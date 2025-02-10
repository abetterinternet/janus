use crate::{
    aggregator::{
        aggregate_step_failure_counter,
        aggregation_job_continue::{compute_helper_aggregate_continue, AggregateContinueMetrics},
        aggregation_job_init::{compute_helper_aggregate_init, AggregateInitMetrics},
        aggregation_job_writer::{
            AggregationJobWriter, AggregationJobWriterMetrics, UpdateWrite,
            WritableReportAggregation,
        },
        batch_mode::CollectableBatchMode,
        error::handle_ping_pong_error,
        http_handlers::AGGREGATION_JOB_ROUTE,
        report_aggregation_success_counter, send_request_to_helper, write_task_aggregation_counter,
        Error, RequestBody,
    },
    cache::HpkeKeypairCache,
    metrics::aggregated_report_share_dimension_histogram,
};
use anyhow::{anyhow, Result};
use backoff::backoff::Backoff;
use bytes::Bytes;
use educe::Educe;
use futures::future::BoxFuture;
use http::{header::RETRY_AFTER, HeaderValue};
use janus_aggregator_core::{
    datastore::{
        self,
        models::{
            AcquiredAggregationJob, AggregationJob, AggregationJobState, Lease, ReportAggregation,
            ReportAggregationState,
        },
        Datastore,
    },
    task::{self, AggregatorTask},
    TIME_HISTOGRAM_BOUNDARIES,
};
use janus_core::{
    retries::{is_retryable_http_client_error, is_retryable_http_status},
    time::Clock,
    vdaf::vdaf_application_context,
    vdaf_dispatch,
};
use janus_messages::{
    batch_mode::{LeaderSelected, TimeInterval},
    AggregationJobContinueReq, AggregationJobInitializeReq, AggregationJobResp,
    PartialBatchSelector, PrepareContinue, PrepareInit, PrepareResp, PrepareStepResult,
    ReportError, ReportMetadata, ReportShare, Role,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter},
    KeyValue,
};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    topology::ping_pong::{PingPongContinuedValue, PingPongState, PingPongTopology},
    vdaf,
};
use rayon::iter::{IndexedParallelIterator as _, IntoParallelIterator as _, ParallelIterator as _};
use reqwest::Method;
use retry_after::RetryAfter;
use std::{
    borrow::Cow,
    collections::HashSet,
    panic,
    sync::Arc,
    time::{Duration, UNIX_EPOCH},
};
use tokio::{
    join,
    sync::{mpsc, Mutex},
    try_join,
};
use tracing::{debug, error, info, info_span, trace_span, warn, Span};

#[cfg(test)]
mod tests;

#[derive(Educe)]
#[educe(Debug)]
pub struct AggregationJobDriver<B> {
    // Configuration.
    batch_aggregation_shard_count: u64,
    task_counter_shard_count: u64,
    hpke_configs_refresh_interval: Duration,
    default_async_poll_interval: Duration,

    // Dependencies.
    http_client: reqwest::Client,
    backoff: B,

    #[educe(Debug(ignore))]
    aggregation_success_counter: Counter<u64>,
    #[educe(Debug(ignore))]
    aggregate_step_failure_counter: Counter<u64>,
    #[educe(Debug(ignore))]
    aggregated_report_share_dimension_histogram: Histogram<u64>,
    #[educe(Debug(ignore))]
    job_cancel_counter: Counter<u64>,
    #[educe(Debug(ignore))]
    job_retry_counter: Counter<u64>,
    #[educe(Debug(ignore))]
    http_request_duration_histogram: Histogram<f64>,
}

impl<R> AggregationJobDriver<R>
where
    R: Backoff + Clone + Send + Sync + 'static,
{
    pub fn new(
        http_client: reqwest::Client,
        backoff: R,
        meter: &Meter,
        batch_aggregation_shard_count: u64,
        task_counter_shard_count: u64,
        hpke_configs_refresh_interval: Duration,
        default_async_poll_interval: Duration,
    ) -> Self {
        let aggregation_success_counter = report_aggregation_success_counter(meter);
        let aggregate_step_failure_counter = aggregate_step_failure_counter(meter);
        let aggregated_report_share_dimension_histogram =
            aggregated_report_share_dimension_histogram(meter);

        let job_cancel_counter = meter
            .u64_counter("janus_job_cancellations")
            .with_description("Count of cancelled jobs.")
            .with_unit("{job}")
            .build();
        job_cancel_counter.add(0, &[]);

        let job_retry_counter = meter
            .u64_counter("janus_job_retries")
            .with_description("Count of retried job steps.")
            .with_unit("{step}")
            .build();
        job_retry_counter.add(0, &[]);

        let http_request_duration_histogram = meter
            .f64_histogram("janus_http_request_duration")
            .with_description(
                "The amount of time elapsed while making an HTTP request to a helper.",
            )
            .with_unit("s")
            .with_boundaries(TIME_HISTOGRAM_BOUNDARIES.to_vec())
            .build();

        Self {
            batch_aggregation_shard_count,
            task_counter_shard_count,
            hpke_configs_refresh_interval,
            default_async_poll_interval,
            http_client,
            backoff,
            aggregation_success_counter,
            aggregate_step_failure_counter,
            aggregated_report_share_dimension_histogram,
            job_cancel_counter,
            job_retry_counter,
            http_request_duration_histogram,
        }
    }

    async fn step_aggregation_job<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error> {
        match lease.leased().batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.step_aggregation_job_generic::<VERIFY_KEY_LENGTH, C, TimeInterval, VdafType>(
                        datastore,
                        hpke_keypairs,
                        Arc::new(vdaf),
                        lease
                    ).await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.step_aggregation_job_generic::<VERIFY_KEY_LENGTH, C, LeaderSelected, VdafType>(
                        datastore,
                        hpke_keypairs,
                        Arc::new(vdaf),
                        lease
                    ).await
                })
            }
        }
    }

    async fn step_aggregation_job_generic<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        for<'a> A::PrepareState:
            PartialEq + Eq + Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // Read all information about the aggregation job.
        let (task, aggregation_job, report_aggregations) = datastore
            .run_tx("step_aggregation_job_generic", |tx| {
                let (lease, vdaf) = (Arc::clone(&lease), Arc::clone(&vdaf));
                Box::pin(async move {
                    let task_future = tx.get_aggregator_task(lease.leased().task_id());
                    let aggregation_job_future = tx.get_aggregation_job::<SEED_SIZE, B, A>(
                        lease.leased().task_id(),
                        lease.leased().aggregation_job_id(),
                    );

                    let (task, aggregation_job) = try_join!(task_future, aggregation_job_future)?;

                    let task = task.ok_or_else(|| {
                        datastore::Error::User(
                            anyhow!("couldn't find task {}", lease.leased().task_id()).into(),
                        )
                    })?;
                    let aggregation_job = aggregation_job.ok_or_else(|| {
                        datastore::Error::User(
                            anyhow!(
                                "couldn't find aggregation job {} for task {}",
                                *lease.leased().aggregation_job_id(),
                                *lease.leased().task_id(),
                            )
                            .into(),
                        )
                    })?;

                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Leader,
                            lease.leased().task_id(),
                            lease.leased().aggregation_job_id(),
                            aggregation_job.aggregation_parameter(),
                        )
                        .await?;

                    Ok((task, aggregation_job, report_aggregations))
                })
            })
            .await?;

        match task.role() {
            Role::Leader => {
                self.step_aggregation_job_leader(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            Role::Helper => {
                self.step_aggregation_job_helper(
                    datastore,
                    hpke_keypairs,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            _ => Err(Error::Internal(format!("unexpected role {}", task.role()))),
        }
    }

    async fn step_aggregation_job_leader<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        for<'a> A::PrepareState:
            PartialEq + Eq + Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // Figure out the next step based on the non-error report aggregation states, and dispatch
        // accordingly.
        let mut saw_init = false;
        let mut saw_continue = false;
        let mut saw_poll = false;
        let mut saw_finished = false;
        for report_aggregation in &report_aggregations {
            match report_aggregation.state() {
                ReportAggregationState::LeaderInit { .. } => saw_init = true,
                ReportAggregationState::LeaderContinue { .. } => saw_continue = true,
                ReportAggregationState::LeaderPoll { .. } => saw_poll = true,

                ReportAggregationState::HelperInitProcessing { .. } => {
                    return Err(Error::Internal(
                        "Leader encountered unexpected ReportAggregationState::HelperInitProcessing"
                            .to_string()
                    ));
                }
                ReportAggregationState::HelperContinue { .. } => {
                    return Err(Error::Internal(
                        "Leader encountered unexpected ReportAggregationState::HelperContinue"
                            .to_string(),
                    ));
                }
                ReportAggregationState::HelperContinueProcessing { .. } => {
                    return Err(Error::Internal(
                        "Leader encountered unexpected ReportAggregationState::HelperContinueProcessing"
                            .to_string()
                    ));
                }

                ReportAggregationState::Finished => saw_finished = true,
                ReportAggregationState::Failed { .. } => (), // ignore failed aggregations
            }
        }
        match (saw_init, saw_continue, saw_poll, saw_finished) {
            // Only saw report aggregations in state "init" (or failed).
            (true, false, false, false) => {
                self.step_aggregation_job_leader_init(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            // Only saw report aggregations in state "continue" (or failed).
            (false, true, false, false) => {
                self.step_aggregation_job_leader_continue(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            // Only saw report aggregations in state "poll" (or failed).
            (false, false, true, false) => {
                self.step_aggregation_job_leader_poll(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            _ => Err(Error::Internal(format!(
                "unexpected combination of report aggregation states (saw_init = {saw_init}, \
                saw_continue = {saw_continue}, saw_poll = {saw_poll}, \
                saw_finished = {saw_finished})",
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn step_aggregation_job_leader_init<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        A::PrepareState: PartialEq + Eq + Send + Sync + Encode,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // Only process non-failed report aggregations.
        let report_aggregations: Vec<_> = report_aggregations
            .into_iter()
            .filter(|report_aggregation| {
                matches!(
                    report_aggregation.state(),
                    &ReportAggregationState::LeaderInit { .. }
                )
            })
            .collect();
        let report_aggregation_count = report_aggregations.len();

        // Compute the next aggregation step.
        //
        // Shutdown on cancellation: if this request is cancelled, the `receiver` will be dropped.
        // This will cause any attempts to send on `sender` to return a `SendError`, which will be
        // returned from the function passed to `try_for_each`; `try_for_each` will terminate early
        // on receiving an error.
        let (ra_sender, mut ra_receiver) = mpsc::unbounded_channel();
        let (pi_and_sa_sender, mut pi_and_sa_receiver) = mpsc::unbounded_channel();
        let aggregation_job = Arc::new(aggregation_job);
        let verify_key = task
            .vdaf_verify_key()
            .map_err(|_| Error::Internal("VDAF verification key has wrong length".to_string()))?;
        let producer_task = tokio::task::spawn_blocking({
            let parent_span = Span::current();
            let vdaf = Arc::clone(&vdaf);
            let task_id = *task.id();
            let aggregation_job = Arc::clone(&aggregation_job);
            let aggregate_step_failure_counter = self.aggregate_step_failure_counter.clone();

            move || {
                let span = info_span!(
                    parent: parent_span,
                    "step_aggregation_job_aggregate_init threadpool task"
                );
                let ctx = vdaf_application_context(&task_id);

                // Compute report shares to send to helper, and decrypt our input shares &
                // initialize preparation state.
                report_aggregations.into_par_iter().try_for_each(
                    |report_aggregation| {
                        let _entered = span.enter();

                    // Extract report data from the report aggregation state.
                    let (
                        public_extensions,
                        public_share,
                        leader_private_extensions,
                        leader_input_share,
                        helper_encrypted_input_share,
                    ) = match report_aggregation.state() {
                        ReportAggregationState::LeaderInit {
                            public_extensions,
                            public_share,
                            leader_private_extensions,
                            leader_input_share,
                            helper_encrypted_input_share,
                        } => (
                            public_extensions,
                            public_share,
                            leader_private_extensions,
                            leader_input_share,
                            helper_encrypted_input_share,
                        ),

                        // Panic safety: this can't happen because we filter to only
                        // LeaderInit-state report aggregations before this loop.
                        _ => panic!(
                            "Unexpected report aggregation state: {:?}",
                            report_aggregation.state()
                        ),
                    };

                    // Check for repeated extensions.
                    let mut extension_types = HashSet::new();
                    if !leader_private_extensions
                        .iter()
                        .chain(public_extensions)
                        .all(|extension| extension_types.insert(extension.extension_type()))
                    {
                        debug!(
                            report_id = %report_aggregation.report_id(),
                            "Received report with duplicate extensions"
                        );
                        aggregate_step_failure_counter
                            .add(1, &[KeyValue::new("type", "duplicate_extension")]);
                        return ra_sender.send(WritableReportAggregation::new(
                            report_aggregation.with_state(ReportAggregationState::Failed {
                                report_error: ReportError::InvalidMessage,
                            }),
                            None,
                        )).map_err(|_| ());
                    }

                    // Initialize the leader's preparation state from the input share.
                    let public_share_bytes = match public_share.get_encoded() {
                        Ok(public_share_bytes) => public_share_bytes,
                        Err(err) => {
                            debug!(report_id = %report_aggregation.report_id(), ?err, "Could not encode public share");
                            aggregate_step_failure_counter
                                .add(1, &[KeyValue::new("type", "public_share_encode_failure")]);
                            return ra_sender.send(WritableReportAggregation::new(
                                report_aggregation.with_state(ReportAggregationState::Failed {
                                    report_error: ReportError::InvalidMessage,
                                }),
                                None,
                            )).map_err(|_| ());
                        }
                    };

                    match trace_span!("VDAF preparation (leader initialization)").in_scope(|| {
                        vdaf.leader_initialized(
                            verify_key.as_bytes(),
                            &ctx,
                            aggregation_job.aggregation_parameter(),
                            // DAP report ID is used as VDAF nonce
                            report_aggregation.report_id().as_ref(),
                            public_share,
                            leader_input_share,
                        )
                        .map_err(|ping_pong_error| {
                            handle_ping_pong_error(
                                &task_id,
                                Role::Leader,
                                report_aggregation.report_id(),
                                ping_pong_error,
                                &aggregate_step_failure_counter,
                            )
                        })
                    }) {
                        Ok((ping_pong_state, ping_pong_message)) => {
                            pi_and_sa_sender.send((
                                report_aggregation.ord(),
                                PrepareInit::new(
                                    ReportShare::new(
                                        ReportMetadata::new(
                                            *report_aggregation.report_id(),
                                            *report_aggregation.time(),
                                            public_extensions.clone(),
                                        ),
                                        public_share_bytes,
                                        helper_encrypted_input_share.clone(),
                                    ),
                                    ping_pong_message,
                                ),
                                SteppedAggregation {
                                    report_aggregation,
                                    leader_state: ping_pong_state,
                                },
                            )).map_err(|_| ())
                        }
                        Err(report_error) => {
                            ra_sender.send(WritableReportAggregation::new(
                                report_aggregation
                                    .with_state(ReportAggregationState::Failed { report_error }),
                                None,
                            )).map_err(|_| ())
                        }
                    }
                })
            }
        });

        let (report_aggregations_to_write, (prepare_inits, stepped_aggregations)) = join!(
            async move {
                let mut report_aggregations_to_write = Vec::with_capacity(report_aggregation_count);
                while ra_receiver
                    .recv_many(&mut report_aggregations_to_write, 10)
                    .await
                    > 0
                {}
                report_aggregations_to_write
            },
            async move {
                let mut pis_and_sas = Vec::with_capacity(report_aggregation_count);
                while pi_and_sa_receiver.recv_many(&mut pis_and_sas, 10).await > 0 {}
                pis_and_sas.sort_unstable_by_key(|(ord, _, _)| *ord);
                let (prepare_inits, stepped_aggregations): (Vec<_>, Vec<_>) =
                    pis_and_sas.into_iter().map(|(_, pi, sa)| (pi, sa)).unzip();
                (prepare_inits, stepped_aggregations)
            },
        );

        // Await the producer task to resume any panics that may have occurred. The only other
        // errors that can occur are: a `JoinError` indicating cancellation, which is impossible
        // because we do not cancel the task; and a `SendError`, which can only happen if this
        // future is cancelled (in which case we will not run this code at all).
        let _ = producer_task.await.map_err(|join_error| {
            if let Ok(reason) = join_error.try_into_panic() {
                panic::resume_unwind(reason);
            }
        });
        assert_eq!(
            report_aggregations_to_write.len() + prepare_inits.len(),
            report_aggregation_count
        );
        assert_eq!(prepare_inits.len(), stepped_aggregations.len());

        let (resp, retry_after) = if !prepare_inits.is_empty() {
            // Construct request, send it to the helper, and process the response.
            let request = AggregationJobInitializeReq::<B>::new(
                aggregation_job
                    .aggregation_parameter()
                    .get_encoded()
                    .map_err(Error::MessageEncode)?,
                PartialBatchSelector::new(aggregation_job.partial_batch_identifier().clone()),
                prepare_inits,
            );

            let http_response = send_request_to_helper(
                &self.http_client,
                self.backoff.clone(),
                Method::PUT,
                task.aggregation_job_uri(aggregation_job.id(), None)?
                    .ok_or_else(|| {
                        Error::InvalidConfiguration("task is leader and has no aggregate share URI")
                    })?,
                AGGREGATION_JOB_ROUTE,
                Some(RequestBody {
                    content_type: AggregationJobInitializeReq::<B>::MEDIA_TYPE,
                    body: Bytes::from(request.get_encoded().map_err(Error::MessageEncode)?),
                }),
                // The only way a task wouldn't have an aggregator auth token in it is in the
                // taskprov case, and Janus never acts as the leader with taskprov enabled.
                task.aggregator_auth_token().ok_or_else(|| {
                    Error::InvalidConfiguration("no aggregator auth token in task")
                })?,
                &self.http_request_duration_histogram,
            )
            .await?;

            let retry_after = http_response
                .headers()
                .get(RETRY_AFTER)
                .map(parse_retry_after)
                .transpose()?;
            let resp = AggregationJobResp::get_decoded(http_response.body())
                .map_err(Error::MessageDecode)?;

            (resp, retry_after)
        } else {
            // If there are no prepare inits to send (because every report aggregation was filtered
            // by the block above), don't send a request to the Helper at all and process an
            // artificial aggregation job response instead, which will finish the aggregation job.
            (
                AggregationJobResp::Finished {
                    prepare_resps: Vec::new(),
                },
                None,
            )
        };

        let aggregation_job: AggregationJob<SEED_SIZE, B, A> =
            Arc::unwrap_or_clone(aggregation_job);
        self.step_aggregation_job_leader_process_response(
            datastore,
            vdaf,
            lease,
            task,
            aggregation_job,
            stepped_aggregations,
            report_aggregations_to_write,
            retry_after.as_ref(),
            resp,
        )
        .await
    }

    async fn step_aggregation_job_leader_continue<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareState: Send + Sync + Encode,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
    {
        // Only process non-failed report aggregations.
        let report_aggregations: Vec<_> = report_aggregations
            .into_iter()
            .filter(|report_aggregation| {
                matches!(
                    report_aggregation.state(),
                    &ReportAggregationState::LeaderContinue { .. }
                )
            })
            .collect();
        let report_aggregation_count = report_aggregations.len();

        // Visit the report aggregations, ignoring any that have already failed; compute our own
        // next step & transitions to send to the helper.
        //
        // Shutdown on cancellation: if this request is cancelled, the `receiver` will be dropped.
        // This will cause any attempts to send on `sender` to return a `SendError`, which will be
        // returned from the function passed to `try_for_each`; `try_for_each` will terminate early
        // on receiving an error.
        let (ra_sender, mut ra_receiver) = mpsc::unbounded_channel();
        let (pc_and_sa_sender, mut pc_and_sa_receiver) = mpsc::unbounded_channel();
        let producer_task = tokio::task::spawn_blocking({
            let parent_span = Span::current();
            let vdaf = Arc::clone(&vdaf);
            let task_id = *task.id();
            let aggregate_step_failure_counter = self.aggregate_step_failure_counter.clone();

            move || {
                let span = info_span!(
                    parent: parent_span,
                    "step_aggregation_job_aggregate_continue threadpool task"
                );
                let ctx = vdaf_application_context(&task_id);

                report_aggregations
                    .into_par_iter()
                    .try_for_each(|report_aggregation| {
                        let _entered = span.enter();

                        let transition = match report_aggregation.state() {
                            ReportAggregationState::LeaderContinue { transition } => transition,
                            // Panic safety: this can't happen because we filter to only
                            // LeaderContinue-state report aggregations before this loop.
                            _ => panic!(
                                "Unexpected report aggregation state: {:?}",
                                report_aggregation.state()
                            ),
                        };

                        let result = trace_span!("VDAF preparation (leader transition evaluation)")
                            .in_scope(|| transition.evaluate(&ctx, vdaf.as_ref()));
                        let (prep_state, message) = match result {
                            Ok((state, message)) => (state, message),
                            Err(error) => {
                                let report_error = handle_ping_pong_error(
                                    &task_id,
                                    Role::Leader,
                                    report_aggregation.report_id(),
                                    error,
                                    &aggregate_step_failure_counter,
                                );
                                return ra_sender
                                    .send(WritableReportAggregation::new(
                                        report_aggregation.with_state(
                                            ReportAggregationState::Failed { report_error },
                                        ),
                                        None,
                                    ))
                                    .map_err(|_| ());
                            }
                        };

                        pc_and_sa_sender
                            .send((
                                report_aggregation.ord(),
                                PrepareContinue::new(*report_aggregation.report_id(), message),
                                SteppedAggregation {
                                    report_aggregation,
                                    leader_state: prep_state,
                                },
                            ))
                            .map_err(|_| ())
                    })
            }
        });

        let (report_aggregations_to_write, (prepare_continues, stepped_aggregations)) = join!(
            async move {
                let mut report_aggregations_to_write = Vec::with_capacity(report_aggregation_count);
                while ra_receiver
                    .recv_many(&mut report_aggregations_to_write, 10)
                    .await
                    > 0
                {}
                report_aggregations_to_write
            },
            async move {
                let mut pcs_and_sas = Vec::with_capacity(report_aggregation_count);
                while pc_and_sa_receiver.recv_many(&mut pcs_and_sas, 10).await > 0 {}
                pcs_and_sas.sort_unstable_by_key(|(ord, _, _)| *ord);
                let (prepare_continues, stepped_aggregations): (Vec<_>, Vec<_>) =
                    pcs_and_sas.into_iter().map(|(_, pc, sa)| (pc, sa)).unzip();
                (prepare_continues, stepped_aggregations)
            }
        );

        // Await the producer task to resume any panics that may have occurred. The only other
        // errors that can occur are: a `JoinError` indicating cancellation, which is impossible
        // because we do not cancel the task; and a `SendError`, which can only happen if this
        // future is cancelled (in which case we will not run this code at all).
        let _ = producer_task.await.map_err(|join_error| {
            if let Ok(reason) = join_error.try_into_panic() {
                panic::resume_unwind(reason);
            }
        });
        assert_eq!(
            report_aggregations_to_write.len() + prepare_continues.len(),
            report_aggregation_count
        );
        assert_eq!(prepare_continues.len(), stepped_aggregations.len());

        // Construct request, send it to the helper, and process the response.
        let request = AggregationJobContinueReq::new(aggregation_job.step(), prepare_continues);

        let http_response = send_request_to_helper(
            &self.http_client,
            self.backoff.clone(),
            Method::POST,
            task.aggregation_job_uri(aggregation_job.id(), None)?
                .ok_or_else(|| {
                    Error::InvalidConfiguration("task is not leader and has no aggregate share URI")
                })?,
            AGGREGATION_JOB_ROUTE,
            Some(RequestBody {
                content_type: AggregationJobContinueReq::MEDIA_TYPE,
                body: Bytes::from(request.get_encoded().map_err(Error::MessageEncode)?),
            }),
            // The only way a task wouldn't have an aggregator auth token in it is in the taskprov
            // case, and Janus never acts as the leader with taskprov enabled.
            task.aggregator_auth_token()
                .ok_or_else(|| Error::InvalidConfiguration("no aggregator auth token in task"))?,
            &self.http_request_duration_histogram,
        )
        .await?;

        let retry_after = http_response
            .headers()
            .get(RETRY_AFTER)
            .map(parse_retry_after)
            .transpose()?;
        let resp =
            AggregationJobResp::get_decoded(http_response.body()).map_err(Error::MessageDecode)?;

        self.step_aggregation_job_leader_process_response(
            datastore,
            vdaf,
            lease,
            task,
            aggregation_job,
            stepped_aggregations,
            report_aggregations_to_write,
            retry_after.as_ref(),
            resp,
        )
        .await
    }

    async fn step_aggregation_job_leader_poll<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareState: Send + Sync + Encode,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
    {
        // Only process non-failed report aggregations; convert non-failed report aggregations into
        // stepped aggregations to be compatible with `process_response_from_helper`.
        let stepped_aggregations: Vec<_> = report_aggregations
            .into_iter()
            .filter_map(|report_aggregation| {
                let leader_state = match report_aggregation.state() {
                    ReportAggregationState::LeaderPoll { leader_state } => {
                        Some(leader_state.clone())
                    }
                    _ => None,
                };

                leader_state.map(|leader_state| SteppedAggregation {
                    report_aggregation,
                    leader_state,
                })
            })
            .collect();

        // Poll the Helper for completion.
        let http_response = send_request_to_helper(
            &self.http_client,
            self.backoff.clone(),
            Method::GET,
            task.aggregation_job_uri(aggregation_job.id(), Some(aggregation_job.step()))?
                .ok_or_else(|| {
                    Error::InvalidConfiguration("task is not leader and has no aggregate share URI")
                })?,
            AGGREGATION_JOB_ROUTE,
            None,
            // The only way a task wouldn't have an aggregator auth token in it is in the taskprov
            // case, and Janus never acts as the leader with taskprov enabled.
            task.aggregator_auth_token()
                .ok_or_else(|| Error::InvalidConfiguration("no aggregator auth token in task"))?,
            &self.http_request_duration_histogram,
        )
        .await?;

        let retry_after = http_response
            .headers()
            .get(RETRY_AFTER)
            .map(parse_retry_after)
            .transpose()?;
        let resp =
            AggregationJobResp::get_decoded(http_response.body()).map_err(Error::MessageDecode)?;

        self.step_aggregation_job_leader_process_response(
            datastore,
            vdaf,
            lease,
            task,
            aggregation_job,
            stepped_aggregations,
            Vec::new(),
            retry_after.as_ref(),
            resp,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn step_aggregation_job_leader_process_response<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        stepped_aggregations: Vec<SteppedAggregation<SEED_SIZE, A>>,
        report_aggregations_to_write: Vec<WritableReportAggregation<SEED_SIZE, A>>,
        retry_after: Option<&RetryAfter>,
        helper_resp: AggregationJobResp,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync + Eq + PartialEq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PrepareShare: Send + Sync,
        A::PrepareState: Send + Sync + Encode,
        A::PublicShare: Send + Sync,
    {
        match helper_resp {
            AggregationJobResp::Processing => {
                self.step_aggregation_job_leader_process_response_processing(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    stepped_aggregations,
                    report_aggregations_to_write,
                    retry_after,
                )
                .await
            }

            AggregationJobResp::Finished { prepare_resps } => {
                self.step_aggregation_job_leader_process_response_finished(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    stepped_aggregations,
                    report_aggregations_to_write,
                    prepare_resps,
                )
                .await
            }
        }
    }

    async fn step_aggregation_job_leader_process_response_processing<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        stepped_aggregations: Vec<SteppedAggregation<SEED_SIZE, A>>,
        mut report_aggregations_to_write: Vec<WritableReportAggregation<SEED_SIZE, A>>,
        retry_after: Option<&RetryAfter>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync + Eq + PartialEq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PrepareShare: Send + Sync,
        A::PrepareState: Send + Sync + Encode,
        A::PublicShare: Send + Sync,
    {
        // Any non-failed report aggregations are set to the Poll state, allowing them to be polled
        // when the aggregation job is next picked up.
        report_aggregations_to_write.extend(stepped_aggregations.into_iter().map(
            |stepped_aggregation| {
                WritableReportAggregation::new(
                    stepped_aggregation.report_aggregation.with_state(
                        ReportAggregationState::LeaderPoll {
                            leader_state: stepped_aggregation.leader_state,
                        },
                    ),
                    // Even if we have recovered an output share (i.e.,
                    // `stepped_aggregation.leader_state` is Finished), we don't include it here: we
                    // aren't done with aggregation until we receive a response from the Helper, so
                    // it would be incorrect to merge the results into the batch aggregations at
                    // this point.
                    None,
                )
            },
        ));

        // Write everything back to storage.
        let task_id = *task.id();
        let mut aggregation_job_writer =
            AggregationJobWriter::<SEED_SIZE, _, _, UpdateWrite, _>::new(
                Arc::new(task),
                self.batch_aggregation_shard_count,
                Some(AggregationJobWriterMetrics {
                    report_aggregation_success_counter: self.aggregation_success_counter.clone(),
                    aggregate_step_failure_counter: self.aggregate_step_failure_counter.clone(),
                    aggregated_report_share_dimension_histogram: self
                        .aggregated_report_share_dimension_histogram
                        .clone(),
                }),
            );
        aggregation_job_writer.put(aggregation_job, report_aggregations_to_write)?;
        let aggregation_job_writer = Arc::new(aggregation_job_writer);

        let retry_after = retry_after
            .map(|ra| retry_after_to_duration(datastore.clock(), ra))
            .transpose()?
            .unwrap_or(self.default_async_poll_interval);
        let counters = datastore
            .run_tx("process_response_from_helper_processing", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let aggregation_job_writer = Arc::clone(&aggregation_job_writer);
                let lease = Arc::clone(&lease);

                Box::pin(async move {
                    let ((_, counters), _) = try_join!(
                        aggregation_job_writer.write(tx, Arc::clone(&vdaf)),
                        tx.release_aggregation_job(&lease, Some(&retry_after)),
                    )?;
                    Ok(counters)
                })
            })
            .await?;

        write_task_aggregation_counter(datastore, self.task_counter_shard_count, task_id, counters);

        Ok(())
    }

    async fn step_aggregation_job_leader_process_response_finished<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        stepped_aggregations: Vec<SteppedAggregation<SEED_SIZE, A>>,
        mut report_aggregations_to_write: Vec<WritableReportAggregation<SEED_SIZE, A>>,
        prepare_resps: Vec<PrepareResp>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync + Eq + PartialEq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PrepareShare: Send + Sync,
        A::PrepareState: Send + Sync + Encode,
        A::PublicShare: Send + Sync,
    {
        // Handle response, computing the new report aggregations to be stored.
        let expected_report_aggregation_count =
            report_aggregations_to_write.len() + stepped_aggregations.len();
        if stepped_aggregations.len() != prepare_resps.len() {
            return Err(Error::Internal(
                "missing, duplicate, out-of-order, or unexpected prepare steps in response"
                    .to_string(),
            ));
        }
        for (stepped_aggregation, helper_prep_resp) in
            stepped_aggregations.iter().zip(&prepare_resps)
        {
            if stepped_aggregation.report_aggregation.report_id() != helper_prep_resp.report_id() {
                return Err(Error::Internal(
                    "missing, duplicate, out-of-order, or unexpected prepare steps in response"
                        .to_string(),
                ));
            }
        }

        // Shutdown on cancellation: if this request is cancelled, the `receiver` will be dropped.
        // This will cause any attempts to send on `sender` to return a `SendError`, which will be
        // returned from the function passed to `try_for_each`; `try_for_each` will terminate early
        // on receiving an error.
        let (ra_sender, mut ra_receiver) = mpsc::unbounded_channel();
        let aggregation_job = Arc::new(aggregation_job);
        let producer_task = tokio::task::spawn_blocking({
            let parent_span = Span::current();
            let vdaf = Arc::clone(&vdaf);
            let task_id = *task.id();
            let aggregation_job = Arc::clone(&aggregation_job);
            let aggregate_step_failure_counter = self.aggregate_step_failure_counter.clone();

            move || {
                let span = info_span!(
                    parent: parent_span,
                    "process_response_from_helper threadpool task"
                );
                let ctx = vdaf_application_context(&task_id);

                stepped_aggregations.into_par_iter().zip(prepare_resps).try_for_each(
                        |(stepped_aggregation, helper_prep_resp)| {
                            let _entered = span.enter();

                            let (new_state, output_share) = match helper_prep_resp.result() {
                                PrepareStepResult::Continue {
                                    message: helper_prep_msg,
                                } => {
                                    let state_and_message = trace_span!("VDAF preparation (leader continuation)")
                                        .in_scope(|| {
                                            vdaf.leader_continued(
                                                &ctx,
                                                stepped_aggregation.leader_state.clone(),
                                                aggregation_job.aggregation_parameter(),
                                                helper_prep_msg,
                                            )
                                            .map_err(|ping_pong_error| {
                                                handle_ping_pong_error(
                                                    &task_id,
                                                    Role::Leader,
                                                    stepped_aggregation.report_aggregation.report_id(),
                                                    ping_pong_error,
                                                    &aggregate_step_failure_counter,
                                                )
                                            })
                                        });

                                    match state_and_message {
                                        Ok(PingPongContinuedValue::WithMessage { transition }) => {
                                            // Leader did not finish. Store our state and outgoing message for the
                                            // next step.
                                            // n.b. it's possible we finished and recovered an output share at the
                                            // VDAF level (i.e., state may be PingPongState::Finished) but we cannot
                                            // finish at the DAP layer and commit the output share until we get
                                            // confirmation from the Helper that they finished, too.
                                            (ReportAggregationState::LeaderContinue { transition }, None)
                                        }
                                        Ok(PingPongContinuedValue::FinishedNoMessage { output_share }) => {
                                            // We finished and have no outgoing message, meaning the Helper was
                                            // already finished. Commit the output share.
                                            (ReportAggregationState::Finished, Some(output_share))
                                        }
                                        Err(report_error) => {
                                            (ReportAggregationState::Failed { report_error }, None)
                                        }
                                    }
                                }

                                PrepareStepResult::Finished => {
                                    if let PingPongState::Finished(output_share) = stepped_aggregation.leader_state
                                    {
                                        // Helper finished and we had already finished. Commit the output share.
                                        (ReportAggregationState::Finished, Some(output_share))
                                    } else {
                                        warn!(
                                            report_id = %stepped_aggregation.report_aggregation.report_id(),
                                            "Helper finished but Leader did not",
                                        );
                                        aggregate_step_failure_counter
                                            .add(1, &[KeyValue::new("type", "finish_mismatch")]);
                                        (
                                            ReportAggregationState::Failed {
                                                report_error: ReportError::VdafPrepError,
                                            },
                                            None,
                                        )
                                    }
                                }

                                PrepareStepResult::Reject(err) => {
                                    // If the helper failed, we move to FAILED immediately.
                                    // TODO(#236): is it correct to just record the transition error that the helper reports?
                                    info!(
                                        report_id = %stepped_aggregation.report_aggregation.report_id(),
                                        helper_error = ?err,
                                        "Helper couldn't step report aggregation",
                                    );
                                    aggregate_step_failure_counter
                                        .add(1, &[KeyValue::new("type", "helper_step_failure")]);
                                    (
                                        ReportAggregationState::Failed {
                                            report_error: *err,
                                        },
                                        None,
                                    )
                                }
                            };

                            ra_sender.send(WritableReportAggregation::new(
                                stepped_aggregation.report_aggregation.with_state(new_state),
                                output_share,
                            ))
                        }
                    )
            }
        });

        while ra_receiver
            .recv_many(&mut report_aggregations_to_write, 10)
            .await
            > 0
        {}

        // Await the producer task to resume any panics that may have occurred. The only other
        // errors that can occur are: a `JoinError` indicating cancellation, which is impossible
        // because we do not cancel the task; and a `SendError`, which can only happen if this
        // future is cancelled (in which case we will not run this code at all).
        let _ = producer_task.await.map_err(|join_error| {
            if let Ok(reason) = join_error.try_into_panic() {
                panic::resume_unwind(reason);
            }
        });
        assert_eq!(
            report_aggregations_to_write.len(),
            expected_report_aggregation_count
        );

        // Write everything back to storage.
        let aggregation_job = Arc::unwrap_or_clone(aggregation_job);
        let task_id = *task.id();
        let mut aggregation_job_writer =
            AggregationJobWriter::<SEED_SIZE, _, _, UpdateWrite, _>::new(
                Arc::new(task),
                self.batch_aggregation_shard_count,
                Some(AggregationJobWriterMetrics {
                    report_aggregation_success_counter: self.aggregation_success_counter.clone(),
                    aggregate_step_failure_counter: self.aggregate_step_failure_counter.clone(),
                    aggregated_report_share_dimension_histogram: self
                        .aggregated_report_share_dimension_histogram
                        .clone(),
                }),
            );
        let new_step = aggregation_job.step().increment();
        aggregation_job_writer.put(
            aggregation_job.with_step(new_step),
            report_aggregations_to_write,
        )?;
        let aggregation_job_writer = Arc::new(aggregation_job_writer);

        let counters = datastore
            .run_tx("process_response_from_helper_finished", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let aggregation_job_writer = Arc::clone(&aggregation_job_writer);
                let lease = Arc::clone(&lease);

                Box::pin(async move {
                    let ((_, counters), _) = try_join!(
                        aggregation_job_writer.write(tx, Arc::clone(&vdaf)),
                        tx.release_aggregation_job(&lease, None),
                    )?;
                    Ok(counters)
                })
            })
            .await?;

        write_task_aggregation_counter(datastore, self.task_counter_shard_count, task_id, counters);

        Ok(())
    }

    async fn step_aggregation_job_helper<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        for<'a> A::PrepareState:
            PartialEq + Eq + Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // Figure out the next step based on the non-error report aggregation states, and dispatch
        // accordingly.
        let mut saw_init = false;
        let mut saw_continue = false;
        let mut saw_finished = false;
        for report_aggregation in &report_aggregations {
            match report_aggregation.state() {
                ReportAggregationState::LeaderInit { .. } => {
                    return Err(Error::Internal(
                        "Helper encountered unexpected ReportAggregationState::LeaderInit"
                            .to_string(),
                    ));
                }
                ReportAggregationState::LeaderContinue { .. } => {
                    return Err(Error::Internal(
                        "Helper encountered unexpected ReportAggregationState::LeaderContinue"
                            .to_string(),
                    ));
                }
                ReportAggregationState::LeaderPoll { .. } => {
                    return Err(Error::Internal(
                        "Leader encountered unexpected ReportAggregationState::LeaderPoll"
                            .to_string(),
                    ));
                }

                ReportAggregationState::HelperInitProcessing { .. } => saw_init = true,
                ReportAggregationState::HelperContinue { .. } => {
                    return Err(Error::Internal(
                        "Helper encountered unexpected ReportAggregationState::HelperContinue"
                            .to_string(),
                    ));
                }
                ReportAggregationState::HelperContinueProcessing { .. } => saw_continue = true,

                ReportAggregationState::Finished => saw_finished = true,
                ReportAggregationState::Failed { .. } => (), // ignore failed aggregations
            }
        }

        match (saw_init, saw_continue, saw_finished) {
            // Only saw report aggregations in state "init processing" (or failed).
            (true, false, false) => {
                self.step_aggregation_job_helper_init(
                    datastore,
                    hpke_keypairs,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            // Only saw report aggregations in state "continue processing" (or failed).
            (false, true, false) => {
                self.step_aggregation_job_helper_continue(
                    datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            _ => Err(Error::Internal(format!(
                "unexpected combination of report aggregation states (saw_init = {saw_init}, \
                saw_continue = {saw_continue}, saw_finished = {saw_finished})",
            ))),
        }
    }

    async fn step_aggregation_job_helper_init<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        hpke_keypairs: Arc<HpkeKeypairCache>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        for<'a> A::PrepareState:
            PartialEq + Eq + Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // Only process report aggregations in the HelperInitProcessing state.
        let report_aggregations = report_aggregations
            .into_iter()
            .filter(|ra| {
                matches!(
                    ra.state(),
                    ReportAggregationState::HelperInitProcessing { .. }
                )
            })
            .collect();

        // Compute the next aggregation step.
        let task = Arc::new(task);
        let aggregation_job =
            Arc::new(aggregation_job.with_state(AggregationJobState::AwaitingRequest));
        let report_aggregations = Arc::new(
            compute_helper_aggregate_init(
                datastore.clock(),
                hpke_keypairs,
                Arc::clone(&vdaf),
                AggregateInitMetrics::new(self.aggregate_step_failure_counter.clone()),
                Arc::clone(&task),
                Arc::clone(&aggregation_job),
                report_aggregations,
            )
            .await?,
        );

        // Write results back to datastore.
        let metrics = AggregationJobWriterMetrics {
            report_aggregation_success_counter: self.aggregation_success_counter.clone(),
            aggregate_step_failure_counter: self.aggregate_step_failure_counter.clone(),
            aggregated_report_share_dimension_histogram: self
                .aggregated_report_share_dimension_histogram
                .clone(),
        };

        let counters = datastore
            .run_tx("aggregate_init_driver_write", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);
                let task = Arc::clone(&task);
                let metrics = metrics.clone();
                let aggregation_job = Arc::clone(&aggregation_job);
                let report_aggregations = Arc::clone(&report_aggregations);
                let batch_aggregation_shard_count = self.batch_aggregation_shard_count;

                Box::pin(async move {
                    // Write aggregation job, report aggregations, and batch aggregations.
                    let report_aggregations =
                        report_aggregations.iter().map(Cow::Borrowed).collect();

                    let mut aggregation_job_writer =
                        AggregationJobWriter::<SEED_SIZE, _, _, UpdateWrite, _>::new(
                            task,
                            batch_aggregation_shard_count,
                            Some(metrics),
                        );
                    aggregation_job_writer
                        .put(aggregation_job.as_ref().clone(), report_aggregations)?;
                    let ((_, counters), _) = try_join!(
                        aggregation_job_writer.write(tx, vdaf),
                        tx.release_aggregation_job(&lease, None),
                    )?;
                    Ok(counters)
                })
            })
            .await?;

        write_task_aggregation_counter(
            datastore,
            self.task_counter_shard_count,
            *task.id(),
            counters,
        );

        Ok(())
    }

    async fn step_aggregation_job_helper_continue<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: AggregatorTask,
        aggregation_job: AggregationJob<SEED_SIZE, B, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        for<'a> A::PrepareState:
            PartialEq + Eq + Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // Only process report aggregations in the HelperContinueProcessing state.
        let report_aggregations = report_aggregations
            .into_iter()
            .filter(|ra| {
                matches!(
                    ra.state(),
                    ReportAggregationState::HelperContinueProcessing { .. }
                )
            })
            .collect();

        // Compute the next aggregation step.
        let task = Arc::new(task);
        let aggregation_job =
            Arc::new(aggregation_job.with_state(AggregationJobState::AwaitingRequest));
        let report_aggregations = Arc::new(
            compute_helper_aggregate_continue(
                Arc::clone(&vdaf),
                AggregateContinueMetrics::new(self.aggregate_step_failure_counter.clone()),
                Arc::clone(&task),
                Arc::clone(&aggregation_job),
                report_aggregations,
            )
            .await,
        );

        // Write results back to datastore.
        let metrics = AggregationJobWriterMetrics {
            report_aggregation_success_counter: self.aggregation_success_counter.clone(),
            aggregate_step_failure_counter: self.aggregate_step_failure_counter.clone(),
            aggregated_report_share_dimension_histogram: self
                .aggregated_report_share_dimension_histogram
                .clone(),
        };

        let counters = datastore
            .run_tx("aggregate_continue_driver_write", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);
                let task = Arc::clone(&task);
                let metrics = metrics.clone();
                let aggregation_job = Arc::clone(&aggregation_job);
                let report_aggregations = Arc::clone(&report_aggregations);

                let batch_aggregation_shard_count = self.batch_aggregation_shard_count;

                Box::pin(async move {
                    let report_aggregations =
                        report_aggregations.iter().map(Cow::Borrowed).collect();
                    let mut aggregation_job_writer =
                        AggregationJobWriter::<SEED_SIZE, _, _, UpdateWrite, _>::new(
                            task,
                            batch_aggregation_shard_count,
                            Some(metrics),
                        );
                    aggregation_job_writer
                        .put(aggregation_job.as_ref().clone(), report_aggregations)?;

                    let ((_, counters), _) = try_join!(
                        aggregation_job_writer.write(tx, vdaf),
                        tx.release_aggregation_job(&lease, None),
                    )?;
                    Ok(counters)
                })
            })
            .await?;

        write_task_aggregation_counter(
            datastore,
            self.task_counter_shard_count,
            *task.id(),
            counters,
        );

        Ok(())
    }

    async fn abandon_aggregation_job<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error> {
        match lease.leased().batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.cancel_aggregation_job_generic::<
                        VERIFY_KEY_LENGTH,
                        C,
                        TimeInterval,
                        VdafType,
                    >(vdaf, datastore, lease)
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.cancel_aggregation_job_generic::<
                        VERIFY_KEY_LENGTH,
                        C,
                        LeaderSelected,
                        VdafType,
                    >(vdaf, datastore, lease)
                    .await
                })
            }
        }
    }

    async fn cancel_aggregation_job_generic<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        A,
    >(
        &self,
        vdaf: A,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
        A::AggregateShare: Send + Sync,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PublicShare: Send + Sync,
    {
        let vdaf = Arc::new(vdaf);
        let batch_aggregation_shard_count = self.batch_aggregation_shard_count;
        let (aggregation_job_uri, aggregator_auth_token) = datastore
            .run_tx("cancel_aggregation_job_generic", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);

                Box::pin(async move {
                    // On abandoning an aggregation job, we update the aggregation job's state field
                    // to Abandoned, but leave all other state (e.g. report aggregations) alone to
                    // ease debugging.
                    let (task, aggregation_job) = try_join!(
                        tx.get_aggregator_task(lease.leased().task_id()),
                        tx.get_aggregation_job::<SEED_SIZE, B, A>(
                            lease.leased().task_id(),
                            lease.leased().aggregation_job_id()
                        ),
                    )?;

                    let task = task.ok_or_else(|| {
                        datastore::Error::User(
                            anyhow!("couldn't find task {}", lease.leased().task_id()).into(),
                        )
                    })?;
                    let aggregation_job = aggregation_job
                        .ok_or_else(|| {
                            datastore::Error::User(
                                anyhow!(
                                    "couldn't find aggregation job {} for task {}",
                                    lease.leased().aggregation_job_id(),
                                    lease.leased().task_id()
                                )
                                .into(),
                            )
                        })?
                        .with_state(AggregationJobState::Abandoned);

                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Leader,
                            lease.leased().task_id(),
                            lease.leased().aggregation_job_id(),
                            aggregation_job.aggregation_parameter(),
                        )
                        .await?
                        .into_iter()
                        .map(|ra| WritableReportAggregation::new(ra, None))
                        .collect();

                    let aggregation_job_uri =
                        task.aggregation_job_uri(lease.leased().aggregation_job_id(), None);
                    let aggregator_auth_token = task.aggregator_auth_token().cloned();

                    let mut aggregation_job_writer =
                        AggregationJobWriter::<SEED_SIZE, _, _, UpdateWrite, _>::new(
                            Arc::new(task),
                            batch_aggregation_shard_count,
                            None,
                        );
                    aggregation_job_writer.put(aggregation_job, report_aggregations)?;

                    try_join!(
                        aggregation_job_writer.write(tx, vdaf),
                        tx.release_aggregation_job(&lease, None),
                    )?;

                    Ok((aggregation_job_uri, aggregator_auth_token))
                })
            })
            .await?;

        // We are giving up on the aggregation job. Delete in the helper so they can clean up
        // resources, too. Because DAP aggregators are not required to implement DELETE on
        // aggregation jobs, we don't check whether this succeeds, though failures will still show
        // up in the HTTP request duration histogram.
        //
        // https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-09#section-4.5.2.2-20
        let _ = send_request_to_helper(
            &self.http_client,
            self.backoff.clone(),
            Method::DELETE,
            aggregation_job_uri?.ok_or_else(|| {
                Error::InvalidConfiguration("task is leader and has no aggregation job URI")
            })?,
            AGGREGATION_JOB_ROUTE,
            None,
            // The only way a task wouldn't have an aggregator auth token in it is in the taskprov
            // case, and Janus never acts as the leader with taskprov enabled.
            &aggregator_auth_token
                .ok_or_else(|| Error::InvalidConfiguration("task has no aggregator auth token"))?,
            &self.http_request_duration_histogram,
        )
        .await;
        Ok(())
    }

    /// Produce a closure for use as a `[JobDriver::JobAcquirer]`.
    pub fn make_incomplete_job_acquirer_callback<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease_duration: Duration,
    ) -> impl Fn(usize) -> BoxFuture<'static, Result<Vec<Lease<AcquiredAggregationJob>>, datastore::Error>>
    {
        move |max_acquire_count: usize| {
            let datastore = Arc::clone(&datastore);

            Box::pin(async move {
                datastore
                    .run_tx("acquire_aggregation_jobs", |tx| {
                        Box::pin(async move {
                            tx.acquire_incomplete_aggregation_jobs(
                                &lease_duration,
                                max_acquire_count,
                            )
                            .await
                        })
                    })
                    .await
            })
        }
    }

    /// Produce a closure for use as a `[JobDriver::JobStepper]`.
    pub fn make_job_stepper_callback<C: Clock>(
        self: Arc<Self>,
        datastore: Arc<Datastore<C>>,
        maximum_attempts_before_failure: usize,
    ) -> impl Fn(Lease<AcquiredAggregationJob>) -> BoxFuture<'static, Result<(), Error>> {
        let hpke_keypairs = Arc::new(Mutex::new(None));

        move |lease| {
            let this = Arc::clone(&self);
            let datastore = Arc::clone(&datastore);
            let hpke_keypairs = Arc::clone(&hpke_keypairs);
            let lease = Arc::new(lease);

            Box::pin(async move {
                let attempts = lease.lease_attempts();
                if attempts > maximum_attempts_before_failure {
                    warn!(
                        attempts = %lease.lease_attempts(),
                        max_attempts = %maximum_attempts_before_failure,
                        "Abandoning job due to too many failed attempts"
                    );
                    this.job_cancel_counter.add(1, &[]);
                    return this.abandon_aggregation_job(datastore, lease).await;
                }

                if attempts > 1 {
                    this.job_retry_counter.add(1, &[]);
                }

                let hpke_keypairs = {
                    let mut hpke_keypairs = hpke_keypairs.lock().await;
                    match hpke_keypairs.as_ref() {
                        Some(hpke_keypairs) => Arc::clone(hpke_keypairs),
                        None => {
                            let hk = Arc::new(
                                HpkeKeypairCache::new(
                                    Arc::clone(&datastore),
                                    this.hpke_configs_refresh_interval,
                                )
                                .await?,
                            );
                            *hpke_keypairs = Some(Arc::clone(&hk));
                            hk
                        }
                    }
                };

                match this
                    .step_aggregation_job(
                        Arc::clone(&datastore),
                        Arc::clone(&hpke_keypairs),
                        Arc::clone(&lease),
                    )
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(error) => {
                        if !Self::is_retryable_error(&error) {
                            // Make a best-effort attempt to immediately cancel the aggregation job.
                            // on fatal errors. This protects the helper from performing wasted
                            // work.
                            //
                            // Cancellation might fail, but we will return the first error, since
                            // that's the more interesting error for debugging purposes.
                            //
                            // If cancellation fails, the job will be picked up again. This isn't
                            // a big deal, since stepping an aggregation job is idempotent. It would
                            // just be some wasted work next time around.
                            warn!(
                                %attempts,
                                max_attempts = %maximum_attempts_before_failure,
                                ?error,
                                "Abandoning job due to fatal error"
                            );
                            this.job_cancel_counter.add(1, &[]);
                            if let Err(error) = this.abandon_aggregation_job(datastore, lease).await
                            {
                                error!(error = ?error, "Failed to abandon job");
                            }
                        }
                        Err(error)
                    }
                }
            })
        }
    }

    /// Determines whether the given [`Error`] is retryable in the context of aggregation job
    /// processing.
    fn is_retryable_error(error: &Error) -> bool {
        match error {
            Error::Http(http_error_response) => {
                is_retryable_http_status(http_error_response.status())
            }
            Error::HttpClient(error) => is_retryable_http_client_error(error),
            Error::Datastore(error) => match error {
                datastore::Error::Db(_) | datastore::Error::Pool(_) => true,
                datastore::Error::User(error) => match error.downcast_ref::<Error>() {
                    Some(error) => Self::is_retryable_error(error),
                    None => false,
                },
                _ => false,
            },
            _ => false,
        }
    }
}

/// SteppedAggregation represents a report aggregation along with the associated preparation-state
/// transition representing the next step for the leader.
struct SteppedAggregation<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
    report_aggregation: ReportAggregation<SEED_SIZE, A>,
    leader_state: PingPongState<SEED_SIZE, 16, A>,
}

fn parse_retry_after(header_value: &HeaderValue) -> Result<RetryAfter, Error> {
    RetryAfter::try_from(header_value)
        .map_err(|err| Error::BadRequest(format!("couldn't parse retry-after header: {err}")))
}

fn retry_after_to_duration<C: Clock>(
    clock: &C,
    retry_after: &RetryAfter,
) -> Result<Duration, Error> {
    match retry_after {
        RetryAfter::Delay(duration) => Ok(*duration),
        RetryAfter::DateTime(next_retry_time) => {
            let now = UNIX_EPOCH + Duration::from_secs(clock.now().as_seconds_since_epoch());
            if &now > next_retry_time {
                return Ok(Duration::ZERO);
            }
            next_retry_time
                .duration_since(now)
                .map_err(|err| Error::Internal(format!("computing retry-after duration: {err}")))
        }
    }
}
