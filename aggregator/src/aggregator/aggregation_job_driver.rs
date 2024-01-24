use super::{error::handle_ping_pong_error, Error, RequestBody};
use crate::aggregator::{
    accumulator::Accumulator, aggregate_step_failure_counter,
    aggregation_job_writer::AggregationJobWriter, http_handlers::AGGREGATION_JOB_ROUTE,
    query_type::CollectableQueryType, send_request_to_helper,
};
use anyhow::{anyhow, Result};
use derivative::Derivative;
use futures::future::{try_join_all, BoxFuture};
use janus_aggregator_core::{
    datastore::{
        self,
        models::{
            AcquiredAggregationJob, AggregationJob, AggregationJobState, Lease, ReportAggregation,
            ReportAggregationState,
        },
        Datastore,
    },
    task::{self, AggregatorTask, VerifyKey},
};
use janus_core::{retries::is_retryable_http_status, time::Clock, vdaf_dispatch};
use janus_messages::{
    query_type::{FixedSize, TimeInterval},
    AggregationJobContinueReq, AggregationJobInitializeReq, AggregationJobResp,
    PartialBatchSelector, PrepareContinue, PrepareError, PrepareInit, PrepareResp,
    PrepareStepResult, ReportId, ReportShare, Role,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter, Unit},
    KeyValue,
};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    topology::ping_pong::{PingPongContinuedValue, PingPongState, PingPongTopology},
    vdaf,
};
use reqwest::Method;
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::try_join;
use tracing::{debug, error, info, trace_span, warn};

#[derive(Derivative)]
#[derivative(Debug)]
pub struct AggregationJobDriver {
    batch_aggregation_shard_count: u64,
    http_client: reqwest::Client,
    #[derivative(Debug = "ignore")]
    aggregate_step_failure_counter: Counter<u64>,
    #[derivative(Debug = "ignore")]
    job_cancel_counter: Counter<u64>,
    #[derivative(Debug = "ignore")]
    job_retry_counter: Counter<u64>,
    #[derivative(Debug = "ignore")]
    http_request_duration_histogram: Histogram<f64>,
}

impl AggregationJobDriver {
    pub fn new(
        http_client: reqwest::Client,
        meter: &Meter,
        batch_aggregation_shard_count: u64,
    ) -> AggregationJobDriver {
        let aggregate_step_failure_counter = aggregate_step_failure_counter(meter);

        let job_cancel_counter = meter
            .u64_counter("janus_job_cancellations")
            .with_description("Count of cancelled jobs.")
            .with_unit(Unit::new("{job}"))
            .init();
        job_cancel_counter.add(0, &[]);

        let job_retry_counter = meter
            .u64_counter("janus_job_retries")
            .with_description("Count of retried job steps.")
            .with_unit(Unit::new("{step}"))
            .init();
        job_retry_counter.add(0, &[]);

        let http_request_duration_histogram = meter
            .f64_histogram("janus_http_request_duration")
            .with_description(
                "The amount of time elapsed while making an HTTP request to a helper.",
            )
            .with_unit(Unit::new("s"))
            .init();

        AggregationJobDriver {
            batch_aggregation_shard_count,
            http_client,
            aggregate_step_failure_counter,
            job_cancel_counter,
            job_retry_counter,
            http_request_duration_histogram,
        }
    }

    async fn step_aggregation_job<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error> {
        match lease.leased().query_type() {
            task::QueryType::TimeInterval => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.step_aggregation_job_generic::<VERIFY_KEY_LENGTH, C, TimeInterval, VdafType>(datastore, Arc::new(vdaf), lease).await
                })
            }
            task::QueryType::FixedSize { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.step_aggregation_job_generic::<VERIFY_KEY_LENGTH, C, FixedSize, VdafType>(datastore, Arc::new(vdaf), lease).await
                })
            }
        }
    }

    async fn step_aggregation_job_generic<
        const SEED_SIZE: usize,
        C: Clock,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error>
    where
        A: 'static + Send + Sync,
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
        let (task, aggregation_job, report_aggregations, verify_key) = datastore
            .run_tx("step_aggregation_job_1", |tx| {
                let (lease, vdaf) = (Arc::clone(&lease), Arc::clone(&vdaf));
                Box::pin(async move {
                    let task = tx
                        .get_aggregator_task(lease.leased().task_id())
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                anyhow!("couldn't find task {}", lease.leased().task_id()).into(),
                            )
                        })?;
                    let verify_key = task.vdaf_verify_key().map_err(|_| {
                        datastore::Error::User(
                            anyhow!("VDAF verification key has wrong length").into(),
                        )
                    })?;

                    let aggregation_job_future = tx.get_aggregation_job::<SEED_SIZE, Q, A>(
                        lease.leased().task_id(),
                        lease.leased().aggregation_job_id(),
                    );
                    let report_aggregations_future = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Leader,
                            lease.leased().task_id(),
                            lease.leased().aggregation_job_id(),
                        );

                    let (aggregation_job, report_aggregations) =
                        try_join!(aggregation_job_future, report_aggregations_future)?;
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

                    Ok((
                        Arc::new(task),
                        aggregation_job,
                        report_aggregations,
                        verify_key,
                    ))
                })
            })
            .await?;

        // Figure out the next step based on the non-error report aggregation states, and dispatch accordingly.
        let (mut saw_start, mut saw_waiting, mut saw_finished) = (false, false, false);
        for report_aggregation in &report_aggregations {
            match report_aggregation.state() {
                ReportAggregationState::StartLeader { .. } => saw_start = true,
                ReportAggregationState::WaitingLeader { .. } => saw_waiting = true,
                ReportAggregationState::WaitingHelper { .. } => {
                    return Err(Error::Internal(
                        "Leader encountered unexpected ReportAggregationState::WaitingHelper"
                            .to_string(),
                    ));
                }
                ReportAggregationState::Finished => saw_finished = true,
                ReportAggregationState::Failed { .. } => (), // ignore failed aggregations
            }
        }
        match (saw_start, saw_waiting, saw_finished) {
            // Only saw report aggregations in state "start" (or failed or invalid).
            (true, false, false) => {
                self.step_aggregation_job_aggregate_init(
                    &datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                    verify_key,
                )
                .await
            }

            // Only saw report aggregations in state "waiting" (or failed or invalid).
            (false, true, false) => {
                self.step_aggregation_job_aggregate_continue(
                    &datastore,
                    vdaf,
                    lease,
                    task,
                    aggregation_job,
                    report_aggregations,
                )
                .await
            }

            _ => Err(Error::Internal(format!(
                "unexpected combination of report aggregation states (saw_start = {saw_start}, \
                saw_waiting = {saw_waiting}, saw_finished = {saw_finished})",
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn step_aggregation_job_aggregate_init<
        const SEED_SIZE: usize,
        C: Clock,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: &Datastore<C>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: Arc<AggregatorTask>,
        aggregation_job: AggregationJob<SEED_SIZE, Q, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
        verify_key: VerifyKey<SEED_SIZE>,
    ) -> Result<(), Error>
    where
        A: 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: PartialEq + Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
        A::PrepareState: PartialEq + Eq + Send + Sync + Encode,
        A::PrepareShare: PartialEq + Eq + Send + Sync,
        A::PrepareMessage: PartialEq + Eq + Send + Sync,
        A::PublicShare: PartialEq + Send + Sync,
    {
        // We currently scrub all reports included in an aggregation job as part of completing the
        // first step. Once we support VDAFs which accept use an aggregation parameter &
        // permit/require multiple aggregations per report, we will need a more complicated strategy
        // where we only scrub a report as part of the _final_ aggregation over the report.
        let report_ids_to_scrub = report_aggregations
            .iter()
            .map(|ra| ra.report_id())
            .copied()
            .collect();

        // Only process non-failed report aggregations.
        let report_aggregations: Vec<_> = report_aggregations
            .into_iter()
            .filter(|report_aggregation| {
                matches!(
                    report_aggregation.state(),
                    &ReportAggregationState::StartLeader { .. }
                )
            })
            .collect();

        // Compute report shares to send to helper, and decrypt our input shares & initialize
        // preparation state.
        let mut report_aggregations_to_write = Vec::new();
        let mut prepare_inits = Vec::new();
        let mut stepped_aggregations = Vec::new();
        for report_aggregation in report_aggregations {
            // Extract report data from the report aggregation state.
            let (public_share, leader_extensions, leader_input_share, helper_encrypted_input_share) =
                match report_aggregation.state() {
                    ReportAggregationState::StartLeader {
                        public_share,
                        leader_extensions,
                        leader_input_share,
                        helper_encrypted_input_share,
                    } => (
                        public_share,
                        leader_extensions,
                        leader_input_share,
                        helper_encrypted_input_share,
                    ),

                    // Panic safety: this can't happen because we filter to only StartLeader-state
                    // report aggregations before this loop.
                    _ => panic!(
                        "Unexpected report aggregation state: {:?}",
                        report_aggregation.state()
                    ),
                };

            // Check for repeated extensions.
            let mut extension_types = HashSet::new();
            if !leader_extensions
                .iter()
                .all(|extension| extension_types.insert(extension.extension_type()))
            {
                debug!(report_id = %report_aggregation.report_id(), "Received report with duplicate extensions");
                self.aggregate_step_failure_counter
                    .add(1, &[KeyValue::new("type", "duplicate_extension")]);
                report_aggregations_to_write.push(report_aggregation.with_state(
                    ReportAggregationState::Failed {
                        prepare_error: PrepareError::InvalidMessage,
                    },
                ));
                continue;
            }

            // Initialize the leader's preparation state from the input share.
            match trace_span!("VDAF preparation").in_scope(|| {
                vdaf.leader_initialized(
                    verify_key.as_bytes(),
                    aggregation_job.aggregation_parameter(),
                    // DAP report ID is used as VDAF nonce
                    report_aggregation.report_id().as_ref(),
                    public_share,
                    leader_input_share,
                )
                .map_err(|ping_pong_error| {
                    handle_ping_pong_error(
                        task.id(),
                        Role::Leader,
                        report_aggregation.report_id(),
                        ping_pong_error,
                        &self.aggregate_step_failure_counter,
                    )
                })
            }) {
                Ok((ping_pong_state, ping_pong_message)) => {
                    prepare_inits.push(PrepareInit::new(
                        ReportShare::new(
                            report_aggregation.report_metadata(),
                            public_share.get_encoded()?,
                            helper_encrypted_input_share.clone(),
                        ),
                        ping_pong_message,
                    ));
                    stepped_aggregations.push(SteppedAggregation {
                        report_aggregation,
                        leader_state: ping_pong_state,
                    });
                }
                Err(prepare_error) => {
                    report_aggregations_to_write.push(
                        report_aggregation
                            .with_state(ReportAggregationState::Failed { prepare_error }),
                    );
                    continue;
                }
            }
        }

        let resp = if !prepare_inits.is_empty() {
            // Construct request, send it to the helper, and process the response.
            let request = AggregationJobInitializeReq::<Q>::new(
                aggregation_job.aggregation_parameter().get_encoded()?,
                PartialBatchSelector::new(aggregation_job.partial_batch_identifier().clone()),
                prepare_inits,
            );

            let resp_bytes = send_request_to_helper(
                &self.http_client,
                Method::PUT,
                task.aggregation_job_uri(aggregation_job.id())?
                    .ok_or_else(|| {
                        Error::InvalidConfiguration("task is leader and has no aggregate share URI")
                    })?,
                AGGREGATION_JOB_ROUTE,
                Some(RequestBody {
                    content_type: AggregationJobInitializeReq::<Q>::MEDIA_TYPE,
                    request,
                }),
                // The only way a task wouldn't have an aggregator auth token in it is in the taskprov
                // case, and Janus never acts as the leader with taskprov enabled.
                task.aggregator_auth_token().ok_or_else(|| {
                    Error::InvalidConfiguration("no aggregator auth token in task")
                })?,
                &self.http_request_duration_histogram,
            )
            .await?;
            AggregationJobResp::get_decoded(&resp_bytes)?
        } else {
            // If there are no prepare inits to send (because every report aggregation was filtered by
            // the block above), don't send a request to the Helper at all and process an artificial
            // aggregation job response instead, which will finish the aggregation job.
            AggregationJobResp::new(Vec::new())
        };

        self.process_response_from_helper(
            datastore,
            vdaf,
            lease,
            task,
            aggregation_job,
            &stepped_aggregations,
            report_aggregations_to_write,
            report_ids_to_scrub,
            resp.prepare_resps(),
        )
        .await
    }

    async fn step_aggregation_job_aggregate_continue<
        const SEED_SIZE: usize,
        C: Clock,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: &Datastore<C>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: Arc<AggregatorTask>,
        aggregation_job: AggregationJob<SEED_SIZE, Q, A>,
        report_aggregations: Vec<ReportAggregation<SEED_SIZE, A>>,
    ) -> Result<(), Error>
    where
        A: 'static,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::AggregateShare: Send + Sync,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareState: Send + Sync + Encode,
        A::PrepareShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        A::PublicShare: Send + Sync,
    {
        // Visit the report aggregations, ignoring any that have already failed; compute our own
        // next step & transitions to send to the helper.
        let mut report_aggregations_to_write = Vec::new();
        let mut prepare_continues = Vec::new();
        let mut stepped_aggregations = Vec::new();
        for report_aggregation in report_aggregations {
            if let ReportAggregationState::WaitingLeader { transition } = report_aggregation.state()
            {
                let (prep_state, message) = match transition.evaluate(vdaf.as_ref()) {
                    Ok((state, message)) => (state, message),
                    Err(error) => {
                        let prepare_error = handle_ping_pong_error(
                            task.id(),
                            Role::Leader,
                            report_aggregation.report_id(),
                            error,
                            &self.aggregate_step_failure_counter,
                        );
                        report_aggregations_to_write.push(
                            report_aggregation
                                .with_state(ReportAggregationState::Failed { prepare_error }),
                        );
                        continue;
                    }
                };

                prepare_continues.push(PrepareContinue::new(
                    *report_aggregation.report_id(),
                    message,
                ));
                stepped_aggregations.push(SteppedAggregation {
                    report_aggregation,
                    leader_state: prep_state,
                });
            }
        }

        // Construct request, send it to the helper, and process the response.
        let request = AggregationJobContinueReq::new(aggregation_job.step(), prepare_continues);

        let resp_bytes = send_request_to_helper(
            &self.http_client,
            Method::POST,
            task.aggregation_job_uri(aggregation_job.id())?
                .ok_or_else(|| {
                    Error::InvalidConfiguration("task is not leader and has no aggregate share URI")
                })?,
            AGGREGATION_JOB_ROUTE,
            Some(RequestBody {
                content_type: AggregationJobContinueReq::MEDIA_TYPE,
                request,
            }),
            // The only way a task wouldn't have an aggregator auth token in it is in the taskprov
            // case, and Janus never acts as the leader with taskprov enabled.
            task.aggregator_auth_token()
                .ok_or_else(|| Error::InvalidConfiguration("no aggregator auth token in task"))?,
            &self.http_request_duration_histogram,
        )
        .await?;
        let resp = AggregationJobResp::get_decoded(&resp_bytes)?;

        self.process_response_from_helper(
            datastore,
            vdaf,
            lease,
            task,
            aggregation_job,
            &stepped_aggregations,
            report_aggregations_to_write,
            Vec::new(), /* reports are only scrubbed on the initial step */
            resp.prepare_resps(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_response_from_helper<
        const SEED_SIZE: usize,
        C: Clock,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: &Datastore<C>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
        task: Arc<AggregatorTask>,
        aggregation_job: AggregationJob<SEED_SIZE, Q, A>,
        stepped_aggregations: &[SteppedAggregation<SEED_SIZE, A>],
        mut report_aggregations_to_write: Vec<ReportAggregation<SEED_SIZE, A>>,
        report_ids_to_scrub: Vec<ReportId>,
        helper_prep_resps: &[PrepareResp],
    ) -> Result<(), Error>
    where
        A: 'static,
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
        if stepped_aggregations.len() != helper_prep_resps.len() {
            return Err(Error::Internal(
                "missing, duplicate, out-of-order, or unexpected prepare steps in response"
                    .to_string(),
            ));
        }
        let mut accumulator = Accumulator::<SEED_SIZE, Q, A>::new(
            Arc::clone(&task),
            self.batch_aggregation_shard_count,
            aggregation_job.aggregation_parameter().clone(),
        );
        for (stepped_aggregation, helper_prep_resp) in
            stepped_aggregations.iter().zip(helper_prep_resps)
        {
            if helper_prep_resp.report_id() != stepped_aggregation.report_aggregation.report_id() {
                return Err(Error::Internal(
                    "missing, duplicate, out-of-order, or unexpected prepare steps in response"
                        .to_string(),
                ));
            }

            let new_state = match helper_prep_resp.result() {
                PrepareStepResult::Continue {
                    message: helper_prep_msg,
                } => {
                    let state_and_message = vdaf
                        .leader_continued(
                            stepped_aggregation.leader_state.clone(),
                            aggregation_job.aggregation_parameter(),
                            helper_prep_msg,
                        )
                        .map_err(|ping_pong_error| {
                            handle_ping_pong_error(
                                task.id(),
                                Role::Leader,
                                stepped_aggregation.report_aggregation.report_id(),
                                ping_pong_error,
                                &self.aggregate_step_failure_counter,
                            )
                        });

                    match state_and_message {
                        Ok(PingPongContinuedValue::WithMessage { transition }) => {
                            // Leader did not finish. Store our state and outgoing message for the
                            // next step.
                            // n.b. it's possible we finished and recovered an output share at the
                            // VDAF level (i.e., state may be PingPongState::Finished) but we cannot
                            // finish at the DAP layer and commit the output share until we get
                            // confirmation from the Helper that they finished, too.
                            ReportAggregationState::WaitingLeader { transition }
                        }
                        Ok(PingPongContinuedValue::FinishedNoMessage { output_share }) => {
                            // We finished and have no outgoing message, meaning the Helper was
                            // already finished. Commit the output share.
                            if let Err(err) = accumulator.update(
                                aggregation_job.partial_batch_identifier(),
                                stepped_aggregation.report_aggregation.report_id(),
                                stepped_aggregation.report_aggregation.time(),
                                &output_share,
                            ) {
                                warn!(
                                    report_id = %stepped_aggregation.report_aggregation.report_id(),
                                    ?err,
                                    "Could not update batch aggregation",
                                );
                                self.aggregate_step_failure_counter
                                    .add(1, &[KeyValue::new("type", "accumulate_failure")]);
                                ReportAggregationState::<SEED_SIZE, A>::Failed {
                                    prepare_error: PrepareError::VdafPrepError,
                                }
                            } else {
                                ReportAggregationState::Finished
                            }
                        }
                        Err(prepare_error) => ReportAggregationState::Failed { prepare_error },
                    }
                }

                PrepareStepResult::Finished => {
                    if let PingPongState::Finished(output_share) = &stepped_aggregation.leader_state
                    {
                        // Helper finished and we had already finished. Commit the output share.
                        if let Err(err) = accumulator.update(
                            aggregation_job.partial_batch_identifier(),
                            stepped_aggregation.report_aggregation.report_id(),
                            stepped_aggregation.report_aggregation.time(),
                            output_share,
                        ) {
                            warn!(
                                report_id = %stepped_aggregation.report_aggregation.report_id(),
                                ?err,
                                "Could not update batch aggregation",
                            );
                            self.aggregate_step_failure_counter
                                .add(1, &[KeyValue::new("type", "accumulate_failure")]);
                            ReportAggregationState::<SEED_SIZE, A>::Failed {
                                prepare_error: PrepareError::VdafPrepError,
                            }
                        } else {
                            ReportAggregationState::Finished
                        }
                    } else {
                        warn!(
                            report_id = %stepped_aggregation.report_aggregation.report_id(),
                            "Helper finished but Leader did not",
                        );
                        self.aggregate_step_failure_counter
                            .add(1, &[KeyValue::new("type", "finish_mismatch")]);
                        ReportAggregationState::Failed {
                            prepare_error: PrepareError::VdafPrepError,
                        }
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
                    self.aggregate_step_failure_counter
                        .add(1, &[KeyValue::new("type", "helper_step_failure")]);
                    ReportAggregationState::Failed {
                        prepare_error: *err,
                    }
                }
            };

            report_aggregations_to_write.push(
                stepped_aggregation
                    .report_aggregation
                    .clone()
                    .with_state(new_state),
            );
        }

        // Write everything back to storage.
        let mut aggregation_job_writer = AggregationJobWriter::new(Arc::clone(&task));
        let new_step = aggregation_job.step().increment();
        aggregation_job_writer.update(
            aggregation_job.with_step(new_step),
            report_aggregations_to_write,
        )?;
        let aggregation_job_writer = Arc::new(aggregation_job_writer);

        let report_ids_to_scrub = Arc::new(report_ids_to_scrub);
        let accumulator = Arc::new(accumulator);
        datastore
            .run_tx("step_aggregation_job_2", |tx| {
                let task_id = *task.id();
                let vdaf = Arc::clone(&vdaf);
                let aggregation_job_writer = Arc::clone(&aggregation_job_writer);
                let accumulator = Arc::clone(&accumulator);
                let report_ids_to_scrub = Arc::clone(&report_ids_to_scrub);
                let lease = Arc::clone(&lease);

                Box::pin(async move {
                    let (unwritable_ra_report_ids, unwritable_ba_report_ids, _, _) = try_join!(
                        aggregation_job_writer.write(tx, Arc::clone(&vdaf)),
                        accumulator.flush_to_datastore(tx, &vdaf),
                        try_join_all(
                            report_ids_to_scrub
                                .iter()
                                .map(|report_id| tx.scrub_client_report(&task_id, report_id))
                        ),
                        tx.release_aggregation_job(&lease),
                    )?;

                    // Currently, writes can fail in two ways: when writing to the batch
                    // aggregations, or when writing the batch/aggregation job/report aggregations.
                    // Until additional work is done to fuse these writes, we must perform a runtime
                    // check that unwritable batch aggregations are a subset of unwritable report
                    // aggregations; this should be guaranteed by the system as we do not make batch
                    // aggregations unwritable until after we make report aggregations unwritable.
                    // But we should certainly check that this is true!
                    // TODO(#1392): remove this check by fusing report aggregation/batch aggregation writes.
                    assert!(unwritable_ba_report_ids.is_subset(&unwritable_ra_report_ids));
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    async fn abandon_aggregation_job<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error> {
        match lease.leased().query_type() {
            task::QueryType::TimeInterval => {
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
            task::QueryType::FixedSize { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.cancel_aggregation_job_generic::<
                        VERIFY_KEY_LENGTH,
                        C,
                        FixedSize,
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
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: A,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredAggregationJob>>,
    ) -> Result<(), Error>
    where
        A: Send + Sync + 'static,
        A::AggregateShare: Send + Sync,
        A::AggregationParam: Send + Sync + PartialEq + Eq,
        A::InputShare: Send + Sync,
        A::OutputShare: Send + Sync,
        A::PrepareMessage: Send + Sync,
        for<'a> A::PrepareState: Send + Sync + Encode + ParameterizedDecode<(&'a A, usize)>,
        A::PublicShare: Send + Sync,
    {
        let vdaf = Arc::new(vdaf);
        let (aggregation_job_uri, aggregator_auth_token) = datastore
            .run_tx("cancel_aggregation_job", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);

                Box::pin(async move {
                    // On abandoning an aggregation job, we update the aggregation job's state field
                    // to Abandoned, but leave all other state (e.g. report aggregations) alone to
                    // ease debugging. (Note that the aggregation_job_writer may still update report
                    // aggregation states to Failed(BatchCollected) if a collection has begun for
                    // the relevant batch.
                    let task = tx
                        .get_aggregator_task(lease.leased().task_id())
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                anyhow!("couldn't find task {}", lease.leased().task_id()).into(),
                            )
                        })?;
                    let aggregation_job = tx
                        .get_aggregation_job::<SEED_SIZE, Q, A>(
                            lease.leased().task_id(),
                            lease.leased().aggregation_job_id(),
                        )
                        .await?
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
                        )
                        .await?;

                    let aggregation_job_uri =
                        task.aggregation_job_uri(lease.leased().aggregation_job_id());
                    let aggregator_auth_token = task.aggregator_auth_token().cloned();

                    let mut aggregation_job_writer = AggregationJobWriter::new(Arc::new(task));
                    aggregation_job_writer.update(aggregation_job, report_aggregations)?;

                    try_join!(
                        aggregation_job_writer.write(tx, vdaf),
                        tx.release_aggregation_job(&lease)
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
            Method::DELETE,
            aggregation_job_uri?.ok_or_else(|| {
                Error::InvalidConfiguration("task is leader and has no aggregation job URI")
            })?,
            AGGREGATION_JOB_ROUTE,
            None as Option<RequestBody<u8>>,
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
        move |lease| {
            let (this, datastore) = (Arc::clone(&self), Arc::clone(&datastore));
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

                match this
                    .step_aggregation_job(Arc::clone(&datastore), Arc::clone(&lease))
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

#[cfg(test)]
mod tests {
    use crate::{
        aggregator::{aggregation_job_driver::AggregationJobDriver, Error},
        binary_utils::job_driver::JobDriver,
    };
    use assert_matches::assert_matches;
    use futures::future::join_all;
    use http::{header::CONTENT_TYPE, StatusCode};
    use janus_aggregator_core::{
        datastore::{
            models::{
                AcquiredAggregationJob, AggregationJob, AggregationJobState, Batch,
                BatchAggregation, BatchAggregationState, BatchState, CollectionJob,
                CollectionJobState, LeaderStoredReport, Lease, ReportAggregation,
                ReportAggregationState,
            },
            test_util::{ephemeral_datastore, EphemeralDatastore},
            Datastore,
        },
        query_type::{AccumulableQueryType, CollectableQueryType},
        task::{test_util::TaskBuilder, AggregatorTask, QueryType, VerifyKey},
        test_util::noop_meter,
    };
    use janus_core::{
        hpke::test_util::generate_test_hpke_config_and_private_key,
        report_id::ReportIdChecksumExt,
        test_util::{install_test_trace_subscriber, run_vdaf, runtime::TestRuntimeManager},
        time::{Clock, IntervalExt, MockClock, TimeExt},
        vdaf::{VdafInstance, VERIFY_KEY_LENGTH},
        Runtime,
    };
    use janus_messages::{
        problem_type::DapProblemType,
        query_type::{FixedSize, TimeInterval},
        AggregationJobContinueReq, AggregationJobInitializeReq, AggregationJobResp,
        AggregationJobStep, Duration, Extension, ExtensionType, FixedSizeQuery, Interval,
        PartialBatchSelector, PrepareContinue, PrepareError, PrepareInit, PrepareResp,
        PrepareStepResult, Query, ReportIdChecksum, ReportMetadata, ReportShare, Role, Time,
    };
    use mockito::ServerGuard;
    use prio::{
        codec::Encode,
        idpf::IdpfInput,
        vdaf::{
            poplar1::{Poplar1, Poplar1AggregationParam},
            prio3::{Prio3, Prio3Count},
            xof::XofTurboShake128,
            Aggregator,
        },
    };
    use rand::random;
    use std::{borrow::Borrow, sync::Arc, time::Duration as StdDuration};
    use trillium_tokio::Stopper;

    #[tokio::test]
    async fn aggregation_job_driver() {
        // This is a minimal test that AggregationJobDriver::run() will successfully find
        // aggregation jobs & step them to completion. More detailed tests of the aggregation job
        // creation logic are contained in other tests which do not exercise the job-acquiry loop.
        // Note that we actually step twice to ensure that lease-release & re-acquiry works as
        // expected.

        // Setup.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let mut runtime_manager = TestRuntimeManager::new();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Poplar1::new_turboshake128(1));
        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Poplar1 { bits: 1 })
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .build();

        let leader_task = task.leader_view().unwrap();

        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();
        let measurement = IdpfInput::from_bools(&[true]);
        let aggregation_param =
            Poplar1AggregationParam::try_from_prefixes(Vec::from([IdpfInput::from_bools(&[true])]))
                .unwrap();

        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &aggregation_param,
            report_metadata.id(),
            &measurement,
        );

        let agg_auth_token = task.aggregator_auth_token().clone();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );

        let aggregation_job_id = random();

        let collection_job = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report, aggregation_param) = (
                    vdaf.clone(),
                    leader_task.clone(),
                    report.clone(),
                    aggregation_param.clone(),
                );
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();
                    tx.mark_report_aggregated(task.id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param.clone(),
                        (),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();
                    tx.put_report_aggregation(
                        &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        batch_identifier,
                        aggregation_param.clone(),
                        BatchState::Closing,
                        1,
                        Interval::from_time(&time).unwrap(),
                    ))
                    .await
                    .unwrap();

                    let collection_job = CollectionJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        random(),
                        Query::new_time_interval(batch_identifier),
                        aggregation_param,
                        batch_identifier,
                        CollectionJobState::Start,
                    );
                    tx.put_collection_job(&collection_job).await.unwrap();

                    Ok(collection_job)
                })
            })
            .await
            .unwrap();

        // Setup: prepare mocked HTTP responses.
        let helper_responses = Vec::from([
            (
                "PUT",
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
                AggregationJobResp::MEDIA_TYPE,
                AggregationJobResp::new(Vec::from([PrepareResp::new(
                    *report.metadata().id(),
                    PrepareStepResult::Continue {
                        message: transcript.helper_prepare_transitions[0].message.clone(),
                    },
                )]))
                .get_encoded()
                .unwrap(),
            ),
            (
                "POST",
                AggregationJobContinueReq::MEDIA_TYPE,
                AggregationJobResp::MEDIA_TYPE,
                AggregationJobResp::new(Vec::from([PrepareResp::new(
                    *report.metadata().id(),
                    PrepareStepResult::Finished,
                )]))
                .get_encoded()
                .unwrap(),
            ),
        ]);
        let mocked_aggregates = join_all(helper_responses.iter().map(
            |(req_method, req_content_type, resp_content_type, resp_body)| {
                let (header, value) = agg_auth_token.request_authentication();
                server
                    .mock(
                        req_method,
                        task.aggregation_job_uri(&aggregation_job_id)
                            .unwrap()
                            .path(),
                    )
                    .match_header(header, value.as_str())
                    .match_header(CONTENT_TYPE.as_str(), *req_content_type)
                    .with_status(200)
                    .with_header(CONTENT_TYPE.as_str(), resp_content_type)
                    .with_body(resp_body)
                    .create_async()
            },
        ))
        .await;
        let aggregation_job_driver = Arc::new(AggregationJobDriver::new(
            reqwest::Client::new(),
            &noop_meter(),
            32,
        ));
        let stopper = Stopper::new();

        // Run. Let the aggregation job driver step aggregation jobs, then kill it.
        let aggregation_job_driver = Arc::new(
            JobDriver::new(
                clock,
                runtime_manager.with_label("stepper"),
                noop_meter(),
                stopper.clone(),
                StdDuration::from_secs(1),
                10,
                StdDuration::from_secs(60),
                aggregation_job_driver.make_incomplete_job_acquirer_callback(
                    Arc::clone(&ds),
                    StdDuration::from_secs(600),
                ),
                aggregation_job_driver.make_job_stepper_callback(Arc::clone(&ds), 5),
            )
            .unwrap(),
        );

        let task_handle = runtime_manager
            .with_label("driver")
            .spawn(aggregation_job_driver.run());

        tracing::info!("awaiting stepper tasks");
        // Wait for all of the aggregation job stepper tasks to complete.
        runtime_manager.wait_for_completed_tasks("stepper", 2).await;
        // Stop the aggregation job driver.
        stopper.stop();
        // Wait for the aggregation job driver task to complete.
        task_handle.await.unwrap();

        // Verify.
        for mocked_aggregate in mocked_aggregates {
            mocked_aggregate.assert_async().await;
        }

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                aggregation_param.clone(),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobStep::from(2),
            );
        let want_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                *report.metadata().id(),
                *report.metadata().time(),
                0,
                None,
                ReportAggregationState::Finished,
            );
        let want_batch =
            Batch::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                batch_identifier,
                aggregation_param.clone(),
                BatchState::Closed,
                0,
                Interval::from_time(&time).unwrap(),
            );
        let want_collection_job = collection_job.with_state(CollectionJobState::Collectable);

        let (got_aggregation_job, got_report_aggregation, got_batch, got_collection_job) = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = task.clone();
                let report_id = *report.metadata().id();
                let collection_job_id = *want_collection_job.id();

                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            &report_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let batch = tx
                        .get_batch(
                            task.id(),
                            &batch_identifier,
                            aggregation_job.aggregation_parameter(),
                        )
                        .await.unwrap()
                        .unwrap();
                    let collection_job = tx
                        .get_collection_job(vdaf.as_ref(), task.id(), &collection_job_id)
                        .await.unwrap()
                        .unwrap();
                    Ok((aggregation_job, report_aggregation, batch, collection_job))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(want_batch, got_batch);
        assert_eq!(want_collection_job, got_collection_job);
    }

    #[tokio::test]
    async fn step_time_interval_aggregation_job_init_single_step() {
        // Setup: insert a client report and add it to a new aggregation job.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Prio3::new_count(2).unwrap());

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .build();

        let leader_task = task.leader_view().unwrap();

        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            report_metadata.id(),
            &false,
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let repeated_extension_report = LeaderStoredReport::generate(
            *task.id(),
            ReportMetadata::new(random(), time),
            helper_hpke_keypair.config(),
            Vec::from([
                Extension::new(ExtensionType::Tbd, Vec::new()),
                Extension::new(ExtensionType::Tbd, Vec::new()),
            ]),
            &transcript,
        );
        let aggregation_job_id = random();

        let lease = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = leader_task.clone();
                let report = report.clone();
                let repeated_extension_report = repeated_extension_report.clone();

                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &repeated_extension_report)
                        .await
                        .unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();
                    tx.scrub_client_report(
                        repeated_extension_report.task_id(),
                        repeated_extension_report.metadata().id(),
                    )
                    .await
                    .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        (),
                        (),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();
                    tx.put_report_aggregation(
                        &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();
                    tx.put_report_aggregation(
                        &repeated_extension_report
                            .as_start_leader_report_aggregation(aggregation_job_id, 1),
                    )
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                        *task.id(),
                        batch_identifier,
                        (),
                        BatchState::Closing,
                        1,
                        Interval::from_time(&time).unwrap(),
                    ))
                    .await
                    .unwrap();

                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Setup: prepare mocked HTTP response. (first an error response, then a success)
        // (This is fragile in that it expects the leader request to be deterministically encoded.
        // It would be nicer to retrieve the request bytes from the mock, then do our own parsing &
        // verification -- but mockito does not expose this functionality at time of writing.)
        let leader_request = AggregationJobInitializeReq::new(
            ().get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            Vec::from([PrepareInit::new(
                ReportShare::new(
                    report.metadata().clone(),
                    report.public_share().get_encoded().unwrap(),
                    report.helper_encrypted_input_share().clone(),
                ),
                transcript.leader_prepare_transitions[0].message.clone(),
            )]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]));
        let mocked_aggregate_failure = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"urn:ietf:params:ppm:dap:error:unauthorizedRequest\"}")
            .create_async()
            .await;
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_success = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created twice.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        let error = aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease.clone()))
            .await
            .unwrap_err();
        assert_matches!(
            error,
            Error::Http(error_response) => {
                assert_eq!(error_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(*error_response.dap_problem_type().unwrap(), DapProblemType::UnauthorizedRequest);
            }
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_failure.assert_async().await;
        mocked_aggregate_success.assert_async().await;

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobStep::from(1),
            );
        let want_report_aggregation = ReportAggregation::<VERIFY_KEY_LENGTH, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            *report.metadata().id(),
            *report.metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        );
        let want_repeated_extension_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                *repeated_extension_report.metadata().id(),
                *repeated_extension_report.metadata().time(),
                1,
                None,
                ReportAggregationState::Failed {
                    prepare_error: PrepareError::InvalidMessage,
                },
            );
        let want_batch = Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
            *task.id(),
            batch_identifier,
            (),
            BatchState::Closing,
            0,
            Interval::from_time(&time).unwrap(),
        );

        let (
            got_aggregation_job,
            got_report_aggregation,
            got_repeated_extension_report_aggregation,
            got_batch,
        ) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_id, repeated_extension_report_id) = (
                    Arc::clone(&vdaf),
                    task.clone(),
                    *report.metadata().id(),
                    *repeated_extension_report.metadata().id(),
                );
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            &report_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let repeated_extension_report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            &repeated_extension_report_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let batch = tx
                        .get_batch(task.id(), &batch_identifier, &())
                        .await
                        .unwrap()
                        .unwrap();
                    Ok((
                        aggregation_job,
                        report_aggregation,
                        repeated_extension_report_aggregation,
                        batch,
                    ))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(
            want_repeated_extension_report_aggregation,
            got_repeated_extension_report_aggregation
        );
        assert_eq!(want_batch, got_batch);
    }

    #[tokio::test]
    async fn step_time_interval_aggregation_job_init_two_steps() {
        // Setup: insert a client report and add it to a new aggregation job.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Poplar1::new_turboshake128(1));

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Poplar1 { bits: 1 })
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .build();

        let leader_task = task.leader_view().unwrap();

        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();
        let measurement = IdpfInput::from_bools(&[true]);
        let aggregation_param =
            Poplar1AggregationParam::try_from_prefixes(Vec::from([IdpfInput::from_bools(&[true])]))
                .unwrap();

        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &aggregation_param,
            report_metadata.id(),
            &measurement,
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let aggregation_job_id = random();

        let lease = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report, aggregation_param) = (
                    vdaf.clone(),
                    leader_task.clone(),
                    report.clone(),
                    aggregation_param.clone(),
                );
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param.clone(),
                        (),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(
                        &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        batch_identifier,
                        aggregation_param,
                        BatchState::Closing,
                        1,
                        Interval::from_time(&time).unwrap(),
                    ))
                    .await
                    .unwrap();

                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Setup: prepare mocked HTTP response. (first an error response, then a success)
        // (This is fragile in that it expects the leader request to be deterministically encoded.
        // It would be nicer to retrieve the request bytes from the mock, then do our own parsing &
        // verification -- but mockito does not expose this functionality at time of writing.)
        let leader_request = AggregationJobInitializeReq::new(
            aggregation_param.get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            Vec::from([PrepareInit::new(
                ReportShare::new(
                    report.metadata().clone(),
                    report.public_share().get_encoded().unwrap(),
                    report.helper_encrypted_input_share().clone(),
                ),
                transcript.leader_prepare_transitions[0].message.clone(),
            )]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]));
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_success = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created twice.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_success.assert_async().await;

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                aggregation_param.clone(),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::InProgress,
                AggregationJobStep::from(1),
            );
        let want_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                *report.metadata().id(),
                *report.metadata().time(),
                0,
                None,
                ReportAggregationState::WaitingLeader {
                    transition: transcript.leader_prepare_transitions[1]
                        .transition
                        .clone()
                        .unwrap(),
                },
            );
        let want_batch =
            Batch::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                batch_identifier,
                aggregation_param,
                BatchState::Closing,
                1,
                Interval::from_time(&time).unwrap(),
            );

        let (got_aggregation_job, got_report_aggregation, got_batch) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_id) =
                    (Arc::clone(&vdaf), task.clone(), *report.metadata().id());
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            &report_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let batch = tx
                        .get_batch(
                            task.id(),
                            &batch_identifier,
                            aggregation_job.aggregation_parameter(),
                        )
                        .await.unwrap()
                        .unwrap();
                    Ok((aggregation_job, report_aggregation, batch))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(want_batch, got_batch);
    }

    #[tokio::test]
    async fn step_time_interval_aggregation_job_init_partially_garbage_collected() {
        // This is a regression test for https://github.com/divviup/janus/issues/2464.

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(500);
        const TIME_PRECISION: Duration = Duration::from_seconds(10);

        // Setup: insert an "old" and "new" client report, and add them to a new aggregation job.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Prio3::new_count(2).unwrap());

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
            .with_time_precision(TIME_PRECISION)
            .build();

        let leader_task = task.leader_view().unwrap();

        let gc_eligible_time = OLDEST_ALLOWED_REPORT_TIMESTAMP
            .sub(&Duration::from_seconds(3 * TIME_PRECISION.as_seconds()))
            .unwrap()
            .to_batch_interval_start(&TIME_PRECISION)
            .unwrap();
        let gc_eligible_batch_identifier =
            TimeInterval::to_batch_identifier(&leader_task, &(), &gc_eligible_time).unwrap();
        let gc_eligible_report_metadata = ReportMetadata::new(random(), gc_eligible_time);

        let gc_ineligible_time = OLDEST_ALLOWED_REPORT_TIMESTAMP
            .add(&Duration::from_seconds(3 * TIME_PRECISION.as_seconds()))
            .unwrap()
            .to_batch_interval_start(&TIME_PRECISION)
            .unwrap();
        let gc_ineligible_batch_identifier =
            TimeInterval::to_batch_identifier(&leader_task, &(), &gc_ineligible_time).unwrap();
        let gc_ineligible_report_metadata = ReportMetadata::new(random(), gc_ineligible_time);

        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let gc_eligible_transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            gc_eligible_report_metadata.id(),
            &false,
        );
        let gc_ineligible_transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            gc_ineligible_report_metadata.id(),
            &false,
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let gc_eligible_report = LeaderStoredReport::generate(
            *task.id(),
            gc_eligible_report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &gc_eligible_transcript,
        );
        let gc_ineligible_report = LeaderStoredReport::generate(
            *task.id(),
            gc_ineligible_report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &gc_ineligible_transcript,
        );

        let aggregation_job_id = random();

        let lease = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let leader_task = leader_task.clone();
                let gc_eligible_report = gc_eligible_report.clone();
                let gc_ineligible_report = gc_ineligible_report.clone();

                Box::pin(async move {
                    tx.put_aggregator_task(&leader_task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &gc_eligible_report)
                        .await
                        .unwrap();
                    tx.put_client_report(vdaf.borrow(), &gc_ineligible_report)
                        .await
                        .unwrap();
                    tx.scrub_client_report(
                        gc_eligible_report.task_id(),
                        gc_eligible_report.metadata().id(),
                    )
                    .await
                    .unwrap();
                    tx.scrub_client_report(
                        gc_ineligible_report.task_id(),
                        gc_ineligible_report.metadata().id(),
                    )
                    .await
                    .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Prio3Count,
                    >::new(
                        *leader_task.id(),
                        aggregation_job_id,
                        (),
                        (),
                        Interval::new(
                            gc_eligible_time,
                            gc_ineligible_time.difference(&gc_eligible_time).unwrap(),
                        )
                        .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();
                    tx.put_report_aggregation(
                        &gc_eligible_report
                            .as_start_leader_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();
                    tx.put_report_aggregation(
                        &gc_ineligible_report
                            .as_start_leader_report_aggregation(aggregation_job_id, 1),
                    )
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                        *leader_task.id(),
                        gc_eligible_batch_identifier,
                        (),
                        BatchState::Closing,
                        1,
                        Interval::from_time(&gc_eligible_time).unwrap(),
                    ))
                    .await
                    .unwrap();
                    tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                        *leader_task.id(),
                        gc_ineligible_batch_identifier,
                        (),
                        BatchState::Closing,
                        1,
                        Interval::from_time(&gc_ineligible_time).unwrap(),
                    ))
                    .await
                    .unwrap();

                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Setup: prepare mocked HTTP response.
        let leader_request = AggregationJobInitializeReq::new(
            ().get_encoded().unwrap(),
            PartialBatchSelector::new_time_interval(),
            Vec::from([
                PrepareInit::new(
                    ReportShare::new(
                        gc_eligible_report.metadata().clone(),
                        gc_eligible_report.public_share().get_encoded().unwrap(),
                        gc_eligible_report.helper_encrypted_input_share().clone(),
                    ),
                    gc_eligible_transcript.leader_prepare_transitions[0]
                        .message
                        .clone(),
                ),
                PrepareInit::new(
                    ReportShare::new(
                        gc_ineligible_report.metadata().clone(),
                        gc_ineligible_report.public_share().get_encoded().unwrap(),
                        gc_ineligible_report.helper_encrypted_input_share().clone(),
                    ),
                    gc_ineligible_transcript.leader_prepare_transitions[0]
                        .message
                        .clone(),
                ),
            ]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([
            PrepareResp::new(
                *gc_eligible_report.metadata().id(),
                PrepareStepResult::Continue {
                    message: gc_eligible_transcript.helper_prepare_transitions[0]
                        .message
                        .clone(),
                },
            ),
            PrepareResp::new(
                *gc_ineligible_report.metadata().id(),
                PrepareStepResult::Continue {
                    message: gc_ineligible_transcript.helper_prepare_transitions[0]
                        .message
                        .clone(),
                },
            ),
        ]));
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_init = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_init.assert_async().await;

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                (),
                (),
                Interval::new(
                    gc_eligible_time,
                    gc_ineligible_time.difference(&gc_eligible_time).unwrap(),
                )
                .unwrap(),
                AggregationJobState::Finished,
                AggregationJobStep::from(1),
            );

        let want_gc_eligible_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                *gc_eligible_report.metadata().id(),
                *gc_eligible_report.metadata().time(),
                0,
                None,
                ReportAggregationState::Finished,
            );
        let want_ineligible_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                *gc_ineligible_report.metadata().id(),
                *gc_ineligible_report.metadata().time(),
                1,
                None,
                ReportAggregationState::Finished,
            );
        let want_report_aggregations = Vec::from([
            want_gc_eligible_report_aggregation,
            want_ineligible_report_aggregation,
        ]);

        let want_batch = Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
            *task.id(),
            gc_ineligible_batch_identifier,
            (),
            BatchState::Closing,
            0,
            Interval::from_time(&gc_ineligible_time).unwrap(),
        );
        let want_batches = Vec::from([want_batch]);

        let (got_aggregation_job, got_report_aggregations, got_batches) = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = task.clone();
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let report_aggregations = tx
                        .get_report_aggregations_for_aggregation_job(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await
                        .unwrap();
                    let batches = tx.get_batches_for_task(task.id()).await.unwrap();
                    Ok((aggregation_job, report_aggregations, batches))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregations, got_report_aggregations);
        assert_eq!(want_batches, got_batches);
    }

    #[tokio::test]
    async fn step_fixed_size_aggregation_job_init_single_step() {
        // Setup: insert a client report and add it to a new aggregation job.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Prio3::new_count(2).unwrap());

        let task = TaskBuilder::new(
            QueryType::FixedSize {
                max_batch_size: 10,
                batch_time_window_size: None,
            },
            VdafInstance::Prio3Count,
        )
        .with_helper_aggregator_endpoint(server.url().parse().unwrap())
        .build();

        let leader_task = task.leader_view().unwrap();

        let report_metadata = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            report_metadata.id(),
            &false,
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let batch_id = random();
        let aggregation_job_id = random();

        let lease = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report) = (vdaf.clone(), leader_task.clone(), report.clone());
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        (),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(
                        &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, FixedSize, Prio3Count>::new(
                        *task.id(),
                        batch_id,
                        (),
                        BatchState::Open,
                        1,
                        Interval::from_time(report.metadata().time()).unwrap(),
                    ))
                    .await
                    .unwrap();

                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Setup: prepare mocked HTTP response. (first an error response, then a success)
        // (This is fragile in that it expects the leader request to be deterministically encoded.
        // It would be nicer to retrieve the request bytes from the mock, then do our own parsing &
        // verification -- but mockito does not expose this functionality at time of writing.)
        let leader_request = AggregationJobInitializeReq::new(
            ().get_encoded().unwrap(),
            PartialBatchSelector::new_fixed_size(batch_id),
            Vec::from([PrepareInit::new(
                ReportShare::new(
                    report.metadata().clone(),
                    report.public_share().get_encoded().unwrap(),
                    report.helper_encrypted_input_share().clone(),
                ),
                transcript.leader_prepare_transitions[0].message.clone(),
            )]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]));
        let mocked_aggregate_failure = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"urn:ietf:params:ppm:dap:error:unauthorizedRequest\"}")
            .create_async()
            .await;
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_success = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created twice.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        let error = aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease.clone()))
            .await
            .unwrap_err();
        assert_matches!(
            error,
            Error::Http(error_response) => {
                assert_eq!(error_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(*error_response.dap_problem_type().unwrap(), DapProblemType::UnauthorizedRequest);
            }
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_failure.assert_async().await;
        mocked_aggregate_success.assert_async().await;

        let want_aggregation_job = AggregationJob::<VERIFY_KEY_LENGTH, FixedSize, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            (),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        );
        let want_report_aggregation = ReportAggregation::<VERIFY_KEY_LENGTH, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            *report.metadata().id(),
            *report.metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        );
        let want_batch = Batch::<VERIFY_KEY_LENGTH, FixedSize, Prio3Count>::new(
            *task.id(),
            batch_id,
            (),
            BatchState::Open,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
        );

        let (got_aggregation_job, got_report_aggregation, got_batch) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_id) =
                    (Arc::clone(&vdaf), task.clone(), *report.metadata().id());
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, FixedSize, Prio3Count>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            &report_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let batch = tx
                        .get_batch(task.id(), &batch_id, &())
                        .await
                        .unwrap()
                        .unwrap();
                    Ok((aggregation_job, report_aggregation, batch))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(want_batch, got_batch);
    }

    #[tokio::test]
    async fn step_fixed_size_aggregation_job_init_two_steps() {
        // Setup: insert a client report and add it to a new aggregation job.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Poplar1::new_turboshake128(1));

        let task = TaskBuilder::new(
            QueryType::FixedSize {
                max_batch_size: 10,
                batch_time_window_size: None,
            },
            VdafInstance::Poplar1 { bits: 1 },
        )
        .with_helper_aggregator_endpoint(server.url().parse().unwrap())
        .build();

        let leader_task = task.leader_view().unwrap();

        let report_metadata = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();
        let measurement = IdpfInput::from_bools(&[true]);
        let aggregation_param =
            Poplar1AggregationParam::try_from_prefixes(Vec::from([IdpfInput::from_bools(&[true])]))
                .unwrap();

        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &aggregation_param,
            report_metadata.id(),
            &measurement,
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let batch_id = random();
        let aggregation_job_id = random();

        let lease = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report, aggregation_param) = (
                    vdaf.clone(),
                    leader_task.clone(),
                    report.clone(),
                    aggregation_param.clone(),
                );
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param.clone(),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(
                        &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        batch_id,
                        aggregation_param.clone(),
                        BatchState::Open,
                        1,
                        Interval::from_time(report.metadata().time()).unwrap(),
                    ))
                    .await
                    .unwrap();

                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Setup: prepare mocked HTTP response. (first an error response, then a success)
        // (This is fragile in that it expects the leader request to be deterministically encoded.
        // It would be nicer to retrieve the request bytes from the mock, then do our own parsing &
        // verification -- but mockito does not expose this functionality at time of writing.)
        let leader_request = AggregationJobInitializeReq::new(
            aggregation_param.get_encoded().unwrap(),
            PartialBatchSelector::new_fixed_size(batch_id),
            Vec::from([PrepareInit::new(
                ReportShare::new(
                    report.metadata().clone(),
                    report.public_share().get_encoded().unwrap(),
                    report.helper_encrypted_input_share().clone(),
                ),
                transcript.leader_prepare_transitions[0].message.clone(),
            )]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]));
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_success = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created twice.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_success.assert_async().await;

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, FixedSize, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                aggregation_param.clone(),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::InProgress,
                AggregationJobStep::from(1),
            );
        let want_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                *report.metadata().id(),
                *report.metadata().time(),
                0,
                None,
                ReportAggregationState::WaitingLeader {
                    transition: transcript.leader_prepare_transitions[1]
                        .transition
                        .clone()
                        .unwrap(),
                },
            );
        let want_batch = Batch::<VERIFY_KEY_LENGTH, FixedSize, Poplar1<XofTurboShake128, 16>>::new(
            *task.id(),
            batch_id,
            aggregation_param.clone(),
            BatchState::Open,
            1,
            Interval::from_time(report.metadata().time()).unwrap(),
        );

        let (got_aggregation_job, got_report_aggregation, got_batch) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_id) =
                    (Arc::clone(&vdaf), task.clone(), *report.metadata().id());
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, FixedSize, Poplar1<XofTurboShake128, 16>>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            &report_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let batch = tx
                        .get_batch(
                            task.id(),
                            &batch_id,
                            aggregation_job.aggregation_parameter(),
                        )
                        .await.unwrap()
                        .unwrap();
                    Ok((aggregation_job, report_aggregation, batch))
                })
            })
            .await
            .unwrap();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(want_batch, got_batch);
    }

    #[tokio::test]
    async fn step_time_interval_aggregation_job_continue() {
        // Setup: insert a client report and add it to an aggregation job whose state has already
        // been stepped once.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Poplar1::new_turboshake128(1));

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Poplar1 { bits: 1 })
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .build();
        let leader_task = task.leader_view().unwrap();
        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let active_batch_identifier =
            TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
        let other_batch_identifier = Interval::new(
            active_batch_identifier
                .start()
                .add(task.time_precision())
                .unwrap(),
            *task.time_precision(),
        )
        .unwrap();
        let collection_identifier = Interval::new(
            *active_batch_identifier.start(),
            Duration::from_seconds(2 * task.time_precision().as_seconds()),
        )
        .unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let aggregation_param = Poplar1AggregationParam::try_from_prefixes(Vec::from([
            IdpfInput::from_bools(&[false]),
        ]))
        .unwrap();
        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &aggregation_param,
            report_metadata.id(),
            &IdpfInput::from_bools(&[true]),
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let aggregation_job_id = random();

        let leader_aggregate_share = vdaf
            .aggregate(&aggregation_param, [transcript.leader_output_share.clone()])
            .unwrap();

        let (lease, want_collection_job) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, aggregation_param, report, transcript) = (
                    vdaf.clone(),
                    leader_task.clone(),
                    aggregation_param.clone(),
                    report.clone(),
                    transcript.clone(),
                );
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();
                    tx.mark_report_aggregated(task.id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param.clone(),
                        (),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(1),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(&ReportAggregation::<
                        VERIFY_KEY_LENGTH,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report.metadata().id(),
                        *report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::WaitingLeader {
                            transition: transcript.leader_prepare_transitions[1]
                                .transition
                                .clone()
                                .unwrap(),
                        },
                    ))
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        active_batch_identifier,
                        aggregation_param.clone(),
                        BatchState::Closing,
                        1,
                        Interval::from_time(report.metadata().time()).unwrap(),
                    ))
                    .await
                    .unwrap();
                    tx.put_batch(&Batch::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        other_batch_identifier,
                        aggregation_param.clone(),
                        BatchState::Closing,
                        1,
                        Interval::EMPTY,
                    ))
                    .await
                    .unwrap();

                    let collection_job = CollectionJob::<
                        VERIFY_KEY_LENGTH,
                        TimeInterval,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        random(),
                        Query::new_time_interval(collection_identifier),
                        aggregation_param,
                        collection_identifier,
                        CollectionJobState::Start,
                    );
                    tx.put_collection_job(&collection_job).await.unwrap();

                    let lease = tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0);

                    Ok((lease, collection_job))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Setup: prepare mocked HTTP responses. (first an error response, then a success)
        // (This is fragile in that it expects the leader request to be deterministically encoded.
        // It would be nicer to retrieve the request bytes from the mock, then do our own parsing &
        // verification -- but mockito does not expose this functionality at time of writing.)
        let leader_request = AggregationJobContinueReq::new(
            AggregationJobStep::from(1),
            Vec::from([PrepareContinue::new(
                *report.metadata().id(),
                transcript.leader_prepare_transitions[1].message.clone(),
            )]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Finished,
        )]));
        let mocked_aggregate_failure = server
            .mock(
                "POST",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"urn:ietf:params:ppm:dap:error:unrecognizedTask\"}")
            .create_async()
            .await;
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_success = server
            .mock(
                "POST",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(CONTENT_TYPE.as_str(), AggregationJobContinueReq::MEDIA_TYPE)
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created twice.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        let error = aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease.clone()))
            .await
            .unwrap_err();
        assert_matches!(
            error,
            Error::Http(error_response) => {
                assert_eq!(error_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(*error_response.dap_problem_type().unwrap(), DapProblemType::UnrecognizedTask);
            }
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_failure.assert_async().await;
        mocked_aggregate_success.assert_async().await;

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                aggregation_param.clone(),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobStep::from(2),
            );
        let want_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                *report.metadata().id(),
                *report.metadata().time(),
                0,
                None,
                ReportAggregationState::Finished,
            );

        let batch_interval_start = report
            .metadata()
            .time()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let want_batch_aggregations = Vec::from([BatchAggregation::<
            VERIFY_KEY_LENGTH,
            TimeInterval,
            Poplar1<XofTurboShake128, 16>,
        >::new(
            *task.id(),
            Interval::new(batch_interval_start, *task.time_precision()).unwrap(),
            aggregation_param.clone(),
            0,
            BatchAggregationState::Aggregating,
            Some(leader_aggregate_share),
            1,
            Interval::from_time(report.metadata().time()).unwrap(),
            ReportIdChecksum::for_report_id(report.metadata().id()),
        )]);
        let want_active_batch =
            Batch::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                active_batch_identifier,
                aggregation_param.clone(),
                BatchState::Closed,
                0,
                Interval::from_time(report.metadata().time()).unwrap(),
            );
        let want_other_batch =
            Batch::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                other_batch_identifier,
                aggregation_param.clone(),
                BatchState::Closing,
                1,
                Interval::EMPTY,
            );

        let (
            got_aggregation_job,
            got_report_aggregation,
            got_batch_aggregations,
            got_active_batch,
            got_other_batch,
            got_collection_job,
        ) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_metadata, aggregation_param, collection_job_id) = (
                    Arc::clone(&vdaf),
                    leader_task.clone(),
                    report.metadata().clone(),
                    aggregation_param.clone(),
                    *want_collection_job.id(),
                );
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, TimeInterval, Poplar1<XofTurboShake128, 16>>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            report_metadata.id(),
                        )
                        .await.unwrap()
                        .unwrap();
                    let batch_aggregations =
                        TimeInterval::get_batch_aggregations_for_collection_identifier::<
                            VERIFY_KEY_LENGTH,
                            Poplar1<XofTurboShake128, 16>,
                            _,
                        >(
                            tx,
                            &task,
                            &vdaf,
                            &Interval::new(
                                report_metadata
                                    .time()
                                    .to_batch_interval_start(task.time_precision())
                                    .unwrap(),
                                *task.time_precision(),
                            )
                            .unwrap(),
                            &aggregation_param,
                        )
                        .await
                        .unwrap();
                    let got_active_batch = tx
                        .get_batch(task.id(), &active_batch_identifier, &aggregation_param)
                        .await.unwrap()
                        .unwrap();
                    let got_other_batch = tx
                        .get_batch(task.id(), &other_batch_identifier, &aggregation_param)
                        .await.unwrap()
                        .unwrap();
                    let got_collection_job = tx
                        .get_collection_job(vdaf.as_ref(), task.id(), &collection_job_id)
                        .await.unwrap()
                        .unwrap();

                    Ok((
                        aggregation_job,
                        report_aggregation,
                        batch_aggregations,
                        got_active_batch,
                        got_other_batch,
                        got_collection_job,
                    ))
                })
            })
            .await
            .unwrap();

        // Map the batch aggregation ordinal value to 0, as it may vary due to sharding.
        let got_batch_aggregations: Vec<_> = got_batch_aggregations
            .into_iter()
            .map(|agg| {
                BatchAggregation::new(
                    *agg.task_id(),
                    *agg.batch_identifier(),
                    aggregation_param.clone(),
                    0,
                    *agg.state(),
                    agg.aggregate_share().cloned(),
                    agg.report_count(),
                    *agg.client_timestamp_interval(),
                    *agg.checksum(),
                )
            })
            .collect();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(want_batch_aggregations, got_batch_aggregations);
        assert_eq!(want_active_batch, got_active_batch);
        assert_eq!(want_other_batch, got_other_batch);
        assert_eq!(want_collection_job, got_collection_job);
    }

    #[tokio::test]
    async fn step_fixed_size_aggregation_job_continue() {
        // Setup: insert a client report and add it to an aggregation job whose state has already
        // been stepped once.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Poplar1::new_turboshake128(1));

        let task = TaskBuilder::new(
            QueryType::FixedSize {
                max_batch_size: 10,
                batch_time_window_size: None,
            },
            VdafInstance::Poplar1 { bits: 1 },
        )
        .with_helper_aggregator_endpoint(server.url().parse().unwrap())
        .build();

        let leader_task = task.leader_view().unwrap();
        let report_metadata = ReportMetadata::new(
            random(),
            clock
                .now()
                .to_batch_interval_start(task.time_precision())
                .unwrap(),
        );
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let aggregation_param = Poplar1AggregationParam::try_from_prefixes(Vec::from([
            IdpfInput::from_bools(&[false]),
        ]))
        .unwrap();
        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &aggregation_param,
            report_metadata.id(),
            &IdpfInput::from_bools(&[true]),
        );

        let agg_auth_token = task.aggregator_auth_token();
        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let batch_id = random();
        let aggregation_job_id = random();
        let leader_aggregate_share = vdaf
            .aggregate(&aggregation_param, [transcript.leader_output_share.clone()])
            .unwrap();

        let (lease, collection_job) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report, aggregation_param, transcript) = (
                    vdaf.clone(),
                    leader_task.clone(),
                    report.clone(),
                    aggregation_param.clone(),
                    transcript.clone(),
                );
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_client_report(vdaf.borrow(), &report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param.clone(),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(1),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(&ReportAggregation::<
                        VERIFY_KEY_LENGTH,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        *report.metadata().id(),
                        *report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::WaitingLeader {
                            transition: transcript.leader_prepare_transitions[1]
                                .transition
                                .clone()
                                .unwrap(),
                        },
                    ))
                    .await
                    .unwrap();

                    tx.put_batch(&Batch::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        batch_id,
                        aggregation_param.clone(),
                        BatchState::Closing,
                        1,
                        Interval::from_time(report.metadata().time()).unwrap(),
                    ))
                    .await
                    .unwrap();

                    let collection_job = CollectionJob::<
                        VERIFY_KEY_LENGTH,
                        FixedSize,
                        Poplar1<XofTurboShake128, 16>,
                    >::new(
                        *task.id(),
                        random(),
                        Query::new_fixed_size(FixedSizeQuery::CurrentBatch),
                        aggregation_param,
                        batch_id,
                        CollectionJobState::Start,
                    );
                    tx.put_collection_job(&collection_job).await.unwrap();

                    let lease = tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap()
                        .remove(0);

                    Ok((lease, collection_job))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        // Setup: prepare mocked HTTP responses. (first an error response, then a success)
        // (This is fragile in that it expects the leader request to be deterministically encoded.
        // It would be nicer to retrieve the request bytes from the mock, then do our own parsing &
        // verification -- but mockito does not expose this functionality at time of writing.)
        let leader_request = AggregationJobContinueReq::new(
            AggregationJobStep::from(1),
            Vec::from([PrepareContinue::new(
                *report.metadata().id(),
                transcript.leader_prepare_transitions[1].message.clone(),
            )]),
        );
        let helper_response = AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Finished,
        )]));
        let mocked_aggregate_failure = server
            .mock(
                "POST",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"urn:ietf:params:ppm:dap:error:unrecognizedTask\"}")
            .create_async()
            .await;
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_success = server
            .mock(
                "POST",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(CONTENT_TYPE.as_str(), AggregationJobContinueReq::MEDIA_TYPE)
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        // Run: create an aggregation job driver & try to step the aggregation we've created twice.
        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        let error = aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease.clone()))
            .await
            .unwrap_err();
        assert_matches!(
            error,
            Error::Http(error_response) => {
                assert_eq!(error_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(*error_response.dap_problem_type().unwrap(), DapProblemType::UnrecognizedTask);
            }
        );
        aggregation_job_driver
            .step_aggregation_job(ds.clone(), Arc::new(lease))
            .await
            .unwrap();

        // Verify.
        mocked_aggregate_failure.assert_async().await;
        mocked_aggregate_success.assert_async().await;

        let want_aggregation_job =
            AggregationJob::<VERIFY_KEY_LENGTH, FixedSize, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                aggregation_param.clone(),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobStep::from(2),
            );
        let want_report_aggregation =
            ReportAggregation::<VERIFY_KEY_LENGTH, Poplar1<XofTurboShake128, 16>>::new(
                *task.id(),
                aggregation_job_id,
                *report.metadata().id(),
                *report.metadata().time(),
                0,
                None,
                ReportAggregationState::Finished,
            );
        let want_batch_aggregations = Vec::from([BatchAggregation::<
            VERIFY_KEY_LENGTH,
            FixedSize,
            Poplar1<XofTurboShake128, 16>,
        >::new(
            *task.id(),
            batch_id,
            aggregation_param.clone(),
            0,
            BatchAggregationState::Aggregating,
            Some(leader_aggregate_share),
            1,
            Interval::from_time(report.metadata().time()).unwrap(),
            ReportIdChecksum::for_report_id(report.metadata().id()),
        )]);
        let want_batch = Batch::<VERIFY_KEY_LENGTH, FixedSize, Poplar1<XofTurboShake128, 16>>::new(
            *task.id(),
            batch_id,
            aggregation_param.clone(),
            BatchState::Closed,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
        );
        let want_collection_job = collection_job.with_state(CollectionJobState::Collectable);

        let (
            got_aggregation_job,
            got_report_aggregation,
            got_batch_aggregations,
            got_batch,
            got_collection_job,
        ) = ds
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_metadata, aggregation_param, collection_job_id) = (
                    Arc::clone(&vdaf),
                    leader_task.clone(),
                    report.metadata().clone(),
                    aggregation_param.clone(),
                    *want_collection_job.id(),
                );
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, FixedSize, Poplar1<XofTurboShake128, 16>>(
                            task.id(),
                            &aggregation_job_id,
                        )
                        .await.unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            aggregation_job.aggregation_parameter(),
                            report_metadata.id(),
                        )
                        .await.unwrap()
                        .unwrap();
                    let batch_aggregations =
                        FixedSize::get_batch_aggregations_for_collection_identifier::<
                            VERIFY_KEY_LENGTH,
                            Poplar1<XofTurboShake128, 16>,
                            _,
                        >(tx, &task, &vdaf, &batch_id, &aggregation_param)
                        .await.unwrap();
                    let batch = tx
                        .get_batch(task.id(), &batch_id, &aggregation_param)
                        .await.unwrap()
                        .unwrap();
                    let collection_job = tx
                        .get_collection_job(vdaf.as_ref(), task.id(), &collection_job_id)
                        .await.unwrap()
                        .unwrap();
                    Ok((
                        aggregation_job,
                        report_aggregation,
                        batch_aggregations,
                        batch,
                        collection_job,
                    ))
                })
            })
            .await
            .unwrap();

        // Map the batch aggregation ordinal value to 0, as it may vary due to sharding.
        let got_batch_aggregations: Vec<_> = got_batch_aggregations
            .into_iter()
            .map(|agg| {
                BatchAggregation::new(
                    *agg.task_id(),
                    *agg.batch_identifier(),
                    aggregation_param.clone(),
                    0,
                    *agg.state(),
                    agg.aggregate_share().cloned(),
                    agg.report_count(),
                    *agg.client_timestamp_interval(),
                    *agg.checksum(),
                )
            })
            .collect();

        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(want_report_aggregation, got_report_aggregation);
        assert_eq!(want_batch_aggregations, got_batch_aggregations);
        assert_eq!(want_batch, got_batch);
        assert_eq!(want_collection_job, got_collection_job);
    }

    struct CancelAggregationJobTestCase {
        task: AggregatorTask,
        vdaf: Arc<Prio3Count>,
        aggregation_job: AggregationJob<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>,
        batch_identifier: Interval,
        report_aggregation: ReportAggregation<VERIFY_KEY_LENGTH, Prio3Count>,
        _ephemeral_datastore: EphemeralDatastore,
        datastore: Arc<Datastore<MockClock>>,
        lease: Lease<AcquiredAggregationJob>,
        mock_helper: ServerGuard,
    }

    async fn setup_cancel_aggregation_job_test() -> CancelAggregationJobTestCase {
        // Setup: insert a client report and add it to a new aggregation job.
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = Arc::new(Prio3::new_count(2).unwrap());
        let mock_helper = mockito::Server::new_async().await;

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .with_helper_aggregator_endpoint(mock_helper.url().parse().unwrap())
            .build()
            .leader_view()
            .unwrap();
        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let batch_identifier = TimeInterval::to_batch_identifier(&task, &(), &time).unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let transcript = run_vdaf(
            vdaf.as_ref(),
            verify_key.as_bytes(),
            &(),
            report_metadata.id(),
            &false,
        );

        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );
        let aggregation_job_id = random();

        let aggregation_job = AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            (),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::InProgress,
            AggregationJobStep::from(0),
        );
        let report_aggregation = report.as_start_leader_report_aggregation(aggregation_job_id, 0);

        let lease = datastore
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report, aggregation_job, report_aggregation) = (
                    vdaf.clone(),
                    task.clone(),
                    report.clone(),
                    aggregation_job.clone(),
                    report_aggregation.clone(),
                );
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await?;
                    tx.put_client_report(vdaf.borrow(), &report).await?;
                    tx.put_aggregation_job(&aggregation_job).await?;
                    tx.put_report_aggregation(&report_aggregation).await?;

                    tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                        *task.id(),
                        batch_identifier,
                        (),
                        BatchState::Open,
                        1,
                        Interval::from_time(report.metadata().time()).unwrap(),
                    ))
                    .await?;

                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await?
                        .remove(0))
                })
            })
            .await
            .unwrap();
        assert_eq!(lease.leased().task_id(), task.id());
        assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

        CancelAggregationJobTestCase {
            task,
            vdaf,
            batch_identifier,
            aggregation_job,
            report_aggregation,
            _ephemeral_datastore: ephemeral_datastore,
            datastore,
            lease,
            mock_helper,
        }
    }

    #[tokio::test]
    async fn cancel_aggregation_job() {
        let mut test_case = setup_cancel_aggregation_job_test().await;

        // Run: create an aggregation job driver & cancel the aggregation job. Mock the helper to
        // verify that we instruct it to delete the aggregation job.
        // https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-09#section-4.5.2.2-20
        let mocked_aggregation_job_delete = test_case
            .mock_helper
            .mock(
                "DELETE",
                test_case
                    .task
                    .aggregation_job_uri(test_case.aggregation_job.id())
                    .unwrap()
                    .unwrap()
                    .path(),
            )
            .with_status(204)
            .create_async()
            .await;

        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        aggregation_job_driver
            .abandon_aggregation_job(Arc::clone(&test_case.datastore), Arc::new(test_case.lease))
            .await
            .unwrap();

        mocked_aggregation_job_delete.assert_async().await;

        // Verify: check that the datastore state is updated as expected (the aggregation job is
        // abandoned, the report aggregation is untouched) and sanity-check that the job can no
        // longer be acquired.
        let want_aggregation_job = test_case
            .aggregation_job
            .with_state(AggregationJobState::Abandoned);
        let want_batch = Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
            *test_case.task.id(),
            test_case.batch_identifier,
            (),
            BatchState::Open,
            0,
            Interval::from_time(test_case.report_aggregation.report_metadata().time()).unwrap(),
        );

        let (got_aggregation_job, got_report_aggregation, got_batch, got_leases) = test_case
            .datastore
            .run_unnamed_tx(|tx| {
                let (vdaf, task, report_id, aggregation_job) = (
                    Arc::clone(&test_case.vdaf),
                    test_case.task.clone(),
                    *test_case.report_aggregation.report_metadata().id(),
                    want_aggregation_job.clone(),
                );
                Box::pin(async move {
                    let aggregation_job = tx
                        .get_aggregation_job::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>(
                            task.id(),
                            aggregation_job.id(),
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let report_aggregation = tx
                        .get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            aggregation_job.id(),
                            aggregation_job.aggregation_parameter(),
                            &report_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let batch = tx
                        .get_batch(task.id(), &test_case.batch_identifier, &())
                        .await
                        .unwrap()
                        .unwrap();
                    let leases = tx
                        .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                        .await
                        .unwrap();
                    Ok((aggregation_job, report_aggregation, batch, leases))
                })
            })
            .await
            .unwrap();
        assert_eq!(want_aggregation_job, got_aggregation_job);
        assert_eq!(test_case.report_aggregation, got_report_aggregation);
        assert_eq!(want_batch, got_batch);
        assert!(got_leases.is_empty());
    }

    #[tokio::test]
    async fn cancel_aggregation_job_helper_aggregation_job_deletion_fails() {
        let mut test_case = setup_cancel_aggregation_job_test().await;

        // DAP does not require that aggregation jobs be deletable and having the leader delete
        // aggregation jobs in the helper on abandonment is merely a SHOULD.
        // https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-09#section-4.5.2.2-20
        // Mock the helper response so that it fails to respond to the delete request. This should
        // not cause the leader to fail.
        let mocked_aggregation_job_delete = test_case
            .mock_helper
            .mock(
                "DELETE",
                test_case
                    .task
                    .aggregation_job_uri(test_case.aggregation_job.id())
                    .unwrap()
                    .unwrap()
                    .path(),
            )
            .with_status(400)
            .create_async()
            .await;

        let aggregation_job_driver = AggregationJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            &noop_meter(),
            32,
        );
        aggregation_job_driver
            .abandon_aggregation_job(Arc::clone(&test_case.datastore), Arc::new(test_case.lease))
            .await
            .unwrap();

        mocked_aggregation_job_delete.assert_async().await;
    }

    #[tokio::test]
    async fn abandon_failing_aggregation_job_with_retryable_error() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let mut runtime_manager = TestRuntimeManager::new();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let stopper = Stopper::new();

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .build();

        let leader_task = task.leader_view().unwrap();
        let agg_auth_token = task.aggregator_auth_token();
        let aggregation_job_id = random();
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();

        let vdaf = Prio3::new_count(2).unwrap();
        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata.id(),
            &false,
        );
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );

        // Set up fixtures in the database.
        ds.run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task = leader_task.clone();
            let report = report.clone();
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&vdaf, &report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<
                    VERIFY_KEY_LENGTH,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *task.id(),
                    aggregation_job_id,
                    (),
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(
                    &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                )
                .await
                .unwrap();

                tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                    *task.id(),
                    batch_identifier,
                    (),
                    BatchState::Open,
                    1,
                    Interval::from_time(report.metadata().time()).unwrap(),
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

        // Set up the aggregation job driver.
        let aggregation_job_driver = Arc::new(AggregationJobDriver::new(
            reqwest::Client::new(),
            &noop_meter(),
            32,
        ));
        let job_driver = Arc::new(
            JobDriver::new(
                clock.clone(),
                runtime_manager.with_label("stepper"),
                noop_meter(),
                stopper.clone(),
                StdDuration::from_secs(1),
                10,
                StdDuration::from_secs(60),
                aggregation_job_driver.make_incomplete_job_acquirer_callback(
                    Arc::clone(&ds),
                    StdDuration::from_secs(600),
                ),
                aggregation_job_driver.make_job_stepper_callback(Arc::clone(&ds), 3),
            )
            .unwrap(),
        );

        // Set up three error responses from our mock helper. These will cause errors in the
        // leader, because the response body is empty and cannot be decoded.
        let (header, value) = agg_auth_token.request_authentication();
        let failure_mock = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .with_status(500)
            .expect(3)
            .create_async()
            .await;
        // Set up an extra response that should never be used, to make sure the job driver doesn't
        // make more requests than we expect. If there were no remaining mocks, mockito would have
        // respond with a fallback error response instead.
        let no_more_requests_mock = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .with_status(500)
            .expect(1)
            .create_async()
            .await;

        // Start up the job driver.
        let task_handle = runtime_manager.with_label("driver").spawn(job_driver.run());

        // Run the job driver until we try to step the collection job four times. The first three
        // attempts make network requests and fail, while the fourth attempt just marks the job
        // as abandoned.
        for i in 1..=4 {
            // Wait for the next task to be spawned and to complete.
            runtime_manager.wait_for_completed_tasks("stepper", i).await;
            // Advance the clock by the lease duration, so that the job driver can pick up the job
            // and try again.
            clock.advance(&Duration::from_seconds(600));
        }
        stopper.stop();
        task_handle.await.unwrap();

        // Check that the job driver made the HTTP requests we expected.
        failure_mock.assert_async().await;
        assert!(!no_more_requests_mock.matched_async().await);

        // Confirm in the database that the job was abandoned.
        let (got_aggregation_job, got_batch) = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    let got_aggregation_job = tx
                        .get_aggregation_job(task.id(), &aggregation_job_id)
                        .await
                        .unwrap()
                        .unwrap();
                    let got_batch = tx
                        .get_batch(task.id(), &batch_identifier, &())
                        .await
                        .unwrap()
                        .unwrap();
                    Ok((got_aggregation_job, got_batch))
                })
            })
            .await
            .unwrap();
        assert_eq!(
            got_aggregation_job,
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Abandoned,
                AggregationJobStep::from(0),
            ),
        );
        assert_eq!(
            got_batch,
            Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                *task.id(),
                batch_identifier,
                (),
                BatchState::Open,
                0,
                Interval::from_time(report.metadata().time()).unwrap(),
            ),
        );
    }

    #[tokio::test]
    async fn abandon_failing_aggregation_job_with_fatal_error() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let mut runtime_manager = TestRuntimeManager::new();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let stopper = Stopper::new();

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .with_helper_aggregator_endpoint(server.url().parse().unwrap())
            .build();

        let leader_task = task.leader_view().unwrap();
        let agg_auth_token = task.aggregator_auth_token();
        let aggregation_job_id = random();
        let verify_key: VerifyKey<VERIFY_KEY_LENGTH> = task.vdaf_verify_key().unwrap();

        let helper_hpke_keypair = generate_test_hpke_config_and_private_key();

        let vdaf = Prio3::new_count(2).unwrap();
        let time = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
        let report_metadata = ReportMetadata::new(random(), time);
        let transcript = run_vdaf(
            &vdaf,
            verify_key.as_bytes(),
            &(),
            report_metadata.id(),
            &false,
        );
        let report = LeaderStoredReport::generate(
            *task.id(),
            report_metadata,
            helper_hpke_keypair.config(),
            Vec::new(),
            &transcript,
        );

        // Set up fixtures in the database.
        ds.run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task = leader_task.clone();
            let report = report.clone();
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&vdaf, &report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<
                    VERIFY_KEY_LENGTH,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *task.id(),
                    aggregation_job_id,
                    (),
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(
                    &report.as_start_leader_report_aggregation(aggregation_job_id, 0),
                )
                .await
                .unwrap();

                tx.put_batch(&Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                    *task.id(),
                    batch_identifier,
                    (),
                    BatchState::Open,
                    1,
                    Interval::from_time(report.metadata().time()).unwrap(),
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

        // Set up the aggregation job driver.
        let aggregation_job_driver = Arc::new(AggregationJobDriver::new(
            reqwest::Client::new(),
            &noop_meter(),
            32,
        ));
        let job_driver = Arc::new(
            JobDriver::new(
                clock.clone(),
                runtime_manager.with_label("stepper"),
                noop_meter(),
                stopper.clone(),
                StdDuration::from_secs(1),
                10,
                StdDuration::from_secs(60),
                aggregation_job_driver.make_incomplete_job_acquirer_callback(
                    Arc::clone(&ds),
                    StdDuration::from_secs(600),
                ),
                aggregation_job_driver.make_job_stepper_callback(Arc::clone(&ds), 3),
            )
            .unwrap(),
        );

        // Set up one fatal error response from our mock helper. These will cause errors in the
        // leader, because the response body is empty and cannot be decoded.
        let (header, value) = agg_auth_token.request_authentication();
        let failure_mock = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .with_status(404)
            .expect(1)
            .create_async()
            .await;
        // Set up an extra response that should never be used, to make sure the job driver doesn't
        // make more requests than we expect. If there were no remaining mocks, mockito would have
        // respond with a fallback error response instead.
        let no_more_requests_mock = server
            .mock(
                "PUT",
                task.aggregation_job_uri(&aggregation_job_id)
                    .unwrap()
                    .path(),
            )
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
            )
            .with_status(500)
            .expect(1)
            .create_async()
            .await;

        // Start up the job driver.
        let task_handle = runtime_manager.with_label("driver").spawn(job_driver.run());
        // Wait for the next task to be spawned and to complete.
        runtime_manager.wait_for_completed_tasks("stepper", 1).await;
        // Advance the clock by the lease duration, so that the job driver can pick up the job
        // and try again.
        clock.advance(&Duration::from_seconds(600));
        stopper.stop();
        task_handle.await.unwrap();

        // Check that the job driver made the HTTP requests we expected.
        failure_mock.assert_async().await;
        assert!(!no_more_requests_mock.matched_async().await);

        // Confirm in the database that the job was abandoned.
        let (got_aggregation_job, got_batch) = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    let got_aggregation_job = tx
                        .get_aggregation_job(task.id(), &aggregation_job_id)
                        .await
                        .unwrap()
                        .unwrap();
                    let got_batch = tx
                        .get_batch(task.id(), &batch_identifier, &())
                        .await
                        .unwrap()
                        .unwrap();
                    Ok((got_aggregation_job, got_batch))
                })
            })
            .await
            .unwrap();
        assert_eq!(
            got_aggregation_job,
            AggregationJob::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                *task.id(),
                aggregation_job_id,
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Abandoned,
                AggregationJobStep::from(0),
            ),
        );
        assert_eq!(
            got_batch,
            Batch::<VERIFY_KEY_LENGTH, TimeInterval, Prio3Count>::new(
                *task.id(),
                batch_identifier,
                (),
                BatchState::Open,
                0,
                Interval::from_time(report.metadata().time()).unwrap(),
            ),
        );
    }
}
