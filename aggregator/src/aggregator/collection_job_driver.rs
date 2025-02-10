//! Implements portions of collect sub-protocol for DAP leader and helper.

use crate::aggregator::{
    aggregate_share::compute_aggregate_share, batch_mode::CollectableBatchMode,
    empty_batch_aggregations, http_handlers::AGGREGATE_SHARES_ROUTE, send_request_to_helper, Error,
    RequestBody,
};
use anyhow::bail;
use backoff::backoff::Backoff;
use bytes::Bytes;
use educe::Educe;
use futures::future::{try_join_all, BoxFuture};
use janus_aggregator_core::{
    datastore::{
        self,
        models::{AcquiredCollectionJob, BatchAggregation, CollectionJobState, Lease},
        Datastore,
    },
    task, TIME_HISTOGRAM_BOUNDARIES,
};
use janus_core::{
    retries::{is_retryable_http_client_error, is_retryable_http_status},
    time::Clock,
    vdaf_dispatch,
};
use janus_messages::{
    batch_mode::{BatchMode, LeaderSelected, TimeInterval},
    AggregateShare, AggregateShareReq, BatchSelector,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter},
    KeyValue, Value,
};
use prio::{
    codec::{Decode, Encode},
    dp::DifferentialPrivacyStrategy,
    vdaf,
};
use reqwest::Method;
use std::{sync::Arc, time::Duration};
use tokio::try_join;
use tracing::{error, info, warn};

/// Drives a collection job.
#[derive(Educe)]
#[educe(Debug)]
pub struct CollectionJobDriver<B> {
    // Dependencies.
    http_client: reqwest::Client,
    backoff: B,
    #[educe(Debug(ignore))]
    metrics: CollectionJobDriverMetrics,

    // Configuration.
    batch_aggregation_shard_count: u64,
    /// The retry strategy to use for collection jobs that attempted to be stepped but was not yet
    /// ready due to pending aggregation.
    collection_retry_strategy: RetryStrategy,
}

impl<R> CollectionJobDriver<R>
where
    R: Backoff + Clone + Send + Sync + 'static,
{
    /// Create a new [`CollectionJobDriver`].
    pub fn new(
        http_client: reqwest::Client,
        backoff: R,
        meter: &Meter,
        batch_aggregation_shard_count: u64,
        collection_retry_strategy: RetryStrategy,
    ) -> Self {
        Self {
            http_client,
            backoff,
            metrics: CollectionJobDriverMetrics::new(meter),
            batch_aggregation_shard_count,
            collection_retry_strategy,
        }
    }

    /// Step the provided collection job, for which a lease should have been acquired (though this
    /// should be idempotent). If the collection job runs to completion, the leader share, helper
    /// share, report count and report ID checksum will be written to the `collection_jobs` table,
    /// and a subsequent request to the collection job URI will yield the aggregate shares. The collect
    /// job's lease is released, though it won't matter since the job will no longer be eligible to
    /// be run.
    ///
    /// If some error occurs (including a failure getting the helper's aggregate share), neither
    /// aggregate share is written to the datastore. A subsequent request to the collection job URI
    /// will not yield a result. The collection job lease will eventually expire, allowing a later run
    /// of the collection job driver to try again. Both aggregate shares will be recomputed at that
    /// time.
    #[tracing::instrument(skip(self, datastore), err)]
    pub async fn step_collection_job<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredCollectionJob>>,
    ) -> Result<(), Error> {
        match lease.leased().batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH, dp_strategy, DpStrategy) => {
                    self.step_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        C,
                        TimeInterval,
                        DpStrategy,
                        VdafType
                    >(datastore, Arc::new(vdaf), lease, dp_strategy)
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH, dp_strategy, DpStrategy) => {
                    self.step_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        C,
                        LeaderSelected,
                        DpStrategy,
                        VdafType
                    >(datastore, Arc::new(vdaf), lease, dp_strategy)
                    .await
                })
            }
        }
    }

    async fn step_collection_job_generic<
        const SEED_SIZE: usize,
        C: Clock,
        B: CollectableBatchMode,
        S: DifferentialPrivacyStrategy,
        A: vdaf::AggregatorWithNoise<SEED_SIZE, 16, S> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredCollectionJob>>,
        dp_strategy: S,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: 'static + Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
    {
        let collection_identifier = Arc::new(
            B::BatchIdentifier::get_decoded(lease.leased().encoded_batch_identifier())
                .map_err(Error::MessageDecode)?,
        );
        let aggregation_param = Arc::new(
            A::AggregationParam::get_decoded(lease.leased().encoded_aggregation_param())
                .map_err(Error::MessageDecode)?,
        );

        let rslt = datastore
            .run_tx("step_collection_job_1", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);
                let collection_identifier = Arc::clone(&collection_identifier);
                let aggregation_param = Arc::clone(&aggregation_param);
                let batch_aggregation_shard_count = self.batch_aggregation_shard_count;
                let collection_retry_strategy = self.collection_retry_strategy.clone();
                let metrics = self.metrics.clone();

                Box::pin(async move {
                    // Read the task & collection job.
                    //
                    // Also, try to read an existing, already-FINISHED collection job for the same
                    // batch & aggregation parameter, so that we can reuse an already-computed
                    // result if available.
                    //
                    // Also, look for unaggregated reports & count the number of created/terminated
                    // aggregation jobs relevant to the collection job, so that we can determine if
                    // we are read to complete the collection job in the case there is no
                    // preexisting FINISHED collection job.
                    let (
                        task,
                        collection_job,
                        finished_collection_job,
                        interval_has_unaggregated_reports,
                        (agg_jobs_created, agg_jobs_terminated),
                    ) = try_join!(
                        tx.get_aggregator_task(lease.leased().task_id()),
                        tx.get_collection_job::<SEED_SIZE, B, A>(
                            vdaf.as_ref(),
                            lease.leased().task_id(),
                            lease.leased().collection_job_id()
                        ),
                        tx.get_finished_collection_job::<SEED_SIZE, B, A>(
                            vdaf.as_ref(),
                            lease.leased().task_id(),
                            &collection_identifier,
                            &aggregation_param,
                        ),
                        {
                            let task_id = *lease.leased().task_id();
                            let collection_identifier = Arc::clone(&collection_identifier);

                            async move {
                                if let Some(collection_interval) =
                                    B::to_batch_interval(&collection_identifier)
                                {
                                    tx.interval_has_unaggregated_reports(
                                        &task_id,
                                        collection_interval,
                                    )
                                    .await
                                } else {
                                    Ok(false)
                                }
                            }
                        },
                        B::get_batch_aggregation_job_count_for_collection_identifier::<
                            SEED_SIZE,
                            A,
                            C,
                        >(
                            tx,
                            lease.leased().task_id(),
                            lease.leased().time_precision(),
                            &collection_identifier,
                            &aggregation_param,
                        ),
                    )?;

                    let task = task.ok_or_else(|| {
                        datastore::Error::User(
                            Error::UnrecognizedTask(*lease.leased().task_id()).into(),
                        )
                    })?;

                    let collection_job = collection_job.ok_or_else(|| {
                        datastore::Error::User(
                            Error::UnrecognizedCollectionJob(
                                *task.id(),
                                *lease.leased().collection_job_id(),
                            )
                            .into(),
                        )
                    })?;

                    // If we found a matching finished collection job, borrow its state & exit
                    // early. We don't need to update the batch aggregations because handling for
                    // the finished collection job will have done so already.
                    if let Some(finished_collection_job) = finished_collection_job {
                        let collection_job =
                            collection_job.with_state(finished_collection_job.state().clone());
                        try_join!(
                            tx.update_collection_job::<SEED_SIZE, B, A>(&collection_job),
                            tx.release_collection_job(&lease, None),
                        )?;
                        metrics.jobs_finished_counter.add(1, &[]);
                        return Ok(None);
                    }

                    // Check if any aggregation jobs relevant to this collection job are incomplete,
                    // and whether there are reports still waiting to be associated to an
                    // aggregation job. If so, we have to wait before we can compute the final
                    // aggregate value.
                    if interval_has_unaggregated_reports || agg_jobs_created != agg_jobs_terminated
                    {
                        let retry_delay = collection_retry_strategy
                            .compute_retry_delay(lease.leased().step_attempts());
                        tx.release_collection_job(&lease, Some(&retry_delay))
                            .await?;
                        return Ok(None);
                    }

                    // There is no pre-existing finished collection job, but the collection job is
                    // ready to be completed. Read batch aggregations so that we can compute the
                    // final aggregate value.
                    let mut batch_aggregations =
                        B::get_batch_aggregations_for_collection_identifier(
                            tx,
                            lease.leased().task_id(),
                            lease.leased().time_precision(),
                            vdaf.as_ref(),
                            &collection_identifier,
                            &aggregation_param,
                        )
                        .await?;

                    // Mark batch aggregations as collected to avoid further aggregation. (We don't
                    // need to do this if there is a FINISHED collection job since that job will
                    // have marked the batch aggregations.)
                    //
                    // To ensure that concurrent aggregations don't write into a
                    // currently-nonexistent batch aggregation, we write (empty) batch aggregations
                    // for any that have not already been written to storage. We do this
                    // transactionally to avoid the possibility of overwriting other transactions'
                    // updates to batch aggregations.
                    batch_aggregations = batch_aggregations
                        .into_iter()
                        .map(|ba| ba.collected())
                        .collect::<Result<Vec<_>, _>>()?;

                    let empty_batch_aggregations = empty_batch_aggregations(
                        &task,
                        batch_aggregation_shard_count,
                        collection_job.batch_identifier(),
                        collection_job.aggregation_parameter(),
                        &batch_aggregations,
                    );

                    try_join!(
                        try_join_all(
                            batch_aggregations
                                .iter()
                                .map(|ba| tx.update_batch_aggregation(ba))
                        ),
                        try_join_all(
                            empty_batch_aggregations
                                .iter()
                                .map(|ba| tx.put_batch_aggregation(ba))
                        ),
                    )?;

                    batch_aggregations = batch_aggregations
                        .into_iter()
                        .chain(empty_batch_aggregations.into_iter())
                        .collect();

                    Ok(Some((task, collection_job, batch_aggregations)))
                })
            })
            .await?;

        let (task, collection_job, batch_aggregations) = match rslt {
            Some((task, collection_job, batch_aggregations)) => {
                (task, collection_job, batch_aggregations)
            }
            None => return Ok(()),
        };

        // Compute our aggregate share and ask the Helper to do the same.
        let (mut leader_aggregate_share, report_count, client_timestamp_interval, checksum) =
            compute_aggregate_share::<SEED_SIZE, B, A>(&task, &batch_aggregations)
                .await
                .map_err(|e| datastore::Error::User(e.into()))?;

        vdaf.add_noise_to_agg_share(
            &dp_strategy,
            collection_job.aggregation_parameter(),
            &mut leader_aggregate_share,
            report_count.try_into()?,
        )
        .map_err(Error::DifferentialPrivacy)?;

        // Send an aggregate share request to the helper.
        let http_response = send_request_to_helper(
            &self.http_client,
            self.backoff.clone(),
            Method::POST,
            task.aggregate_shares_uri()?.ok_or_else(|| {
                Error::InvalidConfiguration("task is not leader and has no aggregate share URI")
            })?,
            AGGREGATE_SHARES_ROUTE,
            Some(RequestBody {
                content_type: AggregateShareReq::<B>::MEDIA_TYPE,
                body: Bytes::from(
                    AggregateShareReq::<B>::new(
                        BatchSelector::new(collection_job.batch_identifier().clone()),
                        collection_job
                            .aggregation_parameter()
                            .get_encoded()
                            .map_err(Error::MessageEncode)?,
                        report_count,
                        checksum,
                    )
                    .get_encoded()
                    .map_err(Error::MessageEncode)?,
                ),
            }),
            // The only way a task wouldn't have an aggregator auth token in it is in the taskprov
            // case, and Janus never acts as the leader with taskprov enabled.
            task.aggregator_auth_token()
                .ok_or_else(|| Error::InvalidConfiguration("no aggregator auth token in task"))?,
            &self.metrics.http_request_duration_histogram,
        )
        .await?;

        // Store the helper aggregate share in the datastore so that a later request to a collect
        // job URI can serve it up. Scrub the batch aggregations, as we are now done with them, too.
        let collection_job = Arc::new(
            collection_job.with_state(CollectionJobState::Finished {
                report_count,
                client_timestamp_interval,
                encrypted_helper_aggregate_share: AggregateShare::get_decoded(http_response.body())
                    .map_err(Error::MessageDecode)?
                    .encrypted_aggregate_share()
                    .clone(),
                leader_aggregate_share,
            }),
        );
        let batch_aggregations = Arc::new(
            batch_aggregations
                .into_iter()
                .map(BatchAggregation::scrubbed)
                .collect::<Vec<_>>(),
        );

        datastore
            .run_tx("step_collection_job_2", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);
                let collection_job = Arc::clone(&collection_job);
                let batch_aggregations = Arc::clone(&batch_aggregations);
                let metrics = self.metrics.clone();

                Box::pin(async move {
                    let maybe_updated_collection_job = tx
                        .get_collection_job::<SEED_SIZE, B, A>(
                            vdaf.as_ref(),
                            lease.leased().task_id(),
                            collection_job.id(),
                        )
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(
                                    *lease.leased().task_id(),
                                    *collection_job.id(),
                                ).into(),
                            )
                        })?;

                    match maybe_updated_collection_job.state() {
                        CollectionJobState::Start => {
                            try_join!(
                                tx.update_collection_job::<SEED_SIZE, B, A>(&collection_job),
                                try_join_all(batch_aggregations.iter().map(|ba| async move {
                                    tx.update_batch_aggregation(ba).await
                                })),
                                tx.release_collection_job(&lease, None),
                            )?;
                            metrics.jobs_finished_counter.add( 1, &[]);
                        }

                        CollectionJobState::Deleted => {
                            // If the collection job was deleted between when we acquired it and
                            // now, discard the aggregate shares and leave the job in the deleted
                            // state so that appropriate status can be returned from polling the
                            // collection job URI and GC can run (#313).
                            info!(
                                collection_job_id = %collection_job.id(),
                                "collection job was deleted while lease was held. Discarding aggregate results.",
                            );
                            metrics.deleted_jobs_encountered_counter.add( 1, &[]);
                        }

                        state => {
                            // It shouldn't be possible for a collection job to move to the
                            // abandoned or finished state while this collection job driver held its
                            // lease, and we should not have acquired a lease if we were in the
                            // start state.
                            metrics.unexpected_job_state_counter.add( 1, &[KeyValue::new("state", Value::from(format!("{state}")))]);
                            panic!(
                                "collection job {} unexpectedly in state {}",
                                collection_job.id(), state
                            );
                        }
                    }

                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    #[tracing::instrument(skip(self, datastore), err)]
    pub async fn abandon_collection_job<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease: Arc<Lease<AcquiredCollectionJob>>,
    ) -> Result<(), Error> {
        match lease.leased().batch_mode() {
            task::BatchMode::TimeInterval => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.abandon_collection_job_generic::<VERIFY_KEY_LENGTH, C, TimeInterval, VdafType>(
                        datastore,
                        Arc::new(vdaf),
                        lease,
                    )
                    .await
                })
            }
            task::BatchMode::LeaderSelected { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.abandon_collection_job_generic::<VERIFY_KEY_LENGTH, C, LeaderSelected, VdafType>(
                        datastore,
                        Arc::new(vdaf),
                        lease,
                    )
                    .await
                })
            }
        }
    }

    async fn abandon_collection_job_generic<
        const SEED_SIZE: usize,
        C: Clock,
        B: BatchMode,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredCollectionJob>>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        datastore
            .run_tx("abandon_collection_job", |tx| {
                let (vdaf, lease) = (Arc::clone(&vdaf), Arc::clone(&lease));
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<SEED_SIZE, B, A>(
                            &vdaf,
                            lease.leased().task_id(),
                            lease.leased().collection_job_id(),
                        )
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::DbState(format!(
                                "collection job {} was leased but no collection job was found",
                                lease.leased().collection_job_id(),
                            ))
                        })?
                        .with_state(CollectionJobState::Abandoned);
                    try_join!(
                        tx.update_collection_job(&collection_job),
                        tx.release_collection_job(&lease, None)
                    )?;
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Produce a closure for use as a `[JobDriver::JobAcquirer`].
    pub fn make_incomplete_job_acquirer_callback<C: Clock>(
        &self,
        datastore: Arc<Datastore<C>>,
        lease_duration: Duration,
    ) -> impl Fn(usize) -> BoxFuture<'static, Result<Vec<Lease<AcquiredCollectionJob>>, datastore::Error>>
    {
        move |maximum_acquire_count| {
            let datastore = Arc::clone(&datastore);
            Box::pin(async move {
                datastore
                    .run_tx("acquire_collection_jobs", |tx| {
                        Box::pin(async move {
                            tx.acquire_incomplete_collection_jobs(
                                &lease_duration,
                                maximum_acquire_count,
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
    ) -> impl Fn(Lease<AcquiredCollectionJob>) -> BoxFuture<'static, Result<(), super::Error>> {
        move |lease: Lease<AcquiredCollectionJob>| {
            let (this, datastore) = (Arc::clone(&self), Arc::clone(&datastore));
            let lease = Arc::new(lease);
            Box::pin(async move {
                let attempts = lease.lease_attempts();
                if attempts > maximum_attempts_before_failure {
                    warn!(
                        %attempts,
                        max_attempts = %maximum_attempts_before_failure,
                        "Abandoning job due to too many failed attempts"
                    );
                    this.metrics.jobs_abandoned_counter.add(1, &[]);
                    return this.abandon_collection_job(datastore, lease).await;
                }

                if attempts > 1 {
                    this.metrics.job_steps_retried_counter.add(1, &[]);
                }

                match this
                    .step_collection_job(Arc::clone(&datastore), Arc::clone(&lease))
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(error) => {
                        if !Self::is_retryable_error(&error) {
                            // Make a best-effort attempt to immediately cancel the collection job.
                            // on fatal errors. This protects the helper from performing wasted
                            // work.
                            //
                            // Cancellation might fail, but we will return the first error, since
                            // that's the more interesting error for debugging purposes.
                            //
                            // If cancellation fails, the job will be picked up again. This isn't
                            // a big deal, since stepping a collection job is idempotent. It would
                            // just be some wasted work next time around.
                            warn!(
                                attempts = %attempts,
                                max_attempts = %maximum_attempts_before_failure,
                                ?error,
                                "Abandoning job due to fatal error"
                            );
                            this.metrics.jobs_abandoned_counter.add(1, &[]);
                            if let Err(error) = this.abandon_collection_job(datastore, lease).await
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

    /// Determines whether the given [`Error`] is retryable in the context of collection job
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

/// Holds various metrics instruments for a collection job driver.
#[derive(Clone)]
struct CollectionJobDriverMetrics {
    jobs_finished_counter: Counter<u64>,
    http_request_duration_histogram: Histogram<f64>,
    jobs_abandoned_counter: Counter<u64>,
    deleted_jobs_encountered_counter: Counter<u64>,
    unexpected_job_state_counter: Counter<u64>,
    job_steps_retried_counter: Counter<u64>,
}

impl CollectionJobDriverMetrics {
    fn new(meter: &Meter) -> Self {
        let jobs_finished_counter = meter
            .u64_counter("janus_collection_jobs_finished")
            .with_description("Count of finished collection jobs.")
            .with_unit("{job}")
            .build();
        jobs_finished_counter.add(0, &[]);

        let http_request_duration_histogram = meter
            .f64_histogram("janus_http_request_duration")
            .with_description(
                "The amount of time elapsed while making an HTTP request to a helper.",
            )
            .with_unit("s")
            .with_boundaries(TIME_HISTOGRAM_BOUNDARIES.to_vec())
            .build();

        let jobs_abandoned_counter = meter
            .u64_counter("janus_collection_jobs_abandoned")
            .with_description("Count of abandoned collection jobs.")
            .with_unit("{job}")
            .build();
        jobs_abandoned_counter.add(0, &[]);

        let deleted_jobs_encountered_counter = meter
            .u64_counter("janus_collect_deleted_jobs_encountered")
            .with_description(
                "Count of collection jobs that were run to completion but found to have been \
                 deleted.",
            )
            .with_unit("{job}")
            .build();
        deleted_jobs_encountered_counter.add(0, &[]);

        let unexpected_job_state_counter = meter
            .u64_counter("janus_collect_unexpected_job_state")
            .with_description(
                "Count of collection jobs that were run to completion but found in an unexpected \
                 state.",
            )
            .with_unit("{job}")
            .build();
        unexpected_job_state_counter.add(0, &[]);

        let job_steps_retried_counter = meter
            .u64_counter("janus_job_retries")
            .with_description("Count of retried job steps.")
            .with_unit("{step}")
            .build();
        job_steps_retried_counter.add(0, &[]);

        Self {
            jobs_finished_counter,
            http_request_duration_histogram,
            jobs_abandoned_counter,
            deleted_jobs_encountered_counter,
            unexpected_job_state_counter,
            job_steps_retried_counter,
        }
    }
}

/// An exponential retry strategy.
#[derive(Debug, Clone)]
pub struct RetryStrategy {
    /// The minimum retry delay.
    min_retry_delay: Duration,

    /// The maximum retry delay.
    max_retry_delay: Duration,

    /// The exponential factor to use when computing the next tery delay.
    exponential_factor: f64,
}

impl RetryStrategy {
    /// A no-delay retry strategy.
    #[cfg(test)]
    const NO_DELAY: Self = Self {
        min_retry_delay: Duration::ZERO,
        max_retry_delay: Duration::ZERO,
        exponential_factor: 1.0,
    };

    pub fn new(
        min_retry_delay: Duration,
        max_retry_delay: Duration,
        exponential_factor: f64,
    ) -> Result<Self, anyhow::Error> {
        if min_retry_delay > max_retry_delay {
            bail!("min_retry_delay ({min_retry_delay:?}) > max_retry_delay ({max_retry_delay:?})");
        }
        if !exponential_factor.is_finite() || exponential_factor < 1.0 {
            // is_finite also checks NaN
            bail!("exponential_factor ({exponential_factor}) is less than 1 (or non-finite/NaN)");
        }
        Ok(Self {
            min_retry_delay,
            max_retry_delay,
            exponential_factor,
        })
    }

    fn compute_retry_delay(&self, step_attempt: u64) -> Duration {
        // Compute: min_retry_delay * (exponential_factor ** step_attempt), clamped to
        // [min_retry_delay, max_retry_delay], avoiding overflow & with reasonable behavior if
        // handed pathological parameter choices.

        let min_retry_delay_secs = self.min_retry_delay.as_secs_f64();
        let max_retry_delay_secs = self.max_retry_delay.as_secs_f64();
        let step_attempt = match i32::try_from(step_attempt).ok() {
            Some(step_attempt) => step_attempt,
            None => i32::MAX, // this will surely overflow, but this is handled by the clamp below.
        };

        let delay = min_retry_delay_secs * self.exponential_factor.powi(step_attempt);
        // Panic safety: min > max guarded against above.
        let delay = delay.clamp(min_retry_delay_secs, max_retry_delay_secs);
        // Panic safety: delay is clamped between min_retry_delay_secs & max_retry_delay_secs, both
        // of which come from duration values & therefore can't be negative/infinite nor overflow
        // Duration. (and we assume values between two valid duration values are also valid)
        Duration::from_secs_f64(delay)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        aggregator::{
            collection_job_driver::{CollectionJobDriver, RetryStrategy},
            test_util::BATCH_AGGREGATION_SHARD_COUNT,
            Error,
        },
        binary_utils::job_driver::JobDriver,
    };
    use assert_matches::assert_matches;
    use http::{header::CONTENT_TYPE, StatusCode};
    use janus_aggregator_core::{
        datastore::{
            models::{
                AcquiredCollectionJob, AggregationJob, AggregationJobState, BatchAggregation,
                BatchAggregationState, CollectionJob, CollectionJobState, LeaderStoredReport,
                Lease, ReportAggregation, ReportAggregationState,
            },
            test_util::ephemeral_datastore,
            Datastore,
        },
        task::{
            test_util::{Task, TaskBuilder},
            AggregationMode, BatchMode,
        },
        test_util::noop_meter,
    };
    use janus_core::{
        retries::test_util::LimitedRetryer,
        test_util::{install_test_trace_subscriber, runtime::TestRuntimeManager},
        time::{Clock, IntervalExt, MockClock, TimeExt},
        vdaf::VdafInstance,
        Runtime,
    };
    use janus_messages::{
        batch_mode::TimeInterval, problem_type::DapProblemType, AggregateShare, AggregateShareReq,
        AggregationJobStep, BatchSelector, Duration, HpkeCiphertext, HpkeConfigId, Interval, Query,
        ReportIdChecksum,
    };
    use prio::{
        codec::{Decode, Encode},
        vdaf::dummy,
    };
    use rand::random;
    use std::{sync::Arc, time::Duration as StdDuration};
    use trillium_tokio::Stopper;

    async fn setup_collection_job_test_case(
        server: &mut mockito::Server,
        clock: MockClock,
        datastore: Arc<Datastore<MockClock>>,
        acquire_lease: bool,
    ) -> (
        Task,
        Option<Lease<AcquiredCollectionJob>>,
        CollectionJob<0, TimeInterval, dummy::Vdaf>,
    ) {
        let time_precision = Duration::from_seconds(500);
        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Fake { rounds: 1 },
        )
        .with_helper_aggregator_endpoint(server.url().parse().unwrap())
        .with_time_precision(time_precision)
        .with_min_batch_size(10)
        .build();

        let leader_task = task.leader_view().unwrap();
        let batch_interval = Interval::new(clock.now(), Duration::from_seconds(2000)).unwrap();
        let aggregation_param = dummy::AggregationParam(0);

        let collection_job = CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            random(),
            Query::new_time_interval(batch_interval),
            aggregation_param,
            batch_interval,
            CollectionJobState::Start,
        );

        let lease = datastore
            .run_unnamed_tx(|tx| {
                let (clock, task, collection_job) =
                    (clock.clone(), leader_task.clone(), collection_job.clone());
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();

                    tx.put_collection_job::<0, TimeInterval, dummy::Vdaf>(&collection_job)
                        .await
                        .unwrap();

                    let aggregation_job_id = random();
                    let report_timestamp = clock
                        .now()
                        .to_batch_interval_start(task.time_precision())
                        .unwrap();
                    tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        (),
                        Interval::from_time(&report_timestamp).unwrap(),
                        AggregationJobState::Finished,
                        AggregationJobStep::from(1),
                    ))
                    .await
                    .unwrap();

                    let report = LeaderStoredReport::new_dummy(*task.id(), report_timestamp);

                    tx.put_client_report(&report).await.unwrap();
                    tx.mark_report_aggregated(task.id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        *report.metadata().id(),
                        *report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Finished,
                    ))
                    .await
                    .unwrap();

                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            Interval::new(clock.now(), time_precision).unwrap(),
                            aggregation_param,
                            0,
                            Interval::new(clock.now(), time_precision).unwrap(),
                            BatchAggregationState::Aggregating {
                                aggregate_share: Some(dummy::AggregateShare(0)),
                                report_count: 5,
                                checksum: ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                                aggregation_jobs_created: 1,
                                aggregation_jobs_terminated: 1,
                            },
                        ),
                    )
                    .await
                    .unwrap();
                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            Interval::new(
                                clock.now().add(&Duration::from_seconds(1000)).unwrap(),
                                time_precision,
                            )
                            .unwrap(),
                            aggregation_param,
                            0,
                            Interval::new(
                                clock.now().add(&Duration::from_seconds(1000)).unwrap(),
                                time_precision,
                            )
                            .unwrap(),
                            BatchAggregationState::Aggregating {
                                aggregate_share: Some(dummy::AggregateShare(0)),
                                report_count: 5,
                                checksum: ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
                                aggregation_jobs_created: 1,
                                aggregation_jobs_terminated: 1,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    if acquire_lease {
                        let lease = tx
                            .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                            .await
                            .unwrap()
                            .remove(0);
                        assert_eq!(task.id(), lease.leased().task_id());
                        assert_eq!(collection_job.id(), lease.leased().collection_job_id());
                        Ok(Some(lease))
                    } else {
                        Ok(None)
                    }
                })
            })
            .await
            .unwrap();

        (task, lease, collection_job)
    }

    #[tokio::test]
    async fn drive_collection_job() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let time_precision = Duration::from_seconds(500);
        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Fake { rounds: 1 },
        )
        .with_helper_aggregator_endpoint(server.url().parse().unwrap())
        .with_time_precision(time_precision)
        .with_min_batch_size(10)
        .build();

        let leader_task = task.leader_view().unwrap();
        let agg_auth_token = task.aggregator_auth_token();
        let batch_interval = Interval::new(clock.now(), Duration::from_seconds(2000)).unwrap();
        let aggregation_param = dummy::AggregationParam(0);
        let report_timestamp = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();
        let report = LeaderStoredReport::new_dummy(*task.id(), report_timestamp);

        let (collection_job_id, lease) = ds
            .run_unnamed_tx(|tx| {
                let task = leader_task.clone();
                let clock = clock.clone();
                let report = report.clone();

                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();

                    let collection_job_id = random();
                    tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        collection_job_id,
                        Query::new_time_interval(batch_interval),
                        aggregation_param,
                        batch_interval,
                        CollectionJobState::Start,
                    ))
                    .await
                    .unwrap();

                    let aggregation_job_id = random();
                    tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        (),
                        Interval::from_time(&report_timestamp).unwrap(),
                        AggregationJobState::Finished,
                        AggregationJobStep::from(1),
                    ))
                    .await
                    .unwrap();

                    tx.put_client_report(&report).await.unwrap();

                    tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        *report.metadata().id(),
                        *report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Finished,
                    ))
                    .await
                    .unwrap();

                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            Interval::new(clock.now(), time_precision).unwrap(),
                            aggregation_param,
                            0,
                            Interval::new(clock.now(), time_precision).unwrap(),
                            BatchAggregationState::Aggregating {
                                aggregate_share: Some(dummy::AggregateShare(0)),
                                report_count: 5,
                                checksum: ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                                aggregation_jobs_created: 1,
                                aggregation_jobs_terminated: 1,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            Interval::new(
                                clock.now().add(&Duration::from_seconds(1000)).unwrap(),
                                time_precision,
                            )
                            .unwrap(),
                            aggregation_param,
                            0,
                            Interval::new(
                                clock.now().add(&Duration::from_seconds(1000)).unwrap(),
                                time_precision,
                            )
                            .unwrap(),
                            BatchAggregationState::Aggregating {
                                aggregate_share: Some(dummy::AggregateShare(0)),
                                report_count: 5,
                                checksum: ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
                                aggregation_jobs_created: 1,
                                aggregation_jobs_terminated: 0,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    let lease = Arc::new(
                        tx.acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                            .await
                            .unwrap()
                            .remove(0),
                    );

                    assert_eq!(task.id(), lease.leased().task_id());
                    assert_eq!(&collection_job_id, lease.leased().collection_job_id());
                    Ok((collection_job_id, lease))
                })
            })
            .await
            .unwrap();

        let collection_job_driver = CollectionJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            LimitedRetryer::new(0),
            &noop_meter(),
            BATCH_AGGREGATION_SHARD_COUNT,
            RetryStrategy::NO_DELAY.clone(),
        );

        // Batch aggregations indicate not all aggregation jobs are complete, and there is an
        // unaggregated report in the interval.
        collection_job_driver
            .step_collection_job(Arc::clone(&ds), Arc::clone(&lease))
            .await
            .unwrap();

        // Collection job in datastore should be unchanged, and batch aggregations should still be
        // in Aggregating state. Update the batch aggregations to indicate that all aggregation jobs
        // are complete, and mark the report aggregated. We must reacquire the lease because the
        // last stepping attempt will have released it.
        let lease = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                let clock = clock.clone();
                let report = report.clone();

                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                            &dummy::Vdaf::new(1),
                            task.id(),
                            &collection_job_id,
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(collection_job.state(), &CollectionJobState::Start);

                    let batch_aggregations = tx
                        .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                            &dummy::Vdaf::new(1),
                            task.id(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(batch_aggregations.len(), 2);
                    for batch_aggregation in batch_aggregations {
                        assert_matches!(
                            batch_aggregation.state(),
                            BatchAggregationState::Aggregating { .. }
                        );
                    }

                    tx.mark_report_aggregated(task.id(), report.metadata().id())
                        .await
                        .unwrap();

                    tx.update_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            Interval::new(clock.now(), time_precision).unwrap(),
                            aggregation_param,
                            0,
                            Interval::new(clock.now(), time_precision).unwrap(),
                            BatchAggregationState::Aggregating {
                                aggregate_share: Some(dummy::AggregateShare(0)),
                                report_count: 5,
                                checksum: ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                                aggregation_jobs_created: 1,
                                aggregation_jobs_terminated: 1,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    tx.update_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            Interval::new(
                                clock.now().add(&Duration::from_seconds(1000)).unwrap(),
                                time_precision,
                            )
                            .unwrap(),
                            aggregation_param,
                            0,
                            Interval::new(
                                clock.now().add(&Duration::from_seconds(1000)).unwrap(),
                                time_precision,
                            )
                            .unwrap(),
                            BatchAggregationState::Aggregating {
                                aggregate_share: Some(dummy::AggregateShare(0)),
                                report_count: 5,
                                checksum: ReportIdChecksum::get_decoded(&[2; 32]).unwrap(),
                                aggregation_jobs_created: 1,
                                aggregation_jobs_terminated: 1,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    let lease = Arc::new(
                        tx.acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                            .await
                            .unwrap()
                            .remove(0),
                    );

                    assert_eq!(task.id(), lease.leased().task_id());
                    assert_eq!(&collection_job_id, lease.leased().collection_job_id());

                    Ok(lease)
                })
            })
            .await
            .unwrap();

        let leader_request = AggregateShareReq::new(
            BatchSelector::new_time_interval(batch_interval),
            aggregation_param.get_encoded().unwrap(),
            10,
            ReportIdChecksum::get_decoded(&[3 ^ 2; 32]).unwrap(),
        );

        // Simulate helper failing to service the aggregate share request.
        let (header, value) = agg_auth_token.request_authentication();
        let mocked_failed_aggregate_share = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregateShareReq::<TimeInterval>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"urn:ietf:params:ppm:dap:error:batchQueriedMultipleTimes\"}")
            .create_async()
            .await;

        let error = collection_job_driver
            .step_collection_job(Arc::clone(&ds), Arc::clone(&lease))
            .await
            .unwrap_err();
        assert_matches!(
            error,
            Error::Http(error_response) => {
                assert_matches!(error_response.dap_problem_type(), Some(DapProblemType::BatchQueriedMultipleTimes));
                assert_eq!(error_response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            }
        );

        mocked_failed_aggregate_share.assert_async().await;

        // Collection job in datastore should be unchanged; all batch aggregations should be
        // collected.
        ds.run_unnamed_tx(|tx| {
            let task_id = *task.id();

            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        &task_id,
                        &collection_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(collection_job.state(), &CollectionJobState::Start);

                let batch_aggregations = tx
                    .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        &task_id,
                    )
                    .await
                    .unwrap();
                assert_eq!(batch_aggregations.len(), 128);
                for batch_aggregation in batch_aggregations {
                    assert_matches!(
                        batch_aggregation.state(),
                        BatchAggregationState::Collected { .. }
                    );
                }
                Ok(())
            })
        })
        .await
        .unwrap();

        // Helper aggregate share is opaque to the leader, so no need to construct a real one
        let helper_response = AggregateShare::new(HpkeCiphertext::new(
            HpkeConfigId::from(100),
            Vec::new(),
            Vec::new(),
        ));

        let (header, value) = agg_auth_token.request_authentication();
        let mocked_aggregate_share = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .match_header(header, value.as_str())
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregateShareReq::<TimeInterval>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded().unwrap())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregateShare::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        collection_job_driver
            .step_collection_job(Arc::clone(&ds), Arc::clone(&lease))
            .await
            .unwrap();

        mocked_aggregate_share.assert_async().await;

        // Should now have recorded helper encrypted aggregate share, too; and batch aggregations
        // should be scrubbed.
        ds.run_unnamed_tx(|tx| {
            let clock = clock.clone();
            let task_id = *task.id();
            let helper_aggregate_share = helper_response.encrypted_aggregate_share().clone();

            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        &task_id,
                        &collection_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();

                assert_matches!(collection_job.state(), CollectionJobState::Finished{ report_count, client_timestamp_interval, encrypted_helper_aggregate_share, leader_aggregate_share } => {
                    assert_eq!(report_count, &10);
                    assert_eq!(
                        client_timestamp_interval,
                        &Interval::new(
                            clock.now(),
                            Duration::from_seconds(3 * time_precision.as_seconds()),
                        ).unwrap()
                    );
                    assert_eq!(encrypted_helper_aggregate_share, &helper_aggregate_share);
                    assert_eq!(leader_aggregate_share, &dummy::AggregateShare(0));
                });

                let batch_aggregations = tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(&dummy::Vdaf::new(1), &task_id).await.unwrap();
                assert_eq!(batch_aggregations.len(), 128);
                for batch_aggregation in batch_aggregations {
                    assert_matches!(batch_aggregation.state(), BatchAggregationState::Scrubbed);
                }

                Ok(())
            })
        })
        .await
        .unwrap();

        // Put another collection job for the same interval & aggregation parameter. Validate that
        // we can still drive it to completion, and that we get the same result, without contacting
        // the Helper.
        let (collection_job_id, lease) = ds
            .run_unnamed_tx(|tx| {
                let task_id = *leader_task.id();

                Box::pin(async move {
                    let collection_job_id = random();
                    tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                        task_id,
                        collection_job_id,
                        Query::new_time_interval(batch_interval),
                        aggregation_param,
                        batch_interval,
                        CollectionJobState::Start,
                    ))
                    .await
                    .unwrap();

                    let lease = Arc::new(
                        tx.acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                            .await
                            .unwrap()
                            .remove(0),
                    );

                    assert_eq!(&task_id, lease.leased().task_id());
                    assert_eq!(&collection_job_id, lease.leased().collection_job_id());
                    Ok((collection_job_id, lease))
                })
            })
            .await
            .unwrap();

        collection_job_driver
            .step_collection_job(Arc::clone(&ds), lease)
            .await
            .unwrap();

        ds.run_unnamed_tx(|tx| {
            let clock = clock.clone();
            let task_id = *task.id();
            let helper_aggregate_share = helper_response.encrypted_aggregate_share().clone();

            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        &task_id,
                        &collection_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();

                assert_matches!(collection_job.state(), CollectionJobState::Finished{ report_count, client_timestamp_interval, encrypted_helper_aggregate_share, leader_aggregate_share } => {
                    assert_eq!(report_count, &10);
                    assert_eq!(
                        client_timestamp_interval,
                        &Interval::new(
                            clock.now(),
                            Duration::from_seconds(3 * time_precision.as_seconds()),
                        ).unwrap()
                    );
                    assert_eq!(encrypted_helper_aggregate_share, &helper_aggregate_share);
                    assert_eq!(leader_aggregate_share, &dummy::AggregateShare(0));
                });

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn abandon_collection_job() {
        // Setup: insert a collection job into the datastore.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let (_, lease, collection_job) =
            setup_collection_job_test_case(&mut server, clock, Arc::clone(&ds), true).await;

        let collection_job_driver = CollectionJobDriver::new(
            reqwest::Client::builder().build().unwrap(),
            LimitedRetryer::new(1),
            &noop_meter(),
            BATCH_AGGREGATION_SHARD_COUNT,
            RetryStrategy::NO_DELAY.clone(),
        );

        // Run: abandon the collection job.
        let lease = Arc::new(lease.unwrap());
        collection_job_driver
            .abandon_collection_job(Arc::clone(&ds), lease)
            .await
            .unwrap();

        // Verify: check that the collection job was abandoned, and that it can no longer be acquired.
        let (abandoned_collection_job, leases) = ds
            .run_unnamed_tx(|tx| {
                let collection_job = collection_job.clone();
                Box::pin(async move {
                    let abandoned_collection_job = tx
                        .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                            &dummy::Vdaf::new(1),
                            collection_job.task_id(),
                            collection_job.id(),
                        )
                        .await
                        .unwrap()
                        .unwrap();

                    let leases = tx
                        .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                        .await
                        .unwrap();

                    Ok((abandoned_collection_job, leases))
                })
            })
            .await
            .unwrap();
        assert_eq!(
            abandoned_collection_job,
            collection_job.with_state(CollectionJobState::Abandoned),
        );
        assert!(leases.is_empty());
    }

    #[tokio::test]
    async fn abandon_failing_collection_job_with_fatal_error() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let mut runtime_manager = TestRuntimeManager::new();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let stopper = Stopper::new();

        let (task, _, collection_job) =
            setup_collection_job_test_case(&mut server, clock.clone(), Arc::clone(&ds), false)
                .await;

        // Set up the collection job driver
        let collection_job_driver = Arc::new(CollectionJobDriver::new(
            reqwest::Client::new(),
            LimitedRetryer::new(0),
            &noop_meter(),
            BATCH_AGGREGATION_SHARD_COUNT,
            RetryStrategy::NO_DELAY.clone(),
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
                collection_job_driver.make_incomplete_job_acquirer_callback(
                    Arc::clone(&ds),
                    StdDuration::from_secs(600),
                ),
                collection_job_driver.make_job_stepper_callback(Arc::clone(&ds), 3),
            )
            .unwrap(),
        );

        // Set up an error response from the server that returns a non-retryable error.
        let failure_mock = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .with_status(404)
            .expect(1)
            .create_async()
            .await;
        // Set up an extra response that should never be used, to make sure the job driver doesn't
        // make more requests than we expect. If there were no remaining mocks, mockito would have
        // respond with a fallback error response instead.
        let no_more_requests_mock = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .with_status(502)
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

        // Shut down the job driver.
        stopper.stop();
        task_handle.await.unwrap();

        // Check that the job driver made the HTTP requests we expected.
        failure_mock.assert_async().await;
        assert!(!no_more_requests_mock.matched_async().await);

        // Confirm that the collection job was abandoned.
        let collection_job_after = ds
            .run_unnamed_tx(|tx| {
                let collection_job = collection_job.clone();
                Box::pin(async move {
                    tx.get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        collection_job.task_id(),
                        collection_job.id(),
                    )
                    .await
                })
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            collection_job_after,
            collection_job.with_state(CollectionJobState::Abandoned),
        );
    }

    #[tokio::test]
    async fn abandon_failing_collection_job_with_retryable_error() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let mut runtime_manager = TestRuntimeManager::new();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let stopper = Stopper::new();

        let (task, _, collection_job) =
            setup_collection_job_test_case(&mut server, clock.clone(), Arc::clone(&ds), false)
                .await;

        // Set up the collection job driver
        let collection_job_driver = Arc::new(CollectionJobDriver::new(
            reqwest::Client::new(),
            LimitedRetryer::new(0),
            &noop_meter(),
            BATCH_AGGREGATION_SHARD_COUNT,
            RetryStrategy::NO_DELAY.clone(),
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
                collection_job_driver.make_incomplete_job_acquirer_callback(
                    Arc::clone(&ds),
                    StdDuration::from_secs(600),
                ),
                collection_job_driver.make_job_stepper_callback(Arc::clone(&ds), 3),
            )
            .unwrap(),
        );

        // Set up three error responses from our mock helper. These will cause errors in the
        // leader, because the response body is empty and cannot be decoded. The error status
        // indicates that the error is retryable.
        let failure_mock = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .with_status(502)
            .expect(3)
            .create_async()
            .await;
        // Set up an extra response that should never be used, to make sure the job driver doesn't
        // make more requests than we expect. If there were no remaining mocks, mockito would have
        // respond with a fallback error response instead.
        let no_more_requests_mock = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
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
        // Shut down the job driver.
        stopper.stop();
        task_handle.await.unwrap();

        // Check that the job driver made the HTTP requests we expected.
        failure_mock.assert_async().await;
        assert!(!no_more_requests_mock.matched_async().await);

        // Confirm that the collection job was abandoned.
        let collection_job_after = ds
            .run_unnamed_tx(|tx| {
                let collection_job = collection_job.clone();
                Box::pin(async move {
                    tx.get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        collection_job.task_id(),
                        collection_job.id(),
                    )
                    .await
                })
            })
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            collection_job_after,
            collection_job.with_state(CollectionJobState::Abandoned),
        );
    }

    #[tokio::test]
    async fn delete_collection_job() {
        // Setup: insert a collection job into the datastore.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let (task, lease, collection_job) =
            setup_collection_job_test_case(&mut server, clock, Arc::clone(&ds), true).await;

        // Delete the collection job
        let collection_job = collection_job.with_state(CollectionJobState::Deleted);

        ds.run_unnamed_tx(|tx| {
            let collection_job = collection_job.clone();
            Box::pin(async move {
                tx.update_collection_job::<0, TimeInterval, dummy::Vdaf>(&collection_job)
                    .await
            })
        })
        .await
        .unwrap();

        // Helper aggregate share is opaque to the leader, so no need to construct a real one
        let helper_response = AggregateShare::new(HpkeCiphertext::new(
            HpkeConfigId::from(100),
            Vec::new(),
            Vec::new(),
        ));

        let mocked_aggregate_share = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregateShare::MEDIA_TYPE)
            .with_body(helper_response.get_encoded().unwrap())
            .create_async()
            .await;

        let collection_job_driver = CollectionJobDriver::new(
            reqwest::Client::new(),
            LimitedRetryer::new(0),
            &noop_meter(),
            BATCH_AGGREGATION_SHARD_COUNT,
            RetryStrategy::NO_DELAY.clone(),
        );

        // Step the collection job. The driver should successfully run the job, but then discard the
        // results when it notices the job has been deleted.
        collection_job_driver
            .step_collection_job(ds.clone(), Arc::new(lease.unwrap()))
            .await
            .unwrap();

        mocked_aggregate_share.assert_async().await;

        // Verify: check that the collection job was abandoned, and that it can no longer be acquired.
        ds.run_unnamed_tx(|tx| {
            let collection_job = collection_job.clone();
            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                        &dummy::Vdaf::new(1),
                        collection_job.task_id(),
                        collection_job.id(),
                    )
                    .await
                    .unwrap()
                    .unwrap();

                assert_eq!(collection_job.state(), &CollectionJobState::Deleted);

                let leases = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                    .await
                    .unwrap();

                assert!(leases.is_empty());

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[test]
    fn retry_strategy() {
        // Acceptable parameters.
        for (step_attempts, min_delay_s, max_delay_s, exponential_factor, want_delay_s) in [
            (0, 100, 1000, 1.1, 100),         // no steps
            (1, 100, 1000, 1.1, 110),         // 1 step
            (10, 100, 1000, 1.1, 259),        // 10 steps
            (10_000, 100, 1000, 1.1, 1000),   // 10,000 steps
            (u64::MAX, 100, 1000, 1.1, 1000), // more steps than can fit in an i32
        ] {
            // We truncate the result of `compute_reacquire_delay` to the nearest second to mitigate
            // floating-point issues.
            let got_delay_s = RetryStrategy::new(
                StdDuration::from_secs(min_delay_s),
                StdDuration::from_secs(max_delay_s),
                exponential_factor,
            )
            .unwrap()
            .compute_retry_delay(step_attempts)
            .as_secs();

            assert_eq!(
                want_delay_s,
                got_delay_s,
                "RetryDelay({min_delay_s}, {max_delay_s}, {exponential_factor}).compute_retry_delay({step_attempts})"
            );
        }

        // Bad parameters.
        for (min_delay_s, max_delay_s, exponential_factor) in [
            (1000, 100, 1.1),               // min_delay > max_delay
            (100, 1000, 0.9),               // exponential_factor < 1
            (100, 1000, -1.1),              // exponential factor negative
            (100, 1000, f64::NAN),          // exponential factor is NaN
            (100, 1000, f64::INFINITY),     // exponential factor is infinity
            (100, 1000, f64::NEG_INFINITY), // exponential factor is -infinity
        ] {
            RetryStrategy::new(
                StdDuration::from_secs(min_delay_s),
                StdDuration::from_secs(max_delay_s),
                exponential_factor,
            )
            .expect_err(&format!(
                "RetryDelay({min_delay_s}, {max_delay_s}, {exponential_factor})"
            ));
        }
    }
}
