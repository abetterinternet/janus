//! Implements portions of collect sub-protocol for DAP leader and helper.

use crate::aggregator::{
    aggregate_share::compute_aggregate_share, empty_batch_aggregations,
    http_handlers::AGGREGATE_SHARES_ROUTE, query_type::CollectableQueryType,
    send_request_to_helper, Error,
};
use derivative::Derivative;
use futures::future::{try_join_all, BoxFuture};
use janus_aggregator_core::{
    datastore::{
        self,
        models::{AcquiredCollectionJob, BatchAggregationState},
        models::{CollectionJobState, Lease},
        Datastore,
    },
    task,
};
use janus_core::{time::Clock, vdaf_dispatch};
use janus_messages::{
    query_type::{FixedSize, QueryType},
    AggregateShare, AggregateShareReq, BatchSelector,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter, Unit},
    Context, KeyValue, Value,
};
use prio::{
    codec::{Decode, Encode},
    vdaf,
};
use reqwest::Method;
use std::{sync::Arc, time::Duration};
use tokio::try_join;
use tracing::{info, warn};

/// Drives a collection job.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct CollectionJobDriver {
    // Dependencies.
    http_client: reqwest::Client,
    #[derivative(Debug = "ignore")]
    metrics: CollectionJobDriverMetrics,

    // Configuration.
    batch_aggregation_shard_count: u64,
}

impl CollectionJobDriver {
    /// Create a new [`CollectionJobDriver`].
    pub fn new(
        http_client: reqwest::Client,
        meter: &Meter,
        batch_aggregation_shard_count: u64,
    ) -> Self {
        Self {
            http_client,
            metrics: CollectionJobDriverMetrics::new(meter),
            batch_aggregation_shard_count,
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
        match lease.leased().query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.step_collection_job_generic::<
                        VERIFY_KEY_LENGTH,
                        C,
                        FixedSize,
                        VdafType
                    >(datastore, Arc::new(vdaf), lease)
                    .await
                })
            }
        }
    }

    async fn step_collection_job_generic<
        const SEED_SIZE: usize,
        C: Clock,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Arc<Lease<AcquiredCollectionJob>>,
    ) -> Result<(), Error>
    where
        A: 'static,
        A::AggregationParam: Send + Sync,
        A::AggregateShare: 'static + Send + Sync,
        A::OutputShare: PartialEq + Eq + Send + Sync,
    {
        let (task, collection_job, batch_aggregations) = datastore
            .run_tx_with_name("step_collection_job_1", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);
                let batch_aggregation_shard_count = self.batch_aggregation_shard_count;

                Box::pin(async move {
                    // TODO(#224): Consider fleshing out `AcquiredCollectionJob` to include a `Task`,
                    // `A::AggregationParam`, etc. so that we don't have to do more DB queries here.
                    let task = tx
                        .get_task(lease.leased().task_id())
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedTask(*lease.leased().task_id()).into(),
                            )
                        })?;

                    let collection_job = tx
                        .get_collection_job::<SEED_SIZE, Q, A>(
                            vdaf.as_ref(),
                            lease.leased().collection_job_id(),
                        )
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(
                                    *lease.leased().collection_job_id(),
                                )
                                .into(),
                            )
                        })?;

                    // Read batch aggregations, and mark them as read-for-collection to avoid
                    // further aggregation.
                    let batch_aggregations: Vec<_> =
                        Q::get_batch_aggregations_for_collection_identifier(
                            tx,
                            &task,
                            vdaf.as_ref(),
                            collection_job.batch_identifier(),
                            collection_job.aggregation_parameter(),
                        )
                        .await?
                        .into_iter()
                        .map(|ba| ba.with_state(BatchAggregationState::Collected))
                        .collect();

                    // To ensure that concurrent aggregations don't write into a
                    // currently-nonexistent batch aggregation, we write (empty) batch aggregations
                    // for any that have not already been written to storage. We do this
                    // transactionally to avoid the possibility of overwriting other transactions'
                    // updates to batch aggregations.
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

                    Ok((task, collection_job, batch_aggregations))
                })
            })
            .await?;

        if matches!(collection_job.state(), CollectionJobState::Finished { .. }) {
            warn!("collection job being stepped already has a computed helper share");
            self.metrics
                .jobs_already_finished_counter
                .add(&Context::current(), 1, &[]);
            return Ok(());
        }

        let (leader_aggregate_share, report_count, checksum) =
            compute_aggregate_share::<SEED_SIZE, Q, A>(&task, &batch_aggregations)
                .await
                .map_err(|e| datastore::Error::User(e.into()))?;

        // Send an aggregate share request to the helper.
        let req = AggregateShareReq::<Q>::new(
            BatchSelector::new(collection_job.batch_identifier().clone()),
            collection_job.aggregation_parameter().get_encoded(),
            report_count,
            checksum,
        );

        let resp_bytes = send_request_to_helper(
            &self.http_client,
            Method::POST,
            task.aggregate_shares_uri()?,
            AGGREGATE_SHARES_ROUTE,
            AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            req,
            task.primary_aggregator_auth_token(),
            &self.metrics.http_request_duration_histogram,
        )
        .await?;

        // Store the helper aggregate share in the datastore so that a later request to a collect
        // job URI can serve it up.
        let collection_job = Arc::new(
            collection_job.with_state(CollectionJobState::Finished {
                report_count,
                encrypted_helper_aggregate_share: AggregateShare::get_decoded(&resp_bytes)?
                    .encrypted_aggregate_share()
                    .clone(),
                leader_aggregate_share,
            }),
        );

        datastore
            .run_tx_with_name("step_collection_job_2", |tx| {
                let vdaf = Arc::clone(&vdaf);
                let lease = Arc::clone(&lease);
                let collection_job = Arc::clone(&collection_job);
                let metrics = self.metrics.clone();

                Box::pin(async move {
                    let maybe_updated_collection_job = tx
                        .get_collection_job::<SEED_SIZE, Q, A>(vdaf.as_ref(), collection_job.id())
                        .await?
                        .ok_or_else(|| {
                            datastore::Error::User(
                                Error::UnrecognizedCollectionJob(*collection_job.id()).into(),
                            )
                        })?;

                    match maybe_updated_collection_job.state() {
                        CollectionJobState::Collectable => {
                            try_join!(
                                tx.update_collection_job::<SEED_SIZE, Q, A>(&collection_job),
                                tx.release_collection_job(&lease),
                            )?;
                            metrics.jobs_finished_counter.add(&Context::current(), 1, &[]);
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
                            metrics.deleted_jobs_encountered_counter.add(&Context::current(), 1, &[]);
                        }

                        state => {
                            // It shouldn't be possible for a collection job to move to the
                            // abandoned or finished state while this collection job driver held its
                            // lease, and we should not have acquired a lease if we were in the
                            // start state.
                            metrics.unexpected_job_state_counter.add(&Context::current(), 1, &[KeyValue::new("state", Value::from(format!("{state}")))]);
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
        lease: Lease<AcquiredCollectionJob>,
    ) -> Result<(), Error> {
        match lease.leased().query_type() {
            task::QueryType::FixedSize { .. } => {
                vdaf_dispatch!(lease.leased().vdaf(), (vdaf, VdafType, VERIFY_KEY_LENGTH) => {
                    self.abandon_collection_job_generic::<VERIFY_KEY_LENGTH, C, FixedSize, VdafType>(
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
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync + 'static,
    >(
        &self,
        datastore: Arc<Datastore<C>>,
        vdaf: Arc<A>,
        lease: Lease<AcquiredCollectionJob>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        let lease = Arc::new(lease);
        datastore
            .run_tx_with_name("abandon_collection_job", |tx| {
                let (vdaf, lease) = (Arc::clone(&vdaf), Arc::clone(&lease));
                Box::pin(async move {
                    let collection_job = tx
                        .get_collection_job::<SEED_SIZE, Q, A>(
                            &vdaf,
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
                    let update_future = tx.update_collection_job(&collection_job);
                    let release_future = tx.release_collection_job(&lease);
                    try_join!(update_future, release_future)?;
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
                    .run_tx_with_name("acquire_collection_jobs", |tx| {
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
        move |collection_job_lease: Lease<AcquiredCollectionJob>| {
            let (this, datastore) = (Arc::clone(&self), Arc::clone(&datastore));
            Box::pin(async move {
                if collection_job_lease.lease_attempts() > maximum_attempts_before_failure {
                    warn!(
                        attempts = %collection_job_lease.lease_attempts(),
                        max_attempts = %maximum_attempts_before_failure,
                        "Abandoning job due to too many failed attempts"
                    );
                    this.metrics
                        .jobs_abandoned_counter
                        .add(&Context::current(), 1, &[]);
                    return this
                        .abandon_collection_job(datastore, collection_job_lease)
                        .await;
                }

                if collection_job_lease.lease_attempts() > 1 {
                    this.metrics
                        .job_steps_retried_counter
                        .add(&Context::current(), 1, &[]);
                }

                this.step_collection_job(datastore, Arc::new(collection_job_lease))
                    .await
            })
        }
    }
}

/// Holds various metrics instruments for a collection job driver.
#[derive(Clone)]
struct CollectionJobDriverMetrics {
    jobs_finished_counter: Counter<u64>,
    http_request_duration_histogram: Histogram<f64>,
    jobs_abandoned_counter: Counter<u64>,
    jobs_already_finished_counter: Counter<u64>,
    deleted_jobs_encountered_counter: Counter<u64>,
    unexpected_job_state_counter: Counter<u64>,
    job_steps_retried_counter: Counter<u64>,
}

impl CollectionJobDriverMetrics {
    fn new(meter: &Meter) -> Self {
        let jobs_finished_counter = meter
            .u64_counter("janus_collection_jobs_finished")
            .with_description("Count of finished collection jobs.")
            .init();
        jobs_finished_counter.add(&Context::current(), 0, &[]);

        let http_request_duration_histogram = meter
            .f64_histogram("janus_http_request_duration_seconds")
            .with_description(
                "The amount of time elapsed while making an HTTP request to a helper.",
            )
            .with_unit(Unit::new("seconds"))
            .init();

        let jobs_abandoned_counter = meter
            .u64_counter("janus_collection_jobs_abandoned")
            .with_description("Count of abandoned collection jobs.")
            .init();
        jobs_abandoned_counter.add(&Context::current(), 0, &[]);

        let jobs_already_finished_counter = meter
            .u64_counter("janus_collection_jobs_already_finished")
            .with_description(
                "Count of collection jobs for which a lease was acquired but were already \
                 finished.",
            )
            .init();
        jobs_already_finished_counter.add(&Context::current(), 0, &[]);

        let deleted_jobs_encountered_counter = meter
            .u64_counter("janus_collect_deleted_jobs_encountered")
            .with_description(
                "Count of collection jobs that were run to completion but found to have been \
                 deleted.",
            )
            .init();
        deleted_jobs_encountered_counter.add(&Context::current(), 0, &[]);

        let unexpected_job_state_counter = meter
            .u64_counter("janus_collect_unexpected_job_state")
            .with_description(
                "Count of collection jobs that were run to completion but found in an unexpected \
                 state.",
            )
            .init();
        unexpected_job_state_counter.add(&Context::current(), 0, &[]);

        let job_steps_retried_counter = meter
            .u64_counter("janus_job_retries")
            .with_description("Count of retried job steps.")
            .init();
        job_steps_retried_counter.add(&Context::current(), 0, &[]);

        Self {
            jobs_finished_counter,
            http_request_duration_histogram,
            jobs_abandoned_counter,
            jobs_already_finished_counter,
            deleted_jobs_encountered_counter,
            unexpected_job_state_counter,
            job_steps_retried_counter,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        aggregator::{collection_job_driver::CollectionJobDriver, DapProblemType, Error},
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
        task::{test_util::TaskBuilder, QueryType, Task},
        test_util::noop_meter,
    };
    use janus_core::{
        task::VdafInstance,
        test_util::{
            dummy_vdaf::{self, AggregationParam},
            install_test_trace_subscriber,
            runtime::TestRuntimeManager,
        },
        time::{Clock, IntervalExt, MockClock, TimeExt},
        Runtime,
    };
    use janus_messages::{
        query_type::FixedSize, AggregateShare, AggregateShareReq, AggregationJobRound,
        BatchSelector, Duration, HpkeCiphertext, HpkeConfigId, Interval, ReportIdChecksum, Role,
    };
    use prio::codec::{Decode, Encode};
    use rand::random;
    use std::{str, sync::Arc, time::Duration as StdDuration};
    use trillium_tokio::Stopper;
    use url::Url;

    async fn setup_collection_job_test_case(
        server: &mut mockito::Server,
        clock: MockClock,
        datastore: Arc<Datastore<MockClock>>,
        acquire_lease: bool,
    ) -> (
        Task,
        Option<Lease<AcquiredCollectionJob>>,
        CollectionJob<0, FixedSize, dummy_vdaf::Vdaf>,
    ) {
        let time_precision = Duration::from_seconds(500);
        let task = TaskBuilder::new(
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_aggregator_endpoints(Vec::from([
            Url::parse("http://irrelevant").unwrap(), // leader URL doesn't matter
            Url::parse(&server.url()).unwrap(),
        ]))
        .with_time_precision(time_precision)
        .with_min_batch_size(10)
        .build();
        let batch_id = random();
        let aggregation_param = AggregationParam(0);

        let collection_job = CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *task.id(),
            random(),
            batch_id,
            aggregation_param,
            CollectionJobState::Collectable,
        );

        let lease = datastore
            .run_tx(|tx| {
                let (clock, task, collection_job) =
                    (clock.clone(), task.clone(), collection_job.clone());
                Box::pin(async move {
                    tx.put_task(&task).await?;

                    tx.put_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(&collection_job)
                        .await?;

                    let aggregation_job_id = random();
                    let report_timestamp = clock
                        .now()
                        .to_batch_interval_start(task.time_precision())
                        .unwrap();
                    tx.put_aggregation_job(&AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        batch_id,
                        Interval::from_time(&report_timestamp).unwrap(),
                        AggregationJobState::Finished,
                        AggregationJobRound::from(1),
                    ))
                    .await?;

                    let report = LeaderStoredReport::new_dummy(*task.id(), report_timestamp);

                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report)
                        .await?;

                    tx.put_report_aggregation(&ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        *report.metadata().id(),
                        *report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Finished,
                    ))
                    .await?;

                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            batch_id,
                            aggregation_param,
                            0,
                            BatchAggregationState::Aggregating,
                            Some(dummy_vdaf::AggregateShare(0)),
                            10,
                            ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                        ),
                    )
                    .await?;
                    if acquire_lease {
                        let lease = tx
                            .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                            .await?
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
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_aggregator_endpoints(Vec::from([
            Url::parse("http://irrelevant").unwrap(), // leader URL doesn't matter
            Url::parse(&server.url()).unwrap(),
        ]))
        .with_time_precision(time_precision)
        .with_min_batch_size(10)
        .build();
        let agg_auth_token = task.primary_aggregator_auth_token();
        let batch_id = random();
        let aggregation_param = AggregationParam(0);
        let report_timestamp = clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap();

        let (collection_job_id, lease) = ds
            .run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    tx.put_task(&task).await?;

                    let collection_job_id = random();
                    tx.put_collection_job(&CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        collection_job_id,
                        batch_id,
                        aggregation_param,
                        CollectionJobState::Collectable,
                    ))
                    .await?;

                    let aggregation_job_id = random();
                    tx.put_aggregation_job(&AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        batch_id,
                        Interval::from_time(&report_timestamp).unwrap(),
                        AggregationJobState::Finished,
                        AggregationJobRound::from(1),
                    ))
                    .await?;

                    let report = LeaderStoredReport::new_dummy(*task.id(), report_timestamp);

                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report)
                        .await?;

                    tx.put_report_aggregation(&ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        *report.metadata().id(),
                        *report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Finished,
                    ))
                    .await?;

                    let lease = Arc::new(
                        tx.acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                            .await?
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
            &noop_meter(),
            1,
        );

        // No batch aggregations inserted yet.
        let error = collection_job_driver
            .step_collection_job(ds.clone(), Arc::clone(&lease))
            .await
            .unwrap_err();
        assert_matches!(error, Error::InvalidBatchSize(error_task_id, 0) => {
            assert_eq!(task.id(), &error_task_id)
        });

        // Put some batch aggregations in the DB.
        ds.run_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.update_batch_aggregation(
                    &BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        batch_id,
                        aggregation_param,
                        0,
                        BatchAggregationState::Aggregating,
                        Some(dummy_vdaf::AggregateShare(0)),
                        10,
                        ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                    ),
                )
                .await?;

                Ok(())
            })
        })
        .await
        .unwrap();

        let leader_request = AggregateShareReq::new(
            BatchSelector::new_fixed_size(batch_id),
            aggregation_param.get_encoded(),
            10,
            ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
        );

        // Simulate helper failing to service the aggregate share request.
        let mocked_failed_aggregate_share = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .match_header(
                "DAP-Auth-Token",
                str::from_utf8(agg_auth_token.as_ref()).unwrap(),
            )
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded())
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"urn:ietf:params:ppm:dap:error:batchQueriedTooManyTimes\"}")
            .create_async()
            .await;

        let error = collection_job_driver
            .step_collection_job(ds.clone(), Arc::clone(&lease))
            .await
            .unwrap_err();
        assert_matches!(
            error,
            Error::Http {
                problem_details,
                dap_problem_type: Some(DapProblemType::BatchQueriedTooManyTimes),
            } => {
                assert_eq!(problem_details.status.unwrap(), StatusCode::INTERNAL_SERVER_ERROR);
            }
        );

        mocked_failed_aggregate_share.assert_async().await;

        // collection job in datastore should be unchanged.
        ds.run_tx(|tx| {
            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &dummy_vdaf::Vdaf::new(),
                        &collection_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(collection_job.state(), &CollectionJobState::Collectable);
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

        let mocked_aggregate_share = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .match_header(
                "DAP-Auth-Token",
                str::from_utf8(agg_auth_token.as_ref()).unwrap(),
            )
            .match_header(
                CONTENT_TYPE.as_str(),
                AggregateShareReq::<FixedSize>::MEDIA_TYPE,
            )
            .match_body(leader_request.get_encoded())
            .with_status(200)
            .with_header(CONTENT_TYPE.as_str(), AggregateShare::MEDIA_TYPE)
            .with_body(helper_response.get_encoded())
            .create_async()
            .await;

        collection_job_driver
            .step_collection_job(ds.clone(), Arc::clone(&lease))
            .await
            .unwrap();

        mocked_aggregate_share.assert_async().await;

        // Should now have recorded helper encrypted aggregate share, too.
        ds.run_tx(|tx| {
            let helper_aggregate_share = helper_response.encrypted_aggregate_share().clone();
            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &dummy_vdaf::Vdaf::new(),
                        &collection_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();

                assert_matches!(collection_job.state(), CollectionJobState::Finished{ encrypted_helper_aggregate_share, .. } => {
                    assert_eq!(encrypted_helper_aggregate_share, &helper_aggregate_share);
                });

                Ok(())
            })
        })
        .await
        .unwrap();

        // Drive collection job again. It should succeed without contacting the helper.
        collection_job_driver
            .step_collection_job(ds.clone(), lease)
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
            &noop_meter(),
            1,
        );

        // Run: abandon the collection job.
        collection_job_driver
            .abandon_collection_job(Arc::clone(&ds), lease.unwrap())
            .await
            .unwrap();

        // Verify: check that the collection job was abandoned, and that it can no longer be acquired.
        let (abandoned_collection_job, leases) = ds
            .run_tx(|tx| {
                let collection_job = collection_job.clone();
                Box::pin(async move {
                    let abandoned_collection_job = tx
                        .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &dummy_vdaf::Vdaf::new(),
                            collection_job.id(),
                        )
                        .await?
                        .unwrap();

                    let leases = tx
                        .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                        .await?;

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
    async fn abandon_failing_collection_job() {
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
            &noop_meter(),
            1,
        ));
        let job_driver = Arc::new(
            JobDriver::new(
                clock.clone(),
                runtime_manager.with_label("stepper"),
                noop_meter(),
                stopper.clone(),
                StdDuration::from_secs(1),
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
        // leader, because the response body is empty and cannot be decoded.
        let failure_mock = server
            .mock("POST", task.aggregate_shares_uri().unwrap().path())
            .with_status(500)
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
            .run_tx(|tx| {
                let collection_job = collection_job.clone();
                Box::pin(async move {
                    tx.get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &dummy_vdaf::Vdaf::new(),
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

        ds.run_tx(|tx| {
            let collection_job = collection_job.clone();
            Box::pin(async move {
                tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(&collection_job)
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
            .with_body(helper_response.get_encoded())
            .create_async()
            .await;

        let collection_job_driver =
            CollectionJobDriver::new(reqwest::Client::new(), &noop_meter(), 1);

        // Step the collection job. The driver should successfully run the job, but then discard the
        // results when it notices the job has been deleted.
        collection_job_driver
            .step_collection_job(ds.clone(), Arc::new(lease.unwrap()))
            .await
            .unwrap();

        mocked_aggregate_share.assert_async().await;

        // Verify: check that the collection job was abandoned, and that it can no longer be acquired.
        ds.run_tx(|tx| {
            let collection_job = collection_job.clone();
            Box::pin(async move {
                let collection_job = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &dummy_vdaf::Vdaf::new(),
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
}
