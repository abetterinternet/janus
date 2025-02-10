use anyhow::{Context, Error, Result};
use futures::future::{join_all, try_join_all, OptionFuture};
use janus_aggregator_core::datastore::{self, Datastore};
use janus_core::time::Clock;
use janus_messages::TaskId;
use opentelemetry::metrics::{Counter, Meter};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::{sync::Semaphore, try_join};
use tracing::error;

pub struct GarbageCollector<C: Clock> {
    // Dependencies.
    datastore: Arc<Datastore<C>>,

    // Configuration.
    report_limit: u64,
    aggregation_limit: u64,
    collection_limit: u64,
    tasks_per_tx: usize,
    concurrent_tx_semaphore: Option<Semaphore>,

    // Metrics.
    deleted_report_counter: Counter<u64>,
    deleted_aggregation_job_counter: Counter<u64>,
    deleted_batch_counter: Counter<u64>,
}

impl<C: Clock> GarbageCollector<C> {
    pub fn new(
        datastore: Arc<Datastore<C>>,
        meter: &Meter,
        report_limit: u64,
        aggregation_limit: u64,
        collection_limit: u64,
        tasks_per_tx: usize,
        concurrent_tx_limit: Option<usize>,
    ) -> Self {
        let deleted_report_counter = meter
            .u64_counter("janus_gc_deleted_reports")
            .with_description("Count of client reports deleted by the garbage collector.")
            .with_unit("{report}")
            .build();
        let deleted_aggregation_job_counter = meter
            .u64_counter("janus_gc_deleted_aggregation_jobs")
            .with_description("Count of aggregation jobs deleted by the garbage collector.")
            .with_unit("{job}")
            .build();
        let deleted_batch_counter = meter
            .u64_counter("janus_gc_deleted_batches")
            .with_description("Count of batches deleted by the garbage collector.")
            .with_unit("{batch}")
            .build();

        deleted_report_counter.add(0, &[]);
        deleted_aggregation_job_counter.add(0, &[]);
        deleted_batch_counter.add(0, &[]);

        let concurrent_tx_semaphore = concurrent_tx_limit.map(Semaphore::new);

        Self {
            datastore,
            report_limit,
            aggregation_limit,
            collection_limit,
            deleted_report_counter,
            deleted_aggregation_job_counter,
            deleted_batch_counter,
            tasks_per_tx,
            concurrent_tx_semaphore,
        }
    }

    #[tracing::instrument(name = "GarbageCollector::run", skip(self))]
    pub async fn run(&self) -> Result<()> {
        // TODO(#224): add support for handling only a subset of tasks in a single job (i.e. sharding).

        // Retrieve tasks.
        let task_ids: Vec<_> = self
            .datastore
            .run_tx("garbage_collector_get_tasks", |tx| {
                Box::pin(async move { tx.get_aggregator_tasks().await })
            })
            .await
            .context("couldn't retrieve tasks")?
            .into_iter()
            .map(|task| *task.id())
            .collect();

        // Run GC for each task.
        join_all(
            task_ids
                .chunks(self.tasks_per_tx)
                .map(|task_ids| async move {
                    // unwrap safety: we never close concurrent_tx_semaphore.
                    let _permit = OptionFuture::from(
                        self.concurrent_tx_semaphore
                            .as_ref()
                            .map(Semaphore::acquire),
                    )
                    .await
                    .transpose()
                    .expect("concurrent_tx_semaphore has been closed");

                    if let Err(err) = self.gc_tasks(task_ids.to_vec()).await {
                        error!(?err, "GC failure")
                    }
                }),
        )
        .await;
        Ok(())
    }

    #[tracing::instrument(name = "GarbageCollector::gc_tasks", skip(self))]
    async fn gc_tasks(&self, task_ids: Vec<TaskId>) -> Result<()> {
        let task_ids = Arc::new(task_ids);
        let (client_reports_deleted, aggregation_jobs_deleted, batches_deleted) = self
            .datastore
            .run_tx("garbage_collector", |tx| {
                let task_ids = Arc::clone(&task_ids);
                let report_limit = self.report_limit;
                let aggregation_limit = self.aggregation_limit;
                let collection_limit = self.collection_limit;

                Box::pin(async move {
                    let client_reports_deleted = Arc::new(AtomicU64::new(0));
                    let aggregation_jobs_deleted = Arc::new(AtomicU64::new(0));
                    let batches_deleted = Arc::new(AtomicU64::new(0));

                    try_join_all(task_ids.iter().map(|task_id| {
                        let client_reports_deleted = Arc::clone(&client_reports_deleted);
                        let aggregation_jobs_deleted = Arc::clone(&aggregation_jobs_deleted);
                        let batches_deleted = Arc::clone(&batches_deleted);

                        async move {
                            let (report_count, agg_job_count, batch_count) = try_join!(
                                tx.delete_expired_client_reports(task_id, report_limit),
                                tx.delete_expired_aggregation_artifacts(task_id, aggregation_limit),
                                tx.delete_expired_collection_artifacts(task_id, collection_limit),
                            )
                            .with_context(|| format!("Couldn't GC {task_id}"))?;

                            client_reports_deleted.fetch_add(report_count, Ordering::Relaxed);
                            aggregation_jobs_deleted.fetch_add(agg_job_count, Ordering::Relaxed);
                            batches_deleted.fetch_add(batch_count, Ordering::Relaxed);

                            Ok::<_, Error>(())
                        }
                    }))
                    .await
                    .map_err(|err| datastore::Error::User(err.into()))?;

                    Ok((
                        client_reports_deleted.load(Ordering::Relaxed),
                        aggregation_jobs_deleted.load(Ordering::Relaxed),
                        batches_deleted.load(Ordering::Relaxed),
                    ))
                })
            })
            .await?;

        self.deleted_report_counter.add(client_reports_deleted, &[]);
        self.deleted_aggregation_job_counter
            .add(aggregation_jobs_deleted, &[]);
        self.deleted_batch_counter.add(batches_deleted, &[]);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::aggregator::garbage_collector::GarbageCollector;
    use janus_aggregator_core::{
        datastore::{
            models::{
                AggregateShareJob, AggregationJob, AggregationJobState, BatchAggregation,
                BatchAggregationState, CollectionJob, CollectionJobState, LeaderStoredReport,
                ReportAggregation, ReportAggregationState,
            },
            test_util::ephemeral_datastore,
        },
        task::{self, test_util::TaskBuilder, AggregationMode},
        test_util::noop_meter,
    };
    use janus_core::{
        test_util::install_test_trace_subscriber,
        time::{Clock, IntervalExt, MockClock, TimeExt},
        vdaf::VdafInstance,
    };
    use janus_messages::{
        batch_mode::{LeaderSelected, TimeInterval},
        AggregationJobStep, Duration, HpkeCiphertext, HpkeConfigId, Interval, Query,
        ReportIdChecksum, ReportMetadata, ReportShare, Role, Time,
    };
    use prio::vdaf::dummy;
    use rand::random;
    use std::sync::Arc;

    const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
    const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(500);

    #[tokio::test]
    async fn gc_task_leader_time_interval() {
        install_test_trace_subscriber();

        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = dummy::Vdaf::new(1);
        let aggregation_param = dummy::AggregationParam(0);

        // Setup.
        let task = ds
            .run_unnamed_tx(|tx| {
                let clock = clock.clone();
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::BatchMode::TimeInterval,
                        AggregationMode::Synchronous,
                        VdafInstance::Fake { rounds: 1 },
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build()
                    .leader_view()
                    .unwrap();
                    tx.put_aggregator_task(&task).await?;

                    // Client report artifacts.
                    let client_timestamp = clock.now().sub(&Duration::from_seconds(2)).unwrap();
                    let report = LeaderStoredReport::new_dummy(*task.id(), client_timestamp);
                    tx.put_client_report(&report).await.unwrap();

                    // Aggregation artifacts.
                    let aggregation_job_id = random();
                    tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        (),
                        Interval::from_time(&client_timestamp).unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(
                        &report.as_leader_init_report_aggregation(aggregation_job_id, 0),
                    )
                    .await
                    .unwrap();

                    // Collection artifacts.
                    let batch_identifier = Interval::from_time(&client_timestamp).unwrap(); // unrealistic, but induces GC
                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            batch_identifier,
                            dummy::AggregationParam(0),
                            0,
                            batch_identifier,
                            BatchAggregationState::Collected {
                                aggregate_share: Some(dummy::AggregateShare(11)),
                                report_count: 1,
                                checksum: random(),
                                aggregation_jobs_created: 3,
                                aggregation_jobs_terminated: 3,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        random(),
                        Query::new_time_interval(batch_identifier),
                        dummy::AggregationParam(0),
                        batch_identifier,
                        CollectionJobState::Start,
                    ))
                    .await
                    .unwrap();

                    Ok(task)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Run.
        let task = Arc::new(task);
        GarbageCollector::new(
            Arc::clone(&ds),
            &noop_meter(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            1,
            Some(1),
        )
        .gc_tasks(Vec::from([*task.id()]))
        .await
        .unwrap();

        // Reset the clock to "undo" read-based expiry.
        clock.set(OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Verify.
        ds.run_unnamed_tx(|tx| {
            let (vdaf, task) = (vdaf.clone(), Arc::clone(&task));
            Box::pin(async move {
                assert!(tx
                    .get_client_reports_for_task::<0, dummy::Vdaf>(&vdaf, task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_aggregation_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Leader,
                        task.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(&vdaf, task.id(),)
                    .await
                    .unwrap()
                    .is_empty());
                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn gc_task_helper_time_interval() {
        install_test_trace_subscriber();

        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = dummy::Vdaf::new(1);
        let aggregation_param = dummy::AggregationParam(0);

        // Setup.
        let task = ds
            .run_unnamed_tx(|tx| {
                let clock = clock.clone();

                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::BatchMode::TimeInterval,
                        AggregationMode::Synchronous,
                        VdafInstance::Fake { rounds: 1 },
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build()
                    .helper_view()
                    .unwrap();
                    tx.put_aggregator_task(&task).await?;

                    // Client report artifacts.
                    let client_timestamp = clock.now().sub(&Duration::from_seconds(2)).unwrap();
                    let report_share = ReportShare::new(
                        ReportMetadata::new(random(), client_timestamp, Vec::new()),
                        Vec::new(),
                        HpkeCiphertext::new(
                            HpkeConfigId::from(13),
                            Vec::from("encapsulated_context_0"),
                            Vec::from("payload_0"),
                        ),
                    );
                    tx.put_scrubbed_report(
                        task.id(),
                        report_share.metadata().id(),
                        report_share.metadata().time(),
                    )
                    .await
                    .unwrap();

                    // Aggregation artifacts.
                    let aggregation_job_id = random();
                    tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        (),
                        Interval::from_time(&client_timestamp).unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();

                    tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        *report_share.metadata().id(),
                        client_timestamp,
                        0,
                        None,
                        ReportAggregationState::Finished,
                    ))
                    .await
                    .unwrap();

                    // Collection artifacts.
                    let batch_identifier = Interval::from_time(&client_timestamp).unwrap(); // unrealistic, but induces GC
                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            batch_identifier,
                            dummy::AggregationParam(0),
                            0,
                            batch_identifier,
                            BatchAggregationState::Collected {
                                aggregate_share: Some(dummy::AggregateShare(11)),
                                report_count: 1,
                                checksum: random(),
                                aggregation_jobs_created: 5,
                                aggregation_jobs_terminated: 5,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    tx.put_aggregate_share_job(
                        &AggregateShareJob::<0, TimeInterval, dummy::Vdaf>::new(
                            *task.id(),
                            batch_identifier,
                            dummy::AggregationParam(0),
                            dummy::AggregateShare(11),
                            0,
                            ReportIdChecksum::default(),
                        ),
                    )
                    .await
                    .unwrap();

                    Ok(task)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Run.
        let task = Arc::new(task);
        GarbageCollector::new(
            Arc::clone(&ds),
            &noop_meter(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            1,
            Some(1),
        )
        .gc_tasks(Vec::from([*task.id()]))
        .await
        .unwrap();

        // Reset the clock to "undo" read-based expiry.
        clock.set(OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Verify.
        ds.run_unnamed_tx(|tx| {
            let (vdaf, task) = (vdaf.clone(), Arc::clone(&task));
            Box::pin(async move {
                assert!(tx
                    .get_client_reports_for_task::<0, dummy::Vdaf>(&vdaf, task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_aggregation_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Leader,
                        task.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_aggregate_share_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id()
                    )
                    .await
                    .unwrap()
                    .is_empty());
                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn gc_task_leader_leader_selected() {
        install_test_trace_subscriber();

        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = dummy::Vdaf::new(1);
        let aggregation_param = dummy::AggregationParam(0);

        // Setup.
        let task = ds
            .run_unnamed_tx(|tx| {
                let clock = clock.clone();
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::BatchMode::LeaderSelected {
                            batch_time_window_size: None,
                        },
                        AggregationMode::Synchronous,
                        VdafInstance::Fake { rounds: 1 },
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build()
                    .leader_view()
                    .unwrap();
                    tx.put_aggregator_task(&task).await?;

                    // Client report artifacts.
                    let client_timestamp = clock
                        .now()
                        .sub(&REPORT_EXPIRY_AGE)
                        .unwrap()
                        .sub(&Duration::from_seconds(2))
                        .unwrap();
                    let report = LeaderStoredReport::new_dummy(*task.id(), client_timestamp);
                    tx.put_client_report(&report).await.unwrap();

                    // Aggregation artifacts.
                    let batch_id = random();
                    let aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                        *task.id(),
                        random(),
                        aggregation_param,
                        batch_id,
                        Interval::from_time(&client_timestamp).unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    );
                    tx.put_aggregation_job(&aggregation_job).await.unwrap();

                    let report_aggregation =
                        report.as_leader_init_report_aggregation(*aggregation_job.id(), 0);
                    tx.put_report_aggregation(&report_aggregation)
                        .await
                        .unwrap();

                    // Collection artifacts.
                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                            *task.id(),
                            batch_id,
                            dummy::AggregationParam(0),
                            0,
                            Interval::from_time(&client_timestamp).unwrap(),
                            BatchAggregationState::Collected {
                                aggregate_share: Some(dummy::AggregateShare(11)),
                                report_count: 1,
                                checksum: random(),
                                aggregation_jobs_created: 5,
                                aggregation_jobs_terminated: 5,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    tx.put_outstanding_batch(task.id(), &batch_id, &None)
                        .await
                        .unwrap();

                    tx.put_collection_job(&CollectionJob::<0, LeaderSelected, dummy::Vdaf>::new(
                        *task.id(),
                        random(),
                        Query::new_leader_selected(),
                        dummy::AggregationParam(0),
                        batch_id,
                        CollectionJobState::Start,
                    ))
                    .await
                    .unwrap();

                    Ok(task)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Run.
        let task = Arc::new(task);
        GarbageCollector::new(
            Arc::clone(&ds),
            &noop_meter(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            1,
            Some(1),
        )
        .gc_tasks(Vec::from([*task.id()]))
        .await
        .unwrap();

        // Reset the clock to "undo" read-based expiry.
        clock.set(OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Verify.
        ds.run_unnamed_tx(|tx| {
            let (vdaf, task) = (vdaf.clone(), Arc::clone(&task));
            Box::pin(async move {
                assert!(tx
                    .get_client_reports_for_task::<0, dummy::Vdaf>(&vdaf, task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Leader,
                        task.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_batch_aggregations_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_unfilled_outstanding_batches(task.id(), &None)
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_collection_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap()
                    .is_empty());
                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn gc_task_helper_leader_selected() {
        install_test_trace_subscriber();

        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let vdaf = dummy::Vdaf::new(1);
        let aggregation_param = dummy::AggregationParam(0);

        // Setup.
        let task = ds
            .run_unnamed_tx(|tx| {
                let clock = clock.clone();

                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::BatchMode::LeaderSelected {
                            batch_time_window_size: None,
                        },
                        AggregationMode::Synchronous,
                        VdafInstance::Fake { rounds: 1 },
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build()
                    .helper_view()
                    .unwrap();
                    tx.put_aggregator_task(&task).await?;

                    // Client report artifacts.
                    let client_timestamp = clock
                        .now()
                        .sub(&REPORT_EXPIRY_AGE)
                        .unwrap()
                        .sub(&Duration::from_seconds(2))
                        .unwrap();
                    let report_share = ReportShare::new(
                        ReportMetadata::new(random(), client_timestamp, Vec::new()),
                        Vec::new(),
                        HpkeCiphertext::new(
                            HpkeConfigId::from(13),
                            Vec::from("encapsulated_context_0"),
                            Vec::from("payload_0"),
                        ),
                    );
                    tx.put_scrubbed_report(
                        task.id(),
                        report_share.metadata().id(),
                        report_share.metadata().time(),
                    )
                    .await
                    .unwrap();

                    // Aggregation artifacts.
                    let batch_id = random();
                    let aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                        *task.id(),
                        random(),
                        aggregation_param,
                        batch_id,
                        Interval::from_time(&client_timestamp).unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    );
                    tx.put_aggregation_job(&aggregation_job).await.unwrap();

                    let report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
                        *task.id(),
                        *aggregation_job.id(),
                        *report_share.metadata().id(),
                        client_timestamp,
                        0,
                        None,
                        ReportAggregationState::Finished,
                    );
                    tx.put_report_aggregation(&report_aggregation)
                        .await
                        .unwrap();

                    // Collection artifacts.
                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                            *task.id(),
                            batch_id,
                            dummy::AggregationParam(0),
                            0,
                            Interval::from_time(&client_timestamp).unwrap(),
                            BatchAggregationState::Collected {
                                aggregate_share: Some(dummy::AggregateShare(11)),
                                report_count: 1,
                                checksum: random(),
                                aggregation_jobs_created: 6,
                                aggregation_jobs_terminated: 6,
                            },
                        ),
                    )
                    .await
                    .unwrap();

                    tx.put_outstanding_batch(task.id(), &batch_id, &None)
                        .await
                        .unwrap();

                    tx.put_aggregate_share_job(
                        &AggregateShareJob::<0, LeaderSelected, dummy::Vdaf>::new(
                            *task.id(),
                            batch_id,
                            dummy::AggregationParam(0),
                            dummy::AggregateShare(11),
                            0,
                            ReportIdChecksum::default(),
                        ),
                    )
                    .await
                    .unwrap();

                    Ok(task)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Run.
        let task = Arc::new(task);
        GarbageCollector::new(
            Arc::clone(&ds),
            &noop_meter(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            u64::try_from(i64::MAX).unwrap(),
            1,
            Some(1),
        )
        .gc_tasks(Vec::from([*task.id()]))
        .await
        .unwrap();

        // Reset the clock to "undo" read-based expiry.
        clock.set(OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Reset the clock to "undo" read-based expiry.
        clock.set(OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Verify.
        ds.run_unnamed_tx(|tx| {
            let (vdaf, task) = (vdaf.clone(), Arc::clone(&task));
            Box::pin(async move {
                assert!(tx
                    .get_client_reports_for_task::<0, dummy::Vdaf>(&vdaf, task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(task.id())
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Leader,
                        task.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_batch_aggregations_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap()
                    .is_empty());
                assert!(tx
                    .get_aggregate_share_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap()
                    .is_empty());
                Ok(())
            })
        })
        .await
        .unwrap();
    }
}
