use crate::aggregator::{
    error::ReportRejectionReason,
    test_util::{create_report, create_report_custom, default_aggregator_config},
    Aggregator, Config, Error,
};
use assert_matches::assert_matches;
use futures::future::try_join_all;
use janus_aggregator_core::{
    datastore::{
        models::{CollectionJob, CollectionJobState, TaskUploadCounter},
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{
        test_util::{Task, TaskBuilder},
        AggregationMode, BatchMode,
    },
    test_util::noop_meter,
};
use janus_core::{
    hpke::{self, HpkeApplicationInfo, HpkeKeypair, Label},
    test_util::{
        install_test_trace_subscriber,
        runtime::{TestRuntime, TestRuntimeManager},
    },
    time::{Clock, MockClock, TimeExt},
    vdaf::{VdafInstance, VERIFY_KEY_LENGTH_PRIO3},
    Runtime,
};
use janus_messages::{
    batch_mode::TimeInterval, Duration, HpkeCiphertext, HpkeConfigId, InputShareAad, Interval,
    PlaintextInputShare, Query, Report, Role,
};
use prio::{codec::Encode, vdaf::prio3::Prio3Count};
use rand::random;
use std::{collections::HashSet, iter, sync::Arc, time::Duration as StdDuration};

struct UploadTest {
    vdaf: Prio3Count,
    aggregator: Aggregator<MockClock>,
    clock: MockClock,
    task: Task,
    datastore: Arc<Datastore<MockClock>>,
    ephemeral_datastore: EphemeralDatastore,
    hpke_keypair: HpkeKeypair,
}

impl UploadTest {
    async fn new(cfg: Config) -> Self {
        Self::new_with_runtime(cfg, TestRuntime::default()).await
    }

    async fn new_with_runtime<R>(cfg: Config, runtime: R) -> Self
    where
        R: Runtime + Send + Sync + 'static,
    {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let vdaf = Prio3Count::new_count(2).unwrap();
        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        )
        .build();

        let leader_task = task.leader_view().unwrap();

        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
        let hpke_keypair = datastore.put_hpke_key().await.unwrap();
        datastore.put_aggregator_task(&leader_task).await.unwrap();

        let aggregator = Aggregator::new(
            Arc::clone(&datastore),
            clock.clone(),
            runtime,
            &noop_meter(),
            cfg,
        )
        .await
        .unwrap();

        Self {
            vdaf,
            aggregator,
            clock,
            task,
            datastore,
            ephemeral_datastore,
            hpke_keypair,
        }
    }
}

#[tokio::test]
async fn upload() {
    let UploadTest {
        vdaf,
        aggregator,
        clock,
        task,
        datastore: ds,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new(Config {
        max_upload_batch_size: 1000,
        max_upload_batch_write_delay: StdDuration::from_millis(500),
        ..Default::default()
    })
    .await;

    let leader_task = task.leader_view().unwrap();
    let report = create_report(&leader_task, &hpke_keypair, clock.now());

    aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap();

    let got_report = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task_id = *task.id();
            let report_id = *report.metadata().id();
            Box::pin(async move { tx.get_client_report(&vdaf, &task_id, &report_id).await })
        })
        .await
        .unwrap()
        .unwrap();
    assert!(got_report.eq_report(&vdaf, &hpke_keypair, &report));

    // Report uploads are idempotent.
    aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap();

    // Even if the report is modified, it is still reported as a duplicate. The original report
    // is stored.
    let mutated_report = create_report_custom(
        &leader_task,
        clock.now(),
        *report.metadata().id(),
        &hpke_keypair,
    );
    aggregator
        .handle_upload(task.id(), &mutated_report.get_encoded().unwrap())
        .await
        .unwrap();

    // Verify that the original report, rather than the modified report, is stored.
    let (got_report, got_counter) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task_id = *task.id();
            let report_id = *report.metadata().id();
            Box::pin(async move {
                Ok((
                    tx.get_client_report(&vdaf, &task_id, &report_id)
                        .await
                        .unwrap(),
                    tx.get_task_upload_counter(&task_id).await.unwrap(),
                ))
            })
        })
        .await
        .unwrap();
    assert!(got_report.unwrap().eq_report(&vdaf, &hpke_keypair, &report));

    assert_eq!(
        got_counter,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 0, 1, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_batch() {
    const BATCH_SIZE: usize = 100;
    let UploadTest {
        vdaf,
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new(Config {
        max_upload_batch_size: BATCH_SIZE,
        max_upload_batch_write_delay: StdDuration::from_secs(86400),
        ..Default::default()
    })
    .await;

    let reports: Vec<_> = iter::repeat_with(|| {
        create_report(&task.leader_view().unwrap(), &hpke_keypair, clock.now())
    })
    .take(BATCH_SIZE)
    .collect();
    let want_report_ids: HashSet<_> = reports.iter().map(|r| *r.metadata().id()).collect();

    let aggregator = Arc::new(aggregator);
    try_join_all(reports.iter().map(|r| {
        let aggregator = Arc::clone(&aggregator);
        let enc = r.get_encoded().unwrap();
        let task_id = task.id();
        async move { aggregator.handle_upload(task_id, &enc).await }
    }))
    .await
    .unwrap();

    let got_report_ids = datastore
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task = task.clone();
            Box::pin(async move { tx.get_client_reports_for_task(&vdaf, task.id()).await })
        })
        .await
        .unwrap()
        .iter()
        .map(|r| *r.metadata().id())
        .collect();

    assert_eq!(want_report_ids, got_report_ids);

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 0, 100, 0, 0, 0
        ))
    );
}

#[tokio::test]
async fn upload_wrong_hpke_config_id() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;
    let leader_task = task.leader_view().unwrap();
    let report = create_report(&leader_task, &hpke_keypair, clock.now());

    let unused_hpke_config_id =
        HpkeConfigId::from(u8::from(*hpke_keypair.config().id()).wrapping_add(1));

    let report = Report::new(
        report.metadata().clone(),
        report.public_share().to_vec(),
        HpkeCiphertext::new(
            unused_hpke_config_id,
            report
                .leader_encrypted_input_share()
                .encapsulated_key()
                .to_vec(),
            report.leader_encrypted_input_share().payload().to_vec(),
        ),
        report.helper_encrypted_input_share().clone(),
    );

    let result = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(result.as_ref(), Error::ReportRejected(rejection) => {
        assert_eq!(task.id(), rejection.task_id());
        assert_eq!(report.metadata().id(), rejection.report_id());
        assert_eq!(report.metadata().time(), rejection.time());
        assert_matches!(rejection.reason(), ReportRejectionReason::OutdatedHpkeConfig(id) => {
            assert_eq!(id, &unused_hpke_config_id);
        })
    });

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 1, 0, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_in_the_future_boundary_condition() {
    let UploadTest {
        vdaf,
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new(default_aggregator_config()).await;
    let report = create_report(
        &task.leader_view().unwrap(),
        &hpke_keypair,
        clock.now().add(task.tolerable_clock_skew()).unwrap(),
    );

    aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap();

    let got_report = datastore
        .run_unnamed_tx(|tx| {
            let (vdaf, task_id, report_id) = (vdaf.clone(), *task.id(), *report.metadata().id());
            Box::pin(async move { tx.get_client_report(&vdaf, &task_id, &report_id).await })
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(task.id(), got_report.task_id());
    assert_eq!(report.metadata(), got_report.metadata());

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 0, 1, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_in_the_future_past_clock_skew() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;
    let report = create_report(
        &task.leader_view().unwrap(),
        &hpke_keypair,
        clock
            .now()
            .add(task.tolerable_clock_skew())
            .unwrap()
            .add(&Duration::from_seconds(1))
            .unwrap(),
    );

    let upload_error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(upload_error.as_ref(), Error::ReportRejected(rejection) => {
        assert_eq!(task.id(), rejection.task_id());
        assert_eq!(report.metadata().id(), rejection.report_id());
        assert_eq!(report.metadata().time(), rejection.time());
        assert_matches!(rejection.reason(), ReportRejectionReason::TooEarly);
    });

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 0, 0, 1, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_for_collected_batch() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;
    let report = create_report(&task.leader_view().unwrap(), &hpke_keypair, clock.now());

    // Insert a collection job for the batch interval including our report.
    let batch_interval = Interval::new(
        report
            .metadata()
            .time()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        *task.time_precision(),
    )
    .unwrap();
    datastore
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.put_collection_job(&CollectionJob::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *task.id(),
                    random(),
                    Query::new_time_interval(batch_interval),
                    (),
                    batch_interval,
                    CollectionJobState::Start,
                ))
                .await
            })
        })
        .await
        .unwrap();

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::IntervalCollected);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            1, 0, 0, 0, 0, 0, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_task_not_started() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;

    // Set the task start time to the future, and generate & upload a report from before that time.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_task_start(Some(
        clock.now().add(&Duration::from_seconds(3600)).unwrap(),
    ))
    .build()
    .leader_view()
    .unwrap();
    datastore.put_aggregator_task(&task).await.unwrap();

    let report = create_report(&task, &hpke_keypair, clock.now());

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::TaskNotStarted);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 0, 0, 0, 1, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_task_ended() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_task_end(Some(clock.now()))
    .build()
    .leader_view()
    .unwrap();
    datastore.put_aggregator_task(&task).await.unwrap();

    // Advance the clock to end the task.
    clock.advance(&Duration::from_seconds(1));
    let report = create_report(&task, &hpke_keypair, clock.now());

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::TaskEnded);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 0, 0, 0, 0, 0, 1
        ))
    )
}

#[tokio::test]
async fn upload_report_report_expired() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_report_expiry_age(Some(Duration::from_seconds(60)))
    .build()
    .leader_view()
    .unwrap();
    datastore.put_aggregator_task(&task).await.unwrap();

    let report = create_report(&task, &hpke_keypair, clock.now());

    // Advance the clock to expire the report.
    clock.advance(&Duration::from_seconds(61));

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::Expired);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 0, 1, 0, 0, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_faulty_encryption() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;

    let task = task.leader_view().unwrap();

    // Encrypt with the wrong key.
    let report = create_report_custom(
        &task,
        clock.now(),
        random(),
        &HpkeKeypair::test_with_id(*hpke_keypair.config().id()),
    );

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::DecryptFailure);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 0, 1, 0, 0, 0, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_public_share_decode_failure() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;

    let task = task.leader_view().unwrap();

    let mut report = create_report(&task, &hpke_keypair, clock.now());
    report = Report::new(
        report.metadata().clone(),
        // Some obviously wrong public share.
        Vec::from([0; 10]),
        report.leader_encrypted_input_share().clone(),
        report.helper_encrypted_input_share().clone(),
    );

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::DecodeFailure);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 1, 0, 0, 0, 0, 0, 0, 0
        ))
    )
}

#[tokio::test]
async fn upload_report_leader_input_share_decode_failure() {
    let mut runtime_manager = TestRuntimeManager::new();
    let UploadTest {
        aggregator,
        clock,
        task,
        datastore,
        ephemeral_datastore: _ephemeral_datastore,
        hpke_keypair,
        ..
    } = UploadTest::new_with_runtime(
        default_aggregator_config(),
        runtime_manager.with_label("aggregator"),
    )
    .await;

    let task = task.leader_view().unwrap();

    let mut report = create_report(&task, &hpke_keypair, clock.now());
    report = Report::new(
        report.metadata().clone(),
        report.public_share().to_vec(),
        hpke::seal(
            hpke_keypair.config(),
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Leader),
            // Some obviously wrong payload.
            &PlaintextInputShare::new(Vec::new(), vec![0; 100])
                .get_encoded()
                .unwrap(),
            &InputShareAad::new(
                *task.id(),
                report.metadata().clone(),
                report.public_share().to_vec(),
            )
            .get_encoded()
            .unwrap(),
        )
        .unwrap(),
        report.helper_encrypted_input_share().clone(),
    );

    // Try to upload the report, verify that we get the expected error.
    let error = aggregator
        .handle_upload(task.id(), &report.get_encoded().unwrap())
        .await
        .unwrap_err();
    assert_matches!(
        error.as_ref(),
        Error::ReportRejected(rejection) => {
            assert_eq!(task.id(), rejection.task_id());
            assert_eq!(report.metadata().id(), rejection.report_id());
            assert_eq!(report.metadata().time(), rejection.time());
            assert_matches!(rejection.reason(), ReportRejectionReason::DecodeFailure);
        }
    );

    // Wait for the report writer to have completed one write task.
    runtime_manager
        .wait_for_completed_tasks("aggregator", 1)
        .await;

    let got_counters = datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move { tx.get_task_upload_counter(&task_id).await })
        })
        .await
        .unwrap();
    assert_eq!(
        got_counters,
        Some(TaskUploadCounter::new_with_values(
            0, 1, 0, 0, 0, 0, 0, 0, 0
        ))
    )
}
