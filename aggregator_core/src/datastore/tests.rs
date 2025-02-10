#![allow(clippy::unit_arg)] // allow reference to dummy::Vdaf's public share, which has the unit type

use crate::{
    batch_mode::CollectableBatchMode,
    datastore::{
        models::{
            AcquiredAggregationJob, AcquiredCollectionJob, AggregateShareJob, AggregationJob,
            AggregationJobState, BatchAggregation, BatchAggregationState, CollectionJob,
            CollectionJobState, CollectionJobStateCode, HpkeKeyState, HpkeKeypair,
            LeaderStoredReport, Lease, OutstandingBatch, ReportAggregation,
            ReportAggregationMetadata, ReportAggregationMetadataState, ReportAggregationState,
            SqlInterval, TaskAggregationCounter, TaskUploadCounter,
        },
        schema_versions_template,
        test_util::{
            ephemeral_datastore_schema_version, generate_aead_key, EphemeralDatastore,
            EphemeralDatastoreBuilder, TEST_DATASTORE_MAX_TRANSACTION_RETRIES,
        },
        Crypter, Datastore, Error, RowExt, Transaction, SUPPORTED_SCHEMA_VERSIONS,
    },
    task::{self, test_util::TaskBuilder, AggregationMode, AggregatorTask},
    taskprov::test_util::PeerAggregatorBuilder,
    test_util::noop_meter,
};
use assert_matches::assert_matches;
use async_trait::async_trait;
use chrono::NaiveDate;
use futures::future::try_join_all;
use janus_core::{
    hpke::{self, HpkeApplicationInfo, Label},
    test_util::{install_test_trace_subscriber, run_vdaf},
    time::{Clock, DurationExt, IntervalExt, MockClock, TimeExt},
    vdaf::{vdaf_dp_strategies, VdafInstance, VERIFY_KEY_LENGTH_PRIO3},
};
use janus_messages::{
    batch_mode::{BatchMode, LeaderSelected, TimeInterval},
    AggregateShareAad, AggregationJobId, AggregationJobStep, BatchId, BatchSelector,
    CollectionJobId, Duration, Extension, ExtensionType, HpkeCiphertext, HpkeConfigId, Interval,
    PrepareInit, PrepareResp, PrepareStepResult, Query, ReportError, ReportId, ReportIdChecksum,
    ReportMetadata, ReportShare, Role, TaskId, Time,
};
use prio::{
    codec::{Decode, Encode},
    dp::{
        distributions::PureDpDiscreteLaplace, DifferentialPrivacyStrategy, PureDpBudget, Rational,
    },
    topology::ping_pong::PingPongMessage,
    vdaf::{dummy, prio3::Prio3Count},
};
use rand::{distributions::Standard, random, thread_rng, Rng};
use std::{
    collections::{HashMap, HashSet},
    iter,
    ops::RangeInclusive,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration as StdDuration,
};
use tokio::{time::timeout, try_join};
use url::Url;

// This function is only used when there are multiple supported versions.
#[allow(unused_imports)]
use crate::datastore::test_util::ephemeral_datastore_schema_version_by_downgrade;

const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);

#[test]
fn check_supported_versions() {
    if SUPPORTED_SCHEMA_VERSIONS[0] != *SUPPORTED_SCHEMA_VERSIONS.iter().max().unwrap() {
        panic!("the latest supported schema version must be first in the list");
    }
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn reject_unsupported_schema_version(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let error = Datastore::new_with_supported_versions(
        ephemeral_datastore.pool(),
        ephemeral_datastore.crypter(),
        MockClock::default(),
        &noop_meter(),
        &[0],
        TEST_DATASTORE_MAX_TRANSACTION_RETRIES,
    )
    .await
    .unwrap_err();

    assert_matches!(error, Error::DbState(_));
}

#[rstest::rstest]
#[case(ephemeral_datastore_schema_version(i64::MAX))]
#[tokio::test]
async fn down_migrations(
    #[future(awt)]
    #[case]
    ephemeral_datastore: EphemeralDatastore,
) {
    ephemeral_datastore.downgrade(0).await;
}

#[tokio::test]
async fn retry_limit() {
    install_test_trace_subscriber();
    let ephemeral_datastore = EphemeralDatastoreBuilder::new().build().await;

    for max_transaction_retries in [0, 1, 1000] {
        let datastore = ephemeral_datastore
            .datastore_with_max_transaction_retries(MockClock::default(), max_transaction_retries)
            .await;

        // The number of times the transaction was actually run.
        let num_runs = Arc::new(AtomicU64::new(0));

        // Induce infinite retry loop.
        let result = datastore
            .run_unnamed_tx(|tx| {
                let num_runs = Arc::clone(&num_runs);
                Box::pin(async move {
                    num_runs.fetch_add(1, Ordering::Relaxed);
                    tx.retry.store(true, Ordering::Relaxed); // artificially force a retry
                    Ok(())
                })
            })
            .await;

        assert_matches!(result, Err(Error::TooManyRetries { .. }));
        assert_eq!(
            num_runs.load(Ordering::Relaxed),
            max_transaction_retries + 1
        );
    }
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_task(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    // Insert tasks, check that they can be retrieved by ID.
    let mut want_tasks = HashMap::new();
    for (vdaf, role) in [
        (VdafInstance::Prio3Count, Role::Leader),
        (
            VdafInstance::Prio3SumVec {
                bits: 1,
                length: 8,
                chunk_length: 3,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
            Role::Leader,
        ),
        (
            VdafInstance::Prio3SumVec {
                bits: 1,
                length: 8,
                chunk_length: 3,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(
                    PureDpDiscreteLaplace::from_budget(
                        PureDpBudget::new(Rational::from_unsigned(1u128, 4u128).unwrap()).unwrap(),
                    ),
                ),
            },
            Role::Leader,
        ),
        (
            VdafInstance::Prio3SumVec {
                bits: 1,
                length: 64,
                chunk_length: 10,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
            Role::Helper,
        ),
        (
            VdafInstance::Prio3Sum {
                max_measurement: 4096,
            },
            Role::Helper,
        ),
        (
            VdafInstance::Prio3Sum {
                max_measurement: 4096,
            },
            Role::Helper,
        ),
        (
            VdafInstance::Prio3Histogram {
                length: 4,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
            },
            Role::Leader,
        ),
        (
            VdafInstance::Prio3Histogram {
                length: 5,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
            },
            Role::Leader,
        ),
    ] {
        let task = TaskBuilder::new(
            task::BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            vdaf,
        )
        .with_task_start(Some(Time::from_seconds_since_epoch(1000)))
        .with_task_end(Some(Time::from_seconds_since_epoch(4000)))
        .with_report_expiry_age(Some(Duration::from_seconds(3600)))
        .build()
        .view_for_role(role)
        .unwrap();
        want_tasks.insert(*task.id(), task.clone());

        let err = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.delete_task(task.id()).await })
            })
            .await
            .unwrap_err();
        assert_matches!(err, Error::MutationTargetNotFound);

        let retrieved_task = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.get_aggregator_task(task.id()).await })
            })
            .await
            .unwrap();
        assert_eq!(None, retrieved_task);

        ds.put_aggregator_task(&task).await.unwrap();

        let retrieved_task = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.get_aggregator_task(task.id()).await })
            })
            .await
            .unwrap();
        assert_eq!(Some(&task), retrieved_task.as_ref());

        ds.run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move { tx.delete_task(task.id()).await })
        })
        .await
        .unwrap();

        let retrieved_task = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.get_aggregator_task(task.id()).await })
            })
            .await
            .unwrap();
        assert_eq!(None, retrieved_task);

        let err = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.delete_task(task.id()).await })
            })
            .await
            .unwrap_err();
        assert_matches!(err, Error::MutationTargetNotFound);

        // Rewrite & retrieve the task again, to test that the delete is "clean" in the sense
        // that it deletes all task-related data (& therefore does not conflict with a later
        // write to the same task_id).
        ds.put_aggregator_task(&task).await.unwrap();

        let retrieved_task = ds
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.get_aggregator_task(task.id()).await })
            })
            .await
            .unwrap();
        assert_eq!(Some(task), retrieved_task);
    }

    let got_tasks: HashMap<TaskId, AggregatorTask> = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.check_timestamp_columns("tasks", "test-put-task", false)
                    .await;
                tx.get_aggregator_tasks().await
            })
        })
        .await
        .unwrap()
        .into_iter()
        .map(|task| (*task.id(), task))
        .collect();
    assert_eq!(want_tasks, got_tasks);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn update_task_end(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_task_end(Some(Time::from_seconds_since_epoch(1000)))
    .build()
    .leader_view()
    .unwrap();
    ds.put_aggregator_task(&task).await.unwrap();

    ds.run_unnamed_tx(|tx| {
        let task_id = *task.id();
        Box::pin(async move {
            let task = tx.get_aggregator_task(&task_id).await.unwrap().unwrap();
            assert_eq!(
                task.task_end().cloned(),
                Some(Time::from_seconds_since_epoch(1000))
            );

            tx.update_task_end(&task_id, Some(&Time::from_seconds_since_epoch(2000)))
                .await
                .unwrap();

            let task = tx.get_aggregator_task(&task_id).await.unwrap().unwrap();
            assert_eq!(
                task.task_end().cloned(),
                Some(Time::from_seconds_since_epoch(2000))
            );

            tx.update_task_end(&task_id, None).await.unwrap();

            let task = tx.get_aggregator_task(&task_id).await.unwrap().unwrap();
            assert_eq!(task.task_end().cloned(), None);

            let result = tx
                .update_task_end(&random(), Some(&Time::from_seconds_since_epoch(2000)))
                .await;
            assert_matches!(result, Err(Error::MutationTargetNotFound));

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn put_task_invalid_aggregator_auth_tokens(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .build()
    .leader_view()
    .unwrap();

    ds.put_aggregator_task(&task).await.unwrap();

    for (auth_token, auth_token_type) in [("NULL", "'BEARER'"), ("'\\xDEADBEEF'::bytea", "NULL")] {
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move {
                let err = tx
                    .query_one(
                        &format!(
                            "--
UPDATE tasks SET aggregator_auth_token = {auth_token},
aggregator_auth_token_type = {auth_token_type}"
                        ),
                        &[],
                    )
                    .await
                    .unwrap_err();

                assert_eq!(
                    err.as_db_error().unwrap().constraint().unwrap(),
                    "aggregator_auth_token_null"
                );
                Ok(())
            })
        })
        .await
        .unwrap();
    }
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn put_task_invalid_collector_auth_tokens(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .build()
    .leader_view()
    .unwrap();

    ds.put_aggregator_task(&task).await.unwrap();

    for (auth_token, auth_token_type) in [("NULL", "'BEARER'"), ("'\\xDEADBEEF'::bytea", "NULL")] {
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move {
                let err = tx
                    .query_one(
                        &format!(
                            "--
UPDATE tasks SET collector_auth_token_hash = {auth_token},
collector_auth_token_type = {auth_token_type}"
                        ),
                        &[],
                    )
                    .await
                    .unwrap_err();

                assert_eq!(
                    err.as_db_error().unwrap().constraint().unwrap(),
                    "collector_auth_token_null"
                );
                Ok(())
            })
        })
        .await
        .unwrap();
    }
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn get_task_ids(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            const TOTAL_TASK_ID_COUNT: usize = 20;
            let tasks: Vec<_> = iter::repeat_with(|| {
                TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap()
            })
            .take(TOTAL_TASK_ID_COUNT)
            .collect();

            let mut task_ids: Vec<_> = tasks.iter().map(AggregatorTask::id).cloned().collect();
            task_ids.sort();

            try_join_all(tasks.iter().map(|task| tx.put_aggregator_task(task)))
                .await
                .unwrap();

            for (i, lower_bound) in iter::once(None)
                .chain(task_ids.iter().cloned().map(Some))
                .enumerate()
            {
                let got_task_ids = tx.get_task_ids(lower_bound).await.unwrap();
                assert_eq!(&got_task_ids, &task_ids[i..]);
            }

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_report(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;
    let report_expiry_age = clock
        .now()
        .difference(&OLDEST_ALLOWED_REPORT_TIMESTAMP)
        .unwrap();

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_report_expiry_age(Some(report_expiry_age))
    .build()
    .leader_view()
    .unwrap();

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        Box::pin(async move { tx.put_aggregator_task(&task).await })
    })
    .await
    .unwrap();

    let report_id = random();
    let report: LeaderStoredReport<0, dummy::Vdaf> = LeaderStoredReport::new(
        *task.id(),
        ReportMetadata::new(
            report_id,
            OLDEST_ALLOWED_REPORT_TIMESTAMP,
            Vec::from([
                // public extensions
                Extension::new(ExtensionType::Tbd, "public_extension_tbd".into()),
                Extension::new(ExtensionType::Taskbind, "public_extension_taskbind".into()),
            ]),
        ),
        (), // public share
        Vec::from([
            // leader private extensions
            Extension::new(ExtensionType::Tbd, "leader_private_extension_tbd".into()),
            Extension::new(
                ExtensionType::Taskbind,
                "leader_private_extension_taskbind".into(),
            ),
        ]),
        dummy::InputShare::default(), // leader input share
        /* Dummy ciphertext for the helper share */
        HpkeCiphertext::new(
            HpkeConfigId::from(13),
            Vec::from("encapsulated_context_1"),
            Vec::from("payload_1"),
        ),
    );

    // Write the report, and then read it back.
    ds.run_tx("test-put-client-report", |tx| {
        let report = report.clone();
        Box::pin(async move { tx.put_client_report(&report).await })
    })
    .await
    .unwrap();

    let retrieved_report = ds
        .run_unnamed_tx(|tx| {
            let task_id = *report.task_id();
            Box::pin(async move {
                tx.get_client_report::<0, dummy::Vdaf>(
                    &dummy::Vdaf::default(),
                    &task_id,
                    &report_id,
                )
                .await
            })
        })
        .await
        .unwrap()
        .unwrap();

    assert_eq!(report, retrieved_report);

    // Try to write a different report with the same ID, and verify we get the expected error.
    let result = ds
        .run_unnamed_tx(|tx| {
            let task_id = *report.task_id();
            Box::pin(async move {
                tx.put_client_report(&LeaderStoredReport::<0, dummy::Vdaf>::new(
                    task_id,
                    ReportMetadata::new(
                        report_id,
                        Time::from_seconds_since_epoch(54321),
                        Vec::new(),
                    ),
                    (), // public share
                    Vec::new(),
                    dummy::InputShare::default(), // leader input share
                    /* Dummy ciphertext for the helper share */
                    HpkeCiphertext::new(
                        HpkeConfigId::from(14),
                        Vec::from("encapsulated_context_2"),
                        Vec::from("payload_2"),
                    ),
                ))
                .await
            })
        })
        .await;
    assert_matches!(result, Err(Error::MutationTargetAlreadyExists));

    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            tx.check_timestamp_columns("client_reports", "test-put-client-report", true)
                .await;
            Ok(())
        })
    })
    .await
    .unwrap();

    // Scrub the report, verify that the expected columns are NULL'ed out, and that we get the
    // expected error if we try to read the report at this point.
    ds.run_unnamed_tx(|tx| {
        let task_id = *report.task_id();

        Box::pin(async move {
            tx.scrub_client_report(&task_id, &report_id).await.unwrap();

            tx.verify_client_report_scrubbed(&task_id, &report_id).await;

            assert_matches!(
                tx.get_client_report::<0, dummy::Vdaf>(
                    &dummy::Vdaf::default(),
                    &task_id,
                    &report_id,
                )
                .await,
                Err(Error::Scrubbed)
            );

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock so that the report is expired, and verify that it does not exist.
    clock.advance(&Duration::from_seconds(1));
    let retrieved_report = ds
        .run_unnamed_tx(|tx| {
            let task_id = *report.task_id();
            Box::pin(async move {
                tx.get_client_report::<0, dummy::Vdaf>(
                    &dummy::Vdaf::default(),
                    &task_id,
                    &report_id,
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(None, retrieved_report);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn report_not_found(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let rslt = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.get_client_report(&dummy::Vdaf::default(), &random(), &random())
                    .await
            })
        })
        .await
        .unwrap();

    assert_eq!(rslt, None);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn get_unaggregated_client_reports_for_task(ephemeral_datastore: EphemeralDatastore) {
    use crate::datastore::models::UnaggregatedReport;

    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;
    let report_interval = Interval::new(
        OLDEST_ALLOWED_REPORT_TIMESTAMP
            .sub(&Duration::from_seconds(1))
            .unwrap(),
        Duration::from_seconds(2),
    )
    .unwrap();
    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .leader_view()
    .unwrap();
    let unrelated_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();

    let first_unaggregated_report =
        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let second_unaggregated_report =
        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let expired_report = LeaderStoredReport::new_dummy(
        *task.id(),
        OLDEST_ALLOWED_REPORT_TIMESTAMP
            .sub(&Duration::from_seconds(1))
            .unwrap(),
    );
    let aggregated_report =
        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let unrelated_report =
        LeaderStoredReport::new_dummy(*unrelated_task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);

    // Set up state.
    ds.run_tx("test-unaggregated-reports", |tx| {
        let task = task.clone();
        let unrelated_task = unrelated_task.clone();
        let first_unaggregated_report = first_unaggregated_report.clone();
        let second_unaggregated_report = second_unaggregated_report.clone();
        let expired_report = expired_report.clone();
        let aggregated_report = aggregated_report.clone();
        let unrelated_report = unrelated_report.clone();

        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();
            tx.put_aggregator_task(&unrelated_task).await.unwrap();

            tx.put_client_report(&first_unaggregated_report)
                .await
                .unwrap();
            tx.put_client_report(&second_unaggregated_report)
                .await
                .unwrap();
            tx.put_client_report(&expired_report).await.unwrap();
            tx.put_client_report(&aggregated_report).await.unwrap();
            tx.put_client_report(&unrelated_report).await.unwrap();

            // Mark aggregated_report as aggregated.
            tx.mark_report_aggregated(task.id(), aggregated_report.metadata().id())
                .await
                .unwrap();
            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    // Verify that we can acquire both unaggregated reports.
    let got_reports_ids: HashSet<_> = ds
        .run_tx("test-unaggregated-reports", |tx| {
            let task = task.clone();
            Box::pin(async move {
                // At this point, first_unaggregated_report and second_unaggregated_report are both
                // unaggregated.
                assert!(tx
                    .interval_has_unaggregated_reports(task.id(), &report_interval)
                    .await
                    .unwrap());

                Ok(tx
                    .get_unaggregated_client_reports_for_task(task.id(), 5000)
                    .await
                    .unwrap())
            })
        })
        .await
        .unwrap()
        .iter()
        .map(UnaggregatedReport::report_id)
        .copied()
        .collect();

    let want_report_ids = HashSet::from([
        *first_unaggregated_report.metadata().id(),
        *second_unaggregated_report.metadata().id(),
    ]);

    assert_eq!(got_reports_ids, want_report_ids);

    // Verify that attempting to acquire again does not return the reports.
    let got_report_ids: HashSet<_> = ds
        .run_tx("test-unaggregated-reports", |tx| {
            let task = task.clone();
            Box::pin(async move {
                // At this point, all reports have started aggregation.
                assert!(!tx
                    .interval_has_unaggregated_reports(task.id(), &report_interval)
                    .await
                    .unwrap());

                Ok(tx
                    .get_unaggregated_client_reports_for_task(task.id(), 5000)
                    .await
                    .unwrap())
            })
        })
        .await
        .unwrap()
        .iter()
        .map(UnaggregatedReport::report_id)
        .copied()
        .collect();

    assert!(got_report_ids.is_empty());

    // Mark one report un-aggregated.
    ds.run_tx("test-unaggregated-reports", |tx| {
        let (task, first_unaggregated_report) = (task.clone(), first_unaggregated_report.clone());
        Box::pin(async move {
            tx.mark_report_unaggregated(task.id(), first_unaggregated_report.metadata().id())
                .await
        })
    })
    .await
    .unwrap();

    // Verify that we can retrieve the un-aggregated report again.
    let got_report_ids: HashSet<_> = ds
        .run_tx("test-unaggregated-reports", |tx| {
            let task = task.clone();
            Box::pin(async move {
                // At this point, first_unaggregated_report is unaggregated.
                assert!(tx
                    .interval_has_unaggregated_reports(task.id(), &report_interval)
                    .await
                    .unwrap());

                Ok(tx
                    .get_unaggregated_client_reports_for_task(task.id(), 5000)
                    .await
                    .unwrap())
            })
        })
        .await
        .unwrap()
        .iter()
        .map(UnaggregatedReport::report_id)
        .copied()
        .collect();

    assert_eq!(
        got_report_ids,
        HashSet::from([*first_unaggregated_report.metadata().id()])
    );

    ds.run_unnamed_tx(|tx| {
        let (first_unaggregated_report, second_unaggregated_report) = (
            first_unaggregated_report.clone(),
            second_unaggregated_report.clone(),
        );
        Box::pin(async move {
            tx.check_timestamp_columns_at_create_time(
                "client_reports",
                "test-unaggregated-reports",
                OLDEST_ALLOWED_REPORT_TIMESTAMP,
                false,
            )
            .await;

            for row in tx
                .query("SELECT report_id, updated_at FROM client_reports", &[])
                .await
                .unwrap()
            {
                let report_id: ReportId =
                    row.get_bytea_and_convert::<ReportId>("report_id").unwrap();
                let updated_at: chrono::NaiveDateTime = row.get("updated_at");
                if report_id == *first_unaggregated_report.metadata().id()
                    || report_id == *second_unaggregated_report.metadata().id()
                {
                    assert_eq!(
                        tx.clock.now().as_naive_date_time().unwrap(),
                        updated_at,
                        "{report_id:?}"
                    );
                } else {
                    assert_eq!(
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .as_naive_date_time()
                            .unwrap(),
                        updated_at
                    );
                }
            }

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn get_unaggregated_client_report_ids_with_agg_param_for_task(
    ephemeral_datastore: EphemeralDatastore,
) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();
    let unrelated_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();

    let first_unaggregated_report =
        LeaderStoredReport::new_dummy(*task.id(), Time::from_seconds_since_epoch(12345));
    let second_unaggregated_report =
        LeaderStoredReport::new_dummy(*task.id(), Time::from_seconds_since_epoch(12346));
    let aggregated_report =
        LeaderStoredReport::new_dummy(*task.id(), Time::from_seconds_since_epoch(12347));
    let unrelated_report =
        LeaderStoredReport::new_dummy(*unrelated_task.id(), Time::from_seconds_since_epoch(12348));

    // Set up state.
    ds.run_unnamed_tx(|tx| {
        let (
            task,
            unrelated_task,
            first_unaggregated_report,
            second_unaggregated_report,
            aggregated_report,
            unrelated_report,
        ) = (
            task.clone(),
            unrelated_task.clone(),
            first_unaggregated_report.clone(),
            second_unaggregated_report.clone(),
            aggregated_report.clone(),
            unrelated_report.clone(),
        );

        Box::pin(async move {
            tx.put_aggregator_task(&task).await?;
            tx.put_aggregator_task(&unrelated_task).await?;

            tx.put_client_report(&first_unaggregated_report).await?;
            tx.put_client_report(&second_unaggregated_report).await?;
            tx.put_client_report(&aggregated_report).await?;
            tx.put_client_report(&unrelated_report).await?;

            // There are no client reports submitted under this task, so we shouldn't see
            // this aggregation parameter at all.
            tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *unrelated_task.id(),
                random(),
                Query::<TimeInterval>::new(
                    Interval::new(
                        Time::from_seconds_since_epoch(0),
                        Duration::from_hours(8).unwrap(),
                    )
                    .unwrap(),
                ),
                dummy::AggregationParam(255),
                Interval::new(
                    Time::from_seconds_since_epoch(0),
                    Duration::from_hours(8).unwrap(),
                )
                .unwrap(),
                CollectionJobState::<0, dummy::Vdaf>::Start,
            ))
            .await
        })
    })
    .await
    .unwrap();

    // Run query & verify results. None should be returned yet, as there are no relevant
    // collect requests.
    let got_reports = ds
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.get_unaggregated_client_report_ids_by_collect_for_task::<0, dummy::Vdaf>(
                    task.id(),
                    5000,
                )
                .await
            })
        })
        .await
        .unwrap();
    assert!(got_reports.is_empty());

    // Add collection jobs, and mark one report as having already been aggregated once.
    ds.run_unnamed_tx(|tx| {
        let (task, aggregated_report) = (task.clone(), aggregated_report.clone());
        Box::pin(async move {
            tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::<TimeInterval>::new(
                    Interval::new(
                        Time::from_seconds_since_epoch(0),
                        Duration::from_hours(8).unwrap(),
                    )
                    .unwrap(),
                ),
                dummy::AggregationParam(0),
                Interval::new(
                    Time::from_seconds_since_epoch(0),
                    Duration::from_hours(8).unwrap(),
                )
                .unwrap(),
                CollectionJobState::<0, dummy::Vdaf>::Start,
            ))
            .await?;
            tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::<TimeInterval>::new(
                    Interval::new(
                        Time::from_seconds_since_epoch(0),
                        Duration::from_hours(8).unwrap(),
                    )
                    .unwrap(),
                ),
                dummy::AggregationParam(1),
                Interval::new(
                    Time::from_seconds_since_epoch(0),
                    Duration::from_hours(8).unwrap(),
                )
                .unwrap(),
                CollectionJobState::<0, dummy::Vdaf>::Start,
            ))
            .await?;
            // No reports fall in this interval, so we shouldn't see it's aggregation
            // parameter at all.
            tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::<TimeInterval>::new(
                    Interval::new(
                        Time::from_seconds_since_epoch(8 * 3600),
                        Duration::from_hours(8).unwrap(),
                    )
                    .unwrap(),
                ),
                dummy::AggregationParam(2),
                Interval::new(
                    Time::from_seconds_since_epoch(8 * 3600),
                    Duration::from_hours(8).unwrap(),
                )
                .unwrap(),
                CollectionJobState::<0, dummy::Vdaf>::Start,
            ))
            .await?;

            let aggregation_job_id = random();
            tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                aggregation_job_id,
                dummy::AggregationParam(0),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Active,
                AggregationJobStep::from(0),
            ))
            .await?;
            tx.put_report_aggregation(
                &aggregated_report.as_leader_init_report_aggregation(aggregation_job_id, 0),
            )
            .await
        })
    })
    .await
    .unwrap();

    // Run query & verify results. We should have two unaggregated reports with one parameter,
    // and three with another.
    let mut got_reports: Vec<_> = ds
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.get_unaggregated_client_report_ids_by_collect_for_task::<0, dummy::Vdaf>(
                    task.id(),
                    5000,
                )
                .await
            })
        })
        .await
        .unwrap()
        .into_iter()
        .map(|(agg_param, report)| (agg_param, *report.report_id()))
        .collect();

    let mut want_reports = Vec::from([
        (
            dummy::AggregationParam(0),
            *first_unaggregated_report.metadata().id(),
        ),
        (
            dummy::AggregationParam(1),
            *first_unaggregated_report.metadata().id(),
        ),
        (
            dummy::AggregationParam(0),
            *second_unaggregated_report.metadata().id(),
        ),
        (
            dummy::AggregationParam(1),
            *second_unaggregated_report.metadata().id(),
        ),
        (
            dummy::AggregationParam(1),
            *aggregated_report.metadata().id(),
        ),
    ]);
    got_reports.sort();
    want_reports.sort();
    assert_eq!(got_reports, want_reports);

    // Add overlapping collection jobs with repeated aggregation parameters. Make sure we don't
    // repeat result tuples, which could lead to double counting in batch aggregations.
    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        Box::pin(async move {
            tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::<TimeInterval>::new(
                    Interval::new(
                        Time::from_seconds_since_epoch(0),
                        Duration::from_hours(16).unwrap(),
                    )
                    .unwrap(),
                ),
                dummy::AggregationParam(0),
                Interval::new(
                    Time::from_seconds_since_epoch(0),
                    Duration::from_hours(16).unwrap(),
                )
                .unwrap(),
                CollectionJobState::Start,
            ))
            .await?;
            tx.put_collection_job(&CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::<TimeInterval>::new(
                    Interval::new(
                        Time::from_seconds_since_epoch(0),
                        Duration::from_hours(16).unwrap(),
                    )
                    .unwrap(),
                ),
                dummy::AggregationParam(1),
                Interval::new(
                    Time::from_seconds_since_epoch(0),
                    Duration::from_hours(16).unwrap(),
                )
                .unwrap(),
                CollectionJobState::Start,
            ))
            .await?;
            Ok(())
        })
    })
    .await
    .unwrap();

    // Verify that we get the same result.
    let mut got_reports: Vec<_> = ds
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.get_unaggregated_client_report_ids_by_collect_for_task::<0, dummy::Vdaf>(
                    task.id(),
                    5000,
                )
                .await
            })
        })
        .await
        .unwrap()
        .into_iter()
        .map(|(agg_param, report)| (agg_param, *report.report_id()))
        .collect();
    got_reports.sort();
    assert_eq!(got_reports, want_reports);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn count_client_reports_for_interval(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .leader_view()
    .unwrap();
    let unrelated_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();
    let no_reports_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();

    let expired_report_in_interval = LeaderStoredReport::new_dummy(
        *task.id(),
        OLDEST_ALLOWED_REPORT_TIMESTAMP
            .sub(&Duration::from_seconds(1))
            .unwrap(),
    );
    let first_report_in_interval =
        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let second_report_in_interval = LeaderStoredReport::new_dummy(
        *task.id(),
        OLDEST_ALLOWED_REPORT_TIMESTAMP
            .add(&Duration::from_seconds(1))
            .unwrap(),
    );
    let report_outside_interval = LeaderStoredReport::new_dummy(
        *task.id(),
        OLDEST_ALLOWED_REPORT_TIMESTAMP
            .add(&Duration::from_seconds(10000))
            .unwrap(),
    );
    let report_for_other_task =
        LeaderStoredReport::new_dummy(*unrelated_task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);

    // Set up state.
    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        let unrelated_task = unrelated_task.clone();
        let no_reports_task = no_reports_task.clone();
        let expired_report_in_interval = expired_report_in_interval.clone();
        let first_report_in_interval = first_report_in_interval.clone();
        let second_report_in_interval = second_report_in_interval.clone();
        let report_outside_interval = report_outside_interval.clone();
        let report_for_other_task = report_for_other_task.clone();

        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();
            tx.put_aggregator_task(&unrelated_task).await.unwrap();
            tx.put_aggregator_task(&no_reports_task).await.unwrap();

            tx.put_client_report(&expired_report_in_interval)
                .await
                .unwrap();
            tx.put_client_report(&first_report_in_interval)
                .await
                .unwrap();
            tx.put_client_report(&second_report_in_interval)
                .await
                .unwrap();
            tx.put_client_report(&report_outside_interval)
                .await
                .unwrap();
            tx.put_client_report(&report_for_other_task).await.unwrap();

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    let (report_count, no_reports_task_report_count) = ds
        .run_unnamed_tx(|tx| {
            let (task, no_reports_task) = (task.clone(), no_reports_task.clone());
            Box::pin(async move {
                let report_count = tx
                    .count_client_reports_for_interval(
                        task.id(),
                        &Interval::new(
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(1))
                                .unwrap(),
                            Duration::from_seconds(5),
                        )
                        .unwrap(),
                    )
                    .await
                    .unwrap();

                let no_reports_task_report_count = tx
                    .count_client_reports_for_interval(
                        no_reports_task.id(),
                        &Interval::new(
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(1))
                                .unwrap(),
                            Duration::from_seconds(5),
                        )
                        .unwrap(),
                    )
                    .await
                    .unwrap();

                Ok((report_count, no_reports_task_report_count))
            })
        })
        .await
        .unwrap();
    assert_eq!(report_count, 2);
    assert_eq!(no_reports_task_report_count, 0);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn count_client_reports_for_batch_id(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

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
    let unrelated_task = TaskBuilder::new(
        task::BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();

    // Set up state.
    let batch_id = ds
        .run_unnamed_tx(|tx| {
            let (task, unrelated_task) = (task.clone(), unrelated_task.clone());

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_aggregator_task(&unrelated_task).await.unwrap();

                // Create a batch for the first task containing two reports, which has started
                // aggregation twice with two different aggregation parameters.
                let batch_id = random();
                let expired_report = LeaderStoredReport::new_dummy(
                    *task.id(),
                    OLDEST_ALLOWED_REPORT_TIMESTAMP
                        .sub(&Duration::from_seconds(2))
                        .unwrap(),
                );
                let report_0 =
                    LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
                let report_1 = LeaderStoredReport::new_dummy(
                    *task.id(),
                    OLDEST_ALLOWED_REPORT_TIMESTAMP
                        .add(&Duration::from_seconds(1))
                        .unwrap(),
                );

                let expired_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    random(),
                    dummy::AggregationParam(22),
                    batch_id,
                    Interval::new(
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(2))
                            .unwrap(),
                        Duration::from_seconds(1),
                    )
                    .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                );
                let expired_report_aggregation = expired_report
                    .as_leader_init_report_aggregation(*expired_aggregation_job.id(), 0);

                let aggregation_job_0 = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    random(),
                    dummy::AggregationParam(22),
                    batch_id,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(2))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                );
                let aggregation_job_0_report_aggregation_0 =
                    report_0.as_leader_init_report_aggregation(*aggregation_job_0.id(), 1);
                let aggregation_job_0_report_aggregation_1 =
                    report_1.as_leader_init_report_aggregation(*aggregation_job_0.id(), 2);

                let aggregation_job_1 = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    random(),
                    dummy::AggregationParam(23),
                    batch_id,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                );
                let aggregation_job_1_report_aggregation_0 =
                    report_0.as_leader_init_report_aggregation(*aggregation_job_1.id(), 0);
                let aggregation_job_1_report_aggregation_1 =
                    report_1.as_leader_init_report_aggregation(*aggregation_job_1.id(), 1);

                tx.put_client_report(&expired_report).await.unwrap();
                tx.put_client_report(&report_0).await.unwrap();
                tx.put_client_report(&report_1).await.unwrap();

                tx.put_aggregation_job(&expired_aggregation_job)
                    .await
                    .unwrap();
                tx.put_report_aggregation(&expired_report_aggregation)
                    .await
                    .unwrap();

                tx.put_aggregation_job(&aggregation_job_0).await.unwrap();
                tx.put_report_aggregation(&aggregation_job_0_report_aggregation_0)
                    .await
                    .unwrap();
                tx.put_report_aggregation(&aggregation_job_0_report_aggregation_1)
                    .await
                    .unwrap();

                tx.put_aggregation_job(&aggregation_job_1).await.unwrap();
                tx.put_report_aggregation(&aggregation_job_1_report_aggregation_0)
                    .await
                    .unwrap();
                tx.put_report_aggregation(&aggregation_job_1_report_aggregation_1)
                    .await
                    .unwrap();

                Ok(batch_id)
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    let report_count = ds
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move {
                tx.count_client_reports_for_batch_id(&task_id, &batch_id)
                    .await
            })
        })
        .await
        .unwrap();
    assert_eq!(report_count, 2);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_scrubbed_report(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .build()
    .leader_view()
    .unwrap();

    let report_id = ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    let client_timestamp = Time::from_seconds_since_epoch(12345);

    ds.run_tx("test-put-report-share", |tx| {
        let task = task.clone();

        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();
            tx.put_scrubbed_report(task.id(), &report_id, &client_timestamp)
                .await
                .unwrap();

            tx.check_timestamp_columns("client_reports", "test-put-report-share", true)
                .await;

            Ok(())
        })
    })
    .await
    .unwrap();

    let (
        got_task_id,
        got_public_extensions,
        got_public_share,
        got_leader_private_extensions,
        got_leader_input_share,
        got_helper_input_share,
    ) = ds
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();

            Box::pin(async move {
                // Verify that attempting to read the report share as a report receives the expected
                // error, and that the expected columns are NULL'ed out.
                assert_matches!(
                    tx.get_client_report::<0, dummy::Vdaf>(
                        &dummy::Vdaf::default(),
                        &task_id,
                        &report_id,
                    )
                    .await,
                    Err(Error::Scrubbed)
                );

                let row = tx
                    .query_one(
                        "--
SELECT
    tasks.task_id,
    client_reports.report_id,
    client_reports.client_timestamp,
    client_reports.public_extensions,
    client_reports.public_share,
    client_reports.leader_private_extensions,
    client_reports.leader_input_share,
    client_reports.helper_encrypted_input_share
FROM client_reports JOIN tasks ON tasks.id = client_reports.task_id
WHERE tasks.task_id = $1 AND client_reports.report_id = $2",
                        &[
                            /* task_id */ &task_id.as_ref(),
                            /* report_id */ &report_id.as_ref(),
                        ],
                    )
                    .await
                    .unwrap();

                let task_id = TaskId::get_decoded(row.get("task_id")).unwrap();

                let maybe_public_extensions: Option<Vec<u8>> = row.get("public_extensions");
                let maybe_public_share: Option<Vec<u8>> = row.get("public_share");
                let maybe_leader_private_extensions: Option<Vec<u8>> =
                    row.get("leader_private_extensions");
                let maybe_leader_input_share: Option<Vec<u8>> = row.get("leader_input_share");
                let maybe_helper_input_share: Option<Vec<u8>> =
                    row.get("helper_encrypted_input_share");

                Ok((
                    task_id,
                    maybe_public_extensions,
                    maybe_public_share,
                    maybe_leader_private_extensions,
                    maybe_leader_input_share,
                    maybe_helper_input_share,
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(task.id(), &got_task_id);
    assert!(got_public_extensions.is_none());
    assert!(got_public_share.is_none());
    assert!(got_leader_private_extensions.is_none());
    assert!(got_leader_input_share.is_none());
    assert!(got_helper_input_share.is_none());
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_aggregation_job(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    // We use a dummy VDAF & leader-selected task for this test, to better exercise the
    // serialization/deserialization roundtrip of the batch_identifier & aggregation_param.
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
    let batch_id = random();
    let leader_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        random(),
        dummy::AggregationParam(23),
        batch_id,
        Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );
    let helper_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        random(),
        dummy::AggregationParam(23),
        random(),
        Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );

    ds.run_tx("test-put-aggregation-jobs", |tx| {
        let (task, leader_aggregation_job, helper_aggregation_job) = (
            task.clone(),
            leader_aggregation_job.clone(),
            helper_aggregation_job.clone(),
        );
        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();
            tx.put_aggregation_job(&leader_aggregation_job)
                .await
                .unwrap();
            tx.put_aggregation_job(&helper_aggregation_job)
                .await
                .unwrap();

            tx.check_timestamp_columns("aggregation_jobs", "test-put-aggregation-jobs", true)
                .await;

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    let (got_leader_aggregation_job, got_helper_aggregation_job) = ds
        .run_unnamed_tx(|tx| {
            let (leader_aggregation_job, helper_aggregation_job) = (
                leader_aggregation_job.clone(),
                helper_aggregation_job.clone(),
            );
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_job(
                        leader_aggregation_job.task_id(),
                        leader_aggregation_job.id(),
                    )
                    .await
                    .unwrap(),
                    tx.get_aggregation_job(
                        helper_aggregation_job.task_id(),
                        helper_aggregation_job.id(),
                    )
                    .await
                    .unwrap(),
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(
        Some(&leader_aggregation_job),
        got_leader_aggregation_job.as_ref()
    );
    assert_eq!(
        Some(&helper_aggregation_job),
        got_helper_aggregation_job.as_ref()
    );

    let new_leader_aggregation_job = leader_aggregation_job
        .clone()
        .with_state(AggregationJobState::Finished);
    let new_helper_aggregation_job = helper_aggregation_job.with_last_request_hash([3; 32]);
    ds.run_tx("test-update-aggregation-jobs", |tx| {
        let (new_leader_aggregation_job, new_helper_aggregation_job) = (
            new_leader_aggregation_job.clone(),
            new_helper_aggregation_job.clone(),
        );
        Box::pin(async move {
            tx.update_aggregation_job(&new_leader_aggregation_job)
                .await
                .unwrap();
            tx.update_aggregation_job(&new_helper_aggregation_job)
                .await
                .unwrap();

            tx.check_timestamp_columns_at_create_time(
                "aggregation_jobs",
                "test-update-aggregation-jobs",
                OLDEST_ALLOWED_REPORT_TIMESTAMP,
                true,
            )
            .await;

            Ok(())
        })
    })
    .await
    .unwrap();

    let (got_leader_aggregation_job, got_helper_aggregation_job) = ds
        .run_unnamed_tx(|tx| {
            let (new_leader_aggregation_job, new_helper_aggregation_job) = (
                new_leader_aggregation_job.clone(),
                new_helper_aggregation_job.clone(),
            );
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_job(
                        new_leader_aggregation_job.task_id(),
                        new_leader_aggregation_job.id(),
                    )
                    .await
                    .unwrap(),
                    tx.get_aggregation_job(
                        new_helper_aggregation_job.task_id(),
                        new_helper_aggregation_job.id(),
                    )
                    .await
                    .unwrap(),
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(
        Some(new_leader_aggregation_job.clone()),
        got_leader_aggregation_job
    );
    assert_eq!(
        Some(new_helper_aggregation_job.clone()),
        got_helper_aggregation_job
    );

    // Trying to write an aggregation job again should fail.
    let new_leader_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        *leader_aggregation_job.id(),
        dummy::AggregationParam(24),
        batch_id,
        Interval::new(
            Time::from_seconds_since_epoch(2345),
            Duration::from_seconds(6789),
        )
        .unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );
    ds.run_unnamed_tx(|tx| {
        let new_leader_aggregation_job = new_leader_aggregation_job.clone();
        Box::pin(async move {
            let error = tx
                .put_aggregation_job(&new_leader_aggregation_job)
                .await
                .unwrap_err();
            assert_matches!(error, Error::MutationTargetAlreadyExists);

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to verify that the aggregation jobs have expired & are no longer
    // returned.
    clock.advance(&Duration::from_seconds(2));
    let (got_leader_aggregation_job, got_helper_aggregation_job) = ds
        .run_unnamed_tx(|tx| {
            let (new_leader_aggregation_job, new_helper_aggregation_job) = (
                new_leader_aggregation_job.clone(),
                new_helper_aggregation_job.clone(),
            );
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_job::<0, LeaderSelected, dummy::Vdaf>(
                        new_leader_aggregation_job.task_id(),
                        new_leader_aggregation_job.id(),
                    )
                    .await
                    .unwrap(),
                    tx.get_aggregation_job::<0, LeaderSelected, dummy::Vdaf>(
                        new_helper_aggregation_job.task_id(),
                        new_helper_aggregation_job.id(),
                    )
                    .await
                    .unwrap(),
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(None, got_leader_aggregation_job);
    assert_eq!(None, got_helper_aggregation_job);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn aggregation_job_acquire_release(ephemeral_datastore: EphemeralDatastore) {
    // Setup: insert a few aggregation jobs.
    install_test_trace_subscriber();

    const LEASE_DURATION: StdDuration = StdDuration::from_secs(300);
    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

    const AGGREGATION_JOB_COUNT: usize = 10;
    let leader_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .leader_view()
    .unwrap();
    let helper_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Prio3Count,
    )
    .build()
    .helper_view()
    .unwrap();

    let mut task_and_aggregation_job_ids: Vec<_> = [*leader_task.id(), *helper_task.id()]
        .into_iter()
        .cycle()
        .zip(thread_rng().sample_iter(Standard))
        .take(AGGREGATION_JOB_COUNT)
        .collect();
    task_and_aggregation_job_ids.sort();

    ds.run_unnamed_tx(|tx| {
        let leader_task = leader_task.clone();
        let helper_task = helper_task.clone();
        let task_and_aggregation_job_ids = task_and_aggregation_job_ids.clone();

        Box::pin(async move {
            // Write a few aggregation jobs we expect to be able to retrieve with
            // acquire_incomplete_aggregation_jobs().
            tx.put_aggregator_task(&leader_task).await.unwrap();
            tx.put_aggregator_task(&helper_task).await.unwrap();

            try_join_all(task_and_aggregation_job_ids.into_iter().map(
                |(task_id, aggregation_job_id)| async move {
                    tx.put_aggregation_job(&AggregationJob::<
                        VERIFY_KEY_LENGTH_PRIO3,
                        TimeInterval,
                        Prio3Count,
                    >::new(
                        task_id,
                        aggregation_job_id,
                        (),
                        (),
                        Interval::new(
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(LEASE_DURATION.as_secs()))
                                .unwrap()
                                .add(&Duration::from_seconds(LEASE_DURATION.as_secs()))
                                .unwrap(),
                            Duration::from_seconds(1),
                        )
                        .unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    ))
                    .await
                },
            ))
            .await
            .unwrap();

            // Write an aggregation job that is finished. We don't want to retrieve this one.
            tx.put_aggregation_job(&AggregationJob::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *leader_task.id(),
                random(),
                (),
                (),
                Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1)).unwrap(),
                AggregationJobState::Finished,
                AggregationJobStep::from(1),
            ))
            .await
            .unwrap();

            // Write an expired aggregation job. We don't want to retrieve this one, either.
            tx.put_aggregation_job(&AggregationJob::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *leader_task.id(),
                random(),
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Active,
                AggregationJobStep::from(0),
            ))
            .await
            .unwrap();

            // Write an aggregation job that is awaiting a request from the Leader. We don't want to
            // retrieve this one, either.
            tx.put_aggregation_job(&AggregationJob::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *helper_task.id(),
                random(),
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::AwaitingRequest,
                AggregationJobStep::from(0),
            ))
            .await
            .unwrap();

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    // Run: run several transactions that all call acquire_incomplete_aggregation_jobs
    // concurrently. (We do things concurrently in an attempt to make sure the
    // mutual-exclusivity works properly.)
    const CONCURRENT_TX_COUNT: usize = 10;
    const MAXIMUM_ACQUIRE_COUNT: usize = 4;

    // Sanity check constants: ensure we acquire jobs across multiple calls to exercise the
    // maximum-jobs-per-call functionality. Make sure we're attempting to acquire enough jobs
    // in total to cover the number of acquirable jobs we created.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(MAXIMUM_ACQUIRE_COUNT < AGGREGATION_JOB_COUNT);
        assert!(
            MAXIMUM_ACQUIRE_COUNT
                .checked_mul(CONCURRENT_TX_COUNT)
                .unwrap()
                >= AGGREGATION_JOB_COUNT
        );
    }

    let want_expiry_time = clock.now().as_naive_date_time().unwrap()
        + chrono::Duration::from_std(LEASE_DURATION).unwrap();
    let want_aggregation_jobs: Vec<_> = task_and_aggregation_job_ids
        .iter()
        .map(|(task_id, aggregation_job_id)| {
            (
                AcquiredAggregationJob::new(
                    *task_id,
                    *aggregation_job_id,
                    task::BatchMode::TimeInterval,
                    VdafInstance::Prio3Count,
                ),
                want_expiry_time,
            )
        })
        .collect();

    let got_leases = timeout(StdDuration::from_secs(10), {
        let ds = Arc::clone(&ds);
        let want_lease_count = want_aggregation_jobs.len();
        async move {
            let mut got_leases = Vec::new();
            loop {
                // Rarely, due to Postgres locking semantics, an aggregation job that could be
                // returned will instead be skipped by all concurrent aggregation attempts. Retry
                // for a little while to keep this from affecting test outcome.
                let results = try_join_all(
                    iter::repeat_with(|| {
                        ds.run_unnamed_tx(|tx| {
                            Box::pin(async move {
                                tx.acquire_incomplete_aggregation_jobs(
                                    &LEASE_DURATION,
                                    MAXIMUM_ACQUIRE_COUNT,
                                )
                                .await
                            })
                        })
                    })
                    .take(CONCURRENT_TX_COUNT),
                )
                .await
                .unwrap();

                for result in results {
                    assert!(result.len() <= MAXIMUM_ACQUIRE_COUNT);
                    got_leases.extend(result.into_iter());
                }

                if got_leases.len() >= want_lease_count {
                    break got_leases;
                }
            }
        }
    })
    .await
    .unwrap();

    // Verify: check that we got all of the desired aggregation jobs, with no duplication, and
    // the expected lease expiry.
    let mut got_aggregation_jobs: Vec<_> = got_leases
        .iter()
        .map(|lease| {
            assert_eq!(lease.lease_attempts(), 1);
            (lease.leased().clone(), *lease.lease_expiry_time())
        })
        .collect();
    got_aggregation_jobs.sort();

    assert_eq!(want_aggregation_jobs, got_aggregation_jobs);

    // Run: release a few jobs with a delay before reacquiry, then attempt to acquire jobs again.
    const RELEASE_COUNT: usize = 2;
    const REACQUIRE_DELAY: StdDuration = StdDuration::from_secs(10);

    // Sanity check constants: ensure we release fewer jobs than we're about to acquire to ensure we
    // can acquire them in all in a single call, while leaving headroom to acquire at least one
    // unwanted job if there is a logic bug. And ensure that our reacquire delay is shorter than the
    // lease duration, to ensure we don't timeout the leases which are not explicitly released.
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(RELEASE_COUNT < MAXIMUM_ACQUIRE_COUNT);
        assert!(REACQUIRE_DELAY < LEASE_DURATION);
    }

    let leases_to_release: Vec<_> = got_leases.into_iter().take(RELEASE_COUNT).collect();
    let mut jobs_to_release: Vec<_> = leases_to_release
        .iter()
        .map(|lease| (lease.leased().clone(), *lease.lease_expiry_time()))
        .collect();
    jobs_to_release.sort();
    ds.run_unnamed_tx(|tx| {
        let leases_to_release = leases_to_release.clone();
        Box::pin(async move {
            for lease in leases_to_release {
                tx.release_aggregation_job(&lease, Some(&REACQUIRE_DELAY))
                    .await
                    .unwrap();
            }
            Ok(())
        })
    })
    .await
    .unwrap();

    // Verify that we can't immediately acquire the jobs again.
    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            assert!(tx
                .acquire_incomplete_aggregation_jobs(&LEASE_DURATION, MAXIMUM_ACQUIRE_COUNT)
                .await
                .unwrap()
                .is_empty());
            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock past the reacquire delay, then reacquire the leases we released with a
    // reacquire delay.
    clock.advance(&Duration::from_seconds(REACQUIRE_DELAY.as_secs()));

    let mut got_aggregation_jobs: Vec<_> = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.acquire_incomplete_aggregation_jobs(
                    &(LEASE_DURATION - REACQUIRE_DELAY),
                    MAXIMUM_ACQUIRE_COUNT,
                )
                .await
            })
        })
        .await
        .unwrap()
        .into_iter()
        .map(|lease| {
            assert_eq!(lease.lease_attempts(), 1);
            (lease.leased().clone(), *lease.lease_expiry_time())
        })
        .collect();
    got_aggregation_jobs.sort();

    // Verify: we should have re-acquired the jobs we released.
    assert_eq!(jobs_to_release, got_aggregation_jobs);

    // Run: advance time by the lease duration (which implicitly releases the jobs), and attempt
    // to acquire aggregation jobs again.
    clock.advance(&Duration::from_seconds(
        LEASE_DURATION.as_secs() - REACQUIRE_DELAY.as_secs(),
    ));
    let want_expiry_time = clock.now().as_naive_date_time().unwrap()
        + chrono::Duration::from_std(LEASE_DURATION).unwrap();
    let want_aggregation_jobs: Vec<_> = task_and_aggregation_job_ids
        .iter()
        .map(|(task_id, aggregation_job_id)| {
            (
                AcquiredAggregationJob::new(
                    *task_id,
                    *aggregation_job_id,
                    task::BatchMode::TimeInterval,
                    VdafInstance::Prio3Count,
                ),
                want_expiry_time,
            )
        })
        .collect();
    let mut got_aggregation_jobs: Vec<_> = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                // This time, we just acquire all jobs in a single go for simplicity -- we've
                // already tested the maximum acquire count functionality above.
                tx.acquire_incomplete_aggregation_jobs(&LEASE_DURATION, AGGREGATION_JOB_COUNT)
                    .await
            })
        })
        .await
        .unwrap()
        .into_iter()
        .map(|lease| {
            let job = (lease.leased().clone(), *lease.lease_expiry_time());
            let expected_attempts = if jobs_to_release.contains(&job) { 1 } else { 2 };
            assert_eq!(lease.lease_attempts(), expected_attempts);
            job
        })
        .collect();
    got_aggregation_jobs.sort();

    // Verify: we got all the jobs.
    assert_eq!(want_aggregation_jobs, got_aggregation_jobs);

    // Run: advance time again to release jobs, acquire a single job, modify its lease token
    // to simulate a previously-held lease, and attempt to release it. Verify that releasing
    // fails.
    clock.advance(&Duration::from_seconds(LEASE_DURATION.as_secs()));
    let lease = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                Ok(tx
                    .acquire_incomplete_aggregation_jobs(&LEASE_DURATION, 1)
                    .await
                    .unwrap()
                    .remove(0))
            })
        })
        .await
        .unwrap();
    let lease_with_random_token = Lease::new(
        lease.leased().clone(),
        *lease.lease_expiry_time(),
        random(),
        lease.lease_attempts(),
    );
    ds.run_unnamed_tx(|tx| {
        let lease_with_random_token = lease_with_random_token.clone();
        Box::pin(async move {
            tx.release_aggregation_job(&lease_with_random_token, None)
                .await
        })
    })
    .await
    .unwrap_err();

    // Replace the original lease token and verify that we can release successfully with it in
    // place.
    ds.run_unnamed_tx(|tx| {
        let lease = lease.clone();
        Box::pin(async move { tx.release_aggregation_job(&lease, None).await })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn aggregation_job_not_found(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let rslt = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.get_aggregation_job::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(
                    &random(),
                    &random(),
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(rslt, None);

    let rslt = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.update_aggregation_job::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(
                    &AggregationJob::new(
                        random(),
                        random(),
                        (),
                        (),
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    ),
                )
                .await
            })
        })
        .await;
    assert_matches!(rslt, Err(Error::MutationTargetNotFound));
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn get_aggregation_jobs_for_task(ephemeral_datastore: EphemeralDatastore) {
    // Setup.
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    // We use a dummy VDAF & leader-selected task for this test, to better exercise the
    // serialization/deserialization roundtrip of the batch_identifier & aggregation_param.
    let task = TaskBuilder::new(
        task::BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();
    let first_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        random(),
        dummy::AggregationParam(23),
        random(),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );
    let second_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        random(),
        dummy::AggregationParam(42),
        random(),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );
    let aggregation_job_with_request_hash = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        random(),
        dummy::AggregationParam(42),
        random(),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    )
    .with_last_request_hash([3; 32]);

    let mut want_agg_jobs = Vec::from([
        first_aggregation_job,
        second_aggregation_job,
        aggregation_job_with_request_hash,
    ]);

    ds.run_unnamed_tx(|tx| {
        let (task, want_agg_jobs) = (task.clone(), want_agg_jobs.clone());
        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();

            for agg_job in want_agg_jobs {
                tx.put_aggregation_job(&agg_job).await.unwrap();
            }

            // Also write an unrelated aggregation job with a different task ID to check that it
            // is not returned.
            let unrelated_task = TaskBuilder::new(
                task::BatchMode::LeaderSelected {
                    batch_time_window_size: None,
                },
                AggregationMode::Synchronous,
                VdafInstance::Fake { rounds: 1 },
            )
            .build()
            .leader_view()
            .unwrap();
            tx.put_aggregator_task(&unrelated_task).await.unwrap();
            tx.put_aggregation_job(&AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                *unrelated_task.id(),
                random(),
                dummy::AggregationParam(82),
                random(),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Active,
                AggregationJobStep::from(0),
            ))
            .await
        })
    })
    .await
    .unwrap();

    // Run.
    want_agg_jobs.sort_by_key(|agg_job| *agg_job.id());
    let mut got_agg_jobs = ds
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move { tx.get_aggregation_jobs_for_task(task.id()).await })
        })
        .await
        .unwrap();
    got_agg_jobs.sort_by_key(|agg_job| *agg_job.id());

    // Verify.
    assert_eq!(want_agg_jobs, got_agg_jobs);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_report_aggregation(ephemeral_datastore: EphemeralDatastore) {
    use janus_messages::PrepareContinue;

    install_test_trace_subscriber();

    let task_id = random();
    let report_id = random();
    let vdaf = Arc::new(dummy::Vdaf::new(2));
    let aggregation_param = dummy::AggregationParam(5);
    let vdaf_transcript = run_vdaf(
        vdaf.as_ref(),
        &task_id,
        &[],
        &aggregation_param,
        &report_id,
        &13,
    );

    for (ord, (role, state)) in [
        (
            Role::Leader,
            ReportAggregationState::LeaderInit {
                public_extensions: Vec::from([Extension::new(
                    ExtensionType::Tbd,
                    "public_extension_tbd".into(),
                )]),
                public_share: vdaf_transcript.public_share,
                leader_private_extensions: Vec::from([Extension::new(
                    ExtensionType::Taskbind,
                    "leader_private_extension_taskbind".into(),
                )]),
                leader_input_share: vdaf_transcript.leader_input_share,
                helper_encrypted_input_share: HpkeCiphertext::new(
                    HpkeConfigId::from(13),
                    Vec::from("encapsulated_context"),
                    Vec::from("payload"),
                ),
            },
        ),
        (
            Role::Leader,
            ReportAggregationState::LeaderContinue {
                transition: vdaf_transcript.leader_prepare_transitions[1]
                    .transition
                    .clone()
                    .unwrap(),
            },
        ),
        (
            Role::Leader,
            ReportAggregationState::LeaderPoll {
                leader_state: vdaf_transcript.leader_prepare_transitions[0].state.clone(),
            },
        ),
        (
            Role::Leader,
            ReportAggregationState::LeaderPoll {
                leader_state: vdaf_transcript.leader_prepare_transitions[1].state.clone(),
            },
        ),
        (
            Role::Helper,
            ReportAggregationState::HelperInitProcessing {
                prepare_init: PrepareInit::new(
                    ReportShare::new(
                        ReportMetadata::new(
                            report_id,
                            Time::from_seconds_since_epoch(25000),
                            Vec::new(),
                        ),
                        vdaf_transcript.public_share.get_encoded().unwrap(),
                        HpkeCiphertext::new(
                            HpkeConfigId::from(13),
                            Vec::from("encapsulated_context"),
                            Vec::from("payload"),
                        ),
                    ),
                    vdaf_transcript.leader_prepare_transitions[0]
                        .message
                        .clone(),
                ),
                require_taskbind_extension: true,
            },
        ),
        (
            Role::Helper,
            ReportAggregationState::HelperContinue {
                prepare_state: *vdaf_transcript.helper_prepare_transitions[0].prepare_state(),
            },
        ),
        (
            Role::Helper,
            ReportAggregationState::HelperContinueProcessing {
                prepare_state: *vdaf_transcript.helper_prepare_transitions[0].prepare_state(),
                prepare_continue: PrepareContinue::new(
                    report_id,
                    vdaf_transcript.leader_prepare_transitions[1]
                        .message
                        .clone(),
                ),
            },
        ),
        (Role::Leader, ReportAggregationState::Finished),
        (Role::Helper, ReportAggregationState::Finished),
        (
            Role::Leader,
            ReportAggregationState::Failed {
                report_error: ReportError::VdafPrepError,
            },
        ),
        (
            Role::Helper,
            ReportAggregationState::Failed {
                report_error: ReportError::VdafPrepError,
            },
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task = TaskBuilder::new(
            task::BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Fake { rounds: 2 },
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build()
        .view_for_role(role)
        .unwrap();
        let aggregation_job_id = random();
        let report_id = random();

        let want_report_aggregation = ds
            .run_tx("test-put-report-aggregations", |tx| {
                let (task, state, aggregation_param) =
                    (task.clone(), state.clone(), aggregation_param);
                Box::pin(async move {
                    tx.put_aggregator_task(&task).await.unwrap();
                    tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param,
                        (),
                        Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::Active,
                        AggregationJobStep::from(0),
                    ))
                    .await
                    .unwrap();
                    tx.put_scrubbed_report(task.id(), &report_id, &OLDEST_ALLOWED_REPORT_TIMESTAMP)
                        .await
                        .unwrap();

                    let report_aggregation = ReportAggregation::new(
                        *task.id(),
                        aggregation_job_id,
                        report_id,
                        OLDEST_ALLOWED_REPORT_TIMESTAMP,
                        ord.try_into().unwrap(),
                        Some(PrepareResp::new(
                            report_id,
                            PrepareStepResult::Continue {
                                message: PingPongMessage::Continue {
                                    prep_msg: format!("prep_msg_{ord}").into(),
                                    prep_share: format!("prep_share_{ord}").into(),
                                },
                            },
                        )),
                        state,
                    );
                    tx.put_report_aggregation(&report_aggregation)
                        .await
                        .unwrap();

                    let row = tx
                        .query_one(
                            "--
SELECT updated_at, updated_by FROM report_aggregations
WHERE client_report_id = $1",
                            &[&report_aggregation.report_id().as_ref()],
                        )
                        .await
                        .unwrap();
                    let updated_at: chrono::NaiveDateTime = row.get("updated_at");
                    let updated_by: &str = row.get("updated_by");

                    assert_eq!(updated_at, tx.clock.now().as_naive_date_time().unwrap());
                    assert_eq!(updated_by, "test-put-report-aggregations");

                    Ok(report_aggregation)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        let got_report_aggregation = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = task.clone();

                Box::pin(async move {
                    tx.get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &role,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                })
            })
            .await
            .unwrap()
            .unwrap();

        assert_eq!(want_report_aggregation, got_report_aggregation);

        let want_report_aggregation = ReportAggregation::new(
            *want_report_aggregation.task_id(),
            *want_report_aggregation.aggregation_job_id(),
            *want_report_aggregation.report_id(),
            *want_report_aggregation.time(),
            want_report_aggregation.ord(),
            Some(PrepareResp::new(
                report_id,
                PrepareStepResult::Continue {
                    message: PingPongMessage::Continue {
                        prep_msg: format!("updated_prep_msg_{ord}").into(),
                        prep_share: format!("updated_prep_share_{ord}").into(),
                    },
                },
            )),
            want_report_aggregation.state().clone(),
        );

        ds.run_tx("test-update-report-aggregation", |tx| {
            let want_report_aggregation = want_report_aggregation.clone();
            Box::pin(async move {
                tx.update_report_aggregation(&want_report_aggregation)
                    .await
                    .unwrap();

                let row = tx
                    .query_one(
                        "--
SELECT updated_at, updated_by FROM report_aggregations
    WHERE client_report_id = $1",
                        &[&want_report_aggregation.report_id().as_ref()],
                    )
                    .await
                    .unwrap();
                let updated_at: chrono::NaiveDateTime = row.get("updated_at");
                let updated_by: &str = row.get("updated_by");

                assert_eq!(updated_at, tx.clock.now().as_naive_date_time().unwrap());
                assert_eq!(updated_by, "test-update-report-aggregation");

                Ok(())
            })
        })
        .await
        .unwrap();

        let got_report_aggregation = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = task.clone();

                Box::pin(async move {
                    tx.get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &role,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(Some(want_report_aggregation), got_report_aggregation);

        // Advance the clock again to expire relevant datastore items.
        clock.advance(&REPORT_EXPIRY_AGE);

        let got_report_aggregation = ds
            .run_unnamed_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                let task = task.clone();

                Box::pin(async move {
                    tx.get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &role,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(None, got_report_aggregation);
    }
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn report_aggregation_not_found(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let vdaf = Arc::new(dummy::Vdaf::default());
    let aggregation_param = dummy::AggregationParam(5);

    let rslt = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);

            Box::pin(async move {
                tx.get_report_aggregation_by_report_id(
                    vdaf.as_ref(),
                    &Role::Leader,
                    &random(),
                    &random(),
                    &ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                    &aggregation_param,
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(rslt, None);

    let rslt = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.update_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    random(),
                    random(),
                    ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                    Time::from_seconds_since_epoch(12345),
                    0,
                    None,
                    ReportAggregationState::Failed {
                        report_error: ReportError::VdafPrepError,
                    },
                ))
                .await
            })
        })
        .await;
    assert_matches!(rslt, Err(Error::MutationTargetNotFound));
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn get_report_aggregations_for_aggregation_job(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let report_id = random();
    let vdaf = Arc::new(dummy::Vdaf::new(2));
    let aggregation_param = dummy::AggregationParam(7);

    let vdaf_transcript = run_vdaf(
        vdaf.as_ref(),
        &task_id,
        &[],
        &aggregation_param,
        &report_id,
        &13,
    );

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .helper_view()
    .unwrap();
    let aggregation_job_id = random();

    let want_report_aggregations = ds
        .run_unnamed_tx(|tx| {
            let (task, vdaf_transcript, aggregation_param) =
                (task.clone(), vdaf_transcript.clone(), aggregation_param);
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                let mut want_report_aggregations = Vec::new();
                for (ord, state) in [
                    ReportAggregationState::LeaderInit {
                        public_extensions: Vec::new(),
                        public_share: vdaf_transcript.public_share,
                        leader_private_extensions: Vec::new(),
                        leader_input_share: vdaf_transcript.leader_input_share,
                        helper_encrypted_input_share: HpkeCiphertext::new(
                            HpkeConfigId::from(13),
                            Vec::from("encapsulated_context"),
                            Vec::from("payload"),
                        ),
                    },
                    ReportAggregationState::HelperContinue {
                        prepare_state: *vdaf_transcript.helper_prepare_transitions[0]
                            .prepare_state(),
                    },
                    ReportAggregationState::Finished,
                    ReportAggregationState::Failed {
                        report_error: ReportError::VdafPrepError,
                    },
                ]
                .iter()
                .enumerate()
                {
                    let report_id = ReportId::from((ord as u128).to_be_bytes());
                    tx.put_scrubbed_report(task.id(), &report_id, &OLDEST_ALLOWED_REPORT_TIMESTAMP)
                        .await
                        .unwrap();

                    let report_aggregation = ReportAggregation::new(
                        *task.id(),
                        aggregation_job_id,
                        report_id,
                        OLDEST_ALLOWED_REPORT_TIMESTAMP,
                        ord.try_into().unwrap(),
                        Some(PrepareResp::new(report_id, PrepareStepResult::Finished)),
                        state.clone(),
                    );
                    tx.put_report_aggregation(&report_aggregation)
                        .await
                        .unwrap();
                    want_report_aggregations.push(report_aggregation);
                }
                Ok(want_report_aggregations)
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    let got_report_aggregations = ds
        .run_unnamed_tx(|tx| {
            let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
            Box::pin(async move {
                tx.get_report_aggregations_for_aggregation_job(
                    vdaf.as_ref(),
                    &Role::Helper,
                    task.id(),
                    &aggregation_job_id,
                    &aggregation_param,
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(want_report_aggregations, got_report_aggregations);

    // Advance the clock again to expire relevant datastore entities.
    clock.advance(&REPORT_EXPIRY_AGE);

    let got_report_aggregations = ds
        .run_unnamed_tx(|tx| {
            let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
            Box::pin(async move {
                tx.get_report_aggregations_for_aggregation_job(
                    vdaf.as_ref(),
                    &Role::Helper,
                    task.id(),
                    &aggregation_job_id,
                    &aggregation_param,
                )
                .await
            })
        })
        .await
        .unwrap();
    assert!(got_report_aggregations.is_empty());
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn create_report_aggregation_from_client_reports_table(
    ephemeral_datastore: EphemeralDatastore,
) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let report_id = random();
    let vdaf = Arc::new(dummy::Vdaf::new(2));
    let aggregation_param = dummy::AggregationParam(7);

    let vdaf_transcript = run_vdaf(
        vdaf.as_ref(),
        &task_id,
        &[],
        &aggregation_param,
        &report_id,
        &13,
    );

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .leader_view()
    .unwrap();
    let aggregation_job_id = random();
    let want_report_aggregations = ds
        .run_unnamed_tx(|tx| {
            let clock = clock.clone();
            let task = task.clone();
            let vdaf_transcript = vdaf_transcript.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                let report_id = random();
                let timestamp = clock.now();
                let leader_stored_report = LeaderStoredReport::<0, dummy::Vdaf>::new(
                    *task.id(),
                    ReportMetadata::new(
                        report_id,
                        timestamp,
                        Vec::from([Extension::new(
                            ExtensionType::Tbd,
                            "public_extension_tbd".into(),
                        )]),
                    ),
                    (),
                    Vec::from([Extension::new(
                        ExtensionType::Taskbind,
                        "leader_private_extension_taskbind".into(),
                    )]),
                    vdaf_transcript.leader_input_share,
                    HpkeCiphertext::new(
                        HpkeConfigId::from(9),
                        Vec::from(b"encapsulated"),
                        Vec::from(b"encrypted helper share"),
                    ),
                );
                tx.put_client_report(&leader_stored_report).await.unwrap();

                let report_aggregation_metadata = ReportAggregationMetadata::new(
                    *task.id(),
                    aggregation_job_id,
                    report_id,
                    timestamp,
                    0,
                    ReportAggregationMetadataState::Init,
                );
                tx.put_leader_report_aggregation(&report_aggregation_metadata)
                    .await
                    .unwrap();

                Ok(Vec::from([ReportAggregation::new(
                    *task.id(),
                    aggregation_job_id,
                    report_id,
                    timestamp,
                    0,
                    None,
                    ReportAggregationState::<0, dummy::Vdaf>::LeaderInit {
                        public_extensions: leader_stored_report
                            .metadata()
                            .public_extensions()
                            .to_vec(),
                        public_share: *leader_stored_report.public_share(),
                        leader_private_extensions: leader_stored_report
                            .leader_private_extensions()
                            .to_vec(),
                        leader_input_share: *leader_stored_report.leader_input_share(),
                        helper_encrypted_input_share: leader_stored_report
                            .helper_encrypted_input_share()
                            .clone(),
                    },
                )]))
            })
        })
        .await
        .unwrap();

    let got_report_aggregations = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task = task.clone();
            Box::pin(async move {
                tx.get_report_aggregations_for_aggregation_job(
                    vdaf.as_ref(),
                    &Role::Leader,
                    task.id(),
                    &aggregation_job_id,
                    &aggregation_param,
                )
                .await
            })
        })
        .await
        .unwrap();
    assert_eq!(want_report_aggregations, got_report_aggregations);
}

#[tokio::test]
async fn crypter() {
    let crypter = Crypter::new(Vec::from([generate_aead_key(), generate_aead_key()]));
    let bad_key = generate_aead_key();

    const TABLE: &str = "some_table";
    const ROW: &[u8] = b"12345";
    const COLUMN: &str = "some_column";
    const PLAINTEXT: &[u8] = b"This is my plaintext value.";

    // Test that roundtripping encryption works.
    let ciphertext = crypter.encrypt(TABLE, ROW, COLUMN, PLAINTEXT).unwrap();
    let plaintext = crypter.decrypt(TABLE, ROW, COLUMN, &ciphertext).unwrap();
    assert_eq!(PLAINTEXT, &plaintext);

    // Roundtripping encryption works even if a non-primary key was used for encryption.
    let ciphertext =
        Crypter::encrypt_with_key(crypter.keys.last().unwrap(), TABLE, ROW, COLUMN, PLAINTEXT)
            .unwrap();
    let plaintext = crypter.decrypt(TABLE, ROW, COLUMN, &ciphertext).unwrap();
    assert_eq!(PLAINTEXT, &plaintext);

    // Roundtripping encryption with an unknown key fails.
    let ciphertext = Crypter::encrypt_with_key(&bad_key, TABLE, ROW, COLUMN, PLAINTEXT).unwrap();
    assert!(crypter.decrypt(TABLE, ROW, COLUMN, &ciphertext).is_err());

    // Roundtripping encryption with a mismatched table, row, or column fails.
    let ciphertext = crypter.encrypt(TABLE, ROW, COLUMN, PLAINTEXT).unwrap();
    assert!(crypter
        .decrypt("wrong_table", ROW, COLUMN, &ciphertext)
        .is_err());
    assert!(crypter
        .decrypt(TABLE, b"wrong_row", COLUMN, &ciphertext)
        .is_err());
    assert!(crypter
        .decrypt(TABLE, ROW, "wrong_column", &ciphertext)
        .is_err());
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn get_collection_job(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .leader_view()
    .unwrap();
    let first_batch_interval =
        Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(100)).unwrap();
    let second_batch_interval = Interval::new(
        OLDEST_ALLOWED_REPORT_TIMESTAMP
            .add(&Duration::from_seconds(100))
            .unwrap(),
        Duration::from_seconds(200),
    )
    .unwrap();
    let aggregation_param = dummy::AggregationParam(13);

    let (first_collection_job, second_collection_job) = ds
        .run_tx("test-put-collection-job", |tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                let first_collection_job = CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    random(),
                    Query::new_time_interval(first_batch_interval),
                    aggregation_param,
                    first_batch_interval,
                    CollectionJobState::Start,
                );
                tx.put_collection_job(&first_collection_job).await.unwrap();

                let second_collection_job = CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    random(),
                    Query::new_time_interval(second_batch_interval),
                    aggregation_param,
                    second_batch_interval,
                    CollectionJobState::Start,
                );
                tx.put_collection_job(&second_collection_job).await.unwrap();

                tx.check_timestamp_columns("collection_jobs", "test-put-collection-job", true)
                    .await;

                Ok((first_collection_job, second_collection_job))
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        let first_collection_job = first_collection_job.clone();
        let second_collection_job = second_collection_job.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let first_collection_job_again = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    first_collection_job.id(),
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(first_collection_job, first_collection_job_again);

            let second_collection_job_again = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    second_collection_job.id(),
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(second_collection_job, second_collection_job_again);

            // We can't get either of the collection jobs via `get_finished_collection_job`, as
            // neither is finished.
            assert!(tx
                .get_finished_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    first_collection_job.batch_identifier(),
                    first_collection_job.aggregation_parameter()
                )
                .await
                .unwrap()
                .is_none());
            assert!(tx
                .get_finished_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    second_collection_job.batch_identifier(),
                    second_collection_job.aggregation_parameter()
                )
                .await
                .unwrap()
                .is_none());

            let encrypted_helper_aggregate_share = hpke::seal(
                task.collector_hpke_config().unwrap(),
                &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
                &[0, 1, 2, 3, 4, 5],
                &AggregateShareAad::new(
                    *task.id(),
                    ().get_encoded().unwrap(),
                    BatchSelector::new_time_interval(first_batch_interval),
                )
                .get_encoded()
                .unwrap(),
            )
            .unwrap();

            let first_collection_job =
                first_collection_job.with_state(CollectionJobState::Finished {
                    report_count: 12,
                    client_timestamp_interval: first_batch_interval,
                    encrypted_helper_aggregate_share,
                    leader_aggregate_share: dummy::AggregateShare(41),
                });

            tx.update_collection_job::<0, TimeInterval, dummy::Vdaf>(&first_collection_job)
                .await
                .unwrap();

            let updated_first_collection_job = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    first_collection_job.id(),
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(first_collection_job, updated_first_collection_job);

            // We can now get the first of the collection jobs via `get_finished_collection_job`, as
            // it is now finished.
            assert_eq!(
                tx.get_finished_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    first_collection_job.batch_identifier(),
                    first_collection_job.aggregation_parameter()
                )
                .await
                .unwrap()
                .unwrap(),
                first_collection_job,
            );
            assert!(tx
                .get_finished_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    second_collection_job.batch_identifier(),
                    second_collection_job.aggregation_parameter()
                )
                .await
                .unwrap()
                .is_none());

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock again to expire everything that has been written.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        let first_collection_job = first_collection_job.clone();
        let second_collection_job = second_collection_job.clone();

        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let first_collection_job = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    first_collection_job.id(),
                )
                .await
                .unwrap();
            assert_eq!(first_collection_job, None);

            let second_collection_job = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    second_collection_job.id(),
                )
                .await
                .unwrap();
            assert_eq!(second_collection_job, None);

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn update_collection_jobs(ephemeral_datastore: EphemeralDatastore) {
    // Setup: write collection jobs to the datastore.
    install_test_trace_subscriber();

    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();
    let abandoned_batch_interval = Interval::new(
        Time::from_seconds_since_epoch(100),
        Duration::from_seconds(100),
    )
    .unwrap();
    let deleted_batch_interval = Interval::new(
        Time::from_seconds_since_epoch(200),
        Duration::from_seconds(100),
    )
    .unwrap();

    ds.run_tx("test-update-collection-jobs", |tx| {
        let task = task.clone();
        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();

            let vdaf = dummy::Vdaf::default();
            let aggregation_param = dummy::AggregationParam(10);
            let abandoned_collection_job = CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::new_time_interval(abandoned_batch_interval),
                aggregation_param,
                abandoned_batch_interval,
                CollectionJobState::Start,
            );
            tx.put_collection_job(&abandoned_collection_job)
                .await
                .unwrap();

            let deleted_collection_job = CollectionJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                random(),
                Query::new_time_interval(deleted_batch_interval),
                aggregation_param,
                deleted_batch_interval,
                CollectionJobState::Start,
            );
            tx.put_collection_job(&deleted_collection_job)
                .await
                .unwrap();

            let abandoned_collection_job_again = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    abandoned_collection_job.id(),
                )
                .await
                .unwrap()
                .unwrap();

            // Verify: initial state.
            assert_eq!(abandoned_collection_job, abandoned_collection_job_again);

            // Setup: update the collection jobs.
            let abandoned_collection_job =
                abandoned_collection_job.with_state(CollectionJobState::Abandoned);
            let deleted_collection_job =
                deleted_collection_job.with_state(CollectionJobState::Deleted);

            tx.update_collection_job::<0, TimeInterval, dummy::Vdaf>(&abandoned_collection_job)
                .await
                .unwrap();
            tx.update_collection_job::<0, TimeInterval, dummy::Vdaf>(&deleted_collection_job)
                .await
                .unwrap();

            let abandoned_collection_job_again = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    abandoned_collection_job.id(),
                )
                .await
                .unwrap()
                .unwrap();

            let deleted_collection_job_again = tx
                .get_collection_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    deleted_collection_job.id(),
                )
                .await
                .unwrap()
                .unwrap();

            // Verify: collection jobs were updated.
            assert_eq!(abandoned_collection_job, abandoned_collection_job_again);
            assert_eq!(deleted_collection_job, deleted_collection_job_again);

            tx.check_timestamp_columns("collection_jobs", "test-update-collection-jobs", true)
                .await;

            // Setup: try to update a job into state `Start`
            let abandoned_collection_job =
                abandoned_collection_job.with_state(CollectionJobState::Start);

            // Verify: Update should fail
            tx.update_collection_job::<0, TimeInterval, dummy::Vdaf>(&abandoned_collection_job)
                .await
                .unwrap_err();
            Ok(())
        })
    })
    .await
    .unwrap();
}

#[derive(Clone)]
struct CollectionJobTestCase<B: BatchMode> {
    should_be_acquired: bool,
    task_id: TaskId,
    batch_identifier: B::BatchIdentifier,
    agg_param: dummy::AggregationParam,
    collection_job_id: Option<CollectionJobId>,
    client_timestamp_interval: Interval,
    state: CollectionJobStateCode,
}

#[derive(Clone)]
struct CollectionJobAcquireTestCase<B: CollectableBatchMode> {
    task_ids: Vec<TaskId>,
    batch_mode: task::BatchMode,
    reports: Vec<LeaderStoredReport<0, dummy::Vdaf>>,
    aggregation_jobs: Vec<AggregationJob<0, B, dummy::Vdaf>>,
    report_aggregations: Vec<ReportAggregation<0, dummy::Vdaf>>,
    collection_job_test_cases: Vec<CollectionJobTestCase<B>>,
}

#[async_trait]
trait TestBatchModeExt: CollectableBatchMode {
    fn query_for_batch_identifier(batch_identifier: &Self::BatchIdentifier) -> Query<Self>;

    fn batch_identifier_for_client_timestamps(client_timestamps: &[Time]) -> Self::BatchIdentifier;

    async fn write_outstanding_batch(
        tx: &Transaction<MockClock>,
        task_id: &TaskId,
        batch_identifier: &Self::BatchIdentifier,
        time_bucket_start: &Option<Time>,
    ) -> Option<(TaskId, BatchId)>;
}

#[async_trait]
impl TestBatchModeExt for TimeInterval {
    fn query_for_batch_identifier(batch_identifier: &Self::BatchIdentifier) -> Query<Self> {
        Query::new_time_interval(*batch_identifier)
    }

    fn batch_identifier_for_client_timestamps(client_timestamps: &[Time]) -> Self::BatchIdentifier {
        let min_client_timestamp = *client_timestamps.iter().min().unwrap();
        let max_client_timestamp = *client_timestamps.iter().max().unwrap();
        Interval::new(
            min_client_timestamp,
            Duration::from_seconds(
                max_client_timestamp
                    .difference(&min_client_timestamp)
                    .unwrap()
                    .as_seconds()
                    + 1,
            ),
        )
        .unwrap()
    }

    async fn write_outstanding_batch(
        _: &Transaction<MockClock>,
        _: &TaskId,
        _: &Self::BatchIdentifier,
        _: &Option<Time>,
    ) -> Option<(TaskId, BatchId)> {
        None
    }
}

#[async_trait]
impl TestBatchModeExt for LeaderSelected {
    fn query_for_batch_identifier(_: &Self::BatchIdentifier) -> Query<Self> {
        Query::new_leader_selected()
    }

    fn batch_identifier_for_client_timestamps(_: &[Time]) -> Self::BatchIdentifier {
        random()
    }

    async fn write_outstanding_batch(
        tx: &Transaction<MockClock>,
        task_id: &TaskId,
        batch_identifier: &Self::BatchIdentifier,
        time_bucket_start: &Option<Time>,
    ) -> Option<(TaskId, BatchId)> {
        tx.put_outstanding_batch(task_id, batch_identifier, time_bucket_start)
            .await
            .unwrap();
        Some((*task_id, *batch_identifier))
    }
}

async fn setup_collection_job_acquire_test_case<B: TestBatchModeExt>(
    ds: &Datastore<MockClock>,
    test_case: CollectionJobAcquireTestCase<B>,
) -> CollectionJobAcquireTestCase<B> {
    ds.run_unnamed_tx(|tx| {
        let mut test_case = test_case.clone();
        Box::pin(async move {
            for task_id in &test_case.task_ids {
                tx.put_aggregator_task(
                    &TaskBuilder::new(
                        test_case.batch_mode,
                        AggregationMode::Synchronous,
                        VdafInstance::Fake { rounds: 1 },
                    )
                    .with_id(*task_id)
                    .build()
                    .leader_view()
                    .unwrap(),
                )
                .await
                .unwrap();
            }

            for report in &test_case.reports {
                tx.put_client_report(report).await.unwrap();
            }
            for aggregation_job in &test_case.aggregation_jobs {
                tx.put_aggregation_job(aggregation_job).await.unwrap();
            }

            for report_aggregation in &test_case.report_aggregations {
                tx.put_report_aggregation(report_aggregation).await.unwrap();
            }

            for test_case in test_case.collection_job_test_cases.iter_mut() {
                tx.put_batch_aggregation(&BatchAggregation::<0, B, dummy::Vdaf>::new(
                    test_case.task_id,
                    test_case.batch_identifier.clone(),
                    test_case.agg_param,
                    0,
                    test_case.client_timestamp_interval,
                    BatchAggregationState::Scrubbed,
                ))
                .await
                .unwrap();

                let collection_job_id = random();
                tx.put_collection_job(&CollectionJob::<0, B, dummy::Vdaf>::new(
                    test_case.task_id,
                    collection_job_id,
                    B::query_for_batch_identifier(&test_case.batch_identifier),
                    test_case.agg_param,
                    test_case.batch_identifier.clone(),
                    match test_case.state {
                        CollectionJobStateCode::Start => CollectionJobState::Start,
                        CollectionJobStateCode::Finished => CollectionJobState::Finished {
                            report_count: 1,
                            client_timestamp_interval: test_case.client_timestamp_interval,
                            encrypted_helper_aggregate_share: HpkeCiphertext::new(
                                HpkeConfigId::from(0),
                                Vec::new(),
                                Vec::new(),
                            ),
                            leader_aggregate_share: dummy::AggregateShare(0),
                        },
                        CollectionJobStateCode::Abandoned => CollectionJobState::Abandoned,
                        CollectionJobStateCode::Deleted => CollectionJobState::Deleted,
                    },
                ))
                .await
                .unwrap();

                test_case.collection_job_id = Some(collection_job_id);
            }

            Ok(test_case)
        })
    })
    .await
    .unwrap()
}

async fn run_collection_job_acquire_test_case<B: TestBatchModeExt>(
    ds: &Datastore<MockClock>,
    test_case: CollectionJobAcquireTestCase<B>,
) -> Vec<Lease<AcquiredCollectionJob>> {
    let test_case = setup_collection_job_acquire_test_case(ds, test_case).await;

    let clock = &ds.clock;
    ds.run_unnamed_tx(|tx| {
        let test_case = test_case.clone();
        let clock = clock.clone();

        Box::pin(async move {
            let leases = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap();

            let mut leased_collection_jobs: Vec<_> = leases
                .iter()
                .map(|lease| (lease.leased().clone(), *lease.lease_expiry_time()))
                .collect();
            leased_collection_jobs.sort();

            let mut expected_collection_jobs: Vec<_> = test_case
                .collection_job_test_cases
                .iter()
                .filter(|c| c.should_be_acquired)
                .map(|c| {
                    (
                        AcquiredCollectionJob::new(
                            c.task_id,
                            c.collection_job_id.unwrap(),
                            test_case.batch_mode,
                            VdafInstance::Fake { rounds: 1 },
                            Duration::from_hours(8).unwrap(),
                            c.batch_identifier.get_encoded().unwrap(),
                            c.agg_param.get_encoded().unwrap(),
                            0,
                        ),
                        clock.now().as_naive_date_time().unwrap()
                            + chrono::Duration::try_seconds(100).unwrap(),
                    )
                })
                .collect();
            expected_collection_jobs.sort();

            assert_eq!(leased_collection_jobs, expected_collection_jobs);

            Ok(leases)
        })
    })
    .await
    .unwrap()
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn time_interval_collection_job_acquire_release_happy_path(
    ephemeral_datastore: EphemeralDatastore,
) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let test_start = clock.now();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let reports = Vec::from([LeaderStoredReport::new_dummy(
        task_id,
        Time::from_seconds_since_epoch(0),
    )]);
    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(0),
        Duration::from_seconds(100),
    )
    .unwrap();
    let aggregation_job_id = random();
    let aggregation_jobs = Vec::from([AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        task_id,
        aggregation_job_id,
        dummy::AggregationParam(0),
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(1),
    )]);
    let report_aggregations = Vec::from([ReportAggregation::<0, dummy::Vdaf>::new(
        task_id,
        aggregation_job_id,
        *reports[0].metadata().id(),
        *reports[0].metadata().time(),
        0,
        None,
        ReportAggregationState::Finished, // Doesn't matter what state the report aggregation is in
    )]);

    let collection_job_test_cases = Vec::from([CollectionJobTestCase::<TimeInterval> {
        should_be_acquired: true,
        task_id,
        batch_identifier: batch_interval,
        agg_param: dummy::AggregationParam(0),
        collection_job_id: None,
        client_timestamp_interval: Interval::EMPTY,
        state: CollectionJobStateCode::Start,
    }]);

    let collection_job_leases = run_collection_job_acquire_test_case(
        &ds,
        CollectionJobAcquireTestCase {
            task_ids: Vec::from([task_id]),
            batch_mode: task::BatchMode::TimeInterval,
            reports,
            aggregation_jobs,
            report_aggregations,
            collection_job_test_cases,
        },
    )
    .await;

    let reacquired_jobs = ds
        .run_tx("test-acquire-leases", |tx| {
            let collection_job_leases = collection_job_leases.clone();
            Box::pin(async move {
                // Try to re-acquire collection jobs. Nothing should happen because the lease is still
                // valid.
                assert!(tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await
                    .unwrap()
                    .is_empty());

                // Release the lease, then re-acquire it.
                tx.release_collection_job(&collection_job_leases[0], None)
                    .await
                    .unwrap();

                tx.check_timestamp_columns("collection_jobs", "test-acquire-leases", true)
                    .await;

                let reacquired_leases = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await
                    .unwrap();
                let reacquired_jobs: Vec<_> = reacquired_leases
                    .iter()
                    .map(|lease| (lease.leased().clone(), lease.lease_expiry_time()))
                    .collect();

                let collection_jobs: Vec<_> = collection_job_leases
                    .iter()
                    .map(|lease| {
                        (
                            lease.leased().clone().with_step_attempts(1),
                            lease.lease_expiry_time(),
                        )
                    })
                    .collect();

                assert_eq!(reacquired_jobs, collection_jobs);

                Ok(reacquired_leases)
            })
        })
        .await
        .unwrap();

    // Advance time by the lease duration
    clock.advance(&Duration::from_seconds(100));

    ds.run_tx("test-reacquire-leases", |tx| {
        let reacquired_jobs = reacquired_jobs.clone();
        Box::pin(async move {
            // Re-acquire the jobs whose lease should have lapsed.
            let acquired_jobs = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap();

            for (acquired_job, reacquired_job) in acquired_jobs.iter().zip(reacquired_jobs) {
                assert_eq!(acquired_job.leased(), reacquired_job.leased());
                assert_eq!(
                    *acquired_job.lease_expiry_time(),
                    *reacquired_job.lease_expiry_time()
                        + chrono::Duration::try_seconds(100).unwrap(),
                );
            }

            tx.check_timestamp_columns_at_create_time(
                "collection_jobs",
                "test-reacquire-leases",
                test_start,
                true,
            )
            .await;

            // Release the job with a reacquire delay, and verify we can't acquire it again.
            tx.release_collection_job(&acquired_jobs[0], Some(&StdDuration::from_secs(600)))
                .await
                .unwrap();

            assert!(tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap()
                .is_empty());

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance time by the reacquire delay, and verify we can reacquire the job.
    clock.advance(&Duration::from_seconds(600));

    ds.run_unnamed_tx(|tx| {
        let collection_job_leases = collection_job_leases.clone();

        Box::pin(async move {
            let reacquired_leases = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap();

            let reacquired_jobs: Vec<_> = reacquired_leases
                .iter()
                .map(|lease| lease.leased().clone())
                .collect();
            let collection_jobs: Vec<_> = collection_job_leases
                .iter()
                .map(|lease| lease.leased().clone().with_step_attempts(2))
                .collect();

            assert_eq!(reacquired_jobs, collection_jobs);

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn leader_selected_collection_job_acquire_release_happy_path(
    ephemeral_datastore: EphemeralDatastore,
) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let reports = Vec::from([LeaderStoredReport::new_dummy(
        task_id,
        Time::from_seconds_since_epoch(0),
    )]);
    let batch_id = random();
    let aggregation_job_id = random();
    let aggregation_jobs = Vec::from([AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        task_id,
        aggregation_job_id,
        dummy::AggregationParam(0),
        batch_id,
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(1),
    )]);
    let report_aggregations = Vec::from([ReportAggregation::<0, dummy::Vdaf>::new(
        task_id,
        aggregation_job_id,
        *reports[0].metadata().id(),
        *reports[0].metadata().time(),
        0,
        None,
        ReportAggregationState::Finished, // Doesn't matter what state the report aggregation is in
    )]);

    let collection_job_leases = run_collection_job_acquire_test_case(
        &ds,
        CollectionJobAcquireTestCase {
            task_ids: Vec::from([task_id]),
            batch_mode: task::BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            reports,
            aggregation_jobs,
            report_aggregations,
            collection_job_test_cases: Vec::from([CollectionJobTestCase::<LeaderSelected> {
                should_be_acquired: true,
                task_id,
                batch_identifier: batch_id,
                agg_param: dummy::AggregationParam(0),
                collection_job_id: None,
                client_timestamp_interval: Interval::new(
                    Time::from_seconds_since_epoch(0),
                    Duration::from_seconds(1),
                )
                .unwrap(),
                state: CollectionJobStateCode::Start,
            }]),
        },
    )
    .await;

    let reacquired_jobs = ds
        .run_unnamed_tx(|tx| {
            let collection_job_leases = collection_job_leases.clone();
            Box::pin(async move {
                // Try to re-acquire collection jobs. Nothing should happen because the lease is still
                // valid.
                assert!(tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10,)
                    .await
                    .unwrap()
                    .is_empty());

                // Release the lease, then re-acquire it.
                tx.release_collection_job(&collection_job_leases[0], None)
                    .await
                    .unwrap();

                let reacquired_leases = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await
                    .unwrap();
                let reacquired_jobs: Vec<_> = reacquired_leases
                    .iter()
                    .map(|lease| (lease.leased().clone(), lease.lease_expiry_time()))
                    .collect();

                let collection_jobs: Vec<_> = collection_job_leases
                    .iter()
                    .map(|lease| {
                        (
                            lease.leased().clone().with_step_attempts(1),
                            lease.lease_expiry_time(),
                        )
                    })
                    .collect();

                assert_eq!(reacquired_jobs, collection_jobs);

                Ok(reacquired_leases)
            })
        })
        .await
        .unwrap();

    // Advance time by the lease duration
    clock.advance(&Duration::from_seconds(100));

    ds.run_unnamed_tx(|tx| {
        let reacquired_jobs = reacquired_jobs.clone();
        Box::pin(async move {
            // Re-acquire the jobs whose lease should have lapsed.
            let acquired_jobs = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap();

            for (acquired_job, reacquired_job) in acquired_jobs.iter().zip(reacquired_jobs) {
                assert_eq!(acquired_job.leased(), reacquired_job.leased());
                assert_eq!(
                    *acquired_job.lease_expiry_time(),
                    *reacquired_job.lease_expiry_time()
                        + chrono::Duration::try_seconds(100).unwrap(),
                );
            }

            // Release the job with a reacquire delay, and verify we can't acquire it again.
            tx.release_collection_job(&acquired_jobs[0], Some(&StdDuration::from_secs(600)))
                .await
                .unwrap();

            assert!(tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap()
                .is_empty());

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance time by the reacquire delay, and verify we can reacquire the job.
    clock.advance(&Duration::from_seconds(600));

    ds.run_unnamed_tx(|tx| {
        let collection_job_leases = collection_job_leases.clone();

        Box::pin(async move {
            let reacquired_leases = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap();

            let reacquired_jobs: Vec<_> = reacquired_leases
                .iter()
                .map(|lease| lease.leased().clone())
                .collect();
            let collection_jobs: Vec<_> = collection_job_leases
                .iter()
                .map(|lease| lease.leased().clone().with_step_attempts(2))
                .collect();

            assert_eq!(reacquired_jobs, collection_jobs);

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn collection_job_acquire_release_job_finished(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let reports = Vec::from([LeaderStoredReport::new_dummy(
        task_id,
        Time::from_seconds_since_epoch(0),
    )]);
    let aggregation_job_id = random();
    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(0),
        Duration::from_seconds(100),
    )
    .unwrap();
    let aggregation_jobs = Vec::from([AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        task_id,
        aggregation_job_id,
        dummy::AggregationParam(0),
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(1),
    )]);

    let report_aggregations = Vec::from([ReportAggregation::<0, dummy::Vdaf>::new(
        task_id,
        aggregation_job_id,
        *reports[0].metadata().id(),
        *reports[0].metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    )]);

    let collection_job_test_cases = Vec::from([CollectionJobTestCase::<TimeInterval> {
        should_be_acquired: false,
        task_id,
        batch_identifier: batch_interval,
        agg_param: dummy::AggregationParam(0),
        collection_job_id: None,
        client_timestamp_interval: Interval::EMPTY,
        // collection job has already run to completion
        state: CollectionJobStateCode::Finished,
    }]);

    run_collection_job_acquire_test_case(
        &ds,
        CollectionJobAcquireTestCase {
            task_ids: Vec::from([task_id]),
            batch_mode: task::BatchMode::TimeInterval,
            reports,
            aggregation_jobs,
            report_aggregations,
            collection_job_test_cases,
        },
    )
    .await;
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn collection_job_acquire_job_max(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let reports = Vec::from([LeaderStoredReport::new_dummy(
        task_id,
        Time::from_seconds_since_epoch(0),
    )]);
    let aggregation_job_ids: [_; 2] = random();
    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(0),
        Duration::from_seconds(100),
    )
    .unwrap();
    let aggregation_jobs = Vec::from([
        AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[0],
            dummy::AggregationParam(0),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        ),
        AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[1],
            dummy::AggregationParam(1),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        ),
    ]);
    let report_aggregations = Vec::from([
        ReportAggregation::<0, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[0],
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        ),
        ReportAggregation::<0, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[1],
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        ),
    ]);

    let collection_job_test_cases = Vec::from([
        CollectionJobTestCase::<TimeInterval> {
            should_be_acquired: true,
            task_id,
            batch_identifier: batch_interval,
            agg_param: dummy::AggregationParam(0),
            collection_job_id: None,
            client_timestamp_interval: batch_interval,
            state: CollectionJobStateCode::Start,
        },
        CollectionJobTestCase::<TimeInterval> {
            should_be_acquired: true,
            task_id,
            batch_identifier: batch_interval,
            agg_param: dummy::AggregationParam(1),
            collection_job_id: None,
            client_timestamp_interval: batch_interval,
            state: CollectionJobStateCode::Start,
        },
    ]);

    let test_case = setup_collection_job_acquire_test_case(
        &ds,
        CollectionJobAcquireTestCase::<TimeInterval> {
            task_ids: Vec::from([task_id]),
            batch_mode: task::BatchMode::TimeInterval,
            reports,
            aggregation_jobs,
            report_aggregations,
            collection_job_test_cases,
        },
    )
    .await;

    ds.run_unnamed_tx(|tx| {
        let test_case = test_case.clone();
        let clock = clock.clone();
        Box::pin(async move {
            // Acquire a single collection job, twice. Each call should yield one job. We don't
            // care what order they are acquired in.
            let mut acquired_collection_jobs = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                .await
                .unwrap();
            assert_eq!(acquired_collection_jobs.len(), 1);

            acquired_collection_jobs.extend(
                tx.acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                    .await
                    .unwrap(),
            );

            assert_eq!(acquired_collection_jobs.len(), 2);

            let mut acquired_collection_jobs: Vec<_> = acquired_collection_jobs
                .iter()
                .map(|lease| (lease.leased().clone(), *lease.lease_expiry_time()))
                .collect();
            acquired_collection_jobs.sort();

            let mut expected_collection_jobs: Vec<_> = test_case
                .collection_job_test_cases
                .iter()
                .filter(|c| c.should_be_acquired)
                .map(|c| {
                    (
                        AcquiredCollectionJob::new(
                            c.task_id,
                            c.collection_job_id.unwrap(),
                            task::BatchMode::TimeInterval,
                            VdafInstance::Fake { rounds: 1 },
                            Duration::from_hours(8).unwrap(),
                            c.batch_identifier.get_encoded().unwrap(),
                            c.agg_param.get_encoded().unwrap(),
                            0,
                        ),
                        clock.now().as_naive_date_time().unwrap()
                            + chrono::Duration::try_seconds(100).unwrap(),
                    )
                })
                .collect();
            expected_collection_jobs.sort();

            assert_eq!(acquired_collection_jobs, expected_collection_jobs);

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn collection_job_acquire_state_filtering(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let task_id = random();
    let reports = Vec::from([LeaderStoredReport::new_dummy(
        task_id,
        Time::from_seconds_since_epoch(0),
    )]);
    let aggregation_job_ids: [_; 3] = random();
    let batch_interval = Interval::new(
        Time::from_seconds_since_epoch(0),
        Duration::from_seconds(100),
    )
    .unwrap();
    let aggregation_jobs = Vec::from([
        AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[0],
            dummy::AggregationParam(0),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        ),
        AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[1],
            dummy::AggregationParam(1),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        ),
        AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[2],
            dummy::AggregationParam(2),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        ),
    ]);
    let report_aggregations = Vec::from([
        ReportAggregation::<0, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[0],
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        ),
        ReportAggregation::<0, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[1],
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        ),
        ReportAggregation::<0, dummy::Vdaf>::new(
            task_id,
            aggregation_job_ids[2],
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        ),
    ]);

    let collection_job_test_cases = Vec::from([
        CollectionJobTestCase::<TimeInterval> {
            should_be_acquired: true,
            task_id,
            batch_identifier: batch_interval,
            agg_param: dummy::AggregationParam(0),
            collection_job_id: None,
            client_timestamp_interval: Interval::EMPTY,
            state: CollectionJobStateCode::Finished,
        },
        CollectionJobTestCase::<TimeInterval> {
            should_be_acquired: true,
            task_id,
            batch_identifier: batch_interval,
            agg_param: dummy::AggregationParam(1),
            collection_job_id: None,
            client_timestamp_interval: Interval::EMPTY,
            state: CollectionJobStateCode::Abandoned,
        },
        CollectionJobTestCase::<TimeInterval> {
            should_be_acquired: true,
            task_id,
            batch_identifier: batch_interval,
            agg_param: dummy::AggregationParam(2),
            collection_job_id: None,
            client_timestamp_interval: Interval::EMPTY,
            state: CollectionJobStateCode::Deleted,
        },
    ]);

    setup_collection_job_acquire_test_case(
        &ds,
        CollectionJobAcquireTestCase {
            task_ids: Vec::from([task_id]),
            batch_mode: task::BatchMode::TimeInterval,
            reports,
            aggregation_jobs,
            report_aggregations,
            collection_job_test_cases,
        },
    )
    .await;

    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            // No collection jobs should be acquired because none of them are in the START state
            let acquired_collection_jobs = tx
                .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                .await
                .unwrap();
            assert!(acquired_collection_jobs.is_empty());

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_batch_aggregation_time_interval(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let time_precision = Duration::from_seconds(100);
    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_time_precision(time_precision)
    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
    .build()
    .leader_view()
    .unwrap();
    let other_task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();
    let aggregation_param = dummy::AggregationParam(12);
    let aggregate_share = dummy::AggregateShare(23);

    let (
        first_batch_aggregation,
        second_batch_aggregation,
        third_batch_aggregation,
        fourth_batch_aggregation,
    ) = ds
        .run_tx("test-put-batch-aggregations", |tx| {
            let task = task.clone();
            let other_task = other_task.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_aggregator_task(&other_task).await.unwrap();

                let first_batch_aggregation = BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    Interval::new(Time::from_seconds_since_epoch(1100), time_precision).unwrap(),
                    aggregation_param,
                    0,
                    Interval::EMPTY,
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(aggregate_share),
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 3,
                        aggregation_jobs_terminated: 2,
                    },
                );

                let second_batch_aggregation =
                    BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        Interval::new(Time::from_seconds_since_epoch(1200), time_precision)
                            .unwrap(),
                        aggregation_param,
                        1,
                        Interval::EMPTY,
                        BatchAggregationState::Collected {
                            aggregate_share: None,
                            report_count: 0,
                            checksum: ReportIdChecksum::default(),
                            aggregation_jobs_created: 4,
                            aggregation_jobs_terminated: 4,
                        },
                    );

                let third_batch_aggregation = BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    Interval::new(Time::from_seconds_since_epoch(1300), time_precision).unwrap(),
                    aggregation_param,
                    2,
                    Interval::EMPTY,
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(aggregate_share),
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 5,
                        aggregation_jobs_terminated: 1,
                    },
                );
                let fourth_batch_aggregation =
                    BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                        *task.id(),
                        Interval::new(Time::from_seconds_since_epoch(1400), time_precision)
                            .unwrap(),
                        aggregation_param,
                        3,
                        Interval::EMPTY,
                        BatchAggregationState::Scrubbed,
                    );

                // Start of this aggregation's interval is before the interval queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    Interval::new(Time::from_seconds_since_epoch(1000), time_precision).unwrap(),
                    aggregation_param,
                    4,
                    Interval::EMPTY,
                    BatchAggregationState::Collected {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 3,
                        aggregation_jobs_terminated: 3,
                    },
                ))
                .await
                .unwrap();

                // Following three batches are within the interval queried below.
                tx.put_batch_aggregation(&first_batch_aggregation)
                    .await
                    .unwrap();
                tx.put_batch_aggregation(&second_batch_aggregation)
                    .await
                    .unwrap();
                tx.put_batch_aggregation(&third_batch_aggregation)
                    .await
                    .unwrap();
                tx.put_batch_aggregation(&fourth_batch_aggregation)
                    .await
                    .unwrap();

                assert_matches!(
                    tx.put_batch_aggregation(&first_batch_aggregation).await,
                    Err(Error::MutationTargetAlreadyExists)
                );

                // Aggregation parameter differs from the one queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    Interval::new(Time::from_seconds_since_epoch(1000), time_precision).unwrap(),
                    dummy::AggregationParam(13),
                    5,
                    Interval::EMPTY,
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(aggregate_share),
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 8,
                        aggregation_jobs_terminated: 6,
                    },
                ))
                .await
                .unwrap();

                // Start of this aggregation's interval is after the interval queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    Interval::new(Time::from_seconds_since_epoch(1500), time_precision).unwrap(),
                    aggregation_param,
                    6,
                    Interval::EMPTY,
                    BatchAggregationState::Collected {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 7,
                        aggregation_jobs_terminated: 7,
                    },
                ))
                .await
                .unwrap();

                // Task ID differs from that queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *other_task.id(),
                    Interval::new(Time::from_seconds_since_epoch(1200), time_precision).unwrap(),
                    aggregation_param,
                    7,
                    Interval::EMPTY,
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(aggregate_share),
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 8,
                        aggregation_jobs_terminated: 2,
                    },
                ))
                .await
                .unwrap();

                tx.check_timestamp_columns(
                    "batch_aggregations",
                    "test-put-batch-aggregations",
                    true,
                )
                .await;

                Ok((
                    first_batch_aggregation,
                    second_batch_aggregation,
                    third_batch_aggregation,
                    fourth_batch_aggregation,
                ))
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        let first_batch_aggregation = first_batch_aggregation.clone();
        let second_batch_aggregation = second_batch_aggregation.clone();
        let third_batch_aggregation = third_batch_aggregation.clone();
        let fourth_batch_aggregation = fourth_batch_aggregation.clone();

        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let batch_aggregations =
                TimeInterval::get_batch_aggregations_for_collection_identifier::<
                    0,
                    dummy::Vdaf,
                    _,
                >(
                    tx,
                    task.id(),
                    task.time_precision(),
                    &vdaf,
                    &Interval::new(
                        Time::from_seconds_since_epoch(1100),
                        Duration::from_seconds(4 * time_precision.as_seconds()),
                    )
                    .unwrap(),
                    &aggregation_param,
                )
                .await
                .unwrap();

            assert_eq!(batch_aggregations.len(), 4, "{batch_aggregations:#?}");
            for batch_aggregation in [
                &first_batch_aggregation,
                &second_batch_aggregation,
                &third_batch_aggregation,
                &fourth_batch_aggregation,
            ] {
                assert!(
                    batch_aggregations.contains(batch_aggregation),
                    "{batch_aggregations:#?}"
                );
            }

            let first_batch_aggregation = BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                *first_batch_aggregation.task_id(),
                *first_batch_aggregation.batch_interval(),
                *first_batch_aggregation.aggregation_parameter(),
                first_batch_aggregation.ord(),
                Interval::new(Time::from_seconds_since_epoch(1100), time_precision).unwrap(),
                BatchAggregationState::Aggregating {
                    aggregate_share: Some(dummy::AggregateShare(92)),
                    report_count: 1,
                    checksum: ReportIdChecksum::get_decoded(&[1; 32]).unwrap(),
                    aggregation_jobs_created: 4,
                    aggregation_jobs_terminated: 3,
                },
            );
            tx.update_batch_aggregation(&first_batch_aggregation)
                .await
                .unwrap();

            let batch_aggregations =
                TimeInterval::get_batch_aggregations_for_collection_identifier::<
                    0,
                    dummy::Vdaf,
                    _,
                >(
                    tx,
                    task.id(),
                    task.time_precision(),
                    &vdaf,
                    &Interval::new(
                        Time::from_seconds_since_epoch(1100),
                        Duration::from_seconds(4 * time_precision.as_seconds()),
                    )
                    .unwrap(),
                    &aggregation_param,
                )
                .await
                .unwrap();

            assert_eq!(batch_aggregations.len(), 4, "{batch_aggregations:#?}");
            for batch_aggregation in [
                &first_batch_aggregation,
                &second_batch_aggregation,
                &third_batch_aggregation,
                &fourth_batch_aggregation,
            ] {
                assert!(
                    batch_aggregations.contains(batch_aggregation),
                    "{batch_aggregations:#?}"
                );
            }

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock again to expire all written entities.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let batch_aggregations: Vec<BatchAggregation<0, TimeInterval, dummy::Vdaf>> =
                TimeInterval::get_batch_aggregations_for_collection_identifier::<
                    0,
                    dummy::Vdaf,
                    _,
                >(
                    tx,
                    task.id(),
                    task.time_precision(),
                    &vdaf,
                    &Interval::new(
                        Time::from_seconds_since_epoch(1100),
                        Duration::from_seconds(3 * time_precision.as_seconds()),
                    )
                    .unwrap(),
                    &aggregation_param,
                )
                .await
                .unwrap();

            assert!(batch_aggregations.is_empty());

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_batch_aggregation_leader_selected(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

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
    let batch_id = random();
    let aggregate_share = dummy::AggregateShare(23);
    let aggregation_param = dummy::AggregationParam(12);
    let batch_aggregation = ds
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                let other_task = TaskBuilder::new(
                    task::BatchMode::LeaderSelected {
                        batch_time_window_size: None,
                    },
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap();

                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_aggregator_task(&other_task).await.unwrap();

                let batch_aggregation = BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    0,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(100))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(aggregate_share),
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 3,
                        aggregation_jobs_terminated: 2,
                    },
                );

                // Following batch aggregations have the batch ID queried below.
                tx.put_batch_aggregation(&batch_aggregation).await.unwrap();

                assert_matches!(
                    tx.put_batch_aggregation(&batch_aggregation).await,
                    Err(Error::MutationTargetAlreadyExists)
                );

                // Wrong batch ID.
                let other_batch_id = random();
                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    other_batch_id,
                    aggregation_param,
                    1,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Collected {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 4,
                        aggregation_jobs_terminated: 4,
                    },
                ))
                .await
                .unwrap();

                // Task ID differs from that queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *other_task.id(),
                    batch_id,
                    aggregation_param,
                    2,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(aggregate_share),
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 5,
                        aggregation_jobs_terminated: 1,
                    },
                ))
                .await
                .unwrap();

                // Index differs from that queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    3,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Collected {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 6,
                        aggregation_jobs_terminated: 6,
                    },
                ))
                .await
                .unwrap();
                Ok(batch_aggregation)
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        let batch_aggregation = batch_aggregation.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let got_batch_aggregation = tx
                .get_batch_aggregation::<0, LeaderSelected, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    &batch_id,
                    &aggregation_param,
                    0,
                )
                .await
                .unwrap();
            assert_eq!(got_batch_aggregation.as_ref(), Some(&batch_aggregation));

            let batch_aggregation = BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                *batch_aggregation.task_id(),
                *batch_aggregation.batch_id(),
                *batch_aggregation.aggregation_parameter(),
                batch_aggregation.ord(),
                Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(200))
                    .unwrap(),
                BatchAggregationState::Aggregating {
                    aggregate_share: None,
                    report_count: 1,
                    checksum: ReportIdChecksum::get_decoded(&[1; 32]).unwrap(),
                    aggregation_jobs_created: 4,
                    aggregation_jobs_terminated: 2,
                },
            );
            tx.update_batch_aggregation(&batch_aggregation)
                .await
                .unwrap();

            let got_batch_aggregation = tx
                .get_batch_aggregation::<0, LeaderSelected, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    &batch_id,
                    &aggregation_param,
                    0,
                )
                .await
                .unwrap();
            assert_eq!(got_batch_aggregation, Some(batch_aggregation));
            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock again to expire all written entities.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let task = task.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let got_batch_aggregation = tx
                .get_batch_aggregation::<0, LeaderSelected, dummy::Vdaf>(
                    &vdaf,
                    task.id(),
                    &batch_id,
                    &aggregation_param,
                    0,
                )
                .await
                .unwrap();
            assert!(got_batch_aggregation.is_none());

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_aggregate_share_job_time_interval(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let aggregate_share_job = ds
        .run_tx("test-roundtrip-aggregate-share-job", |tx| {
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
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(100))
                        .unwrap(),
                    dummy::AggregationParam(11),
                    0,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(100))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(dummy::AggregateShare(0)),
                        report_count: 1,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 7,
                        aggregation_jobs_terminated: 5,
                    },
                ))
                .await
                .unwrap();

                let aggregate_share_job = AggregateShareJob::new(
                    *task.id(),
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(100))
                        .unwrap(),
                    dummy::AggregationParam(11),
                    dummy::AggregateShare(42),
                    10,
                    ReportIdChecksum::get_decoded(&[1; 32]).unwrap(),
                );

                tx.put_aggregate_share_job::<0, TimeInterval, dummy::Vdaf>(&aggregate_share_job)
                    .await
                    .unwrap();

                tx.check_timestamp_columns(
                    "aggregate_share_jobs",
                    "test-roundtrip-aggregate-share-job",
                    false,
                )
                .await;

                Ok(aggregate_share_job)
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let want_aggregate_share_job = aggregate_share_job.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let got_aggregate_share_job = tx
                .get_aggregate_share_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    want_aggregate_share_job.batch_interval(),
                    want_aggregate_share_job.aggregation_parameter(),
                )
                .await
                .unwrap()
                .unwrap();

            assert_eq!(want_aggregate_share_job, got_aggregate_share_job);

            assert!(tx
                .get_aggregate_share_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    &Interval::new(
                        Time::from_seconds_since_epoch(500),
                        Duration::from_seconds(100),
                    )
                    .unwrap(),
                    want_aggregate_share_job.aggregation_parameter(),
                )
                .await
                .unwrap()
                .is_none());

            let want_aggregate_share_jobs = Vec::from([want_aggregate_share_job.clone()]);

            let got_aggregate_share_jobs = tx
                .get_aggregate_share_jobs_intersecting_interval::<0, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    &Interval::new(
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(5))
                            .unwrap(),
                        Duration::from_seconds(10),
                    )
                    .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(got_aggregate_share_jobs, want_aggregate_share_jobs);

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to expire all written entities.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let want_aggregate_share_job = aggregate_share_job.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            assert_eq!(
                tx.get_aggregate_share_job::<0, TimeInterval, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    want_aggregate_share_job.batch_interval(),
                    want_aggregate_share_job.aggregation_parameter(),
                )
                .await
                .unwrap(),
                None
            );

            assert!(tx
                .get_aggregate_share_jobs_intersecting_interval::<0, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    &Interval::new(
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(5))
                            .unwrap(),
                        Duration::from_seconds(10),
                    )
                    .unwrap(),
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

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_aggregate_share_job_leader_selected(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let aggregate_share_job = ds
        .run_unnamed_tx(|tx| {
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
                tx.put_aggregator_task(&task).await.unwrap();

                let batch_id = random();
                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    dummy::AggregationParam(11),
                    0,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(dummy::AggregateShare(0)),
                        report_count: 1,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 7,
                        aggregation_jobs_terminated: 5,
                    },
                ))
                .await
                .unwrap();

                let aggregate_share_job = AggregateShareJob::new(
                    *task.id(),
                    batch_id,
                    dummy::AggregationParam(11),
                    dummy::AggregateShare(42),
                    10,
                    ReportIdChecksum::get_decoded(&[1; 32]).unwrap(),
                );

                tx.put_aggregate_share_job::<0, LeaderSelected, dummy::Vdaf>(&aggregate_share_job)
                    .await
                    .unwrap();

                Ok(aggregate_share_job)
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let want_aggregate_share_job = aggregate_share_job.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            let got_aggregate_share_job = tx
                .get_aggregate_share_job::<0, LeaderSelected, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    want_aggregate_share_job.batch_id(),
                    want_aggregate_share_job.aggregation_parameter(),
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(want_aggregate_share_job, got_aggregate_share_job);

            assert!(tx
                .get_aggregate_share_job::<0, LeaderSelected, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    &random(),
                    want_aggregate_share_job.aggregation_parameter(),
                )
                .await
                .unwrap()
                .is_none());

            let want_aggregate_share_jobs = Vec::from([want_aggregate_share_job.clone()]);

            let got_aggregate_share_jobs = tx
                .get_aggregate_share_jobs_by_batch_id::<0, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    want_aggregate_share_job.batch_id(),
                )
                .await
                .unwrap();
            assert_eq!(got_aggregate_share_jobs, want_aggregate_share_jobs);

            Ok(())
        })
    })
    .await
    .unwrap();

    // Advance the clock to expire all written entities.
    clock.advance(&REPORT_EXPIRY_AGE);

    ds.run_unnamed_tx(|tx| {
        let want_aggregate_share_job = aggregate_share_job.clone();
        Box::pin(async move {
            let vdaf = dummy::Vdaf::default();

            assert_eq!(
                tx.get_aggregate_share_job::<0, LeaderSelected, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    want_aggregate_share_job.batch_id(),
                    want_aggregate_share_job.aggregation_parameter(),
                )
                .await
                .unwrap(),
                None
            );

            assert_eq!(
                tx.get_aggregate_share_jobs_by_batch_id::<0, dummy::Vdaf>(
                    &vdaf,
                    want_aggregate_share_job.task_id(),
                    want_aggregate_share_job.batch_id(),
                )
                .await
                .unwrap(),
                Vec::new()
            );

            Ok(())
        })
    })
    .await
    .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_outstanding_batch(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    let batch_time_window_size = Duration::from_hours(24).unwrap();
    let time_bucket_start = clock
        .now()
        .to_batch_interval_start(&batch_time_window_size)
        .unwrap();

    let (task_id_1, batch_id_1, task_id_2, batch_id_2) = ds
        .run_tx("test-put-outstanding-batches", |tx| {
            let clock = clock.clone();
            Box::pin(async move {
                let task_1 = TaskBuilder::new(
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
                tx.put_aggregator_task(&task_1).await.unwrap();
                let batch_id_1 = random();
                let report_1 =
                    LeaderStoredReport::new_dummy(*task_1.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);

                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task_1.id(),
                    batch_id_1,
                    dummy::AggregationParam(0),
                    0,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(dummy::AggregateShare(0)),
                        report_count: 1,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 3,
                        aggregation_jobs_terminated: 2,
                    },
                ))
                .await
                .unwrap();
                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task_1.id(),
                    batch_id_1,
                    dummy::AggregationParam(0),
                    1,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(dummy::AggregateShare(0)),
                        report_count: 1,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 4,
                        aggregation_jobs_terminated: 1,
                    },
                ))
                .await
                .unwrap();
                tx.put_outstanding_batch(task_1.id(), &batch_id_1, &None)
                    .await
                    .unwrap();

                let task_2 = TaskBuilder::new(
                    task::BatchMode::LeaderSelected {
                        batch_time_window_size: Some(batch_time_window_size),
                    },
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .leader_view()
                .unwrap();
                tx.put_aggregator_task(&task_2).await.unwrap();
                let batch_id_2 = random();
                let report_2 =
                    LeaderStoredReport::new_dummy(*task_2.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);

                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task_2.id(),
                    batch_id_2,
                    dummy::AggregationParam(0),
                    0,
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(dummy::AggregateShare(0)),
                        // Let report_count be 1 without an accompanying report_aggregation in a
                        // terminal state. This captures the case where a FINISHED report_aggregation
                        // was garbage collected and no longer exists in the database.
                        report_count: 1,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 4,
                        aggregation_jobs_terminated: 1,
                    },
                ))
                .await
                .unwrap();
                tx.put_outstanding_batch(task_2.id(), &batch_id_2, &Some(time_bucket_start))
                    .await
                    .unwrap();

                // Write a few aggregation jobs & report aggregations to produce useful
                // min_size/max_size values to validate later.
                let aggregation_job_0 = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task_1.id(),
                    random(),
                    dummy::AggregationParam(0),
                    batch_id_1,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Finished,
                    AggregationJobStep::from(1),
                );
                let report_aggregation_0_0 =
                    report_1.as_leader_init_report_aggregation(*aggregation_job_0.id(), 0);

                let report_id_0_1 = random();
                let transcript = run_vdaf(
                    &dummy::Vdaf::default(),
                    task_1.id(),
                    task_1.vdaf_verify_key().unwrap().as_bytes(),
                    &dummy::AggregationParam(0),
                    &report_id_0_1,
                    &0,
                );

                let report_aggregation_0_1 = ReportAggregation::<0, dummy::Vdaf>::new(
                    *task_1.id(),
                    *aggregation_job_0.id(),
                    report_id_0_1,
                    clock.now(),
                    1,
                    None,
                    // Counted among max_size.
                    ReportAggregationState::LeaderContinue {
                        transition: transcript.helper_prepare_transitions[0].transition.clone(),
                    },
                );
                let report_aggregation_0_2 = ReportAggregation::<0, dummy::Vdaf>::new(
                    *task_1.id(),
                    *aggregation_job_0.id(),
                    random(),
                    clock.now(),
                    2,
                    None,
                    ReportAggregationState::Failed {
                        report_error: ReportError::VdafPrepError,
                    }, // Not counted among min_size or max_size.
                );

                let aggregation_job_1 = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task_1.id(),
                    random(),
                    dummy::AggregationParam(0),
                    batch_id_1,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Finished,
                    AggregationJobStep::from(1),
                );
                let report_aggregation_1_0 = ReportAggregation::<0, dummy::Vdaf>::new(
                    *task_1.id(),
                    *aggregation_job_1.id(),
                    random(),
                    clock.now(),
                    0,
                    None,
                    ReportAggregationState::Finished, // Counted among min_size and max_size.
                );
                let report_aggregation_1_1 = ReportAggregation::<0, dummy::Vdaf>::new(
                    *task_1.id(),
                    *aggregation_job_1.id(),
                    random(),
                    clock.now(),
                    1,
                    None,
                    ReportAggregationState::Finished, // Counted among min_size and max_size.
                );
                let report_aggregation_1_2 = ReportAggregation::<0, dummy::Vdaf>::new(
                    *task_1.id(),
                    *aggregation_job_1.id(),
                    random(),
                    clock.now(),
                    2,
                    None,
                    ReportAggregationState::Failed {
                        report_error: ReportError::VdafPrepError,
                    }, // Not counted among min_size or max_size.
                );

                let aggregation_job_2 = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task_2.id(),
                    random(),
                    dummy::AggregationParam(0),
                    batch_id_2,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Finished,
                    AggregationJobStep::from(1),
                );
                let report_aggregation_2_0 =
                    report_2.as_leader_init_report_aggregation(*aggregation_job_2.id(), 0);

                for aggregation_job in &[aggregation_job_0, aggregation_job_1, aggregation_job_2] {
                    tx.put_aggregation_job(aggregation_job).await.unwrap();
                }
                for report_aggregation in &[
                    report_aggregation_0_0,
                    report_aggregation_0_1,
                    report_aggregation_0_2,
                    report_aggregation_1_0,
                    report_aggregation_1_1,
                    report_aggregation_1_2,
                    report_aggregation_2_0,
                ] {
                    tx.put_client_report(&LeaderStoredReport::<0, dummy::Vdaf>::new(
                        *report_aggregation.task_id(),
                        ReportMetadata::new(
                            *report_aggregation.report_id(),
                            *report_aggregation.time(),
                            Vec::new(),
                        ),
                        (), // Dummy public share
                        Vec::new(),
                        dummy::InputShare::default(), // Dummy leader input share
                        // Dummy helper encrypted input share
                        HpkeCiphertext::new(
                            HpkeConfigId::from(13),
                            Vec::from("encapsulated_context_0"),
                            Vec::from("payload_0"),
                        ),
                    ))
                    .await
                    .unwrap();
                    tx.put_report_aggregation(report_aggregation).await.unwrap();
                }

                tx.check_timestamp_columns(
                    "outstanding_batches",
                    "test-put-outstanding-batches",
                    false,
                )
                .await;

                Ok((*task_1.id(), batch_id_1, *task_2.id(), batch_id_2))
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    let (
        outstanding_batches_task_1,
        outstanding_batches_task_1_after_mark,
        outstanding_batch_1,
        outstanding_batch_2,
        outstanding_batch_3,
        outstanding_batches_task_2,
        outstanding_batches_task_2_after_mark,
        outstanding_batches_empty_time_bucket,
    ) = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let outstanding_batches_task_1 = tx
                    .get_unfilled_outstanding_batches(&task_id_1, &None)
                    .await
                    .unwrap();
                tx.mark_outstanding_batch_filled(&task_id_1, &batch_id_1)
                    .await
                    .unwrap();
                let outstanding_batches_task_1_after_mark = tx
                    .get_unfilled_outstanding_batches(&task_id_1, &None)
                    .await
                    .unwrap();

                let outstanding_batch_1 = tx
                    .acquire_outstanding_batch_with_report_count(&task_id_1, 3)
                    .await
                    .unwrap();
                let outstanding_batch_2 = tx
                    .acquire_outstanding_batch_with_report_count(&task_id_1, 2)
                    .await
                    .unwrap();
                let outstanding_batch_3 = tx
                    .acquire_outstanding_batch_with_report_count(&task_id_1, 1)
                    .await
                    .unwrap();

                let outstanding_batches_task_2 = tx
                    .get_unfilled_outstanding_batches(&task_id_2, &Some(time_bucket_start))
                    .await
                    .unwrap();
                tx.mark_outstanding_batch_filled(&task_id_2, &batch_id_2)
                    .await
                    .unwrap();
                let outstanding_batches_task_2_after_mark = tx
                    .get_unfilled_outstanding_batches(&task_id_2, &Some(time_bucket_start))
                    .await
                    .unwrap();

                let outstanding_batches_empty_time_bucket = tx
                    .get_unfilled_outstanding_batches(
                        &task_id_2,
                        &Some(
                            time_bucket_start
                                .add(&Duration::from_hours(24).unwrap())
                                .unwrap(),
                        ),
                    )
                    .await
                    .unwrap();
                Ok((
                    outstanding_batches_task_1,
                    outstanding_batches_task_1_after_mark,
                    outstanding_batch_1,
                    outstanding_batch_2,
                    outstanding_batch_3,
                    outstanding_batches_task_2,
                    outstanding_batches_task_2_after_mark,
                    outstanding_batches_empty_time_bucket,
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(
        outstanding_batches_task_1,
        Vec::from([OutstandingBatch::new(
            task_id_1,
            batch_id_1,
            RangeInclusive::new(2, 4)
        )])
    );
    assert_eq!(outstanding_batches_task_1_after_mark, Vec::new());
    assert_eq!(outstanding_batch_1, None); // min_report_count too large
    assert_eq!(outstanding_batch_2, Some(batch_id_1));
    assert_eq!(outstanding_batch_3, None); // already retrieved
    assert_eq!(
        outstanding_batches_task_2,
        Vec::from([OutstandingBatch::new(
            task_id_2,
            batch_id_2,
            RangeInclusive::new(1, 2)
        )])
    );
    assert_eq!(outstanding_batches_task_2_after_mark, Vec::new());
    assert_eq!(outstanding_batches_empty_time_bucket, Vec::new());

    // Advance the clock further to trigger expiration of the written batches.
    clock.advance(&REPORT_EXPIRY_AGE);

    // Verify that the batch is no longer available.
    let outstanding_batches = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move { tx.get_unfilled_outstanding_batches(&task_id_1, &None).await })
        })
        .await
        .unwrap();
    assert!(outstanding_batches.is_empty());
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn delete_expired_client_reports(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;
    let vdaf = dummy::Vdaf::default();

    // Setup.
    let report_expiry_age = clock
        .now()
        .difference(&OLDEST_ALLOWED_REPORT_TIMESTAMP)
        .unwrap();
    let (task_id, new_report_id, other_task_id, other_task_report_id) = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(report_expiry_age))
                .build()
                .leader_view()
                .unwrap();
                let other_task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap();
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_aggregator_task(&other_task).await.unwrap();

                let old_report = LeaderStoredReport::new_dummy(
                    *task.id(),
                    OLDEST_ALLOWED_REPORT_TIMESTAMP
                        .sub(&Duration::from_seconds(1))
                        .unwrap(),
                );
                let new_report =
                    LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
                let other_task_report = LeaderStoredReport::new_dummy(
                    *other_task.id(),
                    OLDEST_ALLOWED_REPORT_TIMESTAMP
                        .sub(&Duration::from_seconds(1))
                        .unwrap(),
                );
                tx.put_client_report::<0, dummy::Vdaf>(&old_report)
                    .await
                    .unwrap();
                tx.put_client_report(&new_report).await.unwrap();
                tx.put_client_report(&other_task_report).await.unwrap();

                Ok((
                    *task.id(),
                    *new_report.metadata().id(),
                    *other_task.id(),
                    *other_task_report.metadata().id(),
                ))
            })
        })
        .await
        .unwrap();

    // Run.
    let deleted_report_count = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.delete_expired_client_reports(&task_id, u64::try_from(i64::MAX).unwrap())
                    .await
            })
        })
        .await
        .unwrap();

    // Verify.
    assert_eq!(1, deleted_report_count);
    let want_report_ids = HashSet::from([new_report_id, other_task_report_id]);
    let got_report_ids = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            Box::pin(async move {
                let task_client_reports = tx
                    .get_client_reports_for_task(&vdaf, &task_id)
                    .await
                    .unwrap();
                let other_task_client_reports = tx
                    .get_client_reports_for_task(&vdaf, &other_task_id)
                    .await
                    .unwrap();
                Ok(HashSet::from_iter(
                    task_client_reports
                        .into_iter()
                        .chain(other_task_client_reports)
                        .map(|report| *report.metadata().id()),
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(want_report_ids, got_report_ids);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn delete_expired_client_reports_noop(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::default();
    let ds = ephemeral_datastore.datastore(clock.clone()).await;
    let vdaf = dummy::Vdaf::default();

    // Setup.
    let (task_id, new_report_id, old_report_id) = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(None)
                .build()
                .leader_view()
                .unwrap();
                tx.put_aggregator_task(&task).await.unwrap();

                let old_report = LeaderStoredReport::new_dummy(
                    *task.id(),
                    OLDEST_ALLOWED_REPORT_TIMESTAMP
                        .sub(&Duration::from_seconds(1))
                        .unwrap(),
                );
                let new_report =
                    LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
                tx.put_client_report(&old_report).await.unwrap();
                tx.put_client_report(&new_report).await.unwrap();

                Ok((
                    *task.id(),
                    *new_report.metadata().id(),
                    *old_report.metadata().id(),
                ))
            })
        })
        .await
        .unwrap();

    // Run.
    let deleted_report_count = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.delete_expired_client_reports(&task_id, u64::try_from(i64::MAX).unwrap())
                    .await
            })
        })
        .await
        .unwrap();

    // Verify.
    assert_eq!(0, deleted_report_count);
    let want_report_ids = HashSet::from([new_report_id, old_report_id]);
    let got_report_ids = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            Box::pin(async move {
                let task_client_reports = tx
                    .get_client_reports_for_task(&vdaf, &task_id)
                    .await
                    .unwrap();
                Ok(HashSet::from_iter(
                    task_client_reports
                        .into_iter()
                        .map(|report| *report.metadata().id()),
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(want_report_ids, got_report_ids);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn delete_expired_aggregation_artifacts(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;
    let vdaf = dummy::Vdaf::default();
    let aggregation_param = dummy::AggregationParam(0);

    // Setup.
    async fn write_aggregation_artifacts<B: TestBatchModeExt>(
        tx: &Transaction<'_, MockClock>,
        task_id: &TaskId,
        aggregation_param: &dummy::AggregationParam,
        client_timestamps: &[Time],
    ) -> (
        B::BatchIdentifier,
        AggregationJobId, // aggregation job ID
        Vec<ReportId>,    // client report IDs
    ) {
        let batch_identifier = B::batch_identifier_for_client_timestamps(client_timestamps);

        let mut reports = Vec::new();
        for client_timestamp in client_timestamps {
            let report = LeaderStoredReport::new_dummy(*task_id, *client_timestamp);
            tx.put_client_report(&report).await.unwrap();
            reports.push(report);
        }

        let min_client_timestamp = client_timestamps.iter().min().unwrap();
        let max_client_timestamp = client_timestamps.iter().max().unwrap();
        let client_timestamp_interval = Interval::new(
            *min_client_timestamp,
            max_client_timestamp
                .difference(min_client_timestamp)
                .unwrap()
                .add(&Duration::from_seconds(1))
                .unwrap(),
        )
        .unwrap();

        let aggregation_job = AggregationJob::<0, B, dummy::Vdaf>::new(
            *task_id,
            random(),
            *aggregation_param,
            B::partial_batch_identifier(&batch_identifier).clone(),
            client_timestamp_interval,
            AggregationJobState::Active,
            AggregationJobStep::from(0),
        );
        tx.put_aggregation_job(&aggregation_job).await.unwrap();

        for (ord, report) in reports.iter().enumerate() {
            let report_aggregation = report
                .as_leader_init_report_aggregation(*aggregation_job.id(), ord.try_into().unwrap());
            tx.put_report_aggregation(&report_aggregation)
                .await
                .unwrap();
        }

        (
            batch_identifier,
            *aggregation_job.id(),
            reports
                .into_iter()
                .map(|report| *report.metadata().id())
                .collect(),
        )
    }

    let (
        leader_time_interval_task_id,
        helper_time_interval_task_id,
        leader_leader_selected_task_id,
        helper_leader_selected_task_id,
        want_aggregation_job_ids,
        want_report_ids,
    ) = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let leader_time_interval_task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .leader_view()
                .unwrap();
                let helper_time_interval_task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .helper_view()
                .unwrap();
                let leader_leader_selected_task = TaskBuilder::new(
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
                let helper_leader_selected_task = TaskBuilder::new(
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
                tx.put_aggregator_task(&leader_time_interval_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&helper_time_interval_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&leader_leader_selected_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&helper_leader_selected_task)
                    .await
                    .unwrap();

                let mut aggregation_job_ids = HashSet::new();
                let mut all_report_ids = HashSet::new();

                // Leader, time-interval aggregation job with old reports [GC'ed].
                write_aggregation_artifacts::<TimeInterval>(
                    tx,
                    leader_time_interval_task.id(),
                    &aggregation_param,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(20))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(19))
                            .unwrap(),
                    ],
                )
                .await;

                // Leader, time-interval aggregation job with old & new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<TimeInterval>(
                        tx,
                        leader_time_interval_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(5))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(8))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Leader, time-interval aggregation job with new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<TimeInterval>(
                        tx,
                        leader_time_interval_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(19))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(20))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Helper, time-interval aggregation job with old reports [GC'ed].
                write_aggregation_artifacts::<TimeInterval>(
                    tx,
                    helper_time_interval_task.id(),
                    &aggregation_param,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(20))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(19))
                            .unwrap(),
                    ],
                )
                .await;

                // Helper, time-interval task with old & new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<TimeInterval>(
                        tx,
                        helper_time_interval_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(5))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(8))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Helper, time-interval task with new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<TimeInterval>(
                        tx,
                        helper_time_interval_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(19))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(20))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Leader, leader-selected aggregation job with old reports [GC'ed].
                write_aggregation_artifacts::<LeaderSelected>(
                    tx,
                    leader_leader_selected_task.id(),
                    &aggregation_param,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(20))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(19))
                            .unwrap(),
                    ],
                )
                .await;

                // Leader, leader-selected aggregation job with old & new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<LeaderSelected>(
                        tx,
                        leader_leader_selected_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(5))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(8))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Leader, leader-selected aggregation job with new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<LeaderSelected>(
                        tx,
                        leader_leader_selected_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(19))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(20))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Helper, leader-selected aggregation job with old reports [GC'ed].
                write_aggregation_artifacts::<LeaderSelected>(
                    tx,
                    helper_leader_selected_task.id(),
                    &aggregation_param,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(20))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(19))
                            .unwrap(),
                    ],
                )
                .await;

                // Helper, leader-selected aggregation job with old & new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<LeaderSelected>(
                        tx,
                        helper_leader_selected_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(5))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(8))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                // Helper, leader-selected aggregation job with new reports [not GC'ed].
                let (_, aggregation_job_id, report_ids) =
                    write_aggregation_artifacts::<LeaderSelected>(
                        tx,
                        helper_leader_selected_task.id(),
                        &aggregation_param,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(19))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(20))
                                .unwrap(),
                        ],
                    )
                    .await;
                aggregation_job_ids.insert(aggregation_job_id);
                all_report_ids.extend(report_ids);

                Ok((
                    *leader_time_interval_task.id(),
                    *helper_time_interval_task.id(),
                    *leader_leader_selected_task.id(),
                    *helper_leader_selected_task.id(),
                    aggregation_job_ids,
                    all_report_ids,
                ))
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    // Run.
    let deleted_aggregation_job_counts = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                try_join!(
                    tx.delete_expired_aggregation_artifacts(
                        &leader_time_interval_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_aggregation_artifacts(
                        &helper_time_interval_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_aggregation_artifacts(
                        &leader_leader_selected_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_aggregation_artifacts(
                        &helper_leader_selected_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    )
                )
            })
        })
        .await
        .unwrap();

    // Verify.
    assert_eq!((1, 1, 1, 1), deleted_aggregation_job_counts);
    let (got_aggregation_job_ids, got_report_ids) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            Box::pin(async move {
                let leader_time_interval_aggregation_job_ids = tx
                    .get_aggregation_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &leader_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| *job.id());
                let helper_time_interval_aggregation_job_ids = tx
                    .get_aggregation_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &helper_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| *job.id());
                let leader_leader_selected_aggregation_job_ids = tx
                    .get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &leader_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| *job.id());
                let helper_leader_selected_aggregation_job_ids = tx
                    .get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &helper_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| *job.id());
                let got_aggregation_job_ids = leader_time_interval_aggregation_job_ids
                    .chain(helper_time_interval_aggregation_job_ids)
                    .chain(leader_leader_selected_aggregation_job_ids)
                    .chain(helper_leader_selected_aggregation_job_ids)
                    .collect();

                let leader_time_interval_report_aggregations = tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Leader,
                        &leader_time_interval_task_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                let helper_time_interval_report_aggregations = tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Helper,
                        &helper_time_interval_task_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                let leader_leader_selected_report_aggregations = tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Leader,
                        &leader_leader_selected_task_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                let helper_leader_selected_report_aggregations = tx
                    .get_report_aggregations_for_task::<0, dummy::Vdaf>(
                        &vdaf,
                        &Role::Helper,
                        &helper_leader_selected_task_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                let got_report_ids = leader_time_interval_report_aggregations
                    .into_iter()
                    .chain(helper_time_interval_report_aggregations)
                    .chain(leader_leader_selected_report_aggregations)
                    .chain(helper_leader_selected_report_aggregations)
                    .map(|report_aggregation| *report_aggregation.report_id())
                    .collect();

                Ok((got_aggregation_job_ids, got_report_ids))
            })
        })
        .await
        .unwrap();
    assert_eq!(want_aggregation_job_ids, got_aggregation_job_ids);
    assert_eq!(want_report_ids, got_report_ids);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn delete_expired_collection_artifacts(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();

    let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
    let ds = ephemeral_datastore.datastore(clock.clone()).await;

    // Setup.
    async fn write_collect_artifacts<B: TestBatchModeExt>(
        tx: &Transaction<'_, MockClock>,
        task: &AggregatorTask,
        client_timestamps: &[Time],
    ) -> (
        Option<CollectionJobId>,   // collection job ID
        Option<(TaskId, Vec<u8>)>, // aggregate share job ID (task ID, encoded batch identifier)
        Option<(TaskId, Vec<u8>)>, // batch ID (task ID, encoded batch identifier)
        Option<(TaskId, BatchId)>, // outstanding batch ID
        Option<(TaskId, Vec<u8>)>, // batch aggregation ID (task ID, encoded batch identifier)
        Option<Time>,              // time bucket start
    ) {
        let batch_identifier = B::batch_identifier_for_client_timestamps(client_timestamps);
        let client_timestamp_interval = client_timestamps
            .iter()
            .fold(Interval::EMPTY, |left, right| {
                left.merged_with(right).unwrap()
            });

        let batch_aggregation = BatchAggregation::<0, B, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier.clone(),
            dummy::AggregationParam(0),
            0,
            client_timestamp_interval,
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 3,
                aggregation_jobs_terminated: 2,
            },
        );
        tx.put_batch_aggregation(&batch_aggregation).await.unwrap();
        for ord in 1..8 {
            let batch_aggregation = BatchAggregation::<0, B, dummy::Vdaf>::new(
                *task.id(),
                batch_identifier.clone(),
                dummy::AggregationParam(0),
                ord,
                client_timestamp_interval,
                BatchAggregationState::Aggregating {
                    aggregate_share: None,
                    report_count: 0,
                    checksum: ReportIdChecksum::default(),
                    aggregation_jobs_created: 0,
                    aggregation_jobs_terminated: 0,
                },
            );
            tx.put_batch_aggregation(&batch_aggregation).await.unwrap();
        }

        if task.role() == &Role::Leader {
            let collection_job = CollectionJob::<0, B, dummy::Vdaf>::new(
                *task.id(),
                random(),
                B::query_for_batch_identifier(&batch_identifier),
                dummy::AggregationParam(0),
                batch_identifier.clone(),
                CollectionJobState::Start,
            );
            tx.put_collection_job(&collection_job).await.unwrap();

            let time_bucket_start = match task.batch_mode() {
                task::BatchMode::TimeInterval
                | task::BatchMode::LeaderSelected {
                    batch_time_window_size: None,
                    ..
                } => None,
                task::BatchMode::LeaderSelected {
                    batch_time_window_size: Some(batch_time_window_size),
                    ..
                } => {
                    let time_bucket_start = client_timestamps[0]
                        .to_batch_interval_start(batch_time_window_size)
                        .unwrap();
                    let same_bucket = client_timestamps.iter().all(|ts| {
                        ts.to_batch_interval_start(batch_time_window_size).unwrap()
                            == time_bucket_start
                    });
                    assert!(
                        same_bucket,
                        "client timestamps do not all fall in the same time bucket"
                    );
                    Some(time_bucket_start)
                }
            };

            let outstanding_batch_id =
                B::write_outstanding_batch(tx, task.id(), &batch_identifier, &time_bucket_start)
                    .await;

            (
                Some(*collection_job.id()),
                None,
                Some((*task.id(), batch_identifier.get_encoded().unwrap())),
                outstanding_batch_id,
                Some((*task.id(), batch_identifier.get_encoded().unwrap())),
                time_bucket_start,
            )
        } else {
            tx.put_aggregate_share_job::<0, B, dummy::Vdaf>(&AggregateShareJob::new(
                *task.id(),
                batch_identifier.clone(),
                dummy::AggregationParam(0),
                dummy::AggregateShare(11),
                client_timestamps.len().try_into().unwrap(),
                random(),
            ))
            .await
            .unwrap();

            (
                None,
                Some((*task.id(), batch_identifier.get_encoded().unwrap())),
                Some((*task.id(), batch_identifier.get_encoded().unwrap())),
                None,
                Some((*task.id(), batch_identifier.get_encoded().unwrap())),
                None,
            )
        }
    }

    let (
        leader_time_interval_task_id,
        helper_time_interval_task_id,
        leader_leader_selected_task_id,
        helper_leader_selected_task_id,
        leader_leader_selected_time_bucketed_task_id,
        other_task_id,
        want_collection_job_ids,
        want_aggregate_share_job_ids,
        want_outstanding_batch_ids,
        want_batch_aggregation_ids,
        time_bucket_starts,
    ) = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let leader_time_interval_task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .leader_view()
                .unwrap();
                let helper_time_interval_task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .helper_view()
                .unwrap();
                let leader_leader_selected_task = TaskBuilder::new(
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
                let helper_leader_selected_task = TaskBuilder::new(
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
                let leader_leader_selected_time_bucketed_task = TaskBuilder::new(
                    task::BatchMode::LeaderSelected {
                        batch_time_window_size: Some(Duration::from_hours(24).unwrap()),
                    },
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .leader_view()
                .unwrap();
                let other_task = TaskBuilder::new(
                    task::BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                .build()
                .leader_view()
                .unwrap();

                tx.put_aggregator_task(&leader_time_interval_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&helper_time_interval_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&leader_leader_selected_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&helper_leader_selected_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&leader_leader_selected_time_bucketed_task)
                    .await
                    .unwrap();
                tx.put_aggregator_task(&other_task).await.unwrap();

                let mut collection_job_ids = HashSet::new();
                let mut aggregate_share_job_ids = HashSet::new();
                let mut batch_ids = HashSet::new();
                let mut outstanding_batch_ids = HashSet::new();
                let mut batch_aggregation_ids = HashSet::new();
                let mut time_bucket_starts = HashSet::new();

                // Leader, time-interval collection artifacts with old reports. [GC'ed]
                write_collect_artifacts::<TimeInterval>(
                    tx,
                    &leader_time_interval_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(10))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(9))
                            .unwrap(),
                    ],
                )
                .await;

                // Leader, time-interval collection artifacts with old & new reports. [collection job GC'ed, remainder not GC'ed]
                let (
                    _,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<TimeInterval>(
                    tx,
                    &leader_time_interval_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(5))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(5))
                            .unwrap(),
                    ],
                )
                .await;
                // collection_job_ids purposefully not changed.
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Leader, time-interval collection artifacts with new reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<TimeInterval>(
                    tx,
                    &leader_time_interval_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(9))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(10))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Helper, time-interval collection artifacts with old reports. [GC'ed]
                write_collect_artifacts::<TimeInterval>(
                    tx,
                    &helper_time_interval_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(10))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(9))
                            .unwrap(),
                    ],
                )
                .await;

                // Helper, time-interval collection artifacts with old & new reports. [aggregate share job job GC'ed, remainder not GC'ed]
                let (_, _, batch_id, outstanding_batch_id, batch_aggregation_id, _) =
                    write_collect_artifacts::<TimeInterval>(
                        tx,
                        &helper_time_interval_task,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(5))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .add(&Duration::from_seconds(5))
                                .unwrap(),
                        ],
                    )
                    .await;
                // collection_job_ids purposefully not changed.
                // aggregate_share_job_ids purposefully not changed.
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Helper, time-interval collection artifacts with new reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<TimeInterval>(
                    tx,
                    &helper_time_interval_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(9))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(10))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Leader, leader-selected collection artifacts with old reports. [GC'ed]
                write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &leader_leader_selected_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(10))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(9))
                            .unwrap(),
                    ],
                )
                .await;

                // Leader, leader-selected collection artifacts with old & new reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &leader_leader_selected_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(5))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(5))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Leader, leader-selected collection artifacts with new reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &leader_leader_selected_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(9))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(10))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Helper, leader-selected collection artifacts with old reports. [GC'ed]
                write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &helper_leader_selected_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(10))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(9))
                            .unwrap(),
                    ],
                )
                .await;

                // Helper, leader-selected collection artifacts with old & new reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &helper_leader_selected_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(5))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(5))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Helper, leader-selected collection artifacts with new reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &helper_leader_selected_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(9))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(10))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                // Leader, leader-selected time bucketed collection artifacts with old reports.
                // [GC'ed]
                let (_, _, _, _, _, time_bucket_start) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &leader_leader_selected_time_bucketed_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(10))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(9))
                            .unwrap(),
                    ],
                )
                .await;
                time_bucket_starts.extend(time_bucket_start);

                // Leader, leader-selected time bucketed collection artifacts with old and new
                // reports. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    time_bucket_start,
                ) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &leader_leader_selected_time_bucketed_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(5))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(5))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);
                time_bucket_starts.extend(time_bucket_start);

                // Leader, leader-selected time bucketed collection artifacts with new reports [not
                // GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    time_bucket_start,
                ) = write_collect_artifacts::<LeaderSelected>(
                    tx,
                    &leader_leader_selected_time_bucketed_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(9))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(10))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);
                time_bucket_starts.extend(time_bucket_start);

                // Collection artifacts for different task. [not GC'ed]
                let (
                    collection_job_id,
                    aggregate_share_job_id,
                    batch_id,
                    outstanding_batch_id,
                    batch_aggregation_id,
                    _,
                ) = write_collect_artifacts::<TimeInterval>(
                    tx,
                    &other_task,
                    &[
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(9))
                            .unwrap(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(8))
                            .unwrap(),
                    ],
                )
                .await;
                collection_job_ids.extend(collection_job_id);
                aggregate_share_job_ids.extend(aggregate_share_job_id);
                batch_ids.extend(batch_id);
                outstanding_batch_ids.extend(outstanding_batch_id);
                batch_aggregation_ids.extend(batch_aggregation_id);

                Ok((
                    *leader_time_interval_task.id(),
                    *helper_time_interval_task.id(),
                    *leader_leader_selected_task.id(),
                    *helper_leader_selected_task.id(),
                    *leader_leader_selected_time_bucketed_task.id(),
                    *other_task.id(),
                    collection_job_ids,
                    aggregate_share_job_ids,
                    outstanding_batch_ids,
                    batch_aggregation_ids,
                    time_bucket_starts,
                ))
            })
        })
        .await
        .unwrap();

    // Advance the clock to "enable" report expiry.
    clock.advance(&REPORT_EXPIRY_AGE);

    // Run.
    let deleted_batch_counts = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                try_join!(
                    tx.delete_expired_collection_artifacts(
                        &leader_time_interval_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_collection_artifacts(
                        &helper_time_interval_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_collection_artifacts(
                        &leader_leader_selected_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_collection_artifacts(
                        &helper_leader_selected_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    ),
                    tx.delete_expired_collection_artifacts(
                        &leader_leader_selected_time_bucketed_task_id,
                        u64::try_from(i64::MAX).unwrap(),
                    )
                )
            })
        })
        .await
        .unwrap();

    // Reset the clock to "disable" GC-on-read.
    clock.set(OLDEST_ALLOWED_REPORT_TIMESTAMP);

    // Verify.
    assert_eq!((1, 1, 1, 1, 1), deleted_batch_counts);
    let (
        got_collection_job_ids,
        got_aggregate_share_job_ids,
        got_outstanding_batch_ids,
        got_batch_aggregation_ids,
    ) = ds
        .run_unnamed_tx(|tx| {
            let time_bucket_starts = time_bucket_starts.clone();
            Box::pin(async move {
                let vdaf = dummy::Vdaf::default();

                let leader_time_interval_collection_job_ids = tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &leader_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|collection_job| *collection_job.id());
                let helper_time_interval_collection_job_ids = tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &helper_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|collection_job| *collection_job.id());
                let leader_leader_selected_collection_job_ids = tx
                    .get_collection_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &leader_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|collection_job| *collection_job.id());
                let helper_leader_selected_collection_job_ids = tx
                    .get_collection_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &helper_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|collection_job| *collection_job.id());
                let leader_leader_selected_time_bucketed_collection_job_ids = tx
                    .get_collection_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &leader_leader_selected_time_bucketed_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|collection_job| *collection_job.id());
                let other_task_collection_job_ids = tx
                    .get_collection_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &other_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|collection_job| *collection_job.id());
                let got_collection_job_ids = leader_time_interval_collection_job_ids
                    .chain(helper_time_interval_collection_job_ids)
                    .chain(leader_leader_selected_collection_job_ids)
                    .chain(helper_leader_selected_collection_job_ids)
                    .chain(leader_leader_selected_time_bucketed_collection_job_ids)
                    .chain(other_task_collection_job_ids)
                    .collect();

                let leader_time_interval_aggregate_share_job_ids = tx
                    .get_aggregate_share_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &leader_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| {
                        (
                            *job.task_id(),
                            job.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let helper_time_interval_aggregate_share_job_ids = tx
                    .get_aggregate_share_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &helper_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| {
                        (
                            *job.task_id(),
                            job.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let leader_leader_selected_aggregate_share_job_ids = tx
                    .get_aggregate_share_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &leader_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| {
                        (
                            *job.task_id(),
                            job.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let helper_leader_selected_aggregate_share_job_ids = tx
                    .get_aggregate_share_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &helper_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| {
                        (
                            *job.task_id(),
                            job.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let leader_leader_selected_time_bucketed_aggregate_share_job_ids = tx
                    .get_aggregate_share_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &leader_leader_selected_time_bucketed_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| {
                        (
                            *job.task_id(),
                            job.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let other_task_aggregate_share_job_ids = tx
                    .get_aggregate_share_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &other_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|job| {
                        (
                            *job.task_id(),
                            job.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let got_aggregate_share_job_ids = leader_time_interval_aggregate_share_job_ids
                    .chain(helper_time_interval_aggregate_share_job_ids)
                    .chain(leader_leader_selected_aggregate_share_job_ids)
                    .chain(helper_leader_selected_aggregate_share_job_ids)
                    .chain(leader_leader_selected_time_bucketed_aggregate_share_job_ids)
                    .chain(other_task_aggregate_share_job_ids)
                    .collect();

                let leader_time_interval_outstanding_batch_ids = tx
                    .get_unfilled_outstanding_batches(&leader_time_interval_task_id, &None)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|batch| (*batch.task_id(), *batch.id()));
                let helper_time_interval_outstanding_batch_ids = tx
                    .get_unfilled_outstanding_batches(&helper_time_interval_task_id, &None)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|batch| (*batch.task_id(), *batch.id()));
                let leader_leader_selected_outstanding_batch_ids = tx
                    .get_unfilled_outstanding_batches(&leader_leader_selected_task_id, &None)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|batch| (*batch.task_id(), *batch.id()));
                let helper_leader_selected_outstanding_batch_ids = tx
                    .get_unfilled_outstanding_batches(&helper_leader_selected_task_id, &None)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|batch| (*batch.task_id(), *batch.id()));
                let leader_leader_selected_time_bucketed_outstanding_batch_ids =
                    try_join_all(time_bucket_starts.iter().copied().map(
                        |time_bucket_start| async move {
                            tx.get_unfilled_outstanding_batches(
                                &leader_leader_selected_time_bucketed_task_id,
                                &Some(time_bucket_start),
                            )
                            .await
                        },
                    ))
                    .await
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .map(|batch| (*batch.task_id(), *batch.id()));
                let other_task_outstanding_batch_ids = tx
                    .get_unfilled_outstanding_batches(&other_task_id, &None)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|batch| (*batch.task_id(), *batch.id()));
                let got_outstanding_batch_ids = leader_time_interval_outstanding_batch_ids
                    .chain(helper_time_interval_outstanding_batch_ids)
                    .chain(leader_leader_selected_outstanding_batch_ids)
                    .chain(helper_leader_selected_outstanding_batch_ids)
                    .chain(leader_leader_selected_time_bucketed_outstanding_batch_ids)
                    .chain(other_task_outstanding_batch_ids)
                    .collect();

                let leader_time_interval_batch_aggregation_ids = tx
                    .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &leader_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|agg| {
                        (
                            *agg.task_id(),
                            agg.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let helper_time_interval_batch_aggregation_ids = tx
                    .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &helper_time_interval_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|agg| {
                        (
                            *agg.task_id(),
                            agg.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let leader_leader_selected_batch_aggregation_ids = tx
                    .get_batch_aggregations_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &leader_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|agg| {
                        (
                            *agg.task_id(),
                            agg.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let helper_leader_selected_batch_aggregation_ids = tx
                    .get_batch_aggregations_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &helper_leader_selected_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|agg| {
                        (
                            *agg.task_id(),
                            agg.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let leader_leader_selected_time_bucketed_batch_aggregation_ids = tx
                    .get_batch_aggregations_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        &leader_leader_selected_time_bucketed_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|agg| {
                        (
                            *agg.task_id(),
                            agg.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let other_task_batch_aggregation_ids = tx
                    .get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        &other_task_id,
                    )
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|agg| {
                        (
                            *agg.task_id(),
                            agg.batch_identifier().get_encoded().unwrap(),
                        )
                    });
                let got_batch_aggregation_ids = leader_time_interval_batch_aggregation_ids
                    .chain(helper_time_interval_batch_aggregation_ids)
                    .chain(leader_leader_selected_batch_aggregation_ids)
                    .chain(helper_leader_selected_batch_aggregation_ids)
                    .chain(leader_leader_selected_time_bucketed_batch_aggregation_ids)
                    .chain(other_task_batch_aggregation_ids)
                    .collect();

                Ok((
                    got_collection_job_ids,
                    got_aggregate_share_job_ids,
                    got_outstanding_batch_ids,
                    got_batch_aggregation_ids,
                ))
            })
        })
        .await
        .unwrap();
    assert_eq!(want_collection_job_ids, got_collection_job_ids);
    assert_eq!(want_aggregate_share_job_ids, got_aggregate_share_job_ids);
    assert_eq!(want_outstanding_batch_ids, got_outstanding_batch_ids);
    assert_eq!(want_batch_aggregation_ids, got_batch_aggregation_ids);
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_interval_sql(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let datastore = ephemeral_datastore.datastore(MockClock::default()).await;

    datastore
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let interval = tx
                    .query_one(
                        "SELECT '[2020-01-01 10:00, 2020-01-01 10:30)'::tsrange AS interval",
                        &[],
                    )
                    .await
                    .unwrap()
                    .get::<_, SqlInterval>("interval");
                let ref_interval = Interval::new(
                    Time::from_naive_date_time(
                        &NaiveDate::from_ymd_opt(2020, 1, 1)
                            .unwrap()
                            .and_hms_opt(10, 0, 0)
                            .unwrap(),
                    ),
                    Duration::from_minutes(30).unwrap(),
                )
                .unwrap();
                assert_eq!(interval.as_interval(), ref_interval);

                let interval = tx
                    .query_one(
                        "SELECT '[1970-02-03 23:00, 1970-02-04 00:00)'::tsrange AS interval",
                        &[],
                    )
                    .await
                    .unwrap()
                    .get::<_, SqlInterval>("interval");
                let ref_interval = Interval::new(
                    Time::from_naive_date_time(
                        &NaiveDate::from_ymd_opt(1970, 2, 3)
                            .unwrap()
                            .and_hms_opt(23, 0, 0)
                            .unwrap(),
                    ),
                    Duration::from_hours(1).unwrap(),
                )
                .unwrap();
                assert_eq!(interval.as_interval(), ref_interval);

                let res = tx
                    .query_one(
                        "SELECT '[1969-01-01 00:00, 1970-01-01 00:00)'::tsrange AS interval",
                        &[],
                    )
                    .await
                    .unwrap()
                    .try_get::<_, SqlInterval>("interval");
                assert!(res.is_err());

                let ok = tx
                    .query_one(
                        "--
SELECT (lower(interval) = '1972-07-21 05:30:00' AND
    upper(interval) = '1972-07-21 06:00:00' AND
    lower_inc(interval) AND
    NOT upper_inc(interval)) AS ok
    FROM (VALUES ($1::tsrange)) AS temp (interval)",
                        &[&SqlInterval::from(
                            Interval::new(
                                Time::from_naive_date_time(
                                    &NaiveDate::from_ymd_opt(1972, 7, 21)
                                        .unwrap()
                                        .and_hms_opt(5, 30, 0)
                                        .unwrap(),
                                ),
                                Duration::from_minutes(30).unwrap(),
                            )
                            .unwrap(),
                        )],
                    )
                    .await
                    .unwrap()
                    .get::<_, bool>("ok");
                assert!(ok);

                let ok = tx
                    .query_one(
                        "--
SELECT (lower(interval) = '2021-10-05 00:00:00' AND
    upper(interval) = '2021-10-06 00:00:00' AND
    lower_inc(interval) AND
    NOT upper_inc(interval)) AS ok
    FROM (VALUES ($1::tsrange)) AS temp (interval)",
                        &[&SqlInterval::from(
                            Interval::new(
                                Time::from_naive_date_time(
                                    &NaiveDate::from_ymd_opt(2021, 10, 5)
                                        .unwrap()
                                        .and_hms_opt(0, 0, 0)
                                        .unwrap(),
                                ),
                                Duration::from_hours(24).unwrap(),
                            )
                            .unwrap(),
                        )],
                    )
                    .await
                    .unwrap()
                    .get::<_, bool>("ok");
                assert!(ok);

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_hpke_keypair(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let datastore = ephemeral_datastore.datastore(MockClock::default()).await;
    let clock = datastore.clock.clone();
    let keypair = hpke::HpkeKeypair::test();

    datastore
        .run_tx("test-put-keys", |tx| {
            let keypair = keypair.clone();
            let clock = clock.clone();
            Box::pin(async move {
                assert_eq!(tx.get_hpke_keypairs().await.unwrap(), Vec::new());
                tx.put_hpke_keypair(&keypair).await.unwrap();

                let expected_keypair =
                    HpkeKeypair::new(keypair.clone(), HpkeKeyState::Pending, clock.now());
                assert_eq!(
                    tx.get_hpke_keypairs().await.unwrap(),
                    Vec::from([expected_keypair.clone()])
                );
                assert_eq!(
                    tx.get_hpke_keypair(keypair.config().id())
                        .await
                        .unwrap()
                        .unwrap(),
                    expected_keypair
                );

                // Try modifying state.
                clock.advance(&Duration::from_seconds(100));
                tx.set_hpke_keypair_state(keypair.config().id(), &HpkeKeyState::Active)
                    .await
                    .unwrap();
                assert_eq!(
                    tx.get_hpke_keypair(keypair.config().id())
                        .await
                        .unwrap()
                        .unwrap(),
                    HpkeKeypair::new(keypair.clone(), HpkeKeyState::Active, clock.now())
                );

                clock.advance(&Duration::from_seconds(100));
                tx.set_hpke_keypair_state(keypair.config().id(), &HpkeKeyState::Expired)
                    .await
                    .unwrap();
                assert_eq!(
                    tx.get_hpke_keypair(keypair.config().id())
                        .await
                        .unwrap()
                        .unwrap(),
                    HpkeKeypair::new(keypair.clone(), HpkeKeyState::Expired, clock.now())
                );

                Ok(())
            })
        })
        .await
        .unwrap();

    // Should not be able to set keypair with the same id.
    assert_matches!(
        datastore
            .run_unnamed_tx(|tx| {
                let keypair = keypair.clone();
                Box::pin(async move { tx.put_hpke_keypair(&keypair).await })
            })
            .await,
        Err(Error::Db(_))
    );

    datastore
        .run_unnamed_tx(|tx| {
            let keypair = keypair.clone();
            Box::pin(async move {
                tx.delete_hpke_keypair(keypair.config().id()).await.unwrap();
                assert_eq!(tx.get_hpke_keypairs().await.unwrap(), Vec::new());
                assert_matches!(
                    tx.get_hpke_keypair(keypair.config().id()).await.unwrap(),
                    None
                );

                tx.check_timestamp_columns("hpke_keys", "test-put-keys", true)
                    .await;

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_taskprov_peer_aggregator(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let datastore = ephemeral_datastore.datastore(MockClock::default()).await;

    // Basic aggregator.
    let example_leader_peer_aggregator = PeerAggregatorBuilder::new()
        .with_peer_role(Role::Leader)
        .build();
    let example_helper_peer_aggregator = PeerAggregatorBuilder::new()
        .with_peer_role(Role::Helper)
        .with_aggregator_auth_tokens(Vec::from([random(), random()]))
        .with_collector_auth_tokens(Vec::new())
        .build();
    let another_example_leader_peer_aggregator = PeerAggregatorBuilder::new()
        .with_endpoint(Url::parse("https://another.example.com/").unwrap())
        .with_aggregator_auth_tokens(Vec::new())
        .with_collector_auth_tokens(Vec::from([random(), random()]))
        .build();

    datastore
        .run_tx("test-put-peer-aggregator", |tx| {
            let example_leader_peer_aggregator = example_leader_peer_aggregator.clone();
            let example_helper_peer_aggregator = example_helper_peer_aggregator.clone();
            let another_example_leader_peer_aggregator =
                another_example_leader_peer_aggregator.clone();
            Box::pin(async move {
                tx.put_taskprov_peer_aggregator(&example_leader_peer_aggregator)
                    .await
                    .unwrap();
                tx.put_taskprov_peer_aggregator(&example_helper_peer_aggregator)
                    .await
                    .unwrap();
                tx.put_taskprov_peer_aggregator(&another_example_leader_peer_aggregator)
                    .await
                    .unwrap();
                Ok(())
            })
        })
        .await
        .unwrap();

    // Should not be able to put an aggregator with the same endpoint and role.
    assert_matches!(
        datastore
            .run_unnamed_tx(|tx| {
                Box::pin(async move {
                    let colliding_peer_aggregator = PeerAggregatorBuilder::new().build();
                    tx.put_taskprov_peer_aggregator(&colliding_peer_aggregator)
                        .await
                })
            })
            .await,
        Err(Error::MutationTargetAlreadyExists)
    );

    datastore
        .run_unnamed_tx(|tx| {
            let example_leader_peer_aggregator = example_leader_peer_aggregator.clone();
            let example_helper_peer_aggregator = example_helper_peer_aggregator.clone();
            let another_example_leader_peer_aggregator =
                another_example_leader_peer_aggregator.clone();
            Box::pin(async move {
                for peer_aggregator in [
                    example_leader_peer_aggregator.clone(),
                    example_helper_peer_aggregator.clone(),
                    another_example_leader_peer_aggregator.clone(),
                ] {
                    assert_eq!(
                        tx.get_taskprov_peer_aggregator(
                            peer_aggregator.endpoint(),
                            peer_aggregator.peer_role()
                        )
                        .await
                        .unwrap(),
                        Some(peer_aggregator.clone()),
                    );
                }

                assert_eq!(
                    tx.get_taskprov_peer_aggregators().await.unwrap(),
                    Vec::from([
                        example_leader_peer_aggregator.clone(),
                        example_helper_peer_aggregator.clone(),
                        another_example_leader_peer_aggregator.clone(),
                    ])
                );

                for peer in [
                    example_leader_peer_aggregator.clone(),
                    example_helper_peer_aggregator.clone(),
                    another_example_leader_peer_aggregator.clone(),
                ] {
                    tx.delete_taskprov_peer_aggregator(peer.endpoint(), peer.peer_role())
                        .await
                        .unwrap();
                }
                assert_eq!(
                    tx.get_taskprov_peer_aggregators().await.unwrap(),
                    Vec::new()
                );

                tx.check_timestamp_columns(
                    "taskprov_peer_aggregators",
                    "test-put-peer-aggregator",
                    false,
                )
                .await;

                tx.check_timestamp_columns(
                    "taskprov_aggregator_auth_tokens",
                    "test-put-peer-aggregator",
                    false,
                )
                .await;
                tx.check_timestamp_columns(
                    "taskprov_collector_auth_tokens",
                    "test-put-peer-aggregator",
                    false,
                )
                .await;

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn accept_write_expired_report(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let datastore = ephemeral_datastore.datastore(clock.clone()).await;

    let report_expiry_age = Duration::from_seconds(60);
    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_report_expiry_age(Some(report_expiry_age))
    .build()
    .leader_view()
    .unwrap();

    datastore.put_aggregator_task(&task).await.unwrap();

    // Use same ID for each report.
    let report = LeaderStoredReport::<0, dummy::Vdaf>::new(
        *task.id(),
        ReportMetadata::new(random(), clock.now(), Vec::new()),
        (),
        Vec::new(),
        dummy::InputShare::default(),
        HpkeCiphertext::new(
            HpkeConfigId::from(13),
            Vec::from("encapsulated_context_0"),
            Vec::from("payload_0"),
        ),
    );

    datastore
        .run_unnamed_tx(|tx| {
            let report = report.clone();

            Box::pin(async move {
                tx.put_client_report(&report).await.unwrap();
                Ok(())
            })
        })
        .await
        .unwrap();

    // Advance the clock well past the report expiry age.
    clock.advance(&report_expiry_age.add(&report_expiry_age).unwrap());

    // Validate that the report can't be read, that it can be written, and that even after writing
    // it still can't be read.
    datastore
        .run_unnamed_tx(|tx| {
            let report = report.clone();

            Box::pin(async move {
                assert!(tx
                    .get_client_report(
                        &dummy::Vdaf::default(),
                        report.task_id(),
                        report.metadata().id()
                    )
                    .await
                    .unwrap()
                    .is_none());

                tx.put_client_report(&report).await.unwrap();

                assert!(tx
                    .get_client_report(
                        &dummy::Vdaf::default(),
                        report.task_id(),
                        report.metadata().id()
                    )
                    .await
                    .unwrap()
                    .is_none());

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_task_upload_counter(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let datastore = ephemeral_datastore.datastore(clock.clone()).await;

    let task = TaskBuilder::new(
        task::BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .leader_view()
    .unwrap();

    datastore.put_aggregator_task(&task).await.unwrap();

    datastore
        .run_unnamed_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move {
                // Returns None for non-existent task.
                let counter = tx.get_task_upload_counter(&random()).await.unwrap();
                assert_eq!(counter, None);

                // Returns Some for a task that has just been created and has no counters.
                let counter = tx.get_task_upload_counter(&task_id).await.unwrap();
                assert_eq!(counter, Some(TaskUploadCounter::default()));

                let ord = thread_rng().gen_range(0..32);
                tx.increment_task_upload_counter(
                    &task_id,
                    ord,
                    &TaskUploadCounter::new_with_values(2, 4, 6, 8, 10, 100, 25, 22, 12),
                )
                .await
                .unwrap();

                let ord = thread_rng().gen_range(0..32);
                tx.increment_task_upload_counter(
                    &task_id,
                    ord,
                    &TaskUploadCounter::new_with_values(0, 0, 0, 0, 0, 0, 0, 0, 8),
                )
                .await
                .unwrap();

                let ord = thread_rng().gen_range(0..32);
                tx.increment_task_upload_counter(&task_id, ord, &TaskUploadCounter::default())
                    .await
                    .unwrap();

                let counter = tx.get_task_upload_counter(&task_id).await.unwrap();
                assert_eq!(
                    counter,
                    Some(TaskUploadCounter {
                        interval_collected: 2,
                        report_decode_failure: 4,
                        report_decrypt_failure: 6,
                        report_expired: 8,
                        report_outdated_key: 10,
                        report_success: 100,
                        report_too_early: 25,
                        task_not_started: 22,
                        task_ended: 20,
                    })
                );

                Ok(())
            })
        })
        .await
        .unwrap();
}

#[rstest_reuse::apply(schema_versions_template)]
#[tokio::test]
async fn roundtrip_task_aggregation_counter(ephemeral_datastore: EphemeralDatastore) {
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let datastore = ephemeral_datastore.datastore(clock.clone()).await;

    datastore
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                // Returns None for non-existent task.
                let counter = tx.get_task_aggregation_counter(&random()).await.unwrap();
                assert_eq!(counter, None);

                // Put a task for us to increment counters for.
                let task = TaskBuilder::new(
                    task::BatchMode::LeaderSelected {
                        batch_time_window_size: None,
                    },
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap();

                tx.put_aggregator_task(&task).await.unwrap();

                // Returns Some for a task that has just been created and has no counters.
                let counter = tx.get_task_aggregation_counter(task.id()).await.unwrap();
                assert_eq!(counter, Some(TaskAggregationCounter::default()));

                let ord = thread_rng().gen_range(0..32);
                tx.increment_task_aggregation_counter(
                    task.id(),
                    ord,
                    &TaskAggregationCounter { success: 4 },
                )
                .await
                .unwrap();

                let ord = thread_rng().gen_range(0..32);
                tx.increment_task_aggregation_counter(
                    task.id(),
                    ord,
                    &TaskAggregationCounter { success: 6 },
                )
                .await
                .unwrap();

                let ord = thread_rng().gen_range(0..32);
                tx.increment_task_aggregation_counter(
                    task.id(),
                    ord,
                    &TaskAggregationCounter::default(),
                )
                .await
                .unwrap();

                let counter = tx.get_task_aggregation_counter(task.id()).await.unwrap();
                assert_eq!(counter, Some(TaskAggregationCounter { success: 10 }));

                Ok(())
            })
        })
        .await
        .unwrap();
}
