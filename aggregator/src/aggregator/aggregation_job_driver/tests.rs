#![allow(clippy::unit_arg)] // allow reference to dummy::Vdaf's public share, which has the unit type

use crate::{
    aggregator::{
        aggregation_job_driver::AggregationJobDriver,
        test_util::{
            assert_task_aggregation_counter, generate_helper_report_share,
            BATCH_AGGREGATION_SHARD_COUNT, TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        },
        Error,
    },
    binary_utils::job_driver::JobDriver,
    cache::HpkeKeypairCache,
};
use assert_matches::assert_matches;
use futures::future::join_all;
use http::{header::CONTENT_TYPE, StatusCode};
use janus_aggregator_core::{
    batch_mode::{AccumulableBatchMode, CollectableBatchMode},
    datastore::{
        models::{
            merge_batch_aggregations_by_batch, AcquiredAggregationJob, AggregationJob,
            AggregationJobState, BatchAggregation, BatchAggregationState, LeaderStoredReport,
            Lease, ReportAggregation, ReportAggregationState, TaskAggregationCounter,
        },
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{test_util::TaskBuilder, AggregationMode, AggregatorTask, BatchMode, VerifyKey},
    test_util::noop_meter,
};
use janus_core::{
    hpke::HpkeKeypair,
    report_id::ReportIdChecksumExt,
    retries::test_util::LimitedRetryer,
    test_util::{install_test_trace_subscriber, run_vdaf, runtime::TestRuntimeManager},
    time::{Clock, IntervalExt, MockClock, TimeExt},
    vdaf::{VdafInstance, VERIFY_KEY_LENGTH_PRIO3},
    Runtime,
};
use janus_messages::{
    batch_mode::{LeaderSelected, TimeInterval},
    problem_type::DapProblemType,
    AggregationJobContinueReq, AggregationJobInitializeReq, AggregationJobResp, AggregationJobStep,
    Duration, Extension, ExtensionType, Interval, PartialBatchSelector, PrepareContinue,
    PrepareInit, PrepareResp, PrepareStepResult, ReportError, ReportIdChecksum, ReportMetadata,
    ReportShare, Role, Time,
};
use mockito::ServerGuard;
use prio::{
    codec::Encode,
    vdaf::{
        dummy,
        prio3::{Prio3, Prio3Count},
        Aggregator,
    },
};
use rand::random;
use std::{sync::Arc, time::Duration as StdDuration};
use tokio::time::timeout;
use trillium_tokio::Stopper;

const DEFAULT_ASYNC_POLL_INTERVAL: StdDuration = StdDuration::from_secs(1);

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
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );

    let agg_auth_token = task.aggregator_auth_token().clone();
    let helper_hpke_keypair = HpkeKeypair::test();
    let report = LeaderStoredReport::generate(
        *task.id(),
        report_metadata,
        helper_hpke_keypair.config(),
        Vec::new(),
        &transcript,
    );

    let aggregation_job_id = random();

    ds.run_unnamed_tx(|tx| {
        let (task, report, aggregation_param) =
            (leader_task.clone(), report.clone(), aggregation_param);
        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();
            tx.put_client_report(&report).await.unwrap();
            tx.scrub_client_report(report.task_id(), report.metadata().id())
                .await
                .unwrap();
            tx.mark_report_aggregated(task.id(), report.metadata().id())
                .await
                .unwrap();

            tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                aggregation_job_id,
                aggregation_param,
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
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

            tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                *task.id(),
                batch_identifier,
                aggregation_param,
                0,
                Interval::from_time(&time).unwrap(),
                BatchAggregationState::Aggregating {
                    aggregate_share: None,
                    report_count: 0,
                    checksum: ReportIdChecksum::default(),
                    aggregation_jobs_created: 1,
                    aggregation_jobs_terminated: 0,
                },
            ))
            .await
            .unwrap();

            Ok(())
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
            AggregationJobResp::Finished {
                prepare_resps: Vec::from([PrepareResp::new(
                    *report.metadata().id(),
                    PrepareStepResult::Continue {
                        message: transcript.helper_prepare_transitions[0].message.clone(),
                    },
                )]),
            }
            .get_encoded()
            .unwrap(),
        ),
        (
            "POST",
            AggregationJobContinueReq::MEDIA_TYPE,
            AggregationJobResp::MEDIA_TYPE,
            AggregationJobResp::Finished {
                prepare_resps: Vec::from([PrepareResp::new(
                    *report.metadata().id(),
                    PrepareStepResult::Finished,
                )]),
            }
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
                    task.aggregation_job_uri(&aggregation_job_id, None)
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
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
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
    timeout(
        StdDuration::from_secs(30),
        runtime_manager.wait_for_completed_tasks("stepper", 2),
    )
    .await
    .unwrap();
    // Stop the aggregation job driver.
    stopper.stop();
    // Wait for the aggregation job driver task to complete.
    task_handle.await.unwrap();

    // Verify.
    for mocked_aggregate in mocked_aggregates {
        mocked_aggregate.assert_async().await;
    }

    let want_aggregation_job: AggregationJob<0, TimeInterval, _> =
        AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            aggregation_job_id,
            aggregation_param,
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(2),
        );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );
    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(transcript.leader_aggregate_share),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );
                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn leader_sync_time_interval_aggregation_job_init_single_step() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(Prio3::new_count(2).unwrap());

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<VERIFY_KEY_LENGTH_PRIO3> = task.vdaf_verify_key().unwrap();

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &(),
        report_metadata.id(),
        &false,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
    let report = LeaderStoredReport::generate(
        *task.id(),
        report_metadata,
        helper_hpke_keypair.config(),
        Vec::new(),
        &transcript,
    );
    let repeated_public_extension_report = LeaderStoredReport::generate(
        *task.id(),
        ReportMetadata::new(
            random(),
            time,
            Vec::from([
                Extension::new(ExtensionType::Tbd, Vec::new()),
                Extension::new(ExtensionType::Tbd, Vec::new()),
            ]),
        ),
        helper_hpke_keypair.config(),
        Vec::new(),
        &transcript,
    );
    let repeated_private_extension_report = LeaderStoredReport::generate(
        *task.id(),
        ReportMetadata::new(random(), time, Vec::new()),
        helper_hpke_keypair.config(),
        Vec::from([
            Extension::new(ExtensionType::Tbd, Vec::new()),
            Extension::new(ExtensionType::Tbd, Vec::new()),
        ]),
        &transcript,
    );
    let repeated_public_private_extension_report = LeaderStoredReport::generate(
        *task.id(),
        ReportMetadata::new(
            random(),
            time,
            Vec::from([Extension::new(ExtensionType::Tbd, Vec::new())]),
        ),
        helper_hpke_keypair.config(),
        Vec::from([Extension::new(ExtensionType::Tbd, Vec::new())]),
        &transcript,
    );
    let aggregation_job_id = random();

    let lease = ds
        .run_unnamed_tx(|tx| {
            let task = leader_task.clone();
            let report = report.clone();
            let repeated_public_extension_report = repeated_public_extension_report.clone();
            let repeated_private_extension_report = repeated_private_extension_report.clone();
            let repeated_public_private_extension_report =
                repeated_public_private_extension_report.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                for report in [
                    &report,
                    &repeated_public_extension_report,
                    &repeated_private_extension_report,
                    &repeated_public_private_extension_report,
                ] {
                    tx.put_client_report(report).await.unwrap();
                    tx.scrub_client_report(report.task_id(), report.metadata().id())
                        .await
                        .unwrap();
                }

                tx.put_aggregation_job(&AggregationJob::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *task.id(),
                    aggregation_job_id,
                    (),
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                for (ord, report) in [
                    &report,
                    &repeated_public_extension_report,
                    &repeated_private_extension_report,
                    &repeated_public_private_extension_report,
                ]
                .iter()
                .enumerate()
                {
                    tx.put_report_aggregation(
                        &report.as_leader_init_report_aggregation(aggregation_job_id, ord as u64),
                    )
                    .await
                    .unwrap();
                }

                tx.put_batch_aggregation(&BatchAggregation::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *task.id(),
                    batch_identifier,
                    (),
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]),
    };
    let mocked_aggregate_failure = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
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
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_failure.assert_async().await;
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job =
        AggregationJob::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            (),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        );

    let want_report_aggregation = ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );
    let want_repeated_public_extension_report_aggregation =
        ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            *repeated_public_extension_report.metadata().id(),
            *repeated_public_extension_report.metadata().time(),
            1,
            None,
            ReportAggregationState::Failed {
                report_error: ReportError::InvalidMessage,
            },
        );
    let want_repeated_private_extension_report_aggregation =
        ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            *repeated_private_extension_report.metadata().id(),
            *repeated_private_extension_report.metadata().time(),
            2,
            None,
            ReportAggregationState::Failed {
                report_error: ReportError::InvalidMessage,
            },
        );
    let want_repeated_public_private_extension_report_aggregation =
        ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            *repeated_public_private_extension_report.metadata().id(),
            *repeated_public_private_extension_report.metadata().time(),
            3,
            None,
            ReportAggregationState::Failed {
                report_error: ReportError::InvalidMessage,
            },
        );

    let want_batch_aggregations = Vec::from([BatchAggregation::<
        VERIFY_KEY_LENGTH_PRIO3,
        TimeInterval,
        Prio3Count,
    >::new(
        *task.id(),
        batch_identifier,
        (),
        0,
        Interval::from_time(&time).unwrap(),
        BatchAggregationState::Aggregating {
            aggregate_share: Some(transcript.leader_output_share.clone().into()),
            report_count: 1,
            checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
            aggregation_jobs_created: 1,
            aggregation_jobs_terminated: 1,
        },
    )]);

    let (
        got_aggregation_job,
        got_report_aggregation,
        got_repeated_public_extension_report_aggregation,
        got_repeated_private_extension_report_aggregation,
        got_repeated_public_private_extension_report_aggregation,
        got_batch_aggregations,
    ) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();
            let repeated_public_extension_report_id = *repeated_public_extension_report.metadata().id();
            let repeated_private_extension_report_id = *repeated_private_extension_report.metadata().id();
            let repeated_public_private_extension_report_id = *repeated_public_private_extension_report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let repeated_public_extension_report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &repeated_public_extension_report_id,
                        &(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let repeated_private_extension_report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &repeated_private_extension_report_id,
                        &(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let repeated_public_private_extension_report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &repeated_public_private_extension_report_id,
                        &(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(&vdaf, task.id())
                        .await
                        .unwrap(),
                );

                Ok((
                    aggregation_job,
                    report_aggregation,
                    repeated_public_extension_report_aggregation,
                    repeated_private_extension_report_aggregation,
                    repeated_public_private_extension_report_aggregation,
                    batch_aggregations,
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(
        want_repeated_public_extension_report_aggregation,
        got_repeated_public_extension_report_aggregation
    );
    assert_eq!(
        want_repeated_private_extension_report_aggregation,
        got_repeated_private_extension_report_aggregation
    );
    assert_eq!(
        want_repeated_public_private_extension_report_aggregation,
        got_repeated_public_private_extension_report_aggregation,
    );
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn leader_sync_time_interval_aggregation_job_init_two_steps() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let (task, report, aggregation_param) =
                (leader_task.clone(), report.clone(), aggregation_param);
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
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

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]),
    };
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_success = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(1),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderContinue {
            transition: transcript.leader_prepare_transitions[1]
                .transition
                .clone()
                .unwrap(),
        },
    );
    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let (vdaf, task, report_id) =
                (Arc::clone(&vdaf), task.clone(), *report.metadata().id());
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );
                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_sync_time_interval_aggregation_job_init_partially_garbage_collected() {
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
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(Prio3::new_count(2).unwrap());

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
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
    let gc_eligible_report_metadata = ReportMetadata::new(random(), gc_eligible_time, Vec::new());

    let gc_ineligible_time = OLDEST_ALLOWED_REPORT_TIMESTAMP
        .add(&Duration::from_seconds(3 * TIME_PRECISION.as_seconds()))
        .unwrap()
        .to_batch_interval_start(&TIME_PRECISION)
        .unwrap();
    let gc_ineligible_batch_identifier =
        TimeInterval::to_batch_identifier(&leader_task, &(), &gc_ineligible_time).unwrap();
    let gc_ineligible_report_metadata =
        ReportMetadata::new(random(), gc_ineligible_time, Vec::new());

    let verify_key: VerifyKey<VERIFY_KEY_LENGTH_PRIO3> = task.vdaf_verify_key().unwrap();

    let gc_eligible_transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &(),
        gc_eligible_report_metadata.id(),
        &false,
    );
    let gc_ineligible_transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &(),
        gc_ineligible_report_metadata.id(),
        &false,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let leader_task = leader_task.clone();
            let gc_eligible_report = gc_eligible_report.clone();
            let gc_ineligible_report = gc_ineligible_report.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&leader_task).await.unwrap();
                tx.put_client_report(&gc_eligible_report).await.unwrap();
                tx.put_client_report(&gc_ineligible_report).await.unwrap();
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
                    VERIFY_KEY_LENGTH_PRIO3,
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
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();
                tx.put_report_aggregation(
                    &gc_eligible_report.as_leader_init_report_aggregation(aggregation_job_id, 0),
                )
                .await
                .unwrap();
                tx.put_report_aggregation(
                    &gc_ineligible_report.as_leader_init_report_aggregation(aggregation_job_id, 1),
                )
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *leader_task.id(),
                    gc_eligible_batch_identifier,
                    (),
                    0,
                    Interval::from_time(&gc_eligible_time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();
                tx.put_batch_aggregation(&BatchAggregation::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *leader_task.id(),
                    gc_ineligible_batch_identifier,
                    (),
                    0,
                    Interval::from_time(&gc_ineligible_time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([
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
        ]),
    };
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_init = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_init.assert_async().await;

    let want_aggregation_job =
        AggregationJob::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>::new(
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
        ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            *gc_eligible_report.metadata().id(),
            *gc_eligible_report.metadata().time(),
            0,
            None,
            ReportAggregationState::Finished,
        );
    let want_ineligible_report_aggregation =
        ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
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
    let want_batch_aggregations = Vec::from([BatchAggregation::<
        VERIFY_KEY_LENGTH_PRIO3,
        TimeInterval,
        Prio3Count,
    >::new(
        *leader_task.id(),
        gc_ineligible_batch_identifier,
        (),
        0,
        Interval::from_time(&gc_ineligible_time).unwrap(),
        BatchAggregationState::Aggregating {
            aggregate_share: Some(gc_ineligible_transcript.leader_output_share.clone().into()),
            report_count: 1,
            checksum: ReportIdChecksum::for_report_id(gc_ineligible_report.metadata().id()),
            aggregation_jobs_created: 1,
            aggregation_jobs_terminated: 1,
        },
    )]);

    let (got_aggregation_job, got_report_aggregations, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(
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
                        &(),
                    )
                    .await
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(&vdaf, task.id())
                        .await
                        .unwrap(),
                );
                Ok((aggregation_job, report_aggregations, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregations, got_report_aggregations);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(2))
        .await;
}

#[tokio::test]
async fn leader_sync_leader_selected_aggregation_job_init_single_step() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(Prio3::new_count(2).unwrap());

    let task = TaskBuilder::new(
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
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
        Vec::new(),
    );
    let verify_key: VerifyKey<VERIFY_KEY_LENGTH_PRIO3> = task.vdaf_verify_key().unwrap();

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &(),
        report_metadata.id(),
        &false,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let (task, report) = (leader_task.clone(), report.clone());
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    LeaderSelected,
                    Prio3Count,
                >::new(
                    *task.id(),
                    aggregation_job_id,
                    (),
                    batch_id,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
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

                tx.put_batch_aggregation(&BatchAggregation::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    LeaderSelected,
                    Prio3Count,
                >::new(
                    *task.id(),
                    batch_id,
                    (),
                    0,
                    Interval::from_time(report.metadata().time()).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            ReportShare::new(
                report.metadata().clone(),
                report.public_share().get_encoded().unwrap(),
                report.helper_encrypted_input_share().clone(),
            ),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]),
    };
    let mocked_aggregate_failure = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
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
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    let error = aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease.clone()),
        )
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
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_failure.assert_async().await;
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job =
        AggregationJob::<VERIFY_KEY_LENGTH_PRIO3, LeaderSelected, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            (),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        );
    let want_report_aggregation = ReportAggregation::<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );
    let want_batch_aggregations = Vec::from([BatchAggregation::<
        VERIFY_KEY_LENGTH_PRIO3,
        LeaderSelected,
        Prio3Count,
    >::new(
        *task.id(),
        batch_id,
        (),
        0,
        Interval::from_time(report.metadata().time()).unwrap(),
        BatchAggregationState::Aggregating {
            aggregate_share: Some(transcript.leader_output_share.clone().into()),
            report_count: 1,
            checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
            aggregation_jobs_created: 1,
            aggregation_jobs_terminated: 1,
        },
    )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let (vdaf, task, report_id) =
                (Arc::clone(&vdaf), task.clone(), *report.metadata().id());
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<VERIFY_KEY_LENGTH_PRIO3, LeaderSelected, Prio3Count>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &()
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<VERIFY_KEY_LENGTH_PRIO3, LeaderSelected, Prio3Count>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );
                Ok((
                    aggregation_job,
                    report_aggregation,
                    merge_batch_aggregations_by_batch(batch_aggregations),
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn leader_sync_leader_selected_aggregation_job_init_two_steps() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
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
        Vec::new(),
    );
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let (task, report, aggregation_param) =
                (leader_task.clone(), report.clone(), aggregation_param);
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    batch_id,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
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

                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    0,
                    Interval::from_time(report.metadata().time()).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            ReportShare::new(
                report.metadata().clone(),
                report.public_share().get_encoded().unwrap(),
                report.helper_encrypted_input_share().clone(),
            ),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]),
    };
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_success = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        batch_id,
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(1),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderContinue {
            transition: transcript.leader_prepare_transitions[1]
                .transition
                .clone()
                .unwrap(),
        },
    );
    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
            *task.id(),
            batch_id,
            aggregation_param,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let (vdaf, task, report_id) =
                (Arc::clone(&vdaf), task.clone(), *report.metadata().id());
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, LeaderSelected, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, LeaderSelected, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );
                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_sync_time_interval_aggregation_job_continue() {
    // Setup: insert a client report and add it to an aggregation job whose state has already
    // been stepped once.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
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
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let (task, aggregation_param, report, transcript) = (
                leader_task.clone(),
                aggregation_param,
                report.clone(),
                transcript.clone(),
            );
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();
                tx.mark_report_aggregated(task.id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(1),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderContinue {
                        transition: transcript.leader_prepare_transitions[1]
                            .transition
                            .clone()
                            .unwrap(),
                    },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    active_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(report.metadata().time()).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();
                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    other_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::EMPTY,
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
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
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Finished,
        )]),
    };
    let mocked_aggregate_failure = server
        .mock(
            "POST",
            task.aggregation_job_uri(&aggregation_job_id, None)
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
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(CONTENT_TYPE.as_str(), AggregationJobContinueReq::MEDIA_TYPE)
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(202)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    let error = aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease.clone()),
        )
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
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_failure.assert_async().await;
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(2),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );

    let want_batch_aggregations = Vec::from([
        BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            active_batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(transcript.leader_aggregate_share),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        ),
        BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            other_batch_identifier,
            aggregation_param,
            0,
            Interval::EMPTY,
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        ),
    ]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = leader_task.clone();
            let report_metadata = report.metadata().clone();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn leader_sync_leader_selected_aggregation_job_continue() {
    // Setup: insert a client report and add it to an aggregation job whose state has already
    // been stepped once.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
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
        Vec::new(),
    );
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
        .aggregate(&aggregation_param, [transcript.leader_output_share])
        .unwrap();

    let lease = ds
        .run_unnamed_tx(|tx| {
            let (task, report, aggregation_param, transcript) = (
                leader_task.clone(),
                report.clone(),
                aggregation_param,
                transcript.clone(),
            );
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    batch_id,
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(1),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderContinue {
                        transition: transcript.leader_prepare_transitions[1]
                            .transition
                            .clone()
                            .unwrap(),
                    },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    0,
                    Interval::from_time(report.metadata().time()).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
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
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Finished,
        )]),
    };
    let mocked_aggregate_failure = server
        .mock(
            "POST",
            task.aggregation_job_uri(&aggregation_job_id, None)
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
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(CONTENT_TYPE.as_str(), AggregationJobContinueReq::MEDIA_TYPE)
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(202)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_failure.assert_async().await;
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, LeaderSelected, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        batch_id,
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(2),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );
    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
            *task.id(),
            batch_id,
            aggregation_param,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(leader_aggregate_share),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let (vdaf, task, report_metadata, aggregation_param) = (
                Arc::clone(&vdaf),
                leader_task.clone(),
                report.metadata().clone(),
                aggregation_param,
            );
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, LeaderSelected, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    LeaderSelected::get_batch_aggregations_for_collection_identifier::<
                        0,
                        dummy::Vdaf,
                        _,
                    >(
                        tx,
                        task.id(),
                        task.time_precision(),
                        &vdaf,
                        &batch_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap(),
                );
                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_init_to_pending() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(1));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
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

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
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
    let helper_response = AggregationJobResp::Processing;
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_request = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderPoll {
            leader_state: transcript.leader_prepare_transitions[0].state.clone(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_init_to_pending_two_step() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
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

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
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
    let helper_response = AggregationJobResp::Processing;
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_request = server
        .mock(
            "PUT",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(
            CONTENT_TYPE.as_str(),
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(201)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderPoll {
            leader_state: transcript.leader_prepare_transitions[0].state.clone(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_continue_to_pending() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let transition = transcript.leader_prepare_transitions[1]
                .transition
                .clone()
                .unwrap();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(1),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderContinue { transition },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
    let leader_request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            *report.metadata().id(),
            transcript.leader_prepare_transitions[1].message.clone(),
        )]),
    );
    let helper_response = AggregationJobResp::Processing;
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_request = server
        .mock(
            "POST",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_header(header, value.as_str())
        .match_header(CONTENT_TYPE.as_str(), AggregationJobContinueReq::MEDIA_TYPE)
        .match_body(leader_request.get_encoded().unwrap())
        .with_status(202)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(1),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderPoll {
            leader_state: transcript.leader_prepare_transitions[1].state.clone(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_init_poll_to_pending() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(1));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let leader_state = transcript.leader_prepare_transitions[0].state.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderPoll { leader_state },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
    let helper_response = AggregationJobResp::Processing;
    let (header, value) = agg_auth_token.request_authentication();

    let mocked_aggregate_request = server
        .mock(
            "GET",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_query("step=0")
        .match_header(header, value.as_str())
        .with_status(200)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderPoll {
            leader_state: transcript.leader_prepare_transitions[0].state.clone(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_init_poll_to_pending_two_step() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let leader_state = transcript.leader_prepare_transitions[0].state.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderPoll { leader_state },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
    let helper_response = AggregationJobResp::Processing;
    let (header, value) = agg_auth_token.request_authentication();

    let mocked_aggregate_request = server
        .mock(
            "GET",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_query("step=0")
        .match_header(header, value.as_str())
        .with_status(200)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderPoll {
            leader_state: transcript.leader_prepare_transitions[0].state.clone(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_init_poll_to_finished() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(1));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let leader_state = transcript.leader_prepare_transitions[0].state.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderPoll { leader_state },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]),
    };
    let (header, value) = agg_auth_token.request_authentication();

    let mocked_aggregate_request = server
        .mock(
            "GET",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_query("step=0")
        .match_header(header, value.as_str())
        .with_status(200)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(1),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(transcript.leader_output_share.into()),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_init_poll_to_continue() {
    // Setup: insert a client report and add it to a new aggregation job.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();

    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let aggregation_param = dummy::AggregationParam(0);

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &0,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let leader_state = transcript.leader_prepare_transitions[0].state.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();

                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderPoll { leader_state },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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

    // Setup: prepare mocked HTTP response.
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )]),
    };
    let (header, value) = agg_auth_token.request_authentication();

    let mocked_aggregate_request = server
        .mock(
            "GET",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_query("step=0")
        .match_header(header, value.as_str())
        .with_status(200)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run: create an aggregation job driver & try to step the aggregation we've created twice.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(1),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_request.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(1),
    );

    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderContinue {
            transition: transcript.leader_prepare_transitions[1]
                .transition
                .clone()
                .unwrap(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = task.clone();
            let report_id = *report.metadata().id();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        &report_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_continue_poll_to_pending() {
    // Setup: insert a client report and add it to an aggregation job whose state has already
    // been stepped once.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();
    let leader_task = task.leader_view().unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let active_batch_identifier =
        TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let leader_state = transcript.leader_prepare_transitions[1].state.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();
                tx.mark_report_aggregated(task.id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(1),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderPoll { leader_state },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    active_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(report.metadata().time()).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
            })
        })
        .await
        .unwrap();
    assert_eq!(lease.leased().task_id(), task.id());
    assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

    // Setup: prepare mocked HTTP responses.
    let helper_response = AggregationJobResp::Processing;
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_success = server
        .mock(
            "GET",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_query("step=1")
        .match_header(header, value.as_str())
        .with_status(200)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(1),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::LeaderPoll {
            leader_state: transcript.leader_prepare_transitions[1].state.clone(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            active_batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = leader_task.clone();
            let report_metadata = report.metadata().clone();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn leader_async_aggregation_job_continue_poll_to_finished() {
    // Setup: insert a client report and add it to an aggregation job whose state has already
    // been stepped once.
    install_test_trace_subscriber();
    let mut server = mockito::Server::new_async().await;
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();
    let leader_task = task.leader_view().unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let active_batch_identifier =
        TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let agg_auth_token = task.aggregator_auth_token();
    let helper_hpke_keypair = HpkeKeypair::test();
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
            let task = leader_task.clone();
            let report = report.clone();
            let leader_state = transcript.leader_prepare_transitions[1].state.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();
                tx.mark_report_aggregated(task.id(), report.metadata().id())
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(1),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *task.id(),
                    aggregation_job_id,
                    *report.metadata().id(),
                    *report.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::LeaderPoll { leader_state },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *task.id(),
                    active_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(report.metadata().time()).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
            })
        })
        .await
        .unwrap();
    assert_eq!(lease.leased().task_id(), task.id());
    assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

    // Setup: prepare mocked HTTP responses.
    let helper_response = AggregationJobResp::Finished {
        prepare_resps: Vec::from([PrepareResp::new(
            *report.metadata().id(),
            PrepareStepResult::Finished,
        )]),
    };
    let (header, value) = agg_auth_token.request_authentication();
    let mocked_aggregate_success = server
        .mock(
            "GET",
            task.aggregation_job_uri(&aggregation_job_id, None)
                .unwrap()
                .path(),
        )
        .match_query("step=1")
        .match_header(header, value.as_str())
        .with_status(200)
        .with_header(CONTENT_TYPE.as_str(), AggregationJobResp::MEDIA_TYPE)
        .with_body(helper_response.get_encoded().unwrap())
        .create_async()
        .await;

    // Run.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    mocked_aggregate_success.assert_async().await;

    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(2),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report.metadata().id(),
        *report.metadata().time(),
        0,
        None,
        ReportAggregationState::Finished,
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            active_batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(report.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(transcript.leader_aggregate_share),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let task = leader_task.clone();
            let report_metadata = report.metadata().clone();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn helper_async_init_processing_to_finished() {
    // Setup: insert an aggregation job with a report aggregation in state HelperInitProcessing.
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    let hpke_keypair = ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(1));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let active_batch_identifier =
        TimeInterval::to_batch_identifier(&helper_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let report_share = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata,
        hpke_keypair.config(),
        &transcript.public_share,
        Vec::new(),
        &transcript.helper_input_share,
    );
    let aggregation_job_id = random();

    let lease = ds
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share = report_share.clone();
            let message = transcript.leader_prepare_transitions[0].message.clone();

            Box::pin(async move {
                let report_id = *report_share.metadata().id();
                let report_timestamp = *report_share.metadata().time();

                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(helper_task.id(), &report_id, &report_timestamp)
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    report_id,
                    report_timestamp,
                    0,
                    None,
                    ReportAggregationState::HelperInitProcessing {
                        prepare_init: PrepareInit::new(report_share, message),
                        require_taskbind_extension: false,
                    },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    active_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&report_timestamp).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
            })
        })
        .await
        .unwrap();
    assert_eq!(lease.leased().task_id(), task.id());
    assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

    // Run.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(0),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report_share.metadata().id(),
        *report_share.metadata().time(),
        0,
        Some(PrepareResp::new(
            *report_share.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )),
        ReportAggregationState::Finished,
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            active_batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(report_share.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(transcript.helper_aggregate_share),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report_share.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let helper_task = helper_task.clone();
            let report_metadata = report_share.metadata().clone();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        helper_task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Helper,
                        helper_task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        helper_task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

#[tokio::test]
async fn helper_async_init_processing_to_continue() {
    // Setup: insert an aggregation job with a report aggregation in state HelperInitProcessing.
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    let hpke_keypair = ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let active_batch_identifier =
        TimeInterval::to_batch_identifier(&helper_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let report_share = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata,
        hpke_keypair.config(),
        &transcript.public_share,
        Vec::new(),
        &transcript.helper_input_share,
    );
    let aggregation_job_id = random();

    let lease = ds
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share = report_share.clone();
            let message = transcript.leader_prepare_transitions[0].message.clone();

            Box::pin(async move {
                let report_id = *report_share.metadata().id();
                let report_timestamp = *report_share.metadata().time();

                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(helper_task.id(), &report_id, &report_timestamp)
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    report_id,
                    report_timestamp,
                    0,
                    None,
                    ReportAggregationState::HelperInitProcessing {
                        prepare_init: PrepareInit::new(report_share, message),
                        require_taskbind_extension: false,
                    },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    active_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&report_timestamp).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
            })
        })
        .await
        .unwrap();
    assert_eq!(lease.leased().task_id(), task.id());
    assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

    // Run.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::AwaitingRequest,
        AggregationJobStep::from(0),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report_share.metadata().id(),
        *report_share.metadata().time(),
        0,
        Some(PrepareResp::new(
            *report_share.metadata().id(),
            PrepareStepResult::Continue {
                message: transcript.helper_prepare_transitions[0].message.clone(),
            },
        )),
        ReportAggregationState::HelperContinue {
            prepare_state: *transcript.helper_prepare_transitions[0].prepare_state(),
        },
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            active_batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(report_share.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 0,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let helper_task = helper_task.clone();
            let report_metadata = report_share.metadata().clone();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        helper_task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Helper,
                        helper_task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        helper_task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(0))
        .await;
}

#[tokio::test]
async fn helper_async_continue_processing_to_finished() {
    // Setup: insert an aggregation job with a report aggregation in state HelperInitProcessing.
    install_test_trace_subscriber();
    let clock = MockClock::default();
    let ephemeral_datastore = ephemeral_datastore().await;
    let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);
    let hpke_keypair = ds.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(dummy::Vdaf::new(2));

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let active_batch_identifier =
        TimeInterval::to_batch_identifier(&helper_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();

    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &13,
    );

    let report_share = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata,
        hpke_keypair.config(),
        &transcript.public_share,
        Vec::new(),
        &transcript.helper_input_share,
    );
    let aggregation_job_id = random();

    let lease = ds
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share = report_share.clone();
            let prepare_state = *transcript.helper_prepare_transitions[0].prepare_state();
            let message = transcript.leader_prepare_transitions[1].message.clone();

            Box::pin(async move {
                let report_id = *report_share.metadata().id();
                let report_timestamp = *report_share.metadata().time();

                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(helper_task.id(), &report_id, &report_timestamp)
                    .await
                    .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Active,
                    AggregationJobStep::from(1),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    report_id,
                    report_timestamp,
                    0,
                    None,
                    ReportAggregationState::HelperContinueProcessing {
                        prepare_state,
                        prepare_continue: PrepareContinue::new(report_id, message),
                    },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    active_batch_identifier,
                    aggregation_param,
                    0,
                    Interval::from_time(&report_timestamp).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                let lease = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap()
                    .remove(0);

                Ok(lease)
            })
        })
        .await
        .unwrap();
    assert_eq!(lease.leased().task_id(), task.id());
    assert_eq!(lease.leased().aggregation_job_id(), &aggregation_job_id);

    // Run.
    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
    );
    aggregation_job_driver
        .step_aggregation_job(
            ds.clone(),
            Arc::new(
                HpkeKeypairCache::new(Arc::clone(&ds), HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL)
                    .await
                    .unwrap(),
            ),
            Arc::new(lease),
        )
        .await
        .unwrap();

    // Verify.
    let want_aggregation_job = AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        aggregation_param,
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Finished,
        AggregationJobStep::from(1),
    );
    let want_report_aggregation = ReportAggregation::<0, dummy::Vdaf>::new(
        *task.id(),
        aggregation_job_id,
        *report_share.metadata().id(),
        *report_share.metadata().time(),
        0,
        Some(PrepareResp::new(
            *report_share.metadata().id(),
            PrepareStepResult::Finished,
        )),
        ReportAggregationState::Finished,
    );

    let want_batch_aggregations =
        Vec::from([BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
            *task.id(),
            active_batch_identifier,
            aggregation_param,
            0,
            Interval::from_time(report_share.metadata().time()).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: Some(transcript.helper_aggregate_share),
                report_count: 1,
                checksum: ReportIdChecksum::for_report_id(report_share.metadata().id()),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = Arc::clone(&vdaf);
            let helper_task = helper_task.clone();
            let report_metadata = report_share.metadata().clone();

            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        helper_task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Helper,
                        helper_task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<0, TimeInterval, dummy::Vdaf>(
                        &vdaf,
                        helper_task.id(),
                    )
                    .await
                    .unwrap(),
                );

                Ok((aggregation_job, report_aggregation, batch_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(want_report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);

    assert_task_aggregation_counter(&ds, *task.id(), TaskAggregationCounter::new_with_values(1))
        .await;
}

struct CancelAggregationJobTestCase {
    task: AggregatorTask,
    vdaf: Arc<Prio3Count>,
    aggregation_job: AggregationJob<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>,
    batch_identifier: Interval,
    report_aggregation: ReportAggregation<VERIFY_KEY_LENGTH_PRIO3, Prio3Count>,
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
    datastore.put_hpke_key().await.unwrap();
    let vdaf = Arc::new(Prio3::new_count(2).unwrap());
    let mock_helper = mockito::Server::new_async().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_helper_aggregator_endpoint(mock_helper.url().parse().unwrap())
    .build()
    .leader_view()
    .unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let verify_key: VerifyKey<VERIFY_KEY_LENGTH_PRIO3> = task.vdaf_verify_key().unwrap();

    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &(),
        report_metadata.id(),
        &false,
    );

    let helper_hpke_keypair = HpkeKeypair::test();
    let report = LeaderStoredReport::generate(
        *task.id(),
        report_metadata,
        helper_hpke_keypair.config(),
        Vec::new(),
        &transcript,
    );
    let aggregation_job_id = random();

    let aggregation_job = AggregationJob::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>::new(
        *task.id(),
        aggregation_job_id,
        (),
        (),
        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
        AggregationJobState::Active,
        AggregationJobStep::from(0),
    );
    let report_aggregation = report.as_leader_init_report_aggregation(aggregation_job_id, 0);

    let lease = datastore
        .run_unnamed_tx(|tx| {
            let (task, report, aggregation_job, report_aggregation) = (
                task.clone(),
                report.clone(),
                aggregation_job.clone(),
                report_aggregation.clone(),
            );
            Box::pin(async move {
                tx.put_aggregator_task(&task).await.unwrap();
                tx.put_client_report(&report).await.unwrap();
                tx.scrub_client_report(report.task_id(), report.metadata().id())
                    .await
                    .unwrap();
                tx.put_aggregation_job(&aggregation_job).await.unwrap();
                tx.put_report_aggregation(&report_aggregation)
                    .await
                    .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<
                    VERIFY_KEY_LENGTH_PRIO3,
                    TimeInterval,
                    Prio3Count,
                >::new(
                    *task.id(),
                    batch_identifier,
                    (),
                    0,
                    Interval::from_time(&time).unwrap(),
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 0,
                    },
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
                .aggregation_job_uri(test_case.aggregation_job.id(), None)
                .unwrap()
                .unwrap()
                .path(),
        )
        .with_status(204)
        .create_async()
        .await;

    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
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

    let want_batch_aggregations = Vec::from([BatchAggregation::new(
        *test_case.task.id(),
        test_case.batch_identifier,
        (),
        0,
        Interval::from_time(test_case.report_aggregation.time()).unwrap(),
        BatchAggregationState::Aggregating {
            aggregate_share: None,
            report_count: 0,
            checksum: ReportIdChecksum::default(),
            aggregation_jobs_created: 1,
            aggregation_jobs_terminated: 1,
        },
    )]);

    let (got_aggregation_job, got_report_aggregation, got_batch_aggregations, got_leases) = test_case
        .datastore
        .run_unnamed_tx(|tx| {
            let (vdaf, task, report_id, aggregation_job) = (
                Arc::clone(&test_case.vdaf),
                test_case.task.clone(),
                *test_case.report_aggregation.report_id(),
                want_aggregation_job.clone(),
            );
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(
                        task.id(),
                        aggregation_job.id(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregation = tx
                    .get_report_aggregation_by_report_id(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        aggregation_job.id(),
                        &report_id,
                        &(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(&vdaf, task.id())
                        .await
                        .unwrap(),
                );
                let leases = tx
                    .acquire_incomplete_aggregation_jobs(&StdDuration::from_secs(60), 1)
                    .await
                    .unwrap();
                Ok((aggregation_job, report_aggregation, batch_aggregations, leases))
            })
        })
        .await
        .unwrap();
    assert_eq!(want_aggregation_job, got_aggregation_job);
    assert_eq!(test_case.report_aggregation, got_report_aggregation);
    assert_eq!(want_batch_aggregations, got_batch_aggregations);
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
                .aggregation_job_uri(test_case.aggregation_job.id(), None)
                .unwrap()
                .unwrap()
                .path(),
        )
        .with_status(400)
        .create_async()
        .await;

    let aggregation_job_driver = AggregationJobDriver::new(
        reqwest::Client::builder().build().unwrap(),
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
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
    ds.put_hpke_key().await.unwrap();
    let stopper = Stopper::new();

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();
    let agg_auth_token = task.aggregator_auth_token();
    let aggregation_job_id = random();
    let verify_key: VerifyKey<VERIFY_KEY_LENGTH_PRIO3> = task.vdaf_verify_key().unwrap();

    let helper_hpke_keypair = HpkeKeypair::test();

    let vdaf = Prio3::new_count(2).unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let transcript = run_vdaf(
        &vdaf,
        task.id(),
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
        let task = leader_task.clone();
        let report = report.clone();
        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();

            tx.put_client_report(&report).await.unwrap();
            tx.scrub_client_report(report.task_id(), report.metadata().id())
                .await
                .unwrap();

            tx.put_aggregation_job(&AggregationJob::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *task.id(),
                aggregation_job_id,
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
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

            tx.put_batch_aggregation(&BatchAggregation::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *task.id(),
                batch_identifier,
                (),
                0,
                Interval::from_time(&time).unwrap(),
                BatchAggregationState::Aggregating {
                    aggregate_share: None,
                    report_count: 0,
                    checksum: ReportIdChecksum::default(),
                    aggregation_jobs_created: 1,
                    aggregation_jobs_terminated: 0,
                },
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
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
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
            task.aggregation_job_uri(&aggregation_job_id, None)
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
            task.aggregation_job_uri(&aggregation_job_id, None)
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
    let (got_aggregation_job, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task = task.clone();

            Box::pin(async move {
                let got_aggregation_job = tx
                    .get_aggregation_job(task.id(), &aggregation_job_id)
                    .await
                    .unwrap()
                    .unwrap();
                let got_batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(&vdaf, task.id())
                        .await
                        .unwrap(),
                );

                Ok((got_aggregation_job, got_batch_aggregations))
            })
        })
        .await
        .unwrap();
    assert_eq!(
        got_aggregation_job,
        AggregationJob::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            (),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Abandoned,
            AggregationJobStep::from(0),
        ),
    );
    assert_eq!(
        got_batch_aggregations,
        Vec::from([BatchAggregation::new(
            *task.id(),
            batch_identifier,
            (),
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]),
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
    ds.put_hpke_key().await.unwrap();
    let stopper = Stopper::new();

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_helper_aggregator_endpoint(server.url().parse().unwrap())
    .build();

    let leader_task = task.leader_view().unwrap();
    let agg_auth_token = task.aggregator_auth_token();
    let aggregation_job_id = random();
    let verify_key: VerifyKey<VERIFY_KEY_LENGTH_PRIO3> = task.vdaf_verify_key().unwrap();

    let helper_hpke_keypair = HpkeKeypair::test();

    let vdaf = Prio3::new_count(2).unwrap();
    let time = clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let batch_identifier = TimeInterval::to_batch_identifier(&leader_task, &(), &time).unwrap();
    let report_metadata = ReportMetadata::new(random(), time, Vec::new());
    let transcript = run_vdaf(
        &vdaf,
        task.id(),
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
        let task = leader_task.clone();
        let report = report.clone();
        Box::pin(async move {
            tx.put_aggregator_task(&task).await.unwrap();

            tx.put_client_report(&report).await.unwrap();
            tx.scrub_client_report(report.task_id(), report.metadata().id())
                .await
                .unwrap();

            tx.put_aggregation_job(&AggregationJob::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *task.id(),
                aggregation_job_id,
                (),
                (),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
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

            tx.put_batch_aggregation(&BatchAggregation::<
                VERIFY_KEY_LENGTH_PRIO3,
                TimeInterval,
                Prio3Count,
            >::new(
                *task.id(),
                batch_identifier,
                (),
                0,
                Interval::from_time(&time).unwrap(),
                BatchAggregationState::Aggregating {
                    aggregate_share: None,
                    report_count: 0,
                    checksum: ReportIdChecksum::default(),
                    aggregation_jobs_created: 1,
                    aggregation_jobs_terminated: 0,
                },
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
        LimitedRetryer::new(0),
        &noop_meter(),
        BATCH_AGGREGATION_SHARD_COUNT,
        TASK_AGGREGATION_COUNTER_SHARD_COUNT,
        HpkeKeypairCache::DEFAULT_REFRESH_INTERVAL,
        DEFAULT_ASYNC_POLL_INTERVAL,
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
            task.aggregation_job_uri(&aggregation_job_id, None)
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
            task.aggregation_job_uri(&aggregation_job_id, None)
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
    let (got_aggregation_job, got_batch_aggregations) = ds
        .run_unnamed_tx(|tx| {
            let vdaf = vdaf.clone();
            let task = task.clone();

            Box::pin(async move {
                let got_aggregation_job = tx
                    .get_aggregation_job(task.id(), &aggregation_job_id)
                    .await
                    .unwrap()
                    .unwrap();
                let got_batch_aggregations = merge_batch_aggregations_by_batch(
                    tx.get_batch_aggregations_for_task::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>(&vdaf, task.id())
                        .await
                        .unwrap(),
                );

                Ok((got_aggregation_job, got_batch_aggregations))
            })
        })
        .await
        .unwrap();
    assert_eq!(
        got_aggregation_job,
        AggregationJob::<VERIFY_KEY_LENGTH_PRIO3, TimeInterval, Prio3Count>::new(
            *task.id(),
            aggregation_job_id,
            (),
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Abandoned,
            AggregationJobStep::from(0),
        ),
    );
    assert_eq!(
        got_batch_aggregations,
        Vec::from([BatchAggregation::new(
            *task.id(),
            batch_identifier,
            (),
            0,
            Interval::from_time(&time).unwrap(),
            BatchAggregationState::Aggregating {
                aggregate_share: None,
                report_count: 0,
                checksum: ReportIdChecksum::default(),
                aggregation_jobs_created: 1,
                aggregation_jobs_terminated: 1,
            },
        )]),
    );
}
