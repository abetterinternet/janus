use crate::aggregator::{
    aggregation_job_continue::test_util::{
        post_aggregation_job_and_decode, post_aggregation_job_expecting_error,
    },
    empty_batch_aggregations,
    http_handlers::test_util::HttpHandlerTest,
    test_util::{
        assert_task_aggregation_counter, generate_helper_report_share,
        BATCH_AGGREGATION_SHARD_COUNT,
    },
};
use assert_matches::assert_matches;
use futures::future::try_join_all;
use janus_aggregator_core::{
    batch_mode::CollectableBatchMode,
    datastore::models::{
        merge_batch_aggregations_by_batch, AggregationJob, AggregationJobState, BatchAggregation,
        BatchAggregationState, ReportAggregation, ReportAggregationState, TaskAggregationCounter,
    },
    task::{test_util::TaskBuilder, AggregationMode, BatchMode, VerifyKey},
};
use janus_core::{
    report_id::ReportIdChecksumExt,
    test_util::run_vdaf,
    time::{Clock, IntervalExt, MockClock, TimeExt},
    vdaf::VdafInstance,
};
use janus_messages::{
    batch_mode::TimeInterval, AggregationJobContinueReq, AggregationJobResp, AggregationJobStep,
    Duration, Interval, PrepareContinue, PrepareResp, PrepareStepResult, ReportError, ReportId,
    ReportIdChecksum, ReportMetadata, Role, Time,
};
use prio::{
    topology::ping_pong::PingPongMessage,
    vdaf::{dummy, Aggregator},
};
use rand::random;
use std::sync::Arc;
use trillium::Status;

#[tokio::test]
async fn aggregate_continue_sync() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair: hpke_key,
        ..
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(2));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    // report_share_0 is a "happy path" report.
    let report_metadata_0 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_0 = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_0.id(),
        &measurement,
    );
    let helper_prep_state_0 = transcript_0.helper_prepare_transitions[0].prepare_state();
    let leader_prep_message_0 = &transcript_0.leader_prepare_transitions[1].message;
    let report_share_0 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_0.clone(),
        hpke_key.config(),
        &transcript_0.public_share,
        Vec::new(),
        &transcript_0.helper_input_share,
    );

    // report_share_1 is omitted by the leader's request.
    let report_metadata_1 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_1 = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_1.id(),
        &measurement,
    );

    let helper_prep_state_1 = transcript_1.helper_prepare_transitions[0].prepare_state();
    let report_share_1 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_1.clone(),
        hpke_key.config(),
        &transcript_1.public_share,
        Vec::new(),
        &transcript_1.helper_input_share,
    );

    // report_share_2 falls into a batch that has already been collected.
    let past_clock = MockClock::new(Time::from_seconds_since_epoch(
        task.time_precision().as_seconds() / 2,
    ));
    let report_metadata_2 = ReportMetadata::new(
        random(),
        past_clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_2 = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_2.id(),
        &measurement,
    );
    let helper_prep_state_2 = transcript_2.helper_prepare_transitions[0].prepare_state();
    let leader_prep_message_2 = &transcript_2.leader_prepare_transitions[1].message;
    let report_share_2 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_2.clone(),
        hpke_key.config(),
        &transcript_2.public_share,
        Vec::new(),
        &transcript_2.helper_input_share,
    );

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share_0 = report_share_0.clone();
            let report_share_1 = report_share_1.clone();
            let report_share_2 = report_share_2.clone();
            let helper_prep_state_0 = *helper_prep_state_0;
            let helper_prep_state_1 = *helper_prep_state_1;
            let helper_prep_state_2 = *helper_prep_state_2;
            let report_metadata_0 = report_metadata_0.clone();
            let report_metadata_1 = report_metadata_1.clone();
            let report_metadata_2 = report_metadata_2.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();

                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_0.metadata().id(),
                    report_share_0.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_1.metadata().id(),
                    report_share_1.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_2.metadata().id(),
                    report_share_2.metadata().time(),
                )
                .await
                .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::AwaitingRequest,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    *report_metadata_0.id(),
                    *report_metadata_0.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_0,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    *report_metadata_1.id(),
                    *report_metadata_1.time(),
                    1,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_1,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    *report_metadata_2.id(),
                    *report_metadata_2.time(),
                    2,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_2,
                    },
                ))
                .await
                .unwrap();

                // Write collected batch aggregations for the interval that report_share_2 falls
                // into, which will cause it to fail to prepare.
                try_join_all(
                    empty_batch_aggregations::<0, TimeInterval, dummy::Vdaf>(
                        &helper_task,
                        BATCH_AGGREGATION_SHARD_COUNT,
                        &Interval::new(
                            Time::from_seconds_since_epoch(0),
                            *helper_task.time_precision(),
                        )
                        .unwrap(),
                        &aggregation_param,
                        &[],
                    )
                    .iter()
                    .map(|ba| tx.put_batch_aggregation(ba)),
                )
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([
            PrepareContinue::new(*report_metadata_0.id(), leader_prep_message_0.clone()),
            PrepareContinue::new(*report_metadata_2.id(), leader_prep_message_2.clone()),
        ]),
    );

    // Send request, and parse response.
    let aggregate_resp =
        post_aggregation_job_and_decode(&task, &aggregation_job_id, &request, &handler).await;

    // Validate response.
    assert_eq!(
        aggregate_resp,
        AggregationJobResp::Finished {
            prepare_resps: Vec::from([
                PrepareResp::new(*report_metadata_0.id(), PrepareStepResult::Finished),
                PrepareResp::new(
                    *report_metadata_2.id(),
                    PrepareStepResult::Reject(ReportError::BatchCollected),
                )
            ])
        }
    );

    // Validate datastore.
    let (aggregation_job, report_aggregations) = datastore
        .run_unnamed_tx(|tx| {
            let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregations = tx
                    .get_report_aggregations_for_aggregation_job(
                        vdaf.as_ref(),
                        &Role::Helper,
                        task.id(),
                        &aggregation_job_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                Ok((aggregation_job, report_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(
        aggregation_job,
        AggregationJob::new(
            *task.id(),
            aggregation_job_id,
            aggregation_param,
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        )
        .with_last_request_hash(aggregation_job.last_request_hash().unwrap())
    );
    assert_eq!(
        report_aggregations,
        Vec::from([
            ReportAggregation::new(
                *task.id(),
                aggregation_job_id,
                *report_metadata_0.id(),
                *report_metadata_0.time(),
                0,
                Some(PrepareResp::new(
                    *report_metadata_0.id(),
                    PrepareStepResult::Finished
                )),
                ReportAggregationState::Finished,
            ),
            ReportAggregation::new(
                *task.id(),
                aggregation_job_id,
                *report_metadata_1.id(),
                *report_metadata_1.time(),
                1,
                None,
                ReportAggregationState::Failed {
                    report_error: ReportError::ReportDropped
                },
            ),
            ReportAggregation::new(
                *task.id(),
                aggregation_job_id,
                *report_metadata_2.id(),
                *report_metadata_2.time(),
                2,
                Some(PrepareResp::new(
                    *report_metadata_2.id(),
                    PrepareStepResult::Reject(ReportError::BatchCollected)
                )),
                ReportAggregationState::Failed {
                    report_error: ReportError::BatchCollected
                },
            )
        ])
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(1),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_async() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair: hpke_key,
        ..
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(2));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    // report_share_0 is a "happy path" report.
    let report_metadata_0 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_0 = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_0.id(),
        &measurement,
    );
    let helper_prep_state_0 = transcript_0.helper_prepare_transitions[0].prepare_state();
    let leader_prep_message_0 = &transcript_0.leader_prepare_transitions[1].message;
    let report_share_0 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_0.clone(),
        hpke_key.config(),
        &transcript_0.public_share,
        Vec::new(),
        &transcript_0.helper_input_share,
    );

    // report_share_1 is omitted by the leader's request.
    let report_metadata_1 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_1 = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_1.id(),
        &measurement,
    );

    let helper_prep_state_1 = transcript_1.helper_prepare_transitions[0].prepare_state();
    let report_share_1 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_1.clone(),
        hpke_key.config(),
        &transcript_1.public_share,
        Vec::new(),
        &transcript_1.helper_input_share,
    );

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share_0 = report_share_0.clone();
            let report_share_1 = report_share_1.clone();
            let helper_prep_state_0 = *helper_prep_state_0;
            let helper_prep_state_1 = *helper_prep_state_1;
            let report_metadata_0 = report_metadata_0.clone();
            let report_metadata_1 = report_metadata_1.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();

                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_0.metadata().id(),
                    report_share_0.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_1.metadata().id(),
                    report_share_1.metadata().time(),
                )
                .await
                .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    aggregation_param,
                    (),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::AwaitingRequest,
                    AggregationJobStep::from(0),
                ))
                .await
                .unwrap();

                tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    *report_metadata_0.id(),
                    *report_metadata_0.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_0,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    *report_metadata_1.id(),
                    *report_metadata_1.time(),
                    1,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_1,
                    },
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            *report_metadata_0.id(),
            leader_prep_message_0.clone(),
        )]),
    );

    // Send request, and parse response.
    let aggregate_resp =
        post_aggregation_job_and_decode(&task, &aggregation_job_id, &request, &handler).await;

    // Validate response.
    assert_eq!(aggregate_resp, AggregationJobResp::Processing);

    // Validate datastore.
    let (aggregation_job, report_aggregations) = datastore
        .run_unnamed_tx(|tx| {
            let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
            Box::pin(async move {
                let aggregation_job = tx
                    .get_aggregation_job::<0, TimeInterval, dummy::Vdaf>(
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                let report_aggregations = tx
                    .get_report_aggregations_for_aggregation_job(
                        vdaf.as_ref(),
                        &Role::Helper,
                        task.id(),
                        &aggregation_job_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                Ok((aggregation_job, report_aggregations))
            })
        })
        .await
        .unwrap();

    assert_eq!(
        aggregation_job,
        AggregationJob::new(
            *task.id(),
            aggregation_job_id,
            aggregation_param,
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Active,
            AggregationJobStep::from(1),
        )
        .with_last_request_hash(aggregation_job.last_request_hash().unwrap())
    );
    assert_eq!(
        report_aggregations,
        Vec::from([
            ReportAggregation::new(
                *task.id(),
                aggregation_job_id,
                *report_metadata_0.id(),
                *report_metadata_0.time(),
                0,
                None,
                ReportAggregationState::HelperContinueProcessing {
                    prepare_state: *helper_prep_state_0,
                    prepare_continue: PrepareContinue::new(
                        *report_metadata_0.id(),
                        leader_prep_message_0.clone()
                    ),
                },
            ),
            ReportAggregation::new(
                *task.id(),
                aggregation_job_id,
                *report_metadata_1.id(),
                *report_metadata_1.time(),
                1,
                None,
                ReportAggregationState::Failed {
                    report_error: ReportError::ReportDropped
                },
            ),
        ])
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_accumulate_batch_aggregation() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair: hpke_key,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let aggregation_job_id_0 = random();
    let aggregation_job_id_1 = random();
    let first_batch_interval_clock = MockClock::default();
    let second_batch_interval_clock = MockClock::new(
        first_batch_interval_clock
            .now()
            .add(task.time_precision())
            .unwrap(),
    );

    let vdaf = dummy::Vdaf::new(2);
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    // report_share_0 is a "happy path" report.
    let report_time_0 = first_batch_interval_clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let report_metadata_0 = ReportMetadata::new(random(), report_time_0, Vec::new());
    let transcript_0 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_0.id(),
        &measurement,
    );
    let helper_prep_state_0 = transcript_0.helper_prepare_transitions[0].prepare_state();
    let ping_pong_leader_message_0 = &transcript_0.leader_prepare_transitions[1].message;
    let report_share_0 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_0.clone(),
        hpke_key.config(),
        &transcript_0.public_share,
        Vec::new(),
        &transcript_0.helper_input_share,
    );

    // report_share_1 is another "happy path" report to exercise in-memory accumulation of
    // output shares
    let report_time_1 = first_batch_interval_clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let report_metadata_1 = ReportMetadata::new(random(), report_time_1, Vec::new());
    let transcript_1 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_1.id(),
        &measurement,
    );
    let helper_prep_state_1 = transcript_1.helper_prepare_transitions[0].prepare_state();
    let ping_pong_leader_message_1 = &transcript_1.leader_prepare_transitions[1].message;
    let report_share_1 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_1.clone(),
        hpke_key.config(),
        &transcript_1.public_share,
        Vec::new(),
        &transcript_1.helper_input_share,
    );

    // report_share_2 aggregates successfully, but into a distinct batch aggregation which has
    // already been collected.
    let report_metadata_2 = ReportMetadata::new(
        random(),
        second_batch_interval_clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_2 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_2.id(),
        &measurement,
    );
    let helper_prep_state_2 = transcript_2.helper_prepare_transitions[0].prepare_state();
    let ping_pong_leader_message_2 = &transcript_2.leader_prepare_transitions[1].message;
    let report_share_2 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_2.clone(),
        hpke_key.config(),
        &transcript_2.public_share,
        Vec::new(),
        &transcript_2.helper_input_share,
    );

    let first_batch_identifier = Interval::new(
        report_metadata_0
            .time()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        *task.time_precision(),
    )
    .unwrap();
    let first_batch_interval = Interval::from_time(&report_time_0)
        .unwrap()
        .merged_with(&report_time_1)
        .unwrap();

    let second_batch_identifier = Interval::new(
        report_metadata_2
            .time()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        *task.time_precision(),
    )
    .unwrap();

    let second_batch_want_batch_aggregations: Vec<_> =
        empty_batch_aggregations::<0, TimeInterval, dummy::Vdaf>(
            &helper_task,
            BATCH_AGGREGATION_SHARD_COUNT,
            &second_batch_identifier,
            &aggregation_param,
            &[],
        )
        .into_iter()
        .map(|ba| {
            BatchAggregation::new(
                *ba.task_id(),
                *ba.batch_identifier(),
                *ba.aggregation_parameter(),
                ba.ord(),
                second_batch_identifier,
                ba.state().clone(),
            )
        })
        .collect();

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share_0 = report_share_0.clone();
            let report_share_1 = report_share_1.clone();
            let report_share_2 = report_share_2.clone();
            let helper_prep_state_0 = *helper_prep_state_0;
            let helper_prep_state_1 = *helper_prep_state_1;
            let helper_prep_state_2 = *helper_prep_state_2;
            let report_metadata_0 = report_metadata_0.clone();
            let report_metadata_1 = report_metadata_1.clone();
            let report_metadata_2 = report_metadata_2.clone();
            let second_batch_want_batch_aggregations = second_batch_want_batch_aggregations.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();

                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_0.metadata().id(),
                    report_share_0.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_1.metadata().id(),
                    report_share_1.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_2.metadata().id(),
                    report_share_2.metadata().time(),
                )
                .await
                .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id_0,
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
                    aggregation_job_id_0,
                    *report_metadata_0.id(),
                    *report_metadata_0.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_0,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id_0,
                    *report_metadata_1.id(),
                    *report_metadata_1.time(),
                    1,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_1,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id_0,
                    *report_metadata_2.id(),
                    *report_metadata_2.time(),
                    2,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_2,
                    },
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    first_batch_identifier,
                    aggregation_param,
                    0,
                    first_batch_identifier,
                    BatchAggregationState::Aggregating {
                        aggregate_share: None,
                        report_count: 0,
                        checksum: ReportIdChecksum::default(),
                        aggregation_jobs_created: 2,
                        aggregation_jobs_terminated: 0,
                    },
                ))
                .await
                .unwrap();

                try_join_all(
                    second_batch_want_batch_aggregations
                        .iter()
                        .map(|ba| tx.put_batch_aggregation(ba)),
                )
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([
            PrepareContinue::new(*report_metadata_0.id(), ping_pong_leader_message_0.clone()),
            PrepareContinue::new(*report_metadata_1.id(), ping_pong_leader_message_1.clone()),
            PrepareContinue::new(*report_metadata_2.id(), ping_pong_leader_message_2.clone()),
        ]),
    );

    // Send request, and parse response.
    let _ = post_aggregation_job_and_decode(&task, &aggregation_job_id_0, &request, &handler).await;

    // Map the batch aggregation ordinal value to 0, as it may vary due to sharding.
    let first_batch_got_batch_aggregations = datastore
        .run_unnamed_tx(|tx| {
            let task = helper_task.clone();
            let vdaf = vdaf.clone();

            Box::pin(async move {
                Ok(merge_batch_aggregations_by_batch(
                    TimeInterval::get_batch_aggregations_for_collection_identifier::<
                        0,
                        dummy::Vdaf,
                        _,
                    >(
                        tx,
                        task.id(),
                        task.time_precision(),
                        &vdaf,
                        &first_batch_identifier,
                        &aggregation_param,
                    )
                    .await
                    .unwrap(),
                ))
            })
        })
        .await
        .unwrap();

    let aggregate_share = vdaf
        .aggregate(
            &aggregation_param,
            [
                transcript_0.helper_output_share,
                transcript_1.helper_output_share,
            ],
        )
        .unwrap();
    let checksum = ReportIdChecksum::for_report_id(report_metadata_0.id())
        .updated_with(report_metadata_1.id());

    assert_eq!(
        first_batch_got_batch_aggregations,
        Vec::from([BatchAggregation::new(
            *task.id(),
            first_batch_identifier,
            aggregation_param,
            0,
            first_batch_interval,
            BatchAggregationState::Aggregating {
                aggregate_share: Some(aggregate_share),
                report_count: 2,
                checksum,
                aggregation_jobs_created: 2,
                aggregation_jobs_terminated: 1,
            },
        )])
    );

    let second_batch_got_batch_aggregations = datastore
        .run_unnamed_tx(|tx| {
            let task = helper_task.clone();
            let vdaf = vdaf.clone();

            Box::pin(async move {
                let mut got_batch_aggregations =
                    TimeInterval::get_batch_aggregations_for_collection_identifier::<
                        0,
                        dummy::Vdaf,
                        _,
                    >(
                        tx,
                        task.id(),
                        task.time_precision(),
                        &vdaf,
                        &second_batch_identifier,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                got_batch_aggregations.sort_unstable_by_key(|ba| ba.ord());
                Ok(got_batch_aggregations)
            })
        })
        .await
        .unwrap();
    assert_eq!(
        second_batch_got_batch_aggregations,
        second_batch_want_batch_aggregations
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(2),
    )
    .await;

    // Aggregate some more reports, which should get accumulated into the batch_aggregations
    // rows created earlier.
    // report_share_3 gets aggreated into the first batch interval.
    let report_time_3 = first_batch_interval_clock
        .now()
        .to_batch_interval_start(task.time_precision())
        .unwrap();
    let report_metadata_3 = ReportMetadata::new(random(), report_time_3, Vec::new());
    let transcript_3 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_3.id(),
        &measurement,
    );
    let helper_prep_state_3 = transcript_3.helper_prepare_transitions[0].prepare_state();
    let ping_pong_leader_message_3 = &transcript_3.leader_prepare_transitions[1].message;
    let report_share_3 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_3.clone(),
        hpke_key.config(),
        &transcript_3.public_share,
        Vec::new(),
        &transcript_3.helper_input_share,
    );

    // report_share_4 gets aggregated into the second batch interval (which has already been
    // collected).
    let report_metadata_4 = ReportMetadata::new(
        random(),
        second_batch_interval_clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_4 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_4.id(),
        &measurement,
    );
    let helper_prep_state_4 = transcript_4.helper_prepare_transitions[0].prepare_state();
    let ping_pong_leader_message_4 = &transcript_4.leader_prepare_transitions[1].message;
    let report_share_4 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_4.clone(),
        hpke_key.config(),
        &transcript_4.public_share,
        Vec::new(),
        &transcript_4.helper_input_share,
    );

    // report_share_5 also gets aggregated into the second batch interval (which has already
    // been collected).
    let report_metadata_5 = ReportMetadata::new(
        random(),
        second_batch_interval_clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_5 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata_5.id(),
        &measurement,
    );
    let helper_prep_state_5 = transcript_5.helper_prepare_transitions[0].prepare_state();
    let ping_pong_leader_message_5 = &transcript_5.leader_prepare_transitions[1].message;
    let report_share_5 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_5.clone(),
        hpke_key.config(),
        &transcript_5.public_share,
        Vec::new(),
        &transcript_5.helper_input_share,
    );

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share_3 = report_share_3.clone();
            let report_share_4 = report_share_4.clone();
            let report_share_5 = report_share_5.clone();
            let helper_prep_state_3 = *helper_prep_state_3;
            let helper_prep_state_4 = *helper_prep_state_4;
            let helper_prep_state_5 = *helper_prep_state_5;
            let report_metadata_3 = report_metadata_3.clone();
            let report_metadata_4 = report_metadata_4.clone();
            let report_metadata_5 = report_metadata_5.clone();

            Box::pin(async move {
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_3.metadata().id(),
                    report_share_3.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_4.metadata().id(),
                    report_share_4.metadata().time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_5.metadata().id(),
                    report_share_5.metadata().time(),
                )
                .await
                .unwrap();

                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id_1,
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
                    aggregation_job_id_1,
                    *report_metadata_3.id(),
                    *report_metadata_3.time(),
                    3,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_3,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id_1,
                    *report_metadata_4.id(),
                    *report_metadata_4.time(),
                    4,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_4,
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id_1,
                    *report_metadata_5.id(),
                    *report_metadata_5.time(),
                    5,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: helper_prep_state_5,
                    },
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([
            PrepareContinue::new(*report_metadata_3.id(), ping_pong_leader_message_3.clone()),
            PrepareContinue::new(*report_metadata_4.id(), ping_pong_leader_message_4.clone()),
            PrepareContinue::new(*report_metadata_5.id(), ping_pong_leader_message_5.clone()),
        ]),
    );

    let _ = post_aggregation_job_and_decode(&task, &aggregation_job_id_1, &request, &handler).await;

    // Map the batch aggregation ordinal value to 0, as it may vary due to sharding, and merge
    // batch aggregations over the same interval. (the task & aggregation parameter will always
    // be the same)
    let first_batch_got_batch_aggregations = datastore
        .run_unnamed_tx(|tx| {
            let task = helper_task.clone();
            let vdaf = vdaf.clone();

            Box::pin(async move {
                Ok(merge_batch_aggregations_by_batch(
                    TimeInterval::get_batch_aggregations_for_collection_identifier::<
                        0,
                        dummy::Vdaf,
                        _,
                    >(
                        tx,
                        task.id(),
                        task.time_precision(),
                        &vdaf,
                        &first_batch_identifier,
                        &aggregation_param,
                    )
                    .await
                    .unwrap(),
                ))
            })
        })
        .await
        .unwrap();

    let first_batch_interval = first_batch_interval.merged_with(&report_time_3).unwrap();
    let first_aggregate_share = vdaf
        .aggregate(
            &aggregation_param,
            [
                &transcript_0.helper_output_share,
                &transcript_1.helper_output_share,
                &transcript_3.helper_output_share,
            ]
            .into_iter()
            .cloned(),
        )
        .unwrap();
    let first_checksum = ReportIdChecksum::for_report_id(report_metadata_0.id())
        .updated_with(report_metadata_1.id())
        .updated_with(report_metadata_3.id());

    assert_eq!(
        first_batch_got_batch_aggregations,
        Vec::from([BatchAggregation::new(
            *task.id(),
            first_batch_identifier,
            aggregation_param,
            0,
            first_batch_interval,
            BatchAggregationState::Aggregating {
                aggregate_share: Some(first_aggregate_share),
                report_count: 3,
                checksum: first_checksum,
                aggregation_jobs_created: 2,
                aggregation_jobs_terminated: 2,
            },
        )]),
    );

    let second_batch_got_batch_aggregations = datastore
        .run_unnamed_tx(|tx| {
            let task = helper_task.clone();
            let vdaf = vdaf.clone();

            Box::pin(async move {
                let mut got_batch_aggregations =
                    TimeInterval::get_batch_aggregations_for_collection_identifier::<
                        0,
                        dummy::Vdaf,
                        _,
                    >(
                        tx,
                        task.id(),
                        task.time_precision(),
                        &vdaf,
                        &second_batch_identifier,
                        &aggregation_param,
                    )
                    .await
                    .unwrap();
                got_batch_aggregations.sort_unstable_by_key(|ba| ba.ord());
                Ok(got_batch_aggregations)
            })
        })
        .await
        .unwrap();
    assert_eq!(
        second_batch_got_batch_aggregations,
        second_batch_want_batch_aggregations
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(3),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_leader_sends_non_continue_or_finish_transition() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    // Prepare parameters.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let report_id = random();
    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        &dummy::Vdaf::new(2),
        task.id(),
        task.vdaf_verify_key().unwrap().as_bytes(),
        &aggregation_param,
        &report_id,
        &13,
    );
    let aggregation_job_id = random();
    let report_metadata = ReportMetadata::new(
        ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        Time::from_seconds_since_epoch(54321),
        Vec::new(),
    );

    // Setup datastore.
    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let transcript = transcript.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_metadata.id(),
                    report_metadata.time(),
                )
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
                    *report_metadata.id(),
                    *report_metadata.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: *transcript.helper_prepare_transitions[0].prepare_state(),
                    },
                ))
                .await
            })
        })
        .await
        .unwrap();

    // Make request.
    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            *report_metadata.id(),
            // An AggregationJobContinueReq should only ever contain Continue or Finished
            PingPongMessage::Initialize {
                prep_share: Vec::new(),
            },
        )]),
    );

    let resp =
        post_aggregation_job_and_decode(&task, &aggregation_job_id, &request, &handler).await;
    let prepare_resps =
        assert_matches!(resp, AggregationJobResp::Finished{prepare_resps} => prepare_resps);
    assert_eq!(prepare_resps.len(), 1);
    assert_eq!(
        prepare_resps[0],
        PrepareResp::new(
            *report_metadata.id(),
            PrepareStepResult::Reject(ReportError::VdafPrepError),
        )
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_prep_step_fails() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair: hpke_key,
        ..
    } = HttpHandlerTest::new().await;

    // Prepare parameters.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let vdaf = dummy::Vdaf::new(2);
    let report_id = random();
    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        &vdaf,
        task.id(),
        task.vdaf_verify_key().unwrap().as_bytes(),
        &aggregation_param,
        &report_id,
        &13,
    );
    let aggregation_job_id = random();
    let report_metadata =
        ReportMetadata::new(report_id, Time::from_seconds_since_epoch(54321), Vec::new());
    let helper_report_share = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata.clone(),
        hpke_key.config(),
        &transcript.public_share,
        Vec::new(),
        &transcript.helper_input_share,
    );

    // Setup datastore.
    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let transcript = transcript.clone();
            let helper_report_share = helper_report_share.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    helper_report_share.metadata().id(),
                    helper_report_share.metadata().time(),
                )
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
                    *report_metadata.id(),
                    *report_metadata.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: *transcript.helper_prepare_transitions[0].prepare_state(),
                    },
                ))
                .await
            })
        })
        .await
        .unwrap();

    // Make request.
    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            *report_metadata.id(),
            PingPongMessage::Continue {
                prep_msg: Vec::new(),
                prep_share: Vec::new(),
            },
        )]),
    );

    let aggregate_resp =
        post_aggregation_job_and_decode(&task, &aggregation_job_id, &request, &handler).await;
    assert_eq!(
        aggregate_resp,
        AggregationJobResp::Finished {
            prepare_resps: Vec::from([PrepareResp::new(
                *report_metadata.id(),
                PrepareStepResult::Reject(ReportError::VdafPrepError),
            )])
        }
    );

    // Check datastore state.
    let (aggregation_job, report_aggregation) = datastore
        .run_unnamed_tx(|tx| {
            let (vdaf, task, report_metadata) =
                (vdaf.clone(), task.clone(), report_metadata.clone());
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
                        &vdaf,
                        &Role::Helper,
                        task.id(),
                        &aggregation_job_id,
                        report_metadata.id(),
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                Ok((aggregation_job, report_aggregation))
            })
        })
        .await
        .unwrap();

    assert_eq!(
        aggregation_job,
        AggregationJob::new(
            *task.id(),
            aggregation_job_id,
            aggregation_param,
            (),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobStep::from(1),
        )
        .with_last_request_hash(aggregation_job.last_request_hash().unwrap())
    );
    assert_eq!(
        report_aggregation,
        ReportAggregation::new(
            *task.id(),
            aggregation_job_id,
            *report_metadata.id(),
            *report_metadata.time(),
            0,
            Some(PrepareResp::new(
                *report_metadata.id(),
                PrepareStepResult::Reject(ReportError::VdafPrepError)
            )),
            ReportAggregationState::Failed {
                report_error: ReportError::VdafPrepError
            },
        )
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_unexpected_transition() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    // Prepare parameters.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let report_id = random();
    let aggregation_param = dummy::AggregationParam(7);
    let transcript = run_vdaf(
        &dummy::Vdaf::new(2),
        task.id(),
        task.vdaf_verify_key().unwrap().as_bytes(),
        &aggregation_param,
        &report_id,
        &13,
    );
    let aggregation_job_id = random();
    let report_metadata =
        ReportMetadata::new(report_id, Time::from_seconds_since_epoch(54321), Vec::new());

    // Setup datastore.
    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let transcript = transcript.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_metadata.id(),
                    report_metadata.time(),
                )
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
                    *report_metadata.id(),
                    *report_metadata.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: *transcript.helper_prepare_transitions[0].prepare_state(),
                    },
                ))
                .await
            })
        })
        .await
        .unwrap();

    // Make request.
    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            ReportId::from(
                [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], // not the same as above
            ),
            PingPongMessage::Continue {
                prep_msg: Vec::new(),
                prep_share: Vec::new(),
            },
        )]),
    );

    post_aggregation_job_expecting_error(
        &task,
        &aggregation_job_id,
        &request,
        &handler,
        Status::BadRequest,
        "urn:ietf:params:ppm:dap:error:invalidMessage",
        "The message type for a response was incorrect or the payload was malformed.",
        None,
    )
    .await;

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_out_of_order_transition() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    // Prepare parameters.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 2 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let report_id_0 = random();
    let aggregation_param = dummy::AggregationParam(7);
    let transcript_0 = run_vdaf(
        &dummy::Vdaf::new(2),
        task.id(),
        task.vdaf_verify_key().unwrap().as_bytes(),
        &aggregation_param,
        &report_id_0,
        &13,
    );
    let report_metadata_0 = ReportMetadata::new(
        report_id_0,
        Time::from_seconds_since_epoch(54321),
        Vec::new(),
    );
    let report_id_1 = random();
    let transcript_1 = run_vdaf(
        &dummy::Vdaf::new(2),
        task.id(),
        task.vdaf_verify_key().unwrap().as_bytes(),
        &aggregation_param,
        &report_id_1,
        &13,
    );
    let report_metadata_1 = ReportMetadata::new(
        report_id_1,
        Time::from_seconds_since_epoch(54321),
        Vec::new(),
    );
    let aggregation_job_id = random();

    // Setup datastore.
    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata_0 = report_metadata_0.clone();
            let report_metadata_1 = report_metadata_1.clone();
            let transcript_0 = transcript_0.clone();
            let transcript_1 = transcript_1.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();

                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_metadata_0.id(),
                    report_metadata_0.time(),
                )
                .await
                .unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_metadata_1.id(),
                    report_metadata_1.time(),
                )
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
                    *report_metadata_0.id(),
                    *report_metadata_0.time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: *transcript_0.helper_prepare_transitions[0].prepare_state(),
                    },
                ))
                .await
                .unwrap();
                tx.put_report_aggregation(&ReportAggregation::<0, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    *report_metadata_1.id(),
                    *report_metadata_1.time(),
                    1,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: *transcript_1.helper_prepare_transitions[0].prepare_state(),
                    },
                ))
                .await
            })
        })
        .await
        .unwrap();

    // Make request.
    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([
            // Report IDs are in opposite order to what was stored in the datastore.
            PrepareContinue::new(
                *report_metadata_1.id(),
                PingPongMessage::Continue {
                    prep_msg: Vec::new(),
                    prep_share: Vec::new(),
                },
            ),
            PrepareContinue::new(
                *report_metadata_0.id(),
                PingPongMessage::Continue {
                    prep_msg: Vec::new(),
                    prep_share: Vec::new(),
                },
            ),
        ]),
    );
    post_aggregation_job_expecting_error(
        &task,
        &aggregation_job_id,
        &request,
        &handler,
        Status::BadRequest,
        "urn:ietf:params:ppm:dap:error:invalidMessage",
        "The message type for a response was incorrect or the payload was malformed.",
        None,
    )
    .await;

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_continue_for_non_waiting_aggregation() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    // Prepare parameters.
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let aggregation_job_id = random();
    let report_metadata = ReportMetadata::new(
        ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        Time::from_seconds_since_epoch(54321),
        Vec::new(),
    );

    // Setup datastore.
    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_metadata.id(),
                    report_metadata.time(),
                )
                .await
                .unwrap();
                tx.put_aggregation_job(&AggregationJob::<0, TimeInterval, dummy::Vdaf>::new(
                    *helper_task.id(),
                    aggregation_job_id,
                    dummy::AggregationParam(0),
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
                    *report_metadata.id(),
                    *report_metadata.time(),
                    0,
                    None,
                    ReportAggregationState::Failed {
                        report_error: ReportError::VdafPrepError,
                    },
                ))
                .await
            })
        })
        .await
        .unwrap();

    // Make request.
    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
            PingPongMessage::Continue {
                prep_msg: Vec::new(),
                prep_share: Vec::new(),
            },
        )]),
    );
    post_aggregation_job_expecting_error(
        &task,
        &aggregation_job_id,
        &request,
        &handler,
        Status::BadRequest,
        "urn:ietf:params:ppm:dap:error:invalidMessage",
        "The message type for a response was incorrect or the payload was malformed.",
        None,
    )
    .await;

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}
