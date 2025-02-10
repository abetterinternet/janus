use crate::aggregator::{
    http_handlers::test_util::{decode_response_body, HttpHandlerTest},
    test_util::generate_helper_report_share,
};
use janus_aggregator_core::{
    datastore::models::{
        AggregationJob, AggregationJobState, ReportAggregation, ReportAggregationState,
    },
    task::{
        test_util::{Task, TaskBuilder},
        AggregationMode, BatchMode, VerifyKey,
    },
};
use janus_core::{
    test_util::run_vdaf,
    time::{Clock as _, TimeExt as _},
    vdaf::VdafInstance,
};
use janus_messages::{
    batch_mode::TimeInterval, AggregationJobId, AggregationJobResp, AggregationJobStep, Duration,
    Interval, PrepareInit, PrepareResp, PrepareStepResult, ReportMetadata,
};
use prio::vdaf::dummy;
use rand::random;
use std::sync::Arc;
use trillium::{Handler, Status};
use trillium_testing::{assert_headers, prelude::get, TestConn};

#[tokio::test]
async fn aggregation_job_get_ready() {
    // Prepare state.
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(1));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let report_metadata = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );
    let helper_message = &transcript.helper_prepare_transitions[0].message;

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let helper_message = helper_message.clone();

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
                    Interval::new(*report_metadata.time(), Duration::from_seconds(1)).unwrap(),
                    AggregationJobState::AwaitingRequest,
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
                    Some(PrepareResp::new(
                        *report_metadata.id(),
                        PrepareStepResult::Continue {
                            message: helper_message,
                        },
                    )),
                    ReportAggregationState::Finished,
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    // Send request.
    let aggregate_resp = get_aggregation_job_and_decode(
        &task,
        &aggregation_job_id,
        Some(AggregationJobStep::from(0)),
        &handler,
    )
    .await;

    // Validate result.
    assert_eq!(
        aggregate_resp,
        AggregationJobResp::Finished {
            prepare_resps: Vec::from([PrepareResp::new(
                *report_metadata.id(),
                PrepareStepResult::Continue {
                    message: helper_message.clone(),
                }
            )])
        }
    );
}

#[tokio::test]
async fn aggregation_job_get_unready() {
    // Prepare state.
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(1));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let report_metadata = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );
    let leader_message = &transcript.leader_prepare_transitions[0].message;
    let report_share = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata.clone(),
        hpke_keypair.config(),
        &transcript.public_share,
        Vec::new(),
        &transcript.helper_input_share,
    );

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let report_share = report_share.clone();
            let leader_message = leader_message.clone();

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
                    Interval::new(*report_metadata.time(), Duration::from_seconds(1)).unwrap(),
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
                    ReportAggregationState::HelperInitProcessing {
                        prepare_init: PrepareInit::new(report_share, leader_message),
                        require_taskbind_extension: false,
                    },
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    // Send request.
    let aggregate_resp = get_aggregation_job_and_decode(
        &task,
        &aggregation_job_id,
        Some(AggregationJobStep::from(0)),
        &handler,
    )
    .await;

    // Validate result.
    assert_eq!(aggregate_resp, AggregationJobResp::Processing);
}

#[tokio::test]
async fn aggregation_job_get_wrong_step() {
    // Prepare state.
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(1));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let report_metadata = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );
    let helper_message = &transcript.helper_prepare_transitions[0].message;

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let helper_message = helper_message.clone();

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
                    Interval::new(*report_metadata.time(), Duration::from_seconds(1)).unwrap(),
                    AggregationJobState::AwaitingRequest,
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
                    Some(PrepareResp::new(
                        *report_metadata.id(),
                        PrepareStepResult::Continue {
                            message: helper_message,
                        },
                    )),
                    ReportAggregationState::Finished,
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    // Send request.
    let test_conn = get_aggregation_job(
        &task,
        &aggregation_job_id,
        Some(AggregationJobStep::from(1)),
        &handler,
    )
    .await;

    // Validate result.
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
}

#[tokio::test]
async fn aggregation_job_get_missing_step() {
    // Prepare state.
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(1));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let report_metadata = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );
    let helper_message = &transcript.helper_prepare_transitions[0].message;

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let helper_message = helper_message.clone();

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
                    Interval::new(*report_metadata.time(), Duration::from_seconds(1)).unwrap(),
                    AggregationJobState::AwaitingRequest,
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
                    Some(PrepareResp::new(
                        *report_metadata.id(),
                        PrepareStepResult::Continue {
                            message: helper_message,
                        },
                    )),
                    ReportAggregationState::Finished,
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    // Send request.
    let test_conn = get_aggregation_job(&task, &aggregation_job_id, None, &handler).await;

    // Validate result.
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
}

#[tokio::test]
async fn aggregation_job_get_sync() {
    // Prepare state.
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    let aggregation_job_id = random();
    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();
    let helper_task = task.helper_view().unwrap();

    let vdaf = Arc::new(dummy::Vdaf::new(1));
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 13;
    let aggregation_param = dummy::AggregationParam(7);

    let report_metadata = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript = run_vdaf(
        vdaf.as_ref(),
        task.id(),
        verify_key.as_bytes(),
        &aggregation_param,
        report_metadata.id(),
        &measurement,
    );
    let helper_message = &transcript.helper_prepare_transitions[0].message;

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_metadata = report_metadata.clone();
            let helper_message = helper_message.clone();

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
                    Interval::new(*report_metadata.time(), Duration::from_seconds(1)).unwrap(),
                    AggregationJobState::AwaitingRequest,
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
                    Some(PrepareResp::new(
                        *report_metadata.id(),
                        PrepareStepResult::Continue {
                            message: helper_message,
                        },
                    )),
                    ReportAggregationState::Finished,
                ))
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    // Send request.
    let test_conn = get_aggregation_job(
        &task,
        &aggregation_job_id,
        Some(AggregationJobStep::from(0)),
        &handler,
    )
    .await;

    // Validate result.
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
}

async fn get_aggregation_job(
    task: &Task,
    aggregation_job_id: &AggregationJobId,
    step: Option<AggregationJobStep>,
    handler: &impl Handler,
) -> TestConn {
    let uri = task.aggregation_job_uri(aggregation_job_id, step).unwrap();
    let uri = match uri.query() {
        Some(query) => format!("{}?{}", uri.path(), query),
        None => uri.path().to_string(),
    };

    let (header, value) = task.aggregator_auth_token().request_authentication();
    get(uri)
        .with_request_header(header, value)
        .run_async(handler)
        .await
}

async fn get_aggregation_job_and_decode(
    task: &Task,
    aggregation_job_id: &AggregationJobId,
    step: Option<AggregationJobStep>,
    handler: &impl Handler,
) -> AggregationJobResp {
    let mut test_conn = get_aggregation_job(task, aggregation_job_id, step, handler).await;
    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    decode_response_body::<AggregationJobResp>(&mut test_conn).await
}
