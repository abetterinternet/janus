#![allow(clippy::unit_arg)] // allow reference to dummy::Vdaf's public share, which has the unit type

use crate::aggregator::{
    aggregation_job_init::test_util::{put_aggregation_job, PrepareInitGenerator},
    empty_batch_aggregations,
    http_handlers::test_util::{decode_response_body, take_problem_details, HttpHandlerTest},
    test_util::{
        assert_task_aggregation_counter, generate_helper_report_share,
        generate_helper_report_share_for_plaintext, BATCH_AGGREGATION_SHARD_COUNT,
    },
};
use assert_matches::assert_matches;
use futures::future::try_join_all;
use janus_aggregator_core::{
    datastore::models::{
        AggregationJobState, BatchAggregation, BatchAggregationState, ReportAggregationState,
        TaskAggregationCounter,
    },
    task::{test_util::TaskBuilder, AggregationMode, BatchMode, VerifyKey},
};
use janus_core::{
    auth_tokens::AuthenticationToken,
    hpke::HpkeKeypair,
    report_id::ReportIdChecksumExt,
    test_util::run_vdaf,
    time::{Clock, MockClock, TimeExt},
    vdaf::VdafInstance,
};
use janus_messages::{
    batch_mode::{LeaderSelected, TimeInterval},
    AggregationJobId, AggregationJobInitializeReq, AggregationJobResp, Duration, Extension,
    ExtensionType, HpkeCiphertext, HpkeConfigId, InputShareAad, Interval, PartialBatchSelector,
    PrepareInit, PrepareStepResult, ReportError, ReportIdChecksum, ReportMetadata, ReportShare,
    Role, Time,
};
use prio::{codec::Encode, vdaf::dummy};
use rand::random;
use serde_json::json;
use trillium::{KnownHeaderName, Status};
use trillium_testing::{assert_headers, prelude::put, TestConn};

#[tokio::test]
async fn aggregate_leader() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .build();
    datastore
        .put_aggregator_task(&task.leader_view().unwrap())
        .await
        .unwrap();

    let request = AggregationJobInitializeReq::new(
        Vec::new(),
        PartialBatchSelector::new_time_interval(),
        Vec::new(),
    );
    let aggregation_job_id: AggregationJobId = random();

    let mut test_conn = put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
    assert!(!test_conn.status().unwrap().is_success());

    let problem_details = take_problem_details(&mut test_conn).await;
    assert_eq!(
        problem_details,
        json!({
            "status": 400,
            "type": "urn:ietf:params:ppm:dap:error:unrecognizedTask",
            "title": "An endpoint received a message with an unknown task ID.",
            "taskid": format!("{}", task.id()),
        })
    );
    assert_eq!(
        problem_details
            .as_object()
            .unwrap()
            .get("status")
            .unwrap()
            .as_u64()
            .unwrap(),
        test_conn.status().unwrap() as u16 as u64
    );

    // Check that CORS headers don't bleed over to other routes.
    assert_headers!(
        &test_conn,
        "access-control-allow-origin" => None,
        "access-control-allow-methods" => None,
        "access-control-max-age" => None,
    );

    let test_conn = TestConn::build(
        trillium::Method::Options,
        task.aggregation_job_uri(&aggregation_job_id, None)
            .unwrap()
            .path(),
        (),
    )
    .with_request_header(KnownHeaderName::Origin, "https://example.com/")
    .with_request_header(KnownHeaderName::AccessControlRequestMethod, "PUT")
    .run_async(&handler)
    .await;
    assert_headers!(&test_conn, "access-control-allow-methods" => None);
}

#[tokio::test]
async fn aggregate_wrong_agg_auth_token() {
    let HttpHandlerTest {
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        ..
    } = HttpHandlerTest::new().await;

    let dap_auth_token = AuthenticationToken::DapAuth(random());

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Prio3Count,
    )
    .with_aggregator_auth_token(dap_auth_token.clone())
    .build();

    datastore
        .put_aggregator_task(&task.helper_view().unwrap())
        .await
        .unwrap();

    let request = AggregationJobInitializeReq::new(
        Vec::new(),
        PartialBatchSelector::new_time_interval(),
        Vec::new(),
    );
    let aggregation_job_id: AggregationJobId = random();

    let wrong_token_value = AuthenticationToken::DapAuth(random());

    // Send the right token, but the wrong format: convert the DAP auth token to an equivalent
    // Bearer token, which should be rejected.
    let wrong_token_format =
        AuthenticationToken::new_bearer_token_from_bytes(dap_auth_token.as_ref()).unwrap();

    for auth_tokens in [
        Vec::from([wrong_token_value.clone()]),
        Vec::from([wrong_token_format]),
        Vec::from([]),
        Vec::from([dap_auth_token, wrong_token_value]),
    ] {
        let mut test_conn = put(task
            .aggregation_job_uri(&aggregation_job_id, None)
            .unwrap()
            .path())
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<TimeInterval>::MEDIA_TYPE,
        )
        .with_request_body(request.get_encoded().unwrap());

        for auth_token in auth_tokens {
            let (auth_header, auth_value) = auth_token.request_authentication();
            test_conn = test_conn.with_request_header(auth_header, auth_value);
        }

        let mut test_conn = test_conn.run_async(&handler).await;

        let want_status = u16::from(Status::Forbidden);
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": want_status,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", task.id()),
            })
        );
        assert_eq!(want_status, test_conn.status().unwrap() as u16);
    }
}

#[tokio::test]
async fn aggregate_init_sync() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();

    let helper_task = task.helper_view().unwrap();

    let vdaf = dummy::Vdaf::new(1);
    let verify_key: VerifyKey<0> = task.vdaf_verify_key().unwrap();
    let measurement = 0;
    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        vdaf.clone(),
        dummy::AggregationParam(0),
    );

    // prepare_init_0 is a "happy path" report.
    let (prepare_init_0, transcript_0) = prep_init_generator.next(&measurement);

    // report_share_1 fails decryption.
    let (prepare_init_1, transcript_1) = prep_init_generator.next(&measurement);

    let encrypted_input_share = prepare_init_1.report_share().encrypted_input_share();
    let mut corrupted_payload = encrypted_input_share.payload().to_vec();
    corrupted_payload[0] ^= 0xFF;
    let corrupted_input_share = HpkeCiphertext::new(
        *encrypted_input_share.config_id(),
        encrypted_input_share.encapsulated_key().to_vec(),
        corrupted_payload,
    );

    let prepare_init_1 = PrepareInit::new(
        ReportShare::new(
            prepare_init_1.report_share().metadata().clone(),
            transcript_1.public_share.get_encoded().unwrap(),
            corrupted_input_share,
        ),
        prepare_init_1.message().clone(),
    );

    // prepare_init_2 fails decoding due to an issue with the input share.
    let (prepare_init_2, transcript_2) = prep_init_generator.next(&measurement);

    let mut input_share_bytes = transcript_2.helper_input_share.get_encoded().unwrap();
    input_share_bytes.push(0); // can no longer be decoded.
    let report_share_2 = generate_helper_report_share_for_plaintext(
        prepare_init_2.report_share().metadata().clone(),
        hpke_keypair.config(),
        transcript_2.public_share.get_encoded().unwrap(),
        &input_share_bytes,
        &InputShareAad::new(
            *task.id(),
            prepare_init_2.report_share().metadata().clone(),
            transcript_2.public_share.get_encoded().unwrap(),
        )
        .get_encoded()
        .unwrap(),
    );

    let prepare_init_2 = PrepareInit::new(report_share_2, prepare_init_2.message().clone());

    // prepare_init_3 has an unknown HPKE config ID.
    let (prepare_init_3, transcript_3) = prep_init_generator.next(&measurement);

    let unused_hpke_config_id =
        HpkeConfigId::from(u8::from(*hpke_keypair.config().id()).wrapping_add(1));
    let wrong_hpke_config = HpkeKeypair::test_with_id(unused_hpke_config_id)
        .config()
        .clone();

    let report_share_3 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        prepare_init_3.report_share().metadata().clone(),
        &wrong_hpke_config,
        &transcript_3.public_share,
        Vec::new(),
        &transcript_3.helper_input_share,
    );

    let prepare_init_3 = PrepareInit::new(report_share_3, prepare_init_3.message().clone());

    // prepare_init_4 has already been aggregated in another aggregation job, with the same
    // aggregation parameter.
    let (prepare_init_4, _) = prep_init_generator.next(&measurement);

    // prepare_init_5 falls into a batch that has already been collected.
    let past_clock = MockClock::new(Time::from_seconds_since_epoch(
        task.time_precision().as_seconds() / 2,
    ));
    let report_metadata_5 = ReportMetadata::new(
        random(),
        past_clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_5 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &dummy::AggregationParam(0),
        report_metadata_5.id(),
        &measurement,
    );
    let report_share_5 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_5,
        hpke_keypair.config(),
        &transcript_5.public_share,
        Vec::new(),
        &transcript_5.helper_input_share,
    );

    let prepare_init_5 = PrepareInit::new(
        report_share_5,
        transcript_5.leader_prepare_transitions[0].message.clone(),
    );

    // prepare_init_6 fails decoding due to an issue with the public share.
    let public_share_6 = Vec::from([0]);
    let report_metadata_6 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_6 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &dummy::AggregationParam(0),
        report_metadata_6.id(),
        &measurement,
    );
    let report_share_6 = generate_helper_report_share_for_plaintext(
        report_metadata_6.clone(),
        hpke_keypair.config(),
        public_share_6.clone(),
        &transcript_6.helper_input_share.get_encoded().unwrap(),
        &InputShareAad::new(*task.id(), report_metadata_6, public_share_6)
            .get_encoded()
            .unwrap(),
    );

    let prepare_init_6 = PrepareInit::new(
        report_share_6,
        transcript_6.leader_prepare_transitions[0].message.clone(),
    );

    // prepare_init_7 fails due to having repeated public extensions.
    let report_metadata_7 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::from([
            Extension::new(ExtensionType::Tbd, Vec::new()),
            Extension::new(ExtensionType::Tbd, Vec::new()),
        ]),
    );
    let transcript_7 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &dummy::AggregationParam(0),
        report_metadata_7.id(),
        &measurement,
    );
    let report_share_7 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_7,
        hpke_keypair.config(),
        &transcript_7.public_share,
        Vec::new(),
        &transcript_7.helper_input_share,
    );

    let prepare_init_7 = PrepareInit::new(
        report_share_7,
        transcript_7.leader_prepare_transitions[0].message.clone(),
    );

    // prepare_init_8 fails due to having repeated private extensions.
    let report_metadata_8 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::new(),
    );
    let transcript_8 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &dummy::AggregationParam(0),
        report_metadata_8.id(),
        &measurement,
    );
    let report_share_8 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_8,
        hpke_keypair.config(),
        &transcript_8.public_share,
        Vec::from([
            Extension::new(ExtensionType::Tbd, Vec::new()),
            Extension::new(ExtensionType::Tbd, Vec::new()),
        ]),
        &transcript_8.helper_input_share,
    );

    let prepare_init_8 = PrepareInit::new(
        report_share_8,
        transcript_8.leader_prepare_transitions[0].message.clone(),
    );

    // prepare_init_9 fails due to having repeated extensions between the public & private
    // extensions.
    let report_metadata_9 = ReportMetadata::new(
        random(),
        clock
            .now()
            .to_batch_interval_start(task.time_precision())
            .unwrap(),
        Vec::from([Extension::new(ExtensionType::Tbd, Vec::new())]),
    );
    let transcript_9 = run_vdaf(
        &vdaf,
        task.id(),
        verify_key.as_bytes(),
        &dummy::AggregationParam(0),
        report_metadata_9.id(),
        &measurement,
    );
    let report_share_9 = generate_helper_report_share::<dummy::Vdaf>(
        *task.id(),
        report_metadata_9,
        hpke_keypair.config(),
        &transcript_9.public_share,
        Vec::from([Extension::new(ExtensionType::Tbd, Vec::new())]),
        &transcript_9.helper_input_share,
    );

    let prepare_init_9 = PrepareInit::new(
        report_share_9,
        transcript_9.leader_prepare_transitions[0].message.clone(),
    );

    let mut batch_aggregations_results = Vec::new();
    let mut aggregation_jobs_results = Vec::new();
    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share_4 = prepare_init_4.report_share().clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();

                // report_share_4 is already in the datastore as it was referenced by an existing
                // aggregation job.
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_4.metadata().id(),
                    report_share_4.metadata().time(),
                )
                .await
                .unwrap();

                // Write collected batch aggregations for the interval that report_share_5 falls
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
                        &dummy::AggregationParam(0),
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

    let aggregation_param = dummy::AggregationParam(0);
    let request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_time_interval(),
        Vec::from([
            prepare_init_0.clone(),
            prepare_init_1.clone(),
            prepare_init_2.clone(),
            prepare_init_3.clone(),
            prepare_init_4.clone(),
            prepare_init_5.clone(),
            prepare_init_6.clone(),
            prepare_init_7.clone(),
            prepare_init_8.clone(),
            prepare_init_9.clone(),
        ]),
    );

    // Send request, parse response. Do this twice to prove that the request is idempotent.
    let aggregation_job_id: AggregationJobId = random();
    for _ in 0..2 {
        let mut test_conn =
            put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
        assert_eq!(test_conn.status(), Some(Status::Created));
        assert_headers!(
            &test_conn,
            "content-type" => (AggregationJobResp::MEDIA_TYPE)
        );
        let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
        let prepare_resps = assert_matches!(
            aggregate_resp,
            AggregationJobResp::Finished { prepare_resps } => prepare_resps
        );

        // Validate response.
        assert_eq!(prepare_resps.len(), 10);

        let prepare_step_0 = prepare_resps.first().unwrap();
        assert_eq!(
            prepare_step_0.report_id(),
            prepare_init_0.report_share().metadata().id()
        );
        assert_matches!(prepare_step_0.result(), PrepareStepResult::Continue { message } => {
            assert_eq!(message, &transcript_0.helper_prepare_transitions[0].message);
        });

        let prepare_step_1 = prepare_resps.get(1).unwrap();
        assert_eq!(
            prepare_step_1.report_id(),
            prepare_init_1.report_share().metadata().id()
        );
        assert_matches!(
            prepare_step_1.result(),
            &PrepareStepResult::Reject(ReportError::HpkeDecryptError)
        );

        let prepare_step_2 = prepare_resps.get(2).unwrap();
        assert_eq!(
            prepare_step_2.report_id(),
            prepare_init_2.report_share().metadata().id()
        );
        assert_matches!(
            prepare_step_2.result(),
            &PrepareStepResult::Reject(ReportError::InvalidMessage)
        );

        let prepare_step_3 = prepare_resps.get(3).unwrap();
        assert_eq!(
            prepare_step_3.report_id(),
            prepare_init_3.report_share().metadata().id()
        );
        assert_matches!(
            prepare_step_3.result(),
            &PrepareStepResult::Reject(ReportError::HpkeUnknownConfigId)
        );

        let prepare_step_4 = prepare_resps.get(4).unwrap();
        assert_eq!(
            prepare_step_4.report_id(),
            prepare_init_4.report_share().metadata().id()
        );
        assert_eq!(
            prepare_step_4.result(),
            &PrepareStepResult::Reject(ReportError::ReportReplayed)
        );

        let prepare_step_5 = prepare_resps.get(5).unwrap();
        assert_eq!(
            prepare_step_5.report_id(),
            prepare_init_5.report_share().metadata().id()
        );
        assert_eq!(
            prepare_step_5.result(),
            &PrepareStepResult::Reject(ReportError::BatchCollected)
        );

        let prepare_step_6 = prepare_resps.get(6).unwrap();
        assert_eq!(
            prepare_step_6.report_id(),
            prepare_init_6.report_share().metadata().id()
        );
        assert_eq!(
            prepare_step_6.result(),
            &PrepareStepResult::Reject(ReportError::InvalidMessage),
        );

        let prepare_step_7 = prepare_resps.get(7).unwrap();
        assert_eq!(
            prepare_step_7.report_id(),
            prepare_init_7.report_share().metadata().id()
        );
        assert_eq!(
            prepare_step_7.result(),
            &PrepareStepResult::Reject(ReportError::InvalidMessage),
        );

        let prepare_step_8 = prepare_resps.get(8).unwrap();
        assert_eq!(
            prepare_step_8.report_id(),
            prepare_init_8.report_share().metadata().id()
        );
        assert_eq!(
            prepare_step_8.result(),
            &PrepareStepResult::Reject(ReportError::InvalidMessage),
        );

        let prepare_step_9 = prepare_resps.get(9).unwrap();
        assert_eq!(
            prepare_step_9.report_id(),
            prepare_init_9.report_share().metadata().id()
        );
        assert_eq!(
            prepare_step_9.result(),
            &PrepareStepResult::Reject(ReportError::InvalidMessage),
        );

        // Check aggregation job in datastore.
        let (aggregation_jobs, batch_aggregations) = datastore
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                let vdaf = vdaf.clone();
                Box::pin(async move {
                    Ok((
                        tx.get_aggregation_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(task.id())
                            .await
                            .unwrap(),
                        tx.get_batch_aggregations_for_task::<0, TimeInterval, _>(&vdaf, task.id())
                            .await
                            .unwrap(),
                    ))
                })
            })
            .await
            .unwrap();

        assert_eq!(aggregation_jobs.len(), 1);

        let mut saw_new_aggregation_job = false;
        for aggregation_job in &aggregation_jobs {
            if aggregation_job.task_id().eq(task.id())
                && aggregation_job.id().eq(&aggregation_job_id)
                && aggregation_job.partial_batch_identifier().eq(&())
                && aggregation_job.state().eq(&AggregationJobState::Finished)
            {
                saw_new_aggregation_job = true;
            }
        }
        assert!(saw_new_aggregation_job);

        aggregation_jobs_results.push(aggregation_jobs);
        batch_aggregations_results.push(batch_aggregations);
    }

    assert!(aggregation_jobs_results.windows(2).all(|v| v[0] == v[1]));
    assert!(batch_aggregations_results.windows(2).all(|v| v[0] == v[1]));

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(1),
    )
    .await;
}

#[tokio::test]
async fn aggregate_init_async() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Asynchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();

    let helper_task = task.helper_view().unwrap();

    let vdaf = dummy::Vdaf::new(1);
    let measurement = 0;
    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        vdaf.clone(),
        dummy::AggregationParam(0),
    );

    // prepare_init_0 is a "happy path" report.
    let (prepare_init_0, _) = prep_init_generator.next(&measurement);

    // prepare_init_1 has already been aggregated in another aggregation job, with the same
    // aggregation parameter.
    let (prepare_init_1, _) = prep_init_generator.next(&measurement);

    datastore
        .run_unnamed_tx(|tx| {
            let helper_task = helper_task.clone();
            let report_share_1 = prepare_init_1.report_share().clone();

            Box::pin(async move {
                tx.put_aggregator_task(&helper_task).await.unwrap();

                // report_share_1 is already in the datastore as it was referenced by an existing
                // aggregation job.
                tx.put_scrubbed_report(
                    helper_task.id(),
                    report_share_1.metadata().id(),
                    report_share_1.metadata().time(),
                )
                .await
                .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

    let aggregation_param = dummy::AggregationParam(0);
    let request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_time_interval(),
        Vec::from([prepare_init_0.clone(), prepare_init_1.clone()]),
    );

    // Send request, parse response. Do this twice to prove that the request is idempotent.
    let aggregation_job_id: AggregationJobId = random();
    let mut aggregation_jobs_results = Vec::new();
    let mut report_aggregations_results = Vec::new();
    let mut batch_aggregations_results = Vec::new();
    for _ in 0..2 {
        let mut test_conn =
            put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
        assert_eq!(test_conn.status(), Some(Status::Created));
        assert_headers!(
            &test_conn,
            "content-type" => (AggregationJobResp::MEDIA_TYPE)
        );
        let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
        assert_matches!(aggregate_resp, AggregationJobResp::Processing);

        // Check aggregation job in datastore.
        let (aggregation_jobs, report_aggregations, batch_aggregations) = datastore
            .run_unnamed_tx(|tx| {
                let task = task.clone();
                let vdaf = vdaf.clone();
                Box::pin(async move {
                    Ok((
                        tx.get_aggregation_jobs_for_task::<0, TimeInterval, dummy::Vdaf>(task.id())
                            .await
                            .unwrap(),
                        tx.get_report_aggregations_for_aggregation_job::<0, dummy::Vdaf>(
                            &vdaf,
                            &Role::Helper,
                            task.id(),
                            &aggregation_job_id,
                            &aggregation_param,
                        )
                        .await
                        .unwrap(),
                        tx.get_batch_aggregations_for_task::<0, TimeInterval, _>(&vdaf, task.id())
                            .await
                            .unwrap(),
                    ))
                })
            })
            .await
            .unwrap();

        assert_eq!(aggregation_jobs.len(), 1);

        assert_eq!(aggregation_jobs[0].task_id(), task.id());
        assert_eq!(aggregation_jobs[0].id(), &aggregation_job_id);
        assert_eq!(aggregation_jobs[0].partial_batch_identifier(), &());
        assert_eq!(aggregation_jobs[0].state(), &AggregationJobState::Active);

        assert_eq!(report_aggregations.len(), 2);

        assert_eq!(
            report_aggregations[0].report_id(),
            prepare_init_0.report_share().metadata().id()
        );
        assert_eq!(
            report_aggregations[0].state(),
            &ReportAggregationState::HelperInitProcessing {
                prepare_init: prepare_init_0.clone(),
                require_taskbind_extension: false
            }
        );

        assert_eq!(
            report_aggregations[1].report_id(),
            prepare_init_1.report_share().metadata().id()
        );
        assert_eq!(
            report_aggregations[1].state(),
            &ReportAggregationState::Failed {
                report_error: ReportError::ReportReplayed
            }
        );

        aggregation_jobs_results.push(aggregation_jobs);
        report_aggregations_results.push(report_aggregations);
        batch_aggregations_results.push(batch_aggregations);
    }

    assert!(aggregation_jobs_results.windows(2).all(|v| v[0] == v[1]));
    assert!(report_aggregations_results.windows(2).all(|v| v[0] == v[1]));
    assert!(batch_aggregations_results.windows(2).all(|v| v[0] == v[1]));

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_init_batch_already_collected() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();

    let helper_task = task.helper_view().unwrap();
    datastore.put_aggregator_task(&helper_task).await.unwrap();

    let vdaf = dummy::Vdaf::new(1);
    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        vdaf.clone(),
        dummy::AggregationParam(0),
    );

    let (prepare_init, _) = prep_init_generator.next(&0);

    let aggregation_param = dummy::AggregationParam(0);
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([prepare_init.clone()]),
    );

    // Pretend that we're another concurrently running process: insert some aggregations to the
    // same batch ID and mark them collected.
    datastore
        .run_unnamed_tx(|tx| {
            let task = task.clone();
            let timestamp = *prepare_init.report_share().metadata().time();
            Box::pin(async move {
                let interval = Interval::new(timestamp, Duration::from_seconds(1)).unwrap();

                // Insert for all possible shards, since we non-deterministically assign shards
                // to batches on insertion.
                for shard in 0..BATCH_AGGREGATION_SHARD_COUNT {
                    let batch_aggregation = BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                        *task.id(),
                        batch_id,
                        dummy::AggregationParam(0),
                        shard,
                        interval,
                        BatchAggregationState::Collected {
                            aggregate_share: Some(dummy::OutputShare(0).into()),
                            report_count: 1,
                            checksum: ReportIdChecksum::for_report_id(&random()),
                            aggregation_jobs_created: 1,
                            aggregation_jobs_terminated: 1,
                        },
                    );
                    tx.put_batch_aggregation(&batch_aggregation).await.unwrap();
                }

                Ok(())
            })
        })
        .await
        .unwrap();

    let aggregation_job_id: AggregationJobId = random();
    let (header, value) = task.aggregator_auth_token().request_authentication();
    let mut test_conn = put(task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(header, value)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_body(request.get_encoded().unwrap())
    .run_async(&handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Created));
    let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
    let prepare_resps = assert_matches!(
        aggregate_resp,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );

    let prepare_step = prepare_resps.first().unwrap();
    assert_eq!(
        prepare_step.report_id(),
        prepare_init.report_share().metadata().id()
    );
    assert_eq!(
        prepare_step.result(),
        &PrepareStepResult::Reject(ReportError::BatchCollected)
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_init_prep_init_failed() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::FakeFailsPrepInit,
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        dummy::Vdaf::new(1),
        dummy::AggregationParam(0),
    );

    datastore.put_aggregator_task(&helper_task).await.unwrap();

    let (prepare_init, _) = prep_init_generator.next(&0);
    let request = AggregationJobInitializeReq::new(
        dummy::AggregationParam(0).get_encoded().unwrap(),
        PartialBatchSelector::new_time_interval(),
        Vec::from([prepare_init.clone()]),
    );

    // Send request, and parse response.
    let aggregation_job_id: AggregationJobId = random();
    let mut test_conn = put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
    assert_eq!(test_conn.status(), Some(Status::Created));
    assert_headers!(
        &test_conn,
        "content-type" => (AggregationJobResp::MEDIA_TYPE)
    );
    let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
    let prepare_resps = assert_matches!(
        aggregate_resp,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );

    // Validate response.
    assert_eq!(prepare_resps.len(), 1);

    let prepare_step = prepare_resps.first().unwrap();
    assert_eq!(
        prepare_step.report_id(),
        prepare_init.report_share().metadata().id()
    );
    assert_matches!(
        prepare_step.result(),
        &PrepareStepResult::Reject(ReportError::VdafPrepError)
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_init_prep_step_failed() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::FakeFailsPrepStep,
    )
    .build();
    let helper_task = task.helper_view().unwrap();
    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        dummy::Vdaf::new(1),
        dummy::AggregationParam(0),
    );

    datastore.put_aggregator_task(&helper_task).await.unwrap();

    let (prepare_init, _) = prep_init_generator.next(&0);
    let request = AggregationJobInitializeReq::new(
        dummy::AggregationParam(0).get_encoded().unwrap(),
        PartialBatchSelector::new_time_interval(),
        Vec::from([prepare_init.clone()]),
    );

    let aggregation_job_id: AggregationJobId = random();
    let mut test_conn = put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;
    assert_eq!(test_conn.status(), Some(Status::Created));
    assert_headers!(
        &test_conn,
        "content-type" => (AggregationJobResp::MEDIA_TYPE)
    );
    let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
    let prepare_resps = assert_matches!(
        aggregate_resp,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );

    // Validate response.
    assert_eq!(prepare_resps.len(), 1);

    let prepare_step = prepare_resps.first().unwrap();
    assert_eq!(
        prepare_step.report_id(),
        prepare_init.report_share().metadata().id()
    );
    assert_matches!(
        prepare_step.result(),
        &PrepareStepResult::Reject(ReportError::VdafPrepError)
    );

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}

#[tokio::test]
async fn aggregate_init_duplicated_report_id() {
    let HttpHandlerTest {
        clock,
        ephemeral_datastore: _ephemeral_datastore,
        datastore,
        handler,
        hpke_keypair,
        ..
    } = HttpHandlerTest::new().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build();

    let helper_task = task.helper_view().unwrap();
    let prep_init_generator = PrepareInitGenerator::new(
        clock.clone(),
        helper_task.clone(),
        hpke_keypair.config().clone(),
        dummy::Vdaf::new(1),
        dummy::AggregationParam(0),
    );

    datastore.put_aggregator_task(&helper_task).await.unwrap();

    let (prepare_init, _) = prep_init_generator.next(&0);

    let request = AggregationJobInitializeReq::new(
        dummy::AggregationParam(0).get_encoded().unwrap(),
        PartialBatchSelector::new_time_interval(),
        Vec::from([prepare_init.clone(), prepare_init]),
    );
    let aggregation_job_id: AggregationJobId = random();

    let mut test_conn = put_aggregation_job(&task, &aggregation_job_id, &request, &handler).await;

    let want_status = 400;
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": want_status,
            "type": "urn:ietf:params:ppm:dap:error:invalidMessage",
            "title": "The message type for a response was incorrect or the payload was malformed.",
            "taskid": format!("{}", task.id()),
        })
    );
    assert_eq!(want_status, test_conn.status().unwrap());

    assert_task_aggregation_counter(
        &datastore,
        *task.id(),
        TaskAggregationCounter::new_with_values(0),
    )
    .await;
}
