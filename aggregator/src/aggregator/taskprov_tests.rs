use crate::{
    aggregator::{
        aggregate_init_tests::PrepareInitGenerator,
        http_handlers::{
            aggregator_handler,
            test_util::{decode_response_body, take_problem_details},
        },
        Config,
    },
    config::TaskprovConfig,
};
use assert_matches::assert_matches;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use janus_aggregator_core::{
    datastore::{
        models::{
            AggregateShareJob, AggregationJob, AggregationJobState, Batch, BatchAggregation,
            BatchAggregationState, BatchState, ReportAggregation, ReportAggregationState,
        },
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{
        test_util::{Task, TaskBuilder},
        QueryType,
    },
    taskprov::{test_util::PeerAggregatorBuilder, PeerAggregator},
    test_util::noop_meter,
};
use janus_core::{
    hpke::{
        self, test_util::generate_test_hpke_config_and_private_key, HpkeApplicationInfo,
        HpkeKeypair, Label,
    },
    report_id::ReportIdChecksumExt,
    taskprov::TASKPROV_HEADER,
    test_util::{install_test_trace_subscriber, VdafTranscript},
    time::{Clock, DurationExt, MockClock, TimeExt},
    vdaf::VERIFY_KEY_LENGTH,
};
use janus_messages::{
    codec::{Decode, Encode},
    query_type::FixedSize,
    taskprov::{
        DpConfig, DpMechanism, Query as TaskprovQuery, QueryConfig, TaskConfig, VdafConfig,
        VdafType,
    },
    AggregateShare as AggregateShareMessage, AggregateShareAad, AggregateShareReq,
    AggregationJobContinueReq, AggregationJobId, AggregationJobInitializeReq, AggregationJobResp,
    AggregationJobStep, BatchSelector, Duration, Interval, PartialBatchSelector, PrepareContinue,
    PrepareInit, PrepareResp, PrepareStepResult, ReportIdChecksum, ReportShare, Role, TaskId, Time,
};
use prio::{
    idpf::IdpfInput,
    vdaf::{
        poplar1::{Poplar1, Poplar1AggregationParam},
        xof::XofShake128,
    },
};
use rand::random;
use ring::digest::{digest, SHA256};
use serde_json::json;
use std::sync::Arc;
use trillium::{Handler, KnownHeaderName, Status};
use trillium_testing::{
    assert_headers,
    prelude::{post, put},
};
use url::Url;

type TestVdaf = Poplar1<XofShake128, 16>;

pub struct TaskprovTestCase {
    _ephemeral_datastore: EphemeralDatastore,
    clock: MockClock,
    collector_hpke_keypair: HpkeKeypair,
    datastore: Arc<Datastore<MockClock>>,
    handler: Box<dyn Handler>,
    peer_aggregator: PeerAggregator,
    task: Task,
    task_config: TaskConfig,
    task_id: TaskId,
    vdaf: TestVdaf,
    global_hpke_key: HpkeKeypair,
}

impl TaskprovTestCase {
    async fn new() -> Self {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let global_hpke_key = generate_test_hpke_config_and_private_key();
        let collector_hpke_keypair = generate_test_hpke_config_and_private_key();
        let peer_aggregator = PeerAggregatorBuilder::new()
            .with_endpoint(url::Url::parse("https://leader.example.com/").unwrap())
            .with_role(Role::Leader)
            .with_collector_hpke_config(collector_hpke_keypair.config().clone())
            .build();

        datastore
            .run_tx(|tx| {
                let global_hpke_key = global_hpke_key.clone();
                let peer_aggregator = peer_aggregator.clone();
                Box::pin(async move {
                    tx.put_global_hpke_keypair(&global_hpke_key).await.unwrap();
                    tx.put_taskprov_peer_aggregator(&peer_aggregator)
                        .await
                        .unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        let handler = aggregator_handler(
            Arc::clone(&datastore),
            clock.clone(),
            &noop_meter(),
            Config {
                taskprov_config: TaskprovConfig { enabled: true },
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let time_precision = Duration::from_seconds(1);
        let max_batch_query_count = 1;
        let min_batch_size = 1;
        let max_batch_size = 1;
        let task_expiration = clock.now().add(&Duration::from_hours(24).unwrap()).unwrap();
        let task_config = TaskConfig::new(
            Vec::from("foobar".as_bytes()),
            Vec::from([
                "https://leader.example.com/".as_bytes().try_into().unwrap(),
                "https://helper.example.com/".as_bytes().try_into().unwrap(),
            ]),
            QueryConfig::new(
                time_precision,
                max_batch_query_count,
                min_batch_size,
                TaskprovQuery::FixedSize { max_batch_size },
            ),
            task_expiration,
            VdafConfig::new(
                DpConfig::new(DpMechanism::None),
                VdafType::Poplar1 { bits: 1 },
            )
            .unwrap(),
        )
        .unwrap();

        let task_config_encoded = task_config.get_encoded();

        // We use a real VDAF since taskprov doesn't have any allowance for a test VDAF, and we use
        // Poplar1 so that the VDAF wil take more than one step, so we can exercise aggregation
        // continuation.
        let vdaf = Poplar1::new(1);

        let task_id = TaskId::try_from(digest(&SHA256, &task_config_encoded).as_ref()).unwrap();
        let vdaf_instance = task_config.vdaf_config().vdaf_type().try_into().unwrap();
        let vdaf_verify_key = peer_aggregator.derive_vdaf_verify_key(&task_id, &vdaf_instance);

        let task = TaskBuilder::new(
            QueryType::FixedSize {
                max_batch_size: max_batch_size as u64,
                batch_time_window_size: None,
            },
            vdaf_instance,
        )
        .with_id(task_id)
        .with_leader_aggregator_endpoint(Url::parse("https://leader.example.com/").unwrap())
        .with_helper_aggregator_endpoint(Url::parse("https://helper.example.com/").unwrap())
        .with_vdaf_verify_key(vdaf_verify_key)
        .with_max_batch_query_count(max_batch_query_count as u64)
        .with_task_expiration(Some(task_expiration))
        .with_report_expiry_age(peer_aggregator.report_expiry_age().copied())
        .with_min_batch_size(min_batch_size as u64)
        .with_time_precision(Duration::from_seconds(1))
        .with_tolerable_clock_skew(Duration::from_seconds(1))
        .build();

        Self {
            _ephemeral_datastore: ephemeral_datastore,
            clock,
            collector_hpke_keypair,
            datastore,
            handler: Box::new(handler),
            peer_aggregator,
            task,
            task_config,
            task_id,
            vdaf,
            global_hpke_key,
        }
    }

    fn next_report_share(
        &self,
    ) -> (
        VdafTranscript<16, TestVdaf>,
        ReportShare,
        Poplar1AggregationParam,
    ) {
        let aggregation_param =
            Poplar1AggregationParam::try_from_prefixes(Vec::from([IdpfInput::from_bools(&[true])]))
                .unwrap();
        let measurement = IdpfInput::from_bools(&[true]);
        let (report_share, transcript) = PrepareInitGenerator::new(
            self.clock.clone(),
            self.task.helper_view().unwrap(),
            self.vdaf.clone(),
            aggregation_param.clone(),
        )
        .with_hpke_config(self.global_hpke_key.config().clone())
        .next_report_share(&measurement);
        (transcript, report_share, aggregation_param)
    }
}

#[tokio::test]
async fn taskprov_aggregate_init() {
    let test = TaskprovTestCase::new().await;

    // Use two requests with the same task config. The second request will ensure that a previously
    // provisioned task is usable.
    let (transcript_1, report_share_1, aggregation_param_1) = test.next_report_share();
    let batch_id_1 = random();
    let request_1 = AggregationJobInitializeReq::new(
        aggregation_param_1.get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id_1),
        Vec::from([PrepareInit::new(
            report_share_1.clone(),
            transcript_1.leader_prepare_transitions[0].message.clone(),
        )]),
    );
    let aggregation_job_id_1: AggregationJobId = random();

    let (transcript_2, report_share_2, aggregation_param_2) = test.next_report_share();
    let batch_id_2 = random();
    let request_2 = AggregationJobInitializeReq::new(
        aggregation_param_2.get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id_2),
        Vec::from([PrepareInit::new(
            report_share_2.clone(),
            transcript_2.leader_prepare_transitions[0].message.clone(),
        )]),
    );
    let aggregation_job_id_2: AggregationJobId = random();

    for (name, request, aggregation_job_id, report_share) in [
        ("request_1", request_1, aggregation_job_id_1, report_share_1),
        ("request_2", request_2, aggregation_job_id_2, report_share_2),
    ] {
        let auth = test
            .peer_aggregator
            .primary_aggregator_auth_token()
            .request_authentication();

        let mut test_conn = put(test
            .task
            .aggregation_job_uri(&aggregation_job_id)
            .unwrap()
            .path())
        .with_request_header(auth.0, "Bearer invalid_token")
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
        )
        .with_request_body(request.get_encoded())
        .run_async(&test.handler)
        .await;
        assert_eq!(test_conn.status(), Some(Status::BadRequest), "{}", name);
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": Status::BadRequest as u16,
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test.task_id),
            }),
            "{name}",
        );

        let mut test_conn = put(test
            .task
            .aggregation_job_uri(&aggregation_job_id)
            .unwrap()
            .path())
        .with_request_header(auth.0, auth.1)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
        )
        .with_request_body(request.get_encoded())
        .run_async(&test.handler)
        .await;

        assert_eq!(test_conn.status(), Some(Status::Ok), "{name}");
        assert_headers!(
            &test_conn,
            "content-type" => (AggregationJobResp::MEDIA_TYPE)
        );
        let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;

        assert_eq!(aggregate_resp.prepare_resps().len(), 1, "{}", name);
        let prepare_step = aggregate_resp.prepare_resps().get(0).unwrap();
        assert_eq!(
            prepare_step.report_id(),
            report_share.metadata().id(),
            "{name}",
        );
        assert_matches!(
            prepare_step.result(),
            &PrepareStepResult::Continue { .. },
            "{name}",
        );
    }

    let (aggregation_jobs, got_task) = test
        .datastore
        .run_tx(|tx| {
            let task_id = test.task_id;
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_jobs_for_task::<16, FixedSize, TestVdaf>(&task_id)
                        .await
                        .unwrap(),
                    tx.get_aggregator_task(&task_id).await.unwrap(),
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(aggregation_jobs.len(), 2);
    assert!(
        aggregation_jobs[0].task_id().eq(&test.task_id)
            && aggregation_jobs[0].id().eq(&aggregation_job_id_1)
            && aggregation_jobs[0]
                .partial_batch_identifier()
                .eq(&batch_id_1)
            && aggregation_jobs[0]
                .state()
                .eq(&AggregationJobState::InProgress)
    );
    assert!(
        aggregation_jobs[1].task_id().eq(&test.task_id)
            && aggregation_jobs[1].id().eq(&aggregation_job_id_2)
            && aggregation_jobs[1]
                .partial_batch_identifier()
                .eq(&batch_id_2)
            && aggregation_jobs[1]
                .state()
                .eq(&AggregationJobState::InProgress)
    );
    assert_eq!(test.task.taskprov_helper_view().unwrap(), got_task.unwrap());
}

#[tokio::test]
async fn taskprov_opt_out_task_expired() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();

    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    // Advance clock past task expiry.
    test.clock.advance(&Duration::from_hours(48).unwrap());

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id)
        .unwrap()
        .path())
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
    )
    .with_request_body(request.get_encoded())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:invalidTask",
            "title": "Aggregator has opted out of the indicated task.",
            "taskid": format!("{}", test.task_id),
        })
    );
}

#[tokio::test]
async fn taskprov_opt_out_mismatched_task_id() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let task_expiration = test
        .clock
        .now()
        .add(&Duration::from_hours(24).unwrap())
        .unwrap();
    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        Vec::from([
            "https://leader.example.com/".as_bytes().try_into().unwrap(),
            "https://helper.example.com/".as_bytes().try_into().unwrap(),
        ]),
        // Query configuration is different from the normal test case.
        QueryConfig::new(
            Duration::from_seconds(1),
            100,
            100,
            TaskprovQuery::FixedSize {
                max_batch_size: 100,
            },
        ),
        task_expiration,
        VdafConfig::new(
            DpConfig::new(DpMechanism::None),
            VdafType::Poplar1 { bits: 1 },
        )
        .unwrap(),
    )
    .unwrap();

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    let mut test_conn = put(test
        // Use the test case task's ID.
        .task
        .aggregation_job_uri(&aggregation_job_id)
        .unwrap()
        .path())
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        // Use a different task than the URL's.
        URL_SAFE_NO_PAD.encode(another_task_config.get_encoded()),
    )
    .with_request_body(request.get_encoded())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:invalidMessage",
            "title": "The message type for a response was incorrect or the payload was malformed.",
            "taskid": format!("{}", test.task_id),
        })
    );
}

#[tokio::test]
async fn taskprov_opt_out_missing_aggregator() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let task_expiration = test
        .clock
        .now()
        .add(&Duration::from_hours(24).unwrap())
        .unwrap();
    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        // Only one aggregator!
        Vec::from(["https://leader.example.com/".as_bytes().try_into().unwrap()]),
        QueryConfig::new(
            Duration::from_seconds(1),
            100,
            100,
            TaskprovQuery::FixedSize {
                max_batch_size: 100,
            },
        ),
        task_expiration,
        VdafConfig::new(
            DpConfig::new(DpMechanism::None),
            VdafType::Poplar1 { bits: 1 },
        )
        .unwrap(),
    )
    .unwrap();
    let another_task_config_encoded = another_task_config.get_encoded();
    let another_task_id: TaskId = digest(&SHA256, &another_task_config_encoded)
        .as_ref()
        .try_into()
        .unwrap();

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    let mut test_conn = put(format!(
        "/tasks/{another_task_id}/aggregation_jobs/{aggregation_job_id}"
    ))
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(another_task_config_encoded),
    )
    .with_request_body(request.get_encoded())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:invalidMessage",
            "title": "The message type for a response was incorrect or the payload was malformed.",
            "taskid": format!("{}", another_task_id),
        })
    );
}

#[tokio::test]
async fn taskprov_opt_out_peer_aggregator_wrong_role() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let task_expiration = test
        .clock
        .now()
        .add(&Duration::from_hours(24).unwrap())
        .unwrap();
    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        // Attempt to configure leader as a helper.
        Vec::from([
            "https://helper.example.com/".as_bytes().try_into().unwrap(),
            "https://leader.example.com/".as_bytes().try_into().unwrap(),
        ]),
        QueryConfig::new(
            Duration::from_seconds(1),
            100,
            100,
            TaskprovQuery::FixedSize {
                max_batch_size: 100,
            },
        ),
        task_expiration,
        VdafConfig::new(
            DpConfig::new(DpMechanism::None),
            VdafType::Poplar1 { bits: 1 },
        )
        .unwrap(),
    )
    .unwrap();
    let another_task_config_encoded = another_task_config.get_encoded();
    let another_task_id: TaskId = digest(&SHA256, &another_task_config_encoded)
        .as_ref()
        .try_into()
        .unwrap();

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    let mut test_conn = put(format!(
        "/tasks/{another_task_id}/aggregation_jobs/{aggregation_job_id}"
    ))
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(another_task_config_encoded),
    )
    .with_request_body(request.get_encoded())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:invalidTask",
            "title": "Aggregator has opted out of the indicated task.",
            "taskid": format!("{}", another_task_id),
        })
    );
}

#[tokio::test]
async fn taskprov_opt_out_peer_aggregator_does_not_exist() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let task_expiration = test
        .clock
        .now()
        .add(&Duration::from_hours(24).unwrap())
        .unwrap();
    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        Vec::from([
            // Some non-existent aggregator.
            "https://foobar.example.com/".as_bytes().try_into().unwrap(),
            "https://leader.example.com/".as_bytes().try_into().unwrap(),
        ]),
        QueryConfig::new(
            Duration::from_seconds(1),
            100,
            100,
            TaskprovQuery::FixedSize {
                max_batch_size: 100,
            },
        ),
        task_expiration,
        VdafConfig::new(
            DpConfig::new(DpMechanism::None),
            VdafType::Poplar1 { bits: 1 },
        )
        .unwrap(),
    )
    .unwrap();
    let another_task_config_encoded = another_task_config.get_encoded();
    let another_task_id: TaskId = digest(&SHA256, &another_task_config_encoded)
        .as_ref()
        .try_into()
        .unwrap();

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    let mut test_conn = put(format!(
        "/tasks/{another_task_id}/aggregation_jobs/{aggregation_job_id}"
    ))
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(another_task_config_encoded),
    )
    .with_request_body(request.get_encoded())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:invalidTask",
            "title": "Aggregator has opted out of the indicated task.",
            "taskid": format!("{}", another_task_id),
        })
    );
}

#[tokio::test]
async fn taskprov_aggregate_continue() {
    let test = TaskprovTestCase::new().await;

    let aggregation_job_id = random();
    let batch_id = random();

    let (transcript, report_share, aggregation_param) = test.next_report_share();
    test.datastore
        .run_tx(|tx| {
            let task = test.task.clone();
            let report_share = report_share.clone();
            let transcript = transcript.clone();
            let aggregation_param = aggregation_param.clone();

            Box::pin(async move {
                // Aggregate continue is only possible if the task has already been inserted.
                tx.put_aggregator_task(&task.taskprov_helper_view().unwrap())
                    .await?;

                tx.put_report_share(task.id(), &report_share).await?;

                tx.put_aggregation_job(
                    &AggregationJob::<VERIFY_KEY_LENGTH, FixedSize, TestVdaf>::new(
                        *task.id(),
                        aggregation_job_id,
                        aggregation_param.clone(),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobStep::from(0),
                    ),
                )
                .await?;

                tx.put_report_aggregation::<VERIFY_KEY_LENGTH, TestVdaf>(&ReportAggregation::new(
                    *task.id(),
                    aggregation_job_id,
                    *report_share.metadata().id(),
                    *report_share.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::WaitingHelper(
                        transcript.helper_prepare_transitions[0]
                            .prepare_state()
                            .clone(),
                    ),
                ))
                .await?;

                tx.put_aggregate_share_job::<VERIFY_KEY_LENGTH, FixedSize, TestVdaf>(
                    &AggregateShareJob::new(
                        *task.id(),
                        batch_id,
                        aggregation_param,
                        transcript.helper_aggregate_share,
                        0,
                        ReportIdChecksum::default(),
                    ),
                )
                .await
            })
        })
        .await
        .unwrap();

    let request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            *report_share.metadata().id(),
            transcript.leader_prepare_transitions[1].message.clone(),
        )]),
    );

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    // Attempt using the wrong credentials, should reject.
    let mut test_conn = post(
        test.task
            .aggregation_job_uri(&aggregation_job_id)
            .unwrap()
            .path(),
    )
    .with_request_header(auth.0, "Bearer invalid_token")
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobContinueReq::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
    )
    .with_request_body(request.get_encoded())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test.task_id),
        })
    );

    let mut test_conn = post(
        test.task
            .aggregation_job_uri(&aggregation_job_id)
            .unwrap()
            .path(),
    )
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobContinueReq::MEDIA_TYPE,
    )
    .with_request_body(request.get_encoded())
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
    )
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;

    // We'll only validate the response. Taskprov doesn't touch functionality beyond the
    // authorization of the request.
    assert_eq!(
        aggregate_resp,
        AggregationJobResp::new(Vec::from([PrepareResp::new(
            *report_share.metadata().id(),
            PrepareStepResult::Finished
        )]))
    );
}

#[tokio::test]
async fn taskprov_aggregate_share() {
    let test = TaskprovTestCase::new().await;

    let (transcript, _, aggregation_param) = test.next_report_share();
    let batch_id = random();
    test.datastore
        .run_tx(|tx| {
            let task = test.task.clone();
            let interval =
                Interval::new(Time::from_seconds_since_epoch(6000), *task.time_precision())
                    .unwrap();
            let aggregation_param = aggregation_param.clone();
            let transcript = transcript.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task.taskprov_helper_view().unwrap())
                    .await?;

                tx.put_batch(&Batch::<16, FixedSize, TestVdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param.clone(),
                    BatchState::Closed,
                    0,
                    interval,
                ))
                .await
                .unwrap();

                tx.put_batch_aggregation(&BatchAggregation::<16, FixedSize, TestVdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    0,
                    BatchAggregationState::Aggregating,
                    Some(transcript.helper_aggregate_share),
                    1,
                    interval,
                    ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                ))
                .await
                .unwrap();
                Ok(())
            })
        })
        .await
        .unwrap();

    let request = AggregateShareReq::new(
        BatchSelector::new_fixed_size(batch_id),
        aggregation_param.get_encoded(),
        1,
        ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
    );

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    // Attempt using the wrong credentials, should reject.
    let mut test_conn = post(test.task.aggregate_shares_uri().unwrap().path())
        .with_request_header(auth.0, "Bearer invalid_token")
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregateShareReq::<FixedSize>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
        )
        .with_request_body(request.get_encoded())
        .run_async(&test.handler)
        .await;
    assert_eq!(test_conn.status(), Some(Status::BadRequest));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": Status::BadRequest as u16,
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test.task_id),
        })
    );

    let mut test_conn = post(test.task.aggregate_shares_uri().unwrap().path())
        .with_request_header(auth.0, auth.1)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregateShareReq::<FixedSize>::MEDIA_TYPE,
        )
        .with_request_body(request.get_encoded())
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
        )
        .run_async(&test.handler)
        .await;

    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(
        &test_conn,
        "content-type" => (AggregateShareMessage::MEDIA_TYPE)
    );
    let aggregate_share_resp: AggregateShareMessage = decode_response_body(&mut test_conn).await;

    hpke::open(
        &test.collector_hpke_keypair,
        &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
        aggregate_share_resp.encrypted_aggregate_share(),
        &AggregateShareAad::new(
            test.task_id,
            aggregation_param.get_encoded(),
            request.batch_selector().clone(),
        )
        .get_encoded(),
    )
    .unwrap();
}

/// This runs aggregation job init, aggregation job continue, and aggregate share requests against a
/// taskprov-enabled helper, and confirms that correct results are returned.
#[tokio::test]
async fn end_to_end() {
    let test = TaskprovTestCase::new().await;
    let (auth_header_name, auth_header_value) = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    let batch_id = random();
    let aggregation_job_id = random();

    let (transcript, report_share, aggregation_param) = test.next_report_share();
    let aggregation_job_init_request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded(),
        PartialBatchSelector::new_fixed_size(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id)
        .unwrap()
        .path())
    .with_request_header(auth_header_name, auth_header_value.clone())
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<FixedSize>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
    )
    .with_request_body(aggregation_job_init_request.get_encoded())
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;

    assert_eq!(aggregation_job_resp.prepare_resps().len(), 1);
    let prepare_resp = &aggregation_job_resp.prepare_resps()[0];
    assert_eq!(prepare_resp.report_id(), report_share.metadata().id());
    let message = assert_matches!(
        prepare_resp.result(),
        PrepareStepResult::Continue { message } => message.clone()
    );
    assert_eq!(message, transcript.helper_prepare_transitions[0].message,);

    let aggregation_job_continue_request = AggregationJobContinueReq::new(
        AggregationJobStep::from(1),
        Vec::from([PrepareContinue::new(
            *report_share.metadata().id(),
            transcript.leader_prepare_transitions[1].message.clone(),
        )]),
    );

    let mut test_conn = post(
        test.task
            .aggregation_job_uri(&aggregation_job_id)
            .unwrap()
            .path(),
    )
    .with_request_header(auth_header_name, auth_header_value.clone())
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobContinueReq::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
    )
    .with_request_body(aggregation_job_continue_request.get_encoded())
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;

    assert_eq!(aggregation_job_resp.prepare_resps().len(), 1);
    let prepare_resp = &aggregation_job_resp.prepare_resps()[0];
    assert_eq!(prepare_resp.report_id(), report_share.metadata().id());
    assert_matches!(prepare_resp.result(), PrepareStepResult::Finished);

    let checksum = ReportIdChecksum::for_report_id(report_share.metadata().id());
    let aggregate_share_request = AggregateShareReq::new(
        BatchSelector::new_fixed_size(batch_id),
        aggregation_param.get_encoded(),
        1,
        checksum,
    );

    let mut test_conn = post(test.task.aggregate_shares_uri().unwrap().path())
        .with_request_header(auth_header_name, auth_header_value.clone())
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregateShareReq::<FixedSize>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded()),
        )
        .with_request_body(aggregate_share_request.get_encoded())
        .run_async(&test.handler)
        .await;

    assert_eq!(test_conn.status(), Some(Status::Ok));
    assert_headers!(&test_conn, "content-type" => (AggregateShareMessage::MEDIA_TYPE));
    let aggregate_share_resp: AggregateShareMessage = decode_response_body(&mut test_conn).await;

    let plaintext = hpke::open(
        &test.collector_hpke_keypair,
        &HpkeApplicationInfo::new(&Label::AggregateShare, &Role::Helper, &Role::Collector),
        aggregate_share_resp.encrypted_aggregate_share(),
        &AggregateShareAad::new(
            test.task_id,
            aggregation_param.get_encoded(),
            aggregate_share_request.batch_selector().clone(),
        )
        .get_encoded(),
    )
    .unwrap();
    assert_eq!(plaintext, transcript.helper_aggregate_share.get_encoded());
}
