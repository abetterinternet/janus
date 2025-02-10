use crate::{
    aggregator::{
        aggregation_job_init::test_util::PrepareInitGenerator,
        http_handlers::test_util::{decode_response_body, take_problem_details},
        Config,
    },
    config::TaskprovConfig,
};
use assert_matches::assert_matches;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use janus_aggregator_core::{
    datastore::{
        models::{
            AggregateShareJob, AggregationJob, AggregationJobState, BatchAggregation,
            BatchAggregationState, HpkeKeyState, ReportAggregation, ReportAggregationState,
        },
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{
        test_util::{Task, TaskBuilder},
        AggregationMode, BatchMode,
    },
    taskprov::{taskprov_task_id, test_util::PeerAggregatorBuilder, PeerAggregator},
    test_util::noop_meter,
};
use janus_core::{
    hpke::{self, HpkeApplicationInfo, HpkeKeypair, Label},
    report_id::ReportIdChecksumExt,
    taskprov::TASKPROV_HEADER,
    test_util::{install_test_trace_subscriber, runtime::TestRuntime, VdafTranscript},
    time::{Clock, DurationExt, MockClock, TimeExt},
    vdaf::new_prio3_sum_vec_field64_multiproof_hmacsha256_aes128,
};
use janus_messages::{
    batch_mode::{self, LeaderSelected},
    codec::{Decode, Encode},
    taskprov::{TaskConfig, VdafConfig},
    AggregateShare as AggregateShareMessage, AggregateShareAad, AggregateShareReq,
    AggregationJobContinueReq, AggregationJobId, AggregationJobInitializeReq, AggregationJobResp,
    AggregationJobStep, BatchSelector, Duration, Extension, ExtensionType, Interval,
    PartialBatchSelector, PrepareContinue, PrepareInit, PrepareResp, PrepareStepResult,
    ReportError, ReportIdChecksum, ReportShare, Role, TaskId, Time,
};
use prio::{
    flp::gadgets::ParallelSumMultithreaded,
    vdaf::{dummy, Aggregator, Client, Vdaf},
};
use rand::random;
use serde_json::json;
use std::sync::Arc;
use trillium::{Handler, KnownHeaderName, Status};
use trillium_testing::{
    assert_headers,
    prelude::{post, put},
};
use url::Url;

use super::http_handlers::AggregatorHandlerBuilder;

pub struct TaskprovTestCase<const VERIFY_KEY_SIZE: usize, V: Vdaf> {
    _ephemeral_datastore: EphemeralDatastore,
    clock: MockClock,
    collector_hpke_keypair: HpkeKeypair,
    datastore: Arc<Datastore<MockClock>>,
    handler: Box<dyn Handler>,
    peer_aggregator: PeerAggregator,
    task: Task,
    task_config: TaskConfig,
    task_id: TaskId,
    vdaf: V,
    measurement: V::Measurement,
    aggregation_param: V::AggregationParam,
    hpke_key: HpkeKeypair,
}

impl TaskprovTestCase<0, dummy::Vdaf> {
    async fn new() -> Self {
        let vdaf = dummy::Vdaf::new(2);
        let vdaf_config = VdafConfig::Fake { rounds: 2 };
        let measurement = 13;
        let aggregation_param = dummy::AggregationParam(7);
        Self::with_vdaf(vdaf_config, vdaf, measurement, aggregation_param).await
    }
}

impl<const VERIFY_KEY_SIZE: usize, V> TaskprovTestCase<VERIFY_KEY_SIZE, V>
where
    V: Vdaf + Client<16> + Aggregator<VERIFY_KEY_SIZE, 16> + Clone,
{
    async fn with_vdaf(
        vdaf_config: VdafConfig,
        vdaf: V,
        measurement: V::Measurement,
        aggregation_param: V::AggregationParam,
    ) -> Self {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let hpke_key = HpkeKeypair::test();
        let collector_hpke_keypair = HpkeKeypair::test();
        let peer_aggregator = PeerAggregatorBuilder::new()
            .with_endpoint(url::Url::parse("https://leader.example.com/").unwrap())
            .with_peer_role(Role::Leader)
            .with_collector_hpke_config(collector_hpke_keypair.config().clone())
            .build();

        datastore
            .run_unnamed_tx(|tx| {
                let hpke_key = hpke_key.clone();
                let peer_aggregator = peer_aggregator.clone();
                Box::pin(async move {
                    tx.put_hpke_keypair(&hpke_key).await.unwrap();
                    tx.set_hpke_keypair_state(hpke_key.config().id(), &HpkeKeyState::Active)
                        .await
                        .unwrap();
                    tx.put_taskprov_peer_aggregator(&peer_aggregator)
                        .await
                        .unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        let handler = AggregatorHandlerBuilder::new(
            Arc::clone(&datastore),
            clock.clone(),
            TestRuntime::default(),
            &noop_meter(),
            Config {
                taskprov_config: TaskprovConfig { enabled: true },
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .build()
        .unwrap();

        let time_precision = Duration::from_seconds(1);
        let min_batch_size = 1;
        let task_start = clock.now();
        let task_duration = Duration::from_hours(24).unwrap();
        let task_config = TaskConfig::new(
            Vec::from("foobar".as_bytes()),
            "https://leader.example.com/".as_bytes().try_into().unwrap(),
            "https://helper.example.com/".as_bytes().try_into().unwrap(),
            time_precision,
            min_batch_size,
            batch_mode::Code::LeaderSelected,
            task_start,
            task_duration,
            vdaf_config,
            Vec::new(),
        )
        .unwrap();

        let task_config_encoded = task_config.get_encoded().unwrap();

        let task_id = taskprov_task_id(&task_config_encoded);
        let vdaf_instance = task_config.vdaf_config().try_into().unwrap();
        let vdaf_verify_key = peer_aggregator.derive_vdaf_verify_key(&task_id, &vdaf_instance);

        let task = TaskBuilder::new(
            BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            AggregationMode::Synchronous,
            vdaf_instance,
        )
        .with_id(task_id)
        .with_leader_aggregator_endpoint(Url::parse("https://leader.example.com/").unwrap())
        .with_helper_aggregator_endpoint(Url::parse("https://helper.example.com/").unwrap())
        .with_vdaf_verify_key(vdaf_verify_key)
        .with_task_start(Some(task_start))
        .with_task_end(Some(task_start.add(&task_duration).unwrap()))
        .with_report_expiry_age(peer_aggregator.report_expiry_age().copied())
        .with_min_batch_size(min_batch_size as u64)
        .with_time_precision(Duration::from_seconds(1))
        .with_tolerable_clock_skew(Duration::from_seconds(1))
        .with_taskprov_task_info(task_config.task_info().to_vec())
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
            measurement,
            aggregation_param,
            hpke_key,
        }
    }

    fn next_report_share(
        &self,
    ) -> (
        VdafTranscript<VERIFY_KEY_SIZE, V>,
        ReportShare,
        V::AggregationParam,
    ) {
        self.next_report_share_with_private_extensions(Vec::from([Extension::new(
            ExtensionType::Taskbind,
            Vec::new(),
        )]))
    }

    fn next_report_share_with_private_extensions(
        &self,
        private_extensions: Vec<Extension>,
    ) -> (
        VdafTranscript<VERIFY_KEY_SIZE, V>,
        ReportShare,
        V::AggregationParam,
    ) {
        let (report_share, transcript) = PrepareInitGenerator::new(
            self.clock.clone(),
            self.task.helper_view().unwrap(),
            self.hpke_key.config().clone(),
            self.vdaf.clone(),
            self.aggregation_param.clone(),
        )
        .with_private_extensions(private_extensions)
        .next_report_share(&self.measurement);
        (transcript, report_share, self.aggregation_param.clone())
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
        aggregation_param_1.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id_1),
        Vec::from([PrepareInit::new(
            report_share_1.clone(),
            transcript_1.leader_prepare_transitions[0].message.clone(),
        )]),
    );
    let aggregation_job_id_1: AggregationJobId = random();

    let (transcript_2, report_share_2, aggregation_param_2) = test.next_report_share();
    let batch_id_2 = random();
    let request_2 = AggregationJobInitializeReq::new(
        aggregation_param_2.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id_2),
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
            .aggregation_job_uri(&aggregation_job_id, None)
            .unwrap()
            .path())
        .with_request_header(auth.0, "Bearer invalid_token")
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
        )
        .with_request_body(request.get_encoded().unwrap())
        .run_async(&test.handler)
        .await;
        assert_eq!(test_conn.status(), Some(Status::Forbidden), "{}", name);
        assert_eq!(
            take_problem_details(&mut test_conn).await,
            json!({
                "status": u16::from(Status::Forbidden),
                "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
                "title": "The request's authorization is not valid.",
                "taskid": format!("{}", test.task_id),
            }),
            "{name}",
        );

        let mut test_conn = put(test
            .task
            .aggregation_job_uri(&aggregation_job_id, None)
            .unwrap()
            .path())
        .with_request_header(auth.0, auth.1)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
        )
        .with_request_body(request.get_encoded().unwrap())
        .run_async(&test.handler)
        .await;

        assert_eq!(test_conn.status(), Some(Status::Created), "{name}");
        assert_headers!(
            &test_conn,
            "content-type" => (AggregationJobResp::MEDIA_TYPE)
        );
        let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
        let prepare_resps = assert_matches!(
            aggregate_resp,
            AggregationJobResp::Finished { prepare_resps } => prepare_resps
        );

        assert_eq!(prepare_resps.len(), 1, "{}", name);
        let prepare_step = prepare_resps.first().unwrap();
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
        .run_unnamed_tx(|tx| {
            let task_id = test.task_id;
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(&task_id)
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
                .eq(&AggregationJobState::AwaitingRequest)
    );
    assert!(
        aggregation_jobs[1].task_id().eq(&test.task_id)
            && aggregation_jobs[1].id().eq(&aggregation_job_id_2)
            && aggregation_jobs[1]
                .partial_batch_identifier()
                .eq(&batch_id_2)
            && aggregation_jobs[1]
                .state()
                .eq(&AggregationJobState::AwaitingRequest)
    );
    let got_task = got_task.unwrap();
    assert_eq!(test.task.taskprov_helper_view().unwrap(), got_task);
    assert_eq!(got_task.taskprov_task_info(), Some(b"foobar".as_slice()));
}

#[tokio::test]
async fn taskprov_aggregate_init_missing_extension() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, aggregation_param) =
        test.next_report_share_with_private_extensions(Vec::new());
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
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

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(request.get_encoded().unwrap())
    .run_async(&test.handler)
    .await;

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

    assert_eq!(prepare_resps.len(), 1);
    let prepare_step = prepare_resps.first().unwrap();
    assert_eq!(prepare_step.report_id(), report_share.metadata().id(),);
    assert_eq!(
        prepare_step.result(),
        &PrepareStepResult::Reject(ReportError::InvalidMessage),
    );

    let (aggregation_jobs, got_task) = test
        .datastore
        .run_unnamed_tx(|tx| {
            let task_id = test.task_id;
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(&task_id)
                        .await
                        .unwrap(),
                    tx.get_aggregator_task(&task_id).await.unwrap(),
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(aggregation_jobs.len(), 1);
    assert!(
        aggregation_jobs[0].task_id().eq(&test.task_id)
            && aggregation_jobs[0].id().eq(&aggregation_job_id)
            && aggregation_jobs[0].partial_batch_identifier().eq(&batch_id)
            && aggregation_jobs[0]
                .state()
                .eq(&AggregationJobState::Finished)
    );
    assert_eq!(test.task.taskprov_helper_view().unwrap(), got_task.unwrap());
}

#[tokio::test]
async fn taskprov_aggregate_init_malformed_extension() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, aggregation_param) =
        test.next_report_share_with_private_extensions(Vec::new());
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
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

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(request.get_encoded().unwrap())
    .run_async(&test.handler)
    .await;

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

    assert_eq!(prepare_resps.len(), 1);
    let prepare_step = prepare_resps.first().unwrap();
    assert_eq!(prepare_step.report_id(), report_share.metadata().id(),);
    assert_eq!(
        prepare_step.result(),
        &PrepareStepResult::Reject(ReportError::InvalidMessage),
    );

    let (aggregation_jobs, got_task) = test
        .datastore
        .run_unnamed_tx(|tx| {
            let task_id = test.task_id;
            Box::pin(async move {
                Ok((
                    tx.get_aggregation_jobs_for_task::<0, LeaderSelected, dummy::Vdaf>(&task_id)
                        .await
                        .unwrap(),
                    tx.get_aggregator_task(&task_id).await.unwrap(),
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(aggregation_jobs.len(), 1);
    assert!(
        aggregation_jobs[0].task_id().eq(&test.task_id)
            && aggregation_jobs[0].id().eq(&aggregation_job_id)
            && aggregation_jobs[0].partial_batch_identifier().eq(&batch_id)
            && aggregation_jobs[0]
                .state()
                .eq(&AggregationJobState::Finished)
    );
    assert_eq!(test.task.taskprov_helper_view().unwrap(), got_task.unwrap());
}

#[tokio::test]
async fn taskprov_opt_out_task_ended() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();

    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
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

    // Advance clock past task end time.
    test.clock.advance(&Duration::from_hours(48).unwrap());

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(request.get_encoded().unwrap())
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
        ().get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        "https://leader.example.com/".as_bytes().try_into().unwrap(),
        "https://helper.example.com/".as_bytes().try_into().unwrap(),
        Duration::from_seconds(1),
        100,
        batch_mode::Code::LeaderSelected,
        test.clock.now(),
        Duration::from_hours(24).unwrap(),
        VdafConfig::Fake { rounds: 2 },
        Vec::new(),
    )
    .unwrap();

    let auth = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();

    let mut test_conn = put(test
        // Use the test case task's ID.
        .task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        // Use a different task than the URL's.
        URL_SAFE_NO_PAD.encode(another_task_config.get_encoded().unwrap()),
    )
    .with_request_body(request.get_encoded().unwrap())
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
async fn taskprov_opt_out_peer_aggregator_wrong_role() {
    let test = TaskprovTestCase::new().await;

    let (transcript, report_share, _) = test.next_report_share();
    let batch_id = random();
    let request = AggregationJobInitializeReq::new(
        ().get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        // Attempt to configure leader as a helper.
        "https://helper.example.com/".as_bytes().try_into().unwrap(),
        "https://leader.example.com/".as_bytes().try_into().unwrap(),
        Duration::from_seconds(1),
        100,
        batch_mode::Code::LeaderSelected,
        test.clock.now(),
        Duration::from_hours(24).unwrap(),
        VdafConfig::Fake { rounds: 2 },
        Vec::new(),
    )
    .unwrap();
    let another_task_config_encoded = another_task_config.get_encoded().unwrap();
    let another_task_id = taskprov_task_id(&another_task_config_encoded);

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
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(another_task_config_encoded),
    )
    .with_request_body(request.get_encoded().unwrap())
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
        ().get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let aggregation_job_id: AggregationJobId = random();

    let another_task_config = TaskConfig::new(
        Vec::from("foobar".as_bytes()),
        // Some non-existent aggregator.
        "https://foobar.example.com/".as_bytes().try_into().unwrap(),
        "https://leader.example.com/".as_bytes().try_into().unwrap(),
        Duration::from_seconds(1),
        100,
        batch_mode::Code::LeaderSelected,
        test.clock.now(),
        Duration::from_hours(24).unwrap(),
        VdafConfig::Fake { rounds: 2 },
        Vec::new(),
    )
    .unwrap();
    let another_task_config_encoded = another_task_config.get_encoded().unwrap();
    let another_task_id = taskprov_task_id(&another_task_config_encoded);

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
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(another_task_config_encoded),
    )
    .with_request_body(request.get_encoded().unwrap())
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
        .run_unnamed_tx(|tx| {
            let task = test.task.clone();
            let report_share = report_share.clone();
            let transcript = transcript.clone();

            Box::pin(async move {
                // Aggregate continue is only possible if the task has already been inserted.
                tx.put_aggregator_task(&task.taskprov_helper_view().unwrap())
                    .await?;

                tx.put_scrubbed_report(
                    task.id(),
                    report_share.metadata().id(),
                    report_share.metadata().time(),
                )
                .await?;

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
                .await?;

                tx.put_report_aggregation::<0, dummy::Vdaf>(&ReportAggregation::new(
                    *task.id(),
                    aggregation_job_id,
                    *report_share.metadata().id(),
                    *report_share.metadata().time(),
                    0,
                    None,
                    ReportAggregationState::HelperContinue {
                        prepare_state: *transcript.helper_prepare_transitions[0].prepare_state(),
                    },
                ))
                .await?;

                tx.put_aggregate_share_job::<0, LeaderSelected, dummy::Vdaf>(
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
            .aggregation_job_uri(&aggregation_job_id, None)
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
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(request.get_encoded().unwrap())
    .run_async(&test.handler)
    .await;
    assert_eq!(test_conn.status(), Some(Status::Forbidden));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": u16::from(Status::Forbidden),
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test.task_id),
        })
    );

    let mut test_conn = post(
        test.task
            .aggregation_job_uri(&aggregation_job_id, None)
            .unwrap()
            .path(),
    )
    .with_request_header(auth.0, auth.1)
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobContinueReq::MEDIA_TYPE,
    )
    .with_request_body(request.get_encoded().unwrap())
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Accepted));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregate_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;

    // We'll only validate the response. Taskprov doesn't touch functionality beyond the
    // authorization of the request.
    assert_eq!(
        aggregate_resp,
        AggregationJobResp::Finished {
            prepare_resps: Vec::from([PrepareResp::new(
                *report_share.metadata().id(),
                PrepareStepResult::Finished
            )])
        }
    );
}

#[tokio::test]
async fn taskprov_aggregate_share() {
    let test = TaskprovTestCase::new().await;

    let (transcript, _, aggregation_param) = test.next_report_share();
    let batch_id = random();
    test.datastore
        .run_unnamed_tx(|tx| {
            let task = test.task.clone();
            let interval =
                Interval::new(Time::from_seconds_since_epoch(6000), *task.time_precision())
                    .unwrap();
            let transcript = transcript.clone();

            Box::pin(async move {
                tx.put_aggregator_task(&task.taskprov_helper_view().unwrap())
                    .await?;

                tx.put_batch_aggregation(&BatchAggregation::<0, LeaderSelected, dummy::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    0,
                    interval,
                    BatchAggregationState::Aggregating {
                        aggregate_share: Some(transcript.helper_aggregate_share),
                        report_count: 1,
                        checksum: ReportIdChecksum::get_decoded(&[3; 32]).unwrap(),
                        aggregation_jobs_created: 1,
                        aggregation_jobs_terminated: 1,
                    },
                ))
                .await
                .unwrap();
                Ok(())
            })
        })
        .await
        .unwrap();

    let request = AggregateShareReq::new(
        BatchSelector::new_leader_selected(batch_id),
        aggregation_param.get_encoded().unwrap(),
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
            AggregateShareReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
        )
        .with_request_body(request.get_encoded().unwrap())
        .run_async(&test.handler)
        .await;
    assert_eq!(test_conn.status(), Some(Status::Forbidden));
    assert_eq!(
        take_problem_details(&mut test_conn).await,
        json!({
            "status": u16::from(Status::Forbidden),
            "type": "urn:ietf:params:ppm:dap:error:unauthorizedRequest",
            "title": "The request's authorization is not valid.",
            "taskid": format!("{}", test.task_id),
        })
    );

    let mut test_conn = post(test.task.aggregate_shares_uri().unwrap().path())
        .with_request_header(auth.0, auth.1)
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregateShareReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .with_request_body(request.get_encoded().unwrap())
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
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
            aggregation_param.get_encoded().unwrap(),
            request.batch_selector().clone(),
        )
        .get_encoded()
        .unwrap(),
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
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(auth_header_name, auth_header_value.clone())
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(aggregation_job_init_request.get_encoded().unwrap())
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Created));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
    let prepare_resps = assert_matches!(
        aggregation_job_resp,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );

    assert_eq!(prepare_resps.len(), 1);
    let prepare_resp = &prepare_resps[0];
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
            .aggregation_job_uri(&aggregation_job_id, None)
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
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(aggregation_job_continue_request.get_encoded().unwrap())
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Accepted));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
    let prepare_resps = assert_matches!(
        aggregation_job_resp,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );

    assert_eq!(prepare_resps.len(), 1);
    let prepare_resp = &prepare_resps[0];
    assert_eq!(prepare_resp.report_id(), report_share.metadata().id());
    assert_matches!(prepare_resp.result(), PrepareStepResult::Finished);

    let checksum = ReportIdChecksum::for_report_id(report_share.metadata().id());
    let aggregate_share_request = AggregateShareReq::new(
        BatchSelector::new_leader_selected(batch_id),
        aggregation_param.get_encoded().unwrap(),
        1,
        checksum,
    );

    let mut test_conn = post(test.task.aggregate_shares_uri().unwrap().path())
        .with_request_header(auth_header_name, auth_header_value.clone())
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregateShareReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
        )
        .with_request_body(aggregate_share_request.get_encoded().unwrap())
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
            aggregation_param.get_encoded().unwrap(),
            aggregate_share_request.batch_selector().clone(),
        )
        .get_encoded()
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        plaintext,
        transcript.helper_aggregate_share.get_encoded().unwrap()
    );
}

#[tokio::test]
async fn end_to_end_sumvec_hmac() {
    let vdaf = new_prio3_sum_vec_field64_multiproof_hmacsha256_aes128::<
        ParallelSumMultithreaded<_, _>,
    >(2, 8, 12, 14)
    .unwrap();
    let vdaf_config = VdafConfig::Prio3SumVecField64MultiproofHmacSha256Aes128 {
        length: 12,
        bits: 8,
        chunk_length: 14,
        proofs: 2,
    };
    let measurement = Vec::from([255, 1, 10, 20, 30, 0, 99, 100, 0, 0, 0, 0]);
    let test = TaskprovTestCase::with_vdaf(vdaf_config, vdaf, measurement, ()).await;
    let (auth_header_name, auth_header_value) = test
        .peer_aggregator
        .primary_aggregator_auth_token()
        .request_authentication();
    let batch_id = random();
    let aggregation_job_id = random();
    let (transcript, report_share, aggregation_param) = test.next_report_share();
    let aggregation_job_init_request = AggregationJobInitializeReq::new(
        aggregation_param.get_encoded().unwrap(),
        PartialBatchSelector::new_leader_selected(batch_id),
        Vec::from([PrepareInit::new(
            report_share.clone(),
            transcript.leader_prepare_transitions[0].message.clone(),
        )]),
    );

    let mut test_conn = put(test
        .task
        .aggregation_job_uri(&aggregation_job_id, None)
        .unwrap()
        .path())
    .with_request_header(auth_header_name, auth_header_value.clone())
    .with_request_header(
        KnownHeaderName::ContentType,
        AggregationJobInitializeReq::<LeaderSelected>::MEDIA_TYPE,
    )
    .with_request_header(
        TASKPROV_HEADER,
        URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
    )
    .with_request_body(aggregation_job_init_request.get_encoded().unwrap())
    .run_async(&test.handler)
    .await;

    assert_eq!(test_conn.status(), Some(Status::Created));
    assert_headers!(&test_conn, "content-type" => (AggregationJobResp::MEDIA_TYPE));
    let aggregation_job_resp: AggregationJobResp = decode_response_body(&mut test_conn).await;
    let prepare_resps = assert_matches!(
        aggregation_job_resp,
        AggregationJobResp::Finished { prepare_resps } => prepare_resps
    );

    assert_eq!(prepare_resps.len(), 1);
    let prepare_resp = &prepare_resps[0];
    assert_eq!(prepare_resp.report_id(), report_share.metadata().id());
    let message = assert_matches!(prepare_resp.result(), PrepareStepResult::Continue { message } => message.clone());
    assert_eq!(message, transcript.helper_prepare_transitions[0].message);

    let checksum = ReportIdChecksum::for_report_id(report_share.metadata().id());
    let aggregate_share_request = AggregateShareReq::new(
        BatchSelector::new_leader_selected(batch_id),
        aggregation_param.get_encoded().unwrap(),
        1,
        checksum,
    );

    let mut test_conn = post(test.task.aggregate_shares_uri().unwrap().path())
        .with_request_header(auth_header_name, auth_header_value.clone())
        .with_request_header(
            KnownHeaderName::ContentType,
            AggregateShareReq::<LeaderSelected>::MEDIA_TYPE,
        )
        .with_request_header(
            TASKPROV_HEADER,
            URL_SAFE_NO_PAD.encode(test.task_config.get_encoded().unwrap()),
        )
        .with_request_body(aggregate_share_request.get_encoded().unwrap())
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
            aggregation_param.get_encoded().unwrap(),
            aggregate_share_request.batch_selector().clone(),
        )
        .get_encoded()
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        plaintext,
        transcript.helper_aggregate_share.get_encoded().unwrap()
    );
}
