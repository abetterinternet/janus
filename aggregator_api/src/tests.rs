use crate::{
    aggregator_api_handler,
    models::{
        DeleteTaskprovPeerAggregatorReq, GetTaskAggregationMetricsResp, GetTaskIdsResp,
        GetTaskUploadMetricsResp, HpkeConfigResp, PatchHpkeConfigReq, PostTaskReq,
        PostTaskprovPeerAggregatorReq, PutHpkeConfigReq, TaskResp, TaskprovPeerAggregatorResp,
    },
    Config, CONTENT_TYPE,
};
use assert_matches::assert_matches;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use futures::future::try_join_all;
use janus_aggregator_core::{
    datastore::{
        models::{HpkeKeyState, TaskAggregationCounter, TaskUploadCounter},
        test_util::{ephemeral_datastore, EphemeralDatastore},
        Datastore,
    },
    task::{
        test_util::TaskBuilder, AggregationMode, AggregatorTask, AggregatorTaskParameters,
        BatchMode,
    },
    taskprov::test_util::PeerAggregatorBuilder,
    test_util::noop_meter,
    SecretBytes,
};
use janus_core::{
    auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
    hpke::HpkeKeypair,
    test_util::install_test_trace_subscriber,
    time::MockClock,
    vdaf::{vdaf_dp_strategies, VdafInstance, VERIFY_KEY_LENGTH_PRIO3},
};
use janus_messages::{
    Duration, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId, HpkeKemId, HpkePublicKey, Role,
    TaskId, Time,
};
use rand::{distributions::Standard, random, thread_rng, Rng};
use serde_test::{assert_ser_tokens, assert_tokens, Token};
use std::{iter, sync::Arc};
use trillium::{Handler, Status};
use trillium_testing::{
    assert_body_contains, assert_response, assert_status,
    prelude::{delete, get, patch, post, put},
    Url,
};

const AUTH_TOKEN: &str = "Y29sbGVjdG9yLWFiY2RlZjAw";

async fn setup_api_test() -> (impl Handler, EphemeralDatastore, Arc<Datastore<MockClock>>) {
    install_test_trace_subscriber();
    let ephemeral_datastore = ephemeral_datastore().await;
    let datastore = Arc::new(ephemeral_datastore.datastore(MockClock::default()).await);
    let handler = aggregator_api_handler(
        Arc::clone(&datastore),
        Config {
            auth_tokens: Vec::from([
                AuthenticationToken::new_bearer_token_from_string(AUTH_TOKEN).unwrap(),
            ]),
            public_dap_url: "https://dap.url".parse().unwrap(),
        },
        &noop_meter(),
    );

    (handler, ephemeral_datastore, datastore)
}

#[tokio::test]
async fn get_config() {
    let (handler, ..) = setup_api_test().await;
    let mut conn = get("/")
        .with_request_header("Authorization", format!("Bearer {}", AUTH_TOKEN))
        .with_request_header("Accept", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    assert_body_contains!(
        conn,
        concat!(
            r#""protocol":"DAP-09","dap_url":"https://dap.url/","role":"Either","vdafs":"#,
            r#"["Prio3Count","Prio3Sum","Prio3Histogram","Prio3SumVec"],"#,
            r#""batch_modes":["TimeInterval","LeaderSelected"],"#,
            r#""features":["TokenHash","UploadMetrics","TimeBucketedLeaderSelected","PureDpDiscreteLaplace"],"#,
            r#""software_name":"Janus","software_version":""#,
        )
    );
}

#[tokio::test]
async fn get_task_ids() {
    // Setup: write a few tasks to the datastore.
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let mut task_ids: Vec<_> = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let tasks: Vec<_> = iter::repeat_with(|| {
                    TaskBuilder::new(
                        BatchMode::TimeInterval,
                        AggregationMode::Synchronous,
                        VdafInstance::Fake { rounds: 1 },
                    )
                    .build()
                    .leader_view()
                    .unwrap()
                })
                .take(10)
                .collect();

                try_join_all(tasks.iter().map(|task| tx.put_aggregator_task(task))).await?;

                Ok(tasks.into_iter().map(|task| *task.id()).collect())
            })
        })
        .await
        .unwrap();
    task_ids.sort();

    fn response_for(task_ids: &[TaskId]) -> String {
        serde_json::to_string(&GetTaskIdsResp {
            task_ids: task_ids.to_vec(),
            pagination_token: task_ids.last().cloned(),
        })
        .unwrap()
    }

    // Verify: we can get the task IDs we wrote back from the API.
    assert_response!(
        get("/task_ids")
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Ok,
        response_for(&task_ids),
    );

    // Verify: the lower_bound is respected, if specified.
    assert_response!(
        get(format!(
            "/task_ids?pagination_token={}",
            task_ids.first().unwrap()
        ))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .run_async(&handler)
        .await,
        Status::Ok,
        response_for(&task_ids[1..]),
    );

    // Verify: if the lower bound is large enough, nothing is returned.
    // (also verifies the "last" response will not include a pagination token)
    assert_response!(
        get(format!(
            "/task_ids?pagination_token={}",
            task_ids.last().unwrap()
        ))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .run_async(&handler)
        .await,
        Status::Ok,
        response_for(&[]),
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        get("/task_ids")
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );

    // Verify: requests without the Accept header are denied.
    assert_response!(
        get("/task_ids")
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .run_async(&handler)
            .await,
        Status::NotAcceptable,
        ""
    );
}

#[tokio::test]
async fn post_task_bad_role() {
    // Setup: create a datastore & handler.
    let (handler, _ephemeral_datastore, _) = setup_api_test().await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );
    let aggregator_auth_token = AuthenticationToken::DapAuth(random());

    let req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: None,
        vdaf: VdafInstance::Prio3Count,
        role: Role::Collector,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: Some(Time::from_seconds_since_epoch(12340)),
        task_end: Some(Time::from_seconds_since_epoch(12345)),
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: Some(aggregator_auth_token),
        collector_auth_token_hash: Some(AuthenticationTokenHash::from(&random())),
    };
    assert_response!(
        post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::BadRequest
    );
}

#[tokio::test]
async fn post_task_unauthorized() {
    // Setup: create a datastore & handler.
    let (handler, _ephemeral_datastore, _) = setup_api_test().await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );
    let aggregator_auth_token = AuthenticationToken::DapAuth(random());

    let req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: Some(AggregationMode::Synchronous),
        vdaf: VdafInstance::Prio3Count,
        role: Role::Helper,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: None,
        task_end: Some(Time::from_seconds_since_epoch(12345)),
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: Some(aggregator_auth_token),
        collector_auth_token_hash: Some(AuthenticationTokenHash::from(&random())),
    };
    assert_response!(
        post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            // no Authorization header
            .run_async(&handler)
            .await,
        Status::Unauthorized
    );
}

/// Test the POST /tasks endpoint, with a helper task with no optional fields defined
#[tokio::test]
async fn post_task_helper_no_optional_fields() {
    // Setup: create a datastore & handler.
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );

    // Verify: posting a task creates a new task which matches the request.
    let req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: Some(AggregationMode::Synchronous),
        vdaf: VdafInstance::Prio3Count,
        role: Role::Helper,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: None,
        task_end: None,
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: None,
        collector_auth_token_hash: None,
    };
    let mut conn = post("/tasks")
        .with_request_body(serde_json::to_vec(&req).unwrap())
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    let mut got_task_resp: TaskResp = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();

    // Task creation response will include the aggregator auth token, but it won't be in the
    // datastore or subsequent TaskResps. The token should be a Bearer token.
    assert_matches!(
        got_task_resp.aggregator_auth_token,
        Some(AuthenticationToken::Bearer(_))
    );

    let got_task = ds
        .run_unnamed_tx(|tx| {
            let got_task_resp = got_task_resp.clone();
            Box::pin(async move { tx.get_aggregator_task(&got_task_resp.task_id).await })
        })
        .await
        .unwrap()
        .expect("task was not created");

    // Verify that the task written to the datastore matches the request...
    assert_eq!(
        &req.peer_aggregator_endpoint,
        got_task.peer_aggregator_endpoint()
    );
    assert_eq!(&req.batch_mode, got_task.batch_mode());
    assert_eq!(&req.vdaf, got_task.vdaf());
    assert_eq!(&req.role, got_task.role());
    assert_eq!(req.task_end.as_ref(), got_task.task_end());
    assert_eq!(req.min_batch_size, got_task.min_batch_size());
    assert_eq!(&req.time_precision, got_task.time_precision());
    assert!(got_task.aggregator_auth_token().is_none());
    assert!(got_task.collector_auth_token_hash().is_none());
    assert_eq!(
        &req.collector_hpke_config,
        got_task.collector_hpke_config().unwrap()
    );

    // ...and the response. Clear the aggregator auth token from got_task_resp or it won't match
    // what's in the datastore, as the helper only stores the auth token _hash_.
    got_task_resp.aggregator_auth_token = None;
    assert_eq!(got_task_resp, TaskResp::try_from(&got_task).unwrap());
}

#[tokio::test]
async fn post_task_helper_with_aggregator_auth_token() {
    // Setup: create a datastore & handler.
    let (handler, _ephemeral_datastore, _) = setup_api_test().await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );
    let aggregator_auth_token = AuthenticationToken::DapAuth(random());

    // Verify: posting a task with role = helper and an aggregator auth token fails
    let req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: Some(AggregationMode::Synchronous),
        vdaf: VdafInstance::Prio3Count,
        role: Role::Helper,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: None,
        task_end: Some(Time::from_seconds_since_epoch(12345)),
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: Some(aggregator_auth_token),
        collector_auth_token_hash: None,
    };
    assert_response!(
        post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::BadRequest
    );
}

#[tokio::test]
async fn post_task_idempotence() {
    // Setup: create a datastore & handler.
    let (handler, ephemeral_datastore, _) = setup_api_test().await;
    let ds = ephemeral_datastore.datastore(MockClock::default()).await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );
    let aggregator_auth_token = AuthenticationToken::DapAuth(random());

    // Verify: posting a task creates a new task which matches the request.
    let mut req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: Some(AggregationMode::Synchronous),
        vdaf: VdafInstance::Prio3Count,
        role: Role::Leader,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: Some(Time::from_seconds_since_epoch(12340)),
        task_end: Some(Time::from_seconds_since_epoch(12345)),
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: Some(aggregator_auth_token.clone()),
        collector_auth_token_hash: Some(AuthenticationTokenHash::from(&random())),
    };

    let post_task = || async {
        let mut conn = post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await;
        assert_status!(conn, Status::Ok);
        serde_json::from_slice::<TaskResp>(
            &conn
                .take_response_body()
                .unwrap()
                .into_bytes()
                .await
                .unwrap(),
        )
        .unwrap()
    };

    let first_task_resp = post_task().await;
    let second_task_resp = post_task().await;

    assert_eq!(first_task_resp.task_id, second_task_resp.task_id);
    assert_eq!(
        first_task_resp.vdaf_verify_key,
        second_task_resp.vdaf_verify_key
    );

    let got_tasks = ds
        .run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_tasks().await }))
        .await
        .unwrap();

    assert!(got_tasks.len() == 1);
    assert_eq!(got_tasks[0].id(), &first_task_resp.task_id);

    // Mutate the PostTaskReq and re-send it.
    req.min_batch_size = 332;
    let conn = post("/tasks")
        .with_request_body(serde_json::to_vec(&req).unwrap())
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Conflict);
}

/// Test the POST /tasks endpoint, with a leader task with all of the optional fields provided.
#[tokio::test]
async fn post_task_leader_all_optional_fields() {
    // Setup: create a datastore & handler.
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );
    let aggregator_auth_token = AuthenticationToken::DapAuth(random());
    let collector_auth_token_hash = AuthenticationTokenHash::from(&random());
    // Verify: posting a task creates a new task which matches the request.
    let req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: Some(AggregationMode::Synchronous),
        vdaf: VdafInstance::Prio3Count,
        role: Role::Leader,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: None,
        task_end: Some(Time::from_seconds_since_epoch(12345)),
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: Some(aggregator_auth_token.clone()),
        collector_auth_token_hash: Some(collector_auth_token_hash.clone()),
    };
    let mut conn = post("/tasks")
        .with_request_body(serde_json::to_vec(&req).unwrap())
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    let got_task_resp: TaskResp = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();

    let got_task = ds
        .run_unnamed_tx(|tx| {
            let got_task_resp = got_task_resp.clone();
            Box::pin(async move { tx.get_aggregator_task(&got_task_resp.task_id).await })
        })
        .await
        .unwrap()
        .expect("task was not created");

    // Verify that the task written to the datastore matches the request...
    assert_eq!(
        &req.peer_aggregator_endpoint,
        got_task.peer_aggregator_endpoint()
    );
    assert_eq!(&req.batch_mode, got_task.batch_mode());
    assert_eq!(&req.vdaf, got_task.vdaf());
    assert_eq!(&req.role, got_task.role());
    assert_eq!(&vdaf_verify_key, got_task.opaque_vdaf_verify_key());
    assert_eq!(req.task_end.as_ref(), got_task.task_end());
    assert_eq!(req.min_batch_size, got_task.min_batch_size());
    assert_eq!(&req.time_precision, got_task.time_precision());
    assert_eq!(
        &req.collector_hpke_config,
        got_task.collector_hpke_config().unwrap()
    );
    assert_eq!(
        aggregator_auth_token.as_ref(),
        got_task.aggregator_auth_token().unwrap().as_ref()
    );
    assert_eq!(
        got_task.collector_auth_token_hash().unwrap(),
        &collector_auth_token_hash
    );

    // ...and the response.
    assert_eq!(got_task_resp, TaskResp::try_from(&got_task).unwrap());
}

/// Test the POST /tasks endpoint, with a leader task with all of the optional fields provided.
#[tokio::test]
async fn post_task_leader_no_aggregator_auth_token() {
    // Setup: create a datastore & handler.
    let (handler, _ephemeral_datastore, _) = setup_api_test().await;

    let vdaf_verify_key = SecretBytes::new(
        thread_rng()
            .sample_iter(Standard)
            .take(VERIFY_KEY_LENGTH_PRIO3)
            .collect(),
    );

    // Verify: posting a task with role = Leader and no aggregator auth token fails
    let req = PostTaskReq {
        peer_aggregator_endpoint: "http://aggregator.endpoint".try_into().unwrap(),
        batch_mode: BatchMode::TimeInterval,
        aggregation_mode: Some(AggregationMode::Synchronous),
        vdaf: VdafInstance::Prio3Count,
        role: Role::Leader,
        vdaf_verify_key: URL_SAFE_NO_PAD.encode(&vdaf_verify_key),
        task_start: Some(Time::from_seconds_since_epoch(12340)),
        task_end: Some(Time::from_seconds_since_epoch(12345)),
        min_batch_size: 223,
        time_precision: Duration::from_seconds(62),
        collector_hpke_config: HpkeKeypair::test().config().clone(),
        aggregator_auth_token: None,
        collector_auth_token_hash: Some(AuthenticationTokenHash::from(&random())),
    };

    assert_response!(
        post("/tasks")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::BadRequest
    );
}

#[rstest::rstest]
#[case::leader(Role::Leader)]
#[case::helper(Role::Helper)]
#[tokio::test]
async fn get_task(#[case] role: Role) {
    // Setup: write a task to the datastore.
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .build()
    .view_for_role(role)
    .unwrap();

    ds.put_aggregator_task(&task).await.unwrap();

    // Verify: getting the task returns the expected result.
    let want_task_resp = TaskResp::try_from(&task).unwrap();
    let mut conn = get(format!("/tasks/{}", task.id()))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    let got_task_resp = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(want_task_resp, got_task_resp);

    // Verify: getting a nonexistent task returns NotFound.
    assert_response!(
        get(format!("/tasks/{}", random::<TaskId>()))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NotFound,
        "",
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        get(format!("/tasks/{}", task.id()))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );
}

#[tokio::test]
async fn delete_task() {
    // Setup: write a task to the datastore.
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let task_id = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap();

                tx.put_aggregator_task(&task).await?;

                Ok(*task.id())
            })
        })
        .await
        .unwrap();

    // Verify: deleting a task succeeds (and actually deletes the task).
    assert_response!(
        delete(format!("/tasks/{}", &task_id))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NoContent,
        "",
    );

    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            assert_eq!(tx.get_aggregator_task(&task_id).await.unwrap(), None);
            Ok(())
        })
    })
    .await
    .unwrap();

    // Verify: deleting a task twice returns NoContent.
    assert_response!(
        delete(format!("/tasks/{}", &task_id))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NoContent,
        "",
    );

    // Verify: deleting an arbitrary nonexistent task ID returns NoContent.
    assert_response!(
        delete(format!("/tasks/{}", &random::<TaskId>()))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NoContent,
        "",
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        delete(format!("/tasks/{}", &task_id))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        ""
    );
}

#[rstest::rstest]
#[case::leader(Role::Leader)]
#[case::helper(Role::Helper)]
#[tokio::test]
async fn patch_task(#[case] role: Role) {
    // Setup: write a task to the datastore.
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let task = TaskBuilder::new(
        BatchMode::TimeInterval,
        AggregationMode::Synchronous,
        VdafInstance::Fake { rounds: 1 },
    )
    .with_task_end(Some(Time::from_seconds_since_epoch(1000)))
    .build()
    .view_for_role(role)
    .unwrap();

    ds.put_aggregator_task(&task).await.unwrap();
    let task_id = *task.id();

    // Verify: patching the task with empty body does nothing.
    let want_task_resp = TaskResp::try_from(&task).unwrap();
    let mut conn = patch(format!("/tasks/{}", task.id()))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_body("{}")
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    let got_task_resp = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(want_task_resp, got_task_resp);
    let task = ds
        .run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_task(&task_id).await }))
        .await
        .unwrap();
    assert_eq!(
        task.unwrap().task_end(),
        Some(&Time::from_seconds_since_epoch(1000))
    );

    // Verify: patching the task with a null task end time returns the expected result.
    let mut conn = patch(format!("/tasks/{}", task_id))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_body(r#"{"task_end": null}"#)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    let got_task_resp: TaskResp = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(got_task_resp.task_end, None);
    let task = ds
        .run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_task(&task_id).await }))
        .await
        .unwrap();
    assert_eq!(task.unwrap().task_end(), None);

    // Verify: patching the task with a task end time returns the expected result.
    let expected_time = Some(Time::from_seconds_since_epoch(2000));
    let mut conn = patch(format!("/tasks/{}", task_id))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_body(r#"{"task_end": 2000}"#)
        .run_async(&handler)
        .await;
    assert_status!(conn, Status::Ok);
    let got_task_resp: TaskResp = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(got_task_resp.task_end, expected_time);
    let task = ds
        .run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_task(&task_id).await }))
        .await
        .unwrap();
    assert_eq!(task.unwrap().task_end(), expected_time.as_ref());

    // Verify: patching a nonexistent task returns NotFound.
    assert_response!(
        patch(format!("/tasks/{}", random::<TaskId>()))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_body("{}")
            .run_async(&handler)
            .await,
        Status::NotFound,
        "",
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        patch(format!("/tasks/{}", task_id))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_body("{}")
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );
}

#[tokio::test]
async fn get_task_upload_metrics() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;
    let task_id = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap();
                let task_id = *task.id();
                tx.put_aggregator_task(&task).await.unwrap();

                Ok(task_id)
            })
        })
        .await
        .unwrap();

    // Verify: requesting metrics on a fresh task returns zeroes.
    assert_response!(
        get(format!("/tasks/{}/metrics/uploads", &task_id))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Ok,
        serde_json::to_string(&GetTaskUploadMetricsResp(TaskUploadCounter::default())).unwrap(),
    );

    // Verify: requesting metrics on a task returns the correct result.
    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            tx.increment_task_upload_counter(
                &task_id,
                1,
                &TaskUploadCounter::new_with_values(0, 0, 2, 4, 6, 100, 25, 22, 12),
            )
            .await
        })
    })
    .await
    .unwrap();
    assert_response!(
        get(format!("/tasks/{}/metrics/uploads", &task_id))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Ok,
        serde_json::to_string(&GetTaskUploadMetricsResp(
            TaskUploadCounter::new_with_values(0, 0, 2, 4, 6, 100, 25, 22, 12)
        ))
        .unwrap(),
    );

    // Verify: requesting metrics on a nonexistent task returns NotFound.
    assert_response!(
        get(format!("/tasks/{}/metrics/uploads", &random::<TaskId>()))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NotFound,
        "",
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        get(format!("/tasks/{}/metrics/uploads", &task_id))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );
}

#[tokio::test]
async fn get_task_aggregation_metrics() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;
    let task_id = ds
        .run_unnamed_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Fake { rounds: 1 },
                )
                .build()
                .leader_view()
                .unwrap();
                let task_id = *task.id();
                tx.put_aggregator_task(&task).await.unwrap();

                Ok(task_id)
            })
        })
        .await
        .unwrap();

    // Verify: requesting metrics on a fresh task returns zeroes.
    assert_response!(
        get(format!("/tasks/{task_id}/metrics/aggregations"))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Ok,
        serde_json::to_string(&GetTaskAggregationMetricsResp(
            TaskAggregationCounter::default()
        ))
        .unwrap(),
    );

    // Verify: requesting metrics on a task returns the correct result.
    ds.run_unnamed_tx(|tx| {
        Box::pin(async move {
            tx.increment_task_aggregation_counter(
                &task_id,
                5,
                &TaskAggregationCounter::new_with_values(15),
            )
            .await
        })
    })
    .await
    .unwrap();
    assert_response!(
        get(format!("/tasks/{task_id}/metrics/aggregations"))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Ok,
        serde_json::to_string(&GetTaskAggregationMetricsResp(
            TaskAggregationCounter::new_with_values(15)
        ))
        .unwrap(),
    );

    // Verify: requesting metrics on a nonexistent task returns NotFound.
    assert_response!(
        get(format!(
            "/tasks/{}/metrics/aggregations",
            &random::<TaskId>()
        ))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .run_async(&handler)
        .await,
        Status::NotFound,
        "",
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        get(format!("/tasks/{task_id}/metrics/aggregations"))
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );
}

#[tokio::test]
async fn get_hpke_configs() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let mut conn = get("/hpke_configs")
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_response!(conn, Status::Ok);
    let resp: Vec<HpkeConfigResp> = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(resp, vec![]);

    let keypair1_id: u8 = random();
    let keypair1 = HpkeKeypair::test_with_id(HpkeConfigId::from(keypair1_id));
    let keypair2 = HpkeKeypair::generate(
        HpkeConfigId::from(keypair1_id.wrapping_add(1)),
        HpkeKemId::P256HkdfSha256,
        HpkeKdfId::HkdfSha384,
        HpkeAeadId::Aes128Gcm,
    )
    .unwrap();
    ds.run_unnamed_tx(|tx| {
        let keypair1 = keypair1.clone();
        let keypair2 = keypair2.clone();
        Box::pin(async move {
            tx.put_hpke_keypair(&keypair1).await?;
            tx.put_hpke_keypair(&keypair2).await?;
            tx.set_hpke_keypair_state(keypair2.config().id(), &HpkeKeyState::Active)
                .await?;
            Ok(())
        })
    })
    .await
    .unwrap();

    let mut conn = get("/hpke_configs")
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_response!(conn, Status::Ok);
    let mut resp: Vec<HpkeConfigResp> = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    resp.sort_by(|a, b| a.config.id().cmp(b.config.id()));

    let mut expected = vec![
        HpkeConfigResp {
            config: keypair1.config().clone(),
            state: HpkeKeyState::Pending,
        },
        HpkeConfigResp {
            config: keypair2.config().clone(),
            state: HpkeKeyState::Active,
        },
    ];
    expected.sort_by(|a, b| a.config.id().cmp(b.config.id()));

    assert_eq!(resp, expected);

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        put("/hpke_configs")
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );
}

#[tokio::test]
async fn get_hpke_config() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    // Verify: non-existent key.
    assert_response!(
        get("/hpke_configs/123")
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NotFound
    );

    // Verify: overflow u8.
    assert_response!(
        get("/hpke_configs/1234310294")
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::BadRequest
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        put("/hpke_configs/123")
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );

    let keypair1_id: u8 = random();
    let keypair1 = HpkeKeypair::test_with_id(HpkeConfigId::from(keypair1_id));
    let keypair2 = HpkeKeypair::generate(
        HpkeConfigId::from(keypair1_id.wrapping_add(1)),
        HpkeKemId::P256HkdfSha256,
        HpkeKdfId::HkdfSha384,
        HpkeAeadId::Aes128Gcm,
    )
    .unwrap();
    ds.run_unnamed_tx(|tx| {
        let keypair1 = keypair1.clone();
        let keypair2 = keypair2.clone();
        Box::pin(async move {
            tx.put_hpke_keypair(&keypair1).await?;
            tx.put_hpke_keypair(&keypair2).await?;
            tx.set_hpke_keypair_state(keypair2.config().id(), &HpkeKeyState::Active)
                .await?;
            Ok(())
        })
    })
    .await
    .unwrap();

    for (key, state) in [
        (keypair1, HpkeKeyState::Pending),
        (keypair2, HpkeKeyState::Active),
    ] {
        let mut conn = get(format!("/hpke_configs/{}", key.config().id()))
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await;
        assert_response!(conn, Status::Ok);
        let resp: HpkeConfigResp = serde_json::from_slice(
            &conn
                .take_response_body()
                .unwrap()
                .into_bytes()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            resp,
            HpkeConfigResp {
                config: key.config().clone(),
                state,
            },
        );
    }
}

#[tokio::test]
async fn put_hpke_config() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    // No custom parameters.
    let mut key1_resp = put("/hpke_configs")
        .with_request_body("{}")
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;

    assert_response!(key1_resp, Status::Created);
    let key1: HpkeConfigResp = serde_json::from_slice(
        &key1_resp
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();

    // Choose some custom non-default ciphers.
    let key2_req = PutHpkeConfigReq {
        kem_id: Some(HpkeKemId::X25519HkdfSha256),
        kdf_id: Some(HpkeKdfId::HkdfSha512),
        aead_id: Some(HpkeAeadId::ChaCha20Poly1305),
    };
    let mut key2_resp = put("/hpke_configs")
        .with_request_body(serde_json::to_vec(&key2_req).unwrap())
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;

    assert_response!(key1_resp, Status::Created);
    let key2: HpkeConfigResp = serde_json::from_slice(
        &key2_resp
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();

    let (got_key1, got_key2) = ds
        .run_unnamed_tx(|tx| {
            let key1 = key1.config.clone();
            let key2 = key2.config.clone();
            Box::pin(async move {
                Ok((
                    tx.get_hpke_keypair(key1.id()).await?,
                    tx.get_hpke_keypair(key2.id()).await?,
                ))
            })
        })
        .await
        .unwrap();

    assert_eq!(
        key1,
        HpkeConfigResp {
            config: got_key1.unwrap().hpke_keypair().config().clone(),
            state: HpkeKeyState::Pending,
        }
    );

    assert_eq!(
        key2,
        HpkeConfigResp {
            config: got_key2.unwrap().hpke_keypair().config().clone(),
            state: HpkeKeyState::Pending,
        }
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        put("/hpke_configs")
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );
}

#[tokio::test]
async fn patch_hpke_config() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let req = PatchHpkeConfigReq {
        state: HpkeKeyState::Active,
    };

    // Verify: non-existent key.
    assert_response!(
        patch("/hpke_configs/123")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NotFound
    );

    // Verify: overflow u8.
    assert_response!(
        patch("/hpke_configs/1234310294")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::BadRequest
    );

    // Verify: invalid body.
    assert_response!(
        patch("/hpke_configs/1234310294")
            .with_request_body("{}")
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::UnprocessableEntity
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        patch("/hpke_configs/123")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );

    let keypair = HpkeKeypair::test();
    ds.run_unnamed_tx(|tx| {
        let keypair = keypair.clone();
        Box::pin(async move { tx.put_hpke_keypair(&keypair).await })
    })
    .await
    .unwrap();

    let conn = patch(format!("/hpke_configs/{}", keypair.config().id()))
        .with_request_body(serde_json::to_vec(&req).unwrap())
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_response!(conn, Status::Ok);

    let got_key = ds
        .run_unnamed_tx(|tx| {
            let keypair = keypair.clone();
            Box::pin(async move { tx.get_hpke_keypair(keypair.config().id()).await })
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got_key.state(), &HpkeKeyState::Active);
}

#[tokio::test]
async fn delete_hpke_config() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let req = PatchHpkeConfigReq {
        state: HpkeKeyState::Active,
    };

    // Verify: non-existent key.
    assert_response!(
        delete("/hpke_configs/123")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NoContent
    );

    // Verify: overflow u8.
    assert_response!(
        delete("/hpke_configs/1234310294")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::BadRequest
    );

    // Verify: unauthorized requests are denied appropriately.
    assert_response!(
        delete("/hpke_configs/123")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Accept", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized,
        "",
    );

    let keypair = HpkeKeypair::test();
    ds.run_unnamed_tx(|tx| {
        let keypair = keypair.clone();
        Box::pin(async move { tx.put_hpke_keypair(&keypair).await })
    })
    .await
    .unwrap();

    let conn = delete(format!("/hpke_configs/{}", keypair.config().id()))
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_response!(conn, Status::NoContent);

    assert_eq!(
        ds.run_unnamed_tx(|tx| Box::pin(async move { tx.get_hpke_keypairs().await }))
            .await
            .unwrap(),
        vec![]
    );
}

#[tokio::test]
async fn get_taskprov_peer_aggregator() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let leader = PeerAggregatorBuilder::new()
        .with_endpoint(Url::parse("https://leader.example.com/").unwrap())
        .with_peer_role(Role::Leader)
        .build();
    let helper = PeerAggregatorBuilder::new()
        .with_endpoint(Url::parse("https://helper.example.com/").unwrap())
        .with_peer_role(Role::Helper)
        .build();

    ds.run_unnamed_tx(|tx| {
        let leader = leader.clone();
        let helper = helper.clone();
        Box::pin(async move {
            tx.put_taskprov_peer_aggregator(&leader).await?;
            tx.put_taskprov_peer_aggregator(&helper).await?;
            Ok(())
        })
    })
    .await
    .unwrap();

    // List all.
    let mut conn = get("/taskprov/peer_aggregators")
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_response!(conn, Status::Ok);
    let mut resp: Vec<TaskprovPeerAggregatorResp> = serde_json::from_slice(
        &conn
            .take_response_body()
            .unwrap()
            .into_bytes()
            .await
            .unwrap(),
    )
    .unwrap();
    resp.sort_by(|a, b| a.endpoint.cmp(&b.endpoint));

    let mut expected = vec![
        TaskprovPeerAggregatorResp {
            endpoint: leader.endpoint().clone(),
            peer_role: *leader.peer_role(),
            collector_hpke_config: leader.collector_hpke_config().clone(),
            report_expiry_age: leader.report_expiry_age().cloned(),
            tolerable_clock_skew: *leader.tolerable_clock_skew(),
        },
        TaskprovPeerAggregatorResp {
            endpoint: helper.endpoint().clone(),
            peer_role: *helper.peer_role(),
            collector_hpke_config: helper.collector_hpke_config().clone(),
            report_expiry_age: helper.report_expiry_age().cloned(),
            tolerable_clock_skew: *helper.tolerable_clock_skew(),
        },
    ];
    expected.sort_by(|a, b| a.endpoint.cmp(&b.endpoint));

    assert_eq!(resp, expected);

    // Missing authorization.
    assert_response!(
        get("/taskprov/peer_aggregators")
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized
    );
}

#[tokio::test]
async fn post_taskprov_peer_aggregator() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let endpoint = Url::parse("https://leader.example.com/").unwrap();
    let leader = PeerAggregatorBuilder::new()
        .with_endpoint(endpoint.clone())
        .with_peer_role(Role::Leader)
        .build();

    let req = PostTaskprovPeerAggregatorReq {
        endpoint,
        peer_role: Role::Leader,
        aggregation_mode: Some(AggregationMode::Synchronous),
        collector_hpke_config: leader.collector_hpke_config().clone(),
        verify_key_init: *leader.verify_key_init(),
        report_expiry_age: leader.report_expiry_age().cloned(),
        tolerable_clock_skew: *leader.tolerable_clock_skew(),
        aggregator_auth_tokens: Vec::from(leader.aggregator_auth_tokens()),
        collector_auth_tokens: Vec::from(leader.collector_auth_tokens()),
    };

    let mut conn = post("/taskprov/peer_aggregators")
        .with_request_body(serde_json::to_vec(&req).unwrap())
        .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
        .with_request_header("Accept", CONTENT_TYPE)
        .with_request_header("Content-Type", CONTENT_TYPE)
        .run_async(&handler)
        .await;
    assert_response!(conn, Status::Created);
    assert_eq!(
        serde_json::from_slice::<TaskprovPeerAggregatorResp>(
            &conn
                .take_response_body()
                .unwrap()
                .into_bytes()
                .await
                .unwrap(),
        )
        .unwrap(),
        leader.clone().into()
    );

    assert_eq!(
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move { tx.get_taskprov_peer_aggregators().await })
        })
        .await
        .unwrap(),
        vec![leader]
    );

    // Can't insert the same aggregator.
    assert_response!(
        post("/taskprov/peer_aggregators")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Conflict
    );

    // Missing authorization.
    assert_response!(
        post("/taskprov/peer_aggregators")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized
    );
}

#[tokio::test]
async fn delete_taskprov_peer_aggregator() {
    let (handler, _ephemeral_datastore, ds) = setup_api_test().await;

    let endpoint = Url::parse("https://leader.example.com/").unwrap();
    let leader = PeerAggregatorBuilder::new()
        .with_endpoint(endpoint.clone())
        .with_peer_role(Role::Leader)
        .build();

    ds.run_unnamed_tx(|tx| {
        let leader = leader.clone();
        Box::pin(async move { tx.put_taskprov_peer_aggregator(&leader).await })
    })
    .await
    .unwrap();

    let req = DeleteTaskprovPeerAggregatorReq {
        endpoint,
        peer_role: Role::Leader,
    };

    // Delete target.
    assert_response!(
        delete("/taskprov/peer_aggregators")
            .with_request_body(serde_json::to_vec(&req).unwrap())
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NoContent
    );

    assert_eq!(
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move { tx.get_taskprov_peer_aggregators().await })
        })
        .await
        .unwrap(),
        vec![]
    );

    // Non-existent target.
    assert_response!(
        delete("/taskprov/peer_aggregators")
            .with_request_body(
                serde_json::to_vec(&DeleteTaskprovPeerAggregatorReq {
                    endpoint: Url::parse("https://doesnt-exist.example.com/").unwrap(),
                    peer_role: Role::Leader,
                })
                .unwrap()
            )
            .with_request_header("Authorization", format!("Bearer {AUTH_TOKEN}"))
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::NoContent
    );

    // Missing authorization.
    assert_response!(
        delete("/taskprov/peer_aggregators")
            .with_request_header("Accept", CONTENT_TYPE)
            .with_request_header("Content-Type", CONTENT_TYPE)
            .run_async(&handler)
            .await,
        Status::Unauthorized
    );
}

#[test]
fn get_task_ids_resp_serialization() {
    assert_ser_tokens(
        &GetTaskIdsResp {
            task_ids: Vec::from([TaskId::from([0u8; 32])]),
            pagination_token: None,
        },
        &[
            Token::Struct {
                name: "GetTaskIdsResp",
                len: 1,
            },
            Token::Str("task_ids"),
            Token::Seq { len: Some(1) },
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::SeqEnd,
            Token::StructEnd,
        ],
    );
    assert_ser_tokens(
        &GetTaskIdsResp {
            task_ids: Vec::from([TaskId::from([0u8; 32])]),
            pagination_token: Some(TaskId::from([0u8; 32])),
        },
        &[
            Token::Struct {
                name: "GetTaskIdsResp",
                len: 2,
            },
            Token::Str("task_ids"),
            Token::Seq { len: Some(1) },
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::SeqEnd,
            Token::Str("pagination_token"),
            Token::Some,
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::StructEnd,
        ],
    );
}

#[test]
fn post_task_req_serialization() {
    // helper request with optional fields omitted
    assert_tokens(
        &PostTaskReq {
            peer_aggregator_endpoint: "https://example.com/".parse().unwrap(),
            batch_mode: BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            aggregation_mode: Some(AggregationMode::Synchronous),
            vdaf: VdafInstance::Prio3SumVec {
                bits: 1,
                length: 5,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
            role: Role::Helper,
            vdaf_verify_key: "encoded".to_owned(),
            task_start: None,
            task_end: None,
            min_batch_size: 100,
            time_precision: Duration::from_seconds(3600),
            collector_hpke_config: HpkeConfig::new(
                HpkeConfigId::from(7),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
                HpkePublicKey::from([0u8; 32].to_vec()),
            ),
            aggregator_auth_token: None,
            collector_auth_token_hash: None,
        },
        &[
            Token::Struct {
                name: "PostTaskReq",
                len: 13,
            },
            Token::Str("peer_aggregator_endpoint"),
            Token::Str("https://example.com/"),
            Token::Str("batch_mode"),
            Token::StructVariant {
                name: "BatchMode",
                variant: "LeaderSelected",
                len: 1,
            },
            Token::Str("batch_time_window_size"),
            Token::None,
            Token::StructVariantEnd,
            Token::Str("aggregation_mode"),
            Token::Some,
            Token::UnitVariant {
                name: "AggregationMode",
                variant: "Synchronous",
            },
            Token::Str("vdaf"),
            Token::StructVariant {
                name: "VdafInstance",
                variant: "Prio3SumVec",
                len: 4,
            },
            Token::Str("bits"),
            Token::U64(1),
            Token::Str("length"),
            Token::U64(5),
            Token::Str("chunk_length"),
            Token::U64(2),
            Token::Str("dp_strategy"),
            Token::Struct {
                name: "Prio3SumVec",
                len: 1,
            },
            Token::Str("dp_strategy"),
            Token::Str("NoDifferentialPrivacy"),
            Token::StructEnd,
            Token::StructVariantEnd,
            Token::Str("role"),
            Token::UnitVariant {
                name: "Role",
                variant: "Helper",
            },
            Token::Str("vdaf_verify_key"),
            Token::Str("encoded"),
            Token::Str("task_start"),
            Token::None,
            Token::Str("task_end"),
            Token::None,
            Token::Str("min_batch_size"),
            Token::U64(100),
            Token::Str("time_precision"),
            Token::NewtypeStruct { name: "Duration" },
            Token::U64(3600),
            Token::Str("collector_hpke_config"),
            Token::Struct {
                name: "HpkeConfig",
                len: 5,
            },
            Token::Str("id"),
            Token::NewtypeStruct {
                name: "HpkeConfigId",
            },
            Token::U8(7),
            Token::Str("kem_id"),
            Token::UnitVariant {
                name: "HpkeKemId",
                variant: "X25519HkdfSha256",
            },
            Token::Str("kdf_id"),
            Token::UnitVariant {
                name: "HpkeKdfId",
                variant: "HkdfSha256",
            },
            Token::Str("aead_id"),
            Token::UnitVariant {
                name: "HpkeAeadId",
                variant: "Aes128Gcm",
            },
            Token::Str("public_key"),
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::StructEnd,
            Token::Str("aggregator_auth_token"),
            Token::None,
            Token::Str("collector_auth_token_hash"),
            Token::None,
            Token::StructEnd,
        ],
    );

    // leader request with optional fields set
    assert_tokens(
        &PostTaskReq {
            peer_aggregator_endpoint: "https://example.com/".parse().unwrap(),
            batch_mode: BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            aggregation_mode: None,
            vdaf: VdafInstance::Prio3SumVec {
                bits: 1,
                length: 5,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
            role: Role::Leader,
            vdaf_verify_key: "encoded".to_owned(),
            task_start: Some(Time::from_seconds_since_epoch(500)),
            task_end: Some(Time::from_seconds_since_epoch(1000)),
            min_batch_size: 100,
            time_precision: Duration::from_seconds(3600),
            collector_hpke_config: HpkeConfig::new(
                HpkeConfigId::from(7),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
                HpkePublicKey::from([0u8; 32].to_vec()),
            ),
            aggregator_auth_token: Some(
                AuthenticationToken::new_dap_auth_token_from_string("ZW5jb2RlZA").unwrap(),
            ),
            collector_auth_token_hash: Some(AuthenticationTokenHash::from(
                &AuthenticationToken::new_dap_auth_token_from_string("ZW5jb2RlZA").unwrap(),
            )),
        },
        &[
            Token::Struct {
                name: "PostTaskReq",
                len: 13,
            },
            Token::Str("peer_aggregator_endpoint"),
            Token::Str("https://example.com/"),
            Token::Str("batch_mode"),
            Token::StructVariant {
                name: "BatchMode",
                variant: "LeaderSelected",
                len: 1,
            },
            Token::Str("batch_time_window_size"),
            Token::None,
            Token::StructVariantEnd,
            Token::Str("aggregation_mode"),
            Token::None,
            Token::Str("vdaf"),
            Token::StructVariant {
                name: "VdafInstance",
                variant: "Prio3SumVec",
                len: 4,
            },
            Token::Str("bits"),
            Token::U64(1),
            Token::Str("length"),
            Token::U64(5),
            Token::Str("chunk_length"),
            Token::U64(2),
            Token::Str("dp_strategy"),
            Token::Struct {
                name: "Prio3SumVec",
                len: 1,
            },
            Token::Str("dp_strategy"),
            Token::Str("NoDifferentialPrivacy"),
            Token::StructEnd,
            Token::StructVariantEnd,
            Token::Str("role"),
            Token::UnitVariant {
                name: "Role",
                variant: "Leader",
            },
            Token::Str("vdaf_verify_key"),
            Token::Str("encoded"),
            Token::Str("task_start"),
            Token::Some,
            Token::NewtypeStruct { name: "Time" },
            Token::U64(500),
            Token::Str("task_end"),
            Token::Some,
            Token::NewtypeStruct { name: "Time" },
            Token::U64(1000),
            Token::Str("min_batch_size"),
            Token::U64(100),
            Token::Str("time_precision"),
            Token::NewtypeStruct { name: "Duration" },
            Token::U64(3600),
            Token::Str("collector_hpke_config"),
            Token::Struct {
                name: "HpkeConfig",
                len: 5,
            },
            Token::Str("id"),
            Token::NewtypeStruct {
                name: "HpkeConfigId",
            },
            Token::U8(7),
            Token::Str("kem_id"),
            Token::UnitVariant {
                name: "HpkeKemId",
                variant: "X25519HkdfSha256",
            },
            Token::Str("kdf_id"),
            Token::UnitVariant {
                name: "HpkeKdfId",
                variant: "HkdfSha256",
            },
            Token::Str("aead_id"),
            Token::UnitVariant {
                name: "HpkeAeadId",
                variant: "Aes128Gcm",
            },
            Token::Str("public_key"),
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::StructEnd,
            Token::Str("aggregator_auth_token"),
            Token::Some,
            Token::Struct {
                name: "AuthenticationToken",
                len: 2,
            },
            Token::Str("type"),
            Token::UnitVariant {
                name: "AuthenticationToken",
                variant: "DapAuth",
            },
            Token::Str("token"),
            Token::Str("ZW5jb2RlZA"),
            Token::StructEnd,
            Token::Str("collector_auth_token_hash"),
            Token::Some,
            Token::Struct {
                name: "AuthenticationTokenHash",
                len: 2,
            },
            Token::Str("type"),
            Token::UnitVariant {
                name: "AuthenticationTokenHash",
                variant: "DapAuth",
            },
            Token::Str("hash"),
            Token::Str("hT_ixzv_X1CmJmHGT7jYSEBbdB-CN9H8WxAvjgv4rms"),
            Token::StructEnd,
            Token::StructEnd,
        ],
    );
}

#[test]
fn task_resp_serialization() {
    let task = AggregatorTask::new(
        TaskId::from([0u8; 32]),
        "https://helper.com/".parse().unwrap(),
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
        VdafInstance::Prio3SumVec {
            bits: 1,
            length: 5,
            chunk_length: 2,
            dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
        },
        SecretBytes::new(b"vdaf verify key!".to_vec()),
        None,
        None,
        None,
        100,
        Duration::from_seconds(3600),
        Duration::from_seconds(60),
        AggregatorTaskParameters::Leader {
            aggregator_auth_token: AuthenticationToken::new_dap_auth_token_from_string(
                "Y29sbGVjdG9yLWFiY2RlZjAw",
            )
            .unwrap(),
            collector_auth_token_hash: AuthenticationTokenHash::from(
                &AuthenticationToken::new_dap_auth_token_from_string("Y29sbGVjdG9yLWFiY2RlZjAw")
                    .unwrap(),
            ),
            collector_hpke_config: HpkeConfig::new(
                HpkeConfigId::from(7),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
                HpkePublicKey::from([0u8; 32].to_vec()),
            ),
        },
    )
    .unwrap();
    assert_tokens(
        &TaskResp::try_from(&task).unwrap(),
        &[
            Token::Struct {
                name: "TaskResp",
                len: 14,
            },
            Token::Str("task_id"),
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::Str("peer_aggregator_endpoint"),
            Token::Str("https://helper.com/"),
            Token::Str("batch_mode"),
            Token::StructVariant {
                name: "BatchMode",
                variant: "LeaderSelected",
                len: 1,
            },
            Token::Str("batch_time_window_size"),
            Token::None,
            Token::StructVariantEnd,
            Token::Str("vdaf"),
            Token::StructVariant {
                name: "VdafInstance",
                variant: "Prio3SumVec",
                len: 4,
            },
            Token::Str("bits"),
            Token::U64(1),
            Token::Str("length"),
            Token::U64(5),
            Token::Str("chunk_length"),
            Token::U64(2),
            Token::Str("dp_strategy"),
            Token::Struct {
                name: "Prio3SumVec",
                len: 1,
            },
            Token::Str("dp_strategy"),
            Token::Str("NoDifferentialPrivacy"),
            Token::StructEnd,
            Token::StructVariantEnd,
            Token::Str("role"),
            Token::UnitVariant {
                name: "Role",
                variant: "Leader",
            },
            Token::Str("vdaf_verify_key"),
            Token::Str("dmRhZiB2ZXJpZnkga2V5IQ"),
            Token::Str("task_start"),
            Token::None,
            Token::Str("task_end"),
            Token::None,
            Token::Str("report_expiry_age"),
            Token::None,
            Token::Str("min_batch_size"),
            Token::U64(100),
            Token::Str("time_precision"),
            Token::NewtypeStruct { name: "Duration" },
            Token::U64(3600),
            Token::Str("tolerable_clock_skew"),
            Token::NewtypeStruct { name: "Duration" },
            Token::U64(60),
            Token::Str("aggregator_auth_token"),
            Token::None,
            Token::Str("collector_hpke_config"),
            Token::Struct {
                name: "HpkeConfig",
                len: 5,
            },
            Token::Str("id"),
            Token::NewtypeStruct {
                name: "HpkeConfigId",
            },
            Token::U8(7),
            Token::Str("kem_id"),
            Token::UnitVariant {
                name: "HpkeKemId",
                variant: "X25519HkdfSha256",
            },
            Token::Str("kdf_id"),
            Token::UnitVariant {
                name: "HpkeKdfId",
                variant: "HkdfSha256",
            },
            Token::Str("aead_id"),
            Token::UnitVariant {
                name: "HpkeAeadId",
                variant: "Aes128Gcm",
            },
            Token::Str("public_key"),
            Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Token::StructEnd,
            Token::StructEnd,
        ],
    );
}

#[test]
fn get_task_upload_metrics_serialization() {
    assert_ser_tokens(
        &GetTaskUploadMetricsResp(TaskUploadCounter::new_with_values(
            0, 1, 2, 3, 4, 5, 6, 7, 8,
        )),
        &[
            Token::NewtypeStruct {
                name: "GetTaskUploadMetricsResp",
            },
            Token::Struct {
                name: "TaskUploadCounter",
                len: 9,
            },
            Token::Str("interval_collected"),
            Token::U64(0),
            Token::Str("report_decode_failure"),
            Token::U64(1),
            Token::Str("report_decrypt_failure"),
            Token::U64(2),
            Token::Str("report_expired"),
            Token::U64(3),
            Token::Str("report_outdated_key"),
            Token::U64(4),
            Token::Str("report_success"),
            Token::U64(5),
            Token::Str("report_too_early"),
            Token::U64(6),
            Token::Str("task_not_started"),
            Token::U64(7),
            Token::Str("task_ended"),
            Token::U64(8),
            Token::StructEnd,
        ],
    )
}
