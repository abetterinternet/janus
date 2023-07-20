use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use futures::future::join_all;
use janus_core::{
    task::PRIO3_VERIFY_KEY_LENGTH,
    test_util::{install_test_trace_subscriber, testcontainers::container_client},
};
use janus_interop_binaries::{
    test_util::{await_ready_ok, generate_network_name, generate_unique_name},
    testcontainer::{Aggregator, Client, Collector},
    ContainerLogsDropGuard,
};
use janus_messages::{
    query_type::{FixedSize, QueryType},
    TaskId,
};
use prio::codec::Encode;
use rand::random;
use reqwest::{header::CONTENT_TYPE, StatusCode, Url};
use serde_json::{json, Value};
use std::time::Duration as StdDuration;
use testcontainers::RunnableImage;
use tokio::time::sleep;

#[cfg(feature = "fpvec_bounded_l2")]
use fixed_macro::fixed;

const JSON_MEDIA_TYPE: &str = "application/json";
const TIME_PRECISION: u64 = 3600;

enum QueryKind {
    FixedSize,
}

/// Take a VDAF description and a list of measurements, perform an entire aggregation using
/// interoperation test binaries, and return the aggregate result. This follows the outline of
/// the "Test Runner Operation" section in draft-dcook-ppm-dap-interop-test-design.
async fn run(
    query_kind: QueryKind,
    vdaf_object: serde_json::Value,
    measurements: &[serde_json::Value],
    aggregation_parameter: &[u8],
) -> serde_json::Value {
    install_test_trace_subscriber();

    let (query_type_json, max_batch_size) = match query_kind {
        QueryKind::FixedSize => {
            let query_type = json!(FixedSize::CODE as u8);
            (query_type, Some(json!(measurements.len())))
        }
    };

    // Create and start containers.
    let container_client = container_client();
    let network = generate_network_name();

    let client_container = ContainerLogsDropGuard::new(
        container_client.run(
            RunnableImage::from(Client::default())
                .with_network(&network)
                .with_container_name(generate_unique_name("client")),
        ),
    );
    let client_port = client_container.get_host_port_ipv4(Client::INTERNAL_SERVING_PORT);

    let leader_name = generate_unique_name("leader");
    let leader_container = ContainerLogsDropGuard::new(
        container_client.run(
            RunnableImage::from(Aggregator::default())
                .with_network(&network)
                .with_container_name(leader_name.clone()),
        ),
    );
    let leader_port = leader_container.get_host_port_ipv4(Aggregator::INTERNAL_SERVING_PORT);

    let helper_name = generate_unique_name("helper");
    let helper_container = ContainerLogsDropGuard::new(
        container_client.run(
            RunnableImage::from(Aggregator::default())
                .with_network(&network)
                .with_container_name(helper_name.clone()),
        ),
    );
    let helper_port = helper_container.get_host_port_ipv4(Aggregator::INTERNAL_SERVING_PORT);

    let collector_container = ContainerLogsDropGuard::new(
        container_client.run(
            RunnableImage::from(Collector::default())
                .with_network(&network)
                .with_container_name(generate_unique_name("collector")),
        ),
    );
    let collector_port = collector_container.get_host_port_ipv4(Collector::INTERNAL_SERVING_PORT);

    // Wait for all containers to sucessfully respond to HTTP requests.
    join_all(
        [client_port, leader_port, helper_port, collector_port]
            .into_iter()
            .map(await_ready_ok),
    )
    .await;

    // Generate a random TaskId, random authentication tokens, and a VDAF verification key.
    let task_id: TaskId = random();
    let aggregator_auth_token = URL_SAFE_NO_PAD.encode(random::<[u8; 16]>());
    let collector_auth_token = URL_SAFE_NO_PAD.encode(random::<[u8; 16]>());
    let vdaf_verify_key = rand::random::<[u8; PRIO3_VERIFY_KEY_LENGTH]>();

    let task_id_encoded = URL_SAFE_NO_PAD.encode(task_id.get_encoded());
    let vdaf_verify_key_encoded = URL_SAFE_NO_PAD.encode(vdaf_verify_key);

    // Endpoints, from the POV of this test (i.e. the Docker host).
    let local_client_endpoint = Url::parse(&format!("http://127.0.0.1:{client_port}/")).unwrap();
    let local_leader_endpoint = Url::parse(&format!("http://127.0.0.1:{leader_port}/")).unwrap();
    let local_helper_endpoint = Url::parse(&format!("http://127.0.0.1:{helper_port}/")).unwrap();
    let local_collector_endpoint =
        Url::parse(&format!("http://127.0.0.1:{collector_port}/")).unwrap();

    // Endpoints, from the POV of the containers (i.e. the Docker network).
    let internal_leader_endpoint = Url::parse(&format!("http://{leader_name}:8080/")).unwrap();
    let internal_helper_endpoint = Url::parse(&format!("http://{helper_name}:8080/")).unwrap();

    let http_client = reqwest::Client::new();

    // Send a /internal/test/endpoint_for_task request to the leader.
    let leader_endpoint_response = http_client
        .post(
            local_leader_endpoint
                .join("/internal/test/endpoint_for_task")
                .unwrap(),
        )
        .json(&json!({
            "task_id": task_id_encoded,
            "role": "leader",
            "hostname": local_leader_endpoint.host_str().unwrap(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(leader_endpoint_response.status(), StatusCode::OK);
    assert_eq!(
        leader_endpoint_response
            .headers()
            .get(CONTENT_TYPE)
            .unwrap(),
        JSON_MEDIA_TYPE,
    );
    let leader_endpoint_response_body = leader_endpoint_response.json::<Value>().await.unwrap();
    let leader_endpoint_response_object = leader_endpoint_response_body
        .as_object()
        .expect("endpoint_for_task response is not an object");
    assert_eq!(
        leader_endpoint_response_object
            .get("status")
            .expect("endpoint_for_task response is missing \"status\""),
        "success",
        "error: {:?}",
        leader_endpoint_response_object.get("error"),
    );
    assert_eq!(
        leader_endpoint_response_object
            .get("endpoint")
            .expect("endpoint_for_task response is missing \"endpoint\""),
        "/",
    );

    // Send a /internal/test/endpoint_for_task request to the helper.
    let helper_endpoint_response = http_client
        .post(
            local_helper_endpoint
                .join("/internal/test/endpoint_for_task")
                .unwrap(),
        )
        .json(&json!({
            "task_id": task_id_encoded,
            "role": "helper",
            "hostname": local_helper_endpoint.host_str().unwrap(),
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(helper_endpoint_response.status(), StatusCode::OK);
    assert_eq!(
        helper_endpoint_response
            .headers()
            .get(CONTENT_TYPE)
            .unwrap(),
        JSON_MEDIA_TYPE,
    );
    let helper_endpoint_response_body = helper_endpoint_response.json::<Value>().await.unwrap();
    let helper_endpoint_response_object = helper_endpoint_response_body
        .as_object()
        .expect("endpoint_for_task response is not an object");
    assert_eq!(
        helper_endpoint_response_object
            .get("status")
            .expect("endpoint_for_task response is missing \"status\""),
        "success",
        "error: {:?}",
        helper_endpoint_response_object.get("error"),
    );
    assert_eq!(
        helper_endpoint_response_object
            .get("endpoint")
            .expect("endpoint_for_task response is missing \"endpoint\""),
        "/",
    );

    // Send a /internal/test/add_task request to the collector.
    let collector_add_task_response = http_client
        .post(
            local_collector_endpoint
                .join("/internal/test/add_task")
                .unwrap(),
        )
        .json(&json!({
            "task_id": task_id_encoded,
            "leader": internal_leader_endpoint,
            "vdaf": vdaf_object,
            "collector_authentication_token": collector_auth_token,
            "query_type": query_type_json,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(collector_add_task_response.status(), StatusCode::OK);
    assert_eq!(
        collector_add_task_response
            .headers()
            .get(CONTENT_TYPE)
            .unwrap(),
        JSON_MEDIA_TYPE,
    );
    let collector_add_task_response_body =
        collector_add_task_response.json::<Value>().await.unwrap();
    let collector_add_task_response_object = collector_add_task_response_body
        .as_object()
        .expect("collector add_task response is not an object");
    assert_eq!(
        collector_add_task_response_object
            .get("status")
            .expect("collector add_task response is missing \"status\""),
        "success",
        "error: {:?}",
        collector_add_task_response_object.get("error"),
    );
    let collector_hpke_config_encoded = collector_add_task_response_object
        .get("collector_hpke_config")
        .expect("collector add_task response is missing \"collector_hpke_config\"")
        .as_str()
        .expect("\"collector_hpke_config\" value is not a string");

    // Send a /internal/test/add_task request to the leader.
    let mut leader_add_task_request_body = json!({
        "task_id": task_id_encoded,
        "leader": internal_leader_endpoint,
        "helper": internal_helper_endpoint,
        "vdaf": vdaf_object,
        "leader_authentication_token": aggregator_auth_token,
        "collector_authentication_token": collector_auth_token,
        "role": "leader",
        "vdaf_verify_key": vdaf_verify_key_encoded,
        "max_batch_query_count": 1,
        "query_type": query_type_json,
        "min_batch_size": measurements.len(),
        "time_precision": TIME_PRECISION,
        "collector_hpke_config": collector_hpke_config_encoded,
    });
    if let Some(max_batch_size) = &max_batch_size {
        leader_add_task_request_body
            .as_object_mut()
            .unwrap()
            .insert("max_batch_size".to_string(), max_batch_size.clone());
    }
    let leader_add_task_response = http_client
        .post(
            local_leader_endpoint
                .join("/internal/test/add_task")
                .unwrap(),
        )
        .json(&leader_add_task_request_body)
        .send()
        .await
        .unwrap();
    assert_eq!(leader_add_task_response.status(), StatusCode::OK);
    assert_eq!(
        leader_add_task_response
            .headers()
            .get(CONTENT_TYPE)
            .unwrap(),
        JSON_MEDIA_TYPE,
    );
    let leader_add_task_response_body = leader_add_task_response.json::<Value>().await.unwrap();
    let leader_add_task_response_object = leader_add_task_response_body
        .as_object()
        .expect("leader add_task response is not an object");
    assert_eq!(
        leader_add_task_response_object
            .get("status")
            .expect("leader add_task response is missing \"status\""),
        "success",
        "error: {:?}",
        leader_add_task_response_object.get("error"),
    );

    // Send a /internal/test/add_task request to the helper.
    let mut helper_add_task_request_body = json!({
        "task_id": task_id_encoded,
        "leader": internal_leader_endpoint,
        "helper": internal_helper_endpoint,
        "vdaf": vdaf_object,
        "leader_authentication_token": aggregator_auth_token,
        "role": "helper",
        "vdaf_verify_key": vdaf_verify_key_encoded,
        "max_batch_query_count": 1,
        "query_type": query_type_json,
        "min_batch_size": measurements.len(),
        "time_precision": TIME_PRECISION,
        "collector_hpke_config": collector_hpke_config_encoded,
    });
    if let Some(max_batch_size) = &max_batch_size {
        helper_add_task_request_body
            .as_object_mut()
            .unwrap()
            .insert("max_batch_size".to_string(), max_batch_size.clone());
    }
    let helper_add_task_response = http_client
        .post(
            local_helper_endpoint
                .join("/internal/test/add_task")
                .unwrap(),
        )
        .json(&helper_add_task_request_body)
        .send()
        .await
        .unwrap();
    assert_eq!(helper_add_task_response.status(), StatusCode::OK);
    assert_eq!(
        helper_add_task_response
            .headers()
            .get(CONTENT_TYPE)
            .unwrap(),
        JSON_MEDIA_TYPE,
    );
    let helper_add_task_response_body = helper_add_task_response.json::<Value>().await.unwrap();
    let helper_add_task_response_object = helper_add_task_response_body
        .as_object()
        .expect("helper add_task response is not an object");
    assert_eq!(
        helper_add_task_response_object
            .get("status")
            .expect("helper add_task response is missing \"status\""),
        "success",
        "error: {:?}",
        helper_add_task_response_object.get("error"),
    );

    // Send one or more /internal/test/upload requests to the client.
    for measurement in measurements {
        let upload_response = http_client
            .post(local_client_endpoint.join("/internal/test/upload").unwrap())
            .json(&json!({
                "task_id": task_id_encoded,
                "leader": internal_leader_endpoint,
                "helper": internal_helper_endpoint,
                "vdaf": vdaf_object,
                "measurement": measurement,
                "time_precision": TIME_PRECISION,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(upload_response.status(), StatusCode::OK);
        assert_eq!(
            upload_response.headers().get(CONTENT_TYPE).unwrap(),
            JSON_MEDIA_TYPE,
        );
        let upload_response_body = upload_response.json::<Value>().await.unwrap();
        let upload_response_object = upload_response_body
            .as_object()
            .expect("upload response is not an object");
        assert_eq!(
            upload_response_object
                .get("status")
                .expect("upload response is missing \"status\""),
            "success",
            "error: {:?}",
            upload_response_object.get("error"),
        );
    }

    let query_json = match query_kind {
        QueryKind::FixedSize => {
            json!({
                "type": query_type_json,
                "subtype": 1, // current_batch
            })
        }
    };

    // Try collecting one or more times. For fixed size tasks, a "current batch" query will fail
    // with an invalid batch until enough reports are ready.
    let mut collect_attempt_backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(StdDuration::from_secs(1))
        .with_max_interval(StdDuration::from_secs(1))
        .with_max_elapsed_time(match query_kind {
            QueryKind::FixedSize => Some(StdDuration::from_secs(15)),
        })
        .build();
    loop {
        // Send a /internal/test/collection_start request to the collector.
        let collection_start_response = http_client
            .post(
                local_collector_endpoint
                    .join("/internal/test/collection_start")
                    .unwrap(),
            )
            .json(&json!({
                "task_id": task_id_encoded,
                "agg_param": URL_SAFE_NO_PAD.encode(aggregation_parameter, ),
                "query": query_json,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(collection_start_response.status(), StatusCode::OK);
        assert_eq!(
            collection_start_response
                .headers()
                .get(CONTENT_TYPE)
                .unwrap(),
            JSON_MEDIA_TYPE,
        );
        let collection_start_response_body =
            collection_start_response.json::<Value>().await.unwrap();
        let collection_start_response_object = collection_start_response_body
            .as_object()
            .expect("collection_start response is not an object");
        assert_eq!(
            collection_start_response_object
                .get("status")
                .expect("collection_start response is missing \"status\""),
            "success",
            "error: {:?}",
            collection_start_response_object.get("error"),
        );
        let collection_job_handle = collection_start_response_object
            .get("handle")
            .expect("collection_start response is missing \"handle\"")
            .as_str()
            .expect("\"handle\" value is not a string");

        // Send /internal/test/collection_poll requests to the collector, polling until it is completed.
        let mut collection_poll_backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(StdDuration::from_millis(500))
            .with_max_interval(StdDuration::from_millis(500))
            .with_max_elapsed_time(Some(StdDuration::from_secs(60)))
            .build();
        let (status, collection_poll_response_object) = loop {
            let collection_poll_response = http_client
                .post(
                    local_collector_endpoint
                        .join("/internal/test/collection_poll")
                        .unwrap(),
                )
                .json(&json!({
                    "handle": collection_job_handle,
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(collection_poll_response.status(), StatusCode::OK);
            assert_eq!(
                collection_poll_response
                    .headers()
                    .get(CONTENT_TYPE)
                    .unwrap(),
                JSON_MEDIA_TYPE,
            );
            let collection_poll_response_body =
                collection_poll_response.json::<Value>().await.unwrap();
            let collection_poll_response_object = collection_poll_response_body
                .as_object()
                .expect("collection_poll response is not an object");
            let status = collection_poll_response_object
                .get("status")
                .expect("collection_poll response is missing \"status\"")
                .as_str()
                .expect("\"status\" value is not a string");
            if status == "in progress" {
                if let Some(duration) = collection_poll_backoff.next_backoff() {
                    sleep(duration).await;
                    continue;
                } else {
                    panic!("timed out fetching aggregation result");
                }
            }
            break (status.to_owned(), collection_poll_response_object.clone());
        };

        if status == "error" {
            if let Some(duration) = collect_attempt_backoff.next_backoff() {
                sleep(duration).await;
                continue;
            } else {
                panic!("timed out waiting for collect to succeed");
            }
        }
        assert_eq!(
            status,
            "complete",
            "error: {:?}",
            collection_poll_response_object.get("error"),
        );
        let batch_id_encoded = collection_poll_response_object
            .get("batch_id")
            .expect("completed collection_poll response is missing \"batch_id\"")
            .as_str()
            .expect("\"batch_id\" value is not a string");
        URL_SAFE_NO_PAD.decode(batch_id_encoded).unwrap();
        collection_poll_response_object
            .get("report_count")
            .expect("completed collection_poll response is missing \"report_count\"")
            .as_u64()
            .expect("\"report_count\" is not an integer");
        collection_poll_response_object
            .get("interval_start")
            .expect("completed collection_poll response is missing \"interval_start\"")
            .as_f64()
            .expect("\"interval_start\" is not a number");
        collection_poll_response_object
            .get("interval_duration")
            .expect("completed collection_poll response is missing \"interval_duration\"")
            .as_f64()
            .expect("\"interval_duration\" is not a number");
        return collection_poll_response_object
            .get("result")
            .expect("completed collection_poll response is missing \"result\"")
            .clone();
    }
}

#[tokio::test]
async fn e2e_prio3_count() {
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3Count"}),
        &[
            json!("0"),
            json!("1"),
            json!("1"),
            json!("0"),
            json!("1"),
            json!("0"),
            json!("1"),
            json!("0"),
            json!("1"),
            json!("1"),
            json!("0"),
            json!("1"),
            json!("0"),
            json!("1"),
            json!("0"),
            json!("0"),
            json!("0"),
            json!("0"),
        ],
        b"",
    )
    .await;
    assert!(result.is_string());
}

#[tokio::test]
async fn e2e_prio3_sum() {
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3Sum", "bits": "64"}),
        &[
            json!("0"),
            json!("10"),
            json!("9"),
            json!("21"),
            json!("8"),
            json!("12"),
            json!("14"),
        ],
        b"",
    )
    .await;
    assert!(result.is_string());
}

#[tokio::test]
async fn e2e_prio3_sum_vec() {
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3SumVec", "bits": "64", "length": "4"}),
        &[
            json!(["0", "0", "0", "10"]),
            json!(["0", "0", "10", "0"]),
            json!(["0", "10", "0", "0"]),
            json!(["10", "0", "0", "0"]),
        ],
        b"",
    )
    .await;
    for element in result.as_array().expect("SumVec result should be an array") {
        assert!(element.is_string());
    }
}

#[tokio::test]
async fn e2e_prio3_histogram() {
    let result = run(
        QueryKind::FixedSize,
        json!({
            "type": "Prio3Histogram",
            "buckets": ["0", "1", "10", "100", "1000", "10000", "100000"],
        }),
        &[
            json!("1"),
            json!("4"),
            json!("16"),
            json!("64"),
            json!("256"),
            json!("1024"),
            json!("4096"),
            json!("16384"),
            json!("65536"),
            json!("262144"),
        ],
        b"",
    )
    .await;
    for element in result
        .as_array()
        .expect("Histogram result should be an array")
    {
        assert!(element.is_string());
    }
}

#[tokio::test]
async fn e2e_prio3_count_vec() {
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3CountVec", "length": "4"}),
        &[
            json!(["0", "0", "0", "1"]),
            json!(["0", "0", "1", "0"]),
            json!(["0", "1", "0", "0"]),
            json!(["1", "0", "0", "0"]),
        ],
        b"",
    )
    .await;
    for element in result
        .as_array()
        .expect("CountVec result should be an array")
    {
        assert!(element.is_string());
    }
}

#[tokio::test]
async fn e2e_prio3_fixed16vec() {
    let fp16_4_inv = fixed!(0.25: I1F15);
    let fp16_8_inv = fixed!(0.125: I1F15);
    let fp16_16_inv = fixed!(0.0625: I1F15);
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3FixedPoint16BitBoundedL2VecSum", "length": "3"}),
        &[
            json!([
                fp16_4_inv.to_string(),
                fp16_8_inv.to_string(),
                fp16_8_inv.to_string()
            ]),
            json!([
                fp16_16_inv.to_string(),
                fp16_8_inv.to_string(),
                fp16_16_inv.to_string()
            ]),
            json!([
                fp16_8_inv.to_string(),
                fp16_8_inv.to_string(),
                fp16_4_inv.to_string()
            ]),
            json!([
                fp16_16_inv.to_string(),
                fp16_8_inv.to_string(),
                fp16_4_inv.to_string()
            ]),
        ],
        b"",
    )
    .await;
    assert_eq!(result, json!(["0.5", "0.5", "0.6875"]));
}

#[tokio::test]
async fn e2e_prio3_fixed32vec() {
    let fp32_4_inv = fixed!(0.25: I1F31);
    let fp32_8_inv = fixed!(0.125: I1F31);
    let fp32_16_inv = fixed!(0.0625: I1F31);
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3FixedPoint32BitBoundedL2VecSum", "length": "3"}),
        &[
            json!([
                fp32_4_inv.to_string(),
                fp32_8_inv.to_string(),
                fp32_8_inv.to_string()
            ]),
            json!([
                fp32_16_inv.to_string(),
                fp32_8_inv.to_string(),
                fp32_16_inv.to_string()
            ]),
            json!([
                fp32_8_inv.to_string(),
                fp32_8_inv.to_string(),
                fp32_4_inv.to_string()
            ]),
            json!([
                fp32_16_inv.to_string(),
                fp32_8_inv.to_string(),
                fp32_4_inv.to_string()
            ]),
        ],
        b"",
    )
    .await;
    assert_eq!(result, json!(["0.5", "0.5", "0.6875"]));
}

#[tokio::test]
async fn e2e_prio3_fixed64vec() {
    let fp64_4_inv = fixed!(0.25: I1F63);
    let fp64_8_inv = fixed!(0.125: I1F63);
    let fp64_16_inv = fixed!(0.0625: I1F63);
    let result = run(
        QueryKind::FixedSize,
        json!({"type": "Prio3FixedPoint64BitBoundedL2VecSum", "length": "3"}),
        &[
            json!([
                fp64_4_inv.to_string(),
                fp64_8_inv.to_string(),
                fp64_8_inv.to_string()
            ]),
            json!([
                fp64_16_inv.to_string(),
                fp64_8_inv.to_string(),
                fp64_16_inv.to_string()
            ]),
            json!([
                fp64_8_inv.to_string(),
                fp64_8_inv.to_string(),
                fp64_4_inv.to_string()
            ]),
            json!([
                fp64_16_inv.to_string(),
                fp64_8_inv.to_string(),
                fp64_4_inv.to_string()
            ]),
        ],
        b"",
    )
    .await;
    assert_eq!(result, json!(["0.5", "0.5", "0.6875"]));
}
