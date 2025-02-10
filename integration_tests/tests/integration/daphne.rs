use crate::{
    common::{build_test_task, submit_measurements_and_verify_aggregate, TestContext},
    initialize_rustls,
};
use janus_aggregator_core::task::{test_util::TaskBuilder, AggregationMode, BatchMode};
use janus_core::{test_util::install_test_trace_subscriber, vdaf::VdafInstance};
#[cfg(feature = "testcontainer")]
use janus_integration_tests::janus::JanusContainer;
use janus_integration_tests::{
    client::ClientBackend, daphne::Daphne, janus::JanusInProcess, AggregatorEndpointFragments,
};
use janus_interop_binaries::test_util::generate_network_name;
use janus_messages::Role;
use std::time::Duration;

const VERSION_PATH: &str = "/v09/";

// This test places Daphne in the leader role & Janus in the helper role.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Daphne does not yet publish a leader container image"]
#[cfg(feature = "testcontainer")]
async fn daphne_janus() {
    static TEST_NAME: &str = "daphne_janus";
    install_test_trace_subscriber();
    initialize_rustls();

    // Start servers.
    let network = generate_network_name();
    let (mut task_parameters, task_builder) = build_test_task(
        TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        ),
        TestContext::VirtualNetwork,
        Duration::from_millis(500),
        Duration::from_secs(60),
    );

    // Daphne is hardcoded to serve from a path starting with /v09/.
    task_parameters
        .endpoint_fragments
        .leader
        .set_path(VERSION_PATH.to_string());
    let mut leader_aggregator_endpoint = task_builder.leader_aggregator_endpoint().clone();
    leader_aggregator_endpoint.set_path(VERSION_PATH);
    let task = task_builder
        .with_leader_aggregator_endpoint(leader_aggregator_endpoint)
        .build();

    let leader = Daphne::new(TEST_NAME, &network, &task, Role::Leader, true).await;
    let helper = JanusContainer::new(TEST_NAME, &network, &task, Role::Helper).await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &task_parameters,
        (leader.port(), helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

// This test places Janus in the leader role & Daphne in the helper role.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Daphne does not currently support DAP-07 (issue #1669)"]
#[cfg(feature = "testcontainer")]
async fn janus_daphne() {
    static TEST_NAME: &str = "janus_daphne";
    install_test_trace_subscriber();
    initialize_rustls();

    // Start servers.
    let network = generate_network_name();
    let (mut task_parameters, task_builder) = build_test_task(
        TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        ),
        TestContext::VirtualNetwork,
        Duration::from_millis(500),
        Duration::from_secs(60),
    );

    // Daphne is hardcoded to serve from a path starting with /v09/.
    task_parameters
        .endpoint_fragments
        .helper
        .set_path(VERSION_PATH.to_string());
    let mut helper_aggregator_endpoint = task_builder.helper_aggregator_endpoint().clone();
    helper_aggregator_endpoint.set_path(VERSION_PATH);
    let task = task_builder
        .with_helper_aggregator_endpoint(helper_aggregator_endpoint)
        .build();

    let leader = JanusContainer::new(TEST_NAME, &network, &task, Role::Leader).await;
    let helper = Daphne::new(TEST_NAME, &network, &task, Role::Helper, true).await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &task_parameters,
        (leader.port(), helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test places Janus in the leader role and Daphne in the helper role. Janus is run
/// in-process, while Daphne is run in Docker.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Daphne does not currently support DAP-13"]
async fn janus_in_process_daphne() {
    static TEST_NAME: &str = "janus_in_process_daphne";
    install_test_trace_subscriber();
    initialize_rustls();

    // Start servers.
    let network = generate_network_name();
    let (mut task_parameters, mut task_builder) = build_test_task(
        TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        ),
        TestContext::VirtualNetwork,
        Duration::from_millis(500),
        Duration::from_secs(60),
    );
    task_parameters.endpoint_fragments.leader = AggregatorEndpointFragments::Localhost {
        path: "/".to_owned(),
    };
    task_parameters
        .endpoint_fragments
        .helper
        .set_path(VERSION_PATH.to_owned());
    let helper = Daphne::new(
        TEST_NAME,
        &network,
        &task_builder.clone().build(),
        Role::Helper,
        true,
    )
    .await;
    task_builder = task_builder.with_helper_aggregator_endpoint(
        task_parameters
            .endpoint_fragments
            .helper
            .endpoint_for_host(helper.port()),
    );
    let leader = JanusInProcess::new(&task_builder.build(), Role::Leader).await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &task_parameters,
        (leader.port(), helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}
