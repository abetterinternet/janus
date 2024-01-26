#[cfg(feature = "testcontainer")]
use crate::common::test_task_builder;
use crate::common::{submit_measurements_and_verify_aggregate, test_task_builder_host};
use janus_aggregator_core::task::QueryType;
#[cfg(feature = "testcontainer")]
use janus_core::test_util::testcontainers::container_client;
use janus_core::{test_util::install_test_trace_subscriber, vdaf::VdafInstance};
#[cfg(feature = "testcontainer")]
use janus_integration_tests::janus::JanusContainer;
use janus_integration_tests::{client::ClientBackend, janus::JanusInProcess, TaskParameters};
#[cfg(feature = "testcontainer")]
use janus_interop_binaries::test_util::generate_network_name;
use janus_messages::Role;
use std::time::Duration;
#[cfg(feature = "testcontainer")]
use testcontainers::clients::Cli;

/// A pair of Janus instances, running in containers, against which integration tests may be run.
#[cfg(feature = "testcontainer")]
struct JanusContainerPair<'a> {
    /// Task parameters needed by the client and collector, for the task configured in both Janus
    /// aggregators.
    task_parameters: TaskParameters,

    /// Handle to the leader's resources, which are released on drop.
    leader: JanusContainer<'a>,
    /// Handle to the helper's resources, which are released on drop.
    helper: JanusContainer<'a>,
}

#[cfg(feature = "testcontainer")]
impl<'a> JanusContainerPair<'a> {
    /// Set up a new pair of containerized Janus test instances, and set up a new task in each using
    /// the given VDAF and query type.
    pub async fn new(
        test_name: &str,
        container_client: &'a Cli,
        vdaf: VdafInstance,
        query_type: QueryType,
    ) -> JanusContainerPair<'a> {
        let (task_parameters, task_builder) = test_task_builder(
            vdaf,
            query_type,
            Duration::from_millis(500),
            Duration::from_secs(60),
        );
        let task = task_builder.build();

        let network = generate_network_name();
        let leader =
            JanusContainer::new(test_name, container_client, &network, &task, Role::Leader).await;
        let helper =
            JanusContainer::new(test_name, container_client, &network, &task, Role::Helper).await;

        Self {
            task_parameters,
            leader,
            helper,
        }
    }
}

/// A pair of Janus instances, running in-process, against which integration tests may be run.
struct JanusInProcessPair {
    /// Task parameters needed by the client and collector, for the task configured in both Janus
    /// aggregators.
    task_parameters: TaskParameters,

    /// The leader's resources, which are released on drop.
    leader: JanusInProcess,
    /// The helper's resources, which are released on drop.
    helper: JanusInProcess,
}

impl JanusInProcessPair {
    /// Set up a new pair of in-process Janus test instances, and set up a new task in each using
    /// the given VDAF and query type.
    pub async fn new(vdaf: VdafInstance, query_type: QueryType) -> JanusInProcessPair {
        let (task_parameters, mut task_builder) = test_task_builder_host(
            vdaf,
            query_type,
            Duration::from_millis(500),
            Duration::from_secs(60),
        );

        let helper = JanusInProcess::new(&task_builder.clone().build(), Role::Helper).await;
        let helper_url = task_parameters
            .endpoint_fragments
            .helper
            .endpoint_for_host(helper.port());
        task_builder = task_builder.with_helper_aggregator_endpoint(helper_url);
        let leader = JanusInProcess::new(&task_builder.build(), Role::Leader).await;

        Self {
            task_parameters,
            leader,
            helper,
        }
    }
}

/// This test exercises Prio3Count with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "testcontainer")]
async fn janus_janus_count() {
    static TEST_NAME: &str = "janus_janus_count";
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusContainerPair::new(
        TEST_NAME,
        &container_client,
        VdafInstance::Prio3Count,
        QueryType::TimeInterval,
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Count with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_in_process_count() {
    install_test_trace_subscriber();

    // Start servers.
    let janus_pair =
        JanusInProcessPair::new(VdafInstance::Prio3Count, QueryType::TimeInterval).await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "janus_in_process_count",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Sum with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "testcontainer")]
async fn janus_janus_sum_16() {
    static TEST_NAME: &str = "janus_janus_sum_16";
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusContainerPair::new(
        TEST_NAME,
        &container_client,
        VdafInstance::Prio3Sum { bits: 16 },
        QueryType::TimeInterval,
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Sum with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_in_process_sum_16() {
    install_test_trace_subscriber();

    // Start servers.
    let janus_pair =
        JanusInProcessPair::new(VdafInstance::Prio3Sum { bits: 16 }, QueryType::TimeInterval).await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "janus_in_process_sum_16",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Histogram with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "testcontainer")]
async fn janus_janus_histogram_4_buckets() {
    static TEST_NAME: &str = "janus_janus_histogram_4_buckets";
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusContainerPair::new(
        TEST_NAME,
        &container_client,
        VdafInstance::Prio3Histogram {
            length: 4,
            chunk_length: 2,
        },
        QueryType::TimeInterval,
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Sum with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_in_process_histogram_4_buckets() {
    install_test_trace_subscriber();

    // Start servers.
    let janus_pair = JanusInProcessPair::new(
        VdafInstance::Prio3Histogram {
            length: 4,
            chunk_length: 2,
        },
        QueryType::TimeInterval,
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "janus_in_process_histogram_4_buckets",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises the fixed-size query type with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "testcontainer")]
async fn janus_janus_fixed_size() {
    static TEST_NAME: &str = "janus_janus_fixed_size";
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusContainerPair::new(
        TEST_NAME,
        &container_client,
        VdafInstance::Prio3Count,
        QueryType::FixedSize {
            max_batch_size: Some(50),
            batch_time_window_size: None,
        },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises the fixed-size query type with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_in_process_fixed_size() {
    install_test_trace_subscriber();

    // Start servers.
    let janus_pair = JanusInProcessPair::new(
        VdafInstance::Prio3Count,
        QueryType::FixedSize {
            max_batch_size: Some(50),
            batch_time_window_size: None,
        },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "janus_in_process_fixed_size",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3SumVec with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "testcontainer")]
async fn janus_janus_sum_vec() {
    static TEST_NAME: &str = "janus_janus_sum_vec";
    install_test_trace_subscriber();

    let container_client = container_client();
    let janus_pair = JanusContainerPair::new(
        TEST_NAME,
        &container_client,
        VdafInstance::Prio3SumVec {
            bits: 16,
            length: 15,
            chunk_length: 16,
        },
        QueryType::TimeInterval,
    )
    .await;

    submit_measurements_and_verify_aggregate(
        TEST_NAME,
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3SumVec with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_in_process_sum_vec() {
    install_test_trace_subscriber();

    let janus_pair = JanusInProcessPair::new(
        VdafInstance::Prio3SumVec {
            bits: 16,
            length: 15,
            chunk_length: 16,
        },
        QueryType::TimeInterval,
    )
    .await;

    submit_measurements_and_verify_aggregate(
        "",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}
