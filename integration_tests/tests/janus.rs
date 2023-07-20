use common::{submit_measurements_and_verify_aggregate, test_task_builders};
use janus_aggregator_core::task::QueryType;
use janus_core::{
    task::VdafInstance,
    test_util::{install_test_trace_subscriber, testcontainers::container_client},
};
use janus_integration_tests::{client::ClientBackend, janus::Janus, TaskParameters};
use janus_interop_binaries::test_util::generate_network_name;
use testcontainers::clients::Cli;

mod common;

/// A pair of Janus instances, running in containers, against which integration tests may be run.
struct JanusPair<'a> {
    /// Task parameters needed by the client and collector, for the task configured in both Janus
    /// aggregators.
    task_parameters: TaskParameters,

    /// Handle to the leader's resources, which are released on drop.
    leader: Janus<'a>,
    /// Handle to the helper's resources, which are released on drop.
    helper: Janus<'a>,
}

impl<'a> JanusPair<'a> {
    /// Set up a new pair of containerized Janus test instances, and set up a new task in each using
    /// the given VDAF and query type.
    pub async fn new(
        container_client: &'a Cli,
        vdaf: VdafInstance,
        query_type: QueryType,
    ) -> JanusPair<'a> {
        let (task_parameters, leader_task, helper_task) = test_task_builders(vdaf, query_type);

        let network = generate_network_name();
        let leader = Janus::new(container_client, &network, &leader_task.build()).await;
        let helper = Janus::new(container_client, &network, &helper_task.build()).await;

        Self {
            task_parameters,
            leader,
            helper,
        }
    }
}

/// This test exercises Prio3Count with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_janus_count() {
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusPair::new(
        &container_client,
        VdafInstance::Prio3Count,
        QueryType::FixedSize { max_batch_size: 46 },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Sum with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_janus_sum_16() {
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusPair::new(
        &container_client,
        VdafInstance::Prio3Sum { bits: 16 },
        QueryType::FixedSize { max_batch_size: 46 },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3Histogram with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_janus_histogram_4_buckets() {
    install_test_trace_subscriber();

    let buckets = Vec::from([3, 6, 8]);

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusPair::new(
        &container_client,
        VdafInstance::Prio3Histogram { buckets },
        QueryType::FixedSize { max_batch_size: 46 },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

/// This test exercises Prio3CountVec with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_janus_count_vec_15() {
    install_test_trace_subscriber();

    // Start servers.
    let container_client = container_client();
    let janus_pair = JanusPair::new(
        &container_client,
        VdafInstance::Prio3CountVec { length: 15 },
        QueryType::FixedSize { max_batch_size: 46 },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

// This test exercises Prio3SumVec with Janus as both the leader and the helper.
#[tokio::test(flavor = "multi_thread")]
async fn janus_janus_sum_vec() {
    install_test_trace_subscriber();

    let container_client = container_client();
    let janus_pair = JanusPair::new(
        &container_client,
        VdafInstance::Prio3SumVec {
            bits: 16,
            length: 15,
        },
        QueryType::FixedSize { max_batch_size: 46 },
    )
    .await;

    submit_measurements_and_verify_aggregate(
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}
