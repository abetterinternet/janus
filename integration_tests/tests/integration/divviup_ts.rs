#![cfg(feature = "testcontainer")]
//! These tests check interoperation between the divviup-ts client and Janus aggregators.

use crate::{
    common::{build_test_task, submit_measurements_and_verify_aggregate, TestContext},
    initialize_rustls,
};
use janus_aggregator_core::task::{test_util::TaskBuilder, AggregationMode, BatchMode};
use janus_core::{
    test_util::install_test_trace_subscriber,
    vdaf::{vdaf_dp_strategies, VdafInstance},
};
use janus_integration_tests::{
    client::{ClientBackend, InteropClient},
    janus::JanusContainer,
};
use janus_interop_binaries::test_util::generate_network_name;
use janus_messages::Role;
use std::time::Duration;

async fn run_divviup_ts_integration_test(test_name: &str, vdaf: VdafInstance) {
    let (task_parameters, task_builder) = build_test_task(
        TaskBuilder::new(BatchMode::TimeInterval, AggregationMode::Synchronous, vdaf),
        TestContext::VirtualNetwork,
        Duration::from_millis(500),
        Duration::from_secs(60),
    );
    let task = task_builder.build();
    let network = generate_network_name();
    let leader = JanusContainer::new(test_name, &network, &task, Role::Leader).await;
    let helper = JanusContainer::new(test_name, &network, &task, Role::Helper).await;

    let client_backend = ClientBackend::Container {
        container_image: InteropClient::divviup_ts(),
        network: &network,
    };
    submit_measurements_and_verify_aggregate(
        test_name,
        &task_parameters,
        (leader.port(), helper.port()),
        &client_backend,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "divviup-ts does not expose a DAP-13-compatible image (issue #3479)"]
async fn janus_divviup_ts_count() {
    install_test_trace_subscriber();
    initialize_rustls();

    run_divviup_ts_integration_test("janus_divviup_ts_count", VdafInstance::Prio3Count).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "divviup-ts does not expose a DAP-13-compatible image (issue #3479)"]
async fn janus_divviup_ts_sum() {
    install_test_trace_subscriber();
    initialize_rustls();

    run_divviup_ts_integration_test(
        "janus_divviup_ts_sum",
        VdafInstance::Prio3Sum {
            max_measurement: 255,
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "divviup-ts does not expose a DAP-13-compatible image (issue #3479)"]
async fn janus_divviup_ts_histogram() {
    install_test_trace_subscriber();
    initialize_rustls();

    run_divviup_ts_integration_test(
        "janus_divviup_ts_histogram",
        VdafInstance::Prio3Histogram {
            length: 4,
            chunk_length: 2,
            dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "divviup-ts does not expose a DAP-13-compatible image (issue #3479)"]
async fn janus_divviup_ts_sumvec() {
    install_test_trace_subscriber();
    initialize_rustls();

    run_divviup_ts_integration_test(
        "janus_divviup_ts_sumvec",
        VdafInstance::Prio3SumVec {
            bits: 16,
            length: 15,
            chunk_length: 16,
            dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
        },
    )
    .await;
}
