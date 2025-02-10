#![cfg(feature = "in-cluster")]

use crate::{
    common::{
        build_test_task, collect_aggregate_result_generic,
        submit_measurements_and_verify_aggregate, submit_measurements_generic, TestContext,
    },
    initialize_rustls,
};
use chrono::prelude::*;
use clap::{CommandFactory, FromArgMatches, Parser};
use divviup_client::{
    Client, DivviupClient, Histogram, HpkeConfig, NewAggregator, NewSharedAggregator, NewTask,
    SumVec, Vdaf,
};
use janus_aggregator_core::task::{test_util::TaskBuilder, AggregationMode, BatchMode};
#[cfg(feature = "ohttp")]
use janus_client::OhttpConfig;
use janus_collector::PrivateCollectorCredential;
use janus_core::{
    auth_tokens::AuthenticationToken,
    hpke::HpkeKeypair,
    test_util::{
        install_test_trace_subscriber,
        kubernetes::{Cluster, PortForward},
    },
    time::DurationExt,
    vdaf::{vdaf_dp_strategies, VdafInstance},
};
use janus_integration_tests::{client::ClientBackend, TaskParameters};
use janus_messages::{Duration as JanusDuration, TaskId};
use prio::{
    dp::{
        distributions::PureDpDiscreteLaplace, DifferentialPrivacyStrategy, PureDpBudget, Rational,
    },
    field::{Field128, FieldElementWithInteger},
    vdaf::prio3::Prio3,
};
use std::{env, iter, str::FromStr, time::Duration};
use trillium_rustls::RustlsConfig;
use trillium_tokio::ClientConfig;
use url::Url;
use uuid::Uuid;

/// Options for running tests.
#[derive(Debug, Parser)]
#[clap(
    name = "janus-integration-tests",
    about = "Janus integration test driver",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Options {
    /// If set, the integration tests will be run against remote instances of `divviup-api`, a Janus
    /// leader and a Janus helper. If not set, the integration tests will be run against instances
    /// of `divviup-api`, a Janus leader and a Janus helper in an adjacent Kubernetes cluster.
    ///
    /// See doccomments on InClusterJanusPair::new_in_cloud and InClusterJanusPair::new_in_kind for
    /// discussion of how to configure this test setup.
    #[arg(long, default_value = "false")]
    in_cloud: Option<bool>,
}

struct InClusterJanusPair {
    /// Task parameters needed by the client and collector, for the task configured in both Janus
    /// aggregators.
    task_parameters: TaskParameters,

    /// Handle to the leader's resources, which are released on drop.
    leader: InClusterJanus,
    /// Handle to the helper's resources, which are released on drop.
    helper: InClusterJanus,
}

impl InClusterJanusPair {
    /// Set up a new DAP task, using the given VDAF and batch mode, in a pair of existing Janus
    /// instances, which can be running either in a local Kubernetes cluster accessed over port-
    /// forwards, or remotely, depending on whether the `--in-cloud` flag is set.
    /// `divviup-api` is used to provision the task into the aggregators.
    async fn new(vdaf: VdafInstance, batch_mode: BatchMode) -> Self {
        // The test invocation will be like
        // `cargo test <args for cargo> -- <args for test runner> -- <args for Janus tests>`. We
        // want to parse just that last set of arguments into `struct Options` so we:
        // Get all the args from environment
        let test_args = std::env::args_os()
            // Start from the end of the iterator and take until we encounter the argument separator
            .rev()
            .take_while(|arg| *arg != "--")
            // Tack on a fake argv[0] as otherwise clap can't parse the arguments
            .chain(["janus-integration-tests".into()])
            // Collect back into a Vec because [`std::iter::TakeWhile`] is not a
            // [`DoubleEndedIterator`].
            .collect::<Vec<_>>()
            // Reverse to get the last chunk of args in the correct order.
            .into_iter()
            .rev();

        let options = Options::from_arg_matches(
            &Options::command()
                // Parse arguments permissively so that an invocation like
                // `cargo test <args for cargo> -- <args for test runner>` or
                // `cargo test <args for cargo> -- <args for test runner> --` will be accepted.
                .ignore_errors(true)
                .get_matches_from(test_args),
        )
        .unwrap();

        if options.in_cloud == Some(true) {
            Self::new_in_cloud(vdaf, batch_mode).await
        } else {
            Self::new_in_kind(vdaf, batch_mode).await
        }
    }

    /// Set up a new DAP task in a pair of aggregators. `divviup-api` is used to provision the task
    /// into the two aggregators. Unlike [`Self::new_in_kind`], this does not create a new account
    /// and pair new aggregators. While tasks created by this test can eventually be garbage
    /// collected, accounts and paired aggregators would (currently) hang around forever.
    ///
    /// The following environment variables must be set:
    ///
    ///  - `JANUS_E2E_DIVVIUP_API_URL` (URL): API endpoint for `divviup-api`, used to provision
    ///    tasks.
    ///  - `JANUS_E2E_DIVVIUP_API_TOKEN` (Bearer token): API token with which to authenticate to the
    ///    `divviup-api` instance at `JANUS_E2E_DIVVIUP_API_URL` as a member of account
    ///    `JANUS_E2E_DIVVIUP_ACCOUNT_ID`.
    ///  - `JANUS_E2E_DIVVIUP_ACCOUNT_ID` (UUID): Account ID in which to create a task.
    ///  - `JANUS_E2E_LEADER_AGGREGATOR_ID` (UUID): ID of the aggregator to use as DAP leader in the
    ///    task.
    ///  - `JANUS_E2E_HELPER_AGGREGATOR_ID` (UUID): ID of the aggregator to use as DAP helper in the
    ///    task.
    ///  - `JANUS_E2E_COLLECTOR_CREDENTIAL_ID` (UUID): ID of the collector credential to use when
    ///    collecting aggregate shares in the task.
    ///  - `JANUS_E2E_COLLECTOR_CREDENTIAL_JSON`: JSON representation of the collector credential,
    ///    including the auth token and the HPKE private key. Example:
    ///
    ///    {
    ///      "aead": "AesGcm128",
    ///      "id": 66,
    ///      "kdf": "Sha256",
    ///      "kem": "X25519HkdfSha256",
    ///      "private_key": "uKkTvzKLfYNUPZcoKI7hV64zS06OWgBkbivBL4Sw4mo",
    ///      "public_key": "CcDghts2boltt9GQtBUxdUsVR83SCVYHikcGh33aVlU",
    ///      "token": "Krx-CLfdWo1ULAfsxhr0rA"
    ///    }
    async fn new_in_cloud(vdaf: VdafInstance, batch_mode: BatchMode) -> Self {
        let (
            divviup_api_url,
            divviup_api_token,
            divviup_account_id,
            leader_aggregator_id,
            helper_aggregator_id,
            collector_credential_id,
            collector_credential,
        ) = match (
            env::var("JANUS_E2E_DIVVIUP_API_URL"),
            env::var("JANUS_E2E_DIVVIUP_API_TOKEN"),
            env::var("JANUS_E2E_DIVVIUP_ACCOUNT_ID"),
            env::var("JANUS_E2E_LEADER_AGGREGATOR_ID"),
            env::var("JANUS_E2E_HELPER_AGGREGATOR_ID"),
            env::var("JANUS_E2E_COLLECTOR_CREDENTIAL_ID"),
            env::var("JANUS_E2E_COLLECTOR_CREDENTIAL_JSON"),
        ) {
            (
                Ok(divviup_api_url),
                Ok(divviup_api_token),
                Ok(divviup_account_id),
                Ok(leader_aggregator_id),
                Ok(helper_aggregator_id),
                Ok(collector_credential_id),
                Ok(collector_credential_json),
            ) => (
                divviup_api_url.parse().unwrap(),
                divviup_api_token,
                Uuid::parse_str(&divviup_account_id).unwrap(),
                Uuid::parse_str(&leader_aggregator_id).unwrap(),
                Uuid::parse_str(&helper_aggregator_id).unwrap(),
                Uuid::parse_str(&collector_credential_id).unwrap(),
                serde_json::from_str::<PrivateCollectorCredential>(&collector_credential_json)
                    .unwrap(),
            ),
            _ => panic!("missing or invalid environment variables"),
        };

        let divviup_api = DivviupClient::new(
            divviup_api_token,
            Client::new(RustlsConfig::<ClientConfig>::default()),
        )
        .with_default_pool()
        .with_url(divviup_api_url);

        let aggregators = divviup_api.aggregators(divviup_account_id).await.unwrap();
        let leader_aggregator_dap_url = aggregators
            .iter()
            .find(|a| a.id == leader_aggregator_id)
            .map(|a| a.dap_url.clone())
            .unwrap();
        let helper_aggregator_dap_url = aggregators
            .iter()
            .find(|a| a.id == helper_aggregator_id)
            .map(|a| a.dap_url.clone())
            .unwrap();

        let (task_parameters, task_builder) = build_test_task(
            TaskBuilder::new(batch_mode, AggregationMode::Synchronous, vdaf)
                .with_leader_aggregator_endpoint(leader_aggregator_dap_url)
                .with_helper_aggregator_endpoint(helper_aggregator_dap_url),
            TestContext::Remote,
            Duration::from_secs(30),
            Duration::from_secs(600),
        );

        Self::new_common(
            divviup_api,
            divviup_account_id,
            task_parameters,
            task_builder,
            leader_aggregator_id,
            helper_aggregator_id,
            collector_credential_id,
            collector_credential.authentication_token(),
            collector_credential.hpke_keypair(),
            InClusterJanus {
                aggregator_port_forward: None,
            },
            InClusterJanus {
                aggregator_port_forward: None,
            },
        )
        .await
    }

    /// Set up a new DAP task, using the given VDAF and batch mode, in a pair of existing Janus
    /// instances in a Kubernetes cluster. `divviup-api` is used to create an account, pair both
    /// aggregators and configure the task in each Janus instance. The following environment
    /// variables must be set.
    ///
    ///  - `JANUS_E2E_KUBE_CONFIG_PATH`: The path to a kubeconfig file, containing the information
    ///    needed to connect to the cluster.
    ///  - `JANUS_E2E_KUBECTL_CONTEXT_NAME`: The name of a context in the kubeconfig file.
    ///  - `JANUS_E2E_LEADER_NAMESPACE`: The Kubernetes namespace where the DAP leader is deployed.
    ///  - `JANUS_E2E_LEADER_AGGREGATOR_API_AUTH_TOKEN`: Credential with which requests to the
    ///     leader's aggregator API are authenticated.
    ///  - `JANUS_E2E_HELPER_NAMESPACE`: The Kubernetes namespace where the DAP helper is deployed.
    ///  - `JANUS_E2E_HELPER_AGGREGATOR_API_AUTH_TOKEN`: Credential with which requests to the
    ///     helper's aggregator API are authenticated.
    ///  - `JANUS_E2E_DIVVIUP_API_NAMESPACE`: The Kubernetes namespace where `divviup-api` is
    ///     deployed.
    async fn new_in_kind(vdaf: VdafInstance, batch_mode: BatchMode) -> Self {
        let (
            kubeconfig_path,
            kubectl_context_name,
            leader_namespace,
            leader_aggregator_api_auth_token,
            helper_namespace,
            helper_aggregator_api_auth_token,
            divviup_api_namespace,
        ) = match (
            env::var("JANUS_E2E_KUBE_CONFIG_PATH"),
            env::var("JANUS_E2E_KUBECTL_CONTEXT_NAME"),
            env::var("JANUS_E2E_LEADER_NAMESPACE"),
            env::var("JANUS_E2E_LEADER_AGGREGATOR_API_AUTH_TOKEN"),
            env::var("JANUS_E2E_HELPER_NAMESPACE"),
            env::var("JANUS_E2E_HELPER_AGGREGATOR_API_AUTH_TOKEN"),
            env::var("JANUS_E2E_DIVVIUP_API_NAMESPACE"),
        ) {
            (
                Ok(kubeconfig_path),
                Ok(kubectl_context_name),
                Ok(leader_namespace),
                Ok(leader_aggregator_api_auth_token),
                Ok(helper_namespace),
                Ok(helper_aggregator_api_auth_token),
                Ok(divviup_api_namespace),
            ) => (
                kubeconfig_path,
                kubectl_context_name,
                leader_namespace,
                leader_aggregator_api_auth_token,
                helper_namespace,
                helper_aggregator_api_auth_token,
                divviup_api_namespace,
            ),
            _ => panic!("missing or invalid environment variables"),
        };

        let cluster = Cluster::new(&kubeconfig_path, &kubectl_context_name);

        let (task_parameters, task_builder) = build_test_task(
            TaskBuilder::new(batch_mode, AggregationMode::Synchronous, vdaf),
            TestContext::VirtualNetwork,
            Duration::from_millis(500),
            Duration::from_secs(60),
        );

        // From outside the cluster, the aggregators are reached at a dynamically allocated port on
        // localhost. When the aggregators talk to each other, they do so in the cluster's network,
        // so they need the in-cluster DNS name of the other aggregator, and they can use well-known
        // service port numbers.

        let port_forward = cluster
            .forward_port(&divviup_api_namespace, "divviup-api", 80)
            .await;
        let port = port_forward.local_port();

        let divviup_api = DivviupClient::new(
            "DUATignored".to_string(),
            Client::new(RustlsConfig::<ClientConfig>::default()),
        )
        .with_default_pool()
        .with_url(format!("http://127.0.0.1:{port}").parse().unwrap());

        // Create an account first. (We should be implicitly logged in as a testing user already,
        // assuming divviup-api was built with the integration-testing feature)
        let account = divviup_api
            .create_account("Integration test account")
            .await
            .unwrap();

        // Pair the aggregators. The same Janus instances will get paired multiple times across
        // multiple tests, but it's to a different divviup-api account each time, so that's
        // harmless. The leader aggregator is paired as a *global* aggregator using the admin
        // endpoint. We do this for two reasons:
        //
        // - setting up tasks with one global aggregator and one per-account aggregator is most
        //   representative of the subscriber use cases Divvi Up supports,
        let paired_leader_aggregator = divviup_api
            .create_shared_aggregator(NewSharedAggregator {
                name: "leader".to_string(),
                api_url: Self::in_cluster_aggregator_api_url(&leader_namespace),
                bearer_token: leader_aggregator_api_auth_token,
                is_first_party: true,
            })
            .await
            .unwrap();

        let paired_helper_aggregator = divviup_api
            .create_aggregator(
                account.id,
                NewAggregator {
                    name: "helper".to_string(),
                    api_url: Self::in_cluster_aggregator_api_url(&helper_namespace),
                    bearer_token: helper_aggregator_api_auth_token,
                },
            )
            .await
            .unwrap();

        let hpke_keypair = task_builder.collector_hpke_keypair().clone();
        let hpke_config = hpke_keypair.config();
        let collector_credential = divviup_api
            .create_collector_credential(
                account.id,
                &HpkeConfig::new(
                    u8::from(*hpke_config.id()).into(),
                    u16::from(*hpke_config.kem_id()).into(),
                    u16::from(*hpke_config.kdf_id()).into(),
                    u16::from(*hpke_config.aead_id()).into(),
                    hpke_config.public_key().as_ref().to_vec().into(),
                ),
                Some("Integration test key"),
            )
            .await
            .unwrap();

        Self::new_common(
            divviup_api,
            account.id,
            task_parameters,
            task_builder,
            paired_leader_aggregator.id,
            paired_helper_aggregator.id,
            collector_credential.id,
            collector_credential
                .token
                .map(|t| AuthenticationToken::new_bearer_token_from_string(t).unwrap())
                .unwrap(),
            hpke_keypair.clone(),
            InClusterJanus::new(&cluster, &leader_namespace).await,
            InClusterJanus::new(&cluster, &helper_namespace).await,
        )
        .await
    }

    fn in_cluster_aggregator_api_url(namespace: &str) -> Url {
        Url::parse(&format!(
            "http://aggregator.{namespace}.svc.cluster.local:80/aggregator-api/"
        ))
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    async fn new_common(
        divviup_api: DivviupClient,
        divviup_account_id: Uuid,
        mut task_parameters: TaskParameters,
        task_builder: TaskBuilder,
        leader_aggregator_id: Uuid,
        helper_aggregator_id: Uuid,
        collector_credential_id: Uuid,
        collector_auth_token: AuthenticationToken,
        collector_hpke_keypair: HpkeKeypair,
        leader: InClusterJanus,
        helper: InClusterJanus,
    ) -> Self {
        let task = task_builder.with_min_batch_size(100).build();
        task_parameters.min_batch_size = 100;

        let provision_task_request = NewTask {
            name: format!("Integration test task {}", Utc::now().to_rfc3339()),
            leader_aggregator_id,
            helper_aggregator_id,
            vdaf: match task.vdaf().to_owned() {
                VdafInstance::Prio3Count => Vdaf::Count,
                VdafInstance::Prio3Sum {
                    max_measurement: _max_measurement,
                } => Vdaf::Sum {
                    // TODO(#3436): once divviup_client is updated for DAP-13, plumb max_measurement through
                    bits: 64,
                },
                VdafInstance::Prio3SumVec {
                    bits,
                    length,
                    chunk_length,
                    dp_strategy,
                } => {
                    let dp_strategy =
                        serde_json::from_value(serde_json::to_value(dp_strategy).unwrap()).unwrap();
                    Vdaf::SumVec(SumVec::new(
                        bits.try_into().unwrap(),
                        length.try_into().unwrap(),
                        Some(chunk_length.try_into().unwrap()),
                        dp_strategy,
                    ))
                }
                VdafInstance::Prio3Histogram {
                    length,
                    chunk_length,
                    dp_strategy,
                } => {
                    let dp_strategy =
                        serde_json::from_value(serde_json::to_value(dp_strategy).unwrap()).unwrap();
                    Vdaf::Histogram(Histogram::Length {
                        length: length.try_into().unwrap(),
                        chunk_length: Some(chunk_length.try_into().unwrap()),
                        dp_strategy,
                    })
                }
                other => panic!("unsupported vdaf {other:?}"),
            },
            min_batch_size: task.min_batch_size(),
            // TODO(#3436): once divviup-api is updated for DAP-13, drop max_batch_size entirely.
            max_batch_size: None,
            batch_time_window_size_seconds: match task.batch_mode() {
                BatchMode::TimeInterval => None,
                BatchMode::LeaderSelected {
                    batch_time_window_size,
                    ..
                } => batch_time_window_size.map(|window| window.as_seconds()),
            },
            time_precision_seconds: task.time_precision().as_seconds(),
            collector_credential_id,
        };

        // Provision the task into both aggregators via divviup-api
        let provisioned_task = divviup_api
            .create_task(divviup_account_id, provision_task_request)
            .await
            .unwrap();

        // Update the task parameters with the ID and collector auth token from divviup-api.
        task_parameters.task_id = TaskId::from_str(&provisioned_task.id).unwrap();
        task_parameters.collector_auth_token = collector_auth_token;
        task_parameters.collector_hpke_keypair = collector_hpke_keypair;

        Self {
            task_parameters,
            leader,
            helper,
        }
    }
}

struct InClusterJanus {
    aggregator_port_forward: Option<PortForward>,
}

impl InClusterJanus {
    /// Set up a port forward to an existing Janus instance in a Kubernetes cluster, and provision a
    /// DAP task in it via divviup-api.
    async fn new(cluster: &Cluster, aggregator_namespace: &str) -> Self {
        let aggregator_port_forward = cluster
            .forward_port(aggregator_namespace, "aggregator", 80)
            .await;
        Self {
            aggregator_port_forward: Some(aggregator_port_forward),
        }
    }

    fn port(&self) -> u16 {
        self.aggregator_port_forward
            .as_ref()
            .map(PortForward::local_port)
            .unwrap_or(0)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_count() {
    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let janus_pair =
        InClusterJanusPair::new(VdafInstance::Prio3Count, BatchMode::TimeInterval).await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_count",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "ohttp")]
async fn in_cluster_count_ohttp() {
    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let mut janus_pair =
        InClusterJanusPair::new(VdafInstance::Prio3Count, BatchMode::TimeInterval).await;

    // Set up the client to use OHTTP. The keys and relay are assumed to be deployed adjacent to the
    // leader.
    janus_pair.task_parameters.endpoint_fragments.ohttp_config = Some(OhttpConfig {
        key_configs: janus_pair
            .task_parameters
            .endpoint_fragments
            .leader_endpoint_for_host(0)
            .join("ohttp-keys")
            .unwrap(),
        relay: janus_pair
            .task_parameters
            .endpoint_fragments
            .leader_endpoint_for_host(0)
            .join("gateway")
            .unwrap(),
    });

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_count_ohttp",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_sum() {
    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Sum {
            max_measurement: 65535,
        },
        BatchMode::TimeInterval,
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_sum",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_histogram() {
    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Histogram {
            length: 4,
            chunk_length: 2,
            dp_strategy: vdaf_dp_strategies::Prio3Histogram::NoDifferentialPrivacy,
        },
        BatchMode::TimeInterval,
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_histogram",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_leader_selected() {
    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Count,
        BatchMode::LeaderSelected {
            batch_time_window_size: None,
        },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_leader_selected",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_time_bucketed_leader_selected() {
    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Count,
        BatchMode::LeaderSelected {
            batch_time_window_size: Some(JanusDuration::from_hours(8).unwrap()),
        },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_time_bucketed_leader_selected",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}

#[cfg(feature = "in-cluster-rate-limits")]
mod rate_limits {
    use super::InClusterJanusPair;
    use crate::initialize_rustls;
    use assert_matches::assert_matches;
    use http::Method;
    use janus_aggregator_core::task::BatchMode;
    use janus_core::{test_util::install_test_trace_subscriber, vdaf::VdafInstance};
    use janus_messages::{AggregationJobId, CollectionJobId, TaskId};
    use rand::random;
    use reqwest::StatusCode;
    use serde::Deserialize;
    use std::{
        env,
        fs::File,
        sync::{Arc, LazyLock},
        time::Duration,
    };
    use tokio::sync::Semaphore;
    use url::Url;

    /// Configuration for the rate limit test. We need to know the QPS and the window over which
    /// it is enforced so that we can send the appropriate number of requests. We load this config
    /// from a file so that an integration test harness that knows what rate limits have been
    /// deployed to the cluster we test against can set the values.
    #[derive(Deserialize)]
    struct TestConfig {
        rate_limit_excess: f64,
        window: u64,
        upload_qps: u64,
        aggregation_job_qps: u64,
        collection_job_qps: u64,
    }

    impl TestConfig {
        fn load() -> &'static Self {
            static CONFIG: LazyLock<TestConfig> = LazyLock::new(|| {
                serde_json::from_reader(
                    File::open(env::var("JANUS_E2E_RATE_LIMIT_TEST_CONFIG").unwrap()).unwrap(),
                )
                .unwrap()
            });
            &CONFIG
        }
    }

    struct RequestUrlParameters<'a> {
        leader_url: &'a Url,
        helper_url: &'a Url,
        first_task_id: &'a TaskId,
        second_task_id: &'a TaskId,
    }

    async fn run_rate_limit_test(
        request_url_maker: &dyn Fn(&RequestUrlParameters) -> (Url, Url),
        rate_limit_picker: &dyn Fn(&TestConfig) -> u64,
        method: Method,
    ) {
        install_test_trace_subscriber();
        initialize_rustls();
        let test_config = TestConfig::load();

        let janus_pair =
            InClusterJanusPair::new(VdafInstance::Prio3Count, BatchMode::TimeInterval).await;

        let (leader_url, helper_url) = janus_pair
            .task_parameters
            .endpoint_fragments
            .endpoints_for_host_client(janus_pair.leader.port(), janus_pair.helper.port());
        let other_task_id = random();

        // Send requests to two different tasks, to prove that rate limits are per-task ID. It
        // doesn't matter that one of the tasks doesn't exist for purposes of rate limits.
        let (first_request_url, second_request_url) = request_url_maker(&RequestUrlParameters {
            leader_url: &leader_url,
            helper_url: &helper_url,
            first_task_id: &janus_pair.task_parameters.task_id,
            second_task_id: &other_task_id,
        });
        let rate_limit = rate_limit_picker(test_config);
        let rate_limit_excess = test_config.rate_limit_excess;
        let total_requests_count =
            ((1.0 + rate_limit_excess) * (rate_limit * test_config.window) as f64) as u64;

        let mut acceptable_status_count = 0;
        let mut too_many_requests_count = 0;
        let mut last_retry_after = None;
        let client = reqwest::Client::builder().build().unwrap();

        // Send ten requests at a time, because otherwise the kubectl port-forward gets overwhelmed
        let semaphore = Arc::new(Semaphore::new(10));
        let mut handles = Vec::new();
        for _ in 0..total_requests_count {
            for url in [first_request_url.clone(), second_request_url.clone()] {
                let (client, method, semaphore) =
                    (client.clone(), method.clone(), semaphore.clone());
                handles.push(tokio::spawn(async move {
                    let _permit = semaphore.acquire_owned().await.unwrap();
                    // We avoid using janus_client here because we don't want it to automatically
                    // retry on HTTP 429 for us.
                    let response = client.request(method, url).send().await.unwrap();

                    (
                        response.headers().get("retry-after").cloned(),
                        response.status(),
                    )
                }));
            }
        }

        for handle in handles {
            let (retry_after, status) = handle.await.unwrap();
            if status == StatusCode::TOO_MANY_REQUESTS {
                assert_matches!(retry_after, Some(retry_after) => {
                    let retry_after = retry_after.to_str().unwrap().parse::<u64>().unwrap();
                    assert!(retry_after <= test_config.window);
                    last_retry_after = Some(retry_after);
                });
                too_many_requests_count += 1
            // Every request this test send should get rejected due to a missing body if it gets
            // past the rate limiter.
            } else if status.is_client_error() {
                assert!(retry_after.is_none());
                acceptable_status_count += 1
            } else {
                panic!("unexpected status {status:?}");
            }
        }

        let ratio = too_many_requests_count as f64
            / (acceptable_status_count + too_many_requests_count) as f64;
        // We expect some exact percentage of requests to be rejected with HTTP 429, but allow a
        // margin for error to account for cases where the test takes more than a second to run, or
        // errors introduced by Caddy distributed rate limiting.
        let expected_429_rate = rate_limit_excess / (1.0 + rate_limit_excess);
        assert!(
            ratio > expected_429_rate - 0.05 && ratio <= expected_429_rate + 0.05,
            "ratio: {ratio} expected 429 rate: {expected_429_rate} \
            count of HTTP 429: {too_many_requests_count} \
            count of HTTP 4xx: {acceptable_status_count}",
        );

        let last_retry_after = assert_matches!(last_retry_after, Some(l) => l);

        // Wait for prescribed time to elapse and then try again. Requests should go through.
        std::thread::sleep(Duration::from_secs(last_retry_after));

        for url in [first_request_url.clone(), second_request_url.clone()] {
            let method = method.clone();
            let response = client.request(method, url).send().await.unwrap();
            assert!(response.status() != StatusCode::TOO_MANY_REQUESTS);
            assert!(response.status().is_client_error());
            assert!(response.headers().get("retry-after").is_none());
        }
    }

    #[tokio::test]
    async fn upload() {
        run_rate_limit_test(
            &|RequestUrlParameters {
                  leader_url,
                  first_task_id,
                  second_task_id,
                  ..
              }| {
                (
                    leader_url
                        .join(&format!("tasks/{first_task_id}/reports"))
                        .unwrap(),
                    leader_url
                        .join(&format!("tasks/{second_task_id}/reports"))
                        .unwrap(),
                )
            },
            &|test_config| test_config.upload_qps,
            Method::PUT,
        )
        .await
    }

    #[rstest::rstest]
    #[case::put(Method::PUT)]
    #[case::post(Method::POST)]
    #[case::delete(Method::DELETE)]
    #[tokio::test]
    async fn collection_job(#[case] method: Method) {
        run_rate_limit_test(
            &|RequestUrlParameters {
                  leader_url,
                  first_task_id,
                  second_task_id,
                  ..
              }| {
                let job_id: CollectionJobId = random();

                (
                    leader_url
                        .join(&format!("tasks/{first_task_id}/collection_jobs/{job_id}"))
                        .unwrap(),
                    leader_url
                        .join(&format!("tasks/{second_task_id}/collection_jobs/{job_id}"))
                        .unwrap(),
                )
            },
            &|test_config| test_config.collection_job_qps,
            method,
        )
        .await
    }

    #[rstest::rstest]
    #[case::put(Method::PUT)]
    #[case::post(Method::POST)]
    #[tokio::test]
    async fn aggregation_job_put(#[case] method: Method) {
        run_rate_limit_test(
            &|RequestUrlParameters {
                  helper_url,
                  first_task_id,
                  second_task_id,
                  ..
              }| {
                let job_id: AggregationJobId = random();
                (
                    helper_url
                        .join(&format!("tasks/{first_task_id}/aggregation_jobs/{job_id}"))
                        .unwrap(),
                    helper_url
                        .join(&format!("tasks/{second_task_id}/aggregation_jobs/{job_id}"))
                        .unwrap(),
                )
            },
            &|test_config| test_config.aggregation_job_qps,
            method,
        )
        .await
    }

    #[tokio::test]
    async fn aggregate_share_post() {
        run_rate_limit_test(
            &|RequestUrlParameters {
                  helper_url,
                  first_task_id,
                  second_task_id,
                  ..
              }| {
                (
                    helper_url
                        .join(&format!("tasks/{first_task_id}/aggregate_shares"))
                        .unwrap(),
                    helper_url
                        .join(&format!("tasks/{second_task_id}/aggregate_shares"))
                        .unwrap(),
                )
            },
            &|test_config| test_config.collection_job_qps,
            Method::POST,
        )
        .await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_histogram_dp_noise() {
    static TEST_NAME: &str = "in_cluster_histogram_dp_noise";
    const HISTOGRAM_LENGTH: usize = 100;
    const CHUNK_LENGTH: usize = 10;

    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let epsilon = Rational::from_unsigned(1u128, 10u128).unwrap();
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Histogram {
            length: HISTOGRAM_LENGTH,
            chunk_length: CHUNK_LENGTH,
            dp_strategy: vdaf_dp_strategies::Prio3Histogram::PureDpDiscreteLaplace(
                PureDpDiscreteLaplace::from_budget(PureDpBudget::new(epsilon).unwrap()),
            ),
        },
        BatchMode::LeaderSelected {
            batch_time_window_size: Some(JanusDuration::from_hours(8).unwrap()),
        },
    )
    .await;
    let vdaf = Prio3::new_histogram_multithreaded(2, HISTOGRAM_LENGTH, CHUNK_LENGTH).unwrap();

    let total_measurements: usize = janus_pair
        .task_parameters
        .min_batch_size
        .try_into()
        .unwrap();
    let measurements = iter::repeat(0).take(total_measurements).collect::<Vec<_>>();
    let client_implementation = ClientBackend::InProcess
        .build(
            TEST_NAME,
            &janus_pair.task_parameters,
            (janus_pair.leader.port(), janus_pair.helper.port()),
            vdaf.clone(),
        )
        .await
        .unwrap();
    let before_timestamp = submit_measurements_generic(&measurements, &client_implementation).await;
    let (report_count, aggregate_result) = collect_aggregate_result_generic(
        &janus_pair.task_parameters,
        janus_pair.leader.port(),
        vdaf,
        before_timestamp,
        &(),
    )
    .await;
    assert_eq!(report_count, janus_pair.task_parameters.min_batch_size);

    let mut un_noised_result = [0u128; HISTOGRAM_LENGTH];
    un_noised_result[0] = report_count.into();
    // Smoke test: Just confirm that some noise was added. Since epsilon is small, the noise will be
    // large (drawn from Laplace_Z(20) + Laplace_Z(20)), and it is highly unlikely that all 100
    // noise values will be zero simultaneously.
    assert_ne!(aggregate_result, un_noised_result);

    assert!(aggregate_result
        .iter()
        .all(|x| *x < Field128::modulus() / 4 || *x > Field128::modulus() / 4 * 3));
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_sumvec_dp_noise() {
    static TEST_NAME: &str = "in_cluster_sumvec_dp_noise";
    const VECTOR_LENGTH: usize = 50;
    const BITS: usize = 2;
    const CHUNK_LENGTH: usize = 10;

    install_test_trace_subscriber();
    initialize_rustls();

    // Start port forwards and set up task.
    let epsilon = Rational::from_unsigned(1u128, 10u128).unwrap();
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3SumVec {
            bits: BITS,
            length: VECTOR_LENGTH,
            chunk_length: CHUNK_LENGTH,
            dp_strategy: vdaf_dp_strategies::Prio3SumVec::PureDpDiscreteLaplace(
                PureDpDiscreteLaplace::from_budget(PureDpBudget::new(epsilon).unwrap()),
            ),
        },
        BatchMode::LeaderSelected {
            batch_time_window_size: Some(JanusDuration::from_hours(8).unwrap()),
        },
    )
    .await;
    let vdaf = Prio3::new_sum_vec_multithreaded(2, BITS, VECTOR_LENGTH, CHUNK_LENGTH).unwrap();

    let total_measurements: usize = janus_pair
        .task_parameters
        .min_batch_size
        .try_into()
        .unwrap();
    let measurements = iter::repeat(vec![0; VECTOR_LENGTH])
        .take(total_measurements)
        .collect::<Vec<_>>();
    let client_implementation = ClientBackend::InProcess
        .build(
            TEST_NAME,
            &janus_pair.task_parameters,
            (janus_pair.leader.port(), janus_pair.helper.port()),
            vdaf.clone(),
        )
        .await
        .unwrap();
    let before_timestamp = submit_measurements_generic(&measurements, &client_implementation).await;
    let (report_count, aggregate_result) = collect_aggregate_result_generic(
        &janus_pair.task_parameters,
        janus_pair.leader.port(),
        vdaf,
        before_timestamp,
        &(),
    )
    .await;
    assert_eq!(report_count, janus_pair.task_parameters.min_batch_size);

    let un_noised_result = [0u128; VECTOR_LENGTH];
    // Smoke test: Just confirm that some noise was added. Since epsilon is small, the noise will be
    // large (drawn from Laplace_Z(150) + Laplace_Z(150)), and it is highly unlikely that all 50
    // noise values will be zero simultaneously.
    assert_ne!(aggregate_result, un_noised_result);

    assert!(aggregate_result
        .iter()
        .all(|x| *x < Field128::modulus() / 4 || *x > Field128::modulus() / 4 * 3));
}
