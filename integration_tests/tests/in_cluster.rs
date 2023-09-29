#![cfg(feature = "in-cluster")]
use common::{submit_measurements_and_verify_aggregate, test_task_builders};
use divviup_client::{
    Client, CollectorAuthenticationToken, DivviupClient, Histogram, HpkeConfigContents,
    NewAggregator, NewSharedAggregator, NewTask, Vdaf,
};
use janus_aggregator_core::task::QueryType;
use janus_core::{
    auth_tokens::AuthenticationToken,
    test_util::{
        install_test_trace_subscriber,
        kubernetes::{Cluster, PortForward},
    },
    vdaf::VdafInstance,
};
use janus_integration_tests::{client::ClientBackend, TaskParameters};
use janus_messages::TaskId;
use std::{env, str::FromStr};
use trillium_tokio::ClientConfig;
use url::Url;

mod common;

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
    /// Set up a new DAP task, using the given VDAF and query type, in a pair of existing Janus
    /// instances in a Kubernetes cluster. `divviup-api` is used to configure the task in each Janus
    /// instance. The following environment variables must be set.
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
    async fn new(vdaf: VdafInstance, query_type: QueryType) -> Self {
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

        let (mut task_parameters, task_builder, _) = test_task_builders(vdaf, query_type);
        let task = task_builder.with_min_batch_size(100).build();
        task_parameters.min_batch_size = 100;

        // From outside the cluster, the aggregators are reached at a dynamically allocated port on
        // localhost. When the aggregators talk to each other, they do so in the cluster's network,
        // so they need the in-cluster DNS name of the other aggregator, and they can use well-known
        // service port numbers.

        let port_forward = cluster
            .forward_port(&divviup_api_namespace, "divviup-api", 80)
            .await;
        let port = port_forward.local_port();

        let mut divviup_api = DivviupClient::new(
            "DUATignored".into(),
            Client::new(ClientConfig::new()).with_default_pool(),
        );
        divviup_api.set_url(format!("http://127.0.0.1:{port}").parse().unwrap());

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

        let hpke_config = task.collector_hpke_config().unwrap();
        let collector_hpke_config = divviup_api
            .create_hpke_config(
                account.id,
                &HpkeConfigContents::new(
                    u8::from(*hpke_config.id()).into(),
                    u16::from(*hpke_config.kem_id()).try_into().unwrap(),
                    u16::from(*hpke_config.kdf_id()).try_into().unwrap(),
                    u16::from(*hpke_config.aead_id()).try_into().unwrap(),
                    hpke_config.public_key().as_ref().to_vec().into(),
                ),
                Some("Integration test key"),
            )
            .await
            .unwrap();

        let provision_task_request = NewTask {
            name: "Integration test task".to_string(),
            leader_aggregator_id: paired_leader_aggregator.id,
            helper_aggregator_id: paired_helper_aggregator.id,
            vdaf: match task.vdaf().to_owned() {
                VdafInstance::Prio3Count => Vdaf::Count,
                VdafInstance::Prio3Sum { bits } => Vdaf::Sum {
                    bits: bits.try_into().unwrap(),
                },
                VdafInstance::Prio3SumVec {
                    bits,
                    length,
                    chunk_length,
                } => Vdaf::SumVec {
                    bits: bits.try_into().unwrap(),
                    length: length.try_into().unwrap(),
                    chunk_length: Some(chunk_length.try_into().unwrap()),
                },
                VdafInstance::Prio3Histogram {
                    length,
                    chunk_length,
                } => Vdaf::Histogram(Histogram::Length {
                    length: length.try_into().unwrap(),
                    chunk_length: Some(chunk_length.try_into().unwrap()),
                }),
                VdafInstance::Prio3CountVec {
                    length,
                    chunk_length,
                } => Vdaf::CountVec {
                    length: length.try_into().unwrap(),
                    chunk_length: Some(chunk_length.try_into().unwrap()),
                },
                other => panic!("unsupported vdaf {other:?}"),
            },
            min_batch_size: task.min_batch_size(),
            max_batch_size: match task.query_type() {
                QueryType::TimeInterval => None,
                QueryType::FixedSize { max_batch_size, .. } => Some(*max_batch_size),
            },
            time_precision_seconds: task.time_precision().as_seconds(),
            hpke_config_id: collector_hpke_config.id,
        };

        // Provision the task into both aggregators via divviup-api
        let provisioned_task = divviup_api
            .create_task(account.id, provision_task_request)
            .await
            .unwrap();

        let CollectorAuthenticationToken::Bearer { token } = divviup_api
            .task_collector_auth_tokens(&provisioned_task.id)
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        // Update the task parameters with the ID and collector auth token from divviup-api.
        task_parameters.task_id = TaskId::from_str(&provisioned_task.id).unwrap();
        task_parameters.collector_auth_token =
            AuthenticationToken::new_bearer_token_from_string(token.clone()).unwrap();

        Self {
            task_parameters,
            leader: InClusterJanus::new(&cluster, &leader_namespace).await,
            helper: InClusterJanus::new(&cluster, &helper_namespace).await,
        }
    }

    fn in_cluster_aggregator_api_url(namespace: &str) -> Url {
        Url::parse(&format!(
            "http://aggregator.{namespace}.svc.cluster.local:80/aggregator-api/"
        ))
        .unwrap()
    }
}

struct InClusterJanus {
    aggregator_port_forward: PortForward,
}

impl InClusterJanus {
    /// Set up a port forward to an existing Janus instance in a Kubernetes cluster, and provision a
    /// DAP task in it via divviup-api.
    async fn new(cluster: &Cluster, aggregator_namespace: &str) -> Self {
        let aggregator_port_forward = cluster
            .forward_port(aggregator_namespace, "aggregator", 80)
            .await;
        Self {
            aggregator_port_forward,
        }
    }

    fn port(&self) -> u16 {
        self.aggregator_port_forward.local_port()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_count() {
    install_test_trace_subscriber();

    // Start port forwards and set up task.
    let janus_pair =
        InClusterJanusPair::new(VdafInstance::Prio3Count, QueryType::TimeInterval).await;

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
async fn in_cluster_sum() {
    install_test_trace_subscriber();

    // Start port forwards and set up task.
    let janus_pair =
        InClusterJanusPair::new(VdafInstance::Prio3Sum { bits: 16 }, QueryType::TimeInterval).await;

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

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Histogram {
            length: 4,
            chunk_length: 2,
        },
        QueryType::TimeInterval,
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
async fn in_cluster_fixed_size() {
    install_test_trace_subscriber();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Count,
        QueryType::FixedSize {
            max_batch_size: 110,
            batch_time_window_size: None,
        },
    )
    .await;

    // Run the behavioral test.
    submit_measurements_and_verify_aggregate(
        "in_cluster_fixed_size",
        &janus_pair.task_parameters,
        (janus_pair.leader.port(), janus_pair.helper.port()),
        &ClientBackend::InProcess,
    )
    .await;
}
