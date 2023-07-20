#![cfg(feature = "in-cluster")]

use base64::engine::{
    general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine,
};
use common::{submit_measurements_and_verify_aggregate, test_task_builders};
use janus_aggregator_core::task::QueryType;
use janus_collector::AuthenticationToken;
use janus_core::{
    task::{DapAuthToken, VdafInstance},
    test_util::{
        install_test_trace_subscriber,
        kubernetes::{Cluster, PortForward},
    },
};
use janus_integration_tests::{
    client::ClientBackend,
    divviup_api_client::{DivviupApiClient, NewAggregatorRequest, NewTaskRequest},
    TaskParameters,
};
use janus_messages::TaskId;
use prio::codec::Encode;
use std::{env, str::FromStr};
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

        let divviup_api = DivviupApiClient::new(
            cluster
                .forward_port(&divviup_api_namespace, "divviup-api", 80)
                .await,
        );

        // Create an account first. (We should be implicitly logged in as a testing user already,
        // assuming divviup-api was built with the integration-testing feature)
        let account = divviup_api.create_account().await;

        // Pair the aggregators. The same Janus instances will get paired multiple times across
        // multiple tests, but it's to a different divviup-api account each time, so that's
        // harmless. The leader aggregator is paired as a *global* aggregator using the admin
        // endpoint. We do this for two reasons:
        //
        // - setting up tasks with one global aggregator and one per-account aggregator is most
        //   representative of the subscriber use cases Divvi Up supports,
        // - pairing a global aggregator implictly marks it as "first-party" in divviup-api, which
        //   is necessary for the task we later provision to pass a validity check.
        let paired_leader_aggregator = divviup_api
            .pair_global_aggregator(&NewAggregatorRequest {
                role: "Leader".to_string(),
                name: "leader".to_string(),
                api_url: Self::in_cluster_aggregator_api_url(&leader_namespace).to_string(),
                dap_url: Self::in_cluster_aggregator_dap_url(&leader_namespace).to_string(),
                bearer_token: leader_aggregator_api_auth_token,
            })
            .await;

        let paired_helper_aggregator = divviup_api
            .pair_aggregator(
                &account,
                &NewAggregatorRequest {
                    role: "Helper".to_string(),
                    name: "helper".to_string(),
                    api_url: Self::in_cluster_aggregator_api_url(&helper_namespace).to_string(),
                    dap_url: Self::in_cluster_aggregator_dap_url(&helper_namespace).to_string(),
                    bearer_token: helper_aggregator_api_auth_token,
                },
            )
            .await;

        let provision_task_request = NewTaskRequest {
            name: "Integration test task".to_string(),
            leader_aggregator_id: paired_leader_aggregator.id,
            helper_aggregator_id: paired_helper_aggregator.id,
            vdaf: task.vdaf().try_into().unwrap(),
            min_batch_size: task.min_batch_size(),
            max_batch_size: match task.query_type() {
                QueryType::FixedSize { max_batch_size } => Some(*max_batch_size),
            },
            expiration: "3000-01-01T00:00:00Z".to_owned(),
            time_precision_seconds: task.time_precision().as_seconds(),
            hpke_config: STANDARD.encode(task.collector_hpke_config().get_encoded()),
        };

        // Provision the task into both aggregators via divviup-api
        let provisioned_task = divviup_api
            .create_task(&account, &provision_task_request)
            .await;

        let collector_auth_tokens = divviup_api
            .list_collector_auth_tokens(&provisioned_task)
            .await;

        // Update the task parameters with the ID and collector auth token from divviup-api.
        task_parameters.task_id = TaskId::from_str(provisioned_task.id.as_ref()).unwrap();
        task_parameters.collector_auth_token = AuthenticationToken::DapAuth(
            DapAuthToken::try_from(
                URL_SAFE_NO_PAD
                    .decode(collector_auth_tokens[0].clone())
                    .unwrap(),
            )
            .unwrap(),
        );

        Self {
            task_parameters,
            leader: InClusterJanus::new(&cluster, &leader_namespace).await,
            helper: InClusterJanus::new(&cluster, &helper_namespace).await,
        }
    }

    fn in_cluster_aggregator_dap_url(namespace: &str) -> Url {
        Url::parse(&format!(
            "http://aggregator.{namespace}.svc.cluster.local:80"
        ))
        .unwrap()
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
    let janus_pair = InClusterJanusPair::new(
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

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_sum() {
    install_test_trace_subscriber();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
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

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_histogram() {
    install_test_trace_subscriber();

    // Start port forwards and set up task.
    let buckets = Vec::from([3, 6, 8]);
    let janus_pair = InClusterJanusPair::new(
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

#[tokio::test(flavor = "multi_thread")]
async fn in_cluster_fixed_size() {
    install_test_trace_subscriber();

    // Start port forwards and set up task.
    let janus_pair = InClusterJanusPair::new(
        VdafInstance::Prio3Count,
        QueryType::FixedSize {
            max_batch_size: 110,
        },
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
