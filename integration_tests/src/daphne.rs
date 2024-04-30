//! Functionality for tests interacting with Daphne (<https://github.com/cloudflare/daphne>).

use crate::interop_api;
use janus_aggregator_core::task::test_util::{Task, TaskBuilder};
use janus_interop_binaries::{
    get_rust_log_level, test_util::await_ready_ok, ContainerLogsDropGuard, ContainerLogsSource,
};
use janus_messages::{Role, Time};
use serde_json::json;
use testcontainers::{clients::Cli, GenericImage, RunnableImage};
use url::Url;

const DAPHNE_HELPER_IMAGE_NAME_AND_TAG: &str = "cloudflare/daphne-worker-helper:sha-4c612db";

/// Represents a running Daphne test instance.
pub struct Daphne<'a> {
    daphne_container: ContainerLogsDropGuard<'a, GenericImage>,
}

impl<'a> Daphne<'a> {
    const INTERNAL_SERVING_PORT: u16 = 8788;

    /// Create and start a new hermetic Daphne test instance in the given Docker network, configured
    /// to service the given task. The aggregator port is also exposed to the host.
    pub async fn new(
        test_name: &str,
        container_client: &'a Cli,
        network: &str,
        task: &Task,
        role: Role,
    ) -> Daphne<'a> {
        let (endpoint, image_name_and_tag) = match role {
            Role::Leader => panic!("A leader container image for Daphne is not yet available"),
            Role::Helper => (
                task.helper_aggregator_endpoint(),
                DAPHNE_HELPER_IMAGE_NAME_AND_TAG,
            ),
            Role::Collector | Role::Client => unreachable!(),
        };
        let (image_name, image_tag) = image_name_and_tag
            .rsplit_once(':')
            .unwrap_or((image_name_and_tag, "latest"));

        // Start the Daphne test container running.
        let runnable_image = RunnableImage::from(GenericImage::new(image_name, image_tag))
            .with_network(network)
            // Daphne uses the DAP_TRACING environment variable for its tracing subscriber.
            .with_env_var(("DAP_TRACING", get_rust_log_level().1))
            .with_container_name(endpoint.host_str().unwrap());
        let daphne_container = ContainerLogsDropGuard::new(
            test_name,
            container_client.run(runnable_image),
            ContainerLogsSource::Path("/logs".into()),
        );
        let port = daphne_container.get_host_port_ipv4(Self::INTERNAL_SERVING_PORT);

        // Wait for Daphne container to begin listening on the port.
        await_ready_ok(port).await;

        // Reset Daphne's state.
        {
            let http_client = reqwest::Client::default();
            let resp = http_client
                .post(Url::parse(&format!("http://127.0.0.1:{port}/internal/delete_all")).unwrap())
                .send()
                .await
                .unwrap();
            assert!(
                resp.status().is_success(),
                "unexpected status: {}",
                resp.status()
            );
        }

        // Add an HPKE receiver config to Daphne.
        {
            let hpke_receiver_config = json!({
                "config": {
                    "id": 23,
                    "kem_id": "x25519_hkdf_sha256",
                    "kdf_id": "hkdf_sha256",
                    "aead_id": "aes128_gcm",
                    "public_key": "c63eb66d91f472f586e82e2be84ac1e32269fede0ca80bf9dc3ec2e0c1c6582f"
                },
                "private_key":"8d89ce933d017a73eac6408078c7ed2a6ccc56b5c87ebae0d46f02c8718e26ce",
            });

            let http_client = reqwest::Client::default();
            let resp = http_client
                .post(
                    Url::parse(&format!(
                        "http://127.0.0.1:{port}/internal/test/add_hpke_config"
                    ))
                    .unwrap(),
                )
                .json(&hpke_receiver_config)
                .send()
                .await
                .unwrap();
            assert!(
                resp.status().is_success(),
                "unexpected status: {}",
                resp.status()
            );
        }

        // Daphne does not support unset task expiration values. Work around this by specifying an
        // arbitrary, far-future task expiration time, instead.
        let task = if task.task_expiration().is_none() {
            TaskBuilder::from(task.clone())
                .with_task_expiration(Some(Time::from_seconds_since_epoch(2000000000)))
                .build()
        } else {
            task.clone()
        };

        // Write the given task to the Daphne instance we started.
        interop_api::aggregator_add_task(port, task, role).await;

        Self { daphne_container }
    }

    /// Returns the port of the aggregator on the host.
    pub fn port(&self) -> u16 {
        self.daphne_container
            .get_host_port_ipv4(Self::INTERNAL_SERVING_PORT)
    }
}
