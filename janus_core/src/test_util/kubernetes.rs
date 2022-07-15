//! Testing framework for functionality that interacts with Kubernetes.

use kube::config::{KubeConfigOptions, Kubeconfig};
use rand::{thread_rng, Rng};
use std::process::{Command, Stdio};
use tempfile::{NamedTempFile, TempPath};

/// EphemeralCluster represents a running ephemeral Kubernetes cluster for testing. Dropping an
/// EphemeralCluster will cause the associated Kubernetes cluster to be stopped & cleaned up.
pub struct EphemeralCluster {
    name: String,
    kubeconfig_path: TempPath,
}

impl EphemeralCluster {
    /// Creates & starts a new ephemeral Kubernetes cluster.
    pub fn create() -> Self {
        // Choose a temporary file location for our kube config.
        let kubeconfig_path = NamedTempFile::new().unwrap().into_temp_path();

        // Choose a cluster name.
        let mut randomness = [0u8; 4];
        thread_rng().fill(&mut randomness);
        let cluster_name = format!("janus-ephemeral-{}", hex::encode(&randomness));

        // Use kind to start the cluster.
        assert!(Command::new("kind")
            .args([
                "create",
                "cluster",
                "--kubeconfig",
                &kubeconfig_path.to_string_lossy(),
                "--name",
                &cluster_name,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success());

        Self {
            name: cluster_name,
            kubeconfig_path,
        }
    }

    /// Returns a new [`kube::Client`] configured to interact with this Kubernetes cluster.
    pub async fn client(&self) -> kube::Client {
        kube::Client::try_from(
            kube::Config::from_custom_kubeconfig(
                Kubeconfig::read_from(&self.kubeconfig_path).unwrap(),
                &KubeConfigOptions {
                    context: Some(format!("kind-{}", self.name)),
                    ..KubeConfigOptions::default()
                },
            )
            .await
            .unwrap(),
        )
        .unwrap()
    }
}

impl Drop for EphemeralCluster {
    fn drop(&mut self) {
        // Delete the cluster that was created when we created the EphemeralCluster.
        assert!(Command::new("kind")
            .args([
                "delete",
                "cluster",
                "--kubeconfig",
                &self.kubeconfig_path.to_string_lossy(),
                "--name",
                &self.name,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success())
    }
}

#[cfg(test)]
mod tests {
    use super::EphemeralCluster;

    #[test]
    fn create_clusters() {
        // Create a couple of clusters, then drop them, to test that creating multiple clusters
        // does not lead to collisions in some namespace.
        let _first_cluster = EphemeralCluster::create();
        let _second_cluster = EphemeralCluster::create();
    }
}
