use crate::{
    binary_utils::{database_pool, datastore, initialize_rustls, read_config, CommonBinaryOptions},
    config::{BinaryConfig, CommonConfig},
    metrics::{install_metrics_exporter, MetricsExporterHandle},
    trace::{install_trace_subscriber, TraceGuards},
};
use anyhow::{anyhow, Context, Result};
use aws_lc_rs::aead::AES_128_GCM;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use clap::Parser;
use janus_aggregator_api::git_revision;
use janus_aggregator_core::{
    datastore::{self, models::HpkeKeyState, Datastore},
    task::{AggregationMode, AggregatorTask, SerializedAggregatorTask},
    taskprov::{PeerAggregator, VerifyKeyInit},
};
use janus_core::{
    auth_tokens::AuthenticationToken,
    cli::{AeadAlgorithm, KdfAlgorithm, KemAlgorithm},
    hpke::HpkeKeypair,
    time::{Clock, RealClock},
};
use janus_messages::{
    codec::Encode as _, Duration, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId, HpkeKemId, Role,
};
use k8s_openapi::api::core::v1::Secret;
use kube::api::{ObjectMeta, PostParams};
use opentelemetry::global::meter;
use prio::codec::Decode as _;
use rand::{distributions::Standard, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs,
    runtime::{self, Runtime},
    sync::Mutex,
};
use tracing::{debug, info};
use url::Url;

pub fn run(command_line_options: CommandLineOptions) -> Result<()> {
    initialize_rustls();

    // Read and parse config.
    let config_file: ConfigFile = read_config(&command_line_options.common_options)?;

    let runtime = runtime::Builder::new_multi_thread().enable_all().build()?;

    runtime.block_on(async {
        let _guards =
            install_tracing_and_metrics_handlers(config_file.common_config(), &runtime).await?;

        info!(
            common_options = ?&command_line_options.common_options,
            config = ?config_file,
            version = env!("CARGO_PKG_VERSION"),
            git_revision = git_revision(),
            rust_version = env!("RUSTC_SEMVER"),
            "Starting janus_cli"
        );

        if command_line_options.dry_run {
            info!("DRY RUN: no persistent changes will be made")
        }

        command_line_options
            .cmd
            .execute(&command_line_options, &config_file)
            .await
    })
}

#[derive(Debug, Parser)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Generates & writes a new HPKE key.
    GenerateHpkeKey {
        #[clap(flatten)]
        kubernetes_secret_options: KubernetesSecretOptions,

        /// Numeric identifier of the HPKE configuration to generate
        #[arg(long)]
        id: u8,

        /// HPKE Key Encapsulation Mechanism algorithm
        #[arg(long)]
        kem: KemAlgorithm,

        /// HPKE Key Derivation Function algorithm
        #[arg(long)]
        kdf: KdfAlgorithm,

        /// HPKE Authenticated Encryption with Associated Data algorithm
        #[arg(long)]
        aead: AeadAlgorithm,

        /// The location to write the encoded HpkeConfig
        #[arg(long)]
        hpke_config_out_file: Option<PathBuf>,
    },

    /// Sets the state of an HPKE key.
    SetHpkeKeyState {
        #[clap(flatten)]
        kubernetes_secret_options: KubernetesSecretOptions,

        /// Numeric identifier of the HPKE configuration to modify
        #[arg(long)]
        id: u8,

        /// State to change the HPKE key to
        #[arg(long)]
        state: HpkeKeyState,
    },

    /// Adds a taskprov peer aggregator.
    AddTaskprovPeerAggregator {
        #[clap(flatten)]
        kubernetes_secret_options: KubernetesSecretOptions,

        /// The peer's endpoint, as a URL.
        #[arg(long)]
        peer_endpoint: Url,

        /// The peer aggregator's role.
        #[arg(long)]
        peer_role: Role,

        /// The aggregation mode to use. Specified only if this aggregator is the Helper.
        #[arg(long)]
        aggregation_mode: Option<AggregationMode>,

        /// The taskprov verify_key_init value, in unpadded base64url.
        #[arg(long, env = "VERIFY_KEY_INIT", hide_env_values = true)]
        verify_key_init: VerifyKeyInit,

        /// The location of the collector HPKE config file, which contains an encoded DAP HpkeConfig
        /// (i.e. public key & metadata) used to encrypt to the collector.
        #[arg(long)]
        collector_hpke_config_file: PathBuf,

        /// The age after which reports are considered expired & will be deleted permanently from
        /// the datastore, in seconds.
        #[arg(long)]
        report_expiry_age_secs: Option<u64>,

        /// The amount of clock skew that the system will accept, in seconds.
        #[arg(long)]
        tolerable_clock_skew_secs: u64,

        /// The aggregator auth token, which must be in the format `bearer:value` or `dap:value`.
        #[arg(long, env = "AGGREGATOR_AUTH_TOKEN", hide_env_values = true)]
        aggregator_auth_token: AuthenticationToken,

        /// The collector auth token, which must be in the format `bearer:value` or `dap:value`.
        #[arg(long, env = "COLLECTOR_AUTH_TOKEN", hide_env_values = true)]
        collector_auth_token: Option<AuthenticationToken>,
    },

    /// Write a set of tasks identified in a file to the datastore
    ProvisionTasks {
        #[clap(flatten)]
        kubernetes_secret_options: KubernetesSecretOptions,

        /// A YAML file containing a list of tasks to be written
        ///
        /// Existing tasks (matching by task ID) will be overwritten
        tasks_file: PathBuf,

        /// If true, task parameters omitted from the YAML tasks file will be randomly generated
        #[clap(long, default_value = "false")]
        generate_missing_parameters: bool,

        /// Write the YAML representation of the tasks that are written to stdout
        #[clap(long, default_value = "false")]
        echo_tasks: bool,
    },

    /// Create a datastore key and write it to a Kubernetes secret
    CreateDatastoreKey {
        #[clap(flatten)]
        kubernetes_secret_options: KubernetesSecretOptions,
    },
}

impl Command {
    async fn execute(
        &self,
        command_line_options: &CommandLineOptions,
        config_file: &ConfigFile,
    ) -> Result<()> {
        // Note: to keep this function reasonably-readable, individual command handlers should
        // generally create the command's dependencies based on options/config, then call another
        // function with the main command logic.
        let kube_client = LazyKubeClient::new();
        match self {
            Command::GenerateHpkeKey {
                kubernetes_secret_options,
                id,
                kem,
                kdf,
                aead,
                hpke_config_out_file,
            } => {
                let datastore = datastore_from_opts(
                    kubernetes_secret_options,
                    command_line_options,
                    config_file,
                    &kube_client,
                )
                .await?;

                generate_hpke_key(
                    &datastore,
                    command_line_options.dry_run,
                    (*id).into(),
                    (*kem).into(),
                    (*kdf).into(),
                    (*aead).into(),
                    hpke_config_out_file.as_deref(),
                )
                .await
            }

            Command::SetHpkeKeyState {
                kubernetes_secret_options,
                id,
                state,
            } => {
                let datastore = datastore_from_opts(
                    kubernetes_secret_options,
                    command_line_options,
                    config_file,
                    &kube_client,
                )
                .await?;

                set_hpke_key_state(
                    &datastore,
                    command_line_options.dry_run,
                    (*id).into(),
                    *state,
                )
                .await
            }

            Command::AddTaskprovPeerAggregator {
                kubernetes_secret_options,
                peer_endpoint,
                peer_role,
                aggregation_mode,
                verify_key_init,
                collector_hpke_config_file,
                report_expiry_age_secs,
                tolerable_clock_skew_secs,
                aggregator_auth_token,
                collector_auth_token,
            } => {
                let datastore = datastore_from_opts(
                    kubernetes_secret_options,
                    command_line_options,
                    config_file,
                    &kube_client,
                )
                .await?;

                // Parse flags into proper types.
                let report_expiry_age = report_expiry_age_secs.map(Duration::from_seconds);
                let tolerable_clock_skew = Duration::from_seconds(*tolerable_clock_skew_secs);

                add_taskprov_peer_aggregator(
                    &datastore,
                    command_line_options.dry_run,
                    peer_endpoint,
                    *peer_role,
                    *aggregation_mode,
                    *verify_key_init,
                    collector_hpke_config_file,
                    report_expiry_age,
                    tolerable_clock_skew,
                    aggregator_auth_token,
                    collector_auth_token.as_ref(),
                )
                .await
            }

            Command::ProvisionTasks {
                kubernetes_secret_options,
                tasks_file,
                generate_missing_parameters,
                echo_tasks,
            } => {
                let datastore = datastore_from_opts(
                    kubernetes_secret_options,
                    command_line_options,
                    config_file,
                    &kube_client,
                )
                .await?;

                let written_tasks = provision_tasks(
                    &datastore,
                    tasks_file,
                    *generate_missing_parameters,
                    command_line_options.dry_run,
                )
                .await?;

                if *echo_tasks {
                    let tasks_yaml = serde_yaml::to_string(&written_tasks)
                        .context("couldn't serialize tasks to YAML")?;
                    println!("{tasks_yaml}");
                }

                Ok(())
            }

            Command::CreateDatastoreKey {
                kubernetes_secret_options,
            } => {
                let k8s_namespace = kubernetes_secret_options
                    .secrets_k8s_namespace
                    .as_deref()
                    .context("--secrets-k8s-namespace is required")?;
                create_datastore_key(
                    command_line_options.dry_run,
                    &kube_client,
                    k8s_namespace,
                    &kubernetes_secret_options.datastore_keys_secret_name,
                    &kubernetes_secret_options.datastore_keys_secret_data_key,
                )
                .await
            }
        }
    }
}

async fn install_tracing_and_metrics_handlers(
    config: &CommonConfig,
    runtime: &Runtime,
) -> Result<(TraceGuards, MetricsExporterHandle)> {
    // Discard the trace reload handler, since this program is short-lived.
    let (trace_guard, _) = install_trace_subscriber(&config.logging_config)
        .context("couldn't install tracing subscriber")?;

    let metrics_guard = install_metrics_exporter(&config.metrics_config, runtime)
        .await
        .context("failed to install metrics exporter")?;
    Ok((trace_guard, metrics_guard))
}

async fn generate_hpke_key<C: Clock>(
    datastore: &Datastore<C>,
    dry_run: bool,
    id: HpkeConfigId,
    kem: HpkeKemId,
    kdf: HpkeKdfId,
    aead: HpkeAeadId,
    hpke_config_out_file: Option<&Path>,
) -> Result<()> {
    let hpke_keypair = Arc::new(HpkeKeypair::generate(id, kem, kdf, aead)?);

    if !dry_run {
        datastore
            .run_tx("generate_hpke_key", |tx| {
                let hpke_keypair = Arc::clone(&hpke_keypair);

                Box::pin(async move { tx.put_hpke_keypair(&hpke_keypair).await })
            })
            .await?;
    }

    if let Some(hpke_config_out_file) = hpke_config_out_file {
        fs::write(hpke_config_out_file, hpke_keypair.config().get_encoded()?).await?;
    }

    Ok(())
}

async fn set_hpke_key_state<C: Clock>(
    datastore: &Datastore<C>,
    dry_run: bool,
    id: HpkeConfigId,
    state: HpkeKeyState,
) -> Result<()> {
    if !dry_run {
        datastore
            .run_tx("set_hpke_key_state", |tx| {
                Box::pin(async move { tx.set_hpke_keypair_state(&id, &state).await })
            })
            .await?;
    }

    Ok(())
}

async fn add_taskprov_peer_aggregator<C: Clock>(
    datastore: &Datastore<C>,
    dry_run: bool,
    peer_endpoint: &Url,
    role: Role,
    aggregation_mode: Option<AggregationMode>,
    verify_key_init: VerifyKeyInit,
    collector_hpke_config_file: &Path,
    report_expiry_age: Option<Duration>,
    tolerable_clock_skew: Duration,
    aggregator_auth_token: &AuthenticationToken,
    collector_auth_token: Option<&AuthenticationToken>,
) -> Result<()> {
    let collector_hpke_config = {
        let bytes = fs::read(collector_hpke_config_file).await?;
        HpkeConfig::get_decoded(&bytes)?
    };
    let collector_auth_tokens = collector_auth_token
        .cloned()
        .map(|token| Vec::from([token]))
        .unwrap_or_default();
    let peer_aggregator = Arc::new(PeerAggregator::new(
        peer_endpoint.clone(),
        role,
        aggregation_mode,
        verify_key_init,
        collector_hpke_config,
        report_expiry_age,
        tolerable_clock_skew,
        Vec::from([aggregator_auth_token.clone()]),
        collector_auth_tokens,
    ));

    if !dry_run {
        datastore
            .run_tx("add_taskprov_peer_aggregator", |tx| {
                let peer_aggregator = Arc::clone(&peer_aggregator);

                Box::pin(async move { tx.put_taskprov_peer_aggregator(&peer_aggregator).await })
            })
            .await?;
    }

    Ok(())
}

async fn provision_tasks<C: Clock>(
    datastore: &Datastore<C>,
    tasks_file: &Path,
    generate_missing_parameters: bool,
    dry_run: bool,
) -> Result<Vec<AggregatorTask>> {
    // Read tasks file.
    let tasks: Vec<SerializedAggregatorTask> = {
        let task_file_contents = fs::read_to_string(tasks_file)
            .await
            .with_context(|| format!("couldn't read tasks file {tasks_file:?}"))?;
        serde_yaml::from_str(&task_file_contents)
            .with_context(|| format!("couldn't parse tasks file {tasks_file:?}"))?
    };

    let tasks: Vec<AggregatorTask> = tasks
        .into_iter()
        .map(|mut task| {
            if generate_missing_parameters {
                task.generate_missing_fields();
            }

            AggregatorTask::try_from(task)
        })
        .collect::<Result<_, _>>()?;

    if dry_run {
        info!(task_count = %tasks.len(), "DRY RUN: Not writing tasks");
        return Ok(tasks);
    }

    let tasks = Arc::new(tasks);

    // Write all tasks requested.
    info!(task_count = %tasks.len(), "Writing tasks");
    let written_tasks = datastore
        .run_tx("provision-tasks", |tx| {
            let tasks = Arc::clone(&tasks);
            Box::pin(async move {
                let mut written_tasks = Vec::new();
                for task in tasks.iter() {
                    // We attempt to delete the task, but ignore "task not found" errors since
                    // the task not existing is an OK outcome too.
                    match tx.delete_task(task.id()).await {
                        Ok(()) => {
                            info!(task_id = %task.id(), "replacing existing task");
                        }
                        Err(datastore::Error::MutationTargetNotFound) => (),
                        err => err?,
                    }

                    tx.put_aggregator_task(task).await?;

                    written_tasks.push(task.clone());
                }
                Ok(written_tasks)
            })
        })
        .await
        .context("couldn't write tasks")?;

    Ok(written_tasks)
}

async fn fetch_datastore_keys(
    kube_client: &LazyKubeClient,
    namespace: &str,
    secret_name: &str,
    secret_data_key: &str,
) -> Result<Vec<String>> {
    debug!(
        "Fetching value {} from secret {}/{}",
        secret_data_key, namespace, secret_name,
    );

    let secrets_api: kube::Api<Secret> = kube::Api::namespaced(kube_client.get().await?, namespace);

    let secret = secrets_api
        .get(secret_name)
        .await?
        .data
        .context(format!("no data on secret {secret_name}"))?;
    let secret_value = secret.get(secret_data_key).context(format!(
        "no data key {secret_data_key} on secret {secret_name}"
    ))?;

    Ok(String::from_utf8(secret_value.0.clone())?
        .split(',')
        .map(&str::to_string)
        .collect())
}

async fn create_datastore_key(
    dry_run: bool,
    kube_client: &LazyKubeClient,
    k8s_namespace: &str,
    k8s_secret_name: &str,
    k8s_secret_data_key: &str,
) -> Result<()> {
    info!(
        namespace = k8s_namespace,
        secret_name = k8s_secret_name,
        secret_data_key = k8s_secret_data_key,
        "Creating datastore key"
    );
    let secrets_api: kube::Api<Secret> =
        kube::Api::namespaced(kube_client.get().await?, k8s_namespace);

    // Generate a random datastore key & encode it into unpadded base64 as will be expected by
    // consumers of the secret we are about to write.
    let key_bytes: Vec<_> = thread_rng()
        .sample_iter(Standard)
        .take(AES_128_GCM.key_len())
        .collect();
    let secret_content = URL_SAFE_NO_PAD.encode(key_bytes);

    // Write the secret.
    secrets_api
        .create(
            &PostParams {
                dry_run,
                ..Default::default()
            },
            &Secret {
                metadata: ObjectMeta {
                    namespace: Some(k8s_namespace.to_string()),
                    name: Some(k8s_secret_name.to_string()),
                    ..ObjectMeta::default()
                },
                string_data: Some(BTreeMap::from([(
                    k8s_secret_data_key.to_string(),
                    secret_content,
                )])),
                ..Secret::default()
            },
        )
        .await
        .context("couldn't write datastore key secret")?;
    Ok(())
}

async fn datastore_from_opts(
    kubernetes_secret_options: &KubernetesSecretOptions,
    command_line_options: &CommandLineOptions,
    config_file: &ConfigFile,
    kube_client: &LazyKubeClient,
) -> Result<Datastore<RealClock>> {
    let pool = database_pool(
        &config_file.common_config.database,
        command_line_options
            .common_options
            .database_password
            .as_deref(),
    )
    .await?;

    datastore(
        pool,
        RealClock::default(),
        &meter("janus_aggregator"),
        &kubernetes_secret_options
            .datastore_keys(&command_line_options.common_options, kube_client)
            .await?,
        config_file.common_config().database.check_schema_version,
        config_file.common_config().max_transaction_retries,
    )
    .await
}

#[derive(Debug, Parser)]
#[clap(
    name = "janus_cli",
    about = "Janus CLI tool",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
pub struct CommandLineOptions {
    #[clap(subcommand)]
    cmd: Command,

    #[clap(flatten)]
    common_options: CommonBinaryOptions,

    /// Do not make permanent changes
    ///
    /// The tool will print out what it would do but will not make any real, permanent changes.
    #[clap(long, default_value = "false")]
    dry_run: bool,
}

#[derive(Debug, Parser)]
struct KubernetesSecretOptions {
    /// The Kubernetes namespace where secrets are stored
    ///
    /// Required if --datastore-keys is not set or if the command is `create-datastore-key`.
    #[clap(long, env = "SECRETS_K8S_NAMESPACE", num_args = 1)]
    secrets_k8s_namespace: Option<String>,

    /// Kubernetes secret containing the datastore key(s)
    #[clap(
        long,
        env = "DATASTORE_KEYS_SECRET_NAME",
        num_args = 1,
        default_value = "datastore-key"
    )]
    datastore_keys_secret_name: String,

    /// Key into data of datastore key Kubernetes secret
    #[clap(
        long,
        env = "DATASTORE_KEYS_SECRET_KEY",
        num_args = 1,
        default_value = "datastore_key"
    )]
    datastore_keys_secret_data_key: String,
}

impl KubernetesSecretOptions {
    /// Fetch the datastore keys from the options. If --secrets-k8s-namespace is set, keys are fetched
    /// from a secret therein. Otherwise, returns the keys provided to --datastore-keys. If neither was
    /// set, returns an error.
    async fn datastore_keys(
        &self,
        options: &CommonBinaryOptions,
        kube_client: &LazyKubeClient,
    ) -> Result<Vec<String>> {
        if let Some(ref secrets_namespace) = &self.secrets_k8s_namespace {
            fetch_datastore_keys(
                kube_client,
                secrets_namespace,
                &self.datastore_keys_secret_name,
                &self.datastore_keys_secret_data_key,
            )
            .await
            .context("failed to fetch datastore key(s) from Kubernetes secret")
        } else if !options.datastore_keys.is_empty() {
            Ok(options.datastore_keys.clone())
        } else {
            Err(anyhow!(
                "Either --datastore-keys or --secrets-k8s-namespace must be set"
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(flatten)]
    common_config: CommonConfig,
}

impl BinaryConfig for ConfigFile {
    fn common_config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn common_config_mut(&mut self) -> &mut CommonConfig {
        &mut self.common_config
    }
}

/// A wrapper around [`kube::Client`] adding lazy initialization.
struct LazyKubeClient {
    lock: Mutex<Option<kube::Client>>,
}

impl LazyKubeClient {
    fn new() -> Self {
        Self {
            lock: Mutex::default(),
        }
    }

    /// Return a reference to a client, constructing a client from the default inferred
    /// configuration if it has not been done yet. This will use the local kubeconfig file if
    /// present, use in-cluster environment variables if present, or fail.
    async fn get(&self) -> Result<kube::Client> {
        let mut guard = self.lock.lock().await;
        match guard.as_ref() {
            Some(kube_client) => Ok(kube_client.clone()),
            None => {
                let kube_client = kube::Client::try_default()
                    .await
                    .context("couldn't load Kubernetes configuration")?;
                *guard = Some(kube_client.clone());
                Ok(kube_client)
            }
        }
    }
}

impl From<kube::Client> for LazyKubeClient {
    fn from(kube_client: kube::Client) -> Self {
        Self {
            lock: Mutex::new(Some(kube_client)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        binaries::janus_cli::{
            fetch_datastore_keys, CommandLineOptions, ConfigFile, KubernetesSecretOptions,
            LazyKubeClient,
        },
        binary_utils::{initialize_rustls, CommonBinaryOptions},
        config::{
            default_max_transaction_retries,
            test_util::{generate_db_config, generate_metrics_config, generate_trace_config},
            CommonConfig,
        },
    };
    use aws_lc_rs::aead::{UnboundKey, AES_128_GCM};
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use clap::CommandFactory;
    use janus_aggregator_core::{
        datastore::{models::HpkeKeyState, test_util::ephemeral_datastore, Datastore},
        task::{test_util::TaskBuilder, AggregationMode, AggregatorTask, BatchMode},
        taskprov::{PeerAggregator, VerifyKeyInit},
    };
    use janus_core::{
        auth_tokens::AuthenticationToken,
        hpke::HpkeKeypair,
        test_util::{kubernetes, roundtrip_encoding},
        time::RealClock,
        vdaf::{vdaf_dp_strategies, VdafInstance},
    };
    use janus_messages::{
        codec::Encode, Duration, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId, HpkeKemId, Role,
        TaskId,
    };
    use prio::codec::Decode;
    use rand::random;
    use std::{
        collections::HashMap,
        io::Write,
        net::{Ipv4Addr, SocketAddr},
    };
    use tempfile::{tempdir, NamedTempFile};
    use tokio::fs;
    use url::Url;

    #[test]
    fn verify_app() {
        CommandLineOptions::command().debug_assert()
    }

    #[tokio::test]
    async fn options_datastore_keys() {
        initialize_rustls();
        // Prep: create a Kubernetes cluster and put a secret in it
        let k8s_cluster = kubernetes::EphemeralCluster::create();
        let kube_client = k8s_cluster.cluster().client().await.into();
        super::create_datastore_key(
            false,
            &kube_client,
            "default",
            "secret-name",
            "secret-data-key",
        )
        .await
        .unwrap();

        let expected_datastore_keys =
            Vec::from(["datastore-key-1".to_string(), "datastore-key-2".to_string()]);

        // Keys provided at command line, not present in k8s
        let common_options = CommonBinaryOptions {
            datastore_keys: expected_datastore_keys.clone(),
            ..Default::default()
        };

        let kubernetes_secret_options = KubernetesSecretOptions {
            datastore_keys_secret_name: "secret-name".to_string(),
            datastore_keys_secret_data_key: "secret-data-key".to_string(),
            secrets_k8s_namespace: None,
        };
        let empty_kube_client = LazyKubeClient::new();

        assert_eq!(
            kubernetes_secret_options
                .datastore_keys(&common_options, &empty_kube_client)
                .await
                .unwrap(),
            expected_datastore_keys
        );
        // Shouldn't have set up a kube Client for this, since no namespace was given.
        assert!(empty_kube_client.lock.lock().await.is_none());

        // Keys not provided at command line, present in k8s
        let common_options = CommonBinaryOptions::default();
        let kubernetes_secret_options = KubernetesSecretOptions {
            datastore_keys_secret_name: "secret-name".to_string(),
            datastore_keys_secret_data_key: "secret-data-key".to_string(),
            secrets_k8s_namespace: Some("default".to_string()),
        };

        assert_eq!(
            kubernetes_secret_options
                .datastore_keys(&common_options, &kube_client)
                .await
                .unwrap()
                .len(),
            1
        );

        // Neither flag provided
        let common_options = CommonBinaryOptions::default();
        let kubernetes_secret_options = KubernetesSecretOptions {
            datastore_keys_secret_name: "secret-name".to_string(),
            datastore_keys_secret_data_key: "secret-data-key".to_string(),
            secrets_k8s_namespace: None,
        };

        kubernetes_secret_options
            .datastore_keys(&common_options, &kube_client)
            .await
            .unwrap_err();
    }

    fn task_hashmap_from_slice(tasks: Vec<AggregatorTask>) -> HashMap<TaskId, AggregatorTask> {
        tasks.into_iter().map(|task| (*task.id(), task)).collect()
    }

    // Returns the HPKE config written to disk.
    async fn run_generate_hpke_key_testcase(
        ds: &Datastore<RealClock>,
        dry_run: bool,
        id: HpkeConfigId,
        kem: HpkeKemId,
        kdf: HpkeKdfId,
        aead: HpkeAeadId,
    ) -> HpkeConfig {
        let temp_dir = tempdir().unwrap();
        let hpke_config_out_file = temp_dir.path().join("hpke_config");

        super::generate_hpke_key(ds, dry_run, id, kem, kdf, aead, Some(&hpke_config_out_file))
            .await
            .unwrap();

        HpkeConfig::get_decoded(&fs::read(hpke_config_out_file).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn generate_hpke_key() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let id = HpkeConfigId::from(12);
        let kem = HpkeKemId::P256HkdfSha256;
        let kdf = HpkeKdfId::HkdfSha256;
        let aead = HpkeAeadId::Aes128Gcm;

        let disk_hpke_config =
            run_generate_hpke_key_testcase(&ds, /* dry_run */ false, id, kem, kdf, aead).await;

        let hpke_keypair = ds
            .run_unnamed_tx(|tx| {
                Box::pin(async move { Ok(tx.get_hpke_keypair(&id).await.unwrap()) })
            })
            .await
            .unwrap()
            .unwrap();

        // Verify datastore state matches what was written to disk.
        assert_eq!(hpke_keypair.state(), &HpkeKeyState::Pending);
        assert_eq!(hpke_keypair.hpke_keypair().config(), &disk_hpke_config);

        // Verify HPKE configuration matches what was expected.
        assert_eq!(disk_hpke_config.id(), &id);
        assert_eq!(disk_hpke_config.kem_id(), &kem);
        assert_eq!(disk_hpke_config.kdf_id(), &kdf);
        assert_eq!(disk_hpke_config.aead_id(), &aead);
    }

    #[tokio::test]
    async fn generate_hpke_key_dry_run() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let id = HpkeConfigId::from(43);
        let kem = HpkeKemId::X25519HkdfSha256;
        let kdf = HpkeKdfId::HkdfSha512;
        let aead = HpkeAeadId::ChaCha20Poly1305;

        let disk_hpke_config =
            run_generate_hpke_key_testcase(&ds, /* dry_run */ true, id, kem, kdf, aead).await;

        let hpke_keypairs = ds
            .run_unnamed_tx(|tx| Box::pin(async move { Ok(tx.get_hpke_keypairs().await.unwrap()) }))
            .await
            .unwrap();

        // Verify that nothing was written to the datastore.
        assert!(hpke_keypairs.is_empty());

        // Verify HPKE configuration written to disk matches what was expected.
        assert_eq!(disk_hpke_config.id(), &id);
        assert_eq!(disk_hpke_config.kem_id(), &kem);
        assert_eq!(disk_hpke_config.kdf_id(), &kdf);
        assert_eq!(disk_hpke_config.aead_id(), &aead);
    }

    #[tokio::test]
    async fn set_hpke_key_state() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        // Insert an HPKE key for the command to modify.
        let id = HpkeConfigId::from(26);
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.put_hpke_keypair(
                    &HpkeKeypair::generate(
                        id,
                        HpkeKemId::P256HkdfSha256,
                        HpkeKdfId::HkdfSha256,
                        HpkeAeadId::Aes128Gcm,
                    )
                    .unwrap(),
                )
                .await
                .unwrap();

                let hpke_keypair = tx.get_hpke_keypair(&id).await.unwrap().unwrap();
                assert_eq!(hpke_keypair.state(), &HpkeKeyState::Pending);

                Ok(())
            })
        })
        .await
        .unwrap();

        // Run command.
        super::set_hpke_key_state(&ds, /* dry_run */ false, id, HpkeKeyState::Active)
            .await
            .unwrap();

        // Verify the HPKE key was updated appropriately.
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move {
                let hpke_keypair = tx.get_hpke_keypair(&id).await.unwrap().unwrap();
                assert_eq!(hpke_keypair.state(), &HpkeKeyState::Active);

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn set_hpke_key_state_dry_run() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        // Insert a HPKE key for the command to modify.
        let id = HpkeConfigId::from(26);
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move {
                tx.put_hpke_keypair(
                    &HpkeKeypair::generate(
                        id,
                        HpkeKemId::P256HkdfSha256,
                        HpkeKdfId::HkdfSha256,
                        HpkeAeadId::Aes128Gcm,
                    )
                    .unwrap(),
                )
                .await
                .unwrap();

                let hpke_keypair = tx.get_hpke_keypair(&id).await.unwrap().unwrap();
                assert_eq!(hpke_keypair.state(), &HpkeKeyState::Pending);

                Ok(())
            })
        })
        .await
        .unwrap();

        // Run command.
        super::set_hpke_key_state(&ds, /* dry_run */ true, id, HpkeKeyState::Active)
            .await
            .unwrap();

        // Verify the HPKE key was not updated.
        ds.run_unnamed_tx(|tx| {
            Box::pin(async move {
                let hpke_keypair = tx.get_hpke_keypair(&id).await.unwrap().unwrap();
                assert_eq!(hpke_keypair.state(), &HpkeKeyState::Pending);

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    async fn run_add_taskprov_peer_aggregator_testcase(
        ds: &Datastore<RealClock>,
        dry_run: bool,
        peer_endpoint: &Url,
        peer_role: Role,
        aggregation_mode: Option<AggregationMode>,
        verify_key_init: VerifyKeyInit,
        collector_hpke_config: &HpkeConfig,
        report_expiry_age: Option<Duration>,
        tolerable_clock_skew: Duration,
        aggregator_auth_token: &AuthenticationToken,
        collector_auth_token: Option<&AuthenticationToken>,
    ) {
        let mut collector_hpke_config_file = NamedTempFile::new().unwrap();
        collector_hpke_config_file
            .write_all(&collector_hpke_config.get_encoded().unwrap())
            .unwrap();
        let collector_hpke_config_file = collector_hpke_config_file.into_temp_path();

        super::add_taskprov_peer_aggregator(
            ds,
            dry_run,
            peer_endpoint,
            peer_role,
            aggregation_mode,
            verify_key_init,
            &collector_hpke_config_file,
            report_expiry_age,
            tolerable_clock_skew,
            aggregator_auth_token,
            collector_auth_token,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn add_taskprov_peer_aggregator() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let peer_endpoint = "https://example.com".try_into().unwrap();
        let peer_role = Role::Leader;
        let aggregation_mode = Some(AggregationMode::Synchronous);
        let verify_key_init = random();
        let collector_hpke_config = HpkeKeypair::generate(
            HpkeConfigId::from(96),
            HpkeKemId::P256HkdfSha256,
            HpkeKdfId::HkdfSha256,
            HpkeAeadId::Aes128Gcm,
        )
        .unwrap()
        .config()
        .clone();
        let report_expiry_age = Some(Duration::from_seconds(3600));
        let tolerable_clock_skew = Duration::from_seconds(60);
        let aggregator_auth_token = random();
        let collector_auth_token = random();

        run_add_taskprov_peer_aggregator_testcase(
            &ds,
            /* dry_run */ false,
            &peer_endpoint,
            peer_role,
            aggregation_mode,
            verify_key_init,
            &collector_hpke_config,
            report_expiry_age,
            tolerable_clock_skew,
            &aggregator_auth_token,
            Some(&collector_auth_token),
        )
        .await;

        let want_peer_aggregator = PeerAggregator::new(
            peer_endpoint.clone(),
            peer_role,
            aggregation_mode,
            verify_key_init,
            collector_hpke_config,
            report_expiry_age,
            tolerable_clock_skew,
            Vec::from([aggregator_auth_token]),
            Vec::from([collector_auth_token]),
        );

        let got_peer_aggregator = ds
            .run_unnamed_tx(|tx| {
                let peer_endpoint = peer_endpoint.clone();

                Box::pin(async move {
                    Ok(tx
                        .get_taskprov_peer_aggregator(&peer_endpoint, &peer_role)
                        .await
                        .unwrap()
                        .unwrap())
                })
            })
            .await
            .unwrap();

        assert_eq!(want_peer_aggregator, got_peer_aggregator);
    }

    #[tokio::test]
    async fn add_taskprov_peer_aggregator_dry_run() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        run_add_taskprov_peer_aggregator_testcase(
            &ds,
            /* dry_run */ true,
            &"https://example.com".try_into().unwrap(),
            Role::Leader,
            Some(AggregationMode::Synchronous),
            random(),
            &HpkeKeypair::generate(
                HpkeConfigId::from(96),
                HpkeKemId::P256HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            )
            .unwrap()
            .config()
            .clone(),
            Some(Duration::from_seconds(3600)),
            Duration::from_seconds(60),
            &random(),
            Some(&random()),
        )
        .await;

        let got_peer_aggregators = ds
            .run_unnamed_tx(|tx| {
                Box::pin(async move { Ok(tx.get_taskprov_peer_aggregators().await.unwrap()) })
            })
            .await
            .unwrap();

        assert!(got_peer_aggregators.is_empty())
    }

    async fn run_provision_tasks_testcase(
        ds: &Datastore<RealClock>,
        tasks: &[AggregatorTask],
        dry_run: bool,
    ) -> Vec<AggregatorTask> {
        // Write tasks to a temporary file.
        let mut tasks_file = NamedTempFile::new().unwrap();
        tasks_file
            .write_all(serde_yaml::to_string(&tasks).unwrap().as_ref())
            .unwrap();
        let tasks_path = tasks_file.into_temp_path();

        // Run the program logic.
        super::provision_tasks(ds, &tasks_path, false, dry_run)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn provision_tasks() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let tasks = Vec::from([
            TaskBuilder::new(
                BatchMode::TimeInterval,
                AggregationMode::Synchronous,
                VdafInstance::Prio3Count,
            )
            .build()
            .leader_view()
            .unwrap(),
            TaskBuilder::new(
                BatchMode::TimeInterval,
                AggregationMode::Synchronous,
                VdafInstance::Prio3Sum {
                    max_measurement: 4096,
                },
            )
            .build()
            .helper_view()
            .unwrap(),
        ]);

        let written_tasks = run_provision_tasks_testcase(&ds, &tasks, false).await;

        // Verify that the expected tasks were written.
        let want_tasks = task_hashmap_from_slice(tasks);
        let written_tasks = task_hashmap_from_slice(written_tasks);
        let got_tasks = task_hashmap_from_slice(
            ds.run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_tasks().await }))
                .await
                .unwrap(),
        );
        assert_eq!(want_tasks, got_tasks);
        assert_eq!(want_tasks, written_tasks);
    }

    #[tokio::test]
    async fn provision_task_dry_run() {
        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let tasks = Vec::from([TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        )
        .build()
        .leader_view()
        .unwrap()]);

        let written_tasks = run_provision_tasks_testcase(&ds, &tasks, true).await;

        let want_tasks = task_hashmap_from_slice(tasks);
        let written_tasks = task_hashmap_from_slice(written_tasks);
        assert_eq!(want_tasks, written_tasks);
        let got_tasks = task_hashmap_from_slice(
            ds.run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_tasks().await }))
                .await
                .unwrap(),
        );
        assert!(got_tasks.is_empty());
    }

    #[tokio::test]
    async fn replace_task() {
        let tasks = Vec::from([
            TaskBuilder::new(
                BatchMode::TimeInterval,
                AggregationMode::Synchronous,
                VdafInstance::Prio3Count,
            )
            .build()
            .leader_view()
            .unwrap(),
            TaskBuilder::new(
                BatchMode::TimeInterval,
                AggregationMode::Synchronous,
                VdafInstance::Prio3Sum {
                    max_measurement: 4096,
                },
            )
            .build()
            .leader_view()
            .unwrap(),
        ]);

        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let mut tasks_file = NamedTempFile::new().unwrap();
        tasks_file
            .write_all(serde_yaml::to_string(&tasks).unwrap().as_ref())
            .unwrap();

        super::provision_tasks(&ds, &tasks_file.into_temp_path(), false, false)
            .await
            .unwrap();

        // Construct a "new" task with a previously existing ID.
        let replacement_task = TaskBuilder::new(
            BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            AggregationMode::Synchronous,
            VdafInstance::Prio3SumVec {
                bits: 1,
                length: 4,
                chunk_length: 2,
                dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
            },
        )
        .with_id(*tasks[0].id())
        .build()
        .leader_view()
        .unwrap();

        let mut replacement_tasks_file = NamedTempFile::new().unwrap();
        replacement_tasks_file
            .write_all(
                serde_yaml::to_string(&[&replacement_task])
                    .unwrap()
                    .as_ref(),
            )
            .unwrap();

        let written_tasks =
            super::provision_tasks(&ds, &replacement_tasks_file.into_temp_path(), false, false)
                .await
                .unwrap();
        assert_eq!(written_tasks.len(), 1);
        assert_eq!(written_tasks[0].id(), tasks[0].id());

        // Verify that the expected tasks were written.
        let got_tasks = task_hashmap_from_slice(
            ds.run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_tasks().await }))
                .await
                .unwrap(),
        );
        let want_tasks = HashMap::from([
            (*replacement_task.id(), replacement_task),
            (*tasks[1].id(), tasks[1].clone()),
        ]);

        assert_eq!(want_tasks, got_tasks);
    }

    #[tokio::test]
    async fn provision_task_with_generated_values() {
        // YAML contains no task ID, VDAF verify keys, aggregator auth tokens, collector auth tokens
        // or HPKE keys.
        let serialized_task_yaml = r#"
- peer_aggregator_endpoint: https://helper
  batch_mode: TimeInterval
  vdaf: !Prio3Sum
    max_measurement: 4096
  role: Leader
  vdaf_verify_key:
  task_end: 9000000000
  min_batch_size: 10
  time_precision: 300
  tolerable_clock_skew: 600
  collector_hpke_config:
    id: 23
    kem_id: X25519HkdfSha256
    kdf_id: HkdfSha256
    aead_id: Aes128Gcm
    public_key: 8lAqZ7OfNV2Gi_9cNE6J9WRmPbO-k1UPtu2Bztd0-yc
  aggregator_auth_token:
    type: Bearer
    token: Y29sbGVjdG9yLWFiZjU0MDhlMmIxNjAxODMxNjI1YWYzOTU5MTA2NDU4
  collector_auth_token_hash:
    type: Bearer
    hash: MJOoBO_ysLEuG_lv2C37eEOf1Ngetsr-Ers0ZYj4vdQ
  hpke_keys: []
- peer_aggregator_endpoint: https://leader
  batch_mode: TimeInterval
  aggregation_mode: Asynchronous
  vdaf: !Prio3Sum
    max_measurement: 4096
  role: Helper
  vdaf_verify_key:
  task_end: 9000000000
  min_batch_size: 10
  time_precision: 300
  tolerable_clock_skew: 600
  collector_hpke_config:
    id: 23
    kem_id: X25519HkdfSha256
    kdf_id: HkdfSha256
    aead_id: Aes128Gcm
    public_key: 8lAqZ7OfNV2Gi_9cNE6J9WRmPbO-k1UPtu2Bztd0-yc
  aggregator_auth_token_hash:
    type: Bearer
    hash: MJOoBO_ysLEuG_lv2C37eEOf1Ngetsr-Ers0ZYj4vdQ
  collector_auth_token_hash:
  hpke_keys: []
"#;

        let ephemeral_datastore = ephemeral_datastore().await;
        let ds = ephemeral_datastore.datastore(RealClock::default()).await;

        let mut tasks_file = NamedTempFile::new().unwrap();
        tasks_file
            .write_all(serialized_task_yaml.as_bytes())
            .unwrap();
        let tasks_file_path = tasks_file.into_temp_path();

        super::provision_tasks(
            &ds,
            &tasks_file_path,
            // do not generate missing parameters
            false,
            // not a dry-run
            false,
        )
        .await
        // Should fail because parameters are omitted from task YAML
        .unwrap_err();

        let written_tasks = super::provision_tasks(
            &ds,
            &tasks_file_path,
            // generate missing parameters
            true,
            // not a dry-run
            false,
        )
        .await
        .unwrap();

        // Verify that the expected tasks were written.
        let got_tasks = ds
            .run_unnamed_tx(|tx| Box::pin(async move { tx.get_aggregator_tasks().await }))
            .await
            .unwrap();

        assert_eq!(got_tasks.len(), 2);

        for task in &got_tasks {
            match task.role() {
                Role::Leader => assert!(task.collector_auth_token_hash().is_some()),
                Role::Helper => assert!(task.collector_auth_token_hash().is_none()),
                role => panic!("unexpected role {role}"),
            }
        }

        assert_eq!(
            task_hashmap_from_slice(written_tasks),
            task_hashmap_from_slice(got_tasks)
        );
    }

    #[tokio::test]
    async fn create_datastore_key() {
        initialize_rustls();

        let k8s_cluster = kubernetes::EphemeralCluster::create();
        let kube_client = k8s_cluster.cluster().client().await.into();

        // Create a datastore key.
        const NAMESPACE: &str = "default";
        const SECRET_NAME: &str = "secret-name";
        const SECRET_DATA_KEY: &str = "secret-data-key";
        super::create_datastore_key(
            /* dry_run */ false,
            &kube_client,
            NAMESPACE,
            SECRET_NAME,
            SECRET_DATA_KEY,
        )
        .await
        .unwrap();

        // Verify that the secret was created.
        let secret_data =
            fetch_datastore_keys(&kube_client, NAMESPACE, SECRET_NAME, SECRET_DATA_KEY)
                .await
                .unwrap();

        // Verify that the written secret data can be parsed as a comma-separated list of datastore
        // keys.
        let datastore_key_bytes = URL_SAFE_NO_PAD.decode(&secret_data[0]).unwrap();
        UnboundKey::new(&AES_128_GCM, &datastore_key_bytes).unwrap();
    }

    #[tokio::test]
    async fn create_datastore_key_dry_run() {
        initialize_rustls();

        let k8s_cluster = kubernetes::EphemeralCluster::create();
        let kube_client = k8s_cluster.cluster().client().await.into();

        const NAMESPACE: &str = "default";
        const SECRET_NAME: &str = "secret-name";
        const SECRET_DATA_KEY: &str = "secret-data-key";
        super::create_datastore_key(
            /* dry_run */ true,
            &kube_client,
            NAMESPACE,
            SECRET_NAME,
            SECRET_DATA_KEY,
        )
        .await
        .unwrap();

        // Verify that no secret was created.
        fetch_datastore_keys(&kube_client, NAMESPACE, SECRET_NAME, SECRET_DATA_KEY)
            .await
            .unwrap_err();
    }

    #[test]
    fn roundtrip_config() {
        roundtrip_encoding(ConfigFile {
            common_config: CommonConfig {
                database: generate_db_config(),
                logging_config: generate_trace_config(),
                metrics_config: generate_metrics_config(),
                health_check_listen_address: SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080)),
                max_transaction_retries: default_max_transaction_retries(),
                thread_pool_stack_size: None,
            },
        })
    }

    #[test]
    fn documentation_config_examples() {
        serde_yaml::from_str::<ConfigFile>(include_str!(
            "../../../docs/samples/basic_config/janus_cli.yaml"
        ))
        .unwrap();
        serde_yaml::from_str::<ConfigFile>(include_str!(
            "../../../docs/samples/advanced_config/janus_cli.yaml"
        ))
        .unwrap();
    }
}
