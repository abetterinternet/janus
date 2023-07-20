use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use janus_aggregator_core::task::{QueryType, Task};
use janus_core::{
    hpke::{generate_hpke_config_and_private_key, HpkeKeypair},
    task::VdafInstance,
};
use janus_messages::{
    query_type::{FixedSize, QueryType as _},
    HpkeAeadId, HpkeConfigId, HpkeKdfId, HpkeKemId, Role, TaskId, Time,
};
use prio::codec::Encode;
use rand::random;
use serde::{de::Visitor, Deserialize, Serialize};
use std::{
    collections::HashMap,
    env::{self, VarError},
    fmt::Display,
    fs::create_dir_all,
    io::{stderr, Write},
    marker::PhantomData,
    ops::Deref,
    path::PathBuf,
    process::Command,
    str::FromStr,
    sync::Arc,
    thread::panicking,
};
use testcontainers::{Container, Image};
use tokio::sync::Mutex;
use tracing_log::LogTracer;
use tracing_subscriber::{prelude::*, EnvFilter, Registry};
use trillium::{async_trait, Conn, Handler, Status};
use trillium_api::ApiConnExt;
use url::Url;

#[cfg(feature = "testcontainer")]
pub mod testcontainer;

pub mod status {
    pub static SUCCESS: &str = "success";
    pub static ERROR: &str = "error";
    pub static COMPLETE: &str = "complete";
    pub static IN_PROGRESS: &str = "in progress";
}

/// Helper type to serialize/deserialize a large number as a string.
#[derive(Debug, Clone)]
pub struct NumberAsString<T>(pub T);

impl<T> Serialize for NumberAsString<T>
where
    T: Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de, T> Deserialize<'de> for NumberAsString<T>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
{
    fn deserialize<D>(deserializer: D) -> Result<NumberAsString<T>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(NumberAsStringVisitor::new())
    }
}

struct NumberAsStringVisitor<T>(PhantomData<T>);

impl<T> NumberAsStringVisitor<T> {
    fn new() -> NumberAsStringVisitor<T> {
        NumberAsStringVisitor(PhantomData)
    }
}

impl<'de, T> Visitor<'de> for NumberAsStringVisitor<T>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
{
    type Value = NumberAsString<T>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string with a number in base 10")
    }

    fn visit_str<E>(self, value: &str) -> Result<NumberAsString<T>, E>
    where
        E: serde::de::Error,
    {
        let number = value
            .parse()
            .map_err(|err| E::custom(format!("string could not be parsed into number: {err}")))?;
        Ok(NumberAsString(number))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VdafObject {
    Prio3Count,
    Prio3CountVec {
        length: NumberAsString<usize>,
    },
    Prio3Sum {
        bits: NumberAsString<usize>,
    },
    Prio3SumVec {
        bits: NumberAsString<usize>,
        length: NumberAsString<usize>,
    },
    Prio3Histogram {
        buckets: Vec<NumberAsString<u64>>,
    },
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint16BitBoundedL2VecSum {
        length: NumberAsString<usize>,
    },
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint32BitBoundedL2VecSum {
        length: NumberAsString<usize>,
    },
    #[cfg(feature = "fpvec_bounded_l2")]
    Prio3FixedPoint64BitBoundedL2VecSum {
        length: NumberAsString<usize>,
    },
}

impl From<VdafInstance> for VdafObject {
    fn from(vdaf: VdafInstance) -> Self {
        match vdaf {
            VdafInstance::Prio3Count => VdafObject::Prio3Count,

            VdafInstance::Prio3CountVec { length } => VdafObject::Prio3CountVec {
                length: NumberAsString(length),
            },

            VdafInstance::Prio3Sum { bits } => VdafObject::Prio3Sum {
                bits: NumberAsString(bits),
            },

            VdafInstance::Prio3SumVec { bits, length } => VdafObject::Prio3SumVec {
                bits: NumberAsString(bits),
                length: NumberAsString(length),
            },

            VdafInstance::Prio3Histogram { buckets } => VdafObject::Prio3Histogram {
                buckets: buckets.iter().copied().map(NumberAsString).collect(),
            },

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPoint16BitBoundedL2VecSum { length } => {
                VdafObject::Prio3FixedPoint16BitBoundedL2VecSum {
                    length: NumberAsString(length),
                }
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPoint32BitBoundedL2VecSum { length } => {
                VdafObject::Prio3FixedPoint32BitBoundedL2VecSum {
                    length: NumberAsString(length),
                }
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafInstance::Prio3FixedPoint64BitBoundedL2VecSum { length } => {
                VdafObject::Prio3FixedPoint64BitBoundedL2VecSum {
                    length: NumberAsString(length),
                }
            }
            _ => panic!("Unsupported VDAF: {vdaf:?}"),
        }
    }
}

impl From<VdafObject> for VdafInstance {
    fn from(vdaf: VdafObject) -> Self {
        match vdaf {
            VdafObject::Prio3Count => VdafInstance::Prio3Count,

            VdafObject::Prio3CountVec { length } => {
                VdafInstance::Prio3CountVec { length: length.0 }
            }

            VdafObject::Prio3Sum { bits } => VdafInstance::Prio3Sum { bits: bits.0 },

            VdafObject::Prio3SumVec { bits, length } => VdafInstance::Prio3SumVec {
                bits: bits.0,
                length: length.0,
            },

            VdafObject::Prio3Histogram { buckets } => VdafInstance::Prio3Histogram {
                buckets: buckets.iter().map(|value| value.0).collect(),
            },

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafObject::Prio3FixedPoint16BitBoundedL2VecSum { length } => {
                VdafInstance::Prio3FixedPoint16BitBoundedL2VecSum { length: length.0 }
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafObject::Prio3FixedPoint32BitBoundedL2VecSum { length } => {
                VdafInstance::Prio3FixedPoint32BitBoundedL2VecSum { length: length.0 }
            }

            #[cfg(feature = "fpvec_bounded_l2")]
            VdafObject::Prio3FixedPoint64BitBoundedL2VecSum { length } => {
                VdafInstance::Prio3FixedPoint64BitBoundedL2VecSum { length: length.0 }
            }
        }
    }
}

#[derive(Debug)]
pub struct BadRoleError;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AggregatorRole {
    Leader,
    Helper,
}

impl TryFrom<Role> for AggregatorRole {
    type Error = BadRoleError;

    fn try_from(role: Role) -> Result<Self, Self::Error> {
        match role {
            Role::Collector => Err(BadRoleError),
            Role::Client => Err(BadRoleError),
            Role::Leader => Ok(AggregatorRole::Leader),
            Role::Helper => Ok(AggregatorRole::Helper),
        }
    }
}

impl From<AggregatorRole> for Role {
    fn from(role: AggregatorRole) -> Self {
        match role {
            AggregatorRole::Leader => Role::Leader,
            AggregatorRole::Helper => Role::Helper,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregatorAddTaskRequest {
    pub task_id: TaskId, // uses unpadded base64url
    pub leader: Url,
    pub helper: Url,
    pub vdaf: VdafObject,
    pub leader_authentication_token: String,
    #[serde(default)]
    pub collector_authentication_token: Option<String>,
    pub role: AggregatorRole,
    pub vdaf_verify_key: String, // in unpadded base64url
    pub max_batch_query_count: u64,
    pub query_type: u8,
    pub min_batch_size: u64,
    pub max_batch_size: Option<u64>,
    pub time_precision: u64,           // in seconds
    pub collector_hpke_config: String, // in unpadded base64url
    pub task_expiration: Option<u64>,  // in seconds since the epoch
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddTaskResponse {
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
}

impl From<Task> for AggregatorAddTaskRequest {
    fn from(task: Task) -> Self {
        let (query_type, max_batch_size) = match task.query_type() {
            QueryType::FixedSize { max_batch_size } => {
                (FixedSize::CODE as u8, Some(*max_batch_size))
            }
        };
        Self {
            task_id: *task.id(),
            leader: task.aggregator_url(&Role::Leader).unwrap().clone(),
            helper: task.aggregator_url(&Role::Helper).unwrap().clone(),
            vdaf: task.vdaf().clone().into(),
            leader_authentication_token: String::from_utf8(
                task.primary_aggregator_auth_token().as_ref().to_vec(),
            )
            .unwrap(),
            collector_authentication_token: if task.role() == &Role::Leader {
                Some(
                    String::from_utf8(task.primary_collector_auth_token().as_ref().to_vec())
                        .unwrap(),
                )
            } else {
                None
            },
            role: (*task.role()).try_into().unwrap(),
            vdaf_verify_key: URL_SAFE_NO_PAD
                .encode(task.vdaf_verify_keys().first().unwrap().as_ref()),
            max_batch_query_count: task.max_batch_query_count(),
            query_type,
            min_batch_size: task.min_batch_size(),
            max_batch_size,
            time_precision: task.time_precision().as_seconds(),
            collector_hpke_config: URL_SAFE_NO_PAD
                .encode(task.collector_hpke_config().get_encoded()),
            task_expiration: task.task_expiration().map(Time::as_seconds_since_epoch),
        }
    }
}

pub fn install_tracing_subscriber() -> anyhow::Result<()> {
    let stdout_filter = EnvFilter::builder().from_env()?;
    let layer = tracing_subscriber::fmt::layer()
        .with_thread_ids(true)
        .with_level(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false)
        .pretty();
    let subscriber = Registry::default().with(stdout_filter.and_then(layer));
    tracing::subscriber::set_global_default(subscriber)?;

    LogTracer::init()?;

    Ok(())
}

/// This registry lazily generates up to 256 HPKE key pairs, one with each possible
/// [`HpkeConfigId`].
#[derive(Default)]
pub struct HpkeConfigRegistry {
    keypairs: HashMap<HpkeConfigId, HpkeKeypair>,
}

impl HpkeConfigRegistry {
    pub fn new() -> HpkeConfigRegistry {
        Default::default()
    }

    /// Get the keypair associated with a given ID.
    pub fn fetch_keypair(&mut self, id: HpkeConfigId) -> HpkeKeypair {
        self.keypairs
            .entry(id)
            .or_insert_with(|| {
                generate_hpke_config_and_private_key(
                    id,
                    // These algorithms should be broadly compatible with other DAP implementations, since they
                    // are required by section 6 of draft-ietf-ppm-dap-02.
                    HpkeKemId::X25519HkdfSha256,
                    HpkeKdfId::HkdfSha256,
                    HpkeAeadId::Aes128Gcm,
                )
            })
            .clone()
    }

    /// Choose a random [`HpkeConfigId`], and then get the keypair associated with that ID.
    pub fn get_random_keypair(&mut self) -> HpkeKeypair {
        self.fetch_keypair(random::<u8>().into())
    }
}

/// log_export_path returns the path to export container logs to, or None if container logs are not
/// configured to be exported.
///
/// The resulting value is based directly on the JANUS_E2E_LOGS_PATH environment variable.
pub fn log_export_path() -> Option<PathBuf> {
    match env::var("JANUS_E2E_LOGS_PATH") {
        Ok(logs_path) => Some(PathBuf::from_str(&logs_path).unwrap()),
        Err(VarError::NotPresent) => None,
        Err(err) => panic!("Failed to parse JANUS_E2E_LOGS_PATH: {err}"),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ContainerInspectEntry {
    name: String,
}

pub struct ContainerLogsDropGuard<'d, I: Image> {
    container: Container<'d, I>,
}

impl<'d, I: Image> ContainerLogsDropGuard<'d, I> {
    pub fn new(container: Container<I>) -> ContainerLogsDropGuard<I> {
        ContainerLogsDropGuard { container }
    }
}

impl<'d, I: Image> Drop for ContainerLogsDropGuard<'d, I> {
    fn drop(&mut self) {
        if !panicking() {
            return;
        }
        if let Some(base_dir) = log_export_path() {
            create_dir_all(&base_dir).expect("could not create log output directory");

            let id = self.container.id();

            let inspect_output = Command::new("docker")
                .args(["container", "inspect", id])
                .output()
                .expect("running `docker container inspect` failed");
            stderr().write_all(&inspect_output.stderr).unwrap();
            assert!(inspect_output.status.success());
            let inspect_array: Vec<ContainerInspectEntry> =
                serde_json::from_slice(&inspect_output.stdout).unwrap();
            let inspect_entry = inspect_array
                .first()
                .expect("`docker container inspect` returned no results");
            let name = &inspect_entry.name[inspect_entry
                .name
                .find('/')
                .map(|index| index + 1)
                .unwrap_or_default()..];

            let destination = base_dir.join(name);

            let copy_status = Command::new("docker")
                .arg("cp")
                .arg(format!("{id}:/logs"))
                .arg(destination)
                .status()
                .expect("running `docker cp` failed");
            assert!(copy_status.success());
        }
    }
}

impl<'d, I: Image> Deref for ContainerLogsDropGuard<'d, I> {
    type Target = Container<'d, I>;

    fn deref(&self) -> &Self::Target {
        &self.container
    }
}

pub struct ErrorHandler;

#[async_trait]
impl Handler for ErrorHandler {
    async fn run(&self, conn: Conn) -> Conn {
        conn
    }

    async fn before_send(&self, mut conn: Conn) -> Conn {
        if let Some(error) = conn.take_state::<trillium_api::Error>() {
            // Could not parse the request, return a 400 status code.
            return conn
                .with_json(&serde_json::json!({
                    "status": status::ERROR,
                    "error": format!("{error}"),
                }))
                .with_status(Status::BadRequest);
        }
        conn
    }
}

/// A wrapper around [`HpkeConfigRegistry`] that is suitable for use in Trillium connection
/// state.
#[derive(Clone)]
pub struct Keyring(pub Arc<Mutex<HpkeConfigRegistry>>);

impl Keyring {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HpkeConfigRegistry::new())))
    }
}

impl Default for Keyring {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "test-util")]
pub mod test_util {
    use backoff::{future::retry, ExponentialBackoff};
    use futures::{Future, TryFutureExt};
    use rand::random;
    use std::{fmt::Debug, time::Duration};
    use url::Url;

    lazy_static::lazy_static! {
        static ref DOCKER_HASH_RE: regex::Regex = regex::Regex::new(r"sha256:([0-9a-f]{64})").unwrap();
    }

    async fn await_readiness_condition<
        I,
        E: Debug,
        Fn: FnMut() -> Fut,
        Fut: Future<Output = Result<I, backoff::Error<E>>>,
    >(
        operation: Fn,
    ) {
        retry(
            // (We use ExponentialBackoff as a constant-time backoff as the built-in Constant
            // backoff will never time out.)
            ExponentialBackoff {
                initial_interval: Duration::from_millis(250),
                max_interval: Duration::from_millis(250),
                multiplier: 1.0,
                max_elapsed_time: Some(Duration::from_secs(30)),
                ..Default::default()
            },
            operation,
        )
        .await
        .unwrap();
    }

    /// Waits a while for the given IPv4 port to start responding successfully to
    /// "/internal/test/ready" HTTP requests, panicking if this doesn't happen soon enough.
    pub async fn await_ready_ok(port: u16) {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        // Explicitly connect to IPv4 localhost so that we don't accidentally try to talk to a
        // different container's IPv6 port.
        let url = Url::parse(&format!("http://127.0.0.1:{port}/internal/test/ready")).unwrap();
        await_readiness_condition(|| async {
            http_client
                .post(url.clone())
                .send()
                .await
                .map_err(backoff::Error::transient)?
                .error_for_status()
                .map_err(backoff::Error::transient)
        })
        .await
    }

    /// Waits a while for the given port to start responding to HTTP requests, panicking if this
    /// doesn't happen soon enough.
    pub async fn await_http_server(port: u16) {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(1))
            .build()
            .unwrap();
        let url = Url::parse(&format!("http://127.0.0.1:{port}/")).unwrap();
        await_readiness_condition(|| {
            http_client
                .get(url.clone())
                .send()
                .map_err(backoff::Error::transient)
        })
        .await
    }

    /// Loads a given zstd-compressed docker image into Docker. Returns the hash of the loaded
    /// image, e.g. as referenced by `sha256:$HASH`. Panics on failure.
    pub fn load_zstd_compressed_docker_image(compressed_image: &[u8]) -> String {
        use std::{
            io::{Cursor, Read},
            process::{Command, Stdio},
            thread,
        };

        let mut docker_load_child = Command::new("docker")
            .args(["load", "--quiet"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to execute `docker load`");
        let child_stdin = docker_load_child.stdin.take().unwrap();
        thread::scope(|s| {
            let writer_handle = s.spawn(|| {
                // We write in a separate thread as "writing more than a pipe buffer's
                // worth of input to stdin without also reading stdout and stderr at the
                // same time may cause a deadlock."
                zstd::stream::copy_decode(Cursor::new(compressed_image), child_stdin)
            });
            let reader_handle = s.spawn(|| {
                let mut child_stdout = docker_load_child.stdout.take().unwrap();
                let mut stdout = String::new();
                child_stdout
                    .read_to_string(&mut stdout)
                    .expect("Couldn't read image ID from docker");
                let caps = DOCKER_HASH_RE
                    .captures(&stdout)
                    .expect("Couldn't find image ID from `docker load` output");
                caps.get(1).unwrap().as_str().to_string()
            });

            // The first `expect` catches panics, the second `expect` catches write errors.
            writer_handle
                .join()
                .expect("Couldn't write image to docker")
                .expect("Couldn't write image to docker");
            reader_handle.join().unwrap()
        })
    }

    pub fn generate_network_name() -> String {
        generate_unique_name("janus_ephemeral_network")
    }

    pub fn generate_unique_name(prefix: &str) -> String {
        format!("{}_{}", prefix, hex::encode(random::<[u8; 4]>()))
    }
}
