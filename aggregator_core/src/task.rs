//! Shared parameters for a DAP task.

use crate::SecretBytes;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use derivative::Derivative;
use janus_core::{
    hpke::{generate_hpke_config_and_private_key, HpkeKeypair},
    task::{url_ensure_trailing_slash, AuthenticationToken, VdafInstance},
};
use janus_messages::{
    AggregationJobId, CollectionJobId, Duration, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId,
    HpkeKemId, Role, TaskId, Time,
};
use rand::{distributions::Standard, random, thread_rng, Rng};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use std::{
    array::TryFromSliceError,
    collections::HashMap,
    fmt::{self, Formatter},
};
use url::Url;

/// Errors that methods and functions in this module may return.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid parameter {0}")]
    InvalidParameter(&'static str),
    #[error("URL parse error")]
    Url(#[from] url::ParseError),
    #[error("aggregator verification key size out of range")]
    AggregatorVerifyKeySize,
    #[error("base64 decode error")]
    Base64Decode(#[from] base64::DecodeError),
}

/// Identifiers for query types used by a task, along with query-type specific configuration.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueryType {
    /// Fixed-size: used to support collection of batches as quickly as possible, without aligning
    /// to a fixed batch window.
    FixedSize {
        /// The maximum number of reports in a batch to allow it to be collected.
        max_batch_size: u64,
    },
}

/// A verification key for a VDAF, with a fixed length. It must be kept secret from clients to
/// maintain robustness, and it must be shared between aggregators.
pub struct VerifyKey<const SEED_SIZE: usize>([u8; SEED_SIZE]);

impl<const SEED_SIZE: usize> VerifyKey<SEED_SIZE> {
    pub fn new(array: [u8; SEED_SIZE]) -> VerifyKey<SEED_SIZE> {
        VerifyKey(array)
    }

    pub fn as_bytes(&self) -> &[u8; SEED_SIZE] {
        &self.0
    }
}

impl<const SEED_SIZE: usize> TryFrom<&SecretBytes> for VerifyKey<SEED_SIZE> {
    type Error = TryFromSliceError;

    fn try_from(value: &SecretBytes) -> Result<VerifyKey<SEED_SIZE>, TryFromSliceError> {
        let array = <[u8; SEED_SIZE] as TryFrom<&[u8]>>::try_from(&value.0)?;
        Ok(VerifyKey::new(array))
    }
}

/// The parameters for a DAP task, corresponding to draft-gpew-priv-ppm §4.2.
#[derive(Clone, Derivative, PartialEq, Eq)]
#[derivative(Debug)]
pub struct Task {
    /// Unique identifier for the task.
    task_id: TaskId,
    /// URLs relative to which aggregator API endpoints are found. The first
    /// entry is the leader's.
    #[derivative(Debug(format_with = "fmt_vector_of_urls"))]
    aggregator_endpoints: Vec<Url>,
    /// The query type this task uses to generate batches.
    query_type: QueryType,
    /// The VDAF this task executes.
    vdaf: VdafInstance,
    /// The role performed by the aggregator.
    role: Role,
    /// Secret verification keys shared by the aggregators.
    #[derivative(Debug = "ignore")]
    vdaf_verify_keys: Vec<SecretBytes>,
    /// The maximum number of times a given batch may be collected.
    max_batch_query_count: u64,
    /// The time after which the task is considered invalid.
    task_expiration: Option<Time>,
    /// The age after which a report is considered to be "expired" and will be considered a
    /// candidate for garbage collection. A value of `None` indicates that garbage collection is
    /// disabled.
    report_expiry_age: Option<Duration>,
    /// The minimum number of reports in a batch to allow it to be collected.
    min_batch_size: u64,
    /// The duration to which clients should round their reported timestamps to. For time-interval
    /// tasks, batch intervals must be multiples of this duration.
    time_precision: Duration,
    /// How much clock skew to allow between client and aggregator. Reports from
    /// farther than this duration into the future will be rejected.
    tolerable_clock_skew: Duration,
    /// HPKE configuration for the collector.
    collector_hpke_config: HpkeConfig,
    /// Tokens used to authenticate messages sent to or received from the other aggregator.
    aggregator_auth_tokens: Vec<AuthenticationToken>,
    /// Tokens used to authenticate messages sent to or received from the collector.
    collector_auth_tokens: Vec<AuthenticationToken>,
    /// HPKE configurations & private keys used by this aggregator to decrypt client reports.
    hpke_keys: HashMap<HpkeConfigId, HpkeKeypair>,
}

impl Task {
    /// Create a new [`Task`] from the provided values
    #[allow(clippy::too_many_arguments)]
    pub fn new<I: IntoIterator<Item = HpkeKeypair>>(
        task_id: TaskId,
        mut aggregator_endpoints: Vec<Url>,
        query_type: QueryType,
        vdaf: VdafInstance,
        role: Role,
        vdaf_verify_keys: Vec<SecretBytes>,
        max_batch_query_count: u64,
        task_expiration: Option<Time>,
        report_expiry_age: Option<Duration>,
        min_batch_size: u64,
        time_precision: Duration,
        tolerable_clock_skew: Duration,
        collector_hpke_config: HpkeConfig,
        aggregator_auth_tokens: Vec<AuthenticationToken>,
        collector_auth_tokens: Vec<AuthenticationToken>,
        hpke_keys: I,
    ) -> Result<Self, Error> {
        // Ensure provided aggregator endpoints end with a slash, as we will be joining additional
        // path segments into these endpoints & the Url::join implementation is persnickety about
        // the slash at the end of the path.
        for url in &mut aggregator_endpoints {
            url_ensure_trailing_slash(url);
        }

        // Compute hpke_configs mapping cfg.id -> (cfg, key).
        let hpke_keys: HashMap<HpkeConfigId, HpkeKeypair> = hpke_keys
            .into_iter()
            .map(|keypair| (*keypair.config().id(), keypair))
            .collect();

        let task = Self {
            task_id,
            aggregator_endpoints,
            query_type,
            vdaf,
            role,
            vdaf_verify_keys,
            max_batch_query_count,
            task_expiration,
            report_expiry_age,
            min_batch_size,
            time_precision,
            tolerable_clock_skew,
            collector_hpke_config,
            aggregator_auth_tokens,
            collector_auth_tokens,
            hpke_keys,
        };
        task.validate()?;
        Ok(task)
    }

    fn validate(&self) -> Result<(), Error> {
        // DAP currently only supports configurations of exactly two aggregators.
        if self.aggregator_endpoints.len() != 2 {
            return Err(Error::InvalidParameter("aggregator_endpoints"));
        }
        if !self.role.is_aggregator() {
            return Err(Error::InvalidParameter("role"));
        }
        if self.aggregator_auth_tokens.is_empty() {
            return Err(Error::InvalidParameter("aggregator_auth_tokens"));
        }
        if (self.role == Role::Leader) == (self.collector_auth_tokens.is_empty()) {
            // Collector auth tokens are allowed & required if and only if this task is in the
            // leader role.
            return Err(Error::InvalidParameter("collector_auth_tokens"));
        }
        if self.vdaf_verify_keys.is_empty() {
            return Err(Error::InvalidParameter("vdaf_verify_keys"));
        }
        if self.hpke_keys.is_empty() {
            return Err(Error::InvalidParameter("hpke_keys"));
        }
        let QueryType::FixedSize { max_batch_size } = self.query_type();
        if *max_batch_size < self.min_batch_size() {
            return Err(Error::InvalidParameter("max_batch_size"));
        }
        Ok(())
    }

    /// Retrieves the task ID associated with this task.
    pub fn id(&self) -> &TaskId {
        &self.task_id
    }

    /// Retrieves the aggregator endpoints associated with this task in natural order.
    pub fn aggregator_endpoints(&self) -> &[Url] {
        &self.aggregator_endpoints
    }

    /// Retrieves the query type associated with this task.
    pub fn query_type(&self) -> &QueryType {
        &self.query_type
    }

    /// Retrieves the VDAF associated with this task.
    pub fn vdaf(&self) -> &VdafInstance {
        &self.vdaf
    }

    /// Retrieves the role associated with this task.
    pub fn role(&self) -> &Role {
        &self.role
    }

    /// Retrieves the VDAF verification keys associated with this task.
    pub fn vdaf_verify_keys(&self) -> &[SecretBytes] {
        &self.vdaf_verify_keys
    }

    /// Retrieves the max batch query count parameter associated with this task.
    pub fn max_batch_query_count(&self) -> u64 {
        self.max_batch_query_count
    }

    /// Retrieves the task expiration associated with this task.
    pub fn task_expiration(&self) -> Option<&Time> {
        self.task_expiration.as_ref()
    }

    /// Retrieves the report expiry age associated with this task.
    pub fn report_expiry_age(&self) -> Option<&Duration> {
        self.report_expiry_age.as_ref()
    }

    /// Retrieves the min batch size parameter associated with this task.
    pub fn min_batch_size(&self) -> u64 {
        self.min_batch_size
    }

    /// Retrieves the time precision parameter associated with this task.
    pub fn time_precision(&self) -> &Duration {
        &self.time_precision
    }

    /// Retrieves the tolerable clock skew parameter associated with this task.
    pub fn tolerable_clock_skew(&self) -> &Duration {
        &self.tolerable_clock_skew
    }

    /// Retrieves the collector HPKE config associated with this task.
    pub fn collector_hpke_config(&self) -> &HpkeConfig {
        &self.collector_hpke_config
    }

    /// Retrieves the aggregator authentication tokens associated with this task.
    pub fn aggregator_auth_tokens(&self) -> &[AuthenticationToken] {
        &self.aggregator_auth_tokens
    }

    /// Retrieves the collector authentication tokens associated with this task.
    pub fn collector_auth_tokens(&self) -> &[AuthenticationToken] {
        &self.collector_auth_tokens
    }

    /// Retrieves the HPKE keys in use associated with this task.
    pub fn hpke_keys(&self) -> &HashMap<HpkeConfigId, HpkeKeypair> {
        &self.hpke_keys
    }

    /// Retrieve the "current" HPKE in use for this task.
    #[cfg(feature = "test-util")]
    pub fn current_hpke_key(&self) -> &HpkeKeypair {
        self.hpke_keys
            .values()
            .max_by_key(|keypair| u8::from(*keypair.config().id()))
            .unwrap()
    }

    /// Returns true if the `batch_size` is valid given this task's query type and batch size
    /// parameters, per
    /// <https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6>
    pub fn validate_batch_size(&self, batch_size: u64) -> bool {
        match self.query_type {
            QueryType::FixedSize { max_batch_size } => {
                // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6.2.2
                batch_size >= self.min_batch_size() && batch_size <= max_batch_size
            }
        }
    }

    /// Returns the [`Url`] relative to which the server performing `role` serves its API.
    pub fn aggregator_url(&self, role: &Role) -> Result<&Url, Error> {
        let index = role.index().ok_or(Error::InvalidParameter(role.as_str()))?;
        Ok(&self.aggregator_endpoints[index])
    }

    /// Returns the [`AuthenticationToken`] currently used by this aggregator to authenticate itself
    /// to other aggregators.
    pub fn primary_aggregator_auth_token(&self) -> &AuthenticationToken {
        self.aggregator_auth_tokens.iter().next_back().unwrap()
    }

    /// Checks if the given aggregator authentication token is valid (i.e. matches with an
    /// authentication token recognized by this task).
    pub fn check_aggregator_auth_token(&self, auth_token: &AuthenticationToken) -> bool {
        self.aggregator_auth_tokens
            .iter()
            .rev()
            .any(|t| t == auth_token)
    }

    /// Returns the [`AuthenticationToken`] currently used by the collector to authenticate itself
    /// to the aggregators.
    pub fn primary_collector_auth_token(&self) -> &AuthenticationToken {
        // Unwrap safety: self.collector_auth_tokens is never empty
        self.collector_auth_tokens.iter().next_back().unwrap()
    }

    /// Checks if the given collector authentication token is valid (i.e. matches with an
    /// authentication token recognized by this task).
    pub fn check_collector_auth_token(&self, auth_token: &AuthenticationToken) -> bool {
        self.collector_auth_tokens
            .iter()
            .rev()
            .any(|t| t == auth_token)
    }

    /// Returns the [`VerifyKey`] currently used by this aggregator to prepare report shares with
    /// other aggregators.
    ///
    /// # Errors
    ///
    /// If the verify key is not the correct length as required by the VDAF, an error will be
    /// returned.
    pub fn primary_vdaf_verify_key<const SEED_SIZE: usize>(
        &self,
    ) -> Result<VerifyKey<SEED_SIZE>, Error> {
        // We can safely unwrap this because we maintain an invariant that this vector is
        // non-empty.
        let secret_bytes = self.vdaf_verify_keys.first().unwrap();
        VerifyKey::try_from(secret_bytes).map_err(|_| Error::AggregatorVerifyKeySize)
    }

    /// Returns the relative path for tasks, relative to which other API endpoints are defined.
    fn tasks_path(&self) -> String {
        format!("tasks/{}", self.id())
    }

    /// Returns the URI at which reports may be uploaded for this task.
    pub fn report_upload_uri(&self) -> Result<Url, Error> {
        Ok(self
            .aggregator_url(&Role::Leader)?
            .join(&format!("{}/reports", self.tasks_path()))?)
    }

    /// Returns the URI at which the helper resource for the specified aggregation job ID can be
    /// accessed.
    pub fn aggregation_job_uri(&self, aggregation_job_id: &AggregationJobId) -> Result<Url, Error> {
        Ok(self.aggregator_url(&Role::Helper)?.join(&format!(
            "{}/aggregation_jobs/{aggregation_job_id}",
            self.tasks_path()
        ))?)
    }

    /// Returns the URI at which the helper aggregate shares resource can be accessed.
    pub fn aggregate_shares_uri(&self) -> Result<Url, Error> {
        Ok(self
            .aggregator_url(&Role::Helper)?
            .join(&format!("{}/aggregate_shares", self.tasks_path()))?)
    }

    /// Returns the URI at which the leader resource for the specified collection job ID can be
    /// accessed.
    pub fn collection_job_uri(&self, collection_job_id: &CollectionJobId) -> Result<Url, Error> {
        Ok(self.aggregator_url(&Role::Leader)?.join(&format!(
            "{}/collection_jobs/{collection_job_id}",
            self.tasks_path()
        ))?)
    }
}

fn fmt_vector_of_urls(urls: &Vec<Url>, f: &mut Formatter<'_>) -> fmt::Result {
    let mut list = f.debug_list();
    for url in urls {
        list.entry(&format!("{url}"));
    }
    list.finish()
}

/// SerializedTask is an intermediate representation for tasks being serialized via the Serialize &
/// Deserialize traits.
#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedTask {
    task_id: Option<TaskId>,
    aggregator_endpoints: Vec<Url>,
    query_type: QueryType,
    vdaf: VdafInstance,
    role: Role,
    vdaf_verify_keys: Vec<String>, // in unpadded base64url
    max_batch_query_count: u64,
    task_expiration: Option<Time>,
    report_expiry_age: Option<Duration>,
    min_batch_size: u64,
    time_precision: Duration,
    tolerable_clock_skew: Duration,
    collector_hpke_config: HpkeConfig,
    aggregator_auth_tokens: Vec<AuthenticationToken>,
    collector_auth_tokens: Vec<AuthenticationToken>,
    hpke_keys: Vec<HpkeKeypair>, // uses unpadded base64url
}

impl SerializedTask {
    /// Returns the task ID, if one is set.
    pub fn task_id(&self) -> Option<TaskId> {
        self.task_id
    }

    /// Randomly generates and fills values for the following fields if they are not set in the
    /// [`SerializedTask`]
    ///
    /// - Task ID
    /// - VDAF verify keys (only one key is generated)
    /// - Aggregator authentication tokens (only one token is generated)
    /// - Collector authentication tokens (only one token is generated and only if the task's role
    ///   is leader)
    /// - The aggregator's HPKE keypair (only one keypair is generated)
    pub fn generate_missing_fields(&mut self) {
        if self.task_id.is_none() {
            let task_id: TaskId = random();
            self.task_id = Some(task_id);
        }

        if self.vdaf_verify_keys.is_empty() {
            let vdaf_verify_key = SecretBytes::new(
                thread_rng()
                    .sample_iter(Standard)
                    .take(self.vdaf.verify_key_length())
                    .collect(),
            );

            self.vdaf_verify_keys = Vec::from([URL_SAFE_NO_PAD.encode(vdaf_verify_key.as_ref())]);
        }

        if self.aggregator_auth_tokens.is_empty() {
            self.aggregator_auth_tokens = Vec::from([random()]);
        }

        if self.collector_auth_tokens.is_empty() && self.role == Role::Leader {
            self.collector_auth_tokens = Vec::from([random()]);
        }

        if self.hpke_keys.is_empty() {
            let hpke_keypair = generate_hpke_config_and_private_key(
                random(),
                HpkeKemId::X25519HkdfSha256,
                HpkeKdfId::HkdfSha256,
                HpkeAeadId::Aes128Gcm,
            );

            self.hpke_keys = Vec::from([hpke_keypair]);
        }
    }
}

impl Serialize for Task {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let vdaf_verify_keys: Vec<_> = self
            .vdaf_verify_keys
            .iter()
            .map(|key| URL_SAFE_NO_PAD.encode(key.as_ref()))
            .collect();
        let hpke_keys = self.hpke_keys.values().cloned().collect();

        SerializedTask {
            task_id: Some(self.task_id),
            aggregator_endpoints: self.aggregator_endpoints.clone(),
            query_type: self.query_type,
            vdaf: self.vdaf.clone(),
            role: self.role,
            vdaf_verify_keys,
            max_batch_query_count: self.max_batch_query_count,
            task_expiration: self.task_expiration,
            report_expiry_age: self.report_expiry_age,
            min_batch_size: self.min_batch_size,
            time_precision: self.time_precision,
            tolerable_clock_skew: self.tolerable_clock_skew,
            collector_hpke_config: self.collector_hpke_config.clone(),
            aggregator_auth_tokens: self.aggregator_auth_tokens.clone(),
            collector_auth_tokens: self.collector_auth_tokens.clone(),
            hpke_keys,
        }
        .serialize(serializer)
    }
}

impl TryFrom<SerializedTask> for Task {
    type Error = Error;

    fn try_from(serialized_task: SerializedTask) -> Result<Self, Self::Error> {
        // task_id
        let task_id = serialized_task
            .task_id
            .ok_or(Error::InvalidParameter("missing field task_id"))?;

        // vdaf_verify_keys
        let vdaf_verify_keys: Vec<_> = serialized_task
            .vdaf_verify_keys
            .into_iter()
            .map(|key| Ok(SecretBytes::new(URL_SAFE_NO_PAD.decode(key)?)))
            .collect::<Result<_, Self::Error>>()?;

        Task::new(
            task_id,
            serialized_task.aggregator_endpoints,
            serialized_task.query_type,
            serialized_task.vdaf,
            serialized_task.role,
            vdaf_verify_keys,
            serialized_task.max_batch_query_count,
            serialized_task.task_expiration,
            serialized_task.report_expiry_age,
            serialized_task.min_batch_size,
            serialized_task.time_precision,
            serialized_task.tolerable_clock_skew,
            serialized_task.collector_hpke_config,
            serialized_task.aggregator_auth_tokens,
            serialized_task.collector_auth_tokens,
            serialized_task.hpke_keys,
        )
    }
}

impl<'de> Deserialize<'de> for Task {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Deserialize into intermediate representation.
        let serialized_task = SerializedTask::deserialize(deserializer)?;
        Task::try_from(serialized_task).map_err(D::Error::custom)
    }
}

// This is public to allow use in integration tests.
#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util {
    use crate::{
        task::{QueryType, Task},
        SecretBytes,
    };
    use janus_core::{
        hpke::{test_util::generate_test_hpke_config_and_private_key, HpkeKeypair},
        task::{AuthenticationToken, VdafInstance, PRIO3_VERIFY_KEY_LENGTH},
        time::DurationExt,
    };
    use janus_messages::{Duration, HpkeConfig, HpkeConfigId, Role, TaskId, Time};
    use rand::{distributions::Standard, random, thread_rng, Rng};
    use url::Url;

    /// Returns the expected length of a VDAF verification key for a VDAF of this type.
    fn verify_key_length(vdaf: &VdafInstance) -> usize {
        match vdaf {
            VdafInstance::Fake
            | VdafInstance::FakeFailsPrepInit
            | VdafInstance::FakeFailsPrepStep => 0,

            // All "real" VDAFs use a verify key of length 16 currently. (Poplar1 may not, but it's
            // not yet done being specified, so choosing 16 bytes is fine for testing.)
            _ => PRIO3_VERIFY_KEY_LENGTH,
        }
    }

    /// TaskBuilder is a testing utility allowing tasks to be built based on a template.
    #[derive(Clone)]
    pub struct TaskBuilder(Task);

    impl TaskBuilder {
        /// Create a [`TaskBuilder`] from the provided values, with arbitrary values for the other
        /// task parameters.
        pub fn new(query_type: QueryType, vdaf: VdafInstance, role: Role) -> Self {
            let task_id = random();
            let aggregator_keypair_0 = generate_test_hpke_config_and_private_key();
            let mut aggregator_keypair_1 = generate_test_hpke_config_and_private_key();
            let mut aggregator_config_1 = aggregator_keypair_1.config().clone();
            aggregator_config_1 = HpkeConfig::new(
                HpkeConfigId::from(1),
                *aggregator_config_1.kem_id(),
                *aggregator_config_1.kdf_id(),
                *aggregator_config_1.aead_id(),
                aggregator_config_1.public_key().clone(),
            );
            aggregator_keypair_1 = HpkeKeypair::new(
                aggregator_config_1,
                aggregator_keypair_1.private_key().clone(),
            );

            let vdaf_verify_key = SecretBytes::new(
                thread_rng()
                    .sample_iter(Standard)
                    .take(verify_key_length(&vdaf))
                    .collect(),
            );

            let collector_auth_tokens = if role == Role::Leader {
                Vec::from([random(), AuthenticationToken::DapAuth(random())])
            } else {
                Vec::new()
            };

            Self(
                Task::new(
                    task_id,
                    Vec::from([
                        "https://leader.endpoint".parse().unwrap(),
                        "https://helper.endpoint".parse().unwrap(),
                    ]),
                    query_type,
                    vdaf,
                    role,
                    Vec::from([vdaf_verify_key]),
                    1,
                    None,
                    None,
                    0,
                    Duration::from_hours(8).unwrap(),
                    Duration::from_minutes(10).unwrap(),
                    generate_test_hpke_config_and_private_key().config().clone(),
                    Vec::from([random(), AuthenticationToken::DapAuth(random())]),
                    collector_auth_tokens,
                    Vec::from([aggregator_keypair_0, aggregator_keypair_1]),
                )
                .unwrap(),
            )
        }

        /// Associates the eventual task with the given task ID.
        pub fn with_id(self, task_id: TaskId) -> Self {
            Self(Task { task_id, ..self.0 })
        }

        /// Associates the eventual task with the given aggregator endpoints.
        pub fn with_aggregator_endpoints(self, aggregator_endpoints: Vec<Url>) -> Self {
            Self(Task {
                aggregator_endpoints,
                ..self.0
            })
        }

        /// Retrieves the aggregator endpoints associated with this task builder.
        pub fn aggregator_endpoints(&self) -> &[Url] {
            self.0.aggregator_endpoints()
        }

        /// Associates the eventual task with the given aggregator role.
        pub fn with_role(self, role: Role) -> Self {
            Self(Task { role, ..self.0 })
        }

        /// Associates the eventual task with the given VDAF verification keys.
        pub fn with_vdaf_verify_keys(self, vdaf_verify_keys: Vec<SecretBytes>) -> Self {
            Self(Task {
                vdaf_verify_keys,
                ..self.0
            })
        }

        /// Associates the eventual task with the given max batch query count parameter.
        pub fn with_max_batch_query_count(self, max_batch_query_count: u64) -> Self {
            Self(Task {
                max_batch_query_count,
                ..self.0
            })
        }

        /// Associates the eventual task with the given min batch size parameter.
        pub fn with_min_batch_size(self, min_batch_size: u64) -> Self {
            Self(Task {
                min_batch_size,
                ..self.0
            })
        }

        /// Associates the eventual task with the given time precision parameter.
        pub fn with_time_precision(self, time_precision: Duration) -> Self {
            Self(Task {
                time_precision,
                ..self.0
            })
        }

        /// Associates the eventual task with the given collector HPKE config.
        pub fn with_collector_hpke_config(self, collector_hpke_config: HpkeConfig) -> Self {
            Self(Task {
                collector_hpke_config,
                ..self.0
            })
        }

        /// Associates the eventual task with the given aggregator authentication tokens.
        pub fn with_aggregator_auth_tokens(
            self,
            aggregator_auth_tokens: Vec<AuthenticationToken>,
        ) -> Self {
            Self(Task {
                aggregator_auth_tokens,
                ..self.0
            })
        }

        /// Sets the collector authentication tokens for the task.
        pub fn with_collector_auth_tokens(
            self,
            collector_auth_tokens: Vec<AuthenticationToken>,
        ) -> Self {
            Self(Task {
                collector_auth_tokens,
                ..self.0
            })
        }

        /// Sets the task expiration time.
        pub fn with_task_expiration(self, task_expiration: Option<Time>) -> Self {
            Self(Task {
                task_expiration,
                ..self.0
            })
        }

        /// Sets the report expiry age.
        pub fn with_report_expiry_age(self, report_expiry_age: Option<Duration>) -> Self {
            Self(Task {
                report_expiry_age,
                ..self.0
            })
        }

        /// Sets the task HPKE keys
        pub fn with_hpke_keys(self, hpke_keys: Vec<HpkeKeypair>) -> Self {
            let hpke_keys = hpke_keys
                .into_iter()
                .map(|hpke_keypair| (*hpke_keypair.config().id(), hpke_keypair))
                .collect();
            Self(Task {
                hpke_keys,
                ..self.0
            })
        }

        /// Consumes this task builder & produces a [`Task`] with the given specifications.
        pub fn build(self) -> Task {
            self.0.validate().unwrap();
            self.0
        }
    }

    impl From<Task> for TaskBuilder {
        fn from(task: Task) -> Self {
            Self(task)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        task::{test_util::TaskBuilder, QueryType, Task, VdafInstance},
        SecretBytes,
    };
    use janus_core::{
        hpke::{test_util::generate_test_hpke_config_and_private_key, HpkeKeypair, HpkePrivateKey},
        task::{AuthenticationToken, PRIO3_VERIFY_KEY_LENGTH},
        test_util::roundtrip_encoding,
        time::DurationExt,
    };
    use janus_messages::{
        Duration, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId, HpkeKemId, HpkePublicKey, Role,
        TaskId,
    };
    use rand::random;
    use serde_test::{assert_tokens, Token};
    use url::Url;

    #[test]
    fn task_serialization() {
        roundtrip_encoding(
            TaskBuilder::new(
                QueryType::FixedSize { max_batch_size: 10 },
                VdafInstance::Prio3Count,
                Role::Leader,
            )
            .build(),
        );
    }

    #[test]
    fn deserialize_docs_sample_tasks() {
        serde_yaml::from_str::<Vec<Task>>(include_str!("../../docs/samples/tasks.yaml")).unwrap();
    }

    #[test]
    fn collector_auth_tokens() {
        // As leader, we receive an error if no collector auth token is specified.
        Task::new(
            random(),
            Vec::from([
                "http://leader_endpoint".parse().unwrap(),
                "http://helper_endpoint".parse().unwrap(),
            ]),
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
            Vec::from([SecretBytes::new([0; PRIO3_VERIFY_KEY_LENGTH].into())]),
            0,
            None,
            None,
            0,
            Duration::from_hours(8).unwrap(),
            Duration::from_minutes(10).unwrap(),
            generate_test_hpke_config_and_private_key().config().clone(),
            Vec::from([random()]),
            Vec::new(),
            Vec::from([generate_test_hpke_config_and_private_key()]),
        )
        .unwrap_err();

        // As leader, we receive no error if a collector auth token is specified.
        Task::new(
            random(),
            Vec::from([
                "http://leader_endpoint".parse().unwrap(),
                "http://helper_endpoint".parse().unwrap(),
            ]),
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
            Vec::from([SecretBytes::new([0; PRIO3_VERIFY_KEY_LENGTH].into())]),
            0,
            None,
            None,
            0,
            Duration::from_hours(8).unwrap(),
            Duration::from_minutes(10).unwrap(),
            generate_test_hpke_config_and_private_key().config().clone(),
            Vec::from([random()]),
            Vec::from([random()]),
            Vec::from([generate_test_hpke_config_and_private_key()]),
        )
        .unwrap();

        // As helper, we receive no error if no collector auth token is specified.
        Task::new(
            random(),
            Vec::from([
                "http://leader_endpoint".parse().unwrap(),
                "http://helper_endpoint".parse().unwrap(),
            ]),
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Helper,
            Vec::from([SecretBytes::new([0; PRIO3_VERIFY_KEY_LENGTH].into())]),
            0,
            None,
            None,
            0,
            Duration::from_hours(8).unwrap(),
            Duration::from_minutes(10).unwrap(),
            generate_test_hpke_config_and_private_key().config().clone(),
            Vec::from([random()]),
            Vec::new(),
            Vec::from([generate_test_hpke_config_and_private_key()]),
        )
        .unwrap();

        // As helper, we receive an error if a collector auth token is specified.
        Task::new(
            random(),
            Vec::from([
                "http://leader_endpoint".parse().unwrap(),
                "http://helper_endpoint".parse().unwrap(),
            ]),
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Helper,
            Vec::from([SecretBytes::new([0; PRIO3_VERIFY_KEY_LENGTH].into())]),
            0,
            None,
            None,
            0,
            Duration::from_hours(8).unwrap(),
            Duration::from_minutes(10).unwrap(),
            generate_test_hpke_config_and_private_key().config().clone(),
            Vec::from([random()]),
            Vec::from([random()]),
            Vec::from([generate_test_hpke_config_and_private_key()]),
        )
        .unwrap_err();
    }

    #[test]
    fn aggregator_endpoints_end_in_slash() {
        let task = Task::new(
            random(),
            Vec::from([
                "http://leader_endpoint/foo/bar".parse().unwrap(),
                "http://helper_endpoint".parse().unwrap(),
            ]),
            QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
            Vec::from([SecretBytes::new([0; PRIO3_VERIFY_KEY_LENGTH].into())]),
            0,
            None,
            None,
            0,
            Duration::from_hours(8).unwrap(),
            Duration::from_minutes(10).unwrap(),
            generate_test_hpke_config_and_private_key().config().clone(),
            Vec::from([random()]),
            Vec::from([random()]),
            Vec::from([generate_test_hpke_config_and_private_key()]),
        )
        .unwrap();

        assert_eq!(
            task.aggregator_endpoints,
            Vec::from([
                "http://leader_endpoint/foo/bar/".parse().unwrap(),
                "http://helper_endpoint/".parse().unwrap()
            ])
        );
    }

    #[test]
    fn aggregator_request_paths() {
        for (prefix, task) in [
            (
                "",
                TaskBuilder::new(
                    QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Prio3Count,
                    Role::Leader,
                )
                .build(),
            ),
            (
                "/prefix",
                TaskBuilder::new(
                    QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Prio3Count,
                    Role::Leader,
                )
                .with_aggregator_endpoints(Vec::from([
                    Url::parse("https://leader.com/prefix/").unwrap(),
                    Url::parse("https://helper.com/prefix/").unwrap(),
                ]))
                .build(),
            ),
        ] {
            let prefix = format!("{prefix}/tasks");

            for uri in [
                task.report_upload_uri().unwrap(),
                task.aggregation_job_uri(&random()).unwrap(),
                task.collection_job_uri(&random()).unwrap(),
                task.aggregate_shares_uri().unwrap(),
            ] {
                // Check that path starts with / so it is suitable for use with mockito and that any
                // path components in the aggregator endpoint are still present.
                assert!(
                    uri.path().starts_with(&prefix),
                    "request path {} lacks prefix {prefix}",
                    uri.path()
                );
            }
        }
    }

    #[test]
    fn task_serde() {
        assert_tokens(
            &Task::new(
                TaskId::from([255; 32]),
                Vec::from([
                    "https://example.com/".parse().unwrap(),
                    "https://example.net/".parse().unwrap(),
                ]),
                QueryType::FixedSize { max_batch_size: 10 },
                VdafInstance::Prio3CountVec { length: 8 },
                Role::Helper,
                Vec::from([SecretBytes::new(b"1234567812345678".to_vec())]),
                1,
                None,
                Some(Duration::from_seconds(1800)),
                10,
                Duration::from_seconds(3600),
                Duration::from_seconds(60),
                HpkeConfig::new(
                    HpkeConfigId::from(8),
                    HpkeKemId::X25519HkdfSha256,
                    HpkeKdfId::HkdfSha256,
                    HpkeAeadId::Aes128Gcm,
                    HpkePublicKey::from(b"collector hpke public key".to_vec()),
                ),
                Vec::from([AuthenticationToken::Bearer(b"aggregator token".to_vec())]),
                Vec::new(),
                [HpkeKeypair::new(
                    HpkeConfig::new(
                        HpkeConfigId::from(255),
                        HpkeKemId::X25519HkdfSha256,
                        HpkeKdfId::HkdfSha256,
                        HpkeAeadId::Aes128Gcm,
                        HpkePublicKey::from(b"aggregator hpke public key".to_vec()),
                    ),
                    HpkePrivateKey::new(b"aggregator hpke private key".to_vec()),
                )],
            )
            .unwrap(),
            &[
                Token::Struct {
                    name: "SerializedTask",
                    len: 16,
                },
                Token::Str("task_id"),
                Token::Some,
                Token::Str("__________________________________________8"),
                Token::Str("aggregator_endpoints"),
                Token::Seq { len: Some(2) },
                Token::Str("https://example.com/"),
                Token::Str("https://example.net/"),
                Token::SeqEnd,
                Token::Str("query_type"),
                Token::StructVariant {
                    name: "QueryType",
                    variant: "FixedSize",
                    len: 1,
                },
                Token::Str("max_batch_size"),
                Token::U64(10),
                Token::StructVariantEnd,
                Token::Str("vdaf"),
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3CountVec",
                    len: 1,
                },
                Token::Str("length"),
                Token::U64(8),
                Token::StructVariantEnd,
                Token::Str("role"),
                Token::UnitVariant {
                    name: "Role",
                    variant: "Helper",
                },
                Token::Str("vdaf_verify_keys"),
                Token::Seq { len: Some(1) },
                Token::Str("MTIzNDU2NzgxMjM0NTY3OA"),
                Token::SeqEnd,
                Token::Str("max_batch_query_count"),
                Token::U64(1),
                Token::Str("task_expiration"),
                Token::None,
                Token::Str("report_expiry_age"),
                Token::Some,
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(1800),
                Token::Str("min_batch_size"),
                Token::U64(10),
                Token::Str("time_precision"),
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(3600),
                Token::Str("tolerable_clock_skew"),
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(60),
                Token::Str("collector_hpke_config"),
                Token::Struct {
                    name: "HpkeConfig",
                    len: 5,
                },
                Token::Str("id"),
                Token::NewtypeStruct {
                    name: "HpkeConfigId",
                },
                Token::U8(8),
                Token::Str("kem_id"),
                Token::UnitVariant {
                    name: "HpkeKemId",
                    variant: "X25519HkdfSha256",
                },
                Token::Str("kdf_id"),
                Token::UnitVariant {
                    name: "HpkeKdfId",
                    variant: "HkdfSha256",
                },
                Token::Str("aead_id"),
                Token::UnitVariant {
                    name: "HpkeAeadId",
                    variant: "Aes128Gcm",
                },
                Token::Str("public_key"),
                Token::Str("Y29sbGVjdG9yIGhwa2UgcHVibGljIGtleQ"),
                Token::StructEnd,
                Token::Str("aggregator_auth_tokens"),
                Token::Seq { len: Some(1) },
                Token::Struct {
                    name: "AuthenticationToken",
                    len: 2,
                },
                Token::Str("type"),
                Token::Str("Bearer"),
                Token::Str("token"),
                Token::Str("YWdncmVnYXRvciB0b2tlbg=="),
                Token::StructEnd,
                Token::SeqEnd,
                Token::Str("collector_auth_tokens"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                Token::Str("hpke_keys"),
                Token::Seq { len: Some(1) },
                Token::Struct {
                    name: "HpkeKeypair",
                    len: 2,
                },
                Token::Str("config"),
                Token::Struct {
                    name: "HpkeConfig",
                    len: 5,
                },
                Token::Str("id"),
                Token::NewtypeStruct {
                    name: "HpkeConfigId",
                },
                Token::U8(255),
                Token::Str("kem_id"),
                Token::UnitVariant {
                    name: "HpkeKemId",
                    variant: "X25519HkdfSha256",
                },
                Token::Str("kdf_id"),
                Token::UnitVariant {
                    name: "HpkeKdfId",
                    variant: "HkdfSha256",
                },
                Token::Str("aead_id"),
                Token::UnitVariant {
                    name: "HpkeAeadId",
                    variant: "Aes128Gcm",
                },
                Token::Str("public_key"),
                Token::Str("YWdncmVnYXRvciBocGtlIHB1YmxpYyBrZXk"),
                Token::StructEnd,
                Token::Str("private_key"),
                Token::Str("YWdncmVnYXRvciBocGtlIHByaXZhdGUga2V5"),
                Token::StructEnd,
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }
}
