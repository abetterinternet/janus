//! Shared parameters for a DAP task.

use crate::SecretBytes;
use anyhow::anyhow;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use educe::Educe;
use janus_core::{
    auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
    time::TimeExt,
    vdaf::VdafInstance,
};
use janus_messages::{
    batch_mode, AggregationJobId, AggregationJobStep, Duration, HpkeConfig, Role, TaskId, Time,
};
use postgres_types::{FromSql, ToSql};
use rand::{distributions::Standard, random, thread_rng, Rng};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use std::{array::TryFromSliceError, str::FromStr};
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

/// Identifiers for batch modes used by a task, along with batch mode-specific configuration.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BatchMode {
    /// Time-interval: used to support a collection style based on fixed time intervals.
    TimeInterval,

    /// Leader-selected: used to support collection of batches as quickly as possible, without the
    /// latency of waiting for batch time intervals to pass, and with direct control over the number
    /// of reports per batch.
    LeaderSelected {
        /// If present, reports will be separated into different batches by timestamp, such that
        /// the client timestamp interval duration will not exceed this value. The minimum and
        /// maximum allowed report timestamps for each batch will be multiples of this value as
        /// well. This must be a multiple of the task's time precision.
        ///
        /// This is an implementation-specific configuration parameter, and not part of the query
        /// type as defined in DAP.
        batch_time_window_size: Option<Duration>,
    },
}

impl TryFrom<batch_mode::Code> for BatchMode {
    type Error = Error;

    fn try_from(value: batch_mode::Code) -> Result<Self, Self::Error> {
        match value {
            batch_mode::Code::TimeInterval => Ok(Self::TimeInterval),
            batch_mode::Code::LeaderSelected => Ok(Self::LeaderSelected {
                batch_time_window_size: None,
            }),
            _ => Err(Error::InvalidParameter("unknown batch mode")),
        }
    }
}

/// A verification key for a VDAF, with a fixed length. It must be kept secret from clients to
/// maintain robustness, and it must be shared between aggregators.
#[derive(Educe, Clone, Copy)]
#[educe(Debug)]
pub struct VerifyKey<const SEED_SIZE: usize>(#[educe(Debug(ignore))] [u8; SEED_SIZE]);

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

/// Task parameters common to all views of a DAP task.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommonTaskParameters {
    /// Unique identifier for the task.
    task_id: TaskId,
    /// The batch mode this task uses to generate batches.
    batch_mode: BatchMode,
    /// The VDAF this task executes.
    vdaf: VdafInstance,
    /// Secret verification key shared by the aggregators.
    vdaf_verify_key: SecretBytes,
    /// The time before which the task is considered invalid.
    task_start: Option<Time>,
    /// The time after which the task is considered invalid.
    task_end: Option<Time>,
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
    /// The `task_info` byte string from a Taskprov `TaskConfig` struct. This is only present for
    /// tasks created via Taskprov.
    ///
    /// This field is used to distinguish tasks with otherwise equivalent DAP task parameters.
    taskprov_task_info: Option<Vec<u8>>,
}

impl CommonTaskParameters {
    /// Create a new [`CommonTaskParameters`] with the provided values.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_id: TaskId,
        batch_mode: BatchMode,
        vdaf: VdafInstance,
        vdaf_verify_key: SecretBytes,
        task_start: Option<Time>,
        task_end: Option<Time>,
        report_expiry_age: Option<Duration>,
        min_batch_size: u64,
        time_precision: Duration,
        tolerable_clock_skew: Duration,
    ) -> Result<Self, Error> {
        if min_batch_size == 0 {
            return Err(Error::InvalidParameter("min_batch_size"));
        }

        if let BatchMode::LeaderSelected {
            batch_time_window_size: Some(batch_time_window_size),
        } = batch_mode
        {
            if batch_time_window_size.as_seconds() == 0 {
                return Err(Error::InvalidParameter("batch_time_window_size is zero"));
            }
        }

        // These fields are stored as 64-bit signed integers in the database but are held in
        // memory as unsigned. Reject values that are too large. (perhaps these should be
        // represented by different types?)
        if let Some(report_expiry_age) = report_expiry_age {
            if report_expiry_age > Duration::from_seconds(i64::MAX as u64) {
                return Err(Error::InvalidParameter("report_expiry_age too large"));
            }
        }
        if let Some(task_start) = task_start {
            task_start
                .as_naive_date_time()
                .map_err(|_| Error::InvalidParameter("task_start out of range"))?;
        }
        if let Some(task_end) = task_end {
            task_end
                .as_naive_date_time()
                .map_err(|_| Error::InvalidParameter("task_end out of range"))?;
        }
        if let (Some(task_start), Some(task_end)) = (task_start, task_end) {
            if task_end < task_start {
                return Err(Error::InvalidParameter("task_end before task_start"));
            }
        }

        if time_precision.as_seconds() == 0 {
            return Err(Error::InvalidParameter("time_precision is zero"));
        }

        Ok(Self {
            task_id,
            batch_mode,
            vdaf,
            vdaf_verify_key,
            task_start,
            task_end,
            report_expiry_age,
            min_batch_size,
            time_precision,
            tolerable_clock_skew,
            taskprov_task_info: None,
        })
    }

    /// Returns the [`VerifyKey`] used by this aggregator to prepare report shares with other
    /// aggregators.
    ///
    /// # Errors
    ///
    /// If the verify key is not the correct length as required by the VDAF, an error will be
    /// returned.
    pub fn vdaf_verify_key<const SEED_SIZE: usize>(&self) -> Result<VerifyKey<SEED_SIZE>, Error> {
        VerifyKey::try_from(&self.vdaf_verify_key).map_err(|_| Error::AggregatorVerifyKeySize)
    }
}

/// An aggregator's view of the task's parameters.
#[derive(Clone, Educe, PartialEq, Eq)]
#[educe(Debug)]
pub struct AggregatorTask {
    /// Common task parameters
    common_parameters: CommonTaskParameters,
    /// URL relative to which the peer aggregator's API endpoints are found.
    #[educe(Debug(method(std::fmt::Display::fmt)))]
    peer_aggregator_endpoint: Url,
    /// Parameters specific to either aggregator role
    aggregator_parameters: AggregatorTaskParameters,
}

impl AggregatorTask {
    /// Create a new [`AggregatorTask`] with the provided values.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_id: TaskId,
        peer_aggregator_endpoint: Url,
        batch_mode: BatchMode,
        vdaf: VdafInstance,
        vdaf_verify_key: SecretBytes,
        task_start: Option<Time>,
        task_end: Option<Time>,
        report_expiry_age: Option<Duration>,
        min_batch_size: u64,
        time_precision: Duration,
        tolerable_clock_skew: Duration,
        aggregator_parameters: AggregatorTaskParameters,
    ) -> Result<Self, Error> {
        let common_parameters = CommonTaskParameters::new(
            task_id,
            batch_mode,
            vdaf,
            vdaf_verify_key,
            task_start,
            task_end,
            report_expiry_age,
            min_batch_size,
            time_precision,
            tolerable_clock_skew,
        )?;
        Self::new_with_common_parameters(
            common_parameters,
            peer_aggregator_endpoint,
            aggregator_parameters,
        )
    }

    fn new_with_common_parameters(
        common_parameters: CommonTaskParameters,
        peer_aggregator_endpoint: Url,
        aggregator_parameters: AggregatorTaskParameters,
    ) -> Result<Self, Error> {
        if let BatchMode::LeaderSelected {
            batch_time_window_size: Some(batch_time_window_size),
            ..
        } = common_parameters.batch_mode
        {
            if matches!(
                aggregator_parameters,
                AggregatorTaskParameters::TaskprovHelper { .. },
            ) {
                return Err(Error::InvalidParameter(
                    "batch_time_window_size is not supported for taskprov",
                ));
            } else if batch_time_window_size.as_seconds()
                % common_parameters.time_precision.as_seconds()
                != 0
            {
                return Err(Error::InvalidParameter("batch_time_window_size"));
            }
        }

        Ok(Self {
            common_parameters,
            peer_aggregator_endpoint,
            aggregator_parameters,
        })
    }

    /// Retrieves the task ID associated with this task.
    pub fn id(&self) -> &TaskId {
        &self.common_parameters.task_id
    }

    /// Retrieves the DAP role played by this aggregator.
    pub fn role(&self) -> &Role {
        self.aggregator_parameters.role()
    }

    /// Retrieves the aggregation mode of the task for the Helper, or None for the Leader.
    pub fn aggregation_mode(&self) -> Option<&AggregationMode> {
        self.aggregator_parameters().aggregation_mode()
    }

    /// Retrieves the peer aggregator endpoint associated with this task.
    pub fn peer_aggregator_endpoint(&self) -> &Url {
        &self.peer_aggregator_endpoint
    }

    /// Retrieves the batch mode associated with this task.
    pub fn batch_mode(&self) -> &BatchMode {
        &self.common_parameters.batch_mode
    }

    /// Retrieves the VDAF associated with this task.
    pub fn vdaf(&self) -> &VdafInstance {
        &self.common_parameters.vdaf
    }

    /// Retrieves the VDAF verification key associated with this task, as opaque secret bytes.
    pub fn opaque_vdaf_verify_key(&self) -> &SecretBytes {
        &self.common_parameters.vdaf_verify_key
    }

    /// Retrieves the task start time associated with this task.
    pub fn task_start(&self) -> Option<&Time> {
        self.common_parameters.task_start.as_ref()
    }

    /// Retrieves the task end time associated with this task.
    pub fn task_end(&self) -> Option<&Time> {
        self.common_parameters.task_end.as_ref()
    }

    /// Retrieves the report expiry age associated with this task.
    pub fn report_expiry_age(&self) -> Option<&Duration> {
        self.common_parameters.report_expiry_age.as_ref()
    }

    /// Retrieves the min batch size parameter associated with this task.
    pub fn min_batch_size(&self) -> u64 {
        self.common_parameters.min_batch_size
    }

    /// Retrieves the time precision parameter associated with this task.
    pub fn time_precision(&self) -> &Duration {
        &self.common_parameters.time_precision
    }

    /// Retrieves the tolerable clock skew parameter associated with this task.
    pub fn tolerable_clock_skew(&self) -> &Duration {
        &self.common_parameters.tolerable_clock_skew
    }

    /// Returns true if the `batch_size` is valid given this task's batch mode and batch size
    /// parameters, per
    /// <https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-09#name-batch-validation>
    pub fn validate_batch_size(&self, batch_size: u64) -> bool {
        batch_size >= self.common_parameters.min_batch_size
    }

    /// Returns the [`VerifyKey`] used by this aggregator to prepare report shares with other
    /// aggregators.
    ///
    /// # Errors
    ///
    /// If the verify key is not the correct length as required by the VDAF, an error will be
    /// returned.
    pub fn vdaf_verify_key<const SEED_SIZE: usize>(&self) -> Result<VerifyKey<SEED_SIZE>, Error> {
        self.common_parameters.vdaf_verify_key()
    }

    /// Returns the [`AggregatorTaskParameters`] for this task.
    pub fn aggregator_parameters(&self) -> &AggregatorTaskParameters {
        &self.aggregator_parameters
    }

    /// Returns the relative path for tasks, relative to which other API endpoints are defined.
    fn tasks_path(&self) -> String {
        format!("tasks/{}", self.id())
    }

    /// Returns the URI at which the helper resource for the specified aggregation job ID can be
    /// accessed.
    pub fn aggregation_job_uri(
        &self,
        aggregation_job_id: &AggregationJobId,
        step: Option<AggregationJobStep>,
    ) -> Result<Option<Url>, Error> {
        if matches!(
            self.aggregator_parameters,
            AggregatorTaskParameters::Leader { .. }
        ) {
            let mut uri = self.peer_aggregator_endpoint().join(&format!(
                "{}/aggregation_jobs/{aggregation_job_id}",
                self.tasks_path()
            ))?;

            if let Some(step) = step {
                uri.query_pairs_mut()
                    .append_pair("step", &u16::from(step).to_string());
            }

            Ok(Some(uri))
        } else {
            Ok(None)
        }
    }

    /// Returns the URI at which the helper aggregate shares resource can be accessed.
    pub fn aggregate_shares_uri(&self) -> Result<Option<Url>, Error> {
        if matches!(
            self.aggregator_parameters,
            AggregatorTaskParameters::Leader { .. }
        ) {
            Ok(Some(self.peer_aggregator_endpoint().join(&format!(
                "{}/aggregate_shares",
                self.tasks_path()
            ))?))
        } else {
            Ok(None)
        }
    }

    /// Returns the aggregator [`AuthenticationToken`] for this task, used by the leader to
    /// authenticate aggregation sub-protocol requests sent to the helper, or `None` for the helper.
    pub fn aggregator_auth_token(&self) -> Option<&AuthenticationToken> {
        self.aggregator_parameters.aggregator_auth_token()
    }

    /// Returns the aggregator [`AuthenticationTokenHash`] for this task, used by the helper to
    /// authenticate aggregation sub-protocol requests received from the leader, or `None` for the
    /// leader.
    pub fn aggregator_auth_token_hash(&self) -> Option<&AuthenticationTokenHash> {
        self.aggregator_parameters.aggregator_auth_token_hash()
    }

    /// Returns the collector HPKE configuration for this task, or `None` for taskprov tasks.
    pub fn collector_hpke_config(&self) -> Option<&HpkeConfig> {
        self.aggregator_parameters.collector_hpke_config()
    }

    /// Returns the collector [`AuthenticationTokenHash`] for this task, used by the leader to
    /// authenticate collection sub-protocol requests received from the collector, or `None` for the
    /// helper.
    pub fn collector_auth_token_hash(&self) -> Option<&AuthenticationTokenHash> {
        self.aggregator_parameters.collector_auth_token_hash()
    }

    /// Checks if the given aggregator authentication token is valid (i.e. matches with the
    /// authentication token recognized by this task).
    pub fn check_aggregator_auth_token(
        &self,
        incoming_auth_token: Option<&AuthenticationToken>,
    ) -> bool {
        self.aggregator_auth_token_hash()
            .zip(incoming_auth_token)
            .map(|(own_token_hash, incoming_token)| own_token_hash.validate(incoming_token))
            .unwrap_or(false)
    }

    /// Checks if the given collector authentication token is valid (i.e. matches with the
    /// authentication token recognized by this task).
    pub fn check_collector_auth_token(
        &self,
        incoming_auth_token: Option<&AuthenticationToken>,
    ) -> bool {
        self.collector_auth_token_hash()
            .zip(incoming_auth_token)
            .map(|(own_token_hash, incoming_token)| own_token_hash.validate(incoming_token))
            .unwrap_or(false)
    }

    /// Set the Taskprov `task_info` field for this task.
    pub fn with_taskprov_task_info(mut self, taskprov_task_info: Vec<u8>) -> Self {
        self.common_parameters.taskprov_task_info = Some(taskprov_task_info);
        self
    }

    /// Return the Taskprov `task_info` field for this task.
    pub fn taskprov_task_info(&self) -> Option<&[u8]> {
        self.common_parameters.taskprov_task_info.as_deref()
    }
}

/// Role-specific task parameters for the aggregator DAP roles.
#[derive(Clone, Educe, PartialEq, Eq)]
#[educe(Debug)]
pub enum AggregatorTaskParameters {
    /// Task parameters held exclusively by the DAP leader.
    Leader {
        /// Authentication token used to make requests to the helper during the aggregation
        /// sub-protocol.
        aggregator_auth_token: AuthenticationToken,
        /// Authentication token hash used to validate requests from the collector during the
        /// collection sub-protocol.
        collector_auth_token_hash: AuthenticationTokenHash,
        /// HPKE configuration for the collector.
        collector_hpke_config: HpkeConfig,
    },

    /// Task parameters held exclusively by the DAP helper.
    Helper {
        /// Authentication token hash used to validate requests from the leader during the
        /// aggregation sub-protocol.
        aggregator_auth_token_hash: AuthenticationTokenHash,
        /// HPKE configuration for the collector.
        collector_hpke_config: HpkeConfig,
        /// The aggregation mode to use for this task.
        aggregation_mode: AggregationMode,
    },

    /// Task parameters held exclusively by a DAP helper provisioned via taskprov.
    TaskprovHelper { aggregation_mode: AggregationMode },
}

/// Indicates an aggregation mode: synchronous or asynchronous.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSql, FromSql)]
#[postgres(name = "aggregation_mode")]
pub enum AggregationMode {
    /// Aggregation is completed synchronously, i.e. every successful aggregation initialization or
    /// continuation request will be responded to with a response in the "finished" status.
    #[postgres(name = "SYNCHRONOUS")]
    Synchronous,

    /// Aggregation is completed asynchronously, i.e. every successful aggregation initialization or
    /// continuation request will be responded to with a response in the "processing" status.
    #[postgres(name = "ASYNCHRONOUS")]
    Asynchronous,
}

impl FromStr for AggregationMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "synchronous" => Ok(Self::Synchronous),
            "asynchronous" => Ok(Self::Asynchronous),
            _ => Err(anyhow!("couldn't parse AggregationMode value: {s}")),
        }
    }
}

impl AggregatorTaskParameters {
    /// Returns the [`Role`] that this aggregator plays.
    pub fn role(&self) -> &Role {
        match self {
            Self::Leader { .. } => &Role::Leader,
            Self::Helper { .. } | Self::TaskprovHelper { .. } => &Role::Helper,
        }
    }

    /// Returns the [`AggregationMode`] for this task for the helper, or `None` for the leader.
    fn aggregation_mode(&self) -> Option<&AggregationMode> {
        match self {
            Self::Leader { .. } => None,
            Self::Helper {
                aggregation_mode, ..
            } => Some(aggregation_mode),
            Self::TaskprovHelper {
                aggregation_mode, ..
            } => Some(aggregation_mode),
        }
    }

    /// Returns the aggregator [`AuthenticationToken`] for this task, used by the leader to
    /// authenticate aggregation sub-protocol requests sent to the helper, or `None` for the helper.
    fn aggregator_auth_token(&self) -> Option<&AuthenticationToken> {
        match self {
            Self::Leader {
                aggregator_auth_token,
                ..
            } => Some(aggregator_auth_token),
            _ => None,
        }
    }

    /// Returns the aggregator [`AuthenticationTokenHash`] for this task, used by the helper to
    /// authenticate aggregation sub-protocol requests received from the leader, or `None` for the
    /// leader.
    fn aggregator_auth_token_hash(&self) -> Option<&AuthenticationTokenHash> {
        match self {
            Self::Helper {
                aggregator_auth_token_hash,
                ..
            } => Some(aggregator_auth_token_hash),
            _ => None,
        }
    }

    /// Returns the collector HPKE configuration for this task, or `None` for taskprov tasks.
    fn collector_hpke_config(&self) -> Option<&HpkeConfig> {
        match self {
            Self::Leader {
                collector_hpke_config,
                ..
            }
            | Self::Helper {
                collector_hpke_config,
                ..
            } => Some(collector_hpke_config),
            _ => None,
        }
    }

    /// Returns the collector [`AuthenticationTokenHash`] for this task, used by the leader to
    /// authenticate collection sub-protocol requests received from the collector, or `None` for the
    /// helper.
    fn collector_auth_token_hash(&self) -> Option<&AuthenticationTokenHash> {
        match self {
            Self::Leader {
                collector_auth_token_hash,
                ..
            } => Some(collector_auth_token_hash),
            _ => None,
        }
    }
}

/// SerializedAggregatorTask is an intermediate representation for the aggregator view of tasks
/// being serialized via the Serialize and Deserialize traits.
#[derive(Clone, Serialize, Deserialize)]
pub struct SerializedAggregatorTask {
    task_id: Option<TaskId>,
    peer_aggregator_endpoint: Url,
    batch_mode: BatchMode,
    aggregation_mode: Option<AggregationMode>,
    vdaf: VdafInstance,
    role: Role,
    vdaf_verify_key: Option<String>, // in unpadded base64url
    task_start: Option<Time>,
    task_end: Option<Time>,
    report_expiry_age: Option<Duration>,
    min_batch_size: u64,
    time_precision: Duration,
    tolerable_clock_skew: Duration,
    collector_hpke_config: HpkeConfig,
    aggregator_auth_token: Option<AuthenticationToken>,
    aggregator_auth_token_hash: Option<AuthenticationTokenHash>,
    collector_auth_token_hash: Option<AuthenticationTokenHash>,
}

impl SerializedAggregatorTask {
    /// Returns the task ID, if one is set.
    pub fn task_id(&self) -> Option<TaskId> {
        self.task_id
    }

    /// Randomly generates and fills values for the following fields if they are not set in the
    /// [`SerializedAggregatorTask`]
    ///
    /// - Task ID
    /// - VDAF verify key
    /// - Aggregator authentication token (only if the task's role is helper)
    /// - The aggregator's HPKE keypair (only one keypair is generated)
    pub fn generate_missing_fields(&mut self) {
        if self.task_id.is_none() {
            let task_id: TaskId = random();
            self.task_id = Some(task_id);
        }

        if self.vdaf_verify_key.is_none() {
            let vdaf_verify_key = SecretBytes::new(
                thread_rng()
                    .sample_iter(Standard)
                    .take(self.vdaf.verify_key_length())
                    .collect(),
            );

            self.vdaf_verify_key = Some(URL_SAFE_NO_PAD.encode(vdaf_verify_key.as_ref()));
        }

        if self.aggregator_auth_token.is_none() && self.role == Role::Helper {
            self.aggregator_auth_token = Some(random());
        }
    }
}

impl Serialize for AggregatorTask {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SerializedAggregatorTask {
            task_id: Some(*self.id()),
            peer_aggregator_endpoint: self.peer_aggregator_endpoint().clone(),
            batch_mode: *self.batch_mode(),
            aggregation_mode: self.aggregator_parameters.aggregation_mode().copied(),
            vdaf: self.vdaf().clone(),
            role: *self.role(),
            vdaf_verify_key: Some(URL_SAFE_NO_PAD.encode(self.opaque_vdaf_verify_key())),
            task_start: self.task_start().copied(),
            task_end: self.task_end().copied(),
            report_expiry_age: self.report_expiry_age().copied(),
            min_batch_size: self.min_batch_size(),
            time_precision: *self.time_precision(),
            tolerable_clock_skew: *self.tolerable_clock_skew(),
            collector_hpke_config: self
                .aggregator_parameters
                .collector_hpke_config()
                .expect("serializable tasks must have collector_hpke_config")
                .clone(),
            aggregator_auth_token: self.aggregator_parameters.aggregator_auth_token().cloned(),
            aggregator_auth_token_hash: self
                .aggregator_parameters
                .aggregator_auth_token_hash()
                .cloned(),
            collector_auth_token_hash: self
                .aggregator_parameters
                .collector_auth_token_hash()
                .cloned(),
        }
        .serialize(serializer)
    }
}

impl TryFrom<SerializedAggregatorTask> for AggregatorTask {
    type Error = Error;

    fn try_from(serialized_task: SerializedAggregatorTask) -> Result<Self, Self::Error> {
        // task_id
        let task_id = serialized_task
            .task_id
            .ok_or(Error::InvalidParameter("missing field task_id"))?;

        // vdaf_verify_key
        let vdaf_verify_key = serialized_task
            .vdaf_verify_key
            .ok_or(Error::InvalidParameter("missing vdaf_verify_key"))?;

        let aggregator_parameters = match serialized_task.role {
            Role::Leader => AggregatorTaskParameters::Leader {
                aggregator_auth_token: serialized_task
                    .aggregator_auth_token
                    .ok_or(Error::InvalidParameter("missing aggregator auth token"))?,
                collector_auth_token_hash: serialized_task
                    .collector_auth_token_hash
                    .ok_or(Error::InvalidParameter("missing collector auth token hash"))?,
                collector_hpke_config: serialized_task.collector_hpke_config,
            },
            Role::Helper => AggregatorTaskParameters::Helper {
                aggregator_auth_token_hash: serialized_task.aggregator_auth_token_hash.ok_or(
                    Error::InvalidParameter("missing aggregator auth token hash"),
                )?,
                collector_hpke_config: serialized_task.collector_hpke_config,
                aggregation_mode: serialized_task
                    .aggregation_mode
                    .ok_or(Error::InvalidParameter("missing aggregation mode"))?,
            },
            _ => return Err(Error::InvalidParameter("unexpected role")),
        };

        AggregatorTask::new(
            task_id,
            serialized_task.peer_aggregator_endpoint,
            serialized_task.batch_mode,
            serialized_task.vdaf,
            SecretBytes::new(URL_SAFE_NO_PAD.decode(vdaf_verify_key)?),
            serialized_task.task_start,
            serialized_task.task_end,
            serialized_task.report_expiry_age,
            serialized_task.min_batch_size,
            serialized_task.time_precision,
            serialized_task.tolerable_clock_skew,
            aggregator_parameters,
        )
    }
}

impl<'de> Deserialize<'de> for AggregatorTask {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Deserialize into intermediate representation.
        let serialized_task = SerializedAggregatorTask::deserialize(deserializer)?;
        AggregatorTask::try_from(serialized_task).map_err(D::Error::custom)
    }
}

#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util {
    use crate::{
        task::{
            AggregationMode, AggregatorTask, AggregatorTaskParameters, BatchMode,
            CommonTaskParameters, Error, VerifyKey,
        },
        SecretBytes,
    };
    use educe::Educe;
    use janus_core::{
        auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
        hpke::HpkeKeypair,
        time::DurationExt,
        url_ensure_trailing_slash,
        vdaf::VdafInstance,
    };
    use janus_messages::{
        AggregationJobId, AggregationJobStep, CollectionJobId, Duration, HpkeConfigId, Role,
        TaskId, Time,
    };
    use rand::{distributions::Standard, random, thread_rng, Rng};
    use std::collections::HashMap;
    use url::Url;

    /// All parameters and secrets for a task, for all participants.
    #[derive(Clone, Educe, PartialEq, Eq)]
    #[educe(Debug)]
    pub struct Task {
        /// Common task parameters
        common_parameters: CommonTaskParameters,
        /// URL relative to which the leader aggregator's API endpoints are found.
        #[educe(Debug(method(std::fmt::Display::fmt)))]
        leader_aggregator_endpoint: Url,
        /// URL relative to which the leader aggregator's API endpoints are found.
        #[educe(Debug(method(std::fmt::Display::fmt)))]
        helper_aggregator_endpoint: Url,
        /// The mode used for aggregation by the Helper (synchronous vs asynchronous).
        helper_aggregation_mode: AggregationMode,
        /// HPKE configuration and private key used by the collector to decrypt aggregate shares.
        collector_hpke_keypair: HpkeKeypair,
        /// Token used to authenticate messages exchanged between the aggregators in the aggregation
        /// sub-protocol.
        aggregator_auth_token: AuthenticationToken,
        /// Token used to authenticate messages exchanged between the collector and leader in the
        /// collection sub-protocol.
        collector_auth_token: AuthenticationToken,
        /// HPKE configurations & private keys used by the leader to decrypt client reports.
        leader_hpke_keys: HashMap<HpkeConfigId, HpkeKeypair>,
        /// HPKE configurations & private keys used by the helper to decrypt client reports.
        helper_hpke_keys: HashMap<HpkeConfigId, HpkeKeypair>,
    }

    impl Task {
        /// Create a new [`Task`] from the provided values.
        #[allow(clippy::too_many_arguments, dead_code)]
        pub(crate) fn new<I: IntoIterator<Item = HpkeKeypair>>(
            task_id: TaskId,
            leader_aggregator_endpoint: Url,
            helper_aggregator_endpoint: Url,
            batch_mode: BatchMode,
            helper_aggregation_mode: AggregationMode,
            vdaf: VdafInstance,
            vdaf_verify_key: SecretBytes,
            task_start: Option<Time>,
            task_end: Option<Time>,
            report_expiry_age: Option<Duration>,
            min_batch_size: u64,
            time_precision: Duration,
            tolerable_clock_skew: Duration,
            collector_hpke_keypair: HpkeKeypair,
            aggregator_auth_token: AuthenticationToken,
            collector_auth_token: AuthenticationToken,
            leader_hpke_keys: I,
            helper_hpke_keys: I,
        ) -> Self {
            // Compute hpke_configs mapping cfg.id -> (cfg, key).
            let leader_hpke_keys: HashMap<HpkeConfigId, HpkeKeypair> = leader_hpke_keys
                .into_iter()
                .map(|keypair| (*keypair.config().id(), keypair))
                .collect();

            let helper_hpke_keys: HashMap<HpkeConfigId, HpkeKeypair> = helper_hpke_keys
                .into_iter()
                .map(|keypair| (*keypair.config().id(), keypair))
                .collect();

            Self {
                common_parameters: CommonTaskParameters {
                    task_id,
                    batch_mode,
                    vdaf,
                    vdaf_verify_key,
                    task_start,
                    task_end,
                    report_expiry_age,
                    min_batch_size,
                    time_precision,
                    tolerable_clock_skew,
                    taskprov_task_info: None,
                },
                // Ensure provided aggregator endpoints end with a slash, as we will be joining
                // additional path segments into these endpoints & the Url::join implementation is
                // persnickety about the slash at the end of the path.
                leader_aggregator_endpoint: url_ensure_trailing_slash(leader_aggregator_endpoint),
                helper_aggregator_endpoint: url_ensure_trailing_slash(helper_aggregator_endpoint),
                helper_aggregation_mode,
                aggregator_auth_token,
                collector_auth_token,
                collector_hpke_keypair,
                leader_hpke_keys,
                helper_hpke_keys,
            }
        }

        /// Retrieves the task ID associated with this task.
        pub fn id(&self) -> &TaskId {
            &self.common_parameters.task_id
        }

        /// Retrieves the Leader's aggregator endpoint associated with this task.
        pub fn leader_aggregator_endpoint(&self) -> &Url {
            &self.leader_aggregator_endpoint
        }

        /// Retrieves the Helper's aggregator endpoint associated with this task.
        pub fn helper_aggregator_endpoint(&self) -> &Url {
            &self.helper_aggregator_endpoint
        }

        /// Retrieves the batch mode associated with this task.
        pub fn batch_mode(&self) -> &BatchMode {
            &self.common_parameters.batch_mode
        }

        pub fn helper_aggregation_mode(&self) -> &AggregationMode {
            &self.helper_aggregation_mode
        }

        /// Retrieves the VDAF associated with this task.
        pub fn vdaf(&self) -> &VdafInstance {
            &self.common_parameters.vdaf
        }

        /// Retrieves the VDAF verification key associated with this task, as opaque secret bytes.
        pub fn opaque_vdaf_verify_key(&self) -> &SecretBytes {
            &self.common_parameters.vdaf_verify_key
        }

        /// Retrieves the task start time associated with this task.
        pub fn task_start(&self) -> Option<&Time> {
            self.common_parameters.task_start.as_ref()
        }

        /// Retrieves the task end time associated with this task.
        pub fn task_end(&self) -> Option<&Time> {
            self.common_parameters.task_end.as_ref()
        }

        /// Retrieves the report expiry age associated with this task.
        pub fn report_expiry_age(&self) -> Option<&Duration> {
            self.common_parameters.report_expiry_age.as_ref()
        }

        /// Retrieves the min batch size parameter associated with this task.
        pub fn min_batch_size(&self) -> u64 {
            self.common_parameters.min_batch_size
        }

        /// Retrieves the time precision parameter associated with this task.
        pub fn time_precision(&self) -> &Duration {
            &self.common_parameters.time_precision
        }

        /// Retrieves the tolerable clock skew parameter associated with this task.
        pub fn tolerable_clock_skew(&self) -> &Duration {
            &self.common_parameters.tolerable_clock_skew
        }

        /// Retrieves the collector HPKE keypair associated with this task.
        pub fn collector_hpke_keypair(&self) -> &HpkeKeypair {
            &self.collector_hpke_keypair
        }

        /// Retrieves the aggregator authentication token associated with this task.
        pub fn aggregator_auth_token(&self) -> &AuthenticationToken {
            &self.aggregator_auth_token
        }

        /// Retrieves the collector authentication token associated with this task.
        pub fn collector_auth_token(&self) -> &AuthenticationToken {
            &self.collector_auth_token
        }

        /// Returns the [`VerifyKey`] used by this aggregator to prepare report shares with other
        /// aggregators.
        ///
        /// # Errors
        ///
        /// If the verify key is not the correct length as required by the VDAF, an error will be
        /// returned.
        pub fn vdaf_verify_key<const SEED_SIZE: usize>(
            &self,
        ) -> Result<VerifyKey<SEED_SIZE>, Error> {
            self.common_parameters.vdaf_verify_key()
        }

        /// Returns the relative path for tasks, relative to which other API endpoints are defined.
        fn tasks_path(&self) -> String {
            format!("tasks/{}", self.id())
        }

        /// Returns the URI at which reports may be uploaded for this task.
        pub fn report_upload_uri(&self) -> Result<Url, Error> {
            Ok(self
                .leader_aggregator_endpoint()
                .join(&format!("{}/reports", self.tasks_path()))?)
        }

        /// Returns the URI at which the helper resource for the specified aggregation job ID can be
        /// accessed.
        pub fn aggregation_job_uri(
            &self,
            aggregation_job_id: &AggregationJobId,
            step: Option<AggregationJobStep>,
        ) -> Result<Url, Error> {
            let mut uri = self.helper_aggregator_endpoint().join(&format!(
                "{}/aggregation_jobs/{aggregation_job_id}",
                self.tasks_path()
            ))?;

            if let Some(step) = step {
                uri.query_pairs_mut()
                    .append_pair("step", &u16::from(step).to_string());
            }

            Ok(uri)
        }

        /// Returns the URI at which the helper aggregate shares resource can be accessed.
        pub fn aggregate_shares_uri(&self) -> Result<Url, Error> {
            Ok(self
                .helper_aggregator_endpoint()
                .join(&format!("{}/aggregate_shares", self.tasks_path()))?)
        }

        /// Returns the URI at which the leader resource for the specified collection job ID can be
        /// accessed.
        pub fn collection_job_uri(
            &self,
            collection_job_id: &CollectionJobId,
        ) -> Result<Url, Error> {
            Ok(self.leader_aggregator_endpoint().join(&format!(
                "{}/collection_jobs/{collection_job_id}",
                self.tasks_path()
            ))?)
        }

        /// Render the leader aggregator's view of this task.
        pub fn leader_view(&self) -> Result<AggregatorTask, Error> {
            AggregatorTask::new_with_common_parameters(
                self.common_parameters.clone(),
                self.helper_aggregator_endpoint.clone(),
                AggregatorTaskParameters::Leader {
                    aggregator_auth_token: self.aggregator_auth_token.clone(),
                    collector_auth_token_hash: AuthenticationTokenHash::from(
                        &self.collector_auth_token,
                    ),
                    collector_hpke_config: self.collector_hpke_keypair.config().clone(),
                },
            )
        }

        /// Render the helper aggregator's view of this task.
        pub fn helper_view(&self) -> Result<AggregatorTask, Error> {
            AggregatorTask::new_with_common_parameters(
                self.common_parameters.clone(),
                self.leader_aggregator_endpoint.clone(),
                AggregatorTaskParameters::Helper {
                    aggregator_auth_token_hash: AuthenticationTokenHash::from(
                        &self.aggregator_auth_token,
                    ),
                    collector_hpke_config: self.collector_hpke_keypair.config().clone(),
                    aggregation_mode: self.helper_aggregation_mode,
                },
            )
        }

        /// Render a taskprov helper aggregator's view of this task.
        pub fn taskprov_helper_view(&self) -> Result<AggregatorTask, Error> {
            AggregatorTask::new_with_common_parameters(
                self.common_parameters.clone(),
                self.leader_aggregator_endpoint.clone(),
                AggregatorTaskParameters::TaskprovHelper {
                    aggregation_mode: self.helper_aggregation_mode,
                },
            )
        }

        /// Render the view of the specified aggregator of this task.
        ///
        /// # Errors
        ///
        /// Returns an error if `role` is not an aggregator role.
        pub fn view_for_role(&self, role: Role) -> Result<AggregatorTask, Error> {
            match role {
                Role::Leader => self.leader_view(),
                Role::Helper => self.helper_view(),
                _ => Err(Error::InvalidParameter("role is not an aggregator")),
            }
        }
    }

    /// TaskBuilder is a testing utility allowing tasks to be built based on a template.
    #[derive(Clone)]
    pub struct TaskBuilder(Task);

    impl TaskBuilder {
        /// Create a [`TaskBuilder`] from the provided values, with arbitrary values for the other
        /// task parameters. Defaults to using `AuthenticationToken::Bearer` for the aggregator and
        /// collector authentication tokens.
        pub fn new(
            batch_mode: BatchMode,
            helper_aggregation_mode: AggregationMode,
            vdaf: VdafInstance,
        ) -> Self {
            let task_id = random();

            let leader_hpke_keypairs = [
                HpkeKeypair::test(),
                HpkeKeypair::test_with_id(HpkeConfigId::from(1)),
            ];
            let helper_hpke_keypairs = [
                HpkeKeypair::test(),
                HpkeKeypair::test_with_id(HpkeConfigId::from(1)),
            ];

            let vdaf_verify_key = SecretBytes::new(
                thread_rng()
                    .sample_iter(Standard)
                    .take(vdaf.verify_key_length())
                    .collect(),
            );

            Self(Task::new(
                task_id,
                "https://leader.endpoint".parse().unwrap(),
                "https://helper.endpoint".parse().unwrap(),
                batch_mode,
                helper_aggregation_mode,
                vdaf,
                vdaf_verify_key,
                None,
                None,
                None,
                1,
                Duration::from_hours(8).unwrap(),
                Duration::from_minutes(10).unwrap(),
                /* Collector HPKE keypair */ HpkeKeypair::test(),
                /* Aggregator auth token */ random(),
                /* Collector auth token */ random(),
                leader_hpke_keypairs,
                helper_hpke_keypairs,
            ))
        }

        /// Gets the leader aggregator endpoint for the eventual task.
        pub fn leader_aggregator_endpoint(&self) -> &Url {
            self.0.leader_aggregator_endpoint()
        }

        /// Gets the helper aggregator endpoint for the eventual task.
        pub fn helper_aggregator_endpoint(&self) -> &Url {
            self.0.helper_aggregator_endpoint()
        }

        /// Gets the task ID for the eventual task
        pub fn task_id(&self) -> &TaskId {
            self.0.id()
        }

        /// Associates the eventual task with the given task ID.
        pub fn with_id(self, task_id: TaskId) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    task_id,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Gets the batch mode for the eventual task.
        pub fn batch_mode(&self) -> &BatchMode {
            self.0.batch_mode()
        }

        /// Gets the aggregation mode used by the helper for the eventual task.
        pub fn helper_aggregation_mode(&self) -> &AggregationMode {
            self.0.helper_aggregation_mode()
        }

        /// Gets the VDAF for the eventual task
        pub fn vdaf(&self) -> &VdafInstance {
            self.0.vdaf()
        }

        /// Associates the eventual task with the given aggregator endpoint for the Leader.
        pub fn with_leader_aggregator_endpoint(self, leader_aggregator_endpoint: Url) -> Self {
            Self(Task {
                leader_aggregator_endpoint: url_ensure_trailing_slash(leader_aggregator_endpoint),
                ..self.0
            })
        }

        /// Associates the eventual task with the given aggregator endpoint for the Helper.
        pub fn with_helper_aggregator_endpoint(self, helper_aggregator_endpoint: Url) -> Self {
            Self(Task {
                helper_aggregator_endpoint: url_ensure_trailing_slash(helper_aggregator_endpoint),
                ..self.0
            })
        }

        /// Associates the eventual task with the given VDAF verification key.
        pub fn with_vdaf_verify_key(self, vdaf_verify_key: SecretBytes) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    vdaf_verify_key,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Associates the eventual task with the given min batch size parameter.
        pub fn with_min_batch_size(self, min_batch_size: u64) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    min_batch_size,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Gets the minimum batch size associated with the eventual task.
        pub fn min_batch_size(&self) -> u64 {
            self.0.min_batch_size()
        }

        /// Associates the eventual task with the given time precision parameter.
        pub fn with_time_precision(self, time_precision: Duration) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    time_precision,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Gets the time precision associated with the eventual task.
        pub fn time_precision(&self) -> &Duration {
            self.0.time_precision()
        }

        /// Associates the eventual task with the given tolerable clock skew.
        pub fn with_tolerable_clock_skew(self, tolerable_clock_skew: Duration) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    tolerable_clock_skew,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Associates the eventual task with the given aggregator authentication token.
        pub fn with_aggregator_auth_token(
            self,
            aggregator_auth_token: AuthenticationToken,
        ) -> Self {
            Self(Task {
                aggregator_auth_token,
                ..self.0
            })
        }

        /// Associates the eventual task with a random [`AuthenticationToken::DapAuth`] aggregator
        /// auth token.
        pub fn with_dap_auth_aggregator_token(self) -> Self {
            Self(Task {
                aggregator_auth_token: AuthenticationToken::DapAuth(random()),
                ..self.0
            })
        }

        /// Associates the eventual task with the given collector authentication token.
        pub fn with_collector_auth_token(self, collector_auth_token: AuthenticationToken) -> Self {
            Self(Task {
                collector_auth_token,
                ..self.0
            })
        }

        /// Gets the collector auth token associated with the eventual task.
        pub fn collector_auth_token(&self) -> &AuthenticationToken {
            self.0.collector_auth_token()
        }

        /// Associates the eventual task with a random [`AuthenticationToken::DapAuth`] collector
        /// auth token.
        pub fn with_dap_auth_collector_token(self) -> Self {
            Self(Task {
                collector_auth_token: AuthenticationToken::DapAuth(random()),
                ..self.0
            })
        }

        /// Sets the task start time.
        pub fn with_task_start(self, task_start: Option<Time>) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    task_start,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Sets the task end time.
        pub fn with_task_end(self, task_end: Option<Time>) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    task_end,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Sets the report expiry age.
        pub fn with_report_expiry_age(self, report_expiry_age: Option<Duration>) -> Self {
            Self(Task {
                common_parameters: CommonTaskParameters {
                    report_expiry_age,
                    ..self.0.common_parameters
                },
                ..self.0
            })
        }

        /// Set the Taskprov `task_info` field for this task.
        pub fn with_taskprov_task_info(mut self, taskprov_task_info: Vec<u8>) -> Self {
            self.0.common_parameters.taskprov_task_info = Some(taskprov_task_info);
            self
        }

        /// Gets the colector HPKE keypair for the eventual task.
        pub fn collector_hpke_keypair(&self) -> &HpkeKeypair {
            self.0.collector_hpke_keypair()
        }

        /// Consumes this task builder & produces a [`Task`] with the given specifications.
        pub fn build(self) -> Task {
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
        task::{
            test_util::TaskBuilder, AggregationMode, AggregatorTask, AggregatorTaskParameters,
            BatchMode, VdafInstance,
        },
        SecretBytes,
    };
    use assert_matches::assert_matches;
    use janus_core::{
        auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
        test_util::roundtrip_encoding,
        time::DurationExt,
        vdaf::vdaf_dp_strategies,
    };
    use janus_messages::{
        Duration, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId, HpkeKemId, HpkePublicKey,
        TaskId, Time,
    };
    use rand::random;
    use serde_json::json;
    use serde_test::{assert_de_tokens, assert_tokens, Token};

    #[test]
    fn leader_task_serialization() {
        roundtrip_encoding(
            TaskBuilder::new(
                BatchMode::TimeInterval,
                AggregationMode::Synchronous,
                VdafInstance::Prio3Count,
            )
            .build()
            .leader_view()
            .unwrap(),
        );
    }

    #[test]
    fn helper_task_serialization() {
        roundtrip_encoding(
            TaskBuilder::new(
                BatchMode::TimeInterval,
                AggregationMode::Synchronous,
                VdafInstance::Prio3Count,
            )
            .build()
            .helper_view()
            .unwrap(),
        );
    }

    #[test]
    fn deserialize_docs_sample_tasks() {
        serde_yaml::from_str::<Vec<AggregatorTask>>(include_str!("../../docs/samples/tasks.yaml"))
            .unwrap();
    }

    #[test]
    fn aggregator_endpoints_end_in_slash() {
        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        )
        .with_leader_aggregator_endpoint("http://leader_endpoint/foo/bar".parse().unwrap())
        .with_helper_aggregator_endpoint("http://helper_endpoint".parse().unwrap())
        .build();

        assert_eq!(
            task.leader_aggregator_endpoint(),
            &"http://leader_endpoint/foo/bar/".parse().unwrap(),
        );
        assert_eq!(
            task.helper_aggregator_endpoint(),
            &"http://helper_endpoint/".parse().unwrap(),
        );
    }

    #[test]
    fn aggregator_request_paths() {
        for (prefix, task) in [
            (
                "",
                TaskBuilder::new(
                    BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Prio3Count,
                )
                .build(),
            ),
            (
                "/prefix",
                TaskBuilder::new(
                    BatchMode::TimeInterval,
                    AggregationMode::Synchronous,
                    VdafInstance::Prio3Count,
                )
                .with_leader_aggregator_endpoint("https://leader.com/prefix/".parse().unwrap())
                .with_helper_aggregator_endpoint("https://helper.com/prefix/".parse().unwrap())
                .build(),
            ),
        ] {
            let prefix = format!("{prefix}/tasks");

            for uri in [
                task.report_upload_uri().unwrap(),
                task.aggregation_job_uri(&random(), None).unwrap(),
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
    fn request_authentication() {
        let task = TaskBuilder::new(
            BatchMode::TimeInterval,
            AggregationMode::Synchronous,
            VdafInstance::Prio3Count,
        )
        .build();

        let leader_task = task.leader_view().unwrap();
        let helper_task = task.helper_view().unwrap();

        let incorrect_auth_token = random();

        // Helper should accept valid aggregator auth token
        assert!(helper_task.check_aggregator_auth_token(Some(task.aggregator_auth_token())));
        // Leader should accept valid collector auth token
        assert!(leader_task.check_collector_auth_token(Some(task.collector_auth_token())));

        // Leader should reject absent collector auth token
        assert!(!leader_task.check_collector_auth_token(None));
        // Helper should reject absent aggregator auth token
        assert!(!helper_task.check_aggregator_auth_token(None));
        // Leader should not be able to validate aggregation sub protocol requests
        assert!(!leader_task.check_aggregator_auth_token(Some(task.aggregator_auth_token())));
        // Helper should not be able to validate collection sub protocol requests
        assert!(!helper_task.check_collector_auth_token(Some(task.collector_auth_token())));
        // Incorrect collector token should be rejected by leader
        assert!(!leader_task.check_collector_auth_token(Some(&incorrect_auth_token)));
        // Incorrect aggregator token should be rejected by helper
        assert!(!helper_task.check_aggregator_auth_token(Some(&incorrect_auth_token)));
    }

    #[test]
    fn aggregator_task_serde() {
        assert_tokens(
            &AggregatorTask::new(
                TaskId::from([0; 32]),
                "https://example.net/".parse().unwrap(),
                BatchMode::TimeInterval,
                VdafInstance::Prio3Count,
                SecretBytes::new(b"1234567812345678".to_vec()),
                None,
                None,
                None,
                10,
                Duration::from_seconds(3600),
                Duration::from_seconds(60),
                AggregatorTaskParameters::Leader {
                    aggregator_auth_token: AuthenticationToken::new_dap_auth_token_from_string(
                        "YWdncmVnYXRvciB0b2tlbg",
                    )
                    .unwrap(),
                    collector_auth_token_hash: AuthenticationTokenHash::from(
                        &AuthenticationToken::new_bearer_token_from_string("Y29sbGVjdG9yIHRva2Vu")
                            .unwrap(),
                    ),
                    collector_hpke_config: HpkeConfig::new(
                        HpkeConfigId::from(8),
                        HpkeKemId::X25519HkdfSha256,
                        HpkeKdfId::HkdfSha256,
                        HpkeAeadId::Aes128Gcm,
                        HpkePublicKey::from(b"collector hpke public key".to_vec()),
                    ),
                },
            )
            .unwrap(),
            &[
                Token::Struct {
                    name: "SerializedAggregatorTask",
                    len: 17,
                },
                Token::Str("task_id"),
                Token::Some,
                Token::Str("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                Token::Str("peer_aggregator_endpoint"),
                Token::Str("https://example.net/"),
                Token::Str("batch_mode"),
                Token::UnitVariant {
                    name: "BatchMode",
                    variant: "TimeInterval",
                },
                Token::Str("aggregation_mode"),
                Token::None,
                Token::Str("vdaf"),
                Token::UnitVariant {
                    name: "VdafInstance",
                    variant: "Prio3Count",
                },
                Token::Str("role"),
                Token::UnitVariant {
                    name: "Role",
                    variant: "Leader",
                },
                Token::Str("vdaf_verify_key"),
                Token::Some,
                Token::Str("MTIzNDU2NzgxMjM0NTY3OA"),
                Token::Str("task_start"),
                Token::None,
                Token::Str("task_end"),
                Token::None,
                Token::Str("report_expiry_age"),
                Token::None,
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
                Token::Str("aggregator_auth_token"),
                Token::Some,
                Token::Struct {
                    name: "AuthenticationToken",
                    len: 2,
                },
                Token::Str("type"),
                Token::UnitVariant {
                    name: "AuthenticationToken",
                    variant: "DapAuth",
                },
                Token::Str("token"),
                Token::Str("YWdncmVnYXRvciB0b2tlbg"),
                Token::StructEnd,
                Token::Str("aggregator_auth_token_hash"),
                Token::None,
                Token::Str("collector_auth_token_hash"),
                Token::Some,
                Token::Struct {
                    name: "AuthenticationTokenHash",
                    len: 2,
                },
                Token::Str("type"),
                Token::UnitVariant {
                    name: "AuthenticationTokenHash",
                    variant: "Bearer",
                },
                Token::Str("hash"),
                Token::Str("LdjsTjGZXsaitZonqNIi2LcDLce3OLP6SeWv2eUx4rY"),
                Token::StructEnd,
                Token::StructEnd,
            ],
        );

        assert_tokens(
            &AggregatorTask::new(
                TaskId::from([255; 32]),
                "https://example.com/".parse().unwrap(),
                BatchMode::LeaderSelected {
                    batch_time_window_size: None,
                },
                VdafInstance::Prio3SumVec {
                    bits: 1,
                    length: 8,
                    chunk_length: 3,
                    dp_strategy: vdaf_dp_strategies::Prio3SumVec::NoDifferentialPrivacy,
                },
                SecretBytes::new(b"1234567812345678".to_vec()),
                Some(Time::from_seconds_since_epoch(1000)),
                Some(Time::from_seconds_since_epoch(2000)),
                Some(Duration::from_seconds(1800)),
                10,
                Duration::from_seconds(3600),
                Duration::from_seconds(60),
                AggregatorTaskParameters::Helper {
                    aggregator_auth_token_hash: AuthenticationTokenHash::from(
                        &AuthenticationToken::new_bearer_token_from_string(
                            "YWdncmVnYXRvciB0b2tlbg",
                        )
                        .unwrap(),
                    ),
                    collector_hpke_config: HpkeConfig::new(
                        HpkeConfigId::from(8),
                        HpkeKemId::X25519HkdfSha256,
                        HpkeKdfId::HkdfSha256,
                        HpkeAeadId::Aes128Gcm,
                        HpkePublicKey::from(b"collector hpke public key".to_vec()),
                    ),
                    aggregation_mode: AggregationMode::Synchronous,
                },
            )
            .unwrap(),
            &[
                Token::Struct {
                    name: "SerializedAggregatorTask",
                    len: 17,
                },
                Token::Str("task_id"),
                Token::Some,
                Token::Str("__________________________________________8"),
                Token::Str("peer_aggregator_endpoint"),
                Token::Str("https://example.com/"),
                Token::Str("batch_mode"),
                Token::StructVariant {
                    name: "BatchMode",
                    variant: "LeaderSelected",
                    len: 1,
                },
                Token::Str("batch_time_window_size"),
                Token::None,
                Token::StructVariantEnd,
                Token::Str("aggregation_mode"),
                Token::Some,
                Token::UnitVariant {
                    name: "AggregationMode",
                    variant: "Synchronous",
                },
                Token::Str("vdaf"),
                Token::StructVariant {
                    name: "VdafInstance",
                    variant: "Prio3SumVec",
                    len: 4,
                },
                Token::Str("bits"),
                Token::U64(1),
                Token::Str("length"),
                Token::U64(8),
                Token::Str("chunk_length"),
                Token::U64(3),
                Token::Str("dp_strategy"),
                Token::Struct {
                    name: "Prio3SumVec",
                    len: 1,
                },
                Token::Str("dp_strategy"),
                Token::Str("NoDifferentialPrivacy"),
                Token::StructEnd,
                Token::StructVariantEnd,
                Token::Str("role"),
                Token::UnitVariant {
                    name: "Role",
                    variant: "Helper",
                },
                Token::Str("vdaf_verify_key"),
                Token::Some,
                Token::Str("MTIzNDU2NzgxMjM0NTY3OA"),
                Token::Str("task_start"),
                Token::Some,
                Token::NewtypeStruct { name: "Time" },
                Token::U64(1000),
                Token::Str("task_end"),
                Token::Some,
                Token::NewtypeStruct { name: "Time" },
                Token::U64(2000),
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
                Token::Str("aggregator_auth_token"),
                Token::None,
                Token::Str("aggregator_auth_token_hash"),
                Token::Some,
                Token::Struct {
                    name: "AuthenticationTokenHash",
                    len: 2,
                },
                Token::Str("type"),
                Token::UnitVariant {
                    name: "AuthenticationTokenHash",
                    variant: "Bearer",
                },
                Token::Str("hash"),
                Token::Str("MJOoBO_ysLEuG_lv2C37eEOf1Ngetsr-Ers0ZYj4vdQ"),
                Token::StructEnd,
                Token::Str("collector_auth_token_hash"),
                Token::None,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn batch_mode_serde() {
        assert_tokens(
            &BatchMode::TimeInterval,
            &[Token::UnitVariant {
                name: "BatchMode",
                variant: "TimeInterval",
            }],
        );
        assert_tokens(
            &BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            &[
                Token::StructVariant {
                    name: "BatchMode",
                    variant: "LeaderSelected",
                    len: 1,
                },
                Token::Str("batch_time_window_size"),
                Token::None,
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &BatchMode::LeaderSelected {
                batch_time_window_size: Some(Duration::from_hours(1).unwrap()),
            },
            &[
                Token::StructVariant {
                    name: "BatchMode",
                    variant: "LeaderSelected",
                    len: 1,
                },
                Token::Str("batch_time_window_size"),
                Token::Some,
                Token::NewtypeStruct { name: "Duration" },
                Token::U64(3600),
                Token::StructVariantEnd,
            ],
        );
        assert_tokens(
            &BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            &[
                Token::StructVariant {
                    name: "BatchMode",
                    variant: "LeaderSelected",
                    len: 1,
                },
                Token::Str("batch_time_window_size"),
                Token::None,
                Token::StructVariantEnd,
            ],
        );

        // Backwards compatibility cases:
        assert_de_tokens(
            &BatchMode::LeaderSelected {
                batch_time_window_size: None,
            },
            &[
                Token::StructVariant {
                    name: "BatchMode",
                    variant: "LeaderSelected",
                    len: 1,
                },
                Token::StructVariantEnd,
            ],
        );
        assert_matches!(
            serde_json::from_value(json!({ "LeaderSelected": {} })),
            Ok(BatchMode::LeaderSelected {
                batch_time_window_size: None,
            })
        );
        assert_matches!(
            serde_yaml::from_str("!LeaderSelected {}"),
            Ok(BatchMode::LeaderSelected {
                batch_time_window_size: None,
            })
        );
        assert_matches!(
            serde_yaml::from_str(
                "---
!LeaderSelected
  batch_time_window_size: 3600"
            ),
            Ok(BatchMode::LeaderSelected {
                batch_time_window_size: Some(duration),
            }) => assert_eq!(duration, Duration::from_seconds(3600))
        );
    }
}
