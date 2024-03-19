use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use derivative::Derivative;
use janus_aggregator_core::{
    datastore::models::{GlobalHpkeKeypair, HpkeKeyState, TaskUploadCounter},
    task::{AggregatorTask, QueryType},
    taskprov::{PeerAggregator, VerifyKeyInit},
};
use janus_core::{
    auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
    vdaf::VdafInstance,
};
use janus_messages::{
    query_type::Code as SupportedQueryType, Duration, HpkeAeadId, HpkeConfig, HpkeKdfId, HpkeKemId,
    Role, TaskId, Time,
};
use serde::{Deserialize, Deserializer, Serialize};
use url::Url;

#[allow(dead_code)]
// ^^ allowed in order to fully describe the interface and for later use
#[derive(Serialize, PartialEq, Eq, Debug)]
pub(crate) enum AggregatorRole {
    Either,
    Leader,
    Helper,
}

#[derive(Serialize, PartialEq, Eq, Derivative)]
#[derivative(Debug)]
pub(crate) struct AggregatorApiConfig {
    pub protocol: &'static str,
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    pub dap_url: Url,
    pub role: AggregatorRole,
    pub vdafs: Vec<SupportedVdaf>,
    pub query_types: Vec<SupportedQueryType>,
    pub features: &'static [&'static str],
}

#[allow(clippy::enum_variant_names)]
// ^^ allowed because it just happens to be the case that all of the supported vdafs are prio3
#[derive(Serialize, PartialEq, Eq, Debug)]
pub(crate) enum SupportedVdaf {
    Prio3Count,
    Prio3Sum,
    Prio3Histogram,
    Prio3SumVec,
}

#[derive(Serialize)]
pub(crate) struct GetTaskIdsResp {
    pub(crate) task_ids: Vec<TaskId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pagination_token: Option<TaskId>,
}

#[derive(Derivative, PartialEq, Eq, Serialize, Deserialize)]
#[derivative(Debug)]
pub(crate) struct PostTaskReq {
    /// URL relative to which this task's peer aggregator's DAP API can be found. The peer
    /// aggregator plays the DAP role opposite to the one in the `role` field.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    pub(crate) peer_aggregator_endpoint: Url,
    /// DAP query type for this task.
    pub(crate) query_type: QueryType,
    /// The VDAF being run by this task.
    pub(crate) vdaf: VdafInstance,
    /// The role that this aggregator will play in this task.
    pub(crate) role: Role,
    /// The VDAF verification key used for this DAP task, as Base64 encoded bytes. Task ID is
    /// derived from the verify key.
    pub(crate) vdaf_verify_key: String,
    /// The maximum number of times a given batch may be collected.
    pub(crate) max_batch_query_count: u64,
    /// The time after which the task is considered invalid.
    pub(crate) task_expiration: Option<Time>,
    /// The minimum number of reports in a batch to allow it to be collected.
    pub(crate) min_batch_size: u64,
    /// The duration to which clients should round their reported timestamps, as seconds since
    /// the UNIX epoch.
    pub(crate) time_precision: Duration,
    /// HPKE configuration for the collector.
    pub(crate) collector_hpke_config: HpkeConfig,
    /// If this aggregator is the leader, this is the token to use to authenticate requests to
    /// the helper. If this aggregator is the helper, the value is `None`.
    pub(crate) aggregator_auth_token: Option<AuthenticationToken>,
    /// If this aggregator is the leader, this is the token hash used to authenticate collection
    /// sub-protocol requests received from the helper. If this aggregator is the helper, the value
    /// is `None`.
    pub(crate) collector_auth_token_hash: Option<AuthenticationTokenHash>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PatchTaskReq {
    #[serde(default, deserialize_with = "deserialize_some")]
    pub(crate) task_expiration: Option<Option<Time>>,
}

#[derive(Clone, Derivative, PartialEq, Eq, Serialize, Deserialize)]
#[derivative(Debug)]
pub(crate) struct TaskResp {
    /// ID of the DAP Task.
    pub(crate) task_id: TaskId,
    /// URL relative to which this task's peer aggregator's DAP API can be found. The peer
    /// aggregator plays the DAP role opposite to the one in the `role` field.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    pub(crate) peer_aggregator_endpoint: Url,
    /// DAP query type for this task.
    pub(crate) query_type: QueryType,
    /// The VDAF being run by this task.
    pub(crate) vdaf: VdafInstance,
    /// The role that this aggregator will play in this task.
    pub(crate) role: Role,
    /// The VDAF verification key used for this DAP task, as Base64 encoded bytes. Task ID is
    /// derived from the verify key.
    pub(crate) vdaf_verify_key: String,
    /// The maximum number of times a given batch may be collected.
    pub(crate) max_batch_query_count: u64,
    /// The time after which the task is considered invalid.
    pub(crate) task_expiration: Option<Time>,
    /// The age after which a report is considered to be "expired" and will be considered a
    /// candidate for garbage collection.
    pub(crate) report_expiry_age: Option<Duration>,
    /// The minimum number of reports in a batch to allow it to be collected.
    pub(crate) min_batch_size: u64,
    /// The duration to which clients should round their reported timestamps.
    pub(crate) time_precision: Duration,
    /// How much clock skew to allow between client and aggregator. Reports from
    /// farther than this duration into the future will be rejected.
    pub(crate) tolerable_clock_skew: Duration,
    /// The authentication token for inter-aggregator communication in this task. Only set in the
    /// initial response to a task creation request and only when the role is helper. Subsequent
    /// `TaskResp`s obtained from `GET /tasks/:task_id` will not contain the authentication token.
    pub(crate) aggregator_auth_token: Option<AuthenticationToken>,
    /// HPKE configuration used by the collector to decrypt aggregate shares.
    pub(crate) collector_hpke_config: HpkeConfig,
    /// HPKE configuration(s) used by this aggregator to decrypt report shares.
    pub(crate) aggregator_hpke_configs: Vec<HpkeConfig>,
}

impl TryFrom<&AggregatorTask> for TaskResp {
    type Error = &'static str;

    fn try_from(task: &AggregatorTask) -> Result<Self, Self::Error> {
        let mut aggregator_hpke_configs: Vec<_> = task
            .hpke_keys()
            .values()
            .map(|keypair| keypair.config().clone())
            .collect();
        aggregator_hpke_configs.sort_by_key(|config| *config.id());

        Ok(Self {
            task_id: *task.id(),
            peer_aggregator_endpoint: task.peer_aggregator_endpoint().clone(),
            query_type: *task.query_type(),
            vdaf: task.vdaf().clone(),
            role: *task.role(),
            vdaf_verify_key: URL_SAFE_NO_PAD.encode(task.opaque_vdaf_verify_key().as_ref()),
            max_batch_query_count: task.max_batch_query_count(),
            task_expiration: task.task_expiration().copied(),
            report_expiry_age: task.report_expiry_age().cloned(),
            min_batch_size: task.min_batch_size(),
            time_precision: *task.time_precision(),
            tolerable_clock_skew: *task.tolerable_clock_skew(),
            aggregator_auth_token: None,
            collector_hpke_config: task
                .collector_hpke_config()
                .ok_or("collector_hpke_config is required")?
                .clone(),
            aggregator_hpke_configs,
        })
    }
}

#[derive(Serialize)]
pub(crate) struct GetTaskUploadMetricsResp(pub(crate) TaskUploadCounter);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GlobalHpkeConfigResp {
    pub(crate) config: HpkeConfig,
    pub(crate) state: HpkeKeyState,
}

impl From<GlobalHpkeKeypair> for GlobalHpkeConfigResp {
    fn from(value: GlobalHpkeKeypair) -> Self {
        Self {
            config: value.hpke_keypair().config().clone(),
            state: *value.state(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct PutGlobalHpkeConfigReq {
    pub(crate) kem_id: Option<HpkeKemId>,
    pub(crate) kdf_id: Option<HpkeKdfId>,
    pub(crate) aead_id: Option<HpkeAeadId>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct PatchGlobalHpkeConfigReq {
    pub(crate) state: HpkeKeyState,
}

#[derive(Derivative, PartialEq, Eq, Serialize, Deserialize)]
#[derivative(Debug)]
pub(crate) struct TaskprovPeerAggregatorResp {
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    pub(crate) endpoint: Url,
    pub(crate) role: Role,
    pub(crate) collector_hpke_config: HpkeConfig,
    pub(crate) report_expiry_age: Option<Duration>,
    pub(crate) tolerable_clock_skew: Duration,
}

impl From<PeerAggregator> for TaskprovPeerAggregatorResp {
    fn from(value: PeerAggregator) -> Self {
        // Exclude sensitive values.
        Self {
            endpoint: value.endpoint().clone(),
            role: *value.role(),
            collector_hpke_config: value.collector_hpke_config().clone(),
            report_expiry_age: value.report_expiry_age().cloned(),
            tolerable_clock_skew: *value.tolerable_clock_skew(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct PostTaskprovPeerAggregatorReq {
    pub(crate) endpoint: Url,
    pub(crate) role: Role,
    pub(crate) collector_hpke_config: HpkeConfig,
    pub(crate) verify_key_init: VerifyKeyInit,
    pub(crate) report_expiry_age: Option<Duration>,
    pub(crate) tolerable_clock_skew: Duration,
    pub(crate) aggregator_auth_tokens: Vec<AuthenticationToken>,
    pub(crate) collector_auth_tokens: Vec<AuthenticationToken>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct DeleteTaskprovPeerAggregatorReq {
    pub(crate) endpoint: Url,
    pub(crate) role: Role,
}

// Any value that is present is considered Some value, including null. See
// https://github.com/serde-rs/serde/issues/984#issuecomment-314143738
fn deserialize_some<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(Some)
}
