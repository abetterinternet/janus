//! This module contains models used by the datastore that are not DAP messages.

use crate::{datastore::Error, task};
use base64::{display::Base64Display, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::NaiveDateTime;
use clap::ValueEnum;
use educe::Educe;
use janus_core::{
    auth_tokens::{AuthenticationToken, AuthenticationTokenHash},
    hpke::{self, HpkeCiphersuite},
    report_id::ReportIdChecksumExt,
    time::{DurationExt, IntervalExt, TimeExt},
    vdaf::VdafInstance,
};
use janus_messages::{
    batch_mode::{BatchMode, LeaderSelected, TimeInterval},
    AggregationJobId, AggregationJobStep, BatchId, CollectionJobId, Duration, Extension,
    HpkeCiphertext, HpkeConfigId, Interval, PrepareContinue, PrepareInit, PrepareResp, Query,
    ReportError, ReportId, ReportIdChecksum, ReportMetadata, Role, TaskId, Time,
};
use postgres_protocol::types::{
    range_from_sql, range_to_sql, timestamp_from_sql, timestamp_to_sql, Range, RangeBound,
};
use postgres_types::{accepts, to_sql_checked, FromSql, ToSql};
use prio::{
    codec::{encode_u16_items, Encode},
    topology::ping_pong::{PingPongState, PingPongTransition},
    vdaf::{self, Aggregatable},
};
use rand::{distributions::Standard, prelude::Distribution};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    hash::Hash,
    ops::RangeInclusive,
};

// We have to manually implement [Partial]Eq for a number of types because the derived
// implementations don't play nice with generic fields, even if those fields are constrained to
// themselves implement [Partial]Eq.

/// AuthenticationTokenType represents the type of an authentication token. It corresponds to enum
/// `AUTH_TOKEN_TYPE` in the schema.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ToSql, FromSql)]
#[postgres(name = "auth_token_type")]
pub enum AuthenticationTokenType {
    #[postgres(name = "DAP_AUTH")]
    DapAuthToken,
    #[postgres(name = "BEARER")]
    AuthorizationBearerToken,
}

impl AuthenticationTokenType {
    pub fn as_authentication(&self, token_bytes: &[u8]) -> Result<AuthenticationToken, Error> {
        match self {
            Self::DapAuthToken => AuthenticationToken::new_dap_auth_token_from_bytes(token_bytes),
            Self::AuthorizationBearerToken => {
                AuthenticationToken::new_bearer_token_from_bytes(token_bytes)
            }
        }
        .map_err(|e| Error::DbState(format!("invalid DAP auth token in database: {e:?}")))
    }

    pub fn as_authentication_token_hash(
        &self,
        hash_bytes: &[u8],
    ) -> Result<AuthenticationTokenHash, Error> {
        let hash_array = hash_bytes
            .try_into()
            .map_err(|e| Error::DbState(format!("invalid auth token hash in database: {e:?}")))?;
        Ok(match self {
            Self::DapAuthToken => AuthenticationTokenHash::DapAuth(hash_array),
            Self::AuthorizationBearerToken => AuthenticationTokenHash::Bearer(hash_array),
        })
    }
}

impl From<&AuthenticationToken> for AuthenticationTokenType {
    fn from(value: &AuthenticationToken) -> Self {
        match value {
            AuthenticationToken::DapAuth(_) => Self::DapAuthToken,
            AuthenticationToken::Bearer(_) => Self::AuthorizationBearerToken,
            _ => unreachable!(),
        }
    }
}

impl From<&AuthenticationTokenHash> for AuthenticationTokenType {
    fn from(value: &AuthenticationTokenHash) -> Self {
        match value {
            AuthenticationTokenHash::DapAuth(_) => Self::DapAuthToken,
            AuthenticationTokenHash::Bearer(_) => Self::AuthorizationBearerToken,
            _ => unreachable!(),
        }
    }
}

/// Represents a report as it is stored in the Leader's database, corresponding to an unscrubbed row
/// in `client_reports`.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub struct LeaderStoredReport<const SEED_SIZE: usize, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
{
    task_id: TaskId,
    metadata: ReportMetadata,
    #[educe(Debug(ignore))]
    public_share: A::PublicShare,
    #[educe(Debug(ignore))]
    leader_private_extensions: Vec<Extension>,
    #[educe(Debug(ignore))]
    leader_input_share: A::InputShare,
    #[educe(Debug(ignore))]
    helper_encrypted_input_share: HpkeCiphertext,
}

impl<const SEED_SIZE: usize, A> LeaderStoredReport<SEED_SIZE, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
{
    pub fn new(
        task_id: TaskId,
        metadata: ReportMetadata,
        public_share: A::PublicShare,
        leader_private_extensions: Vec<Extension>,
        leader_input_share: A::InputShare,
        helper_encrypted_input_share: HpkeCiphertext,
    ) -> Self {
        Self {
            task_id,
            metadata,
            public_share,
            leader_private_extensions,
            leader_input_share,
            helper_encrypted_input_share,
        }
    }

    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    pub fn metadata(&self) -> &ReportMetadata {
        &self.metadata
    }

    pub fn public_share(&self) -> &A::PublicShare {
        &self.public_share
    }

    pub fn leader_private_extensions(&self) -> &[Extension] {
        &self.leader_private_extensions
    }

    pub fn leader_input_share(&self) -> &A::InputShare {
        &self.leader_input_share
    }

    pub fn helper_encrypted_input_share(&self) -> &HpkeCiphertext {
        &self.helper_encrypted_input_share
    }

    #[cfg(feature = "test-util")]
    pub fn as_leader_init_report_aggregation(
        &self,
        aggregation_job_id: AggregationJobId,
        ord: u64,
    ) -> ReportAggregation<SEED_SIZE, A> {
        ReportAggregation::new(
            *self.task_id(),
            aggregation_job_id,
            *self.metadata().id(),
            *self.metadata().time(),
            ord,
            None,
            ReportAggregationState::LeaderInit {
                public_extensions: self.metadata().public_extensions().to_vec(),
                public_share: self.public_share().clone(),
                leader_private_extensions: self.leader_private_extensions().to_vec(),
                leader_input_share: self.leader_input_share().clone(),
                helper_encrypted_input_share: self.helper_encrypted_input_share().clone(),
            },
        )
    }

    #[cfg(feature = "test-util")]
    pub fn eq_report(
        &self,
        vdaf: &A,
        leader_hpke_keypair: &hpke::HpkeKeypair,
        report: &janus_messages::Report,
    ) -> bool
    where
        A::PublicShare: PartialEq,
        A::InputShare: PartialEq,
    {
        use janus_core::hpke::{self, HpkeApplicationInfo, Label};
        use janus_messages::{InputShareAad, PlaintextInputShare};
        use prio::codec::{Decode, ParameterizedDecode};

        // Decrypt/decode elements from the Report so that we can check them against the parsed
        // values held by the LeaderStoredReport.
        let public_share = A::PublicShare::get_decoded_with_param(vdaf, report.public_share())
            .expect("couldn't parse Leader's PublicShare");

        let encoded_leader_plaintext_input_share = hpke::open(
            leader_hpke_keypair,
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Leader),
            report.leader_encrypted_input_share(),
            &InputShareAad::new(
                self.task_id,
                report.metadata().clone(),
                report.public_share().to_vec(),
            )
            .get_encoded()
            .unwrap(),
        )
        .expect("couldn't decrypt Leader's PlaintextInputShare");
        let leader_plaintext_input_share =
            PlaintextInputShare::get_decoded(&encoded_leader_plaintext_input_share)
                .expect("couldn't decode Leader's PlaintextInputShare");
        let leader_input_share = A::InputShare::get_decoded_with_param(
            &(vdaf, Role::Leader.index().unwrap()),
            leader_plaintext_input_share.payload(),
        )
        .expect("couldn't decode Leader's InputShare");

        // Check equality of all relevant fields.
        self.metadata() == report.metadata()
            && self.public_share() == &public_share
            && self.leader_private_extensions() == leader_plaintext_input_share.private_extensions()
            && self.leader_input_share() == &leader_input_share
    }

    #[cfg(feature = "test-util")]
    pub fn generate(
        task_id: TaskId,
        report_metadata: ReportMetadata,
        helper_hpke_config: &janus_messages::HpkeConfig,
        leader_private_extensions: Vec<Extension>,
        transcript: &janus_core::test_util::VdafTranscript<SEED_SIZE, A>,
    ) -> Self
    where
        A: vdaf::Client<16>,
    {
        use janus_core::hpke::{self, HpkeApplicationInfo, Label};
        use janus_messages::{InputShareAad, PlaintextInputShare};

        let encrypted_helper_input_share = hpke::seal(
            helper_hpke_config,
            &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, &Role::Helper),
            &PlaintextInputShare::new(
                Vec::new(),
                transcript.helper_input_share.get_encoded().unwrap(),
            )
            .get_encoded()
            .unwrap(),
            &InputShareAad::new(
                task_id,
                report_metadata.clone(),
                transcript.public_share.get_encoded().unwrap(),
            )
            .get_encoded()
            .unwrap(),
        )
        .unwrap();

        LeaderStoredReport::new(
            task_id,
            report_metadata,
            transcript.public_share.clone(),
            leader_private_extensions,
            transcript.leader_input_share.clone(),
            encrypted_helper_input_share,
        )
    }
}

impl<const SEED_SIZE: usize, A> PartialEq for LeaderStoredReport<SEED_SIZE, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
    A::InputShare: PartialEq,
    A::PublicShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
            && self.metadata == other.metadata
            && self.public_share == other.public_share
            && self.leader_private_extensions == other.leader_private_extensions
            && self.leader_input_share == other.leader_input_share
            && self.helper_encrypted_input_share == other.helper_encrypted_input_share
    }
}

impl<const SEED_SIZE: usize, A> Eq for LeaderStoredReport<SEED_SIZE, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
    A::InputShare: Eq,
    A::PublicShare: PartialEq,
{
}

#[cfg(feature = "test-util")]
impl LeaderStoredReport<0, prio::vdaf::dummy::Vdaf> {
    pub fn new_dummy(task_id: TaskId, when: Time) -> Self {
        use rand::random;

        Self::new(
            task_id,
            ReportMetadata::new(random(), when, Vec::new()),
            (),
            Vec::new(),
            prio::vdaf::dummy::InputShare::default(),
            HpkeCiphertext::new(
                HpkeConfigId::from(13),
                Vec::from("encapsulated_context_0"),
                Vec::from("payload_0"),
            ),
        )
    }
}

/// A message providing information about an unaggregated client reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnaggregatedReport {
    report_id: ReportId,
    client_timestamp: Time,
}

impl UnaggregatedReport {
    /// Creates a new [`UnaggregatedReport`].
    pub fn new(report_id: ReportId, client_timestamp: Time) -> Self {
        Self {
            report_id,
            client_timestamp,
        }
    }

    /// Returns the ID of the unaggregated report.
    pub fn report_id(&self) -> &ReportId {
        &self.report_id
    }

    /// Returns the timestamp of the unaggregated report.
    pub fn client_timestamp(&self) -> &Time {
        &self.client_timestamp
    }
}

/// AggregatorRole corresponds to the `AGGREGATOR_ROLE` enum in the schema.
#[derive(Clone, Debug, ToSql, FromSql)]
#[postgres(name = "aggregator_role")]
pub enum AggregatorRole {
    #[postgres(name = "LEADER")]
    Leader,
    #[postgres(name = "HELPER")]
    Helper,
}

impl AggregatorRole {
    /// If the provided [`Role`] is an aggregator, returns the corresponding
    /// [`AggregatorRole`], or `None` otherwise.
    pub fn from_role(role: Role) -> Result<Self, Error> {
        match role {
            Role::Leader => Ok(Self::Leader),
            Role::Helper => Ok(Self::Helper),
            _ => Err(Error::Task(task::Error::InvalidParameter(
                "role is not an aggregator",
            ))),
        }
    }

    /// Returns the [`Role`] corresponding to this value.
    pub fn as_role(&self) -> Role {
        match self {
            Self::Leader => Role::Leader,
            Self::Helper => Role::Helper,
        }
    }
}

/// AggregationJob represents an aggregation job from the DAP specification.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub struct AggregationJob<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>>
{
    /// The ID of the task this aggregation job belongs to.
    task_id: TaskId,
    /// The ID of this aggregation job.
    aggregation_job_id: AggregationJobId,
    /// The aggregation parameter this job is run with.
    #[educe(Debug(ignore))]
    aggregation_parameter: A::AggregationParam,
    /// The partial identifier for the batch this aggregation job contributes to (leader-selected
    /// tasks only; for time interval tasks, aggregation jobs may span multiple batches).
    batch_id: B::PartialBatchIdentifier,
    /// The minimal interval of time spanned by the reports included in this aggregation job.
    client_timestamp_interval: Interval,
    /// The overall state of this aggregation job.
    state: AggregationJobState,
    /// The step of VDAF preparation that this aggregation job is currently on.
    step: AggregationJobStep,
    /// The SHA-256 hash of the most recent [`janus_messages::AggregationJobContinueReq`]
    /// received for this aggregation job. Will only be set for helpers, and only after the
    /// first step of the job.
    last_request_hash: Option<[u8; 32]>,
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>>
    AggregationJob<SEED_SIZE, B, A>
{
    /// Creates a new [`AggregationJob`].
    pub fn new(
        task_id: TaskId,
        aggregation_job_id: AggregationJobId,
        aggregation_parameter: A::AggregationParam,
        batch_id: B::PartialBatchIdentifier,
        client_timestamp_interval: Interval,
        state: AggregationJobState,
        step: AggregationJobStep,
    ) -> Self {
        Self {
            task_id,
            aggregation_job_id,
            aggregation_parameter,
            batch_id,
            client_timestamp_interval,
            state,
            step,
            last_request_hash: None,
        }
    }

    /// Returns the task ID associated with this aggregation job.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Returns the aggregation job ID associated with this aggregation job.
    pub fn id(&self) -> &AggregationJobId {
        &self.aggregation_job_id
    }

    /// Returns the aggregation parameter associated with this aggregation job.
    pub fn aggregation_parameter(&self) -> &A::AggregationParam {
        &self.aggregation_parameter
    }

    /// Gets the partial batch identifier associated with this aggregation job.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call [`Self::batch_id`].
    pub fn partial_batch_identifier(&self) -> &B::PartialBatchIdentifier {
        &self.batch_id
    }

    /// Returns the minimal interval containing all of the client timestamps associated with
    /// this aggregation job.
    pub fn client_timestamp_interval(&self) -> &Interval {
        &self.client_timestamp_interval
    }

    /// Returns the state of the aggregation job.
    pub fn state(&self) -> &AggregationJobState {
        &self.state
    }

    /// Returns a new [`AggregationJob`] corresponding to this aggregation job updated to have
    /// the given state.
    pub fn with_state(self, state: AggregationJobState) -> Self {
        AggregationJob { state, ..self }
    }

    /// Returns the step the aggregation job is on.
    pub fn step(&self) -> AggregationJobStep {
        self.step
    }

    /// Returns a new [`AggregationJob`] corresponding to this aggregation job updated to be on
    /// the given step.
    pub fn with_step(self, step: AggregationJobStep) -> Self {
        Self { step, ..self }
    }

    /// Returns the SHA-256 digest of the most recent
    /// [`janus_messages::AggregationJobInitializeReq`] or
    /// [`janus_messages::AggregationJobContinueReq`] for the job, if any.
    pub fn last_request_hash(&self) -> Option<[u8; 32]> {
        self.last_request_hash
    }

    /// Returns a new [`AggregationJob`] corresponding to this aggregation job updated to have the
    /// given last request hash.
    pub fn with_last_request_hash(self, hash: [u8; 32]) -> Self {
        Self {
            last_request_hash: Some(hash),
            ..self
        }
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    AggregationJob<SEED_SIZE, LeaderSelected, A>
{
    /// Gets the batch ID associated with this aggregation job.
    pub fn batch_id(&self) -> &BatchId {
        self.partial_batch_identifier()
    }
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for AggregationJob<SEED_SIZE, B, A>
where
    A::AggregationParam: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
            && self.aggregation_job_id == other.aggregation_job_id
            && self.aggregation_parameter == other.aggregation_parameter
            && self.batch_id == other.batch_id
            && self.client_timestamp_interval == other.client_timestamp_interval
            && self.state == other.state
            && self.step == other.step
            && self.last_request_hash == other.last_request_hash
    }
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for AggregationJob<SEED_SIZE, B, A>
where
    A::AggregationParam: Eq,
{
}

/// AggregationJobState represents the state of an aggregation job. It corresponds to the
/// AGGREGATION_JOB_STATE enum in the schema.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, ToSql, FromSql)]
#[postgres(name = "aggregation_job_state")]
pub enum AggregationJobState {
    #[postgres(name = "ACTIVE")]
    Active,
    #[postgres(name = "AWAITING_REQUEST")]
    AwaitingRequest,
    #[postgres(name = "FINISHED")]
    Finished,
    #[postgres(name = "ABANDONED")]
    Abandoned,
    #[postgres(name = "DELETED")]
    Deleted,
}

/// LeaseToken represents an opaque value used to determine the identity of a lease.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct LeaseToken([u8; Self::LEN]);

impl LeaseToken {
    /// The length of a lease token in bytes.
    pub const LEN: usize = 16;
}

impl Debug for LeaseToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LeaseToken({})",
            Base64Display::new(&self.0, &URL_SAFE_NO_PAD)
        )
    }
}

impl Display for LeaseToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl TryFrom<&[u8]> for LeaseToken {
    type Error = &'static str;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            "byte slice has incorrect length for LeaseToken"
        })?))
    }
}

impl AsRef<[u8; Self::LEN]> for LeaseToken {
    fn as_ref(&self) -> &[u8; Self::LEN] {
        &self.0
    }
}

impl Distribution<LeaseToken> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> LeaseToken {
        LeaseToken(rng.gen())
    }
}

/// A time-constrained lease for exclusive access to some entity in Janus.
///
/// It has an expiry after which it is no longer valid; another process can take a lease on the
/// same entity after the expiration time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lease<T> {
    leased: T,
    lease_expiry_time: NaiveDateTime,
    lease_token: LeaseToken,
    lease_attempts: usize,
}

impl<T> Lease<T> {
    /// Creates a new [`Lease`].
    pub fn new(
        leased: T,
        lease_expiry_time: NaiveDateTime,
        lease_token: LeaseToken,
        lease_attempts: usize,
    ) -> Self {
        Self {
            leased,
            lease_expiry_time,
            lease_token,
            lease_attempts,
        }
    }

    /// Create a new artificial lease with a random lease token, acquired for the first time;
    /// intended for use in unit tests.
    #[cfg(feature = "test-util")]
    pub fn new_dummy(leased: T, lease_expiry_time: NaiveDateTime) -> Self {
        use rand::random;
        Self {
            leased,
            lease_expiry_time,
            lease_token: random(),
            lease_attempts: 1,
        }
    }

    /// Returns a reference to the leased entity associated with this lease.
    pub fn leased(&self) -> &T {
        &self.leased
    }

    /// Returns the lease expiry time associated with this lease.
    pub fn lease_expiry_time(&self) -> &NaiveDateTime {
        &self.lease_expiry_time
    }

    /// Returns the lease token associated with this lease.
    pub fn lease_token(&self) -> &LeaseToken {
        &self.lease_token
    }

    /// Returns the number of lease acquiries since the last successful release.
    pub fn lease_attempts(&self) -> usize {
        self.lease_attempts
    }
}

/// AcquiredAggregationJob represents an incomplete aggregation job whose lease has been
/// acquired.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AcquiredAggregationJob {
    task_id: TaskId,
    aggregation_job_id: AggregationJobId,
    batch_mode: task::BatchMode,
    vdaf: VdafInstance,
}

impl AcquiredAggregationJob {
    /// Creates a new [`AcquiredAggregationJob`].
    pub fn new(
        task_id: TaskId,
        aggregation_job_id: AggregationJobId,
        batch_mode: task::BatchMode,
        vdaf: VdafInstance,
    ) -> Self {
        Self {
            task_id,
            aggregation_job_id,
            batch_mode,
            vdaf,
        }
    }

    /// Returns the task ID associated with this acquired aggregation job.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Returns the aggregation job ID associated with this acquired aggregation job.
    pub fn aggregation_job_id(&self) -> &AggregationJobId {
        &self.aggregation_job_id
    }

    /// Returns the batch mode associated with this acquired aggregation job.
    pub fn batch_mode(&self) -> &task::BatchMode {
        &self.batch_mode
    }

    /// Returns the VDAF associated with this acquired aggregation job.
    pub fn vdaf(&self) -> &VdafInstance {
        &self.vdaf
    }
}

/// AcquiredCollectionJob represents an incomplete collection job whose lease has been acquired.
#[derive(Clone, Debug, PartialOrd, Ord, Eq, PartialEq)]
pub struct AcquiredCollectionJob {
    task_id: TaskId,
    collection_job_id: CollectionJobId,
    batch_mode: task::BatchMode,
    vdaf: VdafInstance,
    time_precision: Duration,
    encoded_batch_identifier: Vec<u8>,
    encoded_aggregation_param: Vec<u8>,
    step_attempts: u64,
}

impl AcquiredCollectionJob {
    /// Creates a new [`AcquiredCollectionJob`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_id: TaskId,
        collection_job_id: CollectionJobId,
        batch_mode: task::BatchMode,
        vdaf: VdafInstance,
        time_precision: Duration,
        encoded_batch_identifier: Vec<u8>,
        encoded_aggregation_param: Vec<u8>,
        step_attempts: u64,
    ) -> Self {
        Self {
            task_id,
            collection_job_id,
            batch_mode,
            vdaf,
            time_precision,
            encoded_batch_identifier,
            encoded_aggregation_param,
            step_attempts,
        }
    }

    /// Returns the task ID associated with this acquired collection job.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Returns the collection job ID associated with this acquired collection job.
    pub fn collection_job_id(&self) -> &CollectionJobId {
        &self.collection_job_id
    }

    /// Returns the batch mode associated with this acquired collection job.
    pub fn batch_mode(&self) -> &task::BatchMode {
        &self.batch_mode
    }

    /// Returns the VDAF associated with this acquired collection job.
    pub fn vdaf(&self) -> &VdafInstance {
        &self.vdaf
    }

    /// Returns the time precision of the task associated with this acquired collection job.
    pub fn time_precision(&self) -> &Duration {
        &self.time_precision
    }

    /// Returns the batch identifier associated with this acquired collection job, encoded to bytes.
    pub fn encoded_batch_identifier(&self) -> &[u8] {
        &self.encoded_batch_identifier
    }

    /// Returns the aggregation parameter associated with this acquired collection job, encoded to
    /// bytes.
    pub fn encoded_aggregation_param(&self) -> &[u8] {
        &self.encoded_aggregation_param
    }

    /// Returns the number of times this collection job has been stepped without making progress.
    pub fn step_attempts(&self) -> u64 {
        self.step_attempts
    }

    #[cfg(feature = "test-util")]
    pub fn with_step_attempts(self, step_attempts: u64) -> Self {
        Self {
            step_attempts,
            ..self
        }
    }
}

/// ReportAggregation represents a the state of a single client report's ongoing aggregation.
#[derive(Clone, Debug)]
pub struct ReportAggregation<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
    task_id: TaskId,
    aggregation_job_id: AggregationJobId,
    report_id: ReportId,
    time: Time,
    ord: u64,
    last_prep_resp: Option<PrepareResp>,
    state: ReportAggregationState<SEED_SIZE, A>,
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> ReportAggregation<SEED_SIZE, A> {
    /// Creates a new [`ReportAggregation`].
    pub fn new(
        task_id: TaskId,
        aggregation_job_id: AggregationJobId,
        report_id: ReportId,
        time: Time,
        ord: u64,
        last_prep_resp: Option<PrepareResp>,
        state: ReportAggregationState<SEED_SIZE, A>,
    ) -> Self {
        Self {
            task_id,
            aggregation_job_id,
            report_id,
            time,
            ord,
            last_prep_resp,
            state,
        }
    }

    /// Returns the task ID associated with this report aggregation.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Returns the aggregation job ID associated with this report aggregation.
    pub fn aggregation_job_id(&self) -> &AggregationJobId {
        &self.aggregation_job_id
    }

    /// Returns the report ID associated with this report aggregation.
    pub fn report_id(&self) -> &ReportId {
        &self.report_id
    }

    /// Returns the client timestamp associated with this report aggregation.
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Returns the order of this report aggregation in its aggregation job.
    pub fn ord(&self) -> u64 {
        self.ord
    }

    /// Returns the last preparation response returned by the Helper, if any.
    pub fn last_prep_resp(&self) -> Option<&PrepareResp> {
        self.last_prep_resp.as_ref()
    }

    /// Returns a new [`ReportAggregation`] corresponding to this report aggregation updated to
    /// have the given last preparation response.
    pub fn with_last_prep_resp(self, last_prep_resp: Option<PrepareResp>) -> Self {
        Self {
            last_prep_resp,
            ..self
        }
    }

    /// Returns the state of the report aggregation.
    pub fn state(&self) -> &ReportAggregationState<SEED_SIZE, A> {
        &self.state
    }

    /// Returns a new [`ReportAggregation`] corresponding to this report aggregation updated to
    /// have the given state.
    pub fn with_state(self, state: ReportAggregationState<SEED_SIZE, A>) -> Self {
        Self { state, ..self }
    }
}

// This trait implementation is gated on the `test-util` feature as we do not wish to compare
// preparation states in non-test code, since doing so would require a constant-time comparison to
// avoid risking leaking information about the preparation state.
#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for ReportAggregation<SEED_SIZE, A>
where
    A::InputShare: PartialEq,
    A::PrepareShare: PartialEq,
    A::PrepareState: PartialEq,
    A::PublicShare: PartialEq,
    A::OutputShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
            && self.aggregation_job_id == other.aggregation_job_id
            && self.report_id == other.report_id
            && self.time == other.time
            && self.ord == other.ord
            && self.last_prep_resp == other.last_prep_resp
            && self.state == other.state
    }
}

// This trait implementation is gated on the `test-util` feature as we do not wish to compare
// preparation states in non-test code, since doing so would require a constant-time comparison to
// avoid risking leaking information about the preparation state.
#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for ReportAggregation<SEED_SIZE, A>
where
    A::InputShare: Eq,
    A::PrepareShare: Eq,
    A::PrepareState: Eq,
    A::PublicShare: Eq,
    A::OutputShare: Eq,
{
}

/// ReportAggregationState represents the state of a single report aggregation. It corresponds
/// to the REPORT_AGGREGATION_STATE enum in the schema, along with the state-specific data.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub enum ReportAggregationState<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
    //
    // Leader-only states.
    //
    /// The Leader is ready to send an aggregation initialization request to the Helper.
    LeaderInit {
        /// The sequence of public extensions from this report's metadata.
        #[educe(Debug(ignore))]
        public_extensions: Vec<Extension>,
        /// Public share for this report.
        #[educe(Debug(ignore))]
        public_share: A::PublicShare,
        /// The sequence of extensions from the Leader's input share for this report.
        #[educe(Debug(ignore))]
        leader_private_extensions: Vec<Extension>,
        /// The Leader's input share for this report.
        #[educe(Debug(ignore))]
        leader_input_share: A::InputShare,
        /// The Helper's encrypted input share for this report.
        #[educe(Debug(ignore))]
        helper_encrypted_input_share: HpkeCiphertext,
    },
    /// The Leader is ready to send an aggregation continuation request to the Helper.
    LeaderContinue {
        /// Most recent transition for this report aggregation.
        #[educe(Debug(ignore))]
        transition: PingPongTransition<SEED_SIZE, 16, A>,
    },
    /// The Leader received a "processing" response from a previous aggregation initialization or
    /// continuation request, and is ready to poll for completion.
    LeaderPoll {
        /// Leader's current aggregation state.
        #[educe(Debug(ignore))]
        leader_state: PingPongState<SEED_SIZE, 16, A>,
    },

    //
    // Helper-only states.
    //
    /// The Helper has received an aggregation initialization request from the Leader, and is
    /// processing it asynchronously.
    HelperInitProcessing {
        /// The initialization message received for this report aggregation.
        prepare_init: PrepareInit,
        /// Does this report aggregation require the taskprov extension?
        require_taskbind_extension: bool,
    },
    /// The Helper is ready to receive an aggregation continuation request from the Leader.
    HelperContinue {
        /// Helper's current preparation state
        #[educe(Debug(ignore))]
        prepare_state: A::PrepareState,
    },
    /// The Helper has received an aggregation continuation request from the Leader, and is
    /// processing it asynchronously.
    HelperContinueProcessing {
        /// Helper's current preparation state.
        #[educe(Debug(ignore))]
        prepare_state: A::PrepareState,
        /// The message from the Leader for this report aggregation.
        #[educe(Debug(ignore))]
        prepare_continue: PrepareContinue,
    },

    //
    // Common states.
    //
    /// Aggregation has completed successfully.
    Finished,
    /// Aggregation has completed unsuccessfully.
    Failed { report_error: ReportError },
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    ReportAggregationState<SEED_SIZE, A>
{
    pub(super) fn state_code(&self) -> ReportAggregationStateCode {
        match self {
            ReportAggregationState::LeaderInit { .. } => ReportAggregationStateCode::Init,
            ReportAggregationState::LeaderContinue { .. } => ReportAggregationStateCode::Continue,
            ReportAggregationState::LeaderPoll { .. } => ReportAggregationStateCode::Poll,

            ReportAggregationState::HelperInitProcessing { .. } => {
                ReportAggregationStateCode::InitProcessing
            }
            ReportAggregationState::HelperContinue { .. } => ReportAggregationStateCode::Continue,
            ReportAggregationState::HelperContinueProcessing { .. } => {
                ReportAggregationStateCode::ContinueProcessing
            }

            ReportAggregationState::Finished => ReportAggregationStateCode::Finished,
            ReportAggregationState::Failed { .. } => ReportAggregationStateCode::Failed,
        }
    }

    /// Returns the encoded values for the various messages which might be included in a
    /// ReportAggregationState. The order of returned values is preparation state, preparation
    /// message, output share, transition error.
    pub(super) fn encoded_values_from_state(
        &self,
    ) -> Result<EncodedReportAggregationStateValues, Error>
    where
        A::PrepareState: Encode,
    {
        Ok(match self {
            ReportAggregationState::LeaderInit {
                public_extensions,
                public_share,
                leader_private_extensions,
                leader_input_share,
                helper_encrypted_input_share,
            } => {
                let mut encoded_public_extensions = Vec::new();
                encode_u16_items(&mut encoded_public_extensions, &(), public_extensions)?;

                let mut encoded_leader_private_extensions = Vec::new();
                encode_u16_items(
                    &mut encoded_leader_private_extensions,
                    &(),
                    leader_private_extensions,
                )?;

                EncodedReportAggregationStateValues {
                    public_extensions: Some(encoded_public_extensions),
                    public_share: Some(public_share.get_encoded()?),
                    leader_private_extensions: Some(encoded_leader_private_extensions),
                    leader_input_share: Some(leader_input_share.get_encoded()?),
                    helper_encrypted_input_share: Some(helper_encrypted_input_share.get_encoded()?),
                    ..Default::default()
                }
            }
            ReportAggregationState::LeaderContinue { transition } => {
                EncodedReportAggregationStateValues {
                    leader_prep_transition: Some(transition.get_encoded()?),
                    ..Default::default()
                }
            }
            ReportAggregationState::LeaderPoll { leader_state } => {
                let (encoded_leader_prep_state, encoded_leader_output_share) = match leader_state {
                    PingPongState::Continued(prepare_state) => {
                        (Some(prepare_state.get_encoded()?), None)
                    }
                    PingPongState::Finished(output_share) => {
                        (None, Some(output_share.get_encoded()?))
                    }
                };
                EncodedReportAggregationStateValues {
                    leader_prep_state: encoded_leader_prep_state,
                    leader_output_share: encoded_leader_output_share,
                    ..Default::default()
                }
            }

            ReportAggregationState::HelperInitProcessing {
                prepare_init,
                require_taskbind_extension,
            } => EncodedReportAggregationStateValues {
                prepare_init: Some(prepare_init.get_encoded()?),
                require_taskbind_extension: Some(*require_taskbind_extension),
                ..Default::default()
            },

            ReportAggregationState::HelperContinue { prepare_state } => {
                EncodedReportAggregationStateValues {
                    helper_prep_state: Some(prepare_state.get_encoded()?),
                    ..Default::default()
                }
            }

            ReportAggregationState::HelperContinueProcessing {
                prepare_state,
                prepare_continue,
            } => EncodedReportAggregationStateValues {
                helper_prep_state: Some(prepare_state.get_encoded()?),
                prepare_continue: Some(prepare_continue.get_encoded()?),
                ..Default::default()
            },

            ReportAggregationState::Finished => EncodedReportAggregationStateValues::default(),

            ReportAggregationState::Failed { report_error } => {
                EncodedReportAggregationStateValues {
                    report_error: Some(*report_error as i16),
                    ..Default::default()
                }
            }
        })
    }
}

#[derive(Default)]
pub(super) struct EncodedReportAggregationStateValues {
    // State for LeaderInit.
    pub(super) public_extensions: Option<Vec<u8>>,
    pub(super) public_share: Option<Vec<u8>>,
    pub(super) leader_private_extensions: Option<Vec<u8>>,
    pub(super) leader_input_share: Option<Vec<u8>>,
    pub(super) helper_encrypted_input_share: Option<Vec<u8>>,

    // State for LeaderContinue.
    pub(super) leader_prep_transition: Option<Vec<u8>>,

    // State for LeaderPoll.
    pub(super) leader_prep_state: Option<Vec<u8>>,
    pub(super) leader_output_share: Option<Vec<u8>>,

    // State for HelperInitProcessing.
    pub(super) prepare_init: Option<Vec<u8>>,
    pub(super) require_taskbind_extension: Option<bool>,

    // State for HelperContinue & HelperContinueProcessing.
    pub(super) helper_prep_state: Option<Vec<u8>>,

    // State for HelperContinueProcessing.
    pub(super) prepare_continue: Option<Vec<u8>>,

    // State for Failed.
    pub(super) report_error: Option<i16>,
}

// ReportAggregationStateCode exists alongside the public ReportAggregationState because there is no
// apparent way to denote a Postgres enum literal without deriving FromSql/ToSql on a Rust enum
// type, but it is not possible to derive FromSql/ToSql on a non-C-style enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromSql, ToSql)]
#[postgres(name = "report_aggregation_state")]
pub(super) enum ReportAggregationStateCode {
    #[postgres(name = "INIT")]
    Init,
    #[postgres(name = "INIT_PROCESSING")]
    InitProcessing,
    #[postgres(name = "CONTINUE")]
    Continue,
    #[postgres(name = "CONTINUE_PROCESSING")]
    ContinueProcessing,
    #[postgres(name = "POLL")]
    Poll,
    #[postgres(name = "FINISHED")]
    Finished,
    #[postgres(name = "FAILED")]
    Failed,
}

// This trait implementation is gated on the `test-util` feature as we do not wish to compare
// preparation states in non-test code, since doing so would require a constant-time comparison to
// avoid risking leaking information about the preparation state.
#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for ReportAggregationState<SEED_SIZE, A>
where
    A::InputShare: PartialEq,
    A::PrepareShare: PartialEq,
    A::PrepareState: PartialEq,
    A::PublicShare: PartialEq,
    A::OutputShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::LeaderInit {
                    public_extensions: lhs_public_extensions,
                    public_share: lhs_public_share,
                    leader_private_extensions: lhs_leader_private_extensions,
                    leader_input_share: lhs_leader_input_share,
                    helper_encrypted_input_share: lhs_helper_encrypted_input_share,
                },
                Self::LeaderInit {
                    public_extensions: rhs_public_extensions,
                    public_share: rhs_public_share,
                    leader_private_extensions: rhs_leader_private_extensions,
                    leader_input_share: rhs_leader_input_share,
                    helper_encrypted_input_share: rhs_helper_encrypted_input_share,
                },
            ) => {
                lhs_public_extensions == rhs_public_extensions
                    && lhs_public_share == rhs_public_share
                    && lhs_leader_private_extensions == rhs_leader_private_extensions
                    && lhs_leader_input_share == rhs_leader_input_share
                    && lhs_helper_encrypted_input_share == rhs_helper_encrypted_input_share
            }

            (
                Self::LeaderContinue {
                    transition: lhs_transition,
                },
                Self::LeaderContinue {
                    transition: rhs_transition,
                },
            ) => lhs_transition == rhs_transition,

            (
                Self::LeaderPoll {
                    leader_state: lhs_leader_state,
                },
                Self::LeaderPoll {
                    leader_state: rhs_leader_state,
                },
            ) => match (lhs_leader_state, rhs_leader_state) {
                (
                    PingPongState::Continued(lhs_prepare_state),
                    PingPongState::Continued(rhs_prepare_state),
                ) => lhs_prepare_state == rhs_prepare_state,
                (
                    PingPongState::Finished(lhs_output_share),
                    PingPongState::Finished(rhs_output_share),
                ) => lhs_output_share == rhs_output_share,
                _ => false,
            },

            (
                Self::HelperInitProcessing {
                    prepare_init: lhs_prepare_init,
                    require_taskbind_extension: lhs_require_taskbind_extension,
                },
                Self::HelperInitProcessing {
                    prepare_init: rhs_prepare_init,
                    require_taskbind_extension: rhs_require_taskbind_extension,
                },
            ) => {
                lhs_prepare_init == rhs_prepare_init
                    && lhs_require_taskbind_extension == rhs_require_taskbind_extension
            }

            (
                Self::HelperContinue {
                    prepare_state: lhs_state,
                },
                Self::HelperContinue {
                    prepare_state: rhs_state,
                },
            ) => lhs_state == rhs_state,

            (
                Self::HelperContinueProcessing {
                    prepare_state: lhs_state,
                    prepare_continue: lhs_prepare_continue,
                },
                Self::HelperContinueProcessing {
                    prepare_state: rhs_state,
                    prepare_continue: rhs_prepare_continue,
                },
            ) => lhs_state == rhs_state && lhs_prepare_continue == rhs_prepare_continue,

            (
                Self::Failed {
                    report_error: lhs_report_error,
                },
                Self::Failed {
                    report_error: rhs_report_error,
                },
            ) => lhs_report_error == rhs_report_error,

            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

// This trait implementation is gated on the `test-util` feature as we do not wish to compare
// preparation states in non-test code, since doing so would require a constant-time comparison to
// avoid risking leaking information about the preparation state.
#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for ReportAggregationState<SEED_SIZE, A>
where
    A::InputShare: Eq,
    A::PrepareShare: Eq,
    A::PrepareState: Eq,
    A::PublicShare: Eq,
    A::OutputShare: Eq,
{
}

/// Limited set of report aggregation states usable with [`ReportAggregationMetadata`].
///
/// See also [`ReportAggregationState`].
#[derive(Clone, Debug)]
pub enum ReportAggregationMetadataState {
    Init,
    Failed { report_error: ReportError },
}

/// Metadata from the state of a single client report's ongoing aggregation. This is like
/// [`ReportAggregation`], but omits the report aggregation state and report shares.
///
/// This is only used with report aggregations in the `LeaderInit` or `Failed` states.
#[derive(Clone, Debug)]
pub struct ReportAggregationMetadata {
    task_id: TaskId,
    aggregation_job_id: AggregationJobId,
    report_id: ReportId,
    time: Time,
    ord: u64,
    state: ReportAggregationMetadataState,
}

impl ReportAggregationMetadata {
    /// Creates a new [`ReportAggregationMetadata`].
    pub fn new(
        task_id: TaskId,
        aggregation_job_id: AggregationJobId,
        report_id: ReportId,
        time: Time,
        ord: u64,
        state: ReportAggregationMetadataState,
    ) -> Self {
        Self {
            task_id,
            aggregation_job_id,
            report_id,
            time,
            ord,
            state,
        }
    }

    /// Returns the task ID associated with this report aggregation.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Returns the aggregation job ID associated with this report aggregation.
    pub fn aggregation_job_id(&self) -> &AggregationJobId {
        &self.aggregation_job_id
    }

    /// Returns the report ID associated with this report aggregation.
    pub fn report_id(&self) -> &ReportId {
        &self.report_id
    }

    /// Returns the client timestamp associated with this report aggregation.
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Returns the order of this report aggregation in its aggregation job.
    pub fn ord(&self) -> u64 {
        self.ord
    }

    /// Returns the state of the report aggregation.
    pub fn state(&self) -> &ReportAggregationMetadataState {
        &self.state
    }

    /// Returns a new [`ReportAggregationMetadata`] corresponding to this report aggregation updated
    /// to have the given state.
    pub fn with_state(self, state: ReportAggregationMetadataState) -> Self {
        Self { state, ..self }
    }
}

/// Corresponds to a row in the `batch_aggregations` table.
///
/// This represents the possibly-ongoing aggregation of the set of input shares that fall within
/// the batch identified by `batch_identifier` with the aggregation parameter
/// `aggregation_parameter`.  This is the finest-grained possible aggregate share we can emit for
/// this task. The aggregate share constructed to service a collect or aggregate share request
/// consists of one or more `BatchAggregation`s merged together.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub struct BatchAggregation<
    const SEED_SIZE: usize,
    B: BatchMode,
    A: vdaf::Aggregator<SEED_SIZE, 16>,
> {
    /// The task ID for this aggregation result.
    task_id: TaskId,
    /// The identifier of the batch being aggregated over.
    batch_identifier: B::BatchIdentifier,
    /// The VDAF aggregation parameter used to prepare and accumulate input shares.
    #[educe(Debug(ignore))]
    aggregation_parameter: A::AggregationParam,
    /// The index of this batch aggregation among all batch aggregations for
    /// this (task_id, batch_identifier, aggregation_parameter).
    ord: u64,

    /// The minimal interval of time spanned by the reports included in this batch aggregation
    /// shard.
    client_timestamp_interval: Interval,
    /// The current state of the batch aggregation.
    state: BatchAggregationState<SEED_SIZE, A>,
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>>
    BatchAggregation<SEED_SIZE, B, A>
{
    /// Creates a new [`BatchAggregation`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_id: TaskId,
        batch_identifier: B::BatchIdentifier,
        aggregation_parameter: A::AggregationParam,
        ord: u64,
        client_timestamp_interval: Interval,
        state: BatchAggregationState<SEED_SIZE, A>,
    ) -> Self {
        Self {
            task_id,
            batch_identifier,
            aggregation_parameter,
            ord,
            client_timestamp_interval,
            state,
        }
    }

    /// Returns the task ID associated with this batch aggregation.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Gets the batch identifier included in this batch aggregation.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call one of [`Self::batch_interval`] or
    /// [`Self::batch_id`].
    pub fn batch_identifier(&self) -> &B::BatchIdentifier {
        &self.batch_identifier
    }

    /// Returns the aggregation parameter associated with this batch aggregation.
    pub fn aggregation_parameter(&self) -> &A::AggregationParam {
        &self.aggregation_parameter
    }

    /// Returns the index of this batch aggregations among all batch aggregations for this (task_id,
    /// batch_identifier, aggregation_parameter).
    pub fn ord(&self) -> u64 {
        self.ord
    }

    /// Returns the minimal interval of time spanned by the reports included in this batch
    /// aggregation shard.
    pub fn client_timestamp_interval(&self) -> &Interval {
        &self.client_timestamp_interval
    }

    /// Returns the current state associated with this batch aggregation.
    pub fn state(&self) -> &BatchAggregationState<SEED_SIZE, A> {
        &self.state
    }

    // Returns a [`BatchAggregation`] identical to this one, with the given batch aggregation state.
    pub fn with_state(self, state: BatchAggregationState<SEED_SIZE, A>) -> Self {
        Self { state, ..self }
    }

    /// Returns this batch aggregation, marked as collected. The batch aggregation state is kept.
    /// Returns an error only if called on an already-scrubbed batch aggregation.
    pub fn collected(self) -> Result<Self, Error> {
        match self.state {
            BatchAggregationState::Aggregating {
                aggregate_share,
                report_count,
                checksum,
                aggregation_jobs_created,
                aggregation_jobs_terminated,
            } => Ok(Self {
                state: BatchAggregationState::Collected {
                    aggregate_share,
                    report_count,
                    checksum,
                    aggregation_jobs_created,
                    aggregation_jobs_terminated,
                },
                ..self
            }),
            BatchAggregationState::Collected { .. } => Ok(self),
            BatchAggregationState::Scrubbed => Err(Error::Scrubbed),
        }
    }

    /// Returns this batch aggregation, scrubbed of aggregation state.
    pub fn scrubbed(self) -> Self {
        Self {
            state: BatchAggregationState::Scrubbed,
            ..self
        }
    }

    /// Returns a new [`BatchAggregation`] corresponding to the current batch aggregation merged
    /// with the given batch aggregation. Only uncollected, unscrubbed batch aggregations may be
    /// merged.
    pub fn merged_with(self, other: &Self) -> Result<Self, Error> {
        match (self.state, other.state()) {
            (
                BatchAggregationState::Aggregating {
                    aggregate_share: lhs_aggregate_share,
                    report_count: lhs_report_count,
                    checksum: lhs_checksum,
                    aggregation_jobs_created: lhs_aggregation_jobs_created,
                    aggregation_jobs_terminated: lhs_aggregation_jobs_terminated,
                },
                BatchAggregationState::Aggregating {
                    aggregate_share: rhs_aggregate_share,
                    report_count: rhs_report_count,
                    checksum: rhs_checksum,
                    aggregation_jobs_created: rhs_aggregation_jobs_created,
                    aggregation_jobs_terminated: rhs_aggregation_jobs_terminated,
                },
            ) => {
                let merged_aggregate_share = match (lhs_aggregate_share, rhs_aggregate_share) {
                    (Some(mut my_agg), Some(other_agg)) => Some({
                        my_agg
                            .merge(other_agg)
                            .map_err(|err| Error::User(err.into()))?;
                        my_agg
                    }),
                    (Some(my_agg), None) => Some(my_agg),
                    (None, Some(other_agg)) => Some(other_agg.clone()),
                    (None, None) => None,
                };

                Ok(Self {
                    state: BatchAggregationState::Aggregating {
                        aggregate_share: merged_aggregate_share,
                        report_count: lhs_report_count + rhs_report_count,
                        checksum: lhs_checksum.combined_with(rhs_checksum),
                        aggregation_jobs_created: lhs_aggregation_jobs_created
                            + rhs_aggregation_jobs_created,
                        aggregation_jobs_terminated: lhs_aggregation_jobs_terminated
                            + rhs_aggregation_jobs_terminated,
                    },
                    client_timestamp_interval: self
                        .client_timestamp_interval
                        .merge(&other.client_timestamp_interval)?,
                    ..self
                })
            }

            (BatchAggregationState::Scrubbed, _) | (_, BatchAggregationState::Scrubbed) => {
                Err(Error::Scrubbed)
            }

            (BatchAggregationState::Collected { .. }, _)
            | (_, BatchAggregationState::Collected { .. }) => Err(Error::AlreadyCollected),
        }
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    BatchAggregation<SEED_SIZE, TimeInterval, A>
{
    /// Gets the batch interval associated with this batch aggregation.
    pub fn batch_interval(&self) -> &Interval {
        self.batch_identifier()
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    BatchAggregation<SEED_SIZE, LeaderSelected, A>
{
    /// Gets the batch ID associated with this batch aggregation.
    pub fn batch_id(&self) -> &BatchId {
        self.batch_identifier()
    }
}

#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for BatchAggregation<SEED_SIZE, B, A>
where
    A::AggregationParam: PartialEq,
    A::AggregateShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
            && self.batch_identifier == other.batch_identifier
            && self.aggregation_parameter == other.aggregation_parameter
            && self.ord == other.ord
            && self.state == other.state
    }
}

#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for BatchAggregation<SEED_SIZE, B, A>
where
    A::AggregationParam: Eq,
    A::AggregateShare: Eq,
{
}

/// Represents the state of a batch aggregation.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub enum BatchAggregationState<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
    Aggregating {
        /// The aggregate over all the input shares that have been prepared so far by this
        /// aggregator. Will only be None if there are no reports.
        #[educe(Debug(ignore))]
        aggregate_share: Option<A::AggregateShare>,
        /// The number of reports currently included in this aggregate share.
        report_count: u64,
        /// Checksum over the aggregated report shares.
        #[educe(Debug(ignore))]
        checksum: ReportIdChecksum,
        /// The number of aggregation jobs that have been created for this batch.
        aggregation_jobs_created: u64,
        /// The number of aggregation jobs that have been terminated for this batch.
        aggregation_jobs_terminated: u64,
    },

    Collected {
        /// The aggregate over all the input shares that have been prepared so far by this
        /// aggregator. Will only be None if there are no reports.
        #[educe(Debug(ignore))]
        aggregate_share: Option<A::AggregateShare>,
        /// The number of reports currently included in this aggregate share.
        report_count: u64,
        /// Checksum over the aggregated report shares.
        #[educe(Debug(ignore))]
        checksum: ReportIdChecksum,
        /// The number of aggregation jobs that have been created for this batch.
        aggregation_jobs_created: u64,
        /// The number of aggregation jobs that have been terminated for this batch.
        aggregation_jobs_terminated: u64,
    },

    Scrubbed,
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    BatchAggregationState<SEED_SIZE, A>
{
    /// Returns true if this batch aggregation state indicates the batch aggregation is still
    /// accepting additional aggregation.
    pub fn is_accepting_aggregations(&self) -> bool {
        matches!(self, Self::Aggregating { .. })
    }

    pub(super) fn state_code(&self) -> BatchAggregationStateCode {
        match self {
            BatchAggregationState::Aggregating { .. } => BatchAggregationStateCode::Aggregating,
            BatchAggregationState::Collected { .. } => BatchAggregationStateCode::Collected,
            BatchAggregationState::Scrubbed => BatchAggregationStateCode::Scrubbed,
        }
    }

    pub(super) fn encoded_values_from_state(
        &self,
    ) -> Result<EncodedBatchAggregationStateValues, Error> {
        Ok(match self {
            BatchAggregationState::Aggregating {
                aggregate_share,
                report_count,
                checksum,
                aggregation_jobs_created,
                aggregation_jobs_terminated,
            }
            | BatchAggregationState::Collected {
                aggregate_share,
                report_count,
                checksum,
                aggregation_jobs_created,
                aggregation_jobs_terminated,
            } => EncodedBatchAggregationStateValues {
                aggregate_share: aggregate_share
                    .as_ref()
                    .map(Encode::get_encoded)
                    .transpose()?,
                report_count: Some(i64::try_from(*report_count)?),
                checksum: Some(checksum.get_encoded()?),
                aggregation_jobs_created: Some(i64::try_from(*aggregation_jobs_created)?),
                aggregation_jobs_terminated: Some(i64::try_from(*aggregation_jobs_terminated)?),
            },
            BatchAggregationState::Scrubbed => EncodedBatchAggregationStateValues::default(),
        })
    }
}

#[derive(Default)]
pub(super) struct EncodedBatchAggregationStateValues {
    // State for Aggregating & Collected states.
    pub(super) aggregate_share: Option<Vec<u8>>,
    pub(super) report_count: Option<i64>,
    pub(super) checksum: Option<Vec<u8>>,
    pub(super) aggregation_jobs_created: Option<i64>,
    pub(super) aggregation_jobs_terminated: Option<i64>,
}

#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for BatchAggregationState<SEED_SIZE, A>
where
    A::AggregateShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Aggregating {
                    aggregate_share: lhs_aggregate_share,
                    report_count: lhs_report_count,
                    checksum: lhs_checksum,
                    aggregation_jobs_created: lhs_aggregation_jobs_created,
                    aggregation_jobs_terminated: lhs_aggregation_jobs_terminated,
                },
                Self::Aggregating {
                    aggregate_share: rhs_aggregate_share,
                    report_count: rhs_report_count,
                    checksum: rhs_checksum,
                    aggregation_jobs_created: rhs_aggregation_jobs_created,
                    aggregation_jobs_terminated: rhs_aggregation_jobs_terminated,
                },
            ) => {
                lhs_aggregate_share == rhs_aggregate_share
                    && lhs_report_count == rhs_report_count
                    && lhs_checksum == rhs_checksum
                    && lhs_aggregation_jobs_created == rhs_aggregation_jobs_created
                    && lhs_aggregation_jobs_terminated == rhs_aggregation_jobs_terminated
            }
            (
                Self::Collected {
                    aggregate_share: lhs_aggregate_share,
                    report_count: lhs_report_count,
                    checksum: lhs_checksum,
                    aggregation_jobs_created: lhs_aggregation_jobs_created,
                    aggregation_jobs_terminated: lhs_aggregation_jobs_terminated,
                },
                Self::Collected {
                    aggregate_share: rhs_aggregate_share,
                    report_count: rhs_report_count,
                    checksum: rhs_checksum,
                    aggregation_jobs_created: rhs_aggregation_jobs_created,
                    aggregation_jobs_terminated: rhs_aggregation_jobs_terminated,
                },
            ) => {
                lhs_aggregate_share == rhs_aggregate_share
                    && lhs_report_count == rhs_report_count
                    && lhs_checksum == rhs_checksum
                    && lhs_aggregation_jobs_created == rhs_aggregation_jobs_created
                    && lhs_aggregation_jobs_terminated == rhs_aggregation_jobs_terminated
            }
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

#[cfg(feature = "test-util")]
impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for BatchAggregationState<SEED_SIZE, A>
where
    A::AggregateShare: Eq,
{
}

// ReportAggregationStateCode exists alongside the public ReportAggregationState because there is no
// apparent way to denote a Postgres enum literal without deriving FromSql/ToSql on a Rust enum
// type, but it is not possible to derive FromSql/ToSql on a non-C-style enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromSql, ToSql)]
#[postgres(name = "batch_aggregation_state")]
pub(super) enum BatchAggregationStateCode {
    /// This batch aggregation has not been collected & permits further aggregation.
    #[postgres(name = "AGGREGATING")]
    Aggregating,
    /// This batch aggregation has been collected & no longer permits aggregation.
    #[postgres(name = "COLLECTED")]
    Collected,
    #[postgres(name = "SCRUBBED")]
    Scrubbed,
}

/// Merges batch aggregations for the same batch (by task ID, batch identifier, and aggregation
/// parameter), returning a single merged batch aggregation per batch with ord 0.
///
/// Batch aggregations for different batches are returned sorted by task ID, batch identifier, and
/// aggregation parameter.
#[cfg(feature = "test-util")]
pub fn merge_batch_aggregations_by_batch<
    const SEED_SIZE: usize,
    B: BatchMode,
    A: vdaf::Aggregator<SEED_SIZE, 16>,
>(
    mut batch_aggregations: Vec<BatchAggregation<SEED_SIZE, B, A>>,
) -> Vec<BatchAggregation<SEED_SIZE, B, A>>
where
    A::AggregationParam: Ord,
{
    use itertools::Itertools as _;

    batch_aggregations.sort_by(|ba_l, ba_r| {
        (
            ba_l.task_id(),
            ba_l.batch_identifier(),
            ba_l.aggregation_parameter(),
        )
            .cmp(&(
                ba_r.task_id(),
                ba_r.batch_identifier(),
                ba_r.aggregation_parameter(),
            ))
    });
    batch_aggregations
        .into_iter()
        .chunk_by(|ba| {
            (
                *ba.task_id(),
                ba.batch_identifier().clone(),
                ba.aggregation_parameter().clone(),
            )
        })
        .into_iter()
        .map(|(_, bas)| bas.reduce(|l, r| l.merged_with(&r).unwrap()).unwrap())
        .map(|ba| {
            BatchAggregation::new(
                *ba.task_id(),
                ba.batch_identifier().clone(),
                ba.aggregation_parameter().clone(),
                0,
                *ba.client_timestamp_interval(),
                ba.state().clone(),
            )
        })
        .collect()
}

/// CollectionJob represents a row in the `collection_jobs` table, used by leaders to represent
/// running collection jobs and store the results of completed ones.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub struct CollectionJob<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> {
    /// The task ID for this collection job.
    task_id: TaskId,
    /// The unique identifier for the collection job.
    collection_job_id: CollectionJobId,
    /// The Query that was sent to create this collection job.
    query: Query<B>,
    /// The VDAF aggregation parameter used to prepare and aggregate input shares.
    #[educe(Debug(ignore))]
    aggregation_parameter: A::AggregationParam,
    /// The batch interval covered by the collection job.
    batch_identifier: B::BatchIdentifier,
    /// The current state of the collection job.
    state: CollectionJobState<SEED_SIZE, A>,
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>>
    CollectionJob<SEED_SIZE, B, A>
{
    /// Creates a new [`CollectionJob`].
    pub fn new(
        task_id: TaskId,
        collection_job_id: CollectionJobId,
        query: Query<B>,
        aggregation_parameter: A::AggregationParam,
        batch_identifier: B::BatchIdentifier,
        state: CollectionJobState<SEED_SIZE, A>,
    ) -> Self {
        Self {
            task_id,
            collection_job_id,
            query,
            aggregation_parameter,
            batch_identifier,
            state,
        }
    }

    /// Returns the task ID associated with this collection job.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Returns the collection job ID associated with this collection job.
    pub fn id(&self) -> &CollectionJobId {
        &self.collection_job_id
    }

    /// Returns the query that was sent to create this collection job.
    pub fn query(&self) -> &Query<B> {
        &self.query
    }

    /// Returns the aggregation parameter associated with this collection job.
    pub fn aggregation_parameter(&self) -> &A::AggregationParam {
        &self.aggregation_parameter
    }

    /// Takes the aggregation parameter associated with this collection job, consuming the job.
    pub fn take_aggregation_parameter(self) -> A::AggregationParam {
        self.aggregation_parameter
    }

    /// Gets the batch identifier associated with this collection job.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call one of [`Self::batch_interval`] or
    /// [`Self::batch_id`].
    pub fn batch_identifier(&self) -> &B::BatchIdentifier {
        &self.batch_identifier
    }

    /// Returns the state associated with this collection job.
    pub fn state(&self) -> &CollectionJobState<SEED_SIZE, A> {
        &self.state
    }

    /// Returns a new [`CollectionJob`] corresponding to this collection job updated to have the given
    /// state.
    pub fn with_state(self, state: CollectionJobState<SEED_SIZE, A>) -> Self {
        Self { state, ..self }
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    CollectionJob<SEED_SIZE, TimeInterval, A>
{
    /// Gets the batch interval associated with this collection job.
    pub fn batch_interval(&self) -> &Interval {
        self.batch_identifier()
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    CollectionJob<SEED_SIZE, LeaderSelected, A>
{
    /// Gets the batch ID associated with this collection job.
    pub fn batch_id(&self) -> &BatchId {
        self.batch_identifier()
    }
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for CollectionJob<SEED_SIZE, B, A>
where
    A::AggregationParam: PartialEq,
    CollectionJobState<SEED_SIZE, A>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
            && self.collection_job_id == other.collection_job_id
            && self.batch_identifier == other.batch_identifier
            && self.aggregation_parameter == other.aggregation_parameter
            && self.state == other.state
    }
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for CollectionJob<SEED_SIZE, B, A>
where
    A::AggregationParam: Eq,
    CollectionJobState<SEED_SIZE, A>: Eq,
{
}

#[derive(Clone, Educe)]
#[educe(Debug)]
pub enum CollectionJobState<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
    Start,
    Finished {
        /// The number of reports included in this collection job.
        report_count: u64,
        /// The minimal interval containing the timestamps of all reports included in this
        /// collection job, aligned to the task's time precision.
        client_timestamp_interval: Interval,
        /// The helper's encrypted aggregate share over the input shares in the interval.
        encrypted_helper_aggregate_share: HpkeCiphertext,
        /// The leader's aggregate share over the input shares in the interval.
        #[educe(Debug(ignore))]
        leader_aggregate_share: A::AggregateShare,
    },
    Abandoned,
    Deleted,
}

impl<const SEED_SIZE: usize, A> CollectionJobState<SEED_SIZE, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
{
    pub fn collection_job_state_code(&self) -> CollectionJobStateCode {
        match self {
            Self::Start => CollectionJobStateCode::Start,
            Self::Finished { .. } => CollectionJobStateCode::Finished,
            Self::Abandoned => CollectionJobStateCode::Abandoned,
            Self::Deleted => CollectionJobStateCode::Deleted,
        }
    }
}

impl<const SEED_SIZE: usize, A> Display for CollectionJobState<SEED_SIZE, A>
where
    A: vdaf::Aggregator<SEED_SIZE, 16>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Start => "start",
                Self::Finished { .. } => "finished",
                Self::Abandoned => "abandoned",
                Self::Deleted => "deleted",
            }
        )
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for CollectionJobState<SEED_SIZE, A>
where
    A::AggregateShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Finished {
                    report_count: self_report_count,
                    client_timestamp_interval: self_client_timestamp_interval,
                    encrypted_helper_aggregate_share: self_helper_agg_share,
                    leader_aggregate_share: self_leader_agg_share,
                },
                Self::Finished {
                    report_count: other_report_count,
                    client_timestamp_interval: other_client_timestamp_interval,
                    encrypted_helper_aggregate_share: other_helper_agg_share,
                    leader_aggregate_share: other_leader_agg_share,
                },
            ) => {
                self_report_count == other_report_count
                    && self_client_timestamp_interval == other_client_timestamp_interval
                    && self_helper_agg_share == other_helper_agg_share
                    && self_leader_agg_share == other_leader_agg_share
            }
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for CollectionJobState<SEED_SIZE, A>
where
    A::AggregateShare: Eq,
{
}

#[derive(Debug, Copy, Clone, FromSql, ToSql)]
#[postgres(name = "collection_job_state")]
pub enum CollectionJobStateCode {
    #[postgres(name = "START")]
    Start,
    #[postgres(name = "FINISHED")]
    Finished,
    #[postgres(name = "ABANDONED")]
    Abandoned,
    #[postgres(name = "DELETED")]
    Deleted,
}

/// AggregateShareJob represents a row in the `aggregate_share_jobs` table, used by helpers to
/// store the results of handling an AggregateShareReq from the leader.
#[derive(Clone, Educe)]
#[educe(Debug)]
pub struct AggregateShareJob<
    const SEED_SIZE: usize,
    B: BatchMode,
    A: vdaf::Aggregator<SEED_SIZE, 16>,
> {
    /// The task ID for this aggregate share.
    task_id: TaskId,
    /// The batch identifier for the batch covered by the aggregate share.
    batch_identifier: B::BatchIdentifier,
    /// The VDAF aggregation parameter used to prepare and aggregate input shares.
    #[educe(Debug(ignore))]
    aggregation_parameter: A::AggregationParam,
    /// The aggregate share over the input shares in the interval.
    #[educe(Debug(ignore))]
    helper_aggregate_share: A::AggregateShare,
    /// The number of reports included in the aggregate share.
    report_count: u64,
    /// Checksum over the aggregated report shares, as described in §4.4.4.3.
    #[educe(Debug(ignore))]
    checksum: ReportIdChecksum,
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>>
    AggregateShareJob<SEED_SIZE, B, A>
{
    /// Creates a new [`AggregateShareJob`].
    pub fn new(
        task_id: TaskId,
        batch_identifier: B::BatchIdentifier,
        aggregation_parameter: A::AggregationParam,
        helper_aggregate_share: A::AggregateShare,
        report_count: u64,
        checksum: ReportIdChecksum,
    ) -> Self {
        Self {
            task_id,
            batch_identifier,
            aggregation_parameter,
            helper_aggregate_share,
            report_count,
            checksum,
        }
    }

    /// Gets the task ID associated with this aggregate share job.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Gets the batch identifier associated with this aggregate share job.
    pub fn batch_identifier(&self) -> &B::BatchIdentifier {
        &self.batch_identifier
    }

    /// Gets the aggregation parameter associated with this aggregate share job.
    pub fn aggregation_parameter(&self) -> &A::AggregationParam {
        &self.aggregation_parameter
    }

    /// Takes the aggregation parameter associated with this aggregate share job, consuming the job.
    pub fn take_aggregation_parameter(self) -> A::AggregationParam {
        self.aggregation_parameter
    }

    /// Gets the helper aggregate share associated with this aggregate share job.
    pub fn helper_aggregate_share(&self) -> &A::AggregateShare {
        &self.helper_aggregate_share
    }

    /// Gets the report count associated with this aggregate share job.
    pub fn report_count(&self) -> u64 {
        self.report_count
    }

    /// Gets the checksum associated with this aggregate share job.
    pub fn checksum(&self) -> &ReportIdChecksum {
        &self.checksum
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    AggregateShareJob<SEED_SIZE, TimeInterval, A>
{
    /// Gets the batch interval associated with this aggregate share job.
    pub fn batch_interval(&self) -> &Interval {
        self.batch_identifier()
    }
}

impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
    AggregateShareJob<SEED_SIZE, LeaderSelected, A>
{
    /// Gets the batch ID associated with this aggregate share job.
    pub fn batch_id(&self) -> &BatchId {
        self.batch_identifier()
    }
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
    for AggregateShareJob<SEED_SIZE, B, A>
where
    A::AggregationParam: PartialEq,
    A::AggregateShare: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
            && self.batch_identifier == other.batch_identifier
            && self.aggregation_parameter == other.aggregation_parameter
            && self.helper_aggregate_share == other.helper_aggregate_share
            && self.report_count == other.report_count
            && self.checksum == other.checksum
    }
}

impl<const SEED_SIZE: usize, B: BatchMode, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
    for AggregateShareJob<SEED_SIZE, B, A>
where
    A::AggregationParam: Eq,
    A::AggregateShare: Eq,
{
}

/// An outstanding batch, which is a batch which has not yet started collection. Such a batch may
/// have additional reports allocated to it. Only applies to leader-selected batches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutstandingBatch {
    /// The task ID for this outstanding batch.
    task_id: TaskId,
    /// The batch ID for this outstanding batch.
    batch_id: BatchId,
    /// The range of possible sizes of this batch. The minimum size is the count of reports
    /// which have successfully completed the aggregation process, while the maximum size is the
    /// count of reports which are currently being aggregated or have successfully completed the
    /// aggregation process.
    size: RangeInclusive<usize>,
}

impl OutstandingBatch {
    /// Creates a new [`OutstandingBatch`].
    pub fn new(task_id: TaskId, batch_id: BatchId, size: RangeInclusive<usize>) -> Self {
        Self {
            task_id,
            batch_id,
            size,
        }
    }

    /// Gets the [`TaskId`] associated with this outstanding batch.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Gets the [`BatchId`] associated with this outstanding batch.
    pub fn id(&self) -> &BatchId {
        &self.batch_id
    }

    /// Gets the range of possible sizes of this batch. The minimum size is the count of reports
    /// which have successfully completed the aggregation process, while the maximum size is the
    /// count of reports which are currently being aggregated or have successfully completed the
    /// aggregation process.
    pub fn size(&self) -> &RangeInclusive<usize> {
        &self.size
    }
}

/// The SQL timestamp epoch, midnight UTC on 2000-01-01.
const SQL_EPOCH_TIME: Time = Time::from_seconds_since_epoch(946_684_800);

/// Wrapper around [`janus_messages::Interval`] that supports conversions to/from SQL.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SqlInterval(Interval);

impl SqlInterval {
    pub fn as_interval(&self) -> Interval {
        self.0
    }
}

impl From<Interval> for SqlInterval {
    fn from(interval: Interval) -> Self {
        Self(interval)
    }
}

impl From<&Interval> for SqlInterval {
    fn from(interval: &Interval) -> Self {
        Self::from(*interval)
    }
}

impl<'a> FromSql<'a> for SqlInterval {
    fn from_sql(
        _: &postgres_types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        match range_from_sql(raw)? {
            Range::Empty => Ok(SqlInterval(Interval::EMPTY)),
            Range::Nonempty(RangeBound::Inclusive(None), _)
            | Range::Nonempty(RangeBound::Exclusive(None), _)
            | Range::Nonempty(_, RangeBound::Inclusive(None))
            | Range::Nonempty(_, RangeBound::Exclusive(None)) => {
                Err("Interval cannot represent a timestamp range with a null bound".into())
            }
            Range::Nonempty(RangeBound::Unbounded, _)
            | Range::Nonempty(_, RangeBound::Unbounded) => {
                Err("Interval cannot represent an unbounded timestamp range".into())
            }
            Range::Nonempty(RangeBound::Exclusive(_), _)
            | Range::Nonempty(_, RangeBound::Inclusive(_)) => Err(Into::into(
                "Interval can only represent timestamp ranges that are closed at the start \
                     and open at the end",
            )),
            Range::Nonempty(
                RangeBound::Inclusive(Some(start_raw)),
                RangeBound::Exclusive(Some(end_raw)),
            ) => {
                // These timestamps represent the number of microseconds before (if negative) or
                // after (if positive) midnight, 1/1/2000.
                let start_timestamp = timestamp_from_sql(start_raw)?;
                let end_timestamp = timestamp_from_sql(end_raw)?;

                // Convert from SQL timestamp representation to the internal representation.
                let negative = start_timestamp < 0;
                let abs_start_us = start_timestamp.unsigned_abs();
                let abs_start_duration = Duration::from_microseconds(abs_start_us);
                let time = if negative {
                    SQL_EPOCH_TIME.sub(&abs_start_duration).map_err(|_| {
                        "Interval cannot represent timestamp ranges starting before the Unix \
                             epoch"
                    })?
                } else {
                    SQL_EPOCH_TIME
                        .add(&abs_start_duration)
                        .map_err(|_| "overflow when converting to Interval")?
                };

                if end_timestamp < start_timestamp {
                    return Err("timestamp range ends before it starts".into());
                }
                let duration_us = end_timestamp.abs_diff(start_timestamp);
                let duration = Duration::from_microseconds(duration_us);

                Ok(SqlInterval(Interval::new(time, duration)?))
            }
        }
    }

    accepts!(TS_RANGE);
}

fn time_to_sql_timestamp(time: Time) -> Result<i64, Error> {
    if time.is_after(&SQL_EPOCH_TIME) {
        let absolute_difference_us = time.difference(&SQL_EPOCH_TIME)?.as_microseconds()?;
        Ok(absolute_difference_us.try_into()?)
    } else {
        let absolute_difference_us = SQL_EPOCH_TIME.difference(&time)?.as_microseconds()?;
        Ok(-i64::try_from(absolute_difference_us)?)
    }
}

impl ToSql for SqlInterval {
    fn to_sql(
        &self,
        _: &postgres_types::Type,
        out: &mut bytes::BytesMut,
    ) -> Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        // Convert the interval start and end to SQL timestamps.
        let start_sql_usec = time_to_sql_timestamp(*self.0.start())
            .map_err(|_| "millisecond timestamp of Interval start overflowed")?;
        let end_sql_usec = time_to_sql_timestamp(self.0.end())
            .map_err(|_| "millisecond timestamp of Interval end overflowed")?;

        range_to_sql(
            |out| {
                timestamp_to_sql(start_sql_usec, out);
                Ok(postgres_protocol::types::RangeBound::Inclusive(
                    postgres_protocol::IsNull::No,
                ))
            },
            |out| {
                timestamp_to_sql(end_sql_usec, out);
                Ok(postgres_protocol::types::RangeBound::Exclusive(
                    postgres_protocol::IsNull::No,
                ))
            },
            out,
        )?;

        Ok(postgres_types::IsNull::No)
    }

    accepts!(TS_RANGE);

    to_sql_checked!();
}

/// The state of an HPKE key pair, corresponding to the HPKE_KEY_STATE enum in the schema.
#[derive(
    Copy, Clone, Debug, Hash, PartialEq, Eq, ToSql, FromSql, Serialize, Deserialize, ValueEnum,
)]
#[postgres(name = "hpke_key_state")]
#[serde(rename_all = "snake_case")]
pub enum HpkeKeyState {
    /// The key should be advertised to DAP clients, and is a preferred key for new reports to be
    /// encrypted with. The key should only be in this state if all Janus replicas have had
    /// sufficient time to discover the key in at least the [`Self::Pending`] state.
    #[postgres(name = "ACTIVE")]
    Active,
    /// The key should not be advertised to DAP clients, but could be used for decrypting client
    /// reports depending on when Janus replicas pick up the new key. New keys should be created in
    /// this state and remain in this state for as long as it takes replicas to refresh their
    /// caches of HPKE keys.
    #[postgres(name = "PENDING")]
    Pending,
    /// The key is pending deletion. It should not be advertised, but could be used for decrypting
    /// client reports depending on the age of those reports or when clients have refreshed their
    /// key caches. The key should remain in this state for longer than clients are expected to
    /// cache HPKE keys.
    #[postgres(name = "EXPIRED")]
    Expired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HpkeKeypair {
    hpke_keypair: hpke::HpkeKeypair,
    state: HpkeKeyState,
    last_state_change_at: Time,
}

impl HpkeKeypair {
    pub fn new(
        hpke_keypair: hpke::HpkeKeypair,
        state: HpkeKeyState,
        last_state_change_at: Time,
    ) -> Self {
        Self {
            hpke_keypair,
            state,
            last_state_change_at,
        }
    }

    pub fn hpke_keypair(&self) -> &hpke::HpkeKeypair {
        &self.hpke_keypair
    }

    pub fn state(&self) -> &HpkeKeyState {
        &self.state
    }

    pub fn set_state(&mut self, state: HpkeKeyState, time: Time) {
        self.state = state;
        self.last_state_change_at = time;
    }

    pub fn is_pending(&self) -> bool {
        self.state == HpkeKeyState::Pending
    }

    pub fn is_active(&self) -> bool {
        self.state == HpkeKeyState::Active
    }

    pub fn is_expired(&self) -> bool {
        self.state == HpkeKeyState::Expired
    }

    pub fn last_state_change_at(&self) -> &Time {
        &self.last_state_change_at
    }

    pub fn ciphersuite(&self) -> HpkeCiphersuite {
        HpkeCiphersuite::from(self.hpke_keypair().config())
    }

    pub fn id(&self) -> &HpkeConfigId {
        self.hpke_keypair.config().id()
    }
}

/// Per-task counts of uploaded reports and upload attempts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaskUploadCounter {
    /// Reports that fell into a time interval that had already been collected.
    pub(crate) interval_collected: u64,
    /// Reports that could not be decoded.
    pub(crate) report_decode_failure: u64,
    /// Reports that could not be decrypted.
    pub(crate) report_decrypt_failure: u64,
    /// Reports that contained a timestamp too far in the past.
    pub(crate) report_expired: u64,
    /// Reports that were encrypted with an old or unknown HPKE key.
    pub(crate) report_outdated_key: u64,
    /// Reports that were successfully uploaded.
    pub(crate) report_success: u64,
    /// Reports that contain a timestamp too far in the future.
    pub(crate) report_too_early: u64,
    /// Reports that were submitted to the task before the task's start time.
    pub(crate) task_not_started: u64,
    /// Reports that were submitted to the task after the task's end time.
    pub(crate) task_ended: u64,
}

impl TaskUploadCounter {
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "test-util")]
    pub fn new_with_values(
        interval_collected: u64,
        report_decode_failure: u64,
        report_decrypt_failure: u64,
        report_expired: u64,
        report_outdated_key: u64,
        report_success: u64,
        report_too_early: u64,
        task_not_started: u64,
        task_ended: u64,
    ) -> Self {
        Self {
            interval_collected,
            report_decode_failure,
            report_decrypt_failure,
            report_expired,
            report_outdated_key,
            report_success,
            report_too_early,
            task_not_started,
            task_ended,
        }
    }

    pub fn increment_interval_collected(&mut self) {
        self.interval_collected += 1
    }

    pub fn increment_report_decode_failure(&mut self) {
        self.report_decode_failure += 1
    }

    pub fn increment_report_decrypt_failure(&mut self) {
        self.report_decrypt_failure += 1
    }

    pub fn increment_report_expired(&mut self) {
        self.report_expired += 1
    }

    pub fn increment_report_outdated_key(&mut self) {
        self.report_outdated_key += 1
    }

    pub fn increment_report_success(&mut self) {
        self.report_success += 1
    }

    pub fn increment_report_too_early(&mut self) {
        self.report_too_early += 1
    }

    pub fn increment_task_not_started(&mut self) {
        self.task_not_started += 1
    }

    pub fn increment_task_ended(&mut self) {
        self.task_ended += 1
    }

    pub fn interval_collected(&self) -> u64 {
        self.interval_collected
    }

    pub fn report_decode_failure(&self) -> u64 {
        self.report_decode_failure
    }

    pub fn report_decrypt_failure(&self) -> u64 {
        self.report_decrypt_failure
    }

    pub fn report_expired(&self) -> u64 {
        self.report_expired
    }

    pub fn report_outdated_key(&self) -> u64 {
        self.report_outdated_key
    }

    pub fn report_success(&self) -> u64 {
        self.report_success
    }

    pub fn report_too_early(&self) -> u64 {
        self.report_too_early
    }

    pub fn task_not_started(&self) -> u64 {
        self.task_not_started
    }

    pub fn task_ended(&self) -> u64 {
        self.task_ended
    }
}

/// Per-task counts of aggregated reports.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskAggregationCounter {
    /// The number of successfully-aggregated reports.
    pub(crate) success: u64,
}

impl TaskAggregationCounter {
    #[cfg(feature = "test-util")]
    pub fn new_with_values(success: u64) -> Self {
        Self { success }
    }

    /// Increments the counter of successfully-aggregated reports.
    pub fn increment_success(&mut self) {
        self.success += 1
    }

    /// Returns true if and only if this task aggregation counter is "zero", i.e. it would not
    /// change the state of the written task aggregation counters.
    pub fn is_zero(&self) -> bool {
        self == &TaskAggregationCounter::default()
    }
}
