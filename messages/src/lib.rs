//! Messages defined by the [Distributed Aggregation Protocol][dap] with serialization and
//! deserialization support.
//!
//! [dap]: https://datatracker.ietf.org/doc/draft-ietf-ppm-dap/

use self::batch_mode::{BatchMode, LeaderSelected, TimeInterval};
use anyhow::anyhow;
use base64::{display::Base64Display, engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use core::slice;
use educe::Educe;
use num_enum::{FromPrimitive, IntoPrimitive, TryFromPrimitive};
use prio::{
    codec::{
        decode_u16_items, decode_u32_items, encode_u16_items, encode_u32_items, CodecError, Decode,
        Encode,
    },
    topology::ping_pong::PingPongMessage,
};
use rand::{distributions::Standard, prelude::Distribution, Rng};
use serde::{
    de::{self, Visitor},
    Deserialize, Serialize, Serializer,
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    io::{Cursor, Read},
    num::TryFromIntError,
    str,
    str::FromStr,
    time::{SystemTime, SystemTimeError},
};

pub use prio::codec;

pub mod batch_mode;
pub mod problem_type;
pub mod taskprov;
#[cfg(test)]
mod tests;

/// Errors returned by functions and methods in this module
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An invalid parameter was passed.
    #[error("{0}")]
    InvalidParameter(&'static str),
    /// An illegal arithmetic operation on a [`Time`] or [`Duration`].
    #[error("{0}")]
    IllegalTimeArithmetic(&'static str),
    #[error("base64 decode failure: {0}")]
    Base64Decode(#[from] base64::DecodeError),
}

/// Wire-representation of an ASCII-encoded URL with minimum length 1 and maximum
/// length 2^16 - 1.
#[derive(Clone, PartialEq, Eq)]
pub struct Url(Vec<u8>);

impl Url {
    const MAX_LEN: usize = 2usize.pow(16) - 1;
}

impl Encode for Url {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_u16_items(bytes, &(), &self.0)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2 + self.0.len())
    }
}

impl Decode for Url {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Url::try_from(decode_u16_items(&(), bytes)?.as_ref())
    }
}

impl Debug for Url {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            str::from_utf8(&self.0).map_err(|_| std::fmt::Error)?
        )
    }
}

impl Display for Url {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            str::from_utf8(&self.0).map_err(|_| std::fmt::Error)?
        )
    }
}

impl TryFrom<&[u8]> for Url {
    type Error = CodecError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err(CodecError::Other(
                anyhow!("Url must be at least 1 byte long").into(),
            ))
        } else if value.len() > Url::MAX_LEN {
            Err(CodecError::Other(
                anyhow!("Url must be less than {} bytes long", Url::MAX_LEN).into(),
            ))
        } else if !value.iter().all(|i: &u8| i.is_ascii()) {
            Err(CodecError::Other(
                anyhow!("Url must be ASCII encoded").into(),
            ))
        } else {
            Ok(Self(Vec::from(value)))
        }
    }
}

impl TryFrom<&Url> for url::Url {
    type Error = url::ParseError;

    fn try_from(value: &Url) -> Result<Self, Self::Error> {
        // Unwrap safety: this type can't be constructed without being validated
        // as consisting only of ASCII.
        url::Url::parse(str::from_utf8(&value.0).unwrap())
    }
}

/// DAP protocol message representing a duration with a resolution of seconds.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Duration(u64);

impl Duration {
    pub const ZERO: Duration = Duration::from_seconds(0);

    /// Create a duration representing the provided number of seconds.
    pub const fn from_seconds(seconds: u64) -> Self {
        Self(seconds)
    }

    /// Get the number of seconds this duration represents.
    pub fn as_seconds(&self) -> u64 {
        self.0
    }
}

impl Encode for Duration {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

impl Decode for Duration {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u64::decode(bytes)?))
    }
}

impl Display for Duration {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} seconds", self.0)
    }
}

/// DAP protocol message representing an instant in time with a resolution of seconds.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Time(u64);

impl Time {
    /// Construct a [`Time`] representing the instant that is a given number of seconds after
    /// January 1st, 1970, at 0:00:00 UTC (i.e., the instant with the Unix timestamp of
    /// `timestamp`).
    pub const fn from_seconds_since_epoch(timestamp: u64) -> Self {
        Self(timestamp)
    }

    /// Get the number of seconds from January 1st, 1970, at 0:00:00 UTC to the instant represented
    /// by this [`Time`] (i.e., the Unix timestamp for the instant it represents).
    pub fn as_seconds_since_epoch(&self) -> u64 {
        self.0
    }
}

impl Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Encode for Time {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

impl Decode for Time {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u64::decode(bytes)?))
    }
}

impl TryFrom<SystemTime> for Time {
    type Error = SystemTimeError;

    fn try_from(time: SystemTime) -> Result<Self, Self::Error> {
        let duration = time.duration_since(SystemTime::UNIX_EPOCH)?;
        Ok(Time::from_seconds_since_epoch(duration.as_secs()))
    }
}

/// DAP protocol message representing a half-open interval of time with a resolution of seconds;
/// the start of the interval is included while the end of the interval is excluded.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Interval {
    /// The start of the interval.
    start: Time,
    /// The length of the interval.
    duration: Duration,
}

impl Interval {
    pub const EMPTY: Self = Self {
        start: Time::from_seconds_since_epoch(0),
        duration: Duration::ZERO,
    };

    /// Create a new [`Interval`] from the provided start and duration. Returns an error if the end
    /// of the interval cannot be represented as a [`Time`].
    pub fn new(start: Time, duration: Duration) -> Result<Self, Error> {
        start
            .0
            .checked_add(duration.0)
            .ok_or(Error::IllegalTimeArithmetic("duration overflows time"))?;

        Ok(Self { start, duration })
    }

    /// Returns a [`Time`] representing the included start of this interval.
    pub fn start(&self) -> &Time {
        &self.start
    }

    /// Get the duration of this interval.
    pub fn duration(&self) -> &Duration {
        &self.duration
    }
}

impl Encode for Interval {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.start.encode(bytes)?;
        self.duration.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.start.encoded_len()? + self.duration.encoded_len()?)
    }
}

impl Decode for Interval {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let start = Time::decode(bytes)?;
        let duration = Duration::decode(bytes)?;

        Self::new(start, duration).map_err(|e| CodecError::Other(Box::new(e)))
    }
}

impl Display for Interval {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "start: {} duration: {}", self.start, self.duration)
    }
}

/// DAP protocol message representing an ID uniquely identifying a batch, for leader-selected tasks.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BatchId([u8; Self::LEN]);

impl BatchId {
    /// LEN is the length of a batch ID in bytes.
    pub const LEN: usize = 32;
}

impl From<[u8; Self::LEN]> for BatchId {
    fn from(batch_id: [u8; Self::LEN]) -> Self {
        Self(batch_id)
    }
}

impl TryFrom<&[u8]> for BatchId {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            Error::InvalidParameter("byte slice has incorrect length for BatchId")
        })?))
    }
}

impl FromStr for BatchId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from(URL_SAFE_NO_PAD.decode(s)?.as_ref())
    }
}

impl AsRef<[u8; Self::LEN]> for BatchId {
    fn as_ref(&self) -> &[u8; Self::LEN] {
        &self.0
    }
}

impl Debug for BatchId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BatchId({})",
            Base64Display::new(&self.0, &URL_SAFE_NO_PAD)
        )
    }
}

impl Display for BatchId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl Encode for BatchId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.0);
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(Self::LEN)
    }
}

impl Decode for BatchId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut batch_id = [0; Self::LEN];
        bytes.read_exact(&mut batch_id)?;
        Ok(Self(batch_id))
    }
}

impl Distribution<BatchId> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BatchId {
        BatchId(rng.gen())
    }
}

/// DAP protocol message representing an ID uniquely identifying a client report.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReportId([u8; Self::LEN]);

impl ReportId {
    /// LEN is the length of a report ID in bytes.
    pub const LEN: usize = 16;
}

impl From<[u8; Self::LEN]> for ReportId {
    fn from(report_id: [u8; Self::LEN]) -> Self {
        Self(report_id)
    }
}

impl TryFrom<&[u8]> for ReportId {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            Error::InvalidParameter("byte slice has incorrect length for ReportId")
        })?))
    }
}

impl AsRef<[u8; Self::LEN]> for ReportId {
    fn as_ref(&self) -> &[u8; Self::LEN] {
        &self.0
    }
}

impl Debug for ReportId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReportId({})",
            Base64Display::new(&self.0, &URL_SAFE_NO_PAD)
        )
    }
}

impl Display for ReportId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl Encode for ReportId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.0);
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(Self::LEN)
    }
}

impl Decode for ReportId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut report_id = [0; Self::LEN];
        bytes.read_exact(&mut report_id)?;
        Ok(Self(report_id))
    }
}

impl FromStr for ReportId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from(URL_SAFE_NO_PAD.decode(s)?.as_ref())
    }
}

impl Distribution<ReportId> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> ReportId {
        ReportId(rng.gen())
    }
}

/// Checksum over DAP report IDs, defined in §4.4.4.3.
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct ReportIdChecksum([u8; Self::LEN]);

impl ReportIdChecksum {
    /// LEN is the length of a report ID checksum in bytes.
    pub const LEN: usize = 32;
}

impl From<[u8; Self::LEN]> for ReportIdChecksum {
    fn from(checksum: [u8; Self::LEN]) -> Self {
        Self(checksum)
    }
}

impl TryFrom<&[u8]> for ReportIdChecksum {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            Error::InvalidParameter("byte slice has incorrect length for ReportIdChecksum")
        })?))
    }
}

impl AsRef<[u8]> for ReportIdChecksum {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl AsMut<[u8]> for ReportIdChecksum {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl Display for ReportIdChecksum {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

impl Encode for ReportIdChecksum {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.0);
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(Self::LEN)
    }
}

impl Decode for ReportIdChecksum {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut checksum = Self::default();
        bytes.read_exact(&mut checksum.0)?;

        Ok(checksum)
    }
}

#[cfg(feature = "test-util")]
impl Distribution<ReportIdChecksum> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> ReportIdChecksum {
        ReportIdChecksum(rng.gen())
    }
}

/// DAP protocol message representing the different roles that participants can adopt.
#[derive(Copy, Clone, Debug, PartialEq, Eq, TryFromPrimitive, Serialize, Deserialize)]
#[repr(u8)]
pub enum Role {
    Collector = 0,
    Client = 1,
    Leader = 2,
    Helper = 3,
}

impl Role {
    /// True if this [`Role`] is one of the aggregators.
    pub fn is_aggregator(&self) -> bool {
        matches!(self, Role::Leader | Role::Helper)
    }

    /// Returns a VDAF aggregator ID if this [`Role`] is one of the aggregators, or `None` if the
    /// role is not an aggregator. This is also used in [draft-wang-ppm-dap-taskprov-04][1] and earlier
    /// to index into the `aggregator_endpoints` array.
    ///
    /// [1]: https://www.ietf.org/archive/id/draft-wang-ppm-dap-taskprov-04.html#section-3-4
    pub fn index(&self) -> Option<usize> {
        match self {
            Role::Leader => Some(0),
            Role::Helper => Some(1),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Collector => "collector",
            Self::Client => "client",
            Self::Leader => "leader",
            Self::Helper => "helper",
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown role {0}")]
pub struct RoleParseError(String);

impl FromStr for Role {
    type Err = RoleParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "collector" => Ok(Self::Collector),
            "client" => Ok(Self::Client),
            "leader" => Ok(Self::Leader),
            "helper" => Ok(Self::Helper),
            _ => Err(RoleParseError(s.to_owned())),
        }
    }
}

impl Encode for Role {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        (*self as u8).encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(1)
    }
}

impl Decode for Role {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u8::decode(bytes)?;
        Self::try_from(val)
            .map_err(|_| CodecError::Other(anyhow!("unexpected Role value {}", val).into()))
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// DAP protocol message representing an identifier for an HPKE config.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HpkeConfigId(u8);

impl Display for HpkeConfigId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Encode for HpkeConfigId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

impl Decode for HpkeConfigId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u8::decode(bytes)?))
    }
}

impl From<u8> for HpkeConfigId {
    fn from(value: u8) -> HpkeConfigId {
        HpkeConfigId(value)
    }
}

impl From<HpkeConfigId> for u8 {
    fn from(id: HpkeConfigId) -> u8 {
        id.0
    }
}

impl Distribution<HpkeConfigId> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> HpkeConfigId {
        HpkeConfigId(rng.gen())
    }
}

/// DAP protocol message representing an identifier for a DAP task.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId([u8; Self::LEN]);

impl TaskId {
    /// LEN is the length of a task ID in bytes.
    pub const LEN: usize = 32;
}

impl Debug for TaskId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TaskId({})",
            Base64Display::new(&self.0, &URL_SAFE_NO_PAD)
        )
    }
}

impl Display for TaskId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl Encode for TaskId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        bytes.extend_from_slice(&self.0);
        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(Self::LEN)
    }
}

impl Decode for TaskId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut decoded = [0u8; Self::LEN];
        bytes.read_exact(&mut decoded)?;
        Ok(Self(decoded))
    }
}

impl From<[u8; Self::LEN]> for TaskId {
    fn from(task_id: [u8; Self::LEN]) -> Self {
        Self(task_id)
    }
}

impl TryFrom<&[u8]> for TaskId {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            Error::InvalidParameter("byte slice has incorrect length for TaskId")
        })?))
    }
}

impl AsRef<[u8; Self::LEN]> for TaskId {
    fn as_ref(&self) -> &[u8; Self::LEN] {
        &self.0
    }
}

impl FromStr for TaskId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from(URL_SAFE_NO_PAD.decode(s)?.as_ref())
    }
}

impl Distribution<TaskId> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> TaskId {
        TaskId(rng.gen())
    }
}

/// This customized implementation serializes a [`TaskId`] as a base64url-encoded string, instead
/// of as a byte array. This is more compact and ergonomic when serialized to YAML, and aligns with
/// other uses of base64url encoding in DAP.
impl Serialize for TaskId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = URL_SAFE_NO_PAD.encode(self.as_ref());
        serializer.serialize_str(&encoded)
    }
}

struct TaskIdVisitor;

impl Visitor<'_> for TaskIdVisitor {
    type Value = TaskId;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a base64url-encoded string that decodes to 32 bytes")
    }

    fn visit_str<E>(self, value: &str) -> Result<TaskId, E>
    where
        E: de::Error,
    {
        let decoded = URL_SAFE_NO_PAD
            .decode(value)
            .map_err(|_| E::custom("invalid base64url value"))?;

        TaskId::try_from(decoded.as_slice()).map_err(|e| E::custom(e))
    }
}

/// This customized implementation deserializes a [`TaskId`] as a base64url-encoded string, instead
/// of as a byte array. This is more compact and ergonomic when serialized to YAML, and aligns with
/// other uses of base64url encoding in DAP.
impl<'de> Deserialize<'de> for TaskId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(TaskIdVisitor)
    }
}

/// DAP protocol message representing an HPKE key encapsulation mechanism.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromPrimitive, IntoPrimitive, Serialize, Deserialize, Hash,
)]
#[repr(u16)]
#[non_exhaustive]
pub enum HpkeKemId {
    /// NIST P-256 keys and HKDF-SHA256.
    P256HkdfSha256 = 0x0010,
    /// NIST P-384 keys and HKDF-SHA384.
    P384HkdfSha384 = 0x0011,
    /// NIST P-521 keys and HKDF-SHA512.
    P521HkdfSha512 = 0x0012,
    /// X25519 keys and HKDF-SHA256.
    X25519HkdfSha256 = 0x0020,
    /// X448 keys and HKDF-SHA512.
    X448HkdfSha512 = 0x0021,
    /// Unrecognized algorithm identifiers.
    #[num_enum(catch_all)]
    Other(u16),
}

impl Encode for HpkeKemId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        u16::from(*self).encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2)
    }
}

impl Decode for HpkeKemId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u16::decode(bytes)?;
        Ok(Self::from(val))
    }
}

/// DAP protocol message representing an HPKE key derivation function.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromPrimitive, IntoPrimitive, Serialize, Deserialize, Hash,
)]
#[repr(u16)]
#[non_exhaustive]
pub enum HpkeKdfId {
    /// HMAC Key Derivation Function SHA256.
    HkdfSha256 = 0x0001,
    /// HMAC Key Derivation Function SHA384.
    HkdfSha384 = 0x0002,
    /// HMAC Key Derivation Function SHA512.
    HkdfSha512 = 0x0003,
    /// Unrecognized algorithm identifiers.
    #[num_enum(catch_all)]
    Other(u16),
}

impl Encode for HpkeKdfId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        u16::from(*self).encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2)
    }
}

impl Decode for HpkeKdfId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u16::decode(bytes)?;
        Ok(Self::from(val))
    }
}

/// DAP protocol message representing an HPKE AEAD.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromPrimitive, IntoPrimitive, Serialize, Deserialize, Hash,
)]
#[repr(u16)]
#[non_exhaustive]
pub enum HpkeAeadId {
    /// AES-128-GCM.
    Aes128Gcm = 0x0001,
    /// AES-256-GCM.
    Aes256Gcm = 0x0002,
    /// ChaCha20Poly1305.
    ChaCha20Poly1305 = 0x0003,
    /// Unrecognized algorithm identifiers.
    #[num_enum(catch_all)]
    Other(u16),
}

impl Encode for HpkeAeadId {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        u16::from(*self).encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2)
    }
}

impl Decode for HpkeAeadId {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u16::decode(bytes)?;
        Ok(Self::from(val))
    }
}

/// DAP protocol message representing an arbitrary extension included in a client report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extension {
    extension_type: ExtensionType,
    extension_data: Vec<u8>,
}

impl Extension {
    /// Construct an extension from its type and payload.
    pub fn new(extension_type: ExtensionType, extension_data: Vec<u8>) -> Extension {
        Extension {
            extension_type,
            extension_data,
        }
    }

    /// Returns the type of this extension.
    pub fn extension_type(&self) -> &ExtensionType {
        &self.extension_type
    }

    /// Returns the unparsed data representing this extension.
    pub fn extension_data(&self) -> &[u8] {
        &self.extension_data
    }
}

impl Encode for Extension {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.extension_type.encode(bytes)?;
        encode_u16_items(bytes, &(), &self.extension_data)
    }

    fn encoded_len(&self) -> Option<usize> {
        // Type, length prefix, and extension data.
        Some(self.extension_type.encoded_len()? + 2 + self.extension_data.len())
    }
}

impl Decode for Extension {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let extension_type = ExtensionType::decode(bytes)?;
        let extension_data = decode_u16_items(&(), bytes)?;

        Ok(Self {
            extension_type,
            extension_data,
        })
    }
}

/// DAP protocol message representing the type of an extension included in a client report.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, TryFromPrimitive)]
#[repr(u16)]
#[non_exhaustive]
pub enum ExtensionType {
    Tbd = 0,
    Taskbind = 0xFF00,
}

impl Encode for ExtensionType {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        (*self as u16).encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2)
    }
}

impl Decode for ExtensionType {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u16::decode(bytes)?;
        Self::try_from(val).map_err(|_| {
            CodecError::Other(anyhow!("unexpected ExtensionType value {}", val).into())
        })
    }
}

/// DAP protocol message representing an HPKE ciphertext.
#[derive(Clone, Educe, Eq, PartialEq)]
#[educe(Debug)]
pub struct HpkeCiphertext {
    /// An identifier of the HPKE configuration used to seal the message.
    config_id: HpkeConfigId,
    /// An encapsulated HPKE key.
    #[educe(Debug(ignore))]
    encapsulated_key: Vec<u8>,
    /// An HPKE ciphertext.
    #[educe(Debug(ignore))]
    payload: Vec<u8>,
}

impl HpkeCiphertext {
    /// Construct a HPKE ciphertext message from its components.
    pub fn new(
        config_id: HpkeConfigId,
        encapsulated_key: Vec<u8>,
        payload: Vec<u8>,
    ) -> HpkeCiphertext {
        HpkeCiphertext {
            config_id,
            encapsulated_key,
            payload,
        }
    }

    /// Get the configuration identifier associated with this ciphertext.
    pub fn config_id(&self) -> &HpkeConfigId {
        &self.config_id
    }

    /// Get the encapsulated key from this ciphertext message.
    pub fn encapsulated_key(&self) -> &[u8] {
        &self.encapsulated_key
    }

    /// Get the encrypted payload from this ciphertext message.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

impl Encode for HpkeCiphertext {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.config_id.encode(bytes)?;
        encode_u16_items(bytes, &(), &self.encapsulated_key)?;
        encode_u32_items(bytes, &(), &self.payload)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.config_id.encoded_len()?
                + 2
                + self.encapsulated_key.len()
                + 4
                + self.payload.len(),
        )
    }
}

impl Decode for HpkeCiphertext {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let config_id = HpkeConfigId::decode(bytes)?;
        let encapsulated_key = decode_u16_items(&(), bytes)?;
        let payload = decode_u32_items(&(), bytes)?;

        Ok(Self {
            config_id,
            encapsulated_key,
            payload,
        })
    }
}

/// DAP protocol message representing an HPKE public key.
// TODO(#230): refactor HpkePublicKey & HpkeConfig to simplify usage
#[derive(Clone, PartialEq, Eq)]
pub struct HpkePublicKey(Vec<u8>);

impl From<Vec<u8>> for HpkePublicKey {
    fn from(key: Vec<u8>) -> Self {
        Self(key)
    }
}

impl AsRef<[u8]> for HpkePublicKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Encode for HpkePublicKey {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_u16_items(bytes, &(), &self.0)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(2 + self.0.len())
    }
}

impl Decode for HpkePublicKey {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let key = decode_u16_items(&(), bytes)?;
        Ok(Self(key))
    }
}

impl Debug for HpkePublicKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HpkePublicKey({})", self)
    }
}

impl Display for HpkePublicKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl FromStr for HpkePublicKey {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from(URL_SAFE_NO_PAD.decode(s)?))
    }
}

/// This customized implementation serializes a [`HpkePublicKey`] as a base64url-encoded string,
/// instead of as a byte array. This is more compact and ergonomic when serialized to YAML.
impl Serialize for HpkePublicKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = URL_SAFE_NO_PAD.encode(self.as_ref());
        serializer.serialize_str(&encoded)
    }
}

struct HpkePublicKeyVisitor;

impl Visitor<'_> for HpkePublicKeyVisitor {
    type Value = HpkePublicKey;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a base64url-encoded string")
    }

    fn visit_str<E>(self, value: &str) -> Result<HpkePublicKey, E>
    where
        E: de::Error,
    {
        let decoded = URL_SAFE_NO_PAD
            .decode(value)
            .map_err(|_| E::custom("invalid base64url value"))?;
        Ok(HpkePublicKey::from(decoded))
    }
}

/// This customized implementation deserializes a [`HpkePublicKey`] as a base64url-encoded string,
/// instead of as a byte array. This is more compact and ergonomic when serialized to YAML.
impl<'de> Deserialize<'de> for HpkePublicKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(HpkePublicKeyVisitor)
    }
}

/// DAP protocol message representing an HPKE config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HpkeConfig {
    id: HpkeConfigId,
    kem_id: HpkeKemId,
    kdf_id: HpkeKdfId,
    aead_id: HpkeAeadId,
    public_key: HpkePublicKey,
}

impl HpkeConfig {
    /// Construct a HPKE configuration message from its components.
    pub fn new(
        id: HpkeConfigId,
        kem_id: HpkeKemId,
        kdf_id: HpkeKdfId,
        aead_id: HpkeAeadId,
        public_key: HpkePublicKey,
    ) -> HpkeConfig {
        HpkeConfig {
            id,
            kem_id,
            kdf_id,
            aead_id,
            public_key,
        }
    }

    /// Returns the HPKE config ID associated with this HPKE configuration.
    pub fn id(&self) -> &HpkeConfigId {
        &self.id
    }

    /// Retrieve the key encapsulation mechanism algorithm identifier associated with this HPKE configuration.
    pub fn kem_id(&self) -> &HpkeKemId {
        &self.kem_id
    }

    /// Retrieve the key derivation function algorithm identifier associated with this HPKE configuration.
    pub fn kdf_id(&self) -> &HpkeKdfId {
        &self.kdf_id
    }

    /// Retrieve the AEAD algorithm identifier associated with this HPKE configuration.
    pub fn aead_id(&self) -> &HpkeAeadId {
        &self.aead_id
    }

    /// Retrieve the public key from this HPKE configuration.
    pub fn public_key(&self) -> &HpkePublicKey {
        &self.public_key
    }
}

impl Encode for HpkeConfig {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.id.encode(bytes)?;
        self.kem_id.encode(bytes)?;
        self.kdf_id.encode(bytes)?;
        self.aead_id.encode(bytes)?;
        self.public_key.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.id.encoded_len()?
                + self.kem_id.encoded_len()?
                + self.kdf_id.encoded_len()?
                + self.aead_id.encoded_len()?
                + self.public_key.encoded_len()?,
        )
    }
}

impl Decode for HpkeConfig {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let id = HpkeConfigId::decode(bytes)?;
        let kem_id = HpkeKemId::decode(bytes)?;
        let kdf_id = HpkeKdfId::decode(bytes)?;
        let aead_id = HpkeAeadId::decode(bytes)?;
        let public_key = HpkePublicKey::decode(bytes)?;

        Ok(Self {
            id,
            kem_id,
            kdf_id,
            aead_id,
            public_key,
        })
    }
}

/// DAP protocol message representing a list of HPKE configurations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HpkeConfigList(Vec<HpkeConfig>);

impl HpkeConfigList {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-hpke-config-list";

    /// Construct an HPKE configuration list.
    pub fn new(hpke_configs: Vec<HpkeConfig>) -> Self {
        Self(hpke_configs)
    }

    pub fn hpke_configs(&self) -> &[HpkeConfig] {
        &self.0
    }
}

impl Encode for HpkeConfigList {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_u16_items(bytes, &(), &self.0)
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut length = 2;
        for hpke_config in self.0.iter() {
            length += hpke_config.encoded_len()?;
        }
        Some(length)
    }
}

impl Decode for HpkeConfigList {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(decode_u16_items(&(), bytes)?))
    }
}

/// DAP protocol message representing client report metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReportMetadata {
    report_id: ReportId,
    time: Time,
    public_extensions: Vec<Extension>,
}

impl ReportMetadata {
    /// Construct a report's metadata from its components.
    pub fn new(report_id: ReportId, time: Time, public_extensions: Vec<Extension>) -> Self {
        Self {
            report_id,
            time,
            public_extensions,
        }
    }

    /// Retrieve the report ID from this report metadata.
    pub fn id(&self) -> &ReportId {
        &self.report_id
    }

    /// Retrieve the client timestamp from this report metadata.
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Retrieve the public extensions from this report metadata.
    pub fn public_extensions(&self) -> &[Extension] {
        &self.public_extensions
    }
}

impl Encode for ReportMetadata {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.report_id.encode(bytes)?;
        self.time.encode(bytes)?;
        encode_u16_items(bytes, &(), &self.public_extensions)
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut len = self.report_id.encoded_len()? + self.time.encoded_len()? + 2;
        for extension in &self.public_extensions {
            len += extension.encoded_len()?;
        }
        Some(len)
    }
}

impl Decode for ReportMetadata {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let report_id = ReportId::decode(bytes)?;
        let time = Time::decode(bytes)?;
        let public_extensions = decode_u16_items(&(), bytes)?;

        Ok(Self {
            report_id,
            time,
            public_extensions,
        })
    }
}

/// DAP protocol message representing the plaintext of an input share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaintextInputShare {
    private_extensions: Vec<Extension>,
    payload: Vec<u8>,
}

impl PlaintextInputShare {
    /// Construct a plaintext input share from its components.
    pub fn new(private_extensions: Vec<Extension>, payload: Vec<u8>) -> Self {
        Self {
            private_extensions,
            payload,
        }
    }

    /// Retrieve the private extensions from this plaintext input share.
    pub fn private_extensions(&self) -> &[Extension] {
        &self.private_extensions
    }

    /// Retrieve the payload from this plaintext input share.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

impl Encode for PlaintextInputShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_u16_items(bytes, &(), &self.private_extensions)?;
        encode_u32_items(bytes, &(), &self.payload)
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut len = 2;
        for extension in &self.private_extensions {
            len += extension.encoded_len()?;
        }
        len += 4;
        len += self.payload.len();
        Some(len)
    }
}

impl Decode for PlaintextInputShare {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let private_extensions = decode_u16_items(&(), bytes)?;
        let payload = decode_u32_items(&(), bytes)?;

        Ok(Self {
            private_extensions,
            payload,
        })
    }
}

/// DAP protocol message representing a client report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    metadata: ReportMetadata,
    public_share: Vec<u8>,
    leader_encrypted_input_share: HpkeCiphertext,
    helper_encrypted_input_share: HpkeCiphertext,
}

impl Report {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-report";

    /// Construct a report from its components.
    pub fn new(
        metadata: ReportMetadata,
        public_share: Vec<u8>,
        leader_encrypted_input_share: HpkeCiphertext,
        helper_encrypted_input_share: HpkeCiphertext,
    ) -> Self {
        Self {
            metadata,
            public_share,
            leader_encrypted_input_share,
            helper_encrypted_input_share,
        }
    }

    /// Retrieve the metadata from this report.
    pub fn metadata(&self) -> &ReportMetadata {
        &self.metadata
    }

    /// Retrieve the public share from this report.
    pub fn public_share(&self) -> &[u8] {
        &self.public_share
    }

    /// Retrieve the encrypted leader input share from this report.
    pub fn leader_encrypted_input_share(&self) -> &HpkeCiphertext {
        &self.leader_encrypted_input_share
    }

    /// Retrieve the encrypted helper input share from this report.
    pub fn helper_encrypted_input_share(&self) -> &HpkeCiphertext {
        &self.helper_encrypted_input_share
    }
}

impl Encode for Report {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.metadata.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.public_share)?;
        self.leader_encrypted_input_share.encode(bytes)?;
        self.helper_encrypted_input_share.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut length = self.metadata.encoded_len()?;
        length += 4;
        length += self.public_share.len();
        length += self.leader_encrypted_input_share.encoded_len()?;
        length += self.helper_encrypted_input_share.encoded_len()?;
        Some(length)
    }
}

impl Decode for Report {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let metadata = ReportMetadata::decode(bytes)?;
        let public_share = decode_u32_items(&(), bytes)?;
        let leader_encrypted_input_share = HpkeCiphertext::decode(bytes)?;
        let helper_encrypted_input_share = HpkeCiphertext::decode(bytes)?;

        Ok(Self {
            metadata,
            public_share,
            leader_encrypted_input_share,
            helper_encrypted_input_share,
        })
    }
}

/// Represents a query for a specific batch identifier, received from a Collector as part of the
/// collection flow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query<B: BatchMode> {
    query_body: B::QueryBody,
}

impl<B: BatchMode> Query<B> {
    /// Constructs a new query from its components.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call one of [`Self::new_time_interval`] or
    /// [`Self::new_leader_selected`].
    pub fn new(query_body: B::QueryBody) -> Self {
        Self { query_body }
    }

    /// Gets the query body included in this query.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call [`Self::batch_interval`].
    pub fn query_body(&self) -> &B::QueryBody {
        &self.query_body
    }
}

impl Query<TimeInterval> {
    /// Constructs a new query for a time-interval task.
    pub fn new_time_interval(batch_interval: Interval) -> Self {
        Self::new(batch_interval)
    }

    /// Gets the batch interval associated with this query.
    pub fn batch_interval(&self) -> &Interval {
        self.query_body()
    }
}

impl Query<LeaderSelected> {
    /// Constructs a new query for a leader-selected task.
    pub fn new_leader_selected() -> Self {
        Self::new(())
    }
}

impl<B: BatchMode> Encode for Query<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        B::CODE.encode(bytes)?;
        encode_u16_items(bytes, &(), slice::from_ref(&self.query_body))
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(1 + 2 + self.query_body.encoded_len()?)
    }
}

impl<B: BatchMode> Decode for Query<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        batch_mode::Code::decode_expecting_value(bytes, B::CODE)?;

        let buf = decode_u16_items(&(), bytes)?;
        let query_body = B::QueryBody::get_decoded(&buf)?;

        Ok(Self { query_body })
    }
}

/// DAP protocol message representing a request from the collector to the leader to provide
/// aggregate shares for a given batch.
#[derive(Clone, Educe, PartialEq, Eq)]
#[educe(Debug)]
pub struct CollectionJobReq<B: BatchMode> {
    query: Query<B>,
    #[educe(Debug(ignore))]
    aggregation_parameter: Vec<u8>,
}

impl<B: BatchMode> CollectionJobReq<B> {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-collection-job-req";

    /// Constructs a new collect request from its components.
    pub fn new(query: Query<B>, aggregation_parameter: Vec<u8>) -> Self {
        Self {
            query,
            aggregation_parameter,
        }
    }

    /// Gets the query associated with this collect request.
    pub fn query(&self) -> &Query<B> {
        &self.query
    }

    /// Gets the aggregation parameter associated with this collect request.
    pub fn aggregation_parameter(&self) -> &[u8] {
        &self.aggregation_parameter
    }
}

impl<B: BatchMode> Encode for CollectionJobReq<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.query.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.aggregation_parameter)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.query.encoded_len()? + 4 + self.aggregation_parameter.len())
    }
}

impl<B: BatchMode> Decode for CollectionJobReq<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let query = Query::decode(bytes)?;
        let aggregation_parameter = decode_u32_items(&(), bytes)?;

        Ok(Self {
            query,
            aggregation_parameter,
        })
    }
}

/// DAP protocol message representing a partial batch selector, identifying a batch of interest in
/// cases where some batch modes can infer the selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialBatchSelector<B: BatchMode> {
    batch_identifier: B::PartialBatchIdentifier,
}

impl<B: BatchMode> PartialBatchSelector<B> {
    /// Constructs a new partial batch selector.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call one of [`Self::new_time_interval`] or
    /// [`Self::new_leader_selected`].
    pub fn new(batch_identifier: B::PartialBatchIdentifier) -> Self {
        Self { batch_identifier }
    }

    /// Gets the batch identifier associated with this collect response.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call [`Self::batch_id`].
    pub fn batch_identifier(&self) -> &B::PartialBatchIdentifier {
        &self.batch_identifier
    }
}

impl PartialBatchSelector<TimeInterval> {
    /// Constructs a new partial batch selector for a time-interval task.
    pub fn new_time_interval() -> Self {
        Self::new(())
    }
}

impl PartialBatchSelector<LeaderSelected> {
    /// Constructs a new partial batch selector for a leader-selected task.
    pub fn new_leader_selected(batch_id: BatchId) -> Self {
        Self::new(batch_id)
    }

    /// Gets the batch ID associated with this partial batch selector.
    pub fn batch_id(&self) -> &BatchId {
        self.batch_identifier()
    }
}

impl<B: BatchMode> Encode for PartialBatchSelector<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        B::CODE.encode(bytes)?;
        encode_u16_items(bytes, &(), slice::from_ref(&self.batch_identifier))
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(1 + 2 + self.batch_identifier.encoded_len()?)
    }
}

impl<B: BatchMode> Decode for PartialBatchSelector<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        batch_mode::Code::decode_expecting_value(bytes, B::CODE)?;

        let buf = decode_u16_items(&(), bytes)?;
        let batch_identifier = B::PartialBatchIdentifier::get_decoded(&buf)?;

        Ok(Self { batch_identifier })
    }
}

/// DAP protocol message representing an identifier for a collection.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CollectionJobId([u8; Self::LEN]);

impl CollectionJobId {
    /// LEN is the length of a collection ID in bytes.
    pub const LEN: usize = 16;
}

impl AsRef<[u8; Self::LEN]> for CollectionJobId {
    fn as_ref(&self) -> &[u8; Self::LEN] {
        &self.0
    }
}

impl TryFrom<&[u8]> for CollectionJobId {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            Error::InvalidParameter("byte slice has incorrect length for CollectionId")
        })?))
    }
}

impl FromStr for CollectionJobId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from(URL_SAFE_NO_PAD.decode(s)?.as_ref())
    }
}

impl Debug for CollectionJobId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CollectionJobId({})",
            Base64Display::new(&self.0, &URL_SAFE_NO_PAD)
        )
    }
}

impl Display for CollectionJobId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl Distribution<CollectionJobId> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> CollectionJobId {
        CollectionJobId(rng.gen())
    }
}

/// DAP protocol message representing a leader's response to the collector's request to provide
/// aggregate shares for a given query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollectionJobResp<B: BatchMode> {
    Processing,
    Finished {
        partial_batch_selector: PartialBatchSelector<B>,
        report_count: u64,
        interval: Interval,
        leader_encrypted_agg_share: HpkeCiphertext,
        helper_encrypted_agg_share: HpkeCiphertext,
    },
}

impl<B: BatchMode> CollectionJobResp<B> {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-collection-job-resp";
}

impl<B: BatchMode> Encode for CollectionJobResp<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        // The encoding includes an implicit discriminator byte called CollectionJobStatus in the
        // DAP specification.
        match self {
            Self::Processing => 0u8.encode(bytes),
            Self::Finished {
                partial_batch_selector,
                report_count,
                interval,
                leader_encrypted_agg_share,
                helper_encrypted_agg_share,
            } => {
                1u8.encode(bytes)?;
                partial_batch_selector.encode(bytes)?;
                report_count.encode(bytes)?;
                interval.encode(bytes)?;
                leader_encrypted_agg_share.encode(bytes)?;
                helper_encrypted_agg_share.encode(bytes)
            }
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(match self {
            Self::Processing => 1,
            Self::Finished {
                partial_batch_selector,
                report_count,
                interval,
                leader_encrypted_agg_share,
                helper_encrypted_agg_share,
            } => {
                1 + partial_batch_selector.encoded_len()?
                    + report_count.encoded_len()?
                    + interval.encoded_len()?
                    + leader_encrypted_agg_share.encoded_len()?
                    + helper_encrypted_agg_share.encoded_len()?
            }
        })
    }
}

impl<B: BatchMode> Decode for CollectionJobResp<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u8::decode(bytes)?;
        Ok(match val {
            0 => Self::Processing,
            1 => {
                let partial_batch_selector = PartialBatchSelector::decode(bytes)?;
                let report_count = u64::decode(bytes)?;
                let interval = Interval::decode(bytes)?;
                let leader_encrypted_agg_share = HpkeCiphertext::decode(bytes)?;
                let helper_encrypted_agg_share = HpkeCiphertext::decode(bytes)?;

                Self::Finished {
                    partial_batch_selector,
                    report_count,
                    interval,
                    leader_encrypted_agg_share,
                    helper_encrypted_agg_share,
                }
            }
            _ => return Err(CodecError::UnexpectedValue),
        })
    }
}

/// DAP message representing the additional associated data for an input share encryption operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputShareAad {
    task_id: TaskId,
    metadata: ReportMetadata,
    public_share: Vec<u8>,
}

impl InputShareAad {
    /// Constructs a new input share AAD.
    pub fn new(task_id: TaskId, metadata: ReportMetadata, public_share: Vec<u8>) -> Self {
        Self {
            task_id,
            metadata,
            public_share,
        }
    }

    /// Retrieves the task ID associated with this input share AAD.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Retrieves the report metadata associated with this input share AAD.
    pub fn metadata(&self) -> &ReportMetadata {
        &self.metadata
    }

    /// Retrieves the public share associated with this input share AAD.
    pub fn public_share(&self) -> &[u8] {
        &self.public_share
    }
}

impl Encode for InputShareAad {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.task_id.encode(bytes)?;
        self.metadata.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.public_share)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.task_id.encoded_len()?
                + self.metadata.encoded_len()?
                + 4
                + self.public_share.len(),
        )
    }
}

impl Decode for InputShareAad {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let task_id = TaskId::decode(bytes)?;
        let metadata = ReportMetadata::decode(bytes)?;
        let public_share = decode_u32_items(&(), bytes)?;

        Ok(Self {
            task_id,
            metadata,
            public_share,
        })
    }
}

/// DAP message representing the additional associated data for an aggregate share encryption
/// operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateShareAad<B: BatchMode> {
    task_id: TaskId,
    aggregation_parameter: Vec<u8>,
    batch_selector: BatchSelector<B>,
}

impl<B: BatchMode> AggregateShareAad<B> {
    /// Constructs a new aggregate share AAD.
    pub fn new(
        task_id: TaskId,
        aggregation_parameter: Vec<u8>,
        batch_selector: BatchSelector<B>,
    ) -> Self {
        Self {
            task_id,
            aggregation_parameter,
            batch_selector,
        }
    }

    /// Retrieves the task ID associated with this aggregate share AAD.
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Retrieves the aggregation parameter associated with this aggregate share AAD.
    pub fn aggregation_parameter(&self) -> &[u8] {
        &self.aggregation_parameter
    }

    /// Retrieves the batch selector associated with this aggregate share AAD.
    pub fn batch_selector(&self) -> &BatchSelector<B> {
        &self.batch_selector
    }
}

impl<B: BatchMode> Encode for AggregateShareAad<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.task_id.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.aggregation_parameter)?;
        self.batch_selector.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.task_id.encoded_len()?
                + 4
                + self.aggregation_parameter.len()
                + self.batch_selector.encoded_len()?,
        )
    }
}

impl<B: BatchMode> Decode for AggregateShareAad<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let task_id = TaskId::decode(bytes)?;
        let aggregation_parameter = decode_u32_items(&(), bytes)?;
        let batch_selector = BatchSelector::decode(bytes)?;

        Ok(Self {
            task_id,
            aggregation_parameter,
            batch_selector,
        })
    }
}

/// DAP protocol message representing one aggregator's share of a single client report.
#[derive(Educe, Clone, PartialEq, Eq)]
#[educe(Debug)]
pub struct ReportShare {
    metadata: ReportMetadata,
    #[educe(Debug(ignore))]
    public_share: Vec<u8>,
    encrypted_input_share: HpkeCiphertext,
}

impl ReportShare {
    /// Constructs a new report share from its components.
    pub fn new(
        metadata: ReportMetadata,
        public_share: Vec<u8>,
        encrypted_input_share: HpkeCiphertext,
    ) -> Self {
        Self {
            metadata,
            public_share,
            encrypted_input_share,
        }
    }

    /// Gets the metadata associated with this report share.
    pub fn metadata(&self) -> &ReportMetadata {
        &self.metadata
    }

    /// Gets the public share associated with this report share.
    pub fn public_share(&self) -> &[u8] {
        &self.public_share
    }

    /// Gets the encrypted input share associated with this report share.
    pub fn encrypted_input_share(&self) -> &HpkeCiphertext {
        &self.encrypted_input_share
    }
}

impl Encode for ReportShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.metadata.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.public_share)?;
        self.encrypted_input_share.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.metadata.encoded_len()?
                + 4
                + self.public_share.len()
                + self.encrypted_input_share.encoded_len()?,
        )
    }
}

impl Decode for ReportShare {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let metadata = ReportMetadata::decode(bytes)?;
        let public_share = decode_u32_items(&(), bytes)?;
        let encrypted_input_share = HpkeCiphertext::decode(bytes)?;

        Ok(Self {
            metadata,
            public_share,
            encrypted_input_share,
        })
    }
}

/// DAP protocol message representing information required to initialize preparation of a report for
/// aggregation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareInit {
    report_share: ReportShare,
    message: PingPongMessage,
}

impl PrepareInit {
    /// Constructs a new preparation initialization message from its components.
    pub fn new(report_share: ReportShare, message: PingPongMessage) -> Self {
        Self {
            report_share,
            message,
        }
    }

    /// Gets the report share associated with this prep init.
    pub fn report_share(&self) -> &ReportShare {
        &self.report_share
    }

    /// Gets the message associated with this prep init.
    pub fn message(&self) -> &PingPongMessage {
        &self.message
    }
}

impl Encode for PrepareInit {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.report_share.encode(bytes)?;
        let encoded_message = self.message.get_encoded()?;
        encode_u32_items(bytes, &(), &encoded_message)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.report_share.encoded_len()? + 4 + self.message.encoded_len()?)
    }
}

impl Decode for PrepareInit {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let report_share = ReportShare::decode(bytes)?;
        let message_bytes = decode_u32_items(&(), bytes)?;
        let message = PingPongMessage::get_decoded(&message_bytes)?;

        Ok(Self {
            report_share,
            message,
        })
    }
}

/// DAP protocol message representing the response to a preparation step in a VDAF evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareResp {
    report_id: ReportId,
    result: PrepareStepResult,
}

impl PrepareResp {
    /// Constructs a new prepare step from its components.
    pub fn new(report_id: ReportId, result: PrepareStepResult) -> Self {
        Self { report_id, result }
    }

    /// Gets the report ID associated with this prepare step.
    pub fn report_id(&self) -> &ReportId {
        &self.report_id
    }

    /// Gets the result associated with this prepare step.
    pub fn result(&self) -> &PrepareStepResult {
        &self.result
    }
}

impl Encode for PrepareResp {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.report_id.encode(bytes)?;
        self.result.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.report_id.encoded_len()? + self.result.encoded_len()?)
    }
}

impl Decode for PrepareResp {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let report_id = ReportId::decode(bytes)?;
        let result = PrepareStepResult::decode(bytes)?;

        Ok(Self { report_id, result })
    }
}

/// DAP protocol message representing result-type-specific data associated with a preparation step
/// in a VDAF evaluation. Included in a PrepareResp message.
#[derive(Clone, Educe, PartialEq, Eq)]
#[educe(Debug)]
pub enum PrepareStepResult {
    Continue {
        #[educe(Debug(ignore))]
        message: PingPongMessage,
    },
    Finished,
    Reject(ReportError),
}

impl Encode for PrepareStepResult {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        // The encoding includes an implicit discriminator byte, called PrepareStepResult in the
        // DAP spec.
        match self {
            Self::Continue { message: prep_msg } => {
                0u8.encode(bytes)?;
                let encoded_prep_msg = prep_msg.get_encoded()?;
                encode_u32_items(bytes, &(), &encoded_prep_msg)
            }
            Self::Finished => 1u8.encode(bytes),
            Self::Reject(error) => {
                2u8.encode(bytes)?;
                error.encode(bytes)
            }
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        match self {
            Self::Continue { message: prep_msg } => Some(1 + 4 + prep_msg.encoded_len()?),
            Self::Finished => Some(1),
            Self::Reject(error) => Some(1 + error.encoded_len()?),
        }
    }
}

impl Decode for PrepareStepResult {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u8::decode(bytes)?;
        Ok(match val {
            0 => {
                let prep_msg_bytes = decode_u32_items(&(), bytes)?;
                let prep_msg = PingPongMessage::get_decoded(&prep_msg_bytes)?;
                Self::Continue { message: prep_msg }
            }
            1 => Self::Finished,
            2 => Self::Reject(ReportError::decode(bytes)?),
            _ => return Err(CodecError::UnexpectedValue),
        })
    }
}

/// DAP protocol message representing an error while preparing a report share for aggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromPrimitive)]
#[repr(u8)]
pub enum ReportError {
    Reserved = 0,
    BatchCollected = 1,
    ReportReplayed = 2,
    ReportDropped = 3,
    HpkeUnknownConfigId = 4,
    HpkeDecryptError = 5,
    VdafPrepError = 6,
    TaskExpired = 7,
    InvalidMessage = 8,
    ReportTooEarly = 9,
    TaskNotStarted = 10,
}

impl Encode for ReportError {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        (*self as u8).encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(1)
    }
}

impl Decode for ReportError {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u8::decode(bytes)?;
        Self::try_from(val).map_err(|_| {
            CodecError::Other(anyhow!("unexpected ReportShareError value {}", val).into())
        })
    }
}

/// DAP protocol message representing a request to continue preparation of a report share for
/// aggregation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareContinue {
    report_id: ReportId,
    message: PingPongMessage,
}

impl PrepareContinue {
    /// Constructs a new prepare continue from its components.
    pub fn new(report_id: ReportId, message: PingPongMessage) -> Self {
        Self { report_id, message }
    }

    /// Gets the report ID associated with this prepare continue.
    pub fn report_id(&self) -> &ReportId {
        &self.report_id
    }

    /// Gets the message associated with this prepare continue.
    pub fn message(&self) -> &PingPongMessage {
        &self.message
    }
}

impl Encode for PrepareContinue {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.report_id.encode(bytes)?;
        let encoded_message = self.message.get_encoded()?;
        encode_u32_items(bytes, &(), &encoded_message)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(self.report_id.encoded_len()? + 4 + self.message.encoded_len()?)
    }
}

impl Decode for PrepareContinue {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let report_id = ReportId::decode(bytes)?;
        let message_bytes = decode_u32_items(&(), bytes)?;
        let message = PingPongMessage::get_decoded(&message_bytes)?;

        Ok(Self { report_id, message })
    }
}

/// DAP protocol message representing an identifier for an aggregation job.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AggregationJobId([u8; Self::LEN]);

impl AggregationJobId {
    /// LEN is the length of an aggregation job ID in bytes.
    pub const LEN: usize = 16;
}

impl From<[u8; Self::LEN]> for AggregationJobId {
    fn from(aggregation_job_id: [u8; Self::LEN]) -> Self {
        Self(aggregation_job_id)
    }
}

impl TryFrom<&[u8]> for AggregationJobId {
    type Error = Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        Ok(Self(value.try_into().map_err(|_| {
            Error::InvalidParameter("byte slice has incorrect length for AggregationJobId")
        })?))
    }
}

impl AsRef<[u8; Self::LEN]> for AggregationJobId {
    fn as_ref(&self) -> &[u8; Self::LEN] {
        &self.0
    }
}

impl FromStr for AggregationJobId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from(URL_SAFE_NO_PAD.decode(s)?.as_ref())
    }
}

impl Debug for AggregationJobId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AggregationJobId({})",
            Base64Display::new(&self.0, &URL_SAFE_NO_PAD)
        )
    }
}

impl Display for AggregationJobId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Base64Display::new(&self.0, &URL_SAFE_NO_PAD))
    }
}

impl Distribution<AggregationJobId> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> AggregationJobId {
        AggregationJobId(rng.gen())
    }
}

/// DAP protocol message representing an aggregation job initialization request from leader to
/// helper.
#[derive(Clone, Educe, PartialEq, Eq)]
#[educe(Debug)]
pub struct AggregationJobInitializeReq<B: BatchMode> {
    #[educe(Debug(ignore))]
    aggregation_parameter: Vec<u8>,
    partial_batch_selector: PartialBatchSelector<B>,
    prepare_inits: Vec<PrepareInit>,
}

impl<B: BatchMode> AggregationJobInitializeReq<B> {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-aggregation-job-init-req";

    /// Constructs an aggregate initialization request from its components.
    pub fn new(
        aggregation_parameter: Vec<u8>,
        partial_batch_selector: PartialBatchSelector<B>,
        prepare_inits: Vec<PrepareInit>,
    ) -> Self {
        Self {
            aggregation_parameter,
            partial_batch_selector,
            prepare_inits,
        }
    }

    /// Gets the aggregation parameter associated with this aggregate initialization request.
    pub fn aggregation_parameter(&self) -> &[u8] {
        &self.aggregation_parameter
    }

    /// Gets the partial batch selector associated with this aggregate initialization request.
    pub fn batch_selector(&self) -> &PartialBatchSelector<B> {
        &self.partial_batch_selector
    }

    /// Gets the preparation initialization messages associated with this aggregate initialization
    /// request.
    pub fn prepare_inits(&self) -> &[PrepareInit] {
        &self.prepare_inits
    }
}

impl<B: BatchMode> Encode for AggregationJobInitializeReq<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        encode_u32_items(bytes, &(), &self.aggregation_parameter)?;
        self.partial_batch_selector.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.prepare_inits)
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut length = 4 + self.aggregation_parameter.len();
        length += self.partial_batch_selector.encoded_len()?;
        length += 4;
        for prepare_init in &self.prepare_inits {
            length += prepare_init.encoded_len()?;
        }
        Some(length)
    }
}

impl<B: BatchMode> Decode for AggregationJobInitializeReq<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let aggregation_parameter = decode_u32_items(&(), bytes)?;
        let partial_batch_selector = PartialBatchSelector::decode(bytes)?;
        let prepare_inits = decode_u32_items(&(), bytes)?;

        Ok(Self {
            aggregation_parameter,
            partial_batch_selector,
            prepare_inits,
        })
    }
}

/// Type representing the step of an aggregation job.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AggregationJobStep(u16);

impl AggregationJobStep {
    /// Construct a new [`AggregationJobStep`] representing the step after this one.
    pub fn increment(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl Display for AggregationJobStep {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Encode for AggregationJobStep {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.0.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.0.encoded_len()
    }
}

impl Decode for AggregationJobStep {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self(u16::decode(bytes)?))
    }
}

impl From<u16> for AggregationJobStep {
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl From<AggregationJobStep> for u16 {
    fn from(value: AggregationJobStep) -> Self {
        value.0
    }
}

impl TryFrom<i32> for AggregationJobStep {
    // This implementation is convenient for converting from the representation of a step in
    // PostgreSQL, where the smallest type that can store a u16 is `integer`, which is represented
    // as i32 in Rust.

    type Error = TryFromIntError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(AggregationJobStep(u16::try_from(value)?))
    }
}

/// DAP protocol message representing a request to continue an aggregation job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregationJobContinueReq {
    step: AggregationJobStep,
    prepare_continues: Vec<PrepareContinue>,
}

impl AggregationJobContinueReq {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-aggregation-job-continue-req";

    /// Constructs a new aggregate continuation response from its components.
    pub fn new(step: AggregationJobStep, prepare_continues: Vec<PrepareContinue>) -> Self {
        Self {
            step,
            prepare_continues,
        }
    }

    /// Gets the step this aggregation job is on.
    pub fn step(&self) -> AggregationJobStep {
        self.step
    }

    /// Gets the preparation continuation messages associated with this aggregate continuation
    /// request.
    pub fn prepare_continues(&self) -> &[PrepareContinue] {
        &self.prepare_continues
    }
}

impl Encode for AggregationJobContinueReq {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.step.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.prepare_continues)
    }

    fn encoded_len(&self) -> Option<usize> {
        let mut length = self.step.encoded_len()?;
        length += 4;
        for prepare_continue in self.prepare_continues.iter() {
            length += prepare_continue.encoded_len()?;
        }
        Some(length)
    }
}

impl Decode for AggregationJobContinueReq {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let step = AggregationJobStep::decode(bytes)?;
        let prepare_continues = decode_u32_items(&(), bytes)?;
        Ok(Self::new(step, prepare_continues))
    }
}

/// DAP protocol message representing the response to an aggregation job initialization or
/// continuation request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AggregationJobResp {
    Processing,
    Finished { prepare_resps: Vec<PrepareResp> },
}

impl AggregationJobResp {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-aggregation-job-resp";
}

impl Encode for AggregationJobResp {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        // The encoding includes an implicit discriminator byte called AggregationJobStatus in the
        // DAP specification.
        match self {
            Self::Processing => 0u8.encode(bytes),
            Self::Finished { prepare_resps } => {
                1u8.encode(bytes)?;
                encode_u32_items(bytes, &(), prepare_resps)
            }
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(match self {
            Self::Processing => 1,
            Self::Finished { prepare_resps } => {
                let mut len = 5;
                for prepare_resp in prepare_resps {
                    len += prepare_resp.encoded_len()?;
                }
                len
            }
        })
    }
}

impl Decode for AggregationJobResp {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let val = u8::decode(bytes)?;
        Ok(match val {
            0 => Self::Processing,
            1 => {
                let prepare_resps = decode_u32_items(&(), bytes)?;
                Self::Finished { prepare_resps }
            }
            _ => return Err(CodecError::UnexpectedValue),
        })
    }
}

/// DAP protocol message identifying a batch of interest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchSelector<B: BatchMode> {
    batch_identifier: B::BatchIdentifier,
}

impl<B: BatchMode> BatchSelector<B> {
    /// Constructs a new batch selector from its components.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call one of [`Self::new_time_interval`] or
    /// [`Self::new_leader_selected`].
    pub fn new(batch_identifier: B::BatchIdentifier) -> Self {
        Self { batch_identifier }
    }

    /// Gets the batch identifier associated with this batch selector.
    ///
    /// This method would typically be used for code which is generic over the batch mode. Batch
    /// mode-specific code will typically call one of [`Self::batch_interval`] or
    /// [`Self::batch_id`].
    pub fn batch_identifier(&self) -> &B::BatchIdentifier {
        &self.batch_identifier
    }
}

impl BatchSelector<TimeInterval> {
    /// Constructs a new batch selector for time-interval tasks.
    pub fn new_time_interval(batch_interval: Interval) -> Self {
        Self::new(batch_interval)
    }

    /// Gets the batch interval associated with this batch selector.
    pub fn batch_interval(&self) -> &Interval {
        self.batch_identifier()
    }
}

impl BatchSelector<LeaderSelected> {
    /// Constructs a new batch selector for leader-selected tasks.
    pub fn new_leader_selected(batch_id: BatchId) -> Self {
        Self::new(batch_id)
    }

    /// Gets the batch ID associated with this batch selector.
    pub fn batch_id(&self) -> &BatchId {
        self.batch_identifier()
    }
}

impl<B: BatchMode> Encode for BatchSelector<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        B::CODE.encode(bytes)?;
        encode_u16_items(bytes, &(), slice::from_ref(&self.batch_identifier))
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(1 + 2 + self.batch_identifier.encoded_len()?)
    }
}

impl<B: BatchMode> Decode for BatchSelector<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        batch_mode::Code::decode_expecting_value(bytes, B::CODE)?;

        let buf = decode_u16_items(&(), bytes)?;
        let batch_identifier = B::BatchIdentifier::get_decoded(&buf)?;

        Ok(Self { batch_identifier })
    }
}

/// DAP protocol message representing a request from the leader to a helper to provide an
/// encrypted aggregate of its share of data for a given batch interval.
#[derive(Clone, Educe, PartialEq, Eq)]
#[educe(Debug)]
pub struct AggregateShareReq<B: BatchMode> {
    batch_selector: BatchSelector<B>,
    #[educe(Debug(ignore))]
    aggregation_parameter: Vec<u8>,
    report_count: u64,
    checksum: ReportIdChecksum,
}

impl<B: BatchMode> AggregateShareReq<B> {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-aggregate-share-req";

    /// Constructs a new aggregate share request from its components.
    pub fn new(
        batch_selector: BatchSelector<B>,
        aggregation_parameter: Vec<u8>,
        report_count: u64,
        checksum: ReportIdChecksum,
    ) -> Self {
        Self {
            batch_selector,
            aggregation_parameter,
            report_count,
            checksum,
        }
    }

    /// Gets the batch selector associated with this aggregate share request.
    pub fn batch_selector(&self) -> &BatchSelector<B> {
        &self.batch_selector
    }

    /// Gets the aggregation parameter associated with this aggregate share request.
    pub fn aggregation_parameter(&self) -> &[u8] {
        &self.aggregation_parameter
    }

    /// Gets the report count associated with this aggregate share request.
    pub fn report_count(&self) -> u64 {
        self.report_count
    }

    /// Gets the checksum associated with this aggregate share request.
    pub fn checksum(&self) -> &ReportIdChecksum {
        &self.checksum
    }
}

impl<B: BatchMode> Encode for AggregateShareReq<B> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.batch_selector.encode(bytes)?;
        encode_u32_items(bytes, &(), &self.aggregation_parameter)?;
        self.report_count.encode(bytes)?;
        self.checksum.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        Some(
            self.batch_selector.encoded_len()?
                + 4
                + self.aggregation_parameter.len()
                + self.report_count.encoded_len()?
                + self.checksum.encoded_len()?,
        )
    }
}

impl<B: BatchMode> Decode for AggregateShareReq<B> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let batch_selector = BatchSelector::decode(bytes)?;
        let aggregation_parameter = decode_u32_items(&(), bytes)?;
        let report_count = u64::decode(bytes)?;
        let checksum = ReportIdChecksum::decode(bytes)?;

        Ok(Self {
            batch_selector,
            aggregation_parameter,
            report_count,
            checksum,
        })
    }
}

/// DAP protocol message representing a helper's response to the leader's request to provide an
/// encrypted aggregate of its share of data for a given batch interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateShare {
    encrypted_aggregate_share: HpkeCiphertext,
}

impl AggregateShare {
    /// The media type associated with this protocol message.
    pub const MEDIA_TYPE: &'static str = "application/dap-aggregate-share";

    /// Constructs a new aggregate share response from its components.
    pub fn new(encrypted_aggregate_share: HpkeCiphertext) -> Self {
        Self {
            encrypted_aggregate_share,
        }
    }

    /// Gets the encrypted aggregate share associated with this aggregate share response.
    pub fn encrypted_aggregate_share(&self) -> &HpkeCiphertext {
        &self.encrypted_aggregate_share
    }
}

impl Encode for AggregateShare {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.encrypted_aggregate_share.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.encrypted_aggregate_share.encoded_len()
    }
}

impl Decode for AggregateShare {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let encrypted_aggregate_share = HpkeCiphertext::decode(bytes)?;

        Ok(Self {
            encrypted_aggregate_share,
        })
    }
}

#[cfg(test)]
pub(crate) fn roundtrip_encoding<T>(vals_and_encodings: &[(T, &str)])
where
    T: Encode + Decode + Debug + Eq,
{
    struct Wrapper<T>(T);

    impl<T: PartialEq> PartialEq for Wrapper<T> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl<T: Eq> Eq for Wrapper<T> {}

    impl<T: Debug> Debug for Wrapper<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "{:02x?}", &self.0)
        }
    }

    for (val, hex_encoding) in vals_and_encodings {
        let mut encoded_val = Vec::new();
        val.encode(&mut encoded_val).unwrap();
        let expected = Wrapper(hex::decode(hex_encoding).unwrap());
        let encoded_val = Wrapper(encoded_val);
        pretty_assertions::assert_eq!(
            encoded_val,
            expected,
            "Couldn't roundtrip (encoded value differs): {val:?}"
        );
        let decoded_val = T::get_decoded(&encoded_val.0).unwrap();
        pretty_assertions::assert_eq!(
            &decoded_val,
            val,
            "Couldn't roundtrip (decoded value differs): {val:?}"
        );
        pretty_assertions::assert_eq!(
            encoded_val.0.len(),
            val.encoded_len().expect("No encoded length hint"),
            "Encoded length hint is incorrect: {val:?}"
        )
    }
}
