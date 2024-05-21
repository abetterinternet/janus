//! A [DAP](https://datatracker.ietf.org/doc/draft-ietf-ppm-dap/) client
//!
//! This library implements the client role of the DAP-PPM protocol. It uploads measurements to two
//! DAP aggregator servers which in turn compute a statistical aggregate over data from many
//! clients, while preserving the privacy of each client's data.
//!
//! # Examples
//!
//! ```no_run
//! use url::Url;
//! use prio::vdaf::prio3::Prio3Histogram;
//! use janus_messages::{Duration, TaskId};
//! use std::str::FromStr;
//!
//! #[tokio::main]
//! async fn main() {
//!     let leader_url = Url::parse("https://leader.example.com/").unwrap();
//!     let helper_url = Url::parse("https://helper.example.com/").unwrap();
//!     let vdaf = Prio3Histogram::new_histogram(
//!         2,
//!         12,
//!         4
//!     ).unwrap();
//!     let taskid = "rc0jgm1MHH6Q7fcI4ZdNUxas9DAYLcJFK5CL7xUl-gU";
//!     let task = TaskId::from_str(taskid).unwrap();
//!
//!     let client = janus_client::Client::new(
//!         task,
//!         leader_url,
//!         helper_url,
//!         Duration::from_seconds(300),
//!         vdaf
//!     )
//!     .await
//!     .unwrap();
//!     client.upload(&5).await.unwrap();
//! }
//! ```

use backoff::ExponentialBackoff;
use derivative::Derivative;
use http::header::CONTENT_TYPE;
use itertools::Itertools;
use janus_core::{
    hpke::{self, is_hpke_config_supported, HpkeApplicationInfo, Label},
    http::HttpErrorResponse,
    retries::{http_request_exponential_backoff, retry_http_request},
    time::{Clock, RealClock, TimeExt},
    url_ensure_trailing_slash,
};
use janus_messages::{
    Duration, HpkeConfig, HpkeConfigList, InputShareAad, PlaintextInputShare, Report, ReportId,
    ReportMetadata, Role, TaskId, Time,
};
use prio::{
    codec::{Decode, Encode},
    vdaf,
};
use rand::random;
use std::{convert::Infallible, fmt::Debug, time::SystemTimeError};
use tokio::try_join;
use url::Url;

#[cfg(test)]
mod tests;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid parameter {0}")]
    InvalidParameter(&'static str),
    #[error("HTTP client error: {0}")]
    HttpClient(#[from] reqwest::Error),
    #[error("codec error: {0}")]
    Codec(#[from] prio::codec::CodecError),
    #[error("HTTP response status {0}")]
    Http(Box<HttpErrorResponse>),
    #[error("URL parse: {0}")]
    Url(#[from] url::ParseError),
    #[error("VDAF error: {0}")]
    Vdaf(#[from] prio::vdaf::VdafError),
    #[error("HPKE error: {0}")]
    Hpke(#[from] janus_core::hpke::Error),
    #[error("unexpected server response {0}")]
    UnexpectedServerResponse(&'static str),
    #[error("time conversion error: {0}")]
    TimeConversion(#[from] SystemTimeError),
}

impl From<Infallible> for Error {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

static CLIENT_USER_AGENT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
    "/",
    "client"
);

/// The DAP client's view of task parameters.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
struct ClientParameters {
    /// Unique identifier for the task.
    task_id: TaskId,
    /// URL relative to which the Leader's API endpoints are found.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    leader_aggregator_endpoint: Url,
    /// URL relative to which the Helper's API endpoints are found.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    helper_aggregator_endpoint: Url,
    /// The time precision of the task. This value is shared by all parties in the protocol, and is
    /// used to compute report timestamps.
    time_precision: Duration,
    /// Parameters to use when retrying HTTP requests.
    http_request_retry_parameters: ExponentialBackoff,
}

impl ClientParameters {
    /// Creates a new set of client task parameters.
    pub fn new(
        task_id: TaskId,
        leader_aggregator_endpoint: Url,
        helper_aggregator_endpoint: Url,
        time_precision: Duration,
    ) -> Self {
        Self {
            task_id,
            leader_aggregator_endpoint: url_ensure_trailing_slash(leader_aggregator_endpoint),
            helper_aggregator_endpoint: url_ensure_trailing_slash(helper_aggregator_endpoint),
            time_precision,
            http_request_retry_parameters: http_request_exponential_backoff(),
        }
    }

    /// The URL relative to which the API endpoints for the aggregator may be found, if the role is
    /// an aggregator, or an error otherwise.
    fn aggregator_endpoint(&self, role: &Role) -> Result<&Url, Error> {
        match role {
            Role::Leader => Ok(&self.leader_aggregator_endpoint),
            Role::Helper => Ok(&self.helper_aggregator_endpoint),
            _ => Err(Error::InvalidParameter("role is not an aggregator")),
        }
    }

    /// URL from which the HPKE configuration for the server filling `role` may be fetched, per
    /// the [DAP specification][1].
    ///
    /// [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#name-hpke-configuration-request
    fn hpke_config_endpoint(&self, role: &Role) -> Result<Url, Error> {
        Ok(self.aggregator_endpoint(role)?.join("hpke_config")?)
    }

    // URI to which reports may be uploaded for the provided task.
    fn reports_resource_uri(&self, task_id: &TaskId) -> Result<Url, Error> {
        Ok(self
            .leader_aggregator_endpoint
            .join(&format!("tasks/{task_id}/reports"))?)
    }
}

/// Fetches HPKE configuration from the specified aggregator using the aggregator endpoints in the
/// provided [`ClientParameters`].
#[tracing::instrument(err)]
async fn aggregator_hpke_config(
    client_parameters: &ClientParameters,
    aggregator_role: &Role,
    http_client: &reqwest::Client,
) -> Result<HpkeConfig, Error> {
    let mut request_url = client_parameters.hpke_config_endpoint(aggregator_role)?;
    request_url.set_query(Some(&format!("task_id={}", client_parameters.task_id)));
    let hpke_config_response = retry_http_request(
        client_parameters.http_request_retry_parameters.clone(),
        || async { http_client.get(request_url.clone()).send().await },
    )
    .await
    .map_err(|err| match err {
        Ok(http_error_response) => Error::Http(Box::new(http_error_response)),
        Err(error) => error.into(),
    })?;
    let status = hpke_config_response.status();
    if !status.is_success() {
        return Err(Error::Http(Box::new(HttpErrorResponse::from(status))));
    }

    let hpke_configs = HpkeConfigList::get_decoded(hpke_config_response.body())?;

    if hpke_configs.hpke_configs().is_empty() {
        return Err(Error::UnexpectedServerResponse(
            "aggregator provided empty HpkeConfigList",
        ));
    }

    // Take the first supported HpkeConfig from the list. Return the first error otherwise.
    let mut first_error = None;
    for config in hpke_configs.hpke_configs() {
        match is_hpke_config_supported(config) {
            Ok(()) => return Ok(config.clone()),
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }
    }
    // Unwrap safety: we checked that the list is nonempty, and if we fell through to here, we must
    // have seen at least one error.
    Err(first_error.unwrap().into())
}

/// Construct a [`reqwest::Client`] suitable for use in a DAP [`Client`].
pub fn default_http_client() -> Result<reqwest::Client, Error> {
    Ok(reqwest::Client::builder()
        // Clients wishing to override these timeouts may provide their own
        // values using ClientBuilder::with_http_client.
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(10))
        .user_agent(CLIENT_USER_AGENT)
        .build()?)
}

/// Builder for configuring a [`Client`].
pub struct ClientBuilder<V: vdaf::Client<16>> {
    parameters: ClientParameters,
    vdaf: V,
    http_client: Option<reqwest::Client>,
}

impl<V: vdaf::Client<16>> ClientBuilder<V> {
    /// Construct a [`ClientBuilder`] from its required DAP task parameters.
    pub fn new(
        task_id: TaskId,
        leader_aggregator_endpoint: Url,
        helper_aggregator_endpoint: Url,
        time_precision: Duration,
        vdaf: V,
    ) -> Self {
        Self {
            parameters: ClientParameters::new(
                task_id,
                leader_aggregator_endpoint,
                helper_aggregator_endpoint,
                time_precision,
            ),
            vdaf,
            http_client: None,
        }
    }

    /// Finalize construction of a [`Client`]. This will fetch HPKE configurations from each
    /// aggregator via HTTPS.
    pub async fn build(self) -> Result<Client<V>, Error> {
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            default_http_client()?
        };
        let (leader_hpke_config, helper_hpke_config) = try_join!(
            aggregator_hpke_config(&self.parameters, &Role::Leader, &http_client),
            aggregator_hpke_config(&self.parameters, &Role::Helper, &http_client)
        )?;
        Ok(Client {
            parameters: self.parameters,
            vdaf: self.vdaf,
            http_client,
            leader_hpke_config,
            helper_hpke_config,
        })
    }

    /// Finalize construction of a [`Client`], and provide aggregator HPKE configurations through an
    /// out-of-band mechanism.
    pub fn build_with_hpke_configs(
        self,
        leader_hpke_config: HpkeConfig,
        helper_hpke_config: HpkeConfig,
    ) -> Result<Client<V>, Error> {
        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            default_http_client()?
        };
        Ok(Client {
            parameters: self.parameters,
            vdaf: self.vdaf,
            http_client,
            leader_hpke_config,
            helper_hpke_config,
        })
    }

    /// Override the HTTPS client configuration to be used.
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = Some(http_client);
        self
    }

    /// Override the exponential backoff parameters used when retrying HTTPS requests.
    pub fn with_backoff(mut self, http_request_retry_parameters: ExponentialBackoff) -> Self {
        self.parameters.http_request_retry_parameters = http_request_retry_parameters;
        self
    }
}

/// A DAP client.
#[derive(Clone, Debug)]
pub struct Client<V: vdaf::Client<16>> {
    parameters: ClientParameters,
    vdaf: V,
    http_client: reqwest::Client,
    leader_hpke_config: HpkeConfig,
    helper_hpke_config: HpkeConfig,
}

impl<V: vdaf::Client<16>> Client<V> {
    /// Construct a new client from the required set of DAP task parameters.
    pub async fn new(
        task_id: TaskId,
        leader_aggregator_endpoint: Url,
        helper_aggregator_endpoint: Url,
        time_precision: Duration,
        vdaf: V,
    ) -> Result<Self, Error> {
        ClientBuilder::new(
            task_id,
            leader_aggregator_endpoint,
            helper_aggregator_endpoint,
            time_precision,
            vdaf,
        )
        .build()
        .await
    }

    /// Construct a new client, and provide the aggregator HPKE configurations through an
    /// out-of-band means.
    pub fn with_hpke_configs(
        task_id: TaskId,
        leader_aggregator_endpoint: Url,
        helper_aggregator_endpoint: Url,
        time_precision: Duration,
        vdaf: V,
        leader_hpke_config: HpkeConfig,
        helper_hpke_config: HpkeConfig,
    ) -> Result<Self, Error> {
        ClientBuilder::new(
            task_id,
            leader_aggregator_endpoint,
            helper_aggregator_endpoint,
            time_precision,
            vdaf,
        )
        .build_with_hpke_configs(leader_hpke_config, helper_hpke_config)
    }

    /// Creates a [`ClientBuilder`] for further configuration from the required set of DAP task
    /// parameters.
    pub fn builder(
        task_id: TaskId,
        leader_aggregator_endpoint: Url,
        helper_aggregator_endpoint: Url,
        time_precision: Duration,
        vdaf: V,
    ) -> ClientBuilder<V> {
        ClientBuilder::new(
            task_id,
            leader_aggregator_endpoint,
            helper_aggregator_endpoint,
            time_precision,
            vdaf,
        )
    }

    /// Shard a measurement, encrypt its shares, and construct a [`janus_messages::Report`] to be
    /// uploaded.
    fn prepare_report(&self, measurement: &V::Measurement, time: &Time) -> Result<Report, Error> {
        let report_id: ReportId = random();
        let (public_share, input_shares) = self.vdaf.shard(measurement, report_id.as_ref())?;
        assert_eq!(input_shares.len(), 2); // DAP only supports VDAFs using two aggregators.

        let time = time
            .to_batch_interval_start(&self.parameters.time_precision)
            .map_err(|_| Error::InvalidParameter("couldn't round time down to time_precision"))?;
        let report_metadata = ReportMetadata::new(report_id, time);
        let encoded_public_share = public_share.get_encoded()?;

        let (leader_encrypted_input_share, helper_encrypted_input_share) = [
            (&self.leader_hpke_config, &Role::Leader),
            (&self.helper_hpke_config, &Role::Helper),
        ]
        .into_iter()
        .zip(input_shares)
        .map(|((hpke_config, receiver_role), input_share)| {
            hpke::seal(
                hpke_config,
                &HpkeApplicationInfo::new(&Label::InputShare, &Role::Client, receiver_role),
                &PlaintextInputShare::new(
                    Vec::new(), // No extensions supported yet.
                    input_share.get_encoded()?,
                )
                .get_encoded()?,
                &InputShareAad::new(
                    self.parameters.task_id,
                    report_metadata.clone(),
                    encoded_public_share.clone(),
                )
                .get_encoded()?,
            )
            .map_err(Error::Hpke)
        })
        .collect_tuple()
        .expect("iterator to yield two items"); // expect safety: iterator contains two items.

        Ok(Report::new(
            report_metadata,
            encoded_public_share,
            leader_encrypted_input_share?,
            helper_encrypted_input_share?,
        ))
    }

    /// Upload a [`Report`] to the leader, per the [DAP specification][1]. The provided measurement
    /// is sharded into two shares and then uploaded to the leader.
    ///
    /// [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#name-uploading-reports
    #[tracing::instrument(skip(measurement), err)]
    pub async fn upload(&self, measurement: &V::Measurement) -> Result<(), Error> {
        self.upload_with_time(measurement, Clock::now(&RealClock::default()))
            .await
    }

    /// Upload a [`Report`] to the leader, per the [DAP specification][1], and override the report's
    /// timestamp. The provided measurement is sharded into two shares and then uploaded to the
    /// leader.
    ///
    /// [1]: https://www.ietf.org/archive/id/draft-ietf-ppm-dap-07.html#name-uploading-reports
    ///
    /// ```no_run
    /// # use janus_client::{Client, Error};
    /// # use janus_messages::Duration;
    /// # use prio::vdaf::prio3::Prio3;
    /// # use rand::random;
    /// #
    /// # async fn test() -> Result<(), Error> {
    /// # let measurement = true;
    /// # let timestamp = 1_700_000_000;
    /// # let vdaf = Prio3::new_count(2).unwrap();
    /// let client = Client::new(
    ///     random(),
    ///     "https://example.com/".parse().unwrap(),
    ///     "https://example.net/".parse().unwrap(),
    ///     Duration::from_seconds(3600),
    ///     vdaf,
    /// ).await?;
    /// client.upload_with_time(&measurement, std::time::SystemTime::now()).await?;
    /// client.upload_with_time(&measurement, janus_messages::Time::from_seconds_since_epoch(timestamp)).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip(measurement), err)]
    pub async fn upload_with_time<T>(
        &self,
        measurement: &V::Measurement,
        time: T,
    ) -> Result<(), Error>
    where
        T: TryInto<Time> + Debug,
        Error: From<<T as TryInto<Time>>::Error>,
    {
        let report = self
            .prepare_report(measurement, &time.try_into()?)?
            .get_encoded()?;
        let upload_endpoint = self
            .parameters
            .reports_resource_uri(&self.parameters.task_id)?;
        let upload_response = retry_http_request(
            self.parameters.http_request_retry_parameters.clone(),
            || async {
                self.http_client
                    .put(upload_endpoint.clone())
                    .header(CONTENT_TYPE, Report::MEDIA_TYPE)
                    .body(report.clone())
                    .send()
                    .await
            },
        )
        .await
        .map_err(|err| match err {
            Ok(http_error_response) => Error::Http(Box::new(http_error_response)),
            Err(error) => error.into(),
        })?;

        let status = upload_response.status();
        if !status.is_success() {
            return Err(Error::Http(Box::new(HttpErrorResponse::from(status))));
        }

        Ok(())
    }
}
