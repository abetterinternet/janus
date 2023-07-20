//! A [DAP-PPM](https://datatracker.ietf.org/doc/draft-ietf-ppm-dap/) collector
//!
//! This library implements the collector role of the DAP-PPM protocol. It works in concert with
//! two DAP-PPM aggregator servers to compute a statistical aggregate over data from many clients,
//! while preserving the privacy of each client's data.
//!
//! # Examples
//!
//! ```no_run
//! use janus_collector::{AuthenticationToken, Collector, CollectorParameters, default_http_client};
//! use janus_core::{hpke::generate_hpke_config_and_private_key};
//! use janus_messages::{
//!     Duration, FixedSizeQuery, HpkeAeadId, HpkeConfig, HpkeConfigId, HpkeKdfId, HpkeKemId,
//!     Interval, TaskId, Time, Query
//! };
//! use prio::vdaf::prio3::Prio3;
//! use rand::random;
//! use url::Url;
//!
//! # async fn run() {
//! // Supply DAP task parameters.
//! let task_id = random();
//! let hpke_keypair = janus_core::hpke::generate_hpke_config_and_private_key(
//!     HpkeConfigId::from(0),
//!     HpkeKemId::X25519HkdfSha256,
//!     HpkeKdfId::HkdfSha256,
//!     HpkeAeadId::Aes128Gcm,
//! );
//! let parameters = CollectorParameters::new(
//!     task_id,
//!     "https://example.com/dap/".parse().unwrap(),
//!     AuthenticationToken::Bearer(b"my-authentication-token".to_vec()),
//!     hpke_keypair.config().clone(),
//!     hpke_keypair.private_key().clone(),
//! );
//!
//! // Supply a VDAF implementation, corresponding to this task.
//! let vdaf = Prio3::new_count(2).unwrap();
//! // Use the default HTTP client as-is.
//! let http_client = default_http_client().unwrap();
//! let collector = Collector::new(parameters, vdaf, http_client);
//!
//! // Make the request and retrieve the aggregated statistic.
//! let aggregation_result = collector.collect(Query::new_fixed_size(FixedSizeQuery::CurrentBatch), &()).await.unwrap();
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

use backoff::{backoff::Backoff, ExponentialBackoff};
use chrono::{DateTime, Duration, Utc};
use derivative::Derivative;
use http_api_problem::HttpApiProblem;
pub use janus_core::task::AuthenticationToken;
use janus_core::{
    hpke::{self, HpkeApplicationInfo, HpkePrivateKey},
    http::response_to_problem_details,
    retries::{http_request_exponential_backoff, retry_http_request},
    task::url_ensure_trailing_slash,
    time::{DurationExt, TimeExt},
};
use janus_messages::{
    problem_type::DapProblemType,
    query_type::{FixedSize, QueryType},
    AggregateShareAad, BatchSelector, Collection as CollectionMessage, CollectionJobId,
    CollectionReq, HpkeConfig, PartialBatchSelector, Query, Role, TaskId,
};
use prio::{
    codec::{Decode, Encode, ParameterizedDecode},
    vdaf,
};
use rand::random;
use reqwest::{
    header::{HeaderValue, ToStrError, CONTENT_TYPE, RETRY_AFTER},
    Response, StatusCode,
};
use retry_after::FromHeaderValueError;
use retry_after::RetryAfter;
use std::{
    convert::TryFrom,
    time::{Duration as StdDuration, SystemTime},
};
use tokio::time::{sleep, Instant};
use url::Url;

/// Errors that may occur when performing collections.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP client error: {0}")]
    HttpClient(#[from] reqwest::Error),
    #[error("HTTP response status {problem_details}")]
    Http {
        problem_details: Box<HttpApiProblem>,
        dap_problem_type: Option<DapProblemType>,
    },
    #[error("URL parse: {0}")]
    Url(#[from] url::ParseError),
    #[error("missing Location header in See Other response")]
    MissingLocationHeader,
    #[error("invalid bytes in header")]
    InvalidHeader(#[from] ToStrError),
    #[error("wrong Content-Type header: {0:?}")]
    BadContentType(Option<HeaderValue>),
    #[error("invalid Retry-After header value: {0}")]
    InvalidRetryAfterHeader(#[from] FromHeaderValueError),
    #[error("codec error: {0}")]
    Codec(#[from] prio::codec::CodecError),
    #[error("aggregate share decoding error")]
    AggregateShareDecode,
    #[error("expected two aggregate shares, got {0}")]
    AggregateShareCount(usize),
    #[error("VDAF error: {0}")]
    Vdaf(#[from] prio::vdaf::VdafError),
    #[error("HPKE error: {0}")]
    Hpke(#[from] janus_core::hpke::Error),
    #[error("timed out waiting for collection to finish")]
    CollectPollTimeout,
    #[error("report count was too large")]
    ReportCountOverflow,
    #[error("message error: {0}")]
    Message(#[from] janus_messages::Error),
}

impl Error {
    /// Construct an error from an HTTP response's status and problem details document, if present
    /// in the body.
    async fn from_http_response(response: Response) -> Error {
        let problem_details = response_to_problem_details(response).await;
        let dap_problem_type = problem_details
            .type_url
            .as_ref()
            .and_then(|str| str.parse::<DapProblemType>().ok());
        Error::Http {
            problem_details: Box::new(problem_details),
            dap_problem_type,
        }
    }
}

static COLLECTOR_USER_AGENT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
    "/",
    "collector"
);

/// The DAP collector's view of task parameters.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct CollectorParameters {
    /// Unique identifier for the task.
    task_id: TaskId,
    /// The base URL of the leader's aggregator API endpoints.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    leader_endpoint: Url,
    /// The authentication information needed to communicate with the leader aggregator.
    authentication: AuthenticationToken,
    /// HPKE configuration and public key used for encryption of aggregate shares.
    #[derivative(Debug = "ignore")]
    hpke_config: HpkeConfig,
    /// HPKE private key used to decrypt aggregate shares.
    #[derivative(Debug = "ignore")]
    hpke_private_key: HpkePrivateKey,
    /// Parameters to use when retrying HTTP requests.
    #[derivative(Debug = "ignore")]
    http_request_retry_parameters: ExponentialBackoff,
    /// Parameters to use when waiting for a collection job to be processed.
    #[derivative(Debug = "ignore")]
    collect_poll_wait_parameters: ExponentialBackoff,
}

impl CollectorParameters {
    /// Creates a new set of collector task parameters.
    pub fn new(
        task_id: TaskId,
        mut leader_endpoint: Url,
        authentication: AuthenticationToken,
        hpke_config: HpkeConfig,
        hpke_private_key: HpkePrivateKey,
    ) -> CollectorParameters {
        // Ensure the provided leader endpoint ends with a slash.
        url_ensure_trailing_slash(&mut leader_endpoint);

        CollectorParameters {
            task_id,
            leader_endpoint,
            authentication,
            hpke_config,
            hpke_private_key,
            http_request_retry_parameters: http_request_exponential_backoff(),
            collect_poll_wait_parameters: ExponentialBackoff {
                initial_interval: StdDuration::from_secs(15),
                max_interval: StdDuration::from_secs(300),
                multiplier: 1.2,
                max_elapsed_time: None,
                ..Default::default()
            },
        }
    }

    /// Replace the exponential backoff settings used for HTTP requests.
    pub fn with_http_request_backoff(mut self, backoff: ExponentialBackoff) -> CollectorParameters {
        self.http_request_retry_parameters = backoff;
        self
    }

    /// Replace the exponential backoff settings used while polling for aggregate shares.
    pub fn with_collect_poll_backoff(mut self, backoff: ExponentialBackoff) -> CollectorParameters {
        self.collect_poll_wait_parameters = backoff;
        self
    }

    /// Construct a URI for a collection.
    fn collection_job_uri(&self, collection_job_id: CollectionJobId) -> Result<Url, Error> {
        Ok(self.leader_endpoint.join(&format!(
            "tasks/{}/collection_jobs/{collection_job_id}",
            self.task_id
        ))?)
    }
}

/// Construct a [`reqwest::Client`] suitable for use in a DAP [`Collector`].
pub fn default_http_client() -> Result<reqwest::Client, Error> {
    Ok(reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(COLLECTOR_USER_AGENT)
        .build()?)
}

/// Collector state related to a collection job that is in progress.
#[derive(Derivative)]
#[derivative(Debug)]
struct CollectionJob<P, Q>
where
    Q: QueryType,
{
    /// The URL provided by the leader aggregator, where the collect response will be available
    /// upon completion.
    #[derivative(Debug(format_with = "std::fmt::Display::fmt"))]
    collection_job_url: Url,
    /// The collect request's query.
    query: Query<Q>,
    /// The aggregation parameter used in this collect request.
    #[derivative(Debug = "ignore")]
    aggregation_parameter: P,
}

impl<P, Q: QueryType> CollectionJob<P, Q> {
    fn new(
        collection_job_url: Url,
        query: Query<Q>,
        aggregation_parameter: P,
    ) -> CollectionJob<P, Q> {
        CollectionJob {
            collection_job_url,
            query,
            aggregation_parameter,
        }
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
/// The result of a collect request poll operation. This will either provide the collection result
/// or indicate that the collection is still being processed.
enum PollResult<T, Q>
where
    Q: QueryType,
{
    /// The collection result from a completed collect request.
    CollectionResult(#[derivative(Debug = "ignore")] Collection<T, Q>),
    /// The collect request is not yet ready. If present, the [`RetryAfter`] object is the time at
    /// which the leader recommends retrying the request.
    NextAttempt(Option<RetryAfter>),
}

/// The result of a collection operation.
#[derive(Debug)]
pub struct Collection<T, Q>
where
    Q: QueryType,
{
    partial_batch_selector: PartialBatchSelector<Q>,
    report_count: u64,
    interval: (DateTime<Utc>, Duration),
    aggregate_result: T,
}

impl<T, Q> Collection<T, Q>
where
    Q: QueryType,
{
    /// Retrieves the partial batch selector of this collection.
    pub fn partial_batch_selector(&self) -> &PartialBatchSelector<Q> {
        &self.partial_batch_selector
    }

    /// Retrieves the number of client reports included in this collection.
    pub fn report_count(&self) -> u64 {
        self.report_count
    }

    /// Retrieves the interval of time spanned by the reports included in this collection.
    pub fn interval(&self) -> &(DateTime<Utc>, Duration) {
        &self.interval
    }

    /// Retrieves the aggregated result of the client reports included in this collection.
    pub fn aggregate_result(&self) -> &T {
        &self.aggregate_result
    }
}

#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
impl<T, Q> Collection<T, Q>
where
    Q: QueryType,
{
    /// Creates a new [`Collection`].
    pub fn new(
        partial_batch_selector: PartialBatchSelector<Q>,
        report_count: u64,
        interval: (DateTime<Utc>, Duration),
        aggregate_result: T,
    ) -> Self {
        Self {
            partial_batch_selector,
            report_count,
            interval,
            aggregate_result,
        }
    }
}

impl<T, Q> PartialEq for Collection<T, Q>
where
    T: PartialEq,
    Q: QueryType,
{
    fn eq(&self, other: &Self) -> bool {
        self.partial_batch_selector == other.partial_batch_selector
            && self.report_count == other.report_count
            && self.interval == other.interval
            && self.aggregate_result == other.aggregate_result
    }
}

impl<T, Q> Eq for Collection<T, Q>
where
    T: Eq,
    Q: QueryType,
{
}

/// A DAP collector.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Collector<V: vdaf::Collector> {
    parameters: CollectorParameters,
    vdaf_collector: V,
    #[derivative(Debug = "ignore")]
    http_client: reqwest::Client,
}

impl<V: vdaf::Collector> Collector<V> {
    /// Construct a new collector. This requires certain DAP task parameters, an implementation of
    /// the task's VDAF, and a [`reqwest::Client`], configured to never follow redirects, that will
    /// be used to communicate with the leader aggregator.
    pub fn new(
        parameters: CollectorParameters,
        vdaf_collector: V,
        http_client: reqwest::Client,
    ) -> Collector<V> {
        Collector {
            parameters,
            vdaf_collector,
            http_client,
        }
    }

    /// Send a collect request to the leader aggregator.
    #[tracing::instrument(skip(aggregation_parameter), err)]
    async fn start_collection<Q: QueryType>(
        &self,
        query: Query<Q>,
        aggregation_parameter: &V::AggregationParam,
    ) -> Result<CollectionJob<V::AggregationParam, Q>, Error> {
        let collect_request =
            CollectionReq::new(query.clone(), aggregation_parameter.get_encoded());
        let collection_job_url = self.parameters.collection_job_uri(random())?;

        let response_res = retry_http_request(
            self.parameters.http_request_retry_parameters.clone(),
            || async {
                let (auth_header, auth_value) =
                    self.parameters.authentication.request_authentication();
                self.http_client
                    .put(collection_job_url.clone())
                    .header(CONTENT_TYPE, CollectionReq::<FixedSize>::MEDIA_TYPE)
                    .body(collect_request.get_encoded())
                    .header(auth_header, auth_value)
                    .send()
                    .await
            },
        )
        .await;

        match response_res {
            // Successful response or unretryable error status code:
            Ok(response) => {
                let status = response.status();
                if status.is_client_error() || status.is_server_error() {
                    return Err(Error::from_http_response(response).await);
                } else if status != StatusCode::CREATED {
                    // Incorrect success/redirect status code:
                    return Err(Error::Http {
                        problem_details: Box::new(HttpApiProblem::new(status)),
                        dap_problem_type: None,
                    });
                }
            }
            // Retryable error status code, but ran out of retries:
            Err(Ok(response)) => return Err(Error::from_http_response(response).await),
            // Lower level errors, either unretryable or ran out of retries:
            Err(Err(error)) => return Err(Error::HttpClient(error)),
        };

        Ok(CollectionJob::new(
            collection_job_url,
            query,
            aggregation_parameter.clone(),
        ))
    }

    /// Request the results of an in-progress collection from the leader aggregator. This may
    /// return `Ok(None)` if the aggregation is not done yet.
    #[tracing::instrument(err)]
    async fn poll_once<Q: QueryType>(
        &self,
        job: &CollectionJob<V::AggregationParam, Q>,
    ) -> Result<PollResult<V::AggregateResult, Q>, Error> {
        let response_res = retry_http_request(
            self.parameters.http_request_retry_parameters.clone(),
            || async {
                let (auth_header, auth_value) =
                    self.parameters.authentication.request_authentication();
                self.http_client
                    .post(job.collection_job_url.clone())
                    .header(auth_header, auth_value)
                    .send()
                    .await
            },
        )
        .await;

        let response = match response_res {
            // Successful response or unretryable error status code:
            Ok(response) => {
                let status = response.status();
                match status {
                    StatusCode::OK => response,
                    StatusCode::ACCEPTED => {
                        let retry_after_opt = response
                            .headers()
                            .get(RETRY_AFTER)
                            .map(RetryAfter::try_from)
                            .transpose()?;
                        return Ok(PollResult::NextAttempt(retry_after_opt));
                    }
                    _ if status.is_client_error() || status.is_server_error() => {
                        return Err(Error::from_http_response(response).await);
                    }
                    _ => {
                        return Err(Error::Http {
                            problem_details: Box::new(HttpApiProblem::new(status)),
                            dap_problem_type: None,
                        })
                    }
                }
            }
            // Retryable error status code, but ran out of retries:
            Err(Ok(response)) => return Err(Error::from_http_response(response).await),
            // Lower level errors, either unretryable or ran out of retries:
            Err(Err(error)) => return Err(Error::HttpClient(error)),
        };

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .ok_or(Error::BadContentType(None))?;
        if content_type != CollectionMessage::<FixedSize>::MEDIA_TYPE {
            return Err(Error::BadContentType(Some(content_type.clone())));
        }

        let collect_response = CollectionMessage::<Q>::get_decoded(&response.bytes().await?)?;
        if collect_response.encrypted_aggregate_shares().len() != 2 {
            return Err(Error::AggregateShareCount(
                collect_response.encrypted_aggregate_shares().len(),
            ));
        }

        let aggregate_shares_bytes = collect_response
            .encrypted_aggregate_shares()
            .iter()
            .zip(&[Role::Leader, Role::Helper])
            .map(|(encrypted_aggregate_share, role)| {
                hpke::open(
                    &self.parameters.hpke_config,
                    &self.parameters.hpke_private_key,
                    &HpkeApplicationInfo::new(&hpke::Label::AggregateShare, role, &Role::Collector),
                    encrypted_aggregate_share,
                    &AggregateShareAad::new(
                        self.parameters.task_id,
                        BatchSelector::<Q>::new(Q::batch_identifier_for_collection(
                            &job.query,
                            &collect_response,
                        )),
                    )
                    .get_encoded(),
                )
            });
        let aggregate_shares = aggregate_shares_bytes
            .map(|bytes| {
                V::AggregateShare::get_decoded_with_param(
                    &(&self.vdaf_collector, &job.aggregation_parameter),
                    &bytes?,
                )
                .map_err(|_err| Error::AggregateShareDecode)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let report_count = collect_response
            .report_count()
            .try_into()
            .map_err(|_| Error::ReportCountOverflow)?;
        let aggregate_result = self.vdaf_collector.unshard(
            &job.aggregation_parameter,
            aggregate_shares,
            report_count,
        )?;

        Ok(PollResult::CollectionResult(Collection {
            partial_batch_selector: collect_response.partial_batch_selector().clone(),
            report_count: collect_response.report_count(),
            interval: (
                DateTime::<Utc>::from_utc(
                    collect_response.interval().start().as_naive_date_time()?,
                    Utc,
                ),
                collect_response
                    .interval()
                    .duration()
                    .as_chrono_duration()?,
            ),
            aggregate_result,
        }))
    }

    /// A convenience method to repeatedly request the result of an in-progress collection until it
    /// completes.
    async fn poll_until_complete<Q: QueryType>(
        &self,
        job: &CollectionJob<V::AggregationParam, Q>,
    ) -> Result<Collection<V::AggregateResult, Q>, Error> {
        let mut backoff = self.parameters.collect_poll_wait_parameters.clone();
        backoff.reset();
        let deadline = backoff
            .max_elapsed_time
            .map(|duration| Instant::now() + duration);
        loop {
            // poll_once() already retries upon server and connection errors, so propagate any error
            // received from it and return immediately.
            let retry_after = match self.poll_once(job).await? {
                PollResult::CollectionResult(aggregate_result) => return Ok(aggregate_result),
                PollResult::NextAttempt(retry_after) => retry_after,
            };

            // Compute a sleep duration based on the Retry-After header, if available.
            let retry_after_duration = match retry_after {
                Some(RetryAfter::DateTime(system_time)) => {
                    system_time.duration_since(SystemTime::now()).ok()
                }
                Some(RetryAfter::Delay(duration)) => Some(duration),
                None => None,
            };

            let backoff_duration = if let Some(duration) = backoff.next_backoff() {
                duration
            } else {
                // The maximum elapsed time has expired, so return a timeout error.
                return Err(Error::CollectPollTimeout);
            };

            // Sleep for the time indicated in the Retry-After header or the time from our
            // exponential backoff, whichever is longer.
            let sleep_duration = if let Some(retry_after_duration) = retry_after_duration {
                // Check if sleeping for as long as the Retry-After header recommends would result
                // in exceeding the maximum elapsed time, and return a timeout error if so.
                if let Some(deadline) = deadline {
                    let recommendation_is_past_deadline = Instant::now()
                        .checked_add(retry_after_duration)
                        .map_or(true, |recommendation| recommendation > deadline);

                    if recommendation_is_past_deadline {
                        return Err(Error::CollectPollTimeout);
                    }
                }

                std::cmp::max(retry_after_duration, backoff_duration)
            } else {
                backoff_duration
            };
            sleep(sleep_duration).await;
        }
    }

    /// Send a collect request to the leader aggregator, wait for it to complete, and return the
    /// result of the aggregation.
    pub async fn collect<Q: QueryType>(
        &self,
        query: Query<Q>,
        aggregation_parameter: &V::AggregationParam,
    ) -> Result<Collection<V::AggregateResult, Q>, Error> {
        let job = self.start_collection(query, aggregation_parameter).await?;
        self.poll_until_complete(&job).await
    }
}

#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util {
    use crate::{Collection, Collector, Error};
    use janus_messages::{query_type::QueryType, Query};
    use prio::vdaf;

    pub async fn collect_with_rewritten_url<V: vdaf::Collector, Q: QueryType>(
        collector: &Collector<V>,
        query: Query<Q>,
        aggregation_parameter: &V::AggregationParam,
        host: &str,
        port: u16,
    ) -> Result<Collection<V::AggregateResult, Q>, Error> {
        let mut job = collector
            .start_collection(query, aggregation_parameter)
            .await?;
        job.collection_job_url.set_host(Some(host))?;
        job.collection_job_url.set_port(Some(port)).unwrap();
        collector.poll_until_complete(&job).await
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        default_http_client, Collection, CollectionJob, Collector, CollectorParameters, Error,
        PollResult,
    };
    use assert_matches::assert_matches;
    use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
    #[cfg(feature = "fpvec_bounded_l2")]
    use fixed_macro::fixed;
    use janus_core::{
        hpke::{
            self, test_util::generate_test_hpke_config_and_private_key, HpkeApplicationInfo, Label,
        },
        retries::test_http_request_exponential_backoff,
        task::AuthenticationToken,
        test_util::{install_test_trace_subscriber, run_vdaf, VdafTranscript},
    };
    use janus_messages::{
        problem_type::DapProblemType, query_type::FixedSize, AggregateShareAad, BatchId,
        BatchSelector, Collection as CollectionMessage, CollectionJobId, CollectionReq, Duration,
        FixedSizeQuery, HpkeCiphertext, Interval, PartialBatchSelector, Query, Role, TaskId, Time,
    };
    use mockito::Matcher;
    use prio::{
        codec::Encode,
        field::Field64,
        vdaf::{self, prio3::Prio3, AggregateShare, OutputShare},
    };
    use rand::random;
    use reqwest::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        StatusCode, Url,
    };
    use retry_after::RetryAfter;

    fn setup_collector<V: vdaf::Collector>(
        server: &mut mockito::Server,
        vdaf_collector: V,
    ) -> Collector<V> {
        let server_url = Url::parse(&server.url()).unwrap();
        let hpke_keypair = generate_test_hpke_config_and_private_key();
        let parameters = CollectorParameters::new(
            random(),
            server_url,
            AuthenticationToken::Bearer(b"token".to_vec()),
            hpke_keypair.config().clone(),
            hpke_keypair.private_key().clone(),
        )
        .with_http_request_backoff(test_http_request_exponential_backoff())
        .with_collect_poll_backoff(test_http_request_exponential_backoff());
        Collector::new(parameters, vdaf_collector, default_http_client().unwrap())
    }

    fn collection_uri_regex_matcher(task_id: &TaskId) -> Matcher {
        // Matches on the relative path for a collection job resource. The Base64 URL-safe encoding
        // of a collection ID is always 22 characters.
        Matcher::Regex(format!(
            "^/tasks/{task_id}/collection_jobs/[A-Za-z0-9-_]{{22}}$"
        ))
    }

    fn build_collect_response_fixed<const SEED_SIZE: usize, V: vdaf::Aggregator<SEED_SIZE, 16>>(
        transcript: &VdafTranscript<SEED_SIZE, V>,
        parameters: &CollectorParameters,
        batch_id: BatchId,
    ) -> CollectionMessage<FixedSize> {
        let associated_data =
            AggregateShareAad::new(parameters.task_id, BatchSelector::new_fixed_size(batch_id));
        CollectionMessage::new(
            PartialBatchSelector::new_fixed_size(batch_id),
            1,
            Interval::new(
                Time::from_seconds_since_epoch(0),
                Duration::from_seconds(3600),
            )
            .unwrap(),
            Vec::<HpkeCiphertext>::from([
                hpke::seal(
                    &parameters.hpke_config,
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Leader,
                        &Role::Collector,
                    ),
                    &transcript.aggregate_shares[0].get_encoded(),
                    &associated_data.get_encoded(),
                )
                .unwrap(),
                hpke::seal(
                    &parameters.hpke_config,
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Helper,
                        &Role::Collector,
                    ),
                    &transcript.aggregate_shares[1].get_encoded(),
                    &associated_data.get_encoded(),
                )
                .unwrap(),
            ]),
        )
    }

    #[test]
    fn leader_endpoint_end_in_slash() {
        let hpke_keypair = generate_test_hpke_config_and_private_key();
        let collector_parameters = CollectorParameters::new(
            random(),
            "http://example.com/dap".parse().unwrap(),
            AuthenticationToken::Bearer(b"token".to_vec()),
            hpke_keypair.config().clone(),
            hpke_keypair.private_key().clone(),
        );

        assert_eq!(
            collector_parameters.leader_endpoint.as_str(),
            "http://example.com/dap/",
        );

        let collector_parameters = CollectorParameters::new(
            random(),
            "http://example.com".parse().unwrap(),
            AuthenticationToken::Bearer(b"token".to_vec()),
            hpke_keypair.config().clone(),
            hpke_keypair.private_key().clone(),
        );

        assert_eq!(
            collector_parameters.leader_endpoint.as_str(),
            "http://example.com/",
        );
    }

    #[tokio::test]
    async fn successful_collect_prio3_count() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_count(2).unwrap();
        let transcript = run_vdaf(&vdaf, &random(), &(), &random(), &1);
        let collector = setup_collector(&mut server, vdaf);
        let (auth_header, auth_value) =
            collector.parameters.authentication.request_authentication();
        let auth_value = String::from_utf8(auth_value).unwrap();

        let batch_id = random();
        let collect_resp =
            build_collect_response_fixed(&transcript, &collector.parameters, batch_id);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mocked_collect_start_error = server
            .mock("PUT", matcher.clone())
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(500)
            .expect(1)
            .create_async()
            .await;
        let mocked_collect_start_success = server
            .mock("PUT", matcher)
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .match_header(auth_header, auth_value.as_str())
            .with_status(201)
            .expect(1)
            .create_async()
            .await;

        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await;

        mocked_collect_start_error.assert_async().await;
        mocked_collect_start_success.assert_async().await;

        let job = job.unwrap();
        assert_eq!(
            job.query.query_body(),
            &FixedSizeQuery::ByBatchId { batch_id }
        );

        let mocked_collect_error = server
            .mock("POST", job.collection_job_url.path())
            .with_status(500)
            .expect(1)
            .create_async()
            .await;
        let mocked_collect_accepted = server
            .mock("POST", job.collection_job_url.path())
            .with_status(202)
            .expect(2)
            .create_async()
            .await;
        let mocked_collect_complete = server
            .mock("POST", job.collection_job_url.path())
            .match_header(auth_header, auth_value.as_str())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect(1)
            .create_async()
            .await;

        let poll_result = collector.poll_once(&job).await.unwrap();
        assert_matches!(poll_result, PollResult::NextAttempt(None));

        let collection = collector.poll_until_complete(&job).await.unwrap();
        assert_eq!(
            collection,
            Collection::new(
                PartialBatchSelector::new_fixed_size(batch_id),
                1,
                (
                    DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_opt(0, 0).unwrap(),
                        Utc
                    ),
                    chrono::Duration::seconds(3600),
                ),
                1,
            ),
        );

        mocked_collect_error.assert_async().await;
        mocked_collect_accepted.assert_async().await;
        mocked_collect_complete.assert_async().await;
    }

    #[tokio::test]
    async fn successful_collect_prio3_sum() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_sum(2, 8).unwrap();
        let transcript = run_vdaf(&vdaf, &random(), &(), &random(), &144);
        let collector = setup_collector(&mut server, vdaf);

        let batch_id = random();
        let collect_resp =
            build_collect_response_fixed(&transcript, &collector.parameters, batch_id);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mocked_collect_start_success = server
            .mock("PUT", matcher)
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(201)
            .expect(1)
            .create_async()
            .await;

        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap();
        assert_eq!(
            job.query.query_body(),
            &FixedSizeQuery::ByBatchId { batch_id }
        );
        mocked_collect_start_success.assert_async().await;

        let mocked_collect_complete = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect(1)
            .create_async()
            .await;

        let collection = collector.poll_until_complete(&job).await.unwrap();
        assert_eq!(
            collection,
            Collection::new(
                PartialBatchSelector::new_fixed_size(batch_id),
                1,
                (
                    DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_opt(0, 0).unwrap(),
                        Utc
                    ),
                    chrono::Duration::seconds(3600),
                ),
                144
            )
        );

        mocked_collect_complete.assert_async().await;
    }

    #[tokio::test]
    async fn successful_collect_prio3_histogram() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_histogram(2, &[25, 50, 75, 100]).unwrap();
        let transcript = run_vdaf(&vdaf, &random(), &(), &random(), &80);
        let collector = setup_collector(&mut server, vdaf);

        let batch_id = random();
        let collect_resp =
            build_collect_response_fixed(&transcript, &collector.parameters, batch_id);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mocked_collect_start_success = server
            .mock("PUT", matcher)
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(201)
            .expect(1)
            .create_async()
            .await;

        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap();
        assert_eq!(
            job.query.query_body(),
            &FixedSizeQuery::ByBatchId { batch_id }
        );

        mocked_collect_start_success.assert_async().await;

        let mocked_collect_complete = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect(1)
            .create_async()
            .await;

        let collection = collector.poll_until_complete(&job).await.unwrap();
        assert_eq!(
            collection,
            Collection::new(
                PartialBatchSelector::new_fixed_size(batch_id),
                1,
                (
                    DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_opt(0, 0).unwrap(),
                        Utc
                    ),
                    chrono::Duration::seconds(3600),
                ),
                Vec::from([0, 0, 0, 1, 0])
            )
        );

        mocked_collect_complete.assert_async().await;
    }

    #[tokio::test]
    async fn successful_collect_prio3_fixedpoint_boundedl2_vec_sum() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, 3).unwrap();
        let fp32_4_inv = fixed!(0.25: I1F31);
        let fp32_8_inv = fixed!(0.125: I1F31);
        let fp32_16_inv = fixed!(0.0625: I1F31);
        let transcript = run_vdaf(
            &vdaf,
            &random(),
            &(),
            &random(),
            &vec![fp32_16_inv, fp32_8_inv, fp32_4_inv],
        );
        let collector = setup_collector(&mut server, vdaf);

        let batch_id = random();
        let collect_resp =
            build_collect_response_fixed(&transcript, &collector.parameters, batch_id);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mocked_collect_start_success = server
            .mock("PUT", matcher)
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(201)
            .expect(1)
            .create_async()
            .await;

        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap();
        assert_eq!(
            job.query.query_body(),
            &FixedSizeQuery::ByBatchId { batch_id }
        );

        mocked_collect_start_success.assert_async().await;

        let mocked_collect_complete = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect(1)
            .create_async()
            .await;

        let agg_result = collector.poll_until_complete(&job).await.unwrap();
        assert_eq!(
            agg_result,
            Collection::new(
                PartialBatchSelector::new_fixed_size(batch_id),
                1,
                (
                    DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_opt(0, 0).unwrap(),
                        Utc
                    ),
                    chrono::Duration::seconds(3600),
                ),
                Vec::from([0.0625, 0.125, 0.25])
            )
        );

        mocked_collect_complete.assert_async().await;
    }

    #[tokio::test]
    async fn successful_collect_authentication_bearer() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_count(2).unwrap();
        let transcript = run_vdaf(&vdaf, &random(), &(), &random(), &1);
        let server_url = Url::parse(&server.url()).unwrap();
        let hpke_keypair = generate_test_hpke_config_and_private_key();
        let parameters = CollectorParameters::new(
            random(),
            server_url,
            AuthenticationToken::Bearer(Vec::from([0x41u8; 16])),
            hpke_keypair.config().clone(),
            hpke_keypair.private_key().clone(),
        )
        .with_http_request_backoff(test_http_request_exponential_backoff())
        .with_collect_poll_backoff(test_http_request_exponential_backoff());
        let collector = Collector::new(parameters, vdaf, default_http_client().unwrap());

        let batch_id = random();
        let collect_resp =
            build_collect_response_fixed(&transcript, &collector.parameters, batch_id);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mocked_collect_start_success = server
            .mock("PUT", matcher)
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .match_header(AUTHORIZATION.as_str(), "Bearer QUFBQUFBQUFBQUFBQUFBQQ==")
            .with_status(201)
            .expect(1)
            .create_async()
            .await;

        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap();
        assert_eq!(
            job.query.query_body(),
            &FixedSizeQuery::ByBatchId { batch_id }
        );

        mocked_collect_start_success.assert_async().await;

        let mocked_collect_complete = server
            .mock("POST", job.collection_job_url.path())
            .match_header(AUTHORIZATION.as_str(), "Bearer QUFBQUFBQUFBQUFBQUFBQQ==")
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect(1)
            .create_async()
            .await;

        let agg_result = collector.poll_until_complete(&job).await.unwrap();
        assert_eq!(
            agg_result,
            Collection::new(
                PartialBatchSelector::new_fixed_size(batch_id),
                1,
                (
                    DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_opt(0, 0).unwrap(),
                        Utc
                    ),
                    chrono::Duration::seconds(3600),
                ),
                1,
            )
        );

        mocked_collect_complete.assert_async().await;
    }

    #[tokio::test]
    async fn failed_collect_start() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_count(2).unwrap();
        let collector = setup_collector(&mut server, vdaf);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mock_server_error = server
            .mock("PUT", matcher.clone())
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(500)
            .expect_at_least(1)
            .create_async()
            .await;

        let batch_id = random();
        let error = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(dap_problem_type, None);
        });

        mock_server_error.assert_async().await;

        let mock_server_error_details = server
            .mock("PUT", matcher.clone())
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"http://example.com/test_server_error\"}")
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(problem_details.type_url.unwrap(), "http://example.com/test_server_error");
            assert_eq!(dap_problem_type, None);
        });

        mock_server_error_details.assert_async().await;

        let mock_bad_request = server
            .mock("PUT", matcher)
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(400)
            .with_header("Content-Type", "application/problem+json")
            .with_body(concat!(
                "{\"type\": \"urn:ietf:params:ppm:dap:error:unrecognizedMessage\", ",
                "\"detail\": \"The message type for a response was incorrect or the payload was \
                 malformed.\"}"
            ))
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::BAD_REQUEST);
            assert_eq!(problem_details.type_url.unwrap(), "urn:ietf:params:ppm:dap:error:unrecognizedMessage");
            assert_eq!(problem_details.detail.unwrap(), "The message type for a response was incorrect or the payload was malformed.");
            assert_eq!(dap_problem_type, Some(DapProblemType::UnrecognizedMessage));
        });

        mock_bad_request.assert_async().await;
    }

    #[tokio::test]
    async fn failed_collect_poll() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_count(2).unwrap();
        let collector = setup_collector(&mut server, vdaf);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mock_collect_start = server
            .mock("PUT", matcher.clone())
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(201)
            .expect(1)
            .create_async()
            .await;
        let mock_collection_job_server_error = server
            .mock("POST", matcher)
            .with_status(500)
            .expect_at_least(1)
            .create_async()
            .await;

        let batch_id = random();
        let batch_interval = Interval::new(
            Time::from_seconds_since_epoch(1_000_000),
            Duration::from_seconds(3600),
        )
        .unwrap();
        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap();
        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(dap_problem_type, None);
        });

        mock_collect_start.assert_async().await;
        mock_collection_job_server_error.assert_async().await;

        let mock_collection_job_server_error_details = server
            .mock("POST", job.collection_job_url.path())
            .with_status(500)
            .with_header("Content-Type", "application/problem+json")
            .with_body("{\"type\": \"http://example.com/test_server_error\"}")
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(problem_details.type_url.unwrap(), "http://example.com/test_server_error");
            assert_eq!(dap_problem_type, None);
        });

        mock_collection_job_server_error_details
            .assert_async()
            .await;

        let mock_collection_job_bad_request = server
            .mock("POST", job.collection_job_url.path())
            .with_status(400)
            .with_header("Content-Type", "application/problem+json")
            .with_body(concat!(
                "{\"type\": \"urn:ietf:params:ppm:dap:error:unrecognizedMessage\", ",
                "\"detail\": \"The message type for a response was incorrect or the payload was \
                 malformed.\"}"
            ))
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::BAD_REQUEST);
            assert_eq!(problem_details.type_url.unwrap(), "urn:ietf:params:ppm:dap:error:unrecognizedMessage");
            assert_eq!(problem_details.detail.unwrap(), "The message type for a response was incorrect or the payload was malformed.");
            assert_eq!(dap_problem_type, Some(DapProblemType::UnrecognizedMessage));
        });

        mock_collection_job_bad_request.assert_async().await;

        let mock_collection_job_bad_message_bytes = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(b"")
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::Codec(_));

        mock_collection_job_bad_message_bytes.assert_async().await;
        let mock_collection_job_bad_share_count = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(
                CollectionMessage::new(
                    PartialBatchSelector::new_fixed_size(batch_id),
                    0,
                    batch_interval,
                    Vec::new(),
                )
                .get_encoded(),
            )
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::AggregateShareCount(0));

        mock_collection_job_bad_share_count.assert_async().await;

        let mock_collection_job_bad_ciphertext = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(
                CollectionMessage::new(
                    PartialBatchSelector::new_fixed_size(batch_id),
                    1,
                    batch_interval,
                    Vec::from([
                        HpkeCiphertext::new(
                            *collector.parameters.hpke_config.id(),
                            Vec::new(),
                            Vec::new(),
                        ),
                        HpkeCiphertext::new(
                            *collector.parameters.hpke_config.id(),
                            Vec::new(),
                            Vec::new(),
                        ),
                    ]),
                )
                .get_encoded(),
            )
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::Hpke(_));

        mock_collection_job_bad_ciphertext.assert_async().await;

        let associated_data = AggregateShareAad::new(
            collector.parameters.task_id,
            BatchSelector::new_fixed_size(batch_id),
        );
        let collect_resp = CollectionMessage::new(
            PartialBatchSelector::new_fixed_size(batch_id),
            1,
            batch_interval,
            Vec::from([
                hpke::seal(
                    &collector.parameters.hpke_config,
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Leader,
                        &Role::Collector,
                    ),
                    b"bad",
                    &associated_data.get_encoded(),
                )
                .unwrap(),
                hpke::seal(
                    &collector.parameters.hpke_config,
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Helper,
                        &Role::Collector,
                    ),
                    b"bad",
                    &associated_data.get_encoded(),
                )
                .unwrap(),
            ]),
        );
        let mock_collection_job_bad_shares = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::AggregateShareDecode);

        mock_collection_job_bad_shares.assert_async().await;

        let collect_resp = CollectionMessage::new(
            PartialBatchSelector::new_fixed_size(batch_id),
            1,
            batch_interval,
            Vec::from([
                hpke::seal(
                    &collector.parameters.hpke_config,
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Leader,
                        &Role::Collector,
                    ),
                    &AggregateShare::from(OutputShare::from(Vec::from([Field64::from(0)])))
                        .get_encoded(),
                    &associated_data.get_encoded(),
                )
                .unwrap(),
                hpke::seal(
                    &collector.parameters.hpke_config,
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Helper,
                        &Role::Collector,
                    ),
                    &AggregateShare::from(OutputShare::from(Vec::from([
                        Field64::from(0),
                        Field64::from(0),
                    ])))
                    .get_encoded(),
                    &associated_data.get_encoded(),
                )
                .unwrap(),
            ]),
        );
        let mock_collection_job_wrong_length = server
            .mock("POST", job.collection_job_url.path())
            .with_status(200)
            .with_header(
                CONTENT_TYPE.as_str(),
                CollectionMessage::<FixedSize>::MEDIA_TYPE,
            )
            .with_body(collect_resp.get_encoded())
            .expect_at_least(1)
            .create_async()
            .await;

        let error = collector.poll_once(&job).await.unwrap_err();
        assert_matches!(error, Error::AggregateShareDecode);

        mock_collection_job_wrong_length.assert_async().await;

        let mock_collection_job_always_fail = server
            .mock("POST", job.collection_job_url.path())
            .with_status(500)
            .expect_at_least(3)
            .create_async()
            .await;
        let error = collector.poll_until_complete(&job).await.unwrap_err();
        assert_matches!(error, Error::Http { problem_details, dap_problem_type } => {
            assert_eq!(problem_details.status.unwrap(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(dap_problem_type, None);
        });
        mock_collection_job_always_fail.assert_async().await;
    }

    #[tokio::test]
    async fn collect_poll_retry_after() {
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_count(2).unwrap();
        let collector = setup_collector(&mut server, vdaf);
        let matcher = collection_uri_regex_matcher(&collector.parameters.task_id);

        let mock_collect_start = server
            .mock("PUT", matcher.clone())
            .match_header(
                CONTENT_TYPE.as_str(),
                CollectionReq::<FixedSize>::MEDIA_TYPE,
            )
            .with_status(201)
            .expect(1)
            .create_async()
            .await;
        let batch_id = random();
        let job = collector
            .start_collection(
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
                &(),
            )
            .await
            .unwrap();
        mock_collect_start.assert_async().await;

        let mock_collect_poll_no_retry_after = server
            .mock("POST", job.collection_job_url.path())
            .with_status(202)
            .expect(1)
            .create_async()
            .await;
        assert_matches!(
            collector.poll_once(&job).await.unwrap(),
            PollResult::NextAttempt(None)
        );
        mock_collect_poll_no_retry_after.assert_async().await;

        let mock_collect_poll_retry_after_60s = server
            .mock("POST", job.collection_job_url.path())
            .with_status(202)
            .with_header("Retry-After", "60")
            .expect(1)
            .create_async()
            .await;
        assert_matches!(
            collector.poll_once(&job).await.unwrap(),
            PollResult::NextAttempt(Some(RetryAfter::Delay(duration))) => assert_eq!(duration, std::time::Duration::from_secs(60))
        );
        mock_collect_poll_retry_after_60s.assert_async().await;

        let mock_collect_poll_retry_after_date_time = server
            .mock("POST", job.collection_job_url.path())
            .with_status(202)
            .with_header("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT")
            .expect(1)
            .create_async()
            .await;
        let ref_date_time = Utc.with_ymd_and_hms(2015, 10, 21, 7, 28, 0).unwrap();
        assert_matches!(
            collector.poll_once(&job).await.unwrap(),
            PollResult::NextAttempt(Some(RetryAfter::DateTime(system_time))) => assert_eq!(system_time, ref_date_time.into())
        );
        mock_collect_poll_retry_after_date_time.assert_async().await;
    }

    #[tokio::test]
    async fn poll_timing() {
        // This test exercises handling of the different Retry-After header forms. It does not test
        // the amount of time that poll_until_complete() sleeps. `tokio::time::pause()` cannot be
        // used for this because hyper uses `tokio::time::Interval` internally, see issue #234.
        install_test_trace_subscriber();
        let mut server = mockito::Server::new_async().await;
        let vdaf = Prio3::new_count(2).unwrap();
        let mut collector = setup_collector(&mut server, vdaf);
        collector
            .parameters
            .collect_poll_wait_parameters
            .max_elapsed_time = Some(std::time::Duration::from_secs(3));

        let collection_job_id: CollectionJobId = random();
        let collection_job_path = format!(
            "/tasks/{}/collection_jobs/{collection_job_id}",
            collector.parameters.task_id
        );

        let collection_job_url = format!("{}{collection_job_path}", server.url());
        let batch_id = random();
        let job = CollectionJob::new(
            collection_job_url.parse().unwrap(),
            Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
            (),
        );

        let mock_collect_poll_retry_after_1s = server
            .mock("POST", collection_job_path.as_str())
            .with_status(202)
            .with_header("Retry-After", "1")
            .expect(1)
            .create_async()
            .await;
        let mock_collect_poll_retry_after_10s = server
            .mock("POST", collection_job_path.as_str())
            .with_status(202)
            .with_header("Retry-After", "10")
            .expect(1)
            .create_async()
            .await;
        assert_matches!(
            collector.poll_until_complete(&job).await.unwrap_err(),
            Error::CollectPollTimeout
        );
        mock_collect_poll_retry_after_1s.assert_async().await;
        mock_collect_poll_retry_after_10s.assert_async().await;

        let near_future =
            Utc::now() + chrono::Duration::from_std(std::time::Duration::from_secs(1)).unwrap();
        let near_future_formatted = near_future.format("%a, %d %b %Y %H:%M:%S GMT").to_string();
        let mock_collect_poll_retry_after_near_future = server
            .mock("POST", collection_job_path.as_str())
            .with_status(202)
            .with_header("Retry-After", &near_future_formatted)
            .expect(1)
            .create_async()
            .await;
        let mock_collect_poll_retry_after_past = server
            .mock("POST", collection_job_path.as_str())
            .with_status(202)
            .with_header("Retry-After", "Mon, 01 Jan 1900 00:00:00 GMT")
            .expect(1)
            .create_async()
            .await;
        let mock_collect_poll_retry_after_far_future = server
            .mock("POST", collection_job_path.as_str())
            .with_status(202)
            .with_header("Retry-After", "Wed, 01 Jan 3000 00:00:00 GMT")
            .expect(1)
            .create_async()
            .await;
        assert_matches!(
            collector.poll_until_complete(&job).await.unwrap_err(),
            Error::CollectPollTimeout
        );
        mock_collect_poll_retry_after_near_future
            .assert_async()
            .await;
        mock_collect_poll_retry_after_past.assert_async().await;
        mock_collect_poll_retry_after_far_future
            .assert_async()
            .await;

        // Manipulate backoff settings so that we make one or two requests and time out.
        collector
            .parameters
            .collect_poll_wait_parameters
            .max_elapsed_time = Some(std::time::Duration::from_millis(15));
        collector
            .parameters
            .collect_poll_wait_parameters
            .initial_interval = std::time::Duration::from_millis(10);
        let mock_collect_poll_no_retry_after = server
            .mock("POST", collection_job_path.as_str())
            .with_status(202)
            .expect_at_least(1)
            .create_async()
            .await;
        assert_matches!(
            collector.poll_until_complete(&job).await.unwrap_err(),
            Error::CollectPollTimeout
        );
        mock_collect_poll_no_retry_after.assert_async().await;
    }
}
