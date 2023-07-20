use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine,
};
use clap::{
    builder::{NonEmptyStringValueParser, StringValueParser, TypedValueParser},
    error::ErrorKind,
    ArgAction, Args, CommandFactory, FromArgMatches, Parser, ValueEnum,
};
use derivative::Derivative;
#[cfg(feature = "fpvec_bounded_l2")]
use fixed::types::extra::{U15, U31, U63};
#[cfg(feature = "fpvec_bounded_l2")]
use fixed::{FixedI16, FixedI32, FixedI64};
use janus_collector::{default_http_client, AuthenticationToken, Collector, CollectorParameters};
use janus_core::{hpke::HpkePrivateKey, task::DapAuthToken};
use janus_messages::{
    query_type::{FixedSize, QueryType},
    BatchId, FixedSizeQuery, HpkeConfig, PartialBatchSelector, Query, TaskId,
};
#[cfg(feature = "fpvec_bounded_l2")]
use prio::vdaf::prio3::Prio3FixedPointBoundedL2VecSumMultithreaded;
use prio::{
    codec::Decode,
    vdaf::{self, prio3::Prio3},
};
use std::fmt::Debug;
use tracing_log::LogTracer;
use tracing_subscriber::{prelude::*, EnvFilter, Registry};
use url::Url;

/// Enum to propagate errors through this program. Clap errors are handled separately from all
/// others because [`clap::Error::exit`] takes care of its own error formatting, command-line help,
/// and exit code.
#[derive(Debug)]
enum Error {
    Anyhow(anyhow::Error),
    Clap(clap::Error),
}

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Error::Anyhow(error)
    }
}

impl From<clap::Error> for Error {
    fn from(error: clap::Error) -> Self {
        Error::Clap(error)
    }
}

// Parsers for command-line arguments:

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
#[clap(rename_all = "lower")]
enum VdafType {
    /// Prio3Count
    Count,
    /// Prio3CountVec
    CountVec,
    /// Prio3Sum
    Sum,
    /// Prio3SumVec
    SumVec,
    /// Prio3Histogram
    Histogram,
    #[cfg(feature = "fpvec_bounded_l2")]
    /// Prio3FixedPoint16BitBoundedL2VecSum
    FixedPoint16BitBoundedL2VecSum,
    #[cfg(feature = "fpvec_bounded_l2")]
    /// Prio3FixedPoint32BitBoundedL2VecSum
    FixedPoint32BitBoundedL2VecSum,
    #[cfg(feature = "fpvec_bounded_l2")]
    /// Prio3FixedPoint64BitBoundedL2VecSum
    FixedPoint64BitBoundedL2VecSum,
}

#[derive(Clone)]
struct TaskIdValueParser {
    inner: NonEmptyStringValueParser,
}

impl TaskIdValueParser {
    fn new() -> TaskIdValueParser {
        TaskIdValueParser {
            inner: NonEmptyStringValueParser::new(),
        }
    }
}

impl TypedValueParser for TaskIdValueParser {
    type Value = TaskId;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let input = self.inner.parse_ref(cmd, arg, value)?;
        let task_id_bytes: [u8; TaskId::LEN] = URL_SAFE_NO_PAD
            .decode(input)
            .map_err(|err| clap::Error::raw(ErrorKind::ValueValidation, err))?
            .try_into()
            .map_err(|_| {
                clap::Error::raw(ErrorKind::ValueValidation, "task ID length incorrect")
            })?;
        Ok(TaskId::from(task_id_bytes))
    }
}

#[derive(Clone)]
struct BatchIdValueParser {
    inner: NonEmptyStringValueParser,
}

impl BatchIdValueParser {
    fn new() -> BatchIdValueParser {
        BatchIdValueParser {
            inner: NonEmptyStringValueParser::new(),
        }
    }
}

impl TypedValueParser for BatchIdValueParser {
    type Value = BatchId;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let input = self.inner.parse_ref(cmd, arg, value)?;
        let batch_id_bytes: [u8; BatchId::LEN] = URL_SAFE_NO_PAD
            .decode(input)
            .map_err(|err| clap::Error::raw(ErrorKind::ValueValidation, err))?
            .try_into()
            .map_err(|_| {
                clap::Error::raw(ErrorKind::ValueValidation, "batch ID length incorrect")
            })?;
        Ok(BatchId::from(batch_id_bytes))
    }
}

fn parse_authentication_token(value: String) -> Result<AuthenticationToken, anyhow::Error> {
    DapAuthToken::try_from(value.into_bytes()).map(AuthenticationToken::DapAuth)
}

fn parse_authentication_token_base64(value: String) -> Result<AuthenticationToken, anyhow::Error> {
    Ok(AuthenticationToken::Bearer(STANDARD.decode(value)?))
}

#[derive(Clone)]
struct HpkeConfigValueParser {
    inner: NonEmptyStringValueParser,
}

impl HpkeConfigValueParser {
    fn new() -> HpkeConfigValueParser {
        HpkeConfigValueParser {
            inner: NonEmptyStringValueParser::new(),
        }
    }
}

impl TypedValueParser for HpkeConfigValueParser {
    type Value = HpkeConfig;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let input = self.inner.parse_ref(cmd, arg, value)?;
        let bytes = URL_SAFE_NO_PAD
            .decode(input)
            .map_err(|err| clap::Error::raw(ErrorKind::ValueValidation, err))?;
        HpkeConfig::get_decoded(&bytes)
            .map_err(|err| clap::Error::raw(ErrorKind::ValueValidation, err))
    }
}

#[derive(Clone)]
struct PrivateKeyValueParser {
    inner: NonEmptyStringValueParser,
}

impl PrivateKeyValueParser {
    fn new() -> PrivateKeyValueParser {
        PrivateKeyValueParser {
            inner: NonEmptyStringValueParser::new(),
        }
    }
}

impl TypedValueParser for PrivateKeyValueParser {
    type Value = HpkePrivateKey;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let input = self.inner.parse_ref(cmd, arg, value)?;
        let bytes = URL_SAFE_NO_PAD
            .decode(input)
            .map_err(|err| clap::Error::raw(ErrorKind::ValueValidation, err))?;
        Ok(HpkePrivateKey::new(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Buckets(Vec<u64>);

#[derive(Clone)]
struct BucketsValueParser {
    inner: NonEmptyStringValueParser,
}

impl BucketsValueParser {
    fn new() -> BucketsValueParser {
        BucketsValueParser {
            inner: NonEmptyStringValueParser::new(),
        }
    }
}

impl TypedValueParser for BucketsValueParser {
    type Value = Buckets;

    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let input = self.inner.parse_ref(cmd, arg, value)?;
        input
            .split(',')
            .map(|chunk| chunk.trim().parse())
            .collect::<Result<Vec<_>, _>>()
            .map(Buckets)
            .map_err(|err| clap::Error::raw(ErrorKind::ValueValidation, err))
    }
}

#[derive(Derivative, Args, PartialEq, Eq)]
#[derivative(Debug)]
#[group(required = true)]
struct AuthenticationOptions {
    /// Authentication token for the DAP-Auth-Token HTTP header
    #[clap(
        long,
        required = false,
        value_parser = StringValueParser::new().try_map(parse_authentication_token),
        env,
        help_heading = "Authorization",
        display_order = 0,
        conflicts_with = "authorization_bearer_token"
    )]
    #[derivative(Debug = "ignore")]
    dap_auth_token: Option<AuthenticationToken>,

    /// Authentication token for the "Authorization: Bearer ..." HTTP header, in base64
    #[clap(
        long,
        required = false,
        value_parser = StringValueParser::new().try_map(parse_authentication_token_base64),
        env,
        help_heading = "Authorization",
        display_order = 1,
        conflicts_with = "dap_auth_token"
    )]
    #[derivative(Debug = "ignore")]
    authorization_bearer_token: Option<AuthenticationToken>,
}

#[derive(Debug, Args, PartialEq, Eq)]
#[group(required = true)]
struct QueryOptions {
    /// Batch identifier, encoded with base64url
    #[clap(
        long,
        value_parser = BatchIdValueParser::new(),
        conflicts_with_all = ["current_batch"],
        help_heading = "Collect Request Parameters (Fixed Size)",
    )]
    batch_id: Option<BatchId>,
    /// Have the aggregator select a batch that has not yet been collected
    #[clap(
        long,
        action = ArgAction::SetTrue,
        conflicts_with_all = ["batch_id"],
        help_heading = "Collect Request Parameters (Fixed Size)",
    )]
    current_batch: bool,
}

#[derive(Derivative, Parser, PartialEq, Eq)]
#[derivative(Debug)]
#[clap(
    name = "collect",
    version,
    about = "Command-line DAP-PPM collector from ISRG's Divvi Up",
    long_about = None,
)]
struct Options {
    /// DAP task identifier, encoded with base64url
    #[clap(
        long,
        value_parser = TaskIdValueParser::new(),
        help_heading = "DAP Task Parameters",
        display_order = 0
    )]
    task_id: TaskId,
    /// The leader aggregator's endpoint URL
    #[clap(long, help_heading = "DAP Task Parameters", display_order = 1)]
    leader: Url,
    /// DAP message for the collector's HPKE configuration, encoded with base64url
    #[clap(
        long,
        value_parser = HpkeConfigValueParser::new(),
        help_heading = "DAP Task Parameters",
        display_order = 2
    )]
    hpke_config: HpkeConfig,
    /// The collector's HPKE private key, encoded with base64url
    #[clap(
        long,
        value_parser = PrivateKeyValueParser::new(),
        env,
        help_heading = "DAP Task Parameters",
        display_order = 3
    )]
    #[derivative(Debug = "ignore")]
    hpke_private_key: HpkePrivateKey,

    #[clap(flatten)]
    authentication: AuthenticationOptions,

    /// VDAF algorithm
    #[clap(
        long,
        value_enum,
        help_heading = "VDAF Algorithm and Parameters",
        display_order = 0
    )]
    vdaf: VdafType,
    /// Number of vector elements, for use with --vdaf=countvec and --vdaf=sumvec
    #[clap(long, help_heading = "VDAF Algorithm and Parameters")]
    length: Option<usize>,
    /// Bit length of measurements, for use with --vdaf=sum and --vdaf=sumvec
    #[clap(long, help_heading = "VDAF Algorithm and Parameters")]
    bits: Option<usize>,
    /// Comma-separated list of bucket boundaries, for use with --vdaf=histogram
    #[clap(
        long,
        required = false,
        num_args = 1,
        action = ArgAction::Set,
        value_parser = BucketsValueParser::new(),
        help_heading = "VDAF Algorithm and Parameters"
    )]
    buckets: Option<Buckets>,

    #[clap(flatten)]
    query: QueryOptions,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    install_tracing_subscriber()?;

    let mut command = Options::command();
    let mut matches = match command.try_get_matches_from_mut(std::env::args_os()) {
        Ok(matches) => matches,
        Err(err) => err.format(&mut command).exit(),
    };
    let options = match Options::from_arg_matches_mut(&mut matches) {
        Ok(options) => options,
        Err(err) => err.format(&mut command).exit(),
    };

    match run(options).await {
        Ok(()) => Ok(()),
        Err(Error::Anyhow(err)) => Err(err),
        Err(Error::Clap(err)) => err.format(&mut command).exit(),
    }
}

// This function is broken out from `main()` for the sake of testing its argument handling.
async fn run(options: Options) -> Result<(), Error> {
    match (&options.query.batch_id, options.query.current_batch) {
        (Some(batch_id), false) => {
            let batch_id = *batch_id;
            run_with_query(
                options,
                Query::new_fixed_size(FixedSizeQuery::ByBatchId { batch_id }),
            )
            .await
        }
        (None, true) => {
            run_with_query(options, Query::new_fixed_size(FixedSizeQuery::CurrentBatch)).await
        }
        _ => unreachable!(),
    }
}

async fn run_with_query<Q: QueryType>(options: Options, query: Query<Q>) -> Result<(), Error>
where
    Q: QueryTypeExt,
{
    let authentication = match (
        options.authentication.dap_auth_token,
        options.authentication.authorization_bearer_token,
    ) {
        (None, Some(token)) => token,
        (Some(token), None) => token,
        (None, None) | (Some(_), Some(_)) => unreachable!(),
    };
    let parameters = CollectorParameters::new(
        options.task_id,
        options.leader,
        authentication,
        options.hpke_config.clone(),
        options.hpke_private_key.clone(),
    );
    let http_client = default_http_client().map_err(|err| Error::Anyhow(err.into()))?;
    match (options.vdaf, options.length, options.bits, options.buckets) {
        (VdafType::Count, None, None, None) => {
            let vdaf = Prio3::new_count(2).map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        (VdafType::CountVec, Some(length), None, None) => {
            let vdaf = Prio3::new_sum_vec(2, 1, length).map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        (VdafType::Sum, None, Some(bits), None) => {
            let vdaf = Prio3::new_sum(2, bits).map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        (VdafType::SumVec, Some(length), Some(bits), None) => {
            let vdaf =
                Prio3::new_sum_vec(2, bits, length).map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        (VdafType::Histogram, None, None, Some(ref buckets)) => {
            let vdaf =
                Prio3::new_histogram(2, &buckets.0).map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        #[cfg(feature = "fpvec_bounded_l2")]
        (VdafType::FixedPoint16BitBoundedL2VecSum, Some(length), None, None) => {
            let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI16<U15>> =
                Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, length)
                    .map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        #[cfg(feature = "fpvec_bounded_l2")]
        (VdafType::FixedPoint32BitBoundedL2VecSum, Some(length), None, None) => {
            let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI32<U31>> =
                Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, length)
                    .map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        #[cfg(feature = "fpvec_bounded_l2")]
        (VdafType::FixedPoint64BitBoundedL2VecSum, Some(length), None, None) => {
            let vdaf: Prio3FixedPointBoundedL2VecSumMultithreaded<FixedI64<U63>> =
                Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(2, length)
                    .map_err(|err| Error::Anyhow(err.into()))?;
            run_collection_generic(parameters, vdaf, http_client, query, &())
                .await
                .map_err(|err| Error::Anyhow(err.into()))
        }
        _ => Err(clap::Error::raw(
            ErrorKind::ArgumentConflict,
            format!(
                "incorrect VDAF parameter arguments were supplied for {}",
                options
                    .vdaf
                    .to_possible_value()
                    .unwrap()
                    .get_help()
                    .unwrap(),
            ),
        )
        .into()),
    }
}

async fn run_collection_generic<V: vdaf::Collector, Q: QueryTypeExt>(
    parameters: CollectorParameters,
    vdaf: V,
    http_client: reqwest::Client,
    query: Query<Q>,
    agg_param: &V::AggregationParam,
) -> Result<(), janus_collector::Error>
where
    V::AggregateResult: Debug,
{
    let collector = Collector::new(parameters, vdaf, http_client);
    let collection = collector.collect(query, agg_param).await?;
    if !Q::IS_PARTIAL_BATCH_SELECTOR_TRIVIAL {
        println!(
            "Batch: {}",
            Q::format_partial_batch_selector(collection.partial_batch_selector())
        );
    }
    println!("Number of reports: {}", collection.report_count());
    println!(
        "Spanned interval: start: {} length: {}",
        collection.interval().0,
        collection.interval().1
    );
    println!("Aggregation result: {:?}", collection.aggregate_result());
    Ok(())
}

fn install_tracing_subscriber() -> anyhow::Result<()> {
    let stdout_filter = EnvFilter::builder().from_env()?;
    let layer = tracing_subscriber::fmt::layer()
        .with_level(true)
        .with_target(true)
        .pretty();
    let subscriber = Registry::default().with(stdout_filter.and_then(layer));
    tracing::subscriber::set_global_default(subscriber)?;

    LogTracer::init()?;

    Ok(())
}

trait QueryTypeExt: QueryType {
    const IS_PARTIAL_BATCH_SELECTOR_TRIVIAL: bool;

    fn format_partial_batch_selector(partial_batch_selector: &PartialBatchSelector<Self>)
        -> String;
}

impl QueryTypeExt for FixedSize {
    const IS_PARTIAL_BATCH_SELECTOR_TRIVIAL: bool = false;

    fn format_partial_batch_selector(
        partial_batch_selector: &PartialBatchSelector<Self>,
    ) -> String {
        URL_SAFE_NO_PAD.encode(partial_batch_selector.batch_id().as_ref())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AuthenticationOptions, AuthenticationToken, DapAuthToken, Options, QueryOptions, VdafType,
    };
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use clap::{error::ErrorKind, CommandFactory, Parser};
    use janus_core::hpke::test_util::generate_test_hpke_config_and_private_key;
    use janus_messages::{BatchId, TaskId};
    use prio::codec::Encode;
    use rand::random;
    use reqwest::Url;

    #[test]
    fn verify_app() {
        Options::command().debug_assert();
    }

    #[test]
    fn batch_arguments() {
        let task_id: TaskId = random();
        let task_id_encoded = URL_SAFE_NO_PAD.encode(task_id.get_encoded());

        let leader = Url::parse("https://example.com/dap/").unwrap();

        let hpke_keypair = generate_test_hpke_config_and_private_key();
        let encoded_hpke_config = URL_SAFE_NO_PAD.encode(hpke_keypair.config().get_encoded());
        let encoded_private_key = URL_SAFE_NO_PAD.encode(hpke_keypair.private_key().as_ref());
        let auth_token = Some(AuthenticationToken::DapAuth(
            DapAuthToken::try_from(b"collector-authentication-token".to_vec()).unwrap(),
        ));

        // Check parsing arguments for a current batch query.
        let expected = Options {
            task_id,
            leader: leader.clone(),
            authentication: AuthenticationOptions {
                dap_auth_token: auth_token.clone(),
                authorization_bearer_token: None,
            },
            hpke_config: hpke_keypair.config().clone(),
            hpke_private_key: hpke_keypair.private_key().clone(),
            vdaf: VdafType::Count,
            length: None,
            bits: None,
            buckets: None,
            query: QueryOptions {
                batch_id: None,
                current_batch: true,
            },
        };
        let correct_arguments = [
            "collect",
            &format!("--task-id={task_id_encoded}"),
            "--leader",
            leader.as_str(),
            "--dap-auth-token",
            "collector-authentication-token",
            &format!("--hpke-config={encoded_hpke_config}"),
            &format!("--hpke-private-key={encoded_private_key}"),
            "--vdaf",
            "count",
            "--current-batch",
        ];
        match Options::try_parse_from(correct_arguments) {
            Ok(got) => assert_eq!(got, expected),
            Err(e) => panic!("{}\narguments were {:?}", e, correct_arguments),
        }

        // Check parsing arguments for a by-batch-id query.
        let batch_id: BatchId = random();
        let batch_id_encoded = URL_SAFE_NO_PAD.encode(batch_id.as_ref());
        let expected = Options {
            task_id,
            leader: leader.clone(),
            authentication: AuthenticationOptions {
                dap_auth_token: auth_token,
                authorization_bearer_token: None,
            },
            hpke_config: hpke_keypair.config().clone(),
            hpke_private_key: hpke_keypair.private_key().clone(),
            vdaf: VdafType::Count,
            length: None,
            bits: None,
            buckets: None,
            query: QueryOptions {
                batch_id: Some(batch_id),
                current_batch: false,
            },
        };
        let correct_arguments = [
            "collect",
            &format!("--task-id={task_id_encoded}"),
            "--leader",
            leader.as_str(),
            "--dap-auth-token",
            "collector-authentication-token",
            &format!("--hpke-config={encoded_hpke_config}"),
            &format!("--hpke-private-key={encoded_private_key}"),
            "--vdaf",
            "count",
            &format!("--batch-id={batch_id_encoded}"),
        ];
        match Options::try_parse_from(correct_arguments) {
            Ok(got) => assert_eq!(got, expected),
            Err(e) => panic!("{}\narguments were {:?}", e, correct_arguments),
        }

        let base_arguments = Vec::from([
            "collect".to_string(),
            format!("--task-id={task_id_encoded}"),
            "--leader".to_string(),
            "https://example.com/dap/".to_string(),
            "--dap-auth-token".to_string(),
            "collector-authentication-token".to_string(),
            format!("--hpke-config={encoded_hpke_config}"),
            format!("--hpke-private-key={encoded_private_key}"),
            "--vdaf=count".to_string(),
        ]);

        let mut good_arguments = base_arguments.clone();
        good_arguments.push("--current-batch".to_string());
        Options::try_parse_from(good_arguments).unwrap();

        // Check that clap enforces all the constraints we need on combinations of query arguments.
        // This allows us to treat a default match branch as `unreachable!()` when unpacking the
        // argument matches.

        assert_eq!(
            Options::try_parse_from(base_arguments.clone())
                .unwrap_err()
                .kind(),
            ErrorKind::MissingRequiredArgument
        );

        let mut bad_arguments = base_arguments;
        bad_arguments.extend([
            "--batch-id=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_string(),
            "--current-batch".to_string(),
        ]);
        assert_eq!(
            Options::try_parse_from(bad_arguments).unwrap_err().kind(),
            ErrorKind::ArgumentConflict
        );
    }

    #[test]
    fn auth_arguments() {
        let task_id: TaskId = random();
        let task_id_encoded = URL_SAFE_NO_PAD.encode(task_id.get_encoded());

        let hpke_keypair = generate_test_hpke_config_and_private_key();
        let encoded_hpke_config = URL_SAFE_NO_PAD.encode(hpke_keypair.config().get_encoded());
        let encoded_private_key = URL_SAFE_NO_PAD.encode(hpke_keypair.private_key().as_ref());

        let base_arguments = Vec::from([
            "collect".to_string(),
            format!("--task-id={task_id_encoded}"),
            "--leader".to_string(),
            "https://example.com/dap/".to_string(),
            format!("--hpke-config={encoded_hpke_config}"),
            format!("--hpke-private-key={encoded_private_key}"),
            "--current-batch".to_string(),
            "--vdaf=count".to_string(),
        ]);
        let dap_auth_token_arguments = Vec::from([
            "--dap-auth-token".to_string(),
            "collector-authentication-token".to_string(),
        ]);
        let authorization_bearer_token_arguments = Vec::from([
            "--authorization-bearer-token".to_string(),
            "/////////////////////w==".to_string(),
        ]);

        let mut case_1_arguments = base_arguments.clone();
        case_1_arguments.extend(dap_auth_token_arguments.iter().cloned());
        assert_eq!(
            Options::try_parse_from(case_1_arguments)
                .unwrap()
                .authentication,
            AuthenticationOptions {
                dap_auth_token: Some(AuthenticationToken::DapAuth(
                    DapAuthToken::try_from(b"collector-authentication-token".to_vec()).unwrap()
                )),
                authorization_bearer_token: None,
            }
        );

        let mut case_2_arguments = base_arguments.clone();
        case_2_arguments.extend(authorization_bearer_token_arguments.iter().cloned());
        assert_eq!(
            Options::try_parse_from(case_2_arguments)
                .unwrap()
                .authentication,
            AuthenticationOptions {
                dap_auth_token: None,
                authorization_bearer_token: Some(AuthenticationToken::Bearer(Vec::from(
                    [0xff; 16]
                )),)
            }
        );

        let case_3_arguments = base_arguments.clone();
        assert_eq!(
            Options::try_parse_from(case_3_arguments)
                .unwrap_err()
                .kind(),
            ErrorKind::MissingRequiredArgument
        );

        let mut case_4_arguments = base_arguments.clone();
        case_4_arguments.extend(dap_auth_token_arguments.iter().cloned());
        case_4_arguments.extend(authorization_bearer_token_arguments.iter().cloned());
        assert_eq!(
            Options::try_parse_from(case_4_arguments)
                .unwrap_err()
                .kind(),
            ErrorKind::ArgumentConflict
        );

        let mut case_5_arguments = base_arguments;
        case_5_arguments.push("--authorization-bearer-token=not-base-64-!@#$%^&*".to_string());
        assert_eq!(
            Options::try_parse_from(case_5_arguments)
                .unwrap_err()
                .kind(),
            ErrorKind::ValueValidation
        );
    }
}
