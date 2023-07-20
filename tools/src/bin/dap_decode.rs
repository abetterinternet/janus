use anyhow::Result;
use clap::{Parser, ValueEnum};
use janus_messages::{
    query_type::FixedSize, AggregateShare, AggregateShareReq, AggregationJobContinueReq,
    AggregationJobInitializeReq, AggregationJobResp, Collection, CollectionReq, HpkeConfig,
    HpkeConfigList, Report,
};
use prio::codec::Decode;
use std::{
    fmt::Debug,
    fs::File,
    io::{stdin, Read},
};

fn main() -> Result<()> {
    let options = Options::parse();

    let decoded = decode_dap_message(&options.message_file, &options.media_type)?;
    println!("{decoded:#?}");

    Ok(())
}

/// Decode the contents of `message_file` as a DAP message with `media_type`, returning the decoded
/// object.
fn decode_dap_message(message_file: &str, media_type: &MediaType) -> Result<Box<dyn Debug>> {
    let mut reader = if message_file.eq("-") {
        Box::new(stdin()) as Box<dyn Read>
    } else {
        Box::new(File::open(message_file)?) as Box<dyn Read>
    };

    let mut message_buf = Vec::new();
    reader.read_to_end(&mut message_buf)?;

    let decoded: Box<dyn Debug> = match media_type {
        MediaType::HpkeConfig => {
            let message: HpkeConfig = HpkeConfig::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::HpkeConfigList => {
            let message: HpkeConfigList = HpkeConfigList::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::Report => {
            let message: Report = Report::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::AggregationJobInitializeReq => {
            let message: AggregationJobInitializeReq<FixedSize> =
                AggregationJobInitializeReq::<FixedSize>::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::AggregationJobContinueReq => {
            let message: AggregationJobContinueReq =
                AggregationJobContinueReq::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::AggregationJobResp => {
            let message: AggregationJobResp = AggregationJobResp::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::AggregateShareReq => {
            let message: AggregateShareReq<FixedSize> =
                AggregateShareReq::<FixedSize>::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::AggregateShare => {
            let message: AggregateShare = AggregateShare::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::CollectionReq => {
            let message: CollectionReq<FixedSize> =
                CollectionReq::<FixedSize>::get_decoded(&message_buf)?;
            Box::new(message)
        }
        MediaType::Collection => {
            let message: Collection<FixedSize> =
                Collection::<FixedSize>::get_decoded(&message_buf)?;
            Box::new(message)
        }
    };

    Ok(decoded)
}

#[derive(Debug, Clone, ValueEnum)]
#[value()]
enum MediaType {
    #[value(name = "hpke-config")]
    HpkeConfig,
    #[value(name = "hpke-config-list")]
    HpkeConfigList,
    #[value(name = "report")]
    Report,
    #[value(name = "aggregation-job-init-req")]
    AggregationJobInitializeReq,
    #[value(name = "aggregation-job-resp")]
    AggregationJobResp,
    #[value(name = "aggregation-job-continue-req")]
    AggregationJobContinueReq,
    #[value(name = "aggregate-share-req")]
    AggregateShareReq,
    #[value(name = "aggregate-share")]
    AggregateShare,
    #[value(name = "collect-req")]
    CollectionReq,
    #[value(name = "collection")]
    Collection,
}

#[derive(Debug, Parser)]
#[command(
    name = "dap_decode",
    about = "Distributed Aggregation Protocol message decoder",
    version,
    rename_all = "kebab-case"
)]
struct Options {
    /// Path to file containing message to decode. Pass "-" to read from stdin.
    message_file: String,

    /// Media type of the message to decode.
    #[arg(long, short = 't', required = true)]
    media_type: MediaType,
}

#[cfg(test)]
mod tests {
    use crate::Options;
    use clap::CommandFactory;

    #[test]
    fn verify_clap_app() {
        Options::command().debug_assert();
    }
}
