//! Configuration for various Janus actors.

use crate::trace::TraceConfiguration;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use url::Url;

/// Configuration for a Janus server using a database.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DbConfig {
    /// URL at which to connect to the database.
    pub url: Url,
    // TODO: add option for connecting to database over TLS, if necessary
}

/// Non-secret configuration options for a Janus aggregator, deserialized from
/// YAML.
///
/// # Examples
///
/// ```
/// use janus_server::config::AggregatorConfig;
///
/// let yaml_config = r#"
/// ---
/// listen_address: "0.0.0.0:8080"
/// database:
///   url: "postgres://postgres:postgres@localhost:5432/postgres"
/// logging_config: # logging_config is optional
///   force_json_output: true
/// "#;
///
/// let _decoded: AggregatorConfig = serde_yaml::from_str(yaml_config).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Address on which this server should listen for connections and serve its
    /// API endpoints.
    // TODO: Options for terminating TLS, unless that gets handled in a load
    // balancer?
    pub listen_address: SocketAddr,
    /// The aggregator's database configuration.
    pub database: DbConfig,
    /// Logging configuration.
    #[serde(default)]
    pub logging_config: TraceConfiguration,
}

/// Non-secret configuration options for the Janus Aggregation Job Creator job.
///
/// # Examples
///
/// ```
/// use janus_server::config::AggregationJobCreatorConfig;
///
/// let yaml_config = r#"
/// ---
/// database:
///   url: "postgres://postgres:postgres@localhost:5432/postgres"
/// logging_config: # logging_config is optional
///   force_json_output: true
/// tasks_update_frequency_secs: 3600
/// aggregation_job_creation_interval_secs: 60
/// min_aggregation_job_size: 100
/// max_aggregation_job_size: 500
/// "#;
///
/// let _decoded: AggregationJobCreatorConfig = serde_yaml::from_str(yaml_config).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregationJobCreatorConfig {
    /// Configuration for the database backend to connect to.
    pub database: DbConfig,
    /// Logging configuration.
    #[serde(default)]
    pub logging_config: TraceConfiguration,
    /// How frequently we look for new tasks to start creating aggregation jobs for, in seconds.
    pub tasks_update_frequency_secs: u64,
    /// How frequently we attempt to create new aggregation jobs for each task, in seconds.
    pub aggregation_job_creation_interval_secs: u64,
    /// The minimum number of client reports to include in an aggregation job. Applies to the
    /// "current" batch unit only; historical batch units will create aggregation jobs of any size,
    /// on the theory that almost all reports will have be received for these batch units already.
    pub min_aggregation_job_size: usize,
    /// The maximum number of client reports to include in an aggregation job.
    pub max_aggregation_job_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn generate_db_config() -> DbConfig {
        DbConfig {
            url: Url::parse("postgres://postgres:postgres@localhost:5432/postgres").unwrap(),
        }
    }

    #[test]
    fn roundtrip_db_config() {
        let db_config = generate_db_config();
        let encoded = serde_yaml::to_string(&db_config).unwrap();
        let decoded: DbConfig = serde_yaml::from_str(&encoded).unwrap();
        assert_eq!(db_config, decoded);
    }

    #[test]
    fn roundtrip_aggregator_config() {
        let aggregator_config = AggregatorConfig {
            listen_address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8080),
            database: generate_db_config(),
            logging_config: TraceConfiguration::default(),
        };

        let encoded = serde_yaml::to_string(&aggregator_config).unwrap();
        let decoded: AggregatorConfig = serde_yaml::from_str(&encoded).unwrap();
        assert_eq!(aggregator_config, decoded);
    }

    #[test]
    fn roundtrip_aggregation_job_creator_config() {
        let config = AggregationJobCreatorConfig {
            database: generate_db_config(),
            logging_config: TraceConfiguration::default(),
            tasks_update_frequency_secs: 3600,
            aggregation_job_creation_interval_secs: 60,
            min_aggregation_job_size: 100,
            max_aggregation_job_size: 500,
        };

        let encoded = serde_yaml::to_string(&config).unwrap();
        let decoded_config: AggregationJobCreatorConfig = serde_yaml::from_str(&encoded).unwrap();
        assert_eq!(config, decoded_config);
    }
}
