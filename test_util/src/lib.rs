use assert_matches::assert_matches;
use janus::{
    message::{Duration, Nonce, Time},
    time::Clock,
};
use prio::{
    codec::Encode,
    vdaf::{self, PrepareTransition, VdafError},
};
use rand::{thread_rng, Rng};
use ring::aead::{LessSafeKey, UnboundKey, AES_128_GCM};
use std::sync::{Arc, Mutex};

pub mod dummy_vdaf;
pub mod runtime;

/// The Janus database schema.
pub static SCHEMA: &str = include_str!("../../db/schema.sql");

/// This macro injects definitions of `DbHandle` and `ephemeral_datastore()`, for use in tests.
/// It should be invoked once per binary target, and then `ephemeral_datastore()` can be called
/// to set up a database for test purposes. This depends on `janus_server::datastore::Datastore`,
/// `janus_server::datastore::Crypter`, and `janus_server::time::Clock` already being imported into
/// scope, and it expects the following crates to be available: `deadpool_postgres`, `lazy_static`,
/// `ring`, `testcontainers`, `tokio_postgres`, and `tracing`.
#[macro_export]
macro_rules! define_ephemeral_datastore {
    () => {
        ::lazy_static::lazy_static! {
            static ref CONTAINER_CLIENT: ::testcontainers::clients::Cli = ::testcontainers::clients::Cli::default();
        }

        /// DbHandle represents a handle to a running (ephemeral) database. Dropping this value
        /// causes the database to be shut down & cleaned up.
        pub struct DbHandle {
            _db_container: ::testcontainers::Container<'static, ::testcontainers::images::postgres::Postgres>,
            connection_string: String,
            port_number: u16,
            datastore_key_bytes: Vec<u8>,
        }

        impl DbHandle {
            /// Get a PostgreSQL connection string to connect to the temporary database.
            pub fn connection_string(&self) -> &str {
                &self.connection_string
            }

            pub fn datastore_key_bytes(&self) -> &[u8] {
                &self.datastore_key_bytes
            }

            /// Get the port number that the temporary database is exposed on, via the 127.0.0.1
            /// loopback interface.
            pub fn port_number(&self) -> u16 {
                self.port_number
            }

            /// Open an interactive terminal to the database in a new terminal window, and block
            /// until the user exits from the terminal. This is intended to be used while
            /// debugging tests.
            ///
            /// By default, this will invoke `gnome-terminal`, which is readily available on
            /// GNOME-based Linux distributions. To use a different terminal, set the environment
            /// variable `JANUS_SHELL_CMD` to a shell command that will open a new terminal window
            /// of your choice. This command line should include a "{}" in the position appropriate
            /// for what command the terminal should run when it opens. A `psql` invocation will
            /// be substituted in place of the "{}". Note that this shell command must not exit
            /// immediately once the terminal is spawned; it should continue running as long as the
            /// terminal is open. If the command provided exits too soon, then the test will
            /// continue running without intervention, leading to the test's database shutting
            /// down.
            ///
            /// # Example
            ///
            /// ```text
            /// JANUS_SHELL_CMD='xterm -e {}' cargo test
            /// ```
            pub fn interactive_db_terminal(&self) {
                let mut command = match ::std::env::var("JANUS_SHELL_CMD") {
                    Ok(shell_cmd) => {
                        if let None = shell_cmd.find("{}") {
                            panic!("JANUS_SHELL_CMD should contain a \"{{}}\" to denote where the database command should be substituted");
                        }

                        #[cfg(not(windows))]
                        let mut command = {
                            let mut command = ::std::process::Command::new("sh");
                            command.arg("-c");
                            command
                        };

                        #[cfg(windows)]
                        let mut command = {
                            let mut command = ::std::process::Command::new("cmd.exe");
                            command.arg("/c");
                            command
                        };

                        let psql_command = format!(
                            "psql --host=127.0.0.1 --user=postgres -p {}",
                            self.port_number(),
                        );
                        command.arg(shell_cmd.replacen("{}", &psql_command, 1));
                        command
                    }
                    Err(::std::env::VarError::NotPresent) => {
                        let mut command = ::std::process::Command::new("gnome-terminal");
                        command.args(["--wait", "--", "psql", "--host=127.0.0.1", "--user=postgres", "-p"]);
                        command.arg(format!("{}", self.port_number()));
                        command
                    }
                    Err(::std::env::VarError::NotUnicode(_)) => {
                        panic!("JANUS_SHELL_CMD contains invalid unicode data");
                    }
                };
                command.spawn().unwrap().wait().unwrap();
            }
        }

        impl Drop for DbHandle {
            fn drop(&mut self) {
                ::tracing::trace!(connection_string = %self.connection_string, "Dropping ephemeral Postgres container");
            }
        }

        /// ephemeral_datastore creates a new Datastore instance backed by an ephemeral database which
        /// has the Janus schema applied but is otherwise empty.
        ///
        /// Dropping the second return value causes the database to be shut down & cleaned up.
        pub async fn ephemeral_datastore<C: Clock>(clock: C) -> (Datastore<C>, DbHandle) {
            // Start an instance of Postgres running in a container.
            let db_container =
                CONTAINER_CLIENT.run(::testcontainers::RunnableImage::from(::testcontainers::images::postgres::Postgres::default()).with_tag("14-alpine"));

            // Create a connection pool whose clients will talk to our newly-running instance of Postgres.
            const POSTGRES_DEFAULT_PORT: u16 = 5432;
            let port_number = db_container.get_host_port_ipv4(POSTGRES_DEFAULT_PORT);
            let connection_string = format!(
                "postgres://postgres:postgres@127.0.0.1:{}/postgres",
                port_number,
            );
            ::tracing::trace!("Postgres container is up with URL {}", connection_string);
            let cfg = <::tokio_postgres::Config as std::str::FromStr>::from_str(&connection_string).unwrap();
            let conn_mgr = ::deadpool_postgres::Manager::new(cfg, ::tokio_postgres::NoTls);
            let pool = ::deadpool_postgres::Pool::builder(conn_mgr).build().unwrap();

            // Create a crypter with a random (ephemeral) key.
            let datastore_key_bytes = ::janus_test_util::generate_aead_key_bytes();
            let datastore_key =
                ::ring::aead::LessSafeKey::new(::ring::aead::UnboundKey::new(&ring::aead::AES_128_GCM, &datastore_key_bytes).unwrap());
            let crypter = Crypter::new(vec![datastore_key]);

            // Connect to the database & run our schema.
            let client = pool.get().await.unwrap();
            client.batch_execute(::janus_test_util::SCHEMA).await.unwrap();

            (
                Datastore::new(pool, crypter, clock),
                DbHandle {
                    _db_container: db_container,
                    connection_string,
                    port_number,
                    datastore_key_bytes,
                },
            )
        }
    };
}

pub fn generate_aead_key_bytes() -> Vec<u8> {
    let mut key_bytes = vec![0u8; AES_128_GCM.key_len()];
    thread_rng().fill(&mut key_bytes[..]);
    key_bytes
}

pub fn generate_aead_key() -> LessSafeKey {
    let unbound_key = UnboundKey::new(&AES_128_GCM, &generate_aead_key_bytes()).unwrap();
    LessSafeKey::new(unbound_key)
}

/// A mock clock for use in testing. Clones are identical: all clones of a given MockClock will
/// be controlled by a controller retrieved from any of the clones.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MockClock {
    /// The time that this clock will return from [`Self::now`].
    current_time: Arc<Mutex<Time>>,
}

impl MockClock {
    pub fn new(when: Time) -> MockClock {
        MockClock {
            current_time: Arc::new(Mutex::new(when)),
        }
    }

    pub fn advance(&self, dur: Duration) {
        let mut current_time = self.current_time.lock().unwrap();
        *current_time = current_time.add(dur).unwrap();
    }
}

impl Clock for MockClock {
    fn now(&self) -> Time {
        let current_time = self.current_time.lock().unwrap();
        *current_time
    }
}

impl Default for MockClock {
    fn default() -> Self {
        Self {
            // Sunday, September 9, 2001 1:46:40 AM UTC
            current_time: Arc::new(Mutex::new(Time::from_seconds_since_epoch(1000000000))),
        }
    }
}

/// A transcript of a VDAF run. All fields are indexed by natural role index (i.e., index 0 =
/// leader, index 1 = helper).
pub struct VdafTranscript<const L: usize, V: vdaf::Aggregator<L>>
where
    for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
{
    pub input_shares: Vec<V::InputShare>,
    pub prepare_transitions: Vec<Vec<PrepareTransition<V, L>>>,
    pub prepare_messages: Vec<V::PrepareMessage>,
}

/// run_vdaf runs a VDAF state machine from sharding through to generating an output share,
/// returning a "transcript" of all states & messages.
pub fn run_vdaf<const L: usize, V: vdaf::Aggregator<L> + vdaf::Client>(
    vdaf: &V,
    verify_key: &[u8; L],
    aggregation_param: &V::AggregationParam,
    nonce: Nonce,
    measurement: &V::Measurement,
) -> VdafTranscript<L, V>
where
    for<'a> &'a V::AggregateShare: Into<Vec<u8>>,
{
    // Shard inputs into input shares, and initialize the initial PrepareTransitions.
    let input_shares = vdaf.shard(measurement).unwrap();
    let encoded_nonce = nonce.get_encoded();
    let mut prep_trans: Vec<Vec<PrepareTransition<V, L>>> = input_shares
        .iter()
        .enumerate()
        .map(|(agg_id, input_share)| {
            let (prep_state, prep_share) = vdaf.prepare_init(
                verify_key,
                agg_id,
                aggregation_param,
                &encoded_nonce,
                input_share,
            )?;
            Ok(vec![PrepareTransition::Continue(prep_state, prep_share)])
        })
        .collect::<Result<Vec<Vec<PrepareTransition<V, L>>>, VdafError>>()
        .unwrap();
    let mut prep_msgs = Vec::new();

    // Repeatedly step the VDAF until we reach a terminal state.
    loop {
        // Gather messages from last round & combine them into next round's message; if any
        // participants have reached a terminal state (Finish or Fail), we are done.
        let mut prep_shares = Vec::new();
        for pts in &prep_trans {
            match pts.last().unwrap() {
                PrepareTransition::<V, L>::Continue(_, prep_share) => {
                    prep_shares.push(prep_share.clone())
                }
                _ => {
                    return VdafTranscript {
                        input_shares,
                        prepare_transitions: prep_trans,
                        prepare_messages: prep_msgs,
                    }
                }
            }
        }
        let prep_msg = vdaf.prepare_preprocess(prep_shares).unwrap();
        prep_msgs.push(prep_msg.clone());

        // Compute each participant's next transition.
        for pts in &mut prep_trans {
            let prep_state = assert_matches!(pts.last().unwrap(), PrepareTransition::<V, L>::Continue(prep_state, _) => prep_state).clone();
            pts.push(vdaf.prepare_step(prep_state, prep_msg.clone()).unwrap());
        }
    }
}
