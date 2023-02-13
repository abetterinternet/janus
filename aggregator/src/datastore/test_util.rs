use super::{Crypter, Datastore};
use deadpool_postgres::{Manager, Pool};
use janus_core::time::Clock;
use lazy_static::lazy_static;
use rand::{distributions::Standard, random, thread_rng, Rng};
use ring::aead::{LessSafeKey, UnboundKey, AES_128_GCM};
use std::{
    env::{self, VarError},
    mem::take,
    process::Command,
    str::FromStr,
    sync::{Arc, Barrier, Weak},
    thread::{self, JoinHandle},
};
use testcontainers::{images::postgres::Postgres, RunnableImage};
use tokio::sync::{oneshot, Mutex};
use tokio_postgres::{connect, Config, NoTls};
use tracing::trace;

struct EphemeralDatabase {
    port_number: u16,
    shutdown_barrier: Arc<Barrier>,
    join_handle: Option<JoinHandle<()>>,
}

impl EphemeralDatabase {
    async fn shared() -> Arc<Self> {
        // (once Weak::new is stabilized as a const function, replace this with a normal static
        // variable)
        lazy_static! {
            static ref EPHEMERAL_DATABASE: Mutex<Weak<EphemeralDatabase>> = Mutex::new(Weak::new());
        }

        let mut g = EPHEMERAL_DATABASE.lock().await;
        if let Some(ephemeral_database) = g.upgrade() {
            return ephemeral_database;
        }

        let ephemeral_database = Arc::new(EphemeralDatabase::start().await);
        *g = Arc::downgrade(&ephemeral_database);
        ephemeral_database
    }

    async fn start() -> Self {
        let (port_tx, port_rx) = oneshot::channel();
        let shutdown_barrier = Arc::new(Barrier::new(2));
        let join_handle = thread::spawn({
            let shutdown_barrier = Arc::clone(&shutdown_barrier);
            move || {
                // Start an instance of Postgres running in a container.
                let container_client = testcontainers::clients::Cli::default();
                let db_container = container_client
                    .run(RunnableImage::from(Postgres::default()).with_tag("14-alpine"));
                const POSTGRES_DEFAULT_PORT: u16 = 5432;
                let port_number = db_container.get_host_port_ipv4(POSTGRES_DEFAULT_PORT);
                trace!("Postgres container is up with port {port_number}");
                port_tx.send(port_number).unwrap();

                // Wait for the barrier as a shutdown signal.
                shutdown_barrier.wait();
                trace!("Shutting down Postgres container with port {port_number}");
            }
        });
        let port_number = port_rx.await.unwrap();

        Self {
            port_number,
            shutdown_barrier,
            join_handle: Some(join_handle),
        }
    }

    fn connection_string(&self, db_name: &str) -> String {
        format!(
            "postgres://postgres:postgres@127.0.0.1:{}/{db_name}",
            self.port_number
        )
    }
}

impl Drop for EphemeralDatabase {
    fn drop(&mut self) {
        // Wait on the shutdown barrier, which will cause the container-management thread to
        // begin shutdown. Then wait for the container-management thread itself to terminate.
        // This guarantees container shutdown finishes before dropping the EphemeralDatabase
        // completes.
        self.shutdown_barrier.wait();
        if let Some(join_handle) = take(&mut self.join_handle) {
            join_handle.join().unwrap();
        }
    }
}

/// EphemeralDatastore represents an ephemeral datastore instance. It has methods allowing
/// creation of Datastores, as well as the ability to retrieve the underlying connection pool.
///
/// Dropping the EphemeralDatastore will cause it to be shut down & cleaned up.
pub struct EphemeralDatastore {
    db: Arc<EphemeralDatabase>,
    connection_string: String,
    pool: Pool,
    datastore_key_bytes: Vec<u8>,
}

impl EphemeralDatastore {
    /// Creates a Datastore instance based on this EphemeralDatastore. All returned Datastore
    /// instances will refer to the same underlying durable state.
    pub fn datastore<C: Clock>(&self, clock: C) -> Datastore<C> {
        let datastore_key =
            LessSafeKey::new(UnboundKey::new(&AES_128_GCM, &self.datastore_key_bytes).unwrap());
        let crypter = Crypter::new(Vec::from([datastore_key]));
        Datastore::new(self.pool(), crypter, clock)
    }

    /// Retrieves the connection pool used for this EphemeralDatastore. Typically, this would be
    /// used only by tests which need to run custom SQL.
    pub fn pool(&self) -> Pool {
        self.pool.clone()
    }

    /// Retrieves the connection string used to connect to this EphemeralDatastore.
    pub fn connection_string(&self) -> &str {
        &self.connection_string
    }

    /// Get the bytes of the key used to encrypt sensitive datastore values.
    pub fn datastore_key_bytes(&self) -> &[u8] {
        &self.datastore_key_bytes
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
        let mut command = match env::var("JANUS_SHELL_CMD") {
            Ok(shell_cmd) => {
                if !shell_cmd.contains("{}") {
                    panic!("JANUS_SHELL_CMD should contain a \"{{}}\" to denote where the database command should be substituted");
                }

                #[cfg(not(windows))]
                let mut command = {
                    let mut command = Command::new("sh");
                    command.arg("-c");
                    command
                };

                #[cfg(windows)]
                let mut command = {
                    let mut command = Command::new("cmd.exe");
                    command.arg("/c");
                    command
                };

                let psql_command = format!(
                    "psql --host=127.0.0.1 --user=postgres -p {}",
                    self.db.port_number,
                );
                command.arg(shell_cmd.replacen("{}", &psql_command, 1));
                command
            }

            Err(VarError::NotPresent) => {
                let mut command = Command::new("gnome-terminal");
                command.args([
                    "--wait",
                    "--",
                    "psql",
                    "--host=127.0.0.1",
                    "--user=postgres",
                    "-p",
                ]);
                command.arg(format!("{}", self.db.port_number));
                command
            }

            Err(VarError::NotUnicode(_)) => {
                panic!("JANUS_SHELL_CMD contains invalid unicode data");
            }
        };
        command.spawn().unwrap().wait().unwrap();
    }
}

/// Creates a new, empty EphemeralDatastore with no schema applied. Almost all uses will want to
/// call `ephemeral_datastore` instead, which applies the standard schema.
pub async fn ephemeral_datastore_no_schema() -> EphemeralDatastore {
    let db = EphemeralDatabase::shared().await;
    let db_name = format!("janus_test_{}", hex::encode(random::<[u8; 16]>()));

    // Create Postgres DB & apply schema.
    let (client, conn) = connect(&db.connection_string("postgres"), NoTls)
        .await
        .unwrap();
    tokio::spawn(async move { conn.await.unwrap() }); // automatically stops after Client is dropped
    client
        .batch_execute(&format!("CREATE DATABASE {db_name}"))
        .await
        .unwrap();

    // Create a connection pool for the newly-created database.
    let connection_string = db.connection_string(&db_name);
    let cfg = Config::from_str(&connection_string).unwrap();
    let conn_mgr = Manager::new(cfg, NoTls);
    let pool = Pool::builder(conn_mgr).build().unwrap();

    EphemeralDatastore {
        db,
        connection_string,
        pool,
        datastore_key_bytes: generate_aead_key_bytes(),
    }
}

/// Creates a new, empty EphemeralDatastore.
pub async fn ephemeral_datastore() -> EphemeralDatastore {
    let ephemeral_datastore = ephemeral_datastore_no_schema().await;
    let client = ephemeral_datastore.pool().get().await.unwrap();
    client
        .batch_execute(include_str!("../../../db/schema.sql"))
        .await
        .unwrap();
    ephemeral_datastore
}

pub fn generate_aead_key_bytes() -> Vec<u8> {
    thread_rng()
        .sample_iter(Standard)
        .take(AES_128_GCM.key_len())
        .collect()
}

pub fn generate_aead_key() -> LessSafeKey {
    let unbound_key = UnboundKey::new(&AES_128_GCM, &generate_aead_key_bytes()).unwrap();
    LessSafeKey::new(unbound_key)
}
