//! Janus datastore (durable storage) implementation.

use self::models::{
    AcquiredAggregationJob, AcquiredCollectionJob, AggregateShareJob, AggregationJob,
    AggregatorRole, AuthenticationTokenType, BatchAggregation, BatchAggregationState,
    BatchAggregationStateCode, CollectionJob, CollectionJobState, CollectionJobStateCode,
    GlobalHpkeKeypair, HpkeKeyState, LeaderStoredReport, Lease, LeaseToken, OutstandingBatch,
    ReportAggregation, ReportAggregationMetadata, ReportAggregationMetadataState,
    ReportAggregationState, ReportAggregationStateCode, SqlInterval, TaskAggregationCounter,
    TaskUploadCounter,
};
#[cfg(feature = "test-util")]
use crate::VdafHasAggregationParameter;
use crate::{
    query_type::{AccumulableQueryType, CollectableQueryType},
    task::{self, AggregatorTask, AggregatorTaskParameters},
    taskprov::PeerAggregator,
    SecretBytes,
};
use chrono::NaiveDateTime;
use futures::future::try_join_all;
use janus_core::{
    auth_tokens::AuthenticationToken,
    hpke::{HpkeKeypair, HpkePrivateKey},
    time::{Clock, TimeExt},
    vdaf::VdafInstance,
};
use janus_messages::{
    query_type::{FixedSize, QueryType, TimeInterval},
    AggregationJobId, BatchId, CollectionJobId, Duration, Extension, HpkeCiphertext, HpkeConfig,
    HpkeConfigId, Interval, PrepareResp, Query, ReportId, ReportIdChecksum, ReportMetadata,
    ReportShare, Role, TaskId, Time,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter},
    KeyValue,
};
use postgres_types::{FromSql, Json, Timestamp, ToSql};
use prio::{
    codec::{decode_u16_items, encode_u16_items, CodecError, Decode, Encode, ParameterizedDecode},
    topology::ping_pong::PingPongTransition,
    vdaf,
};
use rand::random;
use ring::aead::{self, LessSafeKey, AES_128_GCM};
use std::{
    collections::HashMap,
    convert::TryFrom,
    fmt::{Debug, Display},
    future::Future,
    io::Cursor,
    mem::size_of,
    ops::RangeInclusive,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration as StdDuration, Instant},
};
use tokio::{sync::Barrier, try_join};
use tokio_postgres::{error::SqlState, row::RowIndex, IsolationLevel, Row, Statement, ToStatement};
use tracing::{error, Level};
use url::Url;

pub mod models;
#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util;
#[cfg(test)]
mod tests;

/// This macro stamps out an array of schema versions supported by this version of Janus and an
/// [`rstest_reuse`][1] template that can be applied to tests to have them run against all supported
/// schema versions.
///
/// [1]: https://docs.rs/rstest_reuse/latest/rstest_reuse/
macro_rules! supported_schema_versions {
    ( $i_latest:literal $(,)? $( $i:literal ),* ) => {
        const SUPPORTED_SCHEMA_VERSIONS: &[i64] = &[$i_latest, $($i),*];

        #[cfg(test)]
        #[rstest_reuse::template]
        #[rstest::rstest]
        // Test the latest supported schema version.
        #[case(ephemeral_datastore_schema_version($i_latest))]
        // Test the remaining supported schema versions.
        $(#[case(ephemeral_datastore_schema_version($i))])*
        // Test the remaining supported schema versions by taking a
        // database at the latest schema and downgrading it.
        $(#[case(ephemeral_datastore_schema_version_by_downgrade($i))])*
        async fn schema_versions_template(
            #[future(awt)]
            #[case]
            ephemeral_datastore: EphemeralDatastore,
        ) {
            // This is an rstest template and never gets run.
        }
    }
}

// List of schema versions that this version of Janus can safely run on. If any other schema
// version is seen, [`Datastore::new`] fails.
//
// Note that the latest supported version must be first in the list.
supported_schema_versions!(7);

/// Datastore represents a datastore for Janus, with support for transactional reads and writes.
/// In practice, Datastore instances are currently backed by a PostgreSQL database.
pub struct Datastore<C: Clock> {
    pool: deadpool_postgres::Pool,
    crypter: Crypter,
    clock: C,
    task_infos: Arc<Mutex<HashMap<TaskId, TaskInfo>>>,
    transaction_status_counter: Counter<u64>,
    transaction_retry_histogram: Histogram<u64>,
    rollback_error_counter: Counter<u64>,
    transaction_duration_histogram: Histogram<f64>,
    transaction_pool_wait_histogram: Histogram<f64>,
    max_transaction_retries: u64,
}

impl<C: Clock> Debug for Datastore<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Datastore")
    }
}

impl<C: Clock> Datastore<C> {
    /// `new` creates a new Datastore using the provided connection pool. An error is returned if
    /// the current database migration version is not supported by this version of Janus.
    pub async fn new(
        pool: deadpool_postgres::Pool,
        crypter: Crypter,
        clock: C,
        meter: &Meter,
        max_transaction_retries: u64,
    ) -> Result<Datastore<C>, Error> {
        Self::new_with_supported_versions(
            pool,
            crypter,
            clock,
            meter,
            SUPPORTED_SCHEMA_VERSIONS,
            max_transaction_retries,
        )
        .await
    }

    async fn new_with_supported_versions(
        pool: deadpool_postgres::Pool,
        crypter: Crypter,
        clock: C,
        meter: &Meter,
        supported_schema_versions: &[i64],
        max_transaction_retries: u64,
    ) -> Result<Datastore<C>, Error> {
        let datastore = Self::new_without_supported_versions(
            pool,
            crypter,
            clock,
            meter,
            max_transaction_retries,
        )
        .await;

        let (current_version, migration_description) = datastore
            .run_tx("check schema version", |tx| {
                Box::pin(async move { tx.get_current_schema_migration_version().await })
            })
            .await?;

        if !supported_schema_versions.contains(&current_version) {
            return Err(Error::DbState(format!(
                "unsupported schema version {current_version} / {migration_description}"
            )));
        }

        Ok(datastore)
    }

    /// Creates a new datastore using the provided connection pool.
    pub async fn new_without_supported_versions(
        pool: deadpool_postgres::Pool,
        crypter: Crypter,
        clock: C,
        meter: &Meter,
        max_transaction_retries: u64,
    ) -> Datastore<C> {
        let transaction_status_counter = meter
            .u64_counter(TRANSACTION_METER_NAME)
            .with_description("Count of database transactions run, with their status.")
            .with_unit("{transaction}")
            .init();
        let rollback_error_counter = meter
            .u64_counter(TRANSACTION_ROLLBACK_METER_NAME)
            .with_description(concat!(
                "Count of errors received when rolling back a database transaction, ",
                "with their PostgreSQL error code.",
            ))
            .with_unit("{error}")
            .init();
        let transaction_retry_histogram = meter
            .u64_histogram(TRANSACTION_RETRIES_METER_NAME)
            .with_description("The number of retries before a transaction is committed or aborted.")
            .with_unit("{retry}")
            .init();
        let transaction_duration_histogram = meter
            .f64_histogram(TRANSACTION_DURATION_METER_NAME)
            .with_description(concat!(
                "Duration of database transactions. This counts only the time spent between the ",
                "BEGIN and COMMIT/ROLLBACK statements."
            ))
            .with_unit("s")
            .init();
        let transaction_pool_wait_histogram = meter
            .f64_histogram(TRANSACTION_POOL_WAIT_METER_NAME)
            .with_description(concat!(
                "Time spent waiting for a transaction to BEGIN, because it is waiting for a ",
                "slot to become available in the connection pooler."
            ))
            .with_unit("s")
            .init();

        Self {
            pool,
            crypter,
            clock,
            task_infos: Default::default(),
            transaction_status_counter,
            transaction_retry_histogram,
            rollback_error_counter,
            transaction_duration_histogram,
            transaction_pool_wait_histogram,
            max_transaction_retries,
        }
    }

    /// run_tx runs a transaction, whose body is determined by the given function. The transaction
    /// is committed if the body returns a successful value, and rolled back if the body returns an
    /// error value.
    ///
    /// The datastore will automatically retry some failures (e.g. serialization failures) by
    /// rolling back & retrying with a new transaction, so the given function should support being
    /// called multiple times. Values read from the transaction should not be considered as
    /// "finalized" until the transaction is committed, i.e. after `run_tx` is run to completion.
    ///
    /// This method requires a transaction name for use in database metrics.
    #[tracing::instrument(level = "trace", skip(self, f))]
    pub async fn run_tx<F, T>(&self, name: &'static str, f: F) -> Result<T, Error>
    where
        for<'a> F:
            Fn(&'a Transaction<C>) -> Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>,
    {
        let mut retry_count = 0;
        loop {
            let (mut rslt, retry) = self.run_tx_once(name, &f).await;
            let retries_exceeded = retry_count + 1 > self.max_transaction_retries;
            let status = match (rslt.as_ref(), retry) {
                (_, true) => {
                    if retries_exceeded {
                        "error_too_many_retries"
                    } else {
                        "retry"
                    }
                }
                (Ok(_), _) | (Err(Error::User(_)), _) => "success",
                (Err(Error::Db(_)), _) | (Err(Error::Pool(_)), _) => "error_db",
                (Err(_), _) => "error_other",
            };
            self.transaction_status_counter.add(
                1,
                &[KeyValue::new("status", status), KeyValue::new("tx", name)],
            );

            if retry {
                if retries_exceeded {
                    let err = rslt.err();
                    error!(
                        retry_count,
                        last_err = ?err,
                        "too many retries, aborting transaction"
                    );
                    rslt = Err(Error::TooManyRetries {
                        source: err.map(Box::new),
                    });
                } else {
                    retry_count += 1;
                    continue;
                }
            }

            self.transaction_retry_histogram
                .record(retry_count, &[KeyValue::new("tx", name)]);
            return rslt;
        }
    }

    #[tracing::instrument(level = "trace", skip(self, f))]
    async fn run_tx_once<F, T>(&self, name: &'static str, f: &F) -> (Result<T, Error>, bool)
    where
        for<'a> F:
            Fn(&'a Transaction<C>) -> Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>,
    {
        // Acquire connection from the connection pooler.
        let before = Instant::now();
        let result = self.pool.get().await;
        let elapsed = before.elapsed();
        // We don't record the transaction name for this metric, since it's not particularly
        // interesting. All transactions should get FIFO access to connections.
        self.transaction_pool_wait_histogram.record(
            elapsed.as_secs_f64(),
            &[KeyValue::new(
                "status",
                if result.is_err() { "error" } else { "success" },
            )],
        );
        let mut client = match result {
            Ok(client) => client,
            Err(err) => return (Err(err.into()), false),
        };

        // Open transaction.
        let before = Instant::now();
        let raw_tx = match client
            .build_transaction()
            .isolation_level(IsolationLevel::RepeatableRead)
            .start()
            .await
        {
            Ok(raw_tx) => raw_tx,
            Err(err) => return (Err(err.into()), false),
        };
        let tx = Transaction {
            raw_tx,
            crypter: &self.crypter,
            clock: &self.clock,
            name,
            task_infos: Arc::clone(&self.task_infos),
            retry: AtomicBool::new(false),
            op_group: Mutex::new(Arc::new(Mutex::new(OperationGroup::Running(0)))),
        };

        // Run user-provided function with the transaction, then commit/rollback based on result.
        let rslt = f(&tx).await;
        let (raw_tx, retry) = (tx.raw_tx, tx.retry);
        let rslt = match (rslt, retry.load(Ordering::Relaxed)) {
            // Commit.
            (Ok(val), false) => match check_error(&retry, raw_tx.commit().await) {
                Ok(()) => Ok(val),
                Err(err) => Err(err.into()),
            },

            // Rollback.
            (rslt, _) => {
                if let Err(rollback_err) = check_error(&retry, raw_tx.rollback().await) {
                    error!("Couldn't roll back transaction: {rollback_err}");
                    self.rollback_error_counter.add(
                        1,
                        &[KeyValue::new(
                            "code",
                            rollback_err
                                .code()
                                .map(SqlState::code)
                                .unwrap_or("N/A")
                                .to_string(),
                        )],
                    );
                };
                // We return `rslt` unconditionally here: it will either be an error, or we have the
                // retry flag set so that even if `rslt` is a success we will be retrying the entire
                // transaction & the result of this attempt doesn't matter.
                rslt
            }
        };
        let elapsed = before.elapsed();
        self.transaction_duration_histogram
            .record(elapsed.as_secs_f64(), &[KeyValue::new("tx", name)]);

        (rslt, retry.load(Ordering::Relaxed))
    }

    /// See [`Datastore::run_tx`]. This method provides a placeholder transaction name. It is useful
    /// for tests where the transaction name is not important.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    #[tracing::instrument(level = "trace", skip(self, f))]
    pub fn run_unnamed_tx<'s, F, T>(&'s self, f: F) -> impl Future<Output = Result<T, Error>> + 's
    where
        F: 's,
        T: 's,
        for<'a> F:
            Fn(&'a Transaction<C>) -> Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>,
    {
        self.run_tx("default", f)
    }

    /// Write a task into the datastore.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    pub async fn put_aggregator_task(&self, task: &AggregatorTask) -> Result<(), Error> {
        self.run_tx("test-put-task", |tx| {
            let task = task.clone();
            Box::pin(async move { tx.put_aggregator_task(&task).await })
        })
        .await
    }

    /// Write an arbitrary global HPKE key to the datastore and place it in the
    /// [`HpkeKeyState::Active`] state.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    pub async fn put_global_hpke_key(&self) -> Result<HpkeKeypair, Error> {
        let keypair = HpkeKeypair::test();
        self.run_tx("test-put-global-hpke-key", |tx| {
            let keypair = keypair.clone();
            Box::pin(async move {
                tx.put_global_hpke_keypair(&keypair).await?;
                tx.set_global_hpke_keypair_state(keypair.config().id(), &HpkeKeyState::Active)
                    .await
            })
        })
        .await?;

        Ok(keypair)
    }
}

fn check_error<T>(
    retry: &AtomicBool,
    rslt: Result<T, tokio_postgres::Error>,
) -> Result<T, tokio_postgres::Error> {
    if let Err(err) = &rslt {
        if is_retryable_error(err) {
            retry.store(true, Ordering::Relaxed);
        }
    }
    rslt
}

fn is_retryable_error(err: &tokio_postgres::Error) -> bool {
    err.code().map_or(false, |code| {
        code == &SqlState::T_R_SERIALIZATION_FAILURE || code == &SqlState::T_R_DEADLOCK_DETECTED
    })
}

fn is_transaction_abort_error(err: &tokio_postgres::Error) -> bool {
    err.code()
        .map_or(false, |code| code == &SqlState::IN_FAILED_SQL_TRANSACTION)
}

pub const TRANSACTION_METER_NAME: &str = "janus_database_transactions";
pub const TRANSACTION_ROLLBACK_METER_NAME: &str = "janus_database_rollback_errors";
pub const TRANSACTION_RETRIES_METER_NAME: &str = "janus_database_transaction_retries";
pub const TRANSACTION_DURATION_METER_NAME: &str = "janus_database_transaction_duration";
pub const TRANSACTION_POOL_WAIT_METER_NAME: &str = "janus_database_pool_wait_duration";

/// Transaction represents an ongoing datastore transaction.
pub struct Transaction<'a, C: Clock> {
    raw_tx: deadpool_postgres::Transaction<'a>,
    crypter: &'a Crypter,
    clock: &'a C,
    name: &'a str,
    task_infos: Arc<Mutex<HashMap<TaskId, TaskInfo>>>,

    retry: AtomicBool,
    op_group: Mutex<Arc<Mutex<OperationGroup>>>, // locking discipline: outer lock before inner lock
}

enum OperationGroup {
    Running(usize),         // current operation count
    Draining(Arc<Barrier>), // barrier to wait upon to complete drain
}

impl<C: Clock> Transaction<'_, C> {
    // For some error modes, Postgres will return an error to the caller & then fail all future
    // statements within the same transaction with an "in failed SQL transaction" error. This
    // effectively means one statement will receive a "root cause" error and then all later
    // statements will receive an "in failed SQL transaction" error. In a pipelined scenario, if our
    // code is processing the results of these statements concurrently--e.g. because they are part
    // of a `try_join!`/`try_join_all` group--we might receive & handle one of the "in failed SQL
    // transaction" errors before we handle the "root cause" error, which might cause the "root
    // cause" error's future to be cancelled before we evaluate it. If the "root cause" error would
    // trigger a retry, this would mean we would skip a DB-based retry when one was warranted.
    //
    // To fix this problem, we (internally) wrap all direct DB operations in `run_op`. This function
    // groups concurrent database operations into "operation groups", which allow us to wait for all
    // operations in the group to complete (this waiting operation is called "draining"). If we ever
    // observe an "in failed SQL transaction" error, we drain the operation group before returning.
    // Under the assumption that the "root cause" error is concurrent with the "in failed SQL
    // transactions" errors, this guarantees we will evaluate the "root cause" error for retry
    // before any errors make their way out of the transaction code.
    async fn run_op<T>(
        &self,
        op: impl Future<Output = Result<T, tokio_postgres::Error>>,
    ) -> Result<T, tokio_postgres::Error> {
        // Enter.
        //
        // Before we can run the operation, we need to join this operation into an operation group.
        // Retrieve the current operation group & join it.
        let op_group = {
            let mut tx_op_group = self.op_group.lock().unwrap();
            let new_tx_op_group = {
                let mut op_group = tx_op_group.lock().unwrap();
                match &*op_group {
                    OperationGroup::Running(op_count) => {
                        // If the current op group is running, join it by incrementing the operation
                        // count.
                        *op_group = OperationGroup::Running(*op_count + 1);
                        None
                    }

                    OperationGroup::Draining { .. } => {
                        // If the current op group is draining, we can't join it; instead, create a
                        // new op group to join, and store it as the transaction's current operation
                        // group.
                        Some(Arc::new(Mutex::new(OperationGroup::Running(1))))
                    }
                }
            };
            if let Some(new_tx_op_group) = new_tx_op_group {
                *tx_op_group = new_tx_op_group;
            }
            Arc::clone(&tx_op_group)
        };

        // Run operation, and check if error triggers a retry or requires a drain.
        let rslt = check_error(&self.retry, op.await);
        let needs_drain = rslt
            .as_ref()
            .err()
            .map_or(false, is_transaction_abort_error);

        // Exit.
        //
        // Before we are done running the operation, we have to leave the operation group. If the
        // operation group is running, we just need to decrement the count. If the operation group
        // is draining (because this or another operation encountered an error which requires a
        // drain), we have to wait until all operations in the group are ready to finish.
        let barrier = {
            let mut op_group = op_group.lock().unwrap();
            match &*op_group {
                OperationGroup::Running(op_count) => {
                    if needs_drain {
                        // If the operation group is running & we have determined we need to drain
                        // the operation group, change the operation group to Draining & wait on the
                        // barrier.
                        let barrier = Arc::new(Barrier::new(*op_count));
                        *op_group = OperationGroup::Draining(Arc::clone(&barrier));
                        Some(barrier)
                    } else {
                        // If the operation group is running & we don't need a drain, just decrement
                        // the operation count.
                        *op_group = OperationGroup::Running(op_count - 1);
                        None
                    }
                }

                // If the operation group is already draining, wait on the barrier.
                OperationGroup::Draining(barrier) => Some(Arc::clone(barrier)),
            }
        };
        if let Some(barrier) = barrier {
            barrier.wait().await;
        }
        rslt
    }

    async fn execute<T>(
        &self,
        statement: &T,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<u64, tokio_postgres::Error>
    where
        T: ?Sized + ToStatement,
    {
        self.run_op(self.raw_tx.execute(statement, params)).await
    }

    async fn prepare_cached(&self, query: &str) -> Result<Statement, tokio_postgres::Error> {
        self.run_op(self.raw_tx.prepare_cached(query)).await
    }

    async fn query<T>(
        &self,
        statement: &T,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Vec<Row>, tokio_postgres::Error>
    where
        T: ?Sized + ToStatement,
    {
        self.run_op(self.raw_tx.query(statement, params)).await
    }

    async fn query_one<T>(
        &self,
        statement: &T,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Row, tokio_postgres::Error>
    where
        T: ?Sized + ToStatement,
    {
        self.run_op(self.raw_tx.query_one(statement, params)).await
    }

    async fn query_opt<T>(
        &self,
        statement: &T,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Option<Row>, tokio_postgres::Error>
    where
        T: ?Sized + ToStatement,
    {
        self.run_op(self.raw_tx.query_opt(statement, params)).await
    }

    /// Returns the current schema version of the datastore and the description of the migration
    /// script that applied it.
    async fn get_current_schema_migration_version(&self) -> Result<(i64, String), Error> {
        let stmt = self
            .prepare_cached(
                "-- get_current_schema_migration_version()
SELECT version, description FROM _sqlx_migrations
WHERE success = TRUE ORDER BY version DESC LIMIT(1)",
            )
            .await?;
        let row = self.query_one(&stmt, &[]).await?;

        let version = row.get("version");
        let description = row.get("description");

        Ok((version, description))
    }

    /// Returns the clock used by this transaction.
    pub fn clock(&self) -> &C {
        self.clock
    }

    /// Writes a task into the datastore.
    #[tracing::instrument(skip(self, task), fields(task_id = ?task.id()), err)]
    pub async fn put_aggregator_task(&self, task: &AggregatorTask) -> Result<(), Error> {
        let now = self.clock.now().as_naive_date_time()?;
        // Main task insert.
        let stmt = self
            .prepare_cached(
                "-- put_aggregator_task()
INSERT INTO tasks (
    task_id, aggregator_role, peer_aggregator_endpoint, query_type, vdaf,
    max_batch_query_count, task_expiration, report_expiry_age, min_batch_size,
    time_precision, tolerable_clock_skew, collector_hpke_config, vdaf_verify_key,
    taskprov_task_info, aggregator_auth_token_type, aggregator_auth_token,
    aggregator_auth_token_hash, collector_auth_token_type,
    collector_auth_token_hash, created_at, updated_at, updated_by)
VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18,
    $19, $20, $21, $22
)
ON CONFLICT DO NOTHING",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task.id().as_ref(),
                    /* aggregator_role */ &AggregatorRole::from_role(*task.role())?,
                    /* peer_aggregator_endpoint */
                    &task.peer_aggregator_endpoint().as_str(),
                    /* query_type */ &Json(task.query_type()),
                    /* vdaf */ &Json(task.vdaf()),
                    /* max_batch_query_count */
                    &i64::try_from(task.max_batch_query_count())?,
                    /* task_expiration */
                    &task
                        .task_expiration()
                        .map(Time::as_naive_date_time)
                        .transpose()?,
                    /* report_expiry_age */
                    &task
                        .report_expiry_age()
                        .map(Duration::as_seconds)
                        .map(i64::try_from)
                        .transpose()?,
                    /* min_batch_size */ &i64::try_from(task.min_batch_size())?,
                    /* time_precision */
                    &i64::try_from(task.time_precision().as_seconds())?,
                    /* tolerable_clock_skew */
                    &i64::try_from(task.tolerable_clock_skew().as_seconds())?,
                    /* collector_hpke_config */
                    &task
                        .collector_hpke_config()
                        .map(|cfg| cfg.get_encoded())
                        .transpose()?,
                    /* vdaf_verify_key */
                    &self.crypter.encrypt(
                        "tasks",
                        task.id().as_ref(),
                        "vdaf_verify_key",
                        task.opaque_vdaf_verify_key().as_ref(),
                    )?,
                    /* taskprov_task_info */
                    &task.taskprov_task_info(),
                    /* aggregator_auth_token_type */
                    &task
                        .aggregator_auth_token()
                        .map(AuthenticationTokenType::from)
                        .or_else(|| {
                            task.aggregator_auth_token_hash()
                                .map(AuthenticationTokenType::from)
                        }),
                    /* aggregator_auth_token */
                    &task
                        .aggregator_auth_token()
                        .map(|token| {
                            self.crypter.encrypt(
                                "tasks",
                                task.id().as_ref(),
                                "aggregator_auth_token",
                                token.as_ref(),
                            )
                        })
                        .transpose()?,
                    /* aggregator_auth_token_hash */
                    &task
                        .aggregator_auth_token_hash()
                        .map(|token_hash| token_hash.as_ref()),
                    /* collector_auth_token_type */
                    &task
                        .collector_auth_token_hash()
                        .map(AuthenticationTokenType::from),
                    /* collector_auth_token */
                    &task
                        .collector_auth_token_hash()
                        .map(|token_hash| token_hash.as_ref()),
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                ],
            )
            .await?,
        )?;

        // HPKE keys.
        let mut hpke_config_ids: Vec<i16> = Vec::new();
        let mut hpke_configs: Vec<Vec<u8>> = Vec::new();
        let mut hpke_private_keys: Vec<Vec<u8>> = Vec::new();

        for hpke_keypair in task.hpke_keys().values() {
            let mut row_id = [0u8; TaskId::LEN + size_of::<u8>()];
            row_id[..TaskId::LEN].copy_from_slice(task.id().as_ref());
            row_id[TaskId::LEN..]
                .copy_from_slice(&u8::from(*hpke_keypair.config().id()).to_be_bytes());

            let encrypted_hpke_private_key = self.crypter.encrypt(
                "task_hpke_keys",
                &row_id,
                "private_key",
                hpke_keypair.private_key().as_ref(),
            )?;

            hpke_config_ids.push(u8::from(*hpke_keypair.config().id()) as i16);
            hpke_configs.push(hpke_keypair.config().get_encoded()?);
            hpke_private_keys.push(encrypted_hpke_private_key);
        }
        let stmt = self
            .prepare_cached(
                "-- put_aggregator_task()
INSERT INTO task_hpke_keys (
    task_id, created_at, updated_by, config_id, config, private_key
)
SELECT
    (SELECT id FROM tasks WHERE task_id = $1), $2, $3,
    * FROM UNNEST($4::SMALLINT[], $5::BYTEA[], $6::BYTEA[])",
            )
            .await?;
        let hpke_configs_params: &[&(dyn ToSql + Sync)] = &[
            /* task_id */ &task.id().as_ref(),
            /* created_at */ &self.clock.now().as_naive_date_time()?,
            /* updated_by */ &self.name,
            /* config_id */ &hpke_config_ids,
            /* configs */ &hpke_configs,
            /* private_keys */ &hpke_private_keys,
        ];
        self.execute(&stmt, hpke_configs_params).await?;

        Ok(())
    }

    /// Deletes a task from the datastore, along with all related data (client reports,
    /// aggregations, etc).
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn delete_task(&self, task_id: &TaskId) -> Result<(), Error> {
        // Deletion of other data implemented via ON DELETE CASCADE.
        let stmt = self
            .prepare_cached(
                "-- delete_task()
DELETE FROM tasks WHERE task_id = $1",
            )
            .await?;
        check_single_row_mutation(
            self.execute(&stmt, &[/* task_id */ &task_id.as_ref()])
                .await?,
        )?;
        Ok(())
    }

    /// Sets or unsets the expiration date of a task.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn update_task_expiration(
        &self,
        task_id: &TaskId,
        task_expiration: Option<&Time>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "-- update_task_expiration()
UPDATE tasks SET task_expiration = $1, updated_at = $2, updated_by = $3
   WHERE task_id = $4",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* task_expiration */
                    &task_expiration.map(Time::as_naive_date_time).transpose()?,
                    /* updated_at */ &self.clock.now().as_naive_date_time()?,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_id.as_ref(),
                ],
            )
            .await?,
        )
    }

    /// Fetch the task parameters corresponing to the provided `task_id`.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregator_task(
        &self,
        task_id: &TaskId,
    ) -> Result<Option<AggregatorTask>, Error> {
        let params: &[&(dyn ToSql + Sync)] = &[&task_id.as_ref()];
        let stmt = self
            .prepare_cached(
                "-- get_aggregator_task()
SELECT aggregator_role, peer_aggregator_endpoint, query_type, vdaf,
    max_batch_query_count, task_expiration, report_expiry_age, min_batch_size,
    time_precision, tolerable_clock_skew, collector_hpke_config, vdaf_verify_key,
    taskprov_task_info, aggregator_auth_token_type, aggregator_auth_token,
    aggregator_auth_token_hash, collector_auth_token_type, collector_auth_token_hash
FROM tasks WHERE task_id = $1",
            )
            .await?;
        let task_row = self.query_opt(&stmt, params);

        let stmt = self
            .prepare_cached(
                "-- get_aggregator_task()
SELECT config_id, config, private_key FROM task_hpke_keys
WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)",
            )
            .await?;
        let hpke_key_rows = self.query(&stmt, params);

        let (task_row, hpke_key_rows) = try_join!(task_row, hpke_key_rows,)?;
        task_row
            .map(|task_row| self.task_from_rows(task_id, &task_row, &hpke_key_rows))
            .transpose()
    }

    /// Fetch all the tasks in the database.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregator_tasks(&self) -> Result<Vec<AggregatorTask>, Error> {
        let stmt = self
            .prepare_cached(
                "-- get_aggregator_tasks()
SELECT task_id, aggregator_role, peer_aggregator_endpoint, query_type, vdaf,
    max_batch_query_count, task_expiration, report_expiry_age, min_batch_size,
    time_precision, tolerable_clock_skew, collector_hpke_config, vdaf_verify_key,
    taskprov_task_info, aggregator_auth_token_type, aggregator_auth_token,
    aggregator_auth_token_hash, collector_auth_token_type, collector_auth_token_hash
FROM tasks",
            )
            .await?;
        let task_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "-- get_aggregator_tasks()
SELECT (SELECT tasks.task_id FROM tasks WHERE tasks.id = task_hpke_keys.task_id),
config_id, config, private_key FROM task_hpke_keys",
            )
            .await?;
        let hpke_config_rows = self.query(&stmt, &[]);

        let (task_rows, hpke_config_rows) = try_join!(task_rows, hpke_config_rows,)?;

        let mut task_row_by_id = Vec::new();
        for row in task_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            task_row_by_id.push((task_id, row));
        }

        let mut hpke_config_rows_by_task_id: HashMap<TaskId, Vec<Row>> = HashMap::new();
        for row in hpke_config_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            hpke_config_rows_by_task_id
                .entry(task_id)
                .or_default()
                .push(row);
        }

        task_row_by_id
            .into_iter()
            .map(|(task_id, row)| {
                self.task_from_rows(
                    &task_id,
                    &row,
                    &hpke_config_rows_by_task_id
                        .remove(&task_id)
                        .unwrap_or_default(),
                )
            })
            .collect::<Result<_, _>>()
    }

    /// Construct an [`AggregatorTask`] from the contents of the provided (tasks) `Row` and
    /// `task_hpke_keys` rows.
    fn task_from_rows(
        &self,
        task_id: &TaskId,
        row: &Row,
        hpke_key_rows: &[Row],
    ) -> Result<AggregatorTask, Error> {
        // Scalar task parameters.
        let aggregator_role: AggregatorRole = row.get("aggregator_role");
        let peer_aggregator_endpoint = row.get::<_, String>("peer_aggregator_endpoint").parse()?;
        let query_type = row.try_get::<_, Json<task::QueryType>>("query_type")?.0;
        let vdaf = row.try_get::<_, Json<VdafInstance>>("vdaf")?.0;
        let max_batch_query_count = row.get_bigint_and_convert("max_batch_query_count")?;
        let task_expiration = row
            .get::<_, Option<NaiveDateTime>>("task_expiration")
            .as_ref()
            .map(Time::from_naive_date_time);
        let report_expiry_age = row
            .get_nullable_bigint_and_convert("report_expiry_age")?
            .map(Duration::from_seconds);
        let min_batch_size = row.get_bigint_and_convert("min_batch_size")?;
        let time_precision = Duration::from_seconds(row.get_bigint_and_convert("time_precision")?);
        let tolerable_clock_skew =
            Duration::from_seconds(row.get_bigint_and_convert("tolerable_clock_skew")?);
        let collector_hpke_config = row
            .get::<_, Option<Vec<u8>>>("collector_hpke_config")
            .map(|config| HpkeConfig::get_decoded(&config))
            .transpose()?;
        let encrypted_vdaf_verify_key: Vec<u8> = row.get::<_, Vec<u8>>("vdaf_verify_key");
        let vdaf_verify_key = self
            .crypter
            .decrypt(
                "tasks",
                task_id.as_ref(),
                "vdaf_verify_key",
                &encrypted_vdaf_verify_key,
            )
            .map(SecretBytes::new)?;
        let taskprov_task_info: Option<Vec<u8>> = row.get("taskprov_task_info");

        let aggregator_auth_token_type: Option<AuthenticationTokenType> =
            row.get("aggregator_auth_token_type");

        let aggregator_auth_token = row
            .get::<_, Option<Vec<u8>>>("aggregator_auth_token")
            .zip(aggregator_auth_token_type)
            .map(|(encrypted_token, token_type)| {
                token_type.as_authentication(&self.crypter.decrypt(
                    "tasks",
                    task_id.as_ref(),
                    "aggregator_auth_token",
                    &encrypted_token,
                )?)
            })
            .transpose()?;

        let aggregator_auth_token_hash = row
            .get::<_, Option<Vec<u8>>>("aggregator_auth_token_hash")
            .zip(aggregator_auth_token_type)
            .map(|(token_hash, token_type)| token_type.as_authentication_token_hash(&token_hash))
            .transpose()?;

        let collector_auth_token_hash = row
            .get::<_, Option<Vec<u8>>>("collector_auth_token_hash")
            .zip(row.get::<_, Option<AuthenticationTokenType>>("collector_auth_token_type"))
            .map(|(token_hash, token_type)| token_type.as_authentication_token_hash(&token_hash))
            .transpose()?;

        // HPKE keys.
        let mut hpke_keys = Vec::new();
        for row in hpke_key_rows {
            let config_id = u8::try_from(row.get::<_, i16>("config_id"))?;
            let config = HpkeConfig::get_decoded(row.get("config"))?;
            let encrypted_private_key: Vec<u8> = row.get("private_key");

            let mut row_id = [0u8; TaskId::LEN + size_of::<u8>()];
            row_id[..TaskId::LEN].copy_from_slice(task_id.as_ref());
            row_id[TaskId::LEN..].copy_from_slice(&config_id.to_be_bytes());

            let private_key = HpkePrivateKey::new(self.crypter.decrypt(
                "task_hpke_keys",
                &row_id,
                "private_key",
                &encrypted_private_key,
            )?);

            hpke_keys.push(HpkeKeypair::new(config, private_key));
        }

        let aggregator_parameters = match (
            aggregator_role,
            aggregator_auth_token,
            aggregator_auth_token_hash,
            collector_auth_token_hash,
            collector_hpke_config,
        ) {
            (
                AggregatorRole::Leader,
                Some(aggregator_auth_token),
                None,
                Some(collector_auth_token_hash),
                Some(collector_hpke_config),
            ) => AggregatorTaskParameters::Leader {
                aggregator_auth_token,
                collector_auth_token_hash,
                collector_hpke_config,
            },
            (
                AggregatorRole::Helper,
                None,
                Some(aggregator_auth_token_hash),
                None,
                Some(collector_hpke_config),
            ) => AggregatorTaskParameters::Helper {
                aggregator_auth_token_hash,
                collector_hpke_config,
            },
            (AggregatorRole::Helper, None, None, None, None) => {
                AggregatorTaskParameters::TaskprovHelper
            }
            values => {
                return Err(Error::DbState(format!(
                    "found task row with unexpected combination of values {values:?}",
                )));
            }
        };

        let mut task = AggregatorTask::new(
            *task_id,
            peer_aggregator_endpoint,
            query_type,
            vdaf,
            vdaf_verify_key,
            max_batch_query_count,
            task_expiration,
            report_expiry_age,
            min_batch_size,
            time_precision,
            tolerable_clock_skew,
            hpke_keys,
            aggregator_parameters,
        )?;
        if let Some(taskprov_task_info) = taskprov_task_info {
            task = task.with_taskprov_task_info(taskprov_task_info);
        }
        Ok(task)
    }

    /// Retrieves task IDs, optionally after some specified lower bound. This method returns tasks
    /// IDs in lexicographic order, but may not retrieve the IDs of all tasks in a single call. To
    /// retrieve additional task IDs, make additional calls to this method while specifying the
    /// `lower_bound` parameter to be the last task ID retrieved from the previous call.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_task_ids(&self, lower_bound: Option<TaskId>) -> Result<Vec<TaskId>, Error> {
        let lower_bound = lower_bound.map(|task_id| task_id.as_ref().to_vec());
        let stmt = self
            .prepare_cached(
                "-- get_task_ids()
SELECT task_id FROM tasks
WHERE task_id > $1 OR $1 IS NULL
ORDER BY task_id
LIMIT 5000",
            )
            .await?;
        self.query(&stmt, &[/* task_id */ &lower_bound])
            .await?
            .into_iter()
            .map(|row| Ok(TaskId::get_decoded(row.get("task_id"))?))
            .collect()
    }

    /// get_client_report retrieves a client report by ID.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_client_report<const SEED_SIZE: usize, A>(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        report_id: &ReportId,
    ) -> Result<Option<LeaderStoredReport<SEED_SIZE, A>>, Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16>,
        A::InputShare: PartialEq,
        A::PublicShare: PartialEq,
    {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_client_report()
SELECT
    client_timestamp, extensions, public_share, leader_input_share,
    helper_encrypted_input_share
FROM client_reports
WHERE client_reports.task_id = $1
  AND client_reports.report_id = $2
  AND client_reports.client_timestamp >= $3",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* report_id */ &report_id.as_ref(),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| Self::client_report_from_row(vdaf, *task_id, *report_id, row))
        .transpose()
    }

    #[cfg(feature = "test-util")]
    pub async fn get_report_metadatas_for_task(
        &self,
        task_id: &TaskId,
    ) -> Result<Vec<ReportMetadata>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_report_metadatas_for_task()
SELECT report_id, client_timestamp
FROM client_reports
WHERE client_reports.task_id = $1
  AND client_reports.client_timestamp >= $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            Ok(ReportMetadata::new(
                row.get_bytea_and_convert::<ReportId>("report_id")?,
                Time::from_naive_date_time(&row.get("client_timestamp")),
            ))
        })
        .collect()
    }

    #[cfg(feature = "test-util")]
    pub async fn get_client_reports_for_task<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
    ) -> Result<Vec<LeaderStoredReport<SEED_SIZE, A>>, Error>
    where
        A::InputShare: PartialEq,
        A::PublicShare: PartialEq,
    {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_client_reports_for_task()
SELECT
    report_id, client_timestamp, extensions, public_share, leader_input_share,
    helper_encrypted_input_share
FROM client_reports
WHERE client_reports.task_id = $1
  AND client_reports.client_timestamp >= $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            Self::client_report_from_row(
                vdaf,
                *task_id,
                row.get_bytea_and_convert::<ReportId>("report_id")?,
                row,
            )
        })
        .collect()
    }

    fn client_report_from_row<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>(
        vdaf: &A,
        task_id: TaskId,
        report_id: ReportId,
        row: Row,
    ) -> Result<LeaderStoredReport<SEED_SIZE, A>, Error>
    where
        A::InputShare: PartialEq,
        A::PublicShare: PartialEq,
    {
        let time = Time::from_naive_date_time(&row.get("client_timestamp"));

        let encoded_extensions: Vec<u8> = row
            .get::<_, Option<_>>("extensions")
            .ok_or_else(|| Error::Scrubbed)?;
        let extensions: Vec<Extension> =
            decode_u16_items(&(), &mut Cursor::new(&encoded_extensions))?;

        let encoded_public_share: Vec<u8> = row
            .get::<_, Option<_>>("public_share")
            .ok_or_else(|| Error::Scrubbed)?;
        let public_share = A::PublicShare::get_decoded_with_param(vdaf, &encoded_public_share)?;

        let encoded_leader_input_share: Vec<u8> = row
            .get::<_, Option<_>>("leader_input_share")
            .ok_or_else(|| Error::Scrubbed)?;
        let leader_input_share = A::InputShare::get_decoded_with_param(
            &(vdaf, Role::Leader.index().unwrap()),
            &encoded_leader_input_share,
        )?;

        let encoded_helper_input_share: Vec<u8> = row
            .get::<_, Option<_>>("helper_encrypted_input_share")
            .ok_or_else(|| Error::Scrubbed)?;
        let helper_encrypted_input_share =
            HpkeCiphertext::get_decoded(&encoded_helper_input_share)?;

        Ok(LeaderStoredReport::new(
            task_id,
            ReportMetadata::new(report_id, time),
            public_share,
            extensions,
            leader_input_share,
            helper_encrypted_input_share,
        ))
    }

    /// `get_unaggregated_client_reports_for_task` returns some unaggregated client reports for the
    /// task identified by the given task ID. Returned reports are marked as aggregation-started:
    /// the caller must either create an aggregation job with, or call `mark_report_unaggregated` on
    /// each returned report as part of the same transaction.
    ///
    /// This should only be used with VDAFs that have an aggregation parameter of the unit type. It
    /// relies on this assumption to find relevant reports without consulting collection jobs. For
    /// VDAFs that do have a different aggregation parameter,
    /// `get_unaggregated_client_report_ids_by_collect_for_task` should be used instead.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_unaggregated_client_reports_for_task(
        &self,
        task_id: &TaskId,
        limit: usize,
    ) -> Result<Vec<ReportMetadata>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- get_unaggregated_client_reports_for_task()
WITH unaggregated_reports AS (
    SELECT client_reports.id FROM client_reports
    WHERE client_reports.task_id = $1
      AND client_reports.aggregation_started = FALSE
      AND client_reports.client_timestamp >= $2
    ORDER BY client_timestamp DESC
    FOR UPDATE OF client_reports SKIP LOCKED
    LIMIT $5::BIGINT
)
UPDATE client_reports SET
    aggregation_started = TRUE, updated_at = $3, updated_by = $4
WHERE id IN (SELECT id FROM unaggregated_reports)
RETURNING report_id, client_timestamp",
            )
            .await?;
        let rows = self
            .query(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* limit */ &i64::try_from(limit)?,
                ],
            )
            .await?;

        rows.into_iter()
            .map(|row| {
                Ok(ReportMetadata::new(
                    row.get_bytea_and_convert::<ReportId>("report_id")?,
                    Time::from_naive_date_time(&row.get("client_timestamp")),
                ))
            })
            .collect::<Result<Vec<_>, Error>>()
    }

    /// `get_unaggregated_client_report_ids_by_collect_for_task` returns pairs of report IDs and
    /// aggregation parameters, corresponding to client reports that have not yet been aggregated,
    /// or not aggregated with a certain aggregation parameter, and for which there are collect
    /// jobs, for a given task. Returned client reports are marked as aggregation-started, but this
    /// will not stop additional aggregation jobs from being created later with different
    /// aggregation parameters.
    ///
    /// This should only be used with VDAFs with a non-unit type aggregation parameter. If a VDAF
    /// has the unit type as its aggregation parameter, then
    /// `get_unaggregated_client_report_ids_for_task` should be used instead. In such cases, it is
    /// not necessary to wait for a collection job to arrive before preparing reports.
    ///
    /// This function deliberately ignores the `client_reports.aggregation_started` column, which
    /// only has meaning for VDAFs without aggregation parameters.
    #[tracing::instrument(skip(self), err)]
    #[cfg(feature = "test-util")]
    pub async fn get_unaggregated_client_report_ids_by_collect_for_task<const SEED_SIZE: usize, A>(
        &self,
        task_id: &TaskId,
        limit: usize,
    ) -> Result<Vec<(A::AggregationParam, ReportMetadata)>, Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16> + VdafHasAggregationParameter,
    {
        // TODO(#224): lock retrieved client_reports rows
        // TODO(#225): use get_task_primary_key_and_expiry_threshold as in
        // get_unaggregated_client_reports_for_task
        let stmt = self
            .prepare_cached(
                "-- get_unaggregated_client_report_ids_by_collect_for_task()
WITH unaggregated_client_report_ids AS (
    SELECT DISTINCT report_id, client_timestamp, collection_jobs.aggregation_param
    FROM collection_jobs
    INNER JOIN client_reports
    ON collection_jobs.task_id = client_reports.task_id
    AND client_reports.client_timestamp <@ collection_jobs.batch_interval
    LEFT JOIN (
        SELECT report_aggregations.id, report_aggregations.client_report_id,
            aggregation_jobs.aggregation_param
        FROM report_aggregations
        INNER JOIN aggregation_jobs
        ON aggregation_jobs.id = report_aggregations.aggregation_job_id
        WHERE aggregation_jobs.task_id = (SELECT id FROM tasks WHERE task_id = $1)
    ) AS report_aggs
    ON report_aggs.client_report_id = client_reports.report_id
    -- this join is very inefficient, fix before deploying in non-test scenario
    AND report_aggs.aggregation_param = collection_jobs.aggregation_param
    WHERE client_reports.task_id = (SELECT id FROM tasks WHERE task_id = $1)
    AND collection_jobs.task_id = (SELECT id FROM tasks WHERE task_id = $1)
    AND collection_jobs.state = 'START'
    AND report_aggs.id IS NULL
    LIMIT $2::BIGINT
),
updated_client_reports AS (
    UPDATE client_reports SET aggregation_started = TRUE
    FROM unaggregated_client_report_ids
    WHERE client_reports.task_id = (SELECT id FROM tasks WHERE task_id = $1)
      AND client_reports.report_id = unaggregated_client_report_ids.report_id
      AND client_reports.client_timestamp =
        unaggregated_client_report_ids.client_timestamp
)
SELECT report_id, client_timestamp, aggregation_param
FROM unaggregated_client_report_ids",
            )
            .await?;
        let rows = self
            .query(
                &stmt,
                &[&task_id.as_ref(), /* limit */ &i64::try_from(limit)?],
            )
            .await?;

        rows.into_iter()
            .map(|row| {
                let report_metadata = ReportMetadata::new(
                    row.get_bytea_and_convert::<ReportId>("report_id")?,
                    Time::from_naive_date_time(&row.get("client_timestamp")),
                );
                let agg_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
                Ok((agg_param, report_metadata))
            })
            .collect::<Result<Vec<_>, Error>>()
    }

    /// `mark_report_unaggregated` resets the aggregation-started flag on the given client report,
    /// so that it may once again be returned by `get_unaggregated_client_report_ids_for_task`. It
    /// should generally only be called on report IDs returned from
    /// `get_unaggregated_client_report_ids_for_task`, as part of the same transaction, for any
    /// client reports that are not added to an aggregation job.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn mark_report_unaggregated(
        &self,
        task_id: &TaskId,
        report_id: &ReportId,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- mark_report_unaggregated()
UPDATE client_reports
SET aggregation_started = false, updated_at = $4, updated_by = $5
WHERE client_reports.task_id = $1
  AND client_reports.report_id = $2
  AND client_reports.client_timestamp >= $3",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* report_id */ &report_id.get_encoded()?,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                ],
            )
            .await?,
        )
    }

    #[cfg(feature = "test-util")]
    pub async fn mark_report_aggregated(
        &self,
        task_id: &TaskId,
        report_id: &ReportId,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- mark_report_aggregated()
UPDATE client_reports
SET aggregation_started = TRUE, updated_at = $4, updated_by = $5
WHERE client_reports.task_id = $1
  AND client_reports.report_id = $2
  AND client_reports.client_timestamp >= $3",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* report_id */ &report_id.get_encoded()?,
                /* threshold */ &task_info.report_expiry_threshold(&now)?,
                /* updated_at */ &now,
                /* updated_by */ &self.name,
            ],
        )
        .await?;
        Ok(())
    }

    /// Determines whether the given task includes any client reports which have not yet started the
    /// aggregation process in the given interval.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn interval_has_unaggregated_reports(
        &self,
        task_id: &TaskId,
        batch_interval: &Interval,
    ) -> Result<bool, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(false),
        };

        let stmt = self
            .prepare_cached(
                "-- interval_has_unaggregated_reports()
SELECT EXISTS(
    SELECT 1 FROM client_reports
    WHERE client_reports.task_id = $1
      AND client_reports.client_timestamp >= LOWER($2::TSRANGE)
      AND client_reports.client_timestamp < UPPER($2::TSRANGE)
      AND client_reports.client_timestamp >= $3
      AND client_reports.aggregation_started = FALSE
) AS unaggregated_report_exists",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_interval */ &SqlInterval::from(batch_interval),
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                ],
            )
            .await?;
        Ok(row.get("unaggregated_report_exists"))
    }

    /// Return the number of reports in the provided task whose timestamp falls within the provided
    /// interval, regardless of whether the reports have been aggregated or collected. Applies only
    /// to time-interval queries.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn count_client_reports_for_interval(
        &self,
        task_id: &TaskId,
        batch_interval: &Interval,
    ) -> Result<u64, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(0),
        };

        let stmt = self
            .prepare_cached(
                "-- count_client_reports_for_interval()
SELECT COUNT(1) AS count
FROM client_reports
WHERE client_reports.task_id = $1
  AND client_reports.client_timestamp >= LOWER($2::TSRANGE)
  AND client_reports.client_timestamp < UPPER($2::TSRANGE)
  AND client_reports.client_timestamp >= $3",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_interval */ &SqlInterval::from(batch_interval),
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                ],
            )
            .await?;
        Ok(row
            .get::<_, Option<i64>>("count")
            .unwrap_or_default()
            .try_into()?)
    }

    /// Return the number of reports in the provided task & batch, regardless of whether the reports
    /// have been aggregated or collected. Applies only to fixed-size queries.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn count_client_reports_for_batch_id(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<u64, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(0),
        };

        let stmt = self
            .prepare_cached(
                "-- count_client_reports_for_batch_id()
SELECT COUNT(DISTINCT report_aggregations.client_report_id) AS count
FROM report_aggregations
JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
JOIN tasks ON tasks.id = aggregation_jobs.task_id AND tasks.id = report_aggregations.task_id
WHERE report_aggregations.task_id = $1
  AND aggregation_jobs.batch_id = $2
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $3",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_id */ &batch_id.get_encoded()?,
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                ],
            )
            .await?;
        Ok(row
            .get::<_, Option<i64>>("count")
            .unwrap_or_default()
            .try_into()?)
    }

    /// `put_client_report` stores a client report, the associated plaintext leader input share and
    /// the associated encrypted helper share. Returns `Ok(())` if the write succeeds, or if there
    /// was already a row in the table matching `report`. Returns an error if something goes wrong
    /// or if the report ID is already in use with different values.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_client_report<const SEED_SIZE: usize, A>(
        &self,
        report: &LeaderStoredReport<SEED_SIZE, A>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16>,
        A::InputShare: PartialEq,
        A::PublicShare: PartialEq,
    {
        let task_info = match self.task_info_for(report.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let encoded_public_share = report.public_share().get_encoded()?;
        let encoded_leader_share = report.leader_input_share().get_encoded()?;
        let encoded_helper_share = report.helper_encrypted_input_share().get_encoded()?;
        let mut encoded_extensions = Vec::new();
        encode_u16_items(&mut encoded_extensions, &(), report.leader_extensions())?;

        let stmt = self
            .prepare_cached(
                "-- put_client_report()
INSERT INTO client_reports (
    task_id, report_id, client_timestamp, extensions, public_share,
    leader_input_share, helper_encrypted_input_share, created_at, updated_at,
    updated_by
)
VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
)
ON CONFLICT(task_id, report_id) DO UPDATE
    SET (
        client_timestamp, extensions, public_share, leader_input_share,
        helper_encrypted_input_share, created_at, updated_at, updated_by
    ) = (
        excluded.client_timestamp, excluded.extensions, excluded.public_share,
        excluded.leader_input_share, excluded.helper_encrypted_input_share,
        excluded.created_at, excluded.updated_at, excluded.updated_by
    )
    WHERE client_reports.client_timestamp < $11",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* report_id */ report.metadata().id().as_ref(),
                    /* client_timestamp */
                    &report.metadata().time().as_naive_date_time()?,
                    /* extensions */ &encoded_extensions,
                    /* public_share */ &encoded_public_share,
                    /* leader_input_share */ &encoded_leader_share,
                    /* helper_encrypted_input_share */ &encoded_helper_share,
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// scrub_client_report removes the client report itself from the datastore, retaining only a
    /// small amount of metadata required to perform duplicate-report detection & garbage
    /// collection.
    ///
    /// This method is intended for use by aggregators acting in the Leader role. Scrubbed reports
    /// can no longer be read, so this method should only be called once all aggregations over the
    /// report have stepped past their START state.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn scrub_client_report(
        &self,
        task_id: &TaskId,
        report_id: &ReportId,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- scrub_client_report()
UPDATE client_reports SET
    extensions = NULL,
    public_share = NULL,
    leader_input_share = NULL,
    helper_encrypted_input_share = NULL,
    updated_at = $1,
    updated_by = $2
WHERE task_id = $3
  AND report_id = $4
  AND client_timestamp >= $5",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_info.pkey,
                    /* report_id */ &report_id.as_ref(),
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    pub async fn verify_client_report_scrubbed(&self, task_id: &TaskId, report_id: &ReportId) {
        let task_info = match self.task_info_for(task_id).await.unwrap() {
            Some(task_info) => task_info,
            None => panic!("No such task"),
        };

        let row = self
            .query_one(
                "-- verify_client_report_scrubbed()
SELECT
    extensions, public_share, leader_input_share, helper_encrypted_input_share
FROM client_reports
WHERE task_id = $1
  AND report_id = $2
  AND client_timestamp >= $3",
                &[
                    /* task_id */ &task_info.pkey,
                    /* report_id */ report_id.as_ref(),
                    /* threshold */
                    &task_info
                        .report_expiry_threshold(&self.clock.now().as_naive_date_time().unwrap())
                        .unwrap(),
                ],
            )
            .await
            .unwrap();

        assert_eq!(row.get::<_, Option<Vec<u8>>>("extensions"), None);
        assert_eq!(row.get::<_, Option<Vec<u8>>>("public_share"), None);
        assert_eq!(row.get::<_, Option<Vec<u8>>>("leader_input_share"), None);
        assert_eq!(
            row.get::<_, Option<Vec<u8>>>("helper_encrypted_input_share"),
            None
        );
    }

    /// put_scrubbed_report stores a scrubbed report, given its associated task ID & report share.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_scrubbed_report(
        &self,
        task_id: &TaskId,
        report_share: &ReportShare,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        // On conflict, we update the row, but only if the incoming client timestamp (excluded)
        // matches the existing one. This lets us detect whether there's a row with a mismatching
        // timestamp through the number of rows modified by the statement.
        let stmt = self
            .prepare_cached(
                "-- put_scrubbed_report()
INSERT INTO client_reports (
    task_id, report_id, client_timestamp, created_at, updated_at, updated_by
)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT(task_id, report_id) DO UPDATE
SET (
    client_timestamp, extensions, public_share, leader_input_share,
    helper_encrypted_input_share, created_at, updated_at, updated_by
) = (
    excluded.client_timestamp, excluded.extensions, excluded.public_share,
    excluded.leader_input_share, excluded.helper_encrypted_input_share,
    excluded.created_at, excluded.updated_at, excluded.updated_by
)
WHERE client_reports.client_timestamp < $7",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* report_id */ &report_share.metadata().id().as_ref(),
                    /* client_timestamp */
                    &report_share.metadata().time().as_naive_date_time()?,
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// get_aggregation_job retrieves an aggregation job by ID.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregation_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<Option<AggregationJob<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_aggregation_job()
SELECT
    aggregation_param, batch_id, client_timestamp_interval, state, step,
    last_request_hash
FROM aggregation_jobs
WHERE aggregation_jobs.task_id = $1
  AND aggregation_jobs.aggregation_job_id = $2
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $3",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| Self::aggregation_job_from_row(task_id, aggregation_job_id, &row))
        .transpose()
    }

    /// get_aggregation_jobs_for_task returns all aggregation jobs for a given task ID.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregation_jobs_for_task<const SEED_SIZE: usize, Q, A>(
        &self,
        task_id: &TaskId,
    ) -> Result<Vec<AggregationJob<SEED_SIZE, Q, A>>, Error>
    where
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_aggregation_jobs_for_task()
SELECT
    aggregation_job_id, aggregation_param, batch_id, client_timestamp_interval,
    state, step, last_request_hash
FROM aggregation_jobs
WHERE aggregation_jobs.task_id = $1
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            Self::aggregation_job_from_row(
                task_id,
                &row.get_bytea_and_convert::<AggregationJobId>("aggregation_job_id")?,
                &row,
            )
        })
        .collect()
    }

    fn aggregation_job_from_row<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        row: &Row,
    ) -> Result<AggregationJob<SEED_SIZE, Q, A>, Error> {
        let mut job = AggregationJob::new(
            *task_id,
            *aggregation_job_id,
            A::AggregationParam::get_decoded(row.get("aggregation_param"))?,
            Q::PartialBatchIdentifier::get_decoded(row.get::<_, &[u8]>("batch_id"))?,
            row.get::<_, SqlInterval>("client_timestamp_interval")
                .as_interval(),
            row.get("state"),
            row.get_postgres_integer_and_convert::<i32, _, _>("step")?,
        );

        if let Some(hash) = row.get::<_, Option<Vec<u8>>>("last_request_hash") {
            job = job.with_last_request_hash(hash.try_into().map_err(|h| {
                Error::DbState(format!(
                    "last_request_hash value {h:?} cannot be converted to 32 byte array"
                ))
            })?);
        }

        Ok(job)
    }

    /// acquire_incomplete_aggregation_jobs retrieves & acquires the IDs of unclaimed incomplete
    /// aggregation jobs. At most `maximum_acquire_count` jobs are acquired. The job is acquired
    /// with a "lease" that will time out; the desired duration of the lease is a parameter, and the
    /// returned lease provides the absolute timestamp at which the lease is no longer live.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn acquire_incomplete_aggregation_jobs(
        &self,
        lease_duration: &StdDuration,
        maximum_acquire_count: usize,
    ) -> Result<Vec<Lease<AcquiredAggregationJob>>, Error> {
        let now = self.clock.now().as_naive_date_time()?;
        let lease_expiry_time = add_naive_date_time_duration(&now, lease_duration)?;
        let maximum_acquire_count: i64 = maximum_acquire_count.try_into()?;

        // TODO(#224): verify that this query is efficient. I am not sure if we would currently
        // scan over every (in-progress, not-leased) aggregation job for tasks where we are in the
        // HELPER role.
        // We generate the token on the DB to allow each acquired job to receive its own distinct
        // token. This is not strictly necessary as we only care about token collisions on a
        // per-row basis.
        let stmt = self
            .prepare_cached(
                "-- acquire_incomplete_aggregation_jobs()
WITH incomplete_jobs AS (
    SELECT aggregation_jobs.id FROM aggregation_jobs
    JOIN tasks ON tasks.id = aggregation_jobs.task_id
    WHERE tasks.aggregator_role = 'LEADER'
    AND aggregation_jobs.state = 'IN_PROGRESS'
    AND aggregation_jobs.lease_expiry <= $2
    AND UPPER(aggregation_jobs.client_timestamp_interval) >=
        COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL,
                 '-infinity'::TIMESTAMP)
    FOR UPDATE OF aggregation_jobs SKIP LOCKED LIMIT $3
)
UPDATE aggregation_jobs SET
    lease_expiry = $1,
    lease_token = gen_random_bytes(16),
    lease_attempts = lease_attempts + 1,
    updated_at = $4,
    updated_by = $5
FROM tasks
WHERE tasks.id = aggregation_jobs.task_id
AND aggregation_jobs.id IN (SELECT id FROM incomplete_jobs)
RETURNING tasks.task_id, tasks.query_type, tasks.vdaf,
          aggregation_jobs.aggregation_job_id, aggregation_jobs.lease_token,
          aggregation_jobs.lease_attempts",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* lease_expiry */ &lease_expiry_time,
                /* now */ &now,
                /* limit */ &maximum_acquire_count,
                /* updated_at */ &now,
                /* updated_by */ &self.name,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            let aggregation_job_id =
                row.get_bytea_and_convert::<AggregationJobId>("aggregation_job_id")?;
            let query_type = row.try_get::<_, Json<task::QueryType>>("query_type")?.0;
            let vdaf = row.try_get::<_, Json<VdafInstance>>("vdaf")?.0;
            let lease_token = row.get_bytea_and_convert::<LeaseToken>("lease_token")?;
            let lease_attempts = row.get_bigint_and_convert("lease_attempts")?;
            Ok(Lease::new(
                AcquiredAggregationJob::new(task_id, aggregation_job_id, query_type, vdaf),
                lease_expiry_time,
                lease_token,
                lease_attempts,
            ))
        })
        .collect()
    }

    /// release_aggregation_job releases an acquired (via e.g. acquire_incomplete_aggregation_jobs)
    /// aggregation job. It returns an error if the aggregation job has no current lease.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn release_aggregation_job(
        &self,
        lease: &Lease<AcquiredAggregationJob>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(lease.leased().task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- release_aggregation_job()
UPDATE aggregation_jobs
SET lease_expiry = '-infinity'::TIMESTAMP,
    lease_token = NULL,
    lease_attempts = 0,
    updated_at = $1,
    updated_by = $2
WHERE aggregation_jobs.task_id = $3
  AND aggregation_jobs.aggregation_job_id = $4
  AND aggregation_jobs.lease_expiry = $5
  AND aggregation_jobs.lease_token = $6
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $7",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_info.pkey,
                    /* aggregation_job_id */ &lease.leased().aggregation_job_id().as_ref(),
                    /* lease_expiry */ &lease.lease_expiry_time(),
                    /* lease_token */ &lease.lease_token().as_ref(),
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// put_aggregation_job stores an aggregation job.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_aggregation_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        aggregation_job: &AggregationJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(aggregation_job.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- put_aggregation_job()
INSERT INTO aggregation_jobs
    (task_id, aggregation_job_id, aggregation_param, batch_id,
    client_timestamp_interval, state, step, last_request_hash,
    created_at, updated_at, updated_by)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT(task_id, aggregation_job_id) DO UPDATE
    SET (
        aggregation_param, batch_id, client_timestamp_interval, state, step,
        last_request_hash, created_at, updated_at, updated_by
    ) = (
        excluded.aggregation_param, excluded.batch_id,
        excluded.client_timestamp_interval, excluded.state, excluded.step,
        excluded.last_request_hash, excluded.created_at, excluded.updated_at,
        excluded.updated_by
    )
    WHERE UPPER(aggregation_jobs.client_timestamp_interval) < $12",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* aggregation_job_id */ &aggregation_job.id().as_ref(),
                    /* aggregation_param */
                    &aggregation_job.aggregation_parameter().get_encoded()?,
                    /* batch_id */
                    &aggregation_job.partial_batch_identifier().get_encoded()?,
                    /* client_timestamp_interval */
                    &SqlInterval::from(aggregation_job.client_timestamp_interval()),
                    /* state */ &aggregation_job.state(),
                    /* step */ &(u16::from(aggregation_job.step()) as i32),
                    /* last_request_hash */ &aggregation_job.last_request_hash(),
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// update_aggregation_job updates a stored aggregation job.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn update_aggregation_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        aggregation_job: &AggregationJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(aggregation_job.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- update_aggregation_job()
UPDATE aggregation_jobs SET
    state = $1,
    step = $2,
    last_request_hash = $3,
    updated_at = $4,
    updated_by = $5
WHERE aggregation_jobs.task_id = $6
  AND aggregation_jobs.aggregation_job_id = $7
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $8::TIMESTAMP",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */ &aggregation_job.state(),
                    /* step */ &(u16::from(aggregation_job.step()) as i32),
                    /* last_request_hash */ &aggregation_job.last_request_hash(),
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_info.pkey,
                    /* aggregation_job_id */ &aggregation_job.id().as_ref(),
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// get_report_aggregations_for_aggregation_job retrieves all report aggregations associated
    /// with a given aggregation job, ordered by their natural ordering.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_report_aggregations_for_aggregation_job<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        role: &Role,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<Vec<ReportAggregation<SEED_SIZE, A>>, Error>
    where
        for<'a> A::PrepareState: ParameterizedDecode<(&'a A, usize)>,
    {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_report_aggregations_for_aggregation_job()
SELECT
    ord, client_report_id, client_timestamp, last_prep_resp,
    report_aggregations.state, public_share, leader_extensions, leader_input_share,
    helper_encrypted_input_share, leader_prep_transition, helper_prep_state,
    error_code
FROM report_aggregations
JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
WHERE report_aggregations.task_id = $1
  AND aggregation_jobs.task_id = $1
  AND aggregation_jobs.aggregation_job_id = $2
  AND UPPER(client_timestamp_interval) >= $3
ORDER BY ord ASC",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            Self::report_aggregation_from_row(
                vdaf,
                role,
                task_id,
                aggregation_job_id,
                &row.get_bytea_and_convert::<ReportId>("client_report_id")?,
                &row,
            )
        })
        .collect()
    }

    /// get_report_aggregation_by_report_id gets a report aggregation by report ID.
    #[cfg(feature = "test-util")]
    pub async fn get_report_aggregation_by_report_id<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        role: &Role,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        report_id: &ReportId,
    ) -> Result<Option<ReportAggregation<SEED_SIZE, A>>, Error>
    where
        for<'a> A::PrepareState: ParameterizedDecode<(&'a A, usize)>,
    {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_report_aggregation_by_report_id()
SELECT
    ord, client_timestamp, last_prep_resp, report_aggregations.state,
    public_share, leader_extensions, leader_input_share,
    helper_encrypted_input_share, leader_prep_transition, helper_prep_state,
    error_code
FROM report_aggregations
JOIN aggregation_jobs
    ON aggregation_jobs.id = report_aggregations.aggregation_job_id
WHERE report_aggregations.task_id = $1
    AND aggregation_jobs.task_id = $1
    AND aggregation_jobs.aggregation_job_id = $2
    AND report_aggregations.client_report_id = $3
    AND UPPER(client_timestamp_interval) >= $4",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                /* report_id */ &report_id.as_ref(),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| {
            Self::report_aggregation_from_row(
                vdaf,
                role,
                task_id,
                aggregation_job_id,
                report_id,
                &row,
            )
        })
        .transpose()
    }

    /// get_report_aggregations_for_task retrieves all report aggregations associated with a given
    /// task.
    #[cfg(feature = "test-util")]
    pub async fn get_report_aggregations_for_task<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        role: &Role,
        task_id: &TaskId,
    ) -> Result<Vec<ReportAggregation<SEED_SIZE, A>>, Error>
    where
        for<'a> A::PrepareState: ParameterizedDecode<(&'a A, usize)>,
    {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_report_aggregations_for_task()
SELECT
    aggregation_jobs.aggregation_job_id, ord, client_report_id, client_timestamp,
    last_prep_resp, report_aggregations.state, public_share, leader_extensions,
    leader_input_share, helper_encrypted_input_share, leader_prep_transition,
    helper_prep_state, error_code
FROM report_aggregations
JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
WHERE report_aggregations.task_id = $1
  AND aggregation_jobs.task_id = $1
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            Self::report_aggregation_from_row(
                vdaf,
                role,
                task_id,
                &row.get_bytea_and_convert::<AggregationJobId>("aggregation_job_id")?,
                &row.get_bytea_and_convert::<ReportId>("client_report_id")?,
                &row,
            )
        })
        .collect()
    }

    fn report_aggregation_from_row<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>(
        vdaf: &A,
        role: &Role,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
        report_id: &ReportId,
        row: &Row,
    ) -> Result<ReportAggregation<SEED_SIZE, A>, Error>
    where
        for<'a> A::PrepareState: ParameterizedDecode<(&'a A, usize)>,
    {
        let ord: u64 = row.get_bigint_and_convert("ord")?;
        let time = Time::from_naive_date_time(&row.get("client_timestamp"));
        let state: ReportAggregationStateCode = row.get("state");
        let error_code: Option<i16> = row.get("error_code");
        let last_prep_resp_bytes: Option<Vec<u8>> = row.get("last_prep_resp");

        let last_prep_resp = last_prep_resp_bytes
            .map(|bytes| PrepareResp::get_decoded(&bytes))
            .transpose()?;

        let agg_state = match state {
            ReportAggregationStateCode::Start => {
                let public_share_bytes =
                    row.get::<_, Option<Vec<u8>>>("public_share")
                        .ok_or_else(|| {
                            Error::DbState(
                                "report aggregation in state START but public_share is NULL"
                                    .to_string(),
                            )
                        })?;
                let leader_extensions_bytes = row
                    .get::<_, Option<Vec<u8>>>("leader_extensions")
                    .ok_or_else(|| {
                        Error::DbState(
                            "report aggregation in state START but leader_extensions is NULL"
                                .to_string(),
                        )
                    })?;
                let leader_input_share_bytes = row
                    .get::<_, Option<Vec<u8>>>("leader_input_share")
                    .ok_or_else(|| {
                        Error::DbState(
                            "report aggregation in state START but leader_input_share is NULL"
                                .to_string(),
                        )
                    })?;
                let helper_encrypted_input_share_bytes =
                    row.get::<_, Option<Vec<u8>>>("helper_encrypted_input_share")
                        .ok_or_else(|| {
                            Error::DbState(
                            "report aggregation in state START but helper_encrypted_input_share is NULL"
                                .to_string(),
                        )
                        })?;

                let public_share =
                    A::PublicShare::get_decoded_with_param(vdaf, &public_share_bytes)?;
                let leader_extensions =
                    decode_u16_items(&(), &mut Cursor::new(&leader_extensions_bytes))?;
                let leader_input_share = A::InputShare::get_decoded_with_param(
                    &(vdaf, Role::Leader.index().unwrap()),
                    &leader_input_share_bytes,
                )?;
                let helper_encrypted_input_share =
                    HpkeCiphertext::get_decoded(&helper_encrypted_input_share_bytes)?;

                ReportAggregationState::StartLeader {
                    public_share,
                    leader_extensions,
                    leader_input_share,
                    helper_encrypted_input_share,
                }
            }

            ReportAggregationStateCode::Waiting => {
                match role {
                    Role::Leader => {
                        let leader_prep_transition_bytes = row
                            .get::<_, Option<Vec<u8>>>("leader_prep_transition")
                            .ok_or_else(|| {
                                Error::DbState(
                                    "report aggregation in state WAITING but leader_prep_transition is NULL"
                                        .to_string(),
                                )
                            })?;
                        let ping_pong_transition = PingPongTransition::get_decoded_with_param(
                            &(vdaf, 0 /* leader */),
                            &leader_prep_transition_bytes,
                        )?;

                        ReportAggregationState::WaitingLeader {
                            transition: ping_pong_transition,
                        }
                    }
                    Role::Helper => {
                        let helper_prep_state_bytes = row
                            .get::<_, Option<Vec<u8>>>("helper_prep_state")
                            .ok_or_else(|| {
                                Error::DbState(
                                    "report aggregation in state WAITING but helper_prep_state is NULL"
                                        .to_string(),
                                )
                            })?;
                        let prepare_state = A::PrepareState::get_decoded_with_param(
                            &(vdaf, 1 /* helper */),
                            &helper_prep_state_bytes,
                        )?;

                        ReportAggregationState::WaitingHelper { prepare_state }
                    }
                    _ => panic!("unexpected role"),
                }
            }

            ReportAggregationStateCode::Finished => ReportAggregationState::Finished,

            ReportAggregationStateCode::Failed => {
                let prepare_error = match error_code {
                    Some(c) => {
                        let c: u8 = c.try_into().map_err(|err| {
                            Error::DbState(format!("couldn't convert error_code value: {err}"))
                        })?;
                        Some(c.try_into().map_err(|err| {
                            Error::DbState(format!("couldn't convert error_code value: {err}"))
                        })?)
                    }
                    None => None,
                }
                .ok_or_else(|| {
                    Error::DbState(
                        "report aggregation in state FAILED but error_code is NULL".to_string(),
                    )
                })?;

                ReportAggregationState::Failed { prepare_error }
            }
        };

        Ok(ReportAggregation::new(
            *task_id,
            *aggregation_job_id,
            *report_id,
            time,
            ord,
            last_prep_resp,
            agg_state,
        ))
    }

    /// put_report_aggregation stores aggregation data for a single report.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_report_aggregation<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        report_aggregation: &ReportAggregation<SEED_SIZE, A>,
    ) -> Result<(), Error>
    where
        A::PrepareState: Encode,
    {
        let task_info = match self.task_info_for(report_aggregation.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let encoded_state_values = report_aggregation.state().encoded_values_from_state()?;
        let encoded_last_prep_resp: Option<Vec<u8>> = report_aggregation
            .last_prep_resp()
            .map(PrepareResp::get_encoded)
            .transpose()?;

        let stmt = self
            .prepare_cached(
                "-- put_report_aggregation()
INSERT INTO report_aggregations
    (task_id, aggregation_job_id, ord, client_report_id, client_timestamp,
    last_prep_resp, state, public_share, leader_extensions, leader_input_share,
    helper_encrypted_input_share, leader_prep_transition, helper_prep_state,
    error_code, created_at, updated_at, updated_by)
SELECT
    $1, aggregation_jobs.id, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
    $15, $16, $17
FROM aggregation_jobs
WHERE task_id = $1
  AND aggregation_job_id = $2
ON CONFLICT(task_id, aggregation_job_id, ord) DO UPDATE
    SET (
        client_report_id, client_timestamp, last_prep_resp, state, public_share,
        leader_extensions, leader_input_share, helper_encrypted_input_share,
        leader_prep_transition, helper_prep_state, error_code, created_at,
        updated_at, updated_by
    ) = (
        excluded.client_report_id, excluded.client_timestamp,
        excluded.last_prep_resp, excluded.state, excluded.public_share,
        excluded.leader_extensions, excluded.leader_input_share,
        excluded.helper_encrypted_input_share, excluded.leader_prep_transition,
        excluded.helper_prep_state, excluded.error_code, excluded.created_at,
        excluded.updated_at, excluded.updated_by
    )
    WHERE (SELECT UPPER(client_timestamp_interval)
           FROM aggregation_jobs
           WHERE id = report_aggregations.aggregation_job_id) < $18",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* aggregation_job_id */
                    &report_aggregation.aggregation_job_id().as_ref(),
                    /* ord */ &TryInto::<i64>::try_into(report_aggregation.ord())?,
                    /* client_report_id */ &report_aggregation.report_id().as_ref(),
                    /* client_timestamp */ &report_aggregation.time().as_naive_date_time()?,
                    /* last_prep_resp */ &encoded_last_prep_resp,
                    /* state */ &report_aggregation.state().state_code(),
                    /* public_share */ &encoded_state_values.public_share,
                    /* leader_extensions */ &encoded_state_values.leader_extensions,
                    /* leader_input_share */ &encoded_state_values.leader_input_share,
                    /* helper_encrypted_input_share */
                    &encoded_state_values.helper_encrypted_input_share,
                    /* leader_prep_transition */
                    &encoded_state_values.leader_prep_transition,
                    /* helper_prep_state */ &encoded_state_values.helper_prep_state,
                    /* error_code */ &encoded_state_values.prepare_error,
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Creates a report aggregation in the `StartLeader` state from its metadata.
    ///
    /// Report shares are copied directly from the `client_reports` table.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_leader_report_aggregation(
        &self,
        report_aggregation_metadata: &ReportAggregationMetadata,
    ) -> Result<(), Error> {
        let task_info = match self
            .task_info_for(report_aggregation_metadata.task_id())
            .await?
        {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        match report_aggregation_metadata.state() {
            ReportAggregationMetadataState::Start => {
                let stmt = self
                    .prepare_cached(
                        "-- put_leader_report_aggregation()
INSERT INTO report_aggregations
    (task_id, aggregation_job_id, ord, client_report_id, client_timestamp,
    state, public_share, leader_extensions, leader_input_share,
    helper_encrypted_input_share, created_at, updated_at, updated_by)
SELECT
    $1, aggregation_jobs.id, $3, $4, $5, 'START'::REPORT_AGGREGATION_STATE,
    client_reports.public_share, client_reports.extensions,
    client_reports.leader_input_share,
    client_reports.helper_encrypted_input_share, $6, $7, $8
FROM aggregation_jobs
JOIN client_reports
    ON aggregation_jobs.task_id = client_reports.task_id
AND client_reports.report_id = $4
WHERE aggregation_jobs.task_id = $1
AND aggregation_job_id = $2
ON CONFLICT(task_id, aggregation_job_id, ord) DO UPDATE
    SET (
        client_report_id, client_timestamp, last_prep_resp, state, public_share,
        leader_extensions, leader_input_share, helper_encrypted_input_share,
        leader_prep_transition, helper_prep_state, error_code, created_at,
        updated_at, updated_by
    ) = (
        excluded.client_report_id, excluded.client_timestamp,
        excluded.last_prep_resp, excluded.state, excluded.public_share,
        excluded.leader_extensions, excluded.leader_input_share,
        excluded.helper_encrypted_input_share, excluded.leader_prep_transition,
        excluded.helper_prep_state, excluded.error_code, excluded.created_at,
        excluded.updated_at, excluded.updated_by
    )
    WHERE (SELECT UPPER(client_timestamp_interval)
        FROM aggregation_jobs
        WHERE id = report_aggregations.aggregation_job_id) < $9",
                    )
                    .await?;
                check_insert(
                    self.execute(
                        &stmt,
                        &[
                            /* task_id */ &task_info.pkey,
                            /* aggregation_job_id */
                            &report_aggregation_metadata.aggregation_job_id().as_ref(),
                            /* ord */
                            &TryInto::<i64>::try_into(report_aggregation_metadata.ord())?,
                            /* client_report_id */
                            &report_aggregation_metadata.report_id().as_ref(),
                            /* client_timestamp */
                            &report_aggregation_metadata.time().as_naive_date_time()?,
                            /* created_at */ &now,
                            /* updated_at */ &now,
                            /* updated_by */ &self.name,
                            /* threshold */ &task_info.report_expiry_threshold(&now)?,
                        ],
                    )
                    .await?,
                )
            }
            ReportAggregationMetadataState::Failed { prepare_error } => {
                let stmt = self
                    .prepare_cached(
                        "-- put_leader_report_aggregation()
INSERT INTO report_aggregations
    (task_id, aggregation_job_id, ord, client_report_id, client_timestamp,
    state, error_code, created_at, updated_at, updated_by)
SELECT
    $1, aggregation_jobs.id, $3, $4, $5, 'FAILED'::REPORT_AGGREGATION_STATE,
    $6, $7, $8, $9
FROM aggregation_jobs
JOIN client_reports
    ON client_reports.task_id = aggregation_jobs.task_id
   AND client_reports.report_id = $4
WHERE aggregation_jobs.task_id = $1
AND aggregation_job_id = $2
ON CONFLICT(task_id, aggregation_job_id, ord) DO UPDATE
    SET (
        client_report_id, client_timestamp, last_prep_resp, state, public_share,
        leader_extensions, leader_input_share, helper_encrypted_input_share,
        leader_prep_transition, helper_prep_state, error_code, created_at,
        updated_at, updated_by
    ) = (
        excluded.client_report_id, excluded.client_timestamp,
        excluded.last_prep_resp, excluded.state, excluded.public_share,
        excluded.leader_extensions, excluded.leader_input_share,
        excluded.helper_encrypted_input_share, excluded.leader_prep_transition,
        excluded.helper_prep_state, excluded.error_code, excluded.created_at,
        excluded.updated_at, excluded.updated_by
    )
    WHERE (SELECT UPPER(client_timestamp_interval)
           FROM aggregation_jobs
           WHERE id = report_aggregations.aggregation_job_id) >= $10",
                    )
                    .await?;
                check_insert(
                    self.execute(
                        &stmt,
                        &[
                            /* task_id */ &task_info.pkey,
                            /* aggregation_job_id */
                            &report_aggregation_metadata.aggregation_job_id().as_ref(),
                            /* ord */
                            &TryInto::<i64>::try_into(report_aggregation_metadata.ord())?,
                            /* client_report_id */
                            &report_aggregation_metadata.report_id().as_ref(),
                            /* client_timestamp */
                            &report_aggregation_metadata.time().as_naive_date_time()?,
                            /* error_code */ &(*prepare_error as i16),
                            /* created_at */ &now,
                            /* updated_at */ &now,
                            /* updated_by */ &self.name,
                            /* threshold */ &task_info.report_expiry_threshold(&now)?,
                        ],
                    )
                    .await?,
                )
            }
        }
    }

    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn update_report_aggregation<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        report_aggregation: &ReportAggregation<SEED_SIZE, A>,
    ) -> Result<(), Error>
    where
        A::PrepareState: Encode,
    {
        let task_info = match self.task_info_for(report_aggregation.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let encoded_state_values = report_aggregation.state().encoded_values_from_state()?;
        let encoded_last_prep_resp: Option<Vec<u8>> = report_aggregation
            .last_prep_resp()
            .map(PrepareResp::get_encoded)
            .transpose()?;

        let stmt = self
            .prepare_cached(
                "-- update_report_aggregation()
UPDATE report_aggregations
SET
    last_prep_resp = $1, state = $2, public_share = $3, leader_extensions = $4,
    leader_input_share = $5, helper_encrypted_input_share = $6,
    leader_prep_transition = $7, helper_prep_state = $8, error_code = $9,
    updated_at = $10, updated_by = $11
FROM aggregation_jobs
WHERE report_aggregations.aggregation_job_id = aggregation_jobs.id
  AND aggregation_jobs.aggregation_job_id = $12
  AND aggregation_jobs.task_id = $13
  AND report_aggregations.task_id = $13
  AND report_aggregations.client_report_id = $14
  AND report_aggregations.client_timestamp = $15
  AND report_aggregations.ord = $16
  AND UPPER(aggregation_jobs.client_timestamp_interval) >= $17",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* last_prep_resp */ &encoded_last_prep_resp,
                    /* state */ &report_aggregation.state().state_code(),
                    /* public_share */ &encoded_state_values.public_share,
                    /* leader_extensions */ &encoded_state_values.leader_extensions,
                    /* leader_input_share */ &encoded_state_values.leader_input_share,
                    /* helper_encrypted_input_share */
                    &encoded_state_values.helper_encrypted_input_share,
                    /* leader_prep_transition */
                    &encoded_state_values.leader_prep_transition,
                    /* helper_prep_state */ &encoded_state_values.helper_prep_state,
                    /* error_code */ &encoded_state_values.prepare_error,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* aggregation_job_id */
                    &report_aggregation.aggregation_job_id().as_ref(),
                    /* task_id */ &task_info.pkey,
                    /* client_report_id */ &report_aggregation.report_id().as_ref(),
                    /* client_timestamp */ &report_aggregation.time().as_naive_date_time()?,
                    /* ord */ &TryInto::<i64>::try_into(report_aggregation.ord())?,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Returns the collection job for the provided ID, or `None` if no such collection job exists.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_collection_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        collection_job_id: &CollectionJobId,
    ) -> Result<Option<CollectionJob<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_collection_job()
SELECT
    query, aggregation_param, batch_identifier, state, report_count,
    client_timestamp_interval, helper_aggregate_share, leader_aggregate_share
FROM collection_jobs
WHERE collection_jobs.task_id = $1
  AND collection_jobs.collection_job_id = $2
  AND COALESCE(
          LOWER(collection_jobs.batch_interval),
          (SELECT MAX(UPPER(client_timestamp_interval))
           FROM batch_aggregations
           WHERE batch_aggregations.task_id = collection_jobs.task_id
             AND batch_aggregations.batch_identifier = collection_jobs.batch_identifier
             AND batch_aggregations.aggregation_param = collection_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $3",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* collection_job_id */ &collection_job_id.as_ref(),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| {
            let batch_identifier = Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
            Self::collection_job_from_row(
                vdaf,
                *task_id,
                batch_identifier,
                *collection_job_id,
                &row,
            )
        })
        .transpose()
    }

    /// Returns a collection job in state FINISHED with the given parameters, or `None` if no such
    /// collection job exists.
    pub async fn get_finished_collection_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
        aggregation_param: &A::AggregationParam,
    ) -> Result<Option<CollectionJob<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_finished_collection_job()
SELECT
    collection_job_id, query, aggregation_param, state, report_count,
    client_timestamp_interval, helper_aggregate_share, leader_aggregate_share
FROM collection_jobs
WHERE collection_jobs.task_id = $1
  AND collection_jobs.batch_identifier = $2
  AND collection_jobs.aggregation_param = $3
  AND collection_jobs.state = 'FINISHED'
  AND COALESCE(
          LOWER(collection_jobs.batch_interval),
          (SELECT MAX(UPPER(client_timestamp_interval))
           FROM batch_aggregations
           WHERE batch_aggregations.task_id = collection_jobs.task_id
             AND batch_aggregations.batch_identifier = collection_jobs.batch_identifier
             AND batch_aggregations.aggregation_param = collection_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $4
LIMIT 1",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_identifier */ &batch_identifier.get_encoded()?,
                /* aggregation_param */ &aggregation_param.get_encoded()?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| {
            let collection_job_id =
                row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
            Self::collection_job_from_row(
                vdaf,
                *task_id,
                batch_identifier.clone(),
                collection_job_id,
                &row,
            )
        })
        .transpose()
    }

    /// Returns all collection jobs for the given task which include the given timestamp. Applies
    /// only to time-interval tasks.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_collection_jobs_including_time<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        timestamp: &Time,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, TimeInterval, A>>, Error> {
        // TODO(#1553): write unit test
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_collection_jobs_including_time()
SELECT
    collection_job_id, query, aggregation_param, batch_identifier, state,
    report_count, client_timestamp_interval, helper_aggregate_share,
    leader_aggregate_share
FROM collection_jobs
WHERE task_id = $1
  AND batch_interval @> $2::TIMESTAMP
  AND LOWER(batch_interval) >= $3",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* timestamp */ &timestamp.as_naive_date_time()?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let batch_identifier = Interval::get_decoded(row.get("batch_identifier"))?;
            let collection_job_id =
                row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
            Self::collection_job_from_row(vdaf, *task_id, batch_identifier, collection_job_id, &row)
        })
        .collect()
    }

    /// Returns all collection jobs for the given task whose collect intervals intersect with the
    /// given interval. Applies only to time-interval tasks.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_collection_jobs_intersecting_interval<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_interval: &Interval,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, TimeInterval, A>>, Error> {
        // TODO(#1553): write unit test
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_collection_jobs_intersecting_interval()
SELECT
    collection_job_id, query, aggregation_param, batch_identifier, state,
    report_count, client_timestamp_interval, helper_aggregate_share,
    leader_aggregate_share
FROM collection_jobs
WHERE task_id = $1
  AND batch_interval && $2
  AND LOWER(collection_jobs.batch_interval) >= $3",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_interval */ &SqlInterval::from(batch_interval),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let batch_identifier = Interval::get_decoded(row.get("batch_identifier"))?;
            let collection_job_id =
                row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
            Self::collection_job_from_row::<SEED_SIZE, TimeInterval, A>(
                vdaf,
                *task_id,
                batch_identifier,
                collection_job_id,
                &row,
            )
        })
        .collect()
    }

    /// Retrieves all collection jobs for the given batch ID. Multiple collection jobs may be
    /// returned with distinct aggregation parameters. Applies only to fixed-size tasks.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_collection_jobs_by_batch_id<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, FixedSize, A>>, Error> {
        // TODO(#1553): write unit test
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_collection_jobs_by_batch_id()
SELECT
    collection_job_id, query, aggregation_param, state, report_count,
    client_timestamp_interval, helper_aggregate_share, leader_aggregate_share
FROM collection_jobs
WHERE task_id = $1
  AND batch_identifier = $2
  AND COALESCE(
          (SELECT MAX(UPPER(client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = collection_jobs.task_id
             AND ba.batch_identifier = collection_jobs.batch_identifier
             AND ba.aggregation_param = collection_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $3",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_id */ &batch_id.get_encoded()?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let collection_job_id =
                row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
            Self::collection_job_from_row(vdaf, *task_id, *batch_id, collection_job_id, &row)
        })
        .collect()
    }

    #[cfg(feature = "test-util")]
    pub async fn get_collection_jobs_for_task<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_collection_jobs_for_task()
SELECT
    collection_job_id, query, aggregation_param, batch_identifier, state,
    report_count, client_timestamp_interval, helper_aggregate_share,
    leader_aggregate_share
FROM collection_jobs
WHERE task_id = $1
  AND COALESCE(
          LOWER(batch_interval),
          (SELECT MAX(UPPER(ba.client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = collection_jobs.task_id
             AND ba.batch_identifier = collection_jobs.batch_identifier
             AND ba.aggregation_param = collection_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let collection_job_id =
                row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
            let batch_identifier = Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
            Self::collection_job_from_row(vdaf, *task_id, batch_identifier, collection_job_id, &row)
        })
        .collect()
    }

    fn collection_job_from_row<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        vdaf: &A,
        task_id: TaskId,
        batch_identifier: Q::BatchIdentifier,
        collection_job_id: CollectionJobId,
        row: &Row,
    ) -> Result<CollectionJob<SEED_SIZE, Q, A>, Error> {
        let query = Query::<Q>::get_decoded(row.get("query"))?;
        let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
        let state: CollectionJobStateCode = row.get("state");
        let report_count: Option<i64> = row.get("report_count");
        let client_timestamp_interval: Option<SqlInterval> = row.get("client_timestamp_interval");
        let helper_aggregate_share_bytes: Option<Vec<u8>> = row.get("helper_aggregate_share");
        let leader_aggregate_share_bytes: Option<Vec<u8>> = row.get("leader_aggregate_share");

        let state = match state {
            CollectionJobStateCode::Start => CollectionJobState::Start,

            CollectionJobStateCode::Finished => {
                let report_count = u64::try_from(report_count.ok_or_else(|| {
                    Error::DbState(
                        "collection job in state FINISHED but report_count is NULL".to_string(),
                    )
                })?)?;
                let client_timestamp_interval = client_timestamp_interval
                    .ok_or_else(|| Error::DbState(
                        "collection job in state FINISHED but client_timestamp_interval is NULL".to_string())
                    )?.as_interval();
                let encrypted_helper_aggregate_share = HpkeCiphertext::get_decoded(
                    &helper_aggregate_share_bytes.ok_or_else(|| {
                        Error::DbState(
                            "collection job in state FINISHED but helper_aggregate_share is NULL"
                                .to_string(),
                        )
                    })?,
                )?;
                let leader_aggregate_share = A::AggregateShare::get_decoded_with_param(
                    &(vdaf, &aggregation_param),
                    &leader_aggregate_share_bytes.ok_or_else(|| {
                        Error::DbState(
                            "collection job is in state FINISHED but leader_aggregate_share is \
                             NULL"
                                .to_string(),
                        )
                    })?,
                )?;
                CollectionJobState::Finished {
                    report_count,
                    client_timestamp_interval,
                    encrypted_helper_aggregate_share,
                    leader_aggregate_share,
                }
            }

            CollectionJobStateCode::Abandoned => CollectionJobState::Abandoned,

            CollectionJobStateCode::Deleted => CollectionJobState::Deleted,
        };

        Ok(CollectionJob::new(
            task_id,
            collection_job_id,
            query,
            aggregation_param,
            batch_identifier,
            state,
        ))
    }

    /// Stores a new collection job.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_collection_job<
        const SEED_SIZE: usize,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        collection_job: &CollectionJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Debug,
    {
        let task_info = match self.task_info_for(collection_job.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let batch_interval =
            Q::to_batch_interval(collection_job.batch_identifier()).map(SqlInterval::from);

        let stmt = self
            .prepare_cached(
                "-- put_collection_job()
INSERT INTO collection_jobs
    (task_id, collection_job_id, query, aggregation_param, batch_identifier,
    batch_interval, state, created_at, updated_at, updated_by)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT(task_id, collection_job_id) DO UPDATE
    SET (
        query, aggregation_param, batch_identifier, batch_interval, state,
        created_at, updated_at, updated_by
    ) = (
        excluded.query, excluded.aggregation_param, excluded.batch_identifier,
        excluded.batch_interval, excluded.state, excluded.created_at,
        excluded.updated_at, excluded.updated_by
    )
    WHERE COALESCE(
              LOWER(collection_jobs.batch_interval),
              (SELECT MAX(UPPER(ba.client_timestamp_interval))
               FROM batch_aggregations ba
               WHERE ba.task_id = collection_jobs.task_id
                 AND ba.batch_identifier = collection_jobs.batch_identifier
                 AND ba.aggregation_param = collection_jobs.aggregation_param),
              '-infinity'::TIMESTAMP) < $11",
            )
            .await?;

        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* collection_job_id */ collection_job.id().as_ref(),
                    /* query */ &collection_job.query().get_encoded()?,
                    /* aggregation_param */
                    &collection_job.aggregation_parameter().get_encoded()?,
                    /* batch_identifier */ &collection_job.batch_identifier().get_encoded()?,
                    /* batch_interval */ &batch_interval,
                    /* state */ &collection_job.state().collection_job_state_code(),
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// acquire_incomplete_collection_jobs retrieves & acquires the IDs of unclaimed incomplete
    /// collection jobs. At most `maximum_acquire_count` jobs are acquired. The job is acquired with
    /// a "lease" that will time out; the desired duration of the lease is a parameter, and the
    /// lease expiration time is returned.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn acquire_incomplete_collection_jobs(
        &self,
        lease_duration: &StdDuration,
        maximum_acquire_count: usize,
    ) -> Result<Vec<Lease<AcquiredCollectionJob>>, Error> {
        let now = self.clock.now().as_naive_date_time()?;
        let lease_expiry_time = add_naive_date_time_duration(&now, lease_duration)?;
        let maximum_acquire_count: i64 = maximum_acquire_count.try_into()?;

        let stmt = self
            .prepare_cached(
                "-- acquire_incomplete_collection_jobs()
WITH incomplete_jobs AS (
    SELECT
        collection_jobs.id, collection_jobs.batch_identifier, tasks.task_id,
        tasks.query_type, tasks.vdaf, tasks.time_precision
    FROM collection_jobs
    JOIN tasks ON tasks.id = collection_jobs.task_id
    WHERE tasks.aggregator_role = 'LEADER'
      AND collection_jobs.state = 'START'
      AND collection_jobs.lease_expiry <= $4
      AND COALESCE(
              LOWER(batch_interval),
              (SELECT MAX(UPPER(ba.client_timestamp_interval))
               FROM batch_aggregations ba
               WHERE ba.task_id = collection_jobs.task_id
                 AND ba.batch_identifier = collection_jobs.batch_identifier
                 AND ba.aggregation_param = collection_jobs.aggregation_param),
              '-infinity'::TIMESTAMP)
          >= COALESCE(
                 $4::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL,
                 '-infinity'::TIMESTAMP
             )
    FOR UPDATE OF collection_jobs SKIP LOCKED LIMIT $5
)
UPDATE collection_jobs SET
    lease_expiry = $1,
    lease_token = gen_random_bytes(16),
    lease_attempts = lease_attempts + 1,
    updated_at = $2,
    updated_by = $3
FROM incomplete_jobs
WHERE collection_jobs.id = incomplete_jobs.id
RETURNING
    incomplete_jobs.task_id, incomplete_jobs.query_type, incomplete_jobs.vdaf,
    incomplete_jobs.time_precision, collection_jobs.collection_job_id,
    collection_jobs.batch_identifier, collection_jobs.aggregation_param,
    collection_jobs.lease_token, collection_jobs.lease_attempts,
    collection_jobs.step_attempts",
            )
            .await?;

        self.query(
            &stmt,
            &[
                /* lease_expiry */ &lease_expiry_time,
                /* updated_at */ &now,
                /* updated_by */ &self.name,
                /* now */ &now,
                /* limit */ &maximum_acquire_count,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            let collection_job_id =
                row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
            let query_type = row.try_get::<_, Json<task::QueryType>>("query_type")?.0;
            let vdaf = row.try_get::<_, Json<VdafInstance>>("vdaf")?.0;
            let time_precision =
                Duration::from_seconds(row.get_bigint_and_convert("time_precision")?);
            let encoded_batch_identifier = row.get("batch_identifier");
            let encoded_aggregation_param = row.get("aggregation_param");
            let lease_token = row.get_bytea_and_convert::<LeaseToken>("lease_token")?;
            let lease_attempts = row.get_bigint_and_convert("lease_attempts")?;
            let step_attempts = row.get_bigint_and_convert("step_attempts")?;

            Ok(Lease::new(
                AcquiredCollectionJob::new(
                    task_id,
                    collection_job_id,
                    query_type,
                    vdaf,
                    time_precision,
                    encoded_batch_identifier,
                    encoded_aggregation_param,
                    step_attempts,
                ),
                lease_expiry_time,
                lease_token,
                lease_attempts,
            ))
        })
        .collect()
    }

    /// release_collection_job releases an acquired (via e.g. acquire_incomplete_collection_jobs)
    /// collect job. If given, `reacquire_delay` determines the duration of time that must pass
    /// before the collection job can be reacquired; this method assumes a reacquire delay indicates
    /// that no progress was made, and will increment `step_attempts` accordingly. It returns an
    /// error if the collection job has no current lease.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn release_collection_job(
        &self,
        lease: &Lease<AcquiredCollectionJob>,
        reacquire_delay: Option<&StdDuration>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(lease.leased().task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let lease_expiration = reacquire_delay
            .map(|rd| add_naive_date_time_duration(&now, rd))
            .transpose()?
            .map(Timestamp::Value)
            .unwrap_or_else(|| Timestamp::NegInfinity);

        let stmt = self
            .prepare_cached(
                "-- release_collection_job()
UPDATE collection_jobs
SET lease_expiry = $1,
    lease_token = NULL,
    lease_attempts = 0,
    step_attempts = CASE
            WHEN $6 = '-infinity'::TIMESTAMP THEN 0
            ELSE step_attempts + 1
        END,
    updated_at = $2,
    updated_by = $3
WHERE task_id = $4
  AND collection_job_id = $5
  AND lease_expiry = $6
  AND lease_token = $7
  AND COALESCE(
          LOWER(batch_interval),
          (SELECT MAX(UPPER(ba.client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = collection_jobs.task_id
             AND ba.batch_identifier = collection_jobs.batch_identifier
             AND ba.aggregation_param = collection_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $8",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* lease_expiry */ &lease_expiration,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_info.pkey,
                    /* collection_job_id */ &lease.leased().collection_job_id().as_ref(),
                    /* lease_expiry */ &lease.lease_expiry_time(),
                    /* lease_token */ &lease.lease_token().as_ref(),
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Updates an existing collection job.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn update_collection_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        collection_job: &CollectionJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(collection_job.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let (
            report_count,
            client_timestamp_interval,
            leader_aggregate_share,
            helper_aggregate_share,
        ) = match collection_job.state() {
            CollectionJobState::Start => {
                return Err(Error::InvalidParameter(
                    "cannot update collection job into START state",
                ));
            }

            CollectionJobState::Finished {
                report_count,
                client_timestamp_interval,
                encrypted_helper_aggregate_share,
                leader_aggregate_share,
            } => {
                let report_count = Some(i64::try_from(*report_count)?);
                let client_timestamp_interval = Some(SqlInterval::from(client_timestamp_interval));
                let leader_aggregate_share = Some(leader_aggregate_share.get_encoded()?);
                let helper_aggregate_share = Some(encrypted_helper_aggregate_share.get_encoded()?);

                (
                    report_count,
                    client_timestamp_interval,
                    leader_aggregate_share,
                    helper_aggregate_share,
                )
            }

            CollectionJobState::Abandoned | CollectionJobState::Deleted => (None, None, None, None),
        };

        let stmt = self
            .prepare_cached(
                "-- update_collection_job()
UPDATE collection_jobs SET
    state = $1,
    report_count = $2,
    client_timestamp_interval = $3,
    leader_aggregate_share = $4,
    helper_aggregate_share = $5,
    updated_at = $6,
    updated_by = $7
WHERE task_id = $8
  AND collection_job_id = $9
  AND COALESCE(
          LOWER(batch_interval),
          (SELECT MAX(UPPER(ba.client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = collection_jobs.task_id
             AND ba.batch_identifier = collection_jobs.batch_identifier
             AND ba.aggregation_param = collection_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $10",
            )
            .await?;

        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */ &collection_job.state().collection_job_state_code(),
                    /* report_count */ &report_count,
                    /* client_timestamp_interval */ &client_timestamp_interval,
                    /* leader_aggregate_share */ &leader_aggregate_share,
                    /* helper_aggregate_share */ &helper_aggregate_share,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_info.pkey,
                    /* collection_job_id */ &collection_job.id().as_ref(),
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Retrieves an existing batch aggregation.
    #[tracing::instrument(skip(self, aggregation_parameter), err(level = Level::DEBUG))]
    pub async fn get_batch_aggregation<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
        aggregation_parameter: &A::AggregationParam,
        ord: u64,
    ) -> Result<Option<BatchAggregation<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param) which
        // are _not_ garbage collected. This is used to evaluate whether given batch_aggregations
        // rows are GC'ed.
        let stmt = self
            .prepare_cached(
                "-- get_batch_aggregation()
WITH non_gc_batches AS (
    SELECT batch_identifier, aggregation_param
    FROM batch_aggregations
    WHERE task_id = $1
      AND batch_identifier = $2
      AND aggregation_param = $3
    GROUP BY batch_identifier, aggregation_param
    HAVING MAX(UPPER(COALESCE(batch_interval, client_timestamp_interval))) >= $5
)
SELECT
    client_timestamp_interval, batch_aggregations.state, aggregate_share,
    report_count, checksum, aggregation_jobs_created, aggregation_jobs_terminated
FROM batch_aggregations
WHERE task_id = $1
  AND batch_identifier = $2
  AND aggregation_param = $3
  AND ord = $4
  AND EXISTS(SELECT 1 FROM non_gc_batches
             WHERE batch_identifier = $2
               AND aggregation_param = $3)",
            )
            .await?;

        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_identifier */ &batch_identifier.get_encoded()?,
                /* aggregation_param */ &aggregation_parameter.get_encoded()?,
                /* ord */ &TryInto::<i64>::try_into(ord)?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| {
            Self::batch_aggregation_from_row(
                vdaf,
                *task_id,
                batch_identifier.clone(),
                aggregation_parameter.clone(),
                ord,
                row,
            )
        })
        .transpose()
    }

    /// Retrieves all batch aggregations stored for a given batch, identified by task ID, batch
    /// identifier, and aggregation parameter.
    #[tracing::instrument(skip(self, aggregation_parameter), err(level = Level::DEBUG))]
    pub async fn get_batch_aggregations_for_batch<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
        aggregation_parameter: &A::AggregationParam,
    ) -> Result<Vec<BatchAggregation<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param) which
        // are _not_ garbage collected. This is used to evaluate whether given batch_aggregations
        // rows are GC'ed.
        let stmt = self
            .prepare_cached(
                "-- get_batch_aggregations_for_batch()
WITH non_gc_batches AS (
    SELECT batch_identifier, aggregation_param
    FROM batch_aggregations
    WHERE task_id = $1
      AND batch_identifier = $2
      AND aggregation_param = $3
    GROUP BY batch_identifier, aggregation_param
    HAVING MAX(UPPER(COALESCE(batch_interval, client_timestamp_interval))) >= $4
)
SELECT
    ord, client_timestamp_interval, batch_aggregations.state, aggregate_share,
    report_count, checksum, aggregation_jobs_created, aggregation_jobs_terminated
FROM batch_aggregations
WHERE task_id = $1
  AND batch_identifier = $2
  AND aggregation_param = $3
  AND EXISTS(SELECT 1 FROM non_gc_batches
             WHERE batch_identifier = $2
               AND aggregation_param = $3)",
            )
            .await?;

        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_identifier */ &batch_identifier.get_encoded()?,
                /* aggregation_param */ &aggregation_parameter.get_encoded()?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            Self::batch_aggregation_from_row(
                vdaf,
                *task_id,
                batch_identifier.clone(),
                aggregation_parameter.clone(),
                row.get_bigint_and_convert("ord")?,
                row,
            )
        })
        .collect()
    }

    /// Retrieves the number of aggregation jobs created & terminated for the given batch
    /// identifier.
    #[tracing::instrument(skip(self, aggregation_parameter), err(level = Level::DEBUG))]
    pub async fn get_batch_aggregation_job_count_for_batch<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
        aggregation_parameter: &A::AggregationParam,
    ) -> Result<(u64, u64), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok((0, 0)),
        };

        // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param) which
        // are _not_ garbage collected. This is used to evaluate whether given batch_aggregations
        // rows are GC'ed.
        let stmt = self
            .prepare_cached(
                "-- get_batch_aggregation_job_count_for_batch()
WITH non_gc_batches AS (
    SELECT batch_identifier, aggregation_param
    FROM batch_aggregations
    WHERE task_id = $1
      AND batch_identifier = $2
      AND aggregation_param = $3
    GROUP BY batch_identifier, aggregation_param
    HAVING MAX(UPPER(COALESCE(batch_interval, client_timestamp_interval))) >= $4
)
SELECT
    SUM(aggregation_jobs_created)::BIGINT AS created,
    SUM(aggregation_jobs_terminated)::BIGINT AS terminated
FROM batch_aggregations
WHERE task_id = $1
  AND batch_identifier = $2
  AND aggregation_param = $3
  AND EXISTS(SELECT 1 FROM non_gc_batches
             WHERE batch_identifier = $2
               AND aggregation_param = $3)",
            )
            .await?;

        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_identifier */ &batch_identifier.get_encoded()?,
                    /* aggregation_param */ &aggregation_parameter.get_encoded()?,
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                ],
            )
            .await?;

        let aggregation_jobs_created = row.get_nullable_bigint_and_convert("created")?;
        let aggregation_jobs_terminated = row.get_nullable_bigint_and_convert("terminated")?;

        Ok((
            aggregation_jobs_created.unwrap_or(0),
            aggregation_jobs_terminated.unwrap_or(0),
        ))
    }

    #[cfg(feature = "test-util")]
    pub async fn get_batch_aggregations_for_task<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
    ) -> Result<Vec<BatchAggregation<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param) which
        // are _not_ garbage collected. This is used to evaluate whether given batch_aggregations
        // rows are GC'ed.
        let stmt = self
            .prepare_cached(
                "-- get_batch_aggregations_for_task()
WITH non_gc_batches AS (
    SELECT batch_identifier, aggregation_param
    FROM batch_aggregations
    WHERE task_id = $1
    GROUP BY batch_identifier, aggregation_param
    HAVING MAX(UPPER(COALESCE(batch_interval, client_timestamp_interval))) >= $2
)
SELECT
    client_timestamp_interval, batch_aggregations.batch_identifier,
    batch_aggregations.aggregation_param, ord, batch_aggregations.state,
    aggregate_share, report_count, checksum, aggregation_jobs_created,
    aggregation_jobs_terminated
FROM batch_aggregations
WHERE task_id = $1
  AND EXISTS(SELECT 1 FROM non_gc_batches
             WHERE batch_identifier = batch_aggregations.batch_identifier
               AND aggregation_param = batch_aggregations.aggregation_param)",
            )
            .await?;

        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let batch_identifier = Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
            let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
            let ord = row.get_bigint_and_convert("ord")?;

            Self::batch_aggregation_from_row(
                vdaf,
                *task_id,
                batch_identifier,
                aggregation_param,
                ord,
                row,
            )
        })
        .collect()
    }

    fn batch_aggregation_from_row<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        vdaf: &A,
        task_id: TaskId,
        batch_identifier: Q::BatchIdentifier,
        aggregation_param: A::AggregationParam,
        ord: u64,
        row: Row,
    ) -> Result<BatchAggregation<SEED_SIZE, Q, A>, Error> {
        #[allow(clippy::type_complexity)]
        fn parse_values_from_row<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>(
            vdaf: &A,
            aggregation_param: &A::AggregationParam,
            row: &Row,
        ) -> Result<(Option<A::AggregateShare>, u64, ReportIdChecksum, u64, u64), Error> {
            let aggregate_share = row
                .get::<_, Option<Vec<u8>>>("aggregate_share")
                .map(|bytes| {
                    A::AggregateShare::get_decoded_with_param(&(vdaf, aggregation_param), &bytes)
                })
                .transpose()
                .map_err(|_| Error::DbState("aggregate_share couldn't be parsed".to_string()))?;
            let report_count = row.get_bigint_and_convert("report_count")?;
            let checksum = ReportIdChecksum::get_decoded(row.get("checksum"))?;
            let aggregation_jobs_created =
                row.get_bigint_and_convert("aggregation_jobs_created")?;
            let aggregation_jobs_terminated =
                row.get_bigint_and_convert("aggregation_jobs_terminated")?;

            Ok((
                aggregate_share,
                report_count,
                checksum,
                aggregation_jobs_created,
                aggregation_jobs_terminated,
            ))
        }

        let client_timestamp_interval = row
            .get::<_, SqlInterval>("client_timestamp_interval")
            .as_interval();
        let state: BatchAggregationStateCode = row.get("state");
        let state = match state {
            BatchAggregationStateCode::Aggregating => {
                let (
                    aggregate_share,
                    report_count,
                    checksum,
                    aggregation_jobs_created,
                    aggregation_jobs_terminated,
                ) = parse_values_from_row(vdaf, &aggregation_param, &row)?;
                BatchAggregationState::Aggregating {
                    aggregate_share,
                    report_count,
                    checksum,
                    aggregation_jobs_created,
                    aggregation_jobs_terminated,
                }
            }
            BatchAggregationStateCode::Collected => {
                let (
                    aggregate_share,
                    report_count,
                    checksum,
                    aggregation_jobs_created,
                    aggregation_jobs_terminated,
                ) = parse_values_from_row(vdaf, &aggregation_param, &row)?;
                BatchAggregationState::Collected {
                    aggregate_share,
                    report_count,
                    checksum,
                    aggregation_jobs_created,
                    aggregation_jobs_terminated,
                }
            }
            BatchAggregationStateCode::Scrubbed => BatchAggregationState::Scrubbed,
        };

        Ok(BatchAggregation::new(
            task_id,
            batch_identifier,
            aggregation_param,
            ord,
            client_timestamp_interval,
            state,
        ))
    }

    /// Store a new `batch_aggregations` row in the datastore.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_batch_aggregation<
        const SEED_SIZE: usize,
        Q: AccumulableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        batch_aggregation: &BatchAggregation<SEED_SIZE, Q, A>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Debug,
        A::AggregateShare: Debug,
    {
        let task_info = match self.task_info_for(batch_aggregation.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let batch_interval =
            Q::to_batch_interval(batch_aggregation.batch_identifier()).map(SqlInterval::from);
        let encoded_state_values = batch_aggregation.state().encoded_values_from_state()?;

        let stmt = self
            .prepare_cached(
                "-- put_batch_aggregation()
INSERT INTO batch_aggregations (
    task_id, batch_identifier, batch_interval, aggregation_param, ord,
    client_timestamp_interval, state, aggregate_share, report_count, checksum,
    aggregation_jobs_created, aggregation_jobs_terminated, created_at, updated_at,
    updated_by
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
ON CONFLICT(task_id, batch_identifier, aggregation_param, ord) DO UPDATE
    SET (
        client_timestamp_interval, state, aggregate_share, report_count, checksum,
        aggregation_jobs_created, aggregation_jobs_terminated, created_at,
        updated_at, updated_by
    ) = (
        excluded.client_timestamp_interval, excluded.state,
        excluded.aggregate_share, excluded.report_count, excluded.checksum,
        excluded.aggregation_jobs_created, excluded.aggregation_jobs_terminated,
        excluded.created_at, excluded.updated_at, excluded.updated_by
    )
    WHERE GREATEST(
              UPPER(COALESCE(batch_aggregations.batch_interval,
                             batch_aggregations.client_timestamp_interval)),
              (SELECT MAX(UPPER(COALESCE(ba.batch_interval,
                                         ba.client_timestamp_interval)))
               FROM batch_aggregations ba
               WHERE ba.task_id = batch_aggregations.task_id
                 AND ba.batch_identifier = batch_aggregations.batch_identifier
                 AND ba.aggregation_param = batch_aggregations.aggregation_param)) < $16",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_identifier */
                    &batch_aggregation.batch_identifier().get_encoded()?,
                    /* batch_interval */ &batch_interval,
                    /* aggregation_param */
                    &batch_aggregation.aggregation_parameter().get_encoded()?,
                    /* ord */ &i64::try_from(batch_aggregation.ord())?,
                    /* client_timestamp_interval */
                    &SqlInterval::from(batch_aggregation.client_timestamp_interval()),
                    /* state */ &batch_aggregation.state().state_code(),
                    /* aggregate_share */ &encoded_state_values.aggregate_share,
                    /* report_count */ &encoded_state_values.report_count,
                    /* checksum */ &encoded_state_values.checksum,
                    /* aggregation_jobs_created */
                    &encoded_state_values.aggregation_jobs_created,
                    /* aggregation_jobs_terminated */
                    &encoded_state_values.aggregation_jobs_terminated,
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Update an existing `batch_aggregations` row with the values from the provided batch
    /// aggregation.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn update_batch_aggregation<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        batch_aggregation: &BatchAggregation<SEED_SIZE, Q, A>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: Debug,
        A::AggregateShare: Debug,
    {
        let task_info = match self.task_info_for(batch_aggregation.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;
        let encoded_state_values = batch_aggregation.state().encoded_values_from_state()?;

        let stmt = self
            .prepare_cached(
                "-- update_batch_aggregations()
UPDATE batch_aggregations
SET
    client_timestamp_interval = $1,
    state = $2,
    aggregate_share = $3,
    report_count = $4,
    checksum = $5,
    aggregation_jobs_created = $6,
    aggregation_jobs_terminated = $7,
    updated_at = $8,
    updated_by = $9
WHERE task_id = $10
  AND batch_identifier = $11
  AND aggregation_param = $12
  AND ord = $13
  AND GREATEST(
          UPPER($1::TSRANGE),
          (SELECT MAX(UPPER(COALESCE(batch_interval, client_timestamp_interval)))
           FROM batch_aggregations ba
           WHERE ba.task_id = batch_aggregations.task_id
             AND ba.batch_identifier = batch_aggregations.batch_identifier
             AND ba.aggregation_param = batch_aggregations.aggregation_param)) >= $14",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* client_timestamp_interval */
                    &SqlInterval::from(batch_aggregation.client_timestamp_interval()),
                    /* state */ &batch_aggregation.state().state_code(),
                    /* aggregate_share */ &encoded_state_values.aggregate_share,
                    /* report_count */ &encoded_state_values.report_count,
                    /* checksum */ &encoded_state_values.checksum,
                    /* aggregation_jobs_created */
                    &encoded_state_values.aggregation_jobs_created,
                    /* aggregation_jobs_terminated */
                    &encoded_state_values.aggregation_jobs_terminated,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* task_id */ &task_info.pkey,
                    /* batch_identifier */
                    &batch_aggregation.batch_identifier().get_encoded()?,
                    /* aggregation_param */
                    &batch_aggregation.aggregation_parameter().get_encoded()?,
                    /* ord */
                    &TryInto::<i64>::try_into(batch_aggregation.ord())?,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Fetch an [`AggregateShareJob`] from the datastore corresponding to given parameters, or
    /// `None` if no such job exists.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregate_share_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
        aggregation_parameter: &A::AggregationParam,
    ) -> Result<Option<AggregateShareJob<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_aggregate_share_job()
SELECT helper_aggregate_share, report_count, checksum
FROM aggregate_share_jobs
WHERE task_id = $1
  AND batch_identifier = $2
  AND aggregation_param = $3
  AND COALESCE(
          LOWER(batch_interval),
          (SELECT MAX(UPPER(client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = aggregate_share_jobs.task_id
             AND ba.batch_identifier = aggregate_share_jobs.batch_identifier
             AND ba.aggregation_param = aggregate_share_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $4",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_identifier */ &batch_identifier.get_encoded()?,
                /* aggregation_param */ &aggregation_parameter.get_encoded()?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| {
            Self::aggregate_share_job_from_row(
                vdaf,
                task_id,
                batch_identifier.clone(),
                aggregation_parameter.clone(),
                &row,
            )
        })
        .transpose()
    }

    /// Returns all aggregate share jobs for the given task whose collect intervals intersect with
    /// the given interval. Applies only to time-interval tasks.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregate_share_jobs_intersecting_interval<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        interval: &Interval,
    ) -> Result<Vec<AggregateShareJob<SEED_SIZE, TimeInterval, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_aggregate_share_jobs_intersecting_interval()
SELECT
    batch_identifier, aggregation_param, helper_aggregate_share, report_count,
    checksum
FROM aggregate_share_jobs
WHERE task_id = $1
  AND batch_interval && $2
  AND LOWER(aggregate_share_jobs.batch_interval) >= $3",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* interval */ &SqlInterval::from(interval),
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let batch_identifier = Interval::get_decoded(row.get("batch_identifier"))?;
            let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
            Self::aggregate_share_job_from_row(
                vdaf,
                task_id,
                batch_identifier,
                aggregation_param,
                &row,
            )
        })
        .collect()
    }

    /// Returns all aggregate share jobs for the given task with the given batch identifier.
    /// Multiple aggregate share jobs may be returned with distinct aggregation parameters.
    /// Applies only to fixed-size tasks.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_aggregate_share_jobs_by_batch_id<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<Vec<AggregateShareJob<SEED_SIZE, FixedSize, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_aggregate_share_jobs_by_batch_id()
SELECT
    aggregation_param, helper_aggregate_share, report_count, checksum
FROM aggregate_share_jobs
WHERE task_id = $1
  AND batch_identifier = $2
  AND COALESCE(
          (SELECT MAX(UPPER(client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = aggregate_share_jobs.task_id
             AND ba.batch_identifier = aggregate_share_jobs.batch_identifier
             AND ba.aggregation_param = aggregate_share_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $3",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* batch_id */ &batch_id.get_encoded()?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
            Self::aggregate_share_job_from_row(vdaf, task_id, *batch_id, aggregation_param, &row)
        })
        .collect()
    }

    #[cfg(feature = "test-util")]
    pub async fn get_aggregate_share_jobs_for_task<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
    ) -> Result<Vec<AggregateShareJob<SEED_SIZE, Q, A>>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let stmt = self
            .prepare_cached(
                "-- get_aggregate_share_jobs_for_task()
SELECT
    batch_identifier, aggregation_param, helper_aggregate_share, report_count,
    checksum
FROM aggregate_share_jobs
WHERE task_id = $1
  AND COALESCE(
          LOWER(aggregate_share_jobs.batch_interval),
          (SELECT MAX(UPPER(client_timestamp_interval))
           FROM batch_aggregations ba
           WHERE ba.task_id = aggregate_share_jobs.task_id
             AND ba.batch_identifier = aggregate_share_jobs.batch_identifier
             AND ba.aggregation_param = aggregate_share_jobs.aggregation_param),
          '-infinity'::TIMESTAMP) >= $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let batch_identifier = Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
            let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
            Self::aggregate_share_job_from_row(
                vdaf,
                task_id,
                batch_identifier,
                aggregation_param,
                &row,
            )
        })
        .collect()
    }

    fn aggregate_share_job_from_row<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: Q::BatchIdentifier,
        aggregation_param: A::AggregationParam,
        row: &Row,
    ) -> Result<AggregateShareJob<SEED_SIZE, Q, A>, Error> {
        let helper_aggregate_share =
            row.get_bytea_and_decode("helper_aggregate_share", &(vdaf, &aggregation_param))?;
        Ok(AggregateShareJob::new(
            *task_id,
            batch_identifier,
            aggregation_param,
            helper_aggregate_share,
            row.get_bigint_and_convert("report_count")?,
            ReportIdChecksum::get_decoded(row.get("checksum"))?,
        ))
    }

    /// Put an `aggregate_share_job` row into the datastore.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_aggregate_share_job<
        const SEED_SIZE: usize,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        aggregate_share_job: &AggregateShareJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(aggregate_share_job.task_id()).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;
        let batch_interval =
            Q::to_batch_interval(aggregate_share_job.batch_identifier()).map(SqlInterval::from);

        let stmt = self
            .prepare_cached(
                "-- put_aggregate_share_job()
INSERT INTO aggregate_share_jobs (
    task_id, batch_identifier, batch_interval, aggregation_param,
    helper_aggregate_share, report_count, checksum, created_at, updated_by
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT(task_id, batch_identifier, aggregation_param) DO UPDATE
    SET (
        helper_aggregate_share, report_count, checksum, created_at, updated_by
    ) = (
        excluded.helper_aggregate_share, excluded.report_count, excluded.checksum,
        excluded.created_at, excluded.updated_by
    )
    WHERE COALESCE(
              LOWER(aggregate_share_jobs.batch_interval),
              (SELECT MAX(UPPER(ba.client_timestamp_interval))
               FROM batch_aggregations ba
               WHERE ba.task_id = aggregate_share_jobs.task_id
                 AND ba.batch_identifier = aggregate_share_jobs.batch_identifier
                 AND ba.aggregation_param = aggregate_share_jobs.aggregation_param),
              '-infinity'::TIMESTAMP) < $10",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_identifier */
                    &aggregate_share_job.batch_identifier().get_encoded()?,
                    /* batch_interval */ &batch_interval,
                    /* aggregation_param */
                    &aggregate_share_job.aggregation_parameter().get_encoded()?,
                    /* helper_aggregate_share */
                    &aggregate_share_job.helper_aggregate_share().get_encoded()?,
                    /* report_count */ &i64::try_from(aggregate_share_job.report_count())?,
                    /* checksum */ &aggregate_share_job.checksum().get_encoded()?,
                    /* created_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Writes an outstanding batch. (This method does not take an [`OutstandingBatch`] as several
    /// of the included values are read implicitly.)
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_outstanding_batch(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
        time_bucket_start: &Option<Time>,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param) which
        // are _not_ garbage collected. This is used to evaluate whether given batch_aggregations
        // rows are GC'ed.
        //
        // Note that this ignores aggregation parameter, as `outstanding_batches` does not need to
        // worry about aggregation parameters.
        //
        // TODO(#225): reevaluate whether we can ignore aggregation parameter here once we have
        // experience with VDAFs requiring multiple aggregations per batch.
        let stmt = self
            .prepare_cached(
                "-- put_outstanding_batch()
WITH non_gc_batches AS (
    SELECT batch_identifier
    FROM batch_aggregations
    WHERE task_id = $1
      AND batch_identifier = $2
    GROUP BY batch_identifier
    HAVING MAX(UPPER(client_timestamp_interval)) >= $6
)
INSERT INTO outstanding_batches (
    task_id, batch_id, time_bucket_start, created_at, updated_by
)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT(task_id, batch_id) DO UPDATE
    SET (
        time_bucket_start, created_at, updated_by
    ) = (
        excluded.time_bucket_start, excluded.created_at, excluded.updated_by
    )
    WHERE NOT EXISTS(SELECT 1 FROM non_gc_batches
                     WHERE batch_identifier = outstanding_batches.batch_id)",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_id */ batch_id.as_ref(),
                    /* time_bucket_start */
                    &time_bucket_start
                        .as_ref()
                        .map(Time::as_naive_date_time)
                        .transpose()?,
                    /* created_at */ &now,
                    /* updated_by */ &self.name,
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Retrieves all [`OutstandingBatch`]es for a given task and (if applicable) time bucket which
    /// have not yet been marked filled.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_unfilled_outstanding_batches(
        &self,
        task_id: &TaskId,
        time_bucket_start: &Option<Time>,
    ) -> Result<Vec<OutstandingBatch>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(Vec::new()),
        };

        let rows = if let Some(time_bucket_start) = time_bucket_start {
            // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param)
            // which are _not_ garbage collected. This is used to evaluate whether given
            // batch_aggregations rows are GC'ed.
            let stmt = self
                .prepare_cached(
                    "-- get_unfilled_outstanding_batches()
WITH non_gc_batches AS (
    SELECT batch_identifier
    FROM batch_aggregations
    WHERE task_id = $1
    GROUP BY batch_identifier
    HAVING MAX(UPPER(client_timestamp_interval)) >= $3
)
SELECT batch_id FROM outstanding_batches
WHERE task_id = $1
  AND time_bucket_start = $2
  AND state = 'FILLING'
  AND EXISTS(SELECT 1 FROM non_gc_batches
             WHERE batch_identifier = outstanding_batches.batch_id)",
                )
                .await?;
            self.query(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* time_bucket_start */ &time_bucket_start.as_naive_date_time()?,
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                ],
            )
            .await?
        } else {
            // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param)
            // which are _not_ garbage collected. This is used to evaluate whether given
            // batch_aggregations rows are GC'ed.
            let stmt = self
                .prepare_cached(
                    "-- get_unfilled_outstanding_batches()
WITH non_gc_batches AS (
    SELECT batch_identifier
    FROM batch_aggregations
    WHERE task_id = $1
    GROUP BY batch_identifier
    HAVING MAX(UPPER(client_timestamp_interval)) >= $2
)
SELECT batch_id FROM outstanding_batches
WHERE task_id = $1
  AND time_bucket_start IS NULL
  AND state = 'FILLING'
  AND EXISTS(SELECT 1 FROM non_gc_batches
             WHERE batch_identifier = outstanding_batches.batch_id)",
                )
                .await?;
            self.query(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                ],
            )
            .await?
        };

        try_join_all(rows.into_iter().map(|row| async move {
            let batch_id = BatchId::get_decoded(row.get("batch_id"))?;
            let size = self.read_batch_size(task_info.pkey, &batch_id).await?;
            Ok(OutstandingBatch::new(*task_id, batch_id, size))
        }))
        .await
    }

    // Return value is an inclusive range [min_size, max_size], where:
    //  * min_size is the minimum possible number of reports included in the batch, i.e. all report
    //    aggregations in the batch which have reached the FINISHED state.
    //  * max_size is the maximum possible number of reports included in the batch, i.e. all report
    //    aggregations in the batch which are in a non-failure state (START/WAITING/FINISHED).
    async fn read_batch_size(
        &self,
        task_pkey: i64,
        batch_id: &BatchId,
    ) -> Result<RangeInclusive<usize>, Error> {
        let stmt = self
            .prepare_cached(
                "-- read_batch_size()
WITH report_aggregations_count AS (
    SELECT COUNT(*) AS count FROM report_aggregations
    JOIN aggregation_jobs
        ON report_aggregations.aggregation_job_id = aggregation_jobs.id
    WHERE aggregation_jobs.task_id = $1
    AND report_aggregations.task_id = aggregation_jobs.task_id
    AND aggregation_jobs.batch_id = $2
    AND report_aggregations.state in ('START', 'WAITING')
),
batch_aggregation_count AS (
    SELECT SUM(report_count) AS count FROM batch_aggregations
    WHERE batch_aggregations.task_id = $1
    AND batch_aggregations.batch_identifier = $2
)
SELECT
    (SELECT count FROM batch_aggregation_count)::BIGINT AS min_size,
    (SELECT count FROM report_aggregations_count)::BIGINT
        + (SELECT count FROM batch_aggregation_count)::BIGINT AS max_size",
            )
            .await?;

        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_pkey,
                    /* batch_id */ batch_id.as_ref(),
                ],
            )
            .await?;

        Ok(RangeInclusive::new(
            row.get::<_, Option<i64>>("min_size")
                .unwrap_or_default()
                .try_into()?,
            row.get::<_, Option<i64>>("max_size")
                .unwrap_or_default()
                .try_into()?,
        ))
    }

    /// Retrieves an outstanding batch for the given task with at least the given number of
    /// successfully-aggregated reports, removing it from the datastore.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn acquire_outstanding_batch_with_report_count(
        &self,
        task_id: &TaskId,
        min_report_count: u64,
    ) -> Result<Option<BatchId>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        // non_gc_batches finds batches (by task ID, batch identifier, and aggregation param) which
        // are _not_ garbage collected. This is used to evaluate whether given batch_aggregations
        // rows are GC'ed.
        let stmt = self
            .prepare_cached(
                "-- acquire_outstanding_batch_with_report_count()
WITH non_gc_batches AS (
    SELECT batch_identifier, SUM(report_count) AS report_count
    FROM batch_aggregations
    WHERE task_id = $1
    GROUP BY batch_identifier
    HAVING MAX(UPPER(client_timestamp_interval)) >= $3
),
selected_outstanding_batch AS (
    SELECT outstanding_batches.id
    FROM outstanding_batches
    WHERE task_id = $1
    AND (SELECT report_count FROM non_gc_batches
        WHERE batch_identifier = outstanding_batches.batch_id) >= $2::BIGINT
    LIMIT 1
)
DELETE FROM outstanding_batches WHERE id IN (SELECT id FROM selected_outstanding_batch)
RETURNING batch_id",
            )
            .await?;

        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* min_report_count */ &i64::try_from(min_report_count)?,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
            ],
        )
        .await?
        .map(|row| Ok(BatchId::get_decoded(row.get("batch_id"))?))
        .transpose()
    }

    /// Marks a given outstanding batch as filled, such that it will no longer be considered when
    /// assigning aggregation jobs to batches.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn mark_outstanding_batch_filled(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };
        let now = self.clock.now().as_naive_date_time()?;

        let stmt = self
            .prepare_cached(
                "-- mark_outstanding_batch_filled()
WITH non_gc_batches AS (
    SELECT batch_identifier
    FROM batch_aggregations
    WHERE task_id = $1
    AND batch_identifier = $2
    GROUP BY batch_identifier
    HAVING MAX(UPPER(client_timestamp_interval)) >= $3
)
UPDATE outstanding_batches
SET state = 'FILLED'
WHERE task_id = $1
AND batch_id = $2
AND EXISTS(SELECT 1 FROM non_gc_batches WHERE batch_identifier = $2)",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* batch_id */ batch_id.as_ref(),
                    /* threshold */ &task_info.report_expiry_threshold(&now)?,
                ],
            )
            .await?,
        )
    }

    /// Deletes old client reports for a given task, that is, client reports whose timestamp is
    /// older than the task's report expiry age. Up to `limit` client reports will be deleted.
    /// Returns the number of client reports deleted.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn delete_expired_client_reports(
        &self,
        task_id: &TaskId,
        limit: u64,
    ) -> Result<u64, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(0),
        };

        let stmt = self
            .prepare_cached(
                "-- delete_expired_client_reports()
WITH client_reports_to_delete AS (
    SELECT client_reports.id FROM client_reports
    WHERE client_reports.task_id = $1
        AND client_reports.client_timestamp < $2::TIMESTAMP
    LIMIT $3
)
DELETE FROM client_reports
USING client_reports_to_delete
WHERE client_reports.id = client_reports_to_delete.id",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                /* limit */ &i64::try_from(limit)?,
            ],
        )
        .await
        .map_err(Into::into)
    }

    /// Deletes old aggregation artifacts (aggregation jobs/report aggregations) for a given task,
    /// that is, aggregation artifacts for which the aggregation job's maximum client timestamp is
    /// older than the task's report expiry age. Up to `limit` aggregation jobs will be deleted,
    /// along with all related aggregation artifacts. Returns the number of aggregation jobs
    /// deleted.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn delete_expired_aggregation_artifacts(
        &self,
        task_id: &TaskId,
        limit: u64,
    ) -> Result<u64, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(0),
        };

        let stmt = self
            .prepare_cached(
                "-- delete_expired_aggregation_artifacts()
WITH aggregation_jobs_to_delete AS (
    SELECT aggregation_jobs.id FROM aggregation_jobs
    WHERE task_id = $1
      AND UPPER(aggregation_jobs.client_timestamp_interval) < $2
    LIMIT $3
),
deleted_report_aggregations AS (
    DELETE FROM report_aggregations
    WHERE aggregation_job_id IN (SELECT id FROM aggregation_jobs_to_delete)
)
DELETE FROM aggregation_jobs
WHERE id IN (SELECT id FROM aggregation_jobs_to_delete)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &task_info.pkey,
                /* threshold */
                &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                /* limit */ &i64::try_from(limit)?,
            ],
        )
        .await
        .map_err(Into::into)
    }

    /// Deletes old collection artifacts (outstanding batches/batch aggregations/collection jobs/
    /// aggregate share jobs) for a given task per the following policy:
    ///
    /// * batch_aggregations and outstanding_batches will be considered part of the same entity for
    ///   purposes of GC, and will be considered eligible for GC once the maximum of the
    ///   batch_interval (for time-interval) or merged client_timestamp_interval (for fixed-size) of
    ///   the batch_aggregations rows is older than report_expiry_age.
    /// * collection_jobs and aggregate_share_jobs use a query-type-specific rule to determine
    ///   GC-eligiblity.
    ///   * For time-interval tasks, collection_jobs and aggregate_share_jobs are considered
    ///     eligible for GC if the minimum of the collection interval is older than
    ///     report_expiry_age. (The minimum is used instead of the maximum to ensure that collection
    ///     jobs are not GC'ed after their underlying aggregation information from
    ///     batch_aggregations.)
    ///   * For fixed-size tasks, collection_jobs and aggregate_share_jobs are considered eligible
    ///     for GC if the related batch is eligible for GC, based on the `batch_aggregations` rows.
    ///
    /// Up to `limit` batches will be deleted, along with all related collection artifacts.
    ///
    /// Returns the number of batches deleted.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn delete_expired_collection_artifacts(
        &self,
        task_id: &TaskId,
        limit: u64,
    ) -> Result<u64, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(0),
        };

        let stmt = self
            .prepare_cached(
                "-- delete_expired_collection_artifacts()
WITH candidate_batches_to_delete AS (
    SELECT DISTINCT batch_identifier, aggregation_param
    FROM batch_aggregations
    WHERE task_id = $1
      AND UPPER(COALESCE(batch_interval, client_timestamp_interval)) < $2
),
batches_to_delete AS (
    SELECT ba.batch_identifier, ba.aggregation_param
    FROM batch_aggregations AS ba
    JOIN candidate_batches_to_delete AS candidate
      ON ba.batch_identifier = candidate.batch_identifier
     AND ba.aggregation_param = candidate.aggregation_param
    WHERE ba.task_id = $1
    GROUP BY ba.batch_identifier, ba.aggregation_param
    HAVING MAX(UPPER(COALESCE(ba.batch_interval, ba.client_timestamp_interval))) < $2
    LIMIT $3
),
deleted_outstanding_batches AS (
    DELETE FROM outstanding_batches
    USING batches_to_delete
    WHERE task_id = $1
      AND outstanding_batches.batch_id = batches_to_delete.batch_identifier
),
deleted_collection_jobs AS (
    DELETE FROM collection_jobs
    USING batches_to_delete
    WHERE task_id = $1
      AND (LOWER(batch_interval) < $2
        OR (collection_jobs.batch_identifier = batches_to_delete.batch_identifier
          AND collection_jobs.aggregation_param = batches_to_delete.aggregation_param))
),
deleted_aggregate_share_jobs AS (
    DELETE FROM aggregate_share_jobs
    USING batches_to_delete
    WHERE task_id = $1
      AND (LOWER(batch_interval) < $2
        OR (aggregate_share_jobs.batch_identifier = batches_to_delete.batch_identifier
          AND aggregate_share_jobs.aggregation_param = batches_to_delete.aggregation_param))
),
deleted_batch_aggregations AS (
    DELETE FROM batch_aggregations
    USING batches_to_delete
    WHERE task_id = $1
      AND batch_aggregations.batch_identifier = batches_to_delete.batch_identifier
      AND batch_aggregations.aggregation_param = batches_to_delete.aggregation_param
)
SELECT COUNT(1) AS batch_count FROM batches_to_delete",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* threshold */
                    &task_info.report_expiry_threshold(&self.clock.now().as_naive_date_time()?)?,
                    /* limit */ &i64::try_from(limit)?,
                ],
            )
            .await?;
        row.get_bigint_and_convert("batch_count")
    }

    /// Take an ExclusiveLock on the global_hpke_keys table.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn lock_global_hpke_keypairs(&self) -> Result<(), Error> {
        self.raw_tx
            .batch_execute("LOCK TABLE global_hpke_keys IN EXCLUSIVE MODE")
            .await
            .map_err(|err| err.into())
    }

    /// Retrieve all global HPKE keypairs.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_global_hpke_keypairs(&self) -> Result<Vec<GlobalHpkeKeypair>, Error> {
        let stmt = self
            .prepare_cached(
                "-- get_global_hpke_keypairs()
SELECT config_id, config, private_key, state, last_state_change_at FROM global_hpke_keys",
            )
            .await?;
        let hpke_key_rows = self.query(&stmt, &[]).await?;

        hpke_key_rows
            .iter()
            .map(|row| self.global_hpke_keypair_from_row(row))
            .collect()
    }

    /// Retrieve a global HPKE keypair by config ID.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_global_hpke_keypair(
        &self,
        config_id: &HpkeConfigId,
    ) -> Result<Option<GlobalHpkeKeypair>, Error> {
        let stmt = self
            .prepare_cached(
                "-- get_global_hpke_keypair()
SELECT config_id, config, private_key, state, last_state_change_at FROM global_hpke_keys
    WHERE config_id = $1",
            )
            .await?;
        self.query_opt(&stmt, &[&(u8::from(*config_id) as i16)])
            .await?
            .map(|row| self.global_hpke_keypair_from_row(&row))
            .transpose()
    }

    fn global_hpke_keypair_from_row(&self, row: &Row) -> Result<GlobalHpkeKeypair, Error> {
        let config = HpkeConfig::get_decoded(row.get("config"))?;
        let config_id = u8::try_from(row.get::<_, i16>("config_id"))?;

        let encrypted_private_key: Vec<u8> = row.get("private_key");
        let private_key = HpkePrivateKey::new(self.crypter.decrypt(
            "global_hpke_keys",
            &config_id.to_be_bytes(),
            "private_key",
            &encrypted_private_key,
        )?);
        Ok(GlobalHpkeKeypair::new(
            HpkeKeypair::new(config, private_key),
            row.get("state"),
            Time::from_naive_date_time(&row.get("last_state_change_at")),
        ))
    }

    /// Unconditionally and fully drop a keypair. This is a dangerous operation,
    /// since report shares encrypted with this key will no longer be decryptable.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn delete_global_hpke_keypair(&self, config_id: &HpkeConfigId) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "-- delete_global_hpke_keypair()
DELETE FROM global_hpke_keys WHERE config_id = $1",
            )
            .await?;
        check_single_row_mutation(
            self.execute(&stmt, &[&(u8::from(*config_id) as i16)])
                .await?,
        )
    }

    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn set_global_hpke_keypair_state(
        &self,
        config_id: &HpkeConfigId,
        state: &HpkeKeyState,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "-- set_global_hpke_keypair_state()
UPDATE global_hpke_keys
    SET state = $1, last_state_change_at = $2, updated_at = $3, updated_by = $4
    WHERE config_id = $5",
            )
            .await?;
        let now = self.clock.now().as_naive_date_time()?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */ state,
                    /* last_state_change_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                    /* config_id */ &(u8::from(*config_id) as i16),
                ],
            )
            .await?,
        )
    }

    /// Inserts a new global HPKE keypair and places it in the [`HpkeKeyState::Pending`] state.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_global_hpke_keypair(&self, hpke_keypair: &HpkeKeypair) -> Result<(), Error> {
        let hpke_config_id = u8::from(*hpke_keypair.config().id()) as i16;
        let hpke_config = hpke_keypair.config().get_encoded()?;
        let encrypted_hpke_private_key = self.crypter.encrypt(
            "global_hpke_keys",
            &u8::from(*hpke_keypair.config().id()).to_be_bytes(),
            "private_key",
            hpke_keypair.private_key().as_ref(),
        )?;

        let stmt = self
            .prepare_cached(
                "-- put_global_hpke_keypair()
INSERT INTO global_hpke_keys
    (config_id, config, private_key, last_state_change_at, created_at, updated_at, updated_by)
    VALUES ($1, $2, $3, $4, $5, $6, $7)",
            )
            .await?;
        let now = self.clock.now().as_naive_date_time()?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* config_id */ &hpke_config_id,
                    /* config */ &hpke_config,
                    /* private_key */ &encrypted_hpke_private_key,
                    /* last_state_change_at */ &now,
                    /* created_at */ &now,
                    /* updated_at */ &now,
                    /* updated_by */ &self.name,
                ],
            )
            .await?,
        )
    }

    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_taskprov_peer_aggregators(&self) -> Result<Vec<PeerAggregator>, Error> {
        let stmt = self
            .prepare_cached(
                "-- get_taskprov_peer_aggregators()
SELECT id, endpoint, role, verify_key_init, collector_hpke_config,
        report_expiry_age, tolerable_clock_skew
    FROM taskprov_peer_aggregators",
            )
            .await?;
        let peer_aggregator_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "-- get_taskprov_peer_aggregators()
SELECT (SELECT p.id FROM taskprov_peer_aggregators AS p
    WHERE p.id = a.peer_aggregator_id) AS peer_id,
ord, type, token FROM taskprov_aggregator_auth_tokens AS a
    ORDER BY ord ASC",
            )
            .await?;
        let aggregator_auth_token_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "-- get_taskprov_peer_aggregators()
SELECT (SELECT p.id FROM taskprov_peer_aggregators AS p
    WHERE p.id = a.peer_aggregator_id) AS peer_id,
ord, type, token FROM taskprov_collector_auth_tokens AS a
    ORDER BY ord ASC",
            )
            .await?;
        let collector_auth_token_rows = self.query(&stmt, &[]);

        let (peer_aggregator_rows, aggregator_auth_token_rows, collector_auth_token_rows) = try_join!(
            peer_aggregator_rows,
            aggregator_auth_token_rows,
            collector_auth_token_rows,
        )?;

        let mut aggregator_auth_token_rows_by_peer_id: HashMap<i64, Vec<Row>> = HashMap::new();
        for row in aggregator_auth_token_rows {
            aggregator_auth_token_rows_by_peer_id
                .entry(row.get("peer_id"))
                .or_default()
                .push(row);
        }

        let mut collector_auth_token_rows_by_peer_id: HashMap<i64, Vec<Row>> = HashMap::new();
        for row in collector_auth_token_rows {
            collector_auth_token_rows_by_peer_id
                .entry(row.get("peer_id"))
                .or_default()
                .push(row);
        }

        peer_aggregator_rows
            .into_iter()
            .map(|row| (row.get("id"), row))
            .map(|(peer_id, peer_aggregator_row)| {
                self.taskprov_peer_aggregator_from_rows(
                    &peer_aggregator_row,
                    &aggregator_auth_token_rows_by_peer_id
                        .remove(&peer_id)
                        .unwrap_or_default(),
                    &collector_auth_token_rows_by_peer_id
                        .remove(&peer_id)
                        .unwrap_or_default(),
                )
            })
            .collect()
    }

    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_taskprov_peer_aggregator(
        &self,
        aggregator_url: &Url,
        role: &Role,
    ) -> Result<Option<PeerAggregator>, Error> {
        let aggregator_url = aggregator_url.as_str();
        let role = AggregatorRole::from_role(*role)?;
        let params: &[&(dyn ToSql + Sync)] = &[&aggregator_url, &role];

        let stmt = self
            .prepare_cached(
                "-- get_taskprov_peer_aggregator()
SELECT endpoint, role, verify_key_init, collector_hpke_config,
        report_expiry_age, tolerable_clock_skew
    FROM taskprov_peer_aggregators WHERE endpoint = $1 AND role = $2",
            )
            .await?;
        let peer_aggregator_row = self.query_opt(&stmt, params);

        let stmt = self
            .prepare_cached(
                "-- get_taskprov_peer_aggregator()
SELECT ord, type, token FROM taskprov_aggregator_auth_tokens
    WHERE peer_aggregator_id = (SELECT id FROM taskprov_peer_aggregators
        WHERE endpoint = $1 AND role = $2)
    ORDER BY ord ASC",
            )
            .await?;
        let aggregator_auth_token_rows = self.query(&stmt, params);

        let stmt = self
            .prepare_cached(
                "-- get_taskprov_peer_aggregator()
SELECT ord, type, token FROM taskprov_collector_auth_tokens
    WHERE peer_aggregator_id = (SELECT id FROM taskprov_peer_aggregators
        WHERE endpoint = $1 AND role = $2)
    ORDER BY ord ASC",
            )
            .await?;
        let collector_auth_token_rows = self.query(&stmt, params);

        let (peer_aggregator_row, aggregator_auth_token_rows, collector_auth_token_rows) = try_join!(
            peer_aggregator_row,
            aggregator_auth_token_rows,
            collector_auth_token_rows,
        )?;
        peer_aggregator_row
            .map(|peer_aggregator_row| {
                self.taskprov_peer_aggregator_from_rows(
                    &peer_aggregator_row,
                    &aggregator_auth_token_rows,
                    &collector_auth_token_rows,
                )
            })
            .transpose()
    }

    fn taskprov_peer_aggregator_from_rows(
        &self,
        peer_aggregator_row: &Row,
        aggregator_auth_token_rows: &[Row],
        collector_auth_token_rows: &[Row],
    ) -> Result<PeerAggregator, Error> {
        let endpoint = Url::parse(peer_aggregator_row.get::<_, &str>("endpoint"))?;
        let endpoint_bytes = endpoint.as_str().as_ref();
        let role: AggregatorRole = peer_aggregator_row.get("role");
        let report_expiry_age = peer_aggregator_row
            .get_nullable_bigint_and_convert("report_expiry_age")?
            .map(Duration::from_seconds);
        let tolerable_clock_skew = Duration::from_seconds(
            peer_aggregator_row.get_bigint_and_convert("tolerable_clock_skew")?,
        );
        let collector_hpke_config =
            HpkeConfig::get_decoded(peer_aggregator_row.get("collector_hpke_config"))?;

        let encrypted_verify_key_init: Vec<u8> = peer_aggregator_row.get("verify_key_init");
        let verify_key_init = self
            .crypter
            .decrypt(
                "taskprov_peer_aggregator",
                endpoint_bytes,
                "verify_key_init",
                &encrypted_verify_key_init,
            )?
            .as_slice()
            .try_into()?;

        let decrypt_tokens = |rows: &[Row], table| -> Result<Vec<_>, Error> {
            rows.iter()
                .map(|row| {
                    let ord: i64 = row.get("ord");
                    let auth_token_type: AuthenticationTokenType = row.get("type");
                    let encrypted_token: Vec<u8> = row.get("token");

                    let mut row_id = Vec::new();
                    row_id.extend_from_slice(endpoint_bytes);
                    row_id.extend_from_slice(&role.as_role().get_encoded()?);
                    row_id.extend_from_slice(&ord.to_be_bytes());

                    auth_token_type.as_authentication(&self.crypter.decrypt(
                        table,
                        &row_id,
                        "token",
                        &encrypted_token,
                    )?)
                })
                .collect()
        };

        let aggregator_auth_tokens = decrypt_tokens(
            aggregator_auth_token_rows,
            "taskprov_aggregator_auth_tokens",
        )?;
        let collector_auth_tokens =
            decrypt_tokens(collector_auth_token_rows, "taskprov_collector_auth_tokens")?;

        Ok(PeerAggregator::new(
            endpoint,
            role.as_role(),
            verify_key_init,
            collector_hpke_config,
            report_expiry_age,
            tolerable_clock_skew,
            aggregator_auth_tokens,
            collector_auth_tokens,
        ))
    }

    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn put_taskprov_peer_aggregator(
        &self,
        peer_aggregator: &PeerAggregator,
    ) -> Result<(), Error> {
        let endpoint = peer_aggregator.endpoint().as_str();
        let role = &AggregatorRole::from_role(*peer_aggregator.role())?;
        let encrypted_verify_key_init = self.crypter.encrypt(
            "taskprov_peer_aggregator",
            endpoint.as_ref(),
            "verify_key_init",
            peer_aggregator.verify_key_init().as_ref(),
        )?;

        let stmt = self
            .prepare_cached(
                "-- put_taskprov_peer_aggregator()
INSERT INTO taskprov_peer_aggregators (
    endpoint, role, verify_key_init, tolerable_clock_skew, report_expiry_age,
    collector_hpke_config, created_at, updated_by
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT DO NOTHING",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* endpoint */ &endpoint,
                    /* role */ role,
                    /* verify_key_init */ &encrypted_verify_key_init,
                    /* tolerable_clock_skew */
                    &i64::try_from(peer_aggregator.tolerable_clock_skew().as_seconds())?,
                    /* report_expiry_age */
                    &peer_aggregator
                        .report_expiry_age()
                        .map(Duration::as_seconds)
                        .map(i64::try_from)
                        .transpose()?,
                    /* collector_hpke_config */
                    &peer_aggregator.collector_hpke_config().get_encoded()?,
                    /* created_at */ &self.clock.now().as_naive_date_time()?,
                    /* updated_by */
                    &self.name,
                ],
            )
            .await?,
        )?;

        let encrypt_tokens = |tokens: &[AuthenticationToken], table| -> Result<_, Error> {
            let mut ords = Vec::new();
            let mut types = Vec::new();
            let mut encrypted_tokens = Vec::new();
            for (ord, token) in tokens.iter().enumerate() {
                let ord = i64::try_from(ord)?;

                let mut row_id = Vec::new();
                row_id.extend_from_slice(endpoint.as_ref());
                row_id.extend_from_slice(&role.as_role().get_encoded()?);
                row_id.extend_from_slice(&ord.to_be_bytes());

                let encrypted_auth_token =
                    self.crypter
                        .encrypt(table, &row_id, "token", token.as_ref())?;

                ords.push(ord);
                types.push(AuthenticationTokenType::from(token));
                encrypted_tokens.push(encrypted_auth_token);
            }
            Ok((ords, types, encrypted_tokens))
        };

        let (aggregator_auth_token_ords, aggregator_auth_token_types, aggregator_auth_tokens) =
            encrypt_tokens(
                peer_aggregator.aggregator_auth_tokens(),
                "taskprov_aggregator_auth_tokens",
            )?;
        let stmt = self
            .prepare_cached(
                "-- put_taskprov_peer_aggregator()
INSERT INTO taskprov_aggregator_auth_tokens (
    peer_aggregator_id, created_at, updated_by, ord, type, token
)
SELECT
    (SELECT id FROM taskprov_peer_aggregators WHERE endpoint = $1 AND role = $2),
    $3, $4, * FROM UNNEST($5::BIGINT[], $6::AUTH_TOKEN_TYPE[], $7::BYTEA[])",
            )
            .await?;
        let aggregator_auth_tokens_params: &[&(dyn ToSql + Sync)] = &[
            /* endpoint */ &endpoint,
            /* role */ role,
            /* created_at */ &self.clock.now().as_naive_date_time()?,
            /* updated_by */ &self.name,
            /* ords */ &aggregator_auth_token_ords,
            /* token_types */ &aggregator_auth_token_types,
            /* tokens */ &aggregator_auth_tokens,
        ];
        let aggregator_auth_tokens_future = self.execute(&stmt, aggregator_auth_tokens_params);

        let (collector_auth_token_ords, collector_auth_token_types, collector_auth_tokens) =
            encrypt_tokens(
                peer_aggregator.collector_auth_tokens(),
                "taskprov_collector_auth_tokens",
            )?;
        let stmt = self
            .prepare_cached(
                "-- put_taskprov_peer_aggregator()
INSERT INTO taskprov_collector_auth_tokens (
    peer_aggregator_id, created_at, updated_by, ord,  type, token
)
SELECT
    (SELECT id FROM taskprov_peer_aggregators WHERE endpoint = $1 AND role = $2),
    $3, $4, * FROM UNNEST($5::BIGINT[], $6::AUTH_TOKEN_TYPE[], $7::BYTEA[])",
            )
            .await?;
        let collector_auth_tokens_params: &[&(dyn ToSql + Sync)] = &[
            /* endpoint */ &endpoint,
            /* role */ role,
            /* created_at */ &self.clock.now().as_naive_date_time()?,
            /* updated_by */ &self.name,
            /* ords */ &collector_auth_token_ords,
            /* token_types */ &collector_auth_token_types,
            /* tokens */ &collector_auth_tokens,
        ];
        let collector_auth_tokens_future = self.execute(&stmt, collector_auth_tokens_params);

        try_join!(aggregator_auth_tokens_future, collector_auth_tokens_future)?;
        Ok(())
    }

    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn delete_taskprov_peer_aggregator(
        &self,
        aggregator_url: &Url,
        role: &Role,
    ) -> Result<(), Error> {
        let aggregator_url = aggregator_url.as_str();
        let role = AggregatorRole::from_role(*role)?;

        // Deletion of other data implemented via ON DELETE CASCADE.
        let stmt = self
            .prepare_cached(
                "-- delete_taskprov_peer_aggregator()
DELETE FROM taskprov_peer_aggregators WHERE endpoint = $1 AND role = $2",
            )
            .await?;
        check_single_row_mutation(self.execute(&stmt, &[&aggregator_url, &role]).await?)
    }

    /// Get the [`TaskUploadCounter`] for a task. This is aggregated across all shards. Returns
    /// `None` if the task doesn't exist.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn get_task_upload_counter(
        &self,
        task_id: &TaskId,
    ) -> Result<Option<TaskUploadCounter>, Error> {
        let stmt = self
            .prepare_cached(
                "-- get_task_upload_counter()
SELECT
    tasks.id,
    COALESCE(SUM(interval_collected)::BIGINT, 0) AS interval_collected,
    COALESCE(SUM(report_decode_failure)::BIGINT, 0) AS report_decode_failure,
    COALESCE(SUM(report_decrypt_failure)::BIGINT, 0) AS report_decrypt_failure,
    COALESCE(SUM(report_expired)::BIGINT, 0) AS report_expired,
    COALESCE(SUM(report_outdated_key)::BIGINT, 0) AS report_outdated_key,
    COALESCE(SUM(report_success)::BIGINT, 0) AS report_success,
    COALESCE(SUM(report_too_early)::BIGINT, 0) AS report_too_early,
    COALESCE(SUM(task_expired)::BIGINT, 0) AS task_expired
FROM task_upload_counters
RIGHT JOIN tasks on tasks.id = task_upload_counters.task_id
WHERE tasks.task_id = $1
GROUP BY tasks.id",
            )
            .await?;

        self.query_opt(&stmt, &[task_id.as_ref()])
            .await?
            .map(|row| {
                Ok(TaskUploadCounter {
                    interval_collected: row.get_bigint_and_convert("interval_collected")?,
                    report_decode_failure: row.get_bigint_and_convert("report_decode_failure")?,
                    report_decrypt_failure: row.get_bigint_and_convert("report_decrypt_failure")?,
                    report_expired: row.get_bigint_and_convert("report_expired")?,
                    report_outdated_key: row.get_bigint_and_convert("report_outdated_key")?,
                    report_success: row.get_bigint_and_convert("report_success")?,
                    report_too_early: row.get_bigint_and_convert("report_too_early")?,
                    task_expired: row.get_bigint_and_convert("task_expired")?,
                })
            })
            .transpose()
    }

    /// Add a `TaskUploadCounter` to the counter associated with the given [`TaskId`]. This is sharded,
    /// requiring an `ord` parameter to determine which shard to add to. `ord` should be randomly
    /// generated by the caller.
    #[tracing::instrument(skip(self), err(level = Level::DEBUG))]
    pub async fn increment_task_upload_counter(
        &self,
        task_id: &TaskId,
        ord: u64,
        counter: &TaskUploadCounter,
    ) -> Result<(), Error> {
        let stmt =
            "-- increment_task_upload_counter()
INSERT INTO task_upload_counters (task_id, ord, interval_collected, report_decode_failure,
        report_decrypt_failure, report_expired, report_outdated_key, report_success, report_too_early,
        task_expired)
VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (task_id, ord) DO UPDATE SET
    interval_collected = task_upload_counters.interval_collected + $3,
    report_decode_failure = task_upload_counters.report_decode_failure + $4,
    report_decrypt_failure = task_upload_counters.report_decrypt_failure + $5,
    report_expired = task_upload_counters.report_expired + $6,
    report_outdated_key = task_upload_counters.report_outdated_key + $7,
    report_success = task_upload_counters.report_success + $8,
    report_too_early = task_upload_counters.report_too_early + $9,
    task_expired = task_upload_counters.task_expired + $10";

        let stmt = self.prepare_cached(stmt).await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    task_id.as_ref(),
                    &i64::try_from(ord)?,
                    &i64::try_from(counter.interval_collected)?,
                    &i64::try_from(counter.report_decode_failure)?,
                    &i64::try_from(counter.report_decrypt_failure)?,
                    &i64::try_from(counter.report_expired)?,
                    &i64::try_from(counter.report_outdated_key)?,
                    &i64::try_from(counter.report_success)?,
                    &i64::try_from(counter.report_too_early)?,
                    &i64::try_from(counter.task_expired)?,
                ],
            )
            .await?,
        )
    }

    /// Retrieves the task aggregation counters for a given task. This result reflects the overall
    /// counter values, merged across all shards.
    pub async fn get_task_aggregation_counter(
        &self,
        task_id: &TaskId,
    ) -> Result<Option<TaskAggregationCounter>, Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Ok(None),
        };

        let stmt = self
            .prepare_cached(
                "-- get_task_aggregation_counter()
SELECT
    COALESCE(SUM(success)::BIGINT, 0) AS success
FROM task_aggregation_counters
WHERE task_id = $1",
            )
            .await?;

        self.query_opt(&stmt, &[/* task_id */ &task_info.pkey])
            .await?
            .map(|row| {
                Ok(TaskAggregationCounter {
                    success: row.get_bigint_and_convert("success")?,
                })
            })
            .transpose()
    }

    /// Increments the task aggregation counters for a given task by the values in the provided
    /// counters. The `ord` parameter determines which "shard" of the counters to increment;
    /// generally this value would be randomly-generated.
    pub async fn increment_task_aggregation_counter(
        &self,
        task_id: &TaskId,
        ord: u64,
        counter: &TaskAggregationCounter,
    ) -> Result<(), Error> {
        let task_info = match self.task_info_for(task_id).await? {
            Some(task_info) => task_info,
            None => return Err(Error::MutationTargetNotFound),
        };

        let stmt = self
            .prepare_cached(
                "-- increment_task_aggregation_counter()
INSERT INTO task_aggregation_counters (task_id, ord, success)
VALUES ($1, $2, $3)
ON CONFLICT (task_id, ord) DO UPDATE SET
    success = task_aggregation_counters.success + $3",
            )
            .await?;

        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_info.pkey,
                    /* ord */ &i64::try_from(ord)?,
                    /* success */ &i64::try_from(counter.success)?,
                ],
            )
            .await?,
        )
    }

    /// Helper function to look up (cached) information about a given task. The cache is retained
    /// indefinitely. It is assumed that the parameters stored in a [`TaskInfo`] are never changed
    /// so this should be fine.
    async fn task_info_for(&self, task_id: &TaskId) -> Result<Option<TaskInfo>, Error> {
        // We fetch the task's primary key & task-level information in a separate query. This will
        // allow the query planner to make more accurate row count estimates, by comparing concrete
        // values to the statistics of the tables involved in the main query, and help the query
        // planner reason that there will be only one task_id, so it can satisfy e.g. ordering
        // clauses using a reverse index scan without an intermediate sort.

        // Fast path: the task info is already populated in the cache.
        {
            // unwrap safety: mutex poisoning
            let task_infos = self.task_infos.lock().unwrap();
            if let Some(task_info) = task_infos.get(task_id) {
                return Ok(Some(task_info.clone()));
            }
        }

        // Slow path: the task info was not populated in the cache. Read the task info from the
        // database, populate it into the cache, and return it.
        let stmt = self
            .prepare_cached(
                "-- task_info_for()
SELECT id, report_expiry_age FROM tasks WHERE tasks.task_id = $1",
            )
            .await?;
        let mut rows = self
            .query(&stmt, &[/* task_id */ &task_id.get_encoded()?])
            .await?
            .into_iter();
        let row = if let Some(row) = rows.next() {
            row
        } else {
            // We don't cache that a task does not exist, as there may be a race between propagating
            // a newly-created task & attempting to use it.
            return Ok(None);
        };
        if rows.next().is_some() {
            panic!("task_info_for: found multiple tasks with same ID");
        }

        let pkey = row.get::<_, i64>("id");
        let report_expiry_age = row
            .get::<_, Option<i64>>("report_expiry_age")
            .map(|age| {
                chrono::Duration::try_seconds(age)
                    .ok_or_else(|| Error::TimeOverflow("overflow computing report expiry age"))
            })
            .transpose()?;
        let task_info = TaskInfo {
            pkey,
            report_expiry_age,
        };

        // unwrap safety: mutex poisoning
        let mut task_infos = self.task_infos.lock().unwrap();
        task_infos.insert(*task_id, task_info.clone());
        Ok(Some(task_info))
    }
}

/// Represents cached information about a task.
#[derive(Clone)]
struct TaskInfo {
    /// The task's artificial primary key, corresponding to the tasks.id column.
    pkey: i64,

    /// The task's report expiry age, corresponding to the tasks.report_expiry_age column.
    report_expiry_age: Option<chrono::Duration>,
}

impl TaskInfo {
    /// Computes the report expiry threshold, i.e. the minimum timestamp at which reports will not
    /// be GC'ed, given the current timestamp.
    fn report_expiry_threshold(
        &self,
        now: &NaiveDateTime,
    ) -> Result<Timestamp<NaiveDateTime>, Error> {
        match self.report_expiry_age {
            Some(report_expiry_age) => {
                let report_expiry_threshold =
                    now.checked_sub_signed(report_expiry_age).ok_or_else(|| {
                        Error::TimeOverflow("overflow computing report expiry threshold")
                    })?;
                Ok(Timestamp::Value(report_expiry_threshold))
            }
            None => Ok(Timestamp::NegInfinity),
        }
    }
}

fn check_insert(row_count: u64) -> Result<(), Error> {
    match row_count {
        0 => Err(Error::MutationTargetAlreadyExists),
        1 => Ok(()),
        _ => panic!(
            "insert which should have affected at most one row instead affected {row_count} rows"
        ),
    }
}

fn check_single_row_mutation(row_count: u64) -> Result<(), Error> {
    match row_count {
        0 => Err(Error::MutationTargetNotFound),
        1 => Ok(()),
        _ => panic!(
            "update which should have affected at most one row instead affected {row_count} rows"
        ),
    }
}

/// Add a [`std::time::Duration`] to a [`chrono::NaiveDateTime`].
fn add_naive_date_time_duration(
    time: &NaiveDateTime,
    duration: &StdDuration,
) -> Result<NaiveDateTime, Error> {
    time.checked_add_signed(
        chrono::Duration::from_std(*duration)
            .map_err(|_| Error::TimeOverflow("overflow converting duration to signed duration"))?,
    )
    .ok_or(Error::TimeOverflow("overflow adding duration to time"))
}

/// Extensions for [`tokio_postgres::row::Row`]
trait RowExt {
    /// Get an integer of type `P` from the row, then attempt to convert it to the desired integer
    /// type `T`.
    fn get_postgres_integer_and_convert<'a, P, I, T>(&'a self, idx: I) -> Result<T, Error>
    where
        P: FromSql<'a>,
        I: RowIndex + Display,
        T: TryFrom<P, Error = std::num::TryFromIntError>;

    /// Get a PostgreSQL `BIGINT` from the row, which is represented in Rust as
    /// i64 ([1]), then attempt to convert it to the desired integer type `T`.
    ///
    /// [1]: https://docs.rs/postgres-types/latest/postgres_types/trait.FromSql.html
    fn get_bigint_and_convert<I, T>(&self, idx: I) -> Result<T, Error>
    where
        I: RowIndex + Display,
        T: TryFrom<i64, Error = std::num::TryFromIntError>,
    {
        self.get_postgres_integer_and_convert::<i64, I, T>(idx)
    }

    /// Like [`Self::get_bigint_and_convert`] but handles nullable columns.
    fn get_nullable_bigint_and_convert<I, T>(&self, idx: I) -> Result<Option<T>, Error>
    where
        I: RowIndex + Display,
        T: TryFrom<i64, Error = std::num::TryFromIntError>;

    /// Get a PostgreSQL `BYTEA` from the row and attempt to convert it to `T`.
    fn get_bytea_and_convert<T>(&self, idx: &'static str) -> Result<T, Error>
    where
        for<'a> T: TryFrom<&'a [u8]>,
        for<'a> <T as TryFrom<&'a [u8]>>::Error: Debug;

    /// Get a PostgreSQL `BYTEA` from the row and attempt to decode a `T` value from it.
    fn get_bytea_and_decode<T, P>(
        &self,
        idx: &'static str,
        decoding_parameter: &P,
    ) -> Result<T, Error>
    where
        T: ParameterizedDecode<P>;
}

impl RowExt for Row {
    fn get_postgres_integer_and_convert<'a, P, I, T>(&'a self, idx: I) -> Result<T, Error>
    where
        P: FromSql<'a>,
        I: RowIndex + Display,
        T: TryFrom<P, Error = std::num::TryFromIntError>,
    {
        let postgres_integer: P = self.get(idx);
        Ok(T::try_from(postgres_integer)?)
    }

    fn get_nullable_bigint_and_convert<I, T>(&self, idx: I) -> Result<Option<T>, Error>
    where
        I: RowIndex + Display,
        T: TryFrom<i64, Error = std::num::TryFromIntError>,
    {
        let bigint: Option<i64> = self.get(idx);
        Ok(bigint.map(|bigint| T::try_from(bigint)).transpose()?)
    }

    fn get_bytea_and_convert<T>(&self, idx: &'static str) -> Result<T, Error>
    where
        for<'a> T: TryFrom<&'a [u8]>,
        for<'a> <T as TryFrom<&'a [u8]>>::Error: Debug,
    {
        let encoded: Vec<u8> = self.get(idx);
        T::try_from(&encoded)
            .map_err(|err| Error::DbState(format!("{idx} stored in database is invalid: {err:?}")))
    }

    fn get_bytea_and_decode<T, P>(
        &self,
        idx: &'static str,
        decoding_parameter: &P,
    ) -> Result<T, Error>
    where
        T: ParameterizedDecode<P>,
    {
        let encoded: Vec<u8> = self.get(idx);
        T::get_decoded_with_param(decoding_parameter, &encoded)
            .map_err(|err| Error::DbState(format!("{idx} stored in database is invalid: {err:?}")))
    }
}

/// A Crypter allows a Datastore to encrypt/decrypt sensitive values stored to the datastore. Values
/// are cryptographically bound to the specific location in the datastore in which they are stored.
/// Rollback protection is not provided.
pub struct Crypter {
    keys: Vec<LessSafeKey>,
}

impl Crypter {
    // The internal serialized format of a Crypter encrypted value is:
    //   ciphertext || tag || nonce
    // (the `ciphertext || tag` portion is as returned from `seal_in_place_append_tag`)

    /// Creates a new Crypter instance, using the given set of keys. The first key in the provided
    /// vector is considered to be the "primary" key, used for encryption operations; any of the
    /// provided keys can be used for decryption operations.
    ///
    /// The keys must be for the AES-128-GCM algorithm.
    pub fn new(keys: Vec<LessSafeKey>) -> Self {
        assert!(!keys.is_empty());
        for key in &keys {
            assert_eq!(key.algorithm(), &AES_128_GCM);
        }
        Self { keys }
    }

    fn encrypt(
        &self,
        table: &str,
        row: &[u8],
        column: &str,
        value: &[u8],
    ) -> Result<Vec<u8>, Error> {
        // It is safe to unwrap the key because we have already validated that keys is nonempty
        // in Crypter::new.
        Self::encrypt_with_key(self.keys.first().unwrap(), table, row, column, value)
    }

    fn encrypt_with_key(
        key: &LessSafeKey,
        table: &str,
        row: &[u8],
        column: &str,
        value: &[u8],
    ) -> Result<Vec<u8>, Error> {
        // Generate a random nonce, compute AAD.
        let nonce_bytes: [u8; aead::NONCE_LEN] = random();
        let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
        let aad = aead::Aad::from(Self::aad_bytes_for(table, row, column)?);

        // Encrypt, append nonce.
        let mut result = value.to_vec();
        key.seal_in_place_append_tag(nonce, aad, &mut result)?;
        result.extend(nonce_bytes);
        Ok(result)
    }

    fn decrypt(
        &self,
        table: &str,
        row: &[u8],
        column: &str,
        value: &[u8],
    ) -> Result<Vec<u8>, Error> {
        if value.len() < aead::NONCE_LEN {
            return Err(Error::Crypt);
        }

        // TODO(https://github.com/rust-lang/rust/issues/90091): use `rsplit_array_ref` once it is stabilized.
        let (ciphertext_and_tag, nonce_bytes) = value.split_at(value.len() - aead::NONCE_LEN);
        let nonce_bytes: [u8; aead::NONCE_LEN] = nonce_bytes.try_into().unwrap();
        let aad_bytes = Self::aad_bytes_for(table, row, column)?;

        for key in &self.keys {
            let mut ciphertext_and_tag = ciphertext_and_tag.to_vec();
            if let Ok(plaintext) = key.open_in_place(
                aead::Nonce::assume_unique_for_key(nonce_bytes),
                aead::Aad::from(aad_bytes.clone()),
                &mut ciphertext_and_tag,
            ) {
                let len = plaintext.len();
                ciphertext_and_tag.truncate(len);
                return Ok(ciphertext_and_tag);
            }
        }
        Err(Error::Crypt)
    }

    fn aad_bytes_for(table: &str, row: &[u8], column: &str) -> Result<Vec<u8>, Error> {
        // AAD computation is based on (table, row, column).
        // The serialized AAD is:
        //   (length of table) || table || (length of row) || row || (length of column) || column.
        // Lengths are expressed as 8-byte unsigned integers.

        let aad_length = 3 * size_of::<u64>() + table.len() + row.len() + column.len();
        let mut aad_bytes = Vec::with_capacity(aad_length);
        aad_bytes.extend_from_slice(&u64::try_from(table.len())?.to_be_bytes());
        aad_bytes.extend_from_slice(table.as_ref());
        aad_bytes.extend_from_slice(&u64::try_from(row.len())?.to_be_bytes());
        aad_bytes.extend_from_slice(row);
        aad_bytes.extend_from_slice(&u64::try_from(column.len())?.to_be_bytes());
        aad_bytes.extend_from_slice(column.as_ref());
        assert_eq!(aad_bytes.len(), aad_length);

        Ok(aad_bytes)
    }
}

/// Error represents a datastore-level error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error from the underlying database library.
    #[error("DB error: {0}")]
    Db(#[from] tokio_postgres::Error),
    #[error("DB pool error: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),
    #[error("crypter error")]
    Crypt,
    /// An attempt was made to mutate an entity that does not exist.
    #[error("not found in datastore")]
    MutationTargetNotFound,
    /// An attempt was made to insert an entity that already exists.
    #[error("already in datastore")]
    MutationTargetAlreadyExists,
    /// The database was in an unexpected state.
    #[error("inconsistent database state: {0}")]
    DbState(String),
    /// An error from decoding a value stored encoded in the underlying database.
    #[error("decoding error: {0}")]
    Decode(#[from] CodecError),
    #[error("base64 decoding error: {0}")]
    Base64(#[from] base64::DecodeError),
    /// An arbitrary error returned from the user callback; unrelated to DB internals. This error
    /// will never be generated by the datastore library itself.
    #[error(transparent)]
    User(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("URL parse error: {0}")]
    Url(#[from] url::ParseError),
    #[error("invalid task parameters: {0}")]
    Task(#[from] task::Error),
    #[error("integer conversion failed: {0}")]
    TryFromInt(#[from] std::num::TryFromIntError),
    #[error(transparent)]
    Message(#[from] janus_messages::Error),
    /// An invalid parameter was provided.
    #[error("invalid parameter: {0}")]
    InvalidParameter(&'static str),
    /// An error occurred while manipulating timestamps or durations.
    #[error("{0}")]
    TimeOverflow(&'static str),
    /// The requested operation could not be completed
    #[error("batch already collected")]
    AlreadyCollected,
    /// The requested operation could not be completed because the relevant data has already been
    /// scrubbed from the system.
    #[error("already scrubbed")]
    Scrubbed,
    /// The transaction was aborted because it retried too many times.
    #[error("too many retries")]
    TooManyRetries { source: Option<Box<Error>> },
}

impl From<ring::error::Unspecified> for Error {
    fn from(_: ring::error::Unspecified) -> Self {
        Error::Crypt
    }
}
