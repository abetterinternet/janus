//! Janus datastore (durable storage) implementation.

use self::models::{
    AcquiredAggregationJob, AcquiredCollectionJob, AggregateShareJob, AggregationJob,
    AggregatorRole, AuthenticationTokenType, Batch, BatchAggregation, CollectionJob,
    CollectionJobState, CollectionJobStateCode, LeaderStoredReport, Lease, LeaseToken,
    OutstandingBatch, ReportAggregation, ReportAggregationState, ReportAggregationStateCode,
    SqlInterval,
};
use crate::{
    query_type::{AccumulableQueryType, CollectableQueryType},
    task::{self, Task},
    SecretBytes,
};
use anyhow::anyhow;
use chrono::NaiveDateTime;
use futures::future::try_join_all;
use janus_core::{
    hpke::{HpkeKeypair, HpkePrivateKey},
    task::VdafInstance,
    time::{Clock, TimeExt},
};
use janus_messages::{
    query_type::QueryType, AggregationJobId, BatchId, CollectionJobId, Duration, Extension,
    HpkeCiphertext, HpkeConfig, Interval, PrepareStep, ReportId, ReportIdChecksum, ReportMetadata,
    ReportShare, Role, TaskId, Time,
};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter},
    Context, KeyValue,
};
use postgres_types::{FromSql, Json, ToSql};
use prio::{
    codec::{decode_u16_items, encode_u16_items, CodecError, Decode, Encode, ParameterizedDecode},
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
use tracing::error;
use url::Url;

#[cfg(feature = "test-util")]
#[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
pub mod test_util;

// TODO(#196): retry network-related & other transient failures once we know what they look like

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
supported_schema_versions!(11);

/// Datastore represents a datastore for Janus, with support for transactional reads and writes.
/// In practice, Datastore instances are currently backed by a PostgreSQL database.
pub struct Datastore<C: Clock> {
    pool: deadpool_postgres::Pool,
    crypter: Crypter,
    clock: C,
    transaction_status_counter: Counter<u64>,
    rollback_error_counter: Counter<u64>,
    transaction_duration_histogram: Histogram<f64>,
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
    ) -> Result<Datastore<C>, Error> {
        Self::new_with_supported_versions(pool, crypter, clock, meter, SUPPORTED_SCHEMA_VERSIONS)
            .await
    }

    async fn new_with_supported_versions(
        pool: deadpool_postgres::Pool,
        crypter: Crypter,
        clock: C,
        meter: &Meter,
        supported_schema_versions: &[i64],
    ) -> Result<Datastore<C>, Error> {
        let datastore = Self::new_without_supported_versions(pool, crypter, clock, meter).await;

        let (current_version, migration_description) = datastore
            .run_tx_with_name("check schema version", |tx| {
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
    ) -> Datastore<C> {
        let transaction_status_counter = meter
            .u64_counter("janus_database_transactions")
            .with_description("Count of database transactions run, with their status.")
            .init();
        let rollback_error_counter = meter
            .u64_counter("janus_database_rollback_errors")
            .with_description(concat!(
                "Count of errors received when rolling back a database transaction, ",
                "with their PostgreSQL error code.",
            ))
            .init();
        let transaction_duration_histogram = meter
            .f64_histogram("janus_database_transaction_duration_seconds")
            .with_description("Duration of database transactions.")
            .init();

        Self {
            pool,
            crypter,
            clock,
            transaction_status_counter,
            rollback_error_counter,
            transaction_duration_histogram,
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
    pub fn run_tx<'s, F, T>(&'s self, f: F) -> impl Future<Output = Result<T, Error>> + 's
    where
        F: 's,
        T: 's,
        for<'a> F:
            Fn(&'a Transaction<C>) -> Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>,
    {
        self.run_tx_with_name("default", f)
    }

    /// See [`Datastore::run_tx`]. This method additionally allows specifying a name for the
    /// transaction, for use in database-related metrics.
    #[tracing::instrument(level = "trace", skip(self, f))]
    pub async fn run_tx_with_name<F, T>(&self, name: &'static str, f: F) -> Result<T, Error>
    where
        for<'a> F:
            Fn(&'a Transaction<C>) -> Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>,
    {
        loop {
            let before = Instant::now();
            let (rslt, retry) = self.run_tx_once(&f).await;
            let elapsed = before.elapsed();
            self.transaction_duration_histogram.record(
                &Context::current(),
                elapsed.as_secs_f64(),
                &[KeyValue::new("tx", name)],
            );
            let status = match (rslt.as_ref(), retry) {
                (_, true) => "retry",
                (Ok(_), _) => "success",
                (Err(Error::Db(_)), _) | (Err(Error::Pool(_)), _) => "error_db",
                (Err(_), _) => "error_other",
            };
            self.transaction_status_counter.add(
                &Context::current(),
                1,
                &[KeyValue::new("status", status), KeyValue::new("tx", name)],
            );
            if retry {
                continue;
            }
            return rslt;
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    async fn run_tx_once<F, T>(&self, f: &F) -> (Result<T, Error>, bool)
    where
        for<'a> F:
            Fn(&'a Transaction<C>) -> Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>,
    {
        // Open transaction.
        let mut client = match self.pool.get().await {
            Ok(client) => client,
            Err(err) => return (Err(err.into()), false),
        };
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
                        &Context::current(),
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
        (rslt, retry.load(Ordering::Relaxed))
    }

    /// Write a task into the datastore.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    pub async fn put_task(&self, task: &Task) -> Result<(), Error> {
        self.run_tx(|tx| {
            let task = task.clone();
            Box::pin(async move { tx.put_task(&task).await })
        })
        .await
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

/// Transaction represents an ongoing datastore transaction.
pub struct Transaction<'a, C: Clock> {
    raw_tx: deadpool_postgres::Transaction<'a>,
    crypter: &'a Crypter,
    clock: &'a C,

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

    /// Calling this method will force the transaction to eventually be rolled back and retried; all
    /// datastore writes in this try will be lost. Calling this method does not interrupt or
    /// otherwise directly affect the transaction-processing callback; the caller may wish to e.g.
    /// use an error or some other signalling method to cause the callback to terminate.
    ///
    /// There is no upper limit on the number of retries a single transaction may incur.
    pub fn retry(&self) {
        self.retry.store(true, Ordering::Relaxed);
    }

    /// Returns the current schema version of the datastore and the description of the migration
    /// script that applied it.
    async fn get_current_schema_migration_version(&self) -> Result<(i64, String), Error> {
        let stmt = self
            .prepare_cached(
                "SELECT version, description FROM _sqlx_migrations
                WHERE success = TRUE ORDER BY version DESC LIMIT(1)",
            )
            .await?;
        let row = self.query_one(&stmt, &[]).await?;

        let version = row.try_get("version")?;
        let description = row.try_get("description")?;

        Ok((version, description))
    }

    /// Writes a task into the datastore.
    #[tracing::instrument(skip(self, task), fields(task_id = ?task.id()), err)]
    pub async fn put_task(&self, task: &Task) -> Result<(), Error> {
        let endpoints: Vec<_> = task
            .aggregator_endpoints()
            .iter()
            .map(Url::as_str)
            .collect();

        // Main task insert.
        let stmt = self
            .prepare_cached(
                "INSERT INTO tasks (
                    task_id, aggregator_role, aggregator_endpoints, query_type, vdaf,
                    max_batch_query_count, task_expiration, report_expiry_age, min_batch_size,
                    time_precision, tolerable_clock_skew, collector_hpke_config)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &task.id().as_ref(),
                /* aggregator_role */ &AggregatorRole::from_role(*task.role())?,
                /* aggregator_endpoints */ &endpoints,
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
                /* collector_hpke_config */ &task.collector_hpke_config().get_encoded(),
            ],
        )
        .await?;

        // Aggregator auth tokens.
        let mut aggregator_auth_token_ords = Vec::new();
        let mut aggregator_auth_token_types = Vec::new();
        let mut aggregator_auth_tokens = Vec::new();
        for (ord, token) in task.aggregator_auth_tokens().iter().enumerate() {
            let ord = i64::try_from(ord)?;

            let mut row_id = [0; TaskId::LEN + size_of::<i64>()];
            row_id[..TaskId::LEN].copy_from_slice(task.id().as_ref());
            row_id[TaskId::LEN..].copy_from_slice(&ord.to_be_bytes());

            let encrypted_aggregator_auth_token = self.crypter.encrypt(
                "task_aggregator_auth_tokens",
                &row_id,
                "token",
                token.as_ref(),
            )?;

            aggregator_auth_token_ords.push(ord);
            aggregator_auth_token_types.push(AuthenticationTokenType::from(token));
            aggregator_auth_tokens.push(encrypted_aggregator_auth_token);
        }
        let stmt = self
            .prepare_cached(
                "INSERT INTO task_aggregator_auth_tokens (task_id, ord, type, token)
                SELECT
                    (SELECT id FROM tasks WHERE task_id = $1),
                    * FROM UNNEST($2::BIGINT[], $3::AUTH_TOKEN_TYPE[], $4::BYTEA[])",
            )
            .await?;
        let aggregator_auth_tokens_params: &[&(dyn ToSql + Sync)] = &[
            /* task_id */ &task.id().as_ref(),
            /* ords */ &aggregator_auth_token_ords,
            /* token_types */ &aggregator_auth_token_types,
            /* tokens */ &aggregator_auth_tokens,
        ];
        let aggregator_auth_tokens_future = self.execute(&stmt, aggregator_auth_tokens_params);

        // Collector auth tokens.
        let mut collector_auth_token_ords = Vec::new();
        let mut collector_auth_token_types = Vec::new();
        let mut collector_auth_tokens = Vec::new();
        for (ord, token) in task.collector_auth_tokens().iter().enumerate() {
            let ord = i64::try_from(ord)?;

            let mut row_id = [0; TaskId::LEN + size_of::<i64>()];
            row_id[..TaskId::LEN].copy_from_slice(task.id().as_ref());
            row_id[TaskId::LEN..].copy_from_slice(&ord.to_be_bytes());

            let encrypted_collector_auth_token = self.crypter.encrypt(
                "task_collector_auth_tokens",
                &row_id,
                "token",
                token.as_ref(),
            )?;

            collector_auth_token_ords.push(ord);
            collector_auth_token_types.push(AuthenticationTokenType::from(token));
            collector_auth_tokens.push(encrypted_collector_auth_token);
        }
        let stmt = self
            .prepare_cached(
                "INSERT INTO task_collector_auth_tokens (task_id, ord, type, token)
                SELECT
                    (SELECT id FROM tasks WHERE task_id = $1),
                    * FROM UNNEST($2::BIGINT[], $3::AUTH_TOKEN_TYPE[], $4::BYTEA[])",
            )
            .await?;
        let collector_auth_tokens_params: &[&(dyn ToSql + Sync)] = &[
            /* task_id */ &task.id().as_ref(),
            /* ords */ &collector_auth_token_ords,
            /* token_types */ &collector_auth_token_types,
            /* tokens */ &collector_auth_tokens,
        ];
        let collector_auth_tokens_future = self.execute(&stmt, collector_auth_tokens_params);

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
            hpke_configs.push(hpke_keypair.config().get_encoded());
            hpke_private_keys.push(encrypted_hpke_private_key);
        }
        let stmt = self
            .prepare_cached(
                "INSERT INTO task_hpke_keys (task_id, config_id, config, private_key)
                SELECT
                    (SELECT id FROM tasks WHERE task_id = $1),
                    * FROM UNNEST($2::SMALLINT[], $3::BYTEA[], $4::BYTEA[])",
            )
            .await?;
        let hpke_configs_params: &[&(dyn ToSql + Sync)] = &[
            /* task_id */ &task.id().as_ref(),
            /* config_id */ &hpke_config_ids,
            /* configs */ &hpke_configs,
            /* private_keys */ &hpke_private_keys,
        ];
        let hpke_configs_future = self.execute(&stmt, hpke_configs_params);

        // VDAF verification keys.
        let mut vdaf_verify_keys: Vec<Vec<u8>> = Vec::new();
        for vdaf_verify_key in task.vdaf_verify_keys() {
            let encrypted_vdaf_verify_key = self.crypter.encrypt(
                "task_vdaf_verify_keys",
                task.id().as_ref(),
                "vdaf_verify_key",
                vdaf_verify_key.as_ref(),
            )?;
            vdaf_verify_keys.push(encrypted_vdaf_verify_key);
        }
        let stmt = self
            .prepare_cached(
                "INSERT INTO task_vdaf_verify_keys (task_id, vdaf_verify_key)
                SELECT (SELECT id FROM tasks WHERE task_id = $1), * FROM UNNEST($2::BYTEA[])",
            )
            .await?;
        let vdaf_verify_keys_params: &[&(dyn ToSql + Sync)] = &[
            /* task_id */ &task.id().as_ref(),
            /* vdaf_verify_keys */ &vdaf_verify_keys,
        ];
        let vdaf_verify_keys_future = self.execute(&stmt, vdaf_verify_keys_params);

        try_join!(
            aggregator_auth_tokens_future,
            collector_auth_tokens_future,
            hpke_configs_future,
            vdaf_verify_keys_future
        )?;

        Ok(())
    }

    /// Deletes a task from the datastore, along with all related data (client reports,
    /// aggregations, etc).
    #[tracing::instrument(skip(self))]
    pub async fn delete_task(&self, task_id: &TaskId) -> Result<(), Error> {
        // Deletion of other data implemented via ON DELETE CASCADE.
        let stmt = self
            .prepare_cached("DELETE FROM tasks WHERE task_id = $1")
            .await?;
        check_single_row_mutation(
            self.execute(&stmt, &[/* task_id */ &task_id.as_ref()])
                .await?,
        )?;
        Ok(())
    }

    /// Fetch the task parameters corresponing to the provided `task_id`.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_task(&self, task_id: &TaskId) -> Result<Option<Task>, Error> {
        let params: &[&(dyn ToSql + Sync)] = &[&task_id.as_ref()];
        let stmt = self
            .prepare_cached(
                "SELECT aggregator_role, aggregator_endpoints, query_type, vdaf,
                    max_batch_query_count, task_expiration, report_expiry_age, min_batch_size,
                    time_precision, tolerable_clock_skew, collector_hpke_config
                FROM tasks WHERE task_id = $1",
            )
            .await?;
        let task_row = self.query_opt(&stmt, params);

        let stmt = self
            .prepare_cached(
                "SELECT ord, type, token FROM task_aggregator_auth_tokens
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1) ORDER BY ord ASC",
            )
            .await?;
        let aggregator_auth_token_rows = self.query(&stmt, params);

        let stmt = self
            .prepare_cached(
                "SELECT ord, type, token FROM task_collector_auth_tokens
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1) ORDER BY ord ASC",
            )
            .await?;
        let collector_auth_token_rows = self.query(&stmt, params);

        let stmt = self
            .prepare_cached(
                "SELECT config_id, config, private_key FROM task_hpke_keys
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)",
            )
            .await?;
        let hpke_key_rows = self.query(&stmt, params);

        let stmt = self
            .prepare_cached(
                "SELECT vdaf_verify_key FROM task_vdaf_verify_keys
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)",
            )
            .await?;
        let vdaf_verify_key_rows = self.query(&stmt, params);

        let (
            task_row,
            aggregator_auth_token_rows,
            collector_auth_token_rows,
            hpke_key_rows,
            vdaf_verify_key_rows,
        ) = try_join!(
            task_row,
            aggregator_auth_token_rows,
            collector_auth_token_rows,
            hpke_key_rows,
            vdaf_verify_key_rows,
        )?;
        task_row
            .map(|task_row| {
                self.task_from_rows(
                    task_id,
                    &task_row,
                    &aggregator_auth_token_rows,
                    &collector_auth_token_rows,
                    &hpke_key_rows,
                    &vdaf_verify_key_rows,
                )
            })
            .transpose()
    }

    /// Fetch all the tasks in the database.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_tasks(&self) -> Result<Vec<Task>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT task_id, aggregator_role, aggregator_endpoints, query_type, vdaf,
                    max_batch_query_count, task_expiration, report_expiry_age, min_batch_size,
                    time_precision, tolerable_clock_skew, collector_hpke_config
                FROM tasks",
            )
            .await?;
        let task_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "SELECT (SELECT tasks.task_id FROM tasks
                    WHERE tasks.id = task_aggregator_auth_tokens.task_id),
                ord, type, token FROM task_aggregator_auth_tokens ORDER BY ord ASC",
            )
            .await?;
        let aggregator_auth_token_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "SELECT (SELECT tasks.task_id FROM tasks
                    WHERE tasks.id = task_collector_auth_tokens.task_id),
                ord, type, token FROM task_collector_auth_tokens ORDER BY ord ASC",
            )
            .await?;
        let collector_auth_token_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "SELECT (SELECT tasks.task_id FROM tasks WHERE tasks.id = task_hpke_keys.task_id),
                config_id, config, private_key FROM task_hpke_keys",
            )
            .await?;
        let hpke_config_rows = self.query(&stmt, &[]);

        let stmt = self
            .prepare_cached(
                "SELECT (SELECT tasks.task_id FROM tasks
                    WHERE tasks.id = task_vdaf_verify_keys.task_id),
                vdaf_verify_key FROM task_vdaf_verify_keys",
            )
            .await?;
        let vdaf_verify_key_rows = self.query(&stmt, &[]);

        let (
            task_rows,
            aggregator_auth_token_rows,
            collector_auth_token_rows,
            hpke_config_rows,
            vdaf_verify_key_rows,
        ) = try_join!(
            task_rows,
            aggregator_auth_token_rows,
            collector_auth_token_rows,
            hpke_config_rows,
            vdaf_verify_key_rows
        )?;

        let mut task_row_by_id = Vec::new();
        for row in task_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            task_row_by_id.push((task_id, row));
        }

        let mut aggregator_auth_token_rows_by_task_id: HashMap<TaskId, Vec<Row>> = HashMap::new();
        for row in aggregator_auth_token_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            aggregator_auth_token_rows_by_task_id
                .entry(task_id)
                .or_default()
                .push(row);
        }

        let mut collector_auth_token_rows_by_task_id: HashMap<TaskId, Vec<Row>> = HashMap::new();
        for row in collector_auth_token_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            collector_auth_token_rows_by_task_id
                .entry(task_id)
                .or_default()
                .push(row);
        }

        let mut hpke_config_rows_by_task_id: HashMap<TaskId, Vec<Row>> = HashMap::new();
        for row in hpke_config_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            hpke_config_rows_by_task_id
                .entry(task_id)
                .or_default()
                .push(row);
        }

        let mut vdaf_verify_key_rows_by_task_id: HashMap<TaskId, Vec<Row>> = HashMap::new();
        for row in vdaf_verify_key_rows {
            let task_id = TaskId::get_decoded(row.get("task_id"))?;
            vdaf_verify_key_rows_by_task_id
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
                    &aggregator_auth_token_rows_by_task_id
                        .remove(&task_id)
                        .unwrap_or_default(),
                    &collector_auth_token_rows_by_task_id
                        .remove(&task_id)
                        .unwrap_or_default(),
                    &hpke_config_rows_by_task_id
                        .remove(&task_id)
                        .unwrap_or_default(),
                    &vdaf_verify_key_rows_by_task_id
                        .remove(&task_id)
                        .unwrap_or_default(),
                )
            })
            .collect::<Result<_, _>>()
    }

    /// Construct a [`Task`] from the contents of the provided (tasks) `Row`,
    /// `hpke_aggregator_auth_tokens` rows, and `task_hpke_keys` rows.
    ///
    /// agg_auth_token_rows must be sorted in ascending order by `ord`.
    fn task_from_rows(
        &self,
        task_id: &TaskId,
        row: &Row,
        aggregator_auth_token_rows: &[Row],
        collector_auth_token_rows: &[Row],
        hpke_key_rows: &[Row],
        vdaf_verify_key_rows: &[Row],
    ) -> Result<Task, Error> {
        // Scalar task parameters.
        let aggregator_role: AggregatorRole = row.get("aggregator_role");
        let endpoints = row
            .get::<_, Vec<String>>("aggregator_endpoints")
            .into_iter()
            .map(|endpoint| Ok(Url::parse(&endpoint)?))
            .collect::<Result<_, Error>>()?;
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
        let collector_hpke_config = HpkeConfig::get_decoded(row.get("collector_hpke_config"))?;

        // Aggregator authentication tokens.
        let mut aggregator_auth_tokens = Vec::new();
        for row in aggregator_auth_token_rows {
            let ord: i64 = row.get("ord");
            let auth_token_type: AuthenticationTokenType = row.get("type");
            let encrypted_aggregator_auth_token: Vec<u8> = row.get("token");

            let mut row_id = [0u8; TaskId::LEN + size_of::<i64>()];
            row_id[..TaskId::LEN].copy_from_slice(task_id.as_ref());
            row_id[TaskId::LEN..].copy_from_slice(&ord.to_be_bytes());

            aggregator_auth_tokens.push(auth_token_type.as_authentication(
                &self.crypter.decrypt(
                    "task_aggregator_auth_tokens",
                    &row_id,
                    "token",
                    &encrypted_aggregator_auth_token,
                )?,
            )?);
        }

        // Collector authentication tokens.
        let mut collector_auth_tokens = Vec::new();
        for row in collector_auth_token_rows {
            let ord: i64 = row.get("ord");
            let auth_token_type: AuthenticationTokenType = row.get("type");
            let encrypted_collector_auth_token: Vec<u8> = row.get("token");

            let mut row_id = [0u8; TaskId::LEN + size_of::<i64>()];
            row_id[..TaskId::LEN].copy_from_slice(task_id.as_ref());
            row_id[TaskId::LEN..].copy_from_slice(&ord.to_be_bytes());

            collector_auth_tokens.push(auth_token_type.as_authentication(
                &self.crypter.decrypt(
                    "task_collector_auth_tokens",
                    &row_id,
                    "token",
                    &encrypted_collector_auth_token,
                )?,
            )?);
        }

        // HPKE keys.
        let mut hpke_keypairs = Vec::new();
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

            hpke_keypairs.push(HpkeKeypair::new(config, private_key));
        }

        // VDAF verify keys.
        let mut vdaf_verify_keys = Vec::new();
        for row in vdaf_verify_key_rows {
            let encrypted_vdaf_verify_key: Vec<u8> = row.get("vdaf_verify_key");
            vdaf_verify_keys.push(SecretBytes::new(self.crypter.decrypt(
                "task_vdaf_verify_keys",
                task_id.as_ref(),
                "vdaf_verify_key",
                &encrypted_vdaf_verify_key,
            )?));
        }

        Ok(Task::new(
            *task_id,
            endpoints,
            query_type,
            vdaf,
            aggregator_role.as_role(),
            vdaf_verify_keys,
            max_batch_query_count,
            task_expiration,
            report_expiry_age,
            min_batch_size,
            time_precision,
            tolerable_clock_skew,
            collector_hpke_config,
            aggregator_auth_tokens,
            collector_auth_tokens,
            hpke_keypairs,
        )?)
    }

    /// Retrieves report & report aggregation metrics for a given task: either a tuple
    /// `Some((report_count, report_aggregation_count))`, or None if the task does not exist.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_task_metrics(&self, task_id: &TaskId) -> Result<Option<(u64, u64)>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    (SELECT COUNT(*) FROM tasks WHERE task_id = $1) AS task_count,
                    (SELECT COUNT(*) FROM client_reports
                     JOIN tasks ON tasks.id = client_reports.task_id
                     WHERE tasks.task_id = $1
                       AND client_reports.client_timestamp >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)) AS report_count,
                    (SELECT COUNT(*) FROM aggregation_jobs
                     JOIN tasks ON tasks.id = aggregation_jobs.task_id
                     RIGHT JOIN report_aggregations ON report_aggregations.aggregation_job_id = aggregation_jobs.id
                     WHERE tasks.task_id = $1
                       AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)) AS report_aggregation_count",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ &task_id.as_ref(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?;

        let task_count: u64 = row.get_bigint_and_convert("task_count")?;
        if task_count == 0 {
            return Ok(None);
        }

        Ok(Some((
            row.get_bigint_and_convert("report_count")?,
            row.get_bigint_and_convert("report_aggregation_count")?,
        )))
    }

    /// Retrieves task IDs, optionally after some specified lower bound. This method returns tasks
    /// IDs in lexicographic order, but may not retrieve the IDs of all tasks in a single call. To
    /// retrieve additional task IDs, make additional calls to this method while specifying the
    /// `lower_bound` parameter to be the last task ID retrieved from the previous call.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_task_ids(&self, lower_bound: Option<TaskId>) -> Result<Vec<TaskId>, Error> {
        let lower_bound = lower_bound.map(|task_id| task_id.as_ref().to_vec());
        let stmt = self
            .prepare_cached(
                "SELECT task_id FROM tasks
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
    #[tracing::instrument(skip(self), err)]
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    client_reports.client_timestamp,
                    client_reports.extensions,
                    client_reports.public_share,
                    client_reports.leader_input_share,
                    client_reports.helper_encrypted_input_share
                FROM client_reports
                JOIN tasks ON tasks.id = client_reports.task_id
                WHERE tasks.task_id = $1
                  AND client_reports.report_id = $2
                  AND client_reports.client_timestamp >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* report_id */ &report_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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
        let stmt = self
            .prepare_cached(
                "SELECT client_reports.report_id, client_reports.client_timestamp
                FROM client_reports
                JOIN tasks ON tasks.id = client_reports.task_id
                WHERE tasks.task_id = $1
                  AND client_reports.client_timestamp >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    client_reports.report_id,
                    client_reports.client_timestamp,
                    client_reports.extensions,
                    client_reports.public_share,
                    client_reports.leader_input_share,
                    client_reports.helper_encrypted_input_share
                FROM client_reports
                JOIN tasks ON tasks.id = client_reports.task_id
                WHERE tasks.task_id = $1
                  AND client_reports.client_timestamp >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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

        let encoded_extensions: Vec<u8> = row.get("extensions");
        let extensions: Vec<Extension> =
            decode_u16_items(&(), &mut Cursor::new(&encoded_extensions))?;

        let encoded_public_share: Vec<u8> = row.get("public_share");
        let public_share = A::PublicShare::get_decoded_with_param(vdaf, &encoded_public_share)?;

        let encoded_leader_input_share: Vec<u8> = row.get("leader_input_share");
        let leader_input_share = A::InputShare::get_decoded_with_param(
            &(vdaf, Role::Leader.index().unwrap()),
            &encoded_leader_input_share,
        )?;

        let encoded_helper_input_share: Vec<u8> = row.get("helper_encrypted_input_share");
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

    /// `get_unaggregated_client_report_ids_for_task` returns some report IDs corresponding to
    /// unaggregated client reports for the task identified by the given task ID. Returned client
    /// reports are marked as aggregation-started: the caller must either create an aggregation job
    /// with, or call `mark_reports_unaggregated` on each returned report as part of the same
    /// transaction.
    ///
    /// This should only be used with VDAFs that have an aggregation parameter of the unit type. It
    /// relies on this assumption to find relevant reports without consulting collection jobs. For
    /// VDAFs that do have a different aggregation parameter,
    /// `get_unaggregated_client_report_ids_by_collect_for_task` should be used instead.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_unaggregated_client_report_ids_for_task(
        &self,
        task_id: &TaskId,
    ) -> Result<Vec<(ReportId, Time)>, Error> {
        // TODO(#269): allow the number of returned results to be controlled?
        let stmt = self
            .prepare_cached(
                "WITH unaggregated_reports AS (
                    SELECT client_reports.id FROM client_reports
                    JOIN tasks ON tasks.id = client_reports.task_id
                    WHERE tasks.task_id = $1
                      AND client_reports.aggregation_started = FALSE
                      AND client_reports.client_timestamp >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)
                    FOR UPDATE OF client_reports SKIP LOCKED
                    LIMIT 5000
                )
                UPDATE client_reports SET aggregation_started = TRUE
                WHERE id IN (SELECT id FROM unaggregated_reports)
                RETURNING report_id, client_timestamp",
            )
            .await?;
        let rows = self
            .query(
                &stmt,
                &[
                    /* task_id */ &task_id.as_ref(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?;

        rows.into_iter()
            .map(|row| {
                let report_id = row.get_bytea_and_convert::<ReportId>("report_id")?;
                let time = Time::from_naive_date_time(&row.get("client_timestamp"));
                Ok((report_id, time))
            })
            .collect::<Result<Vec<_>, Error>>()
    }

    /// `mark_reports_unaggregated` resets the aggregation-started flag on the given client reports,
    /// so that they may once again be returned by `get_unaggregated_client_report_ids_for_task`. It
    /// should generally only be called on report IDs returned from
    /// `get_unaggregated_client_report_ids_for_task`, as part of the same transaction, for any
    /// client reports that are not added to an aggregation job.
    #[tracing::instrument(skip(self, report_ids), err)]
    pub async fn mark_reports_unaggregated(
        &self,
        task_id: &TaskId,
        report_ids: &[ReportId],
    ) -> Result<(), Error> {
        let report_ids: Vec<_> = report_ids.iter().map(ReportId::get_encoded).collect();
        let stmt = self
            .prepare_cached(
                "UPDATE client_reports
                SET aggregation_started = false
                FROM tasks
                WHERE client_reports.task_id = tasks.id
                  AND tasks.task_id = $1
                  AND client_reports.report_id IN (SELECT * FROM UNNEST($2::BYTEA[]))
                  AND client_reports.client_timestamp >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        let row_count = self
            .execute(
                &stmt,
                &[
                    /* task_id */ &task_id.as_ref(),
                    /* report_ids */ &report_ids,
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?;
        if TryInto::<usize>::try_into(row_count)? != report_ids.len() {
            return Err(Error::MutationTargetNotFound);
        }
        Ok(())
    }

    #[cfg(feature = "test-util")]
    pub async fn mark_report_aggregated(
        &self,
        task_id: &TaskId,
        report_id: &ReportId,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "UPDATE client_reports
                SET aggregation_started = TRUE
                FROM tasks
                WHERE client_reports.task_id = tasks.id
                  AND tasks.task_id = $1
                  AND client_reports.report_id = $2
                  AND client_reports.client_timestamp >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ task_id.as_ref(),
                /* report_id */ &report_id.get_encoded(),
                /* now */ &self.clock.now().as_naive_date_time()?,
            ],
        )
        .await?;
        Ok(())
    }

    /// Determines whether the given task includes any client reports which have not yet started the
    /// aggregation process in the given interval.
    #[tracing::instrument(skip(self), err)]
    pub async fn interval_has_unaggregated_reports(
        &self,
        task_id: &TaskId,
        batch_interval: &Interval,
    ) -> Result<bool, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT EXISTS(
                    SELECT 1 FROM client_reports
                    JOIN tasks ON tasks.id = client_reports.task_id
                    WHERE tasks.task_id = $1
                    AND client_reports.client_timestamp <@ $2::TSRANGE
                    AND client_reports.client_timestamp >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)
                    AND client_reports.aggregation_started = FALSE
                ) AS unaggregated_report_exists",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ task_id.as_ref(),
                    /* batch_interval */ &SqlInterval::from(batch_interval),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?;
        Ok(row.get("unaggregated_report_exists"))
    }

    /// Return the number of reports in the provided task whose timestamp falls within the provided
    /// interval, regardless of whether the reports have been aggregated or collected. Applies only
    /// to time-interval queries.
    #[tracing::instrument(skip(self), err)]
    pub async fn count_client_reports_for_interval(
        &self,
        task_id: &TaskId,
        batch_interval: &Interval,
    ) -> Result<u64, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT COUNT(1) AS count
                FROM client_reports
                JOIN tasks ON tasks.id = client_reports.task_id
                WHERE tasks.task_id = $1
                  AND client_reports.client_timestamp >= lower($2::TSRANGE)
                  AND client_reports.client_timestamp < upper($2::TSRANGE)
                  AND client_reports.client_timestamp >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ task_id.as_ref(),
                    /* batch_interval */ &SqlInterval::from(batch_interval),
                    /* now */ &self.clock.now().as_naive_date_time()?,
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
    #[tracing::instrument(skip(self), err)]
    pub async fn count_client_reports_for_batch_id(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<u64, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT COUNT(DISTINCT report_aggregations.client_report_id) AS count
                FROM report_aggregations
                JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND aggregation_jobs.batch_id = $2
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ task_id.as_ref(),
                    /* batch_id */ &batch_id.get_encoded(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
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
    /// was already a row in the table matching `new_report`. Returns an error if something goes
    /// wrong or if the report ID is already in use with different values.
    #[tracing::instrument(skip(self), err)]
    pub async fn put_client_report<const SEED_SIZE: usize, A>(
        &self,
        vdaf: &A,
        new_report: &LeaderStoredReport<SEED_SIZE, A>,
    ) -> Result<(), Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16>,
        A::InputShare: PartialEq,
        A::PublicShare: PartialEq,
    {
        let encoded_public_share = new_report.public_share().get_encoded();
        let encoded_leader_share = new_report.leader_input_share().get_encoded();
        let encoded_helper_share = new_report.helper_encrypted_input_share().get_encoded();
        let mut encoded_extensions = Vec::new();
        encode_u16_items(&mut encoded_extensions, &(), new_report.leader_extensions());

        let stmt = self
            .prepare_cached(
                "INSERT INTO client_reports (
                    task_id,
                    report_id,
                    client_timestamp,
                    extensions,
                    public_share,
                    leader_input_share,
                    helper_encrypted_input_share
                )
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3, $4, $5, $6, $7)
                ON CONFLICT DO NOTHING
                RETURNING client_timestamp < COALESCE($3::TIMESTAMP - (SELECT report_expiry_age FROM tasks WHERE task_id = $1) * '1 second'::INTERVAL, '-infinity'::TIMESTAMP) AS is_expired",
            )
            .await?;
        let rows = self
            .query(
                &stmt,
                &[
                    /* task_id */ new_report.task_id().as_ref(),
                    /* report_id */ new_report.metadata().id().as_ref(),
                    /* client_timestamp */
                    &new_report.metadata().time().as_naive_date_time()?,
                    /* extensions */ &encoded_extensions,
                    /* public_share */ &encoded_public_share,
                    /* leader_input_share */ &encoded_leader_share,
                    /* helper_encrypted_input_share */ &encoded_helper_share,
                ],
            )
            .await?;

        if rows.len() > 1 {
            // This should never happen, because the INSERT should affect 0 or 1 rows.
            panic!(
                "INSERT for task ID {} and report ID {} affected multiple rows?",
                new_report.task_id(),
                new_report.metadata().id()
            );
        }

        if let Some(row) = rows.into_iter().next() {
            // Fast path: one row was affected, meaning there was no conflict and we wrote a new
            // report. We check that the report wasn't expired per the task's report_expiry_age,
            // but otherwise we are done. (If the report was expired, we need to delete it; we do
            // this in a separate query rather than the initial insert because the initial insert
            // cannot discriminate between a row that was skipped due to expiry & a row that was
            // skipped due to a write conflict.)
            if row.get("is_expired") {
                let stmt = self
                    .prepare_cached(
                        "DELETE FROM client_reports
                        USING tasks
                        WHERE client_reports.task_id = tasks.id
                          AND tasks.task_id = $1
                          AND client_reports.report_id = $2",
                    )
                    .await?;
                self.execute(
                    &stmt,
                    &[
                        /* task_id */ new_report.task_id().as_ref(),
                        /* report_id */ new_report.metadata().id().as_ref(),
                    ],
                )
                .await?;
            }

            return Ok(());
        }

        // Slow path: no rows were affected, meaning a row with the new report ID already existed
        // and we hit the query's ON CONFLICT DO NOTHING clause. We need to check whether the new
        // report matches the existing one.
        let existing_report = match self
            .get_client_report(vdaf, new_report.task_id(), new_report.metadata().id())
            .await?
        {
            Some(e) => e,
            None => {
                // This codepath can be taken due to a quirk of how the Repeatable Read isolation
                // level works. It cannot occur at the Serializable isolation level.
                //
                // For this codepath to be taken, two writers must concurrently choose to write the
                // same client report (by task & report ID), and this report must not already exist
                // in the datastore.
                //
                // One writer will succeed. The other will receive a unique constraint violation on
                // (task_id, report_id), since unique constraints are still enforced even in the
                // presence of snapshot isolation. They will then receive `None` from the
                // `get_client_report` call, since their snapshot is from before the successful
                // writer's write, and fall into this codepath.
                //
                // The failing writer can't do anything about this problem while in its current
                // transaction: further attempts to read the client report will continue to return
                // `None` (since all reads in the same transaction are from the same snapshot), so
                // so it can't evaluate idempotency. All it can do is give up on this transaction
                // and try again, by calling `retry` and returning an error; once it retries, it
                // will be able to read the report written by the successful writer. (It doesn't
                // matter what error we return here, as the transaction will be retried.)
                self.retry();
                return Err(Error::MutationTargetAlreadyExists);
            }
        };

        // If the existing report does not match the new report, then someone is trying to mutate an
        // existing report, which is forbidden.
        if !existing_report.eq(new_report) {
            return Err(Error::MutationTargetAlreadyExists);
        }

        // If the existing report does match the new one, then there is no error (PUTting a report
        // is idempotent).
        Ok(())
    }

    /// put_report_share stores a report share, given its associated task ID.
    ///
    /// This method is intended for use by aggregators acting in the helper role; notably, it does
    /// not store extensions, public_share, or input_shares, as these are not required to be stored
    /// for the helper workflow (and the helper never observes the entire set of encrypted input
    /// shares, so it could not record the full client report in any case).
    ///
    /// Returns `Err(Error::MutationTargetAlreadyExists)` if an attempt to mutate an existing row
    /// (e.g., changing the timestamp for a known report ID) is detected.
    #[tracing::instrument(skip(self), err)]
    pub async fn put_report_share(
        &self,
        task_id: &TaskId,
        report_share: &ReportShare,
    ) -> Result<(), Error> {
        // On conflict, we update the row, but only if the incoming client timestamp (excluded)
        // matches the existing one. This lets us detect whether there's a row with a mismatching
        // timestamp through the number of rows modified by the statement.
        let stmt = self
            .prepare_cached(
                "INSERT INTO client_reports (task_id, report_id, client_timestamp)
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3)
                ON CONFLICT (task_id, report_id) DO UPDATE
                  SET client_timestamp = client_reports.client_timestamp
                    WHERE excluded.client_timestamp = client_reports.client_timestamp",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &task_id.get_encoded(),
                    /* report_id */ &report_share.metadata().id().as_ref(),
                    /* client_timestamp */
                    &report_share.metadata().time().as_naive_date_time()?,
                ],
            )
            .await?,
        )
    }

    /// get_aggregation_job retrieves an aggregation job by ID.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_aggregation_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        task_id: &TaskId,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<Option<AggregationJob<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    aggregation_param, batch_id, client_timestamp_interval, state, round,
                    last_continue_request_hash
                FROM aggregation_jobs
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND aggregation_jobs.aggregation_job_id = $2
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
            ],
        )
        .await?
        .map(|row| Self::aggregation_job_from_row(task_id, aggregation_job_id, &row))
        .transpose()
    }

    /// get_aggregation_jobs_for_task returns all aggregation jobs for a given task ID.
    #[cfg(feature = "test-util")]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-util")))]
    #[tracing::instrument(skip(self), err)]
    pub async fn get_aggregation_jobs_for_task<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        task_id: &TaskId,
    ) -> Result<Vec<AggregationJob<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    aggregation_job_id, aggregation_param, batch_id, client_timestamp_interval,
                    state, round, last_continue_request_hash
                FROM aggregation_jobs
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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
            row.get_postgres_integer_and_convert::<i32, _, _>("round")?,
        );

        if let Some(hash) = row.try_get::<_, Option<Vec<u8>>>("last_continue_request_hash")? {
            job = job.with_last_continue_request_hash(hash.try_into().map_err(|h| {
                Error::DbState(format!(
                    "last_continue_request_hash value {h:?} cannot be converted to 32 byte array"
                ))
            })?);
        }

        Ok(job)
    }

    /// acquire_incomplete_aggregation_jobs retrieves & acquires the IDs of unclaimed incomplete
    /// aggregation jobs. At most `maximum_acquire_count` jobs are acquired. The job is acquired
    /// with a "lease" that will time out; the desired duration of the lease is a parameter, and the
    /// returned lease provides the absolute timestamp at which the lease is no longer live.
    #[tracing::instrument(skip(self), err)]
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
                "WITH incomplete_jobs AS (
                    SELECT aggregation_jobs.id FROM aggregation_jobs
                    JOIN tasks ON tasks.id = aggregation_jobs.task_id
                    WHERE tasks.aggregator_role = 'LEADER'
                    AND aggregation_jobs.state = 'IN_PROGRESS'
                    AND aggregation_jobs.lease_expiry <= $2
                    AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)
                    FOR UPDATE OF aggregation_jobs SKIP LOCKED LIMIT $3
                )
                UPDATE aggregation_jobs SET
                    lease_expiry = $1,
                    lease_token = gen_random_bytes(16),
                    lease_attempts = lease_attempts + 1
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
    #[tracing::instrument(skip(self), err)]
    pub async fn release_aggregation_job(
        &self,
        lease: &Lease<AcquiredAggregationJob>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "UPDATE aggregation_jobs
                SET lease_expiry = TIMESTAMP '-infinity',
                    lease_token = NULL,
                    lease_attempts = 0
                FROM tasks
                WHERE tasks.id = aggregation_jobs.task_id
                  AND tasks.task_id = $1
                  AND aggregation_jobs.aggregation_job_id = $2
                  AND aggregation_jobs.lease_expiry = $3
                  AND aggregation_jobs.lease_token = $4
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($5::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &lease.leased().task_id().as_ref(),
                    /* aggregation_job_id */
                    &lease.leased().aggregation_job_id().as_ref(),
                    /* lease_expiry */ &lease.lease_expiry_time(),
                    /* lease_token */ &lease.lease_token().as_ref(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?,
        )
    }

    /// put_aggregation_job stores an aggregation job.
    #[tracing::instrument(skip(self), err)]
    pub async fn put_aggregation_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        aggregation_job: &AggregationJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "INSERT INTO aggregation_jobs
                    (task_id, aggregation_job_id, aggregation_param, batch_id,
                    client_timestamp_interval, state, round, last_continue_request_hash)
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT DO NOTHING
                RETURNING UPPER(client_timestamp_interval) < COALESCE($9::TIMESTAMP - (SELECT report_expiry_age FROM tasks WHERE task_id = $1) * '1 second'::INTERVAL, '-infinity'::TIMESTAMP) AS is_expired",
            )
            .await?;
        let rows = self
            .query(
                &stmt,
                &[
                    /* task_id */ &aggregation_job.task_id().as_ref(),
                    /* aggregation_job_id */ &aggregation_job.id().as_ref(),
                    /* aggregation_param */
                    &aggregation_job.aggregation_parameter().get_encoded(),
                    /* batch_id */
                    &aggregation_job.partial_batch_identifier().get_encoded(),
                    /* client_timestamp_interval */
                    &SqlInterval::from(aggregation_job.client_timestamp_interval()),
                    /* state */ &aggregation_job.state(),
                    /* round */ &(u16::from(aggregation_job.round()) as i32),
                    /* last_continue_request_hash */
                    &aggregation_job.last_continue_request_hash(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?;
        let row_count = rows.len().try_into()?;

        if let Some(row) = rows.into_iter().next() {
            // We got a row back, meaning we wrote an aggregation job. We check that it wasn't
            // expired per the task's report_expiry_age, but otherwise we are done.
            if row.get("is_expired") {
                let stmt = self
                    .prepare_cached(
                        "DELETE FROM aggregation_jobs
                        USING tasks
                        WHERE aggregation_jobs.task_id = tasks.id
                          AND tasks.task_id = $1
                          AND aggregation_jobs.aggregation_job_id = $2",
                    )
                    .await?;
                self.execute(
                    &stmt,
                    &[
                        /* task_id */ &aggregation_job.task_id().as_ref(),
                        /* aggregation_job_id */ &aggregation_job.id().as_ref(),
                    ],
                )
                .await?;
                return Ok(());
            }
        }

        check_insert(row_count)
    }

    /// update_aggregation_job updates a stored aggregation job.
    #[tracing::instrument(skip(self), err)]
    pub async fn update_aggregation_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        aggregation_job: &AggregationJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "UPDATE aggregation_jobs SET
                    state = $1,
                    round = $2,
                    last_continue_request_hash = $3
                FROM tasks
                WHERE tasks.task_id = $4
                  AND aggregation_jobs.aggregation_job_id = $5
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($6::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */ &aggregation_job.state(),
                    /* round */ &(u16::from(aggregation_job.round()) as i32),
                    /* last_continue_request_hash */
                    &aggregation_job.last_continue_request_hash(),
                    /* task_id */ &aggregation_job.task_id().as_ref(),
                    /* aggregation_job_id */ &aggregation_job.id().as_ref(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?,
        )
    }

    /// Check whether the report has ever been aggregated with the given parameter, for an
    /// aggregation job besides the given one.
    #[tracing::instrument(skip(self), err)]
    pub async fn check_other_report_aggregation_exists<const SEED_SIZE: usize, A>(
        &self,
        task_id: &TaskId,
        report_id: &ReportId,
        aggregation_param: &A::AggregationParam,
        aggregation_job_id: &AggregationJobId,
    ) -> Result<bool, Error>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    {
        let stmt = self
            .prepare_cached(
                "SELECT 1 FROM report_aggregations
                JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND report_aggregations.client_report_id = $2
                  AND aggregation_jobs.aggregation_param = $3
                  AND aggregation_jobs.aggregation_job_id != $4
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($5::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        Ok(self
            .query_opt(
                &stmt,
                &[
                    /* task_id */ &task_id.as_ref(),
                    /* report_id */ &report_id.as_ref(),
                    /* aggregation_param */ &aggregation_param.get_encoded(),
                    /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await
            .map(|row| row.is_some())?)
    }

    /// get_report_aggregation gets a report aggregation by ID.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_report_aggregation<
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    report_aggregations.client_timestamp, report_aggregations.ord,
                    report_aggregations.state, report_aggregations.prep_state,
                    report_aggregations.prep_msg, report_aggregations.error_code,
                    report_aggregations.last_prep_step
                FROM report_aggregations
                JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND aggregation_jobs.aggregation_job_id = $2
                  AND report_aggregations.client_report_id = $3
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($4::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                /* report_id */ &report_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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

    /// get_report_aggregations_for_aggregation_job retrieves all report aggregations associated
    /// with a given aggregation job, ordered by their natural ordering.
    #[tracing::instrument(skip(self), err)]
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    report_aggregations.client_report_id, report_aggregations.client_timestamp,
                    report_aggregations.ord, report_aggregations.state,
                    report_aggregations.prep_state, report_aggregations.prep_msg,
                    report_aggregations.error_code, report_aggregations.last_prep_step
                FROM report_aggregations
                JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND aggregation_jobs.aggregation_job_id = $2
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($3::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)
                ORDER BY report_aggregations.ord ASC",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* aggregation_job_id */ &aggregation_job_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    aggregation_jobs.aggregation_job_id, report_aggregations.client_report_id,
                    report_aggregations.client_timestamp, report_aggregations.ord,
                    report_aggregations.state, report_aggregations.prep_state,
                    report_aggregations.prep_msg, report_aggregations.error_code,
                    report_aggregations.last_prep_step
                FROM report_aggregations
                JOIN aggregation_jobs ON aggregation_jobs.id = report_aggregations.aggregation_job_id
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* now */ &self.clock.now().as_naive_date_time()?,
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
        let time = Time::from_naive_date_time(&row.get("client_timestamp"));
        let ord: u64 = row.get_bigint_and_convert("ord")?;
        let state: ReportAggregationStateCode = row.get("state");
        let prep_state_bytes: Option<Vec<u8>> = row.get("prep_state");
        let prep_msg_bytes: Option<Vec<u8>> = row.get("prep_msg");
        let error_code: Option<i16> = row.get("error_code");
        let last_prep_step_bytes: Option<Vec<u8>> = row.get("last_prep_step");

        let error_code = match error_code {
            Some(c) => {
                let c: u8 = c.try_into().map_err(|err| {
                    Error::DbState(format!("couldn't convert error_code value: {err}"))
                })?;
                Some(c.try_into().map_err(|err| {
                    Error::DbState(format!("couldn't convert error_code value: {err}"))
                })?)
            }
            None => None,
        };

        let last_prep_step = last_prep_step_bytes
            .map(|bytes| PrepareStep::get_decoded(&bytes))
            .transpose()?;

        let agg_state = match state {
            ReportAggregationStateCode::Start => ReportAggregationState::Start,

            ReportAggregationStateCode::Waiting => {
                let agg_index = role.index().ok_or_else(|| {
                    Error::User(anyhow!("unexpected role: {}", role.as_str()).into())
                })?;
                let prep_state = A::PrepareState::get_decoded_with_param(
                    &(vdaf, agg_index),
                    &prep_state_bytes.ok_or_else(|| {
                        Error::DbState(
                            "report aggregation in state WAITING but prep_state is NULL"
                                .to_string(),
                        )
                    })?,
                )?;
                let prep_msg = prep_msg_bytes
                    .map(|bytes| A::PrepareMessage::get_decoded_with_param(&prep_state, &bytes))
                    .transpose()?;

                ReportAggregationState::Waiting(prep_state, prep_msg)
            }

            ReportAggregationStateCode::Finished => ReportAggregationState::Finished,

            ReportAggregationStateCode::Failed => {
                ReportAggregationState::Failed(error_code.ok_or_else(|| {
                    Error::DbState(
                        "report aggregation in state FAILED but error_code is NULL".to_string(),
                    )
                })?)
            }
        };

        Ok(ReportAggregation::new(
            *task_id,
            *aggregation_job_id,
            *report_id,
            time,
            ord,
            last_prep_step,
            agg_state,
        ))
    }

    /// put_report_aggregation stores aggregation data for a single report.
    #[tracing::instrument(skip(self), err)]
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
        let encoded_state_values = report_aggregation.state().encoded_values_from_state();
        let encoded_last_prep_step = report_aggregation
            .last_prep_step()
            .map(PrepareStep::get_encoded);

        let stmt = self
            .prepare_cached(
                "INSERT INTO report_aggregations
                    (aggregation_job_id, client_report_id, client_timestamp, ord, state, prep_state,
                     prep_msg, error_code, last_prep_step)
                SELECT aggregation_jobs.id, $3, $4, $5, $6, $7, $8, $9, $10
                FROM aggregation_jobs
                JOIN tasks ON tasks.id = aggregation_jobs.task_id
                WHERE tasks.task_id = $1
                  AND aggregation_job_id = $2
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($11::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)
                ON CONFLICT DO NOTHING",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &report_aggregation.task_id().as_ref(),
                /* aggregation_job_id */ &report_aggregation.aggregation_job_id().as_ref(),
                /* client_report_id */ &report_aggregation.report_id().as_ref(),
                /* client_timestamp */ &report_aggregation.time().as_naive_date_time()?,
                /* ord */ &TryInto::<i64>::try_into(report_aggregation.ord())?,
                /* state */ &report_aggregation.state().state_code(),
                /* prep_state */ &encoded_state_values.prep_state,
                /* prep_msg */ &encoded_state_values.prep_msg,
                /* error_code */ &encoded_state_values.report_share_err,
                /* last_prep_step */ &encoded_last_prep_step,
                /* now */ &self.clock.now().as_naive_date_time()?,
            ],
        )
        .await?;
        Ok(())
    }

    #[tracing::instrument(skip(self), err)]
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
        let encoded_state_values = report_aggregation.state().encoded_values_from_state();
        let encoded_last_prep_step = report_aggregation
            .last_prep_step()
            .map(PrepareStep::get_encoded);

        let stmt = self
            .prepare_cached(
                "UPDATE report_aggregations
                SET state = $1, prep_state = $2, prep_msg = $3, error_code = $4, last_prep_step = $5
                FROM aggregation_jobs, tasks
                WHERE report_aggregations.aggregation_job_id = aggregation_jobs.id
                  AND aggregation_jobs.task_id = tasks.id
                  AND aggregation_jobs.aggregation_job_id = $6
                  AND tasks.task_id = $7
                  AND report_aggregations.client_report_id = $8
                  AND report_aggregations.client_timestamp = $9
                  AND report_aggregations.ord = $10
                  AND UPPER(aggregation_jobs.client_timestamp_interval) >= COALESCE($11::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */
                    &report_aggregation.state().state_code(),
                    /* prep_state */ &encoded_state_values.prep_state,
                    /* prep_msg */ &encoded_state_values.prep_msg,
                    /* error_code */ &encoded_state_values.report_share_err,
                    /* last_prep_step */ &encoded_last_prep_step,
                    /* aggregation_job_id */
                    &report_aggregation.aggregation_job_id().as_ref(),
                    /* task_id */ &report_aggregation.task_id().as_ref(),
                    /* client_report_id */ &report_aggregation.report_id().as_ref(),
                    /* client_timestamp */ &report_aggregation.time().as_naive_date_time()?,
                    /* ord */ &TryInto::<i64>::try_into(report_aggregation.ord())?,
                    /* now */ &self.clock.now().as_naive_date_time()?,
                ],
            )
            .await?,
        )
    }

    /// Returns the collection job for the provided ID, or `None` if no such collection job exists.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_collection_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        collection_job_id: &CollectionJobId,
    ) -> Result<Option<CollectionJob<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    tasks.task_id,
                    collection_jobs.batch_identifier,
                    collection_jobs.aggregation_param,
                    collection_jobs.state,
                    collection_jobs.report_count,
                    collection_jobs.helper_aggregate_share,
                    collection_jobs.leader_aggregate_share
                FROM collection_jobs JOIN tasks ON tasks.id = collection_jobs.task_id
                WHERE collection_jobs.collection_job_id = $1",
            )
            .await?;
        self.query_opt(&stmt, &[&collection_job_id.as_ref()])
            .await?
            .map(|row| {
                let task_id = TaskId::get_decoded(row.get("task_id"))?;
                let batch_identifier =
                    Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
                Self::collection_job_from_row(
                    vdaf,
                    task_id,
                    batch_identifier,
                    *collection_job_id,
                    &row,
                )
            })
            .transpose()
    }

    /// Retrieves all collection jobs for the given batch identifier. Multiple collection jobs may be
    /// returned with distinct aggregation parameters.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_collection_jobs_by_batch_identifier<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    collection_jobs.collection_job_id,
                    collection_jobs.aggregation_param,
                    collection_jobs.state,
                    collection_jobs.report_count,
                    collection_jobs.helper_aggregate_share,
                    collection_jobs.leader_aggregate_share
                FROM collection_jobs JOIN tasks ON tasks.id = collection_jobs.task_id
                WHERE tasks.task_id = $1
                  AND collection_jobs.batch_identifier = $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ task_id.as_ref(),
                /* batch_identifier */ &batch_identifier.get_encoded(),
            ],
        )
        .await?
        .into_iter()
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    collection_jobs.collection_job_id,
                    collection_jobs.batch_identifier,
                    collection_jobs.aggregation_param,
                    collection_jobs.state,
                    collection_jobs.report_count,
                    collection_jobs.helper_aggregate_share,
                    collection_jobs.leader_aggregate_share
                FROM collection_jobs JOIN tasks ON tasks.id = collection_jobs.task_id
                WHERE tasks.task_id = $1",
            )
            .await?;
        self.query(&stmt, &[/* task_id */ task_id.as_ref()])
            .await?
            .into_iter()
            .map(|row| {
                let collection_job_id =
                    row.get_bytea_and_convert::<CollectionJobId>("collection_job_id")?;
                let batch_identifier =
                    Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
                Self::collection_job_from_row(
                    vdaf,
                    *task_id,
                    batch_identifier,
                    collection_job_id,
                    &row,
                )
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
        let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
        let state: CollectionJobStateCode = row.get("state");
        let report_count: Option<i64> = row.get("report_count");
        let helper_aggregate_share_bytes: Option<Vec<u8>> = row.get("helper_aggregate_share");
        let leader_aggregate_share_bytes: Option<Vec<u8>> = row.get("leader_aggregate_share");

        let state = match state {
            CollectionJobStateCode::Start => CollectionJobState::Start,

            CollectionJobStateCode::Collectable => CollectionJobState::Collectable,

            CollectionJobStateCode::Finished => {
                let report_count = u64::try_from(report_count.ok_or_else(|| {
                    Error::DbState(
                        "collection job in state FINISHED but report_count is NULL".to_string(),
                    )
                })?)?;
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
            batch_identifier,
            aggregation_param,
            state,
        ))
    }

    /// Stores a new collection job.
    #[tracing::instrument(skip(self), err)]
    pub async fn put_collection_job<
        const SEED_SIZE: usize,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        collection_job: &CollectionJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: std::fmt::Debug,
    {
        let batch_interval =
            Q::to_batch_interval(collection_job.batch_identifier()).map(SqlInterval::from);

        let stmt = self
            .prepare_cached(
                "INSERT INTO collection_jobs
                    (collection_job_id, task_id, batch_identifier, batch_interval,
                        aggregation_param, state)
                VALUES ($1, (SELECT id FROM tasks WHERE task_id = $2), $3, $4, $5, $6)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* collection_job_id */ collection_job.id().as_ref(),
                /* task_id */ collection_job.task_id().as_ref(),
                /* batch_identifier */ &collection_job.batch_identifier().get_encoded(),
                /* batch_interval */ &batch_interval,
                /* aggregation_param */
                &collection_job.aggregation_parameter().get_encoded(),
                /* state */ &collection_job.state().collection_job_state_code(),
            ],
        )
        .await?;

        Ok(())
    }

    /// acquire_incomplete_collection_jobs retrieves & acquires the IDs of unclaimed incomplete
    /// collection jobs. At most `maximum_acquire_count` jobs are acquired. The job is acquired with
    /// a "lease" that will time out; the desired duration of the lease is a parameter, and the
    /// lease expiration time is returned.
    #[tracing::instrument(skip(self), err)]
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
                "WITH incomplete_jobs AS (
                    SELECT collection_jobs.id FROM collection_jobs
                    JOIN tasks ON tasks.id = collection_jobs.task_id
                    WHERE tasks.aggregator_role = 'LEADER'
                    AND collection_jobs.state = 'COLLECTABLE'
                    AND collection_jobs.lease_expiry <= $2
                    FOR UPDATE OF collection_jobs SKIP LOCKED LIMIT $3
                )
                UPDATE collection_jobs SET
                    lease_expiry = $1,
                    lease_token = gen_random_bytes(16),
                    lease_attempts = lease_attempts + 1
                FROM tasks
                WHERE tasks.id = collection_jobs.task_id
                AND collection_jobs.id IN (SELECT id FROM incomplete_jobs)
                RETURNING tasks.task_id, tasks.query_type, tasks.vdaf,
                          collection_jobs.collection_job_id, collection_jobs.id,
                          collection_jobs.lease_token, collection_jobs.lease_attempts",
            )
            .await?;

        self.query(
            &stmt,
            &[
                /* lease_expiry */ &lease_expiry_time,
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
            let lease_token = row.get_bytea_and_convert::<LeaseToken>("lease_token")?;
            let lease_attempts = row.get_bigint_and_convert("lease_attempts")?;
            Ok(Lease::new(
                AcquiredCollectionJob::new(task_id, collection_job_id, query_type, vdaf),
                lease_expiry_time,
                lease_token,
                lease_attempts,
            ))
        })
        .collect()
    }

    /// release_collection_job releases an acquired (via e.g. acquire_incomplete_collection_jobs)
    /// collect job. It returns an error if the collection job has no current lease.
    #[tracing::instrument(skip(self), err)]
    pub async fn release_collection_job(
        &self,
        lease: &Lease<AcquiredCollectionJob>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "UPDATE collection_jobs
                SET lease_expiry = TIMESTAMP '-infinity',
                    lease_token = NULL,
                    lease_attempts = 0
                FROM tasks
                WHERE tasks.id = collection_jobs.task_id
                  AND tasks.task_id = $1
                  AND collection_jobs.collection_job_id = $2
                  AND collection_jobs.lease_expiry = $3
                  AND collection_jobs.lease_token = $4",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &lease.leased().task_id().as_ref(),
                    /* collection_job_id */ &lease.leased().collection_job_id().as_ref(),
                    /* lease_expiry */ &lease.lease_expiry_time(),
                    /* lease_token */ &lease.lease_token().as_ref(),
                ],
            )
            .await?,
        )
    }

    /// Updates an existing collection job.
    #[tracing::instrument(skip(self), err)]
    pub async fn update_collection_job<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        collection_job: &CollectionJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let (report_count, leader_aggregate_share, helper_aggregate_share) = match collection_job
            .state()
        {
            CollectionJobState::Start => {
                return Err(Error::InvalidParameter(
                    "cannot update collection job into START state",
                ));
            }
            CollectionJobState::Finished {
                report_count,
                encrypted_helper_aggregate_share,
                leader_aggregate_share,
            } => {
                let report_count: Option<i64> = Some(i64::try_from(*report_count)?);
                let leader_aggregate_share: Option<Vec<u8>> =
                    Some(leader_aggregate_share.get_encoded());
                let helper_aggregate_share = Some(encrypted_helper_aggregate_share.get_encoded());

                (report_count, leader_aggregate_share, helper_aggregate_share)
            }
            CollectionJobState::Collectable
            | CollectionJobState::Abandoned
            | CollectionJobState::Deleted => (None, None, None),
        };

        let stmt = self
            .prepare_cached(
                "UPDATE collection_jobs SET
                    state = $1,
                    report_count = $2,
                    leader_aggregate_share = $3,
                    helper_aggregate_share = $4
                WHERE collection_job_id = $5",
            )
            .await?;

        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */ &collection_job.state().collection_job_state_code(),
                    /* report_count */ &report_count,
                    /* leader_aggregate_share */ &leader_aggregate_share,
                    /* helper_aggregate_share */ &helper_aggregate_share,
                    /* collection_job_id */ &collection_job.id().as_ref(),
                ],
            )
            .await?,
        )
    }

    /// Retrieves an existing batch aggregation.
    #[tracing::instrument(skip(self, aggregation_parameter), err)]
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
        let stmt = self
            .prepare_cached(
                "SELECT state, aggregate_share, report_count, checksum
                FROM batch_aggregations
                WHERE
                    task_id = (SELECT id FROM tasks WHERE task_id = $1)
                    AND batch_identifier = $2
                    AND aggregation_param = $3
                    AND ord = $4",
            )
            .await?;

        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* batch_identifier */ &batch_identifier.get_encoded(),
                /* aggregation_param */ &aggregation_parameter.get_encoded(),
                /* ord */ &TryInto::<i64>::try_into(ord)?,
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
    #[tracing::instrument(skip(self, aggregation_parameter), err)]
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
        let stmt = self
            .prepare_cached(
                "SELECT ord, state, aggregate_share, report_count, checksum
                FROM batch_aggregations
                WHERE
                    task_id = (SELECT id FROM tasks WHERE task_id = $1)
                    AND batch_identifier = $2
                    AND aggregation_param = $3",
            )
            .await?;

        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* batch_identifier */ &batch_identifier.get_encoded(),
                /* aggregation_param */ &aggregation_parameter.get_encoded(),
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    batch_identifier, aggregation_param, ord, state, aggregate_share, report_count,
                    checksum
                FROM batch_aggregations
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)",
            )
            .await?;

        self.query(&stmt, &[/* task_id */ &task_id.as_ref()])
            .await?
            .into_iter()
            .map(|row| {
                let batch_identifier =
                    Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
                let aggregation_param =
                    A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
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
        aggregation_parameter: A::AggregationParam,
        ord: u64,
        row: Row,
    ) -> Result<BatchAggregation<SEED_SIZE, Q, A>, Error> {
        let state = row.get("state");
        let aggregate_share = row
            .get::<_, Option<Vec<u8>>>("aggregate_share")
            .map(|bytes| {
                A::AggregateShare::get_decoded_with_param(&(vdaf, &aggregation_parameter), &bytes)
            })
            .transpose()
            .map_err(|_| Error::DbState("aggregate_share couldn't be parsed".to_string()))?;
        let report_count = row.get_bigint_and_convert("report_count")?;
        let checksum = ReportIdChecksum::get_decoded(row.get("checksum"))?;
        Ok(BatchAggregation::new(
            task_id,
            batch_identifier,
            aggregation_parameter,
            ord,
            state,
            aggregate_share,
            report_count,
            checksum,
        ))
    }

    /// Store a new `batch_aggregations` row in the datastore.
    #[tracing::instrument(skip(self), err)]
    pub async fn put_batch_aggregation<
        const SEED_SIZE: usize,
        Q: AccumulableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        batch_aggregation: &BatchAggregation<SEED_SIZE, Q, A>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: std::fmt::Debug,
        A::AggregateShare: std::fmt::Debug,
    {
        let batch_interval =
            Q::to_batch_interval(batch_aggregation.batch_identifier()).map(SqlInterval::from);

        let stmt = self
            .prepare_cached(
                "INSERT INTO batch_aggregations (
                    task_id, batch_identifier, batch_interval, aggregation_param, ord, state,
                    aggregate_share, report_count, checksum
                )
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT DO NOTHING",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &batch_aggregation.task_id().as_ref(),
                    /* batch_identifier */
                    &batch_aggregation.batch_identifier().get_encoded(),
                    /* batch_interval */ &batch_interval,
                    /* aggregation_param */
                    &batch_aggregation.aggregation_parameter().get_encoded(),
                    /* ord */ &TryInto::<i64>::try_into(batch_aggregation.ord())?,
                    /* state */ &batch_aggregation.state(),
                    /* aggregate_share */
                    &batch_aggregation.aggregate_share().map(Encode::get_encoded),
                    /* report_count */
                    &i64::try_from(batch_aggregation.report_count())?,
                    /* checksum */ &batch_aggregation.checksum().get_encoded(),
                ],
            )
            .await?,
        )
    }

    /// Update an existing `batch_aggregations` row with the values from the provided batch
    /// aggregation.
    #[tracing::instrument(skip(self), err)]
    pub async fn update_batch_aggregation<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        batch_aggregation: &BatchAggregation<SEED_SIZE, Q, A>,
    ) -> Result<(), Error>
    where
        A::AggregationParam: std::fmt::Debug,
        A::AggregateShare: std::fmt::Debug,
    {
        let stmt = self
            .prepare_cached(
                "UPDATE batch_aggregations
                SET
                    state = $1,
                    aggregate_share = $2,
                    report_count = $3,
                    checksum = $4
                WHERE
                    task_id = (SELECT id from TASKS WHERE task_id = $5)
                    AND batch_identifier = $6
                    AND aggregation_param = $7
                    AND ord = $8",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */
                    &batch_aggregation.state(),
                    /* aggregate_share */
                    &batch_aggregation.aggregate_share().map(Encode::get_encoded),
                    /* report_count */
                    &i64::try_from(batch_aggregation.report_count())?,
                    /* checksum */ &batch_aggregation.checksum().get_encoded(),
                    /* task_id */ &batch_aggregation.task_id().as_ref(),
                    /* batch_identifier */
                    &batch_aggregation.batch_identifier().get_encoded(),
                    /* aggregation_param */
                    &batch_aggregation.aggregation_parameter().get_encoded(),
                    /* ord */ &TryInto::<i64>::try_into(batch_aggregation.ord())?,
                ],
            )
            .await?,
        )?;

        Ok(())
    }

    /// Fetch an [`AggregateShareJob`] from the datastore corresponding to given parameters, or
    /// `None` if no such job exists.
    #[tracing::instrument(skip(self), err)]
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
        let stmt = self
            .prepare_cached(
                "SELECT helper_aggregate_share, report_count, checksum FROM aggregate_share_jobs
                WHERE
                    task_id = (SELECT id FROM tasks WHERE task_id = $1)
                    AND batch_identifier = $2
                    AND aggregation_param = $3",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* batch_identifier */ &batch_identifier.get_encoded(),
                /* aggregation_param */ &aggregation_parameter.get_encoded(),
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

    /// Returns all aggregate share jobs for the given task with the given batch identifier.
    /// Multiple aggregate share jobs may be returned with distinct aggregation parameters.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_aggregate_share_jobs_by_batch_identifier<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
    ) -> Result<Vec<AggregateShareJob<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    aggregate_share_jobs.aggregation_param,
                    aggregate_share_jobs.helper_aggregate_share,
                    aggregate_share_jobs.report_count,
                    aggregate_share_jobs.checksum
                FROM aggregate_share_jobs JOIN tasks ON tasks.id = aggregate_share_jobs.task_id
                WHERE tasks.task_id = $1
                  AND aggregate_share_jobs.batch_identifier = $2",
            )
            .await?;
        self.query(
            &stmt,
            &[
                /* task_id */ &task_id.as_ref(),
                /* batch_identifier */ &batch_identifier.get_encoded(),
            ],
        )
        .await?
        .into_iter()
        .map(|row| {
            let aggregation_param = A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
            Self::aggregate_share_job_from_row(
                vdaf,
                task_id,
                batch_identifier.clone(),
                aggregation_param,
                &row,
            )
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
        let stmt = self
            .prepare_cached(
                "SELECT
                    aggregate_share_jobs.batch_identifier,
                    aggregate_share_jobs.aggregation_param,
                    aggregate_share_jobs.helper_aggregate_share,
                    aggregate_share_jobs.report_count,
                    aggregate_share_jobs.checksum
                FROM aggregate_share_jobs JOIN tasks ON tasks.id = aggregate_share_jobs.task_id
                WHERE tasks.task_id = $1",
            )
            .await?;
        self.query(&stmt, &[/* task_id */ &task_id.as_ref()])
            .await?
            .into_iter()
            .map(|row| {
                let batch_identifier =
                    Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
                let aggregation_param =
                    A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
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
    #[tracing::instrument(skip(self), err)]
    pub async fn put_aggregate_share_job<
        const SEED_SIZE: usize,
        Q: CollectableQueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        job: &AggregateShareJob<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let batch_interval = Q::to_batch_interval(job.batch_identifier()).map(SqlInterval::from);

        let stmt = self
            .prepare_cached(
                "INSERT INTO aggregate_share_jobs (
                    task_id, batch_identifier, batch_interval, aggregation_param,
                    helper_aggregate_share, report_count, checksum
                )
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3, $4, $5, $6, $7)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &job.task_id().as_ref(),
                /* batch_identifier */ &job.batch_identifier().get_encoded(),
                /* batch_interval */ &batch_interval,
                /* aggregation_param */ &job.aggregation_parameter().get_encoded(),
                /* helper_aggregate_share */ &job.helper_aggregate_share().get_encoded(),
                /* report_count */ &i64::try_from(job.report_count())?,
                /* checksum */ &job.checksum().get_encoded(),
            ],
        )
        .await?;

        Ok(())
    }

    /// Writes an outstanding batch. (This method does not take an [`OutstandingBatch`] as several
    /// of the included values are read implicitly.)
    #[tracing::instrument(skip(self), err)]
    pub async fn put_outstanding_batch(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "INSERT INTO outstanding_batches (task_id, batch_id)
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2)",
            )
            .await?;

        self.execute(
            &stmt,
            &[
                /* task_id */ task_id.as_ref(),
                /* batch_id */ batch_id.as_ref(),
            ],
        )
        .await?;

        Ok(())
    }

    /// Retrieves all [`OutstandingBatch`]es for a given task.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_outstanding_batches_for_task(
        &self,
        task_id: &TaskId,
    ) -> Result<Vec<OutstandingBatch>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT batch_id FROM outstanding_batches
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)",
            )
            .await?;

        try_join_all(
            self.query(&stmt, &[/* task_id */ task_id.as_ref()])
                .await?
                .into_iter()
                .map(|row| async move {
                    let batch_id = BatchId::get_decoded(row.get("batch_id"))?;
                    let size = self.read_batch_size(task_id, &batch_id).await?;
                    Ok(OutstandingBatch::new(*task_id, batch_id, size))
                }),
        )
        .await
    }

    // Return value is an inclusive range [min_size, max_size], where:
    //  * min_size is the minimum possible number of reports included in the batch, i.e. all report
    //    aggregations in the batch which have reached the FINISHED state.
    //  * max_size is the maximum possible number of reports included in the batch, i.e. all report
    //    aggregations in the batch which are in a non-failure state (START/WAITING/FINISHED).
    async fn read_batch_size(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<RangeInclusive<usize>, Error> {
        // TODO(#1467): fix this to work in presence of GC.
        let stmt = self
            .prepare_cached(
                "WITH batch_report_aggregation_statuses AS
                    (SELECT report_aggregations.state, COUNT(*) AS count FROM report_aggregations
                     JOIN aggregation_jobs
                        ON report_aggregations.aggregation_job_id = aggregation_jobs.id
                     WHERE aggregation_jobs.task_id = (SELECT id FROM tasks WHERE task_id = $1)
                     AND aggregation_jobs.batch_id = $2
                     GROUP BY report_aggregations.state)
                SELECT
                    (SELECT SUM(count)::BIGINT FROM batch_report_aggregation_statuses
                     WHERE state IN ('FINISHED')) AS min_size,
                    (SELECT SUM(count)::BIGINT FROM batch_report_aggregation_statuses
                     WHERE state IN ('START', 'WAITING', 'FINISHED')) AS max_size",
            )
            .await?;

        let row = self
            .query_one(
                &stmt,
                &[
                    /* task_id */ task_id.as_ref(),
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

    /// Deletes an outstanding batch.
    #[tracing::instrument(skip(self), err)]
    pub async fn delete_outstanding_batch(
        &self,
        task_id: &TaskId,
        batch_id: &BatchId,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "DELETE FROM outstanding_batches
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)
                  AND batch_id = $2",
            )
            .await?;

        self.execute(
            &stmt,
            &[
                /* task_id */ task_id.as_ref(),
                /* batch_id */ batch_id.as_ref(),
            ],
        )
        .await?;
        Ok(())
    }

    /// Retrieves an outstanding batch for the given task with at least the given number of
    /// successfully-aggregated reports.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_filled_outstanding_batch(
        &self,
        task_id: &TaskId,
        min_report_count: u64,
    ) -> Result<Option<BatchId>, Error> {
        // TODO(#1467): fix this to work in presence of GC.
        let stmt = self
            .prepare_cached(
                "WITH batches AS (
                    SELECT
                        outstanding_batches.batch_id AS batch_id,
                        SUM(batch_aggregations.report_count) AS count
                    FROM outstanding_batches
                    JOIN batch_aggregations
                      ON batch_aggregations.batch_identifier = outstanding_batches.batch_id
                    WHERE outstanding_batches.task_id = (SELECT id FROM tasks WHERE task_id = $1)
                    GROUP BY outstanding_batches.batch_id
                )
                SELECT batch_id FROM batches WHERE count >= $2::bigint LIMIT 1",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ task_id.as_ref(),
                /* min_report_count */ &i64::try_from(min_report_count)?,
            ],
        )
        .await?
        .map(|row| Ok(BatchId::get_decoded(row.get("batch_id"))?))
        .transpose()
    }

    /// Puts a `batch` into the datastore. Returns `MutationTargetAlreadyExists` if the batch is
    /// already stored.
    #[tracing::instrument(skip(self), err)]
    pub async fn put_batch<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        batch: &Batch<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "INSERT INTO batches
                    (task_id, batch_identifier, aggregation_param, state,
                    outstanding_aggregation_jobs, client_timestamp_interval)
                VALUES ((SELECT id FROM tasks WHERE task_id = $1), $2, $3, $4, $5, $6)
                ON CONFLICT DO NOTHING",
            )
            .await?;
        check_insert(
            self.execute(
                &stmt,
                &[
                    /* task_id */ &batch.task_id().as_ref(),
                    /* batch_identifier */ &batch.batch_identifier().get_encoded(),
                    /* aggregation_param */ &batch.aggregation_parameter().get_encoded(),
                    /* state */ &batch.state(),
                    /* outstanding_aggregation_jobs */
                    &i64::try_from(batch.outstanding_aggregation_jobs())?,
                    /* client_timestamp_interval */
                    &SqlInterval::from(batch.client_timestamp_interval()),
                ],
            )
            .await?,
        )
    }

    /// Updates a given `batch` in the datastore. Returns `MutationTargetNotFound` if no such batch
    /// is currently stored.
    #[tracing::instrument(skip(self), err)]
    pub async fn update_batch<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        batch: &Batch<SEED_SIZE, Q, A>,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "UPDATE batches
                SET state = $1, outstanding_aggregation_jobs = $2, client_timestamp_interval = $3
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $4)
                  AND batch_identifier = $5
                  AND aggregation_param = $6",
            )
            .await?;
        check_single_row_mutation(
            self.execute(
                &stmt,
                &[
                    /* state */ &batch.state(),
                    /* outstanding_aggregation_jobs */
                    &i64::try_from(batch.outstanding_aggregation_jobs())?,
                    /* client_timestamp_interval */
                    &SqlInterval::from(batch.client_timestamp_interval()),
                    /* task_id */ &batch.task_id().as_ref(),
                    /* batch_identifier */ &batch.batch_identifier().get_encoded(),
                    /* aggregation_param */ &batch.aggregation_parameter().get_encoded(),
                ],
            )
            .await?,
        )
    }

    /// Gets a given `batch` from the datastore, based on the primary key. Returns `None` if no such
    /// batch is stored in the datastore.
    #[tracing::instrument(skip(self), err)]
    pub async fn get_batch<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        task_id: &TaskId,
        batch_identifier: &Q::BatchIdentifier,
        aggregation_parameter: &A::AggregationParam,
    ) -> Result<Option<Batch<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT state, outstanding_aggregation_jobs, client_timestamp_interval FROM batches
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)
                  AND batch_identifier = $2
                  AND aggregation_param = $3",
            )
            .await?;
        self.query_opt(
            &stmt,
            &[
                /* task_id */ task_id.as_ref(),
                /* batch_identifier */ &batch_identifier.get_encoded(),
                /* aggregation_param */ &aggregation_parameter.get_encoded(),
            ],
        )
        .await?
        .map(|row| {
            Self::batch_from_row(
                *task_id,
                batch_identifier.clone(),
                aggregation_parameter.clone(),
                row,
            )
        })
        .transpose()
    }

    #[cfg(feature = "test-util")]
    pub async fn get_batches_for_task<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    >(
        &self,
        task_id: &TaskId,
    ) -> Result<Vec<Batch<SEED_SIZE, Q, A>>, Error> {
        let stmt = self
            .prepare_cached(
                "SELECT
                    batch_identifier, aggregation_param, state, outstanding_aggregation_jobs,
                    client_timestamp_interval
                FROM batches
                WHERE task_id = (SELECT id FROM tasks WHERE task_id = $1)",
            )
            .await?;
        self.query(&stmt, &[/* task_id */ task_id.as_ref()])
            .await?
            .into_iter()
            .map(|row| {
                let batch_identifier =
                    Q::BatchIdentifier::get_decoded(row.get("batch_identifier"))?;
                let aggregation_parameter =
                    A::AggregationParam::get_decoded(row.get("aggregation_param"))?;
                Self::batch_from_row(*task_id, batch_identifier, aggregation_parameter, row)
            })
            .collect()
    }

    fn batch_from_row<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>>(
        task_id: TaskId,
        batch_identifier: Q::BatchIdentifier,
        aggregation_parameter: A::AggregationParam,
        row: Row,
    ) -> Result<Batch<SEED_SIZE, Q, A>, Error> {
        let state = row.get("state");
        let outstanding_aggregation_jobs = row
            .get::<_, i64>("outstanding_aggregation_jobs")
            .try_into()?;
        let client_timestamp_interval = row
            .get::<_, SqlInterval>("client_timestamp_interval")
            .as_interval();
        Ok(Batch::new(
            task_id,
            batch_identifier,
            aggregation_parameter,
            state,
            outstanding_aggregation_jobs,
            client_timestamp_interval,
        ))
    }

    /// Deletes old client reports for a given task, that is, client reports whose timestamp is
    /// older than a given timestamp which are not included in any report aggregations.
    #[tracing::instrument(skip(self), err)]
    pub async fn delete_expired_client_reports(&self, task_id: &TaskId) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "DELETE FROM client_reports
                USING tasks
                WHERE client_reports.task_id = tasks.id
                  AND tasks.task_id = $1
                  AND client_reports.client_timestamp < COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &task_id.get_encoded(),
                /* now */ &self.clock.now().as_naive_date_time()?,
            ],
        )
        .await?;
        Ok(())
    }

    /// Deletes old aggregation artifacts (aggregation jobs/report aggregations/batch aggregations)
    /// for a given task, that is, aggregation artifacts for which all associated client reports
    /// have timestamps older than a given timestamp, and which are not included in any collection
    /// artifacts.
    ///
    /// After calling this function, delete_expired_client_reports must be called in the same
    /// transaction to avoid re-aggregating client reports.
    #[tracing::instrument(skip(self), err)]
    pub async fn delete_expired_aggregation_artifacts(
        &self,
        task_id: &TaskId,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "WITH aggregation_jobs_to_delete AS (
                    SELECT aggregation_jobs.id FROM aggregation_jobs
                    JOIN tasks ON tasks.id = aggregation_jobs.task_id
                    WHERE tasks.task_id = $1
                      AND UPPER(aggregation_jobs.client_timestamp_interval) < COALESCE($2::TIMESTAMP - tasks.report_expiry_age * '1 second'::INTERVAL, '-infinity'::TIMESTAMP)
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
                /* task_id */ &task_id.get_encoded(),
                /* now */ &self.clock.now().as_naive_date_time()?,
            ],
        )
        .await?;
        Ok(())
    }

    /// Deletes old collection artifacts (collection jobs/aggregate share jobs/outstanding batches) for
    /// a given task, that is, collection artifacts for which all associated client reports have
    /// timestamps older than a given timestamp.
    ///
    /// After calling this function, delete_expired_aggregation_artifacts must be called in the same
    /// transaction to avoid re-collecting old aggregations.
    #[tracing::instrument(skip(self), err)]
    pub async fn delete_expired_collection_artifacts(
        &self,
        task_id: &TaskId,
        oldest_allowed_report_timestamp: Time,
    ) -> Result<(), Error> {
        let stmt = self
            .prepare_cached(
                "WITH collection_jobs_to_delete AS (
                    SELECT id FROM collection_jobs
                    JOIN (
                        SELECT
                            collection_jobs.id AS collection_job_id,
                            MAX(UPPER(aggregation_jobs.client_timestamp_interval)) AS max_timestamp
                        FROM collection_jobs
                        JOIN aggregation_jobs
                            ON aggregation_jobs.task_id = collection_jobs.task_id
                            AND (aggregation_jobs.batch_id = collection_jobs.batch_identifier
                              OR aggregation_jobs.client_timestamp_interval &&
                                collection_jobs.batch_interval)
                        GROUP BY collection_jobs.id
                    ) report_max_timestamps
                        ON report_max_timestamps.collection_job_id = collection_jobs.id
                    WHERE collection_jobs.task_id = (SELECT id FROM tasks WHERE task_id = $1)
                        AND report_max_timestamps.max_timestamp <= $2
                ),
                aggregate_share_jobs_to_delete AS (
                    SELECT id FROM aggregate_share_jobs
                    JOIN (
                        SELECT
                            aggregate_share_jobs.id AS aggregate_share_job_id,
                            MAX(UPPER(aggregation_jobs.client_timestamp_interval)) AS max_timestamp
                        FROM aggregate_share_jobs
                        JOIN aggregation_jobs
                            ON aggregation_jobs.task_id = aggregate_share_jobs.task_id
                            AND (aggregation_jobs.batch_id = aggregate_share_jobs.batch_identifier
                              OR aggregation_jobs.client_timestamp_interval &&
                                aggregate_share_jobs.batch_interval)
                        GROUP BY aggregate_share_jobs.id
                    ) report_max_timestamps
                        ON report_max_timestamps.aggregate_share_job_id = aggregate_share_jobs.id
                    WHERE aggregate_share_jobs.task_id = (SELECT id FROM tasks WHERE task_id = $1)
                      AND report_max_timestamps.max_timestamp <= $2
                ),
                deleted_aggregate_share_jobs AS (
                    DELETE FROM aggregate_share_jobs
                    WHERE id IN (SELECT id FROM aggregate_share_jobs_to_delete)
                ),
                outstanding_batches_to_delete AS (
                    SELECT id FROM outstanding_batches
                    JOIN (
                        SELECT
                            outstanding_batches.batch_id,
                            MAX(UPPER(aggregation_jobs.client_timestamp_interval)) AS max_timestamp
                        FROM outstanding_batches
                        JOIN aggregation_jobs
                            ON aggregation_jobs.task_id = outstanding_batches.task_id
                           AND aggregation_jobs.batch_id = outstanding_batches.batch_id
                        GROUP BY outstanding_batches.batch_id
                    ) report_max_timestamps
                        ON report_max_timestamps.batch_id = outstanding_batches.batch_id
                    WHERE outstanding_batches.task_id = (SELECT id FROM tasks WHERE task_id = $1)
                      AND report_max_timestamps.max_timestamp <= $2
                ),
                deleted_outstanding_batches AS (
                    DELETE FROM outstanding_batches
                    WHERE id IN (SELECT id FROM outstanding_batches_to_delete)
                )
                DELETE FROM collection_jobs WHERE id IN (SELECT id FROM collection_jobs_to_delete)",
            )
            .await?;
        self.execute(
            &stmt,
            &[
                /* task_id */ &task_id.get_encoded(),
                /* client_timestamp */
                &oldest_allowed_report_timestamp.as_naive_date_time()?,
            ],
        )
        .await?;
        Ok(())
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
        for<'a> <T as TryFrom<&'a [u8]>>::Error: std::fmt::Debug;

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
        let postgres_integer: P = self.try_get(idx)?;
        Ok(T::try_from(postgres_integer)?)
    }

    fn get_nullable_bigint_and_convert<I, T>(&self, idx: I) -> Result<Option<T>, Error>
    where
        I: RowIndex + Display,
        T: TryFrom<i64, Error = std::num::TryFromIntError>,
    {
        let bigint: Option<i64> = self.try_get(idx)?;
        Ok(bigint.map(|bigint| T::try_from(bigint)).transpose()?)
    }

    fn get_bytea_and_convert<T>(&self, idx: &'static str) -> Result<T, Error>
    where
        for<'a> T: TryFrom<&'a [u8]>,
        for<'a> <T as TryFrom<&'a [u8]>>::Error: std::fmt::Debug,
    {
        let encoded: Vec<u8> = self.try_get(idx)?;
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
        let encoded: Vec<u8> = self.try_get(idx)?;
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
    #[error("batch already collected")]
    AlreadyCollected,
}

impl From<ring::error::Unspecified> for Error {
    fn from(_: ring::error::Unspecified) -> Self {
        Error::Crypt
    }
}

/// This module contains models used by the datastore that are not DAP messages.
pub mod models {
    use crate::{datastore::Error, task};
    use base64::{display::Base64Display, engine::general_purpose::URL_SAFE_NO_PAD};
    use chrono::NaiveDateTime;
    use derivative::Derivative;
    use janus_core::{
        report_id::ReportIdChecksumExt,
        task::{AuthenticationToken, DapAuthToken, VdafInstance},
        time::{DurationExt, IntervalExt, TimeExt},
    };
    use janus_messages::{
        query_type::{FixedSize, QueryType},
        AggregationJobId, AggregationJobRound, BatchId, CollectionJobId, Duration, Extension,
        HpkeCiphertext, Interval, PrepareStep, ReportId, ReportIdChecksum, ReportMetadata,
        ReportShareError, Role, TaskId, Time,
    };
    use postgres_protocol::types::{
        range_from_sql, range_to_sql, timestamp_from_sql, timestamp_to_sql, Range, RangeBound,
    };
    use postgres_types::{accepts, to_sql_checked, FromSql, ToSql};
    use prio::{
        codec::Encode,
        vdaf::{self, Aggregatable},
    };
    use rand::{distributions::Standard, prelude::Distribution};
    use std::{
        fmt::{Debug, Display, Formatter},
        hash::{Hash, Hasher},
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
        pub fn as_authentication(&self, token: &[u8]) -> Result<AuthenticationToken, Error> {
            match self {
                Self::DapAuthToken => DapAuthToken::try_from(token.to_vec())
                    .map(AuthenticationToken::DapAuth)
                    .map_err(|e| {
                        Error::DbState(format!("invalid DAP auth token in database: {e:?}"))
                    }),
                Self::AuthorizationBearerToken => Ok(AuthenticationToken::Bearer(token.into())),
            }
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

    /// Represents a report as it is stored in the leader's database, corresponding to a row in
    /// `client_reports`, where `leader_input_share` and `helper_encrypted_input_share` are required
    /// to be populated.
    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub struct LeaderStoredReport<const SEED_SIZE: usize, A>
    where
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    {
        task_id: TaskId,
        metadata: ReportMetadata,
        #[derivative(Debug = "ignore")]
        public_share: A::PublicShare,
        leader_extensions: Vec<Extension>,
        #[derivative(Debug = "ignore")]
        leader_input_share: A::InputShare,
        #[derivative(Debug = "ignore")]
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
            leader_extensions: Vec<Extension>,
            leader_input_share: A::InputShare,
            helper_encrypted_input_share: HpkeCiphertext,
        ) -> Self {
            Self {
                task_id,
                metadata,
                public_share,
                leader_extensions,
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

        pub fn leader_extensions(&self) -> &[Extension] {
            &self.leader_extensions
        }

        pub fn leader_input_share(&self) -> &A::InputShare {
            &self.leader_input_share
        }

        pub fn helper_encrypted_input_share(&self) -> &HpkeCiphertext {
            &self.helper_encrypted_input_share
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
                && self.leader_extensions == other.leader_extensions
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
    impl LeaderStoredReport<0, janus_core::test_util::dummy_vdaf::Vdaf> {
        pub fn new_dummy(task_id: TaskId, when: Time) -> Self {
            use janus_messages::HpkeConfigId;
            use rand::random;

            Self::new(
                task_id,
                ReportMetadata::new(random(), when),
                (),
                Vec::new(),
                janus_core::test_util::dummy_vdaf::InputShare::default(),
                HpkeCiphertext::new(
                    HpkeConfigId::from(13),
                    Vec::from("encapsulated_context_0"),
                    Vec::from("payload_0"),
                ),
            )
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
    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub struct AggregationJob<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    > {
        /// The ID of the task this aggregation job belongs to.
        task_id: TaskId,
        /// The ID of this aggregation job.
        aggregation_job_id: AggregationJobId,
        /// The aggregation parameter this job is run with.
        #[derivative(Debug = "ignore")]
        aggregation_parameter: A::AggregationParam,
        /// The partial identifier for the batch this aggregation job contributes to (fixed size
        /// tasks only; for time interval tasks, aggregation jobs may span multiple batches).
        batch_id: Q::PartialBatchIdentifier,
        /// The minimal interval of time spanned by the reports included in this aggregation job.
        client_timestamp_interval: Interval,
        /// The overall state of this aggregation job.
        state: AggregationJobState,
        /// The round of VDAF preparation that this aggregation job is currently on.
        round: AggregationJobRound,
        /// The SHA-256 hash of the most recent [`janus_messages::AggregationJobContinueReq`]
        /// received for this aggregation job. Will only be set for helpers, and only after the
        /// first round of the job.
        last_continue_request_hash: Option<[u8; 32]>,
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>>
        AggregationJob<SEED_SIZE, Q, A>
    {
        /// Creates a new [`AggregationJob`].
        pub fn new(
            task_id: TaskId,
            aggregation_job_id: AggregationJobId,
            aggregation_parameter: A::AggregationParam,
            batch_id: Q::PartialBatchIdentifier,
            client_timestamp_interval: Interval,
            state: AggregationJobState,
            round: AggregationJobRound,
        ) -> Self {
            Self {
                task_id,
                aggregation_job_id,
                aggregation_parameter,
                batch_id,
                client_timestamp_interval,
                state,
                round,
                last_continue_request_hash: None,
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
        /// This method would typically be used for code which is generic over the query type.
        /// Query-type specific code will typically call [`Self::batch_id`].
        pub fn partial_batch_identifier(&self) -> &Q::PartialBatchIdentifier {
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

        /// Returns the round of the VDAF preparation protocol the aggregation job is on.
        pub fn round(&self) -> AggregationJobRound {
            self.round
        }

        /// Returns a new [`AggregationJob`] corresponding to this aggregation job updated to be on
        /// the given VDAF preparation round.
        pub fn with_round(self, round: AggregationJobRound) -> Self {
            Self { round, ..self }
        }

        /// Returns the SHA-256 digest of the most recent
        /// [`janus_messages::AggregationJobContinueReq`] for the job, if any.
        pub fn last_continue_request_hash(&self) -> Option<[u8; 32]> {
            self.last_continue_request_hash
        }

        /// Returns a new [`AggregationJob`] corresponding to this aggregation job updated to have
        /// the given last continue request hash.
        pub fn with_last_continue_request_hash(self, hash: [u8; 32]) -> Self {
            Self {
                last_continue_request_hash: Some(hash),
                ..self
            }
        }
    }

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
        AggregationJob<SEED_SIZE, FixedSize, A>
    {
        /// Gets the batch ID associated with this aggregation job.
        pub fn batch_id(&self) -> &BatchId {
            self.partial_batch_identifier()
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for AggregationJob<SEED_SIZE, Q, A>
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
                && self.round == other.round
                && self.last_continue_request_hash == other.last_continue_request_hash
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for AggregationJob<SEED_SIZE, Q, A>
    where
        A::AggregationParam: Eq,
    {
    }

    /// AggregationJobState represents the state of an aggregation job. It corresponds to the
    /// AGGREGATION_JOB_STATE enum in the schema.
    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, ToSql, FromSql)]
    #[postgres(name = "aggregation_job_state")]
    pub enum AggregationJobState {
        #[postgres(name = "IN_PROGRESS")]
        InProgress,
        #[postgres(name = "FINISHED")]
        Finished,
        #[postgres(name = "ABANDONED")]
        Abandoned,
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

    impl<'a> TryFrom<&'a [u8]> for LeaseToken {
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

    /// Lease represents a time-constrained lease for exclusive access to some entity in Janus. It
    /// has an expiry after which it is no longer valid; another process can take a lease on the
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
        query_type: task::QueryType,
        vdaf: VdafInstance,
    }

    impl AcquiredAggregationJob {
        /// Creates a new [`AcquiredAggregationJob`].
        pub fn new(
            task_id: TaskId,
            aggregation_job_id: AggregationJobId,
            query_type: task::QueryType,
            vdaf: VdafInstance,
        ) -> Self {
            Self {
                task_id,
                aggregation_job_id,
                query_type,
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

        /// Returns the query type associated with this acquired aggregation job.
        pub fn query_type(&self) -> &task::QueryType {
            &self.query_type
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
        query_type: task::QueryType,
        vdaf: VdafInstance,
    }

    impl AcquiredCollectionJob {
        /// Creates a new [`AcquiredCollectionJob`].
        pub fn new(
            task_id: TaskId,
            collection_job_id: CollectionJobId,
            query_type: task::QueryType,
            vdaf: VdafInstance,
        ) -> Self {
            Self {
                task_id,
                collection_job_id,
                query_type,
                vdaf,
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

        /// Returns the query type associated with this acquired collection job.
        pub fn query_type(&self) -> &task::QueryType {
            &self.query_type
        }

        /// Returns the VDAF associated with this acquired collection job.
        pub fn vdaf(&self) -> &VdafInstance {
            &self.vdaf
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
        last_prep_step: Option<PrepareStep>,
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
            last_prep_step: Option<PrepareStep>,
            state: ReportAggregationState<SEED_SIZE, A>,
        ) -> Self {
            Self {
                task_id,
                aggregation_job_id,
                report_id,
                time,
                ord,
                last_prep_step,
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

        /// Returns a [`ReportMetadata`] corresponding to this report.
        pub fn report_metadata(&self) -> ReportMetadata {
            ReportMetadata::new(self.report_id, self.time)
        }

        /// Returns the order of this report aggregation in its aggregation job.
        pub fn ord(&self) -> u64 {
            self.ord
        }

        /// Returns the last preparation step returned by the Helper, if any.
        pub fn last_prep_step(&self) -> Option<&PrepareStep> {
            self.last_prep_step.as_ref()
        }

        /// Returns a new [`ReportAggregation`] corresponding to this report aggregation updated to
        /// have the given last preparation step.
        pub fn with_last_prep_step(self, last_prep_step: Option<PrepareStep>) -> Self {
            Self {
                last_prep_step,
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

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for ReportAggregation<SEED_SIZE, A>
    where
        A::PrepareState: PartialEq,
        A::PrepareMessage: PartialEq,
        A::PrepareShare: PartialEq,
        A::OutputShare: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.task_id == other.task_id
                && self.aggregation_job_id == other.aggregation_job_id
                && self.report_id == other.report_id
                && self.time == other.time
                && self.ord == other.ord
                && self.last_prep_step == other.last_prep_step
                && self.state == other.state
        }
    }

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for ReportAggregation<SEED_SIZE, A>
    where
        A::PrepareState: Eq,
        A::PrepareMessage: Eq,
        A::PrepareShare: Eq,
        A::OutputShare: Eq,
    {
    }

    /// ReportAggregationState represents the state of a single report aggregation. It corresponds
    /// to the REPORT_AGGREGATION_STATE enum in the schema, along with the state-specific data.
    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub enum ReportAggregationState<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
        Start,
        Waiting(
            #[derivative(Debug = "ignore")] A::PrepareState,
            #[derivative(Debug = "ignore")] Option<A::PrepareMessage>,
        ),
        Finished,
        Failed(ReportShareError),
    }

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
        ReportAggregationState<SEED_SIZE, A>
    {
        pub fn state_code(&self) -> ReportAggregationStateCode {
            match self {
                ReportAggregationState::Start => ReportAggregationStateCode::Start,
                ReportAggregationState::Waiting(_, _) => ReportAggregationStateCode::Waiting,
                ReportAggregationState::Finished => ReportAggregationStateCode::Finished,
                ReportAggregationState::Failed(_) => ReportAggregationStateCode::Failed,
            }
        }

        /// Returns the encoded values for the various messages which might be included in a
        /// ReportAggregationState. The order of returned values is preparation state, preparation
        /// message, output share, transition error.
        pub(super) fn encoded_values_from_state(&self) -> EncodedReportAggregationStateValues
        where
            A::PrepareState: Encode,
        {
            match self {
                ReportAggregationState::Start => EncodedReportAggregationStateValues::default(),
                ReportAggregationState::Waiting(prep_state, prep_msg) => {
                    EncodedReportAggregationStateValues {
                        prep_state: Some(prep_state.get_encoded()),
                        prep_msg: prep_msg.as_ref().map(Encode::get_encoded),
                        ..Default::default()
                    }
                }
                ReportAggregationState::Finished => EncodedReportAggregationStateValues::default(),
                ReportAggregationState::Failed(report_share_err) => {
                    EncodedReportAggregationStateValues {
                        report_share_err: Some(*report_share_err as i16),
                        ..Default::default()
                    }
                }
            }
        }
    }

    #[derive(Default)]
    pub(super) struct EncodedReportAggregationStateValues {
        pub(super) prep_state: Option<Vec<u8>>,
        pub(super) prep_msg: Option<Vec<u8>>,
        pub(super) report_share_err: Option<i16>,
    }

    // The private ReportAggregationStateCode exists alongside the public ReportAggregationState
    // because there is no apparent way to denote a Postgres enum literal without deriving
    // FromSql/ToSql on a Rust enum type, but it is not possible to derive FromSql/ToSql on a
    // non-C-style enum.
    #[derive(Debug, FromSql, ToSql)]
    #[postgres(name = "report_aggregation_state")]
    pub enum ReportAggregationStateCode {
        #[postgres(name = "START")]
        Start,
        #[postgres(name = "WAITING")]
        Waiting,
        #[postgres(name = "FINISHED")]
        Finished,
        #[postgres(name = "FAILED")]
        Failed,
    }

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for ReportAggregationState<SEED_SIZE, A>
    where
        A::PrepareState: PartialEq,
        A::PrepareMessage: PartialEq,
        A::PrepareShare: PartialEq,
        A::OutputShare: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (
                    Self::Waiting(lhs_prep_state, lhs_prep_msg),
                    Self::Waiting(rhs_prep_state, rhs_prep_msg),
                ) => lhs_prep_state == rhs_prep_state && lhs_prep_msg == rhs_prep_msg,
                (Self::Failed(lhs_report_share_err), Self::Failed(rhs_report_share_err)) => {
                    lhs_report_share_err == rhs_report_share_err
                }
                _ => core::mem::discriminant(self) == core::mem::discriminant(other),
            }
        }
    }

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for ReportAggregationState<SEED_SIZE, A>
    where
        A::PrepareState: Eq,
        A::PrepareMessage: Eq,
        A::PrepareShare: Eq,
        A::OutputShare: Eq,
    {
    }

    /// BatchAggregation corresponds to a row in the `batch_aggregations` table and represents the
    /// possibly-ongoing aggregation of the set of input shares that fall within the batch
    /// identified by `batch_identifier`. This is the finest-grained possible aggregate share we can
    /// emit for this task. The aggregate share constructed to service a collect or aggregate share
    /// request consists of one or more `BatchAggregation`s merged together.
    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub struct BatchAggregation<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    > {
        /// The task ID for this aggregation result.
        task_id: TaskId,
        /// The identifier of the batch being aggregated over.
        batch_identifier: Q::BatchIdentifier,
        /// The VDAF aggregation parameter used to prepare and accumulate input shares.
        #[derivative(Debug = "ignore")]
        aggregation_parameter: A::AggregationParam,
        /// The index of this batch aggregation among all batch aggregations for
        /// this (task_id, batch_identifier, aggregation_parameter).
        ord: u64,
        /// The current state of the batch aggregation.
        state: BatchAggregationState,
        /// The aggregate over all the input shares that have been prepared so far by this
        /// aggregator. Will only be None if there are no reports.
        #[derivative(Debug = "ignore")]
        aggregate_share: Option<A::AggregateShare>,
        /// The number of reports currently included in this aggregate sahre.
        report_count: u64,
        /// Checksum over the aggregated report shares, as described in §4.4.4.3.
        #[derivative(Debug = "ignore")]
        checksum: ReportIdChecksum,
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>>
        BatchAggregation<SEED_SIZE, Q, A>
    {
        /// Creates a new [`BatchAggregation`].
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            task_id: TaskId,
            batch_identifier: Q::BatchIdentifier,
            aggregation_parameter: A::AggregationParam,
            ord: u64,
            state: BatchAggregationState,
            aggregate_share: Option<A::AggregateShare>,
            report_count: u64,
            checksum: ReportIdChecksum,
        ) -> Self {
            Self {
                task_id,
                batch_identifier,
                aggregation_parameter,
                ord,
                state,
                aggregate_share,
                report_count,
                checksum,
            }
        }

        /// Returns the task ID associated with this batch aggregation.
        pub fn task_id(&self) -> &TaskId {
            &self.task_id
        }

        /// Gets the batch identifier included in this batch aggregation.
        ///
        /// This method would typically be used for code which is generic over the query type.
        /// Query-type specific code will typically call one of [`Self::batch_id`].
        pub fn batch_identifier(&self) -> &Q::BatchIdentifier {
            &self.batch_identifier
        }

        /// Returns the aggregation parameter associated with this batch aggregation.
        pub fn aggregation_parameter(&self) -> &A::AggregationParam {
            &self.aggregation_parameter
        }

        /// Returns the index of this batch aggregations among all batch aggregations for this
        /// (task_id, batch_identifier, aggregation_parameter).
        pub fn ord(&self) -> u64 {
            self.ord
        }

        /// Returns the current state associated with this batch aggregation.
        pub fn state(&self) -> &BatchAggregationState {
            &self.state
        }

        // Returns a [`BatchAggregation`] identical to this one, with the given batch aggregation
        // state.
        pub fn with_state(self, state: BatchAggregationState) -> Self {
            Self { state, ..self }
        }

        /// Returns the aggregate share associated with this batch aggregation.
        pub fn aggregate_share(&self) -> Option<&A::AggregateShare> {
            self.aggregate_share.as_ref()
        }

        /// Returns the report count associated with this batch aggregation.
        pub fn report_count(&self) -> u64 {
            self.report_count
        }

        /// Returns the checksum associated with this batch aggregation.
        pub fn checksum(&self) -> &ReportIdChecksum {
            &self.checksum
        }

        /// Returns a new [`BatchAggregation`] corresponding to the current batch aggregation merged
        /// with the given batch aggregation. Only uncollected batch aggregations may be merged.
        pub fn merged_with(self, other: &Self) -> Result<Self, Error> {
            if self.state() == &BatchAggregationState::Collected
                || other.state() == &BatchAggregationState::Collected
            {
                return Err(Error::AlreadyCollected);
            }

            let merged_aggregate_share = match (self.aggregate_share, other.aggregate_share()) {
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
                aggregate_share: merged_aggregate_share,
                report_count: self.report_count + other.report_count(),
                checksum: self.checksum.combined_with(other.checksum()),
                ..self
            })
        }
    }

    impl<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>>
        BatchAggregation<SEED_SIZE, FixedSize, A>
    {
        /// Gets the batch ID associated with this batch aggregation.
        pub fn batch_id(&self) -> &BatchId {
            self.batch_identifier()
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for BatchAggregation<SEED_SIZE, Q, A>
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
                && self.aggregate_share == other.aggregate_share
                && self.report_count == other.report_count
                && self.checksum == other.checksum
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for BatchAggregation<SEED_SIZE, Q, A>
    where
        A::AggregationParam: Eq,
        A::AggregateShare: Eq,
    {
    }

    /// Represents the state of a batch aggregation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, FromSql, ToSql)]
    #[postgres(name = "batch_aggregation_state")]
    pub enum BatchAggregationState {
        /// This batch aggregation has not been collected & permits further aggregation.
        #[postgres(name = "AGGREGATING")]
        Aggregating,
        /// This batch aggregation has been collected & no longer permits aggregation.
        #[postgres(name = "COLLECTED")]
        Collected,
    }

    /// CollectionJob represents a row in the `collection_jobs` table, used by leaders to represent
    /// running collection jobs and store the results of completed ones.
    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub struct CollectionJob<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    > {
        /// The task ID for this collection job.
        task_id: TaskId,
        /// The unique identifier for the collection job.
        collection_job_id: CollectionJobId,
        /// The batch interval covered by the collection job.
        batch_identifier: Q::BatchIdentifier,
        /// The VDAF aggregation parameter used to prepare and aggregate input shares.
        #[derivative(Debug = "ignore")]
        aggregation_parameter: A::AggregationParam,
        /// The current state of the collection job.
        state: CollectionJobState<SEED_SIZE, A>,
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>>
        CollectionJob<SEED_SIZE, Q, A>
    {
        /// Creates a new [`CollectionJob`].
        pub fn new(
            task_id: TaskId,
            collection_job_id: CollectionJobId,
            batch_identifier: Q::BatchIdentifier,
            aggregation_parameter: A::AggregationParam,
            state: CollectionJobState<SEED_SIZE, A>,
        ) -> Self {
            Self {
                task_id,
                collection_job_id,
                batch_identifier,
                aggregation_parameter,
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

        /// Gets the batch identifier associated with this collection job.
        ///
        /// This method would typically be used for code which is generic over the query type.
        /// Query-type specific code will typically call one of [`Self::batch_id`].
        pub fn batch_identifier(&self) -> &Q::BatchIdentifier {
            &self.batch_identifier
        }

        /// Returns the aggregation parameter associated with this collection job.
        pub fn aggregation_parameter(&self) -> &A::AggregationParam {
            &self.aggregation_parameter
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
        CollectionJob<SEED_SIZE, FixedSize, A>
    {
        /// Gets the batch ID associated with this collection job.
        pub fn batch_id(&self) -> &BatchId {
            self.batch_identifier()
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for CollectionJob<SEED_SIZE, Q, A>
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

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for CollectionJob<SEED_SIZE, Q, A>
    where
        A::AggregationParam: Eq,
        CollectionJobState<SEED_SIZE, A>: Eq,
    {
    }

    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub enum CollectionJobState<const SEED_SIZE: usize, A: vdaf::Aggregator<SEED_SIZE, 16>> {
        Start,
        Collectable,
        Finished {
            /// The number of reports included in this collection job.
            report_count: u64,
            /// The helper's encrypted aggregate share over the input shares in the interval.
            encrypted_helper_aggregate_share: HpkeCiphertext,
            /// The leader's aggregate share over the input shares in the interval.
            #[derivative(Debug = "ignore")]
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
                Self::Collectable => CollectionJobStateCode::Collectable,
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
                    Self::Collectable => "collectable",
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
                        encrypted_helper_aggregate_share: self_helper_agg_share,
                        leader_aggregate_share: self_leader_agg_share,
                    },
                    Self::Finished {
                        report_count: other_report_count,
                        encrypted_helper_aggregate_share: other_helper_agg_share,
                        leader_aggregate_share: other_leader_agg_share,
                    },
                ) => {
                    self_report_count == other_report_count
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

    #[derive(Debug, FromSql, ToSql)]
    #[postgres(name = "collection_job_state")]
    pub enum CollectionJobStateCode {
        #[postgres(name = "START")]
        Start,
        #[postgres(name = "COLLECTABLE")]
        Collectable,
        #[postgres(name = "FINISHED")]
        Finished,
        #[postgres(name = "ABANDONED")]
        Abandoned,
        #[postgres(name = "DELETED")]
        Deleted,
    }

    /// AggregateShareJob represents a row in the `aggregate_share_jobs` table, used by helpers to
    /// store the results of handling an AggregateShareReq from the leader.

    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub struct AggregateShareJob<
        const SEED_SIZE: usize,
        Q: QueryType,
        A: vdaf::Aggregator<SEED_SIZE, 16>,
    > {
        /// The task ID for this aggregate share.
        task_id: TaskId,
        /// The batch identifier for the batch covered by the aggregate share.
        batch_identifier: Q::BatchIdentifier,
        /// The VDAF aggregation parameter used to prepare and aggregate input shares.
        #[derivative(Debug = "ignore")]
        aggregation_parameter: A::AggregationParam,
        /// The aggregate share over the input shares in the interval.
        #[derivative(Debug = "ignore")]
        helper_aggregate_share: A::AggregateShare,
        /// The number of reports included in the aggregate share.
        report_count: u64,
        /// Checksum over the aggregated report shares, as described in §4.4.4.3.
        #[derivative(Debug = "ignore")]
        checksum: ReportIdChecksum,
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>>
        AggregateShareJob<SEED_SIZE, Q, A>
    {
        /// Creates a new [`AggregateShareJob`].
        pub fn new(
            task_id: TaskId,
            batch_identifier: Q::BatchIdentifier,
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
        pub fn batch_identifier(&self) -> &Q::BatchIdentifier {
            &self.batch_identifier
        }

        /// Gets the aggregation parameter associated with this aggregate share job.
        pub fn aggregation_parameter(&self) -> &A::AggregationParam {
            &self.aggregation_parameter
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
        AggregateShareJob<SEED_SIZE, FixedSize, A>
    {
        /// Gets the batch ID associated with this aggregate share job.
        pub fn batch_id(&self) -> &BatchId {
            self.batch_identifier()
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for AggregateShareJob<SEED_SIZE, Q, A>
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

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for AggregateShareJob<SEED_SIZE, Q, A>
    where
        A::AggregationParam: Eq,
        A::AggregateShare: Eq,
    {
    }

    /// An outstanding batch, which is a batch which has not yet started collection. Such a batch
    /// may have additional reports allocated to it. Only applies to fixed-size batches.
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

    /// Represents the state of a `Batch`.
    #[derive(Copy, Clone, Debug, FromSql, ToSql, PartialEq, Eq, Hash)]
    #[postgres(name = "batch_state")]
    pub enum BatchState {
        /// This batch can accept the creation of additional aggregation jobs.
        #[postgres(name = "OPEN")]
        Open,
        /// This batch can accept the creation of additional aggregation jobs, but will transition
        /// to state `CLOSED` once there are no outstanding aggregation jobs remaining.
        #[postgres(name = "CLOSING")]
        Closing,
        /// This batch can no longer accept the creation of additional aggregation jobs.
        #[postgres(name = "CLOSED")]
        Closed,
    }

    /// Represents the state of a given batch (and aggregation parameter).

    #[derive(Clone, Derivative)]
    #[derivative(Debug)]
    pub struct Batch<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> {
        task_id: TaskId,
        batch_identifier: Q::BatchIdentifier,
        #[derivative(Debug = "ignore")]
        aggregation_parameter: A::AggregationParam,
        state: BatchState,
        outstanding_aggregation_jobs: u64,
        client_timestamp_interval: Interval,
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>>
        Batch<SEED_SIZE, Q, A>
    {
        /// Creates a new [`Batch`].
        pub fn new(
            task_id: TaskId,
            batch_identifier: Q::BatchIdentifier,
            aggregation_parameter: A::AggregationParam,
            state: BatchState,
            outstanding_aggregation_jobs: u64,
            client_timestamp_interval: Interval,
        ) -> Self {
            Self {
                task_id,
                batch_identifier,
                aggregation_parameter,
                state,
                outstanding_aggregation_jobs,
                client_timestamp_interval,
            }
        }

        /// Gets the task ID associated with this batch.
        pub fn task_id(&self) -> &TaskId {
            &self.task_id
        }

        /// Gets the batch identifier associated with this batch.
        pub fn batch_identifier(&self) -> &Q::BatchIdentifier {
            &self.batch_identifier
        }

        /// Gets the aggregation parameter associated with this batch.
        pub fn aggregation_parameter(&self) -> &A::AggregationParam {
            &self.aggregation_parameter
        }

        /// Gets the state associated with this batch.
        pub fn state(&self) -> &BatchState {
            &self.state
        }

        /// Returns a new batch equivalent to the current batch, but with the given state.
        pub fn with_state(self, state: BatchState) -> Self {
            Self { state, ..self }
        }

        /// Gets the count of outstanding aggregation jobs associated with this batch.
        pub fn outstanding_aggregation_jobs(&self) -> u64 {
            self.outstanding_aggregation_jobs
        }

        /// Returns a new batch equivalent to the current batch, but with the given count of
        /// outstanding aggregation jobs.
        pub fn with_outstanding_aggregation_jobs(self, outstanding_aggregation_jobs: u64) -> Self {
            Self {
                outstanding_aggregation_jobs,
                ..self
            }
        }

        /// Gets the minimal interval of time spanned by the reports included in this batch.
        pub fn client_timestamp_interval(&self) -> &Interval {
            &self.client_timestamp_interval
        }

        /// Returns a new batch equivalent to the current batch, but with the given client timestamp
        /// interval.
        pub fn with_client_timestamp_interval(self, client_timestamp_interval: Interval) -> Self {
            Self {
                client_timestamp_interval,
                ..self
            }
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> PartialEq
        for Batch<SEED_SIZE, Q, A>
    where
        A::AggregationParam: PartialEq,
        Q::BatchIdentifier: PartialEq,
    {
        fn eq(&self, other: &Self) -> bool {
            self.task_id == other.task_id
                && self.batch_identifier == other.batch_identifier
                && self.aggregation_parameter == other.aggregation_parameter
                && self.state == other.state
                && self.outstanding_aggregation_jobs == other.outstanding_aggregation_jobs
                && self.client_timestamp_interval == other.client_timestamp_interval
        }
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> Eq
        for Batch<SEED_SIZE, Q, A>
    where
        A::AggregationParam: Eq,
        Q::BatchIdentifier: Eq,
    {
    }

    impl<const SEED_SIZE: usize, Q: QueryType, A: vdaf::Aggregator<SEED_SIZE, 16>> Hash
        for Batch<SEED_SIZE, Q, A>
    where
        A::AggregationParam: Hash,
    {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.task_id.hash(state);
            self.batch_identifier.hash(state);
            self.aggregation_parameter.hash(state);
            self.state.hash(state);
            self.outstanding_aggregation_jobs.hash(state);
            self.client_timestamp_interval.hash(state);
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
}

#[cfg(test)]
mod tests {
    // This function is only used when there are multiple supported versions.
    #[allow(unused_imports)]
    use crate::datastore::test_util::ephemeral_datastore_schema_version_by_downgrade;

    use crate::{
        datastore::{
            models::{
                AcquiredAggregationJob, AcquiredCollectionJob, AggregateShareJob, AggregationJob,
                AggregationJobState, Batch, BatchAggregation, BatchAggregationState, BatchState,
                CollectionJob, CollectionJobState, LeaderStoredReport, Lease, OutstandingBatch,
                ReportAggregation, ReportAggregationState, SqlInterval,
            },
            schema_versions_template,
            test_util::{
                ephemeral_datastore_schema_version, generate_aead_key, EphemeralDatastore,
            },
            Crypter, Datastore, Error, Transaction, SUPPORTED_SCHEMA_VERSIONS,
        },
        query_type::CollectableQueryType,
        task::{self, test_util::TaskBuilder, Task},
        test_util::noop_meter,
    };

    use assert_matches::assert_matches;
    use async_trait::async_trait;
    use chrono::NaiveDate;
    use futures::future::try_join_all;
    use janus_core::{
        hpke::{self, HpkeApplicationInfo, Label},
        task::{VdafInstance, PRIO3_VERIFY_KEY_LENGTH},
        test_util::{
            dummy_vdaf::{self, AggregateShare, AggregationParam},
            install_test_trace_subscriber, run_vdaf,
        },
        time::{Clock, DurationExt, MockClock, TimeExt},
    };
    use janus_messages::{
        query_type::{FixedSize, QueryType},
        AggregateShareAad, AggregationJobId, AggregationJobRound, BatchId, BatchSelector,
        CollectionJobId, Duration, Extension, ExtensionType, HpkeCiphertext, HpkeConfigId,
        Interval, PrepareStep, PrepareStepResult, ReportId, ReportIdChecksum, ReportMetadata,
        ReportShare, ReportShareError, Role, TaskId, Time,
    };
    use prio::{
        codec::{Decode, Encode},
        vdaf::prio3::{Prio3, Prio3Count},
    };
    use rand::{distributions::Standard, random, thread_rng, Rng};
    use std::{
        collections::{HashMap, HashSet},
        iter,
        ops::RangeInclusive,
        sync::Arc,
        time::Duration as StdDuration,
    };
    use tokio::time::timeout;

    #[test]
    fn check_supported_versions() {
        if SUPPORTED_SCHEMA_VERSIONS[0] != *SUPPORTED_SCHEMA_VERSIONS.iter().max().unwrap() {
            panic!("the latest supported schema version must be first in the list");
        }
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn reject_unsupported_schema_version(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let error = Datastore::new_with_supported_versions(
            ephemeral_datastore.pool(),
            ephemeral_datastore.crypter(),
            MockClock::default(),
            &noop_meter(),
            &[0],
        )
        .await
        .unwrap_err();

        assert_matches!(error, Error::DbState(_));
    }

    #[rstest::rstest]
    #[case(ephemeral_datastore_schema_version(i64::MAX))]
    #[tokio::test]
    async fn down_migrations(
        #[future(awt)]
        #[case]
        ephemeral_datastore: EphemeralDatastore,
    ) {
        ephemeral_datastore.downgrade(0).await;
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_task(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        // Insert tasks, check that they can be retrieved by ID.
        let mut want_tasks = HashMap::new();
        for (vdaf, role) in [
            (VdafInstance::Prio3Count, Role::Leader),
            (VdafInstance::Prio3CountVec { length: 8 }, Role::Leader),
            (VdafInstance::Prio3CountVec { length: 64 }, Role::Helper),
            (VdafInstance::Prio3Sum { bits: 64 }, Role::Helper),
            (VdafInstance::Prio3Sum { bits: 32 }, Role::Helper),
            (
                VdafInstance::Prio3Histogram {
                    buckets: Vec::from([0, 100, 200, 400]),
                },
                Role::Leader,
            ),
            (
                VdafInstance::Prio3Histogram {
                    buckets: Vec::from([0, 25, 50, 75, 100]),
                },
                Role::Leader,
            ),
            (VdafInstance::Poplar1 { bits: 8 }, Role::Helper),
            (VdafInstance::Poplar1 { bits: 64 }, Role::Helper),
        ] {
            let task = TaskBuilder::new(
                task::QueryType::FixedSize { max_batch_size: 10 },
                vdaf,
                role,
            )
            .with_report_expiry_age(Some(Duration::from_seconds(3600)))
            .build();
            want_tasks.insert(*task.id(), task.clone());

            let err = ds
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move { tx.delete_task(task.id()).await })
                })
                .await
                .unwrap_err();
            assert_matches!(err, Error::MutationTargetNotFound);

            let retrieved_task = ds
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move { tx.get_task(task.id()).await })
                })
                .await
                .unwrap();
            assert_eq!(None, retrieved_task);

            ds.put_task(&task).await.unwrap();

            let retrieved_task = ds
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move { tx.get_task(task.id()).await })
                })
                .await
                .unwrap();
            assert_eq!(Some(&task), retrieved_task.as_ref());

            ds.run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.delete_task(task.id()).await })
            })
            .await
            .unwrap();

            let retrieved_task = ds
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move { tx.get_task(task.id()).await })
                })
                .await
                .unwrap();
            assert_eq!(None, retrieved_task);

            let err = ds
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move { tx.delete_task(task.id()).await })
                })
                .await
                .unwrap_err();
            assert_matches!(err, Error::MutationTargetNotFound);

            // Rewrite & retrieve the task again, to test that the delete is "clean" in the sense
            // that it deletes all task-related data (& therefore does not conflict with a later
            // write to the same task_id).
            ds.put_task(&task).await.unwrap();

            let retrieved_task = ds
                .run_tx(|tx| {
                    let task = task.clone();
                    Box::pin(async move { tx.get_task(task.id()).await })
                })
                .await
                .unwrap();
            assert_eq!(Some(task), retrieved_task);
        }

        let got_tasks: HashMap<TaskId, Task> = ds
            .run_tx(|tx| Box::pin(async move { tx.get_tasks().await }))
            .await
            .unwrap()
            .into_iter()
            .map(|task| (*task.id(), task))
            .collect();
        assert_eq!(want_tasks, got_tasks);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn get_task_metrics(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const REPORT_COUNT: usize = 5;
        const REPORT_AGGREGATION_COUNT: usize = 2;
        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);

        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build();
                    let other_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();

                    let reports: Vec<_> = iter::repeat_with(|| {
                        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP)
                    })
                    .take(REPORT_COUNT)
                    .collect();
                    let expired_reports: Vec<_> = iter::repeat_with(|| {
                        LeaderStoredReport::new_dummy(
                            *task.id(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(2))
                                .unwrap(),
                        )
                    })
                    .take(REPORT_COUNT)
                    .collect();
                    let other_reports: Vec<_> = iter::repeat_with(|| {
                        LeaderStoredReport::new_dummy(
                            *other_task.id(),
                            Time::from_seconds_since_epoch(0),
                        )
                    })
                    .take(22)
                    .collect();

                    let aggregation_job = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        random(),
                        AggregationParam(0),
                        random(),
                        Interval::new(
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(1))
                                .unwrap(),
                            Duration::from_seconds(2),
                        )
                        .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    );
                    let expired_aggregation_job =
                        AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            random(),
                            AggregationParam(0),
                            random(),
                            Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(2))
                                    .unwrap(),
                                Duration::from_seconds(1),
                            )
                            .unwrap(),
                            AggregationJobState::InProgress,
                            AggregationJobRound::from(0),
                        );
                    let other_aggregation_job: AggregationJob<0, _, dummy_vdaf::Vdaf> =
                        AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                            *other_task.id(),
                            random(),
                            AggregationParam(0),
                            random(),
                            Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(1))
                                    .unwrap(),
                                Duration::from_seconds(2),
                            )
                            .unwrap(),
                            AggregationJobState::InProgress,
                            AggregationJobRound::from(0),
                        );

                    let report_aggregations: Vec<_> = reports
                        .iter()
                        .take(REPORT_AGGREGATION_COUNT)
                        .enumerate()
                        .map(|(ord, report)| {
                            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                                *task.id(),
                                *aggregation_job.id(),
                                *report.metadata().id(),
                                *report.metadata().time(),
                                ord.try_into().unwrap(),
                                None,
                                ReportAggregationState::Start,
                            )
                        })
                        .collect();
                    let expired_report_aggregations: Vec<_> = expired_reports
                        .iter()
                        .take(REPORT_AGGREGATION_COUNT)
                        .enumerate()
                        .map(|(ord, report)| {
                            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                                *task.id(),
                                *expired_aggregation_job.id(),
                                *report.metadata().id(),
                                *report.metadata().time(),
                                ord.try_into().unwrap(),
                                None,
                                ReportAggregationState::Start,
                            )
                        })
                        .collect();
                    let other_report_aggregations: Vec<_> = other_reports
                        .iter()
                        .take(13)
                        .enumerate()
                        .map(|(ord, report)| {
                            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                                *other_task.id(),
                                *other_aggregation_job.id(),
                                *report.metadata().id(),
                                *report.metadata().time(),
                                ord.try_into().unwrap(),
                                None,
                                ReportAggregationState::Start,
                            )
                        })
                        .collect();

                    tx.put_task(&task).await?;
                    tx.put_task(&other_task).await?;
                    try_join_all(
                        reports
                            .iter()
                            .chain(expired_reports.iter())
                            .chain(other_reports.iter())
                            .map(|report| async move {
                                tx.put_client_report(&dummy_vdaf::Vdaf::new(), report).await
                            }),
                    )
                    .await?;
                    tx.put_aggregation_job(&aggregation_job).await?;
                    tx.put_aggregation_job(&expired_aggregation_job).await?;
                    tx.put_aggregation_job(&other_aggregation_job).await?;
                    try_join_all(
                        report_aggregations
                            .iter()
                            .chain(expired_report_aggregations.iter())
                            .chain(other_report_aggregations.iter())
                            .map(|report_aggregation| async move {
                                tx.put_report_aggregation(report_aggregation).await
                            }),
                    )
                    .await?;

                    Ok(*task.id())
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        ds.run_tx(|tx| {
            Box::pin(async move {
                // Verify we get the correct results when we check metrics on our target task.
                assert_eq!(
                    tx.get_task_metrics(&task_id).await.unwrap(),
                    Some((
                        REPORT_COUNT.try_into().unwrap(),
                        REPORT_AGGREGATION_COUNT.try_into().unwrap()
                    ))
                );

                // Verify that we get None if we ask about a task that doesn't exist.
                assert_eq!(tx.get_task_metrics(&random()).await.unwrap(), None);

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn get_task_ids(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        ds.run_tx(|tx| {
            Box::pin(async move {
                const TOTAL_TASK_ID_COUNT: usize = 20;
                let tasks: Vec<_> = iter::repeat_with(|| {
                    TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build()
                })
                .take(TOTAL_TASK_ID_COUNT)
                .collect();

                let mut task_ids: Vec<_> = tasks.iter().map(Task::id).cloned().collect();
                task_ids.sort();

                try_join_all(tasks.iter().map(|task| tx.put_task(task))).await?;

                for (i, lower_bound) in iter::once(None)
                    .chain(task_ids.iter().cloned().map(Some))
                    .enumerate()
                {
                    let got_task_ids = tx.get_task_ids(lower_bound).await?;
                    assert_eq!(&got_task_ids, &task_ids[i..]);
                }

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_report(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;
        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        let report_expiry_age = clock
            .now()
            .difference(&OLDEST_ALLOWED_REPORT_TIMESTAMP)
            .unwrap();

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_report_expiry_age(Some(report_expiry_age))
        .build();

        ds.run_tx(|tx| {
            let task = task.clone();
            Box::pin(async move { tx.put_task(&task).await })
        })
        .await
        .unwrap();

        let report_id = random();
        let report: LeaderStoredReport<0, dummy_vdaf::Vdaf> = LeaderStoredReport::new(
            *task.id(),
            ReportMetadata::new(report_id, OLDEST_ALLOWED_REPORT_TIMESTAMP),
            (), // public share
            Vec::from([
                Extension::new(ExtensionType::Tbd, Vec::from("extension_data_0")),
                Extension::new(ExtensionType::Tbd, Vec::from("extension_data_1")),
            ]),
            dummy_vdaf::InputShare::default(), // leader input share
            /* Dummy ciphertext for the helper share */
            HpkeCiphertext::new(
                HpkeConfigId::from(13),
                Vec::from("encapsulated_context_1"),
                Vec::from("payload_1"),
            ),
        );

        // Write a report twice to prove it is idempotent
        for _ in 0..2 {
            ds.run_tx(|tx| {
                let report = report.clone();
                Box::pin(async move {
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report)
                        .await
                })
            })
            .await
            .unwrap();

            let retrieved_report = ds
                .run_tx(|tx| {
                    let task_id = *report.task_id();
                    Box::pin(async move {
                        tx.get_client_report::<0, dummy_vdaf::Vdaf>(
                            &dummy_vdaf::Vdaf::new(),
                            &task_id,
                            &report_id,
                        )
                        .await
                    })
                })
                .await
                .unwrap()
                .unwrap();

            assert_eq!(report, retrieved_report);
        }

        // Try to write a different report with the same ID, and verify we get the expected error.
        let result = ds
            .run_tx(|tx| {
                let task_id = *report.task_id();
                Box::pin(async move {
                    tx.put_client_report(
                        &dummy_vdaf::Vdaf::new(),
                        &LeaderStoredReport::<0, dummy_vdaf::Vdaf>::new(
                            task_id,
                            ReportMetadata::new(report_id, Time::from_seconds_since_epoch(54321)),
                            (), // public share
                            Vec::from([
                                Extension::new(ExtensionType::Tbd, Vec::from("extension_data_2")),
                                Extension::new(ExtensionType::Tbd, Vec::from("extension_data_3")),
                            ]),
                            dummy_vdaf::InputShare::default(), // leader input share
                            /* Dummy ciphertext for the helper share */
                            HpkeCiphertext::new(
                                HpkeConfigId::from(14),
                                Vec::from("encapsulated_context_2"),
                                Vec::from("payload_2"),
                            ),
                        ),
                    )
                    .await
                })
            })
            .await;
        assert_matches!(result, Err(Error::MutationTargetAlreadyExists));

        // Advance the clock so that the report is expired, and verify that it does not exist.
        clock.advance(&Duration::from_seconds(1));
        let retrieved_report = ds
            .run_tx(|tx| {
                let task_id = *report.task_id();
                Box::pin(async move {
                    tx.get_client_report::<0, dummy_vdaf::Vdaf>(
                        &dummy_vdaf::Vdaf::new(),
                        &task_id,
                        &report_id,
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(None, retrieved_report);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn report_not_found(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        let rslt = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.get_client_report(&dummy_vdaf::Vdaf::new(), &random(), &random())
                        .await
                })
            })
            .await
            .unwrap();

        assert_eq!(rslt, None);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn get_unaggregated_client_report_ids_for_task(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;
        let report_interval = Interval::new(
            OLDEST_ALLOWED_REPORT_TIMESTAMP
                .sub(&Duration::from_seconds(1))
                .unwrap(),
            Duration::from_seconds(2),
        )
        .unwrap();
        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();
        let unrelated_task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .build();

        let first_unaggregated_report =
            LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let second_unaggregated_report =
            LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let expired_report = LeaderStoredReport::new_dummy(
            *task.id(),
            OLDEST_ALLOWED_REPORT_TIMESTAMP
                .sub(&Duration::from_seconds(1))
                .unwrap(),
        );
        let aggregated_report =
            LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let unrelated_report =
            LeaderStoredReport::new_dummy(*unrelated_task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Set up state.
        ds.run_tx(|tx| {
            let task = task.clone();
            let unrelated_task = unrelated_task.clone();
            let first_unaggregated_report = first_unaggregated_report.clone();
            let second_unaggregated_report = second_unaggregated_report.clone();
            let expired_report = expired_report.clone();
            let aggregated_report = aggregated_report.clone();
            let unrelated_report = unrelated_report.clone();

            Box::pin(async move {
                tx.put_task(&task).await?;
                tx.put_task(&unrelated_task).await?;

                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &first_unaggregated_report)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &second_unaggregated_report)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &expired_report)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &aggregated_report)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &unrelated_report)
                    .await?;

                // Mark aggregated_report as aggregated.
                tx.mark_report_aggregated(task.id(), aggregated_report.metadata().id())
                    .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Verify that we can acquire both unaggregated reports.
        let got_reports = HashSet::from_iter(
            ds.run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    // At this point, first_unaggregated_report and second_unaggregated_report are
                    // both unaggregated.
                    assert!(
                        tx.interval_has_unaggregated_reports(task.id(), &report_interval)
                            .await?
                    );

                    tx.get_unaggregated_client_report_ids_for_task(task.id())
                        .await
                })
            })
            .await
            .unwrap(),
        );

        assert_eq!(
            got_reports,
            HashSet::from([
                (
                    *first_unaggregated_report.metadata().id(),
                    *first_unaggregated_report.metadata().time(),
                ),
                (
                    *second_unaggregated_report.metadata().id(),
                    *second_unaggregated_report.metadata().time(),
                ),
            ]),
        );

        // Verify that attempting to acquire again does not return the reports.
        let got_reports = HashSet::<(ReportId, Time)>::from_iter(
            ds.run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    // At this point, all reports have started aggregation.
                    assert!(
                        !tx.interval_has_unaggregated_reports(task.id(), &report_interval)
                            .await?
                    );

                    tx.get_unaggregated_client_report_ids_for_task(task.id())
                        .await
                })
            })
            .await
            .unwrap(),
        );

        assert!(got_reports.is_empty());

        // Mark one report un-aggregated.
        ds.run_tx(|tx| {
            let (task, first_unaggregated_report) =
                (task.clone(), first_unaggregated_report.clone());
            Box::pin(async move {
                tx.mark_reports_unaggregated(
                    task.id(),
                    &[*first_unaggregated_report.metadata().id()],
                )
                .await
            })
        })
        .await
        .unwrap();

        // Verify that we can retrieve the un-aggregated report again.
        let got_reports = HashSet::from_iter(
            ds.run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move {
                    // At this point, first_unaggregated_report is unaggregated.
                    assert!(
                        tx.interval_has_unaggregated_reports(task.id(), &report_interval)
                            .await?
                    );

                    tx.get_unaggregated_client_report_ids_for_task(task.id())
                        .await
                })
            })
            .await
            .unwrap(),
        );

        assert_eq!(
            got_reports,
            HashSet::from([(
                *first_unaggregated_report.metadata().id(),
                *first_unaggregated_report.metadata().time(),
            ),]),
        );
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn count_client_reports_for_interval(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();
        let unrelated_task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();
        let no_reports_task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();

        let expired_report_in_interval = LeaderStoredReport::new_dummy(
            *task.id(),
            OLDEST_ALLOWED_REPORT_TIMESTAMP
                .sub(&Duration::from_seconds(1))
                .unwrap(),
        );
        let first_report_in_interval =
            LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let second_report_in_interval = LeaderStoredReport::new_dummy(
            *task.id(),
            OLDEST_ALLOWED_REPORT_TIMESTAMP
                .add(&Duration::from_seconds(1))
                .unwrap(),
        );
        let report_outside_interval = LeaderStoredReport::new_dummy(
            *task.id(),
            OLDEST_ALLOWED_REPORT_TIMESTAMP
                .add(&Duration::from_seconds(10000))
                .unwrap(),
        );
        let report_for_other_task =
            LeaderStoredReport::new_dummy(*unrelated_task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);

        // Set up state.
        ds.run_tx(|tx| {
            let task = task.clone();
            let unrelated_task = unrelated_task.clone();
            let no_reports_task = no_reports_task.clone();
            let expired_report_in_interval = expired_report_in_interval.clone();
            let first_report_in_interval = first_report_in_interval.clone();
            let second_report_in_interval = second_report_in_interval.clone();
            let report_outside_interval = report_outside_interval.clone();
            let report_for_other_task = report_for_other_task.clone();

            Box::pin(async move {
                tx.put_task(&task).await?;
                tx.put_task(&unrelated_task).await?;
                tx.put_task(&no_reports_task).await?;

                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &expired_report_in_interval)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &first_report_in_interval)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &second_report_in_interval)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report_outside_interval)
                    .await?;
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report_for_other_task)
                    .await?;

                Ok(())
            })
        })
        .await
        .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        let (report_count, no_reports_task_report_count) = ds
            .run_tx(|tx| {
                let (task, no_reports_task) = (task.clone(), no_reports_task.clone());
                Box::pin(async move {
                    let report_count = tx
                        .count_client_reports_for_interval(
                            task.id(),
                            &Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(1))
                                    .unwrap(),
                                Duration::from_seconds(5),
                            )
                            .unwrap(),
                        )
                        .await?;

                    let no_reports_task_report_count = tx
                        .count_client_reports_for_interval(
                            no_reports_task.id(),
                            &Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(1))
                                    .unwrap(),
                                Duration::from_seconds(5),
                            )
                            .unwrap(),
                        )
                        .await?;

                    Ok((report_count, no_reports_task_report_count))
                })
            })
            .await
            .unwrap();
        assert_eq!(report_count, 2);
        assert_eq!(no_reports_task_report_count, 0);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn count_client_reports_for_batch_id(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();
        let unrelated_task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();

        // Set up state.
        let batch_id = ds
            .run_tx(|tx| {
                let (task, unrelated_task) = (task.clone(), unrelated_task.clone());

                Box::pin(async move {
                    tx.put_task(&task).await?;
                    tx.put_task(&unrelated_task).await?;

                    // Create a batch for the first task containing two reports, which has started
                    // aggregation twice with two different aggregation parameters.
                    let batch_id = random();
                    let expired_report = LeaderStoredReport::new_dummy(
                        *task.id(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(2))
                            .unwrap(),
                    );
                    let report_0 =
                        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
                    let report_1 = LeaderStoredReport::new_dummy(
                        *task.id(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .add(&Duration::from_seconds(1))
                            .unwrap(),
                    );

                    let expired_aggregation_job =
                        AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            random(),
                            AggregationParam(22),
                            batch_id,
                            Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(2))
                                    .unwrap(),
                                Duration::from_seconds(1),
                            )
                            .unwrap(),
                            AggregationJobState::InProgress,
                            AggregationJobRound::from(0),
                        );
                    let expired_report_aggregation = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *expired_aggregation_job.id(),
                        *expired_report.metadata().id(),
                        *expired_report.metadata().time(),
                        0,
                        None,
                        ReportAggregationState::Start,
                    );

                    let aggregation_job_0 = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        random(),
                        AggregationParam(22),
                        batch_id,
                        Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(2))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    );
                    let aggregation_job_0_report_aggregation_0 =
                        ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            *aggregation_job_0.id(),
                            *report_0.metadata().id(),
                            *report_0.metadata().time(),
                            1,
                            None,
                            ReportAggregationState::Start,
                        );
                    let aggregation_job_0_report_aggregation_1 =
                        ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            *aggregation_job_0.id(),
                            *report_1.metadata().id(),
                            *report_1.metadata().time(),
                            2,
                            None,
                            ReportAggregationState::Start,
                        );

                    let aggregation_job_1 = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        random(),
                        AggregationParam(23),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    );
                    let aggregation_job_1_report_aggregation_0 =
                        ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            *aggregation_job_1.id(),
                            *report_0.metadata().id(),
                            *report_0.metadata().time(),
                            0,
                            None,
                            ReportAggregationState::Start,
                        );
                    let aggregation_job_1_report_aggregation_1 =
                        ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            *aggregation_job_1.id(),
                            *report_1.metadata().id(),
                            *report_1.metadata().time(),
                            1,
                            None,
                            ReportAggregationState::Start,
                        );

                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &expired_report)
                        .await?;
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report_0)
                        .await?;
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report_1)
                        .await?;

                    tx.put_aggregation_job(&expired_aggregation_job).await?;
                    tx.put_report_aggregation(&expired_report_aggregation)
                        .await?;

                    tx.put_aggregation_job(&aggregation_job_0).await?;
                    tx.put_report_aggregation(&aggregation_job_0_report_aggregation_0)
                        .await?;
                    tx.put_report_aggregation(&aggregation_job_0_report_aggregation_1)
                        .await?;

                    tx.put_aggregation_job(&aggregation_job_1).await?;
                    tx.put_report_aggregation(&aggregation_job_1_report_aggregation_0)
                        .await?;
                    tx.put_report_aggregation(&aggregation_job_1_report_aggregation_1)
                        .await?;

                    Ok(batch_id)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        let report_count = ds
            .run_tx(|tx| {
                let task_id = *task.id();
                Box::pin(async move {
                    tx.count_client_reports_for_batch_id(&task_id, &batch_id)
                        .await
                })
            })
            .await
            .unwrap();
        assert_eq!(report_count, 2);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_report_share(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .build();
        let report_share = ReportShare::new(
            ReportMetadata::new(
                ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                Time::from_seconds_since_epoch(12345),
            ),
            Vec::from("public_share"),
            HpkeCiphertext::new(
                HpkeConfigId::from(12),
                Vec::from("encapsulated_context_0"),
                Vec::from("payload_0"),
            ),
        );

        ds.run_tx(|tx| {
            let (task, report_share) = (task.clone(), report_share.clone());
            Box::pin(async move {
                tx.put_task(&task).await?;
                tx.put_report_share(task.id(), &report_share).await?;

                Ok(())
            })
        })
        .await
        .unwrap();

        let (got_task_id, got_extensions, got_leader_input_share, got_helper_input_share) = ds
            .run_tx(|tx| {
                let report_share_metadata = report_share.metadata().clone();
                Box::pin(async move {
                    let row = tx
                        .query_one(
                            "SELECT
                                tasks.task_id,
                                client_reports.report_id,
                                client_reports.client_timestamp,
                                client_reports.extensions,
                                client_reports.leader_input_share,
                                client_reports.helper_encrypted_input_share
                            FROM client_reports JOIN tasks ON tasks.id = client_reports.task_id
                            WHERE report_id = $1 AND client_timestamp = $2",
                            &[
                                /* report_id */ &report_share_metadata.id().as_ref(),
                                /* client_timestamp */
                                &report_share_metadata.time().as_naive_date_time()?,
                            ],
                        )
                        .await?;

                    let task_id = TaskId::get_decoded(row.get("task_id"))?;

                    let maybe_extensions: Option<Vec<u8>> = row.get("extensions");
                    let maybe_leader_input_share: Option<Vec<u8>> = row.get("leader_input_share");
                    let maybe_helper_input_share: Option<Vec<u8>> =
                        row.get("helper_encrypted_input_share");

                    Ok((
                        task_id,
                        maybe_extensions,
                        maybe_leader_input_share,
                        maybe_helper_input_share,
                    ))
                })
            })
            .await
            .unwrap();

        assert_eq!(task.id(), &got_task_id);
        assert!(got_extensions.is_none());
        assert!(got_leader_input_share.is_none());
        assert!(got_helper_input_share.is_none());

        // Put the same report share again. This should not cause an error.
        ds.run_tx(|tx| {
            let (task_id, report_share) = (*task.id(), report_share.clone());
            Box::pin(async move {
                tx.put_report_share(&task_id, &report_share).await.unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_aggregation_job(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        // We use a dummy VDAF & fixed-size task for this test, to better exercise the
        // serialization/deserialization roundtrip of the batch_identifier & aggregation_param.
        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();
        let batch_id = random();
        let leader_aggregation_job = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *task.id(),
            random(),
            AggregationParam(23),
            batch_id,
            Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1)).unwrap(),
            AggregationJobState::InProgress,
            AggregationJobRound::from(0),
        );
        let helper_aggregation_job = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *task.id(),
            random(),
            AggregationParam(23),
            random(),
            Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1)).unwrap(),
            AggregationJobState::InProgress,
            AggregationJobRound::from(0),
        );

        ds.run_tx(|tx| {
            let (task, leader_aggregation_job, helper_aggregation_job) = (
                task.clone(),
                leader_aggregation_job.clone(),
                helper_aggregation_job.clone(),
            );
            Box::pin(async move {
                tx.put_task(&task).await.unwrap();
                tx.put_aggregation_job(&leader_aggregation_job)
                    .await
                    .unwrap();
                tx.put_aggregation_job(&helper_aggregation_job)
                    .await
                    .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        let (got_leader_aggregation_job, got_helper_aggregation_job) = ds
            .run_tx(|tx| {
                let (leader_aggregation_job, helper_aggregation_job) = (
                    leader_aggregation_job.clone(),
                    helper_aggregation_job.clone(),
                );
                Box::pin(async move {
                    Ok((
                        tx.get_aggregation_job(
                            leader_aggregation_job.task_id(),
                            leader_aggregation_job.id(),
                        )
                        .await
                        .unwrap(),
                        tx.get_aggregation_job(
                            helper_aggregation_job.task_id(),
                            helper_aggregation_job.id(),
                        )
                        .await
                        .unwrap(),
                    ))
                })
            })
            .await
            .unwrap();
        assert_eq!(
            Some(&leader_aggregation_job),
            got_leader_aggregation_job.as_ref()
        );
        assert_eq!(
            Some(&helper_aggregation_job),
            got_helper_aggregation_job.as_ref()
        );

        let new_leader_aggregation_job = leader_aggregation_job
            .clone()
            .with_state(AggregationJobState::Finished);
        let new_helper_aggregation_job =
            helper_aggregation_job.with_last_continue_request_hash([3; 32]);
        ds.run_tx(|tx| {
            let (new_leader_aggregation_job, new_helper_aggregation_job) = (
                new_leader_aggregation_job.clone(),
                new_helper_aggregation_job.clone(),
            );
            Box::pin(async move {
                tx.update_aggregation_job(&new_leader_aggregation_job)
                    .await
                    .unwrap();
                tx.update_aggregation_job(&new_helper_aggregation_job)
                    .await
                    .unwrap();

                Ok(())
            })
        })
        .await
        .unwrap();

        let (got_leader_aggregation_job, got_helper_aggregation_job) = ds
            .run_tx(|tx| {
                let (new_leader_aggregation_job, new_helper_aggregation_job) = (
                    new_leader_aggregation_job.clone(),
                    new_helper_aggregation_job.clone(),
                );
                Box::pin(async move {
                    Ok((
                        tx.get_aggregation_job(
                            new_leader_aggregation_job.task_id(),
                            new_leader_aggregation_job.id(),
                        )
                        .await
                        .unwrap(),
                        tx.get_aggregation_job(
                            new_helper_aggregation_job.task_id(),
                            new_helper_aggregation_job.id(),
                        )
                        .await
                        .unwrap(),
                    ))
                })
            })
            .await
            .unwrap();
        assert_eq!(
            Some(new_leader_aggregation_job.clone()),
            got_leader_aggregation_job
        );
        assert_eq!(
            Some(new_helper_aggregation_job.clone()),
            got_helper_aggregation_job
        );

        // Trying to write an aggregation job again should fail.
        let new_leader_aggregation_job = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *task.id(),
            *leader_aggregation_job.id(),
            AggregationParam(24),
            batch_id,
            Interval::new(
                Time::from_seconds_since_epoch(2345),
                Duration::from_seconds(6789),
            )
            .unwrap(),
            AggregationJobState::InProgress,
            AggregationJobRound::from(0),
        );
        ds.run_tx(|tx| {
            let new_leader_aggregation_job = new_leader_aggregation_job.clone();
            Box::pin(async move {
                let error = tx
                    .put_aggregation_job(&new_leader_aggregation_job)
                    .await
                    .unwrap_err();
                assert_matches!(error, Error::MutationTargetAlreadyExists);

                Ok(())
            })
        })
        .await
        .unwrap();

        // Advance the clock to verify that the aggregation jobs have expired & are no longer
        // returned.
        clock.advance(&Duration::from_seconds(2));
        let (got_leader_aggregation_job, got_helper_aggregation_job) = ds
            .run_tx(|tx| {
                let (new_leader_aggregation_job, new_helper_aggregation_job) = (
                    new_leader_aggregation_job.clone(),
                    new_helper_aggregation_job.clone(),
                );
                Box::pin(async move {
                    Ok((
                        tx.get_aggregation_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                            new_leader_aggregation_job.task_id(),
                            new_leader_aggregation_job.id(),
                        )
                        .await
                        .unwrap(),
                        tx.get_aggregation_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                            new_helper_aggregation_job.task_id(),
                            new_helper_aggregation_job.id(),
                        )
                        .await
                        .unwrap(),
                    ))
                })
            })
            .await
            .unwrap();
        assert_eq!(None, got_leader_aggregation_job);
        assert_eq!(None, got_helper_aggregation_job);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn aggregation_job_acquire_release(ephemeral_datastore: EphemeralDatastore) {
        // Setup: insert a few aggregation jobs.
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        const LEASE_DURATION: StdDuration = StdDuration::from_secs(300);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        const AGGREGATION_JOB_COUNT: usize = 10;
        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();
        let mut aggregation_job_ids: Vec<_> = thread_rng()
            .sample_iter(Standard)
            .take(AGGREGATION_JOB_COUNT)
            .collect();
        aggregation_job_ids.sort();

        ds.run_tx(|tx| {
            let (task, aggregation_job_ids) = (task.clone(), aggregation_job_ids.clone());
            Box::pin(async move {
                // Write a few aggregation jobs we expect to be able to retrieve with
                // acquire_incomplete_aggregation_jobs().
                tx.put_task(&task).await?;
                try_join_all(aggregation_job_ids.into_iter().map(|aggregation_job_id| {
                    let task_id = *task.id();
                    async move {
                        tx.put_aggregation_job(&AggregationJob::<
                            PRIO3_VERIFY_KEY_LENGTH,
                            FixedSize,
                            Prio3Count,
                        >::new(
                            task_id,
                            aggregation_job_id,
                            (),
                            random(),
                            Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(LEASE_DURATION.as_secs()))
                                    .unwrap()
                                    .add(&Duration::from_seconds(LEASE_DURATION.as_secs()))
                                    .unwrap(),
                                Duration::from_seconds(1),
                            )
                            .unwrap(),
                            AggregationJobState::InProgress,
                            AggregationJobRound::from(0),
                        ))
                        .await
                    }
                }))
                .await?;

                // Write an aggregation job that is finished. We don't want to retrieve this one.
                tx.put_aggregation_job(&AggregationJob::<
                    PRIO3_VERIFY_KEY_LENGTH,
                    FixedSize,
                    Prio3Count,
                >::new(
                    *task.id(),
                    random(),
                    (),
                    random(),
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::Finished,
                    AggregationJobRound::from(1),
                ))
                .await?;

                // Write an expired aggregation job. We don't want to retrieve this one, either.
                tx.put_aggregation_job(&AggregationJob::<
                    PRIO3_VERIFY_KEY_LENGTH,
                    FixedSize,
                    Prio3Count,
                >::new(
                    *task.id(),
                    random(),
                    (),
                    random(),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobRound::from(0),
                ))
                .await?;

                // Write an aggregation job for a task that we are taking on the helper role for.
                // We don't want to retrieve this one, either.
                let helper_task = TaskBuilder::new(
                    task::QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Prio3Count,
                    Role::Helper,
                )
                .build();
                tx.put_task(&helper_task).await?;
                tx.put_aggregation_job(&AggregationJob::<
                    PRIO3_VERIFY_KEY_LENGTH,
                    FixedSize,
                    Prio3Count,
                >::new(
                    *helper_task.id(),
                    random(),
                    (),
                    random(),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobRound::from(0),
                ))
                .await
            })
        })
        .await
        .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Run: run several transactions that all call acquire_incomplete_aggregation_jobs
        // concurrently. (We do things concurrently in an attempt to make sure the
        // mutual-exclusivity works properly.)
        const CONCURRENT_TX_COUNT: usize = 10;
        const MAXIMUM_ACQUIRE_COUNT: usize = 4;

        // Sanity check constants: ensure we acquire jobs across multiple calls to exercise the
        // maximum-jobs-per-call functionality. Make sure we're attempting to acquire enough jobs
        // in total to cover the number of acquirable jobs we created.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(MAXIMUM_ACQUIRE_COUNT < AGGREGATION_JOB_COUNT);
            assert!(
                MAXIMUM_ACQUIRE_COUNT
                    .checked_mul(CONCURRENT_TX_COUNT)
                    .unwrap()
                    >= AGGREGATION_JOB_COUNT
            );
        }

        let want_expiry_time = clock.now().as_naive_date_time().unwrap()
            + chrono::Duration::from_std(LEASE_DURATION).unwrap();
        let want_aggregation_jobs: Vec<_> = aggregation_job_ids
            .iter()
            .map(|&agg_job_id| {
                (
                    AcquiredAggregationJob::new(
                        *task.id(),
                        agg_job_id,
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Prio3Count,
                    ),
                    want_expiry_time,
                )
            })
            .collect();

        let got_leases = timeout(StdDuration::from_secs(10), {
            let ds = Arc::clone(&ds);
            let want_lease_count = want_aggregation_jobs.len();
            async move {
                let mut got_leases = Vec::new();
                loop {
                    // Rarely, due to Postgres locking semantics, an aggregation job that could be
                    // returned will instead be skipped by all concurrent aggregation attempts. Retry
                    // for a little while to keep this from affecting test outcome.
                    let results = try_join_all(
                        iter::repeat_with(|| {
                            ds.run_tx(|tx| {
                                Box::pin(async move {
                                    tx.acquire_incomplete_aggregation_jobs(
                                        &LEASE_DURATION,
                                        MAXIMUM_ACQUIRE_COUNT,
                                    )
                                    .await
                                })
                            })
                        })
                        .take(CONCURRENT_TX_COUNT),
                    )
                    .await
                    .unwrap();

                    for result in results {
                        assert!(result.len() <= MAXIMUM_ACQUIRE_COUNT);
                        got_leases.extend(result.into_iter());
                    }

                    if got_leases.len() >= want_lease_count {
                        break got_leases;
                    }
                }
            }
        })
        .await
        .unwrap();

        // Verify: check that we got all of the desired aggregation jobs, with no duplication, and
        // the expected lease expiry.
        let mut got_aggregation_jobs: Vec<_> = got_leases
            .iter()
            .map(|lease| {
                assert_eq!(lease.lease_attempts(), 1);
                (lease.leased().clone(), *lease.lease_expiry_time())
            })
            .collect();
        got_aggregation_jobs.sort();

        assert_eq!(want_aggregation_jobs, got_aggregation_jobs);

        // Run: release a few jobs, then attempt to acquire jobs again.
        const RELEASE_COUNT: usize = 2;

        // Sanity check constants: ensure we release fewer jobs than we're about to acquire to
        // ensure we can acquire them in all in a single call, while leaving headroom to acquire
        // at least one unwanted job if there is a logic bug.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(RELEASE_COUNT < MAXIMUM_ACQUIRE_COUNT);
        }

        let leases_to_release: Vec<_> = got_leases.into_iter().take(RELEASE_COUNT).collect();
        let mut jobs_to_release: Vec<_> = leases_to_release
            .iter()
            .map(|lease| (lease.leased().clone(), *lease.lease_expiry_time()))
            .collect();
        jobs_to_release.sort();
        ds.run_tx(|tx| {
            let leases_to_release = leases_to_release.clone();
            Box::pin(async move {
                for lease in leases_to_release {
                    tx.release_aggregation_job(&lease).await?;
                }
                Ok(())
            })
        })
        .await
        .unwrap();

        let mut got_aggregation_jobs: Vec<_> = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.acquire_incomplete_aggregation_jobs(&LEASE_DURATION, MAXIMUM_ACQUIRE_COUNT)
                        .await
                })
            })
            .await
            .unwrap()
            .into_iter()
            .map(|lease| {
                assert_eq!(lease.lease_attempts(), 1);
                (lease.leased().clone(), *lease.lease_expiry_time())
            })
            .collect();
        got_aggregation_jobs.sort();

        // Verify: we should have re-acquired the jobs we released.
        assert_eq!(jobs_to_release, got_aggregation_jobs);

        // Run: advance time by the lease duration (which implicitly releases the jobs), and attempt
        // to acquire aggregation jobs again.
        clock.advance(&Duration::from_seconds(LEASE_DURATION.as_secs()));
        let want_expiry_time = clock.now().as_naive_date_time().unwrap()
            + chrono::Duration::from_std(LEASE_DURATION).unwrap();
        let want_aggregation_jobs: Vec<_> = aggregation_job_ids
            .iter()
            .map(|&job_id| {
                (
                    AcquiredAggregationJob::new(
                        *task.id(),
                        job_id,
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Prio3Count,
                    ),
                    want_expiry_time,
                )
            })
            .collect();
        let mut got_aggregation_jobs: Vec<_> = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    // This time, we just acquire all jobs in a single go for simplicity -- we've
                    // already tested the maximum acquire count functionality above.
                    tx.acquire_incomplete_aggregation_jobs(&LEASE_DURATION, AGGREGATION_JOB_COUNT)
                        .await
                })
            })
            .await
            .unwrap()
            .into_iter()
            .map(|lease| {
                let job = (lease.leased().clone(), *lease.lease_expiry_time());
                let expected_attempts = if jobs_to_release.contains(&job) { 1 } else { 2 };
                assert_eq!(lease.lease_attempts(), expected_attempts);
                job
            })
            .collect();
        got_aggregation_jobs.sort();

        // Verify: we got all the jobs.
        assert_eq!(want_aggregation_jobs, got_aggregation_jobs);

        // Run: advance time again to release jobs, acquire a single job, modify its lease token
        // to simulate a previously-held lease, and attempt to release it. Verify that releasing
        // fails.
        clock.advance(&Duration::from_seconds(LEASE_DURATION.as_secs()));
        let lease = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    Ok(tx
                        .acquire_incomplete_aggregation_jobs(&LEASE_DURATION, 1)
                        .await?
                        .remove(0))
                })
            })
            .await
            .unwrap();
        let lease_with_random_token = Lease::new(
            lease.leased().clone(),
            *lease.lease_expiry_time(),
            random(),
            lease.lease_attempts(),
        );
        ds.run_tx(|tx| {
            let lease_with_random_token = lease_with_random_token.clone();
            Box::pin(async move { tx.release_aggregation_job(&lease_with_random_token).await })
        })
        .await
        .unwrap_err();

        // Replace the original lease token and verify that we can release successfully with it in
        // place.
        ds.run_tx(|tx| {
            let lease = lease.clone();
            Box::pin(async move { tx.release_aggregation_job(&lease).await })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn aggregation_job_not_found(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        let rslt = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.get_aggregation_job::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>(
                        &random(),
                        &random(),
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(rslt, None);

        let rslt = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.update_aggregation_job::<PRIO3_VERIFY_KEY_LENGTH, FixedSize, Prio3Count>(
                        &AggregationJob::new(
                            random(),
                            random(),
                            (),
                            random(),
                            Interval::new(
                                Time::from_seconds_since_epoch(0),
                                Duration::from_seconds(1),
                            )
                            .unwrap(),
                            AggregationJobState::InProgress,
                            AggregationJobRound::from(0),
                        ),
                    )
                    .await
                })
            })
            .await;
        assert_matches!(rslt, Err(Error::MutationTargetNotFound));
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn get_aggregation_jobs_for_task(ephemeral_datastore: EphemeralDatastore) {
        // Setup.
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        // We use a dummy VDAF & fixed-size task for this test, to better exercise the
        // serialization/deserialization roundtrip of the batch_identifier & aggregation_param.
        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();
        let first_aggregation_job = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *task.id(),
            random(),
            AggregationParam(23),
            random(),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::InProgress,
            AggregationJobRound::from(0),
        );
        let second_aggregation_job = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            *task.id(),
            random(),
            AggregationParam(42),
            random(),
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::InProgress,
            AggregationJobRound::from(0),
        );
        let aggregation_job_with_request_hash =
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                *task.id(),
                random(),
                AggregationParam(42),
                random(),
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::InProgress,
                AggregationJobRound::from(0),
            )
            .with_last_continue_request_hash([3; 32]);

        let mut want_agg_jobs = Vec::from([
            first_aggregation_job,
            second_aggregation_job,
            aggregation_job_with_request_hash,
        ]);

        ds.run_tx(|tx| {
            let (task, want_agg_jobs) = (task.clone(), want_agg_jobs.clone());
            Box::pin(async move {
                tx.put_task(&task).await?;

                for agg_job in want_agg_jobs {
                    tx.put_aggregation_job(&agg_job).await.unwrap();
                }

                // Also write an unrelated aggregation job with a different task ID to check that it
                // is not returned.
                let unrelated_task = TaskBuilder::new(
                    task::QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Fake,
                    Role::Leader,
                )
                .build();
                tx.put_task(&unrelated_task).await?;
                tx.put_aggregation_job(&AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *unrelated_task.id(),
                    random(),
                    AggregationParam(82),
                    random(),
                    Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobRound::from(0),
                ))
                .await
            })
        })
        .await
        .unwrap();

        // Run.
        want_agg_jobs.sort_by_key(|agg_job| *agg_job.id());
        let mut got_agg_jobs = ds
            .run_tx(|tx| {
                let task = task.clone();
                Box::pin(async move { tx.get_aggregation_jobs_for_task(task.id()).await })
            })
            .await
            .unwrap();
        got_agg_jobs.sort_by_key(|agg_job| *agg_job.id());

        // Verify.
        assert_eq!(want_agg_jobs, got_agg_jobs);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_report_aggregation(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);

        let report_id = random();
        let vdaf = Arc::new(Prio3::new_count(2).unwrap());
        let verify_key: [u8; PRIO3_VERIFY_KEY_LENGTH] = random();
        let vdaf_transcript = run_vdaf(vdaf.as_ref(), &verify_key, &(), &report_id, &0);
        let leader_prep_state = vdaf_transcript.leader_prep_state(0);

        for (ord, state) in [
            ReportAggregationState::<PRIO3_VERIFY_KEY_LENGTH, Prio3Count>::Start,
            ReportAggregationState::Waiting(
                leader_prep_state.clone(),
                Some(vdaf_transcript.prepare_messages[0].clone()),
            ),
            ReportAggregationState::Waiting(leader_prep_state.clone(), None),
            ReportAggregationState::Finished,
            ReportAggregationState::Failed(ReportShareError::VdafPrepError),
        ]
        .into_iter()
        .enumerate()
        {
            let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
            let ds = ephemeral_datastore.datastore(clock.clone()).await;

            let task = TaskBuilder::new(
                task::QueryType::FixedSize { max_batch_size: 10 },
                VdafInstance::Prio3Count,
                Role::Leader,
            )
            .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
            .build();
            let aggregation_job_id = random();
            let report_id = random();

            let want_report_aggregation = ds
                .run_tx(|tx| {
                    let (task, state) = (task.clone(), state.clone());
                    Box::pin(async move {
                        tx.put_task(&task).await?;
                        tx.put_aggregation_job(&AggregationJob::<
                            PRIO3_VERIFY_KEY_LENGTH,
                            FixedSize,
                            Prio3Count,
                        >::new(
                            *task.id(),
                            aggregation_job_id,
                            (),
                            random(),
                            Interval::new(
                                OLDEST_ALLOWED_REPORT_TIMESTAMP,
                                Duration::from_seconds(1),
                            )
                            .unwrap(),
                            AggregationJobState::InProgress,
                            AggregationJobRound::from(0),
                        ))
                        .await?;
                        tx.put_report_share(
                            task.id(),
                            &ReportShare::new(
                                ReportMetadata::new(report_id, OLDEST_ALLOWED_REPORT_TIMESTAMP),
                                Vec::from("public_share"),
                                HpkeCiphertext::new(
                                    HpkeConfigId::from(12),
                                    Vec::from("encapsulated_context_0"),
                                    Vec::from("payload_0"),
                                ),
                            ),
                        )
                        .await?;

                        let report_aggregation = ReportAggregation::new(
                            *task.id(),
                            aggregation_job_id,
                            report_id,
                            OLDEST_ALLOWED_REPORT_TIMESTAMP,
                            ord.try_into().unwrap(),
                            Some(PrepareStep::new(
                                report_id,
                                PrepareStepResult::Continued(format!("prep_msg_{ord}").into()),
                            )),
                            state,
                        );
                        tx.put_report_aggregation(&report_aggregation).await?;
                        Ok(report_aggregation)
                    })
                })
                .await
                .unwrap();

            // Advance the clock to "enable" report expiry.
            clock.advance(&REPORT_EXPIRY_AGE);

            let got_report_aggregation = ds
                .run_tx(|tx| {
                    let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
                    Box::pin(async move {
                        tx.get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            &report_id,
                        )
                        .await
                    })
                })
                .await
                .unwrap()
                .unwrap();

            assert_eq!(want_report_aggregation, got_report_aggregation);

            let want_report_aggregation = ReportAggregation::new(
                *want_report_aggregation.task_id(),
                *want_report_aggregation.aggregation_job_id(),
                *want_report_aggregation.report_id(),
                *want_report_aggregation.time(),
                want_report_aggregation.ord(),
                Some(PrepareStep::new(
                    report_id,
                    PrepareStepResult::Continued(format!("updated_prep_msg_{ord}").into()),
                )),
                want_report_aggregation.state().clone(),
            );

            ds.run_tx(|tx| {
                let want_report_aggregation = want_report_aggregation.clone();
                Box::pin(
                    async move { tx.update_report_aggregation(&want_report_aggregation).await },
                )
            })
            .await
            .unwrap();

            let got_report_aggregation = ds
                .run_tx(|tx| {
                    let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
                    Box::pin(async move {
                        tx.get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            &report_id,
                        )
                        .await
                    })
                })
                .await
                .unwrap();
            assert_eq!(Some(want_report_aggregation), got_report_aggregation);

            // Advance the clock again to expire relevant datastore items.
            clock.advance(&REPORT_EXPIRY_AGE);

            let got_report_aggregation = ds
                .run_tx(|tx| {
                    let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
                    Box::pin(async move {
                        tx.get_report_aggregation(
                            vdaf.as_ref(),
                            &Role::Leader,
                            task.id(),
                            &aggregation_job_id,
                            &report_id,
                        )
                        .await
                    })
                })
                .await
                .unwrap();
            assert_eq!(None, got_report_aggregation);
        }
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn check_other_report_aggregation_exists(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Helper,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();

        ds.put_task(&task).await.unwrap();

        let aggregation_job_id = random();
        let report_id = random();

        ds.run_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move {
                tx.put_aggregation_job(&AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    task_id,
                    aggregation_job_id,
                    dummy_vdaf::AggregationParam(0),
                    random(),
                    Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                        .unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobRound::from(0),
                ))
                .await?;
                tx.put_report_share(
                    &task_id,
                    &ReportShare::new(
                        ReportMetadata::new(report_id, OLDEST_ALLOWED_REPORT_TIMESTAMP),
                        Vec::from("public_share"),
                        HpkeCiphertext::new(
                            HpkeConfigId::from(12),
                            Vec::from("encapsulated_context_0"),
                            Vec::from("payload_0"),
                        ),
                    ),
                )
                .await?;

                let report_aggregation = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                    task_id,
                    aggregation_job_id,
                    report_id,
                    OLDEST_ALLOWED_REPORT_TIMESTAMP,
                    0,
                    None,
                    ReportAggregationState::Start,
                );
                tx.put_report_aggregation(&report_aggregation).await?;
                Ok(())
            })
        })
        .await
        .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        ds.run_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move {
                assert!(tx
                    .check_other_report_aggregation_exists::<0, dummy_vdaf::Vdaf>(
                        &task_id,
                        &report_id,
                        &dummy_vdaf::AggregationParam(0),
                        &random(),
                    )
                    .await
                    .unwrap());

                // Aggregation job ID matches
                assert!(!tx
                    .check_other_report_aggregation_exists::<0, dummy_vdaf::Vdaf>(
                        &task_id,
                        &report_id,
                        &dummy_vdaf::AggregationParam(0),
                        &aggregation_job_id,
                    )
                    .await
                    .unwrap());

                // Wrong task ID
                assert!(!tx
                    .check_other_report_aggregation_exists::<0, dummy_vdaf::Vdaf>(
                        &random(),
                        &report_id,
                        &dummy_vdaf::AggregationParam(0),
                        &random(),
                    )
                    .await
                    .unwrap());

                // Wrong report ID
                assert!(!tx
                    .check_other_report_aggregation_exists::<0, dummy_vdaf::Vdaf>(
                        &task_id,
                        &random(),
                        &dummy_vdaf::AggregationParam(0),
                        &random(),
                    )
                    .await
                    .unwrap());

                // Wrong aggregation param
                assert!(!tx
                    .check_other_report_aggregation_exists::<0, dummy_vdaf::Vdaf>(
                        &task_id,
                        &report_id,
                        &dummy_vdaf::AggregationParam(1),
                        &random(),
                    )
                    .await
                    .unwrap());

                Ok(())
            })
        })
        .await
        .unwrap();

        // Advance the clock again to expire all relevant datastore items.
        clock.advance(&REPORT_EXPIRY_AGE);

        ds.run_tx(|tx| {
            let task_id = *task.id();
            Box::pin(async move {
                assert!(!tx
                    .check_other_report_aggregation_exists::<0, dummy_vdaf::Vdaf>(
                        &task_id,
                        &report_id,
                        &dummy_vdaf::AggregationParam(0),
                        &random(),
                    )
                    .await
                    .unwrap());

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn report_aggregation_not_found(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        let vdaf = Arc::new(dummy_vdaf::Vdaf::default());

        let rslt = ds
            .run_tx(|tx| {
                let vdaf = Arc::clone(&vdaf);
                Box::pin(async move {
                    tx.get_report_aggregation(
                        vdaf.as_ref(),
                        &Role::Leader,
                        &random(),
                        &random(),
                        &ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(rslt, None);

        let rslt = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    tx.update_report_aggregation::<0, dummy_vdaf::Vdaf>(&ReportAggregation::new(
                        random(),
                        random(),
                        ReportId::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                        Time::from_seconds_since_epoch(12345),
                        0,
                        None,
                        ReportAggregationState::Failed(ReportShareError::VdafPrepError),
                    ))
                    .await
                })
            })
            .await;
        assert_matches!(rslt, Err(Error::MutationTargetNotFound));
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn get_report_aggregations_for_aggregation_job(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let report_id = random();
        let vdaf = Arc::new(Prio3::new_count(2).unwrap());
        let verify_key: [u8; PRIO3_VERIFY_KEY_LENGTH] = random();
        let vdaf_transcript = run_vdaf(vdaf.as_ref(), &verify_key, &(), &report_id, &0);

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Prio3Count,
            Role::Leader,
        )
        .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
        .build();
        let aggregation_job_id = random();

        let want_report_aggregations = ds
            .run_tx(|tx| {
                let (task, prep_msg, prep_state) = (
                    task.clone(),
                    vdaf_transcript.prepare_messages[0].clone(),
                    vdaf_transcript.leader_prep_state(0).clone(),
                );
                Box::pin(async move {
                    tx.put_task(&task).await?;
                    tx.put_aggregation_job(&AggregationJob::<
                        PRIO3_VERIFY_KEY_LENGTH,
                        FixedSize,
                        Prio3Count,
                    >::new(
                        *task.id(),
                        aggregation_job_id,
                        (),
                        random(),
                        Interval::new(OLDEST_ALLOWED_REPORT_TIMESTAMP, Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::InProgress,
                        AggregationJobRound::from(0),
                    ))
                    .await?;

                    let mut want_report_aggregations = Vec::new();
                    for (ord, state) in [
                        ReportAggregationState::<PRIO3_VERIFY_KEY_LENGTH, Prio3Count>::Start,
                        ReportAggregationState::Waiting(prep_state.clone(), Some(prep_msg)),
                        ReportAggregationState::Finished,
                        ReportAggregationState::Failed(ReportShareError::VdafPrepError),
                    ]
                    .iter()
                    .enumerate()
                    {
                        let report_id = ReportId::from((ord as u128).to_be_bytes());
                        tx.put_report_share(
                            task.id(),
                            &ReportShare::new(
                                ReportMetadata::new(report_id, OLDEST_ALLOWED_REPORT_TIMESTAMP),
                                Vec::from("public_share"),
                                HpkeCiphertext::new(
                                    HpkeConfigId::from(12),
                                    Vec::from("encapsulated_context_0"),
                                    Vec::from("payload_0"),
                                ),
                            ),
                        )
                        .await?;

                        let report_aggregation = ReportAggregation::new(
                            *task.id(),
                            aggregation_job_id,
                            report_id,
                            OLDEST_ALLOWED_REPORT_TIMESTAMP,
                            ord.try_into().unwrap(),
                            Some(PrepareStep::new(report_id, PrepareStepResult::Finished)),
                            state.clone(),
                        );
                        tx.put_report_aggregation(&report_aggregation).await?;
                        want_report_aggregations.push(report_aggregation);
                    }
                    Ok(want_report_aggregations)
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        let got_report_aggregations = ds
            .run_tx(|tx| {
                let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
                Box::pin(async move {
                    tx.get_report_aggregations_for_aggregation_job(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert_eq!(want_report_aggregations, got_report_aggregations);

        // Advance the clock again to expire relevant datastore entities.
        clock.advance(&REPORT_EXPIRY_AGE);

        let got_report_aggregations = ds
            .run_tx(|tx| {
                let (vdaf, task) = (Arc::clone(&vdaf), task.clone());
                Box::pin(async move {
                    tx.get_report_aggregations_for_aggregation_job(
                        vdaf.as_ref(),
                        &Role::Leader,
                        task.id(),
                        &aggregation_job_id,
                    )
                    .await
                })
            })
            .await
            .unwrap();
        assert!(got_report_aggregations.is_empty());
    }

    #[tokio::test]
    async fn crypter() {
        let crypter = Crypter::new(Vec::from([generate_aead_key(), generate_aead_key()]));
        let bad_key = generate_aead_key();

        const TABLE: &str = "some_table";
        const ROW: &[u8] = b"12345";
        const COLUMN: &str = "some_column";
        const PLAINTEXT: &[u8] = b"This is my plaintext value.";

        // Test that roundtripping encryption works.
        let ciphertext = crypter.encrypt(TABLE, ROW, COLUMN, PLAINTEXT).unwrap();
        let plaintext = crypter.decrypt(TABLE, ROW, COLUMN, &ciphertext).unwrap();
        assert_eq!(PLAINTEXT, &plaintext);

        // Roundtripping encryption works even if a non-primary key was used for encryption.
        let ciphertext =
            Crypter::encrypt_with_key(crypter.keys.last().unwrap(), TABLE, ROW, COLUMN, PLAINTEXT)
                .unwrap();
        let plaintext = crypter.decrypt(TABLE, ROW, COLUMN, &ciphertext).unwrap();
        assert_eq!(PLAINTEXT, &plaintext);

        // Roundtripping encryption with an unknown key fails.
        let ciphertext =
            Crypter::encrypt_with_key(&bad_key, TABLE, ROW, COLUMN, PLAINTEXT).unwrap();
        assert!(crypter.decrypt(TABLE, ROW, COLUMN, &ciphertext).is_err());

        // Roundtripping encryption with a mismatched table, row, or column fails.
        let ciphertext = crypter.encrypt(TABLE, ROW, COLUMN, PLAINTEXT).unwrap();
        assert!(crypter
            .decrypt("wrong_table", ROW, COLUMN, &ciphertext)
            .is_err());
        assert!(crypter
            .decrypt(TABLE, b"wrong_row", COLUMN, &ciphertext)
            .is_err());
        assert!(crypter
            .decrypt(TABLE, ROW, "wrong_column", &ciphertext)
            .is_err());
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn get_collection_job(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();
        let first_batch_id = random();
        let second_batch_id = random();
        let aggregation_param = AggregationParam(13);

        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        ds.run_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.put_task(&task).await.unwrap();

                let vdaf = dummy_vdaf::Vdaf::new();
                let first_collection_job = CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    random(),
                    first_batch_id,
                    aggregation_param,
                    CollectionJobState::Start,
                );
                tx.put_collection_job(&first_collection_job).await.unwrap();

                let second_collection_job = CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    random(),
                    second_batch_id,
                    aggregation_param,
                    CollectionJobState::Start,
                );
                tx.put_collection_job(&second_collection_job).await.unwrap();

                let first_collection_job_again = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        first_collection_job.id(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(first_collection_job, first_collection_job_again);

                let second_collection_job_again = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        second_collection_job.id(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(second_collection_job, second_collection_job_again);

                let encrypted_helper_aggregate_share = hpke::seal(
                    task.collector_hpke_config(),
                    &HpkeApplicationInfo::new(
                        &Label::AggregateShare,
                        &Role::Helper,
                        &Role::Collector,
                    ),
                    &[0, 1, 2, 3, 4, 5],
                    &AggregateShareAad::new(
                        *task.id(),
                        BatchSelector::new_fixed_size(first_batch_id),
                    )
                    .get_encoded(),
                )
                .unwrap();

                let first_collection_job =
                    first_collection_job.with_state(CollectionJobState::Finished {
                        report_count: 12,
                        encrypted_helper_aggregate_share,
                        leader_aggregate_share: AggregateShare(41),
                    });

                tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(&first_collection_job)
                    .await
                    .unwrap();

                let updated_first_collection_job = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        first_collection_job.id(),
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(first_collection_job, updated_first_collection_job);

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn update_collection_jobs(ephemeral_datastore: EphemeralDatastore) {
        // Setup: write collection jobs to the datastore.
        install_test_trace_subscriber();

        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        let task = TaskBuilder::new(
            task::QueryType::FixedSize { max_batch_size: 10 },
            VdafInstance::Fake,
            Role::Leader,
        )
        .build();
        let abandoned_batch_id = random();
        let deleted_batch_id = random();

        ds.run_tx(|tx| {
            let task = task.clone();
            Box::pin(async move {
                tx.put_task(&task).await?;

                let vdaf = dummy_vdaf::Vdaf::new();
                let aggregation_param = AggregationParam(10);
                let abandoned_collection_job: CollectionJob<0, _, dummy_vdaf::Vdaf> =
                    CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        random(),
                        abandoned_batch_id,
                        aggregation_param,
                        CollectionJobState::Start,
                    );
                tx.put_collection_job(&abandoned_collection_job).await?;

                let deleted_collection_job = CollectionJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    random(),
                    deleted_batch_id,
                    aggregation_param,
                    CollectionJobState::Start,
                );
                tx.put_collection_job(&deleted_collection_job).await?;

                let abandoned_collection_job_again = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        abandoned_collection_job.id(),
                    )
                    .await?
                    .unwrap();

                // Verify: initial state.
                assert_eq!(abandoned_collection_job, abandoned_collection_job_again);

                // Setup: update the collection jobs.
                let abandoned_collection_job =
                    abandoned_collection_job.with_state(CollectionJobState::Abandoned);
                let deleted_collection_job =
                    deleted_collection_job.with_state(CollectionJobState::Deleted);

                tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                    &abandoned_collection_job,
                )
                .await?;
                tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(&deleted_collection_job)
                    .await?;

                let abandoned_collection_job_again = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        abandoned_collection_job.id(),
                    )
                    .await?
                    .unwrap();

                let deleted_collection_job_again = tx
                    .get_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        deleted_collection_job.id(),
                    )
                    .await?
                    .unwrap();

                // Verify: collection jobs were updated.
                assert_eq!(abandoned_collection_job, abandoned_collection_job_again);
                assert_eq!(deleted_collection_job, deleted_collection_job_again);

                // Setup: try to update a job into state `Start`
                let abandoned_collection_job =
                    abandoned_collection_job.with_state(CollectionJobState::Start);

                // Verify: Update should fail
                tx.update_collection_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                    &abandoned_collection_job,
                )
                .await
                .unwrap_err();
                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[derive(Copy, Clone)]
    enum CollectionJobTestCaseState {
        Start,
        Collectable,
        Finished,
        Deleted,
        Abandoned,
    }

    #[derive(Clone)]
    struct CollectionJobTestCase<Q: QueryType> {
        should_be_acquired: bool,
        task_id: TaskId,
        batch_identifier: Q::BatchIdentifier,
        agg_param: AggregationParam,
        collection_job_id: Option<CollectionJobId>,
        state: CollectionJobTestCaseState,
    }

    #[derive(Clone)]
    struct CollectionJobAcquireTestCase<Q: CollectableQueryType> {
        task_ids: Vec<TaskId>,
        query_type: task::QueryType,
        reports: Vec<LeaderStoredReport<0, dummy_vdaf::Vdaf>>,
        aggregation_jobs: Vec<AggregationJob<0, Q, dummy_vdaf::Vdaf>>,
        report_aggregations: Vec<ReportAggregation<0, dummy_vdaf::Vdaf>>,
        collection_job_test_cases: Vec<CollectionJobTestCase<Q>>,
    }

    async fn setup_collection_job_acquire_test_case<Q: CollectableQueryType>(
        ds: &Datastore<MockClock>,
        test_case: CollectionJobAcquireTestCase<Q>,
    ) -> CollectionJobAcquireTestCase<Q> {
        ds.run_tx(|tx| {
            let mut test_case = test_case.clone();
            Box::pin(async move {
                for task_id in &test_case.task_ids {
                    tx.put_task(
                        &TaskBuilder::new(test_case.query_type, VdafInstance::Fake, Role::Leader)
                            .with_id(*task_id)
                            .build(),
                    )
                    .await?;
                }

                for report in &test_case.reports {
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), report)
                        .await?;
                }

                for aggregation_job in &test_case.aggregation_jobs {
                    tx.put_aggregation_job(aggregation_job).await?;
                }

                for report_aggregation in &test_case.report_aggregations {
                    tx.put_report_aggregation(report_aggregation).await?;
                }

                for test_case in test_case.collection_job_test_cases.iter_mut() {
                    let collection_job = CollectionJob::<0, Q, dummy_vdaf::Vdaf>::new(
                        test_case.task_id,
                        random(),
                        test_case.batch_identifier.clone(),
                        test_case.agg_param,
                        match test_case.state {
                            CollectionJobTestCaseState::Start => CollectionJobState::Start,
                            CollectionJobTestCaseState::Collectable => {
                                CollectionJobState::Collectable
                            }
                            CollectionJobTestCaseState::Finished => CollectionJobState::Finished {
                                report_count: 1,
                                encrypted_helper_aggregate_share: HpkeCiphertext::new(
                                    HpkeConfigId::from(0),
                                    Vec::new(),
                                    Vec::new(),
                                ),
                                leader_aggregate_share: AggregateShare(0),
                            },
                            CollectionJobTestCaseState::Abandoned => CollectionJobState::Abandoned,
                            CollectionJobTestCaseState::Deleted => CollectionJobState::Deleted,
                        },
                    );
                    tx.put_collection_job(&collection_job).await?;
                    test_case.collection_job_id = Some(*collection_job.id());
                }

                Ok(test_case)
            })
        })
        .await
        .unwrap()
    }

    async fn run_collection_job_acquire_test_case<Q: CollectableQueryType>(
        ds: &Datastore<MockClock>,
        test_case: CollectionJobAcquireTestCase<Q>,
    ) -> Vec<Lease<AcquiredCollectionJob>> {
        let test_case = setup_collection_job_acquire_test_case(ds, test_case).await;

        let clock = &ds.clock;
        ds.run_tx(|tx| {
            let test_case = test_case.clone();
            let clock = clock.clone();
            Box::pin(async move {
                let leases = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await?;

                let mut leased_collection_jobs: Vec<_> = leases
                    .iter()
                    .map(|lease| (lease.leased().clone(), *lease.lease_expiry_time()))
                    .collect();
                leased_collection_jobs.sort();

                let mut expected_collection_jobs: Vec<_> = test_case
                    .collection_job_test_cases
                    .iter()
                    .filter(|c| c.should_be_acquired)
                    .map(|c| {
                        (
                            AcquiredCollectionJob::new(
                                c.task_id,
                                c.collection_job_id.unwrap(),
                                test_case.query_type,
                                VdafInstance::Fake,
                            ),
                            clock.now().as_naive_date_time().unwrap()
                                + chrono::Duration::seconds(100),
                        )
                    })
                    .collect();
                expected_collection_jobs.sort();

                assert_eq!(leased_collection_jobs, expected_collection_jobs);

                Ok(leases)
            })
        })
        .await
        .unwrap()
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn time_interval_collection_job_acquire_release_happy_path(
        ephemeral_datastore: EphemeralDatastore,
    ) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            Time::from_seconds_since_epoch(0),
        )]);
        let batch_id = random();
        let aggregation_job_id = random();
        let aggregation_jobs = Vec::from([AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            AggregationParam(0),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobRound::from(1),
        )]);
        let report_aggregations = Vec::from([ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Start, // Doesn't matter what state the report aggregation is in
        )]);

        let collection_job_test_cases = Vec::from([CollectionJobTestCase::<FixedSize> {
            should_be_acquired: true,
            task_id,
            batch_identifier: batch_id,
            agg_param: AggregationParam(0),
            collection_job_id: None,
            state: CollectionJobTestCaseState::Collectable,
        }]);

        let collection_job_leases = run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases,
            },
        )
        .await;

        let reacquired_jobs = ds
            .run_tx(|tx| {
                let collection_job_leases = collection_job_leases.clone();
                Box::pin(async move {
                    // Try to re-acquire collection jobs. Nothing should happen because the lease is still
                    // valid.
                    assert!(tx
                        .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                        .await
                        .unwrap()
                        .is_empty());

                    // Release the lease, then re-acquire it.
                    tx.release_collection_job(&collection_job_leases[0])
                        .await
                        .unwrap();

                    let reacquired_leases = tx
                        .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                        .await
                        .unwrap();
                    let reacquired_jobs: Vec<_> = reacquired_leases
                        .iter()
                        .map(|lease| (lease.leased().clone(), lease.lease_expiry_time()))
                        .collect();

                    let collection_jobs: Vec<_> = collection_job_leases
                        .iter()
                        .map(|lease| (lease.leased().clone(), lease.lease_expiry_time()))
                        .collect();

                    assert_eq!(reacquired_jobs.len(), 1);
                    assert_eq!(reacquired_jobs, collection_jobs);

                    Ok(reacquired_leases)
                })
            })
            .await
            .unwrap();

        // Advance time by the lease duration
        clock.advance(&Duration::from_seconds(100));

        ds.run_tx(|tx| {
            let reacquired_jobs = reacquired_jobs.clone();
            Box::pin(async move {
                // Re-acquire the jobs whose lease should have lapsed.
                let acquired_jobs = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await
                    .unwrap();

                for (acquired_job, reacquired_job) in acquired_jobs.iter().zip(reacquired_jobs) {
                    assert_eq!(acquired_job.leased(), reacquired_job.leased());
                    assert_eq!(
                        *acquired_job.lease_expiry_time(),
                        *reacquired_job.lease_expiry_time() + chrono::Duration::seconds(100),
                    );
                }

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn fixed_size_collection_job_acquire_release_happy_path(
        ephemeral_datastore: EphemeralDatastore,
    ) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            Time::from_seconds_since_epoch(0),
        )]);
        let batch_id = random();
        let aggregation_job_id = random();
        let aggregation_jobs = Vec::from([AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            AggregationParam(0),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobRound::from(1),
        )]);
        let report_aggregations = Vec::from([ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Start, // Doesn't matter what state the report aggregation is in
        )]);

        let collection_job_leases = run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases: Vec::from([CollectionJobTestCase::<FixedSize> {
                    should_be_acquired: true,
                    task_id,
                    batch_identifier: batch_id,
                    agg_param: AggregationParam(0),
                    collection_job_id: None,
                    state: CollectionJobTestCaseState::Collectable,
                }]),
            },
        )
        .await;

        let reacquired_jobs = ds
            .run_tx(|tx| {
                let collection_job_leases = collection_job_leases.clone();
                Box::pin(async move {
                    // Try to re-acquire collection jobs. Nothing should happen because the lease is still
                    // valid.
                    assert!(tx
                        .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10,)
                        .await
                        .unwrap()
                        .is_empty());

                    // Release the lease, then re-acquire it.
                    tx.release_collection_job(&collection_job_leases[0])
                        .await
                        .unwrap();

                    let reacquired_leases = tx
                        .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                        .await
                        .unwrap();
                    let reacquired_jobs: Vec<_> = reacquired_leases
                        .iter()
                        .map(|lease| (lease.leased().clone(), lease.lease_expiry_time()))
                        .collect();

                    let collection_jobs: Vec<_> = collection_job_leases
                        .iter()
                        .map(|lease| (lease.leased().clone(), lease.lease_expiry_time()))
                        .collect();

                    assert_eq!(reacquired_jobs.len(), 1);
                    assert_eq!(reacquired_jobs, collection_jobs);

                    Ok(reacquired_leases)
                })
            })
            .await
            .unwrap();

        // Advance time by the lease duration
        clock.advance(&Duration::from_seconds(100));

        ds.run_tx(|tx| {
            let reacquired_jobs = reacquired_jobs.clone();
            Box::pin(async move {
                // Re-acquire the jobs whose lease should have lapsed.
                let acquired_jobs = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await
                    .unwrap();

                for (acquired_job, reacquired_job) in acquired_jobs.iter().zip(reacquired_jobs) {
                    assert_eq!(acquired_job.leased(), reacquired_job.leased());
                    assert_eq!(
                        *acquired_job.lease_expiry_time(),
                        *reacquired_job.lease_expiry_time() + chrono::Duration::seconds(100),
                    );
                }

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_no_aggregation_job_with_task_id(
        ephemeral_datastore: EphemeralDatastore,
    ) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let other_task_id = random();

        let batch_id = random();
        let aggregation_jobs = Vec::from([AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            // Aggregation job task ID does not match collection job task ID
            other_task_id,
            random(),
            AggregationParam(0),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobRound::from(1),
        )]);

        let collection_job_test_cases = Vec::from([CollectionJobTestCase::<FixedSize> {
            should_be_acquired: false,
            task_id,
            batch_identifier: batch_id,
            agg_param: AggregationParam(0),
            collection_job_id: None,
            state: CollectionJobTestCaseState::Start,
        }]);

        run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id, other_task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports: Vec::new(),
                aggregation_jobs,
                report_aggregations: Vec::new(),
                collection_job_test_cases,
            },
        )
        .await;
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_no_aggregation_job_with_agg_param(
        ephemeral_datastore: EphemeralDatastore,
    ) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            Time::from_seconds_since_epoch(0),
        )]);

        let batch_id = random();
        let aggregation_jobs = Vec::from([AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            task_id,
            random(),
            // Aggregation job agg param does not match collection job agg param
            AggregationParam(1),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobRound::from(1),
        )]);

        let collection_job_test_cases = Vec::from([CollectionJobTestCase::<FixedSize> {
            should_be_acquired: false,
            task_id,
            batch_identifier: batch_id,
            agg_param: AggregationParam(0),
            collection_job_id: None,
            state: CollectionJobTestCaseState::Start,
        }]);

        run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations: Vec::new(),
                collection_job_test_cases,
            },
        )
        .await;
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_report_shares_outside_interval(
        ephemeral_datastore: EphemeralDatastore,
    ) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let batch_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            // Report associated with the aggregation job is outside the collection job's batch
            // interval
            Time::from_seconds_since_epoch(200),
        )]);
        let aggregation_job_id = random();
        let aggregation_jobs = Vec::from([AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            AggregationParam(0),
            batch_id,
            Interval::new(
                Time::from_seconds_since_epoch(200),
                Duration::from_seconds(1),
            )
            .unwrap(),
            AggregationJobState::Finished,
            AggregationJobRound::from(1),
        )]);
        let report_aggregations = Vec::from([ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Start, // Shouldn't matter what state the report aggregation is in
        )]);

        run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase::<FixedSize> {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases: Vec::from([CollectionJobTestCase::<FixedSize> {
                    should_be_acquired: false,
                    task_id,
                    batch_identifier: batch_id,
                    agg_param: AggregationParam(0),
                    collection_job_id: None,
                    state: CollectionJobTestCaseState::Start,
                }]),
            },
        )
        .await;
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_release_job_finished(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            Time::from_seconds_since_epoch(0),
        )]);
        let aggregation_job_id = random();
        let batch_id = random();
        let aggregation_jobs = Vec::from([AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            AggregationParam(0),
            batch_id,
            Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1)).unwrap(),
            AggregationJobState::Finished,
            AggregationJobRound::from(1),
        )]);

        let report_aggregations = Vec::from([ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
            task_id,
            aggregation_job_id,
            *reports[0].metadata().id(),
            *reports[0].metadata().time(),
            0,
            None,
            ReportAggregationState::Start,
        )]);

        let collection_job_test_cases = Vec::from([CollectionJobTestCase::<FixedSize> {
            should_be_acquired: false,
            task_id,
            batch_identifier: batch_id,
            agg_param: AggregationParam(0),
            collection_job_id: None,
            // collection job has already run to completion
            state: CollectionJobTestCaseState::Finished,
        }]);

        run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases,
            },
        )
        .await;
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_release_aggregation_job_in_progress(
        ephemeral_datastore: EphemeralDatastore,
    ) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([
            LeaderStoredReport::new_dummy(task_id, Time::from_seconds_since_epoch(0)),
            LeaderStoredReport::new_dummy(task_id, Time::from_seconds_since_epoch(50)),
        ]);

        let aggregation_job_ids: [_; 2] = random();
        let batch_id = random();
        let aggregation_jobs = Vec::from([
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[0],
                AggregationParam(0),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            ),
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[1],
                AggregationParam(0),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                // Aggregation job included in collect request is in progress
                AggregationJobState::InProgress,
                AggregationJobRound::from(0),
            ),
        ]);

        let report_aggregations = Vec::from([
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[0],
                *reports[0].metadata().id(),
                *reports[0].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[1],
                *reports[1].metadata().id(),
                *reports[1].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
        ]);

        let collection_job_test_cases = Vec::from([CollectionJobTestCase::<FixedSize> {
            should_be_acquired: false,
            task_id,
            batch_identifier: batch_id,
            agg_param: AggregationParam(0),
            collection_job_id: None,
            state: CollectionJobTestCaseState::Start,
        }]);

        run_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases,
            },
        )
        .await;
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_job_max(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            Time::from_seconds_since_epoch(0),
        )]);
        let aggregation_job_ids: [_; 2] = random();
        let batch_id = random();
        let aggregation_jobs = Vec::from([
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[0],
                AggregationParam(0),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            ),
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[1],
                AggregationParam(1),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            ),
        ]);
        let report_aggregations = Vec::from([
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[0],
                *reports[0].metadata().id(),
                *reports[0].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[1],
                *reports[0].metadata().id(),
                *reports[0].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
        ]);

        let collection_job_test_cases = Vec::from([
            CollectionJobTestCase::<FixedSize> {
                should_be_acquired: true,
                task_id,
                batch_identifier: batch_id,
                agg_param: AggregationParam(0),
                collection_job_id: None,
                state: CollectionJobTestCaseState::Collectable,
            },
            CollectionJobTestCase::<FixedSize> {
                should_be_acquired: true,
                task_id,
                batch_identifier: batch_id,
                agg_param: AggregationParam(1),
                collection_job_id: None,
                state: CollectionJobTestCaseState::Collectable,
            },
        ]);

        let test_case = setup_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase::<FixedSize> {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases,
            },
        )
        .await;

        ds.run_tx(|tx| {
            let test_case = test_case.clone();
            let clock = clock.clone();
            Box::pin(async move {
                // Acquire a single collection job, twice. Each call should yield one job. We don't
                // care what order they are acquired in.
                let mut acquired_collection_jobs = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                    .await?;
                assert_eq!(acquired_collection_jobs.len(), 1);

                acquired_collection_jobs.extend(
                    tx.acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 1)
                        .await?,
                );

                assert_eq!(acquired_collection_jobs.len(), 2);

                let mut acquired_collection_jobs: Vec<_> = acquired_collection_jobs
                    .iter()
                    .map(|lease| (lease.leased().clone(), *lease.lease_expiry_time()))
                    .collect();
                acquired_collection_jobs.sort();

                let mut expected_collection_jobs: Vec<_> = test_case
                    .collection_job_test_cases
                    .iter()
                    .filter(|c| c.should_be_acquired)
                    .map(|c| {
                        (
                            AcquiredCollectionJob::new(
                                c.task_id,
                                c.collection_job_id.unwrap(),
                                task::QueryType::FixedSize { max_batch_size: 10 },
                                VdafInstance::Fake,
                            ),
                            clock.now().as_naive_date_time().unwrap()
                                + chrono::Duration::seconds(100),
                        )
                    })
                    .collect();
                expected_collection_jobs.sort();

                assert_eq!(acquired_collection_jobs, expected_collection_jobs);

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn collection_job_acquire_state_filtering(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let task_id = random();
        let reports = Vec::from([LeaderStoredReport::new_dummy(
            task_id,
            Time::from_seconds_since_epoch(0),
        )]);
        let aggregation_job_ids: [_; 3] = random();
        let batch_id = random();
        let aggregation_jobs = Vec::from([
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[0],
                AggregationParam(0),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            ),
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[1],
                AggregationParam(1),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            ),
            AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[2],
                AggregationParam(2),
                batch_id,
                Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                    .unwrap(),
                AggregationJobState::Finished,
                AggregationJobRound::from(1),
            ),
        ]);
        let report_aggregations = Vec::from([
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[0],
                *reports[0].metadata().id(),
                *reports[0].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[1],
                *reports[0].metadata().id(),
                *reports[0].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
            ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                task_id,
                aggregation_job_ids[2],
                *reports[0].metadata().id(),
                *reports[0].metadata().time(),
                0,
                None,
                ReportAggregationState::Start,
            ),
        ]);

        let collection_job_test_cases = Vec::from([
            CollectionJobTestCase::<FixedSize> {
                should_be_acquired: true,
                task_id,
                batch_identifier: batch_id,
                agg_param: AggregationParam(0),
                collection_job_id: None,
                state: CollectionJobTestCaseState::Finished,
            },
            CollectionJobTestCase::<FixedSize> {
                should_be_acquired: true,
                task_id,
                batch_identifier: batch_id,
                agg_param: AggregationParam(1),
                collection_job_id: None,
                state: CollectionJobTestCaseState::Abandoned,
            },
            CollectionJobTestCase::<FixedSize> {
                should_be_acquired: true,
                task_id,
                batch_identifier: batch_id,
                agg_param: AggregationParam(2),
                collection_job_id: None,
                state: CollectionJobTestCaseState::Deleted,
            },
        ]);

        setup_collection_job_acquire_test_case(
            &ds,
            CollectionJobAcquireTestCase {
                task_ids: Vec::from([task_id]),
                query_type: task::QueryType::FixedSize { max_batch_size: 10 },
                reports,
                aggregation_jobs,
                report_aggregations,
                collection_job_test_cases,
            },
        )
        .await;

        ds.run_tx(|tx| {
            Box::pin(async move {
                // No collection jobs should be acquired because none of them are in the START state
                let acquired_collection_jobs = tx
                    .acquire_incomplete_collection_jobs(&StdDuration::from_secs(100), 10)
                    .await?;
                assert!(acquired_collection_jobs.is_empty());

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_batch_aggregation_fixed_size(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        ds.run_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    task::QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Fake,
                    Role::Leader,
                )
                .build();
                let other_task = TaskBuilder::new(
                    task::QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Fake,
                    Role::Leader,
                )
                .build();
                let vdaf = dummy_vdaf::Vdaf::new();
                let batch_id = random();
                let aggregate_share = AggregateShare(23);
                let aggregation_param = AggregationParam(12);

                tx.put_task(&task).await?;
                tx.put_task(&other_task).await?;

                let batch_aggregation = BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    0,
                    BatchAggregationState::Aggregating,
                    Some(aggregate_share),
                    0,
                    ReportIdChecksum::default(),
                );

                // Following batch aggregations have the batch ID queried below.
                tx.put_batch_aggregation(&batch_aggregation).await?;

                assert_matches!(
                    tx.put_batch_aggregation(&batch_aggregation).await,
                    Err(Error::MutationTargetAlreadyExists)
                );

                // Wrong batch ID.
                tx.put_batch_aggregation(&BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    random(),
                    AggregationParam(13),
                    1,
                    BatchAggregationState::Collected,
                    None,
                    0,
                    ReportIdChecksum::default(),
                ))
                .await?;

                // Task ID differs from that queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *other_task.id(),
                    batch_id,
                    aggregation_param,
                    2,
                    BatchAggregationState::Aggregating,
                    Some(aggregate_share),
                    0,
                    ReportIdChecksum::default(),
                ))
                .await?;

                // Index differs from that queried below.
                tx.put_batch_aggregation(&BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    3,
                    BatchAggregationState::Collected,
                    None,
                    0,
                    ReportIdChecksum::default(),
                ))
                .await?;

                let got_batch_aggregation = tx
                    .get_batch_aggregation::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        task.id(),
                        &batch_id,
                        &aggregation_param,
                        0,
                    )
                    .await?;
                assert_eq!(got_batch_aggregation.as_ref(), Some(&batch_aggregation));

                let batch_aggregation = BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    *batch_aggregation.task_id(),
                    *batch_aggregation.batch_id(),
                    *batch_aggregation.aggregation_parameter(),
                    batch_aggregation.ord(),
                    *batch_aggregation.state(),
                    None,
                    1,
                    ReportIdChecksum::get_decoded(&[1; 32]).unwrap(),
                );
                tx.update_batch_aggregation(&batch_aggregation).await?;

                let got_batch_aggregation = tx
                    .get_batch_aggregation::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        task.id(),
                        &batch_id,
                        &aggregation_param,
                        0,
                    )
                    .await?;
                assert_eq!(got_batch_aggregation, Some(batch_aggregation));
                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_aggregate_share_job(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let ds = ephemeral_datastore.datastore(MockClock::default()).await;

        ds.run_tx(|tx| {
            Box::pin(async move {
                let task = TaskBuilder::new(
                    task::QueryType::FixedSize { max_batch_size: 10 },
                    VdafInstance::Fake,
                    Role::Helper,
                )
                .build();
                tx.put_task(&task).await?;

                let vdaf = dummy_vdaf::Vdaf::new();
                let aggregate_share = AggregateShare(42);
                let batch_id = random();
                let other_batch_id = random();
                let report_count = 10;
                let checksum = ReportIdChecksum::get_decoded(&[1; 32]).unwrap();
                let aggregation_param = AggregationParam(11);

                let aggregate_share_job = AggregateShareJob::new(
                    *task.id(),
                    batch_id,
                    aggregation_param,
                    aggregate_share,
                    report_count,
                    checksum,
                );

                tx.put_aggregate_share_job::<0, FixedSize, dummy_vdaf::Vdaf>(&aggregate_share_job)
                    .await
                    .unwrap();

                let aggregate_share_job_again = tx
                    .get_aggregate_share_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        task.id(),
                        &batch_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(aggregate_share_job, aggregate_share_job_again);

                assert!(tx
                    .get_aggregate_share_job::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        task.id(),
                        &other_batch_id,
                        &aggregation_param,
                    )
                    .await
                    .unwrap()
                    .is_none());

                let want_aggregate_share_jobs = Vec::from([aggregate_share_job]);
                let got_aggregate_share_jobs = tx
                    .get_aggregate_share_jobs_by_batch_identifier::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &vdaf,
                        task.id(),
                        &batch_id,
                    )
                    .await?;
                assert_eq!(got_aggregate_share_jobs, want_aggregate_share_jobs);

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_outstanding_batch(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        let (task_id, batch_id) = ds
            .run_tx(|tx| {
                let clock = clock.clone();
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();
                    tx.put_task(&task).await?;
                    let batch_id = random();
                    tx.put_outstanding_batch(task.id(), &batch_id).await?;

                    // Write a few aggregation jobs & report aggregations to produce useful
                    // min_size/max_size values to validate later.
                    let aggregation_job_0 = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        random(),
                        AggregationParam(0),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::Finished,
                        AggregationJobRound::from(1),
                    );
                    let report_aggregation_0_0 = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *aggregation_job_0.id(),
                        random(),
                        clock.now(),
                        0,
                        None,
                        ReportAggregationState::Start, // Counted among max_size.
                    );
                    let report_aggregation_0_1 = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *aggregation_job_0.id(),
                        random(),
                        clock.now(),
                        1,
                        None,
                        ReportAggregationState::Waiting(
                            dummy_vdaf::PrepareState::default(),
                            Some(()),
                        ), // Counted among max_size.
                    );
                    let report_aggregation_0_2 = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *aggregation_job_0.id(),
                        random(),
                        clock.now(),
                        2,
                        None,
                        ReportAggregationState::Failed(ReportShareError::VdafPrepError), // Not counted among min_size or max_size.
                    );

                    let aggregation_job_1 = AggregationJob::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        random(),
                        AggregationParam(0),
                        batch_id,
                        Interval::new(Time::from_seconds_since_epoch(0), Duration::from_seconds(1))
                            .unwrap(),
                        AggregationJobState::Finished,
                        AggregationJobRound::from(1),
                    );
                    let report_aggregation_1_0 = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *aggregation_job_1.id(),
                        random(),
                        clock.now(),
                        0,
                        None,
                        ReportAggregationState::Finished, // Counted among min_size and max_size.
                    );
                    let report_aggregation_1_1 = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *aggregation_job_1.id(),
                        random(),
                        clock.now(),
                        1,
                        None,
                        ReportAggregationState::Finished, // Counted among min_size and max_size.
                    );
                    let report_aggregation_1_2 = ReportAggregation::<0, dummy_vdaf::Vdaf>::new(
                        *task.id(),
                        *aggregation_job_1.id(),
                        random(),
                        clock.now(),
                        2,
                        None,
                        ReportAggregationState::Failed(ReportShareError::VdafPrepError), // Not counted among min_size or max_size.
                    );

                    for aggregation_job in &[aggregation_job_0, aggregation_job_1] {
                        tx.put_aggregation_job(aggregation_job).await?;
                    }
                    for report_aggregation in &[
                        report_aggregation_0_0,
                        report_aggregation_0_1,
                        report_aggregation_0_2,
                        report_aggregation_1_0,
                        report_aggregation_1_1,
                        report_aggregation_1_2,
                    ] {
                        tx.put_client_report(
                            &dummy_vdaf::Vdaf::new(),
                            &LeaderStoredReport::new(
                                *report_aggregation.task_id(),
                                ReportMetadata::new(
                                    *report_aggregation.report_id(),
                                    *report_aggregation.time(),
                                ),
                                (), // Dummy public share
                                Vec::new(),
                                dummy_vdaf::InputShare::default(), // Dummy leader input share
                                // Dummy helper encrypted input share
                                HpkeCiphertext::new(
                                    HpkeConfigId::from(13),
                                    Vec::from("encapsulated_context_0"),
                                    Vec::from("payload_0"),
                                ),
                            ),
                        )
                        .await?;
                        tx.put_report_aggregation(report_aggregation).await?;
                    }

                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            batch_id,
                            AggregationParam(0),
                            0,
                            BatchAggregationState::Aggregating,
                            Some(AggregateShare(0)),
                            1,
                            ReportIdChecksum::default(),
                        ),
                    )
                    .await?;
                    tx.put_batch_aggregation(
                        &BatchAggregation::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                            *task.id(),
                            batch_id,
                            AggregationParam(0),
                            1,
                            BatchAggregationState::Aggregating,
                            Some(AggregateShare(0)),
                            1,
                            ReportIdChecksum::default(),
                        ),
                    )
                    .await?;

                    Ok((*task.id(), batch_id))
                })
            })
            .await
            .unwrap();

        let (outstanding_batches, outstanding_batch_1, outstanding_batch_2, outstanding_batch_3) =
            ds.run_tx(|tx| {
                Box::pin(async move {
                    let outstanding_batches = tx.get_outstanding_batches_for_task(&task_id).await?;
                    let outstanding_batch_1 = tx.get_filled_outstanding_batch(&task_id, 1).await?;
                    let outstanding_batch_2 = tx.get_filled_outstanding_batch(&task_id, 2).await?;
                    let outstanding_batch_3 = tx.get_filled_outstanding_batch(&task_id, 3).await?;
                    Ok((
                        outstanding_batches,
                        outstanding_batch_1,
                        outstanding_batch_2,
                        outstanding_batch_3,
                    ))
                })
            })
            .await
            .unwrap();
        assert_eq!(
            outstanding_batches,
            Vec::from([OutstandingBatch::new(
                task_id,
                batch_id,
                RangeInclusive::new(2, 4)
            )])
        );
        assert_eq!(outstanding_batch_1, Some(batch_id));
        assert_eq!(outstanding_batch_2, Some(batch_id));
        assert_eq!(outstanding_batch_3, None);

        // Delete the outstanding batch, then check that it is no longer available.
        ds.run_tx(|tx| {
            Box::pin(async move { tx.delete_outstanding_batch(&task_id, &batch_id).await })
        })
        .await
        .unwrap();

        let outstanding_batches = ds
            .run_tx(|tx| {
                Box::pin(async move { tx.get_outstanding_batches_for_task(&task_id).await })
            })
            .await
            .unwrap();
        assert!(outstanding_batches.is_empty());
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_batch(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        ds.run_tx(|tx| {
            Box::pin(async move {
                let want_batch = Batch::<0, FixedSize, dummy_vdaf::Vdaf>::new(
                    random(),
                    random(),
                    AggregationParam(2),
                    BatchState::Closing,
                    1,
                    Interval::new(
                        Time::from_seconds_since_epoch(100),
                        Duration::from_seconds(100),
                    )
                    .unwrap(),
                );

                tx.put_task(
                    &TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .with_id(*want_batch.task_id())
                    .build(),
                )
                .await?;
                tx.put_batch(&want_batch).await?;

                // Try reading the batch back, and verify that modifying any of the primary key
                // attributes causes None to be returned.
                assert_eq!(
                    tx.get_batch(
                        want_batch.task_id(),
                        want_batch.batch_identifier(),
                        want_batch.aggregation_parameter()
                    )
                    .await?
                    .as_ref(),
                    Some(&want_batch)
                );
                assert_eq!(
                    tx.get_batch::<0, FixedSize, dummy_vdaf::Vdaf>(
                        &random(),
                        want_batch.batch_identifier(),
                        want_batch.aggregation_parameter()
                    )
                    .await?,
                    None
                );
                assert_eq!(
                    tx.get_batch::<0, FixedSize, dummy_vdaf::Vdaf>(
                        want_batch.task_id(),
                        &random(),
                        want_batch.aggregation_parameter()
                    )
                    .await?,
                    None
                );
                assert_eq!(
                    tx.get_batch::<0, FixedSize, dummy_vdaf::Vdaf>(
                        want_batch.task_id(),
                        want_batch.batch_identifier(),
                        &AggregationParam(3)
                    )
                    .await?,
                    None
                );

                // Update the batch, then read it again, verifying that the changes are reflected.
                let want_batch = want_batch
                    .with_state(BatchState::Closed)
                    .with_outstanding_aggregation_jobs(0);
                tx.update_batch(&want_batch).await?;

                assert_eq!(
                    tx.get_batch(
                        want_batch.task_id(),
                        want_batch.batch_identifier(),
                        want_batch.aggregation_parameter()
                    )
                    .await?
                    .as_ref(),
                    Some(&want_batch)
                );

                Ok(())
            })
        })
        .await
        .unwrap();
    }

    #[async_trait]
    trait ExpirationQueryTypeExt: CollectableQueryType {
        fn batch_identifier_for_client_timestamps(
            client_timestamps: &[Time],
        ) -> Self::BatchIdentifier;

        fn shortened_batch_identifier(
            batch_identifier: &Self::BatchIdentifier,
        ) -> Self::BatchIdentifier;

        async fn write_outstanding_batch(
            tx: &Transaction<MockClock>,
            task_id: &TaskId,
            batch_identifier: &Self::BatchIdentifier,
        ) -> Option<(TaskId, BatchId)>;
    }

    #[async_trait]
    impl ExpirationQueryTypeExt for FixedSize {
        fn batch_identifier_for_client_timestamps(_: &[Time]) -> Self::BatchIdentifier {
            random()
        }

        fn shortened_batch_identifier(
            batch_identifier: &Self::BatchIdentifier,
        ) -> Self::BatchIdentifier {
            *batch_identifier
        }

        async fn write_outstanding_batch(
            tx: &Transaction<MockClock>,
            task_id: &TaskId,
            batch_identifier: &Self::BatchIdentifier,
        ) -> Option<(TaskId, BatchId)> {
            tx.put_outstanding_batch(task_id, batch_identifier)
                .await
                .unwrap();
            Some((*task_id, *batch_identifier))
        }
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn delete_expired_client_reports(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;
        let vdaf = dummy_vdaf::Vdaf::new();

        // Setup.
        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        let report_expiry_age = clock
            .now()
            .difference(&OLDEST_ALLOWED_REPORT_TIMESTAMP)
            .unwrap();
        let (task_id, new_report_id, other_task_id, other_task_report_id) = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .with_report_expiry_age(Some(report_expiry_age))
                    .build();
                    let other_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();
                    tx.put_task(&task).await?;
                    tx.put_task(&other_task).await?;

                    let old_report = LeaderStoredReport::new_dummy(
                        *task.id(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(1))
                            .unwrap(),
                    );
                    let new_report =
                        LeaderStoredReport::new_dummy(*task.id(), OLDEST_ALLOWED_REPORT_TIMESTAMP);
                    let other_task_report = LeaderStoredReport::new_dummy(
                        *other_task.id(),
                        OLDEST_ALLOWED_REPORT_TIMESTAMP
                            .sub(&Duration::from_seconds(1))
                            .unwrap(),
                    );
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &old_report)
                        .await?;
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &new_report)
                        .await?;
                    tx.put_client_report(&dummy_vdaf::Vdaf::new(), &other_task_report)
                        .await?;

                    Ok((
                        *task.id(),
                        *new_report.metadata().id(),
                        *other_task.id(),
                        *other_task_report.metadata().id(),
                    ))
                })
            })
            .await
            .unwrap();

        // Run.
        ds.run_tx(|tx| Box::pin(async move { tx.delete_expired_client_reports(&task_id).await }))
            .await
            .unwrap();

        // Verify.
        let want_report_ids = HashSet::from([new_report_id, other_task_report_id]);
        let got_report_ids = ds
            .run_tx(|tx| {
                let vdaf = vdaf.clone();
                Box::pin(async move {
                    let task_client_reports =
                        tx.get_client_reports_for_task(&vdaf, &task_id).await?;
                    let other_task_client_reports = tx
                        .get_client_reports_for_task(&vdaf, &other_task_id)
                        .await?;
                    Ok(HashSet::from_iter(
                        task_client_reports
                            .into_iter()
                            .chain(other_task_client_reports)
                            .map(|report| *report.metadata().id()),
                    ))
                })
            })
            .await
            .unwrap();
        assert_eq!(want_report_ids, got_report_ids);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn delete_expired_aggregation_artifacts(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        const REPORT_EXPIRY_AGE: Duration = Duration::from_seconds(1000);
        let clock = MockClock::new(OLDEST_ALLOWED_REPORT_TIMESTAMP);
        let ds = ephemeral_datastore.datastore(clock.clone()).await;
        let vdaf = dummy_vdaf::Vdaf::new();

        // Setup.
        async fn write_aggregation_artifacts<Q: ExpirationQueryTypeExt>(
            tx: &Transaction<'_, MockClock>,
            task_id: &TaskId,
            client_timestamps: &[Time],
        ) -> (
            Q::BatchIdentifier,
            AggregationJobId, // aggregation job ID
            Vec<ReportId>,    // client report IDs
        ) {
            let batch_identifier = Q::batch_identifier_for_client_timestamps(client_timestamps);

            let mut report_ids_and_timestamps = Vec::new();
            for client_timestamp in client_timestamps {
                let report = LeaderStoredReport::new_dummy(*task_id, *client_timestamp);
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report)
                    .await
                    .unwrap();
                report_ids_and_timestamps.push((*report.metadata().id(), *client_timestamp));
            }

            let min_client_timestamp = client_timestamps.iter().min().unwrap();
            let max_client_timestamp = client_timestamps.iter().max().unwrap();
            let client_timestamp_interval = Interval::new(
                *min_client_timestamp,
                max_client_timestamp
                    .difference(min_client_timestamp)
                    .unwrap()
                    .add(&Duration::from_seconds(1))
                    .unwrap(),
            )
            .unwrap();

            let aggregation_job = AggregationJob::<0, Q, dummy_vdaf::Vdaf>::new(
                *task_id,
                random(),
                AggregationParam(0),
                Q::partial_batch_identifier(&batch_identifier).clone(),
                client_timestamp_interval,
                AggregationJobState::InProgress,
                AggregationJobRound::from(0),
            );
            tx.put_aggregation_job(&aggregation_job).await.unwrap();

            for (ord, (report_id, client_timestamp)) in report_ids_and_timestamps.iter().enumerate()
            {
                let report_aggregation = ReportAggregation::new(
                    *task_id,
                    *aggregation_job.id(),
                    *report_id,
                    *client_timestamp,
                    ord.try_into().unwrap(),
                    None,
                    ReportAggregationState::<0, dummy_vdaf::Vdaf>::Start,
                );
                tx.put_report_aggregation(&report_aggregation)
                    .await
                    .unwrap();
            }

            (
                batch_identifier,
                *aggregation_job.id(),
                report_ids_and_timestamps
                    .into_iter()
                    .map(|(report_id, _)| report_id)
                    .collect(),
            )
        }

        let (
            leader_fixed_size_task_id,
            helper_fixed_size_task_id,
            want_aggregation_job_ids,
            want_report_ids,
        ) = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let leader_fixed_size_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build();
                    let helper_fixed_size_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Helper,
                    )
                    .with_report_expiry_age(Some(REPORT_EXPIRY_AGE))
                    .build();
                    tx.put_task(&leader_fixed_size_task).await?;
                    tx.put_task(&helper_fixed_size_task).await?;

                    let mut aggregation_job_ids = HashSet::new();
                    let mut all_report_ids = HashSet::new();

                    // Leader, fixed-size aggregation job with old reports [GC'ed].
                    write_aggregation_artifacts::<FixedSize>(
                        tx,
                        leader_fixed_size_task.id(),
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(20))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(19))
                                .unwrap(),
                        ],
                    )
                    .await;

                    // Leader, fixed-size aggregation job with old & new reports [not GC'ed].
                    let (_, aggregation_job_id, report_ids) =
                        write_aggregation_artifacts::<FixedSize>(
                            tx,
                            leader_fixed_size_task.id(),
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(5))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(8))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    aggregation_job_ids.insert(aggregation_job_id);
                    all_report_ids.extend(report_ids);

                    // Leader, fixed-size aggregation job with new reports [not GC'ed].
                    let (_, aggregation_job_id, report_ids) =
                        write_aggregation_artifacts::<FixedSize>(
                            tx,
                            leader_fixed_size_task.id(),
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(19))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(20))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    aggregation_job_ids.insert(aggregation_job_id);
                    all_report_ids.extend(report_ids);

                    // Helper, fixed-size aggregation job with old reports [GC'ed].
                    write_aggregation_artifacts::<FixedSize>(
                        tx,
                        helper_fixed_size_task.id(),
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(20))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(19))
                                .unwrap(),
                        ],
                    )
                    .await;

                    // Helper, fixed-size aggregation job with old & new reports [not GC'ed].
                    let (_, aggregation_job_id, report_ids) =
                        write_aggregation_artifacts::<FixedSize>(
                            tx,
                            helper_fixed_size_task.id(),
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(5))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(8))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    aggregation_job_ids.insert(aggregation_job_id);
                    all_report_ids.extend(report_ids);

                    // Helper, fixed-size aggregation job with new reports [not GC'ed].
                    let (_, aggregation_job_id, report_ids) =
                        write_aggregation_artifacts::<FixedSize>(
                            tx,
                            helper_fixed_size_task.id(),
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(19))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(20))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    aggregation_job_ids.insert(aggregation_job_id);
                    all_report_ids.extend(report_ids);

                    Ok((
                        *leader_fixed_size_task.id(),
                        *helper_fixed_size_task.id(),
                        aggregation_job_ids,
                        all_report_ids,
                    ))
                })
            })
            .await
            .unwrap();

        // Advance the clock to "enable" report expiry.
        clock.advance(&REPORT_EXPIRY_AGE);

        // Run.
        ds.run_tx(|tx| {
            Box::pin(async move {
                tx.delete_expired_aggregation_artifacts(&leader_fixed_size_task_id)
                    .await?;
                tx.delete_expired_aggregation_artifacts(&helper_fixed_size_task_id)
                    .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

        // Verify.
        let (got_aggregation_job_ids, got_report_ids) = ds
            .run_tx(|tx| {
                let vdaf = vdaf.clone();
                Box::pin(async move {
                    let leader_fixed_size_aggregation_job_ids = tx
                        .get_aggregation_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &leader_fixed_size_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|job| *job.id());
                    let helper_fixed_size_aggregation_job_ids = tx
                        .get_aggregation_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &helper_fixed_size_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|job| *job.id());
                    let got_aggregation_job_ids = leader_fixed_size_aggregation_job_ids
                        .chain(helper_fixed_size_aggregation_job_ids)
                        .collect();

                    let leader_fixed_size_report_aggregations = tx
                        .get_report_aggregations_for_task::<0, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &Role::Leader,
                            &leader_fixed_size_task_id,
                        )
                        .await
                        .unwrap();
                    let helper_fixed_size_report_aggregations = tx
                        .get_report_aggregations_for_task::<0, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &Role::Helper,
                            &helper_fixed_size_task_id,
                        )
                        .await
                        .unwrap();
                    let got_report_ids = leader_fixed_size_report_aggregations
                        .into_iter()
                        .chain(helper_fixed_size_report_aggregations)
                        .map(|report_aggregation| *report_aggregation.report_id())
                        .collect();

                    Ok((got_aggregation_job_ids, got_report_ids))
                })
            })
            .await
            .unwrap();
        assert_eq!(want_aggregation_job_ids, got_aggregation_job_ids);
        assert_eq!(want_report_ids, got_report_ids);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn delete_expired_collection_artifacts(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();

        let clock = MockClock::default();
        let ds = ephemeral_datastore.datastore(clock.clone()).await;

        // Setup.
        async fn write_collect_artifacts<Q: ExpirationQueryTypeExt>(
            tx: &Transaction<'_, MockClock>,
            task: &Task,
            client_timestamps: &[Time],
        ) -> (
            Option<CollectionJobId>,   // collection job ID
            Option<(TaskId, Vec<u8>)>, // aggregate share job ID (task ID, encoded batch identifier)
            Option<(TaskId, BatchId)>, // outstanding batch ID
        ) {
            let batch_identifier = Q::batch_identifier_for_client_timestamps(client_timestamps);
            for client_timestamp in client_timestamps {
                let report = LeaderStoredReport::new_dummy(*task.id(), *client_timestamp);
                tx.put_client_report(&dummy_vdaf::Vdaf::new(), &report)
                    .await
                    .unwrap();

                let aggregation_job = AggregationJob::<0, Q, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    random(),
                    AggregationParam(0),
                    Q::partial_batch_identifier(&batch_identifier).clone(),
                    Interval::new(*client_timestamp, Duration::from_seconds(1)).unwrap(),
                    AggregationJobState::InProgress,
                    AggregationJobRound::from(0),
                );
                tx.put_aggregation_job(&aggregation_job).await.unwrap();

                let report_aggregation = ReportAggregation::new(
                    *task.id(),
                    *aggregation_job.id(),
                    *report.metadata().id(),
                    *client_timestamp,
                    0,
                    None,
                    ReportAggregationState::<0, dummy_vdaf::Vdaf>::Start,
                );
                tx.put_report_aggregation(&report_aggregation)
                    .await
                    .unwrap();
            }

            if task.role() == &Role::Leader {
                let collection_job = CollectionJob::<0, Q, dummy_vdaf::Vdaf>::new(
                    *task.id(),
                    random(),
                    batch_identifier.clone(),
                    AggregationParam(0),
                    CollectionJobState::Start,
                );
                tx.put_collection_job(&collection_job).await.unwrap();

                let outstanding_batch_id =
                    Q::write_outstanding_batch(tx, task.id(), &batch_identifier).await;

                return (Some(*collection_job.id()), None, outstanding_batch_id);
            } else {
                let aggregate_share_job = AggregateShareJob::new(
                    *task.id(),
                    batch_identifier.clone(),
                    AggregationParam(23),
                    AggregateShare(11),
                    client_timestamps.len().try_into().unwrap(),
                    random(),
                );
                tx.put_aggregate_share_job::<0, Q, dummy_vdaf::Vdaf>(&aggregate_share_job)
                    .await
                    .unwrap();

                return (
                    None,
                    Some((*task.id(), batch_identifier.get_encoded())),
                    None,
                );
            }
        }

        const OLDEST_ALLOWED_REPORT_TIMESTAMP: Time = Time::from_seconds_since_epoch(1000);
        let (
            leader_fixed_size_task_id,
            helper_fixed_size_task_id,
            other_task_id,
            want_collection_job_ids,
            want_aggregate_share_job_ids,
            want_outstanding_batch_ids,
        ) = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let leader_fixed_size_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();
                    let helper_fixed_size_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Helper,
                    )
                    .build();
                    let other_task = TaskBuilder::new(
                        task::QueryType::FixedSize { max_batch_size: 10 },
                        VdafInstance::Fake,
                        Role::Leader,
                    )
                    .build();
                    tx.put_task(&leader_fixed_size_task).await?;
                    tx.put_task(&helper_fixed_size_task).await?;
                    tx.put_task(&other_task).await?;

                    let mut collection_job_ids = HashSet::new();
                    let mut aggregate_share_job_ids = HashSet::new();
                    let mut outstanding_batch_ids = HashSet::new();

                    // Leader, fixed-size collection artifacts with old reports. [GC'ed]
                    write_collect_artifacts::<FixedSize>(
                        tx,
                        &leader_fixed_size_task,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(10))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(9))
                                .unwrap(),
                        ],
                    )
                    .await;

                    // Leader, fixed-size collection artifacts with old & new reports. [not GC'ed]
                    let (collection_job_id, aggregate_share_job_id, outstanding_batch_id) =
                        write_collect_artifacts::<FixedSize>(
                            tx,
                            &leader_fixed_size_task,
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(5))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(5))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    collection_job_ids.extend(collection_job_id);
                    aggregate_share_job_ids.extend(aggregate_share_job_id);
                    outstanding_batch_ids.extend(outstanding_batch_id);

                    // Leader, fixed-size collection artifacts with new reports. [not GC'ed]
                    let (collection_job_id, aggregate_share_job_id, outstanding_batch_id) =
                        write_collect_artifacts::<FixedSize>(tx, &leader_fixed_size_task, &[])
                            .await;
                    collection_job_ids.extend(collection_job_id);
                    aggregate_share_job_ids.extend(aggregate_share_job_id);
                    outstanding_batch_ids.extend(outstanding_batch_id);

                    // Helper, fixed-size collection artifacts with old reports. [GC'ed]
                    write_collect_artifacts::<FixedSize>(
                        tx,
                        &helper_fixed_size_task,
                        &[
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(10))
                                .unwrap(),
                            OLDEST_ALLOWED_REPORT_TIMESTAMP
                                .sub(&Duration::from_seconds(9))
                                .unwrap(),
                        ],
                    )
                    .await;

                    // Helper, fixed-size collection artifacts with old & new reports. [not GC'ed]
                    let (collection_job_id, aggregate_share_job_id, outstanding_batch_id) =
                        write_collect_artifacts::<FixedSize>(
                            tx,
                            &helper_fixed_size_task,
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(5))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .add(&Duration::from_seconds(5))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    collection_job_ids.extend(collection_job_id);
                    aggregate_share_job_ids.extend(aggregate_share_job_id);
                    outstanding_batch_ids.extend(outstanding_batch_id);

                    // Helper, fixed-size collection artifacts with new reports. [not GC'ed]
                    let (collection_job_id, aggregate_share_job_id, outstanding_batch_id) =
                        write_collect_artifacts::<FixedSize>(tx, &helper_fixed_size_task, &[])
                            .await;
                    collection_job_ids.extend(collection_job_id);
                    aggregate_share_job_ids.extend(aggregate_share_job_id);
                    outstanding_batch_ids.extend(outstanding_batch_id);

                    // Collection artifacts for different task. [not GC'ed]
                    let (collection_job_id, aggregate_share_job_id, outstanding_batch_id) =
                        write_collect_artifacts::<FixedSize>(
                            tx,
                            &other_task,
                            &[
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(9))
                                    .unwrap(),
                                OLDEST_ALLOWED_REPORT_TIMESTAMP
                                    .sub(&Duration::from_seconds(8))
                                    .unwrap(),
                            ],
                        )
                        .await;
                    collection_job_ids.extend(collection_job_id);
                    aggregate_share_job_ids.extend(aggregate_share_job_id);
                    outstanding_batch_ids.extend(outstanding_batch_id);

                    Ok((
                        *leader_fixed_size_task.id(),
                        *helper_fixed_size_task.id(),
                        *other_task.id(),
                        collection_job_ids,
                        aggregate_share_job_ids,
                        outstanding_batch_ids,
                    ))
                })
            })
            .await
            .unwrap();

        // Run.
        ds.run_tx(|tx| {
            Box::pin(async move {
                tx.delete_expired_collection_artifacts(
                    &leader_fixed_size_task_id,
                    OLDEST_ALLOWED_REPORT_TIMESTAMP,
                )
                .await?;
                tx.delete_expired_collection_artifacts(
                    &helper_fixed_size_task_id,
                    OLDEST_ALLOWED_REPORT_TIMESTAMP,
                )
                .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

        // Verify.
        let (got_collection_job_ids, got_aggregate_share_job_ids, got_outstanding_batch_ids) = ds
            .run_tx(|tx| {
                Box::pin(async move {
                    let vdaf = dummy_vdaf::Vdaf::new();
                    let leader_fixed_size_collection_job_ids = tx
                        .get_collection_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &leader_fixed_size_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|collection_job| *collection_job.id());
                    let helper_fixed_size_collection_job_ids = tx
                        .get_collection_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &helper_fixed_size_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|collection_job| *collection_job.id());
                    let other_task_collection_job_ids = tx
                        .get_collection_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &other_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|collection_job| *collection_job.id());
                    let got_collection_job_ids = leader_fixed_size_collection_job_ids
                        .chain(helper_fixed_size_collection_job_ids)
                        .chain(other_task_collection_job_ids)
                        .collect();

                    let leader_fixed_size_aggregate_share_job_ids = tx
                        .get_aggregate_share_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &leader_fixed_size_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|job| (*job.task_id(), job.batch_identifier().get_encoded()));
                    let helper_fixed_size_aggregate_share_job_ids = tx
                        .get_aggregate_share_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &helper_fixed_size_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|job| (*job.task_id(), job.batch_identifier().get_encoded()));
                    let other_task_aggregate_share_job_ids = tx
                        .get_aggregate_share_jobs_for_task::<0, FixedSize, dummy_vdaf::Vdaf>(
                            &vdaf,
                            &other_task_id,
                        )
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|job| (*job.task_id(), job.batch_identifier().get_encoded()));
                    let got_aggregate_share_job_ids = leader_fixed_size_aggregate_share_job_ids
                        .chain(helper_fixed_size_aggregate_share_job_ids)
                        .chain(other_task_aggregate_share_job_ids)
                        .collect();

                    let leader_fixed_size_outstanding_batch_ids = tx
                        .get_outstanding_batches_for_task(&leader_fixed_size_task_id)
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|batch| (*batch.task_id(), *batch.id()));
                    let helper_fixed_size_outstanding_batch_ids = tx
                        .get_outstanding_batches_for_task(&helper_fixed_size_task_id)
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|batch| (*batch.task_id(), *batch.id()));
                    let other_task_outstanding_batch_ids = tx
                        .get_outstanding_batches_for_task(&other_task_id)
                        .await
                        .unwrap()
                        .into_iter()
                        .map(|batch| (*batch.task_id(), *batch.id()));
                    let got_outstanding_batch_ids = leader_fixed_size_outstanding_batch_ids
                        .chain(helper_fixed_size_outstanding_batch_ids)
                        .chain(other_task_outstanding_batch_ids)
                        .collect();

                    Ok((
                        got_collection_job_ids,
                        got_aggregate_share_job_ids,
                        got_outstanding_batch_ids,
                    ))
                })
            })
            .await
            .unwrap();
        assert_eq!(want_collection_job_ids, got_collection_job_ids);
        assert_eq!(want_aggregate_share_job_ids, got_aggregate_share_job_ids);
        assert_eq!(want_outstanding_batch_ids, got_outstanding_batch_ids);
    }

    #[rstest_reuse::apply(schema_versions_template)]
    #[tokio::test]
    async fn roundtrip_interval_sql(ephemeral_datastore: EphemeralDatastore) {
        install_test_trace_subscriber();
        let datastore = ephemeral_datastore.datastore(MockClock::default()).await;

        datastore
            .run_tx(|tx| {
                Box::pin(async move {
                    let interval = tx
                        .query_one(
                            "SELECT '[2020-01-01 10:00, 2020-01-01 10:30)'::tsrange AS interval",
                            &[],
                        )
                        .await?
                        .get::<_, SqlInterval>("interval");
                    let ref_interval = Interval::new(
                        Time::from_naive_date_time(
                            &NaiveDate::from_ymd_opt(2020, 1, 1)
                                .unwrap()
                                .and_hms_opt(10, 0, 0)
                                .unwrap(),
                        ),
                        Duration::from_minutes(30).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(interval.as_interval(), ref_interval);

                    let interval = tx
                        .query_one(
                            "SELECT '[1970-02-03 23:00, 1970-02-04 00:00)'::tsrange AS interval",
                            &[],
                        )
                        .await?
                        .get::<_, SqlInterval>("interval");
                    let ref_interval = Interval::new(
                        Time::from_naive_date_time(
                            &NaiveDate::from_ymd_opt(1970, 2, 3)
                                .unwrap()
                                .and_hms_opt(23, 0, 0)
                                .unwrap(),
                        ),
                        Duration::from_hours(1).unwrap(),
                    )?;
                    assert_eq!(interval.as_interval(), ref_interval);

                    let res = tx
                        .query_one(
                            "SELECT '[1969-01-01 00:00, 1970-01-01 00:00)'::tsrange AS interval",
                            &[],
                        )
                        .await?
                        .try_get::<_, SqlInterval>("interval");
                    assert!(res.is_err());

                    let ok = tx
                        .query_one(
                            "SELECT (lower(interval) = '1972-07-21 05:30:00' AND
                            upper(interval) = '1972-07-21 06:00:00' AND
                            lower_inc(interval) AND
                            NOT upper_inc(interval)) AS ok
                            FROM (VALUES ($1::tsrange)) AS temp (interval)",
                            &[&SqlInterval::from(
                                Interval::new(
                                    Time::from_naive_date_time(
                                        &NaiveDate::from_ymd_opt(1972, 7, 21)
                                            .unwrap()
                                            .and_hms_opt(5, 30, 0)
                                            .unwrap(),
                                    ),
                                    Duration::from_minutes(30).unwrap(),
                                )
                                .unwrap(),
                            )],
                        )
                        .await?
                        .get::<_, bool>("ok");
                    assert!(ok);

                    let ok = tx
                        .query_one(
                            "SELECT (lower(interval) = '2021-10-05 00:00:00' AND
                            upper(interval) = '2021-10-06 00:00:00' AND
                            lower_inc(interval) AND
                            NOT upper_inc(interval)) AS ok
                            FROM (VALUES ($1::tsrange)) AS temp (interval)",
                            &[&SqlInterval::from(
                                Interval::new(
                                    Time::from_naive_date_time(
                                        &NaiveDate::from_ymd_opt(2021, 10, 5)
                                            .unwrap()
                                            .and_hms_opt(0, 0, 0)
                                            .unwrap(),
                                    ),
                                    Duration::from_hours(24).unwrap(),
                                )
                                .unwrap(),
                            )],
                        )
                        .await?
                        .get::<_, bool>("ok");
                    assert!(ok);

                    Ok(())
                })
            })
            .await
            .unwrap();
    }
}
