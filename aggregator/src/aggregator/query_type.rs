use super::Error;
use async_trait::async_trait;
use janus_aggregator_core::{
    datastore::{self, models::LeaderStoredReport, Transaction},
    query_type::{AccumulableQueryType, CollectableQueryType as CoreCollectableQueryType},
    task::Task,
};
use janus_core::time::Clock;
use janus_messages::{
    query_type::{FixedSize, QueryType},
    Role,
};
use prio::vdaf;

#[async_trait]
pub trait UploadableQueryType: QueryType {
    async fn validate_uploaded_report<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        report: &LeaderStoredReport<SEED_SIZE, A>,
    ) -> Result<(), datastore::Error>
    where
        A::InputShare: Send + Sync,
        A::PublicShare: Send + Sync;
}

#[async_trait]
impl UploadableQueryType for FixedSize {
    async fn validate_uploaded_report<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        _: &Transaction<'_, C>,
        _: &A,
        _: &LeaderStoredReport<SEED_SIZE, A>,
    ) -> Result<(), datastore::Error> {
        // Fixed-size tasks associate reports to batches at time of aggregation rather than at time
        // of upload, and there are no other relevant checks to apply here, so this method simply
        // returns Ok(()).
        Ok(())
    }
}

/// CollectableQueryType represents a query type that can be collected by Janus. This trait extends
/// [`AccumulableQueryType`] with additional functionality required for collection.
#[async_trait]
pub trait CollectableQueryType: CoreCollectableQueryType + AccumulableQueryType {
    /// Validates query count for a given batch, per the size checks in
    /// <https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6>.
    async fn validate_query_count<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task: &Task,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Result<(), datastore::Error>;
}

#[async_trait]
impl CollectableQueryType for FixedSize {
    async fn validate_query_count<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task: &Task,
        batch_id: &Self::BatchIdentifier,
    ) -> Result<(), datastore::Error> {
        let query_count = match task.role() {
            Role::Leader => tx
                .get_collection_jobs_by_batch_identifier::<SEED_SIZE, FixedSize, A>(
                    vdaf,
                    task.id(),
                    batch_id,
                )
                .await?
                .len(),

            Role::Helper => tx
                .get_aggregate_share_jobs_by_batch_identifier::<SEED_SIZE, FixedSize, A>(
                    vdaf,
                    task.id(),
                    batch_id,
                )
                .await?
                .len(),

            _ => panic!("Unexpected task role {:?}", task.role()),
        };

        // Check that the batch query count is being consumed appropriately.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6
        let max_batch_query_count: usize = task.max_batch_query_count().try_into()?;
        if query_count >= max_batch_query_count {
            return Err(datastore::Error::User(
                Error::BatchQueriedTooManyTimes(*task.id(), query_count as u64).into(),
            ));
        }
        Ok(())
    }
}
