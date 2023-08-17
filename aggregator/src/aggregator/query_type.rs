use super::Error;
use async_trait::async_trait;
use janus_aggregator_core::{
    datastore::{self, models::LeaderStoredReport, Transaction},
    query_type::{AccumulableQueryType, CollectableQueryType as CoreCollectableQueryType},
    task::Task,
};
use janus_core::time::Clock;
use janus_messages::{
    query_type::{FixedSize, QueryType, TimeInterval},
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
impl UploadableQueryType for TimeInterval {
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
        A::PublicShare: Send + Sync,
    {
        // Reject reports whose timestamps fall into a batch interval that has already been
        // collected.
        // https://datatracker.ietf.org/doc/html/draft-ietf-ppm-dap-03#section-4.3.2-17
        let conflicting_collect_jobs = tx
            .get_collection_jobs_including_time::<SEED_SIZE, A>(
                vdaf,
                report.task_id(),
                report.metadata().time(),
            )
            .await?;
        if !conflicting_collect_jobs.is_empty() {
            return Err(datastore::Error::User(
                Error::ReportRejected(
                    *report.task_id(),
                    *report.metadata().id(),
                    *report.metadata().time(),
                )
                .into(),
            ));
        }
        Ok(())
    }
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
impl CollectableQueryType for TimeInterval {
    async fn validate_query_count<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task: &Task,
        collect_interval: &Self::BatchIdentifier,
    ) -> Result<(), datastore::Error> {
        // Check how many rows in the relevant table have an intersecting batch interval.
        // Each such row consumes one unit of query count.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6
        let intersecting_intervals: Vec<_> = match task.role() {
            Role::Leader => tx
                .get_collection_jobs_intersecting_interval::<SEED_SIZE, A>(
                    vdaf,
                    task.id(),
                    collect_interval,
                )
                .await?
                .into_iter()
                .map(|job| *job.batch_interval())
                .collect(),

            Role::Helper => tx
                .get_aggregate_share_jobs_intersecting_interval::<SEED_SIZE, A>(
                    vdaf,
                    task.id(),
                    collect_interval,
                )
                .await?
                .into_iter()
                .map(|job| *job.batch_interval())
                .collect(),

            _ => panic!("Unexpected task role {:?}", task.role()),
        };

        // Check that all intersecting collect intervals are equal to this collect interval.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6-5
        if intersecting_intervals
            .iter()
            .any(|interval| interval != collect_interval)
        {
            return Err(datastore::Error::User(
                Error::BatchOverlap(*task.id(), *collect_interval).into(),
            ));
        }

        // Check that the batch query count is being consumed appropriately.
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6
        let max_batch_query_count: usize = task.max_batch_query_count().try_into()?;
        if intersecting_intervals.len() >= max_batch_query_count {
            return Err(datastore::Error::User(
                Error::BatchQueriedTooManyTimes(*task.id(), intersecting_intervals.len() as u64)
                    .into(),
            ));
        }
        Ok(())
    }
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
                .get_collection_jobs_by_batch_id::<SEED_SIZE, A>(vdaf, task.id(), batch_id)
                .await?
                .len(),

            Role::Helper => tx
                .get_aggregate_share_jobs_by_batch_id::<SEED_SIZE, A>(vdaf, task.id(), batch_id)
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
