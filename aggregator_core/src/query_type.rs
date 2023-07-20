use crate::{
    datastore::{
        self,
        models::{AggregateShareJob, Batch, BatchAggregation, CollectionJob},
        Transaction,
    },
    task::Task,
};
use async_trait::async_trait;
use futures::future::try_join_all;
use janus_core::time::Clock;
use janus_messages::{
    query_type::{FixedSize, QueryType},
    FixedSizeQuery, Interval, Query, ReportMetadata, TaskId, Time,
};
use prio::vdaf;
use std::iter;

#[async_trait]
pub trait AccumulableQueryType: QueryType {
    /// This method converts various values related to a client report into a batch identifier. The
    /// arguments are somewhat arbitrary in the sense they are what "works out" to allow the
    /// necessary functionality to be implemented for all query types.
    fn to_batch_identifier(
        _: &Task,
        _: &Self::PartialBatchIdentifier,
        client_timestamp: &Time,
    ) -> Result<Self::BatchIdentifier, datastore::Error>;

    async fn get_conflicting_aggregate_share_jobs<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task_id: &TaskId,
        partial_batch_identifier: &Self::PartialBatchIdentifier,
        report_metadata: &ReportMetadata,
    ) -> Result<Vec<AggregateShareJob<SEED_SIZE, Self, A>>, datastore::Error>;

    /// Retrieves collection jobs which include the given batch identifier.
    async fn get_collection_jobs_including<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, Self, A>>, datastore::Error>;

    /// Some query types (e.g. `TimeInterval`) can represent their batch identifiers as an
    /// interval. This method extracts the interval from such identifiers, or returns `None` if the
    /// query type does not represent batch identifiers as an interval.
    fn to_batch_interval(batch_identifier: &Self::BatchIdentifier) -> Option<&Interval>;

    /// Downgrade a batch identifier into a partial batch identifier.
    fn downgrade_batch_identifier(
        batch_identifier: &Self::BatchIdentifier,
    ) -> &Self::PartialBatchIdentifier;

    /// Upgrade a partial batch identifier into a batch identifier, if possible.
    fn upgrade_partial_batch_identifier(
        partial_batch_identifier: &Self::PartialBatchIdentifier,
    ) -> Option<&Self::BatchIdentifier>;

    /// Get the default value of the partial batch identifier, if applicable.
    fn default_partial_batch_identifier() -> Option<&'static Self::PartialBatchIdentifier>;
}

#[async_trait]
impl AccumulableQueryType for FixedSize {
    fn to_batch_identifier(
        _: &Task,
        batch_id: &Self::PartialBatchIdentifier,
        _: &Time,
    ) -> Result<Self::BatchIdentifier, datastore::Error> {
        Ok(*batch_id)
    }

    async fn get_conflicting_aggregate_share_jobs<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task_id: &TaskId,
        batch_id: &Self::PartialBatchIdentifier,
        _: &ReportMetadata,
    ) -> Result<Vec<AggregateShareJob<SEED_SIZE, Self, A>>, datastore::Error> {
        tx.get_aggregate_share_jobs_by_batch_identifier(vdaf, task_id, batch_id)
            .await
    }

    async fn get_collection_jobs_including<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task_id: &TaskId,
        batch_id: &Self::BatchIdentifier,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, Self, A>>, datastore::Error> {
        tx.get_collection_jobs_by_batch_identifier(vdaf, task_id, batch_id)
            .await
    }

    fn to_batch_interval(_: &Self::BatchIdentifier) -> Option<&Interval> {
        None
    }

    fn downgrade_batch_identifier(
        batch_identifier: &Self::BatchIdentifier,
    ) -> &Self::PartialBatchIdentifier {
        batch_identifier
    }

    fn upgrade_partial_batch_identifier(
        partial_batch_identifier: &Self::PartialBatchIdentifier,
    ) -> Option<&Self::BatchIdentifier> {
        Some(partial_batch_identifier)
    }

    fn default_partial_batch_identifier() -> Option<&'static Self::PartialBatchIdentifier> {
        None
    }
}

/// CollectableQueryType represents a query type that can be collected by Janus. This trait extends
/// [`AccumulableQueryType`] with additional functionality required for collection.
#[async_trait]
pub trait CollectableQueryType: AccumulableQueryType {
    type Iter: Iterator<Item = Self::BatchIdentifier> + Send + Sync;

    /// Retrieves the batch identifier for a given query.
    async fn collection_identifier_for_query<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &Task,
        query: &Query<Self>,
    ) -> Result<Option<Self::BatchIdentifier>, datastore::Error>;

    /// Some query types (e.g. `TimeInterval`) can receive a batch identifier in collection
    /// requests which refers to multiple batches. This method takes a batch identifier received in
    /// a collection request and provides an iterator over the individual batches' identifiers.
    fn batch_identifiers_for_collection_identifier(
        _: &Task,
        collection_identifier: &Self::BatchIdentifier,
    ) -> Self::Iter;

    /// Validates a collection identifier, per the boundary checks in
    /// <https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6>.
    fn validate_collection_identifier(
        task: &Task,
        collection_identifier: &Self::BatchIdentifier,
    ) -> bool;

    /// Returns the number of client reports included in the given collection identifier, whether
    /// they have been aggregated or not.
    async fn count_client_reports<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &Task,
        collection_identifier: &Self::BatchIdentifier,
    ) -> Result<u64, datastore::Error>;

    /// Retrieves batch aggregations corresponding to all batches identified by the given collection
    /// identifier.
    async fn get_batch_aggregations_for_collection_identifier<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
        C: Clock,
    >(
        tx: &Transaction<C>,
        task: &Task,
        vdaf: &A,
        collection_identifier: &Self::BatchIdentifier,
        aggregation_param: &A::AggregationParam,
    ) -> Result<Vec<BatchAggregation<SEED_SIZE, Self, A>>, datastore::Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        Ok(try_join_all(
            Self::batch_identifiers_for_collection_identifier(task, collection_identifier).map(
                |batch_identifier| {
                    let (task_id, aggregation_param) = (*task.id(), aggregation_param.clone());
                    async move {
                        tx.get_batch_aggregations_for_batch(
                            vdaf,
                            &task_id,
                            &batch_identifier,
                            &aggregation_param,
                        )
                        .await
                    }
                },
            ),
        )
        .await?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>())
    }

    /// Retrieves all batches identified by the given collection identifier.
    async fn get_batches_for_collection_identifier<
        const SEED_SIZE: usize,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
        C: Clock,
    >(
        tx: &Transaction<C>,
        task: &Task,
        collection_identifier: &Self::BatchIdentifier,
        aggregation_param: &A::AggregationParam,
    ) -> Result<Vec<Batch<SEED_SIZE, Self, A>>, datastore::Error>
    where
        A::AggregationParam: Send + Sync,
        A::AggregateShare: Send + Sync,
    {
        Ok(try_join_all(
            Self::batch_identifiers_for_collection_identifier(task, collection_identifier).map(
                |batch_identifier| {
                    let (task_id, aggregation_param) = (*task.id(), aggregation_param.clone());
                    async move {
                        tx.get_batch(&task_id, &batch_identifier, &aggregation_param)
                            .await
                    }
                },
            ),
        )
        .await?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>())
    }

    /// Acknowledges that a collection attempt has been made, allowing any query-type specific
    /// updates to be made. For exmaple, a task using fixed-size queries might remove the given
    /// batch to be removed from the list of batches ready to be returned by a `current-batch`
    /// query.
    async fn acknowledge_collection<C: Clock>(
        tx: &Transaction<'_, C>,
        task_id: &TaskId,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Result<(), datastore::Error>;
}

#[async_trait]
impl CollectableQueryType for FixedSize {
    type Iter = iter::Once<Self::BatchIdentifier>;

    async fn collection_identifier_for_query<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &Task,
        query: &Query<Self>,
    ) -> Result<Option<Self::BatchIdentifier>, datastore::Error> {
        match query.fixed_size_query() {
            FixedSizeQuery::ByBatchId { batch_id } => Ok(Some(*batch_id)),
            FixedSizeQuery::CurrentBatch => {
                tx.get_filled_outstanding_batch(task.id(), task.min_batch_size())
                    .await
            }
        }
    }

    fn batch_identifiers_for_collection_identifier(
        _: &Task,
        batch_id: &Self::BatchIdentifier,
    ) -> Self::Iter {
        iter::once(*batch_id)
    }

    fn validate_collection_identifier(_: &Task, _: &Self::BatchIdentifier) -> bool {
        true
    }

    async fn count_client_reports<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &Task,
        batch_id: &Self::BatchIdentifier,
    ) -> Result<u64, datastore::Error> {
        tx.count_client_reports_for_batch_id(task.id(), batch_id)
            .await
    }

    async fn acknowledge_collection<C: Clock>(
        tx: &Transaction<'_, C>,
        task_id: &TaskId,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Result<(), datastore::Error> {
        tx.delete_outstanding_batch(task_id, batch_identifier).await
    }
}
