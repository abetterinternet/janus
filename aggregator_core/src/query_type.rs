use crate::{
    datastore::{
        self,
        models::{BatchAggregation, CollectionJob},
        Transaction,
    },
    task::AggregatorTask,
};
use async_trait::async_trait;
use futures::future::try_join_all;
use janus_core::time::{Clock, IntervalExt as _, TimeExt as _};
use janus_messages::{
    query_type::{FixedSize, QueryType, TimeInterval},
    Duration, FixedSizeQuery, Interval, Query, TaskId, Time,
};
use prio::vdaf;
use std::iter;

#[async_trait]
pub trait AccumulableQueryType: QueryType {
    /// This method converts various values related to a client report into a batch identifier. The
    /// arguments are somewhat arbitrary in the sense they are what "works out" to allow the
    /// necessary functionality to be implemented for all query types.
    fn to_batch_identifier(
        _: &AggregatorTask,
        _: &Self::PartialBatchIdentifier,
        client_timestamp: &Time,
    ) -> Result<Self::BatchIdentifier, datastore::Error>;

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

    /// Some query types (e.g. [`TimeInterval`]) can represent their batch identifiers as an
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

    /// Determine if the batch is expected to be garbage-collected, based on the identifier.
    /// `Some(true)` and `Some(false)` indicate the expected result, and `None` indicates that the
    /// answer cannot be determined based on the batch identifier alone (for e.g. the fixed-size
    /// query type).
    fn is_batch_garbage_collected<C: Clock>(
        clock: &C,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Option<bool>;
}

#[async_trait]
impl AccumulableQueryType for TimeInterval {
    fn to_batch_identifier(
        task: &AggregatorTask,
        _: &Self::PartialBatchIdentifier,
        client_timestamp: &Time,
    ) -> Result<Self::BatchIdentifier, datastore::Error> {
        let batch_interval_start = client_timestamp
            .to_batch_interval_start(task.time_precision())
            .map_err(|e| datastore::Error::User(e.into()))?;
        Interval::new(batch_interval_start, *task.time_precision())
            .map_err(|e| datastore::Error::User(e.into()))
    }

    async fn get_collection_jobs_including<
        const SEED_SIZE: usize,
        C: Clock,
        A: vdaf::Aggregator<SEED_SIZE, 16> + Send + Sync,
    >(
        tx: &Transaction<'_, C>,
        vdaf: &A,
        task_id: &TaskId,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Result<Vec<CollectionJob<SEED_SIZE, Self, A>>, datastore::Error> {
        tx.get_collection_jobs_intersecting_interval(vdaf, task_id, batch_identifier)
            .await
    }

    fn to_batch_interval(collection_identifier: &Self::BatchIdentifier) -> Option<&Interval> {
        Some(collection_identifier)
    }

    fn downgrade_batch_identifier(
        _batch_identifier: &Self::BatchIdentifier,
    ) -> &Self::PartialBatchIdentifier {
        &()
    }

    fn upgrade_partial_batch_identifier(
        _partial_batch_identifier: &Self::PartialBatchIdentifier,
    ) -> Option<&Self::BatchIdentifier> {
        None
    }

    fn default_partial_batch_identifier() -> Option<&'static Self::PartialBatchIdentifier> {
        Some(&())
    }

    fn is_batch_garbage_collected<C: Clock>(
        clock: &C,
        batch_identifier: &Self::BatchIdentifier,
    ) -> Option<bool> {
        Some(batch_identifier.end() < clock.now())
    }
}

#[async_trait]
impl AccumulableQueryType for FixedSize {
    fn to_batch_identifier(
        _: &AggregatorTask,
        batch_id: &Self::PartialBatchIdentifier,
        _: &Time,
    ) -> Result<Self::BatchIdentifier, datastore::Error> {
        Ok(*batch_id)
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
        tx.get_collection_jobs_by_batch_id(vdaf, task_id, batch_id)
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

    fn is_batch_garbage_collected<C: Clock>(_: &C, _: &Self::BatchIdentifier) -> Option<bool> {
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
        task: &AggregatorTask,
        query: &Query<Self>,
    ) -> Result<Option<Self::BatchIdentifier>, datastore::Error>;

    /// Some query types (e.g. [`TimeInterval`]) can receive a batch identifier in collection
    /// requests which refers to multiple batches. This method takes a batch identifier received in
    /// a collection request and provides an iterator over the individual batches' identifiers.
    fn batch_identifiers_for_collection_identifier(
        _: &AggregatorTask,
        collection_identifier: &Self::BatchIdentifier,
    ) -> Self::Iter;

    /// Validates a collection identifier, per the boundary checks in
    /// <https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6>.
    fn validate_collection_identifier(
        task: &AggregatorTask,
        collection_identifier: &Self::BatchIdentifier,
    ) -> bool;

    /// Returns the number of client reports included in the given collection identifier, whether
    /// they have been aggregated or not.
    async fn count_client_reports<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &AggregatorTask,
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
        task: &AggregatorTask,
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
}

#[async_trait]
impl CollectableQueryType for TimeInterval {
    type Iter = TimeIntervalBatchIdentifierIter;

    async fn collection_identifier_for_query<C: Clock>(
        _: &Transaction<'_, C>,
        _: &AggregatorTask,
        query: &Query<Self>,
    ) -> Result<Option<Self::BatchIdentifier>, datastore::Error> {
        Ok(Some(*query.batch_interval()))
    }

    fn batch_identifiers_for_collection_identifier(
        task: &AggregatorTask,
        batch_interval: &Self::BatchIdentifier,
    ) -> Self::Iter {
        TimeIntervalBatchIdentifierIter::new(task, batch_interval)
    }

    fn validate_collection_identifier(
        task: &AggregatorTask,
        collection_identifier: &Self::BatchIdentifier,
    ) -> bool {
        // https://www.ietf.org/archive/id/draft-ietf-ppm-dap-02.html#section-4.5.6.1.1

        // Batch interval should be greater than task's time precision
        collection_identifier.duration().as_seconds() >= task.time_precision().as_seconds()
                // Batch interval start must be a multiple of time precision
                && collection_identifier.start().as_seconds_since_epoch() % task.time_precision().as_seconds() == 0
                // Batch interval duration must be a multiple of time precision
                && collection_identifier.duration().as_seconds() % task.time_precision().as_seconds() == 0
    }

    async fn count_client_reports<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &AggregatorTask,
        batch_interval: &Self::BatchIdentifier,
    ) -> Result<u64, datastore::Error> {
        tx.count_client_reports_for_interval(task.id(), batch_interval)
            .await
    }
}

// This type only exists because the CollectableQueryType trait requires specifying the type of the
// iterator explicitly (i.e. it cannot be inferred or replaced with an `impl Trait` expression), and
// the type of the iterator created via method chaining does not have a type which is expressible.
pub struct TimeIntervalBatchIdentifierIter {
    step: u64,

    total_step_count: u64,
    start: Time,
    time_precision: Duration,
}

impl TimeIntervalBatchIdentifierIter {
    fn new(task: &AggregatorTask, batch_interval: &Interval) -> Self {
        // Sanity check that the given interval is of an appropriate length. We use an assert as
        // this is expected to be checked before this method is used.
        assert_eq!(
            batch_interval.duration().as_seconds() % task.time_precision().as_seconds(),
            0
        );
        let total_step_count =
            batch_interval.duration().as_seconds() / task.time_precision().as_seconds();

        Self {
            step: 0,
            total_step_count,
            start: *batch_interval.start(),
            time_precision: *task.time_precision(),
        }
    }
}

impl Iterator for TimeIntervalBatchIdentifierIter {
    type Item = Interval;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step == self.total_step_count {
            return None;
        }
        // Unwrap safety: errors can only occur if the times being unwrapped cannot be represented
        // as a Time. The relevant times can always be represented since they are internal to the
        // batch interval used to create the iterator.
        let interval = Interval::new(
            self.start
                .add(&Duration::from_seconds(
                    self.step * self.time_precision.as_seconds(),
                ))
                .unwrap(),
            self.time_precision,
        )
        .unwrap();
        self.step += 1;
        Some(interval)
    }
}

#[async_trait]
impl CollectableQueryType for FixedSize {
    type Iter = iter::Once<Self::BatchIdentifier>;

    async fn collection_identifier_for_query<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &AggregatorTask,
        query: &Query<Self>,
    ) -> Result<Option<Self::BatchIdentifier>, datastore::Error> {
        match query.fixed_size_query() {
            FixedSizeQuery::ByBatchId { batch_id } => Ok(Some(*batch_id)),
            FixedSizeQuery::CurrentBatch => {
                tx.acquire_filled_outstanding_batch(task.id(), task.min_batch_size())
                    .await
            }
        }
    }

    fn batch_identifiers_for_collection_identifier(
        _: &AggregatorTask,
        batch_id: &Self::BatchIdentifier,
    ) -> Self::Iter {
        iter::once(*batch_id)
    }

    fn validate_collection_identifier(_: &AggregatorTask, _: &Self::BatchIdentifier) -> bool {
        true
    }

    async fn count_client_reports<C: Clock>(
        tx: &Transaction<'_, C>,
        task: &AggregatorTask,
        batch_id: &Self::BatchIdentifier,
    ) -> Result<u64, datastore::Error> {
        tx.count_client_reports_for_batch_id(task.id(), batch_id)
            .await
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        query_type::CollectableQueryType,
        task::{test_util::TaskBuilder, QueryType},
    };
    use janus_core::vdaf::VdafInstance;
    use janus_messages::{query_type::TimeInterval, Duration, Interval, Time};

    #[test]
    fn validate_collect_identifier() {
        let time_precision_secs = 3600;
        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Fake)
            .with_time_precision(Duration::from_seconds(time_precision_secs))
            .build()
            .leader_view()
            .unwrap();

        struct TestCase {
            name: &'static str,
            input: Interval,
            expected: bool,
        }

        for test_case in Vec::from([
            TestCase {
                name: "same duration as minimum",
                input: Interval::new(
                    Time::from_seconds_since_epoch(time_precision_secs),
                    Duration::from_seconds(time_precision_secs),
                )
                .unwrap(),
                expected: true,
            },
            TestCase {
                name: "interval too short",
                input: Interval::new(
                    Time::from_seconds_since_epoch(time_precision_secs),
                    Duration::from_seconds(time_precision_secs - 1),
                )
                .unwrap(),
                expected: false,
            },
            TestCase {
                name: "interval larger than minimum",
                input: Interval::new(
                    Time::from_seconds_since_epoch(time_precision_secs),
                    Duration::from_seconds(time_precision_secs * 2),
                )
                .unwrap(),
                expected: true,
            },
            TestCase {
                name: "interval duration not aligned with minimum",
                input: Interval::new(
                    Time::from_seconds_since_epoch(time_precision_secs),
                    Duration::from_seconds(time_precision_secs + 1800),
                )
                .unwrap(),
                expected: false,
            },
            TestCase {
                name: "interval start not aligned with minimum",
                input: Interval::new(
                    Time::from_seconds_since_epoch(1800),
                    Duration::from_seconds(time_precision_secs),
                )
                .unwrap(),
                expected: false,
            },
        ]) {
            assert_eq!(
                test_case.expected,
                TimeInterval::validate_collection_identifier(&task, &test_case.input),
                "test case: {}",
                test_case.name
            );
        }
    }
}
