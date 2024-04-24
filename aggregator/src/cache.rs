//! Various in-memory caches that can be used by an aggregator.

use crate::aggregator::{report_writer::ReportWriteBatcher, Error, TaskAggregator};
use janus_aggregator_core::{
    datastore::{models::HpkeKeyState, Datastore},
    taskprov::PeerAggregator,
};
use janus_core::{hpke::HpkeKeypair, time::Clock};
use janus_messages::{HpkeConfig, HpkeConfigId, Role, TaskId};
use moka::{
    future::{Cache, CacheBuilder},
    ops::compute::Op,
    Entry,
};
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, Instant},
};
use tokio::{spawn, task::JoinHandle, time::sleep};
use tracing::{debug, error};
use url::Url;

type HpkeConfigs = Arc<Vec<HpkeConfig>>;
type HpkeKeypairs = HashMap<HpkeConfigId, Arc<HpkeKeypair>>;

#[derive(Debug)]
pub struct GlobalHpkeKeypairCache {
    // We use a std::sync::Mutex in this cache because we won't hold locks across
    // `.await` boundaries. StdMutex is lighter weight than `tokio::sync::Mutex`.
    /// Cache of HPKE configs for advertisement.
    configs: Arc<StdMutex<HpkeConfigs>>,

    /// Cache of HPKE keypairs for report decryption.
    keypairs: Arc<StdMutex<HpkeKeypairs>>,

    /// Handle for task responsible for periodically refreshing the cache.
    refresh_handle: JoinHandle<()>,
}

impl GlobalHpkeKeypairCache {
    pub const DEFAULT_REFRESH_INTERVAL: Duration =
        Duration::from_secs(60 * 30 /* 30 minutes */);

    pub async fn new<C: Clock>(
        datastore: Arc<Datastore<C>>,
        refresh_interval: Duration,
    ) -> Result<Self, Error> {
        let keypairs = Arc::new(StdMutex::new(HashMap::new()));
        let configs = Arc::new(StdMutex::new(Arc::new(Vec::new())));

        // Initial cache load.
        Self::refresh_inner(&datastore, &configs, &keypairs).await?;

        // Start refresh task.
        let refresh_configs = configs.clone();
        let refresh_keypairs = keypairs.clone();
        let refresh_datastore = datastore.clone();
        let refresh_handle = spawn(async move {
            loop {
                sleep(refresh_interval).await;

                let now = Instant::now();
                let result =
                    Self::refresh_inner(&refresh_datastore, &refresh_configs, &refresh_keypairs)
                        .await;
                let elapsed = now.elapsed();

                match result {
                    Ok(_) => debug!(?elapsed, "successfully refreshed HPKE keypair cache"),
                    Err(err) => error!(?err, ?elapsed, "failed to refresh HPKE keypair cache"),
                }
            }
        });

        Ok(Self {
            configs,
            keypairs,
            refresh_handle,
        })
    }

    async fn refresh_inner<C: Clock>(
        datastore: &Datastore<C>,
        configs: &StdMutex<HpkeConfigs>,
        keypairs: &StdMutex<HpkeKeypairs>,
    ) -> Result<(), Error> {
        let global_keypairs = datastore
            .run_tx("refresh_global_hpke_keypairs_cache", |tx| {
                Box::pin(async move { tx.get_global_hpke_keypairs().await })
            })
            .await?;

        let new_configs = Arc::new(
            global_keypairs
                .iter()
                .filter_map(|keypair| match keypair.state() {
                    HpkeKeyState::Active => Some(keypair.hpke_keypair().config().clone()),
                    _ => None,
                })
                .collect(),
        );

        let new_keypairs = global_keypairs
            .iter()
            .map(|keypair| {
                let keypair = keypair.hpke_keypair().clone();
                (*keypair.config().id(), Arc::new(keypair))
            })
            .collect();

        {
            let mut configs = configs.lock().unwrap();
            *configs = new_configs;
        }
        {
            let mut keypairs = keypairs.lock().unwrap();
            *keypairs = new_keypairs;
        }
        Ok(())
    }

    #[cfg(feature = "test-util")]
    pub async fn refresh<C: Clock>(&self, datastore: &Datastore<C>) -> Result<(), Error> {
        Self::refresh_inner(datastore, &self.configs, &self.keypairs).await
    }

    /// Retrieve active configs for config advertisement. This only returns configs
    /// for keypairs that are in the `[HpkeKeyState::Active]` state.
    pub fn configs(&self) -> HpkeConfigs {
        let configs = self.configs.lock().unwrap();
        configs.clone()
    }

    /// Retrieve a keypair by ID for report decryption. This retrieves keypairs that
    /// are in any state.
    pub fn keypair(&self, id: &HpkeConfigId) -> Option<Arc<HpkeKeypair>> {
        let keypairs = self.keypairs.lock().unwrap();
        keypairs.get(id).cloned()
    }

    /// Create a `GlobalHpkeKeypairCacheView` with access to the same caches of configs and
    /// keypairs.
    pub fn view(&self) -> GlobalHpkeKeypairCacheView {
        GlobalHpkeKeypairCacheView {
            configs: Arc::clone(&self.configs),
            keypairs: Arc::clone(&self.keypairs),
        }
    }
}

impl Drop for GlobalHpkeKeypairCache {
    fn drop(&mut self) {
        self.refresh_handle.abort()
    }
}

#[derive(Debug)]
pub struct GlobalHpkeKeypairCacheView {
    // We use a std::sync::Mutex in this cache because we won't hold locks across
    // `.await` boundaries. StdMutex is lighter weight than `tokio::sync::Mutex`.
    /// Cache of HPKE configs for advertisement.
    configs: Arc<StdMutex<HpkeConfigs>>,

    /// Cache of HPKE keypairs for report decryption.
    keypairs: Arc<StdMutex<HpkeKeypairs>>,
}

impl GlobalHpkeKeypairCacheView {
    /// Retrieve active configs for config advertisement. This only returns configs
    /// for keypairs that are in the `[HpkeKeyState::Active]` state.
    pub fn configs(&self) -> HpkeConfigs {
        let configs = self.configs.lock().unwrap();
        configs.clone()
    }

    /// Retrieve a keypair by ID for report decryption. This retrieves keypairs that
    /// are in any state.
    pub fn keypair(&self, id: &HpkeConfigId) -> Option<Arc<HpkeKeypair>> {
        let keypairs = self.keypairs.lock().unwrap();
        keypairs.get(id).cloned()
    }
}

/// Caches taskprov [`PeerAggregator`]'s. This cache is never invalidated, so the process needs to
/// be restarted if there are any changes to peer aggregators.
#[derive(Debug)]
pub struct PeerAggregatorCache {
    peers: Vec<PeerAggregator>,
}

impl PeerAggregatorCache {
    pub async fn new<C: Clock>(datastore: &Datastore<C>) -> Result<Self, Error> {
        Ok(Self {
            peers: datastore
                .run_tx("refresh_peer_aggregators_cache", |tx| {
                    Box::pin(async move { tx.get_taskprov_peer_aggregators().await })
                })
                .await?
                .into_iter()
                .collect(),
        })
    }

    pub fn get(&self, endpoint: &Url, role: &Role) -> Option<&PeerAggregator> {
        // The peer aggregator table is unlikely to be more than a few entries long (1-2 entries),
        // so a linear search should be fine.
        self.peers
            .iter()
            .find(|peer| peer.endpoint() == endpoint && peer.role() == role)
    }
}

#[derive(Debug)]
pub struct TaskAggregatorCache<C: Clock> {
    datastore: Arc<Datastore<C>>,
    report_writer: Arc<ReportWriteBatcher<C>>,
    cache: Cache<TaskId, TaskAggregatorRef<C>>,
    cache_none: bool,
}

/// An Arc reference to a TaskAggregator. None indicates that there is no such task aggregator in
/// the database.
type TaskAggregatorRef<C> = Option<Arc<TaskAggregator<C>>>;

pub const TASK_AGGREGATOR_CACHE_DEFAULT_TTL: Duration = Duration::from_secs(600);
pub const TASK_AGGREGATOR_CACHE_DEFAULT_CAPACITY: u64 = 10_000;

impl<C: Clock> TaskAggregatorCache<C> {
    pub fn new(
        datastore: Arc<Datastore<C>>,
        report_writer: ReportWriteBatcher<C>,
        cache_none: bool,
        capacity: u64,
        ttl: Duration,
    ) -> Self {
        Self {
            datastore,
            report_writer: Arc::new(report_writer),
            cache: CacheBuilder::new(capacity).time_to_live(ttl).build(),
            cache_none,
        }
    }

    pub async fn get(&self, task_id: &TaskId) -> Result<TaskAggregatorRef<C>, Error> {
        Ok(self
            .cache
            .entry(*task_id)
            .and_try_compute_with(|entry| async move {
                match entry {
                    Some(_) => Ok::<_, Error>(Op::Nop),
                    None => {
                        let task = self
                            .datastore
                            .run_tx("task_aggregator_get_task", |tx| {
                                let task_id = *task_id;
                                Box::pin(async move { tx.get_aggregator_task(&task_id).await })
                            })
                            .await?
                            .map(|task| TaskAggregator::new(task, Arc::clone(&self.report_writer)))
                            .transpose()?
                            .map(Arc::new);
                        match task {
                            Some(task) => Ok(Op::Put(Some(task))),
                            None => {
                                if self.cache_none {
                                    Ok(Op::Put(None))
                                } else {
                                    Ok(Op::Nop)
                                }
                            }
                        }
                    }
                }
            })
            .await?
            .into_entry()
            .map_or_else(|| None, Entry::into_value))
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use janus_aggregator_core::{
        datastore::test_util::ephemeral_datastore,
        task::{test_util::TaskBuilder, QueryType},
    };
    use janus_core::{
        test_util::{install_test_trace_subscriber, runtime::TestRuntime},
        time::MockClock,
        vdaf::VdafInstance,
    };
    use janus_messages::Time;
    use tokio::time::sleep;

    use crate::{aggregator::report_writer::ReportWriteBatcher, cache::TaskAggregatorCache};

    #[tokio::test]
    async fn task_aggregator_cache() {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let ttl = Duration::from_millis(500);
        let task_aggregators = TaskAggregatorCache::new(
            Arc::clone(&datastore),
            ReportWriteBatcher::new(
                Arc::clone(&datastore),
                TestRuntime::default(),
                100,                      // doesn't matter
                100,                      // doesn't matter
                Duration::from_secs(100), // doesn't matter
            ),
            false,
            10000,
            ttl,
        );

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .build()
            .leader_view()
            .unwrap();

        assert!(task_aggregators.get(task.id()).await.unwrap().is_none());
        // We shouldn't have cached that last call.
        assert_eq!(task_aggregators.cache.entry_count(), 0);

        // A wild task appears!
        datastore.put_aggregator_task(&task).await.unwrap();
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(task_aggregator.task.id(), task.id());

        // Modify the task.
        let new_expiration = Time::from_seconds_since_epoch(100);
        datastore
            .run_unnamed_tx(|tx| {
                let task_id = *task.id();
                Box::pin(async move {
                    tx.update_task_expiration(&task_id, Some(&new_expiration))
                        .await
                        .unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        // That change shouldn't be reflected yet because we've cached the previous task.
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(
            task_aggregator.task.task_expiration(),
            task.task_expiration()
        );

        // Unfortunately, because moka doesn't provide any facility for a fake clock, we have to resort
        // to sleeps to test TTL functionality.
        sleep(Duration::from_secs(1)).await;

        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(
            task_aggregator.task.task_expiration(),
            Some(&new_expiration)
        );
    }

    #[tokio::test]
    async fn task_aggregator_cache_none() {
        install_test_trace_subscriber();
        let clock = MockClock::default();
        let ephemeral_datastore = ephemeral_datastore().await;
        let datastore = Arc::new(ephemeral_datastore.datastore(clock.clone()).await);

        let ttl = Duration::from_millis(500);
        let task_aggregators = TaskAggregatorCache::new(
            Arc::clone(&datastore),
            ReportWriteBatcher::new(
                Arc::clone(&datastore),
                TestRuntime::default(),
                100,                      // doesn't matter
                100,                      // doesn't matter
                Duration::from_secs(100), // doesn't matter
            ),
            true,
            10000,
            ttl,
        );

        let task = TaskBuilder::new(QueryType::TimeInterval, VdafInstance::Prio3Count)
            .build()
            .leader_view()
            .unwrap();

        assert!(task_aggregators.get(task.id()).await.unwrap().is_none());

        // A wild task appears!
        datastore.put_aggregator_task(&task).await.unwrap();

        // We shouldn't see the new task yet.
        assert!(task_aggregators.get(task.id()).await.unwrap().is_none());

        // Unfortunately, because moka doesn't provide any facility for a fake clock, we have to resort
        // to sleeps to test TTL functionality.
        sleep(Duration::from_secs(1)).await;

        // Now we should see it.
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(task_aggregator.task.id(), task.id());

        // Modify the task.
        let new_expiration = Time::from_seconds_since_epoch(100);
        datastore
            .run_unnamed_tx(|tx| {
                let task_id = *task.id();
                Box::pin(async move {
                    tx.update_task_expiration(&task_id, Some(&new_expiration))
                        .await
                        .unwrap();
                    Ok(())
                })
            })
            .await
            .unwrap();

        // That change shouldn't be reflected yet because we've cached the previous run.
        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(
            task_aggregator.task.task_expiration(),
            task.task_expiration()
        );

        sleep(Duration::from_secs(1)).await;

        let task_aggregator = task_aggregators.get(task.id()).await.unwrap().unwrap();
        assert_eq!(
            task_aggregator.task.task_expiration(),
            Some(&new_expiration)
        );
    }
}
