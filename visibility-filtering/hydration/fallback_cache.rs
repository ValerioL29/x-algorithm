use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

use quick_cache::unsync::Cache;

use crate::hydration::batch::{Hydrated, HydrationBatch};
use crate::hydration::metrics::{record_fallback_cache_entries, record_fallback_cache_keys};

const CACHE_SHARDS: usize = 64;
const OCCUPANCY_SAMPLE_INTERVAL: u64 = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FallbackCacheMode {
    Disabled,
    Shadow,
    ServeStale,
}

impl FallbackCacheMode {
    fn enabled(self) -> bool {
        self != Self::Disabled
    }

    fn serves_stale(self) -> bool {
        self == Self::ServeStale
    }
}

#[derive(Clone)]
struct CacheEntry<V> {
    generation: u64,
    value: Option<V>,
}

type CacheShards<K, V> = Vec<Mutex<Cache<K, CacheEntry<V>>>>;

pub(crate) struct FallbackCache<K, V> {
    facet: &'static str,
    mode: FallbackCacheMode,
    shards: Option<CacheShards<K, V>>,
    next_generation: AtomicU64,
}

impl<K, V> FallbackCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub(crate) fn new(facet: &'static str, capacity: usize, mode: FallbackCacheMode) -> Self {
        let shards = mode.enabled().then(|| {
            let shard_count = CACHE_SHARDS.min(capacity.max(1));
            let shard_capacity = capacity.div_ceil(shard_count);
            (0..shard_count)
                .map(|_| Mutex::new(Cache::<K, CacheEntry<V>>::new(shard_capacity)))
                .collect()
        });
        Self {
            facet,
            mode,
            shards,
            next_generation: AtomicU64::new(0),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.mode.enabled()
    }

    pub(crate) fn begin_request(&self) -> u64 {
        self.next_generation.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn resolve_hydration_batch(
        &self,
        generation: u64,
        batch: HydrationBatch<K, V>,
    ) -> HydrationBatch<K, V> {
        if !self.mode.enabled() {
            return batch;
        }

        let mut fresh = 0;
        let mut stale = 0;
        let mut stale_not_found = 0;
        let mut shadow_hit = 0;
        let mut not_found = 0;
        let mut unavailable = 0;
        let resolved = batch
            .into_hydrated()
            .into_iter()
            .map(|(key, hydrated)| {
                let hydrated = match hydrated {
                    Hydrated::Found(value) => {
                        self.write_entry(generation, &key, Some(value.clone()));
                        fresh += 1;
                        Hydrated::Found(value)
                    }
                    Hydrated::NotFound => {
                        self.write_entry(generation, &key, None);
                        not_found += 1;
                        Hydrated::NotFound
                    }
                    Hydrated::Failed(error) => match self.cached_entry(&key) {
                        Some(CacheEntry {
                            value: Some(value), ..
                        }) if self.mode.serves_stale() => {
                            stale += 1;
                            Hydrated::Found(value)
                        }
                        Some(CacheEntry { value: None, .. }) if self.mode.serves_stale() => {
                            stale_not_found += 1;
                            Hydrated::NotFound
                        }
                        Some(_) => {
                            shadow_hit += 1;
                            Hydrated::Failed(error)
                        }
                        None => {
                            unavailable += 1;
                            Hydrated::Failed(error)
                        }
                    },
                };
                (key, hydrated)
            })
            .collect();

        record_fallback_cache_keys(
            self.facet,
            fresh,
            stale,
            stale_not_found,
            shadow_hit,
            not_found,
            unavailable,
        );
        if generation.is_multiple_of(OCCUPANCY_SAMPLE_INTERVAL) {
            record_fallback_cache_entries(self.facet, self.entry_count());
        }
        HydrationBatch::from_hydrated(resolved)
    }

    fn entry_count(&self) -> usize {
        self.shards()
            .iter()
            .map(|shard| {
                shard
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len()
            })
            .sum()
    }

    fn write_entry(&self, generation: u64, key: &K, value: Option<V>) {
        let shard = cache_shard(self.shards(), key);
        let mut cache = shard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache
            .get(key)
            .is_some_and(|entry| entry.generation > generation)
        {
            return;
        }
        cache.insert(key.clone(), CacheEntry { generation, value });
    }

    fn cached_entry(&self, key: &K) -> Option<CacheEntry<V>> {
        cache_shard(self.shards(), key)
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .cloned()
    }

    fn shards(&self) -> &[Mutex<Cache<K, CacheEntry<V>>>] {
        self.shards
            .as_deref()
            .expect("fallback cache shards unavailable while enabled")
    }
}

fn cache_shard<'a, K, V>(shards: &'a [Mutex<Cache<K, V>>], key: &K) -> &'a Mutex<Cache<K, V>>
where
    K: Eq + Hash,
    V: Clone,
{
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    &shards[hasher.finish() as usize % shards.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::HydrationError;

    fn cache(mode: FallbackCacheMode) -> FallbackCache<u64, String> {
        FallbackCache::new("test", 8, mode)
    }

    fn batch(
        entries: impl IntoIterator<Item = (u64, Hydrated<String>)>,
    ) -> HydrationBatch<u64, String> {
        HydrationBatch::from_hydrated(entries.into_iter().collect())
    }

    fn failed() -> Hydrated<String> {
        Hydrated::Failed(HydrationError::Timeout)
    }

    #[test]
    fn recovers_only_resident_failed_keys() {
        let cache = cache(FallbackCacheMode::ServeStale);
        cache.resolve_hydration_batch(
            cache.begin_request(),
            batch([(1, Hydrated::Found("cached".to_string()))]),
        );

        let resolved = cache.resolve_hydration_batch(
            cache.begin_request(),
            batch([
                (1, failed()),
                (2, failed()),
                (3, Hydrated::Found("fresh".to_string())),
            ]),
        );

        assert_eq!(resolved.get(&1), Some(&"cached".to_string()));
        assert!(matches!(resolved.hydrated(&2), Some(Hydrated::Failed(_))));
        assert_eq!(resolved.get(&3), Some(&"fresh".to_string()));
    }

    #[test]
    fn authoritative_not_found_invalidates_stale_value() {
        let cache = cache(FallbackCacheMode::ServeStale);
        cache.resolve_hydration_batch(
            cache.begin_request(),
            batch([(1, Hydrated::Found("cached".to_string()))]),
        );

        cache.resolve_hydration_batch(cache.begin_request(), batch([(1, Hydrated::NotFound)]));
        let failed = cache.resolve_hydration_batch(cache.begin_request(), batch([(1, failed())]));

        assert!(matches!(failed.hydrated(&1), Some(Hydrated::NotFound)));
    }

    #[test]
    fn shadow_mode_does_not_serve_stale() {
        let cache = cache(FallbackCacheMode::Shadow);
        cache.resolve_hydration_batch(
            cache.begin_request(),
            batch([(1, Hydrated::Found("cached".to_string()))]),
        );

        let failed = cache.resolve_hydration_batch(cache.begin_request(), batch([(1, failed())]));

        assert!(matches!(failed.hydrated(&1), Some(Hydrated::Failed(_))));
    }

    #[test]
    fn shadow_mode_does_not_serve_cached_not_found() {
        let cache = cache(FallbackCacheMode::Shadow);
        cache.resolve_hydration_batch(cache.begin_request(), batch([(1, Hydrated::NotFound)]));

        let failed = cache.resolve_hydration_batch(cache.begin_request(), batch([(1, failed())]));

        assert!(matches!(failed.hydrated(&1), Some(Hydrated::Failed(_))));
    }

    #[test]
    fn disabled_mode_does_not_allocate_cache_shards() {
        let cache = cache(FallbackCacheMode::Disabled);

        assert!(cache.shards.is_none());
    }

    #[test]
    fn late_older_value_does_not_resurrect_newer_not_found() {
        let cache = cache(FallbackCacheMode::ServeStale);
        let older = cache.begin_request();
        let newer = cache.begin_request();

        cache.resolve_hydration_batch(newer, batch([(1, Hydrated::NotFound)]));
        cache.resolve_hydration_batch(older, batch([(1, Hydrated::Found("old".to_string()))]));

        let failed = cache.resolve_hydration_batch(cache.begin_request(), batch([(1, failed())]));
        assert!(matches!(failed.hydrated(&1), Some(Hydrated::NotFound)));
    }

    #[test]
    fn late_older_not_found_does_not_suppress_newer_value() {
        let cache = cache(FallbackCacheMode::ServeStale);
        let older = cache.begin_request();
        let newer = cache.begin_request();

        cache.resolve_hydration_batch(newer, batch([(1, Hydrated::Found("new".to_string()))]));
        cache.resolve_hydration_batch(older, batch([(1, Hydrated::NotFound)]));

        let failed = cache.resolve_hydration_batch(cache.begin_request(), batch([(1, failed())]));
        assert_eq!(failed.get(&1), Some(&"new".to_string()));
    }
}
