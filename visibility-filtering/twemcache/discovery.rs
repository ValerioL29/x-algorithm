use std::collections::HashSet;
use std::time::Duration;

use tokio::sync::watch;

use super::error::{Result, TwemcacheError};
use super::key::{Server, ServerSet};

const POLL_INTERVAL: Duration = Duration::from_secs(30);

pub(crate) trait Discovery: Send + Sync {
    fn subscribe(&self) -> watch::Receiver<ServerSet>;
}

pub struct WilyDiscovery {
    tx: watch::Sender<ServerSet>,
    _handle: tokio::task::JoinHandle<()>,
}

impl Drop for WilyDiscovery {
    fn drop(&mut self) {
        self._handle.abort();
    }
}

impl WilyDiscovery {
    pub async fn new(
        path: impl Into<String>,
        client_name: impl Into<String>,
        zone: impl Into<String>,
    ) -> Result<Self> {
        Self::with_interval(path, client_name, zone, POLL_INTERVAL).await
    }

    async fn with_interval(
        path: impl Into<String>,
        client_name: impl Into<String>,
        zone: impl Into<String>,
        poll_interval: Duration,
    ) -> Result<Self> {
        let path = path.into();
        let config = xai_wily::WilyConfig {
            zone: zone.into(),
            client_name: client_name.into(),
            allow_alternative_zones: false,
            ..xai_wily::WilyConfig::default()
        };
        let wily = xai_wily::WilyNs::new(config)
            .map_err(|e| TwemcacheError::Io(format!("WilyNs init failed: {e}")))?;

        let initial = Self::fetch(&wily, &path).await.unwrap_or_else(|e| {
            tracing::warn!(error = %e, "initial Wily fetch failed, starting empty");
            HashSet::new()
        });

        let (tx, _) = watch::channel(initial);
        let tx_poll = tx.clone();
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(poll_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                match Self::fetch(&wily, &path).await {
                    Ok(servers) => {
                        let _ = tx_poll.send(servers);
                    }
                    Err(e) => tracing::warn!(error = %e, "Wily poll failed"),
                }
            }
        });

        Ok(Self {
            tx,
            _handle: handle,
        })
    }

    async fn fetch(wily: &xai_wily::WilyNs, path: &str) -> Result<ServerSet> {
        let instances = wily
            .resolve(path, "")
            .await
            .map_err(|e| TwemcacheError::Io(format!("Wily resolve failed: {e}")))?;
        Ok(instances
            .into_iter()
            .map(|i| match i.shard_id {
                Some(id) => Server::with_ring_key(i.address, i.port, id),
                None => Server::new(i.address, i.port),
            })
            .collect())
    }
}

impl Discovery for WilyDiscovery {
    fn subscribe(&self) -> watch::Receiver<ServerSet> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
pub(crate) struct ManualDiscovery {
    tx: watch::Sender<ServerSet>,
}

#[cfg(test)]
impl ManualDiscovery {
    pub(crate) fn new(initial: ServerSet) -> Self {
        let (tx, _) = watch::channel(initial);
        Self { tx }
    }

    pub(crate) fn set(&self, servers: ServerSet) {
        let _ = self.tx.send(servers);
    }
}

#[cfg(test)]
impl Discovery for ManualDiscovery {
    fn subscribe(&self) -> watch::Receiver<ServerSet> {
        self.tx.subscribe()
    }
}
