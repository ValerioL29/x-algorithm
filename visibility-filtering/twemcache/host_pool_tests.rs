use super::*;
use crate::twemcache::accrual::FAILURE_THRESHOLD;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

fn key(s: &str) -> Key {
    Key::new(s.as_bytes().to_vec()).unwrap()
}

async fn conn_at_depth(depth: usize) -> Arc<PipelinedConnection> {
    let (client, server) = tokio::io::duplex(256 * 1024);
    tokio::spawn(async move {
        let mut server = server;
        let mut buf = [0u8; 4096];
        loop {
            match server.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    });
    let conn = Arc::new(PipelinedConnection::from_stream(
        Box::new(client),
        Duration::from_secs(60),
        Metrics::disabled(),
    ));
    for _ in 0..depth {
        let c = Arc::clone(&conn);
        tokio::spawn(async move {
            let _ = c.get(&key("k")).await;
        });
    }
    while conn.in_flight() < depth {
        tokio::task::yield_now().await;
    }
    conn
}

fn pool_from_conns(conns: Vec<Arc<PipelinedConnection>>, depth_cap: usize) -> HostPool {
    let slots = conns
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            Mutex::new(ConnSlot {
                conn: Some(c),
                health: ConnHealth::new(seed_for(i)),
            })
        })
        .collect();
    HostPool {
        factory: Arc::new(FakeFactory::failing()),
        slots,
        open_lock: TokioMutex::new(()),
        depth_cap,
        probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
        metrics: Metrics::disabled(),
        clock: Clock::new(),
    }
}

fn trip_slot(pool: &HostPool, idx: usize, now: Instant) {
    let mut g = pool.slots[idx].lock().unwrap();
    for _ in 0..FAILURE_THRESHOLD {
        g.health.note(Outcome::Failure, now);
    }
    assert!(!g.health.is_live());
}

async fn responsive_conn() -> Arc<PipelinedConnection> {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let (sr, mut sw) = tokio::io::split(server);
        let mut sr = BufReader::new(sr);
        let mut line = String::new();
        loop {
            line.clear();
            match sr.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let _ = sw.write_all(b"END\r\n").await;
                }
            }
        }
    });
    Arc::new(PipelinedConnection::from_stream(
        Box::new(client),
        Duration::from_secs(60),
        Metrics::disabled(),
    ))
}

async fn idle_close_once_conn() -> Arc<PipelinedConnection> {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let (sr, _sw) = tokio::io::split(server);
        let mut sr = BufReader::new(sr);
        let mut line = String::new();
        let _ = sr.read_line(&mut line).await;
    });
    Arc::new(PipelinedConnection::from_stream(
        Box::new(client),
        Duration::from_secs(60),
        Metrics::disabled(),
    ))
}

async fn silent_conn(request_timeout: Duration) -> Arc<PipelinedConnection> {
    let (client, server) = tokio::io::duplex(256 * 1024);
    tokio::spawn(async move {
        let mut server = server;
        let mut buf = [0u8; 4096];
        loop {
            match server.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    });
    Arc::new(PipelinedConnection::from_stream(
        Box::new(client),
        request_timeout,
        Metrics::disabled(),
    ))
}

struct FakeFactory {
    fail_forever: bool,
    fails_remaining: AtomicUsize,
    responsive: bool,
    always_closing: bool,
    open_count: AtomicUsize,
}

impl FakeFactory {
    fn failing() -> Self {
        Self {
            fail_forever: true,
            fails_remaining: AtomicUsize::new(0),
            responsive: false,
            always_closing: false,
            open_count: AtomicUsize::new(0),
        }
    }
    fn working() -> Self {
        Self {
            fail_forever: false,
            fails_remaining: AtomicUsize::new(0),
            responsive: false,
            always_closing: false,
            open_count: AtomicUsize::new(0),
        }
    }
    fn fail_then_succeed(n: usize) -> Self {
        Self {
            fail_forever: false,
            fails_remaining: AtomicUsize::new(n),
            responsive: true,
            always_closing: false,
            open_count: AtomicUsize::new(0),
        }
    }
    fn always_closing() -> Self {
        Self {
            fail_forever: false,
            fails_remaining: AtomicUsize::new(0),
            responsive: false,
            always_closing: true,
            open_count: AtomicUsize::new(0),
        }
    }
}

#[tonic::async_trait]
impl ConnFactory for FakeFactory {
    async fn open(&self) -> Result<Arc<PipelinedConnection>> {
        self.open_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_forever {
            return Err(TwemcacheError::Unavailable);
        }
        if self.fails_remaining.load(Ordering::SeqCst) > 0 {
            self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
            return Err(TwemcacheError::Unavailable);
        }
        if self.always_closing {
            Ok(idle_close_once_conn().await)
        } else if self.responsive {
            Ok(responsive_conn().await)
        } else {
            Ok(conn_at_depth(0).await)
        }
    }
}

fn live_conn(sel: Selected) -> Arc<PipelinedConnection> {
    match sel {
        Selected::Live(c, _) => c,
        Selected::Probe(_, _) => panic!("expected Live, got Probe"),
    }
}

#[tokio::test]
async fn least_loaded_picks_shallower() {
    let shallow = conn_at_depth(2).await;
    let deep = conn_at_depth(5).await;
    let pool = pool_from_conns(vec![Arc::clone(&shallow), Arc::clone(&deep)], 100);
    let chosen = live_conn(pool.select(pool.clock.now()).await.unwrap());
    assert!(Arc::ptr_eq(&chosen, &shallow));
}

#[tokio::test]
async fn capped_connection_routes_to_other() {
    let capped = conn_at_depth(100).await;
    let other = conn_at_depth(30).await;
    let pool = pool_from_conns(vec![Arc::clone(&capped), Arc::clone(&other)], 100);
    let chosen = live_conn(pool.select(pool.clock.now()).await.unwrap());
    assert!(Arc::ptr_eq(&chosen, &other));
}

#[tokio::test]
async fn both_at_cap_is_backpressure() {
    let a = conn_at_depth(100).await;
    let b = conn_at_depth(100).await;
    let pool = pool_from_conns(vec![a, b], 100);
    assert!(matches!(
        pool.select(pool.clock.now()).await,
        Err(TwemcacheError::Backpressure)
    ));
}

#[tokio::test]
async fn no_live_connection_is_unavailable() {
    let pool = HostPool::with_factory(
        Arc::new(FakeFactory::failing()),
        2,
        100,
        Clock::new(),
        Metrics::disabled(),
    );
    assert!(matches!(
        pool.select(pool.clock.now()).await,
        Err(TwemcacheError::Unavailable)
    ));
}

#[tokio::test]
async fn first_get_opens_lazily() {
    let factory = Arc::new(FakeFactory::working());
    let pool = HostPool::with_factory(
        Arc::clone(&factory) as Arc<dyn ConnFactory>,
        2,
        100,
        Clock::new(),
        Metrics::disabled(),
    );
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 0);
    let chosen = live_conn(pool.select(pool.clock.now()).await.unwrap());
    assert!(!chosen.is_dead());
    assert!(factory.open_count.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn tripped_connection_skipped_then_probe_eligible() {
    let (clock, mock) = Clock::mock();
    let pool = HostPool::with_factory(
        Arc::new(FakeFactory::working()),
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );
    trip_slot(&pool, 0, clock.now());

    assert!(matches!(
        pool.select(clock.now()).await,
        Err(TwemcacheError::Unavailable)
    ));

    mock.increment(Duration::from_secs(11));
    assert!(matches!(
        pool.select(clock.now()).await,
        Ok(Selected::Probe(_, 0))
    ));
}

#[tokio::test]
async fn all_connections_tripped_returns_unavailable() {
    let (clock, _m) = Clock::mock();
    let pool = HostPool::with_factory(
        Arc::new(FakeFactory::working()),
        2,
        100,
        clock.clone(),
        Metrics::disabled(),
    );
    trip_slot(&pool, 0, clock.now());
    trip_slot(&pool, 1, clock.now());
    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
}

#[tokio::test]
async fn probe_success_promotes_to_live() {
    let (clock, mock) = Clock::mock();
    let pool = HostPool::with_factory(
        Arc::new(FakeFactory::working()),
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );
    trip_slot(&pool, 0, clock.now());
    mock.increment(Duration::from_secs(11));

    assert!(matches!(
        pool.select(clock.now()).await,
        Ok(Selected::Probe(_, 0))
    ));
    pool.record(0, &Ok(None), true, clock.now());
    assert!(pool.slots[0].lock().unwrap().health.is_live());
}

#[tokio::test]
async fn saturated_live_does_not_block_probe() {
    let (clock, mock) = Clock::mock();
    let capped = conn_at_depth(100).await;
    let factory = Arc::new(FakeFactory::fail_then_succeed(0));
    let slots = vec![
        Mutex::new(ConnSlot {
            conn: Some(Arc::clone(&capped)),
            health: ConnHealth::new(seed_for(0)),
        }),
        Mutex::new(ConnSlot {
            conn: None,
            health: ConnHealth::new(seed_for(1)),
        }),
    ]
    .into_iter()
    .collect();
    let pool = HostPool {
        factory: Arc::clone(&factory) as Arc<dyn ConnFactory>,
        slots,
        open_lock: TokioMutex::new(()),
        depth_cap: 100,
        probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
        metrics: Metrics::disabled(),
        clock: clock.clone(),
    };

    trip_slot(&pool, 1, clock.now());
    mock.increment(Duration::from_secs(11));

    let before = factory.open_count.load(Ordering::SeqCst);
    assert!(matches!(
        pool.select(clock.now()).await,
        Ok(Selected::Probe(_, 1))
    ));
    assert!(factory.open_count.load(Ordering::SeqCst) > before);
}

#[tokio::test]
async fn cancelled_probe_slot_is_reclaimed_not_stuck() {
    let (clock, mock) = Clock::mock();
    let factory = Arc::new(FakeFactory::working());
    let pool = HostPool::with_factory(
        Arc::clone(&factory) as Arc<dyn ConnFactory>,
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );
    trip_slot(&pool, 0, clock.now());
    mock.increment(Duration::from_secs(11));

    assert!(matches!(
        pool.select(clock.now()).await,
        Ok(Selected::Probe(_, 0))
    ));
    let after_first = factory.open_count.load(Ordering::SeqCst);

    mock.increment(DEFAULT_PROBE_STALE_AFTER);
    assert!(matches!(
        pool.select(clock.now()).await,
        Err(TwemcacheError::Unavailable)
    ));
    assert_eq!(factory.open_count.load(Ordering::SeqCst), after_first);

    mock.increment(Duration::from_millis(1));
    assert!(matches!(
        pool.select(clock.now()).await,
        Ok(Selected::Probe(_, 0))
    ));
    assert!(factory.open_count.load(Ordering::SeqCst) > after_first);
}

#[tokio::test]
async fn connect_failure_trips_and_halts_reconnect_storm() {
    let (clock, _m) = Clock::mock();
    let factory = Arc::new(FakeFactory::failing());
    let pool = HostPool::with_factory(
        Arc::clone(&factory) as Arc<dyn ConnFactory>,
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );

    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
    let after_first = factory.open_count.load(Ordering::SeqCst);
    assert_eq!(after_first, 1);

    for _ in 0..5 {
        assert!(matches!(
            pool.get(&key("k")).await,
            Err(TwemcacheError::Unavailable)
        ));
    }
    assert_eq!(factory.open_count.load(Ordering::SeqCst), after_first);
}

#[tokio::test]
async fn probe_reconnect_is_throttled_and_backoff_grows() {
    let (clock, mock) = Clock::mock();
    let factory = Arc::new(FakeFactory::failing());
    let pool = HostPool::with_factory(
        Arc::clone(&factory) as Arc<dyn ConnFactory>,
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );

    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 1);
    let dead0 = pool.slots[0]
        .lock()
        .unwrap()
        .health
        .dead_remaining(clock.now())
        .unwrap();

    mock.increment(dead0);
    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 2);
    let dead1 = pool.slots[0]
        .lock()
        .unwrap()
        .health
        .dead_remaining(clock.now())
        .unwrap();

    assert!(dead1 > dead0);
}

#[tokio::test]
async fn host_recovers_via_probe_after_connect_failures() {
    let (clock, mock) = Clock::mock();
    let factory = Arc::new(FakeFactory::fail_then_succeed(2));
    let pool = HostPool::with_factory(
        Arc::clone(&factory) as Arc<dyn ConnFactory>,
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );

    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
    let dead0 = pool.slots[0]
        .lock()
        .unwrap()
        .health
        .dead_remaining(clock.now())
        .unwrap();
    mock.increment(dead0);

    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
    let dead1 = pool.slots[0]
        .lock()
        .unwrap()
        .health
        .dead_remaining(clock.now())
        .unwrap();
    mock.increment(dead1);

    assert_eq!(pool.get(&key("k")).await.unwrap(), None);
    assert!(pool.slots[0].lock().unwrap().health.is_live());
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn idle_close_retries_once_without_tripping() {
    let closing = idle_close_once_conn().await;
    let factory = Arc::new(FakeFactory::fail_then_succeed(0));
    let slots = vec![Mutex::new(ConnSlot {
        conn: Some(closing),
        health: ConnHealth::new(seed_for(0)),
    })];
    let pool = HostPool {
        factory: Arc::clone(&factory) as Arc<dyn ConnFactory>,
        slots,
        open_lock: TokioMutex::new(()),
        depth_cap: 100,
        probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
        metrics: Metrics::disabled(),
        clock: Clock::new(),
    };

    assert_eq!(factory.open_count.load(Ordering::SeqCst), 0);
    assert_eq!(pool.get(&key("k")).await.unwrap(), None);
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 1);
    assert!(pool.slots[0].lock().unwrap().health.is_live());
}

#[tokio::test]
async fn always_closing_accrues_once_per_get() {
    let closing = idle_close_once_conn().await;
    let factory = Arc::new(FakeFactory::always_closing());
    let slots = vec![Mutex::new(ConnSlot {
        conn: Some(closing),
        health: ConnHealth::new(seed_for(0)),
    })];
    let pool = HostPool {
        factory: Arc::clone(&factory) as Arc<dyn ConnFactory>,
        slots,
        open_lock: TokioMutex::new(()),
        depth_cap: 100,
        probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
        metrics: Metrics::disabled(),
        clock: Clock::new(),
    };

    let err = pool.get(&key("k")).await.unwrap_err();
    assert!(
        is_dead_socket_err(&err),
        "expected dead-socket err, got {err:?}"
    );
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 1);
    assert!(pool.slots[0].lock().unwrap().health.is_live());

    for _ in 0..(FAILURE_THRESHOLD - 2) {
        let err = pool.get(&key("k")).await.unwrap_err();
        assert!(is_dead_socket_err(&err), "got {err:?}");
        assert!(pool.slots[0].lock().unwrap().health.is_live());
    }
    let err = pool.get(&key("k")).await.unwrap_err();
    assert!(is_dead_socket_err(&err), "got {err:?}");
    assert!(!pool.slots[0].lock().unwrap().health.is_live());
}

#[tokio::test]
async fn probe_does_not_idle_close_retry() {
    let (clock, mock) = Clock::mock();
    let factory = Arc::new(FakeFactory::always_closing());
    let pool = HostPool::with_factory(
        Arc::clone(&factory) as Arc<dyn ConnFactory>,
        1,
        100,
        clock.clone(),
        Metrics::disabled(),
    );
    trip_slot(&pool, 0, clock.now());
    mock.increment(Duration::from_secs(11));

    let err = pool.get(&key("k")).await.unwrap_err();
    assert!(
        is_dead_socket_err(&err),
        "expected dead-socket err, got {err:?}"
    );
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 1);
    assert!(!pool.slots[0].lock().unwrap().health.is_live());
}

#[tokio::test]
async fn timeout_does_not_idle_close_retry() {
    let silent = silent_conn(Duration::from_millis(20)).await;
    let factory = Arc::new(FakeFactory::failing());
    let slots = vec![Mutex::new(ConnSlot {
        conn: Some(silent),
        health: ConnHealth::new(seed_for(0)),
    })];
    let pool = HostPool {
        factory: Arc::clone(&factory) as Arc<dyn ConnFactory>,
        slots,
        open_lock: TokioMutex::new(()),
        depth_cap: 100,
        probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
        metrics: Metrics::disabled(),
        clock: Clock::new(),
    };

    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Timeout)
    ));
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 0);
    assert!(pool.slots[0].lock().unwrap().health.is_live());
}

#[tokio::test]
async fn idle_close_reconnect_failure_trips() {
    let closing = idle_close_once_conn().await;
    let factory = Arc::new(FakeFactory::failing());
    let slots = vec![Mutex::new(ConnSlot {
        conn: Some(closing),
        health: ConnHealth::new(seed_for(0)),
    })];
    let pool = HostPool {
        factory: Arc::clone(&factory) as Arc<dyn ConnFactory>,
        slots,
        open_lock: TokioMutex::new(()),
        depth_cap: 100,
        probe_stale_after: DEFAULT_PROBE_STALE_AFTER,
        metrics: Metrics::disabled(),
        clock: Clock::new(),
    };

    assert!(matches!(
        pool.get(&key("k")).await,
        Err(TwemcacheError::Unavailable)
    ));
    assert_eq!(factory.open_count.load(Ordering::SeqCst), 1);
    assert!(!pool.slots[0].lock().unwrap().health.is_live());
}
