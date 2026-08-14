use std::time::Duration;

use quanta::Instant;

pub(crate) const FAILURE_THRESHOLD: u32 = 30;
const BASE_MS: u64 = 50;
const CAP_MS: u64 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Success,
    Failure,
    Backpressure,
}

enum State {
    Live,
    Tripped { until: Instant },
    Probing { since: Instant },
}

pub(crate) struct ConnHealth {
    state: State,
    consecutive_failures: u32,
    trip_index: u32,
    seed: u64,
}

impl ConnHealth {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: State::Live,
            consecutive_failures: 0,
            trip_index: 0,
            seed,
        }
    }

    pub(crate) fn is_live(&self) -> bool {
        matches!(self.state, State::Live)
    }

    pub(crate) fn is_probe_eligible(&self, now: Instant, stale_after: Duration) -> bool {
        match self.state {
            State::Tripped { until } => now >= until,
            State::Probing { since } => now > since + stale_after,
            State::Live => false,
        }
    }

    pub(crate) fn begin_probe(&mut self, now: Instant) {
        debug_assert!(!matches!(self.state, State::Live));
        self.state = State::Probing { since: now };
    }

    pub(crate) fn note(&mut self, outcome: Outcome, now: Instant) -> Option<Duration> {
        match outcome {
            Outcome::Success => {
                if matches!(self.state, State::Live) {
                    self.consecutive_failures = 0;
                }
                None
            }
            Outcome::Failure => {
                if !matches!(self.state, State::Live) {
                    return None;
                }
                self.consecutive_failures += 1;
                if self.consecutive_failures >= FAILURE_THRESHOLD {
                    Some(self.trip(now))
                } else {
                    None
                }
            }
            Outcome::Backpressure => None,
        }
    }

    pub(crate) fn note_probe(&mut self, outcome: Outcome, now: Instant) -> Option<Duration> {
        match outcome {
            Outcome::Success => {
                self.state = State::Live;
                self.consecutive_failures = 0;
                self.trip_index = 0;
                None
            }
            Outcome::Failure => Some(self.trip(now)),
            Outcome::Backpressure => {
                self.state = State::Tripped { until: now };
                None
            }
        }
    }

    pub(crate) fn record_connect_failure(&mut self, now: Instant) -> Duration {
        self.trip(now)
    }

    #[cfg(test)]
    pub(crate) fn dead_remaining(&self, now: Instant) -> Option<Duration> {
        match self.state {
            State::Tripped { until } => Some(until.saturating_duration_since(now)),
            _ => None,
        }
    }

    fn trip(&mut self, now: Instant) -> Duration {
        let dead = self.dead_time();
        self.state = State::Tripped { until: now + dead };
        self.trip_index = self.trip_index.saturating_add(1);
        self.consecutive_failures = 0;
        dead
    }

    fn dead_time(&self) -> Duration {
        let factor = 1u64
            .checked_shl(self.trip_index.min(40))
            .unwrap_or(u64::MAX);
        let base_ms = BASE_MS.saturating_mul(factor).min(CAP_MS);
        let base = Duration::from_millis(base_ms);
        let half = base / 2;
        let half_nanos = half.as_nanos() as u64;
        let jitter = if half_nanos == 0 {
            0
        } else {
            self.jitter() % half_nanos
        };
        half + Duration::from_nanos(jitter)
    }

    fn jitter(&self) -> u64 {
        let h = self.seed.wrapping_mul(0x9E3779B97F4A7C15)
            ^ u64::from(self.trip_index)
                .wrapping_mul(0xD1B54A32D192ED03)
                .wrapping_add(0x2545F4914F6CDD1D);
        h ^ (h >> 29)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quanta::Clock;

    const MS25: Duration = Duration::from_millis(25);
    const MS50: Duration = Duration::from_millis(50);
    const MS100: Duration = Duration::from_millis(100);

    fn fail_until_trip(h: &mut ConnHealth, now: Instant) -> Duration {
        loop {
            if let Some(d) = h.note(Outcome::Failure, now) {
                return d;
            }
        }
    }

    #[test]
    fn trips_after_exactly_30_failures() {
        let (clock, _m) = Clock::mock();
        let now = clock.now();
        let mut h = ConnHealth::new(1);
        for _ in 0..FAILURE_THRESHOLD - 1 {
            assert!(h.note(Outcome::Failure, now).is_none());
        }
        assert!(h.note(Outcome::Failure, now).is_some());
    }

    #[test]
    fn success_resets_counter() {
        let (clock, _m) = Clock::mock();
        let now = clock.now();
        let mut h = ConnHealth::new(1);
        for _ in 0..FAILURE_THRESHOLD - 1 {
            h.note(Outcome::Failure, now);
        }
        h.note(Outcome::Success, now);
        assert!(h.note(Outcome::Failure, now).is_none());
    }

    #[test]
    fn backpressure_neither_counts_nor_resets() {
        let (clock, _m) = Clock::mock();
        let now = clock.now();
        let mut h = ConnHealth::new(1);
        for _ in 0..FAILURE_THRESHOLD - 1 {
            h.note(Outcome::Failure, now);
        }
        for _ in 0..5 {
            assert!(h.note(Outcome::Backpressure, now).is_none());
        }
        assert!(h.note(Outcome::Failure, now).is_some());
    }

    #[test]
    fn probe_failure_uses_longer_backoff() {
        let (clock, mock) = Clock::mock();
        let mut h = ConnHealth::new(7);
        let dead0 = fail_until_trip(&mut h, clock.now());
        assert!((MS25..MS50).contains(&dead0));

        mock.increment(dead0);
        let now1 = clock.now();
        assert!(h.is_probe_eligible(now1, MS100));
        h.begin_probe(now1);
        let dead1 = h.note_probe(Outcome::Failure, now1).unwrap();

        assert!(dead1 > dead0);
        assert!((MS50..MS100).contains(&dead1));
    }

    #[test]
    fn probe_success_resets_backoff_to_base() {
        let (clock, mock) = Clock::mock();
        let mut h = ConnHealth::new(7);
        let dead0 = fail_until_trip(&mut h, clock.now());

        mock.increment(dead0);
        let now1 = clock.now();
        h.begin_probe(now1);
        assert!(h.note_probe(Outcome::Success, now1).is_none());
        assert!(h.is_live());

        let dead_after = fail_until_trip(&mut h, now1);
        assert!((MS25..MS50).contains(&dead_after));
    }

    #[test]
    fn backoff_is_bounded_at_10s() {
        let (clock, mock) = Clock::mock();
        let mut h = ConnHealth::new(3);
        let mut dead = fail_until_trip(&mut h, clock.now());
        for _ in 0..25 {
            mock.increment(dead);
            let now = clock.now();
            h.begin_probe(now);
            dead = h.note_probe(Outcome::Failure, now).unwrap();
            assert!(dead <= Duration::from_secs(10));
        }
        assert!(dead >= Duration::from_secs(5));
    }

    #[test]
    fn stale_probe_becomes_eligible_again() {
        let (clock, mock) = Clock::mock();
        let connect_timeout = Duration::from_millis(200);
        let request_timeout = Duration::from_millis(75);
        let worst_case_live = connect_timeout + request_timeout;
        let stale_after = worst_case_live * 2;

        let mut h = ConnHealth::new(5);
        let dead0 = fail_until_trip(&mut h, clock.now());
        mock.increment(dead0);

        let started = clock.now();
        h.begin_probe(started);
        assert!(!h.is_probe_eligible(started, stale_after));

        mock.increment(worst_case_live);
        assert!(!h.is_probe_eligible(clock.now(), stale_after));

        mock.increment(worst_case_live);
        assert!(!h.is_probe_eligible(clock.now(), stale_after));

        mock.increment(Duration::from_millis(1));
        assert!(h.is_probe_eligible(clock.now(), stale_after));
    }
}
