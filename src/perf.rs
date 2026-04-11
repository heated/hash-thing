//! Lightweight per-operation timing tracker.
//!
//! Owned by `App`. Records `Duration` samples in fixed-capacity ring buffers
//! per named metric, exposes mean + p95, and renders a one-line summary for
//! periodic logging.
//!
//! ## Important caveats
//!
//! 1. **CPU-side timings only.** `wgpu` queue submissions and `Queue::write_*`
//!    calls return immediately while the GPU does work later, so `Instant`
//!    around them measures *CPU submit overhead*, not GPU execution time.
//!    Metrics whose work runs on the GPU are named `*_cpu` to make this
//!    explicit. Real GPU timing belongs in a future bead with timestamp queries.
//!
//! 2. **Mixed cadences in one summary.** `step` and `upload_cpu` only run on
//!    a generation tick (~ every 200ms). `render_cpu` runs every redraw, which
//!    in a winit Poll loop is much faster. Both share the 64-sample ring,
//!    so the *time window* covered by `render_cpu` is shorter than the
//!    others. Compare metrics within a category, not across.
//!
//! 3. **Don't call from sim/render/octree code.** Instrumentation lives at
//!    call sites in `main.rs`. Threading `&mut Perf` through `step_flat` or
//!    the render path would touch the determinism contract under h34.1 / h34.2.

use std::collections::HashMap;
use std::time::Duration;

const RING_CAPACITY: usize = 64;

/// Fixed-capacity ring buffer of `Duration`s. Oldest sample is evicted FIFO
/// once capacity is exceeded.
struct Ring {
    buf: [Duration; RING_CAPACITY],
    head: usize,
    len: usize,
}

impl Ring {
    fn new() -> Self {
        Self {
            buf: [Duration::ZERO; RING_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, d: Duration) {
        self.buf[self.head] = d;
        self.head = (self.head + 1) % RING_CAPACITY;
        if self.len < RING_CAPACITY {
            self.len += 1;
        }
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Iterate over the live samples in arbitrary order. Used by aggregation,
    /// not by callers that care about insertion order.
    fn samples(&self) -> &[Duration] {
        &self.buf[..self.len]
    }

    fn mean(&self) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let total: Duration = self.samples().iter().sum();
        total / self.len as u32
    }

    /// 95th percentile via sorted copy. Cheap because cap is 64.
    ///
    /// Index formula: `ceil(len * 0.95) - 1`. At `len == 64` this is index 60
    /// (the 61st sorted entry). At `len == 1` it is index 0.
    fn p95(&self) -> Duration {
        if self.len == 0 {
            return Duration::ZERO;
        }
        let mut sorted: Vec<Duration> = self.samples().to_vec();
        sorted.sort();
        let idx = ((self.len as f64) * 0.95).ceil() as usize - 1;
        sorted[idx]
    }
}

/// Per-operation timing tracker. Owned by `App` in `main.rs`.
pub struct Perf {
    rings: HashMap<&'static str, Ring>,
}

impl Perf {
    pub fn new() -> Self {
        Self {
            rings: HashMap::new(),
        }
    }

    /// Record a duration for a named metric.
    pub fn record(&mut self, name: &'static str, d: Duration) {
        self.rings.entry(name).or_insert_with(Ring::new).push(d);
    }

    /// Convenience: time a closure and record its duration.
    pub fn time<R>(&mut self, name: &'static str, f: impl FnOnce() -> R) -> R {
        let t = std::time::Instant::now();
        let r = f();
        self.record(name, t.elapsed());
        r
    }

    /// Drop all samples from all rings. Call on world reset so post-reset
    /// stats aren't poisoned by stale pre-reset samples.
    pub fn clear(&mut self) {
        for ring in self.rings.values_mut() {
            ring.clear();
        }
    }

    /// One-line summary of all metrics, sorted by name for stable output.
    /// Format: `name1=mean/p95ms name2=mean/p95ms ...`
    pub fn summary(&self) -> String {
        let mut names: Vec<&&'static str> = self.rings.keys().collect();
        names.sort();
        let parts: Vec<String> = names
            .iter()
            .map(|name| {
                let ring = &self.rings[*name];
                format!(
                    "{}={:.2}/{:.2}ms",
                    name,
                    ring.mean().as_secs_f64() * 1000.0,
                    ring.p95().as_secs_f64() * 1000.0,
                )
            })
            .collect();
        parts.join(" ")
    }

    /// Test-only accessor for the number of samples currently held by `name`.
    /// Lets unit tests assert recording happened without depending on
    /// wall-clock granularity.
    #[cfg(test)]
    pub fn sample_count(&self, name: &'static str) -> usize {
        self.rings.get(name).map(|r| r.len).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn ring_mean_zero_when_empty() {
        let r = Ring::new();
        assert_eq!(r.mean(), Duration::ZERO);
        assert_eq!(r.p95(), Duration::ZERO);
        assert_eq!(r.len, 0);
    }

    #[test]
    fn ring_mean_basic() {
        let mut r = Ring::new();
        r.push(ms(1));
        r.push(ms(2));
        r.push(ms(3));
        assert_eq!(r.mean(), ms(2));
    }

    #[test]
    fn ring_p95_handles_single_sample() {
        let mut r = Ring::new();
        r.push(ms(7));
        assert_eq!(r.p95(), ms(7));
    }

    #[test]
    fn ring_p95_at_capacity_is_61st_sorted_entry() {
        // Push 0..64 ms in order. p95 = ceil(64 * 0.95) - 1 = ceil(60.8) - 1 = 60.
        // Sorted is 0..64, so index 60 = 60ms.
        let mut r = Ring::new();
        for i in 0..RING_CAPACITY as u64 {
            r.push(ms(i));
        }
        assert_eq!(r.len, RING_CAPACITY);
        assert_eq!(r.p95(), ms(60));
    }

    #[test]
    fn ring_push_then_wrap_preserves_tail_order_and_stats() {
        // Push 0..70 individually. Capacity is 64, so the retained set is 6..70.
        // Mean = (6+7+...+69)/64 = (sum 0..70 - sum 0..6)/64 = (2415 - 15)/64 = 37.5.
        // We use integer ms so the mean rounds to 37 (Duration / u32 truncates).
        let mut r = Ring::new();
        for i in 0..70u64 {
            r.push(ms(i));
        }
        assert_eq!(r.len, RING_CAPACITY);

        // Tail set must be exactly {6, 7, ..., 69}; verify by sorting samples.
        let mut got: Vec<Duration> = r.samples().to_vec();
        got.sort();
        let expected: Vec<Duration> = (6..70u64).map(ms).collect();
        assert_eq!(got, expected, "retained tail must be 6..70 after wrap");

        // p95 = sorted[60] = 6 + 60 = 66ms
        assert_eq!(r.p95(), ms(66));
    }

    #[test]
    fn perf_time_records_a_sample_without_sleeping() {
        // Don't rely on thread::sleep granularity (flaky on CI). Just check
        // that calling .time() actually inserts into the ring.
        let mut perf = Perf::new();
        let result = perf.time("op", || 42u32);
        assert_eq!(result, 42);
        assert_eq!(perf.sample_count("op"), 1);
    }

    #[test]
    fn perf_record_returns_after_one_call() {
        let mut perf = Perf::new();
        perf.record("a", ms(5));
        perf.record("a", ms(3));
        assert_eq!(perf.sample_count("a"), 2);
    }

    #[test]
    fn perf_summary_sorted_alphabetically() {
        let mut perf = Perf::new();
        // Insert in deliberately wrong order to check the sort.
        perf.record("zebra", ms(10));
        perf.record("apple", ms(20));
        perf.record("mango", ms(30));
        let s = perf.summary();
        let apple_pos = s.find("apple").unwrap();
        let mango_pos = s.find("mango").unwrap();
        let zebra_pos = s.find("zebra").unwrap();
        assert!(apple_pos < mango_pos && mango_pos < zebra_pos);
    }

    #[test]
    fn perf_summary_empty_is_empty_string() {
        let perf = Perf::new();
        assert_eq!(perf.summary(), "");
    }

    #[test]
    fn perf_clear_resets_all_rings() {
        let mut perf = Perf::new();
        perf.record("a", ms(1));
        perf.record("b", ms(2));
        perf.record("a", ms(3));
        assert_eq!(perf.sample_count("a"), 2);
        assert_eq!(perf.sample_count("b"), 1);
        perf.clear();
        assert_eq!(perf.sample_count("a"), 0);
        assert_eq!(perf.sample_count("b"), 0);
        // Ring entries themselves still exist; only their content is wiped.
        // After a fresh push, recording works normally.
        perf.record("a", ms(7));
        assert_eq!(perf.sample_count("a"), 1);
    }
}
