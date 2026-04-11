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
use std::mem::size_of;
use std::time::Duration;

use crate::octree::{Node, NodeId};

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

/// Memory-watchdog metric family for `NodeStore`.
///
/// Orthogonal to `Perf` (which tracks latency). `MemStats` is the "did we
/// just blow the memory budget?" alarm — it watches node count and
/// step-cache size, tracks ratcheting peaks, and exposes an
/// order-of-magnitude byte estimate derived at compile time from
/// `size_of::<Node>()` and `size_of::<NodeId>()`.
///
/// Ratcheting is deliberate: we want the OOM signal to survive a transient
/// dip in live state. Call `reset_peaks()` on a world-reset boundary if you
/// want the signal to follow the new world's growth curve.
///
/// The byte estimate is intentionally pessimistic — see the `NODE_BYTES`
/// constants below for the math. This is an alarm, not an accountant;
/// overestimating keeps it sensitive.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemStats {
    /// Node count at the most recent `update` call.
    pub last_node_count: usize,
    /// Step-cache size at the most recent `update` call.
    pub last_step_cache: usize,
    /// Highest node count seen since the last `reset_peaks` (or `new`).
    pub peak_node_count: usize,
    /// Highest step-cache size seen since the last `reset_peaks` (or `new`).
    pub peak_step_cache: usize,
}

impl MemStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fresh sample from `NodeStore::stats()`. Peaks ratchet — they
    /// never retreat. `last_*` tracks the most recent sample directly.
    pub fn update(&mut self, node_count: usize, step_cache: usize) {
        self.last_node_count = node_count;
        self.last_step_cache = step_cache;
        if node_count > self.peak_node_count {
            self.peak_node_count = node_count;
        }
        if step_cache > self.peak_step_cache {
            self.peak_step_cache = step_cache;
        }
    }

    /// Drop the peaks (and the last-sample) back to zero. Use on a world
    /// reset so the post-reset alarm tracks the new world's growth curve.
    pub fn reset_peaks(&mut self) {
        *self = Self::default();
    }

    /// Order-of-magnitude byte estimate for the current (`last_*`) sample.
    /// See `memory_bytes_estimate` for the math.
    pub fn memory_bytes_estimate(&self) -> usize {
        memory_bytes_estimate(self.last_node_count, self.last_step_cache)
    }

    /// One-line summary suitable for periodic logging. Format:
    /// `mem: nodes=N (peak P), step-cache=C (peak P), ~BYTES`
    pub fn summary(&self) -> String {
        format!(
            "mem: nodes={} (peak {}), step-cache={} (peak {}), ~{}",
            self.last_node_count,
            self.peak_node_count,
            self.last_step_cache,
            self.peak_step_cache,
            fmt_bytes(self.memory_bytes_estimate()),
        )
    }
}

/// Rough byte estimate for a `NodeStore` with `node_count` interned nodes
/// and `step_cache` cached step entries.
///
/// Per-entry sizes are derived at compile time from `size_of::<Node>()`
/// and `size_of::<NodeId>()`, so widening either type (e.g. 32-bit →
/// 64-bit `NodeId` when lazy-root-expansion lands) is picked up
/// automatically. No hardcoded `NODE_BYTES` constant to chase.
///
/// The `* 2` factor on hash-map entries is **deliberate pessimism.** Real
/// hashbrown load factor is ~0.875, so real per-entry cost is ~1.14× the
/// raw payload. Rounding up to 2× keeps the OOM alarm sensitive rather
/// than accurate — overestimating is the safe direction for an alarm.
const NODE_BYTES: usize = size_of::<Node>();
const NODE_ID_BYTES: usize = size_of::<NodeId>();
const NODES_ENTRY: usize = NODE_BYTES; // Vec<Node> slot
const INTERN_ENTRY: usize = (NODE_BYTES + NODE_ID_BYTES) * 2; // FxHashMap<Node, NodeId>
const STEP_CACHE_ENTRY: usize = (NODE_ID_BYTES + NODE_ID_BYTES) * 2; // FxHashMap<NodeId, NodeId>

pub fn memory_bytes_estimate(node_count: usize, step_cache: usize) -> usize {
    // Interned nodes and their NodeId slots are 1:1 — every interned node
    // lives in both the Vec<Node> and the FxHashMap<Node, NodeId>.
    node_count * (NODES_ENTRY + INTERN_ENTRY) + step_cache * STEP_CACHE_ENTRY
}

fn fmt_bytes(bytes: usize) -> String {
    const KB: usize = 1 << 10;
    const MB: usize = 1 << 20;
    const GB: usize = 1 << 30;
    if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
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

    #[test]
    fn memstats_update_tracks_last_sample() {
        let mut m = MemStats::new();
        m.update(100, 50);
        assert_eq!(m.last_node_count, 100);
        assert_eq!(m.last_step_cache, 50);
        m.update(42, 7);
        assert_eq!(m.last_node_count, 42);
        assert_eq!(m.last_step_cache, 7);
    }

    #[test]
    fn memstats_peak_never_retreats() {
        // Ratchet property: peaks only climb, never descend, even when the
        // live sample shrinks. This is the "did we ever blow the budget?"
        // signal — it must survive transient dips in live state.
        let mut m = MemStats::new();
        m.update(100, 50);
        m.update(200, 100);
        m.update(50, 10); // big dip
        assert_eq!(m.last_node_count, 50);
        assert_eq!(m.last_step_cache, 10);
        assert_eq!(m.peak_node_count, 200);
        assert_eq!(m.peak_step_cache, 100);
        // Push above previous peak: peak should climb.
        m.update(300, 150);
        assert_eq!(m.peak_node_count, 300);
        assert_eq!(m.peak_step_cache, 150);
    }

    #[test]
    fn memstats_reset_peaks_zeroes_everything() {
        let mut m = MemStats::new();
        m.update(500, 200);
        m.update(300, 150);
        m.reset_peaks();
        assert_eq!(m.last_node_count, 0);
        assert_eq!(m.last_step_cache, 0);
        assert_eq!(m.peak_node_count, 0);
        assert_eq!(m.peak_step_cache, 0);
    }

    #[test]
    fn memory_bytes_estimate_scales_linearly_with_node_count() {
        // Linearity property: doubling node count doubles the estimate
        // (step_cache contribution held at zero). Pins the formula so a
        // future refactor that tacks on a constant overhead term is caught.
        let zero = memory_bytes_estimate(0, 0);
        assert_eq!(zero, 0);
        let a = memory_bytes_estimate(1000, 0);
        let b = memory_bytes_estimate(2000, 0);
        assert_eq!(b, 2 * a);
        // Step cache contribution is independent and additive.
        let with_cache = memory_bytes_estimate(1000, 500);
        assert!(with_cache > a);
        assert_eq!(with_cache - a, memory_bytes_estimate(0, 500));
    }

    #[test]
    fn memstats_summary_reports_last_and_peak() {
        let mut m = MemStats::new();
        m.update(100, 50);
        m.update(200, 100);
        m.update(50, 10);
        let s = m.summary();
        assert!(s.contains("nodes=50"));
        assert!(s.contains("peak 200"));
        assert!(s.contains("step-cache=10"));
        assert!(s.contains("peak 100"));
    }

    #[test]
    fn fmt_bytes_picks_sensible_units() {
        assert_eq!(fmt_bytes(0), "0B");
        assert_eq!(fmt_bytes(1023), "1023B");
        assert!(fmt_bytes(2 * 1024).ends_with("KB"));
        assert!(fmt_bytes(5 * 1024 * 1024).ends_with("MB"));
        assert!(fmt_bytes(3 * 1024 * 1024 * 1024).ends_with("GB"));
    }
}
