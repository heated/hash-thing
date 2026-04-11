//! Per-operation timing and memory accounting for the three hot paths:
//! terrain generation, CA stepping, and SVDAG serialization.
//!
//! This is the h34.3 "perf measurement infrastructure" bead. The point is
//! not a full profiler — `cargo flamegraph` and `tracy` exist for that. The
//! point is a zero-dep, always-on "is everything still within budget?"
//! signal that lives next to the engine and can be logged every couple of
//! seconds while you're poking at the world interactively.
//!
//! ## Design notes
//!
//! - **No global state.** `PerfCounters` is a plain struct owned by `App`.
//!   No `static mut`, no `OnceLock`, no thread-locals. Consistent with the
//!   h34.2 determinism audit.
//! - **Near-zero overhead.** The only measurement is a pair of
//!   `Instant::now()` calls at each call site (~50 ns each on a modern
//!   host). The struct itself is plain POD.
//! - **Running average, not last-sample.** Interactive sessions rarely
//!   show the *last* step's timing — they show "what's it doing now."
//!   Running averages cut through the noise. We also keep last-sample and
//!   peak so you can catch outliers.
//! - **Bytes estimate, not bytes.** Rust doesn't expose per-allocation
//!   sizing. We compute an order-of-magnitude estimate from
//!   `NodeStore::stats()` plus `size_of::<Node>()` and FxHashMap load
//!   factor. Good enough to catch "oh, the store just went from 10MB to
//!   400MB on that gen call."

use std::mem::size_of;
use std::time::{Duration, Instant};

use crate::octree::{Node, NodeId};

/// Per-operation perf counters. Plain POD; owned by `App`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerfCounters {
    pub step: OpCounter,
    pub gen: OpCounter,
    pub svdag: OpCounter,

    /// Node count seen at the most recent sample. Not `Default` — we can
    /// seed it from `NodeStore::stats()` before the first op.
    pub last_node_count: usize,
    /// Step-cache size at the most recent sample.
    pub last_step_cache: usize,
    /// Peak node count across the session.
    pub peak_node_count: usize,
    /// Peak step-cache size across the session.
    pub peak_step_cache: usize,
}

/// Rolling stats for a single operation type.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpCounter {
    /// Total samples recorded.
    pub samples: u64,
    /// Most recent sample.
    pub last: Duration,
    /// Sum of all samples — used to compute a running average.
    pub total: Duration,
    /// Slowest sample seen this session.
    pub peak: Duration,
}

impl OpCounter {
    pub fn record(&mut self, elapsed: Duration) {
        self.samples += 1;
        self.last = elapsed;
        self.total += elapsed;
        if elapsed > self.peak {
            self.peak = elapsed;
        }
    }

    /// Running average across all samples. `Duration::ZERO` if none.
    pub fn average(&self) -> Duration {
        if self.samples == 0 {
            Duration::ZERO
        } else {
            self.total / self.samples as u32
        }
    }
}

impl PerfCounters {
    /// Record a single `step_flat` call.
    pub fn record_step(&mut self, elapsed: Duration, store_stats: (usize, usize)) {
        self.step.record(elapsed);
        self.update_store_stats(store_stats);
    }

    /// Record a single `gen_region` call.
    pub fn record_gen(&mut self, elapsed: Duration, store_stats: (usize, usize)) {
        self.gen.record(elapsed);
        self.update_store_stats(store_stats);
    }

    /// Record a single SVDAG build+upload call.
    pub fn record_svdag(&mut self, elapsed: Duration, store_stats: (usize, usize)) {
        self.svdag.record(elapsed);
        self.update_store_stats(store_stats);
    }

    fn update_store_stats(&mut self, (nodes, step_cache): (usize, usize)) {
        self.last_node_count = nodes;
        self.last_step_cache = step_cache;
        if nodes > self.peak_node_count {
            self.peak_node_count = nodes;
        }
        if step_cache > self.peak_step_cache {
            self.peak_step_cache = step_cache;
        }
    }

    /// Human-readable single-line log summary. Picks the units that make
    /// the numbers legible (ms if the sample is over a millisecond, µs
    /// otherwise) so eyeballing "is this fast?" doesn't need arithmetic.
    pub fn summary(&self) -> String {
        format!(
            "perf: step[{}] last={} avg={} peak={} | gen[{}] last={} avg={} peak={} | \
             svdag[{}] last={} avg={} peak={} | store={} nodes (peak {}), step-cache={} (peak {}), ~{}",
            self.step.samples,
            fmt_dur(self.step.last),
            fmt_dur(self.step.average()),
            fmt_dur(self.step.peak),
            self.gen.samples,
            fmt_dur(self.gen.last),
            fmt_dur(self.gen.average()),
            fmt_dur(self.gen.peak),
            self.svdag.samples,
            fmt_dur(self.svdag.last),
            fmt_dur(self.svdag.average()),
            fmt_dur(self.svdag.peak),
            self.last_node_count,
            self.peak_node_count,
            self.last_step_cache,
            self.peak_step_cache,
            fmt_bytes(memory_bytes_estimate(
                self.last_node_count,
                self.last_step_cache,
            )),
        )
    }
}

/// Scope timer — runs a closure and records its duration into a counter.
/// Uses `Instant::now()` at both ends; the closure itself is where all the
/// cost is. Returns whatever the closure returns.
///
/// Example:
/// ```ignore
/// let (result, elapsed) = perf::time(|| world.step_flat(&rule));
/// counters.record_step(elapsed, world.store.stats());
/// ```
pub fn time<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

/// Rough estimate of `NodeStore` bytes. See module doc for why this is
/// order-of-magnitude.
///
/// Per-entry sizes are computed at compile time from `size_of::<Node>()`
/// and `size_of::<NodeId>()`, so if the representation of either ever
/// changes (e.g. `Node::Interior` gains a field, `NodeId` widens to u64 on
/// the lazy-root-expansion path), this estimate tracks automatically. The
/// h34.2 audit calls out the step-cache rule-id question explicitly — I do
/// not want to chase a hardcoded `NODE_BYTES` constant the day that lands.
///
/// The `* 2` factor on the hashbrown entries is **deliberate pessimism**:
/// hashbrown's real per-entry cost at its 0.875 max load factor is around
/// `(key + val + 1 control byte) / 0.875 ≈ 1.14×` the raw payload. We
/// round up to 2× because this counter is a "did we just OOM?" alarm, not
/// an accountant — overestimating keeps the alarm sensitive.
const NODE_BYTES: usize = size_of::<Node>();
const NODE_ID_BYTES: usize = size_of::<NodeId>();
const NODES_ENTRY: usize = NODE_BYTES; // Vec<Node>
const INTERN_ENTRY: usize = (NODE_BYTES + NODE_ID_BYTES) * 2; // FxHashMap<Node, NodeId>
const STEP_CACHE_ENTRY: usize = (NODE_ID_BYTES + NODE_ID_BYTES) * 2; // FxHashMap<NodeId, NodeId>

pub fn memory_bytes_estimate(node_count: usize, step_cache: usize) -> usize {
    // intern and nodes are 1:1 — every interned node lives in both.
    node_count * (NODES_ENTRY + INTERN_ENTRY) + step_cache * STEP_CACHE_ENTRY
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_micros();
    if us >= 1_000 {
        format!("{:.2}ms", us as f64 / 1_000.0)
    } else {
        format!("{us}µs")
    }
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

    #[test]
    fn op_counter_records_samples_and_averages() {
        let mut c = OpCounter::default();
        assert_eq!(c.samples, 0);
        assert_eq!(c.average(), Duration::ZERO);

        c.record(Duration::from_micros(100));
        c.record(Duration::from_micros(200));
        c.record(Duration::from_micros(300));

        assert_eq!(c.samples, 3);
        assert_eq!(c.last, Duration::from_micros(300));
        assert_eq!(c.peak, Duration::from_micros(300));
        assert_eq!(c.total, Duration::from_micros(600));
        assert_eq!(c.average(), Duration::from_micros(200));
    }

    #[test]
    fn op_counter_peak_tracks_maximum_across_samples() {
        let mut c = OpCounter::default();
        c.record(Duration::from_micros(500));
        c.record(Duration::from_micros(100)); // lower, should not displace peak
        c.record(Duration::from_micros(300));
        assert_eq!(c.peak, Duration::from_micros(500));
    }

    #[test]
    fn perf_counters_track_peaks_per_op() {
        let mut p = PerfCounters::default();
        p.record_step(Duration::from_micros(10), (100, 5));
        p.record_gen(Duration::from_micros(20), (200, 5));
        p.record_svdag(Duration::from_micros(30), (200, 5));

        assert_eq!(p.step.samples, 1);
        assert_eq!(p.gen.samples, 1);
        assert_eq!(p.svdag.samples, 1);
        assert_eq!(p.last_node_count, 200);
        assert_eq!(p.peak_node_count, 200);
        assert_eq!(p.peak_step_cache, 5);
    }

    #[test]
    fn op_counters_isolate_peaks_across_types() {
        // Regression guard against "record_step accidentally writes into
        // gen.peak" — the kind of thing copy/paste mutations would
        // introduce. Interleave samples across all three ops.
        let mut p = PerfCounters::default();
        p.record_step(Duration::from_micros(100), (0, 0));
        p.record_gen(Duration::from_micros(500), (0, 0));
        p.record_step(Duration::from_micros(200), (0, 0));
        p.record_svdag(Duration::from_micros(50), (0, 0));
        p.record_gen(Duration::from_micros(300), (0, 0));

        // Step peak is 200 (not 500 or 50).
        assert_eq!(p.step.peak, Duration::from_micros(200));
        assert_eq!(p.step.samples, 2);
        // Gen peak is 500 (not 300, not step's 200).
        assert_eq!(p.gen.peak, Duration::from_micros(500));
        assert_eq!(p.gen.samples, 2);
        // SVDAG peak is 50 (its only sample).
        assert_eq!(p.svdag.peak, Duration::from_micros(50));
        assert_eq!(p.svdag.samples, 1);
    }

    #[test]
    fn perf_counters_peak_node_count_never_retreats() {
        let mut p = PerfCounters::default();
        p.record_gen(Duration::from_micros(1), (1000, 0));
        // Simulate a reset: last_node_count drops but peak should hold.
        p.record_step(Duration::from_micros(1), (200, 0));
        assert_eq!(p.last_node_count, 200);
        assert_eq!(p.peak_node_count, 1000);
    }

    #[test]
    fn time_returns_closure_result() {
        // `time` is trivial — the only thing worth asserting is that it
        // threads the closure's return value through. Duration correctness
        // is the OS clock's job. (An earlier version of this test checked
        // `elapsed >= Duration::ZERO`, which is tautological for an
        // unsigned type.)
        let (result, _elapsed) = time(|| 40 + 2);
        assert_eq!(result, 42);
    }

    #[test]
    fn time_measures_a_sleeping_closure() {
        // Soft-check the clock actually ticks. 1ms is loose enough that
        // any CI runner with a working monotonic clock passes. If this
        // ever flakes, the bug is not in `time()`.
        let (_, elapsed) = time(|| std::thread::sleep(Duration::from_millis(2)));
        assert!(
            elapsed >= Duration::from_millis(1),
            "expected >= 1ms, got {elapsed:?}",
        );
    }

    #[test]
    fn node_bytes_constants_are_plausible() {
        // Lock the sizes so a surprise blow-up in Node's representation is
        // caught here rather than silently ballooning the `store=~N bytes`
        // log line. 48 is today's `Interior { level: u32, children: [NodeId;
        // 8], population: u64 }` with niche-optimized enum tag; `NodeId` is
        // a 4-byte `u32` wrapper. If either changes, update the range here
        // and double-check the audit-h34.2 memory notes.
        assert!(
            (32..=96).contains(&NODE_BYTES),
            "NODE_BYTES out of expected range: {NODE_BYTES}",
        );
        assert_eq!(NODE_ID_BYTES, 4);
    }

    #[test]
    fn memory_estimate_scales_linearly_with_node_count() {
        let a = memory_bytes_estimate(1_000, 0);
        let b = memory_bytes_estimate(2_000, 0);
        // Double the nodes → double the bytes (exact, since the formula is
        // pure multiplication).
        assert_eq!(b, 2 * a);
    }

    #[test]
    fn memory_estimate_adds_step_cache_contribution() {
        let base = memory_bytes_estimate(1_000, 0);
        let with_cache = memory_bytes_estimate(1_000, 500);
        assert!(with_cache > base);
        assert_eq!(with_cache - base, 500 * STEP_CACHE_ENTRY);
    }

    #[test]
    fn fmt_dur_switches_between_us_and_ms() {
        assert_eq!(fmt_dur(Duration::from_micros(999)), "999µs");
        assert_eq!(fmt_dur(Duration::from_micros(1_000)), "1.00ms");
        assert_eq!(fmt_dur(Duration::from_micros(1_500)), "1.50ms");
    }

    #[test]
    fn fmt_bytes_picks_readable_unit() {
        assert_eq!(fmt_bytes(512), "512B");
        assert_eq!(fmt_bytes(2 * 1024), "2.00KB");
        assert_eq!(fmt_bytes(3 * 1024 * 1024), "3.00MB");
    }

    #[test]
    fn summary_contains_all_three_op_types() {
        let mut p = PerfCounters::default();
        p.record_step(Duration::from_micros(100), (10, 2));
        p.record_gen(Duration::from_millis(5), (100, 0));
        p.record_svdag(Duration::from_micros(50), (100, 0));
        let s = p.summary();
        assert!(s.contains("step"));
        assert!(s.contains("gen"));
        assert!(s.contains("svdag"));
        assert!(s.contains("store="));
    }
}
