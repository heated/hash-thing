//! Cold-gen latency comparing skip-at-gen vs build-then-collapse
//! (`hash-thing-cswp.8.4`).
//!
//! Four columns, fresh `NodeStore` each run, single shared
//! `PrecomputedHeightmapField` (precompute paid once, outside timing):
//!
//! - (a)  full-detail `gen_region`
//! - (b)  `gen_region` + per-chunk `lod_collapse_chunk` (matches cswp.8.3
//!   runtime shape, pessimal due to per-chunk fresh memos)
//! - (b') `gen_region` + global `lod_collapse(root, target_lod)` (better
//!   baseline for strategy comparison; doesn't match runtime)
//! - (c)  `gen_region_with_lod` (new skip-at-gen path)
//!
//! The recommendation gate uses (b')/(c) median speedup AND a divergence
//! cap (≤5% of skipped chunks for `target_lod ≥ 2`); see Adjustment D of
//! `.ship-notes/plan-spark-hash-thing-cswp.8.4-skip-at-gen.md`.
//!
//! Run with:
//! ```text
//! cargo test --profile bench --test bench_cold_gen_lod -- --ignored --nocapture
//! ```
//!
//! Confidence note: observational only — prints timings and a
//! recommendation, does not enforce a CI gate.

use hash_thing::octree::{Node, NodeId, NodeStore};
use hash_thing::terrain::{
    gen_region, gen_region_with_lod, PrecomputedHeightmapField, TerrainParams,
};
use std::time::Instant;

#[derive(Default)]
struct ColumnStats {
    times_us: Vec<u128>,
}

impl ColumnStats {
    fn record(&mut self, us: u128) {
        self.times_us.push(us);
    }
    fn mean(&self) -> f64 {
        let sum: u128 = self.times_us.iter().sum();
        sum as f64 / self.times_us.len() as f64
    }
    fn min(&self) -> u128 {
        *self.times_us.iter().min().unwrap_or(&0)
    }
    fn max(&self) -> u128 {
        *self.times_us.iter().max().unwrap_or(&0)
    }
    fn median(&self) -> u128 {
        let mut sorted = self.times_us.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }
}

/// Walk two subtrees in lockstep at chunk_level granularity, counting
/// chunks where their NodeIds differ. Both `a` and `b` must live in the
/// same `NodeStore` so hash-cons identity is meaningful.
fn count_divergent_chunks(
    store: &NodeStore,
    root_level: u32,
    a: NodeId,
    b: NodeId,
    chunk_level: u32,
) -> u64 {
    // Hash-cons short-circuit: identical subtrees share a NodeId.
    if a == b {
        return 0;
    }
    if root_level == chunk_level {
        return 1;
    }
    let children_a = expand(store, a);
    let children_b = expand(store, b);
    let mut total = 0u64;
    for oct in 0..8 {
        total += count_divergent_chunks(
            store,
            root_level - 1,
            children_a[oct],
            children_b[oct],
            chunk_level,
        );
    }
    total
}

/// Treat a raw `Leaf(s)` at a non-zero contextual level as 8 copies of
/// itself (uniform-region compression). Interior nodes return their
/// stored children.
fn expand(store: &NodeStore, id: NodeId) -> [NodeId; 8] {
    match store.get(id) {
        Node::Interior { children, .. } => *children,
        Node::Leaf(_) => [id; 8],
    }
}

fn run_bench(level: u32, chunk_level: u32, target_lod: u32, runs: usize) {
    let side = 1u64 << level;
    let chunks_per_axis = 1u64 << (level - chunk_level);
    let n_chunks = chunks_per_axis.pow(3);
    eprintln!(
        "--- skip-at-gen vs build-then-collapse, {side}³ (level={level}), \
         chunk_level={chunk_level}, target_lod={target_lod} ({n_chunks} chunks) ---",
    );

    // Build the precomputed field once — shared across all columns/runs.
    let params = TerrainParams::for_level(level);
    params
        .validate()
        .expect("level-derived terrain params must validate");
    let t_pre = Instant::now();
    let field = PrecomputedHeightmapField::new(params.to_heightmap(), level)
        .expect("validated heightmap precompute must succeed");
    eprintln!(
        "  precompute (one-time): {:.1}ms",
        t_pre.elapsed().as_micros() as f64 / 1000.0,
    );

    let mut a_stats = ColumnStats::default();
    let mut b_stats = ColumnStats::default();
    let mut bp_stats = ColumnStats::default();
    let mut c_stats = ColumnStats::default();
    let mut last_lod_skips = 0u64;

    for run in 0..runs {
        // (a) full-detail gen_region
        let mut s_a = NodeStore::new();
        let t = Instant::now();
        let (root_a, _stats_a) = gen_region(&mut s_a, &field, [0, 0, 0], level);
        a_stats.record(t.elapsed().as_micros());
        let pop_a = match s_a.get(root_a) {
            Node::Interior { population, .. } => *population,
            Node::Leaf(s) if *s != 0 => 1u64 << (3 * level),
            _ => 0,
        };
        let nodes_a = s_a.node_count();
        drop(s_a);

        // (b) gen_region + per-chunk lod_collapse_chunk
        let mut s_b = NodeStore::new();
        let t = Instant::now();
        let (mut root_b, _) = gen_region(&mut s_b, &field, [0, 0, 0], level);
        for cz in 0..chunks_per_axis {
            for cy in 0..chunks_per_axis {
                for cx in 0..chunks_per_axis {
                    root_b = s_b.lod_collapse_chunk(root_b, cx, cy, cz, chunk_level, target_lod);
                }
            }
        }
        b_stats.record(t.elapsed().as_micros());
        let nodes_b = s_b.node_count();
        drop(s_b);

        // (b') gen_region + global lod_collapse
        let mut s_bp = NodeStore::new();
        let t = Instant::now();
        let (root_bp, _) = gen_region(&mut s_bp, &field, [0, 0, 0], level);
        let _root_bp_collapsed = s_bp.lod_collapse(root_bp, target_lod);
        bp_stats.record(t.elapsed().as_micros());
        // nodes_bp is intentionally post-collapse: it's the steady-state
        // memory footprint of the build-then-collapse strategy, which is
        // what we want to compare against (c)'s in-place skip-at-gen output.
        let nodes_bp = s_bp.node_count();
        drop(s_bp);

        // (c) gen_region_with_lod (skip-at-gen)
        let mut s_c = NodeStore::new();
        let t = Instant::now();
        let (_root_c, stats_c) =
            gen_region_with_lod(&mut s_c, &field, [0, 0, 0], level, chunk_level, &|_| {
                target_lod
            });
        c_stats.record(t.elapsed().as_micros());
        let nodes_c = s_c.node_count();
        last_lod_skips = stats_c.lod_skips;
        drop(s_c);

        let fmt_us = |us: u128| -> String {
            if us >= 1000 {
                format!("{:.1}ms", us as f64 / 1000.0)
            } else {
                format!("{us}µs")
            }
        };
        eprintln!(
            "  run {run}: (a) {} / (b) {} / (b') {} / (c) {}; \
             pop={pop_a}, nodes a/b/b'/c = {nodes_a}/{nodes_b}/{nodes_bp}/{nodes_c}, \
             lod_skips(c)={}",
            fmt_us(a_stats.times_us[run]),
            fmt_us(b_stats.times_us[run]),
            fmt_us(bp_stats.times_us[run]),
            fmt_us(c_stats.times_us[run]),
            stats_c.lod_skips,
        );
    }

    let fmt_us = |us: f64| -> String {
        if us >= 1000.0 {
            format!("{:>8.1}ms", us / 1000.0)
        } else {
            format!("{us:>8.0}µs")
        }
    };
    let print_col = |label: &str, s: &ColumnStats| {
        eprintln!(
            "  {label:<26}mean={}  min={}  max={}",
            fmt_us(s.mean()),
            fmt_us(s.min() as f64),
            fmt_us(s.max() as f64),
        );
    };
    print_col("(a)  full-detail:", &a_stats);
    print_col("(b)  per-chunk collapse:", &b_stats);
    print_col("(b') global collapse:", &bp_stats);
    print_col("(c)  skip-at-gen:", &c_stats);

    // Speedup ratios on median (robust to a single outlier run).
    let med_b = b_stats.median() as f64;
    let med_bp = bp_stats.median() as f64;
    let med_c = c_stats.median() as f64;
    let speedup_b_c = if med_c > 0.0 { med_b / med_c } else { 0.0 };
    let speedup_bp_c = if med_c > 0.0 { med_bp / med_c } else { 0.0 };
    eprintln!("  speedup (b)/(c)  median: {speedup_b_c:.2}×");
    eprintln!("  speedup (b')/(c) median: {speedup_bp_c:.2}× (recommendation gate baseline)");

    // Divergence: rebuild (b') and (c) into a single shared store, then
    // walk both trees in lockstep at chunk_level granularity. Hash-cons
    // identity (same NodeId ⇒ same subtree) is the metric — but only
    // after BOTH sides are normalized to lod_collapse's canonical shape
    // (raw Leaf at level==target_lod). The skip-at-gen path emits
    // `self.uniform(level, state)` for proof-collapsed chunks at
    // level==target_lod (full interior chain), whereas lod_collapse emits
    // a single raw Leaf there. Without normalization, those material-
    // identical chunks would register as NodeId-divergent. Post-canonicalizing
    // (c) via lod_collapse(target_lod) is idempotent on already-canonical
    // shapes and fixes only the shape mismatch — material differences
    // (skip-path heuristic vs canonical largest-populated-child) survive
    // and are the divergences we actually want to count.
    let mut s_div = NodeStore::new();
    let (b_root, _) = gen_region(&mut s_div, &field, [0, 0, 0], level);
    let bp_root = s_div.lod_collapse(b_root, target_lod);
    let (c_root_raw, _) =
        gen_region_with_lod(&mut s_div, &field, [0, 0, 0], level, chunk_level, &|_| {
            target_lod
        });
    let c_root = s_div.lod_collapse(c_root_raw, target_lod);
    let divergence = count_divergent_chunks(&s_div, level, bp_root, c_root, chunk_level);
    let div_denom_chunks = last_lod_skips.max(1);
    let div_pct_of_skipped = (divergence as f64) * 100.0 / (div_denom_chunks as f64);
    let div_pct_of_all = (divergence as f64) * 100.0 / (n_chunks.max(1) as f64);
    eprintln!(
        "  divergence: {divergence} chunks differ between (b') and (c); \
         {div_pct_of_skipped:.2}% of skipped chunks ({last_lod_skips}), \
         {div_pct_of_all:.2}% of all {n_chunks} chunks",
    );

    let speedup_ok = speedup_bp_c >= 1.5;
    let div_ok = if target_lod == 1 {
        // target_lod=1: the 8-cell heuristic IS the canonical leaf-only
        // rule. Exact agreement required.
        divergence == 0
    } else {
        // target_lod >= 2: heuristic, ≤5% of skipped chunks acceptable.
        if last_lod_skips == 0 {
            true
        } else {
            (divergence as f64) / (last_lod_skips as f64) <= 0.05
        }
    };
    // Cheap regression check: any divergence between (b') and (c) requires
    // the skip path to have fired at least once. Survives even if the
    // canonicalization step is changed.
    debug_assert!(
        divergence == 0 || last_lod_skips > 0,
        "divergence={divergence} but lod_skips=0 (skip path never fired)",
    );

    let recommendation = match (speedup_ok, div_ok) {
        (true, true) => "skip-at-gen wins (recommend gen_region_with_lod)",
        (true, false) => "build-then-collapse is fine (skip-at-gen too divergent)",
        (false, true) => "build-then-collapse is fine (no meaningful speedup)",
        (false, false) => "build-then-collapse is fine (skip-at-gen neither faster nor exact)",
    };
    eprintln!(
        "  gate: speedup (b')/(c)={speedup_bp_c:.0}× (>=1.5×: {speedup_ok}) | \
         divergence={divergence}/{last_lod_skips} skipped \
         (acceptable: {div_ok}) -> {recommendation}",
    );
    eprintln!(
        "  note: gate compares against (b') global lod_collapse, not (b) per-chunk; \
         the runtime path today is closer to (b).",
    );
    eprintln!();
}

#[test]
#[ignore]
fn bench_cold_gen_lod_1024() {
    // Default scale: 1024³ (level=10), chunk_level=8 (256³ chunks), 3 runs.
    //
    // Two target_lod rows for the speed/fidelity tradeoff:
    // - target_lod=1: skip-at-gen heuristic IS the canonical Leaf-only
    //   rule (the 8 octant samples ARE the 8 leaves). Divergence MUST be 0
    //   by construction; this row is the un-deniable speedup baseline.
    // - target_lod=8: maximum collapse (one Leaf per chunk). Heuristic is
    //   approximate; divergence is empirical. This is the row Adjustment D's
    //   gate is computed against.
    run_bench(10, 8, 1, 3);
    run_bench(10, 8, 8, 3);
}

/// 4096³ row, gated on `HASH_THING_BENCH_4096=1` to stay under the
/// soft-max-command-seconds budget (per CLAUDE.md). 4 columns × 3 runs of
/// ~3-4s each plus per-chunk overhead can push 60s without the gate.
#[test]
#[ignore]
fn bench_cold_gen_lod_4096() {
    if std::env::var("HASH_THING_BENCH_4096").ok().as_deref() != Some("1") {
        eprintln!("skipping 4096³ row (set HASH_THING_BENCH_4096=1 to enable)");
        return;
    }
    run_bench(12, 8, 8, 3);
}
