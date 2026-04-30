//! Hashlife step benchmarks at multiple world scales (m1f.1).
//!
//! Run with: `cargo test --profile perf -p hash-thing --test bench_hashlife -- --ignored --nocapture`
//!
//! These are `#[ignore]` tests, not criterion benchmarks, to avoid adding
//! dependencies. `--profile perf` is the intended perf profile per repo
//! policy — it inherits release opts but drops LTO for fast iteration and
//! stays representative for relative latency comparisons (debug builds are
//! 10-50x slower and not representative).
//!
//! Confidence note: this file is observational only. It prints timings and
//! cache stats for manual comparison; it does not enforce a machine-checked
//! performance budget in CI.

use hash_thing::sim::{BaseCaseStrategy, World};
use hash_thing::terrain::materials::STONE;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

/// Seed a world at the given level with terrain, then step it `n` times,
/// printing per-generation timing and cache stats.
fn bench_step(label: &str, level: u32, generations: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³) ---");

    let t0 = Instant::now();
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    let stats = world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    let seed_ms = t0.elapsed().as_millis();
    eprintln!(
        "  seed: {seed_ms}ms (precompute: {}µs, gen: {}µs), pop: {}, \
         leaves: {}, interiors: {}, collapses: {}, nodes: {}",
        stats.precompute_us,
        stats.gen_region_us,
        world.population(),
        stats.leaves,
        stats.interiors_interned,
        stats.total_collapses(),
        stats.nodes_after_gen,
    );

    let mut times_us = Vec::with_capacity(generations);
    for gen in 0..generations {
        let t = Instant::now();
        world.step_recursive();
        let us = t.elapsed().as_micros();
        times_us.push(us);

        let s = world.hashlife_stats;
        let total_lookups = s.cache_hits + s.cache_misses;
        let hit_rate = if total_lookups > 0 {
            s.cache_hits as f64 / total_lookups as f64 * 100.0
        } else {
            0.0
        };

        if gen < 3 || gen == generations - 1 {
            eprintln!(
                "  gen {gen}: {:.1}ms, pop={}, hits={}, misses={}, empty={}, fixed={}, rate={:.1}%",
                us as f64 / 1000.0,
                world.population(),
                s.cache_hits,
                s.cache_misses,
                s.empty_skips,
                s.fixed_point_skips,
                hit_rate,
            );
            // Print per-level miss distribution
            let level_misses: Vec<(usize, u64)> = s
                .misses_by_level
                .iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(i, &c)| (i + 3, c))
                .collect();
            if !level_misses.is_empty() {
                let parts: Vec<String> = level_misses
                    .iter()
                    .map(|(lvl, cnt)| format!("L{lvl}={cnt}"))
                    .collect();
                eprintln!("    misses by level: {}", parts.join(", "));
            }
        } else if gen == 3 {
            eprintln!("  ...");
        }
    }

    if generations > 0 {
        let total_us: u128 = times_us.iter().sum();
        let mean_us = total_us / generations as u128;
        times_us.sort();
        let median_us = times_us[generations / 2];
        let p95_us = times_us[(generations as f64 * 0.95) as usize];
        eprintln!(
            "  summary: {generations} gens, mean={:.1}ms, median={:.1}ms, p95={:.1}ms, total={:.1}s",
            mean_us as f64 / 1000.0,
            median_us as f64 / 1000.0,
            p95_us as f64 / 1000.0,
            total_us as f64 / 1_000_000.0,
        );
        // hash-thing-tk4j (vqke.3): print final memo_summary so the new
        // memo_skip_empty / memo_skip_fixed rates show up in the bench
        // output. Lifetime ratios across all `generations` steps.
        eprintln!("  memo_summary: {}", world.memo_summary());
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_hashlife_512() {
    bench_step("512³", 9, 20);
}

/// hash-thing-tk4j (vqke.3): 256³ matches the szyh baseline scale that
/// surfaced p1=47ms / step=172ms. Used to read the new memo_skip_empty /
/// memo_skip_fixed tokens against representative terrain so the bead can
/// decide whether the dominant 47ms p1 cost is "skip detection isn't
/// firing" (low skip rate → file optimisation bead) or "skip rate is high
/// but per-cell CaRule is genuinely expensive on the unskipped fraction"
/// (different lever — possibly tk4j follow-up on stable-region detection).
#[test]
#[ignore]
fn bench_hashlife_256_tk4j() {
    bench_step("256³ tk4j", 8, 30);
}

/// hash-thing-ftuu (vqke.4) + hash-thing-ecmn (vqke.4.1): same scale as
/// `bench_hashlife_256_tk4j`, run in three passes so the bench output
/// decomposes serial / per-fanout-rayon (ftuu) / BFS (ecmn) side-by-side.
///
/// Drives strategies directly via `set_base_case_strategy` so a single
/// invocation prints all three columns. Compare:
///
/// ```text
/// cargo test --profile perf --test bench_hashlife \
///     bench_hashlife_256_ftuu_rayon_compare -- --ignored --nocapture
/// ```
///
/// **Acceptance** (per ecmn plan review consensus):
/// - Primary metric: `step_us` median (full-step wall, not just leaf
///   compute) and `step_node_wall_ns` median. p1/p2 are sums of
///   per-worker leaf compute under rayon and don't reflect latency.
/// - Pre-committed slack: BFS step_us median ≤ 1.1× RayonPerFanout
///   step_us median on this default-terrain scene. If exceeded, file a
///   follow-up bead and ship behind the env-var only — do not flip
///   default. (Catches "BFS infrastructure ships but never speeds
///   anything up" — adversarial-claude's primary concern.)
/// - Cold warm-up: skip the first 5 generations from median/p95
///   computation to avoid first-allocation + cold-cache distortion.
/// - Bench is informational (no panic) — the absolute "8 ms" bead
///   gate depends on hash-thing-5e3e (churn injector) landing first.
#[test]
#[ignore]
fn bench_hashlife_256_ftuu_rayon_compare() {
    let serial = bench_step_with_strategy("256³ serial", 8, 30, BaseCaseStrategy::Serial);
    let per_fanout = bench_step_with_strategy(
        "256³ ftuu rayon (per-fanout)",
        8,
        30,
        BaseCaseStrategy::RayonPerFanout,
    );
    let bfs =
        bench_step_with_strategy("256³ ecmn rayon (bfs)", 8, 30, BaseCaseStrategy::RayonBfs);

    eprintln!("--- ecmn slack check ---");
    eprintln!(
        "  step_us median: serial={} per-fanout={} bfs={}",
        serial.step_us_median, per_fanout.step_us_median, bfs.step_us_median,
    );
    let slack_ratio = bfs.step_us_median as f64 / per_fanout.step_us_median.max(1) as f64;
    eprintln!(
        "  bfs/per-fanout ratio: {slack_ratio:.3} (slack target: ≤ 1.10 on default terrain)",
    );
    if slack_ratio > 1.10 {
        eprintln!(
            "  WARN: BFS slower than per-fanout by >10% on default terrain. \
             File follow-up bead — do not flip default to RayonBfs.",
        );
    }
}

#[derive(Clone, Copy, Default)]
struct StepBenchSummary {
    step_us_median: u64,
    #[allow(dead_code)]
    step_us_p95: u64,
    #[allow(dead_code)]
    step_node_wall_ms_median: f64,
}

fn bench_step_with_strategy(
    label: &str,
    level: u32,
    generations: usize,
    strategy: BaseCaseStrategy,
) -> StepBenchSummary {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³, strategy={strategy:?}) ---");

    let t0 = Instant::now();
    let mut world = World::new(level);
    world.set_base_case_strategy(strategy);
    let params = TerrainParams::for_level(level);
    let stats = world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    let seed_ms = t0.elapsed().as_millis();
    eprintln!(
        "  seed: {seed_ms}ms (precompute: {}µs, gen: {}µs), pop: {}, \
         nodes: {}",
        stats.precompute_us,
        stats.gen_region_us,
        world.population(),
        stats.nodes_after_gen,
    );

    let mut step_us = Vec::with_capacity(generations);
    let mut p1_ns_per_step = Vec::with_capacity(generations);
    let mut p2_ns_per_step = Vec::with_capacity(generations);
    let mut step_node_ns_per_step = Vec::with_capacity(generations);
    for _gen in 0..generations {
        let t = Instant::now();
        world.step_recursive();
        step_us.push(t.elapsed().as_micros() as u64);
        p1_ns_per_step.push(world.hashlife_stats.phase1_ns);
        p2_ns_per_step.push(world.hashlife_stats.phase2_ns);
        step_node_ns_per_step.push(world.hashlife_stats.step_node_wall_ns);
    }

    let mut summary = StepBenchSummary::default();
    if generations > 0 {
        // hash-thing-ecmn: skip first 5 generations from median/p95
        // (cold-cache + first-allocation distortion).
        let warmup = (5usize).min(generations.saturating_sub(1));
        let warm_step_us: Vec<u64> = step_us[warmup..].to_vec();
        let warm_count = warm_step_us.len();
        let mut sorted = warm_step_us.clone();
        sorted.sort();
        let median_us = sorted[warm_count / 2];
        let p95_us = sorted[((warm_count as f64) * 0.95) as usize];
        let mean_us = warm_step_us.iter().sum::<u64>() / warm_count as u64;

        let warm_step_node_ns: Vec<u64> = step_node_ns_per_step[warmup..].to_vec();
        let mut step_node_sorted = warm_step_node_ns.clone();
        step_node_sorted.sort();
        let step_node_median_ns = step_node_sorted[warm_count / 2];

        let warm_p1: Vec<u64> = p1_ns_per_step[warmup..].to_vec();
        let mut p1_sorted = warm_p1.clone();
        p1_sorted.sort();
        let p1_median_ns = p1_sorted[warm_count / 2];

        let warm_p2: Vec<u64> = p2_ns_per_step[warmup..].to_vec();
        let mut p2_sorted = warm_p2.clone();
        p2_sorted.sort();
        let p2_median_ns = p2_sorted[warm_count / 2];

        eprintln!(
            "  step (warm, skipped {warmup} cold): mean={:.1}ms median={:.1}ms p95={:.1}ms",
            mean_us as f64 / 1000.0,
            median_us as f64 / 1000.0,
            p95_us as f64 / 1000.0,
        );
        eprintln!(
            "  step_node_wall (warm): median={:.1}ms",
            step_node_median_ns as f64 / 1_000_000.0,
        );
        eprintln!(
            "  p1/p2 sums (info — not latency under rayon): p1={:.1}ms p2={:.1}ms",
            p1_median_ns as f64 / 1_000_000.0,
            p2_median_ns as f64 / 1_000_000.0,
        );
        eprintln!("  memo_summary: {}", world.memo_summary());
        summary.step_us_median = median_us;
        summary.step_us_p95 = p95_us;
        summary.step_node_wall_ms_median = step_node_median_ns as f64 / 1_000_000.0;
    }
    eprintln!();
    summary
}

#[test]
#[ignore]
fn bench_hashlife_1024() {
    bench_step("1024³", 10, 5);
}

/// Seed-only 4096³: measures reachable DAG size + seed timing without any
/// warm step. Completes in ~4s on M1 MBA and is safe under the default 60s
/// test-runner ceiling. Use this for fast node-count measurements (e.g.
/// §5 gap report in `docs/perf/svdag-perf-paper.md`).
#[test]
#[ignore]
fn bench_hashlife_4096_seed() {
    bench_step("4096³ seed only", 12, 0);
}

/// One-step 4096³: seed + exactly one warm step. Exceeds the 60s test-runner
/// ceiling on M1-class hardware — run out-of-band with an explicit longer
/// timeout:
/// ```text
/// cargo test --release --test bench_hashlife bench_hashlife_4096_step \
///     -- --ignored --nocapture
/// ```
/// Required to measure warm-step cost at 4096³ (`render_gpu` / macro-cache
/// hit rate). Kept separate from `_seed` so the cheap node-count probe stays
/// agent-runnable under the soft-max-command-seconds budget.
#[test]
#[ignore]
fn bench_hashlife_4096_step() {
    bench_step("4096³ step", 12, 1);
}

/// Measure edit propagation: seed → warm step → place block → re-step.
/// Reports how much of the cache survives a single-cell edit.
fn bench_edit_propagation(label: &str, level: u32, edits: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³, {edits} edits) ---");

    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");

    // Warm step — populate cache
    let t = Instant::now();
    world.step_recursive();
    let warm_ms = t.elapsed().as_millis();
    let warm_stats = world.hashlife_stats;
    eprintln!(
        "  warm step: {warm_ms}ms, hits={}, misses={}, empty={}",
        warm_stats.cache_hits, warm_stats.cache_misses, warm_stats.empty_skips,
    );

    // Place block(s) in the middle of the world, then re-step
    let center = side / 2;
    for i in 0..edits {
        world.set(
            hash_thing::sim::world::WorldCoord(center as i64 + i as i64),
            hash_thing::sim::world::WorldCoord(center as i64),
            hash_thing::sim::world::WorldCoord(center as i64),
            STONE,
        );
    }

    let t = Instant::now();
    world.step_recursive();
    let edit_ms = t.elapsed().as_millis();
    let edit_stats = world.hashlife_stats;
    let total = edit_stats.cache_hits + edit_stats.cache_misses;
    let hit_rate = if total > 0 {
        edit_stats.cache_hits as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let speedup = if edit_ms > 0 {
        warm_ms as f64 / edit_ms as f64
    } else {
        f64::INFINITY
    };
    eprintln!(
        "  post-edit step: {edit_ms}ms ({speedup:.1}x vs warm), hits={}, misses={}, empty={}, rate={hit_rate:.1}%",
        edit_stats.cache_hits, edit_stats.cache_misses, edit_stats.empty_skips,
    );
    eprintln!();
}

/// Water-scene bench (hash-thing-jw3k): seed terrain + water_and_sand, step
/// long enough to reach the "water has impacted ground, actively propagating"
/// regime where the production app shows 440ms/gen at 256^3. Run at 128^3 by
/// default to stay in the soft-max-command-seconds budget; the dynamics regime
/// is the same, only the cell count differs.
fn bench_water_scene(label: &str, level: u32, generations: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³, water+sand) ---");

    let t0 = Instant::now();
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    world.seed_water_and_sand();
    let seed_ms = t0.elapsed().as_millis();
    eprintln!("  seed: {seed_ms}ms, initial pop: {}", world.population(),);

    let mut times_us = Vec::with_capacity(generations);
    let mut populations = Vec::with_capacity(generations);
    for gen in 0..generations {
        let t = Instant::now();
        world.step_recursive();
        let us = t.elapsed().as_micros();
        times_us.push(us);
        populations.push(world.population());

        let s = world.hashlife_stats;
        let total_lookups = s.cache_hits + s.cache_misses;
        let hit_rate = if total_lookups > 0 {
            s.cache_hits as f64 / total_lookups as f64 * 100.0
        } else {
            0.0
        };

        let want_print = gen < 3
            || gen == generations - 1
            || gen == generations / 4
            || gen == generations / 2
            || gen == generations * 3 / 4;
        if want_print {
            eprintln!(
                "  gen {gen}: {:.1}ms, pop={}, hits={}, misses={}, empty={}, fixed={}, rate={:.1}%",
                us as f64 / 1000.0,
                world.population(),
                s.cache_hits,
                s.cache_misses,
                s.empty_skips,
                s.fixed_point_skips,
                hit_rate,
            );
            let level_misses: Vec<(usize, u64)> = s
                .misses_by_level
                .iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(i, &c)| (i + 3, c))
                .collect();
            if !level_misses.is_empty() {
                let parts: Vec<String> = level_misses
                    .iter()
                    .map(|(lvl, cnt)| format!("L{lvl}={cnt}"))
                    .collect();
                eprintln!("    misses by level: {}", parts.join(", "));
            }
        }
    }

    if generations > 0 {
        let total_us: u128 = times_us.iter().sum();
        let mean_us = total_us / generations as u128;
        let mut sorted = times_us.clone();
        sorted.sort();
        let median_us = sorted[generations / 2];
        let p95_us = sorted[(generations as f64 * 0.95) as usize];
        let max_us = *sorted.last().unwrap();
        let mean_pop = populations.iter().sum::<u64>() / generations as u64;
        let max_pop = *populations.iter().max().unwrap();
        eprintln!(
            "  summary: {generations} gens, mean={:.1}ms, median={:.1}ms, p95={:.1}ms, max={:.1}ms, total={:.1}s",
            mean_us as f64 / 1000.0,
            median_us as f64 / 1000.0,
            p95_us as f64 / 1000.0,
            max_us as f64 / 1000.0,
            total_us as f64 / 1_000_000.0,
        );
        // Warm-only stats: skip first 5 gens (cold cache). Helps distinguish
        // "warmup is slow" from "steady-state is slow" (hash-thing-wsq3).
        let warm_skip = 5usize.min(generations.saturating_sub(1));
        if generations > warm_skip {
            let warm_count = generations - warm_skip;
            let warm_total: u128 = times_us.iter().skip(warm_skip).sum();
            let warm_mean = warm_total / warm_count as u128;
            let mut warm_sorted: Vec<u128> = times_us.iter().skip(warm_skip).copied().collect();
            warm_sorted.sort();
            let warm_median = warm_sorted[warm_count / 2];
            let warm_p95 = warm_sorted[(warm_count as f64 * 0.95) as usize];
            let warm_max = *warm_sorted.last().unwrap();
            eprintln!(
                "  warm (skip {warm_skip}): {warm_count} gens, mean={:.1}ms, median={:.1}ms, p95={:.1}ms, max={:.1}ms",
                warm_mean as f64 / 1000.0,
                warm_median as f64 / 1000.0,
                warm_p95 as f64 / 1000.0,
                warm_max as f64 / 1000.0,
            );
        }
        eprintln!(
            "  pop: mean={mean_pop}, max={max_pop}, ratio_to_volume={:.2}",
            max_pop as f64 / (side as f64).powi(3),
        );
        let ns_per_cell_at_max = (max_us as f64 * 1000.0) / max_pop as f64;
        eprintln!("  ns-per-active-cell at max pop: {ns_per_cell_at_max:.1}");
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_hashlife_water_128() {
    bench_water_scene("128³ water+sand", 7, 200);
}

#[test]
#[ignore]
fn bench_hashlife_water_256() {
    bench_water_scene("256³ water+sand", 8, 200);
}

#[test]
#[ignore]
fn bench_hashlife_water_256_short() {
    bench_water_scene("256³ water+sand (short)", 8, 60);
}

/// Phase-level breakdown of step_recursive on the water scene. Identifies
/// where the 440ms/gen budget goes: memoized step_node descent vs. post-step
/// gravity gap-fill (flatten + fill + from_flat) vs. compaction.
fn bench_water_scene_profiled(label: &str, level: u32, generations: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} PROFILED (level={level}, side={side}³, water+sand) ---");

    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    world.seed_water_and_sand();

    // step_node, column_walk, splice, compact, total, dirty_columns
    let mut agg = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut max_total = 0u64;
    let mut compact_count = 0u64;
    let mut rows = Vec::with_capacity(generations);

    for gen in 0..generations {
        let profile = world.step_recursive_profiled();
        agg.0 += profile.step_node_us;
        agg.1 += profile.column_walk_us;
        agg.2 += profile.splice_us;
        agg.3 += profile.compact_us;
        agg.4 += profile.total_us;
        agg.5 += profile.dirty_columns as u64;
        max_total = max_total.max(profile.total_us);
        if profile.compact_us > 1000 {
            compact_count += 1;
        }
        rows.push(profile);

        let want_print = gen < 3
            || gen == generations - 1
            || gen == generations / 4
            || gen == generations / 2
            || gen == generations * 3 / 4
            || profile.compact_us > 1000;
        if want_print {
            eprintln!(
                "  gen {gen}: total={}µs step_node={}µs column_walk={}µs splice={}µs dirty={} compact={}µs",
                profile.total_us,
                profile.step_node_us,
                profile.column_walk_us,
                profile.splice_us,
                profile.dirty_columns,
                profile.compact_us,
            );
        }
    }

    let gens = generations as u64;
    let mean_total_ms = agg.4 as f64 / gens as f64 / 1000.0;
    let pct = |v: u64| -> f64 { v as f64 * 100.0 / agg.4 as f64 };
    eprintln!();
    eprintln!(
        "  PHASE MEANS over {generations} gens (total={mean_total_ms:.1}ms/gen, max={}ms):",
        max_total / 1000
    );
    eprintln!(
        "    step_node:    {:>7.1}ms/gen  ({:>5.1}%)",
        agg.0 as f64 / gens as f64 / 1000.0,
        pct(agg.0)
    );
    eprintln!(
        "    column_walk:  {:>7.1}ms/gen  ({:>5.1}%)",
        agg.1 as f64 / gens as f64 / 1000.0,
        pct(agg.1)
    );
    eprintln!(
        "    splice:       {:>7.1}ms/gen  ({:>5.1}%)   (subset of column_walk)",
        agg.2 as f64 / gens as f64 / 1000.0,
        pct(agg.2)
    );
    eprintln!(
        "    compact:      {:>7.1}ms/gen  ({:>5.1}%)   [{} compactions]",
        agg.3 as f64 / gens as f64 / 1000.0,
        pct(agg.3),
        compact_count,
    );
    eprintln!(
        "    dirty cols:   {:>7.1}/gen   (mean count of columns that triggered splice)",
        agg.5 as f64 / gens as f64,
    );
    eprintln!();
}

#[test]
#[ignore]
fn bench_hashlife_water_128_profiled() {
    bench_water_scene_profiled("128³ water+sand", 7, 200);
}

#[test]
#[ignore]
fn bench_hashlife_water_256_profiled() {
    bench_water_scene_profiled("256³ water+sand", 8, 40);
}

#[test]
#[ignore]
fn bench_hashlife_water_512_profiled() {
    bench_water_scene_profiled("512³ water+sand", 9, 20);
}

#[test]
#[ignore]
fn bench_hashlife_water_512() {
    bench_water_scene("512³ water+sand", 9, 50);
}

/// hash-thing-wsq3 regression bench. Mimics the production app's hot loop:
/// every step has at least one entity-produced `SetCell` mutation queued,
/// drained by `apply_mutations` before `step_recursive` runs. Pre-wsq3,
/// `set_local` cleared all four hashlife caches on every cell write, so this
/// loop paid cold-cache cost (~2-3s/gen at 512^3) even though the per-step
/// mutation surface is one cell. With the wsq3 fix the cache survives, so
/// warm-mean lands at ~200-400ms/gen and the per-step cache hit rate stays
/// in the steady-state regime (50%+) instead of pinned at the cold-step rate
/// (~15%).
///
/// `mean_hit_rate` is the load-bearing assertion — a regression to per-step
/// cache wipe pins hit_rate near the cold ~15% reading regardless of the
/// noisier wall-clock numbers.
fn bench_water_scene_with_mutations(label: &str, level: u32, generations: usize) {
    use hash_thing::sim::world::WorldCoord;
    use hash_thing::sim::WorldMutation;

    let side = 1u64 << level;
    eprintln!("--- {label} +mutations (level={level}, side={side}³, water+sand) ---");

    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    world.seed_water_and_sand();

    // Pick a coord well inside the realized region and not in a region
    // already heavily populated by water/sand — flipping a single cell here
    // gives the cache wipe the same shape it has in production (one
    // entity-produced SetCell per step).
    let edit_x = WorldCoord((side / 2) as i64);
    let edit_y = WorldCoord((side - 4) as i64);
    let edit_z = WorldCoord((side / 2) as i64);

    let mut times_us = Vec::with_capacity(generations);
    let mut hit_rates = Vec::with_capacity(generations);

    for gen in 0..generations {
        // Alternate state so the same cell doesn't dedup into a noop write.
        let state = if gen.is_multiple_of(2) { STONE } else { 0 };
        world.queue.push(WorldMutation::SetCell {
            x: edit_x,
            y: edit_y,
            z: edit_z,
            state,
        });

        let t = Instant::now();
        world.apply_mutations();
        world.step_recursive();
        let us = t.elapsed().as_micros();
        times_us.push(us);

        let s = world.hashlife_stats;
        let total = s.cache_hits + s.cache_misses;
        let hit_rate = if total > 0 {
            s.cache_hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        hit_rates.push(hit_rate);

        let want_print = gen < 3 || gen == generations - 1 || gen.is_multiple_of(10);
        if want_print {
            eprintln!(
                "  gen {gen}: {:.1}ms  hits={} misses={} hit_rate={:.1}%",
                us as f64 / 1000.0,
                s.cache_hits,
                s.cache_misses,
                hit_rate,
            );
        }
    }

    if generations > 0 {
        let total_us: u128 = times_us.iter().sum();
        let mean_us = total_us / generations as u128;
        let mut sorted = times_us.clone();
        sorted.sort();
        let median_us = sorted[generations / 2];
        let p95_us = sorted[(generations as f64 * 0.95) as usize];
        eprintln!(
            "  summary: {generations} gens, mean={:.1}ms, median={:.1}ms, p95={:.1}ms",
            mean_us as f64 / 1000.0,
            median_us as f64 / 1000.0,
            p95_us as f64 / 1000.0,
        );
        let warm_skip = 5usize.min(generations.saturating_sub(1));
        if generations > warm_skip {
            let warm_count = generations - warm_skip;
            let warm_total: u128 = times_us.iter().skip(warm_skip).sum();
            let warm_mean = warm_total / warm_count as u128;
            let warm_hit_rate_mean: f64 =
                hit_rates.iter().skip(warm_skip).sum::<f64>() / warm_count as f64;
            eprintln!(
                "  warm (skip {warm_skip}): {warm_count} gens, mean={:.1}ms, mean_hit_rate={:.1}%",
                warm_mean as f64 / 1000.0,
                warm_hit_rate_mean,
            );
            // Sharp regression signal: a per-step cache wipe pins hit_rate
            // near the cold ~15% reading. Anything above 30% means the
            // cache is surviving across mutation flushes — the wsq3
            // invariant. Threshold is generous on purpose.
            assert!(
                warm_hit_rate_mean >= 30.0,
                "wsq3 regression: warm mean_hit_rate={warm_hit_rate_mean:.1}% < 30%, \
                 cache likely cleared on every mutation flush"
            );
        }
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_hashlife_water_256_with_mutations() {
    bench_water_scene_with_mutations("256³ water+sand", 8, 30);
}

#[test]
#[ignore]
fn bench_hashlife_water_512_with_mutations() {
    bench_water_scene_with_mutations("512³ water+sand", 9, 20);
}

#[test]
#[ignore]
fn bench_edit_64() {
    bench_edit_propagation("64³ single edit", 6, 1);
    bench_edit_propagation("64³ 10 edits", 6, 10);
}

#[test]
#[ignore]
fn bench_edit_256() {
    bench_edit_propagation("256³ single edit", 8, 1);
    bench_edit_propagation("256³ 10 edits", 8, 10);
}

#[test]
#[ignore]
fn bench_edit_512() {
    bench_edit_propagation("512³ single edit", 9, 1);
    bench_edit_propagation("512³ 10 edits", 9, 10);
}

#[test]
#[ignore]
fn bench_edit_cache_impact_64() {
    // Tiny world to see exact cache invalidation numbers
    let mut world = World::new(6);
    let params = TerrainParams::for_level(6);
    world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");

    world.step_recursive();
    let warm = world.hashlife_stats;
    eprintln!(
        "warm:      hits={}, misses={}, empty={}",
        warm.cache_hits, warm.cache_misses, warm.empty_skips
    );

    // No edit — re-step same world (all cache hits)
    world.step_recursive();
    let cached = world.hashlife_stats;
    eprintln!(
        "no-edit:   hits={}, misses={}, empty={}",
        cached.cache_hits, cached.cache_misses, cached.empty_skips
    );

    // One edit — measure cache survival
    world.set(
        hash_thing::sim::world::WorldCoord(32),
        hash_thing::sim::world::WorldCoord(32),
        hash_thing::sim::world::WorldCoord(32),
        STONE,
    );
    world.step_recursive();
    let edited = world.hashlife_stats;
    eprintln!(
        "one-edit:  hits={}, misses={}, empty={}",
        edited.cache_hits, edited.cache_misses, edited.empty_skips
    );
}
