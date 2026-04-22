//! Hashlife step benchmarks at multiple world scales (m1f.1).
//!
//! Run with: `cargo test --release -p hash-thing --test bench_hashlife -- --ignored --nocapture`
//!
//! These are `#[ignore]` tests, not criterion benchmarks, to avoid adding
//! dependencies. The `--release` flag is critical — debug builds are 10-50x
//! slower and not representative of real performance.
//!
//! Confidence note: this file is observational only. It prints timings and
//! cache stats for manual comparison; it does not enforce a machine-checked
//! performance budget in CI.

use hash_thing::sim::World;
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
    }
    eprintln!();
}

#[test]
#[ignore]
fn bench_hashlife_512() {
    bench_step("512³", 9, 20);
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
    eprintln!(
        "  seed: {seed_ms}ms, initial pop: {}",
        world.population(),
    );

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

    let mut agg = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64); // step_node, flatten, gf_loop, from_flat, compact, total
    let mut max_total = 0u64;
    let mut compact_count = 0u64;
    let mut rows = Vec::with_capacity(generations);

    for gen in 0..generations {
        let profile = world.step_recursive_profiled();
        agg.0 += profile.step_node_us;
        agg.1 += profile.flatten_us;
        agg.2 += profile.gap_fill_loop_us;
        agg.3 += profile.from_flat_us;
        agg.4 += profile.compact_us;
        agg.5 += profile.total_us;
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
                "  gen {gen}: total={}µs step_node={}µs flatten={}µs gf_loop={}µs from_flat={}µs compact={}µs",
                profile.total_us,
                profile.step_node_us,
                profile.flatten_us,
                profile.gap_fill_loop_us,
                profile.from_flat_us,
                profile.compact_us,
            );
        }
    }

    let gens = generations as u64;
    let mean_total_ms = agg.5 as f64 / gens as f64 / 1000.0;
    let pct = |v: u64| -> f64 { v as f64 * 100.0 / agg.5 as f64 };
    eprintln!();
    eprintln!(
        "  PHASE MEANS over {generations} gens (total={mean_total_ms:.1}ms/gen, max={}ms):",
        max_total / 1000
    );
    eprintln!(
        "    step_node:  {:>7.1}ms/gen  ({:>5.1}%)",
        agg.0 as f64 / gens as f64 / 1000.0,
        pct(agg.0)
    );
    eprintln!(
        "    flatten:    {:>7.1}ms/gen  ({:>5.1}%)",
        agg.1 as f64 / gens as f64 / 1000.0,
        pct(agg.1)
    );
    eprintln!(
        "    gf_loop:    {:>7.1}ms/gen  ({:>5.1}%)",
        agg.2 as f64 / gens as f64 / 1000.0,
        pct(agg.2)
    );
    eprintln!(
        "    from_flat:  {:>7.1}ms/gen  ({:>5.1}%)",
        agg.3 as f64 / gens as f64 / 1000.0,
        pct(agg.3)
    );
    eprintln!(
        "    compact:    {:>7.1}ms/gen  ({:>5.1}%)   [{} compactions]",
        agg.4 as f64 / gens as f64 / 1000.0,
        pct(agg.4),
        compact_count,
    );
    let gap_fill_total = agg.1 + agg.2 + agg.3;
    eprintln!(
        "    gap_fill (combined): {:>7.1}ms/gen  ({:>5.1}%)",
        gap_fill_total as f64 / gens as f64 / 1000.0,
        pct(gap_fill_total),
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
