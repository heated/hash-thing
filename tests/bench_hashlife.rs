//! Hashlife step benchmarks at multiple world scales (m1f.1).
//!
//! Run with: `cargo test --profile bench --test bench_hashlife -- --ignored --nocapture`
//! (or `--release` for full LTO builds; bench profile is faster to compile)
//!
//! These are `#[ignore]` tests, not criterion benchmarks, to avoid adding
//! dependencies. Optimized builds are critical — debug builds are 10-50x
//! slower and not representative of real performance.

use hash_thing::octree::Cell;
use hash_thing::sim::world::WorldCoord;
use hash_thing::sim::World;
use hash_thing::terrain::materials::{STONE, WATER_MATERIAL_ID};
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

/// Seed a world at the given level with terrain, then step it `n` times,
/// printing per-generation timing and cache stats.
///
/// `warmup` generations run first (cold cache + parity miss). Only the
/// subsequent `generations` count toward the summary statistics. This
/// separates cold-start overhead from steady-state performance.
fn bench_step(label: &str, level: u32, warmup: usize, generations: usize) {
    let side = 1u64 << level;
    eprintln!(
        "--- {label} (level={level}, side={side}³, {warmup} warmup + {generations} measured) ---"
    );

    let t0 = Instant::now();
    let mut world = World::new(level);
    let params = TerrainParams::default();
    let stats = world.seed_terrain(&params);
    let seed_ms = t0.elapsed().as_millis();
    eprintln!(
        "  seed: {seed_ms}ms, population: {}, gen_region: {}µs",
        world.population(),
        stats.gen_region_us,
    );

    // Warmup: run cold generations, report but don't count
    for gen in 0..warmup {
        let t = Instant::now();
        world.step_recursive();
        let us = t.elapsed().as_micros();
        let s = world.hashlife_stats;
        let total_lookups = s.cache_hits + s.cache_misses;
        let hit_rate = if total_lookups > 0 {
            s.cache_hits as f64 / total_lookups as f64 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  warmup {gen}: {:.1}ms, pop={}, hits={}, misses={}, empty={}, fixed={}, rate={:.1}%",
            us as f64 / 1000.0,
            world.population(),
            s.cache_hits,
            s.cache_misses,
            s.empty_skips,
            s.fixed_point_skips,
            hit_rate,
        );
    }

    // Measured generations
    let mut times_us = Vec::with_capacity(generations);
    for i in 0..generations {
        let gen = warmup + i;
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

        if i < 3 || i == generations - 1 {
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
        } else if i == 3 {
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
            "  summary (warm only): {generations} gens, mean={:.1}ms, median={:.1}ms, p95={:.1}ms, total={:.1}s",
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
    bench_step("512³", 9, 2, 18);
}

#[test]
#[ignore]
fn bench_hashlife_1024() {
    bench_step("1024³", 10, 2, 3);
}

#[test]
#[ignore]
fn bench_hashlife_4096() {
    bench_step("4096³", 12, 2, 2);
}

/// Measure edit propagation: seed → warm step → place block → re-step.
/// Reports how much of the cache survives a single-cell edit.
fn bench_edit_propagation(label: &str, level: u32, edits: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³, {edits} edits) ---");

    let mut world = World::new(level);
    let params = TerrainParams::default();
    world.seed_terrain(&params);

    // Warm step — populate cache
    let t = Instant::now();
    world.step_recursive();
    let warm_ms = t.elapsed().as_millis();
    let warm_stats = world.hashlife_stats;
    eprintln!(
        "  warm step: {warm_ms}ms, hits={}, misses={}, empty={}, fixed={}",
        warm_stats.cache_hits,
        warm_stats.cache_misses,
        warm_stats.empty_skips,
        warm_stats.fixed_point_skips,
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
        "  post-edit step: {edit_ms}ms ({speedup:.1}x vs warm), hits={}, misses={}, empty={}, fixed={}, rate={hit_rate:.1}%",
        edit_stats.cache_hits,
        edit_stats.cache_misses,
        edit_stats.empty_skips,
        edit_stats.fixed_point_skips,
    );
    eprintln!();
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
    let params = TerrainParams::default();
    world.seed_terrain(&params);

    world.step_recursive();
    let warm = world.hashlife_stats;
    eprintln!(
        "warm:      hits={}, misses={}, empty={}, fixed={}",
        warm.cache_hits, warm.cache_misses, warm.empty_skips, warm.fixed_point_skips
    );

    // No edit — re-step same world (all cache hits)
    world.step_recursive();
    let cached = world.hashlife_stats;
    eprintln!(
        "no-edit:   hits={}, misses={}, empty={}, fixed={}",
        cached.cache_hits, cached.cache_misses, cached.empty_skips, cached.fixed_point_skips
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
        "one-edit:  hits={}, misses={}, empty={}, fixed={}",
        edited.cache_hits, edited.cache_misses, edited.empty_skips, edited.fixed_point_skips
    );
}

/// Benchmark a partially-active world: terrain + a column of water that
/// falls and spreads. This exercises the case where most of the world is
/// inert but a small region has active block rules (gravity, lateral spread).
#[test]
#[ignore]
fn bench_active_water_512() {
    let level = 9;
    let side = 1u64 << level;
    eprintln!("--- 512³ with active water column ---");

    let mut world = World::new(level);
    let params = TerrainParams::default();
    let stats = world.seed_terrain(&params);
    eprintln!(
        "  seed: population={}, gen_region={}µs",
        world.population(),
        stats.gen_region_us,
    );

    // Place a column of water at the center, above the terrain surface.
    // At 512³ (level 9), terrain fills roughly the bottom half.
    let center = side / 2;
    let water = Cell::pack(WATER_MATERIAL_ID, 0).raw();
    let mut water_count = 0u64;
    for y in (center..center + 32).rev() {
        for dx in 0..4 {
            for dz in 0..4 {
                world.set(
                    WorldCoord((center + dx) as i64),
                    WorldCoord(y as i64),
                    WorldCoord((center + dz) as i64),
                    water,
                );
                water_count += 1;
            }
        }
    }
    eprintln!(
        "  placed {water_count} water cells, population={}",
        world.population()
    );

    let mut times_us = Vec::with_capacity(20);
    for gen in 0..20 {
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

        if !(5..18).contains(&gen) {
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
        } else if gen == 5 {
            eprintln!("  ...");
        }
    }

    let total_us: u128 = times_us.iter().sum();
    let mean_us = total_us / 20;
    times_us.sort();
    let median_us = times_us[10];
    let p95_us = times_us[19];
    eprintln!(
        "  summary: 20 gens, mean={:.1}ms, median={:.1}ms, p95={:.1}ms, total={:.1}s",
        mean_us as f64 / 1000.0,
        median_us as f64 / 1000.0,
        p95_us as f64 / 1000.0,
        total_us as f64 / 1_000_000.0,
    );
}
