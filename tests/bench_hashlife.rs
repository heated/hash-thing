//! Hashlife step benchmarks at multiple world scales (m1f.1).
//!
//! Run with: `cargo test --release -p hash-thing --test bench_hashlife -- --ignored --nocapture`
//!
//! These are `#[ignore]` tests, not criterion benchmarks, to avoid adding
//! dependencies. The `--release` flag is critical — debug builds are 10-50x
//! slower and not representative of real performance.

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

/// Seed a world at the given level with terrain, then step it `n` times,
/// printing per-generation timing and cache stats.
fn bench_step(label: &str, level: u32, generations: usize, spatial_memo: bool) {
    let side = 1u64 << level;
    let mode = if spatial_memo { "spatial" } else { "origin" };
    eprintln!("--- {label} [{mode}] (level={level}, side={side}³) ---");

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

    world.spatial_memo = spatial_memo;

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
                "  gen {gen}: {:.1}ms, pop={}, hits={}, misses={}, empty={}, rate={:.1}%",
                us as f64 / 1000.0,
                world.population(),
                s.cache_hits,
                s.cache_misses,
                s.empty_skips,
                hit_rate,
            );
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
    bench_step("512³ origin-keyed", 9, 20, false);
    bench_step("512³ spatial-memo", 9, 20, true);
}

#[test]
#[ignore]
fn bench_hashlife_1024() {
    bench_step("1024³ origin-keyed", 10, 5, false);
    bench_step("1024³ spatial-memo", 10, 5, true);
}

#[test]
#[ignore]
fn bench_hashlife_4096() {
    bench_step("4096³ origin-keyed", 12, 2, false);
    bench_step("4096³ spatial-memo", 12, 2, true);
}
