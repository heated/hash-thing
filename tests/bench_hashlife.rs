//! Hashlife step benchmarks at multiple world scales (m1f.1).
//!
//! Run with: `cargo test --release -p hash-thing --test bench_hashlife -- --ignored --nocapture`
//!
//! These are `#[ignore]` tests, not criterion benchmarks, to avoid adding
//! dependencies. The `--release` flag is critical — debug builds are 10-50x
//! slower and not representative of real performance.
//!
//! ## Baseline (2026-04-12, M-series Mac, release build)
//!
//! | Scale | Seed   | Mean/gen | Median/gen | Notes                     |
//! |-------|--------|----------|------------|---------------------------|
//! | 512³  | 363ms  | 31.6s    | 31.2s      | 20 gens, terrain seed     |
//! | 1024³ | 1519ms | ~260s    | ~260s      | 2 gens (too slow for 10)  |
//! | 4096³ | —      | —        | —          | not run (projected >30m)  |
//!
//! Key observation: hashlife cache clears every generation, so no
//! cross-generation memoization. Each step re-traverses the full tree.
//! This is the primary optimization target.

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

/// Seed a world at the given level with terrain, then step it `n` times,
/// printing per-generation timing.
fn bench_step(label: &str, level: u32, generations: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³) ---");

    let t0 = Instant::now();
    let mut world = World::new(level);
    let params = TerrainParams::default();
    let stats = world.seed_terrain(&params);
    let seed_ms = t0.elapsed().as_millis();
    eprintln!(
        "  seed: {seed_ms}ms, nodes: {}, gen_region: {}µs",
        world.population(),
        stats.gen_region_us,
    );

    let mut times_us = Vec::with_capacity(generations);
    for gen in 0..generations {
        let t = Instant::now();
        world.step_recursive();
        let us = t.elapsed().as_micros();
        times_us.push(us);
        if gen < 3 || gen == generations - 1 {
            eprintln!(
                "  gen {gen}: {us}µs ({:.1}ms), nodes: {}",
                us as f64 / 1000.0,
                world.population(),
            );
        } else if gen == 3 {
            eprintln!("  ...");
        }
    }

    let total_us: u128 = times_us.iter().sum();
    let mean_us = total_us / generations as u128;
    times_us.sort();
    let median_us = times_us[generations / 2];
    let p95_us = times_us[(generations as f64 * 0.95) as usize];
    eprintln!(
        "  summary: {generations} gens, mean={mean_us}µs, median={median_us}µs, p95={p95_us}µs, total={:.1}ms",
        total_us as f64 / 1000.0,
    );
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
    bench_step("1024³", 10, 10);
}

#[test]
#[ignore]
fn bench_hashlife_4096() {
    bench_step("4096³", 12, 5);
}
