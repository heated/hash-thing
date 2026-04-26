//! Cold-gen latency at 1024³ and 4096³ — baseline for big-map streaming
//! (`hash-thing-71t7`, `hash-thing-cswp`).
//!
//! Times `World::new + seed_terrain` end-to-end (precompute + gen_region +
//! intern), 3-run mean. Establishes the baseline against which gen-time
//! hash-consing (cswp.3) is later measured.
//!
//! Run with:
//! ```text
//! cargo test --profile bench --test bench_cold_gen_big_map -- --ignored --nocapture
//! ```
//!
//! Confidence note: observational only — prints timings for the perf paper,
//! does not enforce a CI gate.

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

fn bench_cold_gen(label: &str, level: u32, runs: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label} cold gen (level={level}, side={side}³, runs={runs}) ---");

    let mut totals_us = Vec::with_capacity(runs);
    let mut precomp_us = Vec::with_capacity(runs);
    let mut gen_us = Vec::with_capacity(runs);
    let mut last_pop = 0u64;
    let mut last_nodes = 0u64;

    for run in 0..runs {
        let t0 = Instant::now();
        let mut world = World::new(level);
        let params = TerrainParams::for_level(level);
        let stats = world
            .seed_terrain(&params)
            .expect("level-derived terrain params must validate");
        let total = t0.elapsed().as_micros();

        totals_us.push(total);
        precomp_us.push(stats.precompute_us as u128);
        gen_us.push(stats.gen_region_us as u128);
        last_pop = world.population();
        last_nodes = stats.nodes_after_gen as u64;

        eprintln!(
            "  run {run}: total={:.1}ms (precompute={:.1}ms, gen_region={:.1}ms), pop={last_pop}, nodes={last_nodes}",
            total as f64 / 1000.0,
            stats.precompute_us as f64 / 1000.0,
            stats.gen_region_us as f64 / 1000.0,
        );

        // Drop world before next iteration to free memory between runs.
        drop(world);
    }

    let mean = |xs: &[u128]| -> f64 {
        let sum: u128 = xs.iter().sum();
        sum as f64 / xs.len() as f64
    };
    eprintln!(
        "  mean: total={:.1}ms, precompute={:.1}ms, gen_region={:.1}ms, pop={last_pop}, nodes={last_nodes}",
        mean(&totals_us) / 1000.0,
        mean(&precomp_us) / 1000.0,
        mean(&gen_us) / 1000.0,
    );
    eprintln!();
}

#[test]
#[ignore]
fn bench_cold_gen_1024() {
    bench_cold_gen("1024³", 10, 3);
}

/// 4096³ cold gen runs in ~4s on M1-class hardware (per `bench_hashlife_4096_seed`),
/// so a 3-run mean fits comfortably under the 60s soft-max-command-seconds budget.
/// Memory ceiling: peak resident set during seed is dominated by the heightmap
/// precompute (4096² f32 ≈ 64 MB) plus the intermediate octree before
/// hash-consing collapses duplicates. Has been observed to seed and free
/// successfully on a 16 GB M-series machine; if a smaller box OOMs, record the
/// failure mode in the bench output rather than guessing.
#[test]
#[ignore]
fn bench_cold_gen_4096() {
    bench_cold_gen("4096³", 12, 3);
}
