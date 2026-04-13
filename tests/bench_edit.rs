//! Edit propagation benchmarks (m1f.3).
//!
//! Measures the latency of block place/break operations through the octree
//! and the downstream flatten/SVDAG rebuild cost.
//!
//! Run with: `cargo test --release --test bench_edit -- --ignored --nocapture`
//!
//! ## Baseline (2026-04-12, M-series Mac, release build)
//!
//! | Scale | set_cell | flatten  | edit→flatten | Notes                    |
//! |-------|----------|----------|--------------|--------------------------|
//! | 64³   | <1µs     | 560µs    | 560µs        | 0.5MB flat buffer        |
//! | 256³  | <1µs     | 10ms     | 10ms         | 32MB flat buffer         |
//!
//! Key finding: the octree edit (set_cell) is effectively free — it only
//! rewrites the root-to-leaf path via hash-cons. The bottleneck is always
//! `flatten()`, which materializes the full volume into a flat buffer for
//! GPU upload. At 64³ (current default) this is sub-millisecond. At 256³+
//! the flat path is unusable; the SVDAG incremental upload path (5bb.5)
//! bypasses flatten entirely.
//!
//! Cache impact: editing between steps has zero measurable effect on
//! step_recursive timing — the hashlife cache clears every generation
//! regardless of edits.

use hash_thing::octree::Cell;
use hash_thing::sim::{World, WorldCoord};
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

/// Measure single-cell set() latency at various world scales.
fn bench_set_cell(label: &str, level: u32, n_edits: usize) {
    let side = 1u64 << level;
    eprintln!("--- {label}: set_cell (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");

    let material = Cell::pack(1, 0).raw();
    let center = (side / 4) as i64; // avoid boundary

    let mut times_us = Vec::with_capacity(n_edits);
    for i in 0..n_edits {
        let offset = (i % (side as usize / 4)) as i64;
        let t = Instant::now();
        world.set(
            WorldCoord(center + offset),
            WorldCoord(center),
            WorldCoord(center),
            material,
        );
        times_us.push(t.elapsed().as_micros());
    }

    times_us.sort();
    let total_us: u128 = times_us.iter().sum();
    let mean_us = total_us / n_edits as u128;
    let median_us = times_us[n_edits / 2];
    let p95_us = times_us[(n_edits as f64 * 0.95) as usize];
    eprintln!(
        "  {n_edits} edits: mean={mean_us}µs, median={median_us}µs, p95={p95_us}µs, total={:.1}ms",
        total_us as f64 / 1000.0,
    );
}

/// Measure flatten() latency (CPU-side volume extraction for the flat3D path).
fn bench_flatten(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label}: flatten (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");

    // Flatten a few times to get stable numbers.
    let n = 5;
    let mut times_us = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let _data = world.flatten();
        times_us.push(t.elapsed().as_micros());
    }

    times_us.sort();
    let mean_us: u128 = times_us.iter().sum::<u128>() / n as u128;
    let median_us = times_us[n / 2];
    eprintln!("  {n} calls: mean={mean_us}µs, median={median_us}µs");
    eprintln!(
        "  flat buffer size: {} cells ({:.1} MB)",
        side * side * side,
        (side * side * side * 2) as f64 / (1024.0 * 1024.0),
    );
}

/// Measure edit-then-flatten cycle: how long from a block edit to having
/// a fresh flat buffer ready for GPU upload?
fn bench_edit_to_upload(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label}: edit→flatten cycle (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");

    let material = Cell::pack(1, 0).raw();
    let center = (side / 4) as i64;
    let n = 10;
    let mut times_us = Vec::with_capacity(n);

    for i in 0..n {
        let offset = (i % (side as usize / 4)) as i64;
        let t = Instant::now();
        // Edit
        world.set(
            WorldCoord(center + offset),
            WorldCoord(center),
            WorldCoord(center),
            material,
        );
        // Flatten (the CPU part of upload_volume)
        let _data = world.flatten();
        times_us.push(t.elapsed().as_micros());
    }

    times_us.sort();
    let mean_us: u128 = times_us.iter().sum::<u128>() / n as u128;
    let median_us = times_us[n / 2];
    let p95_us = times_us[(n as f64 * 0.95) as usize];
    eprintln!("  {n} cycles: mean={mean_us}µs, median={median_us}µs, p95={p95_us}µs");
}

/// Measure hashlife cache invalidation cost: does editing a cell after
/// stepping (with a warm cache) cause any extra overhead on the next step?
fn bench_edit_cache_impact(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label}: cache impact of edit (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");

    // Step once to warm the cache... except the cache clears each gen.
    // So we're measuring: does an edit between steps change step timing?
    let t = Instant::now();
    world.step_recursive();
    let step_cold = t.elapsed().as_micros();
    eprintln!(
        "  step (no prior edit): {step_cold}µs ({:.1}ms)",
        step_cold as f64 / 1000.0
    );

    // Edit a cell
    let center = (side / 4) as i64;
    let material = Cell::pack(1, 0).raw();
    world.set(
        WorldCoord(center),
        WorldCoord(center),
        WorldCoord(center),
        material,
    );

    let t = Instant::now();
    world.step_recursive();
    let step_after_edit = t.elapsed().as_micros();
    eprintln!(
        "  step (after 1 edit): {step_after_edit}µs ({:.1}ms)",
        step_after_edit as f64 / 1000.0
    );

    let diff_pct = ((step_after_edit as f64 - step_cold as f64) / step_cold as f64) * 100.0;
    eprintln!("  delta: {diff_pct:+.1}% (positive = edit made step slower)");
}

#[test]
#[ignore]
fn bench_edit_512() {
    bench_set_cell("512³", 9, 100);
    bench_flatten("512³", 9);
    bench_edit_to_upload("512³", 9);
}

#[test]
#[ignore]
fn bench_edit_256() {
    bench_set_cell("256³", 8, 100);
    bench_flatten("256³", 8);
    bench_edit_to_upload("256³", 8);
}

#[test]
#[ignore]
fn bench_edit_64() {
    bench_set_cell("64³", 6, 100);
    bench_flatten("64³", 6);
    bench_edit_to_upload("64³", 6);
}

#[test]
#[ignore]
fn bench_edit_cache_impact_64() {
    bench_edit_cache_impact("64³", 6);
}
