//! SVDAG build benchmarks at multiple world scales (m1f.2).
//!
//! Measures CPU-side SVDAG serialization time. GPU raycast framerate
//! requires a window and is measured via the in-app `render_gpu` perf
//! metric (hash-thing-6x3) during interactive sessions.
//!
//! Run with: `cargo test --release --test bench_svdag -- --ignored --nocapture`
//!
//! ## Baseline (2026-04-12, M-series Mac, release build)
//!
//! | Scale | Cold build | Warm rebuild | Nodes  | Size   |
//! |-------|------------|--------------|--------|--------|
//! | 64³   | 91µs       | 49µs         | 640    | 0.0 MB |
//! | 256³  | 934µs      | 566µs        | 6,773  | 0.2 MB |
//! | 512³  | 3.2ms      | 2.1ms        | 20,228 | 0.7 MB |
//! | 1024³ | 15.1ms     | 7.8ms        | 61,399 | 2.1 MB |
//!
//! SVDAG build scales well — even 1024³ cold build is 15ms (well under
//! a frame budget). The incremental rebuild after a single edit is ~50%
//! faster thanks to slot caching.

use hash_thing::render::Svdag;
use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

fn bench_svdag_build(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label}: SVDAG build (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    let _ = world.seed_terrain(&TerrainParams::default());
    eprintln!("  world seeded, population: {}", world.population());

    // Cold build (no cached slots).
    let t = Instant::now();
    let svdag = Svdag::build(&world.store, world.root, world.level);
    let cold_us = t.elapsed().as_micros();
    let node_count = svdag.node_count;
    let byte_size = svdag.nodes.len() * 4;
    eprintln!(
        "  cold build: {cold_us}µs ({:.1}ms), {node_count} nodes, {:.1} MB",
        cold_us as f64 / 1000.0,
        byte_size as f64 / (1024.0 * 1024.0),
    );

    // Incremental rebuild (warm cache — same root, simulates re-upload).
    let mut svdag = Svdag::new();
    svdag.update(&world.store, world.root, world.level);
    // Edit a cell to force some cache invalidation.
    let center = (side / 4) as i64;
    let material = hash_thing::octree::Cell::pack(1, 0).raw();
    world.set(
        hash_thing::sim::WorldCoord(center),
        hash_thing::sim::WorldCoord(center),
        hash_thing::sim::WorldCoord(center),
        material,
    );
    let t = Instant::now();
    svdag.update(&world.store, world.root, world.level);
    let warm_us = t.elapsed().as_micros();
    eprintln!(
        "  warm rebuild (1 edit): {warm_us}µs ({:.1}ms)",
        warm_us as f64 / 1000.0,
    );
    eprintln!();
}

#[test]
#[ignore]
fn bench_svdag_64() {
    bench_svdag_build("64³", 6);
}

#[test]
#[ignore]
fn bench_svdag_256() {
    bench_svdag_build("256³", 8);
}

#[test]
#[ignore]
fn bench_svdag_512() {
    bench_svdag_build("512³", 9);
}

#[test]
#[ignore]
fn bench_svdag_1024() {
    bench_svdag_build("1024³", 10);
}
