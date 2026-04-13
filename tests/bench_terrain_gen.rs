//! Terrain gen benchmarks at multiple world scales (hash-thing-72s).
//!
//! Measures heightmap precomputation + octree gen_region time.
//! Run with: `cargo test --release --test bench_terrain_gen -- --ignored --nocapture`

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

fn bench_terrain(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label}: terrain gen (level={level}, side={side}³) ---");

    let start = Instant::now();
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    let stats = world.seed_terrain(&params).expect("level-derived terrain params must validate");
    let total = start.elapsed();

    eprintln!(
        "  total={:.1}ms  precompute={:.1}ms  gen_region={:.1}ms  nodes={}  leaves={}  collapses={}  pop={}",
        total.as_secs_f64() * 1000.0,
        stats.precompute_us as f64 / 1000.0,
        stats.gen_region_us as f64 / 1000.0,
        stats.nodes_after_gen,
        stats.leaves,
        stats.total_collapses(),
        world.population(),
    );
}

#[test]
#[ignore]
fn bench_terrain_64() {
    bench_terrain("64³", 6);
}

#[test]
#[ignore]
fn bench_terrain_256() {
    bench_terrain("256³", 8);
}

#[test]
#[ignore]
fn bench_terrain_512() {
    bench_terrain("512³", 9);
}

#[test]
#[ignore]
fn bench_terrain_1024() {
    bench_terrain("1024³", 10);
}

#[test]
#[ignore]
fn bench_terrain_2048() {
    bench_terrain("2048³", 11);
}
