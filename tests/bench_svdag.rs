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
//! Confidence note: this file is observational only. The baseline table and
//! printed timings are for manual tracking, not machine-enforced perf gating.
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
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");
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

/// Per-step measurement for docs/perf/svdag-perf-paper.md §4 (Q2/Q4).
///
/// Mirrors the main-loop sync in `src/main.rs` — step_recursive, apply
/// the compaction remap if one fired, incremental `Svdag::update`, and
/// a full `compact` when stale_ratio exceeds 0.5. Per step we record:
///   - u32s appended to `Svdag::nodes` (GPU upload tail, excluding the
///     constant 4-byte root-offset header rewrite at slot 0);
///   - HashlifeStats (cache hits/misses, empty and fixed-point skips);
///   - SVDAG compaction events (full re-uploads).
///
/// Skips a short warmup window per project convention (measure warm
/// frames only; cold-startup generations are not representative of
/// steady-state).
fn bench_svdag_step_deltas(label: &str, level: u32, warmup: u32, measured: u32) {
    let side = 1u64 << level;
    eprintln!(
        "--- {label}: per-step SVDAG upload + memo churn (level={level}, side={side}³, warmup={warmup}, measured={measured}) ---",
    );

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");
    world.seed_water_and_sand();
    eprintln!(
        "  seeded: population={}, store_nodes={}",
        world.population(),
        world.store.node_count(),
    );

    // Cold SVDAG build (not a warm frame; reported separately).
    let mut svdag = Svdag::new();
    let t_cold = Instant::now();
    svdag.update(&world.store, world.root, world.level);
    let cold_us = t_cold.elapsed().as_micros();
    eprintln!(
        "  cold svdag: {}µs, reachable_nodes={}, total_buffer={} u32s ({:.2} MB)",
        cold_us,
        svdag.node_count,
        svdag.nodes.len(),
        svdag.byte_size() as f64 / (1024.0 * 1024.0),
    );

    for _ in 0..warmup {
        step_and_update(&mut world, &mut svdag);
    }

    let mut delta_u32s_per_step: Vec<usize> = Vec::with_capacity(measured as usize);
    let mut compact_events = 0u32;
    let mut hits_total = 0u64;
    let mut misses_total = 0u64;
    let mut empty_skips_total = 0u64;
    let mut fp_skips_total = 0u64;

    for _ in 0..measured {
        let (delta_u32s, compacted) = step_and_update(&mut world, &mut svdag);
        delta_u32s_per_step.push(delta_u32s);
        if compacted {
            compact_events += 1;
        }
        hits_total += world.hashlife_stats.cache_hits;
        misses_total += world.hashlife_stats.cache_misses;
        empty_skips_total += world.hashlife_stats.empty_skips;
        fp_skips_total += world.hashlife_stats.fixed_point_skips;
    }

    let total_u32s: usize = delta_u32s_per_step.iter().sum();
    let mean_u32 = total_u32s as f64 / measured as f64;
    let max_u32 = *delta_u32s_per_step.iter().max().unwrap_or(&0);
    // Per-step upload bytes = 4 (slot-0 root header rewrite) + delta_u32s * 4 (tail append).
    let mean_upload_bytes = 4.0 + mean_u32 * 4.0;
    let max_upload_bytes = 4 + max_u32 * 4;
    let hit_rate = hits_total as f64 / (hits_total as f64 + misses_total as f64).max(1.0);

    eprintln!("  warm-window summary (n={measured}):");
    eprintln!(
        "    upload tail u32s/step: mean={:.1} max={} | upload bytes/step: mean={:.0} max={}",
        mean_u32, max_u32, mean_upload_bytes, max_upload_bytes,
    );
    eprintln!("    svdag compactions fired: {compact_events} (each = full-buffer re-upload)",);
    eprintln!(
        "    hashlife: hits={hits_total} misses={misses_total} hit_rate={:.1}% empty_skips={empty_skips_total} fp_skips={fp_skips_total}",
        hit_rate * 100.0,
    );
    eprintln!(
        "    first {} rows (step: delta_u32s / delta_bytes):",
        (measured as usize).min(10),
    );
    for (i, d) in delta_u32s_per_step.iter().enumerate().take(10) {
        eprintln!("      step {i}: delta_u32s={d}, delta_bytes={}", 4 + d * 4);
    }
    eprintln!();
}

/// One sim step + SVDAG sync, mirroring `src/main.rs:876-886`.
/// Returns `(u32s_appended_this_step, compacted_this_step)`.
fn step_and_update(world: &mut World, svdag: &mut Svdag) -> (usize, bool) {
    world.step_recursive();
    if let Some(remap) = world.last_compaction_remap.take() {
        svdag.apply_remap(&remap);
    }
    let before = svdag.nodes.len();
    svdag.update(&world.store, world.root, world.level);
    let mut delta = svdag.nodes.len().saturating_sub(before);
    let mut compacted = false;
    if svdag.stale_ratio() > 0.5 {
        svdag.compact(&world.store, world.root);
        // Post-compact: fresh buffer, so the renderer re-uploads the whole
        // thing. Charge the full size as the upload "delta" for this step.
        delta = svdag.nodes.len();
        compacted = true;
    }
    (delta, compacted)
}

#[test]
#[ignore]
fn bench_svdag_step_deltas_64() {
    bench_svdag_step_deltas("64³", 6, 5, 40);
}

#[test]
#[ignore]
fn bench_svdag_step_deltas_256() {
    bench_svdag_step_deltas("256³", 8, 5, 40);
}
