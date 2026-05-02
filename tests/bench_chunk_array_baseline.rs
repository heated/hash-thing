//! Chunk-array baseline bench (hash-thing-8ppq.1.1 MVP).
//!
//! This is the **provisional** comparator. It is NOT the closure-grade
//! baseline that 8ppq.1 calls for. Specifically, it is NOT:
//!
//! - regime-coordinate formatted (that's 8ppq.1.4 + 8ppq.9),
//! - cascade-scene measured (that's 8ppq.1.4),
//! - same-scene-seeded against hashlife (that's 8ppq.1.2),
//! - phase / tick-divisor-parity validated (that's 8ppq.1.3).
//!
//! What it IS: 32³ default-terrain idle, 30 generations, raw `(gen, ms, pop)`.
//! L1 leads (`pa24`, `w88i`, cascade-peak) get ballpark perf data without
//! waiting on the 2-3 week honest baseline.
//!
//! Run with:
//!   cargo test --profile perf -p hash-thing --test bench_chunk_array_baseline -- --ignored --nocapture
//!
//! The bench keeps the flat `Vec<CellState>` canonical and calls
//! `World::step_grid` directly, skipping the octree rebuild that
//! `commit_step` performs. `world.generation` is advanced manually after
//! each step so per-material tick_divisor gating stays consistent —
//! although the MVP scene asserts `all_divisors_one`, so that path doesn't
//! actually fire in this bench. Slow-divisor scenes are 8ppq.1.3 scope.

use hash_thing::octree::CellState;
use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

fn popcount(grid: &[CellState]) -> usize {
    grid.iter().filter(|&&c| c != 0).count()
}

#[test]
#[ignore]
fn bench_chunk_array_baseline_32() {
    let level: u32 = 5; // 32³
    let generations = 30;

    let t_seed = Instant::now();
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    let stats = world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    let seed_ms = t_seed.elapsed().as_millis();

    // Note: default terrain has fire/water with `divisor > 1`, so the
    // iowh-gating branches in `step_grid` and `step_blocks` do fire on
    // odd generations. That's part of the brute-force cost we measure —
    // the chunk-array baseline includes its own divisor skip paths.
    // What 8ppq.1.3 covers is hashlife/brute *parity* under those
    // divisors, not whether the bench should exclude the slow-divisor
    // path entirely. Recording the divisor map up-front makes the
    // measurement reproducible.
    let divisors: Vec<u16> = world.materials().tick_divisor_flags().to_vec();

    let mut grid = world.flatten();
    let initial_pop = popcount(&grid);
    let side = 1u64 << level;
    eprintln!("--- chunk-array baseline (level={level}, side={side}³) ---");
    eprintln!(
        "  seed: {seed_ms}ms, pop={initial_pop}, drops=0 \
         (precompute={}µs, gen={}µs)",
        stats.precompute_us, stats.gen_region_us,
    );
    eprintln!("  tick_divisor_flags: {divisors:?}");

    let mut times_us = Vec::with_capacity(generations);
    for gen in 0..generations {
        let t = Instant::now();
        let next = world.step_grid(&grid);
        let us = t.elapsed().as_micros();
        grid = next;
        // The bench advances `generation` itself because `step_grid` skips
        // `commit_step`. The other side effects of `commit_step`
        // (octree rebuild, store compaction, hashlife cache invalidation)
        // are deliberately not replicated — none of them feed into
        // `step_grid`'s output.
        world.generation += 1;
        times_us.push(us);
        let pop = popcount(&grid);
        // drops=0: the brute-force chunk-array path has no entity-drop
        // counter. The bead spec asks for `drops=D`; we honor the shape
        // with a constant zero and own the interpretation here. If any L1
        // lead actually wants Margolus-fall counts, file a follow-up.
        eprintln!(
            "  gen {gen}: {:.3}ms, pop={pop}, drops=0",
            us as f64 / 1000.0
        );
    }

    let total_us: u128 = times_us.iter().sum();
    let mean_us = total_us / generations as u128;
    times_us.sort();
    let median_us = times_us[generations / 2];
    let p95_us = times_us[(generations as f64 * 0.95) as usize];
    eprintln!(
        "  summary: {generations} gens, mean={:.3}ms, median={:.3}ms, p95={:.3}ms, total={:.1}ms",
        mean_us as f64 / 1000.0,
        median_us as f64 / 1000.0,
        p95_us as f64 / 1000.0,
        total_us as f64 / 1000.0,
    );

    // Cheap sanity: the population can never exceed the cell count. No
    // correctness check beyond that — this is a perf bench, not a
    // regression suite. step_grid + Margolus mass conservation are
    // covered by the existing parity tests in src/sim/world.rs.
    let cells = (side as usize).pow(3);
    assert!(popcount(&grid) <= cells);
}
