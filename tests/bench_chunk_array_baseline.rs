//! Chunk-array baseline bench — provisional comparator for the hashlife
//! navigation epic. **Provisional, not closure-grade.** The honest
//! baseline (parity proofs, cascade scene, regime-coordinate output) is
//! the navigation-epic L0 follow-ups; this bench gives game-direction
//! leads ballpark perf data without waiting on the 2–3 week honest scope.
//!
//! Specifically this bench is NOT:
//! - regime-coordinate formatted,
//! - cascade-scene measured,
//! - same-scene-seeded against hashlife,
//! - phase / tick-divisor-parity validated.
//!
//! What it IS: 32³ default-terrain idle, 30 generations, raw
//! `(gen, ms, pop, drops=0)`.
//!
//! Run with:
//!   cargo test --profile perf -p hash-thing --test bench_chunk_array_baseline -- --ignored --nocapture
//!
//! ## What the timing includes (read this before drawing conclusions)
//!
//! The bench times **`World::step_grid` only** — the CA + Margolus
//! kernel that a chunk-array-native engine would pay. It deliberately
//! skips `commit_step`, which today rebuilds the octree, runs store
//! compaction, and clears the hashlife caches. A real chunk-array
//! engine has none of those — the flat `Vec<CellState>` IS its storage.
//!
//! The sibling `bench_hashlife_32` (in `tests/bench_hashlife.rs`) times
//! `World::step_recursive`, which **includes** hashlife's compaction,
//! cache lookups, and generation advance — those are intrinsic to the
//! hashlife engine, not skippable bookkeeping.
//!
//! So the two timings are "engine-cost per generation" measurements,
//! NOT "kernel-cost per generation." Comparing `step_us` directly is
//! valid only at the engine-cost level. Drawing micro-benchmark
//! conclusions ("hashlife's CA kernel is X× faster than chunk-array's")
//! from these numbers is a category error.
//!
//! Default terrain also includes fire/water materials with divisor > 1.
//! Both `step_grid` and `step_recursive` honor the iowh divisor gate, so
//! the comparison stays apples-to-apples — both engines are measured
//! with their own slow-tick skip path active. Pure-brute-force (no
//! divisor short-circuit) is out of MVP scope.

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
        // Mirror `bench_hashlife.rs::bench_step` verbosity (gens 0–2 +
        // last) so the side-by-side comparison is symmetric. drops=0
        // honors the bead spec shape — the chunk-array path has no
        // entity-drop counter; Margolus-fall counting is a follow-up.
        if gen < 3 || gen == generations - 1 {
            eprintln!(
                "  gen {gen}: {:.3}ms, pop={pop}, drops=0",
                us as f64 / 1000.0
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
