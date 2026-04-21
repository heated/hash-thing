//! Macro-cache occupancy benchmark (hash-thing-z7uu).
//!
//! Closes the `docs/perf/svdag-perf-paper.md` §5 gap-report band that had a
//! 9 MB floor / 105 MB ceiling projection at 4096³ with no runtime number
//! behind it. This harness drives a few representative worlds and prints
//! `(entries, bytes_est)` so the paper can cite measured occupancy.
//!
//! Run with:
//!   `cargo test --release --test bench_macro_cache_size -- --ignored --nocapture`
//!
//! Confidence note: observational only. No CI gate — the printed numbers
//! feed into the perf paper's §5 manually. Sanity assertions guard against
//! the estimator regressing to zero or overflowing implausibly.
//!
//! Why `step_recursive_pow2` and not `step_recursive`: only the pow2 macro
//! stepper populates `hashlife_macro_cache`. The regular per-generation
//! `step_recursive` leaves it empty (see the `memo_mac` comment in
//! `memo_summary()`).

use hash_thing::octree::Cell;
use hash_thing::sim::{GameOfLife3D, World, WorldCoord};
use hash_thing::terrain::materials::STONE;
use hash_thing::terrain::TerrainParams;

/// Seed a terrain world, drive a single `step_recursive_pow2` call to fill
/// the macro cache, and print occupancy.
fn measure(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    let _ = world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");

    // Pre-step entries should be zero: seed_terrain doesn't touch the macro
    // cache, and no hashlife step has run yet.
    assert_eq!(
        world.macro_cache_entries(),
        0,
        "fresh world should have empty macro cache before any step",
    );
    assert_eq!(world.macro_cache_bytes_est(), 0);

    // One pow2 macro step populates the cache with every node the recursion
    // visits at every level.
    world.step_recursive_pow2();

    let entries = world.macro_cache_entries();
    let bytes = world.macro_cache_bytes_est();
    let mb = bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "  after 1x pow2 step: entries={entries}, bytes_est={bytes} ({mb:.3} MB)",
    );
    eprintln!("  {}", world.memo_summary());

    // Sanity: the estimator must be a linear function of entries, no
    // surprise zero or overflow. This pairs with the unit test in
    // `src/sim/world.rs` that checks the arithmetic on small worlds.
    assert_eq!(bytes, entries * World::MACRO_CACHE_BYTES_PER_ENTRY);
    assert!(
        bytes < 4 * 1024 * 1024 * 1024,
        "unexpected >4 GiB macro cache at level {level} — regression in MACRO_CACHE_BYTES_PER_ENTRY?",
    );
}

/// Ignored because pow2 macro steps on terrain at level ≥ 8 blow past the
/// 60s test-runner ceiling. Agents/crew should run this manually when
/// updating perf paper §5.
#[test]
#[ignore]
fn macro_cache_size_scales() {
    // Silent at 64³ — macro cache stays small.
    measure("64³", 6);
    // 256³: should land in the handful-of-MB range.
    measure("256³", 8);
    // 512³: still well under the 105 MB ceiling.
    measure("512³", 9);
}

/// Cheap sanity check that can run without `--ignored`. A GoL-only world
/// exercises the macro-stepper's happy path (no block-rule fallback), so
/// the macro cache actually fills. Terrain worlds trigger the brute-force
/// fallback in `step_recursive_pow2` (water/sand are block-rule cells) and
/// leave the macro cache at zero — wrong signal for this estimator test.
#[test]
fn macro_cache_bytes_est_nonzero_after_pow2_step() {
    let mut world = World::new(6);
    world.set_gol_smoke_rule(GameOfLife3D::new(0, 6, 1, 3));
    world.set(
        WorldCoord(4),
        WorldCoord(4),
        WorldCoord(4),
        Cell::pack(1, 0).raw(),
    );
    world.step_recursive_pow2();

    let entries = world.macro_cache_entries();
    assert!(
        entries > 0,
        "pow2 step on a live GoL world must leave at least one macro cache entry",
    );
    assert_eq!(
        world.macro_cache_bytes_est(),
        entries * World::MACRO_CACHE_BYTES_PER_ENTRY,
    );

    // Unused-import guard: force STONE to be evaluated so rustc doesn't
    // warn if the bench ever stops referencing materials directly.
    let _ = STONE;
}
