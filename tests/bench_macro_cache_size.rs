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
//!
//! Why GoL worlds and not terrain: `step_recursive_pow2` bails out to a
//! brute-force per-step loop whenever any block-rule cell is present
//! (`src/sim/hashlife.rs:105-116`). Terrain seeds water/sand, which are
//! block-rule materials, so a terrain world leaves the macro cache empty
//! — the exact opposite of what we want to measure. Seeding a GoL-only
//! world keeps the hashlife macro stepper on its happy path.

use hash_thing::octree::Cell;
use hash_thing::sim::{GameOfLife3D, World, WorldCoord};

/// Seed a GoL world with a scattered live pattern, drive a single
/// `step_recursive_pow2` call to fill the macro cache, and print occupancy.
fn measure(label: &str, level: u32) {
    let side = 1u64 << level;
    eprintln!("--- {label} (level={level}, side={side}³) ---");

    let mut world = World::new(level);
    world.set_gol_smoke_rule(GameOfLife3D::new(0, 6, 1, 3));

    // Scatter live cells on a coarse lattice so the hashlife recursion has
    // to descend multiple branches, producing a representative cache
    // occupancy rather than a degenerate single-branch path. A 16³ seed
    // lattice keeps cost bounded while still exercising many octree nodes.
    let side_i = side as i64;
    let seed_stride = (side_i / 16).max(1);
    let mut seeded = 0usize;
    let mut z = 0i64;
    while z < side_i {
        let mut y = 0i64;
        while y < side_i {
            let mut x = 0i64;
            while x < side_i {
                world.set(
                    WorldCoord(x),
                    WorldCoord(y),
                    WorldCoord(z),
                    Cell::pack(1, 0).raw(),
                );
                seeded += 1;
                x += seed_stride;
            }
            y += seed_stride;
        }
        z += seed_stride;
    }
    eprintln!("  seeded {seeded} live cells (stride={seed_stride})");

    // Pre-step: set() mutates the octree but leaves the macro cache at
    // zero — no hashlife step has run yet.
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

    // Sanity: the macro stepper on a live GoL world must actually fill the
    // cache. Zero would mean either the seed didn't land or the stepper
    // bailed out — either way, the printed number is meaningless for the
    // perf paper.
    assert!(
        entries > 0,
        "pow2 step on seeded GoL world at level {level} produced an empty \
         macro cache — regression in step_recursive_pow2 or seeding?",
    );
    assert_eq!(bytes, entries * World::MACRO_CACHE_BYTES_PER_ENTRY);
    assert!(
        bytes < 4 * 1024 * 1024 * 1024,
        "unexpected >4 GiB macro cache at level {level} — regression in MACRO_CACHE_BYTES_PER_ENTRY?",
    );
}

/// Ignored because pow2 macro steps at level ≥ 8 can run long. Agents/crew
/// should run this manually when updating perf paper §5.
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
}
