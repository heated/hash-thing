//! hash-thing-aqq4 thesis probe — is hashlife actually a multiplier?
//!
//! Steady-state `memo_hit` on the default 256³ / 512³ demo scene was
//! reported at 0.15-0.40 (varying with churn). The bead asks: is the
//! ceiling fixable, or is hashlife genuinely capped at ~40% on
//! edit-heavy worlds?
//!
//! This harness measures four scenarios to triangulate the ceiling:
//!
//! 1. **Default terrain** at 128³ (level 7) — equivalent shape to the
//!    demo scene, smaller for fast iteration. Captures the baseline
//!    "what does live demo behaviour look like".
//! 2. **All-stone** at 128³ — every cell is stone, every level-3
//!    sub-cube is uniform inert. Should hit `inert_uniform_state` /
//!    `is_all_inert` short-circuits at every step. Memo_hit + skip
//!    rates should approach 1.0.
//! 3. **Empty** at 128³ — every cell is air. Should hit the
//!    `population == 0` short-circuit at every step. Memo_skip_empty
//!    should approach 1.0.
//! 4. **Repeating uniform** — a sparse stone pattern at 128³ with
//!    blocks at fixed period boundaries, designed to maximize
//!    cache-friendly subtree reuse. Tests how high `memo_hit` can go
//!    on a non-trivial but stable pattern.
//!
//! Output: per-scenario, per-step memo_hit + skip rates +
//! `misses_by_level` distribution. The data feeds the aqq4 verdict
//! ("the ceiling is X%, hashlife-as-multiplier is {viable / borderline
//! / dead}"). Run with:
//!
//! ```text
//! cargo test --profile perf --test memo_hit_thesis_probe -- \
//!     --ignored --nocapture
//! ```
//!
//! Scope: this is a measurement harness, not a perf gate. No assert!.

use hash_thing::octree::Cell;
use hash_thing::sim::{World, WorldCoord};
use hash_thing::terrain::materials::{SAND_MATERIAL_ID, STONE};
use hash_thing::terrain::TerrainParams;

fn wc(coord: u64) -> WorldCoord {
    WorldCoord(coord as i64)
}

#[derive(Default, Clone, Copy)]
struct StepReport {
    cache_hits: u64,
    cache_misses: u64,
    empty_skips: u64,
    fixed_point_skips: u64,
    misses_by_level: [u64; 32],
}

fn snapshot(world: &World) -> StepReport {
    let s = &world.hashlife_stats;
    StepReport {
        cache_hits: s.cache_hits,
        cache_misses: s.cache_misses,
        empty_skips: s.empty_skips,
        fixed_point_skips: s.fixed_point_skips,
        misses_by_level: s.misses_by_level,
    }
}

fn run_and_report(label: &str, mut world: World, n_steps: usize) {
    eprintln!("\n=== {label} ===");
    eprintln!("  level={} side={}³", world.level, 1u32 << world.level);

    // Discard cold setup — first 2 generations populate caches /
    // first-allocation; report steady-state from the remainder.
    let warmup = 2usize.min(n_steps.saturating_sub(1));
    let mut warm: Vec<StepReport> = Vec::with_capacity(n_steps);
    for gen in 0..n_steps {
        world.step_recursive();
        if gen >= warmup {
            warm.push(snapshot(&world));
        }
    }
    if warm.is_empty() {
        eprintln!("  (no warm steps)");
        return;
    }

    let warm_count = warm.len();
    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    let mut total_empty = 0u64;
    let mut total_fixed = 0u64;
    let mut total_misses_by_level = [0u64; 32];
    for r in &warm {
        total_hits += r.cache_hits;
        total_misses += r.cache_misses;
        total_empty += r.empty_skips;
        total_fixed += r.fixed_point_skips;
        for (dst, src) in total_misses_by_level
            .iter_mut()
            .zip(r.misses_by_level.iter())
        {
            *dst += *src;
        }
    }

    let calls = total_hits + total_misses + total_empty + total_fixed;
    let hit_rate = if total_hits + total_misses == 0 {
        0.0
    } else {
        total_hits as f64 / (total_hits + total_misses) as f64
    };
    let skip_empty_rate = if calls == 0 {
        0.0
    } else {
        total_empty as f64 / calls as f64
    };
    let skip_fixed_rate = if calls == 0 {
        0.0
    } else {
        total_fixed as f64 / calls as f64
    };

    eprintln!(
        "  warm steps: {warm_count} (skipped {warmup} cold)",
    );
    eprintln!(
        "  per-step avg: hits={:>9} misses={:>9} empty={:>9} fixed={:>9}",
        total_hits / warm_count as u64,
        total_misses / warm_count as u64,
        total_empty / warm_count as u64,
        total_fixed / warm_count as u64,
    );
    eprintln!(
        "  rates:        memo_hit={:.3} memo_skip_empty={:.3} memo_skip_fixed={:.3}",
        hit_rate, skip_empty_rate, skip_fixed_rate,
    );
    eprintln!("  misses_by_level (cumulative across warm steps):");
    for (idx, &v) in total_misses_by_level.iter().enumerate() {
        if v > 0 {
            let level = idx as u32 + 3;
            eprintln!("    L{level}: {v}");
        }
    }
    let total_lvl_misses: u64 = total_misses_by_level.iter().sum();
    if total_lvl_misses > 0 {
        eprintln!("  level-3 (base case) miss share: {:.3}",
            total_misses_by_level[0] as f64 / total_lvl_misses as f64);
    }
}

fn fresh_default_terrain(level: u32) -> World {
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    let _stats = world.seed_terrain(&params).expect("terrain params validate");
    world
}

fn fresh_all_stone(level: u32) -> World {
    let mut world = World::new(level);
    let side = world.side() as u64;
    for z in 1..side - 1 {
        for y in 1..side - 1 {
            for x in 1..side - 1 {
                world.set(wc(x), wc(y), wc(z), STONE);
            }
        }
    }
    world
}

fn fresh_empty(level: u32) -> World {
    World::new(level)
}

/// 'Repeating uniform' — sparse stone blocks at lattice positions.
/// Many distinct sub-cubes share the same content (one stone cell at
/// the center) so hash-cons should fold them aggressively.
fn fresh_repeating_lattice(level: u32) -> World {
    let mut world = World::new(level);
    let side = world.side() as u64;
    let step = 8u64; // one stone per 8³ block — aligns with level-3 grid
    for z in (step / 2..side - 1).step_by(step as usize) {
        for y in (step / 2..side - 1).step_by(step as usize) {
            for x in (step / 2..side - 1).step_by(step as usize) {
                world.set(wc(x), wc(y), wc(z), STONE);
            }
        }
    }
    world
}

#[test]
#[ignore]
fn aqq4_thesis_probe_default_terrain() {
    // Default-terrain 128³ — equivalent shape to the demo at smaller
    // scale. Lets us read steady-state memo_hit + miss distribution
    // without running the GUI demo.
    let world = fresh_default_terrain(7);
    run_and_report("default-terrain-128", world, 12);
}

#[test]
#[ignore]
fn aqq4_thesis_probe_all_stone() {
    // All-stone uniform inert — should saturate the inert_uniform /
    // all_inert short-circuits.
    let world = fresh_all_stone(7);
    run_and_report("all-stone-128", world, 6);
}

#[test]
#[ignore]
fn aqq4_thesis_probe_empty() {
    // Empty world — should saturate the population==0 short-circuit.
    let world = fresh_empty(7);
    run_and_report("empty-128", world, 6);
}

#[test]
#[ignore]
fn aqq4_thesis_probe_repeating_lattice() {
    // Sparse stone lattice — many identical level-3 sub-cubes (one
    // stone center each). Tests the hash-cons-driven memo_hit ceiling.
    let world = fresh_repeating_lattice(7);
    run_and_report("repeating-lattice-128", world, 8);
}

/// Edit-heavy churn — inject 20 fresh sand cells per step at random
/// positions over default terrain. Models the demo session's actual
/// load (user dropping sand/water during play) which is what the
/// bead's "40% memo_hit" headline was captured under.
///
/// XorShift64* seeded for reproducibility.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn churn_drop_sand(world: &mut World, rng: &mut Rng, n: usize) {
    let side = world.side() as u64;
    let sand = Cell::pack(SAND_MATERIAL_ID, 0).raw();
    for _ in 0..n {
        let x = rng.next_u64() % (side - 4) + 2;
        let y = side - 4 + (rng.next_u64() % 2);
        let z = rng.next_u64() % (side - 4) + 2;
        world.set(wc(x), wc(y), wc(z), sand);
    }
}

fn run_churn_and_report(label: &str, mut world: World, n_steps: usize, sand_per_step: usize) {
    eprintln!("\n=== {label} ===");
    eprintln!(
        "  level={} side={}³ churn={} sand/step",
        world.level,
        1u32 << world.level,
        sand_per_step,
    );
    let mut rng = Rng::new(0xa994_1234);
    let warmup = 2usize.min(n_steps.saturating_sub(1));
    let mut warm: Vec<StepReport> = Vec::with_capacity(n_steps);
    for gen in 0..n_steps {
        churn_drop_sand(&mut world, &mut rng, sand_per_step);
        world.step_recursive();
        if gen >= warmup {
            warm.push(snapshot(&world));
        }
    }
    if warm.is_empty() {
        eprintln!("  (no warm steps)");
        return;
    }
    let warm_count = warm.len();
    let mut total_hits = 0u64;
    let mut total_misses = 0u64;
    let mut total_empty = 0u64;
    let mut total_fixed = 0u64;
    let mut total_misses_by_level = [0u64; 32];
    for r in &warm {
        total_hits += r.cache_hits;
        total_misses += r.cache_misses;
        total_empty += r.empty_skips;
        total_fixed += r.fixed_point_skips;
        for (dst, src) in total_misses_by_level
            .iter_mut()
            .zip(r.misses_by_level.iter())
        {
            *dst += *src;
        }
    }
    let calls = total_hits + total_misses + total_empty + total_fixed;
    let hit_rate = if total_hits + total_misses == 0 {
        0.0
    } else {
        total_hits as f64 / (total_hits + total_misses) as f64
    };
    let skip_empty_rate = if calls == 0 { 0.0 } else { total_empty as f64 / calls as f64 };
    let skip_fixed_rate = if calls == 0 { 0.0 } else { total_fixed as f64 / calls as f64 };
    eprintln!("  warm steps: {warm_count} (skipped {warmup} cold)");
    eprintln!(
        "  per-step avg: hits={:>9} misses={:>9} empty={:>9} fixed={:>9}",
        total_hits / warm_count as u64,
        total_misses / warm_count as u64,
        total_empty / warm_count as u64,
        total_fixed / warm_count as u64,
    );
    eprintln!(
        "  rates:        memo_hit={:.3} memo_skip_empty={:.3} memo_skip_fixed={:.3}",
        hit_rate, skip_empty_rate, skip_fixed_rate,
    );
    eprintln!("  misses_by_level (cumulative across warm steps):");
    for (idx, &v) in total_misses_by_level.iter().enumerate() {
        if v > 0 {
            let level = idx as u32 + 3;
            eprintln!("    L{level}: {v}");
        }
    }
    let total_lvl_misses: u64 = total_misses_by_level.iter().sum();
    if total_lvl_misses > 0 {
        eprintln!(
            "  level-3 (base case) miss share: {:.3}",
            total_misses_by_level[0] as f64 / total_lvl_misses as f64,
        );
    }
}

#[test]
#[ignore]
fn aqq4_thesis_probe_churn_default_terrain() {
    // Edit-heavy: default terrain + 20 sand cells/step injected at the
    // top of the world. Models the demo's user-drops-sand load that
    // the bead's "40% memo_hit" headline was originally captured under.
    let world = fresh_default_terrain(7);
    run_churn_and_report("churn-default-terrain-128", world, 12, 20);
}

/// Convenience: run all 5 probes back-to-back so a single command
/// produces the full thesis report.
#[test]
#[ignore]
fn aqq4_thesis_probe_full_sweep() {
    aqq4_thesis_probe_default_terrain();
    aqq4_thesis_probe_all_stone();
    aqq4_thesis_probe_empty();
    aqq4_thesis_probe_repeating_lattice();
    aqq4_thesis_probe_churn_default_terrain();
}
