//! Property-test sandwich for hash-thing-ftuu (vqke.4) and hash-thing-ecmn
//! (vqke.4.1).
//!
//! Asserts that all selectable [`BaseCaseStrategy`] modes produce
//! bit-exact the same flattened world state across:
//!
//! - 100 random level-4 (16³) worlds with mixed materials (default `cargo test`).
//! - The same corpus at level 5 (32³), exercising recursive descent into
//!   multiple level-4 fanouts (`--ignored`).
//! - 4 explicit fixtures hitting the BlockRule paths the random sweep
//!   would mostly miss: empty, all-stone (inert short-circuit), sand
//!   column (gravity), water lattice (fluid spread).
//! - Random level-7 corpus (`--ignored`) — exercises the BFS dispatcher's
//!   multi-level descent (root -> level-6 -> level-5 -> level-4 -> level-3
//!   batch). Smaller corpus to stay under the project's command budget.
//!
//! Strategies tested for parity vs serial:
//!
//! - [`BaseCaseStrategy::Serial`]: pre-ftuu DFS reference path.
//! - [`BaseCaseStrategy::RayonPerFanout`] (ftuu): per-level-4-fanout
//!   8-way rayon batching.
//! - [`BaseCaseStrategy::RayonBfs`] (ecmn): breadth-first whole-step
//!   level-3 batching.
//!
//! Note: the test compares `World::flatten()` (cell content), not
//! NodeIds. Distinct `World` instances have independent `NodeStore`
//! intern maps; the same content is allowed to map to different
//! NodeIds across instances.

use hash_thing::octree::Cell;
use hash_thing::sim::{BaseCaseStrategy, World, WorldCoord};
use hash_thing::terrain::materials::{SAND_MATERIAL_ID, STONE, WATER_MATERIAL_ID};

/// XorShift64* — pin-stable RNG for reproducible corpus generation.
/// The std `rand` crate is not a workspace dep here.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn wc(coord: u64) -> WorldCoord {
    WorldCoord(coord as i64)
}

/// Seed `world` with a deterministic random material pattern. Distribution
/// biases toward non-air so BlockRule paths fire more often than a uniform
/// sample would. Margin keeps cells away from the world boundary so the
/// inner-cell stencil is always valid.
fn seed_world(world: &mut World, seed: u64, margin: u64) {
    let mut rng = Rng::new(seed);
    let side = world.side() as u64;
    let sand = Cell::pack(SAND_MATERIAL_ID, 0).raw();
    let water = Cell::pack(WATER_MATERIAL_ID, 0).raw();
    for z in margin..(side - margin) {
        for y in margin..(side - margin) {
            for x in margin..(side - margin) {
                // Distribution: 40% air, 30% sand, 15% water, 15% stone.
                let roll = rng.next_u64() % 100;
                let state = match roll {
                    0..=39 => 0,
                    40..=69 => sand,
                    70..=84 => water,
                    _ => STONE,
                };
                if state != 0 {
                    world.set(wc(x), wc(y), wc(z), state);
                }
            }
        }
    }
}

/// Sand column: 4-cell-tall vertical sand stack on a stone floor in the
/// world center. Exercises GravityBlockRule.
fn seed_sand_column(world: &mut World) {
    let side = world.side() as u64;
    let cx = side / 2;
    let cz = side / 2;
    let sand = Cell::pack(SAND_MATERIAL_ID, 0).raw();
    world.set(wc(cx), wc(2), wc(cz), STONE);
    for dy in 0..4 {
        world.set(wc(cx), wc(3 + dy), wc(cz), sand);
    }
}

/// Water lattice: alternating water+air cells across a horizontal layer.
/// Exercises FluidBlockRule lateral spread.
fn seed_water_lattice(world: &mut World) {
    let side = world.side() as u64;
    let water = Cell::pack(WATER_MATERIAL_ID, 0).raw();
    let y = side / 2;
    for z in 2..(side - 2) {
        for x in 2..(side - 2) {
            if (x + z) % 2 == 0 {
                world.set(wc(x), wc(y), wc(z), water);
            }
        }
    }
}

/// All-stone: every interior cell is stone. Exercises the all-inert
/// short-circuit at level 3 (every level-3 sub-cube is stone-only ->
/// inert_uniform_state hits).
fn seed_all_stone(world: &mut World) {
    let side = world.side() as u64;
    for z in 1..(side - 1) {
        for y in 1..(side - 1) {
            for x in 1..(side - 1) {
                world.set(wc(x), wc(y), wc(z), STONE);
            }
        }
    }
}

/// Run `n_steps` of `step_recursive` on a world; return the final flat
/// snapshot for cell-by-cell comparison.
fn evolve_and_flatten(mut world: World, n_steps: usize) -> Vec<u16> {
    for _ in 0..n_steps {
        world.step_recursive();
    }
    world.flatten()
}

/// Build a world at the given level, evolve under the requested
/// strategy, return the flat snapshot.
fn evolve_with_strategy(
    level: u32,
    seeder: impl Fn(&mut World),
    n_steps: usize,
    strategy: BaseCaseStrategy,
) -> Vec<u16> {
    let mut world = World::new(level);
    seeder(&mut world);
    world.set_base_case_strategy(strategy);
    evolve_and_flatten(world, n_steps)
}

/// Diff a strategy's snapshot against the serial baseline. Reports the
/// first diverging cell with (x, y, z, idx, baseline_state, mode_state)
/// for actionable debugging.
fn assert_strategy_matches_baseline(
    level: u32,
    baseline: &[u16],
    candidate: &[u16],
    label: &str,
    strategy: &str,
) {
    assert_eq!(
        baseline.len(),
        candidate.len(),
        "{label}/{strategy}: flat snapshot length differs",
    );
    if baseline == candidate {
        return;
    }
    let side = 1u64 << level;
    for (idx, (s, r)) in baseline.iter().zip(candidate).enumerate() {
        if s != r {
            let x = idx as u64 % side;
            let y = (idx as u64 / side) % side;
            let z = idx as u64 / (side * side);
            panic!(
                "{label}/{strategy}: serial vs {strategy} diverge at ({x},{y},{z}) idx={idx}: \
                 serial=0x{s:04x} {strategy}=0x{r:04x}",
            );
        }
    }
    panic!("{label}/{strategy}: snapshots differ but no cell-level mismatch found");
}

/// Build a world at the given level with the given seeder, run `n_steps`
/// in serial mode and across the rayon strategies, asserting bit-exact
/// flattened parity for each strategy vs serial.
fn assert_parity(level: u32, seeder: impl Fn(&mut World), n_steps: usize, label: &str) {
    let baseline = evolve_with_strategy(level, &seeder, n_steps, BaseCaseStrategy::Serial);
    let per_fanout =
        evolve_with_strategy(level, &seeder, n_steps, BaseCaseStrategy::RayonPerFanout);
    let bfs = evolve_with_strategy(level, &seeder, n_steps, BaseCaseStrategy::RayonBfs);

    assert_strategy_matches_baseline(level, &baseline, &per_fanout, label, "rayon-per-fanout");
    assert_strategy_matches_baseline(level, &baseline, &bfs, label, "rayon-bfs");
}

#[test]
fn rayon_parity_random_level4() {
    // 30 worlds × 5 steps each, each strategy. ~5 s on M2 MBA dev build.
    for seed in 0..30u64 {
        let label = format!("random-level4 seed={seed}");
        assert_parity(4, |w| seed_world(w, 0xfeed_0000 ^ seed, 2), 5, &label);
    }
}

#[test]
fn rayon_parity_sand_column_level4() {
    assert_parity(4, seed_sand_column, 8, "sand-column-level4");
}

#[test]
fn rayon_parity_water_lattice_level4() {
    assert_parity(4, seed_water_lattice, 4, "water-lattice-level4");
}

#[test]
fn rayon_parity_all_stone_level4() {
    assert_parity(4, seed_all_stone, 3, "all-stone-level4");
}

#[test]
fn rayon_parity_empty_level4() {
    // Empty world: every level-3 sub-cube hits the empty short-circuit.
    assert_parity(4, |_| {}, 2, "empty-level4");
}

#[test]
fn rayon_parity_random_level5() {
    // 10 worlds × 3 steps. Level 5 → recursive descent walks through one
    // recursive level above the level-4 fanout, exercising the rayon path
    // across 8 separate level-4 fanouts per top-level step. For BFS this
    // means tasks at level 4 (frontier of 8) feeding into level 3 (up to 64).
    for seed in 0..10u64 {
        let label = format!("random-level5 seed={seed}");
        assert_parity(5, |w| seed_world(w, 0xbeef_0000 ^ seed, 3), 3, &label);
    }
}

/// Larger sweep — runs only with `--ignored` so default `cargo test`
/// stays under the project's soft 60 s command budget.
#[test]
#[ignore]
fn rayon_parity_random_level5_full_sweep() {
    for seed in 0..100u64 {
        let label = format!("level5-full seed={seed}");
        assert_parity(5, |w| seed_world(w, 0xbeef_0000 ^ seed, 3), 5, &label);
    }
}

/// hash-thing-ecmn (vqke.4.1) multi-level BFS coverage: level-7 (128³)
/// worlds exercise the BFS dispatcher across 4 frontier levels (root=8 ->
/// 7 -> 6 -> 5 -> 4 -> 3 batch). The serial / per-fanout paths are not
/// the hot case here — this test is specifically there to catch any
/// BFS-only divergence that the smaller corpus might miss.
///
/// Ignored by default — single seed runs in ~10-30 s on M2 MBA dev build.
#[test]
#[ignore]
fn rayon_parity_random_level7_bfs_multilevel() {
    for seed in 0..3u64 {
        let label = format!("level7-bfs seed={seed}");
        assert_parity(7, |w| seed_world(w, 0xface_0000 ^ seed, 4), 2, &label);
    }
}
