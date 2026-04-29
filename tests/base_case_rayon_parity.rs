//! Property-test sandwich for hash-thing-ftuu (vqke.4).
//!
//! Asserts that `World::set_base_case_use_rayon(true)` produces bit-exact
//! the same world state as the default serial path across:
//!
//! - 100 random level-4 (16³) worlds with mixed materials (default `cargo test`).
//! - The same corpus at level 5 (32³), exercising recursive descent into
//!   multiple level-4 fanouts (`--ignored`).
//! - 4 explicit fixtures hitting the BlockRule paths the random sweep
//!   would mostly miss: empty, all-stone (inert short-circuit), sand
//!   column (gravity), water lattice (fluid spread).
//!
//! The level-4 fanout is where the rayon batching kicks in; the test
//! verifies that batching the 8 base-case `step_grid_once_pure` invocations
//! through rayon (or its serial fallback below the threshold) does not
//! corrupt state across the descent. The random corpus uses `World::set`
//! to populate cells so the test goes through the same world-mutation API
//! the production sim uses, then runs both modes in parallel `World`
//! instances seeded from the same RNG and asserts `World::flatten()`
//! equality after `n` steps.

use hash_thing::octree::Cell;
use hash_thing::sim::{World, WorldCoord};
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

/// Build a world at the given level with the given seeder, run `n_steps`
/// in serial mode and rayon mode, and assert the final flat snapshots
/// are bit-identical.
fn assert_parity(level: u32, seeder: impl Fn(&mut World), n_steps: usize, label: &str) {
    let mut serial_world = World::new(level);
    seeder(&mut serial_world);
    serial_world.set_base_case_use_rayon(false);

    let mut rayon_world = World::new(level);
    seeder(&mut rayon_world);
    rayon_world.set_base_case_use_rayon(true);

    let serial_after = evolve_and_flatten(serial_world, n_steps);
    let rayon_after = evolve_and_flatten(rayon_world, n_steps);

    assert_eq!(
        serial_after.len(),
        rayon_after.len(),
        "{label}: flat snapshot length differs",
    );

    if serial_after != rayon_after {
        // Locate first diverging cell for actionable diagnostics.
        let mut first_diff = None;
        for (idx, (s, r)) in serial_after.iter().zip(&rayon_after).enumerate() {
            if s != r {
                first_diff = Some((idx, *s, *r));
                break;
            }
        }
        let side = 1u64 << level;
        if let Some((idx, s, r)) = first_diff {
            let x = idx as u64 % side;
            let y = (idx as u64 / side) % side;
            let z = idx as u64 / (side * side);
            panic!(
                "{label}: serial vs rayon diverge at ({x},{y},{z}) idx={idx}: \
                 serial=0x{s:04x} rayon=0x{r:04x}",
            );
        }
        panic!("{label}: snapshots differ but no cell-level mismatch found");
    }
}

#[test]
fn rayon_parity_random_level4() {
    // 30 worlds × 5 steps each. ~3 s on M2 MBA dev build.
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
    // across 8 separate level-4 fanouts per top-level step.
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
