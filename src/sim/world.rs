use super::rule::{CaRule, ALIVE};
use crate::octree::{CellState, NodeId, NodeStore};
use crate::terrain::materials::MaterialRegistry;
use crate::terrain::{carve_caves, gen_region, GenStats, TerrainParams};

/// The simulation world. Owns the octree store and manages stepping.
///
/// For now, stepping works by flattening to a grid, applying rules, and
/// rebuilding the octree. This is O(n³) but correct, and lets us validate
/// the full pipeline. True Hashlife recursive stepping comes next.
pub struct World {
    pub store: NodeStore,
    pub root: NodeId,
    pub level: u32, // root level — grid is 2^level per side
    pub generation: u64,
    pub materials: MaterialRegistry,
}

impl World {
    /// Create a new empty world of given level (side = 2^level).
    pub fn new(level: u32) -> Self {
        let mut store = NodeStore::new();
        let root = store.empty(level);
        let materials = MaterialRegistry::terrain_defaults();
        assert!(
            materials.rule_for_state(ALIVE).is_some(),
            "default material registry must define the legacy ALIVE payload",
        );
        Self {
            store,
            root,
            level,
            generation: 0,
            materials,
        }
    }

    pub fn side(&self) -> usize {
        1 << self.level
    }

    /// Set a cell.
    ///
    /// **Panics** on out-of-bounds coordinates (hash-thing-fb5). Silent data
    /// loss is unacceptable; once `World::ensure_contains` (hash-thing-819
    /// contract half) and lazy root expansion (hash-thing-e9h) land, callers
    /// will route OOB writes through explicit realization — for now the
    /// primitive defends itself.
    #[track_caller]
    pub fn set(&mut self, x: u64, y: u64, z: u64, state: CellState) {
        let side = self.side() as u64;
        assert!(
            x < side && y < side && z < side,
            "World::set: coord ({x}, {y}, {z}) out of bounds for side {side}",
        );
        self.root = self.store.set_cell(self.root, x, y, z, state);
    }

    /// Get a cell.
    ///
    /// Out-of-bounds reads return `0` silently (hash-thing-fb5). `get` is a
    /// pure query over the realized region — outside that region is
    /// conceptually unrealized empty space.
    #[allow(dead_code)]
    pub fn get(&self, x: u64, y: u64, z: u64) -> CellState {
        self.store.get_cell(self.root, x, y, z)
    }

    /// Flatten to a 3D grid for rendering.
    pub fn flatten(&self) -> Vec<CellState> {
        self.store.flatten(self.root, self.side())
    }

    /// Step the simulation forward one generation using brute-force grid evaluation.
    /// This is the simple path — true Hashlife stepping replaces this later.
    pub fn step_flat(&mut self, rule: &dyn CaRule) {
        let side = self.side();
        let grid = self.flatten();
        let mut next = vec![0 as CellState; side * side * side];

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let center = grid[x + y * side + z * side * side];
                    let neighbors = get_neighbors(&grid, side, x, y, z);
                    next[x + y * side + z * side * side] = rule.step_cell(center, &neighbors);
                }
            }
        }

        self.root = self.store.from_flat(&next, side);
        self.generation += 1;

        // Fresh-store compaction: `from_flat` interned a brand new generation
        // into the append-only store, leaving the previous generation's
        // subtrees unreachable but still present. Rebuild into a fresh store
        // so memory tracks live-scene size, not cumulative history.
        // See hash-thing-88d.
        let (new_store, new_root) = self.store.compacted(self.root);
        self.store = new_store;
        self.root = new_root;
    }

    /// Place a random seed pattern in the center of the world.
    ///
    /// The loop bounds are clamped to `[0, side)` so `radius > center` does
    /// not underflow `u64` (which would debug-panic or release-wrap into an
    /// 18-quintillion-iteration hang).
    pub fn seed_center(&mut self, radius: u64, density: f64) {
        let side = self.side() as u64;
        let center = side / 2;
        let lo = center.saturating_sub(radius);
        let hi = center.saturating_add(radius).min(side);
        let mut rng = SimpleRng::new(42);
        for z in lo..hi {
            for y in lo..hi {
                for x in lo..hi {
                    // Spherical mask
                    let dx = x as f64 - center as f64;
                    let dy = y as f64 - center as f64;
                    let dz = z as f64 - center as f64;
                    if dx * dx + dy * dy + dz * dz < (radius as f64 * radius as f64)
                        && rng.next_f64() < density
                    {
                        self.set(x, y, z, ALIVE);
                    }
                }
            }
        }
    }

    pub fn population(&self) -> u64 {
        self.store.population(self.root)
    }

    /// Replace the world with terrain generated from `params`. Uses
    /// a fresh `NodeStore` and clears the step cache — every `NodeId`
    /// from the previous world is invalidated. Resets `generation` to 0.
    ///
    /// ## Fresh-store epoch
    ///
    /// `seed_terrain` is a new epoch: everything from the previous world
    /// is unreachable after this returns, so there is no point interning
    /// the new terrain into a store full of garbage. Building into a
    /// fresh store is cheaper (no post-build compaction walk), keeps
    /// node counts honest, and makes the epoch boundary explicit.
    ///
    /// **The caller MUST keep the simulation paused around this call.**
    pub fn seed_terrain(&mut self, params: &TerrainParams) -> GenStats {
        params.validate().expect("invalid TerrainParams");
        self.store = NodeStore::new();
        self.store.clear_step_cache();
        let field = params.to_heightmap();
        let gen_start = std::time::Instant::now();
        let (mut root, mut stats) = gen_region(&mut self.store, &field, [0, 0, 0], self.level);
        stats.gen_region_us = gen_start.elapsed().as_micros() as u64;
        stats.nodes_after_gen = self.store.stats().0;
        // Opt-in cave-CA post-pass. Runs as a separate stage after the
        // heightmap recursion so the baseline perf path (and every
        // pre-caves test) sees identical work when `params.caves` is
        // `None`.
        if let Some(cave_params) = params.caves {
            let cave_start = std::time::Instant::now();
            root = carve_caves(&mut self.store, root, self.level, &cave_params);
            stats.cave_us = cave_start.elapsed().as_micros() as u64;
        }
        stats.nodes_after_caves = self.store.stats().0;
        self.root = root;
        self.generation = 0;
        stats
    }
}

/// Get the 26 Moore neighbors of a cell, wrapping at boundaries.
fn get_neighbors(grid: &[CellState], side: usize, x: usize, y: usize, z: usize) -> [CellState; 26] {
    let mut neighbors = [0 as CellState; 26];
    let mut idx = 0;
    for dz in [-1i32, 0, 1] {
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = (x as i32 + dx).rem_euclid(side as i32) as usize;
                let ny = (y as i32 + dy).rem_euclid(side as i32) as usize;
                let nz = (z as i32 + dz).rem_euclid(side as i32) as usize;
                neighbors[idx] = grid[nx + ny * side + nz * side * side];
                idx += 1;
            }
        }
    }
    neighbors
}

/// Minimal RNG — xorshift64 so we don't need a dependency.
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    //! Reference tests for the brute-force step_flat.

    use super::*;
    use crate::octree::Cell;
    use crate::sim::rule::{GameOfLife3D, ALIVE};

    /// Helper: build an empty 8^3 world (level=3).
    fn empty_world() -> World {
        World::new(3)
    }

    // -----------------------------------------------------------------
    // 1. Empty world stays empty under amoeba.
    // -----------------------------------------------------------------
    #[test]
    fn empty_world_stays_empty_under_amoeba() {
        let mut world = empty_world();
        let rule = GameOfLife3D::amoeba();

        world.step_flat(&rule);

        assert_eq!(
            world.population(),
            0,
            "empty world must stay empty after one step"
        );
        assert!(
            world.flatten().iter().all(|&c| c == 0),
            "every cell in the flattened grid must still be dead"
        );
    }

    // -----------------------------------------------------------------
    // 2. Single live cell under amoeba dies (zero neighbors).
    // -----------------------------------------------------------------
    #[test]
    fn single_cell_dies_under_amoeba() {
        let mut world = empty_world();
        world.set(4, 4, 4, ALIVE);
        assert_eq!(world.population(), 1);

        let rule = GameOfLife3D::amoeba();
        world.step_flat(&rule);

        assert_eq!(
            world.population(),
            0,
            "lone live cell must die under amoeba"
        );
    }

    // -----------------------------------------------------------------
    // 3. Single live cell under crystal grows to a 3x3x3 cube.
    // -----------------------------------------------------------------
    #[test]
    fn single_cell_grows_to_3x3x3_cube_under_crystal() {
        let mut world = empty_world();
        world.set(4, 4, 4, ALIVE);

        let rule = GameOfLife3D::crystal();
        world.step_flat(&rule);

        assert_eq!(
            world.population(),
            27,
            "crystal single cell must grow to a 3^3 cube"
        );

        for z in 3..=5u64 {
            for y in 3..=5u64 {
                for x in 3..=5u64 {
                    assert_eq!(
                        world.get(x, y, z),
                        ALIVE,
                        "cell ({},{},{}) must be alive inside the expected 3x3x3 cube",
                        x,
                        y,
                        z
                    );
                }
            }
        }

        // Spot-check that cells just outside the cube are still dead.
        assert_eq!(world.get(2, 4, 4), 0);
        assert_eq!(world.get(6, 4, 4), 0);
        assert_eq!(world.get(4, 2, 4), 0);
        assert_eq!(world.get(4, 6, 4), 0);
        assert_eq!(world.get(4, 4, 2), 0);
        assert_eq!(world.get(4, 4, 6), 0);
    }

    // -----------------------------------------------------------------
    // 4. 2x2x2 cube is a still-life under custom S7-7/B3-3.
    // -----------------------------------------------------------------
    #[test]
    fn cube_2x2x2_is_still_life_under_s7_b3() {
        let rule = GameOfLife3D::new(7, 7, 3, 3);

        let mut world = empty_world();
        for z in 3..=4u64 {
            for y in 3..=4u64 {
                for x in 3..=4u64 {
                    world.set(x, y, z, ALIVE);
                }
            }
        }
        assert_eq!(world.population(), 8, "initial cube must have 8 cells");

        for gen in 1..=5 {
            world.step_flat(&rule);
            assert_eq!(
                world.population(),
                8,
                "cube must still have 8 cells at gen {}",
                gen
            );

            for z in 0..8u64 {
                for y in 0..8u64 {
                    for x in 0..8u64 {
                        let expected =
                            if (3..=4).contains(&x) && (3..=4).contains(&y) && (3..=4).contains(&z)
                            {
                                ALIVE
                            } else {
                                0
                            };
                        assert_eq!(
                            world.get(x, y, z),
                            expected,
                            "cell ({},{},{}) has wrong state at gen {}",
                            x,
                            y,
                            z,
                            gen
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // 5. Determinism: same initial state + same rule -> byte-equal result.
    // -----------------------------------------------------------------
    #[test]
    fn step_flat_is_deterministic() {
        let rule = GameOfLife3D::crystal();

        let mut world_a = empty_world();
        let mut world_b = empty_world();

        let seeds: &[(u64, u64, u64)] = &[(4, 4, 4), (3, 4, 5), (5, 2, 4)];
        for &(x, y, z) in seeds {
            world_a.set(x, y, z, ALIVE);
            world_b.set(x, y, z, ALIVE);
        }

        for _ in 0..3 {
            world_a.step_flat(&rule);
            world_b.step_flat(&rule);
        }

        let flat_a = world_a.flatten();
        let flat_b = world_b.flatten();
        assert_eq!(
            flat_a, flat_b,
            "step_flat must produce identical output for identical input"
        );
        assert_eq!(
            world_a.population(),
            world_b.population(),
            "populations must match after identical evolution"
        );
    }

    // -----------------------------------------------------------------
    // 6. Corner-placed single cell still dies under amoeba.
    // -----------------------------------------------------------------
    #[test]
    fn corner_single_cell_dies_under_amoeba() {
        let mut world = empty_world();
        let side = world.side() as u64;
        world.set(side - 1, side - 1, side - 1, ALIVE);
        assert_eq!(world.population(), 1);

        let rule = GameOfLife3D::amoeba();
        world.step_flat(&rule);

        assert_eq!(
            world.population(),
            0,
            "corner live cell must die under amoeba (0 neighbors)"
        );
    }

    // -----------------------------------------------------------------
    // 7. Regression for hash-thing-2o5: seed_center with radius > center.
    // -----------------------------------------------------------------
    #[test]
    fn seed_center_radius_larger_than_center_does_not_underflow() {
        let mut w = World::new(4);
        w.seed_center(12, 0.35);
        assert!(
            w.population() > 0,
            "clamped seed region should still populate some cells"
        );
    }

    // -----------------------------------------------------------------
    // 8. Normal-radius seed_center produces population.
    // -----------------------------------------------------------------
    #[test]
    fn seed_center_normal_radius_produces_population() {
        let mut w = World::new(6);
        w.seed_center(12, 0.35);
        assert!(w.population() > 0);
    }

    // -----------------------------------------------------------------
    // 9-11. Bounds check on World::set / World::get (hash-thing-fb5).
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn world_set_panics_oob() {
        let mut w = World::new(2); // side 4
        w.set(4, 0, 0, ALIVE);
    }

    #[test]
    fn world_set_in_bounds_at_max_corner() {
        let mut w = World::new(2); // side 4
        w.set(3, 3, 3, ALIVE);
        assert_eq!(w.get(3, 3, 3), ALIVE);
        // And the complementary corner — distinct material id 2, metadata 0.
        let mat2 = Cell::pack(2, 0).raw();
        w.set(0, 0, 0, mat2);
        assert_eq!(w.get(0, 0, 0), mat2);
    }

    #[test]
    fn world_get_returns_zero_oob() {
        let w = World::new(2); // side 4
        assert_eq!(w.get(4, 0, 0), 0);
        assert_eq!(w.get(0, 4, 0), 0);
        assert_eq!(w.get(0, 0, 4), 0);
        assert_eq!(w.get(u64::MAX, 0, 0), 0);
    }

    // -----------------------------------------------------------------
    // 12. seed_terrain epoch boundary test.
    // -----------------------------------------------------------------
    #[test]
    fn seed_terrain_is_an_epoch_boundary() {
        let mut world = World::new(6);
        let params = TerrainParams::default();

        let _ = world.seed_terrain(&params);
        let nodes_after_first = world.store.stats().0;

        let _ = world.seed_terrain(&params);
        let nodes_after_second = world.store.stats().0;

        assert_eq!(
            nodes_after_first, nodes_after_second,
            "seed_terrain must start a fresh epoch; node count drifted \
             from {nodes_after_first} to {nodes_after_second} across a \
             deterministic re-seed",
        );
    }
}
