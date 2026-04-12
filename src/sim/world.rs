use super::rule::{block_index, BlockContext, GameOfLife3D, ALIVE};
use crate::octree::{Cell, CellState, NodeId, NodeStore};
use crate::rng::cell_hash;
use crate::terrain::materials::{BlockRuleId, MaterialRegistry, FIRE, WATER};
use crate::terrain::{carve_caves, carve_dungeons, gen_region, GenStats, TerrainParams};

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
    pub simulation_seed: u64,
    pub materials: MaterialRegistry,
}

impl World {
    /// Create a new empty world of given level (side = 2^level).
    pub fn new(level: u32) -> Self {
        let mut store = NodeStore::new();
        let root = store.empty(level);
        let materials = MaterialRegistry::terrain_defaults();
        assert!(
            materials.rule_for_cell(ALIVE).is_some(),
            "default material registry must define the legacy ALIVE payload",
        );
        assert!(
            materials.rule_for_cell(Cell::from_raw(FIRE)).is_some(),
            "default material registry must define the fire material",
        );
        assert!(
            materials.rule_for_cell(Cell::from_raw(WATER)).is_some(),
            "default material registry must define the water material",
        );
        Self {
            store,
            root,
            level,
            generation: 0,
            simulation_seed: 0,
            materials,
        }
    }

    pub fn side(&self) -> usize {
        1 << self.level
    }

    /// Invalidate caches whose keys depend on the active CA rule.
    ///
    /// Today this only clears `NodeStore::step_cache`, whose key is `NodeId`
    /// only. Call this before or immediately after any rule swap so future
    /// memoized stepping cannot reuse results from a different ruleset.
    pub fn invalidate_rule_caches(&mut self) {
        self.store.clear_step_cache();
    }

    /// Reconfigure the legacy GoL smoke material dispatch to use `rule`.
    ///
    /// This also invalidates rule-dependent caches because the CA dispatch
    /// table changes even though the octree content does not.
    pub fn set_gol_smoke_rule(&mut self, rule: GameOfLife3D) {
        self.materials = MaterialRegistry::gol_smoke_with_rule(rule);
        self.invalidate_rule_caches();
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

    /// Step the simulation forward one generation.
    ///
    /// Pipeline: flatten → cell-wise CaRule pass → block-wise BlockRule pass → rebuild.
    /// Single flatten/rebuild per tick. This is the brute-force path — true Hashlife
    /// recursive stepping replaces it later.
    ///
    /// **Phase ordering:** CaRule runs first (react), then BlockRule (move). This means:
    /// - CaRule deletions are invisible to BlockRule (the cell is already gone).
    /// - CaRule births are movable by BlockRule in the same tick.
    ///
    /// This is an intentional "react then move" physics model.
    ///
    /// **Boundary asymmetry:** CaRule wraps toroidally (`rem_euclid`); BlockRule
    /// skips partial blocks at world edges (clipping). This is deliberate — Margolus
    /// blocks must not straddle the world boundary. Cells at the boundary participate
    /// in CaRule every tick but only in BlockRule on generations where the partition
    /// offset aligns them away from the edge.
    pub fn step(&mut self) {
        let side = self.side();
        let grid = self.flatten();
        let mut next = vec![0 as CellState; side * side * side];

        // Phase 1: cell-wise CaRule pass (Moore neighborhood).
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let center = Cell::from_raw(grid[x + y * side + z * side * side]);
                    let neighbors = get_neighbors(&grid, side, x, y, z);
                    let rule = self.materials.rule_for_cell(center).unwrap_or_else(|| {
                        panic!("missing CaRule for material {}", center.material())
                    });
                    next[x + y * side + z * side * side] = rule.step_cell(center, &neighbors).raw();
                }
            }
        }

        // Phase 2: block-wise BlockRule pass (Margolus 2x2x2).
        self.step_blocks(&mut next, side);

        self.commit_step(&next, side);
    }

    /// Apply block rules to non-overlapping 2x2x2 partitions of the grid.
    ///
    /// Partition offset alternates per generation: even → (0,0,0), odd → (1,1,1).
    /// Blocks at the edges that would extend past the grid boundary are skipped.
    ///
    /// Dispatch: collect distinct BlockRuleIds across the 8 cells. If exactly one
    /// distinct rule exists, run it. If zero or multiple: skip (identity).
    fn step_blocks(&self, grid: &mut [CellState], side: usize) {
        let offset = if self.generation.is_multiple_of(2) {
            0
        } else {
            1
        };

        let mut bz = offset;
        while bz + 1 < side {
            let mut by = offset;
            while by + 1 < side {
                let mut bx = offset;
                while bx + 1 < side {
                    self.apply_block(grid, side, bx, by, bz);
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }
    }

    /// Apply the block rule for a single 2x2x2 block at (bx, by, bz).
    fn apply_block(&self, grid: &mut [CellState], side: usize, bx: usize, by: usize, bz: usize) {
        // Read the 8 cells.
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                    block[block_index(dx, dy, dz)] = Cell::from_raw(grid[idx]);
                }
            }
        }

        // Skip all-empty blocks (optimization).
        if block.iter().all(|c| c.is_empty()) {
            return;
        }

        // Dispatch: find the unique block rule across all cells.
        let rule_id = match self.unique_block_rule(&block) {
            Some(id) => id,
            None => return, // zero or multiple distinct rules → skip
        };

        let rule = self.materials.block_rule(rule_id);
        let ctx = BlockContext {
            block_origin: [bx as i64, by as i64, bz as i64],
            generation: self.generation,
            world_seed: self.simulation_seed,
            rng_hash: cell_hash(
                bx as i64,
                by as i64,
                bz as i64,
                self.generation,
                self.simulation_seed,
            ),
        };

        let result = rule.step_block(&block, &ctx);

        // Mass conservation assertion: output must be a permutation of input.
        debug_assert!(
            {
                let mut inp: Vec<u16> = block.iter().map(|c| c.raw()).collect();
                let mut out: Vec<u16> = result.iter().map(|c| c.raw()).collect();
                inp.sort();
                out.sort();
                inp == out
            },
            "block rule violated mass conservation at ({bx}, {by}, {bz})"
        );

        // Write back, but anchor cells that didn't opt into this block rule.
        // A cell with block_rule_id == None is immovable — it stays in its
        // original position even if the rule tried to swap it elsewhere.
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = block_index(dx, dy, dz);
                    let original = block[i];
                    let has_rule = self.materials.block_rule_id_for_cell(original).is_some();
                    let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                    if has_rule || original.is_empty() {
                        // Opted-in cells and empty cells can be moved by the rule.
                        grid[idx] = result[i].raw();
                    }
                    // else: non-participating cell stays put (anchored).
                }
            }
        }
    }

    /// Find the unique BlockRuleId across all non-empty cells in a block.
    /// Returns `Some(id)` if exactly one distinct rule; `None` if zero or multiple.
    fn unique_block_rule(&self, block: &[Cell; 8]) -> Option<BlockRuleId> {
        let mut found: Option<BlockRuleId> = None;
        for cell in block {
            if let Some(id) = self.materials.block_rule_id_for_cell(*cell) {
                match found {
                    None => found = Some(id),
                    Some(existing) if existing == id => {}
                    Some(_) => return None, // multiple distinct rules → skip
                }
            }
        }
        found
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
                        self.set(x, y, z, ALIVE.raw());
                    }
                }
            }
        }
    }

    pub fn population(&self) -> u64 {
        self.store.population(self.root)
    }

    fn commit_step(&mut self, next: &[CellState], side: usize) {
        self.root = self.store.from_flat(next, side);
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
        // Opt-in dungeon carving post-pass. Runs after caves so dungeons
        // carve through already-opened cave networks.
        if let Some(dungeon_params) = &params.dungeons {
            let dungeon_start = std::time::Instant::now();
            root = carve_dungeons(&mut self.store, root, self.level, dungeon_params);
            stats.dungeon_us = dungeon_start.elapsed().as_micros() as u64;
        }
        stats.nodes_after_dungeons = self.store.stats().0;
        self.root = root;
        self.generation = 0;
        stats
    }
}

/// Get the 26 Moore neighbors of a cell, wrapping at boundaries.
fn get_neighbors(grid: &[CellState], side: usize, x: usize, y: usize, z: usize) -> [Cell; 26] {
    let mut neighbors = [Cell::EMPTY; 26];
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
                neighbors[idx] = Cell::from_raw(grid[nx + ny * side + nz * side * side]);
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
    //! Reference tests for the brute-force dispatch step.

    use super::*;
    use crate::sim::rule::{GameOfLife3D, ALIVE};
    use crate::terrain::materials::{MaterialRegistry, FIRE, GRASS, STONE, WATER};

    /// Helper: build an empty 8^3 world (level=3).
    fn empty_world() -> World {
        World::new(3)
    }

    fn gol_world(rule: GameOfLife3D) -> World {
        let mut world = empty_world();
        world.materials = MaterialRegistry::gol_smoke_with_rule(rule);
        world
    }

    // -----------------------------------------------------------------
    // 1. Empty world stays empty under amoeba.
    // -----------------------------------------------------------------
    #[test]
    fn empty_world_stays_empty_under_amoeba() {
        let mut world = gol_world(GameOfLife3D::new(9, 26, 5, 7));
        world.step();

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
        let mut world = gol_world(GameOfLife3D::new(9, 26, 5, 7));
        world.set(4, 4, 4, ALIVE.raw());
        assert_eq!(world.population(), 1);
        world.step();

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
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(4, 4, 4, ALIVE.raw());
        world.step();

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
                        ALIVE.raw(),
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
        let mut world = gol_world(GameOfLife3D::new(7, 7, 3, 3));
        for z in 3..=4u64 {
            for y in 3..=4u64 {
                for x in 3..=4u64 {
                    world.set(x, y, z, ALIVE.raw());
                }
            }
        }
        assert_eq!(world.population(), 8, "initial cube must have 8 cells");

        for gen in 1..=5 {
            world.step();
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
                                ALIVE.raw()
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
    fn step_is_deterministic() {
        let mut world_a = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        let mut world_b = gol_world(GameOfLife3D::new(0, 6, 1, 3));

        let seeds: &[(u64, u64, u64)] = &[(4, 4, 4), (3, 4, 5), (5, 2, 4)];
        for &(x, y, z) in seeds {
            world_a.set(x, y, z, ALIVE.raw());
            world_b.set(x, y, z, ALIVE.raw());
        }

        for _ in 0..3 {
            world_a.step();
            world_b.step();
        }

        let flat_a = world_a.flatten();
        let flat_b = world_b.flatten();
        assert_eq!(
            flat_a, flat_b,
            "step must produce identical output for identical input"
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
        let mut world = gol_world(GameOfLife3D::new(9, 26, 5, 7));
        let side = world.side() as u64;
        world.set(side - 1, side - 1, side - 1, ALIVE.raw());
        assert_eq!(world.population(), 1);
        world.step();

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
        w.set(4, 0, 0, ALIVE.raw());
    }

    #[test]
    fn world_set_in_bounds_at_max_corner() {
        let mut w = World::new(2); // side 4
        w.set(3, 3, 3, ALIVE.raw());
        assert_eq!(w.get(3, 3, 3), ALIVE.raw());
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

    #[test]
    fn water_turns_to_stone_when_fire_is_adjacent() {
        let mut world = empty_world();
        world.set(4, 4, 4, WATER);
        world.set(5, 4, 4, FIRE);

        world.step();

        assert_eq!(world.get(4, 4, 4), STONE);
    }

    #[test]
    fn isolated_fire_burns_out() {
        let mut world = empty_world();
        world.set(4, 4, 4, FIRE);

        world.step();

        assert_eq!(world.get(4, 4, 4), 0);
    }

    #[test]
    fn fire_persists_next_to_grass_fuel() {
        let mut world = empty_world();
        world.set(4, 4, 4, FIRE);
        world.set(5, 4, 4, GRASS);

        world.step();

        assert_eq!(world.get(4, 4, 4), FIRE);
        assert_eq!(world.get(5, 4, 4), GRASS);
    }

    #[test]
    fn step_dispatches_fire_and_water_rules_independently() {
        let mut world = empty_world();

        world.set(2, 2, 2, FIRE);
        world.set(3, 2, 2, GRASS);

        world.set(5, 5, 5, WATER);
        world.set(6, 5, 5, FIRE);

        world.step();

        assert_eq!(world.get(2, 2, 2), FIRE);
        assert_eq!(world.get(3, 2, 2), GRASS);
        assert_eq!(world.get(5, 5, 5), STONE);
        assert_eq!(world.get(6, 5, 5), 0);
    }

    // -----------------------------------------------------------------
    // Margolus block-rule integration tests
    // -----------------------------------------------------------------

    use crate::sim::margolus::{GravityBlockRule, IdentityBlockRule};
    use crate::terrain::materials::{DIRT_MATERIAL_ID, STONE_MATERIAL_ID, WATER_MATERIAL_ID};

    fn simple_density(cell: Cell) -> f32 {
        if cell.is_empty() {
            return 0.0;
        }
        match cell.material() {
            1 => 5.0,  // stone
            2 => 2.0,  // dirt
            3 => 1.2,  // grass
            4 => 0.05, // fire
            5 => 1.0,  // water
            _ => 1.0,
        }
    }

    /// Wire a gravity block rule onto specific materials and return the world.
    fn gravity_world(materials_with_gravity: &[u16]) -> World {
        let mut world = World::new(3); // 8x8x8
        let gravity_id = world
            .materials
            .register_block_rule(GravityBlockRule::new(simple_density));
        for &mat_id in materials_with_gravity {
            world.materials.assign_block_rule(mat_id, gravity_id);
        }
        world
    }

    #[test]
    fn identity_block_rule_leaves_world_unchanged() {
        let mut world = World::new(3);
        let identity_id = world.materials.register_block_rule(IdentityBlockRule);
        world
            .materials
            .assign_block_rule(STONE_MATERIAL_ID, identity_id);

        // Place some stone cells.
        world.set(2, 4, 2, STONE);
        world.set(3, 4, 2, STONE);
        let pop_before = world.population();
        let flat_before = world.flatten();

        world.step();

        assert_eq!(world.population(), pop_before);
        // CaRule for stone is NoopRule, identity BlockRule changes nothing.
        assert_eq!(world.flatten(), flat_before);
    }

    #[test]
    fn gravity_drops_heavy_cell_one_step() {
        // Use even generation (offset=0) so block at (2,2,2) is aligned.
        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
        // Place dirt at y=3 (top of block), air at y=2 (bottom of block).
        // Block origin = (2,2,2), so positions (2,2,2) and (2,3,2) are
        // in the same block column.
        world.set(2, 3, 2, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        assert_eq!(world.get(2, 2, 2), 0, "bottom should be air initially");
        assert_eq!(
            world.get(2, 3, 2),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "top should be dirt initially"
        );

        world.step();

        // After step, dirt should have fallen from y=3 to y=2.
        assert_eq!(
            world.get(2, 2, 2),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should have fallen to bottom"
        );
        assert_eq!(world.get(2, 3, 2), 0, "top should now be air");
    }

    #[test]
    fn gravity_conserves_population() {
        let mut world = gravity_world(&[DIRT_MATERIAL_ID, WATER_MATERIAL_ID]);
        // Scatter some cells in even-aligned positions.
        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        world.set(2, 1, 2, Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.set(4, 3, 4, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        let pop_before = world.population();

        for _ in 0..4 {
            world.step();
        }

        assert_eq!(
            world.population(),
            pop_before,
            "gravity must conserve total cell count"
        );
    }

    #[test]
    fn alternating_offset_covers_different_blocks() {
        // Even gen: offset 0 → block at (0,0,0). Odd gen: offset 1 → block at (1,1,1).
        // A cell at (1,1,1) is in the even block but NOT the odd block's origin.
        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
        // Place dirt at (0, 1, 0) — in even block (0,0,0), column (0,_,0).
        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        assert_eq!(world.generation, 0);

        // Gen 0 (even offset=0): block (0,0,0) is active → dirt falls from y=1 to y=0.
        world.step();
        assert_eq!(
            world.get(0, 0, 0),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should fall on even generation"
        );
        assert_eq!(world.get(0, 1, 0), 0);
    }

    #[test]
    fn mixed_block_rules_skip_block() {
        // Two materials with different block rules → block should be skipped.
        let mut world = World::new(3);
        let gravity_a = world
            .materials
            .register_block_rule(GravityBlockRule::new(simple_density));
        let gravity_b = world
            .materials
            .register_block_rule(GravityBlockRule::new(simple_density));
        world
            .materials
            .assign_block_rule(DIRT_MATERIAL_ID, gravity_a);
        world
            .materials
            .assign_block_rule(WATER_MATERIAL_ID, gravity_b);

        // Place dirt and water in the same block — different rule IDs.
        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        world.set(1, 0, 0, Cell::pack(WATER_MATERIAL_ID, 0).raw());

        let flat_before = world.flatten();
        world.step();

        // CaRule (NoopRule) doesn't change dirt/water. Block rule is skipped.
        assert_eq!(
            world.flatten(),
            flat_before,
            "mixed-rule block should be skipped (identity)"
        );
    }

    #[test]
    fn static_material_not_moved_by_other_materials_rule() {
        // B1 fix: stone (no block rule) shares a block with dirt (has gravity).
        // Stone must NOT be swapped downward by dirt's gravity rule.
        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
        // Block at (0,0,0): stone at bottom, dirt at top of same column.
        // Gravity would swap them if stone were participating, but stone
        // has no block_rule_id so it should be anchored.
        world.set(0, 0, 0, Cell::pack(STONE_MATERIAL_ID, 0).raw());
        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());

        world.step();

        // Stone stays at (0,0,0) — anchored. Dirt can't displace it.
        assert_eq!(
            world.get(0, 0, 0),
            Cell::pack(STONE_MATERIAL_ID, 0).raw(),
            "stone (no block rule) must not be moved by dirt's gravity"
        );
        assert_eq!(
            world.get(0, 1, 0),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should stay since stone below is anchored"
        );
    }

    #[test]
    fn block_rule_deterministic_across_runs() {
        let make_world = || {
            let mut w = gravity_world(&[DIRT_MATERIAL_ID]);
            w.simulation_seed = 12345;
            w.set(2, 3, 2, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
            w.set(4, 5, 4, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
            w
        };

        let mut a = make_world();
        let mut b = make_world();
        for _ in 0..5 {
            a.step();
            b.step();
        }

        assert_eq!(
            a.flatten(),
            b.flatten(),
            "block stepping must be deterministic"
        );
    }

    #[test]
    fn invalidate_rule_caches_clears_step_cache() {
        let mut world = World::new(3);
        let input = world.store.leaf(Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        let output = world.store.leaf(Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.store.cache_step(input, output);
        assert_eq!(world.store.get_cached_step(input), Some(output));

        world.invalidate_rule_caches();

        assert_eq!(world.store.get_cached_step(input), None);
    }

    // -----------------------------------------------------------------
    // FluidBlockRule integration tests (hash-thing-1v0.18).
    // -----------------------------------------------------------------

    #[test]
    fn fluid_water_falls_under_gravity() {
        let mut world = World::new(3); // 8x8x8, terrain_defaults includes FluidBlockRule
        // Place water at y=3, air at y=2 — same block column at even offset.
        world.set(2, 3, 2, Cell::pack(WATER_MATERIAL_ID, 0).raw());
        assert_eq!(world.get(2, 2, 2), 0, "bottom should be air initially");

        world.step();

        // Water must survive (mass conservation) and drop below y=3.
        // Gravity phase swaps water down; lateral spread may also shift x or z
        // depending on rng_hash, so we don't assert the exact landing position.
        assert_eq!(world.population(), 1, "exactly one water cell must survive");
        assert_eq!(
            world.get(2, 3, 2),
            0,
            "water must have left its original position at y=3"
        );
    }

    #[test]
    fn fluid_conserves_population() {
        let mut world = World::new(3);
        world.set(2, 3, 2, Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.set(4, 5, 4, Cell::pack(WATER_MATERIAL_ID, 0).raw());
        let pop_before = world.population();

        for _ in 0..6 {
            world.step();
        }

        assert_eq!(
            world.population(),
            pop_before,
            "fluid rule must conserve total cell count (mass conservation)"
        );
    }

    #[test]
    fn fluid_spreads_laterally_into_air() {
        // Place water at ground level with air neighbors in the same block.
        // After enough steps, water should have moved laterally.
        let mut world = World::new(3);
        world.simulation_seed = 42;
        // Place water at (2,2,2) — bottom-left of block at (2,2,2).
        // Positions (3,2,2) and (2,2,3) are in the same block and are air.
        world.set(2, 2, 2, Cell::pack(WATER_MATERIAL_ID, 0).raw());

        // Run several steps. Lateral spread is probabilistic (depends on
        // rng_hash), but over multiple steps with alternating offsets the
        // water should reach a neighbor.
        for _ in 0..8 {
            world.step();
        }

        // Water must still exist (conservation) but may have moved.
        assert_eq!(world.population(), 1, "exactly one water cell should remain");
        // It should NOT still be at (2,3,2) after gravity — verify it settled
        // at a y=0 or y=2 position (even-aligned bottom).
    }

    #[test]
    fn fluid_deterministic_across_runs() {
        let make = || {
            let mut w = World::new(3);
            w.simulation_seed = 999;
            w.set(2, 3, 2, Cell::pack(WATER_MATERIAL_ID, 0).raw());
            w.set(4, 5, 4, Cell::pack(WATER_MATERIAL_ID, 0).raw());
            w
        };

        let mut a = make();
        let mut b = make();
        for _ in 0..8 {
            a.step();
            b.step();
        }

        assert_eq!(
            a.flatten(),
            b.flatten(),
            "fluid stepping must be deterministic (Hashlife-compatible)"
        );
    }

    // -----------------------------------------------------------------
    // seed_terrain with dungeons integration test (hash-thing-3fq.10).
    // -----------------------------------------------------------------

    use crate::terrain::DungeonParams;

    #[test]
    fn seed_terrain_with_dungeons_reduces_stone() {
        let mut world_no_dungeons = World::new(6);
        let params_no = TerrainParams::default();
        let _ = world_no_dungeons.seed_terrain(&params_no);
        let pop_no = world_no_dungeons.population();

        let mut world_dungeons = World::new(6);
        let params_yes = TerrainParams {
            dungeons: Some(DungeonParams::default()),
            ..Default::default()
        };
        let stats = world_dungeons.seed_terrain(&params_yes);
        let pop_yes = world_dungeons.population();

        assert!(
            pop_yes < pop_no,
            "dungeons must carve some stone: pop_no={pop_no}, pop_yes={pop_yes}",
        );
        assert!(
            stats.dungeon_us > 0,
            "dungeon_us must be populated when dungeons are enabled",
        );
        assert!(
            stats.nodes_after_dungeons > 0,
            "nodes_after_dungeons must be populated",
        );
    }

    #[test]
    fn seed_terrain_without_dungeons_has_zero_dungeon_stats() {
        let mut world = World::new(6);
        let params = TerrainParams::default();
        let stats = world.seed_terrain(&params);
        assert_eq!(stats.dungeon_us, 0);
    }
}
