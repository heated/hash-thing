//! Recursive Hashlife stepper (hash-thing-6gf.1).
//!
//! Computes one simulation step directly on the octree, without flattening the
//! entire grid. The result of stepping a level-n node (2^n × 2^n × 2^n) is a
//! level-(n-1) node representing the center (2^(n-1))³ cube after one generation.
//!
//! **Phase ordering** matches the brute-force path: CaRule first (react), then
//! BlockRule (move). Boundary semantics: CaRule wraps toroidally within the
//! input node; BlockRule clips at edges.
//!
//! **Base case (level 2):** A 4×4×4 input. Flatten, step the center 2×2×2
//! cells through CaRule + BlockRule, rebuild as level-1.
//!
//! **Recursive case (level ≥ 3):** Two phases:
//!   1. *Center* — build 27 intermediate nodes (level n-1) from grandchildren,
//!      then extract each center (level n-2) without time advance.
//!   2. *Step* — group the 27 centered nodes into 8 overlapping sub-cubes
//!      (each level n-1), recursively step each → 8 results at level n-2.
//!      Assemble into the level-(n-1) output.
//!
//! Step results are memoized by (NodeId, origin) so identical subtrees at
//! the same world-space position are computed only once per generation.
//! The cache is cleared after each step (generation changes affect BlockRule
//! rng_hash, and compaction remaps NodeIds).

use super::rule::BlockContext;
use super::world::World;
use crate::octree::node::octant_index;
use crate::octree::{Cell, CellState, NodeId};
use crate::rng::cell_hash;

impl World {
    /// Step the world forward one generation using the recursive Hashlife path.
    pub fn step_recursive(&mut self) {
        assert!(
            self.level >= 3,
            "step_recursive requires level >= 3, got {}",
            self.level
        );
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        // World-space origin of the padded root: the original world is
        // centered, so it starts at 2^(level-1) from the padded root's origin.
        let quarter = 1u64 << (self.level - 1);
        let origin = [-(quarter as i64); 3];
        let parity = (self.generation % 2) as u32;
        let result = self.step_node(padded_root, padded_level, origin, parity);
        self.root = result;
        self.generation += 1;

        self.hashlife_cache.clear();
        self.store.clear_step_cache();
        let (new_store, new_root) = self.store.compacted(self.root);
        self.store = new_store;
        self.root = new_root;
    }

    /// Wrap the current root in a one-level-larger node, padding with empty.
    fn pad_root(&mut self) -> NodeId {
        let empty_child = self.store.empty(self.level - 1);
        let root_children = self.store.children(self.root);

        let mut padded_children = [NodeId::EMPTY; 8];
        for oct in 0..8 {
            let mirror = 7 - oct;
            let mut sub = [empty_child; 8];
            sub[mirror] = root_children[oct];
            padded_children[oct] = self.store.interior(self.level, sub);
        }
        self.store.interior(self.level + 1, padded_children)
    }

    /// Recursively step a node. Input level n (≥ 3), output level n-1.
    /// `origin` is the world-space coordinate of the node's (0,0,0) corner.
    fn step_node(&mut self, node: NodeId, level: u32, origin: [i64; 3], parity: u32) -> NodeId {
        debug_assert!(level >= 3, "step_node requires level >= 3, got {level}");

        let key = (node, origin, parity);
        if let Some(&cached) = self.hashlife_cache.get(&key) {
            return cached;
        }

        let result = if level == 3 {
            self.step_base_case(node, origin, parity)
        } else {
            self.step_recursive_case(node, level, origin, parity)
        };

        self.hashlife_cache.insert(key, result);
        result
    }

    /// Base case: level-3 node (8×8×8). Flatten, run CaRule on interior 6³,
    /// run BlockRule on all aligned blocks, extract center 4³ → level-2 output.
    fn step_base_case(&mut self, node: NodeId, origin: [i64; 3], parity: u32) -> NodeId {
        let side = 8usize;
        let grid = self.store.flatten(node, side);

        // Phase 1: CaRule on interior cells (1..7 on each axis).
        // The outermost ring (0 and 7) can't have correct CaRule results because
        // their neighbors would wrap incorrectly. But we only need the center
        // 4×4×4 (positions 2..6) and the 1-cell border for BlockRule (1..7),
        // so running CaRule on 1..7 suffices.
        let mut next = vec![0 as CellState; side * side * side];
        for z in 1..side - 1 {
            for y in 1..side - 1 {
                for x in 1..side - 1 {
                    let cell = Cell::from_raw(grid[x + y * side + z * side * side]);
                    let neighbors = get_neighbors_from_grid(&grid, side, x, y, z);
                    let rule = self.materials.rule_for_cell(cell).unwrap_or_else(|| {
                        panic!("missing CaRule for material {}", cell.material())
                    });
                    next[x + y * side + z * side * side] = rule.step_cell(cell, &neighbors).raw();
                }
            }
        }

        // Phase 2: BlockRule on all aligned 2×2×2 blocks within the interior.
        let offset = parity as usize;
        // Iterate blocks whose origins are in [1, side-2] so they stay within
        // the CaRule-valid interior.
        let start = offset;
        let mut bz = start;
        while bz + 1 < side - 1 {
            let mut by = start;
            while by + 1 < side - 1 {
                let mut bx = start;
                while bx + 1 < side - 1 {
                    // World-space block origin.
                    let wbx = origin[0] + bx as i64;
                    let wby = origin[1] + by as i64;
                    let wbz = origin[2] + bz as i64;
                    // Only fire if aligned with the generation's offset.
                    if wbx.rem_euclid(2) == offset as i64
                        && wby.rem_euclid(2) == offset as i64
                        && wbz.rem_euclid(2) == offset as i64
                    {
                        self.apply_block_in_grid(&mut next, side, bx, by, bz, wbx, wby, wbz);
                    }
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }

        // Extract center 4×4×4 (positions 2..6) → level-2 node.
        let center_side = 4usize;
        let mut center_grid = vec![0 as CellState; center_side * center_side * center_side];
        for cz in 0..center_side {
            for cy in 0..center_side {
                for cx in 0..center_side {
                    center_grid[cx + cy * center_side + cz * center_side * center_side] =
                        next[(cx + 2) + (cy + 2) * side + (cz + 2) * side * side];
                }
            }
        }
        self.store.from_flat(&center_grid, center_side)
    }

    /// Apply a single block rule within a flat grid.
    #[allow(clippy::too_many_arguments)]
    fn apply_block_in_grid(
        &self,
        grid: &mut [CellState],
        side: usize,
        bx: usize,
        by: usize,
        bz: usize,
        wbx: i64,
        wby: i64,
        wbz: i64,
    ) {
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                    block[octant_index(dx as u32, dy as u32, dz as u32)] =
                        Cell::from_raw(grid[idx]);
                }
            }
        }

        if block.iter().all(|c| c.is_empty()) {
            return;
        }

        let rule_id = match self.unique_block_rule(&block) {
            Some(id) => id,
            None => return,
        };

        let rule = self.materials.block_rule(rule_id);
        let ctx = BlockContext {
            block_origin: [wbx, wby, wbz],
            generation: self.generation,
            world_seed: self.simulation_seed,
            rng_hash: cell_hash(wbx, wby, wbz, self.generation, self.simulation_seed),
        };

        let result = rule.step_block(&block, &ctx);

        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = octant_index(dx as u32, dy as u32, dz as u32);
                    let original = block[i];
                    let has_rule = self.materials.block_rule_id_for_cell(original).is_some();
                    let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                    if has_rule || original.is_empty() {
                        grid[idx] = result[i].raw();
                    }
                }
            }
        }
    }

    /// Recursive case: level ≥ 3.
    fn step_recursive_case(
        &mut self,
        node: NodeId,
        level: u32,
        origin: [i64; 3],
        parity: u32,
    ) -> NodeId {
        let children = self.store.children(node);
        let sub: [[NodeId; 8]; 8] = std::array::from_fn(|i| self.store.children(children[i]));

        // Build 3×3×3 = 27 intermediate nodes at level (n-1).
        let mut inter = [NodeId::EMPTY; 27];
        for pz in 0..3usize {
            for py in 0..3usize {
                for px in 0..3usize {
                    let mut octants = [NodeId::EMPTY; 8];
                    for sz in 0..2usize {
                        for sy in 0..2usize {
                            for sx in 0..2usize {
                                let (parent_x, sub_x) = source_index(px, sx);
                                let (parent_y, sub_y) = source_index(py, sy);
                                let (parent_z, sub_z) = source_index(pz, sz);
                                let parent_oct =
                                    octant_index(parent_x as u32, parent_y as u32, parent_z as u32);
                                let sub_oct =
                                    octant_index(sub_x as u32, sub_y as u32, sub_z as u32);
                                octants[octant_index(sx as u32, sy as u32, sz as u32)] =
                                    sub[parent_oct][sub_oct];
                            }
                        }
                    }
                    inter[px + py * 3 + pz * 9] = self.store.interior(level - 1, octants);
                }
            }
        }

        // Extract center (level n-2) of each intermediate — spatial reindex only.
        let mut centered = [NodeId::EMPTY; 27];
        for i in 0..27 {
            centered[i] = self.center_node(inter[i], level - 1);
        }

        // Group into 8 overlapping sub-cubes (level n-1) and recurse.
        let quarter = 1i64 << (level - 2); // size of a level-(n-2) node
        let half_quarter = quarter / 2; // = 2^(level-3)
        let mut result_children = [NodeId::EMPTY; 8];
        for oz in 0..2usize {
            for oy in 0..2usize {
                for ox in 0..2usize {
                    let mut sub_cube = [NodeId::EMPTY; 8];
                    for dz in 0..2usize {
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                sub_cube[octant_index(dx as u32, dy as u32, dz as u32)] =
                                    centered[(ox + dx) + (oy + dy) * 3 + (oz + dz) * 9];
                            }
                        }
                    }
                    let sub_root = self.store.interior(level - 1, sub_cube);
                    // World-space origin: centered[0] starts at origin + half_quarter.
                    // Each subsequent centered node adds quarter.
                    let sub_origin = [
                        origin[0] + half_quarter + (ox as i64) * quarter,
                        origin[1] + half_quarter + (oy as i64) * quarter,
                        origin[2] + half_quarter + (oz as i64) * quarter,
                    ];
                    result_children[octant_index(ox as u32, oy as u32, oz as u32)] =
                        self.step_node(sub_root, level - 1, sub_origin, parity);
                }
            }
        }

        self.store.interior(level - 1, result_children)
    }

    /// Extract the center (level n-1) of a level-n node.
    fn center_node(&mut self, node: NodeId, level: u32) -> NodeId {
        debug_assert!(level >= 2, "center_node requires level >= 2");
        let children = self.store.children(node);
        let mut center_children = [NodeId::EMPTY; 8];
        for oct in 0..8usize {
            let inner = 7 - oct;
            center_children[oct] = self.store.child(children[oct], inner);
        }
        self.store.interior(level - 1, center_children)
    }
}

/// Map intermediate position p ∈ {0,1,2} and sub-octant s ∈ {0,1} to
/// (parent_child_axis_index, sub_octant_axis_index).
#[inline]
fn source_index(p: usize, s: usize) -> (usize, usize) {
    match (p, s) {
        (0, 0) => (0, 0),
        (0, 1) => (0, 1),
        (1, 0) => (0, 1),
        (1, 1) => (1, 0),
        (2, 0) => (1, 0),
        (2, 1) => (1, 1),
        _ => unreachable!(),
    }
}

fn get_neighbors_from_grid(
    grid: &[CellState],
    side: usize,
    x: usize,
    y: usize,
    z: usize,
) -> [Cell; 26] {
    let mut neighbors = [Cell::EMPTY; 26];
    let mut idx = 0;
    let s = side as i64;
    for dz in [-1i64, 0, 1] {
        for dy in [-1i64, 0, 1] {
            for dx in [-1i64, 0, 1] {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = (x as i64 + dx).rem_euclid(s) as usize;
                let ny = (y as i64 + dy).rem_euclid(s) as usize;
                let nz = (z as i64 + dz).rem_euclid(s) as usize;
                neighbors[idx] = Cell::from_raw(grid[nx + ny * side + nz * side * side]);
                idx += 1;
            }
        }
    }
    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::rule::ALIVE;
    use crate::sim::{GameOfLife3D, WorldCoord};
    use crate::terrain::materials::{DIRT_MATERIAL_ID, STONE, WATER_MATERIAL_ID};

    fn wc(coord: u64) -> WorldCoord {
        WorldCoord(coord as i64)
    }

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
    }

    fn gol_world(level: u32, rule: GameOfLife3D, simulation_seed: u64) -> World {
        let mut world = World::new(level);
        world.simulation_seed = simulation_seed;
        world.set_gol_smoke_rule(rule);
        world
    }

    fn seed_random_alive_cells(world: &mut World, seed: u64, margin: u64) {
        let mut rng = SimpleRng::new(seed);
        let side = world.side() as u64;
        for z in margin..(side - margin) {
            for y in margin..(side - margin) {
                for x in margin..(side - margin) {
                    if rng.next_u64().is_multiple_of(5) {
                        world.set(wc(x), wc(y), wc(z), ALIVE.raw());
                    }
                }
            }
        }
    }

    fn assert_recursive_matches_bruteforce(
        mut brute: World,
        mut recur: World,
        steps: usize,
        label: &str,
    ) {
        assert_eq!(
            brute.flatten(),
            recur.flatten(),
            "{label}: initial state mismatch"
        );
        for step in 0..steps {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "{label}: mismatch at generation {} (step {})",
                brute.generation,
                step
            );
            assert_eq!(
                brute.generation, recur.generation,
                "{label}: generation counter diverged"
            );
        }
    }
    #[test]
    fn recursive_matches_brute_force_empty() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        brute.step();
        recur.step_recursive();
        assert_eq!(brute.flatten(), recur.flatten());
        assert_eq!(brute.generation, recur.generation);
    }

    #[test]
    fn recursive_matches_brute_force_single_cell() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        brute.set(wc(3), wc(3), wc(3), STONE);
        recur.set(wc(3), wc(3), wc(3), STONE);
        brute.step();
        recur.step_recursive();
        assert_eq!(brute.flatten(), recur.flatten());
    }

    #[test]
    fn recursive_matches_brute_force_multiple_steps() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        for &(x, y, z, mat) in &[
            (2u64, 2, 2, STONE),
            (
                3,
                3,
                3,
                crate::octree::Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            ),
            (
                4,
                4,
                4,
                crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw(),
            ),
        ] {
            brute.set(wc(x), wc(y), wc(z), mat);
            recur.set(wc(x), wc(y), wc(z), mat);
        }
        for step in 0..4 {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "mismatch at generation {} (step {})",
                brute.generation,
                step
            );
        }
    }

    #[test]
    fn recursive_matches_brute_force_carule_only() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        for &(x, y, z) in &[
            (1u64, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            (4, 4, 4),
            (5, 5, 5),
            (6, 6, 6),
        ] {
            brute.set(wc(x), wc(y), wc(z), STONE);
            recur.set(wc(x), wc(y), wc(z), STONE);
        }
        for step in 0..6 {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "CaRule-only mismatch at generation {} (step {})",
                brute.generation,
                step
            );
        }
    }

    #[test]
    fn recursive_matches_brute_force_with_terrain() {
        use crate::terrain::TerrainParams;
        let mut brute = World::new(4);
        let mut recur = World::new(4);
        let params = TerrainParams::default();
        brute.seed_terrain(&params);
        recur.seed_terrain(&params);
        assert_eq!(brute.flatten(), recur.flatten(), "initial state must match");
        for _ in 0..2 {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "mismatch at generation {}",
                brute.generation
            );
        }
    }

    #[test]
    fn recursive_matches_brute_force_randomized_gol_presets() {
        let presets = [
            ("amoeba", GameOfLife3D::new(9, 26, 5, 7)),
            ("crystal", GameOfLife3D::new(0, 6, 1, 3)),
            ("445", GameOfLife3D::rule445()),
            ("pyroclastic", GameOfLife3D::new(4, 7, 6, 8)),
        ];
        let cases = [(3_u32, 1_usize, 2_u64), (4_u32, 3_usize, 4_u64)];

        for (preset_idx, &(label, rule)) in presets.iter().enumerate() {
            for (level, steps, margin) in cases {
                for case_seed in 0..4_u64 {
                    let simulation_seed = 0x6f03_u64
                        ^ ((preset_idx as u64) << 16)
                        ^ ((level as u64) << 8)
                        ^ case_seed;
                    let initial_seed = simulation_seed ^ 0xa11ce_u64;
                    let mut brute = gol_world(level, rule, simulation_seed);
                    let mut recur = gol_world(level, rule, simulation_seed);
                    seed_random_alive_cells(&mut brute, initial_seed, margin);
                    seed_random_alive_cells(&mut recur, initial_seed, margin);

                    let case =
                        format!("{label} level={level} steps={steps} seed={simulation_seed:#x}");
                    assert_recursive_matches_bruteforce(brute, recur, steps, &case);
                }
            }
        }
    }

    #[test]
    fn recursive_conserves_population_with_stone() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), STONE);
        world.set(wc(4), wc(4), wc(4), STONE);
        let pop_before = world.population();
        world.step_recursive();
        assert_eq!(world.population(), pop_before);
    }

    #[test]
    fn pad_root_preserves_center() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(4), STONE);
        let flat_before = world.flatten();
        let padded = world.pad_root();
        let padded_side = 16usize;
        let padded_grid = world.store.flatten(padded, padded_side);
        let orig_side = 8usize;
        for z in 0..orig_side {
            for y in 0..orig_side {
                for x in 0..orig_side {
                    let orig_val = flat_before[x + y * orig_side + z * orig_side * orig_side];
                    let (px, py, pz) = (x + 4, y + 4, z + 4);
                    let pad_val =
                        padded_grid[px + py * padded_side + pz * padded_side * padded_side];
                    assert_eq!(orig_val, pad_val, "center mismatch at ({x},{y},{z})");
                }
            }
        }
        for z in 0..padded_side {
            for y in 0..padded_side {
                for x in 0..padded_side {
                    if (4..12).contains(&x) && (4..12).contains(&y) && (4..12).contains(&z) {
                        continue;
                    }
                    let val = padded_grid[x + y * padded_side + z * padded_side * padded_side];
                    assert_eq!(val, 0, "padding not empty at ({x},{y},{z}): {val}");
                }
            }
        }
    }

    #[test]
    fn center_node_extracts_inner_octants() {
        let mut world = World::new(3);
        // Place stone throughout so we have distinct subtrees.
        world.set(wc(3), wc(3), wc(3), STONE);
        world.set(wc(4), wc(4), wc(4), STONE);
        let center = world.center_node(world.root, 3);
        let center_grid = world.store.flatten(center, 4);
        for z in 0..4usize {
            for y in 0..4usize {
                for x in 0..4usize {
                    let expected = world.store.get_cell(
                        world.root,
                        (x + 2) as u64,
                        (y + 2) as u64,
                        (z + 2) as u64,
                    );
                    let actual = center_grid[x + y * 4 + z * 16];
                    assert_eq!(expected, actual, "center mismatch at ({x},{y},{z})");
                }
            }
        }
    }
}
