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
//! Step results are memoized by (NodeId, parity) so identical subtrees anywhere
//! in the world share a single cache entry per generation. Block-rule partition
//! uses node-local alignment (9ww), so origin is not in the cache key.
//! Compaction is deferred (m1f.14): intermediate nodes survive across frames,
//! enabling cache hits at every recursion level. Periodic compaction triggers
//! when the store exceeds 2× its post-compaction size. Since parity alternates
//! 0/1, cache hits occur every OTHER frame for stable subtrees.
//!
//! **Cache-preserving compaction (m1f.15.4):** The recursive descent builds
//! intermediate nodes (27 per recursive level) that become cache keys but are
//! NOT reachable from the step result. When compaction fires, cache keys are
//! passed as extra roots to `compacted_with_remap_keeping`, keeping them alive
//! through GC so cache entries survive.

use super::world::World;
use crate::octree::node::octant_index;
use crate::octree::{Cell, CellState, Node, NodeId};
use rustc_hash::FxHashMap;

const LEVEL3_SIDE: usize = 8;
const LEVEL3_CELL_COUNT: usize = LEVEL3_SIDE * LEVEL3_SIDE * LEVEL3_SIDE;
const CENTER_LEVEL3_SIDE: usize = 4;
const CENTER_LEVEL3_CELL_COUNT: usize =
    CENTER_LEVEL3_SIDE * CENTER_LEVEL3_SIDE * CENTER_LEVEL3_SIDE;

impl World {
    /// Step the world forward one generation using the recursive Hashlife path.
    pub fn step_recursive(&mut self) {
        assert!(
            self.level >= 3,
            "step_recursive requires level >= 3, got {}",
            self.level
        );
        self.hashlife_stats = super::world::HashlifeStats::default();
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        let parity = (self.generation % 2) as u32;
        let result = self.step_node(padded_root, padded_level, parity);
        self.root = result;

        // Post-step gravity gap-fill: prevents Margolus rarefaction.
        // Applied on the flattened grid (same as brute-force path) because
        // gap-fill is a non-local vertical operation that doesn't decompose
        // into hashlife's recursive structure.
        if self.has_block_rule_cells() {
            let side = self.side();
            let mut grid = self.store.flatten(self.root, side);
            super::world::gravity_gap_fill(&mut grid, side, &self.materials);
            self.root = self.store.from_flat(&grid, side);
        }

        self.generation += 1;

        self.maybe_compact();
    }

    /// Number of generations advanced by [`Self::step_recursive_pow2`].
    pub fn recursive_pow2_step_count(&self) -> u64 {
        1u64 << (self.level - 1)
    }

    /// Step the world forward by the largest power-of-two skip supported by
    /// the current realized root size.
    pub fn step_recursive_pow2(&mut self) {
        assert!(
            self.level >= 3,
            "step_recursive_pow2 requires level >= 3, got {}",
            self.level
        );
        if self.has_block_rule_cells() {
            let steps = self.recursive_pow2_step_count();
            for _ in 0..steps {
                self.step();
            }
            self.block_rule_present = None; // brute-force may have consumed block-rule cells
            return;
        }
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        let result = self.step_node_macro(padded_root, padded_level, self.generation);
        self.root = result;
        self.generation += self.recursive_pow2_step_count();

        self.maybe_compact();
    }

    /// Compact the store when it has grown past 2× its post-compaction size,
    /// keeping cache-referenced intermediate nodes alive (m1f.15.4).
    ///
    /// Deferred compaction lets intermediate nodes from recursive descent
    /// survive across frames, enabling cache hits. When compaction fires,
    /// cache keys are passed as extra roots so they survive GC — without
    /// this, every compaction destroys the cache and forces full
    /// recomputation.
    fn maybe_compact(&mut self) {
        let current_size = self.store.stats();
        if self.store_size_at_last_compact == 0 {
            self.store_size_at_last_compact = current_size;
            return;
        }
        if current_size <= self.store_size_at_last_compact * 2 {
            return;
        }
        let extra_roots = self.cache_referenced_nodes();
        let (new_store, new_root, remap) = self
            .store
            .compacted_with_remap_keeping(self.root, &extra_roots);
        self.store = new_store;
        self.root = new_root;
        self.remap_caches(&remap);
        self.store_size_at_last_compact = self.store.stats();
    }

    /// Collect cache key NodeIds that must survive compaction.
    /// Cache keys are intermediate nodes not reachable from the world root;
    /// cache values are step results that ARE reachable from root, so only
    /// keys need to be kept alive as extra roots.
    fn cache_referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes =
            Vec::with_capacity(self.hashlife_cache.len() + self.hashlife_macro_cache.len());
        for &(node, _) in self.hashlife_cache.keys() {
            nodes.push(node);
        }
        for &(node, _) in self.hashlife_macro_cache.keys() {
            nodes.push(node);
        }
        nodes
    }

    fn has_block_rule_cells(&mut self) -> bool {
        if let Some(cached) = self.block_rule_present {
            return cached;
        }
        let result = self.has_block_rule_in_subtree(self.root);
        self.block_rule_present = Some(result);
        result
    }

    /// Walk the octree for any cell with a BlockRule. Short-circuits on empty
    /// nodes — avoids O(n³) flatten for worlds that are mostly empty/inert.
    fn has_block_rule_in_subtree(&self, node: NodeId) -> bool {
        if node == NodeId::EMPTY {
            return false;
        }
        match self.store.get(node) {
            Node::Leaf(state) => self
                .materials
                .block_rule_id_for_cell(Cell::from_raw(*state))
                .is_some(),
            Node::Interior { children, .. } => {
                children.iter().any(|&c| self.has_block_rule_in_subtree(c))
            }
        }
    }

    /// Check if every leaf in the subtree is inert (CaRule::Noop, no BlockRule).
    /// Unlike `inert_uniform_state`, this catches mixed-material subtrees
    /// (e.g. stone/air boundaries). For all-inert nodes, stepping produces
    /// the center extraction — no CA computation needed.
    fn is_all_inert(&mut self, node: NodeId) -> bool {
        if node == NodeId::EMPTY {
            return true;
        }
        if let Some(&cached) = self.hashlife_all_inert_cache.get(&node) {
            return cached;
        }
        let result = match self.store.get(node).clone() {
            Node::Leaf(state) => {
                state == 0
                    || self
                        .materials
                        .cell_is_inert_fixed_point(Cell::from_raw(state))
            }
            Node::Interior { children, .. } => children.into_iter().all(|c| self.is_all_inert(c)),
        };
        self.hashlife_all_inert_cache.insert(node, result);
        result
    }

    /// Check if a subtree is uniformly one inert material.
    /// Returns Some(state) if all leaves are the same inert state.
    fn inert_uniform_state(&mut self, node: NodeId) -> Option<CellState> {
        if let Some(&cached) = self.hashlife_inert_cache.get(&node) {
            return cached;
        }
        let result = match self.store.get(node).clone() {
            Node::Leaf(state) => self
                .materials
                .cell_is_inert_fixed_point(Cell::from_raw(state))
                .then_some(state),
            Node::Interior { children, .. } => {
                let state = self.inert_uniform_state(children[0])?;
                for child in children.into_iter().skip(1) {
                    if self.inert_uniform_state(child) != Some(state) {
                        self.hashlife_inert_cache.insert(node, None);
                        return None;
                    }
                }
                Some(state)
            }
        };
        self.hashlife_inert_cache.insert(node, result);
        result
    }

    /// Remap hashlife cache keys and values through a compaction remap table.
    /// Entries referencing unreachable nodes (not in remap) are dropped.
    fn remap_caches(&mut self, remap: &FxHashMap<NodeId, NodeId>) {
        // Remap hashlife_cache: (NodeId, parity) → NodeId
        let old_cache = std::mem::take(&mut self.hashlife_cache);
        self.hashlife_cache.reserve(old_cache.len());
        for ((node, parity), result) in old_cache {
            if let (Some(&new_node), Some(&new_result)) = (remap.get(&node), remap.get(&result)) {
                self.hashlife_cache.insert((new_node, parity), new_result);
            }
        }

        // Remap hashlife_macro_cache: (NodeId, generation) → NodeId
        let old_macro = std::mem::take(&mut self.hashlife_macro_cache);
        self.hashlife_macro_cache.reserve(old_macro.len());
        for ((node, gen), result) in old_macro {
            if let (Some(&new_node), Some(&new_result)) = (remap.get(&node), remap.get(&result)) {
                self.hashlife_macro_cache
                    .insert((new_node, gen), new_result);
            }
        }

        // Remap hashlife_inert_cache: NodeId → Option<CellState>
        let old_inert = std::mem::take(&mut self.hashlife_inert_cache);
        self.hashlife_inert_cache.reserve(old_inert.len());
        for (node, state) in old_inert {
            if let Some(&new_node) = remap.get(&node) {
                self.hashlife_inert_cache.insert(new_node, state);
            }
        }

        // Remap hashlife_all_inert_cache: NodeId → bool
        let old_all_inert = std::mem::take(&mut self.hashlife_all_inert_cache);
        self.hashlife_all_inert_cache.reserve(old_all_inert.len());
        for (node, inert) in old_all_inert {
            if let Some(&new_node) = remap.get(&node) {
                self.hashlife_all_inert_cache.insert(new_node, inert);
            }
        }
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
    /// Block-rule partition uses node-local alignment, so origin is not needed
    /// and the cache key is just (NodeId, parity) — enabling spatial memoization
    /// for all worlds, including those with block rules (9ww).
    fn step_node(&mut self, node: NodeId, level: u32, parity: u32) -> NodeId {
        assert!(level >= 3, "step_node requires level >= 3, got {level}");

        // Empty nodes step to empty: any rule applied to 26 air neighbors produces air
        // (NoopRule is identity; GoL-family rules have birth_min >= 1). No BlockRule on air.
        if self.store.population(node) == 0 {
            self.hashlife_stats.empty_skips += 1;
            return self.store.empty(level - 1);
        }

        // Fixed-point: uniform inert subtree (all leaves same inert material).
        // Stepping produces the same uniform node one level smaller.
        if let Some(state) = self.inert_uniform_state(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return self.store.uniform(level - 1, state);
        }

        // All-inert: every leaf is CaRule::Noop, no BlockRule (but possibly
        // mixed materials, e.g. stone/air boundary). Stepping = center extract.
        if self.is_all_inert(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return self.center_node(node, level);
        }

        let key = (node, parity);
        if let Some(&cached) = self.hashlife_cache.get(&key) {
            self.hashlife_stats.cache_hits += 1;
            return cached;
        }
        self.hashlife_stats.cache_misses += 1;

        let result = if level == 3 {
            self.step_base_case(node, parity)
        } else {
            self.step_recursive_case(node, level, parity)
        };

        self.hashlife_cache.insert(key, result);
        result
    }

    /// Base case: level-3 node (8×8×8). Flatten, run CaRule on interior 6³,
    /// run BlockRule on all aligned blocks, extract center 4³ → level-2 output.
    fn step_base_case(&mut self, node: NodeId, _parity: u32) -> NodeId {
        let grid = self.store.flatten(node, LEVEL3_SIDE);
        let next = self.step_grid_once(&grid, self.generation);
        self.center_level3_grid_to_node(&next)
    }

    fn step_node_macro(&mut self, node: NodeId, level: u32, generation: u64) -> NodeId {
        debug_assert!(
            level >= 3,
            "step_node_macro requires level >= 3, got {level}"
        );

        // Empty nodes step to empty across any number of generations:
        // identity^N = identity. CaRule-only worlds (macro path prerequisite).
        if self.store.population(node) == 0 {
            return self.store.empty(level - 1);
        }

        // Fixed-point: uniform inert subtree → same material, smaller node.
        if let Some(state) = self.inert_uniform_state(node) {
            return self.store.uniform(level - 1, state);
        }

        // All-inert: mixed materials but all noop → center extract.
        if self.is_all_inert(node) {
            return self.center_node(node, level);
        }

        let key = (node, generation);
        if let Some(&cached) = self.hashlife_macro_cache.get(&key) {
            return cached;
        }

        let result = if level == 3 {
            self.step_base_case_macro(node, generation)
        } else {
            self.step_recursive_case_macro(node, level, generation)
        };

        self.hashlife_macro_cache.insert(key, result);
        result
    }

    fn step_base_case_macro(&mut self, node: NodeId, generation: u64) -> NodeId {
        let grid = self.store.flatten(node, LEVEL3_SIDE);
        let next = self.step_grid_once(&grid, generation);
        let next = self.step_grid_once(&next, generation + 1);
        self.center_level3_grid_to_node(&next)
    }

    fn step_grid_once(
        &self,
        grid: &[CellState],
        generation: u64,
    ) -> [CellState; LEVEL3_CELL_COUNT] {
        let side = LEVEL3_SIDE;
        // Phase 1: CaRule on interior cells (1..side-1 on each axis).
        // The outermost ring cannot be evolved correctly because its neighbors
        // would wrap outside the padded region. Callers only extract the center
        // that remains valid after the requested number of steps.
        let mut next = [0 as CellState; LEVEL3_CELL_COUNT];
        for z in 1..side - 1 {
            for y in 1..side - 1 {
                for x in 1..side - 1 {
                    let cell = Cell::from_raw(grid[x + y * side + z * side * side]);
                    let neighbors = get_neighbors_from_grid(grid, side, x, y, z);
                    let rule = self.materials.rule_for_cell(cell).unwrap_or_else(|| {
                        panic!("missing CaRule for material {}", cell.material())
                    });
                    next[x + y * side + z * side * side] = rule.step_cell(cell, &neighbors).raw();
                }
            }
        }

        // Phase 2: BlockRule on all aligned 2×2×2 blocks within the interior.
        // Node-local alignment (9ww): the partition is determined by grid-local
        // coordinates, not world-space origin. Every block starting at offset
        // is valid — no world-space alignment check needed. This makes the
        // cache key origin-independent, enabling spatial memoization for all
        // worlds including those with block rules.
        let offset = (generation % 2) as usize;
        let mut bz = offset;
        while bz + 1 < side - 1 {
            let mut by = offset;
            while by + 1 < side - 1 {
                let mut bx = offset;
                while bx + 1 < side - 1 {
                    self.apply_block_in_grid(&mut next, side, bx, by, bz);
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }

        next
    }

    fn center_level3_grid_to_node(&mut self, grid: &[CellState]) -> NodeId {
        let mut center_grid = [0 as CellState; CENTER_LEVEL3_CELL_COUNT];
        for cz in 0..CENTER_LEVEL3_SIDE {
            for cy in 0..CENTER_LEVEL3_SIDE {
                for cx in 0..CENTER_LEVEL3_SIDE {
                    center_grid[cx
                        + cy * CENTER_LEVEL3_SIDE
                        + cz * CENTER_LEVEL3_SIDE * CENTER_LEVEL3_SIDE] = grid
                        [(cx + 2) + (cy + 2) * LEVEL3_SIDE + (cz + 2) * LEVEL3_SIDE * LEVEL3_SIDE];
                }
            }
        }
        self.store.from_flat(&center_grid, CENTER_LEVEL3_SIDE)
    }

    /// Apply a single block rule within a flat grid.
    fn apply_block_in_grid(
        &self,
        grid: &mut [CellState],
        side: usize,
        bx: usize,
        by: usize,
        bz: usize,
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
        let result = rule.step_block(&block);

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
    fn step_recursive_case(&mut self, node: NodeId, level: u32, parity: u32) -> NodeId {
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
                    result_children[octant_index(ox as u32, oy as u32, oz as u32)] =
                        self.step_node(sub_root, level - 1, parity);
                }
            }
        }

        self.store.interior(level - 1, result_children)
    }

    fn step_recursive_case_macro(&mut self, node: NodeId, level: u32, generation: u64) -> NodeId {
        debug_assert!(level > 3, "macro recursive case requires level > 3");
        let children = self.store.children(node);
        let sub: [[NodeId; 8]; 8] = std::array::from_fn(|i| self.store.children(children[i]));

        let half_skip = 1u64 << (level - 3);

        // Phase 1: compute each overlapping level-(n-1) intermediate node's
        // center after half of the parent's time skip.
        let mut phased = [NodeId::EMPTY; 27];
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
                    let inter = self.store.interior(level - 1, octants);
                    phased[px + py * 3 + pz * 9] =
                        self.step_node_macro(inter, level - 1, generation);
                }
            }
        }

        // Phase 2: assemble those half-stepped centers into the 8 overlapping
        // sub-cubes, then advance the remaining half-skip from the shifted
        // generation base.
        let mut result_children = [NodeId::EMPTY; 8];
        for oz in 0..2usize {
            for oy in 0..2usize {
                for ox in 0..2usize {
                    let mut sub_cube = [NodeId::EMPTY; 8];
                    for dz in 0..2usize {
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                sub_cube[octant_index(dx as u32, dy as u32, dz as u32)] =
                                    phased[(ox + dx) + (oy + dy) * 3 + (oz + dz) * 9];
                            }
                        }
                    }
                    let sub_root = self.store.interior(level - 1, sub_cube);
                    result_children[octant_index(ox as u32, oy as u32, oz as u32)] =
                        self.step_node_macro(sub_root, level - 1, generation + half_skip);
                }
            }
        }

        self.store.interior(level - 1, result_children)
    }

    /// Extract the center (level n-1) of a level-n node.
    fn center_node(&mut self, node: NodeId, level: u32) -> NodeId {
        assert!(level >= 2, "center_node requires level >= 2");
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

    fn embed_world_in_center(source: &World, target: &mut World) {
        let source_side = source.side() as u64;
        let target_side = target.side() as u64;
        let offset = (target_side - source_side) / 2;
        let grid = source.flatten();

        for z in 0..source_side {
            for y in 0..source_side {
                for x in 0..source_side {
                    let idx = x as usize
                        + y as usize * source_side as usize
                        + z as usize * source_side as usize * source_side as usize;
                    let state = grid[idx];
                    if state != 0 {
                        target.set(wc(offset + x), wc(offset + y), wc(offset + z), state);
                    }
                }
            }
        }
    }

    fn extract_center_cube(world: &World, side: u64) -> Vec<CellState> {
        let world_side = world.side() as u64;
        let offset = (world_side - side) / 2;
        let mut grid = vec![0; (side * side * side) as usize];

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let idx = x as usize
                        + y as usize * side as usize
                        + z as usize * side as usize * side as usize;
                    grid[idx] = world.get(wc(offset + x), wc(offset + y), wc(offset + z));
                }
            }
        }

        grid
    }

    fn seed_random_material_cells(world: &mut World, seed: u64) {
        let mut rng = SimpleRng::new(seed);
        let side = world.side() as u64;
        let water = crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw();

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let roll = rng.next_u64() % 7;
                    let state = match roll {
                        0 | 1 => water,
                        2 => STONE,
                        _ => 0,
                    };
                    if state != 0 {
                        world.set(wc(x), wc(y), wc(z), state);
                    }
                }
            }
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
        // Level 4 (16³) so cells stay well inside the absorbing boundary
        // over 4 steps. Both brute-force and hashlife now use absorbing BC
        // (OOB = empty), but level 4 gives more room for cells to move.
        let mut brute = World::new(4);
        let mut recur = World::new(4);
        for &(x, y, z, mat) in &[
            (6u64, 6, 6, STONE),
            (
                7,
                7,
                7,
                crate::octree::Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            ),
            (
                8,
                8,
                8,
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
        let cases = [(3_u32, 1_usize), (4_u32, 3_usize)];

        for (preset_idx, &(label, rule)) in presets.iter().enumerate() {
            for (level, steps) in cases {
                for case_seed in 0..4_u64 {
                    let simulation_seed = 0x6f03_u64
                        ^ ((preset_idx as u64) << 16)
                        ^ ((level as u64) << 8)
                        ^ case_seed;
                    let initial_seed = simulation_seed ^ 0xa11ce_u64;
                    let mut brute = gol_world(level, rule, simulation_seed);
                    let mut recur = gol_world(level, rule, simulation_seed);
                    seed_random_alive_cells(&mut brute, initial_seed, 0);
                    seed_random_alive_cells(&mut recur, initial_seed, 0);

                    let case =
                        format!("{label} level={level} steps={steps} seed={simulation_seed:#x}");
                    assert_recursive_matches_bruteforce(brute, recur, steps, &case);
                }
            }
        }
    }

    #[test]
    fn recursive_pow2_matches_centered_bruteforce_reference() {
        let presets = [
            ("amoeba", GameOfLife3D::new(9, 26, 5, 7)),
            ("crystal", GameOfLife3D::new(0, 6, 1, 3)),
            ("445", GameOfLife3D::rule445()),
            ("pyroclastic", GameOfLife3D::new(4, 7, 6, 8)),
        ];

        for (preset_idx, &(label, rule)) in presets.iter().enumerate() {
            for level in [3_u32, 4_u32] {
                let simulation_seed = 0x6f07_u64 ^ ((preset_idx as u64) << 16) ^ (level as u64);
                let mut fast = gol_world(level, rule, simulation_seed);
                let mut brute = gol_world(level + 1, rule, simulation_seed);
                seed_random_alive_cells(&mut fast, simulation_seed ^ 0xfeed_u64, 0);
                embed_world_in_center(&fast, &mut brute);

                let skip = fast.recursive_pow2_step_count();
                for _ in 0..skip {
                    brute.step();
                }
                fast.step_recursive_pow2();

                assert_eq!(
                    fast.flatten(),
                    extract_center_cube(&brute, fast.side() as u64),
                    "{label} level={level} skip={skip}: macro-step mismatch"
                );
                assert_eq!(
                    fast.generation, skip,
                    "{label} level={level}: generation should advance by the macro skip"
                );
            }
        }
    }

    #[test]
    fn recursive_pow2_tracks_nonzero_generation_for_material_rules() {
        let mut fast = World::new(3);
        let mut expected = World::new(3);
        fast.generation = 1;
        expected.generation = 1;
        seed_random_material_cells(&mut fast, 0x5eed_u64);
        seed_random_material_cells(&mut expected, 0x5eed_u64);

        let skip = fast.recursive_pow2_step_count();
        for _ in 0..skip {
            expected.step();
        }
        fast.step_recursive_pow2();

        assert_eq!(
            fast.flatten(),
            expected.flatten(),
            "material pow2 fallback should match repeated brute-force stepping"
        );
        assert_eq!(fast.generation, 1 + skip);
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

    /// Timing comparison: brute-force vs recursive Hashlife on 64³ terrain.
    /// Run with `cargo test --release --lib bench_stepper_comparison -- --ignored --nocapture`.
    /// At 64³ the recursive path is roughly at parity with brute-force thanks to
    /// the empty-node short-circuit (6gf.14) and incremental cache (m1f.11/m1f.12).
    #[test]
    #[ignore]
    fn bench_stepper_comparison() {
        use crate::terrain::TerrainParams;
        use std::time::Instant;

        let steps = 10;
        let level = 6; // 64³

        // Set up identical worlds with terrain.
        let params = TerrainParams::default();
        let mut brute = World::new(level);
        let mut recur = World::new(level);
        brute.seed_terrain(&params);
        recur.seed_terrain(&params);

        // Brute-force timing.
        let t0 = Instant::now();
        for _ in 0..steps {
            brute.step();
        }
        let brute_ms = t0.elapsed().as_millis();

        // Recursive timing.
        let t0 = Instant::now();
        for _ in 0..steps {
            recur.step_recursive();
        }
        let recur_ms = t0.elapsed().as_millis();

        // Correctness check.
        assert_eq!(brute.flatten(), recur.flatten(), "results diverged");

        eprintln!(
            "64³ terrain × {steps} steps: brute={brute_ms}ms, recursive={recur_ms}ms, \
             ratio={:.2}x",
            brute_ms as f64 / recur_ms.max(1) as f64
        );
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

    #[test]
    fn recursive_matches_brute_force_level5_stone_only() {
        // Level 5 = 32³ with inert stone only (no BlockRule).
        // Exercises deep recursion without Margolus complexity.
        let mut brute = World::new(5);
        let mut recur = World::new(5);
        let mut rng = SimpleRng::new(0xdee5_u64);
        for z in 0..32u64 {
            for y in 0..32u64 {
                for x in 0..32u64 {
                    if rng.next_u64().is_multiple_of(4) {
                        brute.set(wc(x), wc(y), wc(z), STONE);
                        recur.set(wc(x), wc(y), wc(z), STONE);
                    }
                }
            }
        }
        assert_recursive_matches_bruteforce(brute, recur, 2, "level5-stone");
    }

    /// Seed water and stone with a margin. CaRule boundaries now match
    /// (both absorbing), but BlockRule still differs: brute-force clips
    /// partial blocks at edges, hashlife processes them via overlapping
    /// sub-cubes. Margins keep block-rule-bearing cells away from edges.
    fn seed_random_material_cells_margined(world: &mut World, seed: u64, margin: u64) {
        let mut rng = SimpleRng::new(seed);
        let side = world.side() as u64;
        let water = crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw();
        for z in margin..(side - margin) {
            for y in margin..(side - margin) {
                for x in margin..(side - margin) {
                    let roll = rng.next_u64() % 7;
                    let state = match roll {
                        0 | 1 => water,
                        2 => STONE,
                        _ => 0,
                    };
                    if state != 0 {
                        world.set(wc(x), wc(y), wc(z), state);
                    }
                }
            }
        }
    }

    #[test]
    fn recursive_matches_brute_force_level4_materials() {
        let mut brute = World::new(4);
        let mut recur = World::new(4);
        seed_random_material_cells_margined(&mut brute, 0xdee4_u64, 2);
        seed_random_material_cells_margined(&mut recur, 0xdee4_u64, 2);
        assert_recursive_matches_bruteforce(brute, recur, 3, "level4-materials");
    }

    #[test]
    fn recursive_matches_brute_force_level5_materials() {
        let mut brute = World::new(5);
        let mut recur = World::new(5);
        seed_random_material_cells_margined(&mut brute, 0xdee5_u64, 3);
        seed_random_material_cells_margined(&mut recur, 0xdee5_u64, 3);
        assert_recursive_matches_bruteforce(brute, recur, 2, "level5-materials");
    }

    #[test]
    fn recursive_matches_brute_force_level5_gol() {
        // GoL at level 5 — the CaRule-only path through deep recursion
        let rule = GameOfLife3D::rule445();
        let mut brute = gol_world(5, rule, 0xd33f_u64);
        let mut recur = gol_world(5, rule, 0xd33f_u64);
        seed_random_alive_cells(&mut brute, 0xd33f_u64 ^ 0xa11ce, 0);
        seed_random_alive_cells(&mut recur, 0xd33f_u64 ^ 0xa11ce, 0);
        assert_recursive_matches_bruteforce(brute, recur, 2, "level5-gol-445");
    }

    #[test]
    fn source_index_all_valid_pairs() {
        assert_eq!(source_index(0, 0), (0, 0));
        assert_eq!(source_index(0, 1), (0, 1));
        assert_eq!(source_index(1, 0), (0, 1));
        assert_eq!(source_index(1, 1), (1, 0));
        assert_eq!(source_index(2, 0), (1, 0));
        assert_eq!(source_index(2, 1), (1, 1));
    }

    #[test]
    #[should_panic]
    fn source_index_out_of_range_panics() {
        source_index(3, 0);
    }
}
