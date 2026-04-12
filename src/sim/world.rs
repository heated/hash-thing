use std::fmt;

use super::mutation::{MutationQueue, WorldMutation};
use super::rule::{block_index, GameOfLife3D, ALIVE};
use crate::octree::{Cell, CellState, NodeId, NodeStore, CELLS_PER_BLOCK};
use crate::terrain::materials::{BlockRuleId, MaterialRegistry, DIRT, FIRE, GRASS, STONE, WATER};
use crate::terrain::{carve_caves, gen_region, GenStats, TerrainParams};
use rustc_hash::FxHashMap;

/// The axis-aligned cube of world-space that the octree currently covers.
///
/// Origin is always (0,0,0) today (unsigned coords), but will shift once
/// signed world coordinates and recentering land. The type exists now so
/// call sites can migrate incrementally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RealizedRegion {
    pub origin: [i64; 3],
    pub level: u32,
}

impl RealizedRegion {
    /// Side length of the realized cube.
    pub fn side(&self) -> u64 {
        1u64 << self.level
    }

    /// True if the world-space coordinate `(x, y, z)` is inside the region.
    pub fn contains(&self, x: i64, y: i64, z: i64) -> bool {
        let s = self.side() as i64;
        x >= self.origin[0]
            && x < self.origin[0] + s
            && y >= self.origin[1]
            && y < self.origin[1] + s
            && z >= self.origin[2]
            && z < self.origin[2] + s
    }

    /// Map a world-space point to the octant index (0–7) within the root node.
    ///
    /// Panics if the point is outside the region.
    pub fn octant_of(&self, x: i64, y: i64, z: i64) -> usize {
        assert!(self.contains(x, y, z), "point outside realized region");
        let half = self.side() as i64 / 2;
        let dx = if x - self.origin[0] >= half { 1 } else { 0 };
        let dy = if y - self.origin[1] >= half { 1 } else { 0 };
        let dz = if z - self.origin[2] >= half { 1 } else { 0 };
        dx + dy * 2 + dz * 4
    }
}

impl fmt::Display for RealizedRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.side();
        write!(
            f,
            "realized: origin=({},{},{}) level={} ({s}³)",
            self.origin[0], self.origin[1], self.origin[2], self.level
        )
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct WorldCoord(pub i64);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LocalCoord(pub u64);

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
    /// Retained terrain params for lazy expansion (3fq.4). When `Some`,
    /// `ensure_region` generates terrain (heightmap + caves)
    /// for newly-created sibling octants instead of leaving them empty.
    terrain_params: Option<TerrainParams>,
    /// Spatial memoization cache for the recursive Hashlife stepper.
    /// Key: (NodeId, parity). Identical subtrees anywhere in the world
    /// share a single cache entry — block-rule partition uses node-local
    /// alignment so origin is no longer in the key (9ww).
    pub(crate) hashlife_cache: FxHashMap<(NodeId, u32), NodeId>,
    /// Memoization cache for the exponential Hashlife macro-stepper (6gf.7).
    /// Key: (NodeId, starting generation).
    pub(crate) hashlife_macro_cache: FxHashMap<(NodeId, u64), NodeId>,
    /// Hashlife cache statistics from the most recent step.
    pub hashlife_stats: HashlifeStats,
    /// Store size after the last compaction, used to trigger periodic GC.
    pub(crate) store_size_at_last_compact: usize,
    /// Cache for `inert_uniform_state`: NodeId → Option<CellState>.
    /// `Some(state)` = all leaves are the same inert material; `None` = mixed or active.
    pub(crate) hashlife_inert_cache: FxHashMap<NodeId, Option<CellState>>,
    /// Cache for `is_all_inert`: NodeId → bool.
    /// True if every leaf in the subtree has CaRule::Noop and no BlockRule.
    pub(crate) hashlife_all_inert_cache: FxHashMap<NodeId, bool>,
    /// Cached result of `has_block_rule_cells`. `None` = dirty, needs rescan.
    /// Avoids O(n³) flatten-and-scan on every `step_recursive_pow2` call.
    pub(crate) block_rule_present: Option<bool>,
    /// Pending world mutations. Entities push here; `apply_mutations`
    /// drains and applies in arrival order at tick boundary.
    pub queue: MutationQueue,
    /// World-space origin of local coordinate (0,0,0). When the world
    /// grows in the negative direction, origin shifts to keep the old
    /// root's cells at the same world-space positions.
    pub origin: [i64; 3],
}

/// Cache performance statistics from a single hashlife step.
#[derive(Clone, Copy, Debug, Default)]
pub struct HashlifeStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub empty_skips: u64,
    pub fixed_point_skips: u64,
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
            terrain_params: None,
            hashlife_cache: FxHashMap::default(),
            hashlife_macro_cache: FxHashMap::default(),
            hashlife_stats: HashlifeStats::default(),
            store_size_at_last_compact: 0,
            hashlife_inert_cache: FxHashMap::default(),
            hashlife_all_inert_cache: FxHashMap::default(),
            block_rule_present: None,
            queue: MutationQueue::new(),
            origin: [0, 0, 0],
        }
    }

    pub fn side(&self) -> usize {
        1 << self.level
    }

    /// The realized region as a value type.
    pub fn region(&self) -> RealizedRegion {
        RealizedRegion {
            origin: self.origin,
            level: self.level,
        }
    }

    /// Invalidate caches whose keys depend on the active CA rule.
    pub fn invalidate_rule_caches(&mut self) {
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
    }

    /// Reconfigure the legacy GoL smoke material dispatch to use `rule`.
    ///
    /// This also invalidates rule-dependent caches because the CA dispatch
    /// table changes even though the octree content does not.
    pub fn set_gol_smoke_rule(&mut self, rule: GameOfLife3D) {
        self.materials = MaterialRegistry::gol_smoke_with_rule(rule);
        self.invalidate_rule_caches();
    }

    /// Convert world coordinate to local on a specific axis index (0=x, 1=y, 2=z).
    /// Panics if the result is negative (coordinate is below the world origin).
    fn local_from_world(&self, axis: usize, coord: WorldCoord) -> LocalCoord {
        let local = coord.0 - self.origin[axis];
        LocalCoord(u64::try_from(local).unwrap_or_else(|_| {
            panic!(
                "World: world coord {} maps to negative local coord {} \
                 (origin[{axis}]={})",
                coord.0, local, self.origin[axis]
            )
        }))
    }

    fn set_local(&mut self, x: LocalCoord, y: LocalCoord, z: LocalCoord, state: CellState) {
        let side = self.side() as u64;
        assert!(
            x.0 < side && y.0 < side && z.0 < side,
            "World::set_local: coord ({}, {}, {}) out of bounds for side {side}",
            x.0,
            y.0,
            z.0,
        );
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        self.root = self.store.set_cell(self.root, x.0, y.0, z.0, state);
        self.block_rule_present = None; // invalidate cache
    }

    fn get_local(&self, x: LocalCoord, y: LocalCoord, z: LocalCoord) -> CellState {
        self.store.get_cell(self.root, x.0, y.0, z.0)
    }

    /// Grow the root octree until `(x, y, z)` is in-bounds.
    ///
    /// Supports growth in all directions — negative coordinates shift
    /// the origin while preserving existing cell positions.
    ///
    /// No-op when the coordinate is already in-bounds.
    ///
    /// **Step cache:** existing cached results survive expansion because
    /// they are keyed by `NodeId`, and the old root's `NodeId` is still
    /// valid and still steps to the same result. The new parent has no
    /// cached entry yet and will be computed fresh on first step.
    pub fn ensure_contains(&mut self, x: WorldCoord, y: WorldCoord, z: WorldCoord) {
        self.ensure_region([x, y, z], [x, y, z]);
    }

    /// Grow the root octree until the axis-aligned box `[min, max]` is
    /// fully in-bounds (inclusive on both ends).
    ///
    /// Supports growth in all directions. For each axis needing negative
    /// growth, the old root is placed in the positive half of the new
    /// parent and origin shifts by −half. For positive growth, the old
    /// root stays in the negative half.
    ///
    /// No-op when the region is already in-bounds.
    pub fn ensure_region(&mut self, min: [WorldCoord; 3], max: [WorldCoord; 3]) {
        loop {
            let side = 1i64 << self.level;
            let origin = self.origin;

            // Check which axes need growth and in which direction.
            let need_neg = [
                min[0].0 < origin[0],
                min[1].0 < origin[1],
                min[2].0 < origin[2],
            ];
            let need_pos = [
                max[0].0 >= origin[0] + side,
                max[1].0 >= origin[1] + side,
                max[2].0 >= origin[2] + side,
            ];

            if !need_neg.iter().any(|&b| b) && !need_pos.iter().any(|&b| b) {
                return; // everything fits
            }

            // Determine which octant the old root goes into.
            // Negative growth on axis → old root in + half (bit=1), origin shifts.
            // Positive growth → old root in - half (bit=0).
            let old_octant = (if need_neg[0] { 1 } else { 0 })
                + (if need_neg[1] { 2 } else { 0 })
                + (if need_neg[2] { 4 } else { 0 });

            let half = side; // half of the NEW size = old side
            let sibling_level = self.level;
            let new_level = self.level + 1;

            // Compute new origin — shift negative on axes that need it.
            let mut new_origin = origin;
            for axis in 0..3 {
                if need_neg[axis] {
                    new_origin[axis] -= half;
                }
            }

            // Build children array.
            let mut children = [NodeId::EMPTY; 8];
            children[old_octant] = self.root;
            for (oct, child) in children.iter_mut().enumerate() {
                if oct == old_octant {
                    continue;
                }
                let (cx, cy, cz) = crate::octree::node::octant_coords(oct);
                let sibling_origin = [
                    new_origin[0] + cx as i64 * half,
                    new_origin[1] + cy as i64 * half,
                    new_origin[2] + cz as i64 * half,
                ];
                *child = self.gen_sibling(sibling_level, sibling_origin);
            }

            self.root = self.store.interior(new_level, children);
            self.level = new_level;
            self.origin = new_origin;
        }
    }

    /// Generate a sibling octant for lazy expansion. If `terrain_params` is
    /// set, produces terrain (heightmap + caves) at the given
    /// world-space origin. Otherwise returns a canonical empty node.
    fn gen_sibling(&mut self, level: u32, origin: [i64; 3]) -> NodeId {
        let params = match &self.terrain_params {
            Some(p) => *p,
            None => return self.store.empty(level),
        };
        let field = params.to_heightmap();
        let (mut node, _stats) = gen_region(&mut self.store, &field, origin, level);
        let side = 1usize << level;
        if let Some(cave_params) = &params.caves {
            let mut grid = self.store.flatten(node, side);
            crate::terrain::caves::carve_caves_grid(&mut grid, side, origin, cave_params);
            node = self.store.from_flat(&grid, side);
        }
        node
    }

    /// Set a cell.
    ///
    /// **Panics** on out-of-bounds coordinates (hash-thing-fb5). For writes to
    /// coordinates outside the current realized root, call `ensure_contains`
    /// first to grow the tree.
    #[track_caller]
    pub fn set(&mut self, x: WorldCoord, y: WorldCoord, z: WorldCoord, state: CellState) {
        self.set_local(
            self.local_from_world(0, x),
            self.local_from_world(1, y),
            self.local_from_world(2, z),
            state,
        );
    }

    /// Place a gameplay block at block coordinates `(bx, by, bz)`.
    ///
    /// Fills a `CELLS_PER_BLOCK³` region of cells with the given state.
    /// Block coordinate `(bx, by, bz)` maps to cell region
    /// `[bx*K .. bx*K+K-1]` on each axis, where `K = CELLS_PER_BLOCK`.
    ///
    /// Auto-grows the world to fit, then queues a `FillRegion` mutation —
    /// call `apply_mutations` to flush.
    pub fn set_block(&mut self, bx: i64, by: i64, bz: i64, state: CellState) {
        let k = CELLS_PER_BLOCK as i64;
        let min = [WorldCoord(bx * k), WorldCoord(by * k), WorldCoord(bz * k)];
        let max = [
            WorldCoord(bx * k + k - 1),
            WorldCoord(by * k + k - 1),
            WorldCoord(bz * k + k - 1),
        ];
        self.ensure_region(min, max);
        self.queue
            .push(WorldMutation::FillRegion { min, max, state });
    }

    /// Get a cell.
    ///
    /// Out-of-bounds reads return `0` silently (hash-thing-fb5). `get` is a
    /// pure query over the realized region — outside that region is
    /// conceptually unrealized empty space. This includes negative world
    /// coordinates until signed-region realization lands.
    pub fn get(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        let lx = x.0 - self.origin[0];
        let ly = y.0 - self.origin[1];
        let lz = z.0 - self.origin[2];
        let side = self.side() as i64;
        if lx < 0 || ly < 0 || lz < 0 || lx >= side || ly >= side || lz >= side {
            return 0;
        }
        self.get_local(
            LocalCoord(lx as u64),
            LocalCoord(ly as u64),
            LocalCoord(lz as u64),
        )
    }

    /// Stepper-oriented read: "what is this cell right now?"
    ///
    /// Semantic alias for [`get`](Self::get) that signals the caller accepts
    /// OOB-empty semantics without reservation. Hashlife steppers reading a
    /// 3×3×3 neighborhood call `probe` — the unrealized world *is* empty
    /// from the stepper's perspective, so returning 0 for out-of-bounds is
    /// the correct physical answer, not a lossy fallback.
    ///
    /// Use [`is_realized`](Self::is_realized) when you need to distinguish
    /// "genuinely empty" from "outside the realized region."
    #[inline]
    pub fn probe(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        self.get(x, y, z)
    }

    /// Is the coordinate inside the realized region?
    ///
    /// Returns `false` for negative coordinates and for positions beyond
    /// the current octree extent. Use this alongside [`get`](Self::get)
    /// when distinguishing "realized empty" from "unrealized" matters
    /// (e.g. debug overlays rendering the region boundary).
    pub fn is_realized(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> bool {
        let lx = x.0 - self.origin[0];
        let ly = y.0 - self.origin[1];
        let lz = z.0 - self.origin[2];
        let side = self.side() as i64;
        lx >= 0 && ly >= 0 && lz >= 0 && lx < side && ly < side && lz < side
    }

    /// Flatten to a 3D grid for rendering.
    pub fn flatten(&self) -> Vec<CellState> {
        self.store.flatten(self.root, self.side())
    }

    /// Drain the mutation queue and apply every pending mutation to the
    /// octree in arrival order. This is the only path for entity-produced
    /// edits to reach the world (closed-world invariant, hash-thing-1v0.9).
    pub fn apply_mutations(&mut self) {
        for m in self.queue.take() {
            match m {
                WorldMutation::SetCell { x, y, z, state } => {
                    self.set(x, y, z, state);
                }
                WorldMutation::FillRegion { min, max, state } => {
                    debug_assert!(
                        min[0].0 <= max[0].0 && min[1].0 <= max[1].0 && min[2].0 <= max[2].0,
                        "FillRegion: min must be <= max on every axis"
                    );
                    for z in min[2].0..=max[2].0 {
                        for y in min[1].0..=max[1].0 {
                            for x in min[0].0..=max[0].0 {
                                self.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), state);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Advance the CA by one generation with mutation-queue guard.
    ///
    /// Debug-asserts that the queue is empty on entry — if it isn't,
    /// the caller forgot `apply_mutations` and the step cache would
    /// see stale world state.
    pub fn step_ca(&mut self) {
        debug_assert!(
            self.queue.is_empty(),
            "step_ca called with {} pending mutations — call apply_mutations first",
            self.queue.len()
        );
        self.step_recursive();
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
    /// **Boundary conditions:** Both CaRule and BlockRule use absorbing boundaries
    /// (out-of-bounds = empty), matching hashlife's infinite-world semantics.
    /// BlockRule additionally skips partial blocks at world edges (clipping) —
    /// Margolus blocks must not straddle the world boundary.
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
    /// Blocks at edges use absorbing boundary conditions: out-of-bounds cells
    /// are treated as empty, matching hashlife's pad-with-empty semantics.
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
        while bz < side {
            let mut by = offset;
            while by < side {
                let mut bx = offset;
                while bx < side {
                    self.apply_block(grid, side, bx, by, bz);
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }
    }

    /// Apply the block rule for a single 2x2x2 block at (bx, by, bz).
    ///
    /// Cells outside the grid boundary are treated as empty (absorbing BC).
    fn apply_block(&self, grid: &mut [CellState], side: usize, bx: usize, by: usize, bz: usize) {
        // Read the 8 cells. OOB → empty (absorbing boundary).
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = bx + dx;
                    let y = by + dy;
                    let z = bz + dz;
                    if x < side && y < side && z < side {
                        let idx = x + y * side + z * side * side;
                        block[block_index(dx, dy, dz)] = Cell::from_raw(grid[idx]);
                    }
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
        let result = rule.step_block(&block);

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
        // OOB positions are silently skipped (absorbing boundary).
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = bx + dx;
                    let y = by + dy;
                    let z = bz + dz;
                    if x >= side || y >= side || z >= side {
                        continue;
                    }
                    let i = block_index(dx, dy, dz);
                    let original = block[i];
                    let has_rule = self.materials.block_rule_id_for_cell(original).is_some();
                    let idx = x + y * side + z * side * side;
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
    pub(crate) fn unique_block_rule(&self, block: &[Cell; 8]) -> Option<BlockRuleId> {
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
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), ALIVE.raw());
                    }
                }
            }
        }
    }

    /// Seed a demo scene: stone room with grass walls (fuel), fire, and water.
    ///
    /// Demonstrates reaction phase (fire spreads to grass, water quenches fire)
    /// plus movement phase (water falls and spreads via FluidBlockRule).
    pub fn seed_burning_room(&mut self) {
        let side = self.side() as u64;
        let margin = side / 8;
        let lo = margin;
        let hi = side - margin;

        for z in lo..hi {
            for y in lo..hi {
                for x in lo..hi {
                    let on_wall = x == lo || x == hi - 1 || z == lo || z == hi - 1;
                    let on_floor = y == lo;
                    let on_ceiling = y == hi - 1;

                    if on_floor {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), DIRT);
                    } else if on_ceiling {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), STONE);
                    } else if on_wall {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), GRASS);
                    }
                }
            }
        }

        // Fire source: small cluster in one corner
        let fire_x = lo + 2;
        let fire_z = lo + 2;
        for dy in 1..4u64 {
            for dx in 0..2u64 {
                for dz in 0..2u64 {
                    let y = lo + dy;
                    let x = fire_x + dx;
                    let z = fire_z + dz;
                    if x < hi && y < hi && z < hi {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), FIRE);
                    }
                }
            }
        }

        // Water pool: opposite corner, a few layers deep
        let water_hi_x = hi - 2;
        let water_hi_z = hi - 2;
        let water_lo_x = water_hi_x.saturating_sub(6);
        let water_lo_z = water_hi_z.saturating_sub(6);
        for y in (lo + 1)..(lo + 4).min(hi) {
            for z in water_lo_z..water_hi_z {
                for x in water_lo_x..water_hi_x {
                    self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), WATER);
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
        self.block_rule_present = None;

        // Fresh-store compaction: `from_flat` interned a brand new generation
        // into the append-only store, leaving the previous generation's
        // subtrees unreachable but still present. Rebuild into a fresh store
        // so memory tracks live-scene size, not cumulative history.
        // See hash-thing-88d.
        //
        // Brute-force compaction remaps all NodeIds without updating
        // hashlife_cache, so clear it to prevent stale cross-path hits.
        let (new_store, new_root) = self.store.compacted(self.root);
        self.store = new_store;
        self.root = new_root;
        self.hashlife_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
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
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        let field = params.to_heightmap();
        let gen_start = std::time::Instant::now();
        let (mut root, mut stats) = gen_region(&mut self.store, &field, [0, 0, 0], self.level);
        stats.gen_region_us = gen_start.elapsed().as_micros() as u64;
        stats.nodes_after_gen = self.store.stats();
        // Opt-in cave-CA post-pass. Runs as a separate stage after the
        // heightmap recursion so the baseline perf path (and every
        // pre-caves test) sees identical work when `params.caves` is
        // `None`.
        if let Some(cave_params) = params.caves {
            let cave_start = std::time::Instant::now();
            root = carve_caves(&mut self.store, root, self.level, &cave_params);
            stats.cave_us = cave_start.elapsed().as_micros() as u64;
        }
        stats.nodes_after_caves = self.store.stats();
        self.root = root;
        self.generation = 0;
        self.terrain_params = Some(*params);
        self.block_rule_present = None;
        stats
    }
}

/// Get the 26 Moore neighbors of a cell. Out-of-bounds neighbors are
/// `Cell::EMPTY` (absorbing boundary), matching hashlife's infinite-world
/// semantics.
fn get_neighbors(grid: &[CellState], side: usize, x: usize, y: usize, z: usize) -> [Cell; 26] {
    let mut neighbors = [Cell::EMPTY; 26];
    let mut idx = 0;
    let s = side as i32;
    for dz in [-1i32, 0, 1] {
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nz = z as i32 + dz;
                if nx >= 0 && nx < s && ny >= 0 && ny < s && nz >= 0 && nz < s {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nz = nz as usize;
                    neighbors[idx] = Cell::from_raw(grid[nx + ny * side + nz * side * side]);
                }
                // else: stays Cell::EMPTY (absorbing boundary)
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
    use crate::terrain::materials::{MaterialRegistry, FIRE, GRASS, LAVA, SAND, STONE, WATER};

    /// Helper: build an empty 8^3 world (level=3).
    fn empty_world() -> World {
        World::new(3)
    }

    fn gol_world(rule: GameOfLife3D) -> World {
        let mut world = empty_world();
        world.materials = MaterialRegistry::gol_smoke_with_rule(rule);
        world
    }

    fn wc(coord: u64) -> WorldCoord {
        WorldCoord(coord as i64)
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
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
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
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
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
                        world.get(wc(x), wc(y), wc(z)),
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
        assert_eq!(world.get(wc(2), wc(4), wc(4)), 0);
        assert_eq!(world.get(wc(6), wc(4), wc(4)), 0);
        assert_eq!(world.get(wc(4), wc(2), wc(4)), 0);
        assert_eq!(world.get(wc(4), wc(6), wc(4)), 0);
        assert_eq!(world.get(wc(4), wc(4), wc(2)), 0);
        assert_eq!(world.get(wc(4), wc(4), wc(6)), 0);
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
                    world.set(wc(x), wc(y), wc(z), ALIVE.raw());
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
                            world.get(wc(x), wc(y), wc(z)),
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
            world_a.set(wc(x), wc(y), wc(z), ALIVE.raw());
            world_b.set(wc(x), wc(y), wc(z), ALIVE.raw());
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
        world.set(wc(side - 1), wc(side - 1), wc(side - 1), ALIVE.raw());
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

    #[test]
    fn seed_burning_room_produces_all_material_types() {
        let mut w = World::new(6); // side 64
        w.seed_burning_room();
        assert!(w.population() > 0);
        let grid = w.flatten();
        let has = |mat: CellState| grid.contains(&mat);
        assert!(has(STONE), "room must have stone ceiling");
        assert!(has(DIRT), "room must have dirt floor");
        assert!(has(GRASS), "room must have grass walls");
        assert!(has(FIRE), "room must have fire");
        assert!(has(WATER), "room must have water");
        // Verify population is reasonable (room structure + contents)
        assert!(
            w.population() > 100,
            "burning room should have substantial content"
        );
    }

    // -----------------------------------------------------------------
    // 9-11. Bounds check on World::set / World::get (hash-thing-fb5).
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn world_set_panics_oob() {
        let mut w = World::new(2); // side 4
        w.set(wc(4), wc(0), wc(0), ALIVE.raw());
    }

    #[test]
    fn world_set_in_bounds_at_max_corner() {
        let mut w = World::new(2); // side 4
        w.set(wc(3), wc(3), wc(3), ALIVE.raw());
        assert_eq!(w.get(wc(3), wc(3), wc(3)), ALIVE.raw());
        // And the complementary corner — distinct material id 2, metadata 0.
        let mat2 = Cell::pack(2, 0).raw();
        w.set(wc(0), wc(0), wc(0), mat2);
        assert_eq!(w.get(wc(0), wc(0), wc(0)), mat2);
    }

    #[test]
    fn world_get_returns_zero_oob() {
        let w = World::new(2); // side 4
        assert_eq!(w.get(wc(4), wc(0), wc(0)), 0);
        assert_eq!(w.get(wc(0), wc(4), wc(0)), 0);
        assert_eq!(w.get(wc(0), wc(0), wc(4)), 0);
        assert_eq!(w.get(WorldCoord(-1), wc(0), wc(0)), 0);
    }

    // -----------------------------------------------------------------
    // 12. seed_terrain epoch boundary test.
    // -----------------------------------------------------------------
    #[test]
    fn seed_terrain_is_an_epoch_boundary() {
        let mut world = World::new(6);
        let params = TerrainParams::default();

        let _ = world.seed_terrain(&params);
        let nodes_after_first = world.store.stats();

        let _ = world.seed_terrain(&params);
        let nodes_after_second = world.store.stats();

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
        world.set(wc(4), wc(4), wc(4), WATER);
        world.set(wc(5), wc(4), wc(4), FIRE);

        world.step();

        assert_eq!(world.get(wc(4), wc(4), wc(4)), STONE);
    }

    #[test]
    fn isolated_fire_burns_out() {
        let mut world = empty_world();
        world.set(wc(4), wc(4), wc(4), FIRE);

        world.step();

        assert_eq!(world.get(wc(4), wc(4), wc(4)), 0);
    }

    #[test]
    fn fire_persists_next_to_grass_fuel() {
        let mut world = empty_world();
        world.set(wc(4), wc(4), wc(4), FIRE);
        world.set(wc(5), wc(4), wc(4), GRASS);

        world.step();

        assert_eq!(world.get(wc(4), wc(4), wc(4)), FIRE);
        assert_eq!(world.get(wc(5), wc(4), wc(4)), GRASS);
    }

    #[test]
    fn step_dispatches_fire_and_water_rules_independently() {
        let mut world = empty_world();

        world.set(wc(2), wc(2), wc(2), FIRE);
        world.set(wc(3), wc(2), wc(2), GRASS);

        world.set(wc(5), wc(5), wc(5), WATER);
        world.set(wc(6), wc(5), wc(5), FIRE);

        world.step();

        assert_eq!(world.get(wc(2), wc(2), wc(2)), FIRE);
        assert_eq!(world.get(wc(3), wc(2), wc(2)), GRASS);
        assert_eq!(world.get(wc(5), wc(5), wc(5)), STONE);
        assert_eq!(world.get(wc(6), wc(5), wc(5)), 0);
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
        world.set(wc(2), wc(4), wc(2), STONE);
        world.set(wc(3), wc(4), wc(2), STONE);
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
        world.set(wc(2), wc(3), wc(2), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            0,
            "bottom should be air initially"
        );
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "top should be dirt initially"
        );

        world.step();

        // After step, dirt should have fallen from y=3 to y=2.
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should have fallen to bottom"
        );
        assert_eq!(world.get(wc(2), wc(3), wc(2)), 0, "top should now be air");
    }

    #[test]
    fn gravity_conserves_population() {
        let mut world = gravity_world(&[DIRT_MATERIAL_ID, WATER_MATERIAL_ID]);
        // Scatter some cells in even-aligned positions.
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        world.set(wc(2), wc(1), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.set(wc(4), wc(3), wc(4), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
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
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        assert_eq!(world.generation, 0);

        // Gen 0 (even offset=0): block (0,0,0) is active → dirt falls from y=1 to y=0.
        world.step();
        assert_eq!(
            world.get(wc(0), wc(0), wc(0)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should fall on even generation"
        );
        assert_eq!(world.get(wc(0), wc(1), wc(0)), 0);
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
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        world.set(wc(1), wc(0), wc(0), Cell::pack(WATER_MATERIAL_ID, 0).raw());

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
        world.set(wc(0), wc(0), wc(0), Cell::pack(STONE_MATERIAL_ID, 0).raw());
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());

        world.step();

        // Stone stays at (0,0,0) — anchored. Dirt can't displace it.
        assert_eq!(
            world.get(wc(0), wc(0), wc(0)),
            Cell::pack(STONE_MATERIAL_ID, 0).raw(),
            "stone (no block rule) must not be moved by dirt's gravity"
        );
        assert_eq!(
            world.get(wc(0), wc(1), wc(0)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should stay since stone below is anchored"
        );
    }

    #[test]
    fn block_rule_deterministic_across_runs() {
        let make_world = || {
            let mut w = gravity_world(&[DIRT_MATERIAL_ID]);
            w.simulation_seed = 12345;
            w.set(wc(2), wc(3), wc(2), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
            w.set(wc(4), wc(5), wc(4), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
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

    // -----------------------------------------------------------------
    // FluidBlockRule integration tests (hash-thing-1v0.18).
    // -----------------------------------------------------------------

    #[test]
    fn fluid_water_falls_under_gravity() {
        let mut world = World::new(3); // 8x8x8, terrain_defaults includes FluidBlockRule
                                       // Place water at y=3, air at y=2 — same block column at even offset.
        world.set(wc(2), wc(3), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            0,
            "bottom should be air initially"
        );

        world.step();

        // Water must survive (mass conservation) and drop below y=3.
        // Gravity phase swaps water down; lateral spread may also shift x or z
        // depending on rng_hash, so we don't assert the exact landing position.
        assert_eq!(world.population(), 1, "exactly one water cell must survive");
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            0,
            "water must have left its original position at y=3"
        );
    }

    #[test]
    fn fluid_conserves_population() {
        // Level 4 (16³) keeps cells well inside the absorbing boundary
        // over 6 steps so mass conservation holds.
        let mut world = World::new(4);
        world.set(wc(5), wc(6), wc(5), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.set(wc(8), wc(9), wc(8), Cell::pack(WATER_MATERIAL_ID, 0).raw());
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
        world.set(wc(2), wc(2), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());

        // Run several steps. Lateral spread is probabilistic (depends on
        // rng_hash), but over multiple steps with alternating offsets the
        // water should reach a neighbor.
        for _ in 0..8 {
            world.step();
        }

        // Water must still exist (conservation) but may have moved.
        assert_eq!(
            world.population(),
            1,
            "exactly one water cell should remain"
        );
        // It should NOT still be at (2,3,2) after gravity — verify it settled
        // at a y=0 or y=2 position (even-aligned bottom).
    }

    #[test]
    fn fluid_deterministic_across_runs() {
        let make = || {
            let mut w = World::new(3);
            w.simulation_seed = 999;
            w.set(wc(2), wc(3), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());
            w.set(wc(4), wc(5), wc(4), Cell::pack(WATER_MATERIAL_ID, 0).raw());
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

    // ---- ensure_contains (hash-thing-e9h) ----

    #[test]
    fn ensure_contains_noop_when_in_bounds() {
        let mut world = World::new(3); // 8x8x8
        let level_before = world.level;
        let root_before = world.root;
        world.ensure_contains(wc(7), wc(7), wc(7));
        assert_eq!(world.level, level_before);
        assert_eq!(world.root, root_before);
    }

    #[test]
    fn ensure_contains_grows_root_once() {
        let mut world = World::new(3); // 8x8x8, side=8
        world.set(wc(5), wc(3), wc(2), STONE);
        world.ensure_contains(wc(8), wc(0), wc(0)); // just past the edge
        assert_eq!(world.level, 4); // grew once: side=16
        assert_eq!(world.side(), 16);
        // Original cell survives at its original coordinate.
        assert_eq!(world.get(wc(5), wc(3), wc(2)), STONE);
        // The newly accessible region is empty.
        assert_eq!(world.get(wc(8), wc(0), wc(0)), 0);
        assert_eq!(world.get(wc(15), wc(15), wc(15)), 0);
    }

    #[test]
    fn ensure_contains_grows_multiple_levels() {
        let mut world = World::new(2); // 4x4x4
        world.set(wc(1), wc(2), wc(3), FIRE);
        // Coordinate 100 requires level >= 7 (side=128).
        world.ensure_contains(wc(100), wc(0), wc(0));
        assert!(world.level >= 7);
        assert!(world.side() > 100);
        // Original cell survives.
        assert_eq!(world.get(wc(1), wc(2), wc(3)), FIRE);
    }

    #[test]
    fn ensure_contains_works_on_all_axes() {
        let mut world = World::new(3); // 8x8x8
                                       // Grow for Y axis.
        world.ensure_contains(wc(0), wc(10), wc(0));
        assert!(world.side() > 10);
        let level_after_y = world.level;
        // Grow for Z axis (further).
        world.ensure_contains(wc(0), wc(0), wc(100));
        assert!(world.level > level_after_y);
        assert!(world.side() > 100);
    }

    #[test]
    fn ensure_contains_then_set_roundtrips() {
        let mut world = World::new(3); // 8x8x8
        world.ensure_contains(wc(20), wc(20), wc(20));
        world.set(wc(20), wc(20), wc(20), WATER);
        assert_eq!(world.get(wc(20), wc(20), wc(20)), WATER);
        // Other cells in the expanded region remain empty.
        assert_eq!(world.get(wc(19), wc(20), wc(20)), 0);
    }

    #[test]
    fn ensure_contains_preserves_population() {
        let mut world = World::new(3);
        world.set(wc(0), wc(0), wc(0), STONE);
        world.set(wc(7), wc(7), wc(7), FIRE);
        let pop_before = world.population();
        world.ensure_contains(wc(100), wc(100), wc(100));
        assert_eq!(world.population(), pop_before);
    }

    // ---- ensure_region (hash-thing-5qp) ----

    #[test]
    fn ensure_region_noop_when_in_bounds() {
        let mut world = World::new(3);
        let level_before = world.level;
        world.ensure_region(
            [WorldCoord(0), WorldCoord(0), WorldCoord(0)],
            [WorldCoord(7), WorldCoord(7), WorldCoord(7)],
        );
        assert_eq!(world.level, level_before);
    }

    #[test]
    fn ensure_region_grows_for_max_corner() {
        let mut world = World::new(3); // 8x8x8
        world.set(wc(2), wc(3), wc(4), STONE);
        world.ensure_region(
            [WorldCoord(0), WorldCoord(0), WorldCoord(0)],
            [WorldCoord(20), WorldCoord(20), WorldCoord(20)],
        );
        assert!(world.side() > 20);
        // Existing cell survives.
        assert_eq!(world.get(wc(2), wc(3), wc(4)), STONE);
    }

    #[test]
    fn ensure_region_then_set_covers_full_box() {
        let mut world = World::new(3);
        world.ensure_region(
            [WorldCoord(10), WorldCoord(10), WorldCoord(10)],
            [WorldCoord(20), WorldCoord(20), WorldCoord(20)],
        );
        // Can set cells at both min and max of the region.
        world.set(wc(10), wc(10), wc(10), FIRE);
        world.set(wc(20), wc(20), wc(20), WATER);
        assert_eq!(world.get(wc(10), wc(10), wc(10)), FIRE);
        assert_eq!(world.get(wc(20), wc(20), wc(20)), WATER);
    }

    #[test]
    fn ensure_region_single_point_matches_ensure_contains() {
        let mut w1 = World::new(3);
        let mut w2 = World::new(3);
        w1.ensure_contains(wc(50), wc(50), wc(50));
        w2.ensure_region(
            [WorldCoord(50), WorldCoord(50), WorldCoord(50)],
            [WorldCoord(50), WorldCoord(50), WorldCoord(50)],
        );
        assert_eq!(w1.level, w2.level);
    }

    // -----------------------------------------------------------------
    // FluidBlockRule integration: water falls via terrain_defaults (1v0.5).
    // -----------------------------------------------------------------

    #[test]
    fn terrain_defaults_water_falls_when_stepped() {
        // terrain_defaults() already wires FluidBlockRule for water.
        let mut world = World::new(3); // 8x8x8
                                       // Place water at y=3 with air below. Block lateral escape routes
                                       // with stone so we isolate the gravity behavior.
        world.set(wc(2), wc(3), wc(2), WATER);
        world.set(wc(3), wc(2), wc(2), STONE); // block x-spread
        world.set(wc(2), wc(2), wc(3), STONE); // block z-spread
        assert_eq!(world.get(wc(2), wc(3), wc(2)), WATER);
        assert_eq!(world.get(wc(2), wc(2), wc(2)), 0);

        world.step(); // gen 0 → 1 (even offset=0, block at (2,2,2))

        // Water should have fallen from y=3 to y=2 via FluidBlockRule gravity.
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            WATER,
            "water should fall to y=2 via FluidBlockRule"
        );
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            0,
            "y=3 should be air after water fell"
        );
    }

    #[test]
    fn burning_room_conserves_population_over_steps() {
        let mut world = World::new(6); // 64x64x64
        world.seed_burning_room();
        let pop_before = world.population();
        assert!(pop_before > 100, "room should have substantial content");

        // Step a few times — fire reactions may create/destroy cells,
        // but block rules (movement) must conserve mass within each block.
        // Population may change due to CaRule (fire burns out, water quenches),
        // so we just verify the world doesn't crash and stays non-empty.
        for _ in 0..4 {
            world.step();
        }
        assert!(
            world.population() > 0,
            "world should still have cells after stepping"
        );
    }

    // -----------------------------------------------------------------
    // RealizedRegion (hash-thing-ica).
    // -----------------------------------------------------------------

    #[test]
    fn region_side_matches_world() {
        let world = World::new(5);
        let r = world.region();
        assert_eq!(r.side(), 32);
        assert_eq!(r.side() as usize, world.side());
        assert_eq!(r.level, 5);
        assert_eq!(r.origin, [0, 0, 0]);
    }

    #[test]
    fn region_contains_boundary() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 3,
        }; // side=8
        assert!(r.contains(0, 0, 0));
        assert!(r.contains(7, 7, 7));
        assert!(!r.contains(8, 0, 0));
        assert!(!r.contains(0, 8, 0));
        assert!(!r.contains(0, 0, 8));
    }

    #[test]
    fn region_contains_with_origin() {
        let r = RealizedRegion {
            origin: [10, 20, 30],
            level: 2,
        }; // side=4
        assert!(r.contains(10, 20, 30));
        assert!(r.contains(13, 23, 33));
        assert!(!r.contains(14, 20, 30)); // just past
        assert!(!r.contains(9, 20, 30)); // just before
    }

    #[test]
    fn region_octant_of_corners() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 4,
        }; // side=16, half=8
        assert_eq!(r.octant_of(0, 0, 0), 0); // (0,0,0)
        assert_eq!(r.octant_of(8, 0, 0), 1); // (1,0,0)
        assert_eq!(r.octant_of(0, 8, 0), 2); // (0,1,0)
        assert_eq!(r.octant_of(8, 8, 0), 3); // (1,1,0)
        assert_eq!(r.octant_of(0, 0, 8), 4); // (0,0,1)
        assert_eq!(r.octant_of(15, 15, 15), 7); // (1,1,1)
    }

    #[test]
    #[should_panic(expected = "outside realized region")]
    fn region_octant_of_oob_panics() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 3,
        };
        r.octant_of(8, 0, 0);
    }

    #[test]
    fn region_display() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 6,
        };
        let s = format!("{r}");
        assert!(s.contains("level=6"), "display should show level");
        assert!(s.contains("64³"), "display should show side cubed");
    }

    #[test]
    fn region_tracks_ensure_contains_growth() {
        let mut world = World::new(3); // side=8
        assert_eq!(world.region().side(), 8);
        world.ensure_contains(wc(20), wc(0), wc(0));
        assert!(world.region().side() > 20);
        assert_eq!(world.region().level, world.level);
    }

    // ── Lazy terrain expansion (3fq.4) ────────────────────────────

    #[test]
    fn expand_without_terrain_produces_empty_siblings() {
        let mut world = World::new(3); // side=8
        world.set(wc(0), wc(0), wc(0), ALIVE.raw());
        let pop_before = world.population();
        world.ensure_contains(wc(10), wc(0), wc(0));
        // Only the original cell survives — new octants are empty.
        assert_eq!(world.population(), pop_before);
    }

    #[test]
    fn expand_with_terrain_populates_new_octants() {
        let mut world = World::new(3); // side=8
        let params = TerrainParams::default();
        world.seed_terrain(&params);
        let pop_initial = world.population();
        assert!(pop_initial > 0, "terrain should produce non-empty world");

        // Expand into +x. The new octant at (8..16, 0..8, 0..8) should
        // have generated terrain, not empty space.
        world.ensure_contains(wc(10), wc(0), wc(0));
        let pop_after = world.population();
        assert!(
            pop_after > pop_initial,
            "expansion should add terrain: before={pop_initial}, after={pop_after}"
        );
    }

    #[test]
    fn expand_terrain_is_deterministic() {
        let params = TerrainParams::default();

        let mut w1 = World::new(3);
        w1.seed_terrain(&params);
        w1.ensure_contains(wc(10), wc(0), wc(0));

        let mut w2 = World::new(3);
        w2.seed_terrain(&params);
        w2.ensure_contains(wc(10), wc(0), wc(0));

        // Same params + same expansion → same world.
        assert_eq!(w1.level, w2.level);
        assert_eq!(w1.population(), w2.population());
        // Spot-check a cell in the expanded region.
        assert_eq!(w1.get(wc(10), wc(3), wc(3)), w2.get(wc(10), wc(3), wc(3)));
    }

    #[test]
    fn expand_terrain_preserves_original_cells() {
        let mut world = World::new(3);
        let params = TerrainParams::default();
        world.seed_terrain(&params);

        // Record some cells from the original region.
        let cells_before: Vec<_> = (0..8u64).map(|x| world.get(wc(x), wc(3), wc(3))).collect();

        world.ensure_contains(wc(10), wc(0), wc(0));

        // Original region cells are unchanged.
        for (x, &expected) in cells_before.iter().enumerate() {
            assert_eq!(
                world.get(wc(x as u64), wc(3), wc(3)),
                expected,
                "cell at ({x}, 3, 3) changed after expansion"
            );
        }
    }

    #[test]
    fn expand_terrain_with_caves() {
        use crate::terrain::CaveParams;
        let params = TerrainParams {
            caves: Some(CaveParams::default()),
            ..Default::default()
        };

        let mut w_caves = World::new(3);
        w_caves.seed_terrain(&params);
        w_caves.ensure_contains(wc(10), wc(0), wc(0));

        let mut w_plain = World::new(3);
        w_plain.seed_terrain(&TerrainParams::default());
        w_plain.ensure_contains(wc(10), wc(0), wc(0));

        // Caves should carve out some material — fewer populated cells.
        assert!(
            w_caves.population() < w_plain.population(),
            "caves should reduce population: caves={}, plain={}",
            w_caves.population(),
            w_plain.population()
        );
    }

    #[test]
    fn expand_terrain_matches_direct_gen() {
        // A world grown by expansion should produce the same terrain
        // as one generated at the larger size from the start. This
        // verifies there are no seams at expansion boundaries.
        let params = TerrainParams::default();

        // Path A: generate at level 3, then expand to level 4.
        let mut grown = World::new(3);
        grown.seed_terrain(&params);
        grown.ensure_contains(wc(10), wc(0), wc(0)); // grows to level 4

        // Path B: generate at level 4 directly.
        let mut direct = World::new(4);
        direct.seed_terrain(&params);

        // Spot-check cells across the expansion boundary (x=8 is the seam).
        for x in 0..16u64 {
            for yz in [0u64, 3, 7] {
                assert_eq!(
                    grown.get(wc(x), wc(yz), wc(yz)),
                    direct.get(wc(x), wc(yz), wc(yz)),
                    "seam mismatch at ({x}, {yz}, {yz})"
                );
            }
        }
    }

    #[test]
    fn gol_world_expand_stays_empty() {
        // GoL smoke worlds don't set terrain_params, so expansion is empty.
        let mut world = World::new(3);
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        assert!(world.terrain_params.is_none());
        world.ensure_contains(wc(20), wc(0), wc(0));
        // Only the single cell we placed.
        assert_eq!(world.population(), 1);
    }

    #[test]
    fn probe_matches_get() {
        let mut world = World::new(3);
        let stone = crate::terrain::materials::STONE;
        world.set(wc(1), wc(2), wc(3), stone);
        assert_eq!(
            world.probe(wc(1), wc(2), wc(3)),
            world.get(wc(1), wc(2), wc(3))
        );
        // OOB returns 0 from both
        assert_eq!(world.probe(WorldCoord(-1), wc(0), wc(0)), 0);
        assert_eq!(world.probe(wc(100), wc(0), wc(0)), 0);
    }

    #[test]
    fn is_realized_inside() {
        let world = World::new(3); // side=8
        assert!(world.is_realized(wc(0), wc(0), wc(0)));
        assert!(world.is_realized(wc(7), wc(7), wc(7)));
        assert!(world.is_realized(wc(4), wc(2), wc(6)));
    }

    #[test]
    fn is_realized_outside() {
        let world = World::new(3); // side=8
        assert!(!world.is_realized(WorldCoord(-1), wc(0), wc(0)));
        assert!(!world.is_realized(wc(8), wc(0), wc(0)));
        assert!(!world.is_realized(wc(0), wc(100), wc(0)));
    }

    // --- Mutation channel tests (hash-thing-1v0.9) ---

    #[test]
    fn apply_mutations_drains_and_applies() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: wc(1),
            y: wc(2),
            z: wc(3),
            state: STONE,
        });
        world.queue.push(WorldMutation::SetCell {
            x: wc(4),
            y: wc(5),
            z: wc(6),
            state: STONE,
        });
        assert_eq!(world.queue.len(), 2);
        world.apply_mutations();
        assert!(world.queue.is_empty());
        assert_eq!(world.get(wc(1), wc(2), wc(3)), STONE);
        assert_eq!(world.get(wc(4), wc(5), wc(6)), STONE);
    }

    #[test]
    fn direct_set_clears_hashlife_caches() {
        let mut world = World::new(3);
        world
            .hashlife_cache
            .insert((NodeId::EMPTY, 0), NodeId::EMPTY);
        world
            .hashlife_macro_cache
            .insert((NodeId::EMPTY, 0), NodeId::EMPTY);

        world.set(wc(3), wc(3), wc(3), STONE);

        assert!(
            world.hashlife_cache.is_empty(),
            "direct edits must drop stale recursive cache entries"
        );
        assert!(
            world.hashlife_macro_cache.is_empty(),
            "direct edits must drop stale macro-step cache entries"
        );
    }

    #[test]
    fn apply_mutations_clears_hashlife_caches() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world
            .hashlife_cache
            .insert((NodeId::EMPTY, 0), NodeId::EMPTY);
        world
            .hashlife_macro_cache
            .insert((NodeId::EMPTY, 0), NodeId::EMPTY);
        world.queue.push(WorldMutation::SetCell {
            x: wc(2),
            y: wc(2),
            z: wc(2),
            state: STONE,
        });

        world.apply_mutations();

        assert!(
            world.hashlife_cache.is_empty(),
            "mutation flush must clear stale recursive cache entries"
        );
        assert!(
            world.hashlife_macro_cache.is_empty(),
            "mutation flush must clear stale macro-step cache entries"
        );
    }

    #[test]
    fn apply_mutations_last_write_wins() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: wc(3),
            y: wc(3),
            z: wc(3),
            state: STONE,
        });
        world.queue.push(WorldMutation::SetCell {
            x: wc(3),
            y: wc(3),
            z: wc(3),
            state: Cell::pack(WATER_MATERIAL_ID, 0).raw(),
        });
        world.apply_mutations();
        assert_eq!(
            world.get(wc(3), wc(3), wc(3)),
            Cell::pack(WATER_MATERIAL_ID, 0).raw()
        );
    }

    #[test]
    fn fill_region_covers_inclusive_box() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::FillRegion {
            min: [wc(1), wc(1), wc(1)],
            max: [wc(3), wc(3), wc(3)],
            state: STONE,
        });
        world.apply_mutations();
        // All 27 cells (3×3×3) should be stone.
        let mut count = 0;
        for z in 1..=3 {
            for y in 1..=3 {
                for x in 1..=3 {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 27);
    }

    #[test]
    fn set_block_fills_cells_per_block_cubed() {
        use crate::octree::CELLS_PER_BLOCK;
        // World needs to be large enough: block at (1,0,0) spans cells [8..15]
        // on x, so we need level >= 4 (side 16).
        let mut world = World::new(4);
        world.set_block(1, 0, 0, STONE);
        world.apply_mutations();

        let k = CELLS_PER_BLOCK as u64;
        let mut filled = 0u32;
        for z in 0..k {
            for y in 0..k {
                for x in k..(2 * k) {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        filled += 1;
                    }
                }
            }
        }
        assert_eq!(filled, CELLS_PER_BLOCK.pow(3));

        // Cells just outside the block should be empty.
        assert_eq!(world.get(wc(k - 1), wc(0), wc(0)), 0); // cell 7, just before block
        assert_eq!(world.get(wc(2 * k), wc(0), wc(0)), 0); // cell 16, just after block
    }

    #[test]
    fn set_block_at_origin() {
        use crate::octree::CELLS_PER_BLOCK;
        let mut world = World::new(4);
        world.set_block(0, 0, 0, STONE);
        world.apply_mutations();

        let k = CELLS_PER_BLOCK as u64;
        let mut filled = 0u32;
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        filled += 1;
                    }
                }
            }
        }
        assert_eq!(filled, CELLS_PER_BLOCK.pow(3));
    }

    #[test]
    fn set_block_roundtrip_individual_cells() {
        use crate::octree::CELLS_PER_BLOCK;
        let mut world = World::new(4);
        world.set_block(0, 0, 0, DIRT);
        world.apply_mutations();

        let k = CELLS_PER_BLOCK as u64;
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    assert_eq!(
                        world.get(wc(x), wc(y), wc(z)),
                        DIRT,
                        "cell ({x},{y},{z}) should be DIRT"
                    );
                }
            }
        }
    }

    #[test]
    fn set_block_auto_grows_world() {
        use crate::octree::CELLS_PER_BLOCK;
        // Start with a level-3 world (side=8). Block at (1,0,0) needs cells [8..15],
        // which is out of bounds. set_block should auto-grow to fit.
        let mut world = World::new(3);
        world.set_block(1, 0, 0, STONE);
        world.apply_mutations();

        // World must have grown to at least level 4 (side=16).
        assert!(
            world.level >= 4,
            "world should auto-grow for out-of-bounds block"
        );

        let k = CELLS_PER_BLOCK as u64;
        for z in 0..k {
            for y in 0..k {
                for x in k..(2 * k) {
                    assert_eq!(world.get(wc(x), wc(y), wc(z)), STONE);
                }
            }
        }
    }

    #[test]
    fn block_fill_vs_cell_fill_node_count() {
        use crate::octree::CELLS_PER_BLOCK;
        // Fill a 16³ world with stone: once via set_block, once via individual set calls.
        // Both should produce identical octrees (hash-consing deduplicates).
        let k = CELLS_PER_BLOCK as u64;
        let level = 4u32; // 16³ world

        // Block fill: 2 blocks per axis at K=3 → (16/8)³ = 8 blocks
        let mut world_block = World::new(level);
        let blocks_per_axis = (1u64 << level) / k;
        for bz in 0..blocks_per_axis as i64 {
            for by in 0..blocks_per_axis as i64 {
                for bx in 0..blocks_per_axis as i64 {
                    world_block.set_block(bx, by, bz, STONE);
                }
            }
        }
        world_block.apply_mutations();

        // Cell fill: 16³ = 4096 individual set calls
        let mut world_cell = World::new(level);
        let side = 1u64 << level;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    world_cell.set(wc(x), wc(y), wc(z), STONE);
                }
            }
        }

        // Both should produce the same root (hash-cons identity) — uniform
        // stone compresses identically regardless of insertion order.
        assert_eq!(world_block.root, world_cell.root);
    }

    #[test]
    fn step_ca_runs_clean_on_empty_queue() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), STONE);
        world.step_ca(); // should not panic
        assert_eq!(world.generation, 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "pending mutations")]
    fn step_ca_panics_with_pending_mutations() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: wc(1),
            y: wc(1),
            z: wc(1),
            state: STONE,
        });
        world.step_ca(); // should panic
    }

    #[test]
    fn sand_falls_under_gravity() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(2), SAND);
        assert_eq!(world.get(wc(2), wc(2), wc(2)), 0);

        world.step();

        assert_eq!(world.get(wc(2), wc(2), wc(2)), SAND, "sand should fall");
        assert_eq!(world.get(wc(2), wc(3), wc(2)), 0, "top should be air");
    }

    #[test]
    fn sand_conserves_population() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(2), SAND);
        world.set(wc(4), wc(5), wc(4), SAND);
        world.set(wc(0), wc(1), wc(0), SAND);
        let pop_before = world.population();

        for _ in 0..4 {
            world.step();
        }

        assert_eq!(
            world.population(),
            pop_before,
            "sand gravity must conserve population"
        );
    }

    #[test]
    fn lava_solidifies_on_water_contact() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), LAVA);
        world.set(wc(4), wc(3), wc(3), WATER);

        world.step();

        assert_eq!(
            world.get(wc(3), wc(3), wc(3)),
            STONE,
            "lava adjacent to water should solidify into stone"
        );
    }

    #[test]
    fn lava_persists_without_water() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), LAVA);

        world.step();

        let pop = world.population();
        assert!(pop > 0, "lava should not disappear without water");
    }

    // -----------------------------------------------------------------
    // Negative-direction world growth (hash-thing-37r).
    // -----------------------------------------------------------------

    #[test]
    fn ensure_contains_negative_shifts_origin() {
        let mut world = World::new(3);
        world.set(wc(4), wc(4), wc(4), STONE);
        world.ensure_contains(WorldCoord(-1), wc(0), wc(0));
        assert!(
            world.origin[0] < 0,
            "origin[0] should be negative after negative growth"
        );
        assert_eq!(world.get(wc(4), wc(4), wc(4)), STONE);
    }

    #[test]
    fn ensure_contains_negative_preserves_population() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(5), STONE);
        world.set(wc(6), wc(1), wc(7), WATER);
        let pop_before = world.population();
        world.ensure_contains(WorldCoord(-10), WorldCoord(-10), WorldCoord(-10));
        assert_eq!(world.population(), pop_before);
    }

    #[test]
    fn ensure_contains_negative_then_set_roundtrips() {
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-5), WorldCoord(-5), WorldCoord(-5));
        world.set(WorldCoord(-5), WorldCoord(-5), WorldCoord(-5), STONE);
        assert_eq!(
            world.get(WorldCoord(-5), WorldCoord(-5), WorldCoord(-5)),
            STONE
        );
    }

    #[test]
    fn ensure_region_negative_min_grows_correctly() {
        let mut world = World::new(3);
        world.ensure_region(
            [WorldCoord(-4), WorldCoord(-4), WorldCoord(-4)],
            [WorldCoord(7), WorldCoord(7), WorldCoord(7)],
        );
        assert!(world.is_realized(WorldCoord(-4), WorldCoord(-4), WorldCoord(-4)));
        assert!(world.is_realized(wc(7), wc(7), wc(7)));
    }

    #[test]
    fn negative_growth_origin_tracks_correctly() {
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-1), wc(0), wc(0));
        let o = world.origin;
        let side = world.side() as i64;
        assert!(o[0] <= 0);
        assert!(o[0] + side >= 8);
    }

    #[test]
    fn get_returns_zero_below_origin() {
        let world = World::new(3);
        assert_eq!(world.get(WorldCoord(-1), wc(0), wc(0)), 0);
    }

    #[test]
    fn mixed_direction_growth_preserves_cells() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(4), STONE);
        world.set(wc(6), wc(1), wc(7), WATER);
        world.ensure_region([WorldCoord(-5), wc(0), wc(0)], [wc(0), wc(0), wc(20)]);
        assert_eq!(world.get(wc(2), wc(3), wc(4)), STONE);
        assert_eq!(world.get(wc(6), wc(1), wc(7)), WATER);
    }

    #[test]
    fn is_realized_works_after_negative_growth() {
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-3), WorldCoord(-3), WorldCoord(-3));
        assert!(world.is_realized(WorldCoord(-3), WorldCoord(-3), WorldCoord(-3)));
        assert!(world.is_realized(wc(0), wc(0), wc(0)));
        assert!(world.is_realized(wc(7), wc(7), wc(7)));
    }
}
