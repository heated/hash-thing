use super::rule::CaRule;
use crate::octree::{CellState, NodeId, NodeStore};
use crate::rng::cell_rand_bool;
use crate::terrain::{gen_region, GenStats, TerrainParams};

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
}

impl World {
    /// Create a new empty world of given level (side = 2^level).
    pub fn new(level: u32) -> Self {
        let mut store = NodeStore::new();
        let root = store.empty(level);
        Self {
            store,
            root,
            level,
            generation: 0,
        }
    }

    pub fn side(&self) -> usize {
        1 << self.level
    }

    /// Set a cell.
    pub fn set(&mut self, x: u64, y: u64, z: u64, state: CellState) {
        self.root = self.store.set_cell(self.root, x, y, z, state);
    }

    /// Get a cell.
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
        let mut next = vec![0u8; side * side * side];

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

    /// Place a random spherical seed pattern in the center of the world.
    ///
    /// Randomness is drawn from the stateless `cell_rand_bool(x, y, z, ...)`
    /// primitive in `rng.rs`, not a stream RNG. This makes the output a pure
    /// function of `(position, seed)` — independent of loop order, independent
    /// of which cells are visited first, independent of whether we ever
    /// parallelize this loop. That is the h34.2 determinism rule: no scan-
    /// order dependence for anything that reaches simulation state.
    pub fn seed_center(&mut self, radius: u64, density: f64) {
        const SEED: u64 = 0xD15C_05EE_5EED_5EED; // "disco see seed seed"
        let center = self.side() as u64 / 2;
        for z in (center - radius)..(center + radius) {
            for y in (center - radius)..(center + radius) {
                for x in (center - radius)..(center + radius) {
                    let dx = x as f64 - center as f64;
                    let dy = y as f64 - center as f64;
                    let dz = z as f64 - center as f64;
                    let inside_sphere =
                        dx * dx + dy * dy + dz * dz < (radius as f64 * radius as f64);
                    if inside_sphere
                        && cell_rand_bool(x as i64, y as i64, z as i64, 0, SEED, density)
                    {
                        self.set(x, y, z, 1);
                    }
                }
            }
        }
    }

    pub fn population(&self) -> u64 {
        self.store.population(self.root)
    }

    /// Replace the world with terrain generated from `params`. Starts a
    /// fresh `NodeStore` — every `NodeId` from the previous world is
    /// invalidated. Resets `generation` to 0.
    ///
    /// ## Fresh-store epoch
    ///
    /// `seed_terrain` is a new epoch in the same sense as `step_flat`'s
    /// post-tick `compacted()` call: everything from the previous world
    /// is unreachable after this returns, so there is no point interning
    /// the new terrain into a store full of garbage. Building into a
    /// fresh store is cheaper (no post-build compaction walk), keeps
    /// node counts honest, and makes the epoch boundary explicit.
    ///
    /// Under the closed-world invariant (see `metaverse-design-decision`)
    /// any caller holding a `NodeId` across this call is holding a
    /// dangling id; this matches the `compacted()` contract. Today no
    /// caller does — `World::root` is the only long-lived id holder and
    /// is updated in-place here.
    ///
    /// **The caller MUST keep the simulation paused around this call.** This
    /// method intentionally does NOT touch `paused`, because only the caller
    /// knows whether the current `CaRule` is terrain-safe. The legacy GoL
    /// rule, for example, will immediately treat every solid cell as alive
    /// and annihilate the terrain on the first step. Every in-tree call
    /// site today sets `paused = true` explicitly; keep it that way.
    pub fn seed_terrain(&mut self, params: &TerrainParams) -> GenStats {
        params.validate().expect("invalid TerrainParams");
        self.store = NodeStore::new();
        let field = params.to_heightmap();
        let (root, stats) = gen_region(&mut self.store, &field, [0, 0, 0], self.level);
        self.root = root;
        self.generation = 0;
        stats
    }
}

/// Get the 26 Moore neighbors of a cell, wrapping at boundaries.
fn get_neighbors(grid: &[CellState], side: usize, x: usize, y: usize, z: usize) -> [CellState; 26] {
    let mut neighbors = [0u8; 26];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainParams;

    /// Re-seeding terrain must not accumulate nodes from the previous
    /// world. `seed_terrain` is an epoch boundary in the same sense as
    /// `step_flat`'s `compacted()` tail — after it returns, nothing
    /// from the previous world is reachable, and the store's node
    /// count should reflect only the freshly built terrain.
    ///
    /// Regression guard for the fresh-store pattern: if someone ever
    /// reverts `self.store = NodeStore::new()` back to building into
    /// a polluted store, node counts grow on every re-seed and this
    /// test fires.
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
