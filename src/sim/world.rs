use super::rule::CaRule;
use crate::octree::{CellState, NodeId, NodeStore};
use crate::rng::cell_rand_bool;

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
    }

    /// Place a random seed pattern in the center of the world.
    ///
    /// Uses `cell_rand_bool` keyed on (x, y, z, generation=0, seed) so the
    /// result is a pure function of position and seed — no scan-order
    /// coupling. Identical subtrees seed identically, which keeps the
    /// initial world Hashlife-compatible.
    pub fn seed_center(&mut self, radius: u64, density: f64, seed: u64) {
        let center = self.side() as u64 / 2;
        for z in (center - radius)..(center + radius) {
            for y in (center - radius)..(center + radius) {
                for x in (center - radius)..(center + radius) {
                    // Spherical mask
                    let dx = x as f64 - center as f64;
                    let dy = y as f64 - center as f64;
                    let dz = z as f64 - center as f64;
                    if dx * dx + dy * dy + dz * dz < (radius as f64 * radius as f64)
                        && cell_rand_bool(x as i64, y as i64, z as i64, 0, seed, density)
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

    /// Same seed → same root NodeId. Because seed_center is now a pure
    /// function of (x, y, z, seed), two independently constructed worlds
    /// with identical parameters must share a canonical root via hash-cons.
    /// This is the load-bearing property for Hashlife compatibility: the
    /// step cache can key on NodeId only if identical subtrees intern once.
    #[test]
    fn seed_center_is_deterministic_and_hash_consed() {
        let mut a = World::new(6);
        let mut b = World::new(6);
        a.seed_center(12, 0.35, 42);
        b.seed_center(12, 0.35, 42);
        assert_eq!(a.root, b.root);
        assert_eq!(a.population(), b.population());
        assert!(
            a.population() > 0,
            "seed_center should place at least one cell"
        );
    }

    /// Different seeds produce structurally different worlds. This rules
    /// out the "seed is silently ignored" regression.
    #[test]
    fn seed_center_different_seeds_differ() {
        let mut a = World::new(6);
        let mut b = World::new(6);
        a.seed_center(12, 0.35, 1);
        b.seed_center(12, 0.35, 2);
        assert_ne!(a.root, b.root);
    }

    /// Scan order independence: the set of live cells is a pure function
    /// of position. We verify this by recomputing the expected population
    /// from `cell_rand_bool` directly, with no reference to scan order at
    /// all, and confirming it matches World::population().
    #[test]
    fn seed_center_population_matches_pure_function() {
        let level = 6u32;
        let side = 1u64 << level;
        let center = side / 2;
        let radius = 12u64;
        let density = 0.35;
        let seed = 0xC0FFEE_u64;

        let mut expected = 0u64;
        for z in (center - radius)..(center + radius) {
            for y in (center - radius)..(center + radius) {
                for x in (center - radius)..(center + radius) {
                    let dx = x as f64 - center as f64;
                    let dy = y as f64 - center as f64;
                    let dz = z as f64 - center as f64;
                    if dx * dx + dy * dy + dz * dz < (radius as f64 * radius as f64)
                        && cell_rand_bool(x as i64, y as i64, z as i64, 0, seed, density)
                    {
                        expected += 1;
                    }
                }
            }
        }

        let mut world = World::new(level);
        world.seed_center(radius, density, seed);
        assert_eq!(world.population(), expected);
    }
}
