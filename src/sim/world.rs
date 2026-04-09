use crate::octree::{CellState, NodeId, NodeStore};
use super::rule::CaRule;

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
        Self { store, root, level, generation: 0 }
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
    }

    /// Place a random seed pattern in the center of the world.
    pub fn seed_center(&mut self, radius: u64, density: f64) {
        let center = self.side() as u64 / 2;
        let mut rng = SimpleRng::new(42);
        for z in (center - radius)..(center + radius) {
            for y in (center - radius)..(center + radius) {
                for x in (center - radius)..(center + radius) {
                    // Spherical mask
                    let dx = x as f64 - center as f64;
                    let dy = y as f64 - center as f64;
                    let dz = z as f64 - center as f64;
                    if dx * dx + dy * dy + dz * dz < (radius as f64 * radius as f64) {
                        if rng.next_f64() < density {
                            self.set(x, y, z, 1);
                        }
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
