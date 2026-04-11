use super::rule::CaRule;
use crate::octree::{CellState, NodeId, NodeStore};

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
    //!
    //! These pin the current behavior so that hash-thing-6gf.3 (recursive
    //! Hashlife vs brute-force harness) has a stable reference to compare
    //! against. They cover the floor of correctness items from nna:
    //!
    //!   1. empty world stays empty (amoeba)
    //!   2. single live cell dies under amoeba (0 live neighbors)
    //!   3. single live cell under crystal → 3x3x3 cube (27 cells)
    //!   4. 2x2x2 cube is a still-life under custom S7-7/B3-3
    //!   5. determinism: same initial state + rule → byte-equal output
    //!   6. corner-placed single cell under amoeba still dies
    //!
    //! Not yet covered: a 3D GoL period-2 oscillator. Hand-computing one for
    //! any of the standard presets (amoeba, crystal, rule445, pyroclastic)
    //! needs either a literature lookup or brute-force search and is left as
    //! follow-up — see the nna comment trail.
    //!
    //! Wraparound semantics note: src/sim/world.rs::get_neighbors uses
    //! rem_euclid for neighbor lookup, so the brute-force CA wraps at the
    //! world boundary. These tests are written to be correct *under* that
    //! semantics. If wraparound is ever removed, tests 1-5 still pass as
    //! written (they never rely on wrapping behavior); test 6 also still
    //! passes because the single corner cell has zero live Moore neighbors
    //! whether or not wraparound is enabled.

    use super::*;
    use crate::sim::rule::GameOfLife3D;

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
    //
    // Amoeba is S9-26/B5-7. A lone live cell has 0 live neighbors, which is
    // not in [9,26] → dies. Each of its 26 Moore-1 neighbors has exactly 1
    // live neighbor, which is not in [5,7] → no births.
    // -----------------------------------------------------------------
    #[test]
    fn single_cell_dies_under_amoeba() {
        let mut world = empty_world();
        world.set(4, 4, 4, 1);
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
    //
    // Crystal is S0-6/B1-3.
    //   - The live cell at (4,4,4) has 0 live neighbors, which is in [0,6]
    //     → SURVIVES.
    //   - Each of its 26 Moore-1 neighbors has exactly 1 live neighbor
    //     (the center), which is in [1,3] → BIRTHS.
    //   - Cells at Moore distance ≥2 have 0 live neighbors → no births.
    //
    // Result after one step: the 3x3x3 cube centered at (4,4,4), i.e.
    // 27 cells in (3..=5)^3.
    // -----------------------------------------------------------------
    #[test]
    fn single_cell_grows_to_3x3x3_cube_under_crystal() {
        let mut world = empty_world();
        world.set(4, 4, 4, 1);

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
                        1,
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
    //
    // Cube placed at (3..=4)^3 inside an 8^3 world — well clear of the
    // boundary so wraparound can never contaminate neighbor counts.
    //
    // Every cube cell is Moore-1 adjacent to the other 7 cube cells, so
    // each has exactly 7 live neighbors → in [7,7] → SURVIVES.
    //
    // Dead cells adjacent to the cube have live-neighbor counts that depend
    // on how they touch the cube:
    //   - face-shell cells (e.g. (2,3,3)): 4 live neighbors
    //   - edge-shell cells (e.g. (2,2,3)): 2 live neighbors
    //   - corner-shell cells (e.g. (2,2,2)): 1 live neighbor
    // None of {1, 2, 4} is in [3,3], and cells far from the cube have 0
    // neighbors (also not 3), so nothing births. True still-life under
    // this rule.
    // -----------------------------------------------------------------
    #[test]
    fn cube_2x2x2_is_still_life_under_s7_b3() {
        let rule = GameOfLife3D::new(7, 7, 3, 3);

        let mut world = empty_world();
        for z in 3..=4u64 {
            for y in 3..=4u64 {
                for x in 3..=4u64 {
                    world.set(x, y, z, 1);
                }
            }
        }
        assert_eq!(world.population(), 8, "initial cube must have 8 cells");

        // Step several generations — population and exact configuration must
        // remain identical.
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
                                1
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
    // 5. Determinism: same initial state + same rule → byte-equal result.
    //
    // Builds two independent worlds, seeds them identically, steps both 3
    // times under crystal, and asserts the flattened grids are equal. This
    // is the property 6gf.3 ultimately compares the recursive Hashlife step
    // against, so pin it here.
    // -----------------------------------------------------------------
    #[test]
    fn step_flat_is_deterministic() {
        let rule = GameOfLife3D::crystal();

        let mut world_a = empty_world();
        let mut world_b = empty_world();

        // A handful of seeds placed in the interior to avoid accidental
        // coupling to wraparound semantics.
        let seeds: &[(u64, u64, u64)] = &[(4, 4, 4), (3, 4, 5), (5, 2, 4)];
        for &(x, y, z) in seeds {
            world_a.set(x, y, z, 1);
            world_b.set(x, y, z, 1);
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
    //
    // Places a single live cell at the maximum-index corner of an 8^3 world
    // and steps under amoeba. The cell has 0 live Moore neighbors (all 26
    // neighbor slots — whether wrapped or not — land on dead cells, since
    // there are no other live cells in the world), so it dies.
    //
    // This is not a true no-wraparound assertion. Pinning that down would
    // require multiple live cells straddling opposite faces — filed as
    // follow-up. The value of this test is: it proves step_flat doesn't
    // crash, panic, or misbehave at the max index edge in either wraparound
    // or non-wraparound semantics.
    // -----------------------------------------------------------------
    #[test]
    fn corner_single_cell_dies_under_amoeba() {
        let mut world = empty_world();
        let side = world.side() as u64;
        world.set(side - 1, side - 1, side - 1, 1);
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
    //
    // Level 4 -> side 16, center 8. Pre-fix, radius=12 made the loop
    // range `(center - radius)..(center + radius)` on u64, which
    // underflows to a 2^64-scale range and debug-panics (or release-wraps
    // into an 18-quintillion-iteration hang). Post-fix the bounds are
    // clamped to `[0, side)` and the spherical mask still fills a large
    // fraction of cells at density 0.35.
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
    // 8. Non-underflow path: demo default level 6 (side 64), radius 12,
    //    density 0.35 — matches main.rs VOLUME_SIZE=64. Radius < center
    //    here, so the clamp is a no-op and this exercises the normal
    //    path alongside the clamp regression above.
    // -----------------------------------------------------------------
    #[test]
    fn seed_center_normal_radius_produces_population() {
        let mut w = World::new(6);
        w.seed_center(12, 0.35);
        assert!(w.population() > 0);
    }
}
