use super::mutation::{MutationQueue, WorldMutation};
use super::rule::CaRule;
use crate::octree::{CellState, NodeId, NodeStore};

/// The simulation world. Owns the octree store and manages stepping.
///
/// For now, stepping works by flattening to a grid, applying rules, and
/// rebuilding the octree. This is O(n³) but correct, and lets us validate
/// the full pipeline. True Hashlife recursive stepping comes next.
///
/// ## Tick loop contract (hash-thing-1v0.9)
///
/// Callers drive the world forward with:
///
/// ```text
///   world.apply_mutations();     // drain queued edits
///   world.step_ca(rule);         // CA evolves the world
///   // entities.update(&world, &mut world.queue)  // eventually
/// ```
///
/// [`World::step_ca`] debug-asserts that the mutation queue is empty on
/// entry. That guard is load-bearing for the closed-world invariant: a
/// pending edit at step-time would desync the step cache from the
/// canonical `NodeId`.
pub struct World {
    pub store: NodeStore,
    pub root: NodeId,
    pub level: u32, // root level — grid is 2^level per side
    pub generation: u64,
    /// Pending world mutations. Entities push here; `apply_mutations`
    /// drains and applies in arrival order at tick boundary.
    pub queue: MutationQueue,
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
            queue: MutationQueue::new(),
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

    /// Drain the mutation queue and apply every pending mutation to the
    /// octree in arrival order. This is the *only* way entity-produced
    /// edits reach the world — the closed-world invariant from
    /// hash-thing-sos is enforced by routing all edits through this
    /// method before [`World::step_ca`] runs.
    ///
    /// Arrival order is stable as long as the producer order is
    /// deterministic. Two producers pushing `[A, B]` and `[B, A]`
    /// respectively will diverge — callers who need commutativity must
    /// order their pushes themselves.
    pub fn apply_mutations(&mut self) {
        for m in self.queue.take() {
            match m {
                WorldMutation::SetCell { x, y, z, state } => {
                    self.root = self.store.set_cell(self.root, x, y, z, state);
                }
                WorldMutation::FillRegion {
                    x_min,
                    y_min,
                    z_min,
                    x_max,
                    y_max,
                    z_max,
                    state,
                } => {
                    debug_assert!(
                        x_min <= x_max && y_min <= y_max && z_min <= z_max,
                        "FillRegion: min must be <= max on every axis"
                    );
                    for z in z_min..=z_max {
                        for y in y_min..=y_max {
                            for x in x_min..=x_max {
                                self.root = self.store.set_cell(self.root, x, y, z, state);
                            }
                        }
                    }
                }
            }
        }
        debug_assert!(
            self.queue.is_empty(),
            "apply_mutations: queue must be empty after drain"
        );
    }

    /// Advance the CA by one generation. This is the canonical step
    /// entry point — eventually backed by Hashlife memoization; for now
    /// it delegates to the brute-force flat stepper.
    ///
    /// **Runtime guard:** debug-asserts that the mutation queue is
    /// empty on entry. If it isn't, the caller forgot
    /// [`World::apply_mutations`] and the step cache would see stale
    /// world state. This is the enforcement mechanism for the
    /// closed-world invariant (hash-thing-1v0.9) — same pattern as
    /// `debug_assert!(step_cache.is_empty())` on compaction (hash-thing-88d).
    pub fn step_ca(&mut self, rule: &dyn CaRule) {
        debug_assert!(
            self.queue.is_empty(),
            "step_ca called with {} pending mutations — caller must apply_mutations first",
            self.queue.len()
        );
        self.step_flat(rule);
    }

    /// Step the simulation forward one generation using brute-force grid evaluation.
    /// This is the simple path — true Hashlife stepping replaces this later.
    ///
    /// Prefer [`World::step_ca`] as the public entry point; `step_flat`
    /// exists as a stepper implementation and a no-guard path for tests
    /// and benchmarks that build up state outside the mutation channel.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::rule::GameOfLife3D;

    /// apply_mutations applies queued edits and drains the queue.
    #[test]
    fn apply_mutations_drains_and_applies_in_order() {
        let mut world = World::new(4); // 16^3
        world.queue.push(WorldMutation::SetCell {
            x: 1,
            y: 2,
            z: 3,
            state: 7,
        });
        world.queue.push(WorldMutation::SetCell {
            x: 4,
            y: 5,
            z: 6,
            state: 9,
        });
        assert_eq!(world.queue.len(), 2);

        world.apply_mutations();

        assert!(world.queue.is_empty());
        assert_eq!(world.get(1, 2, 3), 7);
        assert_eq!(world.get(4, 5, 6), 9);
        assert_eq!(world.population(), 2);
    }

    /// Arrival-order determinism: later mutations to the same cell win.
    /// This is the contract callers depend on for "last write wins"
    /// semantics when multiple producers touch the same position.
    #[test]
    fn apply_mutations_arrival_order_last_write_wins() {
        let mut world = World::new(4);
        world.queue.push(WorldMutation::SetCell {
            x: 8,
            y: 8,
            z: 8,
            state: 1,
        });
        world.queue.push(WorldMutation::SetCell {
            x: 8,
            y: 8,
            z: 8,
            state: 2,
        });
        world.queue.push(WorldMutation::SetCell {
            x: 8,
            y: 8,
            z: 8,
            state: 3,
        });
        world.apply_mutations();
        assert_eq!(world.get(8, 8, 8), 3);
    }

    /// FillRegion covers the requested box inclusive on both ends.
    #[test]
    fn fill_region_covers_inclusive_box() {
        let mut world = World::new(4);
        world.queue.push(WorldMutation::FillRegion {
            x_min: 2,
            y_min: 3,
            z_min: 4,
            x_max: 5,
            y_max: 6,
            z_max: 7,
            state: 1,
        });
        world.apply_mutations();

        // Every cell in the box is set.
        for z in 4..=7 {
            for y in 3..=6 {
                for x in 2..=5 {
                    assert_eq!(world.get(x, y, z), 1, "missed ({x},{y},{z})");
                }
            }
        }
        // Cells just outside the box are untouched.
        assert_eq!(world.get(1, 3, 4), 0);
        assert_eq!(world.get(2, 2, 4), 0);
        assert_eq!(world.get(2, 3, 3), 0);
        assert_eq!(world.get(6, 3, 4), 0);

        let expected_population = 4u64 * 4 * 4;
        assert_eq!(world.population(), expected_population);
    }

    /// step_ca debug-asserts that the mutation queue is empty on entry.
    /// This is the runtime enforcement of the closed-world invariant.
    #[test]
    #[should_panic(expected = "pending mutations")]
    #[cfg(debug_assertions)]
    fn step_ca_panics_with_pending_mutations() {
        let mut world = World::new(4);
        let rule = GameOfLife3D::amoeba();
        world.queue.push(WorldMutation::SetCell {
            x: 1,
            y: 1,
            z: 1,
            state: 1,
        });
        // Caller forgot apply_mutations — guard fires.
        world.step_ca(&rule);
    }

    /// step_ca with an empty queue is equivalent to step_flat (for now).
    #[test]
    fn step_ca_runs_cleanly_with_empty_queue() {
        let mut world = World::new(4);
        let rule = GameOfLife3D::amoeba();
        // Seed a small pattern directly through the queue.
        world.queue.push(WorldMutation::FillRegion {
            x_min: 6,
            y_min: 6,
            z_min: 6,
            x_max: 9,
            y_max: 9,
            z_max: 9,
            state: 1,
        });
        world.apply_mutations();
        let pop_before = world.population();
        assert_eq!(pop_before, 64);
        world.step_ca(&rule);
        assert_eq!(world.generation, 1);
    }

    /// End-to-end determinism through the mutation channel: two worlds
    /// that receive the same sequence of mutations and step the same
    /// number of generations must land on the same canonical root
    /// NodeId at every tick. This is the hash-cons-compatibility
    /// canary for hash-thing-1v0.9, analogous to the one added in
    /// hash-thing-i6y for the direct path.
    #[test]
    fn mutation_channel_roundtrip_is_deterministic() {
        let mutations: Vec<WorldMutation> = vec![
            WorldMutation::SetCell {
                x: 8,
                y: 8,
                z: 8,
                state: 1,
            },
            WorldMutation::SetCell {
                x: 9,
                y: 8,
                z: 8,
                state: 1,
            },
            WorldMutation::SetCell {
                x: 8,
                y: 9,
                z: 8,
                state: 1,
            },
            WorldMutation::SetCell {
                x: 8,
                y: 8,
                z: 9,
                state: 1,
            },
            WorldMutation::FillRegion {
                x_min: 10,
                y_min: 10,
                z_min: 10,
                x_max: 12,
                y_max: 12,
                z_max: 12,
                state: 1,
            },
        ];
        let rule = GameOfLife3D::amoeba();

        let mut a = World::new(5); // 32^3
        let mut b = World::new(5);
        for m in &mutations {
            a.queue.push(*m);
            b.queue.push(*m);
        }
        a.apply_mutations();
        b.apply_mutations();
        assert_eq!(a.root, b.root, "roots diverge after initial apply");

        for gen in 0..5 {
            a.step_ca(&rule);
            b.step_ca(&rule);
            assert_eq!(
                a.root,
                b.root,
                "roots diverge at generation {}",
                gen + 1
            );
        }
    }

    /// Second apply_mutations call on an empty queue is a no-op.
    /// This is the ratchet that prevents accidental double-apply.
    #[test]
    fn apply_mutations_is_idempotent_on_empty_queue() {
        let mut world = World::new(4);
        world.apply_mutations(); // empty, no-op
        world.apply_mutations(); // still empty, still no-op
        assert_eq!(world.population(), 0);
        assert!(world.queue.is_empty());
    }
}
