//! Recursive direct-octree builder.
//!
//! Recurses against any `&impl RegionField`, short-circuits on proof-based
//! uniform classification, and interns through `NodeStore`. The recursion
//! shape is the load-bearing pattern for every future direct-octree generator
//! in this codebase (caves 3fq.2, dungeons 3fq.3, infinite worlds 3fq.4,
//! HashDAG edits) — keep it small and uniform.
//!
//! There is exactly one short-circuit path: `RegionField::classify_box`. No
//! corner-agreement heuristic. Heuristic collapse is not allowed in the first
//! direct-octree generator; it would set the wrong project convention.

use rustc_hash::FxHashMap;

use super::field::RegionField;
use crate::octree::{octant_coords, CellState, NodeId, NodeStore};

/// Per-call generation diagnostics. Logged on `seed_terrain`; used by tests
/// to verify that proof-based collapses actually fire.
#[derive(Default, Debug, Clone, Copy)]
pub struct GenStats {
    pub calls_total: u64,
    pub calls_per_level: [u64; 32],
    pub collapses_by_proof: [u64; 32],
    pub leaves: u64,
    pub interiors_interned: u64,
}

impl GenStats {
    pub fn total_collapses(&self) -> u64 {
        self.collapses_by_proof.iter().sum()
    }
}

struct Builder<'a, F: RegionField> {
    store: &'a mut NodeStore,
    field: &'a F,
    /// Cached canonical uniform NodeIds per `(size_log2, state)`. Avoids
    /// O(level) walks through `NodeStore::uniform` on every short-circuit.
    /// Local to one builder run; the canonical NodeIds inside the store
    /// dedup across runs anyway.
    uniform_cache: FxHashMap<(u32, CellState), NodeId>,
    stats: GenStats,
}

impl<'a, F: RegionField> Builder<'a, F> {
    fn new(store: &'a mut NodeStore, field: &'a F) -> Self {
        Self {
            store,
            field,
            uniform_cache: FxHashMap::default(),
            stats: GenStats::default(),
        }
    }

    #[inline]
    fn uniform(&mut self, size_log2: u32, state: CellState) -> NodeId {
        if let Some(&id) = self.uniform_cache.get(&(size_log2, state)) {
            return id;
        }
        let id = self.store.uniform(size_log2, state);
        self.uniform_cache.insert((size_log2, state), id);
        id
    }

    fn build(&mut self, origin: [i64; 3], size_log2: u32) -> NodeId {
        self.stats.calls_total += 1;
        self.stats.calls_per_level[size_log2 as usize] += 1;

        if size_log2 == 0 {
            self.stats.leaves += 1;
            let s = self.field.sample(origin);
            return self.store.leaf(s);
        }

        // Proof-based collapse — the only short-circuit.
        if let Some(state) = self.field.classify_box(origin, size_log2) {
            self.stats.collapses_by_proof[size_log2 as usize] += 1;
            return self.uniform(size_log2, state);
        }

        // Recurse into 8 octants.
        let half = 1i64 << (size_log2 - 1);
        let mut children = [NodeId::EMPTY; 8];
        for (oct, child_slot) in children.iter_mut().enumerate() {
            let (cx, cy, cz) = octant_coords(oct);
            let child_origin = [
                origin[0] + cx as i64 * half,
                origin[1] + cy as i64 * half,
                origin[2] + cz as i64 * half,
            ];
            *child_slot = self.build(child_origin, size_log2 - 1);
        }
        self.stats.interiors_interned += 1;
        self.store.interior(size_log2, children)
    }
}

/// Build a canonical `NodeId` directly from a `RegionField` over the cube
/// `[origin, origin + 2^size_log2)`. World-absolute coordinates.
pub fn gen_region<F: RegionField>(
    store: &mut NodeStore,
    field: &F,
    origin: [i64; 3],
    size_log2: u32,
) -> (NodeId, GenStats) {
    let mut builder = Builder::new(store, field);
    let root = builder.build(origin, size_log2);
    (root, builder.stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::Node;
    use crate::terrain::field::const_field::ConstField;
    use crate::terrain::field::half_space::{HalfSpaceAxis, HalfSpaceField};
    use crate::terrain::field::heightmap::HeightmapField;
    use crate::terrain::materials::{AIR, STONE};

    // -----------------------------------------------------------------
    // ConstField — the trivial test oracle.
    // -----------------------------------------------------------------

    #[test]
    fn const_field_air_collapses_everywhere() {
        let mut store = NodeStore::new();
        let field = ConstField::new(AIR);
        let (root, stats) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let expected = store.uniform(6, AIR);
        assert_eq!(root, expected);
        // One root-level proof collapse, no descent.
        assert_eq!(stats.calls_total, 1);
        assert_eq!(stats.collapses_by_proof[6], 1);
    }

    #[test]
    fn const_field_stone_collapses_everywhere() {
        let mut store = NodeStore::new();
        let field = ConstField::new(STONE);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let expected = store.uniform(6, STONE);
        assert_eq!(root, expected);
    }

    #[test]
    fn const_field_air_has_zero_population() {
        let mut store = NodeStore::new();
        let field = ConstField::new(AIR);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        assert_eq!(store.population(root), 0);
    }

    // -----------------------------------------------------------------
    // HalfSpaceField — independent flatten-vs-sample oracle.
    // -----------------------------------------------------------------

    #[test]
    fn half_space_flatten_matches_sample() {
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 32,
            below: STONE,
            above: AIR,
        };
        let size_log2 = 6u32;
        let side = 1usize << size_log2;
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], size_log2);

        // Independent reference: walk the field directly, not through the
        // builder. This is what makes the test non-self-referential.
        let grid = store.flatten(root, side);
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let expected = field.sample([x as i64, y as i64, z as i64]);
                    let got = grid[x + y * side + z * side * side];
                    assert_eq!(
                        got, expected,
                        "mismatch at ({x},{y},{z}): expected {expected}, got {got}",
                    );
                }
            }
        }
    }

    #[test]
    fn half_space_collapses_above_and_below_but_not_straddle() {
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 32,
            below: STONE,
            above: AIR,
        };

        // Box fully below: y in [0, 32) — fully `below`.
        let (_, s_below) = gen_region(&mut store, &field, [0, 0, 0], 5);
        assert!(
            s_below.collapses_by_proof[5] >= 1,
            "fully-below box should collapse at the root"
        );

        // Box fully above: y in [32, 64) — fully `above`.
        let (_, s_above) = gen_region(&mut store, &field, [0, 32, 0], 5);
        assert!(
            s_above.collapses_by_proof[5] >= 1,
            "fully-above box should collapse at the root"
        );

        // Box straddling: y in [16, 48) — should not collapse at the root.
        let (_, s_mid) = gen_region(&mut store, &field, [0, 16, 0], 5);
        assert_eq!(
            s_mid.collapses_by_proof[5], 0,
            "straddling box must not collapse at the root"
        );
    }

    // -----------------------------------------------------------------
    // HeightmapField — the v1 content path.
    // -----------------------------------------------------------------

    fn default_heightmap(seed: u64) -> HeightmapField {
        HeightmapField {
            seed,
            base_y: 32.0,
            amplitude: 8.0,
            wavelength: 24.0,
            octaves: 4,
        }
    }

    #[test]
    fn heightmap_flatten_matches_sample() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let size_log2 = 6u32;
        let side = 1usize << size_log2;
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], size_log2);
        let grid = store.flatten(root, side);
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let expected = field.sample([x as i64, y as i64, z as i64]);
                    let got = grid[x + y * side + z * side * side];
                    assert_eq!(
                        got, expected,
                        "mismatch at ({x},{y},{z}): expected {expected}, got {got}",
                    );
                }
            }
        }
    }

    #[test]
    fn heightmap_deterministic() {
        // Two stores, same field — identical NodeIds (same intern order).
        let field = default_heightmap(7);
        let mut s1 = NodeStore::new();
        let mut s2 = NodeStore::new();
        let (r1, _) = gen_region(&mut s1, &field, [0, 0, 0], 6);
        let (r2, _) = gen_region(&mut s2, &field, [0, 0, 0], 6);
        assert_eq!(r1, r2);
    }

    /// Walk from a level-6 root down to the level-4 sub-cube at world
    /// `[ox..ox+16, oy..oy+16, oz..oz+16)`. Coordinates must be multiples of 16.
    fn walk_to_level_4(store: &NodeStore, root: NodeId, ox: u64, oy: u64, oz: u64) -> NodeId {
        // Octant at root level 6: which level-5 child (size 32).
        let oct6 = ((ox >> 5) & 1) | (((oy >> 5) & 1) << 1) | (((oz >> 5) & 1) << 2);
        let level5 = store.child(root, oct6 as usize);
        // Octant at level 5: which level-4 child (size 16).
        let oct5 = ((ox >> 4) & 1) | (((oy >> 4) & 1) << 1) | (((oz >> 4) & 1) << 2);
        store.child(level5, oct5 as usize)
    }

    #[test]
    fn heightmap_sky_subtree_is_canonical_air() {
        // At amplitude 8, base 32, surface_max = 40. SURFACE_MARGIN = 2, so the
        // AIR-collapse boundary is y = 42. The level-4 sub-cube at world
        // `[0, 48, 0]..[16, 64, 16]` has y_min = 48 >= 42 — proof-based collapse
        // must produce the canonical level-4 AIR NodeId.
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);

        let canonical_air_4 = store.uniform(4, AIR);
        let sky = walk_to_level_4(&store, root, 0, 48, 0);
        assert_eq!(
            sky, canonical_air_4,
            "level-4 sky cube must be canonical AIR"
        );
    }

    #[test]
    fn heightmap_sky_stable_across_seeds() {
        // The hash-cons flywheel test: the canonical AIR sky cube must be the
        // *same* NodeId regardless of seed. Different seeds → different surface
        // band → identical sky cube above the surface_max + SURFACE_MARGIN line.
        let mut store = NodeStore::new();
        let field_a = default_heightmap(1);
        let field_b = default_heightmap(2);
        let (root_a, _) = gen_region(&mut store, &field_a, [0, 0, 0], 6);
        let (root_b, _) = gen_region(&mut store, &field_b, [0, 0, 0], 6);

        let canonical_air_4 = store.uniform(4, AIR);
        let sky_a = walk_to_level_4(&store, root_a, 0, 48, 0);
        let sky_b = walk_to_level_4(&store, root_b, 0, 48, 0);
        assert_eq!(sky_a, canonical_air_4);
        assert_eq!(sky_b, canonical_air_4);
        assert_eq!(sky_a, sky_b, "sky cubes must dedup across seeds");
    }

    #[test]
    fn heightmap_air_cells_are_exactly_zero() {
        // Defends hash-cons compression against rounding drift: every above-
        // surface cell must be exactly AIR (0), never AIR+1 or similar.
        let mut store = NodeStore::new();
        let field = default_heightmap(3);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let grid = store.flatten(root, 64);
        for z in 0..64 {
            for x in 0..64 {
                // Top of the world is unambiguously above any possible surface.
                let cell = grid[x + 63 * 64 + z * 64 * 64];
                assert_eq!(cell, AIR);
            }
        }
    }

    #[test]
    fn air_invariant_node_id_zero() {
        // Sanity: the canonical empty leaf is NodeId(0) == AIR leaf.
        let store = NodeStore::new();
        // Just touching `store` ensures the constructor reserved the empty leaf.
        let _ = store.population(NodeId::EMPTY);
        // And the constant is 0.
        assert_eq!(AIR, 0);
    }

    // -----------------------------------------------------------------
    // Performance — release-mode budget; debug is much looser.
    // -----------------------------------------------------------------

    #[test]
    fn generation_perf_budget() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let start = std::time::Instant::now();
        let (_, stats) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let elapsed = start.elapsed();

        // Print so it lands in CI logs even when the assertion passes.
        eprintln!(
            "gen_region 64^3: {:?}, calls={}, leaves={}, interiors={}, collapses={}",
            elapsed,
            stats.calls_total,
            stats.leaves,
            stats.interiors_interned,
            stats.total_collapses(),
        );

        // Release budget: 50ms. Debug is ~5x slower; allow 250ms there.
        #[cfg(debug_assertions)]
        let budget = std::time::Duration::from_millis(250);
        #[cfg(not(debug_assertions))]
        let budget = std::time::Duration::from_millis(50);

        assert!(
            elapsed < budget,
            "gen_region 64^3 took {elapsed:?} (budget {budget:?})",
        );
    }

    #[test]
    fn proof_collapses_actually_fire() {
        // Sanity: on a default heightmap at 64^3, classify_box must short-
        // circuit at least once. Otherwise the trait isn't earning its keep.
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (_, stats) = gen_region(&mut store, &field, [0, 0, 0], 6);
        assert!(
            stats.total_collapses() > 0,
            "expected at least one proof-based collapse: {stats:?}",
        );
    }

    // Helps the dead-code lint not complain about Node when only used here.
    #[test]
    fn _node_type_is_in_scope() {
        let store = NodeStore::new();
        match store.get(NodeId::EMPTY) {
            Node::Leaf(0) => {}
            _ => panic!("EMPTY is not Leaf(0)"),
        }
    }
}
