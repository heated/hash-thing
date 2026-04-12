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
use crate::octree::node::octant_coords;
use crate::octree::{CellState, NodeId, NodeStore};

/// Per-call generation diagnostics. Logged on `seed_terrain`; used by tests
/// to verify that proof-based collapses actually fire.
///
/// **Field semantics for the "is noise the bottleneck?" question**
/// (tracked in hash-thing-3fq.5): `leaves` counts `RegionField::sample()`
/// calls one-for-one — every level-0 build dispatches exactly one sample,
/// and sampling is the only place noise evaluation lives. `classify_calls`
/// counts `RegionField::classify_box()` invocations regardless of outcome
/// (collapse or descent). Together with a one-time `probe_sample_ns`
/// microbenchmark, you can estimate `sample_time ≈ leaves * ns_per_sample`
/// and read off the noise fraction of a gen pass without per-call timing
/// overhead (which would double 64³ gen cost).
#[derive(Default, Debug, Clone, Copy)]
pub struct GenStats {
    pub calls_total: u64,
    pub calls_per_level: [u64; 32],
    pub collapses_by_proof: [u64; 32],
    pub leaves: u64,
    /// Total `RegionField::classify_box` invocations, whether they
    /// collapsed or not. Equals `calls_total - leaves` by construction in
    /// the current builder; stored explicitly so a future restructure of
    /// the recursion (e.g. early-outs, lazy child generation) can't silently
    /// break the invariant without a test failing.
    pub classify_calls: u64,
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
        debug_assert!(
            (size_log2 as usize) < self.stats.calls_per_level.len(),
            "size_log2 {size_log2} exceeds GenStats fixed-size arrays (cap 32)",
        );
        self.stats.calls_total += 1;
        self.stats.calls_per_level[size_log2 as usize] += 1;

        if size_log2 == 0 {
            self.stats.leaves += 1;
            let s = self.field.sample(origin);
            return self.store.leaf(s);
        }

        // Proof-based collapse — the only short-circuit.
        self.stats.classify_calls += 1;
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

/// Micro-probe for `RegionField::sample()` cost. Runs `samples` sample
/// calls over a 64×64×64 coordinate walk and returns `ns/call`. Multiply
/// by `GenStats::leaves` to estimate how much of a gen pass was spent
/// inside `sample()` — i.e. "is noise the bottleneck?" — without paying
/// per-call `Instant::now()` cost (which at 262k leaves would roughly
/// double the gen time).
///
/// **Not a benchmark.** No warmup, no variance estimate, no distribution.
/// Runs cold on whatever core you happen to be on. Budget `samples` so a
/// single call takes ~1 ms (typically 10_000 for value noise); any more
/// and it eats into interactive gen latency. The XOR sink + `black_box`
/// keep LLVM from constant-folding the loop body.
///
/// This is the "is this thing slow?" signal the hash-thing-3fq.5 bead
/// asked for. Use it from `main.rs` at terrain reset time and log the
/// estimated sample fraction alongside total gen time.
pub fn probe_sample_ns<F: RegionField>(field: &F, samples: u64) -> f64 {
    debug_assert!(samples >= 1, "probe_sample_ns needs at least one sample");
    let start = std::time::Instant::now();
    let mut sink: i64 = 0;
    for i in 0..samples {
        // Spread across a 64×64×64 coordinate cube so the probe walks
        // genuinely distinct noise inputs instead of hammering one cell.
        let x = (i & 63) as i64;
        let z = ((i >> 6) & 63) as i64;
        let y = ((i >> 12) & 63) as i64;
        sink ^= field.sample([x, y, z]) as i64;
    }
    std::hint::black_box(sink);
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / samples as f64
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
        // Exactly one classify call (the root) and zero leaves.
        assert_eq!(stats.classify_calls, 1);
        assert_eq!(stats.leaves, 0);
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

        // Always print so the number lands in CI logs even when the budget
        // gate is disabled.
        eprintln!(
            "gen_region 64^3: {:?}, calls={}, leaves={}, interiors={}, collapses={}",
            elapsed,
            stats.calls_total,
            stats.leaves,
            stats.interiors_interned,
            stats.total_collapses(),
        );

        // Only enforce the budget in release builds. Debug builds are 5–20×
        // slower in ways that vary wildly by host (emulated CI runners,
        // address-sanitizer, cold caches), and a flake here wastes more time
        // than it saves. Release is the mode that matters for "is 64^3 still
        // sub-frame?" regressions.
        #[cfg(not(debug_assertions))]
        {
            let budget = std::time::Duration::from_millis(50);
            assert!(
                elapsed < budget,
                "gen_region 64^3 took {elapsed:?} (budget {budget:?})",
            );
        }
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

    // -----------------------------------------------------------------
    // hash-thing-3fq.5: noise-fraction accounting.
    // -----------------------------------------------------------------

    #[test]
    fn classify_calls_plus_leaves_equals_calls_total() {
        // Invariant guard: every build() either samples a leaf or
        // classifies a box, never both, never neither. Protects the
        // derived `leaves * ns_per_sample` estimate against future
        // restructures of the recursion.
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (_, stats) = gen_region(&mut store, &field, [0, 0, 0], 6);
        assert_eq!(
            stats.leaves + stats.classify_calls,
            stats.calls_total,
            "every build() must be exactly one sample OR one classify: {stats:?}",
        );
    }

    #[test]
    fn classify_calls_counts_descent_not_just_collapses() {
        // Straddling half-space: the root classify returns None, so we
        // descend into 8 children, each classify called exactly once,
        // two of which collapse (the fully-above and fully-below octants)
        // and six of which descend further. The invariant we care about
        // is just `classify_calls > total_collapses`, proving the counter
        // isn't mistakenly tracking only successful collapses.
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 16,
            below: STONE,
            above: AIR,
        };
        // Size 5 cube at [0, 0, 0] → [0, 32, 0]. Threshold y=16 splits it.
        let (_, stats) = gen_region(&mut store, &field, [0, 0, 0], 5);
        assert!(
            stats.classify_calls > stats.total_collapses(),
            "classify_calls should count all invocations, not just collapses: {stats:?}",
        );
    }

    #[test]
    fn probe_sample_ns_returns_plausible_number() {
        // Soft-check: the probe measures *something* positive for a real
        // field. Exact values are host-dependent (emulated CI vs bare
        // metal spans orders of magnitude) so we only sanity-check the
        // bounds: 1ns < ns_per_call < 1ms. Value noise at four octaves is
        // comfortably inside that range on any machine that can run the
        // renderer at all.
        let field = default_heightmap(1);
        let ns = probe_sample_ns(&field, 10_000);
        assert!(
            ns > 1.0 && ns < 1_000_000.0,
            "probe_sample_ns out of plausible range: {ns}",
        );
    }

    #[test]
    fn probe_sample_ns_does_not_observe_sink_through_dead_code_elim() {
        // Regression guard against a future refactor that drops the
        // `black_box(sink)` and lets LLVM fold the whole loop away.
        // If the probe is ever optimized to zero, this test starts
        // returning ~0ns/call and fails the lower bound.
        let field = ConstField::new(STONE);
        let ns = probe_sample_ns(&field, 1_000);
        assert!(ns > 0.0, "probe must measure something nonzero: {ns}");
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
