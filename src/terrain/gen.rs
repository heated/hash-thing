//! Recursive direct-octree builder.
//!
//! Recurses against any `&impl WorldGen`, short-circuits on proof-based
//! uniform classification, and interns through `NodeStore`. The recursion
//! shape is the load-bearing pattern for every future direct-octree generator
//! in this codebase (infinite worlds 3fq.4, HashDAG edits) — keep it small
//! and uniform.
//!
//! There is exactly one short-circuit path: `WorldGen::classify`. No
//! corner-agreement heuristic. Heuristic collapse is not allowed in the first
//! direct-octree generator; it would set the wrong project convention.

use rustc_hash::FxHashMap;

use super::field::WorldGen;
use crate::octree::node::octant_coords;
use crate::octree::{CellState, NodeId, NodeStore};

/// Per-call generation diagnostics. Logged on `seed_terrain`; used by tests
/// to verify that proof-based collapses actually fire.
///
/// **Field semantics for the "is noise the bottleneck?" question**
/// (tracked in hash-thing-3fq.5): `leaves` counts `WorldGen::sample()`
/// calls one-for-one — every level-0 build dispatches exactly one sample,
/// and sampling is the only place noise evaluation lives. `classify_calls`
/// counts `WorldGen::classify()` invocations regardless of outcome
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
    /// Total `WorldGen::classify` invocations, whether they
    /// collapsed or not. Equals `calls_total - leaves` by construction in
    /// the current builder; stored explicitly so a future restructure of
    /// the recursion (e.g. early-outs, lazy child generation) can't silently
    /// break the invariant without a test failing.
    pub classify_calls: u64,
    pub interiors_interned: u64,
    /// Wall-clock time for the heightmap precomputation pass (set by seed_terrain).
    pub precompute_us: u64,
    /// Wall-clock time for the heightmap gen_region pass (set by seed_terrain).
    pub gen_region_us: u64,
    /// Node count in the store after gen_region.
    pub nodes_after_gen: usize,
    /// Skip-at-gen: count of subtrees the builder shortcut at
    /// `level == target_lod` (one per LOD'd chunk that wasn't already
    /// proof-collapsed). Always zero on plain `gen_region`; only
    /// `gen_region_with_lod` increments it.
    pub lod_skips: u64,
}

impl GenStats {
    pub fn total_collapses(&self) -> u64 {
        self.collapses_by_proof.iter().sum()
    }
}

struct Builder<'a, F: WorldGen> {
    store: &'a mut NodeStore,
    field: &'a F,
    /// Cached canonical uniform NodeIds per `(size_log2, state)`. Avoids
    /// O(level) walks through `NodeStore::uniform` on every short-circuit.
    /// Local to one builder run; the canonical NodeIds inside the store
    /// dedup across runs anyway.
    uniform_cache: FxHashMap<(u32, CellState), NodeId>,
    stats: GenStats,
}

impl<'a, F: WorldGen> Builder<'a, F> {
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
        if let Some(state) = self.field.classify(origin, size_log2) {
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

    /// Top-level recursion for `gen_region_with_lod`. Above the chunk
    /// boundary this is identical to [`Self::build`] (classify-then-recurse,
    /// no LOD logic). At `level == chunk_level`, dispatches into
    /// [`Self::build_with_lod`] using the per-chunk `target_lod` returned
    /// by `chunk_lod_fn`.
    fn top_build(
        &mut self,
        region_origin: [i64; 3],
        origin: [i64; 3],
        level: u32,
        chunk_level: u32,
        chunk_lod_fn: &dyn Fn([u64; 3]) -> u32,
    ) -> NodeId {
        debug_assert!(
            (level as usize) < self.stats.calls_per_level.len(),
            "level {level} exceeds GenStats fixed-size arrays (cap 32)",
        );
        debug_assert!(
            level >= chunk_level,
            "top_build invariant: level {level} < chunk_level {chunk_level}",
        );

        if level == chunk_level {
            // Tail-call into build_with_lod — the dispatch itself doesn't
            // sample or classify, so no counter increment here. build_with_lod
            // will increment its own calls_total/classify_calls.
            let cx = ((origin[0] - region_origin[0]) >> chunk_level) as u64;
            let cy = ((origin[1] - region_origin[1]) >> chunk_level) as u64;
            let cz = ((origin[2] - region_origin[2]) >> chunk_level) as u64;
            let target_lod = chunk_lod_fn([cx, cy, cz]);
            assert!(
                target_lod <= chunk_level,
                "chunk_lod_fn returned target_lod {target_lod} > chunk_level {chunk_level} for chunk ({cx},{cy},{cz})",
            );
            return self.build_with_lod(origin, level, target_lod);
        }

        // level > chunk_level: plain build-style recursion with proof-collapse.
        self.stats.calls_total += 1;
        self.stats.calls_per_level[level as usize] += 1;
        self.stats.classify_calls += 1;
        if let Some(state) = self.field.classify(origin, level) {
            self.stats.collapses_by_proof[level as usize] += 1;
            return self.uniform(level, state);
        }

        let half = 1i64 << (level - 1);
        let mut children = [NodeId::EMPTY; 8];
        for (oct, child_slot) in children.iter_mut().enumerate() {
            let (cx, cy, cz) = octant_coords(oct);
            let child_origin = [
                origin[0] + cx as i64 * half,
                origin[1] + cy as i64 * half,
                origin[2] + cz as i64 * half,
            ];
            *child_slot = self.top_build(
                region_origin,
                child_origin,
                level - 1,
                chunk_level,
                chunk_lod_fn,
            );
        }
        self.stats.interiors_interned += 1;
        self.store.interior(level, children)
    }

    /// LOD-aware recursion inside one chunk subtree. Worker for
    /// [`Self::top_build`]. Invariant: `level >= target_lod`.
    ///
    /// Three short-circuits, evaluated in this order:
    /// 1. `level == 0` → sample one leaf (only reachable when
    ///    `target_lod == 0`).
    /// 2. Proof-collapse via `WorldGen::classify` — same as `build()`. Fires
    ///    before the lod-skip check, so a uniform-AIR chunk still collapses
    ///    to the canonical proof-collapsed form regardless of target_lod.
    /// 3. `level == target_lod` after a classify miss → **lod-skip**. Sample
    ///    8 octant centroids (or 8 cells for `target_lod == 1`), apply the
    ///    canonical Leaf-only representative-state rule from
    ///    `representative_state_memo`, and emit `store.leaf(rep)`. Bumps
    ///    `lod_skips`.
    ///
    /// At `level == target_lod`, the emitted output is a raw `Leaf(rep)`
    /// NodeId standing in for a `2^target_lod`-cube uniform-region — the
    /// same convention `lod_collapse` uses (see
    /// `crates/ht-octree/src/store.rs:983`). Proof-collapse at
    /// `level > target_lod` still emits a `store.uniform(level, state)`
    /// Interior chain; this differs structurally from
    /// `lod_collapse_chunk`'s output (which emits a Leaf at `target_lod`
    /// with Interior chain above) but flattens to the same cell grid.
    fn build_with_lod(&mut self, origin: [i64; 3], level: u32, target_lod: u32) -> NodeId {
        debug_assert!(
            (level as usize) < self.stats.calls_per_level.len(),
            "level {level} exceeds GenStats fixed-size arrays (cap 32)",
        );
        debug_assert!(
            level >= target_lod,
            "build_with_lod invariant: level {level} < target_lod {target_lod}",
        );
        self.stats.calls_total += 1;
        self.stats.calls_per_level[level as usize] += 1;

        if level == 0 {
            // Reachable only when target_lod == 0: descend to leaves.
            self.stats.leaves += 1;
            let s = self.field.sample(origin);
            return self.store.leaf(s);
        }

        // Proof-based collapse — same short-circuit as `build()`. Wins over
        // the lod-skip check, so uniform regions always collapse via the
        // canonical proof rule.
        self.stats.classify_calls += 1;
        if let Some(state) = self.field.classify(origin, level) {
            self.stats.collapses_by_proof[level as usize] += 1;
            return self.uniform(level, state);
        }

        if level == target_lod {
            // lod-skip: 8-octant centroid samples + canonical Leaf-only
            // rule from `representative_state_memo` (store.rs:1009).
            //
            // Sub-octant cell offset:
            //   target_lod == 1 → 0 (the 8 octants ARE the 8 cells).
            //   target_lod >= 2 → 2^(target_lod - 2) (lower-corner-of-upper-half).
            let half = 1i64 << (level - 1);
            let sub_offset: i64 = if level >= 2 { 1i64 << (level - 2) } else { 0 };
            let mut rep_state: CellState = 0;
            let mut rep_pop: u64 = 0;
            for oct in 0..8usize {
                let (cx, cy, cz) = octant_coords(oct);
                let cell = [
                    origin[0] + cx as i64 * half + sub_offset,
                    origin[1] + cy as i64 * half + sub_offset,
                    origin[2] + cz as i64 * half + sub_offset,
                ];
                let s = self.field.sample(cell);
                if s != 0 && (rep_state == 0 || rep_pop == 0) {
                    rep_state = s;
                    rep_pop = 1;
                }
            }
            self.stats.lod_skips += 1;
            // Raw Leaf compression — matches `lod_collapse` output shape at
            // target_lod (store.rs:983).
            return self.store.leaf(rep_state);
        }

        // level > target_lod: recurse into 8 octants.
        let half = 1i64 << (level - 1);
        let mut children = [NodeId::EMPTY; 8];
        for (oct, child_slot) in children.iter_mut().enumerate() {
            let (cx, cy, cz) = octant_coords(oct);
            let child_origin = [
                origin[0] + cx as i64 * half,
                origin[1] + cy as i64 * half,
                origin[2] + cz as i64 * half,
            ];
            *child_slot = self.build_with_lod(child_origin, level - 1, target_lod);
        }
        self.stats.interiors_interned += 1;
        self.store.interior(level, children)
    }
}

/// Build a canonical `NodeId` directly from a `WorldGen` over the cube
/// `[origin, origin + 2^size_log2)`. World-absolute coordinates.
pub fn gen_region<F: WorldGen>(
    store: &mut NodeStore,
    field: &F,
    origin: [i64; 3],
    size_log2: u32,
) -> (NodeId, GenStats) {
    let mut builder = Builder::new(store, field);
    let root = builder.build(origin, size_log2);
    (root, builder.stats)
}

/// LOD-aware variant of [`gen_region`] (cswp.8.4).
///
/// Above the chunk boundary (`level > chunk_level`), behaves identically to
/// [`gen_region`]: classify-based proof-collapse and 8-way recursion. At
/// `level == chunk_level`, queries `chunk_lod_fn` for the chunk's
/// `target_lod` and dispatches into a per-chunk LOD-aware sub-builder. If
/// `target_lod > 0` and proof-collapse misses at `level == target_lod`, the
/// sub-builder skips deeper recursion and emits a single representative-
/// material leaf instead.
///
/// **Convention** (matches `docs/perf/cswp-lod.md` §2.1 and
/// `lod_collapse_chunk`):
/// - `target_lod = 0` is identity (recurse to leaves).
/// - `target_lod = chunk_level` is full chunk-collapse (one Leaf per chunk).
/// - `chunk_lod_fn` takes `[u64; 3]` chunk-relative coordinates indexed
///   `[0, 2^(size_log2 - chunk_level))` per axis.
///
/// **Representative-material proxy** (when lod-skip fires): 8 cells at the
/// octant centroids of the `2^target_lod` cube, with the canonical
/// Leaf-only rule from `NodeStore::representative_state_memo` applied to
/// the samples (first non-empty wins). This is **exact** for
/// `target_lod == 1` (the 8 cells ARE the 8 leaves) and **approximate** for
/// `target_lod >= 2` (treats each octant centroid as the canonical
/// representative of that octant). Callers needing the canonical rule for
/// `target_lod >= 2` should use [`gen_region`] followed by
/// `NodeStore::lod_collapse_chunk`.
///
/// Panics if:
/// - `chunk_level == 0` (degenerate; every leaf would be a chunk)
/// - `chunk_level > size_log2`
/// - any `chunk_lod_fn` output exceeds `chunk_level`
pub fn gen_region_with_lod<F: WorldGen>(
    store: &mut NodeStore,
    field: &F,
    origin: [i64; 3],
    size_log2: u32,
    chunk_level: u32,
    chunk_lod_fn: &dyn Fn([u64; 3]) -> u32,
) -> (NodeId, GenStats) {
    assert!(
        chunk_level >= 1,
        "gen_region_with_lod: chunk_level must be >= 1 (got 0)",
    );
    assert!(
        chunk_level <= size_log2,
        "gen_region_with_lod: chunk_level {chunk_level} > size_log2 {size_log2}",
    );
    let mut builder = Builder::new(store, field);
    let root = builder.top_build(origin, origin, size_log2, chunk_level, chunk_lod_fn);
    (root, builder.stats)
}

/// Micro-probe for `WorldGen::sample()` cost. Runs `samples` sample
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
pub fn probe_sample_ns<F: WorldGen>(field: &F, samples: u64) -> f64 {
    assert!(samples >= 1, "probe_sample_ns needs at least one sample");
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
            sea_level: None,
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
    // Performance — narrow enforced gate, not the repo's general perf story.
    // -----------------------------------------------------------------

    #[test]
    fn generation_perf_budget() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let start = std::time::Instant::now();
        let (_, stats) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let elapsed = start.elapsed();

        // Always print so the number lands in CI logs even when the budget
        // gate is disabled. This test is the narrow machine-enforced perf
        // gate the repo currently has: release-only, 64^3 terrain gen only.
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
        // Sanity: on a default heightmap at 64^3, classify must short-
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

    #[test]
    #[should_panic(expected = "probe_sample_ns needs at least one sample")]
    fn probe_sample_ns_rejects_zero_samples() {
        let field = ConstField::new(AIR);
        let _ = probe_sample_ns(&field, 0);
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

    // -----------------------------------------------------------------
    // hash-thing-cswp.8.4: skip-at-gen LOD path.
    // -----------------------------------------------------------------

    #[test]
    fn lod_target_zero_is_identity_with_gen_region() {
        // target_lod=0 everywhere → recurse to leaves, identical to gen_region.
        let field = default_heightmap(1);
        let mut s_a = NodeStore::new();
        let mut s_b = NodeStore::new();
        let (root_a, _) = gen_region(&mut s_a, &field, [0, 0, 0], 6);
        let (root_b, stats) = gen_region_with_lod(&mut s_b, &field, [0, 0, 0], 6, 4, &|_| 0);
        // Hash-cons: same field + same recursion shape ⇒ same NodeId across
        // independent stores.
        assert_eq!(root_a, root_b, "target_lod=0 must equal gen_region");
        assert_eq!(stats.lod_skips, 0, "target_lod=0 must not lod-skip");
    }

    #[test]
    fn lod_const_field_collapses_via_proof_not_skip() {
        // ConstField → classify proof-collapses at every level. The lod-skip
        // check fires *after* classify, so skip count is zero.
        let mut store = NodeStore::new();
        let field = ConstField::new(STONE);
        let (root, stats) = gen_region_with_lod(&mut store, &field, [0, 0, 0], 6, 4, &|_| 4);
        let expected = store.uniform(6, STONE);
        assert_eq!(
            root, expected,
            "uniform stone must proof-collapse to root uniform"
        );
        assert_eq!(
            stats.lod_skips, 0,
            "classify must short-circuit before lod-skip"
        );
        assert!(
            stats.collapses_by_proof[6] >= 1,
            "uniform field must proof-collapse at the root: {stats:?}",
        );
    }

    #[test]
    fn lod_half_space_below_proof_collapses_no_skip() {
        // HalfSpace fully below threshold → classify proves STONE; lod-skip never fires.
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 1024,
            below: STONE,
            above: AIR,
        };
        let (_, stats) = gen_region_with_lod(&mut store, &field, [0, 0, 0], 6, 6, &|_| 6);
        assert_eq!(stats.lod_skips, 0, "fully-below box must not lod-skip");
        assert!(stats.collapses_by_proof[6] >= 1);
    }

    #[test]
    fn lod_half_space_straddling_skips_at_root() {
        // HalfSpace y=8 with size-4 cube [0,16) → classify returns None at
        // root. With chunk_level=4, target_lod=4, exactly one skip at the root.
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 8,
            below: STONE,
            above: AIR,
        };
        let (_, stats) = gen_region_with_lod(&mut store, &field, [0, 0, 0], 4, 4, &|_| 4);
        assert_eq!(
            stats.lod_skips, 1,
            "exactly one skip at the chunk root: {stats:?}"
        );
        assert_eq!(stats.collapses_by_proof[4], 0, "root cannot proof-collapse");
    }

    #[test]
    fn lod_half_space_skip_picks_first_non_empty() {
        // Threshold y=12 → centroids at y=4 (below=STONE) and y=12 (above=AIR).
        // First non-empty in octant order (oct=0,1,2,3 are y=4 STONE) wins.
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 12,
            below: STONE,
            above: AIR,
        };
        let (root, stats) = gen_region_with_lod(&mut store, &field, [0, 0, 0], 4, 4, &|_| 4);
        assert_eq!(stats.lod_skips, 1);
        match *store.get(root) {
            Node::Leaf(s) => assert_eq!(s, STONE, "leaf rep must be STONE"),
            _ => panic!("expected raw Leaf at chunk root for target_lod=chunk_level"),
        }
    }

    #[test]
    fn lod_classify_calls_invariant_holds() {
        // Re-affirm `leaves + classify_calls == calls_total` on the new path
        // with non-zero lod_skips. lod-skip increments classify_calls
        // because classify is invoked before the skip check fires.
        //
        // Threshold=20 with size_log2=5 (cube [0,32)), chunk_level=4 → 8
        // chunks of size 16. Lower 4 chunks (y∈[0,16)) proof-collapse to
        // STONE. Upper 4 chunks (y∈[16,32)) straddle threshold → lod-skip.
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 20,
            below: STONE,
            above: AIR,
        };
        let (_, stats) = gen_region_with_lod(&mut store, &field, [0, 0, 0], 5, 4, &|_| 4);
        assert!(
            stats.lod_skips > 0,
            "expected at least one lod-skip: {stats:?}"
        );
        assert_eq!(
            stats.leaves + stats.classify_calls,
            stats.calls_total,
            "invariant violated under lod-skip: {stats:?}",
        );
    }

    /// Per-column flatten — handles raw `Leaf(s)` at non-zero levels via
    /// `flatten_column_into`, unlike `NodeStore::flatten` which writes a
    /// single cell for any Leaf regardless of contextual level.
    fn flatten_full(store: &NodeStore, root: NodeId, side: usize) -> Vec<CellState> {
        let mut grid = vec![0 as CellState; side * side * side];
        let mut col = vec![0 as CellState; side];
        for z in 0..side {
            for x in 0..side {
                store.flatten_column_into(root, x as u64, z as u64, &mut col);
                for y in 0..side {
                    grid[x + y * side + z * side * side] = col[y];
                }
            }
        }
        grid
    }

    #[test]
    fn lod_target_one_flatten_matches_lod_collapse_chunk() {
        // Adjustment K: target_lod=1 must flatten-match
        // gen_region + per-chunk lod_collapse_chunk(target_lod=1).
        //
        // Use a threshold offset slightly inside a chunk (y=33) so some
        // chunks proof-collapse and others descend into the chunk. At
        // level=1 inside straddling chunks, the 8-cell heuristic is
        // identical to `representative_state_memo`'s Leaf-only rule on the
        // 8 leaf children — exact equivalence is required.
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 33,
            below: STONE,
            above: AIR,
        };
        let size_log2 = 6u32;
        let chunk_level = 4u32;
        let side = 1usize << size_log2;

        // (a) skip-at-gen path with target_lod=1.
        let mut s_a = NodeStore::new();
        let (root_a, _) =
            gen_region_with_lod(&mut s_a, &field, [0, 0, 0], size_log2, chunk_level, &|_| 1);

        // (b) gen_region + per-chunk lod_collapse_chunk(target_lod=1).
        let mut s_b = NodeStore::new();
        let (mut root_b, _) = gen_region(&mut s_b, &field, [0, 0, 0], size_log2);
        let chunks_per_axis = 1u64 << (size_log2 - chunk_level);
        for cz in 0..chunks_per_axis {
            for cy in 0..chunks_per_axis {
                for cx in 0..chunks_per_axis {
                    root_b = s_b.lod_collapse_chunk(root_b, cx, cy, cz, chunk_level, 1);
                }
            }
        }

        let grid_a = flatten_full(&s_a, root_a, side);
        let grid_b = flatten_full(&s_b, root_b, side);
        assert_eq!(
            grid_a, grid_b,
            "target_lod=1 flatten divergence: skip-at-gen vs lod_collapse_chunk",
        );
    }

    #[test]
    fn lod_mixed_policy_non_skipped_chunks_match_gen_region() {
        // Adjustment J: with mixed target_lod (some 0, some k), the chunks at
        // target_lod=0 must flatten-match gen_region for that chunk.
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 24,
            below: STONE,
            above: AIR,
        };
        let size_log2 = 6u32;
        let chunk_level = 4u32;
        let side = 1usize << size_log2;

        // chunk_lod_fn: target_lod=0 for chunk (0,0,0), target_lod=4 elsewhere.
        let policy = |c: [u64; 3]| -> u32 {
            if c == [0, 0, 0] {
                0
            } else {
                4
            }
        };

        let mut s_a = NodeStore::new();
        let (root_a, _) =
            gen_region_with_lod(&mut s_a, &field, [0, 0, 0], size_log2, chunk_level, &policy);
        let mut s_b = NodeStore::new();
        let (root_b, _) = gen_region(&mut s_b, &field, [0, 0, 0], size_log2);

        let grid_a = s_a.flatten(root_a, side);
        let grid_b = s_b.flatten(root_b, side);

        // chunk (0,0,0) spans x,y,z ∈ [0, 16). Compare cells inside that range.
        let cs = 1usize << chunk_level;
        for z in 0..cs {
            for y in 0..cs {
                for x in 0..cs {
                    let idx = x + y * side + z * side * side;
                    assert_eq!(
                        grid_a[idx], grid_b[idx],
                        "non-skipped chunk (0,0,0) cell ({x},{y},{z}) diverged",
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "chunk_level must be >= 1")]
    fn lod_chunk_level_zero_panics() {
        let mut store = NodeStore::new();
        let field = ConstField::new(AIR);
        let _ = gen_region_with_lod(&mut store, &field, [0, 0, 0], 4, 0, &|_| 0);
    }

    #[test]
    #[should_panic(expected = "chunk_level")]
    fn lod_chunk_level_above_size_log2_panics() {
        let mut store = NodeStore::new();
        let field = ConstField::new(AIR);
        let _ = gen_region_with_lod(&mut store, &field, [0, 0, 0], 4, 5, &|_| 0);
    }

    #[test]
    #[should_panic(expected = "target_lod")]
    fn lod_target_lod_above_chunk_level_panics() {
        let mut store = NodeStore::new();
        let field = HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold: 8,
            below: STONE,
            above: AIR,
        };
        // target_lod=5 > chunk_level=4 → assert at top_build dispatch.
        let _ = gen_region_with_lod(&mut store, &field, [0, 0, 0], 5, 4, &|_| 5);
    }
}
