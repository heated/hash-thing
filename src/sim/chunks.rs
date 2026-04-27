//! Per-chunk LOD policy for camera-anchored streaming (cswp.8.3,
//! single-pass descent in cswp.8.3.1).
//!
//! Each frame, [`ChunkLodPolicy::update`] derives a "view root" from the
//! canonical sim root via a single recursive descent over the SVDAG that
//! detects uniform-LOD subtrees and collapses them in one
//! [`NodeStore::lod_collapse`] call. Per-chunk hysteresis is preserved
//! exactly via a held-divergence sidecar (`held_chunks = { c | prev != raw }`)
//! that gates the fast path. The sim continues to read/write the canonical
//! root; the policy is render-only.
//!
//! Design anchors: `docs/perf/cswp-lod.md` §2 (distance→LOD curve),
//! §4 (hard swap + ±0.25 chunk-unit hysteresis), §6 (chunk_level=7 starting
//! point).
//!
//! Cache key includes the source `world_root` NodeId, so most mutations that
//! change the canonical root — sim step, player edit, scene swap, terrain
//! reseed — auto-invalidate the cached view via key inequality.
//! `World::compact_keeping` is the exception: post-order DFS in
//! `compacted_with_remap_keeping` allocates NodeIds deterministically in the
//! fresh store, so an unchanged subtree's `world.root` can land on the
//! numerically-same NodeId in the new epoch and produce a stale cache HIT.
//! The runtime must call [`ChunkLodPolicy::apply_compaction_remap`] when it
//! drains `World::last_compaction_remap` to rebase the cached pair into the
//! new epoch (or drop it).

use ht_octree::{Node, NodeId, NodeStore};
use rustc_hash::FxHashMap;

/// Octree level for a single chunk subtree. Chunks are 128³ = level 7 per
/// `docs/perf/cswp-lod.md` §6 starting recommendation.
pub const CHUNK_LEVEL: u32 = 7;

/// Side length of a chunk in cells (`1 << CHUNK_LEVEL`).
pub const CHUNK_SIDE: u32 = 1u32 << CHUNK_LEVEL;

/// Maximum LOD level the policy emits per `docs/perf/cswp-lod.md` §2.2 table.
pub const DEFAULT_MAX_LOD: u8 = 4;

/// Hysteresis half-width in chunk units, per `docs/perf/cswp-lod.md` §4.3.
const HYSTERESIS: f64 = 0.25;

/// Default `lod_bias` (1.0 matches the design-doc table verbatim).
pub const DEFAULT_LOD_BIAS: f32 = 1.0;

/// Store-growth ratio at which the runtime compacts the gameplay
/// `NodeStore` to shed ghost interior chains accumulated by repeated
/// `lod_collapse_chunk` calls (hash-thing-e4ep).
///
/// 4× matches the original cswp.8.3 warn-once gate — that breadcrumb's
/// remediation ("consider compaction") is now automated.
pub const LOD_COMPACT_RATIO_THRESHOLD: f32 = 4.0;

/// Chunk-space coordinates: `chunk_xyz = world_cell_xyz / CHUNK_SIDE`. `u32`
/// matches the SVDAG coord space (root world is `[0, 2^level)` cells per
/// axis; chunk count per axis is `1 << (world_level - CHUNK_LEVEL)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkCoord {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl ChunkCoord {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Convert a world-space cell position to chunk-space coords.
    /// Floor-divides by `CHUNK_SIDE`; out-of-world coords clamp to `0`.
    pub fn from_world_pos(pos: [f64; 3]) -> Self {
        let to_chunk = |c: f64| -> u32 {
            if !c.is_finite() || c < 0.0 {
                0
            } else {
                (c / CHUNK_SIDE as f64).floor() as u32
            }
        };
        Self {
            x: to_chunk(pos[0]),
            y: to_chunk(pos[1]),
            z: to_chunk(pos[2]),
        }
    }

    /// Chebyshev (L∞) distance in chunk units between two chunks. Matches
    /// the design doc's "chunk-radius" framing — distance is "how many
    /// chunks away in the worst axis," which is what the LOD curve consumes.
    pub fn radius_to(self, other: ChunkCoord) -> f64 {
        let dx = (self.x as i64 - other.x as i64).unsigned_abs();
        let dy = (self.y as i64 - other.y as i64).unsigned_abs();
        let dz = (self.z as i64 - other.z as i64).unsigned_abs();
        dx.max(dy).max(dz) as f64
    }
}

/// Distance → LOD curve per `docs/perf/cswp-lod.md` §2.2 table:
///
/// | scaled-radius | LOD |
/// |---|---|
/// | [0, 1)        | 0   |
/// | [1, 4)        | 1   |
/// | [4, 16)       | 2   |
/// | [16, 64)      | 3   |
/// | [64, ∞)       | 4   |
///
/// where `scaled = radius / lod_bias`. Equivalently:
/// `LOD = clamp(floor(log_4(scaled)) + 1, 0, max_lod)` for `scaled >= 1`,
/// else `LOD = 0`. The doc's "log_4" formula is off by one vs its own
/// table; the table is the design intent (each band boundary at 1, 4, 16,
/// 64 matches the cswp.8.3 plan-review hysteresis enumeration).
///
/// `lod_bias < 1.0` keeps detail closer; `lod_bias > 1.0` collapses sooner.
pub fn target_lod_for_radius(radius: f64, lod_bias: f32, max_lod: u8) -> u8 {
    let bias = (lod_bias as f64).max(1e-3);
    let scaled = radius / bias;
    if !scaled.is_finite() || scaled < 1.0 {
        return 0;
    }
    let lod = (scaled.log(4.0).floor() as i32).saturating_add(1);
    if lod < 0 {
        return 0;
    }
    (lod as u32).min(max_lod as u32) as u8
}

/// Hysteresis-aware variant: a chunk holds at its `current` LOD while
/// `radius` lies inside the band `[4^(current-1) * bias - h, 4^current *
/// bias + h]` (the LOD-`current` band's open interval expanded by `h =
/// HYSTERESIS` on each side). LOD 0's band has no lower edge.
///
/// Boundary comparisons are strict (`>` / `<`), so exactly hitting
/// `boundary ± HYSTERESIS` falls through to the raw target — the held side
/// does *not* win at exact equality (per cswp.8.3 plan-review enumeration).
/// This keeps behaviour deterministic at the float-grid edge instead of
/// chattering on a single-ulp wiggle.
pub fn target_lod_with_hysteresis(
    radius: f64,
    current: Option<u8>,
    lod_bias: f32,
    max_lod: u8,
) -> u8 {
    let raw = target_lod_for_radius(radius, lod_bias, max_lod);
    let Some(current) = current else { return raw };
    if raw == current {
        return current;
    }
    let bias = (lod_bias as f64).max(1e-3);
    let lo_threshold: f64 = if current == 0 {
        f64::NEG_INFINITY
    } else {
        4f64.powi((current as i32) - 1) * bias - HYSTERESIS
    };
    let hi_threshold = 4f64.powi(current as i32) * bias + HYSTERESIS;
    if radius > lo_threshold && radius < hi_threshold {
        return current;
    }
    raw
}

/// Cache key — any change vs the previous `update()` call triggers a
/// recompute. Pinning `world_root` covers sim step / edit / scene swap /
/// terrain reseed (each produces a fresh root NodeId). Compaction is the
/// exception: NodeId allocation is deterministic, so the runtime must call
/// [`ChunkLodPolicy::apply_compaction_remap`] to rebase the cached pair
/// into the post-compact epoch.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CacheKey {
    world_root: NodeId,
    world_level: u32,
    player_chunk: ChunkCoord,
    lod_bias_bits: u32,
    max_lod: u8,
    enabled: bool,
}

/// Per-frame chunk-LOD policy. Owns hysteresis state, cached view root,
/// and the growth baseline that drives the runtime's compaction trigger
/// (hash-thing-e4ep, threshold [`LOD_COMPACT_RATIO_THRESHOLD`]).
pub struct ChunkLodPolicy {
    pub enabled: bool,
    pub lod_bias: f32,
    pub max_lod: u8,
    chunk_lod: FxHashMap<ChunkCoord, u8>,
    cache: Option<(CacheKey, NodeId)>,
    baseline_store_nodes: Option<usize>,
    #[cfg(test)]
    descent_counters: DescentCounters,
}

/// Box of contiguous chunk coordinates, sized as a power-of-2 cube. Used
/// during the cswp.8.3.1 single-pass descent to track which chunks the
/// current SVDAG subtree covers.
#[derive(Clone, Copy, Debug)]
struct ChunkBox {
    /// Inclusive lower-left chunk coordinate.
    lo: [u32; 3],
    /// Side length in chunks at the current descent level
    /// (`1 << (level - CHUNK_LEVEL)`). At `level == CHUNK_LEVEL`, side==1.
    side: u32,
}

impl ChunkBox {
    /// Box-Chebyshev bounds: returns `(rmin, rmax)` where `rmin` is the
    /// smallest Chebyshev distance from `p` to any chunk inside the box,
    /// and `rmax` is the largest. Uses signed `i64` arithmetic so player
    /// coords outside the world don't underflow `u32` subtraction.
    fn chebyshev_bounds(self, p: ChunkCoord) -> (f64, f64) {
        let pa = [p.x as i64, p.y as i64, p.z as i64];
        let lo = [self.lo[0] as i64, self.lo[1] as i64, self.lo[2] as i64];
        let s = self.side as i64;
        let hi_inc = [lo[0] + s - 1, lo[1] + s - 1, lo[2] + s - 1];
        let mut amin: i64 = 0;
        let mut amax: i64 = 0;
        for a in 0..3 {
            let near = if pa[a] < lo[a] {
                lo[a] - pa[a]
            } else if pa[a] > hi_inc[a] {
                pa[a] - hi_inc[a]
            } else {
                0
            };
            let far = (pa[a] - lo[a]).abs().max((pa[a] - hi_inc[a]).abs());
            if near > amin {
                amin = near;
            }
            if far > amax {
                amax = far;
            }
        }
        (amin as f64, amax as f64)
    }

    /// Sub-box for octant `oct` (per `octant_index(x,y,z) = x + 2y + 4z`).
    fn child(self, oct: usize) -> Self {
        debug_assert!(self.side >= 2 && self.side.is_power_of_two());
        let half = self.side / 2;
        let ox = (oct & 1) as u32;
        let oy = ((oct >> 1) & 1) as u32;
        let oz = ((oct >> 2) & 1) as u32;
        Self {
            lo: [
                self.lo[0] + ox * half,
                self.lo[1] + oy * half,
                self.lo[2] + oz * half,
            ],
            side: half,
        }
    }
}

/// Test-only descent counters. Threaded through `DescentCtx` and bumped at
/// each fast-path hit / node visit so equivalence and fast-path-firing
/// tests can assert structural properties of the descent.
#[cfg(test)]
#[derive(Default, Debug)]
struct DescentCounters {
    fast_path_hits: std::cell::Cell<u32>,
    fast_path_chunks: std::cell::Cell<u64>,
    descent_node_visits: std::cell::Cell<u32>,
}

#[cfg(test)]
impl DescentCounters {
    fn reset(&self) {
        self.fast_path_hits.set(0);
        self.fast_path_chunks.set(0);
        self.descent_node_visits.set(0);
    }
    fn bump_visit(&self) {
        self.descent_node_visits
            .set(self.descent_node_visits.get() + 1);
    }
    fn bump_fast_path(&self, side: u32) {
        self.fast_path_hits.set(self.fast_path_hits.get() + 1);
        let vol = (side as u64).pow(3);
        self.fast_path_chunks.set(self.fast_path_chunks.get() + vol);
    }
}

impl Default for ChunkLodPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkLodPolicy {
    pub fn new() -> Self {
        Self {
            enabled: false,
            lod_bias: DEFAULT_LOD_BIAS,
            max_lod: DEFAULT_MAX_LOD,
            chunk_lod: FxHashMap::default(),
            cache: None,
            baseline_store_nodes: None,
            #[cfg(test)]
            descent_counters: DescentCounters::default(),
        }
    }

    /// Compute the view root for this frame.
    ///
    /// Returns `world_root` unchanged when LOD is disabled or the world fits
    /// inside the near band (`world_level <= CHUNK_LEVEL`, e.g. 256³ default).
    /// Otherwise runs the cswp.8.3.1 single-pass recursive descent over the
    /// canonical SVDAG, detecting uniform-LOD subtrees and collapsing them
    /// in one shot via [`NodeStore::lod_collapse`]; per-chunk hysteresis is
    /// preserved by the held-divergence sidecar built at descent entry.
    ///
    /// A change to `world_root` for sim step / edit / scene swap yields a
    /// different cache key → automatic recompute. Compaction is handled
    /// out-of-band by [`Self::apply_compaction_remap`] (NodeId allocation
    /// is deterministic, so a numerically-equal post-compact root can
    /// otherwise produce a stale cache HIT).
    pub fn update(
        &mut self,
        store: &mut NodeStore,
        world_root: NodeId,
        world_level: u32,
        player_pos: [f64; 3],
    ) -> NodeId {
        if !self.enabled || world_level <= CHUNK_LEVEL {
            return world_root;
        }
        let player_chunk = ChunkCoord::from_world_pos(player_pos);
        let key = CacheKey {
            world_root,
            world_level,
            player_chunk,
            lod_bias_bits: self.lod_bias.to_bits(),
            max_lod: self.max_lod,
            enabled: self.enabled,
        };
        if let Some((prev_key, prev_root)) = self.cache {
            if prev_key == key {
                return prev_root;
            }
        }
        if self.baseline_store_nodes.is_none() {
            self.baseline_store_nodes = Some(store.node_count());
        }
        let view_root = self.recompute(store, world_root, world_level, player_chunk);
        self.cache = Some((key, view_root));
        view_root
    }

    /// Recompute path: cswp.8.3.1 single-pass recursive descent over the
    /// canonical SVDAG.
    ///
    /// Visits each interior node *once* (vs. once per chunk in the cswp.8.3
    /// lex-order loop) and detects "uniform raw LOD" subtrees — collapsing
    /// them in a single [`NodeStore::lod_collapse`] call rather than
    /// walking every chunk under them. Per-chunk hysteresis behaviour is
    /// preserved bit-for-bit: the fast path engages only when no chunk in
    /// the box's `held_chunks` over-approximation diverges from raw, in
    /// which case every chunk would have computed the same LOD via the
    /// per-chunk path anyway.
    ///
    /// Every chunk — including LOD 0 — is recorded in the new map so:
    /// - the next frame sees `current = Some(0)` and can hold transitions
    ///   1→0 with hysteresis (cswp.8.3 review BLOCKER 2: previously chunks
    ///   crossing back into the near band lost their hysteresis hold);
    /// - [`lod_histogram`](Self::lod_histogram) reports an honest hist[0]
    ///   instead of always zero.
    fn recompute(
        &mut self,
        store: &mut NodeStore,
        world_root: NodeId,
        world_level: u32,
        player_chunk: ChunkCoord,
    ) -> NodeId {
        let chunks_per_axis: u32 = 1u32 << (world_level - CHUNK_LEVEL);
        let total_chunks = (chunks_per_axis as usize)
            .saturating_pow(3)
            .max(self.chunk_lod.len());
        let mut next_lod: FxHashMap<ChunkCoord, u8> =
            FxHashMap::with_capacity_and_hasher(total_chunks, Default::default());
        // Build the held-divergence sidecar: chunks whose previous LOD
        // disagrees with the raw target at the new player position. The
        // hysteresis predicate `target_lod_with_hysteresis` can return a
        // value other than `raw` *only* when `prev != raw`, so this set
        // over-approximates the divergent chunks. The descent partitions
        // this slice into per-octant sub-slices at each recursion step, so
        // each held chunk is touched O(D) = O(log_2(C)) times total —
        // worst-case work is O(D · |held|), not O(C³ · |held|) as it would
        // be with a global hashset linearly scanned per node.
        let mut held_chunks: Vec<ChunkCoord> = Vec::new();
        for (&chunk, &prev) in self.chunk_lod.iter() {
            let radius = chunk.radius_to(player_chunk);
            let raw = target_lod_for_radius(radius, self.lod_bias, self.max_lod);
            if prev != raw {
                held_chunks.push(chunk);
            }
        }

        #[cfg(test)]
        self.descent_counters.reset();

        let prev_lod_snapshot: &FxHashMap<ChunkCoord, u8> = &self.chunk_lod;
        let mut ctx = DescentCtx {
            store,
            next_lod: &mut next_lod,
            prev_lod: prev_lod_snapshot,
            bias: self.lod_bias,
            max_lod: self.max_lod,
            player_chunk,
            #[cfg(test)]
            counters: &self.descent_counters,
        };
        let root_box = ChunkBox {
            lo: [0; 3],
            side: chunks_per_axis,
        };
        let view = ctx.descend(world_root, world_level, root_box, &held_chunks);
        self.chunk_lod = next_lod;
        view
    }

    /// Test-only legacy lex-order recompute, retained for equivalence
    /// checks. Mirrors the cswp.8.3 implementation byte-for-byte minus the
    /// surrounding `update()` cache plumbing. The descent path
    /// ([`Self::recompute`]) is the production code; this is the oracle.
    #[cfg(test)]
    fn recompute_lex_loop(
        &mut self,
        store: &mut NodeStore,
        world_root: NodeId,
        world_level: u32,
        player_chunk: ChunkCoord,
    ) -> NodeId {
        let chunks_per_axis: u32 = 1u32 << (world_level - CHUNK_LEVEL);
        let total_chunks = (chunks_per_axis as usize)
            .saturating_pow(3)
            .max(self.chunk_lod.len());
        let mut next_lod: FxHashMap<ChunkCoord, u8> =
            FxHashMap::with_capacity_and_hasher(total_chunks, Default::default());
        let mut view = world_root;
        for cz in 0..chunks_per_axis {
            for cy in 0..chunks_per_axis {
                for cx in 0..chunks_per_axis {
                    let chunk = ChunkCoord::new(cx, cy, cz);
                    let radius = chunk.radius_to(player_chunk);
                    let current = self.chunk_lod.get(&chunk).copied();
                    let lod =
                        target_lod_with_hysteresis(radius, current, self.lod_bias, self.max_lod);
                    next_lod.insert(chunk, lod);
                    if lod > 0 {
                        view = store.lod_collapse_chunk(
                            view,
                            cx as u64,
                            cy as u64,
                            cz as u64,
                            CHUNK_LEVEL,
                            lod as u32,
                        );
                    }
                }
            }
        }
        self.chunk_lod = next_lod;
        view
    }

    /// Telemetry: count of chunks at each LOD band, suitable for HUD.
    /// Index 0 = LOD 0 (full detail), index 4 = LOD 4 (max collapse).
    pub fn lod_histogram(&self) -> [u32; 5] {
        let mut hist = [0u32; 5];
        for &lod in self.chunk_lod.values() {
            let idx = (lod as usize).min(4);
            hist[idx] = hist[idx].saturating_add(1);
        }
        hist
    }

    /// Telemetry: store-node growth ratio since LOD was first enabled.
    /// `None` until the first `update()` after `enabled` flipped on.
    pub fn store_growth_ratio(&self, current_nodes: usize) -> Option<f32> {
        let baseline = self.baseline_store_nodes?;
        if baseline == 0 {
            return Some(f32::INFINITY);
        }
        Some(current_nodes as f32 / baseline as f32)
    }

    /// Reset growth baseline. Call after an external compaction so the
    /// next ratio measurement reflects the post-compact size, not pre.
    pub fn reset_growth_baseline(&mut self) {
        self.baseline_store_nodes = None;
    }

    /// Rebase the cached `(world_root, view_root)` pair into the post-compact
    /// epoch using a `World::compact_keeping` remap. Must be called whenever
    /// the runtime drains `World::last_compaction_remap`.
    ///
    /// `compacted_with_remap_keeping` allocates NodeIds deterministically by
    /// post-order DFS in a fresh store, so an unchanged subtree's
    /// `world.root` can land on the numerically-same NodeId across the
    /// transition. Without this rebase, the next `update()` call would see
    /// `key.world_root == cache.key.world_root` and return the stale
    /// `view_root` from the pre-compact store — wrong cells, or a
    /// dangling-NodeId panic.
    ///
    /// If either id is missing from the remap (the runtime kept neither
    /// `world.root` nor the just-returned `view_root` alive — should never
    /// happen under the documented main.rs flow, where both are passed as
    /// `extra_roots`), drops the cache so the next `update()` recomputes
    /// from scratch.
    pub fn apply_compaction_remap(&mut self, remap: &FxHashMap<NodeId, NodeId>) {
        let Some((mut key, view)) = self.cache.take() else {
            return;
        };
        let (Some(&new_world), Some(&new_view)) = (remap.get(&key.world_root), remap.get(&view))
        else {
            return;
        };
        key.world_root = new_world;
        self.cache = Some((key, new_view));
    }

    /// Test-only counter snapshot: `(fast_path_hits, fast_path_chunks,
    /// descent_node_visits)` from the most recent `recompute()` invocation.
    #[cfg(test)]
    fn descent_counter_snapshot(&self) -> (u32, u64, u32) {
        (
            self.descent_counters.fast_path_hits.get(),
            self.descent_counters.fast_path_chunks.get(),
            self.descent_counters.descent_node_visits.get(),
        )
    }
}

/// Borrowed context threaded through the cswp.8.3.1 descent. Bundles the
/// mutable `store`/`next_lod` and the read-only sidecars (`prev_lod`,
/// hysteresis params, player position) that don't vary across recursion.
/// The held-chunks slice is passed positionally to `descend` instead of
/// living here so it can be partitioned per-octant at each step (see
/// `recompute` for the cost argument).
struct DescentCtx<'a> {
    store: &'a mut NodeStore,
    next_lod: &'a mut FxHashMap<ChunkCoord, u8>,
    prev_lod: &'a FxHashMap<ChunkCoord, u8>,
    bias: f32,
    max_lod: u8,
    player_chunk: ChunkCoord,
    #[cfg(test)]
    counters: &'a DescentCounters,
}

impl<'a> DescentCtx<'a> {
    /// Recursive descent worker. `level` is the contextual level of `node`
    /// as seen from its parent (mirrors `lod_collapse_rec`'s convention so
    /// raw `Leaf` subtrees above `CHUNK_LEVEL` are interpreted at their
    /// true contextual level rather than `Node::level()`'s 0). `held_in_box`
    /// is the slice of held-divergence chunks that lie inside `chunk_box`;
    /// when the recursion descends into octants, the slice is partitioned
    /// so each child sees only its own chunks.
    ///
    /// Returns the new NodeId for this subtree (possibly `node` unchanged
    /// if no chunk under it required collapsing).
    fn descend(
        &mut self,
        node: NodeId,
        level: u32,
        chunk_box: ChunkBox,
        held_in_box: &[ChunkCoord],
    ) -> NodeId {
        #[cfg(test)]
        self.counters.bump_visit();

        // Single-chunk leaf: 1×1×1 box.
        if level == CHUNK_LEVEL {
            debug_assert_eq!(chunk_box.side, 1);
            let chunk = ChunkCoord::new(chunk_box.lo[0], chunk_box.lo[1], chunk_box.lo[2]);
            let radius = chunk.radius_to(self.player_chunk);
            let current = self.prev_lod.get(&chunk).copied();
            let lod = target_lod_with_hysteresis(radius, current, self.bias, self.max_lod);
            self.next_lod.insert(chunk, lod);
            if lod > 0 {
                // `node` may itself be a raw `Leaf` here (uniform chunk).
                // `lod_collapse` asserts `target_lod <= node.level()`; for a
                // Leaf that's 0, so collapsing would panic. A uniform chunk
                // is already its own coarsest representation — return as-is.
                return match self.store.get(node) {
                    Node::Leaf(_) => node,
                    Node::Interior { .. } => self.store.lod_collapse(node, lod as u32),
                };
            }
            return node;
        }

        // Above CHUNK_LEVEL — dispatch on Node type FIRST. The Leaf arm
        // never reaches `lod_collapse(leaf, raw)`, which would panic
        // (`NodeStore::lod_collapse` asserts `target_lod <= self.get(root)
        // .level()`; Leaf's level is 0).
        let node_kind = self.store.get(node).clone();
        match node_kind {
            Node::Leaf(_) => {
                // Uniform region above chunk level — already-flat, can't
                // collapse further. Record per-chunk LODs via hysteresis
                // for next-frame bookkeeping; return `node` unchanged.
                self.record_box_via_hysteresis(chunk_box);
                node
            }
            Node::Interior { children, .. } => {
                let (rmin, rmax) = chunk_box.chebyshev_bounds(self.player_chunk);
                let raw_min = target_lod_for_radius(rmin, self.bias, self.max_lod);
                let raw_max = target_lod_for_radius(rmax, self.bias, self.max_lod);

                if raw_min == raw_max && held_in_box.is_empty() {
                    // Uniform-collapse fast path. Soundness:
                    // - `node` is Interior, so `node.level() == level >
                    //   CHUNK_LEVEL`, satisfying lod_collapse's
                    //   `target_lod <= root_level` assertion.
                    // - `raw <= max_lod <= CHUNK_LEVEL <= level`.
                    // - `held_in_box` over-approximates divergent chunks
                    //   inside the box (any chunk where hysteresis returns
                    //   ≠ raw must have `prev != raw`), so empty guarantees
                    //   every chunk in the box would have computed LOD ==
                    //   raw via the per-chunk path → bit-identical result.
                    let raw = raw_min;
                    self.record_box_uniform(chunk_box, raw);
                    #[cfg(test)]
                    self.counters.bump_fast_path(chunk_box.side);
                    if raw > 0 {
                        return self.store.lod_collapse(node, raw as u32);
                    }
                    return node;
                }

                let partitioned = partition_held(chunk_box, held_in_box);
                let mut new_children = children;
                for oct in 0..8 {
                    let child_box = chunk_box.child(oct);
                    new_children[oct] =
                        self.descend(children[oct], level - 1, child_box, &partitioned[oct]);
                }
                self.store.interior(level, new_children)
            }
        }
    }

    /// Record every chunk in `b` at LOD `lod` (uniform fast-path branch).
    fn record_box_uniform(&mut self, b: ChunkBox, lod: u8) {
        for cz in 0..b.side {
            for cy in 0..b.side {
                for cx in 0..b.side {
                    let chunk = ChunkCoord::new(b.lo[0] + cx, b.lo[1] + cy, b.lo[2] + cz);
                    self.next_lod.insert(chunk, lod);
                }
            }
        }
    }

    /// Record every chunk in `b` at its hysteresis-aware target LOD. Used
    /// when descent encounters a raw `Leaf` above CHUNK_LEVEL — the SVDAG
    /// can't be mutated further but the chunk-LOD map still needs to
    /// reflect each chunk's individual band so the next frame's
    /// hysteresis check has accurate `prev` values.
    fn record_box_via_hysteresis(&mut self, b: ChunkBox) {
        for cz in 0..b.side {
            for cy in 0..b.side {
                for cx in 0..b.side {
                    let chunk = ChunkCoord::new(b.lo[0] + cx, b.lo[1] + cy, b.lo[2] + cz);
                    let radius = chunk.radius_to(self.player_chunk);
                    let current = self.prev_lod.get(&chunk).copied();
                    let lod = target_lod_with_hysteresis(radius, current, self.bias, self.max_lod);
                    self.next_lod.insert(chunk, lod);
                }
            }
        }
    }
}

/// Partition `held` into 8 sub-slices keyed by which octant of `b` each
/// chunk falls in. Caller passes only the chunks already in `b`; this
/// routes them to the matching child box via the same `oct = x | y<<1 |
/// z<<2` encoding used throughout the SVDAG. Each held chunk is touched
/// O(D) = O(log_2(C)) times across the full descent, so total work is
/// O(D · |held|) — independent of the world's chunk count.
fn partition_held(b: ChunkBox, held: &[ChunkCoord]) -> [Vec<ChunkCoord>; 8] {
    let half = b.side / 2;
    let mid_x = b.lo[0] + half;
    let mid_y = b.lo[1] + half;
    let mid_z = b.lo[2] + half;
    // `Default::default()` initializes each Vec empty; capacity grows on
    // first push. Held sets are typically tiny (steady-state empty;
    // worst-case post-teleport bounded by `prev_chunk_lod.len()`), so the
    // up-front allocation cost is negligible compared to the O(C³) lex
    // loop we're replacing.
    let mut out: [Vec<ChunkCoord>; 8] = Default::default();
    for &c in held {
        let oct = ((c.x >= mid_x) as usize)
            | (((c.y >= mid_y) as usize) << 1)
            | (((c.z >= mid_z) as usize) << 2);
        out[oct].push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_coord_from_world_pos_floors_per_axis() {
        // Origin chunk.
        let c = ChunkCoord::from_world_pos([0.0, 0.0, 0.0]);
        assert_eq!(c, ChunkCoord::new(0, 0, 0));
        // Just inside chunk 0 (CHUNK_SIDE - 1).
        let c = ChunkCoord::from_world_pos([
            (CHUNK_SIDE - 1) as f64,
            (CHUNK_SIDE - 1) as f64,
            (CHUNK_SIDE - 1) as f64,
        ]);
        assert_eq!(c, ChunkCoord::new(0, 0, 0));
        // Boundary cell sits in the next chunk.
        let c = ChunkCoord::from_world_pos([CHUNK_SIDE as f64, 0.0, 0.0]);
        assert_eq!(c, ChunkCoord::new(1, 0, 0));
        // Far chunk.
        let c = ChunkCoord::from_world_pos([
            (CHUNK_SIDE * 5) as f64 + 17.0,
            (CHUNK_SIDE * 9) as f64,
            (CHUNK_SIDE * 13) as f64 + 0.5,
        ]);
        assert_eq!(c, ChunkCoord::new(5, 9, 13));
        // Negative / NaN clamp to 0.
        assert_eq!(
            ChunkCoord::from_world_pos([-1.0, -1e9, f64::NAN]),
            ChunkCoord::new(0, 0, 0)
        );
    }

    #[test]
    fn chunk_radius_is_chebyshev() {
        let a = ChunkCoord::new(3, 5, 7);
        assert_eq!(a.radius_to(a), 0.0);
        assert_eq!(a.radius_to(ChunkCoord::new(3, 5, 8)), 1.0);
        assert_eq!(a.radius_to(ChunkCoord::new(0, 5, 7)), 3.0);
        assert_eq!(a.radius_to(ChunkCoord::new(10, 5, 12)), 7.0);
    }

    /// Pins the design-doc §2.2 table cell-by-cell with `lod_bias = 1.0`.
    #[test]
    fn target_lod_curve_matches_design_doc_table() {
        let bias = 1.0_f32;
        let m = 4u8;
        // 0–1 → LOD 0
        assert_eq!(target_lod_for_radius(0.0, bias, m), 0);
        assert_eq!(target_lod_for_radius(0.5, bias, m), 0);
        assert_eq!(target_lod_for_radius(0.999, bias, m), 0);
        // 1–4 → LOD 1
        assert_eq!(target_lod_for_radius(1.0, bias, m), 1);
        assert_eq!(target_lod_for_radius(2.0, bias, m), 1);
        assert_eq!(target_lod_for_radius(3.999, bias, m), 1);
        // 4–16 → LOD 2
        assert_eq!(target_lod_for_radius(4.0, bias, m), 2);
        assert_eq!(target_lod_for_radius(8.0, bias, m), 2);
        assert_eq!(target_lod_for_radius(15.999, bias, m), 2);
        // 16–64 → LOD 3
        assert_eq!(target_lod_for_radius(16.0, bias, m), 3);
        assert_eq!(target_lod_for_radius(32.0, bias, m), 3);
        assert_eq!(target_lod_for_radius(63.999, bias, m), 3);
        // 64+ → LOD 4 (clamped at max_lod)
        assert_eq!(target_lod_for_radius(64.0, bias, m), 4);
        assert_eq!(target_lod_for_radius(256.0, bias, m), 4);
        assert_eq!(target_lod_for_radius(1e6, bias, m), 4);
    }

    /// Hysteresis enumeration. For each band boundary `r ∈ {1, 4, 16, 64}`,
    /// verify both sides of the ±0.25 hold zone.
    #[test]
    fn hysteresis_each_band_boundary_strict_inequality() {
        let bias = 1.0_f32;
        let m = 4u8;
        for (boundary, lower, upper) in [(1.0, 0u8, 1u8), (4.0, 1, 2), (16.0, 2, 3), (64.0, 3, 4)] {
            // Sitting in the upper band, just below the boundary by less
            // than HYSTERESIS — held at upper.
            assert_eq!(
                target_lod_with_hysteresis(boundary - 0.249, Some(upper), bias, m),
                upper,
                "boundary {boundary} hold-down failed"
            );
            // Past the held threshold — flips down to lower.
            assert_eq!(
                target_lod_with_hysteresis(boundary - 0.251, Some(upper), bias, m),
                lower,
                "boundary {boundary} flip-down failed"
            );
            // Sitting in the lower band, just above the boundary — held.
            assert_eq!(
                target_lod_with_hysteresis(boundary + 0.249, Some(lower), bias, m),
                lower,
                "boundary {boundary} hold-up failed"
            );
            // Past the held threshold — flips up to upper.
            assert_eq!(
                target_lod_with_hysteresis(boundary + 0.251, Some(lower), bias, m),
                upper,
                "boundary {boundary} flip-up failed"
            );
            // Exactly at boundary ± HYSTERESIS: strict inequality means the
            // held side wins (no oscillation under float noise).
            assert_eq!(
                target_lod_with_hysteresis(boundary - HYSTERESIS, Some(upper), bias, m),
                lower,
                "boundary {boundary} strict-inequality (lower edge) failed"
            );
            assert_eq!(
                target_lod_with_hysteresis(boundary + HYSTERESIS, Some(lower), bias, m),
                upper,
                "boundary {boundary} strict-inequality (upper edge) failed"
            );
        }
    }

    #[test]
    fn hysteresis_no_state_means_use_raw_target() {
        // First observation of a chunk has no held value; should match raw.
        for &r in &[0.0, 0.5, 1.0, 3.99, 4.0, 16.0, 100.0] {
            assert_eq!(
                target_lod_with_hysteresis(r, None, 1.0, 4),
                target_lod_for_radius(r, 1.0, 4),
                "raw vs hysteresis-no-state diverged at r={r}"
            );
        }
    }

    #[test]
    fn lod_bias_shifts_curve() {
        // bias = 2.0 → halves the effective radius → LOD-1 doesn't kick in
        // until raw radius hits 2.0.
        assert_eq!(target_lod_for_radius(1.5, 2.0, 4), 0);
        assert_eq!(target_lod_for_radius(2.0, 2.0, 4), 1);
        // bias = 0.5 → doubles effective radius → LOD-1 kicks in earlier.
        assert_eq!(target_lod_for_radius(0.5, 0.5, 4), 1);
        assert_eq!(target_lod_for_radius(0.49, 0.5, 4), 0);
    }

    // ─── cswp.8.3.1 descent equivalence tests ──────────────────────────

    /// Tiny LCG for deterministic test scene generation. SplitMix-style
    /// constants borrowed from the standard literature; not for crypto.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(
                seed.wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(0xBF58476D1CE4E5B9),
            )
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    fn stone_cell() -> crate::octree::CellState {
        crate::octree::Cell::pack(1, 0).raw()
    }

    fn plant_seeded_cells(world: &mut crate::sim::World, count: usize, seed: u64) {
        use crate::sim::WorldCoord;
        let side = 1u64 << world.level;
        let mut rng = Lcg::new(seed);
        let stone = stone_cell();
        for _ in 0..count {
            let x = (rng.next_u64() % side) as i64;
            let y = (rng.next_u64() % side) as i64;
            let z = (rng.next_u64() % side) as i64;
            world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), stone);
        }
    }

    fn run_both_and_assert_equiv(world: &mut crate::sim::World, player: ChunkCoord, ctx: &str) {
        let mut p_descent = ChunkLodPolicy::new();
        p_descent.enabled = true;
        let mut p_legacy = ChunkLodPolicy::new();
        p_legacy.enabled = true;
        let view_d = p_descent.recompute(&mut world.store, world.root, world.level, player);
        let view_l = p_legacy.recompute_lex_loop(&mut world.store, world.root, world.level, player);
        assert_eq!(
            view_d, view_l,
            "{ctx}: view_root mismatch (descent={view_d:?}, legacy={view_l:?})"
        );
        assert_eq!(
            p_descent.chunk_lod, p_legacy.chunk_lod,
            "{ctx}: chunk_lod map mismatch"
        );
    }

    /// Plan-review test 1: equivalence vs. lex-order legacy at world level
    /// 8. The legacy lex loop's per-chunk `lod_collapse_chunk` cost is
    /// exactly the slowness this perf optimization removes, so larger
    /// world levels (10/12/13) are impractical as a per-chunk oracle in
    /// a unit test. Multi-position fuzz at level 8 covers fast-path,
    /// recursion, and band-boundary behaviour. Level-10 multi-frame
    /// equivalence is checked separately by
    /// `descent_matches_lex_loop_across_multiple_frames`; level-12
    /// behaviour is exercised by the focused tests below
    /// (`fast_path_engages`, `held_chunk`, `compressed_leaf`,
    /// `player_outside_world`) which do not round-trip the lex loop.
    #[test]
    fn descent_matches_lex_loop_at_level_8() {
        use crate::sim::World;
        let level = 8u32;
        let chunks_per_axis = 1u32 << (level - CHUNK_LEVEL);
        for seed in 0u64..3 {
            let mut world = World::new(level);
            plant_seeded_cells(&mut world, 32, seed);
            let positions = [
                ChunkCoord::new(0, 0, 0),
                ChunkCoord::new(
                    chunks_per_axis / 2,
                    chunks_per_axis / 2,
                    chunks_per_axis / 2,
                ),
                ChunkCoord::new(chunks_per_axis - 1, 0, 0),
                ChunkCoord::new(0, chunks_per_axis / 2, chunks_per_axis - 1),
            ];
            for &player in &positions {
                run_both_and_assert_equiv(
                    &mut world,
                    player,
                    &format!("level={level} seed={seed} player={player:?}"),
                );
            }
        }
    }

    /// Plan-review test 2: at world_level 12 with player at origin, the
    /// far octant (chunks 16..32 on each axis) is entirely beyond the
    /// largest band radius (64 chunks at default bias × max_lod=4 maps to
    /// LOD 4 only for r≥64, but the `hi` corner at (31,31,31) has Chebyshev
    /// 31 — LOD 3, while (16,16,16) has 16 — LOD 3 too). Pick a player
    /// far enough away that one octant is fully uniform-LOD.
    #[test]
    fn descent_fast_path_engages_for_uniform_far_subtree() {
        use crate::sim::World;
        let level = 12u32;
        let mut world = World::new(level);
        // Plant a single far cell so the canonical SVDAG isn't all-empty
        // (which would make the entire descent collapse via the empty
        // root and skip the interior tour).
        use crate::sim::WorldCoord;
        world.set(WorldCoord(7), WorldCoord(7), WorldCoord(7), stone_cell());

        let mut policy = ChunkLodPolicy::new();
        policy.enabled = true;
        // Player at origin (chunk 0,0,0). Octants 1..7 of the level-12
        // root are all far from the player; many sub-boxes within will
        // satisfy raw_min == raw_max and engage the fast path.
        let player = ChunkCoord::new(0, 0, 0);
        let _view = policy.recompute(&mut world.store, world.root, world.level, player);
        let (hits, chunks, _visits) = policy.descent_counter_snapshot();
        assert!(
            hits >= 1,
            "fast path must engage at least once for a 32³ chunk world \
             with player at origin (got hits={hits}, chunks={chunks})"
        );
        // Far octant (cx,cy,cz ≥ 16) has 16³ = 4096 chunks. At least one
        // multi-chunk fast path firing must cover at least 8 chunks (the
        // smallest non-trivial level-8 subtree under CHUNK_LEVEL+1=8).
        // In practice we expect a single firing of 4096+ chunks for the
        // fully-far octant. Lower bound is conservative.
        assert!(
            chunks >= 8,
            "fast path must aggregate at least 8 chunks; got {chunks}"
        );
    }

    /// Plan-review test 3: a held chunk straddling a band boundary causes
    /// the descent to punt back to per-chunk granularity in any subtree
    /// containing it; the resulting view + chunk_lod still matches legacy.
    ///
    /// Setup math (with `lod_bias = 0.25`, `max_lod = 4`):
    /// - `target_lod_for_radius(4, 0.25, 4)`: scaled = 16, log_4 = 2,
    ///   +1 = 3 → raw = 3.
    /// - hysteresis hold-zone for `prev = 2, bias = 0.25`:
    ///   `(4·0.25 − 0.25, 16·0.25 + 0.25) = (0.75, 4.25)`. Since `r = 4 ∈
    ///   (0.75, 4.25)`, hysteresis returns 2 instead of raw=3. So the
    ///   target chunk is genuinely held-divergent: `held` set contains it,
    ///   any fast-path check covering its sub-box must fall through to
    ///   per-chunk recursion. Equivalence vs. the lex loop then proves
    ///   the punt code path produces the same NodeId + chunk_lod map.
    #[test]
    fn descent_with_held_chunk_matches_legacy() {
        use crate::sim::World;
        let level = 10u32;
        let mut world = World::new(level);
        plant_seeded_cells(&mut world, 16, 11);
        let player = ChunkCoord::new(0, 0, 0);
        let target = ChunkCoord::new(4, 0, 0); // Chebyshev radius 4.

        // Sanity-check the divergence math directly so future tweaks to the
        // LOD curve or hysteresis constants surface here as test failures
        // rather than as silent skips of the held path.
        let raw = target_lod_for_radius(4.0, 0.25, 4);
        let held = target_lod_with_hysteresis(4.0, Some(2), 0.25, 4);
        assert_eq!(raw, 3, "test setup: raw at r=4, bias=0.25 must be 3");
        assert_eq!(held, 2, "test setup: held at r=4, prev=2 must stay 2");

        let mut p_descent = ChunkLodPolicy::new();
        p_descent.enabled = true;
        p_descent.lod_bias = 0.25;
        let mut p_legacy = ChunkLodPolicy::new();
        p_legacy.enabled = true;
        p_legacy.lod_bias = 0.25;
        // Hand-poison one chunk's prev to 2 so frame 1's held-divergence
        // sidecar contains `target`. Other chunks' `prev` is unset → enter
        // hysteresis path with `current = None`, no held divergence there.
        p_descent.chunk_lod.insert(target, 2);
        p_legacy.chunk_lod.insert(target, 2);

        let view_d = p_descent.recompute(&mut world.store, world.root, world.level, player);
        let view_l = p_legacy.recompute_lex_loop(&mut world.store, world.root, world.level, player);
        assert_eq!(view_d, view_l, "view_root must match under held divergence");
        assert_eq!(
            p_descent.chunk_lod, p_legacy.chunk_lod,
            "chunk_lod must match under held divergence"
        );
        // Positively assert the held chunk really did diverge: the raw
        // target at r=4 with bias=0.25 is 3, and the recorded LOD must be
        // the held value 2.
        assert_eq!(
            p_descent.chunk_lod.get(&target).copied(),
            Some(2),
            "held chunk must record held LOD (2), not raw (3)"
        );
    }

    /// Plan-review test 4: equivalence on a world with raw `Leaf` nodes
    /// at contextual level > CHUNK_LEVEL. Catches the BLOCKER class
    /// (descent must NOT call `lod_collapse` on a Leaf node — it would
    /// panic the `target_lod <= node.level()` assertion).
    #[test]
    fn descent_handles_compressed_leaf_above_chunk_level() {
        use ht_octree::NodeStore;

        // Build by hand: an Interior at level 12 with all 8 children =
        // a single stone-Leaf NodeId at contextual level 11. This mirrors
        // the post-compaction state where a uniform sub-region is
        // represented by a single Leaf at high contextual level.
        let mut store = NodeStore::new();
        let stone = store.leaf(stone_cell());
        let world_root = store.interior(12, [stone; 8]);
        let world_level = 12u32;
        let chunks_per_axis = 1u32 << (world_level - CHUNK_LEVEL);

        for &player in &[
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(
                chunks_per_axis / 2,
                chunks_per_axis / 2,
                chunks_per_axis / 2,
            ),
            ChunkCoord::new(
                chunks_per_axis - 1,
                chunks_per_axis - 1,
                chunks_per_axis - 1,
            ),
        ] {
            let mut p_descent = ChunkLodPolicy::new();
            p_descent.enabled = true;
            let mut p_legacy = ChunkLodPolicy::new();
            p_legacy.enabled = true;
            let view_d = p_descent.recompute(&mut store, world_root, world_level, player);
            let view_l = p_legacy.recompute_lex_loop(&mut store, world_root, world_level, player);
            assert_eq!(
                view_d, view_l,
                "compressed-leaf scene: view mismatch at player={player:?}"
            );
            assert_eq!(
                p_descent.chunk_lod, p_legacy.chunk_lod,
                "compressed-leaf scene: chunk_lod mismatch at player={player:?}"
            );
        }
    }

    /// Plan-review test 5: player chunk far outside the world bounds. The
    /// Chebyshev-bounds arithmetic must use signed `i64` so this doesn't
    /// underflow `u32`. Both legacy and descent should still agree.
    #[test]
    fn descent_with_player_outside_world_matches_legacy() {
        use crate::sim::World;
        let level = 10u32;
        let mut world = World::new(level);
        plant_seeded_cells(&mut world, 16, 7);
        let chunks_per_axis = 1u32 << (level - CHUNK_LEVEL);

        // `from_world_pos` clamps negative to 0, so we model "outside
        // world" via a chunk coordinate well past `chunks_per_axis`.
        for &player in &[
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(chunks_per_axis * 4, 0, 0),
            ChunkCoord::new(
                chunks_per_axis * 4,
                chunks_per_axis * 4,
                chunks_per_axis * 4,
            ),
        ] {
            run_both_and_assert_equiv(&mut world, player, &format!("player_outside={player:?}"));
        }
    }

    /// Multi-frame equivalence: both paths must agree across a sequence
    /// of recompute calls where the policy state evolves between frames.
    #[test]
    fn descent_matches_lex_loop_across_multiple_frames() {
        use crate::sim::World;
        let level = 10u32;
        let mut world = World::new(level);
        plant_seeded_cells(&mut world, 24, 31);

        let mut p_descent = ChunkLodPolicy::new();
        p_descent.enabled = true;
        let mut p_legacy = ChunkLodPolicy::new();
        p_legacy.enabled = true;

        // Walk the player across the world; both policies should stay in
        // lockstep frame after frame.
        let chunks_per_axis = 1u32 << (level - CHUNK_LEVEL);
        for step in 0..6u32 {
            let p = ChunkCoord::new(step % chunks_per_axis, step % chunks_per_axis, step);
            let view_d = p_descent.recompute(&mut world.store, world.root, world.level, p);
            let view_l = p_legacy.recompute_lex_loop(&mut world.store, world.root, world.level, p);
            assert_eq!(view_d, view_l, "step {step}: view_root mismatch");
            assert_eq!(
                p_descent.chunk_lod, p_legacy.chunk_lod,
                "step {step}: chunk_lod mismatch"
            );
        }
    }

    /// 7hqn idempotency oracle: with no world mutations and the player
    /// held still, two back-to-back `recompute` calls must produce
    /// identical `view_root` + `chunk_lod`. Any churn (iteration-order
    /// non-determinism in the descent, hashing instability, frame-state
    /// leakage) shows up as a failure here. Complements the existing
    /// lex-loop equivalence tests at level 8/10 — those check
    /// correctness against an oracle; this checks self-stability, which
    /// is the only practical invariant at the depths where the lex-loop
    /// oracle is too slow to use.
    ///
    /// Level 10 keeps the default-suite cost low (~1 s). Deeper coverage
    /// at level 12 / 13 is in the `#[ignore]`'d companion tests below.
    #[test]
    fn descent_idempotent_at_level_10() {
        use crate::sim::World;
        let level = 10u32;
        let mut world = World::new(level);
        plant_seeded_cells(&mut world, 24, 17);
        let chunks_per_axis = 1u32 << (level - CHUNK_LEVEL);

        for &player in &[
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(
                chunks_per_axis / 2,
                chunks_per_axis / 2,
                chunks_per_axis / 2,
            ),
            ChunkCoord::new(chunks_per_axis - 1, 0, 0),
        ] {
            let mut policy = ChunkLodPolicy::new();
            policy.enabled = true;

            // Frame 1 primes hysteresis state.
            let view_a = policy.recompute(&mut world.store, world.root, world.level, player);
            let lod_a = policy.chunk_lod.clone();

            // Frame 2 at the same player, same world — must be identical.
            let view_b = policy.recompute(&mut world.store, world.root, world.level, player);
            assert_eq!(
                view_a, view_b,
                "view_root churn across back-to-back recomputes at player={player:?}"
            );
            assert_eq!(
                policy.chunk_lod, lod_a,
                "chunk_lod churn across back-to-back recomputes at player={player:?}"
            );

            // Frame 3 — once stable, stay stable. Catches a hypothetical
            // "settle after two frames" non-determinism the 1-vs-2 check
            // alone would miss.
            let view_c = policy.recompute(&mut world.store, world.root, world.level, player);
            assert_eq!(
                view_a, view_c,
                "view_root drift on third recompute at player={player:?}"
            );
            assert_eq!(
                policy.chunk_lod, lod_a,
                "chunk_lod drift on third recompute at player={player:?}"
            );
        }
    }

    /// Same idempotency check at level 13 (production target). `#[ignore]`'d
    /// because cold-cache `recompute` on a planted-cells level-13 world
    /// runs ~20 s per call; 6 calls (3 positions × 2 frames) overruns the
    /// default suite budget. Run on demand via:
    ///
    ///   cargo test --profile perf --lib -p hash-thing \
    ///       sim::chunks::tests::descent_idempotent_at_level_13 \
    ///       -- --ignored --nocapture
    #[test]
    #[ignore]
    fn descent_idempotent_at_level_13() {
        use crate::sim::World;
        let level = 13u32;
        let mut world = World::new(level);
        plant_seeded_cells(&mut world, 64, 17);
        let chunks_per_axis = 1u32 << (level - CHUNK_LEVEL);

        for &player in &[
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(
                chunks_per_axis / 2,
                chunks_per_axis / 2,
                chunks_per_axis / 2,
            ),
            ChunkCoord::new(chunks_per_axis - 1, 0, 0),
        ] {
            let mut policy = ChunkLodPolicy::new();
            policy.enabled = true;
            let view_a = policy.recompute(&mut world.store, world.root, world.level, player);
            let lod_a = policy.chunk_lod.clone();
            let view_b = policy.recompute(&mut world.store, world.root, world.level, player);
            assert_eq!(view_a, view_b, "view churn at player={player:?}");
            assert_eq!(policy.chunk_lod, lod_a, "chunk_lod churn at player={player:?}");
        }
    }

    /// 7hqn extension: idempotency on a hand-built world with raw `Leaf`
    /// nodes at contextual level above CHUNK_LEVEL. Mirrors the structure
    /// `descent_handles_compressed_leaf_above_chunk_level` exercises, but
    /// scales to level 13 (production target) and checks self-stability
    /// instead of cross-path equivalence. Fast even at level 13 because
    /// the world is a uniform Leaf — the descent's fast path engages
    /// across the full root and avoids per-chunk recursion. Catches
    /// divergences that only manifest when the descent's Leaf arm is hit
    /// at deep levels.
    #[test]
    fn descent_idempotent_on_compressed_leaf_at_level_13() {
        use ht_octree::NodeStore;

        let world_level = 13u32;
        let chunks_per_axis = 1u32 << (world_level - CHUNK_LEVEL);
        let mut store = NodeStore::new();
        let stone = store.leaf(stone_cell());
        // All-stone level-13 root via a single uniform Leaf — the descent
        // sees raw `Leaf` at every contextual level from 13 down.
        let world_root = store.interior(world_level, [stone; 8]);

        for &player in &[
            ChunkCoord::new(0, 0, 0),
            ChunkCoord::new(
                chunks_per_axis / 2,
                chunks_per_axis / 2,
                chunks_per_axis / 2,
            ),
            ChunkCoord::new(
                chunks_per_axis - 1,
                chunks_per_axis - 1,
                chunks_per_axis - 1,
            ),
        ] {
            let mut policy = ChunkLodPolicy::new();
            policy.enabled = true;
            let view_a = policy.recompute(&mut store, world_root, world_level, player);
            let lod_a = policy.chunk_lod.clone();
            let view_b = policy.recompute(&mut store, world_root, world_level, player);
            assert_eq!(
                view_a, view_b,
                "compressed-leaf scene: view_root churn at player={player:?}"
            );
            assert_eq!(
                policy.chunk_lod, lod_a,
                "compressed-leaf scene: chunk_lod churn at player={player:?}"
            );
        }
    }

    /// Box-Chebyshev arithmetic spot checks. Verifies the helper directly
    /// before relying on it through descent.
    #[test]
    fn chunk_box_chebyshev_bounds_basic_cases() {
        // Box at origin, player inside.
        let b = ChunkBox {
            lo: [0, 0, 0],
            side: 4,
        };
        let (lo, hi) = b.chebyshev_bounds(ChunkCoord::new(2, 2, 2));
        assert_eq!((lo, hi), (0.0, 2.0));
        // Box at origin, player just outside.
        let (lo, hi) = b.chebyshev_bounds(ChunkCoord::new(5, 0, 0));
        assert_eq!((lo, hi), (2.0, 5.0));
        // 1×1×1 box (CHUNK_LEVEL).
        let b = ChunkBox {
            lo: [3, 4, 5],
            side: 1,
        };
        let (lo, hi) = b.chebyshev_bounds(ChunkCoord::new(0, 0, 0));
        assert_eq!((lo, hi), (5.0, 5.0));
        // Player far outside (signed safety).
        let b = ChunkBox {
            lo: [0, 0, 0],
            side: 32,
        };
        let (lo, hi) = b.chebyshev_bounds(ChunkCoord::new(1000, 0, 0));
        assert_eq!((lo, hi), ((1000 - 31) as f64, 1000.0));
    }
}
