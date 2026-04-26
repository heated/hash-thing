//! Per-chunk LOD policy for camera-anchored streaming (cswp.8.3).
//!
//! Wires the cswp.8.2 [`NodeStore::lod_collapse_chunk`] primitive into a
//! per-frame chunk-LOD policy. Each frame, [`ChunkLodPolicy::update`]
//! derives a "view root" from the canonical sim root by collapsing far
//! chunks per the design-doc curve. The sim continues to read/write
//! the canonical root; the policy is render-only this iteration.
//!
//! Design anchors: `docs/perf/cswp-lod.md` §2 (distance→LOD curve),
//! §4 (hard swap + ±0.25 chunk-unit hysteresis), §6 (chunk_level=7 starting
//! point).
//!
//! Cache key includes the source `world_root` NodeId, so any mutation that
//! changes the canonical root — sim step, player edit, scene swap, terrain
//! reseed, [`World::compact()`](crate::sim::World) — auto-invalidates the
//! cached view. There is no external `invalidate()` API to forget.

use ht_octree::{NodeId, NodeStore};
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
/// recompute. Pinning `world_root` makes the cache self-defensive: every
/// mutation path that produces a fresh root NodeId (sim step, edit, scene
/// swap, compaction remap) auto-invalidates without any external bookkeeping.
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
/// and the optional growth baseline for `log::warn!` heuristics.
pub struct ChunkLodPolicy {
    pub enabled: bool,
    pub lod_bias: f32,
    pub max_lod: u8,
    chunk_lod: FxHashMap<ChunkCoord, u8>,
    cache: Option<(CacheKey, NodeId)>,
    baseline_store_nodes: Option<usize>,
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
        }
    }

    /// Compute the view root for this frame.
    ///
    /// Returns `world_root` unchanged when LOD is disabled or the world fits
    /// inside the near band (`world_level <= CHUNK_LEVEL`, e.g. 256³ default).
    /// Otherwise walks every chunk in lex order, computes target LOD with
    /// hysteresis, and chains [`NodeStore::lod_collapse_chunk`] calls.
    ///
    /// Self-defensive: any change to `world_root` (sim step, edit, scene
    /// swap, compaction) yields a different cache key → automatic recompute.
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

    /// Recompute path. Invalidates `chunk_lod` against the current world
    /// dimensions, then walks all chunks in lex order. Every chunk —
    /// including LOD 0 — is recorded in the new map so:
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
}
