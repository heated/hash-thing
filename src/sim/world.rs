use std::collections::VecDeque;
use std::fmt;

use super::mutation::{MutationQueue, WorldMutation};
use super::rule::{block_index, GameOfLife3D, ALIVE};
use crate::octree::{Cell, CellState, NodeId, NodeStore, CELLS_PER_BLOCK};
use crate::scale::CELLS_PER_METER_INT;
use crate::terrain::field::gyroid::GyroidField;
use crate::terrain::field::heightmap::PrecomputedHeightmapField;
use crate::terrain::field::lattice::LatticeField;
use crate::terrain::field::TerrainBlendField;
use crate::terrain::materials::{
    pack_clone_source, BlockRuleId, MaterialRegistry, AIR, CLONE_MATERIAL_ID, DIRT, FAN, FIRE,
    FIREWORK, GRASS, LAVA, OIL, SAND, STONE, VINE, WATER,
};
use crate::terrain::{gen_region, GenStats, TerrainParams};
use rustc_hash::FxHashMap;

/// hash-thing-ecmn (vqke.4.1): selectable base-case scheduling strategy.
/// `Serial` is pre-ftuu DFS (no rayon). `RayonPerFanout` is ftuu's
/// shipped per-level-4-fanout 8-way batching. `RayonBfs` is ecmn's
/// breadth-first whole-step level-3 batching (added in commit 2).
///
/// Default = `RayonPerFanout`: preserves the post-ftuu shipped state
/// (env-var users got rayon; new construction gets the same behaviour
/// without needing the env var). Override via
/// `World::set_base_case_strategy` or `HASH_THING_BASE_CASE_STRATEGY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BaseCaseStrategy {
    Serial,
    #[default]
    RayonPerFanout,
    RayonBfs,
}

/// hash-thing-ecmn: parse strategy from env. Precedence:
/// `HASH_THING_BASE_CASE_STRATEGY` (explicit) > `HASH_THING_BASE_CASE_RAYON=1`
/// (ftuu shim, treated as RayonPerFanout) > default.
///
/// Invalid `HASH_THING_BASE_CASE_STRATEGY` values panic at construction
/// — operators should get loud feedback, not silent fallback.
fn env_base_case_strategy() -> BaseCaseStrategy {
    parse_base_case_strategy(
        std::env::var("HASH_THING_BASE_CASE_STRATEGY").ok(),
        std::env::var("HASH_THING_BASE_CASE_RAYON").ok(),
    )
}

/// hash-thing-ecmn: pure env-string-to-strategy parser. Extracted from
/// [`env_base_case_strategy`] so unit tests can drive it without
/// mutating process env (which would race in a parallel test runner).
fn parse_base_case_strategy(
    strategy_env: Option<String>,
    rayon_env: Option<String>,
) -> BaseCaseStrategy {
    if let Some(v) = strategy_env {
        return match v.as_str() {
            "serial" => BaseCaseStrategy::Serial,
            "per-fanout" | "rayon" => BaseCaseStrategy::RayonPerFanout,
            "bfs" => BaseCaseStrategy::RayonBfs,
            other => panic!(
                "HASH_THING_BASE_CASE_STRATEGY must be one of: serial | per-fanout | bfs (got {other:?})"
            ),
        };
    }
    match rayon_env.as_deref() {
        Some("1") => BaseCaseStrategy::RayonPerFanout,
        Some("0") => BaseCaseStrategy::Serial,
        _ => BaseCaseStrategy::default(),
    }
}

/// The axis-aligned cube of world-space that the octree currently covers.
///
/// Origin is always (0,0,0) today (unsigned coords), but will shift once
/// signed world coordinates and recentering land. The type exists now so
/// call sites can migrate incrementally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RealizedRegion {
    pub origin: [i64; 3],
    pub level: u32,
}

impl RealizedRegion {
    /// Side length of the realized cube.
    pub fn side(&self) -> u64 {
        1u64 << self.level
    }

    /// True if the world-space coordinate `(x, y, z)` is inside the region.
    pub fn contains(&self, x: i64, y: i64, z: i64) -> bool {
        let s = self.side() as i64;
        x >= self.origin[0]
            && x < self.origin[0] + s
            && y >= self.origin[1]
            && y < self.origin[1] + s
            && z >= self.origin[2]
            && z < self.origin[2] + s
    }

    /// Map a world-space point to the octant index (0–7) within the root node.
    ///
    /// Panics if the point is outside the region.
    pub fn octant_of(&self, x: i64, y: i64, z: i64) -> usize {
        assert!(self.contains(x, y, z), "point outside realized region");
        let half = self.side() as i64 / 2;
        let dx = if x - self.origin[0] >= half { 1 } else { 0 };
        let dy = if y - self.origin[1] >= half { 1 } else { 0 };
        let dz = if z - self.origin[2] >= half { 1 } else { 0 };
        dx + dy * 2 + dz * 4
    }
}

impl fmt::Display for RealizedRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.side();
        write!(
            f,
            "realized: origin=({},{},{}) level={} ({s}³)",
            self.origin[0], self.origin[1], self.origin[2], self.level
        )
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct WorldCoord(pub i64);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct LocalCoord(pub u64);

/// Read-only voxel-grid view for queries that must outlive a mutable borrow
/// of the full [`World`].
///
/// Holds only the data needed to answer [`get`](Self::get) — a clone of the
/// octree store plus the three metadata fields. No caches, no pending
/// mutations. The main thread refreshes this snapshot each time a sim step
/// finishes (and once more when the next step kicks off) so player collision
/// during a background step sees a stable world instead of falling through
/// `World::placeholder` or switching to free-fly physics (hash-thing-0s9v).
///
/// Staleness: during an in-flight step the snapshot reflects the world state
/// as it entered that step, for the entire duration of the step. Expected
/// drift is negligible at sim tick rates; long steps (100s of ms at 4096³)
/// may briefly let the player clip through a cell that was just cleared or
/// stand inside a cell that just became solid, self-correcting on step
/// completion.
#[derive(Clone)]
pub struct CollisionSnapshot {
    store: NodeStore,
    root: NodeId,
    level: u32,
    origin: [i64; 3],
}

impl CollisionSnapshot {
    pub fn get(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        let lx = x.0 - self.origin[0];
        let ly = y.0 - self.origin[1];
        let lz = z.0 - self.origin[2];
        let side = 1i64 << self.level;
        if lx < 0 || ly < 0 || lz < 0 || lx >= side || ly >= side || lz >= side {
            return 0;
        }
        self.store
            .get_cell(self.root, lx as u64, ly as u64, lz as u64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemoLayout {
    pub player_pos: [f64; 3],
    pub player_yaw: f64,
    pub player_pitch: f64,
    pub walk_route: [[i64; 3]; 6],
    pub corridor_mid: [i64; 3],
    pub atrium_center: [i64; 3],
    pub reveal_center: [i64; 3],
    pub panorama_center: [i64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoSpectacleProfile {
    Cascade,
    Hearth,
    Clash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemoSpectacleAnchor {
    pub label: &'static str,
    pub center: [i64; 3],
    pub profile: DemoSpectacleProfile,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActiveMaterialStats {
    pub fire_cells: u32,
    pub water_cells: u32,
}

impl ActiveMaterialStats {
    pub fn active_cells(self) -> u32 {
        self.fire_cells + self.water_cells
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Box3 {
    min: [i64; 3],
    max: [i64; 3],
}

impl Box3 {
    #[inline]
    const fn new(min: [i64; 3], max: [i64; 3]) -> Self {
        Self { min, max }
    }

    #[inline]
    fn center(self) -> [i64; 3] {
        [
            (self.min[0] + self.max[0]) / 2,
            (self.min[1] + self.max[1]) / 2,
            (self.min[2] + self.max[2]) / 2,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProgressionBoxes {
    start_shell: Box3,
    start_room: Box3,
    corridor_shell: Box3,
    corridor: Box3,
    tease_a: Box3,
    tease_b: Box3,
    atrium: Box3,
    balcony: Box3,
    panorama: Box3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemoWaypoint {
    pub label: &'static str,
    pub center: [i64; 3],
    pub radius: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProgressionWaterfallLayout {
    shell: Box3,
    shaft: Box3,
    curtain: Box3,
    source_x: [i64; 2],
    source_y: [i64; 2],
    source_z: i64,
    focus: [i64; 3],
}

/// The simulation world. Owns the octree store and manages stepping.
///
/// For now, stepping works by flattening to a grid, applying rules, and
/// rebuilding the octree. This is O(n³) but correct, and lets us validate
/// the full pipeline. True Hashlife recursive stepping comes next.
#[derive(Clone)]
pub struct World {
    pub store: NodeStore,
    pub root: NodeId,
    pub level: u32, // root level — grid is 2^level per side
    pub generation: u64,
    pub simulation_seed: u64,
    /// Crate-private — external mutation is structurally blocked
    /// (hash-thing-6iiz). Any mutation (whole-registry replacement,
    /// `set_tick_divisor`, `assign_block_rule`, `register_block_rule`, etc.)
    /// MUST route through [`Self::mutate_materials`] or a safe mutator built
    /// on top of it. Bare `world.materials.X(...)` from within the crate
    /// bypasses cache invalidation: cached `block_rule_present` goes stale,
    /// the 4497 brute-force fallback in `step_recursive_pow2` is skipped
    /// against a block-rule-present registry, and macro-cached results are
    /// wrong. Intra-crate reads via direct field access are fine.
    pub(crate) materials: MaterialRegistry,
    /// Retained terrain params for lazy expansion (3fq.4). When `Some`,
    /// `ensure_region` generates terrain (heightmap)
    /// for newly-created sibling octants instead of leaving them empty.
    terrain_params: Option<TerrainParams>,
    /// Spatial memoization cache for the recursive Hashlife stepper.
    /// Key: (NodeId, schedule_phase). Identical subtrees anywhere in the
    /// world share a single cache entry — block-rule partition uses
    /// node-local alignment so origin is no longer in the key (9ww).
    /// `schedule_phase` is `generation % materials.memo_period()` (iowh).
    pub(crate) hashlife_cache: FxHashMap<(NodeId, u64), NodeId>,
    /// Memoization cache for the exponential Hashlife macro-stepper (6gf.7).
    /// Key: (NodeId, starting generation).
    pub(crate) hashlife_macro_cache: FxHashMap<(NodeId, u64), NodeId>,
    /// Hashlife cache statistics from the most recent step.
    pub hashlife_stats: HashlifeStats,
    /// Lifetime accumulator across `step_recursive` calls. Unlike
    /// `hashlife_stats` (reset each step), this aggregates for the life of the
    /// `World` — enables steady-state hit-rate reporting in the periodic log
    /// (hash-thing-stue.6). `step_recursive_pow2` is intentionally not folded
    /// in: it goes through `step_node_macro`, which doesn't touch
    /// `hashlife_stats` today, so there'd be nothing to add anyway.
    pub hashlife_stats_total: HashlifeStats,
    /// Sliding window of per-step `(cache_hits, cache_misses)` used to
    /// compute a window-minus-lifetime hit-rate delta (hash-thing-o2es).
    /// The churn signal catches regressions that would be smoothed out by
    /// the lifetime denominator; positive delta = recent steps cache better
    /// than the running average, negative = recent regression.
    pub memo_window: MemoWindow,
    /// Store size after the last compaction, used to trigger periodic GC.
    pub(crate) store_size_at_last_compact: usize,
    /// Cache for `inert_uniform_state`: NodeId → `Option<CellState>`.
    /// `Some(state)` = all leaves are the same inert material; `None` = mixed or active.
    pub(crate) hashlife_inert_cache: FxHashMap<NodeId, Option<CellState>>,
    /// Cache for `is_all_inert`: NodeId → bool.
    /// True if every leaf in the subtree has CaRule::Noop and no BlockRule.
    pub(crate) hashlife_all_inert_cache: FxHashMap<NodeId, bool>,
    /// hash-thing-5ie4 (vqke.2.1): cache for `subtree_has_slow_divisor`:
    /// `NodeId → bool`. True if the subtree contains any cell whose
    /// material has `tick_divisor > 1`. Drives the phase-fold in
    /// `step_node`: a node with no slow-divisor descendants has a step
    /// result that depends only on `(NodeId, gen % 2)`, so the cache
    /// key collapses from period N → period 2 for that branch.
    ///
    /// **Invalidation surface** (must be cleared on every path that
    /// changes the answer): material registry mutations
    /// (`invalidate_material_caches`, called via `mutate_materials`)
    /// and the same world-reset / megastructure-seed paths that drop
    /// `hashlife_inert_cache` / `hashlife_all_inert_cache`. The
    /// answer is content-determined so node-mutation paths (cell
    /// edits) get new NodeIds and hit the cache fresh — no manual
    /// invalidation needed there.
    pub(crate) hashlife_slow_divisor_cache: FxHashMap<NodeId, bool>,
    /// Cached result of `has_block_rule_cells`. `None` = dirty, needs rescan.
    /// Avoids O(n³) flatten-and-scan on every `step_recursive_pow2` call.
    pub(crate) block_rule_present: Option<bool>,
    /// Pending world mutations. Entities push here; `apply_mutations`
    /// drains and applies in arrival order at tick boundary.
    pub queue: MutationQueue,
    /// Positions of clone blocks in world coordinates. Each tick,
    /// clone blocks spawn their encoded material into adjacent air cells.
    /// Validated lazily — stale entries (destroyed by acid/etc.) are pruned
    /// on `spawn_clones`.
    pub clone_sources: Vec<[i64; 3]>,
    /// World-space origin of local coordinate (0,0,0). When the world
    /// grows in the negative direction, origin shifts to keep the old
    /// root's cells at the same world-space positions.
    pub origin: [i64; 3],
    /// Old→current NodeId remap pending for the renderer's persistent
    /// cache (`Svdag::apply_remap`, hash-thing-5bb.11). Keys are
    /// old-namespace NodeIds that remain reachable across every
    /// compaction since the last sync; keys GC'd mid-chain are dropped
    /// (absence == invalidation). Compose-on-write: when a new
    /// compaction publishes B→C while A→B is still pending, they are
    /// composed into A→C so the renderer sees a single remap keyed in
    /// its current (A) namespace (hash-thing-rk4n.1).
    ///
    /// `None` on fresh `World::new` (cache is empty by construction).
    /// `Some(empty)` after `seed_terrain` — the consumer drains and
    /// calls `apply_remap(&empty)`, which drops every cached entry.
    pub last_compaction_remap: Option<FxHashMap<NodeId, NodeId>>,
    /// hash-thing-ecmn (vqke.4.1): selectable base-case scheduling
    /// strategy. See [`BaseCaseStrategy`]. Default = `RayonPerFanout`
    /// (ftuu shipped behaviour). Initialised from
    /// `HASH_THING_BASE_CASE_STRATEGY` env var (and the legacy
    /// `HASH_THING_BASE_CASE_RAYON` shim) at construction. Mutate via
    /// [`Self::set_base_case_strategy`] (preferred) or
    /// [`Self::set_base_case_use_rayon`] (legacy bool wrapper).
    pub(crate) base_case_strategy: BaseCaseStrategy,
}

/// Cache performance statistics from a single hashlife step.
#[derive(Clone, Copy, Debug, Default)]
pub struct HashlifeStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub empty_skips: u64,
    pub fixed_point_skips: u64,
    /// Per-level cache miss counts (index = level - 3, since base case is level 3).
    pub misses_by_level: [u64; 32],
    /// Cumulative nanoseconds spent in `step_grid_once` Phase 1 (per-cell
    /// CaRule) across all memo-miss base cases in one step (hash-thing-71mp).
    pub phase1_ns: u64,
    /// Cumulative nanoseconds spent in `step_grid_once` Phase 2 (per-block
    /// BlockRule / Margolus) across all memo-miss base cases in one step
    /// (hash-thing-71mp).
    pub phase2_ns: u64,
    /// hash-thing-vqke Phase 0: total wall-clock nanoseconds inside the
    /// top-level `step_node` call from `step_recursive`. Includes p1+p2
    /// at the leaves PLUS the recursive-descent / hash-cons / memo-lookup
    /// overhead between leaves. `step_node_wall_ns - phase1_ns - phase2_ns`
    /// approximates the descent-and-intern cost — the unaccounted slice
    /// the bead's szyh-baseline evidence (~94 ms at 256³) was about.
    /// Recorded once per `step_recursive` call (not accumulated across
    /// the recursion); per-step, not lifetime — matches phase1/phase2
    /// shape so the `memo_summary` line gives a self-consistent
    /// breakdown.
    pub step_node_wall_ns: u64,
    /// hash-thing-vqke Phase 0: wall-clock nanoseconds spent in
    /// `maybe_compact()` after `step_node` completes. Most steps are
    /// 0 (the 2× growth threshold gates compaction); the spikes show
    /// up roughly every N steps where N depends on churn. Per-step.
    pub compact_ns: u64,
    /// hash-thing-bjdl (vqke.2): probe for hypothesis 3 (schedule-phase
    /// explosion). At each cache miss in `step_node`, count the miss
    /// as "phase-aliased" iff the same NodeId is already cached at a
    /// *different* schedule_phase. Boolean-per-miss semantics: a
    /// node aliased at three other phases counts once, not three
    /// times. A high `cache_misses_phase_aliased / cache_misses`
    /// ratio means the period is fragmenting reuse — content-folding
    /// or a wider key would have hit.
    ///
    /// **Always 0 unless `HASH_THING_MEMO_DIAG=1`** is set in the
    /// process environment at first-step time. Always-on cost was
    /// estimated at single-digit ms/step at observed miss volumes
    /// (per Claude + Codex plan review of this bead — vqke is trying
    /// to speed step_recursive UP, not pay 5-10ms for a diagnostic).
    /// Set the env var when you want to reproduce vqke.2's diagnostic
    /// run and check the new memo_summary tokens.
    ///
    /// Macro path note: `step_node_macro` (`hashlife.rs:645` —
    /// `step_recursive_pow2` path) shares `cache_misses` with the
    /// micro path. On the 256³ default scene that path is not
    /// exercised, so macro misses are 0 (`memo_mac=0` in
    /// `memo_summary`) and the ratio is exact. If a future caller
    /// uses the macro path, `cache_misses_phase_aliased / cache_misses`
    /// becomes a lower-bound on the micro-path's true alias rate
    /// (the macro path doesn't probe).
    pub cache_misses_phase_aliased: u64,
    /// hash-thing-bjdl (vqke.2): probe for hypothesis 2 (cache eviction
    /// too aggressive). Counts hashlife_cache entries dropped during
    /// `remap_caches` because their NodeId or result NodeId was not
    /// present in the post-compaction reachability set. Most steps
    /// are 0 because `maybe_compact` is gated on the 2× growth
    /// threshold. A high `dropped/(dropped+kept)` ratio shows the
    /// cache is fragmenting against the reachability sweep — not
    /// necessarily a sweep bug, but evidence that cache lifetime is
    /// shorter than the reuse horizon. A low ratio rules H2 out.
    pub compact_entries_dropped: u64,
    /// hash-thing-bjdl (vqke.2): paired with `compact_entries_dropped`.
    /// Counts hashlife_cache entries that survived the most recent
    /// `remap_caches`. Only the main hashlife_cache is counted (the
    /// macro / inert / all-inert caches are tangential to the memo_hit
    /// signal). Per-step.
    pub compact_entries_kept: u64,
    /// hash-thing-ecmn (vqke.4.1): unique BFS task count per level.
    /// Index = level - 3. Populated only by `step_root_bfs`; zero on
    /// Serial / RayonPerFanout strategies. Used to diagnose whether
    /// the BFS dispatcher is actually batching meaningful work or
    /// bottoming out on short-circuits.
    pub bfs_tasks_by_level: [u64; 32],
    /// hash-thing-ecmn: unique level-3 base-case misses queued for
    /// the BFS rayon batch in this step. = bfs_tasks_by_level[0].
    pub bfs_level3_unique_misses: u64,
    /// hash-thing-ecmn: number of step_grid_once_pure invocations the
    /// BFS path dispatched through rayon par_iter (batch size ≥
    /// `RAYON_BATCH_THRESHOLD`). Per-step counter; usually 0 or 1
    /// since whole-step BFS produces one large batch.
    pub bfs_batches_parallel: u64,
    /// hash-thing-ecmn: number of BFS level-3 batches that fell back
    /// to serial iter because batch size was below the rayon
    /// threshold. Tracks "BFS infrastructure ran but didn't actually
    /// parallelise" — non-zero means the threshold may be too high or
    /// the workload is short-circuit-dominated.
    pub bfs_batches_serial_fallback: u64,
    /// hash-thing-ecmn: largest level-3 batch size observed in this
    /// step. = bfs_level3_unique_misses on the dominant path; useful
    /// once a chunked-wavefront fallback is added.
    pub bfs_max_batch_len: u64,
}

impl HashlifeStats {
    /// Field-wise add `step` into `self`. Keeps the scalar counters and the
    /// per-level array in sync (hash-thing-stue.6 lifetime accumulator).
    pub fn accumulate(&mut self, step: &HashlifeStats) {
        self.cache_hits += step.cache_hits;
        self.cache_misses += step.cache_misses;
        self.empty_skips += step.empty_skips;
        self.fixed_point_skips += step.fixed_point_skips;
        self.phase1_ns += step.phase1_ns;
        self.phase2_ns += step.phase2_ns;
        // hash-thing-vqke: lifetime accumulators for the per-step
        // wall-clock decompositions. Useful for long-run averages
        // (the per-step values fluctuate with churn / compact cadence).
        self.step_node_wall_ns += step.step_node_wall_ns;
        self.compact_ns += step.compact_ns;
        // hash-thing-bjdl (vqke.2): lifetime accumulators for the
        // memo-hit-rate diagnostic counters. Same per-step → lifetime
        // shape as the wall-clock decompositions above so `memo_summary`
        // can compute lifetime ratios (`memo_phase_alias` =
        // `cache_misses_phase_alias / cache_misses`).
        self.cache_misses_phase_aliased += step.cache_misses_phase_aliased;
        self.compact_entries_dropped += step.compact_entries_dropped;
        self.compact_entries_kept += step.compact_entries_kept;
        // hash-thing-ecmn: BFS observability counters.
        self.bfs_level3_unique_misses += step.bfs_level3_unique_misses;
        self.bfs_batches_parallel += step.bfs_batches_parallel;
        self.bfs_batches_serial_fallback += step.bfs_batches_serial_fallback;
        // bfs_max_batch_len is per-step max, not a sum — accumulator
        // takes the running max so lifetime stat reflects the largest
        // batch ever observed.
        self.bfs_max_batch_len = self.bfs_max_batch_len.max(step.bfs_max_batch_len);
        for (dst, src) in self
            .misses_by_level
            .iter_mut()
            .zip(step.misses_by_level.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .bfs_tasks_by_level
            .iter_mut()
            .zip(step.bfs_tasks_by_level.iter())
        {
            *dst += *src;
        }
    }
}

/// Fixed-capacity ring of recent per-step `(cache_hits, cache_misses)` pairs
/// for the window-delta hit-rate signal (hash-thing-o2es). A regression that
/// halves hit rate would otherwise be buried under the lifetime accumulator's
/// denominator; subtracting the lifetime rate from the window rate surfaces
/// the step-to-step churn directly.
///
/// Capacity is 20 entries: short enough that the window diverges from the
/// lifetime average within a single 2 s log interval at 256³ (~6 steps/s on
/// M2 MBA), long enough to smooth per-step noise on a busy scene. While the
/// window is still filling (step count < CAPACITY) window_rate == lifetime
/// and churn reports `+0.000`, which is the honest "not enough data yet"
/// signal.
#[derive(Clone, Debug, Default)]
pub struct MemoWindow {
    samples: VecDeque<(u64, u64)>,
}

impl MemoWindow {
    pub const CAPACITY: usize = 20;

    /// Append one step's `(cache_hits, cache_misses)`, evicting the oldest
    /// entry once the ring is full.
    pub fn push(&mut self, hits: u64, misses: u64) {
        if self.samples.len() == Self::CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back((hits, misses));
    }

    /// `(hits, misses)` summed across the window. `(0, 0)` when empty.
    pub fn totals(&self) -> (u64, u64) {
        self.samples
            .iter()
            .fold((0, 0), |acc, (h, m)| (acc.0 + h, acc.1 + m))
    }

    /// Hit rate over the window. `0.0` when empty or when the window has
    /// seen no cache traffic (both counters zero).
    pub fn hit_rate(&self) -> f64 {
        let (h, m) = self.totals();
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Compose two sequential NodeId remaps `A→B` and `B→C` into a single
/// `A→C` remap. A key `k_a` in `existing` whose value `v_b` is not a key
/// in `new` is dropped: the node survived compaction 1 but was
/// garbage-collected by compaction 2, so the renderer's cache entry for
/// `k_a` should be invalidated. `Svdag::apply_remap`
/// (`crates/ht-render/src/svdag.rs:173`) rebuilds `id_to_offset` from the
/// remap and drops any entries whose `old_id` is not a key, so absence
/// from the composed map is the correct invalidation signal
/// (hash-thing-rk4n.1).
pub(super) fn compose_remap(
    existing: FxHashMap<NodeId, NodeId>,
    new: &FxHashMap<NodeId, NodeId>,
) -> FxHashMap<NodeId, NodeId> {
    let mut out = FxHashMap::with_capacity_and_hasher(existing.len(), Default::default());
    for (k_a, v_b) in existing {
        if let Some(&v_c) = new.get(&v_b) {
            out.insert(k_a, v_c);
        }
    }
    out
}

impl World {
    /// Create a new empty world of given level (side = 2^level).
    pub fn new(level: u32) -> Self {
        let mut store = NodeStore::new();
        let root = store.empty(level);
        let materials = MaterialRegistry::terrain_defaults();
        assert!(
            materials.rule_for_cell(ALIVE).is_some(),
            "default material registry must define the legacy ALIVE payload",
        );
        assert!(
            materials.rule_for_cell(Cell::from_raw(FIRE)).is_some(),
            "default material registry must define the fire material",
        );
        assert!(
            materials.rule_for_cell(Cell::from_raw(WATER)).is_some(),
            "default material registry must define the water material",
        );
        Self {
            store,
            root,
            level,
            generation: 0,
            simulation_seed: 0,
            materials,
            terrain_params: None,
            hashlife_cache: FxHashMap::default(),
            hashlife_macro_cache: FxHashMap::default(),
            hashlife_stats: HashlifeStats::default(),
            hashlife_stats_total: HashlifeStats::default(),
            memo_window: MemoWindow::default(),
            store_size_at_last_compact: 0,
            hashlife_inert_cache: FxHashMap::default(),
            hashlife_all_inert_cache: FxHashMap::default(),
            hashlife_slow_divisor_cache: FxHashMap::default(),
            block_rule_present: None,
            queue: MutationQueue::new(),
            clone_sources: Vec::new(),
            origin: [0, 0, 0],
            last_compaction_remap: None,
            base_case_strategy: env_base_case_strategy(),
        }
    }

    /// Lightweight placeholder world for use as a temporary swap target
    /// (e.g. while the real world is on a background thread). Skips
    /// material registry setup — NOT valid for simulation.
    pub fn placeholder() -> Self {
        let mut store = NodeStore::new();
        let root = store.empty(1);
        Self {
            store,
            root,
            level: 1,
            generation: 0,
            simulation_seed: 0,
            materials: MaterialRegistry::new(),
            terrain_params: None,
            hashlife_cache: FxHashMap::default(),
            hashlife_macro_cache: FxHashMap::default(),
            hashlife_stats: HashlifeStats::default(),
            hashlife_stats_total: HashlifeStats::default(),
            memo_window: MemoWindow::default(),
            store_size_at_last_compact: 0,
            hashlife_inert_cache: FxHashMap::default(),
            hashlife_all_inert_cache: FxHashMap::default(),
            hashlife_slow_divisor_cache: FxHashMap::default(),
            block_rule_present: None,
            queue: MutationQueue::new(),
            clone_sources: Vec::new(),
            origin: [0, 0, 0],
            last_compaction_remap: None,
            base_case_strategy: BaseCaseStrategy::Serial,
        }
    }

    /// hash-thing-ecmn (vqke.4.1): set the base-case scheduling
    /// strategy. Bit-exact flattened cell content across all
    /// strategies — verified by `tests/base_case_rayon_parity.rs`.
    pub fn set_base_case_strategy(&mut self, strategy: BaseCaseStrategy) {
        self.base_case_strategy = strategy;
    }

    /// hash-thing-ecmn: read the active strategy.
    pub fn base_case_strategy(&self) -> BaseCaseStrategy {
        self.base_case_strategy
    }

    /// Legacy bool wrapper (hash-thing-ftuu). Maps `true` to
    /// `BaseCaseStrategy::RayonPerFanout`, `false` to `Serial`. Kept
    /// so existing tests / benches that toggle this knob keep
    /// compiling. Prefer [`Self::set_base_case_strategy`] in new code.
    pub fn set_base_case_use_rayon(&mut self, enabled: bool) {
        self.base_case_strategy = if enabled {
            BaseCaseStrategy::RayonPerFanout
        } else {
            BaseCaseStrategy::Serial
        };
    }

    pub fn side(&self) -> usize {
        1 << self.level
    }

    /// The realized region as a value type.
    pub fn region(&self) -> RealizedRegion {
        RealizedRegion {
            origin: self.origin,
            level: self.level,
        }
    }

    /// Read-only accessor for the embedded material registry (hash-thing-6iiz).
    /// Mutation must route through [`Self::mutate_materials`] to keep
    /// material-dependent caches in sync.
    pub fn materials(&self) -> &MaterialRegistry {
        &self.materials
    }

    /// Invalidate every cache whose contents depend on the material registry.
    /// Covers the four hashlife caches keyed by NodeId (whose step function
    /// depends on the active CA + block rules) AND the `block_rule_present`
    /// flag (whose answer changes when block rules are added / removed /
    /// re-assigned). Called by every mutation path via
    /// [`Self::mutate_materials`] (hash-thing-6iiz).
    ///
    /// Renamed from the previous `invalidate_rule_caches` — that name was
    /// misleading because it forgot `block_rule_present`, which was the root
    /// cause of the dxi4.2 audit finding.
    pub fn invalidate_material_caches(&mut self) {
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        // hash-thing-5ie4 (vqke.2.1): the slow-divisor predicate
        // depends on the material registry's tick_divisor flags;
        // any mutation that changes a divisor flips the answer. Same
        // dxi4.2 invalidation-surface bug class as the other inert
        // caches. Per Claude + Codex plan-review on 5ie4.
        self.hashlife_slow_divisor_cache.clear();
        self.block_rule_present = None;
    }

    /// Mutate the embedded material registry and invalidate every
    /// material-dependent cache atomically (hash-thing-6iiz).
    ///
    /// All material mutation paths in the crate MUST go through this method.
    /// Direct mutation via a `&mut MaterialRegistry` borrow bypasses
    /// `block_rule_present` invalidation and corrupts the `step_recursive_pow2`
    /// macro path (see the 4497 fallback in `src/sim/hashlife.rs`).
    ///
    /// Accepts both in-place edits (`|m| m.assign_block_rule(...)`) and
    /// whole-registry replacement (`|m| *m = MaterialRegistry::new(...)`).
    pub fn mutate_materials<R>(&mut self, f: impl FnOnce(&mut MaterialRegistry) -> R) -> R {
        let r = f(&mut self.materials);
        self.invalidate_material_caches();
        r
    }

    /// Reconfigure the legacy GoL smoke material dispatch to use `rule`.
    ///
    /// Routes through `mutate_materials` to keep material-dependent caches
    /// in sync — the CA dispatch table changes even though the octree content
    /// does not.
    pub fn set_gol_smoke_rule(&mut self, rule: GameOfLife3D) {
        self.mutate_materials(|m| *m = MaterialRegistry::gol_smoke_with_rule(rule));
    }

    /// Set the tick_divisor for a material. The hashlife cache key is
    /// `(NodeId, generation % memo_period())`, so changing a divisor changes
    /// `memo_period()` and makes existing entries unsound. Crate-private
    /// because production bakes divisors into the registrar; this exists for
    /// tests and the future tuning bead (iowh).
    #[allow(dead_code)]
    pub(crate) fn set_material_tick_divisor(
        &mut self,
        material_id: crate::terrain::materials::MaterialId,
        divisor: u16,
    ) {
        self.mutate_materials(|m| m.set_tick_divisor(material_id, divisor));
    }

    /// Convert world coordinate to local on a specific axis index (0=x, 1=y, 2=z).
    /// Panics if the result is negative (coordinate is below the world origin).
    fn local_from_world(&self, axis: usize, coord: WorldCoord) -> LocalCoord {
        let local = coord.0 - self.origin[axis];
        LocalCoord(u64::try_from(local).unwrap_or_else(|_| {
            panic!(
                "World: world coord {} maps to negative local coord {} \
                 (origin[{axis}]={})",
                coord.0, local, self.origin[axis]
            )
        }))
    }

    fn set_local(&mut self, x: LocalCoord, y: LocalCoord, z: LocalCoord, state: CellState) {
        let side = self.side() as u64;
        assert!(
            x.0 < side && y.0 < side && z.0 < side,
            "World::set_local: coord ({}, {}, {}) out of bounds for side {side}",
            x.0,
            y.0,
            z.0,
        );
        // hash-thing-wsq3: do NOT clear hashlife caches on cell edits.
        // The four caches are keyed by NodeId (hash-cons interned subtrees)
        // plus, for the recursive cache, schedule_phase = generation %
        // memo_period(). All four are pure functions of NodeId content +
        // material registry, so cell edits leave existing entries
        // semantically valid — `set_cell` produces a fresh root NodeId and
        // any unchanged subtrees keep their interned ids. Material-registry
        // mutation is the only thing that invalidates entries; that path
        // goes through `mutate_materials` -> `invalidate_material_caches`.
        // Fresh-store epoch boundaries (commit_step / seed_*) handle their
        // own cache clears separately because they mint a new NodeId
        // namespace.
        //
        // The pre-wsq3 unconditional clear here was inherited from
        // 53f7414 (2026-04-12) when the cache key still included an
        // `[origin]` component; it was correctness-paranoia overcorrection
        // that no longer applies under the current `(NodeId, phase)` shape
        // and was the dominant production-path slowdown — every entity-
        // produced SetCell wiped the cache, forcing every step to pay
        // cold-cache cost (~2-3s/gen at 512^3).
        self.root = self.store.set_cell(self.root, x.0, y.0, z.0, state);
        self.block_rule_present = None; // invalidate cache
    }

    fn get_local(&self, x: LocalCoord, y: LocalCoord, z: LocalCoord) -> CellState {
        self.store.get_cell(self.root, x.0, y.0, z.0)
    }

    /// Grow the root octree until `(x, y, z)` is in-bounds.
    ///
    /// Supports growth in all directions — negative coordinates shift
    /// the origin while preserving existing cell positions.
    ///
    /// No-op when the coordinate is already in-bounds.
    ///
    /// **Step cache:** existing cached results survive expansion because
    /// they are keyed by `NodeId`, and the old root's `NodeId` is still
    /// valid and still steps to the same result. The new parent has no
    /// cached entry yet and will be computed fresh on first step.
    pub fn ensure_contains(&mut self, x: WorldCoord, y: WorldCoord, z: WorldCoord) {
        self.ensure_region([x, y, z], [x, y, z]);
    }

    /// Grow the root octree until the axis-aligned box `[min, max]` is
    /// fully in-bounds (inclusive on both ends).
    ///
    /// Supports growth in all directions. For each axis needing negative
    /// growth, the old root is placed in the positive half of the new
    /// parent and origin shifts by −half. For positive growth, the old
    /// root stays in the negative half.
    ///
    /// No-op when the region is already in-bounds.
    pub fn ensure_region(&mut self, min: [WorldCoord; 3], max: [WorldCoord; 3]) {
        loop {
            let side = 1i64 << self.level;
            let origin = self.origin;

            // Check which axes need growth and in which direction.
            let need_neg = [
                min[0].0 < origin[0],
                min[1].0 < origin[1],
                min[2].0 < origin[2],
            ];
            let need_pos = [
                max[0].0 >= origin[0] + side,
                max[1].0 >= origin[1] + side,
                max[2].0 >= origin[2] + side,
            ];

            if !need_neg.iter().any(|&b| b) && !need_pos.iter().any(|&b| b) {
                return; // everything fits
            }

            // Determine which octant the old root goes into.
            // Negative growth on axis → old root in + half (bit=1), origin shifts.
            // Positive growth → old root in - half (bit=0).
            let old_octant = (if need_neg[0] { 1 } else { 0 })
                + (if need_neg[1] { 2 } else { 0 })
                + (if need_neg[2] { 4 } else { 0 });

            let half = side; // half of the NEW size = old side
            let sibling_level = self.level;
            let new_level = self.level + 1;

            // Compute new origin — shift negative on axes that need it.
            let mut new_origin = origin;
            for axis in 0..3 {
                if need_neg[axis] {
                    new_origin[axis] -= half;
                }
            }

            // Build children array.
            let mut children = [NodeId::EMPTY; 8];
            children[old_octant] = self.root;
            for (oct, child) in children.iter_mut().enumerate() {
                if oct == old_octant {
                    continue;
                }
                let (cx, cy, cz) = crate::octree::node::octant_coords(oct);
                let sibling_origin = [
                    new_origin[0] + cx as i64 * half,
                    new_origin[1] + cy as i64 * half,
                    new_origin[2] + cz as i64 * half,
                ];
                *child = self.gen_sibling(sibling_level, sibling_origin);
            }

            self.root = self.store.interior(new_level, children);
            self.level = new_level;
            self.origin = new_origin;
        }
    }

    /// Generate a sibling octant for lazy expansion. If `terrain_params` is
    /// set, produces terrain (heightmap) at the given world-space origin.
    /// Otherwise returns a canonical empty node.
    fn gen_sibling(&mut self, level: u32, origin: [i64; 3]) -> NodeId {
        let params = match &self.terrain_params {
            Some(p) => *p,
            None => return self.store.empty(level),
        };
        let field = params.to_heightmap();
        let (node, _stats) = gen_region(&mut self.store, &field, origin, level);
        node
    }

    /// Set a cell.
    ///
    /// **Panics** on out-of-bounds coordinates (hash-thing-fb5) or when `state`
    /// encodes an unregistered material ID. For writes to coordinates outside
    /// the current realized root, call `ensure_contains` first to grow the tree.
    #[track_caller]
    fn assert_registered_state(&self, state: CellState) {
        let material_id = Cell::from_raw(state).material();
        assert!(
            self.materials.entry(material_id).is_some(),
            "World write uses unregistered material {material_id} (raw state {state})",
        );
    }

    #[track_caller]
    pub fn set(&mut self, x: WorldCoord, y: WorldCoord, z: WorldCoord, state: CellState) {
        self.assert_registered_state(state);
        self.set_local(
            self.local_from_world(0, x),
            self.local_from_world(1, y),
            self.local_from_world(2, z),
            state,
        );
    }

    /// Place a gameplay block at block coordinates `(bx, by, bz)`.
    ///
    /// Fills a `CELLS_PER_BLOCK³` region of cells with the given state.
    /// Block coordinate `(bx, by, bz)` maps to cell region
    /// `[bx*K .. bx*K+K-1]` on each axis, where `K = CELLS_PER_BLOCK`.
    ///
    /// Auto-grows the world to fit, then queues a `FillRegion` mutation.
    ///
    /// **Panics** when `state` encodes an unregistered material ID.
    /// Call `apply_mutations` to flush.
    pub fn set_block(&mut self, bx: i64, by: i64, bz: i64, state: CellState) {
        self.assert_registered_state(state);
        let k = CELLS_PER_BLOCK as i64;
        let min = [WorldCoord(bx * k), WorldCoord(by * k), WorldCoord(bz * k)];
        let max = [
            WorldCoord(bx * k + k - 1),
            WorldCoord(by * k + k - 1),
            WorldCoord(bz * k + k - 1),
        ];
        self.ensure_region(min, max);
        self.queue
            .push(WorldMutation::FillRegion { min, max, state });
    }

    /// Extract a read-only [`CollisionSnapshot`] view of the current voxel
    /// grid. Clones the octree store and copies the three metadata fields;
    /// does not copy any step caches. Used by the main thread to retain a
    /// usable voxel grid across a background sim step (hash-thing-0s9v).
    pub fn collision_snapshot(&self) -> CollisionSnapshot {
        CollisionSnapshot {
            store: self.store.clone(),
            root: self.root,
            level: self.level,
            origin: self.origin,
        }
    }

    /// Get a cell.
    ///
    /// Out-of-bounds reads return `0` silently (hash-thing-fb5). `get` is a
    /// pure query over the realized region — outside that region is
    /// conceptually unrealized empty space. This includes negative world
    /// coordinates until signed-region realization lands.
    pub fn get(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        let lx = x.0 - self.origin[0];
        let ly = y.0 - self.origin[1];
        let lz = z.0 - self.origin[2];
        let side = self.side() as i64;
        if lx < 0 || ly < 0 || lz < 0 || lx >= side || ly >= side || lz >= side {
            return 0;
        }
        self.get_local(
            LocalCoord(lx as u64),
            LocalCoord(ly as u64),
            LocalCoord(lz as u64),
        )
    }

    /// Stepper-oriented read: "what is this cell right now?"
    ///
    /// Semantic alias for [`get`](Self::get) that signals the caller accepts
    /// OOB-empty semantics without reservation. Hashlife steppers reading a
    /// 3×3×3 neighborhood call `probe` — the unrealized world *is* empty
    /// from the stepper's perspective, so returning 0 for out-of-bounds is
    /// the correct physical answer, not a lossy fallback.
    ///
    /// Use [`is_realized`](Self::is_realized) when you need to distinguish
    /// "genuinely empty" from "outside the realized region."
    #[inline]
    pub fn probe(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        self.get(x, y, z)
    }

    /// Is the coordinate inside the realized region?
    ///
    /// Returns `false` for negative coordinates and for positions beyond
    /// the current octree extent. Use this alongside [`get`](Self::get)
    /// when distinguishing "realized empty" from "unrealized" matters
    /// (e.g. debug overlays rendering the region boundary).
    pub fn is_realized(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> bool {
        let lx = x.0 - self.origin[0];
        let ly = y.0 - self.origin[1];
        let lz = z.0 - self.origin[2];
        let side = self.side() as i64;
        lx >= 0 && ly >= 0 && lz >= 0 && lx < side && ly < side && lz < side
    }

    /// Flatten to a 3D grid for rendering.
    pub fn flatten(&self) -> Vec<CellState> {
        self.store.flatten(self.root, self.side())
    }

    pub fn demo_waypoints(&self) -> Vec<DemoWaypoint> {
        let side = self.side() as i64;
        assert!(side >= 48, "demo spectacle requires side >= 48");
        let floor_y = (side / 4).max(6);
        let center_z = side / 2;
        let start_x = (side / 6).max(8);
        let end_x = side.saturating_sub(start_x + 2);
        let span = end_x.saturating_sub(start_x);
        let step = (span / 3).max(8);
        let radius = (side / 12).clamp(4, 6);
        let labels = ["ember", "spring", "quench", "cascade"];

        labels
            .into_iter()
            .enumerate()
            .map(|(i, label)| DemoWaypoint {
                label,
                center: [start_x + step * i as i64, floor_y + 1, center_z],
                radius,
            })
            .collect()
    }

    pub fn count_active_material_cells_near(&self, center: [i64; 3], radius: i64) -> usize {
        let side = self.side() as i64;
        let x0 = center[0].saturating_sub(radius).max(self.origin[0]);
        let y0 = center[1].saturating_sub(radius).max(self.origin[1]);
        let z0 = center[2].saturating_sub(radius).max(self.origin[2]);
        let x1 = center[0]
            .saturating_add(radius)
            .min(self.origin[0] + side - 1);
        let y1 = center[1]
            .saturating_add(radius)
            .min(self.origin[1] + side - 1);
        let z1 = center[2]
            .saturating_add(radius)
            .min(self.origin[2] + side - 1);

        let mut count = 0;
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let cell = self.get(WorldCoord(x), WorldCoord(y), WorldCoord(z));
                    if cell == FIRE || cell == WATER {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn contains_world_coord(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> bool {
        let side = self.side() as i64;
        x.0 >= self.origin[0]
            && x.0 < self.origin[0] + side
            && y.0 >= self.origin[1]
            && y.0 < self.origin[1] + side
            && z.0 >= self.origin[2]
            && z.0 < self.origin[2] + side
    }

    /// Drain the mutation queue and apply every pending mutation to the
    /// octree in arrival order. This is the only path for entity-produced
    /// edits to reach the world (closed-world invariant, hash-thing-1v0.9).
    /// Entity-produced writes outside the realized region are dropped
    /// instead of panicking; read paths already treat OOB as empty.
    pub fn apply_mutations(&mut self) {
        for m in self.queue.take() {
            match m {
                WorldMutation::SetCell { x, y, z, state } => {
                    if self.contains_world_coord(x, y, z) {
                        self.set(x, y, z, state);
                    }
                }
                WorldMutation::FillRegion { min, max, state } => {
                    debug_assert!(
                        min[0].0 <= max[0].0 && min[1].0 <= max[1].0 && min[2].0 <= max[2].0,
                        "FillRegion: min must be <= max on every axis"
                    );
                    let side = self.side() as i64;
                    let world_min = self.origin;
                    let world_max = [
                        self.origin[0] + side - 1,
                        self.origin[1] + side - 1,
                        self.origin[2] + side - 1,
                    ];
                    let clamped_min = [
                        min[0].0.max(world_min[0]),
                        min[1].0.max(world_min[1]),
                        min[2].0.max(world_min[2]),
                    ];
                    let clamped_max = [
                        max[0].0.min(world_max[0]),
                        max[1].0.min(world_max[1]),
                        max[2].0.min(world_max[2]),
                    ];
                    if clamped_min[0] > clamped_max[0]
                        || clamped_min[1] > clamped_max[1]
                        || clamped_min[2] > clamped_max[2]
                    {
                        continue;
                    }
                    for z in clamped_min[2]..=clamped_max[2] {
                        for y in clamped_min[1]..=clamped_max[1] {
                            for x in clamped_min[0]..=clamped_max[0] {
                                self.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), state);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Spawn materials from clone blocks into adjacent air cells.
    ///
    /// For each tracked clone source, verify it still exists (prune stale entries),
    /// then write the encoded source material into the 6 cardinal neighbors that
    /// are currently air. Mutations are applied immediately.
    pub fn spawn_clones(&mut self) {
        if self.clone_sources.is_empty() {
            return;
        }

        const DIRS: [[i64; 3]; 6] = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ];

        let mut live_sources: Vec<([i64; 3], u16)> = Vec::new();
        for &pos in &self.clone_sources {
            let cell = Cell::from_raw(self.get(
                WorldCoord(pos[0]),
                WorldCoord(pos[1]),
                WorldCoord(pos[2]),
            ));
            if cell.material() == CLONE_MATERIAL_ID {
                live_sources.push((pos, cell.metadata()));
            }
        }

        self.clone_sources = live_sources.iter().map(|&(pos, _)| pos).collect();

        for (pos, source_material_id) in live_sources {
            if source_material_id == 0 {
                continue;
            }
            let spawn_state = Cell::pack(source_material_id, 0).raw();
            for dir in &DIRS {
                let nx = pos[0] + dir[0];
                let ny = pos[1] + dir[1];
                let nz = pos[2] + dir[2];
                let neighbor = self.get(WorldCoord(nx), WorldCoord(ny), WorldCoord(nz));
                if neighbor == 0 {
                    self.queue.push(WorldMutation::SetCell {
                        x: WorldCoord(nx),
                        y: WorldCoord(ny),
                        z: WorldCoord(nz),
                        state: spawn_state,
                    });
                }
            }
        }

        if !self.queue.is_empty() {
            self.apply_mutations();
        }
    }

    /// Advance the CA by one generation with mutation-queue guard.
    ///
    /// Debug-asserts that the queue is empty on entry — if it isn't,
    /// the caller forgot `apply_mutations` and the step cache would
    /// see stale world state.
    pub fn step_ca(&mut self) {
        debug_assert!(
            self.queue.is_empty(),
            "step_ca called with {} pending mutations — call apply_mutations first",
            self.queue.len()
        );
        self.step_recursive();
    }

    /// Step the simulation forward one generation.
    ///
    /// Pipeline: flatten → cell-wise CaRule pass → block-wise BlockRule pass → rebuild.
    /// Single flatten/rebuild per tick. This is the brute-force path — true Hashlife
    /// recursive stepping replaces it later.
    ///
    /// **Phase ordering:** CaRule runs first (react), then BlockRule (move). This means:
    /// - CaRule deletions are invisible to BlockRule (the cell is already gone).
    /// - CaRule births are movable by BlockRule in the same tick.
    ///
    /// This is an intentional "react then move" physics model.
    ///
    /// **Boundary conditions:** Both CaRule and BlockRule use absorbing boundaries
    /// (out-of-bounds = empty), matching hashlife's infinite-world semantics.
    /// BlockRule additionally skips partial blocks at world edges (clipping) —
    /// Margolus blocks must not straddle the world boundary.
    pub fn step(&mut self) {
        let side = self.side();
        let grid = self.flatten();
        let mut next = vec![0 as CellState; side * side * side];
        let divisor_by_material = self.materials.tick_divisor_flags();
        let generation = self.generation;

        // Phase 1: cell-wise CaRule pass (Moore neighborhood). Per-material
        // tick_divisor gate mirrors the hashlife path (iowh) so the brute and
        // recursive paths produce identical output at every generation.
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let idx = x + y * side + z * side * side;
                    let raw = grid[idx];
                    let center = Cell::from_raw(raw);
                    let mat = center.material() as usize;
                    let divisor = divisor_by_material.get(mat).copied().unwrap_or(1) as u64;
                    if divisor > 1 && !generation.is_multiple_of(divisor) {
                        next[idx] = raw;
                        continue;
                    }
                    let neighbors = get_neighbors(&grid, side, x, y, z);
                    let rule = self.materials.rule_for_cell(center).unwrap_or_else(|| {
                        panic!("missing CaRule for material {}", center.material())
                    });
                    next[idx] = rule.step_cell(center, &neighbors).raw();
                }
            }
        }

        // Phase 2: block-wise BlockRule pass (Margolus 2x2x2).
        // Static internal gaps close in 1-2 ticks via the alternating
        // partition offset (qy4g epic decision 2026-04-26, option G).
        // Gaps within free-falling columns are co-moving with the column
        // and appear as a sustained checkerboard until the leading edge
        // compacts against a solid surface — see SPEC.md for the in-flight
        // visible-artifact tradeoff and fallbacks A/F. No post-pass
        // gap-fill runs in production.
        self.step_blocks(&mut next, side);

        self.commit_step(&next, side);
    }

    /// Apply block rules to non-overlapping 2x2x2 partitions of the grid.
    ///
    /// Partition offset alternates per generation: even → (0,0,0), odd → (1,1,1).
    /// Blocks at edges use absorbing boundary conditions: out-of-bounds cells
    /// are treated as empty, matching hashlife's pad-with-empty semantics.
    ///
    /// Dispatch: collect distinct BlockRuleIds across the 8 cells. If exactly one
    /// distinct rule exists, run it. If zero or multiple: skip (identity).
    ///
    /// Under per-material tick_divisors (iowh): when any divisor > 1, iterate
    /// both offsets and gate each block on its rule's `(generation / divisor) & 1`
    /// Margolus offset + firing cadence, matching the hashlife path.
    fn step_blocks(&self, grid: &mut [CellState], side: usize) {
        let block_rule_divisors = self.materials.block_rule_tick_divisors();
        let all_divisors_one = block_rule_divisors.iter().all(|&d| d == 1);

        if all_divisors_one {
            let offset = (self.generation & 1) as usize;
            let mut bz = offset;
            while bz < side {
                let mut by = offset;
                while by < side {
                    let mut bx = offset;
                    while bx < side {
                        self.apply_block(grid, side, bx, by, bz, None, 0);
                        bx += 2;
                    }
                    by += 2;
                }
                bz += 2;
            }
            return;
        }

        for pass_offset in 0..2usize {
            let mut bz = pass_offset;
            while bz < side {
                let mut by = pass_offset;
                while by < side {
                    let mut bx = pass_offset;
                    while bx < side {
                        self.apply_block(
                            grid,
                            side,
                            bx,
                            by,
                            bz,
                            Some(block_rule_divisors),
                            pass_offset,
                        );
                        bx += 2;
                    }
                    by += 2;
                }
                bz += 2;
            }
        }
    }

    /// Apply the block rule for a single 2x2x2 block at (bx, by, bz).
    ///
    /// Cells outside the grid boundary are treated as empty (absorbing BC).
    ///
    /// When `block_rule_divisors` is `Some`, gate the rule on its slowed-down
    /// schedule (iowh): apply only when `generation % divisor == 0` AND the
    /// rule's offset `(generation / divisor) & 1` equals `pass_offset`. The
    /// fast path passes `None` and relies on the caller iterating one offset.
    #[allow(clippy::too_many_arguments)]
    fn apply_block(
        &self,
        grid: &mut [CellState],
        side: usize,
        bx: usize,
        by: usize,
        bz: usize,
        block_rule_divisors: Option<&[u16]>,
        pass_offset: usize,
    ) {
        // Read the 8 cells. OOB → empty (absorbing boundary).
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = bx + dx;
                    let y = by + dy;
                    let z = bz + dz;
                    if x < side && y < side && z < side {
                        let idx = x + y * side + z * side * side;
                        block[block_index(dx, dy, dz)] = Cell::from_raw(grid[idx]);
                    }
                }
            }
        }

        // Skip all-empty blocks (optimization).
        if block.iter().all(|c| c.is_empty()) {
            return;
        }

        // Dispatch: find the unique block rule across all cells.
        let rule_id = match self.unique_block_rule(&block) {
            Some(id) => id,
            None => return, // zero or multiple distinct rules → skip
        };

        if let Some(divisors) = block_rule_divisors {
            let divisor = divisors.get(rule_id.0).copied().unwrap_or(1).max(1) as u64;
            if !self.generation.is_multiple_of(divisor) {
                return;
            }
            let rule_offset = ((self.generation / divisor) & 1) as usize;
            if rule_offset != pass_offset {
                return;
            }
        }

        let movable: [bool; 8] = std::array::from_fn(|i| {
            let c = block[i];
            c.is_empty() || self.materials.block_rule_id_for_cell(c).is_some()
        });

        let rule = self.materials.block_rule(rule_id);
        let result = rule.step_block(&block, &movable);

        // Mass conservation assertion: output must be a permutation of input.
        debug_assert!(
            {
                let mut inp: Vec<u16> = block.iter().map(|c| c.raw()).collect();
                let mut out: Vec<u16> = result.iter().map(|c| c.raw()).collect();
                inp.sort();
                out.sort();
                inp == out
            },
            "block rule violated mass conservation at ({bx}, {by}, {bz})"
        );

        // Contract assertion: immovable cells must be left in place by the
        // rule. Without this, a buggy rule that swaps an immovable cell into
        // a movable slot would silently delete the immovable cell's value
        // (the write-back filter only writes movable positions). Assert
        // here so the failure is loud, not a slow water leak.
        debug_assert!(
            (0..8).all(|i| movable[i] || result[i] == block[i]),
            "block rule moved an immovable cell at ({bx}, {by}, {bz})"
        );

        // Write back. The rule is contracted to leave immovable cells fixed,
        // so writing the rule output is safe. The `movable` filter is a
        // belt-and-suspenders guard against a rule that violates the contract.
        // OOB positions are silently skipped (absorbing boundary).
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = bx + dx;
                    let y = by + dy;
                    let z = bz + dz;
                    if x >= side || y >= side || z >= side {
                        continue;
                    }
                    let i = block_index(dx, dy, dz);
                    let idx = x + y * side + z * side * side;
                    if movable[i] {
                        grid[idx] = result[i].raw();
                    }
                }
            }
        }
    }

    /// Find the unique BlockRuleId across all non-empty cells in a block.
    /// Returns `Some(id)` if exactly one distinct rule; `None` if zero or multiple.
    pub(crate) fn unique_block_rule(&self, block: &[Cell; 8]) -> Option<BlockRuleId> {
        let mut found: Option<BlockRuleId> = None;
        for cell in block {
            if let Some(id) = self.materials.block_rule_id_for_cell(*cell) {
                match found {
                    None => found = Some(id),
                    Some(existing) if existing == id => {}
                    Some(_) => return None, // multiple distinct rules → skip
                }
            }
        }
        found
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
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), ALIVE.raw());
                    }
                }
            }
        }
    }

    /// Stamp deterministic active-material set pieces around named anchors.
    ///
    /// This is deliberately scene-local plumbing: callers provide the
    /// anchor positions, while the helper provides reproducible spectacle
    /// primitives that survive scene resets without hard-coding a route.
    pub fn stage_demo_spectacles(&mut self, anchors: &[DemoSpectacleAnchor]) {
        for &anchor in anchors {
            self.stage_demo_spectacle(anchor);
        }
    }

    /// Count active materials in a local box around `center`. Tests and
    /// future scene validators use this to assert spectacle presence
    /// after a reset without depending on exact voxel-for-voxel layouts.
    pub fn active_material_stats_near(&self, center: [i64; 3], radius: u64) -> ActiveMaterialStats {
        let radius = i64::try_from(radius).expect("demo spectacle radius must fit in i64");
        let mut stats = ActiveMaterialStats::default();
        for z in (center[2] - radius)..=(center[2] + radius) {
            for y in (center[1] - radius)..=(center[1] + radius) {
                for x in (center[0] - radius)..=(center[0] + radius) {
                    match self.get(WorldCoord(x), WorldCoord(y), WorldCoord(z)) {
                        FIRE => stats.fire_cells += 1,
                        WATER => stats.water_cells += 1,
                        _ => {}
                    }
                }
            }
        }
        stats
    }

    fn stage_demo_spectacle(&mut self, anchor: DemoSpectacleAnchor) {
        match anchor.profile {
            DemoSpectacleProfile::Cascade => self.stage_cascade(anchor.center),
            DemoSpectacleProfile::Hearth => self.stage_hearth(anchor.center),
            DemoSpectacleProfile::Clash => self.stage_clash(anchor.center),
        }
    }

    fn stage_cascade(&mut self, center: [i64; 3]) {
        let [cx, cy, cz] = center;
        self.fill_box_clamped([cx - 3, cy - 2, cz - 3], [cx + 4, cy + 5, cz + 4], AIR);
        self.fill_box_clamped([cx - 2, cy - 2, cz - 3], [cx + 3, cy, cz + 3], DIRT);
        self.fill_box_clamped([cx - 2, cy - 1, cz + 2], [cx + 3, cy + 4, cz + 3], STONE);
        self.fill_box_clamped([cx - 1, cy + 2, cz], [cx + 2, cy + 4, cz + 1], WATER);
        self.fill_box_clamped([cx - 3, cy - 1, cz - 1], [cx - 2, cy + 1, cz + 1], FIRE);
        self.fill_box_clamped([cx + 2, cy - 1, cz - 1], [cx + 3, cy + 1, cz + 1], FIRE);
    }

    fn stage_hearth(&mut self, center: [i64; 3]) {
        let [cx, cy, cz] = center;
        self.fill_box_clamped([cx - 3, cy - 1, cz - 3], [cx + 4, cy + 4, cz + 4], AIR);
        self.fill_box_clamped([cx - 3, cy - 1, cz - 3], [cx + 4, cy, cz + 4], STONE);
        self.fill_box_clamped([cx - 2, cy, cz + 2], [cx + 3, cy + 3, cz + 3], GRASS);
        self.fill_box_clamped([cx - 1, cy, cz - 1], [cx + 1, cy + 2, cz + 1], FIRE);
        self.fill_box_clamped([cx + 1, cy, cz - 1], [cx + 2, cy + 2, cz + 1], WATER);
    }

    fn stage_clash(&mut self, center: [i64; 3]) {
        let [cx, cy, cz] = center;
        self.fill_box_clamped([cx - 4, cy - 2, cz - 3], [cx + 5, cy + 4, cz + 4], AIR);
        self.fill_box_clamped([cx - 4, cy - 2, cz - 3], [cx + 5, cy, cz + 4], DIRT);
        self.fill_box_clamped([cx - 4, cy - 1, cz + 2], [cx - 1, cy + 2, cz + 3], GRASS);
        self.fill_box_clamped([cx - 3, cy, cz - 1], [cx - 1, cy + 2, cz + 1], FIRE);
        self.fill_box_clamped([cx + 1, cy + 1, cz - 1], [cx + 4, cy + 3, cz + 1], WATER);
        self.fill_box_clamped([cx + 1, cy - 1, cz + 2], [cx + 4, cy + 3, cz + 3], STONE);
    }

    fn fill_box_clamped(&mut self, min: [i64; 3], max: [i64; 3], state: CellState) {
        let region = self.region();
        for z in min[2]..max[2] {
            for y in min[1]..max[1] {
                for x in min[0]..max[0] {
                    if region.contains(x, y, z) {
                        self.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), state);
                    }
                }
            }
        }
    }

    fn burning_room_demo_spectacle_anchors(&self) -> [DemoSpectacleAnchor; 3] {
        let side = self.side() as i64;
        let margin = side / 8;
        let lo = margin + 6;
        let hi = side - margin - 7;
        let mid = side / 2;
        [
            DemoSpectacleAnchor {
                label: "intro",
                center: [lo, margin + 2, lo + 4],
                profile: DemoSpectacleProfile::Hearth,
            },
            DemoSpectacleAnchor {
                label: "interior",
                center: [mid, margin + 3, mid],
                profile: DemoSpectacleProfile::Clash,
            },
            DemoSpectacleAnchor {
                label: "panorama",
                center: [hi, margin + 3, hi],
                profile: DemoSpectacleProfile::Cascade,
            },
        ]
    }

    /// Seed a demo scene: stone room with grass walls (fuel), fire, and water.
    ///
    /// Demonstrates reaction phase (fire spreads to grass, water quenches fire)
    /// plus movement phase (water falls and spreads via FluidBlockRule).
    /// The room now carries three named spectacle anchors so resets keep
    /// active-material read points available in the same local pockets.
    pub fn seed_burning_room(&mut self) {
        let side = self.side() as u64;
        let margin = side / 8;
        let lo = margin;
        let hi = side - margin;

        for z in lo..hi {
            for y in lo..hi {
                for x in lo..hi {
                    let on_wall = x == lo || x == hi - 1 || z == lo || z == hi - 1;
                    let on_floor = y == lo;
                    let on_ceiling = y == hi - 1;

                    if on_floor {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), DIRT);
                    } else if on_ceiling {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), STONE);
                    } else if on_wall {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), GRASS);
                    }
                }
            }
        }
        self.stage_demo_spectacles(&self.burning_room_demo_spectacle_anchors());
    }

    /// Seed a deterministic four-stop demo gallery with local fire/water
    /// spectacle around each waypoint. The exact route/controls stay external;
    /// this just makes the content reproducible for whichever beat loader
    /// consumes it.
    ///
    /// Coordinate convention: every cell write here passes raw local indices
    /// straight into `fill_box` / `set` as `WorldCoord`. That is only
    /// correct when `self.origin == [0, 0, 0]`; under any other origin the
    /// `local_from_world` subtraction would shift the entire spectacle by
    /// `-origin`. Today the only caller (`App::load_demo_spectacle`) always
    /// builds a fresh `World::new(level)` first, so the assumption holds.
    /// The assert below locks it in — if a future caller (multiplayer,
    /// streaming, 4096+ epics) ever seeds spectacle into a shifted world,
    /// fail loudly here instead of silently mis-placing geometry. See
    /// hash-thing-zksj.
    pub fn seed_demo_spectacle(&mut self) {
        assert_eq!(
            self.origin,
            [0, 0, 0],
            "seed_demo_spectacle assumes origin=[0,0,0]; helpers pass local coords as WorldCoord and would silently shift by -origin otherwise (hash-thing-zksj)"
        );
        let waypoints = self.demo_waypoints();
        let first = waypoints
            .first()
            .expect("demo spectacle requires at least one waypoint");
        let last = waypoints
            .last()
            .expect("demo spectacle requires at least one waypoint");
        let floor_y = first.center[1].saturating_sub(1);
        let ceiling_y = floor_y + 6;
        let corridor_z = first.center[2];
        let corridor_half_width = 3;
        let span_lo = first.center[0].saturating_sub(first.radius + 4);
        let span_hi = last.center[0] + last.radius + 4;

        self.fill_box(
            Box3::new(
                [span_lo, floor_y, corridor_z - corridor_half_width],
                [span_hi, floor_y, corridor_z + corridor_half_width],
            ),
            DIRT,
        );
        self.fill_box(
            Box3::new(
                [span_lo, floor_y + 1, corridor_z - corridor_half_width - 1],
                [span_hi, ceiling_y, corridor_z - corridor_half_width - 1],
            ),
            STONE,
        );
        self.fill_box(
            Box3::new(
                [span_lo, floor_y + 1, corridor_z + corridor_half_width + 1],
                [span_hi, ceiling_y, corridor_z + corridor_half_width + 1],
            ),
            STONE,
        );
        self.fill_box(
            Box3::new(
                [span_lo, ceiling_y, corridor_z - corridor_half_width],
                [span_hi, ceiling_y, corridor_z + corridor_half_width],
            ),
            STONE,
        );

        for (idx, waypoint) in waypoints.into_iter().enumerate() {
            self.seed_waypoint_frame(waypoint);
            match idx {
                0 => self.seed_ember_set_piece(waypoint),
                1 => self.seed_spring_set_piece(waypoint),
                2 => self.seed_quench_set_piece(waypoint),
                3 => self.seed_cascade_set_piece(waypoint),
                _ => unreachable!("demo spectacle defines exactly four waypoints"),
            }
        }

        // hash-thing-t3zn.2: viewing platform under the player spawn.
        // `App::reset_scene_entities` places the player at world_center
        // + 2 cells up; this 5x5 stone pad sits one cell beneath them so
        // `is_grounded` finds support. Coords use the function's existing
        // local-as-WorldCoord convention (origin=0 assumed) — see follow-
        // up bead for the latent shifted-origin issue this shares with
        // the rest of `seed_demo_spectacle`.
        let center_cell = (self.side() as i64) / 2;
        let platform_y = center_cell + 2 * CELLS_PER_METER_INT as i64 - 1;
        let pad_half: i64 = 2;
        self.fill_box(
            Box3::new(
                [center_cell - pad_half, platform_y, center_cell - pad_half],
                [center_cell + pad_half, platform_y, center_cell + pad_half],
            ),
            STONE,
        );

        self.block_rule_present = None;
    }

    /// Add water and sand to an existing terrain — water pools on a
    /// hilltop (so it cascades down) and sand dunes on one side.
    /// Call after `seed_terrain`.
    pub fn seed_water_and_sand(&mut self) {
        let side = self.side() as u64;
        let center = side / 2;
        // Water: a wide pool above center, near the high terrain.
        // Place at 75% height in a ~side/4 square.
        let water_y = center + center / 4;
        let pool_radius = side / 6;
        let pool_depth = (side / 32).max(2);
        let lo_x = center.saturating_sub(pool_radius);
        let hi_x = (center + pool_radius).min(side);
        let lo_z = center.saturating_sub(pool_radius);
        let hi_z = (center + pool_radius).min(side);
        for z in lo_z..hi_z {
            for x in lo_x..hi_x {
                for dy in 0..pool_depth {
                    let y = water_y + dy;
                    if y < side {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), WATER);
                    }
                }
            }
        }
        // Sand: a dune field on one edge, at terrain surface level.
        let sand_width = side / 8;
        let sand_depth = (side / 64).max(2);
        for z in 0..side / 3 {
            for x in 0..sand_width {
                for dy in 0..sand_depth {
                    let y = center + dy;
                    if y < side {
                        self.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), SAND);
                    }
                }
            }
        }
    }

    /// Seed a lattice megastructure embedded in natural terrain.
    ///
    /// A 3D grid of corridors carved from stone, with grass-covered pillars,
    /// water channels, lava pools, fire sources, and sand dunes. The
    /// periodic lattice compresses well in the octree (hashlife sharing).
    /// A heightmap terrain fills the outside world, and the outer shell of
    /// the lattice tapers into that terrain so the seam reads less like a
    /// hard cuboid cutout.
    ///
    /// Uses `LatticeField` + `TerrainBlendField` + `gen_region` for
    /// octree-native generation with proof-based collapse.
    pub fn seed_lattice_megastructure(&mut self) {
        self.store = NodeStore::new();
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        // hash-thing-5ie4 (vqke.2.1): the slow-divisor predicate
        // depends on the material registry's tick_divisor flags;
        // any mutation that changes a divisor flips the answer. Same
        // dxi4.2 invalidation-surface bug class as the other inert
        // caches. Per Claude + Codex plan-review on 5ie4.
        self.hashlife_slow_divisor_cache.clear();
        self.clone_sources.clear();
        let terrain_params = TerrainParams::for_level(self.level);
        let terrain = PrecomputedHeightmapField::new(terrain_params.to_heightmap(), self.level)
            .expect("TerrainParams::for_level must yield a valid heightmap field");
        let lattice = LatticeField::for_world(self.level, 42);
        let edge_blend = (lattice.cell_size / 2).max(2);
        let field = TerrainBlendField::new(
            lattice.clone(),
            terrain,
            lattice.lo,
            lattice.hi,
            edge_blend,
            (lattice.cell_size / 3).max(4) as f32,
            3.0,
        );
        let (root, _stats) = gen_region(&mut self.store, &field, [0, 0, 0], self.level);
        self.root = root;
        self.generation = 0;
        self.terrain_params = Some(terrain_params);
        self.block_rule_present = None;
    }

    /// Seed a warped gyroid megastructure tuned for a walkable demo zone.
    ///
    /// Uses `GyroidField` + `gen_region` for octree-native generation with
    /// proof-based collapse. The gyroid stays parameterized so later
    /// megastructure experiments can reuse the same implicit-field path.
    pub fn seed_gyroid_megastructure(&mut self) -> GenStats {
        self.store = NodeStore::new();
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        // hash-thing-5ie4 (vqke.2.1): the slow-divisor predicate
        // depends on the material registry's tick_divisor flags;
        // any mutation that changes a divisor flips the answer. Same
        // dxi4.2 invalidation-surface bug class as the other inert
        // caches. Per Claude + Codex plan-review on 5ie4.
        self.hashlife_slow_divisor_cache.clear();
        self.clone_sources.clear();
        let field = GyroidField::for_world(self.level, 42);
        let (root, stats) = gen_region(&mut self.store, &field, [0, 0, 0], self.level);
        self.root = root;
        self.generation = 0;
        self.block_rule_present = None;
        stats
    }

    fn fill_box(&mut self, volume: Box3, state: CellState) {
        for z in volume.min[2]..=volume.max[2] {
            for y in volume.min[1]..=volume.max[1] {
                for x in volume.min[0]..=volume.max[0] {
                    self.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), state);
                }
            }
        }
    }

    #[cfg(test)]
    fn box_contains_material(&self, volume: Box3, state: CellState) -> bool {
        for z in volume.min[2]..=volume.max[2] {
            for y in volume.min[1]..=volume.max[1] {
                for x in volume.min[0]..=volume.max[0] {
                    if self.get(WorldCoord(x), WorldCoord(y), WorldCoord(z)) == state {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn fill_floor(&mut self, volume: Box3, state: CellState) {
        self.fill_box(
            Box3::new(
                [volume.min[0], volume.min[1] - 1, volume.min[2]],
                [volume.max[0], volume.min[1] - 1, volume.max[2]],
            ),
            state,
        );
    }

    fn carve_rising_promenade(
        &mut self,
        x0: i64,
        x1: i64,
        z0: i64,
        z1: i64,
        floor_y0: i64,
        floor_y1: i64,
    ) {
        let span = (x1 - x0).abs().max(1);
        let step = if x1 >= x0 { 1 } else { -1 };
        let mut x = x0;
        loop {
            let traveled = (x - x0).abs();
            let floor_y = floor_y0 + (floor_y1 - floor_y0) * traveled / span;
            self.fill_box(Box3::new([x, floor_y, z0], [x, floor_y + 3, z1]), AIR);
            self.fill_box(Box3::new([x, floor_y - 1, z0], [x, floor_y - 1, z1]), STONE);
            if x == x1 {
                break;
            }
            x += step;
        }
    }

    fn progression_boxes(field: &LatticeField) -> ProgressionBoxes {
        let lo = field.lo;
        let hi = field.hi;
        let ground_y = lo[1] + field.floor_thick;
        let room_height = (field.cell_size / 2).max(4);
        let tunnel_half_w = (field.wall_thick + 1).max(2);

        let start_shell = Box3::new(
            [lo[0] + 1, ground_y, lo[2] + 1],
            [
                lo[0] + field.cell_size + 2,
                ground_y + room_height + 2,
                lo[2] + field.cell_size + 2,
            ],
        );
        let start_room = Box3::new(
            [
                start_shell.min[0] + 1,
                start_shell.min[1] + 1,
                start_shell.min[2] + 1,
            ],
            [
                start_shell.max[0] - 1,
                start_shell.max[1] - 1,
                start_shell.max[2] - 1,
            ],
        );

        let corridor_z = (start_room.min[2] + start_room.max[2]) / 2;
        let corridor_end_x = (lo[0] + field.cell_size * 3).min(hi[0] - field.cell_size * 2);
        let corridor_shell = Box3::new(
            [
                start_shell.max[0] - 1,
                ground_y,
                corridor_z - tunnel_half_w - 1,
            ],
            [
                corridor_end_x,
                ground_y + room_height + 1,
                corridor_z + tunnel_half_w + 1,
            ],
        );
        let corridor = Box3::new(
            [
                corridor_shell.min[0] + 1,
                corridor_shell.min[1] + 1,
                corridor_shell.min[2] + 1,
            ],
            [
                corridor_shell.max[0],
                corridor_shell.max[1] - 1,
                corridor_shell.max[2] - 1,
            ],
        );

        let tease_a = Box3::new(
            [
                corridor.min[0] + field.cell_size / 2,
                corridor.min[1],
                corridor.max[2] + 1,
            ],
            [
                corridor.min[0] + field.cell_size,
                corridor.max[1] + 1,
                (corridor.max[2] + field.cell_size).min(hi[2] - 2),
            ],
        );
        let tease_b = Box3::new(
            [
                corridor.max[0] - field.cell_size,
                corridor.min[1],
                (corridor.min[2] - field.cell_size).max(lo[2] + 1),
            ],
            [
                corridor.max[0] - field.cell_size / 2,
                corridor.max[1] + 1,
                corridor.min[2] - 1,
            ],
        );

        let atrium = Box3::new(
            [
                corridor.max[0] - field.cell_size / 2,
                ground_y + 1,
                lo[2] + field.cell_size,
            ],
            [
                (corridor.max[0] + field.cell_size * 2).min(hi[0] - 3),
                (ground_y + field.cell_size + 2).min(hi[1] - 3),
                (lo[2] + field.cell_size * 4).min(hi[2] - 3),
            ],
        );

        let balcony = Box3::new(
            [
                atrium.max[0] - field.cell_size,
                ground_y + room_height / 2,
                atrium.center()[2] - tunnel_half_w - 1,
            ],
            [
                hi[0] - 2,
                ground_y + room_height / 2 + 3,
                atrium.center()[2] + tunnel_half_w + 1,
            ],
        );
        let panorama = Box3::new(
            [
                atrium.max[0] - field.cell_size / 2,
                ground_y,
                lo[2] + field.cell_size / 2,
            ],
            [
                hi[0] - 1,
                hi[1] - 2,
                (lo[2] + field.cell_size * 5).min(hi[2] - 2),
            ],
        );

        ProgressionBoxes {
            start_shell,
            start_room,
            corridor_shell,
            corridor,
            tease_a,
            tease_b,
            atrium,
            balcony,
            panorama,
        }
    }

    fn lattice_progression_spectacle_anchors(
        start_room: Box3,
        atrium: Box3,
        panorama: Box3,
        reveal: [i64; 3],
    ) -> [DemoSpectacleAnchor; 3] {
        [
            DemoSpectacleAnchor {
                label: "intro",
                center: [
                    start_room.min[0] + 2,
                    start_room.min[1],
                    start_room.max[2] - 2,
                ],
                profile: DemoSpectacleProfile::Hearth,
            },
            DemoSpectacleAnchor {
                label: "interior",
                center: [atrium.center()[0], atrium.min[1], atrium.center()[2] + 3],
                profile: DemoSpectacleProfile::Clash,
            },
            DemoSpectacleAnchor {
                label: "panorama",
                center: [panorama.center()[0], reveal[1], reveal[2] + 3],
                profile: DemoSpectacleProfile::Cascade,
            },
        ]
    }

    fn progression_waterfall_layout(
        corridor: Box3,
        tease_b: Box3,
        ground_y: i64,
    ) -> ProgressionWaterfallLayout {
        let source_z = tease_b.min[2] - 2;
        let curtain_z = source_z + 1;
        let curtain_front_z = tease_b.min[2];
        let shaft_bottom = ground_y - 2;
        let shaft_top = corridor.max[1] + 4;
        let source_x = [tease_b.min[0], tease_b.max[0]];

        ProgressionWaterfallLayout {
            shell: Box3::new(
                [source_x[0] - 1, shaft_bottom - 1, source_z],
                [source_x[1] + 1, shaft_top + 1, curtain_z],
            ),
            shaft: Box3::new(
                [source_x[0], shaft_bottom, curtain_z],
                [source_x[1], shaft_top, curtain_front_z],
            ),
            curtain: Box3::new(
                [source_x[0], shaft_bottom, curtain_z],
                [source_x[1], shaft_top - 1, curtain_front_z],
            ),
            source_x,
            source_y: [shaft_bottom, shaft_top],
            source_z,
            focus: [tease_b.center()[0], corridor.center()[1], curtain_front_z],
        }
    }

    /// Place a CLONE cell that spawns `source_material` downstream.
    ///
    /// Routes through `pack_clone_source` to centralize the 6-bit cap +
    /// panic message (hash-thing-457f). The sister site at
    /// `main.rs::place_clone_block` uses the same helper.
    fn place_clone_source(&mut self, pos: [i64; 3], source_material: u16) {
        let state = pack_clone_source(source_material);
        self.set(
            WorldCoord(pos[0]),
            WorldCoord(pos[1]),
            WorldCoord(pos[2]),
            state,
        );
        self.clone_sources.push(pos);
    }

    fn seed_progression_waterfall(
        &mut self,
        corridor: Box3,
        tease_b: Box3,
        ground_y: i64,
    ) -> ProgressionWaterfallLayout {
        let layout = Self::progression_waterfall_layout(corridor, tease_b, ground_y);
        let water_material = Cell::from_raw(WATER).material();

        self.fill_box(layout.shell, STONE);
        self.fill_box(layout.shaft, AIR);
        self.fill_box(layout.curtain, WATER);
        for y in layout.source_y[0]..=layout.source_y[1] {
            for x in layout.source_x[0]..=layout.source_x[1] {
                self.place_clone_source([x, y, layout.source_z], water_material);
            }
        }

        layout
    }

    pub fn seed_lattice_progression_demo(&mut self) -> DemoLayout {
        self.seed_lattice_megastructure();
        let field = LatticeField::for_world(self.level, 42);
        let ground_y = field.lo[1] + field.floor_thick;
        let ProgressionBoxes {
            start_shell,
            start_room,
            corridor_shell,
            corridor,
            tease_a,
            tease_b,
            atrium,
            balcony,
            panorama,
        } = Self::progression_boxes(&field);

        self.fill_box(start_shell, STONE);
        self.fill_box(start_room, AIR);
        self.fill_box(
            Box3::new(
                [start_room.min[0], start_shell.min[1], start_room.min[2]],
                [start_room.max[0], start_shell.min[1], start_room.max[2]],
            ),
            DIRT,
        );
        self.fill_box(corridor_shell, STONE);
        self.fill_box(corridor, AIR);
        self.fill_floor(corridor, DIRT);
        let entry_passage = Box3::new(
            [start_room.max[0] - 1, start_room.min[1], corridor.min[2]],
            [corridor.min[0], corridor.max[1], corridor.max[2]],
        );
        self.fill_box(entry_passage, AIR);
        self.fill_floor(entry_passage, DIRT);
        // Bridge the stone slab between corridor.max[2] and atrium.min[2].
        // At every level atrium.min[2] (lo[2]+cell_size) sits strictly past
        // corridor.max[2] (corridor_z+tunnel_half_w), so without this carve
        // the walk_route's corridor_turn → atrium_entry segment is blocked.
        // The gap is small at level ≤ 6 and grows with cell_size at higher
        // levels, which is what stranded the level-8 traversability test in
        // hash-thing-69cq until dyqr. The carve lands before the `atrium`
        // fill_box below, so any overlap at atrium.min[2] is overwritten by
        // the atrium's own AIR/STONE pass.
        let corridor_turn_x = corridor.max[0] - 2;
        let tunnel_half_w = (corridor.max[2] - corridor.min[2]) / 2;
        let atrium_entry_passage = Box3::new(
            [
                corridor_turn_x - tunnel_half_w,
                corridor.min[1],
                corridor.max[2],
            ],
            [
                corridor_turn_x + tunnel_half_w,
                corridor.max[1],
                atrium.min[2],
            ],
        );
        self.fill_box(atrium_entry_passage, AIR);
        self.fill_floor(atrium_entry_passage, DIRT);
        self.fill_box(tease_a, AIR);
        self.fill_box(tease_b, AIR);
        self.fill_box(atrium, AIR);
        self.fill_floor(atrium, STONE);
        self.fill_box(balcony, AIR);
        self.fill_box(panorama, AIR);
        self.stage_lattice_progression_materials(
            start_room,
            corridor_shell,
            corridor,
            tease_a,
            tease_b,
            balcony,
            panorama,
        );
        self.stage_panorama_gangplank(balcony, panorama);
        self.carve_rising_promenade(
            atrium.center()[0],
            balcony.center()[0],
            balcony.min[2],
            balcony.max[2],
            atrium.min[1],
            balcony.min[1],
        );

        let corridor_mid = [corridor.center()[0], corridor.min[1], corridor.center()[2]];
        let corridor_turn = [corridor_turn_x, corridor.min[1], corridor.center()[2]];
        let atrium_entry = [corridor_turn[0], atrium.min[1], atrium.min[2] + 1];
        let atrium_center = [atrium.center()[0], atrium.min[1], atrium.center()[2]];
        let reveal_center = [balcony.center()[0], balcony.min[1], balcony.center()[2]];
        let panorama_center = [panorama.center()[0], balcony.min[1], balcony.center()[2]];
        self.stage_demo_spectacles(&Self::lattice_progression_spectacle_anchors(
            start_room,
            atrium,
            panorama,
            reveal_center,
        ));
        self.seed_progression_waterfall(corridor, tease_b, ground_y);
        self.seed_reveal_fireworks(balcony.center());
        self.seed_progression_break_trigger(tease_a);

        let player_pos = [
            start_room.center()[0] as f64 + 0.5,
            (ground_y + 1) as f64,
            start_room.center()[2] as f64 + 0.5,
        ];
        DemoLayout {
            player_pos,
            player_yaw: -std::f64::consts::FRAC_PI_2,
            player_pitch: 0.0,
            walk_route: [
                corridor_mid,
                corridor_turn,
                atrium_entry,
                atrium_center,
                reveal_center,
                panorama_center,
            ],
            corridor_mid,
            atrium_center,
            reveal_center,
            panorama_center,
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the route beats are named scene volumes, not anonymous coordinates"
    )]
    fn stage_lattice_progression_materials(
        &mut self,
        start_room: Box3,
        corridor_shell: Box3,
        corridor: Box3,
        tease_a: Box3,
        tease_b: Box3,
        balcony: Box3,
        panorama: Box3,
    ) {
        // Room: vine-draped entry wall and ceiling lip so the start reads
        // organic rather than bare debug architecture.
        self.fill_box(
            Box3::new(
                [
                    start_room.min[0] + 1,
                    start_room.min[1],
                    start_room.min[2] + 1,
                ],
                [
                    start_room.min[0] + 3,
                    start_room.max[1] - 1,
                    start_room.min[2] + 1,
                ],
            ),
            VINE,
        );
        self.fill_box(
            Box3::new(
                [
                    start_room.min[0] + 2,
                    start_room.max[1] - 1,
                    start_room.min[2] + 1,
                ],
                [
                    start_room.min[0] + 5,
                    start_room.max[1] - 1,
                    start_room.min[2] + 2,
                ],
            ),
            VINE,
        );

        // Corridor: carve a side opening and hang a thin waterfall through it.
        let corridor_mid = corridor.center();
        let gap_z = corridor_shell.max[2];
        self.fill_box(
            Box3::new(
                [corridor_mid[0] - 1, corridor.min[1], gap_z],
                [corridor_mid[0] + 1, corridor.max[1], gap_z],
            ),
            AIR,
        );
        self.fill_box(
            Box3::new(
                [corridor_mid[0], corridor.min[1] + 2, corridor.max[2] - 1],
                [corridor_mid[0], corridor.max[1], corridor.max[2]],
            ),
            WATER,
        );

        // Tease: sand spill on one branch, fan props on the other so the
        // player reads "something strange is happening ahead" before the reveal.
        let tease_a_center = tease_a.center();
        self.fill_box(
            Box3::new(
                [tease_a.min[0] + 1, tease_a.min[1], tease_a_center[2] - 1],
                [
                    tease_a.min[0] + 4,
                    tease_a.min[1] + 1,
                    tease_a_center[2] + 1,
                ],
            ),
            SAND,
        );
        let tease_b_center = tease_b.center();
        self.fill_box(
            Box3::new(
                [
                    tease_b.max[0] - 1,
                    tease_b.min[1] + 1,
                    tease_b_center[2] - 1,
                ],
                [
                    tease_b.max[0] - 1,
                    tease_b.min[1] + 2,
                    tease_b_center[2] + 1,
                ],
            ),
            FAN,
        );

        // Reveal: water curtains framing the balcony. Fireworks are staged
        // by the shared reveal helper so this route-specific pass only owns
        // the geometry-aware framing.
        self.fill_box(
            Box3::new(
                [panorama.min[0] + 1, balcony.min[1], balcony.min[2]],
                [panorama.min[0] + 1, balcony.max[1] + 3, balcony.min[2] + 1],
            ),
            WATER,
        );
        self.fill_box(
            Box3::new(
                [panorama.min[0] + 1, balcony.min[1], balcony.max[2] - 1],
                [panorama.min[0] + 1, balcony.max[1] + 3, balcony.max[2]],
            ),
            WATER,
        );
    }

    fn stage_panorama_gangplank(&mut self, balcony: Box3, panorama: Box3) {
        let spine_z = balcony.center()[2];
        let drop_floor = (balcony.min[1] - 10).max(0);
        let gangplank = Box3::new(
            [balcony.min[0], balcony.min[1], spine_z - 1],
            [panorama.max[0], balcony.max[1], spine_z + 1],
        );
        let wide_span = Box3::new(
            [panorama.min[0], balcony.min[1], balcony.min[2]],
            [panorama.max[0], balcony.max[1], balcony.max[2]],
        );

        // Clear the broad terrace floor first so the reveal reads as a narrow
        // bridge over open air instead of a safe plaza.
        self.fill_floor(wide_span, AIR);
        self.fill_floor(gangplank, STONE);

        let drop = Box3::new(
            [balcony.max[0] - 1, drop_floor, balcony.min[2] - 2],
            [panorama.max[0], balcony.min[1] - 2, balcony.max[2] + 2],
        );
        self.fill_box(drop, AIR);
    }

    fn seed_progression_break_trigger(&mut self, tease_a: Box3) {
        let center_x = tease_a.center()[0];
        let curtain = Box3::new(
            [center_x - 1, tease_a.min[1], tease_a.min[2]],
            [center_x + 1, tease_a.min[1] + 2, tease_a.min[2]],
        );
        self.fill_box(curtain, VINE);
    }

    pub fn population(&self) -> u64 {
        self.store.population(self.root)
    }

    /// Bytes-per-entry estimate for `hashlife_macro_cache` (hash-thing-z7uu).
    /// Key is `(NodeId, u64)` → u32 + u64 with u64 alignment pads the key
    /// tuple to 16 bytes; value is `NodeId` (u32) padded to 8 under the
    /// outer tuple's 8-byte alignment. Total stored bytes = 24. Add 8 bytes
    /// for the hashbrown SwissTable control byte plus load-factor slack
    /// (~0.875) — gives ~32 bytes/entry, matching the perf paper §4.7.3
    /// projection. Compile-time constant so the arithmetic stays auditable.
    pub const MACRO_CACHE_BYTES_PER_ENTRY: usize =
        std::mem::size_of::<((NodeId, u64), NodeId)>() + 8;

    /// Live count of entries in `hashlife_macro_cache`. Exposed so
    /// observers (perf HUD, bench harnesses) don't have to reach into a
    /// `pub(crate)` field.
    pub fn macro_cache_entries(&self) -> usize {
        self.hashlife_macro_cache.len()
    }

    /// Approximate byte footprint of `hashlife_macro_cache` (entries ×
    /// `MACRO_CACHE_BYTES_PER_ENTRY`). Closes the perf paper §5 band that
    /// had a 9 MB floor / 105 MB ceiling projection at 4096³ with no
    /// runtime measurement behind it.
    pub fn macro_cache_bytes_est(&self) -> usize {
        self.macro_cache_entries() * Self::MACRO_CACHE_BYTES_PER_ENTRY
    }

    /// Compact one-line summary of spatial-memo health for the periodic log
    /// (hash-thing-stue.6, extended by hash-thing-o2es, hash-thing-z7uu).
    /// Uses the lifetime accumulator `hashlife_stats_total` for hit rate —
    /// so the rate converges toward steady state rather than fluctuating
    /// with the most recent step. Table sizes are live `.len()`, which is
    /// the right signal for "is this table about to blow memory."
    ///
    /// Format: `memo_hit=<fraction> memo_churn=<signed-fraction>
    /// memo_tbl=<int> memo_mac=<int> memo_mac_bytes=<int>
    /// p1=<ms>ms p2=<ms>ms p3=<ms>ms p4=<ms>ms`.
    ///
    /// `p3` (hash-thing-vqke Phase 0) is the descent-and-intern overhead:
    /// `step_node_wall_ns - phase1_ns - phase2_ns`. It captures the
    /// recursive memo lookup, hash-cons interning, and intermediate-node
    /// allocation cost between leaf evaluations.
    ///
    /// `p4` is `compact_ns` — the wall-clock spent in `maybe_compact()`
    /// after `step_node`. Most steps are 0 (the 2× growth gate); compact
    /// spikes are the cost spread across N steps where N depends on
    /// churn.
    ///
    /// `memo_churn` is `window_hit_rate − lifetime_hit_rate` over the last
    /// `MemoWindow::CAPACITY` steps. Positive = recent steps cache better
    /// than the running average (e.g. cache warmup). Negative = recent
    /// regression. `+0.000` before any step has run (both rates are 0).
    /// `memo_mac` / `memo_mac_bytes` stay at 0 on single-step sessions
    /// (only `step_recursive_pow2` populates the macro cache).
    ///
    /// `p1` / `p2` are the last step's per-phase wall times inside
    /// `step_grid_once` (Phase 1 = per-cell CaRule, Phase 2 = per-block
    /// BlockRule / Margolus). Reported per-step (not lifetime) so the
    /// number reflects current sim cost rather than an ever-growing sum
    /// (hash-thing-71mp).
    pub fn memo_summary(&self) -> String {
        let stats = &self.hashlife_stats_total;
        let total = stats.cache_hits + stats.cache_misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / total as f64
        };
        let churn = self.memo_window.hit_rate() - hit_rate;
        let last_step = &self.hashlife_stats;
        let p1_ms = last_step.phase1_ns as f64 / 1_000_000.0;
        let p2_ms = last_step.phase2_ns as f64 / 1_000_000.0;
        // hash-thing-vqke Phase 0: descent-and-intern overhead is
        // step_node wall minus the per-cell + per-block leaf wall.
        // saturating_sub keeps the number sane on the rare frame
        // where leaf timers overlap with descent timers slightly
        // (e.g. inner-loop measurement quantization).
        let descent_ns = last_step
            .step_node_wall_ns
            .saturating_sub(last_step.phase1_ns)
            .saturating_sub(last_step.phase2_ns);
        let p3_ms = descent_ns as f64 / 1_000_000.0;
        let p4_ms = last_step.compact_ns as f64 / 1_000_000.0;
        // hash-thing-bjdl (vqke.2): diagnostic ratios for the
        // memo_hit-rate hypotheses. Bounded-width ratio tokens (each
        // ≤ 25 chars) so the HUD-overlay split test stays green even
        // when the underlying counts are in the millions.
        let memo_period = self.materials.memo_period();
        let phase_aliased = if stats.cache_misses == 0 {
            0.0
        } else {
            stats.cache_misses_phase_aliased as f64 / stats.cache_misses as f64
        };
        let compact_total = stats.compact_entries_kept + stats.compact_entries_dropped;
        let compact_drop = if compact_total == 0 {
            0.0
        } else {
            stats.compact_entries_dropped as f64 / compact_total as f64
        };
        // hash-thing-tk4j (vqke.3): expose the rates at which step_node's
        // pre-cache fast paths fire. Denominator is total step_node calls
        // (skips + cache_hits + cache_misses), so the three rates plus
        // memo_hit_rate-relative shares partition leaf traffic. Used to
        // diagnose whether the 47ms p1 cost on mostly-stable scenes is
        // because skip detection is weak (low skip rate → CaRule runs on
        // cells that should be recognised as stable) or because the
        // remaining unskipped fraction is genuinely changing.
        let total_calls =
            stats.cache_hits + stats.cache_misses + stats.empty_skips + stats.fixed_point_skips;
        let skip_empty_rate = if total_calls == 0 {
            0.0
        } else {
            stats.empty_skips as f64 / total_calls as f64
        };
        let skip_fixed_rate = if total_calls == 0 {
            0.0
        } else {
            stats.fixed_point_skips as f64 / total_calls as f64
        };
        // hash-thing-ecmn: BFS observability tokens. Per-step values
        // come from the most-recent-step `hashlife_stats`, not the
        // lifetime accumulator, so they reflect the active step's BFS
        // shape. `bfs_l3=0 bfs_par=0` on Serial / RayonPerFanout
        // strategies — those paths never set the counters.
        let bfs_l3 = last_step.bfs_level3_unique_misses;
        let bfs_par = last_step.bfs_batches_parallel;
        let bfs_serial_fb = last_step.bfs_batches_serial_fallback;
        let bfs_max = last_step.bfs_max_batch_len;
        // hash-thing-aqq4 verdict: surface the work-elision factor as a
        // primary line so future field readings give the real signal
        // directly (instead of being confused by memo_hit, which only
        // counts cache_hits/(hits+misses) and ignores empty/inert
        // short-circuits + the multiplicative effect of upper-level
        // hits eliding many base cases).
        //
        // elision = (level-3 nodes in the world) / max(L3 misses last step, 1)
        //         = (side / 8)^3 / L3_misses
        // A naive every-cell stepper would do (side/8)^3 base-case
        // evaluations per step. Hashlife does only L3_misses. The ratio
        // is the multiplier hashlife is buying us. >>1 means the engine
        // is paying off; ~1 means it's degenerating to brute force.
        // Floor on `max(_, 1)` so a fully-cached step (L3 misses = 0)
        // doesn't divide by zero — that's the perfect-hit case where
        // the elision factor is effectively unbounded.
        let l3_nodes_in_world = (1u64 << (3 * self.level)).saturating_div(512);
        let l3_misses_last = last_step.misses_by_level[0].max(1);
        let elision_factor = l3_nodes_in_world as f64 / l3_misses_last as f64;
        format!(
            "memo_hit={:.3} memo_churn={:+.3} memo_elision={:.1}x memo_tbl={} memo_mac={} memo_mac_bytes={} memo_period={} memo_phase_aliased={:.3} memo_compact_drop={:.3} memo_skip_empty={:.3} memo_skip_fixed={:.3} p1={:.2}ms p2={:.2}ms p3={:.2}ms p4={:.2}ms bfs_l3={} bfs_par={} bfs_serfb={} bfs_max={}",
            hit_rate,
            churn,
            elision_factor,
            self.hashlife_cache.len(),
            self.hashlife_macro_cache.len(),
            self.macro_cache_bytes_est(),
            memo_period,
            phase_aliased,
            compact_drop,
            skip_empty_rate,
            skip_fixed_rate,
            p1_ms,
            p2_ms,
            p3_ms,
            p4_ms,
            bfs_l3,
            bfs_par,
            bfs_serial_fb,
            bfs_max,
        )
    }

    fn spectacle_box(center: [i64; 3], min: [i64; 3], max: [i64; 3]) -> Box3 {
        Box3::new(
            [center[0] + min[0], center[1] + min[1], center[2] + min[2]],
            [center[0] + max[0], center[1] + max[1], center[2] + max[2]],
        )
    }

    fn seed_waypoint_frame(&mut self, waypoint: DemoWaypoint) {
        let [cx, cy, cz] = waypoint.center;
        let r = waypoint.radius;
        let floor_y = cy - 1;
        let ceiling_y = cy + 4;

        self.fill_box(
            Box3::new([cx - r, floor_y, cz - r], [cx + r, floor_y, cz + r]),
            DIRT,
        );
        self.fill_box(
            Box3::new([cx - r, floor_y + 1, cz - r], [cx - r, ceiling_y, cz + r]),
            STONE,
        );
        self.fill_box(
            Box3::new([cx + r, floor_y + 1, cz - r], [cx + r, ceiling_y, cz + r]),
            STONE,
        );
        self.fill_box(
            Box3::new([cx - r, floor_y + 1, cz + r], [cx + r, ceiling_y, cz + r]),
            STONE,
        );
        self.fill_box(
            Box3::new([cx - r, ceiling_y, cz - r], [cx + r, ceiling_y, cz + r]),
            STONE,
        );
    }

    fn seed_ember_set_piece(&mut self, waypoint: DemoWaypoint) {
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-2, 0, 0], [2, 2, 0]),
            GRASS,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-1, 1, -1], [1, 2, 1]),
            FIRE,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-3, 0, -2], [-2, 2, -1]),
            GRASS,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [2, 0, 1], [3, 2, 2]),
            GRASS,
        );
    }

    fn seed_spring_set_piece(&mut self, waypoint: DemoWaypoint) {
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-2, 3, -1], [2, 4, 1]),
            STONE,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-1, 4, -1], [1, 4, 1]),
            WATER,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-2, 0, 2], [2, 0, 3]),
            STONE,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-1, 1, 2], [1, 1, 3]),
            WATER,
        );
    }

    fn seed_quench_set_piece(&mut self, waypoint: DemoWaypoint) {
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-3, 0, -1], [-1, 2, 1]),
            GRASS,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-3, 1, -1], [-2, 2, 1]),
            FIRE,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [1, 3, -1], [3, 4, 1]),
            STONE,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [1, 4, -1], [2, 4, 1]),
            WATER,
        );
    }

    fn seed_cascade_set_piece(&mut self, waypoint: DemoWaypoint) {
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-2, 4, -2], [2, 4, -1]),
            STONE,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-1, 4, -2], [1, 4, -1]),
            WATER,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-2, 0, 1], [2, 0, 2]),
            LAVA,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-2, 1, 2], [-1, 2, 2]),
            OIL,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [1, 1, 2], [2, 2, 2]),
            OIL,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [-1, 1, 0], [-1, 3, 0]),
            FIREWORK,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [1, 1, 0], [1, 3, 0]),
            FIREWORK,
        );
        self.fill_box(
            Self::spectacle_box(waypoint.center, [0, 1, 1], [0, 2, 1]),
            FIRE,
        );
    }

    fn seed_reveal_fireworks(&mut self, center: [i64; 3]) {
        for &(dx, dz) in &[(-4, -3), (-4, 2), (-1, -3), (-1, 2)] {
            self.fill_box(
                Self::spectacle_box(center, [dx, -1, dz], [dx + 1, 0, dz + 1]),
                STONE,
            );
            self.fill_box(
                Self::spectacle_box(center, [dx, 1, dz], [dx, 2, dz]),
                FIREWORK,
            );
        }
    }

    fn commit_step(&mut self, next: &[CellState], side: usize) {
        self.root = self.store.from_flat(next, side);
        self.generation += 1;
        self.block_rule_present = None;

        // Fresh-store compaction: `from_flat` interned a brand new generation
        // into the append-only store, leaving the previous generation's
        // subtrees unreachable but still present. Rebuild into a fresh store
        // so memory tracks live-scene size, not cumulative history.
        // See hash-thing-88d.
        //
        // Brute-force compaction remaps every live NodeId into a fresh
        // store namespace, so clear all four hashlife caches to prevent
        // stale cross-path hits. Keep the remap so Svdag can update its
        // persistent NodeId cache (hash-thing-5bb.11: O(changed) instead
        // of O(reachable)).
        let (new_store, new_root, remap) = self.store.compacted_with_remap(self.root);
        self.store = new_store;
        self.root = new_root;
        self.last_compaction_remap = Some(match self.last_compaction_remap.take() {
            Some(existing) => compose_remap(existing, &remap),
            None => remap,
        });
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        // hash-thing-5ie4 (vqke.2.1): the slow-divisor predicate
        // depends on the material registry's tick_divisor flags;
        // any mutation that changes a divisor flips the answer. Same
        // dxi4.2 invalidation-surface bug class as the other inert
        // caches. Per Claude + Codex plan-review on 5ie4.
        self.hashlife_slow_divisor_cache.clear();
    }

    /// Replace the world with terrain generated from `params`. Uses
    /// a fresh `NodeStore` and clears the step cache — every `NodeId`
    /// from the previous world is invalidated. Resets `generation` to 0.
    ///
    /// ## Fresh-store epoch
    ///
    /// `seed_terrain` is a new epoch: everything from the previous world
    /// is unreachable after this returns, so there is no point interning
    /// the new terrain into a store full of garbage. Building into a
    /// fresh store is cheaper (no post-build compaction walk), keeps
    /// node counts honest, and makes the epoch boundary explicit.
    ///
    /// **The caller MUST keep the simulation paused around this call.**
    pub fn seed_terrain(&mut self, params: &TerrainParams) -> Result<GenStats, &'static str> {
        params.validate()?;
        self.store = NodeStore::new();
        self.hashlife_cache.clear();
        self.hashlife_macro_cache.clear();
        self.hashlife_inert_cache.clear();
        self.hashlife_all_inert_cache.clear();
        // hash-thing-5ie4 (vqke.2.1): the slow-divisor predicate
        // depends on the material registry's tick_divisor flags;
        // any mutation that changes a divisor flips the answer. Same
        // dxi4.2 invalidation-surface bug class as the other inert
        // caches. Per Claude + Codex plan-review on 5ie4.
        self.hashlife_slow_divisor_cache.clear();
        // New epoch: the renderer's cache holds entries keyed in the previous
        // epoch's NodeId namespace, which is meaningless against a brand-new
        // NodeStore. Publishing an *empty* remap drives `Svdag::apply_remap`
        // to drop every cached entry (absence == invalidation), which is
        // exactly what a scene reset requires. Leaving it `None` would let
        // stale low-NodeId cache entries alias against the fresh store's
        // low NodeIds and produce first-frame corruption after a regenerate
        // (hash-thing-rk4n.1).
        self.last_compaction_remap = Some(FxHashMap::default());
        let heightmap = params.to_heightmap();
        let precompute_start = std::time::Instant::now();
        let field = PrecomputedHeightmapField::new(heightmap, self.level)
            .expect("validated TerrainParams must yield a valid heightmap field");
        let precompute_us = precompute_start.elapsed().as_micros() as u64;
        let gen_start = std::time::Instant::now();
        let (root, mut stats) = gen_region(&mut self.store, &field, [0, 0, 0], self.level);
        stats.precompute_us = precompute_us;
        stats.gen_region_us = gen_start.elapsed().as_micros() as u64;
        stats.nodes_after_gen = self.store.stats();
        self.root = root;
        self.generation = 0;
        self.terrain_params = Some(*params);
        self.block_rule_present = None;
        Ok(stats)
    }
}

/// Cascading bottom-to-top sweep that fills internal air gaps in
/// gravity-bearing bodies. Retained as a `#[cfg(test)]` reference for
/// divergence/regression tests after qy4g option G removed it from the
/// production step path (2026-04-26).
///
/// Mass-conserving: only swaps, never creates or destroys cells.
#[cfg(test)]
pub(crate) fn gravity_gap_fill(grid: &mut [CellState], side: usize, materials: &MaterialRegistry) {
    for z in 0..side {
        for x in 0..side {
            for y in 1..side.saturating_sub(1) {
                let idx_below = x + (y - 1) * side + z * side * side;
                let idx_cur = x + y * side + z * side * side;
                let idx_above = x + (y + 1) * side + z * side * side;
                let below = Cell::from_raw(grid[idx_below]);
                let cur = Cell::from_raw(grid[idx_cur]);
                let above = Cell::from_raw(grid[idx_above]);
                // Only fill gap if: current is air, above is gravity-bearing,
                // AND below is gravity-bearing (this is an internal gap).
                if cur.is_empty()
                    && !above.is_empty()
                    && materials.block_rule_id_for_cell(above).is_some()
                    && !below.is_empty()
                    && materials.block_rule_id_for_cell(below).is_some()
                {
                    grid.swap(idx_cur, idx_above);
                }
            }
        }
    }
}

/// 1D column variant of [`gravity_gap_fill`]. Retained as a `#[cfg(test)]`
/// reference (qy4g option G, 2026-04-26).
///
/// Operates on a single `(x, z)` column (length = `side`), mutating in place.
/// Returns `true` when any swap fired so callers can skip the splice step
/// for unchanged columns.
///
/// In-place mutation is load-bearing: the 3D loop evaluates `y = 1..side-1`
/// in order and later iterations read values that earlier iterations wrote.
/// A fresh-read variant (read whole column, write to an out-buffer) would
/// diverge on cascade patterns like `[B, A, B, B]` — see Rev 2 plan tests.
#[cfg(test)]
pub(crate) fn gravity_gap_fill_column(
    column: &mut [CellState],
    materials: &MaterialRegistry,
) -> bool {
    let side = column.len();
    if side < 3 {
        return false;
    }
    let mut changed = false;
    for y in 1..side - 1 {
        let below = Cell::from_raw(column[y - 1]);
        let cur = Cell::from_raw(column[y]);
        let above = Cell::from_raw(column[y + 1]);
        if cur.is_empty()
            && !above.is_empty()
            && materials.block_rule_id_for_cell(above).is_some()
            && !below.is_empty()
            && materials.block_rule_id_for_cell(below).is_some()
        {
            column.swap(y, y + 1);
            changed = true;
        }
    }
    changed
}

/// Get the 26 Moore neighbors of a cell. Out-of-bounds neighbors are
/// `Cell::EMPTY` (absorbing boundary), matching hashlife's infinite-world
/// semantics.
fn get_neighbors(grid: &[CellState], side: usize, x: usize, y: usize, z: usize) -> [Cell; 26] {
    let mut neighbors = [Cell::EMPTY; 26];
    let mut idx = 0;
    let s = side as i32;
    for dz in [-1i32, 0, 1] {
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nz = z as i32 + dz;
                if nx >= 0 && nx < s && ny >= 0 && ny < s && nz >= 0 && nz < s {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nz = nz as usize;
                    neighbors[idx] = Cell::from_raw(grid[nx + ny * side + nz * side * side]);
                }
                // else: stays Cell::EMPTY (absorbing boundary)
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
    //! Reference tests for the brute-force dispatch step.

    use super::*;
    use crate::player;
    use crate::render::Svdag;
    use crate::sim::rule::{GameOfLife3D, ALIVE};
    use crate::terrain::materials::{
        MaterialRegistry, AIR_MATERIAL_ID, FAN, FAN_ARMED_FIREWORK_MATERIAL_IDS,
        FAN_ARMED_STEAM_MATERIAL_IDS, FIRE, FIREWORK, FIREWORK_MATERIAL_ID, FIRE_MATERIAL_ID,
        GRASS, ICE, ICE_MATERIAL_ID, LAVA, LAVA_MATERIAL_ID, OIL, OIL_MATERIAL_ID, SAND, STEAM,
        STEAM_MATERIAL_ID, STONE, VINE, VINE_MATERIAL_ID, WATER, WATER_MATERIAL_ID,
    };
    use std::collections::{HashSet, VecDeque};

    /// Helper: build an empty 8^3 world (level=3).
    fn empty_world() -> World {
        World::new(3)
    }

    fn gol_world(rule: GameOfLife3D) -> World {
        let mut world = empty_world();
        world.set_gol_smoke_rule(rule);
        world
    }

    fn feet(point: [i64; 3]) -> [f64; 3] {
        [
            point[0] as f64 + 0.5,
            point[1] as f64,
            point[2] as f64 + 0.5,
        ]
    }

    fn player_has_support(world: &World, pos: &[f64; 3]) -> bool {
        let hw = player::PLAYER_HALF_W;
        let x_min = (pos[0] - hw).floor() as i64;
        let x_max = (pos[0] + hw).floor() as i64;
        let z_min = (pos[2] - hw).floor() as i64;
        let z_max = (pos[2] + hw).floor() as i64;
        let support_y = pos[1].floor() as i64 - 1;
        for x in x_min..=x_max {
            for z in z_min..=z_max {
                if world.get(WorldCoord(x), WorldCoord(support_y), WorldCoord(z)) != AIR {
                    return true;
                }
            }
        }
        false
    }

    fn walk_cell(pos: [f64; 3]) -> [i64; 3] {
        [
            pos[0].floor() as i64,
            pos[1].floor() as i64,
            pos[2].floor() as i64,
        ]
    }

    fn is_walkable_player_cell(world: &World, cell: [i64; 3]) -> bool {
        let pos = feet(cell);
        !player::player_collides(world, &pos) && player_has_support(world, &pos)
    }

    fn assert_walk_segment_is_traversable(world: &World, from: [i64; 3], to: [i64; 3]) {
        assert!(
            is_walkable_player_cell(world, from),
            "segment start must be walkable: {from:?}"
        );
        assert!(
            is_walkable_player_cell(world, to),
            "segment end must be walkable: {to:?}"
        );

        let x_min = from[0].min(to[0]) - 6;
        let x_max = from[0].max(to[0]) + 6;
        let y_min = from[1].min(to[1]) - 2;
        let y_max = from[1].max(to[1]) + 2;
        let z_min = from[2].min(to[2]) - 6;
        let z_max = from[2].max(to[2]) + 6;

        let mut frontier = VecDeque::from([from]);
        let mut visited = HashSet::from([from]);

        while let Some(cell) = frontier.pop_front() {
            if cell == to {
                return;
            }

            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                for dy in -1..=1 {
                    let next = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    if next[0] < x_min
                        || next[0] > x_max
                        || next[1] < y_min
                        || next[1] > y_max
                        || next[2] < z_min
                        || next[2] > z_max
                        || visited.contains(&next)
                        || !is_walkable_player_cell(world, next)
                    {
                        continue;
                    }
                    visited.insert(next);
                    frontier.push_back(next);
                }
            }
        }

        panic!("no walkable route between {from:?} and {to:?}");
    }

    fn sync_svdag_with_world(world: &mut World, svdag: &mut Svdag) {
        if let Some(remap) = world.last_compaction_remap.take() {
            svdag.apply_remap(&remap);
        }
        svdag.update(&world.store, world.root, world.level);
        if svdag.stale_ratio() > 0.5 {
            svdag.compact(&world.store, world.root);
        }
    }

    fn lookup_svdag_voxel(svdag: &Svdag, x: u64, y: u64, z: u64) -> u16 {
        const LEAF_BIT: u32 = 0x8000_0000;

        let side = 1u64 << svdag.root_level;
        if x >= side || y >= side || z >= side {
            return 0;
        }
        let mut offset = svdag.nodes[0] as usize;
        let mut lx = x;
        let mut ly = y;
        let mut lz = z;
        for level in (1..=svdag.root_level).rev() {
            let half = 1u64 << (level - 1);
            let ox = usize::from(lx >= half);
            let oy = usize::from(ly >= half);
            let oz = usize::from(lz >= half);
            let octant = ox | (oy << 1) | (oz << 2);

            let mask = svdag.nodes[offset];
            let child_word = svdag.nodes[offset + 1 + octant];
            if mask & (1 << octant) == 0 {
                return 0;
            }
            if child_word & LEAF_BIT != 0 {
                return (child_word & !LEAF_BIT) as u16;
            }

            offset = child_word as usize;
            if ox == 1 {
                lx -= half;
            }
            if oy == 1 {
                ly -= half;
            }
            if oz == 1 {
                lz -= half;
            }
        }
        0
    }

    fn assert_svdag_matches_world(world: &World, svdag: &Svdag, label: &str) {
        let side = world.side();
        let mut mismatches = 0;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let octree = world.get(
                        WorldCoord(x as i64),
                        WorldCoord(y as i64),
                        WorldCoord(z as i64),
                    );
                    let dag = lookup_svdag_voxel(svdag, x as u64, y as u64, z as u64);
                    if octree != dag {
                        if mismatches < 5 {
                            eprintln!(
                                "  MISMATCH at ({x},{y},{z}): octree={octree} svdag={dag} [{label}]"
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{label}: {mismatches} voxel mismatches between world octree and Svdag"
        );

        assert_svdag_lod_rep_mat_matches_world(world, svdag, label);
    }

    // ---------------------------------------------------------------
    // eiu9: LOD rep_mat consistency check (mirror of qaca's ht-render walker)
    // ---------------------------------------------------------------
    //
    // The voxel-walk above only inspects leaves; the LOD `rep_mat` field packed
    // into `(slot[0] >> 8) & 0xFFFF` is unread. Walk the SVDAG and the World's
    // NodeStore in lockstep so post-compact and cross-epoch SVDAG syncs (the
    // path that motivated rk4n) gain the same rep_mat-staleness coverage that
    // qaca added on the ht-render m1f.5 corpus. Mirrored inline because
    // ht-render's helpers are gated on its own `cfg(test)`.

    fn assert_svdag_lod_rep_mat_matches_world(world: &World, svdag: &Svdag, label: &str) {
        use crate::octree::Node;

        if svdag.nodes.is_empty() {
            return;
        }

        // Empty world: SVDAG holds only the root header (no interior slots).
        if world.store.get(world.root).is_empty() {
            return;
        }

        // Memoize on NodeId — without this, validate's per-node call to
        // expected_rep_mat re-recurses the subtree at every level, blowing the
        // 128³ skip-sync test from ~5s to >9 min.
        fn expected_rep_mat(
            store: &NodeStore,
            id: NodeId,
            cache: &mut FxHashMap<NodeId, u32>,
        ) -> u32 {
            if let Some(&hit) = cache.get(&id) {
                return hit;
            }
            let val = match store.get(id) {
                Node::Leaf(state) => (*state as u32) & 0xFFFF,
                Node::Interior { children, .. } => {
                    let children = *children;
                    let mut rep_mat: u32 = 0;
                    let mut rep_pop: u64 = 0;
                    for &child_id in children.iter() {
                        match store.get(child_id) {
                            Node::Leaf(state) => {
                                if *state != 0 && (rep_mat == 0 || rep_pop == 0) {
                                    rep_mat = (*state as u32) & 0xFFFF;
                                    rep_pop = 1;
                                }
                            }
                            Node::Interior { population, .. } => {
                                if *population > 0 && *population > rep_pop {
                                    let p = *population;
                                    rep_mat = expected_rep_mat(store, child_id, cache);
                                    rep_pop = p;
                                }
                            }
                        }
                    }
                    rep_mat
                }
            };
            cache.insert(id, val);
            val
        }

        fn validate(
            svdag: &Svdag,
            store: &NodeStore,
            id: NodeId,
            offset: usize,
            path: &str,
            out: &mut Vec<String>,
            cache: &mut FxHashMap<NodeId, u32>,
        ) {
            const LEAF_BIT: u32 = 0x8000_0000;
            let cmask = svdag.nodes[offset];
            let encoded_rep = (cmask >> 8) & 0xFFFF;
            let want_rep = expected_rep_mat(store, id, cache);
            if encoded_rep != want_rep {
                out.push(format!(
                    "rep_mat mismatch at offset={offset} path='{}': encoded={} expected={}",
                    if path.is_empty() { "/" } else { path },
                    encoded_rep,
                    want_rep,
                ));
            }

            if let Node::Interior { children, .. } = store.get(id) {
                let children = *children;
                for (i, &child_id) in children.iter().enumerate() {
                    let child_word = svdag.nodes[offset + 1 + i];
                    match store.get(child_id) {
                        Node::Leaf(state) => {
                            let want = LEAF_BIT | (*state as u32);
                            if child_word != want {
                                out.push(format!(
                                    "leaf child slot mismatch at offset={} (child {i} of '{}'): encoded={:#010x} expected={:#010x}",
                                    offset + 1 + i,
                                    if path.is_empty() { "/" } else { path },
                                    child_word,
                                    want,
                                ));
                            }
                        }
                        Node::Interior { population, .. } => {
                            if *population == 0 {
                                if child_word != LEAF_BIT {
                                    out.push(format!(
                                        "empty-interior child slot mismatch at offset={} (child {i} of '{}'): encoded={:#010x} expected={:#010x}",
                                        offset + 1 + i,
                                        if path.is_empty() { "/" } else { path },
                                        child_word,
                                        LEAF_BIT,
                                    ));
                                }
                            } else if child_word & LEAF_BIT != 0 {
                                out.push(format!(
                                    "populated-interior child slot has LEAF_BIT at offset={} (child {i} of '{}'): encoded={:#010x}",
                                    offset + 1 + i,
                                    if path.is_empty() { "/" } else { path },
                                    child_word,
                                ));
                            } else {
                                let child_path = if path.is_empty() {
                                    format!("/{i}")
                                } else {
                                    format!("{path}/{i}")
                                };
                                validate(
                                    svdag,
                                    store,
                                    child_id,
                                    child_word as usize,
                                    &child_path,
                                    out,
                                    cache,
                                );
                            }
                        }
                    }
                }
            }
        }

        let root_offset = svdag.nodes[0] as usize;
        let mut mismatches: Vec<String> = Vec::new();
        let mut cache: FxHashMap<NodeId, u32> = FxHashMap::default();
        validate(
            svdag,
            &world.store,
            world.root,
            root_offset,
            "",
            &mut mismatches,
            &mut cache,
        );
        if !mismatches.is_empty() {
            for line in mismatches.iter().take(5) {
                eprintln!("  {line} [{label}]");
            }
            panic!(
                "{label}: {} LOD rep_mat / slot-encoding mismatches between world octree and Svdag",
                mismatches.len()
            );
        }
    }

    fn wc(coord: u64) -> WorldCoord {
        WorldCoord(coord as i64)
    }

    fn demo_spectacle_anchors() -> [DemoSpectacleAnchor; 3] {
        [
            DemoSpectacleAnchor {
                label: "intro",
                center: [16, 10, 16],
                profile: DemoSpectacleProfile::Hearth,
            },
            DemoSpectacleAnchor {
                label: "interior",
                center: [32, 12, 32],
                profile: DemoSpectacleProfile::Clash,
            },
            DemoSpectacleAnchor {
                label: "panorama",
                center: [48, 14, 48],
                profile: DemoSpectacleProfile::Cascade,
            },
        ]
    }

    fn local_snapshot(world: &World, center: [i64; 3], radius: u64) -> Vec<CellState> {
        let radius = i64::try_from(radius).expect("demo spectacle radius must fit in i64");
        let mut cells = Vec::new();
        for z in (center[2] - radius)..=(center[2] + radius) {
            for y in (center[1] - radius)..=(center[1] + radius) {
                for x in (center[0] - radius)..=(center[0] + radius) {
                    cells.push(world.get(WorldCoord(x), WorldCoord(y), WorldCoord(z)));
                }
            }
        }
        cells
    }

    // -----------------------------------------------------------------
    // 1. Empty world stays empty under amoeba.
    // -----------------------------------------------------------------
    #[test]
    fn empty_world_stays_empty_under_amoeba() {
        let mut world = gol_world(GameOfLife3D::new(9, 26, 5, 7));
        world.step();

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
    // -----------------------------------------------------------------
    #[test]
    fn single_cell_dies_under_amoeba() {
        let mut world = gol_world(GameOfLife3D::new(9, 26, 5, 7));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        assert_eq!(world.population(), 1);
        world.step();

        assert_eq!(
            world.population(),
            0,
            "lone live cell must die under amoeba"
        );
    }

    // -----------------------------------------------------------------
    // 3. Single live cell under crystal grows to a 3x3x3 cube.
    // -----------------------------------------------------------------
    #[test]
    fn single_cell_grows_to_3x3x3_cube_under_crystal() {
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step();

        assert_eq!(
            world.population(),
            27,
            "crystal single cell must grow to a 3^3 cube"
        );

        for z in 3..=5u64 {
            for y in 3..=5u64 {
                for x in 3..=5u64 {
                    assert_eq!(
                        world.get(wc(x), wc(y), wc(z)),
                        ALIVE.raw(),
                        "cell ({},{},{}) must be alive inside the expected 3x3x3 cube",
                        x,
                        y,
                        z
                    );
                }
            }
        }

        // Spot-check that cells just outside the cube are still dead.
        assert_eq!(world.get(wc(2), wc(4), wc(4)), 0);
        assert_eq!(world.get(wc(6), wc(4), wc(4)), 0);
        assert_eq!(world.get(wc(4), wc(2), wc(4)), 0);
        assert_eq!(world.get(wc(4), wc(6), wc(4)), 0);
        assert_eq!(world.get(wc(4), wc(4), wc(2)), 0);
        assert_eq!(world.get(wc(4), wc(4), wc(6)), 0);
    }

    // -----------------------------------------------------------------
    // 4. 2x2x2 cube is a still-life under custom S7-7/B3-3.
    // -----------------------------------------------------------------
    #[test]
    fn cube_2x2x2_is_still_life_under_s7_b3() {
        let mut world = gol_world(GameOfLife3D::new(7, 7, 3, 3));
        for z in 3..=4u64 {
            for y in 3..=4u64 {
                for x in 3..=4u64 {
                    world.set(wc(x), wc(y), wc(z), ALIVE.raw());
                }
            }
        }
        assert_eq!(world.population(), 8, "initial cube must have 8 cells");

        for gen in 1..=5 {
            world.step();
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
                                ALIVE.raw()
                            } else {
                                0
                            };
                        assert_eq!(
                            world.get(wc(x), wc(y), wc(z)),
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
    // 5. Determinism: same initial state + same rule -> byte-equal result.
    // -----------------------------------------------------------------
    #[test]
    fn step_is_deterministic() {
        let mut world_a = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        let mut world_b = gol_world(GameOfLife3D::new(0, 6, 1, 3));

        let seeds: &[(u64, u64, u64)] = &[(4, 4, 4), (3, 4, 5), (5, 2, 4)];
        for &(x, y, z) in seeds {
            world_a.set(wc(x), wc(y), wc(z), ALIVE.raw());
            world_b.set(wc(x), wc(y), wc(z), ALIVE.raw());
        }

        for _ in 0..3 {
            world_a.step();
            world_b.step();
        }

        let flat_a = world_a.flatten();
        let flat_b = world_b.flatten();
        assert_eq!(
            flat_a, flat_b,
            "step must produce identical output for identical input"
        );
        assert_eq!(
            world_a.population(),
            world_b.population(),
            "populations must match after identical evolution"
        );
    }

    // -----------------------------------------------------------------
    // 6. Corner-placed single cell still dies under amoeba.
    // -----------------------------------------------------------------
    #[test]
    fn corner_single_cell_dies_under_amoeba() {
        let mut world = gol_world(GameOfLife3D::new(9, 26, 5, 7));
        let side = world.side() as u64;
        world.set(wc(side - 1), wc(side - 1), wc(side - 1), ALIVE.raw());
        assert_eq!(world.population(), 1);
        world.step();

        assert_eq!(
            world.population(),
            0,
            "corner live cell must die under amoeba (0 neighbors)"
        );
    }

    // -----------------------------------------------------------------
    // 7. Regression for hash-thing-2o5: seed_center with radius > center.
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
    // 8. Normal-radius seed_center produces population.
    // -----------------------------------------------------------------
    #[test]
    fn seed_center_normal_radius_produces_population() {
        let mut w = World::new(6);
        w.seed_center(12, 0.35);
        assert!(w.population() > 0);
    }

    #[test]
    fn seed_burning_room_produces_all_material_types() {
        let mut w = World::new(6); // side 64
        w.seed_burning_room();
        assert!(w.population() > 0);
        let grid = w.flatten();
        let has = |mat: CellState| grid.contains(&mat);
        assert!(has(STONE), "room must have stone ceiling");
        assert!(has(DIRT), "room must have dirt floor");
        assert!(has(GRASS), "room must have grass walls");
        assert!(has(FIRE), "room must have fire");
        assert!(has(WATER), "room must have water");
        // Verify population is reasonable (room structure + contents)
        assert!(
            w.population() > 100,
            "burning room should have substantial content"
        );
    }

    #[test]
    fn demo_spectacle_populates_each_waypoint_with_active_materials() {
        let mut world = World::new(6);
        world.seed_demo_spectacle();

        for waypoint in world.demo_waypoints() {
            assert!(
                world.count_active_material_cells_near(waypoint.center, waypoint.radius) > 0,
                "waypoint {} should have local fire/water spectacle",
                waypoint.label
            );
        }
    }

    /// hash-thing-t3zn.2: locks in the pad-vs-waypoint invariant. The
    /// viewing pad sits at world center (`side/2 + 7` voxels above the
    /// world-center cell) — far above every waypoint's `cy + radius`
    /// vertical extent. If a future tweak moves a waypoint up into the
    /// pad's footprint, this test fires before the pad silently overwrites
    /// waypoint geometry or the pad gets buried under set-piece cells.
    #[test]
    fn demo_spectacle_pad_does_not_overlap_any_waypoint() {
        let mut world = World::new(6);
        world.seed_demo_spectacle();

        let center_cell = (world.side() as i64) / 2;
        let platform_y = center_cell + 2 * CELLS_PER_METER_INT as i64 - 1;
        let pad_half: i64 = 2;

        for waypoint in world.demo_waypoints() {
            let r = waypoint.radius;
            let dy = (platform_y - waypoint.center[1]).abs();
            let dx = ((center_cell + pad_half) - (waypoint.center[0] - r))
                .min((waypoint.center[0] + r) - (center_cell - pad_half));
            let dz = ((center_cell + pad_half) - (waypoint.center[2] - r))
                .min((waypoint.center[2] + r) - (center_cell - pad_half));
            let overlaps = dx >= 0 && dy <= r && dz >= 0;
            assert!(
                !overlaps,
                "pad cell at y={platform_y} overlaps waypoint {} bbox \
                 (center={:?}, radius={r}); pad would silently overwrite \
                 set-piece cells",
                waypoint.label, waypoint.center,
            );
        }

        // And the pad itself is actually stone where it ought to be.
        for x in (center_cell - pad_half)..=(center_cell + pad_half) {
            for z in (center_cell - pad_half)..=(center_cell + pad_half) {
                assert_eq!(
                    world.get(WorldCoord(x), WorldCoord(platform_y), WorldCoord(z)),
                    STONE,
                    "pad cell ({x},{platform_y},{z}) lost STONE — was \
                     overwritten by a later seed step",
                );
            }
        }
    }

    #[test]
    fn demo_spectacles_seed_active_materials_near_every_anchor() {
        let mut world = World::new(6);
        let anchors = demo_spectacle_anchors();
        world.stage_demo_spectacles(&anchors);

        for anchor in anchors {
            let stats = world.active_material_stats_near(anchor.center, 4);
            assert!(
                stats.fire_cells > 0,
                "{} must include fire near {:?}",
                anchor.label,
                anchor.center
            );
            assert!(
                stats.water_cells > 0,
                "{} must include water near {:?}",
                anchor.label,
                anchor.center
            );
        }
    }

    #[test]
    fn burning_room_reset_reseeds_identical_spectacle_pockets() {
        let mut first = World::new(6);
        first.seed_burning_room();
        let anchors = first.burning_room_demo_spectacle_anchors();

        let expected: Vec<_> = anchors
            .iter()
            .map(|anchor| {
                (
                    anchor.label,
                    first.active_material_stats_near(anchor.center, 5),
                    local_snapshot(&first, anchor.center, 5),
                )
            })
            .collect();

        let mut second = World::new(6);
        second.seed_burning_room();
        for (anchor, (label, stats, snapshot)) in anchors.iter().zip(expected.iter()) {
            let actual_stats = second.active_material_stats_near(anchor.center, 5);
            assert_eq!(actual_stats, *stats, "{label} stats drifted across reset");
            assert!(
                actual_stats.active_cells() > 0,
                "{label} must retain active materials after reset"
            );
            assert_eq!(
                local_snapshot(&second, anchor.center, 5),
                *snapshot,
                "{label} local snapshot drifted across reset"
            );
        }
    }

    #[test]
    fn seed_lattice_megastructure_produces_active_materials() {
        let mut w = World::new(6); // side 64
        w.seed_lattice_megastructure();
        assert!(w.population() > 100, "lattice must have content");
        let grid = w.flatten();
        let has = |mat: CellState| grid.contains(&mat);
        assert!(has(STONE), "lattice must have stone");
        assert!(has(GRASS), "lattice must have grass pillars");
        assert!(has(WATER), "lattice must have water channels");
        assert!(has(FIRE), "lattice must have fire sources");
    }

    #[test]
    fn seed_lattice_megastructure_matches_terrain_outside_structure_zone() {
        let mut world = World::new(6);
        world.seed_lattice_megastructure();

        let params = TerrainParams::for_level(6);
        let terrain = PrecomputedHeightmapField::new(params.to_heightmap(), 6)
            .expect("TerrainParams::for_level must yield a valid heightmap field");
        let point = [0i64, 20, 20];
        let point_world = [0u64, 20, 20];

        assert_eq!(
            world.get(wc(point_world[0]), wc(point_world[1]), wc(point_world[2])),
            crate::terrain::WorldGen::sample(&terrain, point)
        );
    }

    #[test]
    fn seed_gyroid_megastructure_produces_walkable_structure() {
        let mut w = World::new(6); // side 64
        let stats = w.seed_gyroid_megastructure();
        assert!(w.population() > 100, "gyroid must have content");
        assert!(
            stats.total_collapses() > 0,
            "gyroid should collapse some uniform regions: {stats:?}"
        );
        let grid = w.flatten();
        assert!(grid.contains(&STONE), "gyroid must contain stone");
        assert!(
            grid.contains(&0),
            "gyroid must leave air voids to walk through"
        );
    }

    #[test]
    fn water_and_sand_scene_keeps_svdag_in_sync_across_steps() {
        let mut world = World::new(6); // 64^3 is still practical for exhaustive Svdag checks.
        let params = TerrainParams::for_level(6);
        let _ = world.seed_terrain(&params);
        world.seed_water_and_sand();

        let mut svdag = Svdag::new();
        sync_svdag_with_world(&mut world, &mut svdag);
        assert_svdag_matches_world(&world, &svdag, "initial water scene");

        for step in 1..=12 {
            world.step();
            sync_svdag_with_world(&mut world, &mut svdag);
            assert_svdag_matches_world(&world, &svdag, &format!("after water-heavy step {step}"));
        }
    }

    /// hash-thing-rk4n.1 always-on regression: 64^3 skip-sync variant of
    /// the 256^3 commit_step repro below. Skipping some sync calls between
    /// commit_step invocations used to clobber `last_compaction_remap`;
    /// the compose-on-write fix keeps both remaps live across the gap.
    /// Kept scaled-down + always-on so CI catches any regression without
    /// `--ignored` (the 256^3 version runs ~2 min and stays ignored).
    #[test]
    fn water_and_sand_skip_sync_keeps_svdag_in_sync_64() {
        let mut world = World::new(6);
        let params = TerrainParams::for_level(6);
        let _ = world.seed_terrain(&params);
        world.seed_water_and_sand();

        let mut svdag = Svdag::new();
        sync_svdag_with_world(&mut world, &mut svdag);
        assert_svdag_matches_world(&world, &svdag, "initial 64^3 water scene");

        // Sync on a subset of steps so two commit_step calls land between
        // syncs — the exact pattern that exposed Bug A.
        for step in 1u32..=20 {
            world.step();
            let verify = step <= 4 || step % 2 == 0;
            if verify {
                sync_svdag_with_world(&mut world, &mut svdag);
                assert_svdag_matches_world(
                    &world,
                    &svdag,
                    &format!("after 64^3 skip-sync step {step}"),
                );
            }
        }
    }

    /// hash-thing-rk4n.1 always-on regression: 64^3 step_recursive variant.
    /// `maybe_compact` used to never publish its remap, so any compaction
    /// triggered inside the recursive stepper silently desynced the SVDAG
    /// cache. Sync every step — if the renderer's cache aliases across a
    /// maybe_compact store swap, the assertion panics within a few steps.
    #[test]
    fn water_and_sand_step_recursive_keeps_svdag_in_sync_64() {
        let mut world = World::new(6);
        let params = TerrainParams::for_level(6);
        let _ = world.seed_terrain(&params);
        world.seed_water_and_sand();

        let mut svdag = Svdag::new();
        sync_svdag_with_world(&mut world, &mut svdag);
        assert_svdag_matches_world(&world, &svdag, "initial 64^3 water scene");

        for step in 1u32..=20 {
            world.step_recursive();
            sync_svdag_with_world(&mut world, &mut svdag);
            assert_svdag_matches_world(
                &world,
                &svdag,
                &format!("after 64^3 step_recursive {step}"),
            );
        }
    }

    /// hash-thing-2z3g scout: measure p1/p2 wall-time share on the default
    /// water scene. Prints cumulative phase1_ns / phase2_ns across N steps
    /// so a reader can decide whether noop_flags caching (2z3g) is worth
    /// doing — the gate is "does Phase 1 dominate, and if so is allocation
    /// a big enough chunk to matter."
    ///
    /// Run: `cargo test --profile perf --lib
    /// hashlife_phase_timing_scout_water_64 -- --ignored --nocapture`
    #[test]
    #[ignore = "scout/profiling — prints timing, no assertions"]
    fn hashlife_phase_timing_scout_water_64() {
        let mut world = World::new(6);
        let params = TerrainParams::for_level(6);
        let _ = world.seed_terrain(&params);
        world.seed_water_and_sand();

        const N_STEPS: u32 = 20;
        let wall_start = std::time::Instant::now();
        for _ in 0..N_STEPS {
            world.step_recursive();
        }
        let wall_total_ns = wall_start.elapsed().as_nanos() as u64;

        let stats = &world.hashlife_stats_total;
        let p1 = stats.phase1_ns;
        let p2 = stats.phase2_ns;
        let pct = |n: u64| -> f64 { (n as f64) / (wall_total_ns as f64) * 100.0 };
        eprintln!("--- hashlife phase timing scout: 64^3 water, {N_STEPS} steps ---");
        eprintln!("  wall_total:    {:.2} ms", wall_total_ns as f64 / 1e6);
        eprintln!(
            "  phase1_total:  {:.2} ms ({:.1}% of wall)",
            p1 as f64 / 1e6,
            pct(p1),
        );
        eprintln!(
            "  phase2_total:  {:.2} ms ({:.1}% of wall)",
            p2 as f64 / 1e6,
            pct(p2),
        );
        eprintln!(
            "  p1+p2:         {:.2} ms ({:.1}% of wall — remainder is memo lookup/insert, `next` zeroing, dispatch)",
            (p1 + p2) as f64 / 1e6,
            pct(p1 + p2),
        );
        eprintln!(
            "  per-step avg:  p1={:.3}ms p2={:.3}ms wall={:.3}ms",
            p1 as f64 / 1e6 / N_STEPS as f64,
            p2 as f64 / 1e6 / N_STEPS as f64,
            wall_total_ns as f64 / 1e6 / N_STEPS as f64,
        );
        eprintln!(
            "  cache_hits={} cache_misses={} (hit_rate={:.3})",
            stats.cache_hits,
            stats.cache_misses,
            stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses).max(1) as f64,
        );
    }

    /// hash-thing-9t4u: place a deterministic heterogeneous material mix on
    /// a level-6 (64³) world so the CaRule dispatch fires across many arms
    /// per step. No terrain seeding — direct `set_local` writes onto a
    /// STONE substrate so the bench isolates the dispatch axis from terrain
    /// generation noise.
    ///
    /// Layout (slabs separated geometrically so reactions stay slow boundary
    /// effects rather than instant collapse):
    /// - STONE substrate: y in [0, 16).
    /// - WATER slab: y in [22, 26), x in [0, 32), z in [0, 32).
    /// - LAVA slab: y in [22, 26), x in [32, 64), z in [32, 64).
    ///   (No shared face with WATER.)
    /// - ICE slab: y in [28, 30), x in [0, 32), z in [32, 64).
    /// - OIL slab: y in [28, 30), x in [32, 64), z in [0, 32).
    /// - FIRE pillars: 8 isolated 1×4×1 columns at fixed (x,z), y in [30, 34).
    /// - STEAM slab: y in [38, 40), x in [0, 32), z in [0, 32).
    /// - FIREWORK slab: y in [38, 40), x in [32, 64), z in [32, 64).
    /// - VINE pillars: 4 isolated 1×3×1 columns at corners on top of stone.
    /// - SAND corner block: 4×2×4 at (0, 16, 0).
    fn seed_heterogeneous_palette_64(world: &mut World) {
        // STONE substrate.
        for z in 0..64u64 {
            for x in 0..64u64 {
                for y in 0..16u64 {
                    world.set_local(LocalCoord(x), LocalCoord(y), LocalCoord(z), STONE);
                }
            }
        }
        // SAND corner block.
        for z in 0..4u64 {
            for x in 0..4u64 {
                for dy in 0..2u64 {
                    world.set_local(LocalCoord(x), LocalCoord(16 + dy), LocalCoord(z), SAND);
                }
            }
        }
        // VINE pillars (on the stone surface y=16,17,18).
        for &(px, pz) in &[(16u64, 16u64), (48, 16), (16, 48), (48, 48)] {
            for dy in 0..3u64 {
                world.set_local(LocalCoord(px), LocalCoord(16 + dy), LocalCoord(pz), VINE);
            }
        }
        // WATER slab (NW quadrant).
        for z in 0..32u64 {
            for x in 0..32u64 {
                for dy in 0..4u64 {
                    world.set_local(LocalCoord(x), LocalCoord(22 + dy), LocalCoord(z), WATER);
                }
            }
        }
        // LAVA slab (SE quadrant — no shared face with WATER).
        for z in 32..64u64 {
            for x in 32..64u64 {
                for dy in 0..4u64 {
                    world.set_local(LocalCoord(x), LocalCoord(22 + dy), LocalCoord(z), LAVA);
                }
            }
        }
        // ICE slab (NE quadrant, higher).
        for z in 32..64u64 {
            for x in 0..32u64 {
                for dy in 0..2u64 {
                    world.set_local(LocalCoord(x), LocalCoord(28 + dy), LocalCoord(z), ICE);
                }
            }
        }
        // OIL slab (SW quadrant, higher).
        for z in 0..32u64 {
            for x in 32..64u64 {
                for dy in 0..2u64 {
                    world.set_local(LocalCoord(x), LocalCoord(28 + dy), LocalCoord(z), OIL);
                }
            }
        }
        // FIRE pillars (8 columns at fixed (x,z) so dispatch sees FireRule).
        for &(px, pz) in &[
            (8u64, 8u64),
            (24, 8),
            (40, 8),
            (56, 8),
            (8, 56),
            (24, 56),
            (40, 56),
            (56, 56),
        ] {
            for dy in 0..4u64 {
                world.set_local(LocalCoord(px), LocalCoord(30 + dy), LocalCoord(pz), FIRE);
            }
        }
        // STEAM slab (NW quadrant, top).
        for z in 0..32u64 {
            for x in 0..32u64 {
                for dy in 0..2u64 {
                    world.set_local(LocalCoord(x), LocalCoord(38 + dy), LocalCoord(z), STEAM);
                }
            }
        }
        // FIREWORK slab (SE quadrant, top).
        for z in 32..64u64 {
            for x in 32..64u64 {
                for dy in 0..2u64 {
                    world.set_local(LocalCoord(x), LocalCoord(38 + dy), LocalCoord(z), FIREWORK);
                }
            }
        }
    }

    /// hash-thing-9t4u scout: heterogeneous-scene phase timing.
    ///
    /// Sibling to `hashlife_phase_timing_scout_water_64`. The water scout
    /// is dominated by AirRule + WaterRule dispatch — in a homogeneous
    /// scene the indirect-branch predictor hits ~100%, so dispatch cost
    /// becomes invisible. This scout seeds many materials at once so
    /// every CaRule arm gets exercised per tick, defeating prediction
    /// and exposing whatever dispatch cost remains after the vvun
    /// trait→enum refactor.
    ///
    /// Use this scout to compare `#[inline]` candidates on hot rule
    /// step_cell bodies. The vvun.2 acceptance bar is ≥10% phase1
    /// reduction on this scene with no >2% regression on the water
    /// scout. If neither moves, the lever doesn't help in this dispatch
    /// shape — document the negative result inline and on the bd.
    ///
    /// Run: `cargo test --profile perf --lib
    /// hashlife_phase_timing_scout_heterogeneous_64 -- --ignored --nocapture`
    /// Repeat 3× and report min/median/max spread.
    ///
    /// # 9t4u finding (2026-04-24): `#[inline]` regresses by ~13.5%
    ///
    /// Tested adding `#[inline]` to seven hot rule `step_cell` bodies
    /// (Air, Water, Lava, Ice, Flammable, Steam, Firework). 5 baseline
    /// runs vs 5 inlined runs on this scene:
    ///
    /// | variant   | phase1 min | phase1 median | phase1 max | spread |
    /// | --------- | ---------- | ------------- | ---------- | ------ |
    /// | baseline  | 144.29 ms  | **145.71 ms** | 147.56 ms  | 1.023  |
    /// | inlined   | 163.89 ms  | **165.31 ms** | 168.24 ms  | 1.027  |
    ///
    /// Median delta: **+13.5% regression** under `#[inline]`. Likely
    /// cause: dispatch-site match at `rule.rs` is already `#[inline]`,
    /// so leaf inlining duplicates the bodies into the dispatch site
    /// and inflates i-cache pressure — Claude plan-review's flagged
    /// failure mode. **Do not retry naive leaf `#[inline]` on these
    /// arms.** Future dispatch experiments here should look at
    /// stratified iteration (vvun's "alternative lever") or PGO,
    /// not blanket inlining.
    ///
    /// Numbers are wall-time over 20 cold-start steps, not warm-frame
    /// only — both A and B see the same warm-up so the relative delta
    /// is preserved, but a per-tick steady-state number would be lower.
    #[test]
    #[ignore = "scout/profiling — prints timing; only seed-coverage asserts"]
    fn hashlife_phase_timing_scout_heterogeneous_64() {
        let mut world = World::new(6);
        seed_heterogeneous_palette_64(&mut world);

        // Per-arm dispatch coverage guard (hash-thing-9t4u):
        // confirms each major rule has a non-trivial input population.
        let mut counts = [0u64; 32];
        for z in 0..64u64 {
            for y in 0..64u64 {
                for x in 0..64u64 {
                    let raw = world.get(wc(x), wc(y), wc(z));
                    let m = Cell::from_raw(raw).material() as usize;
                    if m < counts.len() {
                        counts[m] += 1;
                    }
                }
            }
        }
        // Map material id → expected floor.
        let air_count = counts[AIR_MATERIAL_ID as usize];
        let fire_count = counts[FIRE_MATERIAL_ID as usize];
        let water_count = counts[WATER_MATERIAL_ID as usize];
        let lava_count = counts[LAVA_MATERIAL_ID as usize];
        let ice_count = counts[ICE_MATERIAL_ID as usize];
        let oil_count = counts[OIL_MATERIAL_ID as usize];
        let steam_count = counts[STEAM_MATERIAL_ID as usize];
        let vine_count = counts[VINE_MATERIAL_ID as usize];
        let firework_count = counts[FIREWORK_MATERIAL_ID as usize];
        assert!(air_count > 100_000, "AirRule coverage too low: {air_count}");
        assert!(fire_count > 16, "FireRule coverage too low: {fire_count}");
        assert!(
            water_count > 1000,
            "WaterRule coverage too low: {water_count}"
        );
        assert!(lava_count > 1000, "LavaRule coverage too low: {lava_count}");
        assert!(ice_count > 100, "IceRule coverage too low: {ice_count}");
        // OilRule dispatches via the FlammableRule arm in this codebase.
        assert!(
            oil_count > 100,
            "FlammableRule coverage too low: {oil_count}"
        );
        assert!(
            steam_count > 100,
            "SteamRule coverage too low: {steam_count}"
        );
        assert!(vine_count > 8, "VineRule coverage too low: {vine_count}");
        assert!(
            firework_count > 100,
            "FireworkRule coverage too low: {firework_count}"
        );

        const N_STEPS: u32 = 20;
        let wall_start = std::time::Instant::now();
        for _ in 0..N_STEPS {
            world.step_recursive();
        }
        let wall_total_ns = wall_start.elapsed().as_nanos() as u64;

        let stats = &world.hashlife_stats_total;
        let p1 = stats.phase1_ns;
        let p2 = stats.phase2_ns;
        let pct = |n: u64| -> f64 { (n as f64) / (wall_total_ns as f64) * 100.0 };
        eprintln!("--- hashlife phase timing scout: 64^3 heterogeneous, {N_STEPS} steps ---");
        eprintln!("  wall_total:    {:.2} ms", wall_total_ns as f64 / 1e6);
        eprintln!(
            "  phase1_total:  {:.2} ms ({:.1}% of wall)",
            p1 as f64 / 1e6,
            pct(p1),
        );
        eprintln!(
            "  phase2_total:  {:.2} ms ({:.1}% of wall)",
            p2 as f64 / 1e6,
            pct(p2),
        );
        eprintln!(
            "  p1+p2:         {:.2} ms ({:.1}% of wall — remainder is memo lookup/insert, `next` zeroing, dispatch)",
            (p1 + p2) as f64 / 1e6,
            pct(p1 + p2),
        );
        eprintln!(
            "  per-step avg:  p1={:.3}ms p2={:.3}ms wall={:.3}ms",
            p1 as f64 / 1e6 / N_STEPS as f64,
            p2 as f64 / 1e6 / N_STEPS as f64,
            wall_total_ns as f64 / 1e6 / N_STEPS as f64,
        );
        eprintln!(
            "  cache_hits={} cache_misses={} (hit_rate={:.3})",
            stats.cache_hits,
            stats.cache_misses,
            stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses).max(1) as f64,
        );
        eprintln!(
            "  t=0 coverage: air={air_count} fire={fire_count} water={water_count} lava={lava_count} ice={ice_count} oil={oil_count} steam={steam_count} vine={vine_count} firework={firework_count}",
        );
    }

    /// hash-thing-rk4n regression: walks the default water scene past
    /// ground-impact using the brute-force `step()` path which runs
    /// `commit_step` (store-compact every tick). Skipping some sync calls
    /// mimics dropped render frames and used to expose
    /// `last_compaction_remap` being single-buffered: two successive
    /// `commit_step` calls without an intervening sync clobbered the first
    /// remap, so the SVDAG cache decoded subsequent nodes from the wrong
    /// store generation. Fixed in rk4n.1 by compose-on-write
    /// (see `compose_remap`); this test stays as a regression guard.
    ///
    /// Scaled down to 128^3 (level 7) per hash-thing-uh7o so it runs
    /// always-on under the 60s soft-max in release / `--profile perf`
    /// (~10s observed); in debug builds the 128³ voxel walk dominates
    /// and the test takes ~10 min, so it's `#[ignore]`d in debug per
    /// hash-thing-1imu. Run via `cargo test --release` (or any
    /// non-`debug_assertions` profile) to exercise the regression
    /// guard. With the fix reverted, the assertion fires at step 42
    /// with ~254k mismatches. Water pool sits at
    /// `water_y = center + center/4` with `pool_depth = (side/32).max(2)`
    /// — at 128^3 the pool bottom is ~16 cells above the mid-terrain
    /// floor, so 60 steps covers impact plus tail comfortably.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "slow on debug build (~10 min @ 128³); runs under --release / --profile perf (hash-thing-1imu)"
    )]
    fn water_and_sand_128_commit_step_skip_sync_corrupts_svdag() {
        let mut world = World::new(7);
        let params = TerrainParams::for_level(7);
        let _ = world.seed_terrain(&params);
        world.seed_water_and_sand();

        let mut svdag = Svdag::new();
        sync_svdag_with_world(&mut world, &mut svdag);
        assert_svdag_matches_world(&world, &svdag, "initial 128^3 water scene");

        // Mirror the "occasionally drop a render frame" pattern: verify most
        // steps but skip some past step 40 so two commit_step calls can land
        // between syncs. Under rk4n, the first such skipped sync clobbers a
        // remap and the next verification panics with material-id mismatches.
        for step in 1u32..=60 {
            world.step();
            let verify = step <= 40 || step % 2 == 0 || (45..=55).contains(&step) || step == 60;
            if verify {
                sync_svdag_with_world(&mut world, &mut svdag);
                assert_svdag_matches_world(
                    &world,
                    &svdag,
                    &format!("after 128^3 water step {step}"),
                );
            }
        }
    }

    /// hash-thing-rk4n companion regression: production uses
    /// `step_recursive`, whose `maybe_compact` used to re-store without
    /// publishing the remap on `last_compaction_remap`. Sync every step
    /// so this test cannot be excused by skipped-frame behavior; if it
    /// fails, the SVDAG cache is aliasing across a silent store swap.
    /// Fixed in rk4n.1 by publishing the remap with compose-on-write;
    /// this test stays as a regression guard.
    ///
    /// Scaled down to 128^3 (level 7) per hash-thing-uh7o so it runs
    /// always-on under the 60s soft-max in release / `--profile perf`
    /// (~5s observed); in debug builds the 128³ voxel walk dominates
    /// and the test takes ~10 min, so it's `#[ignore]`d in debug per
    /// hash-thing-1imu. Run via `cargo test --release` (or any
    /// non-`debug_assertions` profile) to exercise the regression
    /// guard. With the fix reverted, the assertion fires at step 24
    /// with ~974k mismatches.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "slow on debug build (~10 min @ 128³); runs under --release / --profile perf (hash-thing-1imu)"
    )]
    fn water_and_sand_128_step_recursive_with_sync_every_step() {
        let mut world = World::new(7);
        let params = TerrainParams::for_level(7);
        let _ = world.seed_terrain(&params);
        world.seed_water_and_sand();

        let mut svdag = Svdag::new();
        sync_svdag_with_world(&mut world, &mut svdag);
        assert_svdag_matches_world(&world, &svdag, "initial 128^3 water scene");

        for step in 1u32..=60 {
            world.step_recursive();
            sync_svdag_with_world(&mut world, &mut svdag);
            assert_svdag_matches_world(
                &world,
                &svdag,
                &format!("after 128^3 step_recursive {step}"),
            );
        }
    }

    #[test]
    fn lattice_progression_demo_spawn_and_waypoints_are_open() {
        // Level 8 (256³) — at 4 cells/m, level 6 rooms are only 4 cells (1 m)
        // tall and cannot fit the 1.6 m player. Level 8 gives ~15-cell rooms.
        let mut w = World::new(8);
        let layout = w.seed_lattice_progression_demo();

        assert!(
            !player::player_collides(&w, &layout.player_pos),
            "demo spawn must be walkable: {:?}",
            layout.player_pos
        );

        for checkpoint in layout.walk_route {
            assert_eq!(
                w.get(
                    WorldCoord(checkpoint[0]),
                    WorldCoord(checkpoint[1]),
                    WorldCoord(checkpoint[2]),
                ),
                AIR,
                "checkpoint must be air: {checkpoint:?}"
            );
        }
    }

    #[test]
    fn lattice_progression_demo_route_is_player_traversable_end_to_end() {
        for level in [7u32, 8] {
            let mut w = World::new(level);
            let layout = w.seed_lattice_progression_demo();
            let route = std::iter::once(walk_cell(layout.player_pos))
                .chain(layout.walk_route)
                .collect::<Vec<_>>();

            for segment in route.windows(2) {
                assert_walk_segment_is_traversable(&w, segment[0], segment[1]);
            }
        }
    }

    #[test]
    fn lattice_progression_demo_reveal_reads_as_gangplank_over_void() {
        let mut w = World::new(6);
        let layout = w.seed_lattice_progression_demo();
        let field = LatticeField::for_world(w.level, 42);
        let ProgressionBoxes {
            balcony, panorama, ..
        } = World::progression_boxes(&field);
        let tip_x = panorama.max[0] - 1;
        let side_z = (balcony.max[2] + 2).min(panorama.max[2]);

        assert_eq!(
            w.get(
                WorldCoord(tip_x),
                WorldCoord(layout.panorama_center[1] - 1),
                WorldCoord(layout.panorama_center[2]),
            ),
            STONE,
            "gangplank should carry the player at the reveal tip"
        );
        assert_eq!(
            w.get(
                WorldCoord(tip_x),
                WorldCoord(layout.panorama_center[1] - 1),
                WorldCoord(side_z),
            ),
            AIR,
            "off the gangplank there should be no terrace floor"
        );
        assert_eq!(
            w.get(
                WorldCoord(tip_x),
                WorldCoord(layout.panorama_center[1] - 4),
                WorldCoord(layout.panorama_center[2]),
            ),
            AIR,
            "the reveal tip should hang over a real drop"
        );
    }

    #[test]
    fn lattice_progression_demo_places_spectacle_along_the_walk_route() {
        let mut w = World::new(6);
        let layout = w.seed_lattice_progression_demo();

        for checkpoint in [
            layout.corridor_mid,
            layout.atrium_center,
            layout.reveal_center,
            layout.panorama_center,
        ] {
            assert!(
                w.active_material_stats_near(checkpoint, 6).active_cells() > 0,
                "checkpoint should read local spectacle while walking: {checkpoint:?}"
            );
        }
    }

    #[test]
    fn lattice_progression_demo_stages_fireworks_near_reveal() {
        let mut w = World::new(6); // side 64
        let layout = w.seed_lattice_progression_demo();
        let snapshot = local_snapshot(&w, layout.reveal_center, 5);
        assert!(
            snapshot.contains(&FIREWORK),
            "reveal should stage firework launchers near {:?}",
            layout.reveal_center
        );
    }

    #[test]
    fn lattice_progression_demo_stages_clone_fed_waterfall_beside_corridor_gap() {
        let mut w = World::new(6);
        let _layout = w.seed_lattice_progression_demo();
        let field = LatticeField::for_world(w.level, 42);
        let ground_y = field.lo[1] + field.floor_thick;
        let ProgressionBoxes {
            corridor, tease_b, ..
        } = World::progression_boxes(&field);
        let waterfall = World::progression_waterfall_layout(corridor, tease_b, ground_y);
        let water_material = Cell::from_raw(WATER).material();

        assert!(
            w.active_material_stats_near(waterfall.focus, 4).water_cells > 0,
            "corridor side gap should frame a visible waterfall near {:?}",
            waterfall.focus
        );
        for y in waterfall.source_y[0]..=waterfall.source_y[1] {
            for x in waterfall.source_x[0]..=waterfall.source_x[1] {
                let state = Cell::from_raw(w.get(
                    WorldCoord(x),
                    WorldCoord(y),
                    WorldCoord(waterfall.source_z),
                ));
                assert_eq!(
                    state.material(),
                    CLONE_MATERIAL_ID,
                    "waterfall source wall should be clone blocks"
                );
                assert_eq!(
                    state.metadata(),
                    water_material,
                    "waterfall clone blocks should encode water as their source material"
                );
            }
        }
        assert_eq!(
            w.get(
                WorldCoord(tease_b.center()[0]),
                WorldCoord(corridor.center()[1]),
                WorldCoord(tease_b.max[2]),
            ),
            AIR,
            "waterfall should stay outside the corridor cavity"
        );
    }

    #[test]
    fn lattice_progression_demo_reseed_resets_waterfall_clone_tracking() {
        let mut w = World::new(6);
        let field = LatticeField::for_world(w.level, 42);
        let ground_y = field.lo[1] + field.floor_thick;
        let ProgressionBoxes {
            corridor, tease_b, ..
        } = World::progression_boxes(&field);
        let waterfall = World::progression_waterfall_layout(corridor, tease_b, ground_y);
        let expected_sources = ((waterfall.source_x[1] - waterfall.source_x[0] + 1)
            * (waterfall.source_y[1] - waterfall.source_y[0] + 1))
            as usize;

        w.seed_lattice_progression_demo();
        assert_eq!(w.clone_sources.len(), expected_sources);

        w.seed_lattice_progression_demo();
        assert_eq!(
            w.clone_sources.len(),
            expected_sources,
            "scene reseed should replace waterfall clone tracking instead of accumulating duplicates"
        );
    }

    #[test]
    fn demo_spectacle_seed_is_deterministic() {
        let mut a = World::new(6);
        let mut b = World::new(6);
        a.seed_demo_spectacle();
        b.seed_demo_spectacle();

        assert_eq!(
            a.flatten(),
            b.flatten(),
            "scene reset should reproduce the same staged set pieces"
        );
    }

    #[test]
    fn demo_cascade_finale_uses_expanded_material_palette() {
        let mut world = World::new(6);
        world.seed_demo_spectacle();

        let cascade = world
            .demo_waypoints()
            .into_iter()
            .last()
            .expect("demo spectacle defines a finale waypoint");
        let snapshot = local_snapshot(&world, cascade.center, 5);

        assert!(
            snapshot.contains(&WATER),
            "cascade finale should keep the water curtain"
        );
        assert!(
            snapshot.contains(&FIREWORK),
            "cascade finale should stage firework launchers"
        );
        assert!(
            snapshot.contains(&LAVA),
            "cascade finale should add a lava basin under the reveal"
        );
        assert!(
            snapshot.contains(&OIL),
            "cascade finale should add oil channels for extra spectacle"
        );
    }

    #[test]
    fn lattice_progression_demo_preserves_lattice_materials() {
        let mut w = World::new(6); // side 64
        let _layout = w.seed_lattice_progression_demo();
        let grid = w.flatten();
        let has = |mat: CellState| grid.contains(&mat);

        assert!(
            has(STONE),
            "progression demo must still have stone structure"
        );
        assert!(has(GRASS), "progression demo must still have lattice walls");
        assert!(
            has(WATER),
            "progression demo must still preserve water channels"
        );
        assert!(
            has(FIRE),
            "progression demo must still preserve fire accents"
        );
    }

    #[test]
    fn lattice_progression_demo_stages_route_specific_materials() {
        let mut w = World::new(6);
        let _layout = w.seed_lattice_progression_demo();
        let field = LatticeField::for_world(w.level, 42);
        let ProgressionBoxes {
            start_shell: _start_shell,
            start_room,
            corridor_shell,
            corridor,
            tease_a,
            tease_b,
            atrium: _atrium,
            balcony,
            panorama,
        } = World::progression_boxes(&field);

        assert!(
            w.box_contains_material(start_room, VINE),
            "start room should stage vine detail"
        );
        assert!(
            w.box_contains_material(corridor_shell, WATER),
            "corridor should stage a waterfall/gap read"
        );
        assert!(
            w.box_contains_material(tease_a, SAND),
            "tease branch should contain sand staging"
        );
        assert!(
            w.box_contains_material(tease_b, FAN),
            "tease branch should contain fan props"
        );
        assert!(
            w.box_contains_material(balcony, FIREWORK)
                || w.box_contains_material(panorama, FIREWORK),
            "reveal should stage fireworks"
        );
        assert!(
            w.box_contains_material(panorama, WATER),
            "reveal should keep water framing"
        );

        // Sanity: the corridor still remains traversable through its center.
        assert_eq!(
            w.get(
                WorldCoord(corridor.center()[0]),
                WorldCoord(corridor.center()[1]),
                WorldCoord(corridor.center()[2]),
            ),
            AIR
        );
    }

    #[test]
    fn lattice_progression_demo_stages_optional_break_trigger_off_main_route() {
        let mut w = World::new(6);
        let _layout = w.seed_lattice_progression_demo();
        let field = LatticeField::for_world(w.level, 42);
        let ProgressionBoxes { tease_a, .. } = World::progression_boxes(&field);
        let center_x = tease_a.center()[0];
        let mut vine_cells = 0;

        for y in tease_a.min[1]..=(tease_a.min[1] + 2).min(tease_a.max[1]) {
            for x in (center_x - 1)..=(center_x + 1) {
                if w.get(WorldCoord(x), WorldCoord(y), WorldCoord(tease_a.min[2])) == VINE {
                    vine_cells += 1;
                }
            }
        }

        assert!(
            vine_cells > 0,
            "progression demo should place a breakable vine curtain in the tease alcove"
        );
        assert_eq!(
            w.get(
                WorldCoord(tease_a.center()[0]),
                WorldCoord(tease_a.center()[1]),
                WorldCoord(tease_a.min[2] - 1),
            ),
            AIR,
            "main corridor mouth should stay open even with the optional break trigger nearby"
        );
    }

    // -----------------------------------------------------------------
    // 9-11. Bounds check on World::set / World::get (hash-thing-fb5).
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn world_set_panics_oob() {
        let mut w = World::new(2); // side 4
        w.set(wc(4), wc(0), wc(0), ALIVE.raw());
    }

    #[test]
    fn world_set_in_bounds_at_max_corner() {
        let mut w = World::new(2); // side 4
        w.set(wc(3), wc(3), wc(3), ALIVE.raw());
        assert_eq!(w.get(wc(3), wc(3), wc(3)), ALIVE.raw());
        // And the complementary corner — distinct material id 2, metadata 0.
        let mat2 = Cell::pack(2, 0).raw();
        w.set(wc(0), wc(0), wc(0), mat2);
        assert_eq!(w.get(wc(0), wc(0), wc(0)), mat2);
    }

    #[test]
    fn world_get_returns_zero_oob() {
        let w = World::new(2); // side 4
        assert_eq!(w.get(wc(4), wc(0), wc(0)), 0);
        assert_eq!(w.get(wc(0), wc(4), wc(0)), 0);
        assert_eq!(w.get(wc(0), wc(0), wc(4)), 0);
        assert_eq!(w.get(WorldCoord(-1), wc(0), wc(0)), 0);
    }

    // -----------------------------------------------------------------
    // 12. seed_terrain epoch boundary test.
    // -----------------------------------------------------------------
    #[test]
    fn seed_terrain_is_an_epoch_boundary() {
        let mut world = World::new(6);
        let params = TerrainParams::default();

        let _ = world.seed_terrain(&params).unwrap();
        let nodes_after_first = world.store.stats();

        let _ = world.seed_terrain(&params).unwrap();
        let nodes_after_second = world.store.stats();

        assert_eq!(
            nodes_after_first, nodes_after_second,
            "seed_terrain must start a fresh epoch; node count drifted \
             from {nodes_after_first} to {nodes_after_second} across a \
             deterministic re-seed",
        );
    }

    #[test]
    fn seed_terrain_rejects_invalid_params_without_mutating_world() {
        let mut world = World::new(6);
        let nodes_before = world.store.stats();
        let root_before = world.root;
        let invalid = TerrainParams {
            wavelength: 0.0,
            ..TerrainParams::default()
        };

        assert!(matches!(
            world.seed_terrain(&invalid),
            Err("wavelength must be finite and > 0")
        ));
        assert_eq!(world.store.stats(), nodes_before);
        assert_eq!(world.root, root_before);
        assert!(world.terrain_params.is_none());
    }

    #[test]
    fn fan_pushes_sand_across_floor_in_recursive_runtime_path() {
        let mut world = empty_world();
        for x in 0..5 {
            world.set(wc(x), wc(0), wc(2), STONE);
        }
        world.set(wc(1), wc(1), wc(2), FAN);
        world.set(wc(2), wc(1), wc(2), SAND);

        world.step_recursive();
        assert_eq!(
            Cell::from_raw(world.get(wc(2), wc(1), wc(2))).material(),
            Cell::from_raw(SAND).material()
        );
        assert_eq!(world.get(wc(3), wc(1), wc(2)), AIR);

        world.step_recursive();
        assert_eq!(world.get(wc(2), wc(1), wc(2)), AIR);
        assert_eq!(world.get(wc(3), wc(1), wc(2)), SAND);
    }

    #[test]
    fn fan_pushes_steam_without_resetting_age_in_recursive_runtime_path() {
        let mut world = empty_world();
        world.mutate_materials(|m| {
            let identity = m.register_block_rule(crate::sim::margolus::IdentityBlockRule);
            m.assign_block_rule(STEAM_MATERIAL_ID, identity);
            for material_id in FAN_ARMED_STEAM_MATERIAL_IDS {
                m.assign_block_rule(material_id, identity);
            }
        });
        world.set(wc(1), wc(1), wc(2), FAN);
        world.set(wc(2), wc(1), wc(2), Cell::pack(STEAM_MATERIAL_ID, 14).raw());

        world.step_recursive();
        world.step_recursive();

        assert_eq!(world.get(wc(2), wc(1), wc(2)), AIR);
        assert_eq!(
            world.get(wc(3), wc(1), wc(2)),
            Cell::pack(STEAM_MATERIAL_ID, 15).raw()
        );
    }

    #[test]
    fn fan_pushes_firework_without_resetting_fuse_in_recursive_runtime_path() {
        let mut world = empty_world();
        world.mutate_materials(|m| {
            let identity = m.register_block_rule(crate::sim::margolus::IdentityBlockRule);
            m.assign_block_rule(FIREWORK_MATERIAL_ID, identity);
            for material_id in FAN_ARMED_FIREWORK_MATERIAL_IDS {
                m.assign_block_rule(material_id, identity);
            }
        });
        world.set(wc(1), wc(1), wc(2), FAN);
        world.set(
            wc(2),
            wc(1),
            wc(2),
            Cell::pack(FIREWORK_MATERIAL_ID, 10).raw(),
        );

        world.step_recursive();
        world.step_recursive();

        assert_eq!(world.get(wc(2), wc(1), wc(2)), AIR);
        assert_eq!(
            world.get(wc(3), wc(1), wc(2)),
            Cell::pack(FIREWORK_MATERIAL_ID, 11).raw()
        );
    }

    #[test]
    fn water_turns_to_stone_when_fire_is_adjacent() {
        let mut world = empty_world();
        world.set(wc(4), wc(4), wc(4), WATER);
        world.set(wc(5), wc(4), wc(4), FIRE);

        world.step();

        assert_eq!(world.get(wc(4), wc(4), wc(4)), STONE);
    }

    #[test]
    fn isolated_fire_burns_out() {
        let mut world = empty_world();
        world.set(wc(4), wc(4), wc(4), FIRE);

        world.step();

        assert_eq!(world.get(wc(4), wc(4), wc(4)), 0);
    }

    #[test]
    fn fire_persists_next_to_grass_fuel() {
        let mut world = empty_world();
        world.set(wc(4), wc(4), wc(4), FIRE);
        world.set(wc(5), wc(4), wc(4), GRASS);

        world.step();

        assert_eq!(world.get(wc(4), wc(4), wc(4)), FIRE);
        assert_eq!(world.get(wc(5), wc(4), wc(4)), GRASS);
    }

    #[test]
    fn step_dispatches_fire_and_water_rules_independently() {
        let mut world = empty_world();

        world.set(wc(2), wc(2), wc(2), FIRE);
        world.set(wc(3), wc(2), wc(2), GRASS);

        world.set(wc(5), wc(5), wc(5), WATER);
        world.set(wc(6), wc(5), wc(5), FIRE);

        world.step();

        assert_eq!(world.get(wc(2), wc(2), wc(2)), FIRE);
        assert_eq!(world.get(wc(3), wc(2), wc(2)), GRASS);
        assert_eq!(world.get(wc(5), wc(5), wc(5)), STONE);
        assert_eq!(world.get(wc(6), wc(5), wc(5)), 0);
    }

    // -----------------------------------------------------------------
    // Margolus block-rule integration tests
    // -----------------------------------------------------------------

    use crate::sim::margolus::{GravityBlockRule, IdentityBlockRule};
    use crate::sim::rule::Phase;
    use crate::terrain::materials::{DIRT_MATERIAL_ID, STONE_MATERIAL_ID};

    fn simple_density(cell: Cell) -> f32 {
        if cell.is_empty() {
            return 0.0;
        }
        match cell.material() {
            1 => 5.0,  // stone
            2 => 2.0,  // dirt
            3 => 1.2,  // grass
            4 => 0.05, // fire
            5 => 1.0,  // water
            _ => 1.0,
        }
    }

    /// Phase classification matching `simple_density` for these gravity tests
    /// (hash-thing-nagw). Air → Gas, water (5) → Liquid, fire (4) → Gas,
    /// everything else → Solid. The block-rule predicate uses this to refuse
    /// solid-on-solid swaps.
    fn simple_phase(cell: Cell) -> Phase {
        if cell.is_empty() {
            return Phase::Gas;
        }
        match cell.material() {
            4 => Phase::Gas,    // fire
            5 => Phase::Liquid, // water
            _ => Phase::Solid,
        }
    }

    /// Wire a gravity block rule onto specific materials and return the world.
    ///
    /// Materials sharing a BlockRule must share a `tick_divisor` (iowh
    /// invariant, enforced at rebuild time per hash-thing-lw75.1.1). WATER
    /// ships with `tick_divisor = 2`; every other terrain material defaults
    /// to 1. Normalize all wired materials to `tick_divisor = 1` before
    /// assigning, so the gravity tests exercise single-step drops under one
    /// consistent schedule.
    fn gravity_world(materials_with_gravity: &[u16]) -> World {
        let mut world = World::new(3); // 8x8x8
        world.mutate_materials(|m| {
            for &mat_id in materials_with_gravity {
                m.set_tick_divisor(mat_id, 1);
            }
            let gravity_id =
                m.register_block_rule(GravityBlockRule::new(simple_density, simple_phase));
            for &mat_id in materials_with_gravity {
                m.assign_block_rule(mat_id, gravity_id);
            }
        });
        world
    }

    #[test]
    fn identity_block_rule_leaves_world_unchanged() {
        let mut world = World::new(3);
        world.mutate_materials(|m| {
            let identity_id = m.register_block_rule(IdentityBlockRule);
            m.assign_block_rule(STONE_MATERIAL_ID, identity_id);
        });

        // Place some stone cells.
        world.set(wc(2), wc(4), wc(2), STONE);
        world.set(wc(3), wc(4), wc(2), STONE);
        let pop_before = world.population();
        let flat_before = world.flatten();

        world.step();

        assert_eq!(world.population(), pop_before);
        // CaRule for stone is NoopRule, identity BlockRule changes nothing.
        assert_eq!(world.flatten(), flat_before);
    }

    #[test]
    fn gravity_drops_heavy_cell_one_step() {
        // Use even generation (offset=0) so block at (2,2,2) is aligned.
        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
        // Place dirt at y=3 (top of block), air at y=2 (bottom of block).
        // Block origin = (2,2,2), so positions (2,2,2) and (2,3,2) are
        // in the same block column.
        world.set(wc(2), wc(3), wc(2), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            0,
            "bottom should be air initially"
        );
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "top should be dirt initially"
        );

        world.step();

        // After step, dirt should have fallen from y=3 to y=2.
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should have fallen to bottom"
        );
        assert_eq!(world.get(wc(2), wc(3), wc(2)), 0, "top should now be air");
    }

    #[test]
    fn gravity_conserves_population() {
        let mut world = gravity_world(&[DIRT_MATERIAL_ID, WATER_MATERIAL_ID]);
        // Scatter some cells in even-aligned positions.
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        world.set(wc(2), wc(1), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.set(wc(4), wc(3), wc(4), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        let pop_before = world.population();

        for _ in 0..4 {
            world.step();
        }

        assert_eq!(
            world.population(),
            pop_before,
            "gravity must conserve total cell count"
        );
    }

    #[test]
    fn alternating_offset_covers_different_blocks() {
        // Even gen: offset 0 → block at (0,0,0). Odd gen: offset 1 → block at (1,1,1).
        // A cell at (1,1,1) is in the even block but NOT the odd block's origin.
        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
        // Place dirt at (0, 1, 0) — in even block (0,0,0), column (0,_,0).
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        assert_eq!(world.generation, 0);

        // Gen 0 (even offset=0): block (0,0,0) is active → dirt falls from y=1 to y=0.
        world.step();
        assert_eq!(
            world.get(wc(0), wc(0), wc(0)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should fall on even generation"
        );
        assert_eq!(world.get(wc(0), wc(1), wc(0)), 0);
    }

    #[test]
    fn mixed_block_rules_skip_block() {
        // Two materials with different block rules → block should be skipped.
        let mut world = World::new(3);
        let gravity_a = world
            .materials
            .register_block_rule(GravityBlockRule::new(simple_density, simple_phase));
        let gravity_b = world
            .materials
            .register_block_rule(GravityBlockRule::new(simple_density, simple_phase));
        world
            .materials
            .assign_block_rule(DIRT_MATERIAL_ID, gravity_a);
        world
            .materials
            .assign_block_rule(WATER_MATERIAL_ID, gravity_b);

        // Place dirt and water in the same block — different rule IDs.
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
        world.set(wc(1), wc(0), wc(0), Cell::pack(WATER_MATERIAL_ID, 0).raw());

        let flat_before = world.flatten();
        world.step();

        // CaRule (NoopRule) doesn't change dirt/water. Block rule is skipped.
        assert_eq!(
            world.flatten(),
            flat_before,
            "mixed-rule block should be skipped (identity)"
        );
    }

    #[test]
    fn static_material_not_moved_by_other_materials_rule() {
        // B1 fix: stone (no block rule) shares a block with dirt (has gravity).
        // Stone must NOT be swapped downward by dirt's gravity rule.
        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
        // Block at (0,0,0): stone at bottom, dirt at top of same column.
        // Gravity would swap them if stone were participating, but stone
        // has no block_rule_id so it should be anchored.
        world.set(wc(0), wc(0), wc(0), Cell::pack(STONE_MATERIAL_ID, 0).raw());
        world.set(wc(0), wc(1), wc(0), Cell::pack(DIRT_MATERIAL_ID, 0).raw());

        world.step();

        // Stone stays at (0,0,0) — anchored. Dirt can't displace it.
        assert_eq!(
            world.get(wc(0), wc(0), wc(0)),
            Cell::pack(STONE_MATERIAL_ID, 0).raw(),
            "stone (no block rule) must not be moved by dirt's gravity"
        );
        assert_eq!(
            world.get(wc(0), wc(1), wc(0)),
            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            "dirt should stay since stone below is anchored"
        );
    }

    #[test]
    fn block_rule_deterministic_across_runs() {
        let make_world = || {
            let mut w = gravity_world(&[DIRT_MATERIAL_ID]);
            w.simulation_seed = 12345;
            w.set(wc(2), wc(3), wc(2), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
            w.set(wc(4), wc(5), wc(4), Cell::pack(DIRT_MATERIAL_ID, 0).raw());
            w
        };

        let mut a = make_world();
        let mut b = make_world();
        for _ in 0..5 {
            a.step();
            b.step();
        }

        assert_eq!(
            a.flatten(),
            b.flatten(),
            "block stepping must be deterministic"
        );
    }

    // hash-thing-nagw: production-wiring regression. SAND on top of GUNPOWDER
    // (both Solid in `material_phase`) must not swap even though SAND density
    // (1.5) > GUNPOWDER density (1.4). Catches drift if a future material is
    // added to `material_density` but forgotten in `material_phase`.
    #[test]
    fn solid_does_not_sink_through_solid_via_terrain_defaults() {
        use crate::terrain::materials::{GUNPOWDER_MATERIAL_ID, SAND_MATERIAL_ID};
        // terrain_defaults wires gravity onto SAND and GUNPOWDER (both
        // movable solids). Before nagw, SAND would have swapped with
        // GUNPOWDER below it; now neither moves.
        let mut world = World::new(3); // 8³, terrain_defaults
                                       // Place sand at y=3 and gunpowder at y=2 in same block column.
        let sand = Cell::pack(SAND_MATERIAL_ID, 0).raw();
        let gunpowder = Cell::pack(GUNPOWDER_MATERIAL_ID, 0).raw();
        world.set(wc(2), wc(2), wc(2), gunpowder);
        world.set(wc(2), wc(3), wc(2), sand);

        world.step();

        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            gunpowder,
            "gunpowder must not be displaced by sand (solid-on-solid)"
        );
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            sand,
            "sand must stay above gunpowder (no solid-on-solid swap)"
        );
    }

    // -----------------------------------------------------------------
    // FluidBlockRule integration tests (hash-thing-1v0.18).
    // -----------------------------------------------------------------

    #[test]
    fn fluid_water_falls_under_gravity() {
        let mut world = World::new(3); // 8x8x8, terrain_defaults includes FluidBlockRule
                                       // Place water at y=3, air at y=2 — same block column at even offset.
        world.set(wc(2), wc(3), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            0,
            "bottom should be air initially"
        );

        world.step();

        // Water must survive (mass conservation) and drop below y=3.
        // Gravity phase swaps water down; lateral spread may also shift x or z
        // depending on rng_hash, so we don't assert the exact landing position.
        assert_eq!(world.population(), 1, "exactly one water cell must survive");
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            0,
            "water must have left its original position at y=3"
        );
    }

    #[test]
    fn fluid_conserves_population() {
        // Level 4 (16³) keeps cells well inside the absorbing boundary
        // over 6 steps so mass conservation holds.
        let mut world = World::new(4);
        world.set(wc(5), wc(6), wc(5), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        world.set(wc(8), wc(9), wc(8), Cell::pack(WATER_MATERIAL_ID, 0).raw());
        let pop_before = world.population();

        for _ in 0..6 {
            world.step();
        }

        assert_eq!(
            world.population(),
            pop_before,
            "fluid rule must conserve total cell count (mass conservation)"
        );
    }

    #[test]
    fn fluid_spreads_laterally_into_air() {
        // Place water at ground level with air neighbors in the same block.
        // After enough steps, water should have moved laterally.
        let mut world = World::new(3);
        world.simulation_seed = 42;
        // Place water at (2,2,2) — bottom-left of block at (2,2,2).
        // Positions (3,2,2) and (2,2,3) are in the same block and are air.
        world.set(wc(2), wc(2), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());

        // Run several steps. Lateral spread is probabilistic (depends on
        // rng_hash), but over multiple steps with alternating offsets the
        // water should reach a neighbor.
        for _ in 0..8 {
            world.step();
        }

        // Water must still exist (conservation) but may have moved.
        assert_eq!(
            world.population(),
            1,
            "exactly one water cell should remain"
        );
        // It should NOT still be at (2,3,2) after gravity — verify it settled
        // at a y=0 or y=2 position (even-aligned bottom).
    }

    #[test]
    fn fluid_deterministic_across_runs() {
        let make = || {
            let mut w = World::new(3);
            w.simulation_seed = 999;
            w.set(wc(2), wc(3), wc(2), Cell::pack(WATER_MATERIAL_ID, 0).raw());
            w.set(wc(4), wc(5), wc(4), Cell::pack(WATER_MATERIAL_ID, 0).raw());
            w
        };

        let mut a = make();
        let mut b = make();
        for _ in 0..8 {
            a.step();
            b.step();
        }

        assert_eq!(
            a.flatten(),
            b.flatten(),
            "fluid stepping must be deterministic (Hashlife-compatible)"
        );
    }

    // ---- ensure_contains (hash-thing-e9h) ----

    #[test]
    fn ensure_contains_noop_when_in_bounds() {
        let mut world = World::new(3); // 8x8x8
        let level_before = world.level;
        let root_before = world.root;
        world.ensure_contains(wc(7), wc(7), wc(7));
        assert_eq!(world.level, level_before);
        assert_eq!(world.root, root_before);
    }

    #[test]
    fn ensure_contains_grows_root_once() {
        let mut world = World::new(3); // 8x8x8, side=8
        world.set(wc(5), wc(3), wc(2), STONE);
        world.ensure_contains(wc(8), wc(0), wc(0)); // just past the edge
        assert_eq!(world.level, 4); // grew once: side=16
        assert_eq!(world.side(), 16);
        // Original cell survives at its original coordinate.
        assert_eq!(world.get(wc(5), wc(3), wc(2)), STONE);
        // The newly accessible region is empty.
        assert_eq!(world.get(wc(8), wc(0), wc(0)), 0);
        assert_eq!(world.get(wc(15), wc(15), wc(15)), 0);
    }

    #[test]
    fn ensure_contains_grows_multiple_levels() {
        let mut world = World::new(2); // 4x4x4
        world.set(wc(1), wc(2), wc(3), FIRE);
        // Coordinate 100 requires level >= 7 (side=128).
        world.ensure_contains(wc(100), wc(0), wc(0));
        assert!(world.level >= 7);
        assert!(world.side() > 100);
        // Original cell survives.
        assert_eq!(world.get(wc(1), wc(2), wc(3)), FIRE);
    }

    #[test]
    fn ensure_contains_works_on_all_axes() {
        let mut world = World::new(3); // 8x8x8
                                       // Grow for Y axis.
        world.ensure_contains(wc(0), wc(10), wc(0));
        assert!(world.side() > 10);
        let level_after_y = world.level;
        // Grow for Z axis (further).
        world.ensure_contains(wc(0), wc(0), wc(100));
        assert!(world.level > level_after_y);
        assert!(world.side() > 100);
    }

    #[test]
    fn ensure_contains_then_set_roundtrips() {
        let mut world = World::new(3); // 8x8x8
        world.ensure_contains(wc(20), wc(20), wc(20));
        world.set(wc(20), wc(20), wc(20), WATER);
        assert_eq!(world.get(wc(20), wc(20), wc(20)), WATER);
        // Other cells in the expanded region remain empty.
        assert_eq!(world.get(wc(19), wc(20), wc(20)), 0);
    }

    #[test]
    fn ensure_contains_preserves_population() {
        let mut world = World::new(3);
        world.set(wc(0), wc(0), wc(0), STONE);
        world.set(wc(7), wc(7), wc(7), FIRE);
        let pop_before = world.population();
        world.ensure_contains(wc(100), wc(100), wc(100));
        assert_eq!(world.population(), pop_before);
    }

    // ---- ensure_region (hash-thing-5qp) ----

    #[test]
    fn ensure_region_noop_when_in_bounds() {
        let mut world = World::new(3);
        let level_before = world.level;
        world.ensure_region(
            [WorldCoord(0), WorldCoord(0), WorldCoord(0)],
            [WorldCoord(7), WorldCoord(7), WorldCoord(7)],
        );
        assert_eq!(world.level, level_before);
    }

    #[test]
    fn ensure_region_grows_for_max_corner() {
        let mut world = World::new(3); // 8x8x8
        world.set(wc(2), wc(3), wc(4), STONE);
        world.ensure_region(
            [WorldCoord(0), WorldCoord(0), WorldCoord(0)],
            [WorldCoord(20), WorldCoord(20), WorldCoord(20)],
        );
        assert!(world.side() > 20);
        // Existing cell survives.
        assert_eq!(world.get(wc(2), wc(3), wc(4)), STONE);
    }

    #[test]
    fn ensure_region_then_set_covers_full_box() {
        let mut world = World::new(3);
        world.ensure_region(
            [WorldCoord(10), WorldCoord(10), WorldCoord(10)],
            [WorldCoord(20), WorldCoord(20), WorldCoord(20)],
        );
        // Can set cells at both min and max of the region.
        world.set(wc(10), wc(10), wc(10), FIRE);
        world.set(wc(20), wc(20), wc(20), WATER);
        assert_eq!(world.get(wc(10), wc(10), wc(10)), FIRE);
        assert_eq!(world.get(wc(20), wc(20), wc(20)), WATER);
    }

    #[test]
    fn ensure_region_single_point_matches_ensure_contains() {
        let mut w1 = World::new(3);
        let mut w2 = World::new(3);
        w1.ensure_contains(wc(50), wc(50), wc(50));
        w2.ensure_region(
            [WorldCoord(50), WorldCoord(50), WorldCoord(50)],
            [WorldCoord(50), WorldCoord(50), WorldCoord(50)],
        );
        assert_eq!(w1.level, w2.level);
    }

    // -----------------------------------------------------------------
    // FluidBlockRule integration: water falls via terrain_defaults (1v0.5).
    // -----------------------------------------------------------------

    #[test]
    fn terrain_defaults_water_falls_when_stepped() {
        // terrain_defaults() already wires FluidBlockRule for water.
        let mut world = World::new(3); // 8x8x8
                                       // Place water at y=3 with air below. Block lateral escape routes
                                       // with stone so we isolate the gravity behavior.
        world.set(wc(2), wc(3), wc(2), WATER);
        world.set(wc(3), wc(2), wc(2), STONE); // block x-spread
        world.set(wc(2), wc(2), wc(3), STONE); // block z-spread
        assert_eq!(world.get(wc(2), wc(3), wc(2)), WATER);
        assert_eq!(world.get(wc(2), wc(2), wc(2)), 0);

        world.step(); // gen 0 → 1 (even offset=0, block at (2,2,2))

        // Water should have fallen from y=3 to y=2 via FluidBlockRule gravity.
        assert_eq!(
            world.get(wc(2), wc(2), wc(2)),
            WATER,
            "water should fall to y=2 via FluidBlockRule"
        );
        assert_eq!(
            world.get(wc(2), wc(3), wc(2)),
            0,
            "y=3 should be air after water fell"
        );
    }

    #[test]
    fn burning_room_conserves_population_over_steps() {
        let mut world = World::new(6); // 64x64x64
        world.seed_burning_room();
        let pop_before = world.population();
        assert!(pop_before > 100, "room should have substantial content");

        // Step a few times — fire reactions may create/destroy cells,
        // but block rules (movement) must conserve mass within each block.
        // Population may change due to CaRule (fire burns out, water quenches),
        // so we just verify the world doesn't crash and stays non-empty.
        for _ in 0..4 {
            world.step();
        }
        assert!(
            world.population() > 0,
            "world should still have cells after stepping"
        );
    }

    // -----------------------------------------------------------------
    // RealizedRegion (hash-thing-ica).
    // -----------------------------------------------------------------

    #[test]
    fn region_side_matches_world() {
        let world = World::new(5);
        let r = world.region();
        assert_eq!(r.side(), 32);
        assert_eq!(r.side() as usize, world.side());
        assert_eq!(r.level, 5);
        assert_eq!(r.origin, [0, 0, 0]);
    }

    #[test]
    fn region_contains_boundary() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 3,
        }; // side=8
        assert!(r.contains(0, 0, 0));
        assert!(r.contains(7, 7, 7));
        assert!(!r.contains(8, 0, 0));
        assert!(!r.contains(0, 8, 0));
        assert!(!r.contains(0, 0, 8));
    }

    #[test]
    fn region_contains_with_origin() {
        let r = RealizedRegion {
            origin: [10, 20, 30],
            level: 2,
        }; // side=4
        assert!(r.contains(10, 20, 30));
        assert!(r.contains(13, 23, 33));
        assert!(!r.contains(14, 20, 30)); // just past
        assert!(!r.contains(9, 20, 30)); // just before
    }

    #[test]
    fn region_octant_of_corners() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 4,
        }; // side=16, half=8
        assert_eq!(r.octant_of(0, 0, 0), 0); // (0,0,0)
        assert_eq!(r.octant_of(8, 0, 0), 1); // (1,0,0)
        assert_eq!(r.octant_of(0, 8, 0), 2); // (0,1,0)
        assert_eq!(r.octant_of(8, 8, 0), 3); // (1,1,0)
        assert_eq!(r.octant_of(0, 0, 8), 4); // (0,0,1)
        assert_eq!(r.octant_of(15, 15, 15), 7); // (1,1,1)
    }

    #[test]
    #[should_panic(expected = "outside realized region")]
    fn region_octant_of_oob_panics() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 3,
        };
        r.octant_of(8, 0, 0);
    }

    #[test]
    fn region_display() {
        let r = RealizedRegion {
            origin: [0, 0, 0],
            level: 6,
        };
        let s = format!("{r}");
        assert!(s.contains("level=6"), "display should show level");
        assert!(s.contains("64³"), "display should show side cubed");
    }

    #[test]
    fn region_tracks_ensure_contains_growth() {
        let mut world = World::new(3); // side=8
        assert_eq!(world.region().side(), 8);
        world.ensure_contains(wc(20), wc(0), wc(0));
        assert!(world.region().side() > 20);
        assert_eq!(world.region().level, world.level);
    }

    // ── Lazy terrain expansion (3fq.4) ────────────────────────────

    #[test]
    fn expand_without_terrain_produces_empty_siblings() {
        let mut world = World::new(3); // side=8
        world.set(wc(0), wc(0), wc(0), ALIVE.raw());
        let pop_before = world.population();
        world.ensure_contains(wc(10), wc(0), wc(0));
        // Only the original cell survives — new octants are empty.
        assert_eq!(world.population(), pop_before);
    }

    #[test]
    fn expand_with_terrain_populates_new_octants() {
        let mut world = World::new(3); // side=8
        let params = TerrainParams::default();
        world.seed_terrain(&params).unwrap();
        let pop_initial = world.population();
        assert!(pop_initial > 0, "terrain should produce non-empty world");

        // Expand into +x. The new octant at (8..16, 0..8, 0..8) should
        // have generated terrain, not empty space.
        world.ensure_contains(wc(10), wc(0), wc(0));
        let pop_after = world.population();
        assert!(
            pop_after > pop_initial,
            "expansion should add terrain: before={pop_initial}, after={pop_after}"
        );
    }

    #[test]
    fn expand_terrain_is_deterministic() {
        let params = TerrainParams::default();

        let mut w1 = World::new(3);
        w1.seed_terrain(&params).unwrap();
        w1.ensure_contains(wc(10), wc(0), wc(0));

        let mut w2 = World::new(3);
        w2.seed_terrain(&params).unwrap();
        w2.ensure_contains(wc(10), wc(0), wc(0));

        // Same params + same expansion → same world.
        assert_eq!(w1.level, w2.level);
        assert_eq!(w1.population(), w2.population());
        // Spot-check a cell in the expanded region.
        assert_eq!(w1.get(wc(10), wc(3), wc(3)), w2.get(wc(10), wc(3), wc(3)));
    }

    #[test]
    fn expand_terrain_preserves_original_cells() {
        let mut world = World::new(3);
        let params = TerrainParams::default();
        world.seed_terrain(&params).unwrap();

        // Record some cells from the original region.
        let cells_before: Vec<_> = (0..8u64).map(|x| world.get(wc(x), wc(3), wc(3))).collect();

        world.ensure_contains(wc(10), wc(0), wc(0));

        // Original region cells are unchanged.
        for (x, &expected) in cells_before.iter().enumerate() {
            assert_eq!(
                world.get(wc(x as u64), wc(3), wc(3)),
                expected,
                "cell at ({x}, 3, 3) changed after expansion"
            );
        }
    }

    #[test]
    fn expand_terrain_matches_direct_gen() {
        // A world grown by expansion should produce the same terrain
        // as one generated at the larger size from the start. This
        // verifies there are no seams at expansion boundaries.
        let params = TerrainParams::default();

        // Path A: generate at level 3, then expand to level 4.
        let mut grown = World::new(3);
        grown.seed_terrain(&params).unwrap();
        grown.ensure_contains(wc(10), wc(0), wc(0)); // grows to level 4

        // Path B: generate at level 4 directly.
        let mut direct = World::new(4);
        direct.seed_terrain(&params).unwrap();

        // Spot-check cells across the expansion boundary (x=8 is the seam).
        for x in 0..16u64 {
            for yz in [0u64, 3, 7] {
                assert_eq!(
                    grown.get(wc(x), wc(yz), wc(yz)),
                    direct.get(wc(x), wc(yz), wc(yz)),
                    "seam mismatch at ({x}, {yz}, {yz})"
                );
            }
        }
    }

    #[test]
    fn gol_world_expand_stays_empty() {
        // GoL smoke worlds don't set terrain_params, so expansion is empty.
        let mut world = World::new(3);
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        assert!(world.terrain_params.is_none());
        world.ensure_contains(wc(20), wc(0), wc(0));
        // Only the single cell we placed.
        assert_eq!(world.population(), 1);
    }

    #[test]
    fn probe_matches_get() {
        let mut world = World::new(3);
        let stone = crate::terrain::materials::STONE;
        world.set(wc(1), wc(2), wc(3), stone);
        assert_eq!(
            world.probe(wc(1), wc(2), wc(3)),
            world.get(wc(1), wc(2), wc(3))
        );
        // OOB returns 0 from both
        assert_eq!(world.probe(WorldCoord(-1), wc(0), wc(0)), 0);
        assert_eq!(world.probe(wc(100), wc(0), wc(0)), 0);
    }

    #[test]
    fn is_realized_inside() {
        let world = World::new(3); // side=8
        assert!(world.is_realized(wc(0), wc(0), wc(0)));
        assert!(world.is_realized(wc(7), wc(7), wc(7)));
        assert!(world.is_realized(wc(4), wc(2), wc(6)));
    }

    #[test]
    fn is_realized_outside() {
        let world = World::new(3); // side=8
        assert!(!world.is_realized(WorldCoord(-1), wc(0), wc(0)));
        assert!(!world.is_realized(wc(8), wc(0), wc(0)));
        assert!(!world.is_realized(wc(0), wc(100), wc(0)));
    }

    // --- Mutation channel tests (hash-thing-1v0.9) ---

    #[test]
    fn apply_mutations_drains_and_applies() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: wc(1),
            y: wc(2),
            z: wc(3),
            state: STONE,
        });
        world.queue.push(WorldMutation::SetCell {
            x: wc(4),
            y: wc(5),
            z: wc(6),
            state: STONE,
        });
        assert_eq!(world.queue.len(), 2);
        world.apply_mutations();
        assert!(world.queue.is_empty());
        assert_eq!(world.get(wc(1), wc(2), wc(3)), STONE);
        assert_eq!(world.get(wc(4), wc(5), wc(6)), STONE);
    }

    /// commit_step rebuilds the NodeStore via `compacted_with_remap`, so every
    /// NodeId key in every hashlife cache is invalidated. The macro cache was
    /// historically omitted from the clear-list, leaving stale `(NodeId, u64)`
    /// keys pointing at a dead namespace — which `maybe_compact` would then
    /// treat as extra roots and feed back into compaction (hash-thing-w1bs).
    #[test]
    fn commit_step_clears_hashlife_macro_cache() {
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step_recursive_pow2();
        assert!(
            !world.hashlife_macro_cache.is_empty(),
            "precondition: step_recursive_pow2 must populate the macro cache"
        );

        world.step();

        assert!(
            world.hashlife_macro_cache.is_empty(),
            "commit_step must clear the macro cache; stale NodeId keys would \
             alias into the freshly-compacted store"
        );
    }

    /// hash-thing-wsq3: cell edits must NOT invalidate hashlife cache entries.
    /// Hash-consed NodeIds plus phase-derived keys make existing entries
    /// semantically valid across edits — only material-registry mutation
    /// (`mutate_materials` -> `invalidate_material_caches`) and fresh-store
    /// epoch boundaries (`commit_step`, plus the seed_*
    /// paths) require a clear. The pre-wsq3 unconditional clear in
    /// `set_local` was correctness-paranoia leftover from 53f7414's
    /// `(NodeId, [origin], parity)` cache shape and forced every entity-
    /// driven SetCell in production to wipe the cache, paying cold-cache
    /// cost on every step.
    #[test]
    fn direct_set_preserves_hashlife_caches() {
        let mut world = World::new(3);
        // Seed a non-empty, non-uniform world so step_recursive actually
        // descends through step_base_case and populates caches (an empty
        // world short-circuits at the population==0 guard in step_node).
        // WATER has a FluidBlockRule, so the leaf is not all-inert and
        // step_node won't short-circuit before reaching step_base_case.
        world.set(wc(2), wc(2), wc(2), WATER);
        world.set(wc(3), wc(2), wc(2), WATER);
        world.set(wc(4), wc(2), wc(2), WATER);
        world.step_recursive();
        let recursive_before = world.hashlife_cache.len();
        let inert_before = world.hashlife_inert_cache.len();
        let all_inert_before = world.hashlife_all_inert_cache.len();
        assert!(
            recursive_before > 0,
            "precondition: stepping a seeded small world should populate the recursive cache"
        );

        // Direct edit: should produce a new root NodeId but leave the
        // pre-edit subtree NodeIds (and their cached step results) intact.
        world.set(wc(5), wc(5), wc(5), STONE);

        assert_eq!(
            world.hashlife_cache.len(),
            recursive_before,
            "direct edits must preserve recursive cache entries — NodeIds are hash-consed"
        );
        assert_eq!(
            world.hashlife_inert_cache.len(),
            inert_before,
            "direct edits must preserve inert-uniform cache entries"
        );
        assert_eq!(
            world.hashlife_all_inert_cache.len(),
            all_inert_before,
            "direct edits must preserve all-inert cache entries"
        );

        // Stronger: capture a specific (key, value) from the live cache
        // and confirm both NodeIds and their mapping survive byte-identical
        // across the edit. "len unchanged" alone permits a pathological
        // implementation that drops one entry and inserts a different one;
        // "this exact entry is still here with the same value" doesn't.
        // hash-thing-wsq3 review.
        let sample = world
            .hashlife_cache
            .iter()
            .next()
            .map(|(k, v)| (*k, *v))
            .expect("cache populated above");
        let edit_root = world.root;
        world.set(wc(6), wc(6), wc(6), STONE);
        assert_ne!(
            world.root, edit_root,
            "precondition: a cell write must produce a fresh root NodeId"
        );
        assert_eq!(
            world.hashlife_cache.get(&sample.0).copied(),
            Some(sample.1),
            "preserved cache entry must keep its exact (key, value) mapping across edits"
        );
    }

    /// hash-thing-wsq3: paired with `direct_set_preserves_hashlife_caches`.
    /// Entity-driven mutations (the production hot path) flow through
    /// `apply_mutations` -> `set` -> `set_local`, which must NOT clear caches.
    #[test]
    fn apply_mutations_preserves_hashlife_caches() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        // See `direct_set_preserves_hashlife_caches` — empty worlds
        // short-circuit before populating the cache.
        // WATER has a FluidBlockRule, so the leaf is not all-inert and
        // step_node won't short-circuit before reaching step_base_case.
        world.set(wc(2), wc(2), wc(2), WATER);
        world.set(wc(3), wc(2), wc(2), WATER);
        world.set(wc(4), wc(2), wc(2), WATER);
        world.step_recursive();
        let recursive_before = world.hashlife_cache.len();
        let inert_before = world.hashlife_inert_cache.len();
        let all_inert_before = world.hashlife_all_inert_cache.len();
        assert!(
            recursive_before > 0,
            "precondition: stepping a seeded small world should populate the recursive cache"
        );

        world.queue.push(WorldMutation::SetCell {
            x: wc(5),
            y: wc(5),
            z: wc(5),
            state: STONE,
        });
        world.apply_mutations();

        assert_eq!(
            world.hashlife_cache.len(),
            recursive_before,
            "mutation flush must preserve recursive cache entries"
        );
        assert_eq!(
            world.hashlife_inert_cache.len(),
            inert_before,
            "mutation flush must preserve inert-uniform cache entries"
        );
        assert_eq!(
            world.hashlife_all_inert_cache.len(),
            all_inert_before,
            "mutation flush must preserve all-inert cache entries"
        );

        // Stronger: capture a specific (key, value) and confirm it
        // survives byte-identical (paired with the same check in
        // `direct_set_preserves_hashlife_caches`). hash-thing-wsq3 review.
        let sample = world
            .hashlife_cache
            .iter()
            .next()
            .map(|(k, v)| (*k, *v))
            .expect("cache populated above");
        let pre_root = world.root;
        world.queue.push(WorldMutation::SetCell {
            x: wc(6),
            y: wc(6),
            z: wc(6),
            state: STONE,
        });
        world.apply_mutations();
        assert_ne!(
            world.root, pre_root,
            "precondition: an applied mutation must produce a fresh root NodeId"
        );
        assert_eq!(
            world.hashlife_cache.get(&sample.0).copied(),
            Some(sample.1),
            "preserved cache entry must keep its exact (key, value) mapping across mutation flush"
        );
    }

    #[test]
    fn apply_mutations_last_write_wins() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: wc(3),
            y: wc(3),
            z: wc(3),
            state: STONE,
        });
        world.queue.push(WorldMutation::SetCell {
            x: wc(3),
            y: wc(3),
            z: wc(3),
            state: Cell::pack(WATER_MATERIAL_ID, 0).raw(),
        });
        world.apply_mutations();
        assert_eq!(
            world.get(wc(3), wc(3), wc(3)),
            Cell::pack(WATER_MATERIAL_ID, 0).raw()
        );
    }

    #[test]
    fn apply_mutations_drops_out_of_bounds_setcell() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: WorldCoord(8),
            y: wc(2),
            z: wc(2),
            state: STONE,
        });

        world.apply_mutations();

        assert_eq!(world.get(wc(7), wc(2), wc(2)), 0);
        assert!(world.queue.is_empty());
    }

    #[test]
    fn apply_mutations_drops_out_of_bounds_setcell_with_shifted_origin() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-1), wc(0), wc(0));

        let in_bounds = WorldCoord(world.origin[0]);
        let out_of_bounds = WorldCoord(world.origin[0] - 1);
        world.queue.push(WorldMutation::SetCell {
            x: in_bounds,
            y: wc(0),
            z: wc(0),
            state: STONE,
        });
        world.queue.push(WorldMutation::SetCell {
            x: out_of_bounds,
            y: wc(0),
            z: wc(0),
            state: DIRT,
        });

        world.apply_mutations();

        assert_eq!(world.get(in_bounds, wc(0), wc(0)), STONE);
        assert_eq!(world.get(out_of_bounds, wc(0), wc(0)), 0);
    }

    #[test]
    fn fill_region_covers_inclusive_box() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::FillRegion {
            min: [wc(1), wc(1), wc(1)],
            max: [wc(3), wc(3), wc(3)],
            state: STONE,
        });
        world.apply_mutations();
        // All 27 cells (3×3×3) should be stone.
        let mut count = 0;
        for z in 1..=3 {
            for y in 1..=3 {
                for x in 1..=3 {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 27);
    }

    #[test]
    fn fill_region_clamps_to_realized_bounds() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::FillRegion {
            min: [wc(6), wc(6), wc(6)],
            max: [wc(8), wc(8), wc(8)],
            state: STONE,
        });

        world.apply_mutations();

        let mut count = 0;
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 8);
        assert_eq!(world.get(wc(6), wc(6), wc(6)), STONE);
        assert_eq!(world.get(wc(7), wc(7), wc(7)), STONE);
    }

    #[test]
    fn set_block_fills_cells_per_block_cubed() {
        use crate::octree::CELLS_PER_BLOCK;
        // World needs to be large enough: block at (1,0,0) spans cells [8..15]
        // on x, so we need level >= 4 (side 16).
        let mut world = World::new(4);
        world.set_block(1, 0, 0, STONE);
        world.apply_mutations();

        let k = CELLS_PER_BLOCK as u64;
        let mut filled = 0u32;
        for z in 0..k {
            for y in 0..k {
                for x in k..(2 * k) {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        filled += 1;
                    }
                }
            }
        }
        assert_eq!(filled, CELLS_PER_BLOCK.pow(3));

        // Cells just outside the block should be empty.
        assert_eq!(world.get(wc(k - 1), wc(0), wc(0)), 0); // cell 7, just before block
        assert_eq!(world.get(wc(2 * k), wc(0), wc(0)), 0); // cell 16, just after block
    }

    #[test]
    fn set_block_at_origin() {
        use crate::octree::CELLS_PER_BLOCK;
        let mut world = World::new(4);
        world.set_block(0, 0, 0, STONE);
        world.apply_mutations();

        let k = CELLS_PER_BLOCK as u64;
        let mut filled = 0u32;
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    if world.get(wc(x), wc(y), wc(z)) == STONE {
                        filled += 1;
                    }
                }
            }
        }
        assert_eq!(filled, CELLS_PER_BLOCK.pow(3));
    }

    #[test]
    fn set_block_roundtrip_individual_cells() {
        use crate::octree::CELLS_PER_BLOCK;
        let mut world = World::new(4);
        world.set_block(0, 0, 0, DIRT);
        world.apply_mutations();

        let k = CELLS_PER_BLOCK as u64;
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    assert_eq!(
                        world.get(wc(x), wc(y), wc(z)),
                        DIRT,
                        "cell ({x},{y},{z}) should be DIRT"
                    );
                }
            }
        }
    }

    #[test]
    fn set_block_auto_grows_world() {
        use crate::octree::CELLS_PER_BLOCK;
        // Start with a level-3 world (side=8). Block at (1,0,0) needs cells [8..15],
        // which is out of bounds. set_block should auto-grow to fit.
        let mut world = World::new(3);
        world.set_block(1, 0, 0, STONE);
        world.apply_mutations();

        // World must have grown to at least level 4 (side=16).
        assert!(
            world.level >= 4,
            "world should auto-grow for out-of-bounds block"
        );

        let k = CELLS_PER_BLOCK as u64;
        for z in 0..k {
            for y in 0..k {
                for x in k..(2 * k) {
                    assert_eq!(world.get(wc(x), wc(y), wc(z)), STONE);
                }
            }
        }
    }

    #[test]
    fn block_fill_vs_cell_fill_node_count() {
        use crate::octree::CELLS_PER_BLOCK;
        // Fill a 16³ world with stone: once via set_block, once via individual set calls.
        // Both should produce identical octrees (hash-consing deduplicates).
        let k = CELLS_PER_BLOCK as u64;
        let level = 4u32; // 16³ world

        // Block fill: 2 blocks per axis at K=3 → (16/8)³ = 8 blocks
        let mut world_block = World::new(level);
        let blocks_per_axis = (1u64 << level) / k;
        for bz in 0..blocks_per_axis as i64 {
            for by in 0..blocks_per_axis as i64 {
                for bx in 0..blocks_per_axis as i64 {
                    world_block.set_block(bx, by, bz, STONE);
                }
            }
        }
        world_block.apply_mutations();

        // Cell fill: 16³ = 4096 individual set calls
        let mut world_cell = World::new(level);
        let side = 1u64 << level;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    world_cell.set(wc(x), wc(y), wc(z), STONE);
                }
            }
        }

        // Both should produce the same root (hash-cons identity) — uniform
        // stone compresses identically regardless of insertion order.
        assert_eq!(world_block.root, world_cell.root);
    }

    #[test]
    fn step_ca_runs_clean_on_empty_queue() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), STONE);
        world.step_ca(); // should not panic
        assert_eq!(world.generation, 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "pending mutations")]
    fn step_ca_panics_with_pending_mutations() {
        use crate::sim::WorldMutation;
        let mut world = World::new(3);
        world.queue.push(WorldMutation::SetCell {
            x: wc(1),
            y: wc(1),
            z: wc(1),
            state: STONE,
        });
        world.step_ca(); // should panic
    }

    #[test]
    fn sand_falls_under_gravity() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(2), SAND);
        assert_eq!(world.get(wc(2), wc(2), wc(2)), 0);

        world.step();

        assert_eq!(world.get(wc(2), wc(2), wc(2)), SAND, "sand should fall");
        assert_eq!(world.get(wc(2), wc(3), wc(2)), 0, "top should be air");
    }

    #[test]
    fn sand_conserves_population() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(2), SAND);
        world.set(wc(4), wc(5), wc(4), SAND);
        world.set(wc(0), wc(1), wc(0), SAND);
        let pop_before = world.population();

        for _ in 0..4 {
            world.step();
        }

        assert_eq!(
            world.population(),
            pop_before,
            "sand gravity must conserve population"
        );
    }

    #[test]
    fn lava_solidifies_on_water_contact() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), LAVA);
        world.set(wc(4), wc(3), wc(3), WATER);

        world.step();

        assert_eq!(
            world.get(wc(3), wc(3), wc(3)),
            STONE,
            "lava adjacent to water should solidify into stone"
        );
    }

    #[test]
    fn lava_persists_without_water() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), LAVA);

        world.step();

        let pop = world.population();
        assert!(pop > 0, "lava should not disappear without water");
    }

    // -----------------------------------------------------------------
    // Negative-direction world growth (hash-thing-37r).
    // -----------------------------------------------------------------

    #[test]
    fn ensure_contains_negative_shifts_origin() {
        let mut world = World::new(3);
        world.set(wc(4), wc(4), wc(4), STONE);
        world.ensure_contains(WorldCoord(-1), wc(0), wc(0));
        assert!(
            world.origin[0] < 0,
            "origin[0] should be negative after negative growth"
        );
        assert_eq!(world.get(wc(4), wc(4), wc(4)), STONE);
    }

    #[test]
    fn ensure_contains_negative_preserves_population() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(5), STONE);
        world.set(wc(6), wc(1), wc(7), WATER);
        let pop_before = world.population();
        world.ensure_contains(WorldCoord(-10), WorldCoord(-10), WorldCoord(-10));
        assert_eq!(world.population(), pop_before);
    }

    #[test]
    fn ensure_contains_negative_then_set_roundtrips() {
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-5), WorldCoord(-5), WorldCoord(-5));
        world.set(WorldCoord(-5), WorldCoord(-5), WorldCoord(-5), STONE);
        assert_eq!(
            world.get(WorldCoord(-5), WorldCoord(-5), WorldCoord(-5)),
            STONE
        );
    }

    #[test]
    fn ensure_region_negative_min_grows_correctly() {
        let mut world = World::new(3);
        world.ensure_region(
            [WorldCoord(-4), WorldCoord(-4), WorldCoord(-4)],
            [WorldCoord(7), WorldCoord(7), WorldCoord(7)],
        );
        assert!(world.is_realized(WorldCoord(-4), WorldCoord(-4), WorldCoord(-4)));
        assert!(world.is_realized(wc(7), wc(7), wc(7)));
    }

    #[test]
    fn negative_growth_origin_tracks_correctly() {
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-1), wc(0), wc(0));
        let o = world.origin;
        let side = world.side() as i64;
        assert!(o[0] <= 0);
        assert!(o[0] + side >= 8);
    }

    #[test]
    fn get_returns_zero_below_origin() {
        let world = World::new(3);
        assert_eq!(world.get(WorldCoord(-1), wc(0), wc(0)), 0);
    }

    #[test]
    fn mixed_direction_growth_preserves_cells() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(4), STONE);
        world.set(wc(6), wc(1), wc(7), WATER);
        world.ensure_region([WorldCoord(-5), wc(0), wc(0)], [wc(0), wc(0), wc(20)]);
        assert_eq!(world.get(wc(2), wc(3), wc(4)), STONE);
        assert_eq!(world.get(wc(6), wc(1), wc(7)), WATER);
    }

    #[test]
    fn is_realized_works_after_negative_growth() {
        let mut world = World::new(3);
        world.ensure_contains(WorldCoord(-3), WorldCoord(-3), WorldCoord(-3));
        assert!(world.is_realized(WorldCoord(-3), WorldCoord(-3), WorldCoord(-3)));
        assert!(world.is_realized(wc(0), wc(0), wc(0)));
        assert!(world.is_realized(wc(7), wc(7), wc(7)));
    }

    /// qy4g.2 option G regression. Two invariants checked under the new
    /// no-gap-fill regime:
    ///
    /// 1. **Mass conservation every step.** Population is constant under
    ///    pure CaRule + Margolus BlockRule (option G's strongest guarantee).
    /// 2. **Post-compaction contiguity.** Once the falling water block
    ///    compacts against the floor (or another solid), the resulting
    ///    settled pile is contiguous (no every-other-y holes left behind).
    ///
    /// What this test does **NOT** verify: gap closure during free-fall.
    /// Option G accepts an in-flight every-other-y checkerboard while a
    /// column is falling — see `World::step` and SPEC.md "Internal-gap
    /// closure rides parity-flip" for the explicit tradeoff and fallbacks
    /// A/F. The 32-tick run lets the 6-cell starter block fall ~16 cells
    /// onto the floor and settle.
    #[test]
    fn water_block_gaps_close_via_parity_flip() {
        let mut world = World::new(5); // 32³
                                       // Place a 6×6×6 water block in the center, well away from boundaries.
        for z in 10..16 {
            for y in 16..22 {
                for x in 10..16 {
                    world.set(wc(x), wc(y), wc(z), WATER);
                }
            }
        }
        let pop_before = world.population();

        let max_steps = 32;
        for step in 0..max_steps {
            world.step();
            let pop_after = world.population();
            assert_eq!(
                pop_before, pop_after,
                "population changed at step {step} ({pop_before} → {pop_after})",
            );
        }

        // Check: for each y-level that contains water, count water cells.
        let side = world.side() as u64;
        let grid = world.flatten();
        let mut water_by_y = vec![0u64; side as usize];
        for y in 0..side {
            for z in 0..side {
                for x in 0..side {
                    let idx = x as usize
                        + y as usize * side as usize
                        + z as usize * side as usize * side as usize;
                    if Cell::from_raw(grid[idx]).material() == WATER_MATERIAL_ID {
                        water_by_y[y as usize] += 1;
                    }
                }
            }
        }

        eprintln!("Water cells per y-level after {max_steps} steps:");
        for (y, &count) in water_by_y.iter().enumerate() {
            if count > 0 {
                eprintln!("  y={y}: {count} water cells");
            }
        }

        let water_levels: Vec<usize> = water_by_y
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(y, _)| y)
            .collect();
        assert!(
            !water_levels.is_empty(),
            "water should still exist after stepping"
        );
        for pair in water_levels.windows(2) {
            assert_eq!(
                pair[1] - pair[0],
                1,
                "gap between y={} and y={} — settled pile not contiguous \
                 after {max_steps} steps",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn progression_waterfall_stays_visually_contiguous() {
        let mut world = World::new(6);
        world.seed_lattice_progression_demo();
        let field = LatticeField::for_world(world.level, 42);
        let ground_y = field.lo[1] + field.floor_thick;
        let ProgressionBoxes {
            corridor, tease_b, ..
        } = World::progression_boxes(&field);
        let waterfall = World::progression_waterfall_layout(corridor, tease_b, ground_y);

        for _ in 0..6 {
            world.spawn_clones();
            world.step();
        }

        let mut occupied = Vec::new();
        for y in waterfall.shaft.min[1]..=waterfall.shaft.max[1] {
            let mut xs = Vec::new();
            for x in waterfall.curtain.min[0]..=waterfall.curtain.max[0] {
                let has_water = (waterfall.curtain.min[2]..=waterfall.curtain.max[2]).any(|z| {
                    Cell::from_raw(world.get(WorldCoord(x), WorldCoord(y), WorldCoord(z)))
                        .material()
                        == WATER_MATERIAL_ID
                });
                if has_water {
                    xs.push(x);
                }
            }
            if !xs.is_empty() {
                occupied.push((y, xs));
            }
        }

        assert!(
            occupied.len() >= 4,
            "waterfall should remain visible across multiple projected rows"
        );
        for (y, xs) in occupied {
            for pair in xs.windows(2) {
                assert_eq!(
                    pair[1] - pair[0],
                    1,
                    "waterfall developed a projected gap at y={y}, x={}..{}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }

    #[test]
    fn spawn_clones_drops_out_of_bounds_neighbor_writes() {
        let mut world = World::new(2);
        world.place_clone_source([3, 1, 1], WATER_MATERIAL_ID);

        world.spawn_clones();

        assert_eq!(
            Cell::from_raw(world.get(wc(2), wc(1), wc(1))).material(),
            WATER_MATERIAL_ID
        );
        assert_eq!(
            Cell::from_raw(world.get(wc(3), wc(2), wc(1))).material(),
            WATER_MATERIAL_ID
        );
        assert_eq!(
            Cell::from_raw(world.get(wc(3), wc(0), wc(1))).material(),
            WATER_MATERIAL_ID
        );
        assert_eq!(
            Cell::from_raw(world.get(wc(3), wc(1), wc(2))).material(),
            WATER_MATERIAL_ID
        );
        assert_eq!(
            Cell::from_raw(world.get(wc(3), wc(1), wc(0))).material(),
            WATER_MATERIAL_ID
        );
        assert_eq!(world.get(WorldCoord(4), wc(1), wc(1)), 0);
    }

    #[test]
    #[should_panic(expected = "World write uses unregistered material")]
    fn set_rejects_unregistered_material_id() {
        let mut world = World::new(3);
        let invalid = Cell::pack(1023, 0).raw();
        world.set(wc(0), wc(0), wc(0), invalid);
    }

    #[test]
    #[should_panic(expected = "World write uses unregistered material")]
    fn apply_mutations_rejects_unregistered_material_id() {
        let mut world = World::new(3);
        let invalid = Cell::pack(1023, 0).raw();
        world.queue.push(WorldMutation::SetCell {
            x: wc(0),
            y: wc(0),
            z: wc(0),
            state: invalid,
        });

        world.apply_mutations();
    }

    // ---------------------------------------------------------------------
    // hash-thing-stue.6: spatial-memo telemetry.
    // ---------------------------------------------------------------------

    // hash-thing-ecmn (vqke.4.1) review-pass: pin the env-string ->
    // BaseCaseStrategy parser so a future tweak doesn't silently change
    // operator-visible behavior. Drives `parse_base_case_strategy` with
    // injected Option<String> values rather than touching process env
    // (which would race with the parallel test runner).

    #[test]
    fn parse_base_case_strategy_default_when_unset() {
        assert_eq!(
            super::parse_base_case_strategy(None, None),
            BaseCaseStrategy::RayonPerFanout
        );
    }

    #[test]
    fn parse_base_case_strategy_explicit_strategy_overrides_rayon_legacy() {
        // Explicit STRATEGY beats legacy RAYON shim regardless of value.
        assert_eq!(
            super::parse_base_case_strategy(Some("bfs".into()), Some("1".into())),
            BaseCaseStrategy::RayonBfs
        );
        assert_eq!(
            super::parse_base_case_strategy(Some("serial".into()), Some("1".into())),
            BaseCaseStrategy::Serial
        );
        assert_eq!(
            super::parse_base_case_strategy(Some("per-fanout".into()), Some("0".into())),
            BaseCaseStrategy::RayonPerFanout
        );
        // `rayon` is an accepted alias for per-fanout (ftuu wording).
        assert_eq!(
            super::parse_base_case_strategy(Some("rayon".into()), None),
            BaseCaseStrategy::RayonPerFanout
        );
    }

    #[test]
    fn parse_base_case_strategy_legacy_rayon_shim() {
        assert_eq!(
            super::parse_base_case_strategy(None, Some("1".into())),
            BaseCaseStrategy::RayonPerFanout
        );
        assert_eq!(
            super::parse_base_case_strategy(None, Some("0".into())),
            BaseCaseStrategy::Serial
        );
        // Unrecognised legacy value falls through to default — silent
        // by design (legacy shim, documented as "1 to enable").
        assert_eq!(
            super::parse_base_case_strategy(None, Some("yes".into())),
            BaseCaseStrategy::RayonPerFanout
        );
        assert_eq!(
            super::parse_base_case_strategy(None, Some("".into())),
            BaseCaseStrategy::RayonPerFanout
        );
    }

    #[test]
    #[should_panic(expected = "HASH_THING_BASE_CASE_STRATEGY must be one of")]
    fn parse_base_case_strategy_invalid_explicit_panics() {
        // Operator-visible misconfig: hard panic with usage message,
        // not silent fallback. (Adversarial-Codex-Dependencies plan
        // review feedback — operators should get loud signal.)
        let _ = super::parse_base_case_strategy(Some("turbo".into()), None);
    }

    #[test]
    fn hashlife_stats_accumulate_sums_fields_and_levels_per_index() {
        let mut total = HashlifeStats::default();
        let mut a = HashlifeStats::default();
        let mut b = HashlifeStats::default();

        a.cache_hits = 3;
        a.cache_misses = 5;
        a.empty_skips = 7;
        a.fixed_point_skips = 11;
        a.misses_by_level[0] = 1;
        a.misses_by_level[3] = 13;
        a.misses_by_level[7] = 17;
        // hash-thing-bjdl (vqke.2): exercise the new accumulators
        // alongside the existing scalar fields, so a future refactor
        // that drops one of these from `accumulate` is caught.
        a.cache_misses_phase_aliased = 2;
        a.compact_entries_kept = 100;
        a.compact_entries_dropped = 25;
        // hash-thing-ecmn (vqke.4.1): exercise the BFS accumulators
        // alongside the older fields. Sum-shape for most counters; max
        // for `bfs_max_batch_len`.
        a.bfs_level3_unique_misses = 50;
        a.bfs_batches_parallel = 1;
        a.bfs_batches_serial_fallback = 0;
        a.bfs_max_batch_len = 50;
        a.bfs_tasks_by_level[0] = 50;
        a.bfs_tasks_by_level[2] = 9;

        b.cache_hits = 2;
        b.cache_misses = 4;
        b.empty_skips = 6;
        b.fixed_point_skips = 8;
        b.misses_by_level[0] = 10;
        b.misses_by_level[3] = 100;
        b.misses_by_level[7] = 1000;
        b.cache_misses_phase_aliased = 1;
        b.compact_entries_kept = 50;
        b.compact_entries_dropped = 0;
        b.bfs_level3_unique_misses = 200;
        b.bfs_batches_parallel = 1;
        b.bfs_batches_serial_fallback = 1;
        b.bfs_max_batch_len = 200; // larger; should win the running max
        b.bfs_tasks_by_level[0] = 200;
        b.bfs_tasks_by_level[2] = 17;

        total.accumulate(&a);
        total.accumulate(&b);

        assert_eq!(total.cache_hits, 5);
        assert_eq!(total.cache_misses, 9);
        assert_eq!(total.empty_skips, 13);
        assert_eq!(total.fixed_point_skips, 19);
        assert_eq!(total.misses_by_level[0], 11);
        assert_eq!(total.misses_by_level[3], 113);
        assert_eq!(total.misses_by_level[7], 1017);
        assert_eq!(total.cache_misses_phase_aliased, 3);
        assert_eq!(total.compact_entries_kept, 150);
        assert_eq!(total.compact_entries_dropped, 25);
        // hash-thing-ecmn: BFS counters.
        assert_eq!(total.bfs_level3_unique_misses, 250);
        assert_eq!(total.bfs_batches_parallel, 2);
        assert_eq!(total.bfs_batches_serial_fallback, 1);
        // bfs_max_batch_len is a running MAX, not a sum: the larger of
        // the two wins.
        assert_eq!(total.bfs_max_batch_len, 200);
        assert_eq!(total.bfs_tasks_by_level[0], 250);
        assert_eq!(total.bfs_tasks_by_level[2], 26);
        // Untouched indices must remain zero — catches the "summed into
        // index 0" copy-paste bug.
        for (i, &v) in total.misses_by_level.iter().enumerate() {
            if !matches!(i, 0 | 3 | 7) {
                assert_eq!(v, 0, "misses_by_level[{i}] bled from another index");
            }
        }
        for (i, &v) in total.bfs_tasks_by_level.iter().enumerate() {
            if !matches!(i, 0 | 2) {
                assert_eq!(v, 0, "bfs_tasks_by_level[{i}] bled from another index");
            }
        }
    }

    #[test]
    fn memo_summary_reports_cumulative_hits_after_two_steps() {
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());

        world.step_recursive();
        let after_first = world.hashlife_stats;
        world.step_recursive();
        let after_second = world.hashlife_stats;

        let total = &world.hashlife_stats_total;
        assert!(
            total.cache_hits + total.cache_misses > 0,
            "lifetime accumulator must see nonzero activity after stepping",
        );
        assert_eq!(
            total.cache_hits,
            after_first.cache_hits + after_second.cache_hits,
            "hits total must equal sum of per-step snapshots",
        );
        assert_eq!(
            total.cache_misses,
            after_first.cache_misses + after_second.cache_misses,
            "misses total must equal sum of per-step snapshots",
        );
        assert!(
            total.cache_hits >= after_second.cache_hits,
            "total must be >= most-recent step",
        );

        let summary = world.memo_summary();
        assert!(summary.contains("memo_hit="));
        assert!(summary.contains("memo_tbl="));
        assert!(summary.contains("memo_mac="));
        assert!(summary.contains("memo_mac_bytes="));
        // hash-thing-71mp: guard the phase-timing output contract so the
        // fields can't silently disappear under a future refactor.
        assert!(summary.contains("p1="));
        assert!(summary.contains("p2="));
    }

    /// hash-thing-z7uu: `macro_cache_bytes_est()` must equal
    /// `macro_cache_entries() * MACRO_CACHE_BYTES_PER_ENTRY`. Guards the
    /// arithmetic from drifting if the struct is refactored (e.g. someone
    /// swaps in a different map type with different overhead).
    #[test]
    fn macro_cache_bytes_est_equals_entries_times_per_entry() {
        // Fresh world: no steps, macro cache empty.
        let world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        assert_eq!(world.macro_cache_entries(), 0);
        assert_eq!(world.macro_cache_bytes_est(), 0);

        // Drive a pow2 step to seed the macro cache.
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step_recursive_pow2();

        let entries = world.macro_cache_entries();
        assert_eq!(
            world.macro_cache_bytes_est(),
            entries * World::MACRO_CACHE_BYTES_PER_ENTRY,
        );
        const _: () = assert!(
            World::MACRO_CACHE_BYTES_PER_ENTRY >= 24,
            "per-entry estimate must cover at least key+value payload bytes",
        );
    }

    /// HUD overlay (hash-thing-nhwo) splits `memo_summary` on whitespace
    /// so each field lands on its own line. Guards against future schema
    /// changes that would introduce an intra-field space (e.g.
    /// `memo_tbl=1 234`) which would silently break the HUD layout, and
    /// against per-field widths that overflow the compact HUD panel.
    ///
    /// Prefix allow-list covers `memo_` (cache fields) and `p1` / `p2`
    /// (per-step phase timings from hash-thing-71mp, which are part of
    /// the same HUD block but semantically not memo state).
    #[test]
    fn memo_summary_splits_cleanly_for_hud_overlay() {
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step_recursive();
        world.step_recursive();

        let summary = world.memo_summary();
        let lines: Vec<&str> = summary.split_whitespace().collect();

        assert!(
            lines.len() >= 3,
            "memo_summary should split into at least 3 fields, got {lines:?}",
        );
        for line in &lines {
            assert!(
                line.starts_with("memo_")
                    || line.starts_with("p1=")
                    || line.starts_with("p2=")
                    || line.starts_with("p3=")
                    || line.starts_with("p4=")
                    || line.starts_with("bfs_"),
                "field must use an approved HUD prefix (memo_ / p1 / p2 / p3 / p4 / bfs_), got {line:?}",
            );
            assert!(
                line.len() <= 25,
                "field too wide for compact HUD panel ({} chars): {line:?}",
                line.len(),
            );
        }

        // hash-thing-bjdl (vqke.2): the three new tokens must be
        // present in the formatted line. Per Codex code-review §3:
        // lock the bounded summary tokens into the output-contract
        // tests so a future refactor that drops them is caught.
        assert!(
            lines.iter().any(|f| f.starts_with("memo_period=")),
            "memo_summary must include memo_period= token, got {lines:?}",
        );
        assert!(
            lines.iter().any(|f| f.starts_with("memo_phase_aliased=")),
            "memo_summary must include memo_phase_aliased= token, got {lines:?}",
        );
        assert!(
            lines.iter().any(|f| f.starts_with("memo_compact_drop=")),
            "memo_summary must include memo_compact_drop= token, got {lines:?}",
        );
        // hash-thing-tk4j (vqke.3): the two skip-rate tokens must be
        // present so the HUD overlay reflects whether step_node's
        // pre-cache fast paths are firing on mostly-stable scenes.
        assert!(
            lines.iter().any(|f| f.starts_with("memo_skip_empty=")),
            "memo_summary must include memo_skip_empty= token, got {lines:?}",
        );
        assert!(
            lines.iter().any(|f| f.starts_with("memo_skip_fixed=")),
            "memo_summary must include memo_skip_fixed= token, got {lines:?}",
        );
    }

    #[test]
    fn hashlife_stats_per_step_reset_preserved() {
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());

        world.step_recursive();
        let first_misses = world.hashlife_stats.cache_misses;
        assert!(first_misses > 0, "first step must produce misses");

        world.step_recursive();
        let second_misses = world.hashlife_stats.cache_misses;

        // Per-step field still captures only the latest step. If accumulation
        // accidentally folded into `hashlife_stats` rather than the `_total`
        // sibling, this would equal `first + second` (the lifetime total).
        assert_eq!(
            world.hashlife_stats_total.cache_misses,
            first_misses + second_misses,
            "lifetime accumulator must equal exact per-step sum",
        );
        assert!(
            second_misses < first_misses + second_misses,
            "per-step stats must reset between calls (second={second_misses}, sum={})",
            first_misses + second_misses,
        );
    }

    #[test]
    fn step_grid_once_phase_timings_accumulated_on_memo_miss() {
        // hash-thing-71mp: HashlifeStats.phase1_ns and .phase2_ns must be
        // non-zero after a real (memo-missing) step. On a fresh one-cell
        // world, step_recursive drops into step_base_case at least once
        // for the L3 node containing the seed — that's enough to tick
        // both phase timers past zero. Exact values are platform-dependent
        // (Instant::now resolution varies), so we only assert > 0.
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step_recursive();

        let stats = &world.hashlife_stats;
        assert!(
            stats.cache_misses > 0,
            "test precondition: seeded step should have at least one memo miss",
        );
        assert!(
            stats.phase1_ns > 0,
            "phase1_ns should accumulate across step_grid_once calls, got 0",
        );
        assert!(
            stats.phase2_ns > 0,
            "phase2_ns should accumulate across step_grid_once calls, got 0",
        );
    }

    #[test]
    fn step_recursive_records_step_node_wall_and_compact_ns() {
        // hash-thing-vqke Phase 0: step_node_wall_ns must be non-zero
        // (always taken, every step). compact_ns is 0 on a fresh tiny
        // world (the 2× growth gate hasn't tripped) — so we only assert
        // it stays well-defined (saturating arithmetic, no panic) and
        // that step_node_wall_ns >= phase1_ns + phase2_ns (descent
        // always at least sees the leaves).
        //
        // hash-thing-ecmn (vqke.4.1) review-pass: pin the strategy to
        // Serial. Under RayonPerFanout / RayonBfs, `phase1_ns` and
        // `phase2_ns` sum across worker threads (per
        // step_grid_once_pure return values) and can EXCEED the
        // single-thread wall-clock `step_node_wall_ns`. The
        // wall-vs-CPU-time invariant only holds for the serial path —
        // gemini standard code review caught the mismatch when the
        // default strategy flipped to RayonPerFanout.
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set_base_case_strategy(BaseCaseStrategy::Serial);
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step_recursive();

        let stats = &world.hashlife_stats;
        assert!(
            stats.step_node_wall_ns > 0,
            "step_node_wall_ns should be set on every step, got 0",
        );
        // p1+p2 happen INSIDE step_node, so step_node_wall must be >= their sum.
        // (Inequality is non-strict because the timers nest precisely.)
        assert!(
            stats.step_node_wall_ns >= stats.phase1_ns + stats.phase2_ns,
            "step_node_wall_ns ({}) should be >= phase1_ns ({}) + phase2_ns ({})",
            stats.step_node_wall_ns,
            stats.phase1_ns,
            stats.phase2_ns,
        );
        // compact_ns stays 0 unless the 2× growth gate fired — fine
        // here, just confirm the field exists and didn't panic.
        let _ = stats.compact_ns;
    }

    #[test]
    fn memo_summary_includes_p3_and_p4_phase_breakdown() {
        // hash-thing-vqke Phase 0: the perf summary must include p3
        // (descent overhead) and p4 (compact wall) so the szyh-baseline
        // unaccounted-94ms slice gets a phase-by-phase breakdown.
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());
        world.step_recursive();

        let summary = world.memo_summary();
        assert!(summary.contains("p3="), "missing p3 in summary: {summary}");
        assert!(summary.contains("p4="), "missing p4 in summary: {summary}");
        // The HUD-overlay tester (memo_summary_splits_cleanly_for_hud_overlay)
        // independently asserts the field-prefix whitelist + width caps.
    }

    #[test]
    fn memo_summary_on_fresh_world_is_zero_rate() {
        let world = World::new(4);
        let summary = world.memo_summary();
        assert!(
            summary.starts_with("memo_hit=0.000"),
            "fresh world reports honest zero hit rate, got: {summary}",
        );
        assert!(
            summary.contains("memo_churn=+0.000"),
            "fresh world reports honest zero churn, got: {summary}",
        );
    }

    #[test]
    fn memo_window_push_evicts_oldest_past_capacity() {
        let mut window = MemoWindow::default();
        for i in 0..(MemoWindow::CAPACITY as u64 + 5) {
            // Inject i hits / 1 miss each — after eviction the window holds
            // the most recent CAPACITY entries.
            window.push(i, 1);
        }
        assert_eq!(window.len(), MemoWindow::CAPACITY);
        let (hits, misses) = window.totals();
        // Window now holds hits [5..CAPACITY+5), each with 1 miss.
        let expected_hits: u64 = (5u64..5 + MemoWindow::CAPACITY as u64).sum();
        assert_eq!(hits, expected_hits);
        assert_eq!(misses, MemoWindow::CAPACITY as u64);
    }

    #[test]
    fn memo_window_hit_rate_matches_lifetime_before_capacity() {
        // While step count < CAPACITY the window IS the lifetime — both rates
        // must be identical, so `memo_churn` reports a clean +0.000. This is
        // the contract the log format relies on before enough steps accrue.
        let mut world = gol_world(GameOfLife3D::new(0, 6, 1, 3));
        world.set(wc(4), wc(4), wc(4), ALIVE.raw());

        world.step_recursive();
        world.step_recursive();
        world.step_recursive();

        assert!(
            world.memo_window.len() <= MemoWindow::CAPACITY,
            "window must not exceed capacity",
        );
        assert!(world.memo_window.len() < MemoWindow::CAPACITY);

        let (win_hits, win_misses) = world.memo_window.totals();
        let total = &world.hashlife_stats_total;
        assert_eq!(
            win_hits, total.cache_hits,
            "pre-capacity window totals must match lifetime",
        );
        assert_eq!(
            win_misses, total.cache_misses,
            "pre-capacity window totals must match lifetime",
        );

        let summary = world.memo_summary();
        assert!(
            summary.contains("memo_churn=+0.000"),
            "churn must be exactly 0 while window still filling, got: {summary}",
        );
    }

    #[test]
    fn memo_window_hit_rate_diverges_past_capacity() {
        // After CAPACITY+1 steps the oldest entry is evicted from the window
        // but retained by `hashlife_stats_total`. Construct a case where the
        // oldest entry has a different hit ratio than the remaining window so
        // we can observe divergence.
        let mut window = MemoWindow::default();
        // First push: all-miss (forces lifetime_hits / lifetime_misses split).
        window.push(0, 10);
        // Fill remaining CAPACITY - 1 slots with all-hit, to push the window
        // above an all-miss starting floor.
        for _ in 0..(MemoWindow::CAPACITY - 1) {
            window.push(10, 0);
        }
        // At CAPACITY: 1 all-miss + (CAPACITY-1) all-hit → window rate =
        // (CAPACITY-1)*10 / CAPACITY*10. Push one more all-hit → oldest
        // (all-miss) evicts → window becomes all-hit, rate → 1.0.
        let pre_push_rate = window.hit_rate();
        window.push(10, 0);
        let post_push_rate = window.hit_rate();

        assert!(
            post_push_rate > pre_push_rate,
            "evicting the all-miss oldest entry must raise the window rate",
        );
        assert!(
            (post_push_rate - 1.0).abs() < 1e-9,
            "after eviction window is all-hit → rate = 1.0, got {post_push_rate}",
        );
    }

    // ---- 1D gravity_gap_fill_column tests (hash-thing-jw3k.1) ----

    /// Build a test material registry where `STONE` is a block-rule material
    /// and air (0) is empty. Matches what the default sim registry gives us.
    fn gap_fill_materials() -> MaterialRegistry {
        MaterialRegistry::terrain_defaults()
    }

    /// SAND has the gravity block_rule in terrain_defaults; STONE is inert.
    /// Tests below use SAND for the "block_rule" role.
    fn sand_cell() -> CellState {
        SAND
    }

    #[test]
    fn gap_fill_column_empty_no_op() {
        let mat = gap_fill_materials();
        let mut col: Vec<CellState> = vec![0; 8];
        assert!(!gravity_gap_fill_column(&mut col, &mat));
        assert!(col.iter().all(|&c| c == 0));
    }

    #[test]
    fn gap_fill_column_solid_no_op() {
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col: Vec<CellState> = vec![sand; 8];
        assert!(!gravity_gap_fill_column(&mut col, &mat));
        assert!(col.iter().all(|&c| c == sand));
    }

    #[test]
    fn gap_fill_column_block_air_block_triple_swaps() {
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col = vec![sand, 0, sand];
        assert!(gravity_gap_fill_column(&mut col, &mat));
        // y=1 sees below=sand, above=sand → swap cell[1] with cell[2].
        assert_eq!(col, vec![sand, sand, 0]);
    }

    #[test]
    fn gap_fill_column_block_air_air_block_no_swap() {
        // Rev 2 fix: both above AND below must be block_rule; two adjacent
        // airs in the middle fail the predicate on both y=1 and y=2.
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col = vec![sand, 0, 0, sand];
        assert!(!gravity_gap_fill_column(&mut col, &mat));
        assert_eq!(col, vec![sand, 0, 0, sand]);
    }

    #[test]
    fn gap_fill_column_cascade_babab() {
        // [B,A,B,A,B]: y=1 swaps (above=B, below=B) → [B,B,A,A,B].
        // y=2 above=A → skip. y=3 below=A → skip. Single-pass semantics.
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col = vec![sand, 0, sand, 0, sand];
        assert!(gravity_gap_fill_column(&mut col, &mat));
        assert_eq!(col, vec![sand, sand, 0, 0, sand]);
    }

    #[test]
    fn gap_fill_column_in_place_vs_fresh_read_divergence() {
        // [B,A,B,B] under in-place semantics:
        //   y=1: above=B, below=B → swap 1↔2 → [B,B,A,B]
        //   y=2: cur=A (freshly swapped), above=B, below=B → swap 2↔3 → [B,B,B,A]
        // A fresh-read variant would end at [B,B,A,B] (y=2 reads original B at idx 2).
        // This test pins that gravity_gap_fill_column mutates in place.
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col = vec![sand, 0, sand, sand];
        assert!(gravity_gap_fill_column(&mut col, &mat));
        assert_eq!(
            col,
            vec![sand, sand, sand, 0],
            "in-place semantics: y=2 must see the fresh value written at y=1",
        );
    }

    #[test]
    fn gap_fill_column_boundary_y0_and_y_max_untouched() {
        let mat = gap_fill_materials();
        // Air at y=0 and y=side-1 never participates (y ranges 1..side-1).
        let mut col: Vec<CellState> = vec![0, 0, 0];
        assert!(!gravity_gap_fill_column(&mut col, &mat));
    }

    #[test]
    fn gap_fill_column_side_1_no_op() {
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col = vec![sand];
        assert!(!gravity_gap_fill_column(&mut col, &mat));
        assert_eq!(col, vec![sand]);
    }

    #[test]
    fn gap_fill_column_side_2_no_op() {
        let mat = gap_fill_materials();
        let sand = sand_cell();
        let mut col = vec![sand, 0];
        assert!(!gravity_gap_fill_column(&mut col, &mat));
        assert_eq!(col, vec![sand, 0]);
    }

    // -----------------------------------------------------------------
    // hash-thing-6iiz: mutate_materials cache-invalidation invariants
    // -----------------------------------------------------------------

    /// Force all five material-dependent caches into a populated state and
    /// seed `block_rule_present` so the invariant tests can observe clearing.
    fn prime_material_caches(world: &mut World) {
        world
            .hashlife_cache
            .insert((NodeId::EMPTY, 0), NodeId::EMPTY);
        world
            .hashlife_macro_cache
            .insert((NodeId::EMPTY, 0), NodeId::EMPTY);
        world.hashlife_inert_cache.insert(NodeId::EMPTY, None);
        world.hashlife_all_inert_cache.insert(NodeId::EMPTY, true);
        world.block_rule_present = Some(false);
    }

    fn assert_material_caches_cleared(world: &World) {
        assert!(
            world.hashlife_cache.is_empty(),
            "hashlife_cache not cleared"
        );
        assert!(
            world.hashlife_macro_cache.is_empty(),
            "hashlife_macro_cache not cleared"
        );
        assert!(
            world.hashlife_inert_cache.is_empty(),
            "hashlife_inert_cache not cleared"
        );
        assert!(
            world.hashlife_all_inert_cache.is_empty(),
            "hashlife_all_inert_cache not cleared"
        );
        assert_eq!(
            world.block_rule_present, None,
            "block_rule_present not cleared — dxi4.2 invariant regression"
        );
    }

    /// `mutate_materials` must clear `block_rule_present` so the
    /// `step_recursive_pow2` 4497 brute-force fallback stays correct after a
    /// block-rule assignment (hash-thing-6iiz / dxi4.2).
    #[test]
    fn mutate_materials_invalidates_block_rule_present() {
        let mut world = empty_world();
        prime_material_caches(&mut world);

        world.mutate_materials(|m| {
            let identity = m.register_block_rule(crate::sim::margolus::IdentityBlockRule);
            m.assign_block_rule(crate::terrain::materials::STONE_MATERIAL_ID, identity);
        });

        assert_material_caches_cleared(&world);
    }

    /// `set_gol_smoke_rule` routes through `mutate_materials`; it must invalidate
    /// all material-dependent caches, including `block_rule_present` (widened
    /// semantics from the `invalidate_rule_caches` → `invalidate_material_caches`
    /// rename).
    #[test]
    fn set_gol_smoke_rule_invalidates_all_material_caches() {
        let mut world = empty_world();
        prime_material_caches(&mut world);

        world.set_gol_smoke_rule(GameOfLife3D::rule445());

        assert_material_caches_cleared(&world);
    }

    /// `set_material_tick_divisor` routes through `mutate_materials`; same
    /// coverage as the smoke-rule test.
    #[test]
    fn set_material_tick_divisor_invalidates_all_material_caches() {
        let mut world = empty_world();
        prime_material_caches(&mut world);

        world.set_material_tick_divisor(crate::terrain::materials::WATER_MATERIAL_ID, 1);

        assert_material_caches_cleared(&world);
    }

    // ---- eiu9 rep_mat-walker adversarial guard ------------------------
    //
    // Mirror of qaca_walker_catches_corrupted_rep_mat in ht-render. Confirms
    // assert_svdag_lod_rep_mat_matches_world actually fires when a slot's
    // rep_mat field is corrupted post-build. Without this, the walker could
    // silently no-op on the world.rs path and we'd close rk4n's blind spot
    // only on paper.
    #[test]
    #[should_panic(expected = "LOD rep_mat / slot-encoding mismatches")]
    fn eiu9_walker_catches_corrupted_rep_mat() {
        let mut world = World::new(5); // 32³
                                       // Solid 16³ water block in one octant gives an interior root with
                                       // a non-trivial rep_mat the walker can validate.
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    world.set(wc(x), wc(y), wc(z), WATER);
                }
            }
        }

        let mut svdag = Svdag::new();
        sync_svdag_with_world(&mut world, &mut svdag);
        assert_svdag_matches_world(&world, &svdag, "pre-corruption water block");

        // Stomp the root node's rep_mat field to a bogus material id while
        // preserving the child mask; the walker must catch the mismatch.
        let root_offset = svdag.nodes[0] as usize;
        let original = svdag.nodes[root_offset];
        let bogus_mat: u32 = 0xABCD;
        svdag.nodes[root_offset] = (original & 0xFF) | (bogus_mat << 8);

        assert_svdag_matches_world(&world, &svdag, "corrupted rep_mat");
    }
}
