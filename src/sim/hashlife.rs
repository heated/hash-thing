//! Recursive Hashlife stepper (hash-thing-6gf.1).
//!
//! Computes one simulation step directly on the octree, without flattening the
//! entire grid. The result of stepping a level-n node (2^n × 2^n × 2^n) is a
//! level-(n-1) node representing the center (2^(n-1))³ cube after one generation.
//!
//! **Phase ordering** matches the brute-force path: CaRule first (react), then
//! BlockRule (move). Boundary semantics: CaRule wraps toroidally within the
//! input node; BlockRule clips at edges.
//!
//! **Base case (level 2):** A 4×4×4 input. Flatten, step the center 2×2×2
//! cells through CaRule + BlockRule, rebuild as level-1.
//!
//! **Recursive case (level ≥ 3):** Two phases:
//!   1. *Center* — build 27 intermediate nodes (level n-1) from grandchildren,
//!      then extract each center (level n-2) without time advance.
//!   2. *Step* — group the 27 centered nodes into 8 overlapping sub-cubes
//!      (each level n-1), recursively step each → 8 results at level n-2.
//!      Assemble into the level-(n-1) output.
//!
//! Step results are memoized by (NodeId, schedule_phase) so identical subtrees
//! anywhere in the world share a single cache entry per generation. Block-rule
//! partition uses node-local alignment (9ww), so origin is not in the cache key.
//! `schedule_phase = generation % memo_period()` where `memo_period` is
//! LCM over materials of `2 * tick_divisor` (iowh). With all divisors = 1 this
//! reduces to generation parity (period = 2).
//!
//! Compaction is deferred (m1f.14): intermediate nodes survive across frames,
//! enabling cache hits at every recursion level. Periodic compaction triggers
//! when the store exceeds 2× its post-compaction size. With all divisors = 1 the
//! phase alternates 0/1, so stable subtrees hit the cache every other frame;
//! with slower divisors hits occur every `memo_period` frames at the same phase.
//!
//! **Cache-preserving compaction (m1f.15.4):** The recursive descent builds
//! intermediate nodes (27 per recursive level) that become cache keys but are
//! NOT reachable from the step result. When compaction fires, cache keys are
//! passed as extra roots to `compacted_with_remap_keeping`, keeping them alive
//! through GC so cache entries survive.

use super::world::World;
use crate::octree::node::octant_index;
use crate::octree::{Cell, CellState, Node, NodeId};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU8, Ordering};

/// hash-thing-bjdl (vqke.2): process-wide gate for the memo-hit-rate
/// diagnostic probes inside `step_node`. Lazily initialised from
/// `HASH_THING_MEMO_DIAG=1` on first read; `false` (= probes off) by
/// default. Always-on probes were estimated at single-digit ms/step
/// at observed miss volumes (per Claude + Codex plan review of this
/// bead at notes/PR-bjdl-*-Standard-*.md); vqke is trying to make
/// `step_recursive` *faster*, not pay 5-10 ms for a diagnostic. Set
/// the env var to reproduce the diagnostic run described in the
/// bead and read the new tokens in `memo_summary`.
///
/// State encoding (AtomicU8 instead of bool so tests can still get a
/// "not yet read" sentinel and override before the first hot-path
/// load): 0 = uninitialised, 1 = enabled, 2 = disabled.
static MEMO_DIAG_STATE: AtomicU8 = AtomicU8::new(0);

// MEMO_DIAG_UNINIT (= 0) is the AtomicU8 default; the discriminant
// is checked inline via the `_` arm in `memo_diag_enabled` rather
// than via a named constant, so no `MEMO_DIAG_UNINIT` is exposed.
const MEMO_DIAG_ON: u8 = 1;
const MEMO_DIAG_OFF: u8 = 2;

// hash-thing-bjdl (vqke.2): test-only override for the diag gate,
// kept on the test thread's local storage so a parallel sibling
// test running on a different worker thread continues to observe
// the production global state (per Codex code-review §1 on the
// landed diagnostic — a process-wide override leaks the diag-ON
// signal to whichever sibling happens to call `step_node` during
// the override window). Each test thread reads its own override
// (default `None` = "no override, use the global atomic"); tests
// that need to set it write their value here. Cleared per-thread,
// not per-process.
#[cfg(test)]
thread_local! {
    static MEMO_DIAG_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Read-only accessor for the memo-diag flag. After the first call
/// the env-var lookup is cached as an atomic load on the hot path.
/// In test builds, a thread-local override (`force_memo_diag_for_test`)
/// takes precedence so unit tests can exercise the probe without
/// affecting parallel sibling tests.
fn memo_diag_enabled() -> bool {
    #[cfg(test)]
    {
        if let Some(override_val) = MEMO_DIAG_OVERRIDE.with(|c| c.get()) {
            return override_val;
        }
    }
    match MEMO_DIAG_STATE.load(Ordering::Relaxed) {
        MEMO_DIAG_ON => true,
        MEMO_DIAG_OFF => false,
        _ => {
            let v = std::env::var("HASH_THING_MEMO_DIAG").ok().as_deref() == Some("1");
            MEMO_DIAG_STATE.store(
                if v { MEMO_DIAG_ON } else { MEMO_DIAG_OFF },
                Ordering::Relaxed,
            );
            v
        }
    }
}

/// Test-only override for the memo-diag gate. Sets a thread-local
/// flag that takes precedence over the process-wide atomic for the
/// calling thread only — sibling tests running on other worker
/// threads continue to see the production-default value (OFF unless
/// the env var was set at process start). Pair with
/// `clear_memo_diag_test_override()` at scope exit if a downstream
/// helper on the same thread should resume reading the global gate.
#[cfg(test)]
pub(crate) fn force_memo_diag_for_test(enabled: bool) {
    MEMO_DIAG_OVERRIDE.with(|c| c.set(Some(enabled)));
}

/// Clear the test-thread's override, so subsequent
/// `memo_diag_enabled()` calls on this thread fall through to the
/// process-wide atomic (production default).
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn clear_memo_diag_test_override() {
    MEMO_DIAG_OVERRIDE.with(|c| c.set(None));
}

/// Per-phase micro-timing of one [`World::step_recursive_profiled`] invocation
/// (hash-thing-jw3k diagnostic harness). All `_us` fields are microseconds.
///
/// hash-thing-jw3k.1 split: `flatten_us + gap_fill_loop_us + from_flat_us`
/// (full-world flatten/fill/rebuild round-trip) was replaced by
/// `column_walk_us + splice_us + dirty_columns` (per-column walk + targeted
/// splice for columns that changed). The total phase name is still "gap_fill";
/// see [`Self::gap_fill_us`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StepProfile {
    pub total_us: u64,
    pub step_node_us: u64,
    /// Time spent walking all columns (flatten per column + 1D gap_fill).
    pub column_walk_us: u64,
    /// Time spent in [`crate::octree::NodeStore::splice_column`] for columns that changed.
    /// Subset of `column_walk_us` — reported separately so callers can tell
    /// whether rebuild cost or walk cost dominates.
    pub splice_us: u64,
    /// Count of columns where `gravity_gap_fill_column` reported a change
    /// (i.e. triggered a splice). Lets the bench verify the ~10% dirty
    /// assumption against real scene dynamics.
    pub dirty_columns: u32,
    pub compact_us: u64,
}

impl StepProfile {
    pub fn gap_fill_us(&self) -> u64 {
        self.column_walk_us
    }
}

const LEVEL3_SIDE: usize = 8;
const LEVEL3_CELL_COUNT: usize = LEVEL3_SIDE * LEVEL3_SIDE * LEVEL3_SIDE;
const CENTER_LEVEL3_SIDE: usize = 4;
const CENTER_LEVEL3_CELL_COUNT: usize =
    CENTER_LEVEL3_SIDE * CENTER_LEVEL3_SIDE * CENTER_LEVEL3_SIDE;

/// hash-thing-ecmn (vqke.4.1): one node-step entry in the BFS frontier.
/// `node` is the input at level n; `result` is filled at level-(n-1)
/// time during Phase 2 (level-3 batch) or Phase 3 (ascend).
/// `children[i]` resolves to the i-th sub-cube's stepped result at
/// level (n-2): either Direct (short-circuit / cache hit at descent
/// time) or Pending(idx) (lookup into `tasks[level - 1][idx].result`
/// after the lower level resolves).
#[derive(Clone)]
struct BfsTask {
    node: NodeId,
    effective_phase: u64,
    children: [BfsChildSlot; 8],
    result: Option<NodeId>,
}

#[derive(Clone, Copy, Debug)]
enum BfsChildSlot {
    /// Resolved at descent time — short-circuit or cache hit.
    Direct(NodeId),
    /// Pending until the lower-level task resolves; `usize` is an index
    /// into the next level's `tasks` vec.
    Pending(usize),
}

impl World {
    /// Step the world forward one generation using the recursive Hashlife path.
    pub fn step_recursive(&mut self) {
        assert!(
            self.level >= 3,
            "step_recursive requires level >= 3, got {}",
            self.level
        );
        self.hashlife_stats = super::world::HashlifeStats::default();
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        // Memo key: generation modulo the schedule period of the material
        // registry. With all tick_divisors = 1, memo_period() = 2 and this
        // reduces to the classic Margolus parity. With slower divisors, the
        // period grows to LCM(2*d_i) so identical inputs at the same schedule
        // phase always produce identical outputs (iowh).
        let period = self.materials.memo_period();
        let phase = self.generation % period;
        // hash-thing-vqke Phase 0: bracket step_node (recursive descent
        // + leaf evaluation) and maybe_compact separately so the perf
        // log decomposes the previously-unaccounted slice of `step`.
        let t_step_node = std::time::Instant::now();
        let result = self.step_root_dispatch(padded_root, padded_level, phase, period);
        self.hashlife_stats.step_node_wall_ns = t_step_node.elapsed().as_nanos() as u64;
        self.root = result;
        let step_stats_pre_compact = self.hashlife_stats;

        // No post-pass gap-fill: qy4g option G (2026-04-26) — static
        // internal gaps close in 1-2 ticks via parity-flip; falling
        // columns checkerboard in flight and compact on landing. See
        // World::step and SPEC.md for the visible-artifact tradeoff.
        self.generation += 1;

        let t_compact = std::time::Instant::now();
        self.maybe_compact();
        self.hashlife_stats.compact_ns = t_compact.elapsed().as_nanos() as u64;

        // hash-thing-vqke: accumulate AFTER compact so the lifetime
        // accumulator includes p4 (compact) and p3 (step_node wall).
        // memo_window only tracks hit/miss counts, so it stays based
        // on the pre-compact stats snapshot (compact doesn't change
        // those counters).
        let step_stats = self.hashlife_stats;
        self.hashlife_stats_total.accumulate(&step_stats);
        self.memo_window.push(
            step_stats_pre_compact.cache_hits,
            step_stats_pre_compact.cache_misses,
        );
    }

    /// Per-column gap-fill path. Retained `#[cfg(test)]` after qy4g option G
    /// removed gap-fill from the production step path (2026-04-26).
    #[cfg(test)]
    pub fn apply_gap_fill_column_path(&mut self) {
        let side = self.side();
        let mut column: Vec<CellState> = vec![0; side];
        for z in 0..side as u64 {
            for x in 0..side as u64 {
                self.store.flatten_column_into(self.root, x, z, &mut column);
                if super::world::gravity_gap_fill_column(&mut column, &self.materials) {
                    self.root = self.store.splice_column(self.root, x, z, &column);
                }
            }
        }
    }

    /// Old full-world gap-fill path, preserved for the cross-check regression
    /// test (`step_recursive_column_gap_fill_matches_flatten_rebuild_per_step`).
    /// Flattens the whole world, runs the 3D `gravity_gap_fill`, rebuilds
    /// from the flat buffer. Scoped for deletion in a follow-up bead once the
    /// column path has been running without drift in CI for multiple gens.
    #[cfg(test)]
    pub fn apply_gap_fill_via_flatten_rebuild(&mut self) {
        let side = self.side();
        let mut grid = self.store.flatten(self.root, side);
        super::world::gravity_gap_fill(&mut grid, side, &self.materials);
        self.root = self.store.from_flat(&grid, side);
    }

    /// Test-only: run only the Margolus/hashlife portion of `step_recursive`,
    /// stopping before the post-step gap-fill. Used by the cross-check
    /// regression test to sandwich the column path and the flatten-rebuild
    /// path around a common pre-gap-fill root.
    #[cfg(test)]
    pub fn step_margolus_only(&mut self) {
        assert!(self.level >= 3);
        self.hashlife_stats = super::world::HashlifeStats::default();
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        let period = self.materials.memo_period();
        let phase = self.generation % period;
        let result = self.step_root_dispatch(padded_root, padded_level, phase, period);
        self.root = result;
        let step_stats = self.hashlife_stats;
        self.hashlife_stats_total.accumulate(&step_stats);
        self.memo_window
            .push(step_stats.cache_hits, step_stats.cache_misses);
    }

    /// Test-only: finish a step that was started via `step_margolus_only`
    /// and gap-filled externally. Advances generation and runs `maybe_compact`.
    #[cfg(test)]
    pub fn finalize_step_after_external_gap_fill(&mut self) {
        self.generation += 1;
        self.maybe_compact();
    }

    /// Same work as [`Self::step_recursive`] but returns per-phase timings
    /// (hash-thing-jw3k). Intended for benchmark/diagnostic callers; production
    /// sim uses [`Self::step_recursive`].
    pub fn step_recursive_profiled(&mut self) -> StepProfile {
        assert!(
            self.level >= 3,
            "step_recursive_profiled requires level >= 3, got {}",
            self.level
        );
        let t0 = std::time::Instant::now();
        self.hashlife_stats = super::world::HashlifeStats::default();
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        let period = self.materials.memo_period();
        let phase = self.generation % period;
        let t_step = std::time::Instant::now();
        let result = self.step_root_dispatch(padded_root, padded_level, phase, period);
        let step_node_us = t_step.elapsed().as_micros() as u64;
        self.root = result;
        let step_stats = self.hashlife_stats;
        self.hashlife_stats_total.accumulate(&step_stats);
        self.memo_window
            .push(step_stats.cache_hits, step_stats.cache_misses);

        // qy4g.2 option G: gap-fill post-pass deleted from production. StepProfile
        // fields preserved as zeros for back-compat with bench_hashlife consumers.
        let (column_walk_us, splice_us, dirty_columns) = (0u64, 0u64, 0u32);

        self.generation += 1;

        let t_compact = std::time::Instant::now();
        self.maybe_compact();
        let compact_us = t_compact.elapsed().as_micros() as u64;

        StepProfile {
            total_us: t0.elapsed().as_micros() as u64,
            step_node_us,
            column_walk_us,
            splice_us,
            dirty_columns,
            compact_us,
        }
    }

    /// Number of generations advanced by [`Self::step_recursive_pow2`].
    pub fn recursive_pow2_step_count(&self) -> u64 {
        1u64 << (self.level - 1)
    }

    /// Step the world forward by the largest power-of-two skip supported by
    /// the current realized root size.
    pub fn step_recursive_pow2(&mut self) {
        assert!(
            self.level >= 3,
            "step_recursive_pow2 requires level >= 3, got {}",
            self.level
        );
        if self.has_block_rule_cells() {
            // Fallback: brute-step generation-by-generation when block rules are
            // present. Pre-qy4g.2 this fallback existed to bridge over the missing
            // gap-fill post-pass in `step_node_macro`. With option G (gap-fill
            // deleted from production), the gap-fill divergence reason is gone —
            // but `step_node_macro`'s memoized hop may still need empirical
            // validation against brute Margolus before we trust it for
            // block-rule worlds. Tracked in hash-thing-qy4g (epic) follow-up.
            let steps = self.recursive_pow2_step_count();
            for _ in 0..steps {
                self.step();
            }
            self.block_rule_present = None; // brute-force may have consumed block-rule cells
            return;
        }
        self.hashlife_stats = super::world::HashlifeStats::default();
        let padded_root = self.pad_root();
        let padded_level = self.level + 1;
        let result = self.step_node_macro(padded_root, padded_level, self.generation);
        self.root = result;
        let step_stats = self.hashlife_stats;
        self.hashlife_stats_total.accumulate(&step_stats);
        self.memo_window
            .push(step_stats.cache_hits, step_stats.cache_misses);
        self.generation += self.recursive_pow2_step_count();

        self.maybe_compact();
    }

    /// Compact the store when it has grown past 2× its post-compaction size,
    /// keeping cache-referenced intermediate nodes alive (m1f.15.4).
    ///
    /// Deferred compaction lets intermediate nodes from recursive descent
    /// survive across frames, enabling cache hits. When compaction fires,
    /// cache keys are passed as extra roots so they survive GC — without
    /// this, every compaction destroys the cache and forces full
    /// recomputation.
    fn maybe_compact(&mut self) {
        let current_size = self.store.stats();
        if self.store_size_at_last_compact == 0 {
            self.store_size_at_last_compact = current_size;
            return;
        }
        if current_size <= self.store_size_at_last_compact * 2 {
            return;
        }
        self.compact_keeping(&[]);
    }

    /// Unconditional compaction, preserving the world root, hashlife
    /// cache-referenced nodes, and any caller-provided `extra_roots`.
    ///
    /// Used by:
    /// - [`Self::maybe_compact`] when the 2× growth threshold is met
    ///   (no extra roots beyond cache references).
    /// - The cswp.8.3 chunk-LOD path (hash-thing-e4ep) to shed ghost
    ///   interior chains accumulated by repeated `lod_collapse_chunk`
    ///   calls without losing the live `view_root`. Within the lib,
    ///   prefer `maybe_compact` so the 2× threshold gating stays in one
    ///   place; out-of-lib callers (e.g. the bin-crate's per-frame
    ///   chunk-LOD trigger) drive their own threshold and pass live
    ///   roots that aren't on the hashlife cache list.
    ///
    /// Updates `self.store`, `self.root`, the four hashlife caches (via
    /// [`Self::remap_caches`]), the `store_size_at_last_compact` meter,
    /// and publishes the remap to `last_compaction_remap` with
    /// compose-on-write semantics so back-to-back compactions before
    /// the renderer next drains the slot — for example two sim steps
    /// each tripping `maybe_compact` between frames, or a sim-step
    /// `maybe_compact` followed in the next frame by a chunk-LOD
    /// `compact_keeping` if the renderer hasn't yet drained — don't
    /// clobber the pending A→B map with B→C (hash-thing-rk4n.1).
    pub fn compact_keeping(&mut self, extra_roots: &[NodeId]) {
        let mut roots = self.cache_referenced_nodes();
        roots.extend_from_slice(extra_roots);
        let (new_store, new_root, remap) =
            self.store.compacted_with_remap_keeping(self.root, &roots);
        self.store = new_store;
        self.root = new_root;
        self.remap_caches(&remap);
        self.last_compaction_remap = Some(match self.last_compaction_remap.take() {
            Some(existing) => super::world::compose_remap(existing, &remap),
            None => remap,
        });
        self.store_size_at_last_compact = self.store.stats();
    }

    /// Collect cache key NodeIds that must survive compaction.
    /// Cache keys are intermediate nodes not reachable from the world root;
    /// cache values are step results that ARE reachable from root, so only
    /// keys need to be kept alive as extra roots.
    fn cache_referenced_nodes(&self) -> Vec<NodeId> {
        let mut nodes =
            Vec::with_capacity(self.hashlife_cache.len() + self.hashlife_macro_cache.len());
        for &(node, _) in self.hashlife_cache.keys() {
            nodes.push(node);
        }
        for &(node, _) in self.hashlife_macro_cache.keys() {
            nodes.push(node);
        }
        nodes
    }

    fn has_block_rule_cells(&mut self) -> bool {
        if let Some(cached) = self.block_rule_present {
            return cached;
        }
        let result = self.has_block_rule_in_subtree(self.root);
        self.block_rule_present = Some(result);
        result
    }

    /// Walk the octree for any cell with a BlockRule. Short-circuits on empty
    /// nodes — avoids O(n³) flatten for worlds that are mostly empty/inert.
    fn has_block_rule_in_subtree(&self, node: NodeId) -> bool {
        if node == NodeId::EMPTY {
            return false;
        }
        match self.store.get(node) {
            Node::Leaf(state) => self
                .materials
                .block_rule_id_for_cell(Cell::from_raw(*state))
                .is_some(),
            Node::Interior { children, .. } => {
                children.iter().any(|&c| self.has_block_rule_in_subtree(c))
            }
        }
    }

    /// Check if every leaf in the subtree is inert (CaRule::Noop, no BlockRule).
    /// Unlike `inert_uniform_state`, this catches mixed-material subtrees
    /// (e.g. stone/air boundaries). For all-inert nodes, stepping produces
    /// the center extraction — no CA computation needed.
    fn is_all_inert(&mut self, node: NodeId) -> bool {
        if node == NodeId::EMPTY {
            return true;
        }
        if let Some(&cached) = self.hashlife_all_inert_cache.get(&node) {
            return cached;
        }
        let result = match self.store.get(node).clone() {
            Node::Leaf(state) => {
                state == 0
                    || self
                        .materials
                        .cell_is_inert_fixed_point(Cell::from_raw(state))
            }
            Node::Interior { children, .. } => children.into_iter().all(|c| self.is_all_inert(c)),
        };
        self.hashlife_all_inert_cache.insert(node, result);
        result
    }

    /// hash-thing-5ie4 (vqke.2.1): does the subtree contain any cell
    /// whose material has `tick_divisor > 1`? Drives the phase-fold
    /// in `step_node`: when this returns `false` for a node, its
    /// step result depends only on `(node, generation % 2)`, so the
    /// memo key collapses from the registry's full `memo_period` to
    /// Margolus parity.
    ///
    /// Cached per NodeId on `self.hashlife_slow_divisor_cache`. The
    /// answer is content-determined (NodeIds are content-addressed),
    /// so cell-edit paths produce a fresh NodeId and hit the cache
    /// cold — only material-registry mutation invalidates existing
    /// entries (`World::invalidate_material_caches`).
    ///
    /// Implementation note (per Codex plan-review §1): the
    /// `tick_divisor_flags()` slice is queried INSIDE the `Leaf`
    /// arm, not before the `match`. Holding the slice across the
    /// recursive `Interior` call would conflict with the recursive
    /// `&mut self` borrow.
    fn subtree_has_slow_divisor(&mut self, node: NodeId) -> bool {
        if node == NodeId::EMPTY {
            return false;
        }
        if let Some(&cached) = self.hashlife_slow_divisor_cache.get(&node) {
            return cached;
        }
        let result = match self.store.get(node).clone() {
            Node::Leaf(state) => {
                let cell = Cell::from_raw(state);
                let mat = cell.material() as usize;
                let divisors = self.materials.tick_divisor_flags();
                divisors.get(mat).copied().unwrap_or(1) > 1
            }
            Node::Interior { children, .. } => children
                .into_iter()
                .any(|c| self.subtree_has_slow_divisor(c)),
        };
        self.hashlife_slow_divisor_cache.insert(node, result);
        result
    }

    /// Check if a subtree is uniformly one inert material.
    /// Returns Some(state) if all leaves are the same inert state.
    fn inert_uniform_state(&mut self, node: NodeId) -> Option<CellState> {
        if let Some(&cached) = self.hashlife_inert_cache.get(&node) {
            return cached;
        }
        let result = match self.store.get(node).clone() {
            Node::Leaf(state) => self
                .materials
                .cell_is_inert_fixed_point(Cell::from_raw(state))
                .then_some(state),
            Node::Interior { children, .. } => {
                let state = self.inert_uniform_state(children[0])?;
                for child in children.into_iter().skip(1) {
                    if self.inert_uniform_state(child) != Some(state) {
                        self.hashlife_inert_cache.insert(node, None);
                        return None;
                    }
                }
                Some(state)
            }
        };
        self.hashlife_inert_cache.insert(node, result);
        result
    }

    /// Remap hashlife cache keys and values through a compaction remap table.
    /// Entries referencing unreachable nodes (not in remap) are dropped.
    fn remap_caches(&mut self, remap: &FxHashMap<NodeId, NodeId>) {
        // Remap hashlife_cache: (NodeId, schedule_phase) → NodeId
        let old_cache = std::mem::take(&mut self.hashlife_cache);
        let before = old_cache.len() as u64;
        self.hashlife_cache.reserve(old_cache.len());
        // hash-thing-bjdl (vqke.2): count kept entries directly in the
        // success branch (rather than reading the final map length)
        // so the metric isn't conflated with any future key-coalescing
        // change in this loop. Dropped is `before - kept` — exact.
        // Per Codex plan-review §3.
        let mut kept: u64 = 0;
        for ((node, phase), result) in old_cache {
            if let (Some(&new_node), Some(&new_result)) = (remap.get(&node), remap.get(&result)) {
                self.hashlife_cache.insert((new_node, phase), new_result);
                kept += 1;
            }
        }
        self.hashlife_stats.compact_entries_kept += kept;
        self.hashlife_stats.compact_entries_dropped += before - kept;

        // Remap hashlife_macro_cache: (NodeId, generation) → NodeId
        let old_macro = std::mem::take(&mut self.hashlife_macro_cache);
        self.hashlife_macro_cache.reserve(old_macro.len());
        for ((node, gen), result) in old_macro {
            if let (Some(&new_node), Some(&new_result)) = (remap.get(&node), remap.get(&result)) {
                self.hashlife_macro_cache
                    .insert((new_node, gen), new_result);
            }
        }

        // Remap hashlife_inert_cache: NodeId → Option<CellState>
        let old_inert = std::mem::take(&mut self.hashlife_inert_cache);
        self.hashlife_inert_cache.reserve(old_inert.len());
        for (node, state) in old_inert {
            if let Some(&new_node) = remap.get(&node) {
                self.hashlife_inert_cache.insert(new_node, state);
            }
        }

        // Remap hashlife_all_inert_cache: NodeId → bool
        let old_all_inert = std::mem::take(&mut self.hashlife_all_inert_cache);
        self.hashlife_all_inert_cache.reserve(old_all_inert.len());
        for (node, inert) in old_all_inert {
            if let Some(&new_node) = remap.get(&node) {
                self.hashlife_all_inert_cache.insert(new_node, inert);
            }
        }

        // hash-thing-5ie4 (vqke.2.1): remap hashlife_slow_divisor_cache:
        // NodeId → bool. Same lifecycle as the inert / all-inert
        // caches above. Entries whose NodeId is unreachable in the
        // post-compact namespace are dropped on the same principle.
        let old_slow_div = std::mem::take(&mut self.hashlife_slow_divisor_cache);
        self.hashlife_slow_divisor_cache.reserve(old_slow_div.len());
        for (node, has_slow) in old_slow_div {
            if let Some(&new_node) = remap.get(&node) {
                self.hashlife_slow_divisor_cache.insert(new_node, has_slow);
            }
        }
    }

    /// Wrap the current root in a one-level-larger node, padding with empty.
    fn pad_root(&mut self) -> NodeId {
        let empty_child = self.store.empty(self.level - 1);
        let root_children = self.store.children(self.root);

        let mut padded_children = [NodeId::EMPTY; 8];
        for oct in 0..8 {
            let mirror = 7 - oct;
            let mut sub = [empty_child; 8];
            sub[mirror] = root_children[oct];
            padded_children[oct] = self.store.interior(self.level, sub);
        }
        self.store.interior(self.level + 1, padded_children)
    }

    /// Recursively step a node. Input level n (≥ 3), output level n-1.
    /// Block-rule partition uses node-local alignment, so origin is not needed
    /// and the cache key is (NodeId, schedule_phase). With per-material
    /// tick_divisors all = 1, `schedule_phase = generation % 2` (Margolus
    /// parity). With any divisor > 1 the phase expands to `generation %
    /// memo_period()` where memo_period = LCM over materials of 2*divisor
    /// (iowh). Identical subtrees at the same schedule_phase always produce
    /// identical outputs (9ww + iowh).
    ///
    /// `memo_period` is the period passed by `step_recursive`'s top-level
    /// call (computed once from `self.materials.memo_period()` at line
    /// 98). It's threaded through so `step_node` can run the
    /// hash-thing-bjdl phase-alias diagnostic probe at miss sites without
    /// re-querying the registry on every miss.
    fn step_node(
        &mut self,
        node: NodeId,
        level: u32,
        schedule_phase: u64,
        memo_period: u64,
    ) -> NodeId {
        assert!(level >= 3, "step_node requires level >= 3, got {level}");

        // Empty nodes step to empty: any rule applied to 26 air neighbors produces air
        // (NoopRule is identity; GoL-family rules have birth_min >= 1). No BlockRule on air.
        if self.store.population(node) == 0 {
            self.hashlife_stats.empty_skips += 1;
            return self.store.empty(level - 1);
        }

        // Fixed-point: uniform inert subtree (all leaves same inert material).
        // Stepping produces the same uniform node one level smaller.
        if let Some(state) = self.inert_uniform_state(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return self.store.uniform(level - 1, state);
        }

        // All-inert: every leaf is CaRule::Noop, no BlockRule (but possibly
        // mixed materials, e.g. stone/air boundary). Stepping = center extract.
        if self.is_all_inert(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return self.center_node(node, level);
        }

        // hash-thing-5ie4 (vqke.2.1): phase-fold for fast subtrees.
        // When `memo_period > 2` the registry has at least one
        // material with `tick_divisor > 1`, so the cache key uses
        // `gen % LCM(2 * d_i) > 2`. But for a subtree containing no
        // slow-divisor cells, the step result depends only on
        // `(node, gen % 2)` — the divisor gates in
        // `step_grid_once` are dead code (all divisors = 1) and the
        // BlockRule offset is `(gen / 1) % 2 = gen % 2`. Fold the
        // key for those subtrees so cache reuse extends across
        // every-other-generation pairs.
        //
        // The check is gated on `memo_period > 2` so the only-Margolus
        // (period=2) world does not pay for the predicate: the fold
        // would be a no-op there anyway (`schedule_phase % 2 ==
        // schedule_phase`). The predicate cache is hot in steady
        // state (one map lookup per call); first-population cost
        // amortises across the recursive descent.
        //
        // Correctness note (per Claude plan-review §1): this fold
        // is sound because `step_recursive_case` builds intermediate
        // nodes from grandchildren of `node`, and the content-addressed
        // `store.interior` returns the same NodeId for the same
        // content. A fast subtree's descendants are by induction
        // also fast, so every recursive call inside the fast branch
        // sees the same predicate value and the same fold.
        let effective_phase = if memo_period > 2 && !self.subtree_has_slow_divisor(node) {
            schedule_phase % 2
        } else {
            schedule_phase
        };
        let key = (node, effective_phase);
        if let Some(&cached) = self.hashlife_cache.get(&key) {
            self.hashlife_stats.cache_hits += 1;
            return cached;
        }
        self.hashlife_stats.cache_misses += 1;
        if (level as usize) >= 3 {
            self.hashlife_stats.misses_by_level[(level - 3) as usize] += 1;
        }
        // hash-thing-bjdl (vqke.2): hypothesis-3 probe, env-gated.
        // Mass-zero in production (the LazyLock check is a single
        // atomic load); only fires when HASH_THING_MEMO_DIAG=1 is set
        // at process start. At default-scene period=4 the inner loop
        // is at most 3 hashmap lookups per miss; on a 256³ default
        // scene at ~6-50k misses/step that's 18-150k extra lookups
        // per step (5-10 ms) which the rest of vqke is trying to
        // speed up — hence the gate. Boolean-per-miss semantics: a
        // node aliased at three other phases counts ONCE.
        //
        // Probe is anchored on `effective_phase` (post-fold), so a
        // miss whose fast-subtree fold already collapsed it to
        // phase=0/1 only flags as aliased when the *folded* key has
        // an alternate-phase sibling — which after vqke.2.1 should
        // be rare. The pre-fold raw `schedule_phase` could expose a
        // higher alias rate, but post-fold is the relevant signal
        // for whether further key-shape changes (vqke.2.2) would
        // recoup more.
        if memo_diag_enabled() && memo_period > 1 {
            for p in 0..memo_period {
                if p == effective_phase {
                    continue;
                }
                if self.hashlife_cache.contains_key(&(node, p)) {
                    self.hashlife_stats.cache_misses_phase_aliased += 1;
                    break;
                }
            }
        }

        let result = if level == 3 {
            self.step_base_case(node, effective_phase)
        } else {
            self.step_recursive_case(node, level, effective_phase, memo_period)
        };

        self.hashlife_cache.insert(key, result);
        result
    }

    /// Base case: level-3 node (8×8×8). Flatten, run CaRule on interior 6³,
    /// run BlockRule on all aligned blocks, extract center 4³ → level-2 output.
    ///
    /// The unused `_schedule_phase` parameter matches the cache key component
    /// that the caller uses to distinguish memo entries; it is redundant with
    /// `self.generation % memo_period` for a given step, but kept in the
    /// signature so step_node's call is symmetric with step_recursive_case.
    fn step_base_case(&mut self, node: NodeId, _schedule_phase: u64) -> NodeId {
        // Stack-allocated grid avoids heap allocation per base case (~16K calls).
        let mut grid = [0 as CellState; LEVEL3_CELL_COUNT];
        self.store.flatten_buf(node, &mut grid, LEVEL3_SIDE);
        let (next, p1_ns, p2_ns) = self.step_grid_once(&grid, self.generation);
        self.hashlife_stats.phase1_ns += p1_ns;
        self.hashlife_stats.phase2_ns += p2_ns;
        self.center_level3_grid_to_node(&next)
    }

    fn step_node_macro(&mut self, node: NodeId, level: u32, generation: u64) -> NodeId {
        debug_assert!(
            level >= 3,
            "step_node_macro requires level >= 3, got {level}"
        );

        // Empty nodes step to empty across any number of generations:
        // identity^N = identity. CaRule-only worlds (macro path prerequisite).
        if self.store.population(node) == 0 {
            self.hashlife_stats.empty_skips += 1;
            return self.store.empty(level - 1);
        }

        // Fixed-point: uniform inert subtree → same material, smaller node.
        if let Some(state) = self.inert_uniform_state(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return self.store.uniform(level - 1, state);
        }

        // All-inert: mixed materials but all noop → center extract.
        if self.is_all_inert(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return self.center_node(node, level);
        }

        let key = (node, generation);
        if let Some(&cached) = self.hashlife_macro_cache.get(&key) {
            self.hashlife_stats.cache_hits += 1;
            return cached;
        }
        self.hashlife_stats.cache_misses += 1;
        if (level as usize) >= 3 {
            self.hashlife_stats.misses_by_level[(level - 3) as usize] += 1;
        }

        let result = if level == 3 {
            self.step_base_case_macro(node, generation)
        } else {
            self.step_recursive_case_macro(node, level, generation)
        };

        self.hashlife_macro_cache.insert(key, result);
        result
    }

    fn step_base_case_macro(&mut self, node: NodeId, generation: u64) -> NodeId {
        let mut grid = [0 as CellState; LEVEL3_CELL_COUNT];
        self.store.flatten_buf(node, &mut grid, LEVEL3_SIDE);
        let (next, p1_ns_a, p2_ns_a) = self.step_grid_once(&grid, generation);
        let (next, p1_ns_b, p2_ns_b) = self.step_grid_once(&next, generation + 1);
        self.hashlife_stats.phase1_ns += p1_ns_a + p1_ns_b;
        self.hashlife_stats.phase2_ns += p2_ns_a + p2_ns_b;
        self.center_level3_grid_to_node(&next)
    }

    fn step_grid_once(
        &self,
        grid: &[CellState],
        generation: u64,
    ) -> ([CellState; LEVEL3_CELL_COUNT], u64, u64) {
        step_grid_once_pure(grid, generation, &self.materials)
    }

    fn center_level3_grid_to_node(&mut self, grid: &[CellState]) -> NodeId {
        let mut center_grid = [0 as CellState; CENTER_LEVEL3_CELL_COUNT];
        for cz in 0..CENTER_LEVEL3_SIDE {
            for cy in 0..CENTER_LEVEL3_SIDE {
                for cx in 0..CENTER_LEVEL3_SIDE {
                    center_grid[cx
                        + cy * CENTER_LEVEL3_SIDE
                        + cz * CENTER_LEVEL3_SIDE * CENTER_LEVEL3_SIDE] = grid
                        [(cx + 2) + (cy + 2) * LEVEL3_SIDE + (cz + 2) * LEVEL3_SIDE * LEVEL3_SIDE];
                }
            }
        }
        self.store.from_flat(&center_grid, CENTER_LEVEL3_SIDE)
    }

    /// Recursive case: level ≥ 3.
    fn step_recursive_case(
        &mut self,
        node: NodeId,
        level: u32,
        schedule_phase: u64,
        memo_period: u64,
    ) -> NodeId {
        let children = self.store.children(node);
        let sub: [[NodeId; 8]; 8] = std::array::from_fn(|i| self.store.children(children[i]));

        // Build 3×3×3 = 27 intermediate nodes at level (n-1).
        let mut inter = [NodeId::EMPTY; 27];
        for pz in 0..3usize {
            for py in 0..3usize {
                for px in 0..3usize {
                    let mut octants = [NodeId::EMPTY; 8];
                    for sz in 0..2usize {
                        for sy in 0..2usize {
                            for sx in 0..2usize {
                                let (parent_x, sub_x) = source_index(px, sx);
                                let (parent_y, sub_y) = source_index(py, sy);
                                let (parent_z, sub_z) = source_index(pz, sz);
                                let parent_oct =
                                    octant_index(parent_x as u32, parent_y as u32, parent_z as u32);
                                let sub_oct =
                                    octant_index(sub_x as u32, sub_y as u32, sub_z as u32);
                                octants[octant_index(sx as u32, sy as u32, sz as u32)] =
                                    sub[parent_oct][sub_oct];
                            }
                        }
                    }
                    inter[px + py * 3 + pz * 9] = self.store.interior(level - 1, octants);
                }
            }
        }

        // Extract center (level n-2) of each intermediate — spatial reindex only.
        let mut centered = [NodeId::EMPTY; 27];
        for i in 0..27 {
            centered[i] = self.center_node(inter[i], level - 1);
        }

        // Group into 8 overlapping sub-cubes (level n-1) and assemble their
        // sub_roots up-front. The 8 step_node calls follow — at level==4
        // (i.e. each sub-cube is a level-3 base case) the calls may be
        // batched through rayon if `base_case_strategy == RayonPerFanout`.
        let mut sub_roots = [NodeId::EMPTY; 8];
        for oz in 0..2usize {
            for oy in 0..2usize {
                for ox in 0..2usize {
                    let mut sub_cube = [NodeId::EMPTY; 8];
                    for dz in 0..2usize {
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                sub_cube[octant_index(dx as u32, dy as u32, dz as u32)] =
                                    centered[(ox + dx) + (oy + dy) * 3 + (oz + dz) * 9];
                            }
                        }
                    }
                    sub_roots[octant_index(ox as u32, oy as u32, oz as u32)] =
                        self.store.interior(level - 1, sub_cube);
                }
            }
        }

        let mut result_children = [NodeId::EMPTY; 8];
        if level == 4
            && matches!(
                self.base_case_strategy,
                super::world::BaseCaseStrategy::RayonPerFanout
            )
        {
            // hash-thing-ftuu (vqke.4): batch the 8 level-3 base cases through
            // rayon. Replicates the prelude of `step_node` (empty / inert /
            // cache lookup) inline so the parallel work is restricted to the
            // pure `step_grid_once_pure` calls — the only non-trivial
            // computation that can run without `&mut self`.
            self.step_level4_fanout_rayon(
                &sub_roots,
                schedule_phase,
                memo_period,
                &mut result_children,
            );
        } else {
            for i in 0..8 {
                result_children[i] =
                    self.step_node(sub_roots[i], level - 1, schedule_phase, memo_period);
            }
        }

        self.store.interior(level - 1, result_children)
    }

    /// hash-thing-ftuu (vqke.4): rayon-parallel evaluation of the 8 level-3
    /// base cases under a level-4 fanout. Replaces the serial loop over
    /// `step_node(sub_root, 3, ...)` calls when
    /// `base_case_strategy == RayonPerFanout`.
    ///
    /// Phase A (sequential, mutates stats + reads store/cache): mirrors the
    /// `step_node` prelude — empty-skip, fixed-point-skip, all-inert-skip,
    /// fast-subtree phase fold, cache lookup. Sub-cubes that short-circuit
    /// or hit the cache are recorded directly into `result_children`.
    /// Sub-cubes that miss are recorded into a queue alongside their
    /// flattened 8³ grid (read-only on store).
    ///
    /// Phase B (parallel, no `&self`): rayon `par_iter` over the queued
    /// flat grids invokes `step_grid_once_pure` to compute each output
    /// grid. The threshold below `4` falls back to serial `iter` because
    /// rayon's per-task overhead exceeds the win at very small batch sizes
    /// (a level-4 fanout caps at 8 base cases, and most steady-state
    /// fanouts have fewer than 8 misses).
    ///
    /// Phase C (sequential, mutates store + cache + stats): commits each
    /// output grid via `from_flat` + `center_level3_grid_to_node`, inserts
    /// the result into `hashlife_cache` under the same `(sub_root,
    /// effective_phase)` key the next `step_node` lookup would use, and
    /// folds `phase1_ns` / `phase2_ns` into stats so observability stays
    /// identical to the serial path.
    ///
    /// Bit-exact with the serial path — verified by
    /// `tests/base_case_rayon_parity.rs`.
    fn step_level4_fanout_rayon(
        &mut self,
        sub_roots: &[NodeId; 8],
        schedule_phase: u64,
        memo_period: u64,
        result_children: &mut [NodeId; 8],
    ) {
        use rayon::prelude::*;

        // Per-sub_root classification: either resolved (short-circuit or
        // cache hit) or pending (miss, needs base-case compute).
        //
        // Dedupe at queue time so two equal `sub_roots[i] == sub_roots[j]`
        // entries don't both run `step_grid_once_pure`: the second
        // occurrence is recorded as a deferred dupe pointing at the
        // unique miss's pending index, then resolves from the rayon
        // output post-batch. This both saves recompute (worst case 8×
        // for a fully-aliased fanout) AND makes the (cache_hits,
        // cache_misses) accounting match the serial path's
        // (1 miss + N-1 hits) exactly.
        let mut resolved: [Option<NodeId>; 8] = [None; 8];
        let mut pending: Vec<(usize, u64)> = Vec::with_capacity(8); // (sub_root_index, effective_phase) for unique misses
        let mut pending_grids: Vec<[CellState; LEVEL3_CELL_COUNT]> = Vec::with_capacity(8);
        let mut deferred_dupes: Vec<(usize, usize)> = Vec::new(); // (sub_root_index, pending_index) for duplicate misses

        for i in 0..8 {
            let nid = sub_roots[i];

            // Mirrors `step_node` lines 650-666: empty / uniform-inert /
            // all-inert short-circuits.
            if self.store.population(nid) == 0 {
                self.hashlife_stats.empty_skips += 1;
                resolved[i] = Some(self.store.empty(2));
                continue;
            }
            if let Some(state) = self.inert_uniform_state(nid) {
                self.hashlife_stats.fixed_point_skips += 1;
                resolved[i] = Some(self.store.uniform(2, state));
                continue;
            }
            if self.is_all_inert(nid) {
                self.hashlife_stats.fixed_point_skips += 1;
                resolved[i] = Some(self.center_node(nid, 3));
                continue;
            }

            // Mirrors `step_node` lines 694-703: fast-subtree phase fold +
            // cache lookup.
            let effective_phase = if memo_period > 2 && !self.subtree_has_slow_divisor(nid) {
                schedule_phase % 2
            } else {
                schedule_phase
            };
            let key = (nid, effective_phase);
            if let Some(&cached) = self.hashlife_cache.get(&key) {
                self.hashlife_stats.cache_hits += 1;
                resolved[i] = Some(cached);
                continue;
            }

            // Dedupe: if (nid, effective_phase) is already pending from a
            // prior sub_root in this fanout, the serial path would have
            // hit the cache after the first compute. Match that: count
            // as a cache_hit and defer resolution until Phase C.
            if let Some(pending_idx) = pending
                .iter()
                .position(|&(j, p)| sub_roots[j] == nid && p == effective_phase)
            {
                self.hashlife_stats.cache_hits += 1;
                deferred_dupes.push((i, pending_idx));
                continue;
            }

            // Unique miss — record cache_misses now (matches `step_node`
            // line 704 which counts the miss before evaluation).
            self.hashlife_stats.cache_misses += 1;
            self.hashlife_stats.misses_by_level[0] += 1; // level 3 = misses_by_level[0]

            // hash-thing-bjdl phase-aliasing diag probe (env-gated).
            if memo_diag_enabled() && memo_period > 1 {
                for p in 0..memo_period {
                    if p == effective_phase {
                        continue;
                    }
                    if self.hashlife_cache.contains_key(&(nid, p)) {
                        self.hashlife_stats.cache_misses_phase_aliased += 1;
                        break;
                    }
                }
            }

            let mut grid = [0 as CellState; LEVEL3_CELL_COUNT];
            self.store.flatten_buf(nid, &mut grid, LEVEL3_SIDE);
            pending.push((i, effective_phase));
            pending_grids.push(grid);
        }

        // Phase B: parallel step_grid_once_pure over the pending grids.
        // Below the rayon threshold, fall back to serial iter — at very
        // small batch sizes rayon's per-task overhead dominates the
        // ~14 µs of step_grid_once work.
        const RAYON_BATCH_THRESHOLD: usize = 4;
        let generation = self.generation;
        let materials: &crate::terrain::materials::MaterialRegistry = &self.materials;
        let outputs: Vec<([CellState; LEVEL3_CELL_COUNT], u64, u64)> = if pending_grids.is_empty() {
            Vec::new()
        } else if pending_grids.len() < RAYON_BATCH_THRESHOLD {
            pending_grids
                .iter()
                .map(|grid| step_grid_once_pure(grid, generation, materials))
                .collect()
        } else {
            pending_grids
                .par_iter()
                .map(|grid| step_grid_once_pure(grid, generation, materials))
                .collect()
        };

        // Phase C: commit outputs into store + cache + stats.
        for (out_idx, &(sub_idx, effective_phase)) in pending.iter().enumerate() {
            let (output_grid, p1ns, p2ns) = &outputs[out_idx];
            self.hashlife_stats.phase1_ns += p1ns;
            self.hashlife_stats.phase2_ns += p2ns;
            let centered = self.center_level3_grid_to_node(output_grid);
            self.hashlife_cache
                .insert((sub_roots[sub_idx], effective_phase), centered);
            resolved[sub_idx] = Some(centered);
        }

        // Resolve duplicate-miss sub_roots from their twin's freshly-
        // committed entry. Each dupe stored its `pending_index`; the
        // corresponding pending sub_root's `resolved[...]` is set by
        // Phase C above, so we copy the same NodeId across.
        for &(sub_idx, pending_idx) in &deferred_dupes {
            let original_sub = pending[pending_idx].0;
            resolved[sub_idx] = resolved[original_sub];
        }

        for i in 0..8 {
            result_children[i] =
                resolved[i].expect("step_level4_fanout_rayon: every sub_root must be resolved");
        }
    }

    /// hash-thing-ecmn (vqke.4.1): top-level dispatch picking the active
    /// `BaseCaseStrategy`. Used by `step_recursive`,
    /// `step_recursive_profiled`, and `step_margolus_only` so all three
    /// entry points honor the strategy uniformly. Reviewer feedback
    /// (Standard-Codex, Standard-Codex-Execution, Adversarial-Codex)
    /// flagged that the absence of a shared dispatcher would let the
    /// profiled path silently bypass `RayonBfs`.
    fn step_root_dispatch(
        &mut self,
        padded_root: NodeId,
        padded_level: u32,
        phase: u64,
        period: u64,
    ) -> NodeId {
        match self.base_case_strategy {
            super::world::BaseCaseStrategy::Serial
            | super::world::BaseCaseStrategy::RayonPerFanout => {
                self.step_node(padded_root, padded_level, phase, period)
            }
            super::world::BaseCaseStrategy::RayonBfs => {
                self.step_root_bfs(padded_root, padded_level, phase, period)
            }
        }
    }

    /// hash-thing-ecmn (vqke.4.1): try the short-circuit prelude that
    /// `step_node` runs at lines 650-668. Mirrors that order exactly:
    /// empty -> uniform-inert -> all-inert. Returns the resolved
    /// level-(n-1) node if any short-circuit fires, plus increments the
    /// matching skip counter; returns `None` if the caller should
    /// proceed to cache lookup. `result_level` is the level the result
    /// will live at (= input level - 1 in step_node terms).
    ///
    /// Extracted from step_node + step_level4_fanout_rayon so all three
    /// callers (step_node, the per-fanout rayon path, and step_root_bfs)
    /// share one implementation. Standard-Codex review specifically
    /// asked for this consolidation.
    fn try_resolve_short_circuit(
        &mut self,
        node: NodeId,
        result_level: u32,
    ) -> Option<NodeId> {
        if self.store.population(node) == 0 {
            self.hashlife_stats.empty_skips += 1;
            return Some(self.store.empty(result_level));
        }
        if let Some(state) = self.inert_uniform_state(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return Some(self.store.uniform(result_level, state));
        }
        if self.is_all_inert(node) {
            self.hashlife_stats.fixed_point_skips += 1;
            return Some(self.center_node(node, result_level + 1));
        }
        None
    }

    /// hash-thing-ecmn: shared fast-subtree phase fold (mirror of
    /// `step_node` lines 696-700). When the registry has slow divisors
    /// (memo_period > 2) but the subtree at `node` contains no
    /// slow-divisor cells, fold the schedule phase to `phase % 2` so
    /// fast subtrees alias across every-other-generation pairs.
    fn effective_phase_for(
        &mut self,
        node: NodeId,
        schedule_phase: u64,
        memo_period: u64,
    ) -> u64 {
        if memo_period > 2 && !self.subtree_has_slow_divisor(node) {
            schedule_phase % 2
        } else {
            schedule_phase
        }
    }

    /// hash-thing-ecmn: the env-gated phase-aliasing diagnostic probe
    /// (mirror of `step_node` lines 728-738). Always-on cost was
    /// estimated at single-digit ms/step at observed miss volumes, so
    /// the probe is gated behind `HASH_THING_MEMO_DIAG=1`.
    fn memo_diag_probe_alias(&mut self, node: NodeId, effective_phase: u64, memo_period: u64) {
        if memo_diag_enabled() && memo_period > 1 {
            for p in 0..memo_period {
                if p == effective_phase {
                    continue;
                }
                if self.hashlife_cache.contains_key(&(node, p)) {
                    self.hashlife_stats.cache_misses_phase_aliased += 1;
                    break;
                }
            }
        }
    }

    /// hash-thing-ecmn: build the 27 intermediate nodes + 8 sub-cube
    /// sub_roots at level (n-1) for a level-n parent. Mirrors
    /// `step_recursive_case` lines 853-909 exactly. Extracted so the
    /// BFS descent shares one source of truth with the DFS recursive
    /// case (a future cleanup bead can refactor `step_recursive_case`
    /// to call this helper too).
    fn build_subroots(&mut self, node: NodeId, level: u32) -> [NodeId; 8] {
        let children = self.store.children(node);
        let sub: [[NodeId; 8]; 8] = std::array::from_fn(|i| self.store.children(children[i]));

        let mut inter = [NodeId::EMPTY; 27];
        for pz in 0..3usize {
            for py in 0..3usize {
                for px in 0..3usize {
                    let mut octants = [NodeId::EMPTY; 8];
                    for sz in 0..2usize {
                        for sy in 0..2usize {
                            for sx in 0..2usize {
                                let (parent_x, sub_x) = source_index(px, sx);
                                let (parent_y, sub_y) = source_index(py, sy);
                                let (parent_z, sub_z) = source_index(pz, sz);
                                let parent_oct =
                                    octant_index(parent_x as u32, parent_y as u32, parent_z as u32);
                                let sub_oct =
                                    octant_index(sub_x as u32, sub_y as u32, sub_z as u32);
                                octants[octant_index(sx as u32, sy as u32, sz as u32)] =
                                    sub[parent_oct][sub_oct];
                            }
                        }
                    }
                    inter[px + py * 3 + pz * 9] = self.store.interior(level - 1, octants);
                }
            }
        }

        let mut centered = [NodeId::EMPTY; 27];
        for i in 0..27 {
            centered[i] = self.center_node(inter[i], level - 1);
        }

        let mut sub_roots = [NodeId::EMPTY; 8];
        for oz in 0..2usize {
            for oy in 0..2usize {
                for ox in 0..2usize {
                    let mut sub_cube = [NodeId::EMPTY; 8];
                    for dz in 0..2usize {
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                sub_cube[octant_index(dx as u32, dy as u32, dz as u32)] =
                                    centered[(ox + dx) + (oy + dy) * 3 + (oz + dz) * 9];
                            }
                        }
                    }
                    sub_roots[octant_index(ox as u32, oy as u32, oz as u32)] =
                        self.store.interior(level - 1, sub_cube);
                }
            }
        }
        sub_roots
    }

    /// hash-thing-ecmn (vqke.4.1): breadth-first whole-step level
    /// dispatcher. Walks `step_node` level-by-level instead of
    /// depth-first, collecting all level-3 base-case misses across
    /// the entire step into one large rayon batch.
    ///
    /// Three phases per call:
    ///
    /// 1. **Descending pass** (sequential, mutates store + cache):
    ///    Process the root and each level n ∈ (root_level..=4) by
    ///    building each task's 27 intermediates + 8 sub_roots, then
    ///    classifying each sub_root via the `step_node` prelude
    ///    (short-circuit / cache hit / pending miss). Pending unique
    ///    misses queue as new tasks at level (n-1); duplicates dedupe
    ///    against the per-level seen map and count as cache_hits.
    ///
    /// 2. **Level-3 batch** (parallel, pure): for every unique level-3
    ///    task, flatten the 8³ grid (sequential read), then rayon
    ///    par_iter over `step_grid_once_pure`. Below
    ///    `RAYON_BATCH_THRESHOLD` (=4) fall back to serial iter — same
    ///    threshold as ftuu's per-fanout path. Each output centers
    ///    back into a level-2 NodeId, which becomes the task's result.
    ///
    /// 3. **Ascending pass** (sequential, mutates store + cache): for n
    ///    ∈ 4..=root_level, each task at level n composes its 8 child
    ///    results (level n-2) via `store.interior(n-1, ...)` and
    ///    inserts the result into `hashlife_cache` under the same
    ///    `(node, effective_phase)` key step_node would use.
    ///
    /// Bit-exact flattened-cell parity with the serial / per-fanout
    /// paths (verified by `tests/base_case_rayon_parity.rs`). NodeIds
    /// across `World` instances may differ due to intern allocation
    /// order; the parity test compares `flatten()`, not NodeIds.
    ///
    /// `step_grid_once_pure` receives the raw `self.generation`, NOT
    /// `effective_phase` — the compute path is generation-based and
    /// only the cache key uses effective_phase. (Per Adversarial-Codex
    /// plan-review feedback.)
    ///
    /// Stats accounting matches the serial path for cache_hits /
    /// cache_misses / empty_skips / fixed_point_skips / misses_by_level
    /// counts (modulo per-event ordering, which is intentionally not
    /// part of the contract). Adds new BFS-specific counters
    /// (`bfs_tasks_by_level`, `bfs_level3_unique_misses`,
    /// `bfs_batches_parallel`, `bfs_batches_serial_fallback`,
    /// `bfs_max_batch_len`) wired through `memo_summary()`.
    fn step_root_bfs(
        &mut self,
        root: NodeId,
        root_level: u32,
        schedule_phase: u64,
        memo_period: u64,
    ) -> NodeId {
        debug_assert!(root_level >= 3, "step_root_bfs requires level >= 3");

        // Root prelude — mirrors step_node lines 650-738.
        if let Some(short) = self.try_resolve_short_circuit(root, root_level - 1) {
            return short;
        }
        let root_eff = self.effective_phase_for(root, schedule_phase, memo_period);
        let root_key = (root, root_eff);
        if let Some(&cached) = self.hashlife_cache.get(&root_key) {
            self.hashlife_stats.cache_hits += 1;
            return cached;
        }
        self.hashlife_stats.cache_misses += 1;
        self.hashlife_stats.misses_by_level[(root_level - 3) as usize] += 1;
        self.memo_diag_probe_alias(root, root_eff, memo_period);

        // Level-3 root: no BFS frontier needed — step the base case directly.
        if root_level == 3 {
            let result = self.step_base_case(root, root_eff);
            self.hashlife_cache.insert(root_key, result);
            return result;
        }

        // Per-level frontier. tasks[idx] holds the unique tasks at level
        // `(idx + 3)`. seen[idx] is the dedupe map keyed by
        // (NodeId, effective_phase) -> task index in tasks[idx].
        //
        // Cross-level dedupe is unnecessary because `Node::Interior`
        // carries its level — identical content at different levels gets
        // different NodeIds. NodeId::EMPTY-style sentinels are handled
        // by the empty short-circuit BEFORE reaching the seen/resolved
        // path.
        let max_level_idx = (root_level - 3) as usize;
        let mut tasks: Vec<Vec<BfsTask>> = (0..=max_level_idx).map(|_| Vec::new()).collect();
        let mut seen: Vec<FxHashMap<(NodeId, u64), usize>> =
            (0..=max_level_idx).map(|_| FxHashMap::default()).collect();

        // Seed root.
        tasks[max_level_idx].push(BfsTask {
            node: root,
            effective_phase: root_eff,
            children: [BfsChildSlot::Direct(NodeId::EMPTY); 8],
            result: None,
        });
        seen[max_level_idx].insert(root_key, 0);

        // Phase 1: descending pass. Walk levels root_level..=4 (level-3
        // tasks are pure leaves; they don't need sub_roots built).
        for n in (4..=root_level).rev() {
            let level_idx = (n - 3) as usize;
            let lower_idx = level_idx - 1;
            // Snapshot the count: tasks[level_idx] doesn't grow during
            // this loop (we only append to tasks[lower_idx]), so an
            // index-based loop is safe.
            let n_tasks_count = tasks[level_idx].len();
            for task_idx in 0..n_tasks_count {
                let task_node = tasks[level_idx][task_idx].node;
                let sub_roots = self.build_subroots(task_node, n);
                let lower_level = n - 1;
                let mut child_slots = [BfsChildSlot::Direct(NodeId::EMPTY); 8];
                for i in 0..8 {
                    let sub_nid = sub_roots[i];

                    // Mirror step_node prelude order: short-circuit FIRST.
                    if let Some(short) =
                        self.try_resolve_short_circuit(sub_nid, lower_level - 1)
                    {
                        child_slots[i] = BfsChildSlot::Direct(short);
                        continue;
                    }
                    let sub_eff =
                        self.effective_phase_for(sub_nid, schedule_phase, memo_period);
                    let sub_key = (sub_nid, sub_eff);

                    if let Some(&cached) = self.hashlife_cache.get(&sub_key) {
                        self.hashlife_stats.cache_hits += 1;
                        child_slots[i] = BfsChildSlot::Direct(cached);
                        continue;
                    }

                    // Dedupe against the lower-level seen map. A second
                    // sibling pointing at the same (node, eff) counts as
                    // a cache_hit (matches the per-fanout path's
                    // accounting: the serial path would have hit-cache
                    // after the first compute).
                    if let Some(&pending_idx) = seen[lower_idx].get(&sub_key) {
                        self.hashlife_stats.cache_hits += 1;
                        child_slots[i] = BfsChildSlot::Pending(pending_idx);
                        continue;
                    }

                    // Unique miss — count and queue.
                    self.hashlife_stats.cache_misses += 1;
                    self.hashlife_stats.misses_by_level[(lower_level - 3) as usize] += 1;
                    self.memo_diag_probe_alias(sub_nid, sub_eff, memo_period);

                    let new_idx = tasks[lower_idx].len();
                    tasks[lower_idx].push(BfsTask {
                        node: sub_nid,
                        effective_phase: sub_eff,
                        children: [BfsChildSlot::Direct(NodeId::EMPTY); 8],
                        result: None,
                    });
                    seen[lower_idx].insert(sub_key, new_idx);
                    child_slots[i] = BfsChildSlot::Pending(new_idx);
                }
                tasks[level_idx][task_idx].children = child_slots;
            }
        }

        // Stats: bfs_tasks_by_level (per-step, not lifetime — accumulated
        // by HashlifeStats::accumulate at the call-site of step_recursive).
        for (idx, level_tasks) in tasks.iter().enumerate() {
            self.hashlife_stats.bfs_tasks_by_level[idx] = level_tasks.len() as u64;
        }

        // Phase 2: level-3 parallel batch.
        let level3_count = tasks[0].len();
        self.hashlife_stats.bfs_level3_unique_misses = level3_count as u64;
        self.hashlife_stats.bfs_max_batch_len = level3_count as u64;

        // hash-thing-ecmn (review-pass): soft warning when the
        // unbounded BFS frontier crosses a soft sanity threshold. Plan
        // §11 documented "no cap, but soft warning if
        // bfs_level3_unique_misses > 16384". At ~1 KiB grid + ~1 KiB
        // output per task, 16384 tasks ≈ 32 MiB peak — still safe but
        // the doubling beyond that gets dangerous fast (262K tasks ≈
        // 520 MiB at 256³ worst case, multi-GiB at 1024³). When this
        // warning fires, file a chunked-wavefront follow-up bead. The
        // log only fires on opt-in BFS — Serial/RayonPerFanout never
        // reach this path.
        const BFS_FRONTIER_SOFT_LIMIT: usize = 16_384;
        if level3_count > BFS_FRONTIER_SOFT_LIMIT {
            log::warn!(
                target: "hash_thing::hashlife::bfs",
                "BFS level-3 frontier exceeded soft limit: {level3_count} unique tasks \
                 (limit {BFS_FRONTIER_SOFT_LIMIT}). Memory peak ~{} MiB pending+output. \
                 Consider chunked wavefront follow-up.",
                (level3_count * (LEVEL3_CELL_COUNT * std::mem::size_of::<CellState>() * 2))
                    / (1024 * 1024),
            );
        }

        if !tasks[0].is_empty() {
            let mut pending_grids: Vec<[CellState; LEVEL3_CELL_COUNT]> =
                Vec::with_capacity(level3_count);
            for task in &tasks[0] {
                let mut grid = [0 as CellState; LEVEL3_CELL_COUNT];
                self.store.flatten_buf(task.node, &mut grid, LEVEL3_SIDE);
                pending_grids.push(grid);
            }

            const RAYON_BATCH_THRESHOLD: usize = 4;
            // step_grid_once_pure receives raw self.generation, NOT
            // effective_phase — the rule schedule (CaRule per-cell gates,
            // BlockRule offset parity) is generation-driven; only the
            // cache key uses effective_phase. (Per Adversarial-Codex
            // plan-review feedback — restating the invariant for future
            // readers.)
            let generation = self.generation;
            let materials: &crate::terrain::materials::MaterialRegistry = &self.materials;
            let outputs: Vec<([CellState; LEVEL3_CELL_COUNT], u64, u64)> =
                if level3_count < RAYON_BATCH_THRESHOLD {
                    self.hashlife_stats.bfs_batches_serial_fallback += 1;
                    pending_grids
                        .iter()
                        .map(|grid| step_grid_once_pure(grid, generation, materials))
                        .collect()
                } else {
                    use rayon::prelude::*;
                    self.hashlife_stats.bfs_batches_parallel += 1;
                    pending_grids
                        .par_iter()
                        .map(|grid| step_grid_once_pure(grid, generation, materials))
                        .collect()
                };

            for (out_idx, (output_grid, p1ns, p2ns)) in outputs.into_iter().enumerate() {
                self.hashlife_stats.phase1_ns += p1ns;
                self.hashlife_stats.phase2_ns += p2ns;
                let centered = self.center_level3_grid_to_node(&output_grid);
                let task = &mut tasks[0][out_idx];
                self.hashlife_cache
                    .insert((task.node, task.effective_phase), centered);
                task.result = Some(centered);
            }
        }

        // Phase 3: ascend levels 4..=root_level.
        for n in 4..=root_level {
            let level_idx = (n - 3) as usize;
            let lower_idx = level_idx - 1;
            let n_tasks_count = tasks[level_idx].len();
            for task_idx in 0..n_tasks_count {
                let result_children: [NodeId; 8] =
                    std::array::from_fn(|i| match tasks[level_idx][task_idx].children[i] {
                        BfsChildSlot::Direct(id) => id,
                        BfsChildSlot::Pending(idx) => tasks[lower_idx][idx]
                            .result
                            .expect("BFS ascend: lower-level task must be resolved"),
                    });
                let composed = self.store.interior(n - 1, result_children);
                let task = &mut tasks[level_idx][task_idx];
                self.hashlife_cache
                    .insert((task.node, task.effective_phase), composed);
                task.result = Some(composed);
            }
        }

        tasks[max_level_idx][0]
            .result
            .expect("BFS root must be resolved after ascend")
    }

    fn step_recursive_case_macro(&mut self, node: NodeId, level: u32, generation: u64) -> NodeId {
        debug_assert!(level > 3, "macro recursive case requires level > 3");
        let children = self.store.children(node);
        let sub: [[NodeId; 8]; 8] = std::array::from_fn(|i| self.store.children(children[i]));

        let half_skip = 1u64 << (level - 3);

        // Phase 1: compute each overlapping level-(n-1) intermediate node's
        // center after half of the parent's time skip.
        let mut phased = [NodeId::EMPTY; 27];
        for pz in 0..3usize {
            for py in 0..3usize {
                for px in 0..3usize {
                    let mut octants = [NodeId::EMPTY; 8];
                    for sz in 0..2usize {
                        for sy in 0..2usize {
                            for sx in 0..2usize {
                                let (parent_x, sub_x) = source_index(px, sx);
                                let (parent_y, sub_y) = source_index(py, sy);
                                let (parent_z, sub_z) = source_index(pz, sz);
                                let parent_oct =
                                    octant_index(parent_x as u32, parent_y as u32, parent_z as u32);
                                let sub_oct =
                                    octant_index(sub_x as u32, sub_y as u32, sub_z as u32);
                                octants[octant_index(sx as u32, sy as u32, sz as u32)] =
                                    sub[parent_oct][sub_oct];
                            }
                        }
                    }
                    let inter = self.store.interior(level - 1, octants);
                    phased[px + py * 3 + pz * 9] =
                        self.step_node_macro(inter, level - 1, generation);
                }
            }
        }

        // Phase 2: assemble those half-stepped centers into the 8 overlapping
        // sub-cubes, then advance the remaining half-skip from the shifted
        // generation base.
        let mut result_children = [NodeId::EMPTY; 8];
        for oz in 0..2usize {
            for oy in 0..2usize {
                for ox in 0..2usize {
                    let mut sub_cube = [NodeId::EMPTY; 8];
                    for dz in 0..2usize {
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                sub_cube[octant_index(dx as u32, dy as u32, dz as u32)] =
                                    phased[(ox + dx) + (oy + dy) * 3 + (oz + dz) * 9];
                            }
                        }
                    }
                    let sub_root = self.store.interior(level - 1, sub_cube);
                    result_children[octant_index(ox as u32, oy as u32, oz as u32)] =
                        self.step_node_macro(sub_root, level - 1, generation + half_skip);
                }
            }
        }

        self.store.interior(level - 1, result_children)
    }

    /// Extract the center (level n-1) of a level-n node.
    fn center_node(&mut self, node: NodeId, level: u32) -> NodeId {
        assert!(level >= 2, "center_node requires level >= 2");
        let children = self.store.children(node);
        let mut center_children = [NodeId::EMPTY; 8];
        for oct in 0..8usize {
            let inner = 7 - oct;
            center_children[oct] = self.store.child(children[oct], inner);
        }
        self.store.interior(level - 1, center_children)
    }
}

/// Map intermediate position p ∈ {0,1,2} and sub-octant s ∈ {0,1} to
/// (parent_child_axis_index, sub_octant_axis_index).
#[inline]
fn source_index(p: usize, s: usize) -> (usize, usize) {
    match (p, s) {
        (0, 0) => (0, 0),
        (0, 1) => (0, 1),
        (1, 0) => (0, 1),
        (1, 1) => (1, 0),
        (2, 0) => (1, 0),
        (2, 1) => (1, 1),
        _ => unreachable!(),
    }
}

/// Get 26 Moore neighbors from a flat grid without bounds checking.
/// Caller must ensure `x, y, z` are all in `1..side-1` (interior cells).
/// Uses direct arithmetic instead of `rem_euclid` — safe because interior
/// cells always have valid neighbors at offsets -1, 0, +1.
#[inline]
fn get_neighbors_from_grid_unchecked(
    grid: &[CellState],
    side: usize,
    x: usize,
    y: usize,
    z: usize,
) -> [Cell; 26] {
    let mut neighbors = [Cell::EMPTY; 26];
    let mut idx = 0;
    for dz in [-1i32, 0, 1] {
        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                let nz = (z as i32 + dz) as usize;
                neighbors[idx] = Cell::from_raw(grid[nx + ny * side + nz * side * side]);
                idx += 1;
            }
        }
    }
    neighbors
}

// hash-thing-ftuu (vqke.4): pure free-function form of the level-3 base case
// step. Takes `&MaterialRegistry` instead of `&self` so it can be invoked
// concurrently from rayon worker threads without aliasing `&mut World`.
//
// Bit-exact with the prior `World::step_grid_once` body. The wrapper at
// `World::step_grid_once` forwards here unchanged so existing callers and
// timer attribution (`phase1_ns` / `phase2_ns`) stay identical.
//
// Not part of the public API — only `super::World` and the level-4 rayon
// path call into this module.
pub(super) fn step_grid_once_pure(
    grid: &[CellState],
    generation: u64,
    materials: &crate::terrain::materials::MaterialRegistry,
) -> ([CellState; LEVEL3_CELL_COUNT], u64, u64) {
    let side = LEVEL3_SIDE;
    // Phase 1: CaRule on interior cells (1..side-1 on each axis).
    // The outermost ring cannot be evolved correctly because its neighbors
    // would wrap outside the padded region. Callers only extract the center
    // that remains valid after the requested number of steps.
    let mut next = [0 as CellState; LEVEL3_CELL_COUNT];
    let phase1_start = std::time::Instant::now();
    let noop_by_material = materials.noop_flags();
    let divisor_by_material = materials.tick_divisor_flags();
    let air_is_noop = noop_by_material.first().copied().unwrap_or(false);
    for z in 1..side - 1 {
        for y in 1..side - 1 {
            for x in 1..side - 1 {
                let idx = x + y * side + z * side * side;
                let raw = grid[idx];
                if raw == 0 && air_is_noop {
                    continue;
                }
                let cell = Cell::from_raw(raw);
                let mat = cell.material() as usize;
                if mat < noop_by_material.len() && noop_by_material[mat] {
                    next[idx] = raw;
                    continue;
                }
                let divisor = divisor_by_material.get(mat).copied().unwrap_or(1) as u64;
                if divisor > 1 && !generation.is_multiple_of(divisor) {
                    next[idx] = raw;
                    continue;
                }
                let rule = materials
                    .rule_for_cell(cell)
                    .unwrap_or_else(|| panic!("missing CaRule for material {}", cell.material()));
                let neighbors = get_neighbors_from_grid_unchecked(grid, side, x, y, z);
                next[idx] = rule.step_cell(cell, &neighbors).raw();
            }
        }
    }
    let phase1_ns = phase1_start.elapsed().as_nanos() as u64;

    // Phase 2: BlockRule on aligned 2×2×2 blocks. See the comment at the
    // wrapper site for the parity / divisor schedule details.
    let phase2_start = std::time::Instant::now();
    let block_rule_divisors = materials.block_rule_tick_divisors();
    let all_divisors_one =
        block_rule_divisors.iter().all(|&d| d == 1) && divisor_by_material.iter().all(|&d| d == 1);
    if all_divisors_one {
        let offset = (generation % 2) as usize;
        let mut bz = offset;
        while bz + 1 < side - 1 {
            let mut by = offset;
            while by + 1 < side - 1 {
                let mut bx = offset;
                while bx + 1 < side - 1 {
                    apply_block_in_grid_pure(&mut next, side, bx, by, bz, materials);
                    bx += 2;
                }
                by += 2;
            }
            bz += 2;
        }
    } else {
        for pass_offset in 0..2usize {
            let mut bz = pass_offset;
            while bz + 1 < side - 1 {
                let mut by = pass_offset;
                while by + 1 < side - 1 {
                    let mut bx = pass_offset;
                    while bx + 1 < side - 1 {
                        apply_block_in_grid_with_schedule_pure(
                            &mut next,
                            side,
                            bx,
                            by,
                            bz,
                            pass_offset,
                            generation,
                            block_rule_divisors,
                            materials,
                        );
                        bx += 2;
                    }
                    by += 2;
                }
                bz += 2;
            }
        }
    }
    let phase2_ns = phase2_start.elapsed().as_nanos() as u64;
    (next, phase1_ns, phase2_ns)
}

fn apply_block_in_grid_pure(
    grid: &mut [CellState],
    side: usize,
    bx: usize,
    by: usize,
    bz: usize,
    materials: &crate::terrain::materials::MaterialRegistry,
) {
    let mut block = [Cell::EMPTY; 8];
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                block[octant_index(dx as u32, dy as u32, dz as u32)] = Cell::from_raw(grid[idx]);
            }
        }
    }

    if block.iter().all(|c| c.is_empty()) {
        return;
    }

    let rule_id = match unique_block_rule_pure(materials, &block) {
        Some(id) => id,
        None => return,
    };

    let movable: [bool; 8] = std::array::from_fn(|i| {
        let c = block[i];
        c.is_empty() || materials.block_rule_id_for_cell(c).is_some()
    });

    let rule = materials.block_rule(rule_id);
    let result = rule.step_block(&block, &movable);

    debug_assert!(
        (0..8).all(|i| movable[i] || result[i] == block[i]),
        "block rule moved an immovable cell"
    );

    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let i = octant_index(dx as u32, dy as u32, dz as u32);
                let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                if movable[i] {
                    grid[idx] = result[i].raw();
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_block_in_grid_with_schedule_pure(
    grid: &mut [CellState],
    side: usize,
    bx: usize,
    by: usize,
    bz: usize,
    pass_offset: usize,
    generation: u64,
    block_rule_divisors: &[u16],
    materials: &crate::terrain::materials::MaterialRegistry,
) {
    let mut block = [Cell::EMPTY; 8];
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                block[octant_index(dx as u32, dy as u32, dz as u32)] = Cell::from_raw(grid[idx]);
            }
        }
    }

    if block.iter().all(|c| c.is_empty()) {
        return;
    }

    let rule_id = match unique_block_rule_pure(materials, &block) {
        Some(id) => id,
        None => return,
    };

    let divisor = block_rule_divisors
        .get(rule_id.0)
        .copied()
        .unwrap_or(1)
        .max(1) as u64;
    if !generation.is_multiple_of(divisor) {
        return;
    }
    let rule_offset = ((generation / divisor) & 1) as usize;
    if rule_offset != pass_offset {
        return;
    }

    let movable: [bool; 8] = std::array::from_fn(|i| {
        let c = block[i];
        c.is_empty() || materials.block_rule_id_for_cell(c).is_some()
    });

    let rule = materials.block_rule(rule_id);
    let result = rule.step_block(&block, &movable);

    debug_assert!(
        (0..8).all(|i| movable[i] || result[i] == block[i]),
        "block rule moved an immovable cell"
    );

    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let i = octant_index(dx as u32, dy as u32, dz as u32);
                let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
                if movable[i] {
                    grid[idx] = result[i].raw();
                }
            }
        }
    }
}

fn unique_block_rule_pure(
    materials: &crate::terrain::materials::MaterialRegistry,
    block: &[Cell; 8],
) -> Option<crate::terrain::materials::BlockRuleId> {
    let mut found: Option<crate::terrain::materials::BlockRuleId> = None;
    for cell in block {
        if let Some(id) = materials.block_rule_id_for_cell(*cell) {
            match found {
                None => found = Some(id),
                Some(existing) if existing == id => {}
                Some(_) => return None,
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::rule::ALIVE;
    use crate::sim::{GameOfLife3D, WorldCoord};
    use crate::terrain::materials::{DIRT_MATERIAL_ID, SAND_MATERIAL_ID, STONE, WATER_MATERIAL_ID};

    fn wc(coord: u64) -> WorldCoord {
        WorldCoord(coord as i64)
    }

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
    }

    fn gol_world(level: u32, rule: GameOfLife3D, simulation_seed: u64) -> World {
        let mut world = World::new(level);
        world.simulation_seed = simulation_seed;
        world.set_gol_smoke_rule(rule);
        world
    }

    fn seed_random_alive_cells(world: &mut World, seed: u64, margin: u64) {
        let mut rng = SimpleRng::new(seed);
        let side = world.side() as u64;
        for z in margin..(side - margin) {
            for y in margin..(side - margin) {
                for x in margin..(side - margin) {
                    if rng.next_u64().is_multiple_of(5) {
                        world.set(wc(x), wc(y), wc(z), ALIVE.raw());
                    }
                }
            }
        }
    }

    fn assert_recursive_matches_bruteforce(
        mut brute: World,
        mut recur: World,
        steps: usize,
        label: &str,
    ) {
        assert_eq!(
            brute.flatten(),
            recur.flatten(),
            "{label}: initial state mismatch"
        );
        for step in 0..steps {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "{label}: mismatch at generation {} (step {})",
                brute.generation,
                step
            );
            assert_eq!(
                brute.generation, recur.generation,
                "{label}: generation counter diverged"
            );
        }
    }

    fn embed_world_in_center(source: &World, target: &mut World) {
        let source_side = source.side() as u64;
        let target_side = target.side() as u64;
        let offset = (target_side - source_side) / 2;
        let grid = source.flatten();

        for z in 0..source_side {
            for y in 0..source_side {
                for x in 0..source_side {
                    let idx = x as usize
                        + y as usize * source_side as usize
                        + z as usize * source_side as usize * source_side as usize;
                    let state = grid[idx];
                    if state != 0 {
                        target.set(wc(offset + x), wc(offset + y), wc(offset + z), state);
                    }
                }
            }
        }
    }

    fn extract_center_cube(world: &World, side: u64) -> Vec<CellState> {
        let world_side = world.side() as u64;
        let offset = (world_side - side) / 2;
        let mut grid = vec![0; (side * side * side) as usize];

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let idx = x as usize
                        + y as usize * side as usize
                        + z as usize * side as usize * side as usize;
                    grid[idx] = world.get(wc(offset + x), wc(offset + y), wc(offset + z));
                }
            }
        }

        grid
    }

    fn seed_random_material_cells(world: &mut World, seed: u64) {
        let mut rng = SimpleRng::new(seed);
        let side = world.side() as u64;
        let water = crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw();

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let roll = rng.next_u64() % 7;
                    let state = match roll {
                        0 | 1 => water,
                        2 => STONE,
                        _ => 0,
                    };
                    if state != 0 {
                        world.set(wc(x), wc(y), wc(z), state);
                    }
                }
            }
        }
    }
    #[test]
    fn recursive_matches_brute_force_empty() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        brute.step();
        recur.step_recursive();
        assert_eq!(brute.flatten(), recur.flatten());
        assert_eq!(brute.generation, recur.generation);
    }

    #[test]
    fn recursive_matches_brute_force_single_cell() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        brute.set(wc(3), wc(3), wc(3), STONE);
        recur.set(wc(3), wc(3), wc(3), STONE);
        brute.step();
        recur.step_recursive();
        assert_eq!(brute.flatten(), recur.flatten());
    }

    #[test]
    fn recursive_matches_brute_force_multiple_steps() {
        // Level 4 (16³) so cells stay well inside the absorbing boundary
        // over 4 steps. Both brute-force and hashlife now use absorbing BC
        // (OOB = empty), but level 4 gives more room for cells to move.
        let mut brute = World::new(4);
        let mut recur = World::new(4);
        for &(x, y, z, mat) in &[
            (6u64, 6, 6, STONE),
            (
                7,
                7,
                7,
                crate::octree::Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
            ),
            (
                8,
                8,
                8,
                crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw(),
            ),
        ] {
            brute.set(wc(x), wc(y), wc(z), mat);
            recur.set(wc(x), wc(y), wc(z), mat);
        }
        for step in 0..4 {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "mismatch at generation {} (step {})",
                brute.generation,
                step
            );
        }
    }

    #[test]
    fn recursive_matches_brute_force_carule_only() {
        let mut brute = World::new(3);
        let mut recur = World::new(3);
        for &(x, y, z) in &[
            (1u64, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            (4, 4, 4),
            (5, 5, 5),
            (6, 6, 6),
        ] {
            brute.set(wc(x), wc(y), wc(z), STONE);
            recur.set(wc(x), wc(y), wc(z), STONE);
        }
        for step in 0..6 {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "CaRule-only mismatch at generation {} (step {})",
                brute.generation,
                step
            );
        }
    }

    #[test]
    fn recursive_matches_brute_force_with_terrain() {
        use crate::terrain::TerrainParams;
        let mut brute = World::new(4);
        let mut recur = World::new(4);
        let params = TerrainParams::default();
        brute.seed_terrain(&params).unwrap();
        recur.seed_terrain(&params).unwrap();
        assert_eq!(brute.flatten(), recur.flatten(), "initial state must match");
        for _ in 0..2 {
            brute.step();
            recur.step_recursive();
            assert_eq!(
                brute.flatten(),
                recur.flatten(),
                "mismatch at generation {}",
                brute.generation
            );
        }
    }

    #[test]
    fn recursive_matches_brute_force_randomized_gol_presets() {
        let presets = [
            ("amoeba", GameOfLife3D::new(9, 26, 5, 7)),
            ("crystal", GameOfLife3D::new(0, 6, 1, 3)),
            ("445", GameOfLife3D::rule445()),
            ("pyroclastic", GameOfLife3D::new(4, 7, 6, 8)),
        ];
        let cases = [(3_u32, 1_usize), (4_u32, 3_usize)];

        for (preset_idx, &(label, rule)) in presets.iter().enumerate() {
            for (level, steps) in cases {
                for case_seed in 0..4_u64 {
                    let simulation_seed = 0x6f03_u64
                        ^ ((preset_idx as u64) << 16)
                        ^ ((level as u64) << 8)
                        ^ case_seed;
                    let initial_seed = simulation_seed ^ 0xa11ce_u64;
                    let mut brute = gol_world(level, rule, simulation_seed);
                    let mut recur = gol_world(level, rule, simulation_seed);
                    seed_random_alive_cells(&mut brute, initial_seed, 0);
                    seed_random_alive_cells(&mut recur, initial_seed, 0);

                    let case =
                        format!("{label} level={level} steps={steps} seed={simulation_seed:#x}");
                    assert_recursive_matches_bruteforce(brute, recur, steps, &case);
                }
            }
        }
    }

    #[test]
    fn recursive_pow2_matches_centered_bruteforce_reference() {
        let presets = [
            ("amoeba", GameOfLife3D::new(9, 26, 5, 7)),
            ("crystal", GameOfLife3D::new(0, 6, 1, 3)),
            ("445", GameOfLife3D::rule445()),
            ("pyroclastic", GameOfLife3D::new(4, 7, 6, 8)),
        ];

        for (preset_idx, &(label, rule)) in presets.iter().enumerate() {
            for level in [3_u32, 4_u32] {
                let simulation_seed = 0x6f07_u64 ^ ((preset_idx as u64) << 16) ^ (level as u64);
                let mut fast = gol_world(level, rule, simulation_seed);
                let mut brute = gol_world(level + 1, rule, simulation_seed);
                seed_random_alive_cells(&mut fast, simulation_seed ^ 0xfeed_u64, 0);
                embed_world_in_center(&fast, &mut brute);

                let skip = fast.recursive_pow2_step_count();
                for _ in 0..skip {
                    brute.step();
                }
                fast.step_recursive_pow2();

                assert_eq!(
                    fast.flatten(),
                    extract_center_cube(&brute, fast.side() as u64),
                    "{label} level={level} skip={skip}: macro-step mismatch"
                );
                assert_eq!(
                    fast.generation, skip,
                    "{label} level={level}: generation should advance by the macro skip"
                );
            }
        }
    }

    #[test]
    fn recursive_pow2_tracks_nonzero_generation_for_material_rules() {
        let mut fast = World::new(3);
        let mut expected = World::new(3);
        fast.generation = 1;
        expected.generation = 1;
        seed_random_material_cells(&mut fast, 0x5eed_u64);
        seed_random_material_cells(&mut expected, 0x5eed_u64);

        let skip = fast.recursive_pow2_step_count();
        for _ in 0..skip {
            expected.step();
        }
        fast.step_recursive_pow2();

        assert_eq!(
            fast.flatten(),
            expected.flatten(),
            "material pow2 fallback should match repeated brute-force stepping"
        );
        assert_eq!(fast.generation, 1 + skip);
    }

    #[test]
    fn recursive_conserves_population_with_stone() {
        let mut world = World::new(3);
        world.set(wc(3), wc(3), wc(3), STONE);
        world.set(wc(4), wc(4), wc(4), STONE);
        let pop_before = world.population();
        world.step_recursive();
        assert_eq!(world.population(), pop_before);
    }

    /// hash-thing-4497 investigation: measure the actual divergence
    /// between `step_node_macro` (the macro path) and brute-force
    /// stepping on a world with block-rule cells. The fallback at
    /// step_recursive_pow2:100 currently bypasses the macro path
    /// whenever block-rule cells are present; this test bypasses the
    /// fallback to exercise the macro path directly and report on the
    /// nature of any divergence.
    ///
    /// Expected outcome (hypothesis from the 4497 analysis):
    /// the macro path IS sound w.r.t. Margolus block-rule partition
    /// alignment (sub-cube origins are always at even parent coords),
    /// but it lacks the per-generation `gravity_gap_fill` that
    /// brute-force `step()` applies at phase 3. So the divergence
    /// should be entirely in the gap-fill-affected cells — vertical
    /// rarefaction under water columns — NOT in the block-rule
    /// partition boundaries themselves.
    ///
    /// This is an investigation probe, not a ship assertion: it
    /// prints the divergence rather than failing, so future runs
    /// capture the behavior without pinning it.
    #[test]
    fn investigation_4497_macro_vs_brute_with_block_rules() {
        let level = 4u32;
        let mut macro_world = World::new(level);
        let mut brute = World::new(level);

        // Seed identical water-heavy worlds. Water carries both CaRule
        // (flow) and BlockRule (horizontal spread) + gravity_gap_fill
        // eligibility, so it exercises all three phases that diverge
        // between the two paths.
        seed_random_material_cells(&mut macro_world, 0x4497_u64);
        seed_random_material_cells(&mut brute, 0x4497_u64);
        assert_eq!(macro_world.flatten(), brute.flatten());

        let steps = macro_world.recursive_pow2_step_count();
        for _ in 0..steps {
            brute.step();
        }

        // Run the macro path directly, bypassing the has_block_rule_cells
        // guard at step_recursive_pow2:100. Mirrors the guard-free
        // branch of step_recursive_pow2 exactly.
        let padded_root = macro_world.pad_root();
        let padded_level = macro_world.level + 1;
        let result = macro_world.step_node_macro(padded_root, padded_level, macro_world.generation);
        macro_world.root = result;
        macro_world.generation += steps;

        let macro_flat = macro_world.flatten();
        let brute_flat = brute.flatten();
        let side = macro_world.side();
        assert_eq!(macro_flat.len(), brute_flat.len());

        let mut diff_count = 0usize;
        let mut diff_with_water_present = 0usize;
        for (i, (&m, &b)) in macro_flat.iter().zip(brute_flat.iter()).enumerate() {
            if m != b {
                diff_count += 1;
                let water_raw = crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw();
                if m == water_raw || b == water_raw {
                    diff_with_water_present += 1;
                }
                if diff_count <= 8 {
                    let z = i / (side * side);
                    let y = (i / side) % side;
                    let x = i % side;
                    eprintln!("divergence at ({x},{y},{z}): macro={m:#010x} brute={b:#010x}");
                }
            }
        }
        let total = macro_flat.len();
        eprintln!(
            "4497 probe: level={level} steps={steps} total_cells={total} \
             diffs={diff_count} ({:.2}%) water_touching_diffs={diff_with_water_present}",
            (diff_count as f64 / total as f64) * 100.0
        );
    }

    /// hash-thing-gzio experiment: measure divergence between brute-force
    /// stepping (per-generation gap-fill, the ship baseline) and the
    /// candidate replacement — step_node_macro followed by a SINGLE
    /// gravity_gap_fill at the end of the whole pow2 superstep. The
    /// hypothesis is that most of the 4497 probe's ~16% divergence is
    /// per-generation gap-fill that settles back out under one end-of-
    /// superstep gap-fill, so the post-gap_fill divergence number should
    /// be much smaller than the pre-gap_fill number.
    ///
    /// If this bears out, step_recursive_pow2 could replace its
    /// per-generation brute-force fallback with step_node_macro +
    /// one gap_fill — keeping the O(2^n) macro speedup while approximately
    /// preserving ship-path gap-fill semantics.
    ///
    /// Investigation probe, not a ship assertion: it prints divergence
    /// (pre vs post gap-fill) so future runs can capture the behavior
    /// without pinning it.
    #[test]
    fn investigation_gzio_macro_plus_single_gap_fill_vs_brute() {
        let level = 4u32;
        let mut macro_world = World::new(level);
        let mut brute = World::new(level);

        seed_random_material_cells(&mut macro_world, 0x4497_u64);
        seed_random_material_cells(&mut brute, 0x4497_u64);
        assert_eq!(macro_world.flatten(), brute.flatten());

        let steps = macro_world.recursive_pow2_step_count();
        for _ in 0..steps {
            brute.step();
        }

        // Macro path, bypassing the has_block_rule_cells guard.
        let padded_root = macro_world.pad_root();
        let padded_level = macro_world.level + 1;
        let result = macro_world.step_node_macro(padded_root, padded_level, macro_world.generation);
        macro_world.root = result;
        macro_world.generation += steps;

        let brute_flat = brute.flatten();
        let side = macro_world.side();

        // Measure pre-gap-fill divergence — should match the 4497 probe.
        let pre_flat = macro_world.flatten();
        let mut pre_diffs = 0usize;
        for (&m, &b) in pre_flat.iter().zip(brute_flat.iter()) {
            if m != b {
                pre_diffs += 1;
            }
        }

        // Apply a SINGLE post-macro gravity_gap_fill and re-measure.
        macro_world.apply_gap_fill_via_flatten_rebuild();

        let post_flat = macro_world.flatten();
        let mut post_diffs = 0usize;
        let mut post_water_diffs = 0usize;
        for (i, (&m, &b)) in post_flat.iter().zip(brute_flat.iter()).enumerate() {
            if m != b {
                post_diffs += 1;
                let water_raw = crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw();
                if m == water_raw || b == water_raw {
                    post_water_diffs += 1;
                }
                if post_diffs <= 8 {
                    let z = i / (side * side);
                    let y = (i / side) % side;
                    let x = i % side;
                    eprintln!(
                        "post-gap-fill divergence at ({x},{y},{z}): \
                         macro+gf={m:#010x} brute={b:#010x}"
                    );
                }
            }
        }

        let total = post_flat.len();
        eprintln!(
            "gzio probe: level={level} steps={steps} total_cells={total}\n\
             \tpre_gap_fill_diffs={pre_diffs} ({:.2}%)\n\
             \tpost_gap_fill_diffs={post_diffs} ({:.2}%) \
             water_touching={post_water_diffs}",
            (pre_diffs as f64 / total as f64) * 100.0,
            (post_diffs as f64 / total as f64) * 100.0,
        );
    }

    #[test]
    fn pad_root_preserves_center() {
        let mut world = World::new(3);
        world.set(wc(2), wc(3), wc(4), STONE);
        let flat_before = world.flatten();
        let padded = world.pad_root();
        let padded_side = 16usize;
        let padded_grid = world.store.flatten(padded, padded_side);
        let orig_side = 8usize;
        for z in 0..orig_side {
            for y in 0..orig_side {
                for x in 0..orig_side {
                    let orig_val = flat_before[x + y * orig_side + z * orig_side * orig_side];
                    let (px, py, pz) = (x + 4, y + 4, z + 4);
                    let pad_val =
                        padded_grid[px + py * padded_side + pz * padded_side * padded_side];
                    assert_eq!(orig_val, pad_val, "center mismatch at ({x},{y},{z})");
                }
            }
        }
        for z in 0..padded_side {
            for y in 0..padded_side {
                for x in 0..padded_side {
                    if (4..12).contains(&x) && (4..12).contains(&y) && (4..12).contains(&z) {
                        continue;
                    }
                    let val = padded_grid[x + y * padded_side + z * padded_side * padded_side];
                    assert_eq!(val, 0, "padding not empty at ({x},{y},{z}): {val}");
                }
            }
        }
    }

    /// Timing comparison: brute-force vs recursive Hashlife on 64³ terrain.
    /// Run with `cargo test --release --lib bench_stepper_comparison -- --ignored --nocapture`.
    /// At 64³ the recursive path is roughly at parity with brute-force thanks to
    /// the empty-node short-circuit (6gf.14) and incremental cache (m1f.11/m1f.12).
    #[test]
    #[ignore]
    fn bench_stepper_comparison() {
        use crate::terrain::TerrainParams;
        use std::time::Instant;

        let steps = 10;
        let level = 6; // 64³

        // Set up identical worlds with terrain.
        let params = TerrainParams::default();
        let mut brute = World::new(level);
        let mut recur = World::new(level);
        brute.seed_terrain(&params).unwrap();
        recur.seed_terrain(&params).unwrap();

        // Brute-force timing.
        let t0 = Instant::now();
        for _ in 0..steps {
            brute.step();
        }
        let brute_ms = t0.elapsed().as_millis();

        // Recursive timing.
        let t0 = Instant::now();
        for _ in 0..steps {
            recur.step_recursive();
        }
        let recur_ms = t0.elapsed().as_millis();

        // Correctness check.
        assert_eq!(brute.flatten(), recur.flatten(), "results diverged");

        eprintln!(
            "64³ terrain × {steps} steps: brute={brute_ms}ms, recursive={recur_ms}ms, \
             ratio={:.2}x",
            brute_ms as f64 / recur_ms.max(1) as f64
        );
    }

    #[test]
    fn center_node_extracts_inner_octants() {
        let mut world = World::new(3);
        // Place stone throughout so we have distinct subtrees.
        world.set(wc(3), wc(3), wc(3), STONE);
        world.set(wc(4), wc(4), wc(4), STONE);
        let center = world.center_node(world.root, 3);
        let center_grid = world.store.flatten(center, 4);
        for z in 0..4usize {
            for y in 0..4usize {
                for x in 0..4usize {
                    let expected = world.store.get_cell(
                        world.root,
                        (x + 2) as u64,
                        (y + 2) as u64,
                        (z + 2) as u64,
                    );
                    let actual = center_grid[x + y * 4 + z * 16];
                    assert_eq!(expected, actual, "center mismatch at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn recursive_matches_brute_force_level5_stone_only() {
        // Level 5 = 32³ with inert stone only (no BlockRule).
        // Exercises deep recursion without Margolus complexity.
        let mut brute = World::new(5);
        let mut recur = World::new(5);
        let mut rng = SimpleRng::new(0xdee5_u64);
        for z in 0..32u64 {
            for y in 0..32u64 {
                for x in 0..32u64 {
                    if rng.next_u64().is_multiple_of(4) {
                        brute.set(wc(x), wc(y), wc(z), STONE);
                        recur.set(wc(x), wc(y), wc(z), STONE);
                    }
                }
            }
        }
        assert_recursive_matches_bruteforce(brute, recur, 2, "level5-stone");
    }

    /// Seed water and stone with a margin. CaRule boundaries now match
    /// (both absorbing), but BlockRule still differs: brute-force clips
    /// partial blocks at edges, hashlife processes them via overlapping
    /// sub-cubes. Margins keep block-rule-bearing cells away from edges.
    fn seed_random_material_cells_margined(world: &mut World, seed: u64, margin: u64) {
        let mut rng = SimpleRng::new(seed);
        let side = world.side() as u64;
        let water = crate::octree::Cell::pack(WATER_MATERIAL_ID, 0).raw();
        for z in margin..(side - margin) {
            for y in margin..(side - margin) {
                for x in margin..(side - margin) {
                    let roll = rng.next_u64() % 7;
                    let state = match roll {
                        0 | 1 => water,
                        2 => STONE,
                        _ => 0,
                    };
                    if state != 0 {
                        world.set(wc(x), wc(y), wc(z), state);
                    }
                }
            }
        }
    }

    #[test]
    fn recursive_matches_brute_force_level4_materials() {
        let mut brute = World::new(4);
        let mut recur = World::new(4);
        seed_random_material_cells_margined(&mut brute, 0xdee4_u64, 2);
        seed_random_material_cells_margined(&mut recur, 0xdee4_u64, 2);
        assert_recursive_matches_bruteforce(brute, recur, 3, "level4-materials");
    }

    #[test]
    fn recursive_matches_brute_force_level5_materials() {
        let mut brute = World::new(5);
        let mut recur = World::new(5);
        seed_random_material_cells_margined(&mut brute, 0xdee5_u64, 3);
        seed_random_material_cells_margined(&mut recur, 0xdee5_u64, 3);
        assert_recursive_matches_bruteforce(brute, recur, 2, "level5-materials");
    }

    #[test]
    fn recursive_matches_brute_force_level5_gol() {
        // GoL at level 5 — the CaRule-only path through deep recursion
        let rule = GameOfLife3D::rule445();
        let mut brute = gol_world(5, rule, 0xd33f_u64);
        let mut recur = gol_world(5, rule, 0xd33f_u64);
        seed_random_alive_cells(&mut brute, 0xd33f_u64 ^ 0xa11ce, 0);
        seed_random_alive_cells(&mut recur, 0xd33f_u64 ^ 0xa11ce, 0);
        assert_recursive_matches_bruteforce(brute, recur, 2, "level5-gol-445");
    }

    #[test]
    fn source_index_all_valid_pairs() {
        assert_eq!(source_index(0, 0), (0, 0));
        assert_eq!(source_index(0, 1), (0, 1));
        assert_eq!(source_index(1, 0), (0, 1));
        assert_eq!(source_index(1, 1), (1, 0));
        assert_eq!(source_index(2, 0), (1, 0));
        assert_eq!(source_index(2, 1), (1, 1));
    }

    #[test]
    #[should_panic]
    fn source_index_out_of_range_panics() {
        source_index(3, 0);
    }

    /// Regression test for u4w: water column must conserve mass and spread
    /// symmetrically (no directional bias). Before the FluidBlockRule fix,
    /// water drifted systematically +x due to single-axis selection bias.
    /// Now water uses FluidBlockRule with both-axis spread: lateral movement
    /// is expected but must be mass-conserving and approximately symmetric.
    #[test]
    fn water_column_mass_conservation() {
        use crate::terrain::materials::WATER_MATERIAL_ID;
        let mut world = World::new(5); // 32³
                                       // Place a column of water at (16, y, 16) for y in 8..24.
        for y in 8..24 {
            world.set(
                wc(16),
                wc(y),
                wc(16),
                Cell::pack(WATER_MATERIAL_ID, 0).raw(),
            );
        }
        let initial_water = count_material(&world, 32, WATER_MATERIAL_ID);
        assert_eq!(initial_water, 16);

        for step in 0..8 {
            world.step_recursive();
            let water_count = count_material(&world, 32, WATER_MATERIAL_ID);
            eprintln!("step {step}: water_count={water_count}");
            // Mass must be conserved.
            assert_eq!(
                water_count, initial_water,
                "mass not conserved at step {step}: expected {initial_water}, got {water_count}"
            );
        }
    }

    /// Count cells of a given material in the world.
    fn count_material(world: &World, side: i64, material_id: u16) -> usize {
        let mut count = 0;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let cell =
                        Cell::from_raw(world.get(WorldCoord(x), WorldCoord(y), WorldCoord(z)));
                    if cell.material() == material_id {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// tick_divisor=2 on WATER means the BlockRule fires every other tick.
    /// The gate is "rule fires iff generation % divisor == 0", so with
    /// divisor=2 it fires at gen=0, 2, 4, ... (pre-step generation) and skips
    /// at gen=1, 3, 5. Over 4 steps that's 2 firings and 2 skips, so a
    /// falling water cell should move at most 2 levels rather than 4.
    #[test]
    fn tick_divisor_two_halves_water_block_rule_firing_cadence() {
        let mut fast = World::new(4); // 16³, every tick
        let mut slow = World::new(4); // 16³, every other tick
                                      // rvsh: terrain_defaults now ships water at divisor=2, so `fast`
                                      // must explicitly reset to 1 to keep this test a d=1 vs d=2 contrast
                                      // independent of the registry's default pick.
        fast.mutate_materials(|m| m.set_tick_divisor(WATER_MATERIAL_ID, 1));
        slow.mutate_materials(|m| m.set_tick_divisor(WATER_MATERIAL_ID, 2));

        let water = Cell::pack(WATER_MATERIAL_ID, 0).raw();
        // Water near the top of the world with clear air below.
        for world in [&mut fast, &mut slow] {
            world.set(wc(8), wc(12), wc(8), water);
        }

        // After 1 step the slow world is pre-step gen=0, so it fires identically.
        fast.step_recursive();
        slow.step_recursive();
        assert_eq!(
            fast.flatten(),
            slow.flatten(),
            "gen 0→1: both fire identically"
        );

        // After 2 steps the slow world skips its second firing (pre-step gen=1).
        // Fast world keeps falling; slow world is now one step behind.
        fast.step_recursive();
        let slow_snapshot = slow.flatten();
        slow.step_recursive();
        assert_eq!(
            slow.flatten(),
            slow_snapshot,
            "gen 1→2: slow world must be a no-op under divisor=2"
        );
        assert_ne!(
            fast.flatten(),
            slow.flatten(),
            "fast world must have advanced past slow world"
        );

        // After 3 steps slow fires again (pre-step gen=2); fast keeps falling.
        // Slow should now have 2 firings total; fast had 3. Capture fast's
        // 2-firing state before its 3rd step, so we can assert slow catches
        // up to it after its 2nd firing.
        let fast_after_2_firings = fast.flatten();
        fast.step_recursive();
        slow.step_recursive();
        assert_eq!(
            slow.flatten(),
            fast_after_2_firings,
            "slow world after 2 firings must match fast world after 2 firings \
             (1 firing behind fast's 3 firings)"
        );
    }

    /// T1 (iowh review): CaRule-only divisor gating. FIRE has a CaRule
    /// (FanDrivenRule wrapping FireRule) but no BlockRule, so it exercises the
    /// CaRule gate at `src/sim/hashlife.rs:461` in isolation from the BlockRule
    /// path. A lone fire cell with no fuel disappears on the next rule firing.
    /// With divisor=2 the firing happens at gen 0, 2, 4, ... and is skipped
    /// at 1, 3, 5, ... so a fire cell seeded at gen=1 must survive one step.
    #[test]
    fn tick_divisor_carule_only_fire_gate() {
        use crate::terrain::materials::FIRE_MATERIAL_ID;
        let mut fast = World::new(3); // 8³, divisor=1 (default)
        let mut slow = World::new(3); // 8³, divisor=2 for fire
        slow.mutate_materials(|m| m.set_tick_divisor(FIRE_MATERIAL_ID, 2));

        let fire = Cell::pack(FIRE_MATERIAL_ID, 0).raw();

        // Step 1 (pre-gen=0): both worlds fire the CaRule (0 % 2 == 0 for slow).
        // Seed a lone fire cell (no fuel), both worlds should kill it.
        for w in [&mut fast, &mut slow] {
            w.set(wc(4), wc(4), wc(4), fire);
        }
        fast.step_recursive();
        slow.step_recursive();
        assert_eq!(fast.generation, 1);
        assert_eq!(slow.generation, 1);

        // Re-seed fire at gen=1. Step once more.
        // Fast (d=1): CaRule fires at gen=1 → fire dies.
        // Slow (d=2): CaRule skipped at gen=1 (1 % 2 == 1) → fire stays.
        for w in [&mut fast, &mut slow] {
            w.set(wc(4), wc(4), wc(4), fire);
        }
        fast.step_recursive();
        slow.step_recursive();

        let fast_cell = Cell::from_raw(fast.get(wc(4), wc(4), wc(4)));
        let slow_cell = Cell::from_raw(slow.get(wc(4), wc(4), wc(4)));
        assert!(
            fast_cell.is_empty(),
            "fast world fires CaRule at gen=1 → fire must die"
        );
        assert_eq!(
            slow_cell.material(),
            FIRE_MATERIAL_ID,
            "slow world skips CaRule at gen=1 under divisor=2 → fire must persist"
        );
    }

    /// T2 (iowh review): Margolus alternation under a slow tick schedule
    /// preserves mass conservation. The alternation formula
    /// `(generation / divisor) & 1` means that over the rule's own firing
    /// cadence, offsets alternate 0,1,0,1 — not over the raw generation
    /// counter (which would pin the rule to offset 0 forever when d > 1).
    /// Regression: water with divisor=2 running many generations must
    /// conserve mass exactly the way divisor=1 water does.
    #[test]
    fn tick_divisor_two_preserves_water_mass_conservation() {
        use crate::terrain::materials::WATER_MATERIAL_ID;
        let mut world = World::new(5); // 32³
        world.mutate_materials(|m| m.set_tick_divisor(WATER_MATERIAL_ID, 2));
        for y in 8..24 {
            world.set(
                wc(16),
                wc(y),
                wc(16),
                Cell::pack(WATER_MATERIAL_ID, 0).raw(),
            );
        }
        let initial = count_material(&world, 32, WATER_MATERIAL_ID);
        assert_eq!(initial, 16);

        for step in 0..16 {
            world.step_recursive();
            let n = count_material(&world, 32, WATER_MATERIAL_ID);
            assert_eq!(
                n, initial,
                "mass lost at step {step} under divisor=2: expected {initial}, got {n}"
            );
        }
    }

    /// Memo soundness: a world stepped forward with prior cache entries must
    /// match a fresh world with no cache stepped to the same generation.
    /// Runs with a non-default divisor to exercise the expanded memo period
    /// key (T=4 when water d=2, vs T=2 baseline). If the cache key were too
    /// narrow, the cached world would return a stale result.
    #[test]
    fn memo_soundness_divisor_two_matches_fresh_step() {
        const STEPS: usize = 6;
        const LEVEL: u32 = 4;
        const WATER_COL_X: u64 = 8;
        const WATER_COL_Z: u64 = 8;
        const WATER_COL_Y_RANGE: std::ops::Range<u64> = 4..12;

        let seed_world = || {
            let mut w = World::new(LEVEL);
            w.mutate_materials(|m| m.set_tick_divisor(WATER_MATERIAL_ID, 2));
            for y in WATER_COL_Y_RANGE {
                w.set(
                    wc(WATER_COL_X),
                    wc(y),
                    wc(WATER_COL_Z),
                    Cell::pack(WATER_MATERIAL_ID, 0).raw(),
                );
            }
            w
        };

        // Warm cache: step one world for STEPS generations, then step once more.
        let mut warm = seed_world();
        for _ in 0..STEPS {
            warm.step_recursive();
        }
        let warm_before = warm.flatten();
        warm.step_recursive();
        let warm_after = warm.flatten();

        // Fresh cache: step a NEW world the same number of steps, then once more.
        // Its final `step_recursive` call should see no prior cache entries for
        // the intermediate node shapes (the seed is deterministic, but the
        // cache was never populated).
        let mut fresh = seed_world();
        for _ in 0..STEPS {
            fresh.step_recursive();
        }
        let fresh_before = fresh.flatten();
        assert_eq!(
            fresh_before, warm_before,
            "deterministic seed must produce identical state after {STEPS} steps"
        );
        fresh.step_recursive();
        let fresh_after = fresh.flatten();

        assert_eq!(
            warm_after, fresh_after,
            "cached step must match fresh step at generation {STEPS} + 1"
        );
    }

    // ---- hash-thing-jw3k.1 cross-check: column-path == flatten-rebuild ----

    /// Same-store hash-cons regression for the new column-local gap-fill path.
    /// For each generation we snapshot the pre-gap-fill root, run the column
    /// path, capture its root, reset to the snapshot, run the legacy
    /// flatten/gap_fill/from_flat path on the same store, and assert the two
    /// resulting roots are identical NodeIds. Same-store canonicality means
    /// NodeId equality is sufficient — two hash-consed DAGs that encode the
    /// same grid must land on the same id in the same store.
    #[test]
    fn step_recursive_column_gap_fill_matches_flatten_rebuild_per_step() {
        use crate::terrain::TerrainParams;

        let mut world = World::new(7);
        let params = TerrainParams::for_level(7);
        world.seed_terrain(&params).unwrap();
        world.seed_water_and_sand();

        const GENS: u32 = 16;
        for gen in 0..GENS {
            world.step_margolus_only();
            let pre_root = world.root;

            world.apply_gap_fill_column_path();
            let root_via_column = world.root;

            world.root = pre_root;
            world.apply_gap_fill_via_flatten_rebuild();
            let root_via_flatten = world.root;

            assert_eq!(
                root_via_column, root_via_flatten,
                "gen {gen}: column-path root must match flatten-rebuild root \
                 (same-store hash-cons NodeId equality)",
            );

            world.root = root_via_column;
            world.finalize_step_after_external_gap_fill();
        }
    }

    // ---- hash-thing-jw3k.1 targeted column-path tests ----

    #[test]
    fn column_path_all_air_level3_no_change() {
        let mut world = World::new(3);
        let pre = world.root;
        world.apply_gap_fill_column_path();
        assert_eq!(world.root, pre, "all-air world must not change");
    }

    #[test]
    fn column_path_all_sand_level3_no_change() {
        let mut world = World::new(3);
        let side = world.side() as u64;
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    world.set(wc(x), wc(y), wc(z), crate::terrain::materials::SAND);
                }
            }
        }
        let pre = world.root;
        world.apply_gap_fill_column_path();
        assert_eq!(world.root, pre, "all-sand world has no gaps to fill");
    }

    /// Triple pattern at a boundary (0,0,0)-(0,2,0) → gap_fill_column swaps.
    /// Confirms column walking reaches boundary (x,z) coordinates correctly.
    #[test]
    fn column_path_boundary_corner_triple_swaps() {
        use crate::terrain::materials::SAND;
        let mut world = World::new(3);
        // sand at y=0 and y=2, air at y=1, all at (x=0,z=0)
        world.set(wc(0), wc(0), wc(0), SAND);
        world.set(wc(0), wc(2), wc(0), SAND);

        world.apply_gap_fill_column_path();

        // After gap_fill: sand at y=0 and y=1, air at y=2
        let side = world.side();
        let grid = world.store.flatten(world.root, side);
        let idx = |x: usize, y: usize, z: usize| x + y * side + z * side * side;
        assert_eq!(grid[idx(0, 0, 0)], SAND, "y=0 unchanged");
        assert_eq!(grid[idx(0, 1, 0)], SAND, "y=1 received sand from y=2");
        assert_eq!(grid[idx(0, 2, 0)], 0, "y=2 became air");
    }

    /// hash-thing-9yv2 regression test: stone above sand must not delete the sand.
    ///
    /// Before the fix, `apply_block_in_grid` would dispatch GravityBlockRule on a
    /// block containing sand (rule-cell) + stone (no rule). The rule swapped on
    /// density (stone 5.0 > sand 1.5), and the write-back filter only wrote
    /// positions whose ORIGINAL cell had a rule — so stone's new position
    /// (previously sand) got the stone value (sand deleted), and sand's new
    /// position (previously stone) was skipped (stone stayed). Net: sand
    /// vanished, stone duplicated. After 9yv2, BlockRule honors a `movable`
    /// mask and refuses to swap an anchored cell, so the block is unchanged.
    #[test]
    fn repro_9yv2_inert_swap_deletes_rule_cell() {
        use crate::terrain::materials::{SAND, STONE};
        // Level-3 world = 8³. Place stone on top of sand so gravity rule swaps
        // them in a single 2×2×2 block.
        let mut world = World::new(3);
        // Block (bx=0, by=0, bz=0) covers y=0 and y=1. Stone at y=1, sand at y=0.
        world.set(wc(0), wc(0), wc(0), SAND);
        world.set(wc(0), wc(1), wc(0), STONE);

        let pre_sand = count_material(&world, 8, SAND_MATERIAL_ID);
        let pre_stone = count_material(&world, 8, crate::terrain::materials::STONE_MATERIAL_ID);
        assert_eq!(pre_sand, 1);
        assert_eq!(pre_stone, 1);

        world.step_recursive();

        let post_sand = count_material(&world, 8, SAND_MATERIAL_ID);
        let post_stone = count_material(&world, 8, crate::terrain::materials::STONE_MATERIAL_ID);

        eprintln!("pre: sand={pre_sand} stone={pre_stone}");
        eprintln!("post: sand={post_sand} stone={post_stone}");

        assert_eq!(post_sand, 1, "sand cell was deleted");
        assert_eq!(post_stone, 1, "phantom stone appeared");
    }

    /// hash-thing-9yv2 Phase 1: per-gen sandwich accounting.
    ///
    /// Seeds the same 64³ `water_and_sand` scene the v3b2 repro harness uses,
    /// then at each generation splits `step_recursive` into two measured halves:
    ///
    /// 1. `step_margolus_only` — recursive memoised Margolus/block-rule descent.
    /// 2. `apply_gap_fill_column_path` — per-column gravity settle.
    ///
    /// Records water + sand counts `W_pre / W_mid / W_post` and prints the
    /// per-phase deltas. This tells us which sub-phase owns each cell lost.
    ///
    /// Invariant scope: the `seed_water_and_sand` scene has no fire / lava /
    /// acid, so water and sand have no reactive partners — their counts must
    /// be monotonic-non-decreasing. Any negative delta is the defect.
    #[test]
    #[ignore]
    fn repro_9yv2_phase_mass_accounting_64() {
        use crate::terrain::TerrainParams;

        let level = 6u32;
        let side = 1i64 << level;
        let mut world = World::new(level);
        let params = TerrainParams::for_level(level);
        world
            .seed_terrain(&params)
            .expect("level-derived terrain params must validate");
        world.seed_water_and_sand();

        let w0 = count_material(&world, side, WATER_MATERIAL_ID);
        let s0 = count_material(&world, side, SAND_MATERIAL_ID);
        eprintln!("=== 9yv2 phase mass accounting: 64³ water_and_sand ===");
        eprintln!("initial: water={w0} sand={s0}");
        eprintln!("gen | water (pre→mid→post, Δmarg,Δgf) | sand (pre→mid→post, Δmarg,Δgf)");

        for gen in 0..80u32 {
            let w_pre = count_material(&world, side, WATER_MATERIAL_ID);
            let s_pre = count_material(&world, side, SAND_MATERIAL_ID);

            world.step_margolus_only();

            let w_mid = count_material(&world, side, WATER_MATERIAL_ID);
            let s_mid = count_material(&world, side, SAND_MATERIAL_ID);

            if world.has_block_rule_cells() {
                world.apply_gap_fill_column_path();
            }

            let w_post = count_material(&world, side, WATER_MATERIAL_ID);
            let s_post = count_material(&world, side, SAND_MATERIAL_ID);

            world.finalize_step_after_external_gap_fill();

            let dw_m = w_mid as isize - w_pre as isize;
            let dw_g = w_post as isize - w_mid as isize;
            let ds_m = s_mid as isize - s_pre as isize;
            let ds_g = s_post as isize - s_mid as isize;

            if dw_m != 0 || dw_g != 0 || ds_m != 0 || ds_g != 0 {
                eprintln!(
                    "gen {gen:>2}: W {w_pre}→{w_mid}→{w_post} ({dw_m:+},{dw_g:+}) | \
                     S {s_pre}→{s_mid}→{s_post} ({ds_m:+},{ds_g:+})",
                );
            }
        }

        let w_end = count_material(&world, side, WATER_MATERIAL_ID);
        let s_end = count_material(&world, side, SAND_MATERIAL_ID);
        eprintln!(
            "end: water={w_end} (Δ={}) sand={s_end} (Δ={})",
            w_end as isize - w0 as isize,
            s_end as isize - s0 as isize
        );
    }

    /// hash-thing-9yv2 path comparison: brute `step()` vs hashlife `step_recursive()`
    /// on the same pool-over-terrain scene. If brute preserves water mass and
    /// hashlife doesn't, the defect is in hashlife recursion (step_node /
    /// recursive_case), not in block-rule write-back.
    #[test]
    #[ignore]
    fn repro_9yv2_brute_vs_recursive_water_mass() {
        use crate::terrain::materials::WATER;
        use crate::terrain::TerrainParams;

        let level = 6u32;
        let side = 1i64 << level;

        let make_world = || -> World {
            let mut w = World::new(level);
            let params = TerrainParams::for_level(level);
            w.seed_terrain(&params).expect("terrain");
            let side_u = side as u64;
            let center = side_u / 2;
            let water_y = center + center / 4;
            let pool_radius = side_u / 6;
            let pool_depth = (side_u / 32).max(2);
            let lo_x = center.saturating_sub(pool_radius);
            let hi_x = (center + pool_radius).min(side_u);
            let lo_z = center.saturating_sub(pool_radius);
            let hi_z = (center + pool_radius).min(side_u);
            for z in lo_z..hi_z {
                for x in lo_x..hi_x {
                    for dy in 0..pool_depth {
                        let y = water_y + dy;
                        if y < side_u {
                            w.set(wc(x), wc(y), wc(z), WATER);
                        }
                    }
                }
            }
            w
        };

        let mut brute = make_world();
        let mut recur = make_world();

        let w0 = count_material(&brute, side, WATER_MATERIAL_ID);
        eprintln!("=== 9yv2 brute-vs-recursive: initial water={w0} ===");
        for gen in 0..80u32 {
            brute.step();
            recur.step_recursive();
            let wb = count_material(&brute, side, WATER_MATERIAL_ID);
            let wr = count_material(&recur, side, WATER_MATERIAL_ID);
            if wb != wr {
                eprintln!(
                    "gen {gen:>2}: brute={wb} recur={wr} (diff={})",
                    wb as isize - wr as isize
                );
            }
        }
        let wb = count_material(&brute, side, WATER_MATERIAL_ID);
        let wr = count_material(&recur, side, WATER_MATERIAL_ID);
        eprintln!(
            "end: brute={wb} (Δ={}) recur={wr} (Δ={})",
            wb as isize - w0 as isize,
            wr as isize - w0 as isize,
        );
    }

    /// hash-thing-0z5d divergent-window capture (Angle A from cairn 2026-04-27).
    ///
    /// Steps brute `World::step()` and recursive `World::step_recursive()` in
    /// lockstep on the same pool-over-terrain scene. After each step, flattens
    /// both worlds and finds every cell where the two paths disagree. Dumps:
    ///
    /// 1. Total divergent cell count + per-material breakdown for the gen.
    /// 2. The 8-aligned bucket histogram of divergent x and z coordinates
    ///    (Angle B: cross-leaf-boundary check — if losses cluster at world
    ///    coordinates 7,8,15,16,... that's the 8³ leaf boundary signature).
    /// 3. For the first divergent gen, full hex dumps of the pre-step state,
    ///    brute post-state, and recursive post-state for the 8³ window
    ///    containing the first divergent cell.
    ///
    /// The dump is structured to localize whether the leak is in:
    /// - Phase 1 / Phase 2 within a single 8³ leaf window, or
    /// - the recursive composition between adjacent leaves.
    #[test]
    #[ignore]
    fn repro_0z5d_divergent_window_capture() {
        use crate::octree::Cell;
        use crate::terrain::materials::WATER;
        use crate::terrain::TerrainParams;

        let level = 6u32;
        let side = 1i64 << level;
        let side_u = side as usize;

        let make_world = || -> World {
            let mut w = World::new(level);
            let params = TerrainParams::for_level(level);
            w.seed_terrain(&params).expect("terrain");
            let center = side_u as u64 / 2;
            let water_y = center + center / 4;
            let pool_radius = side_u as u64 / 6;
            let pool_depth = (side_u as u64 / 32).max(2);
            let lo_x = center.saturating_sub(pool_radius);
            let hi_x = (center + pool_radius).min(side_u as u64);
            let lo_z = center.saturating_sub(pool_radius);
            let hi_z = (center + pool_radius).min(side_u as u64);
            for z in lo_z..hi_z {
                for x in lo_x..hi_x {
                    for dy in 0..pool_depth {
                        let y = water_y + dy;
                        if y < side_u as u64 {
                            w.set(wc(x), wc(y), wc(z), WATER);
                        }
                    }
                }
            }
            w
        };

        let mut brute = make_world();
        let mut recur = make_world();

        // Confirm initial states are identical.
        let init_b = brute.flatten();
        let init_r = recur.flatten();
        assert_eq!(init_b, init_r, "initial state mismatch");

        let mut first_divergence_gen: Option<u32> = None;

        for gen in 0..80u32 {
            // Snapshot pre-step state from brute (== recur, by induction over
            // post-condition below).
            let pre = brute.flatten();

            brute.step();
            recur.step_recursive();

            let post_b = brute.flatten();
            let post_r = recur.flatten();

            if post_b == post_r {
                continue;
            }

            // Find every divergent cell.
            let mut diverged: Vec<(usize, usize, usize, u16, u16)> = Vec::new();
            let idx = |x: usize, y: usize, z: usize| x + y * side_u + z * side_u * side_u;
            for z in 0..side_u {
                for y in 0..side_u {
                    for x in 0..side_u {
                        let bb = post_b[idx(x, y, z)];
                        let rr = post_r[idx(x, y, z)];
                        if bb != rr {
                            let mb = Cell::from_raw(bb).material();
                            let mr = Cell::from_raw(rr).material();
                            diverged.push((x, y, z, mb, mr));
                        }
                    }
                }
            }

            eprintln!(
                "\n=== gen {gen}: {n} divergent cells ===",
                n = diverged.len()
            );

            // Per-material summary (brute -> recur).
            let mut counts: std::collections::BTreeMap<(u16, u16), usize> =
                std::collections::BTreeMap::new();
            for &(_, _, _, mb, mr) in &diverged {
                *counts.entry((mb, mr)).or_insert(0) += 1;
            }
            for ((mb, mr), n) in &counts {
                eprintln!("  brute={mb:>3} recur={mr:>3}  n={n}");
            }

            // 8-aligned bucket histogram for x, y, z (Angle B).
            let mut x_buckets: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            let mut y_buckets: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            let mut z_buckets: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            for &(x, y, z, _, _) in &diverged {
                *x_buckets.entry(x % 8).or_insert(0) += 1;
                *y_buckets.entry(y % 8).or_insert(0) += 1;
                *z_buckets.entry(z % 8).or_insert(0) += 1;
            }
            eprintln!("  x % 8 histogram: {x_buckets:?}");
            eprintln!("  y % 8 histogram: {y_buckets:?}");
            eprintln!("  z % 8 histogram: {z_buckets:?}");

            // Dump 8³ window for the first divergence only.
            if first_divergence_gen.is_none() {
                first_divergence_gen = Some(gen);

                let (fx, fy, fz, _, _) = diverged[0];
                // Align to the 8-grid window containing (fx, fy, fz).
                let wx = (fx / 8) * 8;
                let wy = (fy / 8) * 8;
                let wz = (fz / 8) * 8;
                eprintln!("\n  first divergent cell: (x={fx}, y={fy}, z={fz})");
                eprintln!("  dumping 8³ window starting at (wx={wx}, wy={wy}, wz={wz})");
                eprintln!("  window y/z slabs: pre | post-brute | post-recur (cell.material())");
                let dump_window = |grid: &[u16], label: &str| {
                    eprintln!("  {label}:");
                    for dy in 0..8 {
                        let y = wy + dy;
                        if y >= side_u {
                            continue;
                        }
                        eprintln!("    y={y}:");
                        for dz in 0..8 {
                            let z = wz + dz;
                            if z >= side_u {
                                continue;
                            }
                            let mut row = String::new();
                            for dx in 0..8 {
                                let x = wx + dx;
                                if x >= side_u {
                                    row.push_str("  .");
                                    continue;
                                }
                                let m = Cell::from_raw(grid[idx(x, y, z)]).material();
                                row.push_str(&format!(" {m:>2}"));
                            }
                            eprintln!("      z={z}: {row}");
                        }
                    }
                };
                dump_window(&pre, "pre-step");
                dump_window(&post_b, "post-step brute");
                dump_window(&post_r, "post-step recursive");

                // Stop after first window dump to keep output bounded.
                break;
            }
        }

        match first_divergence_gen {
            Some(g) => eprintln!("\n=== first divergence at gen {g} (window dumped above) ==="),
            None => eprintln!("\n=== no divergence in 80 gens ==="),
        }
    }

    // ============================================================
    // hash-thing-bjdl (vqke.2): targeted unit tests for the new
    // memo-hit-rate diagnostic counters. Per Codex plan-review §5,
    // a "step once on a real scene" test is too non-deterministic;
    // these inject the exact precondition the counters care about.
    // ============================================================

    /// `cache_misses_phase_aliased` covers both contracts:
    /// (a) when `HASH_THING_MEMO_DIAG` is on (forced via the test
    ///     hook), a miss whose NodeId is already cached at another
    ///     phase MUST increment the counter; and
    /// (b) when the gate is off (production default), the same
    ///     precondition MUST NOT increment — the always-zero-in-prod
    ///     contract that justifies the env-var gate at all.
    ///
    /// Both arms run inline in the SAME test function so the gate
    /// flag toggle can't race with a sibling test running in parallel
    /// (cargo test runs tests on multiple threads by default; the
    /// gate is process-wide). We use a GoL world with a populated
    /// 4×4×4 region so step_node's inert/all-inert short-circuits
    /// don't fire before the miss path.
    #[test]
    fn cache_misses_phase_aliased_obeys_diag_gate() {
        // Helper: build the same GoL world + alias precondition
        // twice (once per arm), so each arm starts from a clean
        // counter without aliasing across them.
        fn fresh_world_with_alias_precondition() -> (World, NodeId) {
            let mut world = gol_world(3, GameOfLife3D::new(4, 7, 6, 8), 1);
            for x in 2..6 {
                for y in 2..6 {
                    for z in 2..6 {
                        world.set(wc(x), wc(y), wc(z), ALIVE.into());
                    }
                }
            }
            let any_node = world.root;
            world.hashlife_cache.insert((any_node, 0), any_node);
            (world, any_node)
        }

        // Arm A: gate ON → counter must increment.
        force_memo_diag_for_test(true);
        let (mut world_on, node_on) = fresh_world_with_alias_precondition();
        let before_on = world_on.hashlife_stats.cache_misses_phase_aliased;
        world_on.step_node(
            node_on, 3, /*schedule_phase*/ 1, /*memo_period*/ 4,
        );
        let after_on = world_on.hashlife_stats.cache_misses_phase_aliased;
        assert!(
            after_on > before_on,
            "with diag ON the alias precondition must fire the counter (before={before_on} after={after_on})"
        );

        // Arm B: gate OFF → counter must stay at 0.
        force_memo_diag_for_test(false);
        let (mut world_off, node_off) = fresh_world_with_alias_precondition();
        world_off.step_node(node_off, 3, 1, 4);
        assert_eq!(
            world_off.hashlife_stats.cache_misses_phase_aliased, 0,
            "with diag OFF the probe must not fire even when the alias precondition is satisfied"
        );

        // Reset the gate so any later (parallel) test sees the
        // production default.
        force_memo_diag_for_test(false);
    }

    /// `remap_caches` must split its outcome into kept (entries whose
    /// node + result both survived the remap) and dropped (everything
    /// else). Three entries in, two survive, one dropped: counters
    /// land at 2/1.
    #[test]
    fn remap_caches_counts_kept_and_dropped() {
        let mut world = World::new(4);
        let n1 = world.store.empty(2);
        let n2 = world.store.empty(3);
        let n3 = world.store.empty(4);
        // Three cache entries: n1→n2, n2→n3, n3→n1. Two survive
        // remap, one drops.
        world.hashlife_cache.insert((n1, 0), n2);
        world.hashlife_cache.insert((n2, 1), n3);
        world.hashlife_cache.insert((n3, 0), n1);

        let mut remap: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        remap.insert(n1, n1);
        remap.insert(n2, n2);
        // n3 is intentionally NOT in the remap → both keys/values
        // touching n3 should be dropped.

        world.hashlife_stats = super::super::world::HashlifeStats::default();
        world.remap_caches(&remap);

        // Survivors: only the (n1, 0) → n2 entry (both endpoints in
        // the remap). The (n2, 1) → n3 entry drops on the value side
        // (n3 not in remap), and (n3, 0) → n1 drops on the key side.
        assert_eq!(
            world.hashlife_stats.compact_entries_kept, 1,
            "exactly the (n1, 0) → n2 entry should survive"
        );
        assert_eq!(
            world.hashlife_stats.compact_entries_dropped, 2,
            "two entries reference the unmapped n3 and must be dropped"
        );
        assert_eq!(
            world.hashlife_stats.compact_entries_kept
                + world.hashlife_stats.compact_entries_dropped,
            3,
            "kept + dropped must equal the pre-remap cache size (3)"
        );
    }

    /// `remap_caches` on an empty cache must report 0/0 — no
    /// underflow on the `before - kept` subtraction (the dropped
    /// counter is computed as `usize - usize` cast to `u64`).
    #[test]
    fn remap_caches_handles_empty_cache_with_zero_counters() {
        let mut world = World::new(4);
        world.hashlife_stats = super::super::world::HashlifeStats::default();
        // hashlife_cache is empty by default.
        let remap: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        world.remap_caches(&remap);
        assert_eq!(world.hashlife_stats.compact_entries_kept, 0);
        assert_eq!(world.hashlife_stats.compact_entries_dropped, 0);
    }

    // ============================================================
    // hash-thing-5ie4 (vqke.2.1): phase-fold tests for the
    // memo cache key. The fold collapses (NodeId, schedule_phase)
    // to (NodeId, schedule_phase % 2) for subtrees containing no
    // slow-divisor materials.
    // ============================================================

    /// `subtree_has_slow_divisor` correctly classifies a leaf containing
    /// water (tick_divisor=2 in terrain_defaults) as slow, and an empty
    /// or stone-only subtree as fast.
    #[test]
    fn subtree_has_slow_divisor_classifies_water_vs_stone() {
        use crate::terrain::materials::{STONE, WATER};
        let mut world = World::new(4);

        // Empty world root: predicate must be false.
        assert!(
            !world.subtree_has_slow_divisor(world.root),
            "empty subtree must be fast"
        );

        // Stone-only subtree: stone has tick_divisor=1, predicate false.
        world.set(wc(1), wc(1), wc(1), STONE);
        let stone_root = world.root;
        assert!(
            !world.subtree_has_slow_divisor(stone_root),
            "stone-only subtree must be fast (tick_divisor=1)"
        );

        // Insert one water cell anywhere — predicate flips to true.
        world.set(wc(8), wc(8), wc(8), WATER);
        let mixed_root = world.root;
        assert!(
            world.subtree_has_slow_divisor(mixed_root),
            "subtree containing water (tick_divisor=2) must be slow"
        );
    }

    /// hash-thing-5ie4 / Codex plan-review §5: stale-cache regression.
    /// After `mutate_materials` raises a divisor on a material that
    /// was previously fast, the predicate must reflect the new value
    /// for previously-cached nodes. Without invalidation in
    /// `invalidate_material_caches`, the predicate would silently
    /// return the stale `false` and corrupt the fold.
    #[test]
    fn subtree_has_slow_divisor_invalidates_on_material_mutation() {
        use crate::terrain::materials::{STONE, STONE_MATERIAL_ID};
        let mut world = World::new(4);
        world.set(wc(2), wc(2), wc(2), STONE);
        let stone_root = world.root;

        // Initially fast (stone tick_divisor=1).
        assert!(
            !world.subtree_has_slow_divisor(stone_root),
            "stone subtree starts fast"
        );

        // Mutate registry: bump stone's divisor to 4.
        world.set_material_tick_divisor(STONE_MATERIAL_ID, 4);

        // After invalidation, the predicate cache must be cleared
        // so the fresh query reflects the updated divisor.
        assert!(
            world.subtree_has_slow_divisor(stone_root),
            "after raising stone's tick_divisor, the previously-fast subtree must reclassify as slow — otherwise the fold corrupts the memo"
        );
    }

    /// Semantic correctness of the fold: for a fast subtree at a
    /// period-4 world, `step_recursive` at gen=0 and gen=2 must
    /// produce identical world.root values when the world's content
    /// is reset between runs (i.e. without cache reuse). Per Codex
    /// plan-review §5 this is the test that proves the fold is
    /// computationally sound, not just that it caches correctly.
    ///
    /// Per Claude code-review IMPORTANT: stone+air is self-inert
    /// (stone has `CaRule::Noop` and no BlockRule), so a stone-only
    /// world hits the `is_all_inert` short-circuit at the top of
    /// `step_node` and never enters the fold path. **Use SAND** —
    /// it has `tick_divisor=1` (so the predicate returns false →
    /// fold applies) AND a non-trivial `block_rule_id`
    /// (gravity_block_rule) so the world is not all-inert and
    /// step_node descends through the cache + base case.
    #[test]
    fn fast_subtree_step_result_invariant_under_gen_plus_two() {
        use crate::terrain::materials::SAND;

        // Build the same fast world twice (sand has tick_divisor=1
        // → predicate=false → fold applies; gravity block rule →
        // not inert → recursion + base case actually run); step
        // once each at gen=0 and gen=2. Result roots must match.
        fn make_fast_world() -> World {
            let mut world = World::new(4);
            // 4×4×4 sand floating in air. Gravity block rule will
            // step it down, exercising the BlockRule path (which is
            // where Margolus parity lives). The result must be the
            // same at gen=0 and gen=2 because both have parity 0.
            for x in 1..5 {
                for y in 5..9 {
                    for z in 1..5 {
                        world.set(wc(x), wc(y), wc(z), SAND);
                    }
                }
            }
            world
        }

        // Run A: gen=0.
        let mut world_a = make_fast_world();
        let memo_period_a = world_a.materials.memo_period();
        // Sanity: terrain_defaults has water (divisor=2) → period=4.
        assert_eq!(
            memo_period_a, 4,
            "this test relies on memo_period=4 from terrain_defaults"
        );
        world_a.step_recursive();
        let root_a_after_gen0 = world_a.root;

        // Run B: gen=2 (skip ahead by 2 BEFORE stepping). Without
        // the fold this would run with `schedule_phase=2` and
        // (without our cache fold logic) could produce a different
        // root if the implementation accidentally referenced the
        // generation as anything other than `gen % 2`.
        let mut world_b = make_fast_world();
        world_b.generation = 2;
        world_b.step_recursive();
        let root_b_after_gen2 = world_b.root;

        // Sanity: the step actually moved cells (gravity dropped
        // sand by one row). If both roots are unchanged from the
        // pre-step state, the test isn't exercising anything.
        let world_static = make_fast_world();
        let static_root = world_static.root;
        assert_ne!(
            root_a_after_gen0, static_root,
            "step_recursive on a sand world should change the root (gravity); test would be a no-op otherwise"
        );

        // Both worlds did one step on identical content. World A
        // stepped at gen=0 (phase=0), World B at gen=2 (phase=2 in
        // raw period; effective_phase=0 after fold). For a fast
        // subtree the result must be identical — that's the
        // theorem the fold relies on.
        assert_eq!(
            root_a_after_gen0, root_b_after_gen2,
            "fast subtree must produce identical step result at gen=0 and gen=2 (proves the fold's correctness independent of cache reuse)"
        );
    }
}
