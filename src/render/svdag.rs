//! SVDAG (Sparse Voxel DAG) serialization + GPU upload.
//!
//! This converts our hash-consed octree into a flat node buffer for the GPU.
//! The lineage:
//!   - Laine & Karras 2010 "Efficient Sparse Voxel Octrees" (SVO GPU traversal)
//!   - Kämpe, Sintorn, Assarsson 2013 "High Resolution Sparse Voxel DAGs"
//!     (hash-consing SVO subtrees → DAG → 10-100x compression)
//!   - Careil, Billeter, Eisemann 2020 "HashDAG" (dynamic editing of DAGs)
//!   - Molenaar 2023 "Editing Compressed High-Resolution Voxel Scenes with Attributes"
//!     (per-voxel attributes on top of HashDAG)
//!
//! Our hash-consed octree is already a DAG — identical subtrees are interned to the
//! same NodeId. This module serializes that DAG into a flat array the GPU can
//! traverse with an iterative descent shader, and — critically — **keeps the flat
//! array across frames** so that unchanged subtrees never need to be re-uploaded.
//!
//! Buffer layout (u32 per slot):
//!   [0]:        root_offset — absolute index of the current root node's slot
//!   [1..]:      concatenated 9-u32 interior-node slots, append-only
//!
//! Interior node slot (9 u32s = 36 bytes):
//!   [0]:        child_mask (low 8 bits: which octants are non-empty)
//!   [1..=8]:    child entries — packed as (is_leaf << 31) | payload_bits
//!     - is_leaf: payload is the 16-bit material state in low bits
//!     - else:    payload is the absolute offset of the child node in this buffer
//!
//! ## Incremental uploads (hash-thing-5bb.5)
//!
//! `World::step_flat` calls `NodeStore::compacted()` every step, so `NodeId`s are
//! **not** stable across simulation steps — a NodeId-keyed cache is invalidated on
//! every step. Content identity is stable, though: two subtrees with the same
//! `(mask, child_slots)` tuple represent the same voxel volume regardless of which
//! store epoch they came from.
//!
//! `Svdag` exploits this by keeping a persistent `offset_by_slot: FxHashMap<[u32; 9],
//! u32>` map across calls. `update()` walks the new DAG, computes each node's 9-u32
//! slot bottom-up, and checks the map: hit → reuse the existing GPU-side offset;
//! miss → append to `nodes` and insert. The root's slot changes almost every step
//! (its children change), so the root gets a fresh offset; `nodes[0]` is overwritten
//! with it so the shader can find the new root without hardcoding a position.
//!
//! This makes the upload path O(new-content) instead of O(reachable): for a single-
//! cell edit, only the O(log N) nodes on the path from root to leaf are new. The
//! renderer tracks an upload watermark and writes only the appended tail (plus slot
//! 0 for the root header). See `Renderer::upload_svdag`.

use crate::octree::{Node, NodeId, NodeStore};
use rustc_hash::FxHashMap;

/// High bit of a child slot marks it as an inline leaf. Interior-node offsets
/// must fit strictly below this bound, checked in `write_node` at serialize
/// time — a violation is a silent GPU-corruption bug (the decoder would read
/// a legitimate offset as a leaf and emit a garbage material). This mirrors
/// `LEAF_BIT = 0x80000000u` in `src/render/svdag_raycast.wgsl`; if you change
/// one, change the other.
pub(crate) const LEAF_BIT: u32 = 0x8000_0000;

/// GPU-friendly serialized DAG with persistent content-addressed cache.
///
/// `nodes[0]` is reserved for the current root offset. `nodes[1..]` is an
/// append-only stream of 9-u32 interior-node slots. `offset_by_slot` maps each
/// unique slot tuple to its absolute offset in `nodes`, so repeated `update()`
/// calls reuse offsets for content that already lives in the buffer.
pub struct Svdag {
    /// Packed node data. `nodes[0]` = root offset header, rest = slots.
    pub nodes: Vec<u32>,
    /// Persistent content-exact cache: slot bytes → their offset in `nodes`.
    ///
    /// Keyed by the full 9-u32 slot (not a hash) so lookups are collision-free.
    offset_by_slot: FxHashMap<[u32; 9], u32>,
    /// Level of the current root (2^level cells per side).
    pub root_level: u32,
    /// Number of interior nodes reachable from the current root.
    pub node_count: usize,
}

impl Svdag {
    /// Empty builder. Slot 0 is pre-reserved for the root offset header.
    pub fn new() -> Self {
        Self {
            nodes: vec![0u32],
            offset_by_slot: FxHashMap::default(),
            root_level: 0,
            node_count: 0,
        }
    }

    /// One-shot build — creates a fresh `Svdag` and runs `update` once.
    ///
    /// Convenience wrapper for tests and single-snapshot paths. Production code
    /// should construct once with `new()` and call `update()` each frame so the
    /// content cache persists across calls.
    pub fn build(store: &NodeStore, root: NodeId, root_level: u32) -> Self {
        let mut svdag = Self::new();
        svdag.update(store, root, root_level);
        svdag
    }

    /// Rebuild the serialized DAG for a new root, reusing any slots that were
    /// already interned on previous calls. Writes the fresh root offset to
    /// `nodes[0]` and updates `root_level` and `node_count`.
    ///
    /// Leaf roots (a uniform world, a single-voxel test fixture, or a future
    /// root-collapse optimization) are serialized as a **degenerate single-node
    /// interior** with all 8 children pointing at the leaf's inline encoding —
    /// the shader handles that path with no special casing (hash-thing-nch).
    pub fn update(&mut self, store: &NodeStore, root: NodeId, root_level: u32) {
        self.root_level = root_level;

        if let Node::Leaf(state) = store.get(root) {
            // Degenerate single-node DAG: all 8 children point at the same
            // inline leaf. Build the slot and cache it normally.
            let mask = if *state != 0 { 0xffu32 } else { 0u32 };
            let leaf_slot = LEAF_BIT | (*state as u32);
            let mut slot = [0u32; 9];
            slot[0] = mask;
            for i in 0..8 {
                slot[1 + i] = leaf_slot;
            }
            let offset = self.intern_slot(slot);
            self.nodes[0] = offset;
            self.node_count = 1;
            return;
        }

        let mut id_to_offset: FxHashMap<NodeId, u32> = FxHashMap::default();
        let root_offset = self.visit(store, root, &mut id_to_offset);
        self.nodes[0] = root_offset;
        self.node_count = id_to_offset.len();
    }

    /// Recursively visit a node. Returns its absolute offset in `nodes`.
    ///
    /// Bottom-up: child offsets are resolved before the parent's slot is
    /// assembled, so the parent's slot is itself a fully content-determined
    /// identity once the children have been placed.
    fn visit(
        &mut self,
        store: &NodeStore,
        id: NodeId,
        id_to_offset: &mut FxHashMap<NodeId, u32>,
    ) -> u32 {
        if let Some(&offset) = id_to_offset.get(&id) {
            return offset;
        }
        match store.get(id) {
            Node::Leaf(_) => {
                unreachable!(
                    "visit called on a leaf — parents must encode leaves inline, \
                     and leaf roots are intercepted by Svdag::update"
                );
            }
            Node::Interior { children, .. } => {
                let children = *children;
                let mut slot = [0u32; 9];
                let mut mask: u32 = 0;
                for (i, &child_id) in children.iter().enumerate() {
                    match store.get(child_id) {
                        Node::Leaf(state) => {
                            if *state != 0 {
                                mask |= 1 << i;
                            }
                            slot[1 + i] = LEAF_BIT | (*state as u32);
                        }
                        Node::Interior { population, .. } => {
                            if *population > 0 {
                                mask |= 1 << i;
                                let child_offset = self.visit(store, child_id, id_to_offset);
                                slot[1 + i] = child_offset;
                            } else {
                                slot[1 + i] = LEAF_BIT;
                            }
                        }
                    }
                }
                slot[0] = mask;

                let offset = self.intern_slot(slot);
                id_to_offset.insert(id, offset);
                offset
            }
        }
    }

    /// Intern a 9-u32 slot into the persistent cache. Returns the offset.
    fn intern_slot(&mut self, slot: [u32; 9]) -> u32 {
        if let Some(&existing) = self.offset_by_slot.get(&slot) {
            return existing;
        }
        let new_offset = self.nodes.len() as u32;
        assert!(
            new_offset < LEAF_BIT,
            "SVDAG interior node count exceeded 31 bits ({}); \
             widen the encoding (hash-thing-x9r follow-up)",
            self.nodes.len()
        );
        self.nodes.extend_from_slice(&slot);
        self.offset_by_slot.insert(slot, new_offset);
        new_offset
    }

    pub fn byte_size(&self) -> usize {
        self.nodes.len() * 4
    }

    /// Absolute offset of the current root node in `nodes`. Mirrors `nodes[0]`.
    pub fn root_offset(&self) -> u32 {
        self.nodes[0]
    }

    /// Cumulative count of distinct slots interned over this builder's lifetime.
    pub fn total_slot_count(&self) -> usize {
        self.offset_by_slot.len()
    }

    /// Ratio of stale (unreachable) slots to total interned slots.
    /// Returns 0.0 when `total_slot_count` is 0.
    pub fn stale_ratio(&self) -> f64 {
        let total = self.offset_by_slot.len();
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.node_count as f64 / total as f64)
    }

    /// Compact the buffer by rebuilding from the current root into a fresh
    /// Svdag, dropping unreachable slots and the stale cache entries that
    /// pointed at them. The renderer must do a full re-upload after this
    /// (the returned Svdag has a fresh buffer).
    ///
    /// Callers should check `stale_ratio()` first — compaction is O(reachable),
    /// which is the same cost as `update()`, so running it every frame is
    /// wasteful. A threshold of 0.5 (compact when >50% of buffer is stale)
    /// is a reasonable default.
    pub fn compact(&mut self, store: &NodeStore, root: NodeId) {
        let root_level = self.root_level;
        *self = Self::build(store, root, root_level);
    }
}

// ----------------------------------------------------------------------------
// CPU-side replica of svdag_raycast.wgsl for debugging.
//
// This deliberately mirrors the WGSL shader line-for-line so that any bug in
// the traversal reproduces identically on the CPU, where we can dump state.
// ----------------------------------------------------------------------------

// CPU debug-trace helpers. Used ad-hoc when investigating shader bugs;
// no automated tests currently call this module, but keeping it compiling
// is the whole point.
#[cfg(test)]
#[allow(unused_imports, dead_code)]
pub mod cpu_trace {
    use super::*;

    pub const MAX_DEPTH: usize = 14;
    // LEAF_BIT comes from `use super::*` above — kept in one place so the
    // shader mirror stays a single source of truth (hash-thing-x9r).
    const EPS: f32 = 1e-5;

    /// Absolute minimum step budget. Shallow scenes get this even if the
    /// root_level-derived bound is smaller; prevents a tiny analytical bound
    /// from starving demo-sized worlds. Must stay ≥ the pre-2w5 fixed cap
    /// (512) so nothing regresses on the existing regression suite.
    pub const MIN_STEP_BUDGET: usize = 1024;

    /// Per-cell fudge factor on top of `root_side`. A straight diagonal ray
    /// crosses at most `3 * root_side` cells; real rays re-enter neighbor
    /// cells across boundaries and need headroom. `8 *` is conservative —
    /// instrumentable via the new `exhausted` flag on `TraceResult`.
    pub const STEP_BUDGET_FUDGE: usize = 8;

    /// Compute the MAX_STEPS budget for a given `root_level`.
    ///
    /// Result: `max(MIN_STEP_BUDGET, STEP_BUDGET_FUDGE * (1 << root_level))`.
    /// For root_level=6 (256³): `max(1024, 8 * 64) = 1024`.
    /// For root_level=12 (4096³): `max(1024, 8 * 4096) = 32768`.
    /// For root_level=14 (16384³): `max(1024, 8 * 16384) = 131072`.
    /// Saturates above root_level=28 at `(1 << 28) * 8 = 2^31 = 2_147_483_648`.
    ///
    /// The `.min(28)` ceiling is picked to keep the WGSL mirror u32-safe:
    /// `root_side * 8` fits in u32 iff `root_side ≤ 2^28`. Both clamps agree
    /// byte-for-byte at every real root_level so the CPU replica and shader
    /// never diverge on budget. `saturating_mul` is a belt-and-braces guard
    /// for hypothetical 32-bit usize targets; on 64-bit targets the product
    /// trivially fits.
    pub fn step_budget(root_level: u32) -> usize {
        let side = 1usize << root_level.min(28);
        MIN_STEP_BUDGET.max(side.saturating_mul(STEP_BUDGET_FUDGE))
    }

    #[derive(Clone, Copy, Debug)]
    pub enum TraceEvent {
        Descend {
            depth: usize,
            oct: u32,
            child_offset: u32,
        },
        EmptyLeaf {
            depth: usize,
            oct: u32,
        },
        Hit {
            depth: usize,
            mat: u32,
            t: f32,
        },
        StepPast {
            t_old: f32,
            t_new: f32,
        },
        Popped {
            from: usize,
            to: usize,
        },
        ExitedRoot,
    }

    pub struct TraceResult {
        pub hit_cell: Option<u32>,
        pub steps: usize,
        pub events: Vec<TraceEvent>,
        /// True iff traversal ran out of steps before terminating (neither
        /// hit nor exited the root cube). Pre-2w5 this case returned a silent
        /// miss indistinguishable from a real background pixel; now the
        /// shader surfaces it as a magenta sentinel and the CPU replica
        /// exposes it here for test assertions.
        pub exhausted: bool,
        /// Analytical surface normal at the hit point, or `None` on miss /
        /// exhausted / empty-leaf. Computed from which axis of the leaf
        /// AABB the ray entered through — the textbook voxel-raycaster
        /// face normal. Mirrors the analytical path in `svdag_raycast.wgsl`
        /// 1:1 (hash-thing-rv4) so tests can pin face normals for
        /// axis-aligned rays and catch any shader/CPU drift.
        pub hit_normal: Option<[f32; 3]>,
    }

    /// Octant index from a point relative to the node center.
    /// Returns bit-packed index 0..7 with bit 0 = x, bit 1 = y, bit 2 = z.
    ///
    /// hash-thing-6hd: uses STRICT `>` for the HIGH half with `rd >= 0` as a
    /// tiebreaker on exact-midpoint positions. The pre-6hd code used `>=`
    /// which biased midpoint-exact positions to the HIGH half regardless of
    /// ray direction; on simultaneous two-axis exits, the step-past nudge on
    /// a non-dominant axis can underflow to sub-ULP (`dt * |rd_axis|` below
    /// `ULP(pos)`), leaving pos EXACTLY on the boundary. The `>=` bias then
    /// picked the wrong sibling whenever `rd_axis < 0`. Mirrors the shader
    /// fix in `svdag_raycast.wgsl` 1:1 — both sides use `>` + rd tiebreak
    /// and the SAME tiebreak direction so CPU/GPU stay byte-aligned.
    ///
    /// Zero-rd convention: when `rd[axis] == 0.0`, the tiebreak resolves to
    /// HIGH (because `rd[axis] >= 0.0` is true). The ray can't cross the
    /// midpoint on that axis anyway, so the choice is physically inert —
    /// we pick HIGH to stay symmetric with the shader's `sign(0.0) = 0`
    /// behavior and the pre-6hd `>=` default.
    pub(super) fn octant_of(pos: [f32; 3], rd: [f32; 3], node_min: [f32; 3], half: f32) -> u32 {
        let mut idx = 0u32;
        let mid = [node_min[0] + half, node_min[1] + half, node_min[2] + half];
        if pos[0] > mid[0] || (pos[0] == mid[0] && rd[0] >= 0.0) {
            idx |= 1;
        }
        if pos[1] > mid[1] || (pos[1] == mid[1] && rd[1] >= 0.0) {
            idx |= 2;
        }
        if pos[2] > mid[2] || (pos[2] == mid[2] && rd[2] >= 0.0) {
            idx |= 4;
        }
        idx
    }

    fn intersect_aabb(
        origin: [f32; 3],
        inv_dir: [f32; 3],
        box_min: [f32; 3],
        box_max: [f32; 3],
    ) -> (f32, f32) {
        let t1 = [
            (box_min[0] - origin[0]) * inv_dir[0],
            (box_min[1] - origin[1]) * inv_dir[1],
            (box_min[2] - origin[2]) * inv_dir[2],
        ];
        let t2 = [
            (box_max[0] - origin[0]) * inv_dir[0],
            (box_max[1] - origin[1]) * inv_dir[1],
            (box_max[2] - origin[2]) * inv_dir[2],
        ];
        let tmin = [t1[0].min(t2[0]), t1[1].min(t2[1]), t1[2].min(t2[2])];
        let tmax = [t1[0].max(t2[0]), t1[1].max(t2[1]), t1[2].max(t2[2])];
        let t_near = tmin[0].max(tmin[1]).max(tmin[2]);
        let t_far = tmax[0].min(tmax[1]).min(tmax[2]);
        (t_near, t_far)
    }

    /// Raycast with a root-level-derived step budget. Thin wrapper around
    /// `raycast_with_budget` that computes the budget from `root_level` via
    /// `step_budget`. Production callers use this entry point.
    pub fn raycast(
        dag: &[u32],
        root_level: u32,
        ro: [f32; 3],
        rd: [f32; 3],
        record: bool,
    ) -> TraceResult {
        raycast_with_budget(dag, step_budget(root_level), ro, rd, record)
    }

    /// Raycast with an explicit step budget. Exposed so tests can force the
    /// budget below the natural value and assert the `exhausted` path is
    /// reached — a witness `step_budget` covering cannot provide on its own
    /// because every realistic scene completes well under the natural budget.
    pub fn raycast_with_budget(
        dag: &[u32],
        max_steps: usize,
        ro: [f32; 3],
        rd: [f32; 3],
        record: bool,
    ) -> TraceResult {
        let inv_rd = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];
        let mut events = Vec::new();

        let (root_near, root_far) = intersect_aabb(ro, inv_rd, [0.0; 3], [1.0; 3]);
        if root_near > root_far || root_far < 0.0 {
            return TraceResult {
                hit_cell: None,
                steps: 0,
                events,
                exhausted: false,
                hit_normal: None,
            };
        }

        // hash-thing-27m v2: clamp `ro` to the root-cube entry point
        // (Laine-Karras traversal trick). After this, all subsequent
        // `pos = ro_local + rd * t` math uses a t bounded by the in-cube
        // traversal length (|t| ≤ √3), so ULP(t) stays ≤ ~2.4e-7 regardless
        // of how far the original camera was. Without this clamp, far-camera
        // rays hit a zero-forward-progress regime: `t_new = oct_far + dt`
        // rounds back to `t_old` in f32 whenever ULP(t) exceeds the nudge
        // (9.5e-7 at MAX_DEPTH). Flagged by Codex Critical in 27m review.
        let entry = root_near.max(0.0);
        let ro_local = [
            ro[0] + rd[0] * entry,
            ro[1] + rd[1] * entry,
            ro[2] + rd[2] * entry,
        ];
        let t_exit = root_far - entry;

        let mut stack_node = [0u32; MAX_DEPTH];
        let mut stack_min = [[0.0f32; 3]; MAX_DEPTH];
        let mut stack_half = [0.0f32; MAX_DEPTH];
        let mut depth: usize = 0;

        stack_node[0] = dag[0];
        stack_min[0] = [0.0; 3];
        stack_half[0] = 0.5;

        // Local-frame starting t. Just past the root entry; `ro_local` is
        // already on the root face.
        let mut t = EPS;

        for step in 0..max_steps {
            if t > t_exit {
                return TraceResult {
                    hit_cell: None,
                    steps: step,
                    events,
                    exhausted: false,
                    hit_normal: None,
                };
            }
            let pos = [
                ro_local[0] + rd[0] * t,
                ro_local[1] + rd[1] * t,
                ro_local[2] + rd[2] * t,
            ];

            // Pop until top contains pos.
            // NOTE: pop check is STRICT (no EPS slack). Two guarantees make
            // this safe:
            //   (1) ro is clamped to the root-cube entry (27m v2), so `t`
            //       inside the loop is local-frame and bounded by √3;
            //       ULP(t_local) stays ≤ ~2.4e-7.
            //   (2) The step-past below uses a cell-scaled nudge of at least
            //       ~9.5e-7 on the identified exit axis at MAX_DEPTH.
            // Together (1) > (2) means pos advances strictly past the octant
            // boundary on the exit axis in the uncapped path. The cap-bound
            // path (ultra-grazing |rd_axis| at MAX_DEPTH) can still collapse
            // below ULP — that remains the known f32 frontier, resolvable
            // only via integer DDA. Allowing EPS slack here would defeat
            // the strict step-past ordering and reintroduce infinite loops
            // on simultaneous two-axis boundary crossings.
            let mut top = depth;
            loop {
                let node_min = stack_min[top];
                let h2 = stack_half[top] * 2.0;
                let inside = pos[0] >= node_min[0]
                    && pos[1] >= node_min[1]
                    && pos[2] >= node_min[2]
                    && pos[0] < node_min[0] + h2
                    && pos[1] < node_min[1] + h2
                    && pos[2] < node_min[2] + h2;
                if inside {
                    break;
                }
                if top == 0 {
                    if record {
                        events.push(TraceEvent::ExitedRoot);
                    }
                    return TraceResult {
                        hit_cell: None,
                        steps: step,
                        events,
                        exhausted: false,
                        hit_normal: None,
                    };
                }
                top -= 1;
            }
            if record && top != depth {
                events.push(TraceEvent::Popped {
                    from: depth,
                    to: top,
                });
            }
            depth = top;

            // Descend
            for _d in 0..MAX_DEPTH {
                let node_offset = stack_node[depth] as usize;
                let node_min = stack_min[depth];
                let half = stack_half[depth];
                let oct = octant_of(pos, rd, node_min, half);
                let child_slot = dag[node_offset + 1 + oct as usize];

                if child_slot & LEAF_BIT != 0 {
                    let mat = child_slot & 0xFFFF;
                    if mat > 0 {
                        if record {
                            events.push(TraceEvent::Hit { depth, mat, t });
                        }
                        // hash-thing-rv4: analytical face normal for the
                        // leaf AABB. Same math as `svdag_raycast.wgsl` —
                        // the axis with the max `tmin_v` is the last face
                        // the ray crossed to enter the voxel; normal
                        // points opposite `rd` on that axis. This is the
                        // CPU oracle half of the shader fix; tests pin
                        // expected face normals for axis-aligned rays to
                        // catch any shader/CPU drift.
                        let leaf_min = [
                            node_min[0] + (oct & 1) as f32 * half,
                            node_min[1] + ((oct >> 1) & 1) as f32 * half,
                            node_min[2] + ((oct >> 2) & 1) as f32 * half,
                        ];
                        let leaf_max = [leaf_min[0] + half, leaf_min[1] + half, leaf_min[2] + half];
                        let lt1 = [
                            (leaf_min[0] - ro_local[0]) * inv_rd[0],
                            (leaf_min[1] - ro_local[1]) * inv_rd[1],
                            (leaf_min[2] - ro_local[2]) * inv_rd[2],
                        ];
                        let lt2 = [
                            (leaf_max[0] - ro_local[0]) * inv_rd[0],
                            (leaf_max[1] - ro_local[1]) * inv_rd[1],
                            (leaf_max[2] - ro_local[2]) * inv_rd[2],
                        ];
                        let tmin_v = [lt1[0].min(lt2[0]), lt1[1].min(lt2[1]), lt1[2].min(lt2[2])];
                        let tmax_v = [lt1[0].max(lt2[0]), lt1[1].max(lt2[1]), lt1[2].max(lt2[2])];
                        // hash-thing-2nd: inside-leaf fallback. When the ray
                        // origin sits inside the filled voxel, every `tmin_v`
                        // component is negative — the entry-face picker (argmax
                        // of `tmin_v`) then returns the nearest BACK face and
                        // the normal flips inward, shading the voxel as if lit
                        // from behind. Reachable via deep-zoom orbit camera.
                        // Fallback: pick the nearest EXIT face (argmin of
                        // `tmax_v`) and flip the sign so the normal points in
                        // the direction the ray is heading — the "about to
                        // emerge" convention. The outer cascade uses `>=` on
                        // both comparators so ties break identically to the
                        // shader in `svdag_raycast.wgsl` — do NOT flip to `>`.
                        // The inner (inside) cascade uses `<=` for the same
                        // reason: CPU/GPU must break exit-face ties identically.
                        let inside = tmin_v[0] < 0.0 && tmin_v[1] < 0.0 && tmin_v[2] < 0.0;
                        let normal = if inside {
                            if tmax_v[0] <= tmax_v[1] && tmax_v[0] <= tmax_v[2] {
                                [rd[0].signum(), 0.0, 0.0]
                            } else if tmax_v[1] <= tmax_v[2] {
                                [0.0, rd[1].signum(), 0.0]
                            } else {
                                [0.0, 0.0, rd[2].signum()]
                            }
                        } else if tmin_v[0] >= tmin_v[1] && tmin_v[0] >= tmin_v[2] {
                            [-rd[0].signum(), 0.0, 0.0]
                        } else if tmin_v[1] >= tmin_v[2] {
                            [0.0, -rd[1].signum(), 0.0]
                        } else {
                            [0.0, 0.0, -rd[2].signum()]
                        };
                        return TraceResult {
                            hit_cell: Some(mat),
                            steps: step,
                            events,
                            exhausted: false,
                            hit_normal: Some(normal),
                        };
                    }
                    if record {
                        events.push(TraceEvent::EmptyLeaf { depth, oct });
                    }
                    break;
                } else {
                    if depth + 1 >= MAX_DEPTH {
                        break;
                    }
                    let child_min = [
                        node_min[0] + (oct & 1) as f32 * half,
                        node_min[1] + ((oct >> 1) & 1) as f32 * half,
                        node_min[2] + ((oct >> 2) & 1) as f32 * half,
                    ];
                    depth += 1;
                    stack_node[depth] = child_slot;
                    stack_min[depth] = child_min;
                    stack_half[depth] = half * 0.5;
                    if record {
                        events.push(TraceEvent::Descend {
                            depth,
                            oct,
                            child_offset: child_slot,
                        });
                    }
                }
            }

            // Step past the current (empty) octant.
            // hash-thing-27m fix: per-axis exit + cell-scaled, exit-axis-aware
            // nudge. The old fixed `+EPS` advanced the per-axis position by
            // `|rd_axis| * EPS`, which collapses below f32 ULP (~6e-8 near
            // pos ~ 0.5) for exit_rd < 6e-3 — a configuration the traversal
            // naturally enters at deep depths when pos lands close to an
            // inter-octant face on a grazing axis. When that happens the
            // pop/descend loop oscillates on the same boundary until MAX_STEPS
            // fires. The replacement advances `t` so the exit-axis position
            // advances by a world-space nudge of ~1e-5 (or ~half/64 at deep
            // levels where 1e-5 would overshoot a cell), then caps dt so the
            // fastest axis never crosses more than one child span.
            let node_min = stack_min[depth];
            let half = stack_half[depth];
            let oct = octant_of(pos, rd, node_min, half);
            let child_min = [
                node_min[0] + (oct & 1) as f32 * half,
                node_min[1] + ((oct >> 1) & 1) as f32 * half,
                node_min[2] + ((oct >> 2) & 1) as f32 * half,
            ];
            let child_max = [
                child_min[0] + half,
                child_min[1] + half,
                child_min[2] + half,
            ];
            // Per-axis exit times. oct_far is the smallest tmax across the
            // three axes (same value the old `intersect_aabb(...).1` returned)
            // but exposed per-axis so we can identify which axis is the actual
            // exit and scale the post-exit nudge against its |rd|.
            let t1 = [
                (child_min[0] - ro_local[0]) * inv_rd[0],
                (child_min[1] - ro_local[1]) * inv_rd[1],
                (child_min[2] - ro_local[2]) * inv_rd[2],
            ];
            let t2 = [
                (child_max[0] - ro_local[0]) * inv_rd[0],
                (child_max[1] - ro_local[1]) * inv_rd[1],
                (child_max[2] - ro_local[2]) * inv_rd[2],
            ];
            let tmax_v = [t1[0].max(t2[0]), t1[1].max(t2[1]), t1[2].max(t2[2])];
            let mut oct_far = tmax_v[0];
            let mut exit_rd = rd[0].abs();
            if tmax_v[1] < oct_far {
                oct_far = tmax_v[1];
                exit_rd = rd[1].abs();
            }
            if tmax_v[2] < oct_far {
                oct_far = tmax_v[2];
                exit_rd = rd[2].abs();
            }
            // World-space exit-axis nudge. `dt_raw = nudge / exit_rd` means
            // pos advances by exactly `nudge_world` on the exit axis. Clip
            // to EPS at shallow levels (root half/64 = 7.8e-3 would be ~1
            // cell at depth 7, overshooting fine structure); shrink with
            // half at deep levels (at depth 14, half/64 ~ 9.5e-7 ≥ 16x ULP).
            let nudge_world = (half * (1.0 / 64.0)).min(EPS);
            let dt_raw = nudge_world / exit_rd.max(1e-6);
            // Cap dt so the fastest axis advances at most one child span,
            // preventing the step from skipping a non-empty sibling on a
            // dominant axis. For ultra-grazing rays (|rd_axis| < ~1e-5 at
            // MAX_DEPTH) the cap binds and the exit-axis advance collapses
            // below ULP — the f32 frontier noted in the plan.
            let max_rd = rd[0].abs().max(rd[1].abs()).max(rd[2].abs());
            let dt_cap = half / max_rd.max(1e-6);
            let dt = dt_raw.min(dt_cap);
            let t_old = t;
            t = oct_far + dt;
            if record {
                events.push(TraceEvent::StepPast { t_old, t_new: t });
            }
        }

        // Post-loop fall-through: budget exhausted while still inside the
        // root cube (neither hit nor exit). Pre-2w5 this returned a silent
        // miss indistinguishable from background. Now the flag is surfaced
        // so tests can assert on it, and the GPU shader renders the same
        // state as a magenta sentinel pixel.
        TraceResult {
            hit_cell: None,
            steps: max_steps,
            events,
            exhausted: true,
            hit_normal: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{Cell, CellState, NodeStore};

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    /// Pack a material id into a raw `CellState` word (metadata=0) so the
    /// `Cell::from_raw` invariant holds at `set_cell` / `leaf` call sites.
    /// Under the 16-bit tagged encoding raw values 1..=63 are the forbidden
    /// "material 0 with metadata flavor" range, so tests that stored bare
    /// u8 ids before 1v0.1 must round-trip through this helper.
    fn mat(id: u16) -> CellState {
        Cell::pack(id, 0).raw()
    }

    #[test]
    fn single_voxel_center_hit() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        // One voxel at (32,32,32) -> world (0.5, 0.5, 0.5)..(0.515625, ...)
        let mat1 = Cell::pack(1, 0).raw();
        root = store.set_cell(root, 32, 32, 32, mat1);
        let dag = Svdag::build(&store, root, 6);

        // Ray aimed at the voxel's center (0.5078125, 0.5078125, 0.5078125) from outside
        let target = [0.5 + 0.5 / 64.0, 0.5 + 0.5 / 64.0, 0.5 + 0.5 / 64.0];
        let ro = [2.0, 2.0, 2.0];
        let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, true);
        // `hit_cell` is the raw packed cell word, not the decoded material id.
        assert_eq!(
            result.hit_cell,
            Some(mat1 as u32),
            "expected hit, events: {:#?}",
            result.events
        );
    }

    #[test]
    fn eight_corners_each_visible() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        let mat1 = Cell::pack(1, 0).raw();
        for &x in &[16u64, 48] {
            for &y in &[16u64, 48] {
                for &z in &[16u64, 48] {
                    for dx in 0..4u64 {
                        for dy in 0..4u64 {
                            for dz in 0..4u64 {
                                root = store.set_cell(root, x + dx, y + dy, z + dz, mat1);
                            }
                        }
                    }
                }
            }
        }
        let dag = Svdag::build(&store, root, 6);

        // Aim a ray from outside at the CENTER voxel of each corner cluster.
        let mut misses = Vec::new();
        for &x in &[16u64, 48] {
            for &y in &[16u64, 48] {
                for &z in &[16u64, 48] {
                    let target = [
                        (x as f32 + 2.0) / 64.0,
                        (y as f32 + 2.0) / 64.0,
                        (z as f32 + 2.0) / 64.0,
                    ];
                    let ro = [2.0, 2.0, 2.0];
                    let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
                    let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
                    if result.hit_cell.is_none() {
                        misses.push((x, y, z));
                    }
                }
            }
        }
        assert!(
            misses.is_empty(),
            "expected to hit all 8 corners, missed: {:?}",
            misses
        );
    }

    /// Shoot rays at every single voxel in the 8-corner scene from a single outside
    /// point. This stresses the traversal much harder than the cluster-center test:
    /// every ray crosses multiple empty octants at varying depths, and some hit
    /// exact inter-octant boundaries.
    #[test]
    fn eight_corners_per_voxel_visible() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        let mat1 = Cell::pack(1, 0).raw();
        let mut voxels = Vec::new();
        for &x in &[16u64, 48] {
            for &y in &[16u64, 48] {
                for &z in &[16u64, 48] {
                    for dx in 0..4u64 {
                        for dy in 0..4u64 {
                            for dz in 0..4u64 {
                                root = store.set_cell(root, x + dx, y + dy, z + dz, mat1);
                                voxels.push((x + dx, y + dy, z + dz));
                            }
                        }
                    }
                }
            }
        }
        let dag = Svdag::build(&store, root, 6);

        let ro = [2.0, 2.0, 2.0];
        let mut misses = 0;
        for &(cx, cy, cz) in &voxels {
            // Target the center of this voxel in world space.
            let target = [
                (cx as f32 + 0.5) / 64.0,
                (cy as f32 + 0.5) / 64.0,
                (cz as f32 + 0.5) / 64.0,
            ];
            let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
            if result.hit_cell.is_none() {
                misses += 1;
            }
        }
        // Every ray is aimed at the center of a filled voxel, so every ray should hit
        // SOMETHING (either its target or another voxel it passes through first).
        assert_eq!(
            misses,
            0,
            "expected every voxel-targeted ray to hit, {} / {} missed",
            misses,
            voxels.len()
        );
    }

    /// Stress test: a random sparse scatter (matches the main.rs debug scene).
    /// Rays aimed at each filled voxel from an outside point should all hit.
    #[test]
    fn random_sparse_all_hit() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        let mut voxels = Vec::new();
        // Simple hash-based PRNG for reproducibility
        let mut seed: u64 = 0xdead_beef_u64;
        let mut rand_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        let mat1 = Cell::pack(1, 0).raw();
        for _ in 0..300 {
            let x = (rand_u32() % 64) as u64;
            let y = (rand_u32() % 64) as u64;
            let z = (rand_u32() % 64) as u64;
            root = store.set_cell(root, x, y, z, mat1);
            voxels.push((x, y, z));
        }
        let dag = Svdag::build(&store, root, 6);

        let ro = [2.0, 2.0, 2.0];
        let mut misses = 0;
        for &(cx, cy, cz) in &voxels {
            let target = [
                (cx as f32 + 0.5) / 64.0,
                (cy as f32 + 0.5) / 64.0,
                (cz as f32 + 0.5) / 64.0,
            ];
            let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
            if result.hit_cell.is_none() {
                misses += 1;
            }
        }
        assert_eq!(
            misses,
            0,
            "expected every random sparse voxel to be hit, {} / {} missed",
            misses,
            voxels.len()
        );
    }

    /// Pin the Rust-side `LEAF_BIT` to the exact bit pattern the shader reads.
    /// If someone widens the encoding they have to touch this test *and* the
    /// WGSL — the test is the tripwire (hash-thing-x9r).
    #[test]
    fn leaf_bit_matches_shader() {
        assert_eq!(LEAF_BIT, 0x8000_0000);
    }

    /// Regression for hash-thing-nch I4: `Svdag::build` must accept a leaf
    /// root (uniform world, single-cell fixture, future root-collapse) by
    /// emitting a degenerate 9-u32 single-node interior DAG. Before this
    /// fix, `write_node` panicked on leaf roots with a hand-wave comment
    /// that "the root will always be level >= 1 in practice."
    #[test]
    fn leaf_root_builds_degenerate_dag() {
        let mut store = NodeStore::new();
        let mat3 = mat(3);
        let root = store.leaf(mat3);

        // Any root_level should work. Pick something non-trivial to prove
        // the level is preserved on the `Svdag` struct independent of the
        // node encoding.
        let dag = Svdag::build(&store, root, 5);
        assert_eq!(dag.node_count, 1, "degenerate leaf-root DAG is one node");
        // 1 header word + 9 node u32s = 10
        assert_eq!(dag.nodes.len(), 10, "header + one interior node = 10 u32s");
        let root_off = dag.nodes[0] as usize;
        assert_eq!(root_off, 1, "root offset points past header");
        assert_eq!(
            dag.nodes[root_off], 0xffu32,
            "all 8 octants populated for a nonzero leaf root"
        );
        for i in 0..8 {
            assert_eq!(
                dag.nodes[root_off + 1 + i],
                LEAF_BIT | mat3 as u32,
                "child slot {i} must encode the leaf state inline",
            );
        }
        assert_eq!(dag.root_level, 5);

        // Trace a ray straight through the center; must hit the raw packed
        // cell word for material 3 (not the bare material id).
        let ro = [-1.0, 0.5, 0.5];
        let rd = [1.0, 0.0, 0.0];
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        assert_eq!(
            result.hit_cell,
            Some(mat3 as u32),
            "ray through a uniform leaf-root world must hit, events: {:#?}",
            result.events,
        );
    }

    /// Regression for hash-thing-nch I4: empty leaf roots (state 0) build
    /// into an all-empty DAG — the mask is 0, traversal never enters, and
    /// rays miss cleanly instead of panicking in `write_node`.
    #[test]
    fn empty_leaf_root_builds_dag_with_zero_mask() {
        let mut store = NodeStore::new();
        let root = store.leaf(0);
        let dag = Svdag::build(&store, root, 4);
        assert_eq!(dag.node_count, 1);
        let root_off = dag.nodes[0] as usize;
        assert_eq!(dag.nodes[root_off], 0u32, "empty leaf root → zero mask");
        for i in 0..8 {
            assert_eq!(
                dag.nodes[root_off + 1 + i],
                LEAF_BIT,
                "empty leaf slot is LEAF_BIT with state=0"
            );
        }

        let ro = [-1.0, 0.5, 0.5];
        let rd = [1.0, 0.0, 0.0];
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        assert_eq!(
            result.hit_cell, None,
            "ray through empty world must miss cleanly"
        );
        assert!(
            !result.exhausted,
            "empty leaf-root miss must not trip the budget"
        );
    }

    /// Regression: grazing rays at deep depth must make forward progress
    /// through empty space in bounded steps. Before hash-thing-27m the
    /// step-past used a fixed `+EPS` nudge, which on grazing rays (small
    /// |rd_axis|) produced a per-axis positional advance of `|rd_axis| * EPS`
    /// that fell below f32 ULP near pos ~ 0.5, stalling the traversal on
    /// octant boundaries until MAX_STEPS fired.
    ///
    /// These three cases were captured from a fuzz sweep (seed 0xC0FFEE_u64,
    /// `|rd| ~ 1e-3 to 3e-2`, depth 12) where the old code stalled 711/2000
    /// times. With the fix, every case here resolves in <20 steps.
    #[test]
    fn grazing_ray_deep_forward_progress() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        // Single deep voxel near the center so grazing rays sweep a lot of
        // empty space at MAX_DEPTH before hitting (or missing).
        root = store.set_cell(root, 2048, 2048, 2048, mat(1));
        let dag = Svdag::build(&store, root, 12);

        // (origin, raw_direction) — captured from fuzz. Directions are not
        // pre-normalized; the test normalizes below.
        let cases: [([f32; 3], [f32; 3]); 3] = [
            (
                [-1.4759017, 0.49990046, 0.50164545],
                [0.99999964, 5.037533e-5, -0.00083276606],
            ),
            (
                [2.2919006, 0.4996382, 0.50008905],
                [-1.0, 0.00020190649, -4.9705064e-5],
            ),
            (
                [0.50007397, 0.5001334, 2.438742],
                [-3.8144954e-5, -6.881864e-5, -1.0],
            ),
        ];

        for (i, (ro, rd)) in cases.iter().enumerate() {
            let rd_n = normalize(*rd);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, *ro, rd_n, false);
            // 400 << MAX_STEPS (512). Pre-fix code stalled at 512 on all three.
            // Post-fix, observed step count is ≤ 20 on every case.
            assert!(
                result.steps < 400,
                "case {i} (ro={ro:?}, rd={rd:?}): forward-progress regression — \
                 used {steps} steps out of MAX_STEPS=512. Old code stalled here.",
                steps = result.steps,
            );
        }
    }

    /// Regression: far-camera rays must make forward progress even when
    /// the ray-parameter `t` reaches values where `ULP(t)` approaches the
    /// step-past nudge size. Pre-fix (27m v1), this class stalled because
    /// `t_new = oct_far + dt` rounded back to `t_old` in `f32` whenever
    /// `dt < ULP(t)`. Fixed by clamping `ro` to the root-cube entry point
    /// (Laine-Karras), which bounds the in-loop traversal length to
    /// `≤ √3` and keeps `ULP(t_local)` ≤ ~2.4e-7 — always below the
    /// depth-scaled nudge of 9.5e-7.
    ///
    /// Codex Critical reproducer (captured during 27m trident review):
    /// `rd = (-0.86086446, 0.16658753, 0.48079199)`,
    /// `ro ≈ (12.78218, -2.47332, -7.13861)` stalls pre-clamp with
    /// `t ≈ 14.847`, `oct_far ≈ 14.846751`, `dt ≈ 1.11e-6`, and
    /// `t_new = oct_far + dt` rounds back to exactly `t`.
    // The fuzz reproducer is pasted verbatim — don't round it down.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn far_camera_forward_progress() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        root = store.set_cell(root, 2048, 2048, 2048, mat(1));
        let dag = Svdag::build(&store, root, 12);

        // Each case starts the camera well outside the unit root cube so
        // the pre-clamp `t` enters the far-camera regime early in the loop.
        let cases: [([f32; 3], [f32; 3]); 3] = [
            // Codex Critical reproducer (un-normalized direction).
            (
                [12.78218, -2.47332, -7.13861],
                [-0.86086446, 0.16658753, 0.480_792],
            ),
            // Axis-dominant far-camera ray with tiny secondary axes —
            // lands on grazing inter-octant faces at depth ≥ 12.
            ([10.0, 0.5001, 0.4999], [-1.0, 0.0001, -0.0001]),
            // Far camera on a different axis with a different grazing pair.
            ([0.5, 0.5001, -9.5], [0.0001, -0.0001, 1.0]),
        ];

        for (i, (ro, rd)) in cases.iter().enumerate() {
            let rd_n = normalize(*rd);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, *ro, rd_n, false);
            assert!(
                result.steps < 400,
                "case {i} (ro={ro:?}, rd={rd:?}): far-camera forward-progress \
                 regression — used {steps} steps out of MAX_STEPS=512.",
                steps = result.steps,
            );
        }
    }

    /// Regression: ro clamping must also handle the actual deepest
    /// supported depth (MAX_DEPTH=14) where the smallest nudge
    /// (`half/64 = 9.5e-7`) is closest to `f32` ULP and where the
    /// un-clamped pre-fix v1 still stalled on far-camera rays with
    /// any reasonable |ro|.
    #[test]
    fn far_camera_forward_progress_max_depth() {
        let mut store = NodeStore::new();
        let mut root = store.empty(14);
        // Single voxel near the center at full MAX_DEPTH resolution.
        // At depth 14, the cell size is 1/16384 ≈ 6.1e-5.
        root = store.set_cell(root, 8192, 8192, 8192, mat(1));
        let dag = Svdag::build(&store, root, 14);

        // Far camera (|ro| ~ 10) with dominant + grazing axes so the ray
        // descends through deep octants.
        let cases: [([f32; 3], [f32; 3]); 3] = [
            ([10.5, 0.49995, 0.50005], [-1.0, 3e-5, -1e-5]),
            ([0.50003, 10.5, 0.49997], [-1e-5, -1.0, 2e-5]),
            ([0.49999, 0.50002, -10.5], [2e-5, -1e-5, 1.0]),
        ];

        for (i, (ro, rd)) in cases.iter().enumerate() {
            let rd_n = normalize(*rd);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, *ro, rd_n, false);
            assert!(
                result.steps < 400,
                "case {i} (ro={ro:?}, rd={rd:?}): far-camera MAX_DEPTH \
                 forward-progress regression — used {steps} steps out of \
                 MAX_STEPS=512.",
                steps = result.steps,
            );
        }
    }

    /// Static math invariant behind the 27m v2 `ro` clamp: in the
    /// clamped frame, the maximum `|pos|` during traversal is bounded
    /// by `√3 + √3 ≈ 3.46` (clamped `ro_local` is on a cube face, so
    /// `|ro_local| ≤ √3`; the in-cube traversal adds at most `√3` more
    /// along `rd`). For every `t` actually seen inside the loop,
    /// `ULP(t) ≤ ULP(3.46) ≈ 4.77e-7`, which must stay strictly below
    /// the depth-14 step-past nudge `half/64 = 9.54e-7`. If any future
    /// change breaks this inequality — shrinking the nudge, increasing
    /// MAX_DEPTH, or removing the clamp — this test fires.
    #[test]
    fn ro_clamp_preserves_ulp_margin() {
        // Worst-case |pos| after ro clamping: root face + cube diagonal.
        let max_pos: f32 = 2.0 * (3f32.sqrt());
        let ulp_max = f32::from_bits(max_pos.to_bits() + 1) - max_pos;

        // Smallest nudge the step-past will ever produce (deepest cell).
        // At MAX_DEPTH=14, stack_half = 0.5^15 ≈ 3.05e-5.
        const MAX_DEPTH: i32 = 14;
        const EPS: f32 = 1e-5;
        let deepest_half = 0.5_f32.powi(MAX_DEPTH + 1);
        let min_nudge = (deepest_half * (1.0 / 64.0)).min(EPS);

        assert!(
            min_nudge > ulp_max,
            "27m invariant: min nudge {min_nudge:e} must exceed max \
             post-clamp ULP {ulp_max:e} to guarantee forward progress. \
             Breaking this invariant re-opens the zero-progress regime \
             (Codex Critical blocker, 27m review)."
        );
    }

    /// Throughput regression for Claude Critical's deep-level concern:
    /// a dominant-axis ray through a mostly-empty MAX_DEPTH scene should
    /// converge in a reasonable step budget. Claude Critical feared that
    /// `min(half/64, 1e-5)` would starve dominant-axis rays at depth 13
    /// by reducing per-step advance to ~9.5e-7. That analysis conflated
    /// `dt` (ULP padding) with the total step advance. Per-step advance
    /// is `(oct_far - t_prev) + dt`, dominated by the cell-exit time
    /// `oct_far - t_prev` which is O(cell_span / |rd|) — 128x larger
    /// than `dt` for dominant-axis rays. Sparse scenes coalesce empty
    /// space into shallow cells, so most step-pasts happen at coarse
    /// levels anyway. This test asserts the actual observed behavior:
    /// hitting or missing a single-voxel scene at root_level=14 from
    /// multiple ray directions completes in well under MAX_STEPS/4.
    #[test]
    fn dominant_axis_throughput_max_depth() {
        let mut store = NodeStore::new();
        let mut root = store.empty(14);
        root = store.set_cell(root, 8192, 8192, 8192, mat(1));
        let dag = Svdag::build(&store, root, 14);

        // Dominant-axis rays hitting and missing the single deep voxel
        // from different entry angles. `|ro|` kept small here because
        // this test is about throughput, not far-camera clamp behavior
        // (that's covered by step_past_t_is_local_frame).
        let cases: [([f32; 3], [f32; 3]); 6] = [
            // x-axis hit
            ([-0.5, 0.5, 0.5], [1.0, 0.0, 0.0]),
            // y-axis hit
            ([0.5, -0.5, 0.5], [0.0, 1.0, 0.0]),
            // z-axis hit
            ([0.5, 0.5, -0.5], [0.0, 0.0, 1.0]),
            // x-axis miss (grazing)
            ([-0.5, 0.25, 0.25], [1.0, 0.0, 0.0]),
            // Diagonal hit through center
            ([-0.5, -0.5, -0.5], [1.0, 1.0, 1.0]),
            // Axis-dominant with tiny perpendicular components
            ([-0.5, 0.5001, 0.4999], [1.0, 1e-4, -1e-4]),
        ];

        for (i, (ro, rd)) in cases.iter().enumerate() {
            let rd_n = normalize(*rd);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, *ro, rd_n, false);
            assert!(
                result.steps < 128,
                "case {i} (ro={ro:?}, rd={rd:?}): dominant-axis throughput \
                 regression at root_level=14 — used {steps} steps, expected \
                 < 128. If this fires, Claude Critical's `min` vs `max` \
                 concern may be warranted after all.",
                steps = result.steps,
            );
        }
    }

    /// Direct regression for the 27m v2 `ro` clamp: after clamping, the
    /// in-loop `t` variable is local-frame and bounded by the in-cube
    /// traversal length (≤ √3 plus a tiny EPS overshoot). Pre-clamp,
    /// `t` is world-space and scales with `|ro|`. This test traces a
    /// far-camera ray and asserts every `StepPast.t_new` event stays
    /// below a cube-diagonal bound. If someone reverts the clamp,
    /// `t_new` will be ~100x larger on this case and the test fails.
    #[test]
    fn step_past_t_is_local_frame() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        root = store.set_cell(root, 2048, 2048, 2048, mat(1));
        let dag = Svdag::build(&store, root, 12);

        // |ro| ~ 100 — pre-clamp world-space t reaches ~99.5-100.5
        // inside the loop. Post-clamp local-frame t stays ≤ √3 + ε.
        let ro: [f32; 3] = [100.5, 0.5001, 0.4999];
        let rd_n = normalize([-1.0, 1e-5, -1e-5]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd_n, true);

        // Cube diagonal plus a small margin for overshoot.
        const T_BOUND: f32 = 2.0;
        let mut observed_max: f32 = 0.0;
        for ev in &result.events {
            if let cpu_trace::TraceEvent::StepPast { t_new, .. } = ev {
                if *t_new > observed_max {
                    observed_max = *t_new;
                }
                assert!(
                    *t_new <= T_BOUND,
                    "27m v2 regression: StepPast.t_new = {t_new} exceeds \
                     the local-frame bound {T_BOUND}. The `ro` clamp has \
                     been reverted or broken — `t` should be local-frame \
                     after clamping `ro` to the root entry (|t_local| ≤ √3)."
                );
            }
        }
        assert!(
            observed_max > 0.0,
            "expected at least one StepPast event to sanity-check the bound",
        );
    }

    /// Permanent seeded property test for the 27m step-past fix.
    ///
    /// The `grazing_ray_deep_forward_progress` test above preserves three
    /// hand-plucked tuples from an ad-hoc fuzz sweep that caught the
    /// 711/2000 stall rate before 27m. Those tuples are the fossils; this
    /// test is the sweep. Per Claude Critical / Codex Evolutionary / Gemini
    /// Evolutionary in the 27m trident review: "the three frozen cases are
    /// a floor, not a ceiling — promote the fuzz harness to a permanent
    /// seeded property test so future regressions catch the whole
    /// distribution." (hash-thing-ysg).
    ///
    /// Coverage is **distributional, not worst-case**: this test samples
    /// `|rd_secondary|` uniformly in `[-3e-2, 3e-2]`, while the fossils
    /// live at `|rd_secondary| ~ 5e-5` (much deeper into the grazing
    /// regime). Both tests are kept — fossils catch regressions at the
    /// f32 frontier, this sweep catches regressions across the bulk of
    /// the distribution.
    ///
    /// Shape: 2000 axis-dominant rays at seed `0xC0FFEE_u64`, with grazing
    /// secondary `rd` components in `[-3e-2, 3e-2]`, targeting a single
    /// depth-12 voxel near the cube center. For each case:
    ///   1. `steps < 400` — forward progress within the budget (the
    ///      original stall symptom was hitting MAX_STEPS=512).
    ///   2. Every `StepPast` event has strict `t_new > t_old` — pre-27m
    ///      far-camera rays produced `t_new == t_old` (Codex Critical
    ///      zero-forward-progress blocker, resolved by ro clamp in v2).
    ///   3. Aggregate hit count agrees with analytical ray-AABB ground
    ///      truth exactly (or within 2) — catches a "fast miss" regression
    ///      where the traversal silently drops hits (Codex Standard §1
    ///      concern).
    ///
    /// Observed values at commit time (pinned for future readers, not
    /// asserted exactly): `total_steps=6149, traversal_hits=0,
    /// analytic_hits=0, diff=0`. The ~3 steps/ray average comes from the
    /// sparse scene (single voxel in a 4096³ grid coalesces all empty
    /// space into shallow DAG cells, so most rays exit in 2-3 step-pasts).
    /// Zero hits is expected: a ~2.4e-4-wide voxel vs secondary positions
    /// spread over ~0.14 at cube entry gives hit probability ~(1e-3)²
    /// per ray, rounding to 0 per 2000-ray batch.
    #[test]
    fn fuzz_grazing_forward_progress_property() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        root = store.set_cell(root, 2048, 2048, 2048, mat(1));
        let dag = Svdag::build(&store, root, 12);

        // Tight AABB of the single depth-12 voxel in world coordinates.
        // At depth 12, cell width is 1/4096 ≈ 2.44e-4.
        let voxel_min: [f32; 3] = [0.5, 0.5, 0.5];
        let voxel_max: [f32; 3] = [2049.0 / 4096.0, 2049.0 / 4096.0, 2049.0 / 4096.0];

        // Analytical ray-AABB: `t_far >= max(t_near, 0)` → hit. Uses the
        // same slab math as cpu_trace::intersect_aabb so we're comparing
        // against an oracle written in the same floating-point regime.
        //
        // NOTE: this oracle is valid *only* because this geometry has a
        // single leaf that exactly fills one depth-12 AABB. In a
        // multi-voxel scene, "ray hits the scene's bounding box" is not
        // the same as "ray hits a non-empty leaf," and this oracle would
        // no longer be ground truth. Do not generalize without also
        // switching to a scene-wide occupancy check.
        fn hits_voxel(ro: [f32; 3], rd: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
            let inv = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];
            let mut t_near = f32::NEG_INFINITY;
            let mut t_far = f32::INFINITY;
            for i in 0..3 {
                let t1 = (bmin[i] - ro[i]) * inv[i];
                let t2 = (bmax[i] - ro[i]) * inv[i];
                t_near = t_near.max(t1.min(t2));
                t_far = t_far.min(t1.max(t2));
            }
            t_far >= t_near.max(0.0)
        }

        // Same LCG the other svdag fuzz tests use, seeded at the value
        // originally used to mine the three frozen tuples. Single closure
        // so we don't fight the borrow checker with two independent
        // closures both capturing `seed` mutably.
        let mut seed: u64 = 0xC0FFEE_u64;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        // Unit-interval [0, 1) helper. Uses the high 24 bits so the result
        // stays strictly below 1.0 in f32 (`u32::MAX / (2^24)` rounds to
        // exactly 1.0 in f32 with the naïve form, violating [0, 1)).
        let u32_to_unit = |x: u32| ((x >> 8) as f32) / ((1u32 << 24) as f32);

        let mut total_steps: u64 = 0;
        let mut traversal_hits: i32 = 0;
        let mut analytic_hits: i32 = 0;

        for i in 0..2000 {
            // Dominant axis: {0, 1, 2}. Sign: {+, -}.
            let axis = (next_u32() % 3) as usize;
            let dom_pos = (next_u32() & 1) == 0;

            // Dominant ro: outside the unit cube on one side so the ray
            // enters. Pointing toward the cube requires rd_dom's sign to
            // oppose ro_dom's side.
            let dom_u = u32_to_unit(next_u32());
            let dom_ro: f32 = if dom_pos {
                2.0 + dom_u * 2.0 // +2 .. +4, ray heads -axis
            } else {
                -1.0 - dom_u * 2.0 // -3 .. -1, ray heads +axis
            };

            // Secondary ro near 0.5 (close to the voxel) with ±0.01 jitter
            // — large enough to produce a mix of near-grazing hits and
            // near-grazing misses.
            let ro_a = 0.5 + (u32_to_unit(next_u32()) - 0.5) * 0.02;
            let ro_b = 0.5 + (u32_to_unit(next_u32()) - 0.5) * 0.02;

            let mut ro = [0f32; 3];
            ro[axis] = dom_ro;
            ro[(axis + 1) % 3] = ro_a;
            ro[(axis + 2) % 3] = ro_b;

            // Dominant rd: unit magnitude on the dominant axis, sign
            // opposite to ro_dom so the ray points into the cube.
            let rd_dom: f32 = if dom_pos { -1.0 } else { 1.0 };

            // Grazing secondaries in [-3e-2, +3e-2]. Nudge away from zero
            // so `1.0 / rd` never produces a ±inf — the traversal handles
            // it but the ground-truth AABB oracle would divide by zero.
            let mut rd_a = (u32_to_unit(next_u32()) - 0.5) * 6e-2;
            let mut rd_b = (u32_to_unit(next_u32()) - 0.5) * 6e-2;
            if rd_a.abs() < 1e-6 {
                rd_a = if rd_a < 0.0 { -1e-6 } else { 1e-6 };
            }
            if rd_b.abs() < 1e-6 {
                rd_b = if rd_b < 0.0 { -1e-6 } else { 1e-6 };
            }

            let mut rd_raw = [0f32; 3];
            rd_raw[axis] = rd_dom;
            rd_raw[(axis + 1) % 3] = rd_a;
            rd_raw[(axis + 2) % 3] = rd_b;
            let rd_n = normalize(rd_raw);

            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd_n, true);

            // Hard assertion #1: forward progress within the step budget.
            assert!(
                result.steps < 400,
                "ysg fuzz case {i} (axis={axis}, ro={ro:?}, rd={rd_n:?}): \
                 forward-progress regression — used {steps} steps out of \
                 MAX_STEPS=512. Pre-27m, ~35% of this distribution stalled \
                 at MAX_STEPS.",
                steps = result.steps,
            );

            // Hard assertion #2: every step-past strictly advances `t`.
            // Pre-27m-v2 far-camera rays hit a `t_new == t_old` regime
            // (Codex Critical blocker).
            for (ev_idx, ev) in result.events.iter().enumerate() {
                if let cpu_trace::TraceEvent::StepPast { t_old, t_new } = ev {
                    assert!(
                        t_new > t_old,
                        "ysg fuzz case {i} event {ev_idx}: zero-forward-\
                         progress regression — t_old={t_old} t_new={t_new} \
                         (ro={ro:?}, rd={rd_n:?}). The 27m v2 ro clamp has \
                         been broken.",
                    );
                }
            }

            total_steps += result.steps as u64;
            if let Some(hit) = result.hit_cell {
                // Defensive: the scene has exactly one non-empty material
                // — the raw packed word `mat(1)` (= `material=1 << 6 = 64`).
                // Hitting anything else indicates a corruption in the DAG
                // or the traversal reporting an empty leaf as hit.
                let expected = mat(1) as u32;
                assert_eq!(
                    hit, expected,
                    "ysg fuzz case {i}: unexpected hit word {hit}, \
                     only packed cell {expected} exists in this scene",
                );
                traversal_hits += 1;
            }
            if hits_voxel(ro, rd_n, voxel_min, voxel_max) {
                analytic_hits += 1;
            }
        }

        // Aggregate hit-count sanity (Codex Standard §1): a "fast miss"
        // regression would drop traversal_hits to zero while analytic_hits
        // stays put. A `>|tolerance|` gap indicates systematic hit/miss
        // drift. Exact match is unrealistic at the f32 grazing frontier
        // (a handful of tangent rays can disagree on tangent-boundary
        // cases), so we allow a small slack.
        //
        // With this geometry (single depth-12 voxel, |rd_secondary| up
        // to 3e-2, secondary jitter ±0.01), the voxel is ~2.4e-4 wide so
        // most rays miss; both counts should be small and agree to within
        // a handful.
        let diff = (traversal_hits - analytic_hits).abs();
        assert!(
            diff <= 2,
            "ysg fuzz hit-count drift: traversal={traversal_hits} \
             analytic={analytic_hits} diff={diff} (tolerance 2). \
             Observed at commit time was exactly 0; allowing 2 for future \
             edits that might flip one or two tangent-boundary cases. \
             A larger gap indicates systematic hit/miss drift.",
        );

        // Sanity: assert we actually exercised the traversal non-trivially.
        // Observed at commit time: ~6149 total steps (≈3 per ray — the
        // sparse single-voxel scene coalesces empty space into shallow
        // cells, so rays exit in 2–3 step-pasts). Floor is set well below
        // the observed value but high enough to catch a "harness stopped
        // exercising the loop" regression (which would drop to 2000 or
        // lower if every ray short-circuited immediately).
        assert!(
            total_steps > 4000,
            "ysg fuzz: total_steps {total_steps} too low — the harness \
             isn't exercising step-past paths (expected >4000, observed \
             ~6149 at commit time).",
        );
    }

    /// hash-thing-6cc follow-up to 6hd: a midpoint-exact fuzz that pins the
    /// `octant_of` tiebreak fix end-to-end through `cpu_trace::raycast`,
    /// not just at the helper boundary. The 6hd unit tests cover
    /// `octant_of` directly with hand-built float inputs; this test asks
    /// the harder question: does a ray that step-pasts ACROSS a level-
    /// internal midpoint plane with rd<0 on the tied axis still make
    /// forward progress and terminate cleanly?
    ///
    /// Construction: ro sits exactly on the y = 0.5 root midpoint plane
    /// (`ro.y = 0.5_f32`, the bit-exact midpoint of the unit cube), with
    /// `|rd_y|` chosen so the per-step nudge `dt * |rd_y|` underflows
    /// `ULP(0.5) ≈ 5.96e-8`. Under that condition, the post-step `pos.y`
    /// stays at exactly `0.5_f32` for many steps in a row, repeatedly
    /// triggering the `octant_of` midpoint comparator with `rd.y < 0` —
    /// the exact pattern that pre-6hd picked the wrong sibling on
    /// (Codex reproducer `rd = (1.0, -1e-6, 0.0)`).
    ///
    /// The dominant x axis is the workhorse — large `|rd_x|` so the
    /// step-past nudge advances `pos.x` past each octant boundary
    /// normally. The y axis is the tied/secondary axis. Z is forced to
    /// zero, mirroring the Codex reproducer pattern (no z motion → z
    /// component of pos is constant → z classification is direction-
    /// independent and uninvolved in the test).
    ///
    /// Asserted invariants (same shape as the 27m-derived sweep above):
    ///   1. `steps < 400` — forward progress within the budget. Pre-6hd
    ///      this didn't necessarily stall (the wrong-sibling pick still
    ///      makes progress), but a future regression that resurfaces the
    ///      `pos == mid` ambiguity AND breaks step-past nudging would
    ///      hit MAX_STEPS. Belt-and-braces.
    ///   2. Every `StepPast` event has strict `t_new > t_old` — the
    ///      27m v2 invariant; the 6hd fix sits on top of it, so any
    ///      regression there would also surface here.
    ///
    /// Scene: level-2 single-leaf at (2,2,2), so traversal must cross
    /// the y = 0.5 midpoint plane to descend. Tractable in f32 without
    /// rounding drift, per the 6cc bead's "small scene" guidance.
    #[test]
    fn fuzz_midpoint_corner_forward_progress_property() {
        let mut store = NodeStore::new();
        let mut root = store.empty(2);
        root = store.set_cell(root, 2, 2, 2, mat(1));
        let dag = Svdag::build(&store, root, 2);

        let mut seed: u64 = 0xBADF00D_u64;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        let u32_to_unit = |x: u32| ((x >> 8) as f32) / ((1u32 << 24) as f32);

        let mut total_steps: u64 = 0;
        let mut midpoint_pos_witnessed = 0;

        for i in 0..500 {
            // x ro outside the cube on the -x side. Dominant rd is +x.
            let ro_x = -0.5 - u32_to_unit(next_u32()) * 1.5; // -2.0 .. -0.5
                                                             // ro.y = exactly 0.5 (bit-exact root midpoint plane).
            let ro_y = 0.5_f32;
            // ro.z anywhere strictly inside [0, 1) but away from edges.
            let ro_z = 0.1 + u32_to_unit(next_u32()) * 0.8;
            let ro = [ro_x, ro_y, ro_z];

            // Pre-normalize rd. |rd_y| in [1e-7, 1.1e-5] — small enough
            // that `dt * |rd_y|` underflows ULP(0.5) for every dt the
            // traversal will produce on this scene (depth 2 → cell width
            // 0.25, max single-step dt < 0.5, so dt * 1.1e-5 ≈ 5.5e-6,
            // which is two orders of magnitude above ULP(0.5)... but
            // still small enough that pos.y rounds back to 0.5 over the
            // first few steps before drifting away).
            //
            // Critically: rd_y is ALWAYS negative — this is the codex-
            // reproducer pattern (rd_y < 0 + pos.y == mid → pre-6hd
            // picked wrong sibling because `>=` biased to HIGH).
            let rd_y = -(1e-7 + u32_to_unit(next_u32()) * 1e-5);
            let rd_raw = [1.0_f32, rd_y, 0.0_f32];
            let rd_n = normalize(rd_raw);

            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd_n, true);

            assert!(
                result.steps < 400,
                "6cc midpoint case {i} (ro={ro:?}, rd={rd_n:?}): forward-\
                 progress regression — used {steps} steps out of 400 floor.",
                steps = result.steps,
            );

            for (ev_idx, ev) in result.events.iter().enumerate() {
                if let cpu_trace::TraceEvent::StepPast { t_old, t_new } = ev {
                    assert!(
                        t_new > t_old,
                        "6cc midpoint case {i} event {ev_idx}: zero-forward-\
                         progress regression — t_old={t_old} t_new={t_new} \
                         (ro={ro:?}, rd={rd_n:?}). 27m v2 ro clamp broken \
                         OR 6hd tiebreak regressed.",
                    );
                }
            }

            // Witness count: did we actually exercise the midpoint
            // tiebreak path? The harness construction guarantees ro.y
            // == 0.5 exactly going in, so AT LEAST the first descend
            // step must call `octant_of` with pos.y == 0.5. We can't
            // observe `octant_of` calls directly from the trace, but
            // we can infer the witness from the step count: any case
            // that took at least one descend step witnessed the
            // tiebreak with our constructed ro.
            if result.steps > 0 {
                midpoint_pos_witnessed += 1;
            }
            total_steps += result.steps as u64;
        }

        // Sanity: every constructed case must witness the tiebreak.
        // If this drops below 500, the harness isn't exercising the
        // path it was built for.
        assert_eq!(
            midpoint_pos_witnessed, 500,
            "6cc fuzz: only {midpoint_pos_witnessed}/500 cases produced \
             a non-zero step count — the harness isn't reaching the \
             octant_of tiebreak path.",
        );

        // Sanity: the harness exercised the traversal non-trivially.
        // Each ray must take at least one step (the descend that hits
        // octant_of with pos.y == 0.5). The level-2 sparse scene
        // coalesces empty space into shallow cells, so most rays exit
        // in 2 steps (descend → step-past → exit-root). Floor of 500
        // = "every ray got at least one step"; observed at commit time
        // was exactly 1000 (2 per ray).
        assert!(
            total_steps >= 500,
            "6cc fuzz: total_steps {total_steps} too low — the harness \
             isn't reaching the descend loop on every ray.",
        );
    }

    /// hash-thing-2w5: The step budget formula must honor its floor and its
    /// per-cell fudge. Pinning the observable values keeps future tuning
    /// honest — raising MIN_STEP_BUDGET or STEP_BUDGET_FUDGE should be a
    /// visible test change, not a silent drift.
    #[test]
    fn step_budget_respects_floor_and_fudge() {
        // Shallow: fudge × side < floor → floor wins.
        assert_eq!(cpu_trace::step_budget(0), 1024); // 8 * 1 = 8, floored to 1024
        assert_eq!(cpu_trace::step_budget(6), 1024); // 8 * 64 = 512, floored to 1024
        assert_eq!(cpu_trace::step_budget(7), 1024); // 8 * 128 = 1024, still floor
                                                     // Deep: fudge × side > floor → fudge × side wins.
        assert_eq!(cpu_trace::step_budget(8), 2048); // 8 * 256
        assert_eq!(cpu_trace::step_budget(12), 32768); // 8 * 4096 (SPEC target)
        assert_eq!(cpu_trace::step_budget(14), 131072); // 8 * 16384 (MAX_DEPTH)
                                                        // Saturation: an absurd root_level must not panic. The .min(28)
                                                        // clamp keeps the shift in-range AND keeps `root_side * 8` within
                                                        // u32 so the shader mirror can use the same arithmetic. At
                                                        // root_level=28, budget = (1 << 28) * 8 = 2^31 = 2_147_483_648.
                                                        // Exact equality pins the clamp: a future refactor that drops
                                                        // the .min(28) would silently produce a different saturation
                                                        // value (2^33 pre-ysg review catch).
        assert_eq!(
            cpu_trace::step_budget(u32::MAX),
            1usize << 31,
            "step_budget saturation value must match the .min(28) clamp"
        );
        // Boundary: root_level=28 equals the saturation value.
        assert_eq!(cpu_trace::step_budget(28), 1usize << 31);
        // One below the clamp: root_level=27 gives 2^30.
        assert_eq!(cpu_trace::step_budget(27), 1usize << 30);
    }

    /// hash-thing-2w5 invariant pin: at SPEC-target depth (root_level=12,
    /// i.e. 4096³), a single voxel placed at the far corner must still be
    /// reachable via a corner-to-corner ray without the traversal budget
    /// being exhausted. This is *not* a budget-pressure test: single-voxel
    /// sparse scenes coalesce empty space into shallow leaves, so a
    /// full-diagonal ray completes in ~5 outer iterations regardless of
    /// root_level. What this test DOES pin:
    ///
    ///   1. The `exhausted: false` path is exercised on a realistic hit.
    ///      Gives the flag a non-trivial non-exhausted witness.
    ///   2. Hit correctness at SPEC-scale — if a future refactor breaks
    ///      the far-corner traversal this catches it separately from
    ///      budget mechanics.
    ///   3. `step_budget(12) = 32768` is not secretly clamped to something
    ///      smaller than the actually-used step count.
    ///
    /// What this test does NOT pin (and the review artifact calls out
    /// explicitly): that any natural single-voxel scene requires more
    /// than the pre-2w5 MAX_STEPS=512. It doesn't — sparse coalescing
    /// makes budget bite unreachable with single-voxel scenes. The
    /// follow-on `budget_exhaustion_via_low_budget_override` test
    /// exercises the exhausted=true path via an artificially low budget
    /// instead, which is the only reliable way to surface the flag
    /// without constructing a dense pathological scene.
    #[test]
    fn sparse_far_corner_hit_is_not_exhausted_depth12() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        let mat1 = mat(1);
        root = store.set_cell(root, 4095, 4095, 4095, mat1);
        let dag = Svdag::build(&store, root, 12);
        assert_eq!(dag.root_level, 12);

        let ro = [2.0, 2.0, 2.0];
        let target = [4095.5 / 4096.0, 4095.5 / 4096.0, 4095.5 / 4096.0];
        let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);

        assert!(
            !result.exhausted,
            "far-corner depth-12 hit exhausted the budget — expected false, \
             got true (steps={})",
            result.steps,
        );
        assert_eq!(result.hit_cell, Some(mat1 as u32));
        assert!(result.steps < cpu_trace::step_budget(12));
    }

    /// hash-thing-2w5 exhausted-flag witness. Forces the budget-exhaustion
    /// branch via `raycast_with_budget` with an artificially low budget,
    /// since natural single-voxel scenes coalesce empty space aggressively
    /// and never organically need more than a few outer iterations.
    ///
    /// Why this matters: the only way a `raycast` caller ever observes
    /// `exhausted = true` without an override would require a dense
    /// pathological scene. Without this test, a future refactor that
    /// silently dropped the post-loop fall-through (e.g. changed it to
    /// `exhausted: false`, or removed the `exhausted` field entirely)
    /// would compile clean and pass every other test in this module.
    #[test]
    fn budget_exhaustion_via_low_budget_override() {
        // Empty depth-12 scene: every root child is an empty leaf, so
        // the diagonal ray walks through multiple empty root octants in
        // sequence before t_exit. That guarantees a natural step count of
        // > 1, which is what the artificially-low budget needs to bite.
        let mut store = NodeStore::new();
        let root = store.empty(12);
        let dag = Svdag::build(&store, root, 12);

        // Diagonal ray from (2,2,2) toward (-1,-1,-1). Passes through
        // multiple root-level octants before exiting at (0,0,0).
        let ro = [2.0, 2.0, 2.0];
        let rd = normalize([-1.0, -1.0, -1.0]);

        // Sanity: at natural budget the ray exits cleanly (miss), not
        // exhausted, with steps > 1 — otherwise the budget=1 witness
        // below would be vacuously satisfied.
        let natural = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        assert_eq!(natural.hit_cell, None);
        assert_eq!(
            natural.hit_normal, None,
            "clean miss must carry no normal (hash-thing-rv4 invariant)"
        );
        assert!(
            !natural.exhausted,
            "natural diagonal miss must not exhaust at budget=32768"
        );
        let natural_steps = natural.steps;
        assert!(
            natural_steps > 1,
            "prerequisite: natural step count must be > 1 for the exhaustion \
             witness to be non-vacuous (observed {natural_steps})",
        );

        // Force exhaustion. `raycast_with_budget` uses the exact body of
        // `raycast`, so this path exercises the same post-loop fall-through
        // that the shader's magenta sentinel is tied to.
        let budget = 1;
        let result = cpu_trace::raycast_with_budget(&dag.nodes, budget, ro, rd, false);
        assert!(
            result.exhausted,
            "raycast_with_budget({budget}) must set exhausted=true when \
             the natural traversal ({natural_steps} steps) exceeds the \
             override — got exhausted={}, steps={}, hit={:?}",
            result.exhausted, result.steps, result.hit_cell,
        );
        assert_eq!(
            result.hit_cell, None,
            "exhausted traversal must report no hit (CPU path must match \
             shader behavior where the magenta sentinel replaces any hit)",
        );
        assert_eq!(
            result.hit_normal, None,
            "exhausted traversal must carry no normal — guards the \
             post-loop fall-through (hash-thing-rv4 invariant)",
        );
        assert_eq!(
            result.steps, budget,
            "exhausted steps count should equal the budget"
        );
    }

    /// hash-thing-2w5: the `raycast` → `raycast_with_budget` wrapper must
    /// produce byte-identical results when called with the same effective
    /// budget. Without this pin, the refactor split could silently diverge.
    #[test]
    fn raycast_and_raycast_with_budget_agree_at_natural_budget() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        root = store.set_cell(root, 4095, 4095, 4095, mat(1));
        let dag = Svdag::build(&store, root, 12);

        let ro = [2.0, 2.0, 2.0];
        let target = [4095.5 / 4096.0, 4095.5 / 4096.0, 4095.5 / 4096.0];
        let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);

        let via_root_level = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        let via_budget = cpu_trace::raycast_with_budget(
            &dag.nodes,
            cpu_trace::step_budget(dag.root_level),
            ro,
            rd,
            false,
        );
        assert_eq!(via_root_level.hit_cell, via_budget.hit_cell);
        assert_eq!(via_root_level.steps, via_budget.steps);
        assert_eq!(via_root_level.exhausted, via_budget.exhausted);
    }

    /// hash-thing-rv4: the SVDAG shader must produce real per-face surface
    /// normals at leaf hits, not a ray-direction shade, or the output
    /// collapses to a flat-sticker look. The CPU replica computes the same
    /// analytical normal as the shader, so asserting the normals here pins
    /// both paths at once — if either drifts, this test breaks first.
    ///
    /// Six axis-aligned rays fire at a single voxel at the root center; each
    /// ray enters through a distinct cube face, and the reported normal must
    /// match that face. The voxel lives at (32,32,32) in a level-6 grid, so
    /// its world-space AABB is [0.5, 0.515625]^3. Ray origins sit well
    /// outside the root cube and aim at the voxel's center.
    #[test]
    fn leaf_hit_normals_match_entry_face() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        let mat1 = mat(1);
        root = store.set_cell(root, 32, 32, 32, mat1);
        let dag = Svdag::build(&store, root, 6);

        // Voxel world-space center (32.5 / 64, 32.5 / 64, 32.5 / 64).
        let cx = 32.5 / 64.0;
        let cy = 32.5 / 64.0;
        let cz = 32.5 / 64.0;

        // (name, origin, expected_normal). Each origin is on the axis
        // through the voxel center, far outside the root cube on one
        // specific side. The expected normal points back toward the origin
        // (opposite ray direction on the entry axis).
        let cases: &[(&str, [f32; 3], [f32; 3])] = &[
            ("-x face", [-3.0, cy, cz], [-1.0, 0.0, 0.0]),
            ("+x face", [3.0, cy, cz], [1.0, 0.0, 0.0]),
            ("-y face", [cx, -3.0, cz], [0.0, -1.0, 0.0]),
            ("+y face", [cx, 3.0, cz], [0.0, 1.0, 0.0]),
            ("-z face", [cx, cy, -3.0], [0.0, 0.0, -1.0]),
            ("+z face", [cx, cy, 3.0], [0.0, 0.0, 1.0]),
        ];

        for (name, ro, expected_normal) in cases {
            let rd = normalize([cx - ro[0], cy - ro[1], cz - ro[2]]);
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, *ro, rd, false);
            // hit_cell is the raw packed cell word, not the decoded
            // material id (post-1v0.1 16-bit tagged encoding).
            assert_eq!(
                result.hit_cell,
                Some(mat1 as u32),
                "{name}: expected hit, got miss (exhausted={})",
                result.exhausted,
            );
            let normal = result.hit_normal.expect("hit must carry a normal");
            assert_eq!(
                normal, *expected_normal,
                "{name}: ray at {ro:?} → rd {rd:?} must yield normal {expected_normal:?}, got {normal:?}",
            );
        }
    }

    /// hash-thing-rv4 tie-breaker: when two axes have equal `tmin` at the
    /// leaf entry (geometric corner hit), the normal-picking cascade must
    /// still produce a unit face normal, never a zero vector. The shader
    /// uses `>=` to break ties toward x then y — the CPU mirror matches.
    /// This test shoots a ray that enters the voxel exactly on an xy edge
    /// and asserts the normal is a valid unit face vector (one of the two
    /// candidates), not a degenerate zero.
    #[test]
    fn leaf_hit_normal_corner_tiebreak_is_unit_face() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        let mat1 = mat(1);
        root = store.set_cell(root, 32, 32, 32, mat1);
        let dag = Svdag::build(&store, root, 6);

        // Ray along (1, 1, 0) direction, aimed at the voxel's center. The
        // entry tmin on x and y will be (nearly) identical; the tie-breaker
        // picks x (because `tmin.x >= tmin.y`).
        let cx = 32.5 / 64.0;
        let cy = 32.5 / 64.0;
        let cz = 32.5 / 64.0;
        let ro = [cx - 1.0, cy - 1.0, cz];
        let rd = normalize([1.0, 1.0, 0.0]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        assert_eq!(result.hit_cell, Some(mat1 as u32), "corner hit must land");

        let normal = result.hit_normal.expect("hit must carry a normal");
        // The cascade's first comparator is `tmin_v[0] >= tmin_v[1]`, so on
        // an exact xy tie x wins and the ray (coming from -x,-y) gives the
        // -x face normal. Pinning the specific axis here is the whole point
        // of the CPU oracle: if someone flips the cascade to `>` instead of
        // `>=`, CPU would pick y while the shader (also `>=`) still picks x,
        // and this assertion will catch the drift before it hits the GPU.
        assert_eq!(
            normal,
            [-1.0, 0.0, 0.0],
            "xy-edge tie-break must pick x (x-first cascade with `>=`), got {normal:?}",
        );
    }

    /// hash-thing-2nd: inside-filled-leaf ray origins (reachable via deep-zoom
    /// orbit camera) must produce an EXIT-face normal pointing in the direction
    /// of travel — not the inward back-face normal the pre-2nd code returned.
    /// Plants the ray origin at the exact center of a single filled voxel at
    /// (32,32,32) in a level-6 grid and fires along each of the six axes. The
    /// exit face on each is the one the ray is heading toward, so the normal
    /// should equal `sign(rd)` (aligned with the direction of travel).
    #[test]
    fn leaf_hit_normal_inside_origin_picks_exit_face() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        let mat1 = mat(1);
        root = store.set_cell(root, 32, 32, 32, mat1);
        let dag = Svdag::build(&store, root, 6);

        // Exact center of the filled voxel at (32,32,32) — world-space
        // AABB [0.5, 0.515625]^3, center at 32.5/64 on every axis.
        let center = [32.5_f32 / 64.0, 32.5 / 64.0, 32.5 / 64.0];

        // (name, rd, expected exit-face normal = sign(rd) on the exit axis).
        let cases: &[(&str, [f32; 3], [f32; 3])] = &[
            ("+x exit", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ("-x exit", [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            ("+y exit", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            ("-y exit", [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]),
            ("+z exit", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
            ("-z exit", [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]),
        ];

        for (name, rd, expected_normal) in cases {
            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, center, *rd, false);
            assert_eq!(
                result.hit_cell,
                Some(mat1 as u32),
                "{name}: expected inside-leaf hit, got miss (exhausted={})",
                result.exhausted,
            );
            let normal = result.hit_normal.expect("hit must carry a normal");
            assert_eq!(
                normal, *expected_normal,
                "{name}: inside-leaf origin with rd {rd:?} must yield exit-face \
                 normal {expected_normal:?}, got {normal:?} (pre-2nd returned \
                 the inward back-face)",
            );
        }

        // Off-axis diagonal inside-origin ray — all three `tmax_v`
        // components are finite, so the cascade picks the true `argmin`
        // rather than relying on `+inf` on non-moving axes. For a ray
        // starting at the voxel center with `rd = (1, 2, 3)`, the per-axis
        // exit distances from center are `half / |rd_axis|`; z has the
        // largest `|rd|`, so z exits first → normal = `[0, 0, +1]`.
        let rd = normalize([1.0, 2.0, 3.0]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, center, rd, false);
        assert_eq!(
            result.hit_cell,
            Some(mat1 as u32),
            "diagonal: expected inside-leaf hit, got miss (exhausted={})",
            result.exhausted,
        );
        let normal = result.hit_normal.expect("hit must carry a normal");
        assert_eq!(
            normal,
            [0.0, 0.0, 1.0],
            "diagonal rd {rd:?} from voxel center must exit through +z \
             (largest |rd_axis|) → normal [0,0,+1], got {normal:?}",
        );
    }

    /// hash-thing-rv4: `hit_normal` must be `None` on every non-hit return
    /// path. The CPU trace has four miss/exhausted sites; this test
    /// exercises the two most reachable ones (clean miss, pre-root miss).
    /// Empty-leaf and exhausted paths are covered by existing tests via
    /// their `hit_cell: None` assertion — the invariant "miss ⇒ no
    /// normal" holds by construction of the struct literals, but pinning
    /// it here is cheap insurance against future edits.
    #[test]
    fn miss_returns_no_normal() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 32, 32, 32, mat(1));
        let dag = Svdag::build(&store, root, 6);

        // 1. Pre-root miss: ray pointing away from the root cube.
        let ro = [2.0, 2.0, 2.0];
        let rd = normalize([1.0, 1.0, 1.0]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        assert_eq!(result.hit_cell, None, "ray away from root must miss");
        assert_eq!(
            result.hit_normal, None,
            "pre-root miss must carry no normal"
        );

        // 2. Clean in-root miss: enter the cube, traverse empty space, exit.
        let ro = [2.0, 0.1, 0.1];
        let rd = normalize([-1.0, 0.0, 0.0]);
        let result = cpu_trace::raycast(&dag.nodes, dag.root_level, ro, rd, false);
        assert_eq!(
            result.hit_cell, None,
            "ray through empty corner of root must miss (voxel is at center)",
        );
        assert_eq!(result.hit_normal, None, "in-root miss must carry no normal");
        assert!(!result.exhausted, "clean miss must not trip budget");
    }

    /// hash-thing-6hd: `octant_of` must use `rd` sign as a tiebreaker on
    /// exact-midpoint positions. Pre-6hd used `>=` which biased midpoint
    /// positions to the HIGH half regardless of ray direction, producing
    /// wrong-sibling selection on simultaneous two-axis exits where the
    /// step-past nudge on a non-dominant axis underflows to sub-ULP.
    ///
    /// These direct unit tests pin the fix at the call point and exercise
    /// the exact f32 values that the Codex 27m-trident reproducer lands
    /// on in the running shader — the integration path is numerically
    /// fragile to reproduce end-to-end, but the bit-level behavior of
    /// `octant_of` is fully controllable here.
    #[test]
    fn octant_of_midpoint_biases_on_rd_sign() {
        let node_min = [0.0_f32, 0.0, 0.0];
        let half = 0.5_f32;
        // mid = (0.5, 0.5, 0.5) exactly — all three axes representable.
        let mid_pos = [0.5_f32, 0.5, 0.5];

        // --- x axis ---
        // rd.x > 0 → HIGH, rd.x < 0 → LOW. rd.y/z zero; tiebreak for them
        // defaults to HIGH (rd >= 0.0).
        assert_eq!(
            cpu_trace::octant_of(mid_pos, [1.0, 0.0, 0.0], node_min, half),
            1 | 2 | 4,
            "rd.x > 0 at xyz midpoint: x HIGH (target), y HIGH (rd.y==0 tiebreak), z HIGH",
        );
        assert_eq!(
            cpu_trace::octant_of(mid_pos, [-1.0, 0.0, 0.0], node_min, half),
            2 | 4,
            "rd.x < 0 at xyz midpoint: x LOW (target), y HIGH, z HIGH",
        );

        // --- y axis ---
        assert_eq!(
            cpu_trace::octant_of(mid_pos, [0.0, 1.0, 0.0], node_min, half),
            1 | 2 | 4,
            "rd.y > 0 at xyz midpoint: all HIGH",
        );
        assert_eq!(
            cpu_trace::octant_of(mid_pos, [0.0, -1.0, 0.0], node_min, half),
            1 | 4,
            "rd.y < 0 at xyz midpoint: y LOW, x+z HIGH",
        );

        // --- z axis ---
        assert_eq!(
            cpu_trace::octant_of(mid_pos, [0.0, 0.0, 1.0], node_min, half),
            1 | 2 | 4,
            "rd.z > 0 at xyz midpoint: all HIGH",
        );
        assert_eq!(
            cpu_trace::octant_of(mid_pos, [0.0, 0.0, -1.0], node_min, half),
            1 | 2,
            "rd.z < 0 at xyz midpoint: z LOW, x+y HIGH",
        );

        // --- the actual Codex reproducer: rd = (1.0, -1e-6, 0.0) ---
        // Pre-6hd, this returned oct 3 (x-HIGH, y-HIGH, z-LOW) because
        // pos.x=0.50000995 >= 0.5 → HIGH, pos.y=0.5 >= 0.5 → HIGH (wrong
        // under `>=` bias), pos.z=0.25 < 0.5 → LOW. Post-6hd, it returns
        // oct 1 (x-HIGH, y-LOW, z-LOW) because the y tiebreak consults
        // rd.y < 0 → LOW.
        let codex_pos = [0.50000995_f32, 0.5, 0.25];
        let codex_rd = [1.0_f32, -1e-6, 0.0];
        assert_eq!(
            cpu_trace::octant_of(codex_pos, codex_rd, node_min, half),
            1,
            "Codex reproducer: x-HIGH + y-LOW + z-LOW = oct 1 (pre-6hd gave 3)",
        );
    }

    /// hash-thing-6hd: off-midpoint positions must still classify purely
    /// on `pos` (no rd consultation). Away from exact equality the new
    /// `>` predicate and the old `>=` predicate agree by construction.
    #[test]
    fn octant_of_off_midpoint_ignores_rd() {
        let node_min = [0.0_f32, 0.0, 0.0];
        let half = 0.5_f32;

        // 1 ULP above midpoint on x — must be HIGH regardless of rd sign.
        let ulp = f32::from_bits(0.5_f32.to_bits() + 1); // next float up
        let above = [ulp, 0.25, 0.25];
        assert_eq!(
            cpu_trace::octant_of(above, [1.0, 1.0, 1.0], node_min, half),
            1,
            "pos.x just above mid → HIGH-x with rd.x > 0",
        );
        assert_eq!(
            cpu_trace::octant_of(above, [-1.0, -1.0, -1.0], node_min, half),
            1,
            "pos.x just above mid → HIGH-x even with rd.x < 0 (rd ignored off-mid)",
        );

        // 1 ULP below midpoint on x — must be LOW regardless of rd sign.
        let below_x = f32::from_bits(0.5_f32.to_bits() - 1); // next float down
        let below = [below_x, 0.25, 0.25];
        assert_eq!(
            cpu_trace::octant_of(below, [1.0, 1.0, 1.0], node_min, half),
            0,
            "pos.x just below mid → LOW-x even with rd.x > 0",
        );
        assert_eq!(
            cpu_trace::octant_of(below, [-1.0, -1.0, -1.0], node_min, half),
            0,
            "pos.x just below mid → LOW-x with rd.x < 0",
        );
    }

    // ---------------------------------------------------------------
    // 5bb.5: Persistent content-addressed cache regression tests
    // ---------------------------------------------------------------

    #[test]
    fn update_is_idempotent_on_identical_input() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 10, 20, 30, mat(1));
        root = store.set_cell(root, 40, 40, 40, mat(2));

        let mut svdag = Svdag::new();
        svdag.update(&store, root, 6);
        let len_after_first = svdag.nodes.len();
        let cached_after_first = svdag.total_slot_count();
        let reachable_after_first = svdag.node_count;
        let root_offset_after_first = svdag.root_offset();

        svdag.update(&store, root, 6);

        assert_eq!(
            svdag.nodes.len(),
            len_after_first,
            "no new slots should be appended on a no-op update"
        );
        assert_eq!(
            svdag.total_slot_count(),
            cached_after_first,
            "persistent cache size must be stable on a no-op update"
        );
        assert_eq!(
            svdag.node_count, reachable_after_first,
            "reachable node count must be stable on a no-op update"
        );
        assert_eq!(
            svdag.root_offset(),
            root_offset_after_first,
            "root offset must match on a no-op update — same content, same slot"
        );
    }

    #[test]
    fn update_single_cell_edit_appends_few_slots() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 0, 0, 0, mat(1));

        let mut svdag = Svdag::new();
        svdag.update(&store, root, 6);
        let len_before_edit = svdag.nodes.len();

        // Flip one cell deep in the tree.
        root = store.set_cell(root, 33, 17, 41, mat(1));
        svdag.update(&store, root, 6);

        let appended_u32 = svdag.nodes.len() - len_before_edit;
        let appended_slots = appended_u32 / 9;

        assert!(
            appended_u32 % 9 == 0,
            "appended u32 count must be a multiple of 9 (whole slots), got {appended_u32}"
        );
        assert!(
            appended_slots <= 8,
            "single-cell edit appended {appended_slots} slots; expected at most 8 \
             (6 ancestors + slack)"
        );
        assert!(
            appended_slots >= 1,
            "single-cell edit appended zero slots; the edit is a no-op against cache"
        );
    }

    #[test]
    fn update_cross_epoch_cache_reuse() {
        // Simulate a step_flat cycle: build in store A, compact into store B.
        // The Svdag should still recognize identical subtrees.
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 5, 5, 5, mat(1));

        let mut svdag = Svdag::new();
        svdag.update(&store, root, 6);
        let len_after_first = svdag.nodes.len();

        // "Compact" into a fresh store — NodeIds change, content doesn't.
        let (store2, root2) = store.compacted(root);
        svdag.update(&store2, root2, 6);

        assert_eq!(
            svdag.nodes.len(),
            len_after_first,
            "cross-epoch update with identical content must not grow the buffer"
        );
    }

    #[test]
    fn compact_reduces_buffer_after_edits() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 0, 0, 0, mat(1));

        let mut svdag = Svdag::new();
        svdag.update(&store, root, 6);

        // Make several edits to accumulate stale slots.
        for i in 1..10 {
            root = store.set_cell(root, i * 5, i * 3, i * 4, mat(1));
            svdag.update(&store, root, 6);
        }

        let pre_compact_len = svdag.nodes.len();
        let pre_compact_total = svdag.total_slot_count();
        assert!(
            svdag.stale_ratio() > 0.0,
            "edits should produce some stale slots"
        );

        svdag.compact(&store, root);

        assert!(
            svdag.nodes.len() <= pre_compact_len,
            "compact must not grow the buffer"
        );
        assert!(
            svdag.total_slot_count() <= pre_compact_total,
            "compact must not grow the cache"
        );
        assert_eq!(
            svdag.stale_ratio(),
            0.0,
            "after compact, every cached slot is reachable"
        );
    }

    #[test]
    fn root_header_word_points_at_root_node() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 10, 10, 10, mat(1));

        let svdag = Svdag::build(&store, root, 6);
        let root_offset = svdag.nodes[0] as usize;

        // The root offset should point at a valid 9-u32 slot.
        assert!(
            root_offset > 0,
            "root offset 0 would collide with the header word itself"
        );
        assert!(
            root_offset + 9 <= svdag.nodes.len(),
            "root offset {root_offset} points past buffer end {}",
            svdag.nodes.len()
        );
        // The mask at the root offset should have at least one bit set.
        let mask = svdag.nodes[root_offset];
        assert!(
            mask > 0,
            "root mask should be nonzero for a non-empty world"
        );
    }
}
