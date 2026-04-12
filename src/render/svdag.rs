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
//!
//! The buffer is append-only within `Svdag`'s lifetime: stale slots from superseded
//! worlds accumulate but are never referenced, costing memory only. Renderer-side
//! compaction (rebuild from scratch when stale-ratio crosses a threshold) is a
//! follow-up — see hash-thing-5bb epic.

use crate::octree::{Node, NodeId, NodeStore};
use rustc_hash::FxHashMap;

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
    /// This is the load-bearing piece of incremental uploads: across `update()`
    /// calls, any subtree whose slot was interned before reuses its old offset,
    /// and children-of-parent references stay stable.
    offset_by_slot: FxHashMap<[u32; 9], u32>,
    /// Level of the current root (2^level cells per side).
    pub root_level: u32,
    /// Number of interior nodes reachable from the current root.
    /// NOT monotonic — drops when the simulation compacts. Use
    /// `total_slot_count` for cumulative-growth tracking.
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
    pub fn update(&mut self, store: &NodeStore, root: NodeId, root_level: u32) {
        self.root_level = root_level;
        // NodeId-level dedup for this walk only. Discarded at end of update().
        // Two walks can see different NodeIds for the same subtree (store
        // compaction), so this cache cannot persist — the persistent cache
        // lives at the content-slot level, not the NodeId level.
        let mut id_to_offset: FxHashMap<NodeId, u32> = FxHashMap::default();
        let root_offset = self.visit(store, root, &mut id_to_offset);
        self.nodes[0] = root_offset;
        self.node_count = id_to_offset.len();
    }

    /// Recursively visit a node. Returns its absolute offset in `nodes`.
    ///
    /// Bottom-up: child offsets are resolved before the parent's slot is
    /// assembled, so the parent's slot is itself a fully content-determined
    /// identity once the children have been placed. Intern the slot into
    /// `offset_by_slot` — hit → reuse (no append), miss → append.
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
                // Leaves don't get their own slot — they're inlined into the
                // parent's child entries. Reaching here means update() was
                // called with a leaf-level root, which the renderer doesn't
                // support (root_level >= 1 in practice).
                panic!("Svdag::visit called on a leaf — leaves are inlined in parent slots");
            }
            Node::Interior { children, .. } => {
                // Copy out of the store to avoid borrow conflict with the
                // recursive `self.visit` below.
                let children = *children;
                let mut slot = [0u32; 9];
                let mut mask: u32 = 0;
                for (i, &child_id) in children.iter().enumerate() {
                    match store.get(child_id) {
                        Node::Leaf(state) => {
                            if *state != 0 {
                                mask |= 1 << i;
                            }
                            slot[1 + i] = (1u32 << 31) | (*state as u32);
                        }
                        Node::Interior { population, .. } => {
                            if *population > 0 {
                                mask |= 1 << i;
                                let child_offset = self.visit(store, child_id, id_to_offset);
                                slot[1 + i] = child_offset;
                            } else {
                                // Empty interior — same encoding as an empty leaf.
                                slot[1 + i] = 1u32 << 31;
                            }
                        }
                    }
                }
                slot[0] = mask;

                // Persistent content-cache lookup. If this exact slot was ever
                // interned before, reuse the old offset — no buffer growth,
                // no GPU upload. This is the core of the 5bb.5 win.
                let offset = if let Some(&existing) = self.offset_by_slot.get(&slot) {
                    existing
                } else {
                    let new_offset = self.nodes.len() as u32;
                    self.nodes.extend_from_slice(&slot);
                    self.offset_by_slot.insert(slot, new_offset);
                    new_offset
                };
                id_to_offset.insert(id, offset);
                offset
            }
        }
    }

    pub fn byte_size(&self) -> usize {
        self.nodes.len() * 4
    }

    /// Absolute offset of the current root node in `nodes`. Mirrors `nodes[0]`.
    pub fn root_offset(&self) -> u32 {
        self.nodes[0]
    }

    /// Cumulative count of distinct slots interned over this builder's lifetime.
    /// Monotonic; grows only. Useful for deciding when to trigger a renderer-
    /// side compaction pass (follow-up work).
    pub fn total_slot_count(&self) -> usize {
        self.offset_by_slot.len()
    }
}

impl Default for Svdag {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// CPU-side replica of svdag_raycast.wgsl for debugging.
//
// This deliberately mirrors the WGSL shader line-for-line so that any bug in
// the traversal reproduces identically on the CPU, where we can dump state.
// ----------------------------------------------------------------------------

#[cfg(test)]
pub mod cpu_trace {
    use super::*;

    const MAX_DEPTH: usize = 14;
    const MAX_STEPS: usize = 512;
    const LEAF_BIT: u32 = 0x8000_0000;
    const EPS: f32 = 1e-5;

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
        pub hit_material: Option<u32>,
        pub steps: usize,
        pub events: Vec<TraceEvent>,
    }

    fn octant_of(pos: [f32; 3], node_min: [f32; 3], half: f32) -> u32 {
        let mut idx = 0u32;
        if pos[0] >= node_min[0] + half {
            idx |= 1;
        }
        if pos[1] >= node_min[1] + half {
            idx |= 2;
        }
        if pos[2] >= node_min[2] + half {
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

    pub fn raycast(dag: &[u32], ro: [f32; 3], rd: [f32; 3], record: bool) -> TraceResult {
        let inv_rd = [1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2]];
        let mut events = Vec::new();

        let (root_near, root_far) = intersect_aabb(ro, inv_rd, [0.0; 3], [1.0; 3]);
        if root_near > root_far || root_far < 0.0 {
            return TraceResult {
                hit_material: None,
                steps: 0,
                events,
            };
        }

        let mut stack_node = [0u32; MAX_DEPTH];
        let mut stack_min = [[0.0f32; 3]; MAX_DEPTH];
        let mut stack_half = [0.0f32; MAX_DEPTH];
        let mut depth: usize = 0;

        // Root offset lives in `dag[0]` — the rest of `dag` is an append-only
        // stream of 9-u32 interior slots, and the root's slot can be anywhere
        // past index 1 after incremental updates. See `Svdag` module docs.
        stack_node[0] = dag[0];
        stack_min[0] = [0.0; 3];
        stack_half[0] = 0.5;

        let mut t = root_near.max(0.0) + EPS;

        for step in 0..MAX_STEPS {
            if t > root_far {
                return TraceResult {
                    hit_material: None,
                    steps: step,
                    events,
                };
            }
            let pos = [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];

            // Pop until top contains pos.
            // NOTE: pop check is STRICT (no EPS slack). The step-past advances `t`
            // by +EPS which moves pos past the boundary by |rd_axis| * EPS > 0.
            // If we allowed EPS slack here, the pop check would still consider pos
            // "inside" after a step-past across an octant boundary (since the
            // positional advance |rd| * EPS is smaller than the pop slack EPS),
            // causing an infinite loop on boundary crossings. Strict bounds work
            // because t = t_near + EPS on entry also places pos clearly inside.
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
                        hit_material: None,
                        steps: step,
                        events,
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
                let oct = octant_of(pos, node_min, half);
                let child_slot = dag[node_offset + 1 + oct as usize];

                if child_slot & LEAF_BIT != 0 {
                    let mat = child_slot & 0xFFFF;
                    if mat > 0 {
                        if record {
                            events.push(TraceEvent::Hit { depth, mat, t });
                        }
                        return TraceResult {
                            hit_material: Some(mat),
                            steps: step,
                            events,
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

            // Step past
            let node_min = stack_min[depth];
            let half = stack_half[depth];
            let oct = octant_of(pos, node_min, half);
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
            let (_, oct_far) = intersect_aabb(ro, inv_rd, child_min, child_max);
            let t_old = t;
            t = oct_far + EPS;
            if record {
                events.push(TraceEvent::StepPast { t_old, t_new: t });
            }
        }

        TraceResult {
            hit_material: None,
            steps: MAX_STEPS,
            events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::NodeStore;

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    #[test]
    fn single_voxel_center_hit() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        // One voxel at (32,32,32) -> world (0.5, 0.5, 0.5)..(0.515625, ...)
        root = store.set_cell(root, 32, 32, 32, 1);
        let dag = Svdag::build(&store, root, 6);

        // Ray aimed at the voxel's center (0.5078125, 0.5078125, 0.5078125) from outside
        let target = [0.5 + 0.5 / 64.0, 0.5 + 0.5 / 64.0, 0.5 + 0.5 / 64.0];
        let ro = [2.0, 2.0, 2.0];
        let rd = normalize([target[0] - ro[0], target[1] - ro[1], target[2] - ro[2]]);
        let result = cpu_trace::raycast(&dag.nodes, ro, rd, true);
        assert_eq!(
            result.hit_material,
            Some(1),
            "expected hit, events: {:#?}",
            result.events
        );
    }

    #[test]
    fn eight_corners_each_visible() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        for &x in &[16u64, 48] {
            for &y in &[16u64, 48] {
                for &z in &[16u64, 48] {
                    for dx in 0..4u64 {
                        for dy in 0..4u64 {
                            for dz in 0..4u64 {
                                root = store.set_cell(root, x + dx, y + dy, z + dz, 1);
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
                    let result = cpu_trace::raycast(&dag.nodes, ro, rd, false);
                    if result.hit_material.is_none() {
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
        let mut voxels = Vec::new();
        for &x in &[16u64, 48] {
            for &y in &[16u64, 48] {
                for &z in &[16u64, 48] {
                    for dx in 0..4u64 {
                        for dy in 0..4u64 {
                            for dz in 0..4u64 {
                                root = store.set_cell(root, x + dx, y + dy, z + dz, 1);
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
            let result = cpu_trace::raycast(&dag.nodes, ro, rd, false);
            if result.hit_material.is_none() {
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
        for _ in 0..300 {
            let x = (rand_u32() % 64) as u64;
            let y = (rand_u32() % 64) as u64;
            let z = (rand_u32() % 64) as u64;
            root = store.set_cell(root, x, y, z, 1);
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
            let result = cpu_trace::raycast(&dag.nodes, ro, rd, false);
            if result.hit_material.is_none() {
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

    // ------------------------------------------------------------------
    // Incremental-upload tests (hash-thing-5bb.5).
    //
    // These assert the core contract of `Svdag::update`: repeated calls
    // with identical (or near-identical) content must reuse the content-
    // addressed cache and append zero (or near-zero) new slots. Without
    // this, the whole renderer-side tail-upload optimization is a lie.
    // ------------------------------------------------------------------

    /// Reference: `build()` of a simple world produces some set of nodes.
    /// Repeating `update()` on the same `Svdag` with the same root MUST
    /// leave the buffer unchanged — no new slots appended.
    #[test]
    fn update_is_idempotent_on_identical_input() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 10, 20, 30, 1);
        root = store.set_cell(root, 40, 40, 40, 2);

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

    /// A fresh store with the same content produces different NodeIds but
    /// the same 9-u32 slots. The persistent content cache must reuse the
    /// prior offsets so NO new slots get appended.
    ///
    /// This is the cross-epoch case: `NodeStore::compacted()` invalidates
    /// every NodeId but leaves the underlying voxel content identical, so
    /// any NodeId-keyed cache would miss while a slot-keyed cache must hit.
    #[test]
    fn update_reuses_cache_across_fresh_stores() {
        let build_store = || {
            let mut store = NodeStore::new();
            let mut root = store.empty(6);
            for &(x, y, z) in &[(10, 20, 30), (40, 40, 40), (5, 5, 5)] {
                root = store.set_cell(root, x, y, z, 1);
            }
            (store, root)
        };

        let mut svdag = Svdag::new();
        let (store_a, root_a) = build_store();
        svdag.update(&store_a, root_a, 6);
        let len_after_first = svdag.nodes.len();
        let cached_after_first = svdag.total_slot_count();

        // Build a completely fresh store — NodeIds will differ, but every
        // content slot will match something the cache already knows.
        let (store_b, root_b) = build_store();
        // Sanity: the two stores are separate allocations, so a NodeId-
        // keyed cache would miss on this root. We only care about content.
        svdag.update(&store_b, root_b, 6);

        assert_eq!(
            svdag.nodes.len(),
            len_after_first,
            "identical content in a fresh store must hit the content cache — \
             no new slots appended"
        );
        assert_eq!(
            svdag.total_slot_count(),
            cached_after_first,
            "persistent slot cache must not grow across fresh stores of same content"
        );
    }

    /// A single-cell flip on top of an already-cached world should append
    /// only the O(log N) slots along the path from root to the flipped
    /// leaf. For a level-6 (64^3) world that path has 6 ancestors, so
    /// the bound is small and deterministic. We assert a generous cap so
    /// the test is informative but not flaky if the exact count shifts.
    #[test]
    fn update_single_cell_edit_appends_few_slots() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        // Seed an unrelated corner so the tree has a little structure.
        root = store.set_cell(root, 0, 0, 0, 1);

        let mut svdag = Svdag::new();
        svdag.update(&store, root, 6);
        let len_before_edit = svdag.nodes.len();

        // Flip one cell deep in the tree.
        root = store.set_cell(root, 33, 17, 41, 1);
        svdag.update(&store, root, 6);

        let appended_u32 = svdag.nodes.len() - len_before_edit;
        let appended_slots = appended_u32 / 9;

        // Path from root to leaf is `level` ancestors (here 6). One ancestor
        // per level, each touched exactly once, each emitting a fresh slot.
        // Allow a generous ceiling so minor changes to proof-collapse order
        // don't break this test — the point is "O(log N), not O(N)".
        assert!(
            appended_u32 % 9 == 0,
            "appended u32 count must be a multiple of 9 (whole slots), got {appended_u32}"
        );
        assert!(
            appended_slots <= 8,
            "single-cell edit appended {appended_slots} slots; expected at most 8 \
             (6 ancestors + slack). If this fires, the persistent content cache \
             is failing to dedupe unchanged siblings."
        );
        assert!(
            appended_slots >= 1,
            "single-cell edit appended zero slots; the edit is a no-op against cache"
        );
    }

    /// After an update with a changed root, `nodes[0]` must point at the
    /// new root's slot — not the old one. Without this, the shader would
    /// start traversal at a stale root. Verified end-to-end via cpu_trace.
    #[test]
    fn update_rewrites_root_header_and_shader_still_hits() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6);
        root = store.set_cell(root, 10, 10, 10, 1);

        let mut svdag = Svdag::new();
        svdag.update(&store, root, 6);
        let root_offset_a = svdag.root_offset();
        // Shader-equivalent traversal finds the first voxel.
        let target_a = [(10.5) / 64.0, (10.5) / 64.0, (10.5) / 64.0];
        let ro = [2.0, 2.0, 2.0];
        let rd_a = normalize([
            target_a[0] - ro[0],
            target_a[1] - ro[1],
            target_a[2] - ro[2],
        ]);
        let result_a = cpu_trace::raycast(&svdag.nodes, ro, rd_a, false);
        assert_eq!(
            result_a.hit_material,
            Some(1),
            "first update should resolve the first voxel"
        );

        // Replace with a world whose voxel lives in a DIFFERENT octant of
        // the level-6 root. Old was at (10,10,10) → octant 0 (−,−,−);
        // new is at (50,10,10) → octant 1 (+,−,−). Differing masks
        // guarantee a differing root slot → differing root offset.
        let mut store2 = NodeStore::new();
        let mut root2 = store2.empty(6);
        root2 = store2.set_cell(root2, 50, 10, 10, 1);
        svdag.update(&store2, root2, 6);
        let root_offset_b = svdag.root_offset();
        assert_ne!(
            root_offset_a, root_offset_b,
            "a world in a different root-octant must produce a different root offset"
        );

        // Ray aimed at the new voxel should hit, using traversal that
        // starts from `dag[0]` — i.e. through the new root, not the old.
        // From origin (-1, 10.5/64, 10.5/64) the ray is +x only, and only
        // passes through y/z band around 10, which lives in the NEW world
        // but not on any path to the old voxel (old was at x=10, this
        // ray's origin is at x=-1, aiming in +x at x=50.5/64).
        let target_b = [(50.5) / 64.0, (10.5) / 64.0, (10.5) / 64.0];
        let ro_b = [-1.0, (10.5) / 64.0, (10.5) / 64.0];
        let rd_b = normalize([
            target_b[0] - ro_b[0],
            target_b[1] - ro_b[1],
            target_b[2] - ro_b[2],
        ]);
        let result_b = cpu_trace::raycast(&svdag.nodes, ro_b, rd_b, false);
        assert_eq!(
            result_b.hit_material,
            Some(1),
            "after root swap, rays at the new voxel should hit"
        );
    }
}
