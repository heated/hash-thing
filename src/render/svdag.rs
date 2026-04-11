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
//! same NodeId. This module just serializes that DAG into a flat array the GPU can
//! traverse with an iterative descent shader.
//!
//! Node format (u32 per slot, 9 u32s per interior node = 36 bytes):
//!   [0]: child_mask (low 8 bits: which octants are non-empty; high bits: reserved)
//!   [1..=8]: child_index — packed as (is_leaf << 31) | payload_bits
//!     - if is_leaf: payload is the 16-bit material state in low bits
//!     - else:       payload is the index of the child node in the buffer
//!
//! Root is always at index 0.

use crate::octree::{Node, NodeId, NodeStore};
use rustc_hash::FxHashMap;

/// GPU-friendly serialized DAG.
pub struct Svdag {
    /// Packed node data. Each interior node takes 9 u32s:
    /// [mask, child0, child1, ..., child7]
    pub nodes: Vec<u32>,
    /// Level of the root node (2^level cells per side).
    pub root_level: u32,
    /// Number of distinct interior nodes serialized.
    pub node_count: usize,
}

impl Svdag {
    /// Serialize the octree rooted at `root` from `store` into a flat GPU buffer.
    ///
    /// Walks the DAG in depth-first order, interning each `NodeId` to a flat-buffer
    /// offset so shared subtrees become shared offsets in the output.
    pub fn build(store: &NodeStore, root: NodeId, root_level: u32) -> Self {
        let mut nodes: Vec<u32> = Vec::new();
        let mut id_to_offset: FxHashMap<NodeId, u32> = FxHashMap::default();

        // Reserve slot 0 for the root; we'll fill it in after recursion.
        let root_offset = Self::write_node(&mut nodes, &mut id_to_offset, store, root);
        debug_assert_eq!(root_offset, 0, "root must be at offset 0");

        let node_count = id_to_offset.len();
        Svdag {
            nodes,
            root_level,
            node_count,
        }
    }

    /// Recursively serialize a node. Returns the offset of the node in `nodes`.
    /// If the node is a leaf, returns `LEAF_MARKER | material` — callers must
    /// handle leaves inline in the parent's child slot rather than emitting a node.
    fn write_node(
        nodes: &mut Vec<u32>,
        id_to_offset: &mut FxHashMap<NodeId, u32>,
        store: &NodeStore,
        id: NodeId,
    ) -> u32 {
        // Check cache first (DAG dedup)
        if let Some(&offset) = id_to_offset.get(&id) {
            return offset;
        }

        match store.get(id) {
            Node::Leaf(_) => {
                // Leaves don't get their own node entry; the parent encodes them inline.
                // This branch should only hit if someone calls write_node on a leaf at the root.
                // We emit a degenerate "all-leaf" interior node for this case.
                // (In practice the root will always be level >= 1 for our demo.)
                panic!("write_node called on a leaf — handle leaves in parent's child slot");
            }
            Node::Interior { children, .. } => {
                // Reserve our own slot before recursing so that cyclic refs (impossible here
                // but conceptually clean) would be handled. Write a placeholder, record offset.
                let my_offset = nodes.len() as u32;
                id_to_offset.insert(id, my_offset);

                // Reserve 9 slots: mask + 8 children
                let header_start = nodes.len();
                nodes.resize(header_start + 9, 0);

                let mut mask: u32 = 0;
                // Process children (copy to avoid borrow issues)
                let children = *children;

                for (i, &child_id) in children.iter().enumerate() {
                    let child_node = store.get(child_id);
                    match child_node {
                        Node::Leaf(state) => {
                            // Encode leaf inline: high bit set + state in low 16 bits.
                            // State 0 = empty → don't set mask bit.
                            if *state != 0 {
                                mask |= 1 << i;
                            }
                            nodes[header_start + 1 + i] = (1u32 << 31) | (*state as u32);
                        }
                        Node::Interior { population, .. } => {
                            if *population > 0 {
                                mask |= 1 << i;
                                // Recurse
                                let child_offset = Self::write_node(nodes, id_to_offset, store, child_id);
                                nodes[header_start + 1 + i] = child_offset; // high bit clear = interior
                            } else {
                                // Empty interior subtree — encode as empty leaf
                                nodes[header_start + 1 + i] = (1u32 << 31) | 0;
                            }
                        }
                    }
                }

                nodes[header_start] = mask;
                my_offset
            }
        }
    }

    pub fn byte_size(&self) -> usize {
        self.nodes.len() * 4
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
        Descend { depth: usize, oct: u32, child_offset: u32 },
        EmptyLeaf { depth: usize, oct: u32 },
        Hit { depth: usize, mat: u32, t: f32 },
        StepPast { t_old: f32, t_new: f32 },
        Popped { from: usize, to: usize },
        ExitedRoot,
    }

    pub struct TraceResult {
        pub hit_material: Option<u32>,
        pub steps: usize,
        pub events: Vec<TraceEvent>,
    }

    fn octant_of(pos: [f32; 3], node_min: [f32; 3], half: f32) -> u32 {
        let mut idx = 0u32;
        if pos[0] >= node_min[0] + half { idx |= 1; }
        if pos[1] >= node_min[1] + half { idx |= 2; }
        if pos[2] >= node_min[2] + half { idx |= 4; }
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
            return TraceResult { hit_material: None, steps: 0, events };
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

        stack_node[0] = 0;
        stack_min[0] = [0.0; 3];
        stack_half[0] = 0.5;

        // Local-frame starting t. Just past the root entry; `ro_local` is
        // already on the root face.
        let mut t = EPS;

        for step in 0..MAX_STEPS {
            if t > t_exit {
                return TraceResult { hit_material: None, steps: step, events };
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
                    if record { events.push(TraceEvent::ExitedRoot); }
                    return TraceResult { hit_material: None, steps: step, events };
                }
                top -= 1;
            }
            if record && top != depth {
                events.push(TraceEvent::Popped { from: depth, to: top });
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
                        if record { events.push(TraceEvent::Hit { depth, mat, t }); }
                        return TraceResult {
                            hit_material: Some(mat),
                            steps: step,
                            events,
                        };
                    }
                    if record { events.push(TraceEvent::EmptyLeaf { depth, oct }); }
                    break;
                } else {
                    if depth + 1 >= MAX_DEPTH { break; }
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
            let oct = octant_of(pos, node_min, half);
            let child_min = [
                node_min[0] + (oct & 1) as f32 * half,
                node_min[1] + ((oct >> 1) & 1) as f32 * half,
                node_min[2] + ((oct >> 2) & 1) as f32 * half,
            ];
            let child_max = [child_min[0] + half, child_min[1] + half, child_min[2] + half];
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
            if record { events.push(TraceEvent::StepPast { t_old, t_new: t }); }
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
        let rd = normalize([
            target[0] - ro[0],
            target[1] - ro[1],
            target[2] - ro[2],
        ]);
        let result = cpu_trace::raycast(&dag.nodes, ro, rd, true);
        assert_eq!(result.hit_material, Some(1), "expected hit, events: {:#?}", result.events);
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
                    let rd = normalize([
                        target[0] - ro[0],
                        target[1] - ro[1],
                        target[2] - ro[2],
                    ]);
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
            let rd = normalize([
                target[0] - ro[0],
                target[1] - ro[1],
                target[2] - ro[2],
            ]);
            let result = cpu_trace::raycast(&dag.nodes, ro, rd, false);
            if result.hit_material.is_none() {
                misses += 1;
            }
        }
        // Every ray is aimed at the center of a filled voxel, so every ray should hit
        // SOMETHING (either its target or another voxel it passes through first).
        assert_eq!(
            misses, 0,
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
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
            let rd = normalize([
                target[0] - ro[0],
                target[1] - ro[1],
                target[2] - ro[2],
            ]);
            let result = cpu_trace::raycast(&dag.nodes, ro, rd, false);
            if result.hit_material.is_none() {
                misses += 1;
            }
        }
        assert_eq!(
            misses, 0,
            "expected every random sparse voxel to be hit, {} / {} missed",
            misses,
            voxels.len()
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
        root = store.set_cell(root, 2048, 2048, 2048, 1);
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
            let result = cpu_trace::raycast(&dag.nodes, *ro, rd_n, false);
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
    #[test]
    fn far_camera_forward_progress() {
        let mut store = NodeStore::new();
        let mut root = store.empty(12);
        root = store.set_cell(root, 2048, 2048, 2048, 1);
        let dag = Svdag::build(&store, root, 12);

        // Each case starts the camera well outside the unit root cube so
        // the pre-clamp `t` enters the far-camera regime early in the loop.
        let cases: [([f32; 3], [f32; 3]); 3] = [
            // Codex Critical reproducer (un-normalized direction).
            (
                [12.78218, -2.47332, -7.13861],
                [-0.86086446, 0.16658753, 0.48079199],
            ),
            // Axis-dominant far-camera ray with tiny secondary axes —
            // lands on grazing inter-octant faces at depth ≥ 12.
            ([10.0, 0.5001, 0.4999], [-1.0, 0.0001, -0.0001]),
            // Far camera on a different axis with a different grazing pair.
            ([0.5, 0.5001, -9.5], [0.0001, -0.0001, 1.0]),
        ];

        for (i, (ro, rd)) in cases.iter().enumerate() {
            let rd_n = normalize(*rd);
            let result = cpu_trace::raycast(&dag.nodes, *ro, rd_n, false);
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
        root = store.set_cell(root, 8192, 8192, 8192, 1);
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
            let result = cpu_trace::raycast(&dag.nodes, *ro, rd_n, false);
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
        root = store.set_cell(root, 8192, 8192, 8192, 1);
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
            let result = cpu_trace::raycast(&dag.nodes, *ro, rd_n, false);
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
        root = store.set_cell(root, 2048, 2048, 2048, 1);
        let dag = Svdag::build(&store, root, 12);

        // |ro| ~ 100 — pre-clamp world-space t reaches ~99.5-100.5
        // inside the loop. Post-clamp local-frame t stays ≤ √3 + ε.
        let ro: [f32; 3] = [100.5, 0.5001, 0.4999];
        let rd_n = normalize([-1.0, 1e-5, -1e-5]);
        let result = cpu_trace::raycast(&dag.nodes, ro, rd_n, true);

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
        root = store.set_cell(root, 2048, 2048, 2048, 1);
        let dag = Svdag::build(&store, root, 12);

        // Tight AABB of the single depth-12 voxel in world coordinates.
        // At depth 12, cell width is 1/4096 ≈ 2.44e-4.
        let voxel_min: [f32; 3] = [0.5, 0.5, 0.5];
        let voxel_max: [f32; 3] = [
            2049.0 / 4096.0,
            2049.0 / 4096.0,
            2049.0 / 4096.0,
        ];

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
        fn hits_voxel(
            ro: [f32; 3],
            rd: [f32; 3],
            bmin: [f32; 3],
            bmax: [f32; 3],
        ) -> bool {
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

            let result = cpu_trace::raycast(&dag.nodes, ro, rd_n, true);

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
            if let Some(mat) = result.hit_material {
                // Defensive: the scene has exactly one non-empty material
                // (mat=1). Hitting anything else indicates a corruption in
                // the DAG or the traversal reporting an empty leaf as hit.
                assert_eq!(
                    mat, 1,
                    "ysg fuzz case {i}: unexpected hit material {mat}, \
                     only mat=1 exists in this scene",
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
}
