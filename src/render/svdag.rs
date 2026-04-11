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

/// High bit of a child slot marks it as an inline leaf. Interior-node offsets
/// must fit strictly below this bound, checked in `write_node` at serialize
/// time — a violation is a silent GPU-corruption bug (the decoder would read
/// a legitimate offset as a leaf and emit a garbage material). This mirrors
/// `LEAF_BIT = 0x80000000u` in `src/render/svdag_raycast.wgsl`; if you change
/// one, change the other.
pub(crate) const LEAF_BIT: u32 = 0x8000_0000;

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
        debug_assert!(
            root_level >= 1,
            "Svdag::build expects an interior root (level >= 1); got level {root_level}",
        );
        let mut nodes: Vec<u32> = Vec::new();
        let mut id_to_offset: FxHashMap<NodeId, u32> = FxHashMap::default();

        // Reserve slot 0 for the root; we'll fill it in after recursion.
        let root_offset = Self::write_node(&mut nodes, &mut id_to_offset, store, root);
        debug_assert_eq!(root_offset, 0, "root must be at offset 0");

        // Each emitted interior node is exactly NODE_STRIDE u32s (mask + 8 children).
        // Derive node_count from the buffer length so it stays in lock-step with
        // `byte_size() / 36` by construction; the debug_assert below catches any
        // future change that breaks the stride invariant. (Today this happens to
        // equal `id_to_offset.len()` because only interior nodes are interned —
        // leaves are encoded inline in the parent's child slot — but that
        // equivalence is a coincidence of the current writer, not an invariant
        // we want to depend on.)
        const NODE_STRIDE: usize = 9;
        debug_assert_eq!(
            nodes.len() % NODE_STRIDE,
            0,
            "SVDAG buffer length should be a multiple of {NODE_STRIDE}",
        );
        let node_count = nodes.len() / NODE_STRIDE;
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
                // Guard against silent 31-bit overflow: interior offsets MUST fit
                // strictly below LEAF_BIT or the GPU decoder misreads them as leaf
                // words (hash-thing-x9r). Release-panic beats release-corruption.
                // Empirically unreachable today — blowing this assert needs ~8 GB of
                // node storage — but stays cheap as a sanity rail.
                assert!(
                    my_offset < LEAF_BIT,
                    "SVDAG interior node count exceeded 31 bits ({}); \
                     widen the encoding (hash-thing-x9r follow-up)",
                    nodes.len()
                );
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
                            nodes[header_start + 1 + i] = LEAF_BIT | (*state as u32);
                        }
                        Node::Interior { population, .. } => {
                            if *population > 0 {
                                mask |= 1 << i;
                                // Recurse
                                let child_offset =
                                    Self::write_node(nodes, id_to_offset, store, child_id);
                                nodes[header_start + 1 + i] = child_offset; // high bit clear = interior
                            } else {
                                // Empty interior subtree — encode as empty leaf
                                nodes[header_start + 1 + i] = LEAF_BIT;
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

// CPU debug-trace helpers. Used ad-hoc when investigating shader bugs;
// no automated tests currently call this module, but keeping it compiling
// is the whole point.
#[cfg(test)]
#[allow(unused_imports, dead_code)]
pub mod cpu_trace {
    use super::*;

    const MAX_DEPTH: usize = 14;
    const MAX_STEPS: usize = 512;
    // LEAF_BIT comes from `use super::*` above — kept in one place so the
    // shader mirror stays a single source of truth (hash-thing-x9r).
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

        stack_node[0] = 0;
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

    /// Pin the Rust-side `LEAF_BIT` to the exact bit pattern the shader reads.
    /// If someone widens the encoding they have to touch this test *and* the
    /// WGSL — the test is the tripwire (hash-thing-x9r).
    #[test]
    fn leaf_bit_matches_shader() {
        assert_eq!(LEAF_BIT, 0x8000_0000);
    }
}
