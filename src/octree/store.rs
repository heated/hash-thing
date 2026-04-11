use super::node::{octant_index, CellState, Node, NodeId};
use rustc_hash::FxHashMap;

/// Canonical node store — the core of hash-consing.
///
/// Every unique node exists exactly once. Identical subtrees are shared.
/// This gives us Hashlife's spatial compression: a 4096³ world that's
/// mostly empty costs almost nothing.
pub struct NodeStore {
    /// All canonical nodes, indexed by NodeId.
    nodes: Vec<Node>,
    /// Reverse lookup: node -> its canonical id.
    intern: FxHashMap<Node, NodeId>,
    /// Memoized step results: `NodeId -> result_NodeId`.
    ///
    /// The result of a level-n node is the center (level n-1) cube after
    /// `2^(n-2)` steps. **Currently dead code** — the flat stepper in
    /// `World::step_flat` does not use it. Scaffolding for Hashlife
    /// recursive stepping.
    ///
    /// **WARNING — rule-blindness is not yet handled.** The key is only the
    /// input `NodeId`, not `(NodeId, rule_id)`. When the Hashlife stepper
    /// lands, every rule swap in `main.rs` MUST call `clear_step_cache`, or
    /// the cache must be rekeyed by `(NodeId, rule_id)`. Today's rule-swap
    /// path in `main.rs` (keys `1`–`4`) does NOT clear this cache, which is
    /// safe today only because nothing ever populates it. See the
    /// determinism audit at `.ship-notes/audit-h34.2.md`.
    step_cache: FxHashMap<NodeId, NodeId>,
}

impl NodeStore {
    pub fn new() -> Self {
        let mut store = Self {
            nodes: Vec::new(),
            intern: FxHashMap::default(),
            step_cache: FxHashMap::default(),
        };
        // Reserve NodeId(0) as the canonical empty leaf.
        let empty = store.intern_node(Node::Leaf(0));
        debug_assert_eq!(empty, NodeId::EMPTY);
        store
    }

    /// Intern a node: if it already exists, return the existing id.
    /// Otherwise, store it and return a new id.
    pub fn intern_node(&mut self, node: Node) -> NodeId {
        if let Some(&id) = self.intern.get(&node) {
            return id;
        }
        let id = NodeId(self.nodes.len() as u32);
        self.intern.insert(node.clone(), id);
        self.nodes.push(node);
        id
    }

    /// Get a node by id.
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Create a leaf node.
    pub fn leaf(&mut self, state: CellState) -> NodeId {
        self.intern_node(Node::Leaf(state))
    }

    /// Create an interior node from 8 children.
    pub fn interior(&mut self, level: u32, children: [NodeId; 8]) -> NodeId {
        let population = children.iter().map(|&c| self.population(c)).sum();
        self.intern_node(Node::Interior {
            level,
            children,
            population,
        })
    }

    /// Create a uniform node of given level (all cells have the same state).
    pub fn uniform(&mut self, level: u32, state: CellState) -> NodeId {
        if level == 0 {
            return self.leaf(state);
        }
        let child = self.uniform(level - 1, state);
        self.interior(level, [child; 8])
    }

    /// Create an empty node of given level.
    pub fn empty(&mut self, level: u32) -> NodeId {
        self.uniform(level, 0)
    }

    /// Get the population of a node.
    pub fn population(&self, id: NodeId) -> u64 {
        match self.get(id) {
            Node::Leaf(0) => 0,
            Node::Leaf(_) => 1,
            Node::Interior { population, .. } => *population,
        }
    }

    /// Get children of an interior node. Panics if called on a leaf.
    pub fn children(&self, id: NodeId) -> [NodeId; 8] {
        match self.get(id) {
            Node::Interior { children, .. } => *children,
            Node::Leaf(_) => panic!("children() called on leaf"),
        }
    }

    /// Get a specific child by octant index.
    pub fn child(&self, id: NodeId, octant: usize) -> NodeId {
        self.children(id)[octant]
    }

    /// Set a single cell in the octree. Returns a new root.
    /// Coordinates are relative to the node's origin (0,0,0 corner).
    pub fn set_cell(&mut self, root: NodeId, x: u64, y: u64, z: u64, state: CellState) -> NodeId {
        let node = self.get(root).clone();
        match node {
            Node::Leaf(_) => self.leaf(state),
            Node::Interior {
                level, children, ..
            } => {
                let half = 1u64 << (level - 1);
                let ox = if x >= half { 1u32 } else { 0 };
                let oy = if y >= half { 1u32 } else { 0 };
                let oz = if z >= half { 1u32 } else { 0 };
                let idx = octant_index(ox, oy, oz);
                let lx = if ox == 1 { x - half } else { x };
                let ly = if oy == 1 { y - half } else { y };
                let lz = if oz == 1 { z - half } else { z };
                let new_child = self.set_cell(children[idx], lx, ly, lz, state);
                let mut new_children = children;
                new_children[idx] = new_child;
                self.interior(level, new_children)
            }
        }
    }

    /// Get a single cell value.
    pub fn get_cell(&self, root: NodeId, x: u64, y: u64, z: u64) -> CellState {
        match self.get(root) {
            Node::Leaf(s) => *s,
            Node::Interior {
                level, children, ..
            } => {
                let half = 1u64 << (level - 1);
                let ox = if x >= half { 1u32 } else { 0 };
                let oy = if y >= half { 1u32 } else { 0 };
                let oz = if z >= half { 1u32 } else { 0 };
                let idx = octant_index(ox, oy, oz);
                let lx = if ox == 1 { x - half } else { x };
                let ly = if oy == 1 { y - half } else { y };
                let lz = if oz == 1 { z - half } else { z };
                self.get_cell(children[idx], lx, ly, lz)
            }
        }
    }

    /// Flatten a region of the octree into a flat 3D array.
    /// Returns a Vec of size side^3, indexed as [x + y*side + z*side*side].
    pub fn flatten(&self, root: NodeId, side: usize) -> Vec<CellState> {
        let mut grid = vec![0u8; side * side * side];
        self.flatten_into(root, &mut grid, side, 0, 0, 0);
        grid
    }

    fn flatten_into(
        &self,
        id: NodeId,
        grid: &mut [CellState],
        side: usize,
        ox: usize,
        oy: usize,
        oz: usize,
    ) {
        match self.get(id) {
            Node::Leaf(s) => {
                grid[ox + oy * side + oz * side * side] = *s;
            }
            Node::Interior {
                level,
                children,
                population,
                ..
            } => {
                if *population == 0 {
                    return; // skip entirely empty subtrees
                }
                let half = 1usize << (level - 1);
                let children = *children;
                for oct in 0..8 {
                    let (cx, cy, cz) = super::node::octant_coords(oct);
                    self.flatten_into(
                        children[oct],
                        grid,
                        side,
                        ox + cx as usize * half,
                        oy + cy as usize * half,
                        oz + cz as usize * half,
                    );
                }
            }
        }
    }

    /// Build an octree from a flat 3D array.
    /// Grid is indexed as [x + y*side + z*side*side], side must be a power of 2.
    pub fn from_flat(&mut self, grid: &[CellState], side: usize) -> NodeId {
        let level = (side as f64).log2() as u32;
        self.from_flat_recursive(grid, side, 0, 0, 0, level)
    }

    fn from_flat_recursive(
        &mut self,
        grid: &[CellState],
        side: usize,
        ox: usize,
        oy: usize,
        oz: usize,
        level: u32,
    ) -> NodeId {
        if level == 0 {
            return self.leaf(grid[ox + oy * side + oz * side * side]);
        }
        let half = 1usize << (level - 1);
        let mut children = [NodeId::EMPTY; 8];
        for oct in 0..8 {
            let (cx, cy, cz) = super::node::octant_coords(oct);
            children[oct] = self.from_flat_recursive(
                grid,
                side,
                ox + cx as usize * half,
                oy + cy as usize * half,
                oz + cz as usize * half,
                level - 1,
            );
        }
        self.interior(level, children)
    }

    /// Cache a step result.
    pub fn cache_step(&mut self, input: NodeId, result: NodeId) {
        self.step_cache.insert(input, result);
    }

    /// Look up a cached step result.
    pub fn get_cached_step(&self, input: NodeId) -> Option<NodeId> {
        self.step_cache.get(&input).copied()
    }

    /// Clear step cache (call when changing rules).
    pub fn clear_step_cache(&mut self) {
        self.step_cache.clear();
    }

    /// Stats for debugging.
    pub fn stats(&self) -> (usize, usize) {
        (self.nodes.len(), self.step_cache.len())
    }

    /// Return a fresh `NodeStore` containing only the nodes reachable from
    /// `root`, together with the remapped root id.
    ///
    /// ## Why this exists
    ///
    /// `NodeStore` is append-only: `set_cell` and `from_flat` intern new
    /// nodes on every call and never drop the old ones. Over time, that
    /// means every obsolete generation's subtrees leak into `nodes` forever.
    /// `compacted` is the explicit "new epoch" that drops everything
    /// unreachable from `root` by rebuilding a clean store.
    ///
    /// ## Semantics
    ///
    /// The returned store is structurally equivalent (every cell under
    /// `root` is preserved, hash-cons sharing is preserved — structurally
    /// equal subtrees still share a single NodeId). `NodeId::EMPTY` still
    /// lives at slot 0. All other NodeIds from `self` are invalid against
    /// the returned store and must not be mixed.
    ///
    /// ## Cache lifetime
    ///
    /// `compacted` cannot currently preserve `step_cache`: the cache is
    /// keyed on `NodeId`, and the new store uses a fresh id space. The
    /// `debug_assert!` below enforces "cache lifetime == store epoch" —
    /// callers must clear the step cache before compacting (or, once
    /// hash-thing-6gf.2 lands, the memoization layer must live outside
    /// `NodeStore` or remap across compaction).
    pub fn compacted(&self, root: NodeId) -> (NodeStore, NodeId) {
        debug_assert!(
            self.step_cache.is_empty(),
            "compacted() cannot preserve step_cache yet — see hash-thing-6gf.2"
        );
        let mut dst = NodeStore::new();
        let mut remap: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        // Pre-seed EMPTY as a perf short-circuit. Note: this is NOT
        // semantically load-bearing — `dst.leaf(0)` already dedups to
        // `NodeId::EMPTY` via the hash-cons path — but it skips the
        // recursive walk whenever we hit an empty subtree.
        remap.insert(NodeId::EMPTY, NodeId::EMPTY);
        let new_root = clone_reachable(self, &mut dst, &mut remap, root);
        (dst, new_root)
    }
}

/// Recursively clone the subtree rooted at `old` from `src` into `dst`,
/// populating `remap` with old-id → new-id entries.
///
/// ## Post-order ordering is load-bearing
///
/// This function walks **post-order**: every child is cloned into `dst`
/// before the parent is interned. That ordering is required because
/// `NodeStore::interior` re-computes the parent's `population` by
/// calling `dst.population(child)` on each new child — it reads from
/// `dst`, not from `src`. If we interned the parent before cloning its
/// children into `dst`, the lookup would hit uninitialized slots (or
/// worse, stale values from other nodes). Post-order guarantees the
/// children are canonical in `dst` by the time the parent is built, so
/// hash-cons dedup, population folding, and NodeId identity all come out
/// right in a single pass.
///
/// TODO: this is recursive. At level 6 (our current root) that's 7
/// stack frames, which is fine. At level 20+ it would need iteration.
fn clone_reachable(
    src: &NodeStore,
    dst: &mut NodeStore,
    remap: &mut FxHashMap<NodeId, NodeId>,
    old: NodeId,
) -> NodeId {
    if let Some(&new_id) = remap.get(&old) {
        return new_id;
    }
    let new_id = match src.get(old).clone() {
        Node::Leaf(state) => dst.leaf(state),
        Node::Interior {
            level, children, ..
        } => {
            let mut new_children = [NodeId::EMPTY; 8];
            for (i, &child) in children.iter().enumerate() {
                new_children[i] = clone_reachable(src, dst, remap, child);
            }
            dst.interior(level, new_children)
        }
    };
    remap.insert(old, new_id);
    new_id
}

#[cfg(test)]
mod tests {
    //! NodeStore hash-cons test suite.
    //!
    //! Hash-consing is load-bearing for Hashlife: two structurally equal
    //! subtrees must share a single NodeId, and population must fold up
    //! exactly so the step cache can trust its keys. These tests exercise
    //! the invariants that everything downstream (Hashlife stepper, SVDAG
    //! builder, world flatten path) will lean on.
    //!
    //! Out-of-bounds coordinate handling is deliberately NOT tested here —
    //! that's tracked separately by hash-thing-819.
    use super::*;

    /// Canonical empty leaf is at NodeId(0).
    #[test]
    fn empty_leaf_is_node_id_zero() {
        let store = NodeStore::new();
        assert_eq!(NodeId::EMPTY, NodeId(0));
        assert!(matches!(store.get(NodeId::EMPTY), Node::Leaf(0)));
        assert_eq!(store.population(NodeId::EMPTY), 0);
    }

    /// `leaf(0)` after construction is the same id as the pre-reserved
    /// canonical empty leaf. This is the "two paths, one id" invariant
    /// applied at the base case.
    #[test]
    fn leaf_zero_dedups_with_empty() {
        let mut store = NodeStore::new();
        let a = store.leaf(0);
        assert_eq!(a, NodeId::EMPTY);
    }

    /// Interning the same leaf twice yields the same id.
    #[test]
    fn intern_dedup_leaf() {
        let mut store = NodeStore::new();
        let a = store.leaf(7);
        let b = store.leaf(7);
        assert_eq!(a, b);
        // And a different state gets a different id.
        let c = store.leaf(8);
        assert_ne!(a, c);
    }

    /// Interning the same interior node twice yields the same id.
    /// This is the core hash-consing invariant.
    #[test]
    fn intern_dedup_interior() {
        let mut store = NodeStore::new();
        let l0 = store.leaf(0);
        let l1 = store.leaf(1);
        // Two independent calls to build an interior with the same
        // children must return the same NodeId.
        let a = store.interior(1, [l0, l1, l0, l1, l0, l1, l0, l1]);
        let b = store.interior(1, [l0, l1, l0, l1, l0, l1, l0, l1]);
        assert_eq!(a, b);
        // Different child arrangement → different id.
        let c = store.interior(1, [l1, l0, l0, l1, l0, l1, l0, l1]);
        assert_ne!(a, c);
    }

    /// Empty subtrees share one id regardless of build path.
    /// `uniform(level, 0)` and `interior(level, [empty_child_of_level_minus_1; 8])`
    /// must land on the same NodeId — otherwise the node store leaks empties.
    #[test]
    fn empty_subtree_unique_across_build_paths() {
        let mut store = NodeStore::new();
        // Path 1: the uniform helper.
        let via_uniform = store.uniform(3, 0);
        // Path 2: recursively build an empty interior by hand.
        let e0 = store.leaf(0);
        let e1 = store.interior(1, [e0; 8]);
        let e2 = store.interior(2, [e1; 8]);
        let e3 = store.interior(3, [e2; 8]);
        assert_eq!(via_uniform, e3);
        // Path 3: the empty() shorthand.
        let via_empty = store.empty(3);
        assert_eq!(via_uniform, via_empty);
    }

    /// Population at an interior node is the sum of its children's
    /// populations. Checked across a few hand-built shapes.
    #[test]
    fn population_folds_up_correctly() {
        let mut store = NodeStore::new();
        let l0 = store.leaf(0);
        let l1 = store.leaf(1);
        // Interior with 3 live leaves → pop 3.
        let n = store.interior(1, [l1, l0, l1, l0, l1, l0, l0, l0]);
        assert_eq!(store.population(n), 3);
        // A level-2 interior whose children each have known populations.
        let a = store.interior(1, [l1, l1, l1, l1, l1, l1, l1, l1]); // pop 8
        let b = store.interior(1, [l0; 8]); // pop 0
        assert_eq!(store.population(a), 8);
        assert_eq!(store.population(b), 0);
        let parent = store.interior(2, [a, b, a, b, a, b, a, b]);
        assert_eq!(store.population(parent), 8 * 4);
    }

    /// `set_cell` then `get_cell` roundtrips at a few levels.
    #[test]
    fn set_get_roundtrip_small() {
        let mut store = NodeStore::new();
        let mut root = store.empty(3); // 8^3
        root = store.set_cell(root, 0, 0, 0, 5);
        root = store.set_cell(root, 7, 7, 7, 9);
        root = store.set_cell(root, 3, 4, 5, 2);
        assert_eq!(store.get_cell(root, 0, 0, 0), 5);
        assert_eq!(store.get_cell(root, 7, 7, 7), 9);
        assert_eq!(store.get_cell(root, 3, 4, 5), 2);
        // Untouched cells stay empty.
        assert_eq!(store.get_cell(root, 1, 1, 1), 0);
    }

    /// Same test at a deeper level — the recursion has to keep halving
    /// correctly all the way down.
    #[test]
    fn set_get_roundtrip_level_6() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6); // 64^3
        let probes: &[(u64, u64, u64, u8)] = &[
            (0, 0, 0, 1),
            (63, 63, 63, 2),
            (31, 31, 31, 3),
            (32, 32, 32, 4),
            (10, 20, 30, 5),
            (50, 5, 25, 6),
        ];
        for &(x, y, z, s) in probes {
            root = store.set_cell(root, x, y, z, s);
        }
        for &(x, y, z, s) in probes {
            assert_eq!(store.get_cell(root, x, y, z), s, "cell ({x},{y},{z})");
        }
    }

    /// `from_flat` then `flatten` returns the original bytes.
    #[test]
    fn from_flat_flatten_roundtrip() {
        let side = 8usize;
        let mut grid = vec![0u8; side * side * side];
        // Put a recognizable pattern in.
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    grid[x + y * side + z * side * side] = ((x ^ y ^ z) & 0x7) as u8;
                }
            }
        }
        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);
        let out = store.flatten(root, side);
        assert_eq!(out, grid);
    }

    /// An all-empty `from_flat` must dedup down to the canonical empty
    /// subtree — i.e. building an empty 32^3 grid and building it via
    /// `empty(5)` yield the same root.
    #[test]
    fn from_flat_empty_dedups_with_empty_subtree() {
        let side = 32usize;
        let grid = vec![0u8; side * side * side];
        let mut store = NodeStore::new();
        let via_flat = store.from_flat(&grid, side);
        let via_empty = store.empty(5);
        assert_eq!(via_flat, via_empty);
    }

    /// Population on a `from_flat` root matches the number of nonzero
    /// cells in the source grid.
    #[test]
    fn from_flat_population_matches_source() {
        let side = 8usize;
        let mut grid = vec![0u8; side * side * side];
        // Exactly 17 live cells.
        let live: &[(usize, usize, usize)] = &[
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (0, 0, 4),
            (7, 7, 7),
            (6, 6, 6),
            (5, 5, 5),
            (4, 4, 4),
        ];
        for &(x, y, z) in live {
            grid[x + y * side + z * side * side] = 1;
        }
        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);
        assert_eq!(store.population(root), live.len() as u64);
    }

    /// Two independently built octrees of the same content share their
    /// root NodeId. This is what makes the step cache actually save work.
    #[test]
    fn structurally_equal_trees_share_root() {
        let side = 8usize;
        let mut grid = vec![0u8; side * side * side];
        grid[0] = 1;
        grid[side - 1] = 2;
        grid[(side - 1) * side + 4] = 3;
        let mut store = NodeStore::new();
        let a = store.from_flat(&grid, side);
        let b = store.from_flat(&grid, side);
        assert_eq!(a, b);
        // And a grid that differs by one cell must NOT share.
        let mut grid2 = grid.clone();
        grid2[1] = 1;
        let c = store.from_flat(&grid2, side);
        assert_ne!(a, c);
    }

    /// Hash-consing: building a large empty world must not allocate a
    /// node per leaf. A 64^3 empty grid should intern only a handful of
    /// nodes (one per level, plus NodeId::EMPTY) — specifically 7 nodes
    /// (levels 0..=6). This is a canary for "every dedup path actually
    /// dedups."
    #[test]
    fn empty_world_is_compact() {
        let side = 64usize;
        let grid = vec![0u8; side * side * side];
        let mut store = NodeStore::new();
        let _root = store.from_flat(&grid, side);
        let (nodes, _) = store.stats();
        // 7 levels (leaf through level 6), one canonical node per level.
        assert_eq!(nodes, 7, "expected 7 nodes for an empty 64^3, got {nodes}");
    }

    /// Step cache round-trip. Not exercising Hashlife yet — just that
    /// insert / lookup / clear behave.
    #[test]
    fn step_cache_insert_lookup_clear() {
        let mut store = NodeStore::new();
        let a = store.leaf(1);
        let b = store.leaf(2);
        assert_eq!(store.get_cached_step(a), None);
        store.cache_step(a, b);
        assert_eq!(store.get_cached_step(a), Some(b));
        store.clear_step_cache();
        assert_eq!(store.get_cached_step(a), None);
    }

    /// `stats()` reports (nodes, cache_len).
    #[test]
    fn stats_tracks_nodes_and_cache() {
        let mut store = NodeStore::new();
        let (n0, c0) = store.stats();
        assert_eq!(n0, 1); // just the empty leaf
        assert_eq!(c0, 0);
        let a = store.leaf(1);
        let b = store.leaf(2);
        let (n1, _) = store.stats();
        assert_eq!(n1, 3);
        store.cache_step(a, b);
        let (_, c1) = store.stats();
        assert_eq!(c1, 1);
    }

    // ===========================================================================
    // Fresh-store compaction tests (hash-thing-88d, T1–T9).
    //
    // `compacted()` rebuilds the arena into a fresh `NodeStore` containing only
    // nodes reachable from `root`. The invariants we care about:
    //   (a) every cell under `root` round-trips identically,
    //   (b) hash-cons sharing is preserved (structurally-equal subtrees still
    //       share a NodeId),
    //   (c) population folds up correctly at every interior node,
    //   (d) `NodeId::EMPTY` survives at slot 0,
    //   (e) compaction is idempotent.
    // ===========================================================================

    /// T1: empty store compacts to a single-EMPTY store.
    #[test]
    fn compact_nothing_empty() {
        let store = NodeStore::new();
        let (compacted, new_root) = store.compacted(NodeId::EMPTY);
        assert_eq!(new_root, NodeId::EMPTY);
        let (nodes, cache) = compacted.stats();
        assert_eq!(
            nodes, 1,
            "empty compaction should yield only the canonical empty leaf"
        );
        assert_eq!(cache, 0);
    }

    /// T2: after one `set_cell`, compaction's node count matches a
    /// freshly-built one-cell world — NOT the pre-compaction size, because
    /// `set_cell` is persistent and the clone path's old nodes are dead.
    #[test]
    fn compact_after_set_cell_matches_fresh_one_cell_world() {
        // Build store A: empty(6), set one cell.
        let mut a = NodeStore::new();
        let mut root_a = a.empty(6);
        root_a = a.set_cell(root_a, 10, 20, 30, 7);
        let pre_nodes = a.stats().0;
        let (compacted_a, new_root_a) = a.compacted(root_a);

        // Build store B: a fresh one-cell world — empty(6), set the same cell.
        let mut b = NodeStore::new();
        let mut root_b = b.empty(6);
        root_b = b.set_cell(root_b, 10, 20, 30, 7);
        // Compact B too so both sides went through the same canonicalization.
        let (compacted_b, _new_root_b) = b.compacted(root_b);

        assert_eq!(
            compacted_a.stats().0,
            compacted_b.stats().0,
            "compacted one-cell world should match fresh one-cell world node count"
        );
        assert!(
            compacted_a.stats().0 < pre_nodes,
            "compaction should drop dead clone-path nodes ({} → {})",
            pre_nodes,
            compacted_a.stats().0
        );
        // Cell round-trip.
        assert_eq!(compacted_a.get_cell(new_root_a, 10, 20, 30), 7);
        assert_eq!(compacted_a.get_cell(new_root_a, 0, 0, 0), 0);
    }

    /// T3: after several step-like rebuilds (simulate churn by calling
    /// `from_flat` repeatedly on mutated grids), compacted node count is
    /// ≤ pre-compaction count, and every cell still round-trips.
    #[test]
    fn compact_after_churn_preserves_cells() {
        let side = 16usize;
        let mut store = NodeStore::new();
        // Seed a recognizable pattern.
        let mut grid = vec![0u8; side * side * side];
        for i in 0..side {
            grid[i + i * side + i * side * side] = (i as u8 % 7) + 1;
        }
        let mut root = store.from_flat(&grid, side);
        // Churn: rebuild a handful of generations. We don't need real CA
        // semantics — just that the store sees repeated `from_flat` calls
        // with slightly different grids, leaking old subtrees.
        for gen in 0..5 {
            for x in 0..side {
                grid[x] = ((gen + x) as u8 % 5) + 1;
            }
            root = store.from_flat(&grid, side);
        }
        let pre_nodes = store.stats().0;
        let (compacted, new_root) = store.compacted(root);
        assert!(
            compacted.stats().0 <= pre_nodes,
            "compaction should not grow the store ({} → {})",
            pre_nodes,
            compacted.stats().0
        );
        // Every cell under the final root must match.
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let expected = store.get_cell(root, x as u64, y as u64, z as u64);
                    let got = compacted.get_cell(new_root, x as u64, y as u64, z as u64);
                    assert_eq!(
                        got, expected,
                        "cell ({x},{y},{z}) drifted during compaction"
                    );
                }
            }
        }
    }

    /// T4: cell-by-cell equality against a freshly-built world. We deliberately
    /// do NOT compare node counts against a fresh `from_flat` build — that
    /// breaks on the all-empty edge (compacted empty root = 1 node,
    /// `from_flat` empty = 7 nodes one-per-level before the level-0 dedup
    /// cascade). Cell equality is the real contract.
    #[test]
    fn compact_cells_match_fresh_build() {
        let side = 8usize;
        let mut grid = vec![0u8; side * side * side];
        grid[0] = 5;
        grid[side * side * side - 1] = 9;
        grid[3 + 4 * side + 5 * side * side] = 2;

        let mut dirty = NodeStore::new();
        // Churn the store: build, mutate, build again — leaves dead nodes.
        let _waste1 = dirty.from_flat(&grid, side);
        let mut grid2 = grid.clone();
        grid2[1] = 3;
        let _waste2 = dirty.from_flat(&grid2, side);
        let final_root = dirty.from_flat(&grid, side);

        let (compacted, new_root) = dirty.compacted(final_root);

        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    assert_eq!(
                        compacted.get_cell(new_root, x as u64, y as u64, z as u64),
                        grid[x + y * side + z * side * side],
                        "cell ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// T5: hash-cons preservation across compaction. If a non-trivial subtree
    /// is reused in two child slots of a parent in the source, the compacted
    /// store must also land on a single NodeId for both slots. This is the
    /// load-bearing dedup test — it's what makes compaction safe under future
    /// memoization.
    #[test]
    fn compact_preserves_shared_non_empty_interiors() {
        let mut store = NodeStore::new();
        // Build a non-trivial level-1 subtree (not all-empty, not uniform).
        let l0 = store.leaf(0);
        let l1 = store.leaf(1);
        let l2 = store.leaf(2);
        let shared = store.interior(1, [l1, l0, l2, l0, l1, l0, l2, l0]);
        // Use `shared` in two slots of a level-2 parent (alongside other stuff).
        let other = store.interior(1, [l0, l1, l0, l1, l0, l1, l0, l1]);
        let parent = store.interior(
            2,
            [shared, other, shared, other, other, shared, other, shared],
        );
        // Sanity: shared is indeed non-empty.
        assert!(store.population(shared) > 0);

        let (compacted, new_parent) = store.compacted(parent);
        let children = compacted.children(new_parent);
        // Slots 0, 2, 5, 7 were `shared`; slots 1, 3, 4, 6 were `other`.
        assert_eq!(
            children[0], children[2],
            "shared subtree slots 0/2 must dedup"
        );
        assert_eq!(
            children[0], children[5],
            "shared subtree slots 0/5 must dedup"
        );
        assert_eq!(
            children[0], children[7],
            "shared subtree slots 0/7 must dedup"
        );
        assert_eq!(
            children[1], children[3],
            "other subtree slots 1/3 must dedup"
        );
        assert_eq!(
            children[1], children[4],
            "other subtree slots 1/4 must dedup"
        );
        assert_eq!(
            children[1], children[6],
            "other subtree slots 1/6 must dedup"
        );
        // And shared ≠ other post-compaction.
        assert_ne!(children[0], children[1]);
    }

    /// T6: compaction preserves per-cell material payload across multiple
    /// distinct `CellState` values. `CellState` is `u8` so 255 materials;
    /// compaction must be payload-transparent, not just shape-transparent.
    #[test]
    fn compact_preserves_multiple_cell_states() {
        let side = 8usize;
        let mut grid = vec![0u8; side * side * side];
        // Sprinkle a handful of distinct states.
        let cells: &[(usize, usize, usize, u8)] = &[
            (0, 0, 0, 1),
            (1, 0, 0, 2),
            (2, 0, 0, 3),
            (0, 1, 0, 4),
            (0, 2, 0, 5),
            (0, 0, 1, 6),
            (0, 0, 2, 7),
            (7, 7, 7, 255),
            (3, 4, 5, 128),
        ];
        for &(x, y, z, s) in cells {
            grid[x + y * side + z * side * side] = s;
        }
        let mut store = NodeStore::new();
        // Churn so compaction has something to drop.
        let mut g2 = grid.clone();
        g2[0] = 99;
        let _waste = store.from_flat(&g2, side);
        let root = store.from_flat(&grid, side);
        let (compacted, new_root) = store.compacted(root);

        for &(x, y, z, s) in cells {
            assert_eq!(
                compacted.get_cell(new_root, x as u64, y as u64, z as u64),
                s,
                "cell ({x},{y},{z}) material drifted"
            );
        }
        // And a few off-pattern cells should be zero.
        assert_eq!(compacted.get_cell(new_root, 5, 5, 5), 0);
        assert_eq!(compacted.get_cell(new_root, 4, 4, 4), 0);
    }

    /// T7: idempotence. Compacting twice must give a store with identical
    /// `nodes.len()` and identical cell contents — `compacted` is a fixpoint.
    #[test]
    fn compact_is_idempotent() {
        let side = 16usize;
        let mut grid = vec![0u8; side * side * side];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    grid[x + y * side + z * side * side] = ((x * 7 + y * 11 + z * 13) % 17) as u8;
                }
            }
        }
        let mut dirty = NodeStore::new();
        // Churn.
        let mut g2 = grid.clone();
        g2[0] = 254;
        let _w1 = dirty.from_flat(&g2, side);
        let mut g3 = grid.clone();
        g3[1] = 253;
        let _w2 = dirty.from_flat(&g3, side);
        let root = dirty.from_flat(&grid, side);

        let (once, root_once) = dirty.compacted(root);
        let (twice, root_twice) = once.compacted(root_once);
        assert_eq!(
            once.stats().0,
            twice.stats().0,
            "compaction is not idempotent ({} → {})",
            once.stats().0,
            twice.stats().0
        );
        // Cell-by-cell equality across the two compactions.
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let a = once.get_cell(root_once, x as u64, y as u64, z as u64);
                    let b = twice.get_cell(root_twice, x as u64, y as u64, z as u64);
                    assert_eq!(a, b, "cell ({x},{y},{z}) changed across second compaction");
                }
            }
        }
    }

    /// T8: population invariant holds at every interior node in the compacted
    /// store, not just at the root. This is the structural check that
    /// `dst.interior`'s population refold ran in the right order during
    /// `clone_reachable` — the function's post-order walk is load-bearing
    /// because `dst.interior` reads children populations from `dst`.
    #[test]
    fn compact_population_invariant_at_every_interior() {
        let side = 16usize;
        let mut grid = vec![0u8; side * side * side];
        for i in 0..side {
            grid[i + (i % 4) * side + (i % 3) * side * side] = ((i % 7) + 1) as u8;
        }
        let mut store = NodeStore::new();
        // Churn.
        let mut g2 = grid.clone();
        g2[0] = 77;
        let _waste = store.from_flat(&g2, side);
        let root = store.from_flat(&grid, side);
        let (compacted, new_root) = store.compacted(root);

        // Walk every reachable interior node from new_root and assert
        // population = sum(children populations).
        fn walk(store: &NodeStore, id: NodeId, visited: &mut FxHashMap<NodeId, ()>) {
            if visited.insert(id, ()).is_some() {
                return;
            }
            match store.get(id) {
                Node::Leaf(_) => {}
                Node::Interior {
                    children,
                    population,
                    ..
                } => {
                    let sum: u64 = children.iter().map(|&c| store.population(c)).sum();
                    assert_eq!(
                        *population, sum,
                        "interior node {id:?} population {} != sum of children {}",
                        *population, sum
                    );
                    for &c in children {
                        walk(store, c, visited);
                    }
                }
            }
        }
        let mut visited = FxHashMap::default();
        walk(&compacted, new_root, &mut visited);
    }

    /// T9: `NodeId::EMPTY` survives compaction at slot 0. The fresh store is
    /// constructed with the canonical empty leaf pre-reserved; compaction
    /// must not shift it.
    #[test]
    fn compact_preserves_empty_at_slot_zero() {
        let mut store = NodeStore::new();
        let mut root = store.empty(4);
        root = store.set_cell(root, 3, 3, 3, 1);
        let (compacted, _new_root) = store.compacted(root);
        assert_eq!(NodeId::EMPTY, NodeId(0));
        assert!(matches!(compacted.get(NodeId::EMPTY), Node::Leaf(0)));
        assert_eq!(compacted.population(NodeId::EMPTY), 0);
    }
}
