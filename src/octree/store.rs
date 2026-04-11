use super::node::{octant_index, Cell, CellState, Node, NodeId};
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
    /// Memoized step results: (node_id, rule_id) -> result_node_id.
    /// The result of a level-n node is the center (level n-1) cube
    /// after 2^(n-2) steps.
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

    /// Create a leaf node. Validates the `Cell` invariant on `state`
    /// (raw bit patterns in `1..=MAX_METADATA` are rejected because they
    /// would decode to "empty with flavor" and silently alias `NodeId::EMPTY`
    /// after hash-consing).
    pub fn leaf(&mut self, state: CellState) -> NodeId {
        let _ = Cell::from_raw(state);
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
    ///
    /// Validates the `Cell` invariant on `state` at entry — so passing
    /// e.g. a raw `1` (which would decode to material 0, metadata 1 and
    /// corrupt hash-consing) panics immediately instead of silently
    /// aliasing empty.
    pub fn set_cell(&mut self, root: NodeId, x: u64, y: u64, z: u64, state: CellState) -> NodeId {
        let _ = Cell::from_raw(state);
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
        let mut grid = vec![0 as CellState; side * side * side];
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
}
