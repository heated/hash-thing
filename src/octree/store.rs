use rustc_hash::FxHashMap;
use super::node::{CellState, Node, NodeId, octant_index};

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
            Node::Interior { level, children, .. } => {
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
            Node::Interior { level, children, .. } => {
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
            Node::Interior { level, children, population, .. } => {
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
