use super::node::{octant_index, Cell, CellState, Node, NodeId};
use rustc_hash::{FxHashMap, FxHashSet};

/// Output type for the per-chunk reachable-node walker: `Vec` of
/// `((chunk_x, chunk_y, chunk_z), reachable_count)` in row-major
/// (cz outer, cy middle, cx inner) order. See
/// [`NodeStore::reachable_nodes_per_chunk`].
pub type ChunkReachable = Vec<((u64, u64, u64), u64)>;

/// Walker state for [`NodeStore::reachable_summary`]. Bundles the
/// arguments threaded through the recursion so the call site stays
/// within clippy's `too_many_arguments` budget.
struct ChunkWalk {
    chunk_level: u32,
    out: ChunkReachable,
    global: FxHashSet<NodeId>,
}

/// Canonical node store — the core of hash-consing.
///
/// Every unique node exists exactly once. Identical subtrees are shared.
/// This gives us Hashlife's spatial compression: a 4096³ world that's
/// mostly empty costs almost nothing.
///
/// Node indices are `u32`, limiting the store to ~4 billion unique nodes.
/// In practice, hash-consing keeps the count far below this: a 4096³
/// world with terrain + active water typically produces ~200K nodes.
/// The `intern_node` path panics with a diagnostic message if this limit
/// is ever reached.
#[derive(Clone)]
pub struct NodeStore {
    /// All canonical nodes, indexed by NodeId.
    nodes: Vec<Node>,
    /// Reverse lookup: node -> its canonical id.
    intern: FxHashMap<Node, NodeId>,
}

impl Default for NodeStore {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeStore {
    pub fn new() -> Self {
        let mut store = Self {
            nodes: Vec::new(),
            intern: FxHashMap::default(),
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
        let len = self.nodes.len();
        let id = NodeId(u32::try_from(len).unwrap_or_else(|_| {
            panic!(
                "NodeStore overflow: {len} nodes exceeds u32::MAX ({}). \
                 This means the world has more unique subtrees than 2^32 — \
                 consider compacting more aggressively or widening NodeId to u64.",
                u32::MAX
            )
        }));
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
    ///
    /// Coordinates are relative to the node's origin (0,0,0 corner) and MUST
    /// be strictly less than `1 << root_level`. Out-of-bounds writes **panic**
    /// (hash-thing-fb5): silent data corruption is unacceptable, and the
    /// bounds check is a single compare against `side` at the public entry.
    ///
    /// Also validates the `Cell` invariant on `state` at entry (hash-thing-1v0.1):
    /// passing e.g. a raw `1` (which would decode to material 0, metadata 1 and
    /// corrupt hash-consing by aliasing `Leaf(0) == NodeId::EMPTY`) panics
    /// immediately instead of silently poisoning the store.
    ///
    /// Footgun: `root` must be an actual world root whose `Node::level()`
    /// reflects the realized extent. Passing `NodeId::EMPTY` (level 0) turns
    /// every non-origin coordinate into an OOB panic because the entry's
    /// `side` computation resolves to 1.
    #[track_caller]
    pub fn set_cell(&mut self, root: NodeId, x: u64, y: u64, z: u64, state: CellState) -> NodeId {
        let _ = Cell::from_raw(state);
        let level = self.get(root).level();
        let side = 1u64 << level;
        assert!(
            x < side && y < side && z < side,
            "NodeStore::set_cell: coord ({x}, {y}, {z}) out of bounds for side {side} (level {level})",
        );
        self.set_cell_at(root, level, x, y, z, state)
    }

    /// Recursive worker for `set_cell`. `level` is the target level of `node`
    /// as seen from its parent — this may differ from `Node::level()` when a
    /// `Leaf` represents a uniform subtree compressed at an intermediate level
    /// (uniform-subtree compression, empty-subtree collapse, etc.).
    ///
    /// Precondition: coordinates are already in-bounds for `level`. The
    /// public `set_cell` entry enforces this; direct callers would have to
    /// re-derive it. The `debug_assert!` below makes the contract executable
    /// in dev without adding release overhead.
    fn set_cell_at(
        &mut self,
        node: NodeId,
        level: u32,
        x: u64,
        y: u64,
        z: u64,
        state: CellState,
    ) -> NodeId {
        // Unconditional — at level 0 the check is `x == 0 && y == 0 && z == 0`,
        // which is the only valid coord triple inside a single-voxel leaf and
        // is exactly what the public entry's recursion preserves. Keeping the
        // assert fire-able at the base case is load-bearing: if someone ever
        // adds level-0 coord logic, this catches an out-of-range leak.
        debug_assert!(
            x < (1u64 << level) && y < (1u64 << level) && z < (1u64 << level),
            "set_cell_at: coord ({x}, {y}, {z}) out of bounds for level {level}",
        );
        if level == 0 {
            return self.leaf(state);
        }
        // If we encounter a Leaf at level > 0, the caller's context says this
        // node stands for a uniform 2^level × 2^level × 2^level block. Expand
        // it to 8 identical children at level-1 before descending, so we only
        // rewrite the one octant that actually changed.
        let children = match *self.get(node) {
            Node::Leaf(s) => {
                let child = self.uniform(level - 1, s);
                [child; 8]
            }
            Node::Interior { children, .. } => children,
        };
        let half = 1u64 << (level - 1);
        let ox = if x >= half { 1u32 } else { 0 };
        let oy = if y >= half { 1u32 } else { 0 };
        let oz = if z >= half { 1u32 } else { 0 };
        let idx = octant_index(ox, oy, oz);
        let lx = if ox == 1 { x - half } else { x };
        let ly = if oy == 1 { y - half } else { y };
        let lz = if oz == 1 { z - half } else { z };
        let new_child = self.set_cell_at(children[idx], level - 1, lx, ly, lz, state);
        let mut new_children = children;
        new_children[idx] = new_child;
        self.interior(level, new_children)
    }

    /// Get a single cell value.
    ///
    /// Out-of-bounds reads return `0` silently (hash-thing-fb5). `get_cell`
    /// is a pure function — it never grows storage, never panics. Outside
    /// the realized region is conceptually unrealized empty space. The
    /// raycaster, the eventual Hashlife stepper, and any debug visualizer
    /// rely on this contract to probe arbitrarily far without ballooning
    /// storage. Realization happens through separate explicit APIs
    /// (`World::ensure_contains`, future `realize_region`), not here.
    ///
    /// Footgun: `root` must be a real world root. `NodeId::EMPTY` has level
    /// 0, so every non-origin coordinate reflects to 0 via the OOB branch.
    pub fn get_cell(&self, root: NodeId, x: u64, y: u64, z: u64) -> CellState {
        let level = self.get(root).level();
        let side = 1u64 << level;
        if x >= side || y >= side || z >= side {
            return 0;
        }
        self.get_cell_at(root, level, x, y, z)
    }

    /// Recursive worker for `get_cell`. Precondition: coordinates are in
    /// bounds for `level`. The public `get_cell` entry enforces this; the
    /// `debug_assert!` makes the contract executable in dev without any
    /// release overhead.
    ///
    /// Note: unlike `set_cell_at`, this path does not need Leaf-at-
    /// intermediate-level expansion — reads fall through the Leaf arm and
    /// return the uniform value. `level` is the parent-visible level; for
    /// Interior nodes it equals `self.get(node).level()` by the invariant
    /// the public entry establishes. Recursion uses `level` consistently
    /// with `set_cell_at` for symmetry.
    fn get_cell_at(&self, node: NodeId, level: u32, x: u64, y: u64, z: u64) -> CellState {
        debug_assert!(
            x < (1u64 << level) && y < (1u64 << level) && z < (1u64 << level),
            "get_cell_at: coord ({x}, {y}, {z}) out of bounds for level {level}",
        );
        match self.get(node) {
            Node::Leaf(s) => *s,
            Node::Interior { children, .. } => {
                let half = 1u64 << (level - 1);
                let ox = if x >= half { 1u32 } else { 0 };
                let oy = if y >= half { 1u32 } else { 0 };
                let oz = if z >= half { 1u32 } else { 0 };
                let idx = octant_index(ox, oy, oz);
                let lx = if ox == 1 { x - half } else { x };
                let ly = if oy == 1 { y - half } else { y };
                let lz = if oz == 1 { z - half } else { z };
                self.get_cell_at(children[idx], level - 1, lx, ly, lz)
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

    /// Flatten into a caller-provided buffer. The buffer must be exactly
    /// `side³` elements and should be zero-initialized.
    pub fn flatten_buf(&self, root: NodeId, grid: &mut [CellState], side: usize) {
        debug_assert_eq!(grid.len(), side * side * side);
        self.flatten_into(root, grid, side, 0, 0, 0);
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
                for (oct, child) in children.iter().enumerate() {
                    let (cx, cy, cz) = super::node::octant_coords(oct);
                    self.flatten_into(
                        *child,
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
    ///
    /// Release-panics on non-power-of-two `side` or mismatched grid length:
    /// pre-hash-thing-nch this was a `debug_assert!` plus a silent
    /// `(side as f64).log2() as u32` truncation — `side=48` built a depth-5
    /// (32³) tree over 48³ data and read past the buffer end on the upper
    /// 16 cells of every axis. Contract violations are release-panics now.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_flat(&mut self, grid: &[CellState], side: usize) -> NodeId {
        assert!(
            side.is_power_of_two() && side >= 1,
            "from_flat: side must be a power of two >= 1; got {side}",
        );
        assert_eq!(
            grid.len(),
            side * side * side,
            "from_flat: grid length {} does not match side^3 = {}",
            grid.len(),
            side * side * side,
        );
        // `trailing_zeros()` is exact on power-of-two values and can't
        // truncate the way `(side as f64).log2() as u32` did.
        let level = side.trailing_zeros();
        self.from_flat_recursive(grid, side, 0, 0, 0, level)
    }

    #[allow(clippy::wrong_self_convention)]
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
        for (oct, child) in children.iter_mut().enumerate() {
            let (cx, cy, cz) = super::node::octant_coords(oct);
            *child = self.from_flat_recursive(
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

    // ---- Column primitives (hash-thing-jw3k.1) ----

    /// Flatten a single (x, z) column (all y cells) into the caller-provided
    /// buffer. `out.len()` must equal `1 << root.level()`.
    ///
    /// Walks only the 2 y-children at each level (octants `(ox, 0, oz)` and
    /// `(ox, 1, oz)`), honors the `population == 0` early-out, and handles
    /// intermediate `Leaf(state)` nodes by filling the corresponding y-range
    /// with the uniform state.
    ///
    /// Cost per column is `2 * side - 1` node visits — flat and predictable.
    /// Intended for sim hot paths that need per-column state without
    /// materializing the whole 3D grid; cross-crate visibility is required
    /// because the caller lives in the `hash-thing` crate's sim module.
    pub fn flatten_column_into(&self, root: NodeId, x: u64, z: u64, out: &mut [CellState]) {
        let level = self.get(root).level();
        let side = 1u64 << level;
        assert!(
            x < side && z < side,
            "flatten_column_into: ({x}, {z}) out of bounds for side {side}",
        );
        assert_eq!(
            out.len() as u64,
            side,
            "flatten_column_into: buffer len {} != side {}",
            out.len(),
            side,
        );
        self.flatten_column_rec(root, level, x, z, out);
    }

    fn flatten_column_rec(&self, node: NodeId, level: u32, x: u64, z: u64, out: &mut [CellState]) {
        debug_assert_eq!(out.len() as u64, 1u64 << level);
        match self.get(node) {
            Node::Leaf(s) => {
                out.fill(*s);
            }
            Node::Interior {
                children,
                population,
                ..
            } => {
                if *population == 0 {
                    out.fill(0);
                    return;
                }
                let children = *children;
                let half = 1u64 << (level - 1);
                let ox = if x >= half { 1u32 } else { 0 };
                let oz = if z >= half { 1u32 } else { 0 };
                let lx = if ox == 1 { x - half } else { x };
                let lz = if oz == 1 { z - half } else { z };
                let idx_lo = octant_index(ox, 0, oz);
                let idx_hi = octant_index(ox, 1, oz);
                let (lo, hi) = out.split_at_mut(half as usize);
                self.flatten_column_rec(children[idx_lo], level - 1, lx, lz, lo);
                self.flatten_column_rec(children[idx_hi], level - 1, lx, lz, hi);
            }
        }
    }

    /// Splice a column's cells into the tree along (x, z), returning a new
    /// root. `column.len()` must equal `1 << root.level()`.
    ///
    /// Mirrors `flatten_column_into`'s topology walk; rebuilds interior nodes
    /// along the column path and reuses hash-cons for off-path subtrees.
    /// Intermediate `Leaf(state)` nodes are expanded into 8 child `Leaf(state)`
    /// slots before descending (same pattern as `set_cell_at`). On ascent,
    /// canonicalization back to compressed `Leaf(s)` form fires **only when
    /// the input at that level was itself a `Leaf(s)`** that got expanded on
    /// descent — never opportunistically on a pre-existing `Interior` whose
    /// children happen to be uniform.
    ///
    /// The "only when input was a `Leaf(s)`" clause is load-bearing, not
    /// incidental. Heterogeneous trees in this codebase can carry
    /// non-canonical uniform `Interior` nodes (siblings that were never
    /// expanded from a `Leaf`). Opportunistically canonicalizing any
    /// uniform-`Interior` on ascent would silently rewrite those siblings
    /// into `Leaf`, breaking identity-splice `NodeId` equality and
    /// regressing the structural-sharing drift that fix
    /// `hash-thing-gtua` resolved — see the `(gtua)` references on the
    /// expand / ascent sites inside `splice_column_rec`. Any refactor
    /// tempted to "simplify" the ascent into an unconditional
    /// canonicalize-when-uniform MUST preserve the same identity invariant
    /// as the regression test `splice_column_identity_through_raw_leaf_no_node_growth`.
    ///
    /// **Identity guarantee**: if the column is identical to the one
    /// already present, the returned root NodeId equals the input root via
    /// hash-cons, including when a raw `Leaf(state)` sits along the column
    /// path — the expand-then-narrowly-canonicalize round-trip preserves
    /// identity. See tests `splice_column_identity_returns_same_nodeid`
    /// and `splice_column_identity_through_raw_leaf_child_preserves_nodeid`.
    pub fn splice_column(&mut self, root: NodeId, x: u64, z: u64, column: &[CellState]) -> NodeId {
        let level = self.get(root).level();
        let side = 1u64 << level;
        assert!(
            x < side && z < side,
            "splice_column: ({x}, {z}) out of bounds for side {side}",
        );
        assert_eq!(
            column.len() as u64,
            side,
            "splice_column: column len {} != side {}",
            column.len(),
            side,
        );
        self.splice_column_rec(root, level, x, z, column)
    }

    fn splice_column_rec(
        &mut self,
        node: NodeId,
        level: u32,
        x: u64,
        z: u64,
        column: &[CellState],
    ) -> NodeId {
        debug_assert_eq!(column.len() as u64, 1u64 << level);
        if level == 0 {
            return self.leaf(column[0]);
        }
        // Expand intermediate Leaf(s) to [Leaf(s); 8] — each child is
        // itself a compressed uniform-s subtree at the level below. Keeping
        // the compressed form on untouched siblings is what lets the
        // rebuild canonicalize back to Leaf(s) and round-trip to the input
        // NodeId on identity splice. (gtua)
        let (children, was_leaf_state) = match *self.get(node) {
            Node::Leaf(s) => {
                let child = self.leaf(s);
                ([child; 8], Some(s))
            }
            Node::Interior { children, .. } => (children, None),
        };
        let half = 1u64 << (level - 1);
        let ox = if x >= half { 1u32 } else { 0 };
        let oz = if z >= half { 1u32 } else { 0 };
        let lx = if ox == 1 { x - half } else { x };
        let lz = if oz == 1 { z - half } else { z };
        let idx_lo = octant_index(ox, 0, oz);
        let idx_hi = octant_index(ox, 1, oz);
        let (lo, hi) = column.split_at(half as usize);
        let new_lo = self.splice_column_rec(children[idx_lo], level - 1, lx, lz, lo);
        let new_hi = self.splice_column_rec(children[idx_hi], level - 1, lx, lz, hi);
        let mut new_children = children;
        new_children[idx_lo] = new_lo;
        new_children[idx_hi] = new_hi;
        // Canonicalize back to Leaf form *only* when the input was itself a
        // Leaf(s) expanded above: the untouched siblings are guaranteed to
        // be leaf(s), and if the recursions also returned leaf(s), the
        // whole subtree is uniform s and collapses to Leaf(s). Skipping
        // canonicalization on the Interior input case (no opportunistic
        // canonicalization of pre-existing non-canonical uniform interiors)
        // preserves NodeId identity for heterogeneous trees: if we
        // opportunistically collapsed `Interior(L, [leaf(s); 8])` to Leaf(s)
        // on ascent, an identity splice through uniform(L, s) would return
        // a different NodeId than the input. (gtua)
        if let Some(s) = was_leaf_state {
            let target = self.leaf(s);
            if new_children.iter().all(|&c| c == target) {
                return target;
            }
        }
        self.interior(level, new_children)
    }

    // ---- Extraction primitives (hash-thing-6gf.6) ----

    /// Extract the 26 Moore neighbors of a cell, wrapping toroidally.
    ///
    /// Returns `[Cell; 26]` in the same order as `sim::world::get_neighbors`:
    /// all (dx, dy, dz) offsets in {-1, 0, +1}³ excluding (0,0,0), iterated
    /// z-outer / y-middle / x-inner, skipping the center.
    ///
    /// Coordinates wrap via `rem_euclid(side)` so this is a torus — matching
    /// the CaRule boundary semantics established in the brute-force stepper.
    ///
    /// **Performance note:** calls `get_cell` 26 times, each walking the tree
    /// from root. For bulk extraction over an entire grid this is worse than
    /// `flatten()` + grid indexing. The intended use is the recursive Hashlife
    /// stepper's base case, where the neighborhood is extracted from a small
    /// (level-2 or level-3) subtree with context, not the full world root.
    pub fn extract_neighborhood(
        &self,
        root: NodeId,
        side: i64,
        x: i64,
        y: i64,
        z: i64,
    ) -> [Cell; 26] {
        let mut neighbors = [Cell::EMPTY; 26];
        let mut idx = 0;
        for dz in [-1i64, 0, 1] {
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let nx = (x + dx).rem_euclid(side) as u64;
                    let ny = (y + dy).rem_euclid(side) as u64;
                    let nz = (z + dz).rem_euclid(side) as u64;
                    neighbors[idx] = Cell::from_raw(self.get_cell(root, nx, ny, nz));
                    idx += 1;
                }
            }
        }
        neighbors
    }

    /// Extract an axis-aligned 2×2×2 block starting at `(bx, by, bz)`.
    ///
    /// Returns `[Cell; 8]` indexed by `block_index(dx, dy, dz)` — the same
    /// convention as `sim::rule::block_index`. Coordinates are absolute
    /// world-space; no wrapping (cells outside the tree return empty via
    /// `get_cell`'s OOB contract).
    ///
    /// Useful for Margolus-partition extraction in the recursive stepper.
    pub fn extract_block_2x2x2(&self, root: NodeId, bx: u64, by: u64, bz: u64) -> [Cell; 8] {
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2u64 {
            for dy in 0..2u64 {
                for dx in 0..2u64 {
                    let idx = super::node::octant_index(dx as u32, dy as u32, dz as u32);
                    block[idx] = Cell::from_raw(self.get_cell(root, bx + dx, by + dy, bz + dz));
                }
            }
        }
        block
    }

    /// Flatten an arbitrary axis-aligned sub-box of the octree into a flat
    /// 3D array.
    ///
    /// Returns a `Vec<CellState>` of size `w * h * d`, indexed as
    /// `[lx + ly * w + lz * w * h]` where `(lx, ly, lz)` are local
    /// coordinates within the box.
    ///
    /// Coordinates outside the tree silently return 0 (via `get_cell`'s OOB
    /// contract). This lets callers request a region that extends past the
    /// tree boundary to capture border context for the recursive stepper.
    ///
    /// **Performance note:** calls `get_cell` once per voxel. For large
    /// regions, `flatten()` + sub-indexing is faster. This is designed for
    /// small regions (e.g. a level-2 subtree + 1-cell border = 6×6×6 = 216
    /// cells) where the setup cost of a full flatten dominates.
    pub fn flatten_region(
        &self,
        root: NodeId,
        origin: [u64; 3],
        extent: [usize; 3],
    ) -> Vec<CellState> {
        let [w, h, d] = extent;
        let [x0, y0, z0] = origin;
        let mut grid = vec![0 as CellState; w * h * d];
        for lz in 0..d {
            for ly in 0..h {
                for lx in 0..w {
                    grid[lx + ly * w + lz * w * h] =
                        self.get_cell(root, x0 + lx as u64, y0 + ly as u64, z0 + lz as u64);
                }
            }
        }
        grid
    }

    /// Stats for debugging.
    pub fn stats(&self) -> usize {
        self.nodes.len()
    }

    /// Number of unique nodes in the store. Alias for `stats()` with a
    /// clearer name; use this for capacity checks and diagnostics.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total unique reachable nodes from `root`, counting each interior /
    /// leaf NodeId once regardless of how many times it appears.
    ///
    /// This is the global reachable-node count used by the cswp memory-LOD
    /// measurement (`hash-thing-cswp.8.1`); it is the denominator against
    /// which per-chunk numbers are compared (see
    /// [`Self::reachable_nodes_per_chunk`]).
    pub fn reachable_node_count(&self, root: NodeId) -> u64 {
        let mut visited: FxHashSet<NodeId> = FxHashSet::default();
        self.collect_reachable(root, &mut visited);
        visited.len() as u64
    }

    /// Per-chunk reachable-node distribution.
    ///
    /// For a tree with `root_level >= chunk_level > 0`, walks each
    /// level-`chunk_level` subtree once and counts the unique NodeIds
    /// reachable from that chunk's root (the chunk root itself, all of its
    /// interiors, and the leaf at the bottom of every reachable path).
    ///
    /// Note: an "empty" chunk does NOT have count == 1, because
    /// [`Self::empty`] / [`Self::uniform`] build a fresh `Interior` per
    /// level rather than collapsing to a single canonical empty
    /// `NodeId`. An empty chunk at `chunk_level == k` therefore reaches
    /// exactly `k + 1` unique NodeIds (the empty-`Interior` chain at
    /// levels `k`, `k-1`, ..., `1`, plus the canonical empty leaf
    /// `NodeId::EMPTY` at level 0). Callers wanting an "empty?" check
    /// from this output should compare against `k + 1`, or use
    /// [`Self::population`] on the chunk root directly. A future fast
    /// path that short-circuits at `Interior { population: 0, .. }` would
    /// shrink this number to 1, but is intentionally not part of cswp.8.1
    /// (cswp.8 follow-up).
    ///
    /// Returns one entry per chunk in **row-major (cz outer, cy middle,
    /// cx inner)** order, i.e. index `(cz * chunks_per_axis + cy) *
    /// chunks_per_axis + cx`, where `chunks_per_axis = 1 << (root_level -
    /// chunk_level)`.
    ///
    /// ## What this measurement is (and is not)
    ///
    /// Each chunk's count uses a **chunk-local** visited-set. Hash-cons
    /// sharing across chunks is not deduplicated — a node referenced by
    /// `N` chunks is charged once to each, so `sum(per_chunk) >= global`.
    /// The gap (`sum - global`) is the cross-chunk-shared multiplicity.
    ///
    /// Concretely this gives the **chunk-local reachable upper bound with
    /// all cross-chunk sharing charged in full to every chunk**. It is the
    /// right input for "what is the standalone cost of shipping/serializing
    /// this chunk?" It is *not* a "reclaimable bytes if I evict this
    /// chunk" estimate, since shared nodes survive in the surviving
    /// chunks. Callers picking chunk size from this output must read the
    /// `share×` ratio (`sum_per_chunk / global`) alongside per-chunk
    /// distribution; if sharing is high, an exclusive-count follow-up
    /// measurement may be needed.
    ///
    /// ## Cost
    ///
    /// O(reachable per chunk) time per chunk, O(largest chunk visited-set)
    /// memory at any moment. Uniform regions above chunk_level (an
    /// Interior child slot pointing at a raw `Leaf(s)` — a structural
    /// state arising in practice via [`Self::splice_column`]'s
    /// canonicalize path; trees built only via [`Self::empty`] are full
    /// Interior chains and don't hit this arm) short-circuit by emitting
    /// one count=1 entry per nested chunk without descending further.
    /// The walker recursion depth is bounded by `root_level - chunk_level`
    /// (chunk dispatch) plus `chunk_level` (per-chunk reachability),
    /// i.e. `root_level` total — well under any reasonable Rust stack at
    /// the project's max level (12 → 4096³).
    ///
    /// ## Panics
    ///
    /// - if `chunk_level == 0` (cell-level walks are O(side³); use
    ///   [`Self::reachable_node_count`] instead).
    /// - if `chunk_level > root_level` (no chunks fit in the tree).
    pub fn reachable_nodes_per_chunk(&self, root: NodeId, chunk_level: u32) -> ChunkReachable {
        let (per_chunk, _global) = self.reachable_summary(root, chunk_level);
        per_chunk
    }

    /// Combined per-chunk + global reachable counts in a single tree walk.
    ///
    /// Returns `(per_chunk, global)` where `per_chunk` follows the same
    /// shape and ordering as [`Self::reachable_nodes_per_chunk`] and
    /// `global` is the unique-reachable-from-`root` count (also returned
    /// stand-alone by [`Self::reachable_node_count`]).
    ///
    /// Prefer this over calling the two methods separately when you need
    /// both, since each single-method call walks the full DAG.
    pub fn reachable_summary(&self, root: NodeId, chunk_level: u32) -> (ChunkReachable, u64) {
        let root_level = self.get(root).level();
        assert!(
            chunk_level > 0,
            "reachable_nodes_per_chunk: chunk_level=0 is O(side^3) cells and not the intended use; \
             use reachable_node_count for cell-level walks",
        );
        assert!(
            chunk_level <= root_level,
            "reachable_nodes_per_chunk: chunk_level {chunk_level} exceeds root_level {root_level}",
        );
        let chunks_per_axis = 1u64 << (root_level - chunk_level);
        let total = (chunks_per_axis as usize)
            .checked_pow(3)
            .expect("reachable_nodes_per_chunk: chunks_per_axis^3 overflowed usize");
        let mut state = ChunkWalk {
            chunk_level,
            out: Vec::with_capacity(total),
            global: FxHashSet::default(),
        };
        self.walk_chunks_rec(root, root_level, (0, 0, 0), &mut state);
        debug_assert_eq!(state.out.len(), total);
        let global = state.global.len() as u64;
        (state.out, global)
    }

    /// Walker dispatch above `chunk_level`; emits one entry per chunk in
    /// row-major (cz outer, cy middle, cx inner) order.
    fn walk_chunks_rec(
        &self,
        node: NodeId,
        level: u32,
        (cx, cy, cz): (u64, u64, u64),
        state: &mut ChunkWalk,
    ) {
        if level == state.chunk_level {
            let mut local: FxHashSet<NodeId> = FxHashSet::default();
            self.collect_reachable_pair(node, &mut local, &mut state.global);
            state.out.push(((cx, cy, cz), local.len() as u64));
            return;
        }
        match *self.get(node) {
            Node::Leaf(_) => {
                // Uniform region above chunk_level: every nested chunk is
                // the same leaf. Emit one entry per nested chunk in
                // row-major order with count = 1 (the leaf itself, which
                // is added to the global set on this visit; subsequent
                // visits are no-ops by FxHashSet semantics).
                state.global.insert(node);
                let chunks_per_axis_below = 1u64 << (level - state.chunk_level);
                for dz in 0..chunks_per_axis_below {
                    for dy in 0..chunks_per_axis_below {
                        for dx in 0..chunks_per_axis_below {
                            state.out.push((
                                (
                                    cx * chunks_per_axis_below + dx,
                                    cy * chunks_per_axis_below + dy,
                                    cz * chunks_per_axis_below + dz,
                                ),
                                1,
                            ));
                        }
                    }
                }
            }
            Node::Interior { children, .. } => {
                state.global.insert(node);
                let chunks_per_axis_below_child = 1u64 << (level - 1 - state.chunk_level);
                for (oct, &child) in children.iter().enumerate() {
                    let (ox, oy, oz) = super::node::octant_coords(oct);
                    self.walk_chunks_rec(
                        child,
                        level - 1,
                        (
                            cx + ox as u64 * chunks_per_axis_below_child,
                            cy + oy as u64 * chunks_per_axis_below_child,
                            cz + oz as u64 * chunks_per_axis_below_child,
                        ),
                        state,
                    );
                }
            }
        }
    }

    /// Reachability traversal that updates a single visited-set.
    fn collect_reachable(&self, node: NodeId, visited: &mut FxHashSet<NodeId>) {
        if !visited.insert(node) {
            return;
        }
        if let Node::Interior { children, .. } = self.get(node) {
            let children = *children;
            for c in children {
                self.collect_reachable(c, visited);
            }
        }
    }

    /// Reachability traversal that updates two visited-sets in lockstep
    /// (chunk-local + global). Each NodeId is recorded in both sets the
    /// first time it is seen on this chunk; the local set drives the
    /// per-chunk count, the global set accumulates across the whole walk.
    fn collect_reachable_pair(
        &self,
        node: NodeId,
        local: &mut FxHashSet<NodeId>,
        global: &mut FxHashSet<NodeId>,
    ) {
        if !local.insert(node) {
            return;
        }
        global.insert(node);
        if let Node::Interior { children, .. } = self.get(node) {
            let children = *children;
            for c in children {
                self.collect_reachable_pair(c, local, global);
            }
        }
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
    pub fn compacted(&self, root: NodeId) -> (NodeStore, NodeId) {
        let (store, root, _remap) = self.compacted_with_remap(root);
        (store, root)
    }

    /// Like [`compacted`], but also returns the old→new NodeId remap table.
    /// Callers that maintain caches keyed on NodeId can remap their keys
    /// instead of invalidating the entire cache.
    pub fn compacted_with_remap(
        &self,
        root: NodeId,
    ) -> (NodeStore, NodeId, FxHashMap<NodeId, NodeId>) {
        self.compacted_with_remap_keeping(root, &[])
    }

    /// Compact, keeping alive all nodes referenced by `extra_roots` in
    /// addition to those reachable from `root`. Used to preserve hashlife
    /// cache intermediate nodes through GC (m1f.15.4).
    pub fn compacted_with_remap_keeping(
        &self,
        root: NodeId,
        extra_roots: &[NodeId],
    ) -> (NodeStore, NodeId, FxHashMap<NodeId, NodeId>) {
        let mut dst = NodeStore::new();
        let mut remap: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        // Pre-seed EMPTY as a perf short-circuit. `dst.leaf(0)` already
        // dedups to `NodeId::EMPTY` via hash-cons, but this skips the
        // recursive walk for empty subtrees.
        remap.insert(NodeId::EMPTY, NodeId::EMPTY);
        let new_root = clone_reachable(self, &mut dst, &mut remap, root);
        for &extra in extra_roots {
            clone_reachable(self, &mut dst, &mut remap, extra);
        }
        (dst, new_root, remap)
    }

    /// LOD-collapse primitive: collapse the bottom `target_lod` octree levels
    /// of the subtree at `root` into representative-material leaves.
    ///
    /// Convention follows `docs/perf/cswp-lod.md` §2.1:
    /// - `target_lod = 0` is identity (returns `root` unchanged).
    /// - `target_lod = root_level` is full collapse (returns a single
    ///   `Leaf` carrying the whole tree's representative state). For a
    ///   `Leaf` root (level 0), this and `target_lod = 0` coincide.
    /// - In between, every contextual-level-`target_lod` subtree becomes a
    ///   `Leaf`; Interior structure above is preserved and re-interned.
    ///
    /// Panics if `target_lod > root_level` (caller must clamp).
    ///
    /// The representative-state rule mirrors the SVDAG builder
    /// (`crates/ht-render/src/svdag.rs`): largest-populated child wins; ties
    /// among Interior children stay with the earlier octant; among Leaf-only
    /// children, the first non-empty leaf wins (rep_pop seeded to 1). This
    /// guarantees bit-exact agreement with SVDAG's `rep_mat` selection — the
    /// full 16-bit `CellState` is preserved, not just the material field.
    ///
    /// **Population semantics:** rebuilt Interior nodes recompute population
    /// from their (now-Leaf) children, so the collapsed root's `population`
    /// reports the count of *non-empty representative cells* (≤
    /// `8^(root_level - target_lod)`), not the original cell count. Callers
    /// that need the pre-collapse population must read it from the input
    /// root before collapsing.
    ///
    /// **Memory:** each call interns up to `(root_level - target_lod) + 1`
    /// new nodes. The `NodeStore` grows monotonically; repeated per-frame
    /// collapses against a long-lived store will accumulate ghost interior
    /// chains. cswp.8.3 should either run against a transient/scratch
    /// `NodeStore` or compact via [`compact_reachable_from`] periodically.
    pub fn lod_collapse(&mut self, root: NodeId, target_lod: u32) -> NodeId {
        let root_level = self.get(root).level();
        assert!(
            target_lod <= root_level,
            "lod_collapse: target_lod {target_lod} > root_level {root_level}",
        );
        if target_lod == 0 {
            return root;
        }
        // Memo cache for `representative_state` keyed by NodeId. Hash-cons
        // guarantees same NodeId ⇒ same subtree ⇒ same representative state,
        // so the cache is sound for the lifetime of this call. Mutations via
        // `interior(...)` during the descent only mint *new* NodeIds above
        // `target_lod`; the memo is only ever queried for boundary nodes
        // (level == target_lod) that already existed at entry, so existing
        // cache entries can never be invalidated.
        let mut memo: FxHashMap<NodeId, CellState> = FxHashMap::default();
        self.lod_collapse_rec(root, root_level, target_lod, &mut memo)
    }

    /// Recursive worker for [`lod_collapse`]. `level` is the contextual level
    /// of `node` as seen from its parent — when an intermediate subtree is
    /// represented by a raw `Leaf(s)` (uniform-region compression), the
    /// caller's `level` carries the true level rather than `Node::level()`.
    fn lod_collapse_rec(
        &mut self,
        node: NodeId,
        level: u32,
        target_lod: u32,
        memo: &mut FxHashMap<NodeId, CellState>,
    ) -> NodeId {
        if level == target_lod {
            let rep = self.representative_state_memo(node, memo);
            return self.leaf(rep);
        }
        // level > target_lod: descend.
        match *self.get(node) {
            // A raw Leaf at level > target_lod stands for a uniform region;
            // collapsing a uniform region cannot change it, so the input
            // NodeId is already the answer.
            Node::Leaf(_) => node,
            Node::Interior { children, .. } => {
                let mut new_children = children;
                for i in 0..8 {
                    new_children[i] =
                        self.lod_collapse_rec(children[i], level - 1, target_lod, memo);
                }
                self.interior(level, new_children)
            }
        }
    }

    /// Memoized `representative_state`: returns the cached `CellState` for
    /// `node` if present, otherwise computes it via the SVDAG-builder rule
    /// (largest-populated child wins; among Leaf-only children the first
    /// non-empty leaf wins) and inserts the result. Pure read on the store
    /// itself; mutates only the supplied `memo`.
    fn representative_state_memo(
        &self,
        node: NodeId,
        memo: &mut FxHashMap<NodeId, CellState>,
    ) -> CellState {
        if let Some(&cached) = memo.get(&node) {
            return cached;
        }
        let rep = match *self.get(node) {
            Node::Leaf(s) => s,
            Node::Interior { children, .. } => {
                let mut rep_state: CellState = 0;
                let mut rep_pop: u64 = 0;
                for &child in children.iter() {
                    match *self.get(child) {
                        Node::Leaf(s) => {
                            // First non-empty Leaf seeds rep_pop=1 — mirrors
                            // svdag.rs `rep_mat == 0 || rep_pop == 0` guard.
                            if s != 0 && (rep_state == 0 || rep_pop == 0) {
                                rep_state = s;
                                rep_pop = 1;
                            }
                        }
                        Node::Interior { population, .. } => {
                            // Strict inequality: equal-population ties stay
                            // with the earlier octant — matches SVDAG.
                            if population > 0 && population > rep_pop {
                                rep_state = self.representative_state_memo(child, memo);
                                rep_pop = population;
                            }
                        }
                    }
                }
                rep_state
            }
        };
        memo.insert(node, rep);
        rep
    }

    /// Test-only no-cache wrapper for [`representative_state_memo`]. Builds a
    /// fresh memo per call; useful for assertions in unit tests where the
    /// caller doesn't want to thread a cache.
    #[cfg(test)]
    fn representative_state(&self, node: NodeId) -> CellState {
        let mut memo: FxHashMap<NodeId, CellState> = FxHashMap::default();
        self.representative_state_memo(node, &mut memo)
    }

    /// Per-chunk variant of [`lod_collapse`]. Rewrites only the subtree at
    /// chunk `(chunk_x, chunk_y, chunk_z)` (where chunks are
    /// `2^chunk_level`-cube cells of the `2^root_level` root grid),
    /// collapsing it with the given `target_lod`. Other chunks' subtrees are
    /// preserved by NodeId — hash-consing makes the rebuilt root share
    /// every untouched branch with the input root.
    ///
    /// Panics if `chunk_level > root_level`, `target_lod > chunk_level`, or
    /// any chunk coordinate is out of bounds for the
    /// `(2^(root_level - chunk_level))^3` chunk grid.
    ///
    /// **Memory:** like [`lod_collapse`], each call interns a fresh
    /// root-to-chunk Interior chain plus the collapsed leaf. Calling this
    /// once per chunk per frame against a long-lived store will accumulate
    /// ghost chains; cswp.8.3 should batch or compact accordingly.
    pub fn lod_collapse_chunk(
        &mut self,
        root: NodeId,
        chunk_x: u64,
        chunk_y: u64,
        chunk_z: u64,
        chunk_level: u32,
        target_lod: u32,
    ) -> NodeId {
        let root_level = self.get(root).level();
        assert!(
            chunk_level <= root_level,
            "lod_collapse_chunk: chunk_level {chunk_level} > root_level {root_level}",
        );
        assert!(
            target_lod <= chunk_level,
            "lod_collapse_chunk: target_lod {target_lod} > chunk_level {chunk_level}",
        );
        let chunks_per_axis = 1u64 << (root_level - chunk_level);
        assert!(
            chunk_x < chunks_per_axis
                && chunk_y < chunks_per_axis
                && chunk_z < chunks_per_axis,
            "lod_collapse_chunk: coord ({chunk_x},{chunk_y},{chunk_z}) out of bounds for {chunks_per_axis}³ grid",
        );
        self.lod_collapse_chunk_rec(
            root,
            root_level,
            (chunk_x, chunk_y, chunk_z),
            chunk_level,
            target_lod,
        )
    }

    /// Recursive worker for [`lod_collapse_chunk`]. Descends the unique path
    /// to the target chunk, calling `lod_collapse` once at the chunk
    /// boundary. The Leaf early-return preserves uniform regions: if the
    /// path passes through a raw `Leaf(s)` above `chunk_level` (the
    /// uniform-compression case), the chunk is already represented by that
    /// leaf and collapsing it cannot change anything.
    fn lod_collapse_chunk_rec(
        &mut self,
        node: NodeId,
        level: u32,
        (cx, cy, cz): (u64, u64, u64),
        chunk_level: u32,
        target_lod: u32,
    ) -> NodeId {
        if level == chunk_level {
            return self.lod_collapse(node, target_lod);
        }
        match *self.get(node) {
            Node::Leaf(_) => node,
            Node::Interior { children, .. } => {
                // At descent step from `level` to `level-1`, the chunk grid
                // halves. The target chunk's high bit at this depth picks
                // the octant; the remaining bits address the sub-chunk
                // within that octant.
                let stride = 1u64 << (level - 1 - chunk_level);
                let ox = (cx / stride) & 1;
                let oy = (cy / stride) & 1;
                let oz = (cz / stride) & 1;
                let oct = octant_index(ox as u32, oy as u32, oz as u32);
                let local = (cx - ox * stride, cy - oy * stride, cz - oz * stride);
                let mut new_children = children;
                new_children[oct] = self.lod_collapse_chunk_rec(
                    children[oct],
                    level - 1,
                    local,
                    chunk_level,
                    target_lod,
                );
                self.interior(level, new_children)
            }
        }
    }
}

/// Clone the subtree rooted at `old` from `src` into `dst`, populating
/// `remap` with old-id → new-id entries.
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
/// ## Iterative post-order (hash-thing-8m7)
///
/// Originally recursive. At level 6 (current root) that's 7 stack
/// frames; at level 20+ or once chunk-ingest can graft arbitrary
/// subtrees under a high root, the native stack is not a safe place to
/// recurse. Rewritten as an explicit two-phase `Visit`/`Emit` stack:
///
/// - `Visit(old)` dispatches on the node. Leaves intern immediately;
///   interiors push an `Emit(old)` marker, then push `Visit(child)` for
///   each child (in reverse so index 0 pops first — not strictly
///   required for correctness, but keeps the walk left-to-right for
///   anyone reading a debugger).
/// - `Emit(old)` runs after every child has been interned into `dst`
///   and remapped. It reads `children` from `src`, translates them
///   through `remap`, and calls `dst.interior(level, new_children)` —
///   the same single-pass build the recursive version did, just
///   scheduled by the explicit stack instead of the native one.
///
/// Hash-cons sharing (same `NodeId` appearing as multiple children) is
/// still handled at `Visit` pop time: if `remap` already has the id,
/// the visit is a no-op. That keeps the work bounded by the number of
/// *distinct* reachable nodes, not the number of child edges.
fn clone_reachable(
    src: &NodeStore,
    dst: &mut NodeStore,
    remap: &mut FxHashMap<NodeId, NodeId>,
    root: NodeId,
) -> NodeId {
    enum Frame {
        Visit(NodeId),
        Emit(NodeId),
    }

    let mut stack: Vec<Frame> = Vec::new();
    stack.push(Frame::Visit(root));

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Visit(old) => {
                if remap.contains_key(&old) {
                    continue;
                }
                match src.get(old) {
                    Node::Leaf(state) => {
                        let new_id = dst.leaf(*state);
                        remap.insert(old, new_id);
                    }
                    Node::Interior { children, .. } => {
                        let children = *children;
                        stack.push(Frame::Emit(old));
                        // Push children in reverse so index 0 pops (and
                        // is visited) first. Not load-bearing — the
                        // remap/hash-cons contract is index-invariant —
                        // but it makes stack traces read left-to-right.
                        for &child in children.iter().rev() {
                            if !remap.contains_key(&child) {
                                stack.push(Frame::Visit(child));
                            }
                        }
                    }
                }
            }
            Frame::Emit(old) => {
                // By the time we pop Emit, every child has been
                // visited-and-interned (or was already in remap). Read
                // the original children from `src`, translate through
                // `remap`, and intern the parent.
                let (level, children) = match src.get(old) {
                    Node::Interior {
                        level, children, ..
                    } => (*level, *children),
                    Node::Leaf(_) => {
                        // Unreachable: we only push Emit from the
                        // Interior arm above.
                        unreachable!("Emit frame for a leaf node")
                    }
                };
                let mut new_children = [NodeId::EMPTY; 8];
                for (i, &child) in children.iter().enumerate() {
                    new_children[i] = remap[&child];
                }
                let new_id = dst.interior(level, new_children);
                remap.insert(old, new_id);
            }
        }
    }

    remap[&root]
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
    //!
    //! Cell encoding note (hash-thing-1v0.1): `CellState` is a packed
    //! `u16 = material(10 bits) | metadata(6 bits)`, and raw values in
    //! `1..=63` are forbidden (they decode to material 0 with nonzero
    //! metadata, which would silently alias `Cell::EMPTY`). Tests that want
    //! to distinguish "material N" fixture values must use the `mat()`
    //! helper below, which produces a valid raw encoding (`N << 6`).
    use super::super::node::Cell;
    use super::*;

    /// Test helper: encode a material id as a raw `CellState` with metadata=0.
    /// Rejects `material == 0` (empty) loudly so tests can't silently alias.
    fn mat(material: u16) -> CellState {
        Cell::pack(material, 0).raw()
    }

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
        let a = store.leaf(mat(7));
        let b = store.leaf(mat(7));
        assert_eq!(a, b);
        // And a different state gets a different id.
        let c = store.leaf(mat(8));
        assert_ne!(a, c);
    }

    /// Interning the same interior node twice yields the same id.
    /// This is the core hash-consing invariant.
    #[test]
    fn intern_dedup_interior() {
        let mut store = NodeStore::new();
        let l0 = store.leaf(0);
        let l1 = store.leaf(mat(1));
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
        let l1 = store.leaf(mat(1));
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
        root = store.set_cell(root, 0, 0, 0, mat(5));
        root = store.set_cell(root, 7, 7, 7, mat(9));
        root = store.set_cell(root, 3, 4, 5, mat(2));
        assert_eq!(store.get_cell(root, 0, 0, 0), mat(5));
        assert_eq!(store.get_cell(root, 7, 7, 7), mat(9));
        assert_eq!(store.get_cell(root, 3, 4, 5), mat(2));
        // Untouched cells stay empty.
        assert_eq!(store.get_cell(root, 1, 1, 1), 0);
    }

    /// Same test at a deeper level — the recursion has to keep halving
    /// correctly all the way down.
    #[test]
    fn set_get_roundtrip_level_6() {
        let mut store = NodeStore::new();
        let mut root = store.empty(6); // 64^3
        let probes: &[(u64, u64, u64, CellState)] = &[
            (0, 0, 0, mat(1)),
            (63, 63, 63, mat(2)),
            (31, 31, 31, mat(3)),
            (32, 32, 32, mat(4)),
            (10, 20, 30, mat(5)),
            (50, 5, 25, mat(6)),
        ];
        for &(x, y, z, s) in probes {
            root = store.set_cell(root, x, y, z, s);
        }
        for &(x, y, z, s) in probes {
            assert_eq!(store.get_cell(root, x, y, z), s, "cell ({x},{y},{z})");
        }
    }

    /// Regression for hash-thing-nch I5: non-power-of-two `side` must
    /// release-panic at the API boundary. Pre-fix, `(side as f64).log2()
    /// as u32` silently truncated (`side=48` → `level=5`), so
    /// `from_flat_recursive` built a 32³ tree over 48³ data and read past
    /// the source buffer on the upper 16 cells of every axis.
    #[test]
    #[should_panic(expected = "power of two")]
    fn from_flat_rejects_non_pow2() {
        let mut store = NodeStore::new();
        let side = 48usize;
        let grid = vec![0 as CellState; side * side * side];
        let _ = store.from_flat(&grid, side);
    }

    /// Regression for hash-thing-nch I5: each power-of-two `side` must
    /// resolve to the correct octree level via `trailing_zeros()`. Spot
    /// every level from 0 through 8 so a future refactor that re-introduces
    /// the `log2` truncation or shifts the level by one trips on the
    /// smallest possible grid.
    #[test]
    fn from_flat_pow2_levels_roundtrip() {
        for level in 0u32..=6 {
            let side = 1usize << level;
            let mut grid = vec![0 as CellState; side * side * side];
            // Drop a single live cell at the far corner so level-0
            // (single-cell) and level-6 (64³) both exercise a non-empty
            // build.
            let last = side - 1;
            grid[last + last * side + last * side * side] = mat(1);
            let mut store = NodeStore::new();
            let root = store.from_flat(&grid, side);
            assert_eq!(
                store.get(root).level(),
                level,
                "from_flat(side={side}) should build a level-{level} tree",
            );
            // Roundtrip sanity: flatten it back and check the single live
            // cell survives.
            let out = store.flatten(root, side);
            assert_eq!(out[last + last * side + last * side * side], mat(1));
        }
    }

    /// `from_flat` then `flatten` returns the original bytes.
    #[test]
    fn from_flat_flatten_roundtrip() {
        let side = 8usize;
        let mut grid = vec![0 as CellState; side * side * side];
        // Put a recognizable pattern in: each cell gets material id in 0..=7,
        // encoded as `material << METADATA_BITS` so every nonzero value is a
        // valid Cell (material != 0, metadata == 0). Material 0 stays 0 (empty).
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let material = ((x ^ y ^ z) & 0x7) as u16;
                    grid[x + y * side + z * side * side] =
                        if material == 0 { 0 } else { mat(material) };
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
        let grid = vec![0 as CellState; side * side * side];
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
        let mut grid = vec![0 as CellState; side * side * side];
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
            grid[x + y * side + z * side * side] = mat(1);
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
        let mut grid = vec![0 as CellState; side * side * side];
        grid[0] = mat(1);
        grid[side - 1] = mat(2);
        grid[(side - 1) * side + 4] = mat(3);
        let mut store = NodeStore::new();
        let a = store.from_flat(&grid, side);
        let b = store.from_flat(&grid, side);
        assert_eq!(a, b);
        // And a grid that differs by one cell must NOT share.
        let mut grid2 = grid.clone();
        grid2[1] = mat(1);
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
        let grid = vec![0 as CellState; side * side * side];
        let mut store = NodeStore::new();
        let _root = store.from_flat(&grid, side);
        let nodes = store.stats();
        // 7 levels (leaf through level 6), one canonical node per level.
        assert_eq!(nodes, 7, "expected 7 nodes for an empty 64^3, got {nodes}");
    }

    /// `stats()` reports node count.
    #[test]
    fn stats_tracks_nodes() {
        let mut store = NodeStore::new();
        assert_eq!(store.stats(), 1); // just the empty leaf
        store.leaf(mat(1));
        store.leaf(mat(2));
        assert_eq!(store.stats(), 3);
    }

    /// Regression for hash-thing-7rq: when an Interior's child is a Leaf that
    /// stands for a uniform subtree at level > 0, set_cell must only rewrite
    /// the one target cell — not collapse the whole subtree to a single leaf.
    #[test]
    fn set_cell_preserves_siblings_when_child_is_leaf_at_intermediate_level() {
        let mut store = NodeStore::new();

        // Level-2 root (4x4x4). Octant 0 is a raw Leaf(mat(7)): by structural
        // convention a Leaf child of a level-2 Interior represents a uniform
        // 2x2x2 block. The other 7 octants are fully-expanded empty level-1
        // nodes so the bug has somewhere to hide.
        let leaf7 = store.leaf(mat(7));
        let empty_l1 = store.empty(1);
        let root = store.interior(
            2,
            [
                leaf7, empty_l1, empty_l1, empty_l1, empty_l1, empty_l1, empty_l1, empty_l1,
            ],
        );

        // Change one cell inside the compressed uniform block.
        let new_root = store.set_cell(root, 0, 0, 0, mat(9));

        // The written cell is material 9.
        assert_eq!(store.get_cell(new_root, 0, 0, 0), mat(9));
        // The other 7 cells that used to be covered by Leaf(mat(7)) are still mat(7).
        for &(x, y, z) in &[
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ] {
            assert_eq!(
                store.get_cell(new_root, x, y, z),
                mat(7),
                "cell ({x},{y},{z}) should survive set_cell on a sibling",
            );
        }
        // Cells in sibling octants are still empty.
        assert_eq!(store.get_cell(new_root, 2, 0, 0), 0);
        assert_eq!(store.get_cell(new_root, 0, 2, 0), 0);
        assert_eq!(store.get_cell(new_root, 3, 3, 3), 0);
    }

    /// The Leaf-at-intermediate expansion must nest correctly: deep writes
    /// through multiple compressed levels should touch exactly one cell.
    #[test]
    fn set_cell_expands_leaf_through_multiple_intermediate_levels() {
        let mut store = NodeStore::new();

        // Level-3 root (8x8x8) whose octant 0 is Leaf(mat(5)) — a uniform 4x4x4
        // block. set_cell must expand it twice on the way down.
        let leaf5 = store.leaf(mat(5));
        let empty_l2 = store.empty(2);
        let root = store.interior(
            3,
            [
                leaf5, empty_l2, empty_l2, empty_l2, empty_l2, empty_l2, empty_l2, empty_l2,
            ],
        );

        let new_root = store.set_cell(root, 2, 1, 3, mat(9));

        assert_eq!(store.get_cell(new_root, 2, 1, 3), mat(9));
        // Spot-check that the rest of the uniform block survived.
        for &(x, y, z) in &[(0, 0, 0), (3, 3, 3), (0, 3, 0), (2, 1, 2), (2, 0, 3)] {
            assert_eq!(
                store.get_cell(new_root, x, y, z),
                mat(5),
                "cell ({x},{y},{z}) should survive deep set_cell",
            );
        }
    }

    /// Sanity check: set_cell on a bare Leaf root (level 0) still works.
    #[test]
    fn set_cell_on_leaf_root_level_zero() {
        let mut store = NodeStore::new();
        let root = store.leaf(0);
        let new_root = store.set_cell(root, 0, 0, 0, mat(5));
        assert_eq!(store.get_cell(new_root, 0, 0, 0), mat(5));
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
        assert_eq!(
            compacted.stats(),
            1,
            "empty compaction should yield only the canonical empty leaf"
        );
    }

    /// T2: after one `set_cell`, compaction's node count matches a
    /// freshly-built one-cell world — NOT the pre-compaction size, because
    /// `set_cell` is persistent and the clone path's old nodes are dead.
    #[test]
    fn compact_after_set_cell_matches_fresh_one_cell_world() {
        // Build store A: empty(6), set one cell.
        let mut a = NodeStore::new();
        let mut root_a = a.empty(6);
        root_a = a.set_cell(root_a, 10, 20, 30, mat(7));
        let pre_nodes = a.stats();
        let (compacted_a, new_root_a) = a.compacted(root_a);

        // Build store B: a fresh one-cell world — empty(6), set the same cell.
        let mut b = NodeStore::new();
        let mut root_b = b.empty(6);
        root_b = b.set_cell(root_b, 10, 20, 30, mat(7));
        // Compact B too so both sides went through the same canonicalization.
        let (compacted_b, _new_root_b) = b.compacted(root_b);

        assert_eq!(
            compacted_a.stats(),
            compacted_b.stats(),
            "compacted one-cell world should match fresh one-cell world node count"
        );
        assert!(
            compacted_a.stats() < pre_nodes,
            "compaction should drop dead clone-path nodes ({} → {})",
            pre_nodes,
            compacted_a.stats()
        );
        // Cell round-trip.
        assert_eq!(compacted_a.get_cell(new_root_a, 10, 20, 30), mat(7));
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
        let mut grid = vec![0 as CellState; side * side * side];
        for i in 0..side {
            grid[i + i * side + i * side * side] = mat((i as u16 % 7) + 1);
        }
        let mut root = store.from_flat(&grid, side);
        // Churn: rebuild a handful of generations. We don't need real CA
        // semantics — just that the store sees repeated `from_flat` calls
        // with slightly different grids, leaking old subtrees.
        for gen in 0..5 {
            for (x, cell) in grid.iter_mut().enumerate().take(side) {
                *cell = mat(((gen + x) as u16 % 5) + 1);
            }
            root = store.from_flat(&grid, side);
        }
        let pre_nodes = store.stats();
        let (compacted, new_root) = store.compacted(root);
        assert!(
            compacted.stats() <= pre_nodes,
            "compaction should not grow the store ({} → {})",
            pre_nodes,
            compacted.stats()
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
        let mut grid = vec![0 as CellState; side * side * side];
        grid[0] = mat(5);
        grid[side * side * side - 1] = mat(9);
        grid[3 + 4 * side + 5 * side * side] = mat(2);

        let mut dirty = NodeStore::new();
        // Churn the store: build, mutate, build again — leaves dead nodes.
        let _waste1 = dirty.from_flat(&grid, side);
        let mut grid2 = grid.clone();
        grid2[1] = mat(3);
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
        let l1 = store.leaf(mat(1));
        let l2 = store.leaf(mat(2));
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
    /// distinct `CellState` values. `CellState` is now a packed 10-bit
    /// material id + 6-bit metadata `u16`; compaction must be
    /// payload-transparent, not just shape-transparent.
    #[test]
    fn compact_preserves_multiple_cell_states() {
        let side = 8usize;
        let mut grid = vec![0 as CellState; side * side * side];
        // Sprinkle a handful of distinct material ids. Each tuple is
        // (x, y, z, material_id); stored as `mat(material_id)` so the raw
        // cell word is a valid Cell.
        let cells: &[(usize, usize, usize, u16)] = &[
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
        for &(x, y, z, m) in cells {
            grid[x + y * side + z * side * side] = mat(m);
        }
        let mut store = NodeStore::new();
        // Churn so compaction has something to drop.
        let mut g2 = grid.clone();
        g2[0] = mat(99);
        let _waste = store.from_flat(&g2, side);
        let root = store.from_flat(&grid, side);
        let (compacted, new_root) = store.compacted(root);

        for &(x, y, z, m) in cells {
            assert_eq!(
                compacted.get_cell(new_root, x as u64, y as u64, z as u64),
                mat(m),
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
        let mut grid = vec![0 as CellState; side * side * side];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    // Material id 0..=16. 0 stays as raw empty; non-zero wraps
                    // through `mat()` to produce a valid Cell word.
                    let m = ((x * 7 + y * 11 + z * 13) % 17) as u16;
                    grid[x + y * side + z * side * side] = if m == 0 { 0 } else { mat(m) };
                }
            }
        }
        let mut dirty = NodeStore::new();
        // Churn.
        let mut g2 = grid.clone();
        g2[0] = mat(254);
        let _w1 = dirty.from_flat(&g2, side);
        let mut g3 = grid.clone();
        g3[1] = mat(253);
        let _w2 = dirty.from_flat(&g3, side);
        let root = dirty.from_flat(&grid, side);

        let (once, root_once) = dirty.compacted(root);
        let (twice, root_twice) = once.compacted(root_once);
        assert_eq!(
            once.stats(),
            twice.stats(),
            "compaction is not idempotent ({} → {})",
            once.stats(),
            twice.stats()
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
        let mut grid = vec![0 as CellState; side * side * side];
        for i in 0..side {
            grid[i + (i % 4) * side + (i % 3) * side * side] = mat(((i % 7) + 1) as u16);
        }
        let mut store = NodeStore::new();
        // Churn.
        let mut g2 = grid.clone();
        g2[0] = mat(77);
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
        root = store.set_cell(root, 3, 3, 3, mat(1));
        let (compacted, _new_root) = store.compacted(root);
        assert_eq!(NodeId::EMPTY, NodeId(0));
        assert!(matches!(compacted.get(NodeId::EMPTY), Node::Leaf(0)));
        assert_eq!(compacted.population(NodeId::EMPTY), 0);
    }

    /// T10 (hash-thing-8m7): deep-tree clone_reachable. The iterative
    /// `clone_reachable` drives the walk from an explicit heap stack;
    /// this test builds a level-30 tree with two distinct set cells and
    /// compacts it, exercising a full-depth chain of interior nodes plus
    /// a fork where the two cells diverge. Correctness bar is unchanged
    /// from the recursive version (set cells preserved, empty frontier
    /// still zero, shared empty subtrees still dedup through hash-cons)
    /// — the test is here to pin the iterative path specifically.
    #[test]
    fn compact_deep_tree_iterative_clone_reachable() {
        let mut store = NodeStore::new();
        // Level 30 = side 2^30, which keeps coordinates in u64 range and
        // produces a 30-frame chain down each set-cell path. Deeper than
        // the current world root (level 6) by a wide margin.
        let mut root = store.empty(30);
        root = store.set_cell(root, 0, 0, 0, mat(7));
        root = store.set_cell(root, 1, 2, 3, mat(11));

        let (compacted, new_root) = store.compacted(root);

        // Both set cells survive.
        assert_eq!(compacted.get_cell(new_root, 0, 0, 0), mat(7));
        assert_eq!(compacted.get_cell(new_root, 1, 2, 3), mat(11));
        // Frontier cells (still inside-bounds for level 30) read zero.
        assert_eq!(compacted.get_cell(new_root, 100, 200, 300), 0);
        assert_eq!(compacted.get_cell(new_root, 1_000_000, 0, 0), 0);
        // Population is exactly the two set cells.
        assert_eq!(compacted.population(new_root), 2);
    }

    // ===========================================================================
    // Targeted gap-fills (hash-thing-6gf.5).
    // ===========================================================================

    /// `uniform()` with a nonzero state builds a tree where every cell
    /// reads back as that state. Distinct from `empty()` which only
    /// exercises `uniform(level, 0)`.
    #[test]
    fn uniform_nonzero_fills_all_cells() {
        let mut store = NodeStore::new();
        let root = store.uniform(3, mat(4)); // 8x8x8 all material 4
        for z in 0..8u64 {
            for y in 0..8u64 {
                for x in 0..8u64 {
                    assert_eq!(
                        store.get_cell(root, x, y, z),
                        mat(4),
                        "cell ({x},{y},{z}) should be mat(4)",
                    );
                }
            }
        }
        // Population = 512 (all cells live).
        assert_eq!(store.population(root), 512);
    }

    /// Two `uniform()` calls with the same (level, state) return the
    /// same NodeId — the dedup invariant applies to uniform trees too.
    #[test]
    fn uniform_dedup() {
        let mut store = NodeStore::new();
        let a = store.uniform(4, mat(3));
        let b = store.uniform(4, mat(3));
        assert_eq!(a, b);
        // Different state → different id.
        let c = store.uniform(4, mat(5));
        assert_ne!(a, c);
    }

    /// `children()` and `child()` return the correct sub-nodes.
    #[test]
    fn children_and_child_accessors() {
        let mut store = NodeStore::new();
        let l0 = store.leaf(0);
        let l1 = store.leaf(mat(1));
        let l2 = store.leaf(mat(2));
        let kids = [l0, l1, l2, l0, l1, l2, l0, l1];
        let parent = store.interior(1, kids);
        assert_eq!(store.children(parent), kids);
        for (i, &expected) in kids.iter().enumerate() {
            assert_eq!(store.child(parent, i), expected, "child({i})");
        }
    }

    /// `children()` panics on a leaf node.
    #[test]
    #[should_panic(expected = "children()")]
    fn children_panics_on_leaf() {
        let store = NodeStore::new();
        store.children(NodeId::EMPTY);
    }

    /// `Default::default()` produces the same store as `new()`.
    #[test]
    fn default_matches_new() {
        let a = NodeStore::new();
        let b = NodeStore::default();
        assert_eq!(a.stats(), b.stats());
        assert!(matches!(a.get(NodeId::EMPTY), Node::Leaf(0)));
        assert!(matches!(b.get(NodeId::EMPTY), Node::Leaf(0)));
    }

    /// Nonzero metadata survives set/get roundtrip. Previous tests all
    /// use `mat()` which packs metadata=0; this verifies the full 16-bit
    /// `CellState` word is preserved.
    #[test]
    fn metadata_survives_roundtrip() {
        let mut store = NodeStore::new();
        let mut root = store.empty(3);
        // material=5, metadata=42
        let cell = Cell::pack(5, 42).raw();
        root = store.set_cell(root, 3, 3, 3, cell);
        let got = store.get_cell(root, 3, 3, 3);
        assert_eq!(got, cell);
        let decoded = Cell::from_raw(got);
        assert_eq!(decoded.material(), 5);
        assert_eq!(decoded.metadata(), 42);
    }

    /// Writing the same cell twice replaces the first value.
    #[test]
    fn set_cell_overwrites_previous() {
        let mut store = NodeStore::new();
        let mut root = store.empty(3);
        root = store.set_cell(root, 2, 2, 2, mat(7));
        assert_eq!(store.get_cell(root, 2, 2, 2), mat(7));
        root = store.set_cell(root, 2, 2, 2, mat(11));
        assert_eq!(store.get_cell(root, 2, 2, 2), mat(11));
        // Overwriting with 0 (empty) also works.
        root = store.set_cell(root, 2, 2, 2, 0);
        assert_eq!(store.get_cell(root, 2, 2, 2), 0);
    }

    /// `from_flat` / `flatten` roundtrip preserves nonzero metadata.
    #[test]
    fn from_flat_flatten_preserves_metadata() {
        let side = 4usize;
        let mut grid = vec![0 as CellState; side * side * side];
        // Pack (material=3, metadata=17) at a few cells.
        let cell_a = Cell::pack(3, 17).raw();
        let cell_b = Cell::pack(7, 63).raw(); // max metadata
        grid[0] = cell_a;
        grid[side * side * side - 1] = cell_b;

        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);
        let out = store.flatten(root, side);
        assert_eq!(out, grid);
    }

    // ===========================================================================
    // Bounds-check tests (hash-thing-fb5 / 819 bug-fix half).
    //
    // set_cell panics on OOB writes; get_cell silently returns 0 on OOB reads.
    // These tests pin the asymmetric policy: writes are intent (silent loss
    // is unacceptable), reads are queries (far-field probe across an
    // unrealized frontier must not panic).
    // ===========================================================================

    /// Regression for the silent-alias bug: before fb5, `get_cell(8, 0, 0)`
    /// on a level-2 (side 4) root walked the upper-octant branch at every
    /// level and aliased to `(3, 3, 3)`. Now it returns 0.
    #[test]
    fn get_cell_returns_empty_outside_root() {
        let mut store = NodeStore::new();
        let mut root = store.empty(2); // side 4
        root = store.set_cell(root, 3, 3, 3, mat(9)); // corner cell, distinct value
                                                      // Sanity: the corner read works.
        assert_eq!(store.get_cell(root, 3, 3, 3), mat(9));
        // OOB reads on every axis return 0 (not the boundary cell).
        assert_eq!(store.get_cell(root, 4, 0, 0), 0);
        assert_eq!(store.get_cell(root, 0, 4, 0), 0);
        assert_eq!(store.get_cell(root, 0, 0, 4), 0);
        assert_eq!(store.get_cell(root, 100, 100, 100), 0);
        assert_eq!(store.get_cell(root, u64::MAX, 0, 0), 0);
        assert_eq!(store.get_cell(root, 0, u64::MAX, 0), 0);
        assert_eq!(store.get_cell(root, 0, 0, u64::MAX), 0);
    }

    /// OOB writes panic on every axis. Three #[should_panic] tests because
    /// the upper-octant recursion path is independently broken on each axis
    /// and we want to lock all three down.
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_cell_panics_outside_root_x() {
        let mut store = NodeStore::new();
        let root = store.empty(2); // side 4
        store.set_cell(root, 4, 0, 0, mat(1));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_cell_panics_outside_root_y() {
        let mut store = NodeStore::new();
        let root = store.empty(2);
        store.set_cell(root, 0, 4, 0, mat(1));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_cell_panics_outside_root_z() {
        let mut store = NodeStore::new();
        let root = store.empty(2);
        store.set_cell(root, 0, 0, 4, mat(1));
    }

    /// Sanity that the bounds check doesn't off-by-one: max-in-bounds writes
    /// and reads still work for `(side-1, side-1, side-1)` and `(0, 0, 0)`.
    #[test]
    fn set_cell_in_bounds_still_works() {
        let mut store = NodeStore::new();
        let mut root = store.empty(3); // side 8
        root = store.set_cell(root, 7, 7, 7, mat(5));
        root = store.set_cell(root, 0, 0, 0, mat(3));
        assert_eq!(store.get_cell(root, 7, 7, 7), mat(5));
        assert_eq!(store.get_cell(root, 0, 0, 0), mat(3));
    }

    /// Explicit max-in-bounds read — catches `>=` vs `>` mistakes in the
    /// entry-level bounds check.
    #[test]
    fn get_cell_max_in_bounds() {
        let mut store = NodeStore::new();
        let mut root = store.empty(2); // side 4
        root = store.set_cell(root, 3, 3, 3, mat(42));
        assert_eq!(store.get_cell(root, 3, 3, 3), mat(42));
    }

    /// Lock in the "no-assert on OOB read" decision — a future contributor
    /// might see the OOB branch and add a debug_assert!. This test runs
    /// in debug builds and verifies OOB reads don't panic.
    #[test]
    fn get_cell_no_assert_on_oob() {
        let mut store = NodeStore::new();
        let root = store.empty(2);
        // Must not panic in debug builds.
        let _ = store.get_cell(root, 4, 0, 0);
        let _ = store.get_cell(root, 0, 4, 0);
        let _ = store.get_cell(root, 0, 0, 4);
        let _ = store.get_cell(root, u64::MAX, u64::MAX, u64::MAX);
    }

    // ---- Extraction primitives (hash-thing-6gf.6) ----

    /// `extract_neighborhood` returns the correct 26 Moore neighbors for
    /// a cell in the interior of a small world, matching the flat-grid
    /// `get_neighbors` used by the brute-force stepper.
    #[test]
    fn extract_neighborhood_interior_cell() {
        let side = 4usize;
        let mut grid = vec![0 as CellState; side * side * side];
        // Place distinct materials at each cell so we can verify neighbor order.
        // Only use material ids 1..=7 to stay within a small set.
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let id = ((x + y * 4 + z * 16) % 7 + 1) as u16;
                    grid[x + y * side + z * side * side] = mat(id);
                }
            }
        }
        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);

        // Check cell (1,1,1) — all neighbors are in-bounds, no wrapping.
        let neighbors = store.extract_neighborhood(root, side as i64, 1, 1, 1);
        let mut expected = [Cell::EMPTY; 26];
        let mut idx = 0;
        for dz in [-1i64, 0, 1] {
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let nx = (1 + dx) as usize;
                    let ny = (1 + dy) as usize;
                    let nz = (1 + dz) as usize;
                    expected[idx] = Cell::from_raw(grid[nx + ny * side + nz * side * side]);
                    idx += 1;
                }
            }
        }
        assert_eq!(neighbors, expected);
    }

    /// `extract_neighborhood` wraps toroidally: neighbors of a corner
    /// cell pull from the opposite side of the world.
    #[test]
    fn extract_neighborhood_wraps_toroidally() {
        let side = 4usize;
        let mut store = NodeStore::new();
        let mut root = store.empty(2); // 4x4x4
                                       // Place a marker at (3,3,3) — the far corner.
        root = store.set_cell(root, 3, 3, 3, mat(5));
        // Neighbor of (0,0,0) in direction (-1,-1,-1) should wrap to (3,3,3).
        let neighbors = store.extract_neighborhood(root, side as i64, 0, 0, 0);
        // Direction (-1,-1,-1) is the first element in the iteration order
        // (dz=-1, dy=-1, dx=-1, idx=0).
        assert_eq!(neighbors[0], Cell::from_raw(mat(5)));
    }

    /// `extract_block_2x2x2` returns the correct 8 cells for an
    /// axis-aligned block.
    #[test]
    fn extract_block_2x2x2_matches_cells() {
        let mut store = NodeStore::new();
        let mut root = store.empty(3); // 8x8x8
                                       // Place materials in a 2x2x2 block at (2,4,6).
        let coords = [
            (2, 4, 6),
            (3, 4, 6),
            (2, 5, 6),
            (3, 5, 6),
            (2, 4, 7),
            (3, 4, 7),
            (2, 5, 7),
            (3, 5, 7),
        ];
        for (i, &(x, y, z)) in coords.iter().enumerate() {
            root = store.set_cell(root, x, y, z, mat((i + 1) as u16));
        }

        let block = store.extract_block_2x2x2(root, 2, 4, 6);
        for dz in 0..2u64 {
            for dy in 0..2u64 {
                for dx in 0..2u64 {
                    let idx = super::super::node::octant_index(dx as u32, dy as u32, dz as u32);
                    let expected = mat((dx + dy * 2 + dz * 4 + 1) as u16);
                    assert_eq!(
                        block[idx],
                        Cell::from_raw(expected),
                        "block[{idx}] at offset ({dx},{dy},{dz})",
                    );
                }
            }
        }
    }

    /// `extract_block_2x2x2` returns empty cells for OOB coordinates.
    #[test]
    fn extract_block_2x2x2_oob_returns_empty() {
        let mut store = NodeStore::new();
        let root = store.empty(2); // 4x4x4
                                   // Block starting at (3,3,3) extends to (4,4,4) — half OOB.
        let block = store.extract_block_2x2x2(root, 3, 3, 3);
        // Only (3,3,3) is in-bounds; the rest are OOB → empty.
        // All should be empty since the tree is empty.
        for cell in &block {
            assert_eq!(*cell, Cell::EMPTY);
        }
    }

    /// `flatten_region` matches cell-by-cell `get_cell` reads.
    #[test]
    fn flatten_region_matches_get_cell() {
        let side = 8usize;
        let mut grid = vec![0 as CellState; side * side * side];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let id = ((x ^ y ^ z) & 0x7) as u16;
                    grid[x + y * side + z * side * side] = if id == 0 { 0 } else { mat(id) };
                }
            }
        }
        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);

        // Extract a 3x3x3 sub-box at (2,3,4).
        let region = store.flatten_region(root, [2, 3, 4], [3, 3, 3]);
        assert_eq!(region.len(), 27);
        for lz in 0..3usize {
            for ly in 0..3usize {
                for lx in 0..3usize {
                    let expected =
                        store.get_cell(root, 2 + lx as u64, 3 + ly as u64, 4 + lz as u64);
                    assert_eq!(
                        region[lx + ly * 3 + lz * 3 * 3],
                        expected,
                        "region({lx},{ly},{lz})",
                    );
                }
            }
        }
    }

    /// `flatten_region` silently returns 0 for coordinates past the tree.
    #[test]
    fn flatten_region_oob_returns_zero() {
        let mut store = NodeStore::new();
        let mut root = store.empty(2); // 4x4x4
        root = store.set_cell(root, 3, 3, 3, mat(1));

        // Region (3,3,3) to (4,4,4) — extends 1 cell past the tree edge.
        let region = store.flatten_region(root, [3, 3, 3], [2, 2, 2]);
        assert_eq!(region.len(), 8);
        // (0,0,0) in local coords = (3,3,3) world → mat(1)
        assert_eq!(region[0], mat(1));
        // All others are OOB → 0
        for (i, &val) in region.iter().enumerate().skip(1) {
            assert_eq!(val, 0, "region[{i}] should be 0 (OOB)");
        }
    }

    /// `extract_neighborhood` and `flatten` produce consistent results:
    /// for every interior cell, the 26 neighbors from `extract_neighborhood`
    /// match what you'd read from the flat grid.
    #[test]
    fn extract_neighborhood_consistent_with_flatten() {
        let side = 8usize;
        let mut grid = vec![0 as CellState; side * side * side];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let id = ((x * 3 + y * 7 + z * 11) % 5 + 1) as u16;
                    grid[x + y * side + z * side * side] = mat(id);
                }
            }
        }
        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);

        // Check all interior cells (1..7 on each axis to avoid wrap edge cases).
        for z in 1..7 {
            for y in 1..7 {
                for x in 1..7 {
                    let neighbors =
                        store.extract_neighborhood(root, side as i64, x as i64, y as i64, z as i64);
                    let mut idx = 0;
                    for dz in [-1i64, 0, 1] {
                        for dy in [-1i64, 0, 1] {
                            for dx in [-1i64, 0, 1] {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let nx = (x as i64 + dx) as usize;
                                let ny = (y as i64 + dy) as usize;
                                let nz = (z as i64 + dz) as usize;
                                let expected =
                                    Cell::from_raw(grid[nx + ny * side + nz * side * side]);
                                assert_eq!(
                                    neighbors[idx], expected,
                                    "mismatch at ({x},{y},{z}) neighbor ({dx},{dy},{dz})",
                                );
                                idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Column primitive tests (hash-thing-jw3k.1) ----

    /// Build a deterministic heterogeneous grid: cell(x,y,z) = mat(x*7 + y*13 + z*19 + 1).
    /// Avoids raw 1..=63 (would alias empty) by the `+1` shift + `mat()` high-bit encoding.
    fn fill_heterogeneous(store: &mut NodeStore, side: usize) -> (NodeId, Vec<CellState>) {
        let mut grid = vec![0 as CellState; side * side * side];
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let m = ((x * 7 + y * 13 + z * 19) as u16 & 0x3FF).max(1);
                    grid[x + y * side + z * side * side] = mat(m);
                }
            }
        }
        let root = store.from_flat(&grid, side);
        (root, grid)
    }

    #[test]
    fn flatten_column_matches_full_flatten_at_every_xz() {
        let mut store = NodeStore::new();
        let side = 16;
        let (root, grid) = fill_heterogeneous(&mut store, side);
        let mut col = vec![0 as CellState; side];
        for z in 0..side as u64 {
            for x in 0..side as u64 {
                store.flatten_column_into(root, x, z, &mut col);
                for y in 0..side {
                    let expected = grid[x as usize + y * side + z as usize * side * side];
                    assert_eq!(col[y], expected, "cell ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn flatten_column_uniform_stone_returns_all_stone() {
        let mut store = NodeStore::new();
        let stone = mat(STONE_ID);
        let root = store.uniform(4, stone); // 16³
        let mut col = vec![0 as CellState; 16];
        store.flatten_column_into(root, 7, 3, &mut col);
        assert!(col.iter().all(|&c| c == stone));
    }

    const STONE_ID: u16 = 0x100;

    #[test]
    fn flatten_column_through_uniform_subtree_leaf() {
        // Build a world where one quadrant is a uniform-stone subtree and the
        // others are heterogeneous. This forces `flatten_column_rec` to hit a
        // `Leaf(stone)` at an intermediate level for the stone octant.
        let mut store = NodeStore::new();
        let side = 8usize;
        let stone = mat(STONE_ID);
        // Top-level: 2 of 8 children are uniform stone, rest empty leaves.
        let stone_sub = store.uniform(2, stone); // 4³ subtree of stone
        let empty_sub = store.uniform(2, 0);
        let mut children = [empty_sub; 8];
        children[0] = stone_sub; // (ox=0, oy=0, oz=0)
        children[2] = stone_sub; // (ox=0, oy=1, oz=0)
        let root = store.interior(3, children); // 8³ world

        // Column at (x=1, z=1): lies within children[0] for y<4 and children[2] for y>=4.
        let mut col = vec![0 as CellState; side];
        store.flatten_column_into(root, 1, 1, &mut col);
        assert!(
            col.iter().all(|&c| c == stone),
            "expected all stone, got {col:?}"
        );
    }

    #[test]
    fn flatten_column_all_air_returns_all_zero() {
        let mut store = NodeStore::new();
        let root = store.empty(4);
        let mut col = vec![99 as CellState; 16];
        store.flatten_column_into(root, 5, 5, &mut col);
        assert!(col.iter().all(|&c| c == 0));
    }

    #[test]
    fn flatten_column_boundary_corners() {
        let mut store = NodeStore::new();
        let side = 8;
        let (root, grid) = fill_heterogeneous(&mut store, side);
        let mut col = vec![0 as CellState; side];
        for (x, z) in [(0u64, 0u64), (7, 0), (0, 7), (7, 7)] {
            store.flatten_column_into(root, x, z, &mut col);
            for y in 0..side {
                let expected = grid[x as usize + y * side + z as usize * side * side];
                assert_eq!(col[y], expected, "corner ({x},{y},{z})");
            }
        }
    }

    #[test]
    fn flatten_column_side_1() {
        let mut store = NodeStore::new();
        let stone = mat(STONE_ID);
        let root = store.leaf(stone);
        let mut col = vec![0 as CellState; 1];
        store.flatten_column_into(root, 0, 0, &mut col);
        assert_eq!(col, vec![stone]);
    }

    #[test]
    fn flatten_column_side_2() {
        let mut store = NodeStore::new();
        let mut grid = vec![0 as CellState; 8];
        // Side=2 world, populate y=0 with mat(2), y=1 with mat(3) at (x=0,z=0).
        let idx = |x: usize, y: usize, z: usize| x + y * 2 + z * 4;
        grid[idx(0, 0, 0)] = mat(2);
        grid[idx(0, 1, 0)] = mat(3);
        let root = store.from_flat(&grid, 2);
        let mut col = vec![0 as CellState; 2];
        store.flatten_column_into(root, 0, 0, &mut col);
        assert_eq!(col, vec![mat(2), mat(3)]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn flatten_column_out_of_bounds_x() {
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let mut col = vec![0 as CellState; 8];
        store.flatten_column_into(root, 8, 0, &mut col);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn flatten_column_out_of_bounds_z() {
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let mut col = vec![0 as CellState; 8];
        store.flatten_column_into(root, 0, 8, &mut col);
    }

    #[test]
    #[should_panic(expected = "buffer len")]
    fn flatten_column_buffer_len_mismatch() {
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let mut col = vec![0 as CellState; 4]; // wrong — should be 8
        store.flatten_column_into(root, 0, 0, &mut col);
    }

    #[test]
    fn splice_column_identity_returns_same_nodeid() {
        let mut store = NodeStore::new();
        let side = 16;
        let (root, _) = fill_heterogeneous(&mut store, side);
        let mut col = vec![0 as CellState; side];
        store.flatten_column_into(root, 3, 5, &mut col);
        let new_root = store.splice_column(root, 3, 5, &col);
        assert_eq!(
            new_root, root,
            "splicing identical column must yield the same root via hash-cons",
        );
    }

    #[test]
    fn splice_column_no_new_nodes_for_identity() {
        let mut store = NodeStore::new();
        let side = 16;
        let (root, _) = fill_heterogeneous(&mut store, side);
        let before = store.node_count();
        let mut col = vec![0 as CellState; side];
        store.flatten_column_into(root, 5, 5, &mut col);
        let _ = store.splice_column(root, 5, 5, &col);
        assert_eq!(
            store.node_count(),
            before,
            "identity splice must not allocate new nodes",
        );
    }

    #[test]
    fn splice_column_changed_cell_round_trips_via_flatten() {
        let mut store = NodeStore::new();
        let side = 8;
        let (root, mut grid) = fill_heterogeneous(&mut store, side);
        let mut col = vec![0 as CellState; side];
        store.flatten_column_into(root, 2, 3, &mut col);
        col[4] = mat(0x200);
        grid[2 + 4 * side + 3 * side * side] = mat(0x200);
        let new_root = store.splice_column(root, 2, 3, &col);
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let expected = grid[x + y * side + z * side * side];
                    let got = store.get_cell(new_root, x as u64, y as u64, z as u64);
                    assert_eq!(got, expected, "cell ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn splice_column_into_uniform_subtree_expands_leaf() {
        let mut store = NodeStore::new();
        let side = 8usize;
        let stone = mat(STONE_ID);
        // Build a world: children[0] is a uniform-stone level-2 Leaf subtree.
        let stone_sub = store.uniform(2, stone);
        let empty_sub = store.uniform(2, 0);
        let mut children = [empty_sub; 8];
        children[0] = stone_sub;
        let root = store.interior(3, children);

        // Column at (x=1, z=1) passes through children[0] for y<4 and
        // children[2] (empty) for y>=4. The column should read as 4 stone
        // cells followed by 4 air cells.
        let mut col = vec![0 as CellState; side];
        store.flatten_column_into(root, 1, 1, &mut col);
        for (y, expected) in col.iter().enumerate() {
            let want = if y < 4 { stone } else { 0 };
            assert_eq!(*expected, want, "pre-splice cell y={y}");
        }

        // Mutate one cell inside the uniform subtree — forces splice_column
        // to expand the intermediate Leaf(stone) into 8 uniform children.
        col[2] = mat(0x300);
        let new_root = store.splice_column(root, 1, 1, &col);

        // Verify: other cells in that uniform subtree are still stone.
        for y in 0..4 {
            let got = store.get_cell(new_root, 1, y as u64, 1);
            let want = if y == 2 { mat(0x300) } else { stone };
            assert_eq!(got, want, "post-splice (1,{y},1)");
        }
        // Other columns in the stone subtree untouched.
        assert_eq!(store.get_cell(new_root, 0, 2, 0), stone);
        assert_eq!(store.get_cell(new_root, 3, 2, 3), stone);
    }

    /// jw3k.3 canonicalization: identity splice through an intermediate
    /// raw `Leaf(s)` child preserves the original root NodeId. The Leaf is
    /// expanded on descent and canonicalized back to `Leaf(s)` on ascent,
    /// so hash-cons resolves the final interior to the same id as input.
    #[test]
    fn splice_column_identity_through_raw_leaf_child_preserves_nodeid() {
        let mut store = NodeStore::new();
        let stone = mat(0x301);
        // Level-2 interior (side=4). children[0] is raw Leaf(stone) — a
        // compressed level-1 uniform-stone subtree (distinct from
        // uniform(1, stone) which is interior(1, [leaf(stone); 8])).
        let leaf_stone = store.leaf(stone);
        let empty_l1 = store.empty(1);
        let mut children = [empty_l1; 8];
        children[0] = leaf_stone;
        let root = store.interior(2, children);

        let mut col = vec![0 as CellState; 4];
        store.flatten_column_into(root, 0, 0, &mut col);
        assert_eq!(col, vec![stone, stone, 0, 0], "column read correctly");

        let new_root = store.splice_column(root, 0, 0, &col);

        for y in 0..4u64 {
            let expected = if y < 2 { stone } else { 0 };
            assert_eq!(
                store.get_cell(new_root, 0, y, 0),
                expected,
                "grid preserved at (0,{y},0)",
            );
        }
        assert_eq!(
            new_root, root,
            "canonicalization restores original NodeId on identity splice",
        );
    }

    /// jw3k.3 combined stress: mixed heterogeneous tree where children[0]
    /// is a raw Leaf(stone) and other children are a heterogeneous fill.
    /// Identity splice over multiple (x, z) columns (some through the raw
    /// Leaf, some through Interior siblings) must leave node_count stable.
    #[test]
    fn splice_column_identity_mixed_tree_no_node_growth() {
        let mut store = NodeStore::new();
        let side = 4;
        let stone = mat(0x301);
        // Level-2 tree. children[0] is raw Leaf(stone); children[1..7] get
        // a heterogeneous fill (varied cells in each level-1 subtree).
        let leaf_stone = store.leaf(stone);
        let (hetero_sub_seed, _) = fill_heterogeneous(&mut store, 2);
        let hetero_sub = hetero_sub_seed;
        let mut children = [hetero_sub; 8];
        children[0] = leaf_stone;
        let root = store.interior(2, children);

        let before = store.node_count();
        // Repeat identity splice over every (x, z) column — mix of
        // raw-Leaf path (x<2, z<2) and Interior paths (others). None
        // should allocate new nodes.
        for _ in 0..8 {
            for z in 0..side as u64 {
                for x in 0..side as u64 {
                    let mut col = vec![0 as CellState; side];
                    store.flatten_column_into(root, x, z, &mut col);
                    let new_root = store.splice_column(root, x, z, &col);
                    assert_eq!(
                        new_root, root,
                        "identity splice at ({x},{z}) must preserve NodeId",
                    );
                }
            }
        }
        assert_eq!(
            store.node_count(),
            before,
            "mixed-tree identity splices must not grow node count",
        );
    }

    /// jw3k.3: identity splice through a raw Leaf child must not grow the
    /// node count. Regression guard on the canonicalization in
    /// splice_column_rec.
    #[test]
    fn splice_column_identity_through_raw_leaf_no_node_growth() {
        let mut store = NodeStore::new();
        let stone = mat(0x301);
        let leaf_stone = store.leaf(stone);
        let empty_l1 = store.empty(1);
        let mut children = [empty_l1; 8];
        children[0] = leaf_stone;
        let root = store.interior(2, children);

        let mut col = vec![0 as CellState; 4];
        store.flatten_column_into(root, 0, 0, &mut col);
        let before = store.node_count();
        // Many identity splices through the raw-Leaf column must not
        // allocate new nodes — each expand+canonicalize round-trip
        // resolves to the pre-existing Leaf(s) id.
        for _ in 0..32 {
            let _ = store.splice_column(root, 0, 0, &col);
        }
        assert_eq!(
            store.node_count(),
            before,
            "identity splice through raw Leaf must not grow node count",
        );
    }

    #[test]
    fn splice_column_side_1() {
        let mut store = NodeStore::new();
        let root = store.leaf(mat(0x10));
        let new_root = store.splice_column(root, 0, 0, &[mat(0x20)]);
        assert_eq!(store.get_cell(new_root, 0, 0, 0), mat(0x20));
    }

    #[test]
    fn splice_column_side_2() {
        let mut store = NodeStore::new();
        let grid = vec![0 as CellState; 8];
        let root = store.from_flat(&grid, 2);
        let new_root = store.splice_column(root, 1, 0, &[mat(0x11), mat(0x22)]);
        assert_eq!(store.get_cell(new_root, 1, 0, 0), mat(0x11));
        assert_eq!(store.get_cell(new_root, 1, 1, 0), mat(0x22));
        // Other cells remain empty.
        assert_eq!(store.get_cell(new_root, 0, 0, 0), 0);
        assert_eq!(store.get_cell(new_root, 0, 1, 1), 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn splice_column_out_of_bounds_x() {
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let col = vec![0 as CellState; 8];
        let _ = store.splice_column(root, 8, 0, &col);
    }

    #[test]
    #[should_panic(expected = "column len")]
    fn splice_column_length_mismatch() {
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let col = vec![0 as CellState; 4]; // wrong — should be 8
        let _ = store.splice_column(root, 0, 0, &col);
    }

    // ---- reachable_nodes_per_chunk + reachable_node_count + reachable_summary
    //
    // Measurement-only walker for cswp.8.1 — see store.rs docstrings and
    // `tests/bench_chunk_reachable.rs` for the bench that consumes this.

    #[test]
    fn reachable_nodes_per_chunk_empty_world() {
        // empty(level=4) builds an `Interior` per level (no zero-pop
        // collapse), so an empty chunk at chunk_level=2 reaches:
        //   chunk-root Interior(level=2) +
        //   Interior(level=1)             +
        //   NodeId::EMPTY (Leaf 0)
        // = 3 unique NodeIds. chunks_per_axis = 1 << (4-2) = 4, so 4³ = 64
        // chunks, all count=3, all coordinates distinct.
        let mut store = NodeStore::new();
        let root = store.empty(4);
        let chunks = store.reachable_nodes_per_chunk(root, 2);
        assert_eq!(chunks.len(), 64);
        assert!(
            chunks.iter().all(|&(_, n)| n == 3),
            "all empty chunks reach k+1 = 3 nodes; got {:?}",
            chunks.iter().map(|&(_, n)| n).collect::<Vec<_>>(),
        );
        let coords: std::collections::HashSet<_> = chunks.iter().map(|&(c, _)| c).collect();
        assert_eq!(coords.len(), 64);
    }

    #[test]
    fn reachable_nodes_per_chunk_identity_at_root_level() {
        // chunk_level == root_level: a single entry whose count equals the
        // global reachable count.
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let chunks = store.reachable_nodes_per_chunk(root, 3);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, (0, 0, 0));
        assert_eq!(chunks[0].1, store.reachable_node_count(root));
    }

    #[test]
    fn reachable_nodes_per_chunk_distinct_chunks_distinct_counts() {
        // 8³ tree (level 3); place a single non-empty cell in chunk (0,0,0)
        // at chunk_level=1 (chunks_per_axis = 4, chunk_side = 2).
        let mut store = NodeStore::new();
        let mut root = store.empty(3);
        root = store.set_cell(root, 0, 0, 0, mat(7));
        let chunks = store.reachable_nodes_per_chunk(root, 1);
        assert_eq!(chunks.len(), 64);

        let zero_count = chunks.iter().find(|&&(c, _)| c == (0, 0, 0)).unwrap().1;
        // Non-empty chunk (0,0,0): chunk root is Interior(level=1) with one
        // non-empty Leaf in octant 0 and seven empty Leafs elsewhere — so
        // reachable = {Interior(level=1, non-empty), populated-Leaf,
        //              NodeId::EMPTY (the empty Leaf)} = 3 unique nodes.
        assert!(
            zero_count >= 3,
            "non-empty chunk (0,0,0) must reach Interior + populated-Leaf + EMPTY (>= 3); got {zero_count}",
        );

        let far_count = chunks.iter().find(|&&(c, _)| c == (3, 3, 3)).unwrap().1;
        // Empty chunk at chunk_level=1: Interior(level=1, empty) + Leaf 0
        // = 2 unique NodeIds. (See docstring: empty(level) doesn't
        // collapse zero-pop interiors to a single canonical id.)
        assert_eq!(
            far_count, 2,
            "empty chunk at chunk_level=1 reaches the empty Interior + EMPTY leaf",
        );
    }

    #[test]
    fn reachable_node_count_matches_per_chunk_at_root_level() {
        // Whatever the global count is, the per-chunk walker at
        // chunk_level == root_level must agree, because both walks visit
        // the same DAG with one shared visited-set.
        let mut store = NodeStore::new();
        let mut root = store.empty(2);
        root = store.set_cell(root, 1, 2, 3, mat(0x42));
        let global = store.reachable_node_count(root);
        let per_chunk = store.reachable_nodes_per_chunk(root, 2);
        assert_eq!(per_chunk.len(), 1);
        assert_eq!(per_chunk[0].1, global);
    }

    #[test]
    fn reachable_summary_matches_separate_calls() {
        // The combined walker must agree with the two single-purpose
        // walkers on every chunk and on the global denominator.
        let mut store = NodeStore::new();
        let mut root = store.empty(3);
        root = store.set_cell(root, 0, 0, 0, mat(1));
        root = store.set_cell(root, 7, 7, 7, mat(2));

        let separate_chunks = store.reachable_nodes_per_chunk(root, 1);
        let separate_global = store.reachable_node_count(root);
        let (combined_chunks, combined_global) = store.reachable_summary(root, 1);

        assert_eq!(combined_chunks, separate_chunks);
        assert_eq!(combined_global, separate_global);
    }

    #[test]
    #[should_panic(expected = "chunk_level=0")]
    fn reachable_nodes_per_chunk_rejects_chunk_level_zero() {
        let store = NodeStore::new();
        let _ = store.reachable_nodes_per_chunk(NodeId::EMPTY, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds root_level")]
    fn reachable_nodes_per_chunk_rejects_chunk_level_above_root() {
        let mut store = NodeStore::new();
        let root = store.empty(2);
        let _ = store.reachable_nodes_per_chunk(root, 3);
    }

    /// `NodeId::EMPTY` is `Leaf(0)` at level 0. Walking it with any
    /// `chunk_level > 0` must trip the `chunk_level > root_level` panic,
    /// pinning the leaf-root × chunk_level>0 path against off-by-one
    /// drift in the bound.
    #[test]
    #[should_panic(expected = "exceeds root_level")]
    fn reachable_nodes_per_chunk_rejects_chunk_level_above_leaf_root() {
        let store = NodeStore::new();
        let _ = store.reachable_nodes_per_chunk(NodeId::EMPTY, 1);
    }

    /// Exercises the Leaf-above-chunk arm of `walk_chunks_rec`: a raw
    /// `Leaf(s)` sitting at a child slot whose contextual level >
    /// `chunk_level`. This structural state arises in practice via
    /// `splice_column`'s canonicalize path (see the (gtua) refs); we
    /// build it directly with `interior(...)` for a focused test, and
    /// also round-trip through `splice_column` to confirm the live path
    /// preserves the same NodeId.
    #[test]
    fn reachable_nodes_per_chunk_leaf_above_chunk_arm() {
        let mut store = NodeStore::new();
        let stone = mat(0x301);
        // Level-3 tree (side=8). children[0] is a raw Leaf(stone) sitting
        // at contextual level 2; the other 7 children are empty(2)
        // Interior subtrees. octant 0 = (0,0,0) so the raw-Leaf region
        // covers cell coords (0..4, 0..4, 0..4).
        let leaf_stone = store.leaf(stone);
        let empty_l2 = store.empty(2);
        let mut children = [empty_l2; 8];
        children[0] = leaf_stone;
        let root = store.interior(3, children);

        // Round-trip through splice_column on a column inside the
        // raw-Leaf region: the canonicalize path must preserve the
        // raw-Leaf-at-level-2 structural state (NodeId identity).
        let mut col = vec![0 as CellState; 8];
        store.flatten_column_into(root, 0, 0, &mut col);
        let after = store.splice_column(root, 0, 0, &col);
        assert_eq!(
            after, root,
            "identity splice through raw-Leaf-at-level-2 must preserve NodeId",
        );

        // Walk at chunk_level=1 (chunk_side=2, 4×4×4 = 64 chunks).
        let chunks = store.reachable_nodes_per_chunk(root, 1);
        assert_eq!(
            chunks.len(),
            64,
            "4³ chunks_per_axis at level=3,chunk_level=1"
        );

        // The raw-Leaf region covers chunks (cx,cy,cz) ∈ {0,1}³ — 8 chunks
        // emitted via the Leaf-above-chunk arm with count == 1 each.
        let mut leaf_arm_chunks: Vec<(u64, u64, u64)> = Vec::new();
        for &((cx, cy, cz), n) in &chunks {
            if cx < 2 && cy < 2 && cz < 2 {
                assert_eq!(
                    n, 1,
                    "Leaf-above-chunk arm at ({cx},{cy},{cz}) must emit count=1, got {n}",
                );
                leaf_arm_chunks.push((cx, cy, cz));
            } else {
                // empty(2) descended via the Interior arm: each chunk
                // sees Interior(level=1) + Leaf(0) = 2 reachable nodes.
                assert_eq!(
                    n, 2,
                    "Interior-descent chunk at ({cx},{cy},{cz}) reaches Interior(1)+Leaf(0)=2, got {n}",
                );
            }
        }

        // The 8 Leaf-above chunk coords must be dense over the 2×2×2 grid
        // (the region the raw Leaf covers), with no duplicates. Compare
        // as a sorted vec so the assertion is order-agnostic.
        leaf_arm_chunks.sort_unstable();
        let mut expected: Vec<(u64, u64, u64)> = (0..2)
            .flat_map(|cx| (0..2).flat_map(move |cy| (0..2).map(move |cz| (cx, cy, cz))))
            .collect();
        expected.sort_unstable();
        assert_eq!(
            leaf_arm_chunks, expected,
            "Leaf-above arm chunk coords must be dense over the raw-Leaf's 2³ chunk region",
        );
    }

    // -- cswp.8.2: lod_collapse / lod_collapse_chunk ------------------------

    /// `target_lod = 0` is identity by contract — returns the input NodeId.
    #[test]
    fn lod_collapse_identity_at_target_lod_zero() {
        let mut store = NodeStore::new();
        let mut root = store.empty(4);
        root = store.set_cell(root, 1, 2, 3, mat(7));
        root = store.set_cell(root, 8, 8, 8, mat(2));
        let collapsed = store.lod_collapse(root, 0);
        assert_eq!(
            collapsed, root,
            "lod_collapse(_, 0) must return the input NodeId unchanged",
        );
    }

    /// `target_lod = root_level` collapses the entire tree to a single Leaf
    /// carrying the representative state. For a uniform-stone root, that
    /// representative is unambiguously `Leaf(stone)`.
    #[test]
    fn lod_collapse_full_collapse_at_root_level() {
        let mut store = NodeStore::new();
        let stone = mat(1);
        let root = store.uniform(3, stone);
        let collapsed = store.lod_collapse(root, 3);
        assert!(
            matches!(*store.get(collapsed), Node::Leaf(s) if s == stone),
            "full collapse of uniform(3, stone) must be Leaf(stone), got {:?}",
            store.get(collapsed),
        );
    }

    /// Collapsing a uniform region preserves root level and representative
    /// material. The collapsed shape is NOT NodeId-equal to a freshly-built
    /// `uniform(N, stone)`: that helper builds an Interior chain all the way
    /// to a level-0 Leaf, while the collapsed tree stops with a Leaf at
    /// contextual level `target_lod`. The invariants we DO get are:
    /// (a) the collapsed root stays at the input root level, and (b) the
    /// representative state of the collapsed root is the uniform material.
    #[test]
    fn lod_collapse_uniform_preserves_material() {
        let mut store = NodeStore::new();
        let stone = mat(1);
        let original = store.uniform(4, stone);
        let collapsed = store.lod_collapse(original, 2);
        assert_eq!(store.get(collapsed).level(), 4);
        assert_eq!(store.representative_state(collapsed), stone);
    }

    /// Largest-population child wins. Populate one octant with 64 stones,
    /// another with 8 water, leave the rest empty, collapse to root level.
    /// The root's representative must be stone.
    #[test]
    fn lod_collapse_dominant_material_wins() {
        let mut store = NodeStore::new();
        let stone = mat(1);
        let water = mat(2);
        let mut root = store.empty(3); // 8³
                                       // Octant 0 (origin corner): fill a 4³ block with stone (64 cells).
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    root = store.set_cell(root, x, y, z, stone);
                }
            }
        }
        // Octant 7 (far corner): fill a 2³ block with water (8 cells).
        for x in 4..6 {
            for y in 4..6 {
                for z in 4..6 {
                    root = store.set_cell(root, x, y, z, water);
                }
            }
        }
        let collapsed = store.lod_collapse(root, 3);
        assert!(
            matches!(*store.get(collapsed), Node::Leaf(s) if s == stone),
            "dominant-population child (stone, pop 64) must win over water (pop 8); got {:?}",
            store.get(collapsed),
        );
    }

    /// Empty world stays empty under any collapse — verified via population
    /// and full flatten, not NodeId equality. The collapsed tree's
    /// structural shape differs from `empty(N)` for `target_lod > 0` (the
    /// Interior chain stops at `target_lod` rather than continuing to
    /// level 0), so NodeId equality is too strong; the semantic invariant
    /// is "every cell is 0".
    #[test]
    fn lod_collapse_empty_world_stays_empty() {
        let mut store = NodeStore::new();
        let root = store.empty(4);
        let side: usize = 1 << 4;
        for k in 0..=4 {
            let collapsed = store.lod_collapse(root, k);
            assert_eq!(
                store.population(collapsed),
                0,
                "empty world collapsed at target_lod={k} must have population 0",
            );
            // For k < root_level the collapsed tree still has root level 4
            // and we can flatten the full grid; for k == root_level the
            // result is a single Leaf(0) and flatten is meaningless.
            if k < 4 {
                let flat = store.flatten(collapsed, side);
                assert!(
                    flat.iter().all(|&c| c == 0),
                    "empty world collapsed at target_lod={k} must flatten to all zeros",
                );
            } else {
                assert!(
                    matches!(*store.get(collapsed), Node::Leaf(0)),
                    "empty world collapsed at root_level must be Leaf(0)",
                );
            }
        }
    }

    /// `target_lod > root_level` panics.
    #[test]
    #[should_panic(expected = "target_lod")]
    fn lod_collapse_panics_above_root_level() {
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let _ = store.lod_collapse(root, 4);
    }

    /// Property: collapsing must never grow the reachable-node count. The
    /// strict equality property for arbitrary trees is content-dependent
    /// because hash-consing already shares uniform/empty regions, so the
    /// monotonic inequality is the right invariant.
    #[test]
    fn lod_collapse_count_after_le_count_before() {
        let mut store = NodeStore::new();
        // Build a heterogeneous tree: a few scattered cells across the world.
        let mut root = store.empty(4); // 16³
        let probes: &[(u64, u64, u64, u16)] = &[
            (0, 0, 0, 1),
            (15, 15, 15, 2),
            (3, 7, 11, 3),
            (8, 0, 4, 1),
            (12, 12, 12, 2),
            (1, 14, 5, 3),
            (9, 2, 9, 1),
        ];
        for &(x, y, z, m) in probes {
            root = store.set_cell(root, x, y, z, mat(m));
        }
        let count_before = store.reachable_node_count(root);
        for k in 1..=4 {
            let collapsed = store.lod_collapse(root, k);
            let count_after = store.reachable_node_count(collapsed);
            assert!(
                count_after <= count_before,
                "lod_collapse(_, {k}): count_after ({count_after}) > count_before ({count_before})",
            );
        }
    }

    /// `lod_collapse_chunk` only mutates the targeted chunk's subtree:
    /// every other top-level child of the root must remain NodeId-equal.
    #[test]
    fn lod_collapse_chunk_only_touches_one_chunk() {
        let mut store = NodeStore::new();
        // root_level=4, chunk_level=2 → 4 chunks per axis (chunks are 4³).
        let mut root = store.empty(4);
        // Sprinkle one cell into each of two different top-level octants so
        // their subtrees diverge (avoiding full hash-cons sharing across
        // the whole world).
        root = store.set_cell(root, 1, 1, 1, mat(1)); // octant 0 of root
        root = store.set_cell(root, 9, 9, 9, mat(2)); // octant 7 of root
        let original_children = store.children(root);
        // Collapse the chunk at top-level octant 0 (origin chunk at chunk_level=3).
        // chunk_level=3 → 2 chunks per axis; chunk (0,0,0) is the level-3 child
        // at octant 0 of the root.
        let collapsed = store.lod_collapse_chunk(root, 0, 0, 0, 3, 1);
        let collapsed_children = store.children(collapsed);
        // Octant 0's subtree may differ. Octants 1..8 must be unchanged.
        for oct in 1..8 {
            assert_eq!(
                collapsed_children[oct], original_children[oct],
                "octant {oct} subtree must be NodeId-equal after lod_collapse_chunk",
            );
        }
    }

    /// `target_lod > chunk_level` panics.
    #[test]
    #[should_panic(expected = "target_lod")]
    fn lod_collapse_chunk_panics_target_above_chunk_level() {
        let mut store = NodeStore::new();
        let root = store.empty(4);
        // chunk_level=2, target_lod=3 → must panic.
        let _ = store.lod_collapse_chunk(root, 0, 0, 0, 2, 3);
    }

    /// Hash-cons sharing extends to *every* sibling on the descent path,
    /// not just the top-level seven. With a non-zero deeper chunk coord
    /// (`chunk_level=1`, coord `(3, 5, 6)` in a `root_level=4` world), the
    /// recursive walker descends 3 levels; each step picks one octant and
    /// leaves seven siblings untouched. The test asserts NodeId equality
    /// at every descent step. Stresses the non-trivial `stride` / `local`
    /// remap that a single zero-coord case could mask.
    #[test]
    fn lod_collapse_chunk_siblings_on_descent_stay_nodeid_equal() {
        let mut store = NodeStore::new();
        // Heterogeneous root_level=4 (16³) world — sprinkle a few cells
        // across different deep octants so subtrees don't accidentally
        // hash-cons-share globally.
        let mut root = store.empty(4);
        let probes: &[(u64, u64, u64, u16)] = &[
            (1, 1, 1, 1),
            (3, 5, 6, 2),
            (15, 15, 15, 3),
            (8, 0, 4, 1),
            (3, 5, 7, 2),
        ];
        for &(x, y, z, m) in probes {
            root = store.set_cell(root, x, y, z, mat(m));
        }
        // chunk_level=1 (chunks are 2³ cells), target_lod=1 (full collapse
        // inside the chunk). Coord (3, 5, 6): octants visited per-step are
        // not all zero, so the stride/local remap is non-trivial.
        let cx = 3u64;
        let cy = 5u64;
        let cz = 6u64;
        let chunk_level = 1u32;
        let collapsed = store.lod_collapse_chunk(root, cx, cy, cz, chunk_level, 1);

        // Walk both trees in lockstep using the same octant math the rec
        // walker uses. At each descent step, assert the seven non-target
        // octants are NodeId-equal between original and collapsed.
        let mut orig_node = root;
        let mut new_node = collapsed;
        for level in (chunk_level + 1..=4).rev() {
            let orig_children = match *store.get(orig_node) {
                Node::Interior { children, .. } => children,
                Node::Leaf(_) => panic!("test fixture descent should not hit a Leaf"),
            };
            let new_children = match *store.get(new_node) {
                Node::Interior { children, .. } => children,
                Node::Leaf(_) => panic!("collapsed descent should not hit a Leaf"),
            };
            let stride = 1u64 << (level - 1 - chunk_level);
            let ox = (cx / stride) & 1;
            let oy = (cy / stride) & 1;
            let oz = (cz / stride) & 1;
            let oct = octant_index(ox as u32, oy as u32, oz as u32);
            for i in 0..8 {
                if i == oct {
                    continue;
                }
                assert_eq!(
                    new_children[i], orig_children[i],
                    "level={level} sibling octant {i} must be NodeId-equal after lod_collapse_chunk",
                );
            }
            orig_node = orig_children[oct];
            new_node = new_children[oct];
        }
    }

    /// Targeting a chunk that lives inside a uniform-Leaf-above-chunk_level
    /// region must short-circuit at the Leaf early-return arm and return the
    /// input root unchanged. Built directly from a raw `Leaf` at child slot
    /// 0 of an `Interior(level=3)`; the recursive walker descends one step
    /// to `level=2`, hits the `Leaf` arm before reaching `chunk_level=1`,
    /// and returns the original NodeId.
    ///
    /// Probed at both `target_lod=0` (the trivially-no-op inside-the-chunk
    /// case) and `target_lod=1` (the case where the inner `lod_collapse`
    /// would actually rewrite — the Leaf-arm early-return must fire
    /// *before* the inner call, otherwise this assertion regresses).
    #[test]
    fn lod_collapse_chunk_uniform_leaf_above_chunk_level() {
        let mut store = NodeStore::new();
        let stone_leaf = store.leaf(mat(1));
        let empty2 = store.empty(2);
        let mut children = [empty2; 8];
        children[0] = stone_leaf;
        let raw_root = store.interior(3, children);

        // target_lod=0: trivial, but exercises the Leaf-arm path on its own.
        let collapsed_0 = store.lod_collapse_chunk(raw_root, 0, 0, 0, 1, 0);
        assert_eq!(
            collapsed_0, raw_root,
            "uniform-Leaf-above-chunk_level chunk collapse (target_lod=0) must return root unchanged",
        );

        // target_lod=1: pins the Leaf arm firing *before* the inner
        // lod_collapse. If the Leaf check were ever reordered below the
        // `level == chunk_level` check, this would diverge.
        let collapsed_1 = store.lod_collapse_chunk(raw_root, 0, 0, 0, 1, 1);
        assert_eq!(
            collapsed_1, raw_root,
            "uniform-Leaf-above-chunk_level chunk collapse (target_lod=1) must still return root unchanged",
        );
    }

    /// Idempotence: collapsing twice at the same target_lod returns the
    /// same NodeId. Hash-cons identity property.
    ///
    /// Caveat: when `target_lod == root_level` the first collapse already
    /// reduces the tree to a single `Leaf` whose `Node::level() == 0`, so a
    /// re-collapse at `k = root_level` would panic the assert
    /// `target_lod ≤ root_level`. The clean idempotence statement is "the
    /// second collapse uses target_lod clamped to the post-collapse root
    /// level": `lod_collapse(once, min(k, once.level()))` returns `once`.
    #[test]
    fn lod_collapse_idempotent() {
        let mut store = NodeStore::new();
        let mut root = store.empty(4);
        root = store.set_cell(root, 0, 0, 0, mat(1));
        root = store.set_cell(root, 15, 15, 15, mat(2));
        root = store.set_cell(root, 7, 8, 9, mat(3));
        for k in 0..=4 {
            let once = store.lod_collapse(root, k);
            let once_level = store.get(once).level();
            let k2 = k.min(once_level);
            let twice = store.lod_collapse(once, k2);
            assert_eq!(
                once, twice,
                "lod_collapse must be idempotent at target_lod={k} (re-collapse at {k2})",
            );
        }
    }

    /// ip0d memo correctness: a tree where the same NodeId appears as a
    /// child of two *different-shape* parents at the target_lod boundary
    /// still produces the same collapsed result as a fresh-store rebuild.
    /// Hash-cons guarantees the shared subtree dedups; the memo must return
    /// the same `CellState` for that NodeId regardless of which parent
    /// queries it.
    #[test]
    fn lod_collapse_memo_shared_subtree_under_distinct_parents() {
        let stone = mat(1);
        let water = mat(2);
        let dirt = mat(3);
        // Build the same scene twice, in separate stores, to compare.
        // Scene: root_level=3 (8³). Octant 0 holds a 2³ stone block at
        // origin; octant 7 holds a 2³ stone block at (4,4,4). Both octants'
        // octant-0 child (a level-1 subtree) is identical (a uniform stone
        // 2³), so hash-cons shares it. We then differentiate the two
        // top-level octants by adding a unique cell elsewhere in each, so
        // the parents have different shapes but share one identical child.
        fn build(store: &mut NodeStore, stone: u16, water: u16, dirt: u16) -> NodeId {
            let mut root = store.empty(3);
            // Octant 0 (origin): fill 2³ at (0,0,0)..(2,2,2) with stone.
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        root = store.set_cell(root, x, y, z, stone);
                    }
                }
            }
            // Octant 7 (far): fill 2³ at (4,4,4)..(6,6,6) with stone — the
            // octant-0 child of octant-7's level-2 subtree is now a
            // uniform-stone level-1 block, which hash-cons-shares with the
            // identical block in octant 0.
            for x in 4..6 {
                for y in 4..6 {
                    for z in 4..6 {
                        root = store.set_cell(root, x, y, z, stone);
                    }
                }
            }
            // Make octants 0 and 7 distinguishable: drop unique cells.
            root = store.set_cell(root, 3, 0, 0, water);
            root = store.set_cell(root, 7, 4, 4, dirt);
            root
        }

        let mut store_a = NodeStore::new();
        let root_a = build(&mut store_a, stone, water, dirt);
        let collapsed_a = store_a.lod_collapse(root_a, 1);

        // Compare against a fresh build → fresh collapse: same scene must
        // produce a structurally-equal collapsed root, and the
        // representative state must match on a probe.
        let mut store_b = NodeStore::new();
        let root_b = build(&mut store_b, stone, water, dirt);
        let collapsed_b = store_b.lod_collapse(root_b, 1);

        assert_eq!(
            store_a.reachable_node_count(collapsed_a),
            store_b.reachable_node_count(collapsed_b),
            "memo path and reference must produce structurally-equal trees",
        );
        assert_eq!(
            store_a.representative_state(collapsed_a),
            store_b.representative_state(collapsed_b),
            "memo path and reference must agree on root representative state",
        );
        // Cell-by-cell flatten agreement: ensures the materials at every
        // collapsed cell match (catches any memo-induced state drift).
        let side: usize = 1 << 3;
        let flat_a = store_a.flatten(collapsed_a, side);
        let flat_b = store_b.flatten(collapsed_b, side);
        assert_eq!(
            flat_a, flat_b,
            "memo path and reference must produce identical flattened cell grids",
        );
    }

    /// ip0d: `lod_collapse_chunk` calls `lod_collapse` once at the chunk
    /// boundary, so the memo benefits the chunk wrapper too. Verify the
    /// chunk path produces the same result as collapsing the chunk subtree
    /// directly via `lod_collapse`.
    #[test]
    fn lod_collapse_chunk_memo_matches_direct_collapse() {
        let mut store = NodeStore::new();
        let mut root = store.empty(4);
        // Heterogeneous fill so chunk subtrees aren't trivially uniform.
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    root = store.set_cell(root, x, y, z, mat(1));
                }
            }
        }
        for x in 4..8 {
            for y in 0..4 {
                for z in 0..4 {
                    root = store.set_cell(root, x, y, z, mat(2));
                }
            }
        }
        // Chunk at (0,0,0) with chunk_level=2 (8³ chunks → 2 per axis at
        // root_level=4). target_lod=1: collapse one level inside the chunk.
        let chunk_level = 2u32;
        let target_lod = 1u32;
        let via_chunk = store.lod_collapse_chunk(root, 0, 0, 0, chunk_level, target_lod);

        // The chunk-(0,0,0) subtree at chunk_level=2 is the level-2 octant-0
        // child of the level-3 octant-0 child of root. Walk to it and
        // collapse directly.
        let level3_octant0 = match *store.get(root) {
            Node::Interior { children, .. } => children[0],
            _ => panic!("root must be Interior"),
        };
        let level2_chunk = match *store.get(level3_octant0) {
            Node::Interior { children, .. } => children[0],
            _ => panic!("level-3 octant-0 must be Interior for this fixture"),
        };
        let direct = store.lod_collapse(level2_chunk, target_lod);

        // The chunk wrapper rebuilds the path; after the collapse, the
        // root's path-to-chunk-(0,0,0) should terminate in the same NodeId
        // as `direct`.
        let new_l3_oct0 = match *store.get(via_chunk) {
            Node::Interior { children, .. } => children[0],
            _ => panic!("collapsed root must be Interior"),
        };
        let new_l2_chunk = match *store.get(new_l3_oct0) {
            Node::Interior { children, .. } => children[0],
            _ => panic!("new level-3 octant-0 must be Interior"),
        };
        assert_eq!(
            new_l2_chunk, direct,
            "lod_collapse_chunk must produce the same chunk subtree as a direct lod_collapse on it",
        );
    }

    /// For a uniform-stone tree at root_level=N, the collapsed reachable
    /// count at any `target_lod = k` is exactly `(N - k) + 1`: one Interior
    /// link per preserved level above target_lod, plus the terminal
    /// `Leaf(stone)` at level k. This pins the exact compression for the
    /// uniform case (the inequality property test admits silent
    /// undercounting).
    #[test]
    fn lod_collapse_uniform_exact_count() {
        let mut store = NodeStore::new();
        let stone = mat(1);
        let root_level: u32 = 4;
        let root = store.uniform(root_level, stone);
        for k in 0..=root_level {
            let collapsed = store.lod_collapse(root, k);
            let expected = (root_level - k) as u64 + 1;
            let actual = store.reachable_node_count(collapsed);
            assert_eq!(
                actual, expected,
                "uniform(root_level={root_level}, stone) collapsed at target_lod={k}: expected {expected} reachable nodes, got {actual}",
            );
        }
    }
}
