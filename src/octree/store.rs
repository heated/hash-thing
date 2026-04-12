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
        let id = NodeId(
            u32::try_from(self.nodes.len()).expect("NodeStore overflow: exceeded 2^32 nodes"),
        );
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
        let children = match self.get(node).clone() {
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
            compacted.stats(), 1,
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
}
