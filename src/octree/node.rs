/// Cell state — 8 bits for now (256 materials), expandable later.
pub type CellState = u8;

/// Index into the canonical node store. u32 to keep nodes small.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const EMPTY: NodeId = NodeId(0); // reserved: canonical empty node
}

/// A canonical octree node.
///
/// Level 0: a single cell (1x1x1).
/// Level n: a 2^n × 2^n × 2^n cube, with 8 children at level n-1.
///
/// Children are indexed by octant:
///   0 = (−x, −y, −z)  aka (0,0,0) corner
///   1 = (+x, −y, −z)  aka (1,0,0)
///   2 = (−x, +y, −z)  aka (0,1,0)
///   3 = (+x, +y, −z)  aka (1,1,0)
///   4 = (−x, −y, +z)  aka (0,0,1)
///   5 = (+x, −y, +z)  aka (1,0,1)
///   6 = (−x, +y, +z)  aka (0,1,1)
///   7 = (+x, +y, +z)  aka (1,1,1)
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    Leaf(CellState),
    Interior {
        level: u32,
        children: [NodeId; 8],
        /// Population count (number of non-empty cells). Used for quick emptiness checks.
        population: u64,
    },
}

impl Node {
    #[allow(dead_code)]
    pub fn level(&self) -> u32 {
        match self {
            Node::Leaf(_) => 0,
            Node::Interior { level, .. } => *level,
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        match self {
            Node::Leaf(s) => *s == 0,
            Node::Interior { population, .. } => *population == 0,
        }
    }

    /// Side length of this node in cells.
    #[allow(dead_code)]
    pub fn side_len(&self) -> u64 {
        1u64 << self.level()
    }
}

/// Octant index helper: given (x, y, z) each 0 or 1, return octant index.
#[inline]
pub fn octant_index(x: u32, y: u32, z: u32) -> usize {
    (x + y * 2 + z * 4) as usize
}

/// Inverse: octant index to (x, y, z) each 0 or 1.
#[inline]
pub fn octant_coords(idx: usize) -> (u32, u32, u32) {
    (idx as u32 & 1, (idx as u32 >> 1) & 1, (idx as u32 >> 2) & 1)
}
