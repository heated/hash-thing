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

#[cfg(test)]
mod tests {
    //! Pure-helper tests for `node.rs`: the octant index/coords bijection
    //! and `Node` level/emptiness/side_len accessors.
    //!
    //! These helpers are load-bearing contracts. `octant_index` is called
    //! on every subtree build in `store.rs` (set_cell, from_flat_recursive,
    //! interior construction). `octant_coords` is called on every store
    //! traversal (flatten_into, population_recursive). A silent off-by-one
    //! in either direction produces rotated or mirrored children
    //! everywhere — higher-level tests (raycaster visuals, compaction
    //! invariants) would eventually notice, but slowly and expensively.
    //! Direct unit coverage fails in microseconds and pinpoints the break.
    use super::*;

    /// Exhaustive mapping of (x,y,z) ∈ {0,1}³ to the expected octant index,
    /// following the documented layout `idx = x + y*2 + z*4`.
    #[test]
    fn octant_index_matches_formula() {
        assert_eq!(octant_index(0, 0, 0), 0);
        assert_eq!(octant_index(1, 0, 0), 1);
        assert_eq!(octant_index(0, 1, 0), 2);
        assert_eq!(octant_index(1, 1, 0), 3);
        assert_eq!(octant_index(0, 0, 1), 4);
        assert_eq!(octant_index(1, 0, 1), 5);
        assert_eq!(octant_index(0, 1, 1), 6);
        assert_eq!(octant_index(1, 1, 1), 7);
    }

    /// `octant_index(octant_coords(i)) == i` for every valid index.
    /// Catches a `>> 1` vs `>> 2` typo in `octant_coords`.
    #[test]
    fn octant_coords_is_inverse_of_octant_index() {
        for i in 0..8 {
            let (x, y, z) = octant_coords(i);
            assert_eq!(octant_index(x, y, z), i, "roundtrip failed for idx {i}");
        }
    }

    /// `octant_coords(octant_index(x,y,z)) == (x,y,z)` for every input.
    /// Both directions of the roundtrip because they're separately maintained.
    #[test]
    fn octant_roundtrip_coordinates() {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    let i = octant_index(x, y, z);
                    let (rx, ry, rz) = octant_coords(i);
                    assert_eq!(
                        (rx, ry, rz),
                        (x, y, z),
                        "roundtrip failed for ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// Leaves are always level 0, regardless of cell state.
    #[test]
    fn leaf_level_is_zero() {
        assert_eq!(Node::Leaf(0).level(), 0);
        assert_eq!(Node::Leaf(5).level(), 0);
    }

    /// Interior nodes return the stored level unchanged.
    #[test]
    fn interior_level_echoes_field() {
        let n = Node::Interior {
            level: 3,
            children: [NodeId::EMPTY; 8],
            population: 0,
        };
        assert_eq!(n.level(), 3);
    }

    /// `Leaf(0)` is empty; any nonzero cell state is not. Pins the payload-
    /// aware emptiness semantics — a leaf with a live material is *not*
    /// empty even though the sentinel for "empty" happens to be 0.
    #[test]
    fn leaf_is_empty_iff_zero_state() {
        assert!(Node::Leaf(0).is_empty());
        assert!(!Node::Leaf(1).is_empty());
        assert!(!Node::Leaf(CellState::MAX).is_empty());
    }

    /// Interior emptiness is driven by `population`, not children. If this
    /// flips sign, compaction and DAG build both break.
    #[test]
    fn interior_is_empty_iff_population_zero() {
        let empty = Node::Interior {
            level: 2,
            children: [NodeId::EMPTY; 8],
            population: 0,
        };
        assert!(empty.is_empty());
        let full = Node::Interior {
            level: 2,
            children: [NodeId::EMPTY; 8],
            population: 1,
        };
        assert!(!full.is_empty());
    }

    /// `side_len()` is `1 << level` — pinning the documented 2^level.
    #[test]
    fn side_len_is_power_of_two() {
        assert_eq!(Node::Leaf(0).side_len(), 1);
        let l1 = Node::Interior {
            level: 1,
            children: [NodeId::EMPTY; 8],
            population: 0,
        };
        assert_eq!(l1.side_len(), 2);
        let l5 = Node::Interior {
            level: 5,
            children: [NodeId::EMPTY; 8],
            population: 0,
        };
        assert_eq!(l5.side_len(), 32);
        let l10 = Node::Interior {
            level: 10,
            children: [NodeId::EMPTY; 8],
            population: 0,
        };
        assert_eq!(l10.side_len(), 1024);
    }
}
