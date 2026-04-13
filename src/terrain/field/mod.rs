//! `RegionField` trait — the abstraction every direct-octree generator
//! recurses against.
//!
//! Two methods:
//!
//! - `sample(point) -> CellState`: exact value at one integer cell.
//! - `classify_box(origin, size_log2) -> Option<CellState>`: conservative
//!   bound. Must return `Some(state)` only if the entire box is provably
//!   uniform — never `Some` for a mixed box. Heuristic ("probably uniform")
//!   collapse is not allowed here. Heuristic LOD belongs to a different layer.
//!
//! The recursive builder in `terrain::gen` takes any `&impl RegionField` and
//! emits a canonical `NodeId` without ever flattening to a grid. Heightmap,
//! caves (3fq.2) and SDF impls are all sibling structs in
//! this module tree.

use crate::octree::CellState;

pub trait RegionField {
    /// Exact material at a single integer cell.
    fn sample(&self, point: [i64; 3]) -> CellState;

    /// Conservative box classification over `[origin, origin + 2^size_log2)`.
    /// Must return `Some(state)` only if the entire box is provably uniform.
    /// Returns `None` for mixed-or-unknown.
    fn classify_box(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState>;
}

pub mod const_field;
pub mod half_space;
pub mod heightmap;
pub mod lattice;

pub use heightmap::HeightmapField;
pub use lattice::LatticeField;
