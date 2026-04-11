//! Temporary material constants for v1 terrain.
//!
//! 1v0.1 (16-bit tagged cell encoding) will reshape `CellState` and the
//! material registry. **The AIR == 0 invariant must survive that change.**
//! It is load-bearing for:
//!
//! - `NodeStore` population accounting (`Leaf(0).population() == 0`,
//!   see `src/octree/store.rs`)
//! - `NodeStore::flatten_into` empty-subtree skipping
//! - SVDAG empty-subtree skipping (see `src/render/svdag.rs`)
//! - The hash-cons compression of the sky (an entire 32^3 of AIR is one
//!   canonical NodeId reused everywhere)
//!
//! When 1v0.1 lands, the `CellState(0) -> AIR` mapping must be preserved.
//! A test should fail loudly if it ever changes.

use crate::octree::CellState;

pub const AIR: CellState = 0;
pub const STONE: CellState = 1;
pub const DIRT: CellState = 2;
pub const GRASS: CellState = 3;

/// Map "depth below the surface y" to a material.
///
/// `depth` is `surface_y - cell_y`. Above the surface (`depth < 0`) is `AIR`.
/// At the surface (`depth == 0`) is `GRASS`. Within `DEPTH_MARGIN` is `DIRT`.
/// Deeper than that is `STONE`.
#[inline]
pub fn material_from_depth(depth: f32) -> CellState {
    if depth < 0.0 {
        AIR
    } else if depth < 1.0 {
        GRASS
    } else if depth < 4.0 {
        DIRT
    } else {
        STONE
    }
}
