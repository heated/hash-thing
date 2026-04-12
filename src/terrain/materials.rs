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

use crate::octree::{Cell, CellState};

pub const AIR: CellState = 0;
/// Material 1, metadata 0 — encoded via `Cell::pack(1, 0)`.
pub const STONE: CellState = Cell::pack(1, 0).raw();
/// Material 2, metadata 0.
pub const DIRT: CellState = Cell::pack(2, 0).raw();
/// Material 3, metadata 0.
pub const GRASS: CellState = Cell::pack(3, 0).raw();

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
