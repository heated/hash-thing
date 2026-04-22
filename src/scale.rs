//! World-scale knob.
//!
//! `CELLS_PER_METER` is the single constant that converts physical
//! quantities stated in meters (movement speed, gravity, bounding boxes,
//! interact ranges, particle physics) into the cell units the simulation
//! actually runs on. Bumping this value shrinks cells further without
//! changing player / entity feel — a 1 m ledge is still a 1 m ledge; a
//! 9 m/s sprint is still a 9 m/s sprint.
//!
//! The default world size (`DEFAULT_VOLUME_SIZE` in `src/main.rs`) should
//! move in lockstep so the world's size in meters stays fixed while the
//! resolution increases. See hash-thing-69cq for the first bump
//! (1.0 → 4.0 cells/m, 2048³ → 8192³).

/// Cells per world-meter. Physical quantities are expressed in meters
/// and multiplied through this factor to land in cell units.
pub const CELLS_PER_METER: f64 = 4.0;

/// Convenience convertor: meters → cells.
#[inline]
pub const fn meters_to_cells(m: f64) -> f64 {
    m * CELLS_PER_METER
}
