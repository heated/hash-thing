//! World-scale knob.
//!
//! `CELLS_PER_METER` is the single constant that converts physical
//! quantities stated in meters (movement speed, gravity, bounding boxes,
//! interact ranges, particle physics) into the cell units the simulation
//! actually runs on. Bumping this value shrinks cells further without
//! changing player / entity feel — a 1 m ledge is still a 1 m ledge; a
//! 9 m/s sprint is still a 9 m/s sprint.
//!
//! The default world side (`DEFAULT_VOLUME_SIZE`) is derived from the
//! integer scale knob and a fixed `DEFAULT_WORLD_METERS`, so the world
//! stays the same size in meters when `CELLS_PER_METER_INT` bumps. See
//! hash-thing-69cq for the first bump (1.0 → 4.0 cells/m, 2048³ → 8192³).
//!
//! Addressability: `WorldCoord` is `i64`, so any plausible bump of this
//! knob stays far inside the coordinate range — at
//! `CELLS_PER_METER_INT = 16` a 2048-meter world spans 32768 cells per
//! axis, well under `i64::MAX`.

/// Canonical integer scale knob. `DEFAULT_VOLUME_SIZE` pins
/// `DEFAULT_WORLD_METERS * CELLS_PER_METER_INT` to a power-of-two at
/// compile time, so pick both values accordingly.
pub const CELLS_PER_METER_INT: u32 = 4;

/// Cells per world-meter, f64 form for physics paths. Derived from the
/// integer knob — do not write an independent literal here.
pub const CELLS_PER_METER: f64 = CELLS_PER_METER_INT as f64;

/// Default world side in meters. World-in-meters is the invariant we hold
/// fixed across scale-knob bumps.
pub const DEFAULT_WORLD_METERS: u32 = 2048;

/// Default cubic volume side in cells. Derived — bump `CELLS_PER_METER_INT`
/// or `DEFAULT_WORLD_METERS` and this follows.
pub const DEFAULT_VOLUME_SIZE: u32 = DEFAULT_WORLD_METERS * CELLS_PER_METER_INT;

const _: () = assert!(
    DEFAULT_VOLUME_SIZE.is_power_of_two(),
    "DEFAULT_VOLUME_SIZE must stay a power-of-two world side; \
     pick DEFAULT_WORLD_METERS and CELLS_PER_METER_INT so their product is pow2",
);

/// Distance from the active region's edge (in meters) before
/// `ensure_region` grows outward. 8 m preserves responsive streaming at
/// any scale.
pub const GROWTH_MARGIN_METERS: f64 = 8.0;

/// Growth margin in cells (derived from `GROWTH_MARGIN_METERS`).
pub const GROWTH_MARGIN: f64 = GROWTH_MARGIN_METERS * CELLS_PER_METER;
