//! `HeightmapField` — the v1 terrain content. Surface height is a value-noise
//! fractal in `[base_y - amplitude, base_y + amplitude]`. Material is chosen
//! by depth below the surface (see `terrain::materials::material_from_depth`).
//!
//! `classify_box` is **proof-based only**: a y-band check that uses the
//! global noise bounds (not sampled corners) to declare entire boxes above
//! `surface_max + SURFACE_MARGIN` as `AIR`, and entire boxes below
//! `surface_min - DEPTH_MARGIN` as `STONE`. No corner-agreement heuristic.

use super::RegionField;
use crate::octree::CellState;
use crate::terrain::materials::{material_from_depth, AIR, STONE};
use crate::terrain::noise::fractal_2d;

/// Cells above the maximum possible surface before we trust the AIR
/// short-circuit. The margin absorbs a few ulps of FP rounding in
/// `fractal_2d`'s final `sum / total` divide — which, under adversarial
/// input, could push the computed `surface_y` microscopically above the
/// ideal `base_y + amplitude` bound. (At 64³ the `i64 → f32` casts on the
/// box coordinates themselves are exact, so that's not where the slop
/// comes from.) 2 cells of headroom is vastly more than needed and keeps
/// the proof robust against future noise-function changes.
pub const SURFACE_MARGIN: f32 = 2.0;

/// Cells below the minimum possible surface before we trust the STONE
/// short-circuit. Pinned to the dirt-band rule (`d < 4 → DIRT`) so the
/// deep-stone short-circuit is correct by construction: at `depth >= 4`,
/// `material_from_depth` returns `STONE`.
pub const DEPTH_MARGIN: f32 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct HeightmapField {
    pub seed: u64,
    pub base_y: f32,
    pub amplitude: f32,
    pub wavelength: f32,
    pub octaves: u32,
}

impl HeightmapField {
    /// Surface y at world `(x, z)`. Always in
    /// `[base_y - amplitude, base_y + amplitude]`.
    #[inline]
    pub fn surface_y(&self, x: f32, z: f32) -> f32 {
        let n = fractal_2d(
            x / self.wavelength,
            z / self.wavelength,
            self.seed,
            self.octaves,
        );
        // n ∈ [0, 1] (debug_assert in fractal_2d). Map to [-1, 1].
        let centered = n * 2.0 - 1.0;
        self.base_y + centered * self.amplitude
    }
}

impl RegionField for HeightmapField {
    fn sample(&self, p: [i64; 3]) -> CellState {
        let surface = self.surface_y(p[0] as f32, p[2] as f32);
        let depth = surface - p[1] as f32;
        material_from_depth(depth)
    }

    fn classify_box(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;
        let y_min = origin[1] as f32; // inclusive
        let y_max = (origin[1] + size) as f32; // exclusive

        // Conservative global heightmap bounds — NOT sampled.
        let surface_max = self.base_y + self.amplitude;
        let surface_min = self.base_y - self.amplitude;

        // Box fully above the highest possible surface (plus a margin):
        // every cell is at depth < 0, hence AIR.
        if y_min >= surface_max + SURFACE_MARGIN {
            return Some(AIR);
        }
        // Box fully below the lowest possible surface, with at least
        // `DEPTH_MARGIN` cells of clearance: every cell has depth >= 4,
        // hence STONE per `material_from_depth`.
        // y_max - 1 is the largest y in the box, so we need
        //   surface_min - (y_max - 1) >= DEPTH_MARGIN
        //   <=>  y_max <= surface_min - DEPTH_MARGIN + 1
        // We use the slightly looser y_max <= surface_min - DEPTH_MARGIN
        // to keep the comparison exclusive-friendly and conservative.
        if y_max <= surface_min - DEPTH_MARGIN {
            return Some(STONE);
        }
        None
    }
}
