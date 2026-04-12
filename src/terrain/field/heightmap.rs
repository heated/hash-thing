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

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic field with known bounds for testing classify_box.
    /// base_y=32, amplitude=8 → surface ∈ [24, 40].
    fn test_field() -> HeightmapField {
        HeightmapField {
            seed: 42,
            base_y: 32.0,
            amplitude: 8.0,
            wavelength: 16.0,
            octaves: 2,
        }
    }

    #[test]
    fn sky_box_returns_air() {
        let f = test_field();
        // surface_max = 32 + 8 = 40, SURFACE_MARGIN = 2 → threshold = 42.
        // Box at y_min=42: entirely above maximum surface.
        assert_eq!(
            f.classify_box([0, 42, 0], 2),
            Some(AIR),
            "box above surface_max + margin must be AIR",
        );
    }

    #[test]
    fn sky_box_just_below_threshold_returns_none() {
        let f = test_field();
        // Box at y_min=41, size=4 (size_log2=2) → y in [41, 45).
        // Threshold is 42 (surface_max + margin). y_min=41 < 42 → not provably AIR.
        assert_eq!(
            f.classify_box([0, 41, 0], 2),
            None,
            "box with y_min below sky threshold must not short-circuit",
        );
    }

    #[test]
    fn deep_stone_box_returns_stone() {
        let f = test_field();
        // surface_min = 32 - 8 = 24, DEPTH_MARGIN = 4 → threshold = 20.
        // Box y_max must be ≤ 20. With size_log2=2 (size=4): origin y=16 → y_max=20.
        assert_eq!(
            f.classify_box([0, 16, 0], 2),
            Some(STONE),
            "box fully below surface_min - DEPTH_MARGIN must be STONE",
        );
    }

    #[test]
    fn deep_stone_box_just_above_threshold_returns_none() {
        let f = test_field();
        // origin y=17, size=4 → y_max=21 > 20 → not provably all STONE.
        assert_eq!(
            f.classify_box([0, 17, 0], 2),
            None,
            "box with y_max above deep-stone threshold must not short-circuit",
        );
    }

    #[test]
    fn surface_straddling_box_returns_none() {
        let f = test_field();
        // Box around y=32 (base_y) — clearly straddles the surface.
        assert_eq!(
            f.classify_box([0, 30, 0], 3),
            None,
            "box straddling the surface band must return None",
        );
    }

    #[test]
    fn large_sky_box_collapses() {
        let f = test_field();
        // size_log2=5 (size=32), origin y=48 → y_min=48 ≥ 42 → AIR.
        assert_eq!(
            f.classify_box([0, 48, 0], 5),
            Some(AIR),
            "large box well above surface must collapse to AIR",
        );
    }

    #[test]
    fn large_deep_box_collapses() {
        let f = test_field();
        // size_log2=3 (size=8), origin y=0 → y_max=8 ≤ 20 → STONE.
        assert_eq!(
            f.classify_box([0, 0, 0], 3),
            Some(STONE),
            "large box well below surface must collapse to STONE",
        );
    }

    #[test]
    fn sample_agrees_with_surface_y() {
        let f = test_field();
        // A point well above the surface should be AIR.
        let high = f.sample([10, 50, 10]);
        assert_eq!(high, AIR, "point at y=50 must be AIR");
        // A point well below should be STONE.
        let deep = f.sample([10, 0, 10]);
        assert_eq!(deep, STONE, "point at y=0 must be STONE");
    }
}
