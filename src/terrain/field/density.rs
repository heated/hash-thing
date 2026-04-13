//! `DensityField` — 3D noise density with height bias.
//!
//! `density(x,y,z) = fractal_3d(x/wl, y/wl, z/wl, seed, octaves) - height_bias(y)`
//!
//! The height bias is a linear ramp centered on `base_y`:
//!   `height_bias(y) = 0.5 + (y - base_y) / falloff`
//!
//! This produces:
//! - Ground level (y ≈ base_y): ~50% solid, interesting terrain surface
//! - High altitude: mostly air with occasional floating islands
//! - Deep underground: mostly solid with natural caverns
//!
//! `classify` uses tight analytic bounds: fractal_3d ∈ [0, 1] and
//! height_bias is monotonically increasing in y, so density over a box
//! is bounded by `[0 - bias_max, 1 - bias_min]`. If the entire range
//! is positive → solid; if entirely negative → air.

use super::WorldGen;
use crate::octree::CellState;
use crate::terrain::materials::{material_from_depth, AIR, WATER};
use crate::terrain::noise::fractal_3d;

#[derive(Clone, Copy, Debug)]
pub struct DensityField {
    pub seed: u64,
    pub base_y: f32,
    /// Vertical distance over which the terrain transitions from fully
    /// solid to fully air. Larger values → gentler transition, more
    /// floating islands above and more caverns below.
    pub falloff: f32,
    /// Noise wavelength in cells. Controls feature scale.
    pub wavelength: f32,
    pub octaves: u32,
    /// Sea level: air cells below this y become water. `None` disables.
    pub sea_level: Option<f32>,
}

impl DensityField {
    /// Height bias at a given y. Monotonically increasing.
    /// Returns 0.5 at base_y, 0 at base_y - falloff/2, 1 at base_y + falloff/2.
    #[inline]
    fn height_bias(&self, y: f32) -> f32 {
        0.5 + (y - self.base_y) / self.falloff
    }

    /// Density at a world point. Positive → solid, negative → air.
    #[inline]
    fn density(&self, x: f32, y: f32, z: f32) -> f32 {
        let noise = fractal_3d(
            x / self.wavelength,
            y / self.wavelength,
            z / self.wavelength,
            self.seed,
            self.octaves,
        );
        noise - self.height_bias(y)
    }

    /// Approximate depth below the "surface" for material selection.
    /// Uses base_y as the reference surface — cells near base_y get
    /// grass/dirt, deeper cells get stone.
    #[inline]
    fn approx_depth(&self, y: f32) -> f32 {
        self.base_y - y
    }
}

impl WorldGen for DensityField {
    fn sample(&self, p: [i64; 3]) -> CellState {
        let d = self.density(p[0] as f32, p[1] as f32, p[2] as f32);
        if d <= 0.0 {
            // Air — check sea level.
            if let Some(sl) = self.sea_level {
                if (p[1] as f32) < sl {
                    return WATER;
                }
            }
            return AIR;
        }
        // Solid — pick material by approximate depth.
        material_from_depth(self.approx_depth(p[1] as f32))
    }

    fn classify(&self, origin: [i64; 3], level: u32) -> Option<CellState> {
        let size = 1i64 << level;
        let y_min = origin[1] as f32;
        let y_max = (origin[1] + size) as f32;

        // height_bias is monotonically increasing in y.
        let bias_min = self.height_bias(y_min);
        let bias_max = self.height_bias(y_max - 1.0); // inclusive upper y

        // fractal_3d ∈ [0, 1], so:
        //   density_min = 0 - bias_max
        //   density_max = 1 - bias_min
        let density_min = -bias_max;
        let density_max = 1.0 - bias_min;

        // Margin for FP rounding (same rationale as heightmap's SURFACE_MARGIN).
        const MARGIN: f32 = 0.01;

        if density_max < -MARGIN {
            // Entire box is air (density < 0 everywhere).
            if let Some(sl) = self.sea_level {
                if y_max <= sl {
                    return Some(WATER);
                }
                if y_min < sl {
                    return None; // straddles sea level
                }
            }
            return Some(AIR);
        }

        if density_min > MARGIN {
            // Entire box is solid. Check if material is uniform.
            // Deep enough that all cells map to STONE?
            let shallowest_depth = self.approx_depth(y_max - 1.0);
            if shallowest_depth >= 4.0 {
                return Some(material_from_depth(shallowest_depth));
            }
            // Near the surface band — materials vary, can't collapse.
            return None;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::materials::{GRASS, STONE};

    fn test_field() -> DensityField {
        DensityField {
            seed: 42,
            base_y: 32.0,
            falloff: 32.0,
            wavelength: 16.0,
            octaves: 3,
            sea_level: None,
        }
    }

    #[test]
    fn sample_deterministic() {
        let f = test_field();
        let a = f.sample([10, 30, 10]);
        let b = f.sample([10, 30, 10]);
        assert_eq!(a, b);
    }

    #[test]
    fn high_altitude_is_air() {
        let f = test_field();
        // Well above base_y + falloff/2 = 48. height_bias >> 1, so density < 0.
        for x in 0..8 {
            for z in 0..8 {
                assert_eq!(f.sample([x, 100, z]), AIR, "point at y=100 should be AIR");
            }
        }
    }

    #[test]
    fn deep_underground_is_stone() {
        let f = test_field();
        // Well below base_y - falloff/2 = 16. height_bias << 0, so density > 0.
        // Also deep enough that material_from_depth returns STONE.
        for x in 0..8 {
            for z in 0..8 {
                assert_eq!(
                    f.sample([x, -100, z]),
                    STONE,
                    "point at y=-100 should be STONE"
                );
            }
        }
    }

    #[test]
    fn surface_has_mixed_materials() {
        let f = test_field();
        let mut air_count = 0u32;
        let mut solid_count = 0u32;
        // Sample around base_y where we expect a mix.
        for x in 0..32 {
            for z in 0..32 {
                let s = f.sample([x, 32, z]);
                if s == AIR {
                    air_count += 1;
                } else {
                    solid_count += 1;
                }
            }
        }
        assert!(air_count > 0, "expected some air near surface");
        assert!(solid_count > 0, "expected some solid near surface");
    }

    #[test]
    fn classify_high_box_returns_air() {
        let f = test_field();
        // y_min = 100, bias_min = 0.5 + (100-32)/32 = 2.625
        // density_max = 1 - 2.625 = -1.625 < 0 → AIR
        assert_eq!(f.classify([0, 100, 0], 3), Some(AIR));
    }

    #[test]
    fn classify_deep_box_returns_stone() {
        let f = test_field();
        // y_max = -92, bias_max = 0.5 + (-93-32)/32 = -3.40625
        // density_min = -(-3.40625) = 3.40625 > 0 → solid
        // shallowest_depth = base_y - (y_max - 1) = 32 - (-93) = 125 ≥ 4 → STONE
        assert_eq!(f.classify([0, -100, 0], 3), Some(STONE));
    }

    #[test]
    fn classify_surface_returns_none() {
        let f = test_field();
        // Box around base_y where density crosses zero.
        assert_eq!(f.classify([0, 28, 0], 3), None);
    }

    #[test]
    fn classify_consistency_with_sample() {
        let f = test_field();
        // Check boxes at level 3 (8×8×8) over a 64³ grid.
        for oz in (0..64).step_by(8) {
            for oy in (-32i64..32).step_by(8) {
                for ox in (0..64).step_by(8) {
                    if let Some(state) = f.classify([ox, oy, oz], 3) {
                        for z in oz..oz + 8 {
                            for y in oy..oy + 8 {
                                for x in ox..ox + 8 {
                                    let s = f.sample([x, y, z]);
                                    assert_eq!(
                                        s, state,
                                        "classify={state} but sample({x},{y},{z})={s}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sea_level_fills_air_with_water() {
        let mut f = test_field();
        f.sea_level = Some(120.0); // above everything
                                   // High altitude air below sea level → WATER
        let s = f.sample([10, 80, 10]);
        assert_eq!(s, WATER);
        // Very high altitude above sea level → AIR
        let s = f.sample([10, 130, 10]);
        assert_eq!(s, AIR);
    }

    #[test]
    fn gen_region_produces_valid_tree() {
        use crate::octree::NodeStore;
        use crate::terrain::gen::gen_region;

        let f = test_field();
        let mut store = NodeStore::new();
        let (root, stats) = gen_region(&mut store, &f, [0, 0, 0], 6);

        // Must have some collapses (sky and deep stone).
        assert!(
            stats.total_collapses() > 0,
            "expected proof-based collapses: {stats:?}"
        );
        // Population should be nonzero (surface band has solid cells).
        assert!(store.population(root) > 0, "expected nonzero population");
    }

    #[test]
    fn classify_returns_grass_never() {
        // classify should never return GRASS — near-surface solid boxes
        // have mixed materials (grass/dirt/stone) so they can't collapse.
        let f = test_field();
        for oy in -64i64..64 {
            for ox in (0..64).step_by(8) {
                for oz in (0..64).step_by(8) {
                    if let Some(state) = f.classify([ox, oy, oz], 3) {
                        assert_ne!(state, GRASS, "classify should not return GRASS at y={oy}");
                    }
                }
            }
        }
    }
}
