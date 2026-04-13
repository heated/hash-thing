//! Generic scalar-field materializer for organic megastructures.
//!
//! The load-bearing split is:
//! - a scalar field (`ScalarField`) that defines the shape and its bounds
//! - a reusable `ImplicitField` `WorldGen` that handles region clipping,
//!   low-frequency coordinate warping, and proof-based box collapse
//!
//! That keeps "what shape is this?" swappable without rewriting the
//! octree-facing generation logic.

use super::WorldGen;
use crate::octree::CellState;
use crate::terrain::materials::AIR;
use crate::terrain::noise::fractal_3d;

#[derive(Clone, Copy, Debug)]
pub struct ScalarBand {
    pub min: f32,
    pub max: f32,
}

impl ScalarBand {
    #[inline]
    pub fn contains(&self, value: f32) -> bool {
        value >= self.min && value <= self.max
    }

    #[inline]
    pub fn classify(&self, bounds: (f32, f32), solid: CellState) -> Option<CellState> {
        let (min_value, max_value) = bounds;
        if max_value < self.min || min_value > self.max {
            return Some(AIR);
        }
        if min_value >= self.min && max_value <= self.max {
            return Some(solid);
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NoiseWarp {
    /// Maximum coordinate displacement in world cells.
    pub amplitude: f32,
    /// World-cell wavelength of the low-frequency warp noise.
    pub wavelength: f32,
    pub octaves: u32,
}

impl Default for NoiseWarp {
    fn default() -> Self {
        Self {
            amplitude: 0.0,
            wavelength: 1.0,
            octaves: 1,
        }
    }
}

impl NoiseWarp {
    #[inline]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.amplitude < 0.0 || !self.amplitude.is_finite() {
            return Err("warp amplitude must be finite and >= 0");
        }
        if self.amplitude > 0.0 && (!self.wavelength.is_finite() || self.wavelength <= 0.0) {
            return Err("warp wavelength must be finite and > 0");
        }
        if self.amplitude > 0.0 && self.octaves == 0 {
            return Err("warp octaves must be >= 1 when warp is enabled");
        }
        Ok(())
    }

    #[inline]
    fn offset(&self, point: [i64; 3], seed: u64) -> f32 {
        if self.amplitude == 0.0 {
            return 0.0;
        }
        let wavelength = self.wavelength;
        let noise = fractal_3d(
            point[0] as f32 / wavelength,
            point[1] as f32 / wavelength,
            point[2] as f32 / wavelength,
            seed,
            self.octaves,
        );
        self.amplitude * (noise * 2.0 - 1.0)
    }
}

pub trait ScalarField {
    fn sample_scalar(&self, point: [f32; 3]) -> f32;
    fn scalar_bounds(&self, lo: [f32; 3], hi: [f32; 3]) -> (f32, f32);
}

#[derive(Clone, Debug)]
pub struct ImplicitField<S> {
    pub seed: u64,
    pub lo: [i64; 3],
    pub hi: [i64; 3],
    pub shape: S,
    pub band: ScalarBand,
    pub solid: CellState,
    pub warp: NoiseWarp,
}

impl<S> ImplicitField<S> {
    #[inline]
    fn contains_world_point(&self, point: [i64; 3]) -> bool {
        point[0] >= self.lo[0]
            && point[0] < self.hi[0]
            && point[1] >= self.lo[1]
            && point[1] < self.hi[1]
            && point[2] >= self.lo[2]
            && point[2] < self.hi[2]
    }

    #[inline]
    fn warp_seed(&self, axis: usize) -> u64 {
        match axis {
            0 => self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            1 => self.seed.wrapping_add(0xC2B2_AE3D_27D4_EB4F),
            _ => self.seed.wrapping_add(0x1656_67B1_9E37_79F9),
        }
    }

    #[inline]
    fn warped_point(&self, point: [i64; 3]) -> [f32; 3] {
        [
            point[0] as f32 + self.warp.offset(point, self.warp_seed(0)),
            point[1] as f32 + self.warp.offset(point, self.warp_seed(1)),
            point[2] as f32 + self.warp.offset(point, self.warp_seed(2)),
        ]
    }

    #[inline]
    fn warped_bounds(&self, origin: [i64; 3], size_log2: u32) -> ([f32; 3], [f32; 3]) {
        let size = 1i64 << size_log2;
        let hi_inclusive = [
            origin[0] + size - 1,
            origin[1] + size - 1,
            origin[2] + size - 1,
        ];
        let expand = self.warp.amplitude;
        (
            [
                origin[0] as f32 - expand,
                origin[1] as f32 - expand,
                origin[2] as f32 - expand,
            ],
            [
                hi_inclusive[0] as f32 + expand,
                hi_inclusive[1] as f32 + expand,
                hi_inclusive[2] as f32 + expand,
            ],
        )
    }
}

impl<S: ScalarField> WorldGen for ImplicitField<S> {
    fn sample(&self, point: [i64; 3]) -> CellState {
        if !self.contains_world_point(point) {
            return AIR;
        }
        let value = self.shape.sample_scalar(self.warped_point(point));
        if self.band.contains(value) {
            self.solid
        } else {
            AIR
        }
    }

    fn classify(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;
        let box_hi = [origin[0] + size, origin[1] + size, origin[2] + size];

        if box_hi[0] <= self.lo[0]
            || origin[0] >= self.hi[0]
            || box_hi[1] <= self.lo[1]
            || origin[1] >= self.hi[1]
            || box_hi[2] <= self.lo[2]
            || origin[2] >= self.hi[2]
        {
            return Some(AIR);
        }

        if origin[0] < self.lo[0]
            || box_hi[0] > self.hi[0]
            || origin[1] < self.lo[1]
            || box_hi[1] > self.hi[1]
            || origin[2] < self.lo[2]
            || box_hi[2] > self.hi[2]
        {
            return None;
        }

        let (lo, hi) = self.warped_bounds(origin, size_log2);
        self.band
            .classify(self.shape.scalar_bounds(lo, hi), self.solid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::materials::STONE;

    #[derive(Clone, Copy, Debug)]
    struct ConstantShape(f32);

    impl ScalarField for ConstantShape {
        fn sample_scalar(&self, _point: [f32; 3]) -> f32 {
            self.0
        }

        fn scalar_bounds(&self, _lo: [f32; 3], _hi: [f32; 3]) -> (f32, f32) {
            (self.0, self.0)
        }
    }

    fn solid_field(value: f32) -> ImplicitField<ConstantShape> {
        ImplicitField {
            seed: 7,
            lo: [0, 0, 0],
            hi: [8, 8, 8],
            shape: ConstantShape(value),
            band: ScalarBand {
                min: -0.5,
                max: 0.5,
            },
            solid: STONE,
            warp: NoiseWarp::default(),
        }
    }

    #[test]
    fn scalar_band_classifies_air_solid_and_mixed_ranges() {
        let band = ScalarBand {
            min: -0.25,
            max: 0.25,
        };

        assert_eq!(band.classify((-1.0, -0.5), STONE), Some(AIR));
        assert_eq!(band.classify((-0.1, 0.2), STONE), Some(STONE));
        assert_eq!(band.classify((-0.5, 0.5), STONE), None);
    }

    #[test]
    fn noise_warp_validate_rejects_invalid_enabled_params() {
        assert_eq!(
            NoiseWarp {
                amplitude: -1.0,
                wavelength: 4.0,
                octaves: 1,
            }
            .validate(),
            Err("warp amplitude must be finite and >= 0")
        );
        assert_eq!(
            NoiseWarp {
                amplitude: 1.0,
                wavelength: 0.0,
                octaves: 1,
            }
            .validate(),
            Err("warp wavelength must be finite and > 0")
        );
        assert_eq!(
            NoiseWarp {
                amplitude: 1.0,
                wavelength: 4.0,
                octaves: 0,
            }
            .validate(),
            Err("warp octaves must be >= 1 when warp is enabled")
        );
    }

    #[test]
    fn noise_warp_validate_allows_disabled_warp() {
        let warp = NoiseWarp {
            amplitude: 0.0,
            wavelength: 0.0,
            octaves: 0,
        };
        assert_eq!(warp.validate(), Ok(()));
    }

    #[test]
    fn sample_outside_world_bounds_returns_air() {
        let field = solid_field(0.0);
        assert_eq!(field.sample([-1, 0, 0]), AIR);
        assert_eq!(field.sample([0, 8, 0]), AIR);
        assert_eq!(field.sample([0, 0, 8]), AIR);
    }

    #[test]
    fn sample_inside_world_uses_band_membership() {
        assert_eq!(solid_field(0.0).sample([1, 1, 1]), STONE);
        assert_eq!(solid_field(2.0).sample([1, 1, 1]), AIR);
    }

    #[test]
    fn classify_box_fully_outside_world_returns_air() {
        let field = solid_field(0.0);
        assert_eq!(field.classify([-4, 0, 0], 2), Some(AIR));
        assert_eq!(field.classify([8, 0, 0], 0), Some(AIR));
    }

    #[test]
    fn classify_box_partially_overlapping_world_forces_recursion() {
        let field = solid_field(0.0);
        assert_eq!(field.classify([-1, 0, 0], 1), None);
        assert_eq!(field.classify([7, 0, 0], 1), None);
    }

    #[test]
    fn classify_inside_world_collapses_to_solid_or_air() {
        assert_eq!(solid_field(0.0).classify([0, 0, 0], 2), Some(STONE));
        assert_eq!(solid_field(2.0).classify([0, 0, 0], 2), Some(AIR));
    }
}
