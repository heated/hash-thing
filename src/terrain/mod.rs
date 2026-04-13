//! Procedural terrain generation. v1: heightfield seeder built on the
//! `WorldGen` trait + recursive direct-octree builder.
//!
//! The trait abstraction is the load-bearing part. Infinite
//! worlds (3fq.4) and HashDAG edits all reuse the
//! recursion + proof-based collapse + interning shape that lives in
//! `terrain::gen`.

pub mod field;
pub mod gen;
pub mod materials;
pub mod noise;

pub use field::{
    ComposedWorldGen, DensityField, GyroidField, GyroidParams, HeightmapField, TerrainBlendField,
    TerrainSurface, WorldGen,
};
pub use gen::{gen_region, probe_sample_ns, GenStats};

/// Parameters for the v1 heightmap seeder. Lifted into a struct so the
/// runtime can swap them without touching the builder.
#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    pub seed: u64,
    pub base_y: f32,
    pub amplitude: f32,
    pub wavelength: f32,
    pub octaves: u32,
    /// Sea level. Air cells below this y become water. `None` disables water.
    pub sea_level: Option<f32>,
}

impl Default for TerrainParams {
    fn default() -> Self {
        // Sensible 64^3 defaults: surface near mid-world, ±8 amplitude,
        // 24-cell wavelength, 4 octaves of value noise.
        Self {
            seed: 1,
            base_y: 32.0,
            amplitude: 8.0,
            wavelength: 24.0,
            octaves: 4,
            sea_level: Some(28.0),
        }
    }
}

impl TerrainParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.wavelength.is_finite() || self.wavelength <= 0.0 {
            return Err("wavelength must be finite and > 0");
        }
        if self.octaves == 0 {
            return Err("octaves must be >= 1");
        }
        if !self.amplitude.is_finite() || self.amplitude < 0.0 {
            return Err("amplitude must be finite and >= 0");
        }
        if !self.base_y.is_finite() {
            return Err("base_y must be finite");
        }
        Ok(())
    }

    /// Scale-aware defaults: terrain params proportional to the world size.
    /// At level 6 (64³) this matches `Default`; at larger levels, surface
    /// height, amplitude, wavelength, and sea level scale so the terrain
    /// fills the world naturally.
    pub fn for_level(level: u32) -> Self {
        let side = (1u64 << level) as f32;
        let scale = side / 64.0;
        Self {
            seed: 1,
            base_y: 32.0 * scale,
            amplitude: 8.0 * scale,
            wavelength: 24.0 * scale,
            octaves: 4,
            sea_level: Some(28.0 * scale),
        }
    }

    pub fn to_heightmap(self) -> HeightmapField {
        HeightmapField {
            seed: self.seed,
            base_y: self.base_y,
            amplitude: self.amplitude,
            wavelength: self.wavelength,
            octaves: self.octaves,
            sea_level: self.sea_level,
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn for_level_matches_default_at_level_6() {
        let scaled = TerrainParams::for_level(6);
        let default = TerrainParams::default();
        assert_eq!(scaled.base_y, default.base_y);
        assert_eq!(scaled.amplitude, default.amplitude);
        assert_eq!(scaled.wavelength, default.wavelength);
        assert_eq!(scaled.sea_level, default.sea_level);
        assert_eq!(scaled.octaves, default.octaves);
    }

    #[test]
    fn for_level_scales_proportionally() {
        let l6 = TerrainParams::for_level(6);
        let l12 = TerrainParams::for_level(12);
        let ratio = (1u64 << 12) as f32 / (1u64 << 6) as f32; // 64x
        assert!((l12.base_y - l6.base_y * ratio).abs() < 0.01);
        assert!((l12.amplitude - l6.amplitude * ratio).abs() < 0.01);
        assert!((l12.wavelength - l6.wavelength * ratio).abs() < 0.01);
    }

    #[test]
    fn for_level_validates() {
        for level in [3, 6, 9, 12, 16] {
            assert!(TerrainParams::for_level(level).validate().is_ok());
        }
    }

    #[test]
    fn validate_rejects_non_finite_wavelength() {
        for wavelength in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let params = TerrainParams {
                wavelength,
                ..Default::default()
            };
            assert_eq!(
                params.validate(),
                Err("wavelength must be finite and > 0"),
                "unexpected validation result for wavelength={wavelength:?}",
            );
        }
    }

    #[test]
    fn validate_rejects_non_positive_wavelength() {
        for wavelength in [0.0, -1.0] {
            let params = TerrainParams {
                wavelength,
                ..Default::default()
            };
            assert_eq!(
                params.validate(),
                Err("wavelength must be finite and > 0"),
                "unexpected validation result for wavelength={wavelength:?}",
            );
        }
    }

    #[test]
    fn to_heightmap_copies_shape_fields() {
        let params = TerrainParams {
            seed: 42,
            base_y: 12.5,
            amplitude: 3.25,
            wavelength: 19.0,
            octaves: 6,
            sea_level: Some(7.5),
        };

        let field = params.to_heightmap();
        assert_eq!(field.seed, 42);
        assert_eq!(field.base_y, 12.5);
        assert_eq!(field.amplitude, 3.25);
        assert_eq!(field.wavelength, 19.0);
        assert_eq!(field.octaves, 6);
        assert_eq!(field.sea_level, Some(7.5));
    }
}
