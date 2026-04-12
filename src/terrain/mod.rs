//! Procedural terrain generation. v1: heightfield seeder built on the
//! `RegionField` trait + recursive direct-octree builder.
//!
//! The trait abstraction is the load-bearing part. Caves (3fq.2), dungeons
//! (3fq.3), infinite worlds (3fq.4), and HashDAG edits all reuse the
//! recursion + proof-based collapse + interning shape that lives in
//! `terrain::gen`.

pub mod caves;
pub mod field;
pub mod gen;
pub mod materials;
pub mod noise;

#[allow(unused_imports)]
pub use caves::{carve_caves, carve_caves_grid, CaveParams};
pub use field::HeightmapField;
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
    /// When `Some`, run the cave-CA post-pass with these params after
    /// heightmap generation. Default `None` keeps the baseline terrain
    /// path unchanged — tests and perf baselines don't see caves unless
    /// a caller opts in.
    pub caves: Option<CaveParams>,
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
            caves: None,
        }
    }
}

impl TerrainParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.wavelength <= 0.0 {
            return Err("wavelength must be > 0");
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

    pub fn to_heightmap(self) -> HeightmapField {
        HeightmapField {
            seed: self.seed,
            base_y: self.base_y,
            amplitude: self.amplitude,
            wavelength: self.wavelength,
            octaves: self.octaves,
        }
    }
}
