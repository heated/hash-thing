//! Warped gyroid megastructure field.
//!
//! The gyroid is implemented as a scalar field plus a reusable
//! `ImplicitField` materializer. Future experiments can swap the shape while
//! keeping region clipping, warping, and proof-based collapse unchanged.

use super::implicit::{ImplicitField, NoiseWarp, ScalarBand, ScalarField};
use super::WorldGen;
use crate::octree::CellState;
use crate::terrain::materials::STONE;

#[derive(Clone, Copy, Debug)]
struct GyroidShape {
    radians_per_cell: f32,
}

impl GyroidShape {
    #[inline]
    fn new(cell_size: f32) -> Self {
        Self {
            radians_per_cell: std::f32::consts::TAU / cell_size,
        }
    }
}

impl ScalarField for GyroidShape {
    #[inline]
    fn sample_scalar(&self, point: [f32; 3]) -> f32 {
        let x = point[0] * self.radians_per_cell;
        let y = point[1] * self.radians_per_cell;
        let z = point[2] * self.radians_per_cell;
        x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()
    }

    fn scalar_bounds(&self, lo: [f32; 3], hi: [f32; 3]) -> (f32, f32) {
        let phase = |axis: usize| {
            (
                lo[axis] * self.radians_per_cell,
                hi[axis] * self.radians_per_cell,
            )
        };
        let (x_lo, x_hi) = phase(0);
        let (y_lo, y_hi) = phase(1);
        let (z_lo, z_hi) = phase(2);

        let t1 = mul_bounds(sin_bounds(x_lo, x_hi), cos_bounds(y_lo, y_hi));
        let t2 = mul_bounds(sin_bounds(y_lo, y_hi), cos_bounds(z_lo, z_hi));
        let t3 = mul_bounds(sin_bounds(z_lo, z_hi), cos_bounds(x_lo, x_hi));
        (t1.0 + t2.0 + t3.0, t1.1 + t2.1 + t3.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GyroidParams {
    pub seed: u64,
    /// Lower corner of the active megastructure zone (inclusive).
    pub lo: [i64; 3],
    /// Upper corner of the active megastructure zone (exclusive).
    pub hi: [i64; 3],
    /// World cells per gyroid period. Larger means more spacious voids.
    pub cell_size: f32,
    /// Half-width of the solid band around the gyroid zero set.
    pub band_thickness: f32,
    pub warp: NoiseWarp,
    pub solid: CellState,
}

impl GyroidParams {
    pub fn for_world(level: u32, seed: u64) -> Self {
        let side = 1i64 << level;
        let margin = (side / 8).max(4);
        let active_side = (side - 2 * margin).max(24) as f32;
        let cell_size = (active_side / 2.5).max(18.0);
        Self {
            seed,
            lo: [margin, margin, margin],
            hi: [side - margin, side - margin, side - margin],
            cell_size,
            band_thickness: 0.22,
            warp: NoiseWarp {
                amplitude: cell_size * 0.18,
                wavelength: cell_size * 1.75,
                octaves: 3,
            },
            solid: STONE,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.lo[0] >= self.hi[0] || self.lo[1] >= self.hi[1] || self.lo[2] >= self.hi[2] {
            return Err("gyroid bounds must have positive extent on every axis");
        }
        if !self.cell_size.is_finite() || self.cell_size <= 0.0 {
            return Err("cell_size must be finite and > 0");
        }
        if !self.band_thickness.is_finite() || self.band_thickness <= 0.0 {
            return Err("band_thickness must be finite and > 0");
        }
        self.warp.validate()
    }
}

#[derive(Clone, Debug)]
pub struct GyroidField {
    pub params: GyroidParams,
    inner: ImplicitField<GyroidShape>,
}

impl GyroidField {
    pub fn new(params: GyroidParams) -> Self {
        params.validate().expect("invalid gyroid params");
        let inner = ImplicitField {
            seed: params.seed,
            lo: params.lo,
            hi: params.hi,
            shape: GyroidShape::new(params.cell_size),
            band: ScalarBand {
                min: -params.band_thickness,
                max: params.band_thickness,
            },
            solid: params.solid,
            warp: params.warp,
        };
        Self { params, inner }
    }

    pub fn for_world(level: u32, seed: u64) -> Self {
        Self::new(GyroidParams::for_world(level, seed))
    }
}

impl WorldGen for GyroidField {
    #[inline]
    fn sample(&self, point: [i64; 3]) -> CellState {
        self.inner.sample(point)
    }

    #[inline]
    fn classify(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        self.inner.classify(origin, size_log2)
    }
}

fn interval_contains(lo: f32, hi: f32, phase: f32) -> bool {
    let lo = lo as f64;
    let hi = hi as f64;
    let phase = phase as f64;
    let period = std::f64::consts::TAU;
    let k = ((lo - phase) / period).ceil();
    phase + k * period <= hi
}

fn sin_bounds(lo: f32, hi: f32) -> (f32, f32) {
    if hi - lo >= std::f32::consts::TAU {
        return (-1.0, 1.0);
    }
    let mut min = lo.sin().min(hi.sin());
    let mut max = lo.sin().max(hi.sin());
    if interval_contains(lo, hi, std::f32::consts::FRAC_PI_2) {
        max = 1.0;
    }
    if interval_contains(lo, hi, 3.0 * std::f32::consts::FRAC_PI_2) {
        min = -1.0;
    }
    (min, max)
}

fn cos_bounds(lo: f32, hi: f32) -> (f32, f32) {
    if hi - lo >= std::f32::consts::TAU {
        return (-1.0, 1.0);
    }
    let mut min = lo.cos().min(hi.cos());
    let mut max = lo.cos().max(hi.cos());
    if interval_contains(lo, hi, 0.0) {
        max = 1.0;
    }
    if interval_contains(lo, hi, std::f32::consts::PI) {
        min = -1.0;
    }
    (min, max)
}

fn mul_bounds(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let products = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in products {
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::NodeStore;
    use crate::terrain::gen::{gen_region, probe_sample_ns};
    use crate::terrain::materials::AIR;

    fn test_field() -> GyroidField {
        GyroidField::for_world(6, 42)
    }

    #[test]
    fn outside_bounds_is_air() {
        let field = test_field();
        assert_eq!(field.sample([0, 0, 0]), AIR);
        assert_eq!(field.sample([63, 63, 63]), AIR);
    }

    #[test]
    fn deterministic() {
        let field = test_field();
        let mut store_a = NodeStore::new();
        let mut store_b = NodeStore::new();
        let (root_a, _) = gen_region(&mut store_a, &field, [0, 0, 0], 6);
        let (root_b, _) = gen_region(&mut store_b, &field, [0, 0, 0], 6);
        assert_eq!(root_a, root_b, "gyroid generation must be deterministic");
    }

    #[test]
    fn classify_box_conservative() {
        let field = test_field();
        for oz in (0..64).step_by(4) {
            for oy in (0..64).step_by(4) {
                for ox in (0..64).step_by(4) {
                    if let Some(state) = field.classify([ox as i64, oy as i64, oz as i64], 2) {
                        for z in oz..oz + 4 {
                            for y in oy..oy + 4 {
                                for x in ox..ox + 4 {
                                    let sample = field.sample([x as i64, y as i64, z as i64]);
                                    assert_eq!(
                                        sample, state,
                                        "classify said {:?} but sample({x},{y},{z}) = {:?}",
                                        state, sample
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
    fn classify_collapses_some_uniform_boxes() {
        let field = test_field();
        let mut store = NodeStore::new();
        let (_, stats) = gen_region(&mut store, &field, [0, 0, 0], 6);
        assert!(
            stats.total_collapses() > 0,
            "gyroid field should short-circuit some uniform boxes: {stats:?}"
        );
    }

    #[test]
    fn warp_and_band_params_change_shape() {
        let base = GyroidParams::for_world(6, 42);
        let thicker = GyroidField::new(GyroidParams {
            band_thickness: base.band_thickness * 1.5,
            ..base
        });
        let smoother = GyroidField::new(GyroidParams {
            warp: NoiseWarp {
                amplitude: base.warp.amplitude * 0.5,
                ..base.warp
            },
            ..base
        });

        let mut store = NodeStore::new();
        let (root_base, _) = gen_region(&mut store, &GyroidField::new(base), [0, 0, 0], 6);
        let (root_thicker, _) = gen_region(&mut NodeStore::new(), &thicker, [0, 0, 0], 6);
        let (root_smoother, _) = gen_region(&mut NodeStore::new(), &smoother, [0, 0, 0], 6);

        assert_ne!(
            root_base, root_thicker,
            "band thickness should change the structure"
        );
        assert_ne!(
            root_base, root_smoother,
            "warp amplitude should change the structure"
        );
    }

    #[test]
    fn sample_probe_runs() {
        let field = test_field();
        let ns = probe_sample_ns(&field, 2_048);
        assert!(
            ns.is_finite() && ns > 0.0,
            "sample probe must produce a finite time"
        );
    }
}
