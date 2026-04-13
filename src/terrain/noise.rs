//! Hand-rolled value noise over a u64 integer hash. Zero deps, wasm-safe,
//! deterministic across platforms (no float ordering, no floating
//! intermediates in the hash itself).
//!
//! v1 quality is intentionally simple. Swap to simplex/gradient noise behind
//! the `RegionField::sample` boundary if visual review demands it. The
//! consumer (`HeightmapField`) only sees `fractal_2d(x, z, seed, octaves) ->
//! f32` in `[0, 1]`.
//!
//! Exploration note (`hash-thing-fha`): this module also carries small,
//! deterministic candidate primitives for future terrain families so they
//! can be tested in isolation before being threaded through `RegionField`
//! or later post-passes.

use core::f32;

/// SplitMix64 finalizer (Vigna 2015). Strong avalanche on a 64-bit input.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Hash an `(ix, iz, seed)` lattice point to a value in `[0, 1)`.
#[inline]
fn hash_unit(ix: i64, iz: i64, seed: u64) -> f32 {
    // Distinct multipliers per axis to break trivial symmetries.
    let h = (ix as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (iz as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ seed.wrapping_mul(0x165667B19E3779F9);
    let m = mix64(h);
    // Top 24 bits → unit float, mantissa-friendly.
    (m >> 40) as f32 / ((1u32 << 24) as f32)
}

#[inline]
fn hash_pair_unit(ix: i64, iz: i64, seed: u64) -> (f32, f32) {
    let a = hash_unit(ix, iz, seed);
    let b = hash_unit(ix, iz, seed ^ 0xD1B54A32D192ED03);
    (a, b)
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// 2D value noise. Bilinear interpolation between four lattice corners with
/// a smoothstep on the local fraction. Output is in `[0, 1]`.
fn value_2d(x: f32, z: f32, seed: u64) -> f32 {
    let xf = x.floor();
    let zf = z.floor();
    let ix = xf as i64;
    let iz = zf as i64;
    let tx = smoothstep(x - xf);
    let tz = smoothstep(z - zf);

    let v00 = hash_unit(ix, iz, seed);
    let v10 = hash_unit(ix + 1, iz, seed);
    let v01 = hash_unit(ix, iz + 1, seed);
    let v11 = hash_unit(ix + 1, iz + 1, seed);

    let a = v00 * (1.0 - tx) + v10 * tx;
    let b = v01 * (1.0 - tx) + v11 * tx;
    a * (1.0 - tz) + b * tz
}

/// Octave-summed value noise. Result is in `[0, 1]` by construction (each
/// octave is in `[0, 1]`, the weighted sum is divided by the total weight).
///
/// The mathematical result is `<= 1 - 2^-24`, but the final `sum / total`
/// divide can legitimately round to exactly `1.0` under IEEE-754
/// round-to-nearest, and under adversarial input sequences could in principle
/// round a few ulp higher. The explicit `.clamp(0.0, 1.0)` at the return
/// makes the `[0, 1]` bound tight — which in turn makes the y-band proof in
/// `HeightmapField::classify_box` bulletproof rather than "sound with
/// margin". The clamp cost is one min/max per call; at 64³ only a few
/// thousand surface-band calls hit this path (sky and deep stone short-
/// circuit via `classify_box` without sampling).
///
/// `octaves` must be `>= 1`.
pub fn fractal_2d(x: f32, z: f32, seed: u64, octaves: u32) -> f32 {
    debug_assert!(octaves >= 1, "fractal_2d needs at least one octave");
    let mut sum = 0.0f32;
    let mut total = 0.0f32;
    let mut amp = 1.0f32;
    let mut freq = 1.0f32;
    for i in 0..octaves {
        sum += amp * value_2d(x * freq, z * freq, seed.wrapping_add(i as u64));
        total += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    // Tight clamp — makes `classify_box`'s y-band proof bulletproof. See
    // the doc comment above for the rounding story.
    (sum / total).clamp(0.0, 1.0)
}

/// Deterministic 2D Voronoi / Worley sample over jittered lattice points.
///
/// This is not wired into the runtime terrain path yet. It exists as a
/// small, testable primitive for future "bone / coral / cathedral"
/// generators tracked in `hash-thing-fha`. The useful signal is
/// `edge_gap`: it becomes small near cell boundaries, so thresholding
/// low values extracts a Voronoi skeleton without any global state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voronoi2dSample {
    /// Squared distance to the nearest feature point.
    pub nearest_sq: f32,
    /// Squared distance to the second-nearest feature point.
    pub second_sq: f32,
    /// Distance-space gap between the two nearest feature points.
    pub edge_gap: f32,
}

/// Sample a jittered-cell Voronoi field around `(x, z)`.
///
/// Each integer lattice cell owns exactly one feature point with a stable
/// hashed offset in `[0, 1)²`. Sampling scans the local 3×3 neighborhood,
/// which is sufficient in 2D because the nearest and second-nearest feature
/// points to any position must live within one cell of `floor(x), floor(z)`.
pub fn voronoi_2d(x: f32, z: f32, seed: u64) -> Voronoi2dSample {
    let base_x = x.floor() as i64;
    let base_z = z.floor() as i64;

    let mut nearest_sq = f32::INFINITY;
    let mut second_sq = f32::INFINITY;

    for dz in -1..=1 {
        for dx in -1..=1 {
            let cell_x = base_x + dx;
            let cell_z = base_z + dz;
            let (jx, jz) = hash_pair_unit(cell_x, cell_z, seed);
            let fx = cell_x as f32 + jx;
            let fz = cell_z as f32 + jz;
            let dist_sq = (fx - x) * (fx - x) + (fz - z) * (fz - z);

            if dist_sq < nearest_sq {
                second_sq = nearest_sq;
                nearest_sq = dist_sq;
            } else if dist_sq < second_sq {
                second_sq = dist_sq;
            }
        }
    }

    debug_assert!(nearest_sq.is_finite(), "voronoi_2d found no nearest point");
    debug_assert!(
        second_sq.is_finite(),
        "voronoi_2d found no second-nearest point"
    );

    Voronoi2dSample {
        nearest_sq,
        second_sq,
        edge_gap: second_sq.sqrt() - nearest_sq.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractal_in_unit_interval_random_inputs() {
        // 10k randomized inputs — `debug_assert!` would catch a bound break.
        // Use a tiny LCG so we don't pull in any RNG dependency.
        let mut s = 0xDEADBEEFu64;
        for _ in 0..10_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 16) as i32 as f32) * 0.001;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let z = ((s >> 16) as i32 as f32) * 0.001;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let seed = s;
            let v = fractal_2d(x, z, seed, 4);
            assert!((0.0..=1.0).contains(&v), "v={v} x={x} z={z} seed={seed}");
        }
    }

    #[test]
    fn fractal_is_deterministic() {
        let a = fractal_2d(1.5, 2.5, 42, 4);
        let b = fractal_2d(1.5, 2.5, 42, 4);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn voronoi_is_deterministic() {
        let a = voronoi_2d(1.25, -3.5, 42);
        let b = voronoi_2d(1.25, -3.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn voronoi_orders_nearest_and_second() {
        let sample = voronoi_2d(5.125, 9.75, 7);
        assert!(sample.nearest_sq.is_finite());
        assert!(sample.second_sq.is_finite());
        assert!(sample.nearest_sq <= sample.second_sq);
        assert!(sample.edge_gap >= 0.0);
    }

    #[test]
    fn voronoi_random_inputs_stay_finite() {
        let mut s = 0xBAD5EEDu64;
        for _ in 0..10_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 16) as i32 as f32) * 0.01;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let z = ((s >> 16) as i32 as f32) * 0.01;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let sample = voronoi_2d(x, z, s);
            assert!(sample.nearest_sq.is_finite(), "nearest_sq x={x} z={z}");
            assert!(sample.second_sq.is_finite(), "second_sq x={x} z={z}");
            assert!(sample.edge_gap.is_finite(), "edge_gap x={x} z={z}");
            assert!(
                sample.nearest_sq <= sample.second_sq,
                "nearest_sq={} second_sq={} x={x} z={z}",
                sample.nearest_sq,
                sample.second_sq
            );
        }
    }
}
