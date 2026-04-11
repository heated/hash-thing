//! Hand-rolled value noise over a u64 integer hash. Zero deps, wasm-safe,
//! deterministic across platforms (no float ordering, no floating
//! intermediates in the hash itself).
//!
//! v1 quality is intentionally simple. Swap to simplex/gradient noise behind
//! the `RegionField::sample` boundary if visual review demands it. The
//! consumer (`HeightmapField`) only sees `fractal_2d(x, z, seed, octaves) ->
//! f32` in `[0, 1]`.

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
/// `octaves` must be `>= 1`. The `debug_assert!` on the return value is the
/// proof precondition for `HeightmapField::classify_box` — if it ever fires,
/// the y-band short-circuit becomes unsound.
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
    let result = sum / total;
    debug_assert!(
        (0.0..=1.0).contains(&result),
        "fractal_2d out of [0,1]: {result}"
    );
    result
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
}
