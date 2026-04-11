// Public API of a utility module that feeds future consumers (Margolus movement
// phase, probabilistic CA rules, terrain-gen). Some entry points have no call
// sites yet; silence dead_code until those land.
#![allow(dead_code)]

//! Deterministic per-cell randomness for Hashlife-compatible CA rules.
//!
//! ## Why this module exists
//!
//! A probabilistic cellular automaton (e.g. "this sand grain slides left with
//! p=0.3") only stays compatible with Hashlife memoization if the randomness
//! is a **pure function of state**, not a draw from a hidden global PRNG. Two
//! identical subtrees evaluated in different parts of the world must produce
//! identical successors — otherwise the hash-consed step cache silently
//! corrupts.
//!
//! The trick from SPEC.md is `hash(position, generation, seed)`: treat the
//! cell's absolute coordinate, the simulation generation, and a world seed as
//! a key, and hash them into a u64 per step. Pure, reproducible, and local
//! enough that Margolus-block or single-cell rules can query it without
//! touching any shared state.
//!
//! ## Guarantees
//!
//! - **Deterministic.** `cell_hash(x, y, z, g, s)` returns the same value
//!   every call with the same inputs, on every platform.
//! - **Hashlife-compatible.** No global state, no scan-order dependence.
//! - **Avalanche.** Flipping one input bit changes roughly half of the output
//!   bits (empirically; see tests).
//! - **Signed coordinates.** Coordinates are `i64` so infinite-world expansion
//!   with negative-octant growth (SPEC.md lazy root growth) works without a
//!   coordinate remapping. The bit pattern is what gets hashed, so the choice
//!   of i64 vs u64 at the call site is cosmetic.
//!
//! ## Non-goals
//!
//! - **Not cryptographic.** Do not use for anything security-sensitive.
//! - **Not a stream RNG.** There is no mutable state. If you need multiple
//!   draws at one cell per step, pass a small "stream" tag through the seed.

/// SplitMix64 finalizer. Strong avalanche on a 64-bit input.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

// Distinct odd 64-bit multipliers so each axis/field enters the hash with a
// different spread. Breaking axis symmetry prevents trivial collisions like
// `cell_hash(a,b,c,..) == cell_hash(b,a,c,..)`. We deliberately avoid the
// two SplitMix64 finalizer constants used inside `mix64` below, pulling
// instead from the xxHash / Murmur3 families so the per-field multiply and
// the subsequent mix step are doing structurally distinct work.
const MX: u64 = 0x9E3779B97F4A7C15; // 2^64 / φ (golden ratio)
const MY: u64 = 0xC2B2AE3D27D4EB4F; // xxHash64 PRIME2
const MZ: u64 = 0x165667B19E3779F9; // xxHash64 PRIME3
const MG: u64 = 0x85EBCA77C2B2AE63; // Murmur3-class, distinct from mix64 constants

/// Hash a (position, generation, seed) tuple to a pseudo-random `u64`.
///
/// This is the one primitive. Every other helper in the module is a thin
/// adapter over this function.
#[inline]
pub fn cell_hash(x: i64, y: i64, z: i64, generation: u64, seed: u64) -> u64 {
    let mut h = seed;
    h = mix64(h ^ (x as u64).wrapping_mul(MX));
    h = mix64(h ^ (y as u64).wrapping_mul(MY));
    h = mix64(h ^ (z as u64).wrapping_mul(MZ));
    h = mix64(h ^ generation.wrapping_mul(MG));
    h
}

/// Uniform float in `[0.0, 1.0)` for a given cell/generation/seed.
#[inline]
pub fn cell_rand_f64(x: i64, y: i64, z: i64, generation: u64, seed: u64) -> f64 {
    // Top 53 bits → exact uniform over dyadic rationals in [0, 1).
    let bits = cell_hash(x, y, z, generation, seed) >> 11;
    (bits as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Near-uniform integer in `[0, n)`. Panics if `n == 0`.
///
/// Uses Lemire's fast-range trick (64×64→128-bit multiply, take the high
/// half). Residual modulo bias is bounded by `n / 2^64`, which is far below
/// any observable threshold for any `n` a CA rule would realistically ask
/// for. If you need strictly unbiased output, reject the high-bias band.
#[inline]
pub fn cell_rand_range(x: i64, y: i64, z: i64, generation: u64, seed: u64, n: u64) -> u64 {
    assert!(n > 0, "cell_rand_range: n must be > 0, got {n}");
    let h = cell_hash(x, y, z, generation, seed) as u128;
    ((h * n as u128) >> 64) as u64
}

/// Boolean that is true with probability `p` (clamped to `[0, 1]`).
///
/// `NaN` inputs return `false` — `f64::clamp` passes NaN through unchanged,
/// and `_ < NaN` is always false, so the result is well-defined (if not
/// particularly meaningful). Callers that care should not pass NaN.
#[inline]
pub fn cell_rand_bool(x: i64, y: i64, z: i64, generation: u64, seed: u64, p: f64) -> bool {
    let p = p.clamp(0.0, 1.0);
    cell_rand_f64(x, y, z, generation, seed) < p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        // Same inputs → same output, repeatedly.
        let a = cell_hash(1, 2, 3, 7, 0xDEAD_BEEF);
        for _ in 0..100 {
            assert_eq!(cell_hash(1, 2, 3, 7, 0xDEAD_BEEF), a);
        }
    }

    #[test]
    fn different_positions_differ() {
        let base = cell_hash(0, 0, 0, 0, 0);
        assert_ne!(cell_hash(1, 0, 0, 0, 0), base);
        assert_ne!(cell_hash(0, 1, 0, 0, 0), base);
        assert_ne!(cell_hash(0, 0, 1, 0, 0), base);
        // Axis swaps must not alias, so the per-axis multipliers must differ.
        assert_ne!(cell_hash(1, 2, 3, 0, 0), cell_hash(2, 1, 3, 0, 0));
        assert_ne!(cell_hash(1, 2, 3, 0, 0), cell_hash(3, 2, 1, 0, 0));
    }

    #[test]
    fn different_generations_differ() {
        let g0 = cell_hash(5, 5, 5, 0, 0);
        let g1 = cell_hash(5, 5, 5, 1, 0);
        let g2 = cell_hash(5, 5, 5, 2, 0);
        assert_ne!(g0, g1);
        assert_ne!(g1, g2);
        assert_ne!(g0, g2);
    }

    #[test]
    fn different_seeds_differ() {
        assert_ne!(cell_hash(0, 0, 0, 0, 1), cell_hash(0, 0, 0, 0, 2));
    }

    #[test]
    fn negative_coordinates_work() {
        // Infinite-world root expansion produces negative coords. The hash
        // must not panic and must differ from the positive-mirror cell.
        let pos = cell_hash(7, 11, 13, 0, 0);
        let neg = cell_hash(-7, -11, -13, 0, 0);
        assert_ne!(pos, neg);
    }

    #[test]
    fn f64_in_unit_interval() {
        // Sample a bunch of cells; every output must be in [0, 1).
        for x in 0..32 {
            for y in 0..32 {
                let v = cell_rand_f64(x, y, 0, 0, 42);
                assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
            }
        }
    }

    #[test]
    fn f64_roughly_uniform() {
        // Split [0,1) into 10 buckets over 10_000 samples. Each bucket
        // should hold ~1000 samples. Loose bound: [600, 1400]. We vary x, y,
        // AND generation so a hash that's only weak on one axis still fails.
        let mut buckets = [0u32; 10];
        let mut n = 0u32;
        for g in 0..10u64 {
            for x in 0..50i64 {
                for y in 0..20i64 {
                    let v = cell_rand_f64(x, y, 0, g, 0xC0FFEE);
                    buckets[(v * 10.0) as usize] += 1;
                    n += 1;
                }
            }
        }
        assert_eq!(n, 10_000);
        for (i, &c) in buckets.iter().enumerate() {
            assert!(
                (600..=1400).contains(&c),
                "bucket {i} = {c}, expected ~1000"
            );
        }
    }

    #[test]
    fn range_is_bounded() {
        for i in 0..1000i64 {
            let r = cell_rand_range(i, 0, 0, 0, 0, 7);
            assert!(r < 7);
        }
    }

    #[test]
    #[should_panic(expected = "n must be > 0")]
    fn range_zero_panics() {
        cell_rand_range(0, 0, 0, 0, 0, 0);
    }

    #[test]
    fn bool_probability_roughly_matches() {
        // p=0.3 over 10_000 samples → expect ~3000 trues. Loose: [2700, 3300].
        let mut trues = 0u32;
        for i in 0..10_000i64 {
            if cell_rand_bool(i, 0, 0, 0, 0xABCD, 0.3) {
                trues += 1;
            }
        }
        assert!((2700..=3300).contains(&trues), "trues = {trues}");
    }

    #[test]
    fn bool_clamps_probabilities() {
        // Out-of-range p must clamp, not panic or produce NaN comparisons.
        for i in 0..100i64 {
            assert!(!cell_rand_bool(i, 0, 0, 0, 0, -1.0));
            assert!(cell_rand_bool(i, 0, 0, 0, 0, 2.0));
        }
    }

    #[test]
    fn avalanche_single_bit_flip() {
        // Flipping one bit of one input should change ~half of the output
        // bits on average. A good hash gets ~32; we accept [16, 48] as
        // "clearly mixing." We probe all five input channels independently
        // so a weak-on-one-axis mixer still fails.
        fn measure_channel<F: Fn(u64) -> u64>(base: u64, f: F) -> u32 {
            let mut total = 0u32;
            for bit in 0..64 {
                let flipped = f(1u64 << bit);
                total += (base ^ flipped).count_ones();
            }
            total / 64
        }

        // Use nonzero baselines so zero-multiplier corner cases don't sneak
        // a pass. (mix64(0) is still 0, but the other fields prevent that
        // from mattering here.)
        let base = cell_hash(17, 29, 41, 53, 0x01234567_89ABCDEF);
        let avg_x = measure_channel(base, |b| {
            cell_hash(17 ^ b as i64, 29, 41, 53, 0x01234567_89ABCDEF)
        });
        let avg_y = measure_channel(base, |b| {
            cell_hash(17, 29 ^ b as i64, 41, 53, 0x01234567_89ABCDEF)
        });
        let avg_z = measure_channel(base, |b| {
            cell_hash(17, 29, 41 ^ b as i64, 53, 0x01234567_89ABCDEF)
        });
        let avg_g = measure_channel(base, |b| {
            cell_hash(17, 29, 41, 53 ^ b, 0x01234567_89ABCDEF)
        });
        let avg_s = measure_channel(base, |b| {
            cell_hash(17, 29, 41, 53, 0x01234567_89ABCDEF ^ b)
        });
        for (name, avg) in [
            ("x", avg_x),
            ("y", avg_y),
            ("z", avg_z),
            ("generation", avg_g),
            ("seed", avg_s),
        ] {
            assert!(
                (16..=48).contains(&avg),
                "avalanche mean for {name} = {avg} bits, expected ~32"
            );
        }
    }

    #[test]
    fn hashlife_memoization_sanity() {
        // The whole point: two identical "subtrees" at different generations
        // see different randomness, so two different generations of the same
        // block do not collide in a step cache keyed on (node, generation).
        // We simulate that with a 2x2x2 block of cells.
        let block = |g: u64| -> [u64; 8] {
            let mut out = [0u64; 8];
            for (i, slot) in out.iter_mut().enumerate() {
                let dx = (i & 1) as i64;
                let dy = ((i >> 1) & 1) as i64;
                let dz = ((i >> 2) & 1) as i64;
                *slot = cell_hash(dx, dy, dz, g, 0);
            }
            out
        };
        assert_ne!(block(0), block(1));
        // And the same generation is stable.
        assert_eq!(block(0), block(0));
    }
}
