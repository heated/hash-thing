//! `HalfSpaceField` — splits space along an axis at a threshold. Cells with
//! `coord < threshold` are `below`, otherwise `above`.
//!
//! This is the second test oracle. Unlike `ConstField`, it has a real
//! transition surface, so the builder must actually recurse into the
//! straddling box and short-circuit on the boxes that are fully on one side.

// Test-only fixture — used for builder regression tests.

use super::WorldGen;
use crate::octree::CellState;

#[derive(Clone, Copy, Debug)]
pub enum HalfSpaceAxis {
    X,
    Y,
    Z,
}

#[derive(Debug)]
pub struct HalfSpaceField {
    pub axis: HalfSpaceAxis,
    pub threshold: i64,
    pub below: CellState,
    pub above: CellState,
}

impl HalfSpaceField {
    #[inline]
    fn coord(&self, point: [i64; 3]) -> i64 {
        match self.axis {
            HalfSpaceAxis::X => point[0],
            HalfSpaceAxis::Y => point[1],
            HalfSpaceAxis::Z => point[2],
        }
    }
}

impl WorldGen for HalfSpaceField {
    #[inline]
    fn sample(&self, point: [i64; 3]) -> CellState {
        if self.coord(point) < self.threshold {
            self.below
        } else {
            self.above
        }
    }

    fn classify(&self, origin: [i64; 3], level: u32) -> Option<CellState> {
        let size = 1i64 << level;
        let lo = self.coord(origin);
        let hi = lo + size; // exclusive: cells are [lo, hi)
                            // Box is fully on the `below` side iff every cell coord < threshold,
                            // i.e. hi - 1 < threshold, i.e. hi <= threshold.
        if hi <= self.threshold {
            return Some(self.below);
        }
        // Box is fully on the `above` side iff lo >= threshold.
        if lo >= self.threshold {
            return Some(self.above);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn y_split(threshold: i64) -> HalfSpaceField {
        HalfSpaceField {
            axis: HalfSpaceAxis::Y,
            threshold,
            below: 100, // "solid"
            above: 0,   // "air"
        }
    }

    #[test]
    fn sample_below_threshold() {
        let f = y_split(10);
        assert_eq!(f.sample([0, 9, 0]), 100);
        assert_eq!(f.sample([5, -1, 3]), 100);
    }

    #[test]
    fn sample_at_and_above_threshold() {
        let f = y_split(10);
        assert_eq!(f.sample([0, 10, 0]), 0);
        assert_eq!(f.sample([0, 11, 0]), 0);
    }

    #[test]
    fn classify_fully_below() {
        let f = y_split(10);
        // Box [0, 0, 0] size=8 → y ∈ [0, 8). Threshold 10 → hi=8 ≤ 10 → below.
        assert_eq!(f.classify([0, 0, 0], 3), Some(100));
    }

    #[test]
    fn classify_fully_above() {
        let f = y_split(10);
        // Box [0, 16, 0] size=8 → y ∈ [16, 24). lo=16 ≥ 10 → above.
        assert_eq!(f.classify([0, 16, 0], 3), Some(0));
    }

    #[test]
    fn classify_straddle_returns_none() {
        let f = y_split(10);
        // Box [0, 8, 0] size=8 → y ∈ [8, 16). lo=8 < 10 and hi=16 > 10 → mixed.
        assert_eq!(f.classify([0, 8, 0], 3), None);
    }

    #[test]
    fn classify_boundary_exact_hi_equals_threshold() {
        let f = y_split(8);
        // Box [0, 0, 0] size=8 → hi=8, threshold=8 → hi ≤ threshold → fully below.
        assert_eq!(f.classify([0, 0, 0], 3), Some(100));
    }

    #[test]
    fn classify_boundary_lo_equals_threshold() {
        let f = y_split(8);
        // Box [0, 8, 0] size=4 → lo=8 ≥ 8 → fully above.
        assert_eq!(f.classify([0, 8, 0], 2), Some(0));
    }

    #[test]
    fn x_axis_split() {
        let f = HalfSpaceField {
            axis: HalfSpaceAxis::X,
            threshold: 5,
            below: 42,
            above: 0,
        };
        assert_eq!(f.sample([4, 0, 0]), 42);
        assert_eq!(f.sample([5, 0, 0]), 0);
        assert_eq!(f.classify([0, 0, 0], 2), Some(42)); // x ∈ [0, 4) < 5
        assert_eq!(f.classify([8, 0, 0], 2), Some(0)); // x ∈ [8, 12) ≥ 5
        assert_eq!(f.classify([4, 0, 0], 2), None); // x ∈ [4, 8) straddles 5
    }

    #[test]
    fn z_axis_split() {
        let f = HalfSpaceField {
            axis: HalfSpaceAxis::Z,
            threshold: 16,
            below: 77,
            above: 0,
        };
        assert_eq!(f.sample([0, 0, 15]), 77);
        assert_eq!(f.sample([0, 0, 16]), 0);
        assert_eq!(f.classify([0, 0, 0], 4), Some(77)); // z ∈ [0, 16) → hi=16 ≤ 16
        assert_eq!(f.classify([0, 0, 16], 3), Some(0)); // z ∈ [16, 24) ≥ 16
    }
}
