//! `HalfSpaceField` — splits space along an axis at a threshold. Cells with
//! `coord < threshold` are `below`, otherwise `above`.
//!
//! This is the second test oracle. Unlike `ConstField`, it has a real
//! transition surface, so the builder must actually recurse into the
//! straddling box and short-circuit on the boxes that are fully on one side.

// Test-only fixture today; will become a real consumer when caves / dungeons
// land in 3fq.2+.
#![allow(dead_code)]

use super::RegionField;
use crate::octree::CellState;

#[derive(Clone, Copy, Debug)]
pub enum HalfSpaceAxis {
    X,
    Y,
    Z,
}

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

impl RegionField for HalfSpaceField {
    #[inline]
    fn sample(&self, point: [i64; 3]) -> CellState {
        if self.coord(point) < self.threshold {
            self.below
        } else {
            self.above
        }
    }

    fn classify_box(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;
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
