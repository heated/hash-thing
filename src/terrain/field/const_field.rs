//! `ConstField` — every cell is the same state. Used as the trivial test
//! oracle: the builder recursing against `ConstField { AIR }` must produce
//! exactly `store.uniform(level, AIR)`.

// Test-only fixture today; will become a real consumer once 1v0.1 lands.
#![allow(dead_code)]

use super::RegionField;
use crate::octree::CellState;

pub struct ConstField {
    pub state: CellState,
}

impl ConstField {
    pub fn new(state: CellState) -> Self {
        Self { state }
    }
}

impl RegionField for ConstField {
    #[inline]
    fn sample(&self, _point: [i64; 3]) -> CellState {
        self.state
    }

    #[inline]
    fn classify_box(&self, _origin: [i64; 3], _size_log2: u32) -> Option<CellState> {
        Some(self.state)
    }
}
