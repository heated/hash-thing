//! `ConstField` — every cell is the same state. Used as the trivial test
//! oracle: the builder recursing against `ConstField { AIR }` must produce
//! exactly `store.uniform(level, AIR)`.

// Test-only fixture — the trivial WorldGen for builder regression tests.

use super::WorldGen;
use crate::octree::CellState;

#[derive(Debug)]
pub struct ConstField {
    pub state: CellState,
}

impl ConstField {
    pub fn new(state: CellState) -> Self {
        Self { state }
    }
}

impl WorldGen for ConstField {
    #[inline]
    fn sample(&self, _point: [i64; 3]) -> CellState {
        self.state
    }

    #[inline]
    fn classify(&self, _origin: [i64; 3], _level: u32) -> Option<CellState> {
        Some(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_returns_constant() {
        let f = ConstField::new(42);
        assert_eq!(f.sample([0, 0, 0]), 42);
        assert_eq!(f.sample([100, -50, 999]), 42);
    }

    #[test]
    fn classify_always_returns_some() {
        let f = ConstField::new(7);
        assert_eq!(f.classify([0, 0, 0], 0), Some(7));
        assert_eq!(f.classify([-100, 50, 0], 10), Some(7));
    }

    #[test]
    fn empty_field_is_zero() {
        let f = ConstField::new(0);
        assert_eq!(f.sample([0, 0, 0]), 0);
        assert_eq!(f.classify([0, 0, 0], 5), Some(0));
    }
}
