//! `WorldGen` trait — the abstraction every direct-octree generator
//! recurses against (hash-thing-tam).
//!
//! Two methods:
//!
//! - `sample(point) -> CellState`: exact value at one integer cell.
//! - `classify(origin, level) -> Option<CellState>`: conservative bound.
//!   Must return `Some(state)` only if the entire box `[origin, origin + 2^level)`
//!   is provably uniform — never `Some` for a mixed box. Heuristic
//!   ("probably uniform") collapse is not allowed. Heuristic LOD belongs
//!   to a different layer.
//!
//! The recursive builder in `terrain::gen` takes any `&impl WorldGen` and
//! emits a canonical `NodeId` without ever flattening to a grid. Heightmap,
//! SDF, and composed generators are all sibling structs in this module tree.

use crate::octree::CellState;

/// Pluggable world generation trait. Pure function of (position, params) —
/// no state, no caches. The octree builder recurses against this.
pub trait WorldGen {
    /// Exact material at a single integer cell.
    fn sample(&self, point: [i64; 3]) -> CellState;

    /// Conservative box classification over `[origin, origin + 2^level)`.
    /// Must return `Some(state)` only if the entire box is provably uniform.
    /// Returns `None` for mixed-or-unknown (forces recursion).
    fn classify(&self, origin: [i64; 3], level: u32) -> Option<CellState>;
}

/// Layered generator: asks each layer in priority order. First `Some` from
/// `classify` wins; `sample` falls through to the first layer that claims
/// the point. Layers are ordered highest-priority first.
pub struct ComposedWorldGen {
    layers: Vec<Box<dyn WorldGen>>,
}

impl ComposedWorldGen {
    pub fn new(layers: Vec<Box<dyn WorldGen>>) -> Self {
        Self { layers }
    }
}

impl WorldGen for ComposedWorldGen {
    fn sample(&self, point: [i64; 3]) -> CellState {
        // Last layer is the fallback (lowest priority); higher-priority
        // layers override by returning non-AIR.
        for layer in &self.layers {
            let s = layer.sample(point);
            if s != 0 {
                return s;
            }
        }
        0 // AIR
    }

    fn classify(&self, origin: [i64; 3], level: u32) -> Option<CellState> {
        // If all layers agree on the same state, return it.
        // If any layer can't classify (None), we can't short-circuit.
        let mut result: Option<CellState> = None;
        for layer in &self.layers {
            match layer.classify(origin, level) {
                None => return None, // mixed — must recurse
                Some(state) => match result {
                    None => result = Some(state),
                    Some(prev) if prev == state => {} // agree
                    Some(0) => result = Some(state),  // AIR overridden
                    Some(_) if state == 0 => {}       // AIR doesn't override existing
                    Some(_) => return None,           // disagree on non-AIR — must recurse
                },
            }
        }
        result
    }
}

pub mod const_field;
pub mod half_space;
pub mod heightmap;

pub use heightmap::HeightmapField;

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial layer: every cell is a fixed state.
    struct Uniform(CellState);
    impl WorldGen for Uniform {
        fn sample(&self, _p: [i64; 3]) -> CellState {
            self.0
        }
        fn classify(&self, _o: [i64; 3], _l: u32) -> Option<CellState> {
            Some(self.0)
        }
    }

    /// Layer that can't classify (always returns None), but samples to a value.
    struct Unclassifiable(CellState);
    impl WorldGen for Unclassifiable {
        fn sample(&self, _p: [i64; 3]) -> CellState {
            self.0
        }
        fn classify(&self, _o: [i64; 3], _l: u32) -> Option<CellState> {
            None
        }
    }

    #[test]
    fn composed_single_layer() {
        let gen = ComposedWorldGen::new(vec![Box::new(Uniform(42))]);
        assert_eq!(gen.sample([0, 0, 0]), 42);
        assert_eq!(gen.classify([0, 0, 0], 4), Some(42));
    }

    #[test]
    fn composed_higher_priority_overrides_air() {
        // Layer 0 (high priority) is solid, layer 1 (low priority) is AIR.
        let gen = ComposedWorldGen::new(vec![Box::new(Uniform(10)), Box::new(Uniform(0))]);
        assert_eq!(gen.sample([0, 0, 0]), 10);
        assert_eq!(gen.classify([0, 0, 0], 4), Some(10));
    }

    #[test]
    fn composed_air_layer_does_not_override_solid() {
        // Layer 0 is AIR, layer 1 is solid. Sample falls through to solid.
        let gen = ComposedWorldGen::new(vec![Box::new(Uniform(0)), Box::new(Uniform(7))]);
        assert_eq!(gen.sample([0, 0, 0]), 7);
        // classify: AIR then 7 → AIR overridden by 7
        assert_eq!(gen.classify([0, 0, 0], 4), Some(7));
    }

    #[test]
    fn composed_disagreeing_solids_returns_none() {
        let gen = ComposedWorldGen::new(vec![Box::new(Uniform(5)), Box::new(Uniform(9))]);
        // Both non-AIR but different → must recurse
        assert_eq!(gen.classify([0, 0, 0], 4), None);
    }

    #[test]
    fn composed_unclassifiable_forces_recursion() {
        let gen = ComposedWorldGen::new(vec![Box::new(Unclassifiable(42)), Box::new(Uniform(42))]);
        assert_eq!(gen.classify([0, 0, 0], 4), None);
        // sample still works — first non-AIR wins
        assert_eq!(gen.sample([0, 0, 0]), 42);
    }

    #[test]
    fn composed_empty_layers_returns_air() {
        let gen = ComposedWorldGen::new(vec![]);
        assert_eq!(gen.sample([0, 0, 0]), 0);
        assert_eq!(gen.classify([0, 0, 0], 4), None); // no layers → None
    }

    #[test]
    fn composed_all_air_returns_air() {
        let gen = ComposedWorldGen::new(vec![Box::new(Uniform(0)), Box::new(Uniform(0))]);
        assert_eq!(gen.sample([0, 0, 0]), 0);
        assert_eq!(gen.classify([0, 0, 0], 4), Some(0));
    }
}
