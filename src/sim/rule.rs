use crate::octree::CellState;
use std::fmt;

/// A cellular automaton rule operating on the 26-cell Moore neighborhood in 3D.
pub trait CaRule {
    /// Given a cell's current state and its 26 neighbors, return the next state.
    fn step_cell(&self, center: CellState, neighbors: &[CellState; 26]) -> CellState;
}

/// 3D Game of Life (outer totalistic).
///
/// Parameterized by survival and birth ranges over the 26-neighbor Moore
/// neighborhood. The `Display` impl formats as `S{lo}-{hi}/B{lo}-{hi}`, which
/// is the canonical disambiguator for these presets — preset names alone
/// (`Amoeba`, `Crystal`, `445`, `Pyroclastic`) are ambiguous between
/// different rule families in the literature, so always log both.
///
/// Constructor presets (numerics are the source of truth, names are mnemonic):
///   - [`amoeba`]      → `S9-26/B5-7`  (organic spreading growth)
///   - [`crystal`]     → `S0-6/B1-3`   (structured geometric growth)
///   - [`rule445`]     → `S4-4/B4-4`   (the canonical "4/4" crystals rule)
///   - [`pyroclastic`] → `S4-7/B6-8`   (chaotic, fire-like)
///
/// [`amoeba`]: GameOfLife3D::amoeba
/// [`crystal`]: GameOfLife3D::crystal
/// [`rule445`]: GameOfLife3D::rule445
/// [`pyroclastic`]: GameOfLife3D::pyroclastic
pub struct GameOfLife3D {
    /// survive_min..=survive_max: cell stays alive if neighbor count is in this range
    pub survive_min: u8,
    pub survive_max: u8,
    /// birth_min..=birth_max: dead cell becomes alive if neighbor count is in this range
    pub birth_min: u8,
    pub birth_max: u8,
}

impl GameOfLife3D {
    pub fn new(survive_min: u8, survive_max: u8, birth_min: u8, birth_max: u8) -> Self {
        Self { survive_min, survive_max, birth_min, birth_max }
    }

    /// "Amoeba" preset — `S9-26/B5-7`. Organic spreading growth.
    pub fn amoeba() -> Self {
        Self::new(9, 26, 5, 7)
    }

    /// "Crystal" preset — `S0-6/B1-3`. Structured geometric growth.
    pub fn crystal() -> Self {
        Self::new(0, 6, 1, 3)
    }

    /// "445" preset — `S4-4/B4-4`. The canonical 4/4 crystals rule.
    pub fn rule445() -> Self {
        Self::new(4, 4, 4, 4)
    }

    /// "Pyroclastic" preset — `S4-7/B6-8`. Chaotic, fire-like.
    pub fn pyroclastic() -> Self {
        Self::new(4, 7, 6, 8)
    }
}

impl fmt::Display for GameOfLife3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "S{}-{}/B{}-{}",
            self.survive_min, self.survive_max, self.birth_min, self.birth_max
        )
    }
}

impl CaRule for GameOfLife3D {
    fn step_cell(&self, center: CellState, neighbors: &[CellState; 26]) -> CellState {
        let alive_count: u8 = neighbors.iter().map(|&n| if n > 0 { 1u8 } else { 0 }).sum();
        if center > 0 {
            // alive: survive?
            if alive_count >= self.survive_min && alive_count <= self.survive_max {
                center
            } else {
                0
            }
        } else {
            // dead: birth?
            if alive_count >= self.birth_min && alive_count <= self.birth_max {
                1
            } else {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock in the numeric ranges of every preset. This test is the source
    /// of truth for hash-thing-pbx — if you change a preset's numbers, this
    /// test fails and you must also update the doc comments above the
    /// constructors and any caller (main.rs) that displays them.
    #[test]
    fn preset_ranges_match_doc() {
        let amoeba = GameOfLife3D::amoeba();
        assert_eq!((amoeba.survive_min, amoeba.survive_max), (9, 26));
        assert_eq!((amoeba.birth_min, amoeba.birth_max), (5, 7));
        assert_eq!(format!("{}", amoeba), "S9-26/B5-7");

        let crystal = GameOfLife3D::crystal();
        assert_eq!((crystal.survive_min, crystal.survive_max), (0, 6));
        assert_eq!((crystal.birth_min, crystal.birth_max), (1, 3));
        assert_eq!(format!("{}", crystal), "S0-6/B1-3");

        let r445 = GameOfLife3D::rule445();
        assert_eq!((r445.survive_min, r445.survive_max), (4, 4));
        assert_eq!((r445.birth_min, r445.birth_max), (4, 4));
        assert_eq!(format!("{}", r445), "S4-4/B4-4");

        let pyro = GameOfLife3D::pyroclastic();
        assert_eq!((pyro.survive_min, pyro.survive_max), (4, 7));
        assert_eq!((pyro.birth_min, pyro.birth_max), (6, 8));
        assert_eq!(format!("{}", pyro), "S4-7/B6-8");
    }
}
