use crate::octree::CellState;

/// A cellular automaton rule operating on the 26-cell Moore neighborhood in 3D.
pub trait CaRule {
    /// Given a cell's current state and its 26 neighbors, return the next state.
    fn step_cell(&self, center: CellState, neighbors: &[CellState; 26]) -> CellState;
}

/// 3D Game of Life (outer totalistic).
///
/// Parameterized by survival and birth ranges.
/// Default: survives with 4 neighbors, born with 4 neighbors ("4/4" — crystals).
/// Other interesting rules:
///   "5..7/6" — slow growth
///   "4/4" — diamond-like crystals
///   "2..6/5..7" — amoeba
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
        Self {
            survive_min,
            survive_max,
            birth_min,
            birth_max,
        }
    }

    /// "Amoeba" rule — produces organic-looking growth.
    pub fn amoeba() -> Self {
        Self::new(9, 26, 5, 7)
    }

    /// "Crystal" rule — produces structured, geometric growth.
    pub fn crystal() -> Self {
        Self::new(0, 6, 1, 3)
    }

    /// "445" rule — stable structures.
    pub fn rule445() -> Self {
        Self::new(4, 4, 4, 4)
    }

    /// "Pyroclastic" — chaotic, fire-like.
    pub fn pyroclastic() -> Self {
        Self::new(4, 7, 6, 8)
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
