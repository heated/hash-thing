use crate::octree::Cell;
use std::fmt;

/// Default "alive" cell payload used by the legacy GoL smoke scene.
pub(crate) const ALIVE: Cell = Cell::pack(1, 0);

/// A cellular automaton rule operating on the 26-cell Moore neighborhood in 3D.
///
/// Rules are pure local transforms: they inspect only the center cell and its
/// neighbors and return the next cell state for that center.
pub trait CaRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell;
}

/// Identity rule for static materials. Returns the center cell unchanged.
pub struct NoopRule;

impl CaRule for NoopRule {
    fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        center
    }
}

/// Fire persists while fuel is adjacent, and is quenched by water.
pub struct FireRule {
    pub fuel_material: u16,
    pub quencher_material: u16,
}

impl CaRule for FireRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "FireRule should not dispatch for AIR");
        if neighbors
            .iter()
            .any(|neighbor| neighbor.material() == self.quencher_material)
        {
            Cell::EMPTY
        } else if neighbors
            .iter()
            .any(|neighbor| neighbor.material() == self.fuel_material)
        {
            center
        } else {
            Cell::EMPTY
        }
    }
}

/// Water solidifies into a configured product when the reactive material is adjacent.
pub struct WaterRule {
    pub reactive_material: u16,
    pub reaction_product: Cell,
}

impl CaRule for WaterRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "WaterRule should not dispatch for AIR");
        if neighbors
            .iter()
            .any(|neighbor| neighbor.material() == self.reactive_material)
        {
            self.reaction_product
        } else {
            center
        }
    }
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
#[derive(Clone, Copy)]
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
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        let alive_count: u8 = neighbors
            .iter()
            .map(|neighbor| if neighbor.is_empty() { 0u8 } else { 1u8 })
            .sum();
        if !center.is_empty() {
            if alive_count >= self.survive_min && alive_count <= self.survive_max {
                center
            } else {
                Cell::EMPTY
            }
        } else if alive_count >= self.birth_min && alive_count <= self.birth_max {
            ALIVE
        } else {
            Cell::EMPTY
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit coverage for `GameOfLife3D` and the first registry-backed rules.

    use super::*;

    fn mat(material: u16) -> Cell {
        Cell::pack(material, 0)
    }

    /// Construct a 26-neighbor array with exactly `count` live cells
    /// (using material id 1) and the rest empty.
    fn neighbors_with_alive(count: usize) -> [Cell; 26] {
        assert!(count <= 26);
        let mut n = [Cell::EMPTY; 26];
        for slot in n.iter_mut().take(count) {
            *slot = ALIVE;
        }
        n
    }

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

    #[test]
    fn dead_with_zero_neighbors_stays_dead_all_presets() {
        let zero = neighbors_with_alive(0);
        assert_eq!(
            GameOfLife3D::amoeba().step_cell(Cell::EMPTY, &zero),
            Cell::EMPTY
        );
        assert_eq!(
            GameOfLife3D::crystal().step_cell(Cell::EMPTY, &zero),
            Cell::EMPTY
        );
        assert_eq!(
            GameOfLife3D::rule445().step_cell(Cell::EMPTY, &zero),
            Cell::EMPTY
        );
        assert_eq!(
            GameOfLife3D::pyroclastic().step_cell(Cell::EMPTY, &zero),
            Cell::EMPTY
        );
    }

    #[test]
    fn crystal_isolated_live_cell_survives() {
        let zero = neighbors_with_alive(0);
        assert_eq!(GameOfLife3D::crystal().step_cell(ALIVE, &zero), ALIVE);
    }

    #[test]
    fn amoeba_isolated_live_cell_dies() {
        let zero = neighbors_with_alive(0);
        assert_eq!(GameOfLife3D::amoeba().step_cell(ALIVE, &zero), Cell::EMPTY);
    }

    #[test]
    fn rule445_survive_range_boundary() {
        let rule = GameOfLife3D::rule445();
        assert_eq!(rule.step_cell(ALIVE, &neighbors_with_alive(3)), Cell::EMPTY);
        assert_eq!(rule.step_cell(ALIVE, &neighbors_with_alive(4)), ALIVE);
        assert_eq!(rule.step_cell(ALIVE, &neighbors_with_alive(5)), Cell::EMPTY);
    }

    #[test]
    fn rule445_birth_range_boundary() {
        let rule = GameOfLife3D::rule445();
        assert_eq!(
            rule.step_cell(Cell::EMPTY, &neighbors_with_alive(3)),
            Cell::EMPTY
        );
        assert_eq!(rule.step_cell(Cell::EMPTY, &neighbors_with_alive(4)), ALIVE);
        assert_eq!(
            rule.step_cell(Cell::EMPTY, &neighbors_with_alive(5)),
            Cell::EMPTY
        );
    }

    #[test]
    fn survival_preserves_center_payload() {
        let rule = GameOfLife3D::crystal();
        let zero = neighbors_with_alive(0);
        let tagged = Cell::pack(5, 42);
        assert_eq!(rule.step_cell(tagged, &zero), tagged);
    }

    #[test]
    fn nonempty_neighbors_all_count_as_alive() {
        let mut n = [Cell::EMPTY; 26];
        n[0] = Cell::pack(1, 0);
        n[1] = Cell::pack(2, 0);
        n[2] = Cell::pack(7, 3);
        n[3] = Cell::pack(8, 12);
        let rule = GameOfLife3D::rule445();
        assert_eq!(rule.step_cell(Cell::EMPTY, &n), ALIVE);
    }

    #[test]
    fn noop_rule_is_identity_for_varied_cells() {
        let rule = NoopRule;
        let neighbor_sets = [[Cell::EMPTY; 26], [ALIVE; 26], neighbors_with_alive(7), {
            let mut mixed = [Cell::EMPTY; 26];
            mixed[0] = Cell::pack(2, 5);
            mixed[1] = Cell::pack(9, 1);
            mixed[2] = Cell::pack(Cell::MAX_MATERIAL, Cell::MAX_METADATA);
            mixed
        }];

        let centers = [
            Cell::EMPTY,
            ALIVE,
            Cell::pack(11, 1),
            Cell::pack(27, 42),
            Cell::pack(Cell::MAX_MATERIAL, Cell::MAX_METADATA),
        ];

        for neighbors in neighbor_sets {
            for center in centers {
                assert_eq!(
                    rule.step_cell(center, &neighbors),
                    center,
                    "NoopRule must leave {center:?} unchanged"
                );
            }
        }
    }

    #[test]
    fn fire_rule_burns_out_without_fuel() {
        let rule = FireRule {
            fuel_material: 3,
            quencher_material: 5,
        };
        assert_eq!(
            rule.step_cell(mat(4), &[Cell::EMPTY; 26]),
            Cell::EMPTY,
            "isolated fire should burn out"
        );
    }

    #[test]
    fn fire_rule_survives_when_fuel_is_adjacent() {
        let rule = FireRule {
            fuel_material: 3,
            quencher_material: 5,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(3);
        assert_eq!(rule.step_cell(mat(4), &neighbors), mat(4));
    }

    #[test]
    fn fire_rule_quencher_beats_fuel_and_extinguishes() {
        let rule = FireRule {
            fuel_material: 3,
            quencher_material: 5,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(3);
        neighbors[1] = mat(5);

        assert_eq!(rule.step_cell(Cell::pack(4, 9), &neighbors), Cell::EMPTY);
    }

    #[test]
    fn fire_rule_preserves_center_payload_when_fueled() {
        let rule = FireRule {
            fuel_material: 3,
            quencher_material: 5,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(3);
        let tagged_fire = Cell::pack(4, 9);

        assert_eq!(rule.step_cell(tagged_fire, &neighbors), tagged_fire);
    }

    #[test]
    fn water_rule_reacts_to_configured_neighbor() {
        let rule = WaterRule {
            reactive_material: 4,
            reaction_product: mat(1),
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(4);
        assert_eq!(rule.step_cell(mat(5), &neighbors), mat(1));
    }

    #[test]
    fn water_rule_preserves_center_without_reactive_neighbor() {
        let rule = WaterRule {
            reactive_material: 4,
            reaction_product: Cell::pack(1, 7),
        };
        let center = Cell::pack(5, 11);

        assert_eq!(rule.step_cell(center, &[Cell::EMPTY; 26]), center);
    }

    #[test]
    fn water_rule_returns_exact_configured_product() {
        let rule = WaterRule {
            reactive_material: 4,
            reaction_product: Cell::pack(8, 13),
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(4);

        assert_eq!(
            rule.step_cell(Cell::pack(5, 2), &neighbors),
            Cell::pack(8, 13)
        );
    }

    #[test]
    fn all_26_neighbors_alive_survival_by_preset() {
        let full = neighbors_with_alive(26);
        assert_eq!(GameOfLife3D::amoeba().step_cell(ALIVE, &full), ALIVE);
        assert_eq!(GameOfLife3D::crystal().step_cell(ALIVE, &full), Cell::EMPTY);
        assert_eq!(GameOfLife3D::rule445().step_cell(ALIVE, &full), Cell::EMPTY);
        assert_eq!(
            GameOfLife3D::pyroclastic().step_cell(ALIVE, &full),
            Cell::EMPTY
        );
    }
}
