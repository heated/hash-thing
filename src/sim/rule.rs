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

/// Context passed to block rules for deterministic RNG and position awareness.
///
/// All fields are pure functions of position + generation + seed, so block rules
/// remain Hashlife-compatible (no global mutable state).
#[derive(Clone, Copy, Debug)]
pub struct BlockContext {
    /// World-space origin of the block's (0,0,0) corner.
    pub block_origin: [i64; 3],
    /// Current simulation generation.
    pub generation: u64,
    /// Per-world seed for deterministic randomness.
    pub world_seed: u64,
    /// Pre-computed RNG hash from `rng::cell_hash(origin, generation, seed)`.
    pub rng_hash: u64,
}

/// A block-based CA rule operating on 2x2x2 cell blocks.
///
/// Block rules implement mass-conserving permutations: the output must be a
/// rearrangement of the input cells (multiset equality). This invariant enables
/// Margolus-style movement (sand falling, fluid flow) without creating or
/// destroying matter.
///
/// Cell ordering within the block follows octant convention:
///   `block_index(dx, dy, dz) = dx + dy*2 + dz*4`
/// matching `octant_index` in `src/octree/node.rs`.
pub trait BlockRule {
    fn step_block(&self, block: &[Cell; 8], ctx: &BlockContext) -> [Cell; 8];
}

/// Map local (dx, dy, dz) offsets (each 0 or 1) to an index in `[Cell; 8]`.
///
/// Matches `octant_index` in `src/octree/node.rs` — this is a load-bearing
/// invariant. Do not change one without the other.
#[inline]
pub const fn block_index(dx: usize, dy: usize, dz: usize) -> usize {
    dx + dy * 2 + dz * 4
}

/// Identity rule for static materials. Returns the center cell unchanged.
#[derive(Debug)]
pub struct NoopRule;

impl CaRule for NoopRule {
    fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        center
    }
}

/// Fire persists while fuel is adjacent, and is quenched by water.
#[derive(Debug)]
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
#[derive(Debug)]
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
/// The only retained named preset is [`rule445`], which backs the runtime
/// smoke-scene shortcut in `main.rs`. Other historical presets are expressed
/// directly with [`GameOfLife3D::new`] in tests instead of being part of the
/// public scaffold.
///
/// [`rule445`]: GameOfLife3D::rule445
#[derive(Clone, Copy, Debug)]
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

    /// "445" preset — `S4-4/B4-4`. The canonical 4/4 crystals rule.
    pub fn rule445() -> Self {
        Self::new(4, 4, 4, 4)
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
    /// of truth for the retained runtime smoke preset.
    #[test]
    fn rule445_range_matches_doc() {
        let r445 = GameOfLife3D::rule445();
        assert_eq!((r445.survive_min, r445.survive_max), (4, 4));
        assert_eq!((r445.birth_min, r445.birth_max), (4, 4));
        assert_eq!(format!("{}", r445), "S4-4/B4-4");
    }

    #[test]
    fn dead_with_zero_neighbors_stays_dead_across_reference_rules() {
        let zero = neighbors_with_alive(0);
        for rule in [
            GameOfLife3D::new(9, 26, 5, 7),
            GameOfLife3D::new(0, 6, 1, 3),
            GameOfLife3D::rule445(),
            GameOfLife3D::new(4, 7, 6, 8),
        ] {
            assert_eq!(rule.step_cell(Cell::EMPTY, &zero), Cell::EMPTY);
        }
    }

    #[test]
    fn crystal_isolated_live_cell_survives() {
        let zero = neighbors_with_alive(0);
        assert_eq!(GameOfLife3D::new(0, 6, 1, 3).step_cell(ALIVE, &zero), ALIVE);
    }

    #[test]
    fn amoeba_isolated_live_cell_dies() {
        let zero = neighbors_with_alive(0);
        assert_eq!(
            GameOfLife3D::new(9, 26, 5, 7).step_cell(ALIVE, &zero),
            Cell::EMPTY
        );
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
        let rule = GameOfLife3D::new(0, 6, 1, 3);
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
    fn all_26_neighbors_alive_survival_by_reference_rule() {
        let full = neighbors_with_alive(26);
        assert_eq!(
            GameOfLife3D::new(9, 26, 5, 7).step_cell(ALIVE, &full),
            ALIVE
        );
        assert_eq!(
            GameOfLife3D::new(0, 6, 1, 3).step_cell(ALIVE, &full),
            Cell::EMPTY
        );
        assert_eq!(GameOfLife3D::rule445().step_cell(ALIVE, &full), Cell::EMPTY);
        assert_eq!(
            GameOfLife3D::new(4, 7, 6, 8).step_cell(ALIVE, &full),
            Cell::EMPTY
        );
    }

    // ---------------------------------------------------------------
    // BlockRule / BlockContext / block_index tests
    // ---------------------------------------------------------------

    #[test]
    fn block_index_matches_octant_convention() {
        // dx + dy*2 + dz*4
        assert_eq!(block_index(0, 0, 0), 0);
        assert_eq!(block_index(1, 0, 0), 1);
        assert_eq!(block_index(0, 1, 0), 2);
        assert_eq!(block_index(1, 1, 0), 3);
        assert_eq!(block_index(0, 0, 1), 4);
        assert_eq!(block_index(1, 0, 1), 5);
        assert_eq!(block_index(0, 1, 1), 6);
        assert_eq!(block_index(1, 1, 1), 7);
    }

    #[test]
    fn block_index_covers_all_8() {
        let mut seen = [false; 8];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let idx = block_index(dx, dy, dz);
                    assert!(!seen[idx], "duplicate index {idx}");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}
