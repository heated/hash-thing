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

    /// True if this rule always returns the center cell unchanged (identity).
    /// Used in the base-case inner loop to skip neighbor collection.
    fn is_noop(&self) -> bool {
        false
    }

    /// True if this rule is identity when all neighbors are the same material
    /// (self-inert). Used by hashlife subtree-level short-circuit: a subtree
    /// of uniform material with `is_self_inert() == true` can skip stepping
    /// because internal cells never have cross-material neighbors. Boundary
    /// interactions are handled by the 27-intermediate overlap in the parent.
    ///
    /// Default: delegates to `is_noop()`. Override for conditionally-noop rules
    /// like DissolvableRule (identity unless acid is adjacent).
    fn is_self_inert(&self) -> bool {
        self.is_noop()
    }

    /// Clone into a boxed trait object. Enables Clone for MaterialRegistry
    /// (and transitively World) for snapshots, undo, and testing.
    fn clone_box(&self) -> Box<dyn CaRule + Send>;
}

/// A block-based CA rule operating on 2x2x2 cell blocks.
///
/// Block rules are pure functions of block contents — no position, generation,
/// or RNG context. This enables spatial memoization: identical blocks at
/// different positions produce identical results, so hashlife can cache and
/// reuse block-rule outputs across the entire tree.
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
    fn step_block(&self, block: &[Cell; 8]) -> [Cell; 8];

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn BlockRule + Send>;
}

/// Map local (dx, dy, dz) offsets (each 0 or 1) to an index in `[Cell; 8]`.
///
/// Matches `octant_index` in `src/octree/node.rs` — this is a load-bearing
/// invariant. Do not change one without the other.
#[inline]
pub const fn block_index(dx: usize, dy: usize, dz: usize) -> usize {
    dx + dy * 2 + dz * 4
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FanDirection {
    PosX,
    NegX,
    PosZ,
    NegZ,
}

impl FanDirection {
    const ALL: [Self; 4] = [Self::PosX, Self::NegX, Self::PosZ, Self::NegZ];
    const ARMED_METADATA_BASE: u16 = 60;

    fn delta(self) -> (i32, i32, i32) {
        match self {
            Self::PosX => (1, 0, 0),
            Self::NegX => (-1, 0, 0),
            Self::PosZ => (0, 0, 1),
            Self::NegZ => (0, 0, -1),
        }
    }

    fn from_fan_metadata(metadata: u16) -> Option<Self> {
        match metadata {
            1 => Some(Self::PosX),
            2 => Some(Self::NegX),
            3 => Some(Self::PosZ),
            4 => Some(Self::NegZ),
            _ => None,
        }
    }

    fn armed_metadata(self) -> u16 {
        Self::ARMED_METADATA_BASE
            + match self {
                Self::PosX => 0,
                Self::NegX => 1,
                Self::PosZ => 2,
                Self::NegZ => 3,
            }
    }

    fn from_armed_metadata(metadata: u16) -> Option<Self> {
        match metadata {
            60 => Some(Self::PosX),
            61 => Some(Self::NegX),
            62 => Some(Self::PosZ),
            63 => Some(Self::NegZ),
            _ => None,
        }
    }
}

fn neighbor_at(neighbors: &[Cell; 26], dx: i32, dy: i32, dz: i32) -> Cell {
    debug_assert!((-1..=1).contains(&dx));
    debug_assert!((-1..=1).contains(&dy));
    debug_assert!((-1..=1).contains(&dz));
    debug_assert!(dx != 0 || dy != 0 || dz != 0);

    let mut idx = 0;
    for ndz in [-1, 0, 1] {
        for ndy in [-1, 0, 1] {
            for ndx in [-1, 0, 1] {
                if ndx == 0 && ndy == 0 && ndz == 0 {
                    continue;
                }
                if ndx == dx && ndy == dy && ndz == dz {
                    return neighbors[idx];
                }
                idx += 1;
            }
        }
    }
    unreachable!("neighbor offset must resolve to one of the 26 Moore neighbors");
}

fn clear_push_metadata(cell: Cell) -> Cell {
    if cell.is_empty() || FanDirection::from_armed_metadata(cell.metadata()).is_none() {
        cell
    } else {
        Cell::pack(cell.material(), 0)
    }
}

fn fan_pushes_direction(fan: Cell, fan_material: u16, dir: FanDirection) -> bool {
    fan.material() == fan_material
        && (fan.metadata() == 0 || FanDirection::from_fan_metadata(fan.metadata()) == Some(dir))
}

fn arming_direction(neighbors: &[Cell; 26], fan_material: u16) -> Option<FanDirection> {
    FanDirection::ALL.into_iter().find(|&dir| {
        let (dx, dy, dz) = dir.delta();
        let behind = neighbor_at(neighbors, -dx, -dy, -dz);
        let ahead = neighbor_at(neighbors, dx, dy, dz);
        fan_pushes_direction(behind, fan_material, dir) && ahead.is_empty()
    })
}

#[derive(Clone, Copy, Debug)]
pub struct AirRule {
    pub fan_material: u16,
}

impl CaRule for AirRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(center.is_empty(), "AirRule should only dispatch for AIR");
        for dir in FanDirection::ALL {
            let (dx, dy, dz) = dir.delta();
            let source = neighbor_at(neighbors, -dx, -dy, -dz);
            if source.material() == self.fan_material {
                continue;
            }
            if FanDirection::from_armed_metadata(source.metadata()) == Some(dir) {
                return Cell::pack(source.material(), 0);
            }
        }
        Cell::EMPTY
    }

    fn is_self_inert(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(*self)
    }
}

/// Fan cells are static, but their metadata stores optional facing:
/// 0 = omni, 1 = +X, 2 = -X, 3 = +Z, 4 = -Z.
#[derive(Clone, Copy, Debug)]
pub struct FanRule;

impl CaRule for FanRule {
    fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        center
    }

    fn is_noop(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(*self)
    }
}

/// Wrap an existing rule with one-tick fan transport using the high metadata band.
pub struct FanDrivenRule {
    base: Box<dyn CaRule + Send>,
    fan_material: u16,
}

impl fmt::Debug for FanDrivenRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanDrivenRule")
            .field("fan_material", &self.fan_material)
            .finish()
    }
}

impl FanDrivenRule {
    pub fn new<R>(base: R, fan_material: u16) -> Self
    where
        R: CaRule + Send + 'static,
    {
        Self {
            base: Box::new(base),
            fan_material,
        }
    }
}

impl CaRule for FanDrivenRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        let base_next = self.base.step_cell(clear_push_metadata(center), neighbors);
        if base_next.is_empty() || base_next.material() != center.material() {
            return base_next;
        }

        if let Some(dir) = FanDirection::from_armed_metadata(center.metadata()) {
            let (dx, dy, dz) = dir.delta();
            if neighbor_at(neighbors, dx, dy, dz).is_empty() {
                return Cell::EMPTY;
            }
            return Cell::pack(base_next.material(), 0);
        }

        if let Some(dir) = arming_direction(neighbors, self.fan_material) {
            return Cell::pack(base_next.material(), dir.armed_metadata());
        }

        Cell::pack(base_next.material(), 0)
    }

    fn is_self_inert(&self) -> bool {
        self.base.is_self_inert()
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(Self {
            base: self.base.clone_box(),
            fan_material: self.fan_material,
        })
    }
}

/// Identity rule for static materials. Returns the center cell unchanged.
#[derive(Debug)]
pub struct NoopRule;

impl CaRule for NoopRule {
    fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        center
    }

    fn is_noop(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(NoopRule)
    }
}

/// Fire persists while fuel is adjacent, and is quenched by water.
#[derive(Debug, Clone)]
pub struct FireRule {
    pub fuel_materials: Vec<u16>,
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
            .any(|neighbor| self.fuel_materials.contains(&neighbor.material()))
        {
            center
        } else {
            Cell::EMPTY
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(self.clone())
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

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(WaterRule {
            reactive_material: self.reactive_material,
            reaction_product: self.reaction_product,
        })
    }
}

/// Lava converts adjacent water to stone and adjacent grass to fire.
///
/// If any neighbor is water, this cell becomes stone (lava solidifies).
/// Otherwise the lava persists, and any adjacent grass/flammable material
/// is expected to catch fire via the terrain's FireRule on the next tick.
/// The CaRule only transforms the center cell — neighbor ignition is a
/// side-effect of the fire rule's fuel check seeing lava-heated grass.
#[derive(Debug)]
pub struct LavaRule {
    pub water_material: u16,
    pub solidify_product: Cell,
}

impl CaRule for LavaRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "LavaRule should not dispatch for AIR");
        if neighbors
            .iter()
            .any(|neighbor| neighbor.material() == self.water_material)
        {
            self.solidify_product
        } else {
            center
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(LavaRule {
            water_material: self.water_material,
            solidify_product: self.solidify_product,
        })
    }
}

/// Ice melts into water when fire or lava is adjacent.
#[derive(Debug, Clone)]
pub struct IceRule {
    pub heat_materials: Vec<u16>,
    pub melt_product: Cell,
}

impl CaRule for IceRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "IceRule should not dispatch for AIR");
        if neighbors
            .iter()
            .any(|n| self.heat_materials.contains(&n.material()))
        {
            self.melt_product
        } else {
            center
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(self.clone())
    }
}

/// Flammable material catches fire from adjacent fire.
#[derive(Debug)]
pub struct FlammableRule {
    pub fire_material: u16,
    pub fire_product: Cell,
}

impl CaRule for FlammableRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(
            !center.is_empty(),
            "FlammableRule should not dispatch for AIR"
        );
        if neighbors.iter().any(|n| n.material() == self.fire_material) {
            self.fire_product
        } else {
            center
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(FlammableRule {
            fire_material: self.fire_material,
            fire_product: self.fire_product,
        })
    }
}

/// Acid is consumed when touching a dissolvable material.
/// Both acid and the target dissolve (target handled by DissolvableRule).
#[derive(Debug, Clone)]
pub struct AcidRule {
    pub dissolvable_materials: Vec<u16>,
}

impl CaRule for AcidRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "AcidRule should not dispatch for AIR");
        if neighbors
            .iter()
            .any(|n| self.dissolvable_materials.contains(&n.material()))
        {
            Cell::EMPTY
        } else {
            center
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(self.clone())
    }
}

/// Material dissolves when adjacent to acid.
#[derive(Debug)]
pub struct DissolvableRule {
    pub acid_material: u16,
}

impl CaRule for DissolvableRule {
    fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(
            !center.is_empty(),
            "DissolvableRule should not dispatch for AIR"
        );
        if neighbors.iter().any(|n| n.material() == self.acid_material) {
            Cell::EMPTY
        } else {
            center
        }
    }

    fn is_self_inert(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(DissolvableRule {
            acid_material: self.acid_material,
        })
    }
}

/// Steam ages via metadata and condenses to water after a threshold.
#[derive(Debug)]
pub struct SteamRule {
    pub condense_product: Cell,
    pub max_age: u16,
}

impl CaRule for SteamRule {
    fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "SteamRule should not dispatch for AIR");
        let age = center.metadata();
        if age >= self.max_age {
            self.condense_product
        } else {
            Cell::pack(center.material(), age + 1)
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(SteamRule {
            condense_product: self.condense_product,
            max_age: self.max_age,
        })
    }
}

/// Firework rises (via negative density in block rule) and explodes into fire
/// after reaching a metadata age threshold.
#[derive(Debug)]
pub struct FireworkRule {
    pub explode_product: Cell,
    pub fuse_length: u16,
}

impl CaRule for FireworkRule {
    fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(
            !center.is_empty(),
            "FireworkRule should not dispatch for AIR"
        );
        let age = center.metadata();
        if age >= self.fuse_length {
            self.explode_product
        } else {
            Cell::pack(center.material(), age + 1)
        }
    }

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(FireworkRule {
            explode_product: self.explode_product,
            fuse_length: self.fuse_length,
        })
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

    fn clone_box(&self) -> Box<dyn CaRule + Send> {
        Box::new(*self)
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
            fuel_materials: vec![3],
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
            fuel_materials: vec![3],
            quencher_material: 5,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(3);
        assert_eq!(rule.step_cell(mat(4), &neighbors), mat(4));
    }

    #[test]
    fn fire_rule_quencher_beats_fuel_and_extinguishes() {
        let rule = FireRule {
            fuel_materials: vec![3],
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
            fuel_materials: vec![3],
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
    fn lava_rule_solidifies_on_water_contact() {
        let rule = LavaRule {
            water_material: 5,
            solidify_product: mat(1),
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[3] = mat(5); // water neighbor
        assert_eq!(
            rule.step_cell(mat(7), &neighbors),
            mat(1),
            "lava adjacent to water should solidify into stone"
        );
    }

    #[test]
    fn lava_rule_persists_without_water() {
        let rule = LavaRule {
            water_material: 5,
            solidify_product: mat(1),
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(3); // grass, not water
        let lava = Cell::pack(7, 4);
        assert_eq!(
            rule.step_cell(lava, &neighbors),
            lava,
            "lava without water neighbors should persist unchanged"
        );
    }

    #[test]
    fn lava_rule_solidifies_to_configured_product() {
        let product = Cell::pack(2, 5);
        let rule = LavaRule {
            water_material: 5,
            solidify_product: product,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[0] = mat(5);
        assert_eq!(rule.step_cell(mat(7), &neighbors), product);
    }

    fn set_neighbor(neighbors: &mut [Cell; 26], dx: i32, dy: i32, dz: i32, cell: Cell) {
        let mut idx = 0;
        for ndz in [-1, 0, 1] {
            for ndy in [-1, 0, 1] {
                for ndx in [-1, 0, 1] {
                    if ndx == 0 && ndy == 0 && ndz == 0 {
                        continue;
                    }
                    if ndx == dx && ndy == dy && ndz == dz {
                        neighbors[idx] = cell;
                        return;
                    }
                    idx += 1;
                }
            }
        }
        panic!("invalid neighbor offset ({dx}, {dy}, {dz})");
    }

    #[test]
    fn air_rule_accepts_armed_incoming_material() {
        let rule = AirRule { fan_material: 16 };
        let mut neighbors = [Cell::EMPTY; 26];
        set_neighbor(
            &mut neighbors,
            -1,
            0,
            0,
            Cell::pack(6, FanDirection::PosX.armed_metadata()),
        );

        assert_eq!(rule.step_cell(Cell::EMPTY, &neighbors), Cell::pack(6, 0));
    }

    #[test]
    fn fan_driven_rule_arms_material_when_fan_has_space_to_blow_into() {
        let rule = FanDrivenRule::new(NoopRule, 16);
        let mut neighbors = [Cell::EMPTY; 26];
        set_neighbor(&mut neighbors, -1, 0, 0, Cell::pack(16, 0));

        assert_eq!(
            rule.step_cell(Cell::pack(6, 0), &neighbors),
            Cell::pack(6, FanDirection::PosX.armed_metadata())
        );
    }

    #[test]
    fn fan_driven_rule_vacates_after_being_armed() {
        let rule = FanDrivenRule::new(NoopRule, 16);
        let neighbors = [Cell::EMPTY; 26];

        assert_eq!(
            rule.step_cell(
                Cell::pack(6, FanDirection::PosX.armed_metadata()),
                &neighbors
            ),
            Cell::EMPTY
        );
    }

    #[test]
    fn fan_driven_rule_respects_fan_facing_metadata() {
        let rule = FanDrivenRule::new(NoopRule, 16);
        let mut neighbors = [Cell::EMPTY; 26];
        set_neighbor(
            &mut neighbors,
            1,
            0,
            0,
            Cell::pack(16, 1),
        );

        assert_eq!(rule.step_cell(Cell::pack(6, 0), &neighbors), Cell::pack(6, 0));
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
