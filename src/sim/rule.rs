use crate::octree::Cell;
use std::fmt;

/// Default "alive" cell payload used by the legacy GoL smoke scene.
pub(crate) const ALIVE: Cell = Cell::pack(1, 0);
const FACE_NEIGHBOR_INDICES: [usize; 6] = [4, 10, 12, 13, 15, 21];

/// A cellular automaton rule operating on the 26-cell Moore neighborhood in 3D.
///
/// Rules are pure local transforms: they inspect only the center cell and its
/// neighbors and return the next cell state for that center.
///
/// # Dispatch (hash-thing-vvun)
///
/// This is an enum, not a trait — replacing the former `Box<dyn CaRule + Send>`
/// vtable dispatch with a tagged match. Per jw3k.2 scout, dispatch removal
/// targets ~5-25% phase1 reduction at the 64³ water scout. Each payload keeps
/// its current struct and an inherent `step_cell` method; the enum `step_cell`
/// just dispatches to the right arm. `From<X> for CaRule` impls preserve the
/// readable call-site shape `registry.register_rule(AirRule { ... })`.
#[derive(Debug, Clone)]
pub enum CaRule {
    Air(AirRule),
    Fan(FanRule),
    FanDriven(Box<FanDrivenRule>),
    Noop(NoopRule),
    Fire(FireRule),
    Water(WaterRule),
    Lava(LavaRule),
    Ice(IceRule),
    Flammable(FlammableRule),
    Vine(VineRule),
    AirVineGrowth(AirVineGrowthRule),
    Acid(AcidRule),
    Dissolvable(DissolvableRule),
    Steam(SteamRule),
    Firework(FireworkRule),
    GameOfLife3D(GameOfLife3D),
}

impl CaRule {
    #[inline]
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        match self {
            CaRule::Air(r) => r.step_cell(center, neighbors),
            CaRule::Fan(_) => center,
            CaRule::FanDriven(r) => r.step_cell(center, neighbors),
            CaRule::Noop(_) => center,
            CaRule::Fire(r) => r.step_cell(center, neighbors),
            CaRule::Water(r) => r.step_cell(center, neighbors),
            CaRule::Lava(r) => r.step_cell(center, neighbors),
            CaRule::Ice(r) => r.step_cell(center, neighbors),
            CaRule::Flammable(r) => r.step_cell(center, neighbors),
            CaRule::Vine(r) => r.step_cell(center, neighbors),
            CaRule::AirVineGrowth(r) => r.step_cell(center, neighbors),
            CaRule::Acid(r) => r.step_cell(center, neighbors),
            CaRule::Dissolvable(r) => r.step_cell(center, neighbors),
            CaRule::Steam(r) => r.step_cell(center, neighbors),
            CaRule::Firework(r) => r.step_cell(center, neighbors),
            CaRule::GameOfLife3D(r) => r.step_cell(center, neighbors),
        }
    }

    /// True if this rule always returns the center cell unchanged (identity).
    /// Used in the base-case inner loop to skip neighbor collection.
    #[inline]
    pub fn is_noop(&self) -> bool {
        matches!(self, CaRule::Noop(_) | CaRule::Fan(_))
    }

    /// True if this rule is identity when all neighbors are the same material
    /// (self-inert). Used by hashlife subtree-level short-circuit: a subtree
    /// of uniform material with `is_self_inert() == true` can skip stepping
    /// because internal cells never have cross-material neighbors. Boundary
    /// interactions are handled by the 27-intermediate overlap in the parent.
    ///
    /// Every override on a former trait impl must appear here explicitly —
    /// the catch-all uses `is_noop()` (the former trait default). Missing an
    /// override silently regresses the hashlife inert-subtree short-circuit in
    /// `MaterialRegistry::cell_is_inert_fixed_point` (dropping self-inert for
    /// AirVineGrowthRule / FanDrivenRule kills the fast path for fan-static
    /// and uniform-air-with-vine subtrees).
    #[inline]
    pub fn is_self_inert(&self) -> bool {
        match self {
            CaRule::Air(_) => true,
            CaRule::AirVineGrowth(_) => true,
            CaRule::Dissolvable(_) => true,
            CaRule::FanDriven(r) => r.base.is_self_inert(),
            other => other.is_noop(),
        }
    }
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
///
/// `movable[i]` is true iff `block[i]` is opted into this rule (or is empty —
/// empty cells are always valid swap destinations). Rules MUST NOT produce a
/// permutation that moves an immovable cell: swapping rule-cells with
/// non-rule-cells would delete the rule-cell at the write-back layer, since
/// only rule/empty positions are written back.
pub trait BlockRule {
    fn step_block(&self, block: &[Cell; 8], movable: &[bool; 8]) -> [Cell; 8];

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

    fn index(self) -> usize {
        match self {
            Self::PosX => 0,
            Self::NegX => 1,
            Self::PosZ => 2,
            Self::NegZ => 3,
        }
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FanCarriedMaterial {
    pub base_material: u16,
    pub armed_materials: [u16; 4],
}

impl FanCarriedMaterial {
    pub const fn new(base_material: u16, armed_materials: [u16; 4]) -> Self {
        Self {
            base_material,
            armed_materials,
        }
    }

    fn armed_material(self, dir: FanDirection) -> u16 {
        self.armed_materials[dir.index()]
    }

    fn direction_for(self, material: u16) -> Option<FanDirection> {
        FanDirection::ALL
            .into_iter()
            .find(|&dir| self.armed_material(dir) == material)
    }
}

#[derive(Clone, Copy, Debug)]
enum FanTransport {
    MetadataBand,
    MaterialChannel(FanCarriedMaterial),
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

#[derive(Debug, Clone)]
pub struct AirRule {
    pub fan_material: u16,
    pub carried_materials: Vec<FanCarriedMaterial>,
    pub vine_growth: Option<AirVineGrowthRule>,
}

impl AirRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
            if let Some(base_material) = self.carried_materials.iter().find_map(|carried| {
                (carried.armed_material(dir) == source.material()).then_some(carried.base_material)
            }) {
                return Cell::pack(base_material, source.metadata());
            }
        }
        self.vine_growth
            .as_ref()
            .map(|rule| rule.step_cell(center, neighbors))
            .unwrap_or(Cell::EMPTY)
    }
}

impl From<AirRule> for CaRule {
    fn from(r: AirRule) -> CaRule {
        CaRule::Air(r)
    }
}

/// Fan cells are static, but their metadata stores optional facing:
/// 0 = omni, 1 = +X, 2 = -X, 3 = +Z, 4 = -Z.
#[derive(Clone, Copy, Debug)]
pub struct FanRule;

impl FanRule {
    #[inline]
    pub fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        center
    }
}

impl From<FanRule> for CaRule {
    fn from(r: FanRule) -> CaRule {
        CaRule::Fan(r)
    }
}

/// Wrap an existing rule with one-tick fan transport.
///
/// Most materials use the high metadata band for the transient "armed" state.
/// Age-bearing particles can instead route that state through dedicated
/// carry-material IDs so their payload metadata stays intact.
#[derive(Clone)]
pub struct FanDrivenRule {
    pub(crate) base: Box<CaRule>,
    fan_material: u16,
    transport: FanTransport,
}

impl fmt::Debug for FanDrivenRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanDrivenRule")
            .field("fan_material", &self.fan_material)
            .finish()
    }
}

impl FanDrivenRule {
    pub fn new(base: impl Into<CaRule>, fan_material: u16) -> Self {
        Self {
            base: Box::new(base.into()),
            fan_material,
            transport: FanTransport::MetadataBand,
        }
    }

    pub fn new_with_material_transport(
        base: impl Into<CaRule>,
        fan_material: u16,
        carried_material: FanCarriedMaterial,
    ) -> Self {
        Self {
            base: Box::new(base.into()),
            fan_material,
            transport: FanTransport::MaterialChannel(carried_material),
        }
    }

    fn decode_center(&self, center: Cell) -> Cell {
        match self.transport {
            FanTransport::MetadataBand => clear_push_metadata(center),
            FanTransport::MaterialChannel(carried_material) => carried_material
                .direction_for(center.material())
                .map(|_| Cell::pack(carried_material.base_material, center.metadata()))
                .unwrap_or(center),
        }
    }

    fn armed_direction(&self, center: Cell) -> Option<FanDirection> {
        match self.transport {
            FanTransport::MetadataBand => FanDirection::from_armed_metadata(center.metadata()),
            FanTransport::MaterialChannel(carried_material) => {
                carried_material.direction_for(center.material())
            }
        }
    }

    fn arm_cell(&self, next: Cell, dir: FanDirection) -> Cell {
        match self.transport {
            FanTransport::MetadataBand => Cell::pack(next.material(), dir.armed_metadata()),
            FanTransport::MaterialChannel(carried_material) => {
                Cell::pack(carried_material.armed_material(dir), next.metadata())
            }
        }
    }
}

impl FanDrivenRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        let center_base = self.decode_center(center);
        let base_next = self.base.step_cell(center_base, neighbors);
        if base_next.is_empty() || base_next.material() != center_base.material() {
            return base_next;
        }

        if let Some(dir) = self.armed_direction(center) {
            let (dx, dy, dz) = dir.delta();
            if neighbor_at(neighbors, dx, dy, dz).is_empty() {
                return Cell::EMPTY;
            }
            return base_next;
        }

        if let Some(dir) = arming_direction(neighbors, self.fan_material) {
            return self.arm_cell(base_next, dir);
        }

        base_next
    }
}

impl From<FanDrivenRule> for CaRule {
    fn from(r: FanDrivenRule) -> CaRule {
        CaRule::FanDriven(Box::new(r))
    }
}

/// Identity rule for static materials. Returns the center cell unchanged.
#[derive(Debug, Clone, Copy)]
pub struct NoopRule;

impl NoopRule {
    #[inline]
    pub fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        center
    }
}

impl From<NoopRule> for CaRule {
    fn from(r: NoopRule) -> CaRule {
        CaRule::Noop(r)
    }
}

/// Fire persists while fuel is adjacent, and is quenched by water.
#[derive(Debug, Clone)]
pub struct FireRule {
    pub fuel_materials: Vec<u16>,
    pub quencher_material: u16,
}

impl FireRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<FireRule> for CaRule {
    fn from(r: FireRule) -> CaRule {
        CaRule::Fire(r)
    }
}

/// Water solidifies into a configured product when the reactive material is adjacent.
#[derive(Debug, Clone)]
pub struct WaterRule {
    pub reactive_material: u16,
    pub reaction_product: Cell,
}

impl WaterRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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

impl From<WaterRule> for CaRule {
    fn from(r: WaterRule) -> CaRule {
        CaRule::Water(r)
    }
}

/// Lava converts adjacent water to stone and adjacent grass to fire.
///
/// If any neighbor is water, this cell becomes stone (lava solidifies).
/// Otherwise the lava persists, and any adjacent grass/flammable material
/// is expected to catch fire via the terrain's FireRule on the next tick.
/// The CaRule only transforms the center cell — neighbor ignition is a
/// side-effect of the fire rule's fuel check seeing lava-heated grass.
#[derive(Debug, Clone)]
pub struct LavaRule {
    pub water_material: u16,
    pub solidify_product: Cell,
}

impl LavaRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<LavaRule> for CaRule {
    fn from(r: LavaRule) -> CaRule {
        CaRule::Lava(r)
    }
}

/// Ice melts into water when fire or lava is adjacent.
#[derive(Debug, Clone)]
pub struct IceRule {
    pub heat_materials: Vec<u16>,
    pub melt_product: Cell,
}

impl IceRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<IceRule> for CaRule {
    fn from(r: IceRule) -> CaRule {
        CaRule::Ice(r)
    }
}

/// Flammable material catches fire from adjacent fire.
#[derive(Debug, Clone)]
pub struct FlammableRule {
    pub fire_material: u16,
    pub fire_product: Cell,
}

impl FlammableRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<FlammableRule> for CaRule {
    fn from(r: FlammableRule) -> CaRule {
        CaRule::Flammable(r)
    }
}

/// Vine ages slowly, ignites next to fire, and dissolves next to acid.
#[derive(Debug, Clone)]
pub struct VineRule {
    pub fire_material: u16,
    pub acid_material: u16,
    pub max_age: u16,
}

impl VineRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "VineRule should not dispatch for AIR");
        if neighbors.iter().any(|n| n.material() == self.acid_material) {
            Cell::EMPTY
        } else if neighbors.iter().any(|n| n.material() == self.fire_material) {
            Cell::pack(self.fire_material, 0)
        } else {
            Cell::pack(
                center.material(),
                center.metadata().saturating_add(1).min(self.max_age),
            )
        }
    }
}

impl From<VineRule> for CaRule {
    fn from(r: VineRule) -> CaRule {
        CaRule::Vine(r)
    }
}

/// Air is inert unless a mature vine can creep onto a supported face-adjacent cell.
#[derive(Debug, Clone)]
pub struct AirVineGrowthRule {
    pub vine_material: u16,
    pub support_materials: Vec<u16>,
    pub spread_age: u16,
}

impl AirVineGrowthRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(
            center.is_empty(),
            "AirVineGrowthRule should dispatch only for AIR"
        );
        let has_mature_vine = FACE_NEIGHBOR_INDICES.iter().any(|&i| {
            let neighbor = neighbors[i];
            neighbor.material() == self.vine_material && neighbor.metadata() >= self.spread_age
        });
        let has_support = FACE_NEIGHBOR_INDICES
            .iter()
            .any(|&i| self.support_materials.contains(&neighbors[i].material()));
        if has_mature_vine && has_support {
            Cell::pack(self.vine_material, 0)
        } else {
            center
        }
    }
}

impl From<AirVineGrowthRule> for CaRule {
    fn from(r: AirVineGrowthRule) -> CaRule {
        CaRule::AirVineGrowth(r)
    }
}

/// Acid is consumed when touching a dissolvable material.
/// Both acid and the target dissolve (target handled by DissolvableRule).
#[derive(Debug, Clone)]
pub struct AcidRule {
    pub dissolvable_materials: Vec<u16>,
}

impl AcidRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<AcidRule> for CaRule {
    fn from(r: AcidRule) -> CaRule {
        CaRule::Acid(r)
    }
}

/// Material dissolves when adjacent to acid.
#[derive(Debug, Clone)]
pub struct DissolvableRule {
    pub acid_material: u16,
}

impl DissolvableRule {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<DissolvableRule> for CaRule {
    fn from(r: DissolvableRule) -> CaRule {
        CaRule::Dissolvable(r)
    }
}

/// Steam ages via metadata and condenses to water after a threshold.
#[derive(Debug, Clone)]
pub struct SteamRule {
    pub condense_product: Cell,
    pub max_age: u16,
}

impl SteamRule {
    pub fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
        debug_assert!(!center.is_empty(), "SteamRule should not dispatch for AIR");
        let age = center.metadata();
        if age >= self.max_age {
            self.condense_product
        } else {
            Cell::pack(center.material(), age + 1)
        }
    }
}

impl From<SteamRule> for CaRule {
    fn from(r: SteamRule) -> CaRule {
        CaRule::Steam(r)
    }
}

/// Firework rises (via negative density in block rule) and explodes into fire
/// after reaching a metadata age threshold.
#[derive(Debug, Clone)]
pub struct FireworkRule {
    pub explode_product: Cell,
    pub fuse_length: u16,
}

impl FireworkRule {
    pub fn step_cell(&self, center: Cell, _neighbors: &[Cell; 26]) -> Cell {
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
}

impl From<FireworkRule> for CaRule {
    fn from(r: FireworkRule) -> CaRule {
        CaRule::Firework(r)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl GameOfLife3D {
    pub fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell {
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

impl From<GameOfLife3D> for CaRule {
    fn from(r: GameOfLife3D) -> CaRule {
        CaRule::GameOfLife3D(r)
    }
}

#[cfg(test)]
mod tests {
    //! Unit coverage for `GameOfLife3D` and the first registry-backed rules.

    use super::*;

    /// Guards against a silent perf regression: if a future variant bloats the
    /// enum above the cache-friendly budget, force the author to either box the
    /// variant or consciously raise this cap. The 96-byte ceiling is picked per
    /// the vvun plan (~72-byte estimate for the largest inline variant
    /// `AirRule`, with headroom for alignment). See
    /// `.ship-notes/plan-flint-hash-thing-vvun-enum-dispatch-carule.md`.
    #[test]
    fn carule_enum_size_budget() {
        let bytes = std::mem::size_of::<CaRule>();
        eprintln!("size_of::<CaRule>() = {bytes}");
        assert!(
            bytes <= 96,
            "CaRule grew to {bytes} bytes — box the largest variant (likely AirRule) \
             or raise the ceiling with measurement evidence."
        );
    }

    /// Pins the `is_self_inert` truth table. The centralised match in
    /// `CaRule::is_self_inert` uses a wildcard fall-through to `is_noop()` —
    /// without this test, a future variant that needs to be self-inert could
    /// silently regress `MaterialRegistry::cell_is_inert_fixed_point()` and
    /// lose hashlife's inert-subtree fast path. Update the expected arms here
    /// whenever an enum variant is added.
    #[test]
    fn carule_is_self_inert_truth_table() {
        let air: CaRule = AirRule {
            fan_material: 0,
            carried_materials: Vec::new(),
            vine_growth: None,
        }
        .into();
        assert!(air.is_self_inert(), "AirRule must be self-inert");

        let avg: CaRule = AirVineGrowthRule {
            vine_material: 1,
            support_materials: Vec::new(),
            spread_age: 0,
        }
        .into();
        assert!(avg.is_self_inert(), "AirVineGrowthRule must be self-inert");

        let dissolvable: CaRule = DissolvableRule { acid_material: 1 }.into();
        assert!(
            dissolvable.is_self_inert(),
            "DissolvableRule must be self-inert"
        );

        // FanDriven delegates to its base.
        let fan_over_air: CaRule = FanDrivenRule::new(
            AirRule {
                fan_material: 0,
                carried_materials: Vec::new(),
                vine_growth: None,
            },
            1,
        )
        .into();
        assert!(
            fan_over_air.is_self_inert(),
            "FanDriven(Air) must inherit self-inert from its base"
        );
        let fan_over_water: CaRule = FanDrivenRule::new(
            WaterRule {
                reactive_material: 1,
                reaction_product: Cell::EMPTY,
            },
            1,
        )
        .into();
        assert!(
            !fan_over_water.is_self_inert(),
            "FanDriven(Water) must not be self-inert (water is reactive)"
        );

        // Reactive rules are never self-inert.
        let water: CaRule = WaterRule {
            reactive_material: 1,
            reaction_product: Cell::EMPTY,
        }
        .into();
        assert!(!water.is_self_inert(), "WaterRule must not be self-inert");

        // Noop/Fan wildcard arms fall through to is_noop(). This line
        // pins the current behaviour so any future change is forced to
        // update both arms deliberately (not silently).
        let noop: CaRule = NoopRule.into();
        assert_eq!(
            noop.is_self_inert(),
            noop.is_noop(),
            "NoopRule wildcard arm must keep delegating to is_noop()"
        );
    }

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
        let rule = AirRule {
            fan_material: 16,
            carried_materials: Vec::new(),
            vine_growth: None,
        };
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
    fn air_rule_accepts_material_channel_incoming_material() {
        let rule = AirRule {
            fan_material: 16,
            carried_materials: vec![FanCarriedMaterial::new(6, [30, 31, 32, 33])],
            vine_growth: None,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        set_neighbor(&mut neighbors, -1, 0, 0, Cell::pack(30, 11));

        assert_eq!(rule.step_cell(Cell::EMPTY, &neighbors), Cell::pack(6, 11));
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
        set_neighbor(&mut neighbors, 1, 0, 0, Cell::pack(16, 1));

        assert_eq!(
            rule.step_cell(Cell::pack(6, 0), &neighbors),
            Cell::pack(6, 0)
        );
    }

    #[test]
    fn fan_driven_rule_material_channel_preserves_payload_metadata() {
        let rule = FanDrivenRule::new_with_material_transport(
            SteamRule {
                condense_product: Cell::pack(5, 0),
                max_age: 20,
            },
            16,
            FanCarriedMaterial::new(6, [30, 31, 32, 33]),
        );
        let mut arm_neighbors = [Cell::EMPTY; 26];
        set_neighbor(&mut arm_neighbors, -1, 0, 0, Cell::pack(16, 0));
        assert_eq!(
            rule.step_cell(Cell::pack(6, 12), &arm_neighbors),
            Cell::pack(30, 13)
        );

        let mut blocked_neighbors = [Cell::EMPTY; 26];
        set_neighbor(&mut blocked_neighbors, 1, 0, 0, Cell::pack(9, 0));
        assert_eq!(
            rule.step_cell(Cell::pack(30, 13), &blocked_neighbors),
            Cell::pack(6, 14)
        );
    }

    #[test]
    fn vine_rule_ages_when_undisturbed() {
        let rule = VineRule {
            fire_material: 4,
            acid_material: 9,
            max_age: 3,
        };
        assert_eq!(
            rule.step_cell(Cell::pack(15, 0), &[Cell::EMPTY; 26]),
            Cell::pack(15, 1)
        );
        assert_eq!(
            rule.step_cell(Cell::pack(15, 3), &[Cell::EMPTY; 26]),
            Cell::pack(15, 3)
        );
    }

    #[test]
    fn vine_rule_burns_or_dissolves_from_face_neighbors() {
        let rule = VineRule {
            fire_material: 4,
            acid_material: 9,
            max_age: 3,
        };
        let mut fire_neighbors = [Cell::EMPTY; 26];
        fire_neighbors[12] = mat(4);
        assert_eq!(rule.step_cell(Cell::pack(15, 2), &fire_neighbors), mat(4));

        let mut acid_neighbors = [Cell::EMPTY; 26];
        acid_neighbors[21] = mat(9);
        assert_eq!(
            rule.step_cell(Cell::pack(15, 2), &acid_neighbors),
            Cell::EMPTY
        );
    }

    #[test]
    fn air_vine_growth_rule_requires_mature_vine_and_support() {
        let rule = AirVineGrowthRule {
            vine_material: 15,
            support_materials: vec![1, 2],
            spread_age: 2,
        };
        let mut neighbors = [Cell::EMPTY; 26];
        neighbors[12] = Cell::pack(15, 2);
        neighbors[10] = mat(1);
        assert_eq!(rule.step_cell(Cell::EMPTY, &neighbors), mat(15));

        neighbors[12] = Cell::pack(15, 1);
        assert_eq!(rule.step_cell(Cell::EMPTY, &neighbors), Cell::EMPTY);

        neighbors[12] = Cell::pack(15, 2);
        neighbors[10] = Cell::EMPTY;
        assert_eq!(rule.step_cell(Cell::EMPTY, &neighbors), Cell::EMPTY);
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
