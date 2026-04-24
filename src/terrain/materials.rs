//! Material constants and registry for terrain + CA dispatch.
//!
//! `AIR == 0` remains load-bearing for the octree, renderer, and hashing
//! layers. The registry mirrors that encoding: material ID 0 is the empty
//! cell, and every other material ID maps to a validated `CellState`.

use std::fmt;

use crate::octree::{Cell, CellState};
use crate::sim::margolus::FluidBlockRule;
use crate::sim::margolus::GravityBlockRule;
use crate::sim::rule::{
    AcidRule, AirRule, AirVineGrowthRule, BlockRule, CaRule, DissolvableRule, FanCarriedMaterial,
    FanDrivenRule, FanRule, FireRule, FireworkRule, FlammableRule, GameOfLife3D, IceRule, LavaRule,
    NoopRule, SteamRule, VineRule, WaterRule,
};

pub type MaterialId = u16;

pub const INITIAL_MATERIAL_SLOTS: usize = 256;

/// Least common multiple for u64. Panics if the registry would produce a
/// memo period that overflows u64 (requires truly pathological divisors;
/// surface the error at registry-build time rather than silently wrapping).
fn lcm_u64(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd_u64(a, b))
        .checked_mul(b)
        .expect("tick_divisor LCM overflowed u64 — pick smaller divisors")
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub const AIR_MATERIAL_ID: MaterialId = 0;
pub const STONE_MATERIAL_ID: MaterialId = 1;
pub const DIRT_MATERIAL_ID: MaterialId = 2;
pub const GRASS_MATERIAL_ID: MaterialId = 3;
pub const FIRE_MATERIAL_ID: MaterialId = 4;
pub const WATER_MATERIAL_ID: MaterialId = 5;
pub const SAND_MATERIAL_ID: MaterialId = 6;
pub const LAVA_MATERIAL_ID: MaterialId = 7;
pub const ICE_MATERIAL_ID: MaterialId = 8;
pub const ACID_MATERIAL_ID: MaterialId = 9;
pub const OIL_MATERIAL_ID: MaterialId = 10;
pub const GUNPOWDER_MATERIAL_ID: MaterialId = 11;
pub const STEAM_MATERIAL_ID: MaterialId = 12;
pub const GAS_MATERIAL_ID: MaterialId = 13;
pub const METAL_MATERIAL_ID: MaterialId = 14;
pub const VINE_MATERIAL_ID: MaterialId = 15;
pub const FAN_MATERIAL_ID: MaterialId = 16;
pub const FIREWORK_MATERIAL_ID: MaterialId = 17;
pub const CLONE_MATERIAL_ID: MaterialId = 18;
const FAN_STEAM_POS_X_MATERIAL_ID: MaterialId = 19;
const FAN_STEAM_NEG_X_MATERIAL_ID: MaterialId = 20;
const FAN_STEAM_POS_Z_MATERIAL_ID: MaterialId = 21;
const FAN_STEAM_NEG_Z_MATERIAL_ID: MaterialId = 22;
const FAN_FIREWORK_POS_X_MATERIAL_ID: MaterialId = 23;
const FAN_FIREWORK_NEG_X_MATERIAL_ID: MaterialId = 24;
const FAN_FIREWORK_POS_Z_MATERIAL_ID: MaterialId = 25;
const FAN_FIREWORK_NEG_Z_MATERIAL_ID: MaterialId = 26;
pub(crate) const FAN_ARMED_STEAM_MATERIAL_IDS: [MaterialId; 4] = [
    FAN_STEAM_POS_X_MATERIAL_ID,
    FAN_STEAM_NEG_X_MATERIAL_ID,
    FAN_STEAM_POS_Z_MATERIAL_ID,
    FAN_STEAM_NEG_Z_MATERIAL_ID,
];
pub(crate) const FAN_ARMED_FIREWORK_MATERIAL_IDS: [MaterialId; 4] = [
    FAN_FIREWORK_POS_X_MATERIAL_ID,
    FAN_FIREWORK_NEG_X_MATERIAL_ID,
    FAN_FIREWORK_POS_Z_MATERIAL_ID,
    FAN_FIREWORK_NEG_Z_MATERIAL_ID,
];

pub const AIR: CellState = Cell::EMPTY.raw();
pub const STONE: CellState = Cell::pack(STONE_MATERIAL_ID, 0).raw();
pub const DIRT: CellState = Cell::pack(DIRT_MATERIAL_ID, 0).raw();
pub const GRASS: CellState = Cell::pack(GRASS_MATERIAL_ID, 0).raw();
pub const FIRE: CellState = Cell::pack(FIRE_MATERIAL_ID, 0).raw();
pub const WATER: CellState = Cell::pack(WATER_MATERIAL_ID, 0).raw();
pub const SAND: CellState = Cell::pack(SAND_MATERIAL_ID, 0).raw();
pub const LAVA: CellState = Cell::pack(LAVA_MATERIAL_ID, 0).raw();
pub const ICE: CellState = Cell::pack(ICE_MATERIAL_ID, 0).raw();
pub const ACID: CellState = Cell::pack(ACID_MATERIAL_ID, 0).raw();
pub const OIL: CellState = Cell::pack(OIL_MATERIAL_ID, 0).raw();
pub const GUNPOWDER: CellState = Cell::pack(GUNPOWDER_MATERIAL_ID, 0).raw();
pub const STEAM: CellState = Cell::pack(STEAM_MATERIAL_ID, 0).raw();
pub const GAS: CellState = Cell::pack(GAS_MATERIAL_ID, 0).raw();
pub const METAL: CellState = Cell::pack(METAL_MATERIAL_ID, 0).raw();
pub const VINE: CellState = Cell::pack(VINE_MATERIAL_ID, 0).raw();
pub const FAN: CellState = Cell::pack(FAN_MATERIAL_ID, 0).raw();
pub const FIREWORK: CellState = Cell::pack(FIREWORK_MATERIAL_ID, 0).raw();
pub const CLONE: CellState = Cell::pack(CLONE_MATERIAL_ID, 0).raw();

/// Pack a CLONE cell whose metadata slot carries the source material id.
///
/// The "spawn material X" payload is stashed in the Cell's 6-bit metadata
/// field (`Cell::MAX_METADATA = 63`) rather than a side-table, so this
/// path caps source materials at 63 even though `Cell::MAX_MATERIAL` is
/// 1023. Current live materials all fit, but the cap is latent: adding a
/// material id > 63 would silently route through `Cell::pack`'s generic
/// "metadata overflows 6 bits" assert at two call sites
/// (`World::place_clone_source`, main's clone-block placement). Readback
/// lives in `World::spawn_clones` via `cell.metadata()`, which is
/// implicitly bounded the same way — a widener (side-table keyed by CLONE
/// position) would need to update the readback and the
/// `progression_waterfall_stays_visually_contiguous` test's `metadata()`
/// assertion (src/sim/world.rs:3400) that checks the source id round-trip.
/// Tracked by hash-thing-457f.
#[inline]
pub fn pack_clone_source(source_material: MaterialId) -> CellState {
    assert!(
        source_material <= Cell::MAX_METADATA,
        "clone source_material {} exceeds 6-bit cap (MAX_METADATA = {}); \
         widen encoding via side-table (hash-thing-457f)",
        source_material,
        Cell::MAX_METADATA,
    );
    Cell::pack(CLONE_MATERIAL_ID, source_material).raw()
}

/// Density lookup for block rules (gravity, fluid). Maps material ID → density.
///
/// Values here must match `MaterialPhysicalProperties::density` in `terrain_defaults()`.
/// If a material is missing, it gets density 0.0 (floats like air).
pub fn material_density(cell: Cell) -> f32 {
    if cell.is_empty() {
        return 0.0;
    }
    match cell.material() {
        STONE_MATERIAL_ID => 5.0,
        DIRT_MATERIAL_ID => 2.0,
        GRASS_MATERIAL_ID => 1.2,
        WATER_MATERIAL_ID => 1.0,
        FIRE_MATERIAL_ID => 0.05,
        SAND_MATERIAL_ID => 1.5,
        LAVA_MATERIAL_ID => 3.0,
        ICE_MATERIAL_ID => 0.9,
        ACID_MATERIAL_ID => 1.1,
        OIL_MATERIAL_ID => 0.8,
        GUNPOWDER_MATERIAL_ID => 1.4,
        STEAM_MATERIAL_ID => -0.1,
        id if FAN_ARMED_STEAM_MATERIAL_IDS.contains(&id) => -0.1,
        GAS_MATERIAL_ID => -0.2,
        METAL_MATERIAL_ID => 7.0,
        VINE_MATERIAL_ID => 1.1,
        FAN_MATERIAL_ID => 1.0,
        FIREWORK_MATERIAL_ID => -0.3,
        id if FAN_ARMED_FIREWORK_MATERIAL_IDS.contains(&id) => -0.3,
        CLONE_MATERIAL_ID => 10.0, // immovable
        _ => 0.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRuleId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialVisualProperties {
    pub label: &'static str,
    pub base_color: [f32; 4],
    pub texture_ref: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialPhysicalProperties {
    pub density: f32,
    pub flammability: f32,
    pub conductivity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialEntry {
    pub visual: MaterialVisualProperties,
    pub physical: MaterialPhysicalProperties,
    pub rule_id: RuleId,
    pub block_rule_id: Option<BlockRuleId>,
    /// How often this material's rules fire, in whole sim ticks.
    /// Rule (CaRule and BlockRule) applies iff `generation % tick_divisor == 0`.
    /// `1` = every tick (default, current behavior). Must be >= 1.
    /// See `sim/hashlife.rs::step_grid_once` for how this gates dispatch and
    /// `MaterialRegistry::memo_period` for how it flows into the memo key.
    pub tick_divisor: u16,
}

pub struct MaterialRegistry {
    entries: Vec<Option<MaterialEntry>>,
    rules: Vec<CaRule>,
    block_rules: Vec<Box<dyn BlockRule + Send>>,
    // Caches rebuilt after any mutation touching entries / block_rules (hash-thing-5yxk).
    // The sim step hot path reads these every tick; recomputing per step would
    // allocate two Vec<u16> per step, which shows up in jw3k-class profiles.
    cached_tick_divisor_flags: Vec<u16>,
    cached_block_rule_tick_divisors: Vec<u16>,
    // hash-thing-2z3g: per-material noop flag, rebuilt under the same
    // invalidation set as the tick caches. Reads from the sim step inner
    // dispatch; without caching, the old `noop_flags()` allocated a
    // Vec<bool> on every call.
    cached_noop_flags: Vec<bool>,
}

impl Clone for MaterialRegistry {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            rules: self.rules.clone(),
            block_rules: self.block_rules.iter().map(|r| r.clone_box()).collect(),
            cached_tick_divisor_flags: self.cached_tick_divisor_flags.clone(),
            cached_block_rule_tick_divisors: self.cached_block_rule_tick_divisors.clone(),
            cached_noop_flags: self.cached_noop_flags.clone(),
        }
    }
}

impl fmt::Debug for MaterialRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaterialRegistry")
            .field("entries", &self.entries)
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

impl MaterialRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(INITIAL_MATERIAL_SLOTS),
            rules: Vec::new(),
            block_rules: Vec::new(),
            cached_tick_divisor_flags: Vec::new(),
            cached_block_rule_tick_divisors: Vec::new(),
            cached_noop_flags: Vec::new(),
        }
    }

    pub fn terrain_defaults() -> Self {
        let mut registry = Self::new();
        let air_rule = registry.register_rule(AirRule {
            fan_material: FAN_MATERIAL_ID,
            carried_materials: vec![
                FanCarriedMaterial::new(STEAM_MATERIAL_ID, FAN_ARMED_STEAM_MATERIAL_IDS),
                FanCarriedMaterial::new(FIREWORK_MATERIAL_ID, FAN_ARMED_FIREWORK_MATERIAL_IDS),
            ],
            vine_growth: Some(AirVineGrowthRule {
                vine_material: VINE_MATERIAL_ID,
                support_materials: vec![
                    STONE_MATERIAL_ID,
                    DIRT_MATERIAL_ID,
                    GRASS_MATERIAL_ID,
                    SAND_MATERIAL_ID,
                    METAL_MATERIAL_ID,
                    ICE_MATERIAL_ID,
                    FAN_MATERIAL_ID,
                    CLONE_MATERIAL_ID,
                ],
                spread_age: 2,
            }),
        });
        let static_rule = registry.register_rule(NoopRule);
        let fan_rule = registry.register_rule(FanRule);
        let fan_static_rule = registry.register_rule(FanDrivenRule::new(NoopRule, FAN_MATERIAL_ID));
        let vine_rule = registry.register_rule(VineRule {
            fire_material: FIRE_MATERIAL_ID,
            acid_material: ACID_MATERIAL_ID,
            max_age: 3,
        });

        // Stone/dirt/grass dissolve when adjacent to acid.
        let dissolvable_rule = registry.register_rule(DissolvableRule {
            acid_material: ACID_MATERIAL_ID,
        });

        let fan_water_rule = registry.register_rule(FanDrivenRule::new(
            WaterRule {
                reactive_material: FIRE_MATERIAL_ID,
                reaction_product: Cell::pack(STONE_MATERIAL_ID, 0),
            },
            FAN_MATERIAL_ID,
        ));
        let lava_rule = registry.register_rule(LavaRule {
            water_material: WATER_MATERIAL_ID,
            solidify_product: Cell::pack(STONE_MATERIAL_ID, 0),
        });
        let ice_rule = registry.register_rule(IceRule {
            heat_materials: vec![FIRE_MATERIAL_ID, LAVA_MATERIAL_ID],
            melt_product: Cell::pack(WATER_MATERIAL_ID, 0),
        });
        let fan_acid_rule = registry.register_rule(FanDrivenRule::new(
            AcidRule {
                dissolvable_materials: vec![
                    STONE_MATERIAL_ID,
                    DIRT_MATERIAL_ID,
                    GRASS_MATERIAL_ID,
                    ICE_MATERIAL_ID,
                    METAL_MATERIAL_ID,
                    VINE_MATERIAL_ID,
                    OIL_MATERIAL_ID,
                ],
            },
            FAN_MATERIAL_ID,
        ));
        let fan_flammable_rule = registry.register_rule(FanDrivenRule::new(
            FlammableRule {
                fire_material: FIRE_MATERIAL_ID,
                fire_product: Cell::pack(FIRE_MATERIAL_ID, 0),
            },
            FAN_MATERIAL_ID,
        ));
        let steam_rule = registry.register_rule(FanDrivenRule::new_with_material_transport(
            SteamRule {
                condense_product: Cell::pack(WATER_MATERIAL_ID, 0),
                max_age: 20,
            },
            FAN_MATERIAL_ID,
            FanCarriedMaterial::new(STEAM_MATERIAL_ID, FAN_ARMED_STEAM_MATERIAL_IDS),
        ));
        let firework_rule = registry.register_rule(FanDrivenRule::new_with_material_transport(
            FireworkRule {
                explode_product: Cell::pack(FIRE_MATERIAL_ID, 0),
                fuse_length: 12,
            },
            FAN_MATERIAL_ID,
            FanCarriedMaterial::new(FIREWORK_MATERIAL_ID, FAN_ARMED_FIREWORK_MATERIAL_IDS),
        ));
        let fan_fire_rule = registry.register_rule(FanDrivenRule::new(
            FireRule {
                fuel_materials: vec![
                    GRASS_MATERIAL_ID,
                    OIL_MATERIAL_ID,
                    GAS_MATERIAL_ID,
                    GUNPOWDER_MATERIAL_ID,
                    VINE_MATERIAL_ID,
                ],
                quencher_material: WATER_MATERIAL_ID,
            },
            FAN_MATERIAL_ID,
        ));

        // Block rules.
        let gravity_block_rule =
            registry.register_block_rule(GravityBlockRule::new(material_density));
        let water_fluid_block_rule =
            registry.register_block_rule(FluidBlockRule::new(material_density, WATER_MATERIAL_ID));
        let lava_fluid_block_rule =
            registry.register_block_rule(FluidBlockRule::new(material_density, LAVA_MATERIAL_ID));
        let acid_fluid_block_rule =
            registry.register_block_rule(FluidBlockRule::new(material_density, ACID_MATERIAL_ID));
        let oil_fluid_block_rule =
            registry.register_block_rule(FluidBlockRule::new(material_density, OIL_MATERIAL_ID));

        registry.insert(
            AIR_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "air",
                    base_color: [0.0, 0.0, 0.0, 0.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 0.0,
                    flammability: 0.0,
                    conductivity: 0.0,
                },
                rule_id: air_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            STONE_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "stone",
                    base_color: [0.45, 0.45, 0.48, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 5.0,
                    flammability: 0.0,
                    conductivity: 0.2,
                },
                rule_id: dissolvable_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            DIRT_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "dirt",
                    base_color: [0.45, 0.29, 0.15, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 2.0,
                    flammability: 0.0,
                    conductivity: 0.08,
                },
                rule_id: dissolvable_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            GRASS_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "grass",
                    base_color: [0.22, 0.57, 0.19, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.2,
                    flammability: 0.35,
                    conductivity: 0.04,
                },
                rule_id: dissolvable_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            FIRE_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "fire",
                    base_color: [0.98, 0.43, 0.05, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 0.05,
                    flammability: 0.0,
                    conductivity: 0.0,
                },
                rule_id: fan_fire_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            WATER_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "water",
                    base_color: [0.12, 0.35, 0.84, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.0,
                    flammability: 0.0,
                    conductivity: 0.6,
                },
                rule_id: fan_water_rule,
                block_rule_id: Some(water_fluid_block_rule),
                // Water falls on every other tick so it reads as liquid.
                // Expands memo_period from 2 to 4 (see rvsh).
                tick_divisor: 2,
            },
        );
        registry.insert(
            SAND_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "sand",
                    base_color: [0.87, 0.80, 0.55, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.5,
                    flammability: 0.0,
                    conductivity: 0.1,
                },
                rule_id: fan_static_rule,
                block_rule_id: Some(gravity_block_rule),
                tick_divisor: 1,
            },
        );
        registry.insert(
            LAVA_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "lava",
                    base_color: [0.95, 0.25, 0.05, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 3.0,
                    flammability: 0.0,
                    conductivity: 0.9,
                },
                rule_id: lava_rule,
                block_rule_id: Some(lava_fluid_block_rule),
                tick_divisor: 1,
            },
        );

        // --- New Powder Game materials ---

        registry.insert(
            ICE_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "ice",
                    base_color: [0.7, 0.88, 0.97, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 0.9,
                    flammability: 0.0,
                    conductivity: 0.5,
                },
                rule_id: ice_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            ACID_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "acid",
                    base_color: [0.4, 0.95, 0.1, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.1,
                    flammability: 0.0,
                    conductivity: 0.4,
                },
                rule_id: fan_acid_rule,
                block_rule_id: Some(acid_fluid_block_rule),
                tick_divisor: 1,
            },
        );
        registry.insert(
            OIL_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "oil",
                    base_color: [0.15, 0.1, 0.05, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 0.8,
                    flammability: 0.9,
                    conductivity: 0.01,
                },
                rule_id: fan_flammable_rule,
                block_rule_id: Some(oil_fluid_block_rule),
                tick_divisor: 1,
            },
        );
        registry.insert(
            GUNPOWDER_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "gunpowder",
                    base_color: [0.3, 0.3, 0.28, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.4,
                    flammability: 1.0,
                    conductivity: 0.05,
                },
                rule_id: fan_flammable_rule,
                block_rule_id: Some(gravity_block_rule),
                tick_divisor: 1,
            },
        );
        registry.insert(
            STEAM_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "steam",
                    base_color: [0.85, 0.85, 0.9, 0.6],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: -0.1,
                    flammability: 0.0,
                    conductivity: 0.3,
                },
                rule_id: steam_rule,
                block_rule_id: Some(gravity_block_rule),
                tick_divisor: 1,
            },
        );
        registry.insert(
            GAS_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "gas",
                    base_color: [0.6, 0.75, 0.3, 0.5],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: -0.2,
                    flammability: 0.95,
                    conductivity: 0.02,
                },
                rule_id: fan_flammable_rule,
                block_rule_id: Some(gravity_block_rule),
                tick_divisor: 1,
            },
        );
        registry.insert(
            METAL_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "metal",
                    base_color: [0.6, 0.62, 0.65, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 7.0,
                    flammability: 0.0,
                    conductivity: 0.95,
                },
                rule_id: dissolvable_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            VINE_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "vine",
                    base_color: [0.13, 0.42, 0.1, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.1,
                    flammability: 0.6,
                    conductivity: 0.03,
                },
                rule_id: vine_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            FAN_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "fan",
                    base_color: [0.5, 0.7, 0.9, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 1.0,
                    flammability: 0.0,
                    conductivity: 0.1,
                },
                rule_id: fan_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );
        registry.insert(
            FIREWORK_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "firework",
                    base_color: [0.9, 0.2, 0.5, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: -0.3,
                    flammability: 0.8,
                    conductivity: 0.0,
                },
                rule_id: firework_rule,
                block_rule_id: Some(gravity_block_rule),
                tick_divisor: 1,
            },
        );
        for (material_id, label) in [
            (FAN_STEAM_POS_X_MATERIAL_ID, "_fan_steam_pos_x"),
            (FAN_STEAM_NEG_X_MATERIAL_ID, "_fan_steam_neg_x"),
            (FAN_STEAM_POS_Z_MATERIAL_ID, "_fan_steam_pos_z"),
            (FAN_STEAM_NEG_Z_MATERIAL_ID, "_fan_steam_neg_z"),
        ] {
            registry.insert(
                material_id,
                MaterialEntry {
                    visual: MaterialVisualProperties {
                        label,
                        base_color: [0.85, 0.85, 0.9, 0.6],
                        texture_ref: None,
                    },
                    physical: MaterialPhysicalProperties {
                        density: -0.1,
                        flammability: 0.0,
                        conductivity: 0.3,
                    },
                    rule_id: steam_rule,
                    block_rule_id: Some(gravity_block_rule),
                    tick_divisor: 1,
                },
            );
        }
        for (material_id, label) in [
            (FAN_FIREWORK_POS_X_MATERIAL_ID, "_fan_firework_pos_x"),
            (FAN_FIREWORK_NEG_X_MATERIAL_ID, "_fan_firework_neg_x"),
            (FAN_FIREWORK_POS_Z_MATERIAL_ID, "_fan_firework_pos_z"),
            (FAN_FIREWORK_NEG_Z_MATERIAL_ID, "_fan_firework_neg_z"),
        ] {
            registry.insert(
                material_id,
                MaterialEntry {
                    visual: MaterialVisualProperties {
                        label,
                        base_color: [0.9, 0.2, 0.5, 1.0],
                        texture_ref: None,
                    },
                    physical: MaterialPhysicalProperties {
                        density: -0.3,
                        flammability: 0.8,
                        conductivity: 0.0,
                    },
                    rule_id: firework_rule,
                    block_rule_id: Some(gravity_block_rule),
                    tick_divisor: 1,
                },
            );
        }
        registry.insert(
            CLONE_MATERIAL_ID,
            MaterialEntry {
                visual: MaterialVisualProperties {
                    label: "clone",
                    base_color: [0.95, 0.85, 0.2, 1.0],
                    texture_ref: None,
                },
                physical: MaterialPhysicalProperties {
                    density: 10.0,
                    flammability: 0.0,
                    conductivity: 0.0,
                },
                rule_id: static_rule,
                block_rule_id: None,
                tick_divisor: 1,
            },
        );

        registry.validate_shared_block_rule_divisors();
        registry
    }

    pub fn gol_smoke() -> Self {
        Self::gol_smoke_with_rule(GameOfLife3D::rule445())
    }

    pub(crate) fn gol_smoke_with_rule(rule: GameOfLife3D) -> Self {
        let mut registry = Self::terrain_defaults();
        let rule_id = registry.register_rule(rule);
        registry.assign_rule(AIR_MATERIAL_ID, rule_id);
        registry.assign_rule(STONE_MATERIAL_ID, rule_id);
        registry.validate_shared_block_rule_divisors();
        registry
    }

    /// True if any registered material has a BlockRule. O(materials), not O(volume).
    pub fn has_any_block_rules(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.as_ref().is_some_and(|e| e.block_rule_id.is_some()))
    }

    pub fn entry(&self, material_id: MaterialId) -> Option<&MaterialEntry> {
        self.entries
            .get(material_id as usize)
            .and_then(Option::as_ref)
    }

    pub fn rule_for_cell(&self, cell: Cell) -> Option<&CaRule> {
        let entry = self.entry(cell.material())?;
        Some(&self.rules[entry.rule_id.0])
    }

    pub fn block_rule_for_cell(&self, cell: Cell) -> Option<&dyn BlockRule> {
        let entry = self.entry(cell.material())?;
        let block_rule_id = entry.block_rule_id?;
        Some(self.block_rules[block_rule_id.0].as_ref())
    }

    pub fn block_rule_id_for_cell(&self, cell: Cell) -> Option<BlockRuleId> {
        self.entry(cell.material())?.block_rule_id
    }

    /// True if this cell has no block rule and its CA rule is noop (identity).
    /// Such cells never change state — hashlife can skip stepping entire
    /// subtrees composed of these materials.
    pub fn cell_is_inert_fixed_point(&self, cell: Cell) -> bool {
        self.block_rule_id_for_cell(cell).is_none()
            && self
                .rule_for_cell(cell)
                .is_some_and(|rule| rule.is_self_inert())
    }

    /// Precomputed per-material-ID noop flag for hot-loop CaRule skipping.
    /// `result[material_id] == true` iff the material's CaRule is noop (identity).
    /// Avoids vtable dispatch per cell in the base-case inner loop.
    ///
    /// Slice into the cache refreshed by any mutation path that could change
    /// per-material rule assignment (`insert`, `assign_rule`) or material slot
    /// presence (`insert`). Rules themselves are append-only after
    /// registration, so their `is_noop` result is fixed once inserted
    /// (hash-thing-2z3g).
    pub fn noop_flags(&self) -> &[bool] {
        &self.cached_noop_flags
    }

    /// Per-material tick_divisor, indexed by material id. Unregistered slots
    /// return 1 (no skip). Used by the step dispatcher to gate per-tick firing.
    ///
    /// Returns a slice into the cache refreshed by any mutation path
    /// (`insert`, `set_tick_divisor`, `register_block_rule`, `assign_block_rule`)
    /// so the sim step hot path avoids the per-tick Vec alloc (hash-thing-5yxk).
    pub fn tick_divisor_flags(&self) -> &[u16] {
        &self.cached_tick_divisor_flags
    }

    /// Per-block-rule tick_divisor, indexed by BlockRuleId. Materials sharing
    /// a BlockRule must share a tick_divisor — this is validated at registry
    /// build time via [`Self::validate_shared_block_rule_divisors`]. If no
    /// material references a rule id (dead rule), returns 1.
    ///
    /// Slice into the cache; see [`Self::tick_divisor_flags`].
    pub fn block_rule_tick_divisors(&self) -> &[u16] {
        &self.cached_block_rule_tick_divisors
    }

    /// Rebuild the per-registry step caches (tick-divisor flags, block-rule
    /// tick divisors, and noop flags) after any mutation path that could
    /// change per-material divisors, the block-rule table, per-material rule
    /// assignment, or the rule table itself. Cost is O(entries + block_rules)
    /// — small constants in practice. Name retained for git-blame continuity
    /// even though it now owns three caches, not two (hash-thing-2z3g).
    ///
    /// Panics if any two materials sharing one BlockRule have different
    /// `tick_divisor`s (iowh invariant). Enforcing here — rather than only in
    /// [`Self::validate_shared_block_rule_divisors`] called at end-of-
    /// construction — catches mismatch at the mutator boundary, so any
    /// future construction path that forgets the trailing validate call still
    /// fails loudly instead of silently picking last-wins (hash-thing-lw75.1.1).
    fn rebuild_tick_caches(&mut self) {
        self.cached_tick_divisor_flags = self
            .entries
            .iter()
            .map(|entry| entry.as_ref().map(|e| e.tick_divisor.max(1)).unwrap_or(1))
            .collect();
        let mut per_rule: Vec<Option<(u16, MaterialId)>> = vec![None; self.block_rules.len()];
        for (material_id, entry) in self.entries.iter().enumerate() {
            let Some(entry) = entry else { continue };
            let Some(BlockRuleId(id)) = entry.block_rule_id else {
                continue;
            };
            let d = entry.tick_divisor.max(1);
            match per_rule[id] {
                None => per_rule[id] = Some((d, material_id as MaterialId)),
                Some((first_d, first_mat)) if first_d != d => {
                    panic!(
                        "shared BlockRule {} has mixed tick_divisors: material {} = {}, \
                         material {} = {}. Materials that share a BlockRule must share a \
                         tick_divisor (iowh invariant).",
                        id, first_mat, first_d, material_id, d
                    );
                }
                Some(_) => {}
            }
        }
        self.cached_block_rule_tick_divisors = per_rule
            .into_iter()
            .map(|slot| slot.map(|(d, _)| d).unwrap_or(1))
            .collect();
        // hash-thing-2z3g: noop_flags share the same invalidation set as
        // the tick caches (insert / assign_block_rule / set_tick_divisor /
        // register_block_rule) plus `assign_rule`, which also calls this.
        //
        // An out-of-bounds `rule_id` is treated as non-noop rather than
        // panicking here — that keeps rebuild_tick_caches compatible with
        // the "insert corrupt entry, catch it later via `validate()`"
        // pattern that `validate_catches_invalid_rule_id` exercises.
        self.cached_noop_flags = self
            .entries
            .iter()
            .map(|entry| {
                entry
                    .as_ref()
                    .and_then(|e| self.rules.get(e.rule_id.0).map(|r| r.is_noop()))
                    .unwrap_or(false)
            })
            .collect();
    }

    /// Panic if any two materials sharing one BlockRule have different
    /// `tick_divisor`s. Called once at registry construction — follow-up beads
    /// that expose tick_divisor via config must call this after mutation.
    ///
    /// Rationale (iowh review): collapsing mixed divisors via GCD silently
    /// nullifies the slower material's divisor ("A says every 4, B says every
    /// 6, rule fires every 2"). That is a semantic lie. Requiring sharers to
    /// agree keeps "per-material tick_divisor" an honest contract.
    pub fn validate_shared_block_rule_divisors(&self) {
        let mut per_rule: Vec<Option<(u16, MaterialId)>> = vec![None; self.block_rules.len()];
        for (material_id, entry) in self.entries.iter().enumerate() {
            let Some(entry) = entry else { continue };
            let Some(BlockRuleId(id)) = entry.block_rule_id else {
                continue;
            };
            let d = entry.tick_divisor.max(1);
            match per_rule[id] {
                None => per_rule[id] = Some((d, material_id as MaterialId)),
                Some((first_d, first_mat)) if first_d != d => {
                    panic!(
                        "shared BlockRule {} has mixed tick_divisors: material {} = {}, \
                         material {} = {}. Materials that share a BlockRule must share a \
                         tick_divisor (iowh invariant).",
                        id, first_mat, first_d, material_id, d
                    );
                }
                Some(_) => {}
            }
        }
    }

    /// The schedule period of the memo key: the smallest `T` such that for any
    /// `generation g`, the behavior of a step at generation `g` is determined
    /// by `g mod T`. Equal to `LCM(2 * tick_divisor_i)` over registered
    /// materials — the factor-of-2 captures Margolus partition alternation.
    ///
    /// With all divisors = 1, returns 2 (identical to today's `parity` key).
    /// If no materials are registered (empty world), returns 2.
    pub fn memo_period(&self) -> u64 {
        let mut period: u64 = 2;
        for entry in self.entries.iter().filter_map(|e| e.as_ref()) {
            let d = entry.tick_divisor.max(1) as u64;
            period = lcm_u64(period, 2 * d);
        }
        period
    }

    pub fn color_palette_rgba(&self) -> Vec<[f32; 4]> {
        let mut palette =
            vec![[0.0, 0.0, 0.0, 0.0]; self.entries.len().max(INITIAL_MATERIAL_SLOTS)];
        for (material_id, entry) in self.entries.iter().enumerate() {
            if let Some(entry) = entry {
                palette[material_id] = entry.visual.base_color;
            }
        }
        palette
    }

    fn register_rule(&mut self, rule: impl Into<CaRule>) -> RuleId {
        let rule_id = RuleId(self.rules.len());
        self.rules.push(rule.into());
        // hash-thing-2z3g: rebuild participates in the noop cache
        // invariant. Without this, a construction path that inserts a
        // material with a placeholder `rule_id` and only *later* registers
        // the rule would leave `cached_noop_flags` stuck at `false` (the
        // permissive out-of-bounds fallback). All shipping constructors
        // register rules before inserting materials, but the cost of the
        // rebuild here is O(entries + block_rules) — cheap — and it makes
        // the invariant local to each mutator.
        self.rebuild_tick_caches();
        rule_id
    }

    /// Register a block rule and return its id.
    ///
    /// On a registry embedded in a [`crate::sim::world::World`], calling this
    /// directly on `world.materials` bypasses [`crate::sim::world::World::mutate_materials`]
    /// and leaves `block_rule_present` stale (hash-thing-6iiz / dxi4.2 audit).
    /// Route via `world.mutate_materials(|m| m.register_block_rule(...))` instead.
    pub fn register_block_rule<R>(&mut self, rule: R) -> BlockRuleId
    where
        R: BlockRule + Send + 'static,
    {
        let id = BlockRuleId(self.block_rules.len());
        self.block_rules.push(Box::new(rule));
        self.rebuild_tick_caches();
        id
    }

    /// Bind `material_id` to the given block rule.
    ///
    /// On a registry embedded in a [`crate::sim::world::World`], calling this
    /// directly on `world.materials` bypasses [`crate::sim::world::World::mutate_materials`]
    /// and leaves `block_rule_present` stale (hash-thing-6iiz / dxi4.2 audit).
    /// Route via `world.mutate_materials(|m| m.assign_block_rule(...))` instead.
    pub fn assign_block_rule(&mut self, material_id: MaterialId, block_rule_id: BlockRuleId) {
        self.entries[material_id as usize]
            .as_mut()
            .unwrap_or_else(|| {
                panic!("material {material_id} must exist before assigning a block rule")
            })
            .block_rule_id = Some(block_rule_id);
        self.rebuild_tick_caches();
    }

    /// Set the tick divisor for an existing material.
    ///
    /// Panics if `divisor == 0`. On a registry embedded in a
    /// [`crate::sim::world::World`], calling this directly on `world.materials`
    /// bypasses [`crate::sim::world::World::mutate_materials`] and leaves the
    /// four hashlife caches + `block_rule_present` stale (hash-thing-6iiz /
    /// dxi4.2 audit). Prefer
    /// [`crate::sim::world::World::set_material_tick_divisor`] or route via
    /// `world.mutate_materials(|m| m.set_tick_divisor(...))`. Crate-private
    /// because the shipping path bakes divisors into registrar constructors;
    /// this exists for tests and the future tuning bead.
    #[allow(dead_code)]
    pub(crate) fn set_tick_divisor(&mut self, material_id: MaterialId, divisor: u16) {
        assert!(
            divisor >= 1,
            "tick_divisor must be >= 1 (got {divisor} for material {material_id})"
        );
        self.entries[material_id as usize]
            .as_mut()
            .unwrap_or_else(|| panic!("material {material_id} must exist to set tick_divisor"))
            .tick_divisor = divisor;
        self.rebuild_tick_caches();
    }

    /// Look up a block rule by ID. Used by `World::step_blocks()`.
    pub fn block_rule(&self, id: BlockRuleId) -> &dyn BlockRule {
        self.block_rules[id.0].as_ref()
    }

    fn insert(&mut self, material_id: MaterialId, entry: MaterialEntry) {
        let material_id = material_id as usize;
        if self.entries.len() <= material_id {
            self.entries.resize(material_id + 1, None);
        }
        self.entries[material_id] = Some(entry);
        self.rebuild_tick_caches();
    }

    fn assign_rule(&mut self, material_id: MaterialId, rule_id: RuleId) {
        self.entries[material_id as usize]
            .as_mut()
            .expect("material must exist before assigning a rule")
            .rule_id = rule_id;
        // hash-thing-2z3g: rule_id drives the noop cache. Today all
        // `assign_rule` callers run before stepping, but the rebuild is
        // cheap (O(entries + block_rules)) and keeps the invariant local.
        self.rebuild_tick_caches();
    }

    /// Validate the registry for internal consistency. Returns a list of
    /// human-readable diagnostic strings; empty means valid.
    ///
    /// Checks:
    /// - Material ID 0 (AIR) must be present.
    /// - No gaps in the material ID space (None slots between populated entries).
    /// - Every entry's `rule_id` refers to a registered rule.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // AIR (material 0) must exist.
        if self.entry(AIR_MATERIAL_ID).is_none() {
            errors.push("material 0 (AIR) is not registered".into());
        }

        // Check for gaps and invalid rule references.
        let mut last_populated: Option<usize> = None;
        for (id, slot) in self.entries.iter().enumerate() {
            match slot {
                Some(entry) => {
                    if entry.rule_id.0 >= self.rules.len() {
                        errors.push(format!(
                            "material {id} ({}) references rule_id {} but only {} rules registered",
                            entry.visual.label,
                            entry.rule_id.0,
                            self.rules.len()
                        ));
                    }
                    last_populated = Some(id);
                }
                None => {
                    if let Some(prev) = last_populated {
                        // Only flag as a gap if there's a populated entry
                        // after this None slot. Check lazily: if we see a
                        // populated entry later, this gap matters.
                        // We'll do a second pass for gaps to keep it simple.
                        let _ = prev;
                    }
                }
            }
        }

        // Gap detection: find None slots that have populated entries on both sides.
        let mut in_gap = false;
        let mut gap_start = 0usize;
        for (id, slot) in self.entries.iter().enumerate() {
            if slot.is_none() && last_populated.is_some_and(|last| id < last) {
                if !in_gap {
                    gap_start = id;
                    in_gap = true;
                }
            } else if slot.is_some() && in_gap {
                errors.push(format!(
                    "gap in material ID space: slots {gap_start}..{id} are empty"
                ));
                in_gap = false;
            }
        }

        errors
    }
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MaterialRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "MaterialRegistry ({} slots, {} rules):",
            self.entries.len(),
            self.rules.len()
        )?;
        for (id, slot) in self.entries.iter().enumerate() {
            match slot {
                Some(entry) => {
                    writeln!(
                        f,
                        "  [{id:>3}] {} (rule {})",
                        entry.visual.label, entry.rule_id.0
                    )?;
                }
                None => {
                    writeln!(f, "  [{id:>3}] <empty>")?;
                }
            }
        }
        Ok(())
    }
}

/// Map "depth below the surface y" to a material.
///
/// `depth` is `surface_y - cell_y`. Above the surface (`depth < 0`) is `AIR`.
/// At the surface (`depth == 0`) is `GRASS`. Within `DEPTH_MARGIN` is `DIRT`.
/// Deeper than that is `STONE`.
#[inline]
pub fn material_from_depth(depth: f32) -> CellState {
    if depth < 0.0 {
        AIR
    } else if depth < 1.0 {
        GRASS
    } else if depth < 4.0 {
        DIRT
    } else {
        STONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_with(rule_id: RuleId, color: [f32; 4]) -> MaterialEntry {
        MaterialEntry {
            visual: MaterialVisualProperties {
                label: "custom",
                base_color: color,
                texture_ref: Some(7),
            },
            physical: MaterialPhysicalProperties {
                density: 9.0,
                flammability: 0.1,
                conductivity: 0.8,
            },
            rule_id,
            block_rule_id: None,
            tick_divisor: 1,
        }
    }

    #[test]
    fn air_is_zero() {
        assert_eq!(AIR, 0, "AIR must be CellState 0 — load-bearing invariant");
    }

    #[test]
    fn materials_are_distinct() {
        let mats = [
            AIR, STONE, DIRT, GRASS, FIRE, WATER, SAND, LAVA, ICE, ACID, OIL, GUNPOWDER, STEAM,
            GAS, METAL, VINE, FAN, FIREWORK, CLONE,
        ];
        for (i, &a) in mats.iter().enumerate() {
            for (j, &b) in mats.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "material constants must be distinct");
                }
            }
        }
    }

    #[test]
    fn material_from_depth_boundary() {
        assert_eq!(material_from_depth(-1.0), AIR);
        assert_eq!(material_from_depth(0.0), GRASS);
        assert_eq!(material_from_depth(0.5), GRASS);
        assert_eq!(material_from_depth(1.0), DIRT);
        assert_eq!(material_from_depth(3.9), DIRT);
        assert_eq!(material_from_depth(4.0), STONE);
        assert_eq!(material_from_depth(100.0), STONE);
    }

    #[test]
    fn water_ships_with_tick_divisor_two_from_terrain_defaults() {
        // Pins the default so a future materials refactor can't accidentally
        // revert it to 1 (which would visibly double water fall speed)
        // without the test flagging. See rvsh.
        let registry = MaterialRegistry::terrain_defaults();
        let water = registry.entry(WATER_MATERIAL_ID).unwrap();
        assert_eq!(
            water.tick_divisor, 2,
            "water must ship with tick_divisor=2 (edward pick on hash-thing-rvsh)"
        );
    }

    #[test]
    fn non_water_materials_stay_at_tick_divisor_one() {
        // rvsh moved water only. Sand, lava, acid, oil, fire and other
        // materials must remain at 1 until a separate bead tunes them —
        // a sloppy "bump everyone" refactor would slow fire/CA dynamics
        // unintentionally. Enumerate every registered material and flag
        // any that drifted off 1 except water.
        let registry = MaterialRegistry::terrain_defaults();
        let flags = registry.tick_divisor_flags();
        for (id, divisor) in flags.iter().enumerate() {
            if id as u16 == WATER_MATERIAL_ID {
                continue;
            }
            let Some(entry) = registry.entry(id as u16) else {
                continue;
            };
            assert_eq!(
                *divisor, 1,
                "material {id} ({}) must remain at tick_divisor=1; \
                 rvsh moved water only",
                entry.visual.label
            );
        }
    }

    #[test]
    fn terrain_defaults_share_dissolvable_rule_for_terrain_solids() {
        let registry = MaterialRegistry::terrain_defaults();
        let stone = registry.entry(STONE_MATERIAL_ID).unwrap();
        let dirt = registry.entry(DIRT_MATERIAL_ID).unwrap();
        let grass = registry.entry(GRASS_MATERIAL_ID).unwrap();
        let fire = registry.entry(FIRE_MATERIAL_ID).unwrap();
        let water = registry.entry(WATER_MATERIAL_ID).unwrap();

        // Stone, dirt, grass share DissolvableRule (react to acid).
        assert_eq!(stone.rule_id, dirt.rule_id);
        assert_eq!(dirt.rule_id, grass.rule_id);
        assert_ne!(grass.rule_id, fire.rule_id);
        assert_ne!(fire.rule_id, water.rule_id);
        assert_eq!(stone.visual.label, "stone");
        assert!(stone.physical.density > dirt.physical.density);
    }

    #[test]
    fn terrain_defaults_register_known_materials_and_leave_unknown_missing() {
        let registry = MaterialRegistry::terrain_defaults();

        assert_eq!(registry.entry(AIR_MATERIAL_ID).unwrap().visual.label, "air");
        assert_eq!(
            registry.entry(WATER_MATERIAL_ID).unwrap().visual.label,
            "water"
        );
        assert_eq!(
            registry.entry(SAND_MATERIAL_ID).unwrap().visual.label,
            "sand"
        );
        assert_eq!(
            registry.entry(LAVA_MATERIAL_ID).unwrap().visual.label,
            "lava"
        );
        assert!(registry.entry(42).is_none());
    }

    #[test]
    fn terrain_defaults_export_gpu_palette() {
        let registry = MaterialRegistry::terrain_defaults();
        let palette = registry.color_palette_rgba();

        assert_eq!(palette.len(), INITIAL_MATERIAL_SLOTS);
        assert_eq!(palette[AIR_MATERIAL_ID as usize], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(palette[STONE_MATERIAL_ID as usize], [0.45, 0.45, 0.48, 1.0]);
        assert_eq!(palette[DIRT_MATERIAL_ID as usize], [0.45, 0.29, 0.15, 1.0]);
        assert_eq!(palette[GRASS_MATERIAL_ID as usize], [0.22, 0.57, 0.19, 1.0]);
        assert_eq!(palette[FIRE_MATERIAL_ID as usize], [0.98, 0.43, 0.05, 1.0]);
        assert_eq!(palette[WATER_MATERIAL_ID as usize], [0.12, 0.35, 0.84, 1.0]);
        assert_eq!(palette[SAND_MATERIAL_ID as usize], [0.87, 0.80, 0.55, 1.0]);
        assert_eq!(palette[LAVA_MATERIAL_ID as usize], [0.95, 0.25, 0.05, 1.0]);
    }

    #[test]
    fn registry_grows_past_256_slots() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(300, entry_with(rule_id, [0.9, 0.2, 0.1, 1.0]));
        let palette = registry.color_palette_rgba();

        assert_eq!(registry.entry(300).unwrap().rule_id, rule_id);
        assert_eq!(palette.len(), 301);
        assert_eq!(palette[299], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(palette[300], [0.9, 0.2, 0.1, 1.0]);
    }

    #[test]
    fn terrain_defaults_dispatch_registered_rules() {
        let registry = MaterialRegistry::terrain_defaults();
        let neighbors = [Cell::pack(FIRE_MATERIAL_ID, 0); 26];

        assert_eq!(
            registry
                .rule_for_cell(Cell::pack(WATER_MATERIAL_ID, 0))
                .unwrap()
                .step_cell(Cell::pack(WATER_MATERIAL_ID, 0), &neighbors),
            Cell::pack(STONE_MATERIAL_ID, 0),
        );
        assert_eq!(
            registry
                .rule_for_cell(Cell::pack(STONE_MATERIAL_ID, 0))
                .unwrap()
                .step_cell(Cell::pack(STONE_MATERIAL_ID, 0), &neighbors),
            Cell::pack(STONE_MATERIAL_ID, 0),
        );
    }

    #[test]
    fn terrain_defaults_wire_air_and_vine_growth_rules() {
        let registry = MaterialRegistry::terrain_defaults();
        let mut growth_neighbors = [Cell::pack(STONE_MATERIAL_ID, 0); 26];
        growth_neighbors[12] = Cell::pack(VINE_MATERIAL_ID, 2);

        assert_eq!(
            registry
                .rule_for_cell(Cell::EMPTY)
                .unwrap()
                .step_cell(Cell::EMPTY, &growth_neighbors),
            Cell::pack(VINE_MATERIAL_ID, 0),
        );
        assert_eq!(
            registry
                .rule_for_cell(Cell::pack(VINE_MATERIAL_ID, 0))
                .unwrap()
                .step_cell(Cell::pack(VINE_MATERIAL_ID, 0), &[Cell::EMPTY; 26]),
            Cell::pack(VINE_MATERIAL_ID, 1),
        );
    }

    #[test]
    fn rule_for_cell_returns_none_for_unregistered_material() {
        let registry = MaterialRegistry::terrain_defaults();

        assert!(registry.rule_for_cell(Cell::pack(42, 0)).is_none());
    }

    #[test]
    fn gol_smoke_overrides_air_and_alive_dispatch() {
        let registry = MaterialRegistry::gol_smoke();
        let neighbors = [Cell::pack(STONE_MATERIAL_ID, 0); 26];

        assert_eq!(
            registry
                .rule_for_cell(Cell::EMPTY)
                .unwrap()
                .step_cell(Cell::EMPTY, &neighbors),
            Cell::EMPTY,
        );
        assert_eq!(
            registry
                .rule_for_cell(Cell::pack(STONE_MATERIAL_ID, 0))
                .unwrap()
                .step_cell(Cell::pack(STONE_MATERIAL_ID, 0), &neighbors),
            Cell::EMPTY,
        );
    }

    #[test]
    fn gol_smoke_preserves_non_overridden_material_dispatch() {
        let registry = MaterialRegistry::gol_smoke();
        let fire_neighbors = [Cell::pack(FIRE_MATERIAL_ID, 0); 26];

        assert_eq!(
            registry
                .rule_for_cell(Cell::pack(WATER_MATERIAL_ID, 0))
                .unwrap()
                .step_cell(Cell::pack(WATER_MATERIAL_ID, 0), &fire_neighbors),
            Cell::pack(STONE_MATERIAL_ID, 0),
        );
        assert_eq!(
            registry
                .rule_for_cell(Cell::pack(GRASS_MATERIAL_ID, 7))
                .unwrap()
                .step_cell(Cell::pack(GRASS_MATERIAL_ID, 7), &fire_neighbors),
            Cell::pack(GRASS_MATERIAL_ID, 7),
        );
    }

    #[test]
    fn terrain_defaults_validates_clean() {
        let registry = MaterialRegistry::terrain_defaults();
        let errors = registry.validate();
        assert!(
            errors.is_empty(),
            "terrain_defaults must validate: {errors:?}"
        );
    }

    #[test]
    fn validate_catches_missing_air() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        // Insert material 1 without material 0 (AIR).
        registry.insert(1, entry_with(rule_id, [1.0, 0.0, 0.0, 1.0]));
        let errors = registry.validate();
        assert!(
            errors.iter().any(|e| e.contains("AIR")),
            "should flag missing AIR: {errors:?}"
        );
    }

    #[test]
    fn validate_catches_gap() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(0, entry_with(rule_id, [0.0; 4]));
        // Skip material 1, insert material 2.
        registry.insert(2, entry_with(rule_id, [1.0, 0.0, 0.0, 1.0]));
        let errors = registry.validate();
        assert!(
            errors.iter().any(|e| e.contains("gap")),
            "should flag gap at slot 1: {errors:?}"
        );
    }

    #[test]
    fn validate_catches_invalid_rule_id() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(
            0,
            MaterialEntry {
                rule_id: RuleId(99), // bogus
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        let errors = registry.validate();
        assert!(
            errors.iter().any(|e| e.contains("rule_id")),
            "should flag invalid rule_id: {errors:?}"
        );
    }

    #[test]
    fn display_lists_materials() {
        let registry = MaterialRegistry::terrain_defaults();
        let s = format!("{registry}");
        assert!(s.contains("stone"), "Display must list stone");
        assert!(s.contains("fire"), "Display must list fire");
        assert!(s.contains("MaterialRegistry"), "Display must have header");
    }

    #[test]
    fn material_density_matches_registry() {
        let registry = MaterialRegistry::terrain_defaults();
        let ids = [
            AIR_MATERIAL_ID,
            STONE_MATERIAL_ID,
            DIRT_MATERIAL_ID,
            GRASS_MATERIAL_ID,
            FIRE_MATERIAL_ID,
            WATER_MATERIAL_ID,
            SAND_MATERIAL_ID,
            LAVA_MATERIAL_ID,
            ICE_MATERIAL_ID,
            ACID_MATERIAL_ID,
            OIL_MATERIAL_ID,
            GUNPOWDER_MATERIAL_ID,
            STEAM_MATERIAL_ID,
            FAN_STEAM_POS_X_MATERIAL_ID,
            FAN_STEAM_NEG_X_MATERIAL_ID,
            FAN_STEAM_POS_Z_MATERIAL_ID,
            FAN_STEAM_NEG_Z_MATERIAL_ID,
            GAS_MATERIAL_ID,
            METAL_MATERIAL_ID,
            VINE_MATERIAL_ID,
            FAN_MATERIAL_ID,
            FIREWORK_MATERIAL_ID,
            FAN_FIREWORK_POS_X_MATERIAL_ID,
            FAN_FIREWORK_NEG_X_MATERIAL_ID,
            FAN_FIREWORK_POS_Z_MATERIAL_ID,
            FAN_FIREWORK_NEG_Z_MATERIAL_ID,
            CLONE_MATERIAL_ID,
        ];
        for &id in &ids {
            let cell = if id == AIR_MATERIAL_ID {
                Cell::EMPTY
            } else {
                Cell::pack(id, 0)
            };
            let fn_density = material_density(cell);
            let reg_density = registry.entry(id).unwrap().physical.density;
            assert!(
                (fn_density - reg_density).abs() < f32::EPSILON,
                "material_density({}) = {fn_density}, registry = {reg_density}",
                registry.entry(id).unwrap().visual.label
            );
        }
    }

    #[test]
    fn terrain_defaults_water_has_block_rule() {
        let registry = MaterialRegistry::terrain_defaults();
        let water = registry.entry(WATER_MATERIAL_ID).unwrap();
        assert!(
            water.block_rule_id.is_some(),
            "water must have a block rule for fluid flow"
        );
    }

    #[test]
    fn terrain_defaults_stone_grass_no_block_rule() {
        let registry = MaterialRegistry::terrain_defaults();
        assert!(
            registry
                .entry(STONE_MATERIAL_ID)
                .unwrap()
                .block_rule_id
                .is_none(),
            "stone should not have a block rule"
        );
        assert!(
            registry
                .entry(GRASS_MATERIAL_ID)
                .unwrap()
                .block_rule_id
                .is_none(),
            "grass should not have a block rule"
        );
    }

    #[test]
    fn memo_period_terrain_defaults_is_four_under_water_divisor_two() {
        // rvsh: water ships at tick_divisor=2, so memo_period is LCM of
        // 2 (every other material) and 2*2=4 (water) = 4. Before rvsh
        // this was 2 (all divisors at 1). Any future default bump should
        // flow through this assertion.
        let registry = MaterialRegistry::terrain_defaults();
        assert_eq!(
            registry.memo_period(),
            4,
            "with water tick_divisor=2 and every other material at 1, memo_period must be LCM(2, 4) = 4"
        );
    }

    #[test]
    fn memo_period_empty_registry_is_two() {
        let registry = MaterialRegistry::new();
        assert_eq!(registry.memo_period(), 2);
    }

    #[test]
    fn memo_period_expands_for_larger_divisors() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(0, entry_with(rule_id, [0.0; 4]));
        // d=1 → 2, d=4 → 8 → LCM(2, 8) = 8
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 4,
                ..entry_with(rule_id, [1.0, 0.0, 0.0, 1.0])
            },
        );
        assert_eq!(registry.memo_period(), 8);

        // Add d=3 → 6 → LCM(8, 6) = 24
        registry.insert(
            2,
            MaterialEntry {
                tick_divisor: 3,
                ..entry_with(rule_id, [0.0, 1.0, 0.0, 1.0])
            },
        );
        assert_eq!(registry.memo_period(), 24);
    }

    #[test]
    fn tick_divisor_flags_returns_per_material_values() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(0, entry_with(rule_id, [0.0; 4]));
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 4,
                ..entry_with(rule_id, [1.0; 4])
            },
        );
        let flags = registry.tick_divisor_flags();
        assert_eq!(flags[0], 1);
        assert_eq!(flags[1], 4);
    }

    #[test]
    fn noop_flags_returns_per_material_values() {
        let mut registry = MaterialRegistry::new();
        let noop_id = registry.register_rule(NoopRule);
        let non_noop_id = registry.register_rule(FireRule {
            fuel_materials: vec![],
            quencher_material: 0,
        });
        registry.insert(0, entry_with(noop_id, [0.0; 4]));
        registry.insert(1, entry_with(non_noop_id, [1.0; 4]));
        let flags = registry.noop_flags();
        assert!(flags[0], "NoopRule must report as noop");
        assert!(!flags[1], "FireRule must not report as noop");
    }

    #[test]
    fn noop_flags_cache_refreshes_on_assign_rule() {
        // hash-thing-2z3g: assign_rule swaps a material's rule_id, which can
        // flip its noop flag. The cache must reflect the post-assign state.
        let mut registry = MaterialRegistry::new();
        let noop_id = registry.register_rule(NoopRule);
        let non_noop_id = registry.register_rule(FireRule {
            fuel_materials: vec![],
            quencher_material: 0,
        });
        registry.insert(0, entry_with(non_noop_id, [0.0; 4]));
        assert!(!registry.noop_flags()[0], "pre-assign: non-noop rule");
        registry.assign_rule(0, noop_id);
        assert!(registry.noop_flags()[0], "post-assign: noop rule");
    }

    #[test]
    fn noop_flags_cache_refreshes_on_insert() {
        // hash-thing-2z3g: insert on a new slot grows the cache so the caller
        // can index by the new material id without a panic or stale false.
        let mut registry = MaterialRegistry::new();
        let noop_id = registry.register_rule(NoopRule);
        assert!(registry.noop_flags().is_empty());
        registry.insert(2, entry_with(noop_id, [0.0; 4]));
        let flags = registry.noop_flags();
        assert_eq!(flags.len(), 3, "slots 0..=2 all present after insert(2)");
        assert!(!flags[0], "unregistered slot 0 defaults to non-noop");
        assert!(!flags[1], "unregistered slot 1 defaults to non-noop");
        assert!(flags[2], "slot 2 got a NoopRule");
    }

    #[test]
    fn noop_flags_cache_refreshes_on_insert_overwrite() {
        // hash-thing-2z3g follow-up: insert() is both a grow mutator and an
        // overwrite mutator. Overwriting an existing slot with a different
        // rule_id must also invalidate the noop cache for that slot.
        let mut registry = MaterialRegistry::new();
        let noop_id = registry.register_rule(NoopRule);
        let non_noop_id = registry.register_rule(FireRule {
            fuel_materials: vec![],
            quencher_material: 0,
        });
        registry.insert(0, entry_with(noop_id, [0.0; 4]));
        assert!(registry.noop_flags()[0], "pre-overwrite: noop rule");
        registry.insert(0, entry_with(non_noop_id, [1.0; 4]));
        assert!(!registry.noop_flags()[0], "post-overwrite: non-noop rule");
    }

    #[test]
    fn noop_flags_cache_refreshes_on_register_rule() {
        // hash-thing-2z3g critical-review follow-up: register_rule must also
        // rebuild the cache. Construct the exact "insert-with-placeholder,
        // register-rule-later" sequence that motivated the invariant — and
        // verify the noop flag flips from the permissive-fallback `false` to
        // `true` once the rule backing the id is actually registered.
        let mut registry = MaterialRegistry::new();
        // Insert first with a rule_id that doesn't exist yet (validate()
        // would flag this at construction-end; pre-validate state is legal).
        registry.insert(
            0,
            MaterialEntry {
                rule_id: RuleId(0),
                ..entry_with(RuleId(0), [0.0; 4])
            },
        );
        assert!(
            !registry.noop_flags()[0],
            "pre-register: permissive out-of-bounds fallback returns false"
        );
        registry.register_rule(NoopRule);
        assert!(
            registry.noop_flags()[0],
            "post-register: rule_id now resolves, cache reflects is_noop"
        );
    }

    #[test]
    fn tick_divisor_flags_treats_zero_as_one() {
        // Defensive: zero divisor should clamp to 1 (a 0-divisor would divide
        // by zero in the dispatcher). The struct field is u16; we verify the
        // `.max(1)` guard in the accessor.
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 0,
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        assert_eq!(registry.tick_divisor_flags()[0], 1);
    }

    #[test]
    fn block_rule_tick_divisors_matches_uniform_sharers() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        let block_rule_id = registry.register_block_rule(GravityBlockRule::new(material_density));
        // Two materials sharing one BlockRule with the SAME divisor.
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [1.0; 4])
            },
        );
        registry.validate_shared_block_rule_divisors();
        let divisors = registry.block_rule_tick_divisors();
        assert_eq!(divisors[block_rule_id.0], 4);
    }

    /// T3 (iowh review): materials sharing a BlockRule must share a
    /// tick_divisor. Rejecting mixed divisors at registry-build time is the
    /// explicit invariant that replaces the old silent-GCD collapse.
    #[test]
    #[should_panic(expected = "mixed tick_divisors")]
    fn validate_rejects_mixed_divisors_on_shared_block_rule() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        let block_rule_id = registry.register_block_rule(GravityBlockRule::new(material_density));
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 6,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [1.0; 4])
            },
        );
        registry.validate_shared_block_rule_divisors();
    }

    /// hash-thing-lw75.1.1: the shared-BlockRule invariant must fire at the
    /// mutator boundary, not only from a trailing `validate_shared_block_rule_divisors`
    /// call. Omits the explicit validate — if `rebuild_tick_caches` doesn't
    /// catch the mismatch, the test fails. Guards against a future
    /// construction path that forgets the trailing validate call.
    #[test]
    #[should_panic(expected = "mixed tick_divisors")]
    fn insert_rejects_mixed_divisors_on_shared_block_rule_without_explicit_validate() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        let block_rule_id = registry.register_block_rule(GravityBlockRule::new(material_density));
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        // Second insert must panic from inside rebuild_tick_caches — no
        // trailing validate_shared_block_rule_divisors() call.
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 6,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [1.0; 4])
            },
        );
    }

    /// hash-thing-lw75.1.1: parallel coverage for the `assign_block_rule`
    /// mutator path. Two materials with mismatched `tick_divisor`s sharing one
    /// `BlockRule` via post-insert assignment must panic from rebuild, without
    /// any explicit validate call.
    #[test]
    #[should_panic(expected = "mixed tick_divisors")]
    fn assign_block_rule_rejects_mixed_divisors_without_explicit_validate() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        let block_rule_id = registry.register_block_rule(GravityBlockRule::new(material_density));
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: None,
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 6,
                block_rule_id: None,
                ..entry_with(rule_id, [1.0; 4])
            },
        );
        registry.assign_block_rule(0, block_rule_id);
        // Second assign must panic from inside rebuild_tick_caches.
        registry.assign_block_rule(1, block_rule_id);
    }

    /// hash-thing-lw75.1.1: parallel coverage for the `set_tick_divisor`
    /// mutator path. Two materials initially share a BlockRule with a
    /// consistent divisor (4); diverging one via `set_tick_divisor` must
    /// panic from rebuild — the explicit validate call that used to live
    /// inside `set_tick_divisor` was removed in this change, so rebuild is
    /// the sole enforcer.
    #[test]
    #[should_panic(expected = "mixed tick_divisors")]
    fn set_tick_divisor_rejects_mixed_divisors_on_shared_block_rule() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        let block_rule_id = registry.register_block_rule(GravityBlockRule::new(material_density));
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        registry.insert(
            1,
            MaterialEntry {
                tick_divisor: 4,
                block_rule_id: Some(block_rule_id),
                ..entry_with(rule_id, [1.0; 4])
            },
        );
        // Diverging one sharer must panic from inside rebuild_tick_caches.
        registry.set_tick_divisor(1, 6);
    }

    /// T4 (iowh review): `set_tick_divisor(0)` panics rather than silently
    /// clamping. Protects the dispatcher's `is_multiple_of` gate and the
    /// author's expectation that `>= 1` means "fires at least every N ticks".
    #[test]
    #[should_panic(expected = "tick_divisor must be >= 1")]
    fn set_tick_divisor_rejects_zero() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(0, entry_with(rule_id, [0.0; 4]));
        registry.set_tick_divisor(0, 0);
    }

    /// T4 (iowh review): `lcm_u64` panics on u64 overflow rather than
    /// silently wrapping. Five distinct large primes near 2^16, each the
    /// tick_divisor of one material, produce a memo_period of
    /// `2 * p1 * p2 * p3 * p4 * p5 ≈ 2.4e24`, which overflows u64.
    #[test]
    #[should_panic(expected = "overflowed u64")]
    fn memo_period_panics_on_overflow() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        // Distinct primes near 2^16 (coprime to each other and to 2).
        let primes: [u16; 5] = [65521, 65519, 65497, 65479, 65449];
        for (i, &p) in primes.iter().enumerate() {
            registry.insert(
                i as MaterialId,
                MaterialEntry {
                    tick_divisor: p,
                    ..entry_with(rule_id, [0.0; 4])
                },
            );
        }
        let _ = registry.memo_period();
    }

    #[test]
    fn block_rule_tick_divisors_dead_rule_is_one() {
        let mut registry = MaterialRegistry::new();
        let _rule_id = registry.register_rule(NoopRule);
        let block_rule_id = registry.register_block_rule(GravityBlockRule::new(material_density));
        // No material references this block rule — treat as divisor=1.
        let divisors = registry.block_rule_tick_divisors();
        assert_eq!(divisors[block_rule_id.0], 1);
    }

    /// hash-thing-5yxk: mutation paths that don't touch `tick_divisor` directly
    /// (`register_block_rule` growing the slot count; `assign_block_rule`
    /// pointing an existing entry at a newly-registered rule) still need to
    /// rebuild the block-rule divisor cache. Tests that the cache reflects
    /// post-insert reconfiguration rather than being frozen at registry
    /// construction time.
    #[test]
    fn block_rule_tick_divisors_cache_tracks_post_insert_reassignment() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        let first_block_rule =
            registry.register_block_rule(GravityBlockRule::new(material_density));
        registry.insert(
            0,
            MaterialEntry {
                tick_divisor: 3,
                block_rule_id: Some(first_block_rule),
                ..entry_with(rule_id, [0.0; 4])
            },
        );
        // Register a second block rule AFTER the insert — cache length must grow.
        let second_block_rule =
            registry.register_block_rule(FluidBlockRule::new(material_density, 1));
        assert_eq!(
            registry.block_rule_tick_divisors().len(),
            2,
            "cache must grow when register_block_rule runs post-insert"
        );
        // Point the existing material at the new rule — its slot must now
        // carry the material's divisor, and the old slot must revert to 1.
        registry.assign_block_rule(0, second_block_rule);
        let divisors = registry.block_rule_tick_divisors();
        assert_eq!(divisors[second_block_rule.0], 3, "assign must update cache");
        assert_eq!(
            divisors[first_block_rule.0], 1,
            "old slot reverts to 1 when no entry references it"
        );
    }

    #[test]
    fn pack_clone_source_accepts_at_cap() {
        // Boundary: MAX_METADATA itself must pack without panic.
        let state = pack_clone_source(Cell::MAX_METADATA);
        let cell = Cell::from_raw(state);
        assert_eq!(cell.material(), CLONE_MATERIAL_ID);
        assert_eq!(cell.metadata(), Cell::MAX_METADATA);
    }

    #[test]
    #[should_panic(expected = "exceeds 6-bit cap")]
    fn pack_clone_source_rejects_material_over_cap() {
        // Guards the call-site-local message (not the encoding cap itself,
        // which `Cell::pack` still enforces one layer deeper). Widening the
        // encoding per hash-thing-457f would update both messages together.
        let _ = pack_clone_source(Cell::MAX_METADATA + 1);
    }
}
