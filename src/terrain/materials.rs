//! Material constants and registry for terrain + CA dispatch.
//!
//! `AIR == 0` remains load-bearing for the octree, renderer, and hashing
//! layers. The registry mirrors that encoding: material ID 0 is the empty
//! cell, and every other material ID maps to a validated `CellState`.

use std::fmt;

use crate::octree::{Cell, CellState};
use crate::sim::margolus::FluidBlockRule;
use crate::sim::margolus::GravityBlockRule;
use crate::sim::rule::{BlockRule, CaRule, FireRule, GameOfLife3D, LavaRule, NoopRule, WaterRule};

pub type MaterialId = u16;

pub const INITIAL_MATERIAL_SLOTS: usize = 256;

pub const AIR_MATERIAL_ID: MaterialId = 0;
pub const STONE_MATERIAL_ID: MaterialId = 1;
pub const DIRT_MATERIAL_ID: MaterialId = 2;
pub const GRASS_MATERIAL_ID: MaterialId = 3;
pub const FIRE_MATERIAL_ID: MaterialId = 4;
pub const WATER_MATERIAL_ID: MaterialId = 5;
pub const SAND_MATERIAL_ID: MaterialId = 6;
pub const LAVA_MATERIAL_ID: MaterialId = 7;

pub const AIR: CellState = Cell::EMPTY.raw();
pub const STONE: CellState = Cell::pack(STONE_MATERIAL_ID, 0).raw();
pub const DIRT: CellState = Cell::pack(DIRT_MATERIAL_ID, 0).raw();
pub const GRASS: CellState = Cell::pack(GRASS_MATERIAL_ID, 0).raw();
pub const FIRE: CellState = Cell::pack(FIRE_MATERIAL_ID, 0).raw();
pub const WATER: CellState = Cell::pack(WATER_MATERIAL_ID, 0).raw();
pub const SAND: CellState = Cell::pack(SAND_MATERIAL_ID, 0).raw();
pub const LAVA: CellState = Cell::pack(LAVA_MATERIAL_ID, 0).raw();

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
}

pub struct MaterialRegistry {
    entries: Vec<Option<MaterialEntry>>,
    rules: Vec<Box<dyn CaRule + Send>>,
    block_rules: Vec<Box<dyn BlockRule + Send>>,
}

impl Clone for MaterialRegistry {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            rules: self.rules.iter().map(|r| r.clone_box()).collect(),
            block_rules: self.block_rules.iter().map(|r| r.clone_box()).collect(),
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
        }
    }

    pub fn terrain_defaults() -> Self {
        let mut registry = Self::new();
        let static_rule = registry.register_rule(NoopRule);
        let fire_rule = registry.register_rule(FireRule {
            fuel_material: GRASS_MATERIAL_ID,
            quencher_material: WATER_MATERIAL_ID,
        });
        let water_rule = registry.register_rule(WaterRule {
            reactive_material: FIRE_MATERIAL_ID,
            reaction_product: Cell::pack(STONE_MATERIAL_ID, 0),
        });
        let lava_rule = registry.register_rule(LavaRule {
            water_material: WATER_MATERIAL_ID,
            solidify_product: Cell::pack(STONE_MATERIAL_ID, 0),
        });
        let gravity_block_rule =
            registry.register_block_rule(GravityBlockRule::new(material_density));
        let lava_fluid_block_rule =
            registry.register_block_rule(FluidBlockRule::new(material_density, LAVA_MATERIAL_ID));

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
                rule_id: static_rule,
                block_rule_id: None,
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
                rule_id: static_rule,
                block_rule_id: None,
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
                rule_id: static_rule,
                block_rule_id: None,
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
                rule_id: static_rule,
                block_rule_id: None,
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
                rule_id: fire_rule,
                block_rule_id: None,
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
                rule_id: water_rule,
                block_rule_id: Some(gravity_block_rule),
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
                rule_id: static_rule,
                block_rule_id: Some(gravity_block_rule),
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
            },
        );

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

    pub fn rule_for_cell(&self, cell: Cell) -> Option<&dyn CaRule> {
        let entry = self.entry(cell.material())?;
        Some(self.rules[entry.rule_id.0].as_ref())
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
            && self.rule_for_cell(cell).is_some_and(|rule| rule.is_noop())
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

    fn register_rule<R>(&mut self, rule: R) -> RuleId
    where
        R: CaRule + Send + 'static,
    {
        let rule_id = RuleId(self.rules.len());
        self.rules.push(Box::new(rule));
        rule_id
    }

    pub fn register_block_rule<R>(&mut self, rule: R) -> BlockRuleId
    where
        R: BlockRule + Send + 'static,
    {
        let id = BlockRuleId(self.block_rules.len());
        self.block_rules.push(Box::new(rule));
        id
    }

    pub fn assign_block_rule(&mut self, material_id: MaterialId, block_rule_id: BlockRuleId) {
        self.entries[material_id as usize]
            .as_mut()
            .unwrap_or_else(|| {
                panic!("material {material_id} must exist before assigning a block rule")
            })
            .block_rule_id = Some(block_rule_id);
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
    }

    fn assign_rule(&mut self, material_id: MaterialId, rule_id: RuleId) {
        self.entries[material_id as usize]
            .as_mut()
            .expect("material must exist before assigning a rule")
            .rule_id = rule_id;
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
        }
    }

    #[test]
    fn air_is_zero() {
        assert_eq!(AIR, 0, "AIR must be CellState 0 — load-bearing invariant");
    }

    #[test]
    fn materials_are_distinct() {
        let mats = [AIR, STONE, DIRT, GRASS, FIRE, WATER];
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
    fn terrain_defaults_share_static_rule_for_terrain_solids() {
        let registry = MaterialRegistry::terrain_defaults();
        let air = registry.entry(AIR_MATERIAL_ID).unwrap();
        let stone = registry.entry(STONE_MATERIAL_ID).unwrap();
        let dirt = registry.entry(DIRT_MATERIAL_ID).unwrap();
        let grass = registry.entry(GRASS_MATERIAL_ID).unwrap();
        let fire = registry.entry(FIRE_MATERIAL_ID).unwrap();
        let water = registry.entry(WATER_MATERIAL_ID).unwrap();

        assert_eq!(air.rule_id, stone.rule_id);
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
}
