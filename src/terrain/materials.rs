//! Material constants and registry for terrain + CA dispatch.
//!
//! `AIR == 0` remains load-bearing for the octree, renderer, and hashing
//! layers. The registry mirrors that encoding: material ID 0 is the empty
//! cell, and every other material ID maps to a validated `CellState`.

use crate::octree::{Cell, CellState};
use crate::sim::rule::{CaRule, FireRule, GameOfLife3D, NoopRule, WaterRule};

pub type MaterialId = u16;

pub const INITIAL_MATERIAL_SLOTS: usize = 256;

pub const AIR_MATERIAL_ID: MaterialId = 0;
pub const STONE_MATERIAL_ID: MaterialId = 1;
pub const DIRT_MATERIAL_ID: MaterialId = 2;
pub const GRASS_MATERIAL_ID: MaterialId = 3;
pub const FIRE_MATERIAL_ID: MaterialId = 4;
pub const WATER_MATERIAL_ID: MaterialId = 5;

pub const AIR: CellState = Cell::EMPTY.raw();
pub const STONE: CellState = Cell::pack(STONE_MATERIAL_ID, 0).raw();
pub const DIRT: CellState = Cell::pack(DIRT_MATERIAL_ID, 0).raw();
pub const GRASS: CellState = Cell::pack(GRASS_MATERIAL_ID, 0).raw();
pub const FIRE: CellState = Cell::pack(FIRE_MATERIAL_ID, 0).raw();
pub const WATER: CellState = Cell::pack(WATER_MATERIAL_ID, 0).raw();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleId(pub usize);

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialVisualProperties {
    pub label: &'static str,
    pub base_color: [f32; 4],
    pub texture_ref: Option<u32>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialPhysicalProperties {
    pub density: f32,
    pub flammability: f32,
    pub conductivity: f32,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialEntry {
    pub visual: MaterialVisualProperties,
    pub physical: MaterialPhysicalProperties,
    pub rule_id: RuleId,
}

pub struct MaterialRegistry {
    entries: Vec<Option<MaterialEntry>>,
    rules: Vec<Box<dyn CaRule>>,
}

impl MaterialRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(INITIAL_MATERIAL_SLOTS),
            rules: Vec::new(),
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
            },
        );

        registry
    }

    pub fn gol_smoke(rule: GameOfLife3D) -> Self {
        let mut registry = Self::terrain_defaults();
        let rule_id = registry.register_rule(rule);
        registry.assign_rule(AIR_MATERIAL_ID, rule_id);
        registry.assign_rule(STONE_MATERIAL_ID, rule_id);
        registry
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
        R: CaRule + 'static,
    {
        let rule_id = RuleId(self.rules.len());
        self.rules.push(Box::new(rule));
        rule_id
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
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
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
    }

    #[test]
    fn registry_grows_past_256_slots() {
        let mut registry = MaterialRegistry::new();
        let rule_id = registry.register_rule(NoopRule);
        registry.insert(300, entry_with(rule_id, [0.9, 0.2, 0.1, 1.0]));

        assert_eq!(registry.entry(300).unwrap().rule_id, rule_id);
        assert_eq!(registry.color_palette_rgba().len(), 301);
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
    fn gol_smoke_overrides_air_and_alive_dispatch() {
        let registry = MaterialRegistry::gol_smoke(GameOfLife3D::rule445());
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
}
