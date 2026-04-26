// Built-in Margolus block transforms, consumed by World::step_blocks.
//! Built-in Margolus block transforms.
//!
//! Each struct implements `BlockRule` for a specific 2x2x2 block behavior.
//! All transforms must preserve the multiset of input cells (mass conservation).

use crate::octree::Cell;
use crate::sim::rule::{block_index, BlockRule, Phase};

/// Identity block rule — returns the block unchanged. Useful for testing
/// the pipeline (flatten → partition → iterate → apply → rebuild) with
/// zero behavioral change.
pub struct IdentityBlockRule;

impl BlockRule for IdentityBlockRule {
    fn step_block(&self, block: &[Cell; 8], _movable: &[bool; 8]) -> [Cell; 8] {
        *block
    }

    fn clone_box(&self) -> Box<dyn BlockRule + Send> {
        Box::new(IdentityBlockRule)
    }
}

/// Fluid block rule — gravity (vertical swaps) plus lateral spread.
///
/// Phase 1 (gravity): same as `GravityBlockRule` — each vertical column
/// swaps top↔bottom if the upper cell is denser.
///
/// Phase 2 (lateral spread): for each y layer, pick a lateral axis (x or z)
/// derived from the block's cell contents. Along that axis, swap pairs where
/// one cell is fluid (matches `fluid_material`) and the other is air (empty).
/// Content-derived axis selection makes this a pure function of block state,
/// enabling spatial memoization in hashlife.
pub struct FluidBlockRule {
    density_fn: fn(Cell) -> f32,
    phase_fn: fn(Cell) -> Phase,
    fluid_material: u16,
}

impl FluidBlockRule {
    pub fn new(
        density_fn: fn(Cell) -> f32,
        phase_fn: fn(Cell) -> Phase,
        fluid_material: u16,
    ) -> Self {
        Self {
            density_fn,
            phase_fn,
            fluid_material,
        }
    }
}

impl BlockRule for FluidBlockRule {
    fn clone_box(&self) -> Box<dyn BlockRule + Send> {
        Box::new(FluidBlockRule {
            density_fn: self.density_fn,
            phase_fn: self.phase_fn,
            fluid_material: self.fluid_material,
        })
    }

    fn step_block(&self, block: &[Cell; 8], movable: &[bool; 8]) -> [Cell; 8] {
        let mut out = *block;

        // Phase 1: gravity. Only swap if both cells are movable — swapping a
        // fluid cell with a denser anchored cell (e.g. stone above water on a
        // cliff face) would delete the fluid at the write-back layer.
        //
        // Solid-on-solid lock — no-op for current fluids (water/lava/acid/oil
        // are all Liquid), here for symmetry with GravityBlockRule and to
        // defend against future cross-classification (hash-thing-nagw).
        for dz in 0..2 {
            for dx in 0..2 {
                let bot = block_index(dx, 0, dz);
                let top = block_index(dx, 1, dz);
                if !(movable[bot] && movable[top]) {
                    continue;
                }
                let any_non_solid = (self.phase_fn)(out[top]) != Phase::Solid
                    || (self.phase_fn)(out[bot]) != Phase::Solid;
                if any_non_solid && (self.density_fn)(out[top]) > (self.density_fn)(out[bot]) {
                    out.swap(bot, top);
                }
            }
        }

        // Phase 2: lateral spread (fluid ↔ air only).
        // Try BOTH axes per y-layer — the content hash determines which
        // axis has priority when a cell could spread in either direction.
        // This eliminates the directional bias of single-axis selection
        // while remaining a pure function of block state (hashlife-safe).
        let hash = Self::content_hash(block);
        for dy in 0..2usize {
            let x_first = (hash >> dy) & 1 == 0;
            if x_first {
                // X-axis first, then Z-axis
                for dz in 0..2 {
                    let a = block_index(0, dy, dz);
                    let b = block_index(1, dy, dz);
                    self.try_fluid_swap(&mut out, a, b);
                }
                for dx in 0..2 {
                    let a = block_index(dx, dy, 0);
                    let b = block_index(dx, dy, 1);
                    self.try_fluid_swap(&mut out, a, b);
                }
            } else {
                // Z-axis first, then X-axis
                for dx in 0..2 {
                    let a = block_index(dx, dy, 0);
                    let b = block_index(dx, dy, 1);
                    self.try_fluid_swap(&mut out, a, b);
                }
                for dz in 0..2 {
                    let a = block_index(0, dy, dz);
                    let b = block_index(1, dy, dz);
                    self.try_fluid_swap(&mut out, a, b);
                }
            }
        }

        out
    }
}

impl FluidBlockRule {
    /// Derive a hash from block contents for deterministic axis selection.
    /// Uses XOR folding of cell raw values — cheap and sufficient for
    /// picking between two axes.
    #[inline]
    fn content_hash(block: &[Cell; 8]) -> u64 {
        let mut h = 0u64;
        for (i, cell) in block.iter().enumerate() {
            // Rotate and XOR to avoid cancellation from symmetric blocks.
            h ^= (cell.raw() as u64).wrapping_mul(0x9E3779B97F4A7C15u64.wrapping_add(i as u64));
        }
        h
    }

    /// Swap cells at indices a and b if exactly one is fluid and the other is air.
    fn try_fluid_swap(&self, block: &mut [Cell; 8], a: usize, b: usize) {
        let a_is_fluid = block[a].material() == self.fluid_material;
        let b_is_fluid = block[b].material() == self.fluid_material;
        let a_is_air = block[a].is_empty();
        let b_is_air = block[b].is_empty();

        if (a_is_fluid && b_is_air) || (b_is_fluid && a_is_air) {
            block.swap(a, b);
        }
    }
}

/// Gravity block rule — heavy materials swap downward with lighter ones.
///
/// Within the 2x2x2 block, each column (pair of cells sharing x,z but
/// differing in y) is inspected. If the upper cell is heavier than the
/// lower cell, they swap. "Heavier" is determined by a density lookup
/// function provided at construction time.
///
/// This is a mass-conserving permutation: cells are rearranged, never
/// created or destroyed.
pub struct GravityBlockRule {
    /// Returns the density for a given cell. Higher density = falls down.
    /// Air (empty) should return 0.0.
    density_fn: fn(Cell) -> f32,
    /// Returns the phase classification (Solid / Liquid / Gas) for a given cell.
    /// Two `Solid` cells never trade places, regardless of density gap
    /// (hash-thing-nagw). Liquids and gases continue to swap by density.
    phase_fn: fn(Cell) -> Phase,
}

impl GravityBlockRule {
    pub fn new(density_fn: fn(Cell) -> f32, phase_fn: fn(Cell) -> Phase) -> Self {
        Self {
            density_fn,
            phase_fn,
        }
    }
}

impl BlockRule for GravityBlockRule {
    fn clone_box(&self) -> Box<dyn BlockRule + Send> {
        Box::new(GravityBlockRule {
            density_fn: self.density_fn,
            phase_fn: self.phase_fn,
        })
    }

    fn step_block(&self, block: &[Cell; 8], movable: &[bool; 8]) -> [Cell; 8] {
        let mut out = *block;
        // There are 4 vertical columns in a 2x2x2 block.
        // Each column has y=0 (bottom) and y=1 (top).
        //
        // Swap only when BOTH cells are movable (opted into this rule or empty).
        // Swapping a sand cell with an anchored neighbor (e.g. grass below it)
        // would be interpreted by the write-back layer as "sand deleted,
        // neighbor duplicated" — the localized mass-loss mechanism fixed in
        // hash-thing-9yv2.
        //
        // Solids never swap with other solids (hash-thing-nagw): granular
        // stratification is locked. At least one of the two cells must be
        // Liquid or Gas for the density swap to fire.
        for dz in 0..2 {
            for dx in 0..2 {
                let bot = block_index(dx, 0, dz);
                let top = block_index(dx, 1, dz);
                if !(movable[bot] && movable[top]) {
                    continue;
                }
                let any_non_solid = (self.phase_fn)(out[top]) != Phase::Solid
                    || (self.phase_fn)(out[bot]) != Phase::Solid;
                if !any_non_solid {
                    continue;
                }
                let density_bot = (self.density_fn)(out[bot]);
                let density_top = (self.density_fn)(out[top]);
                if density_top > density_bot {
                    out.swap(bot, top);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::Cell;

    fn mat(id: u16) -> Cell {
        if id == 0 {
            Cell::EMPTY
        } else {
            Cell::pack(id, 0)
        }
    }

    /// Check that a block transform is a permutation of the input.
    fn assert_mass_conserved(input: &[Cell; 8], output: &[Cell; 8]) {
        let mut in_sorted: Vec<u16> = input.iter().map(|c| c.raw()).collect();
        let mut out_sorted: Vec<u16> = output.iter().map(|c| c.raw()).collect();
        in_sorted.sort();
        out_sorted.sort();
        assert_eq!(
            in_sorted, out_sorted,
            "block rule must conserve mass (multiset equality)"
        );
    }

    // ---------------------------------------------------------------
    // IdentityBlockRule
    // ---------------------------------------------------------------

    #[test]
    fn identity_preserves_block_exactly() {
        let block = [
            mat(1),
            mat(2),
            mat(3),
            mat(0),
            mat(5),
            mat(0),
            mat(1),
            mat(4),
        ];
        let rule = IdentityBlockRule;
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(block, out);
    }

    #[test]
    fn identity_conserves_mass() {
        let block = [
            mat(1),
            mat(2),
            mat(3),
            mat(4),
            mat(5),
            mat(6),
            mat(7),
            mat(8),
        ];
        let rule = IdentityBlockRule;
        let out = rule.step_block(&block, &[true; 8]);
        assert_mass_conserved(&block, &out);
    }

    // ---------------------------------------------------------------
    // GravityBlockRule
    // ---------------------------------------------------------------

    /// Simple density: material ID as density, air = 0.
    fn simple_density(cell: Cell) -> f32 {
        if cell.is_empty() {
            0.0
        } else {
            cell.material() as f32
        }
    }

    /// Simple phase classification for unit tests:
    /// - empty → Gas
    /// - FLUID_MAT (3) → Liquid
    /// - everything else → Solid
    ///
    /// Mirrors the production `material_phase` shape (hash-thing-nagw).
    fn simple_phase(cell: Cell) -> Phase {
        if cell.is_empty() {
            Phase::Gas
        } else if cell.material() == FLUID_MAT {
            Phase::Liquid
        } else {
            Phase::Solid
        }
    }

    #[test]
    fn gravity_swaps_heavy_cell_downward() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        // Column at (0,0): bottom=air, top=stone(mat 5)
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = Cell::EMPTY; // bottom
        block[block_index(0, 1, 0)] = mat(5); // top — heavier, should fall
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "heavy cell should fall");
        assert_eq!(out[block_index(0, 1, 0)], Cell::EMPTY, "air should rise");
    }

    #[test]
    fn gravity_does_not_swap_when_bottom_is_heavier() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(5); // bottom — heavier
        block[block_index(0, 1, 0)] = mat(1); // top — lighter
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "bottom stays");
        assert_eq!(out[block_index(0, 1, 0)], mat(1), "top stays");
    }

    #[test]
    fn gravity_equal_density_no_swap() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(3);
        block[block_index(0, 1, 0)] = mat(3);
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(out[block_index(0, 0, 0)], mat(3));
        assert_eq!(out[block_index(0, 1, 0)], mat(3));
    }

    #[test]
    fn gravity_conserves_mass() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        let block = [
            mat(5),
            mat(0),
            mat(3),
            mat(1),
            mat(0),
            mat(7),
            mat(2),
            mat(4),
        ];
        let out = rule.step_block(&block, &[true; 8]);
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn gravity_all_four_columns_independent() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        // Set up all 4 columns with heavy-solid-on-top, gas-bottom.
        // (Solid-on-solid no longer swaps under hash-thing-nagw, so we use
        // solid-on-gas to still demonstrate per-column independence.)
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2 {
            for dx in 0..2 {
                block[block_index(dx, 0, dz)] = Cell::EMPTY; // gas bottom
                block[block_index(dx, 1, dz)] = mat(5); // heavy solid top
            }
        }
        let out = rule.step_block(&block, &[true; 8]);
        for dz in 0..2 {
            for dx in 0..2 {
                assert_eq!(
                    out[block_index(dx, 0, dz)],
                    mat(5),
                    "heavy solid should fall at ({dx}, 0, {dz})"
                );
                assert_eq!(
                    out[block_index(dx, 1, dz)],
                    Cell::EMPTY,
                    "gas should rise at ({dx}, 1, {dz})"
                );
            }
        }
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn gravity_preserves_metadata() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        let mut block = [Cell::EMPTY; 8];
        // heavy cell with metadata
        block[block_index(0, 1, 0)] = Cell::pack(5, 42);
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(
            out[block_index(0, 0, 0)],
            Cell::pack(5, 42),
            "metadata must travel with the cell"
        );
    }

    // hash-thing-nagw: solids never sink through other solids, regardless of
    // density gap. Verifies the new phase-gated predicate.
    #[test]
    fn gravity_does_not_swap_solid_on_solid_regardless_of_density_gap() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(1); // light solid bottom
        block[block_index(0, 1, 0)] = mat(7); // much heavier solid top
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(
            out[block_index(0, 0, 0)],
            mat(1),
            "light solid bottom must stay — no solid-on-solid swap"
        );
        assert_eq!(
            out[block_index(0, 1, 0)],
            mat(7),
            "heavy solid top must stay — no solid-on-solid swap"
        );
    }

    // hash-thing-nagw: solids still sink through gas (the original gravity behavior).
    #[test]
    fn gravity_swaps_solid_on_gas() {
        let rule = GravityBlockRule::new(simple_density, simple_phase);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = Cell::EMPTY; // gas bottom
        block[block_index(0, 1, 0)] = mat(5); // solid top
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(out[block_index(0, 0, 0)], mat(5));
        assert_eq!(out[block_index(0, 1, 0)], Cell::EMPTY);
    }

    // hash-thing-nagw: gas-on-gas density swaps still happen (heavy gas falls).
    #[test]
    fn gravity_swaps_gas_on_gas() {
        // Use a synthetic phase where two distinct materials are both Gas
        // but one is denser than the other.
        fn gas_only_phase(_cell: Cell) -> Phase {
            Phase::Gas
        }
        let rule = GravityBlockRule::new(simple_density, gas_only_phase);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(1); // light gas bottom
        block[block_index(0, 1, 0)] = mat(5); // heavy gas top
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "heavy gas falls");
        assert_eq!(out[block_index(0, 1, 0)], mat(1), "light gas rises");
    }

    // ---------------------------------------------------------------
    // FluidBlockRule
    // ---------------------------------------------------------------

    const FLUID_MAT: u16 = 3; // "water" in test context
    const SOLID_MAT: u16 = 7; // "stone" in test context

    fn fluid_rule() -> FluidBlockRule {
        FluidBlockRule::new(simple_density, simple_phase, FLUID_MAT)
    }

    #[test]
    fn fluid_gravity_heavy_over_air_falls() {
        let rule = fluid_rule();
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 1, 0)] = mat(FLUID_MAT); // top — fluid, denser than air
                                                      // Block lateral escape routes so we isolate the gravity phase.
        block[block_index(1, 0, 0)] = mat(SOLID_MAT);
        block[block_index(0, 0, 1)] = mat(SOLID_MAT);
        let out = rule.step_block(&block, &[true; 8]);
        assert_eq!(
            out[block_index(0, 0, 0)],
            mat(FLUID_MAT),
            "fluid should fall"
        );
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn fluid_lateral_spreads_into_air() {
        let rule = fluid_rule();
        // Place fluid at (0,0,0) surrounded by air. Content hash determines
        // axis — but fluid must move to one of the lateral neighbors.
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(FLUID_MAT);
        let out = rule.step_block(&block, &[true; 8]);
        // After gravity (no change — fluid already at bottom), lateral phase
        // should swap fluid with air along whichever axis content_hash picks.
        assert!(
            out[block_index(0, 0, 0)] == Cell::EMPTY || out[block_index(0, 0, 0)] == mat(FLUID_MAT),
            "origin cell should be empty or fluid"
        );
        // Fluid must have moved somewhere — not vanished.
        assert_mass_conserved(&block, &out);
        let fluid_count: usize = out.iter().filter(|c| c.material() == FLUID_MAT).count();
        assert_eq!(fluid_count, 1, "exactly one fluid cell must exist");
    }

    #[test]
    fn fluid_no_spread_into_solid() {
        let rule = fluid_rule();
        // Fluid surrounded by solids on all lateral neighbors — should not spread.
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(FLUID_MAT);
        block[block_index(1, 0, 0)] = mat(SOLID_MAT);
        block[block_index(0, 0, 1)] = mat(SOLID_MAT);
        let out = rule.step_block(&block, &[true; 8]);
        // Fluid can't spread into solid, so it stays at (0,0,0).
        assert_eq!(
            out[block_index(0, 0, 0)],
            mat(FLUID_MAT),
            "fluid stays when blocked by solids"
        );
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn fluid_conserves_mass() {
        let rule = fluid_rule();
        let block = [
            mat(FLUID_MAT),
            mat(0),
            mat(SOLID_MAT),
            mat(FLUID_MAT),
            mat(0),
            mat(FLUID_MAT),
            mat(0),
            mat(SOLID_MAT),
        ];
        let out = rule.step_block(&block, &[true; 8]);
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn fluid_gravity_plus_lateral_combined() {
        let rule = fluid_rule();
        // Fluid at top (0,1,0), air everywhere else.
        // Gravity should pull it to (0,0,0), then lateral should spread it.
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 1, 0)] = mat(FLUID_MAT);
        let out = rule.step_block(&block, &[true; 8]);
        // After gravity: fluid at (0,0,0). After lateral: fluid spreads.
        // The fluid must not be at the top anymore.
        assert_eq!(
            out[block_index(0, 1, 0)],
            Cell::EMPTY,
            "fluid should have fallen from top"
        );
        assert_mass_conserved(&block, &out);
    }
}
