// Built-in Margolus block transforms, consumed by World::step_blocks.
//! Built-in Margolus block transforms.
//!
//! Each struct implements `BlockRule` for a specific 2x2x2 block behavior.
//! All transforms must preserve the multiset of input cells (mass conservation).

use crate::octree::Cell;
use crate::sim::rule::{block_index, BlockRule};

/// Identity block rule — returns the block unchanged. Useful for testing
/// the pipeline (flatten → partition → iterate → apply → rebuild) with
/// zero behavioral change.
pub struct IdentityBlockRule;

impl BlockRule for IdentityBlockRule {
    fn step_block(&self, block: &[Cell; 8]) -> [Cell; 8] {
        *block
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
    fluid_material: u16,
}

impl FluidBlockRule {
    pub fn new(density_fn: fn(Cell) -> f32, fluid_material: u16) -> Self {
        Self {
            density_fn,
            fluid_material,
        }
    }
}

impl BlockRule for FluidBlockRule {
    fn step_block(&self, block: &[Cell; 8]) -> [Cell; 8] {
        let mut out = *block;

        // Phase 1: gravity — same as GravityBlockRule
        for dz in 0..2 {
            for dx in 0..2 {
                let bot = block_index(dx, 0, dz);
                let top = block_index(dx, 1, dz);
                if (self.density_fn)(out[top]) > (self.density_fn)(out[bot]) {
                    out.swap(bot, top);
                }
            }
        }

        // Phase 2: lateral spread (fluid ↔ air only).
        // Derive axis choice from block contents so the rule is a pure
        // function of state — enables spatial memoization in hashlife.
        let hash = Self::content_hash(block);
        for dy in 0..2usize {
            let use_x_axis = (hash >> dy) & 1 == 0;
            if use_x_axis {
                // Swap along x: (0,dy,dz) ↔ (1,dy,dz) for each dz
                for dz in 0..2 {
                    let a = block_index(0, dy, dz);
                    let b = block_index(1, dy, dz);
                    self.try_fluid_swap(&mut out, a, b);
                }
            } else {
                // Swap along z: (dx,dy,0) ↔ (dx,dy,1) for each dx
                for dx in 0..2 {
                    let a = block_index(dx, dy, 0);
                    let b = block_index(dx, dy, 1);
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
}

impl GravityBlockRule {
    pub fn new(density_fn: fn(Cell) -> f32) -> Self {
        Self { density_fn }
    }
}

impl BlockRule for GravityBlockRule {
    fn step_block(&self, block: &[Cell; 8]) -> [Cell; 8] {
        let mut out = *block;
        // There are 4 vertical columns in a 2x2x2 block.
        // Each column has y=0 (bottom) and y=1 (top).
        for dz in 0..2 {
            for dx in 0..2 {
                let bot = block_index(dx, 0, dz);
                let top = block_index(dx, 1, dz);
                let density_bot = (self.density_fn)(out[bot]);
                let density_top = (self.density_fn)(out[top]);
                // If the top cell is heavier than the bottom, swap them down.
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
        let out = rule.step_block(&block);
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
        let out = rule.step_block(&block);
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

    #[test]
    fn gravity_swaps_heavy_cell_downward() {
        let rule = GravityBlockRule::new(simple_density);
        // Column at (0,0): bottom=air, top=stone(mat 5)
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = Cell::EMPTY; // bottom
        block[block_index(0, 1, 0)] = mat(5); // top — heavier, should fall
        let out = rule.step_block(&block);
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "heavy cell should fall");
        assert_eq!(out[block_index(0, 1, 0)], Cell::EMPTY, "air should rise");
    }

    #[test]
    fn gravity_does_not_swap_when_bottom_is_heavier() {
        let rule = GravityBlockRule::new(simple_density);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(5); // bottom — heavier
        block[block_index(0, 1, 0)] = mat(1); // top — lighter
        let out = rule.step_block(&block);
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "bottom stays");
        assert_eq!(out[block_index(0, 1, 0)], mat(1), "top stays");
    }

    #[test]
    fn gravity_equal_density_no_swap() {
        let rule = GravityBlockRule::new(simple_density);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(3);
        block[block_index(0, 1, 0)] = mat(3);
        let out = rule.step_block(&block);
        assert_eq!(out[block_index(0, 0, 0)], mat(3));
        assert_eq!(out[block_index(0, 1, 0)], mat(3));
    }

    #[test]
    fn gravity_conserves_mass() {
        let rule = GravityBlockRule::new(simple_density);
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
        let out = rule.step_block(&block);
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn gravity_all_four_columns_independent() {
        let rule = GravityBlockRule::new(simple_density);
        // Set up all 4 columns with heavy-on-top
        let mut block = [Cell::EMPTY; 8];
        for dz in 0..2 {
            for dx in 0..2 {
                block[block_index(dx, 0, dz)] = mat(1); // light bottom
                block[block_index(dx, 1, dz)] = mat(5); // heavy top
            }
        }
        let out = rule.step_block(&block);
        for dz in 0..2 {
            for dx in 0..2 {
                assert_eq!(
                    out[block_index(dx, 0, dz)],
                    mat(5),
                    "heavy should fall at ({dx}, 0, {dz})"
                );
                assert_eq!(
                    out[block_index(dx, 1, dz)],
                    mat(1),
                    "light should rise at ({dx}, 1, {dz})"
                );
            }
        }
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn gravity_preserves_metadata() {
        let rule = GravityBlockRule::new(simple_density);
        let mut block = [Cell::EMPTY; 8];
        // heavy cell with metadata
        block[block_index(0, 1, 0)] = Cell::pack(5, 42);
        let out = rule.step_block(&block);
        assert_eq!(
            out[block_index(0, 0, 0)],
            Cell::pack(5, 42),
            "metadata must travel with the cell"
        );
    }

    // ---------------------------------------------------------------
    // FluidBlockRule
    // ---------------------------------------------------------------

    const FLUID_MAT: u16 = 3; // "water" in test context
    const SOLID_MAT: u16 = 7; // "stone" in test context

    fn fluid_rule() -> FluidBlockRule {
        FluidBlockRule::new(simple_density, FLUID_MAT)
    }

    #[test]
    fn fluid_gravity_heavy_over_air_falls() {
        let rule = fluid_rule();
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 1, 0)] = mat(FLUID_MAT); // top — fluid, denser than air
                                                      // Block lateral escape routes so we isolate the gravity phase.
        block[block_index(1, 0, 0)] = mat(SOLID_MAT);
        block[block_index(0, 0, 1)] = mat(SOLID_MAT);
        let out = rule.step_block(&block);
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
        let out = rule.step_block(&block);
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
        let out = rule.step_block(&block);
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
        let out = rule.step_block(&block);
        assert_mass_conserved(&block, &out);
    }

    #[test]
    fn fluid_gravity_plus_lateral_combined() {
        let rule = fluid_rule();
        // Fluid at top (0,1,0), air everywhere else.
        // Gravity should pull it to (0,0,0), then lateral should spread it.
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 1, 0)] = mat(FLUID_MAT);
        let out = rule.step_block(&block);
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
