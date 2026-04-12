//! Built-in Margolus block transforms.
//!
//! Each struct implements `BlockRule` for a specific 2x2x2 block behavior.
//! All transforms must preserve the multiset of input cells (mass conservation).

use crate::octree::Cell;
use crate::sim::rule::{block_index, BlockContext, BlockRule};

/// Identity block rule — returns the block unchanged. Useful for testing
/// the pipeline (flatten → partition → iterate → apply → rebuild) with
/// zero behavioral change.
pub struct IdentityBlockRule;

impl BlockRule for IdentityBlockRule {
    fn step_block(&self, block: &[Cell; 8], _ctx: &BlockContext) -> [Cell; 8] {
        *block
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
    fn step_block(&self, block: &[Cell; 8], _ctx: &BlockContext) -> [Cell; 8] {
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
    use crate::sim::rule::BlockContext;

    fn test_ctx() -> BlockContext {
        BlockContext {
            block_origin: [0, 0, 0],
            generation: 0,
            world_seed: 42,
            rng_hash: 0,
        }
    }

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
        let out = rule.step_block(&block, &test_ctx());
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
        let out = rule.step_block(&block, &test_ctx());
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
        let out = rule.step_block(&block, &test_ctx());
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "heavy cell should fall");
        assert_eq!(
            out[block_index(0, 1, 0)],
            Cell::EMPTY,
            "air should rise"
        );
    }

    #[test]
    fn gravity_does_not_swap_when_bottom_is_heavier() {
        let rule = GravityBlockRule::new(simple_density);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(5); // bottom — heavier
        block[block_index(0, 1, 0)] = mat(1); // top — lighter
        let out = rule.step_block(&block, &test_ctx());
        assert_eq!(out[block_index(0, 0, 0)], mat(5), "bottom stays");
        assert_eq!(out[block_index(0, 1, 0)], mat(1), "top stays");
    }

    #[test]
    fn gravity_equal_density_no_swap() {
        let rule = GravityBlockRule::new(simple_density);
        let mut block = [Cell::EMPTY; 8];
        block[block_index(0, 0, 0)] = mat(3);
        block[block_index(0, 1, 0)] = mat(3);
        let out = rule.step_block(&block, &test_ctx());
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
        let out = rule.step_block(&block, &test_ctx());
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
        let out = rule.step_block(&block, &test_ctx());
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
        let out = rule.step_block(&block, &test_ctx());
        assert_eq!(
            out[block_index(0, 0, 0)],
            Cell::pack(5, 42),
            "metadata must travel with the cell"
        );
    }
}
