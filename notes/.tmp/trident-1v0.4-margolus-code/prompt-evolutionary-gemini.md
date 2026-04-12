Read notes/.tmp/trident-1v0.4-margolus-code/CodeReview-Evolutionary.md and follow the instructions.

THIS IS A READ-ONLY REVIEW. Your ONLY output is the review file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.

OUTPUT: Write your review to .ship-notes/1v0.4-code-review-2026-04-12T0825/evolutionary-gemini.md

# CONTEXT

Branch: onyx/1v0.4-margolus
Commits:
202b26b fmt: cargo fmt
749df98 world: integration tests for block stepping pipeline
9cea4b5 world: add simulation_seed and block-wise stepping pass
5650f76 materials: extend MaterialRegistry with block rule storage
8d989d1 sim: add BlockRule trait, BlockContext, and Margolus transforms
bce9600 plan: 1v0.4 Margolus 2x2x2 movement phase

Diff stat:
 .beads/interactions.jsonl |   5 +-
 src/sim/margolus.rs       | 244 ++++++++++++++++++++++++++++++++++++++
 src/sim/mod.rs            |   1 +
 src/sim/rule.rs           |  72 ++++++++++++
 src/sim/world.rs          | 289 +++++++++++++++++++++++++++++++++++++++++++++-
 src/terrain/materials.rs  |  49 +++++++-
 6 files changed, 652 insertions(+), 8 deletions(-)

Full diff follows below.

diff --git a/.beads/interactions.jsonl b/.beads/interactions.jsonl
index 310639c..b818b7d 100644
--- a/.beads/interactions.jsonl
+++ b/.beads/interactions.jsonl
@@ -14,6 +14,5 @@
 {"id":"int-40139d6e","kind":"field_change","created_at":"2026-04-12T02:22:58.644604Z","actor":"spark","issue_id":"hash-thing-bx7","extra":{"field":"status","new_value":"open","old_value":"in_progress"}}
 {"id":"int-534a2a3c","kind":"field_change","created_at":"2026-04-12T02:22:58.645145Z","actor":"spark","issue_id":"hash-thing-bx7","extra":{"field":"assignee","new_value":"","old_value":"spark"}}
 {"id":"int-7964bc34","kind":"field_change","created_at":"2026-04-12T02:45:36.074121Z","actor":"Edward Swernofsky","issue_id":"hash-thing-eg3","extra":{"field":"status","new_value":"closed","old_value":"in_progress","reason":"Closed"}}
-{"id":"int-cb350ba0","kind":"field_change","created_at":"2026-04-12T02:53:16.064492Z","actor":"spark","issue_id":"hash-thing-5bb.5","extra":{"field":"status","new_value":"open","old_value":"closed"}}
-{"id":"int-0518baee","kind":"field_change","created_at":"2026-04-12T02:56:27.307671Z","actor":"spark","issue_id":"hash-thing-5bb.5","extra":{"field":"status","new_value":"in_progress","old_value":"open"}}
-{"id":"int-5482cb2d","kind":"field_change","created_at":"2026-04-12T02:56:27.308546Z","actor":"spark","issue_id":"hash-thing-5bb.5","extra":{"field":"assignee","new_value":"spark","old_value":"Edward Swernofsky"}}
+{"id":"int-47973a34","kind":"field_change","created_at":"2026-04-12T02:47:35.148236Z","actor":"Edward Swernofsky","issue_id":"hash-thing-p3z","extra":{"field":"status","new_value":"closed","old_value":"in_progress","reason":"Closed"}}
+{"id":"int-d694bae2","kind":"field_change","created_at":"2026-04-12T02:50:16.596284Z","actor":"Edward Swernofsky","issue_id":"hash-thing-69z","extra":{"field":"status","new_value":"blocked","old_value":"in_progress"}}
diff --git a/src/sim/margolus.rs b/src/sim/margolus.rs
new file mode 100644
index 0000000..b5f02de
--- /dev/null
+++ b/src/sim/margolus.rs
@@ -0,0 +1,244 @@
+// Public API consumed by future beads (1v0.5 gravity/fluid). Some entry points
+// only have test call sites until main.rs wires them. Silence dead_code until then.
+#![allow(dead_code)]
+//! Built-in Margolus block transforms.
+//!
+//! Each struct implements `BlockRule` for a specific 2x2x2 block behavior.
+//! All transforms must preserve the multiset of input cells (mass conservation).
+
+use crate::octree::Cell;
+use crate::sim::rule::{block_index, BlockContext, BlockRule};
+
+/// Identity block rule — returns the block unchanged. Useful for testing
+/// the pipeline (flatten → partition → iterate → apply → rebuild) with
+/// zero behavioral change.
+pub struct IdentityBlockRule;
+
+impl BlockRule for IdentityBlockRule {
+    fn step_block(&self, block: &[Cell; 8], _ctx: &BlockContext) -> [Cell; 8] {
+        *block
+    }
+}
+
+/// Gravity block rule — heavy materials swap downward with lighter ones.
+///
+/// Within the 2x2x2 block, each column (pair of cells sharing x,z but
+/// differing in y) is inspected. If the upper cell is heavier than the
+/// lower cell, they swap. "Heavier" is determined by a density lookup
+/// function provided at construction time.
+///
+/// This is a mass-conserving permutation: cells are rearranged, never
+/// created or destroyed.
+pub struct GravityBlockRule {
+    /// Returns the density for a given cell. Higher density = falls down.
+    /// Air (empty) should return 0.0.
+    density_fn: fn(Cell) -> f32,
+}
+
+impl GravityBlockRule {
+    pub fn new(density_fn: fn(Cell) -> f32) -> Self {
+        Self { density_fn }
+    }
+}
+
+impl BlockRule for GravityBlockRule {
+    fn step_block(&self, block: &[Cell; 8], _ctx: &BlockContext) -> [Cell; 8] {
+        let mut out = *block;
+        // There are 4 vertical columns in a 2x2x2 block.
+        // Each column has y=0 (bottom) and y=1 (top).
+        for dz in 0..2 {
+            for dx in 0..2 {
+                let bot = block_index(dx, 0, dz);
+                let top = block_index(dx, 1, dz);
+                let density_bot = (self.density_fn)(out[bot]);
+                let density_top = (self.density_fn)(out[top]);
+                // If the top cell is heavier than the bottom, swap them down.
+                if density_top > density_bot {
+                    out.swap(bot, top);
+                }
+            }
+        }
+        out
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::octree::Cell;
+    use crate::sim::rule::BlockContext;
+
+    fn test_ctx() -> BlockContext {
+        BlockContext {
+            block_origin: [0, 0, 0],
+            generation: 0,
+            world_seed: 42,
+            rng_hash: 0,
+        }
+    }
+
+    fn mat(id: u16) -> Cell {
+        if id == 0 {
+            Cell::EMPTY
+        } else {
+            Cell::pack(id, 0)
+        }
+    }
+
+    /// Check that a block transform is a permutation of the input.
+    fn assert_mass_conserved(input: &[Cell; 8], output: &[Cell; 8]) {
+        let mut in_sorted: Vec<u16> = input.iter().map(|c| c.raw()).collect();
+        let mut out_sorted: Vec<u16> = output.iter().map(|c| c.raw()).collect();
+        in_sorted.sort();
+        out_sorted.sort();
+        assert_eq!(
+            in_sorted, out_sorted,
+            "block rule must conserve mass (multiset equality)"
+        );
+    }
+
+    // ---------------------------------------------------------------
+    // IdentityBlockRule
+    // ---------------------------------------------------------------
+
+    #[test]
+    fn identity_preserves_block_exactly() {
+        let block = [
+            mat(1),
+            mat(2),
+            mat(3),
+            mat(0),
+            mat(5),
+            mat(0),
+            mat(1),
+            mat(4),
+        ];
+        let rule = IdentityBlockRule;
+        let out = rule.step_block(&block, &test_ctx());
+        assert_eq!(block, out);
+    }
+
+    #[test]
+    fn identity_conserves_mass() {
+        let block = [
+            mat(1),
+            mat(2),
+            mat(3),
+            mat(4),
+            mat(5),
+            mat(6),
+            mat(7),
+            mat(8),
+        ];
+        let rule = IdentityBlockRule;
+        let out = rule.step_block(&block, &test_ctx());
+        assert_mass_conserved(&block, &out);
+    }
+
+    // ---------------------------------------------------------------
+    // GravityBlockRule
+    // ---------------------------------------------------------------
+
+    /// Simple density: material ID as density, air = 0.
+    fn simple_density(cell: Cell) -> f32 {
+        if cell.is_empty() {
+            0.0
+        } else {
+            cell.material() as f32
+        }
+    }
+
+    #[test]
+    fn gravity_swaps_heavy_cell_downward() {
+        let rule = GravityBlockRule::new(simple_density);
+        // Column at (0,0): bottom=air, top=stone(mat 5)
+        let mut block = [Cell::EMPTY; 8];
+        block[block_index(0, 0, 0)] = Cell::EMPTY; // bottom
+        block[block_index(0, 1, 0)] = mat(5); // top — heavier, should fall
+        let out = rule.step_block(&block, &test_ctx());
+        assert_eq!(out[block_index(0, 0, 0)], mat(5), "heavy cell should fall");
+        assert_eq!(out[block_index(0, 1, 0)], Cell::EMPTY, "air should rise");
+    }
+
+    #[test]
+    fn gravity_does_not_swap_when_bottom_is_heavier() {
+        let rule = GravityBlockRule::new(simple_density);
+        let mut block = [Cell::EMPTY; 8];
+        block[block_index(0, 0, 0)] = mat(5); // bottom — heavier
+        block[block_index(0, 1, 0)] = mat(1); // top — lighter
+        let out = rule.step_block(&block, &test_ctx());
+        assert_eq!(out[block_index(0, 0, 0)], mat(5), "bottom stays");
+        assert_eq!(out[block_index(0, 1, 0)], mat(1), "top stays");
+    }
+
+    #[test]
+    fn gravity_equal_density_no_swap() {
+        let rule = GravityBlockRule::new(simple_density);
+        let mut block = [Cell::EMPTY; 8];
+        block[block_index(0, 0, 0)] = mat(3);
+        block[block_index(0, 1, 0)] = mat(3);
+        let out = rule.step_block(&block, &test_ctx());
+        assert_eq!(out[block_index(0, 0, 0)], mat(3));
+        assert_eq!(out[block_index(0, 1, 0)], mat(3));
+    }
+
+    #[test]
+    fn gravity_conserves_mass() {
+        let rule = GravityBlockRule::new(simple_density);
+        let block = [
+            mat(5),
+            mat(0),
+            mat(3),
+            mat(1),
+            mat(0),
+            mat(7),
+            mat(2),
+            mat(4),
+        ];
+        let out = rule.step_block(&block, &test_ctx());
+        assert_mass_conserved(&block, &out);
+    }
+
+    #[test]
+    fn gravity_all_four_columns_independent() {
+        let rule = GravityBlockRule::new(simple_density);
+        // Set up all 4 columns with heavy-on-top
+        let mut block = [Cell::EMPTY; 8];
+        for dz in 0..2 {
+            for dx in 0..2 {
+                block[block_index(dx, 0, dz)] = mat(1); // light bottom
+                block[block_index(dx, 1, dz)] = mat(5); // heavy top
+            }
+        }
+        let out = rule.step_block(&block, &test_ctx());
+        for dz in 0..2 {
+            for dx in 0..2 {
+                assert_eq!(
+                    out[block_index(dx, 0, dz)],
+                    mat(5),
+                    "heavy should fall at ({dx}, 0, {dz})"
+                );
+                assert_eq!(
+                    out[block_index(dx, 1, dz)],
+                    mat(1),
+                    "light should rise at ({dx}, 1, {dz})"
+                );
+            }
+        }
+        assert_mass_conserved(&block, &out);
+    }
+
+    #[test]
+    fn gravity_preserves_metadata() {
+        let rule = GravityBlockRule::new(simple_density);
+        let mut block = [Cell::EMPTY; 8];
+        // heavy cell with metadata
+        block[block_index(0, 1, 0)] = Cell::pack(5, 42);
+        let out = rule.step_block(&block, &test_ctx());
+        assert_eq!(
+            out[block_index(0, 0, 0)],
+            Cell::pack(5, 42),
+            "metadata must travel with the cell"
+        );
+    }
+}
diff --git a/src/sim/mod.rs b/src/sim/mod.rs
index 415e128..9a53514 100644
--- a/src/sim/mod.rs
+++ b/src/sim/mod.rs
@@ -1,3 +1,4 @@
+pub mod margolus;
 pub mod rule;
 pub mod world;
 
diff --git a/src/sim/rule.rs b/src/sim/rule.rs
index 6925edc..b36543f 100644
--- a/src/sim/rule.rs
+++ b/src/sim/rule.rs
@@ -12,6 +12,46 @@ pub trait CaRule {
     fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell;
 }
 
+/// Context passed to block rules for deterministic RNG and position awareness.
+///
+/// All fields are pure functions of position + generation + seed, so block rules
+/// remain Hashlife-compatible (no global mutable state).
+#[derive(Clone, Copy, Debug)]
+#[allow(dead_code)]
+pub struct BlockContext {
+    /// World-space origin of the block's (0,0,0) corner.
+    pub block_origin: [i64; 3],
+    /// Current simulation generation.
+    pub generation: u64,
+    /// Per-world seed for deterministic randomness.
+    pub world_seed: u64,
+    /// Pre-computed RNG hash from `rng::cell_hash(origin, generation, seed)`.
+    pub rng_hash: u64,
+}
+
+/// A block-based CA rule operating on 2x2x2 cell blocks.
+///
+/// Block rules implement mass-conserving permutations: the output must be a
+/// rearrangement of the input cells (multiset equality). This invariant enables
+/// Margolus-style movement (sand falling, fluid flow) without creating or
+/// destroying matter.
+///
+/// Cell ordering within the block follows octant convention:
+///   `block_index(dx, dy, dz) = dx + dy*2 + dz*4`
+/// matching `octant_index` in `src/octree/node.rs`.
+pub trait BlockRule {
+    fn step_block(&self, block: &[Cell; 8], ctx: &BlockContext) -> [Cell; 8];
+}
+
+/// Map local (dx, dy, dz) offsets (each 0 or 1) to an index in `[Cell; 8]`.
+///
+/// Matches `octant_index` in `src/octree/node.rs` — this is a load-bearing
+/// invariant. Do not change one without the other.
+#[inline]
+pub const fn block_index(dx: usize, dy: usize, dz: usize) -> usize {
+    dx + dy * 2 + dz * 4
+}
+
 /// Identity rule for static materials. Returns the center cell unchanged.
 pub struct NoopRule;
 
@@ -334,4 +374,36 @@ mod tests {
             Cell::EMPTY
         );
     }
+
+    // ---------------------------------------------------------------
+    // BlockRule / BlockContext / block_index tests
+    // ---------------------------------------------------------------
+
+    #[test]
+    fn block_index_matches_octant_convention() {
+        // dx + dy*2 + dz*4
+        assert_eq!(block_index(0, 0, 0), 0);
+        assert_eq!(block_index(1, 0, 0), 1);
+        assert_eq!(block_index(0, 1, 0), 2);
+        assert_eq!(block_index(1, 1, 0), 3);
+        assert_eq!(block_index(0, 0, 1), 4);
+        assert_eq!(block_index(1, 0, 1), 5);
+        assert_eq!(block_index(0, 1, 1), 6);
+        assert_eq!(block_index(1, 1, 1), 7);
+    }
+
+    #[test]
+    fn block_index_covers_all_8() {
+        let mut seen = [false; 8];
+        for dz in 0..2 {
+            for dy in 0..2 {
+                for dx in 0..2 {
+                    let idx = block_index(dx, dy, dz);
+                    assert!(!seen[idx], "duplicate index {idx}");
+                    seen[idx] = true;
+                }
+            }
+        }
+        assert!(seen.iter().all(|&s| s));
+    }
 }
diff --git a/src/sim/world.rs b/src/sim/world.rs
index a11bbf9..64acdf7 100644
--- a/src/sim/world.rs
+++ b/src/sim/world.rs
@@ -1,6 +1,7 @@
-use super::rule::ALIVE;
+use super::rule::{block_index, BlockContext, ALIVE};
 use crate::octree::{Cell, CellState, NodeId, NodeStore};
-use crate::terrain::materials::{MaterialRegistry, FIRE, WATER};
+use crate::rng::cell_hash;
+use crate::terrain::materials::{BlockRuleId, MaterialRegistry, FIRE, WATER};
 use crate::terrain::{carve_caves, gen_region, GenStats, TerrainParams};
 
 /// The simulation world. Owns the octree store and manages stepping.
@@ -13,6 +14,7 @@ pub struct World {
     pub root: NodeId,
     pub level: u32, // root level — grid is 2^level per side
     pub generation: u64,
+    pub simulation_seed: u64,
     pub materials: MaterialRegistry,
 }
 
@@ -39,6 +41,7 @@ impl World {
             root,
             level,
             generation: 0,
+            simulation_seed: 0,
             materials,
         }
     }
@@ -79,13 +82,17 @@ impl World {
         self.store.flatten(self.root, self.side())
     }
 
-    /// Step the simulation forward one generation using per-material rule dispatch.
-    /// This is the simple brute-force path — true Hashlife stepping replaces it later.
+    /// Step the simulation forward one generation.
+    ///
+    /// Pipeline: flatten → cell-wise CaRule pass → block-wise BlockRule pass → rebuild.
+    /// Single flatten/rebuild per tick. This is the brute-force path — true Hashlife
+    /// recursive stepping replaces it later.
     pub fn step(&mut self) {
         let side = self.side();
         let grid = self.flatten();
         let mut next = vec![0 as CellState; side * side * side];
 
+        // Phase 1: cell-wise CaRule pass (Moore neighborhood).
         for z in 0..side {
             for y in 0..side {
                 for x in 0..side {
@@ -99,9 +106,108 @@ impl World {
             }
         }
 
+        // Phase 2: block-wise BlockRule pass (Margolus 2x2x2).
+        self.step_blocks(&mut next, side);
+
         self.commit_step(&next, side);
     }
 
+    /// Apply block rules to non-overlapping 2x2x2 partitions of the grid.
+    ///
+    /// Partition offset alternates per generation: even → (0,0,0), odd → (1,1,1).
+    /// Blocks at the edges that would extend past the grid boundary are skipped.
+    ///
+    /// Dispatch: collect distinct BlockRuleIds across the 8 cells. If exactly one
+    /// distinct rule exists, run it. If zero or multiple: skip (identity).
+    fn step_blocks(&self, grid: &mut [CellState], side: usize) {
+        let offset = if self.generation.is_multiple_of(2) {
+            0
+        } else {
+            1
+        };
+
+        let mut bz = offset;
+        while bz + 1 < side {
+            let mut by = offset;
+            while by + 1 < side {
+                let mut bx = offset;
+                while bx + 1 < side {
+                    self.apply_block(grid, side, bx, by, bz);
+                    bx += 2;
+                }
+                by += 2;
+            }
+            bz += 2;
+        }
+    }
+
+    /// Apply the block rule for a single 2x2x2 block at (bx, by, bz).
+    fn apply_block(&self, grid: &mut [CellState], side: usize, bx: usize, by: usize, bz: usize) {
+        // Read the 8 cells.
+        let mut block = [Cell::EMPTY; 8];
+        for dz in 0..2 {
+            for dy in 0..2 {
+                for dx in 0..2 {
+                    let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
+                    block[block_index(dx, dy, dz)] = Cell::from_raw(grid[idx]);
+                }
+            }
+        }
+
+        // Skip all-empty blocks (optimization).
+        if block.iter().all(|c| c.is_empty()) {
+            return;
+        }
+
+        // Dispatch: find the unique block rule across all cells.
+        let rule_id = match self.unique_block_rule(&block) {
+            Some(id) => id,
+            None => return, // zero or multiple distinct rules → skip
+        };
+
+        let rule = self.materials.block_rule(rule_id);
+        let ctx = BlockContext {
+            block_origin: [bx as i64, by as i64, bz as i64],
+            generation: self.generation,
+            world_seed: self.simulation_seed,
+            rng_hash: cell_hash(
+                bx as i64,
+                by as i64,
+                bz as i64,
+                self.generation,
+                self.simulation_seed,
+            ),
+        };
+
+        let result = rule.step_block(&block, &ctx);
+
+        // Write back the 8 cells.
+        for dz in 0..2 {
+            for dy in 0..2 {
+                for dx in 0..2 {
+                    let idx = (bx + dx) + (by + dy) * side + (bz + dz) * side * side;
+                    grid[idx] = result[block_index(dx, dy, dz)].raw();
+                }
+            }
+        }
+    }
+
+    /// Find the unique BlockRuleId across all non-empty cells in a block.
+    /// Returns `Some(id)` if exactly one distinct rule; `None` if zero or multiple.
+    fn unique_block_rule(&self, block: &[Cell; 8]) -> Option<BlockRuleId> {
+        let mut found: Option<BlockRuleId> = None;
+        for cell in block {
+            if let Some(id) = self.materials.block_rule_id_for_cell(*cell) {
+                match found {
+                    None => found = Some(id),
+                    Some(existing) if existing == id => {}
+                    Some(_) => return None, // multiple distinct rules → skip
+                }
+            }
+        }
+        found
+    }
+
     /// Place a random seed pattern in the center of the world.
     ///
     /// The loop bounds are clamped to `[0, side)` so `radius > center` does
@@ -528,4 +634,179 @@ mod tests {
         assert_eq!(world.get(4, 4, 4), FIRE);
         assert_eq!(world.get(5, 4, 4), GRASS);
     }
+
+    // -----------------------------------------------------------------
+    // Margolus block-rule integration tests
+    // -----------------------------------------------------------------
+
+    use crate::sim::margolus::{GravityBlockRule, IdentityBlockRule};
+    use crate::terrain::materials::{DIRT_MATERIAL_ID, STONE_MATERIAL_ID, WATER_MATERIAL_ID};
+
+    fn simple_density(cell: Cell) -> f32 {
+        if cell.is_empty() {
+            return 0.0;
+        }
+        match cell.material() {
+            1 => 5.0,  // stone
+            2 => 2.0,  // dirt
+            3 => 1.2,  // grass
+            4 => 0.05, // fire
+            5 => 1.0,  // water
+            _ => 1.0,
+        }
+    }
+
+    /// Wire a gravity block rule onto specific materials and return the world.
+    fn gravity_world(materials_with_gravity: &[u16]) -> World {
+        let mut world = World::new(3); // 8x8x8
+        let gravity_id = world
+            .materials
+            .register_block_rule(GravityBlockRule::new(simple_density));
+        for &mat_id in materials_with_gravity {
+            world.materials.assign_block_rule(mat_id, gravity_id);
+        }
+        world
+    }
+
+    #[test]
+    fn identity_block_rule_leaves_world_unchanged() {
+        let mut world = World::new(3);
+        let identity_id = world.materials.register_block_rule(IdentityBlockRule);
+        world
+            .materials
+            .assign_block_rule(STONE_MATERIAL_ID, identity_id);
+
+        // Place some stone cells.
+        world.set(2, 4, 2, STONE);
+        world.set(3, 4, 2, STONE);
+        let pop_before = world.population();
+        let flat_before = world.flatten();
+
+        world.step();
+
+        assert_eq!(world.population(), pop_before);
+        // CaRule for stone is NoopRule, identity BlockRule changes nothing.
+        assert_eq!(world.flatten(), flat_before);
+    }
+
+    #[test]
+    fn gravity_drops_heavy_cell_one_step() {
+        // Use even generation (offset=0) so block at (2,2,2) is aligned.
+        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
+        // Place dirt at y=3 (top of block), air at y=2 (bottom of block).
+        // Block origin = (2,2,2), so positions (2,2,2) and (2,3,2) are
+        // in the same block column.
+        world.set(2, 3, 2, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+        assert_eq!(world.get(2, 2, 2), 0, "bottom should be air initially");
+        assert_eq!(
+            world.get(2, 3, 2),
+            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
+            "top should be dirt initially"
+        );
+
+        world.step();
+
+        // After step, dirt should have fallen from y=3 to y=2.
+        assert_eq!(
+            world.get(2, 2, 2),
+            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
+            "dirt should have fallen to bottom"
+        );
+        assert_eq!(world.get(2, 3, 2), 0, "top should now be air");
+    }
+
+    #[test]
+    fn gravity_conserves_population() {
+        let mut world = gravity_world(&[DIRT_MATERIAL_ID, WATER_MATERIAL_ID]);
+        // Scatter some cells in even-aligned positions.
+        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+        world.set(2, 1, 2, Cell::pack(WATER_MATERIAL_ID, 0).raw());
+        world.set(4, 3, 4, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+        let pop_before = world.population();
+
+        for _ in 0..4 {
+            world.step();
+        }
+
+        assert_eq!(
+            world.population(),
+            pop_before,
+            "gravity must conserve total cell count"
+        );
+    }
+
+    #[test]
+    fn alternating_offset_covers_different_blocks() {
+        // Even gen: offset 0 → block at (0,0,0). Odd gen: offset 1 → block at (1,1,1).
+        // A cell at (1,1,1) is in the even block but NOT the odd block's origin.
+        let mut world = gravity_world(&[DIRT_MATERIAL_ID]);
+        // Place dirt at (0, 1, 0) — in even block (0,0,0), column (0,_,0).
+        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+        assert_eq!(world.generation, 0);
+
+        // Gen 0 (even offset=0): block (0,0,0) is active → dirt falls from y=1 to y=0.
+        world.step();
+        assert_eq!(
+            world.get(0, 0, 0),
+            Cell::pack(DIRT_MATERIAL_ID, 0).raw(),
+            "dirt should fall on even generation"
+        );
+        assert_eq!(world.get(0, 1, 0), 0);
+    }
+
+    #[test]
+    fn mixed_block_rules_skip_block() {
+        // Two materials with different block rules → block should be skipped.
+        let mut world = World::new(3);
+        let gravity_a = world
+            .materials
+            .register_block_rule(GravityBlockRule::new(simple_density));
+        let gravity_b = world
+            .materials
+            .register_block_rule(GravityBlockRule::new(simple_density));
+        world
+            .materials
+            .assign_block_rule(DIRT_MATERIAL_ID, gravity_a);
+        world
+            .materials
+            .assign_block_rule(WATER_MATERIAL_ID, gravity_b);
+
+        // Place dirt and water in the same block — different rule IDs.
+        world.set(0, 1, 0, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+        world.set(1, 0, 0, Cell::pack(WATER_MATERIAL_ID, 0).raw());
+
+        let flat_before = world.flatten();
+        world.step();
+
+        // CaRule (NoopRule) doesn't change dirt/water. Block rule is skipped.
+        assert_eq!(
+            world.flatten(),
+            flat_before,
+            "mixed-rule block should be skipped (identity)"
+        );
+    }
+
+    #[test]
+    fn block_rule_deterministic_across_runs() {
+        let make_world = || {
+            let mut w = gravity_world(&[DIRT_MATERIAL_ID]);
+            w.simulation_seed = 12345;
+            w.set(2, 3, 2, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+            w.set(4, 5, 4, Cell::pack(DIRT_MATERIAL_ID, 0).raw());
+            w
+        };
+
+        let mut a = make_world();
+        let mut b = make_world();
+        for _ in 0..5 {
+            a.step();
+            b.step();
+        }
+
+        assert_eq!(
+            a.flatten(),
+            b.flatten(),
+            "block stepping must be deterministic"
+        );
+    }
 }
diff --git a/src/terrain/materials.rs b/src/terrain/materials.rs
index a786958..7bd3f81 100644
--- a/src/terrain/materials.rs
+++ b/src/terrain/materials.rs
@@ -5,7 +5,7 @@
 //! cell, and every other material ID maps to a validated `CellState`.
 
 use crate::octree::{Cell, CellState};
-use crate::sim::rule::{CaRule, FireRule, GameOfLife3D, NoopRule, WaterRule};
+use crate::sim::rule::{BlockRule, CaRule, FireRule, GameOfLife3D, NoopRule, WaterRule};
 
 pub type MaterialId = u16;
 
@@ -28,6 +28,9 @@ pub const WATER: CellState = Cell::pack(WATER_MATERIAL_ID, 0).raw();
 #[derive(Clone, Copy, Debug, PartialEq, Eq)]
 pub struct RuleId(pub usize);
 
+#[derive(Clone, Copy, Debug, PartialEq, Eq)]
+pub struct BlockRuleId(pub usize);
+
 #[allow(dead_code)]
 #[derive(Clone, Copy, Debug, PartialEq)]
 pub struct MaterialVisualProperties {
@@ -50,11 +53,13 @@ pub struct MaterialEntry {
     pub visual: MaterialVisualProperties,
     pub physical: MaterialPhysicalProperties,
     pub rule_id: RuleId,
+    pub block_rule_id: Option<BlockRuleId>,
 }
 
 pub struct MaterialRegistry {
     entries: Vec<Option<MaterialEntry>>,
     rules: Vec<Box<dyn CaRule>>,
+    block_rules: Vec<Box<dyn BlockRule>>,
 }
 
 impl MaterialRegistry {
@@ -62,6 +67,7 @@ impl MaterialRegistry {
         Self {
             entries: Vec::with_capacity(INITIAL_MATERIAL_SLOTS),
             rules: Vec::new(),
+            block_rules: Vec::new(),
         }
     }
 
@@ -91,6 +97,7 @@ impl MaterialRegistry {
                     conductivity: 0.0,
                 },
                 rule_id: static_rule,
+                block_rule_id: None,
             },
         );
         registry.insert(
@@ -107,6 +114,7 @@ impl MaterialRegistry {
                     conductivity: 0.2,
                 },
                 rule_id: static_rule,
+                block_rule_id: None,
             },
         );
         registry.insert(
@@ -123,6 +131,7 @@ impl MaterialRegistry {
                     conductivity: 0.08,
                 },
                 rule_id: static_rule,
+                block_rule_id: None,
             },
         );
         registry.insert(
@@ -139,6 +148,7 @@ impl MaterialRegistry {
                     conductivity: 0.04,
                 },
                 rule_id: static_rule,
+                block_rule_id: None,
             },
         );
         registry.insert(
@@ -155,6 +165,7 @@ impl MaterialRegistry {
                     conductivity: 0.0,
                 },
                 rule_id: fire_rule,
+                block_rule_id: None,
             },
         );
         registry.insert(
@@ -171,6 +182,7 @@ impl MaterialRegistry {
                     conductivity: 0.6,
                 },
                 rule_id: water_rule,
+                block_rule_id: None,
             },
         );
 
@@ -196,6 +208,17 @@ impl MaterialRegistry {
         Some(self.rules[entry.rule_id.0].as_ref())
     }
 
+    #[allow(dead_code)]
+    pub fn block_rule_for_cell(&self, cell: Cell) -> Option<&dyn BlockRule> {
+        let entry = self.entry(cell.material())?;
+        let block_rule_id = entry.block_rule_id?;
+        Some(self.block_rules[block_rule_id.0].as_ref())
+    }
+
+    pub fn block_rule_id_for_cell(&self, cell: Cell) -> Option<BlockRuleId> {
+        self.entry(cell.material())?.block_rule_id
+    }
+
     pub fn color_palette_rgba(&self) -> Vec<[f32; 4]> {
         let mut palette =
             vec![[0.0, 0.0, 0.0, 0.0]; self.entries.len().max(INITIAL_MATERIAL_SLOTS)];
@@ -216,6 +239,29 @@ impl MaterialRegistry {
         rule_id
     }
 
+    #[allow(dead_code)]
+    pub fn register_block_rule<R>(&mut self, rule: R) -> BlockRuleId
+    where
+        R: BlockRule + 'static,
+    {
+        let id = BlockRuleId(self.block_rules.len());
+        self.block_rules.push(Box::new(rule));
+        id
+    }
+
+    #[allow(dead_code)]
+    pub fn assign_block_rule(&mut self, material_id: MaterialId, block_rule_id: BlockRuleId) {
+        self.entries[material_id as usize]
+            .as_mut()
+            .expect("material must exist before assigning a block rule")
+            .block_rule_id = Some(block_rule_id);
+    }
+
+    /// Look up a block rule by ID. Used by `World::step_blocks()`.
+    pub fn block_rule(&self, id: BlockRuleId) -> &dyn BlockRule {
+        self.block_rules[id.0].as_ref()
+    }
+
     fn insert(&mut self, material_id: MaterialId, entry: MaterialEntry) {
         let material_id = material_id as usize;
         if self.entries.len() <= material_id {
@@ -273,6 +319,7 @@ mod tests {
                 conductivity: 0.8,
             },
             rule_id,
+            block_rule_id: None,
         }
     }
 
