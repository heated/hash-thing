//! Chunk-array baseline world. Flat `Vec<CellState>` storage, no octree,
//! no hashlife memo. Used by the chunk-array baseline comparator for
//! the navigation epic.
//!
//! ## Storage round-trip only — not a parallel seeder
//!
//! `ChunkArrayWorld` is a snapshot of a hashlife `World`'s flatten +
//! materials clone. It is **not** a parallel implementation of the
//! terrain seeder. Per-cell agreement with hashlife is by-construction
//! (we copy hashlife's flatten output); detecting independent-engine
//! seed drift requires a path-(a) native seeder, which is filed as a
//! follow-up.
//!
//! Concretely, when future bench code times
//! [`ChunkArrayWorld::seeded`] the timing is **hashlife's** SVDAG build
//! cost, not a chunk-array engine's seed cost. A native chunk-array
//! engine would skip leaf-coalescing entirely.
//!
//! Both points are intentional limits of the MVP comparator harness;
//! they exist to unblock L1 navigation-epic leads with matched scenes
//! while the honest two-engines-agree parity work matures separately.

use crate::octree::{Cell, CellState};
use crate::sim::World;
use crate::terrain::materials::MaterialRegistry;
use crate::terrain::TerrainParams;

// Dead-code allow: this bead lands the storage type and its parity
// tests; no production caller exists yet. The next bead in the chain
// (phase / tick-divisor parity) wires the first non-test caller via the
// chunk-array baseline bench.
#[allow(dead_code)]
pub(crate) struct ChunkArrayWorld {
    level: u32,
    side: usize,
    grid: Vec<CellState>,
    materials: MaterialRegistry,
}

#[allow(dead_code)]
impl ChunkArrayWorld {
    /// Snapshot `source` into a flat-array world. The chunk grid is a
    /// copy of `source.flatten()`; the materials registry is cloned.
    /// Does not capture `source.generation` — the next bead in this
    /// chain (phase / tick-divisor parity) wires that through when
    /// stepping is added.
    pub(crate) fn from_world(source: &World) -> Self {
        Self {
            level: source.level,
            side: source.side(),
            grid: source.flatten(),
            materials: source.materials().clone(),
        }
    }

    /// Convenience: build a hashlife `World`, seed it with `params`,
    /// snapshot it. For tests that want the chunk-array slice without
    /// holding the source `World` around.
    pub(crate) fn seeded(level: u32, params: &TerrainParams) -> Self {
        let mut world = World::new(level);
        world
            .seed_terrain(params)
            .expect("level-derived TerrainParams must validate");
        Self::from_world(&world)
    }

    pub(crate) fn level(&self) -> u32 {
        self.level
    }

    pub(crate) fn side(&self) -> usize {
        self.side
    }

    pub(crate) fn grid(&self) -> &[CellState] {
        &self.grid
    }

    pub(crate) fn materials(&self) -> &MaterialRegistry {
        &self.materials
    }

    pub(crate) fn population(&self) -> u64 {
        self.grid.iter().filter(|&&c| c != 0).count() as u64
    }

    /// Population indexed by material id. Length matches
    /// `MaterialRegistry::tick_divisor_flags().len()`. Material ids
    /// outside that range (sparse slots / post-mutation truncation) are
    /// silently dropped rather than panicking — those slots read 0.
    pub(crate) fn population_by_material(&self) -> Vec<u64> {
        let n = self.materials.tick_divisor_flags().len();
        let mut counts = vec![0u64; n];
        for &c in &self.grid {
            let mat = Cell::from_raw(c).material() as usize;
            if let Some(slot) = counts.get_mut(mat) {
                *slot += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_by_material(grid: &[CellState], n: usize) -> Vec<u64> {
        let mut counts = vec![0u64; n];
        for &c in grid {
            let mat = Cell::from_raw(c).material() as usize;
            if let Some(slot) = counts.get_mut(mat) {
                *slot += 1;
            }
        }
        counts
    }

    fn round_trip_at(level: u32) {
        let params = TerrainParams::for_level(level);
        let mut hashlife = World::new(level);
        hashlife
            .seed_terrain(&params)
            .expect("validated params must seed");

        let chunk = ChunkArrayWorld::from_world(&hashlife);

        assert_eq!(chunk.level(), level);
        assert_eq!(chunk.side(), hashlife.side());
        assert_eq!(chunk.grid().len(), chunk.side().pow(3));

        let hashlife_grid = hashlife.flatten();
        assert_eq!(chunk.grid(), &hashlife_grid[..]);
        assert_eq!(chunk.population(), hashlife.population());

        let n = hashlife.materials().tick_divisor_flags().len();
        assert_eq!(
            chunk.population_by_material(),
            count_by_material(&hashlife_grid, n)
        );

        // Cached-predicate parity. `MaterialRegistry: Clone` deep-clones
        // the cached `tick_divisor_flags` and `block_rule_tick_divisors`
        // vectors; if a future change to the Clone impl regresses to
        // recomputing-from-stale-state, this assertion catches it.
        assert_eq!(
            chunk.materials().tick_divisor_flags(),
            hashlife.materials().tick_divisor_flags(),
        );

        assert!(
            chunk.population() > 0,
            "level={level} default terrain seeded empty world"
        );
    }

    #[test]
    fn round_trip_level_5() {
        round_trip_at(5);
    }

    #[test]
    fn round_trip_level_6() {
        round_trip_at(6);
    }

    /// 128³ = 2M cells. Gated `#[ignore]` so the regular `cargo test`
    /// run stays under the project's 60s soft-cap. Run via:
    ///   cargo test --profile perf -p hash-thing chunk_array::tests::round_trip_level_7 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn round_trip_level_7() {
        round_trip_at(7);
    }

    /// `seeded(level, params)` is `from_world(World::new + seed_terrain)`
    /// in disguise; verify they produce equivalent snapshots.
    #[test]
    fn seeded_matches_explicit_from_world() {
        let level = 5;
        let params = TerrainParams::for_level(level);
        let convenience = ChunkArrayWorld::seeded(level, &params);

        let mut explicit_world = World::new(level);
        explicit_world
            .seed_terrain(&params)
            .expect("validated params must seed");
        let explicit = ChunkArrayWorld::from_world(&explicit_world);

        assert_eq!(convenience.level(), explicit.level());
        assert_eq!(convenience.side(), explicit.side());
        assert_eq!(convenience.grid(), explicit.grid());
        assert_eq!(convenience.population(), explicit.population());
        assert_eq!(
            convenience.population_by_material(),
            explicit.population_by_material()
        );
    }
}
