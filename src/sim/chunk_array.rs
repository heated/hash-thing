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
use crate::sim::world::brute_step_grid;
use crate::sim::World;
use crate::terrain::materials::MaterialRegistry;
use crate::terrain::TerrainParams;

// Dead-code allow: only the in-module parity tests construct this type
// today. The next bead promoting the chunk-array baseline bench to use
// `ChunkArrayWorld::step` will wire the first non-test caller.
#[allow(dead_code)]
pub(crate) struct ChunkArrayWorld {
    level: u32,
    side: usize,
    grid: Vec<CellState>,
    materials: MaterialRegistry,
    generation: u64,
}

#[allow(dead_code)]
impl ChunkArrayWorld {
    /// Snapshot `source` into a flat-array world. The chunk grid is a
    /// copy of `source.flatten()`; the materials registry is cloned;
    /// `generation` mirrors the source so iterated stepping starts in
    /// phase with a hashlife `World` built the same way.
    pub(crate) fn from_world(source: &World) -> Self {
        Self {
            level: source.level,
            side: source.side(),
            grid: source.flatten(),
            materials: source.materials().clone(),
            generation: source.generation,
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

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn population(&self) -> u64 {
        self.grid.iter().filter(|&&c| c != 0).count() as u64
    }

    /// Advance one generation. Mirrors `World::step` minus `commit_step`
    /// (no octree to commit to). Both paths route through
    /// `brute_step_grid`, so divergence in the brute-force kernel
    /// itself is by-construction impossible — see the parity-test
    /// commentary in this file's `tests` module for what the parity
    /// test actually catches.
    ///
    /// Generation timing matches `World::step`: the call reads
    /// `self.generation`, computes the next grid, then bumps —
    /// equivalent to `World::step` reading generation, calling
    /// `step_grid`, then `commit_step` bumping inside the octree
    /// commit at the `self.generation += 1` line in `commit_step`.
    pub(crate) fn step(&mut self) {
        let next = brute_step_grid(&self.grid, self.side, &self.materials, self.generation);
        self.grid = next;
        self.generation += 1;
    }

    /// Direct cell write — used by microchurn parity tests to drop sand
    /// identically into both engines. **OOB semantics deviate from
    /// `World::set`:** this method silently drops out-of-bounds writes
    /// (absorbing boundary, matching `World::apply_mutations`'s queue
    /// path for OOB `SetCell`), whereas `World::set` panics on negative
    /// or beyond-side coords. Microchurn tests stay in-bounds by
    /// construction; the `debug_assert!` below catches a future caller
    /// who accidentally passes OOB and then can't reproduce the
    /// hashlife-side panic.
    pub(crate) fn set(&mut self, x: i64, y: i64, z: i64, state: CellState) {
        debug_assert!(
            x >= 0
                && y >= 0
                && z >= 0
                && (x as usize) < self.side
                && (y as usize) < self.side
                && (z as usize) < self.side,
            "ChunkArrayWorld::set OOB at ({x}, {y}, {z}) for side={}; \
             World::set would panic — both must use in-bounds coords",
            self.side
        );
        if x < 0 || y < 0 || z < 0 {
            return;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= self.side || y >= self.side || z >= self.side {
            return;
        }
        let idx = x + y * self.side + z * self.side * self.side;
        self.grid[idx] = state;
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

    // --- Tick parity (8ppq.1.3) -----------------------------------------
    //
    // Post-refactor `World::step` and `ChunkArrayWorld::step` both call
    // the same `brute_step_grid` free function, so a "two-engine
    // divergence" reading of the parity claim is structurally
    // tautological — the kernels agree by construction. What these
    // tests actually catch:
    //
    // - `commit_step` round-trip integrity: hashlife flattens its
    //   octree, the brute kernel produces a new flat grid, then
    //   `commit_step` re-builds an octree via `from_flat`,
    //   `compacted_with_remap`, and store swap. If that round-trip
    //   ever perturbs cell values (octree compaction bug, store remap
    //   off-by-one, etc.) the parity test fires.
    // - `ChunkArrayWorld::set` ↔ `World::set` write-path agreement on
    //   the microchurn variant.
    // - Future regressions that bypass the shared `brute_step_grid`
    //   (e.g. a hashlife-only fast path that diverges from brute).
    //
    // The "single source of truth" framing is intentional per
    // 8ppq.1.3's plan-review #7. If a future bead introduces a
    // chunk-array native step (path a, parallel to the SVDAG-rebuild
    // skip in 8ppq.1.2.A's native seeder), the parity test gains its
    // genuine cross-engine bite back.

    use crate::sim::WorldCoord;
    use crate::terrain::materials::SAND;

    /// Failure-localization helper. On byte mismatch, prints the top
    /// 3 (chunk_material, hashlife_material) pairs that disagreed —
    /// per the bead's "material-class divergence map" verification ask.
    fn assert_byte_parity_with_class_map(
        chunk: &ChunkArrayWorld,
        hashlife: &World,
        gen: u64,
    ) {
        let h = hashlife.flatten();
        let c = chunk.grid();
        if c == &h[..] {
            return;
        }
        let mut classes: std::collections::HashMap<(u16, u16), u64> =
            std::collections::HashMap::new();
        let mut first_diffs = 0usize;
        for (i, (&a, &b)) in c.iter().zip(h.iter()).enumerate() {
            if a == b {
                continue;
            }
            let (ma, mb) = (Cell::from_raw(a).material(), Cell::from_raw(b).material());
            *classes.entry((ma, mb)).or_default() += 1;
            if first_diffs < 5 {
                eprintln!("  diff at idx={i}: chunk={a:#x}(mat={ma}) hashlife={b:#x}(mat={mb})");
                first_diffs += 1;
            }
        }
        let mut top: Vec<_> = classes.into_iter().collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        eprintln!("  divergence-class map (top 3, chunk_mat -> hashlife_mat):");
        for ((ma, mb), n) in top.iter().take(3) {
            eprintln!("    {ma:>3} -> {mb:<3}: {n} cells");
        }
        panic!(
            "chunk vs hashlife diverged at gen {gen} (chunk gen={} hashlife gen={})",
            chunk.generation(),
            hashlife.generation,
        );
    }

    fn tick_parity_idle(level: u32, ticks: usize) {
        let params = TerrainParams::for_level(level);
        let mut hashlife = World::new(level);
        hashlife.seed_terrain(&params).unwrap();
        let mut chunk = ChunkArrayWorld::from_world(&hashlife);

        for n in 0..=ticks {
            assert_byte_parity_with_class_map(&chunk, &hashlife, n as u64);
            if n == ticks {
                break;
            }
            chunk.step();
            hashlife.step();
        }
    }

    /// Bead Verification §1: one-tick parity is the cheapest divergence-
    /// localization tool. If this fails, the bug is in single-tick
    /// semantics, not iterated drift.
    #[test]
    fn tick_parity_level_5_one_tick() {
        tick_parity_idle(5, 1);
    }

    #[test]
    fn tick_parity_level_5_idle_30_ticks() {
        tick_parity_idle(5, 30);
    }

    /// Level 6 / 100 ticks gated `#[ignore]` for soft-cap policy.
    /// Run via:
    ///   cargo test --profile perf -p hash-thing chunk_array::tests::tick_parity_level_6_idle_100_ticks -- --ignored --nocapture
    #[test]
    #[ignore]
    fn tick_parity_level_6_idle_100_ticks() {
        tick_parity_idle(6, 100);
    }

    /// Microchurn parity (bead acceptance §microchurn): drop sand at
    /// random top cells identically into both engines between steps.
    /// This is what actually exercises Margolus parity + tick_divisor
    /// — idle terrain reaches near-fixed-point fast.
    fn tick_parity_microchurn(level: u32, ticks: usize, sand_per_step: usize) {
        let params = TerrainParams::for_level(level);
        let mut hashlife = World::new(level);
        hashlife.seed_terrain(&params).unwrap();
        let mut chunk = ChunkArrayWorld::from_world(&hashlife);

        // Deterministic xorshift — both worlds get the same sand drops
        // and the test is reproducible.
        let side = (1i64) << level;
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next_u64 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for n in 0..=ticks {
            assert_byte_parity_with_class_map(&chunk, &hashlife, n as u64);
            if n == ticks {
                break;
            }
            for _ in 0..sand_per_step {
                let x = (next_u64() % (side as u64 - 4)) as i64 + 2;
                let y = side - 4 + (next_u64() % 2) as i64;
                let z = (next_u64() % (side as u64 - 4)) as i64 + 2;
                hashlife.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), SAND);
                chunk.set(x, y, z, SAND);
            }
            chunk.step();
            hashlife.step();
        }
    }

    #[test]
    fn tick_parity_level_5_microchurn_30_ticks() {
        tick_parity_microchurn(5, 30, 4);
    }

    #[test]
    #[ignore]
    fn tick_parity_level_6_microchurn_100_ticks() {
        tick_parity_microchurn(6, 100, 8);
    }
}
