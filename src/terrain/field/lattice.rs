//! `LatticeField` — a 3D periodic grid of corridors, pillars, and rooms
//! with architectural proportions and active materials (water, lava, fire).
//!
//! Replaces the O(n^3) `seed_lattice_megastructure` with an octree-native
//! `RegionField`. `classify_box` collapses large uniform regions (corridor
//! AIR, pillar STONE) without visiting individual cells.
//!
//! The lattice is a repeating grid with period `cell_size` on all axes.
//! Each grid cell has:
//! - **Pillars** at the intersection of x-wall and z-wall zones (STONE)
//! - **Walls** along x and z boundaries (GRASS, flammable)
//! - **Floors** at the bottom of each cell (STONE or SAND, alternating)
//! - **Water channels** in even-z-row corridors on the ground floor
//! - **Lava pools** in rare cells (hash-selected)
//! - **Fire torches** at pillar edges in some cells (hash-selected)
//!
//! Room height varies per grid cell via a deterministic hash, giving visual
//! variety while remaining fully deterministic for `classify_box`.

use super::RegionField;
use crate::octree::CellState;
use crate::terrain::materials::{AIR, FIRE, GRASS, LAVA, SAND, STONE, WATER};

/// SplitMix64 finalizer for deterministic hashing.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Hash a 3D grid cell index to a u64.
#[inline]
fn hash_cell(gx: i64, gy: i64, gz: i64, seed: u64) -> u64 {
    let h = (gx as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (gy as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ (gz as u64).wrapping_mul(0x165667B19E3779F9)
        ^ seed;
    mix64(h)
}

#[derive(Clone, Debug)]
pub struct LatticeField {
    pub seed: u64,
    /// Lower corner of the active lattice region (inclusive).
    pub lo: [i64; 3],
    /// Upper corner of the active lattice region (exclusive).
    pub hi: [i64; 3],
    /// Lattice cell period (same on all three axes).
    pub cell_size: i64,
    /// Wall thickness on x and z axes (pillar = wall_thick x wall_thick).
    pub wall_thick: i64,
    /// Floor slab thickness.
    pub floor_thick: i64,
    /// Water channel depth (above floor).
    pub water_depth: i64,
}

impl LatticeField {
    /// Create a lattice field filling a world of the given level (side = 2^level).
    /// Uses a 1/16 margin on each side, matching the old `seed_lattice_megastructure`.
    pub fn for_world(level: u32, seed: u64) -> Self {
        let side = 1i64 << level;
        let margin = side / 16;
        let lo = margin;
        let hi = side - margin;
        let extent = hi - lo;

        // Cell size: aim for ~8 cells across for visual density, minimum 8.
        let cell_size = (extent / 8).max(8);
        let wall_thick = (cell_size / 5).max(1);
        let floor_thick = 2i64.min(cell_size / 4).max(1);
        let water_depth = (wall_thick + 1).min(cell_size / 3);

        Self {
            seed,
            lo: [lo, lo, lo],
            hi: [hi, hi, hi],
            cell_size,
            wall_thick,
            floor_thick,
            water_depth,
        }
    }

    /// Room height for a given grid cell. Varies by hash of (gx, gz).
    /// Returns the height in cells above the floor where walls end (AIR above).
    /// Range: [cell_size * 5/8, cell_size - floor_thick).
    #[inline]
    fn room_height(&self, gx: i64, gz: i64) -> i64 {
        let min_h = (self.cell_size * 5) / 8;
        let max_h = self.cell_size - self.floor_thick;
        if min_h >= max_h {
            return max_h;
        }
        let h = hash_cell(gx, 0, gz, self.seed.wrapping_add(1));
        let range = max_h - min_h;
        min_h + (h % range as u64) as i64
    }

    /// Whether this grid cell has a water channel on its ground floor.
    #[inline]
    fn has_water(&self, gx: i64, gy: i64, gz: i64) -> bool {
        // Water in even-z-row corridors on the ground floor only.
        gy == 0 && gz % 2 == 0 && gx % 2 == 0
    }

    /// Whether this grid cell has a lava pool.
    #[inline]
    fn has_lava(&self, gx: i64, gy: i64, gz: i64) -> bool {
        // Lava in every 4th cell on the second floor.
        gy == 1 && gx % 4 == 2 && gz % 4 == 2
    }

    /// Whether this grid cell has fire torches.
    #[inline]
    fn has_fire(&self, gx: i64, gy: i64, gz: i64) -> bool {
        gy == 0 && (gx + gz) % 3 == 0
    }

    /// Floor material for a given grid cell.
    #[inline]
    fn floor_material(&self, gx: i64, gz: i64) -> CellState {
        if (gx + gz) % 2 == 0 {
            STONE
        } else {
            SAND
        }
    }

    /// Maximum cy value that could contain a feature (water/lava/fire)
    /// above the floor. Classify_box uses this to know when corridor AIR
    /// is provably feature-free.
    #[inline]
    fn feature_ceiling(&self) -> i64 {
        // Water goes up to floor_thick + water_depth.
        // Fire is at floor_thick + 1. Lava is floor_thick..floor_thick+2.
        let water_top = self.floor_thick + self.water_depth;
        let lava_top = self.floor_thick + 2;
        let fire_top = self.floor_thick + 2;
        water_top.max(lava_top).max(fire_top)
    }

    /// Classify a single point. Assumes the point is within [lo, hi).
    #[inline]
    fn sample_inner(&self, lx: i64, ly: i64, lz: i64) -> CellState {
        let cs = self.cell_size;

        // Grid cell index and cell-local coords.
        let gx = lx.div_euclid(cs);
        let gy = ly.div_euclid(cs);
        let gz = lz.div_euclid(cs);
        let cx = lx.rem_euclid(cs);
        let cy = ly.rem_euclid(cs);
        let cz = lz.rem_euclid(cs);

        let in_x_wall = cx < self.wall_thick;
        let in_z_wall = cz < self.wall_thick;
        let in_floor = cy < self.floor_thick;
        let room_h = self.room_height(gx, gz);

        // Above room height: AIR (open to level above).
        // Pillars extend full height, walls only to room_h.
        let is_pillar = in_x_wall && in_z_wall;
        if !is_pillar && cy >= room_h {
            return AIR;
        }

        // Pillars: always STONE, full height of cell.
        if is_pillar {
            return STONE;
        }

        // Floor slabs.
        if in_floor {
            // Walls on the floor are STONE for structural support.
            if in_x_wall || in_z_wall {
                return STONE;
            }
            return self.floor_material(gx, gz);
        }

        // Walls (not pillar, not floor): GRASS up to room height.
        if in_x_wall || in_z_wall {
            return GRASS;
        }

        // Open corridor — check for features.
        // Water channels.
        if self.has_water(gx, gy, gz) && cy < self.floor_thick + self.water_depth {
            return WATER;
        }

        // Lava pools.
        if self.has_lava(gx, gy, gz) && cy < self.floor_thick + 2 {
            // Only in the center of the cell.
            let center_margin = self.wall_thick + 1;
            if cx > center_margin && cx < cs - 1 && cz > center_margin && cz < cs - 1 {
                return LAVA;
            }
        }

        // Fire torches: at pillar edges, one cell above floor.
        if self.has_fire(gx, gy, gz) && cy == self.floor_thick {
            // Place fire adjacent to the pillar (just outside wall_thick).
            if (cx == self.wall_thick && cz < self.wall_thick + 2)
                || (cz == self.wall_thick && cx < self.wall_thick + 2)
            {
                return FIRE;
            }
        }

        AIR
    }
}

impl RegionField for LatticeField {
    fn sample(&self, point: [i64; 3]) -> CellState {
        // Outside the active region: AIR.
        if point[0] < self.lo[0]
            || point[0] >= self.hi[0]
            || point[1] < self.lo[1]
            || point[1] >= self.hi[1]
            || point[2] < self.lo[2]
            || point[2] >= self.hi[2]
        {
            return AIR;
        }

        let lx = point[0] - self.lo[0];
        let ly = point[1] - self.lo[1];
        let lz = point[2] - self.lo[2];
        self.sample_inner(lx, ly, lz)
    }

    fn classify_box(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;

        // Box corners (inclusive lo, exclusive hi).
        let bx_lo = origin[0];
        let by_lo = origin[1];
        let bz_lo = origin[2];
        let bx_hi = origin[0] + size;
        let by_hi = origin[1] + size;
        let bz_hi = origin[2] + size;

        // Entirely outside the active region → AIR.
        if bx_hi <= self.lo[0]
            || bx_lo >= self.hi[0]
            || by_hi <= self.lo[1]
            || by_lo >= self.hi[1]
            || bz_hi <= self.lo[2]
            || bz_lo >= self.hi[2]
        {
            return Some(AIR);
        }

        // Partially outside → mixed (margin meets lattice).
        if bx_lo < self.lo[0]
            || bx_hi > self.hi[0]
            || by_lo < self.lo[1]
            || by_hi > self.hi[1]
            || bz_lo < self.lo[2]
            || bz_hi > self.hi[2]
        {
            return None;
        }

        let cs = self.cell_size;

        // If box spans a full cell period on any axis, it's definitely mixed.
        if size >= cs {
            return None;
        }

        // Convert to local coordinates.
        let lx_lo = bx_lo - self.lo[0];
        let ly_lo = by_lo - self.lo[1];
        let lz_lo = bz_lo - self.lo[2];
        let lx_hi = bx_hi - self.lo[0]; // exclusive
        let ly_hi = by_hi - self.lo[1];
        let lz_hi = bz_hi - self.lo[2];

        // Check if the box is within a single grid cell on each axis.
        // A box [lo, hi) is within one cell if floor(lo/cs) == floor((hi-1)/cs).
        let gx_lo = lx_lo.div_euclid(cs);
        let gx_hi = (lx_hi - 1).div_euclid(cs);
        let gy_lo = ly_lo.div_euclid(cs);
        let gy_hi = (ly_hi - 1).div_euclid(cs);
        let gz_lo = lz_lo.div_euclid(cs);
        let gz_hi = (lz_hi - 1).div_euclid(cs);

        if gx_lo != gx_hi || gy_lo != gy_hi || gz_lo != gz_hi {
            return None; // spans cell boundary
        }

        let gx = gx_lo;
        let gy = gy_lo;
        let gz = gz_lo;

        // Cell-local coordinate ranges.
        let cx_lo = lx_lo.rem_euclid(cs);
        let cx_hi_incl = (lx_hi - 1).rem_euclid(cs);
        let cy_lo = ly_lo.rem_euclid(cs);
        let cy_hi_incl = (ly_hi - 1).rem_euclid(cs);
        let cz_lo = lz_lo.rem_euclid(cs);
        let cz_hi_incl = (lz_hi - 1).rem_euclid(cs);

        let wt = self.wall_thick;
        let ft = self.floor_thick;
        let room_h = self.room_height(gx, gz);

        // Per-axis wall classification.
        let all_in_x_wall = cx_hi_incl < wt;
        let all_out_x_wall = cx_lo >= wt;
        let all_in_z_wall = cz_hi_incl < wt;
        let all_out_z_wall = cz_lo >= wt;
        let all_in_floor = cy_hi_incl < ft;
        let all_above_floor = cy_lo >= ft;

        // Pillar: both x and z in wall zone, full cell height.
        if all_in_x_wall && all_in_z_wall {
            return Some(STONE);
        }

        // Entirely above room height and provably no pillar cells in the box.
        // Pillars extend full height; walls end at room_h. A pillar cell needs
        // cx < wt AND cz < wt, so if the box is fully outside either wall
        // zone, no pillar is possible.
        if cy_lo >= room_h && (all_out_x_wall || all_out_z_wall) {
            return Some(AIR);
        }

        // Floor slab (not in pillar zone).
        if all_in_floor && all_out_x_wall && all_out_z_wall {
            // Floor material is uniform within a single grid cell.
            return Some(self.floor_material(gx, gz));
        }

        // Floor slab in wall zone: STONE.
        if all_in_floor && (all_in_x_wall || all_in_z_wall) {
            return Some(STONE);
        }

        // Wall (not pillar, not floor): GRASS if below room height.
        if all_above_floor && cy_hi_incl < room_h {
            if (all_in_x_wall && all_out_z_wall) || (all_in_z_wall && all_out_x_wall) {
                return Some(GRASS);
            }
        }

        // Open corridor above feature ceiling → AIR.
        if all_out_x_wall && all_out_z_wall && all_above_floor {
            let feat_ceil = self.feature_ceiling();
            if cy_lo >= feat_ceil && cy_hi_incl < room_h {
                return Some(AIR);
            }
            // No features in this grid cell → AIR for entire corridor above floor.
            if !self.has_water(gx, gy, gz)
                && !self.has_lava(gx, gy, gz)
                && !self.has_fire(gx, gy, gz)
                && cy_hi_incl < room_h
            {
                return Some(AIR);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::NodeStore;
    use crate::terrain::gen::gen_region;

    fn test_lattice() -> LatticeField {
        LatticeField::for_world(6, 42)
    }

    #[test]
    fn outside_bounds_is_air() {
        let f = test_lattice();
        assert_eq!(f.sample([0, 0, 0]), AIR, "below lo must be AIR");
        assert_eq!(f.sample([63, 63, 63]), AIR, "above hi must be AIR");
    }

    #[test]
    fn pillars_are_stone() {
        let f = test_lattice();
        // First cell interior starts at lo. cx=0, cz=0 → pillar zone.
        let lo = f.lo;
        assert_eq!(
            f.sample(lo),
            STONE,
            "corner of lattice (pillar zone) must be STONE"
        );
    }

    #[test]
    fn has_active_materials() {
        let f = test_lattice();
        let mut store = NodeStore::new();
        let (root, _stats) = gen_region(&mut store, &f, [0, 0, 0], 6);
        let grid = store.flatten(root, 64);
        let has = |mat: CellState| grid.contains(&mat);
        assert!(has(STONE), "lattice must have STONE");
        assert!(has(GRASS), "lattice must have GRASS");
        assert!(has(WATER), "lattice must have WATER");
        assert!(has(FIRE), "lattice must have FIRE");
    }

    #[test]
    fn classify_box_outside_is_air() {
        let f = test_lattice();
        // Box entirely below lo on the y axis.
        assert_eq!(f.classify_box([0, 0, 0], 1), Some(AIR));
    }

    #[test]
    fn classify_box_conservative() {
        let f = test_lattice();
        let mut store = NodeStore::new();
        let (_root, _) = gen_region(&mut store, &f, [0, 0, 0], 6);

        // Check a grid of level-2 boxes (4x4x4) for conservativeness.
        let side = 64usize;
        for oz in (0..side).step_by(4) {
            for oy in (0..side).step_by(4) {
                for ox in (0..side).step_by(4) {
                    if let Some(state) =
                        f.classify_box([ox as i64, oy as i64, oz as i64], 2)
                    {
                        for z in oz..oz + 4 {
                            for y in oy..oy + 4 {
                                for x in ox..ox + 4 {
                                    let s = f.sample([x as i64, y as i64, z as i64]);
                                    assert_eq!(
                                        s, state,
                                        "classify_box said {:?} but sample({},{},{}) = {:?}",
                                        state, x, y, z, s,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn flatten_matches_sample() {
        let f = test_lattice();
        let mut store = NodeStore::new();
        let (root, _) = gen_region(&mut store, &f, [0, 0, 0], 6);
        let grid = store.flatten(root, 64);
        for z in 0..64 {
            for y in 0..64 {
                for x in 0..64 {
                    let expected = f.sample([x as i64, y as i64, z as i64]);
                    let got = grid[x + y * 64 + z * 64 * 64];
                    assert_eq!(
                        got, expected,
                        "mismatch at ({x},{y},{z}): expected {expected}, got {got}",
                    );
                }
            }
        }
    }

    #[test]
    fn collapses_fire() {
        let f = test_lattice();
        let mut store = NodeStore::new();
        let (_, stats) = gen_region(&mut store, &f, [0, 0, 0], 6);
        assert!(
            stats.total_collapses() > 0,
            "lattice field must have proof-based collapses: {stats:?}",
        );
        // Most of the volume is AIR margin or corridor — should have many collapses.
        assert!(
            stats.total_collapses() > 10,
            "expected substantial collapses, got {}: {stats:?}",
            stats.total_collapses(),
        );
    }

    #[test]
    fn deterministic() {
        let f = test_lattice();
        let mut s1 = NodeStore::new();
        let mut s2 = NodeStore::new();
        let (r1, _) = gen_region(&mut s1, &f, [0, 0, 0], 6);
        let (r2, _) = gen_region(&mut s2, &f, [0, 0, 0], 6);
        assert_eq!(r1, r2, "lattice gen must be deterministic");
    }
}
