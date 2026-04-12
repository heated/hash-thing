//! `HeightmapField` — the v1 terrain content. Surface height is a value-noise
//! fractal in `[base_y - amplitude, base_y + amplitude]`. Material is chosen
//! by depth below the surface (see `terrain::materials::material_from_depth`).
//!
//! `classify_box` is **proof-based only**: a y-band check that uses the
//! global noise bounds (not sampled corners) to declare entire boxes above
//! `surface_max + SURFACE_MARGIN` as `AIR`, and entire boxes below
//! `surface_min - DEPTH_MARGIN` as `STONE`. No corner-agreement heuristic.
//!
//! `PrecomputedHeightmapField` wraps a `HeightmapField` with a precomputed
//! 2D height grid + min/max mipmap. `classify_box` uses exact local surface
//! bounds instead of global amplitude bounds — still proof-based, much tighter.

use super::RegionField;
use crate::octree::CellState;
use crate::terrain::materials::{material_from_depth, AIR, SAND, STONE, WATER};
use crate::terrain::noise::{biome_2d, fractal_2d};

/// Cells above the maximum possible surface before we trust the AIR
/// short-circuit. The margin absorbs a few ulps of FP rounding in
/// `fractal_2d`'s final `sum / total` divide — which, under adversarial
/// input, could push the computed `surface_y` microscopically above the
/// ideal `base_y + amplitude` bound. (At 64³ the `i64 → f32` casts on the
/// box coordinates themselves are exact, so that's not where the slop
/// comes from.) 2 cells of headroom is vastly more than needed and keeps
/// the proof robust against future noise-function changes.
pub const SURFACE_MARGIN: f32 = 2.0;

/// Cells below the minimum possible surface before we trust the STONE
/// short-circuit. Pinned to the dirt-band rule (`d < 4 → DIRT`) so the
/// deep-stone short-circuit is correct by construction: at `depth >= 4`,
/// `material_from_depth` returns `STONE`.
pub const DEPTH_MARGIN: f32 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct HeightmapField {
    pub seed: u64,
    pub base_y: f32,
    pub amplitude: f32,
    pub wavelength: f32,
    pub octaves: u32,
    /// Sea level: air cells below this y become water. `None` disables.
    pub sea_level: Option<f32>,
}

impl HeightmapField {
    /// Surface y at world `(x, z)`. Always in
    /// `[base_y - amplitude, base_y + amplitude]`.
    #[inline]
    pub fn surface_y(&self, x: f32, z: f32) -> f32 {
        let n = fractal_2d(
            x / self.wavelength,
            z / self.wavelength,
            self.seed,
            self.octaves,
        );
        // n ∈ [0, 1] (debug_assert in fractal_2d). Map to [-1, 1].
        let centered = n * 2.0 - 1.0;
        self.base_y + centered * self.amplitude
    }
}

/// Biome noise threshold: below this value → sandy biome.
const SAND_BIOME_THRESHOLD: f32 = 0.3;

/// Wavelength for biome noise (cells). Large value → big biome regions.
const BIOME_WAVELENGTH: f32 = 64.0;

impl RegionField for HeightmapField {
    fn sample(&self, p: [i64; 3]) -> CellState {
        let surface = self.surface_y(p[0] as f32, p[2] as f32);
        let depth = surface - p[1] as f32;
        let base = material_from_depth(depth);
        // Sea level: fill air below sea_level with water.
        if base == AIR {
            if let Some(sl) = self.sea_level {
                if (p[1] as f32) < sl {
                    return WATER;
                }
            }
            return AIR;
        }
        // In sandy biomes, replace grass and dirt with sand so the terrain
        // has natural sand regions that fall and settle under gravity.
        if base != STONE && depth < 4.0 {
            let biome = biome_2d(
                p[0] as f32 / BIOME_WAVELENGTH,
                p[2] as f32 / BIOME_WAVELENGTH,
                self.seed,
            );
            if biome < SAND_BIOME_THRESHOLD {
                return SAND;
            }
        }
        base
    }

    fn classify_box(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;
        let y_min = origin[1] as f32; // inclusive
        let y_max = (origin[1] + size) as f32; // exclusive

        // Conservative global heightmap bounds — NOT sampled.
        let surface_max = self.base_y + self.amplitude;
        let surface_min = self.base_y - self.amplitude;

        // Box fully above the highest possible surface (plus a margin):
        // every cell is at depth < 0. If also above sea level, all AIR;
        // if also fully below sea level, all WATER. Otherwise can't collapse.
        if y_min >= surface_max + SURFACE_MARGIN {
            if let Some(sl) = self.sea_level {
                if y_max <= sl {
                    return Some(WATER);
                }
                if y_min < sl {
                    return None; // straddles sea level
                }
            }
            return Some(AIR);
        }
        // Box fully below the lowest possible surface, with at least
        // `DEPTH_MARGIN` cells of clearance: every cell has depth >= 4,
        // hence STONE per `material_from_depth`.
        // y_max - 1 is the largest y in the box, so we need
        //   surface_min - (y_max - 1) >= DEPTH_MARGIN
        //   <=>  y_max <= surface_min - DEPTH_MARGIN + 1
        // We use the slightly looser y_max <= surface_min - DEPTH_MARGIN
        // to keep the comparison exclusive-friendly and conservative.
        if y_max <= surface_min - DEPTH_MARGIN {
            return Some(STONE);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// PrecomputedHeightmapField — mipmap-accelerated classify_box
// ---------------------------------------------------------------------------

/// Precomputed 2D surface heights + min/max mipmap for fast `classify_box`.
///
/// The mipmap stores min and max surface_y for every power-of-2-aligned XZ
/// region. At octree level k, the builder's XZ projection is always a
/// 2^k × 2^k square aligned to 2^k boundaries, which maps to exactly one
/// mipmap cell at level k.
pub struct PrecomputedHeightmapField {
    inner: HeightmapField,
    side: usize,
    /// Min surface height per mipmap level. `mip_min[k]` has
    /// `(side >> k)²` entries covering 2^k × 2^k blocks.
    /// Level 0 is the raw heights at leaf resolution.
    mip_min: Vec<Vec<f32>>,
    /// Max surface height per mipmap level.
    mip_max: Vec<Vec<f32>>,
}

impl PrecomputedHeightmapField {
    /// Precompute heights and build the min/max mipmap.
    /// `side_log2` is the world level (e.g. 9 for 512³).
    pub fn new(field: HeightmapField, side_log2: u32) -> Self {
        let side = 1usize << side_log2;
        let n = side * side;

        // Evaluate surface_y at every (x, z) in [0, side).
        let mut heights = vec![0.0f32; n];
        for z in 0..side {
            for x in 0..side {
                heights[z * side + x] = field.surface_y(x as f32, z as f32);
            }
        }

        // Build mipmap. Level 0 = raw heights; higher levels are min/max
        // reductions of 2×2 blocks from the previous level.
        let levels = side_log2 as usize + 1;
        let mut mip_min = Vec::with_capacity(levels);
        let mut mip_max = Vec::with_capacity(levels);

        // Level 0: identity (each cell is its own min/max).
        mip_min.push(heights.clone());
        mip_max.push(heights.clone());

        for k in 1..levels {
            let prev_side = side >> (k - 1);
            let cur_side = side >> k;
            let prev_min = &mip_min[k - 1];
            let prev_max = &mip_max[k - 1];
            let mut cur_min = vec![f32::INFINITY; cur_side * cur_side];
            let mut cur_max = vec![f32::NEG_INFINITY; cur_side * cur_side];

            for cz in 0..cur_side {
                for cx in 0..cur_side {
                    let px = cx * 2;
                    let pz = cz * 2;
                    let i00 = pz * prev_side + px;
                    let i10 = pz * prev_side + px + 1;
                    let i01 = (pz + 1) * prev_side + px;
                    let i11 = (pz + 1) * prev_side + px + 1;

                    let mn = prev_min[i00]
                        .min(prev_min[i10])
                        .min(prev_min[i01])
                        .min(prev_min[i11]);
                    let mx = prev_max[i00]
                        .max(prev_max[i10])
                        .max(prev_max[i01])
                        .max(prev_max[i11]);

                    cur_min[cz * cur_side + cx] = mn;
                    cur_max[cz * cur_side + cx] = mx;
                }
            }

            mip_min.push(cur_min);
            mip_max.push(cur_max);
        }

        Self {
            inner: field,
            side,
            mip_min,
            mip_max,
        }
    }

    /// Look up local min/max surface height for the XZ projection of a box
    /// at `origin` with size `2^size_log2`. Returns `(min_h, max_h)`.
    #[inline]
    fn local_bounds(&self, origin: [i64; 3], size_log2: u32) -> (f32, f32) {
        let k = size_log2 as usize;
        let mip_side = self.side >> k;
        let mx = (origin[0] as usize) >> k;
        let mz = (origin[2] as usize) >> k;
        debug_assert!(mx < mip_side && mz < mip_side);
        let idx = mz * mip_side + mx;
        (self.mip_min[k][idx], self.mip_max[k][idx])
    }

    /// Look up the precomputed surface height at `(x, z)`. Falls back to
    /// live evaluation if the coordinate is out of the precomputed range.
    #[inline]
    fn precomputed_surface_y(&self, x: i64, z: i64) -> f32 {
        let ux = x as usize;
        let uz = z as usize;
        if ux < self.side && uz < self.side {
            self.mip_min[0][uz * self.side + ux]
        } else {
            self.inner.surface_y(x as f32, z as f32)
        }
    }
}

impl RegionField for PrecomputedHeightmapField {
    #[inline]
    fn sample(&self, p: [i64; 3]) -> CellState {
        let surface = self.precomputed_surface_y(p[0], p[2]);
        let depth = surface - p[1] as f32;
        let base = material_from_depth(depth);
        if base == AIR {
            if let Some(sl) = self.inner.sea_level {
                if (p[1] as f32) < sl {
                    return WATER;
                }
            }
            return AIR;
        }
        if base != STONE && depth < 4.0 {
            let biome = biome_2d(
                p[0] as f32 / BIOME_WAVELENGTH,
                p[2] as f32 / BIOME_WAVELENGTH,
                self.inner.seed,
            );
            if biome < SAND_BIOME_THRESHOLD {
                return SAND;
            }
        }
        base
    }

    fn classify_box(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;
        let y_min = origin[1] as f32;
        let y_max = (origin[1] + size) as f32;

        let (surface_min, surface_max) = self.local_bounds(origin, size_log2);

        // Box fully above the highest local surface (plus margin).
        if y_min >= surface_max + SURFACE_MARGIN {
            if let Some(sl) = self.inner.sea_level {
                if y_max <= sl {
                    return Some(WATER);
                }
                if y_min < sl {
                    return None;
                }
            }
            return Some(AIR);
        }
        // Box fully below the lowest local surface (with depth margin).
        if y_max <= surface_min - DEPTH_MARGIN {
            return Some(STONE);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic field with known bounds for testing classify_box.
    /// base_y=32, amplitude=8 → surface ∈ [24, 40].
    fn test_field() -> HeightmapField {
        HeightmapField {
            seed: 42,
            base_y: 32.0,
            amplitude: 8.0,
            wavelength: 16.0,
            octaves: 2,
            sea_level: None,
        }
    }

    #[test]
    fn sky_box_returns_air() {
        let f = test_field();
        // surface_max = 32 + 8 = 40, SURFACE_MARGIN = 2 → threshold = 42.
        // Box at y_min=42: entirely above maximum surface.
        assert_eq!(
            f.classify_box([0, 42, 0], 2),
            Some(AIR),
            "box above surface_max + margin must be AIR",
        );
    }

    #[test]
    fn sky_box_just_below_threshold_returns_none() {
        let f = test_field();
        // Box at y_min=41, size=4 (size_log2=2) → y in [41, 45).
        // Threshold is 42 (surface_max + margin). y_min=41 < 42 → not provably AIR.
        assert_eq!(
            f.classify_box([0, 41, 0], 2),
            None,
            "box with y_min below sky threshold must not short-circuit",
        );
    }

    #[test]
    fn deep_stone_box_returns_stone() {
        let f = test_field();
        // surface_min = 32 - 8 = 24, DEPTH_MARGIN = 4 → threshold = 20.
        // Box y_max must be ≤ 20. With size_log2=2 (size=4): origin y=16 → y_max=20.
        assert_eq!(
            f.classify_box([0, 16, 0], 2),
            Some(STONE),
            "box fully below surface_min - DEPTH_MARGIN must be STONE",
        );
    }

    #[test]
    fn deep_stone_box_just_above_threshold_returns_none() {
        let f = test_field();
        // origin y=17, size=4 → y_max=21 > 20 → not provably all STONE.
        assert_eq!(
            f.classify_box([0, 17, 0], 2),
            None,
            "box with y_max above deep-stone threshold must not short-circuit",
        );
    }

    #[test]
    fn surface_straddling_box_returns_none() {
        let f = test_field();
        // Box around y=32 (base_y) — clearly straddles the surface.
        assert_eq!(
            f.classify_box([0, 30, 0], 3),
            None,
            "box straddling the surface band must return None",
        );
    }

    #[test]
    fn large_sky_box_collapses() {
        let f = test_field();
        // size_log2=5 (size=32), origin y=48 → y_min=48 ≥ 42 → AIR.
        assert_eq!(
            f.classify_box([0, 48, 0], 5),
            Some(AIR),
            "large box well above surface must collapse to AIR",
        );
    }

    #[test]
    fn large_deep_box_collapses() {
        let f = test_field();
        // size_log2=3 (size=8), origin y=0 → y_max=8 ≤ 20 → STONE.
        assert_eq!(
            f.classify_box([0, 0, 0], 3),
            Some(STONE),
            "large box well below surface must collapse to STONE",
        );
    }

    #[test]
    fn sample_agrees_with_surface_y() {
        let f = test_field();
        // A point well above the surface should be AIR.
        let high = f.sample([10, 50, 10]);
        assert_eq!(high, AIR, "point at y=50 must be AIR");
        // A point well below should be STONE.
        let deep = f.sample([10, 0, 10]);
        assert_eq!(deep, STONE, "point at y=0 must be STONE");
    }

    #[test]
    fn surface_y_is_deterministic_for_same_input() {
        let f = test_field();
        let samples = [(0.0, 0.0), (3.5, -9.25), (128.0, 64.0), (-17.0, 41.0)];

        for (x, z) in samples {
            let first = f.surface_y(x, z).to_bits();
            let second = f.surface_y(x, z).to_bits();
            assert_eq!(
                first, second,
                "surface_y must be bitwise deterministic for ({x}, {z})",
            );
        }
    }

    #[test]
    fn surface_y_stays_within_configured_bounds() {
        let f = test_field();
        let min = f.base_y - f.amplitude;
        let max = f.base_y + f.amplitude;
        let samples = [
            (-256.0, -256.0),
            (-32.5, 48.25),
            (0.0, 0.0),
            (17.0, -9.0),
            (64.0, 64.0),
            (512.0, -128.0),
        ];

        for (x, z) in samples {
            let surface = f.surface_y(x, z);
            assert!(
                surface >= min && surface <= max,
                "surface_y({x}, {z}) = {surface} fell outside [{min}, {max}]",
            );
        }
    }

    #[test]
    fn biome_produces_sand_in_some_surface_cells() {
        let f = test_field();
        let mut sand_count = 0u32;
        let mut surface_count = 0u32;
        // Scan a 128×128 grid at the surface level.
        for x in 0..128 {
            for z in 0..128 {
                let surface = f.surface_y(x as f32, z as f32);
                let y = surface.floor() as i64;
                let cell = f.sample([x, y, z]);
                if cell != AIR {
                    surface_count += 1;
                    if cell == SAND {
                        sand_count += 1;
                    }
                }
            }
        }
        assert!(
            sand_count > 0,
            "expected some sand in 128×128 terrain grid, got 0 out of {surface_count} surface cells"
        );
        assert!(
            sand_count < surface_count,
            "expected mixed biomes, not all sand ({sand_count}/{surface_count})"
        );
    }

    #[test]
    fn sea_level_fills_low_air_with_water() {
        let mut f = test_field();
        // Surface range [24, 40]. Set sea level above surface_max so that
        // air cells above the surface but below sea_level become water.
        f.sea_level = Some(45.0);
        // A point well above sea level → AIR
        assert_eq!(f.sample([10, 50, 10]), AIR);
        // A point above the surface (y=41 > surface_max=40) but below sea level (45) → WATER
        assert_eq!(f.sample([10, 41, 10]), WATER);
        // A point deep underground → still STONE
        assert_eq!(f.sample([10, 0, 10]), STONE);
    }

    #[test]
    fn classify_box_water_above_surface_below_sea() {
        let mut f = test_field();
        f.sea_level = Some(28.0);
        // Box fully above surface_max (42) and above sea level → AIR
        assert_eq!(f.classify_box([0, 42, 0], 2), Some(AIR));
        // Box fully above surface_max but fully below sea level:
        // surface_max + margin = 42. If we had sea_level = 50 (above the box),
        // and the box is above the surface, it should be WATER.
        let mut f2 = test_field();
        f2.sea_level = Some(50.0);
        assert_eq!(f2.classify_box([0, 42, 0], 2), Some(WATER));
    }

    // -----------------------------------------------------------------
    // PrecomputedHeightmapField — correctness and performance tests.
    // -----------------------------------------------------------------

    #[test]
    fn precomputed_produces_identical_terrain() {
        use super::super::super::gen::gen_region;
        use crate::octree::NodeStore;

        let field = test_field();
        let precomputed = PrecomputedHeightmapField::new(field, 6);

        let mut store1 = NodeStore::new();
        let mut store2 = NodeStore::new();
        let (root1, _) = gen_region(&mut store1, &field, [0, 0, 0], 6);
        let (root2, stats2) = gen_region(&mut store2, &precomputed, [0, 0, 0], 6);

        // Both must produce identical flattened grids.
        let grid1 = store1.flatten(root1, 64);
        let grid2 = store2.flatten(root2, 64);
        assert_eq!(
            grid1, grid2,
            "precomputed field must produce identical terrain"
        );

        // Precomputed should have MORE collapses (tighter bounds).
        let (_, stats1) = gen_region(&mut NodeStore::new(), &field, [0, 0, 0], 6);
        assert!(
            stats2.total_collapses() >= stats1.total_collapses(),
            "precomputed should collapse at least as many boxes: \
             precomputed={}, original={}",
            stats2.total_collapses(),
            stats1.total_collapses(),
        );
    }

    #[test]
    fn precomputed_classify_box_is_conservative() {
        let field = test_field();
        let precomputed = PrecomputedHeightmapField::new(field, 6);

        // Check a grid of boxes at level 3 (8×8×8).
        for oz in (0..64).step_by(8) {
            for oy in (0..64).step_by(8) {
                for ox in (0..64).step_by(8) {
                    if let Some(state) =
                        precomputed.classify_box([ox as i64, oy as i64, oz as i64], 3)
                    {
                        // Verify every cell in this box matches.
                        for z in oz..oz + 8 {
                            for y in oy..oy + 8 {
                                for x in ox..ox + 8 {
                                    let s = precomputed.sample([x as i64, y as i64, z as i64]);
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
    fn precomputed_more_collapses_than_global() {
        use super::super::super::gen::gen_region;
        use crate::octree::NodeStore;

        let field = test_field();
        let precomputed = PrecomputedHeightmapField::new(field, 6);

        let (_, stats_global) = gen_region(&mut NodeStore::new(), &field, [0, 0, 0], 6);
        let (_, stats_local) = gen_region(&mut NodeStore::new(), &precomputed, [0, 0, 0], 6);

        eprintln!(
            "collapses: global={}, local={}, leaves: global={}, local={}",
            stats_global.total_collapses(),
            stats_local.total_collapses(),
            stats_global.leaves,
            stats_local.leaves,
        );

        // Strictly more collapses with local bounds.
        assert!(
            stats_local.total_collapses() > stats_global.total_collapses(),
            "expected more collapses with precomputed local bounds",
        );
        // Fewer leaf noise evaluations.
        assert!(
            stats_local.leaves < stats_global.leaves,
            "expected fewer leaf samples with precomputed local bounds: \
             local={}, global={}",
            stats_local.leaves,
            stats_global.leaves,
        );
    }
}
