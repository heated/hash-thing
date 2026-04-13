//! `TerrainBlendField` composes a structure with terrain and tapers the
//! structure shell into the terrain near the structure-zone boundary.
//!
//! Outside the structure zone, the wrapped terrain is returned directly.
//! Inside the zone, the structure wins so interior voids stay carved. Only
//! solid structure cells near the outer x/z boundary are softened.

use super::WorldGen;
use crate::octree::CellState;
use crate::terrain::materials::{AIR, WATER};

/// Terrain fields that can expose an approximate surface height for a given
/// world column. Used by seam wrappers that need a cheap "how far above the
/// terrain are we?" test.
pub trait TerrainSurface {
    fn surface_y_at(&self, x: i64, z: i64) -> f32;
}

/// Wraps a structure field and terrain, returning terrain outside the
/// structure zone and the structure inside it, with a seam treatment near the
/// outer x/z boundary.
///
/// Behavior inside the boundary band:
/// - structure cells far above the terrain surface are carved away
/// - shallow structure cells inherit the terrain's surface material
/// - everything else stays as the original structure material
#[derive(Clone, Debug)]
pub struct TerrainBlendField<S, T> {
    pub structure: S,
    pub terrain: T,
    /// Active structure zone. The blend is applied only near the x/z faces
    /// of this box. Y is carried for caller symmetry and future extension.
    pub region_lo: [i64; 3],
    pub region_hi: [i64; 3],
    /// Width of the boundary band in cells. Zero disables the blend.
    pub edge_blend: i64,
    /// Maximum amount the structure may protrude above the terrain surface at
    /// the inner edge of the boundary band.
    pub max_surface_protrusion: f32,
    /// Depth below the terrain surface that should inherit terrain material.
    pub cover_depth: f32,
}

impl<S, T> TerrainBlendField<S, T> {
    pub fn new(
        structure: S,
        terrain: T,
        region_lo: [i64; 3],
        region_hi: [i64; 3],
        edge_blend: i64,
        max_surface_protrusion: f32,
        cover_depth: f32,
    ) -> Self {
        Self {
            structure,
            terrain,
            region_lo,
            region_hi,
            edge_blend,
            max_surface_protrusion,
            cover_depth,
        }
    }

    #[inline]
    fn point_edge_distance_xz(&self, point: [i64; 3]) -> Option<i64> {
        if point[0] < self.region_lo[0]
            || point[0] >= self.region_hi[0]
            || point[2] < self.region_lo[2]
            || point[2] >= self.region_hi[2]
        {
            return None;
        }
        Some(
            (point[0] - self.region_lo[0])
                .min(self.region_hi[0] - 1 - point[0])
                .min(point[2] - self.region_lo[2])
                .min(self.region_hi[2] - 1 - point[2]),
        )
    }

    #[inline]
    fn point_in_region(&self, point: [i64; 3]) -> bool {
        point[0] >= self.region_lo[0]
            && point[0] < self.region_hi[0]
            && point[1] >= self.region_lo[1]
            && point[1] < self.region_hi[1]
            && point[2] >= self.region_lo[2]
            && point[2] < self.region_hi[2]
    }

    #[inline]
    fn box_outside_region(&self, origin: [i64; 3], size: i64) -> bool {
        let hi = [origin[0] + size, origin[1] + size, origin[2] + size];
        hi[0] <= self.region_lo[0]
            || origin[0] >= self.region_hi[0]
            || hi[1] <= self.region_lo[1]
            || origin[1] >= self.region_hi[1]
            || hi[2] <= self.region_lo[2]
            || origin[2] >= self.region_hi[2]
    }

    #[inline]
    fn box_inside_region(&self, origin: [i64; 3], size: i64) -> bool {
        let hi = [origin[0] + size, origin[1] + size, origin[2] + size];
        origin[0] >= self.region_lo[0]
            && hi[0] <= self.region_hi[0]
            && origin[1] >= self.region_lo[1]
            && hi[1] <= self.region_hi[1]
            && origin[2] >= self.region_lo[2]
            && hi[2] <= self.region_hi[2]
    }

    #[inline]
    fn box_touches_blend_band(&self, origin: [i64; 3], size: i64) -> bool {
        if self.edge_blend <= 0 {
            return false;
        }

        let bx_lo = origin[0];
        let bx_hi = origin[0] + size;
        let bz_lo = origin[2];
        let bz_hi = origin[2] + size;

        if bx_hi <= self.region_lo[0]
            || bx_lo >= self.region_hi[0]
            || bz_hi <= self.region_lo[2]
            || bz_lo >= self.region_hi[2]
        {
            return false;
        }

        let core_lo_x = self.region_lo[0] + self.edge_blend;
        let core_hi_x = self.region_hi[0] - self.edge_blend;
        let core_lo_z = self.region_lo[2] + self.edge_blend;
        let core_hi_z = self.region_hi[2] - self.edge_blend;

        if core_lo_x >= core_hi_x || core_lo_z >= core_hi_z {
            return true;
        }

        !(bx_lo >= core_lo_x && bx_hi <= core_hi_x && bz_lo >= core_lo_z && bz_hi <= core_hi_z)
    }
}

impl<S, T> WorldGen for TerrainBlendField<S, T>
where
    S: WorldGen,
    T: WorldGen + TerrainSurface,
{
    fn sample(&self, point: [i64; 3]) -> CellState {
        if !self.point_in_region(point) {
            return self.terrain.sample(point);
        }

        let structure_state = self.structure.sample(point);
        let Some(edge_distance) = self.point_edge_distance_xz(point) else {
            return structure_state;
        };
        if self.edge_blend <= 0 || edge_distance >= self.edge_blend || structure_state == AIR {
            return structure_state;
        }

        let terrain_surface = self.terrain.surface_y_at(point[0], point[2]);
        let protrusion_budget =
            self.max_surface_protrusion * (edge_distance as f32 / self.edge_blend as f32);
        if point[1] as f32 > terrain_surface + protrusion_budget {
            return AIR;
        }

        let terrain_state = self.terrain.sample(point);
        let terrain_depth = terrain_surface - point[1] as f32;
        if terrain_state != AIR
            && terrain_state != WATER
            && terrain_depth >= 0.0
            && terrain_depth <= self.cover_depth
        {
            return terrain_state;
        }

        structure_state
    }

    fn classify(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
        let size = 1i64 << size_log2;

        if self.box_outside_region(origin, size) {
            return self.terrain.classify(origin, size_log2);
        }
        if !self.box_inside_region(origin, size) {
            return None;
        }

        let structure_state = self.structure.classify(origin, size_log2)?;
        if !self.box_touches_blend_band(origin, size) {
            return Some(structure_state);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::materials::{GRASS, STONE};

    #[derive(Clone, Copy, Debug)]
    struct SolidRegion {
        lo: [i64; 3],
        hi: [i64; 3],
        material: CellState,
    }

    impl WorldGen for SolidRegion {
        fn sample(&self, point: [i64; 3]) -> CellState {
            if point[0] >= self.lo[0]
                && point[0] < self.hi[0]
                && point[1] >= self.lo[1]
                && point[1] < self.hi[1]
                && point[2] >= self.lo[2]
                && point[2] < self.hi[2]
            {
                self.material
            } else {
                AIR
            }
        }

        fn classify(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
            let size = 1i64 << size_log2;
            let hi = [origin[0] + size, origin[1] + size, origin[2] + size];
            if hi[0] <= self.lo[0]
                || origin[0] >= self.hi[0]
                || hi[1] <= self.lo[1]
                || origin[1] >= self.hi[1]
                || hi[2] <= self.lo[2]
                || origin[2] >= self.hi[2]
            {
                return Some(AIR);
            }
            if origin[0] >= self.lo[0]
                && hi[0] <= self.hi[0]
                && origin[1] >= self.lo[1]
                && hi[1] <= self.hi[1]
                && origin[2] >= self.lo[2]
                && hi[2] <= self.hi[2]
            {
                return Some(self.material);
            }
            None
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct FlatTerrain {
        surface_y: i64,
    }

    impl TerrainSurface for FlatTerrain {
        fn surface_y_at(&self, _x: i64, _z: i64) -> f32 {
            self.surface_y as f32
        }
    }

    impl WorldGen for FlatTerrain {
        fn sample(&self, point: [i64; 3]) -> CellState {
            if point[1] > self.surface_y {
                AIR
            } else if point[1] == self.surface_y {
                GRASS
            } else {
                STONE
            }
        }

        fn classify(&self, origin: [i64; 3], size_log2: u32) -> Option<CellState> {
            let size = 1i64 << size_log2;
            let y_min = origin[1];
            let y_max = origin[1] + size;
            if y_min > self.surface_y {
                Some(AIR)
            } else if y_max <= self.surface_y {
                Some(STONE)
            } else {
                None
            }
        }
    }

    fn test_field() -> TerrainBlendField<SolidRegion, FlatTerrain> {
        TerrainBlendField::new(
            SolidRegion {
                lo: [10, 0, 10],
                hi: [30, 48, 30],
                material: STONE,
            },
            FlatTerrain { surface_y: 24 },
            [10, 0, 10],
            [30, 48, 30],
            6,
            8.0,
            2.0,
        )
    }

    #[test]
    fn keeps_structure_in_the_core() {
        let field = test_field();
        assert_eq!(field.sample([20, 36, 20]), STONE);
    }

    #[test]
    fn delegates_to_terrain_outside_the_structure_zone() {
        let field = test_field();
        assert_eq!(field.sample([5, 24, 20]), GRASS);
    }

    #[test]
    fn carves_high_structure_at_the_outer_edge() {
        let field = test_field();
        assert_eq!(field.sample([10, 36, 20]), AIR);
    }

    #[test]
    fn applies_terrain_cover_near_the_edge() {
        let field = test_field();
        assert_eq!(field.sample([10, 24, 20]), GRASS);
    }

    #[test]
    fn classify_delegates_outside_the_blend_band() {
        let field = test_field();
        assert_eq!(field.classify([16, 8, 16], 2), Some(STONE));
    }

    #[test]
    fn classify_delegates_to_terrain_outside_the_structure_zone() {
        let field = test_field();
        assert_eq!(field.classify([0, 20, 16], 1), Some(STONE));
    }

    #[test]
    fn classify_stays_conservative_in_the_blend_band() {
        let field = test_field();
        assert_eq!(field.classify([10, 8, 16], 2), None);
    }
}
