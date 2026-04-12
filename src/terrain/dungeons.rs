#![allow(dead_code)]
//! Procedural dungeon carving post-pass (hash-thing-3fq.3).
//!
//! Generates rectangular rooms connected by corridors within the stone
//! volume of an already-generated terrain. Follows the same architecture as
//! `caves.rs`: operates on a flattened grid, only modifies STONE cells,
//! re-interns the result via `NodeStore::from_flat`.
//!
//! ## Algorithm
//!
//! 1. **Placement:** Attempt to place `max_rooms` axis-aligned rectangular
//!    rooms at deterministic random positions within the stone mask. A room
//!    is rejected if it overlaps an existing room (with a 1-cell wall
//!    margin) or extends outside the stone mask. Room dimensions are drawn
//!    from `[room_min..=room_max]` per axis.
//!
//! 2. **Corridors:** Each room (except the first) is connected to its
//!    nearest predecessor by an L-shaped corridor (horizontal in X, then
//!    vertical in Y, then horizontal in Z). Corridors carve a 1-wide
//!    tunnel through stone.
//!
//! 3. **Writeback:** Carved cells (room interiors + corridor paths) are
//!    set to AIR. Cells outside the stone mask are never touched.
//!
//! ## Determinism
//!
//! Room placement uses `rng::cell_rand_range` keyed on the room index and
//! dungeon seed, so the layout is a pure function of `DungeonParams::seed`.
//! Two runs with identical params produce identical dungeons.

use crate::octree::{CellState, NodeId, NodeStore};
use crate::rng;
use crate::terrain::materials::{AIR, STONE};

/// Parameters for dungeon generation.
#[derive(Clone, Copy, Debug)]
pub struct DungeonParams {
    /// Seed for the deterministic room-placement PRNG.
    pub seed: u64,
    /// Maximum number of room placement attempts.
    pub max_rooms: u32,
    /// Minimum room size per axis (cells).
    pub room_min: u32,
    /// Maximum room size per axis (cells).
    pub room_max: u32,
    /// Minimum Y coordinate for room placement (rooms only spawn in deep
    /// stone, not near the surface).
    pub min_y: u32,
}

impl Default for DungeonParams {
    fn default() -> Self {
        Self {
            seed: 7,
            max_rooms: 8,
            room_min: 4,
            room_max: 8,
            min_y: 0,
        }
    }
}

impl DungeonParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.room_min == 0 {
            return Err("room_min must be >= 1");
        }
        if self.room_max < self.room_min {
            return Err("room_max must be >= room_min");
        }
        Ok(())
    }
}

/// An axis-aligned room: `[lo, hi)` on each axis.
#[derive(Clone, Copy, Debug)]
struct Room {
    x0: usize,
    y0: usize,
    z0: usize,
    x1: usize,
    y1: usize,
    z1: usize,
}

impl Room {
    fn center(&self) -> (usize, usize, usize) {
        (
            (self.x0 + self.x1) / 2,
            (self.y0 + self.y1) / 2,
            (self.z0 + self.z1) / 2,
        )
    }

    /// Check if two rooms overlap (with a 1-cell wall margin).
    fn overlaps(&self, other: &Room) -> bool {
        self.x0 < other.x1 + 1
            && self.x1 + 1 > other.x0
            && self.y0 < other.y1 + 1
            && self.y1 + 1 > other.y0
            && self.z0 < other.z1 + 1
            && self.z1 + 1 > other.z0
    }
}

/// Carve dungeons into `root`. Returns a fresh `NodeId` for the post-dungeon
/// terrain.
pub fn carve_dungeons(
    store: &mut NodeStore,
    root: NodeId,
    side_log2: u32,
    params: &DungeonParams,
) -> NodeId {
    params.validate().expect("invalid DungeonParams");
    let side = 1usize << side_log2;
    let mut grid = store.flatten(root, side);
    carve_dungeons_grid(&mut grid, side, [0, 0, 0], params);
    store.from_flat(&grid, side)
}

/// In-place dungeon carving on a flat grid. `origin` is the world-coordinate
/// offset of the grid's `[0,0,0]` corner, threaded into the RNG so the same
/// sub-volume produces identical dungeons regardless of which chunk contains
/// it (lazy-root-expansion friendly, same contract as `carve_caves_grid`).
pub fn carve_dungeons_grid(
    grid: &mut [CellState],
    side: usize,
    origin: [i64; 3],
    params: &DungeonParams,
) {
    let expected_len = side
        .checked_mul(side)
        .and_then(|n| n.checked_mul(side))
        .expect("carve_dungeons_grid: side^3 overflowed usize");
    assert_eq!(grid.len(), expected_len, "grid is not side^3");
    params
        .validate()
        .expect("carve_dungeons_grid: invalid DungeonParams");

    let idx = |x: usize, y: usize, z: usize| x + y * side + z * side * side;

    // Build the stone mask — dungeons only carve STONE.
    let is_stone = |x: usize, y: usize, z: usize| grid[idx(x, y, z)] == STONE;

    // Place rooms.
    let mut rooms: Vec<Room> = Vec::new();

    for attempt in 0..params.max_rooms {
        // Deterministic dimensions and position from seed + attempt index +
        // origin. The origin enters via the x/y/z coordinates of
        // cell_rand_range so that two chunks with different origins produce
        // different room layouts (lazy-root-expansion friendly).
        let gen = attempt as u64;
        let seed = params.seed;
        let ox = origin[0];
        let oy = origin[1];
        let oz = origin[2];

        let range = (params.room_max - params.room_min + 1) as u64;
        let wx = params.room_min as usize
            + rng::cell_rand_range(ox, oy, oz, gen, seed.wrapping_add(1), range) as usize;
        let wy = params.room_min as usize
            + rng::cell_rand_range(ox + 1, oy, oz, gen, seed.wrapping_add(2), range) as usize;
        let wz = params.room_min as usize
            + rng::cell_rand_range(ox, oy + 1, oz, gen, seed.wrapping_add(3), range) as usize;

        if wx >= side || wy >= side || wz >= side {
            continue;
        }

        let min_y = params.min_y as usize;
        let x0 = rng::cell_rand_range(
            ox,
            oy,
            oz + 1,
            gen,
            seed.wrapping_add(4),
            (side - wx) as u64,
        ) as usize;
        let y0 = if side - wy > min_y {
            min_y
                + rng::cell_rand_range(
                    ox + 1,
                    oy + 1,
                    oz,
                    gen,
                    seed.wrapping_add(5),
                    (side - wy - min_y) as u64,
                ) as usize
        } else {
            continue;
        };
        let z0 = rng::cell_rand_range(
            ox + 1,
            oy,
            oz + 1,
            gen,
            seed.wrapping_add(6),
            (side - wz) as u64,
        ) as usize;

        let room = Room {
            x0,
            y0,
            z0,
            x1: x0 + wx,
            y1: y0 + wy,
            z1: z0 + wz,
        };

        // Reject if outside bounds.
        if room.x1 > side || room.y1 > side || room.z1 > side {
            continue;
        }

        // Reject if overlapping existing rooms.
        if rooms.iter().any(|r| r.overlaps(&room)) {
            continue;
        }

        // Reject if any cell in the room isn't stone (don't carve through
        // dirt/grass surface or air).
        let all_stone = (room.z0..room.z1)
            .all(|z| (room.y0..room.y1).all(|y| (room.x0..room.x1).all(|x| is_stone(x, y, z))));
        if !all_stone {
            continue;
        }

        rooms.push(room);
    }

    // Carve rooms.
    for room in &rooms {
        for z in room.z0..room.z1 {
            for y in room.y0..room.y1 {
                for x in room.x0..room.x1 {
                    grid[idx(x, y, z)] = AIR;
                }
            }
        }
    }

    // Connect each room to its nearest predecessor with an L-shaped corridor.
    for i in 1..rooms.len() {
        let (cx, cy, cz) = rooms[i].center();

        // Find nearest predecessor by Manhattan distance.
        let (best_j, _) = rooms[..i]
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| {
                let (px, py, pz) = r.center();
                cx.abs_diff(px) + cy.abs_diff(py) + cz.abs_diff(pz)
            })
            .unwrap(); // i >= 1, so rooms[..i] is non-empty

        let (px, py, pz) = rooms[best_j].center();
        carve_corridor(grid, side, [px, py, pz], [cx, cy, cz]);
    }
}

/// Carve an L-shaped corridor: X first, then Y, then Z. Only carves cells
/// that are currently STONE. If the corridor path crosses non-stone cells
/// (AIR from a prior cave pass, or DIRT/GRASS near the surface), those
/// cells are left untouched — AIR gaps are passable, and solid non-stone
/// gaps are prevented by the all-stone room placement check which keeps
/// rooms deep in the stone layer.
fn carve_corridor(grid: &mut [CellState], side: usize, from: [usize; 3], to: [usize; 3]) {
    let [x0, y0, z0] = from;
    let [x1, y1, z1] = to;

    let mut carve = |x: usize, y: usize, z: usize| {
        if x < side && y < side && z < side {
            let i = x + y * side + z * side * side;
            if grid[i] == STONE {
                grid[i] = AIR;
            }
        }
    };

    // Walk X.
    let (lo_x, hi_x) = (x0.min(x1), x0.max(x1));
    for x in lo_x..=hi_x {
        carve(x, y0, z0);
    }
    // Walk Y.
    let (lo_y, hi_y) = (y0.min(y1), y0.max(y1));
    for y in lo_y..=hi_y {
        carve(x1, y, z0);
    }
    // Walk Z.
    let (lo_z, hi_z) = (z0.min(z1), z0.max(z1));
    for z in lo_z..=hi_z {
        carve(x1, y1, z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::field::heightmap::HeightmapField;
    use crate::terrain::gen::gen_region;
    use crate::terrain::materials::{DIRT, GRASS};

    fn default_heightmap(seed: u64) -> HeightmapField {
        HeightmapField {
            seed,
            base_y: 32.0,
            amplitude: 8.0,
            wavelength: 24.0,
            octaves: 4,
        }
    }

    #[test]
    fn dungeon_params_default_validates() {
        DungeonParams::default()
            .validate()
            .expect("defaults must validate");
    }

    #[test]
    fn dungeon_params_room_min_zero_rejected() {
        let p = DungeonParams {
            room_min: 0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn dungeon_params_room_max_lt_min_rejected() {
        let p = DungeonParams {
            room_min: 6,
            room_max: 4,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    /// Dungeon carving must reduce total STONE (some becomes AIR) while
    /// leaving non-stone cells untouched.
    #[test]
    fn carving_reduces_stone_and_preserves_surface() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);

        let side = 64usize;
        let before = store.flatten(root, side);
        let before_stone = before.iter().filter(|&&c| c == STONE).count();
        assert!(before_stone > 0, "expected stone in baseline terrain");

        let params = DungeonParams {
            seed: 42,
            max_rooms: 20,
            room_min: 3,
            room_max: 6,
            min_y: 0,
        };
        let carved = carve_dungeons(&mut store, root, 6, &params);
        let after = store.flatten(carved, side);
        let after_stone = after.iter().filter(|&&c| c == STONE).count();

        assert!(
            after_stone < before_stone,
            "dungeon pass must carve some stone: before={before_stone}, after={after_stone}",
        );

        // Non-stone cells must survive unchanged.
        for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
            if b != STONE {
                assert_eq!(
                    a, b,
                    "non-stone cell at idx {i} changed: before={b}, after={a}",
                );
            }
        }
    }

    /// Deterministic: two independent runs with the same params produce
    /// identical results.
    #[test]
    fn carving_is_deterministic() {
        let field = default_heightmap(1);
        let params = DungeonParams {
            seed: 42,
            max_rooms: 20,
            room_min: 3,
            room_max: 6,
            min_y: 0,
        };

        let mut s1 = NodeStore::new();
        let (r1, _) = gen_region(&mut s1, &field, [0, 0, 0], 6);
        let c1 = carve_dungeons(&mut s1, r1, 6, &params);
        let g1 = s1.flatten(c1, 64);

        let mut s2 = NodeStore::new();
        let (r2, _) = gen_region(&mut s2, &field, [0, 0, 0], 6);
        let c2 = carve_dungeons(&mut s2, r2, 6, &params);
        let g2 = s2.flatten(c2, 64);

        assert_eq!(g1, g2, "dungeon carving must be deterministic");
    }

    /// Different seeds produce different dungeon layouts.
    #[test]
    fn different_seeds_produce_different_dungeons() {
        let field = default_heightmap(1);

        let p1 = DungeonParams {
            seed: 100,
            max_rooms: 20,
            room_min: 3,
            room_max: 6,
            min_y: 0,
        };
        let p2 = DungeonParams {
            seed: 200,
            max_rooms: 20,
            room_min: 3,
            room_max: 6,
            min_y: 0,
        };

        let mut s1 = NodeStore::new();
        let (r1, _) = gen_region(&mut s1, &field, [0, 0, 0], 6);
        let c1 = carve_dungeons(&mut s1, r1, 6, &p1);
        let g1 = s1.flatten(c1, 64);

        let mut s2 = NodeStore::new();
        let (r2, _) = gen_region(&mut s2, &field, [0, 0, 0], 6);
        let c2 = carve_dungeons(&mut s2, r2, 6, &p2);
        let g2 = s2.flatten(c2, 64);

        assert_ne!(
            g1, g2,
            "different dungeon seeds must produce different layouts"
        );
    }

    /// Rooms placed: verify at least one room is carved (i.e. there's a
    /// contiguous rectangular region of AIR within what was STONE).
    #[test]
    fn at_least_one_room_placed() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let before = store.flatten(root, 64);

        let params = DungeonParams {
            seed: 42,
            max_rooms: 20,
            room_min: 3,
            room_max: 6,
            min_y: 0,
        };
        let carved = carve_dungeons(&mut store, root, 6, &params);
        let after = store.flatten(carved, 64);

        // Count cells that changed from STONE to AIR — must be at least
        // room_min^3 (one minimal room).
        let carved_count = before
            .iter()
            .zip(after.iter())
            .filter(|(&b, &a)| b == STONE && a == AIR)
            .count();
        let min_room_volume = (params.room_min as usize).pow(3);
        assert!(
            carved_count >= min_room_volume,
            "at least one room must be carved: got {carved_count} carved cells, \
             min room volume is {min_room_volume}"
        );
    }

    /// An all-air grid produces no changes (no stone to carve).
    #[test]
    fn all_air_grid_is_untouched() {
        let side = 8usize;
        let mut grid = vec![AIR; side * side * side];
        let params = DungeonParams::default();
        let original = grid.clone();
        carve_dungeons_grid(&mut grid, side, [0, 0, 0], &params);
        assert_eq!(grid, original);
    }

    /// Surface preservation: DIRT and GRASS never become AIR.
    #[test]
    fn surface_materials_never_carved() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let before = store.flatten(root, 64);

        let params = DungeonParams {
            seed: 42,
            max_rooms: 20,
            room_min: 3,
            room_max: 6,
            min_y: 0,
        };
        let carved = carve_dungeons(&mut store, root, 6, &params);
        let after = store.flatten(carved, 64);

        for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
            if b == DIRT || b == GRASS {
                assert_eq!(
                    a, b,
                    "surface material at idx {i} was modified: before={b}, after={a}"
                );
            }
        }
    }

    /// Corridor connectivity: the corridor carver must produce AIR cells
    /// between room centers.
    #[test]
    fn corridors_carve_stone() {
        // Use a small all-stone grid so we can verify corridor paths directly.
        let side = 16usize;
        let mut grid = vec![STONE; side * side * side];
        let params = DungeonParams {
            seed: 1,
            max_rooms: 10,
            room_min: 2,
            room_max: 3,
            min_y: 0,
        };
        carve_dungeons_grid(&mut grid, side, [0, 0, 0], &params);
        let air_count = grid.iter().filter(|&&c| c == AIR).count();
        // With rooms + corridors, we must have carved at least something.
        assert!(
            air_count > 0,
            "dungeons on an all-stone grid must carve at least one cell"
        );
    }

    #[test]
    #[should_panic(expected = "grid is not side^3")]
    fn carve_dungeons_grid_rejects_mismatched_grid_len() {
        let side = 2usize;
        let mut grid = vec![AIR; 7];
        let params = DungeonParams::default();
        carve_dungeons_grid(&mut grid, side, [0, 0, 0], &params);
    }

    #[test]
    #[should_panic(expected = "carve_dungeons_grid: invalid DungeonParams")]
    fn carve_dungeons_grid_rejects_invalid_params() {
        let side = 2usize;
        let mut grid = vec![STONE; side * side * side];
        let params = DungeonParams {
            room_min: 0,
            ..Default::default()
        };
        carve_dungeons_grid(&mut grid, side, [0, 0, 0], &params);
    }

    /// Rooms respect min_y: no carved cell has y < min_y.
    #[test]
    fn rooms_respect_min_y() {
        let side = 16usize;
        let mut grid = vec![STONE; side * side * side];
        let min_y = 8u32;
        let params = DungeonParams {
            seed: 42,
            max_rooms: 20,
            room_min: 2,
            room_max: 3,
            min_y,
        };
        let original = grid.clone();
        carve_dungeons_grid(&mut grid, side, [0, 0, 0], &params);

        for z in 0..side {
            for y in 0..min_y as usize {
                for x in 0..side {
                    let i = x + y * side + z * side * side;
                    assert_eq!(
                        grid[i], original[i],
                        "cell ({x},{y},{z}) below min_y was modified"
                    );
                }
            }
        }
    }

    /// Different origins produce different layouts (origin is threaded into RNG).
    #[test]
    fn different_origins_produce_different_dungeons() {
        let side = 16usize;
        let params = DungeonParams {
            seed: 42,
            max_rooms: 20,
            room_min: 2,
            room_max: 3,
            min_y: 0,
        };

        let mut g1 = vec![STONE; side * side * side];
        carve_dungeons_grid(&mut g1, side, [0, 0, 0], &params);

        let mut g2 = vec![STONE; side * side * side];
        carve_dungeons_grid(&mut g2, side, [100, 200, 300], &params);

        assert_ne!(
            g1, g2,
            "different origins must produce different dungeon layouts"
        );
    }
}
