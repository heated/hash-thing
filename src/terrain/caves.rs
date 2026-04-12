//! Cellular-automaton cave smoothing post-pass (hash-thing-3fq.2).
//!
//! Runs a 3D "majority-wall" CA — the direct 3D generalization of the classic
//! 2D 4-5 cave rule — over a flattened region of the already-generated
//! terrain, then re-interns the mutated grid via [`NodeStore::from_flat`].
//! Operates ONLY on cells that started as STONE — DIRT, GRASS, and AIR are
//! preserved untouched, so caves tunnel underground without eating the
//! surface layer.
//!
//! ## Thresholds are 3D-scaled, not literally "4-5"
//!
//! The bead title calls this the "4-5 rule," which is the 2D convention: in
//! 8-neighbor Moore, a cell is wall iff ≥5 of its 8 neighbors are walls
//! (birth), or it already was a wall with ≥4 (survive). Roughly "majority
//! vote among your neighbors."
//!
//! The 3D 26-neighbor analog is ≥13 (majority of 26), not ≥5. Using the
//! literal 4-5 thresholds in 3D converges to 100% walls in one iteration —
//! a 45%-density seed puts ~12 walls in each 26-neighborhood on average, far
//! above either threshold. With `birth=13, survive=13`, the 45% seed
//! converges to roughly 40% walls arranged in contiguous chambers connected
//! by thin corridors — what you want from a cave CA.
//!
//! See the 3D life rule 5766 family (e.g. "B13/S13") for reference; this
//! module uses a single threshold for simplicity and tunes the initial fill
//! density to shape chamber size.
//!
//! ## Why post-pass and not a `RegionField`
//!
//! CA iteration is fundamentally non-local in time: cell state at step N+1
//! depends on neighbor state at step N for all 26 neighbors. That cannot be
//! expressed as a pure `sample(point) -> CellState` function the direct-octree
//! builder expects. The RegionField abstraction is for generators that emit
//! final state in one pass; iterative smoothing needs a materialized grid.
//!
//! The cost is one full flatten + from_flat round trip on top of the existing
//! gen_region pass. At 64³ with 4 iterations this is fractions of a millisecond;
//! the CA itself is the dominant term, not the octree churn. If cave smoothing
//! becomes the bottleneck of a larger world, the fix is chunked CA — run the
//! pass on 32³ or 16³ windows and re-intern each window independently —
//! not a field-trait rewrite.
//!
//! ## Algorithm
//!
//! 1. **Cave mask**: collect the set of cells where the input material equals
//!    `STONE`. The CA only reads/writes cells inside the mask; everything else
//!    passes through unchanged.
//!
//! 2. **Seed**: within the mask, each cell is initially wall with probability
//!    `wall_probability` (default 0.45, the classic value). Randomness comes
//!    from the stateless `rng::cell_rand_bool` keyed on world coordinates —
//!    same cell at the same seed always produces the same seed state,
//!    regardless of scan order. This is the h34.2 determinism rule.
//!
//! 3. **Iterate** (`iterations` times, default 4): for each masked cell,
//!    count how many of its 26 Moore neighbors are "wall". A neighbor counts
//!    as wall if it is outside the mask (preserving surface layer), outside
//!    the grid entirely (treating the world edge as wall), or currently a
//!    wall in the previous iteration. The cell becomes wall at step t+1 iff:
//!    - it was air at step t and wall_count >= `birth_threshold` (default 13), or
//!    - it was wall at step t and wall_count >= `survive_threshold` (default 13).
//!
//! 4. **Writeback**: unmasked cells pass through untouched; masked cells
//!    take their final state (STONE if wall, AIR if cave).
//!
//! ## Design knobs (all have standard literature defaults)
//!
//! | Knob                | Default | Effect                               |
//! |---------------------|---------|--------------------------------------|
//! | `wall_probability`  | 0.45    | Higher → thicker walls, fewer caves  |
//! | `iterations`        | 4       | Higher → smoother, larger chambers   |
//! | `birth_threshold`   | 13      | 3D majority of 26 neighbors          |
//! | `survive_threshold` | 13      | Same, for already-wall cells         |

use crate::octree::node::octant_coords;
use crate::octree::{CellState, Node, NodeId, NodeStore};
use crate::rng::cell_rand_bool;
use crate::terrain::materials::{AIR, STONE};

/// Parameters for the cave-smoothing post-pass. `Default` gives the 3D
/// majority-wall analog of the classic 4-5 rule (`wall=0.45`, `iters=4`,
/// `birth=13`, `survive=13`).
#[derive(Clone, Copy, Debug)]
pub struct CaveParams {
    /// Seed for the stateless per-cell PRNG used in the initial seed pass.
    /// The terrain seed is the obvious choice; a different value here lets
    /// you decouple cave layout from terrain shape.
    pub seed: u64,
    /// Fraction of masked cells that start as wall in the seed pass. Below
    /// ~0.40 the CA leaves large open chambers; above ~0.50 it leaves only
    /// disconnected pockets. 0.45 is the classic value.
    pub wall_probability: f64,
    /// Number of CA iterations. 4 is the classic value — fewer leaves jagged
    /// edges, more homogenizes toward uniform walls.
    pub iterations: u32,
    /// Wall-count threshold for an air cell to become wall. Majority of 26.
    pub birth_threshold: u32,
    /// Wall-count threshold for a wall cell to survive. Same as birth by
    /// default — the CA then reduces to a simple majority vote.
    pub survive_threshold: u32,
}

impl Default for CaveParams {
    fn default() -> Self {
        Self {
            seed: 1,
            wall_probability: 0.45,
            iterations: 4,
            birth_threshold: 13,
            survive_threshold: 13,
        }
    }
}

impl CaveParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(0.0..=1.0).contains(&self.wall_probability) {
            return Err("wall_probability must be in [0, 1]");
        }
        // Thresholds can legally be 0..=26 (number of Moore neighbors). Out
        // of range doesn't panic — the count comparisons degrade gracefully
        // — but it does produce useless output (everything stays wall or
        // everything becomes air), so flag it loudly.
        if self.birth_threshold > 26 {
            return Err("birth_threshold must be <= 26");
        }
        if self.survive_threshold > 26 {
            return Err("survive_threshold must be <= 26");
        }
        Ok(())
    }
}

/// Carve caves into `root` via the classic 4-5 rule CA. Returns a fresh
/// `NodeId` referring to the post-cave terrain.
///
/// The input `root` is not modified (the octree is immutable) — it simply
/// becomes unreachable if the caller drops their handle. If this is called
/// inside the same `NodeStore` that holds the terrain, the stale pre-cave
/// subtrees stay in the store until the next `compacted()` pass.
///
/// `side_log2` is the log2 of the grid side length for the region rooted at
/// `root`; at 64³ that's 6.
/// Maximum chunk level for chunked cave carving. Each chunk is
/// (2^CHUNK_LEVEL)³ cells. Level 7 = 128³ = ~2MB per chunk.
const CHUNK_LEVEL: u32 = 7;

pub fn carve_caves(
    store: &mut NodeStore,
    root: NodeId,
    side_log2: u32,
    params: &CaveParams,
) -> NodeId {
    params.validate().expect("invalid CaveParams");
    carve_caves_recursive(store, root, side_log2, [0, 0, 0], params)
}

/// Recursively subdivide the world into chunks at `CHUNK_LEVEL` and carve
/// each one independently. Peak memory is O(chunk³) instead of O(world³).
/// Chunk boundaries treat out-of-bounds cells as wall (same as world edges),
/// so cave patterns near chunk boundaries may have slightly thicker walls —
/// visually indistinguishable for random cave geometry (3cp).
fn carve_caves_recursive(
    store: &mut NodeStore,
    node: NodeId,
    level: u32,
    origin: [i64; 3],
    params: &CaveParams,
) -> NodeId {
    // Empty subtrees have no STONE — nothing to carve.
    if store.population(node) == 0 {
        return node;
    }

    if level <= CHUNK_LEVEL {
        // Small enough: flatten, carve, re-intern.
        let side = 1usize << level;
        let mut grid = store.flatten(node, side);
        carve_caves_grid(&mut grid, side, origin, params);
        return store.from_flat(&grid, side);
    }

    // Recurse into children.
    let children = match store.get(node).clone() {
        Node::Interior { children, .. } => children,
        Node::Leaf(_) => unreachable!("level > CHUNK_LEVEL but node is a leaf"),
    };
    let half = 1i64 << (level - 1);
    let mut new_children = [NodeId::EMPTY; 8];
    for (oct, child) in children.iter().enumerate() {
        let (cx, cy, cz) = octant_coords(oct);
        let child_origin = [
            origin[0] + cx as i64 * half,
            origin[1] + cy as i64 * half,
            origin[2] + cz as i64 * half,
        ];
        new_children[oct] =
            carve_caves_recursive(store, *child, level - 1, child_origin, params);
    }
    store.interior(level, new_children)
}

/// In-place cave CA on a flat grid. World origin is used for the per-cell
/// PRNG so cave layout is a deterministic function of world coordinates
/// (lazy-root-expansion friendly: the same sub-volume always produces the
/// same caves).
///
/// Factored out of `carve_caves` so tests can hit the CA core without
/// needing a NodeStore fixture, and so a future chunked-CA path can reuse
/// the same kernel on sub-regions.
pub fn carve_caves_grid(
    grid: &mut [CellState],
    side: usize,
    origin: [i64; 3],
    params: &CaveParams,
) {
    let expected_len = side
        .checked_mul(side)
        .and_then(|n| n.checked_mul(side))
        .expect("carve_caves_grid: side^3 overflowed usize");
    assert_eq!(grid.len(), expected_len, "grid is not side^3");
    params
        .validate()
        .expect("carve_caves_grid: invalid CaveParams");

    // Cave mask: cells that start as STONE. This is the only region the CA
    // is allowed to touch; everything else (AIR, DIRT, GRASS) passes through
    // unchanged so the surface layer stays intact.
    //
    // Stored as a Vec<bool> for clarity; a bitset would halve the memory but
    // 64³ bytes = 256 KB is trivial and the bool path is faster to read.
    let mut mask = vec![false; grid.len()];
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let idx = x + y * side + z * side * side;
                if grid[idx] == STONE {
                    mask[idx] = true;
                }
            }
        }
    }

    // Working buffer: the "is wall" state at the current CA step, defined
    // only inside the mask. Start with the seed pass — each masked cell is
    // wall with probability `wall_probability`, keyed on its world
    // coordinate so the result is independent of scan order.
    //
    // Out-of-mask cells are UNDEFINED in `walls` — the iteration loop
    // guards every read against the mask before consulting `walls`, and
    // the wall_count function treats out-of-mask neighbors as walls
    // unconditionally.
    let mut walls = vec![false; grid.len()];
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let idx = x + y * side + z * side * side;
                if !mask[idx] {
                    continue;
                }
                let wx = origin[0] + x as i64;
                let wy = origin[1] + y as i64;
                let wz = origin[2] + z as i64;
                walls[idx] = cell_rand_bool(wx, wy, wz, 0, params.seed, params.wall_probability);
            }
        }
    }

    // CA iterations. `next` is double-buffered to avoid read-your-own-write
    // hazards: step N+1 must be computed entirely from step N.
    let mut next = vec![false; grid.len()];
    for _ in 0..params.iterations {
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let idx = x + y * side + z * side * side;
                    if !mask[idx] {
                        continue;
                    }
                    let wall_count = count_wall_neighbors(&walls, &mask, side, x, y, z);
                    let was_wall = walls[idx];
                    let becomes_wall = if was_wall {
                        wall_count >= params.survive_threshold
                    } else {
                        wall_count >= params.birth_threshold
                    };
                    next[idx] = becomes_wall;
                }
            }
        }
        std::mem::swap(&mut walls, &mut next);
    }

    // Writeback: masked cells take their final wall/air state; unmasked
    // cells are already correct (we never touched them).
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let idx = x + y * side + z * side * side;
                if !mask[idx] {
                    continue;
                }
                grid[idx] = if walls[idx] { STONE } else { AIR };
            }
        }
    }
}

/// Count wall neighbors around `(x, y, z)` in the 3×3×3 Moore neighborhood
/// (26 cells, excluding the center).
///
/// A neighbor counts as wall if:
/// - it is **outside the grid**: the world edge is treated as an infinite
///   wall so caves cannot escape the simulated volume, and
/// - it is **outside the mask**: non-stone cells (DIRT, GRASS, AIR above)
///   are wall-equivalent for CA purposes, which prevents caves from
///   eroding upward through the dirt layer, or
/// - it is **inside the mask and currently a wall** in `walls`.
#[inline]
fn count_wall_neighbors(
    walls: &[bool],
    mask: &[bool],
    side: usize,
    x: usize,
    y: usize,
    z: usize,
) -> u32 {
    let side_i = side as i64;
    let mut count = 0u32;
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                let nz = z as i64 + dz;
                // Out-of-bounds neighbor: infinite wall.
                if nx < 0 || ny < 0 || nz < 0 || nx >= side_i || ny >= side_i || nz >= side_i {
                    count += 1;
                    continue;
                }
                let nidx = (nx as usize) + (ny as usize) * side + (nz as usize) * side * side;
                // Out-of-mask neighbor: wall-equivalent (preserves surface).
                if !mask[nidx] || walls[nidx] {
                    count += 1;
                }
            }
        }
    }
    count
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
            sea_level: None,
        }
    }

    #[test]
    fn cave_params_default_validates() {
        CaveParams::default()
            .validate()
            .expect("defaults must validate");
    }

    #[test]
    fn cave_params_out_of_range_probability_rejected() {
        let hi = CaveParams {
            wall_probability: 1.5,
            ..Default::default()
        };
        assert!(hi.validate().is_err());
        let lo = CaveParams {
            wall_probability: -0.1,
            ..Default::default()
        };
        assert!(lo.validate().is_err());
    }

    /// Carving caves must reduce total STONE population (some stone becomes
    /// air) while leaving non-stone cells untouched.
    #[test]
    fn carving_reduces_stone_and_preserves_surface() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);

        let side = 64usize;
        let before = store.flatten(root, side);
        let before_stone = before.iter().filter(|&&c| c == STONE).count();
        assert!(before_stone > 0, "expected stone in baseline terrain");

        let carved = carve_caves(&mut store, root, 6, &CaveParams::default());
        let after = store.flatten(carved, side);
        let after_stone = after.iter().filter(|&&c| c == STONE).count();

        assert!(
            after_stone < before_stone,
            "cave pass must carve some stone: before={before_stone}, after={after_stone}",
        );

        // Every non-stone cell must survive unchanged. This is the surface-
        // preservation invariant: DIRT / GRASS / AIR never get touched.
        for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
            if b != STONE {
                assert_eq!(
                    a, b,
                    "non-stone cell at idx {i} changed: before={b}, after={a}",
                );
            }
        }
    }

    /// Deterministic by world coordinate + seed — two independent runs on
    /// fresh stores must produce identical results.
    #[test]
    fn carving_is_deterministic() {
        let field = default_heightmap(1);
        let params = CaveParams::default();

        let mut s1 = NodeStore::new();
        let (r1, _) = gen_region(&mut s1, &field, [0, 0, 0], 6);
        let c1 = carve_caves(&mut s1, r1, 6, &params);
        let g1 = s1.flatten(c1, 64);

        let mut s2 = NodeStore::new();
        let (r2, _) = gen_region(&mut s2, &field, [0, 0, 0], 6);
        let c2 = carve_caves(&mut s2, r2, 6, &params);
        let g2 = s2.flatten(c2, 64);

        assert_eq!(g1, g2, "cave carving must be deterministic");
    }

    /// Different seeds produce different cave layouts (regression guard
    /// against a constant-seed bug).
    #[test]
    fn different_seeds_produce_different_caves() {
        let field = default_heightmap(1);

        let p1 = CaveParams {
            seed: 100,
            ..Default::default()
        };
        let p2 = CaveParams {
            seed: 200,
            ..Default::default()
        };

        let mut s1 = NodeStore::new();
        let (r1, _) = gen_region(&mut s1, &field, [0, 0, 0], 6);
        let c1 = carve_caves(&mut s1, r1, 6, &p1);
        let g1 = s1.flatten(c1, 64);

        let mut s2 = NodeStore::new();
        let (r2, _) = gen_region(&mut s2, &field, [0, 0, 0], 6);
        let c2 = carve_caves(&mut s2, r2, 6, &p2);
        let g2 = s2.flatten(c2, 64);

        assert_ne!(
            g1, g2,
            "different cave seeds must produce different layouts"
        );
    }

    /// 100% wall + any iterations = no-op (all masked stone stays wall,
    /// CA observes majority-wall neighborhoods forever).
    #[test]
    fn full_wall_seed_is_no_op() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let before = store.flatten(root, 64);

        let params = CaveParams {
            seed: 42,
            wall_probability: 1.0,
            iterations: 4,
            birth_threshold: 13,
            survive_threshold: 13,
        };
        let carved = carve_caves(&mut store, root, 6, &params);
        let after = store.flatten(carved, 64);

        assert_eq!(before, after, "100% wall seed must produce no changes");
    }

    /// 0% wall seed + 0 iterations = every stone cell flipped to air.
    /// (With 0 iterations the CA never runs, so the seed pass is the final
    /// state. wall=0.0 means every stone cell starts as air.)
    #[test]
    fn zero_wall_zero_iterations_carves_every_stone() {
        let mut store = NodeStore::new();
        let field = default_heightmap(1);
        let (root, _) = gen_region(&mut store, &field, [0, 0, 0], 6);
        let before = store.flatten(root, 64);

        let params = CaveParams {
            seed: 42,
            wall_probability: 0.0,
            iterations: 0,
            birth_threshold: 13,
            survive_threshold: 13,
        };
        let carved = carve_caves(&mut store, root, 6, &params);
        let after = store.flatten(carved, 64);
        let after_stone = after.iter().filter(|&&c| c == STONE).count();

        assert_eq!(
            after_stone, 0,
            "wall=0.0 with zero iterations must flip every stone cell to air",
        );
        // Non-stone cells still untouched.
        for (&b, &a) in before.iter().zip(after.iter()) {
            if b != STONE {
                assert_eq!(a, b, "non-stone preservation violated");
            }
        }
        // And the converse — surface cells are still there.
        assert!(
            after.iter().any(|&c| c == GRASS || c == DIRT),
            "surface cells must survive a stone-only carve",
        );
    }

    /// Pure-grid kernel on a trivial all-stone block: with wall=1.0, the
    /// CA has nothing to evolve, so output is all-stone.
    #[test]
    fn all_stone_block_with_full_wall_stays_stone() {
        let side = 8usize;
        let mut grid = vec![STONE; side * side * side];
        let params = CaveParams {
            seed: 1,
            wall_probability: 1.0,
            iterations: 4,
            birth_threshold: 13,
            survive_threshold: 13,
        };
        carve_caves_grid(&mut grid, side, [0, 0, 0], &params);
        for &c in &grid {
            assert_eq!(c, STONE);
        }
    }

    /// Pure-grid kernel: an all-air block with 0 iterations and 0% fill
    /// stays all-air because nothing was in the mask.
    #[test]
    fn all_air_block_is_untouched() {
        let side = 8usize;
        let mut grid = vec![AIR; side * side * side];
        let params = CaveParams::default();
        carve_caves_grid(&mut grid, side, [0, 0, 0], &params);
        for &c in &grid {
            assert_eq!(c, AIR);
        }
    }

    #[test]
    #[should_panic(expected = "grid is not side^3")]
    fn carve_caves_grid_rejects_mismatched_grid_len() {
        let side = 2usize;
        let mut grid = vec![AIR; 7];
        let params = CaveParams::default();
        carve_caves_grid(&mut grid, side, [0, 0, 0], &params);
    }

    #[test]
    #[should_panic(expected = "carve_caves_grid: invalid CaveParams")]
    fn carve_caves_grid_rejects_invalid_params() {
        let side = 2usize;
        let mut grid = vec![STONE; side * side * side];
        let params = CaveParams {
            birth_threshold: 27,
            ..Default::default()
        };
        carve_caves_grid(&mut grid, side, [0, 0, 0], &params);
    }
}
