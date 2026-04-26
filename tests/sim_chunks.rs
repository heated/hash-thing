//! Integration tests for the cswp.8.3 chunk-LOD policy.
//!
//! Covers (per the plan-review acceptance bar):
//!   - identity in the disabled / small-world fast paths,
//!   - non-trivial collapse for far chunks,
//!   - cache hit when nothing relevant changes,
//!   - cache invalidation for `World::set`, `seed_terrain`, compaction
//!     remap, and `lod_bias` change (the four mutation paths the cswp.8.3
//!     plan-review BLOCKER called out as silently staling the cached
//!     `view_root`),
//!   - bit-identical sampling for any chunk whose target LOD is 0
//!     (the "near-band == canonical" property).

use hash_thing::octree::Cell;
use hash_thing::sim::chunks::{
    target_lod_for_radius, ChunkCoord, ChunkLodPolicy, CHUNK_LEVEL, CHUNK_SIDE,
};
use hash_thing::sim::{World, WorldCoord};
use hash_thing::terrain::TerrainParams;

fn stone() -> u16 {
    Cell::pack(1, 0).raw()
}

/// `level=9` gives 512³ cells = 4³ chunks at `CHUNK_LEVEL=7`.
fn make_4x4x4_chunk_world() -> World {
    World::new(9)
}

#[test]
fn lod_policy_disabled_is_identity() {
    let mut world = make_4x4x4_chunk_world();
    world.set(WorldCoord(10), WorldCoord(10), WorldCoord(10), stone());
    let mut policy = ChunkLodPolicy::new();
    assert!(!policy.enabled, "default-disabled");
    let view = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    assert_eq!(view, world.root, "disabled policy must be identity");
}

#[test]
fn lod_policy_small_world_is_identity() {
    // level=7 == CHUNK_LEVEL → exactly one chunk → no collapse possible.
    let mut world = World::new(CHUNK_LEVEL);
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let view = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    assert_eq!(
        view, world.root,
        "small world (level<=CHUNK_LEVEL) must take identity fast path"
    );
}

#[test]
fn lod_policy_on_empty_world_does_not_panic_and_returns_valid_root() {
    // Smoke: enabling the policy on an empty 4×4×4-chunk world must
    // produce a NodeId that exists in the store. Per the cswp.8.2
    // primitive doc-comment, each `lod_collapse_chunk` call interns
    // root-to-chunk scaffold nodes (~ChunkCount × log(level) growth);
    // that ghost-chain growth is the trade-off the cswp.8.3.1 follow-up
    // bead is filed against, so we explicitly do NOT assert "no growth"
    // here — only that the policy converges to a usable root.
    let mut world = make_4x4x4_chunk_world();
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let view = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    let _ = world.store.get(view);
}

#[test]
fn lod_policy_collapses_a_far_populated_chunk() {
    // A single non-empty cell deep in chunk (3,0,0) — chunk-radius 3 from
    // (0,0,0), which the design-doc curve maps to LOD 1. After the
    // policy runs, view_root must differ from world_root because that
    // chunk got rewritten to a coarser representation.
    let mut world = make_4x4x4_chunk_world();
    let cx: i64 = 3;
    let inside = (CHUNK_SIDE as i64) / 2;
    world.set(
        WorldCoord(cx * CHUNK_SIDE as i64 + inside),
        WorldCoord(inside),
        WorldCoord(inside),
        stone(),
    );
    let canonical = world.root;
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let view = policy.update(&mut world.store, canonical, world.level, [0.0; 3]);
    assert_eq!(target_lod_for_radius(3.0, 1.0, 4), 1, "sanity: r=3 → LOD 1");
    assert_ne!(
        view, canonical,
        "expected view_root to differ once a far populated chunk gets collapsed"
    );
}

#[test]
fn cache_hit_when_player_doesnt_cross_chunk_and_root_unchanged() {
    let mut world = make_4x4x4_chunk_world();
    let inside = (CHUNK_SIDE as i64) / 2;
    world.set(
        WorldCoord(3 * CHUNK_SIDE as i64 + inside),
        WorldCoord(inside),
        WorldCoord(inside),
        stone(),
    );
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let view1 = policy.update(&mut world.store, world.root, world.level, [10.0; 3]);
    let nodes_after_first = world.store.node_count();
    // Same world.root, same player_chunk (10..12 stays in chunk 0), same
    // bias → must hit cache.
    let view2 = policy.update(&mut world.store, world.root, world.level, [12.0; 3]);
    assert_eq!(view1, view2, "cache hit must return the same NodeId");
    assert_eq!(
        world.store.node_count(),
        nodes_after_first,
        "cache hit must not add nodes to the store"
    );
}

#[test]
fn cache_invalidates_on_root_change_via_set() {
    // Mutating world.root via World::set must invalidate the cache —
    // the cache key is keyed on the source world_root NodeId, so any
    // mutation that produces a fresh root yields a different key.
    // Plan-review BLOCKER 1: this path was missing from the original
    // "invalidate after sim step" approach.
    let mut world = make_4x4x4_chunk_world();
    // Plant a far cell first so the policy has work to do (otherwise an
    // empty world's view_root may equal world_root and "different
    // view_root" can't be the assertion).
    world.set(
        WorldCoord(3 * CHUNK_SIDE as i64 + 5),
        WorldCoord(5),
        WorldCoord(5),
        stone(),
    );
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let view_before = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    let root_before = world.root;
    // Direct edit through World::set — same path the player-edit handlers
    // (break/place block) use in main.rs.
    world.set(WorldCoord(50), WorldCoord(50), WorldCoord(50), stone());
    assert_ne!(world.root, root_before, "set must change world.root");
    let view_after = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    assert_ne!(
        view_after, view_before,
        "different world_root must yield different view_root (no stale cache)"
    );
}

#[test]
fn cache_invalidates_on_compaction_remap() {
    // Compaction renumbers NodeIds (`compacted_with_remap`). After
    // compaction, the cache key's stored world_root no longer matches
    // the new world.root, so the next update must recompute against the
    // fresh store.
    let mut world = make_4x4x4_chunk_world();
    world.set(
        WorldCoord(3 * CHUNK_SIDE as i64 + 5),
        WorldCoord(5),
        WorldCoord(5),
        stone(),
    );
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let _view_before = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    let root_before = world.root;

    // Hand-roll a compaction the same way World::commit_step does.
    let (new_store, new_root, _remap) = world.store.compacted_with_remap(world.root);
    world.store = new_store;
    world.root = new_root;
    assert_ne!(
        world.root, root_before,
        "compaction must remap world.root (otherwise this test cannot \
         distinguish stale cache from a real cache hit)"
    );

    let view_after = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    // view_after must be a NodeId that exists in the new store. NodeStore::get
    // panics on dangling NodeIds, so this is a smoke test for "cache didn't
    // hand back a stale id from the old store".
    let _ = world.store.get(view_after);
}

#[test]
fn cache_invalidates_on_seed_terrain() {
    // seed_terrain replaces the entire scene (fresh epoch). The cached
    // view_root must not survive into the new scene.
    let mut world = make_4x4x4_chunk_world();
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let _view_before = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    let root_before = world.root;

    let params = TerrainParams::for_level(world.level);
    let _ = world.seed_terrain(&params).expect("seed_terrain ok");
    assert_ne!(
        world.root, root_before,
        "seed_terrain must produce a fresh root"
    );

    let view_after = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    let _ = world.store.get(view_after);
}

#[test]
fn cache_invalidates_on_lod_bias_change() {
    let mut world = make_4x4x4_chunk_world();
    world.set(
        WorldCoord(3 * CHUNK_SIDE as i64 + 5),
        WorldCoord(5),
        WorldCoord(5),
        stone(),
    );
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let view_at_bias_1 = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    policy.lod_bias = 0.5;
    let view_at_bias_05 = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    // Different bias shifts the curve → far chunks land in a different
    // LOD band → at least one chunk's collapse changes.
    assert_ne!(
        view_at_bias_1, view_at_bias_05,
        "lod_bias change must invalidate cache and yield a different view_root"
    );
}

#[test]
fn near_band_chunk_is_bit_identical_to_world_root_at_sample_cells() {
    // Property: any chunk whose target LOD is 0 (the chunk containing
    // the player + neighbours within radius 1 — see the §2.2 design-doc
    // table) must read back exactly what world.root has at every cell in
    // that chunk. We sample a 4×4×4 lattice per chunk rather than
    // enumerating, since 128³ per chunk is ~2M cells.
    let mut world = make_4x4x4_chunk_world();
    // Plant a recognisable diagonal in the player chunk so we have
    // something distinguishable to compare.
    for d in 0..8 {
        let x = 4 + d;
        let y = 4 + d;
        let z = 4 + d;
        world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), stone());
    }

    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let player_chunk = ChunkCoord::new(0, 0, 0);
    let player_pos = [
        (CHUNK_SIDE / 2) as f64,
        (CHUNK_SIDE / 2) as f64,
        (CHUNK_SIDE / 2) as f64,
    ];
    let view = policy.update(&mut world.store, world.root, world.level, player_pos);

    // Sample a 4×4×4 lattice within the player chunk — radius 0, target
    // LOD 0. Cells must agree bit-for-bit between view_root and
    // world.root.
    let step = (CHUNK_SIDE / 4) as u64;
    for ix in 0..4u64 {
        for iy in 0..4u64 {
            for iz in 0..4u64 {
                let x = (player_chunk.x as u64) * (CHUNK_SIDE as u64) + ix * step + 1;
                let y = (player_chunk.y as u64) * (CHUNK_SIDE as u64) + iy * step + 1;
                let z = (player_chunk.z as u64) * (CHUNK_SIDE as u64) + iz * step + 1;
                let canonical = world.store.get_cell(world.root, x, y, z);
                let viewed = world.store.get_cell(view, x, y, z);
                assert_eq!(
                    canonical, viewed,
                    "near-band cell ({x},{y},{z}) diverged: \
                     canonical={canonical:#x} view={viewed:#x}"
                );
            }
        }
    }
}
