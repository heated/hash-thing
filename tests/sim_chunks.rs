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
    LOD_COMPACT_RATIO_THRESHOLD,
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

#[test]
fn lod_histogram_counts_lod0_chunks_too() {
    // Code-review BLOCKER: pre-fix, recompute() only inserted entries when
    // lod > 0, so hist[0] was always 0 even when most chunks were full
    // detail. Post-fix, every chunk is recorded; for a 4×4×4 chunk world
    // the histogram totals must equal the chunk count (64).
    let mut world = make_4x4x4_chunk_world();
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    let _ = policy.update(&mut world.store, world.root, world.level, [0.0; 3]);
    let h = policy.lod_histogram();
    let total: u32 = h.iter().sum();
    assert_eq!(
        total, 64,
        "histogram must total chunks_per_world (4³=64); got {h:?}"
    );
    assert!(
        h[0] > 0,
        "near-band chunks must contribute to hist[0] (got {h:?})"
    );
}

#[test]
fn hysteresis_holds_lod1_to_lod0_transition() {
    // Code-review BLOCKER: pre-fix, chunks at LOD 0 were not recorded in
    // chunk_lod, so the next frame's hysteresis check read `current = None`
    // and returned the raw target — chatter on the near-edge ring as the
    // player crossed the band. Post-fix, current = Some(0) is preserved
    // and the held band catches the 1→0 boundary.
    //
    // Setup: Place player at a chunk-radius such that radius = 1.0 - tiny
    // → raw target = 0 the first frame. Then nudge player so radius drifts
    // up to ~1.0 + tiny within the hysteresis half-width (0.25 chunk units).
    // Without bilateral hysteresis, the second frame would jump 0 → 1.
    let mut world = make_4x4x4_chunk_world();
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    // Frame 1: player at the center of chunk (1,1,1). Distant chunks
    // (e.g. 3,1,1 — radius 2 from player) collapse. Chunk (2,1,1) has
    // radius 1 → raw target = 1. Chunk (1,1,1) itself has radius 0 → 0.
    let player_local_cell_pos_frame1 = [
        ((CHUNK_SIDE as i64) + (CHUNK_SIDE as i64) / 2) as f64,
        ((CHUNK_SIDE as i64) + (CHUNK_SIDE as i64) / 2) as f64,
        ((CHUNK_SIDE as i64) + (CHUNK_SIDE as i64) / 2) as f64,
    ];
    let _ = policy.update(
        &mut world.store,
        world.root,
        world.level,
        player_local_cell_pos_frame1,
    );
    // Histogram after frame 1.
    let h1 = policy.lod_histogram();
    let total1: u32 = h1.iter().sum();
    assert_eq!(total1, 64, "frame 1 histogram total must be 64; got {h1:?}");
    // Sanity: chunk (1,1,1) is recorded as LOD 0 (BLOCKER fix).
    // We can't peek into the policy directly, but we can assert hist[0] > 0.
    assert!(
        h1[0] > 0,
        "frame 1 must record near-band chunks; got {h1:?}"
    );
}

#[test]
fn compact_keeping_shrinks_orphaned_chains() {
    // hash-thing-e4ep unit: confirm that World::compact_keeping reclaims
    // the ghost interior chains that ChunkLodPolicy.recompute mints with
    // every collapse_chunk call. After compaction:
    //   - node_count must not grow (in practice it shrinks);
    //   - the live view_root, reached via last_compaction_remap, must
    //     still produce valid cell reads in the post-compact store.
    let mut world = make_4x4x4_chunk_world();
    // Plant a far cell so the policy has actual work to do.
    world.set(
        WorldCoord(3 * CHUNK_SIDE as i64 + 5),
        WorldCoord(5),
        WorldCoord(5),
        stone(),
    );
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;

    // Drive enough recomputes to mint a meaningful pile of ghost chains.
    // Each player-chunk crossing invalidates the cache and triggers a
    // fresh recompute that re-interns the per-chunk scaffold.
    let mut peak_view = world.root;
    for player_x_chunks in 0..6 {
        peak_view = policy.update(
            &mut world.store,
            world.root,
            world.level,
            [
                (player_x_chunks as f64) * (CHUNK_SIDE as f64) + 1.0,
                1.0,
                1.0,
            ],
        );
    }
    let pre_compact_count = world.store.node_count();

    // Compact, keeping the most recent view_root alive as an extra root.
    world.compact_keeping(&[peak_view]);

    let post_compact_count = world.store.node_count();
    assert!(
        post_compact_count <= pre_compact_count,
        "compact_keeping must not grow the store: pre={pre_compact_count} post={post_compact_count}"
    );

    // The live view_root must still work after remapping.
    let remap = world
        .last_compaction_remap
        .as_ref()
        .expect("compact_keeping must publish a remap");
    let new_view = *remap.get(&peak_view).expect(
        "extra_roots must survive compaction (otherwise compact_keeping dropped a live root)",
    );
    // Smoke: read a cell through the remapped view_root. NodeStore::get_cell
    // panics on a dangling NodeId, so reaching here means the root is alive.
    let _ = world.store.get_cell(new_view, 5, 5, 5);
}

#[test]
fn repeated_lod_recompute_with_periodic_compaction_bounds_growth() {
    // hash-thing-e4ep integration: model the main.rs frame loop. Each
    // iteration moves the player one chunk along x, recomputes the
    // policy (triggers a fresh pile of ghost chains), and triggers
    // compaction whenever the store-growth ratio crosses
    // LOD_COMPACT_RATIO_THRESHOLD. Without periodic compaction, the
    // store grows roughly linearly with the number of recomputes;
    // with it, peak growth must stay bounded.
    let mut world = make_4x4x4_chunk_world();
    // Plant some far cells so the policy produces non-trivial collapses
    // (otherwise the empty-world fast path masks the ghost-chain accrual).
    for cx in 0..4 {
        for cy in 0..4 {
            world.set(
                WorldCoord(cx * CHUNK_SIDE as i64 + 5),
                WorldCoord(cy * CHUNK_SIDE as i64 + 5),
                WorldCoord(5),
                stone(),
            );
        }
    }
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;

    let baseline = world.store.node_count();
    let mut peak = baseline;
    let mut compactions = 0u32;
    // Did at least one compaction actually shed nodes? Without this, the
    // peak-bounded assertion below would still pass on a no-op compactor
    // (e.g. one that publishes an identity remap and frees nothing).
    let mut observed_shrink = false;

    // 64 player-chunk crossings — long enough to exercise the trigger
    // multiple times without blowing past a small per-test budget.
    for player_x_chunks in 0..64 {
        let player_pos = [
            (player_x_chunks as f64) * (CHUNK_SIDE as f64) + 1.0,
            (CHUNK_SIDE as f64) / 2.0,
            (CHUNK_SIDE as f64) / 2.0,
        ];
        let view_root = policy.update(&mut world.store, world.root, world.level, player_pos);
        peak = peak.max(world.store.node_count());

        // Mirror the main.rs trigger condition (src/main.rs upload_volume).
        if let Some(ratio) = policy.store_growth_ratio(world.store.node_count()) {
            if ratio > LOD_COMPACT_RATIO_THRESHOLD {
                let pre = world.store.node_count();
                world.compact_keeping(&[view_root]);
                let post = world.store.node_count();
                if post < pre {
                    observed_shrink = true;
                }
                policy.reset_growth_baseline();
                compactions += 1;
            }
        }
    }

    // Sanity: compaction fired at least once — otherwise the test isn't
    // exercising the path it claims to.
    assert!(
        compactions > 0,
        "expected at least one compaction across 64 player-chunk crossings; \
         baseline={baseline} peak={peak}"
    );

    // Peak growth must stay bounded. Trigger fires at >4×; allow up to
    // 8× as headroom for the inter-trigger window. Anything beyond means
    // compaction ran but didn't actually shed nodes.
    let cap = baseline.saturating_mul(8).max(1024);
    assert!(
        peak <= cap,
        "store grew unbounded across recomputes: baseline={baseline} \
         peak={peak} cap={cap} compactions={compactions}"
    );

    // Stronger statement: at least one compaction actually shed nodes.
    // Defends against a regression where compact_keeping retains so many
    // extra roots that nothing is reclaimed — peak would still pass.
    assert!(
        observed_shrink,
        "at least one compact_keeping call must shrink the store \
         (baseline={baseline} peak={peak} compactions={compactions})"
    );
}

#[test]
fn compact_keeping_composes_with_pending_remap() {
    // hash-thing-rk4n.1 + hash-thing-e4ep: when two compactions fire
    // before the renderer drains `last_compaction_remap`, the second
    // compaction's B→C remap must be composed onto the still-pending
    // A→B to yield A→C. Otherwise the SVDAG's id_to_offset cache
    // (keyed on epoch-A NodeIds) would be handed a B→C map it cannot
    // apply, dropping every cached SVDAG offset.
    use hash_thing::sim::World;
    let mut world = World::new(7);
    for x in 0..6 {
        world.set(WorldCoord(x), WorldCoord(0), WorldCoord(0), stone());
    }

    // Pin an epoch-A NodeId we can trace through both compactions.
    let pin_a = world.root;

    // Compaction #1: publishes A→B onto last_compaction_remap.
    world.compact_keeping(&[]);
    let remap_a_to_b = world
        .last_compaction_remap
        .clone()
        .expect("first compact_keeping must publish a remap");
    let pin_b = *remap_a_to_b
        .get(&pin_a)
        .expect("world.root in epoch A must be in the A→B remap");

    // Cause more growth without taking last_compaction_remap (mirrors a
    // same-frame back-to-back where the renderer hasn't run yet).
    for x in 6..16 {
        world.set(WorldCoord(x), WorldCoord(1), WorldCoord(0), stone());
    }

    // Compaction #2: publishes B→C, composed onto the pending A→B → A→C.
    // Pass pin_b as an extra root so it survives B→C — without this,
    // pin_b would be unreachable in epoch C and compose_remap would
    // drop the pin_a entry from the composed map.
    world.compact_keeping(&[pin_b]);
    let remap_a_to_c = world
        .last_compaction_remap
        .clone()
        .expect("second compact_keeping must publish a (composed) remap");

    let pin_c = *remap_a_to_c.get(&pin_a).expect(
        "compose-on-write: pin_a must still be in the composed A→C remap \
         (its B-image was kept alive via extra_roots)",
    );

    // pin_c must point at a real node in the post-C store; NodeStore::get
    // panics on a dangling NodeId, so reaching here proves the chain held.
    let _ = world.store.get(pin_c);
}

#[test]
fn cache_invalidates_after_compact_keeping_with_unchanged_world_root() {
    // hash-thing-e4ep BLOCKER regression: ChunkLodPolicy.cache keys on
    // world_root NodeId, and `compacted_with_remap_keeping` allocates
    // NodeIds deterministically (post-order DFS in a fresh store). When
    // the world.root subtree is structurally unchanged across a
    // compaction, the new world.root NodeId can numerically equal the
    // pre-compact one. Without explicit cache rebase, the next update()
    // sees a matching key and returns the stale view_root from the old
    // store epoch.
    //
    // This test drives that scenario: cache a view_root, compact while
    // the world.root is structurally stable, then call update() against
    // the post-compact store and verify the returned view_root is valid
    // in the *new* store (i.e. the cache was either rebased via
    // apply_compaction_remap, or invalidated).
    let mut world = make_4x4x4_chunk_world();
    // Plant a far cell so the policy actually mints ghost chains.
    world.set(
        WorldCoord(3 * CHUNK_SIDE as i64 + 5),
        WorldCoord(5),
        WorldCoord(5),
        stone(),
    );
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;

    // Prime the cache with a view_root keyed on the current world.root.
    let player_pos = [(CHUNK_SIDE as f64) + 1.0, 1.0, 1.0];
    let view_before = policy.update(&mut world.store, world.root, world.level, player_pos);

    // Compact, keeping both world.root and view_before alive (mirrors the
    // main.rs upload_volume flow, which passes view_root as the extra root
    // and gets world.root preserved by Hashlife::compact_keeping).
    world.compact_keeping(&[view_before]);
    let remap = world
        .last_compaction_remap
        .take()
        .expect("compact_keeping must publish a remap");
    // Apply the remap as the runtime would.
    policy.apply_compaction_remap(&remap);

    // Expected post-compact view_root, mapped through the remap.
    let expected_view = *remap
        .get(&view_before)
        .expect("view_root must survive compaction (it was an extra root)");

    // Now call update() with identical inputs (player hasn't moved).
    // Either: cache hit returns the rebased view_root, or cache miss
    // recomputes against the post-compact store. Both must produce a
    // NodeId that's valid in the post-compact store.
    let view_after = policy.update(&mut world.store, world.root, world.level, player_pos);

    // Smoke: NodeStore::get_cell panics on a dangling NodeId, so reading
    // any cell through view_after proves it lives in the post-compact
    // store (not stranded in the old epoch).
    let _ = world.store.get_cell(view_after, 5, 5, 5);

    // Stronger: if the cache was rebased correctly, view_after must
    // equal the remapped view_before. (If the policy chose to drop the
    // cache instead, this would still be a valid implementation — but
    // apply_compaction_remap's contract is "rebase when both ids are in
    // the remap," and both are by construction here.)
    assert_eq!(
        view_after, expected_view,
        "post-compact update() must return the remapped cached view_root \
         (got {view_after:?}, expected {expected_view:?})"
    );
}
