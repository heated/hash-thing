//! Verification harness for `hash-thing-cswp.3`: confirms gen-time hash-cons
//! is in.
//!
//! cswp.3 was originally filed expecting "≥10× faster than cswp.1's baseline,"
//! but cswp.1's baseline (236 ms @ 1024³, 3.7 s @ 4096³) IS the hash-cons-on
//! number — the 10–50× speedup target was anchored to a stale 25 s analytical
//! estimate (`hash-thing-stue.2`). Recording the dedup factor + the
//! same-seed-same-root identity is the verification this bead actually needs.
//!
//! Two assertions per scale:
//!  1. **Identity**: building twice with the same seed produces identical root
//!     `NodeId`. Proves structural intern at gen time is consistent.
//!  2. **Compression**: voxels/nodes ratio (a lower bound on dedup factor) is
//!     large on noise terrain.
//!
//! Run with:
//! ```text
//! cargo test --profile perf --test verify_gen_hash_cons -- --ignored --nocapture
//! ```
//! The 64³ case runs unignored as a quick CI guard.

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;

/// Build the world twice with the same level + scale-derived params and
/// return `(voxels, nodes_after_gen, dedup_ratio, same_root)`.
fn dedup_check(label: &str, level: u32) -> (u64, u64, u64) {
    let side = 1u64 << level;
    let voxels = side
        .checked_mul(side)
        .and_then(|s2| s2.checked_mul(side))
        .expect("level fits in u64 voxel count");

    let mut a = World::new(level);
    let params = TerrainParams::for_level(level);
    let stats_a = a
        .seed_terrain(&params)
        .expect("level-derived terrain params validate");

    let mut b = World::new(level);
    let stats_b = b
        .seed_terrain(&params)
        .expect("level-derived terrain params validate");

    assert_eq!(
        a.root, b.root,
        "[{label}] gen-twice-same-seed must produce identical root NodeId — gen-time hash-cons identity broken",
    );
    assert_eq!(
        stats_a.nodes_after_gen, stats_b.nodes_after_gen,
        "[{label}] node count must match across two gen runs — non-deterministic intern",
    );
    assert_eq!(
        a.population(),
        b.population(),
        "[{label}] population must match across two gen runs",
    );

    let nodes = stats_a.nodes_after_gen as u64;
    let dedup = if nodes > 0 { voxels / nodes } else { 0 };
    eprintln!(
        "[{label}] level={level}, side={side}^3, voxels={voxels}, nodes={nodes}, dedup={dedup}x, pop={}, root={:?}",
        a.population(),
        a.root,
    );
    (voxels, nodes, dedup)
}

/// Quick CI guard at 64³ (262 144 voxels).
///
/// Identity-of-root is the load-bearing check; the dedup floor is set very
/// conservatively because small worlds offer less repetition than large ones.
#[test]
fn cswp_3_gen_hash_cons_identity_64() {
    let (_, nodes, dedup) = dedup_check("64³", 6);
    assert!(nodes > 0, "64³ seed must produce non-zero nodes");
    assert!(
        dedup >= 4,
        "expected >=4x voxel/node ratio on 64³ noise; got {dedup}",
    );
}

/// cswp.3 acceptance scale.
///
/// Per cswp.1, 1024³ produces ~56k nodes for ~10⁹ voxels (≈19 000× ratio).
/// 1000× is comfortably below that and large enough to detect a regression.
#[test]
#[ignore]
fn cswp_3_gen_hash_cons_dedup_1024() {
    let (_, _, dedup) = dedup_check("1024³", 10);
    assert!(
        dedup >= 1000,
        "expected >=1000x voxel/node ratio on 1024³ noise; got {dedup}",
    );
}

/// 4096³ scale — the streaming target. Per cswp.1, ~548k nodes for ~6.87×10¹⁰
/// voxels (≈125 000× ratio).
#[test]
#[ignore]
fn cswp_3_gen_hash_cons_dedup_4096() {
    let (_, _, dedup) = dedup_check("4096³", 12);
    assert!(
        dedup >= 10_000,
        "expected >=10000x voxel/node ratio on 4096³ noise; got {dedup}",
    );
}
