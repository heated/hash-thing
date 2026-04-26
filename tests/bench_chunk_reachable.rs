//! Per-chunk reachable-node distribution at 256³, 1024³, 4096³ — measurement
//! input for the cswp memory-LOD policy (`hash-thing-cswp.8.1`,
//! `docs/perf/cswp-lod.md` §5.4).
//!
//! Times are not the point — the output is *per-chunk node-count
//! statistics* across chunk sizes 64³ / 128³ / 256³ on the seeded-terrain
//! scene. The reviewer reads the printed distribution + sharing column and
//! answers "what chunk size minimises per-chunk variance?" before
//! cswp.8.2-8.5.
//!
//! Read the `share×` column alongside the per-chunk distribution: the
//! per-chunk numbers are chunk-local upper bounds with cross-chunk sharing
//! charged in full to every chunk. If `share×` is high (say >2), an
//! exclusive-count follow-up may be needed before picking a chunk size.
//! See the `reachable_nodes_per_chunk` docstring on `NodeStore` for the
//! precise framing.
//!
//! Run with:
//! ```text
//! cargo test --profile bench --test bench_chunk_reachable -- --ignored --nocapture
//! ```

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct Row {
    chunk_level: u32,
    chunk_side: u64,
    n: usize,
    min: u64,
    mean: u64,
    median: u64,
    p95: u64,
    max: u64,
    sum_per_chunk: u64,
    global: u64,
}

fn measure(world: &World, chunk_level: u32) -> Row {
    let (chunks, global) = world.store.reachable_summary(world.root, chunk_level);
    let mut counts: Vec<u64> = chunks.iter().map(|&(_, n)| n).collect();
    counts.sort_unstable();
    let n = counts.len();
    let sum_per_chunk: u64 = counts.iter().sum();
    let mean = if n == 0 { 0 } else { sum_per_chunk / n as u64 };
    Row {
        chunk_level,
        chunk_side: 1u64 << chunk_level,
        n,
        min: counts.first().copied().unwrap_or(0),
        mean,
        median: percentile(&counts, 0.5),
        p95: percentile(&counts, 0.95),
        max: counts.last().copied().unwrap_or(0),
        sum_per_chunk,
        global,
    }
}

fn print_summary(label: &str, root_level: u32, rows: &[Row]) {
    eprintln!("--- {label} (root_level={root_level}) ---");
    if let Some(global) = rows.first().map(|r| r.global) {
        eprintln!("  global reachable: {global}");
    }
    eprintln!(
        "  {:>4} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>13} {:>9}",
        "lvl", "side", "chunks", "min", "mean", "median", "p95", "max", "sum_per_chunk", "share×",
    );
    for r in rows {
        let share = r.sum_per_chunk as f64 / r.global.max(1) as f64;
        eprintln!(
            "  {:>4} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>13} {:>9.2}",
            r.chunk_level,
            r.chunk_side,
            r.n,
            r.min,
            r.mean,
            r.median,
            r.p95,
            r.max,
            r.sum_per_chunk,
            share,
        );
    }
    eprintln!();
}

fn run_scene(label: &str, level: u32, chunk_levels: &[u32]) {
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    let _ = world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    let rows: Vec<Row> = chunk_levels
        .iter()
        .copied()
        .filter(|&cl| cl > 0 && cl <= level)
        .map(|cl| measure(&world, cl))
        .collect();
    print_summary(label, level, &rows);
}

/// 256³ fast-iteration target — sub-second on M1-class hardware. Sanity
/// floor for cswp.8.2 / 8.3 iteration; NOT a substitute for the 4096³
/// chunk-size pick.
#[test]
#[ignore]
fn bench_chunk_reachable_256() {
    run_scene("256³", 8, &[4, 5, 6]);
}

/// 1024³, chunk sizes 64³ / 128³ / 256³.
#[test]
#[ignore]
fn bench_chunk_reachable_1024() {
    run_scene("1024³", 10, &[6, 7, 8]);
}

/// 4096³ — design target. Per `docs/perf/cswp-lod.md` §5.4 the chunk-size
/// pick must be validated at this scale; chunk-sharing behavior can change
/// materially with world size, so neither 256³ nor 1024³ alone is enough
/// input for that decision. Cold-gen + walker fits under the 60 s
/// soft-max-command-seconds budget on M1-class hardware (cold-gen ≈ 4 s,
/// walker ≈ 1-2 s on top); record observed times in the bench output if
/// running on a smaller box.
#[test]
#[ignore]
fn bench_chunk_reachable_4096() {
    run_scene("4096³", 12, &[6, 7, 8]);
}
