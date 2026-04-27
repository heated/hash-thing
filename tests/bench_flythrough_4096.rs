//! Fly-through 4096³ acceptance bench for the cswp.8.3 chunk-LOD policy
//! (`hash-thing-cswp.8.5`).
//!
//! Walks a deterministic camera trajectory through a seeded 4096³ world and,
//! for each `lod_bias` value, reports per-frame `ChunkLodPolicy::update` cost
//! plus resident `NodeStore` shape. The numbers feed `docs/perf/svdag-perf-paper.md`
//! §4.7.5a; the bench itself enforces only loose correctness invariants
//! (band coverage, growth tripwire, runaway-detection backstop) so per-frame
//! perf variance does not false-fire it.
//!
//! Run with:
//! ```text
//! cargo test --profile bench --test bench_flythrough_4096 -- --ignored --nocapture
//! ```
//!
//! ## What this bench does NOT measure
//!
//! - **CA-driven cache growth.** `world.step()` is not called, because a
//!   single warm step at 4096³ exceeds the project's
//!   `soft-max-command-seconds: 60` budget (`docs/perf/svdag-perf-paper.md`
//!   §4.7.2a). Consequently `macro_cache_bytes_est()` reads as 0 — that is
//!   not "memory-LOD doesn't reduce memory," it is "this bench did not run
//!   the simulation."
//! - **Per-frame cost vs the cswp.8.3.1 single-pass-recursive-descent
//!   optimization.** The bench reports the current chained-recompute cost;
//!   cswp.8.3.1 measures and lands the optimization. dt_ms here is a
//!   pre-optimization baseline.
//!
//! ## LOD-4 reachability at 4096³ / chunk_level=7
//!
//! Max chunk Chebyshev radius in a 32-chunk-per-axis world is 31. With
//! `target_lod_for_radius` band edges at scaled-radius {1, 4, 16, 64} (per
//! `src/sim/chunks.rs:80`), the LOD-4 boundary at biases {0.5, 1.0, 2.0}
//! sits at raw radius {32, 64, 128} — all ≥ max-radius. So the
//! bead-required biases exercise LOD bands 0..=3 only. Bias=0.25 is
//! included as a diagnostic so bench output validates the LOD-4 collapse
//! path (band edge at raw radius 16, comfortably reachable). This is a
//! curve-vs-world-size property of the design, not a bench limitation.

use hash_thing::sim::chunks::{ChunkLodPolicy, CHUNK_LEVEL};
use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::Instant;

/// Number of trajectory samples per `lod_bias`. Each sample is one call to
/// `policy.update`. Kept small (8) because per-frame cost at 4096³ ranges
/// 1–35 s pre-cswp.8.3.1 and grows monotonically with sample index (the
/// policy *adds* scaffold nodes each frame; later frames walk a larger
/// store). 8 samples × 4 biases × ~30 s mean ≈ 16 min wall-time, within
/// reach of one bench run; a TRAJ_STEPS of 20 produced ≈ 90 min projected
/// wall-time and trips the runaway-detection backstop in pilot runs.
const TRAJ_STEPS: usize = 8;

/// Trajectory anchor — chunk (0, 0, 0). Pinning the camera at one corner
/// keeps the *opposite* corner at the maximum sustainable Chebyshev radius
/// across the entire walk, so the LOD curve's outermost reachable band is
/// exercised every frame instead of fading in mid-trajectory.
const TRAJ_ANCHOR_CHUNK: u32 = 0;

/// Furthest chunk the camera reaches. 31 = `(1 << (12 - CHUNK_LEVEL)) - 1`,
/// the last valid chunk index in a 4096³ / chunk_level-7 world.
const TRAJ_FAR_CHUNK: u32 = 31;

/// `lod_bias` values from the bead description. 0.25 is added as a
/// diagnostic data-point to exercise the LOD-4 collapse path; see the
/// docstring's "LOD-4 reachability" note.
const LOD_BIASES: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

/// Hard ceiling on per-frame `policy.update` cost — runaway-detection
/// backstop only. cswp.8.3.1 owns the actual perf number. Set well above
/// the observed 1–35 s pre-optimization band so the bench produces data
/// rather than panicking; ~5 min trips only on a true runaway, not on
/// known-slow frames at high bias / late trajectory steps.
const DT_RUNAWAY_MS: f64 = 300_000.0;

/// Soft warning threshold; bench prints a `[slow-frame]` line when crossed
/// but does not fail. Reviewers reading the output decide whether the
/// number is acceptable for this iteration.
const DT_WARN_MS: f64 = 1_000.0;

/// Tripwire matching cswp.8.3's own warn-once growth threshold
/// (`src/main.rs` periodic log). Above 4× store-node growth indicates the
/// policy is pathologically scaffolding rather than collapsing.
const GROWTH_TRIPWIRE: f32 = 4.0;

struct FrameSample {
    dt_ms: f64,
    node_count: usize,
    hist: [u32; 5],
    growth: Option<f32>,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Linear sweep from chunk anchor to chunk far_chunk, in *cell* coords at
/// the chunk-center to keep band-edge crossings deterministic.
fn trajectory_pos(step: usize) -> [f64; 3] {
    let chunk_side = (1u32 << CHUNK_LEVEL) as f64;
    let center = chunk_side / 2.0;
    let denom = (TRAJ_STEPS - 1) as f64;
    let t = step as f64 / denom;
    let chunk_axis =
        TRAJ_ANCHOR_CHUNK as f64 + t * (TRAJ_FAR_CHUNK as f64 - TRAJ_ANCHOR_CHUNK as f64);
    let cell = chunk_axis * chunk_side + center;
    [cell, cell, cell]
}

fn run_bias(world: &mut World, bias: f32, total_chunks: u32) -> Vec<FrameSample> {
    let mut policy = ChunkLodPolicy::new();
    policy.enabled = true;
    policy.lod_bias = bias;

    let mut samples = Vec::with_capacity(TRAJ_STEPS);

    eprintln!("--- lod_bias = {bias:.2} ---");
    eprintln!(
        "  {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>30} {:>10}",
        "step", "pos_x", "pos_y", "pos_z", "dt_ms", "nodes", "hist[0..5]", "growth",
    );

    for step in 0..TRAJ_STEPS {
        let pos = trajectory_pos(step);
        let t0 = Instant::now();
        let _view = policy.update(&mut world.store, world.root, world.level, pos);
        let dt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let node_count = world.store.node_count();
        let hist = policy.lod_histogram();
        let growth = policy.store_growth_ratio(node_count);

        // Sanity: the policy walks every chunk each recompute.
        let hist_sum: u32 = hist.iter().sum();
        assert_eq!(
            hist_sum, total_chunks,
            "step {step}: hist sums to {hist_sum}, expected {total_chunks}",
        );
        // Tripwire: cswp.8.3 warn-once threshold. Finite-only so the
        // baseline-zero path (Some(infinity)) gets a named failure.
        if let Some(g) = growth {
            assert!(
                g.is_finite(),
                "step {step}: growth_ratio = {g} (baseline node_count was 0?)",
            );
            assert!(
                g <= GROWTH_TRIPWIRE,
                "step {step}: growth {g}× exceeds {GROWTH_TRIPWIRE}× tripwire — \
                 policy is scaffolding rather than collapsing",
            );
        }
        // Runaway-detection backstop only.
        assert!(
            dt_ms < DT_RUNAWAY_MS,
            "step {step}: dt_ms {dt_ms} exceeds runaway ceiling {DT_RUNAWAY_MS} ms",
        );

        let warn_marker = if dt_ms > DT_WARN_MS {
            "[slow-frame] "
        } else {
            ""
        };
        eprintln!(
            "  {warn_marker}{:>4} {:>10.1} {:>10.1} {:>10.1} {:>10.2} {:>10} \
             [{:>5}/{:>5}/{:>5}/{:>5}/{:>5}] {:>9}",
            step,
            pos[0],
            pos[1],
            pos[2],
            dt_ms,
            node_count,
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            hist[4],
            growth
                .map(|g| format!("{g:.2}x"))
                .unwrap_or_else(|| "—".to_string()),
        );

        samples.push(FrameSample {
            dt_ms,
            node_count,
            hist,
            growth,
        });
    }

    let mut dts: Vec<f64> = samples.iter().map(|s| s.dt_ms).collect();
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dt_min = *dts.first().unwrap();
    let dt_max = *dts.last().unwrap();
    let dt_mean: f64 = dts.iter().sum::<f64>() / dts.len() as f64;
    let dt_p95 = percentile(&dts, 0.95);
    let nodes_min = samples.iter().map(|s| s.node_count).min().unwrap();
    let nodes_max = samples.iter().map(|s| s.node_count).max().unwrap();
    let growth_max = samples
        .iter()
        .filter_map(|s| s.growth)
        .fold(f32::NEG_INFINITY, f32::max);
    eprintln!("  summary: dt_ms min={dt_min:.2} mean={dt_mean:.2} p95={dt_p95:.2} max={dt_max:.2}");
    eprintln!("           nodes min={nodes_min} max={nodes_max} growth_max={growth_max:.2}x",);
    eprintln!();

    samples
}

/// Fly-through bench at 4096³, all four lod_bias values. Shared world; the
/// cold-gen amortizes once across biases. Wall-time is dominated by the
/// per-frame `policy.update` cost — see the docstring's "What this bench
/// does NOT measure" note.
#[test]
#[ignore]
fn bench_flythrough_4096() {
    let level = 12u32;
    eprintln!("=== fly-through 4096³ (level={level}) ===");
    eprintln!(
        "  chunk_level={CHUNK_LEVEL} traj_steps={TRAJ_STEPS} \
               anchor_chunk={TRAJ_ANCHOR_CHUNK} far_chunk={TRAJ_FAR_CHUNK}"
    );
    eprintln!("  biases={LOD_BIASES:?}");
    eprintln!();

    let t_gen = Instant::now();
    let mut world = World::new(level);
    assert_eq!(world.level, level, "world level pin");
    let params = TerrainParams::for_level(level);
    let _ = world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;
    let nodes_after_gen = world.store.node_count();
    eprintln!("  cold-gen: {gen_ms:.1} ms, nodes={nodes_after_gen}");
    eprintln!();

    let chunks_per_axis: u32 = 1u32 << (level - CHUNK_LEVEL);
    assert_eq!(
        chunks_per_axis,
        TRAJ_FAR_CHUNK + 1,
        "TRAJ_FAR_CHUNK must equal chunks_per_axis - 1",
    );
    let total_chunks: u32 = chunks_per_axis.pow(3);

    let mid_step = TRAJ_STEPS / 2;

    // Per-bias rollup row: (bias, dt_mean, dt_max, nodes_max, growth_max,
    // hist_at_mid_traj).
    let mut rollups: Vec<(f32, f64, f64, usize, f32, [u32; 5])> = Vec::new();
    for &bias in LOD_BIASES.iter() {
        let samples = run_bias(&mut world, bias, total_chunks);

        // Band-coverage invariants per the LOD-4-reachability docstring
        // note: at biases ≥ 0.5 the LOD-4 band is unreachable. Pin it as a
        // documented invariant so a future curve change trips loudly.
        if bias >= 0.5 {
            for s in &samples {
                assert_eq!(
                    s.hist[4], 0,
                    "bias {bias}: hist[4] must be 0 (LOD-4 band edge ≥ {} cells, > 4096³ extent), got hist={:?}",
                    32u32 << CHUNK_LEVEL,
                    s.hist,
                );
            }
        } else {
            // bias 0.25 should reach LOD 4 at the far corner.
            let any_lod4 = samples.iter().any(|s| s.hist[4] > 0);
            assert!(
                any_lod4,
                "bias {bias}: expected LOD-4 collapse at far corner, none observed",
            );
        }

        let dts: Vec<f64> = samples.iter().map(|s| s.dt_ms).collect();
        let dt_mean = dts.iter().sum::<f64>() / dts.len() as f64;
        let dt_max = dts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let nodes_max = samples.iter().map(|s| s.node_count).max().unwrap();
        let growth_max = samples
            .iter()
            .filter_map(|s| s.growth)
            .fold(f32::NEG_INFINITY, f32::max);
        let hist_mid = samples[mid_step].hist;
        rollups.push((bias, dt_mean, dt_max, nodes_max, growth_max, hist_mid));
    }

    eprintln!("--- summary across biases (hist column = step {mid_step} of {TRAJ_STEPS}) ---");
    eprintln!(
        "  {:>5} {:>9} {:>9} {:>10} {:>10} {:>30}",
        "bias", "dt_mean", "dt_max", "nodes_max", "growth_max", "hist_at_mid_traj[0..5]",
    );
    for (bias, dt_mean, dt_max, nodes_max, growth_max, hist_mid) in rollups {
        eprintln!(
            "  {bias:>5.2} {dt_mean:>9.2} {dt_max:>9.2} {nodes_max:>10} {growth_max:>9.2}x \
             [{:>5}/{:>5}/{:>5}/{:>5}/{:>5}]",
            hist_mid[0], hist_mid[1], hist_mid[2], hist_mid[3], hist_mid[4],
        );
    }
}
