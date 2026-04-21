//! Per-ray SVDAG traversal-step histogram at 256³ (hash-thing-stue.5).
//!
//! Validates the "~4 steps per ray average" assumption in
//! `docs/perf/svdag-perf-paper.md` §3.3 empirically. For each of three
//! representative camera poses, shoots 960×540 primary rays through the
//! maintained CPU replica of the shader (`ht_render::cpu_trace::raycast`)
//! and reports mean / p50 / p95 / max step counts plus a logarithmic
//! bucket histogram. Hits and misses are reported separately because
//! §3.3 makes different claims for sky rays (exit at depth 1-2) versus
//! terrain rays (4-6).
//!
//! Why CPU and not GPU: `cpu_trace::raycast` is 1:1 with
//! `svdag_raycast.wgsl`, pinned byte-aligned by tests in `svdag::tests`
//! and the `wgsl_drift_guard` suite. `TraceResult.steps` is incremented
//! in the same inner DDA loop as the shader's `step`. For counting
//! steps, CPU and GPU agree. Avoids shader plumbing / readback.
//!
//! Run with:
//! ```
//! cargo test --release --test bench_depth_histogram -- --ignored --nocapture
//! ```
//!
//! ## Camera poses
//!
//! Canonical default-spawn pose mirrors `bench_gpu_raycast::BenchCamera::
//! app_spawn`:
//!
//! ```ignore
//! yaw   = π/4     // 45° around +Y
//! pitch = 0
//! pos   = [0.5, 0.5 + 2.0/side, 0.5]  // 2 voxels above center
//! ```
//!
//! The screen math mirrors `svdag_raycast.wgsl::cs_main` exactly:
//! - `uv.x = (x + 0.5) / W * 2 - 1`, `uv.y = 1 - (y + 0.5) / H * 2`
//! - `rd = normalize(dir + right * uv.x * aspect * fov_tan + up * uv.y * fov_tan)`
//! - `fov_tan = tan(π/8)` (crates/ht-render/src/renderer.rs:1662)
//! - `aspect = W / H` (crates/ht-render/src/renderer.rs:1661)
//! - 960×540 = 1920×1080 × 50% render scale (app default)

use hash_thing::render::{cpu_trace, Svdag};
use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;

const WIDTH: usize = 960;
const HEIGHT: usize = 540;
const LEVEL: u32 = 8;

#[derive(Clone, Copy)]
struct Camera {
    pos: [f32; 3],
    dir: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
    label: &'static str,
}

fn camera_from_yaw_pitch(label: &'static str, pos: [f32; 3], yaw: f32, pitch: f32) -> Camera {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let dir = [-cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw];
    let right = [cos_yaw, 0.0, -sin_yaw];
    let up = [
        right[1] * dir[2] - right[2] * dir[1],
        right[2] * dir[0] - right[0] * dir[2],
        right[0] * dir[1] - right[1] * dir[0],
    ];
    Camera {
        pos,
        dir,
        up,
        right,
        label,
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

struct PoseStats {
    hit_steps: Vec<u32>,
    miss_steps: Vec<u32>,
    exhausted: u32,
}

fn summarize(samples: &[u32]) -> (f64, u32, u32, u32) {
    if samples.is_empty() {
        return (0.0, 0, 0, 0);
    }
    let sum: u64 = samples.iter().map(|&s| s as u64).sum();
    let mean = sum as f64 / samples.len() as f64;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
    let max = *sorted.last().unwrap();
    (mean, p50, p95, max)
}

fn histogram(samples: &[u32]) -> [u64; 11] {
    let edges: [u32; 10] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256];
    let mut buckets = [0u64; 11];
    for &s in samples {
        let mut placed = false;
        for (i, &edge) in edges.iter().enumerate().rev() {
            if s >= edge {
                buckets[i + 1] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            buckets[0] += 1;
        }
    }
    buckets
}

fn print_histogram(label: &str, samples: &[u32]) {
    if samples.is_empty() {
        eprintln!("  {label}: (no samples)");
        return;
    }
    let buckets = histogram(samples);
    let (mean, p50, p95, max) = summarize(samples);
    eprintln!(
        "  {label}: n={}, mean={:.2}, p50={}, p95={}, max={}",
        samples.len(),
        mean,
        p50,
        p95,
        max,
    );
    let edges: [&str; 11] = [
        "[<0]",
        "[0]",
        "[1]",
        "[2-3]",
        "[4-7]",
        "[8-15]",
        "[16-31]",
        "[32-63]",
        "[64-127]",
        "[128-255]",
        "[256+]",
    ];
    for (edge, count) in edges.iter().zip(buckets.iter()) {
        if *count > 0 {
            let pct = *count as f64 / samples.len() as f64 * 100.0;
            eprintln!("    {edge:>10} {count:>8}  {pct:5.1}%");
        }
    }
}

fn run_pose(dag: &[u32], root_level: u32, cam: &Camera) -> PoseStats {
    assert!(
        cam.pos[0] >= 0.0
            && cam.pos[0] <= 1.0
            && cam.pos[1] >= 0.0
            && cam.pos[1] <= 1.0
            && cam.pos[2] >= 0.0
            && cam.pos[2] <= 1.0,
        "camera must sit inside the [0,1]^3 unit cube (pose drift sentinel): {:?}",
        cam.pos
    );

    let aspect = WIDTH as f32 / HEIGHT as f32;
    let fov_tan = (std::f32::consts::FRAC_PI_4 / 2.0).tan();
    let inv_w = 1.0 / WIDTH as f32;
    let inv_h = 1.0 / HEIGHT as f32;

    let mut hit_steps = Vec::with_capacity(WIDTH * HEIGHT / 2);
    let mut miss_steps = Vec::with_capacity(WIDTH * HEIGHT / 2);
    let mut exhausted = 0u32;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let uvx = (x as f32 + 0.5) * inv_w * 2.0 - 1.0;
            let uvy = 1.0 - (y as f32 + 0.5) * inv_h * 2.0;
            let rd_raw = [
                cam.dir[0] + cam.right[0] * uvx * aspect * fov_tan + cam.up[0] * uvy * fov_tan,
                cam.dir[1] + cam.right[1] * uvx * aspect * fov_tan + cam.up[1] * uvy * fov_tan,
                cam.dir[2] + cam.right[2] * uvx * aspect * fov_tan + cam.up[2] * uvy * fov_tan,
            ];
            let rd = normalize(rd_raw);
            let result = cpu_trace::raycast(dag, root_level, cam.pos, rd, false);
            if result.exhausted {
                exhausted += 1;
                continue;
            }
            let steps = result.steps as u32;
            if result.hit_cell.is_some() {
                hit_steps.push(steps);
            } else {
                miss_steps.push(steps);
            }
        }
    }

    PoseStats {
        hit_steps,
        miss_steps,
        exhausted,
    }
}

#[test]
#[ignore]
fn depth_histogram_256() {
    eprintln!("--- traversal-step histogram @ 256³, {WIDTH}×{HEIGHT} rays/pose ---");

    let mut world = World::new(LEVEL);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(LEVEL))
        .expect("level-derived terrain params must validate");
    eprintln!("world population: {}", world.population());

    let svdag = Svdag::build(&world.store, world.root, world.level);
    eprintln!(
        "SVDAG: {} nodes, {:.2} MB",
        svdag.node_count,
        (svdag.nodes.len() * 4) as f64 / (1024.0 * 1024.0),
    );

    let side = (1u32 << LEVEL) as f32;
    let spawn_y = 0.5 + 2.0 / side;
    let poses = [
        camera_from_yaw_pitch(
            "default-spawn (yaw=π/4, pitch=0)",
            [0.5, spawn_y, 0.5],
            std::f32::consts::FRAC_PI_4,
            0.0,
        ),
        camera_from_yaw_pitch(
            "looking-down (yaw=π/4, pitch=+π/2)",
            [0.5, spawn_y, 0.5],
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_2,
        ),
        camera_from_yaw_pitch(
            "horizontal-mid (yaw=0, pitch=0, y=0.5)",
            [0.5, 0.5, 0.5],
            0.0,
            0.0,
        ),
    ];

    let t0 = std::time::Instant::now();
    let mut exhausted_by_pose: Vec<(&str, u32)> = Vec::new();
    let mut mean_drift: Vec<(&str, f64, f64)> = Vec::new();
    for cam in &poses {
        eprintln!();
        eprintln!("pose: {}", cam.label);
        let stats = run_pose(&svdag.nodes, svdag.root_level, cam);
        let total = stats.hit_steps.len() + stats.miss_steps.len() + stats.exhausted as usize;
        let hit_ratio = stats.hit_steps.len() as f64 / total as f64 * 100.0;
        eprintln!(
            "  rays: {} total, hit={} ({:.1}%), miss={}, exhausted={}",
            total,
            stats.hit_steps.len(),
            hit_ratio,
            stats.miss_steps.len(),
            stats.exhausted,
        );
        print_histogram("hit ", &stats.hit_steps);
        print_histogram("miss", &stats.miss_steps);
        let combined: Vec<u32> = stats
            .hit_steps
            .iter()
            .chain(stats.miss_steps.iter())
            .copied()
            .collect();
        print_histogram("all ", &combined);
        exhausted_by_pose.push((cam.label, stats.exhausted));

        // Order-of-magnitude mean-steps floor per pose (hash-thing-v2d1). The
        // [0,1]^3 sentinel above catches gross position drift, but a silent
        // rewrite of yaw/pitch/FOV could keep the camera inside the cube while
        // invalidating the sample. Thresholds are ~50% of the 2026-04-20
        // measured means (15.0 / 6.8 / 13.2) -- low enough to survive terrain
        // tweaks, high enough to notice an order-of-magnitude regression.
        let expected_min = expected_min_mean(cam.label);
        let (mean, _, _, _) = summarize(&combined);
        if mean < expected_min {
            mean_drift.push((cam.label, mean, expected_min));
        }
    }

    let elapsed_s = t0.elapsed().as_secs_f64();
    eprintln!();
    eprintln!(
        "traced {} rays × {} poses in {:.2}s ({:.0} rays/s)",
        WIDTH * HEIGHT,
        poses.len(),
        elapsed_s,
        (WIDTH * HEIGHT * poses.len()) as f64 / elapsed_s,
    );

    let bad: Vec<&(&str, u32)> = exhausted_by_pose.iter().filter(|(_, n)| *n > 0).collect();
    assert!(
        bad.is_empty(),
        "exhausted rays would poison the histogram — step_budget is too low \
         or the scene hit a pathological case. Per-pose counts: {bad:?}"
    );

    assert!(
        mean_drift.is_empty(),
        "mean-steps floor violated — likely silent drift in BenchCamera::app_spawn \
         yaw/pitch/FOV or the screen math in run_pose. Per-pose (label, measured, floor): \
         {mean_drift:?}"
    );
}

fn expected_min_mean(label: &str) -> f64 {
    if label.starts_with("default-spawn") {
        10.0
    } else if label.starts_with("looking-down") {
        3.0
    } else if label.starts_with("horizontal-mid") {
        8.0
    } else {
        0.0
    }
}
