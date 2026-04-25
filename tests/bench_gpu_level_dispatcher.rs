//! GPU breadth-first level dispatcher PoC (hash-thing-abwm.3).
//!
//! Validates that a hashlife-style recursive step can run as a chain of
//! per-level GPU compute dispatches, mirroring the architecture sketched in
//! `docs/perf/svdag-perf-paper.md` §8.3. The dispatcher is exercised on a
//! toy CA rule (3D Life Bays-5766: B{5,6,7} / S{5,6}) so that the spike's
//! conclusion does not depend on the production CaRule + BlockRule path.
//!
//! Spike scope (per `.ship-notes/plan-spark-hash-thing-abwm.3-gpu-level-dispatcher.md`):
//! - Toy rule only. No Margolus, no parity rotation, no schedule_phase.
//! - World root level 4 (16³). One recursive level above the level-3 base.
//! - GPU still derives the 27-intermediate / 8-subcube structure from
//!   uploaded child topology. CPU pre-allocates flat NodeId slots only.
//! - Hash-table primitive copy-pasted from `tests/bench_gpu_hash_table.rs`
//!   (abwm.2). Refactor to a shared module deferred to phase 4.
//!
//! Kernel-internal ordering rule (READ BEFORE EDITING SHADERS):
//! - Every kernel that probes the memo MUST early-return on
//!   `population == 0 → empty leaf` BEFORE the memo probe. Key 0 is the
//!   hash-table empty sentinel (and the canonical `NodeId::EMPTY`); a
//!   key-0 lookup spuriously hits the empty-slot path. Mirror the CPU
//!   short-circuit at `src/sim/hashlife.rs:474`.
//! - Insert phase of dispatch N writes via atomicCompareExchangeWeak;
//!   lookup phase of dispatch N+1 reads the same slots. The atomic
//!   visibility relies on the `queue.submit` boundary between dispatches
//!   — see the cross-dispatch comment in
//!   `tests/bench_gpu_hash_table.rs::cs_insert`. DO NOT fuse insert and
//!   lookup into a single dispatch without adding explicit synchronization.
//!
//! Default `cargo test` runs only the base-case correctness sub-tests
//! (fast, no `--ignored`). The full level-4 dispatcher chain + throughput
//! batch is `--ignored`:
//! ```
//! cargo test --profile bench --test bench_gpu_level_dispatcher -- --ignored --nocapture
//! ```

use std::time::Instant;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Toy rule: 3D Life Bays-5766 (B{5,6,7} / S{5,6}).
//
// Cell ∈ {0, 1}. 1-cell radius (3³ = 27 cell stencil including self).
// Boundary: zero-padded. The base case reads an 8³ block and produces a
// 4³ center result (any cell within 1 of the input boundary cannot be
// evaluated; hashlife shrinks each step by 1 cell of border).
//
// Rule encoded as bitmasks over alive-neighbor count (range 0..=26):
// bit i set ⇒ count i triggers birth (when dead) or survival (when alive).
// ---------------------------------------------------------------------------

/// Birth on alive-neighbor counts ∈ {5, 6, 7}.
const BAYS_BIRTH_MASK: u32 = (1 << 5) | (1 << 6) | (1 << 7);
/// Survival on alive-neighbor counts ∈ {5, 6}.
const BAYS_SURVIVE_MASK: u32 = (1 << 5) | (1 << 6);

/// Apply Bays-5766 to a single cell given (current, alive-neighbor-count).
fn bays_step_cell(current: u32, alive_neighbors: u32) -> u32 {
    debug_assert!(alive_neighbors <= 26, "alive_neighbors out of range");
    if current == 1 {
        (BAYS_SURVIVE_MASK >> alive_neighbors) & 1
    } else {
        (BAYS_BIRTH_MASK >> alive_neighbors) & 1
    }
}

// ---------------------------------------------------------------------------
// CPU base case: 8³ input → 4³ center result (level 3 → level 2 in the
// hashlife sense). Mirrors `src/sim/hashlife.rs::step_base_case` but uses
// the toy rule instead of CaRule + BlockRule.
//
// Indexing convention: linear index = z * 64 + y * 8 + x  (x fastest).
// The 4³ result occupies the center: input cells with 2 ≤ x,y,z < 6
// produce result cells at (x-2, y-2, z-2). Cells with x,y,z ∈ {0, 1, 6, 7}
// are border cells whose stencil can't be fully evaluated; those cells
// contribute neighbor counts to interior cells via the standard
// 1-cell-padded read but are not themselves output.
// ---------------------------------------------------------------------------

const L3_SIDE: usize = 8;
const L3_CELLS: usize = L3_SIDE * L3_SIDE * L3_SIDE;
const L2_SIDE: usize = 4;
const L2_CELLS: usize = L2_SIDE * L2_SIDE * L2_SIDE;

fn cpu_base_case(input: &[u32; L3_CELLS]) -> [u32; L2_CELLS] {
    let mut out = [0u32; L2_CELLS];
    for cz in 0..L2_SIDE {
        for cy in 0..L2_SIDE {
            for cx in 0..L2_SIDE {
                let ix = cx + 2;
                let iy = cy + 2;
                let iz = cz + 2;
                let cur = input[iz * L3_SIDE * L3_SIDE + iy * L3_SIDE + ix];
                let mut alive = 0u32;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nx = ix as i32 + dx;
                            let ny = iy as i32 + dy;
                            let nz = iz as i32 + dz;
                            // 8³ block bounds (zero padding implicit because
                            // we only call this with cx,cy,cz ∈ [0,4) which
                            // means the read is always inside the block,
                            // but we keep the bounds check for clarity).
                            if nx < 0
                                || ny < 0
                                || nz < 0
                                || nx >= L3_SIDE as i32
                                || ny >= L3_SIDE as i32
                                || nz >= L3_SIDE as i32
                            {
                                continue;
                            }
                            alive += input[(nz as usize) * L3_SIDE * L3_SIDE
                                + (ny as usize) * L3_SIDE
                                + nx as usize];
                        }
                    }
                }
                out[cz * L2_SIDE * L2_SIDE + cy * L2_SIDE + cx] = bays_step_cell(cur, alive);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GPU plumbing — forked from `tests/bench_gpu_hash_table.rs` (abwm.2).
// Refactor to a shared module deferred to phase 4 per spike-scope decision.
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 64;

/// Base-case kernel: one thread per (input-NodeId × 64 result cells)
/// — actually one thread per (input-NodeId × output-cell) for clean 64-cell
/// workgroup utilization. Each thread reads its stencil from the upload
/// buffer and writes one cell of the 4³ result.
///
/// Bindings:
///   0: Params { num_inputs: u32, _pad: 3 × u32 }
///   1: input_cells   storage<read>      array<u32>  // len = num_inputs * 512
///   2: output_cells  storage<read_write> array<u32> // len = num_inputs * 64
const SHADER_BASE_CASE: &str = r#"
struct Params {
    num_inputs: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_cells: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_cells: array<u32>;

const L3_SIDE: u32 = 8u;
const L2_SIDE: u32 = 4u;
const L3_CELLS: u32 = 512u;  // 8^3
const L2_CELLS: u32 = 64u;   // 4^3
const BAYS_BIRTH_MASK: u32 = 0xE0u;   // bits 5,6,7
const BAYS_SURVIVE_MASK: u32 = 0x60u; // bits 5,6

fn read_input(input_idx: u32, x: u32, y: u32, z: u32) -> u32 {
    let off = input_idx * L3_CELLS + z * L3_SIDE * L3_SIDE + y * L3_SIDE + x;
    return input_cells[off];
}

@compute @workgroup_size(64)
fn cs_base_case(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Linearize: total threads = num_inputs * 64 (one per output cell).
    let tid = gid.x;
    let total = params.num_inputs * L2_CELLS;
    if (tid >= total) { return; }
    let input_idx = tid / L2_CELLS;
    let cell_idx = tid % L2_CELLS;
    let cx = cell_idx % L2_SIDE;
    let cy = (cell_idx / L2_SIDE) % L2_SIDE;
    let cz = cell_idx / (L2_SIDE * L2_SIDE);
    // Center extract: input cell at (cx+2, cy+2, cz+2).
    let ix = cx + 2u;
    let iy = cy + 2u;
    let iz = cz + 2u;
    let cur = read_input(input_idx, ix, iy, iz);
    var alive: u32 = 0u;
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && dz == 0) { continue; }
                let nx = i32(ix) + dx;
                let ny = i32(iy) + dy;
                let nz = i32(iz) + dz;
                // Inside the 8³ block by construction (ix,iy,iz ∈ [2,6),
                // so neighbors stay in [1, 7] ⊂ [0, 8)). No bounds check
                // needed; kept off the hot path.
                alive = alive + read_input(input_idx, u32(nx), u32(ny), u32(nz));
            }
        }
    }
    var next: u32 = 0u;
    if (cur == 1u) {
        next = (BAYS_SURVIVE_MASK >> alive) & 1u;
    } else {
        next = (BAYS_BIRTH_MASK >> alive) & 1u;
    }
    let out_off = input_idx * L2_CELLS + cz * L2_SIDE * L2_SIDE + cy * L2_SIDE + cx;
    output_cells[out_off] = next;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BaseCaseParams {
    num_inputs: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_base_case: wgpu::ComputePipeline,
    bgl_base_case: wgpu::BindGroupLayout,
    pipeline_recursive_l4: wgpu::ComputePipeline,
    bgl_recursive_l4: wgpu::BindGroupLayout,
    timestamp_supported: bool,
    timestamp_period_ns: f32,
}

impl GpuCtx {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: Default::default(),
            display: Default::default(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let af = adapter.features();
        let timestamp_supported = af.contains(wgpu::Features::TIMESTAMP_QUERY)
            && af.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let mut required_features = wgpu::Features::empty();
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bench_gpu_level_dispatcher"),
            required_features,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;
        let timestamp_period_ns = if timestamp_supported {
            queue.get_timestamp_period()
        } else {
            0.0
        };

        let bgl_base_case = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("base_case_bgl"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("base_case_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_BASE_CASE.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("base_case_pl"),
            bind_group_layouts: &[Some(&bgl_base_case)],
            immediate_size: 0,
        });
        let pipeline_base_case = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cs_base_case"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_base_case"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bgl_recursive_l4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("recursive_l4_bgl"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });
        let shader_r = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("recursive_l4_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_RECURSIVE_L4.into()),
        });
        let pl_r = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("recursive_l4_pl"),
            bind_group_layouts: &[Some(&bgl_recursive_l4)],
            immediate_size: 0,
        });
        let pipeline_recursive_l4 =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cs_step_recursive_l4"),
                layout: Some(&pl_r),
                module: &shader_r,
                entry_point: Some("cs_step_recursive_l4"),
                compilation_options: Default::default(),
                cache: None,
            });

        Some(Self {
            device,
            queue,
            pipeline_base_case,
            bgl_base_case,
            pipeline_recursive_l4,
            bgl_recursive_l4,
            timestamp_supported,
            timestamp_period_ns,
        })
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_input_buffer<T: bytemuck::Pod>(ctx: &GpuCtx, label: &str, data: &[T]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
}

fn make_output_buffer(ctx: &GpuCtx, label: &str, count: u32) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (count as u64) * (std::mem::size_of::<u32>() as u64),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_buffer_u32(ctx: &GpuCtx, src: &wgpu::Buffer, count: u32) -> Vec<u32> {
    let size = (count as u64) * (std::mem::size_of::<u32>() as u64);
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    ctx.queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

// ---------------------------------------------------------------------------
// GPU base case driver.
// ---------------------------------------------------------------------------

fn run_base_case_gpu(ctx: &GpuCtx, inputs: &[[u32; L3_CELLS]]) -> Vec<[u32; L2_CELLS]> {
    let num_inputs = inputs.len() as u32;
    let mut flat: Vec<u32> = Vec::with_capacity(inputs.len() * L3_CELLS);
    for inp in inputs {
        flat.extend_from_slice(inp);
    }
    let input_buf = make_input_buffer(ctx, "base_case_inputs", &flat);
    let output_buf = make_output_buffer(ctx, "base_case_outputs", num_inputs * L2_CELLS as u32);
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("base_case_params"),
            contents: bytemuck::bytes_of(&BaseCaseParams {
                num_inputs,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("base_case_bg"),
        layout: &ctx.bgl_base_case,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("base_case_enc"),
        });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("base_case_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline_base_case);
        cpass.set_bind_group(0, &bg, &[]);
        let total_threads = num_inputs * L2_CELLS as u32;
        let groups = total_threads.div_ceil(WORKGROUP_SIZE);
        cpass.dispatch_workgroups(groups, 1, 1);
    }
    ctx.queue.submit(std::iter::once(enc.finish()));
    let flat_out = read_buffer_u32(ctx, &output_buf, num_inputs * L2_CELLS as u32);
    let mut out: Vec<[u32; L2_CELLS]> = Vec::with_capacity(inputs.len());
    for chunk in flat_out.chunks(L2_CELLS) {
        let mut arr = [0u32; L2_CELLS];
        arr.copy_from_slice(chunk);
        out.push(arr);
    }
    out
}

// ---------------------------------------------------------------------------
// Test fixtures (correctness corpus per plan adjustment 12).
// ---------------------------------------------------------------------------

fn fixture_empty() -> [u32; L3_CELLS] {
    [0u32; L3_CELLS]
}

fn fixture_full() -> [u32; L3_CELLS] {
    [1u32; L3_CELLS]
}

fn fixture_single_live_center() -> [u32; L3_CELLS] {
    let mut g = [0u32; L3_CELLS];
    let i = 4 * L3_SIDE * L3_SIDE + 4 * L3_SIDE + 4;
    g[i] = 1;
    g
}

fn fixture_seeded_random(seed: u64, density: f32) -> [u32; L3_CELLS] {
    // SplitMix64 — same RNG family as bench_gpu_hash_table.rs.
    let mut state: u64 = seed;
    let mut g = [0u32; L3_CELLS];
    for cell in g.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let r = (z as f64) / (u64::MAX as f64);
        *cell = if (r as f32) < density { 1 } else { 0 };
    }
    g
}

#[test]
fn cpu_base_case_empty_stays_empty() {
    let inp = fixture_empty();
    let out = cpu_base_case(&inp);
    assert!(
        out.iter().all(|&c| c == 0),
        "empty L3 → empty L2 (B5/6/7 needs 5+ alive neighbors)"
    );
}

#[test]
fn cpu_base_case_full_dies() {
    // All-alive 8³ → at the center, every cell has 26 alive neighbors.
    // Bays-5766 survives only on 5 or 6 → all cells die.
    let inp = fixture_full();
    let out = cpu_base_case(&inp);
    assert!(
        out.iter().all(|&c| c == 0),
        "full L3 → empty L2 (26 neighbors > S{{5,6}})"
    );
}

#[test]
fn cpu_base_case_single_live_dies() {
    // Single live cell at center has 0 alive neighbors → S{5,6} fails → dies.
    // Surrounding dead cells have 1 alive neighbor → B{5,6,7} fails → stay dead.
    let inp = fixture_single_live_center();
    let out = cpu_base_case(&inp);
    assert!(out.iter().all(|&c| c == 0), "single-live dies (no support)");
}

#[test]
fn gpu_base_case_matches_cpu_on_corpus() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    let inputs: Vec<[u32; L3_CELLS]> = vec![
        fixture_empty(),
        fixture_full(),
        fixture_single_live_center(),
        fixture_seeded_random(0xA5A5_A5A5, 0.30),
        fixture_seeded_random(42, 0.45),
    ];
    let expected: Vec<[u32; L2_CELLS]> = inputs.iter().map(cpu_base_case).collect();
    let got = run_base_case_gpu(&ctx, &inputs);
    assert_eq!(got.len(), expected.len(), "fixture count");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        for (j, (gv, ev)) in g.iter().zip(e.iter()).enumerate() {
            assert_eq!(gv, ev, "fixture {i} cell {j}: GPU={gv} CPU={ev}",);
        }
    }
}

#[test]
#[ignore]
fn gpu_base_case_throughput() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    // Stack a wide batch so the dispatch isn't single-threaded.
    let n = 1024usize;
    let inputs: Vec<[u32; L3_CELLS]> = (0..n)
        .map(|i| fixture_seeded_random(0x0ABC_DEF0 ^ i as u64, 0.30))
        .collect();
    // Warm-up.
    let _ = run_base_case_gpu(&ctx, &inputs);
    let t = Instant::now();
    let _ = run_base_case_gpu(&ctx, &inputs);
    let dt = t.elapsed();
    eprintln!(
        "base-case GPU dispatch: {n} inputs × 64 cells = {} threads in {:?}; {:.1} M cells/s",
        n * L2_CELLS,
        dt,
        ((n * L2_CELLS) as f64) / dt.as_secs_f64() / 1e6,
    );
}

// ===========================================================================
// Level-4 (16³) recursive step.
//
// Hashlife recursion at level 4: input is a 16³ world, output is the inner 8³
// region advanced by 2 generations. Mirrors `src/sim/hashlife.rs::
// step_recursive_case` (line ~859) but specialized to one recursive level
// above the base case (no further recursion below the level-3 base).
//
// Algorithm:
//   1. Form 27 intermediate level-3 nodes (each 8³) by sampling overlapping
//      8³ blocks at offsets (i*4, j*4, k*4) for i,j,k ∈ {0,1,2}. The 27
//      intermediates tile 16³ with 8-cell-wide blocks at 4-cell stride.
//   2. Step each intermediate (1-generation 8³→4³ base case) — 27 level-2
//      results in a 12³ post-gen-1 region [2,14)³ of the 16³ space.
//   3. Form 8 sub-cube level-3 nodes (each 8³) at corners of the 12³ region.
//      Sub-cube (si,sj,sk) ∈ {0,1}³ is composed of 8 of the 27 level-2
//      results: sub-cube oct (sx,sy,sz) sub_cell (cx,cy,cz) reads from
//      intermediate (si+cx, sj+cy, sk+cz). Each sub-cube reads from at most
//      8 intermediates.
//   4. Step each sub-cube (1-generation 8³→4³ base case) — 8 level-2 results.
//   5. Tile the 8 level-2 results into a single 8³ output: sub-cube oct
//      (sx,sy,sz)'s 4³ result lands at output offset (sx*4, sy*4, sz*4).
//
// Net effect: 2 generations on 16³ → 8³ output (lose 4 cells per side).
// ===========================================================================

const L4_SIDE: usize = 16;
const L4_CELLS: usize = L4_SIDE * L4_SIDE * L4_SIDE;
const NUM_INTERMEDIATES: usize = 27;

fn build_intermediate(input: &[u32; L4_CELLS], i: usize, j: usize, k: usize) -> [u32; L3_CELLS] {
    let mut block = [0u32; L3_CELLS];
    for bz in 0..L3_SIDE {
        for by in 0..L3_SIDE {
            for bx in 0..L3_SIDE {
                let gx = i * L2_SIDE + bx;
                let gy = j * L2_SIDE + by;
                let gz = k * L2_SIDE + bz;
                block[bz * L3_SIDE * L3_SIDE + by * L3_SIDE + bx] =
                    input[gz * L4_SIDE * L4_SIDE + gy * L4_SIDE + gx];
            }
        }
    }
    block
}

fn build_subcube_block(
    intermediate_results: &[[u32; L2_CELLS]; NUM_INTERMEDIATES],
    si: usize,
    sj: usize,
    sk: usize,
) -> [u32; L3_CELLS] {
    let mut block = [0u32; L3_CELLS];
    for cz in 0..2 {
        for cy in 0..2 {
            for cx in 0..2 {
                let im_idx = (sk + cz) * 9 + (sj + cy) * 3 + (si + cx);
                let im = &intermediate_results[im_idx];
                for fz in 0..L2_SIDE {
                    for fy in 0..L2_SIDE {
                        for fx in 0..L2_SIDE {
                            let bx = cx * L2_SIDE + fx;
                            let by = cy * L2_SIDE + fy;
                            let bz = cz * L2_SIDE + fz;
                            block[bz * L3_SIDE * L3_SIDE + by * L3_SIDE + bx] =
                                im[fz * L2_SIDE * L2_SIDE + fy * L2_SIDE + fx];
                        }
                    }
                }
            }
        }
    }
    block
}

// CPU level-4 step.
//
// **Oracle independence note (per code review):** this function is *structurally
// similar* to `cs_step_recursive_l4` — it uses the same 27-intermediate /
// 8-subcube decomposition and the same `(sk + cz) * 9 + (sj + cy) * 3 + (si + cx)`
// linearization as the WGSL kernel and as `build_subcube_block`. A bug in the
// shared algorithm would pass the GPU == CPU corpus check on both sides. The
// **third-party** oracle in this chain is `src/sim/hashlife.rs::step_recursive_case`
// (line ~859), which uses the same indexing (line 905: `centered[(ox + dx) +
// (oy + dy) * 3 + (oz + dz) * 9]`); the algorithm is therefore validated against
// production hashlife by inspection plus a hard `live_count == 109` assertion
// on the `random_seed_42` fixture in `gpu_level_4_matches_cpu_on_corpus`.
fn cpu_level_4_step(input: &[u32; L4_CELLS]) -> [u32; L3_CELLS] {
    // Phase 1: 27 intermediate base cases.
    let mut intermediate_results = [[0u32; L2_CELLS]; NUM_INTERMEDIATES];
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                let im = build_intermediate(input, i, j, k);
                intermediate_results[k * 9 + j * 3 + i] = cpu_base_case(&im);
            }
        }
    }
    // Phase 2: 8 sub-cube base cases, tiled into the final 8³ output.
    let mut output = [0u32; L3_CELLS];
    for sk in 0..2 {
        for sj in 0..2 {
            for si in 0..2 {
                let sub_block = build_subcube_block(&intermediate_results, si, sj, sk);
                let sub_result = cpu_base_case(&sub_block);
                for fz in 0..L2_SIDE {
                    for fy in 0..L2_SIDE {
                        for fx in 0..L2_SIDE {
                            let ox = si * L2_SIDE + fx;
                            let oy = sj * L2_SIDE + fy;
                            let oz = sk * L2_SIDE + fz;
                            output[oz * L3_SIDE * L3_SIDE + oy * L3_SIDE + ox] =
                                sub_result[fz * L2_SIDE * L2_SIDE + fy * L2_SIDE + fx];
                        }
                    }
                }
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// GPU recursive-case kernel (dispatch 2 of the breadth-first dispatcher).
//
// Per plan: one thread per (level-4 input × parity). Each thread serially
// runs the 8 sub-cube assemblies and base-case applications. This is the
// architecture under test — the spike validates the dispatcher *shape*
// (per-level breadth-first kernel, results feeding next level via storage
// buffers across a queue.submit fence), NOT throughput at level 4 with one
// thread per world. Throughput driver in commit 3 stacks N=64 worlds.
//
// Cross-dispatch fence: dispatch 1 (cs_base_case on 27*N intermediates)
// writes to `intermediate_results`. Dispatch 2 reads the same buffer.
// The visibility relies on the queue.submit boundary between dispatches.
// WebGPU spec: storage writes are visible across `queue.submit` boundaries
// (regardless of atomic vs non-atomic — abwm.2's cs_insert comment was
// specifically about atomics within a single dispatch, but the host-side
// submit boundary is a stronger guarantee that applies to plain storage
// writes too). WGSL has no in-shader release-store; the host-side submit
// boundary is sufficient. DO NOT fuse these into a single submit without
// explicit synchronization. See the kernel-internal ordering rule in this
// file's module docstring.
//
// NodeId-pre-allocation shortcut: result slots are pre-assigned by CPU,
// so direct buffer indexing replaces hash-table memo lookups for the
// architectural test. The same cross-dispatch fence semantics apply
// (storage write in dispatch 1 → storage read in dispatch 2).
// ---------------------------------------------------------------------------

const SHADER_RECURSIVE_L4: &str = r#"
struct RecursiveParams {
    num_worlds: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> rparams: RecursiveParams;
// intermediate_results layout: [world][intermediate_idx ∈ 0..27][cell_idx ∈ 0..64]
// = world * 27 * 64 + intermediate_idx * 64 + cell_idx
@group(0) @binding(1) var<storage, read> intermediate_results: array<u32>;
// final_outputs layout: [world][cell_idx ∈ 0..512]
@group(0) @binding(2) var<storage, read_write> final_outputs: array<u32>;

const L3_SIDE: u32 = 8u;
const L2_SIDE: u32 = 4u;
const L2_CELLS: u32 = 64u;
const NUM_INTERMEDIATES: u32 = 27u;
const FINAL_CELLS: u32 = 512u;
const BAYS_BIRTH_MASK: u32 = 0xE0u;   // bits 5,6,7
const BAYS_SURVIVE_MASK: u32 = 0x60u; // bits 5,6

fn read_intermediate(world: u32, intermediate_idx: u32, cell_idx: u32) -> u32 {
    let off = world * NUM_INTERMEDIATES * L2_CELLS + intermediate_idx * L2_CELLS + cell_idx;
    return intermediate_results[off];
}

// Per-thread scratch for one sub-cube's 8³ assembly. WGSL `var<private>` is
// per-shader-invocation: each thread gets its own copy.
var<private> scratch: array<u32, 512>;

@compute @workgroup_size(1)
fn cs_step_recursive_l4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let world = gid.x;
    if (world >= rparams.num_worlds) { return; }

    for (var sk: u32 = 0u; sk < 2u; sk = sk + 1u) {
        for (var sj: u32 = 0u; sj < 2u; sj = sj + 1u) {
            for (var si: u32 = 0u; si < 2u; si = si + 1u) {
                // Step A: assemble 8³ scratch from 8 4³ intermediate chunks.
                // sub-cube (si,sj,sk) sub_cell (cx,cy,cz) ← intermediate (si+cx, sj+cy, sk+cz).
                for (var cz: u32 = 0u; cz < 2u; cz = cz + 1u) {
                    for (var cy: u32 = 0u; cy < 2u; cy = cy + 1u) {
                        for (var cx: u32 = 0u; cx < 2u; cx = cx + 1u) {
                            let im_idx = (sk + cz) * 9u + (sj + cy) * 3u + (si + cx);
                            for (var fz: u32 = 0u; fz < L2_SIDE; fz = fz + 1u) {
                                for (var fy: u32 = 0u; fy < L2_SIDE; fy = fy + 1u) {
                                    for (var fx: u32 = 0u; fx < L2_SIDE; fx = fx + 1u) {
                                        let scr_x = cx * L2_SIDE + fx;
                                        let scr_y = cy * L2_SIDE + fy;
                                        let scr_z = cz * L2_SIDE + fz;
                                        let im_cell = fz * L2_SIDE * L2_SIDE + fy * L2_SIDE + fx;
                                        scratch[scr_z * L3_SIDE * L3_SIDE + scr_y * L3_SIDE + scr_x] =
                                            read_intermediate(world, im_idx, im_cell);
                                    }
                                }
                            }
                        }
                    }
                }

                // Step B: apply Bays-5766 base case to scratch → 4³ result;
                // tile into final output at sub-cube offset.
                for (var oz: u32 = 0u; oz < L2_SIDE; oz = oz + 1u) {
                    for (var oy: u32 = 0u; oy < L2_SIDE; oy = oy + 1u) {
                        for (var ox: u32 = 0u; ox < L2_SIDE; ox = ox + 1u) {
                            let ix = ox + 2u;
                            let iy = oy + 2u;
                            let iz = oz + 2u;
                            let cur = scratch[iz * L3_SIDE * L3_SIDE + iy * L3_SIDE + ix];
                            var alive: u32 = 0u;
                            for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
                                for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                                    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                                        if (dx == 0 && dy == 0 && dz == 0) { continue; }
                                        let nx = u32(i32(ix) + dx);
                                        let ny = u32(i32(iy) + dy);
                                        let nz = u32(i32(iz) + dz);
                                        alive = alive + scratch[nz * L3_SIDE * L3_SIDE + ny * L3_SIDE + nx];
                                    }
                                }
                            }
                            var next: u32 = 0u;
                            if (cur == 1u) {
                                next = (BAYS_SURVIVE_MASK >> alive) & 1u;
                            } else {
                                next = (BAYS_BIRTH_MASK >> alive) & 1u;
                            }
                            let fox = si * L2_SIDE + ox;
                            let foy = sj * L2_SIDE + oy;
                            let foz = sk * L2_SIDE + oz;
                            let out_off = world * FINAL_CELLS + foz * L3_SIDE * L3_SIDE + foy * L3_SIDE + fox;
                            final_outputs[out_off] = next;
                        }
                    }
                }
            }
        }
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RecursiveParams {
    num_worlds: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// GPU level-4 dispatcher driver.
//
// Two dispatches across separate `queue.submit` calls:
//   Dispatch 1: cs_base_case on (num_worlds * 27) level-3 intermediates →
//   intermediate_results buffer (length num_worlds * 27 * 64).
//   Dispatch 2: cs_step_recursive_l4 → final_outputs buffer
//   (length num_worlds * 512).
//
// Per plan adjustment 11, dispatches use SEPARATE queue.submit calls, not
// just separate compute passes within one encoder, so dispatch 2's reads
// see dispatch 1's writes via the cross-dispatch fence.
// ---------------------------------------------------------------------------

fn run_level_4_step_gpu(ctx: &GpuCtx, worlds: &[[u32; L4_CELLS]]) -> Vec<[u32; L3_CELLS]> {
    let num_worlds = worlds.len() as u32;
    assert!(num_worlds > 0, "need at least one world");

    // Build all (num_worlds * 27) intermediate inputs CPU-side.
    let mut flat_inputs: Vec<u32> = Vec::with_capacity(worlds.len() * NUM_INTERMEDIATES * L3_CELLS);
    for input in worlds {
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let im = build_intermediate(input, i, j, k);
                    flat_inputs.extend_from_slice(&im);
                }
            }
        }
    }
    let num_intermediate_inputs = num_worlds * NUM_INTERMEDIATES as u32;
    let intermediate_input_buf =
        make_input_buffer(ctx, "dispatcher_intermediate_inputs", &flat_inputs);
    let intermediate_results_buf = make_output_buffer(
        ctx,
        "dispatcher_intermediate_results",
        num_intermediate_inputs * L2_CELLS as u32,
    );
    let final_output_buf = make_output_buffer(
        ctx,
        "dispatcher_final_outputs",
        num_worlds * L3_CELLS as u32,
    );

    let base_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatcher_base_params"),
            contents: bytemuck::bytes_of(&BaseCaseParams {
                num_inputs: num_intermediate_inputs,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bg_base = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dispatcher_bg_base"),
        layout: &ctx.bgl_base_case,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: base_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: intermediate_input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: intermediate_results_buf.as_entire_binding(),
            },
        ],
    });

    // ---- Dispatch 1: 27 base-case calls per world ----
    let mut enc1 = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatcher_enc_dispatch1"),
        });
    {
        let mut cpass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dispatcher_pass1_base_case"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline_base_case);
        cpass.set_bind_group(0, &bg_base, &[]);
        let total_threads = num_intermediate_inputs * L2_CELLS as u32;
        let groups = total_threads.div_ceil(WORKGROUP_SIZE);
        cpass.dispatch_workgroups(groups, 1, 1);
    }
    ctx.queue.submit(std::iter::once(enc1.finish()));
    // (Cross-dispatch fence: queue.submit boundary commits the storage writes.)

    // ---- Dispatch 2: recursive level-4 kernel ----
    let rec_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatcher_rec_params"),
            contents: bytemuck::bytes_of(&RecursiveParams {
                num_worlds,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bg_rec = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dispatcher_bg_rec"),
        layout: &ctx.bgl_recursive_l4,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: rec_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: intermediate_results_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: final_output_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc2 = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatcher_enc_dispatch2"),
        });
    {
        let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dispatcher_pass2_recursive"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline_recursive_l4);
        cpass.set_bind_group(0, &bg_rec, &[]);
        cpass.dispatch_workgroups(num_worlds, 1, 1);
    }
    ctx.queue.submit(std::iter::once(enc2.finish()));

    // ---- Readback ----
    let flat = read_buffer_u32(ctx, &final_output_buf, num_worlds * L3_CELLS as u32);
    let mut out: Vec<[u32; L3_CELLS]> = Vec::with_capacity(worlds.len());
    for chunk in flat.chunks(L3_CELLS) {
        let mut arr = [0u32; L3_CELLS];
        arr.copy_from_slice(chunk);
        out.push(arr);
    }
    out
}

// ---------------------------------------------------------------------------
// Level-4 fixtures (per plan adjustment 12).
// ---------------------------------------------------------------------------

fn fixture_l4_empty() -> [u32; L4_CELLS] {
    [0u32; L4_CELLS]
}

fn fixture_l4_full() -> [u32; L4_CELLS] {
    [1u32; L4_CELLS]
}

fn fixture_l4_single_live_center() -> [u32; L4_CELLS] {
    let mut g = [0u32; L4_CELLS];
    let i = 8 * L4_SIDE * L4_SIDE + 8 * L4_SIDE + 8;
    g[i] = 1;
    g
}

/// 3×3×3 live block at corner (0,0,0). The block touches the world boundary;
/// stencils for cells near (0,0,0) read past the level-4 edge — but those
/// reads are inside the inner 8³ output region only when the live pattern
/// is within 4 cells of the inner boundary. This fixture exercises the
/// zero-padded boundary handling where the level-4 step's outer ring of
/// cells contributes to the inner result via stencil, but the level-4
/// boundary itself is implicitly zero-padded by the intermediate
/// construction (cells outside the 16³ region are not read; intermediates
/// whose blocks would extend past the boundary stay within [0,16)).
fn fixture_l4_boundary_touching() -> [u32; L4_CELLS] {
    let mut g = [0u32; L4_CELLS];
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                g[z * L4_SIDE * L4_SIDE + y * L4_SIDE + x] = 1;
            }
        }
    }
    g
}

fn fixture_l4_seeded_random(seed: u64, density: f32) -> [u32; L4_CELLS] {
    let mut state: u64 = seed;
    let mut g = [0u32; L4_CELLS];
    for cell in g.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let r = (z as f64) / (u64::MAX as f64);
        *cell = if (r as f32) < density { 1 } else { 0 };
    }
    g
}

fn level_4_corpus() -> Vec<(&'static str, [u32; L4_CELLS])> {
    vec![
        ("empty", fixture_l4_empty()),
        ("full", fixture_l4_full()),
        ("single_live_center", fixture_l4_single_live_center()),
        ("boundary_touching", fixture_l4_boundary_touching()),
        ("random_seed_42", fixture_l4_seeded_random(42, 0.30)),
    ]
}

#[test]
fn cpu_level_4_empty_stays_empty() {
    let inp = fixture_l4_empty();
    let out = cpu_level_4_step(&inp);
    assert!(out.iter().all(|&c| c == 0), "empty L4 → empty L3");
}

#[test]
fn cpu_level_4_full_dies() {
    // Dense regions in Bays-5766 collapse: inner cells have 26 alive
    // neighbors, and survive mask is {5,6} only.
    let inp = fixture_l4_full();
    let out = cpu_level_4_step(&inp);
    assert!(out.iter().all(|&c| c == 0), "full L4 → empty L3");
}

#[test]
fn cpu_level_4_single_live_dies() {
    // Single live cell with 0 neighbors → dies. Surrounding dead cells
    // have 1 alive neighbor → don't satisfy B{5,6,7} → stay dead.
    let inp = fixture_l4_single_live_center();
    let out = cpu_level_4_step(&inp);
    assert!(out.iter().all(|&c| c == 0), "single-live L4 → empty L3");
}

#[test]
#[ignore]
fn gpu_level_4_matches_cpu_on_corpus() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    let corpus = level_4_corpus();
    let worlds: Vec<[u32; L4_CELLS]> = corpus.iter().map(|(_, w)| *w).collect();
    let cpu_results: Vec<[u32; L3_CELLS]> = worlds.iter().map(cpu_level_4_step).collect();
    let gpu_results = run_level_4_step_gpu(&ctx, &worlds);
    assert_eq!(gpu_results.len(), worlds.len(), "result count");
    for (idx, (label, _)) in corpus.iter().enumerate() {
        let cpu = &cpu_results[idx];
        let gpu = &gpu_results[idx];
        for (cell_idx, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g, c, "fixture {label} cell {cell_idx}: GPU={g} CPU={c}");
        }
        let live_count = cpu.iter().filter(|&&c| c == 1).count();
        eprintln!("fixture {label}: {live_count} live cells in 8³ result");
        // Hard checksum on the random fixture: independent of the CPU oracle's
        // structural code (which mirrors the GPU's), this number is the
        // committed regression tripwire — if it changes, either the rule, the
        // stencil, or the seeded fixture changed.
        if *label == "random_seed_42" {
            assert_eq!(
                live_count, 109,
                "random_seed_42: expected 109 live cells in inner 8³ (committed regression tripwire); got {live_count}"
            );
        }
    }
}

// ===========================================================================
// Throughput driver (commit 3 — perf measurement for §8.7 paper appendix).
//
// Per plan adjustment 6: N-worlds × 1 iter (N = 64), NOT 1-world × 100 iters.
// At level-4 root with one world, dispatch 2 runs a single thread — measures
// nothing useful. Stack 64 random worlds into one buffer so dispatch 2 has
// 64 active threads.
//
// Per plan adjustment 7: each iteration starts from freshly cleared buffers
// + freshly uploaded inputs (no step chaining — avoids 3D Life's tendency
// to converge to sparse states which would trivialize subsequent iters).
//
// Per plan adjustment 9: in-encoder timestamps PER DISPATCH so we can
// distinguish "primitive is fast, scaling open" from "primitive itself slow."
// ===========================================================================

#[derive(Default, Debug, Clone, Copy)]
struct TimedRun {
    dispatch_1_gpu: Option<std::time::Duration>,
    dispatch_2_gpu: Option<std::time::Duration>,
    wall_total: std::time::Duration,
    // Per code-review feedback: decompose the wall total into measured
    // components so the §8.7 "spike-scale artifact" decomposition is data,
    // not estimate.
    cpu_prep: std::time::Duration,
    readback: std::time::Duration,
}

fn run_level_4_step_gpu_timed(
    ctx: &GpuCtx,
    worlds: &[[u32; L4_CELLS]],
) -> (Vec<[u32; L3_CELLS]>, TimedRun) {
    let num_worlds = worlds.len() as u32;
    assert!(num_worlds > 0, "need at least one world");
    let wall_start = Instant::now();
    let cpu_prep_start = Instant::now();

    // Build all (num_worlds * 27) intermediate inputs CPU-side.
    let mut flat_inputs: Vec<u32> = Vec::with_capacity(worlds.len() * NUM_INTERMEDIATES * L3_CELLS);
    for input in worlds {
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let im = build_intermediate(input, i, j, k);
                    flat_inputs.extend_from_slice(&im);
                }
            }
        }
    }
    let num_intermediate_inputs = num_worlds * NUM_INTERMEDIATES as u32;
    let intermediate_input_buf = make_input_buffer(ctx, "timed_intermediate_inputs", &flat_inputs);
    let intermediate_results_buf = make_output_buffer(
        ctx,
        "timed_intermediate_results",
        num_intermediate_inputs * L2_CELLS as u32,
    );
    let final_output_buf =
        make_output_buffer(ctx, "timed_final_outputs", num_worlds * L3_CELLS as u32);
    let cpu_prep = cpu_prep_start.elapsed();

    let base_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("timed_base_params"),
            contents: bytemuck::bytes_of(&BaseCaseParams {
                num_inputs: num_intermediate_inputs,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bg_base = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("timed_bg_base"),
        layout: &ctx.bgl_base_case,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: base_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: intermediate_input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: intermediate_results_buf.as_entire_binding(),
            },
        ],
    });

    // Per-dispatch timestamp infra. Two QuerySets (one per dispatch), each with
    // 2 slots (start, end). Resolve+copy in the same encoder so the data is
    // available after the corresponding queue.submit.
    let make_ts_set = |label: &str| -> Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)> {
        if !ctx.timestamp_supported {
            return None;
        }
        let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(label),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some((qs, resolve, readback))
    };
    let ts1 = make_ts_set("timed_ts_dispatch1");
    let ts2 = make_ts_set("timed_ts_dispatch2");

    // ---- Dispatch 1 ----
    let mut enc1 = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("timed_enc1"),
        });
    if let Some((qs, _, _)) = &ts1 {
        enc1.write_timestamp(qs, 0);
    }
    {
        let mut cpass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timed_pass1"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline_base_case);
        cpass.set_bind_group(0, &bg_base, &[]);
        let total_threads = num_intermediate_inputs * L2_CELLS as u32;
        let groups = total_threads.div_ceil(WORKGROUP_SIZE);
        cpass.dispatch_workgroups(groups, 1, 1);
    }
    if let Some((qs, resolve, readback)) = &ts1 {
        enc1.write_timestamp(qs, 1);
        enc1.resolve_query_set(qs, 0..2, resolve, 0);
        enc1.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
    }
    ctx.queue.submit(std::iter::once(enc1.finish()));

    // ---- Dispatch 2 ----
    let rec_params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("timed_rec_params"),
            contents: bytemuck::bytes_of(&RecursiveParams {
                num_worlds,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bg_rec = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("timed_bg_rec"),
        layout: &ctx.bgl_recursive_l4,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: rec_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: intermediate_results_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: final_output_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc2 = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("timed_enc2"),
        });
    if let Some((qs, _, _)) = &ts2 {
        enc2.write_timestamp(qs, 0);
    }
    {
        let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timed_pass2"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline_recursive_l4);
        cpass.set_bind_group(0, &bg_rec, &[]);
        cpass.dispatch_workgroups(num_worlds, 1, 1);
    }
    if let Some((qs, resolve, readback)) = &ts2 {
        enc2.write_timestamp(qs, 1);
        enc2.resolve_query_set(qs, 0..2, resolve, 0);
        enc2.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
    }
    ctx.queue.submit(std::iter::once(enc2.finish()));

    // ---- Readback (output cells + timestamps) ----
    let readback_start = Instant::now();
    let flat = read_buffer_u32(ctx, &final_output_buf, num_worlds * L3_CELLS as u32);
    let mut out: Vec<[u32; L3_CELLS]> = Vec::with_capacity(worlds.len());
    for chunk in flat.chunks(L3_CELLS) {
        let mut arr = [0u32; L3_CELLS];
        arr.copy_from_slice(chunk);
        out.push(arr);
    }
    let readback = readback_start.elapsed();

    let read_ts = |readback: &wgpu::Buffer| -> Option<std::time::Duration> {
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        rx.recv().ok()?.ok()?;
        let ticks = {
            let data = slice.get_mapped_range();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            if ts.len() >= 2 && ts[1] > ts[0] {
                Some(ts[1] - ts[0])
            } else {
                None
            }
        };
        readback.unmap();
        ticks.map(|t| {
            std::time::Duration::from_nanos((t as f64 * ctx.timestamp_period_ns as f64) as u64)
        })
    };
    let dispatch_1_gpu = ts1.as_ref().and_then(|(_, _, rb)| read_ts(rb));
    let dispatch_2_gpu = ts2.as_ref().and_then(|(_, _, rb)| read_ts(rb));

    (
        out,
        TimedRun {
            dispatch_1_gpu,
            dispatch_2_gpu,
            wall_total: wall_start.elapsed(),
            cpu_prep,
            readback,
        },
    )
}

#[test]
#[ignore]
fn gpu_level_4_throughput() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    eprintln!(
        "GPU dispatcher throughput (timestamp_supported={}, period_ns={})",
        ctx.timestamp_supported, ctx.timestamp_period_ns
    );

    const N_WORLDS: usize = 64;
    const N_ITERS: usize = 10;

    // Build N=64 random worlds (mixed seeds to defeat content-address aliasing).
    let worlds: Vec<[u32; L4_CELLS]> = (0..N_WORLDS)
        .map(|i| fixture_l4_seeded_random(0x42u64.wrapping_add(i as u64), 0.30))
        .collect();

    // Warm-up (compile shaders, allocate first set of buffers).
    let (_warm_out, _warm_t) = run_level_4_step_gpu_timed(&ctx, &worlds);

    // Measure: each iter starts from freshly cleared buffers (driver creates
    // new buffers per call). No step chaining — per plan adjustment 7.
    let mut runs: Vec<TimedRun> = Vec::with_capacity(N_ITERS);
    for _ in 0..N_ITERS {
        let (_out, t) = run_level_4_step_gpu_timed(&ctx, &worlds);
        runs.push(t);
    }

    // Per-column medians — independent for d1, d2, wall. (Sorting by wall and
    // taking that run's d1/d2 was misleading: wall is dominated by CPU prep +
    // readback in this harness, so the wall-median run is not guaranteed to be
    // the d1- or d2-median run.)
    fn median_dur(values: &[std::time::Duration]) -> std::time::Duration {
        let mut v = values.to_vec();
        v.sort();
        v[v.len() / 2]
    }
    let median_wall = median_dur(&runs.iter().map(|r| r.wall_total).collect::<Vec<_>>());
    let median_cpu_prep = median_dur(&runs.iter().map(|r| r.cpu_prep).collect::<Vec<_>>());
    let median_readback = median_dur(&runs.iter().map(|r| r.readback).collect::<Vec<_>>());
    let d1_samples: Vec<_> = runs.iter().filter_map(|r| r.dispatch_1_gpu).collect();
    let d2_samples: Vec<_> = runs.iter().filter_map(|r| r.dispatch_2_gpu).collect();
    let median_d1 = (!d1_samples.is_empty()).then(|| median_dur(&d1_samples));
    let median_d2 = (!d2_samples.is_empty()).then(|| median_dur(&d2_samples));

    let fmt_dur = |d: std::time::Duration| {
        let ns = d.as_nanos();
        if ns < 10_000 {
            format!("{ns} ns")
        } else {
            format!("{:.3} ms", ns as f64 / 1e6)
        }
    };
    eprintln!(
        "per-column medians (of {N_ITERS} runs, N={N_WORLDS} worlds, fresh buffers each iter):"
    );
    eprintln!(
        "  dispatch 1 (27*N base case, {} threads): {}",
        N_WORLDS * NUM_INTERMEDIATES * L2_CELLS,
        median_d1.map(fmt_dur).unwrap_or_else(|| "n/a".into())
    );
    eprintln!(
        "  dispatch 2 (level-4 recursive, {} threads): {}",
        N_WORLDS,
        median_d2.map(fmt_dur).unwrap_or_else(|| "n/a".into())
    );
    eprintln!(
        "  wall total (incl. CPU prep + readback): {}",
        fmt_dur(median_wall)
    );
    // Measured wall-time decomposition (replaces the §8.7 hand-decomposition).
    eprintln!(
        "  decomposition (median): cpu_prep = {}, readback = {}, residual = {}",
        fmt_dur(median_cpu_prep),
        fmt_dur(median_readback),
        fmt_dur(median_wall.saturating_sub(median_cpu_prep + median_readback)),
    );

    // Plan adjustment 5 — verdict criteria:
    //   GO: per-dispatch ≤ 1 ms AND total step ≤ 5 ms
    //   NO-GO: per-dispatch > 5 ms (we can't fix in 2h)
    //   CAVEAT: 1–5 ms per-dispatch — file follow-up beads per issue
    if let (Some(d1), Some(d2)) = (median_d1, median_d2) {
        let max_dispatch = d1.max(d2);
        let verdict = if max_dispatch.as_micros() <= 1_000 && median_wall.as_micros() <= 5_000 {
            "GO"
        } else if max_dispatch.as_micros() > 5_000 {
            "NO-GO"
        } else {
            "CAVEAT"
        };
        eprintln!("verdict (per plan adjustment 5): {verdict}");
        eprintln!(
            "  max(dispatch_1, dispatch_2) = {} {} 1 ms",
            fmt_dur(max_dispatch),
            if max_dispatch.as_micros() <= 1_000 {
                "≤"
            } else {
                ">"
            }
        );
        eprintln!(
            "  wall total = {} {} 5 ms",
            fmt_dur(median_wall),
            if median_wall.as_micros() <= 5_000 {
                "≤"
            } else {
                ">"
            }
        );
    } else {
        eprintln!("verdict: timestamps unavailable; cannot decide GO/NO-GO from this run");
    }

    // Also dump all individual runs so noise can be inspected manually.
    eprintln!("--- per-iter detail ---");
    for (i, r) in runs.iter().enumerate() {
        eprintln!(
            "  iter {i}: d1={} d2={} cpu_prep={} readback={} wall={}",
            r.dispatch_1_gpu
                .map(fmt_dur)
                .unwrap_or_else(|| "n/a".into()),
            r.dispatch_2_gpu
                .map(fmt_dur)
                .unwrap_or_else(|| "n/a".into()),
            fmt_dur(r.cpu_prep),
            fmt_dur(r.readback),
            fmt_dur(r.wall_total),
        );
    }
}
