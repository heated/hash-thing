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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bench_gpu_level_dispatcher"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;

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

        Some(Self {
            device,
            queue,
            pipeline_base_case,
            bgl_base_case,
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
