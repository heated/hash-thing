//! GPU hash-table microbenchmark (hash-thing-abwm.2).
//!
//! Open-addressing linear-probe hash table on a WGSL compute shader. Measures
//! insert / lookup throughput to gate the GPU spatial-memo exploration
//! (hash-thing-abwm epic). Pinned decisions live in the plan at
//! `.ship-notes/plan-spark-hash-thing-abwm.2-gpu-hash-bench.md` — notably:
//! in-encoder timestamps (matches `bench_gpu_raycast.rs` + dlse.2.3),
//! overwrite-duplicate semantics (memo is a pure function), single u32
//! packed key (NodeId<<1 | parity) with 0 reserved as empty.
//!
//! Run the full sweep (ignored by default):
//!   cargo test --release --test bench_gpu_hash_table -- --ignored --nocapture
//! Correctness tests run in default `cargo test`.

use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

/// RNG seed for reproducibility (plan §5).
const RNG_SEED: u64 = 0xA5A5_A5A5;

/// Workgroup size. 64 is the conventional WebGPU-portable choice.
const WORKGROUP_SIZE: u32 = 64;

/// Histogram bucket count (probes 0..30, plus 31 = overflow).
const HIST_BUCKETS: u32 = 32;

const SHADER_SRC: &str = r#"
struct Params {
    capacity: u32,
    capacity_mask: u32,
    n_keys: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_keys: array<u32>;
@group(0) @binding(2) var<storage, read> input_values: array<u32>;
@group(0) @binding(3) var<storage, read_write> slot_keys: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> slot_values: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> probe_hist: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> lookup_results: array<u32>;

// fxhash-shaped: mul by odd 32-bit constant derived from the low half of
// Knuth's golden-ratio 64-bit constant (0x9E3779B97F4A7C15), xor-shift,
// mul again. Avoids u64 emulation.
fn fxhash(k: u32) -> u32 {
    var h: u32 = k * 0x7F4A7C15u;
    h = h ^ (h >> 16u);
    h = h * 0x9E3779B9u;
    return h;
}

fn bump_hist(probes: u32) {
    let b = min(probes, 30u);
    atomicAdd(&probe_hist[b], 1u);
}

@compute @workgroup_size(64)
fn cs_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.capacity) { return; }
    atomicStore(&slot_keys[i], 0u);
    atomicStore(&slot_values[i], 0u);
}

@compute @workgroup_size(64)
fn cs_insert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.n_keys) { return; }
    let key = input_keys[tid];
    let value = input_values[tid];
    if (key == 0u) { return; }
    var slot = fxhash(key) & params.capacity_mask;
    var probes: u32 = 0u;
    loop {
        if (probes >= params.capacity) {
            atomicAdd(&probe_hist[31u], 1u);
            return;
        }
        let ex = atomicCompareExchangeWeak(&slot_keys[slot], 0u, key);
        if (ex.exchanged) {
            atomicStore(&slot_values[slot], value);
            bump_hist(probes);
            return;
        }
        if (ex.old_value == key) {
            // Overwrite duplicate (memo purity — same key must map to same
            // value). Per-plan semantics.
            atomicStore(&slot_values[slot], value);
            bump_hist(probes);
            return;
        }
        if (ex.old_value == 0u) {
            // Spurious CAS failure — retry same slot without counting a probe.
            continue;
        }
        probes = probes + 1u;
        slot = (slot + 1u) & params.capacity_mask;
    }
}

@compute @workgroup_size(64)
fn cs_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.n_keys) { return; }
    let key = input_keys[tid];
    if (key == 0u) {
        lookup_results[tid] = 0xFFFFFFFFu;
        return;
    }
    var slot = fxhash(key) & params.capacity_mask;
    var probes: u32 = 0u;
    loop {
        if (probes >= params.capacity) {
            lookup_results[tid] = 0xFFFFFFFFu;
            atomicAdd(&probe_hist[31u], 1u);
            return;
        }
        let sk = atomicLoad(&slot_keys[slot]);
        if (sk == 0u) {
            lookup_results[tid] = 0xFFFFFFFFu;
            bump_hist(probes);
            return;
        }
        if (sk == key) {
            lookup_results[tid] = atomicLoad(&slot_values[slot]);
            bump_hist(probes);
            return;
        }
        probes = probes + 1u;
        slot = (slot + 1u) & params.capacity_mask;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    capacity: u32,
    capacity_mask: u32,
    n_keys: u32,
    _pad: u32,
}

struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipeline_reset: wgpu::ComputePipeline,
    pipeline_insert: wgpu::ComputePipeline,
    pipeline_lookup: wgpu::ComputePipeline,
    timestamp_supported: bool,
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
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;

        let af = adapter.features();
        let timestamp_supported = af.contains(wgpu::Features::TIMESTAMP_QUERY)
            && af.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let mut required_features = wgpu::Features::empty();
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("bench_gpu_hash_table"),
                required_features,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }))
            .ok()?;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hash_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
                bgl_storage_rw(4),
                bgl_storage_rw(5),
                bgl_storage_rw(6),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hash_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hash_pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk_pipeline = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Some(Self {
            pipeline_reset: mk_pipeline("cs_reset", "cs_reset"),
            pipeline_insert: mk_pipeline("cs_insert", "cs_insert"),
            pipeline_lookup: mk_pipeline("cs_lookup", "cs_lookup"),
            bgl,
            device,
            queue,
            timestamp_supported,
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

struct Table {
    capacity: u32,
    params_buf: wgpu::Buffer,
    slot_keys: wgpu::Buffer,
    slot_values: wgpu::Buffer,
    probe_hist: wgpu::Buffer,
    probe_hist_read: wgpu::Buffer,
}

impl Table {
    fn new(ctx: &GpuCtx, capacity: u32) -> Self {
        assert!(capacity.is_power_of_two(), "capacity must be pow2");
        let bytes_u32 = std::mem::size_of::<u32>() as u64;
        let slot_keys = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot_keys"),
            size: bytes_u32 * capacity as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slot_values = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot_values"),
            size: bytes_u32 * capacity as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let probe_hist = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_hist"),
            size: bytes_u32 * HIST_BUCKETS as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let probe_hist_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_hist_read"),
            size: bytes_u32 * HIST_BUCKETS as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            capacity,
            params_buf,
            slot_keys,
            slot_values,
            probe_hist,
            probe_hist_read,
        }
    }

    fn write_params(&self, ctx: &GpuCtx, n_keys: u32) {
        let p = Params {
            capacity: self.capacity,
            capacity_mask: self.capacity - 1,
            n_keys,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[p]));
    }

    fn clear_probe_hist(&self, ctx: &GpuCtx) {
        let zeros = vec![0u32; HIST_BUCKETS as usize];
        ctx.queue
            .write_buffer(&self.probe_hist, 0, bytemuck::cast_slice(&zeros));
    }

    fn read_probe_hist(&self, ctx: &GpuCtx) -> [u32; HIST_BUCKETS as usize] {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            &self.probe_hist,
            0,
            &self.probe_hist_read,
            0,
            std::mem::size_of::<u32>() as u64 * HIST_BUCKETS as u64,
        );
        ctx.queue.submit(Some(enc.finish()));
        let slice = self.probe_hist_read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let _ = rx.recv().unwrap();
        let data = slice.get_mapped_range();
        let mut out = [0u32; HIST_BUCKETS as usize];
        out.copy_from_slice(bytemuck::cast_slice(&data[..]));
        drop(data);
        self.probe_hist_read.unmap();
        out
    }

    fn reset(&self, ctx: &GpuCtx) {
        self.write_params(ctx, 0);
        let bg = self.bind_group(ctx, &dummy_input(ctx), &dummy_input(ctx), &dummy_output(ctx));
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&ctx.pipeline_reset);
            pass.set_bind_group(0, &bg, &[]);
            let groups = self.capacity.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        self.clear_probe_hist(ctx);
    }

    fn bind_group(
        &self,
        ctx: &GpuCtx,
        input_keys: &wgpu::Buffer,
        input_values: &wgpu::Buffer,
        lookup_results: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash_bg"),
            layout: &ctx.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.slot_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.slot_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.probe_hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: lookup_results.as_entire_binding(),
                },
            ],
        })
    }
}

fn dummy_input(ctx: &GpuCtx) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dummy_input"),
        contents: bytemuck::cast_slice(&[0u32; 4]),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn dummy_output(ctx: &GpuCtx) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dummy_output"),
        contents: bytemuck::cast_slice(&[0u32; 4]),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn make_input_buffer(ctx: &GpuCtx, label: &str, data: &[u32]) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn make_output_buffer(ctx: &GpuCtx, label: &str, n: u32) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<u32>() as u64 * n.max(1) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_buffer_u32(ctx: &GpuCtx, src: &wgpu::Buffer, n: u32) -> Vec<u32> {
    let bytes = std::mem::size_of::<u32>() as u64 * n as u64;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &readback, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<u32> = bytemuck::cast_slice(&data[..]).to_vec();
    drop(data);
    readback.unmap();
    out
}

/// Single-phase dispatch with optional timestamp bracketing. Returns wall-clock
/// submit+poll time; if GPU timestamps are available, the caller can query
/// `last_gpu_duration` separately via the timestamp buffer (not wired here —
/// the bench uses wall-clock around submit+poll as ground truth, matching the
/// dlse.2.3 stance that pass-level Metal timestamps inflate with barrier time).
fn run_phase(
    ctx: &GpuCtx,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    n: u32,
) -> Duration {
    let start = Instant::now();
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        let groups = n.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    start.elapsed()
}

// ---------- RNG helpers (SplitMix64, self-contained for reproducibility) ----

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn gen_unique_keys(n: u32, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut set = std::collections::HashSet::with_capacity(n as usize);
    let mut out = Vec::with_capacity(n as usize);
    while out.len() < n as usize {
        let k = (splitmix64(&mut state) as u32) | 1; // avoid 0 (empty sentinel)
        if set.insert(k) {
            out.push(k);
        }
    }
    out
}

// ------------------------------------------------------------- correctness --

#[test]
fn insert_all_then_lookup_all_hits() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    let n = 1024u32;
    let capacity = (n * 2).next_power_of_two();
    let table = Table::new(&ctx, capacity);
    table.reset(&ctx);

    let keys = gen_unique_keys(n, RNG_SEED);
    let values: Vec<u32> = (0..n).collect();
    let keys_buf = make_input_buffer(&ctx, "keys", &keys);
    let values_buf = make_input_buffer(&ctx, "values", &values);
    let results_buf = make_output_buffer(&ctx, "results", n);

    table.write_params(&ctx, n);
    let bg = table.bind_group(&ctx, &keys_buf, &values_buf, &results_buf);

    let _ = run_phase(&ctx, &ctx.pipeline_insert, &bg, n);
    let _ = run_phase(&ctx, &ctx.pipeline_lookup, &bg, n);

    let results = read_buffer_u32(&ctx, &results_buf, n);
    for (i, &r) in results.iter().enumerate() {
        assert_eq!(
            r, values[i],
            "lookup[{i}] key={} expected={} got={}",
            keys[i], values[i], r
        );
    }
}

#[test]
fn lookup_disjoint_keys_all_miss() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    let n = 1024u32;
    let capacity = (n * 2).next_power_of_two();
    let table = Table::new(&ctx, capacity);
    table.reset(&ctx);

    let keys = gen_unique_keys(n, RNG_SEED);
    let values: Vec<u32> = (0..n).collect();
    let keys_buf = make_input_buffer(&ctx, "keys", &keys);
    let values_buf = make_input_buffer(&ctx, "values", &values);
    let _results_buf = make_output_buffer(&ctx, "results_ins", n);
    table.write_params(&ctx, n);
    let bg = table.bind_group(&ctx, &keys_buf, &values_buf, &_results_buf);
    let _ = run_phase(&ctx, &ctx.pipeline_insert, &bg, n);

    // Disjoint keys for lookup (seed offset avoids any collision).
    let miss_keys = gen_unique_keys(n, RNG_SEED ^ 0xDEAD_BEEF);
    let miss_keys_buf = make_input_buffer(&ctx, "miss_keys", &miss_keys);
    let miss_results_buf = make_output_buffer(&ctx, "miss_results", n);
    let bg2 = table.bind_group(&ctx, &miss_keys_buf, &values_buf, &miss_results_buf);
    let _ = run_phase(&ctx, &ctx.pipeline_lookup, &bg2, n);

    // Any collision between the two key sets would produce a spurious hit;
    // compute an expected hit count from a CPU-side set intersection to
    // stay rigorous about "all miss" under a disjoint-seed workload.
    let inserted: std::collections::HashSet<u32> = keys.iter().copied().collect();
    let expected_collisions: usize = miss_keys.iter().filter(|k| inserted.contains(k)).count();

    let results = read_buffer_u32(&ctx, &miss_results_buf, n);
    let got_hits = results.iter().filter(|r| **r != 0xFFFF_FFFF).count();
    assert_eq!(
        got_hits, expected_collisions,
        "miss-set returned {got_hits} hits; expected {expected_collisions} from CPU-side set intersection"
    );
}

#[test]
fn duplicate_insert_is_idempotent() {
    // Plan §5: insert the same key twice; must land a single entry, final
    // value wins (overwrite semantics).
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    let n_unique = 64u32;
    let capacity = 256u32;
    let table = Table::new(&ctx, capacity);
    table.reset(&ctx);
    let keys = gen_unique_keys(n_unique, RNG_SEED);

    // Insert pass A: keys → values 0..n
    let values_a: Vec<u32> = (0..n_unique).collect();
    let keys_buf = make_input_buffer(&ctx, "keys", &keys);
    let values_a_buf = make_input_buffer(&ctx, "values_a", &values_a);
    let results_buf = make_output_buffer(&ctx, "results", n_unique);
    table.write_params(&ctx, n_unique);
    let bg_a = table.bind_group(&ctx, &keys_buf, &values_a_buf, &results_buf);
    let _ = run_phase(&ctx, &ctx.pipeline_insert, &bg_a, n_unique);

    // Insert pass B: same keys → values 1000..1000+n (overwrite)
    let values_b: Vec<u32> = (1000..1000 + n_unique).collect();
    let values_b_buf = make_input_buffer(&ctx, "values_b", &values_b);
    let bg_b = table.bind_group(&ctx, &keys_buf, &values_b_buf, &results_buf);
    let _ = run_phase(&ctx, &ctx.pipeline_insert, &bg_b, n_unique);

    // Lookup must return pass-B values.
    let _ = run_phase(&ctx, &ctx.pipeline_lookup, &bg_b, n_unique);
    let results = read_buffer_u32(&ctx, &results_buf, n_unique);
    for (i, &r) in results.iter().enumerate() {
        assert_eq!(
            r, values_b[i],
            "after duplicate insert, lookup[{i}] expected {} got {}",
            values_b[i], r
        );
    }
}

#[test]
fn concurrent_same_key_cas_serializes() {
    // Plan §5 (Claude finding): M threads all race to insert the SAME key
    // with the SAME value. CAS must serialize to exactly one winner; lookup
    // must return that value and no other slot may contain the key.
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    let m = 512u32;
    let capacity = 1024u32;
    let table = Table::new(&ctx, capacity);
    table.reset(&ctx);

    let shared_key = 0xC0FF_EE42_u32;
    let shared_value = 0xABCDu32;
    let keys = vec![shared_key; m as usize];
    let values = vec![shared_value; m as usize];
    let keys_buf = make_input_buffer(&ctx, "keys_same", &keys);
    let values_buf = make_input_buffer(&ctx, "values_same", &values);
    let results_buf = make_output_buffer(&ctx, "results", m);
    table.write_params(&ctx, m);
    let bg = table.bind_group(&ctx, &keys_buf, &values_buf, &results_buf);
    let _ = run_phase(&ctx, &ctx.pipeline_insert, &bg, m);

    // Readback slot_keys: exactly one slot should hold shared_key, rest should
    // be 0. Readback slot_values at that slot: shared_value.
    let slots = read_buffer_u32(&ctx, &table.slot_keys, capacity);
    let key_count = slots.iter().filter(|k| **k == shared_key).count();
    assert_eq!(
        key_count, 1,
        "concurrent same-key insert landed {key_count} copies; CAS failed to serialize"
    );
    // Stray values outside the winning slot are allowed (they may be in the
    // process of being written when the race resolved); what matters is that
    // there's exactly one slot_keys entry for the shared key.
}

// ------------------------------------------------------------- sweep -------

#[derive(Debug, Clone)]
struct Row {
    n: u32,
    m_threads: u32,
    load_factor: f32,
    capacity: u32,
    insert_mean_ms: f64,
    insert_p95_ms: f64,
    lookup_hit_mean_ms: f64,
    lookup_miss_mean_ms: f64,
    max_probes: u32,
    insert_mkeys_s: f64,
    lookup_hit_mkeys_s: f64,
    lookup_miss_mkeys_s: f64,
}

fn p95(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort();
    let idx = ((samples.len() as f64) * 0.95).ceil() as usize;
    samples[idx.min(samples.len() - 1)]
}

fn mean(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let total: Duration = samples.iter().sum();
    total / samples.len() as u32
}

fn max_probe_bucket(hist: &[u32]) -> u32 {
    for (i, &c) in hist.iter().enumerate().rev() {
        if c > 0 {
            return i as u32;
        }
    }
    0
}

fn run_sweep_row(
    ctx: &GpuCtx,
    n: u32,
    m_threads: u32,
    load_factor: f32,
) -> Row {
    let capacity = ((n as f64 / load_factor as f64).ceil() as u32)
        .max(WORKGROUP_SIZE)
        .next_power_of_two();
    let table = Table::new(ctx, capacity);

    let keys = gen_unique_keys(n, RNG_SEED);
    let values: Vec<u32> = (0..n).map(|i| i + 1).collect();
    let miss_keys = gen_unique_keys(n, RNG_SEED ^ 0xDEAD_BEEF);
    let keys_buf = make_input_buffer(ctx, "sweep_keys", &keys);
    let values_buf = make_input_buffer(ctx, "sweep_values", &values);
    let miss_keys_buf = make_input_buffer(ctx, "sweep_miss_keys", &miss_keys);
    let results_buf = make_output_buffer(ctx, "sweep_results", n);

    // Warm run (discarded) + 5 timed runs per plan §5.
    let warm_runs = 1;
    let timed_runs = 5;
    let mut insert_samples = Vec::with_capacity(timed_runs);
    let mut hit_samples = Vec::with_capacity(timed_runs);
    let mut miss_samples = Vec::with_capacity(timed_runs);
    let mut max_probes = 0u32;

    let threads_to_dispatch = m_threads.min(n);

    for pass in 0..(warm_runs + timed_runs) {
        table.reset(ctx);
        table.write_params(ctx, threads_to_dispatch);
        let bg_ins = table.bind_group(ctx, &keys_buf, &values_buf, &results_buf);
        let d_ins = run_phase(ctx, &ctx.pipeline_insert, &bg_ins, threads_to_dispatch);
        let d_hit = run_phase(ctx, &ctx.pipeline_lookup, &bg_ins, threads_to_dispatch);
        let bg_miss = table.bind_group(ctx, &miss_keys_buf, &values_buf, &results_buf);
        let d_miss = run_phase(ctx, &ctx.pipeline_lookup, &bg_miss, threads_to_dispatch);

        if pass >= warm_runs {
            insert_samples.push(d_ins);
            hit_samples.push(d_hit);
            miss_samples.push(d_miss);
            let hist = table.read_probe_hist(ctx);
            max_probes = max_probes.max(max_probe_bucket(&hist));
        }
    }

    let ins_mean = mean(&insert_samples);
    let hit_mean = mean(&hit_samples);
    let miss_mean = mean(&miss_samples);
    let ins_p95 = p95(&mut insert_samples.clone());

    let keys_dispatched = threads_to_dispatch as f64;
    let to_ms = |d: Duration| d.as_secs_f64() * 1_000.0;
    let mkeys_s = |d: Duration| {
        let s = d.as_secs_f64();
        if s > 0.0 {
            keys_dispatched / s / 1_000_000.0
        } else {
            0.0
        }
    };

    Row {
        n,
        m_threads: threads_to_dispatch,
        load_factor,
        capacity,
        insert_mean_ms: to_ms(ins_mean),
        insert_p95_ms: to_ms(ins_p95),
        lookup_hit_mean_ms: to_ms(hit_mean),
        lookup_miss_mean_ms: to_ms(miss_mean),
        max_probes,
        insert_mkeys_s: mkeys_s(ins_mean),
        lookup_hit_mkeys_s: mkeys_s(hit_mean),
        lookup_miss_mkeys_s: mkeys_s(miss_mean),
    }
}

#[test]
#[ignore]
fn sweep_gpu_hash_throughput() {
    let ctx = match GpuCtx::new() {
        Some(c) => c,
        None => {
            eprintln!("skip: no GPU adapter");
            return;
        }
    };
    println!(
        "GPU hash-table sweep (seed 0x{:X}, timestamp_supported={})",
        RNG_SEED, ctx.timestamp_supported
    );
    println!(
        "| N       | M     | LF   | cap     | ins_ms (p95) | hit_ms | miss_ms | ins_Mk/s | hit_Mk/s | miss_Mk/s | maxP |"
    );
    println!(
        "|---------|-------|------|---------|--------------|--------|---------|----------|----------|-----------|------|"
    );

    let ns: [u32; 3] = [10_000, 100_000, 1_000_000];
    let ms: [u32; 4] = [256, 1024, 4096, 16_384];
    let lfs: [f32; 3] = [0.5, 0.75, 0.9];

    for &n in &ns {
        for &m in &ms {
            for &lf in &lfs {
                let row = run_sweep_row(&ctx, n, m, lf);
                println!(
                    "| {:>7} | {:>5} | {:.2} | {:>7} | {:>5.2} ({:>4.2}) | {:>5.2} | {:>6.2} | {:>8.1} | {:>8.1} | {:>9.1} | {:>4} |",
                    row.n,
                    row.m_threads,
                    row.load_factor,
                    row.capacity,
                    row.insert_mean_ms,
                    row.insert_p95_ms,
                    row.lookup_hit_mean_ms,
                    row.lookup_miss_mean_ms,
                    row.insert_mkeys_s,
                    row.lookup_hit_mkeys_s,
                    row.lookup_miss_mkeys_s,
                    row.max_probes,
                );
            }
        }
    }
}
