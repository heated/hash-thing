//! Headless GPU raycast benchmarks at multiple world scales (m1f.2).
//!
//! Renders the SVDAG raycast compute shader to an off-screen storage texture,
//! blits to a render target, and measures GPU execution time via TIMESTAMP_QUERY
//! (falls back to wall-clock if unavailable). No window required.
//!
//! Run with: `cargo test --release --test bench_gpu_raycast -- --ignored --nocapture`
//!
//! Constraint: each scale caps at ~30 seconds total. If a single frame
//! exceeds 5 seconds, the benchmark aborts at that scale.
//!
//! Confidence note: this harness is observational only. GPU-adapter absence,
//! timeout, or timing regressions are logged for human review; they are not
//! enforced as pass/fail confidence gates.

use hash_thing::render::Svdag;
use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const RENDER_WIDTH: u32 = 1920;
const RENDER_HEIGHT: u32 = 1080;
const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const RAYCAST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Camera + scene uniforms matching the SVDAG compute raycast shader layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera_pos: [f32; 4],
    camera_dir: [f32; 4],
    camera_up: [f32; 4],
    camera_right: [f32; 4],
    /// x: volume_size, y: aspect_ratio, z: fov_tan, w: screen_height
    params: [f32; 4],
    /// x: debug_mode, y: lod_bias, z: output_width, w: output_height
    debug: [f32; 4],
}

#[derive(Clone, Copy)]
struct BenchCamera {
    pos: [f32; 3],
    dir: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
}

impl BenchCamera {
    fn empty_corner() -> Self {
        Self {
            pos: [0.0, 0.0, 0.0],
            dir: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
        }
    }

    fn app_spawn(volume_size: u32) -> Self {
        let side = volume_size as f32;
        let yaw = std::f32::consts::FRAC_PI_4;
        let pitch = 0.0f32;
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let dir = [-cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw];
        let right = [cos_yaw, 0.0, -sin_yaw];
        let up = [
            right[1] * dir[2] - right[2] * dir[1],
            right[2] * dir[0] - right[0] * dir[2],
            right[0] * dir[1] - right[1] * dir[0],
        ];
        Self {
            pos: [0.5, 0.5 + 2.0 / side, 0.5],
            dir,
            up,
            right,
        }
    }
}

struct HeadlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Compute raycast pipeline
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: wgpu::Buffer,
    palette_buffer: wgpu::Buffer,
    dag_buffer: Option<wgpu::Buffer>,
    raycast_texture_view: wgpu::TextureView,
    // Blit pipeline
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    render_target: wgpu::TextureView,
    // GPU timing
    timestamp_supported: bool,
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    readback_buffer: Option<wgpu::Buffer>,
    timestamp_period: f32,
}

impl HeadlessRenderer {
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

        let adapter_features = adapter.features();
        let timestamp_supported = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let mut required_features = wgpu::Features::empty();
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bench headless"),
            required_features,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;

        let timestamp_period = if timestamp_supported {
            queue.get_timestamp_period()
        } else {
            0.0
        };

        // --- Render target (final blit output) ---
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench_target"),
            size: wgpu::Extent3d {
                width: RENDER_WIDTH,
                height: RENDER_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RENDER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let render_target = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Raycast storage texture (compute output) ---
        let raycast_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raycast_output"),
            size: wgpu::Extent3d {
                width: RENDER_WIDTH,
                height: RENDER_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RAYCAST_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let raycast_texture_view = raycast_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Uniforms ---
        let camera = BenchCamera::empty_corner();
        let uniforms = Uniforms {
            camera_pos: [camera.pos[0], camera.pos[1], camera.pos[2], 0.0],
            camera_dir: [camera.dir[0], camera.dir[1], camera.dir[2], 0.0],
            camera_up: [camera.up[0], camera.up[1], camera.up[2], 0.0],
            camera_right: [camera.right[0], camera.right[1], camera.right[2], 0.0],
            params: [
                64.0,
                RENDER_WIDTH as f32 / RENDER_HEIGHT as f32,
                (std::f32::consts::FRAC_PI_4).tan(),
                RENDER_HEIGHT as f32,
            ],
            debug: [0.0, 0.0, RENDER_WIDTH as f32, RENDER_HEIGHT as f32],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Palette buffer ---
        let palette: Vec<[f32; 4]> = (0..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                [t * 0.3 + 0.3, t * 0.5 + 0.2, t * 0.2 + 0.4, 1.0]
            })
            .collect();
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // --- Compute pipeline (SVDAG raycast) ---
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("svdag_compute_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: RAYCAST_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svdag raycast compute"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../crates/ht-render/src/svdag_raycast.wgsl").into(),
            ),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("svdag_compute_pl"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("svdag_compute"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Blit pipeline (samples raycast texture → render target) ---
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&raycast_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../crates/ht-render/src/svdag_blit.wgsl").into(),
            ),
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_rp"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: RENDER_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Timestamp query resources ---
        let (query_set, resolve_buffer, readback_buffer) = if timestamp_supported {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("bench_timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_readback"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(qs), Some(resolve), Some(readback))
        } else {
            (None, None, None)
        };

        Some(Self {
            device,
            queue,
            compute_pipeline,
            compute_bind_group_layout,
            compute_bind_group: None,
            uniform_buffer,
            palette_buffer,
            dag_buffer: None,
            raycast_texture_view,
            blit_pipeline,
            blit_bind_group,
            render_target,
            timestamp_supported,
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period,
        })
    }

    fn upload_svdag(&mut self, svdag: &Svdag, volume_size: u32, camera: BenchCamera) {
        let uniforms = Uniforms {
            camera_pos: [camera.pos[0], camera.pos[1], camera.pos[2], 0.0],
            camera_dir: [camera.dir[0], camera.dir[1], camera.dir[2], 0.0],
            camera_up: [camera.up[0], camera.up[1], camera.up[2], 0.0],
            camera_right: [camera.right[0], camera.right[1], camera.right[2], 0.0],
            params: [
                volume_size as f32,
                RENDER_WIDTH as f32 / RENDER_HEIGHT as f32,
                (std::f32::consts::FRAC_PI_4).tan(),
                RENDER_HEIGHT as f32,
            ],
            debug: [0.0, 0.0, RENDER_WIDTH as f32, RENDER_HEIGHT as f32],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let data = bytemuck::cast_slice(&svdag.nodes);
        let dag_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dag_nodes"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.compute_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("svdag_compute_bg"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dag_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.palette_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.raycast_texture_view),
                },
            ],
        }));

        self.dag_buffer = Some(dag_buffer);
    }

    /// Render one frame (compute raycast + blit). Returns GPU duration if
    /// timestamps available, else wall-clock duration.
    fn render_frame(&self) -> Duration {
        let bg = self
            .compute_bind_group
            .as_ref()
            .expect("must upload_svdag first");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench_enc"),
            });

        // Begin timestamp
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, 0);
        }

        // Compute pass: SVDAG raycast → storage texture
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("svdag raycast compute"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, bg, &[]);
            let wg_x = RENDER_WIDTH.div_ceil(8);
            let wg_y = RENDER_HEIGHT.div_ceil(8);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Render pass: blit storage texture → render target
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
            render_pass.draw(0..3, 0..1); // fullscreen triangle
        }

        // End timestamp + resolve
        if let (Some(qs), Some(resolve)) = (&self.query_set, &self.resolve_buffer) {
            encoder.write_timestamp(qs, 1);
            encoder.resolve_query_set(qs, 0..2, resolve, 0);
            if let Some(readback) = &self.readback_buffer {
                encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
            }
        }

        let wall_start = Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));

        // If timestamps are available, read them back synchronously.
        if let Some(readback) = &self.readback_buffer {
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().ok().and_then(|r| r.ok()).is_some() {
                let data = slice.get_mapped_range();
                let timestamps: &[u64] = bytemuck::cast_slice(&data);
                if timestamps.len() >= 2 && timestamps[1] > timestamps[0] {
                    let ticks = timestamps[1] - timestamps[0];
                    let nanos = ticks as f64 * self.timestamp_period as f64;
                    drop(data);
                    readback.unmap();
                    return Duration::from_nanos(nanos as u64);
                }
                drop(data);
                readback.unmap();
            }
        } else {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }

        wall_start.elapsed()
    }
}

fn bench_raycast(label: &str, level: u32, frames: usize, camera: BenchCamera) {
    let side = 1u64 << level;
    eprintln!("--- {label}: GPU raycast benchmark (level={level}, side={side}³) ---");

    // Build world + SVDAG
    let build_start = Instant::now();
    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");
    let svdag = Svdag::build(&world.store, world.root, world.level);
    let build_ms = build_start.elapsed().as_millis();
    eprintln!(
        "  world+svdag built: {}ms, {} nodes, {:.1} MB",
        build_ms,
        svdag.node_count,
        svdag.byte_size() as f64 / (1024.0 * 1024.0),
    );

    // Set up headless renderer
    let mut renderer = match HeadlessRenderer::new() {
        Some(r) => r,
        None => {
            eprintln!("  SKIP: no GPU adapter available");
            return;
        }
    };

    eprintln!(
        "  GPU timestamps: {}",
        if renderer.timestamp_supported {
            "yes"
        } else {
            "no (wall-clock fallback)"
        }
    );

    renderer.upload_svdag(&svdag, side as u32, camera);

    // Warmup frame (shader compilation, etc.)
    let warmup = renderer.render_frame();
    eprintln!("  warmup frame: {:.2}ms", warmup.as_secs_f64() * 1000.0);

    // Benchmark frames
    let mut times = Vec::with_capacity(frames);
    let total_start = Instant::now();
    for i in 0..frames {
        let dt = renderer.render_frame();
        times.push(dt);
        if dt > Duration::from_secs(5) {
            eprintln!(
                "  ABORT: frame {} took {:.1}s (>5s limit)",
                i,
                dt.as_secs_f64()
            );
            break;
        }
        if total_start.elapsed() > Duration::from_secs(30) {
            eprintln!("  TIMEOUT: {} frames in 30s", i + 1);
            break;
        }
    }

    if times.is_empty() {
        eprintln!("  no frames completed");
        return;
    }

    let n = times.len();
    let total_ms: f64 = times.iter().map(|t| t.as_secs_f64() * 1000.0).sum();
    let mean_ms = total_ms / n as f64;
    let mut sorted: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1000.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[n / 2];
    let p95 = sorted[(n as f64 * 0.95) as usize];
    let fps = 1000.0 / mean_ms;

    eprintln!("  {n} frames: mean={mean_ms:.2}ms, p50={p50:.2}ms, p95={p95:.2}ms, ~{fps:.1} fps");
    eprintln!();
}

#[test]
#[ignore]
fn bench_raycast_64() {
    bench_raycast("64³", 6, 100, BenchCamera::empty_corner());
}

#[test]
#[ignore]
fn bench_raycast_256() {
    bench_raycast("256³", 8, 100, BenchCamera::empty_corner());
}

#[test]
#[ignore]
fn bench_raycast_256_app_spawn() {
    bench_raycast("256³ app-spawn", 8, 100, BenchCamera::app_spawn(256));
}

/// Headless game/app smoke path for agents and CI-like environments.
///
/// Runs the real startup ingredients we can support without a visible surface:
/// seed terrain, add dynamic CA materials, render one offscreen frame, step the
/// world once, rebuild the SVDAG, and render again.
///
/// Preferred command:
/// `cargo test --profile bench --test bench_gpu_raycast headless_game_smoke_256 -- --ignored --nocapture`
///
/// The default test profile also works, but it takes close to a minute at
/// 256³ because the simulated step runs unoptimized.
#[test]
#[ignore]
fn headless_game_smoke_256() {
    let level = 8;
    let side = 1u64 << level;

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");
    world.seed_water_and_sand();

    let mut renderer = match HeadlessRenderer::new() {
        Some(r) => r,
        None => {
            eprintln!("SKIP: no GPU adapter available for headless smoke");
            return;
        }
    };

    let mut svdag = Svdag::new();
    svdag.update(&world.store, world.root, world.level);
    assert!(
        svdag.node_count > 0,
        "seeded world should build a non-empty SVDAG"
    );

    renderer.upload_svdag(&svdag, side as u32, BenchCamera::app_spawn(side as u32));
    let first = renderer.render_frame();
    assert!(
        first < Duration::from_secs(5),
        "initial headless frame should complete quickly, got {:.2}s",
        first.as_secs_f64()
    );

    world.step();
    assert!(world.generation >= 1, "smoke step should advance the world");

    svdag.update(&world.store, world.root, world.level);
    assert!(
        svdag.node_count > 0,
        "post-step world should still build a non-empty SVDAG"
    );
    renderer.upload_svdag(&svdag, side as u32, BenchCamera::app_spawn(side as u32));
    let second = renderer.render_frame();
    assert!(
        second < Duration::from_secs(5),
        "post-step headless frame should complete quickly, got {:.2}s",
        second.as_secs_f64()
    );
}

#[test]
#[ignore]
fn bench_raycast_512() {
    bench_raycast("512³", 9, 50, BenchCamera::empty_corner());
}

#[test]
#[ignore]
fn bench_raycast_1024() {
    bench_raycast("1024³", 10, 30, BenchCamera::empty_corner());
}

#[test]
#[ignore]
fn bench_raycast_2048() {
    bench_raycast("2048³", 11, 20, BenchCamera::empty_corner());
}

#[test]
#[ignore]
fn bench_raycast_4096() {
    bench_raycast("4096³", 12, 10, BenchCamera::empty_corner());
}

/// Seeded-terrain variant of the 4096³ raycast bench (hash-thing-e092).
/// The world is already seeded in `bench_raycast`; this variant swaps the
/// camera to `BenchCamera::app_spawn(4096)`, which mirrors the in-app spawn
/// pose and produces the hit-dominated primary-ray traffic the perf paper's
/// §3.10.5 envelope is derived against. Pair with `bench_raycast_4096`
/// (empty-corner / sky-heavy, best-case floor) to bracket the envelope
/// between best-case and typical-scene framerate.
#[test]
#[ignore]
fn bench_raycast_4096_app_spawn() {
    bench_raycast("4096³ app-spawn", 12, 10, BenchCamera::app_spawn(4096));
}

/// 8192³ (level 13) characterization bench. `DEFAULT_VOLUME_SIZE` was
/// bumped to 8192 in hash-thing-69cq; this pair confirms launch
/// feasibility + measures frame time on integrated Metal GPUs
/// (hash-thing-wirw). Frame count kept low (5) because each frame is
/// expected to exceed the per-frame cap at 4096³ and the harness caps
/// total wall time at ~30s per scale.
#[test]
#[ignore]
fn bench_raycast_8192() {
    bench_raycast("8192³", 13, 5, BenchCamera::empty_corner());
}

#[test]
#[ignore]
fn bench_raycast_8192_app_spawn() {
    bench_raycast("8192³ app-spawn", 13, 5, BenchCamera::app_spawn(8192));
}

/// Combined benchmark: SVDAG build + CA step + rebuild + raycast.
/// Simulates real gameplay: step the CA, rebuild SVDAG, render.
#[test]
#[ignore]
fn bench_raycast_with_active_ca() {
    let level = 10; // 1024³
    let side = 1u64 << level;
    eprintln!("--- Active CA benchmark: {side}³ with stepping ---");

    let mut world = World::new(level);
    let _ = world
        .seed_terrain(&TerrainParams::for_level(level))
        .expect("level-derived terrain params must validate");

    let mut renderer = match HeadlessRenderer::new() {
        Some(r) => r,
        None => {
            eprintln!("  SKIP: no GPU adapter available");
            return;
        }
    };

    let mut svdag = Svdag::new();
    svdag.update(&world.store, world.root, world.level);
    renderer.upload_svdag(&svdag, side as u32, BenchCamera::empty_corner());

    // Warmup
    let _ = renderer.render_frame();

    eprintln!(
        "  GPU timestamps: {}",
        if renderer.timestamp_supported {
            "yes"
        } else {
            "no (wall-clock fallback)"
        }
    );

    let total_start = Instant::now();
    for gen in 0..5 {
        let step_t = Instant::now();
        world.step();
        let step_ms = step_t.elapsed().as_millis();

        let build_t = Instant::now();
        svdag.update(&world.store, world.root, world.level);
        let build_ms = build_t.elapsed().as_millis();

        renderer.upload_svdag(&svdag, side as u32, BenchCamera::empty_corner());
        let render_dt = renderer.render_frame();
        let render_ms = render_dt.as_secs_f64() * 1000.0;

        eprintln!(
            "  gen {}: step={}ms, svdag={}ms, render={:.2}ms, total={}ms",
            gen + 1,
            step_ms,
            build_ms,
            render_ms,
            step_t.elapsed().as_millis(),
        );

        if total_start.elapsed() > Duration::from_secs(30) {
            eprintln!("  TIMEOUT at gen {}", gen + 1);
            break;
        }
    }
    eprintln!();
}
