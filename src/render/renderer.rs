use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::window::Window;

// GpuTiming state-machine states. Stored as AtomicU8 because the
// map_async callback fires on whichever thread calls `device.poll`
// (the main thread in our app, but the API signature demands a
// `'static + WasmNotSend` callback, so we synchronize anyway).
const GT_IDLE: u8 = 0;
const GT_PENDING: u8 = 1;
const GT_READY: u8 = 2;

/// Two `u64`s (start + end timestamp) = 16 bytes. `QUERY_RESOLVE_BUFFER_ALIGNMENT`
/// (256) applies to the *destination offset* of `resolve_query_set`, not the
/// buffer size itself — we resolve at offset 0 which is trivially aligned, and
/// `resolve_query_set` only requires the buffer to hold `query_count * 8`
/// bytes (per wgpu-core/src/command/query.rs:475). 16 bytes is exactly that.
const TIMESTAMP_BYTES: u64 = 16;

/// Convert raw GPU timestamp ticks to a `Duration`, using the
/// adapter-reported `timestamp_period` (nanoseconds per tick). Pure
/// function — unit-testable without spinning up a real device.
///
/// Saturates end < start (should only happen if the backend reports
/// non-monotonic ticks, which would be a driver bug, but we'd rather
/// log a zero than underflow).
fn ticks_to_duration(start_ticks: u64, end_ticks: u64, period_ns: f32) -> Duration {
    let delta = end_ticks.saturating_sub(start_ticks);
    // f64 multiply — f32 loses precision for large tick counts
    // (periods are typically < 1000ns, ticks can be in the millions).
    let ns = (delta as f64) * (period_ns as f64);
    // u64 cast saturates on overflow in `as` for positive values on Rust
    // stable. Ceiling it at u64::MAX is fine; such a value would be
    // "absurdly large" per the test rubric and we'd catch it upstream.
    Duration::from_nanos(ns as u64)
}

/// wgpu requires `bytes_per_row` on multi-row texture copies to be a multiple
/// of `COPY_BYTES_PER_ROW_ALIGNMENT` (256). For a 64³ R8Uint volume the natural
/// row length is 64 bytes, which is not aligned — spec-violating even if it
/// happens to work on most backends today. When the format widens to R16Uint
/// (hash-thing-1v0.1) the same row becomes 128 bytes, still not aligned. This
/// helper returns the padded row stride in bytes; callers use it as the upload
/// `bytes_per_row` and pad rows in a staging buffer if the padded value
/// differs from the unpadded one.
fn padded_bytes_per_row(width: u32, bytes_per_texel: u32) -> u32 {
    let unpadded = width * bytes_per_texel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    unpadded.div_ceil(align) * align
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera_pos: [f32; 4],
    camera_dir: [f32; 4],
    camera_up: [f32; 4],
    camera_right: [f32; 4],
    /// x: volume_size, y: aspect_ratio, z: fov_tan, w: time
    params: [f32; 4],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode {
    Flat3D,
    Svdag,
}

/// Outcome of a single `Renderer::render` call. Replaces an earlier `bool`
/// return so the caller can distinguish "frame submitted" from "skip — window
/// is invisible" and stop hot-spinning `request_redraw` while occluded
/// (hash-thing-8jp).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameOutcome {
    /// Frame submitted successfully.
    Rendered,
    /// Surface is occluded (minimized / hidden). Caller should pause redraw
    /// requests until the window becomes visible again.
    Occluded,
    /// Surface was outdated or lost; we reconfigured. Caller should request
    /// the next frame to redraw with the new configuration.
    Reconfigured,
    /// Frame acquisition timed out. Caller should request the next frame;
    /// this is expected transient behavior.
    Timeout,
}

/// GPU-side render pass timing via `wgpu::Features::TIMESTAMP_QUERY`.
///
/// Owns a 2-entry `QuerySet` (pass start + pass end), a resolve buffer
/// (`QUERY_RESOLVE | COPY_SRC`), and a readback buffer (`MAP_READ |
/// COPY_DST`). Each frame, if state == IDLE, the renderer attaches
/// `timestamp_writes` to its render pass descriptor, resolves the set
/// into the resolve buffer, copies to the readback buffer, submits, and
/// issues `map_async`. The callback flips the state to READY. The next
/// frame, `poll()` reads the two `u64`s, converts to `Duration`, unmaps,
/// and returns the value.
///
/// **One readback in flight.** If the previous readback isn't done by
/// the time the next frame hits `try_begin()`, we skip instrumentation
/// for that frame (state != IDLE). That's the "drop samples that are
/// still unmapped when the next frame lands" rule from hash-thing-6x3.
/// Simpler than a ring of staging buffers, and the frames we do capture
/// are still plenty for mean/p95 at winit-Poll cadence.
///
/// **Why AtomicU8 instead of a plain state field.** `map_async`'s
/// callback is typed `impl FnOnce + WasmNotSend + 'static`, which means
/// the closure can't borrow `&mut` anything from self. We need a way
/// for the callback to signal "readback is now ready" back to the
/// render thread. `Arc<AtomicU8>` is the minimum synchronization that
/// lets the closure flip a shared bit.
struct GpuTiming {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    /// Nanoseconds per tick, from `Queue::get_timestamp_period()`.
    /// Varies by adapter; some report 1ns, others 100ns or more.
    period_ns: f32,
    /// Shared state flag. `GT_IDLE`/`GT_PENDING`/`GT_READY`. The
    /// `map_async` callback transitions PENDING → READY on success or
    /// PENDING → IDLE on failure. The render thread transitions IDLE →
    /// PENDING (in `request_readback`) and READY → IDLE (in `poll`).
    state: Arc<AtomicU8>,
}

impl GpuTiming {
    fn new(device: &wgpu::Device, period_ns: f32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timing_qs"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        // wgpu requires resolve buffer size to be at least 16 bytes (2
        // u64 timestamps). `QUERY_RESOLVE_BUFFER_ALIGNMENT` is 256; the
        // allocator rounds up anyway so a 16-byte request is fine.
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timing_resolve"),
            size: TIMESTAMP_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timing_readback"),
            size: TIMESTAMP_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            period_ns,
            state: Arc::new(AtomicU8::new(GT_IDLE)),
        }
    }

    /// If idle, return a `RenderPassTimestampWrites` that captures
    /// pass start + pass end. Returns `None` if a prior readback is
    /// still in flight.
    ///
    /// Does NOT transition state — state stays IDLE until
    /// `request_readback` fires after submit. This lets the caller
    /// back out if it ends up not submitting (e.g. surface occluded
    /// between the check and the draw).
    fn pass_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if self.state.load(Ordering::Acquire) == GT_IDLE {
            Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            })
        } else {
            None
        }
    }

    /// Append `resolve_query_set` + `copy_buffer_to_buffer` commands to
    /// `encoder`. Call only when `pass_writes()` returned `Some` AND the
    /// pass actually submitted (i.e. encoder is about to be finished).
    fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            TIMESTAMP_BYTES,
        );
    }

    /// Start async readback. Transitions IDLE → PENDING and issues
    /// `map_async`; the callback later flips PENDING → READY (success)
    /// or PENDING → IDLE (failure). Call exactly once per `encode_resolve`.
    fn request_readback(&self) {
        if self
            .state
            .compare_exchange(GT_IDLE, GT_PENDING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // Someone else already queued a readback. Shouldn't happen
            // with our single-render-thread model, but silently skip
            // rather than panic.
            return;
        }
        let state = self.state.clone();
        self.readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => state.store(GT_READY, Ordering::Release),
                Err(e) => {
                    log::warn!("gpu timing readback failed: {e:?}");
                    state.store(GT_IDLE, Ordering::Release);
                }
            });
    }

    /// Pump `map_async` callbacks via `device.poll`, then consume a
    /// pending readback if one just became ready. Returns the
    /// resolved `Duration` for the most recently completed frame, or
    /// `None` if no readback is ready.
    ///
    /// Transitions READY → IDLE on success.
    fn poll(&self, device: &wgpu::Device) -> Option<Duration> {
        // Pump callbacks. `PollType::Poll` is non-blocking: it processes
        // whatever's ready and returns immediately. Cheap when there's
        // nothing in flight.
        let _ = device.poll(wgpu::PollType::Poll);

        if self.state.load(Ordering::Acquire) != GT_READY {
            return None;
        }

        let slice = self.readback_buffer.slice(..);
        let data = slice.get_mapped_range();
        // Two little-endian u64s. wgpu reports timestamps in the
        // backend's native byte order, which on every platform we care
        // about is LE. Use from_le_bytes to be explicit.
        let start = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let end = u64::from_le_bytes(data[8..16].try_into().unwrap());
        drop(data);
        self.readback_buffer.unmap();

        // State transitions to IDLE *after* the unmap call, since
        // request_readback asserts the buffer is not already mapped.
        self.state.store(GT_IDLE, Ordering::Release);

        Some(ticks_to_duration(start, end, self.period_ns))
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Flat 3D texture path
    flat_pipeline: wgpu::RenderPipeline,
    flat_bind_group: wgpu::BindGroup,
    volume_texture: wgpu::Texture,

    // SVDAG path
    svdag_pipeline: wgpu::RenderPipeline,
    svdag_bind_group_layout: wgpu::BindGroupLayout,
    svdag_bind_group: Option<wgpu::BindGroup>,
    svdag_buffer: Option<wgpu::Buffer>,
    svdag_buffer_cap: u64, // current allocation in bytes

    // Shared
    uniform_buffer: wgpu::Buffer,
    volume_size: u32,
    /// Bytes-per-texel for the volume format. R8Uint = 1, R16Uint = 2. Stored
    /// alongside `volume_size` so `upload_volume` can compute row padding
    /// without re-deriving it from the texture format enum.
    volume_bytes_per_texel: u32,
    pub mode: RenderMode,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_dist: f32,
    pub camera_target: [f32; 3],

    // GPU-timestamp instrumentation. `None` on adapters without
    // `Features::TIMESTAMP_QUERY` — all timing falls back to the CPU
    // ring in that case (see `hash-thing-6x3`).
    gpu_timing: Option<GpuTiming>,
    /// Most recently resolved GPU render-pass duration. Set by `render()`
    /// when a readback completes, consumed by `take_last_gpu_frame_time()`.
    /// `None` means no new sample since the last take (or the adapter
    /// lacks TIMESTAMP_QUERY entirely). Consume-on-read avoids
    /// double-recording the same duration across frames.
    last_gpu_frame_time: Option<Duration>,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, volume_size: u32) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: Default::default(),
            display: Default::default(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        // hash-thing-6x3: TIMESTAMP_QUERY is a soft requirement — we
        // enable it when the adapter supports it, fall back to CPU-only
        // perf when not. The feature is missing on some older MoltenVK
        // setups and on WebGPU adapters where the spec hasn't stabilized
        // the capability yet. Falling back gracefully keeps the app
        // running on every adapter wgpu can talk to.
        //
        // We check `adapter.features()` (what the adapter *can* expose)
        // not `adapter.limits()` — the latter is the defaults, not the
        // capabilities. `required_features` must be a subset of
        // `adapter.features()` or `request_device` fails.
        let mut required_features = wgpu::Features::empty();
        let adapter_features = adapter.features();
        let timestamp_supported = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        } else {
            log::info!(
                "GPU adapter lacks TIMESTAMP_QUERY — perf will report CPU submit overhead only"
            );
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hash-thing device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume"),
            size: wgpu::Extent3d {
                width: volume_size,
                height: volume_size,
                depth_or_array_layers: volume_size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let volume_view = volume_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let uniforms = Uniforms {
            camera_pos: [0.0; 4],
            camera_dir: [0.0; 4],
            camera_up: [0.0, 1.0, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            params: [volume_size as f32, 1.0, 1.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Flat 3D texture pipeline ===

        let flat_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flat_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let flat_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat_bg"),
            layout: &flat_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
            ],
        });

        let flat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat raycast shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raycast.wgsl").into()),
        });

        let flat_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flat_pl"),
            bind_group_layouts: &[Some(&flat_bind_group_layout)],
            immediate_size: 0,
        });

        let flat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flat_rp"),
            layout: Some(&flat_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &flat_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &flat_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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

        // === SVDAG pipeline ===

        let svdag_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("svdag_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let svdag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svdag raycast shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("svdag_raycast.wgsl").into()),
        });

        let svdag_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("svdag_pl"),
                bind_group_layouts: &[Some(&svdag_bind_group_layout)],
                immediate_size: 0,
            });

        let svdag_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("svdag_rp"),
            layout: Some(&svdag_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &svdag_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &svdag_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
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

        let gpu_timing = if timestamp_supported {
            let period_ns = queue.get_timestamp_period();
            Some(GpuTiming::new(&device, period_ns))
        } else {
            None
        };

        Self {
            surface,
            device,
            queue,
            config,
            flat_pipeline,
            flat_bind_group,
            volume_texture,
            svdag_pipeline,
            svdag_bind_group_layout,
            svdag_bind_group: None,
            svdag_buffer: None,
            svdag_buffer_cap: 0,
            uniform_buffer,
            volume_size,
            // R16Uint: 2 bytes per texel. Update alongside the `format:` line
            // above if the texture format ever widens again.
            volume_bytes_per_texel: 2,
            mode: RenderMode::Svdag,
            camera_yaw: std::f32::consts::FRAC_PI_4,
            camera_pitch: 0.4,
            camera_dist: 2.0,
            camera_target: [0.5, 0.5, 0.5],
            gpu_timing,
            last_gpu_frame_time: None,
        }
    }

    /// Consume and return the most recently resolved GPU render-pass
    /// duration, or `None` if no new sample has been captured since the
    /// last call. Call once per frame after `render()` returns; the
    /// value is intended to be fed into `Perf` as the `render_gpu`
    /// metric (see `hash-thing-6x3`).
    ///
    /// Adapters without `Features::TIMESTAMP_QUERY` always return
    /// `None` — in that case the only render metric is the CPU-submit
    /// `render_cpu` from `main.rs`.
    pub fn take_last_gpu_frame_time(&mut self) -> Option<Duration> {
        self.last_gpu_frame_time.take()
    }

    /// Upload (or re-upload) a serialized SVDAG to the GPU.
    ///
    /// hash-thing-2w5: the shader derives its step budget from `params.x`,
    /// which is `self.volume_size as f32`. The CPU-side raycast replica uses
    /// `dag.root_level` directly. For the two to agree byte-for-byte we
    /// need `self.volume_size == 1 << dag.root_level`. Today this holds by
    /// coincidence — `main.rs` threads the same `VOLUME_SIZE` into both the
    /// renderer and `sim::World::new(VOLUME_SIZE.trailing_zeros())`, and
    /// nothing else resizes either. A future caller that hands us a DAG
    /// whose root_level differs from the renderer's captured volume_size
    /// would silently desync the shader step budget from the CPU replica's
    /// regression suite, so we pin the invariant here with a debug_assert.
    /// When this fires, either wire a dedicated `root_level` uniform field
    /// or call a `set_root_level` helper before `upload_svdag`.
    pub fn upload_svdag(&mut self, dag: &super::Svdag) {
        debug_assert_eq!(
            self.volume_size,
            1u32 << dag.root_level,
            "upload_svdag: renderer.volume_size ({}) must equal 1 << dag.root_level ({}); \
             shader/CPU step budget would desync otherwise (hash-thing-2w5)",
            self.volume_size,
            1u32 << dag.root_level,
        );
        let bytes: &[u8] = bytemuck::cast_slice(&dag.nodes);
        let needed = bytes.len() as u64;

        // Allocate a larger buffer if necessary (grow-only)
        if self.svdag_buffer.is_none() || needed > self.svdag_buffer_cap {
            // Grow to next power of 2 with a floor of 64KB
            let mut cap = self.svdag_buffer_cap.max(65536);
            while cap < needed {
                cap *= 2;
            }
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("svdag_buffer"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.svdag_buffer = Some(buffer);
            self.svdag_buffer_cap = cap;

            // Rebuild bind group
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("svdag_bg"),
                layout: &self.svdag_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.svdag_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            });
            self.svdag_bind_group = Some(bg);
        }

        if let Some(buf) = &self.svdag_buffer {
            self.queue.write_buffer(buf, 0, bytes);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Upload a full `volume_size³` cell grid to the R16Uint 3D texture.
    /// `data` is one `u16` per voxel in x-major, y-stride, z-slice order.
    /// Row strides are padded up to `COPY_BYTES_PER_ROW_ALIGNMENT` via
    /// `padded_bytes_per_row` — see that helper for the full story.
    pub fn upload_volume(&self, data: &[u16]) {
        let w = self.volume_size;
        let bpt = self.volume_bytes_per_texel;
        let unpadded = w * bpt;
        let padded = padded_bytes_per_row(w, bpt);
        let data_bytes: &[u8] = bytemuck::cast_slice(data);

        // Fast path: row length already aligned, pass the caller's slice
        // straight through. Slow path: copy row-by-row into a padded staging
        // buffer. Per-upload allocation is fine — uploads are infrequent
        // (once per generation tick) and ~128KiB for 64³ R16Uint.
        let staging: Vec<u8>;
        let (rows_data, row_stride) = if padded == unpadded {
            (data_bytes, unpadded)
        } else {
            let total_rows = (w * w) as usize; // height × depth
            let mut buf = vec![0u8; total_rows * padded as usize];
            let unpadded_usize = unpadded as usize;
            let padded_usize = padded as usize;
            for row in 0..total_rows {
                let src = &data_bytes[row * unpadded_usize..][..unpadded_usize];
                let dst = &mut buf[row * padded_usize..][..unpadded_usize];
                dst.copy_from_slice(src);
            }
            staging = buf;
            (staging.as_slice(), padded)
        };

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rows_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_stride),
                rows_per_image: Some(w),
            },
            wgpu::Extent3d {
                width: w,
                height: w,
                depth_or_array_layers: w,
            },
        );
    }

    pub fn render(&mut self) -> FrameOutcome {
        use wgpu::CurrentSurfaceTexture;

        // hash-thing-6x3: pump `map_async` callbacks and consume any
        // GPU-timestamp readback that landed since the last frame.
        // `GpuTiming::poll` calls `device.poll(Poll)` internally, so this
        // is the single place we drive wgpu's main-thread callback pump.
        // Must happen before we potentially skip the frame on surface
        // failures — otherwise a long run of Occluded frames would
        // accumulate un-polled callbacks.
        if let Some(gt) = &self.gpu_timing {
            if let Some(d) = gt.poll(&self.device) {
                self.last_gpu_frame_time = Some(d);
            }
        }

        // No catch-all: `CurrentSurfaceTexture` is not `#[non_exhaustive]`, so
        // any future wgpu variant becomes a compile error pointing here,
        // instead of getting silently swallowed (hash-thing-8jp I1a).
        let surface_texture = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(tex) | CurrentSurfaceTexture::Suboptimal(tex) => tex,
            CurrentSurfaceTexture::Occluded => return FrameOutcome::Occluded,
            CurrentSurfaceTexture::Timeout => {
                log::warn!("surface texture acquire: Timeout");
                return FrameOutcome::Timeout;
            }
            CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return FrameOutcome::Reconfigured;
            }
            CurrentSurfaceTexture::Lost => {
                log::error!("surface texture Lost — reconfiguring");
                self.surface.configure(&self.device, &self.config);
                return FrameOutcome::Reconfigured;
            }
            // A validation error was raised inside `get_current_texture`
            // and captured by an error scope. The surface itself is fine —
            // no reconfigure needed — but the caller should request the
            // next frame so rendering resumes after the programmer/driver
            // issue is resolved. Logged at error level so it actually
            // surfaces during GPU debugging.
            CurrentSurfaceTexture::Validation => {
                log::error!("surface texture acquire: Validation error");
                return FrameOutcome::Timeout;
            }
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Camera
        let (sin_yaw, cos_yaw) = self.camera_yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.camera_pitch.sin_cos();

        let cam_dir = [-cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw];
        let cam_pos = [
            self.camera_target[0] - cam_dir[0] * self.camera_dist,
            self.camera_target[1] - cam_dir[1] * self.camera_dist,
            self.camera_target[2] - cam_dir[2] * self.camera_dist,
        ];

        let right = [cos_yaw, 0.0, -sin_yaw];
        let up = [
            right[1] * cam_dir[2] - right[2] * cam_dir[1],
            right[2] * cam_dir[0] - right[0] * cam_dir[2],
            right[0] * cam_dir[1] - right[1] * cam_dir[0],
        ];

        let aspect = self.config.width as f32 / self.config.height as f32;
        let fov_tan = (std::f32::consts::FRAC_PI_4 / 2.0).tan();

        let uniforms = Uniforms {
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            camera_dir: [cam_dir[0], cam_dir[1], cam_dir[2], 0.0],
            camera_up: [up[0], up[1], up[2], 0.0],
            camera_right: [right[0], right[1], right[2], 0.0],
            params: [self.volume_size as f32, aspect, fov_tan, 0.0],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        // hash-thing-6x3: if the previous frame's readback is done,
        // instrument this frame. Otherwise skip — one readback in flight
        // is enough for the mean/p95 the perf line reports.
        //
        // The `Option<RenderPassTimestampWrites>` holds a `&QuerySet`
        // borrowed from `self.gpu_timing`, so it must live across the
        // render-pass scope but can be dropped before the post-pass
        // `encode_resolve` call.
        let timestamp_writes = self.gpu_timing.as_ref().and_then(|gt| gt.pass_writes());
        let captured_this_frame = timestamp_writes.is_some();

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            match self.mode {
                RenderMode::Flat3D => {
                    render_pass.set_pipeline(&self.flat_pipeline);
                    render_pass.set_bind_group(0, &self.flat_bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
                RenderMode::Svdag => {
                    if let Some(bg) = &self.svdag_bind_group {
                        render_pass.set_pipeline(&self.svdag_pipeline);
                        render_pass.set_bind_group(0, bg, &[]);
                        render_pass.draw(0..6, 0..1);
                    } else {
                        // No SVDAG uploaded yet — fall back to flat path.
                        render_pass.set_pipeline(&self.flat_pipeline);
                        render_pass.set_bind_group(0, &self.flat_bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
        }

        if captured_this_frame {
            if let Some(gt) = &self.gpu_timing {
                gt.encode_resolve(&mut encoder);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        if captured_this_frame {
            if let Some(gt) = &self.gpu_timing {
                gt.request_readback();
            }
        }

        FrameOutcome::Rendered
    }
}

#[cfg(test)]
mod tests {
    use super::{padded_bytes_per_row, ticks_to_duration, FrameOutcome};
    use std::time::Duration;

    const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    #[test]
    fn ticks_to_duration_period_one_is_identity() {
        // period=1ns → 1 tick is 1ns. Lets the test read as a
        // "literal tick count is ns count" identity check.
        assert_eq!(
            ticks_to_duration(0, 1_000, 1.0),
            Duration::from_nanos(1_000)
        );
    }

    #[test]
    fn ticks_to_duration_scales_by_period() {
        // 1000 ticks × 100ns/tick = 100_000 ns = 100 µs.
        assert_eq!(
            ticks_to_duration(0, 1_000, 100.0),
            Duration::from_nanos(100_000),
        );
    }

    #[test]
    fn ticks_to_duration_handles_offset_pair() {
        // Adapters don't reset the tick counter at frame start — end_ticks
        // is a large absolute value, duration comes from the delta.
        let dur = ticks_to_duration(10_000_000, 10_001_000, 1.0);
        assert_eq!(dur, Duration::from_nanos(1_000));
    }

    #[test]
    fn ticks_to_duration_saturates_on_inverted_pair() {
        // Driver bug or wraparound: end < start. We return zero rather
        // than underflow. Inverted samples are still recorded (as
        // zero), which shows up in the `render_gpu` summary as a
        // suspiciously-low mean that the user can investigate.
        assert_eq!(ticks_to_duration(500, 100, 1.0), Duration::ZERO);
    }

    #[test]
    fn frame_outcome_variants_are_distinct() {
        // hash-thing-8jp — if a future refactor collapses or removes a
        // variant, this pins the semantic split between the four cases so
        // the main-loop gate (`matches!(_, Occluded)`) keeps working.
        let all = [
            FrameOutcome::Rendered,
            FrameOutcome::Occluded,
            FrameOutcome::Reconfigured,
            FrameOutcome::Timeout,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(a == b, i == j, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn padded_row_rounds_up_to_alignment() {
        // 64 R8Uint texels = 64 bytes; rounds up to the 256-byte alignment.
        assert_eq!(padded_bytes_per_row(64, 1), ALIGN);
    }

    #[test]
    fn padded_row_respects_bytes_per_texel() {
        // 64 R16Uint texels = 128 bytes; still under ALIGN, still rounds up.
        assert_eq!(padded_bytes_per_row(64, 2), ALIGN);
    }

    #[test]
    fn padded_row_passes_through_already_aligned() {
        // 256 R8Uint texels = 256 bytes exactly: one alignment unit, no padding.
        assert_eq!(padded_bytes_per_row(ALIGN, 1), ALIGN);
        // 128 R16Uint texels = 256 bytes: same story.
        assert_eq!(padded_bytes_per_row(ALIGN / 2, 2), ALIGN);
    }

    #[test]
    fn padded_row_rounds_up_beyond_single_unit() {
        // 300 R8Uint texels = 300 bytes; rounds up to 512 (next alignment unit).
        assert_eq!(padded_bytes_per_row(300, 1), 2 * ALIGN);
        // 300 R16Uint texels = 600 bytes; rounds up to 768 (3 alignment units).
        assert_eq!(padded_bytes_per_row(300, 2), 3 * ALIGN);
    }
}
