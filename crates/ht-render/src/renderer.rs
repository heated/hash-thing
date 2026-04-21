use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RendererLifecycleSnapshot {
    has_svdag_buffer: bool,
    has_particle_buffer: bool,
    particle_count: u32,
}

impl RendererLifecycleSnapshot {
    fn from_renderer(renderer: &Renderer) -> Self {
        Self {
            has_svdag_buffer: renderer.svdag_buffer.is_some(),
            has_particle_buffer: renderer.particle_buffer.is_some(),
            particle_count: renderer.particle_count,
        }
    }

    fn rebuild_compute_bind_group_on_texture_recreate(self) -> bool {
        self.has_svdag_buffer
    }

    fn rebuild_compute_bind_group_on_palette_upload(self) -> bool {
        self.has_svdag_buffer
    }

    fn rebuild_particle_bind_group_on_texture_recreate(self) -> bool {
        self.particle_count > 0
    }

    fn rebuild_particle_bind_group_on_palette_upload(self) -> bool {
        self.has_particle_buffer
    }

    fn particle_bind_group_should_exist(self) -> bool {
        self.has_particle_buffer && self.particle_count > 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera_pos: [f32; 4],
    camera_dir: [f32; 4],
    camera_up: [f32; 4],
    camera_right: [f32; 4],
    /// x: volume_size, y: aspect_ratio, z: fov_tan, w: screen_height
    params: [f32; 4],
    /// x: debug_mode (0=normal, 1=step-count heatmap), y/z/w: reserved
    debug: [f32; 4],
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererCpuPhaseTimes {
    pub surface_acquire: Duration,
    pub submit: Duration,
    pub present: Duration,
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
    /// When true, the caller sandwiches the compute pass with
    /// `write_timestamp_begin`/`write_timestamp_end` on the encoder and
    /// `compute_pass_writes()` returns `None`. Requires the adapter to
    /// support `TIMESTAMP_QUERY_INSIDE_ENCODERS`. See hash-thing-dlse.2.3
    /// for why: in-pass timestamps on Metal include barrier/sync waits,
    /// which inflates "compute time" with cross-frame dependencies.
    use_in_encoder: bool,
}

impl GpuTiming {
    /// `label_prefix` gets concatenated with `_qs`, `_resolve`, `_readback`
    /// for the three wgpu objects; distinguishes multiple instances in
    /// RenderDoc / Metal capture tooling.
    fn new(
        device: &wgpu::Device,
        period_ns: f32,
        use_in_encoder: bool,
        label_prefix: &str,
    ) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(&format!("{label_prefix}_qs")),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        // wgpu requires resolve buffer size to be at least 16 bytes (2
        // u64 timestamps). `QUERY_RESOLVE_BUFFER_ALIGNMENT` is 256; the
        // allocator rounds up anyway so a 16-byte request is fine.
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}_resolve")),
            size: TIMESTAMP_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label_prefix}_readback")),
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
            use_in_encoder,
        }
    }

    fn is_idle(&self) -> bool {
        self.state.load(Ordering::Acquire) == GT_IDLE
    }

    fn uses_in_encoder(&self) -> bool {
        self.use_in_encoder
    }

    fn write_timestamp_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.write_timestamp(&self.query_set, 0);
    }

    fn write_timestamp_end(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.write_timestamp(&self.query_set, 1);
    }

    /// If idle, return a `RenderPassTimestampWrites` that captures
    /// pass start + pass end. Returns `None` if a prior readback is
    /// still in flight, OR when the adapter supports
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` — in that mode the caller is
    /// responsible for bracketing the render pass via
    /// `write_timestamp_begin`/`write_timestamp_end` on the encoder
    /// (mirrors `compute_pass_writes()`; avoids double-bracketing).
    ///
    /// Does NOT transition state — state stays IDLE until
    /// `request_readback` fires after submit. This lets the caller
    /// back out if it ends up not submitting (e.g. surface occluded
    /// between the check and the draw).
    fn pass_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if self.use_in_encoder {
            return None;
        }
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

    /// Compute-pass variant of `pass_writes()`. Used for timing the
    /// SVDAG raycast compute dispatch (hash-thing-5bb.6.1).
    ///
    /// Returns `None` when the adapter supports
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` — in that mode the caller is
    /// responsible for bracketing the compute pass via
    /// `write_timestamp_begin`/`write_timestamp_end` on the encoder.
    fn compute_pass_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if self.use_in_encoder {
            return None;
        }
        if self.state.load(Ordering::Acquire) == GT_IDLE {
            Some(wgpu::ComputePassTimestampWrites {
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

    /// Consume a pending readback if one just became ready. Returns
    /// the resolved `Duration` for the most recently completed frame,
    /// or `None` if no readback is ready.
    ///
    /// Does NOT call `device.poll` — callers must invoke
    /// `device.poll(Poll)` exactly once per frame (before polling any
    /// `GpuTiming` instances) to pump `map_async` callbacks. The split
    /// keeps the cost at one poll even when multiple timing brackets
    /// exist in the same frame (dlse.2.4).
    ///
    /// Transitions READY → IDLE on success.
    fn take_resolved(&self) -> Option<Duration> {
        if self.state.load(Ordering::Acquire) != GT_READY {
            return None;
        }

        let slice = self.readback_buffer.slice(..);
        let data = slice.get_mapped_range();
        // Two little-endian u64s. wgpu reports timestamps in the
        // backend's native byte order, which on every platform we care
        // about is LE. Use from_le_bytes to be explicit.
        let start = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .expect("GPU timestamp readback: first 8 bytes must convert to [u8; 8]"),
        );
        let end = u64::from_le_bytes(
            data[8..16]
                .try_into()
                .expect("GPU timestamp readback: second 8 bytes must convert to [u8; 8]"),
        );
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

    // SVDAG compute raycast path (hash-thing-5bb.6.1)
    svdag_compute_pipeline: wgpu::ComputePipeline,
    svdag_compute_bind_group_layout: wgpu::BindGroupLayout,
    svdag_compute_bind_group: Option<wgpu::BindGroup>,
    svdag_buffer: Option<wgpu::Buffer>,
    svdag_buffer_cap: u64, // current allocation in bytes
    /// Tracks how many u32s of `Svdag::nodes` have been uploaded to the GPU.
    /// Only the tail past this watermark (plus the root-offset header at slot 0)
    /// needs re-uploading each frame. Reset to 0 on buffer reallocation.
    svdag_uploaded_len: usize,
    /// Storage texture for compute raycast output (hash-thing-5bb.6.1).
    raycast_texture: wgpu::Texture,
    raycast_texture_view: wgpu::TextureView,

    // Blit pass: samples raycast_texture, writes to swapchain (5bb.6.1)
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,
    blit_sampler: wgpu::Sampler,

    // Particle billboard pass (hash-thing-5bb.9)
    particle_pipeline: wgpu::RenderPipeline,
    particle_bind_group_layout: wgpu::BindGroupLayout,
    particle_bind_group: Option<wgpu::BindGroup>,
    particle_buffer: Option<wgpu::Buffer>,
    particle_buffer_cap: u64,
    particle_count: u32,

    // HUD overlay (hash-thing-5bb.10)
    hud_pipeline: wgpu::RenderPipeline,
    hud_bind_group: wgpu::BindGroup,
    hud_uniform_buffer: wgpu::Buffer,
    pub hud_material_color: [f32; 4],
    pub hud_visible: bool,

    // Hotbar overlay (hash-thing-e7k.6)
    hotbar_pipeline: wgpu::RenderPipeline,
    hotbar_bind_group: wgpu::BindGroup,
    hotbar_uniform_buffer: wgpu::Buffer,
    /// Which material slot is selected (0-based: 0 = material 1/stone).
    pub hotbar_selected_slot: u32,
    pub hotbar_visible: bool,

    // Legend overlay (hash-thing-m1f.7.2)
    legend_pipeline: wgpu::RenderPipeline,
    legend_bind_group_layout: wgpu::BindGroupLayout,
    legend_bind_group: Option<wgpu::BindGroup>,
    legend_uniform_buffer: wgpu::Buffer,
    legend_texture: Option<wgpu::Texture>,
    legend_texture_view: Option<wgpu::TextureView>,
    legend_sampler: wgpu::Sampler,
    legend_tex_w: u32,
    legend_tex_h: u32,
    pub legend_visible: bool,

    // Material palette (shared by all pipelines)
    palette_buffer: wgpu::Buffer,

    // Shared
    uniform_buffer: wgpu::Buffer,
    volume_size: u32,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_dist: f32,
    pub camera_target: [f32; 3],

    /// Debug render mode. 0 = normal, 1 = step-count heatmap.
    pub debug_mode: u32,
    /// LOD bias multiplier. 1.0 = default, higher = more aggressive LOD.
    pub lod_bias: f32,
    /// Render resolution scale. 0.5 = half-res (4x fewer pixels), 1.0 = full.
    pub render_scale: f32,

    // GPU-timestamp instrumentation. `None` on adapters without
    // `Features::TIMESTAMP_QUERY` — all timing falls back to the CPU
    // ring in that case (see `hash-thing-6x3`).
    //
    // Two instances (dlse.2.4): one brackets the SVDAG compute pass,
    // the other brackets the blit + overlay render pass. The metric
    // `render_gpu` historically measures only the compute pass; the
    // new `render_pass_gpu` metric comes from `gpu_timing_render_pass`.
    // Kept as two independent state machines to preserve the
    // one-readback-in-flight invariant per pass.
    gpu_timing: Option<GpuTiming>,
    gpu_timing_render_pass: Option<GpuTiming>,
    /// Most recently resolved GPU compute-pass duration (dlse.2.3 has
    /// always bracketed the compute dispatch only, despite the name).
    /// Set by `render()` when a readback completes, consumed by
    /// `take_last_gpu_frame_time()`. `None` means no new sample since
    /// the last take (or the adapter lacks TIMESTAMP_QUERY entirely).
    /// Consume-on-read avoids double-recording across frames.
    last_gpu_frame_time: Option<Duration>,
    /// Most recently resolved GPU render-pass duration (blit + HUD +
    /// particles + hotbar + legend), bracketed in parallel with
    /// `last_gpu_frame_time`. Set by `render()`, consumed by
    /// `take_last_render_pass_gpu_frame_time()`. dlse.2.4.
    last_render_pass_gpu_frame_time: Option<Duration>,
    /// Most recent CPU-side frame phase timings, consumed by
    /// `take_last_cpu_phase_times()` so callers can record them once.
    last_cpu_phase_times: Option<RendererCpuPhaseTimes>,
    /// Off-surface diagnostic target (hash-thing-dlse.2.2 step 3). When
    /// `Some`, `render()` bypasses `surface.get_current_texture()` and
    /// `surface_texture.present()` entirely, rendering into this throwaway
    /// texture instead. Lets us measure whether the ~25 ms
    /// `surface_acquire_cpu` stall is swapchain/drawable pacing (collapses
    /// when the surface is out of the loop) or something else in the
    /// submit/encode path (persists). Env-var gated at startup; zero-cost
    /// in normal runs.
    off_surface_target: Option<wgpu::Texture>,
    start_time: Instant,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, volume_size: u32) -> Self {
        let size = window.inner_size();

        // hash-thing-dlse.2.2 step 1: log windowing/display context at init so
        // the 25ms surface_acquire_cpu stall on M2 can be correlated with
        // monitor refresh rate, scale factor, and which output the window is on.
        let scale_factor = window.scale_factor();
        match window.current_monitor() {
            Some(m) => log::info!(
                "monitor: name={:?} size={:?} refresh_mHz={:?} scale_factor={} window_scale_factor={}",
                m.name(),
                m.size(),
                m.refresh_rate_millihertz(),
                m.scale_factor(),
                scale_factor,
            ),
            None => log::info!(
                "monitor: none reported by winit (scale_factor={scale_factor})"
            ),
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            backend_options: Default::default(),
            display: Default::default(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create wgpu surface from window");

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
        // hash-thing-dlse.2.3: TIMESTAMP_QUERY_INSIDE_ENCODERS lets us use
        // `encoder.write_timestamp` outside a render/compute pass. Metal
        // appears to attribute in-pass barrier/sync waits to the pass's
        // begin/end timestamps, which makes `ComputePassTimestampWrites`
        // include cross-frame sync waits in what looks like "compute time".
        // The headless bench already uses in-encoder writes (bench_gpu_raycast.rs)
        // and reports ~0.2ms for the same dispatch that the windowed app
        // measures at ~30ms — this flag switches the app to the same
        // pattern so we're comparing like with like.
        let in_encoder_timestamps_supported = timestamp_supported
            && adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        } else {
            log::info!(
                "GPU adapter lacks TIMESTAMP_QUERY — perf will report CPU submit overhead only"
            );
        }
        if in_encoder_timestamps_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
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

        // Default render_scale = 0.5: render at half physical resolution
        // for 4x fewer pixels. Trade sharpness for framerate.
        let render_scale: f32 = 0.5;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: ((size.width as f32 * render_scale) as u32).max(1),
            height: ((size.height as f32 * render_scale) as u32).max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        log::info!(
            "surface_caps: present_modes={:?} alpha_modes={:?} formats={:?}",
            surface_caps.present_modes,
            surface_caps.alpha_modes,
            surface_caps.formats,
        );
        log::info!(
            "surface_config: present_mode={:?} alpha_mode={:?} format={:?} max_frame_latency={} render_scale={} size={}x{} (physical={}x{})",
            config.present_mode, config.alpha_mode, config.format,
            config.desired_maximum_frame_latency, render_scale,
            config.width, config.height, size.width, size.height,
        );
        surface.configure(&device, &config);

        let uniforms = Uniforms {
            camera_pos: [0.0; 4],
            camera_dir: [0.0; 4],
            camera_up: [0.0, 1.0, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0, 0.0],
            params: [volume_size as f32, 1.0, 1.0, 0.0],
            debug: [0.0; 4],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Material palette buffer (shared by both pipelines) ===
        // Initialized with a minimal fallback palette; the real palette is
        // uploaded by main.rs via upload_palette() after creation.
        let initial_palette: [[f32; 4]; 1] = [[0.6, 0.7, 0.8, 1.0]];
        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palette"),
            contents: bytemuck::cast_slice(&initial_palette),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Palette binding entry shared by both pipeline layouts (binding 2).
        // VERTEX | FRAGMENT: SVDAG reads palette in fragment only, but the
        // particle shader reads it in the vertex stage (vs_main looks up
        // material color to pass to the fragment stage).
        let palette_bgl_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // === SVDAG compute raycast pipeline (hash-thing-5bb.6.1) ===

        // Storage texture for compute output. Dimensions match the scaled
        // surface config (already half-res from render_scale).
        let raycast_tex_format = wgpu::TextureFormat::Rgba16Float;
        let (raycast_texture, raycast_texture_view) = Self::create_raycast_texture_static(
            &device,
            config.width,
            config.height,
            raycast_tex_format,
        );

        let svdag_compute_bind_group_layout =
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
                            format: raycast_tex_format,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let svdag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("svdag raycast compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("svdag_raycast.wgsl").into()),
        });

        let svdag_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("svdag_compute_pl"),
                bind_group_layouts: &[Some(&svdag_compute_bind_group_layout)],
                immediate_size: 0,
            });

        let svdag_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("svdag_compute"),
                layout: Some(&svdag_compute_pipeline_layout),
                module: &svdag_shader,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // === Blit pipeline: samples compute output → swapchain (5bb.6.1) ===

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            // The raycast target is often rendered at half-res and then
            // upscaled to the swapchain. Nearest keeps the voxel look crisp
            // instead of bilinear-smearing edges across the upscale.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("svdag_blit.wgsl").into()),
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

        // === Particle billboard pipeline (hash-thing-5bb.9) ===

        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    palette_bgl_entry,
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle billboard shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle.wgsl").into()),
        });

        let particle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle_pl"),
                bind_group_layouts: &[Some(&particle_bind_group_layout)],
                immediate_size: 0,
            });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_rp"),
            layout: Some(&particle_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // HUD overlay (hash-thing-5bb.10)
        let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("hud.wgsl").into()),
        });

        let hud_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud_uniforms"),
            size: 32, // 2 × vec4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hud_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hud_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let hud_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud_bg"),
            layout: &hud_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: hud_uniform_buffer.as_entire_binding(),
            }],
        });

        let hud_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud_pl"),
            bind_group_layouts: &[Some(&hud_bind_group_layout)],
            immediate_size: 0,
        });

        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud_rp"),
            layout: Some(&hud_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &hud_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // Hotbar overlay (hash-thing-e7k.6)
        let hotbar_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hotbar shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("hotbar.wgsl").into()),
        });

        let hotbar_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hotbar_uniforms"),
            size: 16, // 1 × vec4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hotbar_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hotbar_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let hotbar_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hotbar_bg"),
            layout: &hotbar_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hotbar_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: palette_buffer.as_entire_binding(),
                },
            ],
        });

        let hotbar_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hotbar_pl"),
                bind_group_layouts: &[Some(&hotbar_bind_group_layout)],
                immediate_size: 0,
            });

        let hotbar_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hotbar_rp"),
            layout: Some(&hotbar_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &hotbar_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hotbar_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // Legend overlay (hash-thing-m1f.7.2)
        let legend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("legend shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("legend.wgsl").into()),
        });

        let legend_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("legend_uniforms"),
            size: 16, // vec4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let legend_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("legend_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let legend_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("legend_bgl"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let legend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("legend_pl"),
                bind_group_layouts: &[Some(&legend_bind_group_layout)],
                immediate_size: 0,
            });

        let legend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("legend_rp"),
            layout: Some(&legend_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &legend_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &legend_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let adapter_info = adapter.get_info();
        log::info!(
            "GPU adapter: name={:?} backend={:?} driver={:?} device_type={:?}",
            adapter_info.name,
            adapter_info.backend,
            adapter_info.driver_info,
            adapter_info.device_type,
        );
        let (gpu_timing, gpu_timing_render_pass) = if timestamp_supported {
            let period_ns = queue.get_timestamp_period();
            log::info!(
                "GPU timestamp_period={period_ns}ns in_encoder={in_encoder_timestamps_supported}"
            );
            // dlse.2.4: parallel brackets around compute pass and render pass.
            (
                Some(GpuTiming::new(
                    &device,
                    period_ns,
                    in_encoder_timestamps_supported,
                    "gpu_timing_compute",
                )),
                Some(GpuTiming::new(
                    &device,
                    period_ns,
                    in_encoder_timestamps_supported,
                    "gpu_timing_render_pass",
                )),
            )
        } else {
            (None, None)
        };

        Self {
            surface,
            device,
            queue,
            config,
            svdag_compute_pipeline,
            svdag_compute_bind_group_layout,
            svdag_compute_bind_group: None,
            svdag_buffer: None,
            svdag_buffer_cap: 0,
            svdag_uploaded_len: 0,
            raycast_texture,
            raycast_texture_view,
            blit_pipeline,
            blit_bind_group_layout,
            blit_bind_group,
            blit_sampler,
            particle_pipeline,
            particle_bind_group_layout,
            particle_bind_group: None,
            particle_buffer: None,
            particle_buffer_cap: 0,
            particle_count: 0,
            hud_pipeline,
            hud_bind_group,
            hud_uniform_buffer,
            hud_material_color: [1.0, 1.0, 1.0, 1.0],
            hud_visible: false,
            hotbar_pipeline,
            hotbar_bind_group,
            hotbar_uniform_buffer,
            hotbar_selected_slot: 0,
            hotbar_visible: false,
            legend_pipeline,
            legend_bind_group_layout,
            legend_bind_group: None,
            legend_uniform_buffer,
            legend_texture: None,
            legend_texture_view: None,
            legend_sampler,
            legend_tex_w: 0,
            legend_tex_h: 0,
            legend_visible: false,
            palette_buffer,
            uniform_buffer,
            volume_size,
            camera_yaw: std::f32::consts::FRAC_PI_4,
            camera_pitch: 0.4,
            camera_dist: 2.0,
            camera_target: [0.5, 0.5, 0.5],
            debug_mode: 0,
            lod_bias: 1.0,
            render_scale: 0.5,
            gpu_timing,
            gpu_timing_render_pass,
            last_gpu_frame_time: None,
            last_render_pass_gpu_frame_time: None,
            last_cpu_phase_times: None,
            off_surface_target: None,
            start_time: Instant::now(),
        }
    }

    /// Enable off-surface render mode for the dlse.2.2 step-3 diagnostic.
    /// Allocates a throwaway render target matching the current surface
    /// config; subsequent `render()` calls skip `get_current_texture()` +
    /// `present()` and draw into this texture instead.
    ///
    /// Known quirk: `render_gpu` may report `(no samples)` while off-surface
    /// is active. Off-surface frames complete in ~1 ms CPU-side, so the
    /// next frame's `GpuTiming::poll` fires before the prior frame's
    /// `map_async` readback callback lands. The diagnostic's primary signal
    /// (`surface_acquire_cpu`, `render_cpu`) is CPU-side and unaffected.
    pub fn enable_off_surface(&mut self) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("off_surface_target"),
            size: wgpu::Extent3d {
                width: self.config.width.max(1),
                height: self.config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        log::info!(
            "off-surface diagnostic enabled: {}x{} {:?} (dlse.2.2 step 3)",
            self.config.width,
            self.config.height,
            self.config.format,
        );
        self.off_surface_target = Some(tex);
    }

    /// Create the storage texture used as compute raycast output (5bb.6.1).
    /// Separate static method so it can be called from both `new()` and `resize()`.
    fn create_raycast_texture_static(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raycast_output"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    /// Recreate the raycast storage texture and dependent bind groups after resize.
    fn recreate_raycast_texture(&mut self) {
        let lifecycle = RendererLifecycleSnapshot::from_renderer(self);
        let (tex, view) = Self::create_raycast_texture_static(
            &self.device,
            self.config.width,
            self.config.height,
            wgpu::TextureFormat::Rgba16Float,
        );
        self.raycast_texture = tex;
        self.raycast_texture_view = view;

        // Rebuild blit bind group (references the texture view).
        self.blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.raycast_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        // Rebuild compute bind group if SVDAG buffer exists (references texture view).
        if lifecycle.rebuild_compute_bind_group_on_texture_recreate() {
            let Some(svdag_buf) = &self.svdag_buffer else {
                return;
            };
            self.svdag_compute_bind_group =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("svdag_compute_bg"),
                    layout: &self.svdag_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: svdag_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.palette_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &self.raycast_texture_view,
                            ),
                        },
                    ],
                }));
        }

        if lifecycle.rebuild_particle_bind_group_on_texture_recreate() {
            self.rebuild_particle_bind_group();
        }
    }

    fn rebuild_particle_bind_group(&mut self) {
        let lifecycle = RendererLifecycleSnapshot::from_renderer(self);
        let Some(buf) = &self.particle_buffer else {
            self.particle_bind_group = None;
            return;
        };
        if !lifecycle.particle_bind_group_should_exist() {
            self.particle_bind_group = None;
            return;
        }

        self.particle_bind_group =
            Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_bg"),
                layout: &self.particle_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.palette_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.raycast_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            }));
    }

    /// Consume and return the most recently resolved GPU
    /// **compute-pass** duration (SVDAG raycast dispatch), or `None` if
    /// no new sample has been captured since the last call. Call once
    /// per frame after `render()` returns; the value is intended to be
    /// fed into `Perf` as the `render_gpu` metric (see
    /// `hash-thing-6x3`).
    ///
    /// Name is a historical accident — `render_gpu` has always been
    /// compute-only because the blit+overlay render pass was assumed
    /// ~free until dlse.2.2 proved otherwise. The render-pass bracket
    /// lives in `take_last_render_pass_gpu_frame_time()` (dlse.2.4);
    /// `render_gpu` is kept as-is to avoid forking metric history
    /// across the perf paper.
    ///
    /// Adapters without `Features::TIMESTAMP_QUERY` always return
    /// `None` — in that case the only render metric is the CPU-submit
    /// `render_cpu` from `main.rs`.
    pub fn take_last_gpu_frame_time(&mut self) -> Option<Duration> {
        self.last_gpu_frame_time.take()
    }

    /// Consume and return the most recently resolved GPU
    /// **render-pass** duration (blit + particles + HUD + hotbar +
    /// legend), or `None` if no new sample has been captured since the
    /// last call. Companion to `take_last_gpu_frame_time()` — together
    /// they bracket the two GPU-expensive chunks of the frame encoder
    /// (dlse.2.4).
    ///
    /// Returns `None` on adapters without `Features::TIMESTAMP_QUERY`
    /// and when `HASH_THING_DISABLE_TIMESTAMP_RESOLVE=1` is set.
    pub fn take_last_render_pass_gpu_frame_time(&mut self) -> Option<Duration> {
        self.last_render_pass_gpu_frame_time.take()
    }

    pub fn take_last_cpu_phase_times(&mut self) -> Option<RendererCpuPhaseTimes> {
        self.last_cpu_phase_times.take()
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
        // Track the DAG's root level so the shader step budget stays in sync
        // (hash-thing-2w5). When the world grows (hash-thing-m1f.4), the
        // root_level increases and volume_size must follow.
        self.volume_size = 1u32 << dag.root_level;
        let bytes: &[u8] = bytemuck::cast_slice(&dag.nodes);
        let needed = bytes.len() as u64;

        // Grow the GPU buffer if needed. Resets the upload watermark — the
        // fresh buffer has nothing on it, so the next write must cover all.
        let mut require_full = false;
        if self.svdag_buffer.is_none() || needed > self.svdag_buffer_cap {
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
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("svdag_compute_bg"),
                layout: &self.svdag_compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
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
            });
            self.svdag_buffer = Some(buffer);
            self.svdag_buffer_cap = cap;
            self.svdag_uploaded_len = 0;
            self.svdag_compute_bind_group = Some(bg);
            require_full = true;
        }

        let Some(buf) = &self.svdag_buffer else {
            return;
        };

        if require_full || self.svdag_uploaded_len > dag.nodes.len() {
            // Fresh buffer or dag shrank (builder was recreated). Full upload.
            self.queue.write_buffer(buf, 0, bytes);
            self.svdag_uploaded_len = dag.nodes.len();
            return;
        }

        // Slot 0 is the root-offset header — changes almost every frame.
        self.queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&dag.nodes[0..1]));

        // Tail: append-only growth since the last upload.
        if self.svdag_uploaded_len < dag.nodes.len() {
            let tail_byte_offset = self.svdag_uploaded_len as u64 * 4;
            let tail_bytes: &[u8] = bytemuck::cast_slice(&dag.nodes[self.svdag_uploaded_len..]);
            self.queue.write_buffer(buf, tail_byte_offset, tail_bytes);
            self.svdag_uploaded_len = dag.nodes.len();
        }
    }

    /// Upload the material color palette to the GPU. Called once after init
    /// and whenever the palette changes (e.g. terrain reset with different
    /// materials). Recreates the palette buffer and both bind groups since
    /// wgpu bind groups are immutable (hash-thing-5bb.7 / hash-thing-ll6).
    pub fn upload_palette(&mut self, palette: &[[f32; 4]]) {
        let lifecycle = RendererLifecycleSnapshot::from_renderer(self);
        self.palette_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("palette"),
                contents: bytemuck::cast_slice(palette),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Rebuild particle bind group if it exists (it references palette).
        if lifecycle.rebuild_particle_bind_group_on_palette_upload() {
            self.rebuild_particle_bind_group();
        }

        // Rebuild hotbar bind group (it references palette via binding 1).
        self.hotbar_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hotbar_bg"),
            layout: &self.hotbar_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.hotbar_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.palette_buffer.as_entire_binding(),
                },
            ],
        });

        // Rebuild SVDAG compute bind group if it exists (it also references palette).
        if lifecycle.rebuild_compute_bind_group_on_palette_upload() {
            let Some(svdag_buf) = &self.svdag_buffer else {
                return;
            };
            self.svdag_compute_bind_group =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("svdag_compute_bg"),
                    layout: &self.svdag_compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: svdag_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.palette_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                &self.raycast_texture_view,
                            ),
                        },
                    ],
                }));
        }
    }

    /// Upload particle positions/materials for billboard rendering.
    /// Each particle is a `[f32; 4]`: `[x, y, z, bitcast(material_u32)]`.
    /// Called each frame from the app loop after entity update.
    pub fn upload_particles(&mut self, data: &[[f32; 4]]) {
        self.particle_count = data.len() as u32;
        if data.is_empty() {
            self.particle_bind_group = None;
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let needed = bytes.len() as u64;

        if needed > self.particle_buffer_cap {
            // Grow with headroom.
            let cap = needed.next_power_of_two().max(256);
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle_buf"),
                size: cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.particle_buffer = Some(buf);
            self.particle_buffer_cap = cap;
        }

        if let Some(buf) = &self.particle_buffer {
            self.queue.write_buffer(buf, 0, bytes);
            self.rebuild_particle_bind_group();
        }
    }

    /// Upload legend text rendered as a bitmap texture. Call when the
    /// legend content changes (e.g., camera mode switch). Pass empty
    /// slice to clear.
    pub fn set_legend_text(&mut self, lines: &[&str]) {
        if lines.is_empty() {
            self.legend_bind_group = None;
            self.legend_texture = None;
            self.legend_texture_view = None;
            return;
        }

        let scale = 2u32;
        let (pixels, w, h) = super::font::render_text_rgba(lines, scale);
        self.legend_tex_w = w;
        self.legend_tex_h = h;

        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("legend_tex"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            size,
        );
        let view = texture.create_view(&Default::default());

        self.legend_bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("legend_bg"),
            layout: &self.legend_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.legend_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.legend_uniform_buffer.as_entire_binding(),
                },
            ],
        }));

        self.legend_texture_view = Some(view);
        self.legend_texture = Some(texture);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let s = self.render_scale;
            self.config.width = ((width as f32 * s) as u32).max(1);
            self.config.height = ((height as f32 * s) as u32).max(1);
            log::info!(
                "resize: physical={}x{} render_scale={} config={}x{} (dlse.2.2 pixel-workload log)",
                width,
                height,
                s,
                self.config.width,
                self.config.height,
            );
            self.surface.configure(&self.device, &self.config);
            // Recreate storage texture to match new dimensions (5bb.6.1).
            self.recreate_raycast_texture();
            if self.off_surface_target.is_some() {
                self.enable_off_surface();
            }
        }
    }

    pub fn render(&mut self) -> FrameOutcome {
        use wgpu::CurrentSurfaceTexture;

        // hash-thing-6x3: pump `map_async` callbacks and consume any
        // GPU-timestamp readbacks that landed since the last frame.
        // dlse.2.4 adds a second bracket; `device.poll(Poll)` runs
        // exactly once here so cost stays flat regardless of how many
        // `GpuTiming` instances exist. Must happen before we
        // potentially skip the frame on surface failures — otherwise a
        // long run of Occluded frames would accumulate un-polled
        // callbacks.
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(gt) = &self.gpu_timing {
            if let Some(d) = gt.take_resolved() {
                self.last_gpu_frame_time = Some(d);
            }
        }
        if let Some(gt) = &self.gpu_timing_render_pass {
            if let Some(d) = gt.take_resolved() {
                self.last_render_pass_gpu_frame_time = Some(d);
            }
        }

        // No catch-all: `CurrentSurfaceTexture` is not `#[non_exhaustive]`, so
        // any future wgpu variant becomes a compile error pointing here,
        // instead of getting silently swallowed (hash-thing-8jp I1a).
        //
        // Off-surface diagnostic (dlse.2.2 step 3): when
        // `self.off_surface_target` is `Some`, bypass the surface entirely
        // and draw into the throwaway texture. `surface_acquire` records
        // ~0 in that mode.
        let acquire_start = Instant::now();
        let (surface_texture, view) = if let Some(off) = self.off_surface_target.as_ref() {
            let view = off.create_view(&wgpu::TextureViewDescriptor::default());
            (None, view)
        } else {
            let st = match self.surface.get_current_texture() {
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
            let view = st
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            (Some(st), view)
        };
        let surface_acquire = acquire_start.elapsed();

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

        let elapsed_secs = self.start_time.elapsed().as_secs_f32();
        let uniforms = Uniforms {
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], elapsed_secs],
            camera_dir: [cam_dir[0], cam_dir[1], cam_dir[2], 0.0],
            camera_up: [up[0], up[1], up[2], 0.0],
            camera_right: [right[0], right[1], right[2], 0.0],
            params: [
                self.volume_size as f32,
                aspect,
                fov_tan,
                self.config.height as f32,
            ],
            debug: [
                self.debug_mode as f32,
                self.lod_bias,
                self.config.width as f32,
                self.config.height as f32,
            ],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        // hash-thing-5bb.6.1: compute dispatch for SVDAG raycast, then
        // blit + overlays in a single render pass.
        //
        // GPU timing wraps the compute dispatch (the expensive part). When
        // the adapter supports TIMESTAMP_QUERY_INSIDE_ENCODERS, we write
        // timestamps on the encoder instead of via ComputePassTimestampWrites
        // so the measurement excludes cross-pass barrier/sync waits
        // (hash-thing-dlse.2.3).
        // hash-thing-dlse.2.2.1 diagnostic: `HASH_THING_DISABLE_TIMESTAMP_RESOLVE=1`
        // skips both the in-encoder timestamp writes and the resolve/map
        // path so we can isolate whether off-surface's ~26 ms submit_cpu
        // stall is caused by the timestamp fence (hypothesis A) or by
        // Metal submit backpressure (hypothesis B). dlse.2.4 extends
        // the gate to cover the new render-pass bracket.
        let resolve_disabled = std::env::var("HASH_THING_DISABLE_TIMESTAMP_RESOLVE")
            .ok()
            .as_deref()
            == Some("1");

        // --- Compute-pass timing setup ---
        let compute_timestamp_writes = if resolve_disabled {
            None
        } else {
            self.gpu_timing
                .as_ref()
                .and_then(|gt| gt.compute_pass_writes())
        };
        let compute_in_encoder_capturing = !resolve_disabled
            && self.svdag_compute_bind_group.is_some()
            && self
                .gpu_timing
                .as_ref()
                .is_some_and(|gt| gt.uses_in_encoder() && gt.is_idle());
        let captured_compute_this_frame =
            !resolve_disabled && (compute_timestamp_writes.is_some() || compute_in_encoder_capturing);

        // --- Render-pass timing setup (dlse.2.4). Independent state
        // machine — may capture on a frame the compute pass doesn't, or
        // vice versa, whenever one of the two readbacks is still in
        // flight. ---
        let render_pass_timestamp_writes = if resolve_disabled {
            None
        } else {
            self.gpu_timing_render_pass
                .as_ref()
                .and_then(|gt| gt.pass_writes())
        };
        let render_pass_in_encoder_capturing = !resolve_disabled
            && self
                .gpu_timing_render_pass
                .as_ref()
                .is_some_and(|gt| gt.uses_in_encoder() && gt.is_idle());
        let captured_render_pass_this_frame = !resolve_disabled
            && (render_pass_timestamp_writes.is_some() || render_pass_in_encoder_capturing);

        // --- Compute pass: SVDAG raycast → storage texture ---
        if let Some(bg) = &self.svdag_compute_bind_group {
            if compute_in_encoder_capturing {
                if let Some(gt) = &self.gpu_timing {
                    gt.write_timestamp_begin(&mut encoder);
                }
            }
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("svdag raycast compute"),
                    timestamp_writes: compute_timestamp_writes,
                });
                compute_pass.set_pipeline(&self.svdag_compute_pipeline);
                compute_pass.set_bind_group(0, bg, &[]);
                let wg_x = self.config.width.div_ceil(8);
                let wg_y = self.config.height.div_ceil(8);
                compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            if compute_in_encoder_capturing {
                if let Some(gt) = &self.gpu_timing {
                    gt.write_timestamp_end(&mut encoder);
                }
            }
        }

        // --- Render pass: blit + particles + HUD + legend (single pass) ---
        // dlse.2.4: bracket the whole pass with timestamps parallel to
        // the compute pass above. When the adapter supports
        // TIMESTAMP_QUERY_INSIDE_ENCODERS we bracket with
        // encoder.write_timestamp on either side of begin_render_pass;
        // otherwise we thread RenderPassTimestampWrites into the
        // descriptor (which `pass_writes()` returns on that branch).
        if render_pass_in_encoder_capturing {
            if let Some(gt) = &self.gpu_timing_render_pass {
                gt.write_timestamp_begin(&mut encoder);
            }
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit + overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Clear to sky color as fallback if compute hasn't run yet.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.53,
                            g: 0.72,
                            b: 0.92,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: render_pass_timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Blit compute output to swapchain (fullscreen triangle).
            if self.svdag_compute_bind_group.is_some() {
                render_pass.set_pipeline(&self.blit_pipeline);
                render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
                render_pass.draw(0..3, 0..1); // fullscreen triangle = 3 verts
            }

            // Particle overlay — drawn after voxels with alpha blending.
            if self.particle_count > 0 {
                if let Some(bg) = &self.particle_bind_group {
                    render_pass.set_pipeline(&self.particle_pipeline);
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.draw(0..6, 0..self.particle_count);
                }
            }

            // HUD overlay — crosshair + material indicator (5bb.10).
            if self.hud_visible {
                let aspect = self.config.width as f32 / self.config.height as f32;
                let hud_data: [f32; 8] = [
                    self.hud_material_color[0],
                    self.hud_material_color[1],
                    self.hud_material_color[2],
                    self.hud_material_color[3],
                    aspect,
                    1.0,
                    0.0,
                    0.0,
                ];
                self.queue.write_buffer(
                    &self.hud_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&hud_data),
                );
                render_pass.set_pipeline(&self.hud_pipeline);
                render_pass.set_bind_group(0, &self.hud_bind_group, &[]);
                render_pass.draw(0..30, 0..1);
            }

            // Hotbar overlay (e7k.6).
            if self.hotbar_visible {
                let aspect = self.config.width as f32 / self.config.height as f32;
                let hotbar_data: [f32; 4] = [aspect, self.hotbar_selected_slot as f32, 0.0, 0.0];
                self.queue.write_buffer(
                    &self.hotbar_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&hotbar_data),
                );
                render_pass.set_pipeline(&self.hotbar_pipeline);
                render_pass.set_bind_group(0, &self.hotbar_bind_group, &[]);
                render_pass.draw(0..108, 0..1); // 18 slots × 6 verts
            }

            // Legend overlay — keybindings text (m1f.7.2).
            if self.legend_visible {
                if let Some(bg) = &self.legend_bind_group {
                    let aspect = self.config.width as f32 / self.config.height as f32;
                    let tex_aspect = if self.legend_tex_h > 0 {
                        self.legend_tex_w as f32 / self.legend_tex_h as f32
                    } else {
                        1.0
                    };
                    let quad_h = 0.62;
                    let quad_w = quad_h * tex_aspect / aspect;
                    let margin = 0.035;
                    let quad_left = -1.0 + margin;
                    let quad_bottom = -1.0 + margin;
                    let legend_params: [f32; 4] = [quad_left, quad_bottom, quad_w, quad_h];
                    self.queue.write_buffer(
                        &self.legend_uniform_buffer,
                        0,
                        bytemuck::cast_slice(&legend_params),
                    );
                    render_pass.set_pipeline(&self.legend_pipeline);
                    render_pass.set_bind_group(0, bg, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }
        if render_pass_in_encoder_capturing {
            if let Some(gt) = &self.gpu_timing_render_pass {
                gt.write_timestamp_end(&mut encoder);
            }
        }

        if captured_compute_this_frame {
            if let Some(gt) = &self.gpu_timing {
                gt.encode_resolve(&mut encoder);
            }
        }
        if captured_render_pass_this_frame {
            if let Some(gt) = &self.gpu_timing_render_pass {
                gt.encode_resolve(&mut encoder);
            }
        }

        let submit_start = Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        let submit = submit_start.elapsed();
        let present_start = Instant::now();
        if let Some(st) = surface_texture {
            st.present();
        }
        let present = present_start.elapsed();
        self.last_cpu_phase_times = Some(RendererCpuPhaseTimes {
            surface_acquire,
            submit,
            present,
        });

        if captured_compute_this_frame {
            if let Some(gt) = &self.gpu_timing {
                gt.request_readback();
            }
        }
        if captured_render_pass_this_frame {
            if let Some(gt) = &self.gpu_timing_render_pass {
                gt.request_readback();
            }
        }

        FrameOutcome::Rendered
    }
}

#[cfg(test)]
mod tests {
    use super::{ticks_to_duration, FrameOutcome, RendererLifecycleSnapshot};
    use std::time::Duration;

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
    fn lifecycle_snapshot_only_rebuilds_compute_bind_group_when_svdag_buffer_exists() {
        let without_svdag = RendererLifecycleSnapshot {
            has_svdag_buffer: false,
            has_particle_buffer: false,
            particle_count: 0,
        };
        let with_svdag = RendererLifecycleSnapshot {
            has_svdag_buffer: true,
            ..without_svdag
        };

        assert!(!without_svdag.rebuild_compute_bind_group_on_texture_recreate());
        assert!(!without_svdag.rebuild_compute_bind_group_on_palette_upload());
        assert!(with_svdag.rebuild_compute_bind_group_on_texture_recreate());
        assert!(with_svdag.rebuild_compute_bind_group_on_palette_upload());
    }

    #[test]
    fn lifecycle_snapshot_distinguishes_resize_vs_palette_particle_rebuilds() {
        let buffered_but_empty = RendererLifecycleSnapshot {
            has_svdag_buffer: false,
            has_particle_buffer: true,
            particle_count: 0,
        };
        let active_particles = RendererLifecycleSnapshot {
            particle_count: 3,
            ..buffered_but_empty
        };

        assert!(buffered_but_empty.rebuild_particle_bind_group_on_palette_upload());
        assert!(!buffered_but_empty.rebuild_particle_bind_group_on_texture_recreate());

        assert!(active_particles.rebuild_particle_bind_group_on_palette_upload());
        assert!(active_particles.rebuild_particle_bind_group_on_texture_recreate());
    }

    #[test]
    fn lifecycle_snapshot_particle_bind_group_requires_buffer_and_live_particles() {
        let cases = [
            (
                RendererLifecycleSnapshot {
                    has_svdag_buffer: false,
                    has_particle_buffer: false,
                    particle_count: 0,
                },
                false,
            ),
            (
                RendererLifecycleSnapshot {
                    has_svdag_buffer: false,
                    has_particle_buffer: true,
                    particle_count: 0,
                },
                false,
            ),
            (
                RendererLifecycleSnapshot {
                    has_svdag_buffer: false,
                    has_particle_buffer: false,
                    particle_count: 4,
                },
                false,
            ),
            (
                RendererLifecycleSnapshot {
                    has_svdag_buffer: false,
                    has_particle_buffer: true,
                    particle_count: 4,
                },
                true,
            ),
        ];

        for (snapshot, expected) in cases {
            assert_eq!(
                snapshot.particle_bind_group_should_exist(),
                expected,
                "{snapshot:?}"
            );
        }
    }
}
