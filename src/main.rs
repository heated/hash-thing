use hash_thing::perf;
use hash_thing::render;
use hash_thing::sim;
use hash_thing::terrain;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

const VOLUME_SIZE: u32 = 64;

/// Wall-clock cadence for the consolidated perf log line. Decoupled from
/// `world.generation` so the log keeps ticking even when the sim is paused
/// or stepping slowly — see hash-thing-q63.
const LOG_INTERVAL_SECS: f64 = 2.0;

/// One-line gen summary. Centralised so the `App::new` startup path and the
/// `R`-key reset path emit identical formatting. hash-thing-3fq.5 added the
/// classify_calls / nodes_delta / noise_fraction fields; the rest is carried
/// over from the pre-3fq.5 log line.
#[allow(clippy::too_many_arguments)]
fn log_gen_stats(
    label: &str,
    side: usize,
    population: u64,
    nodes: usize,
    nodes_delta: usize,
    stats: &terrain::GenStats,
    elapsed: std::time::Duration,
    noise_ns_per_sample: f64,
) {
    let gen_us = elapsed.as_micros() as f64;
    let gen_ms = gen_us / 1_000.0;
    let sample_us = stats.leaves as f64 * noise_ns_per_sample / 1_000.0;
    let sample_pct = if gen_us > 0.0 {
        (sample_us / gen_us * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let gen_region_ms = stats.gen_region_us as f64 / 1_000.0;
    let cave_ms = stats.cave_us as f64 / 1_000.0;
    let dungeon_ms = stats.dungeon_us as f64 / 1_000.0;
    log::info!(
        "{label}: {side}^3 pop={pop} nodes={nodes} (+{delta}) \
         gen_calls={calls} samples={samples} classifies={classifies} collapses={collapses} \
         gen_time={gen_ms:.2}ms (region={gen_region_ms:.2}ms cave={cave_ms:.2}ms dungeon={dungeon_ms:.2}ms) \
         nodes_after_gen={nag} nodes_after_caves={nac} nodes_after_dungeons={nad} | \
         noise~{ns:.0}ns/sample → ~{sample_pct:.0}% of gen",
        pop = population,
        delta = nodes_delta,
        calls = stats.calls_total,
        samples = stats.leaves,
        classifies = stats.classify_calls,
        collapses = stats.total_collapses(),
        nag = stats.nodes_after_gen,
        nac = stats.nodes_after_caves,
        nad = stats.nodes_after_dungeons,
        ns = noise_ns_per_sample,
    );
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<render::Renderer>,
    world: sim::World,
    gol_smoke_scene: bool,
    /// Persistent serialized DAG. Kept across frames so that its content-
    /// addressed cache lets us upload only new nodes each step (5bb.5).
    svdag: render::Svdag,
    paused: bool,
    step_timer: std::time::Instant,
    /// Wall-clock checkpoint for the next perf summary line. Reset each
    /// time the line fires so cadence stays ~LOG_INTERVAL_SECS regardless
    /// of sim/step rate.
    log_timer: std::time::Instant,
    // Mouse interaction state
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    perf: perf::Perf,
    /// Memory-watchdog metric family — node-count + step-cache ratcheting
    /// peaks and a byte estimate. Orthogonal to `perf` (latency). Sampled
    /// on the wall-clock log path.
    mem_stats: perf::MemStats,
    /// Cached SVDAG stats captured during `upload_volume` so the wall-clock
    /// log line can read them without rebuilding the SVDAG. Updated every
    /// time the DAG is rebuilt for render. Tuple: (node_count, byte_size,
    /// root_level).
    last_svdag_stats: (usize, usize, u32),
    /// Window visibility gate (hash-thing-8jp). When `true`, the redraw
    /// treadmill is paused — `RedrawRequested` becomes a no-op and
    /// `request_redraw` is not called.
    occluded: bool,
    /// One-time microbench of `HeightmapField::sample()` in ns/call.
    /// Used to estimate noise fraction of each gen pass (hash-thing-3fq.5).
    /// Refreshed on terrain reset so a wildly different param set reprobes.
    noise_ns_per_sample: f64,
}

impl App {
    fn new() -> Self {
        let mut world = sim::World::new(VOLUME_SIZE.trailing_zeros());

        // Seed with terrain instead of a GoL sphere.
        let params = terrain::TerrainParams::default();
        let nodes_before = world.store.stats().0;
        let start = std::time::Instant::now();
        let stats = world.seed_terrain(&params);
        let elapsed = start.elapsed();

        // Noise-bottleneck probe. One-time microbench of `sample()`
        // called outside the timed gen region so it does not pollute
        // the gen measurement. See hash-thing-3fq.5.
        let noise_ns_per_sample = terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
        let material_palette_len = world.materials.color_palette_rgba().len();

        let (nodes_after, _) = world.store.stats();
        let nodes_delta = nodes_after.saturating_sub(nodes_before);
        log_gen_stats(
            "Initial terrain",
            world.side(),
            world.population(),
            nodes_after,
            nodes_delta,
            &stats,
            elapsed,
            noise_ns_per_sample,
        );
        log::debug!("Material registry palette slots={material_palette_len}");

        // Start paused so the user opts into stepping explicitly. Terrain
        // defaults are mostly static, but the reactive material rules are
        // now live and should not advance until requested.
        Self {
            window: None,
            renderer: None,
            world,
            gol_smoke_scene: false,
            svdag: render::Svdag::new(),
            paused: true,
            step_timer: std::time::Instant::now(),
            log_timer: std::time::Instant::now(),
            mouse_pressed: false,
            last_mouse: None,
            perf: perf::Perf::new(),
            mem_stats: perf::MemStats::new(),
            last_svdag_stats: (0, 0, 0),
            occluded: false,
            noise_ns_per_sample,
        }
    }

    /// Refresh both GPU uploads (flat3D volume + SVDAG) and cache the
    /// DAG stats for the wall-clock log path.
    ///
    /// Takes explicit field references rather than `&mut self` so
    /// callers can wrap the call in a [`perf::Timer`] via
    /// [`perf::Perf::start`]. A whole-self borrow would conflict with
    /// the timer's borrow on `self.perf`; disjoint field borrows do
    /// not — this is precisely the `hash-thing-yri` fix.
    fn upload_volume(
        renderer: &mut Option<render::Renderer>,
        world: &sim::World,
        svdag: &mut render::Svdag,
        last_svdag_stats: &mut (usize, usize, u32),
    ) {
        if let Some(renderer) = renderer {
            let data = world.flatten();
            renderer.upload_volume(&data);
            // Incremental rebuild: reuses cached offsets for unchanged subtrees.
            svdag.update(&world.store, world.root, world.level);
            // Compact when >50% of the buffer is stale slots (hash-thing-bx7).
            if svdag.stale_ratio() > 0.5 {
                svdag.compact(&world.store, world.root);
            }
            *last_svdag_stats = (svdag.node_count, svdag.byte_size(), svdag.root_level);
            renderer.upload_svdag(svdag);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("hash-thing | 3D Hashlife Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("failed to create main window"),
            );
            self.window = Some(window.clone());

            let mut renderer =
                pollster::block_on(render::Renderer::new(window.clone(), VOLUME_SIZE));
            renderer.upload_palette(&self.world.materials.color_palette_rgba());
            self.renderer = Some(renderer);
            // Initial upload — untimed; we haven't started the render
            // loop yet and there's no perf summary to feed.
            Self::upload_volume(
                &mut self.renderer,
                &self.world,
                &mut self.svdag,
                &mut self.last_svdag_stats,
            );
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }

            // Pause the redraw treadmill while the window is hidden
            // (minimized, behind another window, screen locked). On un-occlude,
            // re-arm the loop with a single `request_redraw`. See
            // hash-thing-8jp.
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key.as_ref() {
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                            self.paused = !self.paused;
                            log::info!("Paused: {}", self.paused);
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        winit::keyboard::Key::Character("s") => {
                            // Single step. Instrumented with the same
                            // `perf.start("step")` Timer as the auto-step
                            // path (hash-thing-5qh + hash-thing-yri).
                            {
                                let _t = self.perf.start("step");
                                self.world.step();
                            }
                            Self::upload_volume(
                                &mut self.renderer,
                                &self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            log::info!(
                                "Gen {}: pop={}",
                                self.world.generation,
                                self.world.population()
                            );
                        }
                        winit::keyboard::Key::Character("c") => {
                            // Re-seed terrain with caves enabled. Stays paused.
                            // hash-thing-3fq.2: drives the cave-CA post-pass
                            // end-to-end so the effect is visible without
                            // editing source.
                            let params = terrain::TerrainParams {
                                caves: Some(terrain::CaveParams::default()),
                                ..Default::default()
                            };
                            let nodes_before = self.world.store.stats().0;
                            let start = std::time::Instant::now();
                            let stats = self.world.seed_terrain(&params);
                            let elapsed = start.elapsed();
                            self.noise_ns_per_sample =
                                terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
                            self.gol_smoke_scene = false;
                            self.paused = true;
                            self.perf.clear();
                            self.mem_stats.reset_peaks();
                            Self::upload_volume(
                                &mut self.renderer,
                                &self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            let (nodes_after, _) = self.world.store.stats();
                            let nodes_delta = nodes_after.saturating_sub(nodes_before);
                            log_gen_stats(
                                "Caves terrain",
                                self.world.side(),
                                self.world.population(),
                                nodes_after,
                                nodes_delta,
                                &stats,
                                elapsed,
                                self.noise_ns_per_sample,
                            );
                        }
                        winit::keyboard::Key::Character("d") => {
                            // Re-seed terrain with caves + dungeons. Stays paused.
                            // hash-thing-3fq.8: drives the dungeon carving
                            // post-pass end-to-end.
                            let params = terrain::TerrainParams {
                                caves: Some(terrain::CaveParams::default()),
                                dungeons: Some(terrain::DungeonParams::default()),
                                ..Default::default()
                            };
                            let nodes_before = self.world.store.stats().0;
                            let start = std::time::Instant::now();
                            let stats = self.world.seed_terrain(&params);
                            let elapsed = start.elapsed();
                            self.noise_ns_per_sample =
                                terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
                            self.legacy_gol_smoke = false;
                            self.paused = true;
                            self.perf.clear();
                            self.mem_stats.reset_peaks();
                            Self::upload_volume(
                                &mut self.renderer,
                                &self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            let (nodes_after, _) = self.world.store.stats();
                            let nodes_delta = nodes_after.saturating_sub(nodes_before);
                            log_gen_stats(
                                "Dungeon terrain",
                                self.world.side(),
                                self.world.population(),
                                nodes_after,
                                nodes_delta,
                                &stats,
                                elapsed,
                                self.noise_ns_per_sample,
                            );
                        }
                        winit::keyboard::Key::Character("r") => {
                            // Re-seed terrain. Stays paused. Clear perf/mem
                            // so post-reset stats aren't poisoned by stale
                            // pre-reset values.
                            let params = terrain::TerrainParams::default();
                            let nodes_before = self.world.store.stats().0;
                            let start = std::time::Instant::now();
                            let stats = self.world.seed_terrain(&params);
                            let elapsed = start.elapsed();
                            // Re-probe in case params drifted. Cheap (~1ms).
                            self.noise_ns_per_sample =
                                terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
                            self.gol_smoke_scene = false;
                            self.paused = true;
                            self.perf.clear();
                            self.mem_stats.reset_peaks();
                            Self::upload_volume(
                                &mut self.renderer,
                                &self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            let (nodes_after, _) = self.world.store.stats();
                            let nodes_delta = nodes_after.saturating_sub(nodes_before);
                            log_gen_stats(
                                "Reset terrain",
                                self.world.side(),
                                self.world.population(),
                                nodes_after,
                                nodes_delta,
                                &stats,
                                elapsed,
                                self.noise_ns_per_sample,
                            );
                        }
                        winit::keyboard::Key::Character("g") => {
                            // Swap to the single retained GoL smoke seed.
                            self.world = sim::World::new(VOLUME_SIZE.trailing_zeros());
                            self.world.materials =
                                terrain::materials::MaterialRegistry::gol_smoke();
                            self.world.seed_center(12, 0.35);
                            self.gol_smoke_scene = true;
                            self.paused = true;
                            self.perf.clear();
                            self.mem_stats.reset_peaks();
                            if let Some(renderer) = &mut self.renderer {
                                renderer.upload_palette(&self.world.materials.color_palette_rgba());
                            }
                            Self::upload_volume(
                                &mut self.renderer,
                                &self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            log::info!("Reset GoL smoke sphere: pop={}", self.world.population());
                        }
                        winit::keyboard::Key::Character("v") => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.mode = match renderer.mode {
                                    render::RenderMode::Flat3D => render::RenderMode::Svdag,
                                    render::RenderMode::Svdag => render::RenderMode::Flat3D,
                                };
                                log::info!("Render mode: {:?}", renderer.mode);
                            }
                        }
                        // hash-thing-hso: on-demand dump of the full perf +
                        // memory summary, independent of the wall-clock log
                        // cadence.
                        winit::keyboard::Key::Character("p") => {
                            let (nodes, cache) = self.world.store.stats();
                            self.mem_stats.update(nodes, cache);
                            let (svdag_nodes, svdag_bytes, svdag_root_level) =
                                self.last_svdag_stats;
                            log::info!(
                                "Gen {} (on demand): pop={} svdag={}/{}KB(L{}) | {} | {}",
                                self.world.generation,
                                self.world.population(),
                                svdag_nodes,
                                svdag_bytes / 1024,
                                svdag_root_level,
                                self.mem_stats.summary(),
                                self.perf.summary(),
                            );
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                    if !self.mouse_pressed {
                        self.last_mouse = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        if let Some(renderer) = &mut self.renderer {
                            renderer.camera_yaw += dx * 0.005;
                            renderer.camera_pitch =
                                (renderer.camera_pitch + dy * 0.005).clamp(-1.4, 1.4);
                        }
                    }
                    self.last_mouse = Some((position.x, position.y));
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                if let Some(renderer) = &mut self.renderer {
                    renderer.camera_dist = (renderer.camera_dist - scroll * 0.1).clamp(0.5, 10.0);
                }
            }

            WindowEvent::RedrawRequested => {
                // If the window is hidden, skip the whole redraw path —
                // stepping the sim + uploading the SVDAG during a 100%-CPU
                // spin on an invisible surface is exactly what 8jp was about.
                // `WindowEvent::Occluded(false)` re-arms the loop.
                if self.occluded {
                    return;
                }

                // Step simulation
                if !self.paused && self.step_timer.elapsed().as_millis() > 200 {
                    // Time the step. `perf.start` returns a Timer that
                    // borrows only self.perf; self.world is disjoint, so
                    // the borrow checker lets the step proceed while the
                    // Timer is alive.
                    {
                        let _t = self.perf.start("step");
                        self.world.step();
                    }
                    // Time upload as one aggregate (flatten + Svdag::build +
                    // upload_volume + upload_svdag). CPU-side submit only —
                    // wgpu queue writes are async. upload_volume takes
                    // explicit field references (not &mut self) precisely
                    // so the Timer can coexist with the call.
                    {
                        let _t = self.perf.start("upload_cpu");
                        Self::upload_volume(
                            &mut self.renderer,
                            &self.world,
                            &mut self.svdag,
                            &mut self.last_svdag_stats,
                        );
                    }

                    self.step_timer = std::time::Instant::now();
                }

                // Wall-clock perf summary (hash-thing-q63). Sits outside the
                // step gate so it fires on its own cadence regardless of
                // sim/step rate — including when paused. Showing the same
                // Gen repeatedly is intentional: it tells the user the app
                // is still alive.
                if self.log_timer.elapsed().as_secs_f64() >= LOG_INTERVAL_SECS {
                    let (nodes, cache) = self.world.store.stats();
                    self.mem_stats.update(nodes, cache);
                    let (svdag_nodes, svdag_bytes, svdag_root_level) = self.last_svdag_stats;
                    log::info!(
                        "Gen {}: pop={} svdag={}/{}KB(L{}) | {} | {}",
                        self.world.generation,
                        self.world.population(),
                        svdag_nodes,
                        svdag_bytes / 1024,
                        svdag_root_level,
                        self.mem_stats.summary(),
                        self.perf.summary(),
                    );
                    self.log_timer = std::time::Instant::now();
                }

                // Time render. Disjoint-field borrows: the Timer holds
                // self.perf, renderer borrows self.renderer — orthogonal.
                // Timer drops at the end of the `if let` arm so the
                // borrow ends before we inspect `outcome` below.
                let outcome = if let Some(renderer) = self.renderer.as_mut() {
                    let _t = self.perf.start("render_cpu");
                    Some(renderer.render())
                } else {
                    None
                };

                // hash-thing-6x3: if the renderer resolved a GPU-side
                // render-pass timing this frame (i.e. the previous
                // frame's `map_async` readback landed), record it into
                // `perf` as `render_gpu`. `take_last_gpu_frame_time`
                // consumes the value so we don't double-record the same
                // sample across frames. Adapters without TIMESTAMP_QUERY
                // always return `None` here — `render_cpu` stays the
                // only render metric on those machines.
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Some(d) = renderer.take_last_gpu_frame_time() {
                        self.perf.record("render_gpu", d);
                    }
                }

                // Belt-and-suspenders: if the surface reports Occluded
                // before winit fires `WindowEvent::Occluded(true)` (some
                // platforms are lazy about that event), latch the flag here
                // so the next RedrawRequested short-circuits at the top of
                // the arm.
                if matches!(outcome, Some(render::FrameOutcome::Occluded)) {
                    self.occluded = true;
                    return;
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("hash-thing: 3D Hashlife Engine");
    log::info!("Controls:");
    log::info!("  Mouse drag: orbit camera");
    log::info!("  Scroll: zoom");
    log::info!("  Space: pause/resume");
    log::info!("  S: single step");
    log::info!("  R: reset terrain (heightmap)");
    log::info!("  C: reset terrain with caves (CA post-pass)");
    log::info!("  D: reset terrain with caves + dungeons");
    log::info!("  G: reset to legacy GoL sphere seed");
    log::info!("  1-4: switch rules (amoeba, crystal, 445, pyroclastic)");
    log::info!("  V: toggle Flat3D / SVDAG rendering");
    log::info!("  P: dump perf + memory summary (on demand)");
    log::info!("  Esc: quit");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
