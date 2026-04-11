mod octree;
mod perf;
mod render;
mod rng;
mod sim;

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

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<render::Renderer>,
    world: sim::World,
    rule: sim::GameOfLife3D,
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
    /// `request_redraw` is not called. Flipped by `WindowEvent::Occluded`
    /// (both edges) and by the renderer's `FrameOutcome::Occluded` as a
    /// belt-and-suspenders signal in case winit doesn't fire the event on
    /// some platform.
    occluded: bool,
}

impl App {
    fn new() -> Self {
        let mut world = sim::World::new(VOLUME_SIZE.trailing_zeros());
        // Seed with a sphere of random cells
        world.seed_center(12, 0.35);

        let (nodes, _) = world.store.stats();
        log::info!(
            "Initial world: {}^3, population={}, nodes={}",
            world.side(),
            world.population(),
            nodes
        );

        Self {
            window: None,
            renderer: None,
            world,
            rule: sim::GameOfLife3D::amoeba(),
            paused: false,
            step_timer: std::time::Instant::now(),
            log_timer: std::time::Instant::now(),
            mouse_pressed: false,
            last_mouse: None,
            perf: perf::Perf::new(),
            mem_stats: perf::MemStats::new(),
            last_svdag_stats: (0, 0, 0),
            occluded: false,
        }
    }

    fn upload_volume(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let data = self.world.flatten();
            renderer.upload_volume(&data);
            // Also rebuild the SVDAG so the other render path stays in sync.
            let dag = render::Svdag::build(&self.world.store, self.world.root, self.world.level);
            // Capture stats here so the wall-clock log path (fb6) reads
            // them without triggering a second build.
            self.last_svdag_stats = (dag.node_count, dag.byte_size(), dag.root_level);
            renderer.upload_svdag(&dag);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("hash-thing | 3D Hashlife Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());

            let renderer = pollster::block_on(render::Renderer::new(window.clone(), VOLUME_SIZE));
            self.renderer = Some(renderer);
            self.upload_volume();
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
            // hash-thing-8jp — previously the main loop ignored
            // `render()`'s outcome and unconditionally requested the next
            // redraw, pegging CPU at 100% when winit kept delivering
            // `RedrawRequested` into a surface that couldn't be acquired.
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
                            // Single step
                            self.world.step_flat(&self.rule);
                            self.upload_volume();
                            log::info!(
                                "Gen {}: pop={}",
                                self.world.generation,
                                self.world.population()
                            );
                        }
                        winit::keyboard::Key::Character("r") => {
                            // Reset — drop perf samples and mem peaks too so
                            // post-reset stats aren't poisoned by stale
                            // pre-reset values. The watchdog follows the new
                            // world's growth curve, not the old world's.
                            self.world = sim::World::new(VOLUME_SIZE.trailing_zeros());
                            self.world.seed_center(12, 0.35);
                            self.perf.clear();
                            self.mem_stats.reset_peaks();
                            self.upload_volume();
                            log::info!("Reset");
                        }
                        // TODO(hash-thing-6gf.1): call self.world.store.clear_step_cache() here
                        // once memoized stepping lands. See store.rs `step_cache` field doc for
                        // the contract — the cache key is NodeId only, so swapping the rule
                        // without clearing yields stale results from the previous rule.
                        winit::keyboard::Key::Character("1") => {
                            self.rule = sim::GameOfLife3D::amoeba();
                            log::info!("Rule: Amoeba ({})", self.rule);
                        }
                        // TODO(hash-thing-6gf.1): clear_step_cache on rule swap (see above).
                        winit::keyboard::Key::Character("2") => {
                            self.rule = sim::GameOfLife3D::crystal();
                            log::info!("Rule: Crystal ({})", self.rule);
                        }
                        // TODO(hash-thing-6gf.1): clear_step_cache on rule swap (see above).
                        winit::keyboard::Key::Character("3") => {
                            self.rule = sim::GameOfLife3D::rule445();
                            log::info!("Rule: 445 ({})", self.rule);
                        }
                        // TODO(hash-thing-6gf.1): clear_step_cache on rule swap (see above).
                        winit::keyboard::Key::Character("4") => {
                            self.rule = sim::GameOfLife3D::pyroclastic();
                            log::info!("Rule: Pyroclastic ({})", self.rule);
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
                    // Time the step. Bind both world+perf as locals before the
                    // closure so the borrow split is unambiguous (the closure
                    // captures locals, not `self`).
                    {
                        let world = &mut self.world;
                        let perf = &mut self.perf;
                        let rule = &self.rule;
                        perf.time("step", || world.step_flat(rule));
                    }
                    // Time upload as one aggregate (flatten + Svdag::build +
                    // upload_volume + upload_svdag). CPU-side submit only —
                    // wgpu queue writes are async.
                    let upload_start = std::time::Instant::now();
                    self.upload_volume();
                    self.perf.record("upload_cpu", upload_start.elapsed());

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

                // Time render. NLL lets us split-borrow self.renderer and
                // self.perf because the renderer borrow ends at `.render()`,
                // before `self.perf.record` reaches for &mut self again.
                let outcome = if let Some(renderer) = self.renderer.as_mut() {
                    let render_start = std::time::Instant::now();
                    let outcome = renderer.render();
                    self.perf.record("render_cpu", render_start.elapsed());
                    Some(outcome)
                } else {
                    None
                };

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
    log::info!("  R: reset");
    log::info!("  1-4: switch rules (amoeba, crystal, 445, pyroclastic)");
    log::info!("  V: toggle Flat3D / SVDAG rendering");
    log::info!("  Esc: quit");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
