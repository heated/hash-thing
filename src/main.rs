mod octree;
mod perf;
mod render;
mod rng;
mod sim;
mod terrain;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

const VOLUME_SIZE: u32 = 64;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<render::Renderer>,
    world: sim::World,
    rule: sim::GameOfLife3D,
    paused: bool,
    step_timer: std::time::Instant,
    /// When to next emit an auto perf summary. Reset on each emission.
    perf_log_timer: std::time::Instant,
    perf: perf::PerfCounters,
    // Mouse interaction state
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        let mut world = sim::World::new(VOLUME_SIZE.trailing_zeros());
        let mut perf_counters = perf::PerfCounters::default();

        let params = terrain::TerrainParams::default();
        let (stats, elapsed) = perf::time(|| world.seed_terrain(&params));
        perf_counters.record_gen(elapsed, world.store.stats());

        let (nodes, _) = world.store.stats();
        log::info!(
            "Initial terrain: {}^3, population={}, nodes={}, gen_calls={}, collapses={}, gen_time={:.2}ms",
            world.side(),
            world.population(),
            nodes,
            stats.calls_total,
            stats.total_collapses(),
            elapsed.as_micros() as f64 / 1_000.0,
        );

        // Start paused so the active CA rule (legacy GoL) does not
        // immediately treat solid terrain as alive and destroy it. Press
        // Space to step.
        Self {
            window: None,
            renderer: None,
            world,
            rule: sim::GameOfLife3D::amoeba(),
            paused: true,
            step_timer: std::time::Instant::now(),
            perf_log_timer: std::time::Instant::now(),
            perf: perf_counters,
            mouse_pressed: false,
            last_mouse: None,
        }
    }

    fn upload_volume(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            // `flatten` is intentionally NOT folded into the perf timer.
            // h34.3's spec names "step, terrain gen, DAG serialization";
            // flatten is the Flat3D render path's own cost, not the
            // SVDAG build's. If the Flat3D renderer is retired later,
            // this call goes away; if it stays, it deserves its own
            // counter rather than being hidden inside `svdag[...]`.
            let data = self.world.flatten();
            renderer.upload_volume(&data);
            // Measure just the rebuild — the GPU upload is wgpu's problem.
            let (dag, elapsed) = perf::time(|| {
                render::Svdag::build(&self.world.store, self.world.root, self.world.level)
            });
            self.perf.record_svdag(elapsed, self.world.store.stats());
            if self.world.generation.is_multiple_of(10) {
                log::info!(
                    "SVDAG: {} nodes, {} bytes, root_level={}",
                    dag.node_count,
                    dag.byte_size(),
                    dag.root_level,
                );
            }
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
                            let (_, elapsed) = perf::time(|| self.world.step_flat(&self.rule));
                            self.perf.record_step(elapsed, self.world.store.stats());
                            self.upload_volume();
                            log::info!(
                                "Gen {}: pop={}",
                                self.world.generation,
                                self.world.population()
                            );
                        }
                        winit::keyboard::Key::Character("r") => {
                            // Re-seed terrain. Stays paused.
                            let params = terrain::TerrainParams::default();
                            let (stats, elapsed) = perf::time(|| self.world.seed_terrain(&params));
                            self.perf.record_gen(elapsed, self.world.store.stats());
                            self.paused = true;
                            self.upload_volume();
                            log::info!(
                                "Reset terrain: pop={}, gen_calls={}, collapses={}, gen_time={:.2}ms",
                                self.world.population(),
                                stats.calls_total,
                                stats.total_collapses(),
                                elapsed.as_micros() as f64 / 1_000.0,
                            );
                        }
                        winit::keyboard::Key::Character("p") => {
                            // Dump perf summary on demand.
                            log::info!("{}", self.perf.summary());
                        }
                        winit::keyboard::Key::Character("g") => {
                            // Swap to legacy GoL sphere seed (kept for CA scaffold demos).
                            self.world = sim::World::new(VOLUME_SIZE.trailing_zeros());
                            self.world.seed_center(12, 0.35);
                            self.paused = true;
                            self.upload_volume();
                            log::info!("Reset GoL sphere: pop={}", self.world.population(),);
                        }
                        winit::keyboard::Key::Character("1") => {
                            self.rule = sim::GameOfLife3D::amoeba();
                            log::info!("Rule: Amoeba");
                        }
                        winit::keyboard::Key::Character("2") => {
                            self.rule = sim::GameOfLife3D::crystal();
                            log::info!("Rule: Crystal");
                        }
                        winit::keyboard::Key::Character("3") => {
                            self.rule = sim::GameOfLife3D::rule445();
                            log::info!("Rule: 445");
                        }
                        winit::keyboard::Key::Character("4") => {
                            self.rule = sim::GameOfLife3D::pyroclastic();
                            log::info!("Rule: Pyroclastic");
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
                // Step simulation
                if !self.paused && self.step_timer.elapsed().as_millis() > 200 {
                    let (_, elapsed) = perf::time(|| self.world.step_flat(&self.rule));
                    self.perf.record_step(elapsed, self.world.store.stats());
                    self.upload_volume();
                    self.step_timer = std::time::Instant::now();

                    if self.world.generation.is_multiple_of(10) {
                        let (nodes, cache) = self.world.store.stats();
                        log::info!(
                            "Gen {}: pop={}, nodes={}, cache={}",
                            self.world.generation,
                            self.world.population(),
                            nodes,
                            cache
                        );
                    }
                }

                // Auto perf summary while the simulation is running. We
                // skip the dump when paused so the log stays quiet if you
                // tab out after seeding a world. Press `P` for an
                // on-demand summary regardless of pause state.
                if !self.paused && self.perf_log_timer.elapsed().as_secs() >= 2 {
                    log::info!("{}", self.perf.summary());
                    self.perf_log_timer = std::time::Instant::now();
                }

                if let Some(renderer) = &mut self.renderer {
                    renderer.render();
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
    log::info!("  G: reset to legacy GoL sphere seed");
    log::info!("  1-4: switch rules (amoeba, crystal, 445, pyroclastic)");
    log::info!("  V: toggle Flat3D / SVDAG rendering");
    log::info!("  P: dump perf summary");
    log::info!("  Esc: quit");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
