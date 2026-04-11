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

/// One-line gen summary. Centralised so the `App::new` startup path and the
/// `R`-key reset path emit identical formatting. hash-thing-3fq.5 added the
/// classify_calls / nodes_delta / noise_fraction fields; the rest is carried
/// over from the pre-3fq.5 log line.
///
/// `noise_ns_per_sample` comes from `terrain::probe_sample_ns`. The "noise
/// fraction" number is `leaves * ns_per_sample / gen_time` — an *estimate*,
/// not a measurement. It deliberately skips timing individual sample calls
/// (which would roughly double gen cost). Read as "if every sample really
/// costs the probe's ns/call, here's how much of the pass it accounts for";
/// deviation from 100% means the bottleneck is elsewhere (classify, intern,
/// allocation). The probe is not precise enough to call anything under ~5%
/// or over ~95% with confidence.
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
    log::info!(
        "{label}: {side}^3 pop={pop} nodes={nodes} (+{delta}) \
         gen_calls={calls} samples={samples} classifies={classifies} collapses={collapses} \
         gen_time={gen_ms:.2}ms | noise~{ns:.0}ns/sample → ~{sample_pct:.0}% of gen",
        pop = population,
        delta = nodes_delta,
        calls = stats.calls_total,
        samples = stats.leaves,
        classifies = stats.classify_calls,
        collapses = stats.total_collapses(),
        ns = noise_ns_per_sample,
    );
}

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
    /// One-time microbench of `HeightmapField::sample()` in ns/call.
    /// Used to estimate noise fraction of each gen pass (hash-thing-3fq.5).
    /// Refreshed on terrain reset so a wildly different param set reprobes.
    noise_ns_per_sample: f64,
    // Mouse interaction state
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
}

impl App {
    fn new() -> Self {
        let mut world = sim::World::new(VOLUME_SIZE.trailing_zeros());
        let mut perf_counters = perf::PerfCounters::default();

        let params = terrain::TerrainParams::default();
        let nodes_before = world.store.stats().0;
        let (stats, elapsed) = perf::time(|| world.seed_terrain(&params));
        perf_counters.record_gen(elapsed, world.store.stats());

        // Noise-bottleneck probe. One-time microbench of `sample()`
        // called outside the timed gen region so it does not pollute
        // `record_gen`. See hash-thing-3fq.5.
        let noise_ns_per_sample = terrain::probe_sample_ns(&params.to_heightmap(), 10_000);

        let (nodes_after, _) = world.store.stats();
        let nodes_delta = nodes_after - nodes_before;
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
            noise_ns_per_sample,
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
                            let nodes_before = self.world.store.stats().0;
                            let (stats, elapsed) = perf::time(|| self.world.seed_terrain(&params));
                            self.perf.record_gen(elapsed, self.world.store.stats());
                            // Re-probe in case params drifted. Cheap (~1ms).
                            self.noise_ns_per_sample =
                                terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
                            self.paused = true;
                            self.upload_volume();
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
