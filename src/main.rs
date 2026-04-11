// Pre-existing clippy noise in this early-stage codebase: dead code paths
// scaffolded for upcoming features, convention-clash on `from_flat`, and
// cosmetic lints. Suppress at the crate level rather than churn unrelated
// files from a bugfix PR (hash-thing-88d is strictly about store compaction).
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(unused_imports)]

mod octree;
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

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<render::Renderer>,
    world: sim::World,
    rule: sim::GameOfLife3D,
    paused: bool,
    step_timer: std::time::Instant,
    // Mouse interaction state
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
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
            mouse_pressed: false,
            last_mouse: None,
        }
    }

    fn upload_volume(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let data = self.world.flatten();
            renderer.upload_volume(&data);
            // Also rebuild the SVDAG so the other render path stays in sync.
            let dag = render::Svdag::build(&self.world.store, self.world.root, self.world.level);
            if self.world.generation % 10 == 0 {
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
                            self.world.step_flat(&self.rule);
                            self.upload_volume();
                            log::info!(
                                "Gen {}: pop={}",
                                self.world.generation,
                                self.world.population()
                            );
                        }
                        winit::keyboard::Key::Character("r") => {
                            // Reset
                            self.world = sim::World::new(VOLUME_SIZE.trailing_zeros());
                            self.world.seed_center(12, 0.35);
                            self.upload_volume();
                            log::info!("Reset");
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
                    self.world.step_flat(&self.rule);
                    self.upload_volume();
                    self.step_timer = std::time::Instant::now();

                    if self.world.generation % 10 == 0 {
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
    log::info!("  R: reset");
    log::info!("  1-4: switch rules (amoeba, crystal, 445, pyroclastic)");
    log::info!("  V: toggle Flat3D / SVDAG rendering");
    log::info!("  Esc: quit");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
