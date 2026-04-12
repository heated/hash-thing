use hash_thing::perf;
use hash_thing::player;
use hash_thing::render;
use hash_thing::sim;
use hash_thing::terrain;

use std::collections::HashSet;
use std::sync::Arc;
use std::thread::JoinHandle;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::KeyCode,
    window::{Window, WindowAttributes},
};

use player::{CameraMode, LOOK_SENSITIVITY, PLAYER_HEIGHT, PLAYER_SPEED, PLAYER_SPRINT};

const DEFAULT_VOLUME_SIZE: u32 = 2048;

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
    log::info!(
        "{label}: {side}^3 pop={pop} nodes={nodes} (+{delta}) \
         gen_calls={calls} samples={samples} classifies={classifies} collapses={collapses} \
         gen_time={gen_ms:.2}ms (region={gen_region_ms:.2}ms cave={cave_ms:.2}ms) \
         nodes_after_gen={nag} nodes_after_caves={nac} | \
         noise~{ns:.0}ns/sample → ~{sample_pct:.0}% of gen",
        pop = population,
        delta = nodes_delta,
        calls = stats.calls_total,
        samples = stats.leaves,
        classifies = stats.classify_calls,
        collapses = stats.total_collapses(),
        nag = stats.nodes_after_gen,
        nac = stats.nodes_after_caves,
        ns = noise_ns_per_sample,
    );
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<render::Renderer>,
    world: sim::World,
    gol_smoke_rule: sim::GameOfLife3D,
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
    /// Currently held keyboard keys (for per-frame movement polling).
    keys_held: HashSet<KeyCode>,
    /// Camera mode: orbit (debug) or first-person (gameplay).
    camera_mode: CameraMode,
    /// The player entity, if spawned.
    player_id: Option<sim::EntityId>,
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
    /// Last frame timestamp for delta-time computation. Player movement
    /// is multiplied by dt so speed is frame-rate-independent (xa7).
    last_frame: std::time::Instant,
    /// Entity store: particles, projectiles, etc. Updated after each
    /// sim step. Entities push mutations onto `world.queue`; those are
    /// applied at the start of the next tick.
    entities: sim::EntityStore,
    volume_size: u32,
    /// Background sim step thread (x5w). While `Some`, `self.world` is a
    /// tiny placeholder — all world reads must use `render_origin` /
    /// `render_inv_size` or be guarded by `is_stepping()`.
    step_handle: Option<JoinHandle<sim::World>>,
    /// Cached world origin for rendering during background step.
    render_origin: [i64; 3],
    /// Cached 1/side for coordinate normalization during background step.
    render_inv_size: f32,
}

impl App {
    fn new(volume_size: u32) -> Self {
        let mut world = sim::World::new(volume_size.trailing_zeros());
        let terrain_params = terrain::TerrainParams {
            caves: Some(terrain::CaveParams::default()),
            ..Default::default()
        };
        let stats = world.seed_terrain(&terrain_params);
        let noise_ns = terrain::probe_sample_ns(&terrain_params.to_heightmap(), 10_000);
        let material_palette_len = world.materials.color_palette_rgba().len();

        log::info!(
            "Initial scene: terrain+caves pop={} nodes={} gen={}µs cave={}µs",
            world.population(),
            world.store.stats(),
            stats.gen_region_us,
            stats.cave_us,
        );
        log::debug!("Material registry palette slots={material_palette_len}");

        // Start running so materials interact immediately (powder-game feel).
        // F5 or Space (orbit mode) toggles pause.
        let render_inv_size = 1.0 / world.side() as f32;
        let render_origin = world.origin;
        let mut app = Self {
            window: None,
            renderer: None,
            world,
            gol_smoke_rule: sim::GameOfLife3D::rule445(),
            gol_smoke_scene: false,
            svdag: render::Svdag::new(),
            paused: false,
            step_timer: std::time::Instant::now(),
            log_timer: std::time::Instant::now(),
            mouse_pressed: false,
            last_mouse: None,
            keys_held: HashSet::new(),
            camera_mode: CameraMode::FirstPerson,
            player_id: None,
            perf: perf::Perf::new(),
            mem_stats: perf::MemStats::new(),
            last_svdag_stats: (0, 0, 0),
            occluded: false,
            noise_ns_per_sample: noise_ns,
            last_frame: std::time::Instant::now(),
            entities: sim::EntityStore::new(),
            volume_size,
            step_handle: None,
            render_origin,
            render_inv_size,
        };

        // Spawn the player entity at the world center, looking forward.
        let center = volume_size as f64 / 2.0;
        let player_id = app.entities.add(
            [center, center + 2.0, center],
            [0.0; 3],
            sim::EntityKind::Player(sim::PlayerState {
                yaw: std::f64::consts::FRAC_PI_4,
                pitch: 0.0,
                held_material: 1, // stone
            }),
        );
        app.player_id = Some(player_id);
        log::info!("Player spawned at ({center}, {}, {center})", center + 2.0);
        log::info!("Controls: WASD=move, mouse=look, LMB=break, RMB=place, scroll/1-7=material, F5=pause, Tab=orbit");

        app
    }

    /// True while the sim step is running on a background thread.
    fn is_stepping(&self) -> bool {
        self.step_handle.is_some()
    }

    /// Update cached render-side world geometry after world changes.
    fn sync_render_cache(&mut self) {
        self.render_origin = self.world.origin;
        self.render_inv_size = 1.0 / self.world.side() as f32;
    }

    /// Get the player's eye position and look direction.
    fn player_eye_ray(&self) -> Option<([f64; 3], [f64; 3])> {
        let pid = self.player_id?;
        let p = self.entities.iter().find(|e| e.id == pid)?;
        if let sim::EntityKind::Player(ref ps) = p.kind {
            Some(player::eye_ray(&p.pos, ps.yaw, ps.pitch))
        } else {
            None
        }
    }

    /// Break the block the player is looking at.
    fn break_block(&mut self) {
        if self.is_stepping() {
            return;
        }
        let Some((eye, dir)) = self.player_eye_ray() else {
            return;
        };
        if let Some((hit, _prev)) = player::raycast_cells(&self.world, eye, dir) {
            self.world.set(
                sim::WorldCoord(hit[0]),
                sim::WorldCoord(hit[1]),
                sim::WorldCoord(hit[2]),
                hash_thing::octree::Cell::EMPTY.raw(),
            );
            // Re-upload volume since we modified the world directly.
            Self::upload_volume(
                &mut self.renderer,
                &self.world,
                &mut self.svdag,
                &mut self.last_svdag_stats,
            );
        }
    }

    /// Place a block on the face the player is looking at.
    fn place_block(&mut self) {
        if self.is_stepping() {
            return;
        }
        let pid = self.player_id;
        let held_material = pid
            .and_then(|id| self.entities.iter().find(|e| e.id == id))
            .and_then(|e| {
                if let sim::EntityKind::Player(ref ps) = e.kind {
                    Some(ps.held_material)
                } else {
                    None
                }
            })
            .unwrap_or(1);

        let Some((eye, dir)) = self.player_eye_ray() else {
            return;
        };
        if let Some((hit, prev)) = player::raycast_cells(&self.world, eye, dir) {
            // Skip if origin is inside a solid cell (prev == hit on first step).
            if prev == hit {
                return;
            }
            // Place at the empty cell just before the hit.
            let state = hash_thing::octree::Cell::pack(held_material, 0).raw();
            self.world.set(
                sim::WorldCoord(prev[0]),
                sim::WorldCoord(prev[1]),
                sim::WorldCoord(prev[2]),
                state,
            );
            Self::upload_volume(
                &mut self.renderer,
                &self.world,
                &mut self.svdag,
                &mut self.last_svdag_stats,
            );
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

    fn select_rule(&mut self, rule: sim::GameOfLife3D, label: &str) {
        self.gol_smoke_rule = rule;
        if self.is_stepping() {
            return;
        }
        self.world.invalidate_rule_caches();
        if self.gol_smoke_scene {
            self.world.set_gol_smoke_rule(self.gol_smoke_rule);
            if let Some(renderer) = &mut self.renderer {
                renderer.upload_palette(&self.world.materials.color_palette_rgba());
            }
        }
        log::info!("Rule: {label}");
    }

    fn select_held_material(&mut self, material_id: u16) {
        let name = match material_id {
            1 => "Stone",
            2 => "Dirt",
            3 => "Grass",
            4 => "Fire",
            5 => "Water",
            6 => "Sand",
            7 => "Lava",
            _ => return,
        };
        if let Some(pid) = self.player_id {
            if let Some(entity) = self.entities.get_mut(pid) {
                if let sim::EntityKind::Player(ref mut ps) = entity.kind {
                    ps.held_material = material_id;
                    log::info!("Held: {name} ({material_id})");
                }
            }
        }
    }

    fn load_burning_room_demo(&mut self, label: &str) {
        if self.is_stepping() {
            return;
        }
        self.world = sim::World::new(self.volume_size.trailing_zeros());
        self.world.seed_burning_room();
        self.gol_smoke_scene = false;
        self.noise_ns_per_sample = 0.0;
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
        self.sync_render_cache();
        log::info!("{label}: pop={}", self.world.population());
    }

    fn load_terrain_scene(&mut self, label: &str, params: terrain::TerrainParams) {
        if self.is_stepping() {
            return;
        }
        let nodes_before = self.world.store.stats();
        let start = std::time::Instant::now();
        let stats = self.world.seed_terrain(&params);
        let elapsed = start.elapsed();
        self.noise_ns_per_sample = terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
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
        let nodes_after = self.world.store.stats();
        let nodes_delta = nodes_after.saturating_sub(nodes_before);
        log_gen_stats(
            label,
            self.world.side(),
            self.world.population(),
            nodes_after,
            nodes_delta,
            &stats,
            elapsed,
            self.noise_ns_per_sample,
        );
        self.sync_render_cache();
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
                pollster::block_on(render::Renderer::new(window.clone(), self.volume_size));
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

            WindowEvent::Focused(false) => {
                self.keys_held.clear();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // Track physical key state for per-frame movement polling.
                if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys_held.insert(code);
                        }
                        ElementState::Released => {
                            self.keys_held.remove(&code);
                        }
                    }
                }
                if event.state == ElementState::Pressed {
                    match event.logical_key.as_ref() {
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                            // In orbit mode, Space toggles pause.
                            // In FPS mode, Space is fly-up (handled per-frame).
                            if self.camera_mode == CameraMode::Orbit {
                                self.paused = !self.paused;
                                log::info!("Paused: {}", self.paused);
                            }
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::F5) => {
                            self.paused = !self.paused;
                            log::info!("Paused: {}", self.paused);
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab) => {
                            self.camera_mode = match self.camera_mode {
                                CameraMode::Orbit => CameraMode::FirstPerson,
                                CameraMode::FirstPerson => CameraMode::Orbit,
                            };
                            log::info!("Camera mode: {:?}", self.camera_mode);
                        }
                        winit::keyboard::Key::Character("s") if !self.is_stepping() => {
                            // Single step via recursive Hashlife path, matching
                            // the auto-step loop (hash-thing-6gf.8).
                            {
                                let _t = self.perf.start("step");
                                self.world.apply_mutations();
                                self.world.step_recursive();
                                let mut queue = std::mem::take(&mut self.world.queue);
                                self.entities.update(&self.world, &mut queue);
                                self.world.queue = queue;
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
                            self.load_terrain_scene(
                                "Caves terrain",
                                terrain::TerrainParams {
                                    caves: Some(terrain::CaveParams::default()),
                                    ..Default::default()
                                },
                            );
                        }
                        winit::keyboard::Key::Character("r") => {
                            // Reset the default CA/materials demo.
                            self.load_burning_room_demo("Reset burning room demo");
                        }
                        winit::keyboard::Key::Character("t") => {
                            // Plain terrain remains available as an explicit
                            // toggle, but is no longer the default scene.
                            self.load_terrain_scene(
                                "Reset terrain",
                                terrain::TerrainParams::default(),
                            );
                        }
                        winit::keyboard::Key::Character("g") if !self.is_stepping() => {
                            // Swap to the single retained GoL smoke seed.
                            self.world = sim::World::new(self.volume_size.trailing_zeros());
                            self.world.set_gol_smoke_rule(self.gol_smoke_rule);
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
                            self.sync_render_cache();
                            log::info!("Reset GoL smoke sphere: pop={}", self.world.population());
                        }
                        winit::keyboard::Key::Character(
                            n @ ("1" | "2" | "3" | "4" | "5" | "6" | "7"),
                        ) => {
                            let digit: u16 = n.parse().unwrap();
                            if self.camera_mode == CameraMode::FirstPerson {
                                // FPS mode: select held material.
                                self.select_held_material(digit);
                            } else {
                                // Orbit mode: select CA rule (1-4 only).
                                match digit {
                                    1 => self
                                        .select_rule(sim::GameOfLife3D::new(9, 26, 5, 7), "Amoeba"),
                                    2 => self
                                        .select_rule(sim::GameOfLife3D::new(0, 6, 1, 3), "Crystal"),
                                    3 => self.select_rule(sim::GameOfLife3D::rule445(), "445"),
                                    4 => self.select_rule(
                                        sim::GameOfLife3D::new(4, 7, 6, 8),
                                        "Pyroclastic",
                                    ),
                                    _ => {}
                                }
                            }
                        }
                        winit::keyboard::Key::Character("b") => {
                            // Burning room demo: fire + water + grass walls.
                            self.load_burning_room_demo("Reset burning room demo");
                        }
                        // hash-thing-hso: on-demand dump of the full perf +
                        // memory summary, independent of the wall-clock log
                        // cadence.
                        winit::keyboard::Key::Character("p") if !self.is_stepping() => {
                            let nodes = self.world.store.stats();
                            self.mem_stats.update(nodes);
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
                    if self.camera_mode == CameraMode::Orbit {
                        self.mouse_pressed = state == ElementState::Pressed;
                        if !self.mouse_pressed {
                            self.last_mouse = None;
                        }
                    } else if state == ElementState::Pressed {
                        // FPS mode: left click = break block.
                        self.break_block();
                    }
                }
                if button == MouseButton::Right
                    && state == ElementState::Pressed
                    && self.camera_mode == CameraMode::FirstPerson
                {
                    self.place_block();
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let should_look = match self.camera_mode {
                    CameraMode::Orbit => self.mouse_pressed,
                    CameraMode::FirstPerson => true, // always look in FPS
                };
                if should_look {
                    if let Some((lx, ly)) = self.last_mouse {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        if self.camera_mode == CameraMode::FirstPerson {
                            // Update player yaw/pitch directly.
                            if let Some(pid) = self.player_id {
                                if let Some(player) = self.entities.get_mut(pid) {
                                    if let sim::EntityKind::Player(ref mut ps) = player.kind {
                                        ps.yaw += dx * LOOK_SENSITIVITY;
                                        ps.pitch =
                                            (ps.pitch + dy * LOOK_SENSITIVITY).clamp(-1.4, 1.4);
                                    }
                                }
                            }
                        } else if let Some(renderer) = &mut self.renderer {
                            renderer.camera_yaw += dx as f32 * 0.005;
                            renderer.camera_pitch =
                                (renderer.camera_pitch + dy as f32 * 0.005).clamp(-1.4, 1.4);
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
                if self.camera_mode == CameraMode::Orbit {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_dist =
                            (renderer.camera_dist - scroll * 0.1).clamp(0.5, 10.0);
                    }
                } else if scroll.abs() > 0.01 {
                    // FPS mode: scroll cycles held material (1-7).
                    let next = self.player_id.and_then(|pid| {
                        let entity = self.entities.get_mut(pid)?;
                        if let sim::EntityKind::Player(ref ps) = entity.kind {
                            let dir = if scroll > 0.0 { 1i16 } else { -1 };
                            let cur = ps.held_material as i16;
                            Some(((cur - 1 + dir).rem_euclid(7) + 1) as u16)
                        } else {
                            None
                        }
                    });
                    if let Some(mat) = next {
                        self.select_held_material(mat);
                    }
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

                // Frame delta time for frame-rate-independent movement (xa7).
                let dt = self.last_frame.elapsed().as_secs_f64().min(0.1);
                self.last_frame = std::time::Instant::now();

                // --- Background step: collect completed result (x5w) ---
                if let Some(ref handle) = self.step_handle {
                    if handle.is_finished() {
                        let handle = self.step_handle.take().unwrap();
                        self.world = handle.join().expect("step thread panicked");
                        // Entity update runs on the main thread (needs
                        // both &World and &mut EntityStore).
                        let mut queue = std::mem::take(&mut self.world.queue);
                        self.entities.update(&self.world, &mut queue);
                        self.world.queue = queue;
                        self.sync_render_cache();
                        // SVDAG rebuild + GPU upload now that world is back.
                        {
                            let _t = self.perf.start("upload_cpu");
                            Self::upload_volume(
                                &mut self.renderer,
                                &self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            if let Some(renderer) = &mut self.renderer {
                                let inv_size = self.render_inv_size;
                                let wo = self.render_origin;
                                let particle_data: Vec<[f32; 4]> = self
                                    .entities
                                    .iter()
                                    .filter_map(|e| {
                                        let mat = match &e.kind {
                                            sim::EntityKind::Particle(p) => p.material as u32,
                                            sim::EntityKind::Player(_) => return None,
                                        };
                                        Some([
                                            (e.pos[0] - wo[0] as f64) as f32 * inv_size,
                                            (e.pos[1] - wo[1] as f64) as f32 * inv_size,
                                            (e.pos[2] - wo[2] as f64) as f32 * inv_size,
                                            f32::from_bits(mat),
                                        ])
                                    })
                                    .collect();
                                renderer.upload_particles(&particle_data);
                            }
                        }
                        self.step_timer = std::time::Instant::now();
                    }
                }

                // --- Kick off background step if due (x5w) ---
                if !self.paused
                    && self.step_timer.elapsed().as_millis() > 200
                    && !self.is_stepping()
                {
                    let _t = self.perf.start("step");
                    // Move world to background thread; replace with tiny
                    // placeholder so self.world remains valid (but inert).
                    let mut world = std::mem::replace(
                        &mut self.world,
                        sim::World::new(1),
                    );
                    self.step_handle = Some(std::thread::spawn(move || {
                        world.apply_mutations();
                        world.step_recursive();
                        world
                    }));
                }

                // Wall-clock perf summary (hash-thing-q63). Sits outside the
                // step gate so it fires on its own cadence regardless of
                // sim/step rate — including when paused. Showing the same
                // Gen repeatedly is intentional: it tells the user the app
                // is still alive.
                if self.log_timer.elapsed().as_secs_f64() >= LOG_INTERVAL_SECS {
                    if self.is_stepping() {
                        // World is on the background thread — just show perf.
                        log::info!("(stepping) | {}", self.perf.summary());
                    } else {
                        let nodes = self.world.store.stats();
                        self.mem_stats.update(nodes);
                        let (svdag_nodes, svdag_bytes, svdag_root_level) =
                            self.last_svdag_stats;
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
                    }
                    self.log_timer = std::time::Instant::now();
                }

                // Player movement (per-frame, not per-tick) and camera sync.
                // Reset HUD each frame — FPS block below sets it true.
                if let Some(renderer) = &mut self.renderer {
                    renderer.hud_visible = false;
                }
                if self.camera_mode == CameraMode::FirstPerson {
                    if let Some(pid) = self.player_id {
                        // Read player yaw for movement direction.
                        let yaw = self
                            .entities
                            .get_mut(pid)
                            .and_then(|e| match &e.kind {
                                sim::EntityKind::Player(ps) => Some(ps.yaw),
                                _ => None,
                            })
                            .unwrap_or(0.0);

                        // Gather input axes from held keys.
                        let fwd = if self.keys_held.contains(&KeyCode::KeyW) {
                            1.0
                        } else if self.keys_held.contains(&KeyCode::KeyS) {
                            -1.0
                        } else {
                            0.0
                        };
                        let right = if self.keys_held.contains(&KeyCode::KeyD) {
                            1.0
                        } else if self.keys_held.contains(&KeyCode::KeyA) {
                            -1.0
                        } else {
                            0.0
                        };
                        let up = if self.keys_held.contains(&KeyCode::Space) {
                            1.0
                        } else if self.keys_held.contains(&KeyCode::ShiftLeft)
                            || self.keys_held.contains(&KeyCode::ShiftRight)
                        {
                            -1.0
                        } else {
                            0.0
                        };

                        let speed = if self.keys_held.contains(&KeyCode::ControlLeft)
                            || self.keys_held.contains(&KeyCode::ControlRight)
                        {
                            PLAYER_SPEED * PLAYER_SPRINT
                        } else {
                            PLAYER_SPEED
                        };

                        let delta = player::compute_move_delta(yaw, [fwd, right, up], speed, dt);

                        let stepping = self.is_stepping();
                        if let Some(p) = self.entities.get_mut(pid) {
                            if stepping {
                                // Free-fly: no collision while world is on
                                // the background thread (x5w).
                                p.pos[0] += delta[0];
                                p.pos[1] += delta[1];
                                p.pos[2] += delta[2];
                            } else {
                                p.pos =
                                    player::apply_movement(&self.world, &p.pos, &delta);
                            }
                        }

                        // hash-thing-m1f.4 / 37r: grow the world when the
                        // player approaches any boundary (positive or negative).
                        // Skipped during background step — world is placeholder.
                        if !self.is_stepping() {
                            if let Some(p) = self.entities.get_mut(pid) {
                                const GROWTH_MARGIN: f64 = 8.0;
                                let origin = self.world.origin;
                                let side = self.world.side() as f64;
                                let pos = p.pos;
                                let margin = GROWTH_MARGIN as i64;
                                let near_pos_edge = pos.iter().enumerate().any(|(i, &c)| {
                                    c > origin[i] as f64 + side - GROWTH_MARGIN
                                });
                                let near_neg_edge = pos.iter().enumerate().any(|(i, &c)| {
                                    c < origin[i] as f64 + GROWTH_MARGIN
                                });
                                if near_pos_edge || near_neg_edge {
                                    let min = [
                                        sim::WorldCoord(pos[0] as i64 - margin),
                                        sim::WorldCoord(pos[1] as i64 - margin),
                                        sim::WorldCoord(pos[2] as i64 - margin),
                                    ];
                                    let max = [
                                        sim::WorldCoord(pos[0] as i64 + margin),
                                        sim::WorldCoord(
                                            (pos[1] + PLAYER_HEIGHT) as i64 + margin,
                                        ),
                                        sim::WorldCoord(pos[2] as i64 + margin),
                                    ];
                                    let old_level = self.world.level;
                                    self.world.ensure_region(min, max);
                                    if self.world.level != old_level {
                                        log::info!(
                                            "World grew: level {} → {} (side {}, origin {:?})",
                                            old_level,
                                            self.world.level,
                                            self.world.side(),
                                            self.world.origin,
                                        );
                                        Self::upload_volume(
                                            &mut self.renderer,
                                            &self.world,
                                            &mut self.svdag,
                                            &mut self.last_svdag_stats,
                                        );
                                        self.sync_render_cache();
                                    }
                                }
                            }
                        }

                        // Sync camera to player position and orientation.
                        // Uses cached render_origin / render_inv_size so this
                        // works even while world is on the step thread (x5w).
                        let inv_size = self.render_inv_size as f64;
                        let wo = self.render_origin;
                        if let Some(player) = self.entities.get_mut(pid) {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera_target = [
                                    (player.pos[0] - wo[0] as f64) as f32 * inv_size as f32,
                                    (player.pos[1] - wo[1] as f64 + PLAYER_HEIGHT * 0.85) as f32
                                        * inv_size as f32,
                                    (player.pos[2] - wo[2] as f64) as f32 * inv_size as f32,
                                ];
                                if let sim::EntityKind::Player(ref ps) = player.kind {
                                    renderer.camera_yaw = ps.yaw as f32;
                                    renderer.camera_pitch = ps.pitch as f32;
                                    if !stepping {
                                        let palette =
                                            self.world.materials.color_palette_rgba();
                                        let mat = ps.held_material as usize;
                                        if mat < palette.len() {
                                            renderer.hud_material_color = palette[mat];
                                        }
                                    }
                                }
                                renderer.camera_dist = 0.0;
                                renderer.hud_visible = true;
                            }
                        }
                    }
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
    log::info!("  Tab: toggle orbit / first-person camera");
    log::info!("  --- Orbit mode ---");
    log::info!("  Mouse drag: orbit camera");
    log::info!("  Scroll: zoom");
    log::info!("  Space: pause/resume");
    log::info!("  --- First-person mode ---");
    log::info!("  WASD: move (relative to look direction)");
    log::info!("  Mouse: look around");
    log::info!("  Space: fly up   Shift: fly down");
    log::info!("  Ctrl: sprint");
    log::info!("  Left click: break block   Right click: place block");
    log::info!("  --- Shared ---");
    log::info!("  F5: pause/resume");
    log::info!("  S: single step");
    log::info!("  R: reset terrain (heightmap)");
    log::info!("  C: reset terrain with caves (CA post-pass)");
    log::info!("  G: reset to legacy GoL sphere seed");
    log::info!("  1-4: switch rules (amoeba, crystal, 445, pyroclastic)");
    log::info!("  P: dump perf + memory summary (on demand)");
    log::info!("  Esc: quit");

    let volume_size = std::env::args()
        .nth(1)
        .map(|s| {
            let n: u32 = s.parse().expect("usage: hash-thing [SIZE]  (SIZE must be a power of 2)");
            assert!(n.is_power_of_two(), "volume size must be a power of 2 (got {n})");
            n
        })
        .unwrap_or(DEFAULT_VOLUME_SIZE);
    log::info!("Volume: {volume_size}^3 (level {})", volume_size.trailing_zeros());

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new(volume_size);
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
