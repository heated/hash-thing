use hash_thing::acquire_harness::{self, Action as HarnessAction};
use hash_thing::perf;
use hash_thing::player;
use hash_thing::render;
use hash_thing::sim;
use hash_thing::terrain;

use std::collections::HashSet;
use std::sync::Arc;
use std::thread::JoinHandle;
#[cfg(target_os = "macos")]
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::KeyCode,
    window::{CursorGrabMode, Window, WindowAttributes},
};

use player::{CameraMode, LOOK_SENSITIVITY, PLAYER_HEIGHT, PLAYER_SPEED, PLAYER_SPRINT};

const DEFAULT_VOLUME_SIZE: u32 = 2048;

/// Wall-clock cadence for the consolidated perf log line. Decoupled from
/// `world.generation` so the log keeps ticking even when the sim is paused
/// or stepping slowly — see hash-thing-q63.
const LOG_INTERVAL_SECS: f64 = 2.0;
const DEV_PROFILE_STEP_WARN_MS: u64 = 500;

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
    log::info!(
        "{label}: {side}^3 pop={pop} nodes={nodes} (+{delta}) \
         gen_calls={calls} samples={samples} classifies={classifies} collapses={collapses} \
         gen_time={gen_ms:.2}ms (region={gen_region_ms:.2}ms) \
         nodes_after_gen={nag} | \
         noise~{ns:.0}ns/sample → ~{sample_pct:.0}% of gen",
        pop = population,
        delta = nodes_delta,
        calls = stats.calls_total,
        samples = stats.leaves,
        classifies = stats.classify_calls,
        collapses = stats.total_collapses(),
        nag = stats.nodes_after_gen,
        ns = noise_ns_per_sample,
    );
}

fn collect_visible_particle_data(
    world: &sim::World,
    entities: &sim::EntityStore,
    camera_pos: [f64; 3],
    render_origin: [i64; 3],
    render_inv_size: f32,
) -> Vec<[f32; 4]> {
    entities
        .iter()
        .filter_map(|entity| {
            let mat = match &entity.kind {
                sim::EntityKind::Player(_) | sim::EntityKind::Emitter(_) => return None,
                sim::EntityKind::Particle(_) | sim::EntityKind::Critter(_) => {
                    entity.render_material().unwrap() as u32
                }
            };
            if !player::has_line_of_sight(world, camera_pos, entity.pos) {
                return None;
            }
            Some([
                (entity.pos[0] - render_origin[0] as f64) as f32 * render_inv_size,
                (entity.pos[1] - render_origin[1] as f64) as f32 * render_inv_size,
                (entity.pos[2] - render_origin[2] as f64) as f32 * render_inv_size,
                f32::from_bits(mat),
            ])
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlayerPose {
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrbitCameraPose {
    target: [f32; 3],
    yaw: f32,
    pitch: f32,
    dist: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingPlayerAction {
    Break,
    Place,
    PlaceClone,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LatticeDemoBeat {
    Intro,
    Interior,
    Panorama,
}

impl LatticeDemoBeat {
    const ALL: [Self; 3] = [Self::Intro, Self::Interior, Self::Panorama];

    fn label(self) -> &'static str {
        match self {
            Self::Intro => "intro",
            Self::Interior => "interior",
            Self::Panorama => "panorama",
        }
    }

    fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|beat| *beat == self)
            .expect("current beat must exist in the beat list");
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    fn previous(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|beat| *beat == self)
            .expect("current beat must exist in the beat list");
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LatticeShortDemoCut {
    started_at: std::time::Instant,
}

impl LatticeShortDemoCut {
    fn beat_for_elapsed(elapsed: std::time::Duration) -> Option<LatticeDemoBeat> {
        let secs = elapsed.as_secs_f32();
        if secs < 2.0 {
            Some(LatticeDemoBeat::Intro)
        } else if secs < 4.5 {
            Some(LatticeDemoBeat::Interior)
        } else if secs < 8.0 {
            Some(LatticeDemoBeat::Panorama)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DemoWaypoint {
    label: &'static str,
    player: PlayerPose,
    camera: OrbitCameraPose,
}

fn lattice_demo_waypoint(side: usize, beat: LatticeDemoBeat) -> DemoWaypoint {
    let side = side as f64;
    let center = side * 0.5;
    match beat {
        LatticeDemoBeat::Intro => {
            let offset = side * 0.28;
            DemoWaypoint {
                label: "intro",
                player: PlayerPose {
                    pos: [center - offset, side * 0.34, center - offset],
                    yaw: -3.0 * std::f64::consts::FRAC_PI_4,
                    pitch: -0.08,
                },
                camera: OrbitCameraPose {
                    target: [0.36, 0.34, 0.36],
                    yaw: -3.0 * std::f32::consts::FRAC_PI_4,
                    pitch: 0.08,
                    dist: 0.34,
                },
            }
        }
        LatticeDemoBeat::Interior => DemoWaypoint {
            label: "interior",
            player: PlayerPose {
                pos: [center + side * 0.04, side * 0.46, center + side * 0.10],
                yaw: std::f64::consts::FRAC_PI_2,
                pitch: -0.12,
            },
            camera: OrbitCameraPose {
                target: [0.54, 0.46, 0.50],
                yaw: std::f32::consts::FRAC_PI_2,
                pitch: 0.10,
                dist: 0.22,
            },
        },
        LatticeDemoBeat::Panorama => {
            let offset = side * 0.22;
            DemoWaypoint {
                label: "panorama",
                player: PlayerPose {
                    pos: [center + offset, side * 0.58, center + offset],
                    yaw: std::f64::consts::FRAC_PI_4,
                    pitch: 0.24,
                },
                camera: OrbitCameraPose {
                    target: [0.62, 0.72, 0.62],
                    yaw: std::f32::consts::FRAC_PI_4,
                    pitch: 0.32,
                    dist: 0.82,
                },
            }
        }
    }
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
    /// Wall-clock checkpoint for the next perf summary line. Reset each
    /// time the line fires so cadence stays ~LOG_INTERVAL_SECS regardless
    /// of sim/step rate.
    log_timer: std::time::Instant,
    // Mouse interaction state
    mouse_pressed: bool,
    last_mouse: Option<(f64, f64)>,
    /// Currently held keyboard keys (for per-frame movement polling).
    keys_held: HashSet<KeyCode>,
    /// Last frame's Space state so jump only triggers on a fresh press.
    jump_was_held: bool,
    /// Pause redraw-driven rendering when the window loses focus.
    focused: bool,
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
    step_handle: Option<JoinHandle<Result<sim::World, String>>>,
    /// When the background step was spawned, for perf timing.
    step_start: std::time::Instant,
    /// Cached world origin for rendering during background step.
    render_origin: [i64; 3],
    /// Cached 1/side for coordinate normalization during background step.
    render_inv_size: f32,
    /// Log the slow-debug-build advisory at most once per run.
    warned_dev_profile_perf: bool,
    /// Legend (keybindings help) overlay toggle (m1f.7.2).
    legend_visible: bool,
    /// True when the legend texture needs re-upload (mode change or toggle).
    legend_dirty: bool,
    /// Last lattice debug-jump preset used from orbit mode.
    current_demo_beat: Option<LatticeDemoBeat>,
    /// Timed intro -> interior -> reveal cut for the short-form demo.
    short_demo_cut: Option<LatticeShortDemoCut>,
    /// Deterministic first-person camera motion layer (bob, settle, sprint cue).
    camera_feel: player::FirstPersonCameraFeel,
    /// Defer expensive initial scene generation until after the window exists.
    startup_scene_pending: bool,
    /// Tracks whether the cursor is currently grabbed/hidden for FPS look.
    cursor_captured: bool,
    /// Replay FPS interactions on the next live-world frame instead of
    /// dropping them while a background step is in flight.
    pending_player_action: Option<PendingPlayerAction>,
    /// Fullscreen toggle state (dlse.2.2). Cycles between None and
    /// Fullscreen::Borderless(None) via F11 or Cmd+Ctrl+F. Initial
    /// state set from HASH_THING_FULLSCREEN=1.
    fullscreen_active: bool,
    /// Latest modifier key state, tracked via ModifiersChanged, used
    /// to detect the Cmd+Ctrl+F fullscreen chord (winit does not
    /// surface chords as NamedKey).
    modifiers: winit::keyboard::ModifiersState,
    /// Cached hashlife memo health summary (hash-thing-stue.6). Refreshed
    /// on the main thread after each step's world result is merged back,
    /// so the periodic log can print it even while the next step is on
    /// the background thread (where `self.world` is a placeholder).
    last_memo_summary: String,
    /// Sim-freeze diagnostic toggle (hash-thing-stue.7). When true, the
    /// background-step dispatch is a no-op: world stays put on the main
    /// thread, generation never advances, and the renderer reads a stable
    /// SVDAG every frame. Used to measure renderer-only frame cost without
    /// sim/render unified-memory contention. Set via `HASH_THING_FREEZE_SIM=1`.
    freeze_sim: bool,
    /// Self-driving windowed-vs-fullscreen acquire measurement
    /// (`dlse.2.2`). `Some` when `HASH_THING_ACQUIRE_HARNESS=1`.
    /// When active, the harness forces a windowed start, burns a
    /// warmup window, records capture samples, flips fullscreen,
    /// records again, logs a side-by-side report, and asks the event
    /// loop to exit.
    acquire_harness: Option<acquire_harness::Harness>,
}

fn should_warn_about_slow_dev_step(
    debug_build: bool,
    volume_size: u32,
    step_elapsed: std::time::Duration,
) -> bool {
    debug_build
        && volume_size >= 256
        && step_elapsed >= std::time::Duration::from_millis(DEV_PROFILE_STEP_WARN_MS)
}

fn default_legend_visibility(mode: CameraMode) -> bool {
    matches!(mode, CameraMode::Orbit)
}

fn should_capture_cursor(camera_mode: CameraMode, focused: bool) -> bool {
    focused && camera_mode == CameraMode::FirstPerson
}

/// Drop any modifier KeyCodes from `keys` whose combined bit is *not* set in
/// `state`. Returns the count removed so callers can log only when something
/// actually changed. hash-thing-hnoh.
fn reconcile_modifier_keys(
    state: winit::keyboard::ModifiersState,
    keys: &mut HashSet<KeyCode>,
) -> usize {
    use winit::keyboard::ModifiersState;
    let mut removed = 0;
    for (bit, lhs, rhs) in [
        (
            ModifiersState::SHIFT,
            KeyCode::ShiftLeft,
            KeyCode::ShiftRight,
        ),
        (
            ModifiersState::CONTROL,
            KeyCode::ControlLeft,
            KeyCode::ControlRight,
        ),
        (ModifiersState::ALT, KeyCode::AltLeft, KeyCode::AltRight),
        (
            ModifiersState::SUPER,
            KeyCode::SuperLeft,
            KeyCode::SuperRight,
        ),
    ] {
        if !state.contains(bit) {
            if keys.remove(&lhs) {
                removed += 1;
            }
            if keys.remove(&rhs) {
                removed += 1;
            }
        }
    }
    removed
}

impl App {
    fn new(volume_size: u32) -> Self {
        let level = volume_size.trailing_zeros();
        let world = sim::World::new(level);

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
            log_timer: std::time::Instant::now(),
            mouse_pressed: false,
            last_mouse: None,
            keys_held: HashSet::new(),
            jump_was_held: false,
            focused: true,
            camera_mode: CameraMode::FirstPerson,
            player_id: None,
            perf: perf::Perf::new(),
            mem_stats: perf::MemStats::new(),
            last_svdag_stats: (0, 0, 0),
            occluded: false,
            noise_ns_per_sample: 0.0,
            last_frame: std::time::Instant::now(),
            entities: sim::EntityStore::new(),
            volume_size,
            step_handle: None,
            step_start: std::time::Instant::now(),
            render_origin,
            render_inv_size,
            warned_dev_profile_perf: false,
            legend_visible: default_legend_visibility(CameraMode::FirstPerson),
            legend_dirty: true,
            current_demo_beat: None,
            short_demo_cut: None,
            camera_feel: player::FirstPersonCameraFeel::default(),
            startup_scene_pending: true,
            cursor_captured: false,
            pending_player_action: None,
            fullscreen_active: std::env::var("HASH_THING_FULLSCREEN").ok().as_deref() == Some("1"),
            modifiers: winit::keyboard::ModifiersState::empty(),
            last_memo_summary: String::new(),
            freeze_sim: std::env::var("HASH_THING_FREEZE_SIM").ok().as_deref() == Some("1"),
            acquire_harness: acquire_harness::Harness::from_env(),
        };
        if app.freeze_sim {
            log::info!("HASH_THING_FREEZE_SIM=1: sim step disabled (stue.7 diagnostic)");
        }
        // The harness owns the windowed→fullscreen transition explicitly;
        // honouring `HASH_THING_FULLSCREEN=1` on top of it would skip the
        // windowed capture phase. Force windowed start when the harness is
        // on.
        if app.acquire_harness.is_some() {
            app.fullscreen_active = false;
        }

        // Seed the cached memo summary so the very first periodic log line
        // has a populated column instead of a blank trailing field
        // (hash-thing-stue.6 reviewer nit).
        app.last_memo_summary = app.world.memo_summary();
        let player_pos = app.reset_scene_entities();
        app.spawn_demo_entities();
        log::info!(
            "Player spawned at ({}, {}, {})",
            player_pos[0],
            player_pos[1],
            player_pos[2]
        );
        log::info!("Controls: WASD=move, mouse=look, LMB=break, RMB=place, scroll/1-9=material, F5=pause, Tab=orbit");

        app
    }

    fn apply_cursor_capture(&mut self, capture: bool) {
        let Some(window) = self.window.as_ref() else {
            self.cursor_captured = false;
            return;
        };

        if capture {
            let grab = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            match grab {
                Ok(()) => {
                    window.set_cursor_visible(false);
                    self.cursor_captured = true;
                    self.last_mouse = None;
                }
                Err(err) => {
                    log::warn!("Failed to grab FPS cursor: {err}");
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                    self.cursor_captured = false;
                }
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
            self.cursor_captured = false;
            self.last_mouse = None;
        }
    }

    fn sync_cursor_capture(&mut self) {
        let should_capture = should_capture_cursor(self.camera_mode, self.focused);
        if should_capture == self.cursor_captured {
            return;
        }
        self.apply_cursor_capture(should_capture);
    }

    /// Apply a raw pixel-delta mouse look to the player. Scales by
    /// `LOOK_SENSITIVITY` and clamps pitch to ±1.4. Shared between the
    /// `CursorMoved` (uncaptured) and `DeviceEvent::MouseMotion` (captured)
    /// paths so the two stay in parity by construction (hash-thing-a6t2).
    fn apply_fps_look(&mut self, dx: f64, dy: f64) {
        let Some(pid) = self.player_id else { return };
        let Some(player) = self.entities.get_mut(pid) else {
            return;
        };
        if let sim::EntityKind::Player(ref mut ps) = player.kind {
            ps.yaw += dx * LOOK_SENSITIVITY;
            ps.pitch = (ps.pitch + dy * LOOK_SENSITIVITY).clamp(-1.4, 1.4);
        }
    }

    /// Handle a `WindowEvent::CursorMoved` at `(x, y)`. Routes to the player
    /// (first-person, uncaptured cursor) or the orbit-camera renderer
    /// depending on mode. Extracted so tests can drive it without a window
    /// (hash-thing-a6t2).
    fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        let should_look = match self.camera_mode {
            CameraMode::Orbit => self.mouse_pressed,
            CameraMode::FirstPerson => !self.cursor_captured,
        };
        if !should_look {
            return;
        }
        if let Some((lx, ly)) = self.last_mouse {
            let dx = x - lx;
            let dy = y - ly;
            match self.camera_mode {
                CameraMode::FirstPerson => self.apply_fps_look(dx, dy),
                CameraMode::Orbit => {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_yaw += dx as f32 * 0.005;
                        renderer.camera_pitch =
                            (renderer.camera_pitch + dy as f32 * 0.005).clamp(-1.4, 1.4);
                    }
                }
            }
        }
        self.last_mouse = Some((x, y));
    }

    /// Handle a captured-cursor `DeviceEvent::MouseMotion` delta. No-op
    /// unless the player is in first-person mode with the cursor grabbed
    /// (hash-thing-w1yq). Extracted for testability (hash-thing-a6t2).
    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.camera_mode != CameraMode::FirstPerson || !self.cursor_captured {
            return;
        }
        self.apply_fps_look(dx, dy);
    }

    fn load_initial_scene(&mut self) {
        let terrain_params = terrain::TerrainParams::for_level(self.volume_size.trailing_zeros());
        let stats = self
            .world
            .seed_terrain(&terrain_params)
            .expect("level-derived terrain params must validate");
        self.world.seed_water_and_sand();
        self.noise_ns_per_sample = terrain::probe_sample_ns(&terrain_params.to_heightmap(), 10_000);
        self.reset_scene_entities();
        self.spawn_demo_entities();
        self.paused = false;
        self.perf.clear();
        self.mem_stats.reset_peaks();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials.color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        log::info!(
            "Initial scene: terrain pop={} nodes={} gen={}µs",
            self.world.population(),
            self.world.store.stats(),
            stats.gen_region_us,
        );
        log::debug!(
            "Material registry palette slots={}",
            self.world.materials.color_palette_rgba().len()
        );
    }

    fn world_center(&self) -> [f64; 3] {
        let side = self.world.side() as f64;
        [
            self.world.origin[0] as f64 + side * 0.5,
            self.world.origin[1] as f64 + side * 0.5,
            self.world.origin[2] as f64 + side * 0.5,
        ]
    }

    fn recenter_player(&mut self) -> bool {
        let Some(pid) = self.player_id else {
            return false;
        };
        let center = self.world_center();
        let Some(player) = self.entities.get_mut(pid) else {
            return false;
        };
        player.pos = [center[0], center[1] + 2.0, center[2]];
        true
    }

    fn reset_scene_entities(&mut self) -> [f64; 3] {
        let player_state = self
            .player_id
            .and_then(|id| self.entities.iter().find(|entity| entity.id == id))
            .and_then(|entity| match &entity.kind {
                sim::EntityKind::Player(state) => Some(state.clone()),
                _ => None,
            })
            .unwrap_or(sim::PlayerState {
                yaw: std::f64::consts::FRAC_PI_4,
                pitch: 0.0,
                held_material: 1,
            });
        let center = self.world_center();
        let pos = [center[0], center[1] + 2.0, center[2]];
        self.entities = sim::EntityStore::new();
        self.player_id = Some(self.entities.add(
            pos,
            [0.0; 3],
            sim::EntityKind::Player(player_state),
        ));
        pos
    }

    fn spawn_demo_entities(&mut self) {
        let center = self.world_center();
        let mid_y = center[1] + 1.0;
        self.entities.add(
            [center[0] - 10.0, mid_y, center[2] - 4.0],
            [0.0; 3],
            sim::EntityKind::Emitter(sim::EmitterState::geyser()),
        );
        self.entities.add(
            [center[0] + 10.0, mid_y + 2.0, center[2] + 6.0],
            [0.0; 3],
            sim::EntityKind::Emitter(sim::EmitterState::volcano()),
        );
        self.entities.add(
            [center[0] + 2.0, mid_y, center[2] - 12.0],
            [0.0; 3],
            sim::EntityKind::Emitter(sim::EmitterState::whirlpool()),
        );
        for offset in [-8.0, 0.0, 8.0] {
            self.entities.add(
                [center[0] + offset, mid_y, center[2] + 12.0],
                [0.0; 3],
                sim::EntityKind::Critter(sim::CritterState::new(
                    hash_thing::terrain::materials::VINE_MATERIAL_ID,
                )),
            );
        }
        log::info!("Spawned environmental demo entities: geyser, volcano, whirlpool, critters");
    }

    fn renderer_camera_world_pos(&self, renderer: &render::Renderer) -> [f64; 3] {
        let target = [
            self.render_origin[0] as f64
                + renderer.camera_target[0] as f64 / self.render_inv_size as f64,
            self.render_origin[1] as f64
                + renderer.camera_target[1] as f64 / self.render_inv_size as f64,
            self.render_origin[2] as f64
                + renderer.camera_target[2] as f64 / self.render_inv_size as f64,
        ];
        let (sin_yaw, cos_yaw) = renderer.camera_yaw.sin_cos();
        let (sin_pitch, cos_pitch) = renderer.camera_pitch.sin_cos();
        let cam_dir = [-cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw];
        [
            target[0]
                - cam_dir[0] as f64 * renderer.camera_dist as f64 / self.render_inv_size as f64,
            target[1]
                - cam_dir[1] as f64 * renderer.camera_dist as f64 / self.render_inv_size as f64,
            target[2]
                - cam_dir[2] as f64 * renderer.camera_dist as f64 / self.render_inv_size as f64,
        ]
    }

    fn upload_visible_particles(&mut self) {
        if self.is_stepping() {
            return;
        }
        let Some(camera_pos) = self
            .renderer
            .as_ref()
            .map(|renderer| self.renderer_camera_world_pos(renderer))
        else {
            return;
        };

        let particle_data = collect_visible_particle_data(
            &self.world,
            &self.entities,
            camera_pos,
            self.render_origin,
            self.render_inv_size,
        );

        if let Some(renderer) = &mut self.renderer {
            renderer.upload_particles(&particle_data);
        }
    }

    /// True while the sim step is running on a background thread.
    fn is_stepping(&self) -> bool {
        self.step_handle.is_some()
    }

    fn maybe_start_background_step(&mut self) {
        if self.paused || self.is_stepping() || self.freeze_sim {
            return;
        }
        self.step_start = std::time::Instant::now();
        // Move world to the background thread only after the frame has
        // consumed the live snapshot for movement, interaction, and render.
        let mut world = std::mem::replace(&mut self.world, sim::World::placeholder());
        self.step_handle = Some(std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                world.apply_mutations();
                world.spawn_clones();
                world.step_recursive();
                world
            }))
            .map_err(|e| {
                if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                }
            })
        }));
    }

    fn run_pending_player_action(&mut self) {
        let Some(action) = self.pending_player_action.take() else {
            return;
        };
        match action {
            PendingPlayerAction::Break => self.break_block(),
            PendingPlayerAction::Place => self.place_block(),
            PendingPlayerAction::PlaceClone => self.place_clone_block(),
        }
    }

    /// Legend text lines for the current camera mode.
    fn legend_lines(mode: CameraMode) -> Vec<&'static str> {
        match mode {
            CameraMode::FirstPerson => vec![
                "  FIELD LINK",
                "",
                "  WASD        Drift",
                "  Mouse       Aim",
                "  Space       Leap",
                "  Ctrl        Surge",
                "  LClick      Carve",
                "  RClick      Cast",
                "  Scroll/1-9  Matter",
                "  Ctrl+RClick Clone source",
                "  Tab         Survey cam",
                "",
                "  T  Terrain    B  Spectacle",
                "  R  Reset      G  GoL bloom",
                "  M  Gyroid     N  Lattice walk",
                "  0  Recenter",
                "  H  Heatmap    +/-  Resolution",
                "  F5 Pause      F1  Signal legend",
                "  F11/Cmd+Ctrl+F Fullscreen   C Clear perf",
                "  Esc Exit",
            ],
            CameraMode::Orbit => vec![
                "  SURVEY CAM",
                "",
                "  LClick+Drag Orbit",
                "  Scroll      Push / pull",
                "  Space       Pause",
                "  S           Single step",
                "  1-4         Rule set",
                "  Tab         Field link",
                "",
                "  T  Terrain    B  Spectacle",
                "  R  Reset      G  GoL bloom",
                "  M  Gyroid     N  Lattice walk",
                "  [/] DEV prev/next jump",
                "  U/I/O DEV intro/interior/reveal",
                "  V  DEV tweet reveal",
                "  0  Recenter",
                "  H  Heatmap    +/-  Resolution",
                "  F5 Pause      F1  Signal legend",
                "  F11/Cmd+Ctrl+F Fullscreen   C Clear perf",
                "  Esc Exit",
            ],
        }
    }

    fn lattice_debug_jumps_enabled(&self) -> bool {
        self.camera_mode == CameraMode::Orbit
    }

    /// Toggle borderless-fullscreen (dlse.2.2). Logs the chosen variant so
    /// post-hoc log forensics can tell which state the app was in at
    /// measurement time.
    fn toggle_fullscreen(&mut self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        self.fullscreen_active = !self.fullscreen_active;
        let mode = if self.fullscreen_active {
            Some(winit::window::Fullscreen::Borderless(None))
        } else {
            None
        };
        log::info!("fullscreen: toggling to {:?}", mode);
        window.set_fullscreen(mode);
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

    fn apply_player_pose(&mut self, pose: PlayerPose) {
        if let Some(pid) = self.player_id {
            if let Some(entity) = self.entities.get_mut(pid) {
                entity.pos = pose.pos;
                if let sim::EntityKind::Player(ref mut ps) = entity.kind {
                    ps.yaw = pose.yaw;
                    ps.pitch = pose.pitch;
                }
            }
        }
    }

    fn apply_orbit_camera_pose(&mut self, pose: OrbitCameraPose) {
        self.camera_mode = CameraMode::Orbit;
        self.camera_feel.reset();
        self.legend_visible = default_legend_visibility(self.camera_mode);
        self.legend_dirty = true;
        if let Some(renderer) = &mut self.renderer {
            renderer.camera_target = pose.target;
            renderer.camera_yaw = pose.yaw;
            renderer.camera_pitch = pose.pitch;
            renderer.camera_dist = pose.dist;
        }
    }

    fn load_lattice_demo_beat(&mut self, beat: LatticeDemoBeat) {
        self.load_lattice_demo();
        let waypoint = lattice_demo_waypoint(self.world.side(), beat);
        self.apply_player_pose(waypoint.player);
        self.apply_orbit_camera_pose(waypoint.camera);
        self.current_demo_beat = Some(beat);
        log::info!("Lattice debug jump: {} ({})", waypoint.label, beat.label());
    }

    fn cycle_lattice_demo_beat(&mut self, step: i32) {
        self.short_demo_cut = None;
        let current = self.current_demo_beat.unwrap_or(LatticeDemoBeat::Intro);
        let next = if step >= 0 {
            current.next()
        } else {
            current.previous()
        };
        self.load_lattice_demo_beat(next);
    }

    fn select_lattice_demo_beat(&mut self, beat: LatticeDemoBeat) {
        self.short_demo_cut = None;
        self.load_lattice_demo_beat(beat);
    }

    fn start_lattice_short_demo_cut(&mut self) {
        self.short_demo_cut = Some(LatticeShortDemoCut {
            started_at: std::time::Instant::now(),
        });
        self.load_lattice_demo_beat(LatticeDemoBeat::Intro);
        log::info!("Lattice short cut: intro -> interior -> reveal");
    }

    fn update_lattice_short_demo_cut(&mut self) {
        let Some(cut) = self.short_demo_cut else {
            return;
        };
        match LatticeShortDemoCut::beat_for_elapsed(cut.started_at.elapsed()) {
            Some(beat) => {
                if self.current_demo_beat != Some(beat) {
                    self.load_lattice_demo_beat(beat);
                }
            }
            None => self.short_demo_cut = None,
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
            // Remove from clone_sources if this was a clone block.
            self.world
                .clone_sources
                .retain(|pos| pos != &[hit[0], hit[1], hit[2]]);
            self.world.set(
                sim::WorldCoord(hit[0]),
                sim::WorldCoord(hit[1]),
                sim::WorldCoord(hit[2]),
                hash_thing::octree::Cell::EMPTY.raw(),
            );
            // Re-upload volume since we modified the world directly.
            Self::upload_volume(
                &mut self.renderer,
                &mut self.world,
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
                &mut self.world,
                &mut self.svdag,
                &mut self.last_svdag_stats,
            );
            if let Some(window) = &self.window {
                window.request_redraw();
            }
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
        world: &mut sim::World,
        svdag: &mut render::Svdag,
        last_svdag_stats: &mut (usize, usize, u32),
    ) {
        if let Some(renderer) = renderer {
            // Apply the NodeId remap from the last compaction so the SVDAG's
            // persistent cache stays valid. This turns update() from O(reachable)
            // to O(changed) — unchanged subtrees hit the cache in O(1)
            // (hash-thing-5bb.11).
            if let Some(remap) = world.last_compaction_remap.take() {
                svdag.apply_remap(&remap);
            }
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

    fn load_demo_spectacle(&mut self, label: &str) {
        if self.is_stepping() {
            return;
        }
        self.world = sim::World::new(self.volume_size.trailing_zeros());
        self.world.seed_demo_spectacle();
        self.reset_scene_entities();
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
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        log::info!("{label}: pop={}", self.world.population());
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
            8 => "Ice",
            9 => "Acid",
            10 => "Oil",
            11 => "Gunpowder",
            12 => "Steam",
            13 => "Gas",
            14 => "Metal",
            15 => "Vine",
            16 => "Fan",
            17 => "Firework",
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

    /// Place a clone block that continuously spawns the currently held material.
    /// The source material ID is encoded in the clone cell's metadata.
    fn place_clone_block(&mut self) {
        if self.is_stepping() {
            return;
        }
        let held_material = self
            .player_id
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
            if prev == hit {
                return;
            }
            // Encode the source material in clone block metadata.
            let state = hash_thing::octree::Cell::pack(
                hash_thing::terrain::materials::CLONE_MATERIAL_ID,
                held_material,
            )
            .raw();
            let pos = [prev[0], prev[1], prev[2]];
            self.world.set(
                sim::WorldCoord(pos[0]),
                sim::WorldCoord(pos[1]),
                sim::WorldCoord(pos[2]),
                state,
            );
            self.world.clone_sources.push(pos);
            log::info!(
                "Placed clone block (spawns material {held_material}) at {:?}",
                pos
            );
            Self::upload_volume(
                &mut self.renderer,
                &mut self.world,
                &mut self.svdag,
                &mut self.last_svdag_stats,
            );
        }
    }

    #[allow(dead_code)]
    fn load_burning_room_demo(&mut self, label: &str) {
        if self.is_stepping() {
            return;
        }
        self.world = sim::World::new(self.volume_size.trailing_zeros());
        self.world.seed_burning_room();
        self.reset_scene_entities();
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
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.current_demo_beat = None;
        log::info!("{label}: pop={}", self.world.population());
    }

    fn reset_player_pose(&mut self, pos: [f64; 3], yaw: f64, pitch: f64) {
        let Some(pid) = self.player_id else {
            return;
        };
        let Some(entity) = self.entities.get_mut(pid) else {
            return;
        };
        entity.pos = pos;
        if let sim::EntityKind::Player(ref mut ps) = entity.kind {
            ps.yaw = yaw;
            ps.pitch = pitch;
        }
    }

    fn load_gyroid_demo(&mut self) {
        if self.is_stepping() {
            return;
        }
        let start = std::time::Instant::now();
        self.world = sim::World::new(self.volume_size.trailing_zeros());
        let stats = self.world.seed_gyroid_megastructure();
        self.reset_scene_entities();
        self.spawn_demo_entities();
        let elapsed = start.elapsed();
        self.gol_smoke_scene = false;
        self.noise_ns_per_sample = 0.0;
        self.paused = false; // Let materials interact immediately.
        self.perf.clear();
        self.mem_stats.reset_peaks();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials.color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.current_demo_beat = None;
        log::info!(
            "Gyroid megastructure: pop={} gen={:.1}ms collapses={} classifies={}",
            self.world.population(),
            elapsed.as_secs_f64() * 1000.0,
            stats.total_collapses(),
            stats.classify_calls,
        );
    }

    fn load_lattice_demo(&mut self) {
        if self.is_stepping() {
            return;
        }
        let start = std::time::Instant::now();
        self.world = sim::World::new(self.volume_size.trailing_zeros());
        let layout = self.world.seed_lattice_progression_demo();
        self.reset_scene_entities();
        self.spawn_demo_entities();
        self.reset_player_pose(layout.player_pos, layout.player_yaw, layout.player_pitch);
        let elapsed = start.elapsed();
        self.gol_smoke_scene = false;
        self.noise_ns_per_sample = 0.0;
        self.paused = false; // Let materials interact immediately.
        self.perf.clear();
        self.mem_stats.reset_peaks();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials.color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.current_demo_beat = None;
        self.short_demo_cut = None;
        log::info!(
            "Lattice progression demo: pop={} gen={:.1}ms reveal={:?}",
            self.world.population(),
            elapsed.as_secs_f64() * 1000.0,
            layout.reveal_center,
        );
    }

    fn load_lattice_panorama_demo(&mut self) {
        self.start_lattice_short_demo_cut();
    }

    fn load_terrain_scene(&mut self, label: &str, params: terrain::TerrainParams) {
        if self.is_stepping() {
            return;
        }
        let nodes_before = self.world.store.stats();
        let start = std::time::Instant::now();
        let stats = self
            .world
            .seed_terrain(&params)
            .expect("UI-generated terrain params must validate");
        self.reset_scene_entities();
        self.spawn_demo_entities();
        let elapsed = start.elapsed();
        self.noise_ns_per_sample = terrain::probe_sample_ns(&params.to_heightmap(), 10_000);
        self.gol_smoke_scene = false;
        self.paused = true;
        self.perf.clear();
        self.mem_stats.reset_peaks();
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
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
        self.current_demo_beat = None;
        self.short_demo_cut = None;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("hash-thing | 3D Hashlife Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));
            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("failed to create main window"),
            );
            // Agent/CLI launches on macOS can leave the app alive but unfocused.
            window.set_visible(true);
            window.focus_window();
            if self.fullscreen_active {
                // dlse.2.2: env-var opt-in. `Borderless(None)` targets the
                // monitor currently containing the window. This will fire
                // `Resized` and transition the macOS Space.
                window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                log::info!(
                    "fullscreen: entering Borderless(None) at startup (HASH_THING_FULLSCREEN=1)"
                );
            }
            self.window = Some(window.clone());

            let mut renderer =
                pollster::block_on(render::Renderer::new(window.clone(), self.volume_size));
            renderer.upload_palette(&self.world.materials.color_palette_rgba());
            // dlse.2.2 step 3: off-surface render-target diagnostic. Bypasses
            // `surface.get_current_texture()` + `present()`; pairs with the
            // acquire harness to measure whether the ~25 ms surface_acquire
            // stall is swapchain-pacing (collapses) or elsewhere (persists).
            if std::env::var("HASH_THING_OFF_SURFACE").ok().as_deref() == Some("1") {
                renderer.enable_off_surface();
            }
            self.renderer = Some(renderer);
            // Initial upload — untimed; we haven't started the render
            // loop yet and there's no perf summary to feed.
            Self::upload_volume(
                &mut self.renderer,
                &mut self.world,
                &mut self.svdag,
                &mut self.last_svdag_stats,
            );
            self.sync_cursor_capture();
            // Some macOS / agent launches do not schedule an initial redraw
            // on their own. Arm the first frame explicitly so startup scene
            // generation and the steady redraw loop can begin.
            window.request_redraw();
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

            WindowEvent::Focused(focused) => {
                self.focused = focused;
                if focused {
                    self.last_mouse = None;
                    self.sync_cursor_capture();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                } else {
                    self.keys_held.clear();
                    self.jump_was_held = false;
                    self.last_mouse = None;
                    self.sync_cursor_capture();
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                // dlse.2.2: track modifier state for Cmd+Ctrl+F chord detection.
                self.modifiers = mods.state();
                // hash-thing-hnoh: macOS can deliver `flagsChanged:` without a
                // usable NSEvent keyCode — e.g. when a system shortcut
                // (Cmd+Shift+3) is intercepted — so winit emits
                // `ModifiersChanged` but no per-side `KeyboardInput { Released }`.
                // Reconcile `keys_held` against the combined state bitflag as
                // a best-effort cleanup.
                let removed = reconcile_modifier_keys(self.modifiers, &mut self.keys_held);
                if removed > 0 {
                    log::debug!("ModifiersChanged reconciled {removed} stuck modifier key(s)");
                }
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
                    // dlse.2.2: Cmd+Ctrl+F fullscreen chord. Check by physical
                    // key + modifier state since winit does not surface chords
                    // as NamedKey and macOS may alter logical_key under Cmd.
                    if let winit::keyboard::PhysicalKey::Code(KeyCode::KeyF) = event.physical_key {
                        if self.modifiers.super_key() && self.modifiers.control_key() {
                            self.toggle_fullscreen();
                            return;
                        }
                    }
                    match event.logical_key.as_ref() {
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                            // In orbit mode, Space toggles pause.
                            // In FPS mode, Space is jump (handled per-frame).
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
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::F11) => {
                            // dlse.2.2: fullscreen toggle. Primary shortcut is
                            // Cmd+Ctrl+F on macOS (reachable even when F11 is
                            // swallowed by the window server).
                            self.toggle_fullscreen();
                        }
                        winit::keyboard::Key::Character("0") => {
                            // Recenter player at world center.
                            if self.recenter_player() {
                                log::info!("Player recentered");
                            }
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::F1) => {
                            self.legend_visible = !self.legend_visible;
                            self.legend_dirty = true;
                        }
                        winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab) => {
                            self.camera_mode = match self.camera_mode {
                                CameraMode::Orbit => CameraMode::FirstPerson,
                                CameraMode::FirstPerson => CameraMode::Orbit,
                            };
                            if self.camera_mode == CameraMode::Orbit {
                                self.camera_feel.reset();
                            }
                            self.legend_visible = default_legend_visibility(self.camera_mode);
                            self.sync_cursor_capture();
                            self.legend_dirty = true;
                            log::info!("Camera mode: {:?}", self.camera_mode);
                        }
                        winit::keyboard::Key::Character("s")
                            if self.camera_mode == CameraMode::Orbit && !self.is_stepping() =>
                        {
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
                            self.last_memo_summary = self.world.memo_summary();
                            Self::upload_volume(
                                &mut self.renderer,
                                &mut self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            log::info!(
                                "Gen {}: pop={}",
                                self.world.generation,
                                self.world.population()
                            );
                        }
                        winit::keyboard::Key::Character("r") => {
                            self.load_terrain_scene(
                                "Reset terrain",
                                terrain::TerrainParams::for_level(
                                    self.volume_size.trailing_zeros(),
                                ),
                            );
                        }
                        winit::keyboard::Key::Character("t") => {
                            // Plain terrain remains available as an explicit
                            // toggle, but is no longer the default scene.
                            self.load_terrain_scene(
                                "Reset terrain",
                                terrain::TerrainParams::for_level(
                                    self.volume_size.trailing_zeros(),
                                ),
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
                                &mut self.world,
                                &mut self.svdag,
                                &mut self.last_svdag_stats,
                            );
                            self.sync_render_cache();
                            log::info!("Reset GoL smoke sphere: pop={}", self.world.population());
                        }
                        winit::keyboard::Key::Character("m") => {
                            self.load_gyroid_demo();
                        }
                        winit::keyboard::Key::Character("n") => {
                            self.load_lattice_demo();
                        }
                        winit::keyboard::Key::Character("[")
                            if self.lattice_debug_jumps_enabled() =>
                        {
                            self.cycle_lattice_demo_beat(-1);
                        }
                        winit::keyboard::Key::Character("]")
                            if self.lattice_debug_jumps_enabled() =>
                        {
                            self.cycle_lattice_demo_beat(1);
                        }
                        winit::keyboard::Key::Character("u")
                            if self.lattice_debug_jumps_enabled() =>
                        {
                            self.select_lattice_demo_beat(LatticeDemoBeat::Intro);
                        }
                        winit::keyboard::Key::Character("i")
                            if self.lattice_debug_jumps_enabled() =>
                        {
                            self.select_lattice_demo_beat(LatticeDemoBeat::Interior);
                        }
                        winit::keyboard::Key::Character("o")
                            if self.lattice_debug_jumps_enabled() =>
                        {
                            self.select_lattice_demo_beat(LatticeDemoBeat::Panorama);
                        }
                        winit::keyboard::Key::Character("v")
                            if self.lattice_debug_jumps_enabled() =>
                        {
                            self.load_lattice_panorama_demo();
                        }
                        winit::keyboard::Key::Character(
                            n @ ("1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"),
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
                            // Default demo gallery: deterministic local fire/water set pieces
                            // staged around the beat waypoints.
                            self.load_demo_spectacle("Reset spectacle gallery");
                        }
                        winit::keyboard::Key::Character("c") => {
                            // dlse.2.2: drain perf histograms so the next `P`
                            // dump reflects only post-clear samples. Needed for
                            // clean windowed-vs-fullscreen comparisons.
                            self.perf.clear();
                            log::info!("perf histograms cleared");
                        }
                        winit::keyboard::Key::Character("h") => {
                            // Toggle step-count heatmap debug mode.
                            if let Some(renderer) = &mut self.renderer {
                                renderer.debug_mode = if renderer.debug_mode == 1 { 0 } else { 1 };
                                log::info!(
                                    "Debug mode: {}",
                                    if renderer.debug_mode == 1 {
                                        "step heatmap"
                                    } else {
                                        "normal"
                                    }
                                );
                            }
                        }
                        winit::keyboard::Key::Character("l") => {
                            // Cycle LOD bias: 1 → 2 → 4 → 8 → 1.
                            if let Some(renderer) = &mut self.renderer {
                                renderer.lod_bias = if renderer.lod_bias >= 8.0 {
                                    1.0
                                } else {
                                    renderer.lod_bias * 2.0
                                };
                                log::info!("LOD bias: {}x", renderer.lod_bias);
                            }
                        }
                        winit::keyboard::Key::Character("=")
                        | winit::keyboard::Key::Character("+") => {
                            // Increase render scale (sharper, slower).
                            if let Some(renderer) = &mut self.renderer {
                                let w = self.window.as_ref().unwrap();
                                let size = w.inner_size();
                                renderer.render_scale = (renderer.render_scale + 0.25).min(1.0);
                                renderer.resize(size.width, size.height);
                                log::info!("Render scale: {:.0}%", renderer.render_scale * 100.0);
                            }
                        }
                        winit::keyboard::Key::Character("-") => {
                            // Decrease render scale (blurrier, faster).
                            if let Some(renderer) = &mut self.renderer {
                                let w = self.window.as_ref().unwrap();
                                let size = w.inner_size();
                                renderer.render_scale = (renderer.render_scale - 0.25).max(0.25);
                                renderer.resize(size.width, size.height);
                                log::info!("Render scale: {:.0}%", renderer.render_scale * 100.0);
                            }
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
                        if self.is_stepping() {
                            self.pending_player_action = Some(PendingPlayerAction::Break);
                        } else {
                            self.break_block();
                        }
                    }
                }
                if button == MouseButton::Right
                    && state == ElementState::Pressed
                    && self.camera_mode == CameraMode::FirstPerson
                {
                    if self.keys_held.contains(&KeyCode::ControlLeft)
                        || self.keys_held.contains(&KeyCode::ControlRight)
                    {
                        if self.is_stepping() {
                            self.pending_player_action = Some(PendingPlayerAction::PlaceClone);
                        } else {
                            self.place_clone_block();
                        }
                    } else {
                        if self.is_stepping() {
                            self.pending_player_action = Some(PendingPlayerAction::Place);
                        } else {
                            self.place_block();
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                // When the cursor is grabbed (FPS look), macOS delivers
                // relative motion via `DeviceEvent::MouseMotion`. `CursorMoved`
                // either stops firing or reports a stationary position, so we
                // must not feed its deltas into look input — see `device_event`
                // below. (hash-thing-w1yq)
                self.handle_cursor_moved(position.x, position.y);
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
                    // FPS mode: scroll cycles held material (1-17).
                    let next = self.player_id.and_then(|pid| {
                        let entity = self.entities.get_mut(pid)?;
                        if let sim::EntityKind::Player(ref ps) = entity.kind {
                            let dir = if scroll > 0.0 { 1i16 } else { -1 };
                            let cur = ps.held_material as i16;
                            Some(((cur - 1 + dir).rem_euclid(17) + 1) as u16)
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
                //
                // Exception: when the acquire-harness is running we want
                // frames to keep coming even if the window manager decides
                // to unfocus us — otherwise a headless crew launch can
                // stall forever on frame 0 waiting on focus it'll never
                // get.
                if self.acquire_harness.is_none() && (self.occluded || !self.focused) {
                    return;
                }

                if self.startup_scene_pending {
                    self.startup_scene_pending = false;
                    self.load_initial_scene();
                }

                // Frame delta time for frame-rate-independent movement (xa7).
                let dt = self.last_frame.elapsed().as_secs_f64().min(0.1);
                self.last_frame = std::time::Instant::now();
                self.update_lattice_short_demo_cut();

                // Update window title with FPS + resolution.
                if let Some(window) = &self.window {
                    let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                    if let Some(renderer) = &self.renderer {
                        let scale_pct = (renderer.render_scale * 100.0) as u32;
                        window.set_title(&format!(
                            "hash-thing | {:.0} FPS | {}³ | scale {}%",
                            fps, self.volume_size, scale_pct,
                        ));
                    }
                }

                // --- Background step: collect completed result (x5w) ---
                if let Some(ref handle) = self.step_handle {
                    if handle.is_finished() {
                        let handle = self.step_handle.take().unwrap();
                        let step_elapsed = self.step_start.elapsed();
                        self.perf.record("step", step_elapsed);
                        if !self.warned_dev_profile_perf
                            && should_warn_about_slow_dev_step(
                                cfg!(debug_assertions),
                                self.volume_size,
                                step_elapsed,
                            )
                        {
                            log::warn!(
                                "Sim step took {:.1}ms at {}^3 in a debug build; use `cargo run --profile bench -- {}` for interactive playtesting.",
                                step_elapsed.as_secs_f64() * 1000.0,
                                self.volume_size,
                                self.volume_size,
                            );
                            self.warned_dev_profile_perf = true;
                        }
                        match handle.join().expect("step thread aborted") {
                            Ok(world) => {
                                self.world = world;
                                // Refresh cached memo-health summary while the
                                // real world is in hand (hash-thing-stue.6):
                                // the (stepping) log branch below cannot read
                                // self.world (it's about to be replaced by a
                                // placeholder again on the next step).
                                self.last_memo_summary = self.world.memo_summary();
                                // Entity update on main thread (needs both
                                // &World and &mut EntityStore).
                                let mut queue = std::mem::take(&mut self.world.queue);
                                self.entities.update(&self.world, &mut queue);
                                self.world.queue = queue;
                                self.sync_render_cache();
                                // SVDAG rebuild + GPU upload.
                                {
                                    let _t = self.perf.start("upload_cpu");
                                    Self::upload_volume(
                                        &mut self.renderer,
                                        &mut self.world,
                                        &mut self.svdag,
                                        &mut self.last_svdag_stats,
                                    );
                                }
                            }
                            Err(msg) => {
                                // Step panicked — log and pause sim. The
                                // placeholder world stays in place; render
                                // continues with the stale SVDAG (4lp).
                                log::error!("Sim step panicked: {msg}");
                                self.paused = true;
                            }
                        }
                    }
                }

                if !self.is_stepping() {
                    self.run_pending_player_action();
                }

                // Wall-clock perf summary (hash-thing-q63). Sits outside the
                // step gate so it fires on its own cadence regardless of
                // sim/step rate — including when paused. Showing the same
                // Gen repeatedly is intentional: it tells the user the app
                // is still alive.
                if self.log_timer.elapsed().as_secs_f64() >= LOG_INTERVAL_SECS {
                    if self.is_stepping() {
                        // World is on the background thread — just show perf.
                        log::info!(
                            "(stepping) | {} | {}",
                            self.perf.summary(),
                            self.last_memo_summary,
                        );
                    } else {
                        let nodes = self.world.store.stats();
                        self.mem_stats.update(nodes);
                        let (svdag_nodes, svdag_bytes, svdag_root_level) = self.last_svdag_stats;
                        log::info!(
                            "Gen {}: pop={} svdag={}/{}KB(L{}) | {} | {} | {}",
                            self.world.generation,
                            self.world.population(),
                            svdag_nodes,
                            svdag_bytes / 1024,
                            svdag_root_level,
                            self.mem_stats.summary(),
                            self.perf.summary(),
                            self.last_memo_summary,
                        );
                    }
                    self.log_timer = std::time::Instant::now();
                }

                // Player movement (per-frame, not per-tick) and camera sync.
                // Reset HUD each frame — FPS block below sets it true.
                if let Some(renderer) = &mut self.renderer {
                    renderer.hud_visible = false;
                    renderer.hotbar_visible = false;

                    // Legend overlay (m1f.7.2): upload text when dirty.
                    if self.legend_dirty {
                        self.legend_dirty = false;
                        renderer.legend_visible = self.legend_visible;
                        if self.legend_visible {
                            renderer.set_legend_text(&Self::legend_lines(self.camera_mode));
                        }
                    }
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
                        let space_held = self.keys_held.contains(&KeyCode::Space);
                        let jump_requested = space_held && !self.jump_was_held;
                        let up = if space_held {
                            1.0
                        } else if self.keys_held.contains(&KeyCode::ShiftLeft)
                            || self.keys_held.contains(&KeyCode::ShiftRight)
                        {
                            -1.0
                        } else {
                            0.0
                        };

                        let sprinting = self.keys_held.contains(&KeyCode::ControlLeft)
                            || self.keys_held.contains(&KeyCode::ControlRight);
                        let speed = if sprinting {
                            PLAYER_SPEED * PLAYER_SPRINT
                        } else {
                            PLAYER_SPEED
                        };

                        let delta = player::compute_move_delta(yaw, [fwd, right, up], speed, dt);
                        let camera_planar_speed =
                            (delta[0] * delta[0] + delta[2] * delta[2]).sqrt() / dt.max(1e-6);

                        let stepping = self.is_stepping();
                        let mut camera_grounded = false;
                        if let Some(p) = self.entities.get_mut(pid) {
                            if stepping {
                                // Free-fly: no collision while world is on
                                // the background thread (x5w).
                                p.pos[0] += delta[0];
                                p.pos[1] += delta[1];
                                p.pos[2] += delta[2];
                            } else {
                                let step = player::step_grounded_movement(
                                    &self.world,
                                    &p.pos,
                                    p.vel[1],
                                    player::GroundedMoveInput {
                                        yaw,
                                        move_input: [fwd, right],
                                        speed,
                                        dt,
                                        jump_requested,
                                    },
                                );
                                p.pos = step.pos;
                                p.vel[1] = step.vertical_velocity;
                            }
                        }
                        if !stepping {
                            if let Some(p) = self.entities.iter().find(|entity| entity.id == pid) {
                                camera_grounded = player::is_grounded(&self.world, &p.pos);
                            }
                        }
                        let camera_motion = (camera_planar_speed, sprinting, camera_grounded);
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
                                let near_pos_edge = pos
                                    .iter()
                                    .enumerate()
                                    .any(|(i, &c)| c > origin[i] as f64 + side - GROWTH_MARGIN);
                                let near_neg_edge = pos
                                    .iter()
                                    .enumerate()
                                    .any(|(i, &c)| c < origin[i] as f64 + GROWTH_MARGIN);
                                if near_pos_edge || near_neg_edge {
                                    let min = [
                                        sim::WorldCoord(pos[0] as i64 - margin),
                                        sim::WorldCoord(pos[1] as i64 - margin),
                                        sim::WorldCoord(pos[2] as i64 - margin),
                                    ];
                                    let max = [
                                        sim::WorldCoord(pos[0] as i64 + margin),
                                        sim::WorldCoord((pos[1] + PLAYER_HEIGHT) as i64 + margin),
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
                                            &mut self.world,
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
                                if let sim::EntityKind::Player(ref ps) = player.kind {
                                    let (camera_planar_speed, camera_sprinting, camera_grounded) =
                                        camera_motion;
                                    let motion = self.camera_feel.tick(
                                        camera_planar_speed,
                                        camera_sprinting,
                                        camera_grounded,
                                        dt,
                                    );
                                    let (sin_yaw, cos_yaw) = ps.yaw.sin_cos();
                                    let right = [cos_yaw, 0.0, -sin_yaw];
                                    let forward = [-sin_yaw, 0.0, -cos_yaw];
                                    let eye = [
                                        player.pos[0]
                                            + right[0] * motion.lateral
                                            + forward[0] * motion.forward,
                                        player.pos[1] + PLAYER_HEIGHT * 0.85 + motion.vertical,
                                        player.pos[2]
                                            + right[2] * motion.lateral
                                            + forward[2] * motion.forward,
                                    ];
                                    renderer.camera_target = [
                                        (eye[0] - wo[0] as f64) as f32 * inv_size as f32,
                                        (eye[1] - wo[1] as f64) as f32 * inv_size as f32,
                                        (eye[2] - wo[2] as f64) as f32 * inv_size as f32,
                                    ];
                                    renderer.camera_yaw = ps.yaw as f32;
                                    renderer.camera_pitch = ps.pitch as f32;
                                    renderer.hotbar_selected_slot =
                                        ps.held_material.saturating_sub(1) as u32;
                                    if !stepping {
                                        let palette = self.world.materials.color_palette_rgba();
                                        let mat = ps.held_material as usize;
                                        if mat < palette.len() {
                                            renderer.hud_material_color = palette[mat];
                                        }
                                    }
                                }
                                renderer.camera_dist = 0.0;
                                renderer.hud_visible = true;
                                renderer.hotbar_visible = false;
                            }
                        }
                    }
                }
                self.jump_was_held = self.keys_held.contains(&KeyCode::Space);

                self.upload_visible_particles();

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
                    if let Some(cpu_times) = renderer.take_last_cpu_phase_times() {
                        self.perf
                            .record("surface_acquire_cpu", cpu_times.surface_acquire);
                        self.perf.record("submit_cpu", cpu_times.submit);
                        self.perf.record("present_cpu", cpu_times.present);
                    }
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

                // Kick off the next background step only after this frame has
                // finished consuming the live world snapshot.
                self.maybe_start_background_step();

                // Advance the self-driving acquire harness (dlse.2.2).
                // Runs after this frame's samples are in `self.perf`, so
                // snapshots reflect the just-completed capture window.
                // Two-pass drain: `step()` holds `&mut self.acquire_harness`
                // + `&self.perf`, so we collect actions first and release
                // those borrows before dispatching (which can touch `&mut
                // self.perf`, `&mut self` for the fullscreen toggle, and
                // `&self.acquire_harness` for the final report).
                let harness_actions: Vec<HarnessAction> =
                    if let Some(harness) = self.acquire_harness.as_mut() {
                        harness.step(&self.perf)
                    } else {
                        Vec::new()
                    };
                for action in harness_actions {
                    match action {
                        HarnessAction::ClearPerf => {
                            self.perf.clear();
                            log::info!("acquire-harness: perf cleared");
                        }
                        HarnessAction::ToggleFullscreen => {
                            log::info!(
                                "acquire-harness: windowed capture done, toggling to fullscreen"
                            );
                            self.toggle_fullscreen();
                        }
                        HarnessAction::Exit => {
                            if let Some(h) = self.acquire_harness.as_ref() {
                                log::info!("{}", h.report());
                            }
                            event_loop.exit();
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // macOS delivers relative mouse motion here when the cursor is
        // grabbed/locked; `WindowEvent::CursorMoved` stops firing or
        // reports a stationary position in that state. Without this,
        // FPS look is dead whenever the cursor is actually captured.
        // hash-thing-w1yq.
        let DeviceEvent::MouseMotion { delta: (dx, dy) } = event else {
            return;
        };
        self.handle_mouse_motion(dx, dy);
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
    log::info!("  Space: jump   Ctrl: sprint");
    log::info!("  Left click: break block   Right click: place block");
    log::info!("  --- Shared ---");
    log::info!("  F5: pause/resume");
    log::info!("  S: single step");
    log::info!("  R: reset terrain (heightmap)");
    log::info!("  B: reset spectacle gallery");
    log::info!("  M: reset gyroid megastructure");
    log::info!("  N: lattice walk-through demo");
    log::info!("  [/] DEV previous/next lattice jump (orbit mode)");
    log::info!("  U/I/O: DEV intro/interior/reveal lattice jumps (orbit mode)");
    log::info!("  V: DEV panoramic lattice reveal jump (orbit mode)");
    log::info!("  G: reset to legacy GoL sphere seed");
    log::info!("  1-4: switch rules (amoeba, crystal, 445, pyroclastic)");
    log::info!("  P: dump perf + memory summary (on demand)");
    log::info!("  Esc: quit");

    let volume_size = std::env::args()
        .nth(1)
        .map(|s| {
            let n: u32 = s
                .parse()
                .expect("usage: hash-thing [SIZE]  (SIZE must be a power of 2)");
            assert!(
                n.is_power_of_two(),
                "volume size must be a power of 2 (got {n})"
            );
            n
        })
        .unwrap_or(DEFAULT_VOLUME_SIZE);
    log::info!(
        "Volume: {volume_size}^3 (level {})",
        volume_size.trailing_zeros()
    );

    let mut event_loop_builder = EventLoop::builder();
    #[cfg(target_os = "macos")]
    {
        // Make launch behavior explicit instead of depending on bundle/agent defaults.
        event_loop_builder.with_activation_policy(ActivationPolicy::Regular);
        event_loop_builder.with_activate_ignoring_other_apps(true);
    }
    let event_loop = event_loop_builder
        .build()
        .expect("failed to create event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new(volume_size);
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use hash_thing::terrain::materials::{FIRE_MATERIAL_ID, VINE_MATERIAL_ID};
    use std::time::Duration;

    #[test]
    fn reconcile_modifier_keys_drops_only_cleared_bits() {
        use winit::keyboard::ModifiersState;

        let pre_populate = || -> HashSet<KeyCode> {
            [
                KeyCode::ShiftLeft,
                KeyCode::ShiftRight,
                KeyCode::ControlLeft,
                KeyCode::ControlRight,
                KeyCode::AltLeft,
                KeyCode::AltRight,
                KeyCode::SuperLeft,
                KeyCode::SuperRight,
                KeyCode::KeyW, // sentinel: must never be touched
            ]
            .into_iter()
            .collect()
        };

        // All modifier bits clear → every modifier KeyCode drops; KeyW stays.
        let mut keys = pre_populate();
        let removed = reconcile_modifier_keys(ModifiersState::empty(), &mut keys);
        assert_eq!(removed, 8);
        assert!(keys.contains(&KeyCode::KeyW));
        assert!(!keys.contains(&KeyCode::ShiftLeft));
        assert!(!keys.contains(&KeyCode::SuperRight));

        // SHIFT bit held → both Shift KeyCodes stay, others drop.
        let mut keys = pre_populate();
        let removed = reconcile_modifier_keys(ModifiersState::SHIFT, &mut keys);
        assert_eq!(removed, 6);
        assert!(keys.contains(&KeyCode::ShiftLeft));
        assert!(keys.contains(&KeyCode::ShiftRight));
        assert!(!keys.contains(&KeyCode::ControlLeft));
        assert!(!keys.contains(&KeyCode::AltRight));

        // CONTROL | ALT held → 4 modifier KeyCodes drop.
        let mut keys = pre_populate();
        let removed =
            reconcile_modifier_keys(ModifiersState::CONTROL | ModifiersState::ALT, &mut keys);
        assert_eq!(removed, 4);
        assert!(keys.contains(&KeyCode::ControlLeft));
        assert!(keys.contains(&KeyCode::AltRight));
        assert!(!keys.contains(&KeyCode::ShiftLeft));
        assert!(!keys.contains(&KeyCode::SuperLeft));

        // All four bits held → nothing removed.
        let all = ModifiersState::SHIFT
            | ModifiersState::CONTROL
            | ModifiersState::ALT
            | ModifiersState::SUPER;
        let mut keys = pre_populate();
        let removed = reconcile_modifier_keys(all, &mut keys);
        assert_eq!(removed, 0);
        assert_eq!(keys.len(), 9);

        // Empty set + all bits clear → no-op, no underflow.
        let mut keys: HashSet<KeyCode> = HashSet::new();
        let removed = reconcile_modifier_keys(ModifiersState::empty(), &mut keys);
        assert_eq!(removed, 0);
    }

    #[test]
    fn lattice_demo_waypoints_stay_inside_world() {
        for beat in [
            LatticeDemoBeat::Intro,
            LatticeDemoBeat::Interior,
            LatticeDemoBeat::Panorama,
        ] {
            let waypoint = lattice_demo_waypoint(2048, beat);
            for axis in [0, 1, 2] {
                assert!(waypoint.player.pos[axis] > 0.0);
                assert!(waypoint.player.pos[axis] < 2048.0);
            }
        }
    }

    #[test]
    fn lattice_panorama_waypoint_faces_inward_and_upward() {
        let waypoint = lattice_demo_waypoint(512, LatticeDemoBeat::Panorama);
        let (_eye, dir) = player::eye_ray(
            &waypoint.player.pos,
            waypoint.player.yaw,
            waypoint.player.pitch,
        );
        assert!(dir[0] < 0.0);
        assert!(dir[1] > 0.0);
        assert!(dir[2] < 0.0);
    }

    #[test]
    fn lattice_demo_waypoints_are_distinct() {
        let intro = lattice_demo_waypoint(512, LatticeDemoBeat::Intro);
        let interior = lattice_demo_waypoint(512, LatticeDemoBeat::Interior);
        let panorama = lattice_demo_waypoint(512, LatticeDemoBeat::Panorama);
        assert_ne!(intro.label, interior.label);
        assert_ne!(interior.label, panorama.label);
        assert_ne!(intro.player.pos, panorama.player.pos);
    }

    #[test]
    fn lattice_demo_beats_cycle_in_story_order() {
        assert_eq!(LatticeDemoBeat::Intro.next(), LatticeDemoBeat::Interior);
        assert_eq!(LatticeDemoBeat::Interior.next(), LatticeDemoBeat::Panorama);
        assert_eq!(LatticeDemoBeat::Panorama.next(), LatticeDemoBeat::Intro);
        assert_eq!(LatticeDemoBeat::Intro.previous(), LatticeDemoBeat::Panorama);
        assert_eq!(
            LatticeDemoBeat::Panorama.previous(),
            LatticeDemoBeat::Interior
        );
    }

    #[test]
    fn lattice_demo_beats_have_stable_labels() {
        assert_eq!(LatticeDemoBeat::Intro.label(), "intro");
        assert_eq!(LatticeDemoBeat::Interior.label(), "interior");
        assert_eq!(LatticeDemoBeat::Panorama.label(), "panorama");
    }

    #[test]
    fn lattice_short_demo_cut_advances_in_story_order() {
        use std::time::Duration;

        assert_eq!(
            LatticeShortDemoCut::beat_for_elapsed(Duration::from_secs_f32(0.0)),
            Some(LatticeDemoBeat::Intro)
        );
        assert_eq!(
            LatticeShortDemoCut::beat_for_elapsed(Duration::from_secs_f32(2.1)),
            Some(LatticeDemoBeat::Interior)
        );
        assert_eq!(
            LatticeShortDemoCut::beat_for_elapsed(Duration::from_secs_f32(5.0)),
            Some(LatticeDemoBeat::Panorama)
        );
        assert_eq!(
            LatticeShortDemoCut::beat_for_elapsed(Duration::from_secs_f32(8.1)),
            None
        );
    }

    #[test]
    fn app_new_defers_initial_scene_generation_until_window_loop() {
        let app = App::new(256);
        assert!(app.startup_scene_pending);
        assert_eq!(app.world.population(), 0);
    }

    #[test]
    fn collect_visible_particle_data_culls_entities_behind_walls() {
        let mut world = sim::World::new(3);
        let wall = hash_thing::octree::Cell::pack(1, 0).raw();
        world.set(
            sim::WorldCoord(3),
            sim::WorldCoord(1),
            sim::WorldCoord(1),
            wall,
        );

        let mut entities = sim::EntityStore::new();
        entities.add(
            [2.5, 1.5, 1.5],
            [0.0; 3],
            sim::EntityKind::Critter(sim::CritterState::new(VINE_MATERIAL_ID)),
        );
        entities.add(
            [5.5, 1.5, 1.5],
            [0.0; 3],
            sim::EntityKind::Particle(sim::ParticleState {
                material: FIRE_MATERIAL_ID,
                ttl: 5,
                on_despawn: None,
            }),
        );

        let visible = collect_visible_particle_data(
            &world,
            &entities,
            [1.5, 1.5, 1.5],
            [0, 0, 0],
            1.0 / world.side() as f32,
        );

        assert_eq!(
            visible.len(),
            1,
            "only the unoccluded critter should upload"
        );
        assert_eq!(visible[0][3].to_bits(), VINE_MATERIAL_ID as u32);
        assert!((visible[0][0] - 2.5 / world.side() as f32).abs() < 1e-6);
    }

    #[test]
    fn first_person_legend_hides_lattice_debug_jumps() {
        let lines = App::legend_lines(CameraMode::FirstPerson);
        assert!(lines.iter().any(|line| line.contains("Space       Leap")));
        assert!(lines.iter().any(|line| line.contains("Scroll/1-9  Matter")));
        assert!(!lines.iter().any(|line| line.contains("Fly up")));
        assert!(!lines.iter().any(|line| line.contains("Fly down")));
        assert!(!lines.iter().any(|line| line.contains("DEV prev/next jump")));
        assert!(!lines
            .iter()
            .any(|line| line.contains("DEV intro/interior/reveal")));
        assert!(!lines.iter().any(|line| line.contains("DEV tweet reveal")));
        assert!(lines.iter().any(|line| line.contains("Lattice walk")));
    }

    #[test]
    fn orbit_legend_marks_lattice_jumps_as_debug() {
        let lines = App::legend_lines(CameraMode::Orbit);
        assert!(lines.iter().any(|line| line.contains("DEV prev/next jump")));
        assert!(lines
            .iter()
            .any(|line| line.contains("DEV intro/interior/reveal")));
        assert!(lines.iter().any(|line| line.contains("DEV tweet reveal")));
        assert!(lines.iter().any(|line| line.contains("Lattice walk")));
    }

    #[test]
    fn legend_defaults_follow_camera_mode() {
        assert!(!default_legend_visibility(CameraMode::FirstPerson));
        assert!(default_legend_visibility(CameraMode::Orbit));
    }

    #[test]
    fn demo_reset_restores_active_materials_near_each_waypoint() {
        let demo_size = 64;
        let mut app = App::new(demo_size);
        app.world = sim::World::new(demo_size.trailing_zeros());

        app.load_demo_spectacle("test reset");

        for waypoint in app.world.demo_waypoints() {
            assert!(
                app.world
                    .count_active_material_cells_near(waypoint.center, waypoint.radius)
                    > 0,
                "reset should restore spectacle at waypoint {}",
                waypoint.label
            );
        }
    }

    #[test]
    fn recenter_player_respects_shifted_world_origin() {
        let mut app = App::new(8);
        app.world.ensure_region(
            [sim::WorldCoord(-4), sim::WorldCoord(0), sim::WorldCoord(0)],
            [sim::WorldCoord(7), sim::WorldCoord(7), sim::WorldCoord(7)],
        );
        assert!(
            app.world.origin[0] < 0,
            "test setup should force a negative origin shift"
        );

        assert!(app.recenter_player());

        let pid = app.player_id.expect("player should exist");
        let player = app
            .entities
            .iter()
            .find(|entity| entity.id == pid)
            .expect("player entity should still exist");
        let center = app.world_center();
        assert_eq!(player.pos, [center[0], center[1] + 2.0, center[2]]);
    }

    #[test]
    fn background_step_starts_after_live_world_player_update() {
        let mut app = App::new(8);
        app.paused = false;
        app.keys_held.insert(KeyCode::KeyW);
        let stone = hash_thing::octree::Cell::pack(1, 0).raw();
        for x in 3..=5 {
            for z in 3..=5 {
                app.world.set(
                    sim::WorldCoord(x),
                    sim::WorldCoord(0),
                    sim::WorldCoord(z),
                    stone,
                );
            }
        }
        app.world.set(
            sim::WorldCoord(4),
            sim::WorldCoord(1),
            sim::WorldCoord(3),
            stone,
        );
        app.world.set(
            sim::WorldCoord(4),
            sim::WorldCoord(2),
            sim::WorldCoord(3),
            stone,
        );
        app.reset_player_pose([4.5, 1.0, 4.5], 0.0, 0.0);

        let pid = app.player_id.expect("player should exist");
        let speed = PLAYER_SPEED;
        let step = player::step_grounded_movement(
            &app.world,
            &[4.5, 1.0, 4.5],
            0.0,
            player::GroundedMoveInput {
                yaw: 0.0,
                move_input: [1.0, 0.0],
                speed,
                dt: 0.1,
                jump_requested: false,
            },
        );
        let expected_z = step.pos[2];

        let yaw = app
            .entities
            .get_mut(pid)
            .and_then(|e| match &e.kind {
                sim::EntityKind::Player(ps) => Some(ps.yaw),
                _ => None,
            })
            .unwrap_or(0.0);
        let delta = player::compute_move_delta(yaw, [1.0, 0.0, 0.0], speed, 0.1);
        assert!(
            4.5 + delta[2] < expected_z,
            "free-fly would move farther through the wall than grounded movement"
        );

        if let Some(player) = app.entities.get_mut(pid) {
            player.pos = step.pos;
        }
        app.maybe_start_background_step();

        let player = app
            .entities
            .iter()
            .find(|entity| entity.id == pid)
            .expect("player entity should still exist");
        assert_eq!(player.pos[2], expected_z);
        assert!(app.is_stepping(), "step should start after player update");

        let handle = app.step_handle.take().expect("step handle should exist");
        app.world = handle
            .join()
            .expect("step thread should not panic")
            .expect("step thread should return a world");
    }

    #[test]
    fn slow_dev_step_warning_requires_debug_large_world_and_slow_step() {
        assert!(should_warn_about_slow_dev_step(
            true,
            256,
            Duration::from_millis(DEV_PROFILE_STEP_WARN_MS)
        ));
        assert!(!should_warn_about_slow_dev_step(
            false,
            256,
            Duration::from_secs(5)
        ));
        assert!(!should_warn_about_slow_dev_step(
            true,
            128,
            Duration::from_secs(5)
        ));
        assert!(!should_warn_about_slow_dev_step(
            true,
            256,
            Duration::from_millis(DEV_PROFILE_STEP_WARN_MS - 1)
        ));
    }

    #[test]
    fn cursor_capture_requires_first_person_and_focus() {
        assert!(should_capture_cursor(CameraMode::FirstPerson, true));
        assert!(!should_capture_cursor(CameraMode::FirstPerson, false));
        assert!(!should_capture_cursor(CameraMode::Orbit, true));
        assert!(!should_capture_cursor(CameraMode::Orbit, false));
    }

    /// Read the player's `(yaw, pitch)` off the entity store, for tests
    /// that exercise the mouse-look split (hash-thing-a6t2).
    fn player_look(app: &mut App) -> (f64, f64) {
        let pid = app.player_id.expect("player spawned by App::new");
        let entity = app.entities.get_mut(pid).expect("player entity present");
        match &entity.kind {
            sim::EntityKind::Player(ps) => (ps.yaw, ps.pitch),
            _ => panic!("player_id does not point at a Player entity"),
        }
    }

    /// Seed yaw/pitch and `last_mouse` so `handle_cursor_moved` produces a
    /// non-zero delta on its first call.
    fn prime_cursor(app: &mut App, start_x: f64, start_y: f64) {
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);
        app.last_mouse = Some((start_x, start_y));
    }

    #[test]
    fn cursor_moved_first_person_uncaptured_applies_look() {
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.cursor_captured = false;
        prime_cursor(&mut app, 100.0, 100.0);

        app.handle_cursor_moved(110.0, 105.0);

        let (yaw, pitch) = player_look(&mut app);
        assert!((yaw - 10.0 * LOOK_SENSITIVITY).abs() < 1e-12);
        assert!((pitch - 5.0 * LOOK_SENSITIVITY).abs() < 1e-12);
    }

    #[test]
    fn cursor_moved_first_person_captured_is_noop() {
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.cursor_captured = true;
        prime_cursor(&mut app, 100.0, 100.0);

        app.handle_cursor_moved(999.0, 999.0);

        let (yaw, pitch) = player_look(&mut app);
        assert_eq!(yaw, 0.0);
        assert_eq!(pitch, 0.0);
    }

    #[test]
    fn mouse_motion_captured_first_person_applies_look() {
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.cursor_captured = true;
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);

        app.handle_mouse_motion(20.0, -8.0);

        let (yaw, pitch) = player_look(&mut app);
        assert!((yaw - 20.0 * LOOK_SENSITIVITY).abs() < 1e-12);
        assert!((pitch - (-8.0 * LOOK_SENSITIVITY)).abs() < 1e-12);
    }

    #[test]
    fn mouse_motion_uncaptured_is_noop() {
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.cursor_captured = false;
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);

        app.handle_mouse_motion(999.0, 999.0);

        let (yaw, pitch) = player_look(&mut app);
        assert_eq!(yaw, 0.0);
        assert_eq!(pitch, 0.0);
    }

    #[test]
    fn apply_fps_look_clamps_pitch() {
        let mut app = App::new(64);
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 1.3);

        // dy large enough to push past the +1.4 clamp.
        app.apply_fps_look(0.0, 1000.0);
        let (_yaw, pitch) = player_look(&mut app);
        assert!((pitch - 1.4).abs() < 1e-12, "pitch should clamp at +1.4");

        // Symmetric negative clamp.
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, -1.3);
        app.apply_fps_look(0.0, -1000.0);
        let (_yaw, pitch) = player_look(&mut app);
        assert!((pitch - -1.4).abs() < 1e-12, "pitch should clamp at -1.4");
    }
}
