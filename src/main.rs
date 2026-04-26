use hash_thing::perf;
use hash_thing::player;
use hash_thing::render;
use hash_thing::scale::{CELLS_PER_METER, DEFAULT_VOLUME_SIZE, GROWTH_MARGIN};
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

/// Wall-clock cadence for the consolidated perf log line. Decoupled from
/// `world.generation` so the log keeps ticking even when the sim is paused
/// or stepping slowly — see hash-thing-q63.
const LOG_INTERVAL_SECS: f64 = 2.0;
const DEV_PROFILE_STEP_WARN_MS: u64 = 500;

/// Minimum interval between `window.set_title` calls. 250 ms = 4 Hz,
/// the threshold at which a human reads a changing number without
/// jitter (hash-thing-4ioh).
const TITLE_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// Bound on consecutive failed FPS cursor-grab attempts before the
/// per-frame retry tap goes dormant (hash-thing-ezx8). 600 frames
/// ~= 10 s at 60 Hz: long enough to ride out the slow-startup
/// activation window lke9 fixed (typically a few frames), short
/// enough that 5 syscalls/frame * 600 = ~3000 calls is a hard
/// ceiling on permanently-unsupported platforms (headless, broken
/// X11). Reset by any successful grab or by an explicit release
/// (camera-mode toggle, focus loss, occlusion).
const MAX_CURSOR_GRAB_RETRIES: u32 = 600;

/// Thin wrapper over macOS `pthread_set_qos_class_self_np` used as the
/// xhi6 diagnostic knob (proxy 2 for SVDAG↔sim cache-locality work).
/// `parse` maps the `HASH_THING_SIM_QOS` env string to a `qos_class_t`
/// constant; `apply` installs it on the *current* thread. No-ops on
/// non-macOS targets so the cross-compile stays green.
mod thread_qos {
    #[cfg(target_os = "macos")]
    mod imp {
        pub const USER_INTERACTIVE: u32 = 0x21;
        pub const USER_INITIATED: u32 = 0x19;
        pub const DEFAULT: u32 = 0x15;
        pub const UTILITY: u32 = 0x11;
        pub const BACKGROUND: u32 = 0x09;

        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }

        pub fn apply(qos: u32) {
            // SAFETY: Apple-documented libc entrypoint. Both inputs are
            // ABI-compatible primitives; no memory is exchanged.
            let rc = unsafe { pthread_set_qos_class_self_np(qos, 0) };
            if rc == 0 {
                log::info!("sim thread QoS set to 0x{qos:02x}");
            } else {
                log::warn!("pthread_set_qos_class_self_np failed: rc={rc} qos=0x{qos:02x}");
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    mod imp {
        pub const USER_INTERACTIVE: u32 = 1;
        pub const USER_INITIATED: u32 = 2;
        pub const DEFAULT: u32 = 3;
        pub const UTILITY: u32 = 4;
        pub const BACKGROUND: u32 = 5;

        pub fn apply(_qos: u32) {
            log::warn!("HASH_THING_SIM_QOS set but target is not macOS; ignoring");
        }
    }

    pub fn parse(s: &str) -> Option<u32> {
        match s {
            "interactive" => Some(imp::USER_INTERACTIVE),
            "initiated" => Some(imp::USER_INITIATED),
            "default" => Some(imp::DEFAULT),
            "utility" => Some(imp::UTILITY),
            "background" => Some(imp::BACKGROUND),
            _ => None,
        }
    }

    pub fn apply(qos: u32) {
        imp::apply(qos);
    }
}

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

/// Scene-swap request queued while a background sim step is in flight.
///
/// Keeps `pending_scene_swap` separate from `pending_player_action`: player
/// actions are FIFO lightweight edits against the next live world, while
/// scene swaps replace the world/entities/render state wholesale. Last
/// write wins: pressing another scene-selection key (`g`, `m`, `r`, `t`,
/// `b`, another `n`/`v`, or a lattice debug jump `[`/`]`/`u`/`i`/`o`)
/// clears the queue *at key-press time*, even when the new loader
/// early-returns under `is_stepping()`. That way a stale queued swap
/// can't land after the user has already picked something else. If both
/// a player action and a scene swap are queued when the step finishes,
/// the scene swap wins and the player action is dropped — the action
/// was queued against a world we're about to discard, so replaying it
/// would just do wasted upload work.
///
/// `SelectLatticeBeat` carries a pre-resolved target beat (computed at
/// press-time from `current_demo_beat` for `[`/`]`). Resolving at press
/// time keeps the cycle-direction intent stable across overwrites
/// (hash-thing-5a5a plan review), so pressing `]` then another
/// scene-swap key doesn't silently re-resolve the target against an
/// unintended `current_demo_beat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingSceneSwap {
    LoadLatticeDemo,
    LoadLatticePanoramaDemo,
    ResetTerrain,
    LoadGyroid,
    LoadDemoSpectacle,
    ResetGolSmoke,
    SelectLatticeBeat(LatticeDemoBeat),
    SelectRule(sim::GameOfLife3D, &'static str),
}

impl PendingSceneSwap {
    fn label(self) -> &'static str {
        match self {
            Self::LoadLatticeDemo => "lattice_demo",
            Self::LoadLatticePanoramaDemo => "lattice_panorama",
            Self::ResetTerrain => "terrain_reset",
            Self::LoadGyroid => "gyroid",
            Self::LoadDemoSpectacle => "demo_spectacle",
            Self::ResetGolSmoke => "gol_smoke_reset",
            Self::SelectLatticeBeat(LatticeDemoBeat::Intro) => "lattice_beat_intro",
            Self::SelectLatticeBeat(LatticeDemoBeat::Interior) => "lattice_beat_interior",
            Self::SelectLatticeBeat(LatticeDemoBeat::Panorama) => "lattice_beat_panorama",
            Self::SelectRule(_, _) => "select_rule",
        }
    }

    /// True if dispatching this swap replaces `world`, making any
    /// concurrently-queued player action stale (it was queued against the
    /// world we're about to throw out — see a9jd / drain comment). False
    /// when the swap only re-keys auxiliary state on the existing world.
    fn discards_world(self) -> bool {
        match self {
            Self::LoadLatticeDemo
            | Self::LoadLatticePanoramaDemo
            | Self::ResetTerrain
            | Self::LoadGyroid
            | Self::LoadDemoSpectacle
            | Self::ResetGolSmoke
            | Self::SelectLatticeBeat(_) => true,
            Self::SelectRule(_, _) => false,
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BeatAdvance {
    /// Advance to a different beat this frame.
    Set(LatticeDemoBeat),
    /// Cut has finished; clear `short_demo_cut`.
    End,
    /// Stay on the current beat.
    Hold,
}

impl LatticeShortDemoCut {
    const INTERIOR_AT: f32 = 2.0;
    const PANORAMA_AT: f32 = 4.5;
    const END_AT: f32 = 8.0;

    /// Decide the next beat given the current beat and wall-clock elapsed.
    /// Advances at most one step per call so a long frame (OS pause,
    /// long sim step, heavy SVDAG upload) cannot jump Intro → Panorama
    /// and skip Interior entirely (hash-thing-x1lg). After a stall, the
    /// next redraw walks forward one step; the frame after that walks the
    /// remaining step, so each beat is displayed for at least one frame.
    fn advance(current: Option<LatticeDemoBeat>, elapsed: std::time::Duration) -> BeatAdvance {
        let secs = elapsed.as_secs_f32();
        match current {
            None => BeatAdvance::Set(LatticeDemoBeat::Intro),
            Some(LatticeDemoBeat::Intro) if secs >= Self::INTERIOR_AT => {
                BeatAdvance::Set(LatticeDemoBeat::Interior)
            }
            Some(LatticeDemoBeat::Interior) if secs >= Self::PANORAMA_AT => {
                BeatAdvance::Set(LatticeDemoBeat::Panorama)
            }
            Some(LatticeDemoBeat::Panorama) if secs >= Self::END_AT => BeatAdvance::End,
            _ => BeatAdvance::Hold,
        }
    }
}

/// Map a `[`/`]`/`u`/`i`/`o` lattice-debug key to its target beat
/// (hash-thing-5a5a). `[` and `]` resolve relative to `current` with
/// `unwrap_or(Intro)`; `u`/`i`/`o` map to their literal beat. Pulled
/// out of the key handler so press-time resolution is unit-testable
/// without stubbing the background-step machinery.
///
/// Caller must already have gated on `lattice_debug_jumps_enabled`
/// and restricted `key` to one of the five recognized characters; any
/// other value trips `unreachable!()`.
fn resolve_lattice_beat_target(current: Option<LatticeDemoBeat>, key: &str) -> LatticeDemoBeat {
    let anchor = current.unwrap_or(LatticeDemoBeat::Intro);
    match key {
        "[" => anchor.previous(),
        "]" => anchor.next(),
        "u" => LatticeDemoBeat::Intro,
        "i" => LatticeDemoBeat::Interior,
        "o" => LatticeDemoBeat::Panorama,
        _ => unreachable!("resolve_lattice_beat_target: unexpected key {key:?}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DemoWaypoint {
    label: &'static str,
    player: PlayerPose,
    camera: OrbitCameraPose,
}

fn lattice_demo_waypoint(
    side: usize,
    layout: &sim::DemoLayout,
    beat: LatticeDemoBeat,
) -> DemoWaypoint {
    // ptew: derive player positions from the actual scene anchors so the
    // spawn is always in carved-AIR cells with scene-provided floor
    // support. The previous implementation used hardcoded `side * k`
    // fractions that drifted deep inside lattice pillars whenever scene
    // geometry evolved.
    let inv_side = 1.0 / side as f32;
    let center_xz =
        |c: [i64; 3]| -> [f64; 3] { [c[0] as f64 + 0.5, c[1] as f64, c[2] as f64 + 0.5] };
    let to_normalized = |p: [f64; 3]| -> [f32; 3] {
        [
            p[0] as f32 * inv_side,
            p[1] as f32 * inv_side,
            p[2] as f32 * inv_side,
        ]
    };
    match beat {
        LatticeDemoBeat::Intro => DemoWaypoint {
            label: "intro",
            player: PlayerPose {
                pos: layout.player_pos,
                yaw: layout.player_yaw,
                pitch: layout.player_pitch,
            },
            camera: OrbitCameraPose {
                target: to_normalized(layout.player_pos),
                yaw: -3.0 * std::f32::consts::FRAC_PI_4,
                pitch: 0.08,
                dist: 0.34,
            },
        },
        LatticeDemoBeat::Interior => {
            let pos = center_xz(layout.atrium_center);
            DemoWaypoint {
                label: "interior",
                player: PlayerPose {
                    pos,
                    yaw: std::f64::consts::FRAC_PI_2,
                    pitch: -0.12,
                },
                camera: OrbitCameraPose {
                    target: to_normalized(pos),
                    yaw: std::f32::consts::FRAC_PI_2,
                    pitch: 0.10,
                    dist: 0.22,
                },
            }
        }
        LatticeDemoBeat::Panorama => {
            let pos = center_xz(layout.reveal_center);
            // Aim the orbit camera at the midpoint between the player on the
            // balcony and the panorama volume so the reveal frames both.
            let panorama_mid: [f64; 3] = [
                (layout.reveal_center[0] + layout.panorama_center[0]) as f64 * 0.5,
                (layout.reveal_center[1] + layout.panorama_center[1]) as f64 * 0.5,
                (layout.reveal_center[2] + layout.panorama_center[2]) as f64 * 0.5,
            ];
            DemoWaypoint {
                label: "panorama",
                player: PlayerPose {
                    pos,
                    yaw: std::f64::consts::FRAC_PI_4,
                    pitch: 0.24,
                },
                camera: OrbitCameraPose {
                    target: to_normalized(panorama_mid),
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
    /// EWMA-smoothed FPS for the window title. Instantaneous `1/dt`
    /// jumps ±5 between frames at 20-30 FPS; the smoothed readout is
    /// just for the human skimming the title bar (hash-thing-d9af).
    /// Zero sentinel: first frame seeds the filter at its instant value.
    smoothed_fps: f64,
    /// Set by reset sites that `last_frame = Instant::now()` outside the
    /// redraw hot path (focus-gain, unocclude). The next RedrawRequested
    /// sees a `dt_wall` that measures the event-dispatch-to-redraw gap
    /// (microseconds), not a real frame; blending `1/dt_wall` into the
    /// EWMA would spike the displayed FPS to millions and decay it over
    /// many seconds. Consumed (cleared) on the first redraw after the
    /// reset so subsequent frames measure normally (hash-thing-dxjb;
    /// same class as hash-thing-6e4a, cold-startup path).
    suppress_next_fps_sample: bool,
    /// Throttle for `window.set_title` — at 60 FPS an unthrottled
    /// title update is unreadable even with EWMA smoothing, and the
    /// OS sees 60 title rewrites per second. Rewrite at ~4 Hz
    /// (hash-thing-4ioh). `None` before the first title update.
    last_title_update: Option<std::time::Instant>,
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
    /// Latches "the most recent grab attempt failed" so the per-frame
    /// retry tap (RedrawRequested) can re-attempt without spamming
    /// `log::warn!` on every frame for platforms where grab is
    /// permanently unsupported. Cleared by any successful grab or by
    /// an explicit release; one warn per failure streak.
    /// hash-thing-lke9.
    cursor_capture_grab_warned: bool,
    /// Counts consecutive failed grab attempts. When it reaches
    /// `MAX_CURSOR_GRAB_RETRIES`, `sync_cursor_capture` stops calling
    /// `apply_cursor_capture(true)` so platforms where grab is
    /// permanently unsupported don't burn syscalls every frame
    /// indefinitely. Reset on success (`record_grab_outcome(true)`)
    /// and on release (`apply_cursor_capture(false)`); legitimate
    /// state transitions (camera-mode, focus, occlusion) all route
    /// through the release path so they re-arm a fresh attempt.
    /// hash-thing-ezx8.
    cursor_capture_grab_failures: u32,
    /// Replay FPS interactions on the next live-world frame instead of
    /// dropping them while a background step is in flight.
    pending_player_action: Option<PendingPlayerAction>,
    /// Defer a scene-swap requested while stepping; drained after step
    /// completion in the same slot as `pending_player_action`. Last-write
    /// wins; any unrelated scene-change key clears it. (hash-thing-a9jd)
    pending_scene_swap: Option<PendingSceneSwap>,
    /// Latest modifier key state, tracked via ModifiersChanged. Feeds
    /// `reconcile_modifier_keys` to clear stuck physical-key entries
    /// when a modifier change arrives without a matching Released event.
    modifiers: winit::keyboard::ModifiersState,
    /// Cached hashlife memo health summary (hash-thing-stue.6). Refreshed
    /// on the main thread after each step's world result is merged back,
    /// so the periodic log can print it even while the next step is on
    /// the background thread (where `self.world` is a placeholder).
    last_memo_summary: String,
    /// Memo HUD overlay toggle (hash-thing-nhwo). On when
    /// `HASH_THING_MEMO_HUD=1` is set at startup; renders memo_* stats
    /// as a top-left text panel. No key-binding — env-var only.
    memo_hud_visible: bool,
    /// True when the memo HUD texture needs re-upload (each time
    /// `last_memo_summary` refreshes or at seed).
    memo_hud_dirty: bool,
    /// Sim-freeze diagnostic toggle (hash-thing-stue.7). When true, the
    /// background-step dispatch is a no-op: world stays put on the main
    /// thread, generation never advances, and the renderer reads a stable
    /// SVDAG every frame. Used to measure renderer-only frame cost without
    /// sim/render unified-memory contention. Set via `HASH_THING_FREEZE_SIM=1`.
    freeze_sim: bool,
    /// macOS QoS class to apply to the background sim thread (hash-thing-xhi6
    /// proxy 2). `Some` → the thread calls `pthread_set_qos_class_self_np`
    /// at entry. `USER_INTERACTIVE` keeps sim in the P-core pool alongside
    /// the main render thread; `BACKGROUND` steers it toward E-cores.
    /// Comparing the two isolates CPU-cycle contention (same pool = slower)
    /// from unified-memory cache contention (no change either way). `None`
    /// leaves the thread at its inherited QoS. Set via
    /// `HASH_THING_SIM_QOS={interactive|initiated|default|utility|background}`.
    sim_qos: Option<u32>,
    /// Read-only voxel-grid view kept on the main thread so player collision
    /// can run every frame even while a background sim step owns the mutable
    /// [`sim::World`] (hash-thing-0s9v). Refreshed at step start, step
    /// completion, and after [`sim::World::ensure_region`] grows the world.
    collision_snapshot: Option<sim::CollisionSnapshot>,
    /// Cached fullscreen state for auto-clearing perf histograms on
    /// windowed↔fullscreen transitions (hash-thing-0zrc). Queried each
    /// `Resized` against `window.fullscreen().is_some()`; a mismatch
    /// means the OS just toggled fullscreen, which matters for dlse.2.2
    /// A/B perf comparisons.
    was_fullscreen: bool,
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

/// EWMA smoother for the window-title FPS readout (hash-thing-d9af).
/// `prev <= 0.0` is the first-frame sentinel — seed the filter at the
/// instant value so the title doesn't crawl up from zero.
fn smooth_fps(prev: f64, instant: f64, alpha: f64) -> f64 {
    if prev <= 0.0 {
        instant
    } else {
        alpha * instant + (1.0 - alpha) * prev
    }
}

/// Split the per-frame elapsed time into `(dt_wall, dt_clamped)`.
///
/// The clamped value keeps xa7 player movement bounded under a hiccup
/// (pos += vel * dt); the wall value feeds the d9af EWMA title readout
/// so real sub-10-FPS frames aren't capped at "10 FPS" (hash-thing-dvzl).
fn compute_frame_dts(elapsed: std::time::Duration) -> (f64, f64) {
    let dt_wall = elapsed.as_secs_f64();
    let dt_clamped = dt_wall.min(0.1);
    (dt_wall, dt_clamped)
}

/// Title-refresh throttle gate (hash-thing-4ioh). Returns `true` on
/// the first call (`last` is `None`) and on any call at least
/// `TITLE_REFRESH_INTERVAL` past the previous refresh.
fn should_refresh_title(last: Option<std::time::Instant>, now: std::time::Instant) -> bool {
    match last {
        None => true,
        Some(t) => now.duration_since(t) >= TITLE_REFRESH_INTERVAL,
    }
}

fn default_legend_visibility(_mode: CameraMode) -> bool {
    // Default-on until proper demo lineup lands (edward 2026-04-21).
    true
}

fn should_capture_cursor(camera_mode: CameraMode, focused: bool, occluded: bool) -> bool {
    // Occlusion (e.g. macOS Mission Control) can revoke the OS-level grab
    // without a Focused(false) event — hash-thing-w0o9. Treat it as
    // defocus-equivalent so cursor_captured cannot outlive the real grab.
    focused && !occluded && camera_mode == CameraMode::FirstPerson
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
            smoothed_fps: 0.0,
            suppress_next_fps_sample: false,
            // None so the first frame refreshes the title immediately
            // — without an `Instant` sentinel before the monotonic
            // epoch, which could underflow on some platforms.
            last_title_update: None,
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
            cursor_capture_grab_warned: false,
            cursor_capture_grab_failures: 0,
            pending_player_action: None,
            pending_scene_swap: None,
            modifiers: winit::keyboard::ModifiersState::empty(),
            last_memo_summary: String::new(),
            memo_hud_visible: std::env::var("HASH_THING_MEMO_HUD").ok().as_deref() == Some("1"),
            memo_hud_dirty: true,
            freeze_sim: std::env::var("HASH_THING_FREEZE_SIM").ok().as_deref() == Some("1"),
            sim_qos: std::env::var("HASH_THING_SIM_QOS")
                .ok()
                .as_deref()
                .and_then(thread_qos::parse),
            collision_snapshot: None,
            was_fullscreen: false,
        };
        if app.freeze_sim {
            log::info!("HASH_THING_FREEZE_SIM=1: sim step disabled (stue.7 diagnostic)");
        }
        if let Some(qos) = app.sim_qos {
            log::info!(
                "HASH_THING_SIM_QOS active: sim thread will request QoS 0x{qos:02x} (xhi6 diagnostic)"
            );
        }

        // Seed the cached memo summary so the very first periodic log line
        // has a populated column instead of a blank trailing field
        // (hash-thing-stue.6 reviewer nit). Dirty the HUD too so reseeding
        // here can never race ahead of the construction-time seed.
        app.last_memo_summary = app.world.memo_summary();
        app.memo_hud_dirty = true;
        let player_pos = app.reset_scene_entities();
        app.spawn_demo_entities();
        log::info!(
            "Player spawned at ({}, {}, {})",
            player_pos[0],
            player_pos[1],
            player_pos[2]
        );
        log::info!(
            "Controls: WASD=move, mouse=look, LMB=break, RMB=place, scroll/1-9=material, F5=pause, Tab=orbit"
        );

        app
    }

    /// Apply a grab attempt's outcome to the cursor-capture state. Pure on
    /// `App` fields; window-side effects (visibility, set_cursor_grab cleanup)
    /// stay with the caller. Returns `true` if a fresh `log::warn!` should
    /// fire for this failure (first failure of a streak), `false` to stay
    /// silent (steady-state failures, or on success).
    ///
    /// Streak contract — hash-thing-lke9:
    /// - success: clear `cursor_capture_grab_warned`. Next failure can warn.
    /// - failure: set the flag; suppress repeat warns until success or
    ///   explicit release clears it.
    ///
    /// Also drives the `cursor_capture_grab_failures` counter
    /// (hash-thing-ezx8): increment on each failure (saturating at
    /// `u32::MAX`), reset to 0 on success. The counter pairs with the
    /// `MAX_CURSOR_GRAB_RETRIES` dormant gate in `sync_cursor_capture`.
    fn record_grab_outcome(&mut self, succeeded: bool) -> bool {
        if succeeded {
            self.cursor_captured = true;
            self.last_mouse = None;
            self.cursor_capture_grab_warned = false;
            self.cursor_capture_grab_failures = 0;
            false
        } else {
            self.cursor_captured = false;
            self.cursor_capture_grab_failures = self.cursor_capture_grab_failures.saturating_add(1);
            let should_warn = !self.cursor_capture_grab_warned;
            self.cursor_capture_grab_warned = true;
            should_warn
        }
    }

    fn apply_cursor_capture(&mut self, capture: bool) {
        if !capture {
            // Release: clear the warn-throttle and the failure counter
            // (hash-thing-ezx8) so a later recapture-then-fail can warn
            // once again and the dormant gate is re-armed. Note that
            // `sync_cursor_capture` also resets the counter on every
            // release-direction call (even when this branch is
            // short-circuited by the state-match early return) — that
            // covers the permanently-failing-platform case where
            // `cursor_captured` was never true. This branch handles the
            // reset for the post-successful-grab release path. Then
            // drop the OS-level grab if we have a window.
            self.cursor_capture_grab_warned = false;
            self.cursor_capture_grab_failures = 0;
            if let Some(window) = self.window.as_ref() {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
            self.cursor_captured = false;
            self.last_mouse = None;
            return;
        }

        // Clone the Arc so the window reference doesn't conflict with the
        // `&mut self` borrow inside `record_grab_outcome`.
        let Some(window) = self.window.clone() else {
            // No window: nothing to grab against. Treat as a soft "couldn't
            // grab" so unit tests can drive the streak lifecycle through
            // App::new (no window). No `log::warn!` here — there is no OS
            // error to surface; the flag move is purely state-tracking.
            let _ = self.record_grab_outcome(false);
            return;
        };

        let grab = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        match grab {
            Ok(()) => {
                window.set_cursor_visible(false);
                let _ = self.record_grab_outcome(true);
            }
            Err(err) => {
                if self.record_grab_outcome(false) {
                    log::warn!("Failed to grab FPS cursor: {err}");
                }
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
        }
    }

    fn sync_cursor_capture(&mut self) {
        let should_capture = should_capture_cursor(self.camera_mode, self.focused, self.occluded);
        // hash-thing-ezx8: any time the predicate says we should NOT
        // capture, clear the failure budget — even when
        // `cursor_captured` was already false (the
        // permanently-failing-platform case where dormancy was reached
        // via 600 failures with no successful grab in between). On
        // those platforms `should_capture==cursor_captured==false`
        // would short-circuit at the state-match early return below,
        // never reaching `apply_cursor_capture(false)`'s reset path,
        // and the user would be pinned dormant forever after the first
        // Tab / focus-loss / occlusion. Reset here covers both the
        // "stale-true cursor_captured" case (handled additionally by
        // the release branch in `apply_cursor_capture`) and the
        // "never-true cursor_captured" case the early return masks.
        if !should_capture {
            self.cursor_capture_grab_failures = 0;
        }
        if should_capture == self.cursor_captured {
            return;
        }
        // Dormant gate: when consecutive grab failures saturate, stop
        // hammering `set_cursor_grab` from the per-frame retry tap.
        // Only gates the *capture* direction — release transitions
        // already reset the counter above and proceed normally.
        if should_capture && self.cursor_capture_grab_failures >= MAX_CURSOR_GRAB_RETRIES {
            return;
        }
        self.apply_cursor_capture(should_capture);
    }

    // Two entry points latch `self.occluded = true`: the winit
    // `WindowEvent::Occluded(true)` event and the `FrameOutcome::Occluded`
    // belt-and-suspenders path in `RedrawRequested`. Both must clear the
    // same transient input state and sync cursor capture or `cursor_captured`
    // drifts from the OS-level grab (hash-thing-w0o9). Extract a helper so
    // the two paths cannot diverge.
    fn enter_occluded_state(&mut self) {
        self.occluded = true;
        self.keys_held.clear();
        self.jump_was_held = false;
        self.last_mouse = None;
        self.sync_cursor_capture();
    }

    fn leave_occluded_state(&mut self) {
        self.occluded = false;
        self.last_mouse = None;
        self.mark_resume_edge();
        self.sync_cursor_capture();
    }

    /// Post-pause resume edge (focus-gain, unocclude). The redraw loop
    /// stops advancing while paused, so the next RedrawRequested would
    /// see `last_frame.elapsed()` clamped to 100 ms — that's a full 100
    /// ms player-movement step on a single frame (hash-thing-xysz) and
    /// a huge dt that decays the EWMA FPS downward for seconds. Reset
    /// `last_frame` to "now" so the next frame measures only the real
    /// gap. BUT the very next RedrawRequested fires microseconds after
    /// this reset, so `dt_wall` becomes the event-dispatch-to-redraw gap
    /// — tiny — and blending `1/dt_wall` into the EWMA would spike the
    /// readout into the millions (hash-thing-dxjb). Pair the reset with
    /// `suppress_next_fps_sample = true` so the next FPS sample is
    /// skipped, not blended. The two operations are one contract; this
    /// helper is the only site that implements it — do not inline it.
    /// All four call sites (focus-gain, unocclude, scene swap, and the
    /// cold-start-post-surface redraw block) share this contract
    /// (hash-thing-v79j, hash-thing-6e4a).
    fn mark_resume_edge(&mut self) {
        self.last_frame = std::time::Instant::now();
        self.suppress_next_fps_sample = true;
    }

    /// Apply a raw pixel-delta mouse look to the player. Scales by
    /// `LOOK_SENSITIVITY`, wraps yaw to `[-π, π)`, and clamps pitch to
    /// ±1.4. Shared between the `CursorMoved` (uncaptured) and
    /// `DeviceEvent::MouseMotion` (captured) paths so the two stay in
    /// parity by construction (hash-thing-a6t2). Wrapping keeps the f32
    /// downcast for rendering precise over arbitrarily long sessions
    /// (hash-thing-u0uf, follow-up to hash-thing-w1yq).
    fn apply_fps_look(&mut self, dx: f64, dy: f64) {
        let Some(pid) = self.player_id else { return };
        let Some(player) = self.entities.get_mut(pid) else {
            return;
        };
        if let sim::EntityKind::Player(ref mut ps) = player.kind {
            let yaw = ps.yaw + dx * LOOK_SENSITIVITY;
            // Centered wrap (around 0, matching reset_player_pose's
            // "straight ahead = 0" convention). rem_euclid would land in
            // [0, τ) instead; [-π, π) keeps sign symmetry with pose code.
            let tau = std::f64::consts::TAU;
            ps.yaw = yaw - tau * ((yaw + std::f64::consts::PI) / tau).floor();
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
    ///
    /// Deltas are intentionally **device-agnostic**: `DeviceId` is not
    /// inspected, so an external mouse and a trackpad driven in the same
    /// frame sum into a single FPS-look delta. A 3D game has no concept
    /// of "which device owns the camera," and winit already delivers one
    /// `MouseMotion` per device per frame, so summing is the natural
    /// behavior (hash-thing-i7gt, follow-up to hash-thing-a6t2).
    fn handle_mouse_motion(&mut self, dx: f64, dy: f64) {
        if self.camera_mode != CameraMode::FirstPerson || !self.cursor_captured {
            return;
        }
        self.apply_fps_look(dx, dy);
    }

    /// Blend `1 / dt_wall` into the EWMA-smoothed FPS readout, or skip
    /// when a reset site has flagged this frame as bogus (hash-thing-dxjb).
    /// A bogus frame is one where `last_frame` was reset outside the
    /// redraw hot path (focus-gain, unocclude) so `dt_wall` is the event-
    /// dispatch-to-redraw gap (microseconds), not a real frame. `dt==0`
    /// (back-to-back redraws within one timer tick) is also skipped to
    /// avoid blending a 0-FPS sample that would silently decay the
    /// readout by 5%.
    fn apply_fps_sample(&mut self, dt_wall: f64) {
        if self.suppress_next_fps_sample {
            self.suppress_next_fps_sample = false;
            return;
        }
        if dt_wall > 0.0 {
            self.smoothed_fps = smooth_fps(self.smoothed_fps, 1.0 / dt_wall, 0.05);
        }
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
        self.reset_scene_perf_state();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.exit_lattice_demo_mode();
        log::info!(
            "Initial scene: terrain pop={} nodes={} gen={}µs",
            self.world.population(),
            self.world.store.stats(),
            stats.gen_region_us,
        );
        log::debug!(
            "Material registry palette slots={}",
            self.world.materials().color_palette_rgba().len()
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
        let blind_pos = [center[0], center[1] + 2.0 * CELLS_PER_METER, center[2]];

        // hash-thing-t3zn.1: blind teleport to `center + 2 cells up` lands
        // inside terrain when the world center is buried. Scan upward at
        // (cx, cz) for the first AABB-clear, grounded pose; on no match the
        // `unwrap_or(blind_pos)` below falls back to the blind pose (covers
        // the empty world — `is_grounded` is false everywhere, so the loop
        // exits without setting `grounded_pos` — and the all-solid column).
        //
        // Ceiling bound is AABB-aware and mirrors `player::player_collides`:
        // that helper iterates cells up to `floor(pos[1] + PLAYER_HEIGHT)`
        // inclusive and reads out-of-region cells as 0, so a naive scan
        // could declare clearance while the player's head cell sits outside
        // the realized region. Stop the scan as soon as the topmost AABB
        // cell `floor(search_y + PLAYER_HEIGHT)` would land at `world_top`.
        let world_top = (self.world.origin[1] + self.world.side() as i64) as f64;
        let mut search_y = blind_pos[1].floor();
        let mut grounded_pos: Option<[f64; 3]> = None;
        while (search_y + player::PLAYER_HEIGHT).floor() < world_top {
            let cand = [blind_pos[0], search_y, blind_pos[2]];
            if !player::player_collides(&self.world, &cand)
                && player::is_grounded(&self.world, &cand)
            {
                grounded_pos = Some(cand);
                break;
            }
            search_y += 1.0;
        }
        let pos = grounded_pos.unwrap_or(blind_pos);

        let Some(player) = self.entities.get_mut(pid) else {
            return false;
        };
        player.pos = pos;
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
        let pos = [center[0], center[1] + 2.0 * CELLS_PER_METER, center[2]];
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
        let mid_y = center[1] + 1.0 * CELLS_PER_METER;
        self.entities.add(
            [
                center[0] - 10.0 * CELLS_PER_METER,
                mid_y,
                center[2] - 4.0 * CELLS_PER_METER,
            ],
            [0.0; 3],
            sim::EntityKind::Emitter(sim::EmitterState::geyser()),
        );
        self.entities.add(
            [
                center[0] + 10.0 * CELLS_PER_METER,
                mid_y + 2.0 * CELLS_PER_METER,
                center[2] + 6.0 * CELLS_PER_METER,
            ],
            [0.0; 3],
            sim::EntityKind::Emitter(sim::EmitterState::volcano()),
        );
        self.entities.add(
            [
                center[0] + 2.0 * CELLS_PER_METER,
                mid_y,
                center[2] - 12.0 * CELLS_PER_METER,
            ],
            [0.0; 3],
            sim::EntityKind::Emitter(sim::EmitterState::whirlpool()),
        );
        for offset in [-8.0, 0.0, 8.0] {
            self.entities.add(
                [
                    center[0] + offset * CELLS_PER_METER,
                    mid_y,
                    center[2] + 12.0 * CELLS_PER_METER,
                ],
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
        // Refresh the main-thread collision snapshot BEFORE moving the world
        // to the step thread. Player collision during the step reads from
        // this snapshot, not the placeholder we're about to swap in
        // (hash-thing-0s9v). Timed so we can validate the clone-cost
        // estimate against live sweeps.
        //
        // Ordering invariant: the snapshot MUST be set before `step_handle`
        // becomes `Some(...)`. `is_stepping()` ⟺ `step_handle.is_some()`, and
        // grounded movement unwraps `collision_snapshot` via `expect` when
        // stepping; inverting these lines would expose that expect.
        {
            let _t = self.perf.start("collision_snapshot_refresh");
            self.collision_snapshot = Some(self.world.collision_snapshot());
        }
        // Move world to the background thread only after the frame has
        // consumed the live snapshot for movement, interaction, and render.
        let mut world = std::mem::replace(&mut self.world, sim::World::placeholder());
        let sim_qos = self.sim_qos;
        self.step_handle = Some(std::thread::spawn(move || {
            if let Some(qos) = sim_qos {
                thread_qos::apply(qos);
            }
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

    /// Queue a scene swap when the sim step is on the background thread, or
    /// run it immediately otherwise (hash-thing-a9jd). The logs name both
    /// sides so repro sessions can tell "queued vs dropped" apart.
    fn request_scene_swap(&mut self, swap: PendingSceneSwap) {
        if self.is_stepping() {
            log::info!("Scene swap queued during step: {}", swap.label());
            self.pending_scene_swap = Some(swap);
        } else {
            self.dispatch_scene_swap(swap);
        }
    }

    fn dispatch_scene_swap(&mut self, swap: PendingSceneSwap) {
        match swap {
            PendingSceneSwap::LoadLatticeDemo => {
                let _ = self.load_lattice_demo();
            }
            PendingSceneSwap::LoadLatticePanoramaDemo => self.load_lattice_panorama_demo(),
            PendingSceneSwap::ResetTerrain => self.load_terrain_scene(
                "Reset terrain",
                terrain::TerrainParams::for_level(self.volume_size.trailing_zeros()),
            ),
            PendingSceneSwap::LoadGyroid => self.load_gyroid_demo(),
            PendingSceneSwap::LoadDemoSpectacle => {
                self.load_demo_spectacle("Reset spectacle gallery")
            }
            PendingSceneSwap::ResetGolSmoke => self.reset_gol_smoke_scene(),
            PendingSceneSwap::SelectLatticeBeat(beat) => self.load_lattice_demo_beat(beat),
            PendingSceneSwap::SelectRule(rule, label) => self.apply_selected_rule(rule, label),
        }
    }

    fn apply_selected_rule(&mut self, rule: sim::GameOfLife3D, label: &'static str) {
        // set_gol_smoke_rule routes through mutate_materials, which already
        // invalidates the material-dependent hashlife caches (sim/world.rs:529).
        // The !gol_smoke_scene branch doesn't change the registry, so it has
        // no caches to invalidate either.
        if self.gol_smoke_scene {
            self.world.set_gol_smoke_rule(rule);
            if let Some(renderer) = &mut self.renderer {
                renderer.upload_palette(&self.world.materials().color_palette_rgba());
            }
        }
        log::info!("Rule: {label}");
    }

    fn run_pending_scene_swap(&mut self) {
        let Some(swap) = self.pending_scene_swap.take() else {
            return;
        };
        log::info!("Scene swap executing after step: {}", swap.label());
        self.dispatch_scene_swap(swap);
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
                "  V  Panorama reveal",
                "  [/] U/I/O  DEV jumps (Tab for orbit)",
                "  0  Recenter",
                "  H  Heatmap    +/-  Resolution",
                "  F5 Pause      F1  Signal legend",
                "  C  Clear perf",
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
                "  V  Panorama reveal",
                "  0  Recenter",
                "  H  Heatmap    +/-  Resolution",
                "  F5 Pause      F1  Signal legend",
                "  C  Clear perf",
                "  Esc Exit",
            ],
        }
    }

    fn lattice_debug_jumps_enabled(&self) -> bool {
        self.camera_mode == CameraMode::Orbit
    }

    /// Update cached render-side world geometry after world changes.
    fn sync_render_cache(&mut self) {
        self.render_origin = self.world.origin;
        self.render_inv_size = 1.0 / self.world.side() as f32;
    }

    /// Drop any perf / FPS state carried over from a prior scene. Called
    /// from every scene loader so a scene swap can't leave the window-title
    /// FPS EWMA, perf histograms, and memory peak-marks anchored to the
    /// previous scene's regime (hash-thing-6nsh).
    ///
    /// Route the `last_frame` + `suppress_next_fps_sample` reset through
    /// `mark_resume_edge` (hash-thing-v79j). The scene loaders that call
    /// this helper then run terrain gen + SVDAG upload — hundreds of ms at
    /// 256³ — before the next redraw. Without the resume edge, the first
    /// post-swap `dt_wall` captures that upload time, and with
    /// `smoothed_fps == 0.0` the `smooth_fps` zero-sentinel seeds the EWMA
    /// at ~3 FPS (`1 / 0.33s`). Same shape as hash-thing-dxjb (focus edge)
    /// and hash-thing-6e4a (cold-start).
    fn reset_scene_perf_state(&mut self) {
        self.perf.clear();
        self.mem_stats.reset_peaks();
        self.smoothed_fps = 0.0;
        self.mark_resume_edge();
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
        // Scene swaps and lattice beats can flip into Orbit from captured
        // FPS (hash-thing-c4sm). Without this sync, `cursor_captured` stays
        // true while `should_capture_cursor` is false, so the cursor is
        // grabbed but no input path drives the orbit.
        self.sync_cursor_capture();
    }

    fn load_lattice_demo_beat(&mut self, beat: LatticeDemoBeat) {
        // Reseed the scene and use the fresh DemoLayout anchors — direct
        // return rather than an App-side cache so a stale layout can never
        // drive a waypoint into unrelated geometry (hash-thing-ptew).
        let Some(layout) = self.load_lattice_demo() else {
            // Background sim step in flight — load_lattice_demo already
            // logged the deferral. Skip the waypoint too; applying a
            // lattice-shaped pose on top of the previous scene is exactly
            // the class of bug that ptew is about.
            return;
        };
        let waypoint = lattice_demo_waypoint(self.world.side(), &layout, beat);
        self.apply_player_pose(waypoint.player);
        self.apply_orbit_camera_pose(waypoint.camera);
        self.current_demo_beat = Some(beat);
        log::info!("Lattice debug jump: {} ({})", waypoint.label, beat.label());
    }

    /// Resolve a `[`/`]`/`u`/`i`/`o` press into a queued
    /// `SelectLatticeBeat` request (hash-thing-5a5a).
    ///
    /// Routes through `PendingSceneSwap` so a press during a background
    /// sim step gets queued instead of silently dropped. Resolves the
    /// target beat against `current_demo_beat` *at press time* so the
    /// cycle-direction intent stays pinned even if a later key
    /// overwrites the queue; dispatch then runs against the pinned
    /// target regardless of intermediate state.
    ///
    /// Clears `short_demo_cut` synchronously — `update_lattice_short_demo_cut`
    /// runs every frame during stepping, so deferring the clear to
    /// dispatch would let the cut timer keep advancing against a
    /// now-cancelled cut.
    ///
    /// Caller is expected to have already gated on
    /// `lattice_debug_jumps_enabled()`. Only the five debug-jump keys
    /// are accepted; any other input hits `unreachable!()`.
    fn request_lattice_beat_jump(&mut self, key: &str) {
        self.short_demo_cut = None;
        let target = resolve_lattice_beat_target(self.current_demo_beat, key);
        self.request_scene_swap(PendingSceneSwap::SelectLatticeBeat(target));
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
        match LatticeShortDemoCut::advance(self.current_demo_beat, cut.started_at.elapsed()) {
            BeatAdvance::Set(beat) => self.load_lattice_demo_beat(beat),
            BeatAdvance::End => self.short_demo_cut = None,
            BeatAdvance::Hold => {}
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

    fn select_rule(&mut self, rule: sim::GameOfLife3D, label: &'static str) {
        self.gol_smoke_rule = rule;
        self.request_scene_swap(PendingSceneSwap::SelectRule(rule, label));
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
        self.reset_scene_perf_state();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.exit_lattice_demo_mode();
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
            // Pack held material as the clone source (see pack_clone_source).
            let state = hash_thing::terrain::materials::pack_clone_source(held_material);
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

    /// Zero the lattice-demo beat/cut state. Called from every non-lattice
    /// scene loader so an in-flight V cut (short_demo_cut = Some(_)) does not
    /// advance after the user explicitly swapped scenes via B/G/M/terrain —
    /// otherwise `update_lattice_short_demo_cut` would next frame call
    /// `load_lattice_demo_beat` and yank the user back into the lattice scene
    /// (hash-thing-0p0s).
    fn exit_lattice_demo_mode(&mut self) {
        self.current_demo_beat = None;
        self.short_demo_cut = None;
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
        self.reset_scene_perf_state();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.exit_lattice_demo_mode();
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
        self.reset_scene_perf_state();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.exit_lattice_demo_mode();
        log::info!(
            "Gyroid megastructure: pop={} gen={:.1}ms collapses={} classifies={}",
            self.world.population(),
            elapsed.as_secs_f64() * 1000.0,
            stats.total_collapses(),
            stats.classify_calls,
        );
    }

    fn reset_gol_smoke_scene(&mut self) {
        if self.is_stepping() {
            return;
        }
        self.world = sim::World::new(self.volume_size.trailing_zeros());
        self.world.set_gol_smoke_rule(self.gol_smoke_rule);
        self.world.seed_center(12, 0.35);
        self.gol_smoke_scene = true;
        self.paused = true;
        self.reset_scene_perf_state();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
        }
        Self::upload_volume(
            &mut self.renderer,
            &mut self.world,
            &mut self.svdag,
            &mut self.last_svdag_stats,
        );
        self.sync_render_cache();
        self.exit_lattice_demo_mode();
        log::info!("Reset GoL smoke sphere: pop={}", self.world.population());
    }

    fn load_lattice_demo(&mut self) -> Option<sim::DemoLayout> {
        if self.is_stepping() {
            // Without this log the `n` key looks dead during a long sim
            // step — see hash-thing-1a1n. The completion log at the end
            // of this function covers the success path.
            log::info!("load_lattice_demo: deferred — background sim step in flight");
            return None;
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
        self.reset_scene_perf_state();
        if let Some(renderer) = &mut self.renderer {
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
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
        Some(layout)
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
        self.reset_scene_perf_state();
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
        self.exit_lattice_demo_mode();
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
            // Do NOT focus-on-launch by default (edward 2026-04-21): the
            // game stealing focus mid-dev-loop blocks keyboard input to the
            // terminal/agent surface. Opt in with HASH_THING_FOCUS=1.
            window.set_visible(true);
            if std::env::var("HASH_THING_FOCUS").ok().as_deref() == Some("1") {
                window.focus_window();
            }
            self.window = Some(window.clone());

            let mut renderer =
                pollster::block_on(render::Renderer::new(window.clone(), self.volume_size));
            renderer.upload_palette(&self.world.materials().color_palette_rgba());
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
            // 4ioh: new window starts with the bootstrap title. Clear
            // the throttle sentinel so the first real frame refreshes
            // the title immediately instead of waiting out the 250 ms
            // interval from a previous window's refresh.
            self.last_title_update = None;
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
                // hash-thing-0zrc: detect OS-initiated windowed<->fullscreen
                // transitions by diffing cached state against the current
                // `window.fullscreen()`. winit 0.30 has no dedicated
                // `FullscreenChanged` event; on macOS/Windows the toggle
                // surfaces as `Resized`, so this is the idiomatic hook.
                // Pre-transition samples belong to a different regime
                // (different render-target size, GPU bottleneck mix), so
                // dropping them gives dlse.2.2 A/B comparisons a clean slate.
                //
                // Scope: strictly windowed<->fullscreen. Same-mode regime
                // changes — monitor swap while fullscreen, DPI change,
                // windowed drag — are intentionally out of scope and left
                // to scene-reset paths or follow-up beads. We call the
                // narrow `self.perf.clear()` (not `reset_scene_perf_state`)
                // because the bead is scoped to perf histograms; FPS EWMA
                // and mem peaks are left alone on purpose.
                if let Some(window) = self.window.as_ref() {
                    let now_fullscreen = window.fullscreen().is_some();
                    if now_fullscreen != self.was_fullscreen {
                        log::info!(
                            "fullscreen transition {}→{}: clearing perf histograms (hash-thing-0zrc)",
                            self.was_fullscreen,
                            now_fullscreen,
                        );
                        self.perf.clear();
                        self.was_fullscreen = now_fullscreen;
                    }
                }
            }

            // Pause the redraw treadmill while the window is hidden
            // (minimized, behind another window, screen locked). On un-occlude,
            // re-arm the loop with a single `request_redraw`. See
            // hash-thing-8jp. Also treat occlusion as a capture-blocking
            // signal — macOS Mission Control / Spaces can revoke the cursor
            // grab here without a Focused(false) event (hash-thing-w0o9).
            WindowEvent::Occluded(occluded) => {
                log::info!("WindowEvent::Occluded({occluded})");
                if occluded {
                    self.enter_occluded_state();
                } else {
                    self.leave_occluded_state();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }

            WindowEvent::Focused(focused) => {
                self.focused = focused;
                if focused {
                    self.last_mouse = None;
                    self.mark_resume_edge();
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
                            if self.camera_mode != CameraMode::Orbit =>
                        {
                            log::debug!("s ignored: single-step only in Orbit mode (current=FPS)");
                        }
                        winit::keyboard::Key::Character("s") if self.is_stepping() => {
                            log::debug!(
                                "s ignored: single-step denied while background step is in flight"
                            );
                        }
                        winit::keyboard::Key::Character("s") => {
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
                            self.memo_hud_dirty = true;
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
                        winit::keyboard::Key::Character("r" | "t") => {
                            // Queue through PendingSceneSwap so a press during
                            // a background sim step lands at the next step
                            // boundary instead of being silently dropped by
                            // load_terrain_scene's is_stepping guard
                            // (hash-thing-u5ik). `r` and `t` share the same
                            // terrain reset today; split in the future if the
                            // keys diverge.
                            self.request_scene_swap(PendingSceneSwap::ResetTerrain);
                        }
                        winit::keyboard::Key::Character("g") => {
                            self.request_scene_swap(PendingSceneSwap::ResetGolSmoke);
                        }
                        winit::keyboard::Key::Character("m") => {
                            self.request_scene_swap(PendingSceneSwap::LoadGyroid);
                        }
                        winit::keyboard::Key::Character("n") => {
                            self.request_scene_swap(PendingSceneSwap::LoadLatticeDemo);
                        }
                        winit::keyboard::Key::Character(k @ ("[" | "]" | "u" | "i" | "o")) => {
                            // Lattice-demo debug jumps are orbit-only:
                            // teleporting the camera between scripted
                            // beats is useful when previewing the demo,
                            // less so while the player is walking around
                            // in FPS mode. Log the silent branch so the
                            // key doesn't feel dead (hash-thing-1a1n).
                            if self.lattice_debug_jumps_enabled() {
                                self.request_lattice_beat_jump(k);
                            } else {
                                // debug!, not info!: winit delivers key-repeat
                                // events and the default filter is `info`, so
                                // holding `[` would spam the console at
                                // ~30 lines/sec. Discoverability now lives in
                                // the FPS legend line; the log is a fallback
                                // for RUST_LOG=debug sessions (hash-thing-ibe0).
                                log::debug!(
                                    "{k}: lattice debug jump — orbit mode only (Tab to switch camera)"
                                );
                            }
                        }
                        winit::keyboard::Key::Character("v") => {
                            // a9jd: the short-demo cut is the user-facing
                            // demo entry (not an orbit-only debug jump),
                            // so no camera-mode gate. `[`, `]`, `u`/`i`/`o`
                            // keep their debug gates — those are genuine
                            // beat-cycle jumps.
                            self.request_scene_swap(PendingSceneSwap::LoadLatticePanoramaDemo);
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
                                    _ => log::debug!(
                                        "digit {digit} ignored in Orbit mode: rule selection uses 1-4 only"
                                    ),
                                }
                            }
                        }
                        winit::keyboard::Key::Character("b") => {
                            // Default demo gallery: deterministic local fire/water set pieces
                            // staged around the beat waypoints.
                            self.request_scene_swap(PendingSceneSwap::LoadDemoSpectacle);
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
                        winit::keyboard::Key::Character("p") if self.is_stepping() => {
                            log::debug!(
                                "p ignored: perf dump denied while background step is in flight"
                            );
                        }
                        // hash-thing-hso: on-demand dump of the full perf +
                        // memory summary, independent of the wall-clock log
                        // cadence.
                        winit::keyboard::Key::Character("p") => {
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
                if self.occluded || !self.focused {
                    return;
                }

                // hash-thing-lke9: per-frame retry tap. If `resumed()`'s
                // initial grab attempt lost the OS-activation race (macOS /
                // agent launches), `cursor_captured` is false even though
                // `should_capture_cursor()` wants it true. `sync_cursor_capture`
                // short-circuits in steady state (one bool comparison) and
                // re-attempts the grab when state mismatches — exactly the
                // post-startup-failure window. Placed BEFORE the
                // `startup_scene_pending` block so the retry isn't delayed
                // behind hundreds of ms of cold-start scene generation.
                self.sync_cursor_capture();

                if self.startup_scene_pending {
                    self.startup_scene_pending = false;
                    self.load_initial_scene();
                    // load_initial_scene runs terrain gen + SVDAG upload —
                    // hundreds of ms at 256³. Re-arm the resume edge AFTER
                    // the upload window so `dt_wall` below measures the
                    // first real frame, not the cold-start work. Symmetric
                    // with focus / unocclude / scene-swap resume edges —
                    // all four share the same pairing contract (last_frame
                    // + suppress). See hash-thing-v79j / hash-thing-6e4a.
                    self.mark_resume_edge();
                }

                // Frame delta time. `dt` is clamped to 0.1s for xa7
                // player movement (a 500ms hiccup must not teleport the
                // player). `dt_wall` is the unclamped wall-clock value for
                // the EWMA title readout — without it, sub-10-FPS frames
                // all report "10 FPS" because 1.0 / 0.1 = 10 (dvzl).
                let (dt_wall, dt) = compute_frame_dts(self.last_frame.elapsed());
                self.last_frame = std::time::Instant::now();
                self.update_lattice_short_demo_cut();

                self.apply_fps_sample(dt_wall);
                // Throttle title refresh to ~4 Hz. EWMA-smoothed
                // FPS still jitters between integer readings on the
                // {:.0} boundary (e.g. 59/60 flicker), and 60 title
                // rewrites per second hands the OS more repaint work
                // than a human can read anyway (hash-thing-4ioh).
                let now = std::time::Instant::now();
                if should_refresh_title(self.last_title_update, now) {
                    if let Some(window) = &self.window {
                        if let Some(renderer) = &self.renderer {
                            let scale_pct = (renderer.render_scale * 100.0) as u32;
                            window.set_title(&format!(
                                "hash-thing | {:.0} FPS | {}³ | scale {}%",
                                self.smoothed_fps, self.volume_size, scale_pct,
                            ));
                            self.last_title_update = Some(now);
                        }
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
                                // Refresh the collision snapshot from the
                                // just-returned world so the next frame's
                                // player physics reads post-step geometry
                                // (hash-thing-0s9v). Same clone-cost surface
                                // as the step-start refresh; timed under the
                                // same metric so sweeps see both samples.
                                {
                                    let _t = self.perf.start("collision_snapshot_refresh");
                                    self.collision_snapshot = Some(self.world.collision_snapshot());
                                }
                                // Refresh cached memo-health summary while the
                                // real world is in hand (hash-thing-stue.6):
                                // the (stepping) log branch below cannot read
                                // self.world (it's about to be replaced by a
                                // placeholder again on the next step).
                                self.last_memo_summary = self.world.memo_summary();
                                self.memo_hud_dirty = true;
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
                    // a9jd: drain a queued scene swap FIRST. If one is set and
                    // it replaces `world`, the pending player action was
                    // queued against the world that's about to be discarded,
                    // so running it first would do a full SVDAG rebuild/upload
                    // that the scene swap immediately throws out. Dropping the
                    // player action in that case matches "latest intent wins."
                    // Swaps that don't discard the world (SelectRule, 9t8m)
                    // preserve the queued action.
                    if self
                        .pending_scene_swap
                        .is_some_and(PendingSceneSwap::discards_world)
                    {
                        self.pending_player_action = None;
                    }
                    self.run_pending_scene_swap();
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

                    // Memo HUD overlay (hash-thing-nhwo): propagate
                    // visibility every frame, re-upload text on dirty.
                    // Visibility assignment must live outside the dirty
                    // gate so a future toggle path doesn't get stuck.
                    renderer.memo_hud_visible = self.memo_hud_visible;
                    if self.memo_hud_visible && self.memo_hud_dirty {
                        self.memo_hud_dirty = false;
                        let lines: Vec<&str> = self.last_memo_summary.split_whitespace().collect();
                        renderer.set_memo_hud_text(&lines);
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
                        // 0s9v: grounded physics every frame. During
                        // background steps, collide against a snapshot of
                        // the step-entry world instead of the placeholder.
                        let grid: &dyn player::VoxelGrid = if stepping {
                            self.collision_snapshot
                                .as_ref()
                                .expect("collision snapshot refreshed at step start")
                        } else {
                            &self.world
                        };
                        if let Some(p) = self.entities.get_mut(pid) {
                            let step = player::step_grounded_movement(
                                grid,
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
                        if let Some(p) = self.entities.iter().find(|entity| entity.id == pid) {
                            camera_grounded = player::is_grounded(grid, &p.pos);
                        }
                        let camera_motion = (camera_planar_speed, sprinting, camera_grounded);
                        // hash-thing-m1f.4 / 37r: grow the world when the
                        // player approaches any boundary (positive or negative).
                        // Skipped during background step — world is placeholder.
                        if !self.is_stepping() {
                            if let Some(p) = self.entities.get_mut(pid) {
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
                                    // 0s9v: ensure_region may grow the
                                    // world; refresh the collision snapshot
                                    // so the next background step starts
                                    // from the grown state. Timed under the
                                    // same metric as the other two refresh
                                    // sites so sweeps observe the full
                                    // clone-cost distribution.
                                    {
                                        let _t = self.perf.start("collision_snapshot_refresh");
                                        self.collision_snapshot =
                                            Some(self.world.collision_snapshot());
                                    }
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
                                        let palette = self.world.materials().color_palette_rgba();
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
                    // dlse.2.4: second bracket around the blit + overlay
                    // render pass. `render_gpu` has always been
                    // compute-only despite the name; the render-pass
                    // GPU cost lands in this separate metric.
                    if let Some(d) = renderer.take_last_render_pass_gpu_frame_time() {
                        self.perf.record("render_pass_gpu", d);
                    }
                }

                // Belt-and-suspenders: if the surface reports Occluded
                // before winit fires `WindowEvent::Occluded(true)` (some
                // platforms are lazy about that event), latch the flag here
                // so the next RedrawRequested short-circuits at the top of
                // the arm. Route through the same helper as the winit event
                // so cursor capture releases on this path too (w0o9).
                if matches!(outcome, Some(render::FrameOutcome::Occluded)) {
                    self.enter_occluded_state();
                    return;
                }

                // Kick off the next background step only after this frame has
                // finished consuming the live world snapshot.
                self.maybe_start_background_step();

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
    log::info!("  V: Panoramic lattice reveal (timed demo cut)");
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
    fn smooth_fps_seeds_from_zero_prev_returns_instant() {
        assert_eq!(smooth_fps(0.0, 60.0, 0.05), 60.0);
        // Negative prev (shouldn't occur in practice) hits the same
        // sentinel branch rather than producing a nonsense blend.
        assert_eq!(smooth_fps(-1.0, 20.0, 0.1), 20.0);
    }

    #[test]
    fn smooth_fps_weights_new_sample_by_alpha() {
        // 0.1 * 20 + 0.9 * 100 = 92.0. Round-trip through f64 to avoid
        // flaky equality; the arithmetic is exact for these values.
        assert!((smooth_fps(100.0, 20.0, 0.1) - 92.0).abs() < 1e-12);
        // alpha=0 pins to prev (filter is frozen).
        assert_eq!(smooth_fps(42.0, 1.0, 0.0), 42.0);
        // alpha=1 snaps to the instant sample (no smoothing).
        assert_eq!(smooth_fps(42.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn smooth_fps_converges_to_steady_instant() {
        // Drive the filter with a constant 60 FPS from a distant start
        // and assert it settles within a plausible window. At alpha=0.05
        // the 99% settle is log(0.01) / log(0.95) ≈ 90 samples, so 200
        // iterations is plenty of headroom.
        let mut s = 10.0;
        for _ in 0..200 {
            s = smooth_fps(s, 60.0, 0.05);
        }
        assert!((s - 60.0).abs() < 0.1, "expected ~60, got {s}");
    }

    // hash-thing-dvzl: the EWMA title readout must not get the 100ms
    // clamp, or frames slower than 10 FPS all round-trip to exactly "10".
    // xa7 movement still uses the clamped value.

    #[test]
    fn compute_frame_dts_preserves_wall_for_slow_frames() {
        let (wall, clamped) = compute_frame_dts(Duration::from_millis(200));
        assert!(
            (wall - 0.2).abs() < 1e-12,
            "wall dt must not clamp; got {wall}"
        );
        assert!(
            (clamped - 0.1).abs() < 1e-12,
            "clamped dt must cap at 0.1; got {clamped}"
        );
    }

    #[test]
    fn compute_frame_dts_matches_below_clamp() {
        // At typical frame times the two values coincide — the clamp only
        // bites above 100ms.
        let (wall, clamped) = compute_frame_dts(Duration::from_millis(16));
        assert!((wall - 0.016).abs() < 1e-12);
        assert!((clamped - 0.016).abs() < 1e-12);
    }

    // hash-thing-4ioh: title-refresh throttle fires on the first frame
    // (`None` seed) and then suppresses rewrites until the interval
    // elapses. At 60 FPS this collapses ~15 per-frame rewrites into one.
    #[test]
    fn should_refresh_title_fires_on_first_frame() {
        let now = std::time::Instant::now();
        assert!(should_refresh_title(None, now));
    }

    #[test]
    fn should_refresh_title_suppresses_within_interval() {
        let last = std::time::Instant::now();
        let soon = last + TITLE_REFRESH_INTERVAL / 2;
        assert!(!should_refresh_title(Some(last), soon));
    }

    #[test]
    fn should_refresh_title_fires_at_interval_boundary() {
        let last = std::time::Instant::now();
        let later = last + TITLE_REFRESH_INTERVAL;
        assert!(should_refresh_title(Some(last), later));
    }

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
        let mut world = sim::World::new(6); // side 64
        let layout = world.seed_lattice_progression_demo();
        let side = world.side();
        for beat in [
            LatticeDemoBeat::Intro,
            LatticeDemoBeat::Interior,
            LatticeDemoBeat::Panorama,
        ] {
            let waypoint = lattice_demo_waypoint(side, &layout, beat);
            for axis in [0, 1, 2] {
                assert!(waypoint.player.pos[axis] > 0.0);
                assert!(waypoint.player.pos[axis] < side as f64);
            }
        }
    }

    #[test]
    fn lattice_panorama_waypoint_faces_inward_and_upward() {
        let mut world = sim::World::new(6);
        let layout = world.seed_lattice_progression_demo();
        let waypoint = lattice_demo_waypoint(world.side(), &layout, LatticeDemoBeat::Panorama);
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
        let mut world = sim::World::new(6);
        let layout = world.seed_lattice_progression_demo();
        let side = world.side();
        let intro = lattice_demo_waypoint(side, &layout, LatticeDemoBeat::Intro);
        let interior = lattice_demo_waypoint(side, &layout, LatticeDemoBeat::Interior);
        let panorama = lattice_demo_waypoint(side, &layout, LatticeDemoBeat::Panorama);
        assert_ne!(intro.label, interior.label);
        assert_ne!(interior.label, panorama.label);
        assert_ne!(intro.player.pos, panorama.player.pos);
    }

    /// ptew: player was landing embedded in solid lattice cells because
    /// the beat waypoints were hardcoded normalized fractions that did
    /// not track the scene geometry. This test seeds the real scene and
    /// asserts that every beat's spawn cell is walkable AND grounded, so
    /// the class of bug cannot regress silently.
    #[test]
    fn lattice_demo_waypoints_do_not_collide_with_scene() {
        // Level 8: 69cq (CELLS_PER_METER=4) requires lattice rooms to fit
        // a 6.4-cell-tall player; the other lattice spawn tests moved to
        // level 8 for the same reason.
        let mut world = sim::World::new(8);
        let layout = world.seed_lattice_progression_demo();
        let side = world.side();
        for beat in [
            LatticeDemoBeat::Intro,
            LatticeDemoBeat::Interior,
            LatticeDemoBeat::Panorama,
        ] {
            let waypoint = lattice_demo_waypoint(side, &layout, beat);
            assert!(
                !player::player_collides(&world, &waypoint.player.pos),
                "{:?} waypoint must spawn in AIR: pos={:?}",
                beat,
                waypoint.player.pos,
            );
            assert!(
                player::is_grounded(&world, &waypoint.player.pos),
                "{:?} waypoint must have floor support: pos={:?}",
                beat,
                waypoint.player.pos,
            );
        }
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
    fn pending_scene_swap_select_lattice_beat_labels_are_stable() {
        // hash-thing-5a5a: the queued log line uses `label()` to
        // distinguish which beat was requested, so every variant must
        // map to a distinct static string.
        assert_eq!(
            PendingSceneSwap::SelectLatticeBeat(LatticeDemoBeat::Intro).label(),
            "lattice_beat_intro",
        );
        assert_eq!(
            PendingSceneSwap::SelectLatticeBeat(LatticeDemoBeat::Interior).label(),
            "lattice_beat_interior",
        );
        assert_eq!(
            PendingSceneSwap::SelectLatticeBeat(LatticeDemoBeat::Panorama).label(),
            "lattice_beat_panorama",
        );
    }

    #[test]
    fn dispatch_scene_swap_select_lattice_beat_sets_current_beat() {
        // hash-thing-5a5a: the drain path must actually load the
        // requested beat, not just decode the variant. Bypass the
        // queue by calling dispatch directly on a non-stepping App so
        // the dispatch arm's body is exercised end-to-end.
        let mut app = App::new(64);
        app.dispatch_scene_swap(PendingSceneSwap::SelectLatticeBeat(
            LatticeDemoBeat::Interior,
        ));
        assert_eq!(app.current_demo_beat, Some(LatticeDemoBeat::Interior));
    }

    #[test]
    fn resolve_lattice_beat_target_snapshots_cycle_direction() {
        // hash-thing-5a5a: `[`/`]` must resolve against the current
        // beat at press time. `]` advances with wraparound, `[`
        // retreats with wraparound, and `u`/`i`/`o` map to the literal
        // beat regardless of current state.
        use LatticeDemoBeat::*;

        assert_eq!(resolve_lattice_beat_target(Some(Interior), "]"), Panorama);
        assert_eq!(resolve_lattice_beat_target(Some(Interior), "["), Intro);
        assert_eq!(resolve_lattice_beat_target(Some(Panorama), "]"), Intro);
        assert_eq!(resolve_lattice_beat_target(Some(Intro), "["), Panorama);

        // `current_demo_beat == None` → `unwrap_or(Intro)`: `[` wraps
        // to Panorama, `]` advances to Interior. Pins the pre-existing
        // fresh-scene behavior.
        assert_eq!(resolve_lattice_beat_target(None, "["), Panorama);
        assert_eq!(resolve_lattice_beat_target(None, "]"), Interior);

        // Literal mappings are independent of current.
        for current in [None, Some(Intro), Some(Interior), Some(Panorama)] {
            assert_eq!(resolve_lattice_beat_target(current, "u"), Intro);
            assert_eq!(resolve_lattice_beat_target(current, "i"), Interior);
            assert_eq!(resolve_lattice_beat_target(current, "o"), Panorama);
        }
    }

    #[test]
    fn request_lattice_beat_jump_clears_short_demo_cut_synchronously() {
        // hash-thing-5a5a: `update_lattice_short_demo_cut` runs every
        // frame regardless of stepping. Deferring the clear to
        // dispatch would let the cut timer keep advancing against a
        // now-cancelled cut, so the clear must happen at press time.
        //
        // On a non-stepping App `request_lattice_beat_jump` dispatches
        // immediately, which re-seeds the lattice demo and then lands
        // the requested beat — so `current_demo_beat` holds the
        // resolved target and `short_demo_cut` is cleared along the
        // way. Both signals together prove the clear happens before
        // the dispatch hands off to `load_lattice_demo_beat`.
        let mut app = App::new(32);
        app.short_demo_cut = Some(LatticeShortDemoCut {
            started_at: std::time::Instant::now(),
        });
        app.current_demo_beat = Some(LatticeDemoBeat::Intro);

        app.request_lattice_beat_jump("]");

        assert!(
            app.short_demo_cut.is_none(),
            "short_demo_cut must be cleared when a beat jump is requested",
        );
        assert_eq!(
            app.current_demo_beat,
            Some(LatticeDemoBeat::Interior),
            "press-time resolution must hand the snapshotted target to dispatch",
        );
    }

    #[test]
    fn pending_scene_swap_select_rule_label_is_stable() {
        // hash-thing-9t8m: routing select_rule through PendingSceneSwap means
        // the queued-line in request_scene_swap reads "Scene swap queued during
        // step: select_rule". Pin that label so the log surface doesn't drift.
        let swap = PendingSceneSwap::SelectRule(sim::GameOfLife3D::rule445(), "445");
        assert_eq!(swap.label(), "select_rule");
    }

    #[test]
    fn pending_scene_swap_select_rule_round_trips_payload() {
        // The variant must compare equal when the rule and label match, so the
        // queue can dedupe / replace earlier queued swaps cleanly.
        let a = PendingSceneSwap::SelectRule(sim::GameOfLife3D::new(9, 26, 5, 7), "Amoeba");
        let b = PendingSceneSwap::SelectRule(sim::GameOfLife3D::new(9, 26, 5, 7), "Amoeba");
        let c = PendingSceneSwap::SelectRule(sim::GameOfLife3D::new(0, 6, 1, 3), "Crystal");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn pending_scene_swap_discards_world_classification() {
        // hash-thing-9t8m post-review: SelectRule must NOT discard a queued
        // player action — it only re-keys the material registry on the
        // existing post-step world. Every other variant replaces `world` and
        // therefore invalidates any concurrently-queued player action.
        assert!(
            !PendingSceneSwap::SelectRule(sim::GameOfLife3D::rule445(), "445").discards_world()
        );
        assert!(PendingSceneSwap::LoadLatticeDemo.discards_world());
        assert!(PendingSceneSwap::LoadLatticePanoramaDemo.discards_world());
        assert!(PendingSceneSwap::ResetTerrain.discards_world());
        assert!(PendingSceneSwap::LoadGyroid.discards_world());
        assert!(PendingSceneSwap::LoadDemoSpectacle.discards_world());
        assert!(PendingSceneSwap::ResetGolSmoke.discards_world());
        assert!(PendingSceneSwap::SelectLatticeBeat(LatticeDemoBeat::Intro).discards_world());
    }

    #[test]
    fn reset_scene_perf_state_zeroes_smoothed_fps() {
        // hash-thing-6nsh: scene loaders previously cleared perf rings and
        // mem_stats peaks but left smoothed_fps anchored to the previous
        // scene, so the window-title FPS EWMA lied for many refreshes
        // after a scene with a very different render cost.
        let mut app = App::new(64);
        app.smoothed_fps = 240.0;

        app.reset_scene_perf_state();

        assert_eq!(app.smoothed_fps, 0.0);
    }

    #[test]
    fn reset_scene_perf_state_routes_through_mark_resume_edge() {
        // hash-thing-v79j: scene loaders call reset_scene_perf_state then
        // spend hundreds of ms on terrain gen + SVDAG upload before the
        // next redraw. Without the mark_resume_edge pairing, the first
        // post-swap dt_wall captures that upload time and seeds the EWMA
        // at ~3 FPS via the smooth_fps zero-sentinel branch.
        let mut app = App::new(64);
        app.last_frame = std::time::Instant::now() - std::time::Duration::from_secs(2);
        app.suppress_next_fps_sample = false;

        app.reset_scene_perf_state();

        assert!(
            app.last_frame.elapsed() < std::time::Duration::from_millis(50),
            "scene reset must refresh last_frame via mark_resume_edge; elapsed was {:?}",
            app.last_frame.elapsed()
        );
        assert!(
            app.suppress_next_fps_sample,
            "scene reset must flag the next FPS sample as bogus via mark_resume_edge"
        );
    }

    #[test]
    fn cold_start_reset_scene_perf_state_is_sufficient_to_suppress_spike() {
        // hash-thing-6e4a: startup-call-site contract test. The cold-start
        // redraw block calls load_initial_scene (which calls
        // reset_scene_perf_state) then mark_resume_edge. This test
        // deliberately does NOT call mark_resume_edge explicitly — it pins
        // that reset_scene_perf_state ALONE is sufficient to suppress the
        // first post-reset FPS sample, so a future refactor that strips
        // mark_resume_edge out of reset_scene_perf_state would fail here.
        let mut app = App::new(64);
        app.last_frame = std::time::Instant::now() - std::time::Duration::from_secs(2);
        app.smoothed_fps = 0.0;
        app.suppress_next_fps_sample = false;

        app.reset_scene_perf_state();
        // reset_scene_perf_state's internal mark_resume_edge set
        // last_frame = now. Sleep so dt_wall below is a realistic
        // small frame-gap rather than ~zero; the suppress flag masks
        // the actual magnitude, but this keeps the test faithful to
        // the real redraw-block shape.
        std::thread::sleep(std::time::Duration::from_millis(5));

        let dt_wall = app.last_frame.elapsed().as_secs_f64();
        app.apply_fps_sample(dt_wall);
        assert_eq!(
            app.smoothed_fps, 0.0,
            "cold-start first sample must be suppressed by reset_scene_perf_state alone; got {}",
            app.smoothed_fps,
        );

        app.apply_fps_sample(1.0 / 60.0);
        assert!(
            (app.smoothed_fps - 60.0).abs() < 1e-9,
            "second sample must seed via zero-sentinel at the instant value; got {}",
            app.smoothed_fps,
        );
        assert!(
            app.smoothed_fps < 1000.0,
            "smoothed_fps must not spike into thousands; got {}",
            app.smoothed_fps,
        );
    }

    #[test]
    fn lattice_short_demo_cut_advances_in_story_order() {
        use std::time::Duration;

        // None → Intro regardless of elapsed (initial entry).
        assert_eq!(
            LatticeShortDemoCut::advance(None, Duration::from_secs_f32(0.0)),
            BeatAdvance::Set(LatticeDemoBeat::Intro)
        );
        // Hold Intro until the 2s boundary.
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Intro),
                Duration::from_secs_f32(1.5)
            ),
            BeatAdvance::Hold
        );
        // Boundary is inclusive (>=): elapsed == INTERIOR_AT must advance.
        // Pins the semantics so a future refactor to strict `>` would flip
        // this test.
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Intro),
                Duration::from_secs_f32(LatticeShortDemoCut::INTERIOR_AT)
            ),
            BeatAdvance::Set(LatticeDemoBeat::Interior)
        );
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Intro),
                Duration::from_secs_f32(2.1)
            ),
            BeatAdvance::Set(LatticeDemoBeat::Interior)
        );
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Interior),
                Duration::from_secs_f32(5.0)
            ),
            BeatAdvance::Set(LatticeDemoBeat::Panorama)
        );
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Panorama),
                Duration::from_secs_f32(8.1)
            ),
            BeatAdvance::End
        );
    }

    #[test]
    fn lattice_short_demo_cut_does_not_skip_interior_on_long_frame() {
        // hash-thing-x1lg: a long frame (OS pause, long sim step, SVDAG
        // upload) that jumps wall-clock from ~0s to >4.5s must not
        // transition Intro → Panorama in one step. Interior must be
        // displayed for at least one frame, preserving story order.
        use std::time::Duration;

        // Elapsed past the Panorama threshold but currently on Intro:
        // should advance to Interior (one step), not skip to Panorama.
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Intro),
                Duration::from_secs_f32(5.0)
            ),
            BeatAdvance::Set(LatticeDemoBeat::Interior)
        );
        // Same elapsed again the next frame, now on Interior: advances
        // to Panorama on the very next frame.
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Interior),
                Duration::from_secs_f32(5.0)
            ),
            BeatAdvance::Set(LatticeDemoBeat::Panorama)
        );
        // Elapsed past the End threshold but still on Intro: one step
        // to Interior, not straight to End. Protects the cut from being
        // torn down before the user sees any middle beat.
        assert_eq!(
            LatticeShortDemoCut::advance(
                Some(LatticeDemoBeat::Intro),
                Duration::from_secs_f32(10.0)
            ),
            BeatAdvance::Set(LatticeDemoBeat::Interior)
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
    fn first_person_legend_notes_lattice_debug_jumps() {
        let lines = App::legend_lines(CameraMode::FirstPerson);
        assert!(lines.iter().any(|line| line.contains("Space       Leap")));
        assert!(lines.iter().any(|line| line.contains("Scroll/1-9  Matter")));
        assert!(!lines.iter().any(|line| line.contains("Fly up")));
        assert!(!lines.iter().any(|line| line.contains("Fly down")));
        // ibe0: FPS legend should mention the orbit-only DEV jumps so users
        // pressing `[/]UIO` in FPS mode know where to look, without copying
        // the full orbit-mode entry wording.
        let dev_jump_line = lines
            .iter()
            .find(|line| line.contains("[/]") && line.contains("U/I/O"))
            .expect("FPS legend should list the orbit-only DEV jump keys");
        assert!(
            dev_jump_line.contains("DEV"),
            "DEV jump legend line should label the keys as DEV: {dev_jump_line}"
        );
        assert!(
            dev_jump_line.contains("Tab") || dev_jump_line.contains("orbit"),
            "DEV jump legend line should point users at Tab/orbit: {dev_jump_line}"
        );
        // a9jd: V is user-facing in every camera mode now (not a DEV jump).
        assert!(lines.iter().any(|line| line.contains("V  Panorama reveal")));
        assert!(lines.iter().any(|line| line.contains("Lattice walk")));
    }

    #[test]
    fn orbit_legend_marks_lattice_jumps_as_debug() {
        let lines = App::legend_lines(CameraMode::Orbit);
        assert!(lines.iter().any(|line| line.contains("DEV prev/next jump")));
        assert!(lines
            .iter()
            .any(|line| line.contains("DEV intro/interior/reveal")));
        // a9jd: `[`, `]`, `U`/`I`/`O` remain DEV beat-cycle jumps, but `V`
        // is the user-facing panorama reveal — not a DEV-only key.
        assert!(lines.iter().any(|line| line.contains("V  Panorama reveal")));
        assert!(lines.iter().any(|line| line.contains("Lattice walk")));
    }

    #[test]
    fn legend_defaults_on_in_all_modes() {
        assert!(default_legend_visibility(CameraMode::FirstPerson));
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
        assert_eq!(
            player.pos,
            [center[0], center[1] + 2.0 * CELLS_PER_METER, center[2]]
        );
    }

    #[test]
    fn background_step_starts_after_live_world_player_update() {
        // 69cq scale: volume_size bumped 8→32 so the 8 m physical world
        // fits after CELLS_PER_METER=4. Fixtures are sized in meters and
        // scaled through CELLS_PER_METER so the physical layout matches
        // the pre-scale intent (3 m × 3 m × 1 m floor slab; 1 m × 2 m ×
        // 1 m column 1 m north of the player).
        let mut app = App::new(32);
        app.paused = false;
        app.keys_held.insert(KeyCode::KeyW);
        let stone = hash_thing::octree::Cell::pack(1, 0).raw();
        let s = CELLS_PER_METER as i64;
        let m = |v: f64| v * CELLS_PER_METER;
        for x in 3 * s..6 * s {
            for y in 0..s {
                for z in 3 * s..6 * s {
                    app.world.set(
                        sim::WorldCoord(x),
                        sim::WorldCoord(y),
                        sim::WorldCoord(z),
                        stone,
                    );
                }
            }
        }
        for x in 4 * s..5 * s {
            for y in s..3 * s {
                for z in 3 * s..4 * s {
                    app.world.set(
                        sim::WorldCoord(x),
                        sim::WorldCoord(y),
                        sim::WorldCoord(z),
                        stone,
                    );
                }
            }
        }
        let spawn = [m(4.5), m(1.0), m(4.5)];
        app.reset_player_pose(spawn, 0.0, 0.0);

        let pid = app.player_id.expect("player should exist");
        let speed = PLAYER_SPEED;
        let step = player::step_grounded_movement(
            &app.world,
            &spawn,
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
            spawn[2] + delta[2] < expected_z,
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
    fn cursor_capture_requires_first_person_focused_and_unoccluded() {
        // Only (FirstPerson, focused, !occluded) should grab. Occluded is the
        // hash-thing-w0o9 axis: macOS Mission Control revokes the OS-level
        // grab via Occluded(true) without firing Focused(false).
        for mode in [CameraMode::FirstPerson, CameraMode::Orbit] {
            for focused in [false, true] {
                for occluded in [false, true] {
                    let expected = mode == CameraMode::FirstPerson && focused && !occluded;
                    assert_eq!(
                        should_capture_cursor(mode, focused, occluded),
                        expected,
                        "mode={mode:?} focused={focused} occluded={occluded}"
                    );
                }
            }
        }
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
    fn apply_fps_look_wraps_yaw_past_pi() {
        // hash-thing-u0uf invariant: apply_fps_look must keep accumulated
        // yaw in [-π, π) so the f32 downcast for rendering stays precise.
        let mut app = App::new(64);
        let pi = std::f64::consts::PI;
        app.reset_player_pose([0.0, 0.0, 0.0], pi - 0.01, 0.0);

        // 10 px * LOOK_SENSITIVITY(0.003) = 0.03 rad → crosses +π by 0.02.
        app.apply_fps_look(10.0, 0.0);
        let (yaw, _pitch) = player_look(&mut app);
        assert!(yaw >= -pi, "yaw {yaw} below -π after wrap");
        assert!(yaw < pi, "yaw {yaw} not strictly below +π after wrap");
        let expected = -pi + 0.02;
        assert!(
            (yaw - expected).abs() < 1e-10,
            "yaw {yaw} did not wrap to {expected}"
        );
    }

    #[test]
    fn apply_fps_look_bounds_yaw_across_many_rotations() {
        // hash-thing-u0uf: even under an absurd event storm the wrap
        // invariant must hold — this is the regression guard for "someone
        // removed the wrap and nothing else noticed."
        let mut app = App::new(64);
        let pi = std::f64::consts::PI;
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);

        for _ in 0..100 {
            app.apply_fps_look(10_000.0, 0.0);
        }
        let (yaw, _pitch) = player_look(&mut app);
        assert!(yaw >= -pi, "yaw {yaw} below -π after accumulation");
        assert!(
            yaw < pi,
            "yaw {yaw} not strictly below +π after accumulation"
        );
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

    // hash-thing-lke9: streak-reset coverage for the per-frame retry tap.
    // record_grab_outcome is the pure, window-less home of the warn-throttle
    // contract; testing it here proves all three transitions
    // (success, failure, repeated failure) without needing a real OS grab.

    #[test]
    fn record_grab_outcome_failure_streak_warns_once_then_throttles() {
        let mut app = App::new(64);
        assert!(!app.cursor_capture_grab_warned, "fresh app starts unwarned");

        // First failure of a streak: caller should warn.
        assert!(app.record_grab_outcome(false));
        assert!(app.cursor_capture_grab_warned);
        assert!(!app.cursor_captured);

        // Subsequent failures throttle.
        assert!(!app.record_grab_outcome(false));
        assert!(!app.record_grab_outcome(false));
        assert!(app.cursor_capture_grab_warned);
        assert!(!app.cursor_captured);
    }

    #[test]
    fn record_grab_outcome_success_clears_streak() {
        // Success path is the primary lke9 acceptance edge: a successful
        // grab must reset the warn-throttle so a *future* failure streak
        // can warn once again. Codex plan-review Important #1.
        let mut app = App::new(64);
        app.cursor_capture_grab_warned = true;

        assert!(!app.record_grab_outcome(true));
        assert!(
            !app.cursor_capture_grab_warned,
            "success must clear the warn-throttle"
        );
        assert!(app.cursor_captured);
    }

    #[test]
    fn record_grab_outcome_failure_then_success_then_failure_re_warns() {
        // Full lifecycle: streak 1 (fail-throttle), success (reset),
        // streak 2 (fail again, must warn). Pins the "warn-once-per-streak"
        // contract from both directions.
        let mut app = App::new(64);

        assert!(app.record_grab_outcome(false));
        assert!(!app.record_grab_outcome(false));

        assert!(!app.record_grab_outcome(true));
        assert!(!app.cursor_capture_grab_warned);

        assert!(
            app.record_grab_outcome(false),
            "fresh streak after success must warn again"
        );
    }

    #[test]
    fn apply_cursor_capture_release_clears_warned_flag() {
        // apply_cursor_capture(false) is the second reset path (alongside
        // record_grab_outcome(true)). Verifies the release branch matches
        // the streak-reset contract.
        let mut app = App::new(64);
        app.cursor_capture_grab_warned = true;

        app.apply_cursor_capture(false);

        assert!(!app.cursor_capture_grab_warned);
        assert!(!app.cursor_captured);
    }

    #[test]
    fn apply_cursor_capture_no_window_sets_warned_flag() {
        // No-window path is treated as a soft "couldn't grab" so unit tests
        // can drive the streak lifecycle through App::new. No log fires
        // (there is no OS error to surface) but the flag moves so callers
        // observe the throttled state.
        let mut app = App::new(64);
        assert!(app.window.is_none());

        app.apply_cursor_capture(true);

        assert!(app.cursor_capture_grab_warned);
        assert!(!app.cursor_captured);
    }

    // hash-thing-ezx8: bounded retry counter + dormant-gate coverage.
    // Pairs with the lke9 warn-throttle tests above. The counter is the
    // syscall-budget half of the same per-frame retry contract.

    #[test]
    fn record_grab_outcome_failure_increments_counter() {
        let mut app = App::new(64);
        assert_eq!(
            app.cursor_capture_grab_failures, 0,
            "fresh app starts at zero failures"
        );

        for expected in 1..=3 {
            let _ = app.record_grab_outcome(false);
            assert_eq!(app.cursor_capture_grab_failures, expected);
        }
    }

    #[test]
    fn record_grab_outcome_success_clears_counter() {
        // Success is the primary reset path: any successful grab must
        // re-arm a fresh failure budget.
        let mut app = App::new(64);
        app.cursor_capture_grab_failures = 5;

        let _ = app.record_grab_outcome(true);

        assert_eq!(app.cursor_capture_grab_failures, 0);
        assert!(app.cursor_captured);
    }

    #[test]
    fn apply_cursor_capture_release_clears_counter() {
        // Second reset path: explicit release re-arms the budget alongside
        // clearing the warn-throttle, so a later recapture-then-fail can
        // both warn once and burn its full retry budget.
        let mut app = App::new(64);
        app.cursor_capture_grab_failures = 5;

        app.apply_cursor_capture(false);

        assert_eq!(app.cursor_capture_grab_failures, 0);
        assert!(!app.cursor_captured);
    }

    #[test]
    fn sync_cursor_capture_dormant_after_max_failures() {
        // Once the counter saturates, sync_cursor_capture must skip the
        // apply_cursor_capture(true) call so the per-frame retry tap
        // stops burning syscalls. Proven by observing the counter does
        // not increment further when sync_cursor_capture is invoked
        // with an active capture-direction state mismatch.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.focused = true;
        app.occluded = false;
        app.cursor_captured = false;
        app.cursor_capture_grab_failures = MAX_CURSOR_GRAB_RETRIES;

        // Sanity: the predicate inputs make should_capture true and
        // cursor_captured is false, so without the dormant gate the call
        // would route through apply_cursor_capture(true) and increment
        // the counter via the no-window soft-failure path.
        assert!(should_capture_cursor(
            app.camera_mode,
            app.focused,
            app.occluded
        ));

        app.sync_cursor_capture();

        assert_eq!(
            app.cursor_capture_grab_failures, MAX_CURSOR_GRAB_RETRIES,
            "dormant gate must skip the retry; counter must not advance"
        );
        assert!(!app.cursor_captured, "dormant retry must not flip the flag");
    }

    #[test]
    fn sync_cursor_capture_release_resets_dormant_counter_when_uncaptured() {
        // hash-thing-ezx8 critical-review B1 regression: on a
        // permanently-failing platform dormancy arrives via 600
        // failures with `cursor_captured` staying false the entire
        // time. A subsequent `should_capture==false` transition
        // (Tab to Orbit / focus loss / occlusion) MUST reset the
        // counter even though `cursor_captured` is already false and
        // the state-match early return would otherwise short-circuit.
        // Without this, returning to FPS later finds the gate still
        // dormant and the retry never re-fires.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::Orbit;
        app.focused = true;
        app.occluded = false;
        app.cursor_captured = false;
        app.cursor_capture_grab_failures = MAX_CURSOR_GRAB_RETRIES;

        assert!(!should_capture_cursor(
            app.camera_mode,
            app.focused,
            app.occluded
        ));

        app.sync_cursor_capture();

        assert_eq!(
            app.cursor_capture_grab_failures, 0,
            "release-direction sync must reset the counter even when \
             cursor_captured was already false (no prior successful grab)"
        );
        assert!(!app.cursor_captured);
    }

    #[test]
    fn sync_cursor_capture_dormant_release_then_recapture_re_arms_budget() {
        // End-to-end on the permanently-failing-platform recovery
        // path: dormant -> release sync -> re-capture sync gets a
        // fresh budget. Without B1's fix the recapture sync would
        // return early at the dormant gate (counter still saturated)
        // and the per-frame retry would never re-fire.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.focused = true;
        app.occluded = false;
        app.cursor_captured = false;
        app.cursor_capture_grab_failures = MAX_CURSOR_GRAB_RETRIES;

        // Simulate Tab to Orbit: should_capture flips false. Counter resets.
        app.camera_mode = CameraMode::Orbit;
        app.sync_cursor_capture();
        assert_eq!(app.cursor_capture_grab_failures, 0);

        // Simulate Tab back to FPS: should_capture flips true again.
        // Counter is back at 0 -> dormant gate doesn't fire ->
        // apply_cursor_capture(true) runs -> no-window soft failure
        // increments the counter to 1.
        app.camera_mode = CameraMode::FirstPerson;
        app.sync_cursor_capture();
        assert_eq!(
            app.cursor_capture_grab_failures, 1,
            "re-capture after release must reach apply_cursor_capture(true) \
             and burn one retry from the fresh budget"
        );
    }

    // hash-thing-9r62 regression: drive the cursor_captured flag through the
    // real apply_cursor_capture entry point rather than direct field writes,
    // so the state machine + event-routing contract are both covered (not
    // just the gating logic).

    #[test]
    fn apply_cursor_capture_without_window_forces_uncaptured() {
        // App::new leaves window = None (winit attaches it in resumed()).
        // The entry point's no-window fallback must force cursor_captured =
        // false no matter which direction is requested — otherwise the flag
        // can report "captured" while there is no actual OS-level grab,
        // which is the state w1yq debugged.
        let mut app = App::new(64);
        assert!(app.window.is_none(), "App::new must not attach a window");

        app.apply_cursor_capture(true);
        assert!(
            !app.cursor_captured,
            "no-window apply_cursor_capture(true) must not claim capture"
        );

        app.apply_cursor_capture(false);
        assert!(
            !app.cursor_captured,
            "no-window apply_cursor_capture(false) must stay uncaptured"
        );
    }

    #[test]
    fn apply_cursor_capture_release_routes_events_uncaptured() {
        // Going through apply_cursor_capture(false) from a captured state
        // must leave the event routing in the uncaptured regime: CursorMoved
        // drives look, DeviceEvent::MouseMotion is a noop.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);

        // Setup pretends we were captured; the *entry point under test* is
        // the release call that follows.
        app.cursor_captured = true;

        app.apply_cursor_capture(false);
        assert!(!app.cursor_captured);

        // MouseMotion must be dropped in uncaptured FP.
        app.handle_mouse_motion(999.0, 999.0);
        let (yaw, pitch) = player_look(&mut app);
        assert_eq!(yaw, 0.0, "MouseMotion must not apply after release");
        assert_eq!(pitch, 0.0);

        // CursorMoved must now apply (after the seeding call).
        prime_cursor(&mut app, 100.0, 100.0);
        app.handle_cursor_moved(110.0, 95.0);
        let (yaw, pitch) = player_look(&mut app);
        assert!((yaw - 10.0 * LOOK_SENSITIVITY).abs() < 1e-12);
        assert!((pitch - (-5.0) * LOOK_SENSITIVITY).abs() < 1e-12);
    }

    #[test]
    fn apply_cursor_capture_flip_drops_in_flight_captured_events() {
        // Cross-stream ordering hazard (F1 from the w1yq triple-tier synth):
        // MouseMotion events in flight across a capture → release flip must
        // stop applying as soon as the flag flips. CursorMoved events must
        // begin applying again after the flip (post-seed).
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);

        // Pre-flip: we believed we had capture. MouseMotion applies here.
        app.cursor_captured = true;
        app.handle_mouse_motion(10.0, 6.0);
        let (yaw_cap, pitch_cap) = player_look(&mut app);
        assert!((yaw_cap - 10.0 * LOOK_SENSITIVITY).abs() < 1e-12);
        assert!((pitch_cap - 6.0 * LOOK_SENSITIVITY).abs() < 1e-12);

        // Same-frame CursorMoved while still captured is a noop.
        prime_cursor_preserving_look(&mut app, 500.0, 500.0);
        app.handle_cursor_moved(999.0, 999.0);
        let (yaw, pitch) = player_look(&mut app);
        assert!((yaw - yaw_cap).abs() < 1e-12);
        assert!((pitch - pitch_cap).abs() < 1e-12);

        // Release via the real entry point.
        app.apply_cursor_capture(false);
        assert!(!app.cursor_captured);

        // Post-flip: MouseMotion must now be dropped.
        app.handle_mouse_motion(999.0, 999.0);
        let (yaw, pitch) = player_look(&mut app);
        assert!(
            (yaw - yaw_cap).abs() < 1e-12,
            "MouseMotion dropped after flip"
        );
        assert!((pitch - pitch_cap).abs() < 1e-12);

        // Post-flip: CursorMoved must apply again. Re-seed so the first
        // real delta is predictable — otherwise an untouched last_mouse
        // from pre-flip would produce a huge one-shot delta.
        prime_cursor_preserving_look(&mut app, 200.0, 200.0);
        app.handle_cursor_moved(220.0, 180.0);
        let (yaw, pitch) = player_look(&mut app);
        assert!((yaw - (yaw_cap + 20.0 * LOOK_SENSITIVITY)).abs() < 1e-12);
        assert!((pitch - (pitch_cap + (-20.0) * LOOK_SENSITIVITY)).abs() < 1e-12);
    }

    // hash-thing-w0o9: occlusion (macOS Mission Control / Spaces) must
    // release cursor capture even when Focused(false) never fires. Drive
    // through `enter_occluded_state` so the winit-event path and the
    // `FrameOutcome::Occluded` latch path are both covered (Codex blocker).

    #[test]
    fn enter_occluded_state_releases_cursor_capture() {
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.focused = true;
        app.occluded = false;
        // Simulate a successful grab from before the occlusion event.
        app.cursor_captured = true;

        app.enter_occluded_state();

        assert!(app.occluded, "enter_occluded_state must latch occluded");
        assert!(
            !app.cursor_captured,
            "occlusion must flip cursor_captured even while focused"
        );
        assert!(
            app.keys_held.is_empty(),
            "transient key state must clear on occlusion"
        );
        assert!(!app.jump_was_held);
        assert!(app.last_mouse.is_none());
    }

    #[test]
    fn mouse_motion_after_occlusion_is_noop() {
        // End-to-end proof: once occlusion has flipped capture off, ghost
        // DeviceEvent::MouseMotion deltas delivered during the Space swap
        // must not reach the player pose.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.focused = true;
        app.occluded = false;
        app.cursor_captured = true;
        app.reset_player_pose([0.0, 0.0, 0.0], 0.0, 0.0);

        app.enter_occluded_state();

        app.handle_mouse_motion(999.0, 999.0);
        let (yaw, pitch) = player_look(&mut app);
        assert_eq!(yaw, 0.0);
        assert_eq!(pitch, 0.0);
    }

    #[test]
    fn leave_occluded_state_restores_should_capture() {
        // Recovery path (Gemini nit): after occlude → un-occlude while
        // still focused in FirstPerson, the capture predicate must want
        // to re-grab. We can't verify the actual OS grab without a window,
        // but the predicate is what sync_cursor_capture consults.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.focused = true;
        app.enter_occluded_state();
        assert!(!should_capture_cursor(
            app.camera_mode,
            app.focused,
            app.occluded,
        ));

        app.leave_occluded_state();

        assert!(!app.occluded);
        assert!(app.last_mouse.is_none());
        assert!(should_capture_cursor(
            app.camera_mode,
            app.focused,
            app.occluded,
        ));
    }

    #[test]
    fn apply_orbit_camera_pose_clears_cursor_captured_from_fps() {
        // hash-thing-c4sm: scene swaps (V panorama) and lattice beats that
        // terminate in Orbit previously left `cursor_captured = true` if the
        // player was in captured FPS, because `apply_orbit_camera_pose`
        // didn't sync cursor state. Symptom: grabbed cursor with no input
        // path driving the orbit until the next focus/occlusion transition.
        let mut app = App::new(64);
        app.camera_mode = CameraMode::FirstPerson;
        app.focused = true;
        app.cursor_captured = true;
        let pose = OrbitCameraPose {
            target: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            dist: 1.0,
        };

        app.apply_orbit_camera_pose(pose);

        assert_eq!(app.camera_mode, CameraMode::Orbit);
        assert!(
            !app.cursor_captured,
            "apply_orbit_camera_pose must sync cursor capture; orbit mode never captures",
        );
    }

    // hash-thing-xysz + hash-thing-dxjb: the redraw handler early-returns
    // while paused so `last_frame` stops advancing. Resuming without a
    // reset clamps the first dt to the 100ms guard (xysz); resetting
    // without suppressing the next FPS sample blends a microsecond
    // dt_wall into the EWMA (dxjb). `mark_resume_edge` does both as one
    // step; `leave_occluded_state_routes_through_mark_resume_edge` (below)
    // verifies the unocclude path calls it.

    // hash-thing-dxjb: the focus/unocclude resets happen outside the
    // redraw hot path. The next RedrawRequested sees a near-zero dt —
    // blending 1/dt_wall into the EWMA would spike the displayed FPS
    // into the millions and take many seconds to decay. The paired
    // `suppress_next_fps_sample` flag defuses that. Both resume edges
    // route through `mark_resume_edge`, so the invariant (reset +
    // suppress in lockstep) cannot drift by construction.

    #[test]
    fn mark_resume_edge_resets_last_frame_and_flags_suppress() {
        let mut app = App::new(64);
        app.last_frame = std::time::Instant::now() - std::time::Duration::from_secs(2);
        app.suppress_next_fps_sample = false;

        app.mark_resume_edge();

        assert!(
            app.last_frame.elapsed() < std::time::Duration::from_millis(50),
            "mark_resume_edge must reset last_frame; elapsed was {:?}",
            app.last_frame.elapsed()
        );
        assert!(
            app.suppress_next_fps_sample,
            "mark_resume_edge must flag the next FPS sample as bogus"
        );
    }

    #[test]
    fn leave_occluded_state_routes_through_mark_resume_edge() {
        let mut app = App::new(64);
        app.enter_occluded_state();
        app.suppress_next_fps_sample = false;
        app.last_frame = std::time::Instant::now() - std::time::Duration::from_secs(2);

        app.leave_occluded_state();

        assert!(
            app.last_frame.elapsed() < std::time::Duration::from_millis(50),
            "unocclude must reset last_frame via mark_resume_edge"
        );
        assert!(
            app.suppress_next_fps_sample,
            "unocclude must flag suppression via mark_resume_edge"
        );
    }

    #[test]
    fn apply_fps_sample_consumes_suppress_flag_without_blending() {
        let mut app = App::new(64);
        app.smoothed_fps = 120.0;
        app.suppress_next_fps_sample = true;

        // A microsecond dt would normally push smoothed_fps past 50000
        // in a single 0.05-alpha blend — verify we skip entirely.
        app.apply_fps_sample(1e-6);

        assert_eq!(
            app.smoothed_fps, 120.0,
            "suppress flag must skip the EWMA update on the flagged frame"
        );
        assert!(
            !app.suppress_next_fps_sample,
            "flag must be consumed after the suppressed frame"
        );
    }

    #[test]
    fn apply_fps_sample_blends_after_suppress_consumed() {
        let mut app = App::new(64);
        app.smoothed_fps = 60.0;
        app.suppress_next_fps_sample = true;

        // First call: bogus frame, skipped.
        app.apply_fps_sample(1e-6);
        // Second call: real 60 FPS frame, should blend (identity here).
        app.apply_fps_sample(1.0 / 60.0);

        assert!(
            (app.smoothed_fps - 60.0).abs() < 1e-9,
            "subsequent real frames must blend normally; got {}",
            app.smoothed_fps
        );
    }

    /// Seed `last_mouse` without resetting player yaw/pitch. Used when a
    /// test builds up yaw/pitch in stages and does not want the anchor
    /// reset to clear the player pose.
    fn prime_cursor_preserving_look(app: &mut App, start_x: f64, start_y: f64) {
        app.last_mouse = Some((start_x, start_y));
    }

    // hash-thing-t3zn: warp-landing audit. Each warp key must drop the
    // player into playable space — inside world bounds, AABB clear of
    // solid cells, with floor support, and facing a finite direction.
    // Failures here are real bugs; file a follow-up fix bead before
    // adjusting the assertion.
    fn assert_player_in_playable_space(app: &App, label: &str) {
        let pid = app
            .player_id
            .unwrap_or_else(|| panic!("{label}: warp must leave player_id set"));
        let player = app
            .entities
            .iter()
            .find(|e| e.id == pid)
            .unwrap_or_else(|| panic!("{label}: player entity missing after warp"));
        let pos = player.pos;
        let origin = app.world.origin;
        let side = app.world.side() as i64;
        for axis in 0..3 {
            let lo = origin[axis] as f64;
            let hi = (origin[axis] + side) as f64;
            assert!(
                pos[axis] >= lo && pos[axis] < hi,
                "{label}: pos[{axis}]={} outside world bounds [{}, {})",
                pos[axis],
                lo,
                hi,
            );
        }
        assert!(
            !player::player_collides(&app.world, &pos),
            "{label}: player AABB at {pos:?} overlaps a solid cell",
        );
        assert!(
            player::is_grounded(&app.world, &pos),
            "{label}: player at {pos:?} has no floor support",
        );
        if let sim::EntityKind::Player(state) = &player.kind {
            assert!(
                state.yaw.is_finite() && state.pitch.is_finite(),
                "{label}: degenerate facing yaw={} pitch={}",
                state.yaw,
                state.pitch,
            );
        } else {
            panic!("{label}: player_id points at a non-Player entity");
        }
    }

    /// Audit `n` (LoadLatticeDemo): seeded lattice spawn must be in playable
    /// space.
    #[test]
    fn warp_n_load_lattice_demo_lands_in_playable_space() {
        let mut app = App::new(256);
        app.load_lattice_demo()
            .expect("load_lattice_demo must succeed when no step is in flight");
        assert_player_in_playable_space(&app, "n / load_lattice_demo");
    }

    /// Audit `v` (LoadLatticePanoramaDemo): the user-facing panorama-cut
    /// entry point starts at the Intro beat, so its landing pose must be
    /// playable.
    #[test]
    fn warp_v_load_lattice_panorama_demo_lands_in_playable_space() {
        let mut app = App::new(256);
        app.load_lattice_panorama_demo();
        assert_player_in_playable_space(&app, "v / load_lattice_panorama_demo");
    }

    /// Audit `[`, `]`, `u`, `i`, `o`: every lattice-demo beat reachable from
    /// the keyboard must land in playable space at the App level. The
    /// per-beat waypoint correctness is already covered cell-by-cell in
    /// `lattice_demo_waypoints_do_not_collide_with_scene`; this test
    /// exercises the full keypress→swap→pose path on a representative
    /// beat (Interior — the only one not implicitly retested by `v`/`n`)
    /// to keep the suite under the soft 60s budget.
    #[test]
    fn warp_lattice_debug_jumps_land_in_playable_space() {
        let mut app = App::new(256);
        app.load_lattice_demo_beat(LatticeDemoBeat::Interior);
        assert_player_in_playable_space(&app, "i / load_lattice_demo_beat(Interior)");
    }

    /// Audit `b` (LoadDemoSpectacle): the spectacle gallery places the
    /// player at world_center + 2 cells up. Verify that pose is in playable
    /// space across the seeded spectacle geometry — `seed_demo_spectacle`
    /// includes a viewing pad at world_center per hash-thing-t3zn.2.
    #[test]
    fn warp_b_load_demo_spectacle_lands_in_playable_space() {
        let mut app = App::new(256);
        app.load_demo_spectacle("warp-audit");
        assert_player_in_playable_space(&app, "b / load_demo_spectacle");
    }

    /// Audit `0` (recenter_player): recenter from inside a seeded scene
    /// must not drop the player AABB into a solid cell, and must land on a
    /// surface with floor support (per the t3zn.1 acceptance line).
    #[test]
    fn warp_0_recenter_player_into_lattice_demo_lands_in_playable_space() {
        let mut app = App::new(256);
        // Seed a real scene so the player exists and the world has solid
        // cells around the spawn — exercises the realistic "press 0 mid-demo"
        // path rather than recentering into an empty world.
        app.load_lattice_demo()
            .expect("scene seed must succeed so player_id is set");
        assert!(
            app.recenter_player(),
            "recenter must succeed once a player exists"
        );

        let pid = app.player_id.expect("player should exist after recenter");
        let player = app
            .entities
            .iter()
            .find(|e| e.id == pid)
            .expect("player entity should exist");
        let pos = player.pos;
        let origin = app.world.origin;
        let side = app.world.side() as i64;
        for axis in 0..3 {
            let lo = origin[axis] as f64;
            let hi = (origin[axis] + side) as f64;
            assert!(
                pos[axis] >= lo && pos[axis] < hi,
                "0 / recenter_player: pos[{axis}]={} outside world bounds [{lo}, {hi})",
                pos[axis],
            );
        }
        assert!(
            !player::player_collides(&app.world, &pos),
            "0 / recenter_player: player AABB at {pos:?} overlaps a solid cell",
        );
        assert!(
            player::is_grounded(&app.world, &pos),
            "0 / recenter_player: player at {pos:?} has no floor support",
        );
    }

    /// hash-thing-t3zn.1: positive scan-path test. Blind candidate is
    /// inside a solid column; a grounded surface exists higher in the same
    /// column. Recenter must climb to that surface, not fall back.
    /// Side=64 so the AABB-aware ceiling has enough headroom above the
    /// stone column for the climb to land (player AABB ≈ 7 cells tall).
    #[test]
    fn recenter_player_climbs_buried_column() {
        let mut app = App::new(64);
        let stone = hash_thing::octree::Cell::pack(1, 0).raw();
        let center = app.world_center();
        let cx = center[0].floor() as i64;
        let cz = center[2].floor() as i64;
        let blind_y = (center[1] + 2.0 * CELLS_PER_METER).floor() as i64;
        // Fill a stone column from y=0 through the bottom half of the blind
        // candidate's AABB. The blind AABB occupies cells [blind_y,
        // blind_y+6] inclusive (PLAYER_HEIGHT=6.4); the column tops at
        // cell `blind_y+3` (`0..column_top` excludes `column_top`). The
        // first clear+grounded scan candidate is `search_y = column_top`:
        // its AABB starts at `column_top`, support cell `column_top-1` is
        // the topmost stone, and the cells above are clear.
        let column_top = blind_y + 4;
        for y in 0..column_top {
            for x in (cx - 2)..=(cx + 2) {
                for z in (cz - 2)..=(cz + 2) {
                    app.world.set(
                        sim::WorldCoord(x),
                        sim::WorldCoord(y),
                        sim::WorldCoord(z),
                        stone,
                    );
                }
            }
        }
        // Spawn a player so recenter has a target entity.
        app.reset_scene_entities();
        assert!(app.recenter_player(), "recenter must succeed");

        let pid = app.player_id.expect("player exists");
        let player = app.entities.iter().find(|e| e.id == pid).expect("player");
        let pos = player.pos;
        assert!(
            !player::player_collides(&app.world, &pos),
            "recenter must climb out of the solid column; pos={pos:?}",
        );
        assert!(
            player::is_grounded(&app.world, &pos),
            "recenter must land on the surface above the column; pos={pos:?}",
        );
        assert_eq!(
            pos[1], column_top as f64,
            "recenter must land exactly on the surface above the stone column: \
             pos.y={} blind_y={} column_top={}",
            pos[1], blind_y, column_top,
        );
    }
}
