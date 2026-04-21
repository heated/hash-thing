//! Self-driving windowed-vs-fullscreen `surface_acquire_cpu` measurement.
//!
//! Removes the "needs human at the keyboard" step from `dlse.2.2` by
//! auto-cycling the app through:
//!
//! 1. `WarmupWindowed`  — burn N frames so the compositor / SVDAG upload
//!    / startup transient has settled.
//! 2. `CaptureWindowed` — record M frames of perf into the shared ring.
//! 3. `WarmupFullscreen` — toggle to `Fullscreen::Borderless(None)`, burn
//!    N frames so the mode-switch transient has settled.
//! 4. `CaptureFullscreen` — record M frames.
//! 5. `Done` — emit a side-by-side log line, ask the event loop to exit.
//!
//! Capture length matches `Perf`'s ring capacity (64) so `n=` in the log
//! equals the number of frames observed in the capture phase (no silent
//! windowing inside the ring).
//!
//! Enable via `HASH_THING_ACQUIRE_HARNESS=1`. When enabled, the harness
//! forces a windowed start regardless of `HASH_THING_FULLSCREEN`.

use std::time::Duration;

use crate::perf::Perf;

const DEFAULT_WARMUP_FRAMES: u32 = 60;
// Match Perf's ring capacity (crates/.../perf.rs Ring::CAPACITY = 64) so
// every captured frame contributes to the reported mean/p95. Previously 180
// — the ring silently retained only the tail 64, inflating the true warmup
// window from 60 to ~176 frames and making n= vs capture_frames confusing
// in the log (mixt IMPORTANT #1). Shifts future measurements vs. commit
// 31df10c by whatever the tail-vs-head delta of those extra 116 frames was
// on a given run — in practice small for a steady-state workload.
const DEFAULT_CAPTURE_FRAMES: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    WarmupWindowed,
    CaptureWindowed,
    WarmupFullscreen,
    CaptureFullscreen,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    ClearPerf,
    ToggleFullscreen,
    Exit,
}

/// Mean / p95 / sample-count for one metric, or `None` if no samples landed
/// during the capture window.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricSnapshot {
    pub mean: Duration,
    pub p95: Duration,
    pub samples: usize,
}

impl MetricSnapshot {
    fn from_perf(perf: &Perf, name: &'static str) -> Option<Self> {
        let (mean, p95, samples) = perf.stats(name)?;
        Some(Self { mean, p95, samples })
    }
}

/// Per-phase snapshot pulled from `Perf` at the moment the capture window
/// closes. Each field is `None` if the metric never fired during capture
/// (e.g. `render_gpu` on an adapter without TIMESTAMP_QUERY).
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseStats {
    pub surface_acquire_cpu: Option<MetricSnapshot>,
    pub render_cpu: Option<MetricSnapshot>,
    pub render_gpu: Option<MetricSnapshot>,
    pub submit_cpu: Option<MetricSnapshot>,
    pub present_cpu: Option<MetricSnapshot>,
}

impl PhaseStats {
    fn snapshot(perf: &Perf) -> Self {
        Self {
            surface_acquire_cpu: MetricSnapshot::from_perf(perf, "surface_acquire_cpu"),
            render_cpu: MetricSnapshot::from_perf(perf, "render_cpu"),
            render_gpu: MetricSnapshot::from_perf(perf, "render_gpu"),
            submit_cpu: MetricSnapshot::from_perf(perf, "submit_cpu"),
            present_cpu: MetricSnapshot::from_perf(perf, "present_cpu"),
        }
    }
}

#[derive(Debug)]
pub struct Harness {
    pub phase: Phase,
    frames_in_phase: u32,
    warmup_frames: u32,
    capture_frames: u32,
    pub windowed: Option<PhaseStats>,
    pub fullscreen: Option<PhaseStats>,
}

impl Harness {
    /// Returns `Some(harness)` iff `HASH_THING_ACQUIRE_HARNESS=1` is set.
    pub fn from_env() -> Option<Self> {
        match std::env::var("HASH_THING_ACQUIRE_HARNESS").ok().as_deref() {
            Some("1") => Some(Self::with_frames(
                DEFAULT_WARMUP_FRAMES,
                DEFAULT_CAPTURE_FRAMES,
            )),
            _ => None,
        }
    }

    pub fn with_frames(warmup_frames: u32, capture_frames: u32) -> Self {
        Self {
            phase: Phase::WarmupWindowed,
            frames_in_phase: 0,
            warmup_frames,
            capture_frames,
            windowed: None,
            fullscreen: None,
        }
    }

    /// Advance the state machine by one rendered frame. Returns the list of
    /// side effects the caller must execute (clear perf rings, flip
    /// fullscreen, exit the event loop). Caller invokes *after* the frame's
    /// perf samples have been recorded for the current frame.
    pub fn step(&mut self, perf: &Perf) -> Vec<Action> {
        if self.phase == Phase::Done {
            return Vec::new();
        }
        self.frames_in_phase = self.frames_in_phase.saturating_add(1);
        let mut actions = Vec::new();
        match self.phase {
            Phase::WarmupWindowed => {
                if self.frames_in_phase >= self.warmup_frames {
                    self.phase = Phase::CaptureWindowed;
                    self.frames_in_phase = 0;
                    actions.push(Action::ClearPerf);
                }
            }
            Phase::CaptureWindowed => {
                if self.frames_in_phase >= self.capture_frames {
                    self.windowed = Some(PhaseStats::snapshot(perf));
                    self.phase = Phase::WarmupFullscreen;
                    self.frames_in_phase = 0;
                    // No ClearPerf here: windowed samples survive through the
                    // toggle + fullscreen warmup, then the
                    // WarmupFullscreen -> CaptureFullscreen boundary clears
                    // before any fullscreen capture starts. Asymmetric vs
                    // the WarmupWindowed -> CaptureWindowed boundary on
                    // purpose (mixt IMPORTANT #2).
                    actions.push(Action::ToggleFullscreen);
                }
            }
            Phase::WarmupFullscreen => {
                if self.frames_in_phase >= self.warmup_frames {
                    self.phase = Phase::CaptureFullscreen;
                    self.frames_in_phase = 0;
                    actions.push(Action::ClearPerf);
                }
            }
            Phase::CaptureFullscreen => {
                if self.frames_in_phase >= self.capture_frames {
                    self.fullscreen = Some(PhaseStats::snapshot(perf));
                    self.phase = Phase::Done;
                    actions.push(Action::Exit);
                }
            }
            Phase::Done => {}
        }
        actions
    }

    /// Multi-line report summarising the two captured phases. Intended for
    /// `log::info!` at the point the harness returns `Action::Exit`.
    pub fn report(&self) -> String {
        let windowed = self.windowed.unwrap_or_default();
        let fullscreen = self.fullscreen.unwrap_or_default();
        let verdict = verdict(windowed.surface_acquire_cpu, fullscreen.surface_acquire_cpu);
        let mut out = String::from("dlse.2.2 acquire-harness report\n");
        out.push_str(&render_phase("windowed", &windowed));
        out.push_str(&render_phase("fullscreen", &fullscreen));
        out.push_str(&render_delta(
            windowed.surface_acquire_cpu,
            fullscreen.surface_acquire_cpu,
        ));
        out.push_str(&format!("  verdict    {verdict}\n"));
        out
    }
}

// Label column widths. PHASE_LABEL_W fits "fullscreen" (10 chars);
// METRIC_LABEL_W fits "surface_acquire_cpu" (19). Keeping them as named
// constants makes future additions (e.g. "mid-transition") a one-line
// change instead of touching every row (mixt NIT #7).
const PHASE_LABEL_W: usize = 10;
const METRIC_LABEL_W: usize = 19;

fn render_phase(label: &str, stats: &PhaseStats) -> String {
    let mut s = String::new();
    for (name, snap) in [
        ("surface_acquire_cpu", stats.surface_acquire_cpu),
        ("render_cpu", stats.render_cpu),
        ("render_gpu", stats.render_gpu),
        ("submit_cpu", stats.submit_cpu),
        ("present_cpu", stats.present_cpu),
    ] {
        match snap {
            Some(snap) => s.push_str(&format!(
                "  {label:<PHASE_LABEL_W$} {name:<METRIC_LABEL_W$} = {:5.2}/{:5.2} ms (n={})\n",
                snap.mean.as_secs_f64() * 1000.0,
                snap.p95.as_secs_f64() * 1000.0,
                snap.samples,
            )),
            None => {
                s.push_str(&format!(
                    "  {label:<PHASE_LABEL_W$} {name:<METRIC_LABEL_W$} = (no samples)\n"
                ));
            }
        }
    }
    s
}

fn render_delta(windowed: Option<MetricSnapshot>, fullscreen: Option<MetricSnapshot>) -> String {
    let (Some(w), Some(f)) = (windowed, fullscreen) else {
        return String::from(
            "  delta      surface_acquire_cpu = (insufficient samples, skipping delta)\n",
        );
    };
    let w_ms = w.mean.as_secs_f64() * 1000.0;
    let f_ms = f.mean.as_secs_f64() * 1000.0;
    let delta_ms = f_ms - w_ms;
    let pct = if w_ms > 0.0 {
        delta_ms / w_ms * 100.0
    } else {
        0.0
    };
    format!(
        "  delta      surface_acquire_cpu mean: {:+.2} ms ({:+.1}%)\n",
        delta_ms, pct,
    )
}

// Verdict thresholds tied to the dlse.2.2 acceptance rubric: "drop to ≤5 ms
// → ship fullscreen-default" (mixt NIT #6).
const SHIP_ACCEPT_MS: f64 = 5.0;
const INVESTIGATE_CEILING_MS: f64 = 10.0;
const INVESTIGATE_FLOOR_MS: f64 = 15.0;
const NOISE_BAND_MS: f64 = 2.0;

/// Turn the windowed vs fullscreen acquire numbers into a short
/// recommendation.
fn verdict(windowed: Option<MetricSnapshot>, fullscreen: Option<MetricSnapshot>) -> &'static str {
    let (Some(w), Some(f)) = (windowed, fullscreen) else {
        return "insufficient samples";
    };
    let w_ms = w.mean.as_secs_f64() * 1000.0;
    let f_ms = f.mean.as_secs_f64() * 1000.0;
    if f_ms <= SHIP_ACCEPT_MS && w_ms > SHIP_ACCEPT_MS {
        "fullscreen meets ≤5 ms acceptance — ship fullscreen-default"
    } else if f_ms <= INVESTIGATE_CEILING_MS && w_ms > INVESTIGATE_FLOOR_MS {
        "big drop but not ≤5 ms — investigate residual stall before shipping"
    } else if (w_ms - f_ms).abs() < NOISE_BAND_MS {
        "fullscreen ≈ windowed — acquire stall is not compositor/windowed-path; queue off-surface diagnostic"
    } else if f_ms > w_ms + NOISE_BAND_MS {
        "fullscreen WORSE than windowed — unexpected; do not ship fullscreen-default"
    } else {
        "modest improvement — queue off-surface diagnostic (step 3)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn new_starts_in_warmup_windowed() {
        let h = Harness::with_frames(2, 2);
        assert_eq!(h.phase, Phase::WarmupWindowed);
        assert!(h.windowed.is_none());
        assert!(h.fullscreen.is_none());
    }

    #[test]
    fn full_cycle_emits_expected_action_sequence() {
        // warmup=2, capture=2. Total frames per cycle = 2 warmup + 2 capture
        // = 4 per mode, 8 total. Expected actions, one per frame:
        // frame 1 (warmup-windowed): none
        // frame 2 (warmup-windowed done): ClearPerf
        // frame 3 (capture-windowed): none
        // frame 4 (capture-windowed done): ToggleFullscreen
        // frame 5 (warmup-fullscreen): none
        // frame 6 (warmup-fullscreen done): ClearPerf
        // frame 7 (capture-fullscreen): none
        // frame 8 (capture-fullscreen done): Exit
        let mut h = Harness::with_frames(2, 2);
        let mut perf = Perf::new();
        // Give the capture phase a sample so PhaseStats isn't empty.
        perf.record("surface_acquire_cpu", ms(10));

        let mut observed: Vec<Action> = Vec::new();
        for _ in 0..8 {
            observed.extend(h.step(&perf));
        }
        assert_eq!(
            observed,
            vec![
                Action::ClearPerf,
                Action::ToggleFullscreen,
                Action::ClearPerf,
                Action::Exit,
            ]
        );
        assert_eq!(h.phase, Phase::Done);
    }

    #[test]
    fn step_after_done_is_noop() {
        let mut h = Harness::with_frames(1, 1);
        let perf = Perf::new();
        // Drive to Done: 1 warmup + 1 capture = 2 frames per mode, 4 total.
        for _ in 0..4 {
            let _ = h.step(&perf);
        }
        assert_eq!(h.phase, Phase::Done);
        let actions = h.step(&perf);
        assert!(actions.is_empty());
    }

    #[test]
    fn snapshot_captures_perf_means_per_phase() {
        // Simulate the windowed phase recording a slow acquire and the
        // fullscreen phase recording a fast one; assert the per-phase
        // snapshot follows the perf state at the moment `step` decides to
        // snapshot.
        let mut h = Harness::with_frames(1, 1);
        let mut perf = Perf::new();

        // Warmup-windowed
        perf.record("surface_acquire_cpu", ms(1));
        let _ = h.step(&perf); // -> CaptureWindowed, ClearPerf (actions include it, but caller is simulated — we also clear explicitly).
        perf.clear();
        // Capture-windowed sample.
        perf.record("surface_acquire_cpu", ms(25));
        let _ = h.step(&perf); // -> WarmupFullscreen, ToggleFullscreen
        assert!(h.windowed.is_some());
        let w = h.windowed.unwrap().surface_acquire_cpu.unwrap();
        assert_eq!(w.mean, ms(25));
        assert_eq!(w.samples, 1);

        // Warmup-fullscreen
        perf.clear();
        perf.record("surface_acquire_cpu", ms(99));
        let _ = h.step(&perf); // -> CaptureFullscreen, ClearPerf
        perf.clear();
        perf.record("surface_acquire_cpu", ms(3));
        let _ = h.step(&perf); // -> Done, Exit
        assert!(h.fullscreen.is_some());
        let f = h.fullscreen.unwrap().surface_acquire_cpu.unwrap();
        assert_eq!(f.mean, ms(3));
    }

    fn snap(mean_ms: u64) -> MetricSnapshot {
        MetricSnapshot {
            mean: ms(mean_ms),
            p95: ms(mean_ms + 2),
            samples: 32,
        }
    }

    #[test]
    fn verdict_recommends_ship_when_fullscreen_drops_below_5ms() {
        assert_eq!(
            verdict(Some(snap(25)), Some(snap(3))),
            "fullscreen meets ≤5 ms acceptance — ship fullscreen-default"
        );
    }

    #[test]
    fn verdict_queues_off_surface_when_fullscreen_no_help() {
        assert_eq!(
            verdict(Some(snap(25)), Some(snap(25))),
            "fullscreen ≈ windowed — acquire stall is not compositor/windowed-path; queue off-surface diagnostic"
        );
    }

    #[test]
    fn verdict_flags_fullscreen_regression() {
        let v = verdict(Some(snap(10)), Some(snap(25)));
        assert!(v.contains("WORSE"), "expected regression verdict, got: {v}");
    }

    #[test]
    fn verdict_big_drop_not_ship() {
        // 25 → 8 ms: huge drop but not ≤5 ms. Want "investigate", not "ship".
        let v = verdict(Some(snap(25)), Some(snap(8)));
        assert!(
            v.contains("investigate"),
            "expected investigate verdict, got: {v}"
        );
    }

    #[test]
    fn verdict_handles_missing_snapshots() {
        assert_eq!(verdict(None, Some(snap(3))), "insufficient samples");
        assert_eq!(verdict(Some(snap(3)), None), "insufficient samples");
    }

    #[test]
    fn report_includes_both_phases_and_verdict() {
        let mut h = Harness::with_frames(1, 1);
        h.windowed = Some(PhaseStats {
            surface_acquire_cpu: Some(snap(25)),
            render_cpu: Some(snap(26)),
            render_gpu: Some(snap(0)),
            submit_cpu: Some(snap(0)),
            present_cpu: Some(snap(0)),
        });
        h.fullscreen = Some(PhaseStats {
            surface_acquire_cpu: Some(snap(3)),
            render_cpu: Some(snap(4)),
            render_gpu: Some(snap(0)),
            submit_cpu: Some(snap(0)),
            present_cpu: Some(snap(0)),
        });
        let r = h.report();
        assert!(r.contains("windowed"), "missing windowed label:\n{r}");
        assert!(r.contains("fullscreen"), "missing fullscreen label:\n{r}");
        assert!(
            r.contains("surface_acquire_cpu"),
            "missing metric name:\n{r}"
        );
        assert!(r.contains("delta"), "missing delta line:\n{r}");
        assert!(
            r.contains("ship fullscreen-default"),
            "missing verdict:\n{r}"
        );
    }

    #[test]
    fn report_handles_absent_phases_without_panic() {
        let h = Harness::with_frames(1, 1);
        // No captured stats at all.
        let r = h.report();
        assert!(
            r.contains("insufficient samples"),
            "expected insufficient-samples verdict, got:\n{r}"
        );
    }
}
