# Render-perf direction (post-2w1u, 2026-04-26, v3)

**Status:** Forward-looking roadmap. v3 incorporates Triple plan review (Claude / Codex / Gemini, see `Review record` section) and a careful re-reading of perf paper §3.9–§3.12 that I should have done before drafting v1. Several conclusions reversed since v2; flagged inline.

## TL;DR (rewritten v3)

Tonight's 30-second captures sampled the documented `dlse.2.2.3` **warmup transient**, not steady state. Per perf paper §3.9 / §3.12, `surface_acquire_cpu` settles from ~25ms to ~11ms after ~30 seconds at 64³, which is the 60 FPS regime — not a 30 FPS pin. The "Apple Metal compositor pacing" framing is wrong on both axes: the mechanism (compositor refuted at §3.9 step 2 by fullscreen-vs-windowed comparison; `maximumDrawableCount` and `displaySyncEnabled` swept null) AND the symptom (the 25-30ms is a transient).

**The actual perf concern that drove tonight's session is at 256³+ scenes** — that's where edward reported the visible-FPS issue ("rendering wasn't that great even at 50% and 256³"). Per §3.11–§3.12, `surface_acquire_cpu` at 256³/100% is 77ms and at 1024³/100% is 142ms — these are NOT warmup, they are pipeline fill scaling with world size and resolution. That's a real GPU/upload pressure issue, and it's the right place to direct the optimization budget.

**Roadmap (revised order):**

1. **Phase 0: extend perf paper §3.13 with steady-state captures at 256³ and 512³.** Tonight's 64³ captures don't generalize; the dlse.2.2.3 warmup confounds them. Need 60s+ runs at edward's actual problem scenes.
2. **Phase 1: cheaper rays.** Promoted back to highest-leverage. The bottleneck at 256³+ is pipeline fill, which is exactly what cheaper rays attack. Caveat: dubiousconst282's numbers are sparse-64-tree; sparse-8-tree adapt with structural changes (see Phase 1 sub-bead notes).
3. **Phase 2: investigate `me6i` (frame_total growth over time at 64³ frozen-sim).** Likely the same warmup transient documented in dlse.2.2.3, but the observation that it persists with sim FROZEN is a new wrinkle worth verifying. Cheap to check.
4. **Phase 3: adaptive resolution.** Cross-platform polish, lower priority. Helps non-Mac targets where the GPU is genuinely the bottleneck after Phase 1 lands.
5. **Phase 4 (deferred design gate): async surface acquire.** Demoted to "uncertain — three prior swapchain/acquire fixes came up null" (`dlse.2.2 exp#4`, `dlse.2.2.2`, moss `displaySyncEnabled` sweep). The async-acquire bet is that an architectural change escapes what those swept; we can't promise it does. Only revisit if Phase 1 lands and 256³+ steady-state still has a non-pipeline-fill stall.

**Reversals from v2:**
- v2 said "demote cheaper rays from highest-leverage on Mac." v3 says **rays are highest-leverage everywhere** — the Mac-specific compositor-bound framing was wrong.
- v2 said "Phase 2-Mac async surface acquire is the highest-leverage Mac fix." v3 says **async acquire is uncertain, deferred.**
- v2 framed Phase 0 as a fresh experiment. v3 frames it as **continuation of perf paper §3.9–§3.12** with the focus shifted to longer-horizon captures at edward's actual problem scenes.

---

## What we measured tonight (with corrected interpretation)

Four 30s headless captures on M2, default scene at 64³, release profile (LTO + opt-3). Logs preserved in `/tmp/2w1u-{autovsync,immediate,frozensim,offsurface}.log` on edward's machine.

| Mode | frame_total | surface_acquire_cpu | submit_cpu | render_gpu (ts query) |
|------|-------------|---------------------|------------|------------------------|
| AutoVsync (sim on)              | 50–80ms growing | 47–80ms     | 0.3ms       | 0.1ms |
| Immediate (sim on)              | 40–50ms growing | 37–46ms     | 0.3ms       | 0.1ms |
| AutoVsync (sim FROZEN, Gen 0)   | 33→70ms growing | 32→69ms     | 0.3ms       | 0.1ms |
| OFF_SURFACE=1 (sim on)          | 14–42ms         | **0.01ms**  | **18–40ms** | **9–26ms** |

### What this actually tells us, in light of §3.9–§3.12

- **All four runs sampled the dlse.2.2.3 warmup transient.** Per §3.9 (onyx 2026-04-21), `surface_acquire_cpu` follows a documented climb-plateau-step-down shape over 0–34s on M2 at 256³: climb 23→34ms in t=2–14, kink at memo cache trim (t=14–16), plateau ~27ms through t=26, then step-down to ~11ms by t=34. Tonight's 30s captures captured exactly the climb + plateau + start-of-step-down phase. **Tonight's 30→80ms surface_acquire growth is the same transient, not a new finding.**
- **`me6i` (frame_total growth over time)** that I filed earlier today is therefore likely a duplicate of dlse.2.2.3. The novelty in tonight's run is that it persisted with sim FROZEN — dlse.2.2.3 attributed the warmup to "CPU-GPU pipeline fill + memo-cache warmup + sim-step-rate stabilisation." If the warmup persists with sim frozen, the "memo-cache warmup" and "sim-step-rate" components shouldn't apply. So either (a) tonight's frozen-sim run has its own pipeline-fill mechanism, or (b) the dlse.2.2.3 attribution is incomplete. Worth a 5-minute Phase-2 investigation.
- **High confidence: present mode is not the lever.** AutoVsync vs Immediate within noise. Confirms §3.9 step 1.
- **High confidence: per-frame compute pass is ~0.16ms.** wgpu Metal HAL timestamps are honest GPU device-tick time. `bench_raycast_256` headless = 0.19ms (`dlse.2.3`).
- **Medium confidence: 9–26ms off-surface render_gpu reading is queue backpressure.** Per §3.9 the more reliable reading is `render_pass_gpu` (which stays <0.35ms on or off surface) — that's the strongest direct evidence. The Phase-0 wider-bracket experiment is required for full disambiguation.
- **Wrong framing in v2 (corrected here): "Apple Metal compositor pacing".** §3.9 step 2 already refuted compositor (fullscreen worse than windowed). Don't say "compositor" without isolating the specific mechanism, which §3.9–§3.12 says is unidentified after sweeping every wgpu-exposed knob.

---

## Phase 0: extend §3.13 with steady-state captures at edward's problem scenes

**Bead:** `9k4w` (existing, P1) **— scope corrected to "Phase-0 measurement extension," not "M2 shader-bound."** The original bead title was based on v1's wrong diagnosis. Will re-title after this v3 lands.

§3.9 / §3.12 captured 256³ but the data points are sparse and edward's actual perf complaint ("256³ at 50% feels slow") needs a steady-state characterization that goes 60–120s past warmup, ideally captured at 256³ + 512³ + the four corners of the dlse.2.2.3 warmup shape.

**Procedure** (cribbed from §3.9 dlse.2.2 exp#4 protocol, extended):

- Hardware: M2 MBA 16GB, 60Hz external display, perf profile (`--profile perf` per `xer4` rename — *NOT* release; future captures use perf for consistency with project policy).
- Worlds: 64³ (control), 256³, 512³ (edward's reported problem-scale).
- Render scale: 0.5 (default-ish), 1.0 (native; for §3.12 alignment).
- Mode: windowed + fullscreen (per §3.9 step 2 — fullscreen avoids WindowServer, partial control on the warmup mechanism).
- Run length: **120s minimum** per arm, capturing the warmup-to-steady transition (per dlse.2.2.3 the transition ends ~t=34; need at least 60s steady to characterize it).
- Capture: every metric in `Perf` (frame_total, surface_acquire_cpu, submit_cpu, present_cpu, render_gpu, render_pass_gpu) + `submit_fence.last_pipeline` + memory-stats + memo-cache stats per generation.
- Use the existing `acquire_harness.rs` from §3.9 (it already does parallel-column capture).

**Acceptance:**
- §3.13 (or new §3.14) extended with the steady-state numbers.
- Steady-state surface_acquire_cpu @ 256³ / 50% / windowed reported with mean ± p95 over the second half of a 120s run.
- If steady-state @ 256³/50% is in the 60 FPS regime (≤17ms frame_total): edward's complaint is the warmup transient, and the optimization target shifts to **shorten warmup** rather than reduce steady-state cost. File a new bead.
- If steady-state @ 256³/50% is NOT in the 60 FPS regime: confirms genuine scaling cost, and Phase 1 (cheaper rays) is the right next step.
- Either way: the data is preserved in the perf paper for future investigation.

**Wider-bracket instrumentation** (Phase-0b, splittable from Phase-0a):
- Add a "whole-pipeline" timestamp pair (one before any encode, one after the final pass).
- Use `TIMESTAMP_QUERY_INSIDE_ENCODERS` for finer brackets (start-of-encode, before/after compute, before/after blit, before submit). Each becomes its own metric.
- Document the surface-vs-off-surface duality in §3.13 explicitly so the next investigator doesn't repeat my mistake.

---

## Phase 1: cheaper rays (promoted to highest-leverage in v3)

**Beads:** `9k4w.1` (ancestor-stack memo — already-landed, see audit below), `9k4w.2` (empty-cell bitmask — partial; the 64-bit grandchildren-mask variant still TODO), `9k4w.3` (ray-octant mirror — already-landed), `9k4w.4` (Apple register flatten — open).

> **Audit update — `k6tm` (cairn) + `e0no` (flint cross-check) 2026-04-27.** Three of the four cheaper-rays sub-beads are already on main; current measurements bake in their benefit:
>
> - **9k4w.3 ray-octant mirror — LANDED.** `crates/ht-render/src/svdag_raycast.wgsl:310-315` (mirror_mask construction) + `:478-479,654` (XOR un-mirror at child lookup). Landing commit `ce97c64` ("render: Laine-Karras Stage 2 ray-octant mirroring"). → **`5m36` closed.**
> - **9k4w.1 ancestor-stack memo — LANDED.** Within-ray pop-to-deepest-still-containing-ancestor at `:396-407` + per-ray register-resident `stack_node[]/stack_min[]/stack_half[]/stack_cmask[]` arrays. Landing commit `e02c55a` (original SVDAG pipeline) extended by `44f7630` (m1f.7.3, stack_cmask caching). The dubiousconst282 article confirms this is the ~1.9× optimization (per-ray stack in registers, not cross-ray) — duke-henry's "Next ray in the tile restarts" prose was a loose paraphrase. → **`5sjh` closed.**
> - **9k4w.2 empty-cell bitmask — PARTIAL.** The 8-bit immediate-children-occupancy mask is landed (`:482-485` cmask pre-check before child_slot read, `44f7630` m1f.7.3; `:595-657` inner-DDA skip-empty-children loop, `d0c8b9a` x5w.1). The 64-bit *grandchildren* occupancy mask described in the bead — single bitscan in the parent skips an entire subtree without descending — is **still TODO**. The +22% memory bloat caveat applies only to that 64-bit grandchildren variant. → **`iqx7` stays open with refined scope.**
> - **9k4w.4 Apple register flatten — open, out of audit scope.**
>
> **Roadmap impact:** Remaining Phase 1 work is `iqx7` refined (64-bit grandchildren bitmask) + `wck2` (Apple register flatten). The ~1.5×–2.2× combined-leverage estimate below was assuming all four sub-beads still to land; subtract the .1 + .3 already-banked contribution + the 8-bit immediate-children part of .2 already-banked, and the **incremental** Phase 1 yield over current main is closer to **1.1×–1.3× at 256³** (driven mostly by iqx7-refined and 9k4w.4 on Mac).

Sources: [dubiousconst282 2024-10-03](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/) (concrete cycle counts on integrated GPU, 2.66× stacked), [Aokana 2025 (arxiv 2505.02017)](https://arxiv.org/html/2505.02017v1) (structural confirmation on Vulkan).

### Why cheaper rays are highest-leverage in v3

- §3.11 documented "the ramp is pipeline fill" at 1024³/100% — the GPU is doing real work that scales with world size and resolution. Cheaper rays directly reduce that.
- The Mac-leverage caveat from v2 ("compositor-bound, GPU is idle") was based on the wrong mechanism. Per §3.9, the steady-state is in the 60 FPS regime where the GPU's per-frame work IS the limit.
- Even in the warmup-transient interpretation: shorter rays → less pipeline-fill work → faster warmup completion. Win-win.
- Cross-platform users (non-Mac) see the full ~2.6× win regardless of M2-specific oddities.

### Sparse-8 vs sparse-64 — the structural caveat (Gemini reviewer surfaced)

dubiousconst282 measured on a sparse-64-tree (depth ~4–5). Our SVDAG is sparse-8-tree (depth 8 at 256³, depth ~12 at 4096³). This isn't a constants caveat — it's a structural one, and the impact on each technique differs:

- **9k4w.1 ancestor-stack memo (8 pointers, 1.9× claim).** Fits depth ≤8 cleanly (256³). At depth 12 (4096³), 8 pointers cover only the bottom of the path; the upper levels still re-traverse from root. **Most likely outcome at 256³: 1.5–1.8× (close to claim). At 4096³: 1.2–1.4×.** Update bead description; tighten acceptance to a measured-on-our-tree delta, not a "±15% of dubiousconst282" gate.
- **9k4w.2 empty-cell bitmask (1.26× claim).** Requires storing grandchildren occupancy in each interior node. Our nodes are 36 bytes; adding a 64-bit bitmask is +22% memory. Either accept the bloat (acceptable but worth flagging), or use an 8-bit children-occupancy mask which is cheaper but lower yield. **Most likely outcome: 1.1–1.2× with smaller mask, 1.3× with full bitmask + memory bloat.** Trident plan review on the design choice.
- **9k4w.3 ray-octant mirror (1.1× claim).** Structural-mismatch-immune (works on any sparse N-tree). Yield should match within ±5%.
- **9k4w.4 Apple register flatten (Apple-specific, mid-yield).** Independent of dubiousconst282. Flatten the recursive traversal helper to fix M-series occupancy collapse past ~32 registers/thread.

**Updated combined leverage estimate:** 1.7–2.2× total at 256³, 1.3–1.7× at 4096³. Still substantial. v2's "2.6× total" was sparse-64-tree's number; v3 honest estimate is lower for our tree.

### Acceptance per sub-bead (updated)

- Phase 0 instrumentation in place.
- Pre-optimization `render_gpu_total` reading captured **on M2 + at least one non-Mac target** (Linux+NV, Linux+AMD, or Windows+integrated).
- Optimization landed via /ship-auto + Triple review.
- Post-optimization reading shows the **internal-baseline-delta** (not a "±15% of dubiousconst282" gate; that was Codex reviewer's correct critique).
  - 9k4w.1: ≥1.4× compute-pass speedup at 256³ on the non-Mac target.
  - 9k4w.2: ≥1.1× on top of 9k4w.1.
  - 9k4w.3: ≥1.05× on top of 9k4w.1+.2.
  - 9k4w.4: ≥1.2× on M2 specifically (Apple-occupancy yield); ≤5% regression on non-Mac.
- Visual regression: `--dump-frame` output within FP-tolerant pixel-percentage threshold (Claude reviewer's correct critique — byte-equivalent is too strict). Specify: ≤0.1% pixels differ by more than 4 LSBs in any channel.

---

## Phase 2: characterize / fix `me6i` warmup transient

**Bead:** `me6i` (existing, P1) — **likely duplicate of dlse.2.2.3 (closed)** but worth a 5-minute check.

The frozen-sim run captured a NEW signal: `frame_total` climbs 32→70ms over 30s with **zero changing scene state**. dlse.2.2.3 attributes the warmup to "CPU-GPU pipeline fill + memo-cache warmup + sim-step-rate stabilisation." With sim frozen, the latter two should not apply. Either:

- (a) dlse.2.2.3 attribution is incomplete and we need to characterize a fourth mechanism. Possible candidates: drawable-pool churn, retained Metal command buffer accumulation, or a subtle wgpu state-cache inflation.
- (b) Tonight's frozen-sim run still has CPU-GPU pipeline fill (the renderer keeps doing per-frame work even with sim frozen — full-frame compute pass + blit + overlay every frame).

(b) is the boring null. (a) is the interesting case. Investigation step:

- Re-run frozen-sim for 120s (long enough to capture warmup → steady → continued growth or stabilization).
- If frame_total stabilizes around ~11ms after t=34: it's the documented warmup transient (b). Close `me6i` as duplicate of dlse.2.2.3.
- If frame_total continues growing or stays elevated: novel mechanism (a). File a new bead with the new shape, drive Mac-side Instruments investigation.

**Acceptance:** one of the two outcomes documented in `me6i` comment.

**Why not parallel with Phase 1:** weakly serial. If me6i shows a leak we can fix cheaply, it changes the steady-state baseline against which Phase-1 optimizations are scored. Better to know first.

---

## Phase 3: adaptive resolution

**Bead:** `adp-res` (filed tonight, P2). Cross-platform polish.

Static GPU-class table at startup + dynamic ramp toward target frame budget. M-series → 0.6 default. Discrete dGPU → 1.0. Other integrated → 0.7. `HASH_THING_RENDER_SCALE` always overrides.

**Acceptance** (Codex reviewer's correct critique applied — name the reference adapter):
- Default Mac M2 launch hits ≤17ms frame_total (60 FPS budget) at chosen render_scale within first 5s of warm frames at 256³ default scene.
- Reference non-Mac: launch on either an Nvidia RTX 3060 (or equivalent ~20 TFLOPs class) or an Intel Iris Xe (or equivalent ~2 TFLOPs class) hits ≤17ms at chosen render_scale at 256³.
- Adaptive ramp converges in <2s with hysteresis ≤±5% over a 10s steady-state window (specified measurement window per Claude reviewer).
- Trident plan review minimum.

**Depends on:** Phase 0 (accurate frame_total; without it the ramp drives off a misleading metric).

---

## Phase 4 (deferred design gate): async surface acquire

**Bead:** `m59h` (filed tonight, P2 design-gated). **Not promoted to Phase 1 in v3** as v2 had it.

**Why deferred** (Claude reviewer's adversarial critique applied):

- §3.9 swept `maximumDrawableCount` ∈ {2,3,4} — null.
- moss 2026-04-15 swept `displaySyncEnabled` ∈ {true,false} — null.
- `dlse.2.2.2` shipped late-acquire reorder (Apple's documented best practice) — null.
- `dlse.2.2 exp#4` swept `frame_latency` — null.

**Async surface acquire is a fifth swapchain-pacing fix.** It's an architectural change rather than an env-knob swap, so it's not strictly redundant — but the prior-null record is enough to deprioritize until we have a reason to believe it would escape what those tests already eliminated.

**Promotion criterion:** Phase 0 and Phase 1 land. If 256³+ steady-state still shows a `surface_acquire_cpu` stall that scales with resolution + isn't accounted for by render_gpu_total + isn't a warmup transient — then async acquire becomes the only remaining lever and gets promoted to P1 with its design phase.

**Acceptance** (Claude reviewer's correct critique applied — explicit gate, not just promotion criterion):
- frame_total ≤ 17ms mean over 60s warm steady-state on M2 windowed at 256³/50%, **OR** documented residual stall + close as can't-fix.

**Risks** (Gemini + Claude reviewers added):
- **Drawable pool exhaustion.** If the stall is a compositor-throughput limit, async acquire will exhaust the 3-drawable swapchain and block the main thread on the channel anyway. The async thread doesn't bypass throughput limits.
- **`me6i` interaction.** If the warmup transient is retained-drawable accumulation, async acquire will retain MORE drawables and may make me6i worse before better.
- **Architectural scope.** This is a thread/resource ownership rewrite, not a "Phase 4 task." Likely a multi-bead epic, not a single bead. Update m59h to reflect.

---

## Standardized measurement protocol (Codex reviewer)

All Phase-0+ measurements use this protocol or override with rationale:

- **Build profile:** `--profile perf` (NOT `--release` and NOT `bench` — see `xer4` for the rename).
- **Hardware:** M2 MBA 16GB, 60Hz external display (windowed) OR M2 internal display (fullscreen).
- **Worlds:** 64³ (control), 256³ (edward's reported problem-scale), 512³ (perf budget edge).
- **Render scale:** 0.5 (default-ish), 1.0 (native).
- **Run length:** 120s minimum per arm. First 34s discarded as warmup (per dlse.2.2.3).
- **Capture:** every metric in `Perf` + memory + memo-cache stats per generation. JSON or CSV serialization for cross-run comparison.
- **Statistic:** mean ± p95 over the second half (post-warmup).
- **Repeats:** 3 independent runs per arm (reduce timer noise).

**Convention:** when a bead reports a perf-relevant measurement, it cites this protocol or explicitly notes deviation.

---

## Cross-cutting: efficiency hygiene

**Bead:** `eff-hyg` (filed tonight, P3).

Standing rule: every PR adding a per-frame code path also adds a `perf::record()` timer. Codified in `.agents/skills/review-tiers/SKILL.md` as a Triple-review checkbox.

CPU-side gap closure (separate from Phase 0b's GPU-side gap):
- winit `AboutToWait` / `RedrawRequested` dispatch.
- scene-state mutation.
- perf-bookkeeping path itself.

---

## Risk register (revised v3)

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Phase 0 confirms 256³/50% steady-state IS in the 60 FPS regime → edward's complaint is the warmup transient | Medium | Reframe optimization target to "shorten warmup" rather than "reduce steady-state cost." Phase 1 still helps. |
| sparse-8 vs sparse-64 knocks down 9k4w.1's yield to ~1.5× and 9k4w.2's to ~1.2× (Medium-High) | Medium-High (Claude+Gemini) | Per-bead acceptance now uses internal-baseline-delta on M2+non-Mac, not a "±15% of dubiousconst282" gate. Reduced expected combined yield from 2.6× to 1.7–2.2×. |
| 9k4w.2 empty-cell bitmask requires +22% memory bloat in SVDAG node format | Medium | Trident plan review on the design choice. May ship a smaller 8-bit mask variant first. |
| Phase 0 implementer prematurely closes as redundant with §3.9 | Low-Medium (Claude) | Acceptance explicitly requires longer-horizon (120s+) captures at 256³/512³, NOT a re-run of §3.9-step-4. |
| async-acquire × `me6i` interaction (retains more drawables → makes leak worse) | Medium (Claude+Gemini) | Phase 4 acceptance gates on `me6i` outcome from Phase 2. |
| async-acquire exhausts 3-drawable pool, blocks main thread anyway | Medium-High (Gemini) | Phase 4 acceptance specifies measured frame_total improvement, not theoretical thread-overlap. If the math doesn't work, document and close. |
| Phase 4 architectural scope is bigger than "Phase 4 task" suggests | High (Codex) | m59h treated as an epic, not a task. Design phase decomposes into sub-beads before implementation begins. |
| Mac-specific tooling (Instruments, Xcode Frame Capture) not available to headless seats | Medium (Codex) | `me6i` and Phase 4 explicitly flagged as "edward-driven or hand-off to a Mac-equipped seat." |
| Phase 1 optimizations conflict with `cswp` chunked LOD | Low | Phase 1 is per-ray; cswp restructures the DAG. Mostly orthogonal. |
| Phase 2-rays optimizations regress non-Mac targets | Low | 9k4w.4 acceptance explicitly requires ≤5% regression on at least one non-Mac target. |

---

## Review record

This document received Triple plan review (Claude + Codex + Gemini), 2026-04-26. Reviewer outputs preserved in `.ship-notes/render-perf-direction-review-{claude,codex,gemini}.md` (gitignored locally; if you need them, ask cairn).

**Synthesis of reviewer asks (v2 → v3):**

| Ask | Reviewer | v3 status |
|-----|----------|-----------|
| Drop "Apple Metal compositor pacing" mechanism claim | Claude+Codex+Gemini | ✅ done; replaced with §3.9 cite + "unidentified mechanism, multiple priors null" |
| Cite dlse.2.2.2, dlse.2.2 exp#4, displaySyncEnabled sweep, §3.9, §3.10, §3.11, §3.12 | Claude | ✅ done |
| Frame Phase 0 as continuation, not invention | Claude | ✅ done (now "extend §3.13 with steady-state captures") |
| Tonight's measurements were warmup transient, not steady state | (Implicit; surfaced via §3.9 reading) | ✅ flagged in TL;DR + What we measured |
| Add measurement protocol section | Codex | ✅ done |
| Use `--profile perf` not `release` for future captures | Codex | ✅ done |
| Phase 1 explicit acceptance criterion (not just promotion) for async-acquire | Claude | ✅ added in Phase 4 |
| Reorder: rays should not be deprioritized just because of mistaken Mac-compositor diagnosis | Claude+Gemini | ✅ done (rays promoted to Phase 1) |
| Sparse-8 vs sparse-64 caveats (depth-12 won't fit 8 pointers; bitmask requires node format change) | Claude+Gemini | ✅ added in Phase 1 |
| Internal-baseline-delta acceptance, not "±15% of dubiousconst282" | Codex+Claude | ✅ done |
| Visual regression: pixel-tolerance threshold, not byte-equivalent | Claude | ✅ done (≤0.1% pixels differ by ≤4 LSBs/channel) |
| Phase 4 = epic, not task | Codex | ✅ flagged as risk + acceptance updates |
| Async-acquire × me6i interaction risk | Claude | ✅ added |
| Async-acquire could exhaust drawable pool | Gemini | ✅ added |
| Phase 0 implementer closing as redundant | Claude | ✅ added |
| Mac-tooling availability for headless seats | Codex | ✅ added |
| Reference adapter for Phase 3 acceptance | Codex | ✅ named (RTX 3060 / Iris Xe) |
| Adaptive-ramp measurement window | Claude | ✅ specified (10s steady-state) |

---

## Cross-references (revised v3)

**Prior measurements (read these before claiming any 9k4w child):**
- `docs/perf/svdag-perf-paper.md` §3.7–§3.12 — present-path investigation, dlse.2.2 candidate enumeration
- `docs/perf/svdag-perf-paper.md` §3.9 specifically — render-pass attribution (dlse.2.4) — **REVISES the "real GPU frame time" inference; tonight's session validated this was the right revision**
- `docs/perf/svdag-perf-paper.md` §3.11 — GPU cost at 1024³/100% is real; "the ramp is pipeline fill" (`hash-thing-dlse.3`)
- `docs/perf/svdag-perf-paper.md` §3.12 — scene-matrix measurement + §3.11 correction

**Beads, organized by Phase:**
- Phase 0: `9k4w` (existing P1, scope corrected) + `9k4w.0` Phase-0a disambiguating experiment + Phase-0b wider-bracket instrumentation (sub-bead to file)
- Phase 1: `9k4w.1` (ancestor-stack memo, P2 → P1 after Phase 0), `9k4w.2` (empty-cell bitmask, P2), `9k4w.3` (ray-octant mirror, P3), `9k4w.4` (Apple register flatten, P2)
- Phase 2: `me6i` (existing P1)
- Phase 3: `adp-res` (filed tonight, P2)
- Phase 4: `m59h` (filed tonight, P2 design-gated)
- Cross-cutting: `eff-hyg` (filed tonight, P3)
- Closed/landed tonight: `xer4` (Cargo profile rename)
- Parent umbrella: `2w1u` (P0, claim released)
- Indirect parent: `dbz5` (perceived-FPS umbrella)

**Closed prior beads, named here so they aren't re-litigated:**
- `dlse.2.2 exp#4` — `frame_latency` ∈ {1,2,3} swept null
- `dlse.2.2.2` — late-acquire reorder shipped null
- `dlse.2.2.3` — warmup transient characterized; this is what tonight's runs sampled
- `dlse.2.3` — proof-of-honest-timestamp; `bench_raycast_256 ≈ 0.19ms`
- `dlse.2.4` — render-pass attribution; **the bead that should have informed v1 of this doc**

---

## What this doc is not (revised v3)

- **Not a bead.** Beads are filed and listed in Cross-references.
- **Not a commitment to ship.** Each bead goes through /ship-auto with its own review tier. Anything that doesn't measure as predicted gets reverted.
- **Not Mac-only.** v2 had a Mac-specific framing built around a wrong mechanism. v3 is cross-platform, with Apple-specific notes called out only where they are.
- **Not measurement-complete.** Phase 0 is the precondition. Skipping it means optimizing without a credible baseline.
