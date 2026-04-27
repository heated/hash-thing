# Render-perf direction (post-69ip/29xk/j98h, 2026-04-27, v4)

**Status:** Forward-looking roadmap. v4 incorporates Triple plan review of the wgbp update (Claude + Gemini lenses landed; Codex hung per hash-thing-pc95) and three closed-bead findings since v3 (`69ip` Phase-0a fence-poll, `29xk` Phase-0 256³ × 0.5 windowed steady-state, `j98h` Phase-0 follow-up arms 64³/512³/1.0). **Four named reversals from v3** (Phase 1 Mac framing, Phase 4 P1 BLOCKED state, Phase 3 promotion, Phase 2 me6i reframing) **+ one new sub-experiment phase** (Phase 0c = `szyh`). v3 reversals from v2 preserved below as history.

## TL;DR (rewritten v4)

**M2 at every measured scale × render-scale, in the post-warmup steady-state windows of perf paper §3.16 / §3.16.1, has `render_gpu` ≈ 0.06–0.10 ms per frame.** Mac frame budget at problem scenes is dominated by `surface_acquire_cpu`, and `surface_acquire_cpu` scales with framebuffer pixels (render-scale), not voxel volume. v3's "tonight's 30 s captures sampled the warmup transient" reading is now upgraded by 69ip's fence-poll evidence: forcing CPU-side serialization (`device.poll(Wait)` after every submit) collapsed `render_gpu` mean from 14.30 ms to **0.10 ms** at 64³, more than two orders of magnitude below the 9k4w.0 decision threshold (~0.16 ms mean); p95 fell from 32.90 ms to 0.20 ms — slightly above the mean-threshold but at the same order of magnitude as the §3.16 steady-state mean (0.10 ms) on the same hardware. The 9–26 ms off-surface `render_gpu` reading was queue/scheduler effects, not real shader cost. **High confidence**, no longer medium.

**Roadmap (revised v4; section order in body matches this list):**

1. **Phase 0: DONE.** §3.16 + §3.16.1 of the perf paper extend §3.13 with 256³ × 0.5 windowed steady-state (37 ms `frame_total`) + 64³, 512³, and 256³ × 1.0 follow-up arms. 9k4w.0 (= `69ip`) fence-poll experiment closed; backpressure confirmed.
2. **Phase 1: cheaper rays — cross-platform-only on Mac.** Highest-leverage on cross-platform targets where compute is the bottleneck (literature projection from dubiousconst282 sparse-64-tree + Aokana 2025; not yet measured on this codebase). On Mac, post-warmup steady-state Mac-incremental yield is bounded above by `render_gpu` ≈ 0.1 ms / frame — effectively zero on a 17 ms budget. Beads stay in flight (cross-platform yield is real); Mac-incremental closure clause added to Phase 1 acceptance.
3. **Phase 2: investigate `me6i` drift — reframed.** v3 framed me6i as "likely duplicate of dlse.2.2.3 warmup transient." j98h finding 4 reframes me6i as a *distinct* mechanism — drift scales with time × world × render-scale, not magnitude alone (arms A and B at 50% scale don't drift; arm C at 256³ × 1.0 drifts 97 → 133 within run, same shape as the §3.16 baseline). This is not the dlse.2.2.3 transient; it persists *post-warmup* in scaled-up cells.
4. **Phase 3: adaptive render-scale (`pfpn`) — promoted to highest-leverage Mac knob.** j98h shows render-scale 0.5 → 1.0 at 256³ inflates `surface_acquire_cpu` 16 → 87 ms (5.5×) and `frame_total` 37 → 115 ms (3.1×). This is the dominant render-side knob on M2; `pfpn`'s adaptive ramp targets it directly. Phase 3 should land before Phase 4's design phase opens.
5. **Phase 0c (new, m59h evidence input): fence-poll experiment at 256³ × 1.0.** Filed as `szyh`. Disambiguates whether the 87 ms `surface_acquire_cpu` post-warmup mean is queue-backpressure-shaped (collapses under fence-poll, m59h's lever is dead) or NOT queue-backpressure-shaped (m59h's lever may be real, but Phase 0c does not by itself prove compositor sync — see Phase 0c body for the full decision rule). Cheap (~30 s harness extension); strong-recommended-prereq for m59h's design phase. Section appears below Phase 3 in body (paired with Phase 4 since it gates m59h's design call).
6. **Phase 4 (m59h): async surface acquire — already-executed P2 → P1 BLOCKED.** v3 deferred this as "uncertain"; spark's 2026-04-27 update on `m59h` post-69ip already promoted it. v4 reflects the existing state. Design call still gated on edward; recommended-prereq sequence is **Phase 0c (szyh) → Phase 3 (pfpn) → re-measure → m59h design call**. Phase 3 attacks the *amount* of compositor work, Phase 4 attacks the *latency* of the compositor-sync wait — different surfaces of the same metric, but Phase 3 is cheaper-and-faster. **v4 off-ramp:** if Phase 3's adaptive ramp meets the 60 FPS budget on M2 at 256³/50%, m59h closes as obsolete-by-Phase-3 without architectural rewrite.

**Reversals from v3:**
- v3 said "rays are highest-leverage everywhere." v4 says **highest-leverage cross-platform; on Mac, post-warmup steady-state incremental yield is bounded above by ≈0.1 ms / frame.** The cross-platform claim is unchanged but flagged as a literature projection until non-Mac measurement runs.
- v3 said "Phase 4 async acquire is uncertain, deferred." v4 reflects **already-executed `m59h` P2 → P1 BLOCKED** (spark 2026-04-27, post-69ip backpressure confirmation), with sequencing prerequisites added (Phase 0c → Phase 3 → re-measure).
- v3 had Phase 3 as "cross-platform polish, lower priority." v4 promotes Phase 3 to **highest-leverage Mac knob** per j98h's render-scale × `surface_acquire_cpu` evidence.
- v3 framed `me6i` as "likely duplicate of dlse.2.2.3 warmup transient." v4 reframes per j98h finding 4 as a **distinct, post-warmup, scale × render-scale × time interaction**.

**Reversals from v2 (v3 history, preserved):**
- v2 said "demote cheaper rays from highest-leverage on Mac." v3 said **rays are highest-leverage everywhere** — the Mac-specific compositor-bound framing was wrong. v4 partially restores the v2 conclusion *with a correct mechanism* (render_gpu = 0.1 ms ceiling, not "compositor pacing").
- v2 said "Phase 2-Mac async surface acquire is the highest-leverage Mac fix." v3 said **async acquire is uncertain, deferred.** v4 reflects spark's already-executed promotion of m59h post-69ip.
- v2 framed Phase 0 as a fresh experiment. v3 framed it as continuation of §3.9–§3.12. v4 marks Phase 0 as **DONE** (§3.16 + §3.16.1 extended; `69ip`, `29xk`, `j98h` all closed).

---

## What we measured (2026-04-26 night + 2026-04-27 follow-up)

Four 2026-04-26 30s headless captures on M2, default scene at 64³, release profile (LTO + opt-3). Logs preserved in `/tmp/2w1u-{autovsync,immediate,frozensim,offsurface}.log` on edward's machine. **Plus 2026-04-27 follow-up rows** for the 9k4w.0 fence-poll experiment (`69ip`) and the Phase-0 steady-state captures (`29xk` + `j98h`); see §3.16 / §3.16.1 of the perf paper for full data.

| Mode (date · scale · scene) | frame_total | surface_acquire_cpu | submit_cpu | render_gpu (ts query) |
|------|-------------|---------------------|------------|------------------------|
| AutoVsync (04-26 · 64³)              | 50–80ms growing | 47–80ms     | 0.3ms       | 0.1ms |
| Immediate (04-26 · 64³)              | 40–50ms growing | 37–46ms     | 0.3ms       | 0.1ms |
| AutoVsync (04-26 · 64³ · sim FROZEN, Gen 0)   | 33→70ms growing | 32→69ms     | 0.3ms       | 0.1ms |
| OFF_SURFACE=1 (04-26 · 64³ · sim on)          | 14–42ms         | **0.01ms**  | **18–40ms** | **9–26ms** |
| OFF_SURFACE=1 + fence-poll (04-27 · 64³ · 9k4w.0=`69ip`) | 46.19/54.09 ms | (n/a, off-surface) | 40.30/42.18 ms | **0.10/0.20 ms** |
| 256³ × 0.5 windowed steady (04-27 · 29xk · §3.16) | **37.25 ms** | 15.62 ms | 0.16 ms | 0.10 ms |
| 256³ × 1.0 windowed steady (04-27 · j98h arm C · §3.16.1) | **114.92 ms** | **86.73 ms** | 0.15 ms | 0.07 ms |
| 64³ × 0.5 windowed steady (04-27 · j98h arm A · §3.16.1) | 24.30 ms | 8.90 ms | 0.15 ms | 0.06 ms |
| 512³ × 0.5 windowed steady (04-27 · j98h arm B · §3.16.1) | 21.37 ms | 15.72 ms | 0.16 ms | 0.08 ms |

### What this actually tells us, in light of §3.9–§3.12 + §3.16 / §3.16.1

- **All four 2026-04-26 runs sampled the dlse.2.2.3 warmup transient.** Per §3.9 (onyx 2026-04-21), `surface_acquire_cpu` follows a documented climb-plateau-step-down shape over 0–34s on M2 at 256³: climb 23→34ms in t=2–14, kink at memo cache trim (t=14–16), plateau ~27ms through t=26, then step-down to ~11ms by t=34. Tonight's 30s captures captured exactly the climb + plateau + start-of-step-down phase. **Tonight's 30→80ms surface_acquire growth was the same transient, not a new finding.**
- **`me6i` (frame_total growth over time) is now characterised by `j98h` finding 4 as a distinct mechanism.** v3 framed it as "likely duplicate of dlse.2.2.3." j98h finding 4 instead shows me6i drift scales with time × world × render-scale, not frame-magnitude alone — arms A (64³ × 0.5) and B (512³ × 0.5) are clean (drift ratio 0.95–1.09); arm C (256³ × 1.0) drifts 97→133 within each post-warmup run, same shape as the §3.16 baseline. **Not a duplicate of the dlse.2.2.3 pre-t=34 transient.**
- **High confidence: present mode is not the lever.** AutoVsync vs Immediate within noise. Confirms §3.9 step 1.
- **High confidence: per-frame compute pass is ~0.06–0.10 ms across every measured Mac cell.** §3.16 + §3.16.1 hold this flat across 64³ → 512³ × 0.5 → 1.0; render_gpu is not the bottleneck on M2 in any regime measured.
- **High confidence (v4 upgrade from v3 medium): 9–26ms off-surface render_gpu reading IS queue/scheduler backpressure.** `69ip` fence-poll collapsed `render_gpu` from 14.30 / 32.90 ms (mean / p95) to **0.10 / 0.20 ms** when frames were CPU-serialized via `device.poll(Wait)`. The mean is more than two orders of magnitude below the 9k4w.0 decision threshold (~0.16 ms mean); the p95 is slightly above the mean-threshold but at the same order of magnitude as the §3.16 steady-state mean (0.10 ms) on the same hardware — consistent with backpressure being the carrier. Pipeline-depth metric `prior_gpu_pipeline_cpu` collapsed from ~20 s (off-surface unbounded) to ~6 ms (one-frame-deep), confirming the experiment was well-formed. The phantom 14 ms `render_gpu` reading was queue scheduling absorbed by CPU-GPU pipelining, not real shader work.
- **Open question, v4: is `surface_acquire_cpu` itself queue-backpressure?** The 87 ms post-warmup mean at 256³ × 1.0 (`j98h` arm C) is wall-clock CPU time the renderer thread spends waiting for the next drawable. We don't yet know whether forcing serialization collapses *this* metric the way it collapsed `render_gpu`. Filed as `szyh` (Phase 0c, m59h evidence input). See Phase 0c subsection below.
- **Wrong framing in v2 (corrected in v3, preserved here): "Apple Metal compositor pacing".** §3.9 step 2 refuted compositor (fullscreen worse than windowed). Don't say "compositor" without isolating the specific mechanism, which §3.9–§3.12 says is unidentified after sweeping every wgpu-exposed knob; v4 keeps the gap open.

---

## Phase 0: DONE (§3.16 + §3.16.1 of the perf paper)

**Beads:** `9k4w` (parent, closed) · `9k4w.0` = `69ip` (Phase-0a fence-poll, closed 2026-04-27) · `29xk` (256³ × 0.5 windowed × 3 runs, closed 2026-04-27, perf paper §3.16 = bbdeab1) · `j98h` (64³ + 512³ + 256³ × 1.0 follow-up arms × 3 runs each, closed 2026-04-27, perf paper §3.16.1).

**Outcome (3-line summary):**
- 9k4w.0 / `69ip`: forcing CPU-side fence-poll (`device.poll(Wait)` after every submit) collapsed `render_gpu` 14.30 / 32.90 ms → **0.10 / 0.20 ms** at 64³. Backpressure interpretation confirmed; the 9–26 ms off-surface `render_gpu` reading from v3 was queue/scheduler effects.
- 29xk / §3.16: 256³ × 0.5 windowed steady-state across 3 runs has `frame_total` mean = **37.25 ms** (within 1.7% across runs), `surface_acquire_cpu` = 15.62 ms, `render_gpu` = 0.10 ms. **Steady-state >> 17 ms 60 FPS budget; ramp is NOT the dlse.2.2.3 warmup transient — it is a true scaling cost at 256³.**
- j98h / §3.16.1: at 50% render-scale, `frame_total` is roughly constant ~21–24 ms across 64³–512³. Render-scale 1.0 inflates `frame_total` by 3.1× and `surface_acquire_cpu` by 5.5× at 256³. **`render_gpu` stays 0.06–0.10 ms across all four cells.** The dominant render-side knob on M2 is render-scale, not voxel work.

**Procedure (preserved as record).** Hardware: M2 MBA 16GB, 60Hz external display (windowed) + M2 internal display (fullscreen, deferred). `--profile perf`. Worlds: 64³ / 256³ / 512³. Render-scale: 0.5 / 1.0. Run length 120 s, first 34 s discarded as warmup. 3 independent runs per arm. Reproduce: `RUST_LOG=info HASH_THING_RENDER_SCALE=0.5 HASH_THING_FOCUS=1 timeout 130 ./target/perf/hash-thing 256 --res 1080p > /tmp/29xk-256-run.log`. Logs preserved; analyzers at `/tmp/29xk-analyze.py`, `/tmp/j98h-aggregate.py`.

**What's left (out of Phase 0 proper):**
- **Arm D** (256³ × 0.5 × fullscreen-internal): parked as gate on j98h close-comment — the binary has no programmatic fullscreen toggle. Sub-task; not a Phase-0 blocker.
- **Phase 0b wider-bracket instrumentation:** v3 proposed a "whole-pipeline" timestamp pair + `TIMESTAMP_QUERY_INSIDE_ENCODERS` finer brackets. 69ip's fence-poll harness gave the same disambiguation more cheaply; Phase 0b not needed unless a sub-investigation surfaces.
- **Phase 0c** (new in v4, `szyh`): `surface_acquire_cpu` fence-poll at 256³ × 1.0. See dedicated subsection below.

---

## Phase 1: cheaper rays — cross-platform-only on Mac (rerated v4)

**Beads:** `9k4w.1` (ancestor-stack memo — already-landed, see audit below), `9k4w.2` = `iqx7` (empty-cell bitmask — partial; the 64-bit grandchildren-mask variant still TODO), `9k4w.3` (ray-octant mirror — already-landed), `9k4w.4` = `wck2` (Apple register flatten — open).

> **Audit update — `k6tm` (cairn) + `e0no` (flint cross-check) 2026-04-27.** Three of the four cheaper-rays sub-beads are already on main; current measurements bake in their benefit:
>
> - **9k4w.3 ray-octant mirror — LANDED.** `crates/ht-render/src/svdag_raycast.wgsl:310-315` (mirror_mask construction) + `:478-479,654` (XOR un-mirror at child lookup). Landing commit `ce97c64` ("render: Laine-Karras Stage 2 ray-octant mirroring"). → **`5m36` closed.**
> - **9k4w.1 ancestor-stack memo — LANDED.** Within-ray pop-to-deepest-still-containing-ancestor at `:396-407` + per-ray register-resident `stack_node[]/stack_min[]/stack_half[]/stack_cmask[]` arrays. Landing commit `e02c55a` (original SVDAG pipeline) extended by `44f7630` (m1f.7.3, stack_cmask caching). The dubiousconst282 article confirms this is the ~1.9× optimization (per-ray stack in registers, not cross-ray) — duke-henry's "Next ray in the tile restarts" prose was a loose paraphrase. → **`5sjh` closed.**
> - **9k4w.2 empty-cell bitmask — PARTIAL.** The 8-bit immediate-children-occupancy mask is landed (`:482-485` cmask pre-check before child_slot read, `44f7630` m1f.7.3; `:595-657` inner-DDA skip-empty-children loop, `d0c8b9a` x5w.1). The 64-bit *grandchildren* occupancy mask described in the bead — single bitscan in the parent skips an entire subtree without descending — is **still TODO**. The +22% memory bloat caveat applies only to that 64-bit grandchildren variant. → **`iqx7` stays open with refined scope.**
> - **9k4w.4 Apple register flatten — open, out of audit scope.**
>
> **Roadmap impact (v4-revised):** Remaining Phase 1 work is `iqx7` refined (64-bit grandchildren bitmask) + `wck2` (Apple register flatten). The 1.5×–2.2× combined-leverage estimate below was assuming all four sub-beads still to land; subtract the .1 + .3 already-banked contribution + the 8-bit immediate-children part of .2 already-banked, and the **incremental** Phase 1 yield over current main is closer to **1.1×–1.3× at 256³ on cross-platform targets where compute is the bottleneck**. On Mac, post-69ip evidence shows `render_gpu` is already at the 0.1 ms floor and Phase-1 incremental yield is bounded above by ≈0.1 ms / frame regardless of which sub-beads land — invisible on a 17 ms budget.

Sources: [dubiousconst282 2024-10-03](https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/) (concrete cycle counts on integrated GPU, 2.66× stacked), [Aokana 2025 (arxiv 2505.02017)](https://arxiv.org/html/2505.02017v1) (structural confirmation on Vulkan). **Both are projections from non-this-codebase measurements**; v4 keeps them as the cross-platform leverage estimate but flags the asymmetry: Mac is now empirically at 0.1 ms `render_gpu` post-warmup; cross-platform leverage on this SVDAG remains a literature projection until a non-Mac steady-state run captures it.

### Why cheaper rays are cross-platform-leverage but Mac-incremental ≈ 0 in v4

- **Steady-state Mac math.** §3.16 + §3.16.1 hold `render_gpu` at 0.06–0.10 ms across 64³–512³ × 0.5–1.0. A 10× reduction in compute saves at most ~0.1 ms / frame on a 17 ms budget = **≤ 0.6 % of frame budget**. Below user-visible threshold; below sampling noise on this hardware. **Mac-incremental yield is bounded above by ≈0.1 ms / frame** regardless of which Phase-1 sub-bead lands.
- **Warmup-envelope and me6i-drift windows are NOT Phase-1 territory either.** The 9–26 ms off-surface `render_gpu` reading at 64³ that v3 saw was queue-backpressure (69ip), not real shader cost — fence-poll collapsed it to 0.10 / 0.20 ms. So even in the warmup envelope `render_gpu` is not the bottleneck; cheaper rays don't help there. me6i drift in arm C (256³ × 1.0) is `surface_acquire_cpu`-carried, not `render_gpu`-carried (j98h finding 4) — Phase 1 doesn't address it. **Those windows are m59h / `me6i` territory, not Phase 1's.**
- §3.11's earlier "the ramp is pipeline fill" at 1024³/100% framing is consistent with v4's read: pipeline fill on M2 is dominated by `surface_acquire_cpu` × framebuffer pixels (Phase 3's lever), not voxel-work on the compute pass.
- **Cross-platform users (non-Mac) see the literature-projected ~2.6× win**, modulated by the sparse-8-vs-sparse-64 caveats below. This is the rationale for keeping the sub-beads in flight; the Mac-incremental clause just lets a Mac-only measurement close successfully.

### Sparse-8 vs sparse-64 — the structural caveat (Gemini reviewer surfaced)

dubiousconst282 measured on a sparse-64-tree (depth ~4–5). Our SVDAG is sparse-8-tree (depth 8 at 256³, depth ~12 at 4096³). This isn't a constants caveat — it's a structural one, and the impact on each technique differs:

- **9k4w.1 ancestor-stack memo (8 pointers, 1.9× claim).** Fits depth ≤8 cleanly (256³). At depth 12 (4096³), 8 pointers cover only the bottom of the path; the upper levels still re-traverse from root. **Most likely outcome at 256³: 1.5–1.8× (close to claim). At 4096³: 1.2–1.4×.** Update bead description; tighten acceptance to a measured-on-our-tree delta, not a "±15% of dubiousconst282" gate.
- **9k4w.2 empty-cell bitmask (1.26× claim).** Requires storing grandchildren occupancy in each interior node. Our nodes are 36 bytes; adding a 64-bit bitmask is +22% memory. Either accept the bloat (acceptable but worth flagging), or use an 8-bit children-occupancy mask which is cheaper but lower yield. **Most likely outcome: 1.1–1.2× with smaller mask, 1.3× with full bitmask + memory bloat.** Trident plan review on the design choice.
- **9k4w.3 ray-octant mirror (1.1× claim).** Structural-mismatch-immune (works on any sparse N-tree). Yield should match within ±5%.
- **9k4w.4 Apple register flatten (Apple-specific, mid-yield).** Independent of dubiousconst282. Flatten the recursive traversal helper to fix M-series occupancy collapse past ~32 registers/thread.

**Updated combined leverage estimate:** 1.7–2.2× total at 256³, 1.3–1.7× at 4096³. Still substantial. v2's "2.6× total" was sparse-64-tree's number; v3 honest estimate is lower for our tree.

### Acceptance per sub-bead (updated; v4 adds Mac-incremental closure clause)

- Phase 0 instrumentation in place — DONE per §3.16 / §3.16.1.
- Pre-optimization `render_gpu_total` reading captured **on M2 + at least one non-Mac target** (Linux+NV, Linux+AMD, or Windows+integrated).
- Optimization landed via /ship-auto + Triple review.
- Post-optimization reading shows the **internal-baseline-delta** (not a "±15% of dubiousconst282" gate; that was Codex reviewer's correct critique).
  - 9k4w.1: ≥1.4× compute-pass speedup at 256³ on the non-Mac target.
  - 9k4w.2 (= `iqx7`): ≥1.1× on top of 9k4w.1.
  - 9k4w.3: ≥1.05× on top of 9k4w.1+.2.
  - 9k4w.4 (= `wck2`): ≥1.2× on M2 specifically (Apple-occupancy yield); ≤5% regression on non-Mac.
- Visual regression: `--dump-frame` output within FP-tolerant pixel-percentage threshold (Claude reviewer's correct critique — byte-equivalent is too strict). Specify: ≤0.1% pixels differ by more than 4 LSBs in any channel.
- **v4 Mac-incremental closure clause:** a Phase-1 sub-bead measuring no Mac-incremental change but a measurable cross-platform speedup may close successfully; the cross-platform target is the gate. (See `Standardized measurement protocol` for the generalised version of this rule.)

---

## Phase 2: characterize / fix `me6i` drift — reframed in v4

**Bead:** `me6i` (existing, P1). v4 reframes the investigation per j98h finding 4.

**v3 framing (superseded):** "likely duplicate of dlse.2.2.3 warmup transient — worth a 5-minute check." j98h's follow-up arms now show this is wrong: drift persists *post-warmup* (post-t=34) and scales with `time × world × render-scale` not `frame_total magnitude alone`. Arms A (64³ × 0.5) and B (512³ × 0.5) are clean (drift ratio 0.95–1.09); arm C (256³ × 1.0) drifts 97 → 133 within each post-warmup run, same shape as the §3.16 baseline. **Not the dlse.2.2.3 pre-t=34 transient** — distinct mechanism.

**v4 framing:** me6i is a `surface_acquire_cpu`-carried drift mechanism active in scaled-up cells (256³ × 1.0 and the §3.16 baseline 256³ × 0.5; 64³ and 512³ × 0.5 are clean). This is exactly the regime Phase 3 (adaptive render-scale, `pfpn`) is designed to ramp out of, AND the regime where Phase 4's `surface_acquire_cpu` lever is most relevant. The Phase-2 investigation question shifts from "is this a warmup transient" to "is this `surface_acquire_cpu`-carried (likely Phase 3/4 territory) or some other in-loop accumulation."

**Updated investigation step:**
- Re-run arm C (256³ × 1.0 windowed) for ≥120 s past warmup, capturing per-window means with `surface_acquire_cpu` / `submit_cpu` / `render_gpu` decomposition.
- If `surface_acquire_cpu` carries the drift (mean climbs while `submit_cpu` + `render_gpu` stay flat): me6i is the same mechanism Phase 3 / Phase 4 attack — close `me6i` as a duplicate of the surface-acquire ladder, fold its acceptance into Phase 3's ramp test.
- If `submit_cpu` or another carrier surfaces: distinct mechanism. File a sub-bead.

**Acceptance:** investigation result documented in `me6i` comment, citing j98h finding 4 and identifying the drift carrier(s). If carrier is `surface_acquire_cpu`-dominant → close as duplicate of Phase 3/4 territory (Phase 3's ramp acceptance subsumes me6i's). If other carrier(s) dominant → file sub-bead(s) per the carrier(s). If mixed → document the decomposition and let the dominant carrier drive close-out. v4 cross-cutting rule applies — see `Standardized measurement protocol` for the closure clause covering Mac-incremental-zero / cross-platform-only-measurable beads.

**Why not parallel with Phase 1:** unchanged. If me6i resolves into Phase 3 territory, Phase 3's ramp acceptance subsumes me6i's; better to know.

---

## Phase 3: adaptive render-scale (`pfpn`) — promoted to highest-leverage Mac knob (v4)

**Bead:** `pfpn` (= adp-res renamed; existing, P1). j98h is the receipts.

**Why this is the dominant Mac knob (j98h evidence):**

- 256³ × 0.5 windowed: `surface_acquire_cpu` = 16 ms post-warmup mean.
- 256³ × 1.0 windowed: `surface_acquire_cpu` = **87 ms** post-warmup mean (5.5×).
- `frame_total` 37 → 115 ms (3.1×) for the same 0.5 → 1.0 render-scale shift.
- `render_gpu` stays 0.07 → 0.10 ms (no shader-side carrier).

**Render-scale is the dominant render-side knob on M2.** Each framebuffer-pixel reduction translates roughly 1:1 into compositor budget. `pfpn`'s adaptive ramp targets it directly.

Static GPU-class table at startup + dynamic ramp toward target frame budget. M-series → 0.6 default. Discrete dGPU → 1.0. Other integrated → 0.7. `HASH_THING_RENDER_SCALE` always overrides.

**Sequencing in v4:** Phase 3 should land *before* Phase 4 (m59h) opens its design phase. If Phase 3's ramp brings 256³ × 0.5 frame_total to within the 17 ms 60 FPS budget on M2, Phase 4's lever is no longer needed for the headline regime; m59h can close as obsolete-by-Phase-3. If Phase 3 does NOT bring frame_total to budget, the residual is the m59h motivation — and Phase 0c (`szyh`) tells us whether m59h's lever is real.

**Acceptance** (Codex reviewer's correct critique applied — name the reference adapter):
- Default Mac M2 launch hits ≤17ms frame_total (60 FPS budget) at chosen render_scale within first 5s of warm frames at 256³ default scene.
- Reference non-Mac: launch on either an Nvidia RTX 3060 (or equivalent ~20 TFLOPs class) or an Intel Iris Xe (or equivalent ~2 TFLOPs class) hits ≤17ms at chosen render_scale at 256³.
- Adaptive ramp converges in <2s with hysteresis ≤±5% over a 10s steady-state window (specified measurement window per Claude reviewer).
- Trident plan review minimum.
- v4 cross-cutting rule applies — see `Standardized measurement protocol` for the closure clause covering Mac-incremental-zero / cross-platform-only-measurable beads.

**Depends on:** Phase 0 — DONE per §3.16 / §3.16.1.

---

## Phase 0c (new in v4): `surface_acquire_cpu` fence-poll experiment at 256³ × 1.0

**Bead:** `szyh` (filed 2026-04-27 by flint per wgbp Triple plan review R1). Sub-experiment under m59h's evidence-input lane; **strong-recommended-prereq for m59h's design call**. Does not modify m59h's bd-graph blocker (m59h is already P1 BLOCKED on edward's design call), but the doc-level gate is: design without Phase 0c evidence risks committing the design phase to a one-way-door that the experiment would have ruled out in 30 s.

**Open question.** 69ip showed `render_gpu` 9–26 ms at 64³ off-surface was queue/scheduler backpressure (collapses to 0.10 / 0.20 ms under fence-poll). j98h showed `surface_acquire_cpu` at 256³ × 1.0 windowed has a post-warmup mean of 86.73 ms (drifts 97 → 133 within window per finding 4 — drift mechanism shared with me6i). **We don't yet know whether the 87 ms `surface_acquire_cpu` reading is queue-backpressure-shaped** (in which case async surface acquire = m59h cannot help — the wait would just shift somewhere else on the queue) **or some other mechanism** (compositor sync, shared-resource contention, HAL CPU-side work, etc. — m59h's lever may or may not help, depending on which).

**Procedure (cribbed from 69ip).** Re-enable `HASH_THING_FENCE_POLL=1` in renderer.rs (the 69ip experimental knob, currently reverted). Experimental code lives on a worktree branch and is never landed on main — same exception-lane treatment as 69ip; results land via comment on `szyh` + `m59h`, branch deleted post-experiment. Run `--profile perf` + 256³ + `HASH_THING_RENDER_SCALE=1.0`, windowed external display, ≥120 s, first 34 s warmup discarded. Compare `surface_acquire_cpu` mean / p95 baseline vs fence-poll. **Cross-validation arm:** also run at 256³ × 0.5 (sanity check that the harness still works at problem scale; expect the fence-poll collapse pattern to hold there too if the harness is healthy).

**Decision rule** (softened from earlier draft per code-review finding 6):
- `surface_acquire_cpu` collapses to <1 ms under fence-poll → **queue-backpressure-shaped confirmed**; m59h's lever (off-thread acquire) cannot help — the wait would just shift somewhere else on the queue. Close m59h as no-help; surface the queue-side mechanism as a follow-up.
- Stays > 50 ms → **NOT queue-backpressure-shaped.** Three sub-cases possible — Phase 0c does not disambiguate among them: (a) true compositor sync (m59h's lever is real); (b) wgpu/Metal HAL CPU-side work that off-thread acquire would relocate but not eliminate; (c) shared-resource contention (M2 unified memory bus, IO, etc.) that off-thread won't help with. m59h's design phase still has to pick a mechanism explanation, but the queue-backpressure null result is logged as evidence.
- Partial (5–50 ms) → document the partial collapse; let m59h design phase weight the partial evidence.

**Acceptance** (mirrors 69ip + adds cross-validation gate):
- Experiment run, both conditions (baseline + fence-poll) captured at 256³ × 1.0.
- **Cross-validation arm (256³ × 0.5) captured under both conditions.** If fence-poll collapse pattern at 256³ × 0.5 disagrees with the 69ip 64³ result (e.g., 256³ × 0.5 fence-poll keeps `surface_acquire_cpu` near baseline instead of collapsing), document and surface — the harness behavior may have shifted since the 69ip revert.
- Outcome posted as comment on `szyh` AND on `m59h`.
- Experimental code reverted (pure diagnostic; not for shipping). Exception-lane work per project CLAUDE.md.

---

## Phase 4: async surface acquire (`m59h`) — already-executed P2 → P1 BLOCKED (v4 reflects)

**Bead:** `m59h` (already P1 BLOCKED per spark's 2026-04-27 update on the bead, post-69ip). v4 reflects the existing state.

**Why P1 BLOCKED post-69ip** (v4 framing):

- §3.9 swept `maximumDrawableCount` ∈ {2,3,4} — null.
- moss 2026-04-15 swept `displaySyncEnabled` ∈ {true,false} — null.
- `dlse.2.2.2` shipped late-acquire reorder (Apple's documented best practice) — null.
- `dlse.2.2 exp#4` swept `frame_latency` — null.
- **`69ip` 2026-04-27 confirmed backpressure interpretation.** GPU compute is genuinely 0.1 ms; the per-frame ceiling on Mac is `surface_acquire_cpu` itself. Either Phase 3 ramps it down (preferred — cheaper, lower-risk) or Phase 4 moves it off-thread.

Async surface acquire is the fifth swapchain-pacing fix and the only architectural-rewrite candidate (vs the four prior env-knob swaps). It's plausible that an architectural change escapes what the prior nulls swept; that plausibility, combined with 69ip's backpressure confirmation, motivates the P2 → P1 BLOCKED promotion. It does NOT motivate skipping Phase 3 — Phase 3 attacks the same metric more cheaply. Recommended sequence in v4: **Phase 0c (`szyh`) → Phase 3 (`pfpn`) lands → re-measure → m59h design call**.

**How Phase 3 and Phase 4 differ.** Phase 3 reduces the *amount* of compositor work (smaller framebuffer = fewer pixels for the OS compositor to schedule). Phase 4 reduces the *latency* of the renderer thread blocking on the compositor sync (moves the wait off-thread). Both attack `surface_acquire_cpu` but for different reasons; Phase 3 is cheaper and lower-risk. Phase 3 first.

**Acceptance** (decision tree; explicit gate, not just promotion criterion):

m59h close-out paths, in evaluation order (whichever fires first wins; later branches don't apply if an earlier branch closed the bead):

1. **`szyh` (Phase 0c) shows queue-backpressure-shaped collapse** → close m59h as no-help; off-thread acquire wouldn't reduce the wait. Architectural rewrite avoided.
2. **Phase 3 (`pfpn`) ramp meets ≤17 ms `frame_total` mean over 60 s warm steady-state on M2 windowed at 256³/50%** → close m59h as obsolete-by-Phase-3; Phase 3's smaller framebuffer absorbs the cost. Architectural rewrite avoided.
3. **m59h design phase opens, ships, and meets ≤17 ms `frame_total` mean over 60 s warm steady-state on M2 windowed at 256³/50%** → close as success.
4. **m59h design phase opens, ships, and does NOT meet the gate** → close as can't-fix with documented residual stall.

Branches 1 and 2 are the v4-introduced off-ramps; branches 3 and 4 are the v3 acceptance preserved. The doc-level rule: don't open the design phase (branch 3 / 4) until branches 1 and 2 are both ruled out.

**Risks** (Gemini + Claude reviewers added; v4 marks Phase-0c-conditional):
- **Drawable pool exhaustion (Phase-0c-conditional probability).** If `szyh` shows `surface_acquire_cpu` collapses under fence-poll → queue-backpressure → exhaustion risk High; async-acquire path is dead. If `szyh` shows it doesn't collapse → compositor-sync → exhaustion risk Low (pool extension mitigates).
- **`me6i` interaction.** If the drift is retained-drawable accumulation, async acquire will retain MORE drawables and may make me6i worse before better. Phase 2 reframing connects me6i to Phase 3 / 4 territory; investigation as part of Phase 2.
- **Architectural scope.** This is a thread/resource ownership rewrite, not a "Phase 4 task." Likely a multi-bead epic, not a single bead. m59h already reflects this in its description.

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
- **Focus knob (qny5):** prefer `HASH_THING_PERF_CAPTURE=1` over `HASH_THING_FOCUS=1` for long-form windowed captures. `PERF_CAPTURE` keeps the redraw treadmill alive while the window is unfocused, so a perf run does not steal focus from whoever is using the machine. `FOCUS=1` remains correct for cases that need the window foregrounded (interactive testing, screenshot workflows that depend on activation). The `occluded` short-circuit (8jp) still applies under `PERF_CAPTURE`: a hidden surface still pauses, since there's no point measuring it.

**Convention:** when a bead reports a perf-relevant measurement, it cites this protocol or explicitly notes deviation.

**v4 cross-cutting rule (generalised from Phase 1 acceptance).** A bead measuring no Mac-incremental change but a measurable cross-platform speedup may close successfully; the cross-platform target is the gate. Conversely, a bead with no cross-platform target may close on Mac-only evidence iff the bead description scopes it as Mac-specific (`9k4w.4` = `wck2` Apple register flatten is the canonical example). This rule applies to every Phase-1+ bead, not just Phase 1.

---

## Cross-cutting: efficiency hygiene

**Bead:** `eff-hyg` (filed 2026-04-26, P3).

Standing rule: every PR adding a per-frame code path also adds a `perf::record()` timer. Codified in `.agents/skills/review-tiers/SKILL.md` as a Triple-review checkbox.

CPU-side gap closure (separate from Phase 0b's GPU-side gap):
- winit `AboutToWait` / `RedrawRequested` dispatch.
- scene-state mutation.
- perf-bookkeeping path itself.

---

## Risk register (revised v4)

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Phase 0 confirms 256³/50% steady-state IS in the 60 FPS regime → edward's complaint is the warmup transient | **Resolved by §3.16** | 256³/50% steady-state `frame_total` = 37 ms (mean of 3 runs), >> 17 ms 60 FPS budget. NOT the warmup transient. Phase 1 (cross-platform) + Phase 3 proceed. |
| sparse-8 vs sparse-64 knocks down 9k4w.1's yield to ~1.5× and 9k4w.2's to ~1.2× (Medium-High) | Medium-High (Claude+Gemini) | Per-bead acceptance now uses internal-baseline-delta on M2+non-Mac, not a "±15% of dubiousconst282" gate. Reduced expected combined yield from 2.6× to 1.7–2.2× cross-platform. |
| `iqx7` empty-cell bitmask requires +22% memory bloat in SVDAG node format | Medium | Trident plan review on the design choice. May ship a smaller 8-bit mask variant first. |
| async-acquire × `me6i` interaction (retains more drawables → makes leak worse) | Medium (Claude+Gemini) | Phase 4 acceptance gates on `me6i` outcome from Phase 2; Phase 2 reframing now connects `me6i` to Phase 3 / 4 territory. |
| async-acquire exhausts 3-drawable pool, blocks main thread anyway | **Phase-0c-conditional** (Gemini) | Resolved by `szyh`: if `surface_acquire_cpu` collapses under fence-poll, exhaustion risk → High and m59h closes as no-help; if it doesn't, exhaustion risk → Low and pool-extension mitigates. |
| Phase 4 architectural scope is bigger than "Phase 4 task" suggests | High (Codex) | m59h treated as an epic, not a task. Design phase decomposes into sub-beads before implementation begins. |
| Mac-specific tooling (Instruments, Xcode Frame Capture) not available to headless seats | Medium (Codex) | `me6i` and Phase 4 explicitly flagged as "edward-driven or hand-off to a Mac-equipped seat." |
| Phase 1 optimizations conflict with `cswp` chunked LOD | Low | Phase 1 is per-ray; cswp restructures the DAG. Mostly orthogonal. |
| Phase-1 sub-bead regresses non-Mac targets | Low | `wck2` acceptance explicitly requires ≤5% regression on at least one non-Mac target. v4 cross-cutting rule lets Mac-incremental-zero close successfully against the cross-platform gate. |
| **v4 NEW: Cross-platform leverage projection unverified on this SVDAG** | Low-Medium (Claude) | v4 keeps v3's 1.7–2.2× cross-platform estimate as a literature projection (dubiousconst282 + Aokana 2025); first non-Mac steady-state run will either confirm or surface the gap. Reference adapters named in Phase 3 acceptance. |
| **v4 NEW: Phase 0c (`szyh`) skipped before m59h design call → m59h's lever-of-effect ambiguous** | Low | Phase 0c filed as `szyh`; documented as strong-recommended-prereq for m59h design call on m59h's bead-comment thread. Cheap experiment (~30 s harness extension); incentive to run pre-design is built in. |

---

## Review record

v3 → v4 received a Triple plan review (Claude + Gemini lenses landed; Codex hung at "Reading additional input from stdin..." per active hash-thing-pc95, killed per project CLAUDE.md 60s budget). Reviewer outputs preserved in `.ship-notes/plan-review-wgbp-{claude,codex,gemini}.md` (gitignored). v4 doc itself receives a Triple code review on the diff; outputs at `.ship-notes/code-review-wgbp-{claude,codex,gemini}.md`.

v3 received Triple plan review (Claude + Codex + Gemini), 2026-04-26. Reviewer outputs preserved in `.ship-notes/render-perf-direction-review-{claude,codex,gemini}.md` (gitignored locally; if you need them, ask cairn).

**Synthesis of reviewer asks (v3 → v4):**

| Ask | Reviewer | v4 status |
|-----|----------|-----------|
| Tighten "Mac-incremental ≈ 0" universal quantifier — true for steady-state windows, not warmup envelope or me6i drift | Claude+Gemini | ✅ qualified as "post-warmup steady-state" everywhere; warmup/me6i windows explicitly noted as m59h/me6i territory |
| Flip Phase 0c gating — file as m59h evidence-input now, run before m59h design call (not after) | Claude+Gemini | ✅ filed as `szyh`; Phase 0c subsection added; m59h's bead-comment cross-links it as strong-recommended-prereq |
| Phase 3 vs Phase 4 sequencing — both attack `surface_acquire_cpu`; Phase 3 is cheaper and should land first | Claude+Gemini | ✅ explicit sequencing: Phase 0c → Phase 3 → re-measure → m59h design call. Phase 4 closes-as-obsolete-by-Phase-3 if Phase 3 hits 60 FPS budget |
| Audit-block trailing sentence ("driven mostly by iqx7-refined and 9k4w.4 on Mac") contradicts v4 framing | Claude | ✅ surgical edit applied: "on cross-platform targets where compute is the bottleneck; on Mac, render_gpu is at the 0.1 ms floor and Phase-1 incremental yield is bounded above by ≈0.1 ms / frame" |
| Cross-platform leverage is unsourced on this codebase — make asymmetry visible | Claude | ✅ Phase 1 framing flagged as literature projection; risk register row added |
| Phase 2 me6i framing is a fifth reversal — j98h finding 4 distinguishes from dlse.2.2.3 | Claude | ✅ added as fourth named v3→v4 reversal; Phase 2 section reframed |
| "5.4×" should be "5.5×" (86.73 / 15.62 = 5.55) | Claude | ✅ corrected in Phase 3 receipts and TL;DR |
| "87 ms steady-state" is loose (arm C drifts within window); say "post-warmup mean" | Claude | ✅ corrected throughout |
| Phase 4 risk-register row probability is Phase-0c-conditional | Claude | ✅ marked Phase-0c-conditional; outcome paths spelled out |
| Generalisable rule (Mac-incremental zero close clause) belongs in Standardized measurement protocol, not Phase 1 only | Claude | ✅ added to Standardized-measurement-protocol section as v4 cross-cutting rule |
| m59h "promoted to P1 BLOCKED" wording — already-executed by spark, not v4-doing-the-promotion | Gemini | ✅ Phase 4 section says "v4 reflects the existing state" |

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

## Cross-references (revised v4)

**Prior measurements (read these before claiming any 9k4w child):**
- `docs/perf/svdag-perf-paper.md` §3.7–§3.12 — present-path investigation, dlse.2.2 candidate enumeration
- `docs/perf/svdag-perf-paper.md` §3.9 specifically — render-pass attribution (dlse.2.4) — **REVISES the "real GPU frame time" inference; v3 + 69ip validated this was the right revision**
- `docs/perf/svdag-perf-paper.md` §3.11 — GPU cost at 1024³/100% is real; "the ramp is pipeline fill" (`hash-thing-dlse.3`)
- `docs/perf/svdag-perf-paper.md` §3.12 — scene-matrix measurement + §3.11 correction
- **`docs/perf/svdag-perf-paper.md` §3.16 — 256³ × 0.5 windowed steady-state** (`29xk`, 3 runs × 130 s, frame_total = 37.25 ms)
- **`docs/perf/svdag-perf-paper.md` §3.16.1 — Phase-0 follow-up arms** (`j98h`, 64³ + 512³ + 256³ × 1.0, render-scale is the dominant Mac knob)

**Beads, organized by Phase (v4-revised priorities):**
- Phase 0: **DONE.** `9k4w` (P1, parent, closed) · `9k4w.0` = `69ip` (Phase-0a fence-poll, closed) · `29xk` (Phase-0 256³ × 0.5 windowed, closed) · `j98h` (Phase-0 follow-up arms, closed)
- Phase 0c: `szyh` (filed 2026-04-27 by flint per wgbp v4 plan review, P2, m59h evidence input)
- Phase 1: `9k4w.1` (closed/landed `5sjh`), `9k4w.2` = `iqx7` (P2 open, refined-scope), `9k4w.3` (closed/landed `5m36`), `9k4w.4` = `wck2` (P2 open). Mac-incremental yield bounded above by ≈0.1 ms / frame; cross-platform yield is the gate.
- Phase 2: `me6i` (P1, reframed in v4 per j98h finding 4)
- Phase 3: `pfpn` (= renamed adp-res; P1; **promoted to highest-leverage Mac knob in v4**)
- Phase 4: `m59h` (already P1 BLOCKED post-69ip; v4 reflects)
- Cross-cutting: `eff-hyg` (P3)
- Closed/landed: `xer4` (Cargo profile rename), `5sjh` (9k4w.1 ancestor-stack memo audit), `5m36` (9k4w.3 ray-octant mirror audit), `69ip` (Phase-0a fence-poll), `29xk` (Phase-0 256³ × 0.5 windowed), `j98h` (Phase-0 follow-up arms)
- Parent umbrella: `2w1u` (P0, claim released)
- Indirect parent: `dbz5` (perceived-FPS umbrella)

**Older closed beads, named here so they aren't re-litigated:**
- `dlse.2.2 exp#4` — `frame_latency` ∈ {1,2,3} swept null
- `dlse.2.2.2` — late-acquire reorder shipped null
- `dlse.2.2.3` — warmup transient characterized; v3 thought tonight's runs sampled it; v4 distinguishes me6i drift from dlse.2.2.3 per j98h finding 4
- `dlse.2.3` — proof-of-honest-timestamp; `bench_raycast_256 ≈ 0.19ms`
- `dlse.2.4` — render-pass attribution; **the bead that should have informed v1 of this doc**

**v4-cycle closed beads** (already in `Closed/landed` above; named here with one-liner for the audit trail):
- `69ip` — Phase-0a fence-poll experiment; backpressure interpretation confirmed (`render_gpu` 14.30 → 0.10 ms mean at 64³)
- `29xk` — Phase-0 256³ × 0.5 windowed steady-state; perf paper §3.16
- `j98h` — Phase-0 follow-up arms (64³ + 512³ + 256³ × 1.0); perf paper §3.16.1; **render-scale is the dominant Mac knob**

---

## What this doc is not (revised v4)

- **Not a bead.** Beads are filed and listed in Cross-references.
- **Not a commitment to ship.** Each bead goes through /ship-auto with its own review tier. Anything that doesn't measure as predicted gets reverted.
- **Not Mac-only.** v2 had a Mac-specific framing built around a wrong mechanism. v3 was cross-platform, with Apple-specific notes called out only where they are. v4 makes the asymmetry visible: Mac is now empirically measured (`render_gpu` ≈ 0.1 ms post-warmup); cross-platform leverage remains a literature projection (dubiousconst282 + Aokana 2025) until non-Mac steady-state captures it.
- **Not a replacement for §3.16 / §3.16.1 of the perf paper.** Those are the steady-state numbers; this doc is the roadmap that interprets them. If they conflict, §3.16 / §3.16.1 win.
- **Not measurement-complete.** Phase 0 is now DONE for the M2 + 50% / 1.0 × 64³ / 256³ / 512³ matrix; non-Mac steady-state and the 256³ × 0.5 × fullscreen-internal arm remain. Phase 0c (`szyh`) is the next experiment in the v4 roadmap.
