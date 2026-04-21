# SVDAG Performance Paper (Living Document)

**Ownership:** mayor owns project-level movement (decomposition, surfacing, keeping it advancing); crew owns execution (claims sub-beads of `hash-thing-stue`, writes into the paper). Roles defined in the `mayor` Claude skill.

**Status:** Skeleton. First real revisions pending (see bead `hash-thing-stue` children — claimable by any seat).

**Companion:** `docs/perf/svdag-perf-spec.md` — short re-derived intuition. The spec is regenerated from this paper, not edited in isolation. If the spec disagrees with the paper, the paper wins.

---

## 0. How to read this document

This is the **long form**. It is intentionally exhaustive and intentionally slow to update. The point is to maintain a top-down model of *what should be possible* on our reference hardware, against which measurements are interpreted.

**Don't adjust the model too quickly.** When measurement disagrees with the model, the default explanation is "we have a bug or architectural flaw," not "the model is wrong." The model only updates when alternatives have been genuinely ruled out.

---

## 1. Reference hardware

**Baseline target: M1 MacBook Air, 8 GB unified memory, 7- or 8-core integrated GPU.**

- Not edward's daily-driver M2 16GB. Numbers from M2 are diagnostic, not authoritative.
- 8 GB unified memory is the binding constraint on SVDAG node budget — it sets the ceiling on representable world detail.
- A possible future bump to 12 GB is tracked separately. Plan around 8 GB until then.
- Frame target: **60 fps sustained** at the demo world size (currently 256³, with ambition for larger).

Apple M1 GPU relevant headline numbers (to be filled in with citations during the literature pass):

- TODO: GFLOPS (FP32), memory bandwidth (GB/s), L1/L2 cache sizes per cluster
- TODO: Metal `timestamp_period` and how it relates to wgpu's reported value (M2 reports 1 ns; verify on M1)
- TODO: Drawable count and surface present cadence on macOS — relevant to dlse.2

---

## 2. Literature survey

First pass (2026-04-20, bead `hash-thing-stue.1`). The list below is the SPEC.md Rendering curation, ordered by when the contribution entered the lineage. Each entry captures core idea, reported number, implied number on M1 MBA, and verdict. Numbers marked **TODO-verify** were taken from secondary summaries because the primary PDFs would not decode through WebFetch during this pass — they are plausible and consistent across sources but have not been eyeballed against the original tables.

Cross-reference for implied-number math: M1 MBA 8-core GPU ≈ 2.6 TFLOPS FP32, ≈ 68 GB/s memory bandwidth, unified memory (no PCIe hop). The baseline papers all target discrete NVIDIA GPUs with 2-10× the bandwidth and 2-20× the compute. SVDAG traversal is dominated by dependent memory loads along the ray, so the **bandwidth ratio** is the right scaling lens, not raw GFLOPS.

### 2.1 Laine & Karras (2010), "Efficient Sparse Voxel Octrees" (ESVO)

- **Core idea.** First paper to make sparse voxel octrees competitive with triangle meshes on a GPU. Iterative descent with explicit ray-position tracking, child-pointer indirection, and per-node "contour" primitives that flatten smooth surfaces. All subsequent SVDAG / HashDAG traversal kernels (including ours) descend from this control-flow shape.
- **Reported number.** Rendering of contoured ESVO scenes in the low tens of milliseconds per frame on a single NVIDIA GTX 480 at 1024×768 primary rays — competitive with triangle rasterization at the time. TODO-verify against the I3D 2010 and extended technical-report tables (cached PDF exists but did not decode through WebFetch).
- **Implied on M1 MBA.** GTX 480 has ≈ 177 GB/s bandwidth and 1.35 TFLOPS; M1 8-core has ≈ 68 GB/s and 2.6 TFLOPS. Expect the same algorithm to land at roughly **2-3× the wall time** versus GTX 480 at the same resolution, i.e. ~30-60 ms/frame on M1 MBA at 1024×768 for large scenes. For our 960×600-ish (256³ at 50% render scale) that is ~20-35 ms/frame on the core descent — in the same order as our headless bench's 0.19 ms mean at 1920×1080 *for small worlds*, consistent with ESVO being memory-walk bound rather than ray-count bound.
- **Verdict.** **We already descend from this lineage.** Our WGSL traversal kernel is iteratively structured in the Laine-Karras sense. We do not use contours; our leaves are full 2³ material-tagged voxels, and contours would bias toward smooth analytic surfaces we do not have.

### 2.2 Kämpe, Sintorn, Assarsson (2013), "High Resolution Sparse Voxel DAGs"

- **Core idea.** Generalize SVO to a DAG by hash-consing subtrees. Empty and repeated regions both get deduplicated. Enables representing binary voxel scenes at 128K³ that would never fit as flat grids or plain SVOs.
- **Reported number.** 170 MRays/sec (ambient occlusion) / 240 MRays/sec (shadows) on NVIDIA GTX 680 at "competitive with or faster than state-of-the-art triangle ray tracing" for the Epic Citadel scene voxelized to 128K³, **945 MB GPU memory** vs. ~5.1 GB for the equivalent SVO. (Source: semanticscholar summary; TODO-verify against the SIGGRAPH 2013 PDF.)
- **Implied on M1 MBA.** GTX 680 has ≈ 192 GB/s bandwidth and 3.1 TFLOPS. M1 8-core is ~3× less bandwidth and ~1.2× less compute. Expect **~50-80 MRays/s primary / shadow** on the same algorithm at the same memory layout, for a scene that actually fits. Our 256³ demo at 50% render scale is ~576k rays/frame, so this predicts ~7-12 ms/frame for just the SVDAG traversal. Our headless bench already reports 0.19 ms mean — but our world is a 256³ instead of 128K³, and the tree is much shallower (log₂ 256 = 8 vs. log₂ 128K ≈ 17), so we do ~half the per-ray memory walk. The numbers are internally consistent.
- **Verdict.** **Our GPU node encoding and upload shape descend from here, but our update model does not.** Kämpe's SVDAG is build-once-render-forever: a scene is offline-DAG-ified and then traced as a static structure. Our implementation rebuilds the CPU-side SVDAG serialization *every sim tick* from the live hash-consed octree — see §4.6 for how that differs from both Kämpe (static) and HashDAG (GPU-resident dynamic). Our octree is still hash-consed; our GPU buffer layout is still Kämpe-shaped. But "this is the paper we are implementing" undersells the delta on update semantics.

### 2.3 Villanueva, Marton, Gobbetti (2016/2017), "Symmetry-aware Sparse Voxel DAGs" (SSVDAG)

- **Core idea.** Extend DAG deduplication to include the mirror and rotational symmetries of a subtree, not just structural equality. Typical scenes have enormous amounts of mirror symmetry (plane-symmetric buildings, rotation-symmetric props). Compression is pure memory win — traversal cost per ray is essentially unchanged because the shader just xors a transform bit into its position update.
- **Reported number.** State-of-the-art lossless compression over SVDAG on CAD models, 3D scans, and game scenes — typical 2-4× additional memory saving on top of SVDAG for scenes with rich symmetry. Real-time GPU tracing at sub-millimetric precision on the full Boeing 777. TODO-verify exact FPS / ms numbers against the I3D 2016 paper (cached PDF did not decode).
- **Implied on M1 MBA.** Traversal shader cost is within a few percent of plain SVDAG, so implied throughput is ~50-80 MRays/s (same as §2.2). The win is the **memory ceiling**: where SVDAG at 256³ already fits comfortably, SSVDAG matters if we push toward 2048³+ on 8 GB unified memory, because symmetry-heavy CA rules may produce large mirror-redundant regions.
- **Verdict.** **Could adopt, not today.** Our CA rules are not symmetry-seeking by design (gravity breaks vertical symmetry; sand/water flow is directional). Expected win for *our scenes* is smaller than for CAD / architectural assets. Keep on the options shelf for post-1024³ scale work.

### 2.4 Careil, Billeter, Eisemann (2020), "Interactively Modifying Compressed Sparse Voxel Representations" (HashDAG)

- **Core idea.** Embed the DAG into a GPU hash table so subtrees can be found, inserted, and reused *during editing* without decompressing and re-hash-consing the whole tree. Editing works bottom-up: compute new leaves, look them up in the hash table, promote found-or-inserted pointers up the tree. Reference implementation at <https://github.com/Phyronnaz/HashDAG>.
- **Reported number.** Interactive carving, filling, copying, and painting on 64K³ and 128K³ scenes at rates described as "interactive" (single-digit-ms edit latencies for localized strokes). TODO-verify per-edit latency table against CGF 2020.
- **Implied on M1 MBA.** The critical-path operation is a single GPU hash probe per leaf being edited, plus log-height promotion. Hash-probe cost is bandwidth-bound on unified memory, so the per-edit latency should scale with our bandwidth ratio (roughly 3× slower than the reported discrete-GPU numbers). Expect **low-ms per localized edit** on M1 MBA at 256³–1024³, assuming the hash table fits in the ≈ 8 GB budget.
- **Verdict.** **Closest match to our use case. Already on the Phase-B roadmap** (SPEC §Rendering step 5). We need dynamic editing because CA rules mutate the world every tick. This is the target shape for the hash-consed-store-on-GPU that our current CPU-side hashlife approximates. Pure "adopt next."

### 2.5 Molenaar & Eisemann (2023), "Editing Compressed High-resolution Voxel Scenes with Attributes"

- **Core idea.** HashDAG as published handles only binary occupancy. This paper extends it with compressed per-voxel attributes (colors, materials) decoupled from geometry and laid out along a 3D Morton curve, indexed by popcount of non-empty voxels preceding the current voxel. Provides one lossless and one lossy attribute-compression variant, both compatible with interactive editing.
- **Reported number.** Benchmarked on Intel i9 (10th gen) + NVIDIA RTX 3070 Ti, Linux, on voxelized Epic Citadel / Lumberyard Bistro / San Miguel. Comparisons against Dado (lossless) and Dolonius (lossy) schemes. TODO-verify exact edit and read latencies from CGF 2023 table.
- **Implied on M1 MBA.** RTX 3070 Ti has ≈ 608 GB/s memory bandwidth; M1 8-core has ≈ 68 GB/s, a 9× gap. Expect **~5-10× slower per attribute lookup** on M1 for the lossless path, which is still sub-ms for localized reads. The 3D Morton-curve indexing is compute-cheap and bandwidth-friendly, so the ratio holds.
- **Verdict.** **Required for our use case.** Our CA is material-typed, not binary — "sand" ≠ "water" ≠ "air" matters. HashDAG alone does not give us this. Adopt together with §2.4; they are a pair. Lossless path likely the right default for CA (distinct material types don't tolerate quantization).

### 2.6 Sparse 64-trees (dubiousconst282, 2024, blog + reference impl)

- **Core idea.** Replace the 8-ary SVO / SVDAG with a 64-ary tree where each node stores a 64-bit occupancy mask covering a 4×4×4 block of children. Child lookup is a 64-bit popcount of the masked mask. Shallower tree means fewer memory indirections per ray; larger nodes mean fewer cache lines touched. Dedicated guide at <https://dubiousconst282.github.io/2024/10/03/voxel-ray-tracing/>; reference impl at <https://github.com/dubiousconst282/VoxelRT>.
- **Reported number (and why it is the most useful one in this survey).** Benchmarked on an **integrated GPU**, same hardware class as ours: 6,358 cycles/ray on the Bistro scene voxelized at 8K³ after all optimizations (memoized ancestors, coalesced single-cell skips, ray-octant mirroring), down from 16,903 cycles naive. Memory footprint ~0.62 B/voxel on Bistro (~0.19 B/voxel theoretical best) vs. ~1.02 B/voxel for the equivalent ESVO — 40-80% smaller than an 8-ary tree on real data.
- **Implied on M1 MBA.** This *is* our hardware class. At a rough 1 GHz iGPU shader clock and 6,358 cycles/ray, the raw ray cost is ~6.4 µs/ray — but that's serial cost; with 1024 SIMD lanes live, per-shader-core throughput is ~1 Gray/s, and M1 has 8 cores, so aggregate ceiling ~8 Gray/s. Our 576k rays/frame fits in <0.1 ms of compute under that model, which is within a factor of 2 of our measured headless 0.19 ms bench. **The 64-tree numbers predict our real measurement, which is a strong signal the underlying per-ray model is right.**
- **Verdict.** **Strongest candidate for post-SVDAG refactor.** Memory density beats SVDAG on real scenes. Shader is simpler (single popcount per step vs. the chained child-mask test + child-pointer fetch of SVDAG). Hashlife-style memoization would need rework — the 64-ary factor breaks our current 8-ary hash-cons boundaries, so it is not a drop-in. Parked as a future optimization with a clear expected win (~1.5-2× throughput) but requiring simultaneous changes to the sim-side octree.

---

**Not yet surveyed (follow-up beads):** Transform-Aware SVDAGs (2025), Occupancy-in-memory-location (Modisett), Aokana (2025), Hybrid voxel formats (2024). These are listed in SPEC.md §Rendering and will be covered once the core lineage (2.1-2.6) is rock-solid. File as `hash-thing-stue.1.x` children.

**For each paper, §2 captures:** core idea in one paragraph; the performance number they report and on what hardware; the number their technique implies on our reference hardware (M1 MBA), accounting for clock/bandwidth/cache differences; whether their technique is something we already do, could adopt, or have ruled out — and why.

---

## 3. Theoretical frame-cost model

What *should* a frame cost on the reference hardware at 256³, 50% render scale (so ~960×600 ≈ 576k rays)?

First pass (2026-04-20, bead `hash-thing-stue.1`-follow-on = `hash-thing-stue.2`). The model is built bottom-up from node-encoding details (`crates/ht-render/src/svdag.rs`) and published M1 specs. Expect precision no better than 2×; the purpose is to catch >10× outliers, not predict the fourth decimal.

### 3.1 Reference hardware specs used by the model

Apple M1 (8-core GPU, MacBook Air):
- **FP32 throughput: ≈ 2.6 TFLOPS** (8 cores × 128 ALUs × 2 FMA ops × ~1.28 GHz).
- **Unified memory bandwidth: ≈ 68 GB/s** (LPDDR4X-4266, 128-bit bus).
- **GPU shader clock: ~1 GHz nominal, ~1.28 GHz boost.**
- **L2 cache: ~4 MB shared across GPU cores; L1 per core ~16 KB.**
- No discrete memory — CPU and GPU share the same 8 GB pool and the same bandwidth ceiling. Any CPU-side traffic during a frame (sim writes, SVDAG uploads, audio DMA) competes with raycast loads for the same 68 GB/s.

TODO-verify: L1 cache size per GPU core; most public sources report "per-cluster" caches but Apple does not publish exact line-size and associativity. 64 B cache lines is a conservative assumption borrowed from the Armv8.6-A inheritance.

### 3.2 Node encoding (from the source)

Our serialized interior node is **9 u32 = 36 bytes**:
- `slot[0]`: child_mask (low 8 bits = octant occupancy, bits 8-23 = representative material for LOD shading).
- `slot[1..=8]`: eight child entries, each a packed `(is_leaf << 31) | payload`. Leaves are inline — no separate leaf-node fetch — with the 16-bit material state in `payload`.

Per traversal step inside the shader:

1. **One dependent u32×9 load** for the current node's 36 bytes.
2. **O(1) arithmetic** to compute the octant index from the ray's position relative to the node's midpoint.
3. **One indexed read** into `slot[1 + octant]` — already in registers from step 1, no additional memory.
4. Branch: `is_leaf` → shade and exit; else follow `payload` (child offset) back to step 1.

So one traversal step costs **~36 B of GPU memory traffic** plus a handful of ALU ops. No hash probes (unlike HashDAG); no popcount (unlike 64-tree); no contour decode (unlike ESVO).

### 3.3 Per-ray cost

**Average traversal steps.** Worst case at 256³ is `log₂(256) = 8` levels of descent, but the relevant quantity for bandwidth is **total DDA steps per ray** (descend + step-past + pop), not just descent depth. Measured empirically for the **default-spawn primary-ray sample** at 256³ / 960×540 (hash-thing-stue.5, `tests/bench_depth_histogram.rs`, seeded terrain via `TerrainParams::for_level(8)`):

| Ray class | Share | Mean steps | p50 | p95 | Max |
|---|---|---|---|---|---|
| Hit (terrain) | 53% | 20.0 | 17 | 38 | 84 |
| Miss (sky) | 47% | 9.3 | 9 | 22 | 64 |
| **All primary rays** | **100%** | **15.0** | **13** | **34** | **84** |

Two cross-check poses from the same harness: looking straight down (100% hit, mean 6.8, p95 10) and horizontal-at-mid (y=0.5, 55% hit, mean 13.2). No rays exhausted the step budget in 3×518k samples.

The prior model guessed ~4 steps/ray. The measured mean is **~4× larger**. The under-count came from conflating "descent depth" (~5-6 for terrain at 256³) with "total DDA steps" — each sibling the ray grazes past counts, and the integer-DDA step-past + pop machinery produces several steps per hit beyond the final descent chain. Sky rays terminate much faster than hits because the root cube's empty neighborhood lets them exit after a few coarse steps, but they are not a tiny fraction of the frame: ~47% at default-spawn.

*Scope note:* these numbers are for the default-spawn primary-ray sample only. Secondary rays (shadows, GI) would have a different distribution; broader scene coverage would need a live-telemetry path, out of scope for stue.5.

**Per-step latency.** On an integrated unified-memory GPU, a dependent u32×9 load:
- L1 hit (same cache line as previous step, common when rays in a warp share ancestors): ~4-8 ns amortized.
- L2 hit (shared 4 MB, hit rate high because the frontier of nodes being traversed fits): ~15-30 ns.
- Memory miss (unlikely inside a hot frame because the reachable node set at 256³ is ≪ 4 MB — see §3.5): 150-200 ns.

Assume an 80/20 L1/L2 mix for the typical ray: `0.8 × 6 ns + 0.2 × 22 ns ≈ 9 ns per step`.

**Per ray:** 15 steps × 9 ns ≈ **135 ns** (using the measured mean from §3.3). SIMD: rays in a 32-lane warp share ancestor loads, so the *warp* costs more like `max(per-lane steps) × per-step` with coherent coalescing. For primary rays with tight screen-space locality, expected throughput is near the bandwidth ceiling, not per-lane latency.

### 3.4 Frame cost at 256³ × 50% render scale

576k rays × 15 steps/ray (measured mean, §3.3) × 36 B/step = **~311 MB of node traffic per frame**.

At 68 GB/s, this is `311 MB / 68 GB/s ≈ 4.6 ms` in bandwidth-limited form. The shader also writes 576k pixels × 8 B (Rgba16Float) = 4.6 MB per frame — another ~0.07 ms. Plus instruction-issue overhead and imperfect bandwidth utilization (realistic peak is ~60% of spec on Apple Silicon for ray-tracing workloads) gives a **predicted envelope of ≈ 7.6 ms / frame** on M1 MBA for pure SVDAG traversal + raycast write. (Pre-stue.5 this section predicted ~2 ms using a 4-step guess; measurement now pushes it upward to ~7.6 ms, *widening* the gap vs the ~0.24 ms measurement — see §3.6.)

**Total frame at 60 FPS target = 16.67 ms.** If the naive 7.6 ms bandwidth envelope were reality, SVDAG traversal alone would eat nearly half the budget. In practice measurement sits near 0.24 ms on M1-implied (§3.6), so the envelope is a *loose upper bound* — the real frame cost is set by how well the per-step cost amortizes against cache reuse, not the worst-case step × bytes product.

### 3.5 Working-set check

At 256³ our hash-consed DAG typically has on the order of 10⁴ to 10⁵ unique interior nodes (measured on steady-state scenes; TODO-verify with a live histogram). At 36 B/node, the entire reachable DAG is **0.4 MB to 4 MB**, which fits the M1 L2. That is why we should not be seeing DRAM-latency (150 ns) loads inside the traversal hot path at this scale.

This is the model's strongest prediction: **at 256³ the SVDAG frame-cost on M1 MBA should be bounded by L1/L2 bandwidth, not DRAM bandwidth and not compute.** If measurement shows DRAM-bound behavior, the model is wrong *or* the working set is somehow cache-thrashing.

### 3.6 Gap vs measurement

The gap-report row (see §5) compares the §3.4 envelope (7.6 ms post-stue.5) against:

- **Headless bench (`bench_gpu_raycast`) at 256³ app-spawn, 1920×1080 on M2 16 GB:** 0.19 ms mean / 0.45 ms p95. Scaled down to M1 MBA (≈ 0.67× memory bandwidth, roughly 1.5× runtime): **predicted M1 MBA headless ≈ 0.3 ms at 1920×1080, ≈ 0.1 ms at our 960×600 windowed target.**
- **Windowed app at 256³ / 50% on M2 16 GB (post-dlse.2.3, commit 8648dc0):** render_gpu ≈ 0.16 ms mean. Scaled to M1: **predicted ≈ 0.24 ms**. Consistent with the bench.
- **Windowed app at 256³ / 50% on M2 16 GB (pre-dlse.2.3):** ~30 ms mean. This was a measurement artifact from ComputePassTimestampWrites charging Metal barrier waits to the compute pass timestamp; the underlying wall time was never 30 ms. See dlse.2.3 for the forensic chain and spark/cairn's hypothesis triage history.

The model predicts **7.6 ms envelope** (post-stue.5, with measured mean = 15 steps/ray) and measurement is **0.24 ms on M1-implied**. Gap is **~32× over-prediction** by the model. Candidate explanations, re-weighted after stue.5 ruled out the "depth too short" hypothesis:

1. ~~Effective traversal depth is shorter than 4.~~ **Refuted by hash-thing-stue.5:** measured mean is ~15 steps/ray, *higher* than the pre-stue.5 guess of 4. Whatever is closing the gap, it is not a shorter effective traversal. This correction makes the gap wider, not narrower — the other two explanations must together account for a ~32× factor instead of ~8×.
2. **Per-step cost is smaller than 9 ns** — the working set fits entirely in L1 per tile, and the shader-wide prefetch overlaps the next load with current ALU. The 9 ns figure assumed an 80/20 L1/L2 mix; if reuse is nearly 100% L1 for primary rays (warp shares the same ancestor chain for most of descent), per-step cost could plausibly be 2-3 ns.
3. **The 60% bandwidth-utilization factor is too pessimistic for this access pattern** — SVDAG access is warp-coherent and hits the DAG hot set hard; utilization closer to 80-90%. More importantly, once the working set is L1-resident, "bandwidth" is L1 bandwidth (per-SM, ~1 TB/s class), not the 68 GB/s unified-memory spec — a full order of magnitude above what the envelope assumes.

Hypotheses 2 and 3 together (`2-3 ns/step × L1-bandwidth regime`) are the **leading-hypothesis** explanation for the 32× factor. stue.5 does not directly measure cache residency or utilization — it refutes explanation 1 and narrows the search, but does not pin the remaining cause. Direct confirmation would take a GPU-side counter pass (hit-rate / throughput telemetry) out of scope for this bead. Treat the paper's bandwidth envelope as a DRAM-limit ceiling, not a predicted steady-state cost.

This is the **"paper wins"** direction from §0: measurement is faster than the model expected, which means the model is conservative and we have headroom to burn on correctness features (attributes, LOD) before we approach the ceiling.

### 3.7 Windowed-path gap (dlse)

The 30 FPS bug on M2 at 256³ / 50% is **not** explained by traversal cost — dlse.2.3 pinned real compute at ~0.16 ms. The 34 ms/frame budget is dominated by `surface_acquire_cpu ≈ 25 ms`. This is outside §3's scope; it is a compositor / swapchain-pacing question, owned by `hash-thing-dlse.2.2` (drawable starvation deep-dive). The perf-paper correlate is §4 ("SVDAG ↔ memo-step interaction") — or, more broadly, that the model should be extended with a §3.8 on surface / presentation cost once we have empirical per-OS numbers.

### 3.8 Present-path inventory (macOS M2 Metal, measured)

Step 1 of `hash-thing-dlse.2.2` logged `surface.get_capabilities` at renderer init (landed commit `4f60ddc` on main, 2026-04-21). M2 MacBook results:

- `surface_caps.present_modes = [Fifo, Immediate]`. **No Mailbox.** wgpu's `AutoVsync` → Fifo, `AutoNoVsync` → Immediate on Metal.
- `surface_caps.alpha_modes = [Opaque, PostMultiplied]`. Selected Opaque.
- `surface_caps.formats = [Bgra8UnormSrgb, Bgra8Unorm, Rgba16Float, Rgb10a2Unorm]`. Selected Bgra8UnormSrgb.
- `desired_maximum_frame_latency = 2`.
- Window at default launch: physical 2940×1782 (retina 2×), render_scale 0.5 → render target 1470×891 (~1.3 M pixels).

Consequences for the `surface_acquire_cpu ≈ 25 ms` hypothesis:
- **No Mailbox** rules out "triple-buffer-style unlocked present was available and we just didn't use it." The wgpu present-mode menu on Metal is thinner than on DXGI or Vulkan.
- moss previously tried `AutoNoVsync` (Immediate) with no improvement. Immediate should have near-zero present-side blocking, yet 25 ms persisted. The wait is therefore **not** a "waiting for vblank / queue full" in the naïve sense.
- The most plausible remaining causes, in order: (a) CoreAnimation compositor tiling of windowed `CAMetalLayer` forcing a compositor fence per frame; (b) a driver-side acquire throttle that serializes across queue submissions; (c) our own pipeline inserting an implicit CPU wait (e.g., the timestamp resolve mapping path introducing a fence on the wrong submit).

Step 2 (fullscreen borderless) tests hypothesis (a). Step 3 (off-surface render target) tests (b)+(c) by bypassing `surface.get_current_texture()` entirely.

**Cold-frame confound check (bead `hash-thing-6hta`, spark 2026-04-20).** The `Perf` ring buffer is a FIFO of 64 samples per metric with no cold-frame skip (`src/perf.rs:61-87`, `src/main.rs:2036` records every frame's `surface_acquire` unconditionally). At ~30 FPS that evicts cold frames within ~2.1 s of wall-clock, so the *first* `log::info` line at `LOG_INTERVAL_SECS = 2.0` s is partially cold-contaminated, but lines 2+ are steady-state. moss's 2026-04-15 repro and spark's 2026-04-19 "three consecutive log lines" observation both sampled steady-state lines, not the opening line. The existing **`C` keybind** (`perf.clear()`, `src/main.rs`) is the correct drain-and-measure mechanism when isolating warm samples is needed. Conclusion: the confound exists in principle but does not explain the dlse.2.2 ~25 ms sustained measurement.

---

## 4. SVDAG ↔ memo-step interaction

Hashlife memo (`step_cache`) and the SVDAG share a deep structural assumption: **subtrees are deduplicated and addressed by content.** This section owns the architectural surface where they meet.

First pass (2026-04-20, bead `hash-thing-stue.3`, spark). Measurements from `tests/bench_svdag.rs::bench_svdag_step_deltas_*`, release profile, seeded world = level-appropriate terrain + `seed_water_and_sand`, warmup=5 steps, measured=40 steps.

### 4.1 Shared store or siloed?

**Shared.** There is a single canonical `NodeStore` (`crates/ht-octree/src/store.rs:16`) per `World`. Both the hashlife memo and the SVDAG build read from it and address content by the same `NodeId` type.

- `World` owns the store: `pub store: NodeStore` (implied by `src/sim/world.rs:187-200` — the hashlife caches are fields on `World`, keyed by `(NodeId, parity)` / `(NodeId, generation)` / `NodeId`).
- Hashlife memo keys **are** store NodeIds (`src/sim/hashlife.rs:228-265`).
- `Svdag::update(&NodeStore, root: NodeId, ...)` descends that same store (`crates/ht-render/src/svdag.rs:124`); its persistent `id_to_offset: FxHashMap<NodeId, u32>` caches render-side offsets keyed by store NodeIds.
- When `NodeStore::compacted_with_remap_keeping` fires (`src/sim/world.rs:1938`, from `hashlife::maybe_compact`), the same remap table is fed into the SVDAG (`src/main.rs:876` via `Svdag::apply_remap`) and into the hashlife caches (`hashlife::remap_caches`). Both sides stay valid across compaction.

**What is siloed is the content-indirection layer, not the store.** The SVDAG maintains two additional caches — `offset_by_slot: FxHashMap<[u32; 9], u32>` (slot-bytes → GPU offset; cross-epoch content dedup) and `id_to_offset` above. These live entirely inside `Svdag`; the sim doesn't see them. They exist because the GPU buffer is an append-only serialization with different stability semantics than `NodeId` (NodeIds churn on compaction; slot bytes are epoch-stable). This is the right shape — pretending the render buffer and the sim store share a single address space would break the incremental upload path.

**Cost of the current split:** the SVDAG carries two hashmaps over the reachable-node set (`offset_by_slot` and `id_to_offset`, ~48 B + ~12 B per entry on a typical LP64 allocator). At 256³ steady-state, `offset_by_slot.len()` converges to the total-slot count including stale slots from earlier epochs — bounded by `stale_ratio()` which triggers `Svdag::compact` at 50% (`src/main.rs:882`). So the render-side bookkeeping stays O(reachable) within a factor of 2.

**Verdict:** shared store, siloed render-side cache. Nothing duplicates the actual octree content; the extra maps are address-translation tables between stable-content and stable-position. No action.

### 4.2 Memo churn vs SVDAG compaction

**Measured at 256³ (terrain + water/sand, 40 warm steps):**

- **Hashlife cache misses (new subtree computations): ~1,847 per step** (73,884 summed).
- **SVDAG new-slot appends: ~544 per step** (mean 4,892 u32s/step ÷ 9 u32s per interior slot).
- **Cache hits: ~2,222 per step** (88,899 summed), **hit rate: 54.6%**.
- **Empty-subtree skips: ~1,670/step** (66,798 summed) — no CA work needed for empty subtrees.
- **Fixed-point skips: ~930/step** (37,219 summed) — inert uniform subtrees that step to themselves.
- **Short-circuit rate (hits + empty + fp) / (hits + misses + empty + fp): ~72%.** Only ~28% of subtree descents actually run the CA kernel.
- **SVDAG compactions fired in 40 steps: 0.** The 50% stale-ratio threshold was never hit.

**Measured at 64³ (same seed pattern):**

- Cache misses ~412/step; SVDAG appends ~115 slots/step; hit rate 45.2%; short-circuit rate ~65%.
- Scales roughly with volume (64× larger volume ≈ 4.5× misses, not 64× — the extra short-circuits absorb most of the growth).

**Implications:**

1. **Memo hit rate alone under-reports memo effectiveness by a lot.** At 256³ the raw hit/miss ratio is 55%, but counting the empty- and fixed-point-skip paths as "did not do CA work," the effective short-circuit rate is ~72%. A plain hit-rate graph would give a false-pessimism signal in steady state.
2. **SVDAG absorbs memo churn well.** Of ~1,847 new subtrees per step, only ~544 translate to SVDAG slot appends (~30%). The gap is because `offset_by_slot` deduplicates by slot bytes across epochs — many "new" NodeIds carry slot bytes already present from prior epochs. So SVDAG slot count grows slower than hashlife memo size in steady state.
3. **We are not near the degenerate limit.** At 256³, a full-reachable-set churn per step would be ~7,340 slot appends/step (from the cold build). We see ~544/step — two orders of magnitude below. Edward's hypothesis ("only parts of the world mutate, and the DAG is super compressible — diff-based uploads should be tiny") is confirmed for this scenario.

### 4.3 Cache locality across the boundary

**Empirical answer (proxy 1, hash-thing-stue.7): contention is present and material at the boundary, ~30 % of windowed frame cost on M2 MBA at 256³.**

**Direct counters remain unavailable.** A causal answer requires shared-L2/L3 traffic counters separated by producer (sim writes vs renderer reads), which Metal does not expose to unprivileged processes. `powermetrics --samplers gpu_power` gives aggregate GPU memory bandwidth but not eviction-attributed-to-side-A.

**Proxy 1 — sim-frozen vs sim-running per-frame latency.** `HASH_THING_FREEZE_SIM=1` disables `maybe_start_background_step` so the sim never advances; renderer keeps reading the same SVDAG every frame. Bench profile, 256³ default scene (terrain + water + sand), windowed at default 50 % render scale, M2 MBA 8 GB:

| run                 | render_cpu mean | render_gpu mean | surface_acquire_cpu mean |
|---------------------|-----------------|-----------------|--------------------------|
| sim-running (n=4)   | ~26.9 ms        | ~0.11 ms        | ~26.5 ms                 |
| sim-frozen  (n=5)   | ~18.8 ms        | ~0.08 ms        | ~18.5 ms                 |
| **delta**           | **−8.1 ms (−30 %)** | −0.03 ms    | **−8.0 ms (−30 %)**      |

Samples taken from a 14-second windowed run, dropping the first log line (warmup) for each mode. Data: `.ship-notes/stue.7-cache-locality-2026-04-21.md` (raw log lines preserved alongside the analysis).

**What proxy 1 cannot distinguish.** This delta could be driven by either:

- **CPU contention.** `step_recursive` is a hot single-threaded user of one P-core. Freezing sim returns that core's cycles to OS / wgpu / display-server work that surface acquire transitively waits on.
- **Unified-memory cache contention.** Sim writes evict renderer-relevant lines from shared L2; freezing sim leaves renderer's working set warm.

The two hypotheses predict the same `surface_acquire_cpu` reduction. Disambiguating needs proxy 2 (same-core vs different-core sim scheduling), which is open as `hash-thing-xhi6` follow-up.

**What proxy 1 does establish.**
1. Renderer-only steady-state at 256 ³ on this hardware lives at **~18 ms surface_acquire_cpu**, not the ~26 ms we see with sim live. The remaining ~18 ms is a windowed-presentation floor (consistent with `dlse.2.2` findings — that path is acquire-dominated independently of sim).
2. **Sim does meaningfully impact frame budget**, even though the renderer reads a coherent SVDAG snapshot per frame and the sim runs on a separate `std::thread`. The "background sim is free" assumption is wrong by ~8 ms at this scale.
3. **render_gpu is unaffected** (0.11 → 0.08 ms, both well below noise). The contention shows up in CPU-side acquire, not on-GPU compute. So at this scale, a GPU-side memo (parent bead `hash-thing-abwm`) is not motivated by *renderer* contention; it would be motivated by step latency itself.

**Implication for §3 frame budget.** The §3.1 budget assumed sim and render were independent on a separate-thread M-series. They aren't quite — sim costs the renderer ~8 ms / frame at 256 ³. At 4096³ the sim cost is ~64× larger (linear in node count for active-region work) but rendering is also more memory-bound, so the contention term may grow faster than linear. Re-derive at 4096³ when `hash-thing-ivms` lands.

**Verdict status:** **contention present, magnitude ~30 % of windowed frame at 256³, driver (CPU vs cache) unresolved.** Follow-up bead filed for proxy 2.

### 4.4 Per-step upload volume to `Svdag::nodes`

**Measured at 256³ (terrain + water/sand, 40 warm steps), release profile:**

- **Upload bytes/step: mean 19.6 KB, max 79 KB.**
- **New u32s appended to `Svdag::nodes` per step: mean 4,892 (~544 new 9-u32 slots), max 19,782 (~2,198 slots).**
- **Full buffer at cold: 66,061 u32s = 258 KB (7,340 reachable nodes).** A full re-upload is 13× the mean incremental upload.
- **Compactions fired in 40 steps: 0.** No full re-uploads happened in the measurement window.

**At 256³, per-frame SVDAG upload traffic is O(10 KB), not O(256 KB).** At 60 FPS this is ~1.2 MB/s upload, well under 0.01% of the M1 68 GB/s memory bandwidth ceiling. Edward's hypothesis is confirmed.

**Upload path is architecturally O(new-content).** Inspection of `Renderer::upload_svdag` (`crates/ht-render/src/renderer.rs:1325`) shows:
- Slot 0 (4-byte root header) is always rewritten.
- The tail `dag.nodes[prev_len..new_len]` is appended.
- A full re-upload only fires when the GPU buffer grows (doubling) or when `Svdag::compact` runs (stale_ratio > 0.5, rare).

**Measured at 64³ for reference:** mean 4 KB/step, max 10 KB/step. Scales sub-linearly with volume (64× volume ≈ 5× upload bytes).

**Implication for §5 (gap report):** SVDAG upload traffic is not a frame-budget concern at 256³, and on the current trend not at 512³ either. Any perf regression that makes upload look expensive is probably a different bug (e.g. unnecessary full re-uploads, buffer thrash, slot dedup broken).

### 4.5 Verdict on the boundary

**No fundamental architectural limit at the SVDAG ↔ memo boundary at 256³.** The two subsystems share a NodeStore; the render-side address-translation maps add bounded overhead; memo short-circuits absorb >70% of subtree descents; SVDAG uploads are O(new-content) at ~10 KB/frame scale. Cache locality across the boundary is the one question we cannot answer with current tools; the proxies in §4.3 will close that question.

The *next* scale where the boundary might matter is the streaming regime (`hash-thing-cswp`): with 4096³+ worlds and view-distance streaming, the reachable-set churn per view-shift is much larger than per-generation churn, and the `offset_by_slot` cache growth becomes a real memory consideration. File a new `stue` child when the streaming work gets close.

### 4.6 Update model comparison (static / GPU-dynamic / rebuild-per-tick)

Our SVDAG is **dynamic**, not static. The literature has two published points on the update-model axis, and we sit at a third.

| System | Update model | Where it lives | Update cost |
|---|---|---|---|
| Kämpe 2013 (§2.2) | **Static.** Build once, trace forever. | CPU offline, then frozen GPU buffer. | No updates. A scene change requires a full offline rebuild. |
| HashDAG / Careil 2020 (§2.4) | **GPU-dynamic.** Edits mutate the same DAG in place. | GPU hash table resident in VRAM. | Single GPU hash probe per edited leaf + log-height promotion. Sub-millisecond localized edits on discrete GPUs. |
| **This codebase** | **CPU-dynamic rebuild-per-tick.** Rebuild the SVDAG serialization each sim tick from the live hash-consed octree, upload the diff. | CPU `NodeStore` + `Svdag::nodes` + `offset_by_slot`; GPU just holds the serialized bytes. | O(new slots) per step. Measured at 256³ steady state: ~544 new slots/step → ~19.6 KB appended/step (§4.4). Update depth is not a special case in the code path. |

**Concrete evidence we handle arbitrary-depth updates natively.** The Margolus stepper in `sim/margolus.rs` mutates the octree at whatever depth the affected 2³ block lands at. The SVDAG rebuild in `crates/ht-render/src/svdag.rs` (via `Svdag::update`) walks whatever subset of NodeIds changed, hash-cons-dedups them against `offset_by_slot`, appends new slots to `Svdag::nodes`, and ships the tail. Nothing in that path is depth-conditional. The ~544 appends/step measurement at 256³ with water+sand churn (§4.4) is direct evidence that water-driven updates at leaf/near-leaf depth flow through the same path as higher-depth structural changes.

**Trade-offs of our model.**

- **Win over Kämpe (static):** the world is a live CA simulation; a static SVDAG would require a full rebuild per tick, which dominates the frame budget even at 256³ (cold rebuild is ~258 KB). We rebuild *the diff*, not the whole thing, and the diff is O(changed subtrees).
- **Win over HashDAG (GPU-dynamic):** no GPU hash table to maintain, no VRAM pressure from the table itself, no inserts racing with reads. We upload bytes, not hash-table operations. Portability win on platforms with limited unified memory (M1 MBA 8 GB target).
- **Loss vs HashDAG:** we pay an upload step per sim tick. At 256³ that's ~1.2 MB/s, deeply irrelevant relative to 68 GB/s unified-memory bandwidth (§4.4). At streaming-scale 4096³+ worlds this stops being free — see §5 gap report and `hash-thing-cswp`.
- **Loss vs Kämpe:** we spend CPU cycles rebuilding the serialization every tick. The rebuild is threaded with the background sim step (`x5w`), so it's not serialized against rendering, but CPU is not infinite and this is the single largest CPU cost outside the stepper itself.
- **Memory overhead specific to our path.** `offset_by_slot` (~48 B/entry) and `id_to_offset` (~12 B/entry) are address-translation tables between content-stable NodeIds and position-stable serialized offsets. Bounded by `Svdag::compact` at `stale_ratio > 0.5`. So render-side bookkeeping stays O(reachable) within a factor of 2 — see §4.1.

**Upload bandwidth degrades when dedup rate drops.** Engineered scenes (terrain + gravity-driven CA) hit ~99% dedup in steady state (4892 new u32s/step ÷ ~66k total u32s ≈ 7.4% of full re-upload). Fully-unique-churn scenes (pathological stress) fall back to linear-in-reachable-set. The current worst case we've measured is the 79 KB/step spike in §4.4, still two orders of magnitude under the cold-rebuild cost.

**Why this section exists.** Conversations around the perf paper occasionally lean on "SVDAG is static" framing from the Kämpe paper. That framing is wrong for our implementation and has architectural implications (edit latency, streaming design). This section pins the distinction so later sections (§5, §6) don't have to re-establish it.

---

## 5. Gap report

Measured vs theoretical, by subsystem. This is the operational output of the model — every entry should cite both the measurement and the model section it's compared against.

Format (placeholder):

| Subsystem | Measured (ref HW) | Theoretical (§3) | Gap | Status |
|---|---|---|---|---|
| Raycast traversal (headless, 256³, 1920×1080) | **0.19 ms mean / 0.45 ms p95** on M2 16 GB (`bench_gpu_raycast::bench_raycast_256_app_spawn`, spark 2026-04-15); M1-implied ≈ 0.3 ms | ~7.6 ms envelope on M1 MBA (§3.4, post-stue.5 with measured 15 steps/ray) | ~25× model over-predicts | Gap widened after hash-thing-stue.5 measured mean steps ≈ 15 (vs prior guess of 4), which pushed the envelope from ~2 ms to ~7.6 ms. The prior "depth < 4" gap explanation was refuted; leading hypothesis for the remaining factor is L1 cache residency + warp-coherent reuse (§3.6 items 2+3), not yet directly measured. |
| Raycast traversal (windowed, 256³ / 50% render scale) | **0.16 ms mean** on M2 16 GB, post-dlse.2.3 (commit 8648dc0, TIMESTAMP_QUERY_INSIDE_ENCODERS); M1-implied ≈ 0.24 ms | ~4.6 ms envelope on M1 MBA at 960×600 (§3.4 × 0.6 for smaller ray count, post-stue.5) | ~19× model over-predicts | Matches headless bench within factor of 2. Windowed ≠ slow; prior "30 ms" was measurement artifact. |
| SVDAG build (cold, 256³ terrain+water+sand) | **~1.08 ms** on M2 16 GB, release (`bench_svdag_step_deltas_256`, spark 2026-04-20); M1-implied ≈ 1.6 ms | Not yet in §3 (cold-build model) | N/A until modelled | One-shot cost; steady-state uses incremental path. |
| SVDAG incremental upload (256³, warm, per step) | **mean 19.6 KB, max 79 KB** on M2 16 GB (`bench_svdag_step_deltas_256`, spark 2026-04-20 — §4.4) | Not yet modelled (§3 is raycast-only) | N/A until modelled | O(new-content). ~1.2 MB/s at 60 FPS — negligible vs 68 GB/s ceiling. |
| SVDAG compaction | Not observed in 40-step warm window at 256³ (§4.4) | N/A | N/A | Fires only when `stale_ratio > 0.5`; not a steady-state cost. |
| Step (memo short-circuit rate, 256³ warm) | **~72% short-circuit** (hits + empty + fp) on M2 16 GB (`bench_svdag_step_deltas_256`, §4.2) | Not yet in §3 (memo model) | N/A until modelled | Raw hit rate ~55%; skips absorb another ~17%. |
| Step (memo miss / cold gen) | TODO | TODO | TODO | TODO |
| Surface acquire (windowed, 256³ / 50%) | **~25 ms/frame** on M2 16 GB (spark 2026-04-19, three consecutive log lines) | Not yet in §3 (surface/presentation model is §3.8 TODO) | N/A until modelled | **This is the real 30 FPS bug.** Owned by `hash-thing-dlse.2.2`. |

A gap of **>2×** is a flag for a bead. A gap of **>10×** is a P0 candidate. A gap near **1×** means the model agrees with reality and we can either ship at that perf or rederive the model with more aggressive techniques.

---

## 6. Open questions

Track here so they don't get lost between revisions.

1. M1 MBA `timestamp_period` confirmation. (M2 = 1 ns; assumed same on M1.)
2. dlse.2 hypotheses 1–3 (per spark): cross-frame texture serialization, drawable starvation, SVDAG content gap between bench and app — each needs a paper-side resolution, not just an empirical fix.
3. Does the per-cell tagged-cell encoding limit SVDAG compression vs an externalized attribute table (Dolonius)? Quantify.
4. What is the right SVDAG node budget for an 8 GB unified-memory machine, given that the OS, browser, and app share that pool?
5. Is there a useful intermediate: a coarser SVDAG for far-field rays plus a fine-grained one for near-field? Cost/benefit on M1.

---

## 7. Limitations and verdicts (what we are confident is *not* possible)

Reserved for things we have actually argued through to a confident "no." Empty for now. Filling this section is as valuable as filling the gap report — it stops us from re-investigating settled ground.

---

## 8. Revision log

| Date | Author | Change |
|---|---|---|
| 2026-04-20 | mayor | Skeleton landed. No content yet — see `hash-thing-stue` for charter and follow-up beads for first revisions. |
| 2026-04-20 | onyx | §2 first pass (bead stue.1). Six-paper survey: ESVO 2010, SVDAG 2013, SSVDAG 2016, HashDAG 2020, HashDAG-attributes 2023, sparse 64-trees 2024. Each with reported number, implied M1-MBA number, verdict. Several exact numbers flagged TODO-verify (PDFs did not decode via WebFetch); headline figures and verdicts stand. |
| 2026-04-20 | onyx | §3 theoretical frame-cost model (bead stue.2). Bottom-up derivation from M1 8-core GPU specs (2.6 TFLOPS, 68 GB/s) and the actual 36-byte interior-node encoding in `crates/ht-render/src/svdag.rs`. Predicts ~2 ms SVDAG traversal envelope at 256³ / 576k rays on M1 MBA. §5 gap report populated with raycast rows — measurement (post-dlse.2.3) is ~5-7× faster than model, consistent with shorter-than-assumed effective traversal depth. Surface-acquire row added to route dlse's 30 FPS bug to dlse.2.2; §3.8 surface/presentation model flagged as TODO. |
| 2026-04-21 | onyx | §3.8 present-path inventory (bead dlse.2.2 step 1, commit `4f60ddc`). macOS M2 Metal exposes only `[Fifo, Immediate]` — no Mailbox. Rules out "triple-buffer-style unlocked present was on the table." The moss/AutoNoVsync null result combined with no-Mailbox narrows the 25 ms `surface_acquire_cpu` bug to compositor fence or driver-internal serialization; step 2 and step 3 of dlse.2.2 will distinguish these. |
| 2026-04-20 | spark | §4 populated (bead stue.3). Answered Q1 (store is shared; render-side cache is a bounded address-translation layer), Q2 (~1,847 hashlife misses/step at 256³ → ~544 SVDAG slot appends/step; ~72% short-circuit rate), Q4 (~19.6 KB mean / ~79 KB max upload per step at 256³, confirming edward's diff-compressibility hypothesis). Q3 (cache locality) deferred with two empirical-proxy experiments for follow-up. Measurements land via new `tests/bench_svdag.rs::bench_svdag_step_deltas_*`. |
| 2026-04-20 | spark | §3.8 cold-frame confound check (bead `hash-thing-6hta`). `Perf` ring buffer is FIFO-64 with no cold-frame skip; at ~30 FPS cold frames evict within ~2.1 s, so the first log line is partially contaminated but steady-state lines (which all prior dlse.2.2 observations sampled) are clean. Does not invalidate the 25 ms surface_acquire finding. |
| 2026-04-21 | onyx | §2.2 verdict + new §4.6 "Update model comparison" (bead `hash-thing-jl4d`). Clarifies that our SVDAG is **dynamic** — rebuild-per-tick on CPU with O(new-slot) diff upload — rather than static Kämpe or GPU-resident HashDAG. Pins the distinction so later discussion does not re-establish it. Evidence: ~544 slot appends/step at 256³ warm with arbitrary-depth CA churn (§4.4) flows through the same code path as structural changes. |
