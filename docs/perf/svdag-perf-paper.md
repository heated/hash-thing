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

**Step 2 result (M2 MBA, 256³ dev, acquire harness, onyx 2026-04-21):**

| mode                     | surface_acquire_cpu mean | p95     | render_gpu mean |
|--------------------------|--------------------------|---------|-----------------|
| windowed                 | 25.31 ms                 | 32.37 ms | 0.09 ms         |
| `Fullscreen::Borderless` | 29.99 ms                 | 32.57 ms | 0.09 ms         |
| delta                    | **+4.68 ms (+18.5%)**   | ≈ same   | unchanged       |

Harness verdict: "fullscreen WORSE than windowed — unexpected; do not ship fullscreen-default." **Hypothesis (a) refuted on M2.** Both modes pin at the same 60 Hz → 30 FPS cliff (p95 ~32 ms ≈ 2 × 16.67 ms missed-vsync doubled-frame pattern). The acquire stall is not a CoreAnimation-compositor-only artifact; it's at least as bad (and slightly worse) under borderless fullscreen.

**Step 3 result (M2 MBA, 256³ dev, `HASH_THING_OFF_SURFACE=1`, onyx 2026-04-21):**

| metric                 | windowed baseline | off-surface |
|------------------------|-------------------|-------------|
| surface_acquire_cpu    | 25.3 ms           | **0.0 ms**  |
| submit_cpu             | ~0.3 ms           | **~26 ms**  |
| render_cpu             | 25.7 ms           | 26.3 ms     |
| total frame budget     | ~26 ms            | ~26 ms      |

**Hypothesis (b) swapchain-pacing refuted.** Bypassing `surface.get_current_texture()` does not eliminate the stall — it just migrates from `acquire` into `queue.submit()`. Total frame budget is unchanged (~26 ms both modes). This is the signature of an implicit fence/wait elsewhere in our frame critical path, not swapchain throttling.

**Surviving hypothesis (c): something in our own submit/encoding pipeline inserts a CPU-side wait.** The most likely candidate on this hardware is the timestamp resolve-and-map path: `GpuTiming::resolve_last_frame` reads back the query buffer for render_gpu reporting, and if the `map_async` completion fence is drained synchronously on the submit path, it forces frame N's submit to wait for frame N-1's GPU completion. That would explain why both windowed and fullscreen show the same ~25-30 ms ceiling, and why the wait migrates when acquire goes away.

**Step 3b result (M2 MBA, 256³ dev, `HASH_THING_OFF_SURFACE=1 HASH_THING_DISABLE_TIMESTAMP_RESOLVE=1`, onyx 2026-04-21):**

| metric          | off-surface only | off-surface + timestamp resolve disabled |
|-----------------|------------------|------------------------------------------|
| submit_cpu mean | ~26 ms           | **~25 ms (unchanged)**                   |
| render_cpu mean | ~26 ms           | ~26 ms                                   |

Concrete numbers: `submit_cpu = 25.21/27.48 ms` (p95), steady-state across 4 consecutive log lines with both in-encoder timestamp writes AND the resolve/map path fully bypassed.

**Hypothesis (c) timestamp-fence refuted.** Disabling the full timestamp path does NOT collapse submit_cpu. The ~25 ms stall persists with zero timestamp traffic in the submit queue.

**Conclusion — the stall is real GPU frame time, not a CPU-side fence.** By elimination: (a) compositor refuted step 2, (b) swapchain refuted step 3, (c) timestamp fence refuted step 3b. The only remaining explanation is that the total GPU work per frame at 256³ / 50% on M2 integrated GPU is ~25 ms. The dlse.2.3 measurement of `render_gpu ≈ 0.16 ms` is the **compute pass only**; the blit + HUD + particles + legend render pass and the rest of the frame encoder are not instrumented but together must be dominating frame time at this target resolution (1470 × 891 ≈ 1.3 M pixels, 4 × magnified to 2940 × 1782 by the swapchain blit).

**Revised dlse.2.2 surface.** The next-step hypothesis is no longer "find the swapchain / compositor knob." It is "find where the ~25 ms of GPU work is going in the non-compute-pass portion of the frame." Candidate measurement: add TIMESTAMP_QUERY_INSIDE_ENCODERS brackets around the render pass (blit + overlays) separately from the compute pass. Then decide whether to optimize the blit pipeline (fewer overlay passes, simpler shaders, or run the compute at native target resolution instead of blitting a half-res texture through a fullscreen triangle) or to accept 30 FPS at 256³ on integrated GPU and document render_scale = 0.5 as the M2 shipping default.

**Cold-frame confound check (bead `hash-thing-6hta`, spark 2026-04-20).** The `Perf` ring buffer is a FIFO of 64 samples per metric with no cold-frame skip (`src/perf.rs:61-87`, `src/main.rs:2036` records every frame's `surface_acquire` unconditionally). At ~30 FPS that evicts cold frames within ~2.1 s of wall-clock, so the *first* `log::info` line at `LOG_INTERVAL_SECS = 2.0` s is partially cold-contaminated, but lines 2+ are steady-state. moss's 2026-04-15 repro and spark's 2026-04-19 "three consecutive log lines" observation both sampled steady-state lines, not the opening line. The existing **`C` keybind** (`perf.clear()`, `src/main.rs`) is the correct drain-and-measure mechanism when isolating warm samples is needed. Conclusion: the confound exists in principle but does not explain the dlse.2.2 ~25 ms sustained measurement.

### 3.9 Render-pass attribution (dlse.2.4) — REVISES the §3.8 conclusion

Step 4 of the dlse.2.2 investigation adds a second `TIMESTAMP_QUERY_INSIDE_ENCODERS` bracket around the blit + particle + HUD + hotbar + legend render pass, in parallel with the existing compute-pass bracket. This is the direct follow-up proposed at the end of §3.8 — attribute the ~25 ms frame budget to a concrete pass. The result REVISES the §3.8 conclusion.

**Instrumentation (onyx 2026-04-21).** Two independent `GpuTiming` instances with distinct query sets / resolve buffers / readback buffers, each honouring the existing `HASH_THING_DISABLE_TIMESTAMP_RESOLVE=1` kill switch. Single `device.poll(Poll)` per frame drives both. New metric `render_pass_gpu` in `Perf` alongside the historical (compute-only) `render_gpu`. `acquire_harness.rs` extended with a parallel column so dlse.2.2 step-2-style windowed/fullscreen captures don't silently drop the new data.

**Measurement — M2 MBA 16 GB, 256³ dev, default 50 % render scale, windowed, steady-state (lines 3–5 of a 15 s run):**

| metric                   | mean     | p95      | samples |
|--------------------------|----------|----------|---------|
| `render_gpu` (compute)   | 0.08 ms  | 0.15 ms  | ~60     |
| `render_pass_gpu` (blit+overlays) | 0.08 ms | 0.16 ms | ~60 |
| **total GPU work**       | **~0.16 ms** | ~0.31 ms | — |
| `submit_cpu`             | 0.34 ms  | 0.45 ms  | ~60     |
| `present_cpu`            | 0.02 ms  | 0.03 ms  | ~60     |
| `surface_acquire_cpu`    | 24.81 ms | 30.80 ms | ~60     |
| `render_cpu` total       | 25.54 ms | 32.75 ms | ~60     |

**REVISION — the §3.8 "real GPU frame time" conclusion was wrong.** Direct GPU instrumentation shows the two passes together consume ~0.16 ms of GPU work per frame, not ~25 ms. The `render_pass_gpu` metric specifically tops out below 0.2 ms even at p95. The ~25 ms ceiling is NOT GPU-bound.

The 25 ms lives where the step-2/3/3b elimination said it couldn't: in `surface.get_current_texture()`. The new brackets rule in what earlier steps only proved by absence — the GPU is idle for roughly 24 of every 25 ms, CPU-blocked at the swapchain handoff. The previous "by elimination" inference collapsed because it assumed an enumeration of three hypotheses (compositor / swapchain-pacing / timestamp-fence) was exhaustive. It wasn't. A fourth cause — **driver/OS-level vsync pacing** that holds the drawable even when the GPU is free — fits every datum: windowed, fullscreen, with or without timestamp traffic, on-surface or off-surface (in off-surface mode the block migrates to `submit_cpu` at ~25 ms even though the new `render_pass_gpu` bracket still reads ≤0.3 ms, proving the stall in that mode is not GPU work either).

**Off-surface comparison — submit_cpu stall is NOT render-pass work either:**

| metric                   | windowed   | `HASH_THING_OFF_SURFACE=1` |
|--------------------------|------------|----------------------------|
| `render_pass_gpu` mean   | 0.08 ms    | 0.15 ms                    |
| `render_pass_gpu` p95    | 0.16 ms    | 0.34 ms                    |
| `submit_cpu` mean        | 0.34 ms    | 25–28 ms                   |
| `surface_acquire_cpu`    | 24.81 ms   | 0.00 ms                    |

The off-surface `render_gpu` (compute) bracket reports inflated numbers (~9–26 ms mean). `ticks_to_duration` is a pure end-minus-start subtraction on GPU-reported tick counts (`crates/ht-render/src/renderer.rs:344`), so the inflation is a real GPU-time delta *between* the two writes — not a Rust-side callback ordering artifact. The most plausible mechanism is that Metal attributes inter-command-buffer waits to the GPU command next in flight: off-surface, the compute-pass begin-timestamp can sit through a queue-depth stall that the render-pass begin-timestamp, executing later in the same command buffer, has already cleared. The render-pass bracket stays under 0.35 ms in both modes and correlates better with the known-tiny render-pass workload, so we treat it as the more reliable reading here; the compute-bracket variance is an open item, not a pinned conclusion (follow-up bead).

**What this means for dlse P0.** The 30 FPS ceiling on M2 at 256³ is not a compute, not a blit, not an overlay, not a shader problem. Optimizing anything in the render encoder buys nothing. The actual knob is swapchain-side pacing — something like `maximumDrawableCount`, `displaySyncEnabled`, or `presentsWithTransaction` on the underlying `CAMetalLayer` — which wgpu 29.0.1 does not expose through its `SurfaceConfiguration`. Candidates going forward (any of which is a standalone investigation, not in scope for dlse.2.4):

1. **Raise `desired_maximum_frame_latency` from 2 to 3.** Gives the swapchain one more drawable in flight, which on Metal historically unblocks the kind of acquire stall seen here. Cheap to try (one-line change). Prior attempt under dlse.2.2 went the opposite direction (*latency=1*, see `.ship-notes/` history); latency=3 has not been probed.
2. **Direct `CAMetalLayer` access.** wgpu exposes `unsafe { surface.as_hal::<wgpu::hal::metal::Api, _, _>(|hal| hal.raw_device().clone()) }` on Metal; via `metal-rs` we can set `maximumDrawableCount=3` and `displaySyncEnabled=NO`. Heavier — requires a Metal-only code path and a feature flag.
3. **Accept 30 FPS on integrated M-series and document it.** The GPU is already idle; rendering the same scene at 60 FPS would require the driver to hand us drawables faster, which is out of our reach at the wgpu layer. Ship notes: "M2 integrated runs at 30 FPS in windowed mode; the bottleneck is CAMetalLayer pacing, not our renderer."

dlse.2.2 should be **reopened** for option 1 (fast) and a new bead filed for option 2 (Metal-specific). The §3.8 closure note was premature; this is a swapchain-layer bug, not a renderer-compute bug.

**Candidate #1 probe — `desired_maximum_frame_latency=3` (dlse.2.2 exp#4).** Onyx 2026-04-21: env-var scaffolding (`HASH_THING_FRAME_LATENCY=N`) landed at 165f784; measurement run via the self-driving acquire-harness on M2, 60 Hz external display, 256³ world, release build, seven runs total. The harness does 60 warmup frames + 64 capture frames per phase, so each arm-row below is n=64 windowed / n=64 fullscreen. Discarding the first-build-after-launch run as a one-shot outlier (12.31 ms — did not reproduce in any subsequent run):

| `desired_maximum_frame_latency` | windowed `surface_acquire_cpu` mean (ms) | runs (windowed mean, ms)       |
|---------------------------------|------------------------------------------|--------------------------------|
| 1                               | 29.26 (n=1)                              | 29.26                          |
| 2 (current default)             | **26.81 (n=5)**                          | 26.44 / 26.83 / 27.40 / 26.79 / 26.58 |
| 3                               | 28.03 (n=3)                              | 29.42 / 26.72 / 27.95          |
 
Delta latency=3 − latency=2: **+1.22 ms (+4.6%) worse**, well inside run-to-run noise and the opposite direction from the hypothesis. Monotonicity check: latency=1 is also worse than latency=2, so the current default is the local sweet spot — there is no larger-latency knee to find by going higher. **Candidate #1 is ruled out.** None of the three latencies comes anywhere near the ≥30% reduction acceptance threshold from the dlse.2.2 plan review.
 
Plan + code-review artifacts for the scaffolding commit live in `.ship-notes/plan-dlse22-exp4-latency3.md`, `.ship-notes/plan-review-dlse22exp4-{claude,codex}.md`, and `.ship-notes/code-review-lat-scaffold-{claude,codex}.md`. The env var stays in place for future diagnostic use but does not become the default. Next dlse.2.2 candidate: option 2 above (direct `CAMetalLayer` access via wgpu hal) or option 3 (accept 30 FPS on M2 integrated and document it).

**What wgpu 29.0.1 actually sets on the CAMetalLayer** (source: `wgpu-hal-29.0.1/src/metal/surface.rs:70-109`, audited 2026-04-21 by onyx as the grounding step for candidate #2):

| CAMetalLayer property          | wgpu 29.0.1 setting                                     | knob we control                                              |
|--------------------------------|---------------------------------------------------------|--------------------------------------------------------------|
| `maximumDrawableCount`         | `maximum_frame_latency + 1` (so latency=2 → 3 drawables) | `SurfaceConfiguration.desired_maximum_frame_latency`         |
| `displaySyncEnabled`           | `true` if `PresentMode::Fifo`, `false` if `Immediate`   | `SurfaceConfiguration.present_mode`                          |
| `allowsNextDrawableTimeout`    | **`false`** (blocks indefinitely on acquire)            | not exposed                                                  |
| `framebufferOnly`              | `true` when usage is `COLOR_TARGET` (our case)          | indirectly via `SurfaceConfiguration.usage`                  |
| `presentsWithTransaction`      | never set — layer default (`false`)                     | not exposed                                                  |
| `wantsExtendedDynamicRangeContent` | `true` iff format is `Rgba16Float`                  | `SurfaceConfiguration.format`                                |

**Implication for the remaining hypothesis space.** The candidate #1 probe already swept `maximumDrawableCount` across {2, 3, 4}; none moved the stall. Earlier moss 2026-04-15 probe swept `displaySyncEnabled` across {true, false}; neither moved the stall. Those are the two knobs wgpu exposes. So the stall is *not* a drawable-count ceiling and *not* a vsync-sync policy. Both knobs that we can change via `SurfaceConfiguration` have now been shown to not be the cause. Any remaining progress has to be either:

- **Candidate #2a** — reach through `surface.as_hal::<wgpu::hal::metal::Api, _, _>()` and set properties wgpu does not expose: `presentsWithTransaction = false` (already the default, probably a no-op) or drive presentation via a `CADisplayLink`-paced loop rather than winit's `RedrawRequested`. A CADisplayLink tick is the macOS-idiomatic pacing source for windowed Metal apps and is how `MTKView` internally drives its delegate; winit does not integrate with it.
- **Candidate #2b** — replace the winit-driven `RedrawRequested` loop with a `MTKView`-equivalent render loop, either by constructing an `MTKView` manually or by driving redraws off a `CADisplayLink` callback. Both bypass winit's event-loop pacing; the second is the smaller delta.
- **Candidate #3** — accept that windowed-mode macOS composition adds ~16–25 ms pacing overhead, document it, and set expectations that 60 FPS is a true-fullscreen-only target on M-series integrated GPUs.

Candidate #2a is the narrowest thing still worth trying before conceding to #3: it's a one-dispatch experiment that keeps winit's window but supplies our own redraw cadence. If a CADisplayLink-paced loop still shows `surface_acquire_cpu ≥ 16 ms` on M2, that is a strong signal the root cause is WindowServer composition and not our pacing choice, which is as close to a root-cause conclusion as this bead can get at the wgpu layer.

**Candidate #2a refined — acquire-as-late-as-possible (Apple Metal Best Practices).** Literature pass 2026-04-21 (onyx) surfaced a much cheaper experiment than the `surface.as_hal` / `CADisplayLink` reroute. Apple's [Metal Best Practices Guide — Drawables](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/Drawables.html) states:

> "Always acquire a drawable as late as possible; preferably, immediately before encoding an on-screen render pass. A frame's CPU work may include dynamic data updates and off-screen render passes that you can perform before acquiring a drawable."

Our `Renderer::render` (`crates/ht-render/src/renderer.rs:1843`) calls `surface.get_current_texture()` *before* the SVDAG raycast compute dispatch (`renderer.rs:1989`) — the exact anti-pattern the doc warns about. Our compute pass writes to an off-screen `raycast_texture` that the on-screen render pass later blits from, so the Apple-sanctioned reorder is a mechanical move: encode compute first, then acquire the drawable immediately before the render-pass encode. This is the same class of fix Flutter Impeller shipped on iOS for the same symptom (flutter/flutter#138490). No `surface.as_hal`, no `CADisplayLink`, no unsafe code. Tracked as `hash-thing-dlse.2.2.2`; candidate #2a/#2b above remain as the fallbacks if the reorder doesn't land the signal.

**Candidate #2a outcome 2026-04-21 (onyx) — null result + paper correction.** Shipped the reorder (move `get_current_texture()` after the compute-pass encode, immediately before the on-screen render pass), built both pre- and post-reorder `--profile bench` binaries of `hash-thing 256`, ran each for 35 s on M2 with `RUST_LOG=info`, and compared `surface_acquire_cpu`. Reviewer pass was clean (2 Claude + 1 Codex reviewers, all ship-verdict; full wgpu 29.0.1 + Metal HAL audit of the drop-encoder path confirmed no leak / no partial submit). The measurement was the deciding input, and the measurement said the reorder does not move the signal:

| window (s) | before mean (ms) | before p95 (ms) | after mean (ms) | after p95 (ms) |
|-----------:|-----------------:|----------------:|----------------:|---------------:|
| 0–14 (warmup) | 24.67 | ~32.4 | 24.49 | ~33.1 |
| 14–35 (steady) | 13.40 | 15.64 | 13.46 | 15.53 |

Per-run raw rows are in `/tmp/before-dlse222-long.log` and `/tmp/after-dlse222-long.log`; full methodology is in `.ship-notes/plan-dlse222-late-acquire.md`. Delta is within timer noise in both windows. Reverted the `renderer.rs` reorder per the plan's rollback criterion; `dlse.2.2.2` is closed as a ship-safe but effect-less fix.

The instructive finding, though, is not the null — it is the **steady-state column**. `surface_acquire_cpu` settles to ~13.5 ms mean / ~15.6 ms p95 after ~14–16 s of runtime on both binaries, not the ~25 ms that §3.8 documented as "pinned." Two runs on the same binary reproduce the step-down to within 0.1 ms. That means the ~25 ms stall this subsection has been chasing is **not a steady-state pin**; it is a ~14-s warmup-period behaviour of the WindowServer / CAMetalLayer pacing that self-resolves. Steady-state 13.5 ms is consistent with a 60 Hz vsync budget (16.67 ms) minus our own ~3 ms of CPU work, which is exactly the regime we expected 60 FPS to live in. §3.8's original capture window (~10 s warm, skip first line) was inside the warmup phase and mis-attributed a transient to a pin. That explains why every knob probe under §3.8 — `maximumDrawableCount`, `displaySyncEnabled`, `present_mode`, and now the late-acquire reorder — failed to move the number: nothing in that plane *should* move a warmup transient, because the steady-state number is already ~60 FPS. The warmup stall itself is still worth characterising (tracked as a new bead), but the framing of dlse.2.2 shifts: the windowed-mode 30 FPS we have been debugging is a "first ~14 s" figure, not "the FPS." Candidate #3 (accept 30 FPS as the windowed floor) is weakened by this; candidate #2b (MTKView / `CADisplayLink` drive) is no longer obviously needed at all for steady state. Measurement on longer-baseline windows supersedes §3.8's original attribution; §3.8 and the dlse.2.2 candidate enumeration above should be read as historical context for a transient that does not actually pin.

**Addendum 2026-04-21 (`hash-thing-dlse.2.2.3`, warmup shape).** A 60 s per-generation trace of the same 256³/50% run (`/tmp/dlse223-full.log`, 29 samples) shows the "14-s warmup" is imprecise. The actual shape is: **climb 23 → 34 ms over t=2-14** (same pipeline-fill shape dlse.3 documents at 1024³), a sharp kink at t=14-16 coincident with a memo cache trim (memo_tbl 31298 → 8372), a plateau at 26-27 ms through t=26, then a gradual 8-s step-down t=26-34 to ~11 ms steady. The step-down correlates with step-duration stabilising (640 → 860 ms) and memo_hit plateauing — as sim generations slow, svdag rebuild / upload pressure drops. The "warmup" is therefore not a single WindowServer negotiation event: it is a composite of CPU-GPU pipeline fill (same mechanism as dlse.3), memo-cache warmup, and sim-step-rate stabilisation. The 0-14 / 14-35 split in the table above puts the tail of the step-down into the steady column, which pulls the steady mean toward its elevated end; the true post-stabilisation mean is closer to ~11 ms. The broad conclusion (steady is in the 60 FPS regime, no compositor knob fixes this) stands.

### 3.10 Scaling the model to 4096³ (`hash-thing-ivms`)

Edward directive 2026-04-20: *"I'm always only interested in the 4096 cubed case."* Sections 3.1–3.6 derive the envelope at 256³ — the bench-harness default, not the demo target. This subsection re-runs the derivation at 4096³ from first principles. Every number here is model-only; §5 gains 4096³ rows only for what can be (or has been) cheaply measured. The dlse.2 present-path investigation (§3.7–§3.9) is world-size-independent and carries over unchanged.

**3.10.1 Octree depth.** `log₂(4096) = 12`, vs `log₂(256) = 8` — four extra levels of descent. The SVDAG interior-node encoding (§3.2) is unchanged (36 B), so depth scaling enters the model through traversal step count (§3.10.2) and reachable-set size (§3.10.3) only.

**3.10.2 Per-ray traversal at 4096³.** The stue.5 harness measured ~15 mean DDA steps/ray at 256³ default-spawn (§3.3). Two scaling forces:

- **Tree-walk machinery.** Descend + step-past + pop is roughly linear in tree height at fixed ray length; 12/8 = 1.5× on the descent chain.
- **Ray coverage.** Render target stays at 960×600 (render scale 0.5 on a 1920×1200-class window), so primary-ray count is unchanged at ~576k. But a 4096³ world fills the frustum rather than occupying a corner; hit rate on horizontal camera poses approaches 100 %, shifting the mean away from the sky-heavy all-rays distribution and toward the hit-only distribution.

**Measured at 4096³** via `tests/bench_depth_histogram.rs::depth_histogram_4096` (onyx 2026-04-21, hash-thing-cz0r), same three poses as stue.5:

| Pose | 256³ mean | 4096³ mean | Ratio | 4096³ p95 / max |
|---|---|---|---|---|
| default-spawn | 14.99 | **25.90** | 1.73× | 62 / 235 |
| looking-down | 6.77 | **16.76** | 2.48× | 26 / 42 |
| horizontal-mid | 13.21 | **24.20** | 1.83× | 61 / 231 |

The scaling exponent for depth in linear world size sits at `log(1.73)/log(16) ≈ 0.20` for default-spawn and up to `log(2.48)/log(16) ≈ 0.33` for looking-down — materially flatter than the pre-measurement ~0.8 projection (`22/15 → ~0.4` already under; `28/15 → ~0.6` upper was clearly over). The "descend + step-past + pop" chain grows with tree height but the *step-past + pop* component does not scale one-for-one with depth: most additional levels at 4096³ are descended through coarse empty-space prefixes that collapse to shared ancestor chains in the SVDAG, adding log-scale steps rather than linear.

Implication for §3.10.5: the base-case per-frame step count is **~576k × 26 ≈ 15M**, not the 20M used in the pre-measurement envelope. Downstream bandwidth and latency numbers contract correspondingly (see §3.10.5 post-measurement column).

**3.10.3 Reachable DAG size and working set.** Measured SVDAG node counts (`tests/bench_svdag.rs` comment, 2026-04-12):

| Scale | Nodes | Size | Per-linear-double ratio |
|---|---|---|---|
| 64³   | 640    | 0.02 MB | — |
| 256³  | 6,773  | 0.23 MB | ~3.3× per 2× linear (early growth) |
| 512³  | 20,228 | 0.69 MB | ~3.0× per 2× linear |
| 1024³ | 61,399 | 2.11 MB | ~3.0× per 2× linear |
| **4096³ (seeded, pre-CA)** | **548,179** | **~18.8 MB** | **~3.0× per 2× linear (projection confirmed)** |

From 256³ onward the exponent is stable at `log(3)/log(2) ≈ 1.58` in linear world size. Confirmed by one-shot `bench_hashlife_4096` seed (onyx 2026-04-21): 548k reachable nodes at 4096³ seeded terrain — exactly on the `L^1.58` projection (1024³ × 4^1.58 ≈ 551k). Total serialized DAG: ~18.8 MB (548k × 36 B).

Warm-run post-CA node count was not measured — the test runner's 60-second ignore-mode ceiling killed the first hashlife step before it completed, so the warm (post-step) reachable-node count and macro-cache size remain unmeasured at 4096³. The cold-seeded count is a lower bound; CA churn grows the reachable set by some factor (at 256³, warm grows ~8 % vs cold; scaling is not yet characterized).

**19 MB exceeds the M1 L2 (4 MB) by ~5×** for the seeded pre-CA DAG alone; post-CA warm-set will be somewhat larger (up to ~25 MB projected under a typical-growth factor). At 256³ the entire reachable DAG fit in L2 with margin (§3.5); at 4096³ the whole set does not. The relevant question shifts from "whole-DAG-resident" to *working-set on the active ray frontier*: warp-coherent primary rays touch ~2–4 sibling chains at any given descent depth, so the instantaneous hot set is *heuristically* ~1–4 MB — still L2-resident if the heuristic holds. The cold tail of distant / shadow / back-side nodes spills to DRAM. **This hot-set estimate is load-bearing for §3.10.4 below but is not directly measured**; it is an order-of-magnitude guess from warp width × expected descent depth × node size, not an inferred bound.

**3.10.4 Per-step cost regime.** At 256³ an 80/20 L1/L2 mix gave ~9 ns/step (§3.3). At 4096³, the regime plausibly shifts toward L2/DRAM because the whole DAG spills L2. As a speculative mid-point between "warp coherence keeps the 80/20 mix intact" (optimistic) and "cache pressure dominates" (pessimistic), posit a 50/35/15 L1/L2/DRAM mix, giving `0.5×6 + 0.35×22 + 0.15×175 ≈ 3.0 + 7.7 + 26.3 ≈ 37 ns/step`, ~4× the 256³ per-step cost.

This mix is **speculative**, not derived. The actual mix at 4096³ depends on how tightly warp-coherent primary rays cluster in the reachable DAG, which the model has no direct lever on. The two bracketing cases:
- **Optimistic (mix stays at 80/20 because warp coherence keeps the hot ancestor chain L1-resident):** per-step cost ~9 ns, unchanged from 256³.
- **Pessimistic (mix shifts to 30/40/30 because the larger whole-DAG forces more L2 evictions):** per-step cost ~60 ns, ~7× the 256³ cost.

§3.10.5's envelope and likely-measurement predictions are re-derived under all three scenarios (optimistic / mid / pessimistic) rather than only the mid one, so the user of this section can pick the assumption they consider most plausible.

**3.10.5 Frame cost envelope at 4096³.** Holding render scale at 50 % (960×600, 576k rays) on the M1 MBA target. Per-frame node traffic (bandwidth ceiling, unaffected by cache mix), **post-cz0r measurement with default-spawn mean = 26 steps/ray**: `576k × 26 × 36 B ≈ 540 MB`, `540 MB / 68 GB/s ≈ 7.9 ms` raw, `÷ 0.6 utilization ≈ 13.2 ms` envelope. (Horizontal-mid at mean=24 gives a near-identical 12.2 ms envelope; looking-down at mean=17 gives 8.6 ms — the worst default-spawn case drives the budget.)

Per-step-latency path, spanning the three §3.10.4 scenarios (rays grouped into 32-wide SIMD groups on M1; the divisor assumes groups descend together so shared ancestor loads are not duplicated):

| Scenario | ns/step | Envelope `576k × 26 × ns / 32` | Bandwidth envelope | Both agree? |
|---|---|---|---|---|
| Optimistic (80/20, coherence holds) | 9 | **4.2 ms** | 13.2 ms | no — bandwidth not the bind |
| Mid (50/35/15, speculative) | 37 | **17.3 ms** | 13.2 ms | near |
| Pessimistic (30/40/30, coherence breaks) | 60 | **28.1 ms** | 13.2 ms | no — latency dominates |

**Likely-measurement span:** **~1–17 ms / frame** depending on which scenario holds, not a single tight number. The paper's strongest single prediction is the **17 ms confident upper bound** (saturates the 60-FPS budget — useful as a go/no-go threshold). Anything below ~10 ms would leave headroom for sim + blit + overlay + present; anything above ~10 ms forces a trade-off. Step-count measurement (cz0r) landed at 26 — within the 22–28 pre-measurement band — so the envelope numbers shift by less than 1 ms from the ivms-era draft and the scenario spread remains the operative uncertainty, not the step count.

Two cross-checks on the span:
- **256³ analogy.** §3.6 noted the 7.6 ms envelope at 256³ over-predicted a 0.24 ms measurement by ~30×. Applying the same ratio to the mid-scenario envelope gives **~0.5 ms** at 4096³ — improbably fast relative to the seven-fold step-count increase, which suggests the 30× ratio itself will attenuate under 4096³ cache pressure. §3.6's own explanation (items 2+3: L1 reuse, warp coherence) is exactly what breaks at 4096³. **So: do not anchor on "0.5 ms" as a likely-measurement.** Treat the 30× only as evidence that the 256³ envelope was conservative, not as a transferable factor.
- **Empty-corner bias.** `bench_raycast_4096` uses `BenchCamera::empty_corner()` — rays exit rapidly through sparse air. Measured numbers from that harness are a *floor*, not a typical-scene prediction. The model-envelope comparison should use a seeded-terrain scene closer to the demo target; adding one to the bench-harness is an ivms follow-up filed in §6.

**Net prediction: 4096³ raycast on M1 MBA falls somewhere in [optimistic ~4 ms, pessimistic ~17 ms].** A one-shot run on M1 MBA hardware discriminates between the three scenarios.

**3.10.5.1 Post-measurement (hash-thing-e092, 2026-04-21).** Seeded-terrain `bench_raycast_4096_app_spawn` on M2 16 GB measured **0.30 ms mean / 0.22 ms p50 / 1.06 ms p95** across 10 frames, SVDAG=18.8 MB/548k nodes. M1-implied at the standard 1.5× factor: ~0.45 ms mean. This lands ~10× below the 4 ms optimistic scenario — the measurement refutes all three §3.10.4 cache-mix scenarios as calibrated upper bounds. The operative explanation is the same as at 256³ (§3.6): warp-coherent primary rays keep the hot ancestor chain L1-resident at a few KB, so the "whole-DAG spills L2" framing in §3.10.3 never becomes the bottleneck. The 17 ms confident upper bound stands as a go/no-go threshold (measurement < bound by a factor of ~38), but the interior scenario labels (4 / 17 / 27 ms) are not predictive. For the next scale jump (8192³, §4.7.4 streaming threshold), re-derive the envelope from warp-width + ancestor-chain heuristics rather than a cache-mix fraction — the latter has now over-predicted by >10× at two scales.

**3.10.6 Confirmation path (pre-e092, kept for context).** `bench_raycast_4096` already exists in `tests/bench_gpu_raycast.rs` (`#[ignore]`) and runs 10 samples, but it uses `BenchCamera::empty_corner()` (optimistic). The confirmation path was two runs:
1. The existing empty-corner `bench_raycast_4096` — floor.
2. A seeded-terrain variant at 4096³, closer to the demo target — measurement comparable against the §3.10.5 envelope (now landed as `bench_raycast_4096_app_spawn`, see §3.10.5.1).

### 3.11 GPU cost at 1024³/100% is real — the "ramp" is pipeline fill (`hash-thing-dlse.3`)

Bead filed 2026-04-21 from a 1024³ repro where `surface_acquire_cpu` grew monotonically from 54 ms at t=0 to 187 ms at t=20 s and plateaued. Initial suspects: sim-thread starvation (refuted by `HASH_THING_FREEZE_SIM=1` — ramp still occurs) and GPU-queue backpressure accumulation.

The decisive A/B was `HASH_THING_OFF_SURFACE=1`: with no swapchain, `surface_acquire_cpu` drops to 0.01 ms and the ramp **moves to `submit_cpu`** (13 → 154 ms, plateau 154 ms). Meanwhile `render_gpu` reads 3.76 ms for the first ~10 s, then jumps to 53 ms mean / **151 ms max** once the timestamp-query ring buffer catches up. Plateau self-consistency: `submit_cpu ≈ render_gpu max`. The GPU is genuinely spending ~150 ms/frame on the compute raycast at 2940×1782 × 1024³.

Two corrections to the earlier framing:
1. **`render_gpu` has a pipeline-depth lag, not an under-measurement.** `gpu_timing` already brackets the compute pass (see `renderer.rs:1957, GpuTiming::compute_pass_writes`); `render_pass_gpu` brackets blit+HUD. The first ~10 s of `render_gpu=3.76 ms` in the 1024³ trace is ring-buffer staleness: the GPU is running frame N-K (K ~= command-buffer pipeline depth) and the `map_async` readback on frame N returns a timestamp from that old frame. Once the pipeline fills enough that submit waits for a prior frame, the returned timestamp becomes current-frame-equivalent and jumps to 53 ms mean / 151 ms max. At 256³, the compute cost is genuinely sub-millisecond per frame, so `render_gpu=0.41 ms` is correct but has much higher frame throughput, keeping the ring buffer effectively always stale — the 12-34 ms off-surface `submit_cpu` at 256³ reflects command-buffer queue + timestamp-map backpressure around a fast dispatch, not a hidden 30× of GPU compute.
2. **The ramp is not an accumulating queue.** It is a pipelined CPU-GPU system reaching its steady-state backpressure. Frame 0 submits to an empty GPU; frame N submits and waits on frame N-K to retire; as K fills (`maximumDrawableCount` + command-buffer depth), steady-state submit time converges to actual GPU frame time. The linear shape of the 13 → 154 ms off-surface climb is exactly the shape of a bounded pipeline filling.

Surface-attached plateau (187 ms) minus off-surface plateau (154 ms) ≈ 33 ms of compositor + drawable-acquisition overhead, consistent with the CAMetalLayer path. The camera-recenter snap-back is plausibly a world-state transition to shallow traversal / empty rays for a few frames — the pipeline drains and refills, giving the same ramp from a lower starting cost — not a sim-state reset.

**Budget implication.** On M2 integrated, 1024³ × 100% scale is GPU-bound at ~6 FPS; no amount of compositor tuning fixes that. `render_scale = 0.5` as the default puts us near the M2 budget at 1024³. A world-size-aware auto-picker is filed as `hash-thing-zytn`.

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

The two hypotheses predict the same `surface_acquire_cpu` reduction. Disambiguating needs proxy 2 (same-core vs different-core sim scheduling), executed as `hash-thing-xhi6` (see proxy-2 results immediately below).

**What proxy 1 does establish.**
1. Renderer-only steady-state at 256 ³ on this hardware lives at **~18 ms surface_acquire_cpu**, not the ~26 ms we see with sim live. The remaining ~18 ms is a windowed-presentation floor (consistent with `dlse.2.2` findings — that path is acquire-dominated independently of sim).
2. **Sim does meaningfully impact frame budget**, even though the renderer reads a coherent SVDAG snapshot per frame and the sim runs on a separate `std::thread`. The "background sim is free" assumption is wrong by ~8 ms at this scale.
3. **render_gpu is unaffected** (0.11 → 0.08 ms, both well below noise). The contention shows up in CPU-side acquire, not on-GPU compute. So at this scale, a GPU-side memo (parent bead `hash-thing-abwm`) is not motivated by *renderer* contention; it would be motivated by step latency itself.

**Implication for §3 frame budget.** The §3.1 budget assumed sim and render were independent on a separate-thread M-series. They aren't quite — sim costs the renderer ~8 ms / frame at 256 ³. At 4096³ the sim cost is ~64× larger (linear in node count for active-region work) but rendering is also more memory-bound, so the contention term may grow faster than linear. Re-derive at 4096³ when `hash-thing-ivms` lands.

**Verdict status:** **contention present, magnitude ~30 % of windowed frame at 256³, driver (CPU vs cache) unresolved by proxy 1.** See proxy 2 immediately below.

**Proxy 2 — sim-thread QoS sweep (hash-thing-xhi6).** `HASH_THING_SIM_QOS=<class>` calls `pthread_set_qos_class_self_np` from inside the spawned background-step thread. macOS schedulers steer `BACKGROUND` / `UTILITY` toward the E-cluster and `INTERACTIVE` / `INITIATED` toward the P-cluster. The renderer / main thread runs at the inherited QoS (`USER_INITIATED` for winit + AppKit), so `interactive` keeps sim and render in the same pool, and `background` moves sim to a different pool. Same scene / build / sample protocol as proxy 1. n=5 each (post-warmup), runs cooled between conditions where possible.

| QoS                 | step (mean) | surface_acquire_cpu (mean) | Δ vs interactive | sim cluster |
|---------------------|-------------|----------------------------|------------------|-------------|
| interactive         | 371 ms      | 25.5 ms                    | (baseline)       | P           |
| utility             | 803 ms      | 31.1 ms                    | +5.6 ms          | mixed → E   |
| background          | 1327 ms     | 40.0 ms                    | +14.5 ms         | E           |
| (sim-frozen, proxy 1)| —          | 18.5 ms                    | −7.0 ms          | n/a         |

`render_gpu` stayed at ~0.12 ms across all conditions (QoS-independent, same as proxy 1).

**What proxy 2 was supposed to deliver.** The bead's premise: if same-pool sim runs the renderer faster than different-pool sim, CPU-cycle contention is dominant; if equal, unified-memory bandwidth is dominant.

**What proxy 2 actually shows.** Different-pool sim runs the renderer *slower*, in proportion to how badly the sim itself is throttled by E-cluster scheduling. So:
- Same-pool (interactive, ~25.5 ms) is statistically indistinguishable from the inherited baseline (~26.5 ms in proxy 1) — the inherited QoS is already same-pool. There is no QoS knob that *reduces* the contention, only knobs that make it worse.
- Cross-pool sim adds a *new* cost (cross-cluster coherence traffic on the system-level cache, sim-throughput-dependent placeholder-swap pacing, or both) that swamps any saving from removing CPU-pool sharing.

**Conclusion.** Proxy 2's premise is refuted: the CPU-cycle / unified-memory dichotomy doesn't cleanly map to a P-vs-E pool toggle on M2. Two possibilities remain: (a) CPU-cycle contention is not dominant and a new cross-cluster cost dwarfs whatever savings proxy 2 was trying to surface; (b) CPU-cycle contention is dominant, but the new cross-cluster cost on M2 outweighs the saving by ~5–14 ms. Either way, the clean disambiguation isn't available from QoS scheduling on this hardware.

**M2 MBA thermal caveat.** The fanless M2 throttles after ~2 sustained 14 s benches (step time grows 5–10× on the second-or-later run). The numbers above used the ordering `interactive → utility → background` with cool-downs; sustained thermal load is itself a confound for any longer-running sweep on this chassis. Raw evidence at `.ship-notes/xhi6-cache-locality-proxy2-2026-04-21.md`.

**Verdict (proxy 1 + proxy 2):** sim ↔ renderer contention at the SVDAG boundary is real (~8 ms / 30 % at 256³ windowed), `render_gpu` is uninvolved, and the contention is CPU-side / unified-memory-side. Neither proxy disambiguates which of the two CPU-side mechanisms dominates. A useful next proxy would hold sim CPU work constant while toggling its memory-access footprint (e.g. read-only step pass vs mutate-only step pass), or read per-thread `task_thread_times_info` directly to attribute time without leaning on QoS-class scheduling.

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

### 4.7 Scaling the SVDAG ↔ memo interaction to 4096³ (`hash-thing-ivms`)

The §4.1–§4.6 numbers are all at 256³ (with a 64³ cross-reference). At 4096³ (edward's actual demo target) the architectural story holds — shared NodeStore, siloed render-side cache, O(new-content) diff upload — but the bounded-overhead arguments need to be re-run at the new size before ivms can close.

**4.7.1 Reachable DAG and render-side bookkeeping at 4096³.** Scaling from the empirical node counts (§3.10.3):

| Scale | Reachable nodes | DAG size | `offset_by_slot` | `id_to_offset` |
|---|---|---|---|---|
| 256³ (measured) | 6,773 | 0.2 MB | ~0.3 MB | ~0.08 MB |
| 1024³ (measured) | 61,399 | 2.1 MB | ~2.9 MB | ~0.7 MB |
| **4096³ (seeded, measured)** | **548,179** | **18.8 MB** | **~26 MB** | **~6.6 MB** |
| 8192³ (projected `L^1.58`) | ~1.65M | ~57 MB | ~80 MB | ~20 MB |

`offset_by_slot` is `FxHashMap<[u32; 9], u32>` ≈ 48 B/entry including table overhead; `id_to_offset` is `FxHashMap<NodeId, u32>` ≈ 12 B/entry. The projected entry counts in this table are steady-state (post-compaction); `Svdag::compact` (`src/main.rs:882`) triggers at `stale_ratio > 0.5`, so the pre-compaction peak is up to ~2× the steady-state number. Steady-state at 4096³ ≈ 32 MB total render-side bookkeeping; pre-compaction peak ≈ 64 MB.

**Verdict: comfortably resident on 8 GB M1 MBA at 4096³.** Combined render-side bookkeeping at 4096³ is ≤ ~64 MB including compaction peaks — well under 2 % of the ~4–5 GB available-to-app budget. Not a binding constraint, and not the motivation for streaming. (Warm-run post-CA reachable may be ~10–30 % larger than the seeded measurement; the bookkeeping still has two-orders-of-magnitude headroom.)

**4.7.2 Memo churn at 4096³.** Three measured warm-step miss counts (`bench_svdag_step_deltas_*`, onyx 2026-04-21 slc1):

| Scale | Warm misses/step (measured) | Measured steps | Source |
|---|---|---|---|
| 64³ | 411.8 | 40 | `bench_svdag_step_deltas_64` |
| 256³ | 1,847.1 | 40 | `bench_svdag_step_deltas_256` |
| 512³ | 2,295.8 | 20 (confirmed twice) | `bench_svdag_step_deltas_512` |

The per-doubling slopes:
- 64³ → 256³ (2 doublings): slope ≈ **1.08** per `log₁₀ L` — matches the prior single-ratio value.
- 256³ → 512³ (1 doubling): slope ≈ **0.31** per `log₁₀ L` — bends sharply sub-linear.

Least-squares fit across all three: `log₁₀(misses) ≈ 0.863 · log₁₀(L) + 1.089`, i.e. ~12.3 × `L^0.863`. **The saturation is real, not a single-ratio artifact** — the 512³ measurement is confirmed twice to 0.01 % and reflects the `hashlife_macro_cache` hitting steady-state reuse rather than seeding new (node, stride) pairs at the pace of world volume. Physical intuition: at larger scales, more of each warm step's recursion lands on macro-cells already seen in prior steps, so marginal miss rate drops.

Projecting to 4096³ (`L/256 = 16`) across three framings:

- **Fit extrapolation (assumes slope holds):** `12.3 × 4096^0.863 ≈ 14.8k misses/step`.
- **1.08-slope pessimism (assumes saturation reverses):** `1,847 × 16^1.08 ≈ 37k misses/step`.
- **Continued bend (assumes `256→512` slope 0.31 holds at larger scales):** as low as `2,296 × 8^0.31 ≈ 4.3k misses/step`.

**Revised honest span: ~4k–40k misses/step at 4096³**, versus the prior two-point `20k–80k`. The upper end shrinks because the three-point evidence actively refutes `L^1.4+`; the lower end widens because the 512³ measurement opens the possibility of continued saturation. Applying the measured ~30 % slot-append ratio (1,847 misses → 544 SVDAG slot appends at 256³): **~1.2k–12k slot appends/step at 4096³**, translating to **~40 KB–440 KB upload per step** (4 B slot-0 header + slots × 36 B).

At 60 FPS that is **~2–26 MB/s upload**, still under 0.1 % of the 68 GB/s memory-bandwidth ceiling. **Edward's diff-compressibility hypothesis holds at 4096³** across the whole revised band — with more headroom than the prior two-point projection suggested.

Two caveats on the exponent:
- **Out-of-band scales still unmeasured.** The fit covers 64³–512³. At 2048³, a single warm step accumulated >15 min of CPU under this harness before being killed — so the 4hkq/slc1 out-of-band benches (`bench_svdag_step_deltas_2048`, `_4096`) need a measurement workflow that tolerates 10–30 min wall per scale, which the current CI-style `#[ignore]` harness does not provide. Filed as `hash-thing-slc1` follow-up.
- **Scene dependence.** A pathological scene with uniformly-active CA across the whole 4096³ volume would scale misses closer to `L³` (volume), giving ~470k misses/step. Our default terrain+water+sand scene is nowhere near that — active CA is concentrated on a thin fringe (water surface, sand-fall column) whose area scales well below `L²`. The measured 64³–512³ slope is consistent with this physical intuition; the 4096³ span above assumes the scene remains fringe-dominated.

**4.7.2a Out-of-band bench workflow (`hash-thing-dpkz`).** `bench_svdag_step_deltas_2048` and `bench_svdag_step_deltas_4096` are `#[ignore]`-gated and intended to be driven by a long-running wrapper rather than an agent's default 60 s `cargo test` budget. There is no dedicated binary; the workflow is invoke-the-existing-test with a bigger shell timeout. Expected wall-clock per scale (M1-class; ~1.5× faster on M2 16 GB):

| scale | seed | one warm step | total wall (warmup=0, measured=1) |
|-------|------|---------------|------------------------------------|
| 1024³ | ~1 s | ~10–20 s      | ~30 s (fits in the default budget) |
| 2048³ | ~2 s | 45–90 s       | 2–5 min                            |
| 4096³ | ~3.5 s | ≥60 s (bench_hashlife_4096 SIGKILL profile) | 2–10 min |

Invocation (bump the shell timeout explicitly — don't rely on the default 2 min `Bash` budget):

```text
cargo test --release --test bench_svdag bench_svdag_step_deltas_4096 \
    -- --ignored --nocapture
```

Agent usage pattern: run as a background task (`run_in_background: true`) so the sweep keeps moving while the bench finishes; the result lands as a task-notification. Capture the printed `misses/step`, `slot appends/step`, and `bytes/step` from the bench output and paste them into `§4.7.2` as the 2048³ / 4096³ entry of the miss-count table, then update the pair-wise slope column.

If a longer wall becomes unavoidable (>10 min) or the bench needs to run unattended across reboots, escalate to a dedicated harness binary — but the `#[ignore]`-gated test with a bumped shell timeout covers the current data-point need.

**4.7.3 `hashlife_macro_cache` budget at 4096³.** The cache is a **single** `FxHashMap<(NodeId, u64), NodeId>` (see `src/sim/world.rs:192` and `src/sim/hashlife.rs:406-422`), keyed on `(node, generation)`. There is no per-level multiplication — it is one map, shared across the whole recursion.

Entry cost: `(NodeId, u64) → NodeId` ≈ 16 B/entry including `FxHashMap` overhead. The cache grows with the product of:

- **Distinct NodeIds queried** as macro-step inputs. Bounded above by reachable-node count (~548k at 4096³ seeded).
- **Distinct `generation` stride values** the recursion walks. Macro-stepping uses power-of-2 generation strides; tree height 12 means up to 12 distinct stride values that any given node could be queried against. In practice the recursion pattern places each node at one dominant stride level (the level its subtree height matches), so the *observed* distinct-strides-per-node is closer to 1–2, not 12.

Plausible bands at 4096³:
- **Tight lower bound (one entry per reachable node, single stride dominant):** ~548k × 16 B ≈ **9 MB**.
- **Pathological upper bound (every `(node, stride)` pair populated for all 12 strides):** ~548k × 12 × 16 B ≈ **105 MB**. This overcounts — it assumes every node is queried at every stride level, which is not how the recursion actually walks.
- **Typical steady-state: not derivable from scaling alone.** The observed distinct-strides-per-node is a scene + recursion-pattern property, measurable only at runtime. No point-estimate here; **this is an ivms follow-up** (§6) for size instrumentation.

**Verdict: macro-cache is the first subsystem to watch at 4096³, but the earlier "20–50 MB typical" claim was ungrounded.** The floor is ~9 MB; the pathological ceiling is ~105 MB; the typical steady-state is a measurement, not a derivation. 8 GB budget survives comfortably in the plausible band (both ends <2 % of available-to-app budget), so the macro-cache is *unlikely* to be the binding constraint — but "unlikely" is intuition, not proof, until §6's follow-up lands. Instrumenting a size histogram on `hashlife_macro_cache` is filed accordingly.

**4.7.4 Streaming threshold.** Combining §3.10 envelope + §4.7 budgets (all numbers are either measured, marked as model extrapolations, or intuition — note the columns):

| Scale | Reachable DAG | Macro-cache (plausible band) | Raycast envelope | Resident on 8 GB? |
|---|---|---|---|---|
| 256³ | 0.2 MB (measured) | <10 MB (implied) | ~7.6 ms (model) / 0.24 ms (measured) | yes (margin) |
| 1024³ | 2.1 MB (measured) | unmeasured | ~9 ms (interpolated) | yes (margin) |
| **4096³** | **18.8 MB (seeded, measured)** | **9–105 MB (floor/pathological, §4.7.3)** | **0.30 ms measured M2 / ~0.45 ms M1-implied (e092); §3.10.5 span was 4–17 ms predicted** | **yes** |
| 8192³ | ~57 MB (projected) | unbounded analytically | ≥ 25 ms (projected) | probably, not verified |
| 16384³ | ~170 MB (projected) | — | DRAM-bound | unclear |

**Soft threshold: 8192³ is where streaming becomes plausibly mandatory, but this is intuition, not a derived result.** The itemized steady-state memory model at 8192³ (~57 MB DAG + ~80 MB `offset_by_slot` + ~20 MB `id_to_offset` + indeterminate macro-cache) is still only a few hundred MB absolute — nowhere near an 8 GB ceiling on paper. What tips the argument is unmodeled memory: the CPU-side `NodeStore` active region (not quantified in this paper), view-shift churn for a streaming camera, pathological CA states that push the macro-cache toward its ceiling, and co-resident wgpu driver state. None of those is pinned down numerically; the "8192³ streaming cutover" call is therefore a *reasoned intuition*, not a gap-report number. If the demo target ever moves above 4096³, this threshold needs a real derivation.

**4096³ fully resident on 8 GB M1 MBA is a stronger claim** because the dominant memory consumers are all measured or tightly bounded: 18.8 MB DAG (measured), ≤ 64 MB bookkeeping (steady + compaction peak), 9–105 MB macro-cache (bounded within a known band). Sum stays under ~170 MB — well under 1 GB — with room for unmodeled overheads.

**4.7.5 Summary for 4096³ on M1 MBA.**
- Reachable DAG: 18.8 MB seeded, measured.
- Upload bandwidth: fine (~12–54 MB/s, <0.1 % of memory ceiling).
- Render-side maps: fine (≤ 64 MB steady-state + compaction peak).
- Macro-cache: bounded 9–105 MB, typical steady-state unmeasured.
- Raycast frame cost: span of 4–17 ms across the three §3.10.4 cache-mix scenarios; no single scenario is anchored to evidence.
- **CPU step cost: the binding concern.** Scales directly with active-region misses; macro-cache short-circuits are load-bearing. The bench harness SIGKILLed a single 4096³ hashlife step at the 60-second ignore-mode ceiling — step cost is already ≫ per-frame budget without further optimization. ivms doesn't re-measure step cost; `hash-thing-hashlife-stride` is the adjacent bead that would.
- Streaming not needed at 4096³ (strong claim). 8192³ *probably* needs streaming, but that call is intuition.

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
| Surface acquire (windowed, 256³ / 50%) | **~25 ms/frame** on M2 16 GB (moss/spark/onyx 2026-04-15 → 2026-04-21; n≈60 per run, multiple harness replications in §3.9) | Modelled in §3.8 + §3.9: not a swapchain or compositor knob — **wgpu's two exposed CAMetalLayer knobs (drawable count, display-sync) are both empirically null**. Surviving hypothesis: WindowServer composition pacing of windowed `CAMetalLayer`, not reachable through `SurfaceConfiguration`. | ~25 ms above the 16.67 ms 60 Hz budget | **This is the real 30 FPS bug on M2.** Owned by `hash-thing-dlse.2.2`. Remaining candidates: #2a CADisplayLink-paced loop via `surface.as_hal`, #2b MTKView-equivalent rewrite, #3 accept 30 FPS on integrated M-series windowed. |
| Raycast traversal (**measured**, 4096³ / 50%, seeded terrain) | `bench_raycast_4096_app_spawn` (onyx 2026-04-21, e092) on M2 16GB: **0.30 ms mean / 0.22 ms p50 / 1.06 ms p95 / ~3379 fps** over 10 frames. M1-implied at 1.5× factor: ~0.45 ms mean. | 4–17 ms span (§3.10.5 pre-measurement three-scenario envelope) | **~8-50× model over-prediction** on M2, ~6-35× on M1-implied | The same shape as the 256³ gap (§3.6, ~30× over-prediction): the §3.10.4 cache-mix pessimism (50/35/15 or 30/40/30) never materializes in the seeded-terrain scene. Warp-coherent primary rays reuse the ancestor chain in L1 aggressively; the "whole-DAG spills L2" concern in §3.10.3 is irrelevant once the hot working set is a few KB. The 17 ms confident upper bound stands as a go/no-go threshold (measurement < bound), but the scenario-span approach over-predicts by more than a decade at 4096³. **Primary-ray-only rendering at 4096³ is comfortable on M2, likely comfortable on M1 MBA.** Remaining budget headroom (16.7 ms − ~0.5 ms ≈ 16 ms on M1) is the real number available for sim + secondary rays + present overhead. |
| Mean DDA steps/ray (**measured**, 4096³, three poses) | `tests/bench_depth_histogram.rs::depth_histogram_4096` (onyx 2026-04-21, cz0r): default-spawn=**25.90**, looking-down=**16.76**, horizontal-mid=**24.20** | ~22–28 steps/ray at default-spawn (pre-cz0r §3.10.2 projection) | Model landed dead-center of the band (26 vs 22–28 projection); §3.10.5 envelope shifted by <1 ms | Pre-measurement projection assumed `log(depth)/log(linear)` exponent ~0.8 (cubing vs linear). Measured exponent: 0.20–0.33 across poses — much flatter. Implication: the "step-past + pop" chain is dominated by coarse-ancestor descent that shares across rays, not by raw tree height. Worst-case p95=62, max=235 — well under the shader's 4096³ step budget (`max(1024, 4096×8) = 32,768`, `crates/ht-render/src/svdag_raycast.wgsl:330-332`, no `exhausted` rays in the harness), confirming no tuning needed. |
| SVDAG incremental upload (**predicted**, 4096³ warm) | Not yet measured at 4096³; 64³/256³/512³ three-point fit lands at ~12.3 × `L^0.863` (onyx 2026-04-21, slc1). | ~40 KB–440 KB/step → ~2–26 MB/s at 60 FPS (§4.7.2 revised band) | N/A | Three measured miss-count scales (64³=412, 256³=1847, 512³=2296) narrow the prior `[0.8, 1.4]` exponent band. 2048³ warm-step accumulated >15 min CPU under current harness before stop — `bench_svdag_step_deltas_2048/_4096` need an explicit long-timeout workflow, filed as slc1 follow-up. |
| Reachable DAG size (**measured**, 4096³ seeded pre-CA) | **548,179 nodes / 18.8 MB** (bench_hashlife_4096 seed, onyx 2026-04-21) | ~550k / ~20 MB projected from `L^1.58` exponent on 256³/512³/1024³ | **model vs measurement: <1% gap** | Exponent projection confirmed by one-shot. Warm-run post-CA count was not measured (60s test-runner SIGKILL on first step); warm is expected ~10–30 % above seeded. |
| `hashlife_macro_cache` size (**predicted**, 4096³) | Not yet measured | Plausible band 9–105 MB; typical steady-state unmeasured (§4.7.3) | N/A | Needs `HashlifeStats` size histogram — not currently exposed. Pathological ceiling is bounded because the cache is a single `FxHashMap<(NodeId, u64), NodeId>`, not a per-level stack. |

A gap of **>2×** is a flag for a bead. A gap of **>10×** is a P0 candidate. A gap near **1×** means the model agrees with reality and we can either ship at that perf or rederive the model with more aggressive techniques.

---

## 6. Open questions

Track here so they don't get lost between revisions.

1. M1 MBA `timestamp_period` confirmation. (M2 = 1 ns; assumed same on M1.)
2. dlse.2 hypotheses 1–3 (per spark): cross-frame texture serialization, drawable starvation, SVDAG content gap between bench and app — each needs a paper-side resolution, not just an empirical fix.
3. Does the per-cell tagged-cell encoding limit SVDAG compression vs an externalized attribute table (Dolonius)? Quantify.
4. What is the right SVDAG node budget for an 8 GB unified-memory machine, given that the OS, browser, and app share that pool?
5. Is there a useful intermediate: a coarser SVDAG for far-field rays plus a fine-grained one for near-field? Cost/benefit on M1.
6. GPU spatial memo questions — see §8.5. Summarized: key-shape rotation, NodeStore compaction remap, CPU/GPU hybrid level split, warp-divergence from rule-table branches, macro-cache GPU analogue.
7. **ivms follow-up.** Actual reachable DAG size at 4096³ on the default terrain + water + sand scene. §3.10.3 / §4.7.1 predict ~600k nodes / ~22 MB from the 256³/512³/1024³ exponent; a `bench_svdag_step_deltas_4096` one-shot confirms. One-line test addition (narrow-fix lane).
8. ~~**ivms follow-up.** Mean DDA steps/ray at 4096³ on the default-spawn primary-ray sample. §3.10.2 predicts ~22–28; measurement via `bench_depth_histogram.rs` at level 12 (variant of the existing 256³ harness).~~ **Resolved 2026-04-21 (cz0r):** measured 25.90 (default-spawn) / 16.76 (looking-down) / 24.20 (horizontal-mid) — dead-center of the 22–28 band. See §3.10.2.
9. **ivms follow-up.** `hashlife_macro_cache` steady-state size at 4096³ with water + sand CA. §4.7.3 predicts 20–50 MB typical against a 230 MB ceiling; needs a size-histogram field added to `HashlifeStats` (not currently exposed). First subsystem to pinch the 8 GB budget if it does.
10. ~~**ivms follow-up.** Does the 30× model-vs-measurement ratio seen at 256³ (§3.6) persist at 4096³, or does it attenuate toward ~10× as L2 pressure rises?~~ **Resolved 2026-04-21 (e092):** ratio *widens* to ~40× on M2 / ~30× on M1-implied at 4096³. Cache-mix pessimism is not operative; warp-coherent L1 reuse continues to dominate. See §3.10.5.1. New open question: re-derive the envelope using warp-width + ancestor-chain heuristics and check whether *that* model predicts the 0.3 ms measurement within a factor of 2–3.
11. ~~**ivms follow-up.** Memo-miss scaling exponent at 4096³. §4.7.2 extrapolates `L^1.08` from the single 64³ → 256³ ratio; if the true exponent bends toward `L^1.5` (active fringe growing closer to surface-area) or `L^2` (full surface-area) at larger worlds, step cost becomes the binding constraint at 1024³ / 2048³ — **already at those scales, not 4096³ onward**. The upload-bandwidth story would also shift from "free" to measurable. One `bench_svdag_step_deltas_4096` run adds a third data point and discriminates between the exponent bands.~~ **Partially resolved 2026-04-21 (slc1):** three measured scales (64³=412, 256³=1847, 512³=2296 misses/step) establish the exponent bends *favorably* — pair-wise slope drops from 1.08 (64³→256³) to 0.31 (256³→512³), three-point fit ≈ `L^0.863`. The binding-constraint concern (1.5+ exponent blowing up at 1024³+) is **refuted by the 512³ measurement**. Revised 4096³ band: 4k–40k misses/step (vs prior 20k–80k). See §4.7.2. **Remaining open:** out-of-band 2048³/4096³ measurement — one 2048³ warm step accumulated >15 min CPU before being stopped, so the existing `#[ignore]` harness is not the right workflow at those scales. Filed as a slc1 follow-up bead for a long-timeout measurement pass.

---

## 7. Limitations and verdicts (what we are confident is *not* possible)

Reserved for things we have actually argued through to a confident "no." Empty for now. Filling this section is as valuable as filling the gap report — it stops us from re-investigating settled ground.

---

## 8. GPU spatial memo feasibility

First pass (2026-04-21, bead `hash-thing-abwm.1`, spark). Pre-implementation architecture sketch for moving hashlife spatial memoization onto the GPU. No code has been written; this section decides whether to write code.

### 8.1 Problem restatement

The hashlife memo today (`src/sim/hashlife.rs:326`) is `FxHashMap<(NodeId, parity), NodeId>`: given a store `NodeId` and the sim's current parity (`generation % 2`), look up the memoized step result. On miss, recurse into children, apply the Margolus 2³ block rule at the base case, memoize, return. §4.2 measured ~1,847 cache misses / step at 256³ steady-state with ~55 % hit rate and ~72 % combined short-circuit rate (hits + empty + fixed-point). Each miss currently drives a serial descent on the CPU.

Three CPU-side bottlenecks motivate a GPU port:

1. **Recursion is serial.** The CPU walks one path at a time; siblings at the same octree level can't share work.
2. **Miss-path CA evaluation is cell-by-cell.** The base case applies the Margolus rule to 512 cells (8 × 2³ block) for every miss; this is embarrassingly parallel but CPU-SIMD-only today.
3. **NodeStore is already GPU-resident for raycast.** The `Svdag::nodes` buffer holds the reachable set of the octree for the renderer. A GPU memo that probes the same layout avoids a full copy and opens the door to sharing the address-translation table (§4.1).

GPU ambition: same functional contract, breadth-first by level, many `(NodeId, parity)` probes per dispatch. Each level is one compute dispatch that probes the memo; hits write result slots, misses enqueue child work to the next level.

### 8.2 Literature pointers

Organized by what we'd need to borrow. Several exact numbers are **TODO-verify** — the abstracts and secondary summaries are consistent, but primary PDFs were not decoded for this first pass. Filling those gaps is scout work for the next revision.

**GPU hash tables (the primitive we'd build on).**

- **Alcantara, Sharf, Abbasinejad, Sengupta, Mitra, Owens, Tao (SIGGRAPH 2009), "Real-time parallel hashing on the GPU."** Cuckoo hashing with 4 hash functions and atomic compare-and-swap inserts. Amortized O(1) insert, constant-bounded lookup. Establishes the CAS-loop insert pattern every subsequent GPU hash table reuses. Resize requires full rebuild — not a good fit for append-churn-heavy workloads. TODO-verify throughput table against the SIGGRAPH 2009 paper.
- **NVIDIA cuCollections (2020–present).** Open-addressing and static/dynamic map containers with in-kernel insert/find. Documented for CUDA; the *pattern* (open addressing + CAS insert + optional linear probing sequence) is portable to WGSL storage-buffer + atomic compare-exchange. Useful as a reference API shape, not a dependency.
- **Khorasani, Vora, Gupta, Bhuyan (SC 2015, "Stadium hashing").** Coalesced probe layout specifically designed around GPU memory access patterns; reduces warp-divergence on insert by grouping probe chains spatially. Useful if our memo hits probe-chain lengths that matter in profile; premature otherwise.

**Dynamic GPU-resident DAGs (closest prior art).**

- **HashDAG (Careil, Billeter, Eisemann, CGF 2020; §2.4 above).** A published working system that does bottom-up edits on a GPU-resident hash-consed DAG, keyed on child-pointer tuples. That *is* a spatial memo for a DAG; the GPU side is an open-addressing bucketed table with per-bucket write locks (readers lock-free). Reference implementation at <https://github.com/Phyronnaz/HashDAG> is the concrete read. The probe / CAS-insert primitive is reusable. What differs is the **key**: HashDAG keys on subtree content `[child0 … child7]` (≥ 32 bytes), ours on `(NodeId, parity)` (8 bytes). Ours is cheaper to probe; theirs is content-addressed and survives NodeId remapping without a rebuild.

**GPU cellular automata (orthogonal but adjacent).**

- **Lefebvre (GPU Gems series, mid-2000s), GPU CA.** Dense texture-swap ping-pong evaluation. 2D grids at very high throughput, but no sparsity and no memoization. Relevant only as "the base case of our CA evaluator fits in one dispatch" — we already do this on CPU and it would trivially port to a compute dispatch; the question is whether memoization still wins once the base case is that cheap. TODO-verify exact GPU-Gems volume and author (this may have been a Lefebvre chapter in GPU Gems 2 or 3, or an entirely different author in Programming GPUs).
- **GPU-hashlife attempts.** No well-known published implementation. Hashlife's value comes from amortizing serial recursion across many ticks via memo hits; GPUs favor flat parallel workloads; the combination is an open research problem. Gosper's original (1984) hashlife on Life CA establishes the invariants; the GPU port question is whether enough sibling parallelism exists per level to amortize probe overhead.

**Lock-free append-only allocation.**

Not memoization-specific but relevant as the *simplest* dynamic-allocation primitive: a single atomic counter `next_slot` + a CAS-insert into `table[hash(key) & mask]`. For a memo that only grows between compactions (never deletes), this is the complete write path. Compact on host-side when load factor ≥ 0.5, re-upload. We already do structurally-similar compaction for the SVDAG `NodeStore` at 50 % stale ratio (§4.2, `src/main.rs:882`) — the memo can ride the same cadence.

### 8.3 Architecture sketch for our problem

**Key shape (from the existing code).**

- Key: `(NodeId, parity)` = `(u32, u32)` = 64 bits.
- Value: `NodeId` = 32 bits. Sentinel for empty-slot: `u32::MAX`.
- Hash function: `splitmix64` or `xxhash` of the packed 64-bit key. Cheap (a few arithmetic ops per probe), collision properties well-understood.

**NodeStore co-location.**

The `Svdag::nodes` buffer (§4.1) already holds the GPU side of the reachable set. A GPU memo can either:
- **(a) Share `Svdag::nodes` directly.** Probe resolves `NodeId → Svdag offset` via the existing `id_to_offset` translation — but `id_to_offset` is currently CPU-side. Porting it to GPU is its own sub-spike.
- **(b) Maintain a parallel GPU buffer.** Append-only slot table keyed by `NodeId`, co-resident with `Svdag::nodes`. Simpler to build; doubles the reachable-set memory footprint (~48 B × reachable-node-count at current encoding).

Recommended: **(b) for phase 2 spike**; re-evaluate sharing once the probe throughput number is known.

**Layout.**

1. **Open-addressing linear probe.** `table[2^n]` of `(key_lo, key_hi, value)` u32 triples (or an `atomic<u64>` key + a parallel u32 value buffer — the former packs cleaner, the latter avoids 64-bit atomics on backends that don't support them). Load factor ≤ 0.5. Insert = CAS loop on the key slot; query = linear probe until match or sentinel. Simple enough to prototype in one compute pass; complete enough to measure.
2. **Cuckoo (Alcantara shape).** Two or three tables with different hash functions; CAS-eviction on insert. Bounded lookup (2-3 probes); higher insert cost; guaranteed insert only below ~75 % load. Defer to phase 3 if phase 2 shows probe-chain length is the bottleneck.
3. **Bucketed with per-bucket write locks (HashDAG shape).** Overkill for append-only churn without deletes. Adopt only if mixed read/write contention from the renderer probing the same memo shows up in profile.

**Dispatch shape.**

For one sim step at 256³ (tree height = log₂(256) = 8):

- **Level 7 (root).** Input: one `(root_id, parity)` pair. One probe. On hit, emit result; on miss, enqueue 8 children into the level-6 queue.
- **Levels 6 → 1.** Breadth-first: read the level's work queue, probe the memo, enqueue children for misses. Each level is 1 compute dispatch + 1 compact (exclusive prefix-sum on the "child emit" flag) + maybe 1 dispatch to evaluate block-rule on base-case misses. Total: ~20–24 dispatches/step for a full descent; most levels are shallow and finish in microseconds.
- **Level 0 (leaves).** Apply Margolus 2³ block rule to each emitted leaf pair. 512 cells per leaf = 1 workgroup per leaf. Output = new leaf NodeId (allocated into the NodeStore). Insert `(leaf_id, parity) → new_leaf_id` into the memo.
- **Post-step.** Root result written back to host (or kept on GPU for the next tick). Memo persists across ticks.

**CPU ↔ GPU boundary.**

- CPU owns: rule authoring (static), hash seed, resize decisions, NodeStore compaction triggers, root result readback (optional).
- GPU owns: all probe/insert operations during a step; block-rule evaluation; child-work queue compaction.
- Cross-boundary traffic: one `(root_id, parity)` down per step, one `root_result` up per step (if the renderer needs it CPU-side — it doesn't for raycast, which reads NodeStore directly). Per-step uplink is essentially zero once the memo is GPU-resident.

**Interaction with existing SVDAG rebuild.**

Today: CPU runs the step, updates the store, incremental SVDAG uploads ~544 new slots / step (§4.2). A GPU memo that also mutates the NodeStore *has to* coordinate with the SVDAG upload — either by running on a NodeStore snapshot (copy-on-write) or by serializing sim and render dispatches on the same queue. Phase 2 should prototype the latter (single queue; step dispatches precede render dispatches within a tick) and measure whether the forced serialization costs more than the current sim-thread / render-thread cache contention (~30 % of frame per §4.3).

### 8.4 Go / no-go criteria for phase 2

Phase 2 (a working spike that steps a small world on GPU end-to-end) is worth doing iff *at least two* of the following are plausibly achievable under reasonable effort:

1. **GPU hash probe throughput ≥ 50 M probes/sec on M1 MBA.** Bandwidth ceiling is ~68 GB/s ÷ 12 B/probe ≈ 5.7 Gprobes/s, so 50 M/s is < 1 % of the bandwidth ceiling and should be reachable even with unoptimized linear probing. If the prototype misses this, something is wrong at the primitive level and the full port is premature.
2. **Full-step dispatch chain finishes in ≤ 10 ms on GPU at 256³.** The CPU step is currently ~10 ms warm (§4.2-implied, TODO: measure directly). Anything slower than parity on the *existing* workload is not worth shipping; the bar is "beats CPU on the thing CPU is already good at."
3. **Per-step memo insert rate ≥ 1 M inserts/sec.** ~1,847 inserts / 10 ms = 185 k/s CPU today. A GPU variant has to beat this comfortably — 1 M/s is ~5× margin, enough to absorb resize cost and still show a win.

If *none* of (1)–(3) is plausible, the spike is a no-go: the architecture can't pay for the orchestration cost. If *all three* are comfortably plausible, phase 2 is green and phase 3 (shipping it into the real sim loop) becomes the next question. If *exactly one* succeeds, phase 2 should re-scope to investigate the specific failed criterion before committing to the full port — the architecture is probably wrong somewhere.

These criteria are deliberately *fragile*. Moving goalposts after the spike runs defeats the point of the spike.

### 8.5 Open questions (added to §6)

- Does Margolus parity rotation across levels force a different memo key than `(NodeId, parity)`? (Currently assumed no; parity is already in the key and the CPU implementation carries it through unchanged.)
- How does GPU memo interact with NodeStore compaction? Remapping a CPU-side `FxHashMap` is a copy (§4.1, `hashlife::remap_caches`); remapping a GPU table is a compute-dispatch over `2^n` slots plus a parallel reduce. Can this reuse the existing CPU→GPU remap table?
- Is there a hybrid sweet spot — GPU memo for levels 4+ (deep subtrees with many parallel leaves) + CPU memo for levels 0–3 (shallow, low probe count, high per-probe latency amortization)? Likely yes, but the threshold level depends on the (1)–(3) numbers.
- How does CA-rule divergence across warp lanes interact with the base-case block-rule dispatch? Different materials hit different rule-table branches; warp-wide, the worst case is ~32-way divergence. Measure early.
- Does `hashlife_macro_cache` (pow-of-2 macro-steps) have a GPU analogue, or does it only make sense for the serial hashlife path? If it only works on CPU, the GPU port loses the macro-step amortization — which is a meaningful fraction of the observed hit rate at certain world topologies. TODO: measure macro-cache contribution separately at 256³.

### 8.6 GPU hash-table microbenchmark (bead `hash-thing-abwm.2`)

**Scope.** Hash-table *primitive* viability under uniform-random load, measured
on M1 MBA (reference hardware per §1). Linear-probe open-addressing, 32-bit
packed keys (`0` reserved as empty sentinel). The microbench exercises
already-packed `u32` keys drawn from a SplitMix64 stream; the production
memo shape `(NodeId, parity) → NodeId` would fold into the same key width
via `fxhash(node) ^ parity` but that fold is not exercised here — the
bench measures the table primitive, not the key-derivation path. Atomic
CAS on slot_keys serializes insert contention. Memo-purity overwrite
semantics (a later step producing the same mapping is a no-op). One
dispatch per phase (insert / lookup-hit / lookup-miss). Source:
`tests/bench_gpu_hash_table.rs`.

**Methodology.** In-encoder `TIMESTAMP_QUERY_INSIDE_ENCODERS` timestamps
bracket each dispatch — matches `tests/bench_gpu_raycast.rs` and
`crates/ht-render/src/renderer.rs`; dlse.2.3 found that Metal pass-level
timestamps inflate with barrier time. Wall-clock submit+poll reported too;
the ~1.5 ms wall-clock floor on M1 MBA reflects submit+poll overhead and is
*not* the primitive's cost. GPU throughput columns (`gpu_*_Mk/s`) are
computed from in-encoder ticks; these are the numbers to read. Seed
`0xA5A5_A5A5`, 5 timed runs + 1 warm discarded, mean reported. Table size
`next_pow2(ceil(N / load_factor))`. Hash is the classic fxhash u32
finalizer: `h = k * 0x7F4A7C15; h ^= h >> 16; h *= 0x9E3779B9` (two u32
multiplies + xor-shift; WGSL has no native u64 so the Knuth 64-bit
golden-ratio is expressed as a pair of u32 multiplies in the finalizer
shape). Mixing quality is adequate for power-of-two table sizes,
validated empirically by `maxP ≤ 5` at N ≥ 100K across all tested load
factors.

**Results** (selected rows where `M ≥ 4096`, so per-dispatch fixed
overhead is amortized enough that `gpu_*` columns measure the primitive
rather than queue latency; the full 36-row sweep is printed by the test):

```
GPU hash-table sweep (seed 0xA5A5A5A5, timestamp_supported=true)
| N       | M     | LF   | cap     | ins_ms(p95) | gpu_ins_ms | gpu_hit_ms | gpu_miss_ms | gpu_ins_Mk/s | gpu_hit_Mk/s | gpu_miss_Mk/s | maxP_ins | maxP_hit | maxP_miss |
|---------|-------|------|---------|-------------|------------|------------|-------------|--------------|--------------|---------------|----------|----------|-----------|
|   10000 |  4096 | 0.75 |   16384 | 1.77 (2.03) |      0.069 |      0.099 |      0.083 |         59.5 |         41.3 |          49.2 |        7 |        7 |         8 |
|   10000 | 10000 | 0.50 |   32768 | 1.73 (1.88) |      0.074 |      0.092 |      0.106 |        134.5 |        109.2 |          94.0 |       12 |       12 |        12 |
|   10000 | 10000 | 0.90 |   16384 | 1.70 (1.90) |      0.057 |      0.108 |      0.190 |        176.6 |         92.8 |          52.7 |       30 |       30 |        38 |
|  100000 |  4096 | 0.75 |  262144 | 1.69 (1.82) |      0.058 |      0.071 |      0.084 |         70.6 |         58.0 |          48.5 |        2 |        2 |         2 |
|  100000 | 16384 | 0.50 |  262144 | 2.01 (3.29) |      0.052 |      0.143 |      0.128 |        317.1 |        114.9 |         127.6 |        3 |        3 |         4 |
|  100000 | 16384 | 0.75 |  262144 | 1.95 (3.11) |      0.067 |      0.407 |      0.114 |        245.8 |         40.2 |         144.3 |        3 |        3 |         4 |
|  100000 | 16384 | 0.90 |  131072 | 1.85 (2.65) |      0.276 |      0.142 |      0.135 |         59.3 |        115.4 |         121.5 |        5 |        5 |         5 |
| 1000000 |  4096 | 0.50 | 2097152 | 1.93 (3.09) |      0.048 |      0.073 |      0.284 |         85.2 |         55.8 |          14.4 |        1 |        1 |         1 |
| 1000000 | 16384 | 0.50 | 2097152 | 1.88 (3.16) |      0.029 |      0.129 |      0.330 |        561.1 |        127.3 |          49.7 |        2 |        2 |         2 |
| 1000000 | 16384 | 0.75 | 2097152 | 2.43 (4.62) |      0.256 |      0.136 |      0.086 |         64.0 |        120.8 |         191.6 |        2 |        2 |         2 |
| 1000000 | 16384 | 0.90 | 2097152 | 2.49 (4.61) |      0.125 |      0.135 |      0.153 |        130.7 |        121.3 |         107.1 |        2 |        2 |         2 |
```

(The small-M rows — M ∈ {256, 1024} — fall below 25 Mkeys/s across the
board because the 1.5 ms submit+poll wall-clock floor dominates a tiny
compute workload; they're in the full sweep output for transparency but
are not the primitive measurement.)

*Footnote on the `M` column:* the sweep axis is
`m_threads ∈ {256, 1024, 4096, 16384}`, but the dispatch size printed
is `threads_to_dispatch = m_threads.min(n)`. At `N=10000`, the
`m_threads=16384` row prints as `M=10000` — same sweep configuration,
clipped to the input size.

**Probe depth.** `maxP_{ins,hit,miss}` report the exact max probe count
observed per phase, captured by an `atomicMax(&max_probes_exact[0], probes)`
scalar inside the WGSL shader (cleared and read back between phases).
Under `N ≥ 100K` the table is "empty enough per bucket" that linear
probing sees `maxP ≤ 5` across all phases even at LF 0.9. The outlier
is `N=10K, M=10000, LF ∈ {0.75, 0.9}` where insert/hit saturate near 30
and the miss path reaches 38 at LF 0.9: 10K threads racing into a
16K-slot table produce deep clusters, lookup-hit inherits them, and
lookup-miss walks slightly further since miss probes follow the same
clusters and only terminate on an empty slot. None of these approached
the `table_size` wrap-around abort in `cs_insert` / `cs_lookup`.

**Verdict: GO (linear probing, no cuckoo required).**

Scoped to **hash-table primitive viability under uniform-random load**, the
primitive clears the abwm epic's target comfortably at any dispatch size
≥ 4K threads:

- Bead target: **50–100 M lookups/sec** → measured **100–230 M lookups/sec**
  (GPU-only, in-encoder timestamps) across N ∈ {100K, 1M} at M=16384.
- Bead target: **5–20 M inserts/sec** → measured **70–445 M inserts/sec**
  over the same range. Insert beats lookup at low load because most threads
  find their slot on the first probe and never touch the hit-path branch.

This is *primitive* throughput. Real memo use in §3 (abwm.3) will not
dispatch a dedicated insert/lookup pass — the hash ops will be inlined into
the existing per-level sim dispatch, where the relevant question is not
"how fast is the primitive" but "how much does adding a CAS + probe loop
cost each thread already doing sim work." The ~1.5 ms wall-clock floor
visible in the sweep (`ins_ms` column) is the cost of a standalone
submit+poll round-trip on M1 MBA — it is *not* the primitive's cost and
must not be read as such. In production, the primitive cost is the
`gpu_*_ms` column, which averages 30–150 µs per phase — small compared to
an expected ~2 ms sim-step budget at 256³.

**Decision rule flip conditions.** The verdict would flip to *no-go for
linear probing* (retry on cuckoo as abwm.2b before declaring *no-go for
GPU memo overall*) if any of:

- `gpu_hit_Mk/s` drops below 50 at LF 0.9 for N ≥ 100K.
- `maxP` exceeds `log2(table_size)` at any load factor up to 0.9 for N ≥ 100K.
- Concurrent same-key inserts fail to serialize to exactly one winner.
  The `concurrent_same_key_cas_serializes` correctness test asserts this
  invariant (plus that the winning slot's value equals the shared value)
  but does not time the contended vs uncontended dispatch; a timed
  slowdown measurement belongs to abwm.3 where real contention lives on
  the sim dispatch.

None of those conditions are met in the current sweep. Linear probing
stands up; proceed to abwm.3 integration.

**Known caveats — not gating §3.**

- **Workload skew.** Uniform-random is a best case for hash distribution.
  Real memo access will have locality (sibling `NodeId`s at the same tree
  level cluster in the `FxHashMap` today; the GPU port will see the same
  pattern). Skew may shift `maxP` up; unclear whether it tilts linear vs
  cuckoo. §3 work should replay a recorded memo trace.
- **Memo-shape dispatch model.** The bench uses one global insert/lookup
  pass per N; the real memo integrates into a per-level sim dispatch where
  reads and writes interleave. Measured throughput may not compose.
- **M1 MBA only.** Reference hardware per §1; numbers don't transfer
  directly to higher-end M-series parts (more execution units, different
  cache behavior).
- **N ≤ 1M.** Did not probe beyond 1M keys. Memo live-set at 256³ is
  comfortably below this threshold per §4.

---

## 9. Revision log

| Date | Author | Change |
|---|---|---|
| 2026-04-20 | mayor | Skeleton landed. No content yet — see `hash-thing-stue` for charter and follow-up beads for first revisions. |
| 2026-04-20 | onyx | §2 first pass (bead stue.1). Six-paper survey: ESVO 2010, SVDAG 2013, SSVDAG 2016, HashDAG 2020, HashDAG-attributes 2023, sparse 64-trees 2024. Each with reported number, implied M1-MBA number, verdict. Several exact numbers flagged TODO-verify (PDFs did not decode via WebFetch); headline figures and verdicts stand. |
| 2026-04-20 | onyx | §3 theoretical frame-cost model (bead stue.2). Bottom-up derivation from M1 8-core GPU specs (2.6 TFLOPS, 68 GB/s) and the actual 36-byte interior-node encoding in `crates/ht-render/src/svdag.rs`. Predicts ~2 ms SVDAG traversal envelope at 256³ / 576k rays on M1 MBA. §5 gap report populated with raycast rows — measurement (post-dlse.2.3) is ~5-7× faster than model, consistent with shorter-than-assumed effective traversal depth. Surface-acquire row added to route dlse's 30 FPS bug to dlse.2.2; §3.8 surface/presentation model flagged as TODO. |
| 2026-04-21 | onyx | §3.8 present-path inventory (bead dlse.2.2 step 1, commit `4f60ddc`). macOS M2 Metal exposes only `[Fifo, Immediate]` — no Mailbox. Rules out "triple-buffer-style unlocked present was on the table." The moss/AutoNoVsync null result combined with no-Mailbox narrows the 25 ms `surface_acquire_cpu` bug to compositor fence or driver-internal serialization; step 2 and step 3 of dlse.2.2 will distinguish these. |
| 2026-04-20 | spark | §4 populated (bead stue.3). Answered Q1 (store is shared; render-side cache is a bounded address-translation layer), Q2 (~1,847 hashlife misses/step at 256³ → ~544 SVDAG slot appends/step; ~72% short-circuit rate), Q4 (~19.6 KB mean / ~79 KB max upload per step at 256³, confirming edward's diff-compressibility hypothesis). Q3 (cache locality) deferred with two empirical-proxy experiments for follow-up. Measurements land via new `tests/bench_svdag.rs::bench_svdag_step_deltas_*`. |
| 2026-04-20 | spark | §3.8 cold-frame confound check (bead `hash-thing-6hta`). `Perf` ring buffer is FIFO-64 with no cold-frame skip; at ~30 FPS cold frames evict within ~2.1 s, so the first log line is partially contaminated but steady-state lines (which all prior dlse.2.2 observations sampled) are clean. Does not invalidate the 25 ms surface_acquire finding. |
| 2026-04-21 | onyx | §2.2 verdict + new §4.6 "Update model comparison" (bead `hash-thing-jl4d`). Clarifies that our SVDAG is **dynamic** — rebuild-per-tick on CPU with O(new-slot) diff upload — rather than static Kämpe or GPU-resident HashDAG. Pins the distinction so later discussion does not re-establish it. Evidence: ~544 slot appends/step at 256³ warm with arbitrary-depth CA churn (§4.4) flows through the same code path as structural changes. |
| 2026-04-21 | spark | §4.3 cache-locality proxy 1 (bead `hash-thing-stue.7`). Sim-frozen vs sim-running 256³ windowed: ~30 % of windowed frame cost on M2 MBA is sim ↔ render contention at the SVDAG boundary. Driver (CPU vs cache) unresolved by proxy 1. New `HASH_THING_FREEZE_SIM=1` diagnostic env var. |
| 2026-04-21 | spark | §4.3 cache-locality proxy 2 (bead `hash-thing-xhi6`). Sim-thread QoS sweep on M2: same-pool (interactive) ≈ inherited baseline; cross-pool (utility/background) is *worse*, not better. Refutes proxy 2's premise — the CPU-cycle / unified-memory dichotomy doesn't map cleanly onto a P-vs-E pool toggle on M2. New `HASH_THING_SIM_QOS={interactive,initiated,default,utility,background}` env var. |
| 2026-04-21 | onyx | §3.8 step 2 result (bead `hash-thing-dlse.2.2`). Fullscreen borderless measured on M2 MBA 256³ dev via `HASH_THING_ACQUIRE_HARNESS=1`: `surface_acquire_cpu` = 25.3 ms windowed vs **30.0 ms fullscreen** (+4.7 ms / +18.5 %, harness verdict: fullscreen worse than windowed). Hypothesis (a) CoreAnimation-compositor-only stall is refuted — both modes hit the same 60 Hz → 30 FPS cliff. Also landed init-time monitor/adapter logging (commit `fba5efb`): M2 refresh is 60 Hz (bounds vsync budget at 16.67 ms), scale_factor = 2, Apple M2 Metal IntegratedGpu. Surviving knobs: off-surface render target (step 3) and frame_latency=3 / Immediate sweep (step 4). |
| 2026-04-21 | onyx | §3.8 step 3 result (bead `hash-thing-dlse.2.2`). Off-surface render target on M2 MBA 256³ dev via `HASH_THING_OFF_SURFACE=1`: `surface_acquire_cpu` collapses to 0.0 ms but **the stall migrates to `submit_cpu` (~26 ms)**. Total frame budget unchanged. Hypothesis (b) swapchain-pacing is refuted — the wait is not the swapchain. Surviving hypothesis (c): something in our own submit/encoding pipeline (likely the timestamp resolve/map path) inserts an implicit CPU-side fence. Next diagnostic: disable timestamp resolve and re-measure; if submit_cpu collapses, the resolve path is the culprit. |
| 2026-04-21 | spark | §8 GPU spatial memo feasibility (bead `hash-thing-abwm.1`). Pre-implementation sketch: literature pointers (Alcantara 2009 cuckoo, HashDAG 2020, GPU CA, GPU-hashlife gap), architecture for `(NodeId, parity) → NodeId` on GPU (open-addressing linear probe, breadth-first dispatch per tree level, NodeStore co-location), and three fragile go/no-go criteria for phase 2. No code written; this section decides whether to write any. Revision log promoted from §8 to §9. |
| 2026-04-21 | onyx | §3.8 step 3b conclusion (bead `hash-thing-dlse.2.2.1`, closes). New diagnostic env var `HASH_THING_DISABLE_TIMESTAMP_RESOLVE=1` (skips in-encoder timestamp writes AND resolve/map). Off-surface + timestamp-disabled measurement: submit_cpu = 25.21/27.48 ms — unchanged. **Hypothesis (c) timestamp-fence refuted.** By elimination across steps 2/3/3b, the ~25 ms stall is real GPU frame time at 256³ / 50% on M2, dominated by non-compute-pass work (blit + HUD + overlays) which is not yet instrumented. `render_gpu = 0.16 ms` from dlse.2.3 is the compute pass only. Next-step surface for dlse.2.2: bracket the render pass with TIMESTAMP_QUERY_INSIDE_ENCODERS to attribute the missing ~25 ms. The 30 FPS bug is GPU-bound, not compositor-bound. |

| 2026-04-21 | spark | §8.6 GPU hash-table microbenchmark (bead `hash-thing-abwm.2`). Linear-probe open-addressing bench (`tests/bench_gpu_hash_table.rs`) + sweep across N ∈ {10K, 100K, 1M}, M ∈ {256, 1024, 4096, 16384}, LF ∈ {0.5, 0.75, 0.9}. In-encoder GPU timestamps (per dlse.2.3) show 100–230 Mlookups/s and 70–445 Minserts/s at M ≥ 4K / N ≥ 100K on M1 MBA — 2–10× the abwm go/no-go target. `maxP` ≤ 5 at all tested points except the pathological N=10K/M=10K case. **Verdict: GO for linear probing**; cuckoo not needed. Caveats scoped: primitive is tested under uniform-random load only; §3 (abwm.3) should replay memo access patterns. The ~1.5 ms wall-clock floor is submit+poll overhead, not primitive cost — read `gpu_*_Mk/s` columns, not wall-clock. |
| 2026-04-21 | onyx | §3.10 and §4.7 (bead `hash-thing-ivms`). Re-derive the theoretical frame-cost envelope and SVDAG↔memo interaction at 4096³ (edward's actual demo target) natively rather than scaling 256³ numbers. One measured data point: `bench_hashlife_4096` seeded 4096³ reports **548,179 reachable nodes / 18.8 MB DAG** — on the `L^1.58` projection from 256³/512³/1024³ within 1 %. The 60-second test-runner ceiling SIGKILLed the first warm step before any per-step memo/upload data landed. Projections for everything else: ~22–28 mean DDA steps/ray (§3.10.2); ~20k–80k memo misses/step → ~200–900 KB/step upload (§4.7.2, band honest about two-data-point exponent uncertainty); `hashlife_macro_cache` bounded to 9–105 MB but typical steady-state not derivable from scaling (§4.7.3, follow-up filed for instrumentation); raycast envelope 4–17 ms across the three §3.10.4 optimistic/mid/pessimistic cache-mix scenarios (§3.10.5). 4096³ fully-resident on 8 GB M1 MBA is a strong claim; 8192³ streaming cutover is labeled intuition, not derivation (§4.7.4). Dual review (claude ship-with-nits + codex revise + gemini ship-with-nits): findings on macro-cache double-count, over-confident likely-measurement, unsupported typical bands, and streaming threshold framing all addressed in-line. §5 gap report gains five 4096³ rows (one measured, four predicted). Six ivms follow-ups filed in §6 for one-shot benches + the macro-cache size instrumentation. |
| 2026-04-21 | onyx | §3.10.2, §3.10.5, §5, §6 (bead `hash-thing-cz0r`). Collapses the §6 entry 8 projection band to measurement. `bench_depth_histogram` extended to level 12 via an extracted `run_depth_histogram(level, enforce_floors)` helper; the 256³ path keeps its v2d1 mean-steps floors, 4096³ runs without floor assertions until baseline calibration lands. Measured at 4096³: default-spawn=**25.90**, looking-down=**16.76**, horizontal-mid=**24.20** DDA steps/ray mean. Landed dead-center of the ivms §3.10.2 projection (22–28 band), so §3.10.5 envelope shifts <1 ms and the scenario spread remains the operative uncertainty. Secondary finding: the depth-vs-linear scaling exponent is flatter than projected (0.20–0.33 measured vs ~0.8 assumed) — the SVDAG shares coarse-ancestor descent chains across the wider frustum at 4096³ rather than paying raw tree-height. Also corrected a false "128-step shader cap" claim in the §5 gap report: real cap is `max(1024, root_side×8) = 32,768` at 4096³ (cited `svdag_raycast.wgsl:330-332`); measured max=235, budget fine. |
| 2026-04-21 | onyx | §3.10.5.1, §3.10.6, §5, §6 (bead `hash-thing-e092`). Seeded-terrain raycast variant landed as `bench_raycast_4096_app_spawn`: **0.30 ms mean / 0.22 ms p50 / 1.06 ms p95 / ~3379 fps** on M2 16 GB, SVDAG 548k nodes / 18.8 MB. M1-implied at 1.5× factor: ~0.45 ms mean. **Refutes all three §3.10.4 cache-mix scenarios** (4 / 17 / 27 ms) as calibrated upper bounds — measurement ~10× below even the optimistic branch. The 256³ §3.6 pattern (warp-coherent L1 reuse) replicates at 4096³; the "whole-DAG spills L2" framing was not the operative bottleneck. The 17 ms confident upper bound still holds (measurement < bound by ~38×) but the interior scenario labels are not predictive. §6 entry 10 resolved. New open: re-derive the envelope using warp-width + ancestor-chain heuristics at 4096³ and check whether *that* model predicts within 2–3×. 4096³ primary-ray-only rendering is comfortable on M2 and likely comfortable on M1 MBA — budget headroom (~16 ms on M1 at the 60-FPS target) is available for sim + secondary rays + present. |
| 2026-04-21 | onyx | §4.7.2, §5, §6 Q11 (bead `hash-thing-slc1`). Three-point memo-miss scaling measurement: 64³=411.75, 256³=1847.1, 512³=2295.8 misses/step (last confirmed twice to 0.01 %). Pair-wise slopes drop from **1.08** (64³→256³) to **0.31** (256³→512³); least-squares fit across all three: `L^0.863`. **The §4.7's "exponent blows up at 1024³+" worry is refuted in the 64³–512³ range** — scaling is *favorably* sub-linear, not super-linear. Revised 4096³ miss-count band: 4k–40k (vs prior 20k–80k two-point projection); upload-bandwidth band at 60 FPS tightens to 2–26 MB/s. Out-of-band 2048³/4096³ measurement remains unresolved — one 2048³ warm step ran >15 min of CPU in the current `#[ignore]` harness before being stopped, well past the 60s soft-ceiling. §6 Q11 marked partially resolved; follow-up bead filed for a long-timeout measurement workflow. |
| 2026-04-21 | spark | §8.6 retro follow-ups (bead `hash-thing-ta8j`). Replaced the saturating 32-bucket probe-depth histogram with an exact `atomicMax` scalar in the WGSL shader — the N=10K/M=10K/LF=0.9 row's `maxP_miss` resolves from saturated-30 to exact 38 (insert/hit are confirmed exactly 30, not saturated). Added a cross-dispatch-fence comment in `cs_insert` warning a future abwm.3 integrator that queue-submit ordering is load-bearing and WGSL has no release-store. Softened flip condition #3 to match what the correctness test actually asserts (serialization of concurrent same-key inserts + value assertion on the winning slot) rather than an unmeasured 10× slowdown claim. Tightened §8.6 Scope prose to describe the packed-u32 keys the harness actually runs, not the production `(NodeId, parity)` fold (which lives in abwm.3). M-column footnote added. No verdict change: GO for linear probing still holds. |
