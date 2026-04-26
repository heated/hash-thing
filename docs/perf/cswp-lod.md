# Near-player vs far LOD policy (cswp.6)

**Status:** Design doc. No code yet — implementation lives in a follow-up sub-bead.

**Parent:** `hash-thing-cswp` (big-map streaming epic, target 4096³+).
**Companion:** `docs/perf/svdag-perf-paper.md` §3.13, §4.7 (cold-gen + 4096³ memory budgets).

This document defines the **memory/streaming LOD policy** for big-map worlds: which cells we keep at full detail, which we collapse, and how that decision shifts as the player moves. It does not redefine the **render LOD** the shader already does (pixel-projected subtree skip with a representative material). The two are related but separable; §1 walks the distinction.

---

## 0. Decision summary (for whoever claims the impl sub-bead)

- **What kind of LOD this is.** *Memory/streaming* LOD — what we hold for far-region cells. Render-side LOD is already intrinsic and stays.
- **Distance → LOD level mapping.** Logarithmic radii anchored to player chunk: full detail within 1 chunk, drop one octree level per doubling of chunk-radius. Tunable; defaults below.
- **Material-collapse policy.** **FEEL-GATE for edward — flag.** Default proposal: keep current "largest populated child" propagation; alternative is fixed per-material priority (rock > sand > water > air). Doc proposes default; edward picks if they disagree.
- **Transition handling.** Hard swap on chunk-radius crossing, not crossfade. Crossfade is an optional follow-up if hard swap reads as poppy in playtesting.
- **GPU buffer impact.** At 4096³ no streaming or LOD policy is required for memory; the LOD layer's value is bounded per-chunk cost for cswp.5's streaming bookkeeping. Memory + cold-gen savings dominate at 8192³+; streaming-plus-LOD becomes mandatory at 16384³+.

The visual lever is the material-collapse policy. The mechanics (radius curve, transition strategy) ship per the gate-tier rule "internal mechanics, ship without asking."

---

## 1. Two LODs: render LOD vs memory LOD

The codebase already has **one** form of LOD, but cswp.6 is asking for a **second**. They are separable and need separate language.

### 1.1 Render LOD (already shipped, see `svdag_raycast.wgsl` and `svdag.rs`)

Each interior SVDAG node packs a **representative material** into bits 8–23 of `slot[0]` (the child_mask word). The shader uses it like this (`crates/ht-render/src/svdag_raycast.wgsl:418-426`):

```wgsl
// LOD cutoff (x5w): if this voxel is sub-pixel, use the
// representative material packed in bits 8-23 of child_mask
// (m1f.7.3.2). No child scan needed — 0 buffer reads.
let lod_bias = max(u.debug.y, 1.0);
let t_dist = max(t + entry, 1e-4);
if pixel_world > 0.0 && half < pixel_world * t_dist * lod_bias {
    let lod_mat = (cmask >> 8u) & 0xFFFFu;
    ...
```

This is **screen-projected, ray-by-ray LOD**: when a node's half-extent projects below one screen pixel at the ray's current depth, the ray takes the representative material and stops descending. The criterion is `half < pixel_world * t_dist * lod_bias`, which already encodes "the farther this node is, the coarser we'll accept." Distance from the player is the dominant term in `t_dist`.

Material propagation during build (`svdag.rs:228-238`) prefers the **largest populated child** (by `population`) so that a sparsely-active subtree doesn't poison the whole coarse cell with a tiny pocket's material.

**This LOD costs nothing at memory level.** All 9 u32s of every node are still resident; the shader just chooses not to recurse. So render LOD does not reduce GPU bookkeeping at large worlds — it only shortens descent.

### 1.2 Memory LOD (what cswp.6 designs)

For 4096³+ worlds, the question is: **do we keep all of those subtrees at all, or do we drop them and let the parent stand in?** Memory LOD physically replaces a subtree with its representative material — the children stop existing in the resident DAG.

Memory LOD is a **superset** of render LOD: any subtree the renderer would skip (sub-pixel at most camera angles) is a candidate for being absent altogether.

Two reasons memory LOD is needed at all:

- **Cold-gen budget.** At 4096³ cold gen is 3.7 s (§3.13). The measured 1024³→4096³ ratio (15.5× time for 64× cells) gives time ∝ cells^0.66 ∝ L^1.98 — so 8192³ projects to ~15 s, 16384³ to ~58 s. Generating full detail for cells the player will never touch is wasted seed time even before memory becomes the bottleneck.
- **Long-tail memory growth.** §4.7.5 calls 4096³ resident on 8 GB a strong claim, but only because the dominant consumers are tightly bounded. Above 4096³, multiple consumers grow: macro-cache (unbounded analytically), `offset_by_slot` (~48 B/entry), and `id_to_offset` (~12 B/entry) all scale with reachable nodes, and reachable nodes grow ~`L^1.58`. Memory LOD breaks that scaling by capping reachable-nodes-as-a-function-of-distance.

A useful framing: **render LOD is a runtime ray-traversal decision; memory LOD is a build/streaming decision.** They share the same representative-material substrate, so adding memory LOD is mostly about *deciding when to build coarser nodes* — the SVDAG build path already produces the right shape.

---

## 2. Distance → LOD level mapping

### 2.1 What "LOD level" means here

The world is a hash-consed octree. At the leaf, every cell is a `MaterialId`. At each interior level, eight children. **LOD level k** for a region means: collapse the bottom k octree levels of that region into representative-material leaves.

- LOD 0 = full detail. Leaves are the actual cells.
- LOD 1 = 2³ blocks become single representative-material cells. 8× memory reduction per region.
- LOD 2 = 4³ blocks become single cells. 64× reduction.
- LOD k = (2^k)³-cells become single cells. 8^k reduction.

### 2.2 Distance → LOD level (proposed default)

Anchor: **player chunk** (one chunk = one octree subtree at some level — TBD by the impl bead, plausibly 64³ or 128³).

| Chunk-radius from player | LOD level | Block size | Per-cell memory savings |
|---:|---:|---|---:|
| 0–1 (containing + neighbors) | 0 (full) | 1³ | 1× |
| 1–4 | 1 | 2³ | 8× |
| 4–16 | 2 | 4³ | 64× |
| 16–64 | 3 | 8³ | 512× |
| 64+ | 4 | 16³ | 4096× |

Pattern: **one octree level coarser per 4× chunk-radius doubling**. In log space the curve is linear (LOD = floor(log₄(chunk_radius))) — geometric falloff matching the geometric memory savings.

**Why anchored on chunk-radius, not absolute distance.** Chunks are the natural unit for streaming (cswp.5 evicts whole chunks). Tying LOD to chunk-radius means every chunk is in exactly one LOD band; we don't re-collapse partway through a chunk's subtree.

**Why log₄ and not log₂.** A log₂ (one level coarser per chunk-radius doubling) is too aggressive: the player would see LOD-2 nodes within 4 chunks, and at chunk_size=64 that's only 256 blocks away — well inside what feels "near." log₄ keeps full detail out to 4 chunks (256 blocks at chunk=64) and reaches LOD-4 only at 64+ chunks (4096+ blocks), which lines up with the human field-of-view's shrinking angular resolution at distance.

**Tunable.** A single u32 uniform (or pair: `near_full_radius`, `falloff_base`) controls the curve. A `lod_bias` analog like the existing render-LOD `u.debug.y` knob lets us A/B aggressive vs gentle in playtesting without recompiling.

### 2.3 Alternative considered: pixel-projection-driven memory LOD

We could compute, at chunk-eviction time, the same `half < pixel_world * t_dist * lod_bias` criterion the shader uses, but at chunk granularity, and pick the LOD level that just-undercuts pixel size from the player's current viewpoint.

- **Pro:** Memory budget is set by the same rule the renderer would use anyway. No "but the renderer can already see detail there" waste.
- **Con:** Couples chunk LOD to camera FOV and screen resolution. A spectator on a different machine with a 4K monitor would need different chunks. Streaming becomes per-client, which is fine for single-player but bad for cswp's eventual interaction with `hash-thing-3bj` (multiplayer).

**Verdict: not chosen for default.** Chunk-radius is camera-independent and survives multiplayer. The pixel-projection criterion already lives in the shader and continues to skip detail the chunk LOD didn't strip.

---

## 3. Material-collapse policy ⚠️ FEEL-GATE for edward

When eight children of an interior node collapse to one representative material, **which material wins?** This affects how the world *looks* at distance — visible to the player, hence flagged.

### 3.1 Current behavior (today, in `svdag.rs:223-238`)

**"Largest populated child" propagation.** During SVDAG build, each interior node's representative material is taken from whichever child has the largest `population` (count of non-air leaves in its subtree). The first occupied octant is the tiebreaker for empty-leaf branches.

The comment in source explains the design tension:

> Propagate representative material from the largest populated child, not the first occupied octant. Water-heavy scenes often create tiny active pockets in low-index octants; "first occupied wins" turns whole coarse LOD nodes into arbitrary materials.

This rule is in production today and drives the existing shader render-LOD path. The cswp.6 question is: **do we also use this rule for memory LOD, or do we change the rule for both?**

### 3.2 Alternatives

| Option | Rule | Effect at distance | Cost |
|---|---|---|---|
| **A — Largest populated child (status quo)** | Pick the representative of the child with the most non-air cells. Recursive — works through arbitrary depth. | Faithful to "what's mostly there." A water-and-sand chunk reads as water if water dominates; sand if sand does. Boundaries between dominant materials read sharply. | None — already implemented. |
| **B — Fixed per-material priority** | Hardcoded order, e.g. STONE > METAL > SAND > GRASS > DIRT > WATER > LAVA > ICE > ... > AIR. Highest-priority occupied child wins. | Distance terrain reads as "the most structural material." A water-and-stone chunk reads as stone. Water becomes invisible at distance unless it's all-water. | Tiny — one priority array. |
| **C — Visually-dominant (perceptual brightness)** | Pick the child whose representative has the highest perceptual brightness × population product. Or the highest saturation. | Distance reads as "what stands out." Mossy stones in shadow disappear; bright sand/snow dominates. Skews aesthetic. | Modest — needs perceptual brightness per material; could derive from `MaterialVisualProperties::base_color`. |
| **D — Mixed dithering** | At LOD-1 collapse, alternate-octant-dither between the two largest children's materials. Hashed by node position. | Distance terrain has visible texture instead of solid blocks. Cost is shader-side: representative material stops being a single u16 and becomes a 2× scheme. | Larger — node encoding changes. |

### 3.3 Recommendation

**Default: keep option A (largest populated child).** Reasons:

1. Already implemented. No new code, no behavioral change for existing 256³ scenes.
2. Locally consistent — a flat sand desert reads as sand at every distance; a stone wall reads as stone.
3. Has a clean override path: edward can swap to B by adding a `MaterialPriority` tiebreaker without touching the build mechanics.

**Worth flagging:** option B is a real second-place candidate for the *demo* specifically. The demo is meant to read visually distinct from Minecraft (per `hash-thing-e7k` mayor note 2026-04-14); priority-driven collapse would emphasize the structural materials (stone, metal) over fluid pockets and might give distant lattice megastructures a more dramatic silhouette. If the cswp follow-up impl bead is also wired into the demo path (`hash-thing-e7k.3` lattice gen), worth A/B-ing.

**Park the choice for edward.** Specifically:

- Default ships with A.
- File a follow-up bead "evaluate option B for demo lattices at LOD-2+" if the demo footage shows distance terrain reading washed-out.
- Option C and D are tracked as future work; D requires shader-side encoding changes that don't pay off until we have a felt visual problem D solves.

---

## 4. Transition handling

When the player moves and a chunk crosses a LOD-radius boundary, what happens?

### 4.1 Options

- **Hard swap.** The chunk's resident representation is replaced from LOD-k to LOD-(k-1) (incoming, near) or LOD-(k+1) (outgoing, far). Frame N renders the old LOD; frame N+1 renders the new. Incoming swaps need the higher-detail nodes loaded; outgoing swaps just drop nodes. Cost: one frame's worth of LOD discontinuity at the chunk boundary.
- **Crossfade.** Both LODs co-resident for a short window (say, 200 ms). Renderer alpha-blends or dithers between them. Cost: doubled memory for the in-flight chunks; shader-side blend; no popping.
- **Re-collapse on the fly (continuous).** As the player moves, chunks *gradually* deepen or coarsen by one level at a time. Cost: bookkeeping, cache thrash, hard to bound the in-flight LOD set.

### 4.2 Recommendation

**Default: hard swap.**

Reasons:

1. The render LOD criterion is `half < pixel_world * t_dist * lod_bias` — it already smooths the transition. A chunk crossing from "memory LOD-2" to "memory LOD-1" still has the renderer using the same representative-material short-circuit at most camera angles. The swap changes "what's resident" more than "what's drawn."
2. Hard swap is the only option that bounds memory budget tightly. Crossfade and continuous-collapse both let the in-flight set grow under load (rapid traversal across many chunks).
3. Eviction (cswp.5) already needs hard chunk transitions for the memory side; making the LOD side match keeps the streaming model uniform.

**If hard swap reads as poppy in playtesting,** crossfade is the obvious upgrade — and it can be added without rewriting the LOD layer (it's purely a renderer concern). File as a follow-up bead.

**Continuous collapse is rejected.** Bounding the in-flight LOD set under arbitrary camera motion is a nontrivial scheduling problem; the value over hard-swap-with-optional-crossfade is small.

### 4.3 Transition trigger details

- The trigger is **chunk centroid distance to player chunk**, recomputed each frame for chunks near a band edge. A small hysteresis (e.g. ±0.25 chunk units) prevents thrashing when the player stands exactly on a band boundary.
- Outgoing transitions (moving away) drop nodes immediately; the SVDAG `compact()` pass (`stale_ratio > 0.5`, see §4.7.1) already handles the cleanup.
- Incoming transitions (moving toward) need the higher-detail nodes available. For seeded scenes those re-derive from `TerrainParams::for_level` — fast, no I/O. For player-edited scenes they come from cswp.2's on-disk format. The impl bead must verify both paths.

---

## 5. GPU buffer impact and budget shrinkage

### 5.1 Without LOD (extrapolation from §4.7)

Reachable-node count scales as `L^1.58` empirically (256³ → 4096³ measurement). Per-node cost: 36 B (DAG) + 48 B (`offset_by_slot`) + 12 B (`id_to_offset`) ≈ 96 B/node total resident.

| Scale | Reachable | Total resident (no LOD) |
|---|---:|---:|
| 4096³ | 548k | 53 MB |
| 8192³ | 1.65M | 158 MB |
| 16384³ | 4.97M | 477 MB |

Plus macro-cache (9–105 MB at 4096³ band, growing with reachable). Total at 8192³ without LOD: ~277 MB; at 16384³: ~700 MB+. Both within an 8 GB envelope numerically, but unmodeled overheads (NodeStore active region, view-shift churn, wgpu driver state) eat the margin.

### 5.2 With LOD policy from §2.2

Per the §2.2 mapping, with chunk-side 64 and player at world center:

| Band | Chunk count | Compression | Full-equivalent chunks |
|---|---:|---:|---:|
| LOD-0 (r ≤ 1) | 27 | 1× | 27 |
| LOD-1 (1 < r ≤ 4) | 9³−3³ = 702 | 8× | 88 |
| LOD-2 (4 < r ≤ 16) | 33³−9³ ≈ 35,200 | 64× | 550 |
| LOD-3 (16 < r ≤ 64) | bounded by world | 512× | depends on scale |
| LOD-4 (r > 64) | bounded by world | 4096× | depends on scale |

The LOD-3 / LOD-4 contribution is world-size-dependent. Worked examples:

- **4096³ (64 chunks across at chunk=64).** Player at center: max chunk-radius 32. LOD-3 ring is 33 < r ≤ 32 (band is empty — world ends inside the LOD-2 reach). Memory dominated by LOD-0..LOD-2 bands ≈ 665 full-equivalent chunks.
- **8192³ (128 chunks across).** Max chunk-radius 64. LOD-3 ring: 33 < r ≤ 64 → ~129³ − 33³ ≈ 2.1M chunks × (1/512) = ~4,100 full-equivalent. Total: ~4,700 full-equivalent.
- **16384³ (256 chunks across).** LOD-3 + LOD-4 bands populate fully. LOD-3 contributes the same ~4,100; LOD-4 ring (r > 64) ≈ 256³ − 129³ ≈ 14.6M chunks × (1/4096) = ~3,560 full-equivalent. Total: ~8,300.

Per-chunk resident cost depends on chunk content; using `L^1.58` extrapolation as a per-chunk proxy gives roughly 1k reachable nodes for a full LOD-0 64³ chunk. Multiplying:

| Scale | LOD policy | No-LOD |
|---|---:|---:|
| 4096³ | ~640k reachable | 548k (already measured) |
| 8192³ | ~4.7M | ~1.65M (projected) |
| 16384³ | ~8.3M | ~5M (projected) |

**Counter-intuitive result.** At 4096³ the LOD policy *adds* overhead because the proposed bands extend past the world boundary into nothing — the 27-chunk LOD-0 core is not fewer nodes than just-render-it-all when the whole world is small. At 8192³ and 16384³ the math also doesn't show the dramatic savings the §5.1 table suggested would be there.

**Why this is actually fine.** The savings story for memory LOD is not "fewer reachable nodes total" but "**bounded growth**." Without LOD, reachable nodes scale as L^1.58 — the curve keeps climbing as the world grows. With LOD, distant chunks contribute a *bounded constant* per chunk, so total reachable scales linearly with chunks-in-LOD-band, which is L³-bounded but with very small coefficients (1/4096 at LOD-4). The crossover where LOD wins is at 16384³ and beyond, where the L^1.58 curve passes the LOD-flattened curve.

This recasts §5.1's framing: **the LOD policy is not primarily a memory-shrinkage tool at 4096³–8192³.** It is a **structural shaping tool** that enforces bounded per-chunk cost and decouples world-side from reachable-node count. The memory win lands at 16384³+ and the cold-gen win lands at 8192³+ (skipping full detail for cells the player won't see).

This re-derivation is back-of-envelope. The impl bead should measure chunk-occupancy distributions on a representative seeded scene before locking the bands; the table above uses uniform fill, which overstates the LOD-3 / LOD-4 cost (most distant chunks are largely empty sky or uniform stone, both of which dedup hard).

### 5.3 Streaming threshold (revised)

§4.7.4 of the perf paper called 8192³ "plausibly mandatory streaming, intuition not derived." With the LOD policy proposed here:

- **8192³ remains feasible without streaming** because (a) total reachable count grows but stays in the few-million-nodes regime (~450 MB worst case at 96 B/node, several × that with overheads), and (b) cold-gen at 8192³ is the binding consumer, not memory. Memory LOD shaves cold-gen by skipping detail in distant chunks even though it doesn't shave the resident DAG much at this scale.
- **16384³ is where streaming-plus-LOD becomes mandatory.** Cold-gen at this scale is ~58 s (§1.2 projection); without LOD, even gen-time hash-consing (cswp.3) won't bring it under the interactive bar. A streaming layer that loads only the seen chunks becomes necessary, and the LOD policy makes the seen-chunk set's memory bounded.

The proposed LOD layer's value at cswp's 4096³+ target is therefore **bounded growth, not absolute shrinkage**. cswp.5 (chunk eviction) lands its full payoff above 8192³ with this LOD layer in place; below that, the layer's main contribution is keeping the per-chunk cost predictable for the streaming bookkeeping, not slashing the total budget.

---

## 6. Open questions for the impl sub-bead

The implementation sub-bead following cswp.6 needs to resolve:

1. **Chunk size.** 64³, 128³, or 256³? Smaller = finer LOD bands, more bookkeeping; larger = coarser LOD bands, less bookkeeping. Probably want the chunk to be at a power-of-2 octree level for hash-cons reuse, so the candidates are 64³ (level-6 subtree), 128³ (level-7), 256³ (level-8). Recommend starting at 128³ as a middle ground; tune from playtesting.
2. **Representative-material recompute on incoming transitions.** Do we recompute representative material when a chunk gains a level (becomes finer), or do we keep the parent's representative? Proposal: recompute on every level change, since the parent's rep was computed against children that no longer exist. Cost is O(chunk-leaves), bounded.
3. **`lod_bias` knob.** Add a single uniform analogous to render-LOD's `u.debug.y` so playtesting can A/B aggressive vs gentle without recompiling. Suggest values 0.5 (more detail kept), 1.0 (default), 2.0 (more aggressive collapse).
4. **Acceptance bench.** A "fly through 4096³ at velocity X" bench that measures resident memory + frame-time over the trajectory. Acceptance: total resident < 100 MB across the trajectory at 4096³, < 200 MB at 8192³.
5. **Multiplayer interaction.** cswp.6 is single-player. Per-client chunk LOD is fine for camera-anchored streaming; if `hash-thing-3bj` lockstep wants synchronized LODs, we need a separate "what's the canonical chunk LOD set" rule — out of scope here, file when 3bj actually starts.

---

## 7. Out of scope

- **Implementation.** This doc is design only. Implementation lives in a follow-up sub-bead (probably `cswp.7`).
- **Per-material rendering tweaks at distance** (atmospheric haze, fog, height-fog tinting). Separable visual bead if needed; orthogonal to memory-LOD policy.
- **LOD for player-edited regions.** The default policy treats all chunks identically; player-edited chunks may want a "stay at LOD-0 forever" override (the player remembers what they built). Filed as a follow-up consideration, not gating cswp.6.
- **Disk-format LOD storage.** cswp.2 owns the on-disk SVDAG serialization; whether it stores per-chunk LOD pyramids or recomputes-on-load is cswp.2's call. Either model interoperates with the policy here.

---

## 8. Acceptance for cswp.6

Per the bead description:

- ✅ Distance → LOD mapping defined (§2.2).
- ✅ Material-collapse policy defined with one competing alternative (§3, options A–D); default A picked, B flagged for edward.
- ✅ Transition strategy defined with one competing alternative (§4, hard swap vs crossfade).
- ✅ Expected node-budget shrinkage at 4096³ / 8192³ documented (§5).
- Unblocks: a concrete impl sub-bead (`cswp.7` recommended); cswp.5 (chunk eviction) for buffer-sizing input.
