# Perf landscape DSL — naming the regimes we measure

**Status:** draft, edward 2026-04-29 ("I want it to be easier to talk about
the space of where we're at and what's possible").

The problem: every perf claim implicitly carries a giant context — *which*
world, *which* scene, *which* cache state, *which* hardware. A claim like
"step is 6.7 ms" is meaningless without that context, and every conversation
right now is re-deriving it from scratch. This doc proposes a small shared
vocabulary so future claims can read like coordinates, not anecdotes.

Not a settled spec. A starting point to sketch on.

---

## A measurement as coordinates

Every perf number lives at four coordinates. A measurement that doesn't name
all four is a measurement with hidden assumptions.

```
(world, scene, intensity, regime) → metric
```

### 1. `world` — physical scale

The cube the sim runs in. Level + side length.

| name        | level | side  | aliases                               |
|-------------|-------|-------|---------------------------------------|
| `tiny`      | 4     | 16³   | unit-test scale, parity sandbox       |
| `small`     | 6     | 64³   | property-test scale                   |
| `medium`    | 7     | 128³  | thesis-probe scale                    |
| `demo`      | 8     | 256³  | the live demo default                 |
| `large`     | 10    | 1024³ | post-edge-prefetch grow regime        |
| `huge`      | 12    | 4096³ | cswp epic target — streaming required |
| `pathological` | —  | —     | adversarial / synthetic worst-case    |

### 2. `scene` — what kind of world content

Different scenes hit different parts of the engine. A measurement is bound
to one of these (or you say "any" if the scene doesn't matter).

| name              | description                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------|
| `empty`           | Air everywhere. Tests the empty short-circuit. Memo doesn't matter.                                  |
| `uniform`         | One material everywhere (e.g. all stone). Tests the inert short-circuit.                             |
| `lattice`         | Sparse identical sub-cubes (e.g. 1 stone every 8³). Tests hash-cons folding.                         |
| `default-terrain` | The demo's terrain seed (heightmap + materials). Mostly stable post-warmup.                          |
| `default-demo`    | Default terrain + volcano + water sheet + critters. The actual demo.                                 |
| `random-mix`      | Synthetic random material per cell. Tests upper levels of cache pressure.                            |

### 3. `intensity` — how much is changing

The amount of fresh content the sim has to produce per step. Three named
levels; "any" if not relevant.

| name       | what it means                                              | example                              |
|------------|------------------------------------------------------------|--------------------------------------|
| `idle`     | No edits, world fully settled                              | demo at gen 200 with no input        |
| `microchurn` | Small synthetic edit rate (~10-30 cells/step)            | `bench_hashlife_256_churn_short`     |
| `cascade`  | Large-volume fluid / particle propagation per step        | demo's water sheet hitting terrain   |
| `edit-active` | User input mid-play (placing/breaking)                  | real demo session                    |
| `adversarial` | Inputs designed to break hashlife (every-cell-different)  | research only                        |

### 4. `regime` — what the cache looks like

Cache state at the time of measurement. Step latency is wildly different
across these.

| name        | what it means                                                              |
|-------------|----------------------------------------------------------------------------|
| `cold`      | First few generations. Cache empty. Compute dominates.                     |
| `warming`   | Cache filling. memo_hit climbing each step.                                |
| `saturated` | Cache at steady state for the world+scene. Hit rate stable.                |
| `churning`  | Cache full but eviction-heavy. Each step produces fresh unique sub-cubes. |
| `compacted` | Just after `maybe_compact()`. Fresh cache shape, may take a few gens to re-warm. |

---

## Headline metrics, in priority order

Numbers that mean something without further context, given the coordinates above.

| metric              | what it answers                                          | reported as                  |
|---------------------|----------------------------------------------------------|------------------------------|
| `frame_total_p95`   | "Will the demo feel laggy?" (1/p95 = worst-case Hz)      | ms (lower is better)         |
| `step_p95`          | "Is the sim keeping up?"                                 | ms (lower is better)         |
| `elision_factor`    | "Is hashlife actually buying anything?"                  | × multiplier (higher better) |
| `memo_hit`          | "How much of the cache is reused?" (post-short-circuit)  | 0.0–1.0                      |
| `step_recursive_p95`| "Worst-case sim work this step"                          | ms                           |

**Single number for thesis verification:** `elision_factor` at the
**busiest gameplay regime that's part of the demo**. Today that's
`(world=demo, scene=default-demo, intensity=cascade, regime=churning)`.
Currently measured at **5.6×** there, vs hundreds-× at saturated-idle.
The thesis is "viable" if this stays >>1 in gameplay regimes; it's "weak"
if it collapses to single-digit when the player actually does something.

**Single number for demo viability:** `frame_total_p95` at the busiest
gameplay regime, with target ≤ 16.6 ms (60 Hz) or ≤ 33.3 ms (30 Hz).

---

## What I've actually been measuring (audit)

Honest inventory of the numbers I've quoted this session, mapped to
coordinates:

| claim                         | (world, scene, intensity, regime)                    | metric           | value      | confidence  |
|-------------------------------|------------------------------------------------------|------------------|------------|-------------|
| "elision 46×, thesis viable"  | (medium, default-terrain, idle, saturated)           | elision_factor   | 46×        | bench       |
| "elision 16× under churn"     | (medium, default-terrain, microchurn, saturated)     | elision_factor   | 16×        | bench       |
| "BFS 1.5× faster than ftuu"   | (demo, default-terrain, microchurn, saturated)       | step_us median   | 3.7 ms     | bench       |
| "step 6.7 ms median post-ite4"| (demo, default-terrain, microchurn, saturated, full)| step_us median   | 6.7 ms     | bench       |
| "step 36 ms / 67 p95"         | (demo, default-demo, cascade, churning)              | step / step_p95  | 36 / 67 ms | demo, n=1   |
| "elision 5.6× at cascade peak"| (demo, default-demo, cascade, churning)              | elision_factor   | 5.6×       | demo, n=1   |
| "memo_hit 0.41 → 0.72"        | (demo, default-terrain, ?, saturated)                | memo_hit         | varies     | mixed       |

The cherry-pick problem is now visible: the rosier numbers were all from
`(microchurn, saturated)` regimes; the demo's interesting regime
(`cascade, churning`) is 5-10× worse on every metric. Future claims should
say which coordinate they're at.

---

## Thesis sub-claims, in this DSL

The thesis "spatial sim / hashlife as a force-multiplier engine that
enables novel games" decomposes into:

| sub-claim     | what it means in this DSL                                                                          | status         |
|---------------|----------------------------------------------------------------------------------------------------|----------------|
| **engine**    | `elision_factor ≥ 10×` for all `(demo, default-demo, *, *)` regimes                                | partial — fails at cascade peak (5.6×) |
| **interactive** | `frame_total_p95 ≤ 33 ms` for all `(demo, default-demo, edit-active, *)`                         | demo says no — render-bound at 50 ms |
| **scale**     | `step_p95 ≤ 100 ms` extends to `(large, *, edit-active, *)` and ≤ 200 ms to `(huge, ...)`          | unknown        |
| **novelty**   | A gameplay capability exists at `(demo, default-demo, edit-active, *)` that a chunk-array sim can't deliver | unmeasured |

When we say "the thesis is viable," we mean *all four* are true at the
relevant coordinates. Today: engine is partial, interactive is failing on
render not sim, scale is unmeasured, novelty is the gap. Saying "viable"
without naming which sub-claim feels uncomfortably loose now.

---

## Concrete next steps if we adopt this

1. **`bench_perf_landscape`** — one bench that runs every named regime once
   and emits a single table. Replaces ad-hoc per-bench claims with a single
   source-of-truth output. Land in `tests/bench_perf_landscape.rs`. Each
   scenario is a tuple of (world, scene, intensity, regime) → metrics.

2. **`memo_summary` token discipline.** Each token gets a doc line saying
   which regime it's representative for. The new `memo_elision` token is
   most useful at `(saturated, churning)` — at `cold` it's meaningless.

3. **Bead tagging.** Perf beads adopt a short tag in their description
   header naming the coordinates they're targeting:
   `regime: (demo, default-demo, cascade, churning)`. Makes "is this bead
   targeting the right regime" answerable at sweep time without re-reading
   the description.

4. **`docs/perf/regimes.md`** (this file) is the living taxonomy. PR'd
   updates as we learn what regimes matter and which sub-claims are
   answered.

5. **Frame-budget table** (separate followup): a small grid of (world,
   scene, target-fps) → required (step + render) ms budgets. So when we
   say "frame_total_p95 = 50 ms" we know exactly which budget we're
   missing and by how much.

---

## Open questions (for edward)

- Are these the right archetypes, or do you have other named regimes in
  mind?
- Is there a 6th coordinate I'm missing? (Maybe `hardware`, but that's
  arguably a separate dimension that multiplies into `metric`.)
- What's the right home for the perf-narrative — this file, or the bd
  description of an epic, or a Notion-style canvas, or the SPEC.md?
- Should the `intensity` axis be more granular? `microchurn / cascade /
  edit-active` lumps things that may behave very differently.

---

## Why this matters

Right now: every conversation about perf re-derives "is hashlife working"
from scratch with whatever numbers happen to be top-of-mind. That's
expensive and produces inconsistent answers (this session: 6.7 ms / 15 ms
/ 36 ms / 67 ms all quoted depending on which bench was open). With a
shared coordinate system the conversation collapses to "is the
$(\text{regime}, \text{metric})$ pair on target?" — and disagreement
becomes a disagreement about which coordinates are the gameplay-load-bearing
ones, which is a productive disagreement.
