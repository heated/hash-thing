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

Every perf number lives at **4 headline coordinates plus 6 mandatory metadata fields** (v2, hash-thing-8ppq.9, 2026-05-02). The 4 headline coordinates are the human-readable shorthand citation; the 6 metadata fields are filled in by the structured record (see [`perf-measurement-schema.md`](./perf-measurement-schema.md) for record shape).

```
(world · scene · intensity · regime) → metric
   + rule_set, backend, hardware, scenario_hash, confidence, schema_version
```

A measurement that doesn't name the 4 headline coordinates is a measurement with hidden assumptions. A measurement that doesn't carry the 6 metadata fields can't be reproduced or compared.

### 1. `world` — physical scale

The cube the sim runs in. Level + side length.

| name        | level | side  | aliases                               |
|-------------|-------|-------|---------------------------------------|
| `tiny`      | 4-5   | 16³-32³ | unit-test / parity-sandbox scale (8ppq.1.1's 32³ MVP comparator lives here) |
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
| `default-demo`    | Default terrain + volcano + water sheet + critters. The current demo.                                |
| `quarantine-atlas`| Quarantine Atlas playtest scene: hazard lane, settlements, and budgeted counter-pattern stamps.       |
| `random-mix`      | Synthetic random material per cell. Tests upper levels of cache pressure.                            |
| `factory-conveyor`| Conveyor-rule scene. `FactoryConveyorRuleV1` tests one-material fixed-direction transport; `FactoryEncodedBeltRoutingV1` tests separate belt substrate, encoded direction, turns, merge pressure, and route-specific throughput telemetry. |
| `puzzle-circuit`  | Wire/signal-propagation scene. Tests sparse-pattern memo + signal cascades.                          |
| `replay-scrub`    | Recorded sim trace replayed at variable speed. Tests temporal cache hits at past timesteps. (5j7e family.) |
| `soup-search`     | Many random initial seeds run in parallel for N gens, classifier picks survivors. (8ppq.5 family.)   |
| `megastructure-stamp` | Pre-computed module stamped at K positions in a huge world. Tests hash-cons folding at scale.    |
| `graph-cellular`  | Non-cube topology (hex / arbitrary graph) with CA on the graph. Tests subgraph-memo hypothesis. (ltt5 family.) |

### 3. `intensity` — how much is changing

The amount of fresh content the sim has to produce per step. Three named
levels; "any" if not relevant.

| name       | what it means                                              | example                              |
|------------|------------------------------------------------------------|--------------------------------------|
| `idle`     | No edits, no emitters, world fully settled                 | empty world at gen 100               |
| `passive-active` | No user input but world has emitters (volcano/water/critters) generating fresh content | demo "just walking around" — edward's 2026-04-30 observation |
| `microchurn` | Small synthetic edit rate (~10-30 cells/step)            | `bench_hashlife_256_churn_short`     |
| `cascade`  | Large-volume fluid / particle propagation per step        | demo's water sheet hitting terrain   |
| `edit-active` | User input mid-play (placing/breaking)                  | real demo session                    |
| `adversarial` | Inputs designed to break hashlife (every-cell-different)  | research only                        |

Scenario runners may also carry a setup identity outside the four headline
coordinates. Examples include `QuarantineAtlasMixedContainmentV1` for the
deterministic six-stamp mixed firebreak/cooling/barrier plan,
`FactoryEncodedBeltRoutingV1` for the routed belt substrate harness,
`TemporalReuseV1` for the replay-scrub recurrence probe, and
`MegastructureStamp10V1` / `MegastructureStamp100V1` for module-stamp count.
Keep this in `setup`, not `intensity`; the intensity still describes the
measured dynamics.

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
| `n/a`       | Backend has no memo cache (e.g. `chunk-array`). Required for non-memoized backends. |

**Backend conditioning:** `regime` is *defined in terms of the hashlife memo cache*. For `backend = chunk-array` (or any future non-memoized backend), `regime` MUST be `n/a` — `(backend=chunk-array, regime=churning)` is a category error. For `backend = hashlife-recursive`, `regime` takes one of the 5 cache-state values above. See [`perf-measurement-schema.md`](./perf-measurement-schema.md) for the per-backend constraint table.

---

## Headline metrics, in priority order

Numbers that mean something without further context, given the coordinates above.

| metric              | what it answers                                          | reported as                  |
|---------------------|----------------------------------------------------------|------------------------------|
| `frame_total_p95`   | "Will the demo feel laggy?" (1/p95 = worst-case Hz)      | ms (lower is better)         |
| `step_p95`          | "Is the sim keeping up?"                                 | ms (lower is better)         |
| `work_elision_factor` | "Is hashlife actually buying anything?"                | × multiplier (higher better) |
| `memo_hit`          | "How much of the cache is reused?" (post-short-circuit)  | 0.0–1.0                      |
| `step_recursive_p95`| "Worst-case sim work this step"                          | ms                           |

`work_elision_factor` is the leaf-work metric from `memo_summary()` /
`work_elision_*` JSON metrics: padded active-leaf nodes divided by active-leaf
misses. It is distinct from the legacy JSON metric `elision_factor_x`, which is
a cache-lookup ratio retained for backward compatibility.

**Single number for thesis verification:** `work_elision_p05_x` at the
**busiest gameplay regime that's part of the demo**. Today that's
`(world=demo, scene=default-demo, intensity=cascade, regime=churning)`.
The historical field reading at that coordinate was **5.6×**, but it pre-dated
the structured `work_elision_*` JSON metric. The first exact-coordinate
structured run (`hash-thing-8ppq.3`) measured `work_elision_p05_x=79.15×`.
The thesis is "viable" if this stays >>1 in gameplay regimes; it's "weak"
if it collapses to single-digit when the player actually does something.

**Single number for demo viability:** `frame_total_p95` at the busiest
gameplay regime, with target ≤ 16.6 ms (60 Hz) or ≤ 33.3 ms (30 Hz).

---

## What I've actually been measuring (audit, v2-shape)

Honest inventory of historical numbers, retrofitted to v2 coordinates with hardware archeology (every existing row was on edward's M2-class MBP). v1 rows pre-date the `scenario_hash` discipline so that field reads `unknown`; future rows are required to fill it. The `cherry_pick_audit` column self-discloses the regime the number came from.

| claim                         | (world · scene · intensity · regime)                 | backend             | rule_set    | hardware     | metric           | value      | source       | cherry_pick_audit |
|-------------------------------|------------------------------------------------------|---------------------|-------------|--------------|------------------|------------|--------------|-------------------|
| "elision 46×, thesis viable"  | medium · default-terrain · idle · saturated          | hashlife-recursive  | default-ca  | m2-pro-mbp   | legacy work-elision | 46×     | bench        | easy_only (8ppq.1.4) |
| "elision 16× under churn"     | medium · default-terrain · microchurn · saturated    | hashlife-recursive  | default-ca  | m2-pro-mbp   | legacy work-elision | 16×     | bench        | mixed             |
| "BFS 1.5× faster than ftuu"   | demo · default-terrain · microchurn · saturated      | hashlife-recursive  | default-ca  | m2-pro-mbp   | step_median_ms   | 3.7        | bench        | mixed             |
| "step 6.7 ms median post-ite4"| demo · default-terrain · microchurn · saturated      | hashlife-recursive  | default-ca  | m2-pro-mbp   | step_median_ms   | 6.7        | bench        | mixed             |
| "step 36 ms / 67 p95"         | demo · default-demo · cascade · churning             | hashlife-recursive  | default-ca  | m2-pro-mbp   | step_p95_ms      | 67         | demo (n=1)   | hard_included     |
| "elision 5.6× at cascade peak"| demo · default-demo · cascade · churning             | hashlife-recursive  | default-ca  | m2-pro-mbp   | legacy field reading | 5.6×    | demo (n=1)   | hard_included     |
| "work elision p05=79.15×"     | demo · default-demo · cascade · churning             | hashlife-recursive  | default-ca  | m2-pro-mbp   | work_elision_p05_x | 79.15×   | bench (n=30) | hard_included     |
| "hashlife p95=20.40ms"        | demo · default-demo · cascade · churning             | hashlife-recursive  | default-ca  | m2-pro-mbp   | step_p95_ms      | 20.40      | bench (n=30) | hard_included     |
| "chunk-array p95=939.17ms"    | demo · default-demo · cascade · n/a                  | chunk-array         | default-ca  | m2-pro-mbp   | step_p95_ms      | 939.17     | bench (n=30) | hard_included     |
| "memo_hit 0.41 → 0.72"        | demo · default-terrain · unknown · saturated         | hashlife-recursive  | default-ca  | m2-pro-mbp   | memo_hit_ratio   | 0.41-0.72  | mixed        | mixed             |
| "chunk-array p95=2.29ms"      | tiny (l=5) · default-terrain · idle · n/a            | chunk-array         | default-ca  | m2-pro-mbp   | step_p95_ms      | 2.29       | bench (n=30) | easy_only (8ppq.1.4) |
| "hashlife p95=1.20ms"         | tiny (l=5) · default-terrain · idle · saturated      | hashlife-recursive  | default-ca  | m2-pro-mbp   | step_p95_ms      | 1.20       | bench (n=30) | easy_only (8ppq.1.4) |

The cherry-pick problem is now structurally visible: the rosier numbers are flagged `easy_only` and carry their `hard_followup_bead` (8ppq.1.4 cascade-regime measurement); the cascade-peak rows are `hard_included`. New claims that flag `easy_only` MUST cite a follow-up bead per the schema's `confidence.cherry_pick_audit` constraint — see [`perf-measurement-schema.md`](./perf-measurement-schema.md).

---

## Thesis sub-claims, in this DSL

The thesis "spatial sim / hashlife as a force-multiplier engine that
enables novel games" decomposes into:

| sub-claim     | what it means in this DSL                                                                          | status         |
|---------------|----------------------------------------------------------------------------------------------------|----------------|
| **engine**    | `work_elision_p05_x ≥ 10×` for all `(demo, default-demo, *, *)` regimes                            | passes known cascade case — 79.15× at `demo · default-demo · cascade · churning` |
| **interactive** | `frame_total_p95 ≤ 33 ms` for all `(demo, default-demo, edit-active, *)`                         | demo says no — render-bound at 50 ms |
| **scale**     | `step_p95 ≤ 100 ms` extends to `(large, *, edit-active, *)` and ≤ 200 ms to `(huge, ...)`          | unknown        |
| **novelty**   | A gameplay capability exists at `(demo, default-demo, edit-active, *)` that a chunk-array sim can't deliver | unmeasured |

When we say "the thesis is viable," we mean *all four* are true at the
relevant coordinates. Today: engine is partial, interactive is failing on
render not sim, scale is unmeasured, novelty is the gap. Saying "viable"
without naming which sub-claim feels uncomfortably loose now.

---

## Concrete next steps if we adopt this

0. **Demo perf trail** — the demo binary appends heuristic snapshots to
   `.ship-notes/demo-perf-trail.jsonl` every wall-clock perf-log tick
   (`hash-thing-x7dl`). These records are `source=demo` and keep raw
   `perf_summary` / `memo_summary` strings so later tooling can reclassify
   rough `intensity` and `regime` guesses. Set
   `HASH_THING_DEMO_PERF_TRAIL=0` to disable, or
   `HASH_THING_DEMO_PERF_TRAIL_PATH=/path/to/file.jsonl` to redirect.

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
- What's the right home for the perf-narrative — this file, or the bd
  description of an epic, or a Notion-style canvas, or the SPEC.md?
- Should the `intensity` axis be more granular? `microchurn / cascade /
  edit-active` lumps things that may behave very differently.

(v2 resolved the v1 "is hardware a 6th coordinate?" question: yes — but as a *metadata field*, not a headline coordinate. See the 6 metadata fields above.)

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
