# SVDAG Research Log (Living Document)

**Ownership:** mayor owns project-level movement (what threads stay live, when an entry graduates, when a thread closes); crew owns execution (runs experiments, writes entries, promotes findings).

**Status:** Skeleton. Threads seeded, no dated experiments yet — first entries land as sub-beads of the threads below.

**Companion docs.**
- `docs/perf/svdag-perf-paper.md` — the settled-model long form. Budgets, derived frame-cost model, literature reconciliation. Slow to update; "what should be possible on the reference hardware."
- `docs/perf/svdag-perf-spec.md` — short re-derived intuition. Regenerated from the paper, not this log.
- This file — research-agenda counterpart. Hypotheses, experiments, refutations, open frontiers. Moves with measurements, not with the wind.

---

## 0. How to read this document

### What this is

A running log of the open research questions we are actually exploring through this codebase, the experiments we run against them, and what we conclude. The intersection we work in — hashlife-class macro-compression × SVDAG × per-tick CA mutation × game-design-coupled performance — has sparse published precedent. That sparseness is not marketing; §1 lays out the specific gaps we think we sit in.

The paper (`svdag-perf-paper.md`) asks: *what does M1 MBA admit in principle, and how far are we from it?* This log asks: *what are we learning that nobody has written down yet?*

### Entry discipline

- **No entry without numbers or a refutation.** "We tried X and it felt faster" is not an entry. "We tried X, phase1 went 68.3ms → 24.3ms on the 64³ water bench, implication: 44ms of phase1 is per-cell dispatch body" is.
- **Dated.** Every experiment log entry has a date. Ongoing threads carry a "last moved" date so stale threads surface.
- **Hypothesis / Method / Results / Implication / Open.** Five-part shape for experiment entries. Brevity over completeness — if the real writeup lives in a bead close-out comment or a scratch file, link to it and summarize.
- **Bead-first.** Entries reference bead IDs (`hash-thing-xxxx`). The bead is canonical; this log is the cross-cutting narrative.

### When an entry graduates

An experiment-log entry **graduates** to the perf paper when:

1. The finding has landed (numbers stable across a re-run, not a single-session artifact).
2. It changes what we believe about what M1 MBA admits, or what our architecture can do.
3. A model or budget digest can be written from it (i.e., it would survive being re-derived in the paper's style).

Graduated entries move to §4 (Frontiers reached) here and get written up in the paper. The research-log retains the stub + pointer; the paper carries the reconciled model.

### When a thread closes

A thread closes when:

- The question is answered (positive: "yes, GPU memo works, here is the architecture"; negative: "no, GPU memo is bandwidth-bound, here is why").
- Or the question is abandoned (circumstances changed, the surface is gone). Say so explicitly; don't just let it rot.

Closed threads stay in §4 for historical record.

---

## 1. The claim — what is novel here

We sit at an intersection that the published literature does not cover well. The claim is not that any single component is new — hashlife, SVDAGs, CA on GPU, cache-aware scene design all exist in prior art — but that **the combination has not been worked through**, and the combination behaves differently than the sum of its parts.

### 1.1 No published GPU-hashlife

Hashlife (Gosper 1984) is CPU-only in the literature. The recursive memoized macro-step is the whole technique, and every published implementation we can find runs it on a single thread. Published GPU cellular-automata work (Lefebvre et al. and successors) does Life-class rules at per-cell parallelism with no pow2-step compression: they get the SIMD win but not the macro-step win.

Our CPU `step_recursive_pow2` + nascent GPU memo exploration (`hash-thing-abwm`) sit in the unclaimed gap. If the spike succeeds, it is novel ground. If it fails for a principled reason (GPU hash-table throughput bounds macro-step dispatch to below CPU throughput at our scales), *that refutation is also novel ground* — nobody has written it down.

### 1.2 No SVDAG with per-tick mutation in the literature

Kämpe et al. (2013) introduced SVDAGs as a **build-once, trace-static** structure — the DAG is computed offline and rendered unchanged. HashDAG (Careil et al., 2020) added **user-edit-per-frame** mutation for authoring tools: sparse diffs flow into the GPU-resident DAG. Neither handles the case we have: **the entire voxel field advances by a CA rule every tick**, and the DAG must follow without full rebuild.

Our per-tick CA-rule mutation + hash-consed upload sits distinct from both. The paper already notes this in §4.6; this log is where the cost-model and churn-model experiments accumulate.

### 1.3 Spatial memo as a game-design knob

The L2-fit / performance-local-optima principle (`SPEC.md` §Core goals, 2026-04-20) says **game-design choices are first-class performance knobs**, not just a downstream constraint. Rain density, material variety, scene structure, progression layout — all of these modulate the dedup rate, the reachable-DAG size, the memo-hit rate, and therefore what frame budget admits what feature.

The existing literature treats performance and content separately: the engine provides a budget, the designer fits inside it. Here they are coupled. That coupling is a design stance we think is honest — worth experiments that either validate it or refute it. See §2 threads.

### 1.4 Margolus-in-hashlife (open theory question)

`hash-thing-4497` asks: can macro-pow2 hashlife composes with 2×2×2 Margolus block rules? The phase shift between recursion levels is not obviously compatible with Gosper's step-doubling. We do not know of prior art that has answered this. It is a genuine open theory question, not a debugging task.

---

## 2. Ongoing threads

Live research questions. Each has a one-liner, current status, "last moved" date, and the beads that feed it. New entries come in via experiment-log (§3); promotions out go to §4.

### 2.1 GPU spatial memo

**Question.** Can the spatial memo (currently a CPU `FxHashMap<(NodeId, parity), NodeId>`) run on GPU, and if so at what throughput and at what cost in hash-table contention?

**Why it matters.** Today `step_recursive` is one CPU core, ~1M memo misses/sec. At 4096³ with realistic churn, the CPU memo is the bottleneck well before 68 GB/s bandwidth or GPU compute. Moving memo + store + rule dispatch to GPU would potentially collapse the entire sim tick to a single dispatch.

**Status (2026-04-23).** Spike phased: `abwm.1` (architecture sketch) closed, `abwm.2` (GPU hash-table microbenchmark) closed, `abwm.3` (breadth-first level dispatcher PoC) open at P3. No dated entries in §3 yet — pending `abwm.3` or a standalone experiment.

**Feeds from.** `hash-thing-abwm`, `hash-thing-abwm.3`.

**Feeds to.** Paper §8 (GPU spatial memo feasibility) — already seeded there with the architecture sketch.

### 2.2 Macro-pow2 hashlife × Margolus block rules

**Question.** Does Gosper's pow2 step-doubling compose with 2×2×2 Margolus partitions, and if not, is there a structured reduction from "block rule + pow2 step" to a pure-CA formulation?

**Why it matters.** If macro-step hashlife cannot compose with Margolus, any material with block-rule semantics (water, sand — the entire falling-sand class) is locked to step=1 and loses the hashlife compression. That is the majority of our interesting dynamics.

**Status (2026-04-23).** Open theory question. `hash-thing-4497` is the root bead. No experiments yet; likely needs a formal framing pass before a numerical experiment makes sense.

**Feeds from.** `hash-thing-4497`.

**Feeds to.** Paper §4 (SVDAG ↔ memo-step interaction) and §6 (open questions) if we reach a reduction.

### 2.3 Spatial memo scaling fit

**Question.** How does memo-hit rate scale with world size, active-cell density, rule mix, and material variety? What is the closed-form (if any)?

**Why it matters.** Budget-gating decisions (how much churn admits 60 fps, what scene shape stays in L2) depend on the memo model. Characterizing the curve is what lets us make "this scene is too expensive, change the design" an honest call.

**Status (2026-04-23).** Partially characterized: `hash-thing-slc1 / heyw / 4hkq / e092 / cz0r` measured scaling on several scenes. Data exists; consolidation pass not yet run. Good candidate for the first dated §3 entry.

**Feeds from.** `hash-thing-slc1`, `hash-thing-heyw`, `hash-thing-4hkq`, `hash-thing-e092`, `hash-thing-cz0r`.

**Feeds to.** Paper §3 (frame-cost model) and §4.

### 2.4 Game-design-tuned performance local optima

**Question.** Can we demonstrate — with numbers, not just rhetoric — that a scene deliberately tuned for L2-fit (or a cousin optimum at a larger scale) outperforms an arbitrarily-structured scene at equivalent "content density"? What does "equivalent content density" even mean operationally?

**Why it matters.** The SPEC.md 2026-04-20 principle is a claim, not a result. We have the intuition (dedup rate, reachable-DAG size), the infrastructure (bench harness, material knobs), and the motive (4096³ with streaming needs *some* optimum to aim for). We do not yet have a paired A/B measurement that pins it down.

**Status (2026-04-23).** No experiments yet. Needs someone to construct the A/B pair (e.g., uniform-random-grid vs. structurally-repetitive scene, same voxel count, same rule mix) and measure render_gpu + upload_cpu + step over a fixed sim horizon. Highest-leverage first §3 entry — this is the claim the paper has not yet numerically defended.

**Feeds from.** SPEC.md §Core goals (2026-04-20 principle).

**Feeds to.** Paper §3.5 (L2-optimum math) and §7 (limitations — what we can't do without an optimum to aim for).

### 2.5 SVDAG live-reachable set drift under sustained sim

**Question.** When a scene sims for a long time, how does the live-reachable DAG shape evolve? Does it converge, drift monotonically, or oscillate? What is the steady-state node count as a function of initial scene + rule?

**Why it matters.** Memo churn and upload cost both scale with live-set size, and our current `retain_reachable` cadence is heuristic. A characterization of drift gives us a principled eviction policy.

**Status (2026-04-23).** No dedicated experiments. Implicit in every perf bench that runs > 100 generations; we have the raw data but no thread pulling it together.

**Feeds from.** Perf paper §3 (discusses reachable set abstractly).

**Feeds to.** Paper §4 and the eventual streaming/eviction policy (`hash-thing-cswp`).

### 2.6 Per-tick SVDAG upload cost model

**Question.** Closed-form cost of `build_svdag_for_world` + GPU upload per tick, as a function of (changed-node count, total-live-node count, buffer-watermark drift).

**Why it matters.** dlse.2 work already established upload is a first-class frame cost. A model lets `render_gpu` and `upload_cpu` budgets actually share a denominator — today they are separately measured but not separately modelled.

**Status (2026-04-23).** `hash-thing-dlse.2.2.x` family ran the per-tick measurements; paper §3.8 has the first sketch. No convergent model yet.

**Feeds from.** `hash-thing-dlse.2.2.x`, paper §3.8.

**Feeds to.** Paper §3 (frame-cost model), §4.6 (per-tick upload).

---

## 3. Experiment log

Dated entries. Shape: **Hypothesis / Method / Results / Implication / Open.** Linked bead is canonical.

*No entries yet. First entries land as sub-beads of §2 threads.*

Entry template (copy, don't edit):

```
### 3.N — YYYY-MM-DD — <one-line title> (<thread §2.x>, <bead>)

- **Hypothesis.**
- **Method.**
- **Results.** (numbers; world size; profile; bench name if reproducible)
- **Implication.**
- **Open.** (what this did not settle)
```

---

## 4. Frontiers reached

Closed questions and what we concluded. Entries move here from §3 when they (a) survive a re-run, (b) change what we believe, (c) can be re-derived in paper style. Each entry keeps a stub + pointer to the paper section that now carries the reconciled model.

*No entries yet.*

---

## 5. Graduation backlog

§3 entries that look ready to become a written-up paper section but have not yet been written up. Light-weight queue — "this is mature, write it up when someone has a session for it." Not gated on review tier; the writeup lives in the paper and runs through the paper's own revision cadence.

*No entries yet.*

---

## 6. Revision log

- **2026-04-23** — Skeleton landed (`hash-thing-ngos`). Threads §2.1–§2.6 seeded; no dated experiments.
