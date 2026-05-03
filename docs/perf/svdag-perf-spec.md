# SVDAG Performance Spec (Re-derived Intuition)

**Last re-derived:** 2026-05-03 (`hash-thing-63gv` literature reconciliation)
**Source of truth:** `docs/perf/svdag-perf-paper.md`. If this spec disagrees with the paper, the paper wins.
**Ownership:** see the paper's ownership note (and the `mayor` Claude skill).

This document is the **short form**. It exists to be referenced from code review, bead descriptions, and other docs without forcing readers through the full paper. Re-derived periodically — never edited in isolation.

---

## Reference hardware

**M1 MacBook Air, 8 GB unified memory, integrated GPU. 60 fps target.**

If you are tuning to a different machine (M2, M3, discrete GPU), you are tuning to the wrong machine.

---

## Budgets

(To be derived from paper §3.)

- **Frame budget:** 16.67 ms / frame total.
- **Raycast pass:** sub-ms on reference-class primary rays. Measured M2 rows are `render_gpu` 0.10-0.30 ms mean across 256³-4096³; M1-implied values in the paper are roughly 1.5× those rows, still sub-ms for primary-ray traversal. Paper-scaled ESVO/SVDAG expectations are now conservative, not missing-optimization evidence.
- **Step pass:** TODO ms.
- **SVDAG sync (build + compact, amortized):** TODO ms.
- **Surface acquire + present + HUD:** TODO ms.
- **Memory budget for SVDAG nodes:** TODO MB out of 8 GB total.

---

## Rules of thumb

(To be derived from paper §3 and §4.)

- **Aim for a performance local optimum, not the middle.** L2-fit (<~4 MB reachable DAG, ~100k interior nodes) is one such optimum on M1 MBA: bandwidth-free raycast, 3–5× faster than the DRAM regime. It is not the only one — at 4096³ with streaming, the relevant optimum is "streaming-bandwidth-matched active region + high dedup rate." The design principle is to pick an optimum at the current scale and build toward it deliberately, rather than drifting into the space between optima where we pay DRAM costs without a corresponding visual payoff. This is a **game-design input**, not just a tuning target. See SPEC.md "Soft requirements" (2026-04-20 entry) and paper §3.5.
- **Do not file shader-opt work from paper-scaled ESVO/SVDAG gaps alone.** The 2026-05-03 primary-source pass says our measured M2 primary-ray traversal is already faster than bandwidth-scaled ESVO/Kämpe expectations; sparse-64 integrated-GPU numbers are the closest qualitative match. Re-measure first, then optimize only if a real `render_gpu_lag <= 1` or fence-polled metric moves.
- TODO — e.g., "Doubling world linear scale costs roughly Nx in raycast time because traversal depth grows by 1 bit and average reachable set grows by ~2x; so 512³ should cost ~2x of 256³, not ~8x."
- TODO — e.g., "Active-material churn above N cells/generation defeats SVDAG compaction; at that point a flat 3D texture is the right structure."
- TODO — e.g., "Cross-frame texture dependencies on integrated GPU cost more than they look like they should; prefer ping-pong over single-target."

---

## Known limits

(To be filled from paper §7.)

Empty until we have argued something through to a confident "no."

---

## How to use this spec

- **Code review:** if a change is in the SVDAG, raycast, or memo-step paths, check it against the budgets and rules. Flag deviations.
- **New perf bead:** before filing, check whether the spec already says what's possible or impossible here. If the bead is "make X faster" and the spec says X is already at limit, escalate to the paper instead of opening a perf bead.
- **Bench results:** compare against the budget, not against yesterday's number. Beating yesterday by 10% while still 5× over budget is not a win.

---

## Revision log

| Date | Re-derived from paper revision | Change |
|---|---|---|
| 2026-04-20 | (skeleton, no paper revision yet) | Initial skeleton. |
| 2026-05-03 | §2.7 (`hash-thing-63gv`) | Literature reconciliation: no missing 3-5× raycast optimization; render/user-visible bottleneck remains surface acquire/render scale, while game-loop bottleneck is sim step. |
