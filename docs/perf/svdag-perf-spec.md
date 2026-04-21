# SVDAG Performance Spec (Re-derived Intuition)

**Last re-derived:** 2026-04-20 (skeleton — no derivation yet)
**Source of truth:** `docs/perf/svdag-perf-paper.md`. If this spec disagrees with the paper, the paper wins.
**Ownership:** the crew, with mayor as the surfacing/coordination layer. See the paper's ownership note for details.

This document is the **short form**. It exists to be referenced from code review, bead descriptions, and other docs without forcing readers through the full paper. Re-derived periodically — never edited in isolation.

---

## Reference hardware

**M1 MacBook Air, 8 GB unified memory, integrated GPU. 60 fps target.**

If you are tuning to a different machine (M2, M3, discrete GPU), you are tuning to the wrong machine.

---

## Budgets

(To be derived from paper §3.)

- **Frame budget:** 16.67 ms / frame total.
- **Raycast pass:** TODO ms.
- **Step pass:** TODO ms.
- **SVDAG sync (build + compact, amortized):** TODO ms.
- **Surface acquire + present + HUD:** TODO ms.
- **Memory budget for SVDAG nodes:** TODO MB out of 8 GB total.

---

## Rules of thumb

(To be derived from paper §3 and §4.)

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
