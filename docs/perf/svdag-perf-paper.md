# SVDAG Performance Paper (Living Document)

**Owner:** mayor seat. Singular. Other seats may comment via bead `hash-thing-stue` or open follow-up beads; direct edits to this document by non-mayor seats should be rare and noted in the revision log.

**Status:** Skeleton. First real revision pending (see bead `hash-thing-stue` children).

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

The SVDAG line of work this codebase descends from:

- **Kämpe, Sintorn, Assarsson (2013), "High Resolution Sparse Voxel DAGs"** — the founding paper. Establishes content-deduplicated octree of voxel materials, GPU traversal pattern, expected compression ratios.
  - TODO: extract their reported traversal cost per ray, and on what hardware.
- **Villanueva, Marton, Gobbetti (2016/2017), "Symmetry-aware Sparse Voxel DAGs"** — extends dedup across reflections/rotations of subtrees. Significant memory wins, traversal cost mostly unchanged.
  - TODO: relevance to our per-cell tagged encoding.
- **Dolonius, Sintorn, Assarsson (2017+), attribute compression for SVDAGs** — separates topology from materials, allows per-attribute compression schemes.
  - TODO: how this maps onto our material-tagged-cell layout and whether it suggests an architectural change.
- **Williams et al., "An Efficient and Robust Ray–Box Intersection Algorithm"** (2005) and follow-ups — the slab test underlying any voxel raycast. Cost floor for traversal.
- TODO: more recent work — anything 2020+ that updates the picture, especially on integrated GPUs.

For each paper, the paper section will eventually capture:

1. Core idea in one paragraph.
2. The performance number they report and on what hardware.
3. The number their technique implies on **our** reference hardware (M1 MBA), accounting for clock/bandwidth/cache differences.
4. Whether their technique is something we already do, could adopt, or have ruled out — and why.

---

## 3. Theoretical frame-cost model

What *should* a frame cost on the reference hardware at 256³, 50% render scale (so ~960×600 ≈ 576k rays)?

The model is built bottom-up:

- **Per ray**: average traversal depth × per-step cost. Traversal depth at 256³ is `log2(256) = 8` levels in the worst case; expected depth on a typical scene is shallower because of large empty regions and large solid regions both terminating early.
- **Per step**: a node fetch (8 bytes? 16? — depends on encoding), a child-mask test, and a child-pointer lookup. Cost dominated by memory bandwidth for the node fetches because the working set exceeds L1 quickly.
- **Total frame**: rays × per-ray cost, plus a small fixed overhead for surface acquire, blit, and HUD/overlay passes.

To be filled in:

- TODO: derive the model with concrete numbers, citing M1 GPU specs.
- TODO: compare against the headless raycast bench (0.19 ms mean at 256³) and the windowed app (~30 ms on M2; need M1 MBA datapoint).
- TODO: characterize the gap. spark's investigation in dlse.2 today suggests at least 100× of the windowed gap is *not* explained by the model — that's exactly the kind of thing this section exists to catch.

---

## 4. SVDAG ↔ memo-step interaction

Hashlife memo (`step_cache`) and the SVDAG share a deep structural assumption: **subtrees are deduplicated and addressed by content.** This section owns the architectural surface where they meet.

Open questions to resolve in the paper:

- **Shared store or siloed?** Currently siloed (TODO: confirm by reading `crates/ht-octree/src/store.rs` — does the SVDAG build walk the same node IDs the sim uses, or does it re-hash from raw cells?). If siloed, what is the cost — duplicated memory, duplicated hashing work, or just convenience? If shared, what would be lost?
- **Memo churn vs SVDAG compaction.** Each generation, active-material rules dirty some subtree set. The SVDAG must either rebuild those subtrees or accept node turnover. Steady-state rate? At the limit, if the entire reachable set churns per generation, the SVDAG offers nothing over a flat 3D texture. Need a measurement and a model.
- **Cache locality across the boundary.** The sim writes cells; the renderer reads cells (via the SVDAG). On a unified-memory architecture (M1), they share L3/system bandwidth. If sim writes are evicting render-side hot lines (or vice versa), we pay an invisible cost that doesn't show up in either subsystem in isolation. spark's hypothesis 2 in dlse.2 (cross-frame `raycast_texture` serialization) is one face of this — there may be others.

If there is a fundamental architectural limit at this boundary, we surface it here. If there isn't, we say so explicitly so future investigations stop suspecting one.

---

## 5. Gap report

Measured vs theoretical, by subsystem. This is the operational output of the model — every entry should cite both the measurement and the model section it's compared against.

Format (placeholder):

| Subsystem | Measured (ref HW) | Theoretical | Gap | Status |
|---|---|---|---|---|
| Raycast traversal (headless, 256³) | TODO | TODO | TODO | TODO |
| Raycast traversal (windowed, 256³) | TODO | TODO | TODO | TODO |
| SVDAG build (cold) | TODO | TODO | TODO | TODO |
| SVDAG compaction | TODO | TODO | TODO | TODO |
| Step (memo hit path) | TODO | TODO | TODO | TODO |
| Step (memo miss / cold gen) | TODO | TODO | TODO | TODO |

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
