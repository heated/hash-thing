# n5uo Hashlife Deep Trident Synthesis

Bead: `hash-thing-n5uo`
Date: 2026-05-03
Module: `src/sim/hashlife.rs`
Pack: `/Users/edward/at/arch/notes/trident-n5uo-hashlife-20260503-0743`

## Coverage

Review inputs:

- Claude standard: complete, `standard-claude.md`
- Claude critical: complete, `critical-claude.md`
- Codex standard / critical / evolutionary: produced substantial logs, then
  hit the known Codex background hang and were stopped
- Gemini standard: failed at startup due the known local Node regex flag
  incompatibility, captured in `standard-gemini.log`

This was a module audit, not a branch diff review. The chosen module was
`src/sim/hashlife.rs`, as recorded on `hash-thing-n5uo` after the whole-codebase
`hash-thing-a08q` synthesis.

## Synthesis

No reviewer validated a cell-state correctness bug in the core recursive
single-step path. The ordinary memo key shape, slow-divisor phase folding,
`RayonBfs` fallback-to-serial behavior, and compaction remap paths all had a
defensible story under static review.

The strongest findings are evidence and cache-contract risks:

1. `step_recursive_profiled()` does not publish the same p3/p4 stats that
   `memo_summary()` reports after `step_recursive()`. It measures
   `step_node_us` and `compact_us` into the returned profile, but it snapshots
   `hashlife_stats` before compaction and never writes `step_node_wall_ns` or
   `compact_ns` back into `self.hashlife_stats`. Perf probes that run the
   profiled path and then inspect `memo_summary()` can read zero or stale
   descent/compaction timing.

2. Default test coverage still does not exercise the highest-risk
   multi-level `RayonBfs` frontier shape. The explicit level-7 parity test is
   ignored, and the earlier attempted ignored run exceeded the soft command
   window. Existing default tests cover useful level-4/5 and forced-fallback
   cases, but not the deeper descent/ascend stack that product-scale worlds use.

3. `compact_keeping()` advertises cache-preserving compaction but preserves
   only cache key nodes as extra roots. `remap_caches()` keeps an entry only
   when both key and result survive the remap, so result values not reachable
   from the current root are dropped. This is safe eviction, not a proven stale
   `NodeId` correctness bug, but it can create recurring cold-cache churn and
   makes `memo_compact_drop` evidence ambiguous unless the contract is clarified
   or value preservation is implemented.

## Follow-Up Beads

- `hash-thing-w93q`: `step_recursive_profiled` stats must match the
  `memo_summary` contract.
- `hash-thing-a08q.6`: bounded non-ignored multi-level `RayonBfs` parity
  coverage. This existing bead was updated with the n5uo corroboration rather
  than duplicated.
- `hash-thing-8qpp`: clarify or preserve hashlife cache values during
  compaction.

## Non-Findings

- No validated memo-key unsoundness for slow-divisor phase folding.
- No validated `RayonBfs` hard-limit fallback divergence.
- No validated compaction remap stale-`NodeId` reuse in `src/sim/hashlife.rs`.
