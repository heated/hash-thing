# hash-thing-aqq4 verdict — is hashlife actually a multiplier?

**Date:** 2026-04-29
**Actor:** ember (worktree silly-imagining-bear)
**Harness:** `tests/memo_hit_thesis_probe.rs`

## TL;DR — VIABLE

**Hashlife is delivering its multiplier.** The "memo_hit ≈ 0.40" headline that
gated the bead is the wrong metric — it conflates "elided via cache" with
"elided via short-circuit" and ignores the multiplicative effect of cache hits
at upper levels. The right metric is **work-elision factor** (= L3 base-case
misses per step ÷ total L3 nodes in the world), and on representative scenes it
is in the 16–50× range — well within "hashlife-class multiplier" territory.

| Scenario                     | memo_hit | skip_empty | skip_fixed | L3 misses/step | Naive baseline | Elision factor |
| ---------------------------- | -------- | ---------- | ---------- | -------------- | -------------- | -------------- |
| All-stone (uniform inert)    | n/a      | 0.00       | 1.00       | 0              | 4096           | ∞ (full inert) |
| Empty world                  | n/a      | 1.00       | 0.00       | 0              | 4096           | ∞ (full empty) |
| Repeating sparse lattice     | n/a      | 0.00       | 1.00       | 0              | 4096           | ∞ (all-inert)  |
| Default terrain (static)     | 0.59     | 0.14       | 0.16       | 88             | 4096           | **46×**        |
| Default terrain + churn (20) | 0.57     | 0.56       | 0.02       | 254            | 4096           | **16×**        |

(All numbers from 128³ probe — level 7 padded to level 8. 10 warm steps, 2 cold
discarded.)

## Why memo_hit is the wrong headline

`memo_hit = cache_hits / (cache_hits + cache_misses)` — but it does **not**
include the empty / inert short-circuits that fire BEFORE the cache lookup. On
the default-terrain run, 30% of `step_node` calls short-circuit before hitting
the cache; those don't appear in either the `cache_hits` or `cache_misses`
counters, but they very much DID elide work.

A better single-number metric is **effective skip rate**:

```
effective_skip = (cache_hits + empty_skips + fixed_point_skips)
                 / (cache_hits + cache_misses + empty_skips + fixed_point_skips)
```

For default terrain (static): `(225 + 78 + 86) / 544 = 71%`.
For default terrain (churn): `(1064 + 2514 + 86) / 4477 = 81%`.

But even that's incomplete — a single cache hit at level 7 elides 8⁴ = 4096
base-case evaluations. The truly load-bearing metric is the L3 miss count
relative to a naive every-cell evaluation:

```
elision_factor = (world_side / 8)³ / L3_misses_per_step
```

For 128³ default-terrain churn case: `4096 / 254 = 16×`.
For 128³ default-terrain static: `4096 / 88 = 46×`.

These are real multipliers. The naive "step every level-3 sub-cube" path would
do 4096 base-case evaluations per step; hashlife is doing 88–254. **Hashlife is
working.**

## Where misses come from

Per-level miss distribution on default-terrain churn (per warm step):

```
L3: 254  (53% — base case work)
L4: 411  (most L4 misses descend into all-cached L3, but the L4 entry itself is
          uncached because (sub_root, phase) hasn't been seen at this generation)
L5: 113
L6:  28
L7:   7
L8:   1  (root level)
```

The L4-spike is interesting: it tells us cache pressure is highest at the level
just above base cases. The cache key `(NodeId, schedule_phase)` mostly hits at
upper levels (large subtrees stable across steps) and misses at lower levels
(churn-affected regions). That's exactly the hashlife behaviour you want.

## Where the room is, if we want to push it

The static-vs-churn comparison reveals the failure mode under load: **memo_hit
stays roughly the same (0.59 → 0.57), but L3 misses jump 3×** (88 → 254) and L4
misses jump 12× (34 → 411).

The churn injects 20 fresh sand cells per step in random positions. Each fresh
cell perturbs ~8 distinct level-3 sub-cubes' content (2×2×2 Margolus block +
Moore neighborhood). At 128³, with 4096 total L3 nodes, 20 churn events
touching ~160 unique L3 nodes per step matches what we observe (254 L3 misses).

The remaining misses are amplified by Margolus phase: a touched L3 node
generates a unique step result for each (NodeId, phase % memo_period) tuple, so
even cache-hits across non-churned regions need to wait for both phases of the
period to populate. Today's `memo_period` for the demo materials is `2 ·
LCM(tick_divisors) = 2` (all divisors = 1) for water/sand. Period 2 means cache
fills in ~2 generations.

Direct levers if we ever want to push elision factor higher:

1. **Macro-skip path** (`step_recursive_pow2`) — already implemented; folds N
   generations into one descent. Disabled in production today (deferred via
   `hash-thing-82bt` epic). Re-enabling on the static branches alone would
   roughly square the elision factor.
2. **GPU spatial-memo** (`hash-thing-co1e`, `hash-thing-abwm`). The same data
   flow ported to compute shaders, with the L3 batch (currently sequentially
   processed per ecmn / ftuu) moved to GPU. Would shrink per-L3-miss latency
   from ~14µs CPU to ~0.1µs GPU, multiplying total speedup beyond the
   elision-factor ceiling.
3. **Phase-key folding** — already done as `vqke.2.1` (`hash-thing-5ie4`).
   Fast-subtree fold extends cache reuse to every-other-generation pairs.
4. **Content-folded cache key** (`vqke.2.2` territory, not yet filed) — keying
   on subtree hash content instead of NodeId could squeeze out the
   not-quite-identical-but-equivalent subtree matches. Diminishing returns
   though — would mostly recover the per-step churn-region miss volume, which
   is already small.

## Per-rule attribution — deferred

The bead asks for memo_hit broken down by quad-step source (gravity_gap_fill
cascade vs ordinary CA step vs water rule vs lava rule). The current
`HashlifeStats` doesn't track that — would need to instrument
`step_grid_once_pure` to count per-rule miss attribution. Recommend filing as
a separate bead if the need re-emerges; the per-level breakdown alone is
enough to make the viability call.

`gravity_gap_fill` is no longer a step phase per `qy4g.2 option G` (deleted
2026-04-26 — see `World::step` and SPEC.md), so that part of the bead's
question is moot.

## Conclusion

**Hashlife is delivering 16-50× work elision on representative loads. The
ceiling is fixable property of the implementation, not an architectural limit.
The thesis (spatial sim / hashlife as force-multiplier engine) is VIABLE.**

The "40% memo_hit" alarm was a false alarm — the cache_hit ratio was
underreporting because it ignored the empty/inert short-circuits and didn't
weight by subtree size. Recommend updating `memo_summary` to surface elision
factor as a primary line so future field readings give the real signal
directly.

No alternative-direction escalation is needed. The macro-skip and GPU paths
remain natural follow-ons but the CPU implementation as-it-stands is
multiplying the load enough to justify the architecture.

## Followups worth considering (not blocking)

- **Surface elision factor in `memo_summary`** — add `elision=46×` token next
  to `memo_hit`. Cheap, makes future field readings less misleading.
- **Per-rule miss attribution** — instrument `step_grid_once_pure` only if the
  per-level data leaves a question unanswered. Probably not needed.
- **Run on 256³** to confirm the ceiling holds at the demo's actual scale
  (probe ran at 128³ for ≤30s test budget). Expectation: similar
  proportional behaviour, maybe slightly lower elision due to larger
  active surface.
- **Re-enable macro-skip path** (`hash-thing-82bt` epic) — biggest near-term
  multiplier. Today's CPU sim ceiling at 256³ is ~6 Hz; macro-skip should
  push that toward 12-25 Hz on the demo scene.
