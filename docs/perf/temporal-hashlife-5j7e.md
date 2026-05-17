# 5j7e temporal hashlife verdict

Status: first pass for `hash-thing-7qob`, 2026-05-03.

## Question

`hash-thing-5j7e` asked whether a 4D hashlife cache, keyed by something like
`(subtree, time-window, phase)`, could produce more total work elision than
independent single-tick spatial memoization.

The close criteria require both halves before disproof:

- measure at least one genuine subtree-bucketing variation;
- show either cache growth is unbounded or the implementation collapses to
  `step_recursive_pow2`.

## Experiment

The first accepted probe is `scenarios/temporal-reuse.ron`:

`small · replay-scrub · passive-active · churning`, setup
`TemporalReuseV1(seed_center_radius=12,density=0.35,rule=crystal)`,
scenario hash `sha256:80d2dd5cda1fc73d`, 16 measured generations.

The probe records structural subtree fingerprints from every measured
generation. It does not rely on raw `NodeId` identity. Empty subtrees are
reported separately, same-generation spatial duplicates do not count as
temporal recurrence, and each generation records population plus whole-state
hash.

Observed result:

| metric | value |
|---|---:|
| active weighted reuse from prior distinct generations | 0.126x |
| parity-shaped weighted reuse from prior distinct generations | 0.124x |
| 4-step bucket recurrence | 0.063x |
| 8-step bucket recurrence | 0.095x |

The temporal bucket variants do not beat the existing parity-shaped view and
are far below the `>1.5x` proof bar.

## Implementation sketch

There are two exact things a temporal cache can mean here.

### 1. Exact multi-tick subtree result

If the cache answers: "given this subtree at generation `g`, what is the
center result after `2^k` ticks?", then this is the existing macro-stepper:

- `step_recursive_pow2()` advances by `1 << (level - 1)`;
- `step_node_macro(node, level, generation)` recursively composes two
  half-skips;
- `hashlife_macro_cache` is keyed by `(NodeId, generation)` and returns the
  exact multi-tick result.

Changing the key name to `(NodeId, generation_bucket, phase)` does not create
a new semantic capability. If the bucket aliases generations that are not
semantically equivalent, it is unsound. If it only aliases equivalent
generations, the key has collapsed back to the phase/generation discipline
already used by `step_recursive` and `step_node_macro`.

So the exact multi-tick variant is not a new 4D cache; it is macro-skip
reenablement. That work belongs under the deferred macro-skip epic
`hash-thing-82bt`, not under a separate temporal hashlife lead.

### 2. Replay/scrub trace cache

If the cache answers: "which subtree fingerprints appeared in previous
generations so replay or rewind can jump to them?", then it is not computing a
new step result. It is retaining a history index.

A history index has linear trace growth unless bounded:

`O(window_size * unique_active_subtrees_per_generation)`.

Bounding the window makes it a fixed replay cache, not a general temporal
hashlife. The measured bounded windows above found less recurrence than the
existing parity-shaped key view. Keeping longer windows only increases memory
pressure unless the workload has delayed recurrence that short windows miss;
that is a new workload-specific replay feature, not evidence for this broad
lead.

## Verdict

Close `hash-thing-5j7e` as not-proven / disproven for the current lead shape.

The measured temporal-bucketing variation is below the `1.5x` bar, and the
remaining exact implementation choices either collapse to the existing
macro-stepper or become a replay trace cache with window-proportional growth.

This does not kill time-as-gameplay forever. It redirects it:

- exact fast-forward/catch-up belongs to `hash-thing-82bt` macro-skip;
- a product rewind mechanic should start from a concrete puzzle prototype and
  can file a fresh lead with narrower cache semantics.
