# Exact Dirty-Cone Skip Prototype

Bead: `hash-thing-vqke.3`

## Verdict

The first idea to try was a narrowed base-case dependency-signature cache. Plan
review falsified the narrow footprint: the current CA plus Margolus/block-rule
step can legally pull from the boundary ring. The only safe first prototype is
therefore a full-leaf dependency-signature cache. That is intentionally
conservative; it is mostly a measurement harness for the dirty-cone hypothesis,
not a promised optimization.

This still targets the `first_seen_or_no_surviving_key` miss bucket from
`hash-thing-vqke.1` without weakening the existing `(NodeId, effective_phase)`
memo contract. If full-leaf signatures show little reuse, the exact dirty-cone
idea is not cheap enough at the base-case level and should be closed negative
or moved up to a more explicit edit-cone metadata design.

At `demo · default-demo · cascade · churning`, `vqke.1` measured miss causes as
`first_seen_or_no_surviving_key=13678`, `parity_aliased=11302`,
`slow_divisor_phase_aliased=1830`, and `compaction_drop=0`. At
`demo · default-terrain · microchurn · saturated`, the same probe measured
`first_seen_or_no_surviving_key=25173`, `parity_aliased=4126`,
`slow_divisor_phase_aliased=1540`, and `compaction_drop=534/35003 remap
entries`. That says eviction policy is not the next lever; the next exact CPU
experiment should ask whether new `NodeId`s differ only outside the cells that
can affect the returned center node.

## Correctness Invariant

For a base-case step at input level `L` and effective schedule phase `p`, the
returned center node is a pure function of only the input cells inside that
base case's one-step dependency footprint:

```text
result = F(level=L, effective_phase=p, dependency_cells(input_grid))
```

The skip is sound if and only if all of these are true:

1. The previous entry has the same `level`.
2. The previous entry has the same `effective_phase`.
3. The current dependency footprint bytes are exactly equal to the stored
   dependency footprint bytes.
4. The stored result `NodeId` still exists after compaction/remap.
5. The material/rule registry has not changed since the entry was created.

The hash of the dependency bytes is only an index. A candidate hit must compare
the stored dependency bytes exactly before returning the cached result.
Approximate equality, ignored deltas, or parity folding beyond
`effective_phase_for` are not allowed.

The cache must follow the same lifecycle as the existing Hashlife caches: clear
on material/rule mutation, and remap or drop `result` through compaction. If a
future entry stores any `NodeId` other than `result`, that field needs the same
remap/drop treatment.

## Bounded Prototype

Prototype only the active base-case miss paths:

- `level == 3`: normal 8x8x8 base case returning the center 4x4x4 node.
- `level == 4`: wide 16x16x16 base case used when block rules with
  `tick_divisor > 1` require the wider halo.

For each base-case cache miss, flatten the input grid as today, compute the
dependency signature, and probe:

```text
BaseFootprintKey {
    level: u8,
    effective_phase: u64,
    footprint_hash: u64,
}

BaseFootprintEntry {
    dependency_bytes: Box<[CellState]>,
    result: NodeId,
}
```

If the key exists and `dependency_bytes == current_dependency_bytes`, return
`result` and count `dirty_cone_hits += 1`. Otherwise compute the base case
normally, insert/replace the entry, and count `dirty_cone_misses += 1`.

With the full-leaf footprint, this only converts misses where a different
`NodeId` has identical flattened leaf bytes under the same effective phase. That
should be rare because `NodeId` is already content-addressed, but measuring the
rate is useful: a low hit rate falsifies the cheap base-case dirty-cone route;
a surprising hit rate points at NodeStore/canonicalization or compaction churn
worth investigating. It does not try to skip recursive assembly, does not change
the main hashlife cache key, and does not infer equality from ancestor dirty
flags.

## Dependency Footprint

Use the full flattened leaf first:

- For `level == 3`, store all 8x8x8 input cells.
- For `level == 4`, store all 16x16x16 input cells.

Plan review found the tempting smaller ranges are not conservative. For
`level == 3`, odd Margolus phase can make output edge cells depend on input
coordinates `0` and `7` through CA neighbors plus the shifted block pass. For
`level == 4`, slowed/mixed-divisor mode can run both Margolus offsets in one
generation; the wide-leaf code comment already notes that a center cell can
depend on a pass-0 cell two positions away before pass 1 runs. Any narrowed
footprint must first be proven against both parities and the mixed-divisor wide
path.

## Validation Plan

Add tests before enabling the fast path by default:

1. Randomized air/stone/water grids for `level == 3` and `level == 4`.
2. For each grid, make two copies with identical dependency footprints and
   randomized outside-footprint cells.
3. Run the normal base-case computation on both and assert equal results.
4. Run the dependency-signature cache path and assert it returns the same
   result as the normal path.
5. Add negative tests where one in-footprint cell differs and assert the cache
   does not hit unless exact bytes still match.
6. Add adversarial boundary-ring tests for both Margolus parities and the
   mixed-divisor level-4 path. These tests should intentionally mutate the
   coordinates a narrowed footprint would omit and prove either that the full
   footprint catches the difference or that a future narrower proof is valid.

Perf probe if code is added:

```text
demo · default-demo · cascade · churning
demo · default-terrain · microchurn · saturated
```

Report `dirty_cone_hits`, `dirty_cone_misses`, hit rate, and the existing
`memo_hit`, `memo_elision`, `p1`, and `p3` tokens. The prototype is worth
keeping only if it converts a meaningful share of `first_seen_or_no_surviving_key`
misses without increasing p95 step time. Expected result for the full-leaf
version is low hit rate; that is useful negative evidence.

## Non-Goals

- No fuzzy keys.
- No parent-level dirty mask until the base-case footprint cache proves there
  is reuse to harvest.
- No cache-policy work; `vqke.1` did not show displacement as the dominant
  miss cause.
- No macro-step integration; production `step_recursive` is the target.
