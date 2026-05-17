# pa24 Factory Substrate Probe

Date: 2026-05-03
Bead: `hash-thing-w4zq`

## Evidence

At `small · factory-conveyor · passive-active`, setup
`FactoryConveyorRuleV1`, rule set `custom:factory-conveyor-v1`, same scenario
hash `sha256:f343616be95e55c5`:

- Hashlife measurement:
  `factory-conveyor-rule-hashlife-recursive-f343616b-d134e37-1777791485357`
  - regime `saturated`
  - `step_p95_ms = 0.520`
  - `work_elision_p05_x = 85.3333`
  - `factory_sinked_total = 49`
  - `factory_backpressure_total = 105`
- Chunk-array comparator:
  `factory-conveyor-rule-chunk-array-f343616b-d134e37-1777791490673`
  - regime `n/a`
  - `step_p95_ms = 18.982`
  - `factory_sinked_total = 49`
  - `factory_backpressure_total = 105`
- Comparison:
  `factory-conveyor-rule-chunk-array-f343616b-d134e37-1777791490673-vs-factory-conveyor-rule-hashlife-recursive-f343616b-d134e37-1777791485357-step_p95_ms`
  - ratio `36.504x`

The comparison validator now rejects mismatched factory totals or per-generation
factory counters, so this pair does not rely only on final material counts.

## Novelty Judgment

This is valid evidence for a factory-like repeated transport rule that Hashlife
can elide aggressively while preserving source/sink throughput parity against
the chunk-array backend.

It is not yet evidence for a full factory conveyor substrate. The rule assigns
motion to the item material itself (`METAL`) and moves it in +X through air.
There is no separate belt material, belt direction encoded in cells, splitter,
merger, routing, or item-on-belt interaction. Source injection, sink drain, and
backpressure accounting are harness operations outside timed `step_us` /
`step_*` latency.

## Recommendation

Accept `w4zq` as a bounded factory sub-probe: "one-material conveyor-rule
transport with source/sink/backpressure harness." Do not promote `pa24` on this
alone as "factory substrate proven." The next useful factory bead should add
cell-encoded belt direction or a separate belt material, then re-run the same
paired throughput comparison.
