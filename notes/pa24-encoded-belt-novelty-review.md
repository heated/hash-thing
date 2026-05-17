# pa24.1 Encoded Belt Routing Novelty Review

## Claim

`FactoryEncodedBeltRoutingV1` is a small but real factory-substrate step beyond
`FactoryConveyorRuleV1`: items are separate `METAL` cells, belt substrate is
separate inert material encoded as +X/+Z direction, the path includes an L-turn
and a merge, and route-specific telemetry proves both backends execute the same
routing behavior. At `small · factory-conveyor · passive-active · saturated`,
hashlife gets `work_elision_p05_x=42.667x` because the repeated belt lattice is
mostly stable substrate with sparse moving fronts; chunk-array still scans the
whole 64^3 grid every tick. This is the factory-shaped reason hashlife matters:
large repeated infrastructure plus sparse active item flow gets memoized, while
the naive grid backend pays for the inactive factory floor.

## Evidence

- Support commit: `3d3528c`.
- Scenario hash: `sha256:f86330eff1ae9181`.
- Hashlife: `factory-encoded-belt-routing-hashlife-recursive-f86330ef-3d3528c-1777869708771`.
- Chunk-array: `factory-encoded-belt-routing-chunk-array-f86330ef-3d3528c-1777869716031`.
- Coordinate: `small · factory-conveyor · passive-active · saturated` for hashlife, `small · factory-conveyor · passive-active · n/a` for chunk-array.
- Hashlife p95: 1.125 ms; chunk-array p95: 19.192 ms; chunk/hash p95 ratio: 17.060x.
- Hashlife `work_elision_p05_x`: 42.667x.
- Matching route totals: `source_x_injected=20`, `source_z_injected=20`, `source_x_backpressure=20`, `source_z_backpressure=20`, `sink_x_drain=7`, `turn_traversal=40`, `merge_winner_x=20`, `merge_winner_z=8`, `merge_stall=12`.

## Reviewer Question

Does this satisfy `hash-thing-pa24.1` as a proven encoded-belt/routing factory
sub-probe, or is the scenario still too scripted/narrow to count as factory
substrate evidence?
