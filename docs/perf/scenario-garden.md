# Scenario garden

Status: first probe, `hash-thing-8ppq.2`, 2026-05-02.

The scenario garden turns a probe into data by putting the scenario shape in RON instead of in a bespoke bench. Run:

```bash
cargo run --bin scenario-runner -- scenarios/default-terrain-idle.ron
cargo run --bin scenario-runner -- scenarios/default-terrain-idle.ron --append .ship-notes/perf-runs.jsonl
```

The runner emits one JSONL measurement record matching `perf-measurement-schema.md` v2. With `--append`, it also appends that record to a JSONL file. The `scenario_hash` uses the v2-B canonical input over world, level, scene, rule_set, intensity, optional setup, and seed. Backend is deliberately not part of the scenario hash so hashlife and chunk-array runs can be compared on the same scenario.

## Schema

```ron
(
    name: "default-terrain-idle",
    world: Tiny,                   // Tiny | Small | Medium | Demo
    level: Some(5),                // optional override for world
    scene: DefaultTerrain,         // DefaultTerrain | DefaultDemo | FactoryConveyor | QuarantineAtlas
    rule_set: DefaultCa,           // DefaultCa | FactoryConveyorV1
    intensity: Idle,               // Idle | Microchurn | PassiveActive | Cascade
    regime: Saturated,             // Saturated | Churning | NotApplicable
    backend: HashlifeRecursive,    // HashlifeRecursive | ChunkArray
    generations: 3,
    warmup_generations: Some(1),
    seed: 1,
    setup: None,                   // optional; e.g. Some(QuarantineAtlasMixedContainmentV1) or Some(FactoryConveyorRuleV1)
    comparator: Some("chunk-array@same-scenario"),
)
```

`backend = ChunkArray` must use `regime = NotApplicable`; memo-cache regimes are only valid for `HashlifeRecursive`.

## Seeders

- `default-terrain`: `World::seed_terrain(TerrainParams::for_level(level))`.
- `default-demo`: default terrain plus water/sand and the demo spectacle when the world is at least 64 cells wide.
- `factory-conveyor`: either the older repeated-lane toy (`setup=None`) or the `FactoryConveyorRuleV1` source/sink/backpressure harness with a scenario-local one-material +X block rule (`rule_set=FactoryConveyorV1`). Source injection, sink drain, and backpressure counting run outside timed `step_us` / `step_*` latency.
- `quarantine-atlas`: deterministic Quarantine Atlas playtest scene. Optional setup `QuarantineAtlasMixedContainmentV1` applies the `oym4` six-stamp mixed containment plan before warmup/measured stepping; it excludes interactive placement/raycast/cache-invalidation cost.

## Current examples

- `scenarios/default-terrain-idle.ron`
- `scenarios/cascade-peak.ron`
- `scenarios/cascade-peak-demo.ron`
- `scenarios/factory-conveyor-toy.ron`
- `scenarios/factory-conveyor-rule.ron`
- `scenarios/quarantine-atlas-mixed-containment.ron`

Automatic comparison-record synthesis is intentionally left out of the first probe. For now, run paired scenarios with different `backend` values and compare records by matching `scenario_hash`, `rule_set`, and hardware.
