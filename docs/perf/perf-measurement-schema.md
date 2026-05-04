# Perf-measurement structured-log schema (v2)

**Status:** v2, hash-thing-8ppq.9, 2026-05-02. Sibling of [`regimes.md`](./regimes.md).

This doc owns the **JSONL record shape** for perf measurements and comparisons. Coordinate values (`world / scene / intensity / regime` enums) are defined in `regimes.md`; this doc owns the record shape, the 6 mandatory metadata fields, the canonical metric names, and migration policy.

If you're authoring a perf claim, read `regimes.md` first. If you're writing a runner that emits perf records, read this doc.

---

## File format and append semantics

- **JSONL** (line-delimited JSON). One record per line. Append-only.
- Canonical filename pattern: `notes/perf-runs.jsonl` (per-project) or `tests/.perf-runs.jsonl` if scoped to a bench harness.
- One record per *run*, not per generation. Per-generation data is nested under `generations: [...]` inside the run record.
- A `comparison` record is a separate JSONL line that *references* two prior measurement records by id. Comparisons aren't measurements — they're relationships between two of them.

---

## Measurement record shape

```jsonc
{
  "schema_version": 2,
  "record_kind": "measurement",
  "measurement_id": "<unique-id>",       // suggest: <bead>-<actor>-<date>-<scene>-<backend>

  // 4 headline coordinates (regimes.md owns the enums)
  "world": "demo",
  "scene": "default-terrain",
  "intensity": "idle",
  "regime": "saturated",                 // or "n/a" for non-memoized backends

  // 6 mandatory metadata fields
  "rule_set": "default-ca",              // see "Metadata enums" below
  "backend": "hashlife-recursive",
  "hardware": "m2-pro-mbp",
  "scenario_hash": "sha256:81aa21c5a72712b2",     // or "none" / "unknown"
  "setup": null,                                   // optional deterministic setup identity; omit/null for none
  "confidence": { /* see Confidence section */ },
  // schema_version is the 6th metadata field; declared up top.

  // Run identifiers
  "level": 5,                            // optional: scale-derived from world
  "side": 32,
  "git_commit": "a9f65c8",
  "bench_fn": "bench_chunk_array_baseline_32",

  // Headline metrics (see "Canonical metric names" below)
  "metrics": {
    "step_mean_ms": 2.097,
    "step_median_ms": 2.081,
    "step_p95_ms": 2.294,
    "wall_total_ms": 62.9
  },

  // Per-generation data, nested
  "generations": [
    { "gen": 0, "step_us": 2044, "pop_count": 16026, "drops": 0, "mat_distribution": null },
    { "gen": 1, "step_us": 2069, "pop_count": 16026, "drops": 0, "mat_distribution": null }
    // ...
  ]
}
```

## Comparison record shape

```jsonc
{
  "schema_version": 2,
  "record_kind": "comparison",
  "comparison_id": "<unique-id>",

  "subject_measurement_id": "8ppq.1.1-ember-2026-05-02-32idle-chunk-array",
  "baseline_measurement_id": "8ppq.1.1-ember-2026-05-02-32idle-hashlife",

  "ratio": 1.91,                         // subject_metric / baseline_metric
  "ratio_metric": "step_p95_ms",         // which metric the ratio is on

  // These MUST agree with both referenced measurements (defensive copy
  // for record-shape sanity, since the JSONL stream may be filtered).
  "scenario_hash": "sha256:81aa21c5a72712b2",
  "setup": null,                                   // optional; omit/null for none
  "rule_set": "default-ca",

  "notes": "free-text"
}
```

A comparison is *not* a measurement. The `ratio_metric` must be a key from one of the referenced measurements' `metrics` objects. The `scenario_hash`, `rule_set`, and optional `setup` MUST match across `subject` and `baseline` for the comparison to be honest — different scenarios produce non-comparable numbers.

---

## Metadata enums

### `rule_set`

| value                  | meaning                                                        |
|------------------------|----------------------------------------------------------------|
| `default-ca`           | The current production CA + Margolus rules.                    |
| `custom:factory-conveyor-v1` | Scenario-local factory conveyor block-rule setup used by `FactoryConveyorRuleV1` (`hash-thing-w4zq`). |
| `custom:factory-encoded-belt-routing-v1` | Scenario-local encoded-belt routing setup used by `FactoryEncodedBeltRoutingV1` (`hash-thing-pa24.1`). |
| `custom:soup-search-v1` | Scenario-local tiled 3D Game-of-Life soup ensemble used by `SoupSearchV1(tile=16,soup_side=8,density_per_1000=180,rule=445)` and `SoupSearchSparseV1(tile=16,soup_side=8,density_per_1000=45,rule=445)` (`hash-thing-8ppq.5`). |
| `water-margolus`       | Water-only rules (for fluid-only benches).                     |
| `particle-cellular`    | Particle CA (sand/dust/etc).                                   |
| `circuit-signal`       | Wire/signal-propagation rules (puzzle-circuit scene families). |
| `graph-cellular`       | Non-cube topology CA (graph-cellular family).                  |
| `custom:<short-id>`    | Anything not in the enum; cite the bead that defines it.       |

### `backend`

| value                  | meaning                                                        |
|------------------------|----------------------------------------------------------------|
| `chunk-array`          | Flat `Vec<CellState>` storage; CA + Margolus kernel; no memo.  |
| `hashlife-recursive`   | The hashlife memo path (`World::step_recursive`).              |
| `gpu-compute`          | GPU compute shader.                                            |
| `hybrid`               | Mixed CPU/GPU; document the split in `confidence.notes`.       |
| `custom:<short-id>`    | Anything not in the enum; cite the bead.                       |

### `hardware`

| value                  | meaning                                                        |
|------------------------|----------------------------------------------------------------|
| `m2-pro-mbp`           | Edward's M2 Pro MacBook Pro (the canonical dev box).           |
| `m2-ultra-mac-pro`     | M2 Ultra workstation.                                          |
| `ci-runner-x86`        | x86_64 GitHub Actions runner.                                  |
| `unknown`              | Genuinely unrecoverable. Do not use for new claims; archeology rule lets v1 audit rows back-fill. |

### `rule_set` is a code-program axis, not a content axis

Two scenes seeded with the same `rule_set=default-ca` but different live materials (one with fire+water, one with only stone) will have very different measured costs because the rule branches that fire are content-conditioned. That's a `scene` / `scenario_hash` distinction, not a `rule_set` distinction. `rule_set` names which CA + Margolus + block-rule program is loaded; what runs inside it depends on what cells are present.

If a future bench needs to compare runs that share `rule_set` but have very different content, the comparison record should also gate on `scene` (not just `scenario_hash`). Document the concern in `confidence.notes` if this matters for the measurement at hand.

### `scenario_hash` algorithm (v2-A)

The hash is computed over a canonical UTF-8 byte sequence that names the deterministic inputs to the scenario. Two records that share `scenario_hash` MUST produce the same seeded scene; two records that legitimately compare against each other MUST agree on it.

**v2-A recipe** (used until 8ppq.2 lands a RON-driven scenario format that supersedes it):

```
v2-A canonical input: "v2-A|" + rule_set + "|" + scene + "|" + intensity + "|"
                      + level + "|" + generations + "|" + terrain_params_json
```

where `terrain_params_json` is `serde_json::to_string(&TerrainParams)` rendered with sorted keys (or the closest canonical form). `scenario_hash` = `"sha256:" + first 16 hex chars of sha256(canonical_input)`.

**Why `bench_fn` is NOT in the hash:** two records that ran the same scenario via different bench functions are still measuring the same scenario. `bench_fn` is provenance metadata in the measurement record, not a scenario distinguisher. (Same logic applies to `backend` below.)

**v2-B recipe** (active once 8ppq.2 lands):

```
v2-B canonical input: "v2-B-canonical|world=<world>|level=<level>|scene=<scene>|"
                      + "rule_set=<rule_set>|intensity=<intensity>"
                      + optional("|setup=<setup>") + "|seed=<seed>"
```

The version prefix (`v2-A|` / `v2-B-canonical|`) lets tooling tell pre-8ppq.2 records from RON-driven ones without breaking the comparison-match contract. Omit the setup component entirely when no explicit deterministic setup is applied; this preserves historical no-setup v2-B hashes. The RON file's `backend`, `regime`, `comparator`, display `name`, sample count, and warmup policy are deliberately excluded: they change the measurement record, not the deterministic seeded scene.

**Why `backend` is NOT in the hash:** the same scenario should produce the same hash regardless of which engine measures it — that's exactly what makes paired-run comparisons valid (chunk-array vs hashlife on the same scene). Engine identity goes in `backend`, not `scenario_hash`.

### `regime` × `backend` constraint table

A single, hard cross-axis constraint:

| backend             | allowed `regime` values                                          |
|---------------------|------------------------------------------------------------------|
| `hashlife-recursive`| `cold / warming / saturated / churning / compacted` (see regimes.md) |
| `chunk-array`       | `n/a` only                                                       |
| `gpu-compute`       | `n/a` (until a GPU memo design lands; then revisit this table)   |
| `hybrid`            | the regime of the engine that holds the load-bearing memo cache; if the hybrid has multiple memo layers, `n/a` and use the optional `regime_components` field. |

Records that violate this constraint are malformed.

### Hybrid escape hatch: `regime_components`

For `backend = hybrid`, the top-level `regime` field can either name the load-bearing engine's regime (most common — e.g., a hashlife-on-CPU + GPU-precompute hybrid uses the hashlife regime), or be `n/a` with a sibling `regime_components` object describing each engine separately:

```json
"regime": "n/a",
"regime_components": {
  "cpu_hashlife": "saturated",
  "gpu_compute": "n/a"
}
```

Use `regime_components` when comparison-record `regime` matching is too coarse to be honest about the hybrid's mixed cache state.

---

## Confidence record

```jsonc
{
  "n": 30,                                // sample count for percentiles
  "warm_frame_policy": "skip-first-5",    // or "all-frames", "warm-only", etc.
  "source": "bench",                      // "bench" | "demo" | "manual" | "spec"
  "cherry_pick_audit": "easy_only",       // see below
  "hard_followup_bead": "hash-thing-8ppq.1.4",  // required iff cherry_pick_audit == "easy_only"
  "notes": "free-text caveats"
}
```

### `cherry_pick_audit` values

| value           | meaning                                                            |
|-----------------|--------------------------------------------------------------------|
| `easy_only`     | Measured at a regime that flatters the engine (e.g. idle/saturated). MUST also cite `hard_followup_bead` pointing at the bead that will measure the hard regime. |
| `hard_included` | The measurement covers the hard regime (cascade/churning/edit-active). |
| `mixed`         | Some hard, some easy; `notes` describes the split.                |
| `n/a`           | Cherry-picking discipline doesn't apply (e.g., spec-tier non-empirical claim). |

The schema can't enforce the `easy_only → hard_followup_bead` link at write-time. Code-review checklist enforces it at human-review time: any record with `cherry_pick_audit: "easy_only"` and no `hard_followup_bead` is a review failure.

---

## Canonical metric names

Units in the name. Mixing units across records breaks downstream tooling.

| name                  | unit          | semantics                                     |
|-----------------------|---------------|-----------------------------------------------|
| `step_us`             | µs            | Per-generation wall-time inside `generations[].step_us`. Aggregate metrics use ms; per-gen detail uses µs because sub-ms generations are common at small worlds. |
| `step_mean_ms`        | ms            | Arithmetic mean of per-step wall time.        |
| `step_median_ms`      | ms            | Median per-step wall time.                    |
| `step_p95_ms`         | ms            | 95th percentile per-step wall time.           |
| `frame_total_p95_ms`  | ms            | 95th percentile per-frame wall (step+render). |
| `wall_total_ms`       | ms            | Sum of per-step wall over the run.            |
| `pop_count`           | int           | Live cell count at end-of-run.                |
| `memo_hit_ratio`      | 0.0–1.0       | Hashlife memo hit rate (post-warmup).         |
| `elision_factor_x`    | × multiplier  | See "elision_factor_x formula" below.         |
| `work_elision_min_x`  | × multiplier  | Minimum warm-frame Hashlife work elision over the run. See "work_elision_* formula" below. |
| `work_elision_mean_x` | × multiplier  | Mean warm-frame Hashlife work elision over the run. |
| `work_elision_p05_x`  | × multiplier  | 5th-percentile warm-frame Hashlife work elision; anti-cherry-pick thesis metric for churning runs. |
| `leaf_misses_mean`    | active leaves | Mean active-leaf misses per warm frame.       |
| `work_elision_leaf_level` | octree level | Active Hashlife leaf level used for work-elision accounting. Usually 3; 4 when slowed block-rule materials need the wider base-case halo. |
| `memo_table_entries`  | entries       | Per-generation Hashlife spatial memo table size in `generations[].memo_table_entries`. |
| `memo_table_entries_final` | entries  | Final Hashlife spatial memo table size after the measured run. |
| `bfs_l3_unique_misses` | active leaves | Per-generation active leaf-level unique misses in `generations[].bfs_l3_unique_misses`; legacy name still says L3 even when the active leaf level is 4. |
| `bfs_l3_unique_misses_mean` | active leaves | Mean active leaf-level unique misses over measured frames. |
| `bfs_l3_unique_misses_p95` | active leaves | 95th-percentile active leaf-level unique misses over measured frames. |
| `bfs_max_batch_len` | tasks | Per-generation largest BFS frontier/batch length in `generations[].bfs_max_batch_len`. Equals `bfs_l3_unique_misses` on the normal leaf-batch path; can differ when a higher-level hard-cap fallback trips. |
| `bfs_max_batch_len_mean` | tasks | Mean largest BFS frontier/batch length over measured frames. |
| `bfs_max_batch_len_p95` | tasks | 95th-percentile largest BFS frontier/batch length over measured frames. |
| `miss_cause_table`   | JSON object    | Structured miss-cause attribution table. With `HASH_THING_MEMO_DIAG=1`, status is `ok` and rows split misses by level into first-seen/no-surviving-key, parity alias, slow-divisor phase alias, residual unknown, and compaction kept/dropped counts. Without the diagnostic gate, scenario-runner records use `{status: "todo", dependency: "hash-thing-vqke.1"}`. |
| `factory_sinked_total` | items         | Scenario-specific factory harness sink throughput over measured frames; source/sink harness work is outside timed `step_us` / `step_*` latency. |
| `factory_backpressure_total` | blocked source attempts | Scenario-specific count of source attempts blocked by occupied lane inputs before the CA step; source/sink harness work is outside timed `step_us` / `step_*` latency. |
| `factory_routing_total` | JSON object | Route-specific factory harness totals for encoded-belt setups: per-leg source injection/backpressure, sink drain, turn traversal, and merge winner/stall counts. Comparison validation requires hashlife/chunk-array equality. |
| `seed_ms`             | ms            | Wall-time of the seed step.                   |
| `compaction_ns`       | ns            | Last `maybe_compact` wall.                    |

New metric names: pick one with the unit suffix; document here in the same PR.

### `soup_search` summary

`scene=soup-search` records must include a top-level `soup_search` object. It
contains `setup`, aggregate `tile_count` / `survivor_count` /
`candidate_stable_count` / `extinct_count`, and a `tiles` array. Each tile row
records the tile coordinate, `pop_history`, final and max population, measured
lifespan, survivor/candidate booleans, and `final_state_hash`.

This is a measured-window classifier, not a proof of true oscillator stability.
Comparison validation requires the whole summary to match between backends.

### Numerical precision

- All metric values stored at f64 full precision (no pre-rounding at write-time).
- Audit-table renderings in `regimes.md` may round to 2 decimals for readability; the canonical record is the JSONL file. Tooling that dedupes or compares MUST read the JSONL value, not the table.
- Comparison `ratio` values are stored at full f64 precision and rounded only at display time.

### `elision_factor_x` formula

```
elision_factor_x = (memo_hits + memo_misses) / (memo_misses + 1)
```

- Counts are **post-warmup**, gated by the same `confidence.warm_frame_policy` that gates `step_p95_ms`. If `warm_frame_policy` is `skip-first-5`, the `memo_hits` / `memo_misses` totals come from generations 5..end.
- The `+1` in the denominator is Laplace smoothing — guarantees a finite value on a fully-cold or zero-miss run rather than dividing by zero.
- `memo_evictions` are NOT in this formula. A high-eviction cache still has `elision_factor_x` ≥ 1; eviction pressure is a separate metric (`memo_evict_ratio`, file follow-up if useful).

### `work_elision_*` formula

`work_elision_*` is the aqq4 thesis metric, not the cache-lookup ratio above.
It mirrors the `memo_elision=` token from `World::memo_summary()`:

```
active_leaf_level = 4 if any block-rule tick divisor > 1 else 3
leaf_nodes_in_world = 2^(3 * (world_level + 1)) / 2^(3 * active_leaf_level)
work_elision_factor_x(frame) = leaf_nodes_in_world / max(leaf_misses(frame), 1)
```

The `world_level + 1` term is intentional: `step_recursive` pads the root before
stepping, and `memo_summary()` uses the same padded denominator. Per-generation
records may include `work_elision_factor_x` and `leaf_misses`; aggregate metrics
summarize those per-generation values. For churning/thesis claims, prefer
`work_elision_p05_x` or `work_elision_min_x` over final-frame-only readings so a
run cannot pass by ending after the active cascade calms down.

---

## Schema versioning

Every record carries `schema_version: <integer>`. This rev is `2`.

### Migration policy

- **Additive change**: new OPTIONAL metadata field with a documented default; new metric name; new enum value. Minor — bump only if downstream tooling needs it. v2 records remain valid.
- **Breaking change**: making a previously-optional field required; renaming a field or enum value; removing a field; changing the meaning of an existing field; tightening a constraint that existing records may violate. Major bump (v3, v4). Schema doc gains a "v3 changes vs v2" section.

### Migration tooling contract

When v3 lands, the same PR MUST land a `scripts/migrate-perf-records.py` (or equivalent) that takes a v2 JSONL file and emits a v3 JSONL file. The script is the migration contract; the prose section is the rationale. Without an executable migrator the schema bump is incomplete.

JSONL is append-only — already-emitted v2 records are NOT rewritten. The migrator produces a v3-shaped *copy* that downstream tools consume; v2 records remain canonical for their write window. Consumer tools either read both shapes (parallel-record-window, ≥1 month) or read only the migrated v3 copy.

### Enum-extension policy

Applies to `rule_set`, `backend`, `hardware`, and any other enum metadata field.

- **Adding a value** is additive (minor bump). Past records that had no occasion to use the new value remain valid.
- **Renaming or splitting a value** is breaking (major bump). Example: if `m2-pro-mbp` ever splits into `m2-pro-mbp-2023` vs `m2-pro-mbp-2024` for thermal-class differences, that's a v3 change with a migrator.
- **Cross-machine compatibility classes** (e.g., "all M-series MBPs are roughly comparable") are NOT a coordinate property — they're a consumer-side opt-in: the consumer documents a compatibility table and explicitly compares across the table. Records always store the exact enum value.

### Headline-archetype contract freeze

The 4 headline coordinates (`world / scene / intensity / regime`) and their enum values are part of the v2 contract. The v1 "Open questions" in `regimes.md` (whether intensity is granular enough, whether the archetypes are right) remain open, but resolutions that **add** archetype values are additive (minor); resolutions that **rename or split** existing values are breaking (major).

If a future bead wants to split `cascade` into `cascade-edit` vs `cascade-physics`, that's a v3 schema bump with the migrator path documented above. Same for renaming `passive-active` to `light-traffic`. Don't sneak the change in as additive.

### Versioning of the schema doc itself

When the schema bumps to v3, the schema-doc filename stays `perf-measurement-schema.md` and the v3 entry is added at the top with the v2 spec preserved below as "v2 (legacy)." Don't fork the doc.

---

## Worked example (8ppq.1.1 idle, paired comparison)

The 8ppq.1.1 MVP comparator landed two measurements at level 5 idle, default-terrain. Per the cherry-pick discipline, both are `easy_only` with the hard regime pointed at 8ppq.1.4 (cascade).

```jsonl
{"schema_version":2,"record_kind":"measurement","measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-chunk-array","world":"tiny","scene":"default-terrain","intensity":"idle","regime":"n/a","rule_set":"default-ca","backend":"chunk-array","hardware":"m2-pro-mbp","scenario_hash":"sha256:81aa21c5a72712b2","confidence":{"n":30,"warm_frame_policy":"all-frames","source":"bench","cherry_pick_audit":"easy_only","hard_followup_bead":"hash-thing-8ppq.1.4","notes":"32^3 default-terrain idle. CA kernel only, skips commit_step rebuild that a chunk-array-native engine wouldn't pay."},"level":5,"side":32,"git_commit":"a9f65c8","bench_fn":"bench_chunk_array_baseline_32","metrics":{"step_mean_ms":2.097,"step_median_ms":2.081,"step_p95_ms":2.294,"wall_total_ms":62.9},"generations":[]}
{"schema_version":2,"record_kind":"measurement","measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-hashlife","world":"tiny","scene":"default-terrain","intensity":"idle","regime":"saturated","rule_set":"default-ca","backend":"hashlife-recursive","hardware":"m2-pro-mbp","scenario_hash":"sha256:81aa21c5a72712b2","confidence":{"n":30,"warm_frame_policy":"all-frames","source":"bench","cherry_pick_audit":"easy_only","hard_followup_bead":"hash-thing-8ppq.1.4","notes":"32^3 default-terrain idle. step_recursive (full step including maybe_compact + generation advance). Idle reaches near-fixed-point by gen 2; memo_elision=64x."},"level":5,"side":32,"git_commit":"a9f65c8","bench_fn":"bench_hashlife_32","metrics":{"step_mean_ms":0.10,"step_median_ms":0.0,"step_p95_ms":1.20,"wall_total_ms":3.0,"memo_hit_ratio":1.0,"elision_factor_x":64.0},"generations":[]}
{"schema_version":2,"record_kind":"comparison","comparison_id":"8ppq.1.1-ember-2026-05-02-32idle-pair","subject_measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-chunk-array","baseline_measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-hashlife","ratio":1.91,"ratio_metric":"step_p95_ms","scenario_hash":"sha256:81aa21c5a72712b2","rule_set":"default-ca","notes":"chunk-array p95=2.29ms vs hashlife p95=1.20ms. Engine-cost framing (chunk-array kernel-only, hashlife full step). Both records share scenario_hash, so the comparison-record honesty constraint passes — they are measuring the same scenario, differing only in backend. NOT a closure-grade comparator until 8ppq.1.4 cascade lands."}
```

Reading this: the 32³ idle comparison shows hashlife is 1.91× faster on `step_p95_ms`, but BOTH records self-disclose that idle is the easy regime; the load-bearing measurement is 8ppq.1.4's cascade pair, due once that lands.

---

## 8ppq.2 fitness check

8ppq.2 (scenario garden / config DSL) wants a RON-shaped scenario file that drives a runner. Each scenario field maps to a v2 record location:

| 8ppq.2 RON field      | v2 schema location                                     |
|-----------------------|-------------------------------------------------------|
| `world-size`          | `world` headline coord (resolves to level/side)       |
| `scene-seeder`        | `scene` headline coord                                |
| `rule-set`            | `rule_set` metadata field                             |
| `intensity-injector`  | `intensity` headline coord                            |
| `regime-target`       | `regime` headline coord (constrained by `backend`)    |
| `frames` (or generations) | `generations[]` length                            |
| `comparator`          | comparison record's `baseline_measurement_id`         |

`bench_fn` and `scenario_hash` are runner outputs (the runner computes the v2-B canonical field hash and writes it back into the measurement record). 8ppq.2 doesn't need to enumerate them; it only needs to emit the runner that fills them.

If 8ppq.2 lands a scenario shape that v2 doesn't cover, file an additive-change bead and bump the schema (minor or major per migration policy above). Until then, v2 is the contract.

---

## What's NOT in scope of v2

- A formal JSONSchema/Avro/Protobuf schema file. Markdown spec + worked example is enough for the next harness to consume.
- Per-CA-cell histograms (`mat_distribution` is a placeholder; populate when 8ppq.1.4 needs it).
- Streaming aggregation across runs (the consumer reads JSONL, groups by `scenario_hash` + `rule_set` + `hardware`, and produces the comparison records itself).
- Tooling enforcement of the `easy_only → hard_followup_bead` link. Code-review-time check, not write-time.
