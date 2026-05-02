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
  "scenario_hash": "sha256:7c3f...",     // or "none" / "unknown"
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
  "scenario_hash": "sha256:7c3f...",
  "rule_set": "default-ca",

  "notes": "free-text"
}
```

A comparison is *not* a measurement. The `ratio_metric` must be a key from one of the referenced measurements' `metrics` objects. The `scenario_hash` and `rule_set` MUST match across `subject` and `baseline` for the comparison to be honest — different scenarios produce non-comparable numbers.

---

## Metadata enums

### `rule_set`

| value                  | meaning                                                        |
|------------------------|----------------------------------------------------------------|
| `default-ca`           | The current production CA + Margolus rules.                    |
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

### `regime` × `backend` constraint table

A single, hard cross-axis constraint:

| backend             | allowed `regime` values                                          |
|---------------------|------------------------------------------------------------------|
| `hashlife-recursive`| `cold / warming / saturated / churning / compacted` (see regimes.md) |
| `chunk-array`       | `n/a` only                                                       |
| `gpu-compute`       | `n/a` (until a GPU memo design lands; then revisit this table)   |
| `hybrid`            | `n/a` (caller must choose one of the above per-component or document) |

Records that violate this constraint are malformed.

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
| `step_mean_ms`        | ms            | Arithmetic mean of per-step wall time.        |
| `step_median_ms`      | ms            | Median per-step wall time.                    |
| `step_p95_ms`         | ms            | 95th percentile per-step wall time.           |
| `frame_total_p95_ms`  | ms            | 95th percentile per-frame wall (step+render). |
| `wall_total_ms`       | ms            | Sum of per-step wall over the run.            |
| `pop_count`           | int           | Live cell count at end-of-run.                |
| `memo_hit_ratio`      | 0.0–1.0       | Hashlife memo hit rate (post-warmup).         |
| `elision_factor_x`    | × multiplier  | Hashlife elision factor (memo hits ÷ misses+1). |
| `seed_ms`             | ms            | Wall-time of the seed step.                   |
| `compaction_ns`       | ns            | Last `maybe_compact` wall.                    |

New metric names: pick one with the unit suffix; document here in the same PR.

---

## Schema versioning

Every record carries `schema_version: <integer>`. This rev is `2`.

### Migration policy

- **Additive change** (new optional metadata field, new metric name, new enum value): minor — bump only if downstream tooling needs it. v2 records remain valid.
- **Coordinate semantics change or rename, or breaking change to record shape**: major bump (v3, v4). Schema doc gains a "v3 changes vs v2" section. Tooling carries a parallel-record-window where it reads both v2 and v3 for ≥1 month while old runs are migrated or aged out.

### Versioning of the schema doc itself

When the schema bumps to v3, the schema-doc filename stays `perf-measurement-schema.md` and the v3 entry is added at the top with the v2 spec preserved below as "v2 (legacy)." Don't fork the doc.

---

## Worked example (8ppq.1.1 idle, paired comparison)

The 8ppq.1.1 MVP comparator landed two measurements at level 5 idle, default-terrain. Per the cherry-pick discipline, both are `easy_only` with the hard regime pointed at 8ppq.1.4 (cascade).

```jsonl
{"schema_version":2,"record_kind":"measurement","measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-chunk-array","world":"demo","scene":"default-terrain","intensity":"idle","regime":"n/a","rule_set":"default-ca","backend":"chunk-array","hardware":"m2-pro-mbp","scenario_hash":"unknown","confidence":{"n":30,"warm_frame_policy":"all-frames","source":"bench","cherry_pick_audit":"easy_only","hard_followup_bead":"hash-thing-8ppq.1.4","notes":"32^3 default-terrain idle. CA kernel only, skips commit_step rebuild that a chunk-array-native engine wouldn't pay."},"level":5,"side":32,"git_commit":"a9f65c8","bench_fn":"bench_chunk_array_baseline_32","metrics":{"step_mean_ms":2.097,"step_median_ms":2.081,"step_p95_ms":2.294,"wall_total_ms":62.9},"generations":[]}
{"schema_version":2,"record_kind":"measurement","measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-hashlife","world":"demo","scene":"default-terrain","intensity":"idle","regime":"saturated","rule_set":"default-ca","backend":"hashlife-recursive","hardware":"m2-pro-mbp","scenario_hash":"unknown","confidence":{"n":30,"warm_frame_policy":"all-frames","source":"bench","cherry_pick_audit":"easy_only","hard_followup_bead":"hash-thing-8ppq.1.4","notes":"32^3 default-terrain idle. step_recursive (full step including maybe_compact + generation advance). Idle reaches near-fixed-point by gen 2; memo_elision=64x."},"level":5,"side":32,"git_commit":"a9f65c8","bench_fn":"bench_hashlife_32","metrics":{"step_mean_ms":0.10,"step_median_ms":0.0,"step_p95_ms":1.20,"wall_total_ms":3.0,"memo_hit_ratio":1.0,"elision_factor_x":64.0},"generations":[]}
{"schema_version":2,"record_kind":"comparison","comparison_id":"8ppq.1.1-ember-2026-05-02-32idle-pair","subject_measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-chunk-array","baseline_measurement_id":"8ppq.1.1-ember-2026-05-02-32idle-hashlife","ratio":1.91,"ratio_metric":"step_p95_ms","scenario_hash":"unknown","rule_set":"default-ca","notes":"chunk-array p95=2.29ms vs hashlife p95=1.20ms. Engine-cost framing (chunk-array kernel-only, hashlife full step). NOT a closure-grade comparator until 8ppq.1.4 cascade lands."}
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

`bench_fn` and `scenario_hash` are runner outputs (the runner computes the hash of the scenario file's content + seed and writes it back into the measurement record). 8ppq.2 doesn't need to enumerate them; it only needs to emit the runner that fills them.

If 8ppq.2 lands a scenario shape that v2 doesn't cover, file an additive-change bead and bump the schema (minor or major per migration policy above). Until then, v2 is the contract.

---

## What's NOT in scope of v2

- A formal JSONSchema/Avro/Protobuf schema file. Markdown spec + worked example is enough for the next harness to consume.
- Per-CA-cell histograms (`mat_distribution` is a placeholder; populate when 8ppq.1.4 needs it).
- Streaming aggregation across runs (the consumer reads JSONL, groups by `scenario_hash` + `rule_set` + `hardware`, and produces the comparison records itself).
- Tooling enforcement of the `easy_only → hard_followup_bead` link. Code-review-time check, not write-time.
