# Whole-Codebase Trident Review Round 2

Bead: `hash-thing-a08q`
Date: 2026-05-03
Pack: `/Users/edward/at/arch/notes/trident-review-whole-codebase-round2-pack-20260503-0307`

## Coverage

Review inputs:

- Claude standard: complete, `standard-claude.md`
- Claude critical: complete, `critical-claude.md`
- Codex standard / critical / evolutionary: launched and produced substantial
  logs, then hit the known Codex background-hang failure and were stopped
- Gemini standard: failed at startup due the known local Node regex flag
  incompatibility; see `standard-gemini.log`

Baseline validation supplied to reviewers:

- `cargo fmt --check`: pass
- `cargo test -p hash-thing`: pass
- `cargo test -p hash-thing --bin scenario-runner factory_conveyor -- --nocapture`: pass
- Factory scenario comparison at `small · factory-conveyor · passive-active`: pass

## Synthesis

No reviewer found a broad "stop the line" collapse in the sim kernels or the
recent perf-evidence stack. The strongest findings are concentrated in two
areas:

1. App-level async world ownership can still lose user-visible scene state.
2. Perf evidence tooling can still admit or emit records that are easier to
overclaim than the schema intends.

The single most important code finding is the stale world-prefetch race:
`maybe_start_world_prefetch` clones the current world, scene swaps can run while
that prefetch is pending, and `apply_world_grow_result` later accepts the stale
world without an epoch/token check. A scene key pressed during prefetch can be
overwritten by the old grown world when the worker returns.

The second important system finding is that default `RayonBfs` only logs when
the leaf frontier crosses its soft limit; it still allocates the whole
frontier/output batch. This is known in comments, but the whole-codebase review
agrees it needs a hard cap, batching, or fallback before large/adversarial
worlds become product evidence.

The scenario-runner findings are about evidence integrity:

- append mode ignores untracked files, so a closure-grade JSONL row can cite an
  untracked scenario file;
- compare mode's record validator is much weaker than run-mode scenario
  validation;
- drifted comparisons still emit ratio records with only a text caveat.

The recommended single-module deep review successor is `src/sim/hashlife.rs`.
The whole-codebase pass repeatedly converged on BFS/default-dispatch memory
behavior, cache/phase parity, and comparison to brute/chunk-array as the most
load-bearing surface for correctness and product evidence.

## Triage Beads

Filed from this synthesis:

- `hash-thing-a08q.1`: stale world-prefetch result can overwrite newer scene
  swap.
- `hash-thing-a08q.2`: hard-cap or chunk default RayonBfs leaf frontier.
- `hash-thing-a08q.3`: scenario-runner append mode must reject untracked
  closure inputs.
- `hash-thing-a08q.4`: strengthen compare-mode measurement-record validation.
- `hash-thing-a08q.5`: make trajectory drift structural or opt-in, not only a
  notes caveat.
- `hash-thing-a08q.6`: add bounded multi-level BFS parity coverage to default
  tests.
- `hash-thing-a08q.7`: update stale bench comments from `--release` / `bench`
  to `--profile perf`.

Existing successor:

- `hash-thing-n5uo`: single most critical module deep trident. Module choice:
  `src/sim/hashlife.rs`.
