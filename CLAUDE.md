# Hash Thing

3D voxel cellular automaton engine. See `SPEC.md` for architecture; `bd ready` for live task list.

## Roadmap execution: /ship every task

**Every task from the beads roadmap MUST be completed via `/ship`.** Don't `bd close` a task from a direct commit or informal verification. The flow per task is:

1. `bd update <id> --claim` (atomic, sets in_progress + assignee)
2. Implement
3. `/ship` (runs the full ship workflow: tests, lints, build, commit, push, close)

`/ship` is the single chokepoint that enforces quality gates and "landing the plane" for every roadmap item. No shortcuts.

## Agent names

`BEADS_ACTOR` draws from a fixed crew pool: short natural-object words that are not confusable with people names.

Pool: `flint · cairn · onyx · wisp · mote · ember · spark · quill · thorn`

Current assignments:
- `flint` — simulation / octree / physics / CA rules
- `onyx` — SVDAG rendering / GPU / shaders
- `ember` — generalist / floating worker

Reuse a name if an agent comes back on the same lane; otherwise pull the next available name from the pool. Past closed issues retain whatever historical assignee they had — don't rewrite history.

```bash
export BEADS_ACTOR=flint   # or onyx, ember, etc.
```

## Session completion ≡ /ship

bd auto-init appends a generic "push before stopping" block; it does not apply here. `/ship` is the single chokepoint and handles tests → commit → push → `bd close` as one atomic unit. Don't run `git push` or `bd close` manually outside of `/ship`.
