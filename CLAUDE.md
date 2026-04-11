# hash-thing (Ashfall)

3D voxel cellular automaton engine. See `SPEC.md` for architecture; `bd ready` for live task list.

## Roadmap execution: /ship every task

**Every task from the beads roadmap MUST be completed via `/ship`.** Don't `bd close` a task from a direct commit or informal verification. The flow per task is:

1. `bd update <id> --claim` (atomic, sets in_progress + assignee)
2. Implement
3. `/ship` (runs the full ship workflow: tests, lints, build, commit, push, close)

`/ship` is the single chokepoint that enforces quality gates and "landing the plane" for every roadmap item. No shortcuts.

## Agent names

Per-session `BEADS_ACTOR` in this repo uses crew names (not path-derived). Set once at session start:
- `margo` — simulation / octree / Hashlife stepping worktrees (Norman Margolus, 2x2x2 blocks)
- `kampe` — SVDAG rendering (Kämpe et al. 2013, foundational SVDAG paper)
- Additional crew names drawn from CA / SVDAG / voxel research lineage as needed

```bash
export BEADS_ACTOR=margo   # or kampe, etc.
```
