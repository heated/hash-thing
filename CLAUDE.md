# Hash Thing

3D voxel cellular automaton engine. See `SPEC.md` for architecture; `bd ready` for live task list.

## The crew

Named agents working on this repo. `BEADS_ACTOR` draws from a fixed pool of short natural-object words that are not confusable with people names.

Pool: `flint · cairn · onyx · wisp · mote · ember · spark · quill · thorn`

Current crew:
- **flint** — simulation / octree / physics / CA rules
- **onyx** — SVDAG rendering / GPU / shaders
- **ember** — generalist / floating worker
- **cairn** — foundations / bug fixes
- **spark** — terrain generation / infinite worlds

Reuse a name if an agent comes back on the same lane; otherwise pull the next available name from the pool. Past closed issues retain whatever historical assignee they had — don't rewrite history.

```bash
export BEADS_ACTOR=flint   # or onyx, ember, cairn, spark
```

## /ship is the default — not /ship-auto

**Every roadmap task MUST run through `/ship` (not `/ship-auto`).** `/ship` is the full workflow:

```
Route → Dialogue → Plan → Plan-review → Implement → Commit → Code-review → Fix → Push → bd close → Finish
```

**The plan gate is a real human gate.** Crew must stop after writing the plan and wait for human approval before implementing, unless the task is explicitly trivial enough to start with `/ship --from=implement`.

Shortcuts only when the human explicitly asks for them:
- `/ship --from=implement <task>` — skip dialogue + plan (only for obvious single-action fixes)
- `/ship-auto` — never the default; only use if the human says so

### Per-task flow
1. `bd update <id> --claim` (atomic, sets `in_progress` + assignee)
2. `/ship <id>` (runs dialogue → plan → plan-review → …)
3. Stop at the plan gate for human approval
4. `/ship` continues: implement → tests → commit → code-review → fix → push → `bd close`

**Never** `git push` or `bd close` manually outside of `/ship`. `/ship` is the single atomic chokepoint that enforces quality gates and "lands the plane" for every roadmap item.

## Plan-review tier selection

Not every task needs a 9-way plan review. The rough rubric:

| Priority / shape | Plan-review tier |
|---|---|
| P1 invariant-bearing / foundational / keystone | **Trident** or **Triple** (3-4 reviewers) |
| P2 straightforward feature or bug | **Single** (Claude only) or **Dual** |
| P3 nit / cleanup / docs | **None** (just plan, no review) |

Code-review tier is separate and is biased bigger per global CLAUDE.md (Trident ~half the time). See the `review-tiers` skill for the full rubric.

**When unsure, pick the bigger tier.** An extra review is cheap; a shipped bug in foundational code is not.

## Open policy work

`hash-thing-3au` tracks the ongoing design of crew coordination, /ship defaults, and plan-review gating. Amend this file as the policy firms up.
