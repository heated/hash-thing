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

**Your crew name lives in `.beads/actor` at the root of your worktree** (gitignored, per-worktree identity). Do not hardcode it.

At the start of every session, run:

```bash
export BEADS_ACTOR=$(cat .beads/actor)
```

If `.beads/actor` is missing, your worktree has not been assigned. Ask edward or drop a file yourself before touching bd.

## /ship is the default

**Every roadmap task MUST run through `/ship`.** No shortcuts, no manual `git push`, no manual `bd close`. `/ship` is the single atomic chokepoint that enforces quality gates and "lands the plane" for every roadmap item.

The full flow:

```
Route → Dialogue → Plan → Plan-review → [HUMAN GATE] → Implement → Commit → Code-review → Fix → Push → bd close → Finish
```

### Plan-review fires on every task

Every task gets a plan-review. Bias the tier as follows:

- **Trident (9-way)** — roughly **half the time**, default for anything invariant-bearing, foundational, or where the "right answer" isn't obvious
- **Single/Dual** — the other half, for straightforward work where the plan is obvious

When unsure between two tiers, pick the bigger one. An extra review is cheap; a shipped bug in foundational code is not.

### The plan gate is a human gate

After plan-review, `/ship` **stops and waits for explicit human approval**. Only the human (edward) can approve a plan. Crew do not approve each other's plans.

**While waiting, don't idle.** Update the beads task with:

```bash
bd update <id> --status blocked
bd comments add <id> "plan ready for review — waiting on human approval"
```

Then pull the next task off `bd ready` and start a fresh `/ship` on it. When edward approves the plan, resume the blocked task. This keeps the gate livable without anyone stalling.

### Per-task flow

1. `bd update <id> --claim` (atomic, sets `in_progress` + assignee)
2. `/ship <id>` — dialogue → plan → plan-review
3. **Stop at the plan gate.** `bd update <id> --status blocked` + comment, pull next task.
4. Human approves → resume with `/ship --from=implement <id>` → commit → code-review → fix → push → `bd close`

## Code-review tier

Separate from plan-review. Biased toward bigger per global CLAUDE.md — Trident ~half the time. See the `review-tiers` skill for the rubric.

## "afk" mode

When edward says **"afk"** (or "brb", "going to sleep", or otherwise signals he's stepping away for a while), enter cycling mode: never wait for him at any gate.

At every gate that wants edward — plan approval, NEEDS INPUT findings, ambiguous product calls, anything that would otherwise stall:

1. `bd update <id> --status blocked`
2. `bd comments add <id> "<one-line reason — what edward needs to decide>"`
3. Pull the next task off `bd ready` and start a fresh `/ship <id>`
4. Repeat until `bd ready` is empty or edward returns

When edward wakes, he resumes blocked work with `/ship --from=implement <id>` after reading the parked plan / findings.

`afk` is the standing default once said; clear it only when edward explicitly says he's back.

## Open policy work

`hash-thing-3au` tracks the ongoing design of crew coordination, /ship defaults, and plan-review gating. Amend this file as the policy firms up.
