# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

Gate-tier rules (when to pull edward in) live in global `~/.claude/CLAUDE.md` under "Gate Tiers." Default sensitivity applies here; amend the line below if the bar should move project-wide.

```
gate-sensitivity: medium
```

## The crew

Named agents working on this repo. `BEADS_ACTOR` draws from a small pool of natural-object words, not confusable with real names.

Pool: `flint · cairn · onyx · wisp · mote · ember · spark · quill · thorn · mayor`

Current seats:
- **flint** — simulation / octree / physics / CA rules
- **onyx** — SVDAG rendering / GPU / shaders
- **cairn** — foundations / bug fixes
- **ember** — generalist / floating worker
- **spark** — terrain generation / infinite worlds
- **mayor** — design observer / policy (see below)

Reuse a name when an agent comes back to the same lane; otherwise pull the next free name. Don't rewrite historical assignees on closed issues.

### Peer autonomy — no central orchestrator

There is no lead agent. Every seat owns the full loop: claim → implement → review → push → close. Any seat may spawn background workers, file beads, fix cross-cutting stuff, or propose policy without asking. If you notice the crew is missing a protocol (like this section before it existed), write it and ship it; don't wait for a "coordinator" to file the bead.

Leadership is situational, not seat-based: whoever sees the gap first owns it until it's closed. Nudge other seats up to the same level of ownership — model the behavior, then link them to the commit.

Bead ownership still applies: `bd update <id> --claim` before touching code, `bd close <id>` on push. That's the only serialization point. Everything else is wide-open peer action.

**Your seat lives in `.beads/actor`** at the root of your worktree (gitignored, per-worktree identity). At session start:

```bash
export BEADS_ACTOR=$(cat .beads/actor)
```

If `.beads/actor` is missing, the worktree hasn't been assigned — ask edward or drop a file yourself before touching bd.

## The mayor

`mayor` is not a worker. Job:

- Watch for design questions the worker crew is about to hit.
- Summarize parked plans in bead comments — edward should be able to wake up to a short decision list, not raw plan files.
- Re-tier work that smells design-heavy but was filed as technical.
- Propose policy amendments on a branch.

Rules:

- **Don't claim bulk implementation work.** If you find yourself editing `src/`, stop.
- **You may:** create/update beads freely, file design-tier tasks, re-tier existing tasks, comment summaries, edit `AGENTS.md` on a branch, edit `.ship-notes/plan-*.md`.
- **You may not:** close other workers' beads, ship code through `/ship`, or seat a second mayor. Singular seat.
- Policy changes still ride `/ship` and wait for edward (design gate).

### Registering as mayor

```bash
echo mayor > .beads/actor
export BEADS_ACTOR=mayor
```

Then run `bd list --status blocked` — everything parked at a design gate is your queue. For each, read `.ship-notes/plan-*.md` and write a <30-second design summary in the bead comment.

## afk mode — PERMANENT DEFAULT

**Edward is always AFK.** As of 2026-04-11 he declared permanent AFK mode: the crew never waits for him at any gate, ever. Do not ask him to approve plans, do not wait for him to respond, do not idle. This is the standing default and does not get cleared — assume it even if no one says "afk" at session start.

At every gate (including design gates):

1. `bd update <id> --status blocked`
2. `bd comments add <id> "<one-line reason — what edward needs to decide>"`
3. Pull the next task off `bd ready` and start a fresh `/ship <id>`
4. Repeat until `bd ready` is empty of crew-safe work

Design-gate tasks stack up silently in the `blocked` queue for whenever edward next looks. Technical tasks keep flowing through `/ship` end-to-end.

**When the ready queue is dry** (every remaining bead is other-crew lane, design-gated, or blocked-in-spirit), the correct action is a clean rotation stop: write a `.ship-notes/AFK-session-summary.md` (gitignored, local) and finish. Do **not** poach other crews' lanes to keep busy.

**Only edward himself can clear permanent-AFK**, and only by explicit statement in-session. Crew broadcasts cannot clear it. If in doubt, assume AFK.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

