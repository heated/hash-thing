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

## autopilot — the permanent default operating mode

**Edward is never at the yoke.** As of 2026-04-11 the crew runs on autopilot permanently: we never wait for him at any gate, ever. Don't ask him to approve plans, don't wait for him to respond, don't idle. This is the standing default and does not get cleared — assume it even if no one mentions autopilot at session start. (Older docs and comments may say "afk mode"; it's the same thing, renamed because "afk" implies a temporary step-away, which no longer applies.)

At every gate (including design gates):

1. `bd update <id> --status blocked`
2. `bd comments add <id> "<one-line reason — what edward needs to decide>"`
3. Pull the next task off `bd ready` and start a fresh `/ship <id>`
4. Keep cycling — see the dry-queue section below

Design-gate tasks stack up silently in the `blocked` queue for whenever edward next looks. Technical tasks keep flowing through `/ship` end-to-end.

### When `bd ready` is dry — **do not stop**

Running out of ember-safe ready beads is not a rotation stop; it's a prompt to do scout-tier work. The crew only stops for pull-the-plug reasons (the remote is down, the tree won't compile, edward says "that's enough"). Never stop just because the ready queue ran dry.

Scout-tier moves in priority order:

1. **Re-scan with fresh eyes.** Some beads you skipped as "design gate" may have a technical half ember can do (stub scripts, infra scaffolding, test harness, doc pass). Re-read the description, not the summary line.
2. **File new beads.** Walk the code, find a weak spot, file it. `grep -rn TODO\|FIXME\|XXX src/` is a 30-second start. Look for: unchecked `unwrap` / `expect` in non-test code, `assert_eq!` with magic numbers, modules without unit tests, shaders without CPU-side validation, places where the bead graph is obviously missing a link.
3. **Improve crew infra.** SPEC.md gaps, AGENTS.md gaps, `.ship-notes/` templates, missing skills, hygiene fixes (the `.beads/ permissions 0755` warning on every git command is a standing one — `chmod 700 .beads` in the worktree silences it).
4. **Epic decomposition.** Read an epic description (`bd show hash-thing-6gf`), identify the next 2-3 technical sub-beads, file them with dependencies on the epic. Don't do the keystone design decisions — just the boring decomposition.
5. **Scout lane.** If another crew has a parked plan, read it and leave a bead comment with any technical concerns you spotted. Don't touch their code; do contribute review signal.
6. **Transcript review (yls).** When the session has accumulated enough action that transcripts are interesting — read `~/.claude/projects/-Users-edward-projects-hash-thing*/*.jsonl` and file beads for patterns worth fixing. This is one of only a few meta-beads ember can actually advance.
7. **Only after all of the above come up empty** in the same session: write a `.ship-notes/autopilot-session-summary.md` (gitignored, local) and finish. This should be rare.

**Do not poach other crews' lanes** to stay busy — that's the one move that isn't allowed. Lane discipline still applies. But scout work is cheap, wide, and always available.

**Only edward himself can clear autopilot**, and only by explicit statement in-session ("I'm back", "pause autopilot", or similar). Crew broadcasts cannot clear it. If in doubt, stay on autopilot.

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

