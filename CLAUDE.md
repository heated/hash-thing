# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

Gate-tier rules (when to pull edward in) live in global `~/.claude/CLAUDE.md` under "Gate Tiers." Default sensitivity applies here; amend the line below if the bar should move project-wide.

```
gate-sensitivity: medium
```

## Agent surface — where project skills and commands live

This project is designed to work with **any of three CLI agents**: Claude Code (the primary seat), Codex Exec, and Gemini CLI. To keep the three in sync:

- **`AGENTS.md`** is a symlink to this file (`CLAUDE.md`). Codex and Gemini auto-load `AGENTS.md`; Claude Code auto-loads `CLAUDE.md`. Same content, one source of truth.
- **`.agents/commands/`** holds the project-local slash-command / prompt definitions that `/ship` and related workflows depend on: `code_review.md`, `code_review_critical.md`, `trident-code-review.md`, `trident-plan-review.md`, `ship.md`, `diagram.md`. These are **copies**, not symlinks into `~/.claude/`, so a fresh `git clone` on another machine has everything needed.
- **`.agents/skills/`** holds the skill definitions the workflows reference, most importantly `review-tiers/SKILL.md` which `/ship` reads to pick its review tier.
- **Refreshing from `~/.claude/`**: if edward updates his global Claude config, re-sync via `.agents/README.md`'s refresh command. Drift is a file the claude-md-edit queue (or any seat noticing stale content) should refile.

**Why this matters for cost**: edward's Claude Code $200/month plan is maxing out. Shifting review workload from all-Claude trident (9 Claude agents) to real-trident (3 Claude + 3 Codex + 3 Gemini) saves ~2/3 of Claude token spend. That only works if Codex and Gemini sessions can read the same project instructions and workflows — which is what this section, the AGENTS.md symlink, and `.agents/` together make possible.

**When invoking Codex or Gemini on this project** (e.g. from `/ship` phase 6 review), the review prompt should point at `.agents/commands/code_review.md` (project-local) rather than `~/.claude/commands/code_review.md` (user-global). The `/ship` skill itself may still hardcode the user-global path — if so, file a bead or fix it inline. Until then, splicing the project-local file into the prompt is sufficient.

## The crew

Named agents working on this repo. `BEADS_ACTOR` draws from a small pool of natural-object words, not confusable with real names.

Auto-assign pool (used by the `claude -w` seat hook): `flint · cairn · onyx · ember · spark`
Explicit-only seat: `mayor` — singular, never auto-assigned, see the mayor skill

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

`mayor` is a **singular, mostly-passive design-observer seat.** Comments on design gates, re-tiers misfiled work, surfaces decisions for edward. Does not claim drafting, planning, bead-restructure, `.ship-notes`, or `src/` work. Not in the auto-assign pool — only invoked explicitly.

**What non-mayor workers need to know:**
- Don't point tasks at mayor. If you think "mayor should own this," you are almost certainly wrong — own it yourself or hand it to a working seat.
- If edward asks for a CLAUDE.md edit while you're mid-task, file it in the `claude-md-edit` queue and spawn a mayor background session to process it (see "Editing CLAUDE.md" below).
- Never spawn a second mayor at the same time — singular seat.

**All mayor process — starter queue, rules, pre-flight, drift-unparking, on-demand-only, anti-patterns — lives in `skills/mayor/SKILL.md`.** That skill is the authoritative source. This section is the minimum stub for non-mayor workers; if the two disagree, the skill wins.

## How the crew runs

**Edward is never at the yoke.** The crew never waits for him at any gate, ever. Don't ask him to approve plans, don't wait for him to respond, don't idle.

At every gate (including design gates):

1. `bd update <id> --status blocked`
2. `bd comments add <id> "<one-line reason — what edward needs to decide>"`
3. Pull the next task off `bd ready` and start a fresh `/ship <id>`
4. Keep cycling — see the dry-queue section below

Design-gate tasks stack up silently in the `blocked` queue for whenever edward next looks. Technical tasks keep flowing through `/ship` end-to-end.

**Design gates are narrower than they look.** If a scout finds two independent implementations of the same bead, do NOT park at a human gate just because there's a choice to make. Review both and pick one yourself. Only genuine user-facing design calls (what the system *is*, not how it's built) warrant the gate. Internal-API / implementation-detail decisions are autonomous. See `hash-thing-52b` for the precedent.

**But "autonomous" ≠ "single reviewer."** Picking internally still needs external validation — a single seat's pick is too easy to be wrong about. Run a **trident code review** on the pick (9-way preferred) or **at minimum a three-way review** (Claude + Codex + Gemini). The review doesn't ask edward to approve; it catches the cases where the picker missed something that a fresh reviewer would spot instantly. The rule is: no one-seat implementation picks land on main without at least 3 independent reviewer perspectives.

### When `bd ready` is dry — **do not stop**

Running out of ember-safe ready beads is not a rotation stop; it's a prompt to do scout-tier work. The crew only stops for pull-the-plug reasons (the remote is down, the tree won't compile, edward says "that's enough"). Never stop just because the ready queue ran dry.

Scout-tier moves in priority order:

1. **Re-scan with fresh eyes.** Some beads you skipped as "design gate" may have a technical half ember can do (stub scripts, infra scaffolding, test harness, doc pass). Re-read the description, not the summary line.
2. **File new beads.** Walk the code, find a weak spot, file it. `grep -rn TODO\|FIXME\|XXX src/` is a 30-second start. Look for: unchecked `unwrap` / `expect` in non-test code, `assert_eq!` with magic numbers, modules without unit tests, shaders without CPU-side validation, places where the bead graph is obviously missing a link.
3. **Improve crew infra.** SPEC.md gaps, CLAUDE.md gaps, `.ship-notes/` templates, missing skills, hygiene fixes (the `.beads/ permissions 0755` warning on every git command is a standing one — `chmod 700 .beads` in the worktree silences it).
4. **Epic decomposition.** Read an epic description (`bd show hash-thing-6gf`), identify the next 2-3 technical sub-beads, file them with dependencies on the epic. Don't do the keystone design decisions — just the boring decomposition.
5. **Scout lane.** If another crew has a parked plan, read it and leave a bead comment with any technical concerns you spotted. Don't touch their code; do contribute review signal.
6. **Transcript review (yls).** When the session has accumulated enough action that transcripts are interesting — read `~/.claude/projects/-Users-edward-projects-hash-thing*/*.jsonl` and file beads for patterns worth fixing. This is one of only a few meta-beads ember can actually advance.
7. **Only after all of the above come up empty** in the same session: write a `.ship-notes/session-summary.md` (gitignored, local) and finish. This should be rare.

**Do not poach other crews' lanes** to stay busy — that's the one move that isn't allowed. Lane discipline still applies. But scout work is cheap, wide, and always available.

### Drift-unparking autonomy

Any agent may autonomously unblock a bead whose park comment lists **no actual design call** under the Gate Tiers rubric. Bug fixes, refactors, naming, on-disk formats, module layout, and plan-file open questions that are just style/scope picks are drift-parked by reflex, not real gates — reopen them and let the owner ship.

Procedure:
1. Re-read the park comment. If the question maps to "ship without asking" in the gate-tier rubric, it's drift.
2. `bd update <id> --status open` and add a comment citing drift-unparking autonomy + the one-line reason the parked question wasn't actually a design call.
3. Do not claim work on someone else's bead as part of unparking — just unblock so the original owner can resume.

Mayor should sweep for drift-parked beads during any invocation. Workers may unpark their own beads when they realize the park was reflexive.

## Editing CLAUDE.md

**Agent-initiated edits** to project `CLAUDE.md` or `~/.claude/CLAUDE.md` are queued, not direct-committed and not gated on edward. File a bead labeled `claude-md-edit`, status `blocked`, with the proposed diff + a one-line rationale in the description. Also spawn a mayor background session via `Agent(subagent_type=general-purpose, prompt="act as mayor — process claude-md-edit queue")` so the queue doesn't stall. Continue other work; do not wait.

**Human-initiated edits** (edward says "change CLAUDE.md, do X") are direct — any agent executes without queueing.

## Commit cadence

Commit at every natural boundary — plan file written, first test green, helper extracted, review comment addressed, design gate reached. Push after each commit.

## Landing to main — don't let feature branches accumulate

**Feature branches are not a staging area.** Per edward 2026-04-11: "are you just not merging to main? you should be." Any work that's tested and ready belongs on `origin/main`, not parked on a long-lived `worktree-*` or crew branch.

**The default lifecycle for any seat:**

1. Claim the bead (`bd update <id> --claim`) on your worktree branch.
2. Implement / review / fix on the worktree branch — normal `/ship` flow, this part doesn't change.
3. **Before declaring the bead closed, land it on main.** Either fast-forward `origin/main` or create a merge commit to it. If the tree hasn't moved, fast-forward. If it has, pull main first, resolve locally, push main forward. No PR dance, no waiting for edward.
4. `bd close <id>` fires *after* main has the work, not after the worktree branch does.

**What this means in practice:**

- `git push origin HEAD:main` is a routine operation. Main is not protected.
- If your branch and main diverged — pull main, merge or rebase it in, resolve conflicts, push main. Do not leave the worktree branch ahead of main "for later."
- If two worktrees diverged on the same file (the `hash-thing-52b` pattern), don't park at a gate — review both, pick one, land the pick. The losing branch rebases over the landed pick when it's next touched.
- Multi-seat collisions resolve by whoever sees them first: any seat can land any other seat's worktree-ready commit to main, as long as it's actually ready. Lane discipline applies to *authorship*, not to the landing step.

**Exceptions (the only times a worktree branch stays ahead of main):**

- Work in progress that isn't review-passed yet — expected; that's what branches are for.
- Design-gate parking — the bead is `blocked`, not ready; the branch goes quiet until unparked.
- A seat found a genuine user-facing design question and is waiting on edward — rare, always a bead comment first.

**What this does NOT mean:** We're not adopting GitLab/GitHub PR review as a gate. The `/ship` review tiers (dual/triple/trident) still run; the crew still cross-reviews. The difference is that "push passes" → "land on main" is one step, not two.

The prior divergence backlog (18 commits on `worktree-vast-leaping-allen`, 9 on `worktree-sleepy-wiggling-fountain`, both ahead of main) was a bug, not a feature. 52b-A is the cleanup for that specific instance.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, complete all steps below. Work is NOT complete until the reviewed commits are on `origin/main`.

**Workflow:**

1. **File issues for remaining work** — Create beads for anything that needs follow-up.
2. **Run quality gates** — Tests, linters, builds. Everything the affected service requires.
3. **Update issue status** — Close finished work, update in-progress items.
4. **LAND ON MAIN** — this is mandatory and is the actual completion step, not just a branch push:
   ```bash
   git fetch origin
   git merge origin/main           # or rebase, if you prefer a linear history
   # resolve conflicts, run tests again
   git push origin HEAD             # update your worktree branch first (fallback record)
   git push origin HEAD:main        # then fast-forward main
   git status                       # MUST show "up to date with origin"
   ```
   See the "Landing to main" section above for the rationale and multi-seat conventions.
5. **Clean up** — Clear stashes, prune remote branches.
6. **Verify** — All changes committed AND landed on main AND `bd close <id>` done.
7. **Hand off** — Provide context for next session in `.ship-notes/` if the work has open follow-ups.

**Rules:**
- Work is NOT complete until `origin/main` has it. Pushing to your worktree branch alone is an intermediate step, not the finish line.
- NEVER stop before landing — that leaves work stranded locally or on a feature branch.
- NEVER say "ready to push when you are" — you land main yourself.
- If landing main fails, resolve and retry until it succeeds. If you hit a genuine merge conflict you can't resolve, file a bead with the conflict surface and park at a design gate — don't just leave the branch dangling.

