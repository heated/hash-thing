# Edward's AI Coding Setup

Repo-local snapshot of the setup currently used around hash-thing. This file is intentionally descriptive, not aspirational. It captures what is visible on disk from this worktree on 2026-04-12.

Path choice: this lives in the repo as `ai-coding-setup.md` so it is shareable, versioned, and does not mutate `~/.claude/`.

## Layer Model

The setup has five layers:

1. Global Claude layer in `~/.claude/`
2. Project layer in this repo (`CLAUDE.md`, `.agents/`, `.beads/`, `.ship-notes/`)
3. Worktree + seat layer in `.claude/worktrees/*`
4. Terminal/runtime layer (`cmux`, Ghostty, shell env)
5. Execution patterns on top of that (`/ship`, trident review, bd workflow, manual multi-seat operation)

The important pattern is that the global layer teaches habits, the project layer overrides policy, the worktree layer gives each agent an identity, and the runtime layer makes the multi-agent workflow physically usable.

## 1. Global Claude Layer

### `~/.claude/CLAUDE.md`

This is the always-loaded instruction file. It is opinionated and operational, not just style guidance. The major sections are:

- Override + autonomy: user instructions win, bias to action, use bash freely when needed
- Push ownership up: every prompt should offer to own one level higher in abstraction
- Attention header: a mandatory directive header for every response
- Communication + writing style: dense, fast, low-fluff, plain-text file refs
- File lookup rules: narrow-first search, prefer Glob/Grep over broad shell scans
- Workflow preferences: invoke `/diagram` when it would help
- Self-improvement: propose concrete `CLAUDE.md` edits when patterns are discovered
- Bias to action: test/run/validate immediately rather than narrating
- Merge-surface estimation: prefer `git merge-tree` over guessing from commit counts
- beads (`bd`) workflow: actor identity, claim-before-work, queue discipline
- cmux usage: read-screen, send, browser, and env vars like `CMUX_SURFACE_ID`
- Language triggers: `diagram`, `dialogue`, `loopy`, `ralph`, `ensemble`, `clear`
- Clear agents / multi-model analysis: headless Claude/Codex/Gemini runs and synthesis
- Parallelization rules
- Review tier defaults
- Gate tiers: when to pull the human in
- Poll-don't-stop loop when blocked
- Thrashing recognition

This file is effectively the behavioral kernel for every Claude session.

### `~/.claude/settings.json`

Current verified settings:

- Model: `opus`
- Default permissions mode: `bypassPermissions`
- Sandbox: disabled
- Status line command: `~/.claude/statusline.sh`
- Voice: enabled
- Auto-memory: disabled
- Effort level: `medium`
- Hooks: `{}` at the user-global level right now
- Enabled plugin: `document-skills@anthropic-agent-skills`

The allowlist is broad and tuned for Edward's normal workflow: `git`, `rg`, `find`, npm/test commands, `WebSearch`, specific `WebFetch` domains, and a few MCP tools (`sentry`, `arch-logs`).

### Global slash commands

Current files in `~/.claude/commands/`:

- `capture-skill.md`: turn discovered workflows into reusable config/skills
- `code_review.md`: standard review flow
- `code_review_critical.md`: adversarial review flow
- `compushar.md`: commit/push/PR shortcut
- `diagram.md`: render a Mermaid diagram via the external `diagram` command
- `fix-the-things.md`: parallel detection + sequential fixing loop
- `investigate.md`: multi-agent bug investigation
- `meta-learn.md`: analyze transcript history for process improvements
- `meta-review.md`: synthesize multiple reviews into one plan
- `orchestrate.md`: plan/delegate complex work
- `ship.md`: end-to-end task workflow
- `team_three_review.md`: 6-agent cross-model review
- `trident-code-review.md`: 9-agent code review
- `trident-plan-review.md`: 9-agent plan review

These are personal commands. Hash-thing mirrors the project-relevant subset into `.agents/commands/` so Codex and Gemini can use the same workflow prompts.

### Global skills

Current verified skills in `~/.claude/skills/`:

- `adversarial`: attack your own conclusion before shipping it
- `creating-skills`: how to create a Claude skill
- `creating-slash-commands`: how to create/edit slash commands
- `ensemble`: parallel Claude + Codex + Gemini synthesis
- `human-copy`: write less-LLM-sounding prose
- `steelman`: construct the strongest version of each position before judging
- `using-cmux`: cmux CLI + concepts + integration patterns

## 2. Project Layer: hash-thing

### `CLAUDE.md` / `AGENTS.md`

In this repo, `AGENTS.md` is a symlink to `CLAUDE.md`. Same content, one source of truth.

The project file adds the crew operating model:

- use `bd` / beads for issue tracking
- keep a `gate-sensitivity: medium` bar
- treat the repo as multi-agent by default
- maintain a named seat model (`flint`, `cairn`, `onyx`, `ember`, `spark`, plus singular `mayor`)
- no central orchestrator; every seat owns claim -> implement -> review -> push -> close
- infra/process beads are human-gated
- work in short chunks and re-poll `bd ready`
- park real design decisions, but do not over-park internal implementation choices
- land work to `origin/main` before declaring a bead done

It also records repo-specific policies around worktree drift, CLAUDE edits, landing etiquette, and dry-queue scout work.

### `.agents/`

`.agents/` is the project-local agent surface. It exists so Claude Code, Codex, and Gemini can all read the same project prompts without depending on Edward's home directory.

Verified contents:

- `.agents/commands/`: `code_review.md`, `code_review_critical.md`, `diagram.md`, `ship.md`, `trident-code-review.md`, `trident-plan-review.md`
- `.agents/skills/`: `mayor/SKILL.md`, `review-tiers/SKILL.md`
- `.agents/README.md`: explains why these are copies, not symlinks, and how to refresh from `~/.claude/`

This is one of the main enablers for "real trident" review across models.

### `.beads/`

This worktree has a local `.beads/`, but it is not the primary tracker store.

Verified files:

- `.beads/actor`: per-worktree seat identity (`cedar` in this worktree)
- `.beads/redirect`: points to `/Users/edward/projects/hash-thing/.beads`
- `.beads/issues.jsonl`, `.beads/interactions.jsonl`, config files

The redirect pattern matters: a worktree keeps its own actor identity while sharing the main repo's tracker state.

### `.ship-notes/`

This repo uses `.ship-notes/` for working notes that should not live under `.claude/` in-project. Right now the tracked anchor is `design-ledger.md`; other plans/reviews/handoffs are typically gitignored working files.

### Worktree structure

Codex/Claude/Gemini seats live under `.claude/worktrees/` as sibling worktrees, one per seat/session. Current snapshot from this machine:

- `codex-cedar -> cedar`
- `codex-satyr -> satyr`
- `composed-stargazing-sutherland -> flint`
- `hidden-snacking-parrot -> cairn`
- `lively-waddling-scott -> onyx`
- `sleepy-wiggling-fountain -> spark`
- `sparkling-baking-cookie -> cairn`
- `vast-leaping-allen -> ember`
- `wiggly-nibbling-trinket -> mayor`

The naming pattern is mixed:

- human-readable random worktree names from `git worktree add`
- explicit `codex-<seat>` worktrees created by `scripts/launch-codex-crew.sh`

## 3. Runtime / Terminal Layer

### Ghostty

Verified `~/.config/ghostty/config`:

- shell command: `/usr/local/bin/fish`
- working directory default: `/Users/edward/projects`
- font: `SF Mono`
- font size: `10.5`

This is the base terminal app underneath cmux.

### cmux

Verified binary path:

- `/Applications/cmux.app/Contents/Resources/bin/cmux`

Verified local support directories:

- `~/.cmuxterm`
- `~/Library/Application Support/cmux`
- `~/Library/Application Support/com.cmuxterm.app`

Important note on drift: global `~/.claude/CLAUDE.md` still says the cmux source repo lives at `~/cmux-repo`, but that path does not exist on this machine snapshot. The installed app is present; the source checkout is not.

cmux concepts, per the installed `using-cmux` skill:

- Window -> Workspace -> Pane -> Surface -> Panel
- `CMUX_WORKSPACE_ID` and `CMUX_SURFACE_ID` identify where the current agent is running
- `cmux read-screen`, `cmux send`, `cmux send-key`, `cmux notify`, and browser panes are the core automation primitives

### Session spawning

There is no central orchestrator process. Sessions are launched manually or by helper scripts.

Project helper:

- `scripts/launch-codex-crew.sh <seat> [--workspace <ref>] [--exec] [--dry-run]`

That script:

- creates or reuses `.claude/worktrees/codex-<seat>`
- writes `.beads/actor`
- writes `.beads/redirect` back to the repo root
- writes a seat-specific `.codex-crew-prompt.md`
- launches Codex in interactive mode by default
- exports `GIT_EDITOR=true` / `EDITOR=true` to avoid interactive git editor stalls

## 4. Multi-Agent Architecture

### Crew model

hash-thing runs as a peer crew, not a lead-agent tree. The model is:

- one seat per worktree
- any non-mayor seat can take any ready bead
- the mayor is a singular observer/gate-commenter seat
- ownership is serialized only by bead claim, not by role

### Review architecture

The review stack has tiers:

- single review
- dual / triple review
- trident review

"Trident" here means 9-way review across three models and three lenses. The cost reason is explicit in project docs: moving from all-Claude trident to Claude + Codex + Gemini reduces Claude token burn while preserving review diversity.

### `/ship`

`/ship` is the main end-to-end workflow. The global `ship.md` describes it as:

- route
- dialogue
- plan
- human plan gate
- plan review
- implement
- code review
- fix
- push

Commit cadence is inline, not "one big commit at the end."

## 5. Issue Tracking: beads / bd

This repo uses beads 1.0+ with embedded Dolt.

Working rules:

- export `BEADS_ACTOR` at session start
- `bd ready` to find work
- `bd update <id> --claim` before touching code
- `bd close <id>` only after the work is landed on `origin/main`
- actor identity lives in each worktree's `.beads/actor`
- tracker state is shared through the repo-root `.beads/`

## Workflow Fit Snapshot: Gastown vs Gas City vs Current/Minimal

This snapshot was added during `hash-thing-kqy6.1.3` on 2026-04-14 to compare likely adoption shapes for this repo's crew workflow.

### Bottom line

The best near-term shape for hash-thing is a hybrid biased toward the current/minimal loop, not a wholesale port to Gas Town.

Gas Town is the most mature and opinionated option, but its native loop is Mayor-led and role-heavy. Gas City is concrete enough to evaluate as real software, but it is still more of an orchestration SDK than a finished workflow answer. For this repo, Gas City is the only serious next-substrate candidate, and only as a narrow hybrid if hands-on trials remain positive.

### Core loops

Current/minimal loop:
1. Poll `.bin/bd ready`
2. Claim bead
3. Work in a seat worktree
4. Validate
5. Run review tier
6. Land on `main`
7. Close bead
8. Poll again

Gas Town native loop:
1. Start in Mayor
2. Create convoy around beads
3. Sling work to crew or polecats
4. Witness/Deacon/Dogs supervise health and recovery
5. Refinery handles merge discipline
6. Mayor reports status and routes escalations

Gas City native loop:
1. Define a city in `city.toml`
2. Add rigs and providers
3. Start the controller/supervisor loop
4. Create beads or sessions
5. Attach to sessions and let desired state reconcile to running state
6. Use formulas, waits, mail, and orders where needed

### Fit for this repo

Current/minimal:
- best Codex-first fit today
- lowest ceremony
- already aligned with cmux and peer-seat ownership

Gas Town:
- strongest built-in operating model
- biggest workflow change
- weakest fit for "any seat owns the full loop" and "mayor is mostly observer"

Gas City:
- better long-term fit than Gas Town because it is runtime-agnostic and config-first
- concrete enough to evaluate now: active repo, releases, docs, runtimes, beads integration
- still requires us to define our own house style for peer seats, review policy, direct-to-main landing, and cmux supervision

### Adoption shapes

Wholesale Gas Town port:
- highest automation
- highest process overhead
- poor fit unless the explicit goal becomes a Mayor-centered town

Native Gas City adoption:
- medium/high migration cost
- plausible if we want a reusable orchestration substrate
- still requires design work before it feels like hash-thing

Hybrid:
- keep `bd`, worktrees, direct seat ownership, current review tiers, and direct-to-main landing
- borrow only the pieces that replace bespoke glue cleanly
- best fit now

Likely hybrid imports, if trials justify them:
- runtime/provider abstraction
- controller/supervisor for session lifecycle
- declarative config or pack surface
- better health/recovery loops

### Recommendation shape

1. Keep the current/minimal crew loop as the baseline.
2. Do not wholesale-port this repo to Gas Town.
3. Treat Gas City as the only serious next-substrate candidate.
4. If Gas City trials stay positive, prototype a narrow hybrid that preserves peer-seat pull semantics and direct seat ownership.
5. If Gas City feels heavy even in narrow use, tighten the current setup instead of adopting either system wholesale.

Important project convention: Edward is not expected to operate `bd` directly. If a bead action is attributed to a human name, the project docs treat that as a likely missing `BEADS_ACTOR`, not evidence of a human action.

## 6. Browser Automation / MCP

Verified directly in `settings.json`:

- no user-global hooks configured
- MCP-related allowlist entries exist for `sentry` and `arch-logs`

Verified indirectly from local Claude debug/changelog files:

- Claude-in-Chrome support has been used on this machine before
- debug logs show Chrome profiles and the Claude in Chrome extension being detected

What is not directly verified in this repo snapshot:

- a checked-in project `.mcp.json`
- a user-global config file explicitly wiring a `claude-in-chrome` server for this repo

So the conservative statement is: browser automation capability exists in the surrounding Claude install, but the exact current MCP wiring is not documented in this repo and was not reconstructed here from a single authoritative config file.

## 7. What Makes This Setup Distinct

The distinctive parts are not any single tool. It is the composition:

- global prompt kernel in `~/.claude/CLAUDE.md`
- repo-local agent surface in `.agents/`
- worktree-per-seat execution model
- bd as shared issue memory with per-worktree actor identity
- cmux as the physical substrate for running many agents at once
- review workflows designed for cross-model parallelism rather than single-agent heroics

That stack makes the setup feel less like "an AI assistant in one terminal" and more like a lightweight software crew with a shared queue, shared policy, and separate seats.

## 8. Replicating the Setup

Minimum viable reproduction for another repo:

1. Create a strong global `~/.claude/CLAUDE.md` that defines behavior, review tiers, gate policy, and runtime assumptions
2. Add project `CLAUDE.md` plus `AGENTS.md -> CLAUDE.md`
3. Track project-local commands/skills under `.agents/`
4. Use `bd` with per-worktree `BEADS_ACTOR`
5. Run one worktree per seat
6. Use cmux or an equivalent terminal multiplexer so agents have addressable surfaces
7. Land review-passed work directly to `main`

## 9. Known Drift / Open Follow-Ups

- `~/.claude/CLAUDE.md` still references `~/cmux-repo`, but that source checkout is not present here
- browser automation is evidenced by logs, but its active config surface is not captured in one obvious file here
- this file is a snapshot, not a self-updating system; if Edward wants it to stay current, it should either become part of the `.agents/` refresh discipline or get its own upkeep bead
