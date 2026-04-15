# `.agents/` — project-local agent surface

This directory is the **shared surface** that Claude Code, Codex Exec, and Gemini CLI all read when operating on hash-thing. It exists so a `git clone` on a fresh machine (or a Codex/Gemini session that has never seen edward's `~/.claude/` config) finds the project's review workflows, skills, and slash-command prompts without any out-of-band setup.

## Layout

```
.agents/
├── commands/          # slash-command / prompt definitions
│   ├── code_review.md
│   ├── code_review_critical.md
│   ├── trident-code-review.md
│   ├── trident-plan-review.md
│   ├── ship.md
│   └── diagram.md
├── prompts/           # tracked prompt frameworks used by project-local commands
│   ├── CodeReview-Evolutionary.md
│   ├── PlanReview-Standard.md
│   ├── PlanReview-Adversarial.md
│   └── PlanReview-Evolutionary.md
└── skills/
    └── review-tiers/
        └── SKILL.md   # tier selection algorithm for /ship phase 6
```

These are **copies**, not symlinks into `~/.claude/`. Absolute symlinks break on other machines; relative symlinks break out of the repo. Copies mean the project is self-contained — at the cost of occasional drift that has to be reconciled (see below).

## Invariants

- `AGENTS.md` at the repo root is a symlink to `CLAUDE.md`. Codex and Gemini auto-load `AGENTS.md`; Claude Code auto-loads `CLAUDE.md`. Content is identical because it's the same file.
- `.agents/` is tracked in git (not in `.gitignore`). Every worktree and every fresh clone has it.
- Project-local trident review commands must resolve every review framework they execute from `.agents/`, not `~/.claude/`, so fresh clones and non-Claude seats can run without private global config.

## Refreshing from `~/.claude/`

When edward updates his user-global Claude config (`~/.claude/commands/*`, `~/.claude/skills/review-tiers/`), this directory drifts. To resync:

```bash
# From the repo root of any hash-thing worktree:
cp ~/.claude/commands/code_review.md          .agents/commands/code_review.md
cp ~/.claude/commands/code_review_critical.md .agents/commands/code_review_critical.md
cp ~/.claude/commands/trident-code-review.md  .agents/commands/trident-code-review.md
cp ~/.claude/commands/trident-plan-review.md  .agents/commands/trident-plan-review.md
cp ~/.claude/commands/ship.md                 .agents/commands/ship.md
cp ~/.claude/commands/diagram.md              .agents/commands/diagram.md
cp ~/.claude/agents/CodeReview-Evolutionary.md .agents/prompts/CodeReview-Evolutionary.md
cp ~/.claude/agents/PlanReview-Standard.md     .agents/prompts/PlanReview-Standard.md
cp ~/.claude/agents/PlanReview-Adversarial.md  .agents/prompts/PlanReview-Adversarial.md
cp ~/.claude/agents/PlanReview-Evolutionary.md .agents/prompts/PlanReview-Evolutionary.md

# review-tiers lives in _archived in ~/.claude but is still the authoritative copy:
cp -r ~/.claude/skills/_archived/review-tiers/* .agents/skills/review-tiers/
```

Then `git diff .agents/` to see what drifted, commit with a message like `chore(.agents): refresh from ~/.claude/ snapshot <date>`, land on main.

Drift detection is the crew's responsibility — if any seat notices a command or skill that looks newer in `~/.claude/` than in `.agents/`, either run the refresh above or file a `claude-md-edit`-style bead and let the next session pick it up.

## Why copies and not symlinks

Three failed alternatives, all rejected:

1. **Absolute symlinks into `~/.claude/`** — break on any other machine / fresh clone. Tight coupling between the project and edward's private config.
2. **Relative symlinks out of the worktree** — git either stores the link target literally (breaking on any different machine) or follows it and commits the content (which defeats the "link to a canonical copy" intent).
3. **Submodule pointing at `~/.claude/`** — `~/.claude/` is not a git repo, and making it one would entangle unrelated projects with hash-thing's history.

Copies are the least-bad option: the project is self-contained, git shows drift clearly, and refresh is one `cp` command away.

## Why this exists at all

Recorded here for context that won't age well in a commit message: edward's Claude Code $200/month plan was maxing out in early April 2026. Trident reviews that ran all-Claude (9 Claude agents) were eating budget. Shifting to real trident (3 Claude + 3 Codex + 3 Gemini) cuts Claude token spend roughly in thirds — but only works if Codex and Gemini can *find* the project's review workflows. That's what this directory is.

See `hash-thing-al1` for the bead that created this layout.
