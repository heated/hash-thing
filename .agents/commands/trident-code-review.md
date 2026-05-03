---
name: trident-code-review
description: "Trident Code Review: Six-Agent Code Review (2-3-1)"
---

# Trident Code Review: Six-Agent Code Review (2-3-1)

Run parallel code reviews with a Codex-heavy six-review mix: Claude runs two lenses (standard / critical), Codex runs the three canonical lenses (standard / critical / evolutionary), and Gemini runs a single standard pass. Per hash-thing-2kkt, the evolutionary-Claude lens is dropped (Codex evolutionary covers the angle at lower cost) and the evolutionary synthesis step is skipped. Output lives in a review pack folder under `~/at/arch/notes/`.

**Usage:**
- `/trident-code-review` — Review current branch with all six agents
- `/trident-code-review [context]` — Review with additional context: `$ARGUMENTS`

---

## Critical Rule: READ-ONLY

**This is a read-only command.** All six review agents and both synthesis agents must ONLY:
- Read existing files (source code, prompt files, related context)
- Write their single designated output file to the review pack folder

They must NEVER:
- Commit or push to git
- Modify any existing files
- Create files outside the review pack folder or the tmp staging directory
- Run destructive commands of any kind

Enforce this in every launch message. Agents that try to commit, modify source files, or take actions beyond writing their review output are violating the contract.

---

## Architecture

```
Main Agent (Opus 4.6)
├── Git operations (once)
├── Test suite (once)
├── Generate context document
├── Identify related context files for agents
├── Write prompt files for Codex/Gemini (file-reference approach)
├── Launch 6 review agents
│   ├── 2x Claude — standard, critical
│   ├── 3x Codex — standard, critical, evolutionary
│   └── 1x Gemini — standard only
├── Wait and collect all 6
├── Quality gate — verify output files exist and have substance
├── Launch 2 synthesis Agents (standard, critical — no evolutionary synthesis)
├── Open synthesis docs
└── Clean up tmp staging directory
```

After all 6 complete, launch 2 synthesis agents (standard + critical), then open the synthesis docs. Evolutionary-Codex output is surfaced as-is without a synthesis pass (single source; nothing to synthesize).

---

## Agent Configuration

| Agent | Type | Lens | Command |
|-------|------|------|---------|
| Claude | Agent | Standard | Agent tool with standard prompt |
| Claude | Agent | Critical | Agent tool with critical prompt |
| Codex | Bash | Standard | `env -u CLAUDECODE .bin/codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read {prompt_file}..." < /dev/null 2>&1 \| tee {log}` |
| Codex | Bash | Critical | same pattern |
| Codex | Bash | Evolutionary | same pattern |
| Gemini | Bash | Standard | `.bin/gemini -m gemini-3-pro-preview --yolo "Read {prompt_file}..." 2>&1 \| tee {log}` |

**IMPORTANT:** All prompt files MUST be written inside the workspace (e.g., `notes/.tmp/`) — NOT `/tmp/`. Gemini CLI sandboxes file reads to workspace directories only.

---

## Instructions

### Phase 1: Gather All Context (Main Agent Does This Once)

Run these commands to build the complete context:

```bash
# Branch info
git branch --show-current

# Determine base branch
BASE_BRANCH=$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main)
git fetch origin "${BASE_BRANCH#origin/}"

# Merge-base against the repo default branch
git merge-base HEAD "$BASE_BRANCH"

# Commits on this branch
git log --oneline "$BASE_BRANCH"..HEAD

# Full diff (not just stat)
git diff "$BASE_BRANCH"...HEAD

# File change summary
git diff "$BASE_BRANCH"...HEAD --stat
```

Derive `REVIEW_ID` from the branch name, including enough path context to prevent collisions:
- `es/add-investment-filter` -> `add-investment-filter`
- `fix/auth-handler` -> `auth-handler`
- `es/investment-perf-fixes` -> `investment-perf-fixes`

If the branch name is unclear, ask the user what to use as the review namespace.

### Phase 1b: Identify Related Context

Look for files that would help reviewers ground their feedback:
- Sibling files in changed directories (e.g., `CLAUDE.md`, existing tests, config files)
- Related modules or projects with similar patterns
- Any files referenced or imported by the changed code

Build a `CONTEXT_NOTE` string listing these paths, e.g.:
> "For additional context, you may also read these related files: `src/common/server/routes/CommonRoutes.ts`, `src/portal/server/src/App.ts` (route registration)."

If no relevant context is found, set `CONTEXT_NOTE` to empty.

### Phase 2: Run Test Suite (Main Agent Does This Once)

Detect which services were changed by looking at the diff:

```bash
git diff "$BASE_BRANCH"...HEAD --stat
```

Run checks only for changed services:

- **`src/common/`** changed: `cd src/common && npm run typecheck && npm run test && npm run lint`
- **`src/portal/server/`** changed: `cd src/portal/server && npm run check` (runs typecheck, test, and lint — may take a while)
- **`src/portal/client/`** changed: `cd src/portal/client && npm run check`
- **`src/admin/server/`** changed: `cd src/admin/server && npm run check`
- **`src/admin/client/`** changed: `cd src/admin/client && npm run check`
- If only common changed, just run common checks
- If no service-specific changes, skip tests

Run sequentially, capture all output. Any failures are noted but don't block the review — reviewers should see the current state.

### Phase 3: Generate Context Document

Create workspace temp directory:
```bash
mkdir -p notes/.tmp/trident-{REVIEW_ID}
```

Write `notes/.tmp/trident-{REVIEW_ID}/team-review-context.md` with all gathered information:

```markdown
# Code Review Context

## Branch Information
- **Branch:** {branch name}
- **Review ID:** {REVIEW_ID}
- **Base:** {BASE_BRANCH} at {merge-base hash}
- **Latest Commit:** {HEAD hash}

## Test Results (Baseline)
- Common typecheck: {PASS/FAIL/SKIPPED}
- Common tests: {PASS/FAIL/SKIPPED}
- Common lint: {PASS/FAIL/SKIPPED}
- Portal server: {PASS/FAIL/SKIPPED}
- Portal client: {PASS/FAIL/SKIPPED}
- Admin server: {PASS/FAIL/SKIPPED}
- Admin client: {PASS/FAIL/SKIPPED}

## Commits on This Branch
{git log output}

## Files Changed
{git diff --stat output}

## Full Diff
{If diff is small: embed here}
{If diff is large: "The diff is {N} lines. Read it from notes/.tmp/trident-{REVIEW_ID}/full-diff.txt"}
```

### Phase 3b: Smoke Check External CLIs

Before writing prompt files or launching background review agents, verify the
repo-local wrappers can start. This catches broken Node/npm environments before
the review silently degrades after launch.

```bash
CODEX_AVAILABLE=1
GEMINI_AVAILABLE=1
printf '%s\n' "$CODEX_AVAILABLE" >/tmp/trident-{REVIEW_ID}-codex-available
printf '%s\n' "$GEMINI_AVAILABLE" >/tmp/trident-{REVIEW_ID}-gemini-available

if ! CODEX_NO_HANG_REMINDER=1 .bin/codex --version >/tmp/trident-{REVIEW_ID}-codex-version.txt 2>&1; then
    CODEX_AVAILABLE=0
    printf '%s\n' "$CODEX_AVAILABLE" >/tmp/trident-{REVIEW_ID}-codex-available
    echo "Codex CLI preflight failed; do not launch Codex reviewers." >&2
    cat /tmp/trident-{REVIEW_ID}-codex-version.txt >&2
    echo "Fallback: run the review with available Claude/Gemini agents and record the Codex outage in the synthesis/audit." >&2
fi

if ! .bin/gemini --version >/tmp/trident-{REVIEW_ID}-gemini-version.txt 2>&1; then
    GEMINI_AVAILABLE=0
    printf '%s\n' "$GEMINI_AVAILABLE" >/tmp/trident-{REVIEW_ID}-gemini-available
    echo "Gemini CLI preflight failed; do not launch Gemini reviewers." >&2
    cat /tmp/trident-{REVIEW_ID}-gemini-version.txt >&2
    echo "Fallback: run the review with available Claude/Codex agents and record the Gemini outage in the synthesis/audit." >&2
fi
```

### Phase 4: Build the Three Review Prompts

Generate a timestamp:
```bash
date +%Y%m%d-%H%M
```
Save for report filenames.

**For Codex and Gemini**, use the file-reference approach to avoid shell escaping issues. Write each agent's full prompt to a tmp file rather than passing it inline.

First, concatenate the canonical review prompts with the context document. Claude uses two lenses (standard + critical); the evolutionary lens is Codex-only:

```bash
# Standard prompt (used by Claude, Codex, Gemini)
cat .agents/commands/code_review.md > notes/.tmp/trident-{REVIEW_ID}/trident-review-standard.md
echo -e "\n\n---\n\n**Additional Context:** $ARGUMENTS\n\n---\n\n# CONTEXT DOCUMENT\n" >> notes/.tmp/trident-{REVIEW_ID}/trident-review-standard.md
cat notes/.tmp/trident-{REVIEW_ID}/team-review-context.md >> notes/.tmp/trident-{REVIEW_ID}/trident-review-standard.md

# Critical prompt (used by Claude, Codex)
cat .agents/commands/code_review_critical.md > notes/.tmp/trident-{REVIEW_ID}/trident-review-critical.md
echo -e "\n\n---\n\n**Additional Context:** $ARGUMENTS\n\n---\n\n# CONTEXT DOCUMENT\n" >> notes/.tmp/trident-{REVIEW_ID}/trident-review-critical.md
cat notes/.tmp/trident-{REVIEW_ID}/team-review-context.md >> notes/.tmp/trident-{REVIEW_ID}/trident-review-critical.md

# Evolutionary prompt (Codex-only — no Claude evolutionary pass)
cat .agents/prompts/CodeReview-Evolutionary.md > notes/.tmp/trident-{REVIEW_ID}/trident-review-evolutionary.md
echo -e "\n\n---\n\n**Additional Context:** $ARGUMENTS\n\n---\n\n# CONTEXT DOCUMENT\n" >> notes/.tmp/trident-{REVIEW_ID}/trident-review-evolutionary.md
cat notes/.tmp/trident-{REVIEW_ID}/team-review-context.md >> notes/.tmp/trident-{REVIEW_ID}/trident-review-evolutionary.md
```

Then, write per-agent prompt files for Codex and Gemini (4 total: 3 Codex, 1 Gemini):

```bash
# Example for standard-codex — repeat the pattern for critical-codex and evolutionary-codex.
cat > notes/.tmp/trident-{REVIEW_ID}/prompt-standard-codex.md <<'PROMPT_EOF'
Read notes/.tmp/trident-{REVIEW_ID}/trident-review-standard.md and follow the instructions.

**THIS IS A READ-ONLY REVIEW.** Your ONLY output is the review file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.

**OUTPUT:** Write your review to `/tmp/trident-{REVIEW_ID}/standard-codex.md`

Use these values: STORY_ID={REVIEW_ID}, MODEL=Codex.
{CONTEXT_NOTE}

Now follow the instructions in the review prompt file.
PROMPT_EOF
```

For Gemini, copy the prompt framework files into the workspace (Gemini now runs only the standard pass):
```bash
cp .agents/commands/code_review.md notes/.tmp/trident-{REVIEW_ID}/
cp .agents/commands/code_review_critical.md notes/.tmp/trident-{REVIEW_ID}/
cp .agents/prompts/CodeReview-Evolutionary.md notes/.tmp/trident-{REVIEW_ID}/
```

For Gemini prompt files, use workspace-local paths for both the framework file and the context document.

### Phase 5: Create Review Pack and Launch All Six Agents

Create the review pack folder:
```bash
PACK_DIR=~/at/arch/notes/trident-review-{REVIEW_ID}-pack-{TIMESTAMP}
mkdir -p "$PACK_DIR"
rm -rf /tmp/trident-{REVIEW_ID}
mkdir -p /tmp/trident-{REVIEW_ID}

CODEX_AVAILABLE=$(cat /tmp/trident-{REVIEW_ID}-codex-available 2>/dev/null || echo 1)
if [ "$CODEX_AVAILABLE" -eq 0 ]; then
    for lens in standard critical evolutionary; do
        cat > "$PACK_DIR/${lens}-codex.md" <<'EOF'
# Codex Review Unavailable

Codex CLI preflight failed before review launch, so this is an explicit outage placeholder rather than a model review.

No Codex findings were produced. Synthesis agents must treat this file as unavailable-provider metadata, not as an affirmative no-findings review.

See `/tmp/trident-{REVIEW_ID}-codex-version.txt` for the captured preflight output and record the outage in the review audit.

Do not infer consensus from this placeholder. If the corresponding Claude or
Gemini review found no issue, the correct interpretation is "no Codex signal was
available," not "Codex agreed." Keep the outage visible in the synthesis summary
and in any final merge/ship note.
EOF
    done
fi

GEMINI_AVAILABLE=$(cat /tmp/trident-{REVIEW_ID}-gemini-available 2>/dev/null || echo 1)
if [ "$GEMINI_AVAILABLE" -eq 0 ]; then
    cat > "$PACK_DIR/standard-gemini.md" <<'EOF'
# Gemini Review Unavailable

Gemini CLI preflight failed before review launch, so this is an explicit outage placeholder rather than a model review.

No Gemini findings were produced. Synthesis agents must treat this file as unavailable-provider metadata, not as an affirmative no-findings review.

See `/tmp/trident-{REVIEW_ID}-gemini-version.txt` for the captured preflight output and record the outage in the review audit.

Do not infer consensus from this placeholder. If the corresponding Claude or
Codex review found no issue, the correct interpretation is "no Gemini signal was
available," not "Gemini agreed." Keep the outage visible in the synthesis
summary and in any final merge/ship note.
EOF
fi
```

Launch all six review agents with `run_in_background: true` for Bash agents. Claude agents use the Agent tool (2 calls in a single message).
Use the `CODEX_AVAILABLE` and `GEMINI_AVAILABLE` values from Phase 3b: launch
only the external providers whose preflight passed, and record skipped providers
in the quality gate and synthesis/audit notes.

**Launch message template** (for Claude subagents — Codex/Gemini use their prompt files from Phase 4):

> Read notes/.tmp/trident-{REVIEW_ID}/trident-review-{lens}.md and follow the instructions.
>
> **THIS IS A READ-ONLY REVIEW.** Your ONLY output is the review file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.
>
> **OUTPUT:** Write your review to `{PACK_DIR}/{lens}-claude.md`
>
> Use these values: STORY_ID={REVIEW_ID}, MODEL=Claude.
> {CONTEXT_NOTE}
> Additional context from the user: {ADDITIONAL_NOTES}
>
> Now follow the instructions in the review prompt file.

---

**Claude agents** (2x) — use the **Agent tool** (parallel subagents):

Launch two Agent tool calls in a single message with the launch template, substituting:
- Standard: `lens=standard`
- Critical: `lens=critical`

(No Claude evolutionary pass — Codex evolutionary covers that angle at lower cost per hash-thing-2kkt.)

---

**Codex agents** (3x) — use Bash with `run_in_background: true`:

**Important:** Use `env -u CLAUDECODE` to strip the environment variable that blocks nested sessions.
If `CODEX_AVAILABLE=0` from Phase 3b, skip this section and note the outage in
the quality gate and synthesis/audit notes.

Each agent reads its prompt from the file written in Phase 4:

**Important:** Create `/tmp/trident-{REVIEW_ID}/` for Codex output and logs (Codex sandbox can't write to PACK_DIR):
```bash
mkdir -p /tmp/trident-{REVIEW_ID}
```

Always redirect stdin to `/dev/null` — codex waits on stdin for EOF before running and hangs indefinitely if bash leaves stdin open (pc95):

```bash
env -u CLAUDECODE .bin/codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read notes/.tmp/trident-{REVIEW_ID}/prompt-standard-codex.md and follow the instructions exactly." < /dev/null 2>&1 | tee /tmp/trident-{REVIEW_ID}/codex-standard.log

env -u CLAUDECODE .bin/codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read notes/.tmp/trident-{REVIEW_ID}/prompt-critical-codex.md and follow the instructions exactly." < /dev/null 2>&1 | tee /tmp/trident-{REVIEW_ID}/codex-critical.log

env -u CLAUDECODE .bin/codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read notes/.tmp/trident-{REVIEW_ID}/prompt-evolutionary-codex.md and follow the instructions exactly." < /dev/null 2>&1 | tee /tmp/trident-{REVIEW_ID}/codex-evolutionary.log
```

---

**Gemini agents** (1x) — use Bash with `run_in_background: true`:
If `GEMINI_AVAILABLE=0` from Phase 3b, skip this section and note the outage in
the quality gate and synthesis/audit notes.

Each agent reads its prompt from the file written in Phase 4:

```bash
.bin/gemini -m gemini-3-pro-preview --yolo "Read notes/.tmp/trident-{REVIEW_ID}/prompt-standard-gemini.md and follow the instructions exactly." < /dev/null 2>&1 | tee /tmp/trident-{REVIEW_ID}/gemini-standard.log
```

---

**Tell the user:**
> "Context prepared. Test results: {service: PASS/FAIL for each changed service}
>
> Review launch:
> - Claude (Standard, Critical)
> - Codex (Standard, Critical, Evolutionary) if `CODEX_AVAILABLE=1`; otherwise placeholder outage files written
> - Gemini (Standard) if `GEMINI_AVAILABLE=1`; otherwise placeholder outage file written
>
> Review pack: `{PACK_DIR}/`
>
> You can continue working. I'll synthesize when complete."

### Phase 6: Wait, Collect, and Quality Gate

- For Claude agents: the Agent tool calls will return when complete.
- For Codex/Gemini agents: use `TaskOutput` with `block: true` for each background Bash task.

Find review files:
```bash
ls -la {PACK_DIR}/*.md 2>/dev/null
```

**Collect Codex output from /tmp:** Codex writes to `/tmp/trident-{REVIEW_ID}/` because PACK_DIR is outside its sandbox. Copy any output files to PACK_DIR:
```bash
CODEX_AVAILABLE=$(cat /tmp/trident-{REVIEW_ID}-codex-available 2>/dev/null || echo 1)
if [ "$CODEX_AVAILABLE" -eq 1 ]; then
    for f in /tmp/trident-{REVIEW_ID}/*codex*.md; do
        [ -f "$f" ] && cp "$f" "{PACK_DIR}/"
    done
fi
```

**Stdout fallback:** Codex/Gemini may write findings to stdout instead of the output file. For each missing pack file, check the corresponding log in `/tmp/` — if it has review content (>500 bytes of non-boilerplate), copy it as the output file. Use the actual agent suffix (`standard-codex`, `critical-codex`, `evolutionary-codex`, `standard-gemini`):
```bash
# Example for the standard codex pass:
if [ ! -f "{PACK_DIR}/standard-codex.md" ] && [ -s "/tmp/trident-{REVIEW_ID}/codex-standard.log" ]; then
    cp "/tmp/trident-{REVIEW_ID}/codex-standard.log" "{PACK_DIR}/standard-codex.md"
fi
```

**Quality gate:** Verify output files exist and have substance:
```bash
wc -c {PACK_DIR}/*.md
```

- Files over 500 bytes: good, proceed to synthesis.
- Files under 500 bytes: likely failed or empty. Note the gap.
- Missing files: check logs, note which agent failed.

Report any gaps to the user before proceeding to synthesis.

Expected files: `standard-claude.md`, `standard-codex.md`, `standard-gemini.md`, `critical-claude.md`, `critical-codex.md`, `evolutionary-codex.md`. (No `evolutionary-claude.md` — that lens was dropped per hash-thing-2kkt.)
Provider-outage placeholder files are expected when a preflight flag is `0`;
they satisfy path availability but must not be counted as successful model
reviews.

### Phase 7: Two Synthesis Agents

Launch **two Agent tool calls** in a single message — one for standard, one for critical. The evolutionary lens is not synthesized: Codex's evolutionary pass is the only source and is surfaced as-is.

**Synthesis-Standard Agent:**
> Read all Standard code reviews for {REVIEW_ID}:
> - `{PACK_DIR}/standard-claude.md`
> - `{PACK_DIR}/standard-codex.md`
> - `{PACK_DIR}/standard-gemini.md`
>
> **THIS IS A READ-ONLY REVIEW.** Your ONLY output is the synthesis file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.
>
> Synthesize into a single document. Focus on:
> 1. Consensus issues — what do 2+ models agree on? (highest confidence)
> 2. Divergent views — where do models disagree? (signal worth examining)
> 3. Unique findings — issues only one pass raised
> 4. Consolidated blockers, important items, and suggestions (deduplicated)
>
> Write to: `{PACK_DIR}/synthesis-standard.md`
> Format as numbered lists. Start with executive summary and merge verdict.

**Synthesis-Critical Agent:**
> Read all Critical code reviews for {REVIEW_ID}:
> - `{PACK_DIR}/critical-claude.md`
> - `{PACK_DIR}/critical-codex.md`
>
> **THIS IS A READ-ONLY REVIEW.** Your ONLY output is the synthesis file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.
>
> Synthesize into a single document. Focus on:
> 1. Consensus risks — failure modes multiple models identified (highest priority)
> 2. Unique concerns — risks only one pass raised (worth investigating)
> 3. The ugly truths — hard messages that recur across models
> 4. Consolidated blockers and production risk assessment
>
> Write to: `{PACK_DIR}/synthesis-critical.md`
> Format as numbered lists. Start with executive summary and production readiness verdict.

(No Synthesis-Evolutionary agent — the evolutionary lens has a single source, `evolutionary-codex.md`, which reviewers read directly.)

### Phase 8: Launch for Review and Clean Up

Once both synthesis docs are written, also open the raw evolutionary-codex review directly (no synthesis for it):

```bash
code "{PACK_DIR}/synthesis-standard.md"
code "{PACK_DIR}/synthesis-critical.md"
code "{PACK_DIR}/evolutionary-codex.md"
```

**Clean up tmp staging directory** — only after confirming review pack files are present:
```bash
# Verify review pack is complete before cleanup
ACTUAL=$(ls {PACK_DIR}/*.md 2>/dev/null | wc -l | tr -d ' ')
if [ "$ACTUAL" -ge 6 ]; then
    rm -rf notes/.tmp/trident-{REVIEW_ID}
fi
```

(Use 6 as the threshold — if at least 6 reviews exist, the tmp files served their purpose.)

Tell the user:
> "Trident code review complete. Two synthesis docs opened plus raw evolutionary:
> - `synthesis-standard.md` — analytical consensus and merge verdict
> - `synthesis-critical.md` — adversarial findings and production readiness
> - `evolutionary-codex.md` — creative opportunities (Codex-only; no synthesis pass)
>
> Full review pack at `{PACK_DIR}/` (6 review slots, with explicit outage placeholders for unavailable providers, plus 2 syntheses)"

---

## Fallback

If an agent fails or isn't available, proceed with successful agents. Note failures in the relevant synthesis doc. If an entire model is unavailable, use the preflight-created placeholder files so synthesis has stable inputs and the outage remains visible.

---

Now begin. Prepare context, run tests, and orchestrate the six-agent trident code review.
