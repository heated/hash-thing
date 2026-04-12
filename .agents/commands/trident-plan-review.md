# Trident Plan Review: Nine-Agent Plan Review

Run parallel plan reviews using Claude, Codex, and Gemini — each provider runs three lenses: Standard (analytical), Adversarial (critical), and Evolutionary (exploratory). Output lives alongside the input file in a review pack folder.

**Usage:**
- `/user:trident-plan-review <file>` — Review a plan/document through all nine agents
- `/user:trident-plan-review <file> <additional notes>` — Review with additional context
- `$ARGUMENTS` contains the file path and optional additional notes

---

## Critical Rule: READ-ONLY

**This is a read-only command.** All nine review agents and all three synthesis agents must ONLY:
- Read existing files (the plan, prompt files, related context)
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
├── Parse file + notes from $ARGUMENTS
├── Create review pack folder alongside input file
├── Identify related context files for agents
├── Write prompt files for Codex/Gemini (file-reference approach)
├── Launch 9 agents
│   ├── 3x Claude — via Agent tool (subagents)
│   ├── 3x Codex — via Bash (headless CLI)
│   └── 3x Gemini — via Bash (headless CLI)
├── Wait and collect all 9
├── Quality gate — verify output files exist and have substance
├── Launch 3 synthesis Agents (one per lens)
├── Open synthesis docs
└── Clean up tmp staging directory
```

---

## Instructions

### Phase 1: Parse Arguments

Extract from `$ARGUMENTS`:
- **First argument:** file path (required)
- **Everything after:** additional user notes (optional)

Read the input file. If it doesn't exist or is empty, stop and tell the user.

Derive `REVIEW_ID` from **both the parent directory and filename** (strip path prefix, lowercase, hyphens for spaces):
- `projects/music_fetcher/PLAN.md` -> `music-fetcher-plan`
- `notes/api-redesign-plan.md` -> `api-redesign-plan`
- `plans/Q3 Roadmap.md` -> `q3-roadmap`

This prevents collisions when reviewing identically-named files in different directories.

Determine the directory where the input file lives (`INPUT_DIR`).

### Phase 1b: Identify Related Context

Look for files that would help reviewers ground their feedback:
- Sibling files in the same directory (e.g., `CLAUDE.md`, other docs)
- Related projects with similar patterns (e.g., `movie_fetcher` when reviewing `music_fetcher`)
- Any files referenced in the plan itself

Build a `CONTEXT_NOTE` string listing these paths, e.g.:
> "For additional context, you may also read these related files: `projects/music_fetcher/CLAUDE.md`, `projects/movie_fetcher/PLAN.md` (sibling project with same pattern)."

If no relevant context is found, set `CONTEXT_NOTE` to empty.

### Phase 2: Create Review Pack

Generate a timestamp:
```bash
date +%Y-%m-%dT%H%M
```

Create the review pack folder alongside the input file:
```bash
mkdir -p "{INPUT_DIR}/{REVIEW_ID}-review-pack-{TIMESTAMP}"
```

Example: If reviewing `projects/music_fetcher/PLAN.md`, the pack goes at `projects/music_fetcher/music-fetcher-plan-review-pack-2026-03-10T1423/`.

Store this path as `PACK_DIR`.

### Phase 3: Prepare Prompt Files

Create a tmp staging directory using **absolute path** based on INPUT_DIR:
```bash
STAGING_DIR="{INPUT_DIR}/.tmp/trident-{REVIEW_ID}"
mkdir -p "$STAGING_DIR"
```

**IMPORTANT — Codex sandbox constraints:**
- Codex can only write to: its workdir (`~/arch`), `/tmp`, and `$TMPDIR`
- If PACK_DIR is outside the workdir (e.g. `~/at/arch/notes/...`), Codex can't write there
- Solution: Codex prompt files tell it to write output to `/tmp/trident-{REVIEW_ID}/{lens}-codex.md`, then main agent copies to PACK_DIR after collection
- Also create a workspace-local copy of prompt files so Codex can read them:

```bash
# Workspace-local staging (Codex reads from here)
WS_STAGING="notes/.tmp/trident-{REVIEW_ID}"
mkdir -p "$WS_STAGING"

# Also create /tmp output dir for Codex
mkdir -p /tmp/trident-{REVIEW_ID}
```

**For Codex and Gemini**, use the file-reference approach to avoid shell escaping issues. Write each agent's full prompt to a tmp file rather than passing it inline.

For each of the 6 Codex/Gemini agents, write a prompt file. **Codex prompt files go in BOTH staging dirs** (workspace-local for reading, /tmp output path for writing). **Codex output paths must use `/tmp/`**:

```bash
# Codex prompt — note /tmp output path
cat > "$WS_STAGING/prompt-standard-codex.md" <<'EOF'
Read ~/.claude/agents/PlanReview-Standard.md for your review approach and thinking framework. Then read {INPUT_FILE} — that is the document you are reviewing.

**THIS IS A READ-ONLY REVIEW.** Your ONLY output is the review file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.

**OUTPUT:** Write your review to `/tmp/trident-{REVIEW_ID}/standard-codex.md`

Use these values: PLAN_ID={REVIEW_ID}, MODEL=Codex.
{CONTEXT_NOTE}
{ADDITIONAL_NOTES}

Now follow the instructions in the prompt file.
EOF
```

Gemini prompt files go in the workspace-local staging dir and use PACK_DIR for output (Gemini sandbox is more permissive):
```bash
# Gemini prompt
cat > "$WS_STAGING/prompt-standard-gemini.md" <<'EOF'
...same as Codex but OUTPUT path is {PACK_DIR}/standard-gemini.md...
EOF
```

Copy the prompt framework files into the workspace-local staging (both Codex and Gemini sandbox reads to the workspace):
```bash
cp ~/.claude/agents/PlanReview-Standard.md "$WS_STAGING/"
cp ~/.claude/agents/PlanReview-Adversarial.md "$WS_STAGING/"
cp ~/.claude/agents/PlanReview-Evolutionary.md "$WS_STAGING/"
```

If the input file is outside the workspace, also copy it:
```bash
cp {INPUT_FILE} "$WS_STAGING/plan-input.md"
```

For Gemini prompt files, use workspace-local paths for both the framework file and the input file.

### Phase 4: Launch All Nine Agents

**Launch message template** (for Claude subagents — Codex/Gemini use their prompt files from Phase 3):
> Read {PROMPT_PATH} for your review approach and thinking framework. Then read {INPUT_FILE} — that is the document you are reviewing.
>
> **THIS IS A READ-ONLY REVIEW.** Your ONLY output is the review file specified below. Do NOT commit, push, modify existing files, or take any action beyond writing this single file.
>
> **OUTPUT:** Write your review to `{PACK_DIR}/{lens}-{model}.md` — NOT to the path mentioned in the prompt file.
>
> Use these values: PLAN_ID={REVIEW_ID}, MODEL={MODEL}.
> {CONTEXT_NOTE}
> Additional context from the user: {ADDITIONAL_NOTES}
>
> Now follow the instructions in the prompt file.

---

**Claude agents** (3x) — use the **Agent tool** (parallel subagents):

Launch three Agent tool calls in a single message with the launch template, substituting:
- Standard: `PROMPT_PATH=~/.claude/agents/PlanReview-Standard.md`, `lens=standard`, `MODEL=Claude`
- Adversarial: `PROMPT_PATH=~/.claude/agents/PlanReview-Adversarial.md`, `lens=adversarial`, `MODEL=Claude`
- Evolutionary: `PROMPT_PATH=~/.claude/agents/PlanReview-Evolutionary.md`, `lens=evolutionary`, `MODEL=Claude`

---

**Codex agents** (3x) — use Bash with `run_in_background: true`:

**Important:** Use `env -u CLAUDECODE` to strip the environment variable that blocks nested sessions.

Each agent reads its prompt from the **workspace-local** staging dir (not the absolute INPUT_DIR path). Log files also go to `/tmp/` to avoid tee failures:

```bash
env -u CLAUDECODE codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read notes/.tmp/trident-{REVIEW_ID}/prompt-standard-codex.md and follow the instructions exactly." 2>&1 | tee /tmp/trident-{REVIEW_ID}/codex-standard.log

env -u CLAUDECODE codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read notes/.tmp/trident-{REVIEW_ID}/prompt-adversarial-codex.md and follow the instructions exactly." 2>&1 | tee /tmp/trident-{REVIEW_ID}/codex-adversarial.log

env -u CLAUDECODE codex exec --full-auto -s danger-full-access --skip-git-repo-check "Read notes/.tmp/trident-{REVIEW_ID}/prompt-evolutionary-codex.md and follow the instructions exactly." 2>&1 | tee /tmp/trident-{REVIEW_ID}/codex-evolutionary.log
```

---

**Gemini agents** (3x) — use Bash with `run_in_background: true`:

Each agent reads its prompt from the file written in Phase 3:

```bash
gemini -m gemini-3-pro-preview --yolo "Read notes/.tmp/trident-{REVIEW_ID}/prompt-standard-gemini.md and follow the instructions exactly." 2>&1 | tee /tmp/trident-{REVIEW_ID}/gemini-standard.log

gemini -m gemini-3-pro-preview --yolo "Read notes/.tmp/trident-{REVIEW_ID}/prompt-adversarial-gemini.md and follow the instructions exactly." 2>&1 | tee /tmp/trident-{REVIEW_ID}/gemini-adversarial.log

gemini -m gemini-3-pro-preview --yolo "Read notes/.tmp/trident-{REVIEW_ID}/prompt-evolutionary-gemini.md and follow the instructions exactly." 2>&1 | tee /tmp/trident-{REVIEW_ID}/gemini-evolutionary.log
```

---

**Tell the user:**
> "Plan: {input_file} | Review Pack: {PACK_DIR}
>
> Launched 9 agents (3 Standard, 3 Adversarial, 3 Evolutionary) across Claude, Codex, and Gemini.
>
> You can continue working. I'll synthesize when complete."

### Phase 5: Wait, Collect, and Quality Gate

- For Claude agents: the Agent tool calls will return when complete.
- For Codex/Gemini agents: use `TaskOutput` with `block: true` for each background Bash task.

**Collect Codex output from /tmp:** Codex writes to `/tmp/trident-{REVIEW_ID}/` because PACK_DIR may be outside its sandbox. Copy any output files to PACK_DIR:
```bash
for f in /tmp/trident-{REVIEW_ID}/*-codex.md; do
    [ -f "$f" ] && cp "$f" "{PACK_DIR}/"
done
```

**Quality gate:** Verify output files exist and have substance:
```bash
wc -c {PACK_DIR}/*.md
```

- Files over 500 bytes: good, proceed to synthesis.
- Files under 500 bytes: likely failed or empty. Check the corresponding log in `notes/.tmp/trident-{REVIEW_ID}/` and note the gap.
- Missing files: check logs, note which agent failed.

Report any gaps to the user before proceeding to synthesis.

Expected files: `standard-claude.md`, `standard-codex.md`, `standard-gemini.md`, `adversarial-claude.md`, `adversarial-codex.md`, `adversarial-gemini.md`, `evolutionary-claude.md`, `evolutionary-codex.md`, `evolutionary-gemini.md`.

### Phase 6: Three Synthesis Agents

Launch **three Agent tool calls** in a single message. Each synthesizes one lens across all three models.

**Synthesis-Standard Agent:**
> Read the following three reviews and synthesize them into a single cohesive document:
> - `{PACK_DIR}/standard-claude.md`
> - `{PACK_DIR}/standard-codex.md`
> - `{PACK_DIR}/standard-gemini.md`
>
> These are Standard/Analytical reviews of the same plan from three different AI models. Your job:
> 1. Where do models agree? (highest confidence findings)
> 2. Where do they diverge? (the disagreement itself is signal)
> 3. What unique insights did only one model surface?
> 4. Consolidated questions for the plan author (deduplicated, numbered)
> 5. Overall readiness verdict synthesized across all three
>
> Write your synthesis to: `{PACK_DIR}/synthesis-standard.md`
> Format as a clean, numbered-list-heavy document. Start with an executive summary.

**Synthesis-Adversarial Agent:**
> Read the following three reviews and synthesize them into a single cohesive document:
> - `{PACK_DIR}/adversarial-claude.md`
> - `{PACK_DIR}/adversarial-codex.md`
> - `{PACK_DIR}/adversarial-gemini.md`
>
> These are Adversarial/Critical reviews of the same plan from three different AI models. Your job:
> 1. Consensus risks — what do multiple models flag? (highest priority)
> 2. Unique concerns — risks only one model raised (worth investigating)
> 3. Assumption audit — merge and deduplicate all surfaced assumptions
> 4. The uncomfortable truths — what hard messages recur across models?
> 5. Consolidated hard questions for the plan author (deduplicated, numbered)
>
> Write your synthesis to: `{PACK_DIR}/synthesis-adversarial.md`
> Format as a clean, numbered-list-heavy document. Start with an executive summary.

**Synthesis-Evolutionary Agent:**
> Read the following three reviews and synthesize them into a single cohesive document:
> - `{PACK_DIR}/evolutionary-claude.md`
> - `{PACK_DIR}/evolutionary-codex.md`
> - `{PACK_DIR}/evolutionary-gemini.md`
>
> These are Evolutionary/Exploratory reviews of the same plan from three different AI models. Your job:
> 1. Consensus direction — evolution paths multiple models identified
> 2. Best concrete suggestions — the most actionable ideas across all three
> 3. Wildest mutations — the most creative/ambitious ideas (even if risky)
> 4. Flywheel opportunities — self-reinforcing loops any model identified
> 5. Strategic questions for the plan author (deduplicated, numbered)
>
> Write your synthesis to: `{PACK_DIR}/synthesis-evolutionary.md`
> Format as a clean, numbered-list-heavy document. Start with an executive summary.

### Phase 7: Launch for Review and Clean Up

Once all three synthesis docs are written:

```bash
code "{PACK_DIR}/synthesis-standard.md"
code "{PACK_DIR}/synthesis-adversarial.md"
code "{PACK_DIR}/synthesis-evolutionary.md"
```

**Clean up tmp staging directory** — only after confirming all review pack files are present:
```bash
# Verify review pack is complete before cleanup
EXPECTED=12  # 9 reviews + 3 syntheses
ACTUAL=$(ls {PACK_DIR}/*.md | wc -l | tr -d ' ')
if [ "$ACTUAL" -ge 9 ]; then
    rm -rf notes/.tmp/trident-{REVIEW_ID}
fi
```

(Use 9 as the threshold, not 12 — syntheses are the proof that reviews were consumed. If at least 9 reviews exist, the tmp files served their purpose.)

Tell the user:
> "Trident review complete. Three synthesis docs opened:
> - `synthesis-standard.md` — analytical consensus
> - `synthesis-adversarial.md` — critical findings
> - `synthesis-evolutionary.md` — evolutionary opportunities
>
> Full review pack (12 files) at: `{PACK_DIR}/`"

---

## Fallback

If an agent fails or isn't available, proceed with successful agents. Note failures in the relevant synthesis doc. If an entire model is unavailable (e.g., Codex is down), the synthesis still works with 2 inputs instead of 3.

---

Now begin. Parse the file from $ARGUMENTS and orchestrate the nine-agent trident plan review.
