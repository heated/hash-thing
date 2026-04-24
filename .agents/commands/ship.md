# Ship

End-to-end task workflow: route → dialogue → plan → **plan-gate (human)** → plan-review → implement → code-review → fix → push. **Commit happens inline at every natural milestone** (plan marker → each implementation unit → each fix group) — there is no single end-of-ship commit. See "Commit cadence" below.

**Critical ordering:** the human plan-gate comes **BEFORE** plan review, not after. The human approves the plan first; the review runs on the approved plan; review deltas get folded back and presented as a delta to the already-approved plan. This prevents wasting reviewer cycles on a plan the human would have rejected anyway.

**Usage:**
```
/ship [task description or Notion link]
/ship --from=plan [task]           # skip dialogue, start at plan
/ship --from=implement [task]      # skip plan, just build it
/ship --from=review                # review current branch
/ship --from=fix                   # fix existing review findings
/ship --from=preview               # spin up dev servers only
/ship --from=push                  # push current branch
```

---

## Commit cadence

`/ship` commits at every natural milestone, **not** just at the end. A single invocation should produce multiple commits, each reflecting a complete logical unit. This gives us:

- **Bisectability** — every milestone is a green, working state
- **Granular history** — reviewers can walk the change in tractable steps
- **Safe resume** — `--from=implement` picks up from `HEAD` cleanly regardless of where the previous session stopped
- **Park/handoff safety** — if the agent blocks or hands off mid-`/ship`, the committed history is the authoritative state

**Commit points** are injected inline into each phase. There is no "commit phase" at the end — the old Phase 5 is now a shared reference doc for commit hygiene, not a sequential step:

| When | What | Flag |
|---|---|---|
| Phase 2 (plan written) | `--allow-empty` plan marker with the ~100-word plan summary in the message body. Plan files live in `.ship-notes/` which is gitignored, so the marker is how the plan lands in history. | `--allow-empty` |
| Phase 4 (implement) | One commit per logical unit as it lands green. Minimum: one commit for the phase; typically one per module + one per test group. Never let an un-committed green state slip into the next unit. | — |
| Phase 7 (fix) | One commit per fix group (grouped by file or concern). | — |
| Phase 8 (push) | `git push` carries all accumulated commits in a single push. | — |

**What counts as a "logical unit"** — default shapes, use judgment:

- A new module with its first passing test
- A migration step that leaves tests green (e.g., "add new API" is one unit, "migrate callers" is another)
- A self-contained refactor that passes the existing test suite without behavioral change
- Tests added for existing code (one commit per file or per concern)

**Do NOT wait** until the entire implementation is done to commit. If Phase 4 produces five logical units, it should produce five commits. A single monolithic end-of-ship commit is an anti-pattern — it loses bisectability, makes review harder, and forces resume-from-implement to re-do work that's already green.

**Resume semantics.** `/ship --from=implement` picks up from `HEAD` — whatever was committed on the branch is the authoritative state, regardless of which `/ship` session put it there. If you're resuming, run `git log --oneline $BASE_BRANCH..HEAD` first to see what's already in.

**Plan-gate park.** When parking at the plan gate (Phase 2b), the plan marker commit is enough — no code has been written yet. On resume, Phase 3 runs against the plan file pointed at by the metadata; the marker commit is the git-level proof that the plan existed at park time.

**Trident/plan-review interactions.** Reviews themselves never commit. Fix rounds (Phase 7) commit once per fix group. A trident review that surfaces three fix groups across two files produces 1–2 commits, not three.

Message format (trailers, `--no-verify`, matching style) is shared across all commit points — see Phase 5 "Commit hygiene" below.

---

## Resolve constants (do this first, before routing)

Resolve in this order, first match wins:

**`NOTES_DIR`** — review artifacts, metadata, plan files
1. `$SHIP_NOTES_DIR` env var
2. `ship.notes_dir` key in project `CLAUDE.md` (`grep -m1 '^ship.notes_dir:'`)
3. `$(git rev-parse --show-toplevel)/.ship-notes` if inside a git repo (must be in `.gitignore` — add it if not)
4. `~/at/arch/notes` (legacy arch default)

**`BASE_BRANCH`** — what to diff/push against
1. `$SHIP_BASE_BRANCH` env var
2. `git rev-parse --abbrev-ref origin/HEAD 2>/dev/null` (e.g. `origin/main`)
3. `origin/main`

**`REPO_KIND`** — determines whether arch-specific helpers run
- `arch` if `$(git rev-parse --show-toplevel)` is inside `~/arch/` AND `~/arch/.runtime/start-dev.sh` exists
- `generic` otherwise

Echo resolved values once at session start so the human can see what you picked.

---

## Beads integration (optional, auto-detected)

If the first arg to `/ship` matches a bd issue id (e.g. `hash-thing-5bb.4`, `bd-42`) AND `bd show <arg>` exits 0, treat it as `BD_ID`:

1. **On entry** (before Phase 0 routing): `bd update $BD_ID --claim`. If the claim fails because already claimed by a different actor, stop and report — don't overwrite.
2. **On successful Phase 8 push**: `bd close $BD_ID`. If push fails or any earlier phase aborts, leave the claim in place (agent retains ownership until resolution).
3. **On abort/handoff**: leave a `bd comments add $BD_ID` note describing state so the next actor can resume.

If `BD_ID` is not set, behavior is unchanged.

---

## Phase 0: Route

Determine the right entry point based on input:

| Input | Entry Point |
|-------|-------------|
| Notion link or pasted task description | Phase 1 (Dialogue) |
| "plan this", ambiguous/open-ended task | Phase 1 (Dialogue) |
| "just do X", obvious single-action fix | Phase 4 (Implement) — skip dialogue and plan |
| `--from=<phase>` flag | Jump directly to that phase |
| Unclear | Ask: "Want me to plan this out or just go ahead?" |

Derive `BRANCH_ID` from the task:
- If Notion link: extract task title slug
- If branch already exists: use branch name
- Otherwise: derive from task description (e.g., `es/add-investment-filter`)

Create metadata file at `$NOTES_DIR/ship-{BRANCH_ID}.md`:
```markdown
# Ship: {BRANCH_ID}

Branch: _(not yet created)_
Started: {timestamp}

## Preferences
_(filled during dialogue)_

## Plan
_(filled during plan phase)_

## Review Findings
_(filled during review phase)_

## Summary
_(filled at finish)_
```

---

## Phase 1: Dialogue

**Goal:** Fully understand what we're building before writing any code.

1. **Explore the codebase** — search for relevant files, understand current patterns
2. **Interview the human.** Use `AskUserQuestion` liberally. Surface:
   - Requirements and scope — is this the right size? Should it be split?
   - Assumptions you're making so the human can correct early
   - Tradeoffs and edge cases the human may not have considered
3. **Ask preferences:**
   - **Plan review tier:** trident / triple / dual / single / none — runs automatically in Phase 3 **after** the human approves the plan in Phase 2b. **Default comes from the bead priority table in the project `CLAUDE.md` / `AGENTS.md` "Reviews" section** — not free-form agent pick. If `$BD_ID` is set, read `bd show $BD_ID` for the priority and map via that table. If unsure, go *up* a tier; the cost of an extra review is negligible vs a shipped bug in invariant-bearing code. Only downgrade with explicit rationale noted in the metadata file. Never skip plan review on P0/P1 without the human's explicit say-so.
   - **Preview:** spin up dev servers on completion? (Agent decides which service based on what changed.)
4. **Record preferences** in the metadata file

**Exit condition:** Human says "do it", "go", "looks good", or equivalent.

---

## Phase 2: Plan

**Goal:** Write a concrete implementation plan that the human will gate in Phase 2b.

1. **Deep exploration** — read all relevant files, trace code paths, understand the change surface
2. **Write the plan** to `$NOTES_DIR/plan-{BRANCH_ID}.md`:
   - Files to modify/create (with what changes)
   - Key decisions and why
   - Risks or open questions
   - Testing approach
   - The interview (questions asked, answers received) so context isn't lost
3. **Show the human ~100 words** in conversation — key decisions, files touched, tradeoffs. Ask for approval explicitly: "approve the plan and I'll run plan review on it next".
4. **Update metadata file** with plan reference
5. **Commit a plan marker.** The plan file itself is in `.ship-notes/` (gitignored) so it can't be committed directly. Instead, create an empty commit with the ~100-word plan summary in the message body — this lands the plan in git history for bisectability, park safety, and resume correctness:
   ```bash
   git commit --allow-empty --no-verify -m "plan: {BRANCH_ID}

   <paste the ~100-word plan summary here>

   Plan file: $NOTES_DIR/plan-{BRANCH_ID}.md (gitignored)"
   ```
   If this is the first commit on the branch, first create the feature branch: `git checkout -b {BRANCH_ID}`. This plan marker is also the commit that Phase 2b's park path leaves behind — see "Plan-gate park" in Commit cadence at the top of this file.

---

### Phase 2b GATE: Wait for explicit human approval (BEFORE plan review)

**The human plan-gate comes BEFORE plan review, not after.** The human approves the plan first; review runs on the approved plan; review deltas come back as a delta to the already-approved plan. This prevents wasting reviewer cycles on a plan the human would have rejected anyway, and prevents the human from being asked to gate a fresh plan they've never seen.

After writing the plan, **stop and wait for explicit human approval**. Only the human can approve a plan; crew do not approve each other's plans, and plan-review agents do not approve plans. This overrides "autonomy over confirmation" — the plan gate is a real gate, not a defensive prompt.

The human may adjust the plan, ask questions, or reject it. Update the plan with any adjustments and re-present until approved.

**While waiting, don't idle.** Per project policy:

```bash
bd update <id> --status blocked
bd comments add <id> "plan ready for review — waiting on human approval"
```

Then pull the next task off `bd ready` and start a fresh `/ship` on it. When the human approves, resume the blocked task with `/ship --from=plan-review <id>`.

---

## Phase 3: Plan Review

Runs automatically once the human has approved the plan in Phase 2b, using the **plan review tier** chosen in Phase 1.

- **Trident:** 8-way review of `$NOTES_DIR/plan-{BRANCH_ID}.md` (see `/trident-plan-review`)
- **Triple:** 2 Claude (standard + critical) + 1 Codex + 1 Gemini (standard each) + synthesis
- **Dual:** 1 Claude + 1 Codex (standard)
- **Single:** Claude only
- **None:** Skip

**No second human gate.** Phase 2b is the only plan-stage human interaction in /ship. After plan review:

1. Read the synthesis and the underlying reviews.
2. Decide which deltas to adopt vs reject. Use judgment grounded in the human's dialogue answers — when reviewers contradict choices the human already made, hold the line; when reviewers find real bugs, scope creep, or contradictions, fold them in.
3. Update the plan file with the adopted deltas, including a `## Trident deltas` table that records what was adopted, what was rejected, and why.
4. **Proceed directly to Phase 4 (Implement).** Do not pause, do not re-present, do not ask for approval again. The human's Phase 2b approval covers the reviewed plan as long as you're exercising judgment consistent with the dialogue.

The only exception: if review surfaces a finding that genuinely contradicts something the human said in Phase 1 dialogue (e.g., reveals their request is impossible as stated), abort to the human with a clear "this contradicts X — how do you want to proceed?" — but treat that as an abort, not a routine re-gate.

---

## Phase 4: Implement

1. **Ensure on a feature branch:**
   - If already on a feature branch, continue
   - If on the base branch (master/main), create branch: `git checkout -b {BRANCH_ID}`
   - Update metadata with branch name
   - **If `REPO_KIND=arch`:** append `{"branch": "{BRANCH_ID}", "workspace": "$CMUX_WORKSPACE_ID", "started": "{timestamp}"}` to `~/arch/.claude/crew/branch-owners.jsonl` (create if missing). This lets Rita route GitLab findings to the right workspace. Skip in `generic` repos.

2. **Implement the changes** described in the plan (or in `$ARGUMENTS` if skipping to implement). **Commit as you go** — not once at the end. Every logical unit that reaches green is its own commit. See "Commit cadence" at the top of this file for what counts as a logical unit; the short version:
   - New module + first passing test → one commit
   - Migration step that keeps the suite green → one commit
   - Self-contained refactor with no behavior change → one commit
   - A group of tests added for existing code → one commit per file/concern

   Before starting the next unit, verify the working tree is clean: `git status --porcelain` should be empty. If it isn't, you skipped a commit — go back and land it. Each commit should pass the per-service checks from step 5 **at the commit point**, not just at the end of the phase. If you're about to touch a file, commit any pending green state first.

   On resume (`/ship --from=implement`), run `git log --oneline $BASE_BRANCH..HEAD` first to see what's already landed and continue from the next un-committed unit.

3. **Write tests** — Load `/test-writing-step`. Follow the domain rubric to decide test type and coverage. Write tests as part of implementation, not as a separate trailing phase. Tests committed alongside the code they cover is preferred over a separate "add tests" commit — but if tests lag, one-commit-per-test-file is the fallback shape. If skipping tests, note why in the metadata file.

5. **Validate** — run checks only for changed services:
   - **If `REPO_KIND=arch`:** detect changed services from `git diff --stat` and run the corresponding per-service check:
     - `src/common/` → `cd src/common && npm run typecheck && npm run test && npm run lint`
     - `src/portal/server/` → `cd src/portal/server && npm run check`
     - `src/portal/client/` → `cd src/portal/client && npm run check`
     - `src/admin/server/` → `cd src/admin/server && npm run check`
     - `src/admin/client/` → `cd src/admin/client && npm run check`
     - These can be slow — only check what changed
   - **If `REPO_KIND=generic`:** infer from repo type:
     - Rust (`Cargo.toml`): `cargo build`, `cargo test`, `cargo clippy -- -D warnings`
     - Node (`package.json`): `npm test`, `npm run lint` (if scripts exist)
     - Go (`go.mod`): `go build ./...`, `go test ./...`, `go vet ./...`
     - Python (`pyproject.toml`): run whatever the project CLAUDE.md specifies
     - Fall back to: whatever the project CLAUDE.md `ship.validate` key says, else just `git status` and note no validation command configured
   - Fix any issues before proceeding

6. **Run formatter** on changed files: arch → `prettier --write`, Rust → `cargo fmt`, Go → `gofmt -w`, else whatever the project CLAUDE.md `ship.format` key says

---

## Phase 5: Commit hygiene (shared reference — not a sequential step)

This phase is **not** a step you "run" — it's the shared rulebook for every `git commit` invocation triggered by Phase 2 (plan marker), Phase 4 (implementation milestones), and Phase 7 (fix groups). `/ship` no longer has a single monolithic commit phase at the end of implementation; see "Commit cadence" at the top of this file.

When any phase commits:

1. **Stage specific files**, not `git add -A`. Per-file staging avoids sweeping in stray artifacts and makes the commit scope visible at review time.
2. **Commit with `--no-verify`**. Pre-commit hooks are for humans doing ad-hoc commits; `/ship` already ran its own validation in Phase 4 step 5, and hook flakes must not block a green logical unit from landing.
3. **Match recent commit message style** — check `git log --oneline -5` on the current branch (or `git log --oneline -5 $BASE_BRANCH` if the branch is fresh). The first line should fit the style already present in the repo.
4. **First commit on the branch only:** add trailers to the commit message body:
   ```
   NOTION=<link if available>

   TEST=<test plan>
   ```
   If nothing actionable to test, use `TEST=passes CI`. Subsequent commits on the same branch do not repeat these trailers.
5. **Always** update `$NOTES_DIR/ship-{BRANCH_ID}.md` with the new commit hash after committing — this is how resume / park / handoff recovers the running log of what's landed.
6. **Message shape by phase:**
   - Plan marker (Phase 2): `plan: {BRANCH_ID}` + ~100-word summary in body + `--allow-empty --no-verify`
   - Implementation milestones (Phase 4): `<scope>: <logical unit>` — e.g. `sim: add WorldMutation enum` or `octree: hash-cons compaction path`
   - Review fixes (Phase 7): `fix: address review findings — <group>` where `<group>` names the file or concern
7. **Keep commits atomic and green.** Every commit `/ship` produces should pass the Phase 4 step 5 validation at its own commit point — never land red on the branch. If validation for a unit is flaky, split the unit smaller until each sub-unit is green on its own.

---

## Phase 6: Review

Load the `review-tiers` skill, compute the tier from the diff, and log the result in the metadata file. Then run the matching review:

### Trident (6-way)

Run `/trident-code-review` — 6 agents (2 Claude + 3 Codex + 1 Gemini) + 2 synthesis (standard + critical; evolutionary not synthesized per hash-thing-2kkt). Full details in that command.

### Triple (4-way)

2 Claude (standard + critical) + 1 Codex (standard) + 1 Gemini (standard) + 1 synthesis. If Gemini fails, proceed with the other 3.

**Context gathering:**
```bash
git fetch origin "${BASE_BRANCH#origin/}"
git log --oneline "$BASE_BRANCH"..HEAD
git diff "$BASE_BRANCH"...HEAD --stat
git diff "$BASE_BRANCH"...HEAD
```

**Build prompt files:**
```bash
mkdir -p notes/.tmp/ship-{BRANCH_ID}
cat .agents/commands/code_review.md > notes/.tmp/ship-{BRANCH_ID}/review-standard.md
cat .agents/commands/code_review_critical.md > notes/.tmp/ship-{BRANCH_ID}/review-critical.md
echo -e "\n\n---\n\n# CONTEXT\n\nBranch: ...\nCommits: ...\nDiff: ..." >> notes/.tmp/ship-{BRANCH_ID}/review-standard.md
echo -e "\n\n---\n\n# CONTEXT\n\nBranch: ...\nCommits: ...\nDiff: ..." >> notes/.tmp/ship-{BRANCH_ID}/review-critical.md
cp notes/.tmp/ship-{BRANCH_ID}/review-standard.md notes/.tmp/ship-{BRANCH_ID}/review-standard-gemini.md  # Gemini needs workspace-local file
```

**Launch 4 agents in parallel:**

1. **Claude Standard** — Agent tool:
   > Read notes/.tmp/ship-{BRANCH_ID}/review-standard.md and follow the instructions. READ-ONLY. Write to `$NOTES_DIR/CR-{BRANCH_ID}-Claude-Standard-{TIMESTAMP}.md`

2. **Claude Critical** — Agent tool:
   > Read notes/.tmp/ship-{BRANCH_ID}/review-critical.md and follow the instructions. READ-ONLY. Write to `$NOTES_DIR/CR-{BRANCH_ID}-Claude-Critical-{TIMESTAMP}.md`

3. **Codex Standard** — Bash (`run_in_background: true`). Always redirect stdin to `/dev/null` — codex waits on stdin for EOF before running and hangs indefinitely if bash leaves stdin open (pc95):
   ```bash
   env -u CLAUDECODE codex exec --full-auto --skip-git-repo-check \
     "Read notes/.tmp/ship-{BRANCH_ID}/review-standard.md and follow the instructions. Write to $NOTES_DIR/CR-{BRANCH_ID}-Codex-Standard-{TIMESTAMP}.md" \
     < /dev/null 2>&1 | tee notes/.tmp/ship-{BRANCH_ID}/codex.log
   ```

4. **Gemini Standard** — Bash (`run_in_background: true`):
   ```bash
   gemini -m gemini-3-pro-preview --yolo \
     "Read notes/.tmp/ship-{BRANCH_ID}/review-standard-gemini.md and follow the instructions. Write to $NOTES_DIR/CR-{BRANCH_ID}-Gemini-Standard-{TIMESTAMP}.md" \
     < /dev/null 2>&1 | tee notes/.tmp/ship-{BRANCH_ID}/gemini.log
   ```

**Collect:** Wait for all 4 (don't block on Gemini if it fails). If an output file is missing but the log has review content (>500 bytes), copy the log as the output file (Codex/Gemini sometimes write to stdout). Verify output >500 bytes. Consolidate and deduplicate into metadata file.

### Dual (2-way)

1 Claude standard + 1 Codex standard. Same context gathering and prompt build as triple, but only launch 2 agents.

### Single

Claude only via Agent tool with standard code review prompt.

### None

Skip to Phase 7.

---

## Phase 7: Validate & Fix

Read review findings (from review pack synthesis or individual reviews).

For each blocker and important issue:
1. **Read the actual code** at the referenced location
2. **Trace the logic** — understand why the reviewer flagged it
3. **Classify:**
   - ✅ **VALID** — real issue, needs fix
   - ❌ **FALSE POSITIVE** — not an issue (explain why)
   - ⬇️ **DOWNGRADE** — valid but lower priority
   - ❓ **NEEDS INPUT** — uncertain, requires human judgment on product/business intent

4. **Make the call yourself.** For technical decisions surfaced by reviewers (API compatibility, error handling strategy, performance tradeoffs), investigate the codebase, trace the code paths, and decide. Only escalate to the human for genuine product/business ambiguity — not for things you can verify by reading code or docs. If you're 70%+ confident, just do it and note your reasoning in the metadata.

5. **Fix VALID issues:**
   - Group by file for parallelization
   - Same file → same bucket (sequential)
   - Different files → can parallelize via sub-agents
   - Complex or uncertain → main thread
   - Each fix agent: make the change, run service checks, report

6. **Re-validate** after fixes: run checks for affected services

7. **Commit fixes — one commit per fix group**, not one monolithic "address review findings" lump. Groups come from step 5's file/concern grouping; each group lands as its own `fix: address review findings — <group>` commit with `--no-verify`. A trident review that produces 3 fix groups → 3 commits; a dual review that produces 1 group → 1 commit. If fixes span multiple validation runs, commit after each passing group rather than waiting for all of them. See Phase 5 "Commit hygiene" for message-format rules.

8. **Update metadata** with the fix summary and commit hashes (one row per fix-group commit)

---

## Phase 8: Push

Push carries every commit accumulated by this `/ship` invocation — plan marker, implementation milestones, review-fix groups — in one shot:

```bash
# Pre-push sanity: working tree must be clean (no un-committed green state left behind)
test -z "$(git status --porcelain)" || { echo "refusing to push with dirty tree"; exit 1; }

# Show what's about to land, for the metadata record
git log --oneline "$BASE_BRANCH"..HEAD

git push -u origin HEAD
```

If `git status --porcelain` is not empty, that's a Phase 4 or Phase 7 bug — some logical unit finished green but never got committed. Fix it by going back to the last relevant phase and committing the missing unit, not by sweeping everything into a trailing "final" commit.

**If `BD_ID` is set and push succeeded:** `bd close $BD_ID`. Leave a brief `bd comments add $BD_ID` summary pointing at the branch and listing the commits (`git log --oneline $BASE_BRANCH..HEAD`). The beads task is closed as a function of the push landing — this is the single closing chokepoint per the project CLAUDE.md policy.

Do **NOT** create an MR. The human creates MRs.

---

## Phase 9: Manual Testing

**Only runs when `REPO_KIND=arch`.** Generic repos: skip to Phase 11 (or run `cargo test` / language-native test runner manually if the plan called for it). The rest of this phase assumes arch-labs infrastructure.

1. **Determine minimum service set** from `git diff --stat "$BASE_BRANCH"...HEAD`:

   **UI mode** — pick the minimal set:
   | Changed paths | Flag |
   |---|---|
   | Only `src/portal/` | `--portal` |
   | Only `src/admin/` | `--admin` |
   | Both, or `src/common/` | `--all` |
   | No UI paths (only supporting services) | Skip UI — use `service-ctl.sh` directly |

   **Supporting services** — append `--with=<name>` for each:
   | Changed path | Flag |
   |---|---|
   | `src/updateprocessor/` | `--with=updateprocessor` |
   | `src/activityprocessor/` | `--with=activityprocessor` |
   | `src/scheduled-tasks/` | `--with=scheduled-tasks` |

   Also add supporting services when changes affect async/queued flows that need background processing (e.g., activity creation that triggers update-processor jobs), even if the service code itself wasn't changed.

2. **Start dev servers:**
   ```bash
   bash ~/arch/.runtime/start-dev.sh <worktree-path> --portal [--with=updateprocessor]
   ```
   Supporting services are managed by `service-ctl.sh` (called internally). Locks prevent duplicate instances — if another agent already started a service, the current agent reuses the running instance.

   For supporting-service-only testing (no UI):
   ```bash
   bash ~/arch/.runtime/service-ctl.sh ensure updateprocessor <worktree-path>
   ```

3. **Pick test data** — query the test DB for good object IDs while servers spin up.

4. **Run LLM-driven verification.** Agent decides method based on the change:
   - **UI changes:** Use cmux browser — open preview URLs, login (credentials from project CLAUDE.md), navigate to affected pages, verify the change renders correctly, interact with new UI elements, check for visual regressions.
   - **API/backend changes:** Hit endpoints directly via curl or the dev server, verify response shape and data correctness.
   - **Background processing:** Trigger the relevant flow, then check service logs:
     ```bash
     tail -50 <worktree>/.dev-logs/updateprocessor.log
     ```
   - **Mixed:** Combine as needed.

5. **Report results** — what was tested, what passed, anything suspicious. Note in metadata file.

6. **Leave servers running** for human preview if requested.

---

## Phase 10: Human Preview (if requested)

Only runs if human said "yes" to preview in Phase 1. If not, stop dev servers after Phase 9.

If cmux is available (`$CMUX_WORKSPACE_ID` is set):
- Open preview URLs in cmux browser
- Navigate to the right pages for human testing

Otherwise, tell the human the URLs.

**Wait for human to finish testing.** This is a human gate.

To stop (arch only): `bash ~/arch/.runtime/stop-dev.sh <worktree-path>`. In `generic` repos, stop whatever dev server you started in Phase 9 the same way you started it.

---

## Phase 11: Finish

1. **Write summary** to the metadata file:
   - What was done
   - Review results (tier, findings count, fixes applied)
   - Any caveats or follow-ups

2. **Present NEEDS INPUT items** to the human (if any):
   - Clear description of each issue
   - Why it's uncertain
   - Options: "Fix it" / "Skip it" / "Let me explain more"

3. **Output final status:**
```
════════════════════════════════════════════
SHIP COMPLETE
════════════════════════════════════════════

Branch: {branch}
Commits: {count}
Review: {tier} — {blockers} blockers, {important} important, {false_positives} false positives
Fixes: {count} applied
Push: ✅

Metadata: $NOTES_DIR/ship-{BRANCH_ID}.md
Review: $NOTES_DIR/trident-review-{BRANCH_ID}-pack-{TIMESTAMP}/ (if trident)

NEEDS INPUT: {count or "None"}
════════════════════════════════════════════
```

---

## Handling GitLab Findings (post-push)

When Rita routes a finding to this session (CI failure, reviewer comment, etc.):

| Type | Action |
|------|--------|
| CI failure | Fix it, commit, push |
| Nit / trivial fix | Just do it, commit, push |
| Substantive reviewer comment | Evaluate critically. Fix if it improves the code. If not, draft a reply explaining why — show it to the human |
| Question / clarification | Draft a reply, show to the human |

**Don't blindly acquiesce** to every reviewer suggestion. Push back on suggestions that make things worse or miss context.

**Never comment on GitLab directly.** Draft replies are shown to the human in conversation — the human posts them. Local commits and pushes are fine.

Update the metadata file with each finding and how it was handled.

---

## Error Handling

- **Agent timeout:** Use partial results, note in output
- **Agent failure:** Continue with remaining agents (degrade gracefully)
- **Fix breaks tests:** Investigate, may need to back out fix
- **Service check slow:** Note in output, don't block forever

---

## Persistence

All state lives in `$NOTES_DIR/ship-{BRANCH_ID}.md`. If context is compacted or session restarts:
1. Read the metadata file
2. Check git status and branch state
3. Resume from the last incomplete phase

This is intentionally a plain markdown file. If a structured system (beads, tasks, etc.) is added later, this is what gets replaced.

---

Now begin. Route the task and start the workflow.
