# Ship-Auto

Headless, zero-human-interaction variant of `/ship`. Drives a single beads issue from claim to push without any interview, plan gate, or human preview. Escalates to the human only by parking the bd issue with a structured note; never blocks the main thread on an `AskUserQuestion`.

**Usage:**
```
/ship-auto <bd-issue-id>
```

The input is **always** a bd issue id. If no id is provided, fail fast with "ship-auto requires a bd issue id; use /ship for free-form tasks."

---

## Resolve constants (same as /ship)

`NOTES_DIR`, `BASE_BRANCH`, `REPO_KIND` resolve identically to `/ship`. See that file for the resolution order.

Additional defaults locked in by ship-auto (no interview to set them):

- **Plan review tier:** `dual` (1 Claude + 1 Codex)
- **Code review tier:** inherit from `review-tiers` skill based on diff size; override to `dual` minimum
- **Preview:** `never`
- **Manual testing:** `cargo test` / language-native only; no dev server spin-up except in `REPO_KIND=arch`

---

## Beads integration (mandatory)

ship-auto requires `BD_ID`. Behavior:

1. **On entry:** `bd update $BD_ID --claim`. If the claim fails (already claimed by a different actor), stop immediately and report — do not overwrite another agent's claim.
2. **Crew lane check:** if project `CLAUDE.md` defines lane assignments and `$BEADS_ACTOR` is bound to a lane, verify the issue's subsystem matches. If it does not, **release the claim** (`bd update $BD_ID --status open --assignee ""`) and exit with a note like "ship-auto: {issue} is outside {actor}'s lane; release back to pool." This prevents crew cross-talk.
3. **On successful push:** `bd close $BD_ID` with a summary comment referencing the branch/commit.
4. **On any parked state** (escalation): leave the claim **in place** (so nobody else grabs the half-done work), set `bd update $BD_ID --status blocked`, attach an ESCALATION note (see "Escalation" below), and exit.
5. **On crash/abort:** leave a `bd note $BD_ID` describing state so the next actor can resume.

---

## Phase 0: Load task

Do NOT run the /ship routing table. The entry point is fixed.

1. Read the bd issue: `bd show $BD_ID`. Capture title, description, labels, priority, dependencies, and all notes/comments.
2. Read any linked/dependency issues (transitively, one level) for context.
3. Derive `BRANCH_ID` from the bd id + slug, e.g. `onyx/hash-thing-2w5-svdag-max-steps`. Include the actor prefix so parallel crew work is easy to identify on the branch listing.
4. Create metadata file at `$NOTES_DIR/ship-{BRANCH_ID}.md`:
   ```markdown
   # Ship-Auto: {BRANCH_ID}

   BD: {BD_ID} — {title}
   Actor: {BEADS_ACTOR}
   Branch: _(not yet created)_
   Started: {timestamp}
   Mode: ship-auto (headless)

   ## Plan
   _(filled during plan phase)_

   ## Plan Review
   _(filled during plan review)_

   ## Review Findings
   _(filled during code review)_

   ## Summary
   _(filled at finish)_
   ```

---

## Phase 1: ~~Dialogue~~ (SKIPPED)

The bd issue description + comments + linked issues ARE the spec. If after reading them you cannot identify a concrete change surface with >=70% confidence, **park** (see Escalation) instead of guessing.

---

## Phase 2: Plan (no gate)

1. Deep exploration: read all relevant files, trace code paths.
2. Write the plan to `$NOTES_DIR/plan-{BRANCH_ID}.md` with the same structure as /ship's plan (files to modify, decisions, risks, testing approach). There is no interview to include.
3. Update metadata file.
4. **Proceed immediately** to Phase 3. There is no human approval gate in this phase.

---

## Phase 3: Plan Review (automated)

Run a **dual** plan review by default: 1 Claude (standard) + 1 Codex (standard), launched in parallel. Same mechanics as `/ship` Phase 3 but hardwired to dual. Do not wait longer than ~5 min per agent after the fastest peer lands — if one stalls (429 retry loop, etc.), TaskStop it and proceed with the other. Single-agent plan review is acceptable as a fallback, but if BOTH fail, park.

**Interpret review feedback:**

- **Both reviews agree "proceed with adjustments"** → update the plan, proceed to Phase 4.
- **Both reviews agree "fundamental problems" / "reject" / "rewrite from scratch"** → **park** with the review findings attached.
- **Reviews disagree significantly** → if you can resolve the disagreement by re-reading code at >=70% confidence, proceed. Otherwise **park** with both reviews attached.
- **Both reviews find only nits** → proceed without adjustments.

---

## Phase 4: Implement

Same as `/ship` Phase 4. Ensure you're on the feature branch (create from `BASE_BRANCH` if needed). Implement the plan. Write tests per the `/test-writing-step` rubric.

If an unexpected state arrives (merge conflict, failing unrelated test, missing file, type error you can't classify at 70%+ confidence within 2 tries), **park** instead of guessing or deleting state.

---

## Phase 5: Validate

Run language-native checks only for changed files (same as /ship's generic-repo path):

- Rust (`Cargo.toml`): `cargo build`, `cargo test`, `cargo clippy -- -D warnings`
- Node: `npm test`, `npm run lint` (if scripts exist)
- Go: `go build ./...`, `go test ./...`, `go vet ./...`
- Else: whatever project CLAUDE.md `ship.validate` key says, or none.

Run `cargo fmt` / `prettier --write` / language formatter on changed files.

If validation fails and you can fix it in 1-2 focused edits at >=70% confidence, fix it and re-run. Otherwise **park** with the failure output attached.

---

## Phase 6: Commit

Same as `/ship` Phase 6:
1. Stage specific files (not `git add -A`).
2. Commit with `--no-verify`.
3. Match recent commit message style (`git log --oneline -5`).
4. First commit on branch: add `BD=$BD_ID` and `TEST=` trailers.

---

## Phase 7: Code Review (automated)

Load `review-tiers` skill, compute the tier from the diff, but **override floor to `dual`** — ship-auto never runs `none` or `single` review. Run the review (same mechanics as /ship Phase 6).

---

## Phase 8: Validate & Fix findings

Same as `/ship` Phase 7, with the following overrides:

- **Classification confidence threshold is 70% for technical calls, 95% for product/scope calls.**
- **For any finding classified `NEEDS INPUT`:** do NOT present to a human. Instead, **park** the whole task with that finding attached to the ESCALATION note. (Rationale: if we push code with a NEEDS-INPUT finding unresolved, we're shipping something we're not sure about. Parking is cheaper than reverting.)
- All VALID blockers/importants get auto-fixed, grouped by file for parallelism. Commit fixes with `fix: address review findings`.
- Re-run validation after fixes.

---

## Phase 9: Push

```bash
git push -u origin HEAD
```

Then `bd close $BD_ID` with a summary comment:
```
Landed via ship-auto.
Branch: {branch}
Commit: {commit}
Review: {tier} — {blockers} blockers, {important} important, {fixes} fixes applied
```

Do NOT create an MR.

---

## Phase 10: ~~Manual testing~~ (conditional)

- **Generic repos:** `cargo test` / language-native test runner only. Already run in Phase 5.
- **REPO_KIND=arch + UI change:** run the reduced manual test flow from /ship Phase 9 (pick minimal service set, verify via cmux browser). **But do not block on a human preview** — verify LLM-driven, report in metadata, park if the verification itself is uncertain.

---

## Phase 11: ~~Human Preview~~ (SKIPPED)

Never runs. Any dev servers spun up in Phase 10 are stopped at the end of Phase 10.

---

## Phase 12: Finish

1. Write summary to metadata file (same as /ship Phase 11).
2. Write summary as a bd comment on the (now-closed) issue.
3. **No NEEDS INPUT presentation.** If the run made it to Phase 9, everything was handled.
4. Output final status:
   ```
   ════════════════════════════════════════════
   SHIP-AUTO COMPLETE
   ════════════════════════════════════════════

   BD: {BD_ID}
   Actor: {BEADS_ACTOR}
   Branch: {branch}
   Commits: {count}
   Review: {tier} — {blockers} blockers, {important} important, {fixes} applied
   Push: ✅
   BD: closed

   Metadata: $NOTES_DIR/ship-{BRANCH_ID}.md
   ════════════════════════════════════════════
   ```
5. **Next issue:** print `bd ready --assignee "" --limit 3` so the next loop knows what's next. Do NOT automatically start a new ship-auto; leave that to the outer loop.

---

## Escalation — how to park a task

When any phase hits something that genuinely needs human judgment (product call, architectural pivot, review-finding you can't classify at the confidence threshold, unexpected state), park as follows:

1. **Do not commit broken/incomplete code.** If there are uncommitted changes, either:
   - `git stash push -u -m "ship-auto park: $BD_ID"` if the changes are salvageable, OR
   - `git restore .` if they're not.
   Note the choice in the escalation block.
2. **If a branch was created:** leave it; note the branch name in the escalation.
3. **Set the bd issue status:** `bd update $BD_ID --status blocked`. Leave `assignee=$BEADS_ACTOR` in place so the parked work has an owner.
4. **Attach an ESCALATION note** with `bd note $BD_ID`:
   ```
   ESCALATION — needs human

   From: {BEADS_ACTOR}
   Phase: {which phase parked: dialogue-skip / plan / plan-review / implement / validate / code-review / fix}
   Timestamp: {ISO timestamp}

   Question: {one concise sentence — what do I need to know to proceed?}

   Options the human can pick:
   - A) {concrete option with consequences}
   - B) {concrete option with consequences}
   - C) {third option if obvious, else "something else — tell me"}

   Context:
   - What I was trying to do: {1-2 sentences}
   - What I tried: {actions taken, bullet list}
   - What I found: {the blocker, as concrete as possible — file:line, error output, review excerpt}
   - Branch: {branch name or "none"}
   - Stash: {stash ref or "none"}
   - Metadata: {path to ship-{BRANCH_ID}.md}

   Confidence: {your estimate, %}
   ```
5. **Exit cleanly.** Print the park summary to the main thread:
   ```
   ════════════════════════════════════════════
   SHIP-AUTO PARKED: {BD_ID}
   ════════════════════════════════════════════
   Phase: {phase}
   Reason: {one line}
   See: bd show {BD_ID} (latest note)
   ════════════════════════════════════════════
   ```

The human sweeps parked tasks with `bd list --status blocked`. When the human resolves the question (via comment or by unblocking the issue), the next ship-auto run picks it up from where it left off (Phase 2/4/7 depending on where it parked).

---

## What ship-auto will NEVER do

- Call `AskUserQuestion` (there is no human in the loop).
- Wait on a plan-approval gate.
- Show plans / diffs / findings to the human in conversation and block.
- Spin up dev servers for manual preview.
- Auto-close a bd issue whose code review flagged NEEDS INPUT.
- Claim a task outside the current actor's crew lane.
- Bypass the plan review or the code review. These run regardless.

---

## Error handling

Same as /ship, plus:
- **Stuck background agent:** peek output file at fastest-peer + ~3 min; TaskStop on visible retry loops; never block the whole run on a single stuck reviewer.
- **bd daemon unresponsive:** park the task with a "bd infrastructure failure" escalation; do not proceed without a working issue tracker.
- **Branch collision with another actor:** detect via `git fetch` before Phase 4 branch creation; if the target branch name exists on origin under a different actor's prefix, pick a unique suffix.

---

Now begin. Read the bd issue, confirm lane, and start Phase 0.
