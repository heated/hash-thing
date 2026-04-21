---
name: review-tiers
description: Select the right code review tier based on diff stats and risk signals
---

# Review Tier Selection

Select the right code review tier based on diff stats and risk signals. Use when deciding review tier for /ship, /code_review, Rita patrols, or any code review.

## Tiers

| Tier | Agents | Target % |
|------|--------|----------|
| **trident (code)** | 2 Claude (standard + critical) + 3 Codex (standard + critical + evolutionary) + 1 Gemini (standard) + 2 synthesis (standard + critical) | 40% |
| **trident (plan)** | 2 Claude (standard + adversarial) + 5 Codex (standard + standard-execution + adversarial + adversarial-dependencies + evolutionary) + 1 Gemini (standard) + 2 synthesis (standard + adversarial) | 40% |
| **triple** | 2 Claude (standard + critical) + 1 Codex (standard) + 1 Gemini (standard) + 1 synthesis | 40% |
| **dual** | 1 Claude + 1 Codex (standard) | 20% |

Per hash-thing-2kkt: the evolutionary-Claude lens is dropped across both trident variants (Codex covers the evolutionary angle at ~1/3 the Claude token cost), and the evolutionary synthesis step is skipped in both (single remaining source doesn't need synthesis).

Gemini is rate-limited and fails ~50% of the time. If Gemini fails, proceed with remaining results — don't retry.

## Selection Algorithm (calibrated 2026-04-07, n=183 reviewable MRs)

### Step 1: Compute diff stats

```bash
BASE_BRANCH=$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main)
git diff "$BASE_BRANCH"...HEAD --stat
git diff "$BASE_BRANCH"...HEAD --numstat
```

Extract: **lines added**, **lines deleted**, **files changed**.

### Step 2: Base tier from lines added

| Lines added | Base tier |
|-------------|-----------|
| >= 75 | Trident |
| >= 50 AND >= 7 files | Trident (spread signal) |
| 19-74 | Triple |
| < 19 | Dual |

### Step 3: Detect risk signals from file paths and MR title

| Signal | Trigger paths/keywords | Risk level |
|--------|----------------------|------------|
| schema-ddl | `sql/schema-changes/`, CREATE/ALTER in SQL paths | HIGH |
| auth-code | `auth/`, `login`, `permission`, `middleware/auth` paths | HIGH |
| infra-security | `.tf` files touching VPC, security groups, RDS, IAM | HIGH |
| api-surface | API schema/route/controller/mutation files | HIGH |
| security-config | SAML, SSO, MFA, credential in title | MODERATE |
| infra-sso | SSO permission set `.tf` files | MODERATE |

### Step 4: Apply risk bumps

- Any HIGH or MODERATE signal on a Dual MR -> bump to Triple
- 2+ HIGH signals on a Triple MR -> bump to Trident
- Never bump down

### Step 5: Apply caps and skips

- Mass deletion (deletions > 5x additions AND > 100 deletions) -> cap at Triple
- Promotion MRs ("promote approved changes to production", "dump-db") -> skip review entirely

### Step 6: Project-specific risk signals (hash-thing)

The generic signals in Step 3 target enterprise webapps. For this game engine, use these instead:

| Signal | Trigger | Risk level |
|--------|---------|------------|
| shader-parity | `*.wgsl` AND `svdag.rs` both in diff | HIGH |
| hashlife-invariants | `hashlife.rs` or `store.rs` compaction/memoization paths | HIGH |
| buffer-layout | `svdag.rs` serialization + `renderer.rs` upload changed together | MODERATE |
| rule-system | `rule.rs` or `margolus.rs` | MODERATE |

Apply the same bump logic as Step 4: any HIGH/MODERATE on Dual -> Triple, 2+ HIGH on Triple -> Trident.

### Step 7: Log review cost (observational)

After each review completes, append a JSONL line to `.ship-notes/review-cost-log.jsonl`:

```json
{"bead":"hash-thing-xxx","tier":"dual","agents":2,"output_bytes":12345,"timestamp":"2026-04-12T14:00:00Z"}
```

This is data collection only — used to calibrate thresholds over time.

### Default bias

When in doubt, pick higher tier.
