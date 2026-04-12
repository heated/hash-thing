---
name: review-tiers
description: Select the right code review tier based on diff stats and risk signals
---

# Review Tier Selection

Select the right code review tier based on diff stats and risk signals. Use when deciding review tier for /ship, /code_review, Rita patrols, or any code review.

## Tiers

| Tier | Agents | Target % |
|------|--------|----------|
| **trident** | 9 (3 models x 3 lenses) + 3 synthesis | 40% |
| **triple** | 2 Claude (standard + critical) + 1 Codex (standard) + 1 Gemini (standard) + 1 synthesis | 40% |
| **dual** | 1 Claude + 1 Codex (standard) | 20% |

Gemini is rate-limited and fails ~50% of the time. If Gemini fails, proceed with remaining results — don't retry.

## Selection Algorithm (calibrated 2026-04-07, n=183 reviewable MRs)

### Step 1: Compute diff stats

```bash
git diff origin/master...HEAD --stat
git diff origin/master...HEAD --numstat
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

### Default bias

When in doubt, pick higher tier.
