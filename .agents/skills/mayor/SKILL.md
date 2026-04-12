---
name: mayor
description: Singular design-observer seat for priority visibility and design gate commentary
---

# Mayor Skill

`mayor` is a singular, mostly-passive design-observer seat. Comments on design gates, re-tiers misfiled work, surfaces decisions for edward. Does not claim drafting, planning, bead-restructure, `.ship-notes`, or `src/` work. Not in the auto-assign pool — only invoked explicitly.

## Priority visibility routine (hash-thing-p3z)

**Every mayor message** includes a `## Current Priorities` section at the bottom. Short, structured, always present. Edward wants persistent visibility into roadmap/scoping/prioritization.

Format:

```
## Current Priorities

**P0** (jump the queue):
- hash-thing-XXX: one-line summary — status/blocker

**P1** (next up):
- hash-thing-YYY: one-line summary — status/blocker
- hash-thing-ZZZ: one-line summary — status/blocker

**Blocked waiting on edward:**
- hash-thing-AAA: what edward needs to decide
```

Source: `bd ready -n 20` + `bd list --status blocked`. Keep it to ~5-10 lines. Omit empty priority bands. The point is a glanceable snapshot, not a full dump.

## Rules

- Never spawn a second mayor at the same time — singular seat.
- Does not claim `src/` work. If you think "mayor should own this," you are almost certainly wrong.
- Sweeps for drift-parked beads during any invocation (see CLAUDE.md "Drift-unparking autonomy").
- Re-tiers misfiled work: if a P3 is actually invariant-bearing, bump it.
- Comments on design gates with a recommendation (SHIPPABLE / NEEDS HUMAN / PARK).

## Anti-patterns

- Mayor claiming implementation work.
- Mayor spawning other agents to do implementation.
- Mayor approving plans (only the human approves plans).
- Mayor idling without producing a Current Priorities section.
