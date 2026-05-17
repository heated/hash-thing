# u4b4 Prototype Review 1

Date: 2026-05-03
Reviewer: `moss`
Input: Quarantine Atlas prototype landed at `03ab866`
Scene: `q` / `--dump-scene quarantine-atlas`
Status: reviewer 1 of 3; does not close `hash-thing-u4b4`

## Protocol

This review evaluates the first playable Quarantine Atlas slice against the
closure question from `hash-thing-u4b4`:

> Does this gameplay loop hang together, or does it devolve into wall-painting /
> hand-editing?

Evidence used:

- Headless scene smoke: `cargo run --bin hash-thing -- 128 --dump-frame /tmp/quarantine-atlas.png --dump-scene quarantine-atlas`
- Implementation inspection of the landed scene/action layer.
- Targeted tests proving scene contents, hazard-lane spread, budget enforcement,
  small-world guard, direct-loader cleanup, and Quarantine-specific legend.

This is an agent prototype review, not a substitute for an interactive human
feel pass. It should count as one written reviewer perspective only.

## Verdict

**Conditional yes: the loop hangs together enough to keep testing.**

The current prototype now has the minimum ingredients for a real containment
loop:

- A large quiet floor with separated protected settlements.
- A live lava/fire source feeding a combustible lane toward the middle
  settlement.
- A finite intervention budget.
- Reusable counter-pattern stamps: barrier, cooling trench, and firebreak.
- Raw left-click carving disabled in Quarantine mode.
- Scene-specific legend text that points the player at stamps/patterns instead
  of hand-editing raw materials.

That combination avoids the worst version of the earlier falsifier. The player
cannot simply free-paint every material or carve perfect one-block fixes without
spending the intended action budget. The scene also starts pointed at the hazard
lane, so the first read is "front to contain" rather than "empty sandbox with
props."

## Distinctness

It is meaningfully different from pure Powder Toy because the player is not
given unlimited painting as the main verb. It is also not Factorio-like yet:
there is no production chain, only triage and containment. The closest familiar
shape is a sparse tower-defense or disaster-response puzzle, but the material
front and reusable voxel stamps are enough to justify another prototype pass.

## Hashlife-Relevant Decision

The hashlife-relevant decision is still present, but only in seed form:

**Where do I spend one of a few interventions on a mostly quiet atlas, knowing
the active front is the expensive region?**

Without large quiet regions being cheap, this should collapse into a small arena
hazard puzzle. Without reusable structures sharing work, the counter-pattern
library is just cosmetic prefabs. The prototype needs a larger next map or
instrumented review to prove those properties are felt, not just asserted.

## Falsifier Check

The prototype no longer obviously fails as hand-editing, because raw carve is
disabled and right-click deploys budgeted stamps. It still may fail as
wall-painting if reviewers discover that the dominant strategy is always:

1. Select barrier.
2. Stamp a straight perimeter across the lane.
3. Wait.

The next review should specifically test whether cooling trench and firebreak
create different tactical outcomes from barrier spam. If not, Quarantine Atlas
needs either more hazard topology or a different objective than "block the
front."

## Required Next Evidence

`u4b4` should remain open. Closure still requires at least two more written
prototype reviewers, with at least two of three total reviewers saying the loop
hangs together.

Suggested prompt for the next reviewers:

1. Use `q` or `--dump-scene quarantine-atlas`.
2. Try to solve the scene using only barrier spam.
3. Try again using cooling trench or firebreak.
4. Report whether the non-barrier choices were meaningfully better in any
   situation.

