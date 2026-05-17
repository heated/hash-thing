# u4b4 Prototype Review 2

Date: 2026-05-03
Reviewer: `ember`
Input: Quarantine Atlas prototype on `origin/main` after `7768627`
Scene: `q` / `--dump-scene quarantine-atlas`
Status: reviewer 2 of 3; does not close `hash-thing-u4b4`

## Protocol

This review follows the prompt from `notes/u4b4-prototype-1-review.md`:

1. Check whether the prototype is still a budgeted containment loop rather than
   free hand-editing.
2. Inspect whether barrier spam obviously dominates.
3. Inspect whether cooling trench and firebreak create distinct choices.
4. Answer whether the Quarantine Atlas loop hangs together.

Evidence used:

- Fresh scene smoke:
  `cargo run --bin hash-thing -- 128 --dump-frame /tmp/quarantine-atlas-review-2.png --dump-scene quarantine-atlas`
- Visual inspection of `/tmp/quarantine-atlas-review-2.png`.
- Implementation inspection of Quarantine Atlas input routing, pattern stamps,
  map seeding, and tests.
- Existing main-target validation after the `n8hy` merge:
  `cargo test -p hash-thing --bin hash-thing` passed 133 tests.

This is an agent prototype review. It counts as one written reviewer
perspective, not as an interactive human feel pass.

## Verdict

**Weak yes: the gameplay loop hangs together enough to keep Quarantine Atlas as
the active u4b4 prototype.**

The prototype has crossed the minimum line from "sandbox scene" to "game-loop
slice":

- `Q` / `--dump-scene quarantine-atlas` loads a specific hazard-map setup.
- Quarantine mode has a finite budget of 6 interventions.
- Left-click carve is disabled in Quarantine mode, so raw hand-editing is not
  the main verb.
- Right-click deploys the selected counter-pattern, and `1` / `2` / `3` switch
  between barrier, cooling trench, and firebreak.
- The map seed includes settlements, a live lava/fire hazard, and a flammable
  lane toward the middle settlement.

That is enough for "predict, isolate, deploy limited counter-patterns" to be a
real loop rather than a pure Powder Toy sandbox.

## Barrier Spam Check

Barrier spam does not obviously invalidate the prototype yet, but it remains the
main threat.

The barrier stamp is a hard stone/metal obstruction, so it is the most legible
first move. However it is also narrow: it builds a wall line at one `z`, while
the hazard lane and grass/vine threat band span multiple `z` rows. A player
trying to solve the whole lane with barriers likely has to spend several of the
6 interventions.

That gives room for the other tools to matter:

- Cooling trench covers a wider lane slice with water and stone banks, so it
  looks better when the goal is to slow or dampen a front rather than place a
  single hard stop.
- Firebreak replaces a wider patch with sand and clears the layer above it, so
  it looks better when the goal is to remove fuel from the path.

The current prototype therefore passes the "not immediately just perimeter-wall
painting" check, but only weakly. The third review should explicitly test a
barrier-only run against a mixed firebreak/cooling run.

## Hashlife-Relevant Decision

The hashlife-relevant decision is visible but not proven:

**Spend scarce edits only where a large mostly-quiet atlas becomes locally
expensive.**

The playable loop needs large stable regions to be cheap enough that the player
can care about the few active fronts. The current scene expresses that at the
layout level: quiet settlements and terrain are broad, while lava/fire/vine
fronts are localized. It does not yet make the engine advantage felt directly to
the player; a larger map, long-running objective, or perf/heat overlay would make
the thesis sharper.

## Distinctness

This remains distinct from the comparison set in the sketch packet:

- Not Powder Toy, because edits are budgeted and the map gives external stakes.
- Not Factorio, because there is no production chain or throughput factory.
- Not Minecraft, because placement is pattern deployment against material
  dynamics, not block construction.
- Not Noita, because the player is operating an atlas-level containment problem,
  not an avatar combat/exploration loop.
- Not Zachtronics, because this is not a bounded transform puzzle; it is a live
  spatial front-management problem.

The closest genre shape is tactical disaster-response / tower-defense, but the
material-sim front and reusable voxel counter-patterns are enough to justify one
more prototype review.

## Next Required Evidence

`u4b4` should stay open for reviewer 3. Closure needs at least one more written
prototype review and at least two of three total reviewers saying the loop hangs
together.

Reviewer 3 should focus less on implementation and more on play shape:

1. Try a barrier-only containment plan.
2. Try a mixed plan using cooling trench and firebreak.
3. Report whether the mixed plan creates a meaningful choice, or whether the
   answer is still "spend walls until the lane stops."

