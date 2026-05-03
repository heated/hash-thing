# u4b4 Prototype Review 3

Date: 2026-05-03
Reviewer: `darwin`
Input: Quarantine Atlas prototype on `origin/main` after `8e80406`
Scene: `q` / `--dump-scene quarantine-atlas`
Status: reviewer 3 of 3; completes the written prototype-review panel

## Protocol

Darwin was asked for an independent read-only review of the Quarantine Atlas
prototype. The review prompt asked:

1. Does the loop hang together?
2. Does barrier-only dominate versus cooling trench/firebreak?
3. Is the prototype distinct from Powder Toy, Noita, Minecraft, Factorio, and
   Zachtronics-style puzzle games?
4. Does this count as yes/no for the `>=2/3` closure criterion?

Evidence used by Darwin:

- Notes: `notes/u4b4-action-layer-research.md`,
  `notes/u4b4-sketch-review-1.md`,
  `notes/u4b4-prototype-1-review.md`, and
  `notes/u4b4-prototype-2-review.md`.
- Quarantine Atlas code in `src/main.rs` and `src/sim/world.rs`.
- Smoke run:
  `cargo run --bin hash-thing -- 128 --dump-frame /tmp/quarantine-atlas-review-3-darwin.png --dump-scene quarantine-atlas`

## Verdict

**Weak yes.**

The loop hangs together as a prototype slice. Quarantine mode is not freeform
hand-editing: carve is disabled, right-click deploys one of three budgeted
stamps, and the scene has a live hazard lane, settlements, and only six
interventions. That is enough to read as "predict, isolate, deploy
counter-patterns" rather than a Powder Toy sandbox.

Barrier-only does **not** clearly dominate from code inspection. The barrier is
a narrow hard obstruction, while cooling trench and firebreak cover wider lane
areas and interact with the actual rules: water quenches/solidifies near
fire/lava, and firebreak removes fuel. The caveat is that this is still weakly
proven. The prototype needs a real side-by-side play run or scripted simulation
comparison to show mixed tools outperform wall spam in at least one common
situation.

Distinctness is sufficient for this stage. It is not Factorio production, not
Minecraft block construction, not Noita avatar combat, not Zachtronics transform
proof, and not pure Powder Toy because budgeted atlas-scale triage gives the
material sim external stakes. The closest comparison is tactical disaster
response / tower defense, but the material-front and reusable stamp layer make
it distinct enough to keep.

Hashlife relevance is present but still mostly implicit: the important decision
is where to spend scarce edits on a mostly quiet large map with localized churn.
That matches the engine thesis, but the current slice does not yet make
stable-region skipping or structural sharing felt directly.

For the `>=2/3` closure criterion: **yes, count this as a yes**, but mark it as
weak/conditional. Combined with reviews 1 and 2, Quarantine Atlas meets the
written-review threshold to close the prototype-loop question, with follow-up
work focused on proving non-barrier tool choice.

