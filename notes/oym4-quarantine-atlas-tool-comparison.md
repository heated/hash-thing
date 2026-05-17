# Quarantine Atlas Tool Comparison

Date: 2026-05-03
Bead: `hash-thing-oym4`

## Question

The `u4b4` prototype panel accepted Quarantine Atlas only weakly because
barrier-only play might dominate cooling trench and firebreak. This comparison
tests one common lane situation: six interventions placed across the central
hazard path between the lava/fire source and the middle settlement.

## Protocol

The landed regression test
`quarantine_atlas_mixed_plan_beats_barrier_only_on_wide_lane` builds the
deterministic `128^3` Quarantine Atlas scene, applies two six-stamp plans, runs
16 recursive sim steps, then counts:

- active hazard cells, `FIRE` or `LAVA`, in the central threat lane
- remaining `GRASS` in the middle settlement

Plans:

- Barrier-only: six barrier stamps at `x = 40, 50, 60, 70, 80, 90`.
- Mixed: firebreak at `x = 40`, cooling trench at `x = 50`, firebreak at
  `x = 60`, cooling trench at `x = 70`, then barriers at `x = 80, 90`.

Both plans spend the full six-intervention budget.

## Result

Targeted test output:

```text
barrier = QuarantineThreatMetrics { fire_or_lava_in_lane: 62, intact_settlement_grass: 75 }
mixed   = QuarantineThreatMetrics { fire_or_lava_in_lane: 4, intact_settlement_grass: 75 }
```

## Verdict

Barrier spam does not dominate this common wide-lane situation. The mixed plan
is meaningfully better because firebreak and cooling trench cover the wider
fuel band, while the barrier stamp only makes a narrow hard stop. With the same
six-stamp budget and equal settlement outcome, mixed play leaves 58 fewer
active hazard cells in the lane after the warm sim run.

No immediate design fix is needed for this bead. The next balance question is
human feel: whether players discover the mixed plan naturally, or whether the
UI/tutorial surface makes barriers look like the only intended tool.
