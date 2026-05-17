# u4b4 Action-Layer Research Packet

Date: 2026-05-02
Bead: `hash-thing-u4b4`
Status: pre-panel sketch packet; not a prototype commitment

## Premise

The engine evidence says hashlife can be a real multiplier on stable or
partially-stable worlds. The missing game-design question is narrower: what can
the player do where that multiplier matters to the loop, not just to a benchmark?

This packet proposes five loops for reviewer-panel triage. It should not close
`u4b4`: the next step is written reviewer verdicts, then one 30-minute playable
prototype for the best surviving sketch.

## Adjacent Games

| Game / genre | Substrate | Main loop | What it owns | Gap for this project |
|---|---|---|---|---|
| The Powder Toy / falling-sand sandbox | 2D particle and cellular material sim | Place materials, observe reactions, share contraptions | Material curiosity and emergent setups | Mostly toy/sandbox; weak external stakes or progression |
| Noita | 2D pixel material sim plus roguelite action | Explore, fight, compose wands, survive self-inflicted chaos | "Every pixel is simulated" as adventure texture | Real-time action foregrounds combat and authored progression more than large-scale stable computation |
| Minecraft | 3D voxels, local block updates, crafting/survival | Gather, build, survive, automate with redstone | Legible 3D world editing and mod culture | Simulation is coarse and mostly local; stable megastructure reuse is not the central verb |
| Factorio | Discrete logistics graph and factory simulation | Extract, route, automate, scale production under pressure | Scaling optimization with readable throughput goals | The world is a factory graph, not a mutable material field |
| Zachtronics-style engineering puzzles | Deterministic machine or program rules | Build a mechanism that transforms inputs to outputs under metrics | Dense design-space optimization and replayable solutions | Usually level-bounded, not a live world with persistent large-scale material consequences |
| Oxygen Not Included / colony sim | Material, thermal, gas, agent-needs simulation | Dig/build systems that keep a colony alive | Survival pressure from environmental systems | Colony agents and needs dominate; hashlife-scale repetition is background optimization |

The opportunity is not "3D Powder Toy" by itself. The distinctive target is a
game where the player edits a huge material computer/ecology, then uses time,
scale, and repeated stable structure as resources.

## Hashlife Strengths as Player Verbs

| Substrate property | Design translation |
|---|---|
| Large stable regions become cheap | Let players build persistent infrastructure that can run while attention is elsewhere. |
| Repeated subtrees can share work | Make modular stamping, cloning, templates, or self-similar machines a first-class strategy. |
| Local churn is expensive | Treat damage, pests, sabotage, weather, or extraction as localized disturbances the player manages. |
| Multi-tick patterns can be memoized | Reward oscillators, timed gates, clocks, conveyor rhythms, and periodic reactors. |
| Sparse edits can invalidate large cached areas | Make edits deliberate: the player plans windows of intervention instead of constantly painting. |
| Scale can be simulated unevenly | Let the player zoom from hand-built cells to city/continent infrastructure without changing rule language. |
| Deterministic replay is natural | Let players scrub, fork, audit, or prove a process, turning time into a tool. |
| Material rules are inspectable | Let machines be made from the same stuff as terrain, not hidden components. |

## Sketch 1: Fault Garden

**One-line pitch:** A survival automation game where your base is a self-healing
material ecology, and the main job is designing stable patterns that absorb
localized disasters.

**Setting:** A living industrial garden on an unstable moon. The world grows
resources, heat, pressure, spores, and corrosive fluids through CA-like material
rules. The player builds repeated "organs" that harvest, filter, vent, and
repair.

**Core verb:** Stamp and tune modules, then intervene surgically when faults
propagate. The player does not babysit every cell; they design stable organs and
patch local churn before it escapes.

**Progression:** Start with one resource loop: water/soil/heat. Unlock larger
organs: pumps, thermal lungs, filter beds, memory relays, spore fences. Later
maps add harsher fault sources and require larger repeated structure.

**Why hashlife matters:** Stable infrastructure should be cheap while local
faults remain expensive and meaningful. Repeated organs should share simulation
work, making city-scale living factories plausible.

**Why anyone cares:** The fantasy is "gardening a machine that mostly runs
itself." Success feels like resilient design, not throughput maximization.

**Differentiation:** Unlike Factorio, the factory is not a graph of placed
machines; it is a mutable material ecology. Unlike Oxygen Not Included, survival
is about pattern resilience more than colonist micromanagement. Unlike Powder Toy,
fault containment gives the sandbox a reason to matter.

**Likely falsifier:** If the best strategies become ordinary pipe/conveyor
layouts, or if local faults are either trivial or unrecoverably chaotic, the
loop collapses into existing automation or sandbox play.

## Sketch 2: Time Claim

**One-line pitch:** A puzzle-builder where land is conquered by proving that a
material process remains stable for long simulated futures.

**Setting:** The world is a disputed frontier made of volatile cellular matter.
The player claims regions by building mechanisms that meet a target after N
ticks: purify a lake, keep a reactor bounded, route sand pulses, maintain a seal.

**Core verb:** Build a local mechanism, fast-forward it, inspect failures, rewind
to edit, and submit a "proof run" that survives the requested horizon.

**Progression:** Early levels ask for 100-tick stability in small rooms. Later
levels ask for million-tick periodicity, multiple interacting regions, and
resource-efficient proofs.

**Why hashlife matters:** The game literally asks the engine to skip stable
future time. If proof horizons are long and structures become periodic, temporal
memoization becomes the experience instead of hidden optimization.

**Why anyone cares:** Players get the Zachtronics satisfaction of a working
machine, but the output is a living material process whose long-term behavior is
the objective.

**Differentiation:** Unlike Zachtronics, the machine is embedded in a mutable
world with terrain and fluids. Unlike Baba Is You, the rules are not manipulated
as text; the player proves behavior through construction. Unlike Minecraft
redstone, long-horizon simulation is central.

**Likely falsifier:** If long-horizon proof feels like waiting for a progress bar
instead of revealing design flaws, or if short bounded simulations are enough,
hashlife is incidental.

## Sketch 3: Pattern Prospectors

**One-line pitch:** An exploration game where players mine a vast rule-space for
rare stable structures, then domesticate them as tools.

**Setting:** Different biomes are rule regimes with distinct material chemistry.
Players seed soups, disturb terrain, and search for gliders, oscillators,
filters, eaters, pumps, and other reusable motifs.

**Core verb:** Prospect, classify, capture, and breed useful patterns. The
player carries a library of discovered structures and stamps them into later
expeditions.

**Progression:** Early goals ask for simple stable filters. Midgame asks for
transport and timing patterns. Late game asks for composite structures that work
across hostile rule regimes.

**Why hashlife matters:** Soup search and repeated motifs are hashlife-native.
The engine advantage should show up in running many long-lived candidates and
then reusing discovered structures at scale.

**Why anyone cares:** The player is not just building; they are discovering an
ecology of computational artifacts and turning them into a toolkit.

**Differentiation:** Unlike Powder Toy, the loop has collection, classification,
and reuse. Unlike Noita, the core reward is pattern discovery rather than combat
or spell composition. Unlike Factorio, automation comes from found cellular
organisms, not authored machines.

**Likely falsifier:** If discovery is more fun as offline tooling than live play,
or if useful structures are too rare or too opaque, it becomes a research toy.

## Sketch 4: Sleeping City

**One-line pitch:** A city-builder where most of the city sleeps in memoized
steady state until player choices wake districts into expensive local crises.

**Setting:** The player manages a vertical voxel city whose infrastructure is
made from material rules: air, heat, waste, power pulses, transport grains,
signal gels. Districts can settle into stable routines.

**Core verb:** Zone and stamp district templates, then route scarce active
attention to crises: a heat plume, waste bloom, riot of signal material, or
structural leak.

**Progression:** Grow from one block to a layered city. New districts introduce
materials that create new stability constraints. The win condition is not raw
population; it is how much city can run asleep.

**Why hashlife matters:** The core metric is sleeping area versus active churn.
Repeated districts should amortize heavily; crises should wake local chunks.

**Why anyone cares:** It gives city-building a strong systems fantasy: designing
urban metabolism that becomes quiet when healthy.

**Differentiation:** Unlike SimCity-style builders, the city is not mostly
abstract service radii. Unlike Minecraft, buildings are not just geometry. Unlike
Oxygen Not Included, the unit of success is large-scale district stability rather
than individual agent survival.

**Likely falsifier:** If the best UI hides the material sim behind ordinary
city-builder overlays, then the substrate is not pulling its weight.

## Sketch 5: Quarantine Atlas

**One-line pitch:** A tactical logistics game where the player contains spreading
material phenomena across a huge map by deploying reusable counter-patterns.

**Setting:** A world map is full of slow material hazards: crystal blooms, lava
veins, gas fronts, fungal logic, pressure storms. The player has mobile crews
and a library of stamped countermeasures.

**Core verb:** Predict, isolate, and deploy patterns. The player cannot edit
everything; they choose where to disturb the map and what stable barriers or
reactors to place.

**Progression:** Start with one hazard type and small towns. Later maps combine
hazards, terrain constraints, and limited intervention budgets. Victory means
keeping settlements connected while letting most wilderness continue simulating.

**Why hashlife matters:** Most of the map should remain quiet or periodic while
fronts and interventions churn. Reusable barriers and reactor templates should
share work.

**Why anyone cares:** It turns simulation scale into strategic pressure: the map
is too large to hand-control, so judgment and pattern libraries matter.

**Differentiation:** Unlike Noita, the player is not a combat avatar in caves.
Unlike Factorio, logistics supports containment rather than production growth.
Unlike Powder Toy, the map and stakes force prioritization.

**Likely falsifier:** If the player mostly paints walls around hazards, the loop
is too shallow. The hazards must create prediction and tradeoff problems.

## Review Candidates

**Current-engine candidate: Fault Garden.** It best matches the current engine
without requiring temporal-hashlife first and gives the P0 demo a legible visual
target. Its risk is that hashlife may remain an engine optimization instead of a
player-facing constraint. Reviewers should only advance it if they can identify
a player decision that becomes worse without stable-region skipping or structural
sharing.

**Thesis candidate: Time Claim.** It makes long-horizon simulation the objective,
so hashlife is more directly part of the play loop. Its risk is implementation
sequence: the prototype may need temporal/macro-skip work before it can feel
good.

**Do not pick yet:** The next step is a written sketch panel.

## Runnable Sketch Panel

Create `notes/u4b4-sketch-review-1.md`. Collect at least three independent
written verdicts; use a mixed agent panel if available, with at least one
adversarial pass. Do not synthesize the verdicts into a pick yourself before the
three raw verdicts are recorded.

Suggested reviewer prompt:

```text
Read notes/u4b4-action-layer-research.md. For each sketch, answer:
1. Does this hang together as a gameplay loop?
2. Is it clearly distinct from Powder Toy, Noita, Minecraft, Factorio, and
   Zachtronics-style puzzle games?
3. What player decision becomes impossible or less meaningful without
   hashlife-scale skipping, temporal skipping, or structural sharing?
4. Is it a better thesis candidate, current-engine prototype candidate, both, or
   neither?
5. What would falsify it fastest in a 30-minute playable prototype?

End with one prototype recommendation and one reject/kill recommendation.
```

After the sketch panel, `u4b4` still remains open until the selected sketch also
produces a playable 30-minute prototype where at least two of three written
reviewers report that the gameplay loop hangs together.

Minimum questions for `notes/u4b4-sketch-review-1.md`:

1. Which sketches hang together as gameplay loops?
2. Which are clearly distinct from Powder Toy, Noita, Minecraft, Factorio, and
   Zachtronics-style puzzle games?
3. What player decision depends on hashlife-scale skipping, temporal skipping,
   or structural sharing?
4. Which sketch should be prototyped first, and what would falsify it fastest?

## Sources

- Project evidence: `notes/aqq4-thesis-verdict.md`,
  `notes/nav-epic-directive.md`, `docs/perf/lead-ledger.md`.
- Adjacent-game references: [The Powder Toy Steam page](https://store.steampowered.com/app/1148350/The_Powder_Toy/)
  describes a physics sandbox with air, heat, gravity, and material
  interactions; [Noita's official site](https://noitagame.com/) describes a
  roguelite where every pixel is simulated; [Factorio's official content page](https://www.factorio.com/game/content)
  frames factory-building, automation, scaling, and production objectives;
  [Zachtronics' Opus Magnum page](https://www.zachtronics.com/opus-magnum)
  frames open-ended physical-machine engineering puzzles; [Klei's Oxygen Not
  Included page](https://www.klei.com/games/oxygen-not-included) frames colony
  survival around oxygen, warmth, and sustenance threats.
