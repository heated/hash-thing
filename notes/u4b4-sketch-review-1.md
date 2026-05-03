# u4b4 Sketch Review 1

Date: 2026-05-03
Input: `notes/u4b4-action-layer-research.md`
Bead: `hash-thing-u4b4`
Status: sketch panel complete; prototype review still required before closing

## Panel Contract

Each reviewer answered:

1. Does this hang together as a gameplay loop?
2. Is it clearly distinct from Powder Toy, Noita, Minecraft, Factorio, and
   Zachtronics-style puzzle games?
3. What player decision becomes impossible or less meaningful without
   hashlife-scale skipping, temporal skipping, or structural sharing?
4. Is it a better thesis candidate, current-engine prototype candidate, both, or
   neither?
5. What would falsify it fastest in a 30-minute playable prototype?

## Raw Verdict 1

### Sketch 1: Fault Garden

1. Yes. Build stable organs, let them run, respond to localized faults before
   they cascade.
2. Mostly distinct, especially from Factorio and Powder Toy, if the "organ
   ecology" is real and not just pipes with organic names.
3. Deciding to invest in repeated self-healing modules becomes less meaningful
   without stable-region skipping and structural sharing. If every district
   costs the same forever, modular resilience is just aesthetics.
4. Both, but stronger as a current-engine prototype.
5. Falsifier: in 30 minutes, if the best play is drawing simple barriers/pipes
   and watching messes resolve or explode with little surgical intervention,
   kill or heavily revise.

### Sketch 2: Time Claim

1. Yes. Build, fast-forward, inspect failure, rewind, prove stability.
2. Clearly distinct from the listed games. It overlaps Zachtronics in
   satisfaction, but the long-horizon material proof is a real difference.
3. Choosing designs for million-tick stability becomes impossible without
   temporal skipping. The core decision is "will this remain bounded or periodic
   far into the future?"
4. Best thesis candidate; weaker current-engine candidate unless temporal
   skipping is already usable enough.
5. Falsifier: if the proof run feels like waiting, or failures are only visible
   in the first few hundred ticks, the hashlife thesis is not doing work.

### Sketch 3: Pattern Prospectors

1. Partial. Prospect, classify, reuse is coherent, but "live play" is
   under-specified.
2. Distinct in theory, but risks becoming either Powder Toy with a collection log
   or external CA research tooling.
3. Deciding which discovered motifs are worth domesticating at scale depends on
   structural sharing and long-lived candidate search. Without that, discovery
   becomes small-scale tinkering.
4. Thesis candidate only, unless the current engine already supports quick
   pattern search and library stamping.
5. Falsifier: if useful patterns are too rare, opaque, or more fun to find with
   offline batch search than in-game play.

### Sketch 4: Sleeping City

1. Yes conceptually: grow districts, make them settle, intervene when they wake.
2. Distinct if material simulation remains visible. Weakly distinct if the UI
   collapses into normal city-builder overlays.
3. The decision to stamp stable district templates and optimize "sleeping area"
   loses meaning without stable-region skipping and structural sharing.
4. Both, but probably too UI-heavy for the first prototype.
5. Falsifier: if the material layer wants to be hidden behind heat/waste/service
   overlays, the city fantasy is carrying the game, not hashlife.

### Sketch 5: Quarantine Atlas

1. Yes. Predict fronts, place counter-patterns, choose where limited
   intervention matters.
2. Fairly distinct from the listed games; strongest separation is containment
   logistics over production, combat, or sandbox curiosity.
3. Deciding where not to intervene depends on scale skipping. The map must be too
   large to babysit, with quiet regions cheap and active fronts expensive.
4. Current-engine prototype candidate; possible thesis candidate if prediction
   over huge time/space becomes central.
5. Falsifier: if containment reduces to painting walls around hazards, the loop
   is too shallow.

Prototype recommendation: **Fault Garden**. It is the best 30-minute
current-engine test because it can show stable repeated modules, localized churn,
and visible repair decisions without requiring the full temporal proof stack
first.

Reject/kill recommendation: **Pattern Prospectors** for now. It has the highest
risk of becoming an offline CA-search toy rather than a playable loop, and its
fun depends on rare discoveries that are hard to guarantee in a short prototype.

## Raw Verdict 2

### Sketch 1: Fault Garden

1. Gameplay loop: Mostly hangs together. "Design stable organs, wait, patch
   faults" is a real loop, but it is under-specified on pressure: what forces
   the player to improve instead of just overbuilding buffers?
2. Distinctness: Moderately distinct. It dodges pure Powder Toy by adding
   survival/fault stakes, and dodges Factorio by using material ecology instead
   of belts/machines. Its weakest comparison is Oxygen Not Included plus
   Factorio: if organs become named infrastructure blocks, it collapses into
   familiar automation.
3. Hashlife-dependent decision: Whether to stamp a known resilient organ many
   times across a huge base versus hand-tuning bespoke local fixes. Without
   stable-region skipping/structural sharing, "scale by repeated living modules"
   becomes less meaningful because the game cannot afford the city-scale idle
   ecology that makes resilience design matter.
4. Candidate: Both, but better current-engine prototype than thesis. It can show
   a loop before temporal skipping exists. It is weaker as a thesis because
   hashlife risks being invisible optimization.
5. Fast falsifier: Give the player one harvesting organ, one spreading fault,
   and one repeatable repair/filter module. If optimal play is just walling off
   the fault or painting more material, kill or radically reframe it.

### Sketch 2: Time Claim

1. Gameplay loop: Hangs together cleanly. Build, fast-forward, inspect failure,
   rewind, improve, submit proof is coherent and immediately legible.
2. Distinctness: Strongest distinctness. It borrows Zachtronics satisfaction,
   but long-horizon material stability as the win condition is meaningfully
   different from fixed input/output machine puzzles.
3. Hashlife-dependent decision: Choosing designs that become periodic, quiescent,
   or provably bounded over long futures. Without temporal skipping, the player
   cannot reasonably compare "messy but works for 500 ticks" against "elegant
   and stable for 1,000,000 ticks"; the core horizon decision disappears.
4. Candidate: Best thesis candidate. Not a great current-engine prototype unless
   the prototype can fake or instrument fast-forward well enough to make proof
   horizons feel real.
5. Fast falsifier: Make a tiny level with a leak/reactor/seal and a 10,000-tick
   claim. If players solve it by watching a progress bar rather than discovering
   surprising late failures, the loop is dead.

### Sketch 3: Pattern Prospectors

1. Gameplay loop: Weak-to-moderate. Prospect, classify, capture, reuse is
   compelling intellectually, but it risks being a toolchain rather than a game.
2. Distinctness: Distinct from the listed games, but dangerously close to "Life
   enthusiast workstation." Its differentiation from Powder Toy depends entirely
   on whether classification/reuse becomes playable rather than archival.
3. Hashlife-dependent decision: Running many candidate soups long enough to find
   rare stable/periodic motifs, then choosing which discovered motif to reuse at
   scale. Without temporal skipping and structural sharing, prospecting rare
   long-lived patterns becomes too slow or too small to be the main verb.
4. Candidate: Thesis candidate, not current-engine prototype candidate. It
   proves the engine idea if it works, but the fun risk is high.
5. Fast falsifier: Give players a seeded soup generator, a simple classifier,
   and one goal requiring a discovered oscillator/filter. If the fun is all in
   the search UI and not in deploying the found pattern, reject it as a game
   loop.

### Sketch 4: Sleeping City

1. Gameplay loop: Conceptually hangs together, but it is the blurriest. "Make
   city sleep" is a strong metric; "zone/stamp/route attention to crises" could
   become ordinary city-builder management unless the sleeping state is deeply
   playable.
2. Distinctness: Moderately distinct, but fragile. It can separate from
   SimCity/Oxygen Not Included if district metabolism is materially inspectable.
   If the UI naturally becomes overlays and alerts, it loses the material-sim
   identity.
3. Hashlife-dependent decision: Whether to design repeated districts that settle
   into cheap steady states versus high-output districts that constantly churn.
   Without stable-region skipping, "asleep city area" is just a score label, not
   a functional strategy.
4. Candidate: Thesis candidate maybe; current-engine prototype maybe not. It
   likely needs too much UI and too much city simulation scaffolding before the
   core loop can be judged.
5. Fast falsifier: Prototype two district templates: one efficient but churny,
   one lower-output but sleep-stable. If players cannot feel why sleep-stability
   is strategically better, kill the city wrapper.

### Sketch 5: Quarantine Atlas

1. Gameplay loop: Hangs together well. Predict, isolate, deploy reusable
   counter-patterns across a too-large map is a solid tactical loop.
2. Distinctness: Fairly distinct. It is less like Factorio because logistics
   serves containment, not growth; less like Powder Toy because the large map
   creates prioritization. The main risk is becoming a tower-defense/wall-painting
   game.
3. Hashlife-dependent decision: Choosing where to intervene on a huge
   mostly-quiet map, and which reusable containment pattern to stamp. Without
   scale skipping, the "atlas" cannot be large enough for triage; without
   structural sharing, reusable countermeasures are just prefabs rather than
   computationally privileged strategy.
4. Candidate: Both, with a better balance than Fault Garden. It can be
   prototyped now as a map-scale containment toy, and it gives hashlife a clearer
   player-facing reason: most of the world must keep running while attention goes
   elsewhere.
5. Fast falsifier: Create one spreading hazard, three towns, limited
   interventions, and two counter-patterns. If the dominant move is drawing
   perimeter walls and waiting, reject the loop.

Prototype recommendation: **Quarantine Atlas** first. It has the best mix of
immediate playable pressure and hashlife-relevant decisions: huge mostly-idle map,
localized churn, reusable patterns, limited interventions, and visible triage.

Reject/kill recommendation: **Sleeping City** for now. It needs the most UI, the
most genre baggage, and the most abstraction before the distinctive engine thesis
becomes visible.

## Raw Verdict 3

### Sketch 1: Fault Garden

1. Gameplay loop: Yes, but conditionally. "Stamp resilient organs, fast-forward
   normal operation, surgically patch faults" is a real loop if faults are
   diagnosable and recovery is local.
2. Distinctness: Moderately distinct. It risks becoming ONI/Factorio-with-
   materials unless the material substrate is visibly the machine, not decoration
   around pipes.
3. Hashlife-dependent decision: "Can I afford to replicate this organ 200 times
   and let the city run while I manage only disturbed zones?" Without
   stable-region skipping or sharing, modular resilience becomes either too
   small-scale or just cosmetic.
4. Candidate: Both, but stronger as a current-engine prototype than thesis
   candidate.
5. Fast falsifier: Build one harvest/filter/vent organ plus one spreading fault.
   If the best play is just draw a wall/pipe, or if the fault is unreadable
   chaos, kill or radically narrow it.

### Sketch 2: Time Claim

1. Gameplay loop: Yes. Build, fast-forward, inspect failure, rewind/edit, submit
   proof is clean and repeatable.
2. Distinctness: Strong. It overlaps Zachtronics, but the objective being
   long-term material behavior in terrain gives it a different center.
3. Hashlife-dependent decision: "Do I design for million-tick periodic stability
   instead of short-term success?" Without temporal skipping, long-horizon proof
   is waiting, not play.
4. Candidate: Best thesis candidate. Weak current-engine prototype candidate
   unless the engine already supports convincing time skipping/proof horizons.
5. Fast falsifier: Make a 5-minute level with a 10k+ tick proof goal. If skipping
   does not reveal failures quickly and interestingly, or if 200 ticks is enough
   to solve it, the thesis collapses.

### Sketch 3: Pattern Prospectors

1. Gameplay loop: Maybe. Prospect/classify/capture/reuse is coherent, but it
   depends on discoverability and legibility.
2. Distinctness: Good on paper, but fragile. It could become "CA research UI
   with progression" rather than a game.
3. Hashlife-dependent decision: "Which soups or motifs are worth running for long
   futures and adding to my reusable library?" Without temporal skipping and
   structural sharing, search scale and motif reuse lose force.
4. Candidate: Thesis candidate only if discovery is live and playable. Not a
   good current-engine prototype first.
5. Fast falsifier: Give players three seed tools and one goal structure. If they
   cannot intentionally improve odds or understand why a pattern matters within
   30 minutes, reject.

### Sketch 4: Sleeping City

1. Gameplay loop: Yes, but vague. "Grow districts, maximize sleeping area,
   respond to wakes" is promising, but the player verbs need sharper definition.
2. Distinctness: Moderate. The sleeping/churn metric is distinctive; the
   city-builder framing may hide the CA substrate behind overlays.
3. Hashlife-dependent decision: "Should I stamp another proven district template,
   knowing it will remain cheap while stable?" Without structural sharing and
   stable-region skipping, city scale is just normal optimization pressure.
4. Candidate: Both in theory, neither as first prototype. It needs too much UI
   and too many systems before the thesis is visible.
5. Fast falsifier: Prototype three district templates and one crisis. If players
   mostly read service overlays instead of material behavior, kill the
   city-builder wrapper.

### Sketch 5: Quarantine Atlas

1. Gameplay loop: Yes. Predict, isolate, deploy counter-patterns, triage limited
   attention is a strong loop.
2. Distinctness: Stronger than Fault Garden against Factorio/ONI, weaker against
   generic strategy unless the material simulation is central.
3. Hashlife-dependent decision: "Where do I spend my few interventions on a huge
   mostly-stable map?" Without large-scale skipping, the map cannot be huge
   enough for triage to matter.
4. Candidate: Both. It is especially plausible as a current-engine prototype
   because it can use a small number of hazard/front rules and still expose
   scale.
5. Fast falsifier: Build one town, one spreading hazard, three reusable
   counter-patterns, limited actions. If the winning move is always perimeter
   walls, reject or redesign hazards.

Prototype recommendation: **Quarantine Atlas** first. It exposes the thesis
better than Fault Garden while staying closer to current-engine feasibility than
Time Claim. The 30-minute prototype should force triage on a map too large to
hand-edit: one spreading hazard, settlements to protect, limited intervention
budget, and two or three stamped counter-patterns. The key test is whether
players make meaningful choices about where to wake/disturb the map and which
stable pattern to reuse.

Reject/kill recommendation: **Pattern Prospectors** for now. It is
thesis-adjacent, but its fastest path likely becomes offline CA search tooling,
not a playable current-engine loop. It can come back later as a subsystem for
Time Claim or Quarantine Atlas, not as the first action-layer prototype.

## Panel Synthesis

Verdict count:

- Prototype recommendation: Quarantine Atlas 2, Fault Garden 1.
- Reject/shelve recommendation: Pattern Prospectors 2, Sleeping City 1.
- Strongest thesis candidate: Time Claim, unanimous in substance.
- Best near-term current-engine candidate: Quarantine Atlas or Fault Garden;
  Quarantine Atlas wins the panel because it makes "huge mostly-stable map plus
  limited intervention" more player-visible.

Panel decision: **prototype Quarantine Atlas first**.

Prototype shape:

- One large mostly-quiet map.
- One spreading hazard/front.
- Two or three reusable counter-patterns.
- Settlements or critical regions to protect.
- Limited intervention budget, so the player must decide where to disturb the
  map and which stable pattern to reuse.

Fastest falsifier: if the winning move is perimeter-wall painting or hand-editing
everything, reject or redesign the hazards. The prototype must make triage,
pattern choice, and stable-region scale visible.

`u4b4` remains open after this sketch panel. Closure still requires a playable
30-minute prototype and at least two of three written reviewers saying the loop
hangs together.
