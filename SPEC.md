# Hash Thing — Design Spec

> A 3D voxel cellular automaton engine with hash-consed octree storage, Hashlife-style simulation stepping, Margolus-block physics, material-type CA rules, and SVDAG rendering. Targeting infinitely explorable worlds with emergent physics and redstone-style computation.

Repo: `git@github.com:heated/hash-thing.git`

---

## The user's words (source of truth)

**Everything in this section is verbatim from the user during the design conversation.** These are the source requirements. When the spec and the user's words disagree, the user's words win. Do not paraphrase or "update" these — add clarifications below them instead.

### Initial framing

> Let's make a new repo here and explore making essentially 3D Hashlife, except instead of Conway's Game of Life we're gonna just have a 3D cellular automata in general that can support some game-like physics. and we'll want voxels and everything, and we'll want to explore first maybe what kind of architecture or language etc this is gonna be. In terms of the simulation I was thinking something like to start with something like powder game but in 3d but we can talk more also it would be great if we could you reuse voxel quest for this like the display engine but i'm not tied to it

### Core goals

> 1. help simulating a ton of stuff, possibly redstone-like computation
> 2. pure CA to start, but support particles
> 3. gonna try with Voxel Quest first
> 4. we'll start with 64^3 to demo, then scale up. def shooting for 4096+, prob will aim for infinitely explorable world
> 5. physics: lets experiment
>
> the one level up is what we'll go for, for architecture. but one level up in *your ownership of this* is a different thing

### Validation and cross-platform

> Basically just get this up and running now. Do what you have to, to do that. So validate this end-to-end. Definitely check if VoxelQuest can handle this sort of thing. If it can't, throw it out and let's do Rust. But validate, make absolute sure that it can't first. But I'm not sure that Rust will fit, maybe. But I'd like this to be cross-platform, to work on Mac, Linux and Windows. But let's talk, because that might come with tradeoffs.

> Before you go further, can we talk architecture for distribution? Or just supporting the different platforms? I know games on Steam, for example, typically don't support macOS. What are our options with all this?

> I kind of just want the absolute most power. So, I get your cross-platform thing but what alternatives do we have to using something else? It sounds like you just reached for WASM immediately but I kind of want to do native binaries. Is Steam easy to add later or should we plan for it from the start?

> How much work is it to do native and support all three platforms? Definitely keep WebBuild as an open option.

### On the simulation path

> Okay, but to be clear, somebody is making 3D Noita and I don't think they're doing it on a cellular automaton. So I'm definitely gonna lean toward material type CA here. Yep, okay, let's do a material type CA but let's explore a little bit just what kind of architecture we want.

### On materials, infinite worlds, octree-direct operation, Margolus

> A few things, it'd be nice if we had more material types. I'm willing to start with 256 but it could be cool to just support more unless it becomes a performance issue. Maybe we fold meta into the material, some sort of tagged pointer space? Similarly, ideally this thing is infinitely generating in each direction and it'd be nice to generate some interesting dungeon-like things or just terrain. Sort of like how Minecraft generates stuff. and I'm super curious about these phases and physics in general. I'd love to dive into this further. And again, with saving memory... I don't know but can we just store the Octree? It sounds like your understanding is like oh we have to translate it into pure... we have to evaluate everything. But I'm not sure that's the case. I'm still not really sure what the Margulis stuff is. It'd be nice for you to expand a bit on what going in this direction would look like or where, what the lineage of this thing is. We absolutely do not want to do scan-based like Noita. Probabilistic, I don't know if that works with hash life. I do wonder if we could do hash life with a scan based thing though, aside. Like, we could just eat the issue with the bugs with locality or whatever. I think we should certainly support material interactions to be hand authored, and im down to have a lot of this (and less, emergent. whatever works). Okay, so before you design more let's talk about some of these points

### On terrain-gen and direct octree operation

> * terrain-gen - Worried that if we don't use CA for this it'll be slow, but worth exploring.
> * flatten-to-grid - I'm down to use this for testing, but definitely want to see if we can avoid this for every single aspect of the game. Especially rendering, though that sounds super complex. lets talk more about this

### Final direction

> * sounds good on terrain gen, but lets keep an eye on performance and surface issues early
> * lets also do some SVDAG rendering today to reserve a spot for it in the complexity budget. and can we aim for the state of the art? web search for what that is.
> * ok, everything sounds good. lets start. first, save all of this to a spec. especially save *everything I said* and emphasize its importance in the spec
>
> and then build it!

### Product vision and demo direction (2026-04-12)

> We definitely want it to be a game. It'd be great if the engine could support other games, a la the Metaverse thing. But definitely not a tech demo or just a basic toy — we'll try to shoot a little higher but I'm willing to walk it back a bit.
>
> When they launch it, we want to start with a basic voxel thing — like, okay this is normal — and then they walk around and start seeing crazier and crazier things. For a demo, maybe it starts looking like Minecraft, but I'm not married to that. I just want to say we could do other things. And as they keep moving and walking through the demo they start seeing the whole world come to life and crazy cool interactions and really novel shit.
>
> A character, and still able to interact with the world. Probably placing material, but definitely seeing stuff.
>
> For physics: primarily powder game stuff or maybe fire. Not really redstone to begin with.
>
> For terrain/dungeons: I could see a lot of different possible generation — a lot of different worlds to step into. I don't want terrain to just be one kind of terrain. I might want to start in an infinite 3D dungeon or whatever, but I'm also down to support the normal terrain.
>
> For materials: it could be good to have different material stacks — different namespaces of materials depending on the front end game on top of this. One of the sets should be whatever standard Minecraft-y materials, except more in the direction of powder game. Any sort of material stack should be a little more in the direction of powder game — stuff moving around by default, a little more alive.
>
> Done enough to show someone: it's fast enough, things are moving enough, it looks like a relatively complete thing. More of a complete thing than what it is now.
>
> I'm gonna want to talk more about what specific set of features we want to put in. There's a standard RPG stack, but I don't necessarily want to do that. I don't want it to look exactly like Minecraft. I don't know if I want crafting, stuff like that. So let's talk more about those.
>
> If it's clear from what my vision is, the answer to whatever downstream design question pops up — don't require a human gate. Only gate if it's not obvious from the spec.

### World ideas (2026-04-12, to workshop further)

> **Infinite 3D dungeon.** Infinite in all directions. You don't see much of it yet but it opens into big spaces periodically.
>
> **Natural megastructures.** Much more open space by default. Showcases big movement and the alive CA environment. Something that looks more natural and outdoorsy but with lots of 3D structure. The ground forms crazy 3D megastructures you can see up — maybe a world tree? Fine with it looking a little alien, but we'll have to workshop it. Worry: stuff that comes to mind might not look great.
>
> **Minecraft-style.** But might be strictly worse than the megastructure world. Talk more.
>
> **Tiny scale.** You are very small in a normal-sized world. Lots of stuff that's sized for you as well. Normal objects are huge terrain features.

These are terrain/world presets — each would be a different terrain stack + material set + generation parameters. Still needs workshopping; don't build any of these without design conversation first.

### Interactions, identity, wow moment (2026-04-12 session 2)

> **Materials:** Some materials should be static (stone, dirt). Some fall (sand). Some flow (water). Maybe fire. Not everything needs to move — it's okay for some to just sit there.
>
> **Voxel scale:** open question, depends on performance. Either roughly player-sized voxels or much smaller. TBD.
>
> **Game mode:** Far more sandbox than builder. Survival with possible roguelike layer. NOT structure-building focused — Minecraft already does that. Sandbox or survival/roguelike, possibly something else.
>
> **Interactions — direction:** Entities. Player places machines or entities that do things in the world. That's the interaction layer, not block-by-block construction. Also: Metroidvania-style progression — items unlock areas, movement upgrades (walking → breaking → swimming → flight). Macro-scale player presence — player has velocity, crashes into things, stuff shifts around them, every action is impactful and changes the world. The player is always physically affecting their surroundings just by moving through them.
>
> **The trailhead imagery:** A 3D sandbox where stuff is constantly moving, the player is always displacing and interacting with the material around them, everything the player does ripples outward. The player isn't placing blocks — they're a physical presence crashing through a living world.
>
> **First wow moment — 30-second version:** Something 3D and beyond normal simulation. Rain (3D particle field). Or a powder-game setup with lots of movement. Fireworks or explosions. A big sand pile collapsing. Water in the air.
>
> **First wow moment — aspirational:** Open space with floating islands, water flowing between them, stuff happening in the air. The megastructure world but with a LOT going on. Work toward this; simpler version first.
>
> **Material interactions for demo:** Water flowing, something airborne (rain/particles), a big sand collapse or avalanche, explosions/fireworks. These are the "prove it works" moments.
>
> **"Done enough to show someone":** Both a 30-second and a 5-minute experience. Each is foreplay building to the big wow reveal.
>
> **Game identity (still forming):** Not Minecraft's crafting/RPG stack. Materials will always struggle to be a perfect sandbox — lean into what's unique: 3D CA physics at scale, player-as-physical-presence, living world. Explore more. Play Noita for reference.
>
> **Clarification:** Sandbox is a MODE within the game, coexisting with other game mechanics. Door is open on roguelike combat. Other identities worth exploring — player-as-physical-presence is the starting trailhead, not the final answer.

### Feature decisions (2026-04-12 session 3)

> **Entities/machines — Powder Game inspiration:** Clone block (material source) is in. Entities are mobile creatures that interact with the CA world — like mobs but more physics-native. Not necessarily Minecraft mobs. Specifics TBD but direction is clear: things that move around and interact with materials.
>
> **Powder Game material candidates:** Clone (source), fan (wind/push), firework, magma/lava, ice (freezes water), acid (dissolves), oil (flammable), gunpowder, steam (rises), gas (explosive), metal (heat conductor), plant/vine (grows). Classic Powder Game set mapped to 3D CA.
>
> **Entity ideas to explore:** Physics creatures (water elemental, sand worm), automata-native entities (behavior IS CA rules), environmental entities (geysers, volcanos, whirlpools), critters (small scatter creatures that make world feel alive).
>
> **Roguelike:** Deferred. Get sandbox + metroidvania + destructible environment solid first. Risk of Rain 2 as possible inspiration when the time comes. Don't build roguelike mechanics yet.
>
> **Metroidvania progression (concrete):**
> - Start: walk only, navigate terrain puzzles
> - Unlock 1: block-breaking (maybe limited to some materials first)
> - Unlock 2: material creation abilities
> - Unlock 3: swimming/traversal
> - Unlock 4: flight — leaning Superman-style (powerful, fast), possibly a weaker tier before it
> - Progression IS the tutorial — no explicit tutorial text, gated areas teach by doing
> - Each unlock opens a new zone containing the next unlock
>
> **Inventory:** Hotbar yes, inventory yes, no crafting. Items auto-equip on pickup, hotbar to switch active tool. "Permanent equip with loadout swapping" worth exploring later but start with hotbar. Standard hotbar selection controls.
>
> **Demo world:** Megastructures. Player starts in a confined/tame environment, then it opens into the spectacle.

### Material encoding idea (2026-04-12 session 3)

> It would be great if our material system was tagged pointer-like. So one of the materials was special — special means go look up what this should be in another table. And special means it's destructible by the player but indestructible by explosions and stuff. The idea is these special blocks would be treated the same way by Hashlife all the time. Maybe a few types — hashlife would still interact with them differently but they'd just be breakable or unbreakable materials. Something to workshop — not married to it, but want to make space for it.

### Demo world and progression (2026-04-12 session 3)

> **World preset ranking:**
> 1. **Lattice** — first demo target. Enormous repeating geometric structures with active materials flowing through gaps. Escher meets nature.
> 2. **Inverted Mountains** — follow-up. Terrain hangs from above, open chasms below, waterfalls into mist.
> 3. **World Tree + floating islands** or **Organic Towers** — third option, pick later.
>
> **Demo progression (5 minutes):**
> The demo is a visual journey, not a gameplay tutorial. Minimal player interaction, maximum spectacle. Metroidvania unlocks happen over the full game (hours), not the demo. The key principle: scale is withheld until the final reveal — that's the wow moment.
>
> 1. ROOM — Small enclosed space. Quiet. Stone, water dripping. One exit. No sense of what's outside.
> 2. CORRIDOR — Growing hints: water stream, sand sifting, light/sound changes. Still enclosed, can't see out.
> 3. TEASES — Brief glimpses through cracks/gaps. Movement, color, something big. The lattice is felt before it's seen.
> 4. THE WALK — Path weaves through lattice interiors. Chambers, tunnels within the geometry. Material interactions at local scale (water, fire, lava→steam). Player thinks "this is cool" but doesn't grasp scale. Maybe one interaction.
> 5. THE REVEAL — Exit onto open vantage. Full lattice visible: infinite repeating geometry, every surface alive. THIS is when scale hits. Everything before was inside it.
>
> 30-second demo: compressed 1→5 or 3→5. Still needs the confined-to-open contrast to land the reveal — just faster.
>
> **Demo materials (Powder Game set):** Sand, water, lava, steam, fire, oil, gunpowder, ice, acid, clone, plant/vine, stone, dirt. Space reserved for non-powder mechanics later. Fan and metal are easy future adds.

### Alternate game directions to track (2026-04-21)

> Basically, I think there's two more directions we could take this game. In one of them, we try to be satisfactory or factorio but in 3D. And in the other, we try to do something like Zachtronics or a puzzle game. So maybe a puzzle game that involves cellular automata mechanics or something. or just massive physical scale things. I want you to track those as directions we can go, and potentially block some future stuff on that sort of thing. We're not going to block the demo for now, but we might want to pivot to this direction and then make a different demo.

Two alternate directions, tracked but not yet chosen:
1. **3D Factorio / Satisfactory** — logistics / automation game in 3D voxel CA space.
2. **Zachtronics-style puzzle** — CA-mechanics puzzles, or puzzles that exploit massive physical scale.

Current demo (megastructure "sandbox → wow reveal") is not blocked on this choice. But future work that would lock us out of either direction should be flagged before landing. If the crew is ever weighing two implementations and one preserves optionality toward these, prefer that one.

---

## Extracted requirements (derived from the user's words above)

These are the operational specifications. If one of these ever disagrees with the verbatim section above, the verbatim section wins.

### Hard requirements

- **3D cellular automata, not just 3D Game of Life.** General-purpose CA that can support "game-like physics." (Verbatim: "3D cellular automata in general that can support some game-like physics")
- **Voxel-based.** (Verbatim: "we'll want voxels and everything")
- **Material-type CA, not scan-based.** The user explicitly rejected scan-based: "We absolutely do not want to do scan-based like Noita." The user chose material-type CA over pure CA for physics reasons.
- **Hand-authored material interactions.** (Verbatim: "I think we should certainly support material interactions to be hand authored, and im down to have a lot of this")
- **Support "a ton of stuff" and redstone-like computation.** (Verbatim: "help simulating a ton of stuff, possibly redstone-like computation")
- **Particles supported alongside pure CA.** (Verbatim: "pure CA to start, but support particles")
- **64³ initial demo, scaling to 4096+, aiming for infinitely explorable worlds.** (Verbatim: "we'll start with 64^3 to demo, then scale up. def shooting for 4096+, prob will aim for infinitely explorable world")
- **Cross-platform: Mac, Linux, Windows.** (Verbatim: "I'd like this to be cross-platform, to work on Mac, Linux and Windows")
- **Native binaries are primary, web build stays an open option.** (Verbatim: "I kind of want to do native binaries… Definitely keep WebBuild as an open option.")
- **Maximum power.** (Verbatim: "I kind of just want the absolute most power.")
- **Infinite procedural world generation with dungeons and terrain, Minecraft-style.** (Verbatim: "ideally this thing is infinitely generating in each direction and it'd be nice to generate some interesting dungeon-like things or just terrain. Sort of like how Minecraft generates stuff.")
- **State-of-the-art SVDAG rendering is a design target, researched today.** (Verbatim: "lets also do some SVDAG rendering today to reserve a spot for it in the complexity budget. and can we aim for the state of the art?")
- **Direct octree operation, avoiding flatten-to-grid for everything except testing/scaffolding.** (Verbatim: "I'm down to use this for testing, but definitely want to see if we can avoid this for every single aspect of the game.")
- **Keep an eye on terrain-gen performance and surface issues early.**

- **This is a game, not a tech demo.** The engine should support multiple games (metaverse direction), but the first product is a game with a player character. (Verbatim: "definitely not a tech demo or just a basic toy")
- **Player character with world interaction.** First-person or third-person character that can walk around and place/interact with materials. Not god-mode-only. (Verbatim: "a character, and still able to interact with the world. Probably placing material, but definitely seeing stuff.")
- **Demo experience: normal→extraordinary gradient.** Launch should look familiar (Minecraft-ish), then reveal increasingly alive/dynamic world as player explores. (Verbatim: "start with a basic voxel thing... then they start seeing crazier and crazier things")
- **Powder-game physics first, not redstone.** Gravity, fluids, fire/temperature. Redstone-style logic comes later. (Verbatim: "primarily powder game stuff or maybe fire. Not really redstone to begin with.")
- **Materials should feel alive.** Default behavior is movement/interaction, not static blocks. All material stacks lean powder-game. (Verbatim: "stuff moving around by default, a little more alive")
- **Multiple terrain types / world presets.** Not locked to one biome. Could start in a dungeon, could start on terrain. Configurable. (Verbatim: "a lot of different worlds to step into")
- **Material namespaces per game.** Engine supports different material stacks for different front-end games. One standard Minecraft-ish-but-alive set as baseline. (Verbatim: "different namespaces of materials depending on the front end game on top of this")
- **Feature set TBD — not copying Minecraft.** Crafting, RPG mechanics, specific features not yet decided. Require human gate. (Verbatim: "I don't necessarily want to do that. I don't know if I want crafting.")

### Soft requirements

- **More than 256 material types is desirable if performance allows.** (Verbatim: "I'm willing to start with 256 but it could be cool to just support more unless it becomes a performance issue")
- **Fold metadata into material as tagged pointer space.** (Verbatim: "Maybe we fold meta into the material, some sort of tagged pointer space?")
- **Experiment with physics.** (Verbatim: "physics: lets experiment")
- **Emergent behavior is acceptable alongside hand-authored.** (Verbatim: "and im down to have a lot of this (and less, emergent. whatever works)")
- **Aim for performance local optima in game design.** (Edward 2026-04-20: "I legitimately might be pretty interested in making... designing the game toward stuff that is like higher performance that like fits in L2 or something. So let's just keep that in mind and add that to our high level design stuff." Refined 2026-04-20: "L2 is negotiable, but I want to aim for great performance local optima.") The principle is that game-design choices (rain density, per-material churn, procedural variety, scene structure) should consciously target a performance local optimum available at the current scale — not drift into the space between optima where we pay DRAM costs without the visual payoff of a truly-unique world. L2-fit (<~4 MB reachable DAG, ~100k interior nodes) is one such optimum on M1 MBA: bandwidth-free raycast, 3–5× faster than the DRAM regime. At 4096³ with streaming, the relevant optimum shifts to something like "streaming-bandwidth-matched active region + high dedup rate" — different number, same principle. Favored shapes in general: structural repetition (high SVDAG dedup), localized active regions, "many drops few subtree shapes" vs "every cell unique." See `docs/perf/svdag-perf-paper.md` §3.5 for the current L2-optimum math; other optima get characterized in the paper as they become relevant.

- **Game logic rides spatial memo.** (Edward 2026-04-22, following jw3k.1 landing: "I was hoping that any kind of water stuff we had would ride Spatial Memo or whatever. Like I was hoping that basically the whole game engine would run on Spatial Memo and be fast, and I really hope we're not doing any kind of linear thing with any moving stuff.") The architectural target is: every per-generation operation that touches the world — CA rule, block rules, gravity / gap-fill, fluid propagation, particle behavior, eventually entity updates — should be expressible as something hashlife's `(NodeId, parity)` memo can cache across ticks. Linear passes over active cells or active columns are an anti-pattern, regardless of constant factor. The payoff is structural: a memoized step is O(unique sub-tree shapes) per generation with compounding reuse across frames, while a linear pass is O(active region) every frame forever. The measured cost today (jw3k.1 baseline 2026-04-23: `column_walk` 85% of step, `step_node` 12%) is the exact pattern this principle rules out — the fast path is dwarfed by the non-memoized gravity pass. Implication for design: pick simulation rules whose per-generation light cone is bounded at every resolution (hashlife's precondition). If a visually-desirable rule breaks that (gap_fill's vertical unboundedness in densely packed columns is the current counterexample), the default response is to **redesign the rule** to be local-per-tick (multi-tick falling with a `falling` marker, etc.) rather than accept a linear workaround. **Resolved 2026-04-24:** gap_fill's cascading in-place sweep is being killed; replacement is a non-cascading 1-radius local rule (gaps bubble at 1 cell/tick, multi-tick settle). Tracked: `hash-thing-qy4g` (epic, deprioritized to P3 2026-04-24 — see below; option G already shipped) + `hash-thing-vfrr` (P2 research, explores rarefaction-free Margolus alternatives that would drop the gap-fill phase entirely). Light-cone evidence: `hash-thing-7nj6`. See `docs/perf/svdag-perf-paper.md` for the theoretical frame cost assuming this invariant holds.

- **Per-tick spatial memo is the production path; multi-tick macro-skip ("true hashlife") is deferred.** (Edward 2026-04-24.) The hashlife machinery in `src/sim/hashlife.rs` has two layers: (1) `step_recursive` — single-tick step that memoizes shared subtrees within one tick, used by production main loop; this is what "spatial memo" refers to in this spec, and it's the load-bearing perf path (m1f.15: 37× speedup, settled worlds 0.0 ms/frame). (2) `step_recursive_pow2` + `step_node_macro` family — multi-tick macro-skip that does 2^k generations per memo lookup; **currently test-only, deprioritized to P3.** Kept-on-file use case: catching the sim back up if it lags a lot at once (large time-step jump after a stall, scrubbing forward through stored state). May find other uses later. Until then, do not pull macro-skip beads as priority work — they don't block production. Umbrella: `hash-thing-82bt` (deferred-macro-skip epic, P3). Deprioritized children: `hash-thing-qy4g`, `hash-thing-qy4g.3`, `hash-thing-qy4g.4`, `hash-thing-krbw`, `hash-thing-sjpq` (the y=0 macro-vs-brute divergence sits in this deferred layer, not in production `step_recursive`). **If you trip over a macro-skip-related test failure or error, do not investigate** — add a one-line "tripped" comment to `hash-thing-lyqz` (the silencing throttle); when ≥3 trips from ≥2 seats accumulate, any seat may claim and `#[ignore]` the offending tests. Don't delete macro-skip code itself — the whole point of deferring rather than deleting is keeping the catch-up use case available.

### Design gates (what needs edward vs what's clear from spec)

**Autonomous (no gate needed):**
- Powder-game physics: gravity, fluids, fire, temperature — this is the priority
- Making materials feel alive / move by default
- Multiple terrain presets / world types
- Player character movement and camera
- Performance work, engine internals, octree ops
- Material interactions that are obvious powder-game staples (sand falls, water flows, fire spreads)
- Sandbox mode (free placement, observation, experimentation)
- Player-as-physical-presence: velocity, collision displacement, world impact from movement
- Metroidvania progression: walk → break (limited materials) → create materials → swim → fly (Superman-style). Progression-as-tutorial, no text prompts.
- Entity/machine placement as the primary interaction model (not block-by-block construction)
- Clone block and Powder Game material set (fan, firework, lava, ice, acid, oil, gunpowder, steam, gas, metal, vine)
- Hotbar + inventory (no crafting). Items auto-equip, hotbar to switch.
- Megastructures as first demo world (confined start → spectacle reveal)
- Physics creatures, environmental entities (geysers, volcanos), critters as entity types to explore
- Demo scenes: confined-space → big-reveal with visible CA activity at scale
- 30-sec and 5-min demo experiences building to a wow moment

**Requires human gate:**
- Roguelike layer specifics — deferred until sandbox + metroidvania are solid (Risk of Rain 2 as reference when ready)
- Voxel scale decision (player-sized vs much smaller) — awaiting perf data from m1f
- Specific entity creature designs beyond the explored categories (physics creatures, environmental, critters)
- Anything that makes it look/feel like a specific existing game (Minecraft clone, Terraria clone, etc.)
- "Permanent equip with loadout swapping" as alternative to hotbar — explore later

### Current game direction (crew reference — no gate needed to build toward this)

**Identity:** Player-as-physical-presence in a living world. Sandbox + metroidvania + destructible environment first. Roguelike layered on later (Risk of Rain 2 as reference). No crafting.

**Interaction model:** Entities/machines, not block construction. Clone blocks (material sources). Mobile physics creatures. Environmental entities (geysers, volcanos). Player places things that DO things. Hotbar + inventory, no crafting. Movement progression: walk → break → create → swim → fly (Superman-style).

**Progression:** Metroidvania-as-tutorial. Confined start, each unlock opens a new zone with the next unlock. No tutorial text — learn by doing. Hotbar fills as player progresses.

**Demo target materials:** Sand (gravity/collapse), water (flow/fill), airborne particles (rain/gas), explosions/fireworks. Static materials (stone/dirt) as structural backdrop. Full Powder Game set: clone, fan, firework, lava, ice, acid, oil, gunpowder, steam, gas, metal, vine.

**Demo world:** Lattice (repeating geometric megastructures). Confined start → corridor with hints → lattice interiors → panoramic reveal. Scale is withheld until the final vantage point. Follow-up worlds: Inverted Mountains, then World Tree or Organic Towers.

**Wow moment ladder:**
1. 30-sec: something visibly 3D and alive — rain, sand avalanche, explosion
2. 5-min: confined start → big reveal with lots of CA activity
3. Aspirational: floating islands, water cascading between them, particles in air, the whole world in motion

**What to optimize for when making judgment calls:** Does it make the world feel more alive and responsive to the player? Does it showcase 3D CA physics doing something no other engine does? If yes, build it. If it's a Minecraft feature that doesn't leverage CA physics, skip it.

### Questions the user raised that are not yet resolved

- **Can we do Hashlife with scan-based despite correctness bugs?** (Verbatim: "I do wonder if we could do hash life with a scan based thing though, aside. Like, we could just eat the issue with the bugs with locality or whatever.") — Current position: not planning to, but keeping the thought alive.
- **Does probabilistic CA work with Hashlife?** (Verbatim: "Probabilistic, I don't know if that works with hash life.") — Yes, if randomness is a deterministic function of `(position, generation)`. Noted for later.

### Ownership

The user explicitly granted autonomy: "the one level up is what we'll go for, for architecture. but one level up in *your ownership of this* is a different thing." This means: when the user specifies an approach, offer to own the problem; when they give a problem, own the goal; when they give a goal, own the strategy. Make decisive calls. Don't ask permission for things that can be decided.

---

## Architecture (current plan)

### Language and toolchain

**Rust + wgpu.** Chosen for:
- Cross-platform graphics (Metal / Vulkan / DX12) with zero platform-specific code
- Natural fit for hash-consed data structures (HashMap, ownership)
- No GC pauses
- Easy parallelism via rayon when needed
- Native binaries on Mac/Linux/Windows from a single codebase
- Web/WASM build remains available via `wgpu` → WebGPU target

**Rejected: Voxel Quest.** Investigated during the design session. Found to be an isometric-only GPU procedural city generator, not a general-purpose voxel engine. GenerateVolume shader is GPU-resident, no external data injection API, Windows-centric build, abandoned since 2016, OpenGL-legacy. Would be more work to gut than to write from scratch.

**Compilation time strategy:** Optimize dependencies even in dev builds (`[profile.dev.package."*"] opt-level = 2` in Cargo.toml) so the first build is slow but incremental rebuilds are fast (~4s). wgpu is heavy but this gives us the best of both worlds.

### Core data structure: hash-consed 3D octree

- **Nodes are canonical.** Every unique octree node exists exactly once; identical subtrees share a single `NodeId`. This is the same trick as 2D Hashlife's quadtree interning, extended to 8-way branching.
- **Level 0 = single cell. Level n = 2^n on each side.** 8 children per interior node, indexed by octant `(x+y*2+z*4)`.
- **Empty space is free.** A 4096³ mostly-empty world costs almost nothing because empty subtrees are all the same `NodeId`.
- **Store tracks population, supports step-result memoization.**

### Cell representation

- **Start: 16 bits per cell.** This gives 65536 distinct states, which is "material ID + packed metadata" folded into a single tagged integer. Hashlife doesn't care about the semantic split — it just memoizes patterns of states. State ID → material properties via a lookup table on CPU and GPU.
- **Expand to 32 bits if we hit the wall.** Doubles memory but the DAG compression means we only pay for *distinct* configurations that actually appear, not the theoretical maximum.
- **Rationale from user:** "Maybe we fold meta into the material, some sort of tagged pointer space?" — Yes, that's exactly this approach.

#### Possible direction: structural material class (workshopping)

Not committed — making space for this idea. The material ID space could reserve a range for **structural materials** that hashlife treats as fixed points (always return self). Two sub-classes:

- **Structural-breakable:** player can mine, but CA rules and explosions treat them as inert walls. Terrain, dungeon walls, wood.
- **Structural-unbreakable:** nothing breaks these. Bedrock, barriers, puzzle boundaries.

Why this is interesting for hashlife: a subtree composed entirely of air + structural materials is **inert** — it memos on first touch and never recomputes. Since the megastructure demo world is mostly structural with active materials only on surfaces/edges, this could make the vast majority of the world free to step. Fixed-point detection (m1f.14) reduces to a range check on leaf IDs.

Possible encoding (illustrative, not final):
```
0x0000              = AIR (inert)
0x0001..0x7FFF      = active materials (CA rules apply)
0x8000..0xBFFF      = structural-breakable (visual variety, hashlife inert)
0xC000..0xFFFF      = structural-unbreakable (hashlife inert)
```

Open questions: whether the breakable/unbreakable distinction belongs in the cell ID or in a material property table. Whether "structural" should be a bit flag or a range. Whether indirection (cell points to extension table) is worth the hashlife-memo complexity.

### World scale

**One cell ≠ one meter.** Cells are a sub-meter spatial unit; `CELLS_PER_METER`
(declared in src/scale.rs, currently `4.0`) converts player-facing physical
quantities (m/s, gravity, bounding box, interact range) to cells. At the
current setting each cell is 0.25 m along a world axis, so the default
`DEFAULT_VOLUME_SIZE = 8192` world is 2048 m on a side, with 8192³ ≈ 5.5 × 10¹¹
cells worth of addressable resolution. The DAG never pays for unique cells,
only for unique sub-tree structure, so the cell count is set aggressively.

**The knob is designed to move.** Bumping `CELLS_PER_METER` (and
`DEFAULT_VOLUME_SIZE` in lockstep so the world-in-meters stays fixed) shrinks
cells further — move away from Minecraft-scale blocks toward
particle/sand-grain resolution. Terrain generator parameters are expressed in
scaled cells via `TerrainParams::for_level` and follow along automatically;
demo waypoints are fractions of the world side; rendering normalizes to
`[0,1]` so neither needs per-bump retuning. See hash-thing-69cq for the first
bump (1.0 → 4.0 cells/m, 2048³ → 8192³).

**Addressability headroom.** `WorldCoord` is `i64`, so per-axis coordinates
can represent cell positions far past any plausible scale-knob bump: at
`CELLS_PER_METER_INT = 16` a 2048-meter world spans 32,768 cells per axis,
comfortably inside `i64::MAX`.

### Simulation model

**Hybrid Path D: reaction-phase (pure CA) + movement-phase (Margolus blocks).** Both phases operate directly on the octree via recursive Hashlife stepping; the flatten-to-grid path is a temporary testing scaffold only.

**Operating model: two phases (CaRule + BlockRule).** A third phase (LocalRule) is wired into the stepper as an extension point but currently runs only `is_self_inert` rules — i.e., it is a no-op in production. See "Phase 3" below for why it exists and the conditions under which it would be activated.

**Phase 1 — Reaction phase (pure CA, 26-neighborhood).**
Fire spreads, temperature diffuses, signals propagate, chemical reactions fire. Strictly local, no cell movement. Hashlife memoizes this perfectly because every cell independently computes its next state from its neighborhood. **No gap-fill or post-Margolus fix-up runs here** — see Phase 2 for how internal gaps close.

**Phase 2 — Movement phase (Margolus 2×2×2 blocks).**
Gravity, fluid flow, particle displacement. The world is partitioned into non-overlapping 2×2×2 blocks. A block rule takes 8 cells in, produces 8 cells out — which means it can *permute* cells within the block (swap sand with air below it = gravity). The partition offset alternates each generation (even steps: blocks at (0,0,0); odd steps: blocks at (1,1,1)) so cells can travel across block boundaries over multiple steps. Mass is conserved automatically because you're rearranging, not creating or destroying.

**Solid stratification (hash-thing-nagw).** GravityBlockRule and FluidBlockRule gate the vertical-swap predicate on a `Phase` classification (`Solid | Liquid | Gas`) in addition to density: a swap fires only when the density-heavier cell is on top **and at least one of the two cells is non-Solid**. Solid-on-solid pairs never swap regardless of density gap, so heavy granular materials (e.g. SAND, density 1.5) cannot sift through lighter granular materials below them (e.g. GUNPOWDER, density 1.4). Solid-on-gas, gas-on-gas, and any liquid-involving pair still swap by density. Phase is a pure function of the cell (`material_phase` in `terrain/materials.rs`), keeping the block rule a pure function of block contents and preserving the `(NodeId, parity)` hashlife memo key.

**Internal-gap closure rides parity-flip, not a separate pass** (qy4g epic decision 2026-04-26). For **static** SOLID-AIR-SOLID configurations (e.g. a settled pile sitting on the floor), the gap is inside a block on the opposite parity, where the existing gravity rule closes it within 1-2 ticks. **For free-falling columns the parity-flip does NOT close gaps in flight** — the gap pattern is "co-moving" with the falling water and the entire column appears as an every-other-y checkerboard for the duration of the fall, only compacting once the leading edge hits a solid surface. This is a known visible artifact of option G; the test `water_block_gaps_close_via_parity_flip` validates **post-compaction** contiguity, not mid-flight closure. Pointwise-parallel mass-conserving gap-fill is an unsolved problem in the literature (Hahm 2023, Wikipedia Block CA, Noita GDC 2019); the field's accepted answer is parity-flip resolution OR sequential post-pass. We pick parity-flip. **Documented fallbacks if the in-flight checkerboard proves unacceptable in playtest:** (A) accept mass non-conservation in a CaRule-folded promote-only rule, or (F) re-introduce a cheap dirty-column post-pass walking only BlockRule-touched columns. See qy4g epic notes.

**Phase 3 — Local-context phase (LocalRule, ±1y read). Wired but unused in production.**
Tightly-scoped per-cell rules that read self + immediate vertical neighbors only (above, below) and run **after** Phase 2. The trait + recursive-stepper integration ship in qy4g.1 as an extension point, but no non-`is_self_inert` LocalRule is registered today; the only impl is `IdentityLocalRule`, which exists to pin the equivalence test. **Activating Phase 3 with a non-trivial rule reopens an architectural problem** (the 8³ level-3 base case is short by 2 y-cells once Phase 3 contributes a 1-cell light cone; fixing it requires threading y-context through every step_node recursion — see hash-thing-mzct for the parked design work). Default route for any new "fix-up after Margolus" rule: fold it into Phase 1 (CaRule) and accept the 1-frame delay. Reach for Phase 3 only if you genuinely need post-Margolus state THIS generation, and unblock mzct first.

**Phase ceiling: 3 phases, with Phase 3 gated.**
Each phase pays a per-gen light cone slot, a buffer-geometry assumption, a cache-key dimension, and a recursion-threading obligation. Adding a 4th phase requires explicit justification (a bead with a concrete value-vs-tax case) — not a one-liner. Default disposition for any new "I want to read neighbors" rule: fold it into Phase 1 or Phase 2, or run it as a non-recursive pass (see `apply_gap_fill_column_path`). Activating Phase 3 with a non-trivial rule is a one-way door: it requires landing the slab-threading architecture (hash-thing-mzct) first. The 3-phase cap was originally pinned by the hash-thing-mzct postmortem (qy4g.1's addition of Phase 3 over-budgeted the level-3 base case y-light-cone, forcing slab-threading through every recursion level); the two-phase operating decision (qy4g 2026-04-26) defers actually paying that tax.

**Why Margolus:**
- Pure CA can't swap cells, so gravity is hard without tricks.
- Scan-based sims (Noita) are explicitly rejected.
- Margolus preserves mass, is deterministic, and is Hashlife-compatible because the partition offset is a function of generation parity.
- Lineage: Norman Margolus (MIT, 1980s), used throughout CA research for physics-like simulations.

**Why this is Hashlife-compatible (PARTIALLY INCORRECT — see below):**
- Both phases are deterministic functions of local state.
- No scan order dependency, no global state.
- Memoization key is `(NodeId, parity)` — node contents + generation parity. BlockRule is now a pure function of block contents (no coord/generation context), so spatial memoization works for all worlds (m1f.9, m1f.13, 9ww).
- Margolus partition parity is implicit in the generation count, which maps cleanly onto tree levels in the Hashlife recursive step.

**RESOLVED (m1f.9, m1f.13, 9ww):** This tension was resolved by making BlockRule a pure function of block contents only (no position/generation context), then enabling node-local Margolus alignment so the hashlife cache key is `(NodeId, parity)` — fully spatial memoization-compatible. All worlds, including those with block rules, now share cache entries for identical subtrees.

**Material interactions: hand-authored.** Per-material-pair rules like "sand + air-below → swap," "fire + wood → fire," "water + lava → obsidian." The user explicitly endorsed hand-authored over emergent-only.

### Terrain generation

- **Multi-resolution procedural generation at octree node granularity.** Not cell-by-cell — `gen(node_region) → NodeId` evaluates noise at region corners; if the region is uniform, return a uniform node without descending. Only subdivide where noise transitions occur (surface, biome boundaries).
- **Lineage:** similar to how SVO terrain engines work; different from Minecraft's flat-chunk approach.
- **Noise functions (Perlin / simplex / 3D Voronoi) for macro structure.** Heightmap, biomes, continents, ore distribution.
- **CA for micro refinement.** Erosion, vegetation spread. Lineage: classic technique going back to the 1990s, used in Dwarf Fortress, Terraria, many roguelikes.
- **Dungeon generation as a layer on top.** Carve rooms/corridors, via procedural placement or WFC/graph grammars.
- **Infinite worlds via lazy octree expansion.** When the player/simulation reaches the edge of the current root, wrap the root in a bigger parent node whose other 7 children are "default terrain" nodes, generated lazily as needed.
- **Watch item:** monitor generation performance early. If per-chunk noise evaluation becomes the bottleneck, consider CA-based generation or SIMD noise.

### Rendering

**Phase A (now, up to ~256³–512³): flat 3D texture raycaster.** Upload the octree as a flat `R8Uint` (or wider) 3D texture each frame. Fragment shader does DDA raymarching. This is what's currently working. Adequate for demo and early development.

**Phase B (scale milestone, 1024³+): SVDAG raycaster.**
At 1024³ flat textures become 1GB (impossible). SVDAG is the state-of-the-art sparse voxel representation and is literally what our hash-consed octree already is.

**State-of-the-art research reviewed:**
- **Laine & Karras 2010, "Efficient Sparse Voxel Octrees"** (NVIDIA) — seminal GPU traversal algorithm, iterative descent with position tracking.
- **Kämpe, Sintorn, Assarsson 2013, "High Resolution Sparse Voxel DAGs"** — the foundational paper for SVDAG. Shows that hash-consing SVO subtrees gives 10-100× memory compression. 128K³ binary scenes interactively.
- **Jaspe Villanueva et al., "Symmetry-aware Sparse Voxel DAGs (SSVDAGs)"** — exploits mirror symmetries in addition to structural equality for additional compression.
- **Careil, Billeter, Eisemann 2020, "Interactively Modifying Compressed Sparse Voxel Representations" (HashDAG)** — **closest match to our use case.** SVDAGs are normally static. HashDAG embeds the DAG in a GPU hash table so it can be edited interactively (carving, filling, copying, painting) without decompression. This is *exactly* what we need for a dynamic simulation.
- **Molenaar et al. 2023, "Editing Compressed High-resolution Voxel Scenes with Attributes"** — extends HashDAG with per-voxel attributes (colors, materials). Required for our material-type CA.
- **Transform-Aware SVDAGs (2025)** — further generalizes symmetry exploitation to translations and novel transforms. Memory wins on top of everything above.
- **Occupancy-in-memory-location (Modisett, recent CGF)** — encodes presence bits via memory position, eliminating explicit masks for further compactness.
- **Sparse 64-trees (dubiousconst282, 2024)** — branching factor 64 (4×4×4) instead of 8, with a 64-bit occupancy mask. Claims ~3× better memory density than 8-ary SVO (~0.19 B/voxel vs ~0.57). Uses popcount on 64-bit masks for fast child lookup. Alternative spatial decomposition; would affect our Hashlife step. Noted as a future optimization direction, not adopted today.
- **Aokana 2025** — GPU-driven SVDAG framework with LOD + streaming, tens of billions of voxels, open-world games. The "full modern stack." Represents the scale ceiling of the technique.
- **Hybrid voxel formats (2024)** — different spatial structures at different tree levels for Pareto-optimal memory/speed tradeoffs.

**Our target for SVDAG rendering:** HashDAG-style structure (because we need dynamic editing), with attributes per leaf (because we need materials), and an iterative GPU traversal shader in the Laine-Karras lineage. Keep sparse-64-tree and SSVDAG compression as future optimization paths.

**Rendering phase roadmap:**
1. ✅ Flat 3D texture raycaster (working).
2. Serialize hash-consed DAG to GPU buffers as SVDAG.
3. Iterative DAG traversal shader. Start with correctness, then optimize.
4. Hook it up as an alternative render path. Toggle between flat and DAG.
5. Add HashDAG-style editing (diff-based upload of changed nodes).
6. Consider SSVDAG / transform-aware compression once baseline is stable.
7. Consider LOD streaming (Aokana-style) when scaling to 4096³+.

**Performance ownership:** the crew owns the SVDAG performance story top-down — any seat may pick up sub-beads. Mayor surfaces decisions and re-derives the spec from the paper. See `docs/perf/svdag-perf-paper.md` (long form, living) and `docs/perf/svdag-perf-spec.md` (re-derived intuition). For open research frontiers, dated experiments, and threads that have not yet graduated to the settled model, see `docs/perf/svdag-research-log.md`. Reference hardware: M1 MacBook Air 8 GB. Bead: `hash-thing-stue`.

### Distribution

- **Native binaries for Mac, Linux, Windows from day one.** `cargo build --release` with target triples. CI matrix via GitHub Actions.
- **macOS notarization** required for distribution outside the App Store. Scriptable with `xcrun notarytool`.
- **Linux: AppImage** for simplest single-file distribution (or bare binary).
- **Windows: bare .exe** works out of the box.
- **Steam integration via `steamworks-rs` when there's a game to ship.** Architecturally a thin layer added later, not a day-one concern.
- **Web build stays available but not the primary target.** `wgpu` compiles to WebGPU via WASM. Useful for shareable demos.

### Cross-cutting principles

- **Operate on the octree directly wherever possible.** Flatten-to-grid is only allowed as a testing scaffold, never as a permanent implementation.
- **Determinism is sacred.** Every simulation step, terrain-gen call, rendering pass must be a pure function of input state. No global PRNG, no scan order. Randomness comes from `hash(position, generation, seed)`.
- **Hash-consing is load-bearing.** The entire memory story — simulation, rendering, terrain — depends on it. Identical subtrees must share a single `NodeId`.
- **Measure early, especially on scale-sensitive paths.** Terrain generation, Margolus stepping, DAG serialization. Track memory and time per operation.

---

## Current state

> Live task tracking lives in `.beads/issues.jsonl` (via `bd`). This section is a
> narrative summary; `bd ready` / `bd list` is the source of truth for open work.

### Landed

- ✅ Cargo workspace: `ht-octree` (octree+rng), `ht-render` (renderer+svdag+shaders), root crate (sim+terrain+main). Incremental build 24s→8s for sim edits (cvu)
- ✅ Hash-consed octree with NodeStore (intern/lookup, flatten, from_flat, set_cell, stats)
- ✅ 3D Game of Life CA rules (4 presets: amoeba, crystal, 445, pyroclastic) — scaffolding for experimentation, to be replaced by material-type CA
- ✅ Brute-force grid stepping (temporary, will be replaced by recursive Hashlife stepping)
- ✅ Flat 3D-texture wgpu raycaster with orbit camera (mouse drag + scroll), fullscreen-quad fragment shader, DDA voxel traversal, directional lighting
- ✅ Main loop: keyboard controls (space pause, S step, R reset, 1-4 rule switch, V render-mode toggle, Esc exit)
- ✅ Builds and runs on macOS with Metal backend
- ✅ Pushed to `git@github.com:heated/hash-thing.git`
- ✅ **SVDAG rendering pipeline (hash-thing-5bb.1, 5bb.2, 5bb.3)**
  - `Svdag` serializes the DAG to a flat GPU buffer (9 u32 per interior: mask + 8 children; leaves inlined via high-bit marker); handles leaf roots via degenerate single-node interior (nch)
  - **Incremental uploads (5bb.5)**: persistent `FxHashMap<[u32; 9], u32>` content cache across frames — `update()` reuses offsets for unchanged subtrees, only appends new slots. Buffer layout: `nodes[0]` = root-offset header, `nodes[1..]` = append-only slots. Renderer uploads only the tail past a watermark + root header each frame
  - **Stale-slot compaction (bx7)**: `compact()` rebuilds from scratch when >50% of buffer is unreachable, reclaiming memory from superseded subtrees
  - `svdag_raycast.wgsl` iterative stack-based descent: pop-until-contains, descend-until-leaf, step-past-empty-octant
  - Dual renderer pipelines (Flat3D / Svdag), V toggles at runtime
  - CPU-side trace replica of the shader (`src/render/svdag.rs::cpu_trace`) with comprehensive test suite (~1700 lines)
  - **Bug fixes**: octant_of corner-exit tiebreak (6hd), inside-leaf exit-face normal (2nd), far-camera Laine-Karras entry clamp (27m), root_side-scaled step budget with magenta exhaustion sentinel (2w5), analytical entry-face normals (rv4), sub-pixel LOD cutoff (x5w — stops descent when voxel < 1 pixel, shades with first non-empty child color)
  - **CPU/GPU drift guards** (`src/render/mod.rs::wgsl_drift_guard`): 6 textual pins covering METADATA_BITS shift, octant_of tiebreak (6hd), entry-face normal cascade (rv4), inside-leaf fallback (2nd), traversal constants MAX_DEPTH/MIN_STEP_BUDGET/STEP_BUDGET_FUDGE (2w5), material palette vs registry (xev)
  - **Property tests**: midpoint-exact corner fuzz (6cc, 500 cases), forward-progress invariants, 27-ray sweep at MAX_DEPTH, far-camera stall regression, budget saturation
  - GPU timestamp queries for render timing (6x3), graceful TIMESTAMP_QUERY fallback
  - FrameOutcome enum with Occluded suppression, no hot-spin (8jp)
  - 31-bit node-count overflow assertion at write time (x9r)
  - `padded_bytes_per_row` helper for COPY_BYTES_PER_ROW_ALIGNMENT (mys)
  - Material palette sync with registry (xev): both shaders now match `MaterialRegistry::terrain_defaults()` colors
- ✅ **Material-type CA (epic `1v0`, complete)**
  - `1v0.1` 16-bit tagged cell: `Cell` packs 10-bit `material_id` + 6-bit `metadata` into `u16`. `METADATA_BITS=6` is pinned by drift guard in both shaders.
  - `1v0.2` `MaterialRegistry` with per-material rules, visual properties (label, base_color, texture_ref), physical properties (density, flammability, conductivity). `terrain_defaults()` registers air/stone/dirt/grass/fire/water. `gol_smoke()` bridges legacy GoL presets. GPU palette export via `color_palette_rgba()`.
  - `1v0.3` Per-material `CaRule` dispatch in `World::step`: `trait CaRule { fn step_cell(&self, center: Cell, neighbors: &[Cell; 26]) -> Cell; }`. `NoopRule` for static materials, `FireRule` (spreads from fuel, quenched by water), `WaterRule` (reacts with fire → stone). No central interaction table.
  - `1v0.4` Margolus 2×2×2 movement phase: `trait BlockRule { fn step_block(&self, block: &[Cell; 8]) -> [Cell; 8]; }`. Pure function of block contents (no position/generation context), enabling spatial memoization in hashlife. Alternating partition offset (even/odd gen). `GravityBlockRule` for density-based vertical swaps. Mass conservation enforced via debug_assert.
  - `1v0.5` `FluidBlockRule`: 2-phase block rule — gravity (vertical swaps by density) then lateral spread (fluid↔air swaps). Axis selection via `content_hash(block)` for determinism.
  - `1v0.6` Entity system: `EntityStore` with `EntityKind::Particle` (continuous position, velocity, TTL, collision, gravity). Mutation queue integration.
  - `1v0.7` Burning room demo scene: stone room with grass walls (fuel), fire corner, water pool. Key binding: B.
  - `1v0.8` Cell/block granularity: `CELLS_PER_BLOCK_LOG2=3` (8³ cells per gameplay block), `World::set_block()` API with auto-grow.
  - `1v0.9` Mutation channel: `WorldMutation` enum + `apply_mutations()` for entity→world edit API with runtime guards.
  - `1v0.10` Player entity: `EntityKind::Player(PlayerState)` with first-person camera (Tab toggle), WASD+mouse look, AABB collision, DDA raycast block place/break.
- ✅ **Hash-cons compaction (`88d`)**: `NodeStore::compacted(root)` rebuilds reachable graph into a fresh store via iterative post-order traversal (`8m7`). Called after every `commit_step` and on `seed_terrain` epoch boundaries.
- ✅ **Foundations progress (epic `h34`, 4/4 — complete)**
  - `h34.1` cell_hash PRNG: `hash(x, y, z, generation, seed) → u32` Hashlife-compatible deterministic source (src/rng.rs)
  - `h34.2` determinism audit: walked every file in sim + terrain-gen paths, no global PRNG / scan-order dependencies found, two minor follow-ups filed and closed (`99e` step_cache rule_id doc fix, `c6k` seed_center migrating to cell_rand_bool). Audit notes in `.ship-notes/ship-h34.2-determinism-audit.md`
  - `h34.3` perf measurement infra: `src/perf.rs` 64-sample ring buffer + `Perf::time(name, closure)` + consolidated per-generation log line with mean/p95 on `step_cpu`, `upload_cpu`, `render_cpu`; wall-clock auto-log cadence (q63); memory-watchdog MemStats with ratcheting peaks and `memory_bytes_estimate` (yb5); GPU `_gpu` metric family via timestamp queries (6x3)
  - `h34.4` Retired GoL3D scaffolding: legacy presets removed, `set_gol_smoke_rule` replaces old rule API
- ✅ **NodeStore hash-cons unit tests (`1lq`, `6gf.5`)**: intern idempotency, lookup round-trip, flatten/from_flat determinism, set_cell paths, uniform collapse, metadata, overwrite, accessor coverage
- ✅ **Octree extraction primitives (`6gf.6`)**: `extract_neighborhood` (26 Moore neighbors, toroidal wrap), `extract_block_2x2x2` (Margolus partition extraction), `flatten_region` (arbitrary AABB sub-box). Foundation for recursive Hashlife step.
- ✅ **Lazy root expansion (`e9h`, `5qp`, `37r`)**: `World::ensure_contains(x,y,z)` grows root octree by wrapping in bigger parent nodes. `ensure_region([min],[max])` is the batch form. OOB bounds checking on set_cell/get_cell (`819`) also landed. Bidirectional growth (`37r`): `World.origin` tracks world-space offset; negative-direction growth places old root at appropriate octant and shifts origin.
- ✅ **Out-of-bounds coordinate safety (`819`, `fb5`)**: `set_cell` panics on OOB with `#[track_caller]`; `get_cell` silently returns 0 for OOB (no storage growth). 7 boundary tests. `probe()` stepper-oriented read with OOB-empty promise (`90w`). `is_realized()` for boundary detection.
- ✅ **Terrain generation (epic `3fq`)**
  - Heightmap-based recursive octree gen with proof-based collapse (sky/deep-stone short-circuit)
  - ⚠️ Dungeon carving (3fq.3): was built and landed, then **reverted** from main (hash-thing-t2n.1 — landed without design gate). Code lives on `feature/dungeons` branch pending edward's design approval.
  - Per-phase perf tracking in GenStats: gen_region_us, node counts, noise bottleneck estimate (3fq.5)
  - Lazy terrain expansion (3fq.4): `ensure_region` generates heightmap for new sibling octants when `terrain_params` is set. Non-terrain worlds expand with empty nodes.
- ✅ **RealizedRegion value type (`ica`)**: `RealizedRegion { origin: [i64; 3], level: u32 }` encapsulates the world's spatial extent. `contains()`, `side()`, `octant_of()`, `Display` impl. `World::region()` returns a derived view.
- ✅ **WorldCoord/LocalCoord newtypes (`7zc`)**: `WorldCoord(i64)` for world-space, `LocalCoord(u64)` for octree-internal. Compiler-enforced separation prevents the coordinate-mixing bug class that 819 fixed. All World API methods migrated.
- ✅ **CI + release (epic `xb7`, 5/7)**
  - `xb7.1` 3-platform CI matrix (Linux + Mac + Windows), all actions SHA-pinned, `rust-toolchain.toml` channel pin, `Cargo.toml [lints.rust]` for first-party warning gating, actionlint job, gating `cargo check --all-targets`, gating `cargo fmt --check` (k5r), gating `cargo clippy -- -D warnings` (00f)
  - `xb7.3` Linux AppImage via linuxdeploy (packaging/linux/ desktop entry + placeholder icon)
  - `xb7.4` tag-triggered release workflow producing Windows `hash-thing.exe` via `gh release` CLI (no new third-party action deps). Sibling jobs for Mac (`xb7.2` notarization) extend as they land

### Completed epics

- ✅ **Recursive Hashlife stepping** (epic `6gf`, complete): all 13 beads landed including `6gf.8` main loop switchover, `6gf.10` defensive hardening, `6gf.11` deep recursion tests, `6gf.13` vestigial step_cache removal. `018` brute-force stepper absorbing boundary fix (CaRule matches hashlife). `m1f.12` stable NodeIds across compaction (cache key remapping instead of clearing).
- ✅ **Material-type CA** (epic `1v0`, complete): all 18 beads landed including `1v0.1` 16-bit cells, `1v0.6` entity system, `1v0.8` cell/block granularity, `1v0.10` player entity with first-person camera + AABB collision + DDA raycast block interaction.
- ✅ **SVDAG rendering** (epic `5bb`, 10/11): `5bb.1`–`5bb.5` serialization through incremental uploads, `bx7` stale-slot compaction, `5bb.7` palette sync, `5bb.8` zero-direction guard, `5bb.9` particle renderer, `5bb.10` HUD overlay. Remaining: `5bb.6` SSVDAG/LOD research (P4).
- ✅ Foundations & determinism (`h34`, complete): all 4 beads landed including `h34.4` retire GoL3D + `h34.5` iterative clone_reachable (8m7)
- ✅ Terrain generation & infinite worlds (`3fq`): heightmap gen + lazy terrain expansion + perf tracking. Cave CA removed (72s — O(n³) bottleneck). Sand biomes (`y2s`): low-freq noise selects sandy regions where grass/dirt → sand (gravity block rule). Sea-level water (`4t6`): `HeightmapField.sea_level` fills air below sea level with water; `classify_box` proof updated. Dungeon carving reverted (t2n.1, design gate); code on `feature/dungeons`. `l1t` terrain gen 15× speedup via `PrecomputedHeightmapField`: 2D heightmap + min/max mipmap for O(1) `classify_box` bounds + precomputed biome grid. `m1f.7.1` scale-aware `TerrainParams::for_level()`.
- ✅ **Build infra (`4yb`, `3sw`)**: shared `CARGO_TARGET_DIR` across worktrees (saves ~15GB), `[profile.bench]` for fast representative builds, benchmarks use `--profile bench` instead of `--release`. `3sw`: sccache + per-worktree target dirs to eliminate cargo lock contention.
- ✅ **Bug fixes**: `0xq` BlockRule boundary asymmetry (absorbing BCs instead of clipping), `08c` has_block_rule_cells O(n³)→O(nodes) walk, `bhi` NodeStore/SVDAG overflow diagnostics.
- ✅ **Demo playability (`x5w`)**: sim step moved to background thread (render loop unblocked), SVDAG raycast LOD cutoff (sub-pixel voxels short-circuit to nearest color, cutting GPU step count for complex/distant geometry).

### Recently completed

- ✅ **Core engine validation** (epic `m1f`, 15/15 complete): full loop works at scale. Hashlife stepping, SVDAG rendering, player interaction, edits — all validated. Key milestones: hashlife 37x speedup via spatial memoization, SVDAG depth 24 (16M³), half-res rendering for 60fps at 512-1024³, lattice megastructure demo scene with active materials. Renderer: step-count heatmap debug (H key), runtime LOD bias (L), render scale (+/-).
  - **Hashlife performance (m1f.15):** 512³ terrain 7147ms→192ms mean (37x), settled worlds 0.0ms/frame.
  - **Renderer performance (m1f.7.3):** 512³ render_gpu 49ms→14ms via half-res. 1024³ terrain at 17ms GPU.

### Later (P2+, from bd)

- ☐ Cross-platform distribution (`xb7`, 5/7): ✅ CI matrix (`xb7.1`), ✅ macOS notarization (`xb7.2`), ✅ Linux AppImage (`xb7.3`), ✅ Windows exe (`xb7.4`), ✅ WASM/WebGPU (`xb7.5`). Remaining: `brt` AppImage branding (P4), `xb7.6` Steam (P4 deferred)
- ☐ SVDAG research (`5bb.6`): SSVDAG / sparse-64 / LOD streaming once baseline is stable

---

## Minimum spec target (rendering / perf budget)

This is the hardware we **target for 60 FPS at the listed resolution**. It's an invariant we want to hit, not a "might work" floor — it's the budget that constrains what rendering / sim features we ship by default. Heavier features may live behind a higher tier (LOD bias, optional shadows, etc.) but the default experience must hit 60 FPS on this rig.

Sourced from edward's read of public hardware surveys (Steam HW Survey and similar). Numbers are not pinned to any specific survey snapshot; revise when the public-PC floor visibly moves.

| Component | Target |
|---|---|
| RAM | **8 GB** |
| CPU | **4 physical cores** (clock speed unconstrained) |
| GPU VRAM | **2 GB** (open to pushback; some features may gate behind higher tiers) |
| Display resolution | **1440p (2560×1440)** at 60 FPS |
| Disk install footprint | **≤ 2 GB** (negotiable) |
| Architecture | **x86_64** + **Apple Silicon (arm64)** |
| Operating systems | macOS, Linux, Windows (per `xb7` distribution work) |

Notes:
- "Minimum" here means **the default experience runs at 60 FPS at 1440p on this rig.** If we can't hit that, either we lower the rendering budget or the spec target moves up — not both silently.
- Felt FPS is the metric, not log-reported `render_gpu` (see hash-thing-dbz5). A fix that moves a single number without moving the felt experience does not count as hitting the target.
- Tiered features are allowed: e.g. shadows / higher-quality LOD can require more VRAM, as long as the 2 GB tier still renders cleanly with the defaults.
- Higher resolutions (4K, ultrawide) and beefier rigs are nice-to-have, not the design target.

---

## Anti-goals

Things we are NOT doing, to be clear about scope:

- **Not a scan-based Powder Game clone.** Explicitly rejected by the user.
- **Not a standard game engine.** We're building the simulation substrate; game mechanics come after.
- **Not physically-accurate continuum physics.** No Navier-Stokes solver, no lattice Boltzmann. Material-type CA with hand-authored rules is the path.
- **Not a voxel editor first.** Editing is a consequence of needing dynamic simulation, not the primary goal.
- **Not using Voxel Quest.** Investigated and rejected.
- **Not web-first.** Native is primary; web is an optional shareable path.
