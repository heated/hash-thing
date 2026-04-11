# Ashfall — Design Spec

> A 3D voxel cellular automaton engine with hash-consed octree storage, Hashlife-style simulation stepping, Margolus-block physics, material-type CA rules, and SVDAG rendering. Targeting infinitely explorable worlds with emergent physics and redstone-style computation.

Repo: `git@github.com:heated/ashfall.git`

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

### Soft requirements

- **More than 256 material types is desirable if performance allows.** (Verbatim: "I'm willing to start with 256 but it could be cool to just support more unless it becomes a performance issue")
- **Fold metadata into material as tagged pointer space.** (Verbatim: "Maybe we fold meta into the material, some sort of tagged pointer space?")
- **Experiment with physics.** (Verbatim: "physics: lets experiment")
- **Emergent behavior is acceptable alongside hand-authored.** (Verbatim: "and im down to have a lot of this (and less, emergent. whatever works)")

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

### Simulation model

**Hybrid Path D: reaction-phase (pure CA) + movement-phase (Margolus blocks).** Both phases operate directly on the octree via recursive Hashlife stepping; the flatten-to-grid path is a temporary testing scaffold only.

**Phase 1 — Reaction phase (pure CA, 26-neighborhood).**
Fire spreads, temperature diffuses, signals propagate, chemical reactions fire. Strictly local, no cell movement. Hashlife memoizes this perfectly because every cell independently computes its next state from its neighborhood.

**Phase 2 — Movement phase (Margolus 2×2×2 blocks).**
Gravity, fluid flow, particle displacement. The world is partitioned into non-overlapping 2×2×2 blocks. A block rule takes 8 cells in, produces 8 cells out — which means it can *permute* cells within the block (swap sand with air below it = gravity). The partition offset alternates each generation (even steps: blocks at (0,0,0); odd steps: blocks at (1,1,1)) so cells can travel across block boundaries over multiple steps. Mass is conserved automatically because you're rearranging, not creating or destroying.

**Why Margolus:**
- Pure CA can't swap cells, so gravity is hard without tricks.
- Scan-based sims (Noita) are explicitly rejected.
- Margolus preserves mass, is deterministic, and is Hashlife-compatible because the partition offset is a function of generation parity.
- Lineage: Norman Margolus (MIT, 1980s), used throughout CA research for physics-like simulations.

**Why this is Hashlife-compatible:**
- Both phases are deterministic functions of local state.
- No scan order dependency, no global state.
- Memoization key is just the node contents + phase.
- Margolus partition parity is implicit in the generation count, which maps cleanly onto tree levels in the Hashlife recursive step.

**Material interactions: hand-authored.** Per-material-pair rules like "sand + air-below → swap," "fire + wood → fire," "water + lava → obsidian." The user explicitly endorsed hand-authored over emergent-only.

### Terrain generation

- **Multi-resolution procedural generation at octree node granularity.** Not cell-by-cell — `gen(node_region) → NodeId` evaluates noise at region corners; if the region is uniform, return a uniform node without descending. Only subdivide where noise transitions occur (surface, caves, biome boundaries).
- **Lineage:** similar to how SVO terrain engines work; different from Minecraft's flat-chunk approach.
- **Noise functions (Perlin / simplex / 3D Voronoi) for macro structure.** Heightmap, biomes, continents, ore distribution.
- **CA for micro refinement.** Cave smoothing ("4-5 rule" for organic caves), erosion, vegetation spread. Lineage: classic technique going back to the 1990s, used in Dwarf Fortress, Terraria, many roguelikes.
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

- ✅ Cargo project scaffold, `src/{octree,sim,render}` modules
- ✅ Hash-consed octree with NodeStore (intern/lookup, flatten, from_flat, set_cell, stats)
- ✅ 3D Game of Life CA rules (4 presets: amoeba, crystal, 445, pyroclastic) — scaffolding for experimentation, to be replaced by material-type CA
- ✅ Brute-force grid stepping (temporary, will be replaced by recursive Hashlife stepping)
- ✅ Flat 3D-texture wgpu raycaster with orbit camera (mouse drag + scroll), fullscreen-quad fragment shader, DDA voxel traversal, directional lighting
- ✅ Main loop: keyboard controls (space pause, S step, R reset, 1-4 rule switch, V render-mode toggle, Esc exit)
- ✅ Builds and runs on macOS with Metal backend
- ✅ Pushed to `git@github.com:heated/ashfall.git`
- ✅ **SVDAG rendering pipeline (hash-thing-5bb.1, 5bb.2, 5bb.3)**
  - `Svdag::build` serializes the DAG to a flat GPU buffer (9 u32 per interior: mask + 8 children; leaves inlined via high-bit marker)
  - `svdag_raycast.wgsl` iterative stack-based descent: pop-until-contains, descend-until-leaf, step-past-empty-octant
  - Dual renderer pipelines (Flat3D / Svdag), V toggles at runtime
  - CPU-side trace replica of the shader (`src/render/svdag.rs::cpu_trace`) + 4 regression tests
  - Epsilon bug fixed: pop-check slack was beating step-past advance on multi-axis boundary crossings
- ✅ **Foundations progress (epic `h34`, 3/4)**
  - `h34.1` cell_hash PRNG: `hash(x, y, z, generation, seed) → u32` Hashlife-compatible deterministic source (src/rng.rs)
  - `h34.2` determinism audit: walked every file in sim + terrain-gen paths, no global PRNG / scan-order dependencies found, two minor follow-ups filed and closed (`99e` step_cache rule_id doc fix, `c6k` seed_center migrating to cell_rand_bool). Audit notes in `.ship-notes/ship-h34.2-determinism-audit.md`
  - `h34.3` perf measurement infra: `src/perf.rs` 64-sample ring buffer + `Perf::time(name, closure)` + consolidated per-generation log line with mean/p95 on `step_cpu`, `upload_cpu`, `render_cpu`. One bead remains (retire-GoL `h34.4`, blocked on 1v0 material CA)
- ✅ **NodeStore hash-cons unit tests (`1lq`)**: intern idempotency, lookup round-trip, flatten/from_flat determinism, set_cell paths
- ✅ **CI + release (epic `xb7`, 2/6)**
  - `xb7.1` 3-platform CI matrix (Linux + Mac + Windows), all actions SHA-pinned, `rust-toolchain.toml` channel pin, `Cargo.toml [lints.rust]` for first-party warning gating, actionlint job, gating `cargo check --all-targets` before warn-only clippy
  - `xb7.4` tag-triggered release workflow producing Windows `hash-thing.exe` via `gh release` CLI (no new third-party action deps). Sibling jobs for Mac (`xb7.2` notarization) and Linux (`xb7.3` AppImage) extend as they land

### Next up (P1, from bd)

- ☐ **Recursive Hashlife stepping** (epic `6gf`: `6gf.1` recursive step, `6gf.2` memoize by (NodeId, phase), `6gf.3` correctness harness vs brute-force, `6gf.4` Margolus parity threading). Currently we flatten-then-step; this is the biggest single perf unlock.
- ☐ **Material-type CA** (epic `1v0`: `1v0.1` 16-bit tagged cell, `1v0.2` material registry, `1v0.3` hand-authored interaction table, `1v0.4` Margolus movement phase). Replaces the GoL3D scaffolding.
- ☐ **SVDAG continuation**: `5bb.4` per-leaf material attributes (Molenaar-style), `5bb.5` HashDAG-style incremental edit uploads (so we don't re-serialize the whole DAG every step).
- ☐ **Hash-cons compaction (`88d`)**: NodeStore is currently append-only — every dead generation's subtrees are retained forever. Plan in flight (`plan-flint-88d.md`, fresh-store rebuild via `NodeStore::compacted`).

### Later (P2+, from bd)

- ☐ Foundations & determinism (`h34`): retire GoL3D scaffolding (`h34.4` — blocked on 1v0 material CA landing)
- ☐ Terrain generation & infinite worlds (`3fq`): multi-res `gen(node_region) → NodeId`, cave smoothing, dungeon carving, lazy root expansion, terrain-gen perf tracking
- ☐ Cross-platform distribution (`xb7`): macOS notarization (`xb7.2`, credentials-gated), Linux AppImage (`xb7.3`), WASM/WebGPU (`xb7.5`, design-gated), Steam (`xb7.6`, P4 deferred)
- ☐ SVDAG research (`5bb.6`): SSVDAG / sparse-64 / LOD streaming once baseline is stable

---

## Anti-goals

Things we are NOT doing, to be clear about scope:

- **Not a scan-based Powder Game clone.** Explicitly rejected by the user.
- **Not a standard game engine.** We're building the simulation substrate; game mechanics come after.
- **Not physically-accurate continuum physics.** No Navier-Stokes solver, no lattice Boltzmann. Material-type CA with hand-authored rules is the path.
- **Not a voxel editor first.** Editing is a consequence of needing dynamic simulation, not the primary goal.
- **Not using Voxel Quest.** Investigated and rejected.
- **Not web-first.** Native is primary; web is an optional shareable path.
