# Design Ledger

Running list of design calls edward has made on hash-thing. Mayor owns this file.

**Rules:** one sentence per entry. Link out for detail. Update on every park/resolve. Don't try to be exhaustive on day one — seed from session memory and grow.

---

## Decided

- **Cell encoding is 16-bit tagged: `(material << 6) | metadata`** — `hash-thing-1v0.1` (closed). Material in high 10 bits, metadata in low 6.
- **Determinism comes from `cell_hash(pos, generation, seed)`, not global PRNG** — `hash-thing-h34.1` (closed). All cell-level randomness derives from a single hash function with explicit inputs.
- **`cell_rand_bool` must be scan-order-independent** — `hash-thing-c6k` (closed, per session memory). CA rules cannot depend on iteration order.
- **Hash-cons store needs explicit lifetime management; append-only is a bug** — `hash-thing-88d`. Decided to ship per-step `NodeStore::compacted(root) → (NodeStore, NodeId)` whole, with the fresh-store recursive clone shape.
- **Cache lifetime == store epoch** (architectural rule that falls out of 88d) — any future memo cache must be epoch-scoped to the store, moved out of the store, or remapped across compaction. Rolls forward as a constraint on `hash-thing-6gf.2`.
- **SVDAG is the primary render path; flat 3D-texture is debug/fallback** — `hash-thing-5bb.1/2/3` (all closed). Iterative GPU traversal in Laine-Karras lineage.
- **NodeStore is hash-consed by structural identity** — implicit since project start. `intern_node` dedups on `(level, children…)`.
- **Mayor seat exists** — singular policy/observer role, no implementation work in `src/`. `CLAUDE.md` "The crew" + "The mayor" sections (`AGENTS.md` is a symlink to `CLAUDE.md`).
- **Gate-tier rubric** — `~/.claude/CLAUDE.md` "Gate Tiers" section. Default `gate-sensitivity: medium`. Bug fixes / refactors / on-disk formats / module layout / naming = ship without asking. User-visible behavior / capability shifts / one-way doors = pull human in.
- **afk mode** — when edward says "afk", crew never waits at gates: park + comment + pull next.
- **Crew /ship-by-default + plan-review tier table** — `hash-thing-3au` (closed). Landed across global `~/.claude/CLAUDE.md` "Gate Tiers" + "Reviews" default-tier-by-priority table + project `CLAUDE.md` "Peer autonomy" + "afk mode" + "The mayor" sections. Agent proposing the task picks the tier; override up allowed.
- **`m0p` metaverse direction — middle path, game-leaning** — `hash-thing-m0p` (closed 2026-04-11 16:25). Foundational principle: **The World Sim is Closed** — entities push world edits, world state never reads entity state, so Hashlife stays load-bearing because rules remain pure-local. 1v0.x epic reframed: `1v0.1`/`1v0.2`/`1v0.4` keep, `1v0.3` reframes to "per-material CaRule registration." Follow-ups: `hash-thing-sos` (1v0.x reframe draft, flint), `hash-thing-ore` (plugin-surface survey, flint), `hash-thing-0et` (multiplayer exploration). Full resolution in `.ship-notes/metaverse-design-decision.md` (flint's groovy-launching-piglet worktree).
- **Rules must stay pure-local — hard-enforced invariant** — falls out of m0p middle-path + World-Sim-Closed. `CaRule::step_cell` cannot read `world.generation`, `player.pos`, or global RNG; one almost-pure rule silently kills Hashlife memoization downstream. Enforcement mechanism (lint / trait bound / runtime check) is a follow-up design call, not yet filed.
- **Stepper is 1-step Hashlife (spatial dedup), not time-doubling** — `hash-thing-6gf.1` (closed as design, 2026-04-11). Hashlife's classical `2^(k-2)` time-doubling is async (different regions at different generations during computation — see Wikipedia), which rules it out for a real-time voxel game loop. Golly ships QuickLife alongside HashLife for exactly this reason. Decision: 6gf.1 builds `step(node) -> center advanced exactly 1 gen`, memoized by `(NodeId, rule_id) → NodeId`. Recursion is one phase (27 subnodes → inner 2×2×2 of the 3×3×3 result grid), not the two-phase 27+8 Gosper combine. Speedup is polynomial (spatial dedup only), not exponential, but real-time-compatible. Time-doubling variant filed as `hash-thing-8o5` for later fast-forward / offline use. Mayor made the call per edward's delegation ("research + consider with reviews + decide").

---

## Pending (parked design gates)

- **`819` OOB-bounds split** — bug-fix half is shippable; contract half (`ensure_contains` stub + SPEC.md commitments around lazy expansion + PROVISIONAL u64 signature) is design. Edward's options: (a) approve unified, (b) split, (c) trim. Mayor recommends (b).
- **`1v0.2` material registry shape** — how materials are identified and referenced; the schema everything CA, physics, and rendering will key off. Blocks `1v0.3-1v0.5`, `5bb.4`.
- **`5bb.5` HashDAG incremental upload protocol** — CBE 2020 protocol vs custom protocol coupled to step cadence + 88d compaction. GPU memory model + sim-renderer handshake. Pre-plan dialogue, 4 open questions.
- **`3fq.1` terrain gen API + PRNG family** — `gen(node_region) -> NodeId` shape and the terrain seed scheme (cell_hash family vs separate). One-way door on terrain APIs.
- **`1v0.6` particles substrate** — second simulation substrate outside the octree state plus a coupling protocol. Capability shift.
- **`xb7.5` WASM/WebGPU build target** — is web a real target or shareable-demo extra? Capability + platform call.
- **`l7d` trident review collapse (9→6 or 9→4)** — review process policy. Wait for 2-3 more trident runs of cost data.
- **`0uw` stuck-task detection rule** — proposed rule for global CLAUDE.md (peek output files, kill stuck agents on 429 loops). Sibling of l7d.

---

## Open questions the crew is circling

- **Compaction cadence beyond per-step** — once `6gf.2` memoizes, per-step compaction throws away the memo cache. The right cadence (per N steps, per GPU upload, on memory pressure) is unframed.
- **`Stepper` trait existence** — separate `Stepper` struct vs free function vs `NodeStore` method recurs across `6gf.*` and `88d` planning. No commitment yet.
- **Coordinate type contract** — `WorldCoord(i64)` newtype, `RealizedRegion`, `LocalCoord` deferred from `819` trident review. Will become a design call when `e9h` lazy root expansion lands.
- **Rule identity in memo keys** — does `CaRule::rule_id()` enter the key? `6gf.1` flagged this as evolutionary insight; not yet committed.
- **Terrain seed family vs cell seed family** — currently only `cell_hash` is committed. Whether terrain shares the family or gets its own is unframed.
- **Rule-purity enforcement mechanism** — m0p middle-path promoted "rules stay pure-local" to hard invariant, but enforcement shape (lint pass, `CaRule: Pure` trait bound, runtime check on `world.generation` / `player.pos` access) is open. Pick before the plugin surface lands (`hash-thing-ore`).
- **Entity → world edit protocol** — "World Sim is Closed" principle implies entities push edits into the world via a bounded API; the shape of that API (immediate, queued, per-tick batched, conflict resolution) is unframed. Needed before `1v0.6` particles or any entity work.
