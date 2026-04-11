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
- **Mayor seat exists** — singular policy/observer role, no implementation work in `src/`. AGENTS.md "The crew" + "The mayor" sections.
- **Gate-tier rubric** — `~/.claude/CLAUDE.md` "Gate Tiers" section. Default `gate-sensitivity: medium`. Bug fixes / refactors / on-disk formats / module layout / naming = ship without asking. User-visible behavior / capability shifts / one-way doors = pull human in.
- **afk mode** — when edward says "afk", crew never waits at gates: park + comment + pull next.
- **Crew /ship-by-default + plan-review tier table** — `hash-thing-3au` (closed). Landed across global `~/.claude/CLAUDE.md` "Gate Tiers" + "Reviews" default-tier-by-priority table + AGENTS.md "Peer autonomy" + "afk mode" + "The mayor" sections. Agent proposing the task picks the tier; override up allowed.

---

## Pending (parked design gates)

- **`m0p` substrate vs middle vs game-first** — six-month direction call, **active live dialogue with flint** (not dormant). Three paths framed in `.ship-notes/metaverse-design-decision.md` (flint's groovy-launching-piglet worktree): substrate-first (CA library, Hashlife sacred, u8 `CellState`), middle ("substrate-enabled game engine" — blocks-as-CaRules + separate entity store, Hashlife caches stable regions), game-first (Minecraft-in-Rust, Hashlife collapses to background optimization). Flint's lean: **middle**. Repo tension flint surfaced: *infrastructure* (Hashlife, hash-cons, determinism, cell_hash) is substrate-coherent but *active 1v0.x roadmap* (material registry, reaction table, Margolus movement) has been soft-committing game-ward — the drift was natural, each step reasonable, aggregate locks in. Lock in effect: no fanout on m0p, no pulling on 1v0.x, no touching downstream of m0p until edward picks. Everything else routes through this.
- **`6gf.1` Hashlife stepper architecture** — stepper module placement, memo key shape, rule-identity-into-key, cache-lifetime epoch boundary inherited from 88d. Plan ready, 5 reviewers approve. Edward's call on the algorithmic spine.
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
- **Rules-must-stay-pure invariant (conditional on m0p)** — flint's m0p decision doc surfaces this: for Hashlife memoization to work, `CaRule::step_cell` must be pure-local (no `world.generation`, no `player.pos`, no global RNG). Under substrate it's implicit contract; under middle it becomes a **hard-enforced invariant** (lint? trait bound? runtime check?) because one "almost-pure" rule silently kills memoization downstream; under game-first it's moot. Decision is downstream of m0p, not separate.
- **1v0.x has been soft-committing game-first** — flint's diagnostic (in decision doc): material registry + hand-authored reaction table + Margolus movement phase are game-engine features, not substrate features. Nothing about this is wrong in isolation; the aggregate is what locks in. This observation is the lens through which m0p needs to be resolved — "do we stay on this drift or course-correct?"
