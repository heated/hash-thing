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

---

## Pending (parked design gates)

- **`m0p` substrate-first vs game-first** — six-month direction call. Repo today is substrate-first; vision file (ONE_POSSIBLE_METAVERSE_VISION.md, truncated) proposes a pivot to game-first. One-way doors: CellState shape, modding ABI (Rust trait vs Lua), networking protocol existence, what artifact ships (engine vs world). Mayor's soft rec: substrate-first. Edward's call. Everything else routes through this.
- **`6gf.1` Hashlife stepper architecture** — stepper module placement, memo key shape, rule-identity-into-key, cache-lifetime epoch boundary inherited from 88d. Plan ready, 5 reviewers approve. Edward's call on the algorithmic spine.
- **`819` OOB-bounds split** — bug-fix half is shippable; contract half (`ensure_contains` stub + SPEC.md commitments around lazy expansion + PROVISIONAL u64 signature) is design. Edward's options: (a) approve unified, (b) split, (c) trim. Mayor recommends (b).
- **`1v0.2` material registry shape** — how materials are identified and referenced; the schema everything CA, physics, and rendering will key off. Blocks `1v0.3-1v0.5`, `5bb.4`.
- **`5bb.5` HashDAG incremental upload protocol** — CBE 2020 protocol vs custom protocol coupled to step cadence + 88d compaction. GPU memory model + sim-renderer handshake. Pre-plan dialogue, 4 open questions.
- **`3fq.1` terrain gen API + PRNG family** — `gen(node_region) -> NodeId` shape and the terrain seed scheme (cell_hash family vs separate). One-way door on terrain APIs.
- **`1v0.6` particles substrate** — second simulation substrate outside the octree state plus a coupling protocol. Capability shift.
- **`xb7.5` WASM/WebGPU build target** — is web a real target or shareable-demo extra? Capability + platform call.
- **`3au` crew /ship-by-default + plan-review policy** — when /ship gates, who picks plan-review tier, compliance audit. Policy change, design-gated per AGENTS.md.
- **`l7d` trident review collapse (9→6 or 9→4)** — review process policy. Wait for 2-3 more trident runs of cost data.
- **`0uw` stuck-task detection rule** — proposed rule for global CLAUDE.md (peek output files, kill stuck agents on 429 loops). Sibling of l7d.

---

## Open questions the crew is circling

- **Compaction cadence beyond per-step** — once `6gf.2` memoizes, per-step compaction throws away the memo cache. The right cadence (per N steps, per GPU upload, on memory pressure) is unframed.
- **`Stepper` trait existence** — separate `Stepper` struct vs free function vs `NodeStore` method recurs across `6gf.*` and `88d` planning. No commitment yet.
- **Coordinate type contract** — `WorldCoord(i64)` newtype, `RealizedRegion`, `LocalCoord` deferred from `819` trident review. Will become a design call when `e9h` lazy root expansion lands.
- **Rule identity in memo keys** — does `CaRule::rule_id()` enter the key? `6gf.1` flagged this as evolutionary insight; not yet committed.
- **Terrain seed family vs cell seed family** — currently only `cell_hash` is committed. Whether terrain shares the family or gets its own is unframed.
