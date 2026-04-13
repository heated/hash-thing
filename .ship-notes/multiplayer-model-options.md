# Multiplayer Model Options

Reconstructed on 2026-04-13 from closed bead `hash-thing-0et`, epic text in `hash-thing-3bj` / `hash-thing-ij0`, and the surrounding design ledger after the original note went missing.

## Recommendation

Ship **deterministic lockstep** as multiplayer v1.

Why this wins for hash-thing:

- The mutation channel (`1v0.9`) is already the right wire shape: peers exchange intended edits, not full world state.
- The closed-world invariant already forces the discipline lockstep needs: world state evolves from local rules plus queued mutations, not hidden external state.
- Hash-consed `NodeId` roots make desync detection cheap: compare roots every N ticks and treat mismatch as a first-class fault.
- The determinism audit (`h34.2`, follow-up `i6y`) is useful even if multiplayer never ships, so the prep work is not speculative waste.
- The target ceiling is modest cooperative play, not public-server scale.

## Options Survey

### 1. Deterministic lockstep

**Model:** every peer simulates the full world, exchanges stamped inputs / mutations, and advances the step loop only when all required inputs for that frame are present.

**Pros**

- Lowest architectural distortion: matches the current simulation model.
- Bandwidth cost tracks player actions, not world size.
- Desync detection is trivial via root `NodeId`.
- Initial sync can stay small if subtree graph transfer is solved.

**Cons**

- Slowest-peer latency caps practical player count.
- Requires strict determinism discipline across the entire sim path.
- Join / catch-up flow still needs subtree transfer or an equivalent world snapshot transport.

**Best fit**

- 4-16 player cooperative sandbox sessions.

### 2. Authoritative server

**Model:** one process owns the canonical world; clients send intents and receive state or deltas.

**Pros**

- Easier anti-cheat and easier handling of mildly nondeterministic clients.
- Late join is conceptually simpler because the server is the source of truth.
- No slowest-peer hard stall.

**Cons**

- Pushes the project toward server infrastructure before the game has proven it needs it.
- Requires a replication protocol for a large octree world instead of reusing the closed-world + mutation discipline.
- Gives up the unusual advantage hash-thing already has for lockstep and rollback.

**Best fit**

- Public servers, moderation, persistence, or adversarial environments.

### 3. Rollback networking

**Model:** predict remote inputs locally, simulate ahead, rewind and re-simulate on mismatch.

**Pros**

- hash-thing is unusually good at the expensive rollback steps:
  - world snapshots are cheap because a root `NodeId` is an O(1) snapshot handle
  - re-simulation is cheaper than usual because shared subtrees hit memoized step results
- Excellent feel for small-player-count competitive modes.

**Cons**

- Higher implementation complexity than lockstep.
- Prediction quality degrades as player count rises.
- Needs a real gameplay target that cares about tight competitive feel.

**Best fit**

- 2-4 player competitive or duel modes after lockstep v1 proves out.

## Concrete v1 Implementation Plan

1. Determinism audit of sim-adjacent code paths and follow-up fixes.
2. Add a `DeterminismGuard`-style debug wrapper around the CA step / mutation application boundary.
3. Serialize `WorldMutation` for network transport.
4. Stamp and buffer local/remote inputs by simulation tick.
5. Advance the step loop in lockstep once all required inputs for a frame are present.
6. Compare root `NodeId` values every N ticks for desync detection.
7. Add local-latency hiding for block placement / break actions.
8. Add basic lobby / join flow once subtree sync format is chosen.

## Rust / Engine Footguns To Keep Out Of Lockstep

- `std::collections::HashMap` / `HashSet` iteration order is process-randomized.
- `rand::thread_rng()` or OS-seeded randomness in the sim path is poison.
- Wall clock (`Instant`, system time) must never influence the simulation.
- Floats must not leak into the mutation channel or CA rule decisions.
- Parallel float reductions are order-dependent unless the combiner is made deterministic.
- Pointer addresses or allocator identity must never become part of simulation state.

These checks are what `hash-thing-i6y` exists to formalize.

## Why Not Commit Harder Yet

Multiplayer is still future-gated on single-player being worth sharing. This note is a direction record, not an instruction to start implementation now:

- `hash-thing-3bj` is the v1 lockstep epic once the game is fun enough.
- `hash-thing-ij0` is the later rollback branch for competitive modes.
- Authoritative-server work stays off the critical path unless the product shifts toward public or adversarial play.
