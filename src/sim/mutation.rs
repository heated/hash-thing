//! World mutation channel.
//!
//! This module enforces the **closed-world invariant**: entities edit the
//! world by pushing typed mutations onto a queue; the world applies them
//! at a well-defined point in the tick loop. There is no direct path for
//! entity code to touch `NodeStore` mid-tick, and no path at all for a
//! CA rule to read or write entity state.
//!
//! ## Why this exists
//!
//! Hashlife memoization keys the step cache on `NodeId`. Any mutation
//! that bypasses the tick boundary can desync the world from the cache,
//! because two structurally identical subtrees must always produce the
//! same successor. The runtime guard on [`World::step_ca`] panics if
//! mutations are still pending when CA evaluation starts — that's the
//! one-line enforcement mechanism.
//!
//! ## Tick loop
//!
//! ```text
//!   world.apply_mutations()     // drain queue → octree edits
//!   world.step_ca(rule)         // CA evolves the world (guard: queue must be empty)
//!   entities.update(&world, &mut queue)  // read new world, push mutations for next tick
//! ```
//!
//! Next tick starts at `apply_mutations` again, so there's no explicit
//! "clear queue" step — `apply_mutations` drains.
//!
//! ## What goes in the queue
//!
//! [`WorldMutation`] is deliberately a bounded, typed set. It's the wire
//! format for lockstep multiplayer (hash-thing-3bj) and the authorization
//! surface for sandboxed gameplay plugins (hash-thing-ore). Adding a new
//! variant is a deliberate, review-gated action — if you need something
//! novel, design it at the enum level, not as a back-door through a
//! `&mut NodeStore`.

use crate::octree::CellState;

/// A single world edit. Arrival-order application is deterministic as
/// long as the producer order is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMutation {
    /// Set a single cell.
    SetCell {
        x: u64,
        y: u64,
        z: u64,
        state: CellState,
    },
    /// Fill an axis-aligned inclusive region with a constant state.
    /// Bounds are inclusive on both sides; `min == max` fills one cell.
    /// Caller is responsible for ensuring `min <= max` on every axis.
    FillRegion {
        x_min: u64,
        y_min: u64,
        z_min: u64,
        x_max: u64,
        y_max: u64,
        z_max: u64,
        state: CellState,
    },
}

/// A bounded FIFO of pending world mutations. Entities push; the world
/// drains in arrival order during `apply_mutations`.
///
/// The queue is deliberately a thin `Vec` — no priority, no coalescing,
/// no deduplication. Arrival order IS the order of application. If a
/// future producer needs different semantics (e.g., ordering by position
/// for commutativity), that's a new abstraction on top of this, not a
/// change to the queue.
#[derive(Default, Debug)]
pub struct MutationQueue {
    pending: Vec<WorldMutation>,
}

impl MutationQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Push a mutation onto the tail of the queue.
    pub fn push(&mut self, m: WorldMutation) {
        self.pending.push(m);
    }

    /// Number of pending mutations.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True if the queue has no pending mutations.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Take all pending mutations, leaving the queue empty. Used by
    /// `World::apply_mutations` to drain before application.
    pub(crate) fn take(&mut self) -> Vec<WorldMutation> {
        std::mem::take(&mut self.pending)
    }

    /// Read-only view of the pending mutations (for inspection / tests).
    #[allow(dead_code)]
    pub fn pending(&self) -> &[WorldMutation] {
        &self.pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain_preserve_order() {
        let mut q = MutationQueue::new();
        q.push(WorldMutation::SetCell {
            x: 1,
            y: 2,
            z: 3,
            state: 1,
        });
        q.push(WorldMutation::SetCell {
            x: 4,
            y: 5,
            z: 6,
            state: 2,
        });
        q.push(WorldMutation::SetCell {
            x: 7,
            y: 8,
            z: 9,
            state: 3,
        });
        assert_eq!(q.len(), 3);
        let drained = q.take();
        assert!(q.is_empty());
        assert_eq!(drained.len(), 3);
        // Arrival order preserved.
        match drained[0] {
            WorldMutation::SetCell { x, state, .. } => {
                assert_eq!((x, state), (1, 1));
            }
            _ => panic!("unexpected variant"),
        }
        match drained[2] {
            WorldMutation::SetCell { x, state, .. } => {
                assert_eq!((x, state), (7, 3));
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn take_empties_the_queue() {
        let mut q = MutationQueue::new();
        q.push(WorldMutation::SetCell {
            x: 0,
            y: 0,
            z: 0,
            state: 1,
        });
        let _ = q.take();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        // A second take yields nothing.
        assert_eq!(q.take().len(), 0);
    }

    #[test]
    fn default_is_empty() {
        let q = MutationQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }
}
