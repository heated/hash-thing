//! World mutation channel.
//!
//! Enforces the **closed-world invariant**: entities edit the world by
//! pushing typed mutations onto a queue; the world applies them at a
//! well-defined point in the tick loop. No direct path for entity code
//! to touch `NodeStore` mid-tick, no path for a CA rule to read entity state.
//!
//! ## Tick loop
//!
//! ```text
//!   world.apply_mutations()     // drain queue → octree edits
//!   world.step()                // CA evolves the world
//!   // entities.update(&world, &mut world.queue)  // lands in 1v0.6
//! ```
//!
//! [`World::step_ca`] debug-asserts the queue is empty on entry.

use super::world::WorldCoord;
use crate::octree::CellState;

/// A single world edit. Arrival-order application is deterministic as
/// long as the producer order is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMutation {
    /// Set a single cell.
    SetCell {
        x: WorldCoord,
        y: WorldCoord,
        z: WorldCoord,
        state: CellState,
    },
    /// Fill an axis-aligned inclusive region with a constant state.
    FillRegion {
        min: [WorldCoord; 3],
        max: [WorldCoord; 3],
        state: CellState,
    },
}

/// A bounded FIFO of pending world mutations. Entities push; the world
/// drains in arrival order during `apply_mutations`.
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

    pub fn push(&mut self, m: WorldMutation) {
        self.pending.push(m);
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Drain all pending mutations. Used by `World::apply_mutations`.
    pub(crate) fn take(&mut self) -> Vec<WorldMutation> {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wc(v: i64) -> WorldCoord {
        WorldCoord(v)
    }

    #[test]
    fn push_and_drain_preserve_order() {
        let mut q = MutationQueue::new();
        q.push(WorldMutation::SetCell {
            x: wc(1),
            y: wc(2),
            z: wc(3),
            state: 1,
        });
        q.push(WorldMutation::SetCell {
            x: wc(4),
            y: wc(5),
            z: wc(6),
            state: 2,
        });
        assert_eq!(q.len(), 2);
        let drained = q.take();
        assert!(q.is_empty());
        assert_eq!(drained.len(), 2);
        match drained[0] {
            WorldMutation::SetCell { x, state, .. } => {
                assert_eq!((x.0, state), (1, 1));
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn take_empties_the_queue() {
        let mut q = MutationQueue::new();
        q.push(WorldMutation::SetCell {
            x: wc(0),
            y: wc(0),
            z: wc(0),
            state: 1,
        });
        let _ = q.take();
        assert!(q.is_empty());
        assert_eq!(q.take().len(), 0);
    }

    #[test]
    fn default_is_empty() {
        let q = MutationQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }
}
