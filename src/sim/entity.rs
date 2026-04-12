//! Entity system (hash-thing-1v0.6).
//!
//! Out-of-grid things with continuous positions. Entities read the octree
//! but never mutate it directly — they push `WorldMutation`s onto the
//! mutation queue, applied at tick boundaries by `World::apply_mutations`.
//!
//! ## Update order
//!
//! ```text
//!   world.apply_mutations()       // drain entity edits from last tick
//!   world.step()                  // CA evolves the world
//!   entities.update(&world)       // read new world, update positions, push new mutations
//! ```

use super::mutation::{MutationQueue, WorldMutation};
use super::world::{World, WorldCoord};
use crate::octree::{Cell, CellState};

/// Opaque entity identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

/// 3D position/velocity as `[f64; 3]`. No external math lib needed yet.
pub type Vec3 = [f64; 3];

/// What kind of entity this is. Particles are the first and simplest.
#[derive(Clone, Debug)]
pub enum EntityKind {
    /// Short-lived visual effect: sparks, smoke, debris.
    Particle(ParticleState),
}

/// Per-particle state.
#[derive(Clone, Debug)]
pub struct ParticleState {
    /// Material to render as (for color lookup).
    pub material: u8,
    /// Remaining lifetime in ticks. 0 = despawn this tick.
    pub ttl: u32,
    /// Optional: cell state to write to the world on despawn (e.g., ash).
    pub on_despawn: Option<CellState>,
}

/// A single entity in the world.
#[derive(Clone, Debug)]
pub struct Entity {
    pub id: EntityId,
    pub pos: Vec3,
    pub vel: Vec3,
    pub kind: EntityKind,
}

/// Storage for all entities. Slab-style: entities indexed by id, with a
/// free list for reuse. For the initial particle-only implementation,
/// a simple Vec + generation counter is sufficient.
pub struct EntityStore {
    entities: Vec<Option<Entity>>,
    next_id: u64,
}

impl Default for EntityStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an entity, returning its id.
    pub fn add(&mut self, pos: Vec3, vel: Vec3, kind: EntityKind) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        let entity = Entity { id, pos, vel, kind };
        // Find a free slot or push.
        if let Some(slot) = self.entities.iter_mut().find(|s| s.is_none()) {
            *slot = Some(entity);
        } else {
            self.entities.push(Some(entity));
        }
        id
    }

    /// Remove an entity by id. Returns true if it existed.
    pub fn remove(&mut self, id: EntityId) -> bool {
        for slot in &mut self.entities {
            if let Some(e) = slot {
                if e.id == id {
                    *slot = None;
                    return true;
                }
            }
        }
        false
    }

    /// Iterate over all live entities.
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter().filter_map(|s| s.as_ref())
    }

    /// Number of live entities.
    pub fn len(&self) -> usize {
        self.entities.iter().filter(|s| s.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Update all entities: apply velocity, tick down TTLs, despawn dead
    /// particles. Reads cell state from `world` for collision/spawning.
    /// Pushes mutations onto `queue` for world edits.
    pub fn update(&mut self, world: &World, queue: &mut MutationQueue) {
        let mut to_remove = Vec::new();

        for slot in &mut self.entities {
            let Some(entity) = slot else { continue };

            // Apply velocity.
            entity.pos[0] += entity.vel[0];
            entity.pos[1] += entity.vel[1];
            entity.pos[2] += entity.vel[2];

            match &mut entity.kind {
                EntityKind::Particle(state) => {
                    if state.ttl == 0 {
                        // Despawn: optionally write a cell.
                        if let Some(cell_state) = state.on_despawn {
                            let x = entity.pos[0].round() as i64;
                            let y = entity.pos[1].round() as i64;
                            let z = entity.pos[2].round() as i64;
                            queue.push(WorldMutation::SetCell {
                                x: WorldCoord(x),
                                y: WorldCoord(y),
                                z: WorldCoord(z),
                                state: cell_state,
                            });
                        }
                        to_remove.push(entity.id);
                    } else {
                        state.ttl -= 1;

                        // Simple collision: if inside a solid cell, stop.
                        let gx = entity.pos[0].round() as i64;
                        let gy = entity.pos[1].round() as i64;
                        let gz = entity.pos[2].round() as i64;
                        let cell = Cell::from_raw(world.get(
                            WorldCoord(gx),
                            WorldCoord(gy),
                            WorldCoord(gz),
                        ));
                        if !cell.is_empty() {
                            // Bounce back: undo velocity, zero it.
                            entity.pos[0] -= entity.vel[0];
                            entity.pos[1] -= entity.vel[1];
                            entity.pos[2] -= entity.vel[2];
                            entity.vel = [0.0; 3];
                        }

                        // Gravity.
                        entity.vel[1] -= 0.05;
                    }
                }
            }
        }

        for id in to_remove {
            self.remove(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_iterate() {
        let mut store = EntityStore::new();
        let id1 = store.add(
            [1.0, 2.0, 3.0],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 10,
                on_despawn: None,
            }),
        );
        let id2 = store.add(
            [4.0, 5.0, 6.0],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 2,
                ttl: 5,
                on_despawn: None,
            }),
        );
        assert_eq!(store.len(), 2);
        let ids: Vec<EntityId> = store.iter().map(|e| e.id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn remove_entity() {
        let mut store = EntityStore::new();
        let id = store.add(
            [0.0; 3],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 10,
                on_despawn: None,
            }),
        );
        assert!(store.remove(id));
        assert_eq!(store.len(), 0);
        assert!(!store.remove(id)); // double remove returns false
    }

    #[test]
    fn slot_reuse() {
        let mut store = EntityStore::new();
        let id1 = store.add(
            [0.0; 3],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 10,
                on_despawn: None,
            }),
        );
        store.remove(id1);
        let _id2 = store.add(
            [0.0; 3],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 10,
                on_despawn: None,
            }),
        );
        // Should reuse the first slot, not grow the vec.
        assert_eq!(store.entities.len(), 1);
    }

    #[test]
    fn update_applies_velocity() {
        let mut store = EntityStore::new();
        store.add(
            [5.0, 5.0, 5.0],
            [1.0, 0.0, -1.0],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 10,
                on_despawn: None,
            }),
        );
        let world = World::new(3);
        let mut queue = MutationQueue::new();
        store.update(&world, &mut queue);
        let e = store.iter().next().unwrap();
        assert!((e.pos[0] - 6.0).abs() < 1e-9);
        assert!((e.pos[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn particle_despawns_at_ttl_zero() {
        let mut store = EntityStore::new();
        store.add(
            [3.0, 3.0, 3.0],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 0, // despawn immediately
                on_despawn: None,
            }),
        );
        let world = World::new(3);
        let mut queue = MutationQueue::new();
        store.update(&world, &mut queue);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn particle_despawn_pushes_mutation() {
        let mut store = EntityStore::new();
        let ash_state: CellState = 42;
        store.add(
            [3.0, 3.0, 3.0],
            [0.0; 3],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 0,
                on_despawn: Some(ash_state),
            }),
        );
        let world = World::new(3);
        let mut queue = MutationQueue::new();
        store.update(&world, &mut queue);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn default_is_empty() {
        let store = EntityStore::default();
        assert!(store.is_empty());
    }
}
