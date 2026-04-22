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
use crate::scale::CELLS_PER_METER;
use crate::terrain::materials::{
    FIRE_MATERIAL_ID, LAVA, LAVA_MATERIAL_ID, WATER, WATER_MATERIAL_ID,
};

/// Opaque entity identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

/// 3D position/velocity as `[f64; 3]`. No external math lib needed yet.
pub type Vec3 = [f64; 3];

/// What kind of entity this is.
#[derive(Clone, Debug)]
pub enum EntityKind {
    /// Short-lived visual effect: sparks, smoke, debris.
    Particle(ParticleState),
    /// The player — continuous position, input-driven movement.
    Player(PlayerState),
    /// Environmental emitters that periodically spawn particles.
    Emitter(EmitterState),
    /// Simple autonomous creature rendered through the particle path.
    Critter(CritterState),
}

/// Per-player state. Movement and camera are driven by input, not physics.
#[derive(Clone, Debug)]
pub struct PlayerState {
    /// Horizontal look angle (radians).
    pub yaw: f64,
    /// Vertical look angle (radians), clamped to ±π/2.
    pub pitch: f64,
    /// Material to place when right-clicking.
    pub held_material: u16,
}

/// Per-particle state.
#[derive(Clone, Debug)]
pub struct ParticleState {
    /// Material to render as (for color lookup).
    pub material: u16,
    /// Remaining lifetime in ticks. 0 = despawn this tick.
    pub ttl: u32,
    /// Optional: cell state to write to the world on despawn (e.g., ash).
    pub on_despawn: Option<CellState>,
}

/// Environmental emitter profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitterKind {
    Geyser,
    Volcano,
    Whirlpool,
}

/// Periodic environmental source.
#[derive(Clone, Debug)]
pub struct EmitterState {
    pub kind: EmitterKind,
    pub period: u32,
    pub burst_count: u8,
    pub particle_ttl: u32,
    pub speed: f64,
    pub spread: f64,
    pub age: u32,
}

impl EmitterState {
    pub fn geyser() -> Self {
        Self {
            kind: EmitterKind::Geyser,
            period: 5,
            burst_count: 2,
            particle_ttl: 18,
            speed: 0.85 * CELLS_PER_METER,
            spread: 0.10 * CELLS_PER_METER,
            age: 0,
        }
    }

    pub fn volcano() -> Self {
        Self {
            kind: EmitterKind::Volcano,
            period: 7,
            burst_count: 3,
            particle_ttl: 24,
            speed: 0.75 * CELLS_PER_METER,
            spread: 0.16 * CELLS_PER_METER,
            age: 0,
        }
    }

    pub fn whirlpool() -> Self {
        Self {
            kind: EmitterKind::Whirlpool,
            period: 4,
            burst_count: 3,
            particle_ttl: 14,
            speed: 0.22 * CELLS_PER_METER,
            spread: 0.08 * CELLS_PER_METER,
            age: 0,
        }
    }
}

/// Autonomous hopper used for simple creature motion.
#[derive(Clone, Debug)]
pub struct CritterState {
    pub material: u16,
    pub age: u32,
    pub hop_period: u32,
    pub hop_speed: f64,
    pub hop_height: f64,
}

impl CritterState {
    pub fn new(material: u16) -> Self {
        Self {
            material,
            age: 0,
            hop_period: 18,
            hop_speed: 0.16 * CELLS_PER_METER,
            hop_height: 0.26 * CELLS_PER_METER,
        }
    }
}

/// A single entity in the world.
#[derive(Clone, Debug)]
pub struct Entity {
    pub id: EntityId,
    pub pos: Vec3,
    pub vel: Vec3,
    pub kind: EntityKind,
}

impl Entity {
    pub fn render_material(&self) -> Option<u16> {
        match &self.kind {
            EntityKind::Particle(state) => Some(state.material),
            EntityKind::Critter(state) => Some(state.material),
            EntityKind::Player(_) | EntityKind::Emitter(_) => None,
        }
    }
}

/// Storage for all entities. Slab-style: entities indexed by id, with a
/// free list for reuse. For the initial particle-only implementation,
/// a simple Vec + generation counter is sufficient.
#[derive(Clone)]
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

    /// Get a mutable reference to an entity by id.
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities
            .iter_mut()
            .filter_map(|s| s.as_mut())
            .find(|e| e.id == id)
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
        let mut to_add = Vec::new();

        for slot in &mut self.entities {
            let Some(entity) = slot else { continue };
            let Entity { id, pos, vel, kind } = entity;

            match kind {
                EntityKind::Player(_) => {
                    // Player movement is input-driven (handled in main.rs).
                    // No velocity, gravity, or collision here.
                    continue;
                }
                EntityKind::Particle(state) => {
                    update_particle(*id, pos, vel, state, world, queue, &mut to_remove);
                }
                EntityKind::Emitter(state) => {
                    update_emitter(*id, *pos, state, &mut to_add);
                }
                EntityKind::Critter(state) => {
                    update_critter(*id, pos, vel, state, world);
                }
            }
        }

        for (pos, vel, kind) in to_add {
            self.add(pos, vel, kind);
        }
        for id in to_remove {
            self.remove(id);
        }
    }
}

fn update_particle(
    id: EntityId,
    pos: &mut Vec3,
    vel: &mut Vec3,
    state: &mut ParticleState,
    world: &World,
    queue: &mut MutationQueue,
    to_remove: &mut Vec<EntityId>,
) {
    pos[0] += vel[0];
    pos[1] += vel[1];
    pos[2] += vel[2];

    if state.ttl == 0 {
        if let Some(cell_state) = state.on_despawn {
            let x = pos[0].round() as i64;
            let y = pos[1].round() as i64;
            let z = pos[2].round() as i64;
            queue.push(WorldMutation::SetCell {
                x: WorldCoord(x),
                y: WorldCoord(y),
                z: WorldCoord(z),
                state: cell_state,
            });
        }
        to_remove.push(id);
        return;
    }
    state.ttl -= 1;

    if !cell_at(world, *pos).is_empty() {
        pos[0] -= vel[0];
        pos[1] -= vel[1];
        pos[2] -= vel[2];
        *vel = [0.0; 3];
    }

    vel[1] -= 0.05 * CELLS_PER_METER;
}

fn update_emitter(
    id: EntityId,
    pos: Vec3,
    state: &mut EmitterState,
    to_add: &mut Vec<(Vec3, Vec3, EntityKind)>,
) {
    if state.age.is_multiple_of(state.period) {
        for i in 0..u32::from(state.burst_count) {
            let (spawn_pos, vel, particle) = emitter_particle(id, pos, state, i);
            to_add.push((spawn_pos, vel, EntityKind::Particle(particle)));
        }
    }
    state.age = state.age.wrapping_add(1);
}

fn emitter_particle(
    id: EntityId,
    pos: Vec3,
    state: &EmitterState,
    burst_idx: u32,
) -> (Vec3, Vec3, ParticleState) {
    let seed = id.0 as f64 * 0.173 + state.age as f64 * 0.947 + burst_idx as f64 * 1.618;
    let jitter_x = noise(seed);
    let jitter_z = noise(seed + 2.3);
    let angle = seed * 0.7;
    match state.kind {
        EmitterKind::Geyser => (
            pos,
            [
                jitter_x * state.spread,
                state.speed + noise(seed + 4.1) * 0.12 * CELLS_PER_METER,
                jitter_z * state.spread,
            ],
            ParticleState {
                material: WATER_MATERIAL_ID,
                ttl: state.particle_ttl,
                on_despawn: Some(WATER),
            },
        ),
        EmitterKind::Volcano => (
            pos,
            [
                jitter_x * state.spread,
                state.speed + noise(seed + 5.7) * 0.18 * CELLS_PER_METER,
                jitter_z * state.spread,
            ],
            ParticleState {
                material: if burst_idx.is_multiple_of(3) {
                    FIRE_MATERIAL_ID
                } else {
                    LAVA_MATERIAL_ID
                },
                ttl: state.particle_ttl,
                on_despawn: Some(LAVA),
            },
        ),
        EmitterKind::Whirlpool => {
            let radius = (0.75 + noise(seed + 8.4) * 0.2) * CELLS_PER_METER;
            let orbit_x = angle.cos() * radius;
            let orbit_z = angle.sin() * radius;
            (
                [pos[0] + orbit_x, pos[1], pos[2] + orbit_z],
                [
                    -orbit_z * state.speed + jitter_x * state.spread,
                    -0.03 * CELLS_PER_METER,
                    orbit_x * state.speed + jitter_z * state.spread,
                ],
                ParticleState {
                    material: WATER_MATERIAL_ID,
                    ttl: state.particle_ttl,
                    on_despawn: None,
                },
            )
        }
    }
}

fn update_critter(
    id: EntityId,
    pos: &mut Vec3,
    vel: &mut Vec3,
    state: &mut CritterState,
    world: &World,
) {
    let grounded = !cell_at(world, [pos[0], pos[1] - 0.75 * CELLS_PER_METER, pos[2]]).is_empty();

    if grounded && vel[1] <= 0.0 && state.age.is_multiple_of(state.hop_period) {
        let heading = critter_heading(id, state.age);
        vel[0] = heading.cos() * state.hop_speed;
        vel[2] = heading.sin() * state.hop_speed;
        vel[1] = state.hop_height;
    }

    let next = [pos[0] + vel[0], pos[1] + vel[1], pos[2] + vel[2]];
    if !cell_at(world, next).is_empty() {
        vel[0] *= -0.45;
        vel[2] *= -0.45;
        vel[1] = vel[1].max(0.12 * CELLS_PER_METER);
    } else {
        *pos = next;
    }

    vel[1] -= 0.04 * CELLS_PER_METER;
    vel[0] *= 0.94;
    vel[2] *= 0.94;
    state.age = state.age.wrapping_add(1);
}

fn cell_at(world: &World, pos: Vec3) -> Cell {
    let coords = [
        pos[0].round() as i64,
        pos[1].round() as i64,
        pos[2].round() as i64,
    ];
    Cell::from_raw(world.get(
        WorldCoord(coords[0]),
        WorldCoord(coords[1]),
        WorldCoord(coords[2]),
    ))
}

fn critter_heading(id: EntityId, age: u32) -> f64 {
    let phase = noise(id.0 as f64 * 0.31 + age as f64 * 0.17);
    phase * std::f64::consts::TAU
}

fn noise(seed: f64) -> f64 {
    (seed.sin() * 13.37 + (seed * 0.7).cos() * 3.11).sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::materials::{STONE, VINE_MATERIAL_ID};

    fn floor_world(level: u32) -> World {
        let mut world = World::new(level);
        let side = world.side() as i64;
        let thickness = CELLS_PER_METER as i64;
        for x in 0..side {
            for y in 0..thickness {
                for z in 0..side {
                    world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), STONE);
                }
            }
        }
        world
    }

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
    fn particle_despawn_outside_realized_region_is_dropped_on_apply() {
        let mut store = EntityStore::new();
        store.add(
            [7.6, 3.0, 3.0],
            [0.8, 0.0, 0.0],
            EntityKind::Particle(ParticleState {
                material: 1,
                ttl: 0,
                on_despawn: Some(WATER),
            }),
        );

        let mut world = World::new(3);
        let mut queue = MutationQueue::new();
        store.update(&world, &mut queue);
        assert_eq!(queue.len(), 1);

        world.queue = queue;
        world.apply_mutations();

        assert_eq!(world.get(WorldCoord(7), WorldCoord(3), WorldCoord(3)), 0);
        assert_eq!(world.get(WorldCoord(0), WorldCoord(3), WorldCoord(3)), 0);
    }

    #[test]
    fn default_is_empty() {
        let store = EntityStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn emitter_spawns_particles_on_schedule() {
        let mut store = EntityStore::new();
        store.add(
            [8.0, 8.0, 8.0],
            [0.0; 3],
            EntityKind::Emitter(EmitterState::geyser()),
        );
        let world = World::new(4);
        let mut queue = MutationQueue::new();
        store.update(&world, &mut queue);
        assert_eq!(store.len(), 3);
        let particles = store
            .iter()
            .filter(|entity| matches!(entity.kind, EntityKind::Particle(_)))
            .count();
        assert_eq!(particles, 2);
    }

    #[test]
    fn emitter_updates_are_deterministic() {
        let mut a = EntityStore::new();
        let mut b = EntityStore::new();
        a.add(
            [8.0, 8.0, 8.0],
            [0.0; 3],
            EntityKind::Emitter(EmitterState::volcano()),
        );
        b.add(
            [8.0, 8.0, 8.0],
            [0.0; 3],
            EntityKind::Emitter(EmitterState::volcano()),
        );
        let world = World::new(4);
        let mut queue_a = MutationQueue::new();
        let mut queue_b = MutationQueue::new();
        for _ in 0..6 {
            a.update(&world, &mut queue_a);
            b.update(&world, &mut queue_b);
        }
        let snapshot_a: Vec<(Vec3, Vec3, Option<u16>)> = a
            .iter()
            .map(|entity| (entity.pos, entity.vel, entity.render_material()))
            .collect();
        let snapshot_b: Vec<(Vec3, Vec3, Option<u16>)> = b
            .iter()
            .map(|entity| (entity.pos, entity.vel, entity.render_material()))
            .collect();
        assert_eq!(snapshot_a.len(), snapshot_b.len());
        for (left, right) in snapshot_a.iter().zip(snapshot_b.iter()) {
            assert_eq!(left.2, right.2);
            for axis in 0..3 {
                assert!((left.0[axis] - right.0[axis]).abs() < 1e-9);
                assert!((left.1[axis] - right.1[axis]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn critter_hops_from_ground() {
        let mut store = EntityStore::new();
        let s = CELLS_PER_METER;
        let start_y = s;
        store.add(
            [2.0 * s, start_y, 2.0 * s],
            [0.0; 3],
            EntityKind::Critter(CritterState::new(VINE_MATERIAL_ID)),
        );
        let world = floor_world(4);
        let mut queue = MutationQueue::new();
        store.update(&world, &mut queue);
        let critter = store.iter().next().unwrap();
        assert!(matches!(critter.kind, EntityKind::Critter(_)));
        assert!(critter.pos[1] > start_y);
    }
}
