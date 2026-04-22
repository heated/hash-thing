//! Player controller: collision, raycast, movement.
//!
//! Pure functions operating on a voxel grid + position data, extracted from
//! main.rs for testability (hash-thing-d6m).
//!
//! Collision helpers are generic over the [`VoxelGrid`] trait so the main
//! thread can feed them either the live [`World`] or a cheap read-only
//! [`CollisionSnapshot`] cloned at sim-step boundaries. The snapshot variant
//! keeps player physics running every frame while a background sim step owns
//! the mutable `World` (hash-thing-0s9v).

use crate::octree::CellState;
use crate::sim::{CollisionSnapshot, World, WorldCoord};

/// Read-only voxel-grid surface needed by player collision / raycast.
///
/// The trait exists only to let the generic helpers in this module accept
/// both the live [`World`] and a cheap read-only [`CollisionSnapshot`]; it
/// is not intended for external impls. The two impls below are the only
/// ones expected to exist.
pub trait VoxelGrid {
    fn cell_at(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState;
}

impl VoxelGrid for World {
    fn cell_at(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        self.get(x, y, z)
    }
}

impl VoxelGrid for CollisionSnapshot {
    fn cell_at(&self, x: WorldCoord, y: WorldCoord, z: WorldCoord) -> CellState {
        self.get(x, y, z)
    }
}

/// Camera mode: orbit around a target point, or first-person at the player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraMode {
    /// Debug orbit camera (original). Mouse drag rotates, scroll zooms.
    Orbit,
    /// First-person: eye at player position, mouse look, WASD movement.
    FirstPerson,
}

/// Player movement speed in cells per second.
pub const PLAYER_SPEED: f64 = 9.0;
/// Sprint multiplier.
pub const PLAYER_SPRINT: f64 = 2.5;
/// Jump launch speed in cells per second.
pub const PLAYER_JUMP_SPEED: f64 = 8.5;
/// Downward acceleration in cells per second squared.
pub const PLAYER_GRAVITY: f64 = 28.0;
/// Player bounding box half-width on X/Z (cells).
pub const PLAYER_HALF_W: f64 = 0.3;
/// Player height (cells).
pub const PLAYER_HEIGHT: f64 = 1.6;
/// Maximum vertical lift the collision solver will try to clear a low ledge.
const STEP_UP_HEIGHT: f64 = 1.0;
/// Maximum drop to snap down to a nearby floor when moving horizontally.
const GROUND_SNAP_HEIGHT: f64 = 0.35;
/// Mouse look sensitivity (radians per pixel).
pub const LOOK_SENSITIVITY: f64 = 0.003;
/// Maximum raycast range for block place/break (in cells).
pub const INTERACT_RANGE: f64 = 40.0;

const GROUND_CONTACT_EPSILON: f64 = 0.05;
const VERTICAL_COLLISION_STEPS: usize = 12;

/// Result of one grounded-movement step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundedMoveResult {
    pub pos: [f64; 3],
    pub vertical_velocity: f64,
    pub grounded: bool,
}

/// Per-frame inputs for grounded first-person movement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundedMoveInput {
    pub yaw: f64,
    pub move_input: [f64; 2],
    pub speed: f64,
    pub dt: f64,
    pub jump_requested: bool,
}

/// Camera-space offsets layered on top of the first-person eye position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstPersonCameraOffset {
    pub vertical: f64,
    pub lateral: f64,
    pub forward: f64,
}

/// Small deterministic state machine that adds weight to first-person camera
/// motion without entangling renderer-side camera math.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FirstPersonCameraFeel {
    phase: f64,
    vertical: f64,
    lateral: f64,
    impulse: f64,
    moving: bool,
    grounded: bool,
    initialized: bool,
}

impl FirstPersonCameraFeel {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn tick(
        &mut self,
        planar_speed: f64,
        sprinting: bool,
        grounded: bool,
        dt: f64,
    ) -> FirstPersonCameraOffset {
        if dt <= 0.0 {
            return FirstPersonCameraOffset {
                vertical: self.vertical + self.impulse,
                lateral: self.lateral,
                forward: -self.impulse * 0.5,
            };
        }

        let active = grounded && planar_speed > 0.1;
        if !self.initialized {
            self.moving = active;
            self.grounded = grounded;
            self.initialized = true;
        }
        if active {
            let cadence_hz = 1.2 + planar_speed * 0.12;
            self.phase = (self.phase + dt * cadence_hz * std::f64::consts::TAU)
                .rem_euclid(std::f64::consts::TAU);
        }

        let speed_ratio = (planar_speed / (PLAYER_SPEED * PLAYER_SPRINT)).clamp(0.0, 1.0);
        let bob_gain = if sprinting { 1.3 } else { 1.0 };
        let target_vertical = if active {
            -self.phase.sin().abs() * (0.012 + 0.014 * speed_ratio) * bob_gain
        } else {
            0.0
        };
        let target_lateral = if active {
            self.phase.cos() * (0.005 + 0.007 * speed_ratio) * bob_gain
        } else {
            0.0
        };

        let smoothing = 1.0 - (-dt * 12.0).exp();
        self.vertical += (target_vertical - self.vertical) * smoothing;
        self.lateral += (target_lateral - self.lateral) * smoothing;

        if self.initialized && active != self.moving {
            self.impulse = if active { 0.010 } else { -0.012 };
        }
        if self.initialized && grounded && !self.grounded {
            self.impulse = self.impulse.min(-0.016);
        }
        self.impulse *= (-dt * 10.0).exp();

        self.moving = active;
        self.grounded = grounded;

        FirstPersonCameraOffset {
            vertical: self.vertical + self.impulse,
            lateral: self.lateral,
            forward: -self.impulse * 0.5,
        }
    }
}

/// Check if the player's AABB at `pos` overlaps any solid cell.
pub fn player_collides<W: VoxelGrid + ?Sized>(world: &W, pos: &[f64; 3]) -> bool {
    let hw = PLAYER_HALF_W;
    let x_min = (pos[0] - hw).floor() as i64;
    let x_max = (pos[0] + hw).floor() as i64;
    let y_min = pos[1].floor() as i64;
    let y_max = (pos[1] + PLAYER_HEIGHT).floor() as i64;
    let z_min = (pos[2] - hw).floor() as i64;
    let z_max = (pos[2] + hw).floor() as i64;
    for cx in x_min..=x_max {
        for cy in y_min..=y_max {
            for cz in z_min..=z_max {
                let cell = world.cell_at(WorldCoord(cx), WorldCoord(cy), WorldCoord(cz));
                if cell != 0 {
                    return true;
                }
            }
        }
    }
    false
}

fn player_supported_at_y<W: VoxelGrid + ?Sized>(world: &W, pos: &[f64; 3], support_y: i64) -> bool {
    let hw = PLAYER_HALF_W;
    let x_min = (pos[0] - hw).floor() as i64;
    let x_max = (pos[0] + hw).floor() as i64;
    let z_min = (pos[2] - hw).floor() as i64;
    let z_max = (pos[2] + hw).floor() as i64;
    for cx in x_min..=x_max {
        for cz in z_min..=z_max {
            if world.cell_at(WorldCoord(cx), WorldCoord(support_y), WorldCoord(cz)) != 0 {
                return true;
            }
        }
    }
    false
}

fn snap_down_to_support<W: VoxelGrid + ?Sized>(
    world: &W,
    pos: &[f64; 3],
    max_drop: f64,
) -> Option<[f64; 3]> {
    let top_support = pos[1].floor() as i64 - 1;
    let bottom_support = (pos[1] - max_drop).floor() as i64 - 1;
    for support_y in (bottom_support..=top_support).rev() {
        let candidate_y = support_y as f64 + 1.0;
        if pos[1] - candidate_y < -1e-9 || pos[1] - candidate_y > max_drop + 1e-9 {
            continue;
        }
        let mut candidate = *pos;
        candidate[1] = candidate_y;
        if !player_collides(world, &candidate)
            && player_supported_at_y(world, &candidate, support_y)
        {
            return Some(candidate);
        }
    }
    None
}

fn try_step_move<W: VoxelGrid + ?Sized>(
    world: &W,
    pos: &[f64; 3],
    axis: usize,
    delta: f64,
) -> Option<[f64; 3]> {
    if axis == 1 || delta.abs() < 1e-9 {
        return None;
    }
    snap_down_to_support(world, pos, GROUND_SNAP_HEIGHT)?;
    let mut raised = *pos;
    raised[1] += STEP_UP_HEIGHT;
    if player_collides(world, &raised) {
        return None;
    }
    raised[axis] += delta;
    if player_collides(world, &raised) {
        return None;
    }
    snap_down_to_support(world, &raised, STEP_UP_HEIGHT)
}

fn is_contact_grounded<W: VoxelGrid + ?Sized>(world: &W, pos: &[f64; 3]) -> bool {
    let mut probe = *pos;
    probe[1] -= GROUND_CONTACT_EPSILON;
    player_collides(world, &probe)
}

/// Report whether the player is standing on support or close enough to snap
/// onto it; camera feel should read those states as grounded locomotion.
pub fn is_grounded<W: VoxelGrid + ?Sized>(world: &W, pos: &[f64; 3]) -> bool {
    let support_y = pos[1].floor() as i64 - 1;
    player_supported_at_y(world, pos, support_y)
        || snap_down_to_support(world, pos, GROUND_SNAP_HEIGHT).is_some()
}

fn raycast_cells_with_range<W: VoxelGrid + ?Sized>(
    world: &W,
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
) -> Option<([i64; 3], [i64; 3])> {
    if max_dist <= 0.0 {
        return None;
    }

    let mut pos = [
        origin[0].floor() as i64,
        origin[1].floor() as i64,
        origin[2].floor() as i64,
    ];
    let step = [
        if dir[0] >= 0.0 { 1i64 } else { -1 },
        if dir[1] >= 0.0 { 1i64 } else { -1 },
        if dir[2] >= 0.0 { 1i64 } else { -1 },
    ];
    let mut t_max = [0.0f64; 3];
    let mut t_delta = [0.0f64; 3];
    for i in 0..3 {
        if dir[i].abs() < 1e-12 {
            t_max[i] = f64::INFINITY;
            t_delta[i] = f64::INFINITY;
        } else {
            let boundary = if dir[i] > 0.0 {
                (pos[i] + 1) as f64
            } else {
                pos[i] as f64
            };
            t_max[i] = (boundary - origin[i]) / dir[i];
            t_delta[i] = (step[i] as f64) / dir[i];
        }
    }

    let max_steps = (max_dist.ceil() as usize).saturating_mul(2).max(1);
    let mut prev = pos;
    for _ in 0..max_steps {
        let cell = world.cell_at(WorldCoord(pos[0]), WorldCoord(pos[1]), WorldCoord(pos[2]));
        if cell != 0 {
            return Some((pos, prev));
        }
        prev = pos;

        if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
            pos[0] += step[0];
            t_max[0] += t_delta[0];
        } else if t_max[1] < t_max[2] {
            pos[1] += step[1];
            t_max[1] += t_delta[1];
        } else {
            pos[2] += step[2];
            t_max[2] += t_delta[2];
        }

        let travel_t = t_max[0].min(t_max[1]).min(t_max[2]);
        if travel_t > max_dist {
            break;
        }
    }
    None
}

fn ray_aabb_entry_t(origin: [f64; 3], dir: [f64; 3], cell: [i64; 3]) -> Option<f64> {
    let mut t_min = f64::NEG_INFINITY;
    let mut t_max = f64::INFINITY;
    for axis in 0..3 {
        let min = cell[axis] as f64;
        let max = min + 1.0;
        if dir[axis].abs() < 1e-12 {
            if origin[axis] < min || origin[axis] > max {
                return None;
            }
            continue;
        }
        let inv_dir = 1.0 / dir[axis];
        let mut axis_t0 = (min - origin[axis]) * inv_dir;
        let mut axis_t1 = (max - origin[axis]) * inv_dir;
        if axis_t0 > axis_t1 {
            std::mem::swap(&mut axis_t0, &mut axis_t1);
        }
        t_min = t_min.max(axis_t0);
        t_max = t_max.min(axis_t1);
        if t_max < t_min {
            return None;
        }
    }
    Some(t_min.max(0.0))
}

/// Return whether `target` is directly visible from `origin` through empty
/// cells. Used to keep particle billboards from drawing through voxel walls.
pub fn has_line_of_sight<W: VoxelGrid + ?Sized>(
    world: &W,
    origin: [f64; 3],
    target: [f64; 3],
) -> bool {
    let delta = [
        target[0] - origin[0],
        target[1] - origin[1],
        target[2] - origin[2],
    ];
    let dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if dist <= 1e-9 {
        return true;
    }
    let dir = [delta[0] / dist, delta[1] / dist, delta[2] / dist];
    match raycast_cells_with_range(world, origin, dir, dist) {
        None => true,
        Some((hit, _)) => ray_aabb_entry_t(origin, dir, hit).is_some_and(|entry_t| entry_t >= dist),
    }
}

/// DDA raycast through the cell grid. Returns `(hit_cell, prev_cell)` —
/// `hit_cell` is the first solid cell along the ray, `prev_cell` is the
/// empty cell just before it (for block placement).
pub fn raycast_cells<W: VoxelGrid + ?Sized>(
    world: &W,
    origin: [f64; 3],
    dir: [f64; 3],
) -> Option<([i64; 3], [i64; 3])> {
    raycast_cells_with_range(world, origin, dir, INTERACT_RANGE)
}

/// Compute the player's eye position and look direction from position + angles.
pub fn eye_ray(pos: &[f64; 3], yaw: f64, pitch: f64) -> ([f64; 3], [f64; 3]) {
    let eye = [pos[0], pos[1] + PLAYER_HEIGHT * 0.85, pos[2]];
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let dir = [-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw];
    (eye, dir)
}

/// Compute a movement delta from yaw + input directions + speed + dt.
///
/// `input` is a unit-ish direction in player-local space:
/// `[forward, right, up]` where each component is -1, 0, or 1.
/// Returns the world-space delta vector, normalized to `speed * dt` length.
pub fn compute_move_delta(yaw: f64, input: [f64; 3], speed: f64, dt: f64) -> [f64; 3] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let fwd = [-sin_yaw, 0.0, -cos_yaw];
    let right = [cos_yaw, 0.0, -sin_yaw];

    let move_dir = [
        fwd[0] * input[0] + right[0] * input[1],
        input[2],
        fwd[2] * input[0] + right[2] * input[1],
    ];

    let len =
        (move_dir[0] * move_dir[0] + move_dir[1] * move_dir[1] + move_dir[2] * move_dir[2]).sqrt();

    if len < 1e-9 {
        return [0.0; 3];
    }

    let inv_len = (speed * dt) / len;
    [
        move_dir[0] * inv_len,
        move_dir[1] * inv_len,
        move_dir[2] * inv_len,
    ]
}

/// Apply a movement delta with per-axis collision resolution.
/// Returns the new position after sliding along walls.
fn apply_movement_internal<W: VoxelGrid + ?Sized>(
    world: &W,
    pos: &[f64; 3],
    delta: &[f64; 3],
    grounded: bool,
) -> [f64; 3] {
    let mut new_pos = *pos;
    for axis in 0..3 {
        let old = new_pos[axis];
        new_pos[axis] += delta[axis];
        if player_collides(world, &new_pos) {
            new_pos[axis] = old;
            if grounded {
                if let Some(stepped) = try_step_move(world, &new_pos, axis, delta[axis]) {
                    new_pos = stepped;
                }
            }
        }
    }
    if grounded && delta[1] <= 0.0 {
        if let Some(snapped) = snap_down_to_support(world, &new_pos, GROUND_SNAP_HEIGHT) {
            new_pos = snapped;
        }
    }
    new_pos
}

pub fn apply_movement<W: VoxelGrid + ?Sized>(
    world: &W,
    pos: &[f64; 3],
    delta: &[f64; 3],
) -> [f64; 3] {
    apply_movement_internal(world, pos, delta, true)
}

fn apply_vertical_movement<W: VoxelGrid + ?Sized>(
    world: &W,
    pos: &[f64; 3],
    delta_y: f64,
) -> ([f64; 3], bool) {
    if delta_y.abs() < 1e-9 {
        return (*pos, false);
    }

    let mut target = *pos;
    target[1] += delta_y;
    if !player_collides(world, &target) {
        return (target, false);
    }

    let mut settled = *pos;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..VERTICAL_COLLISION_STEPS {
        let mid = (low + high) * 0.5;
        let mut probe = *pos;
        probe[1] += delta_y * mid;
        if player_collides(world, &probe) {
            high = mid;
        } else {
            low = mid;
            settled = probe;
        }
    }
    (settled, true)
}

/// Apply one grounded first-person movement step with jump/gravity.
pub fn step_grounded_movement<W: VoxelGrid + ?Sized>(
    world: &W,
    pos: &[f64; 3],
    vertical_velocity: f64,
    input: GroundedMoveInput,
) -> GroundedMoveResult {
    let grounded_before = is_contact_grounded(world, pos);
    let horizontal = compute_move_delta(
        input.yaw,
        [input.move_input[0], input.move_input[1], 0.0],
        input.speed,
        input.dt,
    );
    let mut new_pos = apply_movement_internal(world, pos, &horizontal, grounded_before);
    let mut grounded = is_contact_grounded(world, &new_pos);
    let mut vertical_velocity = if grounded && vertical_velocity < 0.0 {
        0.0
    } else {
        vertical_velocity
    };

    if grounded {
        if input.jump_requested {
            vertical_velocity = PLAYER_JUMP_SPEED;
            grounded = false;
        } else {
            vertical_velocity = 0.0;
        }
    } else {
        vertical_velocity -= PLAYER_GRAVITY * input.dt;
    }

    let (vertical_pos, blocked) =
        apply_vertical_movement(world, &new_pos, vertical_velocity * input.dt);
    new_pos = vertical_pos;
    if blocked {
        grounded = vertical_velocity <= 0.0 && is_contact_grounded(world, &new_pos);
        vertical_velocity = 0.0;
    } else if vertical_velocity <= 0.0 && is_contact_grounded(world, &new_pos) {
        grounded = true;
        vertical_velocity = 0.0;
    }

    GroundedMoveResult {
        pos: new_pos,
        vertical_velocity,
        grounded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::Cell;

    fn world_with_wall(x: i64, y: i64, z: i64) -> World {
        let mut world = World::new(3); // 8³
        let material = Cell::pack(1, 0).raw();
        world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), material);
        world
    }

    fn world_with_floor(y: i64, x0: i64, x1: i64, z0: i64, z1: i64) -> World {
        let mut world = World::new(3);
        let material = Cell::pack(1, 0).raw();
        for x in x0..=x1 {
            for z in z0..=z1 {
                world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), material);
            }
        }
        world
    }

    #[test]
    fn collides_with_solid() {
        let world = world_with_wall(4, 4, 4);
        assert!(player_collides(&world, &[4.0, 4.0, 4.0]));
    }

    #[test]
    fn no_collision_in_empty() {
        let world = World::new(3);
        assert!(!player_collides(&world, &[4.0, 4.0, 4.0]));
    }

    #[test]
    fn raycast_hits_wall() {
        let world = world_with_wall(4, 4, 0);
        // Shoot from (4, 4, 3) toward -Z
        let result = raycast_cells(&world, [4.5, 4.5, 3.5], [0.0, 0.0, -1.0]);
        assert!(result.is_some());
        let (hit, prev) = result.unwrap();
        assert_eq!(hit, [4, 4, 0]);
        assert_eq!(prev, [4, 4, 1]);
    }

    #[test]
    fn raycast_misses_empty() {
        let world = World::new(3);
        let result = raycast_cells(&world, [4.5, 4.5, 4.5], [0.0, 0.0, -1.0]);
        assert!(result.is_none());
    }

    #[test]
    fn raycast_origin_in_solid_returns_prev_eq_hit() {
        let world = world_with_wall(4, 4, 4);
        let result = raycast_cells(&world, [4.5, 4.5, 4.5], [0.0, 0.0, -1.0]);
        assert!(result.is_some());
        let (hit, prev) = result.unwrap();
        assert_eq!(hit, prev); // documented behavior: prev == hit on first step
    }

    #[test]
    fn eye_ray_forward_at_zero_angles() {
        let (eye, dir) = eye_ray(&[5.0, 5.0, 5.0], 0.0, 0.0);
        assert!((eye[1] - (5.0 + PLAYER_HEIGHT * 0.85)).abs() < 1e-9);
        // At yaw=0 pitch=0, look direction is [0, 0, -1]
        assert!(dir[0].abs() < 1e-9);
        assert!(dir[1].abs() < 1e-9);
        assert!((dir[2] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn eye_ray_looking_right() {
        // yaw = pi/2 → look along -X
        let (_eye, dir) = eye_ray(&[5.0, 5.0, 5.0], std::f64::consts::FRAC_PI_2, 0.0);
        assert!((dir[0] - (-1.0)).abs() < 1e-9);
        assert!(dir[2].abs() < 1e-9);
    }

    #[test]
    fn compute_move_no_input_returns_zero() {
        let delta = compute_move_delta(0.0, [0.0, 0.0, 0.0], PLAYER_SPEED, 1.0 / 60.0);
        assert!(delta[0].abs() < 1e-9);
        assert!(delta[1].abs() < 1e-9);
        assert!(delta[2].abs() < 1e-9);
    }

    #[test]
    fn compute_move_forward_normalized() {
        let dt = 1.0 / 60.0;
        let delta = compute_move_delta(0.0, [1.0, 0.0, 0.0], PLAYER_SPEED, dt);
        let len = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        assert!((len - PLAYER_SPEED * dt).abs() < 1e-9);
    }

    #[test]
    fn compute_move_diagonal_same_speed() {
        let dt = 1.0 / 60.0;
        let forward = compute_move_delta(0.0, [1.0, 0.0, 0.0], PLAYER_SPEED, dt);
        let diagonal = compute_move_delta(0.0, [1.0, 1.0, 0.0], PLAYER_SPEED, dt);
        let len_fwd =
            (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
        let len_diag =
            (diagonal[0] * diagonal[0] + diagonal[1] * diagonal[1] + diagonal[2] * diagonal[2])
                .sqrt();
        // Diagonal should be same speed as forward (normalized)
        assert!((len_fwd - len_diag).abs() < 1e-9);
    }

    #[test]
    fn camera_feel_idle_stays_neutral() {
        let mut feel = FirstPersonCameraFeel::default();
        for _ in 0..30 {
            let offset = feel.tick(0.0, false, true, 1.0 / 60.0);
            assert!(offset.vertical.abs() < 1e-6);
            assert!(offset.lateral.abs() < 1e-6);
            assert!(offset.forward.abs() < 1e-6);
        }
    }

    #[test]
    fn camera_feel_sprint_bobs_more_than_walk() {
        let mut walk = FirstPersonCameraFeel::default();
        let mut sprint = FirstPersonCameraFeel::default();
        let mut walk_peak = 0.0f64;
        let mut sprint_peak = 0.0f64;

        for _ in 0..90 {
            let walk_offset = walk.tick(PLAYER_SPEED, false, true, 1.0 / 60.0);
            let sprint_offset = sprint.tick(PLAYER_SPEED * PLAYER_SPRINT, true, true, 1.0 / 60.0);
            walk_peak = walk_peak.max(walk_offset.vertical.abs() + walk_offset.lateral.abs());
            sprint_peak =
                sprint_peak.max(sprint_offset.vertical.abs() + sprint_offset.lateral.abs());
        }

        assert!(
            sprint_peak > walk_peak,
            "sprint bob should read stronger than walk bob"
        );
    }

    #[test]
    fn camera_feel_landing_adds_downward_impulse() {
        let mut feel = FirstPersonCameraFeel::default();
        let airborne = feel.tick(PLAYER_SPEED, false, false, 1.0 / 60.0);
        let landing = feel.tick(PLAYER_SPEED, false, true, 1.0 / 60.0);

        assert!(
            landing.vertical < airborne.vertical,
            "landing should add a downward camera response"
        );
        assert!(
            landing.forward > 0.0,
            "landing should add a small forward settle cue"
        );
    }

    #[test]
    fn line_of_sight_reaches_visible_target() {
        let world = World::new(3);
        assert!(has_line_of_sight(&world, [1.5, 1.5, 1.5], [5.5, 1.5, 1.5]));
    }

    #[test]
    fn line_of_sight_blocks_target_behind_wall() {
        let world = world_with_wall(3, 1, 1);
        assert!(!has_line_of_sight(&world, [1.5, 1.5, 1.5], [5.5, 1.5, 1.5]));
    }

    #[test]
    fn apply_movement_slides_along_wall() {
        // Wall at z=2, player at z=3.5 moving into it on Z
        let world = world_with_wall(5, 4, 2);
        let pos = [4.5, 4.5, 3.5];
        let delta = [0.5, 0.0, -1.5]; // would go into wall on Z
        let new_pos = apply_movement(&world, &pos, &delta);
        // X should have moved (no wall at x=5 for the new Z)
        assert!((new_pos[0] - 5.0).abs() < 1e-9);
        // Z should be rejected — moving to 2.0 would collide with wall at (5,4,2)
        assert!((new_pos[2] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn apply_movement_no_collision_moves_freely() {
        let world = World::new(3);
        let pos = [4.0, 4.0, 4.0];
        let delta = [1.0, 0.0, -1.0];
        let new_pos = apply_movement(&world, &pos, &delta);
        assert!((new_pos[0] - 5.0).abs() < 1e-9);
        assert!((new_pos[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn grounded_step_jumps_only_when_supported() {
        let world = world_with_floor(0, 0, 7, 0, 7);
        let dt = 1.0 / 60.0;
        let start = [4.0, 1.0, 4.0];

        let jumped = step_grounded_movement(
            &world,
            &start,
            0.0,
            GroundedMoveInput {
                yaw: 0.0,
                move_input: [0.0, 0.0],
                speed: PLAYER_SPEED,
                dt,
                jump_requested: true,
            },
        );
        assert!(jumped.pos[1] > start[1]);
        assert!(jumped.vertical_velocity > 0.0);
        assert!(!jumped.grounded);

        let midair = step_grounded_movement(
            &world,
            &jumped.pos,
            jumped.vertical_velocity,
            GroundedMoveInput {
                yaw: 0.0,
                move_input: [0.0, 0.0],
                speed: PLAYER_SPEED,
                dt,
                jump_requested: true,
            },
        );
        assert!(midair.pos[1] > jumped.pos[1]);
        assert!(midair.vertical_velocity < jumped.vertical_velocity);
    }

    #[test]
    fn grounded_step_gravity_lands_back_on_floor() {
        let world = world_with_floor(0, 0, 7, 0, 7);
        let dt = 1.0 / 60.0;
        let mut pos = [4.0, 4.0, 4.0];
        let mut vertical_velocity = 0.0;
        let mut grounded = false;

        for _ in 0..240 {
            let step = step_grounded_movement(
                &world,
                &pos,
                vertical_velocity,
                GroundedMoveInput {
                    yaw: 0.0,
                    move_input: [0.0, 0.0],
                    speed: PLAYER_SPEED,
                    dt,
                    jump_requested: false,
                },
            );
            pos = step.pos;
            vertical_velocity = step.vertical_velocity;
            grounded = step.grounded;
            if grounded {
                break;
            }
        }

        assert!(grounded);
        assert!((pos[1] - 1.0).abs() < 0.01);
        assert!(vertical_velocity.abs() < 1e-9);
    }

    #[test]
    fn apply_movement_steps_up_one_block_ledge() {
        let mut world = world_with_floor(0, 0, 7, 0, 7);
        let material = Cell::pack(1, 0).raw();
        world.set(WorldCoord(4), WorldCoord(1), WorldCoord(3), material);
        let pos = [3.5, 1.0, 3.5];
        let delta = [0.8, 0.0, 0.0];
        let new_pos = apply_movement(&world, &pos, &delta);
        assert!(
            (new_pos[0] - 4.3).abs() < 1e-9,
            "horizontal move should clear the ledge"
        );
        assert!(
            (new_pos[1] - 2.0).abs() < 1e-9,
            "step-up should land the player on top of the ledge"
        );
    }

    #[test]
    fn apply_movement_snaps_down_small_floor_gap() {
        let world = world_with_floor(0, 0, 7, 0, 7);
        let pos = [3.5, 1.2, 3.5];
        let delta = [0.4, 0.0, 0.0];
        let new_pos = apply_movement(&world, &pos, &delta);
        assert!(
            (new_pos[0] - 3.9).abs() < 1e-9,
            "horizontal movement should still apply"
        );
        assert!(
            (new_pos[1] - 1.0).abs() < 1e-9,
            "small drops should snap back onto the floor"
        );
    }

    #[test]
    fn collision_snapshot_matches_live_world() {
        // 0s9v: the snapshot used during background steps must answer
        // collision queries identically to the live world it was cloned
        // from. Otherwise grounded physics drifts between step boundaries.
        let mut world = world_with_floor(0, 0, 7, 0, 7);
        let material = Cell::pack(1, 0).raw();
        world.set(WorldCoord(4), WorldCoord(1), WorldCoord(3), material);
        world.set(WorldCoord(2), WorldCoord(2), WorldCoord(2), material);
        let snapshot = world.collision_snapshot();

        let sample_positions = [
            [3.5, 1.0, 3.5],
            [4.0, 1.0, 3.5],
            [4.0, 2.0, 3.5],
            [3.5, 1.2, 3.5],
            [1.5, 2.5, 1.5],
            [7.0, 0.5, 7.0],
        ];
        for pos in sample_positions {
            assert_eq!(
                player_collides(&world, &pos),
                player_collides(&snapshot, &pos),
                "player_collides disagrees at {:?}",
                pos,
            );
            assert_eq!(
                is_grounded(&world, &pos),
                is_grounded(&snapshot, &pos),
                "is_grounded disagrees at {:?}",
                pos,
            );
        }

        let step_input = GroundedMoveInput {
            yaw: 0.5,
            move_input: [1.0, 0.0],
            speed: PLAYER_SPEED,
            dt: 1.0 / 60.0,
            jump_requested: false,
        };
        for pos in sample_positions {
            let live = step_grounded_movement(&world, &pos, 0.0, step_input);
            let snap = step_grounded_movement(&snapshot, &pos, 0.0, step_input);
            assert_eq!(live.pos, snap.pos, "step pos disagrees at {:?}", pos);
            assert_eq!(
                live.vertical_velocity, snap.vertical_velocity,
                "step vy disagrees at {:?}",
                pos,
            );
            assert_eq!(
                live.grounded, snap.grounded,
                "step grounded disagrees at {:?}",
                pos,
            );
        }
    }
}
