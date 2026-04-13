//! Player controller: collision, raycast, movement.
//!
//! Pure functions operating on `World` + position data, extracted from
//! main.rs for testability (hash-thing-d6m).

use crate::sim::{World, WorldCoord};

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
/// Mouse look sensitivity (radians per pixel).
pub const LOOK_SENSITIVITY: f64 = 0.003;
/// Maximum raycast range for block place/break (in cells).
pub const INTERACT_RANGE: f64 = 40.0;

const GROUND_PROBE_EPSILON: f64 = 0.05;
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

/// Check if the player's AABB at `pos` overlaps any solid cell.
pub fn player_collides(world: &World, pos: &[f64; 3]) -> bool {
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
                let cell = world.get(WorldCoord(cx), WorldCoord(cy), WorldCoord(cz));
                if cell != 0 {
                    return true;
                }
            }
        }
    }
    false
}

/// DDA raycast through the cell grid. Returns `(hit_cell, prev_cell)` —
/// `hit_cell` is the first solid cell along the ray, `prev_cell` is the
/// empty cell just before it (for block placement).
pub fn raycast_cells(
    world: &World,
    origin: [f64; 3],
    dir: [f64; 3],
) -> Option<([i64; 3], [i64; 3])> {
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

    let max_steps = (INTERACT_RANGE * 2.0) as usize;
    let mut prev = pos;
    for _ in 0..max_steps {
        let cell = world.get(WorldCoord(pos[0]), WorldCoord(pos[1]), WorldCoord(pos[2]));
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
    }
    None
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
pub fn apply_movement(world: &World, pos: &[f64; 3], delta: &[f64; 3]) -> [f64; 3] {
    let mut new_pos = *pos;
    for axis in 0..3 {
        let old = new_pos[axis];
        new_pos[axis] += delta[axis];
        if player_collides(world, &new_pos) {
            new_pos[axis] = old;
        }
    }
    new_pos
}

/// True when the player is standing on solid ground (or within a tiny
/// settling epsilon of it).
pub fn is_grounded(world: &World, pos: &[f64; 3]) -> bool {
    let mut probe = *pos;
    probe[1] -= GROUND_PROBE_EPSILON;
    player_collides(world, &probe)
}

fn apply_vertical_movement(world: &World, pos: &[f64; 3], delta_y: f64) -> ([f64; 3], bool) {
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
pub fn step_grounded_movement(
    world: &World,
    pos: &[f64; 3],
    vertical_velocity: f64,
    input: GroundedMoveInput,
) -> GroundedMoveResult {
    let horizontal = compute_move_delta(
        input.yaw,
        [input.move_input[0], input.move_input[1], 0.0],
        input.speed,
        input.dt,
    );
    let mut new_pos = apply_movement(world, pos, &horizontal);
    let mut grounded = is_grounded(world, &new_pos);
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
        grounded = vertical_velocity <= 0.0 && is_grounded(world, &new_pos);
        vertical_velocity = 0.0;
    } else if vertical_velocity <= 0.0 && is_grounded(world, &new_pos) {
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

    fn world_with_floor() -> World {
        let mut world = World::new(3);
        let material = Cell::pack(1, 0).raw();
        for x in 0..8 {
            for z in 0..8 {
                world.set(WorldCoord(x), WorldCoord(0), WorldCoord(z), material);
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
        let world = world_with_floor();
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
        let world = world_with_floor();
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
}
