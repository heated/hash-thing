// SVDAG raycasting shader.
//
// Iterative descent through a sparse voxel DAG, rendered via a fullscreen pass.
//
// Node layout (9 u32s per interior node):
//   [0]: child_mask (low 8 bits = octant occupancy)
//   [1..=8]: child entries, where each entry is:
//     - bit 31 set → leaf, low 16 bits = material state
//     - bit 31 clear → interior, value = offset of child node in buffer
//
// Root is at offset 0. Root bounds are the unit cube [0,1]^3.
//
// Traversal: explicit stack with depth capped at MAX_DEPTH (12 levels → 4096^3).
// At each step we descend into the octant the ray is currently in. If empty,
// we step the ray to the next octant boundary using DDA on the current level's
// cell size. If leaf, shade it. If interior, push and descend.

struct Uniforms {
    camera_pos: vec4<f32>,
    camera_dir: vec4<f32>,
    camera_up: vec4<f32>,
    camera_right: vec4<f32>,
    // x: root_side_cells (2^root_level), y: aspect, z: fov_tan, w: time
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> dag_nodes: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[id], 0.0, 1.0);
    out.uv = positions[id];
    return out;
}

fn material_color(mat: u32) -> vec3<f32> {
    switch mat {
        case 1u: { return vec3<f32>(0.4, 0.8, 0.3); }
        case 2u: { return vec3<f32>(0.8, 0.3, 0.2); }
        case 3u: { return vec3<f32>(0.2, 0.4, 0.9); }
        case 4u: { return vec3<f32>(0.9, 0.8, 0.2); }
        default: { return vec3<f32>(0.6, 0.7, 0.8); }
    }
}

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

// Octant index from a point relative to the node center.
// Returns bit-packed index 0..7 with bit 0 = x, bit 1 = y, bit 2 = z.
fn octant_of(pos: vec3<f32>, node_min: vec3<f32>, half: f32) -> u32 {
    var idx: u32 = 0u;
    if pos.x >= node_min.x + half { idx |= 1u; }
    if pos.y >= node_min.y + half { idx |= 2u; }
    if pos.z >= node_min.z + half { idx |= 4u; }
    return idx;
}

const MAX_DEPTH: u32 = 14u;
const MAX_STEPS: u32 = 512u;
const LEAF_BIT: u32 = 0x80000000u;

// Iterative DAG descent.
//
// Strategy: for each ray, start at root. At each level, compute which octant
// the ray is currently in, look at the child. If leaf non-empty → hit.
// If empty → step ray past the octant boundary and re-test. If interior →
// descend. When we exit the node, pop the stack.
//
// This is a straightforward first implementation — not Laine-Karras optimal,
// but correct enough to prove the pipeline works and reserve the complexity
// budget. Optimizations (ray-octant mirroring, ancestor memoization, bitmask
// coalescing) can be layered on later.
fn raycast(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let inv_rd = 1.0 / rd;

    // Intersect ray with root cube [0,1]^3
    let root_hit = intersect_aabb(ro, inv_rd, vec3<f32>(0.0), vec3<f32>(1.0));
    if root_hit.x > root_hit.y || root_hit.y < 0.0 {
        return vec4<f32>(0.0);
    }

    // Stack of (node_offset, node_min, half_size)
    var stack_node: array<u32, MAX_DEPTH>;
    var stack_min: array<vec3<f32>, MAX_DEPTH>;
    var stack_half: array<f32, MAX_DEPTH>;
    var depth: u32 = 0u;

    stack_node[0] = 0u;
    stack_min[0] = vec3<f32>(0.0);
    stack_half[0] = 0.5;

    // Start the ray just past the root entry point.
    var t = max(root_hit.x, 0.0) + 1e-5;

    for (var step = 0u; step < MAX_STEPS; step = step + 1u) {
        if t > root_hit.y { break; }
        let pos = ro + rd * t;

        // If the current top-of-stack no longer contains `pos`, pop.
        // NOTE: STRICT bounds (no EPS slack). The step-past advances `t` by +EPS
        // which moves pos past the boundary by |rd_axis| * EPS > 0. If we allowed
        // EPS slack here, the pop would consider pos "inside" after a step-past
        // across an octant boundary (since |rd| * EPS < EPS on the pop side),
        // causing infinite loops on simultaneous two-axis boundary crossings.
        var top = depth;
        loop {
            let node_min = stack_min[top];
            let h2 = stack_half[top] * 2.0;
            if all(pos >= node_min) && all(pos < node_min + h2) {
                break;
            }
            if top == 0u {
                // Exited root
                return vec4<f32>(0.0);
            }
            top = top - 1u;
        }
        depth = top;

        // Descend until leaf or empty
        var descended = false;
        for (var d = 0u; d < MAX_DEPTH; d = d + 1u) {
            let node_offset = stack_node[depth];
            let node_min = stack_min[depth];
            let half = stack_half[depth];
            let oct = octant_of(pos, node_min, half);
            let child_slot = dag_nodes[node_offset + 1u + oct];

            if (child_slot & LEAF_BIT) != 0u {
                let mat = child_slot & 0xFFFFu;
                if mat > 0u {
                    // HIT
                    let base = material_color(mat);
                    // Cheap lighting: distance fog + flat shading based on ray dir
                    let fog = exp(-t * 1.2);
                    let shade = 0.35 + 0.65 * abs(rd.y);
                    return vec4<f32>(base * shade * fog, 1.0);
                }
                // Empty leaf — break to step ray
                break;
            } else {
                // Interior node — descend
                if depth + 1u >= MAX_DEPTH { break; }
                let child_min = vec3<f32>(
                    node_min.x + f32(oct & 1u) * half,
                    node_min.y + f32((oct >> 1u) & 1u) * half,
                    node_min.z + f32((oct >> 2u) & 1u) * half,
                );
                depth = depth + 1u;
                stack_node[depth] = child_slot;
                stack_min[depth] = child_min;
                stack_half[depth] = half * 0.5;
                descended = true;
            }
        }

        // Step the ray past the current (empty) octant.
        // Compute t-value at which the ray exits the octant we're in.
        let node_min = stack_min[depth];
        let half = stack_half[depth];
        let oct = octant_of(pos, node_min, half);
        let child_min = vec3<f32>(
            node_min.x + f32(oct & 1u) * half,
            node_min.y + f32((oct >> 1u) & 1u) * half,
            node_min.z + f32((oct >> 2u) & 1u) * half,
        );
        let child_max = child_min + vec3<f32>(half);
        let oct_hit = intersect_aabb(ro, inv_rd, child_min, child_max);
        // Advance t past the octant exit
        t = oct_hit.y + 1e-5;
    }

    return vec4<f32>(0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = u.params.y;
    let fov_tan = u.params.z;

    let rd = normalize(
        u.camera_dir.xyz +
        u.camera_right.xyz * in.uv.x * aspect * fov_tan +
        u.camera_up.xyz * in.uv.y * fov_tan
    );
    let ro = u.camera_pos.xyz;

    let hit = raycast(ro, rd);
    if hit.a > 0.0 {
        return hit;
    }

    // Background gradient
    let t = in.uv.y * 0.5 + 0.5;
    let bg = mix(vec3<f32>(0.05, 0.05, 0.08), vec3<f32>(0.1, 0.1, 0.15), t);
    return vec4<f32>(bg, 1.0);
}
