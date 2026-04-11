struct Uniforms {
    camera_pos: vec4<f32>,
    camera_dir: vec4<f32>,
    camera_up: vec4<f32>,
    camera_right: vec4<f32>,
    params: vec4<f32>, // x: volume_size, y: aspect, z: fov_tan, w: time
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_3d<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle-pair via vertex ID
@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    // Two triangles covering [-1,1]
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

// Ray-AABB intersection (slab method)
fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

// Material color palette. Input is a packed cell word:
//   bits 15..6 material_id (10 bits)
//   bits  5..0 metadata    ( 6 bits)
// We index the palette by material_id only; metadata is ignored here
// (the material registry in 1v0.2 will replace this hard-coded switch).
//
// DRIFT GUARD: the shift amount `6u` below mirrors `Cell::METADATA_BITS`
// in src/octree/node.rs. If that constant ever changes, update this
// shader AND src/render/svdag_raycast.wgsl together. There is a pinning
// test in src/render/mod.rs (test `wgsl_metadata_shift_matches_rust`).
fn material_color(packed: u32) -> vec3<f32> {
    let mat_id = packed >> 6u;
    switch mat_id {
        case 1u: { return vec3<f32>(0.4, 0.8, 0.3); }  // green (life)
        case 2u: { return vec3<f32>(0.8, 0.3, 0.2); }  // red
        case 3u: { return vec3<f32>(0.2, 0.4, 0.9); }  // blue
        case 4u: { return vec3<f32>(0.9, 0.8, 0.2); }  // yellow
        default: { return vec3<f32>(0.6, 0.7, 0.8); }  // default gray-blue
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let vol_size = u.params.x;
    let aspect = u.params.y;
    let fov_tan = u.params.z;

    // Construct ray
    let rd = normalize(
        u.camera_dir.xyz +
        u.camera_right.xyz * in.uv.x * aspect * fov_tan +
        u.camera_up.xyz * in.uv.y * fov_tan
    );
    let ro = u.camera_pos.xyz;

    // Volume sits in [0, 1]³
    let hit = intersect_aabb(ro, 1.0 / rd, vec3<f32>(0.0), vec3<f32>(1.0));

    if hit.x > hit.y || hit.y < 0.0 {
        // Miss — background gradient
        let t = in.uv.y * 0.5 + 0.5;
        let bg = mix(vec3<f32>(0.05, 0.05, 0.08), vec3<f32>(0.1, 0.1, 0.15), t);
        return vec4<f32>(bg, 1.0);
    }

    // DDA voxel traversal
    let step_size = 1.0 / vol_size;
    var t = max(hit.x, 0.0) + step_size * 0.01;
    let max_steps = u32(vol_size * 3.0);

    var color = vec3<f32>(0.0);
    var hit_something = false;

    for (var i = 0u; i < max_steps; i = i + 1u) {
        if t > hit.y { break; }

        let pos = ro + rd * t;
        let voxel_coord = vec3<i32>(pos * vol_size);

        // Bounds check
        if all(voxel_coord >= vec3<i32>(0)) && all(voxel_coord < vec3<i32>(i32(vol_size))) {
            let val = textureLoad(volume, voxel_coord, 0).r;
            if val > 0u {
                hit_something = true;

                // Basic lighting: normal from gradient
                let base_color = material_color(val);

                // Approximate normal via central differences
                let eps = vec3<i32>(1, 0, 0);
                let nx = f32(textureLoad(volume, voxel_coord + eps, 0).r) -
                         f32(textureLoad(volume, voxel_coord - eps, 0).r);
                let ny = f32(textureLoad(volume, voxel_coord + vec3<i32>(0, 1, 0), 0).r) -
                         f32(textureLoad(volume, voxel_coord - vec3<i32>(0, 1, 0), 0).r);
                let nz = f32(textureLoad(volume, voxel_coord + vec3<i32>(0, 0, 1), 0).r) -
                         f32(textureLoad(volume, voxel_coord - vec3<i32>(0, 0, 1), 0).r);
                var normal = normalize(vec3<f32>(-nx, -ny, -nz));
                if length(vec3<f32>(nx, ny, nz)) < 0.01 {
                    // Flat surface — use ray direction for shading
                    normal = -rd;
                }

                // Directional light + ambient
                let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
                let diffuse = max(dot(normal, light_dir), 0.0);
                let ambient = 0.25;

                // Depth fog
                let fog = exp(-t * 1.5);
                color = base_color * (ambient + diffuse * 0.75) * fog;

                break;
            }
        }

        t += step_size;
    }

    if !hit_something {
        let t_bg = in.uv.y * 0.5 + 0.5;
        color = mix(vec3<f32>(0.05, 0.05, 0.08), vec3<f32>(0.1, 0.1, 0.15), t_bg);
    }

    return vec4<f32>(color, 1.0);
}
