// Particle billboard shader (hash-thing-5bb.9).
//
// Renders EntityStore particles as camera-facing quads. Each particle is
// one instance; the vertex shader expands 6 vertices (two triangles) into
// a billboard aligned to the camera's up/right vectors. The fragment
// shader draws a soft circular dot colored by the material palette.

struct Uniforms {
    camera_pos: vec4<f32>,
    camera_dir: vec4<f32>,
    camera_up: vec4<f32>,
    camera_right: vec4<f32>,
    params: vec4<f32>, // x: volume_size, y: aspect, z: fov_tan, w: time
};

// Per-particle data packed into a storage buffer.
// Each entry: vec4(pos.x, pos.y, pos.z, bitcast<f32>(material_u32))
struct Particle {
    pos_mat: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(3) var t_scene: texture_2d<f32>;
@group(0) @binding(4) var s_scene: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) screen_uv: vec2<f32>,
    @location(3) ray_t: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VertexOutput {
    // Quad corners: two triangles covering [-1,1]².
    // Vertex order: 0-1-2, 2-1-3 → (TL, TR, BL, BL, TR, BR)
    let quad_uvs = array<vec2<f32>, 6>(
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
    );
    let uv = quad_uvs[vid];

    let p = particles[iid];
    let world_pos = p.pos_mat.xyz;
    let material = bitcast<u32>(p.pos_mat.w);
    let center_ray_t = length(world_pos - u.camera_pos.xyz);

    // Billboard size in world units. Particles are small — 1/64 of the
    // volume (one voxel-ish). Scale slightly larger so they're visible.
    let size = 1.5 / u.params.x;

    // Expand quad in camera space.
    let right = u.camera_right.xyz;
    let up = u.camera_up.xyz;
    let offset = right * uv.x * size + up * uv.y * size;
    let pos = world_pos + offset;

    // View-projection: perspective projection matching the raycast camera.
    let eye = u.camera_pos.xyz;
    let fwd = u.camera_dir.xyz;
    let aspect = u.params.y;
    let fov_tan = u.params.z;

    // View-space transform.
    let rel = pos - eye;
    let vz = dot(rel, fwd);
    let vx = dot(rel, right);
    let vy = dot(rel, up);

    // Perspective divide — match the implicit projection of the raycast.
    let clip_x = vx / (vz * fov_tan * aspect);
    let clip_y = vy / (vz * fov_tan);

    var out: VertexOutput;
    out.position = vec4<f32>(clip_x, clip_y, 0.5, 1.0);
    // Hide particles behind the camera.
    if vz < 0.0 {
        out.position = vec4<f32>(0.0, 0.0, -1.0, 1.0); // clip away
    }
    out.uv = uv;
    out.color = palette[material];
    out.screen_uv = vec2<f32>(clip_x * 0.5 + 0.5, 0.5 - clip_y * 0.5);
    out.ray_t = center_ray_t;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(t_scene, s_scene, in.screen_uv);
    let depth_epsilon = 2.0 / u.params.x;
    if scene.a > 0.0 && scene.a + depth_epsilon < in.ray_t {
        discard;
    }

    // Soft circular dot — fade from center to edge.
    let dist = length(in.uv);
    if dist > 1.0 {
        discard;
    }
    let alpha = 1.0 - smoothstep(0.5, 1.0, dist);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
