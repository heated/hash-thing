// HUD overlay shader (hash-thing-5bb.10).
//
// Draws a crosshair and material color indicator in screen space.
// No bind groups needed — all parameters come via push constants or
// are baked into the vertex positions.

struct HudUniforms {
    // x: material R, y: material G, z: material B, w: material A
    material_color: vec4<f32>,
    // x: aspect ratio (width/height), y: crosshair_on (0 or 1)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: HudUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Crosshair: 4 thin rectangles forming a + shape, plus a material indicator square.
// Total: 5 quads = 30 vertices.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    // Crosshair parameters (in NDC, where screen is [-1,1]²).
    let arm_len = 0.015;  // half-length of each arm
    let arm_w = 0.002;    // half-width of each arm
    let gap = 0.004;      // gap in center

    // Material indicator square.
    let sq_size = 0.02;
    let sq_x = -0.06;     // offset left of center
    let sq_y = -0.04;     // offset below center

    let aspect = u.params.x;

    // Which quad (0-4) and which vertex within the quad (0-5).
    let quad_id = vid / 6u;
    let local_v = vid % 6u;

    // Quad corners: TL, TR, BL, BL, TR, BR
    let corner_u = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let corner_v = array<f32, 6>(1.0, 1.0, -1.0, -1.0, -1.0, 1.0);
    let cu = corner_u[local_v];
    let cv = corner_v[local_v];

    var pos = vec2<f32>(0.0, 0.0);
    var color = vec4<f32>(1.0, 1.0, 1.0, 0.8); // white crosshair

    if quad_id == 0u {
        // Right arm
        pos = vec2<f32>((gap + arm_len * (cu * 0.5 + 0.5)) / aspect, arm_w * cv);
    } else if quad_id == 1u {
        // Left arm
        pos = vec2<f32>(-(gap + arm_len * (cu * 0.5 + 0.5)) / aspect, arm_w * cv);
    } else if quad_id == 2u {
        // Top arm
        pos = vec2<f32>(arm_w * cu / aspect, gap + arm_len * (cv * 0.5 + 0.5));
    } else if quad_id == 3u {
        // Bottom arm
        pos = vec2<f32>(arm_w * cu / aspect, -(gap + arm_len * (cv * 0.5 + 0.5)));
    } else if quad_id == 4u {
        // Material indicator square — bottom-left of crosshair
        pos = vec2<f32>((sq_x + sq_size * cu) / aspect, sq_y + sq_size * cv);
        color = u.material_color;
    }

    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
