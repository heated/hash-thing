// HUD overlay shader (hash-thing-5bb.10).
//
// Draws an open reticle and material signal bar in screen space.
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

// Reticle: two open side rails, two short vertical cues, and one
// material signal bar below center. Total: 5 quads = 30 vertices.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    // Reticle geometry (in NDC, where screen is [-1,1]²).
    let side_len = 0.018;
    let side_w = 0.0018;
    let side_gap = 0.010;
    let cue_len = 0.009;
    let cue_w = 0.0014;
    let cue_offset = 0.016;

    // Material signal bar.
    let sig_w = 0.018;
    let sig_h = 0.004;
    let sig_y = -0.050;

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
    var color = vec4<f32>(0.97, 0.91, 0.82, 0.86);

    if quad_id == 0u {
        // Left rail
        let x = -(side_gap + side_len * (cu * 0.5 + 0.5));
        pos = vec2<f32>(x / aspect, side_w * cv);
    } else if quad_id == 1u {
        // Right rail
        let x = side_gap + side_len * (cu * 0.5 + 0.5);
        pos = vec2<f32>(x / aspect, side_w * cv);
    } else if quad_id == 2u {
        // Upper cue
        let y = cue_offset + cue_len * (cv * 0.5 + 0.5);
        pos = vec2<f32>(cue_w * cu / aspect, y);
    } else if quad_id == 3u {
        // Lower cue
        let y = -(cue_offset + cue_len * (cv * 0.5 + 0.5));
        pos = vec2<f32>(cue_w * cu / aspect, y);
    } else if quad_id == 4u {
        // Material signal bar below center.
        pos = vec2<f32>(sig_w * cu / aspect, sig_y + sig_h * cv);
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
