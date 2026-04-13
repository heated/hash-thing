// Legend overlay shader (hash-thing-m1f.7.2).
//
// Draws a textured quad for the keybindings legend.
// The texture is CPU-rendered from a bitmap font.

@group(0) @binding(0) var legend_tex: texture_2d<f32>;
@group(0) @binding(1) var legend_samp: sampler;
// params: x = quad left (NDC), y = quad bottom (NDC),
//         z = quad width (NDC), w = quad height (NDC)
@group(0) @binding(2) var<uniform> params: vec4<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    // Two triangles forming a quad.
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    );
    let uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
    );

    let c = corners[vid];
    // Map [0,1] quad to NDC position using params.
    let ndc_x = params.x + c.x * params.z;
    let ndc_y = params.y + c.y * params.w;

    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(legend_tex, legend_samp, in.uv);
}
