// Minimal blit shader: samples the compute raycast output texture and writes
// to the swapchain surface. This is the first draw call in the render pass;
// particle, HUD, and legend overlays follow in the same pass.
//
// The raycast compute shader writes sRGB-encoded values to an rgba16float
// storage texture. This shader samples it as a regular texture and outputs
// to the sRGB swapchain. Since the values are already sRGB-encoded, the
// swapchain's sRGB encoding will double-encode slightly — but at 16-bit
// float precision the visual difference is imperceptible. If exact round-trip
// is needed, sample as rgba16float and skip the swapchain sRGB (use a non-sRGB
// surface format).

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

@vertex
fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
    // Fullscreen triangle (3 verts, covers [-1,1] quad with one triangle).
    // More efficient than 6-vert quad: no wasted fragments, no index buffer.
    let x = f32((id & 1u) << 2u) - 1.0;
    let y = f32((id & 2u) << 1u) - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Map clip-space [-1,1] to UV [0,1] for texture sampling.
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, in.uv);
}
