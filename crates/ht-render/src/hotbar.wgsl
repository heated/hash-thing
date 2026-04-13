// Hotbar overlay shader (hash-thing-e7k.6).
// Draws 18 material slots as colored quads at the bottom of the screen.
// Each slot is a 6-vertex quad (two triangles). Total: 108 vertices.

struct HotbarUniforms {
    aspect: f32,
    selected_slot: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: HotbarUniforms;
@group(0) @binding(1) var<storage, read> palette: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

const SLOT_COUNT: u32 = 18u;
const SLOT_SIZE: f32 = 0.05;   // NDC half-width of each slot
const SLOT_GAP: f32 = 0.008;   // gap between slots
const BAR_Y: f32 = -0.92;      // vertical center of the bar (NDC)

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let slot = vid / 6u;
    let vert = vid % 6u;

    // Total bar width: SLOT_COUNT slots + (SLOT_COUNT - 1) gaps.
    let total_w = f32(SLOT_COUNT) * SLOT_SIZE * 2.0 + f32(SLOT_COUNT - 1u) * SLOT_GAP;
    let bar_left = -total_w * 0.5;

    // Slot center x.
    let cx = bar_left + SLOT_SIZE + f32(slot) * (SLOT_SIZE * 2.0 + SLOT_GAP);
    let cy = BAR_Y;

    // Aspect-corrected half-sizes (slots are square on screen).
    let hx = SLOT_SIZE / uniforms.aspect;
    let hy = SLOT_SIZE;

    // Quad vertices: 0-1-2, 2-1-3 (two triangles).
    var dx: f32;
    var dy: f32;
    switch vert {
        case 0u: { dx = -1.0; dy = -1.0; }
        case 1u: { dx =  1.0; dy = -1.0; }
        case 2u: { dx = -1.0; dy =  1.0; }
        case 3u: { dx = -1.0; dy =  1.0; }
        case 4u: { dx =  1.0; dy = -1.0; }
        default: { dx =  1.0; dy =  1.0; }
    }

    let px = cx + dx * hx;
    let py = cy + dy * hy;

    // Color from palette. Slot 0 = material 0 (AIR = black), etc.
    var col = vec4<f32>(0.1, 0.1, 0.1, 0.7); // default dark
    if slot < arrayLength(&palette) {
        let p = palette[slot];
        col = vec4<f32>(p.rgb, 0.85);
    }

    // Highlight selected slot with a brighter border effect.
    if slot == u32(uniforms.selected_slot) {
        col = vec4<f32>(min(col.rgb * 1.4 + 0.15, vec3<f32>(1.0)), 0.95);
    }

    var out: VertexOutput;
    out.position = vec4<f32>(px, py, 0.0, 1.0);
    out.color = col;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
