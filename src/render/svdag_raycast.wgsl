// SVDAG raycasting shader.
//
// Iterative descent through a sparse voxel DAG, rendered via a fullscreen pass.
//
// Buffer layout (dag_nodes: array<u32>):
//   [0]:     root_offset — absolute index of the current root node's slot
//   [1..]:   append-only stream of 9-u32 interior-node slots
//
// Interior node slot (9 u32s):
//   [0]: child_mask (low 8 bits = octant occupancy)
//   [1..=8]: child entries, where each entry is:
//     - bit 31 set → leaf, low 16 bits = material state
//     - bit 31 clear → interior, value = absolute offset of child node in buffer
//
// Root lives at `dag_nodes[dag_nodes[0]]` — the root header updates every frame
// while all other reachable slots stay stable, letting the CPU-side builder
// serve incremental edits without rewriting the buffer. Root bounds are [0,1]^3.
//
// Traversal: explicit stack with depth capped at MAX_DEPTH (20 levels → 1M^3).
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
@group(0) @binding(2) var<storage, read> palette: array<vec4<f32>>;

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

// Material color palette. Input is the packed 16-bit cell word:
//   bits 15..6 material_id (10 bits)
//   bits  5..0 metadata    ( 6 bits)
// We index by material_id; metadata is reserved for future per-cell variation.
//
// DRIFT GUARD: the shift amount `6u` below mirrors `Cell::METADATA_BITS`
// in src/octree/node.rs. If that constant ever changes, update this
// shader accordingly. There is a pinning
// test in src/render/mod.rs (test `wgsl_metadata_shift_matches_rust`).
fn material_color(packed: u32) -> vec3<f32> {
    let mat_id = packed >> 6u;
    // Look up from GPU palette buffer uploaded by Renderer::upload_palette
    // from MaterialRegistry::color_palette_rgba(). No more hardcoded switch —
    // palette is always in sync with the Rust registry (hash-thing-5bb.7).
    if mat_id < arrayLength(&palette) {
        return palette[mat_id].xyz;
    }
    return vec3<f32>(0.6, 0.7, 0.8); // fallback for out-of-range mat_id
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
//
// hash-thing-6hd: uses STRICT `>` for the HIGH half with `rd >= 0` as a
// tiebreaker on exact-midpoint positions. The pre-6hd code used `>=` which
// biased midpoint-exact positions to the HIGH half regardless of ray
// direction; on simultaneous two-axis exits, the step-past nudge on a
// non-dominant axis can underflow to sub-ULP (dt * |rd_axis| < ULP(pos)),
// leaving pos EXACTLY on the boundary. The `>=` bias then picked the
// wrong sibling whenever `rd_axis < 0`. Codex reproducer (float32):
// rd = (1.0, -1e-6, 0.0). Post-step pos.x = 0.50000995, pos.y = 0.5
// (unchanged by the sub-ULP nudge on y), pos.z = 0.25 — old code returned
// oct 3 (x-HIGH, y-HIGH), physically correct is oct 1 (x-HIGH, y-LOW)
// because rd.y < 0.
//
// Zero-rd convention: when rd.axis == 0.0, the tiebreak resolves to HIGH
// (the ray can't cross the midpoint on that axis, so the choice is
// physically inert). Must match the CPU oracle in svdag.rs.
fn octant_of(pos: vec3<f32>, rd: vec3<f32>, node_min: vec3<f32>, half: f32) -> u32 {
    var idx: u32 = 0u;
    let mid = node_min + vec3<f32>(half);
    if pos.x > mid.x || (pos.x == mid.x && rd.x >= 0.0) { idx |= 1u; }
    if pos.y > mid.y || (pos.y == mid.y && rd.y >= 0.0) { idx |= 2u; }
    if pos.z > mid.z || (pos.z == mid.z && rd.z >= 0.0) { idx |= 4u; }
    return idx;
}

const MAX_DEPTH: u32 = 20u;
const LEAF_BIT: u32 = 0x80000000u;
// Integer DDA (hash-thing-pck): leaf-resolution grid.
const RESOLUTION: u32 = 1u << MAX_DEPTH;
const INV_RES: f32 = 1.0 / f32(RESOLUTION);

// hash-thing-2w5: step budget scales with root_level so deep scenes don't
// silently black out on long sparse traversals. Must stay in lockstep with
// `render::svdag::tests::step_budget` on the CPU replica.
//   MIN_STEP_BUDGET = 1024  (floor for shallow demo worlds)
//   STEP_BUDGET_FUDGE = 8   (per-cell multiplier above the 3*root_side
//                            diagonal worst case)
const MIN_STEP_BUDGET: u32 = 1024u;
const STEP_BUDGET_FUDGE: u32 = 8u;

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
    // Laine-Karras Stage 2: mirror ray so rd is componentwise non-negative.
    // Octant child indices are XORed with mirror_mask during DAG lookup.
    let neg = rd < vec3<f32>(0.0);
    let mirror_mask = select(0u, 1u, neg.x) | select(0u, 2u, neg.y) | select(0u, 4u, neg.z);
    let rd_m = abs(rd);
    let ro_m = select(ro, vec3<f32>(1.0) - ro, neg);

    // Guard zero ray-direction components (hash-thing-5bb.8).
    let rd_eps = vec3<f32>(1e-30);
    let safe_rd = max(rd_m, rd_eps);
    let inv_rd = 1.0 / safe_rd;

    // Per-ray step budget (hash-thing-2w5). Derived from u.params.x which the
    // host sets to `1 << dag.root_level` cells per side. Must stay in
    // lockstep with `cpu_trace::step_budget` in svdag.rs — the clamp at
    // `1 << 29` mirrors the CPU replica's `root_level.min(30)` +
    // `saturating_mul` guard so a pathological root_level can't overflow
    // `root_side * STEP_BUDGET_FUDGE` silently in u32. At the upper clamp
    // (root_side = 2^29), `* 8 = 2^32` would wrap — so the clamp is one
    // notch tighter than the CPU to leave headroom in u32.
    let root_side_raw = u32(u.params.x);
    let root_side = min(root_side_raw, 1u << 28u);
    let max_steps = max(MIN_STEP_BUDGET, root_side * STEP_BUDGET_FUDGE);

    // Intersect mirrored ray with root cube [0,1]^3
    let root_hit = intersect_aabb(ro_m, inv_rd, vec3<f32>(0.0), vec3<f32>(1.0));
    if root_hit.x > root_hit.y || root_hit.y < 0.0 {
        return vec4<f32>(0.0);
    }

    // hash-thing-27m v2: clamp ro to the root-cube entry (Laine-Karras).
    let entry = max(root_hit.x, 0.0);
    let ro_local = ro_m + rd_m * entry;
    let t_exit = root_hit.y - entry;

    // Integer DDA state (hash-thing-pck).
    var int_pos: vec3<u32> = vec3<u32>(
        u32(clamp(floor(ro_local.x * f32(RESOLUTION)), 0.0, f32(RESOLUTION - 1u))),
        u32(clamp(floor(ro_local.y * f32(RESOLUTION)), 0.0, f32(RESOLUTION - 1u))),
        u32(clamp(floor(ro_local.z * f32(RESOLUTION)), 0.0, f32(RESOLUTION - 1u))),
    );

    // Stack of (node_offset, node_min, half_size)
    var stack_node: array<u32, MAX_DEPTH>;
    var stack_min: array<vec3<f32>, MAX_DEPTH>;
    var stack_half: array<f32, MAX_DEPTH>;
    var depth: u32 = 0u;

    // Slot 0 of the buffer holds the current root offset; real slots start at 1.
    stack_node[0] = dag_nodes[0];
    stack_min[0] = vec3<f32>(0.0);
    stack_half[0] = 0.5;

    // Local-frame starting t — just past the root entry. `ro_local`
    // already sits on the root face, so we nudge forward by EPS.
    var t = 1e-5;

    for (var step = 0u; step < max_steps; step = step + 1u) {
        // Normal miss path: walked past the root exit cleanly. Must be an
        // explicit background return, NOT a `break` — the post-loop
        // fall-through is now the hash-thing-2w5 magenta exhaustion sentinel,
        // so any `break` here would render every clean root-exit as magenta.
        if t > t_exit { return vec4<f32>(0.0); }
        // Integer DDA (hash-thing-pck): pos from integer cell center.
        let pos = (vec3<f32>(int_pos) + 0.5) * INV_RES;

        // If the current top-of-stack no longer contains `pos`, pop.
        // NOTE: STRICT bounds (no EPS slack). Integer DDA (hash-thing-pck)
        // derives t from exact cell boundaries, so pos = ro_local + rd_m * t
        // always advances strictly past the octant boundary.
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
            let oct = octant_of(pos, rd_m, node_min, half);
            // Stage 2: XOR to un-mirror the octant for DAG lookup.
            let child_slot = dag_nodes[node_offset + 1u + (oct ^ mirror_mask)];

            if (child_slot & LEAF_BIT) != 0u {
                let mat = child_slot & 0xFFFFu;
                if mat > 0u {
                    // HIT — analytical face normal from the leaf AABB.
                    //
                    // hash-thing-rv4: pre-rv4 the shade was
                    //   `0.35 + 0.65 * abs(rd.y)`
                    // which depends only on ray direction, not surface
                    // orientation — every cube face got the same shade,
                    // collapsing all depth cues. Edward: "it's like weirdly
                    // flat." This block derives the true surface normal
                    // from which axis of the leaf AABB the ray entered
                    // through (textbook voxel raycaster normal), then
                    // applies Lambertian shading.
                    let leaf_min = vec3<f32>(
                        node_min.x + f32(oct & 1u) * half,
                        node_min.y + f32((oct >> 1u) & 1u) * half,
                        node_min.z + f32((oct >> 2u) & 1u) * half,
                    );
                    let leaf_max = leaf_min + vec3<f32>(half);
                    // Per-axis slab entry times. `tmin_v` is the max-across
                    // axes at entry, and the axis holding that max is the
                    // last face the ray crossed to get inside the voxel.
                    //
                    // hash-thing-2nd: inside-leaf fallback. When the ray
                    // origin sits inside the filled voxel, every `tmin_v`
                    // component is negative — the entry-face picker then
                    // returns the nearest back face and the normal flips
                    // inward, shading the voxel as if lit from behind. The
                    // orbit camera reaches this case on deep zoom. Fallback:
                    // pick the nearest exit face (argmin of `tmax_v`) and
                    // flip the sign so the normal points in the direction
                    // the ray is heading — the "about to emerge" convention.
                    // The outer cascade uses `>=` on both comparators so
                    // entry-face ties break identically to the CPU oracle
                    // in svdag.rs — do NOT flip to `>`. The inner (inside)
                    // cascade uses `<=` for the same reason: CPU/GPU must
                    // break exit-face ties identically.
                    let lt1 = (leaf_min - ro_local) * inv_rd;
                    let lt2 = (leaf_max - ro_local) * inv_rd;
                    let tmin_v = min(lt1, lt2);
                    let tmax_v = max(lt1, lt2);
                    var normal = vec3<f32>(0.0);
                    let inside = tmin_v.x < 0.0 && tmin_v.y < 0.0 && tmin_v.z < 0.0;
                    if inside {
                        if tmax_v.x <= tmax_v.y && tmax_v.x <= tmax_v.z {
                            normal = vec3<f32>(sign(rd.x), 0.0, 0.0);
                        } else if tmax_v.y <= tmax_v.z {
                            normal = vec3<f32>(0.0, sign(rd.y), 0.0);
                        } else {
                            normal = vec3<f32>(0.0, 0.0, sign(rd.z));
                        }
                    } else if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {
                        normal = vec3<f32>(-sign(rd.x), 0.0, 0.0);
                    } else if tmin_v.y >= tmin_v.z {
                        normal = vec3<f32>(0.0, -sign(rd.y), 0.0);
                    } else {
                        normal = vec3<f32>(0.0, 0.0, -sign(rd.z));
                    }
                    // Lambertian + ambient + fog.
                    let base = material_color(mat);
                    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
                    let diffuse = max(dot(normal, light_dir), 0.0);
                    let ambient = 0.3;
                    let lit = base * (ambient + diffuse * 0.7);
                    let t_world = t + entry;
                    let fog = exp(-t_world * 1.5);
                    let fog_color = vec3<f32>(0.55, 0.70, 0.90);
                    return vec4<f32>(mix(fog_color, lit, fog), 1.0);
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

        // Step past the current (empty) octant.
        // hash-thing-pck: integer DDA. Advance int_pos on exit axis, then
        // derive t as max of all three axes' boundary times.
        let node_min = stack_min[depth];
        let half = stack_half[depth];
        let oct = octant_of(pos, rd_m, node_min, half);
        let child_min = vec3<f32>(
            node_min.x + f32(oct & 1u) * half,
            node_min.y + f32((oct >> 1u) & 1u) * half,
            node_min.z + f32((oct >> 2u) & 1u) * half,
        );
        let child_max = child_min + vec3<f32>(half);
        // Stage 2: rd_m is all non-negative, so t2 > t1 on every axis.
        let t2 = (child_max - ro_local) * inv_rd;
        // Find exit axis (smallest exit time).
        var exit_axis: u32 = 0u;
        if t2.y < t2.x {
            exit_axis = 1u;
        }
        if t2.z < select(t2.x, t2.y, exit_axis == 1u) {
            exit_axis = 2u;
        }
        // Integer DDA: advance int_pos on exit axis.
        let step_cells = 1u << (MAX_DEPTH - 1u - depth);
        let cmin_exit = select(select(child_min.x, child_min.y, exit_axis == 1u), child_min.z, exit_axis == 2u);
        let octant_base = u32(cmin_exit * f32(RESOLUTION));
        let boundary_cell = octant_base + step_cells;
        if boundary_cell >= RESOLUTION {
            return vec4<f32>(0.0); // exited volume
        }
        if exit_axis == 0u { int_pos.x = boundary_cell; }
        else if exit_axis == 1u { int_pos.y = boundary_cell; }
        else { int_pos.z = boundary_cell; }
        // Derive t from exact exit-axis boundary. Monotonicity
        // guaranteed: if boundary t doesn't advance (rare corner case),
        // nudge by 1 ULP via bitcast. Safe because pos comes from int_pos.
        let boundary = f32(boundary_cell) * INV_RES;
        let ro_exit = select(select(ro_local.x, ro_local.y, exit_axis == 1u), ro_local.z, exit_axis == 2u);
        let inv_rd_exit = select(select(inv_rd.x, inv_rd.y, exit_axis == 1u), inv_rd.z, exit_axis == 2u);
        let t_boundary = (boundary - ro_exit) * inv_rd_exit;
        if t_boundary > t {
            t = t_boundary;
        } else {
            t = bitcast<f32>(bitcast<u32>(t) + 1u);
        }
    }

    // hash-thing-2w5: post-loop fall-through means we blew the step budget
    // while still inside the root cube (neither hit nor exited). Pre-2w5
    // this returned a silent black pixel indistinguishable from the real
    // background — budget exhaustion would just visually black out deep
    // worlds. Now we surface it as a magenta sentinel so the failure mode
    // is impossible to miss on screen. The CPU replica sets
    // `TraceResult.exhausted = true` for the same condition so tests can
    // assert on it directly.
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
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

    // Sky gradient: lighter blue at top, desaturated at horizon.
    let t = in.uv.y * 0.5 + 0.5;
    let sky_top = vec3<f32>(0.35, 0.55, 0.90);
    let sky_bot = vec3<f32>(0.65, 0.78, 0.92);
    let bg = mix(sky_bot, sky_top, t);
    return vec4<f32>(bg, 1.0);
}
