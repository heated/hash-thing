pub(crate) mod font;
mod renderer;
mod svdag;

pub use renderer::{FrameOutcome, Renderer};
pub use svdag::Svdag;

#[cfg(test)]
mod wgsl_drift_guard {
    //! Pins the Rust `Cell::METADATA_BITS` constant to the hardcoded
    //! `>> 6u` shifts in both raycast shaders. If this test fails, update
    //! `METADATA_BITS` in src/octree/node.rs AND both shader files in the
    //! same change. See comments next to `material_color` in each shader.
    use ht_octree::Cell;

    const RENDERER_RS: &str = include_str!("renderer.rs");
    const SVDAG_RAYCAST_WGSL: &str = include_str!("svdag_raycast.wgsl");
    const PARTICLE_WGSL: &str = include_str!("particle.wgsl");

    #[test]
    fn wgsl_metadata_shift_matches_rust() {
        let expected = format!("packed >> {}u", Cell::METADATA_BITS);
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected),
            "svdag_raycast.wgsl must contain `{expected}` — \
             Cell::METADATA_BITS drifted from the hardcoded shift. \
             Update the shader."
        );
    }

    #[test]
    fn wgsl_svdag_entry_face_normal_cascade_matches_rust() {
        let expected_lines = [
            "} else if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {",
            "normal = vec3<f32>(-sign(rd.x), 0.0, 0.0);",
            "} else if tmin_v.y >= tmin_v.z {",
            "normal = vec3<f32>(0.0, -sign(rd.y), 0.0);",
            "normal = vec3<f32>(0.0, 0.0, -sign(rd.z));",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the entry-face normal cascade drifted from the CPU \
                 oracle in svdag.rs::cpu_trace::raycast_with_budget. \
                 Both sides must use `>=` ties on `argmax(tmin_v)` with \
                 `-sign(rd.*)` on the entry axis (hash-thing-rv4). \
                 Update whichever side is wrong so they stay byte-equivalent."
            );
        }
    }

    #[test]
    fn wgsl_svdag_inside_leaf_normal_fallback_matches_rust() {
        let expected_lines = [
            "let tmax_v = max(lt1, lt2);",
            "let inside = tmin_v.x < 0.0 && tmin_v.y < 0.0 && tmin_v.z < 0.0;",
            "if tmax_v.x <= tmax_v.y && tmax_v.x <= tmax_v.z {",
            "normal = vec3<f32>(sign(rd.x), 0.0, 0.0);",
            "} else if tmax_v.y <= tmax_v.z {",
            "normal = vec3<f32>(0.0, sign(rd.y), 0.0);",
            "normal = vec3<f32>(0.0, 0.0, sign(rd.z));",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the inside-leaf exit-face fallback drifted from the CPU \
                 oracle in svdag.rs::cpu_trace::raycast_with_budget. Both \
                 sides must detect inside-leaf via `all(tmin_v < 0)` and \
                 pick the nearest exit face with `sign(rd)` on that axis \
                 (hash-thing-2nd). Update whichever side is wrong so they \
                 stay byte-equivalent."
            );
        }
    }

    #[test]
    fn wgsl_svdag_inside_lod_node_normal_fallback_matches_rust() {
        let expected_lines = [
            "let ntmax_v = max(nt1, nt2);",
            "let inside_node = ntmin_v.x < 0.0 && ntmin_v.y < 0.0 && ntmin_v.z < 0.0;",
            "if ntmax_v.x <= ntmax_v.y && ntmax_v.x <= ntmax_v.z {",
            "normal = vec3<f32>(sign(rd.x), 0.0, 0.0);",
            "} else if ntmax_v.y <= ntmax_v.z {",
            "normal = vec3<f32>(0.0, sign(rd.y), 0.0);",
            "normal = vec3<f32>(0.0, 0.0, sign(rd.z));",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the representative-material LOD path must detect inside-node \
                 origins via `all(ntmin_v < 0)` and pick the nearest exit face \
                 with `sign(rd)` on that axis (hash-thing-p9dd). Update \
                 whichever side is wrong so CPU and WGSL stay aligned."
            );
        }
    }

    #[test]
    fn wgsl_leaf_shading_uses_surface_hit_position() {
        let expected_lines = [
            "let hit_t = select(",
            "ro_local + rd * max(hit_t, 0.0) - normal * (INV_RES * 0.25);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — leaf shading \
                 should sample the actual surface hit position (nudged inward), \
                 not the leaf center, or collapsed leaves turn into rectangular \
                 lighting artifacts."
            );
        }
    }

    #[test]
    fn wgsl_lod_shading_uses_surface_hit_position() {
        let expected_lines = [
            "let lod_hit_t = select(",
            "ro_local + rd * max(lod_hit_t, 0.0) - normal * (INV_RES * 0.25);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — representative-\
                 material LOD hits should shade the actual surface hit position \
                 (nudged inward), not the node center, or coarse interiors turn \
                 into rectangular ghost patches."
            );
        }
    }

    #[test]
    fn wgsl_hit_alpha_tracks_scene_distance() {
        let expected_lines = [
            "vec4<f32>(lit, max(entry + max(lod_hit_t, 0.0), 1e-4))",
            "vec4<f32>(lit, max(entry + max(hit_t, 0.0), 1e-4))",
            "final_color = vec4<f32>(bg, 0.0);",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` — the raycast \
                 output alpha now carries scene hit distance for overlay \
                 occlusion; background pixels must keep alpha 0."
            );
        }
    }

    #[test]
    fn wgsl_particles_sample_scene_depth_for_occlusion() {
        let expected_lines = [
            "@group(0) @binding(3) var t_scene: texture_2d<f32>;",
            "let scene = textureSample(t_scene, s_scene, in.screen_uv);",
            "if scene.a > 0.0 && scene.a + depth_epsilon < in.ray_t {",
        ];
        for expected in expected_lines {
            assert!(
                PARTICLE_WGSL.contains(expected),
                "particle.wgsl must contain `{expected}` — billboard overlays \
                 should sample the raycast texture's scene depth and discard \
                 particles hidden behind voxel geometry."
            );
        }
    }

    #[test]
    fn renderer_particle_bind_group_helper_covers_all_runtime_bindings() {
        let expected_lines = [
            "fn rebuild_particle_bind_group(&mut self) {",
            "binding: 0,",
            "binding: 1,",
            "binding: 2,",
            "binding: 3,",
            "binding: 4,",
            "resource: wgpu::BindingResource::TextureView(&self.raycast_texture_view),",
            "resource: wgpu::BindingResource::Sampler(&self.blit_sampler),",
        ];
        for expected in expected_lines {
            assert!(
                RENDERER_RS.contains(expected),
                "renderer.rs must contain `{expected}` — particle bind-group \
                 rebuilds should flow through the shared helper with the full \
                 live binding set, or layout drift will survive compile-time \
                 checks and fail only at runtime."
            );
        }
    }

    #[test]
    fn renderer_rebuild_call_sites_delegate_to_particle_helper() {
        let expected_lines = [
            "if self.particle_count > 0 {\n            self.rebuild_particle_bind_group();\n        }",
            "if self.particle_buffer.is_some() {\n            self.rebuild_particle_bind_group();\n        }",
            "self.rebuild_particle_bind_group();",
        ];
        for expected in expected_lines {
            assert!(
                RENDERER_RS.contains(expected),
                "renderer.rs must contain `{expected}` — particle bind-group \
                 refreshes on raycast-texture recreate, palette upload, and \
                 particle upload should all delegate through the shared helper \
                 instead of open-coding stale binding lists."
            );
        }
    }

    #[test]
    fn wgsl_svdag_octant_of_tiebreak_matches_rust() {
        let expected_lines = [
            "if pos.x > mid.x || (pos.x == mid.x && rd.x >= 0.0) { idx |= 1u; }",
            "if pos.y > mid.y || (pos.y == mid.y && rd.y >= 0.0) { idx |= 2u; }",
            "if pos.z > mid.z || (pos.z == mid.z && rd.z >= 0.0) { idx |= 4u; }",
        ];
        for expected in expected_lines {
            assert!(
                SVDAG_RAYCAST_WGSL.contains(expected),
                "svdag_raycast.wgsl must contain `{expected}` verbatim — \
                 the octant_of midpoint tiebreak drifted from the CPU \
                 oracle in svdag.rs::cpu_trace::octant_of. Both sides \
                 must use strict `>` with `rd >= 0.0` tiebreak on exact \
                 midpoint matches (hash-thing-6hd). Update whichever \
                 side is wrong so they stay byte-equivalent."
            );
        }
    }

    #[test]
    fn wgsl_material_palette_uses_buffer_lookup() {
        let expected = "palette[mat_id].xyz";
        assert!(
            SVDAG_RAYCAST_WGSL.contains(expected),
            "svdag_raycast.wgsl must contain `{expected}` — material_color \
             should read from the GPU palette buffer, not a hardcoded switch \
             (hash-thing-5bb.7)."
        );
    }

    #[test]
    fn wgsl_svdag_traversal_constants_match_rust() {
        use crate::svdag::cpu_trace;

        let expected_depth = format!("const MAX_DEPTH: u32 = {}u;", cpu_trace::MAX_DEPTH);
        let expected_min = format!(
            "const MIN_STEP_BUDGET: u32 = {}u;",
            cpu_trace::MIN_STEP_BUDGET
        );
        let expected_fudge = format!(
            "const STEP_BUDGET_FUDGE: u32 = {}u;",
            cpu_trace::STEP_BUDGET_FUDGE
        );

        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_depth),
            "svdag_raycast.wgsl must contain `{expected_depth}` — \
             MAX_DEPTH drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_min),
            "svdag_raycast.wgsl must contain `{expected_min}` — \
             MIN_STEP_BUDGET drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_fudge),
            "svdag_raycast.wgsl must contain `{expected_fudge}` — \
             STEP_BUDGET_FUDGE drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );

        let expected_stack = format!("const MAX_STACK: u32 = {}u;", cpu_trace::MAX_STACK);
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected_stack),
            "svdag_raycast.wgsl must contain `{expected_stack}` — \
             MAX_STACK drifted from the CPU oracle in \
             svdag.rs::cpu_trace. Update whichever side is wrong."
        );
    }
}
