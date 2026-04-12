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

    const SVDAG_RAYCAST_WGSL: &str = include_str!("svdag_raycast.wgsl");

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
    }
}
