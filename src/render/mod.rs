mod renderer;
mod svdag;

pub use renderer::{FrameOutcome, RenderMode, Renderer};
pub use svdag::Svdag;

#[cfg(test)]
mod wgsl_drift_guard {
    //! Pins the Rust `Cell::METADATA_BITS` constant to the hardcoded
    //! `>> 6u` shifts in both raycast shaders. If this test fails, update
    //! `METADATA_BITS` in src/octree/node.rs AND both shader files in the
    //! same change. See comments next to `material_color` in each shader.
    use crate::octree::Cell;

    const RAYCAST_WGSL: &str = include_str!("raycast.wgsl");
    const SVDAG_RAYCAST_WGSL: &str = include_str!("svdag_raycast.wgsl");

    #[test]
    fn wgsl_metadata_shift_matches_rust() {
        let expected = format!("packed >> {}u", Cell::METADATA_BITS);
        assert!(
            RAYCAST_WGSL.contains(&expected),
            "raycast.wgsl must contain `{expected}` — Cell::METADATA_BITS \
             drifted from the hardcoded shift. Update the shader."
        );
        assert!(
            SVDAG_RAYCAST_WGSL.contains(&expected),
            "svdag_raycast.wgsl must contain `{expected}` — \
             Cell::METADATA_BITS drifted from the hardcoded shift. \
             Update the shader."
        );
    }

    /// hash-thing-rv4 retroactive scout: pins the entry-face normal cascade
    /// in the SVDAG shader to the CPU oracle in `src/render/svdag.rs`.
    /// The rv4 fix added analytical face normals via `argmax(tmin_v)`
    /// with `>=` cascade ties (x then y then z); the comparator shape
    /// AND the `-sign(rd.*)` flip direction must stay byte-equivalent
    /// with the CPU oracle. CPU drift is caught by
    /// `leaf_hit_normal_corner_tiebreak_is_unit_face`; this guard fills
    /// the symmetric shader-only drift gap. Filed as scout work after
    /// hash-thing-2nd witnessed the same gap pattern twice (6hd + 2nd).
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

    /// hash-thing-2nd: pins the inside-leaf face-normal fallback in the
    /// SVDAG shader to the CPU oracle in `src/render/svdag.rs`. When the
    /// ray origin sits inside a filled voxel, the entry-face picker
    /// (`argmax(tmin_v)`) returns an inward-facing normal (all `tmin_v`
    /// components are negative → argmax is the least-negative axis → the
    /// nearest back face). Both sides must detect this via
    /// `all(tmin_v < 0)` and fall back to the nearest exit face
    /// (`argmin(tmax_v)`) with `+sign(rd)` on that axis. Reachable via
    /// deep-zoom orbit camera. This guard catches the case where a
    /// future edit drifts ONLY the shader.
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

    /// hash-thing-6hd: pins the `octant_of` midpoint tiebreak rule in the
    /// SVDAG shader to the CPU oracle in `src/render/svdag.rs`. Both sides
    /// must use strict `>` with `rd >= 0.0` as the tiebreaker on exact
    /// midpoint matches, otherwise two-axis tied exits pick the wrong
    /// sibling whenever `rd_axis < 0` (see Codex reproducer in the bead
    /// description). Unit tests cover the CPU side directly; this guard
    /// catches the case where a future edit drifts ONLY the shader and
    /// the existing parity claim silently breaks.
    #[test]
    fn wgsl_svdag_octant_of_tiebreak_matches_rust() {
        // Exact strings the shader must contain — comparator shape AND
        // rd tiebreak direction AND bit assignment. Any drift on any of
        // the three axes fails the test with the offending line named.
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

    /// hash-thing-2w5 retroactive scout: pins the step-budget constants in
    /// the SVDAG shader to the CPU oracle in `src/render/svdag.rs`. Both
    /// sides must agree on `MIN_STEP_BUDGET` and `STEP_BUDGET_FUDGE` so
    /// that exhaustion behavior (the magenta sentinel) fires at the same
    /// step count. CPU drift is caught by `step_budget_respects_floor_and_fudge`;
    /// this guard fills the symmetric shader-only drift gap.
    #[test]
    fn wgsl_svdag_step_budget_constants_match_rust() {
        use crate::render::svdag::cpu_trace;

        let expected_min = format!(
            "const MIN_STEP_BUDGET: u32 = {}u;",
            cpu_trace::MIN_STEP_BUDGET
        );
        let expected_fudge = format!(
            "const STEP_BUDGET_FUDGE: u32 = {}u;",
            cpu_trace::STEP_BUDGET_FUDGE
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
