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
}
