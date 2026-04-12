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
}
