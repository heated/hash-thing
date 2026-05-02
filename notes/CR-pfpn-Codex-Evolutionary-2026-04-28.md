## Evolutionary Code Review
- **Date:** 2026-04-28
- **Model:** Codex / GPT-5
- **Branch:** flint/hash-thing-pfpn-narrow
- **Latest Commit:** 8127465
- **Lens:** Evolutionary, narrow static slice
---

Static correctness looks sound: the class defaults implement the scoped spec, `Option<&AdapterInfo>` is a reasonable seam for unit tests / wasm-ish callers, and splitting the dynamic ramp into `uzrr` avoided most of the plan-review risk around `RampState` lifecycle. I found no BLOCKERs.

Validation run:
- `cargo test -p ht-render resolved_` -> 15 passed
- `cargo test -p ht-render classify_` -> 8 passed

1. **IMPORTANT - Initial-scale decision metadata is discarded too early, which will make `uzrr` re-plumb source/class data.**  
   In `resolved_render_scale`, the class is immediately dropped via `.map(|(_class, pick)| pick.clamp(...))` at `crates/ht-render/src/renderer.rs:678`, and `AutoPickedByGpuClass` carries no payload. The call site then re-runs `classify_gpu(&adapter_info)` only for logging at `renderer.rs:870`. That is workable for this narrow slice, but it is setting up against the deferred ramp: `uzrr` will need to log/attribute post-ramp steps to the initial decision, and currently the returned decision only says "class path" or "invalid env fallback", not which class or which auto fallback. The sharpest case is invalid env + known adapter at `renderer.rs:693`: it returns the class-picked scale but `RenderScaleSource::EnvInvalidFallback`, so the scale provenance and override provenance are conflated. Tests cover valid env with adapter (`renderer.rs:3062`) but not invalid env with adapter. Before or during `uzrr`, turn this into an `InitialScaleDecision` (or make `RenderScaleSource` carry `AutoScaleSource::{GpuClass(GpuClass), PixelBudget}` plus invalid-env metadata) so ramp enablement and ramp-step logs do not have to infer/reclassify state.

2. **IMPORTANT - The classifier is branch-shaped, not table-shaped, so each new bucket will touch code and tests in several places.**  
   `classify_gpu` is an `if/else` ladder over names and `DeviceType` at `renderer.rs:594`, and the tests mirror each branch with one-off test functions at `renderer.rs:2933` through `renderer.rs:3007`. That is fine for five buckets, but the prompt explicitly expects the table to grow with new GPU families and real-user telemetry. Adding, say, "Apple M5/M6 = 0.8" or "old/mobile dGPU = 0.7" is not a one-line table edit: it requires a new enum variant, matcher surgery, new bespoke tests, and probably log/test updates. A small rule table like `{ class, scale, device_type, name_prefixes/name_contains }` plus a parameterized fixture test would make additions mostly data edits while preserving the same public `classify_gpu(&AdapterInfo) -> Option<...>` API. This also gives future telemetry reports an obvious place to paste new observed adapter strings.

3. **NIT - The class-pick log is useful, but not yet a self-contained calibration record.**  
   The new `AutoPickedByGpuClass` log at `renderer.rs:874` includes `name`, `device_type`, `class`, `class_pick`, and `pixel_pick`, which is enough to debug the local decision. For the six-month telemetry loop, it should also include the inputs that make `pixel_pick` interpretable: `volume_size`, physical `width x height`, and ideally `backend`, `driver_info`, numeric `vendor/device`, or a stable reference to the adjacent adapter line. Otherwise future table-tuning from pasted logs depends on correlating multiple startup lines and may lose the display/world context that explains why a class default was or was not corrected later by the ramp.

Overall: this is a good static split, but `uzrr` should not build on the current `(f32, RenderScaleSource)` tuple as-is. Promote the startup result to a decision object before the ramp starts accumulating logic around it.
