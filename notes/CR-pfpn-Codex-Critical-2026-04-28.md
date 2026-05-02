## Critical Code Review: hash-thing-pfpn

- **Date:** 2026-04-28
- **Model:** Codex (GPT-5)
- **Branch:** `flint/hash-thing-pfpn-narrow`
- **Latest Commit:** `8127465`
- **Scope:** narrow static GPU-class default only
- **Validation:** `cargo test -p ht-render --lib` passed: 111 passed, 1 ignored

1. **BLOCKER - Low-end/mobile dGPUs are silently promoted to RTX-3060-class defaults.**
   `crates/ht-render/src/renderer.rs:614` maps every `wgpu::DeviceType::DiscreteGpu` to `(DiscreteDgpu, 1.0)`. The cited audience spec defends `RTX 3060+ -> 1.0`, and `docs/perf/render-perf-direction.md` names the reference non-Mac target as RTX 3060-equivalent or Iris Xe-equivalent, not all discrete adapters. Real adapters like GT 1030, MX-series, GTX 1050 mobile, and some low-power laptop dGPUs report `DiscreteGpu` but are weaker than, or close to, integrated-class hardware. At 1440p / 256^3, this changes the default from the old pixel budget's ~0.74 to native 1.0, about 1.8x the rendered pixels, and the dynamic ramp is explicitly deferred. The only discrete fixture is `NVIDIA GeForce RTX 3060`, so tests do not defend the lower boundary. Fix by whitelisting known median+ families, adding a lower/unknown dGPU bucket, or falling weak/unknown discrete names back to the conservative pixel/integrated policy.

2. **IMPORTANT - `EnvOverride` reports the wrong "would have been" default.**
   `crates/ht-render/src/renderer.rs:857-860` says `auto-pick would have been ...` but computes it with raw `auto_render_scale`, not the new class-aware default. On an RTX 3060 at 1440p / 256^3 the actual no-env default is 1.0, while this log reports ~0.74. This was explicitly flagged in the plan-review pack and remains unaddressed. Either reuse the class-aware decision for diagnostics or rename the field to `pixel-budget pick would have been`.

3. **IMPORTANT - Invalid env fallback hides which auto policy actually supplied the scale.**
   `crates/ht-render/src/renderer.rs:697` returns `auto_pair.0` with source `EnvInvalidFallback`. With `HASH_THING_RENDER_SCALE=garbage` on a classified M2, the actual scale is the GPU-class 0.5, but `crates/ht-render/src/renderer.rs:884-887` only says `auto-picked 0.500 instead`. That loses the distinction between class default and pixel fallback precisely in the typo-debug path. Add an invalid-env + classified-adapter test and either log the fallback kind or return a richer decision (`invalid_env=true`, `auto_source=...`) instead of collapsing the source.

4. **IMPORTANT - Apple classification is too dependent on `IntegratedGpu` plus exact friendly names.**
   `crates/ht-render/src/renderer.rs:597-615` only runs the Apple ladder inside `DeviceType::IntegratedGpu`, despite the nearby comment saying older paths may have an empty/mis-bucketed device type. An adapter named `Apple M2` with `DeviceType::Other` falls to the pixel-budget path; on the documented 2940x1782 / 256^3 case that is ~0.62, not the spec's 0.5. A sanitized WebGPU-style name such as `Apple GPU` under `IntegratedGpu` becomes `IntegratedNonApple` at 0.7. Add tests for `Apple M2 + Other` and `Apple GPU + IntegratedGpu`, then either handle Apple vendor/name before device-type bucketing or explicitly document those platforms as unsupported fallback cases.

5. **IMPORTANT - The tests mostly pin happy-path implementation, not the contract boundaries.**
   The new suite verifies M1/M2/M3/M4 friendly names, one RTX 3060, and unknown `Other`. It does not cover low-end discrete names, Apple mis-bucketing, future-name boundaries like `Apple M10`, or the two log-contract regressions above. Those are the exact places where real adapters can silently fall through to the wrong default.
