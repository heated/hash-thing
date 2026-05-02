## Code Review
- **Date:** 2026-04-28T00:21:03Z
- **Model:** Codex / GPT-5
- **Branch:** flint/hash-thing-pfpn-narrow
- **Latest Commit:** 81274651f4a2
- **Story:** hash-thing-pfpn
---

No BLOCKER findings. The narrowed implementation keeps the risky dynamic ramp out, preserves override precedence, and the static class-default path is small and testable. `cargo test -p ht-render` passes: 111 passed, 1 ignored.

### Findings

1. **IMPORTANT - Env override diagnostics still report the old pixel-only auto pick.**  
   [crates/ht-render/src/renderer.rs](/Users/edward/projects/hash-thing/.claude/worktrees/nested-hopping-boot/crates/ht-render/src/renderer.rs:857) logs `auto-pick would have been ...` but computes that value with `auto_render_scale(physical_pixels, volume_size)` at line 860. After this change, the real no-env auto pick for a known adapter is class-aware. Example: an RTX 3060 at 1440p / 256^3 would auto-pick `1.0`, but the env-override log would report the pixel-budget value around `0.74`. This was explicitly flagged in the plan-review synthesis, and it will mislead the next tuning/debugging pass. Fix by sharing the class-aware auto calculation with the log path, or change the wording to say `pixel-budget pick would have been`.
   Confirmed against lines 849-861 and the new `AutoPickedByGpuClass` path at lines 870-882.

2. **IMPORTANT - `DiscreteGpu` over-buckets low-end/mobile discrete GPUs as RTX-3060-class.**  
   [crates/ht-render/src/renderer.rs](/Users/edward/projects/hash-thing/.claude/worktrees/nested-hopping-boot/crates/ht-render/src/renderer.rs:614) maps every `wgpu::DeviceType::DiscreteGpu` to `DiscreteDgpu, 1.0`. The cited audience spec defends `RTX 3060+ -> 1.0`, not necessarily GT 1030 / MX-series / GTX 1650-class cards. With the dynamic ramp deferred, those users will stay at native scale until they manually intervene. Either make the coarse policy explicit with a test fixture for a low-end discrete name, or refine the classifier so only known median+ discrete families get the 1.0 default and weaker/unknown dGPUs fall back lower.
   Confirmed: the only discrete fixture is `NVIDIA GeForce RTX 3060`; no lower-end discrete case covers the boundary.

3. **IMPORTANT - Missing coverage for invalid env + known adapter fallback.**  
   [crates/ht-render/src/renderer.rs](/Users/edward/projects/hash-thing/.claude/worktrees/nested-hopping-boot/crates/ht-render/src/renderer.rs:695) correctly appears to use `auto_pair.0` for `EnvInvalidFallback`, so `HASH_THING_RENDER_SCALE=garbage` with an Apple M2 should fall back to `0.5`, not the pixel-budget value. The tests cover invalid env with `adapter_info=None`, and valid env/CLI with an adapter, but not this combined branch. Add a test like `resolved_invalid_env_uses_class_default_when_adapter_known` to lock that contract.
   Confirmed by the tests around lines 2858-2870 and 3062-3078.

API shape looks acceptable for this slice: `Option<&AdapterInfo>` keeps pure tests cheap, and `pub(crate)` is not externally leaky. Moving `adapter.get_info()` earlier looks mechanically safe; the adapter log remains single-emission.
