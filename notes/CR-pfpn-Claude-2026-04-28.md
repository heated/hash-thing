## Code Review — hash-thing-pfpn (narrow scope)
- **Date:** 2026-04-28T00:00:00Z
- **Model:** Claude Opus 4.7 (claude-opus-4-7)
- **Branch:** flint/hash-thing-pfpn-narrow
- **Latest Commit:** 8127465
- **Bead:** hash-thing-pfpn

---

### Summary

Delivers what the narrowed plan promised: pure `classify_gpu`, `AutoPickedByGpuClass` source unclamped by pixel-budget, env/CLI precedence preserved, log line moved up. Both top trident plan-review blockers (D5 unclamped class default, wgpu-29 `AdapterInfo` field shape) are resolved with explicit tests. All 50 renderer-mod tests pass; build is green. Findings below are logging-clarity + edge-case coverage. No correctness blockers.

### Findings

1. **IMPORTANT — `EnvInvalidFallback` silently consumes the GPU-class signal.**
   `renderer.rs:697` returns `auto_pair.0` (the *class* scale when adapter is classifiable) tagged as `EnvInvalidFallback`. The log at `:884` prints `auto-picked 0.500 instead`, hiding whether the value came from class default or pixel-budget. A user typo-debugging `HASH_THING_RENDER_SCALE` on an Apple M1 can't tell which path rescued them — matters because uzrr's dynamic ramp will start from one or the other. Fix: preserve `AutoPickedByGpuClass` as the source on the invalid-env path and emit a separate `warn!` for the typo (mirrors the `CliOverride` pattern at `:893–:897`); keeps the source enum semantically truthful. Test gap: no case covers `Some("garbage") + Some(adapter)` — add one.

2. **IMPORTANT — `classify_gpu` is called twice on the auto-by-class path.**
   `renderer.rs:679` runs it inside `resolved_render_scale`, then `:871` re-runs with `.expect(...)` to recover `class` for logging. Cheap, but the `.expect` is an invariant in two places: if anyone refactors `resolved_render_scale` to short-circuit before `classify_gpu`, the panic message lies. Cleaner: have `resolved_render_scale` return `Option<GpuClass>` alongside the source.

3. **IMPORTANT — Apple-Silicon adapters reporting `DeviceType::Other` fall through silently.**
   The comment at `:599–:603` acknowledges Apple names "sometimes come back ... with empty device_type on older wgpu paths," but the match arm only handles `IntegratedGpu`. An "Apple M2" name with `DeviceType::Other` returns `None`, dropping to pixel-budget. wgpu 29 on current macOS is fine; MoltenVK / Vulkan backends can mis-categorize. Fix: one extra arm — `DeviceType::Other if name.starts_with("Apple ") => /* same Apple ladder */` — plus a mirrored test. If left as-is, document the fall-through explicitly so the next seat tuning the table doesn't waste an hour.

4. **NIT — Wasm/WebGPU sanitized names land in `IntegratedNonApple`@0.7.**
   Chrome WebGPU returns `name = "GPU 0"` with `device_type = IntegratedGpu`, so a wasm session on an RTX 4090 classifies as `IntegratedNonApple` at 0.7 — wrong but conservative. Fine for narrow scope; one-line mention in `GpuClass` doc-comment so uzrr doesn't inherit the surprise.

5. **NIT — `pub(crate)` asymmetry.**
   `auto_render_scale` / `target_pixels_for_volume` are private `fn`; `classify_gpu` / `GpuClass` are `pub(crate)` but unused outside `renderer.rs` (verified). Lower to private, or pick one convention.

6. **NIT — Test coverage gaps.**
   - Missing: `Some(invalid env) + Some(classifiable adapter)` (re #1).
   - Missing: `DeviceType::Other` + Apple name (re #3).
   - Present and good: dGPU-not-clamped, adapter=None, env-wins-over-class, cli-wins-over-class, all five class buckets, `Cpu/VirtualGpu/Other → None`.

7. **NIT — `RTX 3060+` ≠ "all DiscreteGpu".**
   Implementation picks 1.0 for *every* `DiscreteGpu` including older Quadro NVS / GTX 750 Ti class cards. Out of scope for narrow pfpn (uzrr's ramp catches this), but worth a one-liner in the bead-close so uzrr's scope is unambiguous.

### Migration safety

Moving `adapter.get_info()` is safe: `&self` method, no side effects, nothing between new/old call sites mutates adapter state. Old log line removed cleanly. Intervening callers (`adapter.features()`, `request_device(...)`, `surface.get_capabilities(&adapter)`) don't care about ordering vs. `get_info()`.

### Verdict

Ship-ready after addressing #1 (user-facing logging gap). #2–3 worth fixing in the same PR; #4–7 fine as follow-ups or bead-close notes.
