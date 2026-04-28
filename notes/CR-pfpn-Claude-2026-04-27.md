# CR pfpn — narrow scope (D1+D2+D7) — Claude

Reviewer: Claude Opus 4.7 (claude-opus-4-7)
Branch / commit: flint/hash-thing-pfpn-narrow @ 8127465
Scope: `crates/ht-render/src/renderer.rs` only (+303 / −25)
Spec defended: `docs/perf/audience-hw-distribution.md` v3.1; trident-flagged dGPU clamp dropped.
All 17 `resolved_*` + `classify_*` tests pass locally (`cargo test --package ht-render --lib`).

## Summary

Solid narrow-slice implementation. The unclamped class-default path is correctly enforced by both code and a dedicated regression test (`resolved_dgpu_class_default_not_clamped_by_pixel_budget`). Override precedence (env > CLI > class > pixel) is preserved. The `synthetic_adapter_info` fail-to-compile pattern for wgpu 29 fields (`device_pci_bus_id`, `subgroup_min_size`, `subgroup_max_size`, `transient_saves_memory`) is exactly what the trident pack asked for, and I verified those fields against `wgpu-types-29.0.1/src/adapter.rs`.

No BLOCKERs. Five IMPORTANT items, plus a handful of NITs.

---

## IMPORTANT

### I1. `EnvOverride` and (less critically) `EnvInvalidFallback` log lines now lie about the auto-pick

`renderer.rs:857-861` — the `EnvOverride` arm prints:

```
"render_scale={:.3} (HASH_THING_RENDER_SCALE override; auto-pick would have been {:.3})",
render_scale, auto_render_scale(physical_pixels, volume_size),
```

That second value is the **pixel-budget** pick, but the actual auto-pick that would have fired with the adapter known is the class default. Concrete cases:

- RTX 3060 @ 1440p, user sets `HASH_THING_RENDER_SCALE=0.5` to test something. Log says "auto-pick would have been ~0.74" — actually would have been **1.0**. User has no signal that they're leaving 35 % of pixels on the table.
- M2 + `HASH_THING_RENDER_SCALE=0.7`. Log says "auto-pick would have been 0.498" — actually would have been **0.5** (class default). Off by ~0.2 % so harmless here, but the principle is wrong everywhere.

This was Codex's plan-review Q-18 / Adversarial #28 ("existing env/CLI log lines that say 'auto would have picked' will continue reporting the old pixel-only auto-pick unless explicitly updated"). The implementation didn't update them. Fix: extract the auto-pick selection into a helper that returns `(scale, source, class_pick_opt, pixel_pick)` and let every log arm interpolate from one source of truth — same shape as the existing `AutoPickedByGpuClass` arm.

The same gap exists in `EnvInvalidFallback` at `renderer.rs:884` where `auto-picked {:.3} instead` doesn't say whether the value came from class or pixel — diagnostic ambiguity for the next seat tuning the class table.

### I2. Test gap: `EnvInvalidFallback` × adapter-known is uncovered

The 8 new tests cover env-valid, env-invalid-no-adapter, CLI, class-only, dgpu-no-clamp, unknown-device-type, and adapter-None — but **no test exercises `Some(invalid env)` + `None CLI` + `Some(known adapter)`**. The branch at `renderer.rs:697` returns `(auto_pair.0, EnvInvalidFallback)` where `auto_pair.0` is the class pick when adapter is known. That's the right value, but the contract isn't pinned. If a future refactor flips it to `auto_render_scale(physical_pixels, volume_size)` "for clarity", every M-series user with a typo'd env var silently drops to the pixel-budget pick. Two-line addition: M2 adapter + `Some("garbage")` + no CLI → source `EnvInvalidFallback`, value 0.5.

### I3. `DiscreteGpu → 1.0` over-buckets low-end and mobile dGPUs

`renderer.rs:614` maps **any** `DeviceType::DiscreteGpu` to 1.0. Spec is "RTX 3060+ → 1.0." Real-world false positives that get launched at 1.0:

- GT 1030 (~1.1 TFLOPS, ~0.4× M1 base)
- MX-150 / MX-250 mobile (~1.2–1.5 TFLOPS)
- GTX 950M / 960M laptops still common in second-hand markets

These users get a stuttery first-launch experience and have to discover `=` / `-` keys or `HASH_THING_RENDER_SCALE` to recover. The deferred ramp (`uzrr`) was the safety net that absorbed this; with the ramp out of scope, the trade-off is now bare. Two acceptable fixes:

- **Cheapest**: comment the trade-off in `GpuClass::DiscreteDgpu` doc and on the bead so the next seat (or the `uzrr` ramp) knows to address it. Don't expand the table — that's `uzrr`'s job.
- **Better**: vendor + name-substring check (NVIDIA + RTX/4060 Mobile/etc.) gates 1.0; otherwise fall through to a conservative 0.7 dGPU bucket. ~10 LOC, no dynamic state.

I'd accept the cheaper fix given the narrow scope, but it should be explicit.

### I4. `classify_gpu` comment promises a fallback the code does not implement

`renderer.rs:599-603` (in the `IntegratedGpu` arm) the comment reads:

> Apple-Silicon-shaped names sometimes come back through either the IntegratedGpu bucket (Metal) or **with empty device_type on older wgpu paths**. Match on the prefix first so the early-Apple buckets fire before the generic integrated fallback.

But the outer `match info.device_type` collapses `Cpu | VirtualGpu | Other => None` *before* the name is consulted. So an Apple-named adapter that returns `DeviceType::Other` (the comment's "older wgpu paths" case) silently drops to the pixel-budget path. Either reorder to name-first, or strike that sentence. I lean strike — wgpu 29 Metal reliably reports `IntegratedGpu` for Apple Silicon today.

### I5. Test gap for the `(invalid env, Some(cli), Some(adapter))` combo

Existing `resolved_render_scale_cli_used_when_env_invalid` (`renderer.rs:2904`) passes `None` for adapter. Adding `Some(M2 adapter)` to that case validates that CLI still wins over class default in the env-invalid path — the precedence chain's last untested cell.

---

## NIT

- **N1.** `name.contains("Apple M1")` matches hypothetical `"Apple M10"` / `"Apple M11"`. Plan R2 noted this as conservative; tighter would be `contains("Apple M1 ")` + `== "Apple M1"`. Low priority.
- **N2.** `classify_gpu` is called twice in `Renderer::new` (once in `resolved_render_scale`, once in the log arm with `.expect(…)`). Pure + cheap, but the `expect` re-establishes an invariant that was just satisfied. Threading `(class, class_pick, pixel_pick)` out of `resolved_render_scale` removes the duplicate call and the `expect`.
- **N3.** Two adjacent log lines (`renderer.rs:779` and `:875`) both print `name` and `device_type`. The class line could drop those and keep only `class={:?} class_pick={:.3} pixel_pick={:.3}`.
- **N4.** `pub(crate)` on `GpuClass` / `classify_gpu` — both only used inside `renderer.rs`. File-private (`fn classify_gpu`, `enum GpuClass` with no visibility) would be tighter; `pub(crate)` is fine if a sibling module is anticipated.
- **N5.** `synthetic_adapter_info` uses `Backend::Noop` — verified that variant exists in wgpu 29 (`wgpu-types-29.0.1`). ✓ no action.

---

## Trident plan-review carryovers — verified

- D5 `class_pick.min(pixel_pick)` clamp: **dropped**, `resolved_dgpu_class_default_not_clamped_by_pixel_budget` regression test in place. ✓
- `synthetic_adapter_info` constructs all wgpu 29 fields explicitly. ✓ (verified against `wgpu-types-29.0.1/src/adapter.rs`)
- Override precedence (env > CLI > class > pixel) preserved across all 17 tests. ✓
- Dynamic ramp (D3+D4) confirmed deferred to `uzrr`; no scope creep. ✓
- `Renderer::new` adapter-info relocation: nothing between old (`:1543`) and new (`:778`) position depended on `adapter_info`. Safe. ✓

---

## Recommendation

Ship after **I1** (log-line correction is the highest-leverage finding — Codex flagged this in the plan review and it slipped through), **I2** (one-test gap on the documented EnvInvalidFallback × class path), and a one-line comment on **I3** (acknowledge the GT 1030 / mobile-MX trade-off so `uzrr` inherits it cleanly). **I4** is a one-sentence comment edit. **I5** is two lines. NITs optional.
