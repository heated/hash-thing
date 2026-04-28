## Critical Code Review — hash-thing-pfpn (narrow scope)
- **Date:** 2026-04-28
- **Model:** Claude Opus 4.7 (claude-opus-4-7), critical lens
- **Branch:** flint/hash-thing-pfpn-narrow @ 8127465
- **Bead:** hash-thing-pfpn

This pass is adversarial. The non-critical Claude review on the same date called this ship-ready with no blockers; I disagree on at least three counts. The narrow scope claims "static GPU-class default for the whole v3.1 audience-distribution spec" — the implementation does not actually deliver that for several real-world configurations the spec covers, and three trident-plan-review findings the implementation didn't address are still wrong defaults today, not future concerns.

---

### BLOCKER 1 — `DiscreteGpu` blanket → 1.0 contradicts the spec for ~bottom-third of the dGPU audience.

`renderer.rs:614` maps **every** `DeviceType::DiscreteGpu` to 1.0. The spec says "RTX 3060+ → 1.0" — that is a substantially narrower set than "every discrete GPU on Steam". Real adapters that hit the wrong default *today*:

- GTX 1050 / 1050 Ti / GT 1030 (low-end Pascal — still common, 5+ years deployed)
- AMD RX 540 / 550 / 560 (low-end Polaris)
- Mobile MX150 / MX250 / MX350, Quadro M-series — laptop dGPUs explicitly weaker than RTX 3060
- Intel Arc A310 / A380 (entry Arc)
- Older eGPUs (HD 7000, R9 200)

These users start at `render_scale = 1.0` and get a stuttery first-second until the deferred ramp (uzrr) lands — but the static-pick PR claims to ship a usable default in the absence of the ramp. The trident plan-review flagged this **five times** across reviewers (synthesis-standard #6, synthesis-adversarial #11/A8/Q-14, adversarial-codex 27/56/87, standard-codex Q1) and the implementation added no guard. At minimum: clamp dGPU to `min(1.0, pixel_budget_pick)` *for unknown discrete only*, or split `DiscreteDgpu` into `DiscreteKnownHigh` (RTX 3060+, Radeon RX 6700+) vs `DiscreteUnknown` (conservative 0.7).

This is a BLOCKER because the bead's marquee acceptance criterion ("RTX 3060+ → 1.0") is implemented as a strictly weaker policy ("every dGPU → 1.0"), and the harm lands on the audience-distribution slice the bead is supposed to serve.

### BLOCKER 2 — Apple Silicon under Vulkan/MoltenVK gets `DiscreteDgpu = 1.0`.

`classify_gpu` matches on `device_type` first. The Apple name ladder lives **only** under the `IntegratedGpu` arm. If wgpu's Vulkan/MoltenVK path reports an M1/M2 adapter as `DiscreteGpu` (observed historically — wgpu issue tracker has multiple reports of MoltenVK mis-classifying Apple integrated GPUs), the name check never fires and an M1 user gets `render_scale = 1.0` instead of `0.5`. This *exactly inverts* the v3.1 spec for the bottom-quartile audience.

The classify_gpu doc-comment at `:599–:603` explicitly acknowledges "Apple-Silicon-shaped names sometimes come back through either the IntegratedGpu bucket (Metal) or with empty device_type on older wgpu paths" — but the code only handles `IntegratedGpu`. The comment promises a defense the code doesn't deliver. Fix: hoist `if name.starts_with("Apple ")` *outside* the device_type match (or duplicate the Apple ladder under `DiscreteGpu` and `Other`). No test covers any of these cross-bucket cases.

### BLOCKER 3 — `RenderScaleSource::EnvInvalidFallback` lies about its provenance.

`renderer.rs:697`: when env is invalid, no CLI, *and* adapter is classifiable, code returns `(auto_pair.0, EnvInvalidFallback)` — the *value* is the class pick (e.g. 0.5 on M1), but the source enum says "fell back to pixel-budget auto". This is the only signal uzrr's dynamic ramp will use to decide whether to engage. Tagging a class-default decision as `EnvInvalidFallback` will silently break uzrr's enable-rule. The log at `:884` compounds the problem: `auto-picked 0.500 instead` hides whether the rescue came from class default or pixel budget.

The trident plan-review explicitly raised this (Codex Q13, synthesis-adversarial Q-18) — the implementation didn't address it. **No test covers `Some("garbage") + Some(classifiable_adapter)`**, so the regression would not be caught. Fix: either preserve `AutoPickedByGpuClass` on the invalid-env path and emit a `warn!` for the typo (mirroring `:893–:897`), or split `EnvInvalidFallback` into pixel- vs class-flavored variants.

### IMPORTANT 4 — `EnvOverride` log line's "would have been" value is now actively misleading.

`renderer.rs:858`: prints `auto_render_scale(...)` — the **old** pixel-budget pick. On RTX 3060 with `HASH_THING_RENDER_SCALE=0.6`, user sees "auto-pick would have been 0.74" when the actual auto path now picks 1.0. The diagnostic counterfactual must be class-aware. Trident plan-review flagged this (Codex Q13). Fix: build the auto pair via the existing `resolved_render_scale` machinery and surface both class_pick and pixel_pick on this branch too.

### IMPORTANT 5 — `name.contains("Apple M1")` future-bug.

Substring (not anchored) match: when Apple ships "Apple M10", every M10 user silently gets `AppleM1M2 = 0.5`. Trident-plan-review evolutionary-codex #6 / synthesis-evolutionary #10 flagged this; not addressed. Fix: anchor the digit (`starts_with("Apple M1") && next_char_is_not_digit`).

### IMPORTANT 6 — `Apple M2 Ultra → 0.5` is locked in by test.

Mac Studio / Mac Pro are decisively *not* bottom-quartile audience but get the same default as base M1 MBA. Either justify in `GpuClass` doc-comment ("audience-distribution heuristic, not per-chip estimate") or split M-Ultra out.

### IMPORTANT 7 — Test gaps: invariants not defended.

Missing: invalid-env + classifiable adapter (BLOCKER 3 regression). Missing: `DiscreteGpu + "Apple M2"` (BLOCKER 2). Missing: `Other + "Apple M1"`. Missing: empty `name` on each `DeviceType`. Tests defend the implementation as written, not the contract — a future refactor sees green CI for spec-violating behavior.

### NIT 8 — `pub(crate)` asymmetry.

`classify_gpu` / `GpuClass` are `pub(crate)` but unused outside `renderer.rs`. Either lower to private or document the uzrr consumer plan.

---

### Verdict

**Do not ship as-is.** BLOCKER 1 + 2 produce wrong startup defaults on real, non-trivial slices of the spec'd audience. BLOCKER 3 will silently corrupt uzrr's enable-signal. The plan-review identified these; the implementation went around them.
