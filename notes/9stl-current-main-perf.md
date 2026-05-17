# hash-thing-9stl current-main perf capture

Date: 2026-05-03
Actor: moss
Commit measured: `8329e22`
Hardware: Apple M2 / Metal integrated GPU
Coordinate: `demo · default-demo · passive-active · warming`

## Commands

```bash
cargo build --profile perf
RUST_LOG=info HASH_THING_PERF_CAPTURE=1 timeout 25s target/perf/hash-thing 256 --demo
RUST_LOG=info HASH_THING_PERF_CAPTURE=1 HASH_THING_FREEZE_SIM=1 timeout 25s target/perf/hash-thing 256 --demo
RUST_LOG=info HASH_THING_PERF_CAPTURE=1 timeout 25s target/perf/hash-thing 256 --res 1080p
```

Logs:

- `.ship-notes/9stl/sim-running-256-demo.log`
- `.ship-notes/9stl/freeze-sim-256-demo.log`
- `.ship-notes/9stl/terrain-running-256-1080p.log`

The third command was intended as a non-demo terrain comparison, but current
main still launched the pyroclastic default scene without `--demo`; treat it
as a same-scene repeat, not a terrain comparison.

## Findings

At `demo · default-demo · passive-active · warming`, the sim-running 256^3
capture reproduces the reported frame-pacing shape:

- `frame_total` starts around 54/92 ms and settles mostly in the 46-73 ms mean
  range during the 25 s capture.
- `step` rises from about 60/93 ms to about 96/135 ms as the cache fills and
  the active scene grows.
- `step_poll_lag` remains material, mostly about 26-40 ms mean.
- `surface_acquire_cpu` is also material, about 22-48 ms mean depending on the
  sample window.
- `render_gpu` falls to about 0.08-0.12 ms once timestamp lag catches up.

The frozen-sim comparison says the frame-rate problem is not only the sim
worker:

- With `HASH_THING_FREEZE_SIM=1`, `step`, `step_poll_lag`, and phase timings
  are zero.
- `frame_total` still climbs to about 70 ms mean.
- `surface_acquire_cpu` carries almost all of that cost, about 69 ms mean with
  p95 around 82-85 ms.
- `prior_gpu_pipeline_cpu` sits around 140-150 ms while measured GPU work stays
  around 0.09-0.13 ms.

The "felt FPS much lower than reciprocal frame_total" part was not directly
measured; that still needs either a displayed-present cadence metric or human
visual confirmation. But the logs do explain why the user experience is bad:
the renderer is commonly spending multiple 60 Hz frame intervals blocked in
surface acquisition / prior-pipeline pacing, and the live sim adds tens of ms
of wait on top.

## Answers to the bead asks

1. `present_cpu` is not the culprit; it stays near 0.02 ms. The visible frame
   budget is dominated by `surface_acquire_cpu` / `acq_inflight_cpu` and
   `prior_gpu_pipeline_cpu`, with live-sim `step_poll_lag` as an additional
   contributor.

2. The current 25 s captures do not prove pyroclastic is heavier than a true
   terrain-only scene, because the attempted non-demo run still used the
   pyroclastic default scene. Within the captured scene, however, the sim cost
   grows from about 60 ms to about 100 ms over 25 s, so the scene is definitely
   sim-heavy enough to contribute to perceived stutter.

3. Existing perf logging was enough to answer the main diagnostic question.
   The carrier is mixed: live sim wait matters, but frozen sim still leaves a
   roughly 70 ms frame, so the primary render-side follow-up is the existing
   `hash-thing-dbz5.1` surface-acquire / submission-fence instrumentation bead.

## Conclusion

`hash-thing-9stl` should close as characterized, not fixed. The actionable
follow-ups already exist:

- `hash-thing-vqke` owns the sim-step side.
- `hash-thing-dbz5.1` owns the surface-acquire / submission-fence
  instrumentation side.
