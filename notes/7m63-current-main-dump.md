# 7m63 Current-Main Dump Characterization

Date: 2026-05-03
Bead: `hash-thing-7m63`
Commit tested: `849a3b6`
Actor: `moss`
Status: diagnostic only; do not close without edward verification

## Commands

```bash
cargo run --bin hash-thing -- 256 --dump-frame .ship-notes/7m63/moss-2026-05-03-default-256-firstframe.png
cargo run --bin hash-thing -- 512 --dump-frame .ship-notes/7m63/moss-2026-05-03-default-512-firstframe.png
cargo run --bin hash-thing -- 256 --dump-frame .ship-notes/7m63/moss-2026-05-03-spectacle-256-firstframe.png --dump-scene spectacle
```

All three commands completed and wrote 1920x1080 PNGs.

## Observations

The default-scene dumps now launch into the pyroclastic chamber, so the first
frame is a close wall face rather than the older terrain/sand-cross view. Both
the 256^3 and 512^3 default dumps show a repeatable, screen-horizontal shading
discontinuity across the wall near the crosshair height. The discontinuity is
visible on the same broad face, not on an isolated free-standing cube.

That may be the current automated manifestation of the reported one-face
shading artifact, but this diagnostic does not prove it. It could also be an
LOD/procedural-sampling discontinuity on a large wall plane. Edward's live-eye
verification is still required by the bead.

The `--dump-scene spectacle` frame was not useful for block-face
characterization: it rendered mostly sky from the inherited camera/player pose.
That supports the earlier recommendation that visual bug work needs dump-camera
pose presets or a debug-color/normal dump, not just scene swapping.

## What This Rules In / Out

- Rules in: current main has a stable first-frame image with a visible
  horizontal wall-face discontinuity at both 256^3 and 512^3.
- Rules out: nothing about the original human-visible artifact. Static
  dump-frame screenshots still cannot answer whether the wrong face is
  camera-relative, world-fixed, material-specific, or only visible during live
  movement.
- Still blocked for closure: edward must confirm in a live build that the
  artifact is gone, or annotate the wrong face/material if it remains.

## Next Useful Technical Step

Add a diagnostic dump pose surface, for example `--dump-pose <name>`, with a few
known camera/player poses aimed at free-standing blocks, the default chamber
wall, and high-LOD terrain. That would let reviewers compare the same face and
material across commits without relying on interactive focus or memory.

