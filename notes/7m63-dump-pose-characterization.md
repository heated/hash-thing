# 7m63 Dump-Pose Characterization

Date: 2026-05-03
Bead: `hash-thing-7m63`
Status: diagnostic only; do not close without edward verification

## Inputs

Inspected the deterministic dump-pose captures from `hash-thing-jszv`:

- `.ship-notes/7m63/moss-2026-05-03-default-256-wall-pose.png`
- `.ship-notes/7m63/moss-2026-05-03-default-256-terrain-wide-pose.png`
- `.ship-notes/7m63/moss-2026-05-03-spectacle-256-blocks-pose.png`

Validation that the captures exist and are full-size:

```bash
sips -g pixelWidth -g pixelHeight .ship-notes/7m63/*.png
```

All inspected captures are `1920x1080`.

## Observations

The wall pose reproduces the already-noted horizontal discontinuity across the
pyroclastic chamber wall. The break is screen-horizontal, spans most of the
wall, and appears on a large continuous face rather than on one isolated block.

The terrain-wide pose shows a stronger version of the same class: a broad
horizontal band on a large LOD wall/terrain face, plus vertical LOD-scale
texture variation. This makes an LOD/procedural-sampling seam more plausible
than a simple per-face normal sign error.

The spectacle blocks pose is useful as a counterexample. It shows a long
free-standing block structure with visible top, side, and underside faces, but
does not obviously show one face shaded with the wrong sign. The visible
differences read mostly as expected material/procedural variation and geometry
edges.

## Interpretation

Current deterministic captures point at a large-surface LOD/procedural-shading
discontinuity, not a clean "one face normal is flipped" repro. That does not
disprove edward's live observation: the original artifact may be camera-motion
dependent, material-specific, or easier to see interactively than in these
static captures.

## Next Useful Technical Step

Add a debug material/normal dump or shader mode for `--dump-frame` that colors:

- selected normal axis
- LOD node versus leaf hit
- representative material versus leaf material

That would turn the current visual ambiguity into a pixel-checkable artifact.
Until then, `hash-thing-7m63` should stay open under
`do-not-close-without-human-verify`.
