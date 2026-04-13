# World Generation Evolution

`hash-thing-fha` is a future-facing exploration bead. The seam that matters on
`origin/main` is not "some future generator system"; it is the concrete
`WorldGen` contract that already feeds the recursive octree builder.

- `WorldGen::sample` decides the exact material at a coordinate.
- `WorldGen::classify` decides whether a whole region is uniform enough to
  short-circuit recursion.
- Bounded structure passes still make sense, but only if they can be expressed
  as deterministic owned-region generation rather than mutable global state.
- Determinism is the non-negotiable invariant: a generator must remain a pure
  function of coordinates, params, and seed.

That means "future worldgen" is not one abstract lane. It is three concrete
lanes with different costs.

## Lane 1: Pure density fields

Best fit for:

- Voronoi skeletons
- Reaction-diffusion snapshots baked into a field
- ML materializers, if inference is frozen and deterministic

Why this lane fits:

- `classify` can still prove large uniform regions, so the recursive builder
  keeps its asymptotic win.
- Sampling composes with the existing `gen_region` path.
- These approaches scale naturally to infinite-space evaluation.

What must stay true:

- The field must supply conservative bounds for `classify`, or it becomes a
  leaf-by-leaf generator and loses the whole octree advantage.
- Any learned model must be version-locked and platform-stable; "same seed,
  different GPU" is unacceptable.

Current repo foothold:

- `src/terrain/noise.rs::voronoi_2d` is a minimal deterministic primitive for
  Voronoi-cell boundary experiments. The useful signal is `edge_gap`: low
  values mark walls / bones / cathedral ribs.

## Lane 2: Bounded structure post-passes

Best fit for:

- WFC interiors
- Narrative / grammar placement
- Pre-committed structure catalogs with side-table caches

Why this lane fits:

- These systems usually reason about adjacency, rooms, or authored structure
  graphs rather than continuous scalar fields.
- They are easier to stage as "generate a local scaffold, then carve or stamp
  it" than as a closed-form `sample(x, y, z)`.

What must stay true:

- Ownership rules across region boundaries have to be explicit up front.
- Cached side tables are only acceptable if the cache key is deterministic and
  the generated result can be re-derived from seed + coordinates.
- WFC backtracking budgets need hard caps; unbounded retries would inject
  frame-time cliffs into region generation.

## Lane 3: Hybrid discrete anchors + pure local detail

Best fit for:

- Large authored landmarks plus procedural infill
- Grammar-selected macro structures with deterministic local materials
- The "option 5" shape mentioned in the bead: pre-committed discrete
  structures, pure functions for everything between them

Why this lane fits:

- It preserves long-range coherence without requiring the whole world to be a
  closed-form function.
- It keeps most cells cheap: only anchor lookup is discrete; local surface or
  material detail can stay purely sampled.

What must stay true:

- Anchor ownership must be globally canonical.
- The anchor index must be benchmarked before adoption. If lookup cost
  dominates, the design is fighting the architecture instead of using it.

## Suggested next technical probes

Small, landable experiments that would add real signal:

1. Add a 3D Voronoi density field and measure whether `classify` can still
   prove large empty / solid bands.
2. Prototype a tiny grammar side table keyed by region pair and benchmark the
   hash-table lookup cost during region generation.
3. Build a bounded WFC room filler that runs only inside an already-owned room
   volume, never across chunk ownership boundaries.

The pattern is simple: future generators should first prove they respect the
existing deterministic octree seams, then earn integration.
