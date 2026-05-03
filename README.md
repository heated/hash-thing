# hash-thing

`hash-thing` is an experimental 3D voxel cellular-automata engine for building
large material worlds that keep simulating while most of the space is quiet.

The bet: if a world is stored as a hash-consed octree, rendered as a GPU SVDAG,
and stepped with Hashlife-style memoization, then stable and repeated structures
can become cheap enough to support games that ordinary chunk sims struggle to
run.

## What It Is

Today the project is a real-time 3D powder-game sandbox and engine proof. You
can launch into a voxel world with stone, sand, water, lava, fire, ice, acid,
vine, metal, clone sources, and other material rules running together. The
current direction is to make that sandbox feel like a dense material toy first,
then use the engine's scale and memoization to find the game loop that only this
substrate can carry.

Under the hood:

- A hash-consed sparse octree stores the world, so identical subtrees share
  memory.
- A dynamic SVDAG renderer raycasts the octree-derived scene on the GPU.
- Material rules are local and deterministic, which keeps memoized stepping
  valid.
- Hashlife-style recursive stepping can skip empty, inert, cached, and repeated
  regions instead of touching every voxel every tick.
- Streaming and LOD work are aimed at worlds far beyond the first 256^3 demo
  scale, with 4096^3 as the long-range target.

The project is not just chasing "3D Powder Toy." The current research question
is sharper: what player action layer becomes possible when huge quiet regions,
localized active fronts, and repeated structures are computationally privileged?

## Current Demo

The demo is moving toward a compact 3D powder-game slice: a visually rich
volcanic/material chamber, immediate editing tools, visible cascades, and one
scale reveal that shows the simulation is not a tiny test box.

Recent prototype work also includes **Quarantine Atlas**, a playable containment
sketch where a live hazard front moves across a mostly quiet map and the player
spends a small budget on reusable counter-pattern stamps. It is an action-layer
test, not the final game direction.

Screenshot/video placeholder: TODO add a fresh capture from the current demo
once the scene art stabilizes.

## Why It Is Interesting

Most voxel games pay for local simulation by limiting the amount of world that
is actually active. `hash-thing` is testing a different path: make repeated,
stable, and empty structure structurally shareable, then spend compute on the
small regions where material behavior is actually changing.

Structured thesis probes show Hashlife is already a real multiplier in both
easy and hard regimes. At `demo · default-demo · cascade · churning`, the
current ledger records 79.15x work-elision p05 and a 20.40 ms hashlife p95
against a 939.17 ms chunk-array p95 on the same cascade scenario. At
`demo · default-terrain · microchurn · saturated`, the post-ite4 default
`RayonBfs` path records a 6.7 ms step median. These are still regime-specific
numbers, not a blanket victory claim; every new perf claim should cite its
world, scene, intensity, and regime.

That honesty matters. The goal is not a benchmark trick; it is a game where
scale, stability, and reusable material patterns are player-facing resources.

## Run It

Requirements:

- Rust via [rustup](https://rustup.rs)
- A GPU supported by `wgpu` through Metal, Vulkan, or DX12

Use the demo wrapper:

```bash
scripts/hash-thing-demo
```

Useful one-shot overrides:

```bash
scripts/hash-thing-demo --world 256
scripts/hash-thing-demo --res 1440p
```

The wrapper stores defaults in
`${XDG_CONFIG_HOME:-$HOME/.config}/hash-thing/demo.toml`, prefers an existing
`target/stable/hash-thing` or `target/release/hash-thing` binary, and falls back
to building the release binary when needed.

For repeat launches from anywhere:

```bash
mkdir -p ~/bin
ln -sf "$(pwd)/scripts/hash-thing-demo" ~/bin/hash-thing-demo
```

Then make sure `~/bin` is on your `PATH` and run `hash-thing-demo`.

## Status

This is research software, not a packaged game. The engine pieces are real, the
demo is playable, and the product direction is still being tested through small
prototype loops and measured perf regimes.
