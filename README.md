# hash-thing

3D voxel engine built on a hash-consed octree with SVDAG raycasting and cellular automaton simulation.

## Quick start

```bash
cargo run --release
```

First build takes ~60s (wgpu + dependencies). Subsequent builds are incremental.

## What you're looking at

A 64^3 procedural terrain (heightmap + value noise) rendered via an SVDAG raycaster on the GPU. The simulation engine supports material-aware cellular automaton rules (fire spreads to grass, water quenches fire) and Margolus 2x2x2 block rules for gravity and fluid flow.

The octree is hash-consed: identical subtrees share storage. This is the foundation for Hashlife-style memoized stepping (in progress).

## Controls

| Key | Action |
|---|---|
| Mouse drag | Orbit camera |
| Scroll | Zoom |
| Space | Pause / resume simulation |
| S | Single step |
| R | Reset terrain (heightmap only) |
| C | Reset terrain with caves (CA post-pass) |
| D | Reset terrain with caves + dungeons |
| G | Switch to legacy Game of Life sphere seed |
| 1-4 | Switch GoL rule (Amoeba, Crystal, 445, Pyroclastic) |
| V | Toggle Flat3D / SVDAG rendering |
| P | Dump perf + memory summary |
| Esc | Quit |

## Architecture

```
src/
  octree/       Hash-consed octree (node.rs, store.rs)
  sim/          Simulation: CaRule + BlockRule dispatch, Margolus 2x2x2
  terrain/      Procedural generation: heightmap, caves, dungeons, noise
  render/       wgpu renderer: flat 3D texture + SVDAG raycaster
  rng.rs        Deterministic per-cell PRNG (Hashlife-compatible)
  perf.rs       Ring-buffer latency tracker + memory watchdog
```

Key invariant: simulation rules are pure functions of their neighborhood. No global PRNG, no scan-order dependence. This keeps Hashlife memoization structurally valid.

## Requirements

- Rust 1.80+ (uses `is_multiple_of` stabilized in 1.87 -- nightly or recent stable)
- A GPU with Vulkan, Metal, or DX12 support (wgpu backend)

## Tests

```bash
cargo test
cargo clippy -- -D warnings
```
