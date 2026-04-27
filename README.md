# hash-thing

3D voxel cellular automaton engine. Hash-consed octree storage, SVDAG rendering, material-type CA rules.

## Quick demo (pre-built)

If you've already built once on this machine:

```
./target/release/hash-thing
```

## Demo command

Standardized demo runner with a per-user config file:

```
scripts/hash-thing-demo                   # run with current config
scripts/hash-thing-demo --world 256       # one-shot world override
scripts/hash-thing-demo --res 1440p       # one-shot resolution override
scripts/hash-thing-demo set world 256     # persist
scripts/hash-thing-demo set res 1440p     # persist
scripts/hash-thing-demo show              # current config + which binary will run
```

Defaults: `world=512`, `res=1080p`, `scene=default`. Config lives at
`${XDG_CONFIG_HOME:-$HOME/.config}/hash-thing/demo.toml` so changes
propagate to other sessions on next invocation (no `source` needed).

The wrapper picks a binary in this priority order: `target/stable/hash-thing`
→ `target/release/hash-thing` → `cargo build --release`. It always sets
`HASH_THING_FOCUS=1` so the demo window starts focused.

To invoke from any cwd, symlink the wrapper into `$HOME/bin/`:

```
mkdir -p ~/bin
ln -sf "$(pwd)/scripts/hash-thing-demo" ~/bin/hash-thing-demo
```

Make sure `~/bin` is on your `$PATH`. Then any session can run
`! hash-thing-demo` (or just `hash-thing-demo` from a regular shell).

## Build from source

```
cargo run --release
```

First build ~60s. Subsequent builds are incremental (~2-5s).

Requires Rust (install via [rustup](https://rustup.rs)).

## What you're looking at

A 64^3 procedural terrain (heightmap + value noise) rendered via an SVDAG raycaster on the GPU. The simulation engine supports material-aware cellular automaton rules (fire spreads to grass, water quenches fire) and Margolus 2x2x2 block rules for gravity and fluid flow.

The octree is hash-consed: identical subtrees share storage. This is the foundation for Hashlife-style memoized stepping (in progress).

## Controls

| Key | Action |
|-----|--------|
| **Mouse drag** | Orbit camera |
| **Scroll** | Zoom in/out |
| **Space** | Pause / resume simulation |
| **S** | Single step |
| **R** | Reset terrain (heightmap) |
| **G** | Switch to legacy Game of Life sphere seed |
| **1-4** | Switch CA rule (Amoeba, Crystal, Rule445, Pyroclastic) |
| **V** | Toggle render mode (Flat3D / SVDAG) |
| **P** | Print perf stats |
| **Esc** | Quit |

## Architecture

```
src/
  octree/       Hash-consed octree (node.rs, store.rs)
  sim/          Simulation: CaRule + BlockRule dispatch, Margolus 2x2x2
  terrain/      Procedural generation: heightmap, noise
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
