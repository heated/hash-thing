# hash-thing

3D voxel cellular automaton engine. Hash-consed octree storage, SVDAG rendering, material-type CA rules.

## Quick demo (pre-built)

If you've already built once on this machine:

```
./target/release/hash-thing
```

## Build from source

```
cargo run --release
```

First build ~60s. Subsequent builds are incremental (~2-5s).

Requires Rust (install via [rustup](https://rustup.rs)).

## Controls

| Key | Action |
|-----|--------|
| **Space** | Pause / resume simulation |
| **S** | Single step |
| **R** | Reset world |
| **1-4** | Switch CA rule (Amoeba, Crystal, Rule445, Pyroclastic) |
| **V** | Toggle render mode (Flat3D / SVDAG) |
| **P** | Print perf stats |
| **Esc** | Quit |
| **Mouse drag** | Orbit camera |
| **Scroll** | Zoom in/out |
