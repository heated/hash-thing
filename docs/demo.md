# Demo Runner

`scripts/hash-thing-demo` is the standard way to launch the current demo from a
repo checkout or from a symlink on your `PATH`.

```bash
scripts/hash-thing-demo
scripts/hash-thing-demo --world 256
scripts/hash-thing-demo --res 1440p
scripts/hash-thing-demo show
```

For repeat launches from any seat:

```bash
ln -sf /path/to/hash-thing/scripts/hash-thing-demo ~/bin/hash-thing-demo
hash-thing-demo
```

The wrapper resolves the repo root from its own path, so the symlink can be run
outside the checkout.

## Configuration

Persistent settings live at:

```text
${XDG_CONFIG_HOME:-$HOME/.config}/hash-thing/demo.toml
```

Defaults:

```toml
world = 512
resolution = "1080p"
scene = "default"
```

Update persistent settings with:

```bash
hash-thing-demo set world 256
hash-thing-demo set res 1440p
hash-thing-demo set scene default
hash-thing-demo show
```

`world` must be a positive power of two. `res` accepts `720p`, `1080p`,
`1440p`, `2160p`, `4k`, or `WxH` such as `1920x1080`. `scene` only supports
`default` today.

One-shot flags do not rewrite `demo.toml`:

```bash
hash-thing-demo --world 256
hash-thing-demo --res 720p
```

## Binary Selection

The wrapper chooses a binary in this order:

1. `target/stable/hash-thing`
2. `target/release/hash-thing`
3. `cargo build --release`, then `target/release/hash-thing`

`target/stable/hash-thing` is the stable demo-cut location. The release fallback
is intentional for demo distribution; perf work should use the project's perf
profile directly, not the wrapper.

## CLI Flags

The binary accepts:

```bash
hash-thing [SIZE] [--demo | --res 720p|1080p|1440p|2160p|4k|WxH]
```

`--demo` and `--res` are mutually exclusive. `--demo` focuses the window on
launch and uses the fixed demo render scale. `--res` keeps pixel-budget render
scale semantics and does not imply demo focus by itself.

The wrapper passes `--demo` when the configured resolution is `1080p`; for any
other configured or one-shot resolution it passes `--res VALUE`. The wrapper
also sets `HASH_THING_FOCUS=1`, so wrapper launches start focused either way.

Plain `cargo run` starts unfocused by default. Use one of these when you want
the window to activate on launch:

```bash
cargo run -- --demo
HASH_THING_FOCUS=1 cargo run
```
