#!/usr/bin/env bash
# One-command demo: build and run hash-thing from a fresh clone.
# Usage: ./scripts/demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Building hash-thing (release)..."
cargo build --release 2>&1 | tail -1

echo ""
echo "Controls: mouse-drag to orbit, scroll to zoom, Space to pause/resume,"
echo "  S=step, R=reset, G=GoL, 1-4=rules, V=toggle render, Esc=quit"
echo ""

exec cargo run --release
