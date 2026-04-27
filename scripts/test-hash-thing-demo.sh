#!/usr/bin/env bash
# Test harness for scripts/hash-thing-demo.
# Runs deterministic asserts against the wrapper using a tempdir
# XDG_CONFIG_HOME and HASH_THING_DEMO_DRY_RUN=1 so the actual game
# binary is never invoked.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WRAPPER="$REPO_ROOT/scripts/hash-thing-demo"

TMP_HOME=$(mktemp -d)
trap 'rm -rf "$TMP_HOME"' EXIT

export XDG_CONFIG_HOME="$TMP_HOME/.config"
export HASH_THING_DEMO_DRY_RUN=1

CONFIG="$XDG_CONFIG_HOME/hash-thing/demo.toml"
PASS=0
FAIL=0

assert_eq() {
    # assert_eq <label> <actual> <expected>
    local label="$1" actual="$2" expected="$3"
    if [ "$actual" = "$expected" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $label"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $label"
        echo "    expected: $expected"
        echo "    actual:   $actual"
    fi
}

assert_contains() {
    local label="$1" haystack="$2" needle="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        PASS=$((PASS + 1))
        echo "  PASS: $label"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $label (no match for '$needle')"
        echo "    output: $haystack"
    fi
}

assert_status() {
    local label="$1" expected="$2"; shift 2
    set +e
    "$@" > /dev/null 2>&1
    local rc=$?
    set -e
    if [ "$rc" -eq "$expected" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $label"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $label (expected exit $expected, got $rc)"
    fi
}

echo "== test 1: show with no config prints defaults =="
out=$("$WRAPPER" show)
assert_contains "show prints world default" "$out" "world:      512"
assert_contains "show prints res default" "$out" "resolution: 1080p"
assert_contains "show prints scene default" "$out" "scene:      default"
assert_contains "show notes config absent" "$out" "(does not exist; using defaults)"

echo "== test 2: set world 256 persists =="
"$WRAPPER" set world 256 > /dev/null
[ -f "$CONFIG" ] || { echo "  FAIL: config file not created"; FAIL=$((FAIL + 1)); }
out=$("$WRAPPER" show)
assert_contains "show reflects set world" "$out" "world:      256"

echo "== test 3: set res 1440p persists; round-trip preserves world =="
"$WRAPPER" set res 1440p > /dev/null
out=$("$WRAPPER" show)
assert_contains "show reflects set res" "$out" "resolution: 1440p"
assert_contains "set res preserved world" "$out" "world:      256"

echo "== test 4: round-trip — set world then re-read produces same file content =="
"$WRAPPER" set world 128 > /dev/null
first=$(cat "$CONFIG")
"$WRAPPER" set world 128 > /dev/null
second=$(cat "$CONFIG")
assert_eq "config round-trip stable" "$first" "$second"

echo "== test 5: set rejects non-power-of-two world =="
assert_status "set world 100 rejected" 1 "$WRAPPER" set world 100
assert_status "set world 0 rejected" 1 "$WRAPPER" set world 0
assert_status "set world abc rejected" 1 "$WRAPPER" set world abc

echo "== test 6: set rejects bad res =="
assert_status "set res 999p rejected" 1 "$WRAPPER" set res 999p
assert_status "set res garbage rejected" 1 "$WRAPPER" set res garbage
assert_status "set res 0x1080 rejected" 1 "$WRAPPER" set res 0x1080

echo "== test 7: set rejects unknown key =="
assert_status "set unknown rejected" 1 "$WRAPPER" set foo bar

echo "== test 8: set scene default ok, other rejected =="
assert_status "set scene default ok" 0 "$WRAPPER" set scene default
assert_status "set scene cave rejected" 1 "$WRAPPER" set scene cave

echo "== test 9: dry-run with default config emits --demo =="
"$WRAPPER" set world 512 > /dev/null
"$WRAPPER" set res 1080p > /dev/null
out=$("$WRAPPER")
assert_contains "default config picks --demo" "$out" " --demo"
assert_contains "default config passes 512" "$out" " 512 "
assert_contains "default config sets focus env" "$out" "HASH_THING_FOCUS=1"

echo "== test 10: dry-run with non-1080p res emits --res =="
"$WRAPPER" set res 1440p > /dev/null
out=$("$WRAPPER")
assert_contains "non-1080p picks --res" "$out" " --res 1440p"
out_no_demo=$(echo "$out" | grep -v -- '--demo' || true)
assert_eq "non-1080p does not pass --demo" "$out" "$out_no_demo"

echo "== test 11: --world one-shot does not write config =="
"$WRAPPER" set world 256 > /dev/null
"$WRAPPER" --world 64 > /dev/null
persisted=$("$WRAPPER" show | grep "^world:")
assert_contains "one-shot --world did not persist" "$persisted" "256"

echo "== test 12: --res one-shot does not write config =="
"$WRAPPER" set res 1440p > /dev/null
"$WRAPPER" --res 720p > /dev/null
persisted=$("$WRAPPER" show | grep "^resolution:")
assert_contains "one-shot --res did not persist" "$persisted" "1440p"

echo "== test 13: --help exits 0 with usage =="
out=$("$WRAPPER" --help)
assert_contains "--help prints usage" "$out" "usage: hash-thing-demo"

echo "== test 14: show prints binary metadata when present =="
# Create a fake binary at target/release/hash-thing
mkdir -p "$REPO_ROOT/target/release"
created_fake=0
if [ ! -e "$REPO_ROOT/target/release/hash-thing" ]; then
    touch "$REPO_ROOT/target/release/hash-thing"
    chmod +x "$REPO_ROOT/target/release/hash-thing"
    created_fake=1
fi
out=$("$WRAPPER" show)
assert_contains "show prints binary line" "$out" "binary:"
assert_contains "show prints binary mtime" "$out" "mtime "
if [ "$created_fake" = 1 ]; then
    rm -f "$REPO_ROOT/target/release/hash-thing"
fi

echo "== test 15: hand-edited TOML with quotes / whitespace tolerated =="
mkdir -p "$XDG_CONFIG_HOME/hash-thing"
cat > "$CONFIG" <<'EOF'
# user-edited
world  =  64
resolution  =  "720p"
scene = "default"
EOF
out=$("$WRAPPER" show)
assert_contains "tolerates whitespace world" "$out" "world:      64"
assert_contains "tolerates quoted res" "$out" "resolution: 720p"

echo
echo "== summary: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
