#!/usr/bin/env bash
# reap-worktrees.sh — recover disk by deleting target/ in worktrees whose
# claimed bead is CLOSED.
#
# hash-thing-kqw0. The project's disk steady state is dominated by stale
# closed-bead worktrees keeping their multi-GB target/ around for days or
# weeks. Active per-worktree cost (post-incremental=off, hash-thing-a5mv)
# is ~600 MB-1.5 GB, so the only reason we sit at >25 GiB project-wide is
# nothing reaps the closed-bead targets.
#
# Conservative by design:
#   - default --dry-run; --apply is the destructive opt-in.
#   - skips active builds (cargo lock held).
#   - skips on any bd-query failure (transient outage MUST NOT widen into
#     a multi-worktree reap incident).
#   - skips codex pool (persistent, no bead claim) and worktree-* cruft
#     (mayor scratchpads — needs human review).
#   - never removes the worktree itself, only target/.
#
# Usage:
#   scripts/reap-worktrees.sh                 # dry-run, summary only
#   scripts/reap-worktrees.sh --verbose       # dry-run, show every worktree
#   scripts/reap-worktrees.sh --apply         # actually delete
#   scripts/reap-worktrees.sh --apply -v      # delete + verbose

set -euo pipefail
shopt -s nullglob

# --- Resolve repo root (independent of cwd, works from any worktree) -------
REPO_ROOT=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null \
              | sed 's|/\.git$||' || true)
if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/.beads" ]]; then
    echo "reap-worktrees: not in a hash-thing git checkout (no .beads/)" >&2
    exit 1
fi

BD="${REPO_ROOT}/.bin/bd"
if [[ ! -x "${BD}" ]]; then
    echo "reap-worktrees: ${BD} not found or not executable" >&2
    exit 1
fi

# --- Args ------------------------------------------------------------------
DRY_RUN=true
VERBOSE=false
for arg in "$@"; do
    case "$arg" in
        --apply)        DRY_RUN=false ;;
        --dry-run)      DRY_RUN=true ;;
        --verbose|-v)   VERBOSE=true ;;
        --help|-h)
            # Print the usage block (lines 3-24, before `set -euo pipefail`).
            sed -n '3,24p' "$0"
            exit 0
            ;;
        *)
            echo "reap-worktrees: unknown flag '$arg' (try --help)" >&2
            exit 1
            ;;
    esac
done

# --- Branch-shape classification -------------------------------------------
# Echo the bead id on stdout for a known seat/<id>... branch, or one of:
#   ""                empty branch (detached, etc.)
#   "@@codex-pool"    codex/* branches
#   "@@unowned"       worktree-* branches
#   "@@unparseable"   anything else
# bd show accepts both `ite4` and `hash-thing-ite4` — we strip the prefix
# off uniformly so callers can pass the bare id.
classify_branch() {
    local branch="$1"
    if [[ -z "$branch" ]]; then echo ""; return; fi
    case "$branch" in
        codex/*)        echo "@@codex-pool"; return ;;
        worktree-*)     echo "@@unowned";    return ;;
    esac
    # <seat>/<rest> — accepted seats are the auto-pool members + mayor.
    local seat rest
    seat="${branch%%/*}"
    rest="${branch#*/}"
    case "$seat" in
        flint|cairn|onyx|ember|spark|mayor) ;;
        *) echo "@@unparseable"; return ;;
    esac
    if [[ -z "$rest" || "$rest" == "$branch" ]]; then
        echo "@@unparseable"; return
    fi
    # Strip optional `hash-thing-` prefix; take the leading id token
    # (alphanumeric, optionally followed by `.<n>` for sub-beads like 8ppq.1).
    local id="${rest#hash-thing-}"
    id="${id%%-*}"
    if [[ -z "$id" || ! "$id" =~ ^[a-z0-9]+(\.[0-9]+)?$ ]]; then
        echo "@@unparseable"; return
    fi
    echo "$id"
}

# --- Build-in-progress guard -----------------------------------------------
# Cargo's lock files live PER PROFILE at `target/<profile>/.cargo-lock`
# (e.g. target/debug/.cargo-lock, target/release/.cargo-lock). There is NO
# top-level target/.cargo-lock — checking that path would always pass the
# "no lock" branch and let the reaper rm -rf during a live build (Claude
# code-review BLOCKER). Plus, cargo+rustc invocations are short-lived and
# can elude a single-shot lsof/pgrep probe. Belt-and-suspenders: any of
# these fire ⇒ skip.
#
#   1. `flock -n -s` on each `target/<profile>/.cargo-lock` (where
#      available) — exclusive holder ⇒ active.
#   2. `lsof +D <target>` shows any open FD inside the tree — captures
#      cargo, rust-analyzer, IDE indexers.
#   3. `find <target> -mmin -1` — anything modified in the last minute
#      means a writer was here recently; conservative skip.
#   4. `pgrep` for any cargo/rustc/sccache process system-wide — coarse,
#      over-skips when a different worktree is building, but the cost
#      is "wait for next reaper run." Never under-skips.
#
# Always skip-on-error. Never reap if uncertain.
build_in_progress() {
    local target="$1"
    local lockfile
    local found_lock=false

    # (1) per-profile flocks
    if command -v flock >/dev/null 2>&1; then
        for lockfile in "$target"/*/.cargo-lock; do
            [[ -f "$lockfile" ]] || continue
            found_lock=true
            if ! flock -n -s "$lockfile" -c true 2>/dev/null; then
                return 0   # exclusive holder exists ⇒ build in progress
            fi
        done
        # found_lock=true and all acquired ⇒ idle by lock. Continue to
        # (2)-(4) anyway so a transient cargo invocation between locks
        # still gets caught.
    fi

    # (2) lsof open-FD check
    if command -v lsof >/dev/null 2>&1 \
       && lsof +D "$target" >/dev/null 2>&1; then
        return 0
    fi

    # (3) recent-mtime check
    if [[ -n "$(find "$target" -mmin -1 -print -quit 2>/dev/null)" ]]; then
        return 0
    fi

    # (4) any cargo/rustc anywhere. Deliberately NOT including sccache —
    # it runs as a long-lived server daemon (`sccache --start-server`),
    # so its presence isn't a build signal. Including it would make the
    # reaper a permanent no-op on any machine running sccache (i.e. all
    # of them, post-a5mv).
    if pgrep -x cargo >/dev/null 2>&1 \
       || pgrep -x rustc >/dev/null 2>&1; then
        return 0
    fi

    return 1
}

# --- bd status query -------------------------------------------------------
# Echoes one of OPEN | IN_PROGRESS | CLOSED | BLOCKED | DEFERRED, or empty
# string on any failure (caller treats empty as "skip with bd-error").
bd_status() {
    local id="$1"
    local out
    out=$("$BD" show "$id" 2>/dev/null) || { echo ""; return; }
    # Anchor to the bracketed `[● P0 · CLOSED]` header pattern (Claude
    # code-review IMPORTANT). Description text containing words like
    # "OPEN-source" or "CLOSED-loop" can't false-match this. The pattern
    # tolerates any priority token between the dot/dot bullets.
    head -n 5 <<< "$out" \
        | grep -oE '\[● [^]]+ · (IN_PROGRESS|CLOSED|BLOCKED|DEFERRED|OPEN)\]' \
        | head -1 \
        | grep -oE '(IN_PROGRESS|CLOSED|BLOCKED|DEFERRED|OPEN)'
}

# --- Pretty size ----------------------------------------------------------
human_size() {
    local path="$1"
    du -sh "$path" 2>/dev/null | cut -f1
}

# --- Main loop -------------------------------------------------------------
declare -i reaped=0
declare -i skipped=0

WORKTREE_ROOTS=(
    "${REPO_ROOT}/.claude/worktrees"
    "${REPO_ROOT}/.worktrees"
)

action_line() {
    # $1 = verb (REAP|SKIP|DRY-RUN), $2 = size or "-", $3 = name, $4 = reason
    printf "  %-9s %-7s %-40s %s\n" "$1" "$2" "$3" "$4"
}

emit_skip() {
    local name="$1" reason="$2"
    skipped+=1
    if $VERBOSE; then
        action_line "SKIP" "-" "$name" "$reason"
    fi
}

for root in "${WORKTREE_ROOTS[@]}"; do
    [[ -d "$root" ]] || continue
    for wt in "$root"/*/; do
        wt="${wt%/}"
        name=$(basename "$wt")
        target="${wt}/target"
        if [[ ! -d "$target" ]]; then
            emit_skip "$name" "no target/"
            continue
        fi

        branch=$(git -C "$wt" branch --show-current 2>/dev/null || true)
        cls=$(classify_branch "$branch")
        case "$cls" in
            "")              emit_skip "$name" "detached HEAD"; continue ;;
            "@@codex-pool")  emit_skip "$name" "codex pool ($branch)"; continue ;;
            "@@unowned")     emit_skip "$name" "unowned ($branch)"; continue ;;
            "@@unparseable") emit_skip "$name" "unparseable ($branch)"; continue ;;
        esac
        bead="$cls"

        if build_in_progress "$target"; then
            emit_skip "$name" "build-in-progress (bead=$bead)"
            continue
        fi

        status=$(bd_status "$bead")
        if [[ -z "$status" ]]; then
            emit_skip "$name" "bd-error (bead=$bead)"
            continue
        fi
        if [[ "$status" != "CLOSED" ]]; then
            emit_skip "$name" "bead $bead $status"
            continue
        fi

        size=$(human_size "$target")
        if $DRY_RUN; then
            action_line "DRY-RUN" "$size" "$name" "would reap (bead=$bead CLOSED)"
        else
            rm -rf "$target"
            action_line "REAP" "$size" "$name" "(bead=$bead CLOSED)"
        fi
        reaped+=1
    done
done

mode=$($DRY_RUN && echo "dry-run" || echo "apply")
echo ""
echo "reap-worktrees [$mode]: ${reaped} reapable, ${skipped} skipped"
if $DRY_RUN && (( reaped > 0 )); then
    echo "  (re-run with --apply to actually delete)"
fi
exit 0
