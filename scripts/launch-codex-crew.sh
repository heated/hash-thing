#!/usr/bin/env bash
# launch-codex-crew.sh — spawn a Codex crew seat in a cmux workspace
#
# Usage: scripts/launch-codex-crew.sh <seat-name> [worktree-name]
#
# Creates (or reuses) a git worktree, writes .beads/actor, opens a new cmux
# workspace at that path, and seeds it with a codex exec command that reads
# project instructions and starts claiming+shipping beads.
#
# hash-thing-d13. Option A from the bead description.

set -euo pipefail

SEAT="${1:?Usage: launch-codex-crew.sh <seat-name> [worktree-name]}"
WORKTREE_NAME="${2:-codex-$SEAT}"

# Resolve repo root — works from any worktree or the primary checkout.
REPO_ROOT="$(cd "$(git rev-parse --show-toplevel)" && git rev-parse --path-format=absolute --git-common-dir | sed 's|/\.git$||')"
WORKTREES_DIR="$REPO_ROOT/.claude/worktrees"
WORKTREE_PATH="$WORKTREES_DIR/$WORKTREE_NAME"

# --- Validate seat name is in the auto-pool ---
POOL=(flint cairn onyx ember spark)
valid=false
for s in "${POOL[@]}"; do [[ "$s" == "$SEAT" ]] && valid=true; done
if [[ "$valid" != true ]]; then
    echo "error: seat '$SEAT' not in pool (${POOL[*]}). mayor is explicit-only." >&2
    exit 1
fi

# --- Check seat not already claimed by another worktree ---
for actor_file in "$WORKTREES_DIR"/*/.beads/actor; do
    [[ ! -f "$actor_file" ]] && continue
    existing_seat="$(cat "$actor_file")"
    existing_wt="$(basename "$(dirname "$(dirname "$actor_file")")")"
    if [[ "$existing_seat" == "$SEAT" && "$existing_wt" != "$WORKTREE_NAME" ]]; then
        echo "warning: seat '$SEAT' already active in worktree '$existing_wt'" >&2
        echo "  proceeding anyway — Codex will use the same BEADS_ACTOR identity" >&2
    fi
done

# --- Create worktree if needed ---
if [[ ! -d "$WORKTREE_PATH" ]]; then
    echo "Creating worktree '$WORKTREE_NAME'..."
    git fetch origin 2>/dev/null || true
    git worktree add "$WORKTREE_PATH" origin/main --detach 2>/dev/null
fi

# --- Write seat identity ---
mkdir -p "$WORKTREE_PATH/.beads"
echo "$SEAT" > "$WORKTREE_PATH/.beads/actor"

# --- Build the codex seed command ---
# codex exec runs in full-auto mode with no git repo check (worktree .git is a
# file, not a dir — some tools get confused). The prompt points at AGENTS.md
# which is a symlink to CLAUDE.md, giving Codex the same project context as
# Claude Code sessions.
CODEX_CMD="export BEADS_ACTOR=$SEAT && codex exec --full-auto --skip-git-repo-check \
\"You are seat '$SEAT' on the hash-thing crew. \
Read AGENTS.md for project instructions. \
Run 'bd ready -n 10' to see available work. \
Pick the highest-priority unclaimed bead. \
Claim it: bd update <id> --claim. \
Implement it, write tests, validate (cargo test, cargo clippy -- -D warnings, cargo fmt --check). \
Commit and land on main: git push origin HEAD:main. \
Close the bead: bd close <id>. \
Repeat until bd ready is empty or you hit a design gate.\""

# --- Launch cmux workspace with the codex command ---
echo "Launching Codex crew seat '$SEAT' in worktree '$WORKTREE_NAME'..."
cmux new-workspace --cwd "$WORKTREE_PATH" --command "$CODEX_CMD"

echo "Done. Codex seat '$SEAT' is running in workspace '$WORKTREE_NAME'."
