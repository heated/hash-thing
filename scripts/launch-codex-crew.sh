#!/usr/bin/env bash
# launch-codex-crew.sh — spawn a Codex crew seat in a worktree
#
# Usage:
#   scripts/launch-codex-crew.sh <seat-name> [--workspace <ref>] [--exec] [--dry-run]
#
# Creates a fresh git worktree at .claude/worktrees/codex-<seat>, writes
# .beads/actor, and launches Codex in interactive mode with full crew context.
# Interactive mode keeps the session alive between tasks — no relaunch needed.
#
# Options:
#   --workspace <ref>   Send to an existing cmux workspace instead of creating one
#   --exec              Use non-interactive exec mode (exits after one pass)
#   --dry-run           Print what would happen without doing it
#
# hash-thing-d13

set -euo pipefail

# --- Parse args ---
SEAT=""
WORKSPACE_REF=""
EXEC_MODE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace) WORKSPACE_REF="$2"; shift 2 ;;
        --exec) EXEC_MODE=true; shift ;;
        --interactive) shift ;;  # no-op, interactive is default now
        --dry-run) DRY_RUN=true; shift ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$SEAT" ]]; then SEAT="$1"
            else echo "Too many arguments. Usage: launch-codex-crew.sh <seat-name> [options]" >&2; exit 1
            fi
            shift ;;
    esac
done

if [[ -z "$SEAT" ]]; then
    echo "Usage: scripts/launch-codex-crew.sh <seat-name> [--workspace <ref>] [--exec] [--dry-run]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  scripts/launch-codex-crew.sh cedar --workspace ws:18  # interactive (default)" >&2
    echo "  scripts/launch-codex-crew.sh cedar --exec             # one-shot exec mode" >&2
    exit 1
fi

# --- Resolve repo root (works from any worktree) ---
REPO_ROOT="$(cd "$(git rev-parse --show-toplevel)" && git rev-parse --path-format=absolute --git-common-dir | sed 's|/\.git$||')"
WORKTREES_DIR="$REPO_ROOT/.claude/worktrees"
WORKTREE_NAME="codex-$SEAT"
WORKTREE_PATH="$WORKTREES_DIR/$WORKTREE_NAME"

# --- Create worktree ---
if [[ ! -d "$WORKTREE_PATH" ]]; then
    echo "Creating worktree '$WORKTREE_NAME'..."
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] git worktree add $WORKTREE_PATH origin/main --detach"
    else
        git fetch origin 2>/dev/null || true
        git worktree add "$WORKTREE_PATH" origin/main --detach 2>/dev/null
    fi
else
    echo "Reusing existing worktree '$WORKTREE_NAME'"
    if [[ "$DRY_RUN" != true ]]; then
        (cd "$WORKTREE_PATH" && git fetch origin 2>/dev/null && git reset --hard origin/main 2>/dev/null) || true
    fi
fi

# --- Write seat identity + set up beads redirect ---
if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$WORKTREE_PATH/.beads"
    chmod 700 "$WORKTREE_PATH/.beads" 2>/dev/null || true
    echo "$SEAT" > "$WORKTREE_PATH/.beads/actor"

    # Redirect to main repo's .beads/ so worktree uses the shared Dolt server.
    # Codex sandbox blocks port binding (can't start its own server) but allows
    # client connections — this redirect makes bd commands work transparently.
    echo "$REPO_ROOT/.beads" > "$WORKTREE_PATH/.beads/redirect"
fi
echo "Seat: $SEAT → $WORKTREE_PATH/.beads/actor"

# --- Build the codex prompt ---
PROMPT="You are crew seat '$SEAT' on the hash-thing project (a 3D voxel engine in Rust).

CRITICAL SETUP — run these first:
  export BEADS_ACTOR=$SEAT

IMPORTANT: Use '.bin/bd' wrapper, NOT bare 'bd' — it forces the repo-root shared .beads server, preserves BEADS_ACTOR, and guards bd close.

Read AGENTS.md for full project instructions, then:
  .bin/bd ready -n 10

WORKFLOW — autonomous bead loop:
  1. .bin/bd update <id> --claim
  2. .bin/bd show <id> — read the full description
  3. Implement in Rust
  4. Validate: cargo test && cargo clippy --all-targets --locked -- -D warnings && cargo fmt
  5. Commit (match style from git log --oneline -5)
  6. Land on main: git fetch origin && git rebase origin/main && git push origin HEAD:main
  7. .bin/bd close <id>
  8. Re-poll: .bin/bd ready — take the top item

Work in short chunks (5-30 min). Re-check .bin/bd ready between beads —
a higher-priority bead may have appeared while you were working.

DESIGN GATES — only park for genuine user-visible behavior changes:
- User sees/does something different → park as blocked with a one-line reason
- Implementation details, naming, module layout, on-disk format, refactors → decide yourself
- When in doubt, check the gate-tier rubric in AGENTS.md

WHEN QUEUE IS DRY — don't stop. Scout-tier work:
- grep -rn 'TODO\|FIXME\|XXX' src/ and file new beads
- Improve tests, docs, or crew infra
- Only stop if nothing productive remains after scouting

RULES:
- Never pick beads with status 'blocked' or type 'epic'
- Always validate before committing
- Land every completed bead on origin/main before picking the next"

# Always write the prompt file (used by both modes)
PROMPT_FILE="$WORKTREE_PATH/.codex-crew-prompt.md"
if [[ "$DRY_RUN" != true ]]; then
    printf '%s' "$PROMPT" > "$PROMPT_FILE"
fi

# Write a wrapper script that sets env and launches codex.
# cmux new-workspace --command doesn't expand $(cat) reliably, so we use a
# self-contained script the terminal shell can exec.
LAUNCH_SCRIPT="$WORKTREE_PATH/.codex-launch.sh"
if [[ "$DRY_RUN" != true ]]; then
    if [[ "$EXEC_MODE" == true ]]; then
        cat > "$LAUNCH_SCRIPT" <<LAUNCHER
#!/usr/bin/env bash
export BEADS_ACTOR=$SEAT
export GIT_EDITOR=true
export EDITOR=true
exec codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check "\$(cat .codex-crew-prompt.md)"
LAUNCHER
    else
        cat > "$LAUNCH_SCRIPT" <<LAUNCHER
#!/usr/bin/env bash
export BEADS_ACTOR=$SEAT
export GIT_EDITOR=true
export EDITOR=true
exec codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen "\$(cat .codex-crew-prompt.md)"
LAUNCHER
    fi
    chmod +x "$LAUNCH_SCRIPT"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "[dry-run] Would launch in workspace=${WORKSPACE_REF:-new}"
    echo "[dry-run] Mode: $(if $EXEC_MODE; then echo exec; else echo interactive; fi)"
    exit 0
fi

# --- Launch ---
if [[ -n "$WORKSPACE_REF" ]]; then
    echo "Sending to existing workspace $WORKSPACE_REF..."
    cmux send --workspace "$WORKSPACE_REF" $'\x03'  # Ctrl-C
    sleep 1
    cmux send --workspace "$WORKSPACE_REF" "cd '$WORKTREE_PATH' && bash .codex-launch.sh"
    cmux send-key --workspace "$WORKSPACE_REF" enter
else
    echo "Opening new cmux workspace..."
    cmux new-workspace --cwd "$WORKTREE_PATH" --command "bash .codex-launch.sh"
fi

echo ""
echo "Codex crew seat '$SEAT' launched"
echo "  Worktree: $WORKTREE_PATH"
echo "  Actor:    $SEAT"
echo "  Mode:     $(if $EXEC_MODE; then echo exec; else echo interactive; fi)"
