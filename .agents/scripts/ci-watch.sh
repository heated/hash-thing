#!/usr/bin/env bash
# Fire-and-forget CI watcher.
# Usage: .agents/scripts/ci-watch.sh <run-id> &
# Exits silently on success/cancelled. Files a P1 bead on failure.
# Mayor reconciles missed runs via .ship-notes/ci-watch.log (gitignored).

set -u
RUN="${1:?run-id required}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
LOG="$REPO_ROOT/.ship-notes/ci-watch.log"
mkdir -p "$(dirname "$LOG")"
echo "$(date -u +%FT%TZ) start run=$RUN" >> "$LOG"

while [ "$(gh run view "$RUN" --json status -q .status 2>/dev/null)" != "completed" ]; do
  sleep 30
done

CONCL=$(gh run view "$RUN" --json conclusion -q .conclusion)
TITLE=$(gh run view "$RUN" --json displayTitle -q .displayTitle)
echo "$(date -u +%FT%TZ) end run=$RUN conclusion=$CONCL" >> "$LOG"

if [ "$CONCL" = "failure" ]; then
  cd "$REPO_ROOT"
  # Dedup: if an open bead already exists for this run, don't file a duplicate
  # (covers re-runs of the same RUN id or mayor-reconcile races; hash-thing-t02q #4).
  # Closed beads are NOT matches — a recurrence means the failure is back.
  DUPE=$(./.bin/bd list --title-contains "CI failure on main run $RUN" --status open 2>/dev/null | grep -oE 'hash-thing-[a-z0-9.]+' | head -1)
  if [ -n "$DUPE" ]; then
    echo "$(date -u +%FT%TZ) dedup run=$RUN existing=$DUPE" >> "$LOG"
    exit 0
  fi
  ./.bin/bd create "CI failure on main run $RUN ($TITLE)" \
    --type bug --priority 1 \
    --description "Fire-and-forget CI watcher landed. Run https://github.com/heated/hash-thing/actions/runs/$RUN failed on \"$TITLE\". First seat to notice sweeps per mayor skill (CI reconciliation). Diagnose via \`gh run view $RUN --log-failed\`."
fi
