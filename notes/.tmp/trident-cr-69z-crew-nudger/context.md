# CODE REVIEW CONTEXT

Branch: onyx/69z-crew-nudger
Commits: 54d8ca8 feat(crew): crew-nudge supervisor with idle detection + LaunchAgent 48e0c44 plan: 69z crew anti-stop nudger 

## Diff stat
 .bin/com.hash-thing.crew-nudge.plist |  28 ++++
 .bin/crew-nudge                      | 257 +++++++++++++++++++++++++++++------
 2 files changed, 242 insertions(+), 43 deletions(-)

## Full diff
diff --git a/.bin/com.hash-thing.crew-nudge.plist b/.bin/com.hash-thing.crew-nudge.plist
new file mode 100644
index 0000000..55174ee
--- /dev/null
+++ b/.bin/com.hash-thing.crew-nudge.plist
@@ -0,0 +1,28 @@
+<?xml version="1.0" encoding="UTF-8"?>
+<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
+<plist version="1.0">
+<dict>
+    <key>Label</key>
+    <string>com.hash-thing.crew-nudge</string>
+    <key>ProgramArguments</key>
+    <array>
+        <string>/bin/bash</string>
+        <string>/Users/edward/projects/hash-thing/.bin/crew-nudge</string>
+    </array>
+    <key>WorkingDirectory</key>
+    <string>/Users/edward/projects/hash-thing</string>
+    <key>StartInterval</key>
+    <integer>120</integer>
+    <key>RunAtLoad</key>
+    <true/>
+    <key>EnvironmentVariables</key>
+    <dict>
+        <key>PATH</key>
+        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
+    </dict>
+    <key>StandardOutPath</key>
+    <string>/Users/edward/projects/hash-thing/.ship-notes/crew-nudge-launchd.log</string>
+    <key>StandardErrorPath</key>
+    <string>/Users/edward/projects/hash-thing/.ship-notes/crew-nudge-launchd.log</string>
+</dict>
+</plist>
diff --git a/.bin/crew-nudge b/.bin/crew-nudge
index b7c8f57..700879e 100755
--- a/.bin/crew-nudge
+++ b/.bin/crew-nudge
@@ -1,83 +1,254 @@
 #!/usr/bin/env bash
 # Crew anti-stop nudger (hash-thing-69z).
 #
-# Checks each hash-thing crew workspace for idle surfaces and sends a
-# wake prompt. Run via cron every 30-60 minutes, or manually when you
-# notice a seat has gone quiet.
+# LaunchAgent-based supervisor that detects idle cmux crew surfaces and
+# sends `go\n` to wake them. Runs every ~2 minutes via launchd.
 #
 # Usage:
-#   .bin/crew-nudge              # nudge all idle crew surfaces
+#   .bin/crew-nudge              # nudge idle crew surfaces
 #   .bin/crew-nudge --dry-run    # show what would be nudged
-#   .bin/crew-nudge onyx         # nudge only the onyx workspace
+#   .bin/crew-nudge --install    # install LaunchAgent plist
+#   .bin/crew-nudge --uninstall  # uninstall LaunchAgent plist
+#
+# Pause affordance:
+#   touch .crew-paused           # pause all nudging
+#   touch .crew-paused-onyx      # pause nudging for onyx only
+#   rm .crew-paused              # resume
 
 set -euo pipefail
 
+REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
+STATE_DIR="/tmp/crew-nudge-state"
+LOCKFILE="/tmp/crew-nudge.lock"
+LOG="${REPO_ROOT}/.ship-notes/crew-nudge.log"
+PLIST_NAME="com.hash-thing.crew-nudge"
+PLIST_SRC="${REPO_ROOT}/.bin/${PLIST_NAME}.plist"
+PLIST_DST="${HOME}/Library/LaunchAgents/${PLIST_NAME}.plist"
+
+# Cooldowns (seconds)
+DEFAULT_COOLDOWN=300    # 5 min between nudges to same surface
+RATELIMIT_COOLDOWN=900  # 15 min if rate-limited
+MAX_CONSECUTIVE=5       # suppress after N consecutive nudges with no screen change
+
+# Crew seats to check
+SEATS=(flint cairn onyx ember spark codex)
+
 DRY_RUN=false
-TARGET_SEAT=""
+ACTION="nudge"
 
 for arg in "$@"; do
     case "$arg" in
-        --dry-run) DRY_RUN=true ;;
-        *) TARGET_SEAT="$arg" ;;
+        --dry-run)   DRY_RUN=true ;;
+        --install)   ACTION="install" ;;
+        --uninstall) ACTION="uninstall" ;;
     esac
 done
 
-# Crew seats to check (non-mayor working seats + codex)
-SEATS=(flint cairn onyx ember spark codex)
+# --- Install / Uninstall ---
+
+do_install() {
+    if [ ! -f "$PLIST_SRC" ]; then
+        echo "Error: plist not found at $PLIST_SRC"
+        echo "Generate it first (it should be committed alongside this script)."
+        exit 1
+    fi
+    cp "$PLIST_SRC" "$PLIST_DST"
+    launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null || true
+    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
+    echo "Installed and loaded ${PLIST_NAME}"
+}
+
+do_uninstall() {
+    launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null || true
+    rm -f "$PLIST_DST"
+    echo "Uninstalled ${PLIST_NAME}"
+}
+
+case "$ACTION" in
+    install)   do_install; exit 0 ;;
+    uninstall) do_uninstall; exit 0 ;;
+esac
+
+# --- Nudge Logic ---
 
+now=$(date +%s)
+mkdir -p "$STATE_DIR" "$(dirname "$LOG")"
+
+# Lockfile — prevent concurrent runs (macOS-compatible)
+# Clean stale locks older than 5 minutes (script should finish in <30s)
+if [ -d "$LOCKFILE" ]; then
+    lock_age=$(( now - $(stat -f %m "$LOCKFILE" 2>/dev/null || echo "$now") ))
+    if [ "$lock_age" -gt 300 ]; then
+        rmdir "$LOCKFILE" 2>/dev/null || true
+    fi
+fi
+if ! mkdir "$LOCKFILE" 2>/dev/null; then
+    exit 0  # another instance running, silently skip
+fi
+trap 'rmdir "$LOCKFILE" 2>/dev/null' EXIT
+
+# Global pause check
+if [ -f "${REPO_ROOT}/.crew-paused" ]; then
+    exit 0
+fi
 nudge_count=0
 
+log_msg() {
+    local ts
+    ts=$(date '+%Y-%m-%d %H:%M:%S')
+    echo "[$ts] $*" >> "$LOG"
+    if [ "$DRY_RUN" = true ]; then
+        echo "[$ts] $*"
+    fi
+}
+
+# Get last N non-empty lines from screen content
+last_lines() {
+    echo "$1" | sed '/^[[:space:]]*$/d' | tail -5
+}
+
+# Hash screen content for change detection
+screen_hash() {
+    echo "$1" | shasum -a 256 | cut -d' ' -f1
+}
+
+# Check if screen indicates idle state. Returns 0 if idle, 1 if active.
+# Also sets IDLE_REASON and IS_RATELIMITED globals.
+classify_screen() {
+    local screen="$1"
+    IDLE_REASON=""
+    IS_RATELIMITED=false
+
+    local tail_lines
+    tail_lines=$(last_lines "$screen")
+
+    if [ -z "$tail_lines" ]; then
+        return 1  # empty screen, don't nudge
+    fi
+
+    # Rate limit detection — special case, longer cooldown
+    if echo "$tail_lines" | grep -qiE 'rate.?limit|429|overloaded|too many requests'; then
+        IDLE_REASON="rate-limited"
+        IS_RATELIMITED=true
+        return 0
+    fi
+
+    # Agent finished work / waiting for input
+    if echo "$tail_lines" | grep -qiE 'Compacted|compact.*summary'; then
+        IDLE_REASON="post-compact-idle"
+        return 0
+    fi
+
+    # Claude Code waiting for user input (the > prompt at end of output)
+    if echo "$tail_lines" | grep -qE '^\s*>\s*$'; then
+        IDLE_REASON="waiting-at-prompt"
+        return 0
+    fi
+
+    # Shell prompt (agent session exited back to shell)
+    if echo "$tail_lines" | grep -qE '^\$\s*$|^[a-z]+@.*\$\s*$'; then
+        IDLE_REASON="shell-prompt"
+        return 0
+    fi
+
+    # Explicit completion / stop signals
+    if echo "$tail_lines" | grep -qiE 'session ended|exiting|goodbye|task complete'; then
+        IDLE_REASON="session-ended"
+        return 0
+    fi
+
+    # Agent said it's done but didn't pull next work
+    if echo "$tail_lines" | grep -qiE 'standing by|waiting for.*go|no more work|queue.*empty|bd ready.*empty'; then
+        IDLE_REASON="idle-announced"
+        return 0
+    fi
+
+    return 1  # appears active
+}
+
 for seat in "${SEATS[@]}"; do
-    if [ -n "$TARGET_SEAT" ] && [ "$seat" != "$TARGET_SEAT" ]; then
+    # Per-seat pause check
+    if [ -f "${REPO_ROOT}/.crew-paused-${seat}" ]; then
         continue
     fi
 
-    # Find the workspace for this seat
-    ws=$(cmux list-workspaces 2>/dev/null | grep -i "$seat" | head -1 | awk '{print $1}')
+    # Find workspace for this seat (strip leading whitespace and * indicator)
+    ws=$(cmux list-workspaces 2>/dev/null | grep -i "$seat" | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^workspace:/) {print $i; exit}}')
     if [ -z "$ws" ]; then
         continue
     fi
 
-    # List surfaces in the workspace
-    surfaces=$(cmux list-pane-surfaces --workspace "$ws" 2>/dev/null || true)
-    if [ -z "$surfaces" ]; then
+    # Get first surface only (v1: single surface per workspace)
+    surface=$(cmux list-pane-surfaces --workspace "$ws" 2>/dev/null | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^surface:/) {print $i; exit}}')
+    if [ -z "$surface" ]; then
         continue
     fi
 
-    # Check each surface for recent activity via screen content
-    while IFS= read -r line; do
-        surface_ref=$(echo "$line" | awk '{print $1}')
-        if [ -z "$surface_ref" ]; then
-            continue
-        fi
+    # Read screen
+    screen=$(cmux read-screen --surface "$surface" --lines 20 2>/dev/null || true)
+    if [ -z "$screen" ]; then
+        continue  # empty screen, skip
+    fi
 
-        # Read the last few lines of screen content
-        screen=$(cmux read-screen --surface "$surface_ref" --lines 5 2>/dev/null || true)
+    # Classify
+    if ! classify_screen "$screen"; then
+        continue  # active, skip
+    fi
 
-        # Heuristics for "idle": prompt waiting for input, or empty
-        is_idle=false
-        if echo "$screen" | grep -qE '^\$\s*$|^>\s*$|waiting for input|^human:'; then
-            is_idle=true
-        fi
-        # Also check if the screen shows a completed message with no follow-up
-        if echo "$screen" | grep -qE 'SHIP COMPLETE|session ended|exiting|goodbye'; then
-            is_idle=true
+    # --- Cooldown / Dedupe ---
+    state_file="${STATE_DIR}/${seat}.state"
+    current_hash=$(screen_hash "$screen")
+
+    if [ -f "$state_file" ]; then
+        last_nudge_time=$(sed -n '1p' "$state_file")
+        last_hash=$(sed -n '2p' "$state_file")
+        consecutive=$(sed -n '3p' "$state_file")
+        consecutive=${consecutive:-0}
+
+        # Choose cooldown based on state
+        cooldown=$DEFAULT_COOLDOWN
+        if [ "$IS_RATELIMITED" = true ]; then
+            cooldown=$RATELIMIT_COOLDOWN
         fi
 
-        if [ "$is_idle" = true ]; then
-            nudge_count=$((nudge_count + 1))
+        elapsed=$((now - last_nudge_time))
+        if [ "$elapsed" -lt "$cooldown" ]; then
             if [ "$DRY_RUN" = true ]; then
-                echo "[dry-run] Would nudge $seat ($surface_ref)"
-            else
-                # Send a wake prompt
-                cmux send --surface "$surface_ref" "Wake check: bd ready has work. Pull the top bead and /ship it."
-                cmux send --surface "$surface_ref" $'\n'
-                echo "Nudged $seat ($surface_ref)"
+                log_msg "SKIP $seat ($surface): cooldown (${elapsed}s/${cooldown}s) reason=$IDLE_REASON"
             fi
+            continue
         fi
-    done <<< "$surfaces"
+
+        # Check if screen changed since last nudge
+        if [ "$current_hash" = "$last_hash" ]; then
+            consecutive=$((consecutive + 1))
+            if [ "$consecutive" -ge "$MAX_CONSECUTIVE" ]; then
+                log_msg "SUPPRESS $seat ($surface): ${consecutive} consecutive nudges with no screen change — likely stuck"
+                # Update state but don't nudge
+                printf '%s\n%s\n%s\n' "$now" "$current_hash" "$consecutive" > "$state_file"
+                continue
+            fi
+        else
+            consecutive=0
+        fi
+    else
+        consecutive=0
+    fi
+
+    # --- Nudge ---
+    nudge_count=$((nudge_count + 1))
+
+    if [ "$DRY_RUN" = true ]; then
+        log_msg "WOULD-NUDGE $seat ($surface): reason=$IDLE_REASON hash=$current_hash consecutive=$consecutive"
+    else
+        cmux send --surface "$surface" $'go\n'
+        log_msg "NUDGE $seat ($surface): reason=$IDLE_REASON consecutive=$consecutive"
+    fi
+
+    # Persist state
+    printf '%s\n%s\n%s\n' "$now" "$current_hash" "$((consecutive + 1))" > "$state_file"
 done
 
-if [ "$nudge_count" -eq 0 ]; then
-    echo "All crew surfaces appear active (or no matching workspaces found)."
+if [ "$nudge_count" -eq 0 ] && [ "$DRY_RUN" = true ]; then
+    echo "All crew surfaces appear active (or paused/cooled-down)."
 fi
