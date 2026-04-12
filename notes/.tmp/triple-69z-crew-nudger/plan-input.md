# Plan: hash-thing-69z — Crew Anti-Stop Nudger

## Summary

Rewrite `.bin/crew-nudge` as a reliable LaunchAgent-based supervisor. Every ~2 minutes, scan each crew cmux workspace for idle surfaces via `cmux read-screen`, send `go\n` via `cmux send` to wake them. Add `.crew-paused` file convention for human pause affordance.

## Files to modify/create

1. **`.bin/crew-nudge`** — Complete rewrite:
   - Iterate crew workspaces (flint, cairn, onyx, ember, spark, codex) via `cmux list-workspaces`
   - For each: `cmux list-pane-surfaces` → `cmux read-screen --surface <ref> --lines 10`
   - Idle detection heuristics:
     - Agent finished work: screen shows `$` prompt, "Task completed", "no more work", `/compact` output, "Compacted" with no follow-up
     - Failed handoff: screen shows error/traceback then idle prompt
     - Rate limit: screen shows "rate limit", "429", "overloaded"
     - Generic idle: no screen change in last read (compare with previous run's cached content)
   - Skip if `.crew-paused` exists at repo root (human pause)
   - Skip if per-seat pause: `.crew-paused-{seat}` exists
   - Nudge: `cmux send --surface <ref> $'go\n'`
   - `--dry-run` mode for testing
   - Log each nudge to `.ship-notes/crew-nudge.log` (gitignored)

2. **`com.hash-thing.crew-nudge.plist`** — LaunchAgent plist:
   - RunAtLoad, StartInterval 120 (2 min)
   - WorkingDirectory: repo root
   - Log to `.ship-notes/crew-nudge-launchd.log`
   - Will live at `~/Library/LaunchAgents/`

3. **`.ship-notes/crew-nudge-alternatives.md`** — Log of considered alternatives:
   - LaunchAgent + fresh `claude -p --resume` (rejected: we use interactive sessions)
   - agtx/Batty-style dedicated supervisor agent (heavier, future option)
   - `/loop` skill with 30m poll (too slow for 2min cadence, uses context)
   - Cron instead of LaunchAgent (either works; plist is macOS-native and already used for dolt-server)

## Key decisions

- **2-minute interval**: aggressive but cheap (screen reads are fast). Can tune via plist edit.
- **No `-p` invocations**: agreed with edward. `go\n` into existing interactive session is the mechanism.
- **Screen-content heuristic, not timing**: we read what's on screen rather than tracking timestamps. Simpler, no state file needed beyond optional last-screen cache.
- **Pause affordance**: touch `.crew-paused` to pause all, `.crew-paused-{seat}` for one seat. rm to resume.

## Risks

- Screen heuristics may false-positive (nudge a working agent) — mitigated by `go\n` being harmless to an active Claude session.
- Screen heuristics may false-negative (miss an idle state) — iterate on patterns as we observe failures.
- cmux read-screen returns empty for some surfaces (observed in prior testing) — skip empty reads, don't nudge.

## Testing approach

- `--dry-run` mode shows what would be nudged without sending
- Manual babysitting: run once, check output, verify nudges land correctly
- Watch `.ship-notes/crew-nudge.log` after enabling LaunchAgent

## Interview context

Edward's design input: three idle cases (finished-without-pulling-next, failed-handoff, rate-limits), nudge via `cmux send 'go\n'`, detect often (~2min), pause affordance, log alternatives, no `-p`, interactive sessions stay.

## Review tier

Dual (P0 infra, but not invariant-bearing code — it's a bash script supervisor)
