# Dolt-server arch-migration recipe

Procedural doc for re-pointing the launchd-managed `dolt sql-server` after
swapping the host's macOS architecture (most commonly Rosetta x86 → arm64).
Machine-specific to edward's M-series box; not invoked by the autonomous-crew
workflow. See `hash-thing-2ccg` (closed) for the original incident.

## Background

bd 1.0.x talks to a long-running `dolt sql-server` over local TCP. On this
machine the server is **launchd-managed**, not started by `bd` itself, via:

    ~/Library/LaunchAgents/com.hash-thing.dolt-server.plist

The plist hardcodes the dolt binary path in `ProgramArguments[0]`. launchd
reads that absolute path directly — `PATH` is not consulted — so installing a
new dolt under a different prefix (e.g. `/opt/homebrew/bin/dolt` after an
arm64 brew flip) is not enough. The plist itself has to be edited.

bd 1.0.4+ may ship embedded-dolt and remove the external server; in that
world this whole recipe goes away. Check `bd doctor` before sinking time
into a migration.

## Recipe

1. **Verify the new dolt binary.** After the brew migration:

       /opt/homebrew/bin/dolt version

   Should print arm64 (or whatever target arch). The path you confirm here
   is what will go into the plist.

2. **Inspect the current plist.** Look for the `<key>ProgramArguments</key>`
   block; the first string is the binary path:

       plutil -p ~/Library/LaunchAgents/com.hash-thing.dolt-server.plist

3. **Edit `ProgramArguments[0]`.** Replace the old absolute path (e.g.
   `/usr/local/bin/dolt`, `/usr/local/Cellar/dolt/<old>/bin/dolt`) with the
   new one (e.g. `/opt/homebrew/bin/dolt`). `plutil -replace` works, or
   open it in a text editor — it's plain XML.

4. **Reload launchd.** `unload` first to stop the running x86 server, then
   `load` to spawn the new one:

       launchctl unload ~/Library/LaunchAgents/com.hash-thing.dolt-server.plist
       launchctl load ~/Library/LaunchAgents/com.hash-thing.dolt-server.plist

5. **Verify.** Confirm the running server is the new binary:

       ps aux | grep 'dolt sql-server'

   Expect to see the new path (e.g. `/opt/homebrew/bin/dolt sql-server
   --config .../.beads/dolt/config.yaml`). Then sanity-check bd:

       bd ready

   should return without errors.

## Persistent state

Dolt's database lives under `<repo>/.beads/dolt/` and is independent of the
binary path. The binary swap does not touch the database; no migration or
backup is needed for this step.

## When this is moot

- bd ships embedded-dolt (no external server, no plist).
- The plist is removed in favor of a different supervisor.
- The dolt binary becomes a universal binary that doesn't care about arch.

If you notice any of the above, drop this file rather than maintain it.
