Based on the current conversation context, generate a mermaid diagram of whatever is being discussed. Do not ask clarifying questions — just pick the most useful visualization and render it.

1. Generate mermaid syntax based on context
2. Write it to a `.mmd` file in `/tmp/claude/` with a descriptive random name (e.g. `auth-flow-x7k.mmd`) using `printf` via Bash (not the Write tool — avoids edit acceptance prompts)
3. Aim for squarish aspect ratios: prefer `LR` direction for deep trees, break long chains into parallel paths, use subgraphs to add width
4. Run `diagram <file>` to render and open it (must set `dangerouslyDisableSandbox: true` — the sandbox blocks `open` from launching browsers)
