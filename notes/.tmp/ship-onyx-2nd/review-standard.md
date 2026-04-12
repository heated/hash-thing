# Single Code Review Command

You are a senior software engineer conducting a code review. Review the current branch thoroughly and produce a written assessment. We need your expert perspective to ensure this is professional quality work (without being overengineered). We always want to ensure that Android and iOS will be equally good and, if fixing a bug on one platform, that it does not negatively impact the other platform.

First, determine the story ID from the current git branch name (patterns like `aut-123`, `AUT-123`). If not found, ask the user.

## Phase 0: Branch Setup

Run these in parallel where possible:
1. `git branch --show-current` — confirm branch
2. `git fetch origin && git pull origin [current-branch]` — sync with remote
3. `git fetch origin dev` — ensure dev is current for accurate diffs

Verify branch is up to date before proceeding.

## Phase 1: Understand the Branch

Execute to understand what's being reviewed:
1. `git log --oneline origin/dev..HEAD` — commits on this branch
2. `git show --stat [commit]` for each commit — files changed per commit
3. `git diff origin/dev...HEAD --stat` — this branch's total changes (three dots)

Only review changes in this branch's commits. Ignore untouched code (though you can understand it, to ensure this is the correct solution and/or feature).

## Phase 2: Test the Work

Run sequentially, stop on first failure:
1. `npm run type-check` — TypeScript validation
2. `npm run lint` — code style (errors only)
3. `npm test` — unit tests

Report exact counts of failures and errors. Any test failures, lint errors, or type errors are blockers—communicate this prominently at the top of the review.

## Phase 3: Understand the Changes

Produce a clear assessment of what this branch does:
- Trace the logic flow through the changed code
- Create mermaid diagrams or numbered sequences where helpful
- Reference relevant existing code for context

Output this to console. This helps the reviewer understand what the story actually accomplishes.

## Phase 4: Write the Review

Create `notes/CR-[STORY-ID]-[ModelShortName]-Standard-[YYYYMMDD-HHMM].md` with header:

Use a short model name for the filename (e.g., "Claude", "Codex", "GPT5"). This should match your model identity.

```
## Code Review
- **Date:** [UTC timestamp in ISO 8601 format, e.g., 2025-12-28T18:30:00Z]
- **Model:** [Your model name + exact model ID]
- **Branch:** [branch name]
- **Latest Commit:** [commit hash]
- **Linear Story:** [In format 'AUT-123', if available]
---
```

Conduct a thorough review at two levels:

**Architectural**: Evaluate deep-seated design decisions. Does this solution address the root cause or work around a deeper issue? Are there better patterns? Flag architectural concerns clearly—propose proper solutions while balancing startup pragmatism.

**Tactical**: Evaluate the specific implementation. Consider (non-exhaustive):
- Test coverage adequacy
- Security (auth, data exposure, input validation)
- Performance (re-renders, memory, query efficiency)
- TypeScript quality
- Pattern consistency
- Code clarity and redundancy

Provide general feedback first, then a single consecutively-numbered list organized as:
- **Blockers** — must fix before merge
- **Important** — should fix, not blocking
- **Potential** — nice-to-have or uncertain items

Be specific with file names and line numbers. Flag security or data loss risks as blockers.

## Phase 5: Validation Pass

After completing the initial review, validate each Blocker and Important item:
- Re-read the specific code section
- Verify the issue exists AND is part of this branch's changes
- Test assumptions where practical (run type-check, trace calls)

Update the review file inline (one authoritative list) with:
- ✅ Confirmed
- ❌ ~~Struck through~~ false positives with explanation
- ❓ Uncertain or needs input and/or discussion
- ⬇️ Lower priority, valid but non-blocking

Complete by informing the user of the review file created and summarizing key findings, listing all blockers or important issues in an easy to scan overview. 


---

# CONTEXT

Branch: onyx/2nd-inside-leaf-normal
Bead: hash-thing-2nd
Base: origin/main

## Commits

0d43ec9 fix(render/svdag): inside-leaf origins use exit-face normal (hash-thing-2nd)

## Diff

diff --git a/src/render/svdag.rs b/src/render/svdag.rs
index 6964446..ebef1b0 100644
--- a/src/render/svdag.rs
+++ b/src/render/svdag.rs
@@ -515,14 +515,31 @@ pub mod cpu_trace {
                             (leaf_max[2] - ro_local[2]) * inv_rd[2],
                         ];
                         let tmin_v = [lt1[0].min(lt2[0]), lt1[1].min(lt2[1]), lt1[2].min(lt2[2])];
-                        // Assumes ray origin is outside the leaf. Inside-leaf
-                        // origins pick the nearest back face (all `tmin_v`
-                        // negative → argmax is the least-negative axis);
-                        // tracked as a follow-up to hash-thing-rv4. The
-                        // cascade uses `>=` on both comparators so ties
-                        // break identically to the shader in
-                        // `svdag_raycast.wgsl` — do NOT flip to `>`.
-                        let normal = if tmin_v[0] >= tmin_v[1] && tmin_v[0] >= tmin_v[2] {
+                        let tmax_v = [lt1[0].max(lt2[0]), lt1[1].max(lt2[1]), lt1[2].max(lt2[2])];
+                        // hash-thing-2nd: inside-leaf fallback. When the ray
+                        // origin sits inside the filled voxel, every `tmin_v`
+                        // component is negative — the entry-face picker (argmax
+                        // of `tmin_v`) then returns the nearest BACK face and
+                        // the normal flips inward, shading the voxel as if lit
+                        // from behind. Reachable via deep-zoom orbit camera.
+                        // Fallback: pick the nearest EXIT face (argmin of
+                        // `tmax_v`) and flip the sign so the normal points in
+                        // the direction the ray is heading — the "about to
+                        // emerge" convention. The outer cascade uses `>=` on
+                        // both comparators so ties break identically to the
+                        // shader in `svdag_raycast.wgsl` — do NOT flip to `>`.
+                        // The inner (inside) cascade uses `<=` for the same
+                        // reason: CPU/GPU must break exit-face ties identically.
+                        let inside = tmin_v[0] < 0.0 && tmin_v[1] < 0.0 && tmin_v[2] < 0.0;
+                        let normal = if inside {
+                            if tmax_v[0] <= tmax_v[1] && tmax_v[0] <= tmax_v[2] {
+                                [rd[0].signum(), 0.0, 0.0]
+                            } else if tmax_v[1] <= tmax_v[2] {
+                                [0.0, rd[1].signum(), 0.0]
+                            } else {
+                                [0.0, 0.0, rd[2].signum()]
+                            }
+                        } else if tmin_v[0] >= tmin_v[1] && tmin_v[0] >= tmin_v[2] {
                             [-rd[0].signum(), 0.0, 0.0]
                         } else if tmin_v[1] >= tmin_v[2] {
                             [0.0, -rd[1].signum(), 0.0]
@@ -1689,6 +1706,53 @@ mod tests {
         );
     }
 
+    /// hash-thing-2nd: inside-filled-leaf ray origins (reachable via deep-zoom
+    /// orbit camera) must produce an EXIT-face normal pointing in the direction
+    /// of travel — not the inward back-face normal the pre-2nd code returned.
+    /// Plants the ray origin at the exact center of a single filled voxel at
+    /// (32,32,32) in a level-6 grid and fires along each of the six axes. The
+    /// exit face on each is the one the ray is heading toward, so the normal
+    /// should equal `sign(rd)` (aligned with the direction of travel).
+    #[test]
+    fn leaf_hit_normal_inside_origin_picks_exit_face() {
+        let mut store = NodeStore::new();
+        let mut root = store.empty(6);
+        let mat1 = mat(1);
+        root = store.set_cell(root, 32, 32, 32, mat1);
+        let dag = Svdag::build(&store, root, 6);
+
+        // Exact center of the filled voxel at (32,32,32) — world-space
+        // AABB [0.5, 0.515625]^3, center at 32.5/64 on every axis.
+        let center = [32.5_f32 / 64.0, 32.5 / 64.0, 32.5 / 64.0];
+
+        // (name, rd, expected exit-face normal = sign(rd) on the exit axis).
+        let cases: &[(&str, [f32; 3], [f32; 3])] = &[
+            ("+x exit", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
+            ("-x exit", [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
+            ("+y exit", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
+            ("-y exit", [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]),
+            ("+z exit", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
+            ("-z exit", [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]),
+        ];
+
+        for (name, rd, expected_normal) in cases {
+            let result = cpu_trace::raycast(&dag.nodes, dag.root_level, center, *rd, false);
+            assert_eq!(
+                result.hit_material,
+                Some(mat1 as u32),
+                "{name}: expected inside-leaf hit, got miss (exhausted={})",
+                result.exhausted,
+            );
+            let normal = result.hit_normal.expect("hit must carry a normal");
+            assert_eq!(
+                normal, *expected_normal,
+                "{name}: inside-leaf origin with rd {rd:?} must yield exit-face \
+                 normal {expected_normal:?}, got {normal:?} (pre-2nd returned \
+                 the inward back-face)",
+            );
+        }
+    }
+
     /// hash-thing-rv4: `hit_normal` must be `None` on every non-hit return
     /// path. The CPU trace has four miss/exhausted sites; this test
     /// exercises the two most reachable ones (clean miss, pre-root miss).
diff --git a/src/render/svdag_raycast.wgsl b/src/render/svdag_raycast.wgsl
index 0becc4f..43b39d4 100644
--- a/src/render/svdag_raycast.wgsl
+++ b/src/render/svdag_raycast.wgsl
@@ -245,18 +245,35 @@ fn raycast(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
                     // axes at entry, and the axis holding that max is the
                     // last face the ray crossed to get inside the voxel.
                     //
-                    // Assumes the ray origin is outside the leaf. Inside-leaf
-                    // origins produce an inward-facing normal (all `tmin_v`
-                    // negative → picker returns the nearest back face); the
-                    // orbit camera can reach this case on deep zoom. Tracked
-                    // as a follow-up to hash-thing-rv4. The cascade below
-                    // uses `>=` on both comparators so ties break identically
-                    // to the CPU oracle in svdag.rs — do NOT flip to `>`.
+                    // hash-thing-2nd: inside-leaf fallback. When the ray
+                    // origin sits inside the filled voxel, every `tmin_v`
+                    // component is negative — the entry-face picker then
+                    // returns the nearest back face and the normal flips
+                    // inward, shading the voxel as if lit from behind. The
+                    // orbit camera reaches this case on deep zoom. Fallback:
+                    // pick the nearest exit face (argmin of `tmax_v`) and
+                    // flip the sign so the normal points in the direction
+                    // the ray is heading — the "about to emerge" convention.
+                    // The outer cascade uses `>=` on both comparators so
+                    // entry-face ties break identically to the CPU oracle
+                    // in svdag.rs — do NOT flip to `>`. The inner (inside)
+                    // cascade uses `<=` for the same reason: CPU/GPU must
+                    // break exit-face ties identically.
                     let lt1 = (leaf_min - ro_local) * inv_rd;
                     let lt2 = (leaf_max - ro_local) * inv_rd;
                     let tmin_v = min(lt1, lt2);
+                    let tmax_v = max(lt1, lt2);
                     var normal = vec3<f32>(0.0);
-                    if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {
+                    let inside = tmin_v.x < 0.0 && tmin_v.y < 0.0 && tmin_v.z < 0.0;
+                    if inside {
+                        if tmax_v.x <= tmax_v.y && tmax_v.x <= tmax_v.z {
+                            normal = vec3<f32>(sign(rd.x), 0.0, 0.0);
+                        } else if tmax_v.y <= tmax_v.z {
+                            normal = vec3<f32>(0.0, sign(rd.y), 0.0);
+                        } else {
+                            normal = vec3<f32>(0.0, 0.0, sign(rd.z));
+                        }
+                    } else if tmin_v.x >= tmin_v.y && tmin_v.x >= tmin_v.z {
                         normal = vec3<f32>(-sign(rd.x), 0.0, 0.0);
                     } else if tmin_v.y >= tmin_v.z {
                         normal = vec3<f32>(0.0, -sign(rd.y), 0.0);
