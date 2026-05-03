# Audience GPU distribution & implications for tuning priority

**Status:** Forward-looking strategic doc. cairn 2026-04-26 (research agent + bead `aud-hw`). Edward direction: "If GPU is mostly an issue with Macs, it could be good to do some research on just the distribution of what people who might play this game will have in terms of rendering capability... it's quite possible that an M1 Mac is just laughably weak compared to most of our audience. And if that's true, then we don't really care about making things fast for a Mac."

## Summary

**Apple M1/M2 integrated sits below the 25th percentile of the active Steam GPU population.** The median Steam gamer has ~3.5–5× the FP32 throughput of M1/M2 base. macOS is 2.35% of Steam users, and within that slice M-series is now dominant. SPEC.md now chooses the low end deliberately: **Apple M1 base / GTX 1650-class is the spec floor, M2 is the daily measurement rig, and median Steam GPUs scale up through quality knobs rather than defining the minimum experience.**

## 1. Integrated vs discrete share

Steam's audience skews heavily **discrete**. The March 2026 Steam Hardware Survey's top-15 GPU list is dominated by NVIDIA discrete parts (RTX 3060, RTX 4060, RTX 4060 Laptop, etc.). Explicit integrated entries are limited to "AMD Radeon Graphics" (~4.3%) and "Intel Iris Xe Graphics" (~1.9%) — roughly **6–10% of Steam systems are integrated-only**. Vendor totals on Steam: ~73% NVIDIA / ~18% AMD / ~8% Intel (mostly iGPUs).

Note this is **survivorship-biased**: when you widen the lens to *all* PCs (not just gaming-active Steam clients), Intel iGPUs lead (~61% of all-PC GPU share Q3 2025). But those machines mostly aren't running 3D games. For an indie 3D voxel raymarcher, **assume 85–90%+ of your Windows/Linux audience has discrete**.

Sources:
- [Steam Hardware Survey](https://store.steampowered.com/hwsurvey/videocard/)
- [AMD Radeon GPU market share — TweakTown](https://www.tweaktown.com/news/103653/amds-radeon-gpu-market-share-is-only-8-according-to-steam/index.html)

## 2. M-series vs typical-player GPU perf

Concrete FP32 throughput:

| GPU | TFLOPS | Typical Steam percentile |
|-----|-------:|--------------------------|
| Apple M1 (8-core GPU) | ~2.6 | bottom ~10–15% |
| Apple M2 (10-core GPU) | ~3.6 | bottom ~25% |
| GTX 1650 | ~3.0 | low (peer to M1/M2 base) |
| GTX 1660 | ~5.0 | low-mid |
| **RTX 3060** (Steam #1) | **~12.7** | **median** |
| RTX 4060 | ~15 | median+ |
| RTX 5070 | ~30+ | high |

Base M1 sits **below GTX 1650** in raw FP32 — i.e., near the bottom 10–15% of the active Steam GPU population. Base M2 (~3.6 TF) is roughly GTX 1650-tier, also below the 25th percentile. The Steam median GPU (RTX 3060-class, ~12 TF) is **3.5–5× faster in raw compute** than M1/M2 base. Unified-memory bandwidth narrows the gap on bandwidth-bound shaders, but for an FP32-heavy raymarcher the ratio is the right ballpark.

Sources:
- [Apple M1 specs — Wikipedia](https://en.wikipedia.org/wiki/Apple_M1)
- [M2 GPU analysis — Tom's Hardware](https://www.tomshardware.com/news/apple-m2-gpu-analysis)
- [M1 Pro/Max vs NVIDIA — AppleInsider](https://appleinsider.com/articles/21/10/19/m1-pro-and-m1-max-gpu-performance-versus-nvidia-and-amd)
- [GTX 1650 FP32 — GPU Monkey](https://www.gpu-monkey.com/en/benchmark-nvidia_geforce_gtx_1650_gddr6-fp32)
- [RTX 3060 FP32 — GPU Monkey](https://www.gpu-monkey.com/en/benchmark-nvidia_geforce_rtx_3060-fp32)
- [M1 vs RTX 3060 — Notebookcheck](https://www.notebookcheck.net/M1-8-Core-GPU-vs-NVIDIA-GeForce-RTX-3060_10552_10960.247598.0.html)

## 3. Steam Hardware Survey snapshot (March 2026)

Top GPUs by share:

| GPU | Share |
|-----|-------|
| RTX 3060 | 4.10% |
| RTX 4060 Laptop | 4.04% |
| RTX 4060 | 3.92% |
| RTX 3050 | 3.14% |
| RTX 5070 | 2.87% |
| GTX 1650 | 2.74% |
| RTX 4060 Ti | 2.50% |
| RTX 5060 | 2.42% |
| RTX 3060 Ti | 2.32% |
| RTX 3070 | 2.19% |

VRAM: 8 GB still leads (~27.5%), 16 GB rising fast (~21.5%), 12 GB ~18.7%. **Median Steam gamer is RTX 3060/4060-class with 8–12 GB VRAM** — comfortably 1080p60 with modest shader work, often 1440p60.

Sources:
- [Steam Hardware Survey](https://store.steampowered.com/hwsurvey/videocard/)
- [March 2026 Steam Survey — PC Guide](https://www.pcguide.com/news/rtx-5070-takes-the-top-spot-in-latest-steam-survey-and-new-vram-scores-are-in-after-valve-fixes-reporting-bug/)

## 4. Mac slice

macOS is **2.35% of Steam users** (March 2026). Within that slice Apple Silicon is now dominant — M4 recently overtook M1 as the most common chip; M2 alone is ~13% of Apple Silicon Steam users. Intel-Mac + AMD-discrete is a shrinking long tail.

**"Mac player" ≈ "M-series integrated"** is now a fair approximation; design for it.

Sources:
- [Linux 5.33% / macOS 2.35% on Steam — VideoCardz](https://videocardz.com/newz/steam-on-linux-reaches-5-33-in-march-steam-survey)
- [M4 dominant Apple Silicon on Steam — AppleInsider](https://appleinsider.com/articles/26/01/12/forget-m1-m4-is-the-dominant-apple-silicon-chip-on-steam)

## 5. Implication for hash-thing tuning priority

> **UPDATED by `hash-thing-3q4a` (2026-05-03).** SPEC.md now sets the spec rig at **10th-percentile Steam ≈ Apple M1 base**, with edward's M2 as the daily dev measurement target. The distribution data above still matters, but it no longer means "optimize for RTX 3060 first." It means the project has deliberately chosen the low end of the distribution as the contract.

SPEC.md's current hardware target maps to roughly a **GTX 1650 / Apple M1 base** floor: 8 GB RAM, 4 physical CPU cores, ~2.5-3 TFLOPS GPU throughput, 1080p with `render_scale <= 0.5` acceptable, and **50-60 FPS felt** on that tier. M2 base is one tier above the floor and is the practical day-to-day measurement rig the crew owns.

**Recommendation:**
- Tune first for the **M1-class spec floor**, not the median Steam GPU. Median RTX 3060/4060-class machines get higher `render_scale`, draw distance, post-FX, and FPS through continuous quality knobs; they do not define the minimum viable experience.
- Treat M1/M2 performance work as first-class when it moves the spec contract. Mac-only work is no longer low-value just because macOS is a small Steam slice; M1-class hardware is the chosen low-end proxy for the whole audience.
- Keep the adaptive/default-resolution path central: `render_scale <= 0.5` on M1-class hardware is part of the contract, not a shame fallback.
- Specific re-prioritization for `docs/perf/render-perf-direction.md`:
  - `9k4w.4` (Apple-specific register-flatten): **restore to P2 or higher**. If it improves M-series occupancy without non-Mac regression, it directly supports the spec rig.
  - `m59h` (async surface acquire, Mac-only): **keep promoted / design-gated, not deferred as low audience leverage**. It remains gated because it is an architectural rewrite, but the gating question is mechanism evidence and design risk, not whether Mac matters.
  - `9k4w-audit` (cheaper-rays already-landed audit): stays P1. It helps all GPU classes and keeps the render roadmap honest.
  - `adp-res` / `pfpn` (adaptive resolution): stays **P1 and central**. Under the new frame it is the explicit path for meeting the M1-class 50-60 FPS contract, not merely a Mac workaround.
  - `render-meas` (256^3+ steady-state captures): stays P1, and should report **M1 + M2** when possible. M2 is the daily rig; M1 is the contract cross-check.

## 6. Audience definition (hash-thing-specific caveat)

This research uses Steam's general gaming audience as a proxy. The hash-thing-specific audience may skew different — voxel/sandbox/sim genres tend to over-index on:
- Lower-end hardware (Minecraft demographics)
- Linux users (slightly tech-y, ~5% of Steam)
- Older hardware (long-tail of 4-year-old GPUs)

A more conservative read: **median hash-thing player is probably GTX 1660 / RTX 3050-class**, not RTX 3060. That reinforces the SPEC.md choice to make M1-class hardware the floor and scale richer visuals upward through quality knobs.

If hash-thing ever ships demographically-targeted (e.g., explicit "runs on potato" indie marketing), the median may pull lower toward GTX 1650 / Iris Xe — closer to M1/M2 territory, raising the priority of M-series optimization. Worth re-running this analysis at distribution-launch or if early-access feedback shows unexpectedly low-end audience.

## Related beads

- `aud-hw` (this research; close after this doc lands)
- `adp-res` (Phase 3 adaptive resolution — primary consumer of this recommendation)
- `2w1u` (parent perceived-FPS bead)
- `9k4w.4` (Apple register flatten — re-prioritized per §5)
- `m59h` (async surface acquire — promoted/design-gated per §5)

## Cross-references

- `docs/perf/render-perf-direction.md` — primary roadmap; re-prioritization recommendations land here.
- `SPEC.md` hardware spec target (8 GB / 4-core / 4 GB discrete VRAM or 8 GB unified / 1080p with `render_scale <= 0.5` / 50-60 FPS felt) — referenced in §5.
- `docs/perf/svdag-perf-paper.md` §3.9–§3.12 — empirical M2 numbers feeding into §2.
