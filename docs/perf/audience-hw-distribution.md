# Audience GPU distribution & implications for tuning priority

**Status:** Forward-looking strategic doc. cairn 2026-04-26 (research agent + bead `aud-hw`). Edward direction: "If GPU is mostly an issue with Macs, it could be good to do some research on just the distribution of what people who might play this game will have in terms of rendering capability... it's quite possible that an M1 Mac is just laughably weak compared to most of our audience. And if that's true, then we don't really care about making things fast for a Mac."

## Summary

**Apple M1/M2 integrated sits below the 25th percentile of the active Steam GPU population.** The median Steam gamer has ~3.5–5× the FP32 throughput of M1/M2 base. macOS is 2.35% of Steam users, and within that slice M-series is now dominant. The implication for hash-thing: **set defensive Mac defaults (render_scale=0.5) and direct optimization budget at the median Steam GPU**, not at making the M1/M2 case faster.

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

SPEC.md's minimum-spec target (8GB / 4-core / 2GB-VRAM / 1440p60) maps to roughly a **GTX 1650 / GTX 1060 3GB-class** discrete card (~3 TF). M1 base (2.6 TF, unified memory) is *just below* that line; M2 base (3.6 TF) is *just above*.

**Recommendation:**
- Keep M1/M2 at `render_scale = 0.5` default (tracked as `adp-res` Phase 3 per render-perf-direction.md).
- **Do not over-spend the optimization budget chasing Mac perf.** ~98% of the audience is Windows/Linux on RTX 3060+ class hardware (3–5× M1 compute).
- Tune to the **median Steam GPU first** (RTX 3060 / 4060). M1/M2 = "minimum-spec smoke test" — must run, doesn't have to look pretty.
- Specific re-prioritization for `docs/perf/render-perf-direction.md`:
  - `9k4w.4` (Apple-specific register-flatten): demoted from P2 to P3. Mac-only optimization at <2.5% of audience is low-leverage.
  - `m59h` (async surface acquire, Mac-only): stays at P2 design-gated. Only revisit if `aud-hw`-derived render_scale=0.5 default still doesn't deliver acceptable feel on M1/M2.
  - `9k4w-audit` (cheaper-rays already-landed audit): stays P1. The cheaper-rays optimizations already in the shader help all GPU classes, so confirming they're landed is high-value regardless of audience composition.
  - `adp-res` (adaptive resolution): promoted from P2 to **P1**. With M1/M2 = bottom-25% and median = 5× faster, the per-GPU-class default IS the Mac perf strategy.
  - `render-meas` (256^3+ steady-state captures): stays P1. We still need real numbers at edward's actual problem-scale, but those numbers will inform "when do we stop optimizing" rather than "how much harder to optimize."

## 6. Audience definition (hash-thing-specific caveat)

This research uses Steam's general gaming audience as a proxy. The hash-thing-specific audience may skew different — voxel/sandbox/sim genres tend to over-index on:
- Lower-end hardware (Minecraft demographics)
- Linux users (slightly tech-y, ~5% of Steam)
- Older hardware (long-tail of 4-year-old GPUs)

A more conservative read: **median hash-thing player is probably GTX 1660 / RTX 3050-class**, not RTX 3060. That's still 1.5–2× M1/M2 perf; the qualitative recommendation stands.

If hash-thing ever ships demographically-targeted (e.g., explicit "runs on potato" indie marketing), the median may pull lower toward GTX 1650 / Iris Xe — closer to M1/M2 territory, raising the priority of M-series optimization. Worth re-running this analysis at distribution-launch or if early-access feedback shows unexpectedly low-end audience.

## Related beads

- `aud-hw` (this research; close after this doc lands)
- `adp-res` (Phase 3 adaptive resolution — primary consumer of this recommendation)
- `2w1u` (parent perceived-FPS bead)
- `9k4w.4` (Apple register flatten — re-prioritized per §5)
- `m59h` (async surface acquire — confirmed deferred per §5)

## Cross-references

- `docs/perf/render-perf-direction.md` — primary roadmap; re-prioritization recommendations land here.
- `SPEC.md` minimum-spec line (8GB / 4-core / 2GB-VRAM / 1440p60) — referenced in §5.
- `docs/perf/svdag-perf-paper.md` §3.9–§3.12 — empirical M2 numbers feeding into §2.
