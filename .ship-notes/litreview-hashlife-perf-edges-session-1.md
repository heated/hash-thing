# Litreview session 1 - hashlife performance edges + non-real-time uses

Claimed bead: `hash-thing-jh3c.2`

Scope note: this is a first-pass literature and implementation review, not a benchmark of this repo. Almost all hard numbers below are 2D Life-family numbers; treat relevance to a 3D, edit-heavy, real-time voxel sandbox as an inference unless explicitly marked otherwise.

## Q1. Where hashlife stops being a multiple

- Finding 1: Hashlife's core premise is repeated space-time, not "cellular automata are always faster in trees." Gosper's paper frames the algorithm as recording intermediate computations over a compressed space-time representation, and says the intended use is exploring future place-times of large initial configurations when they are sufficiently repetitious in structure and behavior.
  Source: R. Wm. Gosper, "Exploiting Regularities in Large Cellular Spaces", Physica D 10 (1984), pp. 75-80. PDF: https://gwern.net/doc/cs/cellular-automaton/1984-gosper.pdf

- Finding 2: Continuous real-time display is a known mismatch. LifeWiki says HashLife is generally not suitable for showing a continuous display because the algorithm works asynchronously: different regions are often evolved through different generation counts at a given moment.
  Source: LifeWiki HashLife: https://conwaylife.com/wiki/HashLife

- Finding 3: Highly chaotic patterns are the clearest documented cliff. Golly's own HashLife help says it achieves high speed on patterns with time and/or space regularity, but performs very poorly on highly chaotic patterns and recommends QuickLife for those cases.
  Source: Golly HashLife help: https://golly.sourceforge.io/Help/Algorithms/HashLife.html

- Finding 4: Golly gives operational stall signals: if hash GC reclaims less than about 70% of memory, or more than about ten GCs occur for a single step, further progress is likely extremely slow. It also warns that raising hash memory can help but increases step-size-change work and GC pauses.
  Source: Golly Control menu help: https://golly.sourceforge.io/Help/control.html

- Finding 5: Rokicki's `hlife` README gives a concrete failure case. For rule `2/1`, one live cell to generation 512 took `hlife` 18 seconds, consumed more than 140 MB, and generated more than 6 million nodes; `qlife` took 1.6 seconds and less than 1 MB. The same note explains the reason: the cache stores essentially every other generation of a highly chaotic pattern, and repeated GC can turn cached recursion into exponential recursive work.
  Source: Tomas Rokicki, HLife README: https://tomas.rokicki.com/hlife/

- Finding 6: Dense random Life is bad for hashlife during the active/random period, even in modern comparisons. Rokicki's 2018 "Life Algorithms" comparison used a 4000x4000 random universe at 30% density. At 2K generations, `hlife` took 53.533 s, versus `qlife` 3.804 s and `ulifelib` 2.491 s; `lifelib` was 9.496 s. Hashlife caught up only after the random pattern had mostly stabilized: advancing the same case from 16K to 4M generations took almost no extra time for the hashlife algorithms, while non-hash algorithms either timed out or took more than a minute.
  Source: Tomas Rokicki, "Life Algorithms", Gathering 4 Gardner 13 gift exchange paper, 2018: https://www.gathering4gardner.org/g4g13gift/math/RokickiTomas-GiftExchange-LifeAlgorithms-G4G13.pdf

- Finding 7: The same 2018 paper says hashlife can be much slower for patterns that stay random or high-entropy for long periods, and that its performance is dominated by hashtable efficiency, not primitive cell computation. That matters for this repo because chunk-array dirty tracking and GPU/bit-parallel methods compete directly on cache locality and parallel throughput, while Hashlife trades that for hash-consed indirection and memo cache hits.
  Source: https://www.gathering4gardner.org/g4g13gift/math/RokickiTomas-GiftExchange-LifeAlgorithms-G4G13.pdf

- Finding 8: There is a "small step size" tradeoff. Rokicki says smaller step sizes mitigate memory use but also reduce Hashlife's ability to gallop forward. That maps directly onto real-time simulation: if the app must expose every visible frame or every player edit, the macro-skip benefit is constrained by the output cadence.
  Source: https://www.gathering4gardner.org/g4g13gift/math/RokickiTomas-GiftExchange-LifeAlgorithms-G4G13.pdf

- Finding 9: Alignment and neighborhood assumptions are real but less numerically documented. LifeWiki describes HashLife's reuse around power-of-two quadtree tiles, and notes applicability to outer-totalistic Moore/von Neumann Life-like rules; larger neighborhoods require increasing base tile size. Wikipedia's drawback summary also flags poor alignment to quadtree node boundaries as a memory/time problem, but the strongest primary support found in this session is architectural rather than measured.
  Sources: LifeWiki HashLife: https://conwaylife.com/wiki/HashLife ; Wikipedia Hashlife drawback summary: https://en.wikipedia.org/wiki/Hashlife

- Finding 10: 3D adaptation exists, but as specialized offline simulation, not evidence for a real-time 3D sandbox. Stevens adapted Gosper's algorithm to a three-dimensional kinematic environment by decomposing the environment into interleaved periods with different CA rules and storing global state to decide transitions. That is encouraging for "possible in 3D", but the abstract describes a specific modeling environment and does not claim interactive editing/frame-by-frame rendering performance.
  Source: UWE repository record: https://uwe-repository.worktribe.com/output/988553/adapting-gospers-hashlife-algorithm-for-kinematic-environments

## Q2. Non-real-time hashlife uses

- Use case 1: Far-future behavior/proof support for Life research. Gosper's motivating example is deciding whether apparent puffer trains regularize or fail after long transients. The paper explicitly frames long experimental runs as needed to show unbounded growth or disintegration.
  Implementation/source: Gosper 1984 paper: https://gwern.net/doc/cs/cellular-automaton/1984-gosper.pdf
  Multiple over baseline: not given as a clean multiple in the paper; qualitative claim is ability to explore large future space-times when repetitive.

- Use case 2: Golly/HashLife for enormous structured constructions. LifeWiki reports Paul Rendell's Life Turing machine at generation 6,366,548,773,467,669,985,195,496,000 computed in under 30 seconds on a 2 GHz Core Duo using HashLife in Golly.
  Source: LifeWiki HashLife: https://conwaylife.com/wiki/HashLife
  Multiple over baseline: not given against a tuned baseline, but the generation count is the practical result; ordinary step-by-step simulation is not a serious comparator at that horizon.

- Use case 3: Population growth classification for quadratic-growth patterns. Rokicki used `hlife` to simulate metacatacryst to many trillions of generations in the 2006 Dr. Dobb's article mirror, and LifeWiki says he tracked metacatacryst to 2^130 generations, supporting quadratic-growth evidence.
  Sources: Dr. Dobb's article mirror: https://github.com/mafm/HashLife ; Tomas Rokicki LifeWiki page: https://conwaylife.com/wiki/Tomas_Rokicki
  Multiple over baseline: not reported as a ratio. The practical gain is reaching horizons large enough to distinguish growth law.

- Use case 4: Compressed snapshots and replay/scrubbing of huge Life universes. Rokicki's `hlife` macrocell example generates generation 1,000,000,000 of breeder with population 1,302,084,180,212,404 and writes a macrocell file around 52 KB; the README says RLE would be thousands of terabytes. This is a strong storage/canonicalization use even when the result is not being animated every generation.
  Source: Tomas Rokicki HLife README: https://tomas.rokicki.com/hlife/
  Multiple over baseline: storage comparison is huge but format-specific; useful as qualitative evidence for repeated-structure compression.

- Use case 5: Pattern exploration and visualization at scale. `hdraw` in the `hlife` package exists specifically to browse huge universes, and the README calls out fractal-like patterns such as metacatacryst. This is not "real-time gameplay"; it is offline generation followed by zoom/pan inspection of compressed output.
  Source: https://tomas.rokicki.com/hlife/
  Multiple over baseline: no direct ratio.

- Use case 6: Soup search and census pipelines use HashLife-derived infrastructure as one backend, but not as a universal best engine. apgsearch 4.x introduced lifelib with support for both HashLife and `vlife` containers, plus object detection for high-period oscillators/spaceships and long methuselahs. This supports "offline search/census" as a good adjacent domain, but also shows mature tools mix engines instead of assuming HashLife wins every soup.
  Source: LifeWiki apgsearch: https://conwaylife.com/wiki/Apgsearch
  Multiple over baseline: apgsearch page reports an 8% lifelib speed increase over vlife in one version note, plus later GPU improvements; it does not isolate HashLife as the cause of all search speed.

- Use case 7: Self-replicating programmable constructor / kinematic CA studies. Stevens' 2010 paper record says Hashlife was adapted for a 3D kinematic environment; the abstract and indexed text say recent SRPC simulations that were previously too large had been simulated using HashLife. This is evidence for research-scale simulation of large constructed automata, not for interactive games.
  Sources: https://uwe-repository.worktribe.com/output/988553/adapting-gospers-hashlife-algorithm-for-kinematic-environments ; https://www.srm.org.uk/downloads.html
  Multiple over baseline: no ratio found in session 1.

- Use case 8: Rule-family exploration where repeated structure dominates. Golly's hash-based algorithms support multiple rule families and 256-state automata; Rokicki's 2018 paper says caching primitive computation across many uses helps more complex rules, while warning random patterns can still defeat memory. This suggests a good fit for deterministic, repeated, mostly sparse or periodic rule experiments.
  Sources: Golly HashLife help: https://golly.sourceforge.io/Help/Algorithms/HashLife.html ; Rokicki 2018 paper: https://www.gathering4gardner.org/g4g13gift/math/RokickiTomas-GiftExchange-LifeAlgorithms-G4G13.pdf

## Direct implications for hash-thing

- Real-time edit-heavy 3D CA is exactly where the literature gives the fewest positive numbers. The known cliffs - high entropy, frequent edits, visible frame cadence, cache/GC pressure, poor locality, and parallelization difficulty - are all plausible in this repo's target demo.
- Hashlife is still worth treating as a substrate for offline tools: far-future analysis, compressed deterministic replays, scrubbing, puzzle/search/census tasks, and procedural precomputation where repeated structure is intentionally induced.
- For the main renderer/sim loop, the literature suggests benchmarking Hashlife against the current dirty-chunk or GPU-friendly baseline on repo-shaped scenes, not against naive per-cell stepping. A hashlife win against naive Life is not enough evidence.

## Open questions / things worth a session 2

- Find exact performance numbers from Stevens' 3D kinematic Hashlife implementation or source. The UWE repository record loaded in browser, but direct PDF/content endpoints blocked `curl` during validation.
- Mine `lifelib` and apgsearch implementation docs for when the HashLife container is selected versus `vlife`, and whether any heuristics switch based on entropy/stabilization.
- Search ConwayLife forum threads for abandoned 3D Hashlife attempts or hybrid algorithms that detect chaotic regions and fall back to QuickLife-style stepping.
- Compare the project's actual regimes from `docs/perf/regimes.md` to the literature cases: random dense, structured breeder, glider-stream/QGC-like, and edit-heavy local disturbance.
- Investigate whether macrocell-style canonical snapshots would help this repo's replay/debug tooling even if the live sim stays chunk-array based.
