use hash_thing::octree::CellState;
use hash_thing::sim::{World, WorldCoord};
use hash_thing::terrain::materials::{SAND, STONE, WATER};
use hash_thing::terrain::TerrainParams;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    name: String,
    world: WorldCoordName,
    level: Option<u32>,
    scene: Scene,
    rule_set: RuleSet,
    intensity: Intensity,
    regime: Regime,
    backend: Backend,
    generations: usize,
    warmup_generations: Option<usize>,
    seed: u64,
    comparator: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum WorldCoordName {
    Tiny,
    Small,
    Medium,
    Demo,
}

impl WorldCoordName {
    fn level(self) -> u32 {
        match self {
            Self::Tiny => 5,
            Self::Small => 6,
            Self::Medium => 7,
            Self::Demo => 8,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Demo => "demo",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum Scene {
    DefaultTerrain,
    DefaultDemo,
    FactoryConveyor,
}

impl Scene {
    fn as_str(self) -> &'static str {
        match self {
            Self::DefaultTerrain => "default-terrain",
            Self::DefaultDemo => "default-demo",
            Self::FactoryConveyor => "factory-conveyor",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum RuleSet {
    DefaultCa,
}

impl RuleSet {
    fn as_str(self) -> &'static str {
        match self {
            Self::DefaultCa => "default-ca",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum Intensity {
    Idle,
    PassiveActive,
    Cascade,
}

impl Intensity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::PassiveActive => "passive-active",
            Self::Cascade => "cascade",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
enum Regime {
    Saturated,
    Churning,
    NotApplicable,
}

impl Regime {
    fn as_str(self) -> &'static str {
        match self {
            Self::Saturated => "saturated",
            Self::Churning => "churning",
            Self::NotApplicable => "n/a",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum Backend {
    HashlifeRecursive,
    ChunkArray,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Self::HashlifeRecursive => "hashlife-recursive",
            Self::ChunkArray => "chunk-array",
        }
    }
}

#[derive(Deserialize, Serialize)]
struct ConfidenceRecord {
    n: usize,
    warm_frame_policy: String,
    source: String,
    cherry_pick_audit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hard_followup_bead: Option<String>,
    notes: String,
}

#[derive(Deserialize, Serialize)]
struct GenerationRecord {
    gen: usize,
    step_us: u128,
    pop_count: usize,
    drops: usize,
    mat_distribution: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize)]
struct MetricsRecord {
    step_mean_ms: f64,
    step_median_ms: f64,
    step_p95_ms: f64,
    wall_total_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    memo_hit_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    elision_factor_x: Option<f64>,
}

#[derive(Deserialize, Serialize)]
struct MeasurementRecord {
    schema_version: u32,
    record_kind: String,
    measurement_id: String,
    world: String,
    scene: String,
    intensity: String,
    regime: String,
    rule_set: String,
    backend: String,
    hardware: String,
    scenario_hash: String,
    confidence: ConfidenceRecord,
    level: u32,
    side: u64,
    git_commit: String,
    bench_fn: String,
    comparator: Option<String>,
    metrics: MetricsRecord,
    generations: Vec<GenerationRecord>,
}

#[derive(Serialize)]
struct ComparisonRecord {
    schema_version: u32,
    record_kind: &'static str,
    comparison_id: String,
    subject_measurement_id: String,
    baseline_measurement_id: String,
    ratio: f64,
    ratio_metric: String,
    scenario_hash: String,
    rule_set: String,
    notes: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("scenario-runner: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let line = match args.mode {
        Mode::Run { scenario, hardware } => {
            let text = read_utf8(&scenario)?;
            let scenario_record: Scenario = ron::from_str(&text)
                .map_err(|err| format!("parse {}: {err}", scenario.display()))?;
            let record = run_scenario(&scenario, &scenario_record, &hardware)?;
            serde_json::to_string(&record).map_err(|err| format!("serialize record: {err}"))?
        }
        Mode::Compare {
            jsonl,
            subject_id,
            baseline_id,
            metric,
        } => {
            let record = compare_records(&jsonl, &subject_id, &baseline_id, &metric)?;
            serde_json::to_string(&record).map_err(|err| format!("serialize comparison: {err}"))?
        }
    };
    println!("{line}");
    if let Some(append_path) = args.append {
        append_jsonl(&append_path, &line)?;
    }
    Ok(())
}

struct Args {
    mode: Mode,
    append: Option<PathBuf>,
}

enum Mode {
    Run {
        scenario: PathBuf,
        hardware: String,
    },
    Compare {
        jsonl: PathBuf,
        subject_id: String,
        baseline_id: String,
        metric: String,
    },
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args_os().skip(1);
    let first = raw.next().ok_or_else(usage)?;
    let mut append = None;
    let mut hardware = "m2-pro-mbp".to_string();
    let mode = if first == "--compare" {
        let jsonl = raw
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| format!("--compare requires <jsonl>\n{}", usage()))?;
        let subject_id = raw
            .next()
            .map(|s| s.to_string_lossy().into_owned())
            .ok_or_else(|| format!("--compare requires <subject-id>\n{}", usage()))?;
        let baseline_id = raw
            .next()
            .map(|s| s.to_string_lossy().into_owned())
            .ok_or_else(|| format!("--compare requires <baseline-id>\n{}", usage()))?;
        let mut metric = "step_p95_ms".to_string();
        while let Some(arg) = raw.next() {
            if arg == "--metric" {
                metric = raw
                    .next()
                    .map(|s| s.to_string_lossy().into_owned())
                    .ok_or_else(|| "--metric requires a metric name".to_string())?;
            } else if arg == "--append" {
                append = Some(
                    raw.next()
                        .map(PathBuf::from)
                        .ok_or_else(|| "--append requires a path".to_string())?,
                );
            } else {
                return Err(format!("unknown argument: {arg:?}\n{}", usage()));
            }
        }
        Mode::Compare {
            jsonl,
            subject_id,
            baseline_id,
            metric,
        }
    } else {
        let scenario = PathBuf::from(first);
        while let Some(arg) = raw.next() {
            if arg == "--hardware" {
                hardware = raw
                    .next()
                    .map(|s| s.to_string_lossy().into_owned())
                    .ok_or_else(|| "--hardware requires a hardware enum value".to_string())?;
                validate_hardware(&hardware)?;
            } else if arg == "--append" {
                append = Some(
                    raw.next()
                        .map(PathBuf::from)
                        .ok_or_else(|| "--append requires a path".to_string())?,
                );
            } else {
                return Err(format!("unknown argument: {arg:?}\n{}", usage()));
            }
        }
        validate_hardware(&hardware)?;
        Mode::Run { scenario, hardware }
    };
    Ok(Args { mode, append })
}

fn read_utf8(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    String::from_utf8(bytes).map_err(|err| format!("{} is not UTF-8: {err}", path.display()))
}

fn validate_hardware(hardware: &str) -> Result<(), String> {
    match hardware {
        "m2-pro-mbp" | "m2-ultra-mac-pro" | "ci-runner-x86" | "unknown" => Ok(()),
        _ => Err(format!("invalid hardware coordinate {hardware:?}")),
    }
}

fn usage() -> String {
    "usage: cargo run --bin scenario-runner -- <scenario.ron> [--hardware <enum>] [--append <out.jsonl>]\n       cargo run --bin scenario-runner -- --compare <jsonl> <subject-id> <baseline-id> [--metric step_p95_ms] [--append <out.jsonl>]".to_string()
}

fn append_jsonl(path: &Path, line: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("open {}: {err}", path.display()))?;
    writeln!(file, "{line}").map_err(|err| format!("write {}: {err}", path.display()))
}

fn compare_records(
    jsonl: &Path,
    subject_id: &str,
    baseline_id: &str,
    metric: &str,
) -> Result<ComparisonRecord, String> {
    let text = read_utf8(jsonl)?;
    let mut subject = None;
    let mut baseline = None;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|err| format!("parse {} line {}: {err}", jsonl.display(), i + 1))?;
        if value.get("record_kind").and_then(|v| v.as_str()) != Some("measurement") {
            continue;
        }
        let record: MeasurementRecord = serde_json::from_value(value).map_err(|err| {
            format!(
                "parse measurement {} line {}: {err}",
                jsonl.display(),
                i + 1
            )
        })?;
        if record.measurement_id == subject_id {
            subject = Some(record);
        } else if record.measurement_id == baseline_id {
            baseline = Some(record);
        }
    }
    let subject = subject.ok_or_else(|| format!("subject measurement not found: {subject_id}"))?;
    let baseline =
        baseline.ok_or_else(|| format!("baseline measurement not found: {baseline_id}"))?;
    validate_comparable(&subject, &baseline, metric)?;
    let subject_value = metric_value(&subject.metrics, metric)?;
    let baseline_value = metric_value(&baseline.metrics, metric)?;
    if baseline_value == 0.0 {
        return Err(format!("baseline metric {metric} is zero"));
    }
    let ratio = subject_value / baseline_value;
    Ok(ComparisonRecord {
        schema_version: 2,
        record_kind: "comparison",
        comparison_id: format!("{subject_id}-vs-{baseline_id}-{metric}"),
        subject_measurement_id: subject.measurement_id.clone(),
        baseline_measurement_id: baseline.measurement_id.clone(),
        ratio,
        ratio_metric: metric.to_string(),
        scenario_hash: subject.scenario_hash.clone(),
        rule_set: subject.rule_set.clone(),
        notes: format!(
            "{} {}={:.3} vs {} {}={:.3}; ratio={:.3}",
            subject.measurement_id,
            metric,
            subject_value,
            baseline.measurement_id,
            metric,
            baseline_value,
            ratio
        ),
    })
}

fn validate_comparable(
    subject: &MeasurementRecord,
    baseline: &MeasurementRecord,
    metric: &str,
) -> Result<(), String> {
    let checks = [
        (
            "scenario_hash",
            &subject.scenario_hash,
            &baseline.scenario_hash,
        ),
        ("rule_set", &subject.rule_set, &baseline.rule_set),
        ("hardware", &subject.hardware, &baseline.hardware),
    ];
    for (name, a, b) in checks {
        if a != b {
            return Err(format!("comparison mismatch on {name}: {a:?} vs {b:?}"));
        }
    }
    if subject.level != baseline.level || subject.side != baseline.side {
        return Err(format!(
            "comparison mismatch on level/side: l{} {}³ vs l{} {}³",
            subject.level, subject.side, baseline.level, baseline.side
        ));
    }
    metric_value(&subject.metrics, metric)?;
    metric_value(&baseline.metrics, metric)?;
    Ok(())
}

fn metric_value(metrics: &MetricsRecord, metric: &str) -> Result<f64, String> {
    match metric {
        "step_mean_ms" => Ok(metrics.step_mean_ms),
        "step_median_ms" => Ok(metrics.step_median_ms),
        "step_p95_ms" => Ok(metrics.step_p95_ms),
        "wall_total_ms" => Ok(metrics.wall_total_ms),
        "memo_hit_ratio" => metrics
            .memo_hit_ratio
            .ok_or_else(|| "metric memo_hit_ratio missing".to_string()),
        "elision_factor_x" => metrics
            .elision_factor_x
            .ok_or_else(|| "metric elision_factor_x missing".to_string()),
        _ => Err(format!("unknown metric {metric:?}")),
    }
}

fn run_scenario(
    path: &Path,
    scenario: &Scenario,
    hardware: &str,
) -> Result<MeasurementRecord, String> {
    validate_backend_regime(scenario)?;
    let level = scenario.level.unwrap_or_else(|| scenario.world.level());
    let side = 1u64 << level;
    let scenario_hash = scenario_hash(scenario, level);
    let mut world = World::new(level);
    seed_scene(&mut world, scenario)?;
    let warmup_generations = scenario.warmup_generations.unwrap_or(0);

    let (generations, metrics) = match scenario.backend {
        Backend::HashlifeRecursive => run_hashlife(world, warmup_generations, scenario.generations),
        Backend::ChunkArray => run_chunk_array(world, warmup_generations, scenario.generations),
    };

    let hash_suffix = scenario_hash
        .strip_prefix("sha256:")
        .unwrap_or(&scenario_hash)
        .chars()
        .take(8)
        .collect::<String>();

    Ok(MeasurementRecord {
        schema_version: 2,
        record_kind: "measurement".to_string(),
        measurement_id: format!(
            "{}-{}-{}",
            scenario.name,
            scenario.backend.as_str(),
            hash_suffix
        ),
        world: scenario.world.as_str().to_string(),
        scene: scenario.scene.as_str().to_string(),
        intensity: scenario.intensity.as_str().to_string(),
        regime: scenario.regime.as_str().to_string(),
        rule_set: scenario.rule_set.as_str().to_string(),
        backend: scenario.backend.as_str().to_string(),
        hardware: hardware.to_string(),
        scenario_hash,
        confidence: ConfidenceRecord {
            n: scenario.generations,
            warm_frame_policy: warm_frame_policy(warmup_generations),
            source: "scenario-runner".to_string(),
            cherry_pick_audit: cherry_pick_audit(scenario).to_string(),
            hard_followup_bead: hard_followup_bead(scenario),
            notes: confidence_notes(scenario, path, warmup_generations),
        },
        level,
        side,
        git_commit: git_commit(),
        bench_fn: "scenario-runner".to_string(),
        comparator: scenario.comparator.clone(),
        metrics,
        generations,
    })
}

fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn validate_backend_regime(scenario: &Scenario) -> Result<(), String> {
    match scenario.backend {
        Backend::ChunkArray if scenario.regime != Regime::NotApplicable => {
            Err("backend=chunk-array requires regime=\"n/a\"".to_string())
        }
        Backend::HashlifeRecursive if scenario.regime == Regime::NotApplicable => {
            Err("backend=hashlife-recursive requires a memo-cache regime".to_string())
        }
        _ => Ok(()),
    }
}

fn seed_scene(world: &mut World, scenario: &Scenario) -> Result<(), String> {
    match scenario.scene {
        Scene::DefaultTerrain => {
            let params = TerrainParams::for_level(world.level);
            world.seed_terrain(&params).map_err(str::to_string)?;
        }
        Scene::DefaultDemo => {
            let params = TerrainParams::for_level(world.level);
            world.seed_terrain(&params).map_err(str::to_string)?;
            world.seed_water_and_sand();
            if world.side() >= 64 {
                world.seed_demo_spectacle();
            }
        }
        Scene::FactoryConveyor => seed_factory_conveyor_toy(world, scenario.seed),
    }
    Ok(())
}

fn seed_factory_conveyor_toy(world: &mut World, seed: u64) {
    let side = world.side() as i64;
    let y = side / 2;
    let lane_spacing = 4 + (seed % 3) as i64;
    for z in (4..side - 4).step_by(lane_spacing as usize) {
        for x in 4..side - 4 {
            let state = if (x / 4 + z / lane_spacing) % 2 == 0 {
                SAND
            } else {
                WATER
            };
            world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), state);
            if x % 8 == 0 {
                world.set(WorldCoord(x), WorldCoord(y - 1), WorldCoord(z), STONE);
            }
        }
    }
}

fn warm_frame_policy(warmup_generations: usize) -> String {
    if warmup_generations == 0 {
        "all-frames".to_string()
    } else {
        format!("skip-first-{warmup_generations}")
    }
}

fn hard_followup_bead(scenario: &Scenario) -> Option<String> {
    (cherry_pick_audit(scenario) == "easy_only").then(|| "hash-thing-8ppq.1.4".to_string())
}

fn confidence_notes(scenario: &Scenario, path: &Path, warmup_generations: usize) -> String {
    let seed_note = match scenario.backend {
        Backend::ChunkArray => {
            "chunk-array path snapshots a hashlife-seeded world; per-step metrics are comparator data, seed cost is not a chunk-array-native seed benchmark"
        }
        Backend::HashlifeRecursive => "hashlife path times step_recursive",
    };
    format!(
        "scenario={}, path={}, warmup_generations={}, {}",
        scenario.name,
        path.display(),
        warmup_generations,
        seed_note
    )
}

fn run_hashlife(
    mut world: World,
    warmup_generations: usize,
    generations: usize,
) -> (Vec<GenerationRecord>, MetricsRecord) {
    for _ in 0..warmup_generations {
        world.step_recursive();
    }
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    let mut memo_hits = 0u64;
    let mut memo_misses = 0u64;
    for gen in 0..generations {
        let start = Instant::now();
        world.step_recursive();
        let step_us = start.elapsed().as_micros();
        let stats = world.hashlife_stats;
        memo_hits += stats.cache_hits;
        memo_misses += stats.cache_misses;
        let grid = world.flatten();
        times.push(step_us);
        records.push(GenerationRecord {
            gen,
            step_us,
            pop_count: popcount(&grid),
            drops: 0,
            mat_distribution: Some(material_distribution(&grid)),
        });
    }
    let mut metrics = metrics(times);
    let memo_total = memo_hits + memo_misses;
    if memo_total > 0 {
        metrics.memo_hit_ratio = Some(memo_hits as f64 / memo_total as f64);
        metrics.elision_factor_x = Some(memo_total as f64 / (memo_misses + 1) as f64);
    }
    (records, metrics)
}

fn run_chunk_array(
    mut world: World,
    warmup_generations: usize,
    generations: usize,
) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut grid = world.flatten();
    for _ in 0..warmup_generations {
        let next = world.step_grid(&grid);
        grid = next;
        world.generation += 1;
    }
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    for gen in 0..generations {
        let start = Instant::now();
        let next = world.step_grid(&grid);
        let step_us = start.elapsed().as_micros();
        grid = next;
        world.generation += 1;
        times.push(step_us);
        records.push(GenerationRecord {
            gen,
            step_us,
            pop_count: popcount(&grid),
            drops: 0,
            mat_distribution: Some(material_distribution(&grid)),
        });
    }
    (records, metrics(times))
}

fn popcount(grid: &[CellState]) -> usize {
    grid.iter().filter(|&&cell| cell != 0).count()
}

fn material_distribution(grid: &[CellState]) -> serde_json::Value {
    let mut counts = std::collections::BTreeMap::<u16, u64>::new();
    for &cell in grid {
        if cell == 0 {
            continue;
        }
        let material = hash_thing::octree::Cell::from_raw(cell).material();
        *counts.entry(material).or_insert(0) += 1;
    }
    serde_json::json!(counts)
}

fn metrics(mut times_us: Vec<u128>) -> MetricsRecord {
    if times_us.is_empty() {
        return MetricsRecord {
            step_mean_ms: 0.0,
            step_median_ms: 0.0,
            step_p95_ms: 0.0,
            wall_total_ms: 0.0,
            memo_hit_ratio: None,
            elision_factor_x: None,
        };
    }
    let total_us: u128 = times_us.iter().sum();
    times_us.sort_unstable();
    let last = times_us.len() - 1;
    let p95_idx = ((times_us.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(last);
    MetricsRecord {
        step_mean_ms: total_us as f64 / times_us.len() as f64 / 1000.0,
        step_median_ms: times_us[times_us.len() / 2] as f64 / 1000.0,
        step_p95_ms: times_us[p95_idx] as f64 / 1000.0,
        wall_total_ms: total_us as f64 / 1000.0,
        memo_hit_ratio: None,
        elision_factor_x: None,
    }
}

fn scenario_hash(scenario: &Scenario, level: u32) -> String {
    let canonical = format!(
        "v2-B-canonical|world={}|level={level}|scene={}|rule_set={}|intensity={}|generations={}|warmup_generations={}|seed={}",
        scenario.world.as_str(),
        scenario.scene.as_str(),
        scenario.rule_set.as_str(),
        scenario.intensity.as_str(),
        scenario.generations,
        scenario.warmup_generations.unwrap_or(0),
        scenario.seed
    );
    format!("sha256:{}", hex16(canonical.as_bytes()))
}

fn hex16(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn cherry_pick_audit(scenario: &Scenario) -> &'static str {
    match scenario.intensity {
        Intensity::Cascade => "hard_included",
        Intensity::Idle => "easy_only",
        Intensity::PassiveActive => "mixed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_hash_changes_with_seed() {
        let mut a = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        let mut b = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        b.seed = 2;
        assert_ne!(scenario_hash(&a, 5), scenario_hash(&b, 5));
        a.backend = Backend::ChunkArray;
        a.regime = Regime::NotApplicable;
        b = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        assert_eq!(scenario_hash(&a, 5), scenario_hash(&b, 5));
    }

    #[test]
    fn chunk_array_requires_na_regime() {
        let scenario = test_scenario(Backend::ChunkArray, Regime::Saturated);
        assert!(validate_backend_regime(&scenario).is_err());
    }

    fn test_scenario(backend: Backend, regime: Regime) -> Scenario {
        Scenario {
            name: "test".to_string(),
            world: WorldCoordName::Tiny,
            level: None,
            scene: Scene::DefaultTerrain,
            rule_set: RuleSet::DefaultCa,
            intensity: Intensity::Idle,
            regime,
            backend,
            generations: 1,
            warmup_generations: None,
            seed: 1,
            comparator: None,
        }
    }
}
