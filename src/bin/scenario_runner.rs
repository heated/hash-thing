use hash_thing::octree::CellState;
use hash_thing::sim::world::{
    quarantine_atlas_mixed_containment_plan, QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP,
};
use hash_thing::sim::{World, WorldCoord};
use hash_thing::terrain::materials::{SAND, STONE, WATER};
use hash_thing::terrain::TerrainParams;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    setup: Option<ScenarioSetup>,
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
    QuarantineAtlas,
}

impl Scene {
    fn as_str(self) -> &'static str {
        match self {
            Self::DefaultTerrain => "default-terrain",
            Self::DefaultDemo => "default-demo",
            Self::FactoryConveyor => "factory-conveyor",
            Self::QuarantineAtlas => "quarantine-atlas",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
enum ScenarioSetup {
    QuarantineAtlasMixedContainmentV1,
}

impl ScenarioSetup {
    fn as_str(self) -> &'static str {
        match self {
            Self::QuarantineAtlasMixedContainmentV1 => QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP,
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
    Microchurn,
    PassiveActive,
    Cascade,
}

impl Intensity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Microchurn => "microchurn",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    setup: Option<String>,
    confidence: ConfidenceRecord,
    level: u32,
    side: u64,
    git_commit: String,
    bench_fn: String,
    comparator: Option<String>,
    metrics: MetricsRecord,
    generations: Vec<GenerationRecord>,
}

#[derive(Debug, Serialize)]
struct ComparisonRecord {
    schema_version: u32,
    record_kind: &'static str,
    comparison_id: String,
    subject_measurement_id: String,
    baseline_measurement_id: String,
    ratio: f64,
    ratio_metric: String,
    scenario_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    setup: Option<String>,
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
    let mut hardware = "unknown".to_string();
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
    let mut seen_measurement_ids = std::collections::HashSet::new();
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
        if !seen_measurement_ids.insert(record.measurement_id.clone()) {
            return Err(format!(
                "duplicate measurement_id in {}: {}",
                jsonl.display(),
                record.measurement_id
            ));
        }
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
    let drift = generation_drift_note(&subject, &baseline)
        .map(|note| format!("; {note}"))
        .unwrap_or_default();
    Ok(ComparisonRecord {
        schema_version: 2,
        record_kind: "comparison",
        comparison_id: format!("{subject_id}-vs-{baseline_id}-{metric}"),
        subject_measurement_id: subject.measurement_id.clone(),
        baseline_measurement_id: baseline.measurement_id.clone(),
        ratio,
        ratio_metric: metric.to_string(),
        scenario_hash: subject.scenario_hash.clone(),
        setup: subject.setup.clone(),
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
        ) + &drift,
    })
}

fn generation_drift_note(
    subject: &MeasurementRecord,
    baseline: &MeasurementRecord,
) -> Option<String> {
    let last_subject = subject.generations.last()?;
    let last_baseline = baseline.generations.last()?;
    if last_subject.pop_count == last_baseline.pop_count
        && last_subject.mat_distribution == last_baseline.mat_distribution
    {
        return None;
    }
    Some(format!(
        "trajectory caveat: final measured pop/material distribution differ (subject pop={}, baseline pop={}); see hash-thing-neql",
        last_subject.pop_count, last_baseline.pop_count
    ))
}

fn validate_comparable(
    subject: &MeasurementRecord,
    baseline: &MeasurementRecord,
    metric: &str,
) -> Result<(), String> {
    validate_measurement_record(subject)?;
    validate_measurement_record(baseline)?;
    compare_u32(
        "schema_version",
        subject.schema_version,
        baseline.schema_version,
    )?;
    compare_str(
        "scenario_hash",
        &subject.scenario_hash,
        &baseline.scenario_hash,
    )?;
    compare_str("rule_set", &subject.rule_set, &baseline.rule_set)?;
    compare_str("hardware", &subject.hardware, &baseline.hardware)?;
    compare_str("world", &subject.world, &baseline.world)?;
    compare_str("scene", &subject.scene, &baseline.scene)?;
    compare_str("intensity", &subject.intensity, &baseline.intensity)?;
    if subject.setup != baseline.setup {
        return Err(format!(
            "comparison mismatch on setup: {:?} vs {:?}",
            subject.setup, baseline.setup
        ));
    }
    compare_usize("confidence.n", subject.confidence.n, baseline.confidence.n)?;
    compare_str(
        "confidence.warm_frame_policy",
        &subject.confidence.warm_frame_policy,
        &baseline.confidence.warm_frame_policy,
    )?;
    compare_u32("level", subject.level, baseline.level)?;
    compare_u64("side", subject.side, baseline.side)?;
    if subject.backend == baseline.backend {
        return Err(format!(
            "comparison requires distinct backends, got {:?}",
            subject.backend
        ));
    }
    metric_value(&subject.metrics, metric)?;
    metric_value(&baseline.metrics, metric)?;
    Ok(())
}

fn validate_measurement_record(record: &MeasurementRecord) -> Result<(), String> {
    if record.schema_version != 2 {
        return Err(format!(
            "unsupported measurement schema_version {} for {}",
            record.schema_version, record.measurement_id
        ));
    }
    if record.record_kind != "measurement" {
        return Err(format!(
            "record_kind is not measurement: {}",
            record.record_kind
        ));
    }
    if !matches!(
        record.confidence.source.as_str(),
        "bench" | "demo" | "manual" | "spec"
    ) {
        return Err(format!(
            "invalid confidence.source {:?} for {}",
            record.confidence.source, record.measurement_id
        ));
    }
    Ok(())
}

fn compare_str(name: &str, a: &str, b: &str) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("comparison mismatch on {name}: {a:?} vs {b:?}"))
    }
}

fn compare_usize(name: &str, a: usize, b: usize) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("comparison mismatch on {name}: {a} vs {b}"))
    }
}

fn compare_u32(name: &str, a: u32, b: u32) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("comparison mismatch on {name}: {a} vs {b}"))
    }
}

fn compare_u64(name: &str, a: u64, b: u64) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("comparison mismatch on {name}: {a} vs {b}"))
    }
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

    let microchurn = microchurn_sand_per_step(scenario, level);
    let (generations, metrics) = match scenario.backend {
        Backend::HashlifeRecursive => run_hashlife(
            world,
            warmup_generations,
            scenario.generations,
            microchurn,
            scenario.seed,
        ),
        Backend::ChunkArray => run_chunk_array(
            world,
            warmup_generations,
            scenario.generations,
            microchurn,
            scenario.seed,
        ),
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
            "{}-{}-{}-{}-{}",
            scenario.name,
            scenario.backend.as_str(),
            hash_suffix,
            git_commit(),
            run_epoch_millis()
        ),
        world: scenario.world.as_str().to_string(),
        scene: scenario.scene.as_str().to_string(),
        intensity: scenario.intensity.as_str().to_string(),
        regime: scenario.regime.as_str().to_string(),
        rule_set: scenario.rule_set.as_str().to_string(),
        backend: scenario.backend.as_str().to_string(),
        hardware: hardware.to_string(),
        scenario_hash,
        setup: scenario.setup.map(|setup| setup.as_str().to_string()),
        confidence: ConfidenceRecord {
            n: scenario.generations,
            warm_frame_policy: warm_frame_policy(warmup_generations),
            source: "bench".to_string(),
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

fn run_epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
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
        Scene::QuarantineAtlas => seed_quarantine_atlas(world, scenario)?,
    }
    Ok(())
}

fn seed_quarantine_atlas(world: &mut World, scenario: &Scenario) -> Result<(), String> {
    if world.side() < 64 {
        return Err("quarantine-atlas scene requires side >= 64".to_string());
    }
    let layout = world.seed_quarantine_atlas_demo();
    match scenario.setup {
        Some(ScenarioSetup::QuarantineAtlasMixedContainmentV1) => {
            let plan = quarantine_atlas_mixed_containment_plan(layout);
            for (pattern, center) in plan {
                world.apply_quarantine_atlas_pattern(pattern, center);
            }
        }
        None => {}
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

fn microchurn_sand_per_step(scenario: &Scenario, level: u32) -> Option<usize> {
    (matches!(scenario.scene, Scene::DefaultTerrain)
        && matches!(scenario.intensity, Intensity::Microchurn))
    .then_some(if level >= 6 { 8 } else { 4 })
}

fn microchurn_note(microchurn_sand_per_step: Option<usize>, warmup_generations: usize) -> String {
    microchurn_sand_per_step
        .map(|drops| format!("; microchurn injects {drops} sand writes/step after {warmup_generations} warmup generations"))
        .unwrap_or_default()
}

struct Microchurn {
    state: u64,
    side: i64,
    sand_per_step: usize,
}

impl Microchurn {
    fn new(seed: u64, level: u32, sand_per_step: usize) -> Self {
        Self {
            state: 0x9E3779B97F4A7C15 ^ seed,
            side: 1i64 << level,
            sand_per_step,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_drop(&mut self) -> (i64, i64, i64) {
        let x = (self.next_u64() % (self.side as u64 - 4)) as i64 + 2;
        let y = self.side - 4 + (self.next_u64() % 2) as i64;
        let z = (self.next_u64() % (self.side as u64 - 4)) as i64 + 2;
        (x, y, z)
    }

    fn apply_world(&mut self, world: &mut World) {
        for _ in 0..self.sand_per_step {
            let (x, y, z) = self.next_drop();
            world.set(WorldCoord(x), WorldCoord(y), WorldCoord(z), SAND);
        }
    }

    fn apply_grid(&mut self, grid: &mut [CellState]) {
        let side = self.side as usize;
        for _ in 0..self.sand_per_step {
            let (x, y, z) = self.next_drop();
            let (x, y, z) = (x as usize, y as usize, z as usize);
            grid[x + y * side + z * side * side] = SAND;
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
    let microchurn = microchurn_sand_per_step(
        scenario,
        scenario.level.unwrap_or_else(|| scenario.world.level()),
    );
    let seed_note = match scenario.backend {
        Backend::ChunkArray => {
            "chunk-array path snapshots a hashlife-seeded world; per-step metrics are comparator data, seed cost is not a chunk-array-native seed benchmark"
        }
        Backend::HashlifeRecursive => "hashlife path times step_recursive",
    };
    let drift_note = if matches!(scenario.scene, Scene::DefaultDemo)
        && matches!(scenario.intensity, Intensity::Cascade)
    {
        "; same-seed comparator does not by itself prove byte-identical trajectory; inspect mat_distribution/pop_count and hash-thing-neql"
    } else {
        ""
    };
    let setup_note = scenario
        .setup
        .map(|setup| {
            format!(
                "; setup={} (scripted pre-measurement intervention setup; excludes interactive placement/raycast/cache-invalidation cost)",
                setup.as_str()
            )
        })
        .unwrap_or_default();
    format!(
        "scenario={}, path={}, warmup_generations={}, {}",
        scenario.name,
        path.display(),
        warmup_generations,
        seed_note
    ) + &microchurn_note(microchurn, warmup_generations)
        + &setup_note
        + drift_note
}

fn run_hashlife(
    mut world: World,
    warmup_generations: usize,
    generations: usize,
    microchurn_sand_per_step: Option<usize>,
    seed: u64,
) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut microchurn =
        microchurn_sand_per_step.map(|sand| Microchurn::new(seed, world.level, sand));
    for _ in 0..warmup_generations {
        if let Some(churn) = &mut microchurn {
            churn.apply_world(&mut world);
        }
        world.step_recursive();
    }
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    let mut memo_hits = 0u64;
    let mut memo_misses = 0u64;
    for gen in 0..generations {
        let drops = microchurn_sand_per_step.unwrap_or(0);
        if let Some(churn) = &mut microchurn {
            churn.apply_world(&mut world);
        }
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
            drops,
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
    microchurn_sand_per_step: Option<usize>,
    seed: u64,
) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut grid = world.flatten();
    let mut microchurn =
        microchurn_sand_per_step.map(|sand| Microchurn::new(seed, world.level, sand));
    for _ in 0..warmup_generations {
        if let Some(churn) = &mut microchurn {
            churn.apply_grid(&mut grid);
        }
        let next = world.step_grid(&grid);
        grid = next;
        world.generation += 1;
    }
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    for gen in 0..generations {
        let drops = microchurn_sand_per_step.unwrap_or(0);
        if let Some(churn) = &mut microchurn {
            churn.apply_grid(&mut grid);
        }
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
            drops,
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
    format!(
        "sha256:{}",
        hex16(scenario_hash_canonical_input(scenario, level).as_bytes())
    )
}

fn scenario_hash_canonical_input(scenario: &Scenario, level: u32) -> String {
    let setup = scenario
        .setup
        .map(|setup| format!("|setup={}", setup.as_str()))
        .unwrap_or_default();
    format!(
        "v2-B-canonical|world={}|level={level}|scene={}|rule_set={}|intensity={}{}|seed={}",
        scenario.world.as_str(),
        scenario.scene.as_str(),
        scenario.rule_set.as_str(),
        scenario.intensity.as_str(),
        setup,
        scenario.seed
    )
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
        Intensity::Microchurn => "mixed",
        Intensity::PassiveActive => "mixed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

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
    fn scenario_hash_changes_with_setup() {
        let mut raw = test_scenario(Backend::HashlifeRecursive, Regime::Churning);
        raw.scene = Scene::QuarantineAtlas;
        raw.intensity = Intensity::Cascade;
        raw.level = Some(7);
        let mut mixed = test_scenario(Backend::HashlifeRecursive, Regime::Churning);
        mixed.scene = Scene::QuarantineAtlas;
        mixed.intensity = Intensity::Cascade;
        mixed.level = Some(7);
        mixed.setup = Some(ScenarioSetup::QuarantineAtlasMixedContainmentV1);

        assert_ne!(scenario_hash(&raw, 7), scenario_hash(&mixed, 7));
    }

    #[test]
    fn scenario_hash_preserves_no_setup_v2b_inputs() {
        let mut cascade = test_scenario(Backend::HashlifeRecursive, Regime::Churning);
        cascade.name = "cascade-peak".to_string();
        cascade.world = WorldCoordName::Medium;
        cascade.level = Some(7);
        cascade.scene = Scene::DefaultDemo;
        cascade.intensity = Intensity::Cascade;
        cascade.generations = 30;
        cascade.warmup_generations = Some(5);

        assert_eq!(
            scenario_hash(&cascade, 7),
            "sha256:d8ba69a0e324d707",
            "no-setup scenarios must keep historical v2-B hashes"
        );
    }

    #[test]
    fn chunk_array_requires_na_regime() {
        let scenario = test_scenario(Backend::ChunkArray, Regime::Saturated);
        assert!(validate_backend_regime(&scenario).is_err());
    }

    #[test]
    fn microchurn_intensity_selects_synthetic_churn() {
        let mut scenario = test_scenario(Backend::HashlifeRecursive, Regime::Churning);
        scenario.intensity = Intensity::Microchurn;
        assert_eq!(microchurn_sand_per_step(&scenario, 5), Some(4));
        assert_eq!(microchurn_sand_per_step(&scenario, 7), Some(8));

        scenario.intensity = Intensity::PassiveActive;
        assert_eq!(microchurn_sand_per_step(&scenario, 7), None);
    }

    #[test]
    fn compare_records_rejects_duplicate_measurement_ids() {
        let a = test_measurement("same-id", "chunk-array", "n/a", 10.0);
        let b = test_measurement("same-id", "hashlife-recursive", "saturated", 1.0);
        let path = write_jsonl(&[&a, &b]);
        let err = compare_records(&path, "same-id", "missing", "step_p95_ms").unwrap_err();
        assert!(err.contains("duplicate measurement_id"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_emits_ratio_and_drift_caveat() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        let path = write_jsonl(&[&a, &b]);
        let comparison =
            compare_records(&path, "chunk", "hashlife", "step_p95_ms").expect("comparison");
        assert_eq!(comparison.ratio, 5.0);
        assert!(comparison.notes.contains("trajectory caveat"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_mismatched_hardware() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        b.hardware = "m2-ultra-mac-pro".to_string();
        let path = write_jsonl(&[&a, &b]);
        let err = compare_records(&path, "chunk", "hashlife", "step_p95_ms").unwrap_err();
        assert!(err.contains("hardware"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn microchurn_keeps_backends_on_same_trajectory() {
        let level = 7;
        let params = TerrainParams::for_level(level);
        let mut hashlife_world = World::new(level);
        hashlife_world.seed_terrain(&params).unwrap();
        let mut chunk_world = World::new(level);
        chunk_world.seed_terrain(&params).unwrap();

        let (hashlife, _) = run_hashlife(hashlife_world, 1, 4, Some(8), 7);
        let (chunk, _) = run_chunk_array(chunk_world, 1, 4, Some(8), 7);

        assert_eq!(hashlife.len(), chunk.len());
        for (gen, (h, c)) in hashlife.iter().zip(chunk.iter()).enumerate() {
            assert_eq!(h.pop_count, c.pop_count, "pop drift at gen {gen}");
            assert_eq!(
                h.mat_distribution, c.mat_distribution,
                "material drift at gen {gen}"
            );
        }
    }

    #[test]
    fn quarantine_atlas_mixed_setup_keeps_backends_on_same_trajectory() {
        let level = 7;
        let mut hashlife_world = World::new(level);
        let layout = hashlife_world.seed_quarantine_atlas_demo();
        for (pattern, center) in quarantine_atlas_mixed_containment_plan(layout) {
            hashlife_world.apply_quarantine_atlas_pattern(pattern, center);
        }

        let mut chunk_world = World::new(level);
        let layout = chunk_world.seed_quarantine_atlas_demo();
        for (pattern, center) in quarantine_atlas_mixed_containment_plan(layout) {
            chunk_world.apply_quarantine_atlas_pattern(pattern, center);
        }

        let (hashlife, _) = run_hashlife(hashlife_world, 1, 4, None, 1);
        let (chunk, _) = run_chunk_array(chunk_world, 1, 4, None, 1);

        assert_eq!(hashlife.len(), chunk.len());
        for (gen, (h, c)) in hashlife.iter().zip(chunk.iter()).enumerate() {
            assert_eq!(h.pop_count, c.pop_count, "pop drift at gen {gen}");
            assert_eq!(
                h.mat_distribution, c.mat_distribution,
                "material drift at gen {gen}"
            );
        }
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
            setup: None,
            comparator: None,
        }
    }

    fn test_measurement(
        measurement_id: &str,
        backend: &str,
        regime: &str,
        step_p95_ms: f64,
    ) -> MeasurementRecord {
        MeasurementRecord {
            schema_version: 2,
            record_kind: "measurement".to_string(),
            measurement_id: measurement_id.to_string(),
            world: "medium".to_string(),
            scene: "default-demo".to_string(),
            intensity: "cascade".to_string(),
            regime: regime.to_string(),
            rule_set: "default-ca".to_string(),
            backend: backend.to_string(),
            hardware: "m2-pro-mbp".to_string(),
            scenario_hash: "sha256:test".to_string(),
            setup: None,
            confidence: ConfidenceRecord {
                n: 2,
                warm_frame_policy: "skip-first-1".to_string(),
                source: "bench".to_string(),
                cherry_pick_audit: "hard_included".to_string(),
                hard_followup_bead: None,
                notes: "test".to_string(),
            },
            level: 7,
            side: 128,
            git_commit: "test".to_string(),
            bench_fn: "scenario-runner".to_string(),
            comparator: None,
            metrics: MetricsRecord {
                step_mean_ms: step_p95_ms,
                step_median_ms: step_p95_ms,
                step_p95_ms,
                wall_total_ms: step_p95_ms * 2.0,
                memo_hit_ratio: None,
                elision_factor_x: None,
            },
            generations: vec![GenerationRecord {
                gen: 0,
                step_us: (step_p95_ms * 1000.0) as u128,
                pop_count: if backend == "chunk-array" { 11 } else { 10 },
                drops: 0,
                mat_distribution: Some(serde_json::json!({"1": 10})),
            }],
        }
    }

    fn write_jsonl(records: &[&MeasurementRecord]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hash-thing-scenario-runner-test-{}-{}.jsonl",
            std::process::id(),
            TEST_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let body = records
            .iter()
            .map(|record| serde_json::to_string(record).expect("serialize test record"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, format!("{body}\n")).expect("write test jsonl");
        path
    }
}
