use hash_thing::octree::{Cell, CellState};
use hash_thing::sim::margolus::ConveyorBlockRule;
use hash_thing::sim::world::{
    quarantine_atlas_mixed_containment_plan, WorkElisionStats,
    QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP,
};
use hash_thing::sim::{GameOfLife3D, World, WorldCoord};
use hash_thing::terrain::materials::{METAL, METAL_MATERIAL_ID, SAND, STONE, WATER};
use hash_thing::terrain::TerrainParams;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SOUP_SEARCH_SETUP_V1: &str =
    "SoupSearchV1(tile=16,soup_side=8,density_per_1000=180,rule=445)";
const SOUP_SEARCH_SPARSE_V1: &str =
    "SoupSearchSparseV1(tile=16,soup_side=8,density_per_1000=45,rule=445)";
const SOUP_SEARCH_ALIVE: CellState = Cell::pack(1, 0).raw();

#[derive(Debug, Clone, Deserialize)]
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
    SoupSearch,
}

impl Scene {
    fn as_str(self) -> &'static str {
        match self {
            Self::DefaultTerrain => "default-terrain",
            Self::DefaultDemo => "default-demo",
            Self::FactoryConveyor => "factory-conveyor",
            Self::QuarantineAtlas => "quarantine-atlas",
            Self::SoupSearch => "soup-search",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
enum ScenarioSetup {
    QuarantineAtlasMixedContainmentV1,
    FactoryConveyorRuleV1,
    SoupSearchV1,
    SoupSearchSparseV1,
}

impl ScenarioSetup {
    fn as_str(self) -> &'static str {
        match self {
            Self::QuarantineAtlasMixedContainmentV1 => QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP,
            Self::FactoryConveyorRuleV1 => "FactoryConveyorRuleV1",
            Self::SoupSearchV1 => SOUP_SEARCH_SETUP_V1,
            Self::SoupSearchSparseV1 => SOUP_SEARCH_SPARSE_V1,
        }
    }
}

#[derive(Clone, Copy)]
struct SoupSearchParams {
    setup: &'static str,
    tile: i64,
    soup_side: i64,
    density_per_1000: u64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
enum RuleSet {
    DefaultCa,
    FactoryConveyorV1,
    SoupSearchV1,
}

impl RuleSet {
    fn as_str(self) -> &'static str {
        match self {
            Self::DefaultCa => "default-ca",
            Self::FactoryConveyorV1 => "custom:factory-conveyor-v1",
            Self::SoupSearchV1 => "custom:soup-search-v1",
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
    #[serde(skip_serializing_if = "Option::is_none")]
    work_elision_factor_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    leaf_misses: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    factory_sinked: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    factory_backpressure: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state_hash: Option<String>,
    mat_distribution: Option<serde_json::Value>,
    #[serde(skip)]
    grid: Option<Vec<CellState>>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    work_elision_min_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_elision_mean_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_elision_p05_x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    leaf_misses_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    work_elision_leaf_level: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    factory_sinked_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    factory_backpressure_total: Option<u64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    soup_search: Option<SoupSearchSummary>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct SoupSearchSummary {
    setup: String,
    tile_count: usize,
    survivor_count: usize,
    candidate_stable_count: usize,
    extinct_count: usize,
    tiles: Vec<SoupTileSummary>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct SoupTileSummary {
    tile: [i64; 3],
    final_pop: usize,
    max_pop: usize,
    lifespan_generations: usize,
    survived_window: bool,
    candidate_stable: bool,
    final_state_hash: String,
    pop_history: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct TrajectoryDriftRecord {
    generation_index: Option<usize>,
    subject_gen: Option<usize>,
    baseline_gen: Option<usize>,
    subject_final_pop: usize,
    baseline_final_pop: usize,
    final_material_distribution_equal: bool,
    final_state_hash_equal: bool,
    generation_count_equal: bool,
    subject_final_mat_distribution: Option<serde_json::Value>,
    baseline_final_mat_distribution: Option<serde_json::Value>,
    subject_final_state_hash: Option<String>,
    baseline_final_state_hash: Option<String>,
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
    trajectory_equivalent: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    trajectory_drift: Option<TrajectoryDriftRecord>,
    notes: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("scenario-runner: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let line = run_with_args(std::env::args_os().skip(1))?;
    println!("{line}");
    Ok(())
}

fn run_with_args<I>(raw: I) -> Result<String, String>
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args = parse_args_from(raw)?;
    if let Some(append_path) = &args.append {
        ensure_clean_git_tree_for_append(append_path, args.mode.closure_input_paths())?;
    }
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
            options,
        } => {
            let record = compare_records(&jsonl, &subject_id, &baseline_id, &metric, options)?;
            serde_json::to_string(&record).map_err(|err| format!("serialize comparison: {err}"))?
        }
    };
    if let Some(append_path) = args.append {
        append_jsonl(&append_path, &line)?;
    }
    Ok(line)
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
        options: CompareOptions,
    },
}

impl Mode {
    fn closure_input_paths(&self) -> Vec<&Path> {
        match self {
            Mode::Run { scenario, .. } => vec![scenario.as_path()],
            Mode::Compare { jsonl, .. } => vec![jsonl.as_path()],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompareOptions {
    allow_trajectory_drift: bool,
}

fn parse_args_from<I>(raw: I) -> Result<Args, String>
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let mut raw = raw.into_iter();
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
        let mut options = CompareOptions::default();
        while let Some(arg) = raw.next() {
            if arg == "--metric" {
                metric = raw
                    .next()
                    .map(|s| s.to_string_lossy().into_owned())
                    .ok_or_else(|| "--metric requires a metric name".to_string())?;
            } else if arg == "--allow-trajectory-drift" {
                options.allow_trajectory_drift = true;
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
            options,
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
    "usage: cargo run --bin scenario-runner -- <scenario.ron> [--hardware <enum>] [--append <out.jsonl>]\n       cargo run --bin scenario-runner -- --compare <jsonl> <subject-id> <baseline-id> [--metric step_p95_ms] [--allow-trajectory-drift] [--append <out.jsonl>]".to_string()
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

fn ensure_clean_git_tree_for_append(
    append_path: &Path,
    closure_inputs: Vec<&Path>,
) -> Result<(), String> {
    let allowed = git_status_path(append_path)?;
    let required = closure_inputs
        .into_iter()
        .map(git_tracked_input_path)
        .collect::<Result<Vec<_>, _>>()?;
    for input in &required {
        ensure_git_tracked_path(input)?;
    }
    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|err| format!("check git status before append: {err}"))?;
    if !status.status.success() {
        return Err("check git status before append failed".to_string());
    }
    let dirty = String::from_utf8_lossy(&status.stdout);
    let unexpected_dirty = unexpected_dirty_paths(&dirty, &allowed);
    if unexpected_dirty.is_empty() {
        return Ok(());
    }
    Err(format!(
        "refusing to append closure-grade perf record with dirty paths other than {}: {}",
        allowed,
        unexpected_dirty.join(", ")
    ))
}

fn ensure_git_tracked_path(path: &str) -> Result<(), String> {
    let root = git_root()?;
    let status = Command::new("git")
        .args(["ls-files", "--error-unmatch", "--", path])
        .current_dir(root)
        .output()
        .map_err(|err| format!("check tracked input {path}: {err}"))?;
    if status.status.success() {
        Ok(())
    } else {
        Err(format!(
            "refusing to append closure-grade perf record from untracked input: {path}"
        ))
    }
}

fn unexpected_dirty_paths(status: &str, allowed: &str) -> Vec<String> {
    status
        .lines()
        .filter_map(porcelain_path)
        .filter(|path| path != allowed)
        .collect::<Vec<_>>()
}

fn git_status_path(path: &Path) -> Result<String, String> {
    let root = git_root()?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|err| format!("read cwd before append: {err}"))?
            .join(path)
    };
    let absolute = canonical_status_path(&absolute)?;
    let relative = absolute.strip_prefix(&root).unwrap_or(path);
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn canonical_status_path(path: &Path) -> Result<PathBuf, String> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|err| format!("canonicalize {}: {err}", path.display()));
    }
    if let Some(parent) = path.parent() {
        let parent = parent
            .canonicalize()
            .map_err(|err| format!("canonicalize {}: {err}", parent.display()))?;
        if let Some(name) = path.file_name() {
            return Ok(parent.join(name));
        }
    }
    Ok(path.to_path_buf())
}

fn git_tracked_input_path(path: &Path) -> Result<String, String> {
    if !path.is_absolute() {
        let pathspec = path.to_string_lossy().replace('\\', "/");
        if is_git_tracked_pathspec(&pathspec)? {
            return Ok(pathspec);
        }
    }
    git_status_path(path)
}

fn git_root() -> Result<PathBuf, String> {
    let root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|err| format!("find git root before append: {err}"))?;
    if !root.status.success() {
        return Err("find git root before append failed".to_string());
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&root.stdout).trim().to_string(),
    ))
}

fn is_git_tracked_pathspec(path: &str) -> Result<bool, String> {
    let root = git_root()?;
    let status = Command::new("git")
        .args(["ls-files", "--error-unmatch", "--", path])
        .current_dir(root)
        .output()
        .map_err(|err| format!("check tracked input {path}: {err}"))?;
    Ok(status.status.success())
}

fn porcelain_path(line: &str) -> Option<String> {
    let path = line.get(3..)?.trim();
    Some(
        path.split(" -> ")
            .last()
            .unwrap_or(path)
            .trim_matches('"')
            .to_string(),
    )
}

fn compare_records(
    jsonl: &Path,
    subject_id: &str,
    baseline_id: &str,
    metric: &str,
    options: CompareOptions,
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
    let trajectory_drift = trajectory_drift(&subject, &baseline);
    if trajectory_drift.is_some() && !options.allow_trajectory_drift {
        return Err(trajectory_drift_rejection(&subject, &baseline));
    }
    let subject_value = metric_value(&subject.metrics, metric)?;
    let baseline_value = metric_value(&baseline.metrics, metric)?;
    if baseline_value == 0.0 {
        return Err(format!("baseline metric {metric} is zero"));
    }
    let ratio = subject_value / baseline_value;
    let drift = trajectory_drift
        .as_ref()
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
        trajectory_equivalent: trajectory_drift.is_none(),
        trajectory_drift,
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

fn trajectory_drift(
    subject: &MeasurementRecord,
    baseline: &MeasurementRecord,
) -> Option<TrajectoryDriftRecord> {
    let last_subject = subject.generations.last()?;
    let last_baseline = baseline.generations.last()?;
    let first_drift = subject
        .generations
        .iter()
        .zip(&baseline.generations)
        .enumerate()
        .find(|(_, (subject_gen, baseline_gen))| {
            subject_gen.gen != baseline_gen.gen
                || subject_gen.pop_count != baseline_gen.pop_count
                || subject_gen.mat_distribution != baseline_gen.mat_distribution
                || subject_gen.state_hash != baseline_gen.state_hash
        });
    if first_drift.is_none() && subject.generations.len() == baseline.generations.len() {
        return None;
    }
    let (generation_index, subject_gen, baseline_gen) =
        if let Some((index, (subject_gen, baseline_gen))) = first_drift {
            (Some(index), Some(subject_gen.gen), Some(baseline_gen.gen))
        } else {
            (None, None, None)
        };
    Some(TrajectoryDriftRecord {
        generation_index,
        subject_gen,
        baseline_gen,
        subject_final_pop: last_subject.pop_count,
        baseline_final_pop: last_baseline.pop_count,
        final_material_distribution_equal: last_subject.mat_distribution
            == last_baseline.mat_distribution,
        final_state_hash_equal: last_subject.state_hash == last_baseline.state_hash,
        generation_count_equal: subject.generations.len() == baseline.generations.len(),
        subject_final_mat_distribution: last_subject.mat_distribution.clone(),
        baseline_final_mat_distribution: last_baseline.mat_distribution.clone(),
        subject_final_state_hash: last_subject.state_hash.clone(),
        baseline_final_state_hash: last_baseline.state_hash.clone(),
    })
}

fn trajectory_drift_rejection(subject: &MeasurementRecord, baseline: &MeasurementRecord) -> String {
    let Some(drift) = trajectory_drift(subject, baseline) else {
        return "comparison trajectory is equivalent".to_string();
    };
    format!(
        "comparison trajectory drift between {} and {}: final measured pop/material distribution differ (subject pop={}, baseline pop={}); pass --allow-trajectory-drift to emit an explicit drift comparison",
        subject.measurement_id,
        baseline.measurement_id,
        drift.subject_final_pop,
        drift.baseline_final_pop
    )
}

impl std::fmt::Display for TrajectoryDriftRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
        "trajectory caveat: final measured pop/material/state hash differ (subject pop={}, baseline pop={}); see hash-thing-neql",
            self.subject_final_pop, self.baseline_final_pop
        )
    }
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
    validate_factory_parity(subject, baseline)?;
    validate_soup_search_parity(subject, baseline)?;
    metric_value(&subject.metrics, metric)?;
    metric_value(&baseline.metrics, metric)?;
    Ok(())
}

fn validate_soup_search_parity(
    subject: &MeasurementRecord,
    baseline: &MeasurementRecord,
) -> Result<(), String> {
    if subject.soup_search != baseline.soup_search {
        return Err(format!(
            "comparison mismatch on soup_search summary: {:?} vs {:?}",
            subject.soup_search, baseline.soup_search
        ));
    }
    Ok(())
}

fn validate_factory_parity(
    subject: &MeasurementRecord,
    baseline: &MeasurementRecord,
) -> Result<(), String> {
    compare_optional_u64(
        "metrics.factory_sinked_total",
        subject.metrics.factory_sinked_total,
        baseline.metrics.factory_sinked_total,
    )?;
    compare_optional_u64(
        "metrics.factory_backpressure_total",
        subject.metrics.factory_backpressure_total,
        baseline.metrics.factory_backpressure_total,
    )?;

    let factory_subject = subject.metrics.factory_sinked_total.is_some()
        || subject.metrics.factory_backpressure_total.is_some();
    let factory_baseline = baseline.metrics.factory_sinked_total.is_some()
        || baseline.metrics.factory_backpressure_total.is_some();
    if !(factory_subject || factory_baseline) {
        return Ok(());
    }
    compare_usize(
        "generations.len",
        subject.generations.len(),
        baseline.generations.len(),
    )?;
    for (idx, (subject_gen, baseline_gen)) in subject
        .generations
        .iter()
        .zip(baseline.generations.iter())
        .enumerate()
    {
        compare_optional_u64(
            &format!("generations[{idx}].factory_sinked"),
            subject_gen.factory_sinked,
            baseline_gen.factory_sinked,
        )?;
        compare_optional_u64(
            &format!("generations[{idx}].factory_backpressure"),
            subject_gen.factory_backpressure,
            baseline_gen.factory_backpressure,
        )?;
    }
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
    if record.generations.is_empty() {
        return Err(format!(
            "measurement {} has no generation records",
            record.measurement_id
        ));
    }
    validate_measurement_coordinates(record)?;
    validate_measurement_confidence(record)?;
    validate_measurement_metrics(record)?;
    Ok(())
}

fn validate_measurement_coordinates(record: &MeasurementRecord) -> Result<(), String> {
    match record.backend.as_str() {
        "chunk-array" if record.regime != "n/a" => {
            return Err(format!(
                "backend=chunk-array requires regime=\"n/a\" for {}",
                record.measurement_id
            ));
        }
        "hashlife-recursive" if record.regime == "n/a" => {
            return Err(format!(
                "backend=hashlife-recursive requires a memo-cache regime for {}",
                record.measurement_id
            ));
        }
        "chunk-array" | "hashlife-recursive" => {}
        other => {
            return Err(format!(
                "invalid backend {:?} for {}",
                other, record.measurement_id
            ));
        }
    }
    validate_one_of(
        "regime",
        &record.regime,
        &["saturated", "churning", "n/a"],
        record,
    )?;
    validate_one_of(
        "world",
        &record.world,
        &["tiny", "small", "medium", "demo"],
        record,
    )?;
    validate_one_of(
        "scene",
        &record.scene,
        &[
            "default-terrain",
            "default-demo",
            "factory-conveyor",
            "quarantine-atlas",
            "soup-search",
        ],
        record,
    )?;
    validate_one_of(
        "intensity",
        &record.intensity,
        &["idle", "microchurn", "passive-active", "cascade"],
        record,
    )?;
    validate_one_of(
        "rule_set",
        &record.rule_set,
        &[
            "default-ca",
            "custom:factory-conveyor-v1",
            "custom:soup-search-v1",
        ],
        record,
    )?;
    validate_hardware(&record.hardware)?;
    validate_setup_coordinates(record)?;
    validate_scenario_hash(record)?;
    validate_side(record)?;
    Ok(())
}

fn validate_setup_coordinates(record: &MeasurementRecord) -> Result<(), String> {
    match record.setup.as_deref() {
        Some(QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP) if record.scene != "quarantine-atlas" => {
            Err(format!(
                "setup={} is invalid for scene={} in {}",
                QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP, record.scene, record.measurement_id
            ))
        }
        Some("FactoryConveyorRuleV1") if record.scene != "factory-conveyor" => Err(format!(
            "setup=FactoryConveyorRuleV1 is invalid for scene={} in {}",
            record.scene, record.measurement_id
        )),
        Some(SOUP_SEARCH_SETUP_V1 | SOUP_SEARCH_SPARSE_V1) if record.scene != "soup-search" => {
            Err(format!(
                "setup=SoupSearchV1 is invalid for scene={} in {}",
                record.scene, record.measurement_id
            ))
        }
        Some(QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP) if record.rule_set != "default-ca" => {
            Err(format!(
                "setup={} requires matching rule_set (got {}) in {}",
                QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP, record.rule_set, record.measurement_id
            ))
        }
        Some("FactoryConveyorRuleV1") if record.rule_set != "custom:factory-conveyor-v1" => {
            Err(format!(
                "setup=FactoryConveyorRuleV1 requires matching rule_set (got {}) in {}",
                record.rule_set, record.measurement_id
            ))
        }
        Some(SOUP_SEARCH_SETUP_V1 | SOUP_SEARCH_SPARSE_V1)
            if record.rule_set != "custom:soup-search-v1" =>
        {
            Err(format!(
                "setup=SoupSearchV1 requires matching rule_set (got {}) in {}",
                record.rule_set, record.measurement_id
            ))
        }
        Some(
            QUARANTINE_ATLAS_MIXED_CONTAINMENT_SETUP
            | "FactoryConveyorRuleV1"
            | SOUP_SEARCH_SETUP_V1
            | SOUP_SEARCH_SPARSE_V1,
        ) => Ok(()),
        Some(other) => Err(format!(
            "invalid setup {:?} for {}",
            other, record.measurement_id
        )),
        None if record.scene == "soup-search" => Err(format!(
            "scene=soup-search requires a soup setup in {}",
            record.measurement_id
        )),
        None if record.rule_set == "default-ca" => Ok(()),
        None => Err(format!(
            "rule_set={} requires an explicit setup in {}",
            record.rule_set, record.measurement_id
        )),
    }
}

fn validate_scenario_hash(record: &MeasurementRecord) -> Result<(), String> {
    let Some(suffix) = record.scenario_hash.strip_prefix("sha256:") else {
        return Err(format!(
            "malformed scenario_hash {:?} for {}",
            record.scenario_hash, record.measurement_id
        ));
    };
    if suffix.len() != 16
        || !suffix
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
    {
        return Err(format!(
            "malformed scenario_hash {:?} for {}",
            record.scenario_hash, record.measurement_id
        ));
    }
    Ok(())
}

fn validate_side(record: &MeasurementRecord) -> Result<(), String> {
    let expected = 1u64.checked_shl(record.level).ok_or_else(|| {
        format!(
            "invalid level {} for {}",
            record.level, record.measurement_id
        )
    })?;
    if record.side != expected {
        return Err(format!(
            "side {} does not match level {} for {}",
            record.side, record.level, record.measurement_id
        ));
    }
    Ok(())
}

fn validate_measurement_confidence(record: &MeasurementRecord) -> Result<(), String> {
    if record.confidence.n == 0 {
        return Err(format!(
            "confidence.n must be positive for {}",
            record.measurement_id
        ));
    }
    if record.confidence.warm_frame_policy == "all-frames" {
        return Ok(());
    }
    let Some(skipped) = record
        .confidence
        .warm_frame_policy
        .strip_prefix("skip-first-")
    else {
        return Err(format!(
            "invalid confidence.warm_frame_policy {:?} for {}",
            record.confidence.warm_frame_policy, record.measurement_id
        ));
    };
    if skipped.is_empty() || skipped.parse::<usize>().is_err() {
        return Err(format!(
            "invalid confidence.warm_frame_policy {:?} for {}",
            record.confidence.warm_frame_policy, record.measurement_id
        ));
    }
    Ok(())
}

fn validate_measurement_metrics(record: &MeasurementRecord) -> Result<(), String> {
    for (name, value) in [
        ("metrics.step_mean_ms", record.metrics.step_mean_ms),
        ("metrics.step_median_ms", record.metrics.step_median_ms),
        ("metrics.step_p95_ms", record.metrics.step_p95_ms),
        ("metrics.wall_total_ms", record.metrics.wall_total_ms),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{name} must be finite and non-negative for {}",
                record.measurement_id
            ));
        }
    }
    match record.backend.as_str() {
        "chunk-array" => validate_chunk_array_metrics(record),
        "hashlife-recursive" => validate_hashlife_metrics(record),
        _ => Ok(()),
    }?;
    validate_soup_search_summary(record)
}

fn validate_chunk_array_metrics(record: &MeasurementRecord) -> Result<(), String> {
    if record.metrics.memo_hit_ratio.is_some()
        || record.metrics.elision_factor_x.is_some()
        || record.metrics.work_elision_min_x.is_some()
        || record.metrics.work_elision_mean_x.is_some()
        || record.metrics.work_elision_p05_x.is_some()
        || record.metrics.leaf_misses_mean.is_some()
        || record.metrics.work_elision_leaf_level.is_some()
        || record
            .generations
            .iter()
            .any(|gen| gen.work_elision_factor_x.is_some() || gen.leaf_misses.is_some())
    {
        return Err(format!(
            "chunk-array measurement {} must not include hashlife metrics",
            record.measurement_id
        ));
    }
    Ok(())
}

fn validate_hashlife_metrics(record: &MeasurementRecord) -> Result<(), String> {
    let has_memo_metrics =
        record.metrics.memo_hit_ratio.is_some() && record.metrics.elision_factor_x.is_some();
    let has_all_work_elision_metrics = record.metrics.work_elision_min_x.is_some()
        && record.metrics.work_elision_mean_x.is_some()
        && record.metrics.work_elision_p05_x.is_some()
        && record.metrics.leaf_misses_mean.is_some()
        && record.metrics.work_elision_leaf_level.is_some()
        && record
            .generations
            .iter()
            .all(|gen| gen.work_elision_factor_x.is_some() && gen.leaf_misses.is_some());
    let has_any_work_elision_metrics = record.metrics.work_elision_min_x.is_some()
        || record.metrics.work_elision_mean_x.is_some()
        || record.metrics.work_elision_p05_x.is_some()
        || record.metrics.leaf_misses_mean.is_some()
        || record.metrics.work_elision_leaf_level.is_some()
        || record
            .generations
            .iter()
            .any(|gen| gen.work_elision_factor_x.is_some() || gen.leaf_misses.is_some());
    if !has_memo_metrics {
        return Err(format!(
            "hashlife measurement {} has partial/missing backend-specific metrics",
            record.measurement_id
        ));
    }
    if record.scene == "soup-search" && !has_all_work_elision_metrics {
        return Err(format!(
            "hashlife soup-search measurement {} requires work-elision metrics",
            record.measurement_id
        ));
    }
    if has_any_work_elision_metrics && !has_all_work_elision_metrics {
        return Err(format!(
            "hashlife measurement {} has partial/missing backend-specific metrics",
            record.measurement_id
        ));
    }
    validate_optional_ratio(
        "metrics.memo_hit_ratio",
        record.metrics.memo_hit_ratio,
        record,
    )?;
    validate_optional_non_negative(
        "metrics.elision_factor_x",
        record.metrics.elision_factor_x,
        record,
    )?;
    validate_optional_non_negative(
        "metrics.work_elision_min_x",
        record.metrics.work_elision_min_x,
        record,
    )?;
    validate_optional_non_negative(
        "metrics.work_elision_mean_x",
        record.metrics.work_elision_mean_x,
        record,
    )?;
    validate_optional_non_negative(
        "metrics.work_elision_p05_x",
        record.metrics.work_elision_p05_x,
        record,
    )?;
    validate_optional_non_negative(
        "metrics.leaf_misses_mean",
        record.metrics.leaf_misses_mean,
        record,
    )?;
    Ok(())
}

fn validate_soup_search_summary(record: &MeasurementRecord) -> Result<(), String> {
    let Some(summary) = &record.soup_search else {
        if record.scene == "soup-search" {
            return Err(format!(
                "soup-search measurement {} requires soup_search summary",
                record.measurement_id
            ));
        }
        return Ok(());
    };
    if record.scene != "soup-search" {
        return Err(format!(
            "non-soup measurement {} must not include soup_search summary",
            record.measurement_id
        ));
    }
    if summary.setup != SOUP_SEARCH_SETUP_V1 && summary.setup != SOUP_SEARCH_SPARSE_V1 {
        return Err(format!(
            "invalid soup_search setup {:?} for {}",
            summary.setup, record.measurement_id
        ));
    }
    if summary.tile_count != summary.tiles.len() {
        return Err(format!(
            "soup_search tile_count mismatch for {}",
            record.measurement_id
        ));
    }
    let survivor_count = summary
        .tiles
        .iter()
        .filter(|tile| tile.survived_window)
        .count();
    let candidate_stable_count = summary
        .tiles
        .iter()
        .filter(|tile| tile.candidate_stable)
        .count();
    let extinct_count = summary
        .tiles
        .iter()
        .filter(|tile| tile.final_pop == 0)
        .count();
    if summary.survivor_count != survivor_count
        || summary.candidate_stable_count != candidate_stable_count
        || summary.extinct_count != extinct_count
    {
        return Err(format!(
            "soup_search aggregate counts mismatch for {}",
            record.measurement_id
        ));
    }
    Ok(())
}

fn validate_one_of(
    name: &str,
    value: &str,
    allowed: &[&str],
    record: &MeasurementRecord,
) -> Result<(), String> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "invalid {name} {:?} for {}",
            value, record.measurement_id
        ))
    }
}

fn validate_optional_non_negative(
    name: &str,
    value: Option<f64>,
    record: &MeasurementRecord,
) -> Result<(), String> {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "{name} must be finite and non-negative for {}",
                record.measurement_id
            ));
        }
    }
    Ok(())
}

fn validate_optional_ratio(
    name: &str,
    value: Option<f64>,
    record: &MeasurementRecord,
) -> Result<(), String> {
    if let Some(value) = value {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!(
                "{name} must be a finite 0..=1 ratio for {}",
                record.measurement_id
            ));
        }
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

fn compare_optional_u64(name: &str, a: Option<u64>, b: Option<u64>) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("comparison mismatch on {name}: {a:?} vs {b:?}"))
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
        "work_elision_min_x" => metrics
            .work_elision_min_x
            .ok_or_else(|| "metric work_elision_min_x missing".to_string()),
        "work_elision_mean_x" => metrics
            .work_elision_mean_x
            .ok_or_else(|| "metric work_elision_mean_x missing".to_string()),
        "work_elision_p05_x" => metrics
            .work_elision_p05_x
            .ok_or_else(|| "metric work_elision_p05_x missing".to_string()),
        "leaf_misses_mean" => metrics
            .leaf_misses_mean
            .ok_or_else(|| "metric leaf_misses_mean missing".to_string()),
        "work_elision_leaf_level" => metrics
            .work_elision_leaf_level
            .map(|level| level as f64)
            .ok_or_else(|| "metric work_elision_leaf_level missing".to_string()),
        "factory_sinked_total" => metrics
            .factory_sinked_total
            .map(|count| count as f64)
            .ok_or_else(|| "metric factory_sinked_total missing".to_string()),
        "factory_backpressure_total" => metrics
            .factory_backpressure_total
            .map(|count| count as f64)
            .ok_or_else(|| "metric factory_backpressure_total missing".to_string()),
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
    let factory = FactoryHarness::for_scenario(scenario, side as i64);
    let (generations, metrics) = match scenario.backend {
        Backend::HashlifeRecursive => run_hashlife(
            world,
            warmup_generations,
            scenario.generations,
            microchurn,
            factory,
            scenario.seed,
        ),
        Backend::ChunkArray => run_chunk_array(
            world,
            warmup_generations,
            scenario.generations,
            microchurn,
            factory,
            scenario.seed,
        ),
    };
    let soup_search = soup_search_summary_for(scenario, &generations, side as usize);

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
        soup_search,
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
        _ => validate_setup_scene(scenario),
    }
}

fn validate_setup_scene(scenario: &Scenario) -> Result<(), String> {
    match (scenario.setup, scenario.scene) {
        (Some(ScenarioSetup::QuarantineAtlasMixedContainmentV1), Scene::QuarantineAtlas)
        | (Some(ScenarioSetup::FactoryConveyorRuleV1), Scene::FactoryConveyor)
        | (Some(ScenarioSetup::SoupSearchV1), Scene::SoupSearch)
        | (Some(ScenarioSetup::SoupSearchSparseV1), Scene::SoupSearch)
        | (None, _) => Ok(()),
        (Some(setup), scene) => Err(format!(
            "setup={} is invalid for scene={}",
            setup.as_str(),
            scene.as_str()
        )),
    }?;
    if matches!(scenario.scene, Scene::SoupSearch) && scenario.setup.is_none() {
        return Err(format!(
            "scene=soup-search requires setup={}",
            SOUP_SEARCH_SETUP_V1
        ));
    }
    match (scenario.setup, scenario.rule_set) {
        (Some(ScenarioSetup::FactoryConveyorRuleV1), RuleSet::FactoryConveyorV1)
        | (Some(ScenarioSetup::QuarantineAtlasMixedContainmentV1), RuleSet::DefaultCa)
        | (Some(ScenarioSetup::SoupSearchV1), RuleSet::SoupSearchV1)
        | (Some(ScenarioSetup::SoupSearchSparseV1), RuleSet::SoupSearchV1)
        | (None, RuleSet::DefaultCa) => Ok(()),
        (Some(setup), rule_set) => Err(format!(
            "setup={} requires matching rule_set (got {})",
            setup.as_str(),
            rule_set.as_str()
        )),
        (None, rule_set) => Err(format!(
            "rule_set={} requires an explicit setup",
            rule_set.as_str()
        )),
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
        Scene::FactoryConveyor => seed_factory_conveyor(world, scenario)?,
        Scene::QuarantineAtlas => seed_quarantine_atlas(world, scenario)?,
        Scene::SoupSearch => seed_soup_search(world, scenario)?,
    }
    Ok(())
}

fn seed_soup_search(world: &mut World, scenario: &Scenario) -> Result<(), String> {
    match scenario.setup {
        Some(ScenarioSetup::SoupSearchV1 | ScenarioSetup::SoupSearchSparseV1) => {}
        None => {
            return Err("soup-search scene requires setup=SoupSearchV1".to_string());
        }
        Some(other) => {
            return Err(format!(
                "setup={} is invalid for soup-search",
                other.as_str()
            ));
        }
    }
    if world.side() < 32 {
        return Err("soup-search scene requires side >= 32".to_string());
    }

    world.set_gol_smoke_rule(GameOfLife3D::rule445());
    let side = world.side() as i64;
    let params = soup_search_params(scenario.setup.expect("validated soup setup"));
    let margin = (params.tile - params.soup_side) / 2;
    let mut rng = SoupSearchRng::new(scenario.seed);
    for tile_z in 0..side / params.tile {
        for tile_y in 0..side / params.tile {
            for tile_x in 0..side / params.tile {
                let origin = [
                    tile_x * params.tile + margin,
                    tile_y * params.tile + margin,
                    tile_z * params.tile + margin,
                ];
                for dz in 0..params.soup_side {
                    for dy in 0..params.soup_side {
                        for dx in 0..params.soup_side {
                            if rng.next_mod(1000) < params.density_per_1000 {
                                world.set(
                                    WorldCoord(origin[0] + dx),
                                    WorldCoord(origin[1] + dy),
                                    WorldCoord(origin[2] + dz),
                                    SOUP_SEARCH_ALIVE,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn soup_search_params(setup: ScenarioSetup) -> SoupSearchParams {
    match setup {
        ScenarioSetup::SoupSearchV1 => SoupSearchParams {
            setup: SOUP_SEARCH_SETUP_V1,
            tile: 16,
            soup_side: 8,
            density_per_1000: 180,
        },
        ScenarioSetup::SoupSearchSparseV1 => SoupSearchParams {
            setup: SOUP_SEARCH_SPARSE_V1,
            tile: 16,
            soup_side: 8,
            density_per_1000: 45,
        },
        other => panic!("invalid soup-search setup: {}", other.as_str()),
    }
}

struct SoupSearchRng {
    state: u64,
}

impl SoupSearchRng {
    fn new(seed: u64) -> Self {
        let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
        if state == 0 {
            state = 0xA076_1D64_78BD_642F;
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 7;
        self.state ^= self.state >> 9;
        self.state = self.state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.state
    }

    fn next_mod(&mut self, modulo: u64) -> u64 {
        self.next_u64() % modulo
    }
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
        Some(other) => {
            return Err(format!(
                "setup={} is invalid for quarantine-atlas",
                other.as_str()
            ));
        }
    }
    Ok(())
}

fn seed_factory_conveyor(world: &mut World, scenario: &Scenario) -> Result<(), String> {
    match scenario.setup {
        Some(ScenarioSetup::FactoryConveyorRuleV1) => {
            world.mutate_materials(|materials| {
                let conveyor =
                    materials.register_block_rule(ConveyorBlockRule::new(METAL_MATERIAL_ID));
                materials.assign_block_rule(METAL_MATERIAL_ID, conveyor);
            });
            seed_factory_conveyor_rule(world);
        }
        None => seed_factory_conveyor_toy(world, scenario.seed),
        Some(other) => {
            return Err(format!(
                "setup={} is invalid for factory-conveyor",
                other.as_str()
            ));
        }
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

fn seed_factory_conveyor_rule(world: &mut World) {
    let side = world.side() as i64;
    let harness = FactoryHarness::new(side);
    for &z in &harness.lane_z {
        for x in (harness.source_x + 2..harness.sink_x).step_by(4) {
            world.set(WorldCoord(x), WorldCoord(harness.y), WorldCoord(z), METAL);
        }
    }
}

#[derive(Clone)]
struct FactoryHarness {
    source_x: i64,
    sink_x: i64,
    y: i64,
    lane_z: Vec<i64>,
}

#[derive(Clone, Copy, Default)]
struct FactoryStepStats {
    sinked: u64,
    backpressure: u64,
}

impl FactoryHarness {
    fn new(side: i64) -> Self {
        let y = side / 2;
        let lanes = ((side - 8) / 8).max(1);
        let start_z = 4;
        let lane_z = (0..lanes).map(|i| start_z + i * 8).collect::<Vec<_>>();
        Self {
            source_x: 2,
            sink_x: side - 3,
            y,
            lane_z,
        }
    }

    fn for_scenario(scenario: &Scenario, side: i64) -> Option<Self> {
        (matches!(scenario.setup, Some(ScenarioSetup::FactoryConveyorRuleV1)))
            .then(|| Self::new(side))
    }

    fn apply_sources_world(&self, world: &mut World) -> u64 {
        let mut backpressure = 0;
        for &z in &self.lane_z {
            if world.get(WorldCoord(self.source_x), WorldCoord(self.y), WorldCoord(z)) == 0 {
                world.set(
                    WorldCoord(self.source_x),
                    WorldCoord(self.y),
                    WorldCoord(z),
                    METAL,
                );
            } else {
                backpressure += 1;
            }
        }
        backpressure
    }

    fn drain_sinks_world(&self, world: &mut World) -> u64 {
        let mut sinked = 0;
        for &z in &self.lane_z {
            if world.get(WorldCoord(self.sink_x), WorldCoord(self.y), WorldCoord(z)) == METAL {
                world.set(
                    WorldCoord(self.sink_x),
                    WorldCoord(self.y),
                    WorldCoord(z),
                    0,
                );
                sinked += 1;
            }
        }
        sinked
    }

    fn apply_sources_grid(&self, grid: &mut [CellState], side: usize) -> u64 {
        let mut backpressure = 0;
        for &z in &self.lane_z {
            let idx = self.idx(side, self.source_x, self.y, z);
            if grid[idx] == 0 {
                grid[idx] = METAL;
            } else {
                backpressure += 1;
            }
        }
        backpressure
    }

    fn drain_sinks_grid(&self, grid: &mut [CellState], side: usize) -> u64 {
        let mut sinked = 0;
        for &z in &self.lane_z {
            let idx = self.idx(side, self.sink_x, self.y, z);
            if grid[idx] == METAL {
                grid[idx] = 0;
                sinked += 1;
            }
        }
        sinked
    }

    fn idx(&self, side: usize, x: i64, y: i64, z: i64) -> usize {
        let (x, y, z) = (x as usize, y as usize, z as usize);
        x + y * side + z * side * side
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
    let setup_note = match scenario.setup {
        Some(ScenarioSetup::FactoryConveyorRuleV1) => {
            "; setup=factory-conveyor-rule-v1 (scripted pre-measurement intervention setup plus per-step source/sink harness; step_us/step_* time only CA stepping and exclude source injection, sink drain, and backpressure accounting)".to_string()
        }
        Some(ScenarioSetup::SoupSearchV1) => {
            "; setup=soup-search-v1 (deterministic tiled 3D GoL soup ensemble; this first probe measures the search workload, while survivor classification is tracked in hash-thing-8ppq.5.2)".to_string()
        }
        Some(setup) => format!(
            "; setup={} (scripted pre-measurement intervention setup; excludes interactive placement/raycast/cache-invalidation cost)",
            setup.as_str()
        ),
        None => String::new(),
    };
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
    factory: Option<FactoryHarness>,
    seed: u64,
) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut microchurn =
        microchurn_sand_per_step.map(|sand| Microchurn::new(seed, world.level, sand));
    for _ in 0..warmup_generations {
        if let Some(churn) = &mut microchurn {
            churn.apply_world(&mut world);
        }
        if let Some(factory) = &factory {
            factory.apply_sources_world(&mut world);
        }
        world.step_recursive();
        if let Some(factory) = &factory {
            factory.drain_sinks_world(&mut world);
        }
    }
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    let mut memo_hits = 0u64;
    let mut memo_misses = 0u64;
    let mut work_elision = Vec::with_capacity(generations);
    let mut factory_total = FactoryStepStats::default();
    for gen in 0..generations {
        let drops = microchurn_sand_per_step.unwrap_or(0);
        if let Some(churn) = &mut microchurn {
            churn.apply_world(&mut world);
        }
        let mut factory_step = FactoryStepStats::default();
        if let Some(factory) = &factory {
            factory_step.backpressure = factory.apply_sources_world(&mut world);
        }
        let start = Instant::now();
        world.step_recursive();
        let step_us = start.elapsed().as_micros();
        if let Some(factory) = &factory {
            factory_step.sinked = factory.drain_sinks_world(&mut world);
        }
        let stats = world.hashlife_stats;
        let elision_stats = world.work_elision_stats();
        memo_hits += stats.cache_hits;
        memo_misses += stats.cache_misses;
        work_elision.push(elision_stats);
        let grid = world.flatten();
        times.push(step_us);
        records.push(GenerationRecord {
            gen,
            step_us,
            pop_count: popcount(&grid),
            drops,
            work_elision_factor_x: Some(elision_stats.factor_x),
            leaf_misses: Some(elision_stats.leaf_misses),
            factory_sinked: factory.as_ref().map(|_| factory_step.sinked),
            factory_backpressure: factory.as_ref().map(|_| factory_step.backpressure),
            state_hash: Some(grid_hash(&grid)),
            mat_distribution: Some(material_distribution(&grid)),
            grid: Some(grid),
        });
        factory_total.sinked += factory_step.sinked;
        factory_total.backpressure += factory_step.backpressure;
    }
    let mut metrics = metrics(times);
    let memo_total = memo_hits + memo_misses;
    if memo_total > 0 {
        metrics.memo_hit_ratio = Some(memo_hits as f64 / memo_total as f64);
        metrics.elision_factor_x = Some(memo_total as f64 / (memo_misses + 1) as f64);
    }
    apply_work_elision_metrics(&mut metrics, &work_elision);
    apply_factory_metrics(&mut metrics, factory, factory_total);
    (records, metrics)
}

fn run_chunk_array(
    mut world: World,
    warmup_generations: usize,
    generations: usize,
    microchurn_sand_per_step: Option<usize>,
    factory: Option<FactoryHarness>,
    seed: u64,
) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut grid = world.flatten();
    let mut microchurn =
        microchurn_sand_per_step.map(|sand| Microchurn::new(seed, world.level, sand));
    let side = world.side() as usize;
    for _ in 0..warmup_generations {
        if let Some(churn) = &mut microchurn {
            churn.apply_grid(&mut grid);
        }
        if let Some(factory) = &factory {
            factory.apply_sources_grid(&mut grid, side);
        }
        let next = world.step_grid(&grid);
        grid = next;
        if let Some(factory) = &factory {
            factory.drain_sinks_grid(&mut grid, side);
        }
        world.generation += 1;
    }
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    let mut factory_total = FactoryStepStats::default();
    for gen in 0..generations {
        let drops = microchurn_sand_per_step.unwrap_or(0);
        if let Some(churn) = &mut microchurn {
            churn.apply_grid(&mut grid);
        }
        let mut factory_step = FactoryStepStats::default();
        if let Some(factory) = &factory {
            factory_step.backpressure = factory.apply_sources_grid(&mut grid, side);
        }
        let start = Instant::now();
        let next = world.step_grid(&grid);
        let step_us = start.elapsed().as_micros();
        grid = next;
        if let Some(factory) = &factory {
            factory_step.sinked = factory.drain_sinks_grid(&mut grid, side);
        }
        world.generation += 1;
        times.push(step_us);
        records.push(GenerationRecord {
            gen,
            step_us,
            pop_count: popcount(&grid),
            drops,
            work_elision_factor_x: None,
            leaf_misses: None,
            factory_sinked: factory.as_ref().map(|_| factory_step.sinked),
            factory_backpressure: factory.as_ref().map(|_| factory_step.backpressure),
            state_hash: Some(grid_hash(&grid)),
            mat_distribution: Some(material_distribution(&grid)),
            grid: Some(grid.clone()),
        });
        factory_total.sinked += factory_step.sinked;
        factory_total.backpressure += factory_step.backpressure;
    }
    let mut metrics = metrics(times);
    apply_factory_metrics(&mut metrics, factory, factory_total);
    (records, metrics)
}

fn apply_factory_metrics(
    metrics: &mut MetricsRecord,
    factory: Option<FactoryHarness>,
    totals: FactoryStepStats,
) {
    if factory.is_some() {
        metrics.factory_sinked_total = Some(totals.sinked);
        metrics.factory_backpressure_total = Some(totals.backpressure);
    }
}

fn soup_search_summary_for(
    scenario: &Scenario,
    generations: &[GenerationRecord],
    side: usize,
) -> Option<SoupSearchSummary> {
    scenario.setup.and_then(|setup| match setup {
        ScenarioSetup::SoupSearchV1 | ScenarioSetup::SoupSearchSparseV1 => Some(
            soup_search_summary(generations, side, soup_search_params(setup)),
        ),
        _ => None,
    })
}

fn soup_search_summary(
    generations: &[GenerationRecord],
    side: usize,
    params: SoupSearchParams,
) -> SoupSearchSummary {
    let tiles_per_axis = side as i64 / params.tile;
    let mut tiles = Vec::with_capacity((tiles_per_axis * tiles_per_axis * tiles_per_axis) as usize);
    for tile_z in 0..tiles_per_axis {
        for tile_y in 0..tiles_per_axis {
            for tile_x in 0..tiles_per_axis {
                let tile = [tile_x, tile_y, tile_z];
                let pop_history = generations
                    .iter()
                    .map(|gen| soup_tile_pop(gen, side, tile, params))
                    .collect::<Vec<_>>();
                let final_pop = *pop_history.last().unwrap_or(&0);
                let max_pop = pop_history.iter().copied().max().unwrap_or(0);
                let lifespan_generations = pop_history.iter().filter(|&&pop| pop > 0).count();
                let survived_window = final_pop > 0;
                let candidate_stable = survived_window
                    && pop_history.len() >= 3
                    && pop_history[pop_history.len() - 3..]
                        .iter()
                        .all(|&pop| pop == final_pop);
                tiles.push(SoupTileSummary {
                    tile,
                    final_pop,
                    max_pop,
                    lifespan_generations,
                    survived_window,
                    candidate_stable,
                    final_state_hash: generations
                        .last()
                        .map(|gen| soup_tile_hash(gen, side, tile, params))
                        .unwrap_or_else(|| "sha256:e3b0c44298fc1c14".to_string()),
                    pop_history,
                });
            }
        }
    }
    let survivor_count = tiles.iter().filter(|tile| tile.survived_window).count();
    let candidate_stable_count = tiles.iter().filter(|tile| tile.candidate_stable).count();
    let extinct_count = tiles.iter().filter(|tile| tile.final_pop == 0).count();
    SoupSearchSummary {
        setup: params.setup.to_string(),
        tile_count: tiles.len(),
        survivor_count,
        candidate_stable_count,
        extinct_count,
        tiles,
    }
}

fn soup_tile_pop(
    gen: &GenerationRecord,
    side: usize,
    tile: [i64; 3],
    params: SoupSearchParams,
) -> usize {
    soup_tile_cells(gen, side, tile, params)
        .filter(|&cell| cell != 0)
        .count()
}

fn soup_tile_hash(
    gen: &GenerationRecord,
    side: usize,
    tile: [i64; 3],
    params: SoupSearchParams,
) -> String {
    let mut bytes = Vec::with_capacity((params.tile as usize).pow(3) * 2);
    for cell in soup_tile_cells(gen, side, tile, params) {
        bytes.extend_from_slice(&cell.to_le_bytes());
    }
    format!("sha256:{}", hex16(&bytes))
}

fn soup_tile_cells<'a>(
    gen: &'a GenerationRecord,
    side: usize,
    tile: [i64; 3],
    params: SoupSearchParams,
) -> impl Iterator<Item = CellState> + 'a {
    let grid = gen.grid.as_deref().unwrap_or(&[]);
    let origin = [
        tile[0] * params.tile,
        tile[1] * params.tile,
        tile[2] * params.tile,
    ];
    (0..params.tile).flat_map(move |dz| {
        (0..params.tile).flat_map(move |dy| {
            (0..params.tile).map(move |dx| {
                let x = (origin[0] + dx) as usize;
                let y = (origin[1] + dy) as usize;
                let z = (origin[2] + dz) as usize;
                grid.get(x + y * side + z * side * side)
                    .copied()
                    .unwrap_or(0)
            })
        })
    })
}

fn apply_work_elision_metrics(metrics: &mut MetricsRecord, stats: &[WorkElisionStats]) {
    if stats.is_empty() {
        return;
    }
    let mut factors = stats.iter().map(|s| s.factor_x).collect::<Vec<_>>();
    factors.sort_by(|a, b| a.total_cmp(b));
    let total_factor: f64 = factors.iter().sum();
    let total_leaf_misses: u64 = stats.iter().map(|s| s.leaf_misses).sum();
    let last = factors.len() - 1;
    let p05_idx = ((factors.len() as f64 * 0.05).ceil() as usize)
        .saturating_sub(1)
        .min(last);

    metrics.work_elision_min_x = Some(factors[0]);
    metrics.work_elision_mean_x = Some(total_factor / factors.len() as f64);
    metrics.work_elision_p05_x = Some(factors[p05_idx]);
    metrics.leaf_misses_mean = Some(total_leaf_misses as f64 / stats.len() as f64);
    metrics.work_elision_leaf_level = Some(stats[0].leaf_level);
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

fn grid_hash(grid: &[CellState]) -> String {
    let mut bytes = Vec::with_capacity(grid.len() * std::mem::size_of::<CellState>());
    for cell in grid {
        bytes.extend_from_slice(&cell.to_le_bytes());
    }
    format!("sha256:{}", hex16(&bytes))
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
            work_elision_min_x: None,
            work_elision_mean_x: None,
            work_elision_p05_x: None,
            leaf_misses_mean: None,
            work_elision_leaf_level: None,
            factory_sinked_total: None,
            factory_backpressure_total: None,
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
        work_elision_min_x: None,
        work_elision_mean_x: None,
        work_elision_p05_x: None,
        leaf_misses_mean: None,
        work_elision_leaf_level: None,
        factory_sinked_total: None,
        factory_backpressure_total: None,
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
    use std::sync::Mutex;

    static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static CWD_TEST_LOCK: Mutex<()> = Mutex::new(());

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
    fn factory_setup_changes_hash_and_requires_matching_scene_and_rule_set() {
        let mut factory = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        factory.scene = Scene::FactoryConveyor;
        factory.rule_set = RuleSet::FactoryConveyorV1;
        factory.setup = Some(ScenarioSetup::FactoryConveyorRuleV1);

        let mut raw = factory.clone();
        raw.rule_set = RuleSet::DefaultCa;
        raw.setup = None;

        assert_ne!(scenario_hash(&factory, 6), scenario_hash(&raw, 6));
        assert!(validate_backend_regime(&factory).is_ok());

        let mut bad_scene = factory.clone();
        bad_scene.scene = Scene::DefaultTerrain;
        assert!(validate_backend_regime(&bad_scene).is_err());

        let mut bad_rule = factory;
        bad_rule.rule_set = RuleSet::DefaultCa;
        assert!(validate_backend_regime(&bad_rule).is_err());
    }

    #[test]
    fn soup_search_setup_changes_hash_and_requires_matching_rule_set() {
        let mut soup = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        soup.scene = Scene::SoupSearch;
        soup.rule_set = RuleSet::SoupSearchV1;
        soup.intensity = Intensity::PassiveActive;
        soup.setup = Some(ScenarioSetup::SoupSearchV1);

        let mut raw = soup.clone();
        raw.rule_set = RuleSet::DefaultCa;
        raw.setup = None;

        assert_ne!(scenario_hash(&soup, 6), scenario_hash(&raw, 6));
        assert!(validate_backend_regime(&soup).is_ok());

        let mut bad_scene = soup.clone();
        bad_scene.scene = Scene::DefaultTerrain;
        assert!(validate_backend_regime(&bad_scene).is_err());

        let mut bad_rule = soup.clone();
        bad_rule.rule_set = RuleSet::DefaultCa;
        assert!(validate_backend_regime(&bad_rule).is_err());

        let mut missing_setup = soup;
        missing_setup.setup = None;
        missing_setup.rule_set = RuleSet::DefaultCa;
        assert!(validate_backend_regime(&missing_setup).is_err());

        let mut sparse = test_scenario(Backend::HashlifeRecursive, Regime::Churning);
        sparse.scene = Scene::SoupSearch;
        sparse.rule_set = RuleSet::SoupSearchV1;
        sparse.intensity = Intensity::PassiveActive;
        sparse.setup = Some(ScenarioSetup::SoupSearchSparseV1);
        assert!(validate_backend_regime(&sparse).is_ok());
        assert_ne!(scenario_hash(&sparse, 6), scenario_hash(&bad_rule, 6));
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
        let err = compare_records(
            &path,
            "same-id",
            "missing",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();
        assert!(err.contains("duplicate measurement_id"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn append_hygiene_allows_only_append_target_dirty() {
        let status = " M .ship-notes/perf-runs.jsonl\n";

        let dirty = unexpected_dirty_paths(status, ".ship-notes/perf-runs.jsonl");

        assert!(dirty.is_empty());
    }

    #[test]
    fn append_hygiene_reports_untracked_non_append_input() {
        let status = "?? scenarios/local-only.ron\n M .ship-notes/perf-runs.jsonl\n";

        let dirty = unexpected_dirty_paths(status, ".ship-notes/perf-runs.jsonl");

        assert_eq!(dirty, vec!["scenarios/local-only.ron"]);
    }

    #[test]
    fn append_hygiene_rejects_untracked_closure_input() {
        let _guard = CWD_TEST_LOCK.lock().expect("cwd test lock");
        let repo = temp_git_repo();
        let input = repo.join("scenario.ron");
        std::fs::write(&input, "untracked").expect("write untracked input");

        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&repo).expect("cd temp repo");
        let err = ensure_git_tracked_path("scenario.ron").unwrap_err();
        std::env::set_current_dir(cwd).expect("restore cwd");

        assert!(err.contains("untracked input"), "{err}");
    }

    #[test]
    fn append_hygiene_accepts_tracked_closure_input() {
        let _guard = CWD_TEST_LOCK.lock().expect("cwd test lock");
        let repo = temp_git_repo();
        let input = repo.join("scenario.ron");
        std::fs::write(&input, "tracked").expect("write tracked input");
        Command::new("git")
            .args(["add", "scenario.ron"])
            .current_dir(&repo)
            .status()
            .expect("git add");

        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&repo).expect("cd temp repo");
        ensure_git_tracked_path("scenario.ron").expect("tracked input");
        std::env::set_current_dir(cwd).expect("restore cwd");
    }

    #[test]
    fn append_hygiene_accepts_repo_relative_tracked_input_from_subdir() {
        let _guard = CWD_TEST_LOCK.lock().expect("cwd test lock");
        let repo = temp_git_repo();
        let scenarios = repo.join("scenarios");
        let subdir = repo.join("subdir");
        std::fs::create_dir_all(&scenarios).expect("create scenarios");
        std::fs::create_dir_all(&subdir).expect("create subdir");
        std::fs::write(scenarios.join("tracked.ron"), "tracked").expect("write tracked input");
        let append = repo.join(".ship-notes").join("perf.jsonl");
        std::fs::create_dir_all(append.parent().unwrap()).expect("create append parent");
        Command::new("git")
            .args(["add", "scenarios/tracked.ron"])
            .current_dir(&repo)
            .status()
            .expect("git add");
        Command::new("git")
            .args(["commit", "-q", "-m", "track scenario"])
            .current_dir(&repo)
            .status()
            .expect("git commit");

        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&subdir).expect("cd subdir");
        ensure_clean_git_tree_for_append(&append, vec![Path::new("scenarios/tracked.ron")])
            .expect("repo-relative tracked input from subdir");
        std::env::set_current_dir(cwd).expect("restore cwd");
    }

    #[test]
    fn compare_records_rejects_trajectory_drift_by_default() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        let path = write_jsonl(&[&a, &b]);

        let err = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();

        assert!(err.contains("trajectory drift"), "{err}");
        assert!(err.contains("--allow-trajectory-drift"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_allows_explicit_trajectory_drift_with_structured_fields() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        let path = write_jsonl(&[&a, &b]);

        let comparison = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions {
                allow_trajectory_drift: true,
            },
        )
        .expect("comparison");

        assert_eq!(comparison.ratio, 5.0);
        assert!(!comparison.trajectory_equivalent);
        let drift = comparison.trajectory_drift.expect("trajectory drift");
        assert_eq!(drift.generation_index, Some(0));
        assert_eq!(drift.subject_final_pop, 11);
        assert_eq!(drift.baseline_final_pop, 10);
        assert!(drift.generation_count_equal);
        assert!(drift.final_material_distribution_equal);
        assert!(comparison.notes.contains("trajectory caveat"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_emits_ratio_for_equivalent_trajectory() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        b.generations[0].pop_count = a.generations[0].pop_count;
        let path = write_jsonl(&[&a, &b]);

        let comparison = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .expect("comparison");

        assert_eq!(comparison.ratio, 5.0);
        assert!(comparison.trajectory_equivalent);
        assert!(comparison.trajectory_drift.is_none());
        assert!(!comparison.notes.contains("trajectory caveat"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_command_rejects_drift_without_allow_flag() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        let path = write_jsonl(&[&a, &b]);

        let err = run_with_args(compare_args(&path, false)).unwrap_err();

        assert!(err.contains("trajectory drift"), "{err}");
        assert!(err.contains("--allow-trajectory-drift"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_command_allow_flag_emits_structured_drift_json() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        let path = write_jsonl(&[&a, &b]);

        let line = run_with_args(compare_args(&path, true)).expect("comparison");
        let value: serde_json::Value = serde_json::from_str(&line).expect("comparison json");

        assert_eq!(value["record_kind"], "comparison");
        assert_eq!(value["trajectory_equivalent"], false);
        assert_eq!(value["trajectory_drift"]["subject_final_pop"], 11);
        assert_eq!(value["trajectory_drift"]["baseline_final_pop"], 10);
        assert_eq!(
            value["trajectory_drift"]["final_material_distribution_equal"],
            true
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_mismatched_hardware() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        b.hardware = "m2-ultra-mac-pro".to_string();
        let path = write_jsonl(&[&a, &b]);
        let err = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();
        assert!(err.contains("hardware"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_invalid_backend_regime_pair() {
        let mut a = test_measurement("chunk", "chunk-array", "saturated", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.generations[0].pop_count = b.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(err.contains("backend=chunk-array requires regime"), "{err}");
    }

    #[test]
    fn compare_records_rejects_invalid_setup_rule_set_pair() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.scene = "factory-conveyor".to_string();
        a.setup = Some("FactoryConveyorRuleV1".to_string());
        a.rule_set = "default-ca".to_string();
        a.generations[0].pop_count = b.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(err.contains("requires matching rule_set"), "{err}");
    }

    #[test]
    fn compare_records_rejects_soup_search_without_setup() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "churning", 2.0);
        a.scene = "soup-search".to_string();
        a.intensity = "passive-active".to_string();
        a.generations[0].pop_count = b.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(
            err.contains("scene=soup-search requires a soup setup"),
            "{err}"
        );
    }

    #[test]
    fn compare_records_detects_state_hash_drift() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        b.generations[0].pop_count = a.generations[0].pop_count;
        b.generations[0].mat_distribution = a.generations[0].mat_distribution.clone();
        b.generations[0].state_hash = Some("sha256:ffffffffffffffff".to_string());
        let path = write_jsonl(&[&a, &b]);

        let err = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();

        assert!(err.contains("trajectory drift"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_malformed_scenario_hash() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.scenario_hash = "sha256:test".to_string();
        a.generations[0].pop_count = b.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(err.contains("malformed scenario_hash"), "{err}");
    }

    #[test]
    fn compare_records_rejects_partial_hashlife_metrics() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        b.metrics.work_elision_mean_x = None;
        b.generations[0].pop_count = a.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(
            err.contains("partial/missing backend-specific metrics"),
            "{err}"
        );
    }

    #[test]
    fn compare_records_rejects_hashlife_without_backend_metrics() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        strip_hashlife_metrics(&mut b);
        b.generations[0].pop_count = a.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(
            err.contains("partial/missing backend-specific metrics"),
            "{err}"
        );
    }

    #[test]
    fn compare_records_accepts_legacy_hashlife_memo_only_metrics() {
        let a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        strip_work_elision_metrics(&mut b);
        b.generations[0].pop_count = a.generations[0].pop_count;
        let path = write_jsonl(&[&a, &b]);

        let comparison = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .expect("comparison");

        assert_eq!(comparison.ratio, 5.0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_soup_search_hashlife_without_work_elision_metrics() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "churning", 2.0);
        a.world = "small".to_string();
        a.scene = "soup-search".to_string();
        a.intensity = "passive-active".to_string();
        a.rule_set = "custom:soup-search-v1".to_string();
        a.setup = Some(SOUP_SEARCH_SETUP_V1.to_string());
        a.level = 6;
        a.side = 64;
        b.world = a.world.clone();
        b.scene = a.scene.clone();
        b.intensity = a.intensity.clone();
        b.rule_set = a.rule_set.clone();
        b.setup = a.setup.clone();
        b.level = a.level;
        b.side = a.side;
        b.generations[0].pop_count = a.generations[0].pop_count;
        let summary = SoupSearchSummary {
            setup: SOUP_SEARCH_SETUP_V1.to_string(),
            tile_count: 1,
            survivor_count: 1,
            candidate_stable_count: 0,
            extinct_count: 0,
            tiles: vec![SoupTileSummary {
                tile: [0, 0, 0],
                final_pop: 1,
                max_pop: 1,
                lifespan_generations: 1,
                survived_window: true,
                candidate_stable: false,
                final_state_hash: "sha256:aaaaaaaaaaaaaaaa".to_string(),
                pop_history: vec![1],
            }],
        };
        a.soup_search = Some(summary.clone());
        b.soup_search = Some(summary);
        strip_work_elision_metrics(&mut b);

        let err = compare_error(&a, &b);

        assert!(err.contains("requires work-elision metrics"), "{err}");
    }

    #[test]
    fn compare_records_rejects_chunk_array_hashlife_metrics() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.metrics.memo_hit_ratio = Some(0.5);
        a.generations[0].pop_count = b.generations[0].pop_count;

        let err = compare_error(&a, &b);

        assert!(err.contains("must not include hashlife metrics"), "{err}");
    }

    #[test]
    fn compare_records_accepts_all_frames_warm_policy() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.confidence.warm_frame_policy = "all-frames".to_string();
        b.confidence.warm_frame_policy = "all-frames".to_string();
        b.generations[0].pop_count = a.generations[0].pop_count;
        let path = write_jsonl(&[&a, &b]);

        let comparison = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .expect("comparison");

        assert_eq!(comparison.ratio, 5.0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_factory_total_drift() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.metrics.factory_sinked_total = Some(4);
        a.metrics.factory_backpressure_total = Some(2);
        a.generations[0].factory_sinked = Some(4);
        a.generations[0].factory_backpressure = Some(2);
        b.metrics.factory_sinked_total = Some(3);
        b.metrics.factory_backpressure_total = Some(2);
        b.generations[0].factory_sinked = Some(3);
        b.generations[0].factory_backpressure = Some(2);

        let path = write_jsonl(&[&a, &b]);
        let err = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();

        assert!(err.contains("metrics.factory_sinked_total"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compare_records_rejects_factory_generation_drift() {
        let mut a = test_measurement("chunk", "chunk-array", "n/a", 10.0);
        let mut b = test_measurement("hashlife", "hashlife-recursive", "saturated", 2.0);
        a.metrics.factory_sinked_total = Some(4);
        a.metrics.factory_backpressure_total = Some(2);
        a.generations[0].factory_sinked = Some(4);
        a.generations[0].factory_backpressure = Some(2);
        b.metrics.factory_sinked_total = Some(4);
        b.metrics.factory_backpressure_total = Some(2);
        b.generations[0].factory_sinked = Some(3);
        b.generations[0].factory_backpressure = Some(2);

        let path = write_jsonl(&[&a, &b]);
        let err = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();

        assert!(err.contains("generations[0].factory_sinked"), "{err}");
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

        let (hashlife, _) = run_hashlife(hashlife_world, 1, 4, Some(8), None, 7);
        let (chunk, _) = run_chunk_array(chunk_world, 1, 4, Some(8), None, 7);

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
    fn hashlife_records_work_elision_metrics() {
        let level = 5;
        let params = TerrainParams::for_level(level);
        let mut world = World::new(level);
        world.seed_terrain(&params).unwrap();

        let (generations, metrics) = run_hashlife(world, 1, 3, None, None, 1);

        assert_eq!(generations.len(), 3);
        assert!(generations
            .iter()
            .all(|gen| gen.work_elision_factor_x.is_some() && gen.leaf_misses.is_some()));
        assert!(metrics.work_elision_min_x.unwrap() > 0.0);
        assert!(metrics.work_elision_mean_x.unwrap() > 0.0);
        assert!(metrics.work_elision_p05_x.unwrap() > 0.0);
        assert!(metrics.leaf_misses_mean.unwrap() >= 0.0);
        assert!(matches!(metrics.work_elision_leaf_level, Some(3 | 4)));
    }

    #[test]
    fn chunk_array_omits_hashlife_work_elision_metrics() {
        let level = 5;
        let params = TerrainParams::for_level(level);
        let mut world = World::new(level);
        world.seed_terrain(&params).unwrap();

        let (generations, metrics) = run_chunk_array(world, 1, 3, None, None, 1);

        assert!(generations
            .iter()
            .all(|gen| gen.work_elision_factor_x.is_none() && gen.leaf_misses.is_none()));
        assert!(metrics.work_elision_min_x.is_none());
        assert!(metrics.work_elision_mean_x.is_none());
        assert!(metrics.work_elision_p05_x.is_none());
        assert!(metrics.leaf_misses_mean.is_none());
        assert!(metrics.work_elision_leaf_level.is_none());
    }

    #[test]
    fn old_measurement_json_without_work_elision_fields_still_compares() {
        let body = r#"{"schema_version":2,"record_kind":"measurement","measurement_id":"chunk","world":"medium","scene":"default-demo","intensity":"cascade","regime":"n/a","rule_set":"default-ca","backend":"chunk-array","hardware":"m2-pro-mbp","scenario_hash":"sha256:0123456789abcdef","confidence":{"n":2,"warm_frame_policy":"skip-first-1","source":"bench","cherry_pick_audit":"hard_included","notes":"test"},"level":7,"side":128,"git_commit":"test","bench_fn":"scenario-runner","comparator":null,"metrics":{"step_mean_ms":10.0,"step_median_ms":10.0,"step_p95_ms":10.0,"wall_total_ms":20.0},"generations":[{"gen":0,"step_us":10000,"pop_count":11,"drops":0,"mat_distribution":{"1":10}}]}
{"schema_version":2,"record_kind":"measurement","measurement_id":"hashlife","world":"medium","scene":"default-demo","intensity":"cascade","regime":"saturated","rule_set":"default-ca","backend":"hashlife-recursive","hardware":"m2-pro-mbp","scenario_hash":"sha256:0123456789abcdef","confidence":{"n":2,"warm_frame_policy":"skip-first-1","source":"bench","cherry_pick_audit":"hard_included","notes":"test"},"level":7,"side":128,"git_commit":"test","bench_fn":"scenario-runner","comparator":null,"metrics":{"step_mean_ms":2.0,"step_median_ms":2.0,"step_p95_ms":2.0,"wall_total_ms":4.0,"memo_hit_ratio":0.5,"elision_factor_x":1.5},"generations":[{"gen":0,"step_us":2000,"pop_count":11,"drops":0,"mat_distribution":{"1":10}}]}
	"#;
        let path = std::env::temp_dir().join(format!(
            "hash-thing-scenario-runner-old-json-{}-{}.jsonl",
            std::process::id(),
            TEST_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, body).expect("write old jsonl");

        let comparison = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .expect("comparison");

        assert_eq!(comparison.ratio, 5.0);
        let _ = std::fs::remove_file(path);
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

        let (hashlife, _) = run_hashlife(hashlife_world, 1, 4, None, None, 1);
        let (chunk, _) = run_chunk_array(chunk_world, 1, 4, None, None, 1);

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
    fn factory_conveyor_setup_keeps_backends_on_same_trajectory() {
        let mut scenario = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        scenario.world = WorldCoordName::Small;
        scenario.level = Some(6);
        scenario.scene = Scene::FactoryConveyor;
        scenario.rule_set = RuleSet::FactoryConveyorV1;
        scenario.intensity = Intensity::PassiveActive;
        scenario.setup = Some(ScenarioSetup::FactoryConveyorRuleV1);

        let mut hashlife_world = World::new(6);
        seed_scene(&mut hashlife_world, &scenario).expect("seed hashlife");
        let mut chunk_world = World::new(6);
        seed_scene(&mut chunk_world, &scenario).expect("seed chunk");
        let factory = FactoryHarness::for_scenario(&scenario, 64);

        let (hashlife, hashlife_metrics) =
            run_hashlife(hashlife_world, 2, 6, None, factory.clone(), 1);
        let (chunk, chunk_metrics) = run_chunk_array(chunk_world, 2, 6, None, factory, 1);

        assert_eq!(hashlife.len(), chunk.len());
        for (gen, (h, c)) in hashlife.iter().zip(chunk.iter()).enumerate() {
            assert_eq!(h.pop_count, c.pop_count, "pop drift at gen {gen}");
            assert_eq!(
                h.mat_distribution, c.mat_distribution,
                "material drift at gen {gen}"
            );
            assert_eq!(
                h.factory_sinked, c.factory_sinked,
                "sink drift at gen {gen}"
            );
            assert_eq!(
                h.factory_backpressure, c.factory_backpressure,
                "backpressure drift at gen {gen}"
            );
        }
        assert!(hashlife_metrics.factory_sinked_total.unwrap() > 0);
        assert_eq!(
            hashlife_metrics.factory_sinked_total,
            chunk_metrics.factory_sinked_total
        );
    }

    #[test]
    fn factory_conveyor_harness_makes_multi_step_progress() {
        let mut scenario = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        scenario.world = WorldCoordName::Small;
        scenario.level = Some(6);
        scenario.scene = Scene::FactoryConveyor;
        scenario.rule_set = RuleSet::FactoryConveyorV1;
        scenario.intensity = Intensity::PassiveActive;
        scenario.setup = Some(ScenarioSetup::FactoryConveyorRuleV1);

        let mut world = World::new(6);
        seed_scene(&mut world, &scenario).expect("seed factory conveyor");
        let factory = FactoryHarness::for_scenario(&scenario, 64);
        let (generations, metrics) = run_hashlife(world, 0, 16, None, factory, 1);

        assert!(generations
            .iter()
            .any(|gen| gen.factory_sinked.unwrap_or(0) > 0));
        assert!(metrics.factory_sinked_total.unwrap() > 0);
    }

    #[test]
    fn soup_search_seed_is_deterministic_and_nonempty() {
        let mut scenario = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        scenario.world = WorldCoordName::Small;
        scenario.level = Some(6);
        scenario.scene = Scene::SoupSearch;
        scenario.rule_set = RuleSet::SoupSearchV1;
        scenario.intensity = Intensity::PassiveActive;
        scenario.setup = Some(ScenarioSetup::SoupSearchV1);

        let mut a = World::new(6);
        seed_scene(&mut a, &scenario).expect("seed soup a");
        let mut b = World::new(6);
        seed_scene(&mut b, &scenario).expect("seed soup b");

        assert_eq!(a.flatten(), b.flatten());
        assert!(a.population() > 0);

        scenario.seed += 1;
        let mut c = World::new(6);
        seed_scene(&mut c, &scenario).expect("seed soup c");
        assert_ne!(a.flatten(), c.flatten());
    }

    #[test]
    fn soup_search_rng_does_not_stick_on_zero_state() {
        let mut rng = SoupSearchRng::new(0xD1B5_4A32_D192_ED03);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn soup_search_summary_classifies_extinct_survivor_and_candidate_tiles() {
        let side = 32;
        let mut grid = vec![0; side * side * side];
        grid[4 + 4 * side + 4 * side * side] = SOUP_SEARCH_ALIVE;
        let gen = |i| GenerationRecord {
            gen: i,
            step_us: 0,
            pop_count: popcount(&grid),
            drops: 0,
            work_elision_factor_x: None,
            leaf_misses: None,
            factory_sinked: None,
            factory_backpressure: None,
            state_hash: Some(grid_hash(&grid)),
            mat_distribution: Some(material_distribution(&grid)),
            grid: Some(grid.clone()),
        };

        let summary = soup_search_summary(
            &[gen(0), gen(1), gen(2)],
            side,
            soup_search_params(ScenarioSetup::SoupSearchV1),
        );

        assert_eq!(summary.tile_count, 8);
        assert_eq!(summary.survivor_count, 1);
        assert_eq!(summary.candidate_stable_count, 1);
        assert_eq!(summary.extinct_count, 7);
    }

    #[test]
    fn soup_search_keeps_backends_on_same_trajectory() {
        let mut scenario = test_scenario(Backend::HashlifeRecursive, Regime::Saturated);
        scenario.world = WorldCoordName::Small;
        scenario.level = Some(6);
        scenario.scene = Scene::SoupSearch;
        scenario.rule_set = RuleSet::SoupSearchV1;
        scenario.intensity = Intensity::PassiveActive;
        scenario.setup = Some(ScenarioSetup::SoupSearchV1);

        let mut hashlife_world = World::new(6);
        seed_scene(&mut hashlife_world, &scenario).expect("seed hashlife soup");
        let mut chunk_world = World::new(6);
        seed_scene(&mut chunk_world, &scenario).expect("seed chunk soup");

        let (hashlife, _) = run_hashlife(hashlife_world, 1, 4, None, None, 11);
        let (chunk, _) = run_chunk_array(chunk_world, 1, 4, None, None, 11);

        assert_eq!(hashlife.len(), chunk.len());
        for (gen, (h, c)) in hashlife.iter().zip(chunk.iter()).enumerate() {
            assert_eq!(h.pop_count, c.pop_count, "pop drift at gen {gen}");
            assert_eq!(
                h.mat_distribution, c.mat_distribution,
                "material drift at gen {gen}"
            );
            assert_eq!(h.state_hash, c.state_hash, "state hash drift at gen {gen}");
        }
        let hashlife_summary = soup_search_summary(
            &hashlife,
            64,
            soup_search_params(ScenarioSetup::SoupSearchV1),
        );
        let chunk_summary =
            soup_search_summary(&chunk, 64, soup_search_params(ScenarioSetup::SoupSearchV1));
        assert_eq!(hashlife_summary, chunk_summary);
        assert_eq!(hashlife_summary.tile_count, 64);
        assert!(hashlife_summary.survivor_count > 0);
        assert!(hashlife_summary
            .tiles
            .iter()
            .any(|tile| tile.lifespan_generations > 0));
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
            scenario_hash: "sha256:0123456789abcdef".to_string(),
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
                memo_hit_ratio: (backend == "hashlife-recursive").then_some(0.5),
                elision_factor_x: (backend == "hashlife-recursive").then_some(1.5),
                work_elision_min_x: (backend == "hashlife-recursive").then_some(1.0),
                work_elision_mean_x: (backend == "hashlife-recursive").then_some(1.5),
                work_elision_p05_x: (backend == "hashlife-recursive").then_some(1.0),
                leaf_misses_mean: (backend == "hashlife-recursive").then_some(2.0),
                work_elision_leaf_level: (backend == "hashlife-recursive").then_some(3),
                factory_sinked_total: None,
                factory_backpressure_total: None,
            },
            generations: vec![GenerationRecord {
                gen: 0,
                step_us: (step_p95_ms * 1000.0) as u128,
                pop_count: if backend == "chunk-array" { 11 } else { 10 },
                drops: 0,
                work_elision_factor_x: (backend == "hashlife-recursive").then_some(1.5),
                leaf_misses: (backend == "hashlife-recursive").then_some(2),
                factory_sinked: None,
                factory_backpressure: None,
                state_hash: Some("sha256:aaaaaaaaaaaaaaaa".to_string()),
                mat_distribution: Some(serde_json::json!({"1": 10})),
                grid: None,
            }],
            soup_search: None,
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

    fn compare_error(a: &MeasurementRecord, b: &MeasurementRecord) -> String {
        let path = write_jsonl(&[a, b]);
        let err = compare_records(
            &path,
            "chunk",
            "hashlife",
            "step_p95_ms",
            CompareOptions::default(),
        )
        .unwrap_err();
        let _ = std::fs::remove_file(path);
        err
    }

    fn strip_hashlife_metrics(record: &mut MeasurementRecord) {
        record.metrics.memo_hit_ratio = None;
        record.metrics.elision_factor_x = None;
        strip_work_elision_metrics(record);
    }

    fn strip_work_elision_metrics(record: &mut MeasurementRecord) {
        record.metrics.work_elision_min_x = None;
        record.metrics.work_elision_mean_x = None;
        record.metrics.work_elision_p05_x = None;
        record.metrics.leaf_misses_mean = None;
        record.metrics.work_elision_leaf_level = None;
        for generation in &mut record.generations {
            generation.work_elision_factor_x = None;
            generation.leaf_misses = None;
        }
    }

    fn compare_args(path: &Path, allow_trajectory_drift: bool) -> Vec<std::ffi::OsString> {
        let mut args = vec![
            std::ffi::OsString::from("--compare"),
            path.as_os_str().to_owned(),
            std::ffi::OsString::from("chunk"),
            std::ffi::OsString::from("hashlife"),
            std::ffi::OsString::from("--metric"),
            std::ffi::OsString::from("step_p95_ms"),
        ];
        if allow_trajectory_drift {
            args.push(std::ffi::OsString::from("--allow-trajectory-drift"));
        }
        args
    }

    fn temp_git_repo() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hash-thing-scenario-runner-git-test-{}-{}",
            std::process::id(),
            TEST_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("create temp git repo");
        let status = Command::new("git")
            .args(["init", "-q"])
            .current_dir(&path)
            .status()
            .expect("git init");
        assert!(status.success(), "git init failed");
        for (key, value) in [
            ("user.email", "scenario-runner-test@example.invalid"),
            ("user.name", "Scenario Runner Test"),
        ] {
            let status = Command::new("git")
                .args(["config", key, value])
                .current_dir(&path)
                .status()
                .expect("git config");
            assert!(status.success(), "git config failed");
        }
        path
    }
}
