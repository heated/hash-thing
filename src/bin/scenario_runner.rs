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

#[derive(Serialize)]
struct ConfidenceRecord {
    n: usize,
    warm_frame_policy: &'static str,
    source: &'static str,
    cherry_pick_audit: &'static str,
    notes: String,
}

#[derive(Serialize)]
struct GenerationRecord {
    gen: usize,
    step_us: u128,
    pop_count: usize,
    drops: usize,
    mat_distribution: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct MetricsRecord {
    step_mean_ms: f64,
    step_median_ms: f64,
    step_p95_ms: f64,
    wall_total_ms: f64,
}

#[derive(Serialize)]
struct MeasurementRecord {
    schema_version: u32,
    record_kind: &'static str,
    measurement_id: String,
    world: &'static str,
    scene: &'static str,
    intensity: &'static str,
    regime: &'static str,
    rule_set: &'static str,
    backend: &'static str,
    hardware: &'static str,
    scenario_hash: String,
    confidence: ConfidenceRecord,
    level: u32,
    side: u64,
    git_commit: String,
    bench_fn: &'static str,
    comparator: Option<String>,
    metrics: MetricsRecord,
    generations: Vec<GenerationRecord>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("scenario-runner: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let path = args.scenario;
    let bytes = fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|err| format!("{} is not UTF-8: {err}", path.display()))?;
    let scenario: Scenario =
        ron::from_str(text).map_err(|err| format!("parse {}: {err}", path.display()))?;
    let record = run_scenario(&path, &bytes, &scenario)?;
    let line = serde_json::to_string(&record).map_err(|err| format!("serialize record: {err}"))?;
    println!("{line}");
    if let Some(append_path) = args.append {
        append_jsonl(&append_path, &line)?;
    }
    Ok(())
}

struct Args {
    scenario: PathBuf,
    append: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args_os().skip(1);
    let scenario = raw.next().map(PathBuf::from).ok_or_else(usage)?;
    let mut append = None;
    while let Some(arg) = raw.next() {
        if arg == "--append" {
            let path = raw
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| "--append requires a path".to_string())?;
            append = Some(path);
        } else {
            return Err(format!("unknown argument: {arg:?}\n{}", usage()));
        }
    }
    Ok(Args { scenario, append })
}

fn usage() -> String {
    "usage: cargo run --bin scenario-runner -- <scenario.ron> [--append <out.jsonl>]".to_string()
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

fn run_scenario(
    path: &Path,
    bytes: &[u8],
    scenario: &Scenario,
) -> Result<MeasurementRecord, String> {
    validate_backend_regime(scenario)?;
    let level = scenario.level.unwrap_or_else(|| scenario.world.level());
    let side = 1u64 << level;
    let scenario_hash = scenario_hash(bytes, scenario.seed);
    let mut world = World::new(level);
    seed_scene(&mut world, scenario)?;

    let (generations, metrics) = match scenario.backend {
        Backend::HashlifeRecursive => run_hashlife(world, scenario.generations),
        Backend::ChunkArray => run_chunk_array(world, scenario.generations),
    };

    let hash_suffix = scenario_hash
        .strip_prefix("sha256:")
        .unwrap_or(&scenario_hash)
        .chars()
        .take(8)
        .collect::<String>();

    Ok(MeasurementRecord {
        schema_version: 2,
        record_kind: "measurement",
        measurement_id: format!(
            "{}-{}-{}",
            scenario.name,
            scenario.backend.as_str(),
            hash_suffix
        ),
        world: scenario.world.as_str(),
        scene: scenario.scene.as_str(),
        intensity: scenario.intensity.as_str(),
        regime: scenario.regime.as_str(),
        rule_set: scenario.rule_set.as_str(),
        backend: scenario.backend.as_str(),
        hardware: "local-dev",
        scenario_hash,
        confidence: ConfidenceRecord {
            n: scenario.generations,
            warm_frame_policy: "all-frames",
            source: "scenario-runner",
            cherry_pick_audit: cherry_pick_audit(scenario),
            notes: format!("scenario={}, path={}", scenario.name, path.display()),
        },
        level,
        side,
        git_commit: git_commit(),
        bench_fn: "scenario-runner",
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

fn run_hashlife(mut world: World, generations: usize) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut times = Vec::with_capacity(generations);
    let mut records = Vec::with_capacity(generations);
    for gen in 0..generations {
        let start = Instant::now();
        world.step_recursive();
        let step_us = start.elapsed().as_micros();
        times.push(step_us);
        records.push(GenerationRecord {
            gen,
            step_us,
            pop_count: popcount(&world.flatten()),
            drops: 0,
            mat_distribution: None,
        });
    }
    (records, metrics(times))
}

fn run_chunk_array(mut world: World, generations: usize) -> (Vec<GenerationRecord>, MetricsRecord) {
    let mut grid = world.flatten();
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
            mat_distribution: None,
        });
    }
    (records, metrics(times))
}

fn popcount(grid: &[CellState]) -> usize {
    grid.iter().filter(|&&cell| cell != 0).count()
}

fn metrics(mut times_us: Vec<u128>) -> MetricsRecord {
    if times_us.is_empty() {
        return MetricsRecord {
            step_mean_ms: 0.0,
            step_median_ms: 0.0,
            step_p95_ms: 0.0,
            wall_total_ms: 0.0,
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
    }
}

fn scenario_hash(bytes: &[u8], seed: u64) -> String {
    let file_hash = hex16(bytes);
    let canonical = format!("v2-B|{file_hash}|{seed}");
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
        let bytes = b"(name:\"x\")";
        assert_ne!(scenario_hash(bytes, 1), scenario_hash(bytes, 2));
    }

    #[test]
    fn chunk_array_requires_na_regime() {
        let scenario = Scenario {
            name: "bad".to_string(),
            world: WorldCoordName::Tiny,
            level: None,
            scene: Scene::DefaultTerrain,
            rule_set: RuleSet::DefaultCa,
            intensity: Intensity::Idle,
            regime: Regime::Saturated,
            backend: Backend::ChunkArray,
            generations: 1,
            seed: 1,
            comparator: None,
        };
        assert!(validate_backend_regime(&scenario).is_err());
    }
}
