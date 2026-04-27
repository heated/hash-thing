//! 14yp horizontal-checkerboard reproducer.
//!
//! Edward observed water settling into a horizontal (xz-plane) checkerboard
//! pattern on the floor of the default scene — distinct from the in-flight
//! every-other-y artifact tracked by hash-thing-iomz. This harness builds
//! controlled mini-scenes (flat stone floor + water column or pre-placed
//! checkerboard) and dumps the floor-adjacent xz slice across multiple
//! steps so the mechanism can be characterized.
//!
//! Run: `cargo test --profile bench --test repro_14yp_water_floor_checkerboard
//!       -- --ignored --nocapture`
//!
//! Scope: pure CPU, deterministic, small (level=5, 32³). No GPU, no winit.

use hash_thing::octree::Cell;
use hash_thing::sim::margolus::FluidBlockRule;
use hash_thing::sim::rule::{block_index, BlockRule};
use hash_thing::sim::world::WorldCoord;
use hash_thing::sim::World;
use hash_thing::terrain::materials::{
    material_density, material_phase, STONE, STONE_MATERIAL_ID, WATER, WATER_MATERIAL_ID,
};

fn ascii_floor(world: &World, y: u64) -> Vec<String> {
    let side = world.side() as u64;
    let mut rows = Vec::with_capacity(side as usize);
    for z in 0..side {
        let mut row = String::with_capacity(side as usize);
        for x in 0..side {
            let state = world.get(
                WorldCoord(x as i64),
                WorldCoord(y as i64),
                WorldCoord(z as i64),
            );
            let m = (state >> 6) as u32;
            row.push(match m {
                0 => '.',
                1 => '#',
                5 => '~',
                _ => '?',
            });
        }
        rows.push(row);
    }
    rows
}

fn dump_floor(world: &World, y: u64, label: &str) {
    let rows = ascii_floor(world, y);
    eprintln!("--- {label}: xz slice at y={y} ({} side) ---", world.side());
    for (z, row) in rows.iter().enumerate() {
        eprintln!("  z={z:>2}: {row}");
    }
    eprintln!();
}

fn count_water_at(world: &World, y: u64) -> u64 {
    let side = world.side() as u64;
    let mut n = 0;
    for z in 0..side {
        for x in 0..side {
            let s = world.get(
                WorldCoord(x as i64),
                WorldCoord(y as i64),
                WorldCoord(z as i64),
            );
            if (s >> 6) == WATER_MATERIAL_ID {
                n += 1;
            }
        }
    }
    n
}

/// Score how "checkerboard-like" the xz water pattern at row y is.
///
/// For each pair of horizontally-adjacent cells within a square region,
/// count whether one is water and the other is air. A pure checkerboard
/// gives ~100%; a contiguous puddle gives ~0%.
fn checkerboard_score(world: &World, y: u64, lo: u64, hi: u64) -> f64 {
    let mut alt = 0u64;
    let mut total = 0u64;
    let is_water = |x: u64, z: u64| -> bool {
        let s = world.get(
            WorldCoord(x as i64),
            WorldCoord(y as i64),
            WorldCoord(z as i64),
        );
        (s >> 6) == WATER_MATERIAL_ID
    };
    let is_air = |x: u64, z: u64| -> bool {
        let s = world.get(
            WorldCoord(x as i64),
            WorldCoord(y as i64),
            WorldCoord(z as i64),
        );
        s == 0
    };
    for z in lo..hi {
        for x in lo..(hi - 1) {
            let a_w = is_water(x, z);
            let b_w = is_water(x + 1, z);
            let a_a = is_air(x, z);
            let b_a = is_air(x + 1, z);
            if (a_w && b_a) || (b_w && a_a) {
                alt += 1;
            }
            if (a_w || a_a) && (b_w || b_a) {
                total += 1;
            }
        }
    }
    for z in lo..(hi - 1) {
        for x in lo..hi {
            let a_w = is_water(x, z);
            let b_w = is_water(x, z + 1);
            let a_a = is_air(x, z);
            let b_a = is_air(x, z + 1);
            if (a_w && b_a) || (b_w && a_a) {
                alt += 1;
            }
            if (a_w || a_a) && (b_w || b_a) {
                total += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        alt as f64 / total as f64
    }
}

/// Build a small world with a flat stone floor at y=floor_y.
fn make_world_with_floor(level: u32, floor_y: u64) -> World {
    let mut world = World::new(level);
    let side = world.side() as u64;
    for z in 0..side {
        for x in 0..side {
            world.set(WorldCoord(x as i64), WorldCoord(floor_y as i64), WorldCoord(z as i64), STONE);
        }
    }
    world
}

/// Scenario A: Drop a water column above a flat floor.
/// This is the "natural" repro path — water falls, lands, and we observe
/// the floor pattern across steps.
#[test]
#[ignore]
fn repro_water_column_falls_to_flat_floor() {
    let level = 5; // 32³
    let side = 1u64 << level;

    for floor_parity_label in ["floor_y=4 (even)", "floor_y=5 (odd)"] {
        let floor_y: u64 = if floor_parity_label.contains("even") { 4 } else { 5 };
        let water_y_lo = floor_y + 4;
        let water_y_hi = floor_y + 12;

        eprintln!("=== Scenario A — water column on flat floor — {floor_parity_label} ===");
        let mut world = make_world_with_floor(level, floor_y);
        // Solid 8x8 column of water above the floor.
        let lo = side / 2 - 4;
        let hi = side / 2 + 4;
        for z in lo..hi {
            for x in lo..hi {
                for y in water_y_lo..water_y_hi {
                    world.set(WorldCoord(x as i64), WorldCoord(y as i64), WorldCoord(z as i64), WATER);
                }
            }
        }
        let initial_water = world.population() as i64
            - (side * side) as i64; // subtract floor count
        eprintln!("initial water cells: {initial_water}");
        dump_floor(&world, floor_y + 1, "gen 0 (floor+1)");

        let checkpoints: [u64; 8] = [4, 8, 16, 24, 32, 48, 64, 96];
        let mut g = 0u64;
        for &c in &checkpoints {
            while g < c {
                world.step_recursive();
                g += 1;
            }
            let pop_y = count_water_at(&world, floor_y + 1);
            let score = checkerboard_score(&world, floor_y + 1, lo - 2, hi + 2);
            eprintln!(
                "gen {c:>3}: pop={}  water@floor+1={pop_y}  cb_score={score:.2}",
                world.population()
            );
        }
        dump_floor(&world, floor_y + 1, "final (floor+1)");
        // Also dump floor+0 (should still be stone floor) for sanity.
        dump_floor(&world, floor_y, "final (floor exact y=stone)");
    }
}

/// Scenario B: pre-place a horizontal checkerboard layer of water on a flat
/// floor. If the rule is locally stable on a checkerboard, the pattern
/// persists. If not, it should compact within a few steps.
#[test]
#[ignore]
fn repro_horizontal_checkerboard_stability() {
    let level = 5; // 32³
    let side = 1u64 << level;

    for floor_parity_label in ["floor_y=4 (even)", "floor_y=5 (odd)"] {
        let floor_y = if floor_parity_label.contains("even") { 4 } else { 5 };
        let water_y = floor_y + 1;

        eprintln!("=== Scenario B — pre-placed horizontal checkerboard — {floor_parity_label} ===");
        let mut world = make_world_with_floor(level, floor_y);
        // Pre-place a checkerboard of water at floor+1 over a 12x12 region.
        let lo = side / 2 - 6;
        let hi = side / 2 + 6;
        let mut placed = 0u64;
        for z in lo..hi {
            for x in lo..hi {
                if (x + z) % 2 == 0 {
                    world.set(WorldCoord(x as i64), WorldCoord(water_y as i64), WorldCoord(z as i64), WATER);
                    placed += 1;
                }
            }
        }
        eprintln!("placed {placed} water cells in {}x{} checkerboard at y={water_y}", hi - lo, hi - lo);
        dump_floor(&world, water_y, "gen 0");
        let initial_score = checkerboard_score(&world, water_y, lo - 2, hi + 2);
        eprintln!("initial cb_score={initial_score:.2}");

        let checkpoints: [u64; 7] = [2, 4, 8, 16, 32, 64, 128];
        let mut g = 0u64;
        for &c in &checkpoints {
            while g < c {
                world.step_recursive();
                g += 1;
            }
            let pop_y = count_water_at(&world, water_y);
            let score = checkerboard_score(&world, water_y, lo - 2, hi + 2);
            eprintln!(
                "gen {c:>3}: pop={}  water@y={water_y}: {pop_y}  cb_score={score:.2}",
                world.population()
            );
        }
        dump_floor(&world, water_y, "final");
    }
}

/// Scenario C: brute vs recursive on the same checkerboard — does the
/// artifact appear in only one path or both?
#[test]
#[ignore]
fn repro_horizontal_checkerboard_brute_vs_recursive() {
    let level = 5; // 32³
    let side = 1u64 << level;
    let floor_y = 4;
    let water_y = floor_y + 1;
    let lo = side / 2 - 6;
    let hi = side / 2 + 6;

    fn build(level: u32, floor_y: u64, water_y: u64, lo: u64, hi: u64) -> World {
        let mut world = make_world_with_floor(level, floor_y);
        for z in lo..hi {
            for x in lo..hi {
                if (x + z) % 2 == 0 {
                    world.set(WorldCoord(x as i64), WorldCoord(water_y as i64), WorldCoord(z as i64), WATER);
                }
            }
        }
        world
    }

    let mut brute = build(level, floor_y, water_y, lo, hi);
    let mut recursive = build(level, floor_y, water_y, lo, hi);

    let max_gens = 32u64;
    eprintln!("=== Scenario C — brute vs recursive over {max_gens} gens ===");
    for g in 1..=max_gens {
        brute.step();
        recursive.step_recursive();
        let bs = checkerboard_score(&brute, water_y, lo - 2, hi + 2);
        let rs = checkerboard_score(&recursive, water_y, lo - 2, hi + 2);
        let bp = count_water_at(&brute, water_y);
        let rp = count_water_at(&recursive, water_y);
        eprintln!(
            "gen {g:>3}: brute(water={bp}, cb={bs:.2})  recursive(water={rp}, cb={rs:.2})"
        );
    }
    dump_floor(&brute, water_y, "brute final");
    dump_floor(&recursive, water_y, "recursive final");
}

/// Scenario D: pre-placed striped pattern (alternating x-rows) — distinguishes
/// "block rule fails to spread laterally" from "rule prefers checkerboard".
#[test]
#[ignore]
fn repro_horizontal_stripes_x() {
    let level = 5;
    let side = 1u64 << level;
    let floor_y = 4;
    let water_y = floor_y + 1;
    let lo = side / 2 - 6;
    let hi = side / 2 + 6;

    let mut world = make_world_with_floor(level, floor_y);
    for z in lo..hi {
        for x in lo..hi {
            if x % 2 == 0 {
                world.set(WorldCoord(x as i64), WorldCoord(water_y as i64), WorldCoord(z as i64), WATER);
            }
        }
    }
    eprintln!("=== Scenario D — x-stripes (alternating x-cols, full z-rows) ===");
    dump_floor(&world, water_y, "gen 0");

    let checkpoints: [u64; 6] = [2, 4, 8, 16, 32, 64];
    let mut g = 0u64;
    for &c in &checkpoints {
        while g < c {
            world.step_recursive();
            g += 1;
        }
        let score = checkerboard_score(&world, water_y, lo - 2, hi + 2);
        eprintln!(
            "gen {c:>3}: pop={} cb_score={score:.2}",
            world.population()
        );
    }
    dump_floor(&world, water_y, "final");
}

/// Scenario E: direct `FluidBlockRule::step_block` against a 2x2x2 block
/// holding the diagonal-water configuration that arises when a horizontal
/// checkerboard at the floor-adjacent layer falls inside a Margolus block.
/// If the block-rule is a fixed point on this configuration, we have the
/// mechanism explanation for Scenario B's stability.
///
/// Block layout: stone floor at y=0 (movable=false), checker-water at y=1.
/// Two diagonal arrangements: (0,1,0)+(1,1,1) and (1,1,0)+(0,1,1).
#[test]
#[ignore]
fn repro_fluid_block_rule_fixed_point_on_diagonal_water() {
    let rule = FluidBlockRule::new(material_density, material_phase, WATER_MATERIAL_ID);

    let stone_cell = Cell::pack(STONE_MATERIAL_ID, 0);
    let water_cell = Cell::pack(WATER_MATERIAL_ID, 0);
    let air_cell = Cell::EMPTY;

    fn name(cell: Cell) -> &'static str {
        if cell.is_empty() {
            "."
        } else if cell.material() == WATER_MATERIAL_ID {
            "~"
        } else if cell.material() == STONE_MATERIAL_ID {
            "#"
        } else {
            "?"
        }
    }

    fn dump_block(label: &str, block: &[Cell; 8]) {
        eprintln!("  {label}:");
        for dy in (0..2).rev() {
            for dz in 0..2 {
                let mut row = String::from("    ");
                for dx in 0..2 {
                    row.push_str(name(block[block_index(dx, dy, dz)]));
                }
                eprintln!("{row}    (dy={dy}, dz={dz})");
            }
        }
    }

    eprintln!("=== Scenario E — direct FluidBlockRule on diagonal-water block ===");

    // Configuration 1: water at (0,1,0) and (1,1,1); air at (1,1,0) and (0,1,1).
    let mut block: [Cell; 8] = [air_cell; 8];
    for dx in 0..2 {
        for dz in 0..2 {
            block[block_index(dx, 0, dz)] = stone_cell;
        }
    }
    block[block_index(0, 1, 0)] = water_cell;
    block[block_index(1, 1, 1)] = water_cell;
    let movable = [
        false, false, false, false, // y=0 stone, immovable
        true, true, true, true, // y=1 air/water, movable
    ];
    eprintln!("config 1 (diagonal A — water at (0,1,0) and (1,1,1)):");
    dump_block("input", &block);
    let out = rule.step_block(&block, &movable);
    dump_block("output", &out);
    let net_water = out
        .iter()
        .filter(|c| c.material() == WATER_MATERIAL_ID)
        .count();
    eprintln!("  net water cells: input=2, output={net_water}");
    let unchanged = out == block;
    eprintln!("  output == input: {unchanged}");
    assert_eq!(net_water, 2, "fluid count must be preserved");

    // Configuration 2: water at (1,1,0) and (0,1,1) — the other diagonal.
    let mut block2: [Cell; 8] = [air_cell; 8];
    for dx in 0..2 {
        for dz in 0..2 {
            block2[block_index(dx, 0, dz)] = stone_cell;
        }
    }
    block2[block_index(1, 1, 0)] = water_cell;
    block2[block_index(0, 1, 1)] = water_cell;
    eprintln!();
    eprintln!("config 2 (diagonal B — water at (1,1,0) and (0,1,1)):");
    dump_block("input", &block2);
    let out2 = rule.step_block(&block2, &movable);
    dump_block("output", &out2);
    let net_water2 = out2
        .iter()
        .filter(|c| c.material() == WATER_MATERIAL_ID)
        .count();
    eprintln!("  net water cells: input=2, output={net_water2}");
    let unchanged2 = out2 == block2;
    eprintln!("  output == input: {unchanged2}");
    assert_eq!(net_water2, 2, "fluid count must be preserved");

    eprintln!();
    eprintln!("Conclusion: if both `output == input` lines above print true, the");
    eprintln!("FluidBlockRule is a fixed point on a diagonal-water block above a");
    eprintln!("solid floor. The horizontal-checkerboard symptom from 14yp follows.");
}
