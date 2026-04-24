//! v3b2 water-bunching reproducer. Builds a small water pool, steps it N
//! generations with the default CA + block-rule + gap-fill pipeline, and
//! dumps a vertical y-slice as ASCII so a reader can see whether water
//! settles cleanly or bunches/striates.
//!
//! Run: `cargo test --profile bench --test repro_v3b2_water_bunching
//!       -- --ignored --nocapture`
//!
//! Scope: pure CPU, no GPU / winit. Visual artifacts observable only at
//! 256^3 via the app window are out of scope for this harness; this one
//! produces text output showing the column structure at scale=64^3 in
//! the default water scene.

use hash_thing::sim::World;
use hash_thing::terrain::TerrainParams;

fn ascii_cell(state: u16) -> char {
    // material lives in the upper 10 bits (METADATA_BITS = 6)
    let m = (state >> 6) as u32;
    match m {
        0 => '.',  // air
        1 => '#',  // stone
        2 => 'd',  // dirt
        3 => 'g',  // grass
        4 => 'f',  // fire
        5 => '~',  // water
        6 => 's',  // sand
        7 => 'L',  // lava
        8 => 'i',  // ice
        9 => 'a',  // acid
        _ => char::from_digit(m % 36, 36).unwrap_or('?'),
    }
}

fn dump_yz_slice(world: &World, x_fixed: u64, gen: u64) {
    use hash_thing::sim::world::WorldCoord;
    let side = world.side() as u64;
    eprintln!("--- y-z slice at x={x_fixed}, gen={gen}, side={side}³ ---");
    for y in (0..side).rev() {
        let mut row = String::with_capacity(side as usize);
        for z in 0..side {
            let state = world.get(
                WorldCoord(x_fixed as i64),
                WorldCoord(y as i64),
                WorldCoord(z as i64),
            );
            row.push(ascii_cell(state));
        }
        eprintln!("  y={y:>3}: {row}");
    }
    eprintln!();
}

fn column_summary(world: &World, x_fixed: u64, z_fixed: u64) -> String {
    use hash_thing::sim::world::WorldCoord;
    let side = world.side() as u64;
    let mut s = String::with_capacity(side as usize);
    for y in 0..side {
        let state = world.get(
            WorldCoord(x_fixed as i64),
            WorldCoord(y as i64),
            WorldCoord(z_fixed as i64),
        );
        s.push(ascii_cell(state));
    }
    s
}

/// Seed the default water scene at 64^3, step N generations, dump a slice
/// through the middle of the water pool.
#[test]
#[ignore]
fn repro_v3b2_water_bunching_64() {
    let level = 6u32;
    let side = 1u64 << level;
    let mut world = World::new(level);
    let params = TerrainParams::for_level(level);
    world
        .seed_terrain(&params)
        .expect("level-derived terrain params must validate");
    world.seed_water_and_sand();

    let x_mid = side / 2;
    let z_mid = side / 2;

    eprintln!("=== v3b2 repro: 64³ default water scene ===");
    eprintln!("initial pop: {}", world.population());
    dump_yz_slice(&world, x_mid, 0);

    let checkpoints = [10u64, 20, 30, 40, 60, 80];
    let mut last_gen = 0u64;
    for &target in &checkpoints {
        while last_gen < target {
            world.step_recursive();
            last_gen += 1;
        }
        eprintln!(
            "gen {target}, pop={}, center column (x={x_mid}, z={z_mid}): {}",
            world.population(),
            column_summary(&world, x_mid, z_mid),
        );
    }

    dump_yz_slice(&world, x_mid, last_gen);
}
