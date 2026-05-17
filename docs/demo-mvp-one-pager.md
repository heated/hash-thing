# Demo MVP One-Pager: 3D Powder Game

## Thesis

The MVP is a real-time 3D Powder Game sandbox: many cellular-automata materials
running together at fine voxel resolution, in a world large enough to feel
surprising, with god-like tools for creating and disturbing the simulation.

This is a game-shaped engine proof. The player edits the world now and the world
reacts now. No combat, spells, enemies, roguelite structure, or time-skip
mechanic belongs in the MVP.

## Experience

Start in a constrained volcanic/lattice cavern that reads at human scale: stone
walls, sand shelves, water pockets, lava seams, vents, vines/wood, and clone
sources. The first minute is free play with obvious tools and immediate material
reactions. The reveal is a pull-back or fly-out from the chamber into a much
larger generated structure, showing that the sandbox is not a tiny contained
box.

Target demo shape:

- 30 seconds: launch, spawn/edit materials, trigger one obvious cascade, reveal
  scale.
- 1-3 minutes: free play in the same scene, with enough tools to create new
  interactions without a tutorial layer.
- Infinite tail: keep playing with the sandbox after the scripted arc is over.

## Camera

Ship free-fly god-cam as the default. It matches the sandbox verb set: move fast,
look anywhere, paint anywhere, and pull back for scale.

Add a first-person walking toggle if it stays cheap. Walking sells voxel scale
and makes the big-world reveal land, but it should not be the primary editing
mode.

## Materials

MVP target: 10-13 materials, not a two-material demo.

Core set:

- Stone and dirt: structure and terrain.
- Sand and dust: granular collapse.
- Water and steam: flow, fill, rise, phase contrast.
- Lava and fire: heat, ignition, solidification pressure.
- Oil or gas: flammable spread.
- Ice: cooling/freezing contrast.
- Acid: destructive fluid.
- Wood/plant/vine: burnable/growing organic material.
- Clone/source: continuous material injection for sustained scenes.

Nice-to-have after the core set works: fan/wind, metal/heat conductor, gunpowder,
mercury/heavy fluid.

## Tools

Use diegetic god-tool verbs, not a Photoshop-style panel.

MVP tool palette:

- Brush spawn selected material.
- Brush erase / carve.
- Grab and throw a small chunk or blob.
- Ignite / heat.
- Cool / freeze.
- Gust / push.
- Clone/source placer.
- Brush size and material selection.

The minimum viable interaction loop is: spawn material, disturb it, watch it
react, then combine it with another material.

## World And Scale

Use one visually rich generated scene, not a menu of small test boxes. The scene
should justify the material orchestra: lava seams below, water pockets above,
sand shelves, vents, organic patches, and structural voids.

Voxel target is roughly 25 cm, 4x finer per axis than Minecraft-scale blocks.
The world can start at 256^3 for the first shippable demo, but the scene should
be framed so 4096^3 streaming work naturally extends it instead of replacing it.

## Success Criteria

- The world visibly reacts while the player edits it.
- At least 8 materials are active in one scene without feeling staged one at a
  time.
- Sand, water, smoke/steam, and fire/lava each read as distinct behavior.
- The viewer gets one scale surprise.
- The demo still works as free play after the reveal.
- Engine-fit evidence remains positive at the target scale; if perf walls appear,
  the demo reports that honestly instead of hiding it behind a smaller box.

## Explicitly Post-MVP

- Combat, enemies, and AI agents.
- Wand/spell systems.
- Roguelite/run structure.
- Multiplayer.
- Time-skip or rewind as a player mechanic.
- Save/share scene browser.
- Alternate game directions: factory, Zachtronics/puzzle, Noita-like action,
  giant-on-living-world, temporal/4D, graph/non-Euclidean.
