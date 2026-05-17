# Graph Hashlife Probe (ltt5)

**Status:** research-only CPU toy probe, hash-thing-ltt5, 2026-05-03.

This probe does not touch the production SVDAG renderer or GPU upload path. It
tests only whether content-addressed memoization can find repeated work on a
non-cube topology with changing, non-uniform state.

## Probe

`tests/bench_graph_hashlife.rs` builds a 48-module irregular factory graph
(1152 nodes). Each module has identical internal topology, sparse cross-module
links, five material classes, and six initial state phases. The update rule is a
small graph CA: each node updates from its own state, material, degree, and the
count of active neighbors. The scene runs 120 generations.

The radius-2 memo key is a canonicalized local neighborhood signature used to
return one node's next state: layer distance, material, degree, in-radius degree,
and current node state. It is a toy stand-in for "subgraph content + local
shape," not a full graph-isomorphism engine and not a radius-1 multi-node
evolution cache.

## Result

At `small·graph-cellular·passive-active·n/a` (toy 48-module graph), the probe
reported:

```text
updates=138240 changed=119200 final_state_kinds=6
naive_misses=16790 naive_elision=8.23x
radius2_misses=17280 radius2_elision=8.00x
```

This clears ltt5's per-probe bar of 5x elision on a non-uniform changing graph.
It does not prove a production graph-hashlife engine is worth building. It also
does not show radius-2 memo is better than a simpler local-state cache: the
naive cache is slightly stronger in this toy (8.23x vs 8.00x). The evidence is
narrower: graph-local content-addressed reuse exists on this non-cube topology,
so the research lead stays alive.

## Game Unlock

The interesting game shape is not "Minecraft but graph-shaped." It is a factory
or circuit world where the substrate is a designed network: pipes, conveyors,
signals, gates, and machines arranged in repeated motifs that are not embedded
in a cubic grid. A graph memo could let repeated subcircuits or production cells
simulate once and reuse across the network while still allowing non-uniform,
phase-shifted dynamics.

## Next Decision

Stay in scope-a unless a navigation review explicitly promotes the lead. The
next probe should cache a radius-1 multi-node result from a real canonical
subgraph hash, then try one less-regular scene. Any production path still
requires a separate renderer/SVDAG rewrite decision.
