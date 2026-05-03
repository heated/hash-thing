//! Research-only graph-hashlife probe for hash-thing-ltt5.
//!
//! This intentionally does not integrate with the production SVDAG renderer.
//! It asks one narrow question: can content-addressed radius-2 graph
//! neighborhoods elide repeated single-node CA updates on a non-cube topology
//! with changing, non-uniform state?

use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug)]
struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    material: u8,
}

#[derive(Clone, Copy, Debug)]
struct ProbeStats {
    total_updates: usize,
    naive_misses: usize,
    radius2_misses: usize,
    changed_updates: usize,
    final_state_kinds: usize,
}

#[test]
fn graph_hashlife_radius2_probe_elides_repeated_nonuniform_modules() {
    let graph = modular_factory_graph(48);
    let stats = run_probe(&graph, 120);
    let radius2_elision = stats.total_updates as f64 / stats.radius2_misses as f64;
    let naive_elision = stats.total_updates as f64 / stats.naive_misses as f64;

    eprintln!(
        "graph-hashlife probe: updates={} changed={} final_state_kinds={} naive_misses={} naive_elision={:.2}x radius2_misses={} radius2_elision={:.2}x",
        stats.total_updates,
        stats.changed_updates,
        stats.final_state_kinds,
        stats.naive_misses,
        naive_elision,
        stats.radius2_misses,
        radius2_elision,
    );

    assert!(
        stats.changed_updates > stats.total_updates / 3,
        "probe must have non-trivial dynamics"
    );
    assert!(
        stats.final_state_kinds >= 4,
        "probe must remain non-uniform at the end"
    );
    assert!(
        radius2_elision >= 5.0,
        "radius-2 neighborhood memo should clear ltt5's 5x per-probe bar"
    );
}

fn modular_factory_graph(module_count: usize) -> Graph {
    const MODULE_NODES: usize = 24;
    let mut nodes = Vec::with_capacity(module_count * MODULE_NODES);
    let mut edges = vec![Vec::new(); module_count * MODULE_NODES];

    for module in 0..module_count {
        for local in 0..MODULE_NODES {
            nodes.push(Node {
                material: ((local / 4 + local % 3) % 5) as u8,
            });
        }
        for local in 0..MODULE_NODES {
            let a = module * MODULE_NODES + local;
            connect(
                &mut edges,
                a,
                module * MODULE_NODES + ((local + 1) % MODULE_NODES),
            );
            connect(
                &mut edges,
                a,
                module * MODULE_NODES + ((local + 6) % MODULE_NODES),
            );
            if local % 4 == 0 {
                connect(
                    &mut edges,
                    a,
                    module * MODULE_NODES + ((local + 11) % MODULE_NODES),
                );
            }
        }
    }

    for module in 0..module_count {
        let next = (module + 1) % module_count;
        let prev = (module + module_count - 1) % module_count;
        connect(
            &mut edges,
            module * MODULE_NODES + 3,
            next * MODULE_NODES + 19,
        );
        connect(
            &mut edges,
            module * MODULE_NODES + 17,
            prev * MODULE_NODES + 5,
        );
    }

    Graph { nodes, edges }
}

fn connect(edges: &mut [Vec<usize>], a: usize, b: usize) {
    if !edges[a].contains(&b) {
        edges[a].push(b);
    }
    if !edges[b].contains(&a) {
        edges[b].push(a);
    }
}

fn run_probe(graph: &Graph, generations: usize) -> ProbeStats {
    let mut state = initial_state(graph);
    let mut naive_cache = HashMap::new();
    let mut radius2_cache = HashMap::new();
    let mut changed_updates = 0;

    for _ in 0..generations {
        let mut next = vec![0u8; state.len()];
        for node in 0..graph.nodes.len() {
            let naive_key = naive_signature(graph, &state, node);
            naive_cache
                .entry(naive_key)
                .or_insert_with(|| step_node(graph, &state, node));

            let radius2_key = radius_signature(graph, &state, node, 2);
            let value = *radius2_cache
                .entry(radius2_key)
                .or_insert_with(|| step_node(graph, &state, node));
            next[node] = value;
            if value != state[node] {
                changed_updates += 1;
            }
        }
        state = next;
    }

    ProbeStats {
        total_updates: graph.nodes.len() * generations,
        naive_misses: naive_cache.len(),
        radius2_misses: radius2_cache.len(),
        changed_updates,
        final_state_kinds: state.into_iter().collect::<BTreeSet<_>>().len(),
    }
}

fn initial_state(graph: &Graph) -> Vec<u8> {
    graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let module = i / 24;
            let local = i % 24;
            ((module % 6 + local % 7 + node.material as usize) % 6) as u8
        })
        .collect()
}

fn step_node(graph: &Graph, state: &[u8], node: usize) -> u8 {
    let material = graph.nodes[node].material;
    let degree = graph.edges[node].len() as u8;
    let active_neighbors = graph.edges[node]
        .iter()
        .filter(|&&neighbor| state[neighbor] >= 3)
        .count() as u8;
    (state[node] + active_neighbors + material + degree) % 6
}

fn naive_signature(graph: &Graph, state: &[u8], node: usize) -> Vec<u8> {
    let mut key = vec![
        graph.nodes[node].material,
        graph.edges[node].len() as u8,
        state[node],
    ];
    let mut neighbors = graph.edges[node]
        .iter()
        .map(|&n| {
            (
                graph.nodes[n].material,
                graph.edges[n].len() as u8,
                state[n],
            )
        })
        .collect::<Vec<_>>();
    neighbors.sort_unstable();
    for (material, degree, neighbor_state) in neighbors {
        key.extend([material, degree, neighbor_state]);
    }
    key
}

fn radius_signature(graph: &Graph, state: &[u8], origin: usize, radius: usize) -> Vec<u8> {
    let mut layers: BTreeMap<usize, Vec<(u8, u8, u8, u8)>> = BTreeMap::new();
    let mut frontier = BTreeSet::from([origin]);
    let mut seen = BTreeSet::from([origin]);

    for distance in 0..=radius {
        let mut layer = Vec::new();
        for &node in &frontier {
            let in_radius_degree = graph.edges[node]
                .iter()
                .filter(|neighbor| seen.contains(neighbor) || distance < radius)
                .count() as u8;
            layer.push((
                graph.nodes[node].material,
                graph.edges[node].len() as u8,
                in_radius_degree,
                state[node],
            ));
        }
        layer.sort_unstable();
        layers.insert(distance, layer);

        let mut next = BTreeSet::new();
        for node in frontier {
            for &neighbor in &graph.edges[node] {
                if seen.insert(neighbor) {
                    next.insert(neighbor);
                }
            }
        }
        frontier = next;
    }

    let mut key = Vec::new();
    for (distance, layer) in layers {
        key.push(distance as u8);
        key.push(layer.len() as u8);
        for (material, degree, in_radius_degree, node_state) in layer {
            key.extend([material, degree, in_radius_degree, node_state]);
        }
    }
    key
}
