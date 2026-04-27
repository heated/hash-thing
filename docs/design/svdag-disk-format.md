# SVDAG on-disk format (v1)

Status: shipped (cswp.2). Source: `crates/ht-octree/src/disk.rs`.

## Goals

- **Byte-identical NodeStore round-trip.** Save then load reproduces the
  same nodes at the same NodeIds. Anything keyed on NodeId — caches,
  the renderer's `Svdag` slab, in-flight chunk-loaders — works after
  reload without remap acrobatics.
- **mmap-ready layout.** Fixed-size records at predictable offsets. v1
  ships a `Read`/`Write` API; the layout itself is alignment-clean and
  ready for `bytemuck::cast_slice` over an mmap'd page once cswp.5
  needs it.
- **Strictly validated load.** Corrupt files return typed `DiskError`,
  never panic. The plan-review round flagged `Cell::from_raw`'s
  release-panic on the forbidden range as the highest-priority case;
  the loader pre-validates every leaf state to prevent it.

## File layout

```
[0..32]              DiskHeader
[32..32 + 48*N]      N x DiskNode  (NodeId-positional: NodeId(i) <-> record i)
```

Total file size: exactly `32 + 48*N` bytes. A loader rejects any other
size with `SizeMismatch`.

### Header (32 bytes, little-endian)

| offset | size | field        | meaning                                                |
|--------|------|--------------|--------------------------------------------------------|
| 0      | 8    | `magic`      | `b"HTOCTRE\0"`                                         |
| 8      | 4    | `version`    | `1`                                                    |
| 12     | 4    | `flags`      | reserved; v1 requires `== 0`                           |
| 16     | 8    | `node_count` | length of the node array                               |
| 24     | 4    | `root_id`    | NodeId of the persisted root (`< node_count`)          |
| 28     | 4    | `root_level` | extent level of the root (≥ root node's intrinsic level) |

`root_level` is renderer/sim metadata: a `Leaf(s)` may serve as a uniform
root over a 32³ block, with `root_level=5` capturing that extent. The
loader validates `root_level >= store.get(root).level()`.

### Node record (48 bytes, little-endian)

| offset | size | field        | leaf semantics                          | interior semantics                  |
|--------|------|--------------|-----------------------------------------|-------------------------------------|
| 0      | 4    | `level`      | `u32::MAX` (sentinel)                   | actual level (≥ 1, < `u32::MAX`)    |
| 4      | 4    | `state`      | low 16 bits = `CellState`; high 16 = 0  | must be 0                           |
| 8      | 8    | `population` | must be 0                               | matches `sum(child populations)`    |
| 16     | 32   | `children`   | all 8 slots = 0                         | NodeIds of the 8 children           |

Total: 4 + 4 + 8 + 32 = **48 bytes**. The layout is naturally aligned to
8 bytes (record size is a multiple of 8, `population` sits at an 8-byte
offset within the record), which lets a future loader cast an mmap'd
page directly to `&[DiskNode]` without copying.

### Why `level == u32::MAX` for leaves

We need to distinguish leaf records from interior records on load. The
two practical options are a 1-byte tag or a sentinel value in an
existing field. A separate tag forces 4 bytes of padding to keep
alignment; folding the tag into `level` is alignment-free. `u32::MAX`
is dead space — we'll never have a 4-billion-deep octree (current worlds
sit at level 6–12), so reserving the maximum representable level as the
leaf marker costs nothing in practice.

## Save path

`save(store, root, root_level, w)` first runs `store.compacted(root)`,
which produces a fresh `NodeStore` containing only nodes reachable from
`root`. Compaction is post-order: a child is interned into the new
store before its parent. The resulting store therefore has a
**children-strictly-before-parent** ordering on NodeIds — a property
the load path depends on.

After compacting, save writes the header followed by one 48-byte record
per NodeId in order. NodeId stability across save/load follows directly:
record `i` becomes NodeId(`i`) on load.

## Load path

`load(r)` reads the header, validates magic / version / flags / sizing,
then walks records in order. Each record goes through the ordinary
public `NodeStore` API:

- **Leaf** records call `store.leaf(state)` (after inline validation
  has proven the state is outside the forbidden range, so the
  `Cell::from_raw` call inside `leaf` cannot panic).
- **Interior** records call `store.interior(level, children)`, which
  recomputes `population` from the already-interned children.

Because compaction guarantees children come first, every child NodeId
referenced by an interior record has already been added to the store
by the time the parent is processed. The intern map (`FxHashMap<Node,
NodeId>`) is rebuilt as a side effect of going through `intern_node`,
so post-load mutations dedup correctly against the loaded world.

### Validation rules

The loader enforces:

1. Magic, version, flags. Any deviation → typed error.
2. File size = `32 + 48 * node_count`. Truncation **and** trailing bytes
   reject with `SizeMismatch`.
3. `root_id < node_count`.
4. Record 0 is `Leaf(0)` (the canonical empty leaf).
5. Leaf records have `state >> 16 == 0`, `state` outside the forbidden
   range `1..=Cell::MAX_METADATA` (except at NodeId 0 where it must be
   exactly 0), `population == 0`, and all `children` slots are zero.
6. Interior records have `level >= 1` and `level != u32::MAX`,
   `state == 0`, every `children[k] < at` (forces topological order),
   and `population` equal to the recomputed sum of child populations.
7. `root_level >= store.get(root).level()`.
8. No two records dedup to the same `Node` (compaction guarantees
   uniqueness; a duplicate signals a hand-crafted file).

### Why recompute population on load

Hash-cons keys on the tuple `(level, children, population)`. A
corrupt-but-internally-consistent population would create a node that
*cannot* dedup against future `interior(level, children)` calls,
silently doubling memory. The recomputation is O(N) anyway —
compaction's child-before-parent ordering means the population fold
runs in a single pass. For v2, a per-file CRC32C (gated by `flags &
0x1`) would let us trade the recompute for a faster integrity check.

## Versioning

`version = 1` is locked. Future `flags` bit assignments:

- bit 0: per-node CRC32C trailer present (v2)
- bit 1: optional sidecar payload appended after the node array (v2)

A v1 loader rejects any nonzero flag bit. A future major version bump
(`version > 1`) returns `UnsupportedVersion(v)` rather than guessing.

## Out of scope (deferred to cswp.5)

- mmap-faulted loader. The byte layout is mmap-ready; the v1 API is
  `Read`-based for portability.
- Cross-file hash-cons. cswp.5 will own a long-lived shared `NodeStore`
  that loaded chunks intern into; per-file dedup is automatic via the
  existing `intern_node` path.
- Chunk index (chunk-key → byte offset). The streaming runtime will
  sit *above* the format, addressing chunks by filename. v1 is one
  store per file.

## Trade-offs explicitly accepted

- **Endianness:** declared LE-only. A BE host would produce/consume
  files with byte-swapped fields; the build fails with a
  `compile_error!` on BE targets to make the limitation loud.
- **No CRC in v1.** Population recompute catches the corruption case
  that breaks hash-cons; bit-flip detection in cold reads is deferred
  to v2 via `flags & 0x1`.
- **Header is 32 bytes, no padding for future fields.** v2 adds via
  the flag-payload mechanism rather than reserving empty bytes we
  might guess wrong about.
