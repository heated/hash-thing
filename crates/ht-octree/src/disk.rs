//! On-disk serialization for [`NodeStore`].
//!
//! Format spec lives in `docs/design/svdag-disk-format.md`. Quick reference:
//!
//! ```text
//! [0..32]              DiskHeader (LE-only)
//! [32..32 + 48*N]      N x DiskNode, NodeId-positional
//! ```
//!
//! Every record is a fixed 48 bytes; a record's file index *is* its NodeId.
//! Compaction is performed on `save`, which puts children before parents in
//! the file — load can then walk records in order and feed each one through
//! the ordinary `NodeStore::leaf` / `NodeStore::interior` API with all
//! children already interned.

use crate::node::{Cell, Node, NodeId};
use crate::store::NodeStore;
use std::io::{Read, Write};

/// 8-byte magic prefix. `\0` terminator catches truncated/zero-padded files.
pub const MAGIC: [u8; 8] = *b"HTOCTRE\0";
/// Format version. v1 requires `flags == 0`. Bit assignments for v2:
///   - bit 0: per-node CRC32C trailer present (deferred to v2).
///   - bit 1: optional sidecar payload appended after the node array.
pub const VERSION: u32 = 1;
/// Total header size, in bytes.
pub const HEADER_BYTES: usize = 32;
/// Total per-node record size, in bytes.
pub const NODE_BYTES: usize = 48;

/// Sentinel `level` value indicating a record is a leaf, not an interior
/// node. `u32::MAX` is dead space in any sane octree (we'll never have
/// 4-billion-deep trees) and folds the leaf/interior tag into an existing
/// field, keeping the on-disk layout alignment-clean.
const LEAF_LEVEL: u32 = u32::MAX;

#[cfg(target_endian = "big")]
compile_error!("ht-octree disk format is little-endian only; this build is big-endian");

/// All errors `save` and `load` can return. Every load-time failure produces
/// a typed error — corrupt files do *not* panic.
#[derive(Debug)]
pub enum DiskError {
    /// Underlying I/O failure on the reader/writer.
    Io(std::io::Error),
    /// File magic did not match `b"HTOCTRE\0"`.
    BadMagic { found: [u8; 8] },
    /// File `version` field did not equal `VERSION`.
    UnsupportedVersion(u32),
    /// File `flags` had an unrecognized bit set.
    UnknownFlags(u32),
    /// Reader length did not match `HEADER_BYTES + NODE_BYTES * node_count`.
    SizeMismatch { expected: u64, found: u64 },
    /// Header `root_id` was out of bounds for the persisted node count.
    InvalidRootId { root_id: u32, node_count: u64 },
    /// Interior `children[k]` referenced a NodeId outside the file or
    /// later than the parent — children must come before parents (a
    /// consequence of compaction-on-save).
    InvalidNodeId { at: u64, child: u32, max: u32 },
    /// Interior `level == 0`. Level 0 is leaf-only.
    InvalidLevel { at: u64, level: u32 },
    /// Persisted `root_level` is below the actual root node's `level()`.
    /// `root_level >= node.level()` is required so the renderer / sim can
    /// treat the loaded root as covering at least the persisted extent.
    InvalidRootLevel { node_level: u32, root_level: u32 },
    /// Leaf at `NodeId(0)` had nonzero state. The empty-leaf invariant is
    /// load-bearing for hash-cons.
    EmptyLeafCorrupted { state: u16 },
    /// Leaf state lay in the forbidden range `1..=Cell::MAX_METADATA` —
    /// would decode to "empty with flavor" via [`Cell::from_raw`] and
    /// silently alias `NodeId::EMPTY` after interning.
    BadLeafState { at: u64, state: u16 },
    /// `state` field had nonzero high 16 bits (leaf) or any nonzero bits
    /// (interior). Raw leaf state is `u16`; the high 16 bits of the on-
    /// disk `u32` field must be zero.
    BadStatePayload { at: u64, state: u32 },
    /// Interior `population` did not match `sum(child populations)`. Load
    /// recomputes population during interning and rejects mismatches —
    /// without this check a corrupt population would break hash-cons
    /// dedup, since the intern key is `(level, children, population)`.
    BadPopulation { at: u64, expected: u64, found: u64 },
    /// Two records hashed to the same `Node`. Save's compaction pass
    /// guarantees uniqueness; this signals a corrupt / hand-crafted file.
    DuplicateRecord { at: u64, dedup_to: u32 },
}

impl From<std::io::Error> for DiskError {
    fn from(e: std::io::Error) -> Self {
        DiskError::Io(e)
    }
}

impl std::fmt::Display for DiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for DiskError {}

/// Serialize a compacted view of `store` rooted at `root` to `w`.
///
/// `store` is compacted before writing, so only nodes reachable from `root`
/// (plus `NodeId::EMPTY`) end up in the file. After compaction the children
/// of every interior node have strictly smaller NodeIds than the interior;
/// that ordering is the load path's input contract.
pub fn save(
    store: &NodeStore,
    root: NodeId,
    root_level: u32,
    w: &mut impl Write,
) -> Result<(), DiskError> {
    let (compact, new_root) = store.compacted(root);

    let node_count = compact.node_count() as u64;
    let mut header = [0u8; HEADER_BYTES];
    header[0..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&0u32.to_le_bytes()); // flags
    header[16..24].copy_from_slice(&node_count.to_le_bytes());
    header[24..28].copy_from_slice(&new_root.0.to_le_bytes());
    header[28..32].copy_from_slice(&root_level.to_le_bytes());
    w.write_all(&header)?;

    for i in 0..compact.node_count() {
        let id = NodeId(i as u32);
        let mut buf = [0u8; NODE_BYTES];
        match compact.get(id) {
            Node::Leaf(state) => {
                buf[0..4].copy_from_slice(&LEAF_LEVEL.to_le_bytes());
                buf[4..8].copy_from_slice(&u32::from(*state).to_le_bytes());
                // population (8) and children (32) stay zeroed.
            }
            Node::Interior {
                level,
                children,
                population,
            } => {
                buf[0..4].copy_from_slice(&level.to_le_bytes());
                // state stays zero.
                buf[8..16].copy_from_slice(&population.to_le_bytes());
                for (k, child) in children.iter().enumerate() {
                    let off = 16 + k * 4;
                    buf[off..off + 4].copy_from_slice(&child.0.to_le_bytes());
                }
            }
        }
        w.write_all(&buf)?;
    }

    Ok(())
}

/// Deserialize a `NodeStore` from `r`. Returns `(store, root, root_level)`.
///
/// Validates every documented format invariant. Corrupt files return
/// `DiskError`, never panic.
pub fn load(r: &mut impl Read) -> Result<(NodeStore, NodeId, u32), DiskError> {
    let mut header = [0u8; HEADER_BYTES];
    r.read_exact(&mut header)?;

    let mut magic = [0u8; 8];
    magic.copy_from_slice(&header[0..8]);
    if magic != MAGIC {
        return Err(DiskError::BadMagic { found: magic });
    }
    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(DiskError::UnsupportedVersion(version));
    }
    let flags = u32::from_le_bytes(header[12..16].try_into().unwrap());
    if flags != 0 {
        return Err(DiskError::UnknownFlags(flags));
    }
    let node_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let root_id_raw = u32::from_le_bytes(header[24..28].try_into().unwrap());
    let root_level = u32::from_le_bytes(header[28..32].try_into().unwrap());

    if (root_id_raw as u64) >= node_count {
        return Err(DiskError::InvalidRootId {
            root_id: root_id_raw,
            node_count,
        });
    }

    let expected_size = HEADER_BYTES as u64 + NODE_BYTES as u64 * node_count;
    let mut store = NodeStore::new();

    for at in 0..node_count {
        let mut buf = [0u8; NODE_BYTES];
        match r.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(DiskError::SizeMismatch {
                    expected: expected_size,
                    found: HEADER_BYTES as u64 + NODE_BYTES as u64 * at,
                });
            }
            Err(e) => return Err(DiskError::Io(e)),
        }

        let level = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let state = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let population = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let mut children = [NodeId::EMPTY; 8];
        for (k, slot) in children.iter_mut().enumerate() {
            let off = 16 + k * 4;
            *slot = NodeId(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }

        let id = if level == LEAF_LEVEL {
            if state >> 16 != 0 {
                return Err(DiskError::BadStatePayload { at, state });
            }
            let raw = state as u16;
            if at == 0 {
                if raw != 0 {
                    return Err(DiskError::EmptyLeafCorrupted { state: raw });
                }
            } else if raw != 0 && raw <= Cell::MAX_METADATA {
                return Err(DiskError::BadLeafState { at, state: raw });
            }
            if population != 0 {
                return Err(DiskError::BadPopulation {
                    at,
                    expected: 0,
                    found: population,
                });
            }
            for child in &children {
                if child.0 != 0 {
                    return Err(DiskError::InvalidNodeId {
                        at,
                        child: child.0,
                        max: 0,
                    });
                }
            }
            // Inline validation has already proven `Cell::from_raw` cannot
            // panic, so `store.leaf(raw)` is safe.
            store.leaf(raw)
        } else {
            if level == 0 {
                return Err(DiskError::InvalidLevel { at, level });
            }
            if at == 0 {
                // First record must be the canonical empty leaf.
                return Err(DiskError::EmptyLeafCorrupted { state: 0 });
            }
            if state != 0 {
                return Err(DiskError::BadStatePayload { at, state });
            }
            // Children must be strictly less than `at` so `interior` can
            // fold their populations from the in-progress store. This is
            // the contract compaction guarantees on save.
            for child in &children {
                if (child.0 as u64) >= at {
                    return Err(DiskError::InvalidNodeId {
                        at,
                        child: child.0,
                        max: at as u32,
                    });
                }
            }
            let id = store.interior(level, children);
            let actual_pop = store.population(id);
            if actual_pop != population {
                return Err(DiskError::BadPopulation {
                    at,
                    expected: actual_pop,
                    found: population,
                });
            }
            id
        };

        if id.0 as u64 != at {
            return Err(DiskError::DuplicateRecord { at, dedup_to: id.0 });
        }
    }

    let mut probe = [0u8; 1];
    match r.read(&mut probe) {
        Ok(0) => {}
        Ok(n) => {
            return Err(DiskError::SizeMismatch {
                expected: expected_size,
                found: expected_size + n as u64,
            });
        }
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {}
        Err(e) => return Err(DiskError::Io(e)),
    }

    let root = NodeId(root_id_raw);
    let actual_root_level = store.get(root).level();
    if root_level < actual_root_level {
        return Err(DiskError::InvalidRootLevel {
            node_level: actual_root_level,
            root_level,
        });
    }

    Ok((store, root, root_level))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn save_to_vec(store: &NodeStore, root: NodeId, root_level: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        save(store, root, root_level, &mut buf).expect("save");
        buf
    }

    fn load_from_slice(bytes: &[u8]) -> Result<(NodeStore, NodeId, u32), DiskError> {
        let mut cursor = Cursor::new(bytes);
        load(&mut cursor)
    }

    /// Helper: assert load failed and return the error. Avoids needing
    /// `Debug` on the `Ok` side, which `NodeStore` deliberately doesn't
    /// implement (the store is not human-printable).
    fn expect_err(r: Result<(NodeStore, NodeId, u32), DiskError>) -> DiskError {
        match r {
            Ok(_) => panic!("expected DiskError, got Ok"),
            Err(e) => e,
        }
    }

    // --- Round-trip tests ---

    #[test]
    fn roundtrip_empty() {
        let store = NodeStore::new();
        let bytes = save_to_vec(&store, NodeId::EMPTY, 0);
        let (loaded, root, root_level) = load_from_slice(&bytes).unwrap();
        assert_eq!(loaded.node_count(), 1);
        assert_eq!(root, NodeId::EMPTY);
        assert_eq!(root_level, 0);
    }

    #[test]
    fn roundtrip_single_cell() {
        let mut store = NodeStore::new();
        let level = 5u32;
        let mut root = store.empty(level);
        let cell = Cell::pack(7, 0).raw();
        root = store.set_cell(root, 10, 20, 30, cell);

        let bytes = save_to_vec(&store, root, level);
        let (loaded, lroot, lroot_level) = load_from_slice(&bytes).unwrap();
        assert_eq!(lroot_level, level);
        assert_eq!(loaded.get_cell(lroot, 10, 20, 30), cell);
        assert_eq!(loaded.get_cell(lroot, 0, 0, 0), 0);
    }

    #[test]
    fn roundtrip_full_grid() {
        let side: usize = 16;
        let level = side.trailing_zeros();
        let mut grid = vec![0u16; side * side * side];
        // Sparse pattern: every cell where (x ^ y ^ z) % 5 == 0 gets a
        // distinct material to stress hash-cons.
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    if (x ^ y ^ z) % 5 == 0 {
                        let mat = ((x + y + z) % 100 + 1) as u16;
                        grid[x + y * side + z * side * side] = Cell::pack(mat, 0).raw();
                    }
                }
            }
        }
        let mut store = NodeStore::new();
        let root = store.from_flat(&grid, side);

        let bytes = save_to_vec(&store, root, level);
        let (loaded, lroot, _) = load_from_slice(&bytes).unwrap();
        let loaded_grid = loaded.flatten(lroot, side);
        assert_eq!(loaded_grid, grid);
    }

    #[test]
    fn roundtrip_node_count_stable() {
        let mut store = NodeStore::new();
        let level = 4u32;
        let mut root = store.empty(level);
        for i in 0..16 {
            root = store.set_cell(root, i, 0, 0, Cell::pack(1, 0).raw());
        }
        let (compact, _) = store.compacted(root);
        let expected = compact.node_count();

        let bytes = save_to_vec(&store, root, level);
        let (loaded, _, _) = load_from_slice(&bytes).unwrap();
        assert_eq!(loaded.node_count(), expected);
    }

    #[test]
    fn roundtrip_population_preserved() {
        let mut store = NodeStore::new();
        let level = 4u32;
        let mut root = store.empty(level);
        for i in 0..7u64 {
            root = store.set_cell(root, i, i, i, Cell::pack(2, 0).raw());
        }
        let want_pop = store.population(root);

        let bytes = save_to_vec(&store, root, level);
        let (loaded, lroot, _) = load_from_slice(&bytes).unwrap();
        assert_eq!(loaded.population(lroot), want_pop);
    }

    #[test]
    fn roundtrip_after_compact() {
        let mut store = NodeStore::new();
        let level = 5u32;
        let mut root = store.empty(level);
        // Two rounds of writes with a compaction in between, to exercise
        // the post-compact-write path (no stale subtrees in the file).
        for i in 0..8u64 {
            root = store.set_cell(root, i, 0, 0, Cell::pack(3, 0).raw());
        }
        let (mut compact, croot) = store.compacted(root);
        let mut root = croot;
        for i in 0..8u64 {
            root = compact.set_cell(root, 16, i, 0, Cell::pack(4, 0).raw());
        }

        let bytes = save_to_vec(&compact, root, level);
        let (loaded, lroot, _) = load_from_slice(&bytes).unwrap();
        for i in 0..8u64 {
            assert_eq!(
                loaded.get_cell(lroot, i, 0, 0),
                Cell::pack(3, 0).raw(),
                "x={i}"
            );
            assert_eq!(
                loaded.get_cell(lroot, 16, i, 0),
                Cell::pack(4, 0).raw(),
                "y={i}"
            );
        }
    }

    #[test]
    fn roundtrip_leaf_root_with_root_level() {
        // A `Leaf(s)` at root_level=5 represents a uniform 32^3 region of
        // `s`. The renderer relies on this exact case (uniform world cube).
        let mut store = NodeStore::new();
        let leaf = store.leaf(Cell::pack(9, 0).raw());
        let bytes = save_to_vec(&store, leaf, 5);
        let (loaded, lroot, lroot_level) = load_from_slice(&bytes).unwrap();
        // root_level is renderer/sim metadata describing the extent the
        // leaf-root represents (here, a 32³ uniform block). It must
        // round-trip exactly. The intrinsic node level is still 0 — that's
        // the contract `get_cell` operates against.
        assert_eq!(lroot_level, 5);
        assert_eq!(loaded.get(lroot).level(), 0);
        assert_eq!(loaded.get_cell(lroot, 0, 0, 0), Cell::pack(9, 0).raw());
    }

    #[test]
    fn byte_size_matches_formula() {
        for &level in &[0u32, 2, 4, 6] {
            let mut store = NodeStore::new();
            let root = store.empty(level);
            let bytes = save_to_vec(&store, root, level);
            let (compact, _) = store.compacted(root);
            let want = HEADER_BYTES + NODE_BYTES * compact.node_count();
            assert_eq!(bytes.len(), want, "level={level}");
        }
    }

    #[test]
    fn roundtrip_intern_dedup_after_load() {
        // Save / load, then mutate the loaded store. The intern map must
        // be rehydrated such that re-creating an existing subtree dedups
        // back to its existing NodeId rather than appending a duplicate.
        let mut store = NodeStore::new();
        let level = 3u32;
        let mut root = store.empty(level);
        root = store.set_cell(root, 0, 0, 0, Cell::pack(1, 0).raw());
        let bytes = save_to_vec(&store, root, level);
        let (mut loaded, lroot, _) = load_from_slice(&bytes).unwrap();
        let before = loaded.node_count();

        // Re-set the same cell to the same value. Nothing should change
        // in the store — every interior on the path is already canonical.
        let lroot2 = loaded.set_cell(lroot, 0, 0, 0, Cell::pack(1, 0).raw());
        assert_eq!(lroot, lroot2);
        assert_eq!(loaded.node_count(), before);
    }

    // --- Header validation ---

    #[test]
    fn header_validation_bad_magic() {
        let store = NodeStore::new();
        let mut bytes = save_to_vec(&store, NodeId::EMPTY, 0);
        bytes[0] = b'X';
        match expect_err(load_from_slice(&bytes)) {
            DiskError::BadMagic { .. } => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn header_validation_bad_version() {
        let store = NodeStore::new();
        let mut bytes = save_to_vec(&store, NodeId::EMPTY, 0);
        bytes[8..12].copy_from_slice(&999u32.to_le_bytes());
        match expect_err(load_from_slice(&bytes)) {
            DiskError::UnsupportedVersion(999) => {}
            other => panic!("expected UnsupportedVersion(999), got {other:?}"),
        }
    }

    #[test]
    fn header_validation_unknown_flags() {
        let store = NodeStore::new();
        let mut bytes = save_to_vec(&store, NodeId::EMPTY, 0);
        bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
        match expect_err(load_from_slice(&bytes)) {
            DiskError::UnknownFlags(1) => {}
            other => panic!("expected UnknownFlags(1), got {other:?}"),
        }
    }

    // --- Per-record validation ---

    fn write_le_u32(buf: &mut [u8], offset: usize, value: u32) {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn write_le_u64(buf: &mut [u8], offset: usize, value: u64) {
        buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    /// Build a minimal valid file with: NodeId(0) = Leaf(0), one interior
    /// at NodeId(1) of `level=1` with all 8 children = NodeId(0). Then
    /// hand the tweaked bytes back to the caller for corruption.
    fn minimal_file() -> Vec<u8> {
        let mut store = NodeStore::new();
        let empty1 = store.empty(1);
        save_to_vec(&store, empty1, 1)
    }

    fn record_offset(at: usize) -> usize {
        HEADER_BYTES + at * NODE_BYTES
    }

    #[test]
    fn node_validation_child_oob() {
        let mut bytes = minimal_file();
        // The interior at NodeId(1) has 8 children — corrupt children[0].
        // It currently points to NodeId(0); set it to a NodeId beyond
        // node_count.
        let off = record_offset(1) + 16; // children[0] starts at offset 16
        write_le_u32(&mut bytes, off, 0xFFFF_FFFF);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::InvalidNodeId { at: 1, child, .. } => {
                assert_eq!(child, 0xFFFF_FFFF);
            }
            other => panic!("expected InvalidNodeId, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_empty_leaf_nonzero() {
        let mut bytes = minimal_file();
        // NodeId(0) is the canonical empty leaf — its state is at record
        // offset 4..8. Plant a value of 0x100 (out of forbidden range to
        // bypass BadLeafState; this is the EmptyLeafCorrupted path).
        let off = record_offset(0) + 4;
        write_le_u32(&mut bytes, off, 0x100);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::EmptyLeafCorrupted { state: 0x100 } => {}
            other => panic!("expected EmptyLeafCorrupted, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_bad_leaf_state() {
        // A leaf at NodeId>0 with state in the forbidden range 1..=63
        // would panic via Cell::from_raw if interned. Loader must Err.
        let mut store = NodeStore::new();
        let leaf = store.leaf(Cell::pack(2, 0).raw()); // valid leaf
        let mut bytes = save_to_vec(&store, leaf, 0);
        // Corrupt that leaf's state to 42 (forbidden).
        let off = record_offset(leaf.0 as usize) + 4;
        write_le_u32(&mut bytes, off, 42);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::BadLeafState { state: 42, .. } => {}
            other => panic!("expected BadLeafState, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_state_high_bits_set() {
        let mut store = NodeStore::new();
        let leaf = store.leaf(Cell::pack(2, 0).raw());
        let mut bytes = save_to_vec(&store, leaf, 0);
        // Set high 16 bits of state.
        let off = record_offset(leaf.0 as usize) + 4;
        let cur = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        write_le_u32(&mut bytes, off, cur | 0xDEAD_0000);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::BadStatePayload { state, .. } => {
                assert_eq!(state >> 16, 0xDEAD);
            }
            other => panic!("expected BadStatePayload, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_interior_state_nonzero() {
        let mut bytes = minimal_file();
        // Interior at NodeId(1) — overwrite its state from 0 to 1.
        let off = record_offset(1) + 4;
        write_le_u32(&mut bytes, off, 1);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::BadStatePayload { at: 1, state: 1 } => {}
            other => panic!("expected BadStatePayload, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_interior_level_zero() {
        let mut bytes = minimal_file();
        // Interior at NodeId(1) — overwrite its level from 1 to 0.
        let off = record_offset(1);
        write_le_u32(&mut bytes, off, 0);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::InvalidLevel { at: 1, level: 0 } => {}
            other => panic!("expected InvalidLevel, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_truncated_file() {
        let bytes = minimal_file();
        // Drop the last record entirely — header still says N nodes.
        let truncated = &bytes[..bytes.len() - NODE_BYTES];
        match expect_err(load_from_slice(truncated)) {
            DiskError::SizeMismatch { .. } => {}
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_oversized_file() {
        let mut bytes = minimal_file();
        // Append junk after the last record. Header says N nodes; there
        // are now > N records' worth of bytes.
        bytes.extend_from_slice(&[0u8; 8]);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::SizeMismatch { .. } => {}
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_population_mismatch() {
        let mut bytes = minimal_file();
        // Empty interior at level 1: population is 0. Corrupt to 1 to
        // force a BadPopulation error.
        let off = record_offset(1) + 8;
        write_le_u64(&mut bytes, off, 1);
        match expect_err(load_from_slice(&bytes)) {
            DiskError::BadPopulation {
                at: 1,
                expected: 0,
                found: 1,
            } => {}
            other => panic!("expected BadPopulation, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_root_level_below_node() {
        // Build a store whose root is an interior at level=3, then save
        // with root_level=2. Expect InvalidRootLevel on load.
        let mut store = NodeStore::new();
        let root = store.empty(3);
        let mut bytes = Vec::new();
        save(&store, root, 3, &mut bytes).expect("save");
        // Patch root_level field in header to 2.
        bytes[28..32].copy_from_slice(&2u32.to_le_bytes());
        match expect_err(load_from_slice(&bytes)) {
            DiskError::InvalidRootLevel {
                node_level: 3,
                root_level: 2,
            } => {}
            other => panic!("expected InvalidRootLevel, got {other:?}"),
        }
    }

    #[test]
    fn node_validation_root_id_oob() {
        let store = NodeStore::new();
        let mut bytes = save_to_vec(&store, NodeId::EMPTY, 0);
        // Header says node_count=1, root_id=0. Plant root_id=99.
        bytes[24..28].copy_from_slice(&99u32.to_le_bytes());
        match expect_err(load_from_slice(&bytes)) {
            DiskError::InvalidRootId {
                root_id: 99,
                node_count: 1,
            } => {}
            other => panic!("expected InvalidRootId, got {other:?}"),
        }
    }
}
