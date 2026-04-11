/// Packed cell state: 10-bit material ID + 6-bit metadata in a single u16.
///
/// Layout (MSB → LSB):
///   bits 15..6  material_id  (10 bits, 0..=1023)
///   bits  5..0  metadata     ( 6 bits, 0..=63)
///
/// Material 0 is reserved as "empty" — the whole cell word is 0 iff the cell
/// is empty, which preserves the hash-cons invariant that `NodeId::EMPTY`
/// (a `Leaf(0)`) represents pure void. Attempting to attach metadata to
/// material 0 is a programming error (there is no "empty with flavor");
/// `Cell::pack` and `Cell::from_raw` panic — in both debug **and release** —
/// because a violation silently corrupts hash-consing.
///
/// Hashlife treats `CellState` as an opaque bit pattern — it memoizes by
/// pattern equality and never inspects the material/metadata split. The
/// simulation kernel and renderer unpack as needed.
///
/// **Rust↔WGSL drift guard:** `METADATA_BITS = 6` is duplicated in
/// `src/render/raycast.wgsl` and `src/render/svdag_raycast.wgsl` as the
/// hardcoded `packed >> 6u` material decode. If `METADATA_BITS` changes
/// here, both shaders must be updated to match. A unit test in
/// `src/render/mod.rs` (or equivalent) should pin these together.
pub type CellState = u16;

/// Typed view over a `CellState` word. `Copy` and `repr(transparent)` so it
/// can be stored interchangeably with a raw `u16`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Cell(pub CellState);

impl Cell {
    /// Canonical empty cell (material 0, metadata 0).
    pub const EMPTY: Cell = Cell(0);

    /// Number of bits reserved for the metadata field.
    pub const METADATA_BITS: u32 = 6;

    /// Maximum representable material ID (inclusive).
    pub const MAX_MATERIAL: u16 = (1 << (16 - Self::METADATA_BITS)) - 1; // 1023

    /// Maximum representable metadata value (inclusive).
    pub const MAX_METADATA: u16 = (1 << Self::METADATA_BITS) - 1; // 63

    const METADATA_MASK: u16 = Self::MAX_METADATA;

    /// Pack a (material, metadata) pair. Panics (in both debug AND release)
    /// if either field overflows, or if metadata is nonzero on material 0
    /// (ambiguous "empty with flavor" — would silently alias `Cell::EMPTY`
    /// and corrupt hash-consing). `assert!` is deliberate: foundational
    /// invariant, the cost of a branch beats shipping a corrupt tree.
    #[inline]
    pub const fn pack(material: u16, metadata: u16) -> Cell {
        assert!(
            material <= Self::MAX_MATERIAL,
            "material id overflows 10 bits"
        );
        assert!(metadata <= Self::MAX_METADATA, "metadata overflows 6 bits");
        assert!(
            material != 0 || metadata == 0,
            "material 0 is reserved for empty; metadata must also be 0",
        );
        Cell((material << Self::METADATA_BITS) | metadata)
    }

    /// Construct a cell from a raw `CellState` word (e.g. when loading from
    /// the octree store or a GPU buffer). Validates the "material 0 ⇒
    /// metadata 0" invariant and panics in both debug and release if it is
    /// violated. Raw u16 values ≤ `MAX_METADATA` that are nonzero are the
    /// forbidden range — any such value decodes to `material=0` with
    /// nonzero metadata.
    ///
    /// The overflow fields (material / metadata bit-width) cannot be
    /// violated by a `u16` input, so we only check the empty-invariant.
    #[inline]
    pub const fn from_raw(raw: CellState) -> Cell {
        // Any raw value in 1..=MAX_METADATA decodes to material=0 with
        // nonzero metadata, which is the forbidden "empty with flavor".
        assert!(
            raw == 0 || raw > Self::MAX_METADATA,
            "raw cell in forbidden range: material 0 with nonzero metadata",
        );
        Cell(raw)
    }

    /// Construct a cell from a raw `CellState` word without validating the
    /// "material 0 ⇒ metadata 0" invariant. Only use this on bit patterns
    /// that are already known to be well-formed (e.g. bits that came out of
    /// a previously validated `Cell`). Violating the invariant silently
    /// corrupts hash-consing — there is no safety net downstream.
    #[inline]
    pub const fn from_raw_unchecked(raw: CellState) -> Cell {
        Cell(raw)
    }

    /// Raw `CellState` word. This is what Hashlife memoizes on.
    #[inline]
    pub const fn raw(self) -> CellState {
        self.0
    }

    /// Material ID (10 bits).
    #[inline]
    pub const fn material(self) -> u16 {
        self.0 >> Self::METADATA_BITS
    }

    /// Metadata (6 bits).
    #[inline]
    pub const fn metadata(self) -> u16 {
        self.0 & Self::METADATA_MASK
    }

    /// `true` iff this cell is the canonical empty cell (material 0, metadata 0).
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Produce a new cell with the material replaced, metadata preserved.
    /// Panics in debug if `material` overflows or if `material==0` while
    /// metadata is nonzero (the resulting cell would alias empty-with-flavor).
    #[inline]
    pub const fn with_material(self, material: u16) -> Cell {
        Cell::pack(material, self.metadata())
    }

    /// Produce a new cell with the metadata replaced, material preserved.
    #[inline]
    pub const fn with_metadata(self, metadata: u16) -> Cell {
        Cell::pack(self.material(), metadata)
    }
}

// Intentionally no `impl From<u16> for Cell` — a silent `u16 -> Cell`
// conversion bypasses the "material 0 ⇒ metadata 0" invariant. Callers
// must go through `Cell::pack` (for logical values) or `Cell::from_raw`
// (for storage round-trips, validated) or `Cell::from_raw_unchecked`
// (documented escape hatch).

impl From<Cell> for u16 {
    #[inline]
    fn from(cell: Cell) -> u16 {
        cell.0
    }
}

/// Index into the canonical node store. u32 to keep nodes small.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const EMPTY: NodeId = NodeId(0); // reserved: canonical empty node
}

/// A canonical octree node.
///
/// Level 0: a single cell (1x1x1).
/// Level n: a 2^n × 2^n × 2^n cube, with 8 children at level n-1.
///
/// Children are indexed by octant:
///   0 = (−x, −y, −z)  aka (0,0,0) corner
///   1 = (+x, −y, −z)  aka (1,0,0)
///   2 = (−x, +y, −z)  aka (0,1,0)
///   3 = (+x, +y, −z)  aka (1,1,0)
///   4 = (−x, −y, +z)  aka (0,0,1)
///   5 = (+x, −y, +z)  aka (1,0,1)
///   6 = (−x, +y, +z)  aka (0,1,0)
///   7 = (+x, +y, +z)  aka (1,1,1)
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    Leaf(CellState),
    Interior {
        level: u32,
        children: [NodeId; 8],
        /// Population count (number of non-empty cells). Used for quick emptiness checks.
        population: u64,
    },
}

impl Node {
    pub fn level(&self) -> u32 {
        match self {
            Node::Leaf(_) => 0,
            Node::Interior { level, .. } => *level,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Node::Leaf(s) => *s == 0,
            Node::Interior { population, .. } => *population == 0,
        }
    }

    /// Side length of this node in cells.
    pub fn side_len(&self) -> u64 {
        1u64 << self.level()
    }
}

/// Octant index helper: given (x, y, z) each 0 or 1, return octant index.
#[inline]
pub fn octant_index(x: u32, y: u32, z: u32) -> usize {
    (x + y * 2 + z * 4) as usize
}

/// Inverse: octant index to (x, y, z) each 0 or 1.
#[inline]
pub fn octant_coords(idx: usize) -> (u32, u32, u32) {
    (idx as u32 & 1, (idx as u32 >> 1) & 1, (idx as u32 >> 2) & 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cell_roundtrip() {
        let c = Cell::EMPTY;
        assert!(c.is_empty());
        assert_eq!(c.material(), 0);
        assert_eq!(c.metadata(), 0);
        assert_eq!(c.raw(), 0);
    }

    #[test]
    fn pack_unpack_basic() {
        let c = Cell::pack(5, 3);
        assert_eq!(c.material(), 5);
        assert_eq!(c.metadata(), 3);
        assert!(!c.is_empty());
    }

    #[test]
    fn pack_unpack_boundary() {
        let c = Cell::pack(Cell::MAX_MATERIAL, Cell::MAX_METADATA);
        assert_eq!(c.material(), Cell::MAX_MATERIAL);
        assert_eq!(c.metadata(), Cell::MAX_METADATA);
        assert_eq!(c.raw(), 0xFFFF);
    }

    #[test]
    fn raw_value_layout() {
        // Material 1, metadata 0 -> bits 15..6 = 0000000001, bits 5..0 = 000000
        // = 0b0000_0000_0100_0000 = 0x0040 = 64
        assert_eq!(Cell::pack(1, 0).raw(), 0x0040);
        // Material 1, metadata 1 -> 0x0041
        assert_eq!(Cell::pack(1, 1).raw(), 0x0041);
        // Material 2, metadata 0 -> 0x0080
        assert_eq!(Cell::pack(2, 0).raw(), 0x0080);
    }

    #[test]
    fn with_material_preserves_metadata() {
        let c = Cell::pack(7, 42);
        let d = c.with_material(9);
        assert_eq!(d.material(), 9);
        assert_eq!(d.metadata(), 42);
    }

    #[test]
    fn with_metadata_preserves_material() {
        let c = Cell::pack(7, 42);
        let d = c.with_metadata(11);
        assert_eq!(d.material(), 7);
        assert_eq!(d.metadata(), 11);
    }

    #[test]
    fn from_raw_roundtrip() {
        // Only valid raws: 0 (empty) and anything where material > 0
        // (i.e. raw > MAX_METADATA). Raws in 1..=63 are forbidden.
        for raw in [0u16, 0x0040, 0x0041, 0x1234, 0xFFFF] {
            assert_eq!(Cell::from_raw(raw).raw(), raw);
        }
    }

    #[test]
    #[should_panic(expected = "forbidden range")]
    fn from_raw_rejects_empty_with_flavor() {
        let _ = Cell::from_raw(1);
    }

    #[test]
    #[should_panic(expected = "forbidden range")]
    fn from_raw_rejects_max_metadata_alone() {
        // raw = MAX_METADATA (63) = material 0, metadata 63 → forbidden.
        let _ = Cell::from_raw(Cell::MAX_METADATA);
    }

    #[test]
    fn from_raw_unchecked_skips_validation() {
        // Escape hatch does not panic even on forbidden bit patterns.
        let bogus = Cell::from_raw_unchecked(1);
        assert_eq!(bogus.raw(), 1);
    }

    #[test]
    #[should_panic(expected = "material id overflows")]
    fn pack_material_overflow_panics() {
        // Cell::MAX_MATERIAL is 1023; 1024 overflows.
        let _ = Cell::pack(1024, 0);
    }

    #[test]
    #[should_panic(expected = "metadata overflows")]
    fn pack_metadata_overflow_panics() {
        let _ = Cell::pack(1, 64);
    }

    #[test]
    #[should_panic(expected = "material 0 is reserved")]
    fn empty_with_flavor_panics() {
        let _ = Cell::pack(0, 1);
    }
}
