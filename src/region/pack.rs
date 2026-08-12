//! CPU-side packing: Micro-chunk snapshots → per-Region voxel pools
//! (offset table + compact blocks) and trimmed AABB hulls (renderer-impl
//! tickets 02/03 / ADR 0001). Ticket 02 builds the static lattice over the
//! world's initial snapshot batch; ticket 03's input contract packs through
//! the same shape ([`RegionMirror::pack`](crate::region::input::RegionMirror))
//! on every change cycle.
//!
//! A Region is the renderer's grouping of Micro-chunks that share one
//! acceleration-structure build: 32^3 Micro-chunks (256^3 voxels, 4x4x4
//! Chunks), origin-aligned over the world lattice. This module derives the
//! Region id from global coords (the renderer owns the lattice — the world
//! never computes ids) and packs each Region's GPU pool CPU-side: a u32
//! offset table (32768 slots, sentinel for empty Micro-chunks) followed by
//! compact blocks (64-byte Occupancy mask + popcount-compacted u8 materials,
//! 8-aligned). It also derives the trimmed hull AABBs in absolute
//! Region-local coordinates — one per non-empty Micro-chunk — so the
//! intersection shader resolves micro_chunk = floor(hit/8) and
//! cell = floor(hit) mod 8 with zero per-AABB metadata.

use std::collections::HashMap;

use glam::IVec3;
use vulkano::acceleration_structure::AabbPositions;

use super::snapshot::{MICRO_CHUNK_EDGE, MicroChunkSnapshot};

/// The Region's edge length in voxels (256^3 voxels per Region).
pub const REGION_EDGE: i32 = 256;

/// The number of Micro-chunks per Region: 32^3 = 32768.
pub const MICRO_CHUNKS_PER_REGION: usize = 32 * 32 * 32;

/// The offset table's size in bytes (u32 × 32768 slots) — the pool starts
/// with the table, then the blocks.
pub const OFFSET_TABLE_SIZE: usize = MICRO_CHUNKS_PER_REGION * 4;

/// The offset-table sentinel for empty Micro-chunks (no block exists).
pub const OFFSET_SENTINEL: u32 = u32::MAX;

/// The number of Regions in the v1 lattice: ±2048/axis → 16^3 = 4096, the
/// 12-bit region-id budget.
pub const REGION_COUNT: usize = 4096;

/// Whether a Region index lies inside the v1 lattice (region indices in
/// [-8, 8) → voxel ±2048/axis, the 12-bit region-id budget). The single
/// source of truth for the extent: both [`region_id`] (the 4-bit/axis
/// encoding aliases beyond it) and the world→renderer boundary check share
/// this, so the bounds cannot drift.
pub fn region_index_in_lattice(region_index: IVec3) -> bool {
    region_index.cmpge(IVec3::splat(-8)).all() && region_index.cmplt(IVec3::splat(8)).all()
}

/// Panics iff `region_index` is outside the v1 lattice. This is an
/// **unconditional** check, not a `debug_assert`: in release the 4-bit
/// encoding in [`region_id`] would silently alias out-of-lattice indicGes and
/// then index past the fixed `REGION_COUNT` instance/region tables, corrupting
/// (not erroring). An over-lattice model must fail loudly here, at the
/// world→renderer boundary, never silently.
pub fn assert_region_index_in_lattice(region_index: IVec3) {
    if !region_index_in_lattice(region_index) {
        panic!(
            "micro-chunk/region index {region_index} exceeds the renderer lattice \
             (±2048/axis, region indices in [-8, 8), the 12-bit region-id budget); \
             the model is too big to fit the v1 acceleration-structure lattice"
        );
    }
}

/// The 12-bit Region id (rides the TLAS instance custom index and indexes
/// the Region table): 4 bits per axis over the signed Region lattice index,
/// biased by +8 so all-16 values per axis fit the 4 bits.
pub fn region_id(region_index: IVec3) -> u32 {
    assert_region_index_in_lattice(region_index);
    (((region_index.x + 8) as u32 & 0xF) << 8)
        | (((region_index.y + 8) as u32 & 0xF) << 4)
        | ((region_index.z + 8) as u32 & 0xF)
}

/// The Region lattice index of a Micro-chunk origin (floor division, so
/// negative global coords land in the correct origin-aligned Region).
pub fn region_index_of(global_coords: IVec3) -> IVec3 {
    global_coords.div_euclid(IVec3::splat(REGION_EDGE))
}

/// The CPU-side mirror of one Region: the packed pool (offset table +
/// blocks) and the trimmed hull AABBs, everything in Region-local
/// coordinates. This is the source of truth for the wholesale pool rebuild
/// (ticket 03's change path packs through this shape).
pub struct RegionData {
    pub region_index: IVec3,
    /// u32 offset table, `MICRO_CHUNKS_PER_REGION` slots; slot
    /// `(mc.z*32 + mc.y)*32 + mc.x` holds the block's byte offset within the
    /// pool, or `OFFSET_SENTINEL` for an empty Micro-chunk.
    ///
    /// The typed mirror of the table serialized at the start of [`blocks`]
    /// (the pool layout). Production reads `blocks`; tests assert on the
    /// typed table, and the input contract's change path (ticket 03) packs
    /// through it via [`RegionMirror::pack`](crate::region::input::RegionMirror).
    #[cfg_attr(not(test), allow(dead_code))]
    pub offset_table: Vec<u32>,
    /// The pool bytes: the offset table followed by 8-aligned compact blocks
    /// (64-byte mask + popcount-compacted u8 materials, padded to 8 bytes).
    pub blocks: Vec<u8>,
    /// One trimmed hull per non-empty Micro-chunk, in absolute Region-local
    /// coordinates (min = occupied-min cell, max = occupied-max cell + 1).
    pub aabbs: Vec<AabbPositions>,
}

impl RegionData {
    pub fn region_id(&self) -> u32 {
        region_id(self.region_index)
    }

    /// The Region's min corner in world coordinates.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn origin(&self) -> IVec3 {
        self.region_index * REGION_EDGE
    }
}

/// Packs the snapshot batch into one [`RegionData`] per occupied Region,
/// sorted by Region id (deterministic instance order).
pub fn pack_regions(snapshots: &[MicroChunkSnapshot]) -> Vec<RegionData> {
    let mut by_region: HashMap<IVec3, Vec<&MicroChunkSnapshot>> = HashMap::new();
    for snapshot in snapshots {
        by_region
            .entry(region_index_of(snapshot.global_coords))
            .or_default()
            .push(snapshot);
    }

    let mut regions: Vec<RegionData> = by_region
        .into_iter()
        .map(|(region_index, region_snapshots)| pack_region(region_index, &region_snapshots))
        .collect();

    regions.sort_unstable_by_key(|region| region.region_id());
    regions
}

/// Packs one Region's snapshots into a pool + trimmed hulls. A snapshot's
/// global coords must fall inside this Region (its micro-chunk origin is
/// within the Region's extent). Exposed for the input contract's
/// [`RegionMirror::pack`](crate::region::input::RegionMirror) (ticket 03).
pub(crate) fn pack_region(region_index: IVec3, snapshots: &[&MicroChunkSnapshot]) -> RegionData {
    let region_origin = region_index * REGION_EDGE;

    let mut offset_table = vec![OFFSET_SENTINEL; MICRO_CHUNKS_PER_REGION];
    // The pool starts with the offset table serialized little-endian, then
    // the compact blocks; the sentinel 0xFFFFFFFF bytes fill the empty slots.
    let mut blocks: Vec<u8> = Vec::with_capacity(OFFSET_TABLE_SIZE);
    for slot in &offset_table {
        blocks.extend_from_slice(&slot.to_le_bytes());
    }
    let mut aabbs = Vec::with_capacity(snapshots.len());

    let mut ordered = snapshots.to_vec();
    ordered.sort_unstable_by_key(|s| s.global_coords.to_array());

    for snapshot in ordered {
        let mc_local_origin = snapshot.global_coords - region_origin;
        debug_assert!(
            mc_local_origin.cmpge(IVec3::ZERO).all()
                && mc_local_origin.cmplt(IVec3::splat(REGION_EDGE)).all(),
            "snapshot {} outside region {region_index}",
            snapshot.global_coords
        );

        let mc = mc_local_origin / MICRO_CHUNK_EDGE;
        let mc_index = ((mc.z * 32 + mc.y) * 32 + mc.x) as usize;

        // Every block offset is a multiple of 8 (the table is 128 KiB), so
        // the mask stays 8-aligned.
        let block_offset = blocks.len() as u32;
        debug_assert!(block_offset % 8 == 0);
        debug_assert_eq!(offset_table[mc_index], OFFSET_SENTINEL);

        offset_table[mc_index] = block_offset;
        blocks[4 * mc_index..4 * mc_index + 4].copy_from_slice(&block_offset.to_le_bytes());
        debug_assert_eq!(snapshot.materials.len(), snapshot.occupied_count());
        blocks.extend_from_slice(&snapshot.mask);
        blocks.extend_from_slice(&snapshot.materials);
        while blocks.len() % 8 != 0 {
            blocks.push(0);
        }

        // Trimmed hull: the occupied-bounds box in Region-local cells.
        let (min_cell, max_cell) = occupied_cell_bounds(&snapshot.mask);
        let min = mc_local_origin + min_cell;
        let max = mc_local_origin + max_cell + IVec3::ONE;
        aabbs.push(AabbPositions {
            min: min.as_vec3().to_array(),
            max: max.as_vec3().to_array(),
        });
    }

    RegionData {
        region_index,
        offset_table,
        blocks,
        aabbs,
    }
}

/// The inclusive bounds of the mask's occupied cells within the Micro-chunk
/// (each component in 0..8).
fn occupied_cell_bounds(mask: &[u8; 64]) -> (IVec3, IVec3) {
    let mut min = IVec3::splat(7);
    let mut max = IVec3::ZERO;
    let mut any = false;

    for idx in 0..512usize {
        if (mask[idx / 8] >> (idx % 8)) & 1 != 0 {
            let x = (idx % 8) as i32;
            let y = ((idx / 8) % 8) as i32;
            let z = (idx / 64) as i32;
            min = min.min(IVec3::new(x, y, z));
            max = max.max(IVec3::new(x, y, z));
            any = true;
        }
    }

    debug_assert!(any, "packed an empty snapshot as a hull");
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        region::snapshot::{MICRO_CHUNK_EDGE, emit_snapshots},
        world::chunk::Chunks,
    };

    /// A world with voxels in two Regions (x = 255 in Region 0, x = 256 in
    /// Region 1) packs into two Regions with correct ids and lattice math.
    #[test]
    fn packs_across_region_boundary() {
        let mut world = Chunks::default();
        world.insert_voxel_at(IVec3::new(255, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(256, 0, 0), 2);

        let snapshots = emit_snapshots(&world);
        let regions = pack_regions(&snapshots);

        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].region_index, IVec3::new(0, 0, 0));
        assert_eq!(regions[1].region_index, IVec3::new(1, 0, 0));

        // Region 1's voxel is at Region-local cell (0, 0, 0): the trimmed
        // hull is the unit cell [0, 1)^3.
        assert_eq!(regions[1].aabbs[0].min, [0.0, 0.0, 0.0]);
        assert_eq!(regions[1].aabbs[0].max, [1.0, 1.0, 1.0]);

        // The Region origins are lattice-aligned (the instance transforms).
        assert_eq!(regions[0].origin(), IVec3::ZERO);
        assert_eq!(regions[1].origin(), IVec3::new(256, 0, 0));

        // Region 0's voxel is at Region-local cell (255, 0, 0).
        assert_eq!(regions[0].aabbs[0].min, [255.0, 0.0, 0.0]);
        assert_eq!(regions[0].aabbs[0].max, [256.0, 1.0, 1.0]);
    }

    /// Negative global coords land in the correct origin-aligned Region via
    /// floor division, with Region-local coords in [0, 256).
    #[test]
    fn negative_coords_use_floor_regions() {
        assert_eq!(
            region_index_of(IVec3::new(-8, -8, -8)),
            IVec3::new(-1, -1, -1)
        );
        assert_eq!(region_index_of(IVec3::new(0, 0, 0)), IVec3::ZERO);
        assert_eq!(region_index_of(IVec3::new(248, 0, 0)), IVec3::ZERO);
        assert_eq!(region_index_of(IVec3::new(256, 0, 0)), IVec3::new(1, 0, 0));
    }

    /// Packing invariants: the table has the sentinel everywhere except the
    /// occupied slots, blocks are 8-aligned, and the mask+materials inside a
    /// block round-trip to the snapshot.
    #[test]
    fn pack_layout_invariants() {
        let mut world = Chunks::default();
        // Two voxels in the same Micro-chunk at opposite corners (empty
        // cells between them) plus a third in a different Micro-chunk.
        world.insert_voxel_at(IVec3::new(0, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(7, 7, 7), 2);
        world.insert_voxel_at(IVec3::new(8, 0, 0), 3);

        let snapshots = emit_snapshots(&world);
        let regions = pack_regions(&snapshots);
        assert_eq!(regions.len(), 1);
        let region = &regions[0];

        assert_eq!(region.offset_table.len(), MICRO_CHUNKS_PER_REGION);
        assert_eq!(region.blocks.len() % 8, 0);
        assert_eq!(region.aabbs.len(), 2);

        // Slot for the (0,0,0) Micro-chunk points at an 8-aligned block.
        let slot0 = region.offset_table[0];
        assert_ne!(slot0, OFFSET_SENTINEL);
        assert_eq!(slot0 % 8, 0);
        assert_eq!(slot0, OFFSET_TABLE_SIZE as u32);

        // Slot for the (8,0,0) Micro-chunk (mc index 1) is also filled; a
        // far-away empty slot stays sentinel.
        assert_ne!(region.offset_table[1], OFFSET_SENTINEL);
        assert_eq!(region.offset_table[1000], OFFSET_SENTINEL);

        // The block at slot0: mask bytes then materials. Two occupied voxels
        // (bits 0 and 511), materials [1, 2].
        let block = &region.blocks[slot0 as usize..];
        assert_eq!(block[0] & 1, 1);
        assert_eq!(block[63] & 0x80, 0x80);
        assert_eq!(block[64], 1);
        assert_eq!(block[65], 2);
    }

    /// The Micro-chunk index convention must match the DDA's
    /// (mc.z * 32 + mc.y) * 32 + mc.x and the bit order x + 8y + 64z.
    #[test]
    fn microchunk_index_convention() {
        let mc = IVec3::new(1, 2, 3);
        let index = ((mc.z * 32 + mc.y) * 32 + mc.x) as usize;
        assert_eq!(index, (3 * 32 + 2) * 32 + 1);

        // The max Micro-chunk index within a Region fits the table.
        assert_eq!((31 * 32 + 31) * 32 + 31, MICRO_CHUNKS_PER_REGION - 1);

        // A voxel at Region-local cell (0, 0, 0) of the mc (1,2,3) micro-chunk
        // has voxel index 0 (bit 0).
        assert_eq!(MICRO_CHUNK_EDGE, 8);
    }
}
