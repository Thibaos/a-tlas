//! Micro-chunk snapshots and the minimal emitter (renderer-impl tickets
//! 02/03).
//!
//! The input contract (ADR 0004): the world hands the renderer Micro-chunk
//! snapshots — {global coords, 64-byte Occupancy mask, u8 material indices} —
//! one message for create, update, and removal (an emptied Micro-chunk
//! re-snapshots with a zero mask). Ticket 03's `crate::region::input`
//! implements the contract (enqueue-only `submit_microchunk` / `submit_batch`,
//! worker drain into per-Region mirrors); this module keeps the snapshot
//! shape (the contract) and the minimal emitter, which the world side uses
//! to voice its initial state as one `submit_batch` (the harness and tests
//! do exactly that).

use std::collections::HashMap;

use glam::IVec3;

use crate::world::chunk::Chunks;

/// The render unit's edge length in voxels (8^3 = 512 voxels).
pub const MICRO_CHUNK_EDGE: i32 = 8;

/// One Micro-chunk's occupancy: global coords (origin, a multiple of 8), the
/// 512-bit Occupancy mask, and the u8 material indices of the occupied
/// voxels.
///
/// The mask's bit order is `idx = x + 8*y + 64*z` for the cell (x, y, z)
/// within the Micro-chunk; the materials array follows the same increasing
/// bit order and has exactly `popcount(mask)` entries. The bit order is the
/// contract with the DDA in shaders/region/intersect.rint and the packer in
/// `crate::region::pack` — no sentinel material exists (palette index 0 is a
/// real color); the mask defines which voxels exist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicroChunkSnapshot {
    /// Global coords of the Micro-chunk's min corner (a multiple of 8).
    pub global_coords: IVec3,
    /// The 512-bit Occupancy mask: bit `idx` at byte `idx / 8`, bit `idx % 8`
    /// (little-endian bit order).
    pub mask: [u8; 64],
    /// u8 material indices of the occupied voxels, in increasing bit order.
    pub materials: Vec<u8>,
}

impl MicroChunkSnapshot {
    /// The number of occupied voxels (the mask's popcount).
    pub fn occupied_count(&self) -> usize {
        self.mask
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum()
    }
}

/// Emits one snapshot per non-empty Micro-chunk over the whole world, from
/// the world's own voxel storage (the world side of the input contract).
///
/// This is the minimal emitter: deterministic (snapshots sorted by global
/// coords), covering every occupied voxel — interior included, since the
/// Occupancy mask (not surface-ness) defines existence.
pub fn emit_snapshots(world: &Chunks) -> Vec<MicroChunkSnapshot> {
    // Voxel → (micro-chunk origin, (bit index, material)).
    let mut per_microchunk: HashMap<IVec3, Vec<(u32, u8)>> = HashMap::new();

    for (global, voxel) in world.iter_voxels() {
        let origin = global.div_euclid(IVec3::splat(MICRO_CHUNK_EDGE)) * MICRO_CHUNK_EDGE;
        let local = global - origin;
        debug_assert!(local.cmpge(IVec3::ZERO).all());
        debug_assert!(local.cmplt(IVec3::splat(MICRO_CHUNK_EDGE)).all());

        let idx = (local.x + 8 * local.y + 64 * local.z) as u32;
        per_microchunk
            .entry(origin)
            .or_default()
            .push((idx, voxel.material_index() as u8));
    }

    let mut snapshots: Vec<MicroChunkSnapshot> = per_microchunk
        .into_iter()
        .map(|(global_coords, mut cells)| {
            cells.sort_unstable_by_key(|&(idx, _)| idx);

            let mut mask = [0u8; 64];
            let mut materials = Vec::with_capacity(cells.len());
            for (idx, material) in cells {
                mask[(idx / 8) as usize] |= 1 << (idx % 8);
                materials.push(material);
            }

            debug_assert_eq!(
                materials.len(),
                mask.iter().map(|b| b.count_ones() as usize).sum()
            );

            MicroChunkSnapshot {
                global_coords,
                mask,
                materials,
            }
        })
        .collect();

    snapshots.sort_unstable_by_key(|s| s.global_coords.to_array());
    snapshots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::chunk::Chunks;

    /// Bit idx = x + 8*y + 64*z must round-trip through the mask bytes.
    #[test]
    fn mask_bit_convention() {
        let mut mask = [0u8; 64];
        for idx in [0u32, 1, 7, 8, 63, 64, 511] {
            mask[(idx / 8) as usize] |= 1 << (idx % 8);
        }
        for idx in 0..512u32 {
            let byte = mask[(idx / 8) as usize];
            assert_eq!(
                (byte >> (idx % 8)) & 1 != 0,
                matches!(idx, 0 | 1 | 7 | 8 | 63 | 64 | 511)
            );
        }
    }

    /// The emitter groups voxels by Micro-chunk and reports every occupied
    /// voxel exactly once (interior voxels included).
    #[test]
    fn emitter_covers_all_voxels() {
        let mut world = Chunks::default();
        // Voxels straddling a Micro-chunk boundary, one interior to a solid
        // block (no exposed face).
        for x in 0..10 {
            for y in 0..3 {
                for z in 0..3 {
                    world.insert_voxel_at(IVec3::new(x, y, z), 3);
                }
            }
        }
        // A negative-coordinate voxel (floor-division Micro-chunk origin).
        world.insert_voxel_at(IVec3::new(-1, -1, -1), 5);

        let snapshots = emit_snapshots(&world);
        let total: usize = snapshots.iter().map(|s| s.occupied_count()).sum();
        assert_eq!(total, world.voxel_count());

        // Snapshot origins are multiples of 8; the -1 voxel lands in the
        // Micro-chunk rooted at -8 at bit idx = 7 + 8*7 + 64*7 = 511.
        assert!(snapshots.iter().any(|s| {
            s.global_coords == IVec3::new(-8, -8, -8) && s.mask[63] & 0b1000_0000 != 0
        }));
    }

    /// Snapshot materials must be in increasing bit order (the packer and the
    /// DDA rely on the rank = popcount below the bit).
    #[test]
    fn materials_in_bit_order() {
        let mut world = Chunks::default();
        world.insert_voxel_at(IVec3::new(0, 0, 0), 1); // idx 0
        world.insert_voxel_at(IVec3::new(7, 0, 0), 2); // idx 7
        world.insert_voxel_at(IVec3::new(0, 1, 0), 3); // idx 8
        world.insert_voxel_at(IVec3::new(0, 0, 1), 4); // idx 64

        let snapshots = emit_snapshots(&world);
        assert_eq!(snapshots.len(), 1);
        let snapshot = &snapshots[0];
        assert_eq!(snapshot.materials, vec![1, 2, 3, 4]);
        assert_eq!(snapshot.occupied_count(), 4);
    }
}
