use anyhow::Context;
use std::collections::HashMap;

use glam::{IVec3, UVec3};

use crate::core::world::{
    World,
    grid::{MICRO_CHUNK_LENGTH, grid_origin},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicroChunkSnapshot {
    pub global_coords: IVec3,
    pub mask: [u8; 64],
    pub materials: Vec<u8>,
}

impl MicroChunkSnapshot {
    #[allow(clippy::as_conversions)] // count_ones is bounded by the mask width
    pub fn occupied_count(&self) -> usize {
        self.mask
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum()
    }
}

pub fn emit_snapshots(world: &World) -> anyhow::Result<Vec<MicroChunkSnapshot>> {
    let mut per_microchunk: HashMap<IVec3, Vec<(u32, u8)>> = HashMap::new();

    for (global, voxel) in world.iter_voxels() {
        let origin = grid_origin(global, MICRO_CHUNK_LENGTH);
        let local = global
            .checked_sub(origin)
            .context("voxel below its micro chunk origin")?;

        debug_assert!(local.cmpge(IVec3::ZERO).all());
        debug_assert!(
            local
                .cmplt(UVec3::splat(MICRO_CHUNK_LENGTH).as_ivec3())
                .all()
        );

        let idx = u32::try_from(
            local
                .x
                .strict_add(local.y.strict_mul(8))
                .strict_add(local.z.strict_mul(64)),
        )?;

        per_microchunk
            .entry(origin)
            .or_default()
            .push((idx, u8::try_from(*voxel)?));
    }

    let mut snapshots: Vec<MicroChunkSnapshot> = per_microchunk
        .into_iter()
        .map(|(global_coords, mut cells)| -> anyhow::Result<MicroChunkSnapshot> {
            cells.sort_unstable_by_key(|&(idx, _)| idx);

            let mut mask = [0u8; 64];
            let mut materials = Vec::with_capacity(cells.len());

            for (idx, material) in cells {
                let slot = mask
                    .get_mut(usize::try_from(idx / 8)?)
                    .context(format!("mask byte for cell {idx} out of range"))?;
                *slot |= 1 << (idx % 8);
                materials.push(material);
            }

            let snapshot = MicroChunkSnapshot {
                global_coords,
                mask,
                materials,
            };

            debug_assert_eq!(snapshot.materials.len(), snapshot.occupied_count());

            Ok(snapshot)
        })
        .collect::<anyhow::Result<_>>()?;

    snapshots.sort_unstable_by_key(|s| s.global_coords.to_array());

    Ok(snapshots)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn emitter_covers_all_voxels() {
        let mut world = World::default();
        for x in 0..10 {
            for y in 0..3 {
                for z in 0..3 {
                    world.insert_voxel_at(IVec3::new(x, y, z), 3);
                }
            }
        }

        world.insert_voxel_at(IVec3::new(-1, -1, -1), 5);

        let snapshots = emit_snapshots(&world).unwrap();
        let total: usize = snapshots.iter().map(|s| s.occupied_count()).sum();
        assert_eq!(total, world.voxel_count());

        assert!(snapshots.iter().any(|s| {
            s.global_coords == IVec3::new(-8, -8, -8) && s.mask[63] & 0b1000_0000 != 0
        }));
    }

    #[test]
    fn materials_in_bit_order() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(0, 0, 0), 1); // idx 0
        world.insert_voxel_at(IVec3::new(7, 0, 0), 2); // idx 7
        world.insert_voxel_at(IVec3::new(0, 1, 0), 3); // idx 8
        world.insert_voxel_at(IVec3::new(0, 0, 1), 4); // idx 64

        let snapshots = emit_snapshots(&world).unwrap();
        assert_eq!(snapshots.len(), 1);
        let snapshot = &snapshots[0];
        assert_eq!(snapshot.materials, vec![1, 2, 3, 4]);
        assert_eq!(snapshot.occupied_count(), 4);
    }
}
