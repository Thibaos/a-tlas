#[cfg(test)]
use std::collections::HashMap;

use anyhow::{Context, bail};
use glam::{IVec3, UVec3};
use vulkano::acceleration_structure::AabbPositions;

use crate::world::{
    grid::{MICRO_CHUNK_LENGTH, REGION_HALF_EXTENT, REGION_LENGTH, region_id},
    snapshot::MicroChunkSnapshot,
};

#[cfg(test)]
use crate::world::grid::region_index_of;

#[allow(clippy::as_conversions)]
pub const MC_PER_REGION_SIDE: usize = (REGION_LENGTH / MICRO_CHUNK_LENGTH) as usize;
pub const MICRO_CHUNKS_PER_REGION: usize =
    MC_PER_REGION_SIDE * MC_PER_REGION_SIDE * MC_PER_REGION_SIDE;
pub const OFFSET_TABLE_SIZE: usize = MICRO_CHUNKS_PER_REGION * 4;
pub const OFFSET_SENTINEL: u32 = u32::MAX;
#[allow(clippy::as_conversions)]
pub const REGION_COUNT: usize = (2 * REGION_HALF_EXTENT as usize).pow(3);

pub struct RegionData {
    pub region_index: IVec3,
    #[cfg(test)]
    pub offset_table: Vec<u32>,
    pub blocks: Vec<u8>,
    pub aabbs: Vec<AabbPositions>,
}

impl RegionData {
    pub fn region_id(&self) -> u32 {
        region_id(self.region_index)
    }

    #[cfg(test)]
    pub fn origin(&self) -> IVec3 {
        self.region_index * i32::try_from(REGION_LENGTH).unwrap()
    }
}

#[cfg(test)]
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
        .map(|(region_index, region_snapshots)| {
            pack_region(region_index, &region_snapshots).unwrap()
        })
        .collect();

    regions.sort_unstable_by_key(|region| region.region_id());
    regions
}

pub fn pack_region(
    region_index: IVec3,
    snapshots: &[&MicroChunkSnapshot],
) -> anyhow::Result<RegionData> {
    let region_origin = region_index.saturating_mul(UVec3::splat(REGION_LENGTH).as_ivec3());

    let mut offset_table = vec![OFFSET_SENTINEL; MICRO_CHUNKS_PER_REGION];
    let mut blocks: Vec<u8> = Vec::with_capacity(OFFSET_TABLE_SIZE);

    for slot in &offset_table {
        blocks.extend_from_slice(&slot.to_le_bytes());
    }

    let mut aabbs = Vec::with_capacity(snapshots.len());

    let mut ordered = snapshots.to_vec();
    ordered.sort_unstable_by_key(|s| s.global_coords.to_array());

    for snapshot in ordered {
        let mc_local_origin = snapshot
            .global_coords
            .checked_sub(region_origin)
            .with_context(|| {
                format!(
                    "snapshot {} outside region {region_index}",
                    snapshot.global_coords
                )
            })?;

        debug_assert!(
            mc_local_origin.cmpge(IVec3::ZERO).all()
                && mc_local_origin
                    .cmplt(UVec3::splat(REGION_LENGTH).as_ivec3())
                    .all(),
            "snapshot {} outside region {region_index}",
            snapshot.global_coords
        );

        let mc = mc_local_origin
            .checked_div(UVec3::splat(MICRO_CHUNK_LENGTH).as_ivec3())
            .context("MICRO_CHUNK_LENGTH is zero")?;

        let side = i32::try_from(MC_PER_REGION_SIDE)?;
        let mc_index = usize::try_from(
            (mc.z.saturating_mul(side).saturating_add(mc.y))
                .saturating_mul(side)
                .saturating_add(mc.x),
        )?;

        let block_offset = u32::try_from(blocks.len())?;
        debug_assert_eq!(block_offset % 8, 0);

        let offset = offset_table
            .get_mut(mc_index)
            .with_context(|| format!("offset table slot {mc_index} out of range"))?;
        debug_assert_eq!(offset, &OFFSET_SENTINEL);

        *offset = block_offset;

        let block = blocks
            .get_mut(4usize.strict_mul(mc_index)..4usize.strict_mul(mc_index).strict_add(4))
            .with_context(|| format!("block offset slot {mc_index} out of range"))?;
        block.copy_from_slice(&block_offset.to_le_bytes());
        debug_assert_eq!(snapshot.materials.len(), snapshot.occupied_count());

        blocks.extend_from_slice(&snapshot.mask);
        blocks.extend_from_slice(&snapshot.materials);

        while !blocks.len().is_multiple_of(8) {
            blocks.push(0);
        }

        let (min_cell, max_cell) = occupied_cell_bounds(&snapshot.mask)?;

        let min = mc_local_origin.saturating_add(min_cell);
        let max = mc_local_origin
            .saturating_add(max_cell)
            .saturating_add(IVec3::ONE);

        aabbs.push(AabbPositions {
            min: min.as_vec3().to_array(),
            max: max.as_vec3().to_array(),
        });
    }

    Ok(RegionData {
        region_index,
        #[cfg(test)]
        offset_table,
        blocks,
        aabbs,
    })
}

fn occupied_cell_bounds(mask: &[u8; 64]) -> anyhow::Result<(IVec3, IVec3)> {
    let mut min = IVec3::splat(i32::try_from(MICRO_CHUNK_LENGTH.strict_sub(1))?);
    let mut max = IVec3::ZERO;
    let mut any = false;

    for (z, row) in mask.as_chunks::<8>().0.iter().enumerate() {
        for (y, &byte) in row.iter().enumerate() {
            if byte == 0 {
                continue;
            }

            let x_min = i32::try_from(byte.trailing_zeros())?;
            let x_max = i32::try_from(7u32.strict_sub(byte.leading_zeros()))?;
            let y = i32::try_from(y)?;
            let z = i32::try_from(z)?;

            min = min.min(IVec3::new(x_min, y, z));
            max = max.max(IVec3::new(x_max, y, z));

            any = true;
        }
    }

    if !any {
        bail!("packed an empty snapshot as a hull");
    }

    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use crate::world::{World, snapshot::emit_snapshots};

    use super::*;

    #[test]
    fn packs_across_region_boundary() {
        let mut world = World::default();
        world.insert_voxel_at(IVec3::new(255, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(256, 0, 0), 2);

        let snapshots = emit_snapshots(&world).unwrap();
        let regions = pack_regions(&snapshots);

        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].region_index, IVec3::new(0, 0, 0));
        assert_eq!(regions[1].region_index, IVec3::new(1, 0, 0));

        assert_eq!(regions[1].aabbs[0].min, [0.0, 0.0, 0.0]);
        assert_eq!(regions[1].aabbs[0].max, [1.0, 1.0, 1.0]);

        assert_eq!(regions[0].origin(), IVec3::ZERO);
        assert_eq!(regions[1].origin(), IVec3::new(256, 0, 0));

        assert_eq!(regions[0].aabbs[0].min, [255.0, 0.0, 0.0]);
        assert_eq!(regions[0].aabbs[0].max, [256.0, 1.0, 1.0]);
    }

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

    #[test]
    fn pack_layout_invariants() {
        let mut world = World::default();

        world.insert_voxel_at(IVec3::new(0, 0, 0), 1);
        world.insert_voxel_at(IVec3::new(7, 7, 7), 2);
        world.insert_voxel_at(IVec3::new(8, 0, 0), 3);

        let snapshots = emit_snapshots(&world).unwrap();
        let regions = pack_regions(&snapshots);
        assert_eq!(regions.len(), 1);
        let region = &regions[0];

        assert_eq!(region.offset_table.len(), MICRO_CHUNKS_PER_REGION);
        assert_eq!(region.blocks.len() % 8, 0);
        assert_eq!(region.aabbs.len(), 2);

        let slot0 = region.offset_table[0];
        assert_ne!(slot0, OFFSET_SENTINEL);
        assert_eq!(slot0 % 8, 0);
        assert_eq!(slot0, OFFSET_TABLE_SIZE as u32);

        assert_ne!(region.offset_table[1], OFFSET_SENTINEL);
        assert_eq!(region.offset_table[1000], OFFSET_SENTINEL);

        let block = &region.blocks[slot0 as usize..];
        assert_eq!(block[0] & 1, 1);
        assert_eq!(block[63] & 0x80, 0x80);
        assert_eq!(block[64], 1);
        assert_eq!(block[65], 2);
    }

    #[test]
    fn microchunk_index_convention() {
        let mc = IVec3::new(1, 2, 3);
        let index = ((mc.z * 32 + mc.y) * 32 + mc.x) as usize;

        assert_eq!(index, (3 * 32 + 2) * 32 + 1);
        assert_eq!((31 * 32 + 31) * 32 + 31, MICRO_CHUNKS_PER_REGION - 1);
        assert_eq!(MICRO_CHUNK_LENGTH, 8);
    }
}

#[cfg(test)]
mod contract {
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::render::region::task::RenderMode;

    use super::*;

    const CONSTS: &str = include_str!("../../../shaders/common/common.glsl");

    const NAMES: &[&str] = &[
        "RAY_T_MIN",
        "RAY_T_MAX",
        "REGION_TABLE_ENTRIES",
        "REGION_ID_MASK",
        "MC_STRIDE_Y",
        "MC_STRIDE_Z",
        "VOXEL_STRIDE_Y",
        "VOXEL_STRIDE_Z",
        "MASK_BYTES",
        "OFFSET_SENTINEL",
        "MODE_VOXEL",
        "MODE_HULL",
        "MODE_NORMAL",
    ];

    fn define(name: &str) -> String {
        CONSTS
            .lines()
            .find_map(|line| {
                let mut parts = line.trim().split_whitespace();

                (parts.next() == Some("#define") && parts.next() == Some(name))
                    .then(|| parts.next().expect("define without value").to_string())
            })
            .unwrap_or_else(|| panic!("consts.glsl missing {name}"))
    }

    fn uint(name: &str) -> u64 {
        let value = define(name);

        match value.strip_prefix("0x") {
            Some(hex) => u64::from_str_radix(hex.trim_end_matches('u'), 16).unwrap(),
            None => value.trim_end_matches('u').parse().unwrap(),
        }
    }

    fn float(name: &str) -> f64 {
        define(name).parse().unwrap()
    }

    #[test]
    fn glsl_matches_rust() {
        let side = MC_PER_REGION_SIDE as u64;
        let voxel = MICRO_CHUNK_LENGTH as u64;

        assert_eq!(uint("REGION_TABLE_ENTRIES"), REGION_COUNT as u64);
        assert_eq!(uint("REGION_ID_MASK"), REGION_COUNT as u64 - 1);
        assert_eq!(uint("MC_STRIDE_Y"), side);
        assert_eq!(uint("MC_STRIDE_Z"), side * side);
        assert_eq!(uint("VOXEL_STRIDE_Y"), voxel);
        assert_eq!(uint("VOXEL_STRIDE_Z"), voxel * voxel);
        assert_eq!(uint("MASK_BYTES"), voxel * voxel * voxel / 8);
        assert_eq!(uint("OFFSET_SENTINEL"), OFFSET_SENTINEL as u64);

        assert_eq!(uint("MODE_VOXEL"), RenderMode::Voxel as u32 as u64);

        #[cfg(debug_assertions)]
        {
            assert_eq!(uint("MODE_HULL"), RenderMode::Hull as u32 as u64);
            assert_eq!(uint("MODE_NORMAL"), RenderMode::Normal as u32 as u64);
        }

        assert_eq!(float("RAY_T_MIN"), 0.01);
        assert_eq!(float("RAY_T_MAX"), 10000.0);
    }

    #[test]
    fn contract_declared_once() {
        let mut sources = Vec::new();
        collect_shaders(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders"),
            &mut sources,
        );

        for path in sources {
            if path.file_name().is_some_and(|name| name == "common.glsl") {
                continue;
            }

            let text = fs::read_to_string(&path).unwrap();

            for name in NAMES {
                let redeclared = text.contains(&format!("#define {name} "))
                    || text.contains(&format!("{name} ="));

                assert!(
                    !redeclared,
                    "{} declares contract constant {name}",
                    path.display()
                );
            }
        }
    }

    fn collect_shaders(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();

            if path.is_dir() {
                collect_shaders(&path, out);
            } else {
                out.push(path);
            }
        }
    }
}
