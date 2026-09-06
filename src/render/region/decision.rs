use anyhow::Context;
use glam::IVec3;

use crate::render::region::pack::RegionData;
use crate::world::grid::region_id;

#[derive(Clone, Copy, Debug)]
pub struct RegionSlot {
    pub pool_capacity: u64,
    pub aabb_capacity: u32,
}

pub enum RegionEffect {
    Ignore,
    Enter {
        pool_bytes: u64,
        aabbs: u32,
        pack: RegionData,
    },
    Exit {
        retire_pool: u64,
        retire_blas: u32,
    },
    Update {
        pool_bytes: u64,
        aabbs: u32,
        retire_pool: Option<u64>,
        retire_blas: Option<u32>,
        pack: RegionData,
    },
}

#[derive(Default)]
pub struct ResidencyDecision {
    pub effects: Vec<(u32, RegionEffect)>,
    pub resident_ids: Vec<u32>,
    pub became_resident: Vec<IVec3>,
    pub left_resident: Vec<IVec3>,
    pub dirty: Vec<IVec3>,
    pub blas_replaced: Vec<IVec3>,
    pub tlas_dirty: bool,
    pub table_changed: bool,
}

pub fn decide(
    slots: &[Option<RegionSlot>],
    resident_ids: &[u32],
    packs: impl IntoIterator<Item = (IVec3, Option<RegionData>)>,
) -> anyhow::Result<ResidencyDecision> {
    let mut decision = ResidencyDecision {
        resident_ids: resident_ids.to_vec(),
        ..ResidencyDecision::default()
    };

    for (region_index, pack) in packs {
        let id = region_id(region_index);

        let was_resident = slots
            .get(usize::try_from(id)?)
            .context(format!("region slot {id} out of range"))?
            .is_some();

        let effect = match (was_resident, pack) {
            (false, None) => RegionEffect::Ignore,

            (false, Some(pack)) => {
                insert_resident(&mut decision.resident_ids, id);
                decision.became_resident.push(region_index);
                decision.tlas_dirty = true;
                decision.table_changed = true;
                RegionEffect::Enter {
                    pool_bytes: u64::try_from(pack.blocks.len())?,
                    aabbs: u32::try_from(pack.aabbs.len())?,
                    pack,
                }
            }

            (true, None) => {
                let slot = slot_of(slots, id)?;

                remove_resident(&mut decision.resident_ids, id);
                decision.left_resident.push(region_index);
                decision.tlas_dirty = true;
                decision.table_changed = true;

                RegionEffect::Exit {
                    retire_pool: slot.pool_capacity,
                    retire_blas: slot.aabb_capacity,
                }
            }

            (true, Some(pack)) => {
                let slot = slot_of(slots, id)?;

                let pool_grows = slot.pool_capacity < u64::try_from(pack.blocks.len())?;
                let blas_grows = slot.aabb_capacity < u32::try_from(pack.aabbs.len())?;

                if pool_grows {
                    decision.table_changed = true;
                }

                if blas_grows {
                    decision.blas_replaced.push(region_index);
                    decision.tlas_dirty = true;
                }

                decision.dirty.push(region_index);

                RegionEffect::Update {
                    pool_bytes: u64::try_from(pack.blocks.len())?,
                    aabbs: u32::try_from(pack.aabbs.len())?,
                    retire_pool: pool_grows.then_some(slot.pool_capacity),
                    retire_blas: blas_grows.then_some(slot.aabb_capacity),
                    pack,
                }
            }
        };

        decision.effects.push((id, effect));
    }

    Ok(decision)
}

fn slot_of(slots: &[Option<RegionSlot>], id: u32) -> anyhow::Result<&RegionSlot> {
    slots
        .get(usize::try_from(id)?)
        .context(format!("region slot {id} out of range"))?
        .as_ref()
        .context(format!("region {id} is not resident"))
}

fn insert_resident(resident_ids: &mut Vec<u32>, id: u32) {
    match resident_ids.binary_search(&id) {
        Ok(_) => panic!("region {id} already resident"),
        Err(position) => resident_ids.insert(position, id),
    }
}

fn remove_resident(resident_ids: &mut Vec<u32>, id: u32) {
    let position = resident_ids
        .binary_search(&id)
        .unwrap_or_else(|_| panic!("region {id} not resident"));
    resident_ids.remove(position);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::render::region::pack::REGION_COUNT;

    use vulkano::acceleration_structure::AabbPositions;

    fn empty_slots() -> Vec<Option<RegionSlot>> {
        vec![None; REGION_COUNT]
    }
    fn slot(pool_capacity: u64, aabb_capacity: u32) -> Option<RegionSlot> {
        Some(RegionSlot {
            pool_capacity,
            aabb_capacity,
        })
    }

    fn pack(pool_bytes: u64, aabbs: u32) -> Option<RegionData> {
        Some(RegionData {
            region_index: IVec3::ZERO,
            offset_table: Vec::new(),
            blocks: vec![0; pool_bytes as usize],
            aabbs: vec![
                AabbPositions {
                    min: [0.0; 3],
                    max: [1.0; 3]
                };
                aabbs as usize
            ],
        })
    }

    #[test]
    fn enter_makes_an_empty_region_resident() {
        let index = IVec3::new(1, 2, 3);
        let packs = [(index, pack(512, 4))];

        let decision = decide(&empty_slots(), &[], packs).unwrap();

        assert!(matches!(
            &decision.effects[0].1,
            RegionEffect::Enter {
                pool_bytes: 512,
                aabbs: 4,
                ..
            }
        ));
        assert_eq!(decision.became_resident, [index]);
        assert_eq!(decision.resident_ids, [region_id(index)]);
        assert!(decision.tlas_dirty);
        assert!(decision.table_changed);
    }

    #[test]
    fn exit_retires_the_slot_and_leaves_the_lattice() {
        let index = IVec3::ZERO;
        let id = region_id(index);
        let mut slots = empty_slots();
        slots[id as usize] = slot(256, 8);

        let decision = decide(&slots, &[id], [(index, None)]).unwrap();

        assert!(matches!(
            &decision.effects[0].1,
            RegionEffect::Exit {
                retire_pool: 256,
                retire_blas: 8
            }
        ));
        assert_eq!(decision.left_resident, [index]);
        assert!(decision.resident_ids.is_empty());
        assert!(decision.tlas_dirty);
        assert!(decision.table_changed);
    }

    #[test]
    fn update_grows_only_what_does_not_fit() {
        let index = IVec3::ZERO;
        let id = region_id(index);
        let mut slots = empty_slots();
        slots[id as usize] = slot(256, 8);

        let decision = decide(&slots, &[id], [(index, pack(512, 4))]).unwrap();

        assert!(matches!(
            &decision.effects[0].1,
            RegionEffect::Update {
                pool_bytes: 512,
                aabbs: 4,
                retire_pool: Some(256),
                retire_blas: None,
                ..
            }
        ));
        assert_eq!(decision.dirty, [index]);
        assert!(decision.blas_replaced.is_empty());
        assert!(!decision.tlas_dirty);
        assert!(decision.table_changed);
    }

    #[test]
    fn update_growing_the_blas_dirties_the_tlas() {
        let index = IVec3::ZERO;
        let id = region_id(index);
        let mut slots = empty_slots();
        slots[id as usize] = slot(256, 8);

        let decision = decide(&slots, &[id], [(index, pack(128, 16))]).unwrap();

        assert!(matches!(
            &decision.effects[0].1,
            RegionEffect::Update {
                pool_bytes: 128,
                aabbs: 16,
                retire_pool: None,
                retire_blas: Some(8),
                ..
            }
        ));
        assert_eq!(decision.blas_replaced, [index]);
        assert!(decision.tlas_dirty);
        assert!(!decision.table_changed);
    }

    #[test]
    fn update_without_growth_rebuilds_in_place() {
        let index = IVec3::ZERO;
        let id = region_id(index);
        let mut slots = empty_slots();
        slots[id as usize] = slot(256, 8);

        let decision = decide(&slots, &[id], [(index, pack(128, 4))]).unwrap();

        assert!(matches!(
            &decision.effects[0].1,
            RegionEffect::Update {
                pool_bytes: 128,
                aabbs: 4,
                retire_pool: None,
                retire_blas: None,
                ..
            }
        ));
        assert_eq!(decision.dirty, [index]);
        assert!(!decision.tlas_dirty);
        assert!(!decision.table_changed);
    }

    #[test]
    fn pending_frees_follow_pack_order_and_never_reenter_the_free_lists() {
        let left = IVec3::new(-1, 0, 0);
        let grown = IVec3::ZERO;
        let entered = IVec3::new(1, 0, 0);

        let mut slots = empty_slots();
        slots[region_id(left) as usize] = slot(64, 2);
        slots[region_id(grown) as usize] = slot(128, 4);
        let resident = [region_id(left), region_id(grown)];

        let packs = [(left, None), (grown, pack(256, 8)), (entered, pack(32, 1))];

        let decision = decide(&slots, &resident, packs).unwrap();

        let effect_ids: Vec<u32> = decision.effects.iter().map(|&(id, _)| id).collect();
        assert_eq!(
            effect_ids,
            [region_id(left), region_id(grown), region_id(entered)]
        );

        assert!(matches!(
            &decision.effects[0].1,
            RegionEffect::Exit {
                retire_pool: 64,
                retire_blas: 2
            }
        ));
        assert!(matches!(
            &decision.effects[1].1,
            RegionEffect::Update {
                retire_pool: Some(128),
                retire_blas: Some(4),
                ..
            }
        ));
        assert!(matches!(&decision.effects[2].1, RegionEffect::Enter { .. }));

        assert_eq!(
            decision.resident_ids,
            [region_id(grown), region_id(entered)]
        );
    }

    #[test]
    fn an_empty_region_with_no_pack_is_ignored() {
        let index = IVec3::new(2, 3, 4);

        let decision = decide(&empty_slots(), &[], [(index, None)]).unwrap();

        assert!(matches!(&decision.effects[0].1, RegionEffect::Ignore));
        assert!(decision.resident_ids.is_empty());
        assert!(decision.became_resident.is_empty());
        assert!(decision.left_resident.is_empty());
        assert!(decision.dirty.is_empty());
        assert!(decision.blas_replaced.is_empty());
        assert!(!decision.tlas_dirty);
        assert!(!decision.table_changed);
    }
}
