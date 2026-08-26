//! Origin-aligned grid math and the renderer lattice, the single source of
//! truth for the floor-division and lattice-bounds logic the renderer and the
//! world share. The renderer owns the grid: its extent is the 12-bit
//! region-id budget, and the world stays region-agnostic by
//! importing only the voxel-space extent and [`in_lattice`], never a Region
//! or Micro-chunk constant.
//!
//! Before this module, floor-division of a possibly-negative coord into an
//! origin-aligned cell was written three ways (a hand-rolled `+1` trick in
//! `world/chunk.rs`, `div_euclid` in `region/pack.rs`, and
//! `region/snapshot.rs`), and the world's extent
//! (±4096) disagreed with the renderer lattice (±2048). These primitives
//! concentrate the math and pin the extent to one constant.

use glam::IVec3;

pub const MICRO_CHUNK_LENGTH: i32 = 8;
pub const REGION_LENGTH: i32 = 256;
pub const REGION_HALF_EXTENT: i32 = 8;
pub const LATTICE_HALF_EXTENT: i32 = REGION_HALF_EXTENT * REGION_LENGTH;

pub fn grid_index(global: IVec3, edge: i32) -> IVec3 {
    global.div_euclid(IVec3::splat(edge))
}

pub fn grid_origin(global: IVec3, edge: i32) -> IVec3 {
    grid_index(global, edge) * edge
}

pub fn in_lattice(global: IVec3) -> bool {
    global.cmpge(IVec3::splat(-LATTICE_HALF_EXTENT)).all()
        && global.cmplt(IVec3::splat(LATTICE_HALF_EXTENT)).all()
}

pub fn region_index_in_lattice(region_index: IVec3) -> bool {
    region_index.cmpge(IVec3::splat(-REGION_HALF_EXTENT)).all()
        && region_index.cmplt(IVec3::splat(REGION_HALF_EXTENT)).all()
}

pub fn assert_region_index_in_lattice(region_index: IVec3) {
    if !region_index_in_lattice(region_index) {
        panic!(
            "micro-chunk/region index {region_index} exceeds the renderer lattice (±{LATTICE_HALF_EXTENT}/axis, region indices in [-{REGION_HALF_EXTENT}, {REGION_HALF_EXTENT}), the 12-bit region-id budget); the model is too big to fit the v1 acceleration-structure lattice"
        );
    }
}

pub fn region_id(region_index: IVec3) -> u32 {
    assert_region_index_in_lattice(region_index);
    (((region_index.x + REGION_HALF_EXTENT) as u32 & 0xF) << 8)
        | (((region_index.y + REGION_HALF_EXTENT) as u32 & 0xF) << 4)
        | ((region_index.z + REGION_HALF_EXTENT) as u32 & 0xF)
}

pub fn region_index_of(global_coords: IVec3) -> IVec3 {
    grid_index(global_coords, REGION_LENGTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_index_floors_negative_coords() {
        assert_eq!(
            grid_index(IVec3::new(-8, -8, -8), 8),
            IVec3::new(-1, -1, -1)
        );
        assert_eq!(grid_index(IVec3::new(-1, 0, 1), 8), IVec3::new(-1, 0, 0));
        assert_eq!(grid_index(IVec3::new(0, 0, 0), 8), IVec3::ZERO);
        assert_eq!(grid_index(IVec3::new(7, 0, 0), 8), IVec3::ZERO);
        assert_eq!(grid_index(IVec3::new(8, 0, 0), 8), IVec3::new(1, 0, 0));
        assert_eq!(grid_index(IVec3::new(255, 0, 0), 256), IVec3::ZERO);
        assert_eq!(grid_index(IVec3::new(256, 0, 0), 256), IVec3::new(1, 0, 0));
        assert_eq!(
            grid_index(IVec3::new(-256, 0, 0), 256),
            IVec3::new(-1, 0, 0)
        );
    }

    #[test]
    fn grid_origin_round_trips() {
        for &(p, edge) in &[
            (IVec3::new(-1, -1, -1), 8),
            (IVec3::new(0, 0, 0), 8),
            (IVec3::new(3000, -3000, 7), 256),
        ] {
            let origin = grid_origin(p, edge);
            assert_eq!(grid_index(origin, edge) * edge, origin);
            assert_eq!(grid_index(origin, edge), grid_index(p, edge));
        }
    }

    #[test]
    fn lattice_is_half_open() {
        assert!(in_lattice(IVec3::new(-2048, -2048, -2048)));
        assert!(in_lattice(IVec3::new(2047, 2047, 2047)));
        assert!(!in_lattice(IVec3::new(2048, 0, 0)));
        assert!(!in_lattice(IVec3::new(-2049, 0, 0)));
    }

    #[test]
    fn extent_derives_from_region_budget() {
        assert_eq!(REGION_HALF_EXTENT, 8);
        assert_eq!(LATTICE_HALF_EXTENT, REGION_HALF_EXTENT * REGION_LENGTH);
        assert_eq!(LATTICE_HALF_EXTENT, 2048);
    }

    #[test]
    fn region_index_floor_division() {
        assert_eq!(
            region_index_of(IVec3::new(-8, -8, -8)),
            IVec3::new(-1, -1, -1)
        );
        assert_eq!(region_index_of(IVec3::new(0, 0, 0)), IVec3::ZERO);
        assert_eq!(region_index_of(IVec3::new(248, 0, 0)), IVec3::ZERO);
        assert_eq!(region_index_of(IVec3::new(256, 0, 0)), IVec3::new(1, 0, 0));
    }

    #[test]
    fn region_id_encodes_12_bits_at_budget_ends() {
        assert_eq!(region_id(IVec3::new(-8, -8, -8)), 0);
        assert_eq!(region_id(IVec3::new(7, 7, 7)), 0xFFF);
        assert!(region_index_in_lattice(IVec3::new(-8, 0, 0)));
        assert!(region_index_in_lattice(IVec3::new(7, 0, 0)));
        assert!(!region_index_in_lattice(IVec3::new(8, 0, 0)));
        assert!(!region_index_in_lattice(IVec3::new(-9, 0, 0)));
    }
}
