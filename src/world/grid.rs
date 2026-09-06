#![allow(clippy::as_conversions)]
// Lattice edges are crate constants, bounded far below i32 ranges.
#![allow(clippy::cast_possible_wrap)]
use glam::{IVec3, UVec3};

pub const MICRO_CHUNK_LENGTH: u32 = 8;
pub const REGION_LENGTH: u32 = 256;
pub const REGION_HALF_EXTENT: u32 = 8;
pub const LATTICE_HALF_EXTENT: u32 = REGION_HALF_EXTENT * REGION_LENGTH;

pub fn grid_index(global: IVec3, edge: u32) -> IVec3 {
    global.div_euclid(IVec3::splat(edge as i32))
}

pub fn grid_origin(global: IVec3, edge: u32) -> IVec3 {
    grid_index(global, edge).saturating_mul(IVec3::splat(edge as i32))
}

pub fn in_lattice(global: IVec3) -> bool {
    let half = IVec3::splat(LATTICE_HALF_EXTENT as i32);

    global.cmpge(half.saturating_mul(IVec3::splat(-1))).all() && global.cmplt(half).all()
}

pub fn region_index_in_lattice(region_index: IVec3) -> bool {
    let half = IVec3::splat(REGION_HALF_EXTENT as i32);

    region_index.cmpge(half.saturating_mul(IVec3::splat(-1))).all()
        && region_index.cmplt(half).all()
}

pub fn assert_region_index_in_lattice(region_index: IVec3) {
    assert!(
        region_index_in_lattice(region_index),
        "region index {region_index} exceeds the renderer lattice (±{LATTICE_HALF_EXTENT}/axis, region indices in [-{REGION_HALF_EXTENT}, {REGION_HALF_EXTENT})"
    );
}

pub fn region_id(region_index: IVec3) -> u32 {
    assert_region_index_in_lattice(region_index);
    let UVec3 { x, y, z } = region_index.as_uvec3();

    (((x.wrapping_add(REGION_HALF_EXTENT)) & 0xF) << 8)
        | ((y.wrapping_add(REGION_HALF_EXTENT) & 0xF) << 4)
        | (z.wrapping_add(REGION_HALF_EXTENT) & 0xF)
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
            assert_eq!(grid_index(origin, edge) * edge as i32, origin);
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
