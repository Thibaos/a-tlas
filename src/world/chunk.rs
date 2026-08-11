#![allow(unused)]
use std::{collections::HashMap, fmt::Display};

use dot_vox::DotVoxData;
use glam::{IVec3, UVec3, Vec4, Vec4Swizzles};

use crate::world::{HostVoxel, loader::SceneGraphTraverser};

#[cfg(debug_assertions)]
use super::Vertex3DColor;

// The voxel length in meters
pub const VOXEL_PHYSICAL_LENGTH: f32 = 1.0 / 16.0;

// The amount of voxels per chunk dimension
pub const CHUNK_WIDTH: u32 = 64;

// The amount of chunks in the world's X axis
pub const WORLD_WIDTH: i32 = 64;
// The amount of chunks in the world's Y axis
pub const WORLD_HEIGHT: i32 = 64;
// The amount of chunks in the world's Z axis
pub const WORLD_DEPTH: i32 = 64;

struct Bounds(i32, i32);

impl Bounds {
    pub const fn inside(&self, value: i32) -> bool {
        value > self.0 && value < self.1
    }
}

#[derive(Debug)]
pub struct Chunk {
    voxels: HashMap<UVec3, HostVoxel>,
}

impl Default for Chunk {
    fn default() -> Self {
        Chunk {
            voxels: HashMap::new(),
        }
    }
}

impl Chunk {
    pub fn empty(&self) -> bool {
        self.voxels.is_empty()
    }

    pub fn contains(&self, position: &UVec3) -> bool {
        self.voxels.contains_key(position)
    }

    pub fn insert(&mut self, position: UVec3, voxel: HostVoxel) -> bool {
        if position.x >= CHUNK_WIDTH || position.y >= CHUNK_WIDTH || position.z >= CHUNK_WIDTH {
            panic!("Inserted voxel outside of chunk bounds: {position}");
        }

        if self.voxels.contains_key(&position) {
            return false;
        }

        self.voxels.insert(position, voxel).is_none()
    }

    #[cfg(debug_assertions)]
    pub fn debug_lines(&self, grid_position: IVec3) -> Vec<Vertex3DColor> {
        let color = if self.empty() {
            [0.5, 0.5, 0.5, 0.1]
        } else {
            [0.0, 1.0, 0.0, 1.0]
        };

        let origin = grid_position * CHUNK_WIDTH as i32;
        let origin_array = origin.to_array();
        let origin_array = [
            origin_array[0] as f32 - 0.5,
            origin_array[1] as f32 - 0.5,
            origin_array[2] as f32 - 0.5,
        ];

        let width = CHUNK_WIDTH as f32;

        let dlf = origin_array;
        let dlb = [origin_array[0], origin_array[1], origin_array[2] + width];
        let drf = [origin_array[0] + width, origin_array[1], origin_array[2]];
        let drb = [
            origin_array[0] + width,
            origin_array[1],
            origin_array[2] + width,
        ];

        let ulf = [origin_array[0], origin_array[1] + width, origin_array[2]];
        let ulb = [
            origin_array[0],
            origin_array[1] + width,
            origin_array[2] + width,
        ];
        let urf = [
            origin_array[0] + width,
            origin_array[1] + width,
            origin_array[2],
        ];
        let urb = [
            origin_array[0] + width,
            origin_array[1] + width,
            origin_array[2] + width,
        ];

        vec![
            // bottom
            Vertex3DColor {
                position: dlf,
                color,
            },
            Vertex3DColor {
                position: dlb,
                color,
            },
            Vertex3DColor {
                position: dlb,
                color,
            },
            Vertex3DColor {
                position: drb,
                color,
            },
            Vertex3DColor {
                position: drb,
                color,
            },
            Vertex3DColor {
                position: drf,
                color,
            },
            Vertex3DColor {
                position: drf,
                color,
            },
            Vertex3DColor {
                position: dlf,
                color,
            },
            // up
            Vertex3DColor {
                position: ulf,
                color,
            },
            Vertex3DColor {
                position: ulb,
                color,
            },
            Vertex3DColor {
                position: ulb,
                color,
            },
            Vertex3DColor {
                position: urb,
                color,
            },
            Vertex3DColor {
                position: urb,
                color,
            },
            Vertex3DColor {
                position: urf,
                color,
            },
            Vertex3DColor {
                position: urf,
                color,
            },
            Vertex3DColor {
                position: ulf,
                color,
            },
            // sides
            Vertex3DColor {
                position: dlf,
                color,
            },
            Vertex3DColor {
                position: ulf,
                color,
            },
            Vertex3DColor {
                position: dlb,
                color,
            },
            Vertex3DColor {
                position: ulb,
                color,
            },
            Vertex3DColor {
                position: drf,
                color,
            },
            Vertex3DColor {
                position: urf,
                color,
            },
            Vertex3DColor {
                position: drb,
                color,
            },
            Vertex3DColor {
                position: urb,
                color,
            },
        ]
    }
}

pub type ChunksInner = HashMap<IVec3, Chunk>;

#[derive(Default)]
pub struct Chunks {
    inner: ChunksInner,
}

impl Chunks {
    const X_BOUNDS: Bounds = Bounds(
        -WORLD_WIDTH * CHUNK_WIDTH as i32,
        WORLD_WIDTH * CHUNK_WIDTH as i32,
    );

    const Y_BOUNDS: Bounds = Bounds(
        -WORLD_HEIGHT * CHUNK_WIDTH as i32,
        WORLD_HEIGHT * CHUNK_WIDTH as i32,
    );

    const Z_BOUNDS: Bounds = Bounds(
        -WORLD_DEPTH * CHUNK_WIDTH as i32,
        WORLD_DEPTH * CHUNK_WIDTH as i32,
    );

    const fn in_bounds(position: &IVec3) -> bool {
        Chunks::X_BOUNDS.inside(position.x)
            && Chunks::Y_BOUNDS.inside(position.y)
            && Chunks::Z_BOUNDS.inside(position.z)
    }

    fn create_empty_chunks() -> ChunksInner {
        (-WORLD_WIDTH..WORLD_WIDTH)
            .flat_map(move |x| {
                (-WORLD_HEIGHT..WORLD_HEIGHT).flat_map(move |y| {
                    (-WORLD_DEPTH..WORLD_DEPTH)
                        .map(move |z| (IVec3::new(x, y, z), Chunk::default()))
                })
            })
            .collect()
    }

    fn translation_to_position(position: &IVec3) -> (IVec3, UVec3) {
        if !Chunks::in_bounds(position) {
            panic!("Out of bounds: {position}");
        }

        let IVec3 { x, y, z } = position;

        let chunk_width = CHUNK_WIDTH as i32;

        let grid_position = IVec3::new(
            if x.is_negative() {
                -chunk_width + x + 1
            } else {
                *x
            } / chunk_width,
            if y.is_negative() {
                -chunk_width + y + 1
            } else {
                *y
            } / chunk_width,
            if z.is_negative() {
                -chunk_width + z + 1
            } else {
                *z
            } / chunk_width,
        );

        let chunk_min_corner = grid_position * chunk_width;
        let IVec3 { x, y, z } = position - chunk_min_corner;

        assert!(!x.is_negative());
        assert!(x <= chunk_width);

        assert!(!y.is_negative());
        assert!(y <= chunk_width);

        assert!(!z.is_negative());
        assert!(z <= chunk_width);

        let local_position = UVec3::new(x as u32, y as u32, z as u32);

        (grid_position, local_position)
    }

    pub fn new(voxel_data: &DotVoxData) -> Self {
        let mut chunks = Chunks::create_empty_chunks();

        let mut loader = SceneGraphTraverser {
            chunks: &mut chunks,
            scene: voxel_data,
            models: vec![],
        };

        loader.traverse();

        for (translation, rotation, size, voxels) in loader.models {
            let transform = SceneGraphTraverser::to_transform(translation, rotation, size);

            for voxel in voxels {
                let local_position =
                    UVec3::new(voxel.x as u32, voxel.z as u32, size.y - voxel.y as u32 - 1)
                        .as_ivec3();

                let position = (transform
                    * Vec4::new(
                        local_position.x as f32,
                        local_position.y as f32,
                        local_position.z as f32,
                        1.0,
                    ))
                .xyz()
                .as_ivec3();

                let p = IVec3::new(position.x, position.y, -position.z);

                Chunks::insert_voxel(&mut chunks, p, HostVoxel::new(voxel.i.into()));
            }
        }

        Self { inner: chunks }
    }

    fn from(inner: ChunksInner) -> Self {
        Self { inner }
    }

    #[cfg(debug_assertions)]
    pub fn debug_lines(&self) -> Vec<Vertex3DColor> {
        self.inner
            .iter()
            .filter(|(_, c)| !c.empty())
            .flat_map(|(grid_position, chunk)| chunk.debug_lines(*grid_position))
            .collect()
    }

    pub fn contains(&self, position: &IVec3) -> bool {
        let (grid_position, local_position) = Chunks::translation_to_position(position);

        let chunk = self.inner.get(&grid_position).unwrap();

        chunk.contains(&local_position)
    }

    pub fn get_voxel(&self, position: &IVec3) -> Option<&HostVoxel> {
        let (grid_position, local_position) = Chunks::translation_to_position(position);

        let chunk = self.inner.get(&grid_position).unwrap();

        chunk.voxels.get(&local_position)
    }

    /// Like [`Chunks::get_voxel`] but returns `None` instead of panicking for
    /// positions outside the world bounds (used by the reference tracer's grid
    /// walk, which may probe one cell past the occupied set).
    pub(crate) fn try_get_voxel(&self, position: &IVec3) -> Option<&HostVoxel> {
        if !Chunks::in_bounds(position) {
            return None;
        }

        let (grid_position, local_position) = Chunks::translation_to_position(position);

        self.inner.get(&grid_position)?.voxels.get(&local_position)
    }

    /// Inserts a voxel with the given palette index (test/validate helper).
    /// Lazily creates the chunk, so small hand-built worlds stay cheap.
    pub(crate) fn insert_voxel_at(&mut self, position: IVec3, material_index: u32) {
        let (grid_position, local_position) = Chunks::translation_to_position(&position);
        self.inner
            .entry(grid_position)
            .or_default()
            .voxels
            .insert(local_position, HostVoxel::new(material_index));
    }

    /// Removes the voxel at `position` (world-side edit helper used by the
    /// validator's edit-at-the-seam flow). Returns whether a voxel was
    /// present. Empty chunks stay allocated (the lattice is pre-allocated);
    /// `iter_voxels` skips them.
    pub(crate) fn remove_voxel_at(&mut self, position: IVec3) -> bool {
        let (grid_position, local_position) = Chunks::translation_to_position(&position);
        let Some(chunk) = self.inner.get_mut(&grid_position) else {
            return false;
        };
        chunk.voxels.remove(&local_position).is_some()
    }

    /// Iterates every occupied voxel in the world as (global position, voxel).
    ///
    /// The validator's reference tracer reads the world side of the
    /// renderer input contract through this iterator (plus `get_voxel` and the
    /// palette) — it never touches renderer state.
    pub fn iter_voxels(&self) -> impl Iterator<Item = (IVec3, &HostVoxel)> + '_ {
        self.inner.iter().flat_map(|(grid_position, chunk)| {
            let cw = CHUNK_WIDTH as i32;
            let chunk_origin = *grid_position * cw;

            chunk.voxels.iter().map(move |(local_position, voxel)| {
                (
                    IVec3::new(
                        chunk_origin.x + local_position.x as i32,
                        chunk_origin.y + local_position.y as i32,
                        chunk_origin.z + local_position.z as i32,
                    ),
                    voxel,
                )
            })
        })
    }

    /// The inclusive axis-aligned bounds of the occupied voxel set, or `None`
    /// for an empty world.
    pub fn voxel_bounds(&self) -> Option<(IVec3, IVec3)> {
        let mut min: Option<IVec3> = None;
        let mut max: Option<IVec3> = None;

        for (position, _) in self.iter_voxels() {
            min = Some(match min {
                None => position,
                Some(m) => m.min(position),
            });
            max = Some(match max {
                None => position,
                Some(m) => m.max(position),
            });
        }

        min.zip(max)
    }

    /// The number of occupied voxels in the world.
    pub fn voxel_count(&self) -> usize {
        self.iter_voxels().count()
    }

    pub fn insert_voxel(
        chunks: &mut ChunksInner,
        position: IVec3,
        voxel: HostVoxel,
    ) -> Option<IVec3> {
        let (grid_position, local_position) = Chunks::translation_to_position(&position);

        let current_chunk = chunks.get_mut(&grid_position).unwrap();

        if !current_chunk.insert(local_position, voxel) {
            return None;
        }

        Some(grid_position)
    }
}

impl Display for Chunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (grid_position, voxel_count) in self
            .inner
            .iter()
            .map(|(grid_position, chunk)| (grid_position, chunk.voxels.len()))
        {
            writeln!(f, "({grid_position:?}, voxels: {voxel_count})")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use glam::{IVec3, UVec3};

    use super::{CHUNK_WIDTH, Chunk, Chunks};
    use crate::world::{HostVoxel, chunk::WORLD_WIDTH};

    #[test]
    fn chunk_insert() {
        use glam::UVec3;

        let mut chunk = Chunk::default();

        for x in 0..CHUNK_WIDTH {
            for y in 0..CHUNK_WIDTH {
                for z in 0..CHUNK_WIDTH {
                    chunk.insert(UVec3::new(x, y, z), HostVoxel { material_index: 0 });
                }
            }
        }

        assert!(chunk.voxels.len() as u32 == CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_WIDTH);
    }

    #[test]
    fn chunks_insert() {
        let mut chunks = Chunks::create_empty_chunks();

        let size = CHUNK_WIDTH as i32;

        for x in 0..WORLD_WIDTH * size {
            let position = IVec3::new(x, 0, 0);

            let res = Chunks::insert_voxel(&mut chunks, position, HostVoxel { material_index: 0 });

            assert!(res.is_some());

            let grid_position = res.unwrap();

            assert!(grid_position == IVec3::new(x / size, 0, 0));
        }

        let sum = chunks.values().map(|c| c.voxels.len() as u32).sum::<u32>();

        assert!(sum == WORLD_WIDTH as u32 * CHUNK_WIDTH);
    }

    #[test]
    fn chunk_contains() {
        let mut chunk = Chunk::default();

        let pos1 = UVec3::new(1, 1, 1);
        let pos2 = UVec3::new(32, 19, 15);
        let pos3 = UVec3::new(8, 8, 8);

        chunk.insert(pos1, HostVoxel::default());
        chunk.insert(pos2, HostVoxel::default());
        chunk.insert(pos3, HostVoxel::default());

        assert!(chunk.contains(&pos1));
        assert!(chunk.contains(&pos2));
        assert!(chunk.contains(&pos3));
    }

    #[test]
    fn chunks_contains() {
        let mut inner = Chunks::create_empty_chunks();

        let pos1 = IVec3::new(1, 1, 1);
        let pos2 = IVec3::new(120, 129, -215);

        Chunks::insert_voxel(&mut inner, pos1, HostVoxel::default());
        Chunks::insert_voxel(&mut inner, pos2, HostVoxel::default());

        let chunks = Chunks::from(inner);

        assert!(chunks.contains(&pos1));
        assert!(chunks.contains(&pos2));
    }
}
