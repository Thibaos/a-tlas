//! Hand-authored test worlds (assets/test/) and their cameras.
//!
//! The validator renders each world through the real renderer and compares the
//! captured frame against the CPU reference tracer. The worlds are chosen to
//! exercise the renderer input contract's edge cases (rendering-core ticket 06
//! / ADR 0003): hull-empty space, Micro-chunk/Chunk/Region boundaries,
//! camera-in-voxel, far-plane miss, material index 0 (a real color via the
//! 8-bit hitKind), and a solid volume with an interior camera (DDA interior
//! voxels, not hollow shells). Each world is authored as a
//! deterministic .vox file written to assets/test/ on demand, so the validator
//! also exercises the real world loader (open_file + Chunks::new).
//!
//! Authoring convention (verified by the unit tests below): a scene-less .vox
//! model voxel (x, y, z) lands at world (x, z, y) — the loader's direct
//! mapping. Voxel coordinates are u8 in the .vox format, so positions beyond
//! 255 (the Region boundary) are authored through scene-graph transforms,
//! whose translation lands at world (t.x, t.z, t.y + 1) for an unrotated
//! 1-voxel model.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use dot_vox::{Color, Dict, DotVoxData, Frame, Model, SceneNode, ShapeModel, Size, Voxel};
use glam::IVec3;

pub const TEST_WORLDS_DIR: &str = "assets/test";

/// A fixed camera for a test world — part of the validator's camera inputs,
/// shared verbatim by the GPU ray pass and the reference tracer.
#[derive(Clone, Copy, Debug)]
pub struct CameraSpec {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
}

/// A `None` camera means "frame the world's bounding box" (used by the smoke
/// world).
pub type WorldCamera = Option<CameraSpec>;

pub struct WorldSpec {
    pub name: String,
    /// Path relative to the repo root (assets/test/<name>.vox for the test
    /// worlds, assets/custom.vox for the smoke world).
    pub path: String,
    pub description: String,
    pub camera: WorldCamera,
    /// Optional edit-at-the-seam script (renderer-impl tickets 03/04): after
    /// the first frame passes, each step mutates the world (voxel
    /// removals/additions), voices the change through the input contract,
    /// and renders another frame compared against the edited world. The
    /// edit-seam world empties one Micro-chunk (ticket 03); the residency
    /// world empties a whole Region and re-populates it (ticket 04).
    pub edit: Option<EditScript>,
}

/// One edit step between frames: world edits voiced through the input
/// contract, then a frame compared against the edited world.
#[derive(Clone, Debug)]
pub struct EditStep {
    /// Suffixes the frame's artifacts ("-after-empty", "-after-repop").
    pub label: String,
    /// Micro-chunks to empty: the world removes its voxels, then voices a
    /// zero-mask snapshot per Micro-chunk (removal = one message).
    pub remove_microchunks: Vec<IVec3>,
    /// Voxels to insert (world edit), voiced as fresh snapshots of the
    /// affected Micro-chunks.
    pub add_voxels: Vec<(IVec3, u8)>,
}

/// The edit-at-the-seam script (one step per frame after the first).
#[derive(Clone, Debug)]
pub struct EditScript {
    pub steps: Vec<EditStep>,
}

/// The edge-case worlds + palette-zero + solid-cube + edit-seam + the smoke
/// world.
pub fn all_worlds() -> Vec<WorldSpec> {
    let mut worlds = test_suite();
    worlds.push(smoke_world());
    worlds
}

/// The edge-case worlds + palette-zero + solid-cube + edit-seam + the smoke
/// world. One data entry per world (the body is declarative specs, not
/// logic, so the line budget is allowed).
#[allow(clippy::too_many_lines)]
pub fn test_suite() -> Vec<WorldSpec> {
    vec![
        // The depth-sorting regression (validation gap): nuke.vox from the
        // app's default camera pitched to the clamp (-π/2 + 0.01 = looking
        // straight down) — a large dense multi-Region world viewed along a
        // near-vertical axis. The former `TerminateOnFirstHit` traversal
        // committed the first hit found in BVH order, which is not the
        // closest: stacked Regions along the ray let a farther surface win
        // stably (~2400 hard mismatches before the fix — see
        // shaders/region/production.rgen). The reference tracer is
        // closest-hit, so this world guards the ray flags.
        WorldSpec {
            name: "nuke-down".to_string(),
            path: "assets/nuke.vox".to_string(),
            description: "nuke.vox, the app's default camera looking straight down (pitch clamped at -pi/2 + 0.01) — dense multi-Region depth-sorting regression".to_string(),
            camera: Some(CameraSpec {
                eye: [-16.0, 620.0, -16.0],
                target: [-16.0, 619.0, -16.01],
                up: [0.0, 0.01, -1.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "single".to_string(),
            path: "assets/test/single.vox".to_string(),
            description: "a single voxel at the origin (known-good minimal case)".to_string(),
            camera: Some(CameraSpec {
                eye: [-6.0, 2.0, 6.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "palette-zero".to_string(),
            path: "assets/test/palette-zero.vox".to_string(),
            description: "voxels with material index 0 — palette index 0 is a real color, and the 8-bit hitKind must carry it (no sentinel material)".to_string(),
            camera: Some(CameraSpec {
                eye: [-6.0, 2.0, 6.0],
                target: [0.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "hollow-box".to_string(),
            path: "assets/test/hollow-box.vox".to_string(),
            description: "a hollow 12x12x12 box, empty interior (known-good case)".to_string(),
            camera: Some(CameraSpec {
                eye: [-16.0, 5.0, 16.0],
                target: [5.5, 5.5, 5.5],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "solid-cube".to_string(),
            path: "assets/test/solid-cube.vox".to_string(),
            description: "a solid 12x12x12 cube with the camera inside — interior voxels (no exposed face) render through the DDA, not a hollow shell; every ray commits the enclosing voxel at t_min".to_string(),
            camera: Some(CameraSpec {
                eye: [6.5, 6.5, 6.5],
                target: [7.0, 6.5, 6.5],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "hull-empty".to_string(),
            path: "assets/test/hull-empty.vox".to_string(),
            description: "two voxels with empty space between (empty space is background, no ghost geometry)".to_string(),
            camera: Some(CameraSpec {
                eye: [20.0, 4.0, 9.0],
                target: [20.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "boundaries".to_string(),
            path: "assets/test/boundaries.vox".to_string(),
            description: "voxels at Micro-chunk (8), Chunk (64) and Region (256) boundaries".to_string(),
            camera: Some(CameraSpec {
                eye: [-40.0, 1.0, 10.0],
                target: [150.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "camera-in-voxel".to_string(),
            path: "assets/test/camera-in-voxel.vox".to_string(),
            description: "camera inside a solid voxel — the DDA commits the enclosing voxel at t_min (the old triangle-per-voxel driver reported the AABB exit t; the DDA path fixes it)".to_string(),
            camera: Some(CameraSpec {
                eye: [0.2, 0.2, 0.2],
                target: [1.0, 0.2, 0.2],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "far-miss".to_string(),
            path: "assets/test/far-miss.vox".to_string(),
            description: "camera looking away from the only voxel (everything is background)".to_string(),
            camera: Some(CameraSpec {
                eye: [0.0, 0.0, 12.0],
                target: [0.0, 0.0, 30.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "edit-seam".to_string(),
            path: "assets/test/edit-seam.vox".to_string(),
            description: "three voxels in a line; after the first frame the middle Micro-chunk (8,0,0) is removed through the input contract (zero-mask snapshot) and the second frame must match the edited world".to_string(),
            camera: Some(CameraSpec {
                eye: [-14.0, 1.0, 6.0],
                target: [10.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: Some(EditScript {
                steps: vec![EditStep {
                    label: "-after-edit".to_string(),
                    remove_microchunks: vec![IVec3::new(8, 0, 0)],
                    add_voxels: vec![],
                }],
            }),
        },
        WorldSpec {
            name: "multi-region".to_string(),
            path: "assets/test/multi-region.vox".to_string(),
            description: "a 3x3 cluster per Region along +x (Regions (-1,0,0)..(3,0,0)) — negative and multi-Region occupancy through the full lattice".to_string(),
            camera: Some(CameraSpec {
                eye: [-160.0, 25.0, 60.0],
                target: [380.0, 0.0, 15.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: None,
        },
        WorldSpec {
            name: "residency".to_string(),
            path: "assets/test/residency.vox".to_string(),
            description: "a 2x2x2 cube per Region along +x (Regions (0,0,0)..(2,0,0)); after the first frame Region (1,0,0) is emptied through the contract (left residency), then re-populated (became resident again) — both frames must match the reference".to_string(),
            camera: Some(CameraSpec {
                eye: [-90.0, 25.0, 45.0],
                target: [256.0, 1.0, 1.0],
                up: [0.0, 1.0, 0.0],
            }),
            edit: Some(EditScript {
                steps: vec![
                    EditStep {
                        label: "-after-empty".to_string(),
                        remove_microchunks: vec![IVec3::new(256, 0, 0)],
                        add_voxels: vec![],
                    },
                    EditStep {
                        label: "-after-repop".to_string(),
                        remove_microchunks: vec![],
                        add_voxels: vec![(256, 0, 0), (257, 0, 0), (256, 1, 0), (257, 1, 0)]
                            .into_iter()
                            .flat_map(|(x, y, z)| {
                                [(IVec3::new(x, y, z), 2), (IVec3::new(x, y, z + 1), 2)]
                            })
                            .collect(),
                    },
                ],
            }),
        },
    ]
}

pub fn smoke_world() -> WorldSpec {
    WorldSpec {
        name: "custom".to_string(),
        path: "assets/custom.vox".to_string(),
        description: "custom.vox — the smoke world (camera frames the world's bounding box)"
            .to_string(),
        camera: None,
        edit: None,
    }
}

/// The shared 256-color test palette. Index 0 is a real color (per
/// CONTEXT.md) and is exercised as a voxel material by the palette-zero
/// world; the low indices used by the test worlds are distinct, non-black
/// colors so background black stays distinct.
pub fn test_palette() -> [Color; 256] {
    std::array::from_fn(|i| Color {
        r: ((i * 37 + 11) % 256) as u8,
        g: ((i * 71 + 23) % 256) as u8,
        b: ((i * 109 + 41) % 256) as u8,
        a: 255,
    })
}

/// Regenerates every test world in assets/test/ (idempotent). Returns the
/// written paths.
pub fn generate_all(out_dir: &Path) -> io::Result<Vec<PathBuf>> {
    fs::create_dir_all(out_dir)?;

    let worlds: Vec<(&str, DotVoxData)> = vec![
        ("single", single_world()),
        ("palette-zero", palette_zero_world()),
        ("hollow-box", hollow_box_world()),
        ("solid-cube", solid_cube_world()),
        ("hull-empty", hull_empty_world()),
        ("boundaries", boundaries_world()),
        ("camera-in-voxel", camera_in_voxel_world()),
        ("far-miss", far_miss_world()),
        ("edit-seam", edit_seam_world()),
        ("multi-region", multi_region_world()),
        ("residency", residency_world()),
    ];

    let mut written = Vec::new();
    for (name, data) in worlds {
        let path = out_dir.join(format!("{name}.vox"));
        let mut file = fs::File::create(&path)?;
        data.write_vox(&mut file)?;
        written.push(path);
    }

    Ok(written)
}

// ---------------------------------------------------------------------------
// World data
// ---------------------------------------------------------------------------

/// A scene-less .vox world: one model, voxel (x, y, z) → world (x, z, y).
pub(crate) fn scene_less_world(voxels: &[(IVec3, u8)]) -> DotVoxData {
    let max = voxels
        .iter()
        .map(|(p, _)| *p)
        .fold(IVec3::ZERO, |acc, p| acc.max(p));

    let model = Model {
        size: Size {
            x: max.x as u32 + 1,
            y: max.y as u32 + 1,
            z: max.z as u32 + 1,
        },
        voxels: voxels
            .iter()
            .map(|(p, i)| Voxel {
                x: p.x as u8,
                y: p.z as u8,
                z: p.y as u8,
                i: *i,
            })
            .collect(),
    };

    base_data(vec![model], vec![])
}

/// A scene-graph .vox world: one 1-voxel model per (world position, material),
/// each placed through an unrotated transform node. An unrotated 1-voxel model
/// with frame translation t lands at world (t.x, t.z, t.y + 1) — verified by
/// `scene_graph_placement` below. Used to reach positions beyond u8 (the
/// Region boundary at x = 256).
fn transformed_world(voxels: &[(IVec3, u8)]) -> DotVoxData {
    let n = voxels.len();

    let models: Vec<Model> = voxels
        .iter()
        .map(|(_, i)| Model {
            size: Size { x: 1, y: 1, z: 1 },
            voxels: vec![Voxel {
                x: 0,
                y: 0,
                z: 0,
                i: *i,
            }],
        })
        .collect();

    let mut scenes = vec![SceneNode::Group {
        attributes: Dict::new(),
        children: (1..=n as u32).collect(),
    }];

    for (index, (world_position, _)) in voxels.iter().enumerate() {
        // Invert world (wx, wy, wz) = (t.x, t.z, t.y + 1).
        let t = IVec3::new(world_position.x, world_position.z - 1, world_position.y);
        let mut attributes = Dict::new();
        attributes.insert("_t".to_string(), format!("{} {} {}", t.x, t.y, t.z));

        scenes.push(SceneNode::Transform {
            attributes: Dict::new(),
            frames: vec![Frame::new(attributes)],
            child: (n as u32 + 1 + index as u32),
            layer_id: 0,
        });
    }

    for index in 0..n {
        scenes.push(SceneNode::Shape {
            attributes: Dict::new(),
            models: vec![ShapeModel {
                model_id: index as u32,
                attributes: Dict::new(),
            }],
        });
    }

    base_data(models, scenes)
}

fn base_data(models: Vec<Model>, scenes: Vec<SceneNode>) -> DotVoxData {
    DotVoxData {
        version: 150,
        index_map: (0..=255).collect(),
        models,
        palette: test_palette().to_vec(),
        materials: vec![],
        scenes,
        layers: vec![],
    }
}

fn single_world() -> DotVoxData {
    scene_less_world(&[(IVec3::new(0, 0, 0), 1)])
}

/// Voxels whose material index is 0: palette index 0 is a real color (the
/// Occupancy mask — not a sentinel material — defines existence), and the
/// 8-bit hitKind must carry it through the intersection shader.
fn palette_zero_world() -> DotVoxData {
    scene_less_world(&[(IVec3::new(0, 0, 0), 0), (IVec3::new(3, 0, 0), 0)])
}

/// A solid 12x12x12 cube with the camera inside: every interior voxel has no
/// exposed face, so a surface-only path renders this frame black; the DDA
/// path commits the enclosing voxel at t_min for every ray.
fn solid_cube_world() -> DotVoxData {
    let mut voxels = Vec::new();
    for x in 0..12 {
        for y in 0..12 {
            for z in 0..12 {
                voxels.push((IVec3::new(x, y, z), 7));
            }
        }
    }
    scene_less_world(&voxels)
}

fn hollow_box_world() -> DotVoxData {
    // Shell of the box [0, 11]^3: voxels where any coordinate is on a face.
    let mut voxels = Vec::new();
    for x in 0..12 {
        for y in 0..12 {
            for z in 0..12 {
                if x == 0 || x == 11 || y == 0 || y == 11 || z == 0 || z == 11 {
                    voxels.push((IVec3::new(x, y, z), 2));
                }
            }
        }
    }
    scene_less_world(&voxels)
}

fn hull_empty_world() -> DotVoxData {
    scene_less_world(&[(IVec3::new(0, 0, 0), 1), (IVec3::new(40, 0, 0), 2)])
}

fn boundaries_world() -> DotVoxData {
    // Voxels straddling the renderer's lattice boundaries (all at y = z = 0):
    // Micro-chunk boundary 7/8, Chunk boundary 63/64, Region boundary 255/256.
    // x = 256 exceeds the .vox u8 coordinate range, so the whole world goes
    // through scene-graph transforms.
    transformed_world(&[
        (IVec3::new(7, 0, 0), 1),
        (IVec3::new(8, 0, 0), 2),
        (IVec3::new(63, 0, 0), 3),
        (IVec3::new(64, 0, 0), 4),
        (IVec3::new(255, 0, 0), 5),
        (IVec3::new(256, 0, 0), 6),
    ])
}

fn camera_in_voxel_world() -> DotVoxData {
    scene_less_world(&[(IVec3::new(0, 0, 0), 1)])
}

fn far_miss_world() -> DotVoxData {
    scene_less_world(&[(IVec3::new(0, 0, 0), 1)])
}

/// Three voxels in a line along +x, one per Micro-chunk (origin =
/// floor(voxel/8)*8): (0,0,0) in mc (0,0,0), (12,0,0) in mc (8,0,0),
/// (24,0,0) in mc (24,0,0). The validator empties the middle Micro-chunk
/// (8,0,0) after the first frame (ticket 03's edit-at-the-seam: a zero-mask
/// snapshot through the input contract) — the world survives (two voxels
/// left) and the removal is dead ahead of the camera, so the second frame
/// visibly differs.
fn edit_seam_world() -> DotVoxData {
    scene_less_world(&[
        (IVec3::new(0, 0, 0), 1),
        (IVec3::new(12, 0, 0), 2),
        (IVec3::new(24, 0, 0), 3),
    ])
}

/// A 3x3 cluster per Region along +x: Regions (-1,0,0), (0,0,0), (1,0,0),
/// (2,0,0), (3,0,0). The negative-coordinate cluster exercises floor-division
/// Region derivation end-to-end (the renderer derives every Region id from
/// global coords); the whole world renders as one batch over 5 resident
/// Regions.
fn multi_region_world() -> DotVoxData {
    let mut voxels = Vec::new();
    for (material, region_x) in (1u8..).zip([-1, 0, 1, 2, 3]) {
        for dx in 0..3 {
            for dz in 0..3 {
                voxels.push((IVec3::new(region_x * 256 + dx, 0, dz), material));
            }
        }
    }
    transformed_world(&voxels)
}

/// A 2x2x2 cube per Region along +x: Regions (0,0,0), (1,0,0), (2,0,0). The
/// validator empties Region (1,0,0)'s Micro-chunk (256,0,0) after the first
/// frame — its last voxel leaves residency — and re-populates it after the
/// second (re-created: BLAS + pool buffer, ideally reused from the free
/// lists). Each frame must match the reference over the world at that point.
fn residency_world() -> DotVoxData {
    let mut voxels = Vec::new();
    for (region_x, material) in [(0, 1), (1, 2), (2, 3)] {
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    voxels.push((IVec3::new(region_x * 256 + dx, dy, dz), material));
                }
            }
        }
    }
    transformed_world(&voxels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{chunk::Chunks, voxel::open_file};

    /// Writes a world to a unique temp file (tests run in parallel), loads it
    /// through the real loader, and returns the world.
    fn load(world: DotVoxData) -> Chunks {
        use std::sync::atomic::{AtomicU32, Ordering};

        static COUNTER: AtomicU32 = AtomicU32::new(0);

        let dir = std::env::temp_dir().join("atlas-rt-test-worlds");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "test-{}-{}.vox",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = fs::File::create(&path).unwrap();
        world.write_vox(&mut file).unwrap();
        let data = open_file(path.to_str().unwrap());
        assert_eq!(data.palette.len(), 256);
        Chunks::new(&data)
    }

    #[test]
    fn scene_less_placement() {
        // The authoring helper swaps y/z, and the loader swaps them back, so
        // a voxel authored for world (3, 4, 5) lands exactly there.
        let world = load(scene_less_world(&[(IVec3::new(3, 4, 5), 1)]));
        assert!(world.get_voxel(&IVec3::new(3, 4, 5)).is_some());
        assert!(world.get_voxel(&IVec3::new(3, 5, 4)).is_none());
    }

    #[test]
    fn scene_graph_placement() {
        // Frame translation t lands a 1-voxel model at world (t.x, t.z, t.y+1).
        let world = load(transformed_world(&[(IVec3::new(256, 0, 0), 6)]));
        let voxel = world.get_voxel(&IVec3::new(256, 0, 0));
        assert!(voxel.is_some(), "expected voxel at (256, 0, 0)");
        assert_eq!(voxel.unwrap().material_index(), 6);
    }

    #[test]
    fn boundaries_world_loads_at_expected_positions() {
        let world = load(boundaries_world());
        for x in [7, 8, 63, 64, 255, 256] {
            assert!(
                world.get_voxel(&IVec3::new(x, 0, 0)).is_some(),
                "missing boundary voxel at x = {x}"
            );
        }
        assert_eq!(world.voxel_count(), 6);
    }

    #[test]
    fn every_test_world_writes_and_loads() {
        let dir = std::env::temp_dir().join("atlas-rt-test-worlds");
        let paths = generate_all(&dir).unwrap();
        assert_eq!(paths.len(), 11);

        for path in &paths {
            let data = open_file(path.to_str().unwrap());
            assert_eq!(data.palette.len(), 256, "palette must be 256 entries");
            let world = Chunks::new(&data);
            assert!(
                world.voxel_count() > 0,
                "{} loaded as empty",
                path.display()
            );
        }
    }

    #[test]
    fn test_palette_index_zero_is_a_real_color() {
        let palette = test_palette();
        assert_eq!(palette[0].a, 255);
    }
}
