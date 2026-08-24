use super::path_tracer::{PathTracer, Scene, render_path};
use super::reference::CameraInputs;
use crate::world::World;
use crate::world::material::{MaterialTable, get_material_table};
use dot_vox::{Color, DotVoxData, Model, Size, Voxel};
use glam::Vec3;

fn scene() -> Scene {
    let sun_dir = Vec3::new(0.45, 0.8, 0.35).normalize();
    let cos_disk = (0.5_f32 * std::f32::consts::PI / 180.0).cos();
    let omega = 2.0 * std::f32::consts::PI * (1.0 - cos_disk);

    Scene {
        sun_dir,
        sky_knots: [0.15, 0.6, 1.2],
        e_sun: 16.0,
        cos_disk,
        l_disk: 16.0 / omega,
    }
}

fn world_floor() -> World {
    let mut world = World::default();

    for x in -2..=2 {
        for z in -2..=2 {
            world.insert_voxel_at(glam::IVec3::new(x, 0, z), 200);
        }
    }

    world
}

fn data_with_palette() -> DotVoxData {
    DotVoxData {
        version: 150,
        index_map: (0..=255).collect(),
        models: vec![Model {
            size: Size { x: 1, y: 1, z: 1 },
            voxels: vec![Voxel {
                x: 0,
                y: 0,
                z: 0,
                i: 200,
            }],
        }],
        palette: (0..256)
            .map(|i| {
                if i == 200 {
                    Color {
                        r: 200,
                        g: 30,
                        b: 30,
                        a: 255,
                    }
                } else {
                    Color {
                        r: 0,
                        g: 0,
                        b: 0,
                        a: 255,
                    }
                }
            })
            .collect(),
        materials: vec![],
        scenes: vec![],
        layers: vec![],
    }
}

// Mirrors shaders/composite.comp exactly.
fn composite(radiance: Vec3, ev: f32) -> Vec3 {
    let a = Vec3::splat(2.51);
    let b = Vec3::splat(0.03);
    let c = Vec3::splat(2.43);
    let d = Vec3::splat(0.59);
    let e = Vec3::splat(0.14);
    let x = radiance * ev.exp2();
    let mapped = (x * (a * x + b)) / (x * (c * x + d) + e);

    mapped.clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2)
}

fn hsv_saturation(c: Vec3) -> f32 {
    let mx = c.max_element();
    let mn = c.min_element();

    if mx <= 0.0 { 0.0 } else { (mx - mn) / mx }
}

/// A lit saturated voxel must render near its authored saturation: the
/// palette decodes sRGB to linear reflectance and the default Composite
/// exposure keeps diffuse highlights off the ACES shoulder.
#[test]
fn lit_voxel_keeps_authored_saturation() {
    let materials: MaterialTable = get_material_table(&data_with_palette());
    let world = world_floor();
    let tracer = PathTracer::new(&world, materials, scene());

    let eye = Vec3::new(2.5, 8.0, 2.5);
    let view = glam::camera::lh::view::look_to_mat4(eye, Vec3::NEG_Y, Vec3::Z);
    let proj = glam::camera::lh::proj::vulkan::perspective(
        std::f32::consts::FRAC_PI_2,
        1.0,
        0.01,
        10000.0,
    );
    let camera = CameraInputs::new(view, proj, 64, 64);
    let render = render_path(&tracer, &camera, 64, 64, 64);

    let radiance = render.display[32 * 64 + 32];
    let out = composite(radiance, super::super::region::task::default_ev());

    let sat = hsv_saturation(out);
    println!(
        "lit red (200,30,30): radiance {radiance:.3?} -> rgb {:.0?}, S = {sat:.3}",
        out * 255.0
    );

    // Authored bytes carry HSV S = 0.85; the ACES shoulder costs a little.
    assert!(
        sat > 0.75,
        "lit saturated voxel desaturated: S = {sat:.3} (authored 0.85)"
    );
}
