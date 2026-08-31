use super::format::get_palette;

pub const DEFAULT_ROUGHNESS: f32 = 1.0;
pub const EMISSION_SCALE: f32 = 1000.0;
pub const MATFLAG_GLASS: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub emission: [f32; 3],
    pub glass: bool,
}

impl Material {
    fn default_for(albedo: [f32; 3]) -> Self {
        Self {
            albedo,
            metallic: 0.0,
            roughness: DEFAULT_ROUGHNESS,
            emission: [0.0; 3],
            glass: false,
        }
    }
}

pub type MaterialTable = [Material; 256];

pub fn get_material_table(data: &dot_vox::DotVoxData) -> MaterialTable {
    use super::format::srgb_to_linear;

    let palette = get_palette(data);

    let mut table: MaterialTable = std::array::from_fn(|i| {
        let color = palette[i];

        Material::default_for([
            srgb_to_linear(color.x),
            srgb_to_linear(color.y),
            srgb_to_linear(color.z),
        ])
    });

    for material in &data.materials {
        let id = material.id as usize;

        if id >= 256 {
            continue;
        }

        let entry = &mut table[id];

        if let Some(metallic) = material.metalness() {
            entry.metallic = metallic.clamp(0.0, 1.0);
        }

        // The editor writes an untouched `_rough: 0.1` into every palette slot;
        // roughness is authored data only for _type presets.
        if material.material_type().is_some() {
            if let Some(roughness) = material.roughness() {
                entry.roughness = roughness.clamp(0.0, 1.0);
            }
        }

        if material.material_type() == Some("_glass") {
            entry.glass = true;
        }

        if let Some(emit) = material.emission() {
            let emit = emit.clamp(0.0, 1.0);
            entry.emission = [
                entry.albedo[0] * emit * EMISSION_SCALE,
                entry.albedo[1] * emit * EMISSION_SCALE,
                entry.albedo[2] * emit * EMISSION_SCALE,
            ];
        }
    }

    table
}

#[cfg(test)]
mod tests {
    use crate::core::world::format::srgb_to_linear;

    use super::*;
    use dot_vox::{Color, DotVoxData, Material as VoxMaterial, Model, Size, Voxel};

    fn data_with(materials: Vec<VoxMaterial>) -> DotVoxData {
        DotVoxData {
            version: 150,
            index_map: (0..=255).collect(),
            models: vec![Model {
                size: Size { x: 1, y: 1, z: 1 },
                voxels: vec![Voxel {
                    x: 0,
                    y: 0,
                    z: 0,
                    i: 1,
                }],
            }],
            palette: (0..256)
                .map(|i| Color {
                    r: i as u8,
                    g: 0,
                    b: 0,
                    a: 255,
                })
                .collect(),
            materials,
            scenes: vec![],
            layers: vec![],
        }
    }

    fn matl(id: u32, props: &[(&str, &str)]) -> VoxMaterial {
        VoxMaterial {
            id,
            properties: props
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn palette_red(i: usize) -> f32 {
        srgb_to_linear(f32::from(i as u8) / 255.0)
    }

    #[test]
    fn defaults_and_albedo_match_palette() {
        let data = data_with(vec![]);
        let table = get_material_table(&data);
        let palette = get_palette(&data);

        for i in 0..256 {
            assert_eq!(
                table[i].albedo,
                [
                    srgb_to_linear(palette[i].x),
                    srgb_to_linear(palette[i].y),
                    srgb_to_linear(palette[i].z),
                ],
                "albedo must be the palette decoded to linear at index {i}"
            );
            assert_eq!(table[i].metallic, 0.0);
            assert_eq!(table[i].roughness, DEFAULT_ROUGHNESS);
            assert_eq!(table[i].emission, [0.0; 3]);
            assert!(!table[i].glass, "missing MATL degrades to opaque");
        }
    }

    #[test]
    fn matl_properties_override_defaults() {
        let data = data_with(vec![
            matl(
                3,
                &[("_type", "_metal"), ("_metal", "0.5"), ("_rough", "0.2")],
            ),
            matl(7, &[("_type", "_metal"), ("_rough", "0.9")]),
            matl(9, &[("_metal", "0.5"), ("_rough", "0.2")]),
        ]);
        let table = get_material_table(&data);

        assert_eq!(table[1].metallic, 0.0);
        assert_eq!(table[3].metallic, 0.5);
        assert_eq!(table[3].roughness, 0.2);
        assert_eq!(table[7].roughness, 0.9);
        assert_eq!(table[7].metallic, 0.0, "missing _metal keeps the default");
        assert_eq!(table[7].emission, [0.0; 3]);
        assert_eq!(
            table[9].metallic, 0.5,
            "_metal is authored even without a preset"
        );
        assert_eq!(
            table[9].roughness, DEFAULT_ROUGHNESS,
            "typeless _rough is the editor's untouched default, not authored data"
        );
    }

    #[test]
    fn glass_marker_sets_the_flag() {
        let data = data_with(vec![
            matl(5, &[("_type", "_glass"), ("_rough", "0.1")]),
            matl(6, &[("_type", "_glass")]),
            matl(8, &[("_type", "_metal")]),
        ]);
        let table = get_material_table(&data);

        assert!(table[5].glass);
        assert!(table[6].glass);

        assert!(!table[8].glass, "non-glass presets stay opaque");
        assert!(!table[1].glass, "missing MATL degrades to opaque");

        assert_eq!(table[5].roughness, 0.1, "glass keeps _rough flowing");
        assert_eq!(
            table[6].roughness, DEFAULT_ROUGHNESS,
            "glass without _rough keeps the default"
        );
    }

    #[test]
    fn emission_is_emit_times_albedo_times_scale() {
        let data = data_with(vec![
            matl(6, &[("_emit", "1.0")]),
            matl(7, &[("_emit", "0.25")]),
        ]);
        let table = get_material_table(&data);
        let palette = get_palette(&data);

        let scale = EMISSION_SCALE;
        for (i, emit) in [(6usize, 1.0f32), (7, 0.25)] {
            for channel in 0..3 {
                let expected = srgb_to_linear(palette[i][channel]) * emit * scale;
                assert!(
                    (table[i].emission[channel] - expected).abs() < 1e-6,
                    "emission[{i}][{channel}]: expected {expected}, got {}",
                    table[i].emission[channel]
                );
            }
        }

        assert_eq!(table[1].emission, [0.0; 3]);
    }

    #[test]
    fn properties_clamp_to_unit_range() {
        let data = data_with(vec![matl(
            4,
            &[
                ("_type", "_metal"),
                ("_metal", "2.0"),
                ("_rough", "-0.5"),
                ("_emit", "3.0"),
            ],
        )]);
        let table = get_material_table(&data);
        assert_eq!(table[4].metallic, 1.0);
        assert_eq!(table[4].roughness, 0.0);

        assert_eq!(table[4].emission[0], palette_red(4) * EMISSION_SCALE);
    }

    #[test]
    fn out_of_range_matl_id_is_skipped() {
        let data = data_with(vec![matl(300, &[("_metal", "1.0")])]);
        let table = get_material_table(&data);
        for entry in &table {
            assert_eq!(entry.metallic, 0.0);
        }
    }
}
