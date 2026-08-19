//! Materials: the per-palette-index surface properties and the CPU Material
//! table (ADR 0008). Loaded from the .vox MATL chunk (see `format`).

use super::format::get_palette;

/// The default roughness for a palette index with no MATL `_rough` property.
pub const DEFAULT_ROUGHNESS: f32 = 0.3;

/// The emission radiance scale: a fully emissive voxel (`_emit` 1.0) with
/// white albedo contributes `EMISSION_SCALE` linear radiance, bright enough
/// to read clearly through the ACES tonemap (and, later, to act as a real
/// path-hit light source). Tunable; the firefly-clamp policy stays in the
/// effort's fog (the map's "Firefly control" item).
pub const EMISSION_SCALE: f32 = 10.0;

/// The per-palette-index surface properties (CONTEXT.md: **Material**):
/// albedo (the Palette color), metallic, roughness, and emission (linear RGB
/// radiance). Loaded from the .vox MATL chunk; one Material per Palette
/// index (256 max). This is the CPU mirror, the single source of truth
/// whose packed twin the GPU reads (uploaded once at startup by
/// `RegionStore`, read by the DDA closest-hit and the production raygen) and
/// the table the validator's reference tracer shades with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// The Palette color for this index (linear RGB; == `get_palette`'s
    /// entry by construction. The byte-exact capture path depends on it).
    pub albedo: [f32; 3],
    /// `_metal`, clamped to [0, 1]; 0 = dielectric.
    pub metallic: f32,
    /// `_rough`, clamped to [0, 1]; 0.3 when the property is absent.
    pub roughness: f32,
    /// Emissive radiance, linear RGB: `_emit` × albedo × [`EMISSION_SCALE`]
    /// (0 for non-emissive materials).
    pub emission: [f32; 3],
}

impl Material {
    /// The default surface for a palette index with no MATL entry (or no
    /// property): diffuse, non-metallic, mid roughness, no emission.
    fn default_for(albedo: [f32; 3]) -> Self {
        Self {
            albedo,
            metallic: 0.0,
            roughness: DEFAULT_ROUGHNESS,
            emission: [0.0; 3],
        }
    }
}

/// The full Material table: one [`Material`] per palette index. The MATL
/// chunk's material id is the palette index, so the GPU table is indexed by
/// the 8-bit hitKind (the material index) directly.
pub type MaterialTable = [Material; 256];

/// Builds the CPU Material table from the .vox data (ADR 0008): every
/// palette index starts at the [`Material::default_for`] defaults, then the
/// MATL chunk's entries override the properties they name. VOX property
/// ranges are 0–1; out-of-range values clamp. `_type` is informational in
/// v1. All types keep the PBR triad (the map rules `glass` out of scope,
/// treated as opaque). Malformed material ids (≥ 256) are skipped, not
/// fatal: a bad MATL entry must not take down the loader.
pub fn get_material_table(data: &dot_vox::DotVoxData) -> MaterialTable {
    let palette = get_palette(data);
    let mut table: MaterialTable = std::array::from_fn(|i| {
        let color = palette[i];
        Material::default_for([color.x, color.y, color.z])
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
        if let Some(roughness) = material.roughness() {
            entry.roughness = roughness.clamp(0.0, 1.0);
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
    use super::*;
    use dot_vox::{Color, DotVoxData, Material as VoxMaterial, Model, Size, Voxel};

    /// A minimal scene-less .vox dataset with a red-ramp palette (index i →
    /// (i, 0, 0)) and the given MATL chunk.
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

    /// Every palette index starts from the defaults; the albedo column is
    /// the palette (the byte-exact capture path's invariant).
    #[test]
    fn defaults_and_albedo_match_palette() {
        let data = data_with(vec![]);
        let table = get_material_table(&data);
        let palette = get_palette(&data);

        for i in 0..256 {
            assert_eq!(
                table[i].albedo,
                [palette[i].x, palette[i].y, palette[i].z],
                "albedo must equal the palette at index {i}"
            );
            assert_eq!(table[i].metallic, 0.0);
            assert_eq!(table[i].roughness, DEFAULT_ROUGHNESS);
            assert_eq!(table[i].emission, [0.0; 3]);
        }
    }

    #[test]
    fn matl_properties_override_defaults() {
        let data = data_with(vec![
            matl(3, &[("_metal", "0.5"), ("_rough", "0.2")]),
            matl(7, &[("_rough", "0.9")]),
        ]);
        let table = get_material_table(&data);

        // Only the named indices changed.
        assert_eq!(table[1].metallic, 0.0);
        assert_eq!(table[3].metallic, 0.5);
        assert_eq!(table[3].roughness, 0.2);
        assert_eq!(table[7].roughness, 0.9);
        assert_eq!(table[7].metallic, 0.0, "missing _metal keeps the default");
        assert_eq!(table[7].emission, [0.0; 3]);
    }

    /// The emission mapping: linear RGB radiance = `_emit` × albedo ×
    /// EMISSION_SCALE.
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
                let expected = palette[i][channel] * emit * scale;
                assert!(
                    (table[i].emission[channel] - expected).abs() < 1e-6,
                    "emission[{i}][{channel}]: expected {expected}, got {}",
                    table[i].emission[channel]
                );
            }
        }
        // Non-emissive stays zero.
        assert_eq!(table[1].emission, [0.0; 3]);
    }

    #[test]
    fn properties_clamp_to_unit_range() {
        let data = data_with(vec![matl(
            4,
            &[("_metal", "2.0"), ("_rough", "-0.5"), ("_emit", "3.0")],
        )]);
        let table = get_material_table(&data);
        assert_eq!(table[4].metallic, 1.0);
        assert_eq!(table[4].roughness, 0.0);
        // _emit clamps to 1.0 before the albedo × scale mapping.
        assert_eq!(table[4].emission[0], palette_red(4) * EMISSION_SCALE);
    }

    /// A malformed MATL entry (id outside the palette) is skipped, not fatal.
    #[test]
    fn out_of_range_matl_id_is_skipped() {
        let data = data_with(vec![matl(300, &[("_metal", "1.0")])]);
        let table = get_material_table(&data);
        for entry in &table {
            assert_eq!(entry.metallic, 0.0);
        }
    }

    /// Helper: palette index i's red channel (the ramp is (i, 0, 0)).
    fn palette_red(i: usize) -> f32 {
        f32::from(i as u8) / 255.0
    }
}
