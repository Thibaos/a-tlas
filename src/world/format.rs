//! The .vox file format: loading (`open_file`) and the palette. Materials
//! live in `material`.

/// Loads a .vox file (the world's committed format).
pub fn open_file(path: &str) -> dot_vox::DotVoxData {
    let vox_data = dot_vox::load(path).unwrap();

    #[cfg(debug_assertions)]
    assert!(vox_data.palette.len() <= 256);

    vox_data
}

/// The palette: one linear RGBA per index (0–255), the reference tracer's
/// and material table's color source.
pub fn get_palette(data: &dot_vox::DotVoxData) -> [glam::Vec4; 256] {
    let mut array = [glam::Vec4::ZERO; 256];

    for (i, color) in data.palette.iter().enumerate() {
        array[i] = glam::Vec4::new(
            f32::from(color.r) / 255.0,
            f32::from(color.g) / 255.0,
            f32::from(color.b) / 255.0,
            f32::from(color.a) / 255.0,
        );
    }

    array
}
