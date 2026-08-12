pub fn open_file(path: &str) -> dot_vox::DotVoxData {
    let vox_data = dot_vox::load(path).unwrap();

    #[cfg(debug_assertions)]
    assert!(vox_data.palette.len() <= 256);

    vox_data
}

pub fn get_palette(data: &dot_vox::DotVoxData) -> [glam::Vec4; 256] {
    let mut array = [glam::Vec4::ZERO; 256];
    for (i, value) in array.iter_mut().enumerate() {
        let color = data.palette.get(i).unwrap();
        *value = glam::Vec4::new(
            f32::from(color.r) / 255.0,
            f32::from(color.g) / 255.0,
            f32::from(color.b) / 255.0,
            f32::from(color.a) / 255.0,
        )
    }

    array
}
