use glam::Vec3;

pub fn sample_uniform_sphere(radius: f32) -> (f32, f32, f32) {
    let sample = Vec3::new(
        rand::random_range(-1.0..=1.0),
        rand::random_range(-1.0..=1.0),
        rand::random_range(-1.0..=1.0),
    )
    .normalize()
        * rand::random_range(0.0..=1.0)
        * radius;

    (sample.x.floor(), sample.y.floor(), sample.z.floor())
}
