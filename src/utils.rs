pub fn sample_uniform_sphere(radius: f32) -> (f32, f32, f32) {
    let theta = rand::random_range(0.0..=2.0 * std::f32::consts::PI);
    let phi = (rand::random_range::<f32, _>(0.0..=1.0) * 2.0 - 1.0).acos();
    let r = rand::random_range::<f32, _>(0.0..=1.0).powf(1.0 / 3.0) * radius;

    let x = r * phi.sin() * theta.cos();
    let y = r * phi.sin() * theta.sin();
    let z = r * phi.cos();

    (x.floor(), y.floor(), z.floor())
}
