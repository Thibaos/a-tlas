pub mod acceleration_structure;

pub(crate) mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "shaders/rt/simple.rgen",
        vulkan_version: "1.3"
    }
}

pub(crate) mod intersection {
    vulkano_shaders::shader! {
        ty: "intersection",
        path: "shaders/rt/simple.rint",
        vulkan_version: "1.3"
    }
}

pub(crate) mod miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "shaders/rt/simple.rmiss",
        vulkan_version: "1.3"
    }
}

pub(crate) mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "shaders/rt/simple.rchit",
        vulkan_version: "1.3"
    }
}
