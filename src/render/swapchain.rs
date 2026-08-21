use vulkano::{
    image::{ImageLayout, view::ImageView},
    swapchain::Swapchain,
};
use vulkano_taskgraph::{Id, descriptor_set::StorageImageId, resource::Resources};

pub fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
) -> Vec<StorageImageId> {
    let bcx = resources.bindless_context().unwrap();
    let swapchain_state = resources.swapchain(swapchain_id);
    let images = swapchain_state.images();

    images
        .iter()
        .map(|image| {
            let image_view = ImageView::new_default(image).unwrap();

            bcx.global_set()
                .add_storage_image(image_view, ImageLayout::General)
        })
        .collect()
}
