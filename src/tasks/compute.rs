use std::sync::Arc;

use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
    },
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult, command_buffer::RecordingCommandBuffer,
    descriptor_set::StorageBufferId, resource::HostAccessType,
};

use crate::{
    App, RenderContext,
    voxel::voxel_map::{VoxelMap, generate_tree},
};

pub mod tree64 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/compute/tree64.glsl",
        vulkan_version: "1.3"
    }
}

pub struct RendererTree64 {
    pub map: VoxelMap,
    // tree_scale: u32,
    // clusters: HashMap<u64, RenderClusterInfo>,
    pub swapchain_id: Id<Swapchain>,
    pub params_buffer_id: Id<Buffer>,
    params_storage_buffer_id: StorageBufferId,
    pub node_buffer_id: Id<Buffer>,
    nodes_storage_buffer_id: StorageBufferId,
    pub material_buffer_id: Id<Buffer>,
    materials_storage_buffer_id: StorageBufferId,
    pipeline: Arc<ComputePipeline>,
}

impl RendererTree64 {
    pub fn new(app: &App, virtual_swapchain_id: Id<Swapchain>, map: VoxelMap) -> Self {
        let bcx = app.resources.bindless_context().unwrap();

        let pipeline = {
            let shader = tree64::load(&app.device)
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(&shader);

            let layout = bcx
                .pipeline_layout_from_stages(std::slice::from_ref(&stage))
                .unwrap();

            ComputePipeline::new(
                &app.device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        let params_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_sized::<tree64::SceneParams>(),
            )
            .unwrap();

        let params_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                params_buffer_id,
                0,
                size_of::<tree64::SceneParams>() as DeviceSize,
            )
            .unwrap();

        let buffer_size = 32 * 32 * 32;

        let node_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[tree64::Node]>(buffer_size).unwrap(),
            )
            .unwrap();

        let nodes_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                node_buffer_id,
                0,
                (size_of::<tree64::Node>() * buffer_size as usize) as DeviceSize,
            )
            .unwrap();

        let material_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::new_unsized::<[tree64::Material]>(buffer_size).unwrap(),
            )
            .unwrap();

        let materials_storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                material_buffer_id,
                0,
                (size_of::<tree64::Material>() * buffer_size as usize) as DeviceSize,
            )
            .unwrap();

        app.resources
            .flight(app.graphics_flight_id)
            .unwrap()
            .wait_idle()
            .unwrap();

        Self {
            map,
            swapchain_id: virtual_swapchain_id,
            params_buffer_id,
            params_storage_buffer_id,
            node_buffer_id,
            nodes_storage_buffer_id,
            material_buffer_id,
            materials_storage_buffer_id,
            pipeline,
        }
    }

    pub fn sync_map(&mut self, app: &App) {
        self.map.dirty_locs.clear();

        let (nodes, _) = generate_tree(&mut self.map, 10);

        dbg!(nodes.len());

        unsafe {
            vulkano_taskgraph::execute(
                &app.transfer_queue,
                &app.resources,
                app.graphics_flight_id,
                |_cbf, tcx| {
                    let write_nodes_buffer =
                        tcx.write_buffer::<[tree64::Node]>(self.node_buffer_id, ..)?;

                    for (o, node) in write_nodes_buffer.iter_mut().zip(&nodes) {
                        let hi32 = (node.child_mask >> 32) as u32;
                        let lo32 = node.child_mask as u32;

                        *o = tree64::Node {
                            is_leaf: node.is_leaf as u32,
                            child_ptr: node.child_ptr,
                            child_mask_0: lo32,
                            child_mask_1: hi32,
                            // packed_data: [node.is_leaf as u32 | (node.child_ptr >> 1), lo32, hi32],
                        };
                    }

                    let write_materials_buffer =
                        tcx.write_buffer::<[tree64::Material]>(self.material_buffer_id, ..)?;

                    for (o, _) in write_materials_buffer.iter_mut().zip(nodes) {
                        *o = tree64::Material {
                            data: [rand::random::<u32>(), 0],
                        };
                    }

                    Ok(())
                },
                [
                    (self.node_buffer_id, HostAccessType::Write),
                    (self.material_buffer_id, HostAccessType::Write),
                ],
                [],
                [],
            )
        }
        .unwrap();
    }
}

impl Task for RendererTree64 {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.swapchain_id)?;
        let image_index = swapchain_state.current_image_index().unwrap();
        let extent = swapchain_state.images()[0].extent();

        unsafe { cbf.update_buffer(self.params_buffer_id, 0, &rcx.scene_params) }?;

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &tree64::PushConstants {
                    image_id: rcx.swapchain_storage_image_ids[image_index as usize],
                    scene_params_id: self.params_storage_buffer_id,
                    node_buffer_id: self.nodes_storage_buffer_id,
                    material_buffer_id: self.materials_storage_buffer_id,
                },
            )
        }?;

        unsafe {
            cbf.bind_pipeline_compute(&self.pipeline)?;
        }

        unsafe { cbf.dispatch([extent[0] / 8, extent[1] / 8, 1]) }?;

        Ok(())
    }
}
