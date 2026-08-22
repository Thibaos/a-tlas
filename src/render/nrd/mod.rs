//! NRD ReBLUR denoiser: instance lifecycle (pools, pipelines, descriptor
//! machinery) and the per-frame recording of the dispatch list the library
//! returns. The library makes no GPU calls itself; this module executes its
//! compute dispatches on the graphics flight between the trace pass and the
//! composite.

pub mod sys;

use core::ffi::CStr;
use core::slice;
use std::sync::Arc;

use vulkano::{
    buffer::Buffer,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType},
        sys::RawDescriptorSet,
        DescriptorBufferInfo, DescriptorImageInfo, WriteDescriptorSet,
    },
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::PipelineLayoutCreateInfo,
        ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
};

use crate::{core::gpu::GpuStack, render::region::task::{RegionRenderContext, RenderMode}};

const CONSTANTS_SLOTS: u64 = 64;

pub struct PoolTexture {
    pub id: Id<Image>,
    pub view: Arc<ImageView>,
}

pub struct NrdInputs {
    pub diff_radiance: Arc<ImageView>,
    pub spec_radiance: Arc<ImageView>,
    pub normal_roughness: Arc<ImageView>,
    pub viewz: Arc<ImageView>,
    pub mv: Arc<ImageView>,
    pub diff_out: Arc<ImageView>,
    pub spec_out: Arc<ImageView>,
}

struct PipelineData {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    texture_layout: Arc<DescriptorSetLayout>,
    constants_layout: Arc<DescriptorSetLayout>,
}

pub struct NrdInstance {
    instance: *mut sys::Instance,
    identifier: sys::Identifier,
    width: u32,
    height: u32,
    texture_offset: u32,
    storage_offset: u32,
    constants_offset: u32,

    pipelines: Vec<PipelineData>,
    set_allocator: Arc<StandardDescriptorSetAllocator>,
    // Immutable samplers are cloned into the constants set layouts; the Vec
    // only documents ownership.
    #[allow(dead_code)]
    samplers: Vec<Arc<Sampler>>,

    permanent_pool: Vec<PoolTexture>,
    transient_pool: Vec<PoolTexture>,

    pub constants_buffer_id: Id<Buffer>,
    constants_buffer: Arc<Buffer>,
    constants_stride: u64,
    constants_max_data_size: u64,
}

// The instance lives for the app's lifetime (or until a resize swap) and is
// only dereferenced from taskgraph task execution, which runs on one thread.
unsafe impl Send for NrdInstance {}

impl Drop for NrdInstance {
    fn drop(&mut self) {
        if !self.instance.is_null() {
            unsafe { sys::DestroyInstance(self.instance) };
        }
    }
}

fn map_format(format: u32) -> vulkano::format::Format {
    use vulkano::format::Format;

    match format {
        sys::format::RGBA8_UNORM => Format::R8G8B8A8_UNORM,
        sys::format::R16_SFLOAT => Format::R16_SFLOAT,
        sys::format::RG16_SFLOAT => Format::R16G16_SFLOAT,
        sys::format::RGBA16_SFLOAT => Format::R16G16B16A16_SFLOAT,
        sys::format::R32_SFLOAT => Format::R32_SFLOAT,
        sys::format::R10_G10_B10_A2_UNORM => Format::A2B10G10R10_UNORM_PACK32,
        sys::format::R11_G11_B10_UFLOAT => Format::B10G11R11_UFLOAT_PACK32,
        _ => Format::R16G16B16A16_SFLOAT,
    }
}

impl NrdInstance {
    pub fn new(gpu: &GpuStack, width: u32, height: u32) -> Result<Self, String> {
        let library = unsafe { sys::GetLibraryDesc().as_ref() }.ok_or("NRD: no library desc")?;
        let offsets = library.spirv_binding_offsets;

        let denoisers = [sys::DenoiserDesc {
            identifier: 0,
            denoiser: sys::denoiser::REBLUR_DIFFUSE_SPECULAR,
        }];
        let creation = sys::InstanceCreationDesc {
            denoisers: denoisers.as_ptr(),
            denoisers_num: 1,
            allocation_callbacks: [core::ptr::null_mut(); 4],
        };

        let mut instance: *mut sys::Instance = core::ptr::null_mut();
        let result = unsafe { sys::CreateInstance(&creation, &mut instance) };
        if result != sys::result::SUCCESS || instance.is_null() {
            return Err(format!("NRD: CreateInstance failed ({result})"));
        }

        let desc = unsafe { sys::GetInstanceDesc(instance).as_ref() }.ok_or("NRD: no instance desc")?;

        let sampler = |filter: Filter| -> Result<Arc<Sampler>, String> {
            Sampler::new(
                &gpu.device,
                &SamplerCreateInfo {
                    mag_filter: filter,
                    min_filter: filter,
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    ..SamplerCreateInfo::default()
                },
            )
            .map_err(|e| format!("NRD: sampler: {e}"))
        };
        let samplers = vec![sampler(Filter::Nearest)?, sampler(Filter::Linear)?];

        let pool_texture = |texture: &sys::TextureDesc| -> Result<PoolTexture, String> {
            let factor = u32::from(texture.downsample_factor.max(1));
            let extent = [width.div_ceil(factor).max(1), height.div_ceil(factor).max(1), 1];
            let id = gpu
                .resources
                .create_image(
                    &ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: map_format(texture.format),
                        extent,
                        usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                        initial_layout: ImageLayout::General,
                        ..Default::default()
                    },
                    &AllocationCreateInfo::default(),
                )
                .map_err(|e| format!("NRD: pool image: {e}"))?;
            let image = gpu.resources.image(id).image().clone();
            let view = ImageView::new_default(&image).map_err(|e| format!("NRD: pool view: {e}"))?;

            Ok(PoolTexture { id, view })
        };

        let permanent_pool =
            unsafe { slice::from_raw_parts(desc.permanent_pool, desc.permanent_pool_size as usize) }
                .iter()
                .map(&pool_texture)
                .collect::<Result<Vec<_>, String>>()?;
        let transient_pool =
            unsafe { slice::from_raw_parts(desc.transient_pool, desc.transient_pool_size as usize) }
                .iter()
                .map(&pool_texture)
                .collect::<Result<Vec<_>, String>>()?;

        let set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &gpu.device,
            &Default::default(),
        ));

        let constants_max_data_size = u64::from(desc.constant_buffer_max_data_size.max(16));
        let constants_stride = constants_max_data_size.max(256);
        let constants_size = constants_stride
            .checked_mul(CONSTANTS_SLOTS)
            .ok_or("NRD: constants size overflow")?;
        let constants_buffer_id = gpu
            .resources
            .create_buffer(
                &vulkano::buffer::BufferCreateInfo {
                    usage: vulkano::buffer::BufferUsage::UNIFORM_BUFFER
                        | vulkano::buffer::BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                &Default::default(),
                DeviceLayout::new_unsized::<[u8]>(constants_size).ok_or("NRD: constants layout")?,
            )
            .map_err(|e| format!("NRD: constants buffer: {e}"))?;
        let constants_buffer = gpu.resources.buffer(constants_buffer_id).buffer().clone();

        let entry_point_name = unsafe { CStr::from_ptr(desc.shader_entry_point.cast()) };

        let pipelines =
            unsafe { slice::from_raw_parts(desc.pipelines, desc.pipelines_num as usize) }
                .iter()
                .map(|pipeline_desc| build_pipeline(gpu, &samplers, &offsets, pipeline_desc, entry_point_name))
                .collect::<Result<Vec<_>, String>>()?;

        let settings = sys::ReblurSettings::default();
        let result = unsafe {
            sys::SetDenoiserSettings(instance, denoisers[0].identifier, core::ptr::from_ref(&settings).cast())
        };
        if result != sys::result::SUCCESS {
            return Err(format!("NRD: SetDenoiserSettings failed ({result})"));
        }

        Ok(Self {
            instance,
            identifier: denoisers[0].identifier,
            width,
            height,
            texture_offset: offsets.texture_offset,
            storage_offset: offsets.storage_texture_and_buffer_offset,
            constants_offset: offsets.constant_buffer_offset,
            pipelines,
            set_allocator,
            samplers,
            permanent_pool,
            transient_pool,
            constants_buffer_id,
            constants_buffer,
            constants_stride,
            constants_max_data_size,
        })
    }

    pub fn extent(&self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// Pool textures plus the constants buffer, for deferred destruction by
    /// the owner when the instance is replaced (resize).
    pub fn resource_ids(&self) -> (Vec<Id<Image>>, Id<Buffer>) {
        let images = self
            .permanent_pool
            .iter()
            .chain(&self.transient_pool)
            .map(|texture| texture.id)
            .collect();
        (images, self.constants_buffer_id)
    }

    unsafe fn record(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        settings: &sys::CommonSettings,
        inputs: &NrdInputs,
    ) -> Result<(), String> {
        let result = unsafe { sys::SetCommonSettings(self.instance, core::ptr::from_ref(settings)) };
        if result != sys::result::SUCCESS {
            return Err(format!("NRD: SetCommonSettings failed ({result})"));
        }

        let mut dispatch_descs: *const sys::DispatchDesc = core::ptr::null();
        let mut dispatch_num: u32 = 0;
        let result = unsafe {
            sys::GetComputeDispatches(
                self.instance,
                core::ptr::from_ref(&self.identifier),
                1,
                &mut dispatch_descs,
                &mut dispatch_num,
            )
        };
        if result != sys::result::SUCCESS {
            return Err(format!("NRD: GetComputeDispatches failed ({result})"));
        }

        let dispatches = unsafe { slice::from_raw_parts(dispatch_descs, dispatch_num as usize) };

        for (index, dispatch) in dispatches.iter().enumerate() {
            if dispatch.constant_buffer_data_matches_previous_dispatch
                || dispatch.constant_buffer_data.is_null()
                || dispatch.constant_buffer_data_size == 0
            {
                continue;
            }

            let offset = self.constants_slot_offset(index)?;
            let size = u64::from(dispatch.constant_buffer_data_size);
            if size > self.constants_max_data_size {
                return Err("NRD: constant data exceeds slot".to_string());
            }

            let data = unsafe {
                slice::from_raw_parts(dispatch.constant_buffer_data, dispatch.constant_buffer_data_size as usize)
            }
            .to_vec();

            unsafe { cbf.update_buffer(self.constants_buffer_id, offset, data.as_slice()) };
        }

        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: &[MemoryBarrier {
                    src_access: vulkano::sync::AccessFlags::TRANSFER_WRITE,
                    dst_access: vulkano::sync::AccessFlags::SHADER_READ
                        | vulkano::sync::AccessFlags::SHADER_STORAGE_READ
                        | vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
                    src_stages: vulkano::sync::PipelineStages::ALL_TRANSFER,
                    dst_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER,
                    ..Default::default()
                }],
                ..Default::default()
            })
        };

        let mut deferred_sets: Vec<RawDescriptorSet> = Vec::new();

        for (index, dispatch) in dispatches.iter().enumerate() {
            let pipeline_data = self
                .pipelines
                .get(usize::from(dispatch.pipeline_index))
                .ok_or("NRD: pipeline index out of range")?;

            let resources =
                unsafe { slice::from_raw_parts(dispatch.resources, dispatch.resources_num as usize) };

            let mut views: Vec<Arc<ImageView>> = Vec::with_capacity(resources.len());
            let mut bindings: Vec<u32> = Vec::with_capacity(resources.len());
            let mut next_sampled = 0u32;
            let mut next_storage = 0u32;

            for resource in resources {
                views.push(self.view_for(resource, inputs)?);

                if resource.descriptor_type == sys::descriptor_type::STORAGE_TEXTURE {
                    bindings.push(self.storage_offset + next_storage);
                    next_storage += 1;
                } else {
                    bindings.push(self.texture_offset + next_sampled);
                    next_sampled += 1;
                }
            }

            let image_infos: Vec<DescriptorImageInfo> = views
                .iter()
                .map(|view| DescriptorImageInfo {
                    sampler: None,
                    image_view: Some(view),
                    image_layout: ImageLayout::General,
                })
                .collect();
            let texture_writes: Vec<WriteDescriptorSet> = bindings
                .iter()
                .zip(&image_infos)
                .map(|(binding, info)| WriteDescriptorSet::image(*binding, info))
                .collect();

            let texture_set =
                RawDescriptorSet::new(&self.set_allocator, &pipeline_data.texture_layout, 0)
                    .map_err(|e| format!("NRD: texture set: {e}"))?;
            unsafe { texture_set.update(&texture_writes, &[]) };

            let constants_info = DescriptorBufferInfo {
                buffer: Some(&self.constants_buffer),
                offset: self.constants_slot_offset(index)?,
                range: Some(self.constants_max_data_size),
            };
            let constants_set =
                RawDescriptorSet::new(&self.set_allocator, &pipeline_data.constants_layout, 0)
                    .map_err(|e| format!("NRD: constants set: {e}"))?;
            unsafe {
                constants_set.update(
                    &[WriteDescriptorSet::buffer(self.constants_offset, &constants_info)],
                    &[],
                )
            };

            unsafe {
                cbf.bind_pipeline(&pipeline_data.pipeline);
                cbf.as_raw().bind_descriptor_sets_unchecked(
                    PipelineBindPoint::Compute,
                    &pipeline_data.layout,
                    0,
                    &[&texture_set, &constants_set],
                    &[],
                );
                cbf.dispatch([
                    u32::from(dispatch.grid_width.max(1)),
                    u32::from(dispatch.grid_height.max(1)),
                    1,
                ]);
            }

            deferred_sets.push(texture_set);
            deferred_sets.push(constants_set);

            if index + 1 < dispatches.len() {
                unsafe {
                    cbf.pipeline_barrier(&DependencyInfo {
                        memory_barriers: &[MemoryBarrier {
                            src_access: vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
                            dst_access: vulkano::sync::AccessFlags::SHADER_READ
                                | vulkano::sync::AccessFlags::SHADER_STORAGE_READ
                                | vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
                            src_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER,
                            dst_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER,
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                };
            }
        }

        for set in deferred_sets {
            cbf.destroy_object(set);
        }

        Ok(())
    }

    fn view_for(
        &self,
        resource: &sys::ResourceDesc,
        inputs: &NrdInputs,
    ) -> Result<Arc<ImageView>, String> {
        match resource.kind {
            sys::resource_type::IN_DIFF_RADIANCE_HITDIST => Ok(inputs.diff_radiance.clone()),
            sys::resource_type::IN_SPEC_RADIANCE_HITDIST => Ok(inputs.spec_radiance.clone()),
            sys::resource_type::IN_NORMAL_ROUGHNESS => Ok(inputs.normal_roughness.clone()),
            sys::resource_type::IN_VIEWZ => Ok(inputs.viewz.clone()),
            sys::resource_type::IN_MV => Ok(inputs.mv.clone()),
            sys::resource_type::OUT_DIFF_RADIANCE_HITDIST => Ok(inputs.diff_out.clone()),
            sys::resource_type::OUT_SPEC_RADIANCE_HITDIST => Ok(inputs.spec_out.clone()),
            sys::resource_type::TRANSIENT_POOL | sys::resource_type::PERMANENT_POOL => self
                .pool_view(resource.kind == sys::resource_type::PERMANENT_POOL, resource.index_in_pool),
            _ => Err(format!("NRD: unhandled resource kind {}", resource.kind)),
        }
    }

    fn pool_view(&self, permanent: bool, index_in_pool: u16) -> Result<Arc<ImageView>, String> {
        let pool = if permanent { &self.permanent_pool } else { &self.transient_pool };

        pool.get(usize::from(index_in_pool))
            .map(|texture| texture.view.clone())
            .ok_or_else(|| format!("NRD: pool index {} out of range", index_in_pool))
    }

    fn constants_slot_offset(&self, index: usize) -> Result<u64, String> {
        let slot = u64::try_from(index).map_err(|_| "NRD: slot index overflow")?;

        slot.checked_mul(self.constants_stride)
            .filter(|offset| offset.checked_add(self.constants_max_data_size).is_some())
            .ok_or_else(|| "NRD: constants offset overflow".to_string())
    }
}

unsafe impl Sync for NrdInstance {}

fn build_pipeline(
    gpu: &GpuStack,
    samplers: &[Arc<Sampler>],
    offsets: &sys::SpirvBindingOffsets,
    pipeline_desc: &sys::PipelineDesc,
    entry_point_name: &CStr,
) -> Result<PipelineData, String> {
    let bytecode = unsafe {
        slice::from_raw_parts(
            pipeline_desc.compute_shader_spirv.bytecode.cast::<u8>(),
            pipeline_desc.compute_shader_spirv.size as usize,
        )
    };

    let entry_name = entry_point_name.to_str().map_err(|e| format!("NRD: entry point: {e}"))?;
    let module = unsafe {
        ShaderModule::new(
            &gpu.device,
            &ShaderModuleCreateInfo::new(bytemuck::cast_slice::<u8, u32>(bytecode)),
        )
    }
    .map_err(|e| format!("NRD: shader module: {e}"))?;
    let entry_point = module.entry_point(entry_name).ok_or("NRD: entry point not found")?;
    let stage = PipelineShaderStageCreateInfo::new(&entry_point);

    let ranges = unsafe {
        slice::from_raw_parts(
            pipeline_desc.resource_ranges,
            pipeline_desc.resource_ranges_num as usize,
        )
    };

    let mut texture_bindings: Vec<DescriptorSetLayoutBinding> = Vec::new();
    let mut next_sampled = offsets.texture_offset;
    let mut next_storage = offsets.storage_texture_and_buffer_offset;

    for range in ranges {
        let storage = range.descriptor_type == sys::descriptor_type::STORAGE_TEXTURE;
        let descriptor_type = if storage {
            DescriptorType::StorageImage
        } else {
            DescriptorType::SampledImage
        };
        let binding = if storage { next_storage } else { next_sampled };

        texture_bindings.push(DescriptorSetLayoutBinding {
            binding,
            descriptor_count: range.descriptors_num,
            stages: ShaderStages::COMPUTE,
            ..DescriptorSetLayoutBinding::new(descriptor_type)
        });

        if storage {
            next_storage += range.descriptors_num;
        } else {
            next_sampled += range.descriptors_num;
        }
    }

    let texture_layout = DescriptorSetLayout::new(
        &gpu.device,
        &DescriptorSetLayoutCreateInfo {
            bindings: &texture_bindings,
            ..Default::default()
        },
    )
    .map_err(|e| format!("NRD: texture layout: {e}"))?;

    let sampler_refs: Vec<&Arc<Sampler>> = samplers.iter().collect();
    let mut constants_bindings: Vec<DescriptorSetLayoutBinding> = Vec::new();
    for (index, sampler) in sampler_refs.iter().enumerate() {
        constants_bindings.push(DescriptorSetLayoutBinding {
            binding: offsets.sampler_offset + index as u32,
            immutable_samplers: slice::from_ref(sampler),
            stages: ShaderStages::COMPUTE,
            ..DescriptorSetLayoutBinding::new(DescriptorType::Sampler)
        });
    }
    constants_bindings.push(DescriptorSetLayoutBinding {
        binding: offsets.constant_buffer_offset,
        stages: ShaderStages::COMPUTE,
        ..DescriptorSetLayoutBinding::new(DescriptorType::UniformBuffer)
    });

    let constants_layout = DescriptorSetLayout::new(
        &gpu.device,
        &DescriptorSetLayoutCreateInfo {
            bindings: &constants_bindings,
            ..Default::default()
        },
    )
    .map_err(|e| format!("NRD: constants layout: {e}"))?;

    let set_layouts: Vec<&Arc<DescriptorSetLayout>> = vec![&texture_layout, &constants_layout];
    let layout = PipelineLayout::new(
        &gpu.device,
        &PipelineLayoutCreateInfo {
            set_layouts: &set_layouts,
            ..Default::default()
        },
    )
    .map_err(|e| format!("NRD: pipeline layout: {e}"))?;

    let pipeline = ComputePipeline::new(
        &gpu.device,
        None,
        &ComputePipelineCreateInfo::new(stage, &layout),
    )
    .map_err(|e| format!("NRD: compute pipeline: {e}"))?;

    Ok(PipelineData {
        pipeline,
        layout,
        texture_layout,
        constants_layout,
    })
}

pub struct DenoiseTask {
    pub instance: Option<Arc<NrdInstance>>,
    pub inputs: Option<NrdInputs>,
}

impl DenoiseTask {
    pub fn new() -> Self {
        Self { instance: None, inputs: None }
    }
}

impl Default for DenoiseTask {
    fn default() -> Self {
        Self::new()
    }
}

impl Task for DenoiseTask {
    type World = RegionRenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let (Some(instance), Some(inputs)) = (&self.instance, &self.inputs) else {
            return Ok(());
        };

        if rcx.mode != RenderMode::Voxel {
            return Ok(());
        }

        let extent = instance.extent();
        let mut settings = sys::CommonSettings::new([
            u16::try_from(extent[0]).unwrap_or(u16::MAX),
            u16::try_from(extent[1]).unwrap_or(u16::MAX),
        ]);
        settings.view_to_clip_matrix = rcx.nrd.view_to_clip;
        settings.view_to_clip_matrix_prev = rcx.nrd.view_to_clip_prev;
        settings.world_to_view_matrix = rcx.nrd.world_to_view;
        settings.world_to_view_matrix_prev = rcx.nrd.world_to_view_prev;
        settings.frame_index = rcx.nrd.frame_index;
        settings.accumulation_mode = if rcx.nrd.clear {
            sys::accumulation_mode::CLEAR_AND_RESTART
        } else if rcx.nrd.reset {
            sys::accumulation_mode::RESTART
        } else {
            sys::accumulation_mode::CONTINUE
        };

        if let Err(error) = unsafe { instance.record(cbf, &settings, inputs) } {
            eprintln!("{error}");
        }

        Ok(())
    }
}
