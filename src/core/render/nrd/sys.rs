//! Raw FFI bindings for the vendored NRD SDK (third_party/nrd, pinned v4.17.3),
//! hand-mirrored from Include/NRD.h + NRDDescs.h + NRDSettings.h. Struct
//! layouts are asserted against the C++ ABI probed at vendoring time.

// The full ABI is mirrored for documentation; not every constant is
// consumed by this integration.
#![allow(dead_code)]

use core::ffi::c_void;

pub const NRD_VERSION_MAJOR: u32 = 4;
pub const NRD_VERSION_MINOR: u32 = 17;
pub const NRD_VERSION_BUILD: u32 = 3;

#[repr(C)]
pub struct Instance {
    _opaque: [u8; 0],
}

pub type Identifier = u32;

pub mod result {
    pub const SUCCESS: u32 = 0;
    pub const FAILURE: u32 = 1;
    pub const INVALID_ARGUMENT: u32 = 2;
    pub const UNSUPPORTED: u32 = 3;
}

pub mod resource_type {
    pub const IN_MV: u32 = 0;
    pub const IN_NORMAL_ROUGHNESS: u32 = 1;
    pub const IN_VIEWZ: u32 = 2;
    pub const IN_DIFF_RADIANCE_HITDIST: u32 = 6;
    pub const IN_SPEC_RADIANCE_HITDIST: u32 = 7;
    pub const OUT_DIFF_RADIANCE_HITDIST: u32 = 18;
    pub const OUT_SPEC_RADIANCE_HITDIST: u32 = 19;
    pub const OUT_VALIDATION: u32 = 29;
    pub const TRANSIENT_POOL: u32 = 30;
    pub const PERMANENT_POOL: u32 = 31;
}

pub mod descriptor_type {
    pub const TEXTURE: u32 = 0;
    pub const STORAGE_TEXTURE: u32 = 1;
}

pub mod format {
    pub const R8_UNORM: u32 = 0;
    pub const RG8_UNORM: u32 = 4;
    pub const RGBA8_UNORM: u32 = 8;
    pub const R16_SFLOAT: u32 = 17;
    pub const RG16_SFLOAT: u32 = 22;
    pub const RGBA16_SFLOAT: u32 = 27;
    pub const R16_UINT: u32 = 15;
    pub const R32_SFLOAT: u32 = 30;
    pub const R32_UINT: u32 = 28;
    pub const R10_G10_B10_A2_UNORM: u32 = 40;
    pub const R11_G11_B10_UFLOAT: u32 = 42;
}

pub mod sampler {
    pub const NEAREST_CLAMP: u32 = 0;
    pub const LINEAR_CLAMP: u32 = 1;
}

pub mod accumulation_mode {
    pub const CONTINUE: u8 = 0;
    pub const RESTART: u8 = 1;
    pub const CLEAR_AND_RESTART: u8 = 2;
}

pub mod hit_distance_reconstruction_mode {
    pub const OFF: u8 = 0;
    pub const AREA_3X3: u8 = 1;
}

pub mod denoiser {
    pub use super::Identifier;
    pub const REBLUR_DIFFUSE_SPECULAR: Identifier = 6;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SpirvBindingOffsets {
    pub sampler_offset: u32,
    pub texture_offset: u32,
    pub constant_buffer_offset: u32,
    pub storage_texture_and_buffer_offset: u32,
}

#[repr(C)]
pub struct LibraryDesc {
    pub spirv_binding_offsets: SpirvBindingOffsets,
    pub supported_denoisers: *const Identifier,
    pub supported_denoisers_num: u32,
    pub version_major: u8,
    pub version_minor: u8,
    pub version_build: u8,
    pub normal_encoding: u8,
    pub roughness_encoding: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DenoiserDesc {
    pub identifier: Identifier,
    pub denoiser: Identifier,
}

#[repr(C)]
pub struct InstanceCreationDesc {
    pub allocation_callbacks: [*mut c_void; 4],
    pub denoisers: *const DenoiserDesc,
    pub denoisers_num: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TextureDesc {
    pub format: u32,
    pub downsample_factor: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResourceDesc {
    pub descriptor_type: u32,
    pub kind: u32,
    pub index_in_pool: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResourceRangeDesc {
    pub descriptor_type: u32,
    pub descriptors_num: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputeShaderDesc {
    pub bytecode: *const c_void,
    pub size: u64,
}

#[repr(C)]
pub struct PipelineDesc {
    pub compute_shader_dxbc: ComputeShaderDesc,
    pub compute_shader_dxil: ComputeShaderDesc,
    pub compute_shader_spirv: ComputeShaderDesc,
    pub resource_ranges: *const ResourceRangeDesc,
    pub resource_ranges_num: u32,
    pub has_constant_data: bool,
    pub shader_identifier: [u8; 256],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DescriptorPoolDesc {
    pub per_set_textures_max_num: u32,
    pub per_set_storage_textures_max_num: u32,
    pub total_textures_num: u32,
    pub total_storage_textures_num: u32,
    pub sets_max_num: u32,
}

#[repr(C)]
pub struct InstanceDesc {
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_register_index: u32,
    pub samplers_base_register_index: u32,
    pub resources_base_register_index: u32,
    pub constant_buffer_max_data_size: u32,
    pub samplers: *const u32,
    pub samplers_num: u32,
    pub shader_entry_point: *const u8,
    pub pipelines: *const PipelineDesc,
    pub pipelines_num: u32,
    pub permanent_pool: *const TextureDesc,
    pub permanent_pool_size: u32,
    pub transient_pool: *const TextureDesc,
    pub transient_pool_size: u32,
    pub descriptor_pool_desc: DescriptorPoolDesc,
}

#[repr(C)]
pub struct DispatchDesc {
    pub name: *const u8,
    pub identifier: Identifier,
    pub resources: *const ResourceDesc,
    pub resources_num: u32,
    pub constant_buffer_data: *const u8,
    pub constant_buffer_data_size: u32,
    pub constant_buffer_data_matches_previous_dispatch: bool,
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
}

#[repr(C)]
pub struct CommonSettings {
    pub view_to_clip_matrix: [f32; 16],
    pub view_to_clip_matrix_prev: [f32; 16],
    pub world_to_view_matrix: [f32; 16],
    pub world_to_view_matrix_prev: [f32; 16],
    pub world_prev_to_world_matrix: [f32; 16],
    pub motion_vector_scale: [f32; 3],
    pub camera_jitter: [f32; 2],
    pub camera_jitter_prev: [f32; 2],
    pub resource_size: [u16; 2],
    pub resource_size_prev: [u16; 2],
    pub rect_size: [u16; 2],
    pub rect_size_prev: [u16; 2],
    pub view_z_scale: f32,
    pub time_delta_between_frames: f32,
    pub denoising_range: f32,
    pub disocclusion_threshold: f32,
    pub disocclusion_threshold_alternate: f32,
    pub camera_attached_reflection_material_id: f32,
    pub strand_material_id: f32,
    pub history_fix_alternate_pixel_stride_material_id: f32,
    pub strand_thickness: f32,
    pub split_screen: f32,
    pub printf_at: [u16; 2],
    pub debug: f32,
    pub rect_origin: [u32; 2],
    pub frame_index: u32,
    pub accumulation_mode: u8,
    pub is_motion_vector_in_world_space: bool,
    pub is_history_confidence_available: bool,
    pub is_disocclusion_threshold_mix_available: bool,
    pub enable_validation: bool,
}

impl CommonSettings {
    pub fn new(rect_size: [u16; 2]) -> Self {
        Self {
            view_to_clip_matrix: [0.0; 16],
            view_to_clip_matrix_prev: [0.0; 16],
            world_to_view_matrix: [0.0; 16],
            world_to_view_matrix_prev: [0.0; 16],
            world_prev_to_world_matrix: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            motion_vector_scale: [1.0, 1.0, 1.0],
            camera_jitter: [0.0; 2],
            camera_jitter_prev: [0.0; 2],
            resource_size: rect_size,
            resource_size_prev: rect_size,
            rect_size,
            rect_size_prev: rect_size,
            view_z_scale: 1.0,
            time_delta_between_frames: 0.0,
            denoising_range: 500_000.0,
            disocclusion_threshold: 0.01,
            disocclusion_threshold_alternate: 0.05,
            camera_attached_reflection_material_id: 999.0,
            strand_material_id: 999.0,
            history_fix_alternate_pixel_stride_material_id: 999.0,
            strand_thickness: 80.0e-6,
            split_screen: 0.0,
            printf_at: [9999, 9999],
            debug: 0.0,
            rect_origin: [0; 2],
            frame_index: 0,
            accumulation_mode: accumulation_mode::CONTINUE,
            is_motion_vector_in_world_space: false,
            is_history_confidence_available: false,
            is_disocclusion_threshold_mix_available: false,
            enable_validation: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReblurSettings {
    pub hit_distance_parameters_a: f32,
    pub hit_distance_parameters_b: f32,
    pub hit_distance_parameters_c: f32,
    pub antilag_luminance_sigma_scale: f32,
    pub antilag_luminance_sensitivity: f32,
    pub responsive_accumulation_roughness_threshold: f32,
    pub responsive_accumulation_min_accumulated_frame_num: u32,
    pub convergence_s: f32,
    pub convergence_b: f32,
    pub convergence_p: f32,
    pub max_accumulated_frame_num: u32,
    pub max_fast_accumulated_frame_num: u32,
    pub max_stabilized_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub history_fix_base_pixel_stride: u32,
    pub history_fix_alternate_pixel_stride: u32,
    pub fast_history_clamping_sigma_scale: f32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_hit_distance_weight: f32,
    pub min_blur_radius: f32,
    pub max_blur_radius: f32,
    pub lobe_angle_fraction: f32,
    pub roughness_fraction: f32,
    pub plane_distance_sensitivity: f32,
    pub specular_probability_thresholds_for_mv_modification: [f32; 2],
    pub firefly_suppressor_min_relative_scale: f32,
    pub min_material_for_diffuse: f32,
    pub min_material_for_specular: f32,
    pub checkerboard_mode: u8,
    pub hit_distance_reconstruction_mode: u8,
    pub enable_anti_firefly: bool,
    pub use_prepass_only_for_specular_motion_estimation: bool,
    pub return_history_length_instead_of_occlusion: bool,
}

impl Default for ReblurSettings {
    fn default() -> Self {
        Self {
            hit_distance_parameters_a: 3.0,
            hit_distance_parameters_b: 0.1,
            hit_distance_parameters_c: 20.0,
            antilag_luminance_sigma_scale: 2.0,
            antilag_luminance_sensitivity: 3.0,
            responsive_accumulation_roughness_threshold: 0.0,
            responsive_accumulation_min_accumulated_frame_num: 3,
            convergence_s: 1.0,
            convergence_b: 0.2,
            convergence_p: 0.8,
            max_accumulated_frame_num: 30,
            max_fast_accumulated_frame_num: 6,
            max_stabilized_frame_num: 63,
            history_fix_frame_num: 3,
            history_fix_base_pixel_stride: 14,
            history_fix_alternate_pixel_stride: 14,
            fast_history_clamping_sigma_scale: 2.0,
            diffuse_prepass_blur_radius: 30.0,
            specular_prepass_blur_radius: 50.0,
            min_hit_distance_weight: 0.1,
            min_blur_radius: 1.0,
            max_blur_radius: 30.0,
            lobe_angle_fraction: 0.15,
            roughness_fraction: 0.15,
            plane_distance_sensitivity: 0.02,
            specular_probability_thresholds_for_mv_modification: [0.5, 0.9],
            firefly_suppressor_min_relative_scale: 2.0,
            min_material_for_diffuse: 4.0,
            min_material_for_specular: 4.0,
            checkerboard_mode: 0,
            hit_distance_reconstruction_mode: hit_distance_reconstruction_mode::AREA_3X3,
            enable_anti_firefly: true,
            use_prepass_only_for_specular_motion_estimation: false,
            return_history_length_instead_of_occlusion: false,
        }
    }
}

// ABI pins probed against the vendored headers (third_party/probe_sizes.cpp).
const _: () = assert!(size_of::<LibraryDesc>() == 40);
const _: () = assert!(size_of::<InstanceCreationDesc>() == 48);
const _: () = assert!(size_of::<InstanceDesc>() == 112);
const _: () = assert!(size_of::<PipelineDesc>() == 320);
const _: () = assert!(size_of::<ComputeShaderDesc>() == 16);
const _: () = assert!(size_of::<ResourceRangeDesc>() == 8);
const _: () = assert!(size_of::<ResourceDesc>() == 12);
const _: () = assert!(size_of::<TextureDesc>() == 8);
const _: () = assert!(size_of::<DescriptorPoolDesc>() == 20);
const _: () = assert!(size_of::<DispatchDesc>() == 56);
const _: () = assert!(size_of::<CommonSettings>() == 432);
const _: () = assert!(size_of::<ReblurSettings>() == 128);
const _: () = assert!(size_of::<SpirvBindingOffsets>() == 16);

unsafe extern "system" {
    pub fn CreateInstance(
        instance_creation_desc: *const InstanceCreationDesc,
        instance: *mut *mut Instance,
    ) -> u32;

    pub fn DestroyInstance(instance: *mut Instance);

    pub fn GetLibraryDesc() -> *const LibraryDesc;

    pub fn GetInstanceDesc(instance: *const Instance) -> *const InstanceDesc;

    pub fn SetCommonSettings(
        instance: *mut Instance,
        common_settings: *const CommonSettings,
    ) -> u32;

    pub fn SetDenoiserSettings(
        instance: *mut Instance,
        identifier: Identifier,
        denoiser_settings: *const c_void,
    ) -> u32;

    pub fn GetComputeDispatches(
        instance: *mut Instance,
        identifiers: *const Identifier,
        identifiers_num: u32,
        dispatch_descs: *mut *const DispatchDesc,
        dispatch_descs_num: *mut u32,
    ) -> u32;
}
