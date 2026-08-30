//! The frame images (CONTEXT.md): every extent-bound image the frame's
//! passes read and write. The set spans the Trace pass's outputs, the
//! Denoise pass's outputs, and the swapchain's bindless storage views. The
//! task graph references the set virtually; physical images and bindless
//! registrations attach per extent, and a resize destroys and recreates the
//! whole set in one deferred batch.

use vulkano::{
    format::Format,
    image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, view::ImageView},
    memory::allocator::AllocationCreateInfo,
    swapchain::Swapchain,
};
use vulkano_taskgraph::{
    Id,
    descriptor_set::StorageImageId,
    graph::{TaskGraph, TaskNodeBuilder},
    resource::{AccessTypes, ImageLayoutType, Resources},
};

use crate::core::render::{nrd::NrdInputs, region::task::RegionRenderContext};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameImageKind {
    DiffRadiance,
    SpecRadiance,
    NormalRoughness,
    ViewZ,
    Mv,
    AlbedoMetal,
    DenoisedDiff,
    DenoisedSpec,
    Validation,
}

use FrameImageKind::*;

const KINDS: [FrameImageKind; 9] = [
    DiffRadiance,
    SpecRadiance,
    NormalRoughness,
    ViewZ,
    Mv,
    AlbedoMetal,
    DenoisedDiff,
    DenoisedSpec,
    Validation,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    TraceOutput,
    DenoiserInput,
    DenoiserOutput,
    CompositeRead,
}

struct Entry {
    kind: FrameImageKind,
    format: Format,
    virtual_id: Id<Image>,
    physical_id: Id<Image>,
    storage_id: StorageImageId,
}

fn format_of(kind: FrameImageKind) -> Format {
    match kind {
        DiffRadiance | SpecRadiance | Mv | DenoisedDiff | DenoisedSpec => {
            Format::R16G16B16A16_SFLOAT
        }
        NormalRoughness | AlbedoMetal | Validation => Format::R8G8B8A8_UNORM,
        ViewZ => Format::R32_SFLOAT,
    }
}

fn roles_of(kind: FrameImageKind) -> &'static [Role] {
    match kind {
        DiffRadiance => &[Role::TraceOutput, Role::DenoiserInput, Role::CompositeRead],
        SpecRadiance | NormalRoughness | Mv => &[Role::TraceOutput, Role::DenoiserInput],
        ViewZ => &[Role::TraceOutput, Role::DenoiserInput, Role::CompositeRead],
        AlbedoMetal => &[Role::TraceOutput],
        DenoisedDiff | DenoisedSpec | Validation => &[Role::DenoiserOutput, Role::CompositeRead],
    }
}

pub struct FrameImages {
    entries: Vec<Entry>,
    attached: bool,
    swapchain_storage: Vec<StorageImageId>,
}

impl FrameImages {
    pub fn declare(task_graph: &mut TaskGraph<RegionRenderContext>) -> Self {
        let entries = KINDS
            .into_iter()
            .map(|kind| {
                let format = format_of(kind);

                let virtual_id = task_graph.add_image(&ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format,
                    usage: ImageUsage::STORAGE,
                    ..Default::default()
                });

                Entry {
                    kind,
                    format,
                    virtual_id,
                    physical_id: Id::INVALID,
                    storage_id: StorageImageId::INVALID,
                }
            })
            .collect();

        Self {
            entries,
            attached: false,
            swapchain_storage: Vec::new(),
        }
    }

    pub fn recreate(
        &mut self,
        resources: &Resources,
        swapchain_id: Id<Swapchain>,
        extent: [u32; 3],
    ) {
        if self.attached {
            let mut batch = resources.create_deferred_batch();

            for entry in &self.entries {
                batch.destroy_image(entry.physical_id);
                batch.destroy_storage_image(entry.storage_id);
            }

            for id in &self.swapchain_storage {
                batch.destroy_storage_image(*id);
            }

            batch.enqueue();
        }

        self.swapchain_storage = swapchain_storage_views(resources, swapchain_id);

        let bcx = resources.bindless_context().unwrap();

        for entry in &mut self.entries {
            let physical_id = resources
                .create_image(
                    &ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: entry.format,
                        extent: [extent[0], extent[1], 1],
                        usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    &AllocationCreateInfo::default(),
                )
                .unwrap();

            let image = resources.image(physical_id).image().clone();
            let view = ImageView::new_default(&image).unwrap();

            entry.physical_id = physical_id;
            entry.storage_id = bcx
                .global_set()
                .add_storage_image(view, ImageLayout::General);
        }

        self.attached = true;
    }

    pub fn bind_into(&self, region: &mut RegionRenderContext) {
        for entry in &self.entries {
            *image_slot_mut(region, entry.kind) = entry.storage_id;
        }

        region.swapchain_storage_image_ids = self.swapchain_storage.clone();
    }

    pub fn resource_pairs(&self) -> impl Iterator<Item = (Id<Image>, Id<Image>)> + '_ {
        self.entries.iter().map(|e| (e.virtual_id, e.physical_id))
    }

    pub fn nrd_inputs(&self, resources: &Resources) -> NrdInputs {
        let view = |kind: FrameImageKind| {
            let physical_id = self.get(kind).physical_id;
            ImageView::new_default(&resources.image(physical_id).image().clone()).unwrap()
        };

        NrdInputs {
            diff_radiance: view(DiffRadiance),
            spec_radiance: view(SpecRadiance),
            normal_roughness: view(NormalRoughness),
            viewz: view(ViewZ),
            mv: view(Mv),
            diff_out: view(DenoisedDiff),
            spec_out: view(DenoisedSpec),
            validation: view(Validation),
        }
    }

    pub fn declare_trace_outputs(&self, node: &mut TaskNodeBuilder<'_>) {
        for entry in self.entries_with_role(Role::TraceOutput) {
            node.image_access(
                entry.virtual_id,
                AccessTypes::RAY_TRACING_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            );
        }
    }

    /// The NRD constants buffer stays undeclared on purpose: declaring the
    /// physical id here trips ResourceMap validation (InvalidSlotError). Its
    /// hazards are covered anyway: update_buffer lands in the same recording
    /// as the dispatches behind an explicit TRANSFER_WRITE barrier, and
    /// frames are serialized by the per-frame wait_idle.
    pub fn declare_denoise_io(&self, node: &mut TaskNodeBuilder<'_>) {
        for entry in self.entries_with_role(Role::DenoiserInput) {
            node.image_access(
                entry.virtual_id,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                ImageLayoutType::General,
            );
        }

        for entry in self.entries_with_role(Role::DenoiserOutput) {
            node.image_access(
                entry.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                ImageLayoutType::General,
            );
        }
    }

    pub fn declare_composite_reads(&self, node: &mut TaskNodeBuilder<'_>) {
        for entry in self.entries_with_role(Role::CompositeRead) {
            node.image_access(
                entry.virtual_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_READ,
                ImageLayoutType::General,
            );
        }
    }

    fn entries_with_role(&self, role: Role) -> impl Iterator<Item = &Entry> + '_ {
        self.entries
            .iter()
            .filter(move |e| roles_of(e.kind).contains(&role))
    }

    fn get(&self, kind: FrameImageKind) -> &Entry {
        self.entries.iter().find(|e| e.kind == kind).unwrap()
    }
}

fn image_slot_mut(region: &mut RegionRenderContext, kind: FrameImageKind) -> &mut StorageImageId {
    match kind {
        DiffRadiance => &mut region.diff_radiance_image_id,
        SpecRadiance => &mut region.spec_radiance_image_id,
        NormalRoughness => &mut region.normal_roughness_image_id,
        ViewZ => &mut region.viewz_image_id,
        Mv => &mut region.mv_image_id,
        AlbedoMetal => &mut region.albedo_metal_image_id,
        DenoisedDiff => &mut region.denoised_diff_image_id,
        DenoisedSpec => &mut region.denoised_spec_image_id,
        Validation => &mut region.validation_image_id,
    }
}

fn swapchain_storage_views(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
) -> Vec<StorageImageId> {
    let bcx = resources.bindless_context().unwrap();
    let swapchain_state = resources.swapchain(swapchain_id);
    let images = swapchain_state.images();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image).unwrap();

            bcx.global_set()
                .add_storage_image(view, ImageLayout::General)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::render::region::task::{
        NrdFrame, RenderMode, default_scene, production_raygen,
    };

    fn test_images() -> FrameImages {
        FrameImages {
            entries: KINDS
                .into_iter()
                .map(|kind| Entry {
                    kind,
                    format: format_of(kind),
                    virtual_id: Id::INVALID,
                    physical_id: Id::INVALID,
                    storage_id: StorageImageId::INVALID,
                })
                .collect(),
            attached: false,
            swapchain_storage: vec![StorageImageId::INVALID; 3],
        }
    }

    fn test_region() -> RegionRenderContext {
        RegionRenderContext {
            camera: production_raygen::Camera {
                proj_inverse: [[0.0; 4]; 4],
                view_inverse: [[0.0; 4]; 4],
                view_prev: [[0.0; 4]; 4],
                proj_prev: [[0.0; 4]; 4],
            },
            scene: default_scene(),
            swapchain_storage_image_ids: Vec::new(),
            diff_radiance_image_id: StorageImageId::INVALID,
            spec_radiance_image_id: StorageImageId::INVALID,
            normal_roughness_image_id: StorageImageId::INVALID,
            viewz_image_id: StorageImageId::INVALID,
            mv_image_id: StorageImageId::INVALID,
            denoised_diff_image_id: StorageImageId::INVALID,
            denoised_spec_image_id: StorageImageId::INVALID,
            validation_image_id: StorageImageId::INVALID,
            denoiser_active: false,
            nrd: NrdFrame::default(),
            albedo_metal_image_id: StorageImageId::INVALID,
            ev: 0.0,
            mode: RenderMode::default(),
            frame_seed: 0,
        }
    }

    #[test]
    fn kinds_are_unique() {
        let mut seen = Vec::new();

        for kind in KINDS {
            assert!(!seen.contains(&kind));
            seen.push(kind);
        }

        assert_eq!(seen.len(), KINDS.len());
    }

    #[test]
    fn formats_match_the_output_contract() {
        assert_eq!(format_of(DiffRadiance), Format::R16G16B16A16_SFLOAT);
        assert_eq!(format_of(SpecRadiance), Format::R16G16B16A16_SFLOAT);
        assert_eq!(format_of(NormalRoughness), Format::R8G8B8A8_UNORM);
        assert_eq!(format_of(ViewZ), Format::R32_SFLOAT);
        assert_eq!(format_of(Mv), Format::R16G16B16A16_SFLOAT);
        assert_eq!(format_of(AlbedoMetal), Format::R8G8B8A8_UNORM);
        assert_eq!(format_of(DenoisedDiff), Format::R16G16B16A16_SFLOAT);
        assert_eq!(format_of(DenoisedSpec), Format::R16G16B16A16_SFLOAT);
        assert_eq!(format_of(Validation), Format::R8G8B8A8_UNORM);
    }

    fn with_role(role: Role) -> Vec<FrameImageKind> {
        KINDS
            .into_iter()
            .filter(|k| roles_of(*k).contains(&role))
            .collect()
    }

    #[test]
    fn trace_outputs_are_the_six_production_writes() {
        assert_eq!(
            with_role(Role::TraceOutput),
            vec![
                DiffRadiance,
                SpecRadiance,
                NormalRoughness,
                ViewZ,
                Mv,
                AlbedoMetal
            ]
        );
    }

    #[test]
    fn denoiser_inputs_exclude_albedo_metal() {
        assert_eq!(
            with_role(Role::DenoiserInput),
            vec![DiffRadiance, SpecRadiance, NormalRoughness, ViewZ, Mv]
        );
    }

    #[test]
    fn denoiser_outputs_are_the_denoised_pair_plus_validation() {
        assert_eq!(
            with_role(Role::DenoiserOutput),
            vec![DenoisedDiff, DenoisedSpec, Validation]
        );
    }

    #[test]
    fn composite_reads_radiance_viewz_and_denoiser_outputs() {
        assert_eq!(
            with_role(Role::CompositeRead),
            vec![DiffRadiance, ViewZ, DenoisedDiff, DenoisedSpec, Validation]
        );
    }

    #[test]
    fn image_slots_are_pairwise_distinct() {
        let mut region = test_region();
        let mut slots: Vec<usize> = KINDS
            .into_iter()
            .map(|kind| image_slot_mut(&mut region, kind) as *mut StorageImageId as usize)
            .collect();

        slots.sort_unstable();
        slots.dedup();

        assert_eq!(slots.len(), KINDS.len());
    }

    #[test]
    fn bind_into_hands_over_the_swapchain_views() {
        let images = test_images();
        let mut region = test_region();

        images.bind_into(&mut region);

        assert_eq!(region.swapchain_storage_image_ids.len(), 3);
    }
}
