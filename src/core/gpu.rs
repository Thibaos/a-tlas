//! The shared GPU stack: instance, device, queues, allocator, and the
//! taskgraph resources/flights. Constructed once per event loop by the app.
//! Every GPU layer builds on this module (core: no dependencies).

use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    swapchain::Surface,
};
use vulkano_taskgraph::{
    Id,
    descriptor_set::BindlessContext,
    resource::{Flight, Resources, ResourcesCreateInfo},
};
use winit::event_loop::EventLoop;

pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;
pub const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

/// The shared GPU stack (instance, device, queues, allocator, taskgraph
/// resources and flights). Constructed once per event loop by [`App::new`] and
/// by the offline validator.
pub struct GpuStack {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,

    pub graphics_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
    pub transfer_queue: Arc<Queue>,

    pub memory_allocator: Arc<dyn MemoryAllocator>,

    pub resources: Arc<Resources>,
    pub graphics_flight_id: Id<Flight>,
    pub compute_flight_id: Id<Flight>,
}

impl GpuStack {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let required_extensions = Surface::required_extensions(event_loop);

        let library = unsafe { VulkanLibrary::new() }.unwrap();
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &InstanceExtensions {
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_acceleration_structure: true,
            khr_deferred_host_operations: true,
            khr_ray_tracing_maintenance1: true,
            khr_ray_tracing_pipeline: true,
            khr_synchronization2: true,
            khr_shader_clock: true,
            khr_swapchain: true,
            ..BindlessContext::required_extensions(&instance)
        };
        let device_features = DeviceFeatures {
            acceleration_structure: true,
            descriptor_binding_acceleration_structure_update_after_bind: true,
            ray_tracing_pipeline: true,
            buffer_device_address: true,
            storage_push_constant8: true,
            synchronization2: true,
            shader_float64: true,
            shader_int64: true,
            shader_int8: true,
            shader_subgroup_clock: true,
            shader_device_clock: cfg!(debug_assertions),
            storage_buffer8_bit_access: true,
            ..BindlessContext::required_features(&instance)
        };

        let (physical_device, graphics_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().contains(&device_features)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        let compute_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
            .min_by_key(|(_, q)| q.queue_flags.count())
            .unwrap()
            .0 as u32;

        let transfer_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_flags.intersects(QueueFlags::TRANSFER))
            .min_by_key(|(_, q)| q.queue_flags.count())
            .unwrap()
            .0 as u32;

        let (device, mut queues) = {
            let mut queue_create_infos = vec![QueueCreateInfo {
                queue_family_index: graphics_family_index,
                ..Default::default()
            }];

            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: compute_family_index,
                ..Default::default()
            });

            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: transfer_family_index,
                ..Default::default()
            });

            Device::new(
                &physical_device,
                &DeviceCreateInfo {
                    enabled_extensions: &device_extensions,
                    enabled_features: &device_features,
                    queue_create_infos: &queue_create_infos,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let graphics_queue = queues.next().unwrap();
        let compute_queue = queues.next().unwrap();
        let transfer_queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let resources = Resources::new(
            &device,
            &ResourcesCreateInfo {
                bindless_context: Some(&Default::default()),
                ..Default::default()
            },
        )
        .unwrap();

        let graphics_flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();
        let compute_flight_id = resources.create_flight(1).unwrap();

        Self {
            instance,
            device,
            graphics_queue,
            compute_queue,
            transfer_queue,
            memory_allocator,
            resources,
            graphics_flight_id,
            compute_flight_id,
        }
    }
}
