use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use glam::IVec3;
use vulkano::{acceleration_structure::AccelerationStructure, device::Queue};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, Resources},
    resource_map,
};

use crate::{
    app::AsyncRenderContext, tasks::update_as::UpdateAccelerationStructureTask,
    world::chunk::Chunks,
};

fn init_worker(
    update_as_task: UpdateAccelerationStructureTask,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
) -> ExecutableTaskGraph<AsyncRenderContext> {
    let mut task_graph = TaskGraph::new(&resources);

    task_graph.add_host_buffer_access(update_as_task.instance_buffer_id, HostAccessType::Write);

    let instance_buffer_id = update_as_task.instance_buffer_id;

    task_graph
        .create_task_node("Update TLAS", QueueFamilyType::Compute, update_as_task)
        .buffer_access(
            instance_buffer_id,
            AccessTypes::ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE,
        )
        .build();

    unsafe {
        task_graph.compile(&CompileInfo {
            queues: &[&queue],
            flight_id,
            ..Default::default()
        })
    }
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
pub fn run_worker(
    channel: mpsc::Receiver<IVec3>,
    update_as_task: UpdateAccelerationStructureTask,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    graphics_flight_id: Id<Flight>,
    compute_flight_id: Id<Flight>,
    acceleration_structures: [Arc<AccelerationStructure>; 2],
    current_as_index: Arc<AtomicBool>,
    world: Arc<Chunks>,
    worker_available: Arc<AtomicBool>,
) {
    let task_graph = init_worker(update_as_task, queue, resources.clone(), compute_flight_id);

    thread::spawn(move || {
        let mut last_frame = 0;

        while let Ok(position) = channel.recv() {
            let now = std::time::Instant::now();

            worker_available.store(false, Ordering::Release);

            let graphics_flight = resources.flight(graphics_flight_id);

            while last_frame == graphics_flight.current_frame() {
                thread::sleep(Duration::from_millis(1));
            }

            graphics_flight.wait_for_frame(last_frame, None).unwrap();

            let back_index = !current_as_index.load(Ordering::Acquire);

            let resource_map = resource_map!(&task_graph).unwrap();

            unsafe {
                task_graph.execute(
                    resource_map,
                    &AsyncRenderContext {
                        acceleration_structures: acceleration_structures.clone(),
                        current_as_index: current_as_index.clone(),
                        world: world.clone(),
                        position,
                    },
                    || {},
                )
            }
            .unwrap();

            resources.flight(compute_flight_id).wait_idle().unwrap();

            last_frame = graphics_flight.current_frame();

            current_as_index.store(back_index, Ordering::Release);

            worker_available.store(true, Ordering::Release);

            println!("async tlas update took: {:.2}ms", now.elapsed().as_millis())
        }
    });
}
