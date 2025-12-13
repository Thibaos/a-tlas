use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use vulkano::{acceleration_structure::AccelerationStructure, device::Queue};
use vulkano_taskgraph::{
    Id, QueueFamilyType,
    graph::{CompileInfo, ExecutableTaskGraph, TaskGraph},
    resource::{Flight, HostAccessType, Resources},
    resource_map,
};

use crate::tasks::update_as::{AsyncRenderContext, UpdateAccelerationStructureTask};

const TRANSFER_GRANULARITY: u32 = 4096;

fn init_worker(
    update_as_task: UpdateAccelerationStructureTask,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
) -> ExecutableTaskGraph<AsyncRenderContext> {
    let mut task_graph = TaskGraph::new(&resources);

    task_graph.add_host_buffer_access(update_as_task.instance_buffer_id, HostAccessType::Write);

    task_graph
        .create_task_node("Update TLAS", QueueFamilyType::Compute, update_as_task)
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
    channel: mpsc::Receiver<()>,
    update_as_task: UpdateAccelerationStructureTask,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    graphics_flight_id: Id<Flight>,
    compute_flight_id: Id<Flight>,
    acceleration_structures: [Arc<AccelerationStructure>; 2],
    current_as_index: Arc<AtomicBool>,
    show_current_index: Arc<AtomicBool>,
) {
    let task_graph = init_worker(update_as_task, queue, resources.clone(), compute_flight_id);

    thread::spawn(move || {
        let mut last_frame = 0;

        while let Ok(()) = channel.recv() {
            let now = Instant::now();

            let graphics_flight = resources.flight(graphics_flight_id).unwrap();

            while last_frame == graphics_flight.current_frame() {
                thread::sleep(Duration::from_millis(1));
            }

            graphics_flight.wait_for_frame(last_frame, None).unwrap();

            let back_index = !current_as_index.load(Ordering::Relaxed);
            // println!("Updating TLAS at index: {back_index}");

            let resource_map = resource_map!(&task_graph).unwrap();

            unsafe {
                task_graph.execute(
                    resource_map,
                    &AsyncRenderContext {
                        tlas: acceleration_structures[back_index as usize].clone(),
                    },
                    || {},
                )
            }
            .unwrap();

            resources
                .flight(compute_flight_id)
                .unwrap()
                .wait_idle()
                .unwrap();

            last_frame = graphics_flight.current_frame();

            current_as_index.store(back_index, Ordering::Relaxed);
            show_current_index.store(true, Ordering::Relaxed);
            println!(
                "TLAS update took: {:.2}ms",
                now.elapsed().as_micros() as f64 / 1000.
            );
        }
    });
}
