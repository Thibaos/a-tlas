use atlas_rt::app::App;
use std::error::Error;
use winit::event_loop::EventLoop;

fn main() -> Result<(), impl Error> {
    let args: Vec<String> = std::env::args().collect();

    // `atlas-rt validate [options]` — the offline correctness validator.
    if args.get(1).map(String::as_str) == Some("validate") {
        return match atlas_rt::validate::run(&args[2..]) {
            Ok(()) => Ok(()),
            Err(message) => {
                eprintln!("validate: {message}");
                std::process::exit(1);
            }
        };
    }

    // `atlas-rt --measure` (renderer-impl ticket 07): attach the GPU
    // measurement — per-stage timestamps (trace_rays / AS rebuild / flight),
    // min/avg/p95 in the FPS log, the 16 ms gate as the GPU timestamp sum
    // with wall-clock beside it. On demand: off by default, and the
    // validator never measures (the harness's captured frames are
    // unaffected).
    let measure = args.iter().any(|arg| arg == "--measure");
    let clip_oob = args.iter().any(|arg| arg == "--clip-oob");
    let world_path = args
        .iter()
        .position(|arg| arg == "--world")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("assets/bistro.vox");

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop, measure, world_path, clip_oob);

    event_loop.run_app(&mut app)
}
