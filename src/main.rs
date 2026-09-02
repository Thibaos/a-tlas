use atlas_rt::app::App;
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let clip_oob = args.iter().any(|arg| arg == "--clip-oob");
    let world_path = args
        .iter()
        .position(|arg| arg == "--world")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("assets/nuke.vox");

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop, world_path, clip_oob)?;

    let result = event_loop.run_app(&mut app)?;

    Ok(result)
}
