use atlas_rt::app::App;
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let clip_oob = args.iter().any(|arg| arg == "--clip-oob");
    let world_path = args
        .iter()
        .position(|arg| arg == "--world")
        .and_then(|i| args.get(i.strict_add(1)))
        .map_or("assets/nuke.vox", String::as_str);

    let event_loop = EventLoop::new()?;

    let mut app = App::new(&event_loop, world_path, clip_oob)?;

    event_loop.run_app(&mut app)?;

    Ok(())
}
