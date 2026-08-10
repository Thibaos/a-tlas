use a_tlas::app::App;
use std::error::Error;
use winit::event_loop::EventLoop;

fn main() -> Result<(), impl Error> {
    let args: Vec<String> = std::env::args().collect();

    // `a-tlas harness [options]` — the offline validation harness.
    if args.get(1).map(String::as_str) == Some("harness") {
        return match a_tlas::harness::run(&args[2..]) {
            Ok(()) => Ok(()),
            Err(message) => {
                eprintln!("harness: {message}");
                std::process::exit(1);
            }
        };
    }

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
