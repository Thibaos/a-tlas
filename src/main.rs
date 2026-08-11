use a_tlas::app::App;
use std::error::Error;
use winit::event_loop::EventLoop;

fn main() -> Result<(), impl Error> {
    let args: Vec<String> = std::env::args().collect();

    // `a-tlas validate [options]` — the offline correctness validator.
    if args.get(1).map(String::as_str) == Some("validate") {
        return match a_tlas::validate::run(&args[2..]) {
            Ok(()) => Ok(()),
            Err(message) => {
                eprintln!("validate: {message}");
                std::process::exit(1);
            }
        };
    }

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
