use std::time::Instant;

use crate::TICKS_PER_SECOND;

pub struct PhysicsController {
    last_update: Instant,
}

impl PhysicsController {
    pub fn new() -> Self {
        PhysicsController {
            last_update: Instant::now(),
        }
    }

    fn should_step(&self) -> bool {
        let elapsed = self.last_update.elapsed().as_secs_f32();

        elapsed > 1.0 / TICKS_PER_SECOND as f32
    }

    fn step(&mut self) {
        self.last_update = Instant::now();
    }

    pub fn request_update(&mut self) {
        if self.should_step() {
            self.step();
        }
    }
}
