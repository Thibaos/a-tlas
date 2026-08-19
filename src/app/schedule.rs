use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use either::Either;

pub type Condition = Either<(usize, usize), Duration>;
pub struct ScheduleController {
    schedules: HashMap<&'static str, (Instant, Condition)>,
}

impl ScheduleController {
    pub fn new() -> Self {
        Self {
            schedules: HashMap::new(),
        }
    }

    pub fn add_schedule_duration(&mut self, name: &'static str, duration: Duration) {
        let instant = Instant::now();
        self.schedules
            .insert(name, (instant, Either::Right(duration)));
    }

    pub fn add_schedule_frames(&mut self, name: &'static str, frames: usize) {
        let instant = Instant::now();
        self.schedules
            .insert(name, (instant, Either::Left((frames, frames))));
    }

    pub fn check(&mut self, key: &str) -> Option<Duration> {
        if let Some((last_update_instant, condition)) = self.schedules.get_mut(key) {
            match condition {
                Either::Left((remaining, total_frames)) => {
                    if *remaining <= 1 {
                        let duration = last_update_instant.elapsed();
                        *last_update_instant = Instant::now();
                        *remaining = *total_frames;
                        return Some(duration);
                    }

                    *remaining -= 1;

                    return None;
                }
                Either::Right(duration) => {
                    if (*last_update_instant).elapsed() > *duration {
                        *last_update_instant = Instant::now();
                        return Some(*duration);
                    } else {
                        return None;
                    }
                }
            }
        }

        None
    }
}
