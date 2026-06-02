use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

pub struct ScheduleController {
    schedules: HashMap<&'static str, (Instant, Option<Duration>)>,
}

impl ScheduleController {
    pub fn new() -> Self {
        Self {
            schedules: HashMap::new(),
        }
    }

    pub fn add_schedule(&mut self, name: &'static str, duration: Option<Duration>) {
        let instant = Instant::now();
        self.schedules.insert(name, (instant, duration));
    }

    pub fn check(&mut self, key: &str) -> Option<Duration> {
        if let Some((last_update_instant, duration_opt)) = self.schedules.get_mut(key) {
            if let Some(duration) = duration_opt {
                if (*last_update_instant).elapsed() > *duration {
                    *last_update_instant = Instant::now();
                    return Some(*duration);
                }
            } else {
                let duration = last_update_instant.elapsed();
                *last_update_instant = Instant::now();
                return Some(duration);
            }
        }

        None
    }
}
