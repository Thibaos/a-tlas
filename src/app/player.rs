use core::f32;
use std::{
    f32::consts::{FRAC_PI_2, TAU},
    ops::{Add, Mul, Sub},
    time::Duration,
};

use glam::{Mat4, Quat, Vec3, vec3};

use crate::app::input::{Input, InputKey};

pub struct PlayerController {
    pub speed: f32,
    pub sensitivity: f64,
    pub translation: Vec3,

    yaw: f32,
    pitch: f32,

    view: Mat4,
    needs_view_update: bool,
}

impl Default for PlayerController {
    fn default() -> Self {
        let translation = Vec3::new(124.0, 110.0, 320.0);

        Self {
            speed: 64.0,
            sensitivity: 0.001,
            translation,
            yaw: 0.0,
            pitch: 0.0,
            view: Mat4::IDENTITY,
            needs_view_update: true,
        }
    }
}

impl PlayerController {
    const MAX_PITCH: f32 = FRAC_PI_2 - 0.01;
    const MIN_PITCH: f32 = -Self::MAX_PITCH;

    pub fn view(&mut self) -> Mat4 {
        if self.needs_view_update {
            self.compute_view();
        }

        self.view
    }

    pub fn fly_movement(&mut self, delta_time: Duration, input: &Input) {
        if input.scroll_delta > 0.0 {
            self.speed *= 1.5;
        } else if input.scroll_delta < 0.0 {
            self.speed /= 1.5;
        }

        let view_inverse = self.view().inverse();
        let absolute_forward = view_inverse.transform_vector3(Vec3::Z);
        let forward = vec3(absolute_forward.x, 0.0, absolute_forward.z).normalize();
        let right = view_inverse.transform_vector3(-Vec3::X);

        let mut velocity = glam::Vec3::ZERO;

        if input.down.contains(&InputKey::Forward) {
            velocity = velocity.add(forward);
        } else if input.down.contains(&InputKey::Backward) {
            velocity = velocity.sub(forward);
        }
        if input.down.contains(&InputKey::Left) {
            velocity = velocity.add(right);
        } else if input.down.contains(&InputKey::Right) {
            velocity = velocity.sub(right);
        }
        if input.down.contains(&InputKey::Up) {
            velocity = velocity.add(glam::Vec3::Y);
        } else if input.down.contains(&InputKey::Down) {
            velocity = velocity.sub(glam::Vec3::Y);
        }

        velocity = velocity.normalize_or_zero();

        self.translation = self
            .translation
            .add(velocity.mul(delta_time.as_secs_f32()).mul(self.speed));

        self.needs_view_update = true;
    }

    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    pub fn rotate(&mut self, delta: (f64, f64)) {
        self.yaw = self.yaw.add(delta.0.mul(self.sensitivity) as f32);
        self.pitch = self.pitch.sub((delta.1.mul(self.sensitivity)) as f32);

        self.yaw = self.yaw.rem_euclid(TAU);

        self.pitch = self.pitch.clamp(Self::MIN_PITCH, Self::MAX_PITCH);

        self.needs_view_update = true;
    }

    fn orientation(&self) -> Quat {
        let yaw_q = Quat::from_rotation_y(self.yaw);
        let pitch_q = Quat::from_rotation_x(self.pitch);

        yaw_q.mul(pitch_q)
    }

    fn compute_view(&mut self) {
        let rot = self.orientation();
        let forward = rot.mul_vec3(Vec3::NEG_Z);
        let up = rot.mul_vec3(Vec3::Y);

        self.view = glam::camera::lh::view::look_at_mat4(
            self.translation,
            self.translation.add(forward),
            up,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_with(down: &[InputKey]) -> Input {
        let mut input = Input::default();
        for &key in down {
            input.down.insert(key);
        }
        input
    }

    #[test]
    fn forward_moves_along_look_axis() {
        let mut player = PlayerController::default();
        let base_translation = player.translation;

        let input = input_with(&[InputKey::Forward]);

        player.fly_movement(Duration::from_secs(1), &input);

        assert_eq!(
            player.translation,
            base_translation.add(Vec3::new(0.0, 0.0, -64.0))
        );
    }

    #[test]
    fn held_keys_drive_velocity() {
        let cases: &[(InputKey, Vec3)] = &[
            (InputKey::Forward, Vec3::new(0.0, 0.0, -1.0)),
            (InputKey::Backward, Vec3::new(0.0, 0.0, 1.0)),
            (InputKey::Left, Vec3::new(1.0, 0.0, 0.0)),
            (InputKey::Right, Vec3::new(-1.0, 0.0, 0.0)),
            (InputKey::Up, Vec3::new(0.0, 1.0, 0.0)),
            (InputKey::Down, Vec3::new(0.0, -1.0, 0.0)),
        ];

        for &(key, direction) in cases {
            let mut player = PlayerController::default();
            let start = player.translation;
            let input = input_with(&[key]);

            player.fly_movement(Duration::from_secs(1), &input);

            assert_eq!(
                player.translation - start,
                direction * player.speed,
                "holding {key:?} should move {direction:?} * speed"
            );
        }
    }

    #[test]
    fn scroll_delta_changes_speed() {
        let mut player = PlayerController::default();

        let mut up = Input::default();
        up.down.insert(InputKey::Forward);
        up.scroll_delta = 1.0;
        let start = player.translation;
        player.fly_movement(Duration::from_secs(1), &up);
        assert!(64.0f32.mul_add(-1.5, player.speed).abs() < 0.0001);
        assert_eq!(player.translation - start, Vec3::new(0.0, 0.0, -96.0));

        let mut down = Input::default();
        down.down.insert(InputKey::Forward);
        down.scroll_delta = -1.0;
        let start = player.translation;
        player.fly_movement(Duration::from_secs(1), &down);
        assert!((player.speed - 64.0).abs() < 0.0001);
        assert_eq!(player.translation - start, Vec3::new(0.0, 0.0, -64.0));
    }
}
