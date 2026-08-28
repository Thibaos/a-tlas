//! The Denoise pass's temporal history (CONTEXT.md): the previous frame's
//! camera matrices, their validity, and the accumulation frame index. The
//! app reports camera updates, region edits, and swapchain resizes; the
//! history decides each frame's clear, reset, and frame index.

use glam::Mat4;

use crate::core::render::region::task::NrdFrame;

pub struct NrdHistory {
    prev_view: Mat4,
    prev_proj: Mat4,
    camera_valid: bool,
    frame_index: u32,
    clear_pending: bool,
    camera_frame: CameraFrame,
}

struct CameraFrame {
    view_to_clip: [f32; 16],
    world_to_view: [f32; 16],
    view_to_clip_prev: [f32; 16],
    world_to_view_prev: [f32; 16],
}

pub struct CameraPrev {
    pub view: Mat4,
    pub proj: Mat4,
}

impl NrdHistory {
    pub fn new() -> Self {
        Self {
            prev_view: Mat4::IDENTITY,
            prev_proj: Mat4::IDENTITY,
            camera_valid: false,
            frame_index: 0,
            clear_pending: true,
            camera_frame: CameraFrame {
                view_to_clip: [0.0; 16],
                world_to_view: Mat4::IDENTITY.to_cols_array(),
                view_to_clip_prev: [0.0; 16],
                world_to_view_prev: Mat4::IDENTITY.to_cols_array(),
            },
        }
    }

    pub fn observe_camera(&mut self, view: Mat4, proj: Mat4) -> CameraPrev {
        let prev = CameraPrev {
            view: self.prev_view,
            proj: self.prev_proj,
        };

        let (view_to_clip_prev, world_to_view_prev) = if self.camera_valid {
            (
                self.prev_proj.to_cols_array(),
                self.prev_view.to_cols_array(),
            )
        } else {
            (proj.to_cols_array(), view.to_cols_array())
        };

        self.camera_frame = CameraFrame {
            view_to_clip: proj.to_cols_array(),
            world_to_view: view.to_cols_array(),
            view_to_clip_prev,
            world_to_view_prev,
        };

        self.prev_view = view;
        self.prev_proj = proj;
        self.camera_valid = true;

        prev
    }

    pub fn resized(&mut self) {
        self.clear_pending = true;
    }

    pub fn advance(&mut self, edited: bool, denoiser_present: bool) -> NrdFrame {
        let clear = self.clear_pending && denoiser_present;
        let reset = edited && !clear;

        self.frame_index = if clear || reset {
            1
        } else {
            self.frame_index.wrapping_add(1)
        };
        self.clear_pending = false;

        let camera = &self.camera_frame;

        NrdFrame {
            view_to_clip: camera.view_to_clip,
            view_to_clip_prev: camera.view_to_clip_prev,
            world_to_view: camera.world_to_view,
            world_to_view_prev: camera.world_to_view_prev,
            frame_index: if clear || reset { 0 } else { self.frame_index },
            reset,
            clear,
        }
    }
}

impl Default for NrdHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn view(at: [f32; 3]) -> Mat4 {
        Mat4::from_translation(Vec3::new(at[0], at[1], at[2]))
    }

    fn proj(scale: f32) -> Mat4 {
        Mat4::from_scale(Vec3::splat(scale))
    }

    #[test]
    fn first_frame_clears_with_self_prev() {
        let mut history = NrdHistory::new();
        history.observe_camera(view([1.0, 2.0, 3.0]), proj(2.0));

        let frame = history.advance(false, true);

        assert!(frame.clear && !frame.reset);
        assert_eq!(frame.frame_index, 0);
        assert_eq!(frame.view_to_clip_prev, proj(2.0).to_cols_array());
        assert_eq!(
            frame.world_to_view_prev,
            view([1.0, 2.0, 3.0]).to_cols_array()
        );
    }

    #[test]
    fn camera_prev_tracks_the_last_observed_frame() {
        let mut history = NrdHistory::new();

        let prev = history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));
        assert_eq!(prev.view, Mat4::IDENTITY);
        assert_eq!(prev.proj, Mat4::IDENTITY);

        let prev = history.observe_camera(view([0.0, 1.0, 0.0]), proj(3.0));
        assert_eq!(prev.view, view([1.0, 0.0, 0.0]));
        assert_eq!(prev.proj, proj(2.0));
    }

    #[test]
    fn region_edit_restarts_without_clearing() {
        let mut history = NrdHistory::new();
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));
        history.advance(false, true);
        history.observe_camera(view([0.0, 1.0, 0.0]), proj(3.0));

        let frame = history.advance(true, true);

        assert!(!frame.clear && frame.reset);
        assert_eq!(frame.frame_index, 0);
        assert_eq!(frame.view_to_clip_prev, proj(2.0).to_cols_array());
        assert_eq!(
            frame.world_to_view_prev,
            view([1.0, 0.0, 0.0]).to_cols_array()
        );
    }

    #[test]
    fn frame_index_counts_up_between_restarts() {
        let mut history = NrdHistory::new();
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));
        history.advance(false, true);
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));

        assert_eq!(history.advance(false, true).frame_index, 2);
        assert_eq!(history.advance(false, true).frame_index, 3);
    }

    #[test]
    fn resize_clears_on_the_next_frame() {
        let mut history = NrdHistory::new();
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));
        history.advance(false, true);
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));
        history.advance(false, true);

        history.resized();
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));

        let frame = history.advance(false, true);

        assert!(frame.clear && frame.frame_index == 0);
    }

    #[test]
    fn absent_denoiser_never_clears() {
        let mut history = NrdHistory::new();
        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));

        assert!(!history.advance(false, false).clear);

        history.observe_camera(view([1.0, 0.0, 0.0]), proj(2.0));

        assert!(history.advance(true, false).reset);
    }
}
