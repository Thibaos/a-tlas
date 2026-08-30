mod input;
mod player;
mod schedule;

use std::{sync::Arc, time::Duration};

use glam::Mat4;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes},
};

use crate::{
    app::{
        input::{Input, InputButton, InputKey},
        player::PlayerController,
        schedule::ScheduleController,
    },
    core::{
        render::{
            gpu::GpuDesc,
            pipeline::{FrameInput, FramePipeline},
        },
        world::{World, format::open_file, grid::LATTICE_HALF_EXTENT},
    },
};

pub struct App {
    close_requested: bool,

    pub gpu: GpuDesc,

    delta_time: Duration,
    focused: bool,

    pub voxel_data: dot_vox::DotVoxData,
    pub world: Arc<World>,

    player_controller: PlayerController,
    player_input: Input,
    schedule_controller: ScheduleController,

    window: Option<Arc<Window>>,
    pipeline: Option<FramePipeline>,

    resize_pending: bool,
    mode_toggle_pending: bool,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>, world_path: &str, clip_oob: bool) -> Self {
        let gpu = GpuDesc::new(event_loop);

        let voxel_data = open_file(world_path);
        let (world, clipped) = if clip_oob {
            World::new_clipped(&voxel_data)
        } else {
            (World::new(&voxel_data), 0)
        };
        if clipped > 0 {
            println!(
                "clipped {clipped} voxels outside the ±{} lattice",
                LATTICE_HALF_EXTENT
            );
        }
        let world = Arc::new(world);

        let mut schedule_controller = ScheduleController::new();
        schedule_controller.add_schedule_frames("delta", 1);
        schedule_controller.add_schedule_duration("log", Duration::from_secs(1));

        App {
            close_requested: false,

            gpu,

            delta_time: Duration::ZERO,
            focused: false,

            player_controller: PlayerController::default(),
            player_input: Input::default(),
            schedule_controller,

            voxel_data,
            world,

            window: None,
            pipeline: None,

            resize_pending: false,
            mode_toggle_pending: false,
        }
    }

    pub fn toggle_capture_mouse(&mut self) {
        let window = self.window.as_ref().unwrap();

        if self.focused {
            self.focused = false;
            window
                .set_cursor_grab(winit::window::CursorGrabMode::None)
                .unwrap();
            window.set_cursor_visible(true);
        } else {
            self.focused = true;
            window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .unwrap();
            window.set_cursor_visible(false);
        }
    }

    #[cfg(debug_assertions)]
    fn handle_toggle_render_mode(&mut self) {
        if self
            .player_input
            .just_pressed
            .contains(&InputKey::ToggleRenderMode)
        {
            self.mode_toggle_pending = true;
        }
    }

    fn update_delta_time(&mut self) {
        self.delta_time = self
            .schedule_controller
            .check("delta")
            .expect("Delta time calculation returned None!");
    }

    fn request_log(&mut self) {
        if self.schedule_controller.check("log").is_some() {
            println!("{:.2} fps", 1.0 / self.delta_time.as_secs_f32());
        }
    }

    fn player_view(&mut self) -> Mat4 {
        if self.focused {
            self.player_controller
                .rotate(self.player_input.mouse_motion);
        }

        self.player_controller
            .fly_movement(self.delta_time, &self.player_input);

        self.player_controller.view()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes =
            WindowAttributes::default().with_inner_size(PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.pipeline = Some(FramePipeline::new(
            &self.gpu,
            window.clone(),
            &self.voxel_data,
            &self.world,
        ));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.close_requested = true;
            }
            WindowEvent::Resized(_) => {
                self.resize_pending = true;
            }
            WindowEvent::RedrawRequested => {
                self.update_delta_time();
                self.request_log();

                let view = self.player_view();

                let resized = std::mem::take(&mut self.resize_pending);
                let next_mode = std::mem::take(&mut self.mode_toggle_pending);

                self.pipeline.as_mut().unwrap().run_frame(
                    &self.gpu,
                    FrameInput {
                        view,
                        resized,
                        next_mode,
                    },
                );
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(mapped) = input::map_mouse_button(button) {
                    match state {
                        ElementState::Pressed => {
                            if mapped == InputButton::Right {
                                self.toggle_capture_mouse();
                            }
                            self.player_input.buttons_down.insert(mapped);
                        }
                        ElementState::Released => {
                            self.player_input.buttons_down.remove(&mapped);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => {
                self.player_input.scroll_delta += y;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(key) = input::map_key(&event.logical_key) {
                    match event.state {
                        ElementState::Pressed => {
                            self.player_input.down.insert(key);
                            self.player_input.just_pressed.insert(key);
                        }
                        ElementState::Released => {
                            self.player_input.down.remove(&key);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.player_input.just_pressed.contains(&InputKey::Close) {
            self.close_requested = true;
        }

        #[cfg(debug_assertions)]
        self.handle_toggle_render_mode();
        self.player_input.clear();

        if self.close_requested {
            event_loop.exit();
        } else {
            self.window.as_ref().unwrap().request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.player_input.mouse_motion.0 += delta.0;
            self.player_input.mouse_motion.1 += delta.1;
        };
    }
}
