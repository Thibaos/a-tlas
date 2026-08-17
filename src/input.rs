//! Player input layer.
//!
//! One winit-free input structure owned by the app and passed by reference:
//! held keys, just-pressed keys (an edge drained every frame), a this-frame
//! scroll delta, a this-frame mouse-motion pair, and held mouse buttons.
//! Before this module, input was split between the player controller (which
//! owned a held-key set and imported winit) and the app (wheel-to-speed and
//! right-button cursor capture), with no shared per-frame state.
//!
//! [`Input`] concentrates that state. Movement polls [`Input::down`] and
//! reads [`Input::scroll_delta`]; look reads [`Input::mouse_motion`];
//! cursor capture toggles on the right-button press edge (held state in
//! [`Input::buttons_down`]); close and the render-mode toggle read
//! [`Input::just_pressed`]. The winit-to-semantic mapping
//! ([map_key] / [map_button]) lives here at the app boundary — nothing else
//! imports winit input types.

use std::collections::HashSet;

use winit::{
    event::MouseButton,
    keyboard::{Key, NamedKey},
};

/// The semantic action a key maps to (winit-free; the mapping lives in
/// [map_key]). Consumers read the action, not a keycode, so key bindings are
/// confined to one place at the app boundary.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InputKey {
    /// Move forward (Z).
    Forward,
    /// Move backward (S).
    Backward,
    /// Move left (Q).
    Left,
    /// Move right (D).
    Right,
    /// Move up (Space).
    Up,
    /// Move down (Control).
    Down,
    /// Toggle the Render mode (Tab)
    ToggleRenderMode,
    /// Raise the composite's manual exposure (]) — +0.5 EV per press.
    ExposureUp,
    /// Lower the composite's manual exposure ([) — -0.5 EV per press.
    ExposureDown,
    /// Request the app to close (Escape).
    Close,
}

/// The semantic mouse button (winit-free; the mapping lives in [map_button]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InputButton {
    Left,
    Middle,
    Right,
}

/// The app-owned per-frame input state. Held state ([Input::down],
/// [Input::buttons_down]) persists across frames; the edge state
/// ([Input::just_pressed], [Input::scroll_delta], [Input::mouse_motion]) is
/// drained at the end of each frame by [Input::end_frame].
#[derive(Default)]
pub struct Input {
    /// Keys held down this frame — movement polls this (continuous).
    pub down: HashSet<InputKey>,
    /// Keys pressed this frame (an edge) — close/toggle read this;
    /// drained by [Input::end_frame].
    pub just_pressed: HashSet<InputKey>,
    /// This-frame wheel delta (+ = up) — movement reads this for speed.
    pub scroll_delta: f32,
    /// This-frame mouse-motion delta — look reads this.
    pub mouse_motion: (f64, f64),
    /// Mouse buttons held down this frame — cursor capture toggles on the
    /// right-button press edge (Right).
    pub buttons_down: HashSet<InputButton>,
}

impl Input {
    /// Drains the per-frame edge state (just-pressed keys, scroll delta,
    /// mouse motion) at the end of a frame. Held state is untouched.
    pub fn end_frame(&mut self) {
        self.just_pressed.clear();
        self.scroll_delta = 0.0;
        self.mouse_motion = (0.0, 0.0);
    }
}

/// Maps a winit key to its semantic action, or `None` for keys the app does
/// not bind. This is the one place winit keys become [InputKey].
pub fn map_key(key: &Key) -> Option<InputKey> {
    match key {
        Key::Character(ch) => match ch.as_str() {
            "z" => Some(InputKey::Forward),
            "s" => Some(InputKey::Backward),
            "q" => Some(InputKey::Left),
            "d" => Some(InputKey::Right),
            "=" => Some(InputKey::ExposureUp),
            ")" => Some(InputKey::ExposureDown),
            _ => None,
        },
        Key::Named(NamedKey::Space) => Some(InputKey::Up),
        Key::Named(NamedKey::Control) => Some(InputKey::Down),
        Key::Named(NamedKey::Escape) => Some(InputKey::Close),
        Key::Named(NamedKey::Tab) => Some(InputKey::ToggleRenderMode),
        _ => None,
    }
}

/// Maps a winit mouse button to its semantic button, or `None` for buttons
/// the app does not bind. This is the one place winit buttons become
/// [InputButton].
pub fn map_button(button: MouseButton) -> Option<InputButton> {
    match button {
        MouseButton::Left => Some(InputButton::Left),
        MouseButton::Middle => Some(InputButton::Middle),
        MouseButton::Right => Some(InputButton::Right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::SmolStr;

    /// A press lands in both the held set and the edge set; the edge is
    /// drained at end of frame while the held key survives until release.
    #[test]
    fn held_vs_just_pressed_set_semantics() {
        let mut input = Input::default();

        input.down.insert(InputKey::Forward);
        input.just_pressed.insert(InputKey::Forward);
        assert!(input.down.contains(&InputKey::Forward));
        assert!(input.just_pressed.contains(&InputKey::Forward));

        // End of frame drains the edge; the held key remains.
        input.end_frame();
        assert!(input.down.contains(&InputKey::Forward));
        assert!(!input.just_pressed.contains(&InputKey::Forward));

        // Release removes the held key.
        input.down.remove(&InputKey::Forward);
        assert!(!input.down.contains(&InputKey::Forward));
    }

    /// end_frame clears exactly the per-frame edge state (just-pressed keys,
    /// scroll delta, mouse motion) and leaves held state intact.
    #[test]
    fn end_frame_drains_per_frame_state() {
        let mut input = Input::default();
        input.down.insert(InputKey::Forward);
        input.just_pressed.insert(InputKey::Close);
        input.scroll_delta = 3.0;
        input.mouse_motion = (10.0, -5.0);
        input.buttons_down.insert(InputButton::Right);

        input.end_frame();

        assert!(input.just_pressed.is_empty());
        assert_eq!(input.scroll_delta, 0.0);
        assert_eq!(input.mouse_motion, (0.0, 0.0));
        // Held state is not drained.
        assert!(input.down.contains(&InputKey::Forward));
        assert!(input.buttons_down.contains(&InputButton::Right));
    }

    /// The winit key-to-action mapping: every bound key maps to its action,
    /// and unbound keys map to None.
    #[test]
    fn maps_winit_keys_to_actions() {
        let char_key = |s: &str| Key::Character(SmolStr::new(s));

        assert_eq!(map_key(&char_key("z")), Some(InputKey::Forward));
        assert_eq!(map_key(&char_key("s")), Some(InputKey::Backward));
        assert_eq!(map_key(&char_key("q")), Some(InputKey::Left));
        assert_eq!(map_key(&char_key("d")), Some(InputKey::Right));
        assert_eq!(map_key(&char_key("=")), Some(InputKey::ExposureUp));
        assert_eq!(map_key(&char_key(")")), Some(InputKey::ExposureDown));
        assert_eq!(map_key(&Key::Named(NamedKey::Space)), Some(InputKey::Up));
        assert_eq!(
            map_key(&Key::Named(NamedKey::Control)),
            Some(InputKey::Down)
        );
        assert_eq!(
            map_key(&Key::Named(NamedKey::Escape)),
            Some(InputKey::Close)
        );
        assert_eq!(
            map_key(&Key::Named(NamedKey::Tab)),
            Some(InputKey::ToggleRenderMode)
        );

        // Unbound keys are ignored.
        assert_eq!(map_key(&char_key("a")), None);
        assert_eq!(map_key(&char_key("w")), None);
        assert_eq!(map_key(&Key::Named(NamedKey::Enter)), None);
    }

    /// The winit button-to-action mapping: left/middle/right map, side and
    /// "other" buttons do not.
    #[test]
    fn maps_winit_buttons_to_actions() {
        assert_eq!(map_button(MouseButton::Left), Some(InputButton::Left));
        assert_eq!(map_button(MouseButton::Middle), Some(InputButton::Middle));
        assert_eq!(map_button(MouseButton::Right), Some(InputButton::Right));
        assert_eq!(map_button(MouseButton::Back), None);
        assert_eq!(map_button(MouseButton::Forward), None);
        assert_eq!(map_button(MouseButton::Other(1)), None);
    }
}
