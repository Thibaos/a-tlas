use std::collections::HashSet;

use winit::{
    event::MouseButton,
    keyboard::{Key, NamedKey},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InputKey {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
    ToggleRenderMode,
    Close,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InputButton {
    Left,
    Middle,
    Right,
}

#[derive(Default)]
pub struct Input {
    pub down: HashSet<InputKey>,
    pub just_pressed: HashSet<InputKey>,
    pub scroll_delta: f32,
    pub mouse_motion: (f64, f64),
    pub buttons_down: HashSet<InputButton>,
}

impl Input {
    pub fn clear(&mut self) {
        self.just_pressed.clear();
        self.scroll_delta = 0.0;
        self.mouse_motion = (0.0, 0.0);
    }
}

pub fn map_key(key: &Key) -> Option<InputKey> {
    match key {
        Key::Character(ch) => match ch.as_str() {
            "z" => Some(InputKey::Forward),
            "s" => Some(InputKey::Backward),
            "q" => Some(InputKey::Left),
            "d" => Some(InputKey::Right),
            _ => None,
        },
        Key::Named(NamedKey::Space) => Some(InputKey::Up),
        Key::Named(NamedKey::Control) => Some(InputKey::Down),
        Key::Named(NamedKey::Escape) => Some(InputKey::Close),
        Key::Named(NamedKey::Tab) => Some(InputKey::ToggleRenderMode),
        _ => None,
    }
}

pub fn map_mouse_button(button: MouseButton) -> Option<InputButton> {
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

    #[test]
    fn held_vs_just_pressed_set_semantics() {
        let mut input = Input::default();

        input.down.insert(InputKey::Forward);
        input.just_pressed.insert(InputKey::Forward);
        assert!(input.down.contains(&InputKey::Forward));
        assert!(input.just_pressed.contains(&InputKey::Forward));

        input.clear();
        assert!(input.down.contains(&InputKey::Forward));
        assert!(!input.just_pressed.contains(&InputKey::Forward));

        input.down.remove(&InputKey::Forward);
        assert!(!input.down.contains(&InputKey::Forward));
    }

    #[test]
    fn end_frame_clears_per_frame_state() {
        let mut input = Input::default();
        input.down.insert(InputKey::Forward);
        input.just_pressed.insert(InputKey::Close);
        input.scroll_delta = 3.0;
        input.mouse_motion = (10.0, -5.0);
        input.buttons_down.insert(InputButton::Right);

        input.clear();

        assert!(input.just_pressed.is_empty());
        assert_eq!(input.scroll_delta, 0.0);
        assert_eq!(input.mouse_motion, (0.0, 0.0));

        assert!(input.down.contains(&InputKey::Forward));
        assert!(input.buttons_down.contains(&InputButton::Right));
    }

    #[test]
    fn maps_winit_keys_to_actions() {
        let char_key = |s: &str| Key::Character(SmolStr::new(s));

        assert_eq!(map_key(&char_key("z")), Some(InputKey::Forward));
        assert_eq!(map_key(&char_key("s")), Some(InputKey::Backward));
        assert_eq!(map_key(&char_key("q")), Some(InputKey::Left));
        assert_eq!(map_key(&char_key("d")), Some(InputKey::Right));
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

        assert_eq!(map_key(&char_key("w")), None);
        assert_eq!(map_key(&Key::Named(NamedKey::Enter)), None);
    }

    #[test]
    fn maps_winit_buttons_to_actions() {
        assert_eq!(map_mouse_button(MouseButton::Left), Some(InputButton::Left));
        assert_eq!(
            map_mouse_button(MouseButton::Middle),
            Some(InputButton::Middle)
        );
        assert_eq!(
            map_mouse_button(MouseButton::Right),
            Some(InputButton::Right)
        );
        assert_eq!(map_mouse_button(MouseButton::Back), None);
        assert_eq!(map_mouse_button(MouseButton::Forward), None);
        assert_eq!(map_mouse_button(MouseButton::Other(1)), None);
    }
}
