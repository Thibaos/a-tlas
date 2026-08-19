//! The renderer: the destination side of the renderer input contract.
//! Everything GPU: the ray pass, the composite, the debug overlays, and
//! the correctness validator that checks the rendered frames against the
//! CPU mirrors.

pub mod accel;
pub mod composite;
#[cfg(debug_assertions)]
pub mod debug;
pub mod region;
pub mod swapchain;
pub mod validate;
