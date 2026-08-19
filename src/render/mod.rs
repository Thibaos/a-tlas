//! The renderer: the destination side of the renderer input contract.
//! Everything GPU — the ray pass, the composite, the debug overlays, the
//! timing contract, and the correctness validator that checks the rendered
//! frames against the CPU mirrors.

pub mod accel;
pub mod composite;
#[cfg(debug_assertions)]
pub mod debug;
pub mod measure;
pub mod region;
pub mod swapchain;
pub mod validate;
