pub mod app;
mod core;
mod render;
mod world;

/// The offline correctness validator (`atlas-rt validate`), re-exported at
/// the crate root.
pub use render::validate;
