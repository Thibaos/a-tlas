# AGENTS.md

## What this is
Vulkan ray tracing voxel renderer. Loads MagicaVoxel `.vox` files and renders them using hardware ray tracing with a fly camera. Built with Rust + vulkano + winit.

## Build & Run
- `cargo run` — the only command. No args, no env vars, no special profiles.
- Requires a GPU with Vulkan RT support (`khr_ray_tracing_pipeline`, `khr_acceleration_structure`, `khr_synchronization2`).
- The Vulkan SDK must be installed (vulkano-shaders uses `glslc` at build time to compile GLSL to SPIR-V).

## Project structure

```
src/
├── main.rs                    # Entrypoint: winit event loop
├── lib.rs                     # Module declarations
├── app.rs                     # Vulkan init, swapchain, frame loop, event handling
├── async_tlas.rs              # Worker thread for async TLAS rebuild (double-buffered)
├── player.rs                  # Fly camera (AZERTY: ZQSD + Space/Ctrl)
├── schedule.rs                # Periodic task scheduler
├── physics/mod.rs             # Physics tick stub
├── utils.rs                   # Uniform sphere sampling
├── rt/
│   ├── mod.rs                 # Shader module declarations (vulkano_shaders! macros)
│   └── acceleration_structure.rs  # BLAS/TLAS builders
├── tasks/
│   ├── mod.rs
│   ├── render.rs              # Ray tracing render task (vulkano-taskgraph)
│   ├── update_as.rs           # TLAS update task (runs on worker thread)
│   └── debug.rs               # Debug wireframe overlay (debug-only)
└── world/
    ├── mod.rs                 # Vertex types, HostVoxel
    ├── chunk.rs               # 64³ chunk system, Chunks world container
    ├── loader.rs              # .vox scene graph traversal
    └── voxel.rs               # .vox file opening, palette extraction, cube geometry

shaders/
├── rt/                        # GLSL ray tracing stages (compiled by vulkano-shaders)
│   ├── simple.rgen            # Ray generation shader
│   ├── simple.rchit           # Closest-hit shader
│   ├── simple.rmiss           # Miss shader
│   └── simple.rint            # Intersection shader
├── debug/lines/               # Debug wireframe vertex/fragment shaders
├── compute/                   # Legacy GLSL compute shaders (not currently used)
└── slang/                     # Experimental Slang shaders (not used in build)
```

## Key architecture notes

- **Double-buffered TLAS**: Two top-level acceleration structures are alternated between frames. The render task reads the "front" AS while a worker thread rebuilds the "back" AS asynchronously. Synchronized via `AtomicBool`.
- **Chunk system**: World is `64³` chunks, each chunk is `64³` voxels. Chunks are pre-allocated empty; only those with voxels are "active".
- **Renderer**: Single ray tracing pipeline with raygen, miss, procedural hit (intersection + closest-hit) stages. Uses vulkano-taskgraph with bindless descriptors.
- **Debug overlay**: `#[cfg(debug_assertions)]` gates the debug wireframe overlay and chunk boundary rendering. Absent in release builds.
- **max_instance_count** is capped at `device.max_instance_count / 512` to stay within Vulkan instance limits.

## Gotchas

- **vulkano is pulled from git**, not crates.io (`git = "https://github.com/vulkano-rs/vulkano"`). If build breaks, vulkano main branch may have changed.
- **Rust edition 2024** — requires Rust ≥ 1.85.
- **Assets are not tracked**: `assets/` is in `.gitignore`. The code currently hardcodes `assets/castle.vox`. Other .vox files in `assets/` (Church_Of_St_Sophia, custom, nuke, sponza) exist locally but are not committed.
- **`docs/vulkan/`** is a vendored copy of the Vulkan specification — do not modify.
- **`shaders/slang/` and `shaders/compute/`** contain legacy/experimental shaders not wired into the current build.
- **AZERTY keyboard layout**: Movement keys are hardcoded — Z (forward), Q (left), S (backward), D (right), Space (up), Ctrl (down).
- **Mouse capture**: Right-click toggles cursor confinement. Mouse sensitivity is `0.001`.
- **No CI, no formatter/linter config**, no README. Only unit tests exist (in `chunk.rs`).

## Controls
| Action | Key/Mouse |
|--------|-----------|
| Move | Z / Q / S / D |
| Up / Down | Space / Left Ctrl |
| Look | Mouse (when captured) |
| Toggle cursor capture | Right-click |
| Change speed | Mouse wheel |
| Quit | Escape |
