# Ray pass output contract: direct storage-image writes to the swapchain

The ray generation shader writes the primary-visibility result directly to
the presentable swapchain images as bindless storage images
(image usage STORAGE | COLOR_ATTACHMENT), presented with
PresentMode::Immediate; the debug wireframe overlay and any later
diagnostic overlay (measurement heatmap) render into the same image as
subsequent nodes on the same taskgraph, after the ray pass and before
present. There is no intermediate render target and no copy/present step.

The ray parameter range matches the camera: t_min = near (0.01),
t_max = far (10000), so the ray pass clips exactly like the camera.

## Status

accepted (rendering-core ticket 05, 2026-08-10). **Superseded by
[0007](0007-path-trace-output-contract.md)** (path-tracing ticket 02,
2026-08-17) for the app path; retained unchanged for the validator's
capture path.

## Considered Options

- **Render target + copy/present**. Rejected: an extra image + an extra
  copy pass per frame; the swapchain image would still need storage access
  or a blit, and the debug overlay ordering would need a separate resolve
  edge. Direct write keeps one image and one pass.
- **PresentMode::Fifo (vsync)**. Rejected for v1: measurement (ticket 06)
  is GPU-timestamp based and present-independent; Immediate keeps latency
  low and frame pacing in the renderer's hands. Revisit only if tearing
  becomes an issue on the RTX 3070.
- **t_min = EPSILON / t_max = FLT_MAX (ray hits everything)**. Rejected:
  the world is bounded (+-2048), but making the ray range equal the camera's
  near/far keeps primary visibility semantically identical to a camera
  frustum (nothing closer than near, nothing beyond far) for free, and
  gives the CPU reference tracer (ticket 06) one less divergence.

## Consequences

- Swapchain images must keep STORAGE | COLOR_ATTACHMENT usage (already the
  case) and transition General layout for the raygen storage write, then
  Optimal for the overlay's color attachment. The taskgraph edges encode
  this today (RT -> Debug) and any diagnostic overlay joins the chain.
- Barrier correctness between the ray pass and the overlay is expressed as
  a taskgraph edge, not an extra pipeline stage; no resolve/copy stages.
- t_min = near means geometry closer than the near plane is never
  committed; a camera inside a solid voxel renders the enclosing voxel at
  t_min (DDA commits from the clamped entry cell. See ticket 05 Q4, edge
  case for ticket 06).
