# Path-trace output contract: radiance pair + aux set + composite

Under the path-tracing output contract, the Region ray pass no longer writes the swapchain directly. In Voxel mode the trace pass writes a noisy radiance pair and an auxiliary buffer set (the Denoise pass's inputs, per the NRD REBLUR research, ADR 0007's basis, ticket 01); a Composite node exposes the radiance to the swapchain with manual EV exposure and the ACES tonemap. The debug Render modes (Hull, Ray latency, hull-crossed) are unchanged in behavior: they paint the swapchain storage image directly from the raygen, and the composite no-ops for them.

The trace pass's per-pixel output set (all full-res, window-sized):

- **Diffuse radiance + hit distance**. RGBA16F; rgb = de-modulated diffuse radiance, a = normalized in-lobe hit distance (the DDA's first-bounce t).
- **Specular radiance + hit distance**. RGBA16F; same shape, specular lobe.
- **Normal + roughness**. RGBA8; octahedral world normal + linear roughness (sqrt(m)); diffuse pixels carry roughness 1.0.
- **ViewZ**. R32F; linear view-space Z of the primary hit.
- **Motion vectors**. RGBA16F; backward-pointing screen-space motion in pixels per frame, 2.5D (z = viewZ_prev - viewZ). Computed from prev view-proj (plumbing lands with the Denoise pass).
- **Albedo + metalness**. RGBA8; the palette albedo + MATL metalness (ticket 03). NRD itself does not read albedo/metalness. The composite re-modulates the denoised radiance with them (diffuse × albedo; specular via F0/envBRDF).

Emission rides inside the de-modulated diffuse radiance (the de-modulation
divide is guarded against zero albedo), so emissive voxels are denoised with
the diffuse signal. There is no separate emission channel (matches the
map's "no emissive NEE; emission via path hits").

Sky/Background pixels are radiance written into the diffuse buffer with a viewZ beyond NRD's denoisingRange (excluded from denoising, passed through the composite), per the ticket-01 research.

## Status

accepted (path-tracing ticket 02, 2026-08-17). Supersedes 0002. The validator's capture path (0002's direct-write shape) was removed with the validation teardown (2026-08-27).

## Considered Options

- **Single combined Beauty buffer**. Rejected: NRD REBLUR requires diffuse and specular radiance separated and de-modulated at the primary hit; a combined buffer would need a prepass split before denoising.
- **Combined Beauty + pair**. Rejected: extra memory + a write per pixel; the composite can derive a combined view from the pair when needed.
- **Separate emission channel**. Rejected: black+emissive materials would avoid the eps-guard divide, but emission denoises fine inside the diffuse signal and a separate channel adds a buffer + a composite add.
- **Debug modes through the composite**. Rejected: a dedicated preview buffer + copy per frame; the existing pattern (the heatmap's no-op) lets the debug modes keep writing the swapchain directly with the composite no-oping on `mode != Voxel`.
- **Auto exposure**. Deferred: manual EV is the default (map defaults); auto folds in later if the scene demands it.

## Consequences

- The production raygen writes six storage images in Voxel mode (the push constant block grows by six bindless ids).
- The composite node sits after the render node (and the hull-crossed heatmap overlay in debug builds), last before present; its reads are `IMAGE_LAYOUT_GENERAL` like the heatmap's, so no layout transitions are introduced.
- The trace-pass images are virtual graph resources mapped to physical images per frame (the same mechanism as the virtual swapchain), so window resize destroys and recreates the physical images without rebuilding the graph.
- Exposure is manual EV and the tonemap is the ACES filmic fit (Narkowicz 2015); both live in the composite shader, so ticket 08 (the Denoise pass) and later tuning touch one place.
- The `--measure` flight bracket still covers only the render node's work. The composite runs after `FLIGHT_END`. Whether the composite joins the bracket or gets its own per-stage slot is ticket 09's measurement-revision decision.
- Slice contents land over the next tickets: viewZ is real from the slice on; normal/roughness, motion vectors, metalness, and the radiance values are placeholders until tickets 03/04/05/08 fill them (the composite currently shows the palette-color stub through exposure + tonemap).
