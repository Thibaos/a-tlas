# Auto exposure

Interiors in this world meter at demodulated luminance ~1e-4 while the sky
through the same frame's apertures sits at ~1e-1, and both ends ride on fixed
sun/sky radiance constants. A single manual EV calibrated for one regime makes
the other unviewable, so the Composite replaces the manual EV with an adapted
one, measured from the frame's own assembled pre-tonemap radiance.

## Status

accepted (2026-09-01, built and verified in-session on the 3070 harness:
histogram counts full-rate every frame, EV converges and plateaus stable over
2k+ frames at 60-77 fps with no measurable meter cost).

## Decision

- **Meter the assembled display radiance, post-denoise, pre-tonemap.** The
  same value the viewer sees drives the adaptation; the denoiser already
  removes firefly noise the meter must not chase, and NRD's exposure coupling
  is not fed (its CommonSettings.exposure stays fixed, so the meter's EV
  change cannot loop back into the denoiser's normalization).
- **Histogram, not accumulator.** Per-pixel CIE luminance is clamped at
  LUM_CLAMP (above the brightest legitimate surface, below the Sun disk) and
  binned onto a 64-bin log-luminance histogram at half-stop steps from 2^-25.
  A linear fixed-point accumulator (`uint(lum·2048+0.5)`) truncates this
  world's interior luminances to zero; the histogram counts every pixel, and
  each deposit is a constant-valued `1u` atomic.
- **One unconditional `atomicExchange` per field per frame.** The integrate
  pass (1 thread) reads the histogram, computes the log-space mean, targets
  KEY / mean, adapts EV exponentially (TAU_BRIGHT 1.5 s brightening, TAU_DARK
  2.5 s dimming), clamps to [EV_MIN, EV_MAX] = [-10, +8], and consumes the
  histogram by exchanging bins to zero and prev_sky to the running sky count.
  Non-atomic stores to this buffer and conditional guarded writes proved
  unreliable on this driver; every buffer write is an integer atomic with a
  locally computed value.
- **All-sky hold.** A frame whose sky fraction equals its pixel count holds
  the last EV (star-field views carry no meterable surface; the meter must
  not chase the background).
- **The Composite reads the EV from the buffer, not a push constant.** The
  display uses the same frame's adapted EV with zero CPU round-trip; the
  swapchain is 8-bit so the tonemapped value is dithered by interleaved
  gradient noise.

## Considered Options

- CPU readback of the adapted EV and a push-constant display. Rejected: adds
  a host round-trip per frame and a one-frame display lag for no benefit.
- Meter the raw pre-denoise beauty pair. Rejected: 1 spp fireflies would
  drag the meter upward between NRD clamping boundaries.
- Feed NRD's CommonSettings.exposure with the adapted EV. Rejected: makes
  the denoiser's internal thresholds move with the meter, a feedback loop
  for no measured benefit; NRD operates on absolute-radiance input.

## Consequences

- The dark-interior saturation at EV_MAX (+8) is deliberate: beyond it the
  1 spp noise floor amplifies visibly. Moving outdoors the same frame's
  meter pulls EV down through TAU_DARK; the transition is exponential, not
  a cut.
- The meter's EV is also the tonemap's only exposure knob; manual EV
  control (an offset on top of the adapted value) remains unimplemented
  until asked for.
