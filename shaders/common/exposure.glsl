// Auto exposure (ADR 0020): composite.comp meters the assembled pre-tonemap
// radiance — per-pixel CIE luminance clamped at LUM_CLAMP, quantized onto a
// 64-bin log-luminance histogram (half-stop bins from 2^-25 up) — so sub-LSB
// dark scenes still count every pixel, where a linear fixed-point accumulator
// would truncate them to zero. exposure_integrate.comp turns the frame
// histogram into a log-space mean, targets KEY / mean, and adapts the stored
// EV exponentially: TAU_BRIGHT when the image gets brighter, TAU_DARK when
// darker, clamped to the EV bounds. An all-sky frame (frame_sky == pixels)
// holds the last EV. Per-frame histogram state is consumed by exchanging prev
// bins, never by zeroing mid-flight.
#define EXPOSURE_KEY 0.18
#define EXPOSURE_LUM_CLAMP 8.0
#define EXPOSURE_BINS 64
#define EXPOSURE_EV_MAX 8.0
#define EXPOSURE_BIN_STEPS 0.5
#define EXPOSURE_EV_MIN -10.0
#define EXPOSURE_BIN_LOG2_MIN (-25.0)
#define EXPOSURE_TAU_BRIGHT 1.5
#define EXPOSURE_TAU_DARK 2.5
