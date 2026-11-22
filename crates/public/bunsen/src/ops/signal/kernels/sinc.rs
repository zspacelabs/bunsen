//! Windowed-sinc low-pass kernels: [`WindowedSinc`], and the phase tables a
//! rational rate conversion applies ([`SincPhaseTable`]).
//!
//! A low-pass filter with its cut-off at `fc` (a fraction of a rate's
//! Nyquist frequency) has the impulse response `fc · sinc(fc · u)`, `u` in
//! that rate's samples; truncated to a half-width, it is windowed by a
//! [`WindowTaper`] over that half-width, so the truncation does not ring.
//! In the sinc's own units, `t = fc · u` (its zero crossings at the
//! integers), the kernel is `sinc(t) · taper(t / zero_crossings)`.
//!
//! The rate converters in [`ops::signal::resample`](crate::ops::signal::resample)
//! differ in how they place the kernel, not in the kernel:
//! [`HannSincResampler`](crate::ops::signal::resample::HannSincResampler)
//! (`torchaudio`'s `sinc_interp_hann`) tabulates it under Hann at each output
//! phase of a reduced ratio ([`SincPhaseTable::torchaudio`]);
//! [`StreamingResampler`](crate::ops::signal::resample::StreamingResampler)
//! evaluates it under Blackman at each output's exact position, over `16 /
//! cutoff` input samples, and normalizes each output's taps to unit DC gain.

use alloc::vec::Vec;
use core::f64::consts::PI;

// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::ops::signal::kernels::window::WindowTaper;

/// The normalized sinc, `sin(πx) / (πx)`; 1 at 0.
pub fn normalized_sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let t = x * PI;
    t.sin() / t
}

/// A windowed-sinc kernel in the sinc's own units: `sinc(t) · taper(t /
/// zero_crossings)`, zero beyond `zero_crossings` either side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowedSinc<W> {
    /// The window over the kernel's half-width.
    pub taper: W,

    /// The half-width, in the sinc's zero crossings.
    pub zero_crossings: f64,
}

impl<W: WindowTaper> WindowedSinc<W> {
    /// The kernel at `t`, in the sinc's zero crossings from its centre.
    pub fn at(
        &self,
        t: f64,
    ) -> f64 {
        normalized_sinc(t) * self.taper.taper(t / self.zero_crossings)
    }
}

/// The greatest common divisor of `a` and `b`.
fn gcd(
    a: usize,
    b: usize,
) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// One rational conversion's kernels: `phases` rows of `taps` each, and the
/// padding before the signal.
#[derive(Clone, Debug, PartialEq)]
pub struct SincPhaseTable {
    /// The reduced input rate, `in'`: the stride between output frames.
    pub stride: usize,

    /// The zeros before the signal; `width + stride` follow it.
    pub width: usize,

    /// The kernels, `[phases, taps]` row-major, `phases = out'`, `taps =
    /// 2 * width + stride`.
    pub kernels: Vec<f32>,
}

impl SincPhaseTable {
    /// The kernels' length, `2 * width + stride`.
    pub fn taps(&self) -> usize {
        2 * self.width + self.stride
    }

    /// The output phases, `out'`.
    pub fn phases(&self) -> usize {
        self.kernels.len() / self.taps()
    }

    /// `torchaudio`'s table from `in_rate` to `out_rate`
    /// (`torchaudio.functional.resample`'s `_get_sinc_resample_kernel`), the
    /// window `kernel.taper` over `kernel.zero_crossings` (its
    /// `lowpass_filter_width`).
    ///
    /// The rates are reduced by their greatest common divisor to `in' :
    /// out'`; the cut-off is `base = min(in', out') · rolloff`, in `in'`'s
    /// units; the half-width is `width = ceil(zero_crossings · in' / base)`
    /// input samples. Phase `i`'s tap `k` is the kernel at `t = (k - width) /
    /// in' - i / out'`, times `base`, clamped to the zero crossings, scaled by
    /// `base / in'`: computed in float64 (the phase offset `i / out'` in
    /// float32, as `torchaudio` divides it) and rounded to float32, as
    /// `torchaudio` stores it.
    ///
    /// # Panics
    /// If either rate is zero, or the zero crossings are not positive.
    pub fn torchaudio<W: WindowTaper>(
        kernel: &WindowedSinc<W>,
        rolloff: f64,
        in_rate: usize,
        out_rate: usize,
    ) -> Self {
        assert!(in_rate > 0 && out_rate > 0, "rates must be non-zero");
        let lpw = kernel.zero_crossings;
        assert!(lpw > 0.0, "the zero crossings must be positive");
        let g = gcd(in_rate, out_rate);
        let (orig, new) = (in_rate / g, out_rate / g);
        let base = orig.min(new) as f64 * rolloff;
        let width = (lpw * orig as f64 / base).ceil() as usize;
        let taps = 2 * width + orig;
        let scale = base / orig as f64;
        let mut kernels = Vec::with_capacity(new * taps);
        for phase in 0..new {
            // `torchaudio` divides the phase in float32 (an int64 `arange`
            // over an int is float32's default dtype), then widens it.
            let shift = f64::from(-(phase as f32) / new as f32);
            for k in 0..taps {
                let idx = (k as f64 - width as f64) / orig as f64;
                let t = ((shift + idx) * base).clamp(-lpw, lpw);
                kernels.push((kernel.at(t) * scale) as f32);
            }
        }
        Self {
            stride: orig,
            width,
            kernels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::signal::kernels::window::CosineWindowConfig;

    /// The sinc is 1 at 0, zero at the other integers, and even.
    #[test]
    fn test_normalized_sinc() {
        assert_eq!(normalized_sinc(0.0), 1.0);
        for n in 1..8 {
            assert!(normalized_sinc(n as f64).abs() < 1e-15);
            assert_eq!(
                normalized_sinc(-0.3 - n as f64),
                normalized_sinc(0.3 + n as f64)
            );
        }
        assert!((normalized_sinc(0.5) - 2.0 / PI).abs() < 1e-15);
    }

    /// The windowed kernel is the sinc at its centre, and zero at and beyond
    /// its half-width under Hann.
    #[test]
    fn test_windowed_sinc() {
        let kernel = WindowedSinc {
            taper: CosineWindowConfig::hann(true),
            zero_crossings: 6.0,
        };
        assert_eq!(kernel.at(0.0), 1.0);
        assert!(kernel.at(6.0).abs() < 1e-30);
        assert_eq!(kernel.at(6.5), 0.0);
        let expected = normalized_sinc(0.5) * (PI / 24.0).cos().powi(2);
        assert!((kernel.at(0.5) - expected).abs() < 1e-15);
    }

    /// The geometry, as `torchaudio` sizes it: 24 kHz to 16 kHz is 3:2, a
    /// cut-off of `2 · 0.99`, so `ceil(6 · 3 / 1.98) = 10` zeros of padding
    /// and kernels of 23 taps, two phases; phase 0 peaks at its frame's first
    /// sample.
    #[test]
    fn test_torchaudio_geometry() {
        let kernel = WindowedSinc {
            taper: CosineWindowConfig::hann(true),
            zero_crossings: 6.0,
        };
        let table = SincPhaseTable::torchaudio(&kernel, 0.99, 24000, 16000);
        assert_eq!(table.stride, 3);
        assert_eq!(table.width, 10);
        assert_eq!(table.taps(), 23);
        assert_eq!(table.phases(), 2);
        let peak = table.kernels[table.width];
        assert!(table.kernels[..23].iter().all(|&w| w <= peak));
    }
}
