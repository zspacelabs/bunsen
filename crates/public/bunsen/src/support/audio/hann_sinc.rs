//! [`HannSincResampler`]: `torchaudio`'s `sinc_interp_hann` resample.

use alloc::vec::Vec;
use core::f64::consts::PI;

use burn::config::Config;

/// A whole-signal rate conversion by a Hann-windowed sinc, as
/// `torchaudio.transforms.Resample` computes it (`sinc_interp_hann`), for a
/// model trained behind it (ZONOS3's speaker encoder).
///
/// The rates are reduced by their greatest common divisor to `in' : out'`.
/// Each of the `out'` output phases has its own kernel over
/// `2 * width + in'` input samples, `width = ceil(lowpass_filter_width *
/// in' / base)`, with `base = min(in', out') * rolloff` the cut-off: the sinc
/// at `base`, under a `cos²` window clamped to `lowpass_filter_width` zero
/// crossings, scaled by `base / in'`. The signal is zero-padded by `width`
/// before and `width + in'` after, and output `j * out' + i` is phase `i`'s
/// kernel over the padded samples from `j * in'`; the output keeps
/// `ceil(n * out / in)` samples.
///
/// The kernels are computed in float64 (each phase's offset in float32, as
/// `torchaudio` divides it) and rounded to float32, as `torchaudio` stores
/// them, and each output is accumulated in float64. So
/// this is `torchaudio`'s float32 kernel applied exactly; `torchaudio`'s
/// float32 convolution is off it by float32's accumulation.
///
/// Unlike [`Resampler`](super::Resampler), this is not streaming, and its
/// output is not phase-aligned to the output rate's grid: it reproduces a
/// reference, sample for sample.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct HannSincResampler {
    /// The kernel's half-width in zero crossings of the sinc; at least 1.
    #[config(default = "6")]
    pub lowpass_filter_width: usize,

    /// The cut-off, as a fraction of the lower rate's Nyquist frequency.
    #[config(default = "0.99")]
    pub rolloff: f64,
}

/// The greatest common divisor of `a` and `b`.
fn gcd(
    a: usize,
    b: usize,
) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// One conversion's kernels: `phases` rows of `taps` each, and the padding
/// before the signal.
#[derive(Clone, Debug, PartialEq)]
pub struct HannSincKernels {
    /// The reduced input rate, `in'`: the stride between output frames.
    pub stride: usize,

    /// The zeros before the signal; `width + stride` follow it.
    pub width: usize,

    /// The kernels, `[phases, taps]` row-major, `phases = out'`, `taps =
    /// 2 * width + stride`.
    pub kernels: Vec<f32>,
}

impl HannSincKernels {
    /// The kernels' length, `2 * width + stride`.
    pub fn taps(&self) -> usize {
        2 * self.width + self.stride
    }

    /// The output phases, `out'`.
    pub fn phases(&self) -> usize {
        self.kernels.len() / self.taps()
    }
}

impl HannSincResampler {
    /// The kernels from `in_rate` to `out_rate`.
    ///
    /// # Panics
    /// If either rate is zero, or the filter width is.
    pub fn kernels(
        &self,
        in_rate: usize,
        out_rate: usize,
    ) -> HannSincKernels {
        assert!(in_rate > 0 && out_rate > 0, "rates must be non-zero");
        assert!(
            self.lowpass_filter_width > 0,
            "lowpass_filter_width must be non-zero"
        );
        let g = gcd(in_rate, out_rate);
        let (orig, new) = (in_rate / g, out_rate / g);
        let lpw = self.lowpass_filter_width as f64;
        let base = orig.min(new) as f64 * self.rolloff;
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
                let window = (t * PI / lpw / 2.0).cos().powi(2);
                let t = t * PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                kernels.push((sinc * window * scale) as f32);
            }
        }
        HannSincKernels {
            stride: orig,
            width,
            kernels,
        }
    }

    /// `input`, a whole signal at `in_rate`, at `out_rate`: `ceil(n *
    /// out_rate / in_rate)` samples; `input` itself when the rates are
    /// equal.
    ///
    /// # Panics
    /// If either rate is zero, or the filter width is.
    pub fn resample(
        &self,
        input: &[f32],
        in_rate: usize,
        out_rate: usize,
    ) -> Vec<f32> {
        if in_rate == out_rate {
            assert!(in_rate > 0, "rates must be non-zero");
            return input.to_vec();
        }
        let kernels = self.kernels(in_rate, out_rate);
        let (stride, width, taps, phases) = (
            kernels.stride,
            kernels.width,
            kernels.taps(),
            kernels.phases(),
        );
        // `ceil(n * new / orig)`, in integers.
        let keep = (input.len() * phases).div_ceil(stride);
        let sample = |m: usize| -> f64 {
            m.checked_sub(width)
                .and_then(|i| input.get(i))
                .map_or(0.0, |&x| x as f64)
        };
        (0..keep)
            .map(|n| {
                let (frame, phase) = (n / phases, n % phases);
                let kernel = &kernels.kernels[phase * taps..(phase + 1) * taps];
                let start = frame * stride;
                kernel
                    .iter()
                    .enumerate()
                    .map(|(k, &w)| w as f64 * sample(start + k))
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

#[cfg(all(test, feature = "store"))]
mod tests {
    use std::path::PathBuf;

    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::{
        ToyFixture,
        assert_io_close_within,
        performance_device,
    };

    /// `torchaudio.transforms.Resample`'s float32 kernels applied exactly
    /// (`tools/gen_resample_fixtures.py`), down and up, at reduced ratios
    /// of 3:2, 441:160, 441:320, 2:3 and 160:441. Bit-equal where measured
    /// (wgpu's host, 2026-11-21); the `1e-6` bound leaves room for another
    /// platform's `sin` and `cos` in the kernels' last bits.
    #[test]
    fn test_against_torchaudio() {
        let fixture = ToyFixture::load_io(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("testdata/resample/hann_sinc.safetensors"),
            &performance_device(),
        );
        let resampler = HannSincResampler::new();
        for (orig, new) in [
            (24000, 16000),
            (44100, 16000),
            (22050, 16000),
            (16000, 24000),
            (16000, 44100),
        ] {
            let case = format!("{orig}_{new}");
            let input = fixture.io::<2>(&format!("{case}.input"));
            let expected = fixture.io::<2>(&format!("{case}.output"));
            let [rows, _] = input.dims();
            let outputs: Vec<Vec<f32>> = (0..rows)
                .map(|r| {
                    let row = input
                        .clone()
                        .slice_dim(0, r..r + 1)
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap();
                    resampler.resample(&row, orig, new)
                })
                .collect();
            let len = outputs[0].len();
            let actual = burn::Tensor::<2>::from_data(
                burn::tensor::TensorData::new(outputs.concat(), [rows, len]),
                fixture.device(),
            );
            assert_io_close_within(&case, actual, expected, Tolerance::rel_abs(1e-6, 1e-6));
        }
    }

    /// The kernels' geometry, as `torchaudio` sizes it: 24 kHz to 16 kHz is
    /// 3:2, a cut-off of `2 * 0.99`, so `ceil(6 * 3 / 1.98) = 10` zeros of
    /// padding and kernels of 23 taps, two phases.
    #[test]
    fn test_kernel_geometry() {
        let kernels = HannSincResampler::new().kernels(24000, 16000);
        assert_eq!(kernels.stride, 3);
        assert_eq!(kernels.width, 10);
        assert_eq!(kernels.taps(), 23);
        assert_eq!(kernels.phases(), 2);
        // Phase 0 is centered on its frame's first sample: the sinc's peak.
        let peak = kernels.kernels[kernels.width];
        assert!(kernels.kernels[..23].iter().all(|&w| w <= peak));
    }

    /// Equal rates pass the signal through; the length rounds up.
    #[test]
    fn test_identity_and_length() {
        let resampler = HannSincResampler::new();
        let x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        assert_eq!(resampler.resample(&x, 16000, 16000), x);
        assert_eq!(resampler.resample(&x[..7], 3, 2).len(), 5);
        assert_eq!(resampler.resample(&x[..7], 2, 3).len(), 11);
    }
}
