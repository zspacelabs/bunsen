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

use burn::{
    Tensor,
    module::Module,
    tensor::{
        TensorCreationOptions,
        TensorData,
        module::conv1d,
        ops::ConvOptions,
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::ops::{
    arange::tensor_arange_start_step,
    signal::kernels::window::WindowTaper,
};

/// The normalized sinc, `sin(πx) / (πx)`; 1 at 0.
pub fn normalized_sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let t = x * PI;
    t.sin() / t
}

/// [`normalized_sinc`] at each of `x`'s values, on its device.
pub fn normalized_sinc_tensor<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let zero = x.clone().equal_elem(0.0);
    let t = x.mul_scalar(PI);
    t.clone().sin().div(t).mask_fill(zero, 1.0)
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

    /// [`at`](Self::at) at each of `t`'s values, on its device.
    pub fn at_tensor<const D: usize>(
        &self,
        t: Tensor<D>,
    ) -> Tensor<D> {
        let window = self
            .taper
            .taper_tensor(t.clone().div_scalar(self.zero_crossings));
        normalized_sinc_tensor(t).mul(window)
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

    /// `torchaudio`'s table ([`torchaudio`](Self::torchaudio)), built on a
    /// device in its float: the grid `t` as one integer, `(k - width) · out' -
    /// i · in'`, over `in' · out'`, exact while it is below 2^24, times
    /// `base`, so `t` carries one rounding.
    ///
    /// It is not the host's table to the bit: the host keeps `torchaudio`'s
    /// float32 phase offset `i / out'` (its rounding times `base`, up to 316.8
    /// at 441 : 320, moves `t`), where this computes the offset exactly. On
    /// wgpu f32 the two are within 7.5e-6.
    ///
    /// # Panics
    /// As [`torchaudio`](Self::torchaudio).
    pub fn torchaudio_tensor<W: WindowTaper>(
        kernel: &WindowedSinc<W>,
        rolloff: f64,
        in_rate: usize,
        out_rate: usize,
        options: impl Into<TensorCreationOptions>,
    ) -> SincPhaseKernels {
        assert!(in_rate > 0 && out_rate > 0, "rates must be non-zero");
        let lpw = kernel.zero_crossings;
        assert!(lpw > 0.0, "the zero crossings must be positive");
        let options = options.into();
        let g = gcd(in_rate, out_rate);
        let (orig, new) = (in_rate / g, out_rate / g);
        let base = orig.min(new) as f64 * rolloff;
        let width = (lpw * orig as f64 / base).ceil() as usize;
        let taps = 2 * width + orig;
        // `(k - width) / in' - i / out'` as one integer over `in' · out'`:
        // exact in the device's float while it is below 2^24, so `t` carries
        // one rounding, not the cancellation of two order-1 terms.
        let shift = tensor_arange_start_step(new, 0.0, Some(-(orig as f64)), options.clone())
            .reshape([new as i32, 1]);
        let idx =
            tensor_arange_start_step(taps, -((width * new) as f64), Some(new as f64), options)
                .reshape([1, taps as i32]);
        let t = shift
            .add(idx)
            .mul_scalar(base / (orig * new) as f64)
            .clamp(-lpw, lpw);
        SincPhaseKernels {
            stride: orig,
            width,
            kernels: kernel.at_tensor(t).mul_scalar(base / orig as f64),
        }
    }

    /// The table on a device, as it is.
    pub fn to_kernels(
        &self,
        options: impl Into<TensorCreationOptions>,
    ) -> SincPhaseKernels {
        let data = TensorData::new(self.kernels.clone(), [self.phases(), self.taps()]);
        SincPhaseKernels {
            stride: self.stride,
            width: self.width,
            kernels: Tensor::from_data(data, options),
        }
    }
}

/// A [`SincPhaseTable`] on a device, and its polyphase convolution.
///
/// The kernels are a bare tensor, not a `Param`: they move with
/// `to_device` but are not recorded.
#[derive(Module, Debug)]
pub struct SincPhaseKernels {
    /// The reduced input rate, `in'`: the stride between output frames.
    pub stride: usize,

    /// The zeros before the signal; `width + stride` follow it.
    pub width: usize,

    /// The kernels, `[phases, taps]`.
    pub kernels: Tensor<2>,
}

impl SincPhaseKernels {
    /// The output phases, `out'`.
    pub fn phases(&self) -> usize {
        self.kernels.dims()[0]
    }

    /// `input`, `[B, N]` at the table's input rate, at its output rate:
    /// `[B, ceil(N · out' / in')]`.
    ///
    /// The signal is zero-padded by `width` before and `width + stride`
    /// after, and convolved with each phase's kernel at `stride`: phase `i`'s
    /// output of frame `j` is output `j·out' + i`, as `torchaudio` lays them.
    pub fn resample(
        &self,
        input: Tensor<2>,
    ) -> Tensor<2> {
        let [batch, n] = input.dims();
        let phases = self.phases();
        let weight = self.kernels.clone().unsqueeze_dim::<3>(1);
        let options = ConvOptions {
            stride: [self.stride],
            padding: [(self.width, self.width + self.stride)],
            dilation: [1],
            groups: 1,
        };
        let out = conv1d(input.unsqueeze_dim::<3>(1), weight, None, options);
        let frames = out.dims()[2];
        let keep = (n * phases).div_ceil(self.stride);
        out.swap_dims(1, 2)
            .reshape([batch as i32, (frames * phases) as i32])
            .narrow(1, 0, keep)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        ops::signal::kernels::window::CosineWindowConfig,
        support::testing::{
            assert_tensor_close_to_vec,
            performance_device,
            tensor_to_vec_as,
        },
    };

    /// The most a device-built kernel may stray from the host's, absolute.
    /// Measured on wgpu f32 (2026-11-22): 7.5e-6 at 441 : 320 (the host's
    /// float32 phase offset, times the largest `base`).
    const KERNEL_ABS: f32 = 2e-5;

    fn hann(zero_crossings: f64) -> WindowedSinc<CosineWindowConfig> {
        WindowedSinc {
            taper: CosineWindowConfig::hann(true),
            zero_crossings,
        }
    }

    /// The device's sinc and windowed sinc are the host's, element-wise.
    #[test]
    fn test_tensor_sinc_is_the_host_sinc() {
        let device = performance_device();
        let ts: Vec<f64> = (-700..=700).map(|i| i as f64 / 100.0).collect();
        let t = Tensor::<1>::from_data(ts.as_slice(), &device);
        let sinc: Vec<f64> = ts.iter().map(|&t| normalized_sinc(t)).collect();
        let tol = Tolerance::rel_abs(1e-5, 1e-6);
        assert_tensor_close_to_vec(&normalized_sinc_tensor(t.clone()), &sinc, tol);
        let kernel = hann(6.0);
        let windowed: Vec<f64> = ts.iter().map(|&t| kernel.at(t)).collect();
        assert_tensor_close_to_vec(&kernel.at_tensor(t), &windowed, tol);
    }

    /// `torchaudio`'s table built on a device is the host's, at each of the
    /// fixture ratios; lifted as it is, it is the host's exactly.
    #[test]
    fn test_torchaudio_tensor_is_the_host_table() {
        let device = performance_device();
        for (orig, new) in [
            (24000, 16000),
            (44100, 16000),
            (22050, 16000),
            (16000, 24000),
            (16000, 44100),
        ] {
            let host = SincPhaseTable::torchaudio(&hann(6.0), 0.99, orig, new);
            let built = SincPhaseTable::torchaudio_tensor(&hann(6.0), 0.99, orig, new, &device);
            assert_eq!((built.stride, built.width), (host.stride, host.width));
            assert_eq!(built.kernels.dims(), [host.phases(), host.taps()]);
            let worst = tensor_to_vec_as::<f32, 2>(&built.kernels)
                .iter()
                .zip(&host.kernels)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            eprintln!("{orig} -> {new}: the device's table within {worst:.2e} of the host's");
            assert!(worst <= KERNEL_ABS, "{orig} -> {new}: {worst}");
            let lifted = host.to_kernels(&device);
            assert_eq!(tensor_to_vec_as::<f32, 2>(&lifted.kernels), host.kernels);
        }
    }

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
        let kernel = hann(6.0);
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
        let table = SincPhaseTable::torchaudio(&hann(6.0), 0.99, 24000, 16000);
        assert_eq!(table.stride, 3);
        assert_eq!(table.width, 10);
        assert_eq!(table.taps(), 23);
        assert_eq!(table.phases(), 2);
        let peak = table.kernels[table.width];
        assert!(table.kernels[..23].iter().all(|&w| w <= peak));
    }
}
