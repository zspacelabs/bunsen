//! [`HannSincResampler`]: `torchaudio`'s `sinc_interp_hann` resample.

use alloc::vec::Vec;

use burn::config::Config;

use crate::ops::signal::kernels::{
    sinc::{
        SincPhaseTable,
        WindowedSinc,
    },
    window::CosineWindowConfig,
};

/// A whole-signal rate conversion by a Hann-windowed sinc, as
/// `torchaudio.transforms.Resample` computes it (`sinc_interp_hann`), for a
/// model trained behind it (ZONOS3's speaker encoder).
///
/// The kernels are a Hann-windowed sinc over `lowpass_filter_width` zero
/// crossings, tabulated per output phase of the reduced ratio `in' : out'` as
/// `torchaudio` tabulates them ([`SincPhaseTable::torchaudio`]). The signal is
/// zero-padded by `width` before and `width + in'` after; phase `i`'s output
/// of frame `j` (output `j·out' + i`) is its kernel over the padded samples
/// from `j·in'`, and the output keeps `ceil(n · out / in)` samples. Each
/// output is accumulated in float64, so this is `torchaudio`'s float32 kernel
/// applied exactly; `torchaudio`'s float32 convolution is off it by float32's
/// accumulation.
///
/// Unlike [`StreamingResampler`](super::StreamingResampler), this is not
/// streaming, and its output is not phase-aligned to the output rate's grid:
/// it reproduces a reference, sample for sample.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct HannSincResampler {
    /// The kernel's half-width in zero crossings of the sinc; at least 1.
    #[config(default = "6")]
    pub lowpass_filter_width: usize,

    /// The cut-off, as a fraction of the lower rate's Nyquist frequency.
    #[config(default = "0.99")]
    pub rolloff: f64,
}

impl HannSincResampler {
    /// The windowed-sinc kernel: Hann over `lowpass_filter_width` zero
    /// crossings.
    pub fn kernel(&self) -> WindowedSinc<CosineWindowConfig> {
        WindowedSinc {
            taper: CosineWindowConfig::hann(true),
            zero_crossings: self.lowpass_filter_width as f64,
        }
    }

    /// The kernels from `in_rate` to `out_rate`
    /// ([`SincPhaseTable::torchaudio`]).
    ///
    /// # Panics
    /// If either rate is zero, or the filter width is.
    pub fn kernels(
        &self,
        in_rate: usize,
        out_rate: usize,
    ) -> SincPhaseTable {
        assert!(
            self.lowpass_filter_width > 0,
            "lowpass_filter_width must be non-zero"
        );
        SincPhaseTable::torchaudio(&self.kernel(), self.rolloff, in_rate, out_rate)
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
