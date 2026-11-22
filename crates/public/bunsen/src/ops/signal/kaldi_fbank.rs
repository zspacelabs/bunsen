//! Kaldi-compatible filterbank features.

use alloc::{
    string::ToString,
    vec::Vec,
};

use burn::{
    prelude::*,
    tensor::signal::rfft,
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::signal::kernels::window::{
        SamplingWindowBuilder,
        StftWindowConfig,
    },
};

/// Common meta for [`KaldiFbankConfig`] and [`KaldiFbank`].
pub trait KaldiFbankMeta {
    /// A frame's length, in samples.
    fn frame_len(&self) -> usize;

    /// The shift between frames, in samples.
    fn frame_shift(&self) -> usize;

    /// The FFT size: the frame length, rounded up to a power of two.
    fn fft_size(&self) -> usize;

    /// The mel bins, `M`.
    fn num_mel_bins(&self) -> usize;

    /// The frames `samples` samples give: `1 + (samples - frame_len) /
    /// frame_shift`, the whole frames (Kaldi's `snip_edges`); 0 for fewer
    /// samples than a frame.
    fn frames_for(
        &self,
        samples: usize,
    ) -> usize {
        if samples < self.frame_len() {
            0
        } else {
            1 + (samples - self.frame_len()) / self.frame_shift()
        }
    }
}

/// Config for [`KaldiFbank`]: Kaldi's `compute-fbank-feats`, as
/// `torchaudio.compliance.kaldi.fbank` computes it.
///
/// The defaults are `torchaudio`'s, and the window has none: Kaldi's default
/// (Povey's) is not one this offers, so a config names its window.
///
/// Kaldi's options this does not offer: dither (its default is random noise;
/// this is dither 0, as a deterministic front end sets it), the raw energy
/// and its floor (fbank's energy column is off by default), VTLN warping,
/// frames past the signal's edges (`snip_edges` false), and an FFT that is
/// not a power of two (`round_to_power_of_two` false).
///
/// Implements [`KaldiFbankMeta`].
#[derive(Config, Debug)]
pub struct KaldiFbankConfig {
    /// The analysis window over a frame, symmetric as Kaldi's are: Kaldi's
    /// `hamming` is [`StftWindowConfig::Hamming`], `hanning`
    /// [`StftWindowConfig::Hann`], `blackman` [`StftWindowConfig::Blackman`]
    /// (each with `periodic: false`), `rectangular`
    /// [`StftWindowConfig::Ones`].
    pub window: StftWindowConfig,

    /// The signal's sample rate, in Hz.
    #[config(default = "16000.0")]
    pub sample_rate: f64,

    /// A frame's length, in milliseconds; the samples are truncated to a
    /// whole number.
    #[config(default = "25.0")]
    pub frame_length_ms: f64,

    /// The shift between frames, in milliseconds; truncated as the length
    /// is.
    #[config(default = "10.0")]
    pub frame_shift_ms: f64,

    /// The mel bins, `M`; at least 4.
    #[config(default = "23")]
    pub num_mel_bins: usize,

    /// The lowest bin's lower edge, in Hz.
    #[config(default = "20.0")]
    pub low_freq: f64,

    /// The highest bin's upper edge, in Hz; a value of 0 or less is an
    /// offset from the Nyquist frequency.
    #[config(default = "0.0")]
    pub high_freq: f64,

    /// The pre-emphasis coefficient, in `[0, 1]`: each frame's sample `n`
    /// less this times sample `n - 1` (the first, less itself).
    #[config(default = "0.97")]
    pub preemphasis: f64,

    /// Whether each frame's mean is removed before the pre-emphasis.
    #[config(default = "true")]
    pub remove_dc_offset: bool,

    /// The power spectrum (`true`) or the magnitude.
    #[config(default = "true")]
    pub use_power: bool,

    /// The log of each bin's energy, floored at `f32::EPSILON`, or the
    /// energy.
    #[config(default = "true")]
    pub use_log: bool,

    /// Whether each bin's mean over the frames is subtracted
    /// (`torchaudio`'s `subtract_mean`: per-utterance cepstral mean
    /// normalization).
    #[config(default = "false")]
    pub subtract_mean: bool,
}

impl KaldiFbankMeta for KaldiFbankConfig {
    fn frame_len(&self) -> usize {
        (self.sample_rate * self.frame_length_ms * 0.001) as usize
    }

    fn frame_shift(&self) -> usize {
        (self.sample_rate * self.frame_shift_ms * 0.001) as usize
    }

    fn fft_size(&self) -> usize {
        self.frame_len().max(1).next_power_of_two()
    }

    fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }
}

/// Kaldi's mel scale: `1127 ln(1 + f / 700)`.
fn kaldi_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

impl KaldiFbankConfig {
    /// The bins' upper edge, in Hz: [`high_freq`](Self::high_freq), or the
    /// Nyquist frequency plus it when it is 0 or less.
    pub fn high_freq_hz(&self) -> f64 {
        if self.high_freq <= 0.0 {
            self.high_freq + 0.5 * self.sample_rate
        } else {
            self.high_freq
        }
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause: a sample rate that is not positive and
    /// finite; a frame length or shift of no samples; fewer than 4 mel bins;
    /// a low or high edge outside `[0, nyquist]`, or a low edge not below the
    /// high; a pre-emphasis outside `[0, 1]`.
    pub fn validate(&self) -> BunsenResult<()> {
        const OWNER: &str = "KaldiFbankConfig";
        if !(self.sample_rate.is_finite() && self.sample_rate > 0.0) {
            return Err(ConstraintError::out_of_range(
                OWNER,
                "sample_rate",
                self.sample_rate,
                "the positive rates",
            )
            .into());
        }
        if self.frame_len() == 0 {
            return Err(ConstraintError::zero_or_empty(OWNER, "frame_length_ms").into());
        }
        if self.frame_shift() == 0 {
            return Err(ConstraintError::zero_or_empty(OWNER, "frame_shift_ms").into());
        }
        if self.num_mel_bins < 4 {
            return Err(ConstraintError::out_of_range(
                OWNER,
                "num_mel_bins",
                self.num_mel_bins,
                "4 and more",
            )
            .into());
        }
        let nyquist = 0.5 * self.sample_rate;
        if !(0.0..nyquist).contains(&self.low_freq) {
            return Err(ConstraintError::out_of_range(
                OWNER,
                "low_freq",
                self.low_freq,
                "[0, nyquist)",
            )
            .into());
        }
        let high = self.high_freq_hz();
        if !(high > 0.0 && high <= nyquist) {
            return Err(ConstraintError::out_of_range(
                OWNER,
                "high_freq",
                self.high_freq,
                "(0, nyquist], or (-nyquist, 0] as an offset from it",
            )
            .into());
        }
        if self.low_freq >= high {
            return Err(ConstraintError::new(
                OWNER,
                "",
                Rule::Relation {
                    lhs: ("low_freq".into(), self.low_freq.to_string()),
                    op: "<",
                    rhs: ("high_freq".into(), high.to_string()),
                },
            )
            .into());
        }
        if !(0.0..=1.0).contains(&self.preemphasis) {
            return Err(ConstraintError::out_of_range(
                OWNER,
                "preemphasis",
                self.preemphasis,
                "[0, 1]",
            )
            .into());
        }
        Ok(())
    }

    /// The mel filterbank, `[fft_size / 2 + 1, M]` row-major: bin `k`'s
    /// weight in mel bin `m` at `k * M + m`.
    ///
    /// Each mel bin is a triangle on Kaldi's mel scale, its edges equally
    /// spaced between the low and high edges, over each FFT bin's center
    /// frequency; the Nyquist bin weighs nothing in any.
    pub fn mel_banks(&self) -> Vec<f64> {
        let (bins, fft_bins) = (self.num_mel_bins, self.fft_size() / 2);
        let (mel_low, mel_high) = (kaldi_mel(self.low_freq), kaldi_mel(self.high_freq_hz()));
        let delta = (mel_high - mel_low) / (bins + 1) as f64;
        let bin_width = self.sample_rate / self.fft_size() as f64;
        let mut banks = vec![0.0; (fft_bins + 1) * bins];
        for k in 0..fft_bins {
            let mel = kaldi_mel(bin_width * k as f64);
            for m in 0..bins {
                let left = mel_low + m as f64 * delta;
                let (center, right) = (left + delta, left + 2.0 * delta);
                let up = (mel - left) / (center - left);
                let down = (right - mel) / (right - center);
                banks[k * bins + m] = up.min(down).max(0.0);
            }
        }
        banks
    }
}

impl ModuleInit<KaldiFbank> for KaldiFbankConfig {
    /// Builds a [`KaldiFbank`] on `device`: the window and the filterbank.
    ///
    /// # Errors
    /// See [`validate`](KaldiFbankConfig::validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<KaldiFbank> {
        self.validate()?;
        let frame_len = self.frame_len();
        let to_f32 = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<_>>();
        let window = Tensor::from_data(
            TensorData::new(to_f32(self.window.to_vec_window(frame_len)), [frame_len]),
            device,
        );
        let banks = Tensor::from_data(
            TensorData::new(
                to_f32(self.mel_banks()),
                [self.fft_size() / 2 + 1, self.num_mel_bins],
            ),
            device,
        );
        Ok(KaldiFbank {
            frame_shift: self.frame_shift(),
            fft_size: self.fft_size(),
            preemphasis: self.preemphasis,
            remove_dc_offset: self.remove_dc_offset,
            use_power: self.use_power,
            use_log: self.use_log,
            subtract_mean: self.subtract_mean,
            window,
            banks,
        })
    }
}

/// Kaldi-compatible filterbank features of whole signals: frames, each with
/// its mean removed, pre-emphasized and windowed, then each frame's spectrum
/// through a mel filterbank, logged.
///
/// Built by [`KaldiFbankConfig`]. Implements [`KaldiFbankMeta`].
///
/// # Module semantics
///
/// A `Module` over **bare tensors, never `Param`s**: the window and the
/// filterbank move with `to_device`, are not recorded (they are rebuilt by
/// [`try_init`](KaldiFbankConfig::try_init)), and a `ModuleMapper` pass
/// (dtype casting, quantization) does not see them.
#[derive(Module, Debug)]
pub struct KaldiFbank {
    frame_shift: usize,
    fft_size: usize,
    preemphasis: f64,
    remove_dc_offset: bool,
    use_power: bool,
    use_log: bool,
    subtract_mean: bool,

    /// The analysis window, `[frame_len]`.
    pub window: Tensor<1>,

    /// The mel filterbank, `[fft_size / 2 + 1, M]`.
    pub banks: Tensor<2>,
}

impl KaldiFbankMeta for KaldiFbank {
    fn frame_len(&self) -> usize {
        self.window.dims()[0]
    }

    fn frame_shift(&self) -> usize {
        self.frame_shift
    }

    fn fft_size(&self) -> usize {
        self.fft_size
    }

    fn num_mel_bins(&self) -> usize {
        self.banks.dims()[1]
    }
}

impl KaldiFbank {
    /// The features of `signal`, `[B, N]`: `[B, F, M]`, `F`
    /// [`frames_for(N)`](KaldiFbankMeta::frames_for). Each row is a signal
    /// of its own; samples past the last whole frame are not read.
    ///
    /// # Panics
    /// When a signal is shorter than a frame.
    pub fn forward(
        &self,
        signal: Tensor<2>,
    ) -> Tensor<3> {
        let [batch, samples] = crate::contracts::unpack_shape_contract!(
            ["batch", "samples"],
            &signal,
            &["batch", "samples"],
        );
        let (frame_len, shift) = (self.frame_len(), self.frame_shift);
        let frames = self.frames_for(samples);
        assert!(
            frames > 0,
            "KaldiFbank: a signal of {samples} samples is shorter than a frame ({frame_len})"
        );

        // [batch, frames, frame_len]; the trim keeps `unfold` to whole frames.
        let covered = (frames - 1) * shift + frame_len;
        let x = signal
            .slice_dim(1, 0..covered as isize)
            .unfold(1, frame_len, shift);
        let x = if self.remove_dc_offset {
            x.clone() - x.mean_dim(2)
        } else {
            x
        };
        let x = if self.preemphasis != 0.0 {
            let previous = Tensor::cat(
                vec![
                    x.clone().slice_dim(2, 0..1),
                    x.clone().slice_dim(2, 0..(frame_len - 1) as isize),
                ],
                2,
            );
            x - previous * self.preemphasis
        } else {
            x
        };
        let x = x * self.window.clone().reshape([1, 1, frame_len]);

        // [batch, frames, fft_size / 2 + 1]
        let (re, im) = rfft(x, 2, Some(self.fft_size));
        let power = re.clone() * re + im.clone() * im;
        let spectrum = if self.use_power { power } else { power.sqrt() };

        let bins = self.fft_size / 2 + 1;
        let mel = spectrum
            .reshape([batch * frames, bins])
            .matmul(self.banks.clone())
            .reshape([batch, frames, self.num_mel_bins()]);
        let mel = if self.use_log {
            mel.clamp_min(f32::EPSILON).log()
        } else {
            mel
        };
        if self.subtract_mean {
            mel.clone() - mel.mean_dim(1)
        } else {
            mel
        }
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

    fn fixture() -> ToyFixture {
        ToyFixture::load_io(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("testdata/kaldi_fbank/fbank.safetensors"),
            &performance_device(),
        )
    }

    fn zonos3() -> KaldiFbankConfig {
        KaldiFbankConfig::new(StftWindowConfig::Hamming { periodic: false }).with_num_mel_bins(80)
    }

    /// A config and the fbank it builds answer the meta alike.
    #[test]
    fn test_meta() {
        let config = KaldiFbankConfig::new(StftWindowConfig::Hann { periodic: false })
            .with_sample_rate(8000.0)
            .with_frame_length_ms(20.0)
            .with_frame_shift_ms(8.0)
            .with_num_mel_bins(40);
        let fbank = config.init(&performance_device());
        for meta in [&config as &dyn KaldiFbankMeta, &fbank] {
            assert_eq!(meta.frame_len(), 160);
            assert_eq!(meta.frame_shift(), 64);
            assert_eq!(meta.fft_size(), 256);
            assert_eq!(meta.num_mel_bins(), 40);
            assert_eq!(meta.frames_for(4800), 73);
            assert_eq!(meta.frames_for(160), 1);
            assert_eq!(meta.frames_for(159), 0);
        }
        assert_eq!(fbank.banks.dims(), [129, 40]);
    }

    /// Each config's refusals.
    #[test]
    fn test_validate() {
        assert!(zonos3().validate().is_ok());
        for bad in [
            zonos3().with_sample_rate(0.0),
            zonos3().with_frame_length_ms(0.01),
            zonos3().with_frame_shift_ms(0.0),
            zonos3().with_num_mel_bins(3),
            zonos3().with_low_freq(8000.0),
            zonos3().with_high_freq(9000.0),
            zonos3().with_high_freq(-8000.0),
            zonos3().with_low_freq(7000.0).with_high_freq(-1500.0),
            zonos3().with_preemphasis(1.5),
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        assert_eq!(zonos3().with_high_freq(-200.0).high_freq_hz(), 7800.0);
        assert_eq!(zonos3().with_high_freq(6000.0).high_freq_hz(), 6000.0);
    }

    /// Every bank is a triangle that peaks below one and covers some bin;
    /// the Nyquist bin weighs nothing.
    #[test]
    fn test_mel_banks() {
        let config = zonos3();
        let (bins, rows) = (config.num_mel_bins, config.fft_size() / 2 + 1);
        let banks = config.mel_banks();
        for m in 0..bins {
            let column: Vec<f64> = (0..rows).map(|k| banks[k * bins + m]).collect();
            let peak = column.iter().copied().fold(0.0, f64::max);
            assert!(peak > 0.0 && peak <= 1.0, "bin {m}: peak {peak}");
        }
        assert!((0..bins).all(|m| banks[(rows - 1) * bins + m] == 0.0));
    }

    /// `torchaudio.compliance.kaldi.fbank`'s answers, computed in float64
    /// (`tools/gen_kaldi_fbank_fixtures.py`). A float32 spectrum rounds
    /// relative to a frame's peak energy, so the quietest bins (about 23
    /// nats below the loudest) carry the error: log features within `2e-3`
    /// absolute, energies within `2e-3` relative. (torchaudio's own float32
    /// run is off by up to `6.6e-4` there; this, measured on wgpu, by
    /// `1.3e-3`.)
    #[test]
    fn test_against_torchaudio() {
        let fixture = fixture();
        let device = fixture.device().clone();
        let log = Tolerance::rel_abs(1e-4, 2e-3);
        let cases = [
            ("zonos3", "signal_16k", zonos3(), log),
            (
                "zonos3_mean",
                "signal_16k",
                zonos3().with_subtract_mean(true),
                log,
            ),
            (
                "variant",
                "signal_8k",
                KaldiFbankConfig::new(StftWindowConfig::Hann { periodic: false })
                    .with_sample_rate(8000.0)
                    .with_frame_length_ms(20.0)
                    .with_frame_shift_ms(8.0)
                    .with_num_mel_bins(40)
                    .with_low_freq(64.0)
                    .with_high_freq(-200.0)
                    .with_preemphasis(0.5)
                    .with_remove_dc_offset(false)
                    .with_use_power(false),
                log,
            ),
            (
                "linear",
                "signal_8k",
                KaldiFbankConfig::new(StftWindowConfig::Blackman { periodic: false })
                    .with_sample_rate(8000.0)
                    .with_frame_length_ms(20.0)
                    .with_frame_shift_ms(8.0)
                    .with_preemphasis(0.0)
                    .with_use_log(false),
                Tolerance::rel_abs(2e-3, 1e-3),
            ),
        ];
        for (name, signal, config, tolerance) in cases {
            let fbank = config.init(&device);
            let actual = fbank.forward(fixture.io::<2>(signal));
            assert_io_close_within(name, actual, fixture.io::<3>(name), tolerance);
        }
    }
}
