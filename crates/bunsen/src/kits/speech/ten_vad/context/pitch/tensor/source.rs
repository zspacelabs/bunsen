//! # The device-side pitch source.
//!
//! Composes the four stages into a [`TenVadPitchSource`]:
//!
//! ```text
//! bin_power ──prefilter──▶ lpc ──┐
//!                                ├──excitation──▶ exc ──correlate──▶ xcorr ──track──▶ Hz
//! raw ───────────────────────────┘
//! ```
//!
//! Only two stages carry state — the excitation branch's filters and FIFO, and
//! the tracker's Viterbi accumulator. The pre-filter design and the lag search
//! are pure functions of their inputs, which is why a whole sequence runs
//! through them in one pass while the other two thread a carry.
//!
//! ## Reference and optimized tiers
//!
//! Both are this type; they differ in one sub-step, the anti-alias filter
//! ([`PitchAntiAliasConfig`]). Selecting [`Recurrence`] gives a literal
//! transcription of the reference's IIR cascade — the correctness reference,
//! sample-sequential and viable only on short inputs. Selecting
//! [`TruncatedFir`] gives the default, which is both faster *and* more accurate
//! than the recurrence.
//!
//! [`Recurrence`]: PitchAntiAliasConfig::Recurrence
//! [`TruncatedFir`]: PitchAntiAliasConfig::TruncatedFir

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    super::{
        coeff::LPC_ORDER,
        source::TenVadPitchSource,
    },
    antialias::PitchAntiAliasConfig,
    correlate::{
        PitchCorrelate,
        PitchCorrelateConfig,
    },
    excitation::{
        PitchExcitation,
        PitchExcitationConfig,
        PitchExcitationState,
    },
    prefilter::{
        PitchPrefilter,
        PitchPrefilterConfig,
    },
    track::{
        PitchTrack,
        PitchTrackConfig,
        PitchTrackState,
    },
};
use crate::errors::{
    BunsenResult,
    WithOkOrPanic,
};

/// Config for [`TensorPitch`].
#[derive(Config, Debug)]
pub struct TensorPitchConfig {
    /// Stage 1: the whitening filter design.
    #[config(default = "PitchPrefilterConfig::new()")]
    pub prefilter: PitchPrefilterConfig,

    /// Stage 2: whitening and decimation.
    #[config(default = "PitchExcitationConfig::new()")]
    pub excitation: PitchExcitationConfig,

    /// Stage 3: the normalized lag search.
    #[config(default = "PitchCorrelateConfig::new()")]
    pub correlate: PitchCorrelateConfig,

    /// Stage 4: period tracking.
    #[config(default = "PitchTrackConfig::new()")]
    pub track: PitchTrackConfig,
}

impl TensorPitchConfig {
    /// Selects how the anti-alias filter is realized.
    ///
    /// The only place the reference and optimized tiers differ, so this is the
    /// knob that chooses between them.
    pub fn with_anti_alias(
        mut self,
        anti_alias: PitchAntiAliasConfig,
    ) -> Self {
        self.excitation = self.excitation.with_anti_alias(anti_alias);
        self
    }

    /// A config selecting the literal-transcription tier.
    ///
    /// Sample-sequential, so this is a correctness reference for short inputs
    /// rather than a workload path.
    pub fn reference() -> Self {
        Self::new().with_anti_alias(PitchAntiAliasConfig::Recurrence)
    }

    /// The number of frequency bins this source expects.
    pub fn n_bins(&self) -> usize {
        self.prefilter.n_bins()
    }

    /// The hop size, in samples at 16 kHz.
    pub fn hop_size(&self) -> usize {
        self.excitation.hop_size
    }

    /// Validates every stage, and that they agree on geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`](crate::errors::BunsenError::Invalid) if any
    /// stage is invalid, or if the excitation history the excitation stage
    /// emits is not the length the lag search reads.
    pub fn validate(&self) -> BunsenResult<()> {
        self.prefilter.validate()?;
        self.excitation.validate()?;
        self.correlate.validate()?;
        self.track.validate()?;

        if self.excitation.exc_len() != self.correlate.exc_len() {
            return Err(crate::errors::BunsenError::Invalid(format!(
                "TensorPitch excitation emits {} samples but the lag search reads {}",
                self.excitation.exc_len(),
                self.correlate.exc_len(),
            )));
        }
        Ok(())
    }

    /// Builds the coefficients.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TensorPitch<B>> {
        self.validate()?;
        Ok(TensorPitch {
            prefilter: self.prefilter.try_init(device)?,
            excitation: self.excitation.try_init(device)?,
            correlate: self.correlate.try_init(device)?,
            track: self.track.try_init(device)?,
        })
    }

    /// Builds the coefficients, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> TensorPitch<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// The device-side pitch estimator's fixed coefficients.
///
/// Stateless, so one instance serves any number of streams; state lives in
/// [`TensorPitchContext`]. Built by [`TensorPitchConfig::try_init`].
#[derive(Debug, Clone)]
pub struct TensorPitch<B: Backend> {
    /// Stage 1.
    pub prefilter: PitchPrefilter<B>,
    /// Stage 2.
    pub excitation: PitchExcitation<B>,
    /// Stage 3.
    pub correlate: PitchCorrelate<B>,
    /// Stage 4.
    pub track: PitchTrack<B>,
}

impl<B: Backend> TensorPitch<B> {
    /// The number of frequency bins this source expects.
    pub fn n_bins(&self) -> usize {
        self.prefilter.n_bins()
    }

    /// The hop size, in samples at 16 kHz.
    pub fn hop_size(&self) -> usize {
        self.excitation.hop_size()
    }

    /// Binds a start-of-stream state over `batch_size` independent streams.
    pub fn init_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> TensorPitchContext<B> {
        TensorPitchContext {
            excitation: self.excitation.init_state(batch_size, device),
            track: self.track.init_state(batch_size, device),
            batch_size,
            coef: self.clone(),
        }
    }
}

/// The device-side pitch estimator's streaming state.
///
/// Implements [`TenVadPitchSource`]. Built by [`TensorPitch::init_state`].
#[derive(Debug, Clone)]
pub struct TensorPitchContext<B: Backend> {
    /// The fixed coefficients.
    pub coef: TensorPitch<B>,

    /// Stage 2's carried buffers.
    pub excitation: PitchExcitationState<B>,

    /// Stage 4's carried accumulator and slot history.
    pub track: PitchTrackState<B>,

    batch_size: usize,
}

impl<B: Backend> TensorPitchContext<B> {
    /// The batch size; each row is an independent stream.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl<B: Backend> TenVadPitchSource<B> for TensorPitchContext<B> {
    fn forward(
        &mut self,
        raw: Tensor<B, 2>,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [batch, hop] = raw.dims();
        let n_bins = bin_power.dims()[1];

        // One hop is just a one-step sequence; the stages are written for the
        // batched form and a `steps` of 1 costs nothing extra.
        let out = self.forward_sequence(
            raw.reshape([1, batch, hop]),
            bin_power.reshape([1, batch, n_bins]),
        );
        out.reshape([batch, 1])
    }

    fn forward_sequence(
        &mut self,
        raw: Tensor<B, 3>,
        bin_power: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [steps, batch, _] = raw.dims();
        let n_bins = bin_power.dims()[2];
        assert_eq!(batch, self.batch_size, "TensorPitch batch mismatch");
        assert_eq!(n_bins, self.coef.n_bins(), "TensorPitch bin count mismatch");

        let rows = steps * batch;

        // Stage 1 carries nothing, so the whole sequence designs in one pass.
        let lpc = self
            .coef
            .prefilter
            .forward(bin_power.reshape([rows, n_bins]))
            .reshape([steps, batch, LPC_ORDER]);

        let (exc, excitation) = self
            .coef
            .excitation
            .forward(raw, lpc, self.excitation.clone());
        self.excitation = excitation;

        // Stage 3 likewise: stateless given the history it is handed.
        let exc_len = self.coef.correlate.exc_len();
        let (xcorr, energy) = self.coef.correlate.forward(exc.reshape([rows, exc_len]));

        let max_period = self.coef.correlate.max_period();
        let subs = super::correlate::SUBS_PER_HOP;
        let (pitch, track) = self.coef.track.forward(
            xcorr.reshape([steps, batch, subs, max_period]),
            energy.reshape([steps, batch, subs]),
            self.track.clone(),
        );
        self.track = track;

        pitch
    }

    fn reset(&mut self) {
        let device = self.track.path_score.device();
        self.excitation = self.coef.excitation.init_state(self.batch_size, &device);
        self.track = self.coef.track.init_state(self.batch_size, &device);
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::{
        super::super::{
            TenVadPitchEstimator,
            TenVadPitchScalarSource,
        },
        *,
    };
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    const HOP: usize = 256;
    const N_BINS: usize = 513;

    fn pulse_hop(
        f0: f32,
        at: usize,
    ) -> Vec<f32> {
        let period = 16000.0 / f0;
        (0..HOP)
            .map(|i| {
                let pos = (at + i) as f32 % period;
                8000.0 * (-pos / (period * 0.08)).exp()
            })
            .collect()
    }

    fn spectrum(step: usize) -> Vec<f32> {
        (0..N_BINS)
            .map(|k| {
                let k = k as f32;
                1e7 * (-k / 70.0).exp() * (1.0 + 0.4 * (k * 0.05 + step as f32 * 0.3).sin())
            })
            .collect()
    }

    /// Drives the host over `steps` hops, returning the inputs it saw and the
    /// pitch it reported.
    fn host_reference(steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut est = TenVadPitchEstimator::new();
        let (mut raw, mut power, mut hz) = (Vec::new(), Vec::new(), Vec::new());
        for step in 0..steps {
            let hop = pulse_hop(150.0, step * HOP);
            let spec = spectrum(step);
            hz.push(est.frame_pitch(&hop, &spec));
            raw.extend_from_slice(&hop);
            power.extend_from_slice(&spec);
        }
        (raw, power, hz)
    }

    fn run(
        cfg: &TensorPitchConfig,
        steps: usize,
        raw: &[f32],
        power: &[f32],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<f32> {
        let coef: TensorPitch<B> = cfg.init(device);
        let mut ctx = coef.init_state(1, device);

        let raw_t = Tensor::<B, 1>::from_floats(raw, device).reshape([steps, 1, HOP]);
        let pow_t = Tensor::<B, 1>::from_floats(power, device).reshape([steps, 1, N_BINS]);

        ctx.forward_sequence(raw_t, pow_t)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap()
    }

    #[test]
    fn test_config_meta() {
        let cfg = TensorPitchConfig::new();
        assert_eq!(cfg.n_bins(), N_BINS);
        assert_eq!(cfg.hop_size(), HOP);
        assert!(cfg.validate().is_ok());
        assert!(TensorPitchConfig::reference().validate().is_ok());
    }

    #[test]
    fn test_reference_tier_selects_the_recurrence() {
        assert_eq!(
            TensorPitchConfig::reference().excitation.anti_alias,
            PitchAntiAliasConfig::Recurrence,
        );
        assert_eq!(
            TensorPitchConfig::new().excitation.anti_alias,
            PitchAntiAliasConfig::default(),
        );
    }

    #[test]
    fn test_validate_rejects_mismatched_stage_geometry() {
        let cfg =
            TensorPitchConfig::new().with_correlate(PitchCorrelateConfig::new().with_hop_size(128));
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_forward_sequence_matches_the_host_estimator() {
        // The whole device path against the whole host path, over the same
        // inputs. This is what the stage-level differential tests add up to.
        let device = Default::default();
        let steps = 16;
        let (raw, power, want) = host_reference(steps);

        let got = run(&TensorPitchConfig::new(), steps, &raw, &power, &device);

        for (t, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(*g > 0.0, *w > 0.0, "hop {t}: voicing disagrees, {g} vs {w}");
            if *w > 0.0 {
                let rel = (g - w).abs() / w;
                assert!(rel < 1e-3, "hop {t}: {g} Hz vs {w} Hz (rel {rel})");
            }
        }
        assert!(
            want.iter().filter(|v| **v > 0.0).count() * 2 > steps,
            "fixture should be mostly voiced",
        );
    }

    #[test]
    fn test_the_two_tiers_agree() {
        // Short, because the reference tier is sample-sequential: five cascade
        // sections stepping one sample at a time.
        let device = Default::default();
        let steps = 3;
        let (raw, power, _) = host_reference(steps);

        let fast = run(&TensorPitchConfig::new(), steps, &raw, &power, &device);
        let exact = run(
            &TensorPitchConfig::reference(),
            steps,
            &raw,
            &power,
            &device,
        );

        for (t, (f, e)) in fast.iter().zip(exact.iter()).enumerate() {
            assert_eq!(*f > 0.0, *e > 0.0, "hop {t}: voicing disagrees, {f} vs {e}");
            if *e > 0.0 {
                let rel = (f - e).abs() / e;
                assert!(rel < 1e-3, "hop {t}: fast {f} Hz vs reference {e} Hz");
            }
        }
    }

    #[test]
    fn test_forward_sequence_matches_stepwise() {
        let device = Default::default();
        let steps = 8;
        let (raw, power, _) = host_reference(steps);
        let cfg = TensorPitchConfig::new();

        let whole = run(&cfg, steps, &raw, &power, &device);

        let coef: TensorPitch<B> = cfg.init(&device);
        let mut ctx = coef.init_state(1, &device);
        let mut stepwise = Vec::new();
        for step in 0..steps {
            let r = Tensor::<B, 1>::from_floats(&raw[step * HOP..(step + 1) * HOP], &device)
                .reshape([1, HOP]);
            let p =
                Tensor::<B, 1>::from_floats(&power[step * N_BINS..(step + 1) * N_BINS], &device)
                    .reshape([1, N_BINS]);
            stepwise.extend(
                ctx.forward(r, p)
                    .to_data_as::<f32>()
                    .to_vec_as::<f32>()
                    .unwrap(),
            );
        }

        TensorData::from(whole.as_slice()).assert_approx_eq::<f32>(
            &TensorData::from(stepwise.as_slice()),
            Tolerance::relative(1e-4),
        );
    }

    #[test]
    fn test_reset_rewinds_the_stream() {
        let device = Default::default();
        let steps = 4;
        let (raw, power, _) = host_reference(steps);
        let cfg = TensorPitchConfig::new();
        let coef: TensorPitch<B> = cfg.init(&device);
        let mut ctx = coef.init_state(1, &device);

        let raw_t = Tensor::<B, 1>::from_floats(raw.as_slice(), &device).reshape([steps, 1, HOP]);
        let pow_t =
            Tensor::<B, 1>::from_floats(power.as_slice(), &device).reshape([steps, 1, N_BINS]);

        let first: Vec<f32> = ctx
            .forward_sequence(raw_t.clone(), pow_t.clone())
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        ctx.reset();
        let again: Vec<f32> = ctx
            .forward_sequence(raw_t, pow_t)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        assert_eq!(first, again);
    }

    #[test]
    fn test_batch_rows_are_independent() {
        let device = Default::default();
        let steps = 6;
        let (raw, power, _) = host_reference(steps);
        let cfg = TensorPitchConfig::new();
        let coef: TensorPitch<B> = cfg.init(&device);

        // Row 1 is silence, which must stay unvoiced whatever row 0 does.
        let mut raw_pair = Vec::new();
        let mut pow_pair = Vec::new();
        for step in 0..steps {
            raw_pair.extend_from_slice(&raw[step * HOP..(step + 1) * HOP]);
            raw_pair.extend(std::iter::repeat_n(0.0f32, HOP));
            pow_pair.extend_from_slice(&power[step * N_BINS..(step + 1) * N_BINS]);
            pow_pair.extend(std::iter::repeat_n(0.0f32, N_BINS));
        }

        let mut ctx = coef.init_state(2, &device);
        let got: Vec<f32> = ctx
            .forward_sequence(
                Tensor::<B, 1>::from_floats(raw_pair.as_slice(), &device).reshape([steps, 2, HOP]),
                Tensor::<B, 1>::from_floats(pow_pair.as_slice(), &device)
                    .reshape([steps, 2, N_BINS]),
            )
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        let solo = run(&cfg, steps, &raw, &power, &device);
        for step in 0..steps {
            assert!(
                (got[step * 2] - solo[step]).abs() < 1e-3,
                "hop {step}: row 0 {} vs solo {}",
                got[step * 2],
                solo[step],
            );
            assert_eq!(got[step * 2 + 1], 0.0, "hop {step}: silent row 1 is voiced");
        }
    }
}
