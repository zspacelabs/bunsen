//! # The step sampler
//!
//! [`Zonos3Sampler`]: one row's codes from its logits, every codebook
//! sampled at once, as the release samples them.

use alloc::vec::Vec;

use burn::{
    Tensor,
    config::Config,
};

use crate::kits::support::decoding::{
    CodeSampler,
    MinP,
    RepetitionPenalty,
    SuppressTokens,
    TopK,
    TopP,
};

/// The lowest temperature [`Zonos3Sampler`] samples at: a lower one, zero
/// included, is raised to it, so the sampler always draws (upstream's).
pub const ZONOS3_MIN_TEMPERATURE: f64 = 1e-6;

/// Config for [`Zonos3Sampler`]; its defaults are the release's
/// (`config.json`'s `generation`,
/// [`Zonos3GenerationMirror::sampler_config`](super::interop::zonos::Zonos3GenerationMirror::sampler_config)).
#[derive(Config, Debug, PartialEq)]
pub struct Zonos3SamplerConfig {
    /// The temperature, at least [`ZONOS3_MIN_TEMPERATURE`].
    #[config(default = 0.75)]
    pub temperature: f64,

    /// Codes kept per codebook, the most probable.
    #[config(default = 80)]
    pub top_k: usize,

    /// The probability mass kept per codebook ([`TopP`]).
    #[config(default = 0.69)]
    pub top_p: f64,

    /// The fraction of the top probability a code must reach ([`MinP`]);
    /// zero keeps every code.
    #[config(default = 0.0)]
    pub min_p: f64,

    /// The repetition penalty ([`RepetitionPenalty`]); one is none.
    #[config(default = 2.4)]
    pub repetition_penalty: f64,

    /// The generated rows the penalty looks back over.
    #[config(default = 96)]
    pub repetition_window: usize,

    /// The codebooks the penalty covers, the first ones.
    #[config(default = 6)]
    pub repetition_codebooks: usize,
}

impl Zonos3SamplerConfig {
    /// The sampler, for codebooks of `codebook_size` codes followed by the
    /// end of audio (`eoa_id`) and pads up to `audio_vocab`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`](crate::errors::ConstraintError), if the end of
    /// audio is not past the codes and in the vocabulary, or the repetition
    /// penalty is not positive.
    pub fn try_init(
        &self,
        codebook_size: usize,
        eoa_id: usize,
        audio_vocab: usize,
    ) -> crate::errors::BunsenResult<Zonos3Sampler> {
        use crate::errors::ConstraintError;
        if !(codebook_size..audio_vocab).contains(&eoa_id) {
            return Err(ConstraintError::out_of_range(
                "Zonos3SamplerConfig::try_init",
                "eoa_id",
                eoa_id,
                alloc::format!("{codebook_size}..{audio_vocab}: past the codes, in the vocabulary"),
            )
            .into());
        }
        if self.repetition_penalty.is_nan() || self.repetition_penalty <= 0.0 {
            return Err(ConstraintError::out_of_range(
                "Zonos3SamplerConfig",
                "repetition_penalty",
                self.repetition_penalty,
                "> 0",
            )
            .into());
        }
        Ok(self.init_unchecked(codebook_size, eoa_id, audio_vocab))
    }

    /// [`try_init`](Self::try_init), panicking with its error's report.
    ///
    /// # Panics
    /// As `try_init`'s errors.
    pub fn init(
        &self,
        codebook_size: usize,
        eoa_id: usize,
        audio_vocab: usize,
    ) -> Zonos3Sampler {
        use crate::errors::WithOkOrPanic;
        self.try_init(codebook_size, eoa_id, audio_vocab)
            .ok_or_panic()
    }

    fn init_unchecked(
        &self,
        codebook_size: usize,
        eoa_id: usize,
        audio_vocab: usize,
    ) -> Zonos3Sampler {
        let temperature = self.temperature.max(ZONOS3_MIN_TEMPERATURE);
        let eoa = eoa_id as i64;
        Zonos3Sampler {
            code: CodeSampler {
                suppress: Some(SuppressTokens::new(
                    (codebook_size as i64..audio_vocab as i64).filter(|&id| id != eoa),
                )),
                penalty: Some(
                    RepetitionPenalty::new(self.repetition_penalty)
                        .with_window(self.repetition_window)
                        .with_exempt(codebook_size as i64..audio_vocab as i64),
                ),
                penalized_rows: Some(self.repetition_codebooks),
                top_k: Some(TopK { k: self.top_k }),
                top_p: Some(TopP {
                    p: self.top_p,
                    temperature,
                }),
                min_p: (self.min_p > 0.0).then_some(MinP {
                    p: self.min_p,
                    temperature,
                }),
                temperature,
            },
        }
    }
}

/// One row's codes from its logits, as the release samples them, every
/// codebook independently, in this order:
///
/// 1. every id past the codes but the end of audio is never sampled (the pad);
///    the end of audio stays sampleable in every codebook;
/// 2. the repetition penalty, over the first
///    [`repetition_codebooks`](Zonos3SamplerConfig::repetition_codebooks), on
///    the codes they generated within the window (the end of audio and the pad
///    never count);
/// 3. top-k, then top-p and min-p at the temperature;
/// 4. one draw per codebook at the temperature
///    ([`sample_softmax`](crate::kits::support::decoding::sample_softmax)), on
///    the backend's random numbers (`device.seed(..)` repeats a take).
///
/// Top-p is [`TopP`]'s, the standard nucleus: the release's sampler also
/// keeps a code whose preceding mass is exactly `p`, which this does not.
/// The draws are not torch's either, so a take matches the release's in
/// distribution, never code for code.
#[derive(Clone, Debug)]
pub struct Zonos3Sampler {
    code: CodeSampler,
}

impl Zonos3Sampler {
    /// The temperature it samples at.
    pub fn temperature(&self) -> f64 {
        self.code.temperature
    }

    /// One row's codes, one per codebook.
    ///
    /// # Arguments
    /// - `logits`: `[K, V_a]`, the row's.
    /// - `history`: per codebook, the codes it generated so far, oldest first
    ///   (the prompt's rows not among them); a codebook past its end has none.
    pub fn sample(
        &self,
        logits: Tensor<2>,
        history: &[Vec<i64>],
    ) -> Vec<i64> {
        self.code
            .choose(logits, history, &[])
            .into_data()
            .try_into_vec_as::<i64>()
            .expect("the drawn ids")
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::cpu_device;

    /// Codebooks of 4 codes, the end of audio 4, the pad 5.
    fn sampler(config: &Zonos3SamplerConfig) -> Zonos3Sampler {
        config.init(4, 4, 6)
    }

    fn logits(rows: &[[f32; 6]]) -> Tensor<2> {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Tensor::from_data(TensorData::new(flat, [rows.len(), 6]), &cpu_device())
    }

    #[test]
    fn test_defaults_are_the_release() {
        let config = Zonos3SamplerConfig::new();
        assert_eq!(
            (
                config.temperature,
                config.top_k,
                config.top_p,
                config.min_p,
                config.repetition_penalty,
                config.repetition_window,
                config.repetition_codebooks,
            ),
            (0.75, 80, 0.69, 0.0, 2.4, 96, 6)
        );
        let raised = sampler(&Zonos3SamplerConfig::new().with_temperature(0.0));
        assert_eq!(raised.temperature(), ZONOS3_MIN_TEMPERATURE);
    }

    /// An end of audio among the codes or past the vocabulary, or a penalty
    /// that is not positive, is refused by name.
    #[test]
    fn test_try_init_refusals() {
        let config = Zonos3SamplerConfig::new();
        for eoa in [3, 6] {
            crate::errors::testing::ErrorMatcher::illegal()
                .constraint("Zonos3SamplerConfig::try_init", "eoa_id")
                .assert_err(&config.try_init(4, eoa, 6).map(|_| ()));
        }
        crate::errors::testing::ErrorMatcher::illegal()
            .constraint("Zonos3SamplerConfig", "repetition_penalty")
            .assert_err(
                &config
                    .clone()
                    .with_repetition_penalty(0.0)
                    .try_init(4, 4, 6)
                    .map(|_| ()),
            );
    }

    #[test]
    fn test_pad_never_eoa_always() {
        let s = sampler(&Zonos3SamplerConfig::new().with_temperature(4.0));
        let device = cpu_device();
        device.seed(11);
        let mut saw_eoa = false;
        for _ in 0..32 {
            let ids = s.sample(logits(&[[0.0, 0.0, 0.0, 0.0, 0.0, 50.0]]), &[]);
            assert_ne!(ids[0], 5, "the pad is never sampled");
            saw_eoa |= ids[0] == 4;
        }
        assert!(saw_eoa, "the end of audio is sampleable");
    }

    #[test]
    fn test_repetition_penalty_covers_the_first_codebooks() {
        // Code 0 just over code 1 in all three codebooks; each generated
        // code 0 before. Top-k 1 makes the draw the argmax.
        let s = sampler(
            &Zonos3SamplerConfig::new()
                .with_top_k(1)
                .with_repetition_codebooks(2),
        );
        let row = [1.0, 0.9, -1.0, -1.0, -5.0, -5.0];
        let history = vec![vec![0], vec![0], vec![0]];
        assert_eq!(s.sample(logits(&[row, row, row]), &history), [1, 1, 0]);

        // Outside the window, code 0 is not penalized.
        let s = sampler(
            &Zonos3SamplerConfig::new()
                .with_top_k(1)
                .with_repetition_window(1),
        );
        assert_eq!(s.sample(logits(&[row]), &[vec![0, 2]]), [0]);
    }
}
