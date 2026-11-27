//! # The samplers
//!
//! [`Qwen3TtsSampler`]: a code from a row of logits, as upstream's
//! `generate` chooses one, for the talker's first codebook and for the code
//! predictor's.

use alloc::vec::Vec;

use burn::{
    Tensor,
    config::Config,
    prelude::Int,
};

use crate::kits::support::decoding::{
    CodeSampler,
    RepetitionPenalty,
    SuppressTokens,
    TopK,
    TopP,
};

/// Config for [`Qwen3TtsSampler`]; its defaults are the release's talker's.
#[derive(Config, Debug, PartialEq)]
pub struct Qwen3TtsSamplerConfig {
    /// Whether it samples; else it takes the argmax (upstream's
    /// `do_sample = false`), and the temperature and truncations are not
    /// read.
    #[config(default = true)]
    pub sample: bool,

    /// The temperature; above zero when it samples.
    #[config(default = 0.9)]
    pub temperature: f64,

    /// Codes kept, the most probable.
    #[config(default = 50)]
    pub top_k: usize,

    /// The probability mass kept ([`TopP`]); one keeps every code top-k
    /// does.
    #[config(default = 1.0)]
    pub top_p: f64,

    /// The repetition penalty over the codes it chose ([`RepetitionPenalty`]);
    /// one is none.
    #[config(default = 1.0)]
    pub repetition_penalty: f64,
}

impl Qwen3TtsSamplerConfig {
    /// The sampler, never choosing `suppressed`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`](crate::errors::ConstraintError), if the repetition
    /// penalty is not positive.
    pub fn try_init(
        &self,
        suppressed: impl IntoIterator<Item = i64>,
    ) -> crate::errors::BunsenResult<Qwen3TtsSampler> {
        if self.repetition_penalty.is_nan() || self.repetition_penalty <= 0.0 {
            return Err(crate::errors::ConstraintError::out_of_range(
                "Qwen3TtsSamplerConfig",
                "repetition_penalty",
                self.repetition_penalty,
                "> 0",
            )
            .into());
        }
        Ok(self.init_unchecked(suppressed))
    }

    /// [`try_init`](Self::try_init), panicking with its error's report.
    ///
    /// # Panics
    /// As `try_init`'s errors.
    pub fn init(
        &self,
        suppressed: impl IntoIterator<Item = i64>,
    ) -> Qwen3TtsSampler {
        use crate::errors::WithOkOrPanic;
        self.try_init(suppressed).ok_or_panic()
    }

    fn init_unchecked(
        &self,
        suppressed: impl IntoIterator<Item = i64>,
    ) -> Qwen3TtsSampler {
        let suppressed: Vec<i64> = suppressed.into_iter().collect();
        Qwen3TtsSampler {
            code: CodeSampler {
                suppress: (!suppressed.is_empty()).then(|| SuppressTokens::new(suppressed)),
                penalty: (self.repetition_penalty != 1.0)
                    .then(|| RepetitionPenalty::new(self.repetition_penalty)),
                penalized_rows: None,
                top_k: self.sample.then_some(TopK { k: self.top_k }),
                top_p: (self.sample && self.top_p < 1.0).then_some(TopP {
                    p: self.top_p,
                    temperature: self.temperature,
                }),
                min_p: None,
                temperature: if self.sample { self.temperature } else { 0.0 },
            },
        }
    }
}

/// A code from a row of logits, as `transformers`' `generate` chooses one
/// under upstream's settings, in this order:
///
/// 1. the suppressed ids, `-inf` (the talker's control rows but EOS);
/// 2. the repetition penalty over the codes chosen so far (the talker's);
/// 3. the blocked ids, `-inf`, for this choice alone (EOS before the take's
///    first frames);
/// 4. sampling: top-k, then top-p, then a draw at the temperature
///    ([`sample_softmax`](crate::kits::support::decoding::sample_softmax), on
///    the backend's random numbers); or the argmax.
///
/// Its [`CodeSampler`] runs the chain.
///
/// The draws are not torch's, so a sampled take matches upstream's in
/// distribution, never code for code; an argmax take is upstream's.
#[derive(Clone, Debug)]
pub struct Qwen3TtsSampler {
    code: CodeSampler,
}

impl Qwen3TtsSampler {
    /// The temperature it samples at; zero for the argmax.
    pub fn temperature(&self) -> f64 {
        self.code.temperature
    }

    /// A code per row.
    ///
    /// # Arguments
    /// - `logits`: `[B, V]`.
    /// - `history`: per row, the codes chosen before, oldest first.
    /// - `blocked`: ids never chosen this time.
    ///
    /// # Returns
    /// `[B]`.
    pub fn choose(
        &self,
        logits: Tensor<2>,
        history: &[Vec<i64>],
        blocked: &[i64],
    ) -> Tensor<1, Int> {
        self.code.choose(logits, history, blocked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// The argmax, after the suppressed ids, the penalty and the blocked
    /// ones.
    #[test]
    fn test_argmax_order() {
        let device = cpu_device();
        let sampler = Qwen3TtsSamplerConfig::new()
            .with_sample(false)
            .with_repetition_penalty(2.0)
            .init([4]);
        let logits = || Tensor::<2>::from_floats([[1.0, 3.0, 2.5, 0.0, 9.0]], &device);
        let choose = |history: &[Vec<i64>], blocked: &[i64]| {
            sampler
                .choose(logits(), history, blocked)
                .into_data()
                .iter::<i64>()
                .next()
                .expect("an id")
        };
        assert_eq!(choose(&[vec![]], &[]), 1);
        // 3 / 2 < 2.5
        assert_eq!(choose(&[vec![1]], &[]), 2);
        assert_eq!(choose(&[vec![1]], &[2]), 1);
    }
}
