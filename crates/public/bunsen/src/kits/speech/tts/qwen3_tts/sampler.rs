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
    tensor::DType,
};
use serde::Deserialize;

use crate::kits::support::decoding::{
    LogitFilter,
    RepetitionPenalty,
    SuppressTokens,
    TopK,
    TopP,
    sample_softmax,
};

/// A model's `generation_config.json`, under upstream's names: the talker's
/// settings, and the code predictor's (`subtalker_*`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsGenerationJson {
    /// Whether the talker samples; else it takes the argmax.
    pub do_sample: bool,
    /// The talker's repetition penalty.
    pub repetition_penalty: f64,
    /// The talker's temperature.
    pub temperature: f64,
    /// The talker's top-p.
    pub top_p: f64,
    /// The talker's top-k.
    pub top_k: usize,
    /// Whether the code predictor samples.
    pub subtalker_dosample: bool,
    /// Its temperature.
    pub subtalker_temperature: f64,
    /// Its top-p.
    pub subtalker_top_p: f64,
    /// Its top-k.
    pub subtalker_top_k: usize,
    /// The most codebook-0 tokens a take generates.
    pub max_new_tokens: usize,
}

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
    /// The talker's settings of a `generation_config.json`.
    pub fn talker(generation: &Qwen3TtsGenerationJson) -> Self {
        Self::new()
            .with_sample(generation.do_sample)
            .with_temperature(generation.temperature)
            .with_top_k(generation.top_k)
            .with_top_p(generation.top_p)
            .with_repetition_penalty(generation.repetition_penalty)
    }

    /// The code predictor's settings of a `generation_config.json`: its own
    /// sampling, no repetition penalty (upstream passes it none).
    pub fn code_predictor(generation: &Qwen3TtsGenerationJson) -> Self {
        Self::new()
            .with_sample(generation.subtalker_dosample)
            .with_temperature(generation.subtalker_temperature)
            .with_top_k(generation.subtalker_top_k)
            .with_top_p(generation.subtalker_top_p)
            .with_repetition_penalty(1.0)
    }

    /// The sampler, never choosing `suppressed`.
    pub fn init(
        &self,
        suppressed: impl IntoIterator<Item = i64>,
    ) -> Qwen3TtsSampler {
        let suppressed: Vec<i64> = suppressed.into_iter().collect();
        Qwen3TtsSampler {
            suppress: (!suppressed.is_empty()).then(|| SuppressTokens::new(suppressed)),
            penalty: (self.repetition_penalty != 1.0)
                .then(|| RepetitionPenalty::new(self.repetition_penalty)),
            truncation: self.sample.then(|| {
                (
                    TopK { k: self.top_k },
                    (self.top_p < 1.0).then_some(TopP {
                        p: self.top_p,
                        temperature: self.temperature,
                    }),
                )
            }),
            temperature: if self.sample { self.temperature } else { 0.0 },
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
///    ([`sample_softmax`], on the backend's random numbers); or the argmax.
///
/// The draws are not torch's, so a sampled take matches upstream's in
/// distribution, never code for code; an argmax take is upstream's.
#[derive(Clone, Debug)]
pub struct Qwen3TtsSampler {
    suppress: Option<SuppressTokens>,
    penalty: Option<RepetitionPenalty>,
    truncation: Option<(TopK, Option<TopP>)>,
    temperature: f64,
}

impl Qwen3TtsSampler {
    /// The temperature it samples at; zero for the argmax.
    pub fn temperature(&self) -> f64 {
        self.temperature
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
        let [b, _] = logits.dims();
        let mut logits = logits.cast(DType::F32);
        if let Some(suppress) = &self.suppress {
            logits = suppress.apply(logits, &[], 0);
        }
        if let Some(penalty) = &self.penalty {
            logits = penalty.apply(logits, history, 0);
        }
        if !blocked.is_empty() {
            logits = SuppressTokens::new(blocked.iter().copied()).apply(logits, &[], 0);
        }
        if let Some((top_k, top_p)) = &self.truncation {
            logits = top_k.apply(logits, &[], 0);
            if let Some(top_p) = top_p {
                logits = top_p.apply(logits, &[], 0);
            }
        }
        sample_softmax(logits, self.temperature).reshape([b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    fn release() -> Qwen3TtsGenerationJson {
        serde_json::from_str(
            r#"{"do_sample": true, "repetition_penalty": 1.05, "temperature": 0.9,
                "top_p": 1.0, "top_k": 50, "subtalker_dosample": true,
                "subtalker_temperature": 0.9, "subtalker_top_p": 1.0,
                "subtalker_top_k": 50, "max_new_tokens": 8192}"#,
        )
        .expect("parse")
    }

    /// The release's settings: the config's defaults for the talker; the
    /// predictor's without a penalty.
    #[test]
    fn test_release_settings() {
        assert_eq!(
            Qwen3TtsSamplerConfig::talker(&release()).repetition_penalty,
            1.05
        );
        assert_eq!(
            Qwen3TtsSamplerConfig::talker(&release()),
            Qwen3TtsSamplerConfig::new().with_repetition_penalty(1.05)
        );
        let predictor = Qwen3TtsSamplerConfig::code_predictor(&release());
        assert_eq!(
            (
                predictor.temperature,
                predictor.top_k,
                predictor.repetition_penalty
            ),
            (0.9, 50, 1.0)
        );
    }

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
