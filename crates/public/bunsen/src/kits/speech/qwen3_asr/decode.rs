//! Greedy decoding of a Qwen3-ASR prompt.
use burn::{
    Tensor,
    config::Config,
};

use super::blocks::Qwen3Asr;
use crate::{
    kits::decoding::{
        GreedySearch,
        decode_cached,
    },
    ops::transformers::attention::KVCacheStorage,
};

/// `Qwen3-ASR-0.6B`'s end tokens: `<|endoftext|>` and `<|im_end|>`.
pub const QWEN3_ASR_EOS: [usize; 2] = [151_643, 151_645];

/// Config for [`Qwen3AsrGreedyDecoder`].
#[derive(Config, Debug)]
pub struct Qwen3AsrGreedyDecoderConfig {
    /// The most tokens one decode generates.
    #[config(default = "512")]
    pub max_new_tokens: usize,

    /// The tokens that end a decode; the last one generated is kept.
    #[config(default = "QWEN3_ASR_EOS.to_vec()")]
    pub eos: Vec<usize>,
}

impl Qwen3AsrGreedyDecoderConfig {
    /// Builds the decoder.
    pub fn init(&self) -> Qwen3AsrGreedyDecoder {
        Qwen3AsrGreedyDecoder {
            config: self.clone(),
        }
    }
}

/// Greedy decoding of a prompt: a cached prefill, then
/// [`decode_cached`] under a [`GreedySearch`] that stops on any of the end
/// tokens, until one or the budget.
///
/// Built by [`Qwen3AsrGreedyDecoderConfig`].
#[derive(Clone, Debug)]
pub struct Qwen3AsrGreedyDecoder {
    config: Qwen3AsrGreedyDecoderConfig,
}

impl Qwen3AsrGreedyDecoder {
    /// The config this was built from.
    pub fn config(&self) -> &Qwen3AsrGreedyDecoderConfig {
        &self.config
    }

    /// Decodes a prompt.
    ///
    /// # Arguments
    /// * `model` - the model.
    /// * `embeds` - `[1, seq, d_text]`, the prompt
    ///   ([`Qwen3Asr::prompt_embeds`]).
    ///
    /// # Returns
    /// The generated tokens, the end token included when one was generated.
    pub fn decode(
        &self,
        model: &Qwen3Asr,
        embeds: Tensor<3>,
    ) -> Vec<usize> {
        let lm = &model.language_model;
        let device = embeds.device();
        let seq = embeds.dims()[1];
        let budget = self.config.max_new_tokens;
        let mut state = lm.decode_state(1, seq + budget, KVCacheStorage::Auto, &device);
        let first = lm.prefill(embeds, &mut state);
        let stops: Vec<i64> = self.config.eos.iter().map(|&t| t as i64).collect();
        // A finished row is fed a filler, which is never stepped for one row.
        let filler = stops.first().copied().unwrap_or(0);
        let mut search = GreedySearch::new(stops.iter().copied(), filler);
        let out = decode_cached(
            lm,
            &mut state,
            vec![Vec::new()],
            first,
            budget,
            &mut search,
            &[],
        );
        out[0][0]
            .tokens_with_stop()
            .into_iter()
            .map(|t| t as usize)
            .collect()
    }
}
