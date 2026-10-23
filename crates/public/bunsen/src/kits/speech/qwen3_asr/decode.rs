//! Greedy decoding of a Qwen3-ASR prompt.
use burn::{
    Tensor,
    config::Config,
    prelude::{
        Int,
        TensorData,
    },
};

use super::blocks::Qwen3Asr;
use crate::ops::transformers::attention::KVCacheModel;

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

/// Greedy decoding of a prompt: a cached prefill, then one cached step per
/// token, the argmax each time, until an end token or the budget.
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
        let total = seq + self.config.max_new_tokens;
        let r_emb = lm.rotary(total, &device);
        // The rotary table holds `total` positions, so the cache does too.
        let mut cache = lm
            .kv_cache_config()
            .with_seq_len(total)
            .with_max_seq_len(Some(total))
            .init();

        let argmax_last = |hidden: Tensor<3>| -> usize {
            let n = hidden.dims()[1];
            lm.logits(hidden.slice_dim(1, n - 1..n))
                .argmax(2)
                .into_scalar::<i64>() as usize
        };
        let mut token = argmax_last(lm.forward_hidden(embeds, &r_emb, Some(&mut cache)));
        let mut out = vec![token];
        while !self.config.eos.contains(&token) && out.len() < self.config.max_new_tokens {
            let ids =
                Tensor::<2, Int>::from_data(TensorData::new(vec![token as i64], [1, 1]), &device);
            token = argmax_last(lm.forward_hidden(lm.embed(ids), &r_emb, Some(&mut cache)));
            out.push(token);
        }
        out
    }
}
