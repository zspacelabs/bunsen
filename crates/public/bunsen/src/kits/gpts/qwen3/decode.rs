//! Cached decoding of a Qwen3.
use burn::{
    Tensor,
    module::Module,
    prelude::Int,
    tensor::Device,
};

use super::{
    Qwen3Meta,
    Qwen3Model,
};
use crate::{
    blocks::transformers::embedding::RotaryEmbedding,
    kits::decoding::CachedStep,
    ops::transformers::attention::{
        KVCache,
        KVCacheModel,
        KVCacheStorage,
    },
};

/// What a cached Qwen3 decode carries between steps: its KV cache, and the
/// rotary table for every position it may reach.
///
/// Made by [`Qwen3Model::decode_state`], filled by
/// [`Qwen3Model::prefill`], then stepped by
/// [`decode_cached`](crate::kits::decoding::decode_cached) through the
/// model's [`CachedStep`]. The model never owns one, so one loaded model
/// serves any number of decodes.
#[derive(Module, Debug)]
pub struct Qwen3DecodeState {
    /// The KV cache, one batch row per decode row.
    pub cache: KVCache,

    /// The rotary table, covering the cache's capacity.
    pub rotary: RotaryEmbedding,
}

impl Qwen3Model {
    /// A decode state for `rows` rows of at most `positions` positions each,
    /// under `storage`: the cache allocated whole and capped there, as the
    /// rotary table is.
    pub fn decode_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> Qwen3DecodeState {
        Qwen3DecodeState {
            cache: self
                .kv_cache_config()
                .with_batch_size(rows)
                .with_seq_len(positions)
                .with_max_seq_len(Some(positions))
                .with_storage(storage)
                .init(),
            rotary: self.rotary(positions, device),
        }
    }

    /// Runs prompt embeddings through the model into `state`.
    ///
    /// # Arguments
    /// * `embeds` - `[rows, seq, d_model]`: [`embed`](Self::embed)'s tokens, or
    ///   a prompt with other frames in it.
    ///
    /// # Returns
    /// `[rows, vocab]`, the logits that follow the prompt.
    pub fn prefill(
        &self,
        embeds: Tensor<3>,
        state: &mut Qwen3DecodeState,
    ) -> Tensor<2> {
        self.last_logits(self.forward_hidden(embeds, &state.rotary, Some(&mut state.cache)))
    }

    /// The logits of the last position of `hidden`, `[rows, seq, d_model]`.
    fn last_logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<2> {
        let [rows, seq, _] = hidden.dims();
        self.logits(hidden.slice_dim(1, seq - 1..seq))
            .reshape([rows, self.n_vocab()])
    }
}

impl CachedStep for Qwen3Model {
    type State = Qwen3DecodeState;

    fn step(
        &self,
        ids: Tensor<2, Int>,
        state: &mut Qwen3DecodeState,
    ) -> Tensor<2> {
        self.prefill(self.embed(ids), state)
    }

    fn reorder(
        &self,
        state: &mut Qwen3DecodeState,
        sources: &[usize],
    ) {
        state.cache.reorder(sources);
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            decoding::{
                SearchConfig,
                decode_cached,
            },
            gpts::qwen3::Qwen3Config,
        },
        support::testing::cpu_device,
    };

    /// A greedy decode through the loop is the argmax of an uncached
    /// forward over the prompt and every token before it; a beam of three
    /// over three copies of the prompt finishes each of its candidates
    /// within the budget.
    #[test]
    fn test_decode_cached() {
        let device = cpu_device();
        let model: Qwen3Model = Qwen3Config::new(40, 12, 2, 3, 1, 6, 20).init(&device);
        let prompt = vec![1i64, 5, 7];
        let ids = |ids: &[i64], rows: usize| {
            let data: Vec<i64> = (0..rows).flat_map(|_| ids.iter().copied()).collect();
            Tensor::<2, Int>::from_data(TensorData::new(data, [rows, ids.len()]), &device)
        };

        let budget = 5;
        let mut state = model.decode_state(1, prompt.len() + budget, KVCacheStorage::Auto, &device);
        let first = model.prefill(model.embed(ids(&prompt, 1)), &mut state);
        let mut search = SearchConfig::new().init_search(&[39], 0);
        let out = decode_cached(
            &model,
            &mut state,
            vec![prompt.clone()],
            first,
            budget,
            search.as_mut(),
            &[],
        );
        let generated = out[0][0].tokens_with_stop();
        assert!(!generated.is_empty() && generated.len() <= budget);

        let mut seen = prompt.clone();
        for &token in &generated {
            let r_emb = model.rotary(seen.len(), &device);
            let hidden = model.forward_hidden(model.embed(ids(&seen, 1)), &r_emb, None);
            let expected = model.last_logits(hidden).argmax(1).into_scalar::<i64>();
            assert_eq!(token, expected, "after {seen:?}");
            seen.push(token);
        }

        let mut state = model.decode_state(3, prompt.len() + budget, KVCacheStorage::Auto, &device);
        let first = model.prefill(model.embed(ids(&prompt, 3)), &mut state);
        let mut beam = SearchConfig::new().with_beam_size(3).init_search(&[39], 0);
        let out = decode_cached(
            &model,
            &mut state,
            vec![prompt.clone(); 3],
            first,
            budget,
            beam.as_mut(),
            &[],
        );
        assert_eq!(out.len(), 1, "one group");
        assert_eq!(out[0].len(), 3);
        assert!(out[0].iter().all(|c| c.tokens.len() <= budget));
    }
}
