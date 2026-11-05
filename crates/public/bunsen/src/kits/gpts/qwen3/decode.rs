//! Cached decoding of a Qwen3.
use burn::{
    Tensor,
    prelude::Int,
};

use super::Qwen3Model;
use crate::{
    blocks::transformers::decoders::stack::DecoderStack,
    kits::support::decoding::StackedLm,
};

/// A Qwen3 decodes through its stack's
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache):
/// [`StackedLm::decode_state`], [`StackedLm::prefill`], then
/// [`decode_cached`](crate::kits::support::decoding::decode_cached).
impl StackedLm for Qwen3Model {
    fn stack(&self) -> &DecoderStack {
        &self.stack
    }

    fn embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        Qwen3Model::embed(self, ids)
    }

    fn head(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.logits(hidden)
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            gpts::qwen3::Qwen3Config,
            support::decoding::{
                SearchConfig,
                decode_cached,
            },
        },
        ops::transformers::attention::KVCacheStorage,
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
            let n = hidden.dims()[1];
            let expected = model
                .logits(hidden.slice_dim(1, n - 1..n))
                .argmax(2)
                .into_scalar::<i64>();
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
