//! Cached decoding of a nanochat model.
use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
        s,
    },
};

use super::{
    NanoChatGpt,
    NanoChatGptMeta,
};
use crate::{
    kits::decoding::CachedStep,
    ops::transformers::attention::{
        KVCache,
        KVCacheModel,
        KVCacheStorage,
    },
};

/// What a cached nanochat decode carries between steps: its KV cache, and
/// each row's last token, which a smeared model (upstream's 2026 one) mixes
/// into the next position.
///
/// Made by [`NanoChatGpt::decode_state`], filled by
/// [`NanoChatGpt::prefill`], then stepped by
/// [`decode_cached`](crate::kits::decoding::decode_cached) through the
/// model's [`CachedStep`]. The model never owns one, so one loaded model
/// serves any number of decodes.
#[derive(Debug)]
pub struct NanoChatDecodeState {
    /// The KV cache, one batch row per decode row.
    pub cache: KVCache,

    /// `[rows, 1]`: the last token each row ran, or `None` before any.
    pub last: Option<Tensor<2, Int>>,
}

impl NanoChatDecodeState {
    /// Rewinds to position `pos`, `last` the token at `pos - 1` of each row
    /// (`None` at 0): as though nothing past `pos` had run. A decode that
    /// changes its mind about a suffix keeps the prefix it shares.
    ///
    /// # Panics
    /// As [`KVCache::truncate`]; and when `last` is `None` past 0, or given
    /// at 0.
    pub fn rewind(
        &mut self,
        pos: usize,
        last: Option<Tensor<2, Int>>,
    ) {
        assert_eq!(
            pos == 0,
            last.is_none(),
            "a decode rewound to {pos} needs the token before it exactly when past 0"
        );
        self.cache.truncate(pos);
        self.last = last;
    }
}

impl NanoChatGpt {
    /// A decode state for `rows` rows of at most `positions` positions each,
    /// under `storage`: the cache allocated whole and capped there.
    ///
    /// # Panics
    /// When `positions` exceeds the rotary table's `max_seq_len`.
    pub fn decode_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
    ) -> NanoChatDecodeState {
        assert!(
            positions <= self.max_seq_len(),
            "{positions} positions exceed the rotary table's {}",
            self.max_seq_len()
        );
        NanoChatDecodeState {
            cache: self
                .kv_cache_config()
                .with_batch_size(rows)
                .with_seq_len(positions)
                .with_max_seq_len(Some(positions))
                .with_storage(storage)
                .init(),
            last: None,
        }
    }

    /// Runs `ids`, `[rows, seq]`, through the model into `state`, after
    /// whatever it holds.
    ///
    /// # Returns
    /// `[rows, vocab]`, the logits that follow the last position.
    pub fn prefill(
        &self,
        ids: Tensor<2, Int>,
        state: &mut NanoChatDecodeState,
    ) -> Tensor<2> {
        let [rows, seq] = ids.dims();
        let last = ids.clone().slice(s![.., seq - 1..seq]);
        let logits = self.forward_after(ids, state.last.take(), &mut Some(&mut state.cache));
        state.last = Some(last);
        logits
            .slice(s![.., seq - 1..seq])
            .reshape([rows, self.vocab_size()])
    }
}

impl CachedStep for NanoChatGpt {
    type State = NanoChatDecodeState;

    fn step(
        &self,
        ids: Tensor<2, Int>,
        state: &mut NanoChatDecodeState,
    ) -> Tensor<2> {
        self.prefill(ids, state)
    }

    fn reorder(
        &self,
        state: &mut NanoChatDecodeState,
        sources: &[usize],
    ) {
        state.cache.reorder(sources);
        if let Some(last) = state.last.take() {
            let indices = Tensor::<1, Int>::from_data(
                TensorData::new(
                    sources.iter().map(|&s| s as i64).collect::<Vec<_>>(),
                    [sources.len()],
                ),
                &last.device(),
            );
            state.last = Some(last.select(0, indices));
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            decoding::{
                SearchConfig,
                decode_cached,
            },
            gpts::nanochat::NanoChatGptContractConfig,
        },
        support::testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    };

    fn ids(
        ids: &[i64],
        rows: usize,
        device: &Device,
    ) -> Tensor<2, Int> {
        let data: Vec<i64> = (0..rows).flat_map(|_| ids.iter().copied()).collect();
        Tensor::from_data(TensorData::new(data, [rows, ids.len()]), device)
    }

    /// A tiny 2026 model, so the smear is in play; its smear strength set
    /// away from its neutral start.
    fn model(device: &Device) -> NanoChatGpt {
        let mut gpt: NanoChatGpt = NanoChatGptContractConfig::new()
            .with_vocab_size(40)
            .with_n_layer(2)
            .with_n_head(2)
            .with_n_kv_head(1)
            .with_n_embed(32)
            .with_init_seq_len(4)
            .with_upstream_2026("SL")
            .init(device);
        gpt.set_smear_lambda_for_tests(0.8, device);
        gpt
    }

    /// A greedy decode through the loop is the argmax of an uncached forward
    /// over the prompt and every token before it, the smear and windows
    /// carried across steps; a beam of three over three copies of the
    /// prompt finishes each of its candidates within the budget.
    #[test]
    #[serial]
    fn test_decode_cached() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let model = model(&device);
        let prompt = vec![1i64, 5, 7];

        let budget = 6;
        let mut state = model.decode_state(1, prompt.len() + budget, KVCacheStorage::Auto);
        let first = model.prefill(ids(&prompt, 1, &device), &mut state);
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
            let logits = model.forward(ids(&seen, 1, &device), &mut None);
            let n = seen.len();
            let expected = logits
                .slice(s![.., n - 1..n])
                .reshape([1, 40])
                .argmax(1)
                .into_scalar::<i64>();
            assert_eq!(token, expected, "after {seen:?}");
            seen.push(token);
        }

        let mut state = model.decode_state(3, prompt.len() + budget, KVCacheStorage::Auto);
        let first = model.prefill(ids(&prompt, 3, &device), &mut state);
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

    /// A rewound decode, given the token before its new end, runs on as
    /// though it had never gone past it.
    #[test]
    #[serial]
    fn test_rewind() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let model = model(&device);

        let mut state = model.decode_state(1, 12, KVCacheStorage::Auto);
        model.prefill(ids(&[3, 1, 4, 1, 5], 1, &device), &mut state);
        state.rewind(3, Some(ids(&[4], 1, &device)));
        let rewound = model.prefill(ids(&[2, 6], 1, &device), &mut state);

        let mut fresh = model.decode_state(1, 12, KVCacheStorage::Auto);
        let expected = model.prefill(ids(&[3, 1, 4, 2, 6], 1, &device), &mut fresh);
        rewound.into_data().assert_approx_eq::<f32>(
            &expected.into_data(),
            burn::tensor::Tolerance::rel_abs(1e-3, 1e-4),
        );
    }
}
