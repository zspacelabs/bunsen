//! [`Zonos3Lm`] as a [`TimedDecode`].

use burn::{
    Tensor,
    prelude::Int,
    tensor::Device,
};

use super::blocks::{
    Zonos3Lm,
    Zonos3LmMeta,
    Zonos3RowConditioning,
};
use crate::{
    blocks::transformers::decoders::stack::DecoderStackCache,
    kits::support::timing::{
        TimedDecode,
        synthetic_ids,
    },
    ops::transformers::attention::KVCacheStorage,
};

/// A take's phases, as the engine's `generate_codes` runs them: the prompt's
/// rows prefilled through the language model's cache (no speaker or
/// condition rows: they replace a row's embedding and cost nothing apart),
/// then one row per step, `K` codebook codes and a text pad. A step makes
/// one codec frame; the sampler's choice from the logits is outside the
/// window, as a timing of the model.
impl TimedDecode for Zonos3Lm {
    type State = DecoderStackCache;

    fn step_unit(&self) -> &'static str {
        "frame"
    }

    fn timed_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        self.decode_cache(rows, positions, storage, device)
    }

    fn timed_prefill(
        &self,
        state: &mut DecoderStackCache,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2> {
        self.timed_rows(state, rows, len, device)
    }

    fn timed_step(
        &self,
        state: &mut DecoderStackCache,
        rows: usize,
        device: &Device,
    ) -> Tensor<2> {
        self.timed_rows(state, rows, 1, device)
    }
}

impl Zonos3Lm {
    /// `len` synthetic rows through the cache: every id below each table's
    /// rows, the codebooks' and the text's.
    ///
    /// # Returns
    /// `[rows, K * V_a]`: the last row's logits, every codebook's.
    fn timed_rows(
        &self,
        state: &mut DecoderStackCache,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2> {
        let width = self.n_codebooks() + 1;
        let bound = self.audio_vocab().min(self.text_rows()) as i64;
        let ids: Tensor<3, Int> = synthetic_ids(rows, 0, len * width, device)
            .reshape([rows, len, width])
            .remainder_scalar(bound);
        let logits = self.forward(ids, &Zonos3RowConditioning::default(), Some(state));
        let [_, _, k, v] = logits.dims();
        logits.slice_dim(1, len - 1..len).reshape([rows, k * v])
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::tts::zonos3::{
            blocks::Zonos3LmConfig,
            interop::zonos::Zonos3LmMirror,
        },
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// The toy's language model config (`testdata/zonos3/toy.config.json`).
    fn toy_config() -> Zonos3LmConfig {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/zonos3/toy.config.json");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read the toy config"))
                .expect("parse the toy config");
        let lm: Zonos3LmMirror =
            serde_json::from_value(json["lm"].clone()).expect("the toy's lm section");
        lm.lm_config().expect("the toy's config")
    }

    /// A timed ZONOS3 decode is its language model's: the prefill's and a
    /// step's logits are the uncached forward's over the same rows.
    #[test]
    fn test_timed_zonos3_is_its_lm() {
        let device = cpu_device();
        let lm: Zonos3Lm = toy_config().init(&device);
        let (rows, len) = (2, 3);
        let mut state = lm.timed_state(rows, len + 1, KVCacheStorage::Auto, &device);
        assert_eq!(TimedDecode::step_unit(&lm), "frame");
        assert!(lm.timed_encode(&mut state, rows, &device).is_none());
        let prefilled = lm.timed_prefill(&mut state, rows, len, &device);
        let stepped = lm.timed_step(&mut state, rows, &device);

        let width = lm.n_codebooks() + 1;
        let bound = lm.audio_vocab().min(lm.text_rows()) as i64;
        let ids = Tensor::cat(
            vec![
                synthetic_ids(rows, 0, len * width, &device).reshape([rows, len, width]),
                synthetic_ids(rows, 0, width, &device).reshape([rows, 1, width]),
            ],
            1,
        )
        .remainder_scalar(bound);
        let whole = lm.forward(ids, &Zonos3RowConditioning::default(), None);
        let [_, _, k, v] = whole.dims();
        let at = |t: usize| whole.clone().slice_dim(1, t..t + 1).reshape([rows, k * v]);
        let tolerance = Tolerance::rel_abs(1e-4, 1e-5);
        assert_tensors_close(&prefilled, &at(len - 1), tolerance);
        assert_tensors_close(&stepped, &at(len), tolerance);
    }
}
