//! The toy language model against upstream's reference.
//!
//! `testdata/zonos3/toy.safetensors` holds upstream's `TINY` language model
//! (2 layers of width 64: a gated `DeltaNet` with the dense `SwiGLU`, then a
//! global attention with a latent mixture of 4 experts) under the `-hf`
//! file's names, and the reference's answers over 72 rows, computed in
//! float64 (`tools/gen_zonos3_toy_fixtures.py`). The geometry comes from
//! `toy.config.json`'s `lm` section, and the weights load through
//! [`load_zonos3_lm_tensors`], so the reader and the loader are under test
//! too.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    Zonos3Lm,
    Zonos3LmConfig,
    Zonos3LmJson,
    Zonos3RowConditioning,
    weights::load_zonos3_lm_tensors,
};
use crate::{
    burner::module::ModuleInit,
    ops::transformers::attention::KVCacheStorage,
    support::testing::{
        ToyFixture,
        assert_io_close_within,
        performance_device,
    },
};

/// The toy's tolerance: relative `1e-3`, absolute `5e-4`. The reference
/// keeps its routes' scores, its rotary angles, its attention scores, its
/// delta rule and its routing in float32 whatever the model's dtype, so it is
/// float32's, not float64's.
const TOLERANCE: (f32, f32) = (1e-3, 5e-4);

fn close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    let (rel, abs) = TOLERANCE;
    assert_io_close_within(
        what,
        actual,
        expected,
        burn::tensor::Tolerance::rel_abs(rel, abs),
    );
}

struct Toy {
    config: Zonos3LmConfig,
    model: Zonos3Lm,
    fixture: ToyFixture,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let device = performance_device();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/zonos3");
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("toy.config.json")).expect("read the toy config"),
        )
        .expect("parse the toy config");
        let lm: Zonos3LmJson =
            serde_json::from_value(json["lm"].clone()).expect("the toy's lm section");
        let config = Zonos3LmConfig::from_json(&lm).expect("the toy's config");
        let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
        let mut model: Zonos3Lm = config.init(&device);
        load_zonos3_lm_tensors(&mut model, &config, fixture.take_weights(), None)
            .expect("load the toy weights");
        Self {
            config,
            model,
            fixture,
            device,
        }
    }

    /// `[1, 72, 11]`.
    fn ids(&self) -> Tensor<3, Int> {
        self.fixture.io::<2>("lm.ids").int().unsqueeze()
    }

    /// Row 0 a speaker row, rows 1-3 the conditions.
    fn rows(&self) -> Zonos3RowConditioning {
        Zonos3RowConditioning {
            speaker: Some((0, self.fixture.io::<1>("lm.speaker_embedding").unsqueeze())),
            conditions: Some((
                1,
                self.fixture
                    .io::<1>("lm.condition_buckets")
                    .int()
                    .unsqueeze(),
            )),
        }
    }

    fn expected<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        self.fixture.io::<D>(key)
    }
}

/// The geometry the toy's `lm` section gives.
#[test]
fn test_toy_config() {
    let toy = Toy::load();
    let c = &toy.config;
    assert_eq!((c.d_model, c.n_layers, c.global_period), (64, 2, 2));
    assert_eq!((c.n_codebooks, c.audio_vocab(), c.text_rows), (10, 18, 21));
    assert_eq!((c.gdn_heads, c.gdn_head_dim), (2, 16));
    assert_eq!((c.n_experts, c.top_k, c.moe_latent), (4, 2, 32));
    assert!(!c.layer_attends(0) && c.layer_attends(1));
    assert_eq!(toy.model.stack.n_sources(), 5);
}

/// Every stage of a full pass: the summed and the conditioned embeddings,
/// the bank (the normalized embedding, then each sublayer's output), the
/// final norm and the soft-capped logits.
#[test]
fn test_toy_stages() {
    let toy = Toy::load();
    let (logits, trace) = toy.model.forward_traced(toy.ids(), &toy.rows(), None);
    let one = |t: Tensor<3>| t.squeeze_dim::<2>(0);
    close("emb_sum", one(trace.emb_sum), toy.expected("lm.emb_sum"));
    close("emb_cond", one(trace.emb_cond), toy.expected("lm.emb_cond"));
    close(
        "emb_norm",
        one(trace.bank[0].clone()),
        toy.expected("lm.emb_norm"),
    );
    for l in 0..toy.config.n_layers {
        close(
            &format!("attn_out_{l}"),
            one(trace.bank[2 * l + 1].clone()),
            toy.expected(&format!("lm.attn_out_{l}")),
        );
        close(
            &format!("ffn_out_{l}"),
            one(trace.bank[2 * l + 2].clone()),
            toy.expected(&format!("lm.ffn_out_{l}")),
        );
    }
    close("out_norm", one(trace.hidden), toy.expected("lm.out_norm"));
    close(
        "logits",
        logits.squeeze_dim::<3>(0),
        toy.expected("lm.logits"),
    );
}

/// A chunked prefill and a decode a row at a time, through a decode cache,
/// give the reference's logits for the rows they run.
#[test]
fn test_toy_cached() {
    let toy = Toy::load();
    let ids = toy.ids();
    let rows = toy.rows();
    let n = ids.dims()[1];

    let mut cache = toy
        .model
        .decode_cache(1, n, KVCacheStorage::Auto, &toy.device);
    let _ = toy
        .model
        .forward(ids.clone().slice(s![.., 0..40]), &rows, Some(&mut cache));
    let chunk = toy
        .model
        .forward(ids.clone().slice(s![.., 40..n]), &rows, Some(&mut cache));
    close(
        "chunked_logits",
        chunk.squeeze_dim::<3>(0),
        toy.expected("lm.chunked_logits"),
    );

    let mut cache = toy
        .model
        .decode_cache(1, n, KVCacheStorage::Auto, &toy.device);
    let _ = toy
        .model
        .forward(ids.clone().slice(s![.., 0..64]), &rows, Some(&mut cache));
    let steps: Vec<_> = (64..n)
        .map(|r| {
            toy.model
                .forward(ids.clone().slice(s![.., r..r + 1]), &rows, Some(&mut cache))
        })
        .collect();
    close(
        "decode_logits",
        Tensor::cat(steps, 1).squeeze_dim::<3>(0),
        toy.expected("lm.decode_logits"),
    );
}
