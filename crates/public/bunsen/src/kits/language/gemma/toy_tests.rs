//! The toy models against `transformers`' own forward.
//!
//! `testdata/gemma/` holds four tiny seeded Gemmas' weights and their
//! answers, computed in float64 (`tools/gen_gemma_toy_fixtures.py`): two
//! Gemma 3s (grouped-query and multi-query), a Gemma 2 and a Gemma 1, each
//! with the features of its generation set to bind (a window of 4 against 16
//! tokens, a `query_pre_attn_scalar` that is not the head width, a linear
//! factor on the global table alone, caps of 2 and 3, random norm weights).
//! Each geometry comes from its `config.json` through
//! [`gemma_config_from_transformers`], and the weights load through
//! [`load_transformers_tensors`], so the reader and the loader are under test
//! too.

use core::ops::Range;
use std::path::PathBuf;

use burn::prelude::*;

use super::{
    GemmaModel,
    weights::{
        gemma_config_from_transformers,
        load_transformers_tensors,
    },
};
use crate::{
    blocks::transformers::models::LmKvForwardHidden,
    burner::module::ModuleInit,
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
    support::testing::{
        DecoderToy,
        ToyFixture,
        assert_io_close,
        check_decoder_toy,
        performance_device,
    },
};

struct Toy {
    model: GemmaModel,
    fixture: ToyFixture,
    device: Device,
}

/// The toys: their names under `testdata/gemma/`.
const TOYS: [&str; 4] = ["gemma3_toy", "gemma3_toy_mqa", "gemma2_toy", "gemma_toy"];

impl Toy {
    fn load_on(
        name: &str,
        device: &Device,
    ) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/gemma");
        let json = std::fs::read_to_string(dir.join(format!("{name}.config.json")))
            .expect("read the toy config");
        let config = gemma_config_from_transformers(&json).expect("parse the toy config");
        let mut fixture = ToyFixture::load(dir.join(format!("{name}.safetensors")), device);
        let mut model: GemmaModel = config.init(device);
        let applied = load_transformers_tensors(&mut model, fixture.take_weights())
            .expect("load the toy weights");
        assert!(applied.unused.is_empty(), "{name}: {:?}", applied.unused);
        Self {
            model,
            fixture,
            device: device.clone(),
        }
    }

    fn ids(&self) -> Tensor<2, Int> {
        self.fixture.ids()
    }
}

impl DecoderToy for Toy {
    fn positions(&self) -> usize {
        self.ids().dims()[1]
    }

    fn kv_cache_config(&self) -> KVCacheConfig {
        self.model.kv_cache_config()
    }

    fn logits(
        &self,
        range: Range<usize>,
        cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let embeds = self.model.embed(self.ids().slice_dim(1, range));
        let r_emb = self.model.rotary(self.positions(), &self.device);
        self.model
            .logits(self.model.forward_hidden(embeds, &r_emb, cache))
    }
}

/// An uncached pass is `transformers`' forward: the first layer, the
/// normalized hidden states, and the logits.
#[test]
fn test_toy_forward() {
    for name in TOYS {
        let toy = Toy::load_on(name, &performance_device());
        let model = &toy.model;
        let embeds = model.embed(toy.ids());
        let r_emb = model.rotary(embeds.dims()[1], &toy.device);
        let layer0 =
            model.lm.stack.layers[0].forward(embeds.clone(), r_emb.for_layer(0), &mut None);
        assert_io_close(&format!("{name} layer 0"), layer0, toy.fixture.io("layer0"));
        let hidden = model.forward_hidden(embeds, &r_emb, None);
        assert_io_close(
            &format!("{name} hidden"),
            hidden.clone(),
            toy.fixture.io("hidden"),
        );
        assert_io_close(
            &format!("{name} logits"),
            model.logits(hidden),
            toy.fixture.io("logits"),
        );
    }
}

/// A prefill and then one step per token, through a cache of every storage
/// that starts too small and grows, is `transformers`' uncached forward:
/// past the window, the sliding layers' caches keep only it.
#[test]
fn test_toy_cached_decode() {
    for name in TOYS {
        let toy = Toy::load_on(name, &performance_device());
        let logits: Tensor<3> = toy.fixture.io("logits");
        check_decoder_toy(&toy, logits.clone(), logits, 3);
    }
}

/// bf16 parameters on an f32 CPU device, as a `LoadPolicy::as_stored` load
/// places them: the model runs end to end with no mixed op, which burn-flex
/// refuses, so a missed cast fails here; the logits leave it in f32, and
/// land near the f32 toy's.
#[test]
fn test_toy_mixed_dtype() {
    use burn::{
        module::Module,
        tensor::DType,
    };

    use crate::{
        burner::module::DTypeMapper,
        support::testing::cpu_device,
    };

    let device = cpu_device();
    for name in TOYS {
        let toy = Toy::load_on(name, &device);
        let half = toy.model.clone().map(&mut DTypeMapper::new(DType::BF16));
        let embeds = half.embed(toy.ids());
        assert_eq!(embeds.dtype(), DType::BF16, "{name}: the stream is bf16");
        let r_emb = half.rotary(embeds.dims()[1], &device);
        let logits = half.logits(half.forward_hidden(embeds, &r_emb, None));
        assert_eq!(
            logits.dtype(),
            DType::F32,
            "{name}: the logits leave in f32"
        );
        let expected: Tensor<3> = toy.fixture.io("logits");
        let gap = (logits - expected.clone()).abs().max().into_scalar::<f32>();
        let scale = expected.abs().max().into_scalar::<f32>();
        assert!(
            gap <= 0.05 * scale,
            "{name}: bf16 logits {gap} from f32 (max {scale})"
        );
    }
}
