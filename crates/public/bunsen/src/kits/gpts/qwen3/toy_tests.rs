//! The toy models against `transformers`' own forward.
//!
//! `testdata/qwen3/toy.safetensors` holds a tiny seeded `Qwen3ForCausalLM`'s
//! weights and its answers, computed in float64
//! (`tools/gen_qwen3_toy_fixtures.py`), and an `lm_head.weight` that is not the
//! embedding, which a tied loader must not read. `toy_untied.safetensors` is
//! the same with a head of its own, which the logits are computed through.
//! Each geometry comes from its `config.json` through
//! [`qwen3_config_from_hf`], and the weights load through
//! [`load_hf_tensors`], so the loader is under test too.

use std::{
    ops::Range,
    path::PathBuf,
};

use burn::prelude::*;

use super::{
    Qwen3Model,
    weights::{
        load_hf_tensors,
        qwen3_config_from_hf,
    },
};
use crate::{
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
    model: Qwen3Model,
    fixture: ToyFixture,
    device: Device,
}

/// The tied toy and the untied one: their names under `testdata/qwen3/`.
const TOYS: [&str; 2] = ["toy", "toy_untied"];

impl Toy {
    fn load(name: &str) -> Self {
        Self::load_on(name, &performance_device())
    }

    fn load_on(
        name: &str,
        device: &Device,
    ) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/qwen3");
        let json = std::fs::read_to_string(dir.join(format!("{name}.config.json")))
            .expect("read the toy config");
        let config = qwen3_config_from_hf(&json).expect("parse the toy config");
        let mut fixture = ToyFixture::load(dir.join(format!("{name}.safetensors")), device);
        let device = device.clone();
        let mut model: Qwen3Model = config.init(&device);
        let applied =
            load_hf_tensors(&mut model, fixture.take_weights()).expect("load the toy weights");
        if config.tied_head {
            assert_eq!(applied.unused, ["lm_head.weight"], "{name}");
        } else {
            assert!(applied.unused.is_empty(), "{name}: {:?}", applied.unused);
            assert!(applied.applied.iter().any(|p| p == "lm_head.weight"));
        }
        Self {
            model,
            fixture,
            device,
        }
    }

    fn io<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        self.fixture.io(key)
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
/// normalized hidden states, and the head's logits, tied or its own.
#[test]
fn test_toy_forward() {
    for name in TOYS {
        check_forward(&Toy::load(name));
    }
}

fn check_forward(toy: &Toy) {
    let model = &toy.model;
    let embeds = model.embed(toy.ids());
    let seq = embeds.dims()[1];
    let r_emb = model.rotary(seq, &toy.device);

    let layer0 = model.stack.layers[0].forward(embeds.clone(), &r_emb, &mut None);
    assert_io_close("layer 0", layer0, toy.io("layer0"));
    let hidden = model.forward_hidden(embeds, &r_emb, None);
    assert_io_close("hidden", hidden.clone(), toy.io("hidden"));
    assert_io_close("logits", model.logits(hidden), toy.io("logits"));
}

/// A prefill and then one step per token, through a cache of every storage
/// that starts too small and grows, is `transformers`' uncached forward.
#[test]
fn test_toy_cached_decode() {
    for name in TOYS {
        let toy = Toy::load(name);
        check_decoder_toy(&toy, toy.io("logits"), toy.io("logits"), 3);
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
        let expected: Tensor<3> = toy.io("logits");
        let gap = (logits - expected.clone()).abs().max().into_scalar::<f32>();
        let scale = expected.abs().max().into_scalar::<f32>();
        assert!(
            gap <= 0.05 * scale,
            "{name}: bf16 logits {gap} from f32 (max {scale})"
        );
    }
}
