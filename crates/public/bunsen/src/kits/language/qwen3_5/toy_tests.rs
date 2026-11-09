//! The toy models against `transformers`' own forward.
//!
//! `testdata/qwen3_5/` holds two tiny seeded
//! `Qwen3_5ForConditionalGeneration`s' weights and their answers, computed in
//! float64 (`tools/gen_qwen3_5_toy_fixtures.py`), saved as the published
//! checkpoints are (a `text_config`, a vision tower and `mtp.` tensors beside
//! the text model): `toy` (three linear layers to one full, equal linear key
//! and value heads, a tied head and a decoy `lm_head.weight`) and `toy_untied`
//! (linear and full alternating, twice the value heads, one key/value head, a
//! head of its own), over 72 tokens, past the delta rule's chunk of 64. Each
//! geometry comes from its `config.json` through
//! [`qwen3_5_config_from_transformers`], and the weights load through
//! [`load_transformers_tensors`], so the reader and the loader are under test
//! too.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    Qwen3_5Model,
    weights::{
        load_transformers_tensors,
        qwen3_5_config_from_transformers,
    },
};
use crate::{
    blocks::transformers::{
        decoders::stack::DecoderStackMeta,
        models::language_decoder::{
            LmDecoderOps,
            LmUncachedForwardHidden,
        },
    },
    burner::module::ModuleInit,
    kits::support::decoding::{
        CachedStep,
        StackedLm,
    },
    ops::transformers::attention::KVCacheStorage,
    support::testing::{
        KV_CACHE_STORAGES,
        ToyFixture,
        assert_io_close_within,
        performance_device,
    },
};

/// The toys' tolerance: relative `1e-3`, absolute `5e-4`. `transformers`
/// runs the norms, the delta rule and the rotary angles in float32 whatever
/// the model's dtype, so the reference is float32's, not float64's, and the
/// toys' 72 positions of random recurrence carry its rounding: the worst
/// value is off by 8.2e-5 on one CPU, 1.03e-4 on another (GitHub's runner),
/// over the shared `1e-4`.
const TOLERANCE: (f32, f32) = (1e-3, 5e-4);

/// [`assert_io_close_within`] the toys' [`TOLERANCE`].
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
    model: Qwen3_5Model,
    fixture: ToyFixture,
    device: Device,
}

/// The toys: their names under `testdata/qwen3_5/`.
const TOYS: [&str; 2] = ["toy", "toy_untied"];

impl Toy {
    fn load_on(
        name: &str,
        device: &Device,
    ) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/qwen3_5");
        let json = std::fs::read_to_string(dir.join(format!("{name}.config.json")))
            .expect("read the toy config");
        let config = qwen3_5_config_from_transformers(&json).expect("parse the toy config");
        let mut fixture = ToyFixture::load(dir.join(format!("{name}.safetensors")), device);
        let mut model: Qwen3_5Model = config.init(device);
        let applied = load_transformers_tensors(&mut model, fixture.take_weights())
            .expect("load the toy weights");
        // Unused: the dropped vision tower and prediction head, and a tied
        // toy's decoy head, which a tied model has nowhere to load.
        let dropped = |n: &String| n.starts_with("model.visual.") || n.starts_with("mtp.");
        let rest: Vec<_> = applied.unused.iter().filter(|n| !dropped(n)).collect();
        if config.tied_head {
            assert_eq!(rest, ["head.proj.weight"], "{name}");
        } else {
            assert!(rest.is_empty(), "{name}: {rest:?}");
        }
        assert!(
            applied
                .unused
                .iter()
                .any(|n| n.starts_with("model.visual."))
                && applied.unused.iter().any(|n| n.starts_with("mtp.")),
            "{name}: the toy carries a vision tower and a prediction head"
        );
        Self {
            model,
            fixture,
            device: device.clone(),
        }
    }

    fn ids(&self) -> Tensor<2, Int> {
        self.fixture.ids()
    }

    /// The first full-attention layer's index.
    fn first_full(&self) -> usize {
        self.model
            .lm
            .stack
            .layer_attends()
            .iter()
            .position(|&a| a)
            .expect("a full layer")
    }
}

/// An uncached pass is `transformers`' forward: the first (linear) layer,
/// the first full layer, the normalized hidden states, and the logits.
#[test]
fn test_toy_forward() {
    for name in TOYS {
        let toy = Toy::load_on(name, &performance_device());
        let model = &toy.model;
        let embeds = model.embed(toy.ids());
        let r_emb = model.rotary(embeds.dims()[1], &toy.device);
        let full = toy.first_full();
        let mut x = embeds.clone();
        for (i, layer) in model.lm.stack.layers.iter().enumerate().take(full + 1) {
            x = layer.forward(x, r_emb.for_layer(i), &mut None);
            if i == 0 {
                close(
                    &format!("{name} layer 0"),
                    x.clone(),
                    toy.fixture.io("layer0"),
                );
            }
        }
        close(
            &format!("{name} full layer {full}"),
            x,
            toy.fixture.io("full0"),
        );
        let hidden = model.forward_hidden(embeds, &r_emb);
        close(
            &format!("{name} hidden"),
            hidden.clone(),
            toy.fixture.io("hidden"),
        );
        close(
            &format!("{name} logits"),
            model.logits(hidden),
            toy.fixture.io("logits"),
        );
    }
}

/// A prefill and then one step per token, through a decode state under every
/// KV storage, is `transformers`' uncached forward: each step's logits are
/// the forward's at that position. The prefill runs the delta rule a chunk at
/// a time, the steps one position, and the decode crosses the chunk.
#[test]
fn test_toy_cached_decode() {
    for name in TOYS {
        let toy = Toy::load_on(name, &performance_device());
        let expected: Tensor<3> = toy.fixture.io("logits");
        let n = toy.ids().dims()[1];
        let prefill = 3;
        for storage in KV_CACHE_STORAGES {
            check_cached_decode(&toy, &expected, n, prefill, storage);
        }
    }
}

fn check_cached_decode(
    toy: &Toy,
    expected: &Tensor<3>,
    n: usize,
    prefill: usize,
    storage: KVCacheStorage,
) {
    let model = &toy.model;
    let ids = toy.ids();
    let mut state = model.decode_state(1, n, storage, &toy.device);
    let first = model.prefill(
        model.embed(ids.clone().slice_dim(1, 0..prefill)),
        &mut state,
    );
    let at = |pos: usize| expected.clone().slice_dim(1, pos..=pos).reshape([1, -1]);
    close(&format!("{storage:?} prefill"), first, at(prefill - 1));
    for pos in prefill..n {
        let logits = model.step(ids.clone().slice_dim(1, pos..=pos), &mut state);
        close(&format!("{storage:?} step {pos}"), logits, at(pos));
    }
}

/// How far `transformers`' own forward of each toy in bf16 lands from its
/// float64 logits (the largest absolute difference; `toy`'s logits reach
/// 3.7, `toy_untied`'s 4.0): the toys' random recurrences amplify bf16's
/// rounding, so a fixed share of the logits is no bound.
const TRANSFORMERS_BF16_GAP: [(&str, f32); 2] = [("toy", 1.40), ("toy_untied", 0.44)];

/// bf16 parameters on an f32 CPU device, as a `LoadPolicy::as_stored` load
/// places them: the model runs end to end with no mixed op, which burn-flex
/// refuses, so a missed cast fails here; the logits leave it in f32, and
/// land within twice `transformers`' own bf16 gap of the f32 toy's
/// ([`TRANSFORMERS_BF16_GAP`]).
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
        let logits = half.logits(half.forward_hidden(embeds, &r_emb));
        assert_eq!(
            logits.dtype(),
            DType::F32,
            "{name}: the logits leave in f32"
        );
        let expected: Tensor<3> = toy.fixture.io("logits");
        let gap = (logits - expected).abs().max().into_scalar::<f32>();
        let bound = 2.0
            * TRANSFORMERS_BF16_GAP
                .iter()
                .find(|(toy, _)| *toy == name)
                .expect("a measured gap")
                .1;
        eprintln!("{name}: bf16 logits off by {gap}");
        assert!(
            gap <= bound,
            "{name}: bf16 logits {gap} from f32, over {bound}"
        );
    }
}
