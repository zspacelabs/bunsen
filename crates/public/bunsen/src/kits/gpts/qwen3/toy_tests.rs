//! The toy model against `transformers`' own forward.
//!
//! `testdata/qwen3/toy.safetensors` holds a tiny seeded `Qwen3ForCausalLM`'s
//! weights and its answers, computed in float64
//! (`tools/gen_qwen3_toy_fixtures.py`), and an `lm_head.weight` that is not the
//! embedding, which a tied loader must not read. The geometry comes from
//! `toy.config.json` through [`qwen3_config_from_hf`], and the weights load
//! through [`load_hf_tensors`], so the loader is under test too.

use std::collections::BTreeMap;

use burn::{
    prelude::*,
    store::{
        ModuleStore,
        SafetensorsStore,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::Tolerance,
};

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
        KVCacheModel,
        KVCacheStorage,
    },
    support::testing::performance_device,
};

struct Toy {
    model: Qwen3Model,
    io: BTreeMap<String, PackTensor>,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let config = qwen3_config_from_hf(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3/toy.config.json"
        )))
        .expect("read the toy config");
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3/toy.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store
            .get_all_tensors()
            .expect("read the toy fixture")
            .clone();
        let (io, weights): (Vec<_>, Vec<_>) =
            all.into_iter().partition(|(n, _)| n.starts_with("io."));
        let device = performance_device();
        let mut model: Qwen3Model = config.init(&device);
        let applied = load_hf_tensors(&mut model, weights.into_iter().map(|(_, t)| t))
            .expect("load the toy weights");
        assert_eq!(applied.unused, ["lm_head.weight"]);
        let io = io
            .into_iter()
            .map(|(n, t)| (n.strip_prefix("io.").unwrap().to_string(), t))
            .collect();
        Self { model, io, device }
    }

    fn io<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        Tensor::from_data(
            bridge::to_data(&self.io[key]).expect("read an io tensor"),
            &self.device,
        )
    }

    fn ids(&self) -> Tensor<2, Int> {
        self.io::<2>("input_ids").int()
    }
}

fn assert_close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    assert_eq!(actual.dims(), expected.dims(), "{what}: shape");
    actual.to_data_as::<f32>().assert_approx_eq::<f32>(
        &expected.to_data_as::<f32>(),
        Tolerance::rel_abs(1e-3, 1e-4),
    );
}

/// An uncached pass is `transformers`' forward: the first layer, the
/// normalized hidden states, and the tied head's logits.
#[test]
fn test_toy_forward() {
    let toy = Toy::load();
    let model = &toy.model;
    let embeds = model.embed(toy.ids());
    let seq = embeds.dims()[1];
    let r_emb = model.rotary(seq, &toy.device);

    let layer0 = model.layers[0].forward(embeds.clone(), &r_emb, &mut None);
    assert_close("layer 0", layer0, toy.io("layer0"));
    let hidden = model.forward_hidden(embeds, &r_emb, None);
    assert_close("hidden", hidden.clone(), toy.io("hidden"));
    assert_close("logits", model.logits(hidden), toy.io("logits"));
}

/// A prefill and then one step per token, through a cache of every storage
/// that starts too small and grows, is `transformers`' uncached forward.
#[test]
fn test_toy_cached_decode() {
    let toy = Toy::load();
    let model = &toy.model;
    let embeds = model.embed(toy.ids());
    let seq = embeds.dims()[1];
    let r_emb = model.rotary(seq, &toy.device);
    let prefill = 3;

    for storage in [
        KVCacheStorage::Auto,
        KVCacheStorage::Concat,
        KVCacheStorage::PerLayer,
        KVCacheStorage::MultiLayer,
    ] {
        let mut cache = model
            .kv_cache_config()
            .with_seq_len(2)
            .with_chunk_size(2)
            .with_storage(storage)
            .init();
        let mut parts = vec![model.forward_hidden(
            embeds.clone().slice_dim(1, 0..prefill),
            &r_emb,
            Some(&mut cache),
        )];
        for pos in prefill..seq {
            parts.push(model.forward_hidden(
                embeds.clone().slice_dim(1, pos..pos + 1),
                &r_emb,
                Some(&mut cache),
            ));
        }
        assert_eq!(cache.pos(), seq, "{storage:?}");
        let logits = model.logits(Tensor::cat(parts, 1));
        assert_close(&format!("{storage:?} logits"), logits, toy.io("logits"));
    }
}
