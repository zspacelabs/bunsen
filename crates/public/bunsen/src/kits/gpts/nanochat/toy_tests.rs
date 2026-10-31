//! The toy models against upstream nanochat's own forward.
//!
//! `testdata/nanochat/toy_2025/` and `toy_2026/` each hold a tiny seeded
//! `nanochat.gpt.GPT` of one upstream generation, written as upstream's
//! checkpoint manager writes a checkpoint (`model_000001.pt`,
//! `meta_000001.json`), and its answers in fp32 (`io.safetensors`;
//! `tools/gen_nanochat_toy_fixtures.py`). Each loads through
//! [`load_pt_checkpoint`], so the generation's detection, its config and the
//! loader are under test with the model.

use std::{
    collections::BTreeMap,
    path::PathBuf,
};

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
use serial_test::serial;

use super::{
    NanoChatGpt,
    weights::{
        NanoChatGeneration,
        load_pt_checkpoint,
    },
};
use crate::{
    ops::transformers::attention::KVCacheModel,
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

/// The prefill the cached logits were recorded with; one token per step
/// after it.
const PREFILL: usize = 3;

struct Toy {
    model: NanoChatGpt,
    generation: NanoChatGeneration,
    io: BTreeMap<String, PackTensor>,
    device: Device,
}

impl Toy {
    fn load(
        generation: &str,
        device: &Device,
    ) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/nanochat")
            .join(format!("toy_{generation}"));
        let (model, generation) = load_pt_checkpoint(
            &dir.join("meta_000001.json"),
            &dir.join("model_000001.pt"),
            device,
        )
        .expect("load the toy checkpoint");
        let io = SafetensorsStore::from_file(dir.join("io.safetensors"))
            .get_all_tensors()
            .expect("read the toy's answers")
            .clone();
        Self {
            model,
            generation,
            io,
            device: device.clone(),
        }
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
        let ids: Tensor<2> = self.io("input_ids");
        ids.int()
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

/// The whole sequence: the head's input and the logits.
fn check_full(toy: &Toy) {
    let hidden = toy.model.hidden_after(toy.ids(), None, &mut None);
    assert_close("hidden", hidden.clone(), toy.io("hidden"));
    assert_close("logits", toy.model.logits(hidden), toy.io("logits"));
}

/// The sequence through a KV cache, as upstream's engine decodes it: a
/// prefill, then one token per step, each given the token before it.
fn check_cached(toy: &Toy) {
    let ids = toy.ids();
    let n = ids.dims()[1];
    let mut cache = toy.model.new_kv_cache(1);
    let mut steps = vec![
        toy.model
            .forward(ids.clone().slice(s![.., ..PREFILL]), &mut Some(&mut cache)),
    ];
    for i in PREFILL..n {
        steps.push(toy.model.forward_after(
            ids.clone().slice(s![.., i..i + 1]),
            Some(ids.clone().slice(s![.., i - 1..i])),
            &mut Some(&mut cache),
        ));
    }
    assert_close(
        "cached logits",
        Tensor::cat(steps, 1),
        toy.io("logits_cached"),
    );
}

#[test]
#[serial]
fn test_toy_2025() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let toy = Toy::load("2025", &device);
    assert_eq!(toy.generation, NanoChatGeneration::Upstream2025);
    assert!(!toy.model.is_smeared());
    check_full(&toy);
    check_cached(&toy);
}

/// The 2026 toy has every 2026 part, a long window shorter than its
/// sequence, and a vocabulary its embedding pads.
#[test]
#[serial]
fn test_toy_2026() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let toy = Toy::load("2026", &device);
    assert_eq!(toy.generation, NanoChatGeneration::Upstream2026);
    assert!(toy.model.is_smeared());
    check_full(&toy);
    check_cached(&toy);
}
