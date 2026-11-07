//! The toy models against upstream nanochat's own forward.
//!
//! `testdata/nanochat/toy_2025/` and `toy_2026/` each hold a tiny seeded
//! `nanochat.gpt.GPT` of one upstream generation, written as upstream's
//! checkpoint manager writes a checkpoint (`model_000001.pt`,
//! `meta_000001.json`), and its answers in fp32 (`io.safetensors`;
//! `tools/gen_nanochat_toy_fixtures.py`). Each loads through
//! [`load_pt_checkpoint`], so the generation's detection, its config and the
//! loader are under test with the model.

use core::ops::Range;
use std::path::PathBuf;

use burn::prelude::*;
use serial_test::serial;

use super::{
    NanoChatGpt,
    weights::{
        NanoChatGeneration,
        load_pt_checkpoint,
    },
};
use crate::{
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
    support::testing::{
        DecoderToy,
        DeviceMemoryGuard,
        ToyFixture,
        assert_io_close,
        check_decoder_toy,
        performance_device,
    },
};

/// The prefill the cached logits were recorded with; one token per step
/// after it.
const PREFILL: usize = 3;

struct Toy {
    model: NanoChatGpt,
    generation: NanoChatGeneration,
    fixture: ToyFixture,
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
            crate::kits::support::construct::LoadPolicy::at_least(burn::tensor::FloatDType::F32),
        )
        .expect("load the toy checkpoint");
        Self {
            model,
            generation,
            fixture: ToyFixture::load_io(dir.join("io.safetensors"), device),
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

/// Upstream's engine: a step is given the token before it, which the
/// prefill's first position has none of.
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
        mut cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let ids = self.ids();
        let prev =
            (range.start > 0).then(|| ids.clone().slice_dim(1, range.start - 1..range.start));
        self.model
            .forward_after(ids.slice_dim(1, range), prev, &mut cache)
    }
}

/// The whole sequence: the head's input and the logits; then through a KV
/// cache, as upstream's engine decodes it: a prefill, then one token per
/// step, each given the token before it.
fn check(toy: &Toy) {
    let hidden = toy.model.hidden_after(toy.ids(), None, &mut None);
    assert_io_close("hidden", hidden, toy.io("hidden"));
    check_decoder_toy(toy, toy.io("logits"), toy.io("logits_cached"), PREFILL);
}

#[test]
#[serial]
fn test_toy_2025() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let toy = Toy::load("2025", &device);
    assert_eq!(toy.generation, NanoChatGeneration::Upstream2025);
    assert!(!toy.model.is_smeared());
    check(&toy);
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
    check(&toy);
}

/// The load's dtype is the policy's, never the device's default: on the CPU,
/// whose default float is f32, the f32 toy keeps its stored f32 under
/// `as_stored`, and its tensors loaded at bf16 make a bf16 model.
#[test]
fn test_the_load_dtype_is_the_policy_s() {
    use burn::tensor::{
        DType,
        FloatDType,
    };

    use super::weights::{
        NanoChatMeta,
        load_pt_tensors,
    };
    use crate::{
        burner::module::ModuleInit,
        data::pretrained::read_pytorch_tensors,
        kits::support::construct::LoadPolicy,
        support::testing::cpu_device,
    };

    let device = cpu_device();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/nanochat/toy_2025");
    let (meta, model) = (dir.join("meta_000001.json"), dir.join("model_000001.pt"));
    let (stored, _) = load_pt_checkpoint(&meta, &model, &device, LoadPolicy::as_stored())
        .expect("load the toy as stored");
    assert_eq!(stored.float_dtype(), DType::F32);

    let meta = NanoChatMeta::from_file(&meta).unwrap();
    let tensors = read_pytorch_tensors(&model).unwrap();
    let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    let config = meta.gpt_config(meta.generation(&names).unwrap());
    let mut half: NanoChatGpt = config.try_init(&device).unwrap();
    load_pt_tensors(&mut half, tensors, FloatDType::BF16).unwrap();
    assert_eq!(half.float_dtype(), DType::BF16);
}
