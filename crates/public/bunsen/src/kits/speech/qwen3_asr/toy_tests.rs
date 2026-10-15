//! The toy model against `transformers`' own forward.
//!
//! `testdata/qwen3_asr/toy.safetensors` holds a tiny seeded
//! `Qwen3ASRForConditionalGeneration`'s weights and its answers, computed in
//! float64 (`tools/gen_qwen3_asr_toy_fixtures.py`). The weights load through
//! [`load_hf_tensors`], so the loader is under test too.

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
    Qwen3AsrGreedyDecoderConfig,
    blocks::{
        Qwen3Asr,
        Qwen3AsrAudioEncoderConfig,
        Qwen3AsrConfig,
    },
    weights::load_hf_tensors,
};
use crate::{
    burner::module::ModuleInit,
    kits::gpts::qwen3::Qwen3Config,
    support::testing::performance_device,
};

/// The audio placeholder of the toy's prompt.
const AUDIO_TOKEN: usize = 7;

/// The valid mel frames of the toy's input.
const FEATURE_LEN: usize = 250;

/// `toy.json`'s config.
fn toy_config() -> Qwen3AsrConfig {
    Qwen3AsrConfig::new(
        Qwen3AsrAudioEncoderConfig::new(16, 32, 4, 64, 2, 8).with_n_window_infer(200),
        Qwen3Config::new(50, 24, 2, 4, 2, 8, 48),
    )
    .with_audio_token(AUDIO_TOKEN)
}

struct Toy {
    model: Qwen3Asr,
    io: BTreeMap<String, PackTensor>,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/qwen3_asr/toy.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store
            .get_all_tensors()
            .expect("read the toy fixture")
            .clone();
        let (weights, io): (Vec<_>, Vec<_>) =
            all.into_iter().partition(|(n, _)| n.starts_with("model."));
        let device = performance_device();
        let mut model: Qwen3Asr = toy_config().init(&device);
        let applied = load_hf_tensors(&mut model, weights.into_iter().map(|(_, t)| t))
            .expect("load the toy weights");
        assert!(applied.unused.is_empty(), "unused: {:?}", applied.unused);
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

    fn features(&self) -> Tensor<2> {
        let f: Tensor<3> = self.io("features");
        let [_, m, t] = f.dims();
        f.reshape([m, t])
    }

    fn ids(&self) -> Vec<usize> {
        let ids: Tensor<2> = self.io("input_ids");
        ids.into_data()
            .try_into_vec_as::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| v as usize)
            .collect()
    }
}

fn assert_close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    assert_eq!(actual.dims(), expected.dims(), "{what}: shape");
    eprintln!("checking {what}");
    actual.to_data_as::<f32>().assert_approx_eq::<f32>(
        &expected.to_data_as::<f32>(),
        Tolerance::rel_abs(1e-3, 1e-4),
    );
}

#[test]
fn test_toy_audio_tower() {
    let toy = Toy::load();
    let tower = &toy.model.audio_tower;
    let stem = tower.stem(toy.features(), FEATURE_LEN);
    assert_close("conv stem", stem.clone(), toy.io("conv"));

    let mask = tower.window_mask(stem.dims()[0], &toy.device);
    let layer0 = tower.layers[0].forward_masked(stem.unsqueeze_dim(0), Some(mask));
    assert_close(
        "audio layer 0",
        layer0.squeeze_dim::<2>(0),
        toy.io("audio_layer0"),
    );

    let audio = tower.forward(toy.features(), FEATURE_LEN);
    assert_close("audio tower", audio.clone(), toy.io("audio"));
    assert_close(
        "projector",
        toy.model.multi_modal_projector.forward(audio),
        toy.io("audio_projected"),
    );
}

#[test]
fn test_toy_text_model() {
    let toy = Toy::load();
    let audio = toy.model.encode_audio(toy.features(), FEATURE_LEN);
    let embeds = toy.model.prompt_embeds(&toy.ids(), audio).unwrap();
    let lm = &toy.model.language_model;
    let seq = embeds.dims()[1];
    let r_emb = lm.rotary(seq, &toy.device);

    let layer0 = lm.layers[0].forward(embeds.clone(), &r_emb, &mut None);
    assert_close("text layer 0", layer0, toy.io("text_layer0"));
    let hidden = lm.forward_hidden(embeds.clone(), &r_emb, None);
    assert_close("text hidden", hidden, toy.io("text_hidden"));
    assert_close("logits", toy.model.forward_logits(embeds), toy.io("logits"));
}

/// A prompt with the wrong number of placeholders is refused.
#[test]
fn test_prompt_must_match_audio() {
    let toy = Toy::load();
    let audio = toy.model.encode_audio(toy.features(), FEATURE_LEN);
    assert!(
        toy.model
            .prompt_embeds(&[1, AUDIO_TOKEN, 2], audio)
            .is_err()
    );
}

/// The cached decode chooses, at every step, the argmax an uncached forward
/// over the same tokens chooses.
#[test]
fn test_toy_cached_decode_matches_uncached() {
    let toy = Toy::load();
    let audio = toy.model.encode_audio(toy.features(), FEATURE_LEN);
    let mut embeds = toy.model.prompt_embeds(&toy.ids(), audio).unwrap();
    let generated = Qwen3AsrGreedyDecoderConfig::new()
        .with_max_new_tokens(6)
        .with_eos(vec![])
        .init()
        .decode(&toy.model, embeds.clone());
    assert_eq!(generated.len(), 6);

    let lm = &toy.model.language_model;
    for &token in &generated {
        let logits = toy.model.forward_logits(embeds.clone());
        let n = logits.dims()[1];
        let expected = logits.slice_dim(1, n - 1..n).argmax(2).into_scalar::<i64>() as usize;
        assert_eq!(token, expected);
        let next =
            Tensor::<2, Int>::from_data(TensorData::new(vec![token as i64], [1, 1]), &toy.device);
        embeds = Tensor::cat(vec![embeds, lm.embed(next)], 1);
    }
}
