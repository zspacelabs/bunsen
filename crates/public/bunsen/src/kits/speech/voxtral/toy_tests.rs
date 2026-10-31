//! The toy model against `transformers`' own forward.
//!
//! `testdata/voxtral/toy.safetensors` holds a tiny seeded
//! `VoxtralRealtimeForConditionalGeneration`'s weights, under the names the
//! published checkpoint gives them, and its answers, computed in float64
//! (`tools/gen_voxtral_toy_fixtures.py`). The weights load through
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
    blocks::{
        VoxtralAudioEncoderConfig,
        VoxtralRealtime,
        VoxtralRealtimeConfig,
        VoxtralTextConfig,
    },
    weights::load_hf_tensors,
};
use crate::{
    burner::module::ModuleInit,
    support::testing::performance_device,
};

/// The delays the toy's logits were recorded at; the first is its default.
const DELAYS: [usize; 2] = [6, 2];

/// `toy.json`'s config.
fn toy_config() -> VoxtralRealtimeConfig {
    VoxtralRealtimeConfig::new(
        VoxtralAudioEncoderConfig::new(16, 32, 4, 8, 64, 2).with_sliding_window(Some(5)),
        VoxtralTextConfig::new(64, 48, 2, 4, 2, 10, 96).with_sliding_window(Some(7)),
    )
}

struct Toy {
    model: VoxtralRealtime,
    io: BTreeMap<String, PackTensor>,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/voxtral/toy.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store
            .get_all_tensors()
            .expect("read the toy fixture")
            .clone();
        let (io, weights): (Vec<_>, Vec<_>) =
            all.into_iter().partition(|(n, _)| n.starts_with("io."));
        let device = performance_device();
        let mut model: VoxtralRealtime = toy_config().init(&device);
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

#[test]
fn test_toy_audio_tower() {
    let toy = Toy::load();
    let tower = &toy.model.audio_tower;
    let features: Tensor<3> = toy.io("features");
    let stem = tower.stem(features.clone());
    assert_close("audio stem", stem.clone(), toy.io("audio_stem"));

    let r_emb = tower.rotary(stem.dims()[1], &toy.device);
    let layer0 = tower.layers[0].forward(stem, &r_emb, &mut None);
    assert_close("audio layer 0", layer0, toy.io("audio_layer0"));

    let audio = tower.forward(features.clone());
    assert_close("audio tower", audio, toy.io("audio"));
    assert_close(
        "audio embeds",
        toy.model.audio_embeds(features),
        toy.io("audio_embeds"),
    );
}

#[test]
fn test_toy_text_model() {
    let toy = Toy::load();
    let audio = toy.model.audio_embeds(toy.io("features"));
    let lm = &toy.model.language_model;
    let embeds = lm.embed(toy.ids()) + audio.clone();
    let seq = embeds.dims()[1];
    let r_emb = lm.rotary(seq, &toy.device);
    let cond = lm.delay_condition(DELAYS[0], &toy.device);

    let layer0 =
        lm.layers[0].forward_conditioned(embeds.clone(), &r_emb, &mut None, Some(cond.clone()));
    assert_close("text layer 0", layer0, toy.io("text_layer0"));
    let hidden = lm.forward_hidden(embeds, &r_emb, &cond, None);
    assert_close("text hidden", hidden, toy.io("text_hidden"));
    for delay in DELAYS {
        assert_close(
            &format!("logits at delay {delay}"),
            toy.model.logits(toy.ids(), audio.clone(), delay),
            toy.io(&format!("logits_delay_{delay}")),
        );
    }
}
