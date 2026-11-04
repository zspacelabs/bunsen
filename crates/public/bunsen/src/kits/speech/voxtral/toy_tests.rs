//! The toy model against `transformers`' own forward.
//!
//! `testdata/voxtral/toy.safetensors` holds a tiny seeded
//! `VoxtralRealtimeForConditionalGeneration`'s weights, under the names the
//! published checkpoint gives them, and its answers, computed in float64
//! (`tools/gen_voxtral_toy_fixtures.py`). The weights load through
//! [`load_hf_tensors`], so the loader is under test too.

use std::ops::Range;

use burn::prelude::*;

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
    fixture: ToyFixture,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        Self::load_on(&performance_device())
    }

    fn load_on(device: &Device) -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/voxtral/toy.safetensors"
        );
        let mut fixture = ToyFixture::load(path, device);
        let device = device.clone();
        let mut model: VoxtralRealtime = toy_config().init(&device);
        let applied =
            load_hf_tensors(&mut model, fixture.take_weights()).expect("load the toy weights");
        assert!(applied.unused.is_empty(), "unused: {:?}", applied.unused);
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

/// The text model at a delay, its input the toy's tokens with the audio
/// embeds added, position by position.
struct TextToy<'a> {
    toy: &'a Toy,
    audio: Tensor<3>,
    cond: Tensor<3>,
}

impl<'a> TextToy<'a> {
    fn new(
        toy: &'a Toy,
        delay: usize,
    ) -> Self {
        Self {
            toy,
            audio: toy.model.audio_embeds(toy.io("features")),
            cond: toy.model.language_model.delay_condition(delay, &toy.device),
        }
    }
}

impl DecoderToy for TextToy<'_> {
    fn positions(&self) -> usize {
        self.toy.ids().dims()[1]
    }

    fn kv_cache_config(&self) -> KVCacheConfig {
        self.toy.model.language_model.kv_cache_config()
    }

    fn logits(
        &self,
        range: Range<usize>,
        cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let lm = &self.toy.model.language_model;
        let embeds = lm.embed(self.toy.ids().slice_dim(1, range.clone()))
            + self.audio.clone().slice_dim(1, range);
        let r_emb = lm.rotary(self.positions(), &self.toy.device);
        lm.logits(lm.forward_hidden(embeds, &r_emb, &self.cond, cache))
    }
}

#[test]
fn test_toy_audio_tower() {
    let toy = Toy::load();
    let tower = &toy.model.audio_tower;
    let features: Tensor<3> = toy.io("features");
    let stem = tower.stem(features.clone());
    assert_io_close("audio stem", stem.clone(), toy.io("audio_stem"));

    let r_emb = tower.rotary(stem.dims()[1], &toy.device);
    let layer0 = tower.stack.layers[0].forward(stem, &r_emb, &mut None);
    assert_io_close("audio layer 0", layer0, toy.io("audio_layer0"));

    let audio = tower.forward(features.clone());
    assert_io_close("audio tower", audio, toy.io("audio"));
    assert_io_close(
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

    let layer0 = lm.stack.layers[0].forward_conditioned(
        embeds.clone(),
        &r_emb,
        &mut None,
        Some(cond.clone()),
    );
    assert_io_close("text layer 0", layer0, toy.io("text_layer0"));
    let hidden = lm.forward_hidden(embeds, &r_emb, &cond, None);
    assert_io_close("text hidden", hidden, toy.io("text_hidden"));
    for delay in DELAYS {
        assert_io_close(
            &format!("logits at delay {delay}"),
            toy.model.logits(toy.ids(), audio.clone(), delay),
            toy.io(&format!("logits_delay_{delay}")),
        );
    }
}

/// At each delay, a prefill and then one step per token, through a cache of
/// every storage that starts too small and grows, is `transformers`'
/// uncached forward; the text model's sliding window (7) is shorter than
/// the toy's eight tokens.
#[test]
fn test_toy_text_cached_decode() {
    let toy = Toy::load();
    for delay in DELAYS {
        let expected: Tensor<3> = toy.io(&format!("logits_delay_{delay}"));
        check_decoder_toy(&TextToy::new(&toy, delay), expected.clone(), expected, 3);
    }
}

/// bf16 parameters on an f32 CPU device, as a `LoadPolicy::as_stored` load
/// places them, fed f32 features: the model runs end to end with no mixed
/// op, which burn-flex refuses, so a missed cast fails here; the logits
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
    let toy = Toy::load_on(&device);
    let half = toy.model.clone().map(&mut DTypeMapper::new(DType::BF16));
    let audio = half.audio_embeds(toy.io("features"));
    let logits = half.logits(toy.ids(), audio, DELAYS[0]);
    let expected: Tensor<3> = toy.io(&format!("logits_delay_{}", DELAYS[0]));
    let gap = (logits.cast(DType::F32) - expected.clone())
        .abs()
        .max()
        .into_scalar::<f32>();
    let scale = expected.abs().max().into_scalar::<f32>();
    assert!(
        gap <= 0.05 * scale,
        "bf16 logits {gap} from f32 (max {scale})"
    );
}
