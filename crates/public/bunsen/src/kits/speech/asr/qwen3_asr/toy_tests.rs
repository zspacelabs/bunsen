//! The toy model against `transformers`' own forward.
//!
//! `testdata/qwen3_asr/toy.safetensors` holds a tiny seeded
//! `Qwen3ASRForConditionalGeneration`'s weights and its answers, computed in
//! float64 (`tools/gen_qwen3_asr_toy_fixtures.py`). The weights load through
//! [`load_qwen3_asr_tensors`], so the loader is under test too.

use core::ops::Range;

use burn::prelude::*;

use super::{
    Qwen3AsrConstruct,
    Qwen3AsrGreedyDecoderConfig,
    Qwen3AsrPromptConfig,
    blocks::{
        Qwen3Asr,
        Qwen3AsrAudioEncoderConfig,
        Qwen3AsrConfig,
    },
    interop::transformers::load_qwen3_asr_tensors,
};
use crate::{
    blocks::transformers::models::language_decoder::{
        LmDecoderOps,
        LmKvForwardHidden,
    },
    burner::module::ModuleInit,
    errors::testing::ErrorMatcher,
    kits::{
        language::generation::qwen3::Qwen3ContractConfig,
        speech::asr::qwen3_asr::blocks::Qwen3AsrFrontEndConfig,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
        testing::{
            DecoderToy,
            check_decoder_toy,
        },
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        performance_device,
        tensor_to_vec_as,
        testdata,
    },
};

/// The audio placeholder of the toy's prompt.
const AUDIO_TOKEN: usize = 7;

/// The valid mel frames of the toy's input.
const FEATURE_LEN: usize = 250;

/// `toy.json`'s config.
fn toy_config() -> Qwen3AsrConfig {
    Qwen3AsrConfig::new(
        Qwen3AsrAudioEncoderConfig::new(16, 32, 4, 64, 2, 8).with_n_window_infer(200),
        Qwen3ContractConfig::new(50, 24, 2, 4, 2, 8, 48),
    )
    .with_audio_token(AUDIO_TOKEN)
}

struct Toy {
    model: Qwen3Asr,
    fixture: ToyFixture,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        Self::load_on(&performance_device())
    }

    fn load_on(device: &Device) -> Self {
        let path = testdata!("qwen3_asr/toy.safetensors");
        let mut fixture = ToyFixture::load(path, device);
        let device = device.clone();
        let mut model: Qwen3Asr = toy_config().init(&device);
        let applied = load_qwen3_asr_tensors(&mut model, fixture.take_weights(), None)
            .expect("load the toy weights");
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

    fn features(&self) -> Tensor<2> {
        let f: Tensor<3> = self.io("features");
        let [_, m, t] = f.dims();
        f.reshape([m, t])
    }

    fn ids(&self) -> Vec<usize> {
        let ids: Tensor<2> = self.io("input_ids");
        tensor_to_vec_as::<f32, _>(&ids)
            .into_iter()
            .map(|v| v as usize)
            .collect()
    }
}

/// The text model over the toy's prompt, its audio frames placed.
struct PromptToy<'a> {
    toy: &'a Toy,
    embeds: Tensor<3>,
}

impl DecoderToy for PromptToy<'_> {
    fn positions(&self) -> usize {
        self.embeds.dims()[1]
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
        let r_emb = lm.rotary(self.positions(), &self.toy.device);
        lm.logits(lm.forward_hidden(self.embeds.clone().slice_dim(1, range), &r_emb, cache))
    }
}

#[test]
fn test_toy_audio_tower() {
    let toy = Toy::load();
    let tower = &toy.model.audio_tower;
    let stem = tower.stem(toy.features(), FEATURE_LEN);
    assert_io_close("conv stem", stem.clone(), toy.io("conv"));

    let n = stem.dims()[0];
    let layer0 = tower.stack.layers[0].forward(
        stem.unsqueeze_dim(0),
        tower.stack.rotary(n, &toy.device).for_layer(0),
        &mut None,
    );
    assert_io_close(
        "audio layer 0",
        layer0.squeeze_dim::<2>(0),
        toy.io("audio_layer0"),
    );

    let audio = tower.forward(toy.features(), FEATURE_LEN);
    assert_io_close("audio tower", audio.clone(), toy.io("audio"));
    assert_io_close(
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

    let layer0 =
        lm.lm.decoder.stack.layers[0].forward(embeds.clone(), r_emb.for_layer(0), &mut None);
    assert_io_close("text layer 0", layer0, toy.io("text_layer0"));
    let hidden = lm.forward_hidden(embeds.clone(), &r_emb, None);
    assert_io_close("text hidden", hidden, toy.io("text_hidden"));
    assert_io_close("logits", toy.model.forward_logits(embeds), toy.io("logits"));
}

/// A prompt with the wrong number of placeholders is refused.
#[test]
fn test_prompt_must_match_audio() {
    let toy = Toy::load();
    let audio = toy.model.encode_audio(toy.features(), FEATURE_LEN);
    ErrorMatcher::illegal().assert_err(&toy.model.prompt_embeds(&[1, AUDIO_TOKEN, 2], audio));
}

/// A prefill and then one step per position, through a cache of every
/// storage that starts too small and grows, is `transformers`' uncached
/// forward over the prompt.
#[test]
fn test_toy_text_cached_decode() {
    let toy = Toy::load();
    let audio = toy.model.encode_audio(toy.features(), FEATURE_LEN);
    let embeds = toy.model.prompt_embeds(&toy.ids(), audio).unwrap();
    let logits: Tensor<3> = toy.io("logits");
    check_decoder_toy(&PromptToy { toy: &toy, embeds }, logits.clone(), logits, 3);
}

/// The cached decode chooses, at every step, the argmax an uncached forward
/// over the same tokens chooses.
#[test]
fn test_toy_cached_decode_matches_uncached() {
    let toy = Toy::load();
    let audio = toy.model.encode_audio(toy.features(), FEATURE_LEN);
    let mut embeds = toy.model.prompt_embeds(&toy.ids(), audio).unwrap();
    let generated = Qwen3AsrGreedyDecoderConfig::new(vec![])
        .with_max_new_tokens(6)
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

/// A vocabulary spelling each of the toy's 50 ids as ` t<id>`.
fn toy_vocab() -> crate::kits::support::tokens::ByteLevelVocabulary {
    let entries: Vec<String> = (0..50).map(|k| format!("\"Ġt{k}\": {k}")).collect();
    crate::kits::support::tokens::ByteLevelVocabulary::from_vocab_json(
        &format!("{{{}}}", entries.join(", ")),
        None,
    )
    .unwrap()
}

/// The construct runs the toy end to end, from samples to a parsed transcript;
/// a long recording is the concatenation of its windows' transcriptions.
#[test]
fn test_toy_construct() {
    let toy = Toy::load();
    let device = toy.device.clone();
    let construct = Qwen3AsrConstruct {
        model: toy.model,
        front_end: Qwen3AsrFrontEndConfig::new()
            .with_n_mels(16)
            .with_chunk_seconds(1)
            .init(&device),
        vocab: toy_vocab(),
        prompt: Qwen3AsrPromptConfig::new(vec![1, 2], vec![3, 4], 5),
        decoder: Qwen3AsrGreedyDecoderConfig::new(vec![6]).with_max_new_tokens(4),
        policy: Default::default(),
        presets: Default::default(),
    };
    let samples: Vec<f32> = (0..40_000)
        .map(|k| 0.1 * ((k as f32) * 0.05).sin())
        .collect();

    let one = construct.transcribe(&samples[..16_000], &device).unwrap();
    assert!(!one.tokens.is_empty() && one.tokens.len() <= 4);
    let expected = construct
        .prompt
        .parse(one.tokens.clone(), &[6], &construct.vocab)
        .unwrap();
    assert_eq!(one, expected);

    let long = construct
        .transcribe_long(&samples, 16_000, &device)
        .unwrap();
    let windows = crate::ops::signal::quiet_cuts(&samples, 16_000, 160);
    assert!(windows.len() >= 3);
    let mut tokens = Vec::new();
    for (a, b) in windows {
        tokens.extend(
            construct
                .transcribe(&samples[a..b], &device)
                .unwrap()
                .tokens,
        );
    }
    assert_eq!(long.tokens, tokens);
}

/// A reply's attention onto the audio comes from the positions that predict
/// it: an uncached pass over the prompt and the reply predicts the reply, and
/// each row is part of a distribution.
#[test]
fn test_toy_audio_attention() {
    use super::blocks::{
        Qwen3AsrAudioEncoderMeta,
        Qwen3AsrMeta,
    };

    let toy = Toy::load();
    let device = toy.device.clone();
    let construct = Qwen3AsrConstruct {
        model: toy.model,
        front_end: Qwen3AsrFrontEndConfig::new()
            .with_n_mels(16)
            .with_chunk_seconds(1)
            .init(&device),
        vocab: toy_vocab(),
        prompt: Qwen3AsrPromptConfig::new(vec![1, 2], vec![3, 4], 5),
        decoder: Qwen3AsrGreedyDecoderConfig::new(vec![6]).with_max_new_tokens(4),
        policy: Default::default(),
        presets: Default::default(),
    };
    let samples: Vec<f32> = (0..16_000)
        .map(|k| 0.1 * ((k as f32) * 0.05).sin())
        .collect();
    let (features, valid) = construct.front_end.forward(&samples, &device).unwrap();
    let reply = construct
        .transcribe_features(features.clone(), valid)
        .unwrap()
        .tokens;
    let n_audio = construct.model.audio_tower.output_len(valid);

    let layers = [1, 0];
    let weights = construct
        .audio_attention(features.clone(), valid, &reply, &layers)
        .unwrap();
    assert_eq!(weights.len(), 2);
    for w in &weights {
        assert_eq!(w.dims()[1..], [reply.len(), n_audio]);
        let sums: Vec<f32> = w.clone().sum_dim(2).into_data().try_to_vec().unwrap();
        assert!(
            sums.iter().all(|&m| (0.0..=1.0 + 1e-5).contains(&m)),
            "{sums:?}"
        );
    }
    // a head past the text model's two layers or four heads is refused
    for heads in [[(2, 0)], [(0, 4)]] {
        assert_eq!(
            construct
                .token_frames(features.clone(), valid, &reply, &heads)
                .unwrap_err()
                .kind(),
            crate::errors::BunsenErrorKind::Illegal
        );
    }
    assert_eq!(
        construct
            .token_frames(features.clone(), valid, &reply, &[(1, 3), (0, 0)])
            .unwrap()
            .len(),
        reply.len()
    );

    let audio = construct.model.encode_audio(features, valid);
    let mut ids = construct
        .prompt
        .prompt(n_audio, construct.model.audio_token());
    let rows = ids.len() - 1;
    ids.extend_from_slice(&reply[..reply.len() - 1]);
    let logits = construct
        .model
        .forward_logits(construct.model.prompt_embeds(&ids, audio).unwrap());
    let predicted: Vec<i64> = logits
        .slice_dim(1, rows..rows + reply.len())
        .argmax(2)
        .into_data()
        .convert::<i64>()
        .try_to_vec()
        .unwrap();
    let reply: Vec<i64> = reply.iter().map(|&t| t as i64).collect();
    assert_eq!(predicted, reply);
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
    let audio = half.encode_audio(toy.features(), FEATURE_LEN);
    let embeds = half.prompt_embeds(&toy.ids(), audio).unwrap();
    let logits = half.forward_logits(embeds);
    let expected: Tensor<3> = toy.io("logits");
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
