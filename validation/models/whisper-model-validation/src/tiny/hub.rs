//! `hf:openai/whisper-tiny` against `openai/tiny`: the same weights through
//! two files, two layouts and two readers. Fetches both into the weights
//! cache.

use bunsen::{
    data::{
        pretrained::Prefabs,
        resources::{
            Provenance,
            ResourceCache,
            ResourceCacheOptions,
        },
    },
    kits::speech::asr::whisper::{
        WhisperApiConfig,
        WhisperMeta,
        pretrained::{
            CHECKPOINT,
            default_whisper_factory,
        },
    },
    support::testing::{
        assert_tensors_close,
        cpu_device,
    },
};
use burn::tensor::{
    Tensor,
    Tolerance,
};

/// `hf:openai/whisper-tiny` resolves through the hub's listing to a
/// pinned checkpoint, loads to the `tiny` geometry with the
/// multilingual layout and vocabulary, and its weights are
/// `openai/tiny`'s: the same numbers through two files, two layouts and
/// two readers.
#[test]
fn test_hf_whisper_tiny_is_openai_tiny() {
    let device = cpu_device();
    let cache = ResourceCache::new(ResourceCacheOptions::default()).unwrap();
    let factory = default_whisper_factory().unwrap();

    let model = factory.resolve("hf:openai/whisper-tiny", &cache).unwrap();
    let checkpoint = model.to_resource_set();
    let checkpoint = checkpoint.get(CHECKPOINT).unwrap();
    assert_eq!(checkpoint.file, "model.safetensors");
    assert!(checkpoint.is_pinned(), "pinned by the hub's listing");

    let loaded = model
        .load(
            &cache,
            &device,
            bunsen::data::formats::LoadPolicy::load_as_stored(),
        )
        .unwrap();
    assert!(matches!(
        loaded.resources.get(CHECKPOINT).unwrap().provenance,
        Provenance::Cached | Provenance::Downloaded
    ));
    let hf = loaded.handle;
    let tiny = WhisperApiConfig::expect_prefab("tiny").geometry();
    assert_eq!(hf.model.n_mels(), tiny.n_mels);
    assert_eq!(hf.model.vocab_size(), tiny.vocab_size);
    assert_eq!(hf.model.d_model(), tiny.d_model);
    assert_eq!(hf.model.max_audio_ctx(), tiny.max_audio_ctx);
    assert_eq!(hf.model.max_text_ctx(), tiny.max_text_ctx);
    assert_eq!(hf.model.encoder.stack.layers.len(), tiny.n_encoder_layers);
    assert_eq!(hf.model.decoder.blocks.len(), tiny.n_decoder_layers);
    assert!(hf.layout.ids().is_multilingual());
    assert_eq!(hf.ranks.as_ref().map(|r| r.len()), Some(50257));

    let openai = factory
        .load_construct(
            "openai/tiny",
            &cache,
            &device,
            bunsen::data::formats::LoadPolicy::load_as_stored(),
        )
        .unwrap();
    let close = |a: Tensor<2>, b: Tensor<2>| {
        assert_tensors_close(&a, &b, Tolerance::<f32>::default());
    };
    let c_q = |model: &bunsen::kits::speech::asr::whisper::Whisper| {
        model.encoder.stack.layers[0]
            .attn
            .branch
            .attention()
            .c_q
            .weight
            .val()
    };
    close(c_q(&hf.model), c_q(&openai.model));
    close(
        hf.model.decoder.blocks[3]
            .cross_attn
            .branch
            .output
            .weight
            .val(),
        openai.model.decoder.blocks[3]
            .cross_attn
            .branch
            .output
            .weight
            .val(),
    );
    close(
        hf.model.decoder.blocks[1].mlp.branch.linear2.weight.val(),
        openai.model.decoder.blocks[1]
            .mlp
            .branch
            .linear2
            .weight
            .val(),
    );
    close(
        hf.model.encoder.positional_embedding.val(),
        openai.model.encoder.positional_embedding.val(),
    );
    close(
        hf.model.decoder.token_embedding.weight.val(),
        openai.model.decoder.token_embedding.weight.val(),
    );
    close(
        hf.model.decoder.ln.gamma.val().unsqueeze(),
        openai.model.decoder.ln.gamma.val().unsqueeze(),
    );
}
