//! The generated reference models against onnxruntime's run of the published
//! graphs.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// The generated audio encoder computes what onnxruntime computes for the
/// published graph: a call from empty caches past its attention window, then
/// a second through the caches the first returned.
#[test]
#[serial_test::serial]
fn test_encoder_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().encoder;
    let model = reference::VoxtralAudioEncoderOnnx::load_pretrained(&device);

    let (embeds, cache) = reference_encode(
        &model,
        synthetic_features(fixture.frames, 0, &device),
        &EncoderCache::empty(&device),
    );
    fixture.embeds.assert_matches("embeds", embeds, 1e-3, 1e-3);
    fixture
        .padding_cache
        .assert_matches("padding cache", cache.padding.clone(), 1e-4, 1e-4);
    fixture
        .key_0
        .assert_matches("key 0", cache.kv[0].0.clone(), 1e-3, 1e-3);
    fixture
        .value_31
        .assert_matches("value 31", cache.kv[31].1.clone(), 1e-3, 1e-3);

    let (embeds, cache) = reference_encode(
        &model,
        synthetic_features(fixture.step_frames, fixture.frames, &device),
        &cache,
    );
    fixture
        .step_embeds
        .assert_matches("step embeds", embeds, 1e-3, 1e-3);
    fixture.step_padding_cache.assert_matches(
        "step padding cache",
        cache.padding.clone(),
        1e-4,
        1e-4,
    );
    fixture
        .step_key_31
        .assert_matches("step key 31", cache.kv[31].0.clone(), 1e-3, 1e-3);
}

/// The generated token embedding is onnxruntime's.
#[test]
#[serial_test::serial]
fn test_embed_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().embed;
    let model = reference::VoxtralEmbedTokensOnnx::load_pretrained(&device);

    fixture.embeds.assert_matches(
        "embeds",
        reference_embed(&model, &fixture.ids, &device),
        1e-6,
        1e-6,
    );
}

/// The generated text decoder computes what onnxruntime computes for the
/// published graph: a prefill from an empty cache, then one step over the
/// cache it returned, choosing the same tokens.
#[test]
#[serial_test::serial]
fn test_decoder_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().decoder;
    let model = reference::VoxtralDecoderOnnx::load_pretrained(&device);

    let (logits, cache) = reference_decode(
        &model,
        synthetic_embeds(fixture.prefill, 0, &device),
        &empty_decoder_cache(&device),
    );
    let argmax: Vec<i64> = logits
        .clone()
        .argmax(2)
        .into_data()
        .try_into_vec_as::<i64>()
        .unwrap();
    assert_eq!(argmax, fixture.argmax, "prefill argmax");
    fixture
        .logits
        .assert_matches("prefill logits", logits, 1e-2, 2e-3);
    fixture
        .key_0
        .assert_matches("prefill key 0", cache[0].0.clone(), 1e-3, 1e-4);
    fixture
        .value_25
        .assert_matches("prefill value 25", cache[25].1.clone(), 1e-2, 1e-3);

    let (logits, cache) = reference_decode(
        &model,
        synthetic_embeds(1, fixture.prefill, &device),
        &cache,
    );
    let step: i64 = logits.clone().argmax(2).into_scalar::<i64>();
    assert_eq!(step, fixture.step_argmax, "step argmax");
    fixture
        .step_logits
        .assert_matches("step logits", logits, 1e-2, 2e-3);
    fixture
        .step_key_25
        .assert_matches("step key 25", cache[25].0.clone(), 1e-2, 1e-3);
}
