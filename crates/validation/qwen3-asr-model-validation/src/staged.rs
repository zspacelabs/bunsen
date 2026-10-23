//! bunsen's Qwen3-ASR on the base weights, stage by stage, against
//! onnxruntime's run of the export on the same synthetic input.

use bunsen::{
    ops::transformers::attention::KVCacheModel,
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;

/// The audio tower and projector compute what the export's encoder computes,
/// across two attention windows.
#[test]
#[serial_test::serial]
fn test_bunsen_audio_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().encoder;
    let bundle = load_bundle("qwen/qwen3-asr-0.6b", &device);

    let frames = fixture.chunks * CHUNK;
    let features = synthetic_features(frames, &device).squeeze_dim::<2>(0);
    let audio = bundle.model.encode_audio(features, frames);
    fixture
        .audio_embeds
        .assert_matches("audio embeds", audio.unsqueeze_dim::<3>(0), 1e-2, 2e-3);
}

/// The text model embeds as the export does, and computes its logits over a
/// causal prefill and then one cached step.
#[test]
#[serial_test::serial]
fn test_bunsen_text_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference();
    let bundle = load_bundle("qwen/qwen3-asr-0.6b", &device);
    let lm = &bundle.model.language_model;

    let embed = &fixture.embed;
    let ids = Tensor::<2, Int>::from_data(
        TensorData::new(embed.ids.clone(), [1, embed.ids.len()]),
        &device,
    );
    embed
        .inputs_embeds
        .assert_matches("embeds", lm.embed(ids), 1e-6, 1e-6);

    let decoder = &fixture.decoder;
    let seq = decoder.seq;
    let r_emb = lm.rotary(seq + 1, &device);
    let mut cache = lm.kv_cache_config().with_seq_len(seq + 1).init();
    let hidden = lm.forward_hidden(synthetic_embeds(seq, 0, &device), &r_emb, Some(&mut cache));
    decoder
        .prefill_logits
        .assert_matches("prefill logits", lm.logits(hidden), 1e-2, 2e-3);
    let hidden = lm.forward_hidden(synthetic_embeds(1, 1, &device), &r_emb, Some(&mut cache));
    decoder
        .step_logits
        .assert_matches("step logits", lm.logits(hidden), 1e-2, 2e-3);
}
