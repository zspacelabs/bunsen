//! The generated reference models against onnxruntime's run of the same graphs.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// The base model's export.
mod qwen3_asr_0_6b {
    use super::*;
    /// The generated encoder computes what onnxruntime computes for the same
    /// graph and input, across two attention windows.
    #[test]
    #[serial_test::serial]
    fn encoder_matches_onnxruntime() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference().encoder;
        let n = fixture.chunks * FRAMES_PER_CHUNK;

        let model = reference::EncoderModel::load_pretrained(&device);
        let valid = Tensor::<1, Int>::arange(0..n as i64, &device);
        let audio = model.forward(
            synthetic_features(fixture.chunks * CHUNK, &device),
            valid,
            window_bias(n, &device),
        );
        fixture
            .audio_embeds
            .assert_matches("audio embeds", audio, 1e-2, 2e-3);
    }

    /// The generated embedding is the export's table.
    #[test]
    #[serial_test::serial]
    fn embed_matches_onnxruntime() {
        let device = performance_device();
        let fixture = onnx_reference().embed;
        let model = reference::EmbedModel::load_pretrained(&device);
        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(fixture.ids.clone(), [1, fixture.ids.len()]),
            &device,
        );
        fixture
            .inputs_embeds
            .assert_matches("embeds", model.forward(ids), 1e-6, 1e-6);
    }

    /// The generated decoder computes what onnxruntime computes: a causal
    /// prefill from an empty cache, then one step over the cache it returned.
    #[test]
    #[serial_test::serial]
    fn decoder_matches_onnxruntime() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference().decoder;
        let seq = fixture.seq;
        let model = reference::DecoderModel::load_pretrained(&device);

        let empty: ReferenceCache = (0..N_LAYERS)
            .map(|_| {
                let t = Tensor::<4>::zeros([1, N_KV_HEADS, 0, HEAD_DIM], &device);
                (t.clone(), t)
            })
            .collect();
        let causal: Vec<f32> = (0..seq * seq)
            .map(|k| {
                if k % seq <= k / seq {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let (logits, cache) = reference_decode(
            &model,
            synthetic_embeds(seq, 0, &device),
            Tensor::from_data(TensorData::new(causal, [1, 1, seq, seq]), &device),
            Tensor::<1, Int>::arange(0..seq as i64, &device).unsqueeze_dim(0),
            &empty,
        );
        fixture
            .prefill_logits
            .assert_matches("prefill logits", logits, 1e-2, 2e-3);
        fixture
            .prefill_key_0
            .assert_matches("prefill key 0", cache[0].0.clone(), 1e-3, 1e-4);
        fixture.prefill_value_27.assert_matches(
            "prefill value 27",
            cache[27].1.clone(),
            1e-2,
            1e-3,
        );

        let (logits, cache) = reference_decode(
            &model,
            synthetic_embeds(1, 1, &device),
            Tensor::zeros([1, 1, 1, seq + 1], &device),
            Tensor::<2, Int>::from_data(TensorData::new(vec![seq as i64], [1, 1]), &device),
            &cache,
        );
        fixture
            .step_logits
            .assert_matches("step logits", logits, 1e-2, 2e-3);
        fixture
            .step_key_27
            .assert_matches("step key 27", cache[27].0.clone(), 1e-2, 1e-3);
    }
}
