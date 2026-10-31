//! The generated reference model against onnxruntime's run of the published
//! graph.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// `Qwen3-0.6B`'s export.
mod qwen3_0_6b {
    use super::*;

    /// The generated model computes what onnxruntime computes for the published
    /// graph: a prefill from an empty cache, then one step over the cache it
    /// returned.
    #[test]
    #[serial_test::serial]
    fn matches_onnxruntime() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference();
        let model = reference::Qwen3OnnxModel::load_pretrained(&device);

        let (logits, cache) = reference_decode(
            &model,
            &fixture.ids,
            &empty_reference_cache(&device),
            &device,
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

        let (logits, cache) = reference_decode(&model, &[fixture.step_id], &cache, &device);
        fixture
            .step_logits
            .assert_matches("step logits", logits, 1e-2, 2e-3);
        fixture
            .step_key_27
            .assert_matches("step key 27", cache[27].0.clone(), 1e-2, 1e-3);
    }
}
