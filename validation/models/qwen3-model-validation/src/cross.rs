//! bunsen's `Qwen3-0.6B` against onnxruntime's run of the published graph,
//! and against the generated model step by step over a long decode, through
//! every [`KVCacheStorage`].

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// The steps of [`qwen3_0_6b::tracks_the_reference`]'s decode.
#[cfg(feature = "reference")]
const LONG_DECODE: usize = 48;

/// The largest logit difference [`qwen3_0_6b::tracks_the_reference`] allows
/// at any step.
#[cfg(feature = "reference")]
const LONG_DECODE_ATOL: f32 = 5e-3;

/// `Qwen3-0.6B`: the checkpoint the export is of.
mod qwen3_0_6b {
    use super::*;

    /// bunsen's logits over the fixture's prefill, every position, and then its
    /// step, are onnxruntime's.
    #[test]
    #[serial_test::serial]
    fn matches_onnxruntime() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference();
        let bundle = load_bundle("qwen/qwen3-0.6b", &device);
        let model = &bundle.model;
        let seq = fixture.ids.len();

        let r_emb = model.rotary(seq + 1, &device);
        let mut cache = model.kv_cache_config().with_seq_len(seq + 1).init();
        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(fixture.ids.clone(), [1, seq]), &device);
        let hidden = model.forward_hidden(model.embed(ids), &r_emb, Some(&mut cache));
        fixture
            .prefill_logits
            .assert_matches("prefill logits", model.logits(hidden), 1e-2, 2e-3);

        let step =
            Tensor::<2, Int>::from_data(TensorData::new(vec![fixture.step_id], [1, 1]), &device);
        let hidden = model.forward_hidden(model.embed(step), &r_emb, Some(&mut cache));
        fixture
            .step_logits
            .assert_matches("step logits", model.logits(hidden), 1e-2, 2e-3);
    }

    /// Over a greedy decode of [`LONG_DECODE`] steps, through a cache that
    /// grows every [`CACHE_CHUNK`] positions, bunsen's logits at every step are
    /// the generated model's, under every storage.
    #[cfg(feature = "reference")]
    #[test]
    #[serial_test::serial]
    fn tracks_the_reference() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference();
        let bundle = load_bundle("qwen/qwen3-0.6b", &device);
        let reference = reference::Qwen3OnnxModel::load_pretrained(&device);

        // The reference decodes greedily; bunsen is fed the same tokens.
        let mut fed = Vec::new();
        let mut expected = Vec::new();
        let (logits, mut cache) = reference_decode(
            &reference,
            &fixture.ids,
            &empty_reference_cache(&device),
            &device,
        );
        let n = logits.dims()[1];
        let mut last: Tensor<2> = logits.slice_dim(1, n - 1..n).squeeze_dim(1);
        for _ in 0..LONG_DECODE {
            let id = argmax(&last);
            expected.push(last);
            fed.push(id);
            let (logits, next) = reference_decode(&reference, &[id], &cache, &device);
            cache = next;
            last = logits.squeeze_dim(1);
        }
        expected.push(last);
        drop(cache);
        drop(reference);

        for storage in STORAGES {
            let actual = bunsen_decode(&bundle, &fixture.ids, &fed, storage, &device);
            for (step, (a, e)) in actual.iter().zip(&expected).enumerate() {
                let worst = (a.clone() - e.clone()).abs().max().into_scalar::<f32>();
                assert!(
                    worst <= LONG_DECODE_ATOL,
                    "{storage:?}, step {step}: max |logit difference| {worst}"
                );
            }
            eprintln!("{storage:?}: {} steps agree", actual.len());
        }
    }
}
