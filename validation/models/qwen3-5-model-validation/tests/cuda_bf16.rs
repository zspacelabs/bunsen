//! bunsen's Qwen3.5 held in bf16 on CUDA, against `transformers`' bf16
//! decodes: the short decode of `qwen3.5-9b`, its prefill's logits within
//! twice `transformers`' own bf16 error from f32, and every token but a bf16
//! near-tie.
//!
//! The device is CUDA's, its default float f32; the weights load in bf16
//! under [`LoadPolicy::performance`]. A test binary of its own, so the f32
//! suites stay off CUDA. Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use bunsen::data::formats::LoadPolicy;
use burn::{
    prelude::*,
    tensor::DType,
};
use qwen3_5_model_validation::{
    BF16_NEAR_TIE,
    STORAGES,
    bunsen_decode,
    check_decode,
    load_construct_with,
    transformers_bf16_reference,
};

fn check_row(row: &str) {
    let device = Device::cuda(0);
    let fixture = transformers_bf16_reference();
    let record = &fixture[row];
    let own = record.f32_last_logits_max_abs.expect("bf16's own error");
    let bundle = load_construct_with(row, &device, LoadPolicy::performance());
    assert_eq!(
        bundle.model.lm.embedding.weight.val().dtype(),
        DType::BF16,
        "{row}: performance() keeps bf16 where CUDA accelerates it"
    );
    for storage in STORAGES {
        let steps = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &record.ids[..record.ids.len() - 1],
            storage,
            &device,
        );
        assert_eq!(
            steps[0].dtype(),
            DType::F32,
            "{row}: the logits leave in f32"
        );
        check_decode(
            &format!("{row} {storage:?} bf16 (transformers' own bf16 error {own:.3})"),
            record,
            &steps,
            (5e-2, (2.0 * own).max(0.5)),
            BF16_NEAR_TIE,
        );
    }
}

mod bf16 {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn qwen3_5_9b() {
        check_row("qwen3.5-9b");
    }
}
