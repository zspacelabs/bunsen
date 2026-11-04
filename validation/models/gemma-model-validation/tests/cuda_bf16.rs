//! bunsen's Gemma held in bf16 on CUDA, against `transformers`' bf16 decode
//! of `gemma-2-9b`: also the measure of how far bunsen's bf16 norms drift
//! from `transformers`' (`[adapters_norm_drift]`). Its gains load as `1 + w`
//! and are kept in f32, so the fold is exact, and its RMS norms normalize in
//! f32; what is left is where the two round to bf16.
//!
//! The device is CUDA's, its default float f32; the weights load in bf16
//! under [`LoadPolicy::performance`]. A test binary of its own, so the f32
//! suites stay off CUDA. Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use bunsen::kits::support::construct::LoadPolicy;
use burn::{
    prelude::*,
    tensor::DType,
};
use gemma_model_validation::{
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
        bundle.model.embed_tokens.weight.val().dtype(),
        DType::BF16,
        "{row}: performance() keeps bf16 where CUDA accelerates it"
    );
    for storage in STORAGES {
        let (steps, _) = bunsen_decode(
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
        let steps: Vec<_> = steps.into_iter().map(|s| s.cast(DType::F32)).collect();
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
    fn gemma_2_9b() {
        check_row("gemma-2-9b");
    }
}
