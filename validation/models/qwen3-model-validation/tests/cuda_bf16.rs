//! bunsen's Qwen3 held in bf16 on CUDA, against `transformers`' bf16 decodes.
//!
//! A test binary of its own: the CUDA device's default float dtype is set to
//! bf16 here, once per process, and must not reach the float32 suite.
//! Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use burn::{
    prelude::*,
    tensor::{
        DType,
        DeviceConfig,
        FloatDType,
    },
};
use qwen3_model_validation::{
    BF16_NEAR_TIE,
    STORAGES,
    argmax,
    bunsen_decode,
    load_bundle,
    transformers_bf16_reference,
};

/// The bf16 CUDA device, configured once for this process.
fn bf16_device() -> Device {
    let mut device = Device::cuda(0);
    match device.configure(DeviceConfig::default().float_dtype(FloatDType::BF16)) {
        Ok(()) => {}
        Err(_) => assert_eq!(device.settings().float_dtype, FloatDType::BF16),
    }
    device
}

/// bunsen in bf16, fed `transformers`' bf16 tokens, predicts each of them at
/// every step but a bf16 near-tie ([`BF16_NEAR_TIE`]), and its prefill's last
/// logits are `transformers`' within bf16's reach: two bf16 implementations
/// round differently.
fn check_row(row: &str) {
    let device = bf16_device();
    let fixture = transformers_bf16_reference();
    let record = &fixture[row];
    let bundle = load_bundle(row, &device);
    assert!(bundle.model.lm_head.is_some(), "{row}: the head is its own");

    for storage in STORAGES {
        let steps = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &record.ids[..record.ids.len() - 1],
            storage,
            &device,
        );
        assert_eq!(steps[0].dtype(), DType::BF16, "{row}: the logits are bf16");
        let error = record.last_logits.assert_matches(
            &format!("{row} {storage:?} bf16 prefill logits"),
            steps[0].clone().cast(DType::F32),
            5e-2,
            0.5,
        );
        let mut ties = 0;
        for (k, logits) in steps.iter().enumerate() {
            let got = argmax(logits);
            if got != record.ids[k] {
                assert!(
                    record.margins[k] < BF16_NEAR_TIE,
                    "{row} {storage:?}, step {k}: {got}, transformers {} by a margin of {}",
                    record.ids[k],
                    record.margins[k]
                );
                ties += 1;
            }
        }
        eprintln!(
            "{row} {storage:?} bf16 (transformers' own bf16 vs f32: {:?}): prefill logits {error:?}; {} of {} tokens agree ({ties} near-ties)",
            record.f32_last_logits_max_abs,
            steps.len() - ties,
            steps.len(),
        );
    }
}

/// Each row in bf16, under its id.
mod bf16 {
    use super::*;

    #[test]
    fn qwen3_8b() {
        check_row("qwen3-8b");
    }
}
