//! bunsen's nanochat held in bf16 on CUDA, against upstream's bf16 decodes.
//!
//! A test binary of its own: the CUDA device's default float dtype is set to
//! bf16 here, once per process, and must not reach the float32 suite.
//! Needs `download`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "download", feature = "gpu-tests", feature = "cuda"))]

use std::sync::Arc;

use bunsen::{
    data::pretrained::read_pytorch_tensors,
    kits::gpts::nanochat::{
        CHECKPOINT,
        NanoChatBundle,
    },
};
use burn::{
    prelude::*,
    store::{
        ModuleSnapshot,
        bridge,
    },
    tensor::{
        DType,
        DeviceConfig,
        FloatDType,
    },
};
use nanochat_model_validation::{
    BF16_NEAR_TIE,
    ROWS,
    STORAGES,
    argmax,
    bunsen_decode,
    load_bundle,
    upstream_bf16_reference,
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

/// The checkpoint's bf16 tensors are the model's, bit for bit: loaded as
/// they are, never through f32.
fn assert_bf16_tensors_load_as_they_are(
    row: &str,
    bundle: &Arc<NanoChatBundle>,
) {
    let path = bundle.file(CHECKPOINT).expect("the checkpoint's path");
    let file = read_pytorch_tensors(path).expect("read the checkpoint");
    let wte = file
        .iter()
        .find(|t| t.name == "transformer.wte.weight")
        .expect("the embedding");
    assert_eq!(
        wte.dtype,
        DType::BF16,
        "{row}: the file's embedding is bf16"
    );
    let model = bundle
        .model
        .collect(None, None, true)
        .into_iter()
        .find(|t| t.name == "wte.weight")
        .expect("the model's embedding");
    let expected = bridge::to_data(wte).expect("the file's embedding");
    let actual = bridge::to_data(&model).expect("the model's embedding");
    assert_eq!(
        actual.dtype(),
        DType::BF16,
        "{row}: the model's embedding is bf16"
    );
    assert_eq!(actual.as_bytes(), expected.as_bytes(), "{row}: bit for bit");
}

/// bunsen in bf16, fed upstream's bf16 tokens, predicts each of them at
/// every step but a bf16 near-tie ([`BF16_NEAR_TIE`]), and its prefill's last
/// logits are upstream's within bf16's reach: two bf16 implementations round
/// differently.
fn check_row(row: &str) {
    let device = bf16_device();
    let fixture = upstream_bf16_reference();
    let record = &fixture.rows[row];
    let bundle = load_bundle(row, &device);
    assert_bf16_tensors_load_as_they_are(row, &bundle);

    for storage in STORAGES {
        let steps = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &record.ids[..record.ids.len() - 1],
            storage,
            &device,
        );
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
                    "{row} {storage:?}, step {k}: {got}, upstream {} by a margin of {}",
                    record.ids[k],
                    record.margins[k]
                );
                ties += 1;
            }
        }
        eprintln!(
            "{row} {storage:?} bf16 (upstream on torch {}): prefill logits {error:?}; {} of {} tokens agree ({ties} near-ties)",
            fixture.torch,
            steps.len() - ties,
            steps.len(),
        );
    }
}

#[test]
fn test_bf16_rows() {
    for row in ROWS {
        check_row(row);
    }
}
