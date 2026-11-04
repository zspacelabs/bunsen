//! bunsen's Qwen3 held in bf16 on CUDA, against `transformers`' bf16 decodes.
//!
//! `bf16::` holds the short decodes of `testdata/transformers_bf16.json`.
//! `long_bf16::` reports how far bunsen in bf16 is from `transformers` at
//! the end of the long conversation (`testdata/transformers_long.json`), and
//! asserts nothing of it yet: the bf16 load does not keep `transformers`'
//! f32 islands until `LoadPolicy` (`[s5_load_policy]`), and this is the gap
//! it is held to from then on.
//!
//! A test binary of its own: the CUDA device's default float dtype is set to
//! bf16 here, once per process, and must not reach the float32 suite.
//! Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use bunsen::ops::transformers::attention::KVCacheStorage;
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
    load_construct,
    transformers_bf16_reference,
    transformers_long_reference,
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
    let bundle = load_construct(row, &device);
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

/// bunsen in bf16 at the end of the long conversation, fed `transformers`'
/// bf16 tokens: how far its prefill's last logits are from `transformers`'
/// bf16 and float32 ones, beside `transformers`' own bf16 error, and how many
/// of the tokens it predicts as `transformers` in bf16 does. Reported, not
/// asserted (the module's docs).
fn report_long(row: &str) {
    let device = bf16_device();
    let fixture = transformers_long_reference();
    let record = &fixture.rows[row];
    let reference = &record.bf16;
    let bundle = load_construct(row, &device);
    let steps = bunsen_decode(
        &bundle,
        &record.prompt_ids,
        &reference.ids[..reference.ids.len() - 1],
        KVCacheStorage::Auto,
        &device,
    );
    let prefill = steps[0].clone().cast(DType::F32);
    let vs_bf16 = reference.last_logits.max_abs_error(prefill.clone());
    let vs_f32 = record.f32.last_logits.max_abs_error(prefill);
    let predicted: Vec<i64> = steps.iter().map(argmax).collect();
    let agree = predicted
        .iter()
        .zip(&reference.ids)
        .filter(|(got, want)| got == want)
        .count();
    let first = predicted
        .iter()
        .zip(&reference.ids)
        .position(|(got, want)| got != want);
    eprintln!(
        "{row} long bf16 ({} positions): prefill logits {vs_bf16} from transformers' bf16, \
         {vs_f32} from its float32 (transformers' own bf16 is {:.1e} from its float32); \
         {agree} of {} tokens as transformers' bf16, first apart at {first:?} \
         (transformers' bf16 and float32 decodes agree on {})",
        record.prompt_ids.len(),
        record.bf16_vs_f32_last_logits_max_abs,
        predicted.len(),
        record.bf16_f32_agree,
    );
}

/// Each of the long fixture's rows in bf16, under its id.
mod long_bf16 {
    use super::*;

    #[test]
    fn qwen3_0_6b() {
        report_long("qwen3-0.6b");
    }

    #[test]
    fn qwen3_4b() {
        report_long("qwen3-4b");
    }

    /// The 8B's weights and a 2263-position prefill fill a 24 GB card: run
    /// it on one nothing else uses (`CUDA_VISIBLE_DEVICES`), as a desktop's
    /// 1.4 GB is enough to run it out of memory.
    #[test]
    fn qwen3_8b() {
        report_long("qwen3-8b");
    }
}
