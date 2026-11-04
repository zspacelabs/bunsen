//! bunsen's Qwen3 held in bf16 on CUDA, against `transformers`' bf16 decodes.
//!
//! `bf16::` holds the short decodes of `testdata/transformers_bf16.json`.
//! `long_bf16::` holds bunsen in bf16 to `transformers` at the end of the
//! long conversation (`testdata/transformers_long.json`): its prefill's
//! logits within twice `transformers`' own bf16 error from f32, and all but
//! [`LONG_BF16_MISSES`] of the tokens.
//!
//! The device is CUDA's, its default float f32; the weights load in bf16
//! under [`LoadPolicy::performance`], as CUDA accelerates bf16, and the
//! model keeps `transformers`' f32 islands (the softmax, the rope tables,
//! the norms, the logits). A test binary of its own, so the f32 suites stay
//! off CUDA. Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use bunsen::{
    kits::support::construct::LoadPolicy,
    ops::transformers::attention::KVCacheStorage,
};
use burn::{
    prelude::*,
    tensor::DType,
};
use qwen3_model_validation::{
    BF16_NEAR_TIE,
    STORAGES,
    argmax,
    bunsen_decode,
    load_construct_with,
    transformers_bf16_reference,
    transformers_long_reference,
};

/// The CUDA device, its default float f32: the weights' dtype is the load
/// policy's.
fn bf16_device() -> Device {
    Device::cuda(0)
}

/// bunsen in bf16, fed `transformers`' bf16 tokens, predicts each of them at
/// every step but a bf16 near-tie ([`BF16_NEAR_TIE`]), and its prefill's last
/// logits are `transformers`' within bf16's reach: two bf16 implementations
/// round differently.
fn check_row(row: &str) {
    let device = bf16_device();
    let fixture = transformers_bf16_reference();
    let record = &fixture[row];
    let bundle = load_construct_with(row, &device, LoadPolicy::performance());
    assert_eq!(
        bundle.model.embed_tokens.weight.val().dtype(),
        DType::BF16,
        "{row}: performance() keeps bf16 where CUDA accelerates it"
    );
    assert!(bundle.model.lm_head.proj.is_some(), "{row}: the head is its own");

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
            "{row}: the logits leave the model in f32"
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

/// The tokens bunsen in bf16 may predict otherwise than `transformers` in
/// bf16 at the end of the long conversation: two bf16 implementations round
/// differently, and a near-tie late in the decode may go either way.
const LONG_BF16_MISSES: usize = 2;

/// bunsen in bf16 at the end of the long conversation, fed `transformers`'
/// bf16 tokens: its prefill's last logits are within twice `transformers`'
/// own bf16 error (its bf16 logits' distance from its f32 ones) of
/// `transformers`' bf16 logits, and it predicts all but
/// [`LONG_BF16_MISSES`] of the tokens as `transformers` in bf16 does.
fn check_long(row: &str) {
    let device = bf16_device();
    let fixture = transformers_long_reference();
    let record = &fixture.rows[row];
    let reference = &record.bf16;
    let bundle = load_construct_with(row, &device, LoadPolicy::performance());
    assert_eq!(
        bundle.model.embed_tokens.weight.val().dtype(),
        DType::BF16,
        "{row}: performance() keeps bf16 where CUDA accelerates it"
    );
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
    assert!(
        vs_bf16.abs <= 2.0 * record.bf16_vs_f32_last_logits_max_abs,
        "{row}: prefill logits {vs_bf16} from transformers' bf16, over twice its own bf16 error {}",
        record.bf16_vs_f32_last_logits_max_abs
    );
    assert!(
        agree + LONG_BF16_MISSES >= predicted.len(),
        "{row}: {agree} of {} tokens as transformers' bf16",
        predicted.len()
    );
}

/// Each of the long fixture's rows in bf16, under its id.
mod long_bf16 {
    use super::*;

    #[test]
    fn qwen3_0_6b() {
        check_long("qwen3-0.6b");
    }

    #[test]
    fn qwen3_4b() {
        check_long("qwen3-4b");
    }

    /// The 8B's weights and a 2263-position prefill fill a 24 GB card: run
    /// it on one nothing else uses (`CUDA_VISIBLE_DEVICES`), as a desktop's
    /// 1.4 GB is enough to run it out of memory.
    #[test]
    fn qwen3_8b() {
        check_long("qwen3-8b");
    }
}
