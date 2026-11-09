//! bunsen's Gemma held in bf16 on CUDA, against `transformers`' bf16 decodes.
//!
//! `bf16::` holds the short decode of `gemma-2-9b`: also the measure of how
//! far bunsen's bf16 norms drift from `transformers`'
//! (`[adapters_norm_drift]`). `long_bf16::` holds bunsen in bf16 to
//! `transformers` at the end of the long conversation
//! (`testdata/transformers_long.json`): its prefill's logits within twice
//! `transformers`' own bf16 error from f32, and all but [`LONG_BF16_MISSES`]
//! of the tokens. Its gains load as `1 + w`
//! and are kept in f32, so the fold is exact, and its RMS norms normalize in
//! f32; what is left is where the two round to bf16.
//!
//! The device is CUDA's, its default float f32; the weights load in bf16
//! under [`LoadPolicy::performance`]. A test binary of its own, so the f32
//! suites stay off CUDA. Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use bunsen::{
    kits::support::construct::LoadPolicy,
    ops::transformers::attention::KVCacheStorage,
};
use burn::{
    prelude::*,
    tensor::DType,
};
use gemma_model_validation::{
    BF16_NEAR_TIE,
    STORAGES,
    argmax,
    bunsen_decode,
    check_decode,
    load_construct_with,
    transformers_bf16_reference,
    transformers_long_reference,
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
    let device = Device::cuda(0);
    let fixture = transformers_long_reference();
    let record = &fixture.rows[row];
    let reference = &record.bf16;
    let bundle = load_construct_with(row, &device, LoadPolicy::performance());
    assert_eq!(
        bundle.model.lm.embedding.weight.val().dtype(),
        DType::BF16,
        "{row}: performance() keeps bf16 where CUDA accelerates it"
    );
    let (steps, _) = bunsen_decode(
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
    #[serial_test::serial]
    fn gemma_3_1b_it() {
        check_long("gemma-3-1b-it");
    }

    #[test]
    #[serial_test::serial]
    fn gemma_2_2b_it() {
        check_long("gemma-2-2b-it");
    }

    /// The 9B's bf16 weights and a 2048-position prefill nearly fill a 24 GB
    /// card: run it on one nothing else uses (`CUDA_VISIBLE_DEVICES`).
    #[test]
    #[serial_test::serial]
    fn gemma_2_9b_it() {
        check_long("gemma-2-9b-it");
    }
}
