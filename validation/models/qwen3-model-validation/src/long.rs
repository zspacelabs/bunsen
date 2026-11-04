//! bunsen's float32 decodes at the end of a long multi-turn conversation
//! (`testdata/transformers_long.json`) against `transformers`', for each of
//! [`LONG_F32_ROWS`], through every [`KVCacheStorage`].
//!
//! The prompt is long enough (2263 positions) for rounding to build up over
//! the context, and its last question asks about a ledger line about 2200
//! positions back. bunsen in float32 must decode as `transformers` does in
//! float32; the bf16 runs, which only report today's gap
//! (`tests/cuda_bf16.rs`), are measured against the same fixture.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// bunsen, fed `transformers`' float32 tokens after the long prompt,
/// predicts each of them at every step but a near-tie ([`NEAR_TIE`]), and
/// its prefill's last logits are `transformers`'.
fn check_row(row: &str) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = transformers_long_reference();
    let record = &fixture.rows[row];
    let reference = &record.f32;
    let bundle = load_construct(row, &device);

    for storage in STORAGES {
        let steps = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &reference.ids[..reference.ids.len() - 1],
            storage,
            &device,
        );
        let error = reference.last_logits.assert_matches(
            &format!("{row} {storage:?} long prefill logits"),
            steps[0].clone(),
            1e-2,
            5e-3,
        );
        let mut ties = 0;
        for (k, logits) in steps.iter().enumerate() {
            let got = argmax(logits);
            if got != reference.ids[k] {
                assert!(
                    reference.margins[k] < NEAR_TIE,
                    "{row} {storage:?}, step {k}: {got}, transformers {} by a margin of {}",
                    reference.ids[k],
                    reference.margins[k]
                );
                ties += 1;
            }
        }
        eprintln!(
            "{row} {storage:?} long ({} positions): prefill logits {error:?}; {} of {} tokens agree ({ties} near-ties): {:?}",
            record.prompt_ids.len(),
            steps.len() - ties,
            steps.len(),
            reference.text
        );
    }
}

#[test]
#[serial_test::serial]
fn qwen3_0_6b() {
    check_row("qwen3-0.6b");
}

#[test]
#[serial_test::serial]
fn qwen3_4b() {
    check_row("qwen3-4b");
}
