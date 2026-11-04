//! bunsen's float32 decodes at the end of a long multi-turn conversation
//! (`testdata/transformers_long.json`) against `transformers`', for each of
//! [`LONG_F32_ROWS`], through every [`KVCacheStorage`].
//!
//! The prompt is long enough (2048 positions or more) for rounding to build
//! up over the context, and its last question asks about a ledger line far
//! back, past `gemma-3-1b-it`'s 512-position window. bunsen in float32 must
//! decode as `transformers` does in float32; the bf16 runs
//! (`tests/cuda_bf16.rs`, `long_bf16::`) are measured against the same
//! fixture.

use bunsen::support::testing::DeviceMemoryGuard;

use super::*;

/// bunsen, fed `transformers`' float32 tokens after the long prompt,
/// predicts each of them at every step but a near-tie ([`NEAR_TIE`]), and
/// its prefill's last logits are `transformers`'.
fn check_row(row: &str) {
    let device = f32_device(row);
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = transformers_long_reference();
    let record = &fixture.rows[row];
    let reference = &record.f32;
    let bundle = load_construct(row, &device);

    for storage in STORAGES {
        let (steps, _) = bunsen_decode(
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
fn gemma_3_1b_it() {
    check_row("gemma-3-1b-it");
}

#[test]
#[serial_test::serial]
fn gemma_2_2b_it() {
    check_row("gemma-2-2b-it");
}
