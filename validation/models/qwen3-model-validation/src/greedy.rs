//! bunsen's greedy decodes against `transformers`', for every row, through
//! every [`KVCacheStorage`].

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// bunsen, fed `transformers`' tokens, predicts each of them at every step
/// but a near-tie ([`NEAR_TIE`]), and its prefill's last logits are
/// `transformers`'.
fn check_row(row: &str) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = transformers_reference();
    let record = &fixture[row];
    let bundle = load_bundle(row, &device);

    for storage in STORAGES {
        let steps = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &record.ids[..record.ids.len() - 1],
            storage,
            &device,
        );
        record.last_logits.assert_matches(
            &format!("{row} {storage:?} prefill logits"),
            steps[0].clone(),
            1e-2,
            5e-3,
        );
        let mut ties = 0;
        for (k, logits) in steps.iter().enumerate() {
            let got = argmax(logits);
            if got != record.ids[k] {
                assert!(
                    record.margins[k] < NEAR_TIE,
                    "{row} {storage:?}, step {k}: {got}, transformers {} by a margin of {}",
                    record.ids[k],
                    record.margins[k]
                );
                ties += 1;
            }
        }
        eprintln!(
            "{row} {storage:?}: {} of {} tokens agree ({ties} near-ties): {:?}",
            steps.len() - ties,
            steps.len(),
            record.text
        );
    }
}

#[test]
#[serial_test::serial]
fn test_qwen3_0_6b_greedy() {
    check_row("qwen3-0.6b");
}

#[test]
#[serial_test::serial]
fn test_qwen3_1_7b_greedy() {
    check_row("qwen3-1.7b");
}

#[test]
#[serial_test::serial]
fn test_qwen3_4b_greedy() {
    check_row("qwen3-4b");
}

#[test]
#[serial_test::serial]
fn test_qwen3_4b_instruct_2507_greedy() {
    check_row("qwen3-4b-instruct-2507");
}
