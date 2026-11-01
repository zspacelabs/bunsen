//! bunsen's greedy decodes against upstream's, for every row, through every
//! [`KVCacheStorage`].

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// bunsen, fed upstream's tokens, predicts each of them at every step but a
/// near-tie ([`NEAR_TIE`]), and its prefill's last logits are upstream's.
fn check_row(row: &str) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = upstream_reference();
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
        let error = record.last_logits.assert_matches(
            &format!("{row} {storage:?} prefill logits"),
            steps[0].clone(),
            1e-3,
            1e-3,
        );
        let mut ties = 0;
        for (k, logits) in steps.iter().enumerate() {
            let got = argmax(logits);
            if got != record.ids[k] {
                assert!(
                    record.margins[k] < NEAR_TIE,
                    "{row} {storage:?}, step {k}: {got}, upstream {} by a margin of {}",
                    record.ids[k],
                    record.margins[k]
                );
                ties += 1;
            }
        }
        eprintln!(
            "{row} {storage:?}: prefill logits {error:?}; {} of {} tokens agree ({ties} near-ties): {:?}",
            steps.len() - ties,
            steps.len(),
            record.text
        );
    }
}

#[test]
#[serial_test::serial]
fn nanochat_d20_base() {
    check_row("nanochat-students/base-d20");
}

#[test]
#[serial_test::serial]
fn nanochat_d20_mid() {
    check_row("nanochat-students/mid-d20");
}

#[test]
#[serial_test::serial]
fn nanochat_d20_rl() {
    check_row("nanochat-students/rl-d20");
}

#[test]
#[serial_test::serial]
fn nanochat_d18_base() {
    check_row("nitishpandey04/nanochat-d18-base");
}

#[test]
#[serial_test::serial]
fn nanochat_d18_sft() {
    check_row("nitishpandey04/nanochat-d18-sft");
}
