//! bunsen's float32 greedy decodes against `transformers`', for every row,
//! through every [`KVCacheStorage`].

use bunsen::support::testing::DeviceMemoryGuard;

use super::*;

/// bunsen, fed `transformers`' tokens, predicts each of them at every step
/// but a near-tie ([`NEAR_TIE`]), and its prefill's last logits are
/// `transformers`'.
fn check_row(row: &str) {
    let device = f32_device(row);
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = transformers_reference();
    let record = &fixture[row];
    let bundle = load_construct(row, &device);
    for storage in STORAGES {
        let (steps, _) = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &record.ids[..record.ids.len() - 1],
            storage,
            &device,
        );
        check_decode(
            &format!("{row} {storage:?}"),
            record,
            &steps,
            (1e-2, 5e-3),
            NEAR_TIE,
        );
    }
}

#[test]
#[serial_test::serial]
fn gemma_3_270m() {
    check_row("gemma-3-270m");
}

#[test]
#[serial_test::serial]
fn gemma_3_1b_pt() {
    check_row("gemma-3-1b-pt");
}

#[test]
#[serial_test::serial]
fn gemma_3_4b_pt() {
    check_row("gemma-3-4b-pt");
}

#[test]
#[serial_test::serial]
fn gemma_2_2b() {
    check_row("gemma-2-2b");
}
