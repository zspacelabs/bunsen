//! bunsen's float32 greedy decodes against `transformers`', for every row,
//! through every [`KVCacheStorage`].

use bunsen::support::testing::DeviceMemoryGuard;

use super::*;

/// bunsen, fed `transformers`' tokens, predicts each of them at every step
/// but a near-tie ([`NEAR_TIE`]), and its prefill's last logits are
/// `transformers`'.
pub(crate) fn check_row(
    row: &str,
    fixture: &BTreeMap<String, GreedyRecord>,
) {
    let device = f32_device(row);
    let _memory = DeviceMemoryGuard::new(&device);
    let record = &fixture[row];
    let bundle = load_construct(row, &device);
    for storage in STORAGES {
        let steps = bunsen_decode(
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
fn qwen3_5_0_8b() {
    check_row("qwen3.5-0.8b", &transformers_reference());
}

#[test]
#[serial_test::serial]
fn qwen3_5_2b() {
    check_row("qwen3.5-2b", &transformers_reference());
}

#[test]
#[serial_test::serial]
fn qwen3_5_4b() {
    check_row("qwen3.5-4b", &transformers_reference());
}
