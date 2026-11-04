//! bunsen's float32 decodes past each row's sliding window against
//! `transformers`': a prefill longer than the window, then a step per token,
//! through a cache whose sliding layers keep only their window.

use bunsen::support::testing::DeviceMemoryGuard;

use super::*;

/// As [`greedy`](super::greedy)'s check, after a prompt longer than the
/// row's window; and the sliding layers' caches dropped what the window
/// passed.
fn check_row(row: &str) {
    let device = f32_device(row);
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = transformers_window_reference();
    let record = &fixture[row];
    let window = record.window.expect("a window");
    let bundle = load_construct(row, &device);
    for storage in [KVCacheStorage::Auto, KVCacheStorage::MultiLayer] {
        let (steps, retained_from) = bunsen_decode(
            &bundle,
            &record.prompt_ids,
            &record.ids[..record.ids.len() - 1],
            storage,
            &device,
        );
        let total = record.prompt_ids.len() + record.ids.len() - 1;
        assert_eq!(
            retained_from,
            total - (window - 1),
            "{row} {storage:?}: the sliding layers keep their window"
        );
        check_decode(
            &format!(
                "{row} {storage:?}, {} positions past a window of {window}",
                total
            ),
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
