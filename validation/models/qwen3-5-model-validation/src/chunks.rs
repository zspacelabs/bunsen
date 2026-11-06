//! bunsen's float32 decodes after a prompt of several of the delta rule's
//! chunks against `transformers`': a prefill whose linear layers carry their
//! state from chunk to chunk, and a decode from where it ends, inside one.

use super::*;

#[test]
#[serial_test::serial]
fn qwen3_5_0_8b() {
    crate::greedy::check_row("qwen3.5-0.8b", &transformers_chunks_reference());
}

#[test]
#[serial_test::serial]
fn qwen3_5_2b() {
    crate::greedy::check_row("qwen3.5-2b", &transformers_chunks_reference());
}

#[test]
#[serial_test::serial]
fn qwen3_5_4b() {
    crate::greedy::check_row("qwen3.5-4b", &transformers_chunks_reference());
}
