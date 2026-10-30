//! `bunsen-chat`'s sessions against `transformers`' greedy chat replies.

use std::sync::Arc;

use bunsen::{
    kits::decoding::SearchConfig,
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};
use bunsen_chat::{
    ChatModel,
    ChatSessionConfig,
    StopReason,
};

use super::*;

/// A greedy session, fed `transformers`' user turns, writes each of its
/// replies: the same prompt ids, the same reply ids, up to a step that is a
/// near-tie ([`NEAR_TIE`]), after which the two conversations may part and
/// the session's remaining turns are not compared. Every turn after the
/// first prefills less than its whole prompt: the cache holds the rest.
fn check_row(row: &str) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = chat_replies_reference();
    let chat = Arc::new(ChatModel::from_bundle(load_bundle(row, &device)).unwrap());

    for session_record in &fixture[row] {
        let config = ChatSessionConfig::new()
            .with_search(Some(SearchConfig::new()))
            .with_enable_thinking(session_record.enable_thinking)
            .with_max_new_tokens(session_record.budget)
            .with_context(Some(1024));
        let mut session = chat.session(config, &device);
        'turns: for (turn, expected) in session_record.replies.iter().enumerate() {
            let what = format!("{row} {}, turn {turn}", session_record.name);
            let mut streamed = String::new();
            let reply = session
                .send(&expected.user, &mut |delta| streamed.push_str(delta))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(streamed, reply.text, "{what}: the stream is the reply");
            assert_eq!(reply.prompt_tokens, expected.prompt_ids.len(), "{what}");
            if turn > 0 {
                assert!(reply.prefilled_tokens < reply.prompt_tokens, "{what}");
            }
            for (k, (&got, &want)) in reply.ids.iter().zip(&expected.ids).enumerate() {
                if got != want {
                    assert!(
                        expected.margins[k] < NEAR_TIE,
                        "{what}, step {k}: {got}, transformers {want} by {}",
                        expected.margins[k]
                    );
                    eprintln!("{what}: parts at a near-tie, step {k}");
                    break 'turns;
                }
            }
            assert_eq!(reply.ids, expected.ids, "{what}");
            assert_eq!(reply.text, expected.text, "{what}");
            let stopped = matches!(reply.stop, StopReason::StopToken(_));
            assert_eq!(
                stopped,
                expected.ids.len() < session_record.budget,
                "{what}"
            );
            eprintln!(
                "{what}: {} tokens agree, {} of {} prompt tokens prefilled",
                reply.ids.len(),
                reply.prefilled_tokens,
                reply.prompt_tokens
            );
        }
    }
}

#[test]
#[serial_test::serial]
fn test_qwen3_0_6b_sessions() {
    check_row("qwen3-0.6b");
}

#[test]
#[serial_test::serial]
fn test_qwen3_4b_instruct_2507_sessions() {
    check_row("qwen3-4b-instruct-2507");
}
