//! `bunsen_tool::chat`'s sessions against `transformers`' greedy chat replies.

use std::sync::Arc;

use bunsen::{
    kits::support::decoding::SearchConfig,
    support::testing::DeviceMemoryGuard,
};
use bunsen_tool::chat::{
    ChatModel,
    ChatSessionConfig,
    Qwen3_5Chat,
    StopReason,
};

use super::*;

/// A greedy session, fed `transformers`' user turns, writes each of its
/// replies: the same prompt ids, the same reply ids, up to a step that is a
/// near-tie ([`NEAR_TIE`]), after which the two conversations may part and
/// the session's remaining turns are not compared. A reply that stops
/// before its budget stops on a stop token, as `transformers`' did.
///
/// A turn's rendering drops the last reply's empty thinking, so it parts
/// from what the cache holds inside the last reply; the linear layers'
/// states cannot rewind there, so the turn prefills its whole prompt again,
/// from a reset cache.
fn check_row(row: &str) {
    let device = f32_device(row);
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = chat_replies_reference();
    let chat = Arc::new(ChatModel::new(
        Qwen3_5Chat::from_construct(load_construct(row, &device)).unwrap(),
    ));

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
fn qwen3_5_0_8b() {
    check_row("qwen3.5-0.8b");
}
