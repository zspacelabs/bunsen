//! `bunsen_tool::chat`'s nanochat sessions against upstream's greedy chat
//! replies.

use std::sync::Arc;

use bunsen::{
    kits::decoding::SearchConfig,
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};
use bunsen_tool::chat::{
    ChatModel,
    ChatSessionConfig,
    NanoChatChat,
    StopReason,
};

use super::*;

/// A greedy session, fed upstream's user turns, writes each of its replies:
/// the same prompt length, the same reply ids, up to a step that is a
/// near-tie ([`NEAR_TIE`]), after which the two conversations may part and
/// the session's remaining turns are not compared. Every turn after the
/// first prefills less than its whole prompt: the cache holds the rest, and
/// a smeared model (the 2026 one) carries the token before the cut.
fn check_row(row: &str) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = chat_replies_reference();
    let chat = Arc::new(ChatModel::new(
        NanoChatChat::from_bundle(load_bundle(row, &device)).unwrap(),
    ));

    for (n, turns) in fixture[row].iter().enumerate() {
        let config = ChatSessionConfig::new()
            .with_search(Some(SearchConfig::new()))
            .with_max_new_tokens(REPLY_TOKENS);
        let mut session = chat.session(config, &device);
        'turns: for (turn, expected) in turns.iter().enumerate() {
            let what = format!("{row} chat {n}, turn {turn}");
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
                        "{what}, step {k}: {got}, upstream {want} by {}",
                        expected.margins[k]
                    );
                    eprintln!("{what}: parts at a near-tie, step {k}");
                    break 'turns;
                }
            }
            assert_eq!(reply.ids, expected.ids, "{what}");
            assert_eq!(reply.text, expected.text, "{what}");
            let stopped = matches!(reply.stop, StopReason::StopToken(_));
            assert_eq!(stopped, expected.ids.len() < REPLY_TOKENS, "{what}");
            eprintln!(
                "{what}: {} tokens agree, {} of {} prompt tokens prefilled: {:?}",
                reply.ids.len(),
                reply.prefilled_tokens,
                reply.prompt_tokens,
                reply.text
            );
        }
    }
}

#[test]
#[serial_test::serial]
fn test_rl_d20_sessions() {
    check_row("nanochat-students/rl-d20");
}

#[test]
#[serial_test::serial]
fn test_d18_sft_sessions() {
    check_row("nitishpandey04/nanochat-d18-sft");
}
