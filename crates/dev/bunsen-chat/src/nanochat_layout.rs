//! nanochat's conversation layout: a conversation as the ids its model reads.
use bunsen::errors::{
    BunsenError,
    BunsenResult,
};

use crate::{
    ChatMessage,
    NanoChatTokenizer,
};

/// nanochat's conversation layout, as upstream's tokenizer renders one
/// (`render_conversation`, and `render_for_completion` for the assistant's
/// next turn): `<|bos|>`, then each user turn between `<|user_start|>` and
/// `<|user_end|>` and each assistant turn between `<|assistant_start|>` and
/// `<|assistant_end|>`, the turns alternating from a user's; a system
/// message is folded into the first user turn, `system + "\n\n" + user`.
/// The markers are their special tokens, and a message's text is encoded as
/// text, a special token's spelling included.
///
/// Upstream's own tool calls (`<|python_start|>` and the calculator's
/// `<|output_start|>`) are its engine's, not the layout's: a reply that
/// opens one shows it as text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NanoChatLayout {
    /// `<|bos|>`: the conversation's first token, and a reply's end.
    pub bos: i64,
    /// `<|user_start|>`.
    pub user_start: i64,
    /// `<|user_end|>`.
    pub user_end: i64,
    /// `<|assistant_start|>`: the next turn's prompt ends with it.
    pub assistant_start: i64,
    /// `<|assistant_end|>`: a reply's end.
    pub assistant_end: i64,
}

impl NanoChatLayout {
    /// The layout of `tokenizer`'s special tokens.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// naming a special token the tokenizer lacks.
    pub fn from_tokenizer(tokenizer: &NanoChatTokenizer) -> BunsenResult<Self> {
        let special = |name: &str| {
            tokenizer.special(name).ok_or_else(|| {
                BunsenError::invalid_resource(format!("a nanochat tokenizer without {name}"))
            })
        };
        Ok(Self {
            bos: special("<|bos|>")?,
            user_start: special("<|user_start|>")?,
            user_end: special("<|user_end|>")?,
            assistant_start: special("<|assistant_start|>")?,
            assistant_end: special("<|assistant_end|>")?,
        })
    }

    /// The tokens that end a reply, as upstream's engine stops one:
    /// `<|assistant_end|>`, then `<|bos|>`.
    pub fn stop_ids(&self) -> Vec<i64> {
        vec![self.assistant_end, self.bos]
    }

    /// `messages` as the prompt of the assistant's next turn: upstream's
    /// `render_for_completion` of them.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) when the
    /// conversation is empty, does not alternate from a user's turn (a system
    /// message first aside), or ends with the assistant's; as the tokenizer.
    pub fn render(
        &self,
        tokenizer: &NanoChatTokenizer,
        messages: &[ChatMessage],
    ) -> BunsenResult<Vec<i64>> {
        let folded;
        let messages = match messages {
            [system, user, rest @ ..] if system.role == "system" => {
                if user.role != "user" {
                    return Err(BunsenError::illegal(
                        "a nanochat conversation's system message must be followed by a user's",
                    ));
                }
                folded = std::iter::once(ChatMessage::user(format!(
                    "{}\n\n{}",
                    system.content, user.content
                )))
                .chain(rest.iter().cloned())
                .collect::<Vec<_>>();
                folded.as_slice()
            }
            [system] if system.role == "system" => {
                return Err(BunsenError::illegal(
                    "a nanochat conversation's system message must be followed by a user's",
                ));
            }
            _ => messages,
        };
        match messages.last() {
            None => return Err(BunsenError::illegal("an empty nanochat conversation")),
            Some(last) if last.role != "user" => {
                return Err(BunsenError::illegal(
                    "a nanochat conversation to reply to ends with a user's turn",
                ));
            }
            Some(_) => {}
        }
        let mut ids = vec![self.bos];
        for (i, message) in messages.iter().enumerate() {
            let (role, start, end) = if i % 2 == 0 {
                ("user", self.user_start, self.user_end)
            } else {
                ("assistant", self.assistant_start, self.assistant_end)
            };
            if message.role != role {
                return Err(BunsenError::illegal(format!(
                    "a nanochat conversation's turn {i} is the {}'s, where it alternates to the {role}'s",
                    message.role
                )));
            }
            ids.push(start);
            ids.extend(tokenizer.encode(&message.content)?);
            ids.push(end);
        }
        ids.push(self.assistant_start);
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use bunsen::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    use super::*;
    use crate::nanochat_tokenizer::toy_pickle;

    fn toy() -> (NanoChatTokenizer, NanoChatLayout) {
        let tok = NanoChatTokenizer::from_pickle(&toy_pickle()).unwrap();
        let layout = NanoChatLayout::from_tokenizer(&tok).unwrap();
        (tok, layout)
    }

    #[test]
    fn test_render() {
        let (tok, layout) = toy();
        assert_eq!(layout.stop_ids(), [264, 260]);
        let hello = 259;
        let ids = layout
            .render(
                &tok,
                &[
                    ChatMessage::user("hello"),
                    ChatMessage::assistant("hello"),
                    ChatMessage::user("hello"),
                ],
            )
            .unwrap();
        assert_eq!(
            ids,
            [260, 261, hello, 262, 263, hello, 264, 261, hello, 262, 263]
        );
    }

    /// A system message is folded into the first user turn.
    #[test]
    fn test_system_message_folds_into_the_first_turn() {
        let (tok, layout) = toy();
        let folded = layout
            .render(
                &tok,
                &[ChatMessage::system("hello"), ChatMessage::user("hello")],
            )
            .unwrap();
        let joined = layout
            .render(&tok, &[ChatMessage::user("hello\n\nhello")])
            .unwrap();
        assert_eq!(folded, joined);
    }

    #[test]
    fn test_refuses_what_upstream_cannot_render() {
        let (tok, layout) = toy();
        let illegal = ErrorMatcher::kind(BunsenErrorKind::Illegal);
        illegal.assert_err(&layout.render(&tok, &[]));
        illegal.assert_err(&layout.render(&tok, &[ChatMessage::system("hello")]));
        illegal.assert_err(&layout.render(&tok, &[ChatMessage::assistant("hello")]));
        illegal.assert_err(&layout.render(
            &tok,
            &[ChatMessage::user("hello"), ChatMessage::assistant("hello")],
        ));
        illegal.assert_err(&layout.render(
            &tok,
            &[ChatMessage::user("hello"), ChatMessage::user("hello")],
        ));
    }
}
