//! The ids-to-text seam: `Detokenizer` and `Tokenizer`.

use core::fmt::Debug;

use crate::errors::BunsenResult;

/// Turns token ids into text.
///
/// The seam a kit holds as `Option<Arc<dyn Detokenizer>>`: without one, its
/// output is ids, which is still a complete result. Ids are `i64`, as the
/// model's token tensors are; the implementation narrows them.
///
/// `Debug` is a supertrait so that a module holding one can still derive
/// `Debug`; `Send + Sync` so that one can be shared across streams.
pub trait Detokenizer: Send + Sync + Debug {
    /// Renders `ids` as text, a special token as its own spelling.
    ///
    /// Bytes are joined **before** the UTF-8 decode, so a character whose
    /// bytes span two ids survives; decoding each id alone would lose both
    /// halves. Which ids to render — whether to drop the prompt, the
    /// timestamps, the stop token — is the caller's policy, applied before
    /// the call; the detokenizer renders whatever it is handed.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if an id is outside the vocabulary. Nothing is skipped silently. The
    /// ids are a model's output, so an id the vocabulary does not cover means
    /// the model and its vocabulary disagree; a caller that built the ids
    /// itself can re-mark the error
    /// [`as_illegal`](crate::errors::ResultContext::as_illegal).
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String>;
}

/// A checkpoint's tokenizer, text to ids and back: what a kit that builds
/// its own prompts, or holds a conversation, reads and writes text through.
///
/// Ids are `i64`, as the model's token tensors are. It is a [`Detokenizer`],
/// and shares its supertraits for the same reasons.
pub trait Tokenizer: Detokenizer {
    /// Encodes `text`, adding no special tokens of its own, an added token's
    /// spelling anywhere in it as that token: `transformers`'
    /// `tokenizer(text, add_special_tokens=False)`.
    ///
    /// # Errors
    /// As the encoder.
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// Encodes `text` as [`encode`](Self::encode) does, but with a special
    /// added token's spelling as text: `transformers`'
    /// `encode_special_tokens`, `tiktoken`'s `encode_ordinary`. An added token
    /// that is not special still encodes as itself.
    ///
    /// # Errors
    /// As the encoder.
    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// The id of an added token, by its content: `[MASK]`, `<end_of_turn>`.
    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64>;

    /// Whether `id` is a token.
    fn contains(
        &self,
        id: i64,
    ) -> bool;

    /// The token count, added tokens included.
    fn len(&self) -> usize;

    /// Whether there are no tokens.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
