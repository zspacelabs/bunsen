//! # Tokens
//!
//! The token side of a model kit: what is shared between kits that emit ids
//! and want text back. Today that is one seam, [`Detokenizer`], and one
//! implementation of it, `WordchipperDetokenizer`, behind the `tokenizer`
//! feature.
//!
//! A model kit owns its own token *layout* — which ids are special, what they
//! mean, how a prompt is built — because that is model-specific and needs no
//! dependency; Whisper's is [`TokenPolicy`]. What it does not own is the
//! tokenizer. Ids-to-text is the same operation for every byte-level
//! vocabulary, and the crate that does it well is `wordchipper`, so that
//! lives here once and a kit hands it a table.
//!
//! Only decoding is here. Encoding — the merge table, the pre-tokenizer, the
//! special-token splitter — is `wordchipper`'s whole job, and a kit that needs
//! it should use that crate directly.
//!
//! [`TokenPolicy`]: crate::kits::speech::whisper::driver::TokenPolicy

use std::fmt::Debug;

use crate::errors::BunsenResult;

#[cfg(feature = "tokenizer")]
mod wordchipper;
#[cfg(feature = "tokenizer")]
pub use wordchipper::WordchipperDetokenizer;

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
    /// If an id is outside the vocabulary. Nothing is skipped silently.
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A detokenizer with no assets: renders each id as its number.
    #[derive(Debug)]
    struct Numbers;

    impl Detokenizer for Numbers {
        fn detokenize(
            &self,
            ids: &[i64],
        ) -> BunsenResult<String> {
            Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
        }
    }

    /// The shape a kit holds: shareable, dynamic, and debuggable.
    #[test]
    fn test_trait_object() {
        let detokenizer: Option<Arc<dyn Detokenizer>> = Some(Arc::new(Numbers));

        assert_eq!(format!("{detokenizer:?}"), "Some(Numbers)");
        assert_eq!(
            detokenizer
                .as_ref()
                .unwrap()
                .detokenize(&[1, 2, 3])
                .unwrap(),
            "1 2 3"
        );
    }
}
