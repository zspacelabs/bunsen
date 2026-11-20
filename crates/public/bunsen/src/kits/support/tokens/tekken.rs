//! Mistral's tekken vocabularies.
use alloc::{
    collections::BTreeMap,
    string::String,
    vec::Vec,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The ids and bytes of a tekken vocabulary, Mistral's byte-level BPE
/// (`tekken.json`, as `mistral_common` reads it, through
/// [`TekkenMirror`](crate::kits::support::interop::mistral_common::TekkenMirror)): the `{ id -> bytes }` table
/// `WordchipperDetokenizer::from_spans` (feature `tokenizer`) takes.
///
/// The first `num_special` ids are the special tokens, in rank order (`<s>`
/// is 1, `</s>` 2, Voxtral Realtime's `[STREAMING_PAD]` 32); vocabulary entry
/// of rank `r` is id `num_special + r`, its bytes the file's base64
/// `token_bytes`. The file may list more entries than the model's vocabulary
/// holds; only the ids below `vocab_size` are read. A special token renders
/// as its own spelling; which ids to render is the caller's policy
/// ([`is_special`](Self::is_special)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TekkenVocabulary {
    spans: BTreeMap<usize, Vec<u8>>,
    specials: BTreeMap<String, usize>,
    num_special: usize,
    vocab_size: usize,
}

impl TekkenVocabulary {
    /// A vocabulary of `spans` (each id's bytes, the specials' their
    /// spellings), `specials` (each special's id by its spelling), the
    /// special count and the vocabulary size: what
    /// [`TekkenMirror::tekken_vocabulary`](crate::kits::support::interop::mistral_common::TekkenMirror::tekken_vocabulary)
    /// reads.
    pub(crate) fn from_parts(
        spans: BTreeMap<usize, Vec<u8>>,
        specials: BTreeMap<String, usize>,
        num_special: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            spans,
            specials,
            num_special,
            vocab_size,
        }
    }

    /// The token count, special tokens included.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The model's vocabulary size: one past the largest id read.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The special-token count: the first id that is a vocabulary entry.
    pub fn num_special(&self) -> usize {
        self.num_special
    }

    /// Whether `id` is a special token's.
    pub fn is_special(
        &self,
        id: usize,
    ) -> bool {
        id < self.num_special
    }

    /// The id of a special token, by its spelling: `[STREAMING_PAD]`, say.
    pub fn special_id(
        &self,
        spelling: &str,
    ) -> Option<usize> {
        self.specials.get(spelling).copied()
    }

    /// The bytes of `id`, if the vocabulary has it.
    pub fn bytes(
        &self,
        id: usize,
    ) -> Option<&[u8]> {
        self.spans.get(&id).map(Vec::as_slice)
    }

    /// The `{ id -> bytes }` table, in id order.
    pub fn spans(&self) -> impl Iterator<Item = (usize, Vec<u8>)> + '_ {
        self.spans.iter().map(|(&id, b)| (id, b.clone()))
    }

    /// The text `ids` spell: their bytes joined, then decoded as UTF-8,
    /// replacing what is not.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if an id is outside the vocabulary.
    pub fn text(
        &self,
        ids: &[usize],
    ) -> BunsenResult<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.bytes(id).ok_or_else(|| {
                BunsenError::invalid_resource(format!("token id {id} is outside the vocabulary"))
            })?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
