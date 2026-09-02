//! # Token ids to text, for Whisper.
//!
//! The seam is [`Detokenizer`], shared by every kit; what is Whisper's here
//! is only the **table** — the base ranks of a `.tiktoken` file followed by
//! the generated special-token spellings, which is what `tiktoken` builds
//! for `whisper/tokenizer.py`. [`token_spans`] is that table and needs no
//! dependency; [`detokenizer`] hands it to `wordchipper` behind the
//! `tokenizer` feature.
//!
//! What is deliberately *not* here is a specials policy. Whisper's own
//! `Tokenizer.decode` drops the timestamp tokens; its transcriber drops
//! everything from `<|endoftext|>` up; `decode_with_timestamps` keeps them
//! all. Those are three filters over ids, and [`TokenPolicy::text_ids`] is
//! one of them. The detokenizer renders whatever it is handed, a special as
//! its `<|name|>`.
//!
//! [`Detokenizer`]: crate::kits::tokens::Detokenizer
//! [`TokenPolicy::text_ids`]: super::tokens::TokenPolicy::text_ids

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::driver::support::{
        tokens::WhisperSpecialIds,
        vocab::TiktokenRanks,
    },
};

/// The full `{ id -> bytes }` table of a Whisper vocabulary, indexed by id.
///
/// The base ranks first, then every special in [`WhisperSpecialIds`]'s
/// order, spelled as `<|name|>` — `n_vocab` entries in all.
///
/// # Errors
/// [`BunsenError::Invalid`] if the ranks are not the base of `ids`' layout:
/// the vocabulary file and the checkpoint disagree.
pub fn token_spans(
    ranks: &TiktokenRanks,
    ids: &WhisperSpecialIds,
) -> BunsenResult<Vec<Vec<u8>>> {
    if ranks.len() != ids.n_base {
        return Err(BunsenError::Invalid(format!(
            "the vocabulary has {} base ranks but the layout expects {}",
            ranks.len(),
            ids.n_base,
        )));
    }

    let mut spans = Vec::with_capacity(ids.n_vocab());
    spans.extend(ranks.iter().map(<[u8]>::to_vec));
    spans.extend(ids.special_names().map(String::into_bytes));
    Ok(spans)
}

/// A [`Detokenizer`](crate::kits::tokens::Detokenizer) over a Whisper
/// vocabulary.
///
/// [`token_spans`] handed to `wordchipper`'s decode-only path.
///
/// # Errors
/// As [`token_spans`].
#[cfg(feature = "tokenizer")]
pub fn detokenizer(
    ranks: &TiktokenRanks,
    ids: &WhisperSpecialIds,
) -> BunsenResult<crate::kits::tokens::WordchipperDetokenizer<u16>> {
    crate::kits::tokens::WordchipperDetokenizer::from_spans(
        token_spans(ranks, ids)?.into_iter().enumerate(),
    )
}

/// [`detokenizer`] over a `.tiktoken` file.
///
/// # Errors
/// As [`TiktokenRanks::load`] and [`detokenizer`].
#[cfg(feature = "tokenizer")]
pub fn load_detokenizer(
    path: impl AsRef<std::path::Path>,
    ids: &WhisperSpecialIds,
) -> BunsenResult<crate::kits::tokens::WordchipperDetokenizer<u16>> {
    detokenizer(&TiktokenRanks::load(path)?, ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Five base ranks, including an empty one, laid out with two languages.
    fn tiny() -> (TiktokenRanks, WhisperSpecialIds) {
        let ranks = TiktokenRanks::parse("b2s= 0\nIPCfjg== 1\niQ== 2\nIGRvbmU= 3\n= 4\n").unwrap();
        let ids = WhisperSpecialIds::new(5, 2).unwrap();
        (ranks, ids)
    }

    #[test]
    fn test_token_spans() {
        let (ranks, ids) = tiny();
        let spans = token_spans(&ranks, &ids).unwrap();

        assert_eq!(spans.len(), ids.n_vocab());
        assert_eq!(spans[0], b"ok");
        assert_eq!(spans[4], b"", "the empty token is present");
        assert_eq!(spans[ids.eot as usize], b"<|endoftext|>");
        assert_eq!(spans[ids.language_begin as usize + 1], b"<|zh|>");
        assert_eq!(spans[ids.timestamp_begin as usize + 100], b"<|2.00|>");
        assert_eq!(spans[ids.timestamp_end() as usize], b"<|30.00|>");
    }

    #[test]
    fn test_token_spans_rejects_a_mismatched_layout() {
        let (ranks, _) = tiny();
        let wrong = WhisperSpecialIds::new(4, 2).unwrap();

        let err = token_spans(&ranks, &wrong).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err:?}");
    }

    #[cfg(feature = "tokenizer")]
    mod wordchipper {
        use super::*;
        use crate::kits::tokens::Detokenizer;

        /// Specials render as their spellings, straight from the layout.
        #[test]
        fn test_detokenizer_renders_specials() {
            let (ranks, ids) = tiny();
            let detok = detokenizer(&ranks, &ids).unwrap();
            assert_eq!(detok.vocab_size(), ids.n_vocab());

            let window = [
                ids.sot,
                ids.language_begin,
                ids.transcribe,
                ids.timestamp_begin,
                0,
                4,
                ids.timestamp_begin + 100,
                ids.eot,
            ];
            assert_eq!(
                detok.detokenize(&window).unwrap(),
                "<|startoftranscript|><|en|><|transcribe|><|0.00|>ok<|2.00|><|endoftext|>",
            );
            assert_eq!(detok.detokenize(&[0, 1, 2, 3]).unwrap(), "ok 🎉 done");
            assert!(detok.detokenize(&[ids.n_vocab() as i64]).is_err());
        }

        /// Against the real assets, with ids produced by `whisper.tokenizer`.
        #[cfg(feature = "whisper-weights")]
        mod bundled {
            use super::*;
            use crate::kits::speech::whisper::driver::support::tokens::{
                ENGLISH_BASE_RANKS,
                MULTILINGUAL_BASE_RANKS,
                TokenPolicy,
            };

            #[test]
            fn test_multilingual_vocabulary() {
                let ranks =
                    TiktokenRanks::load(bunsen_bundled_whisper::multilingual_tiktoken()).unwrap();
                assert_eq!(ranks.len(), MULTILINGUAL_BASE_RANKS);
                assert_eq!(ranks.get(50256), Some(&b""[..]), "the empty token");
                assert_eq!(
                    ranks.get(50255),
                    Some("场".as_bytes()),
                    "rank 50255 is `5Zy6`"
                );

                // The layout the checkpoint implies is the layout the file has.
                let policy = TokenPolicy::from_vocab_size(51865).unwrap();
                let detok = detokenizer(&ranks, policy.ids()).unwrap();
                assert_eq!(detok.vocab_size(), 51865);

                // Plain text, including non-ASCII.
                let sentence = [
                    15947, 1002, 11, 341, 307, 257, 1500, 295, 264, 1667, 15487, 303, 979, 19866,
                    3466, 220, 27311, 31348, 886, 13,
                ];
                assert_eq!(
                    detok.detokenize(&sentence).unwrap(),
                    "Hello world, this is a test of the naïve decoder — 日本語 too.",
                );

                // A codepoint split across two ids: 19034 is `b" \xf0\x9f\x8e"`
                // and 231 is `b"\x89"`.
                assert_eq!(
                    detok.detokenize(&[453, 19034, 231, 1096]).unwrap(),
                    "ok 🎉 done"
                );

                // A window as the model emits it, with and without its
                // specials.
                let window = [50258, 50259, 50359, 50364, 15947, 1002, 13, 50464, 50257];
                assert_eq!(
                    detok.detokenize(&window).unwrap(),
                    "<|startoftranscript|><|en|><|transcribe|><|0.00|>Hello world.<|2.00|><|endoftext|>",
                );
                assert_eq!(
                    detok.detokenize(&policy.text_ids(&window)).unwrap(),
                    "Hello world."
                );

                // Every id renders; none is a hole.
                let all: Vec<i64> = (0..51865).collect();
                assert!(detok.detokenize(&all).is_ok());
            }

            #[test]
            fn test_english_vocabulary() {
                let ranks = TiktokenRanks::load(bunsen_bundled_whisper::gpt2_tiktoken()).unwrap();
                assert_eq!(ranks.len(), ENGLISH_BASE_RANKS);
                assert_eq!(ranks.get(50255), Some(&b" gazed"[..]));

                let policy = TokenPolicy::from_vocab_size(51864).unwrap();
                let detok = load_detokenizer(bunsen_bundled_whisper::gpt2_tiktoken(), policy.ids())
                    .unwrap();
                assert_eq!(detok.vocab_size(), 51864);

                // The same text, tokenized differently: this vocabulary splits
                // the party popper across three ids.
                assert_eq!(detok.detokenize(&[15496, 995, 13]).unwrap(), "Hello world.");
                assert_eq!(
                    detok.detokenize(&[482, 12520, 236, 231, 1760]).unwrap(),
                    "ok 🎉 done"
                );
                assert_eq!(
                    detok.detokenize(&[50257, 50362, 50363, 50256]).unwrap(),
                    "<|startoftranscript|><|notimestamps|><|0.00|><|endoftext|>",
                );

                // The multilingual layout does not fit this file.
                let wrong = TokenPolicy::from_vocab_size(51865).unwrap();
                assert!(detokenizer(&ranks, wrong.ids()).is_err());
            }
        }
    }
}
