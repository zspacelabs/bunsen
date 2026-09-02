//! Bunsen Whisper bundled model assets.
//!
//! **This crate hosts nothing.** `OpenAI`'s `base.pt` is 145 MB and the ONNX
//! export is ~290 MB, all far too large to commit — so unlike the Silero
//! bundle, whose graph ships in the crate, this one *fetches* its assets, pins
//! each to a SHA-256, and caches them under `cache/`. The two `.tiktoken`
//! vocabularies are small enough to commit, and are fetched anyway so that
//! every Whisper asset arrives the same way.
//!
//! That size is also why the weights are a **path** rather than bytes:
//! `include_bytes!` of 145 MB would dominate compile time and binary size, so
//! the checkpoint stays a file and [`base_pt`] names it.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// The fetched `base.pt`: `OpenAI`'s multilingual Whisper *base* checkpoint.
///
/// Resolved at build time — either the digest-pinned download, or whatever
/// `WHISPER_BASE_PT` pointed at.
///
/// Nothing here loads a model; that needs bunsen's Whisper kit, which depends
/// on this crate rather than the other way round. Reach for
/// `bunsen::kits::speech::whisper::Whisper::load_pretrained`, which is this
/// path fed through `PytorchWhisperScanner`.
///
/// # Panics
/// Never at run time. If the asset could not be obtained the build itself
/// fails, so reaching this means the file was present and verified *when the
/// crate was compiled*. It can still have been deleted since; callers that
/// care should check [`Path::is_file`](std::path::Path::is_file).
#[cfg(feature = "checkpoint")]
pub fn base_pt() -> &'static std::path::Path {
    std::path::Path::new(env!("WHISPER_BASE_PT_PATH"))
}

/// The fetched `multilingual.tiktoken`: the base vocabulary of every
/// multilingual Whisper checkpoint, `base_pt()` included.
///
/// A `tiktoken` rank file — one `<base64 bytes> <rank>` per line, 50257 of
/// them. bunsen's `kits::speech::whisper::TiktokenRanks` parses it; its
/// last line, `= 50256`, is a genuinely empty token that a strict base64
/// reader rejects, which is why that parser exists.
///
/// Resolved at build time — the digest-pinned download, or whatever
/// `WHISPER_MULTILINGUAL_TIKTOKEN` pointed at.
///
/// # Panics
/// Never at run time: a missing asset fails the build, as for `base_pt()`.
#[cfg(feature = "vocab")]
pub fn multilingual_tiktoken() -> &'static std::path::Path {
    std::path::Path::new(env!("WHISPER_MULTILINGUAL_TIKTOKEN_PATH"))
}

/// The fetched `gpt2.tiktoken`: the base vocabulary of the English-only
/// (`*.en`) Whisper checkpoints, 50256 ranks.
///
/// Not the vocabulary of `base_pt()`, which is multilingual. It is here so
/// that an English-only checkpoint supplied through `WHISPER_BASE_PT` has
/// its tokenizer too, and so that both layouts can be pinned against
/// `whisper.tokenizer`.
///
/// Resolved at build time — the digest-pinned download, or whatever
/// `WHISPER_GPT2_TIKTOKEN` pointed at.
///
/// # Panics
/// Never at run time: a missing asset fails the build, as for `base_pt()`.
#[cfg(feature = "vocab")]
pub fn gpt2_tiktoken() -> &'static std::path::Path {
    std::path::Path::new(env!("WHISPER_GPT2_TIKTOKEN_PATH"))
}

/// The ONNX-generated reference models.
///
/// A transliteration of `onnx-community/whisper-base`, which is a conversion of
/// the same checkpoint [`base_pt`] returns. That shared provenance is what
/// makes comparing them meaningful: they agree only if bunsen's loader and its
/// forward pass are both right.
///
/// Weights are loaded from `OUT_DIR` at run time rather than embedded —
/// together they are ~290 MB.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    /// The reference audio encoder.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod encoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/whisper_base_encoder.rs"));

        impl<B: Backend> Model<B> {
            /// Loads the reference encoder from the generated weights.
            pub fn load_pretrained(device: &B::Device) -> Self {
                Self::from_file(
                    std::path::Path::new(env!("WHISPER_ONNX_OUT_DIR"))
                        .join("whisper_base_encoder.bpk"),
                    device,
                )
            }
        }
    }
    pub use encoder::Model as EncoderModel;

    /// The reference text decoder.
    ///
    /// This is the KV-cache-free export: it consumes a whole token sequence at
    /// once, matching `TextDecoder::forward`. Its `forward` returns the logits
    /// followed by 24 present-key/value tensors, which callers usually ignore.
    /// (`decoder_with_past_model.onnx` is the incremental variant.)
    #[allow(warnings, clippy::all)]
    mod decoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/whisper_base_decoder.rs"));

        impl<B: Backend> Model<B> {
            /// Loads the reference decoder from the generated weights.
            pub fn load_pretrained(device: &B::Device) -> Self {
                Self::from_file(
                    std::path::Path::new(env!("WHISPER_ONNX_OUT_DIR"))
                        .join("whisper_base_decoder.bpk"),
                    device,
                )
            }
        }
    }
    pub use decoder::Model as DecoderModel;
}

#[cfg(test)]
mod tests {
    /// The checkpoint the build resolved must still be on disk, and must be
    /// the size of a `base` checkpoint rather than an error page.
    #[cfg(feature = "checkpoint")]
    #[test]
    fn test_base_pt_is_present() {
        let path = super::base_pt();
        assert!(path.is_file(), "{} is missing", path.display());

        let len = std::fs::metadata(path).expect("stat the checkpoint").len();
        assert!(
            len > 100 * 1024 * 1024,
            "{} is {len} bytes, too small to be base.pt",
            path.display(),
        );
    }

    /// Both rank files must be on disk and the size of a vocabulary rather
    /// than an error page — and the multilingual one must end with its empty
    /// token, which is the line a naive fetch-and-strip would lose.
    #[cfg(feature = "vocab")]
    #[test]
    fn test_tiktoken_files_are_present() {
        // Both are ~800 KB; an error page is a few KB.
        for (path, min_len) in [
            (super::multilingual_tiktoken(), 700 * 1024),
            (super::gpt2_tiktoken(), 700 * 1024),
        ] {
            assert!(path.is_file(), "{} is missing", path.display());

            let len = std::fs::metadata(path).expect("stat the rank file").len();
            assert!(
                len > min_len,
                "{} is {len} bytes, too small to be a rank file",
                path.display(),
            );
        }

        let multilingual =
            std::fs::read_to_string(super::multilingual_tiktoken()).expect("read the rank file");
        assert_eq!(multilingual.lines().count(), 50257);
        assert_eq!(multilingual.lines().next_back(), Some("= 50256"));

        let gpt2 = std::fs::read_to_string(super::gpt2_tiktoken()).expect("read the rank file");
        assert_eq!(gpt2.lines().count(), 50256);
    }

    /// Without a feature the crate is deliberately empty; this documents that
    /// rather than leaving a suite that silently has nothing in it.
    #[cfg(not(any(feature = "checkpoint", feature = "vocab", feature = "onnx_gen")))]
    #[test]
    #[ignore = "no assets fetched; rerun with --features checkpoint, vocab and/or onnx_gen"]
    fn test_assets_need_a_feature() {}
}
