//! Bunsen Whisper bundled model assets.
//!
//! **This crate hosts nothing.** `OpenAI`'s "base.pt" is 145 MB and the ONNX
//! export is ~290 MB, all far too large to commit — so unlike the Silero
//! bundle, whose graph ships in the crate, this one *fetches* its assets, pins
//! each to a SHA-256, and keeps them in `OUT_DIR`. The two `.tiktoken`
//! vocabularies are small enough to commit, and are fetched anyway so that
//! every Whisper asset arrives the same way.
//!
//! The pretrained assets are laid out in `OUT_DIR` as a **pretrained cache**,
//! `pretrained/whisper/openai/<sha256>/<file>`, which is how bunsen's
//! `data::pretrained::WeightsCache` roots a resource. A cache pointed at
//! `cache_dir()`, offline, hits `openai/base` and both vocabularies without
//! knowing they were bundled: bundling is a populated cache directory, not a
//! kind of source. bunsen's own tests and `whisper-model-validation` do
//! exactly that.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::WeightsCache` is pointed at to read the
/// bundled assets. Nothing under it changes after the build; a cache that
/// fetches other models into it will, so a test wants it offline.
///
/// # Panics
/// Never at run time; the directory was written when the crate was
/// compiled.
#[cfg(any(feature = "checkpoint", feature = "vocab"))]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("WHISPER_CACHE_DIR"))
}

/// The fetched `base.pt`: `OpenAI`'s multilingual Whisper *base* checkpoint.
///
/// Resolved at build time — either the digest-pinned download, or whatever
/// `WHISPER_BASE_PT` pointed at; laid out under [`cache_dir`] unless the
/// override was a different file.
///
/// Nothing here loads a model; that needs bunsen's Whisper kit, which depends
/// on this crate rather than the other way round. Reach for
/// `bunsen::kits::speech::whisper::pretrained::default_whisper_factory()`
/// and its `load("openai/base", …)`, with a cache rooted at [`cache_dir`].
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
/// `WHISPER_MULTILINGUAL_TIKTOKEN` pointed at; laid out under [`cache_dir`]
/// unless the override was a different file.
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
/// `WHISPER_GPT2_TIKTOKEN` pointed at; laid out under [`cache_dir`] unless
/// the override was a different file.
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
    /// Every pretrained asset sits in the cache layout under the root the
    /// crate exposes: `pretrained/whisper/openai/<sha256>/<file>`, under the
    /// file name the Whisper kit's resource uses. (An override that is not
    /// the pinned file lands elsewhere; this test assumes none is set.)
    #[cfg(any(feature = "checkpoint", feature = "vocab"))]
    #[test]
    fn test_assets_are_laid_out_as_a_pretrained_cache() {
        let root = super::cache_dir()
            .join("pretrained")
            .join("whisper")
            .join("openai");
        assert!(root.is_dir(), "{} is missing", root.display());

        let mut assets: Vec<&'static std::path::Path> = Vec::new();
        #[cfg(feature = "checkpoint")]
        assets.push(super::base_pt());
        #[cfg(feature = "vocab")]
        assets.extend([super::multilingual_tiktoken(), super::gpt2_tiktoken()]);

        for path in assets {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or_else(|_| panic!("{} is not under {}", path.display(), root.display()));
            let mut parts = rel.components();
            let digest = parts.next().unwrap().as_os_str().to_str().unwrap();
            assert_eq!(digest.len(), 64, "{}", path.display());
            assert!(
                digest
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
                "{}",
                path.display()
            );
            let file = parts.next().unwrap().as_os_str().to_str().unwrap();
            assert!(
                ["base.pt", "multilingual.tiktoken", "gpt2.tiktoken"].contains(&file),
                "{}",
                path.display()
            );
            assert!(parts.next().is_none(), "{}", path.display());
        }
    }

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
