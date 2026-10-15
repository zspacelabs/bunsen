//! Bunsen Parakeet TDT bundled model assets.
//!
//! **This crate hosts nothing.** `FermionResearch`'s Phonon-2 is 164 MB,
//! NVIDIA's `parakeet-tdt-0.6b-v3` 2.5 GB, and its ONNX export as much again,
//! all far too large to commit, so this crate *fetches* them, pins each file
//! to a SHA-256, and keeps them in `OUT_DIR`.
//!
//! The pretrained assets are laid out in `OUT_DIR` as a **pretrained
//! cache**, `pretrained/parakeet/<namespace>/<sha256>/<file>`, which is how
//! bunsen's `data::pretrained::PretrainedCache` roots a resource. A cache
//! pointed at `cache_dir()`, offline, hits `fermion/phonon-2` and
//! `nvidia/parakeet-tdt-0.6b-v3` without knowing they were bundled.
//!
//! ## What it exposes
//!
//! Each item exists only under the feature that fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `cache_dir()` | `checkpoint` or `base_checkpoint` | `OUT_DIR`, laid out as a pretrained cache |
//! | `phonon_2_archive()` | `checkpoint` | Phonon-2's `phonon-2.bps.tar.zst` |
//! | `parakeet_safetensors()`, `parakeet_tokenizer()` | `base_checkpoint` | `parakeet-tdt-0.6b-v3`'s `model.safetensors` and `tokenizer.json` |
//! | `onnx_gen::{EncoderModel, DecoderJointModel}` | `onnx_gen` | reference models generated from `istupakov/parakeet-tdt-0.6b-v3-onnx` |
//!
//! ## Using local copies: the `PARAKEET_*` variables
//!
//! A build without the network, or one that should not download gigabytes
//! again after a `cargo clean`, can point the build script at local files.
//! Each variable replaces one asset, and is read only when the feature that
//! fetches it is on:
//!
//! | Variable | Replaces |
//! |---|---|
//! | `PHONON_2_ARCHIVE` | `phonon-2.bps.tar.zst`, behind `phonon_2_archive()` |
//! | `PARAKEET_SAFETENSORS` | `model.safetensors`, behind `parakeet_safetensors()` |
//! | `PARAKEET_TOKENIZER` | `tokenizer.json`, behind `parakeet_tokenizer()` |
//! | `PARAKEET_ONNX_ENCODER` | `encoder-model.onnx`, behind `onnx_gen::EncoderModel` |
//! | `PARAKEET_ONNX_ENCODER_DATA` | `encoder-model.onnx.data`, the encoder's external weights |
//! | `PARAKEET_ONNX_DECODER_JOINT` | `decoder_joint-model.onnx`, behind `onnx_gen::DecoderJointModel` |
//!
//! The variable must name a file, or the build fails. Nothing is downloaded
//! for that asset. A file that is the pinned asset is put where the download
//! would have gone (a symlink on Unix, a copy elsewhere); any other file is
//! used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::PretrainedCache` is pointed at to read the
/// bundled assets. A cache that fetches other models into it will change it,
/// so a test wants it offline.
#[cfg(any(feature = "checkpoint", feature = "base_checkpoint"))]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("PARAKEET_CACHE_DIR"))
}

/// The fetched `phonon-2.bps.tar.zst`: `FermionResearch`'s Phonon-2.
///
/// Resolved at build time: the digest-pinned download, or whatever
/// `PHONON_2_ARCHIVE` pointed at. A missing asset fails the build, so this
/// never panics; the file can still have been deleted since.
#[cfg(feature = "checkpoint")]
pub fn phonon_2_archive() -> &'static std::path::Path {
    std::path::Path::new(env!("PHONON_2_ARCHIVE_PATH"))
}

/// The fetched `model.safetensors` of NVIDIA's `parakeet-tdt-0.6b-v3`.
///
/// Resolved at build time, as [`phonon_2_archive`] (`PARAKEET_SAFETENSORS`).
#[cfg(feature = "base_checkpoint")]
pub fn parakeet_safetensors() -> &'static std::path::Path {
    std::path::Path::new(env!("PARAKEET_SAFETENSORS_PATH"))
}

/// The fetched `tokenizer.json` of NVIDIA's `parakeet-tdt-0.6b-v3`.
///
/// Resolved at build time, as [`phonon_2_archive`] (`PARAKEET_TOKENIZER`).
#[cfg(feature = "base_checkpoint")]
pub fn parakeet_tokenizer() -> &'static std::path::Path {
    std::path::Path::new(env!("PARAKEET_TOKENIZER_PATH"))
}

/// The ONNX-generated reference models.
///
/// A transliteration of `istupakov/parakeet-tdt-0.6b-v3-onnx`, `NeMo`'s own
/// export of NVIDIA's `parakeet-tdt-0.6b-v3`. The two graphs are the two
/// model stages of a transcription; the export's feature extractor is not
/// generated, as `burn-onnx` cannot import it.
///
/// | Model | Inputs | Outputs |
/// |---|---|---|
/// | [`EncoderModel`](onnx_gen::EncoderModel) | `audio_signal [B, 128, T]`, `length [B]` | `outputs [B, 1024, T/8]`, `encoded_lengths [B]` |
/// | [`DecoderJointModel`](onnx_gen::DecoderJointModel) | `encoder_outputs [B, 1024, 1]`, `targets [B, 1]`, `target_length [B]`, two LSTM states `[2, B, 640]` | `outputs [B, 1, 1, 8198]`, `prednet_lengths [B]`, two LSTM states |
///
/// The joint's 8198 outputs are 8192 tokens, the blank (8192), and the five
/// TDT duration logits for durations 0 through 4.
///
/// Weights are loaded from `OUT_DIR` at run time rather than embedded.
#[cfg(feature = "onnx_gen")]
pub mod onnx_gen {
    use burn::prelude::*;

    /// The directory the build generated the weights into.
    fn out_dir() -> &'static std::path::Path {
        std::path::Path::new(env!("PARAKEET_ONNX_OUT_DIR"))
    }

    /// The reference `FastConformer` encoder.
    // Machine-generated: not held to this crate's lint bar.
    #[allow(warnings, clippy::all)]
    mod encoder {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/parakeet_encoder.rs"));

        impl Model {
            /// Loads the reference encoder from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("parakeet_encoder.bpk"), device)
            }
        }
    }
    pub use encoder::Model as EncoderModel;

    /// The reference TDT prediction network and joint, one step at a time.
    #[allow(warnings, clippy::all)]
    mod decoder_joint {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/parakeet_decoder_joint.rs"));

        impl Model {
            /// Loads the reference prediction network and joint from the
            /// generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("parakeet_decoder_joint.bpk"), device)
            }
        }
    }
    pub use decoder_joint::Model as DecoderJointModel;
}

#[cfg(test)]
mod tests {
    /// Without a feature the crate is deliberately empty; this documents that
    /// rather than leaving a suite that silently has nothing in it.
    #[cfg(not(any(
        feature = "checkpoint",
        feature = "base_checkpoint",
        feature = "onnx_gen"
    )))]
    #[test]
    #[ignore = "no assets fetched; rerun with --features checkpoint, base_checkpoint and/or onnx_gen"]
    fn test_assets_need_a_feature() {}

    /// Every pretrained asset sits in the cache layout under the root the
    /// crate exposes: `pretrained/parakeet/<namespace>/<sha256>/<file>`.
    #[cfg(any(feature = "checkpoint", feature = "base_checkpoint"))]
    #[test]
    fn test_assets_are_laid_out_as_a_pretrained_cache() {
        let root = super::cache_dir().join("pretrained").join("parakeet");
        let mut assets: Vec<(&std::path::Path, &str)> = Vec::new();
        #[cfg(feature = "checkpoint")]
        assets.push((super::phonon_2_archive(), "fermion"));
        #[cfg(feature = "base_checkpoint")]
        assets.extend([
            (super::parakeet_safetensors(), "nvidia"),
            (super::parakeet_tokenizer(), "nvidia"),
        ]);
        for (path, namespace) in assets {
            assert!(path.is_file(), "{} is missing", path.display());
            let rel = path.strip_prefix(root.join(namespace)).unwrap();
            assert_eq!(rel.components().count(), 2, "{}", path.display());
        }
    }
}
