//! Bunsen Sortformer bundled model assets.
//!
//! **This crate hosts nothing.** NVIDIA's `Nemotron-3-Diarization` is 397 MB,
//! the streaming Sortformer v2 and v2.1 `.nemo`s 471 MB each, and the ONNX
//! exports as much again, far too large to commit, so this crate *fetches*
//! them, pins each file to a SHA-256, and keeps them in `OUT_DIR`.
//!
//! The pretrained assets are laid out in `OUT_DIR` as a **pretrained
//! cache**, `pretrained/sortformer/<namespace>/<sha256>/<file>`, which is how
//! bunsen's `data::pretrained::PretrainedCache` roots a resource. A cache
//! pointed at `cache_dir()`, offline, hits the bundled models without knowing
//! they were bundled.
//!
//! ## What it exposes
//!
//! Each item exists only under the feature that fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `cache_dir()` | any `checkpoint_*` | `OUT_DIR`, laid out as a pretrained cache |
//! | `nemotron_safetensors()`, `nemotron_config()` | `checkpoint_nemotron` | `Nemotron-3-Diarization`'s `model.safetensors` and `config.json` |
//! | `v2_nemo()` | `checkpoint_v2` | `diar_streaming_sortformer_4spk-v2.nemo` |
//! | `v2_1_nemo()` | `checkpoint_v2_1` | `diar_streaming_sortformer_4spk-v2.1.nemo` |
//! | `onnx_gen::NemotronDiarizationModel` | `onnx_gen` | the reference model generated from `altunenes/parakeet-rs`'s Nemotron export |
//! | `onnx_gen::SortformerV21Model` | `onnx_gen_v2_1` | the reference model generated from `Scrybl`'s v2.1 export |
//!
//! ## Using local copies: the `SORTFORMER_*` variables
//!
//! A build without the network, or one that should not download hundreds of
//! megabytes again after a `cargo clean`, can point the build script at local
//! files. Each variable replaces one asset, and is read only when the feature
//! that fetches it is on:
//!
//! | Variable | Replaces |
//! |---|---|
//! | `SORTFORMER_NEMOTRON_SAFETENSORS` | `model.safetensors`, behind `nemotron_safetensors()` |
//! | `SORTFORMER_NEMOTRON_CONFIG` | `config.json`, behind `nemotron_config()` |
//! | `SORTFORMER_ONNX_NEMOTRON` | `nemotron3_diar_v3.onnx`, behind `onnx_gen::NemotronDiarizationModel` |
//! | `SORTFORMER_V2_NEMO` | the v2 `.nemo`, behind `v2_nemo()` |
//! | `SORTFORMER_V2_1_NEMO` | the v2.1 `.nemo`, behind `v2_1_nemo()` |
//! | `SORTFORMER_ONNX_V2_1` | `diar_streaming_sortformer_4spk-v2.1.onnx`, behind `onnx_gen::SortformerV21Model` |
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
#[cfg(any(
    feature = "checkpoint_nemotron",
    feature = "checkpoint_v2",
    feature = "checkpoint_v2_1"
))]
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("SORTFORMER_CACHE_DIR"))
}

/// The fetched `model.safetensors` of NVIDIA's `Nemotron-3-Diarization`.
///
/// Resolved at build time: the digest-pinned download, or whatever
/// `SORTFORMER_NEMOTRON_SAFETENSORS` pointed at. A missing asset fails the
/// build, so this never panics; the file can still have been deleted since.
#[cfg(feature = "checkpoint_nemotron")]
pub fn nemotron_safetensors() -> &'static std::path::Path {
    std::path::Path::new(env!("SORTFORMER_NEMOTRON_SAFETENSORS_PATH"))
}

/// The fetched `config.json` of NVIDIA's `Nemotron-3-Diarization`.
///
/// Resolved at build time, as [`nemotron_safetensors`]
/// (`SORTFORMER_NEMOTRON_CONFIG`).
#[cfg(feature = "checkpoint_nemotron")]
pub fn nemotron_config() -> &'static std::path::Path {
    std::path::Path::new(env!("SORTFORMER_NEMOTRON_CONFIG_PATH"))
}

/// The fetched `.nemo` of NVIDIA's streaming Sortformer v2
/// (`diar_streaming_sortformer_4spk-v2`).
///
/// Resolved at build time, as [`nemotron_safetensors`] is
/// (`SORTFORMER_V2_NEMO`).
#[cfg(feature = "checkpoint_v2")]
pub fn v2_nemo() -> &'static std::path::Path {
    std::path::Path::new(env!("SORTFORMER_V2_NEMO_PATH"))
}

/// The fetched `.nemo` of NVIDIA's streaming Sortformer v2.1
/// (`diar_streaming_sortformer_4spk-v2.1`).
///
/// Resolved at build time, as [`nemotron_safetensors`] is
/// (`SORTFORMER_V2_1_NEMO`).
#[cfg(feature = "checkpoint_v2_1")]
pub fn v2_1_nemo() -> &'static std::path::Path {
    std::path::Path::new(env!("SORTFORMER_V2_1_NEMO_PATH"))
}

/// The ONNX-generated reference models.
///
/// `NemotronDiarizationModel` (feature `onnx_gen`): a transliteration of
/// `altunenes/parakeet-rs`'s export of NVIDIA's `Nemotron-3-Diarization`: one
/// streaming step, `NeMo`'s `forward_for_export` with the 10 ms predictions
/// added. The speaker cache and FIFO are the caller's to keep, as in `NeMo`.
///
/// | Inputs | Outputs |
/// |---|---|
/// | `chunk [B, T, 128]`, `chunk_lengths [1]`, `spkcache [B, C, 512]`, `spkcache_lengths [1]`, `fifo [B, F, 512]`, `fifo_lengths [1]` | `preds_diar [B, C+F+T/8, 8]` (80 ms), `preds_hires [B, 8*(C+F+T/8), 8]` (10 ms), `chunk_pre_encode_embs [B, T/8, 512]`, `chunk_pre_encode_lengths [1]` |
///
/// The lengths are `i64`: build them with an explicit `DType::I64`, as the
/// graph's constants are. `preds_diar` is the 8-frame mean of `preds_hires`.
/// Output frames past the valid lengths are zero.
///
/// The cache and FIFO must each hold at least one frame: the generated model
/// cannot run an empty one, as a stream's first step would.
///
/// `SortformerV21Model` (feature `onnx_gen_v2_1`): a transliteration of
/// `Scrybl`'s export of NVIDIA's streaming Sortformer v2.1, `NeMo`'s own
/// `forward_for_export` step at opset 17.
///
/// | Inputs | Outputs |
/// |---|---|
/// | `chunk [B, T, 128]`, `chunk_lengths [1]`, `spkcache [B, C, 512]`, `spkcache_lengths [1]`, `fifo [B, F, 512]`, `fifo_lengths [1]` | `spkcache_fifo_chunk_preds [B, C+F+T/8, 4]` (80 ms), `chunk_pre_encode_embs [B, T/8, 512]`, `chunk_pre_encode_lengths [1]` |
///
/// The lengths are `i64`, as for Nemotron's. Batch norm is folded into the
/// convolutions, so the graph's weights compare to the checkpoint's only
/// after folding; its outputs compare directly.
///
/// Weights are loaded from `OUT_DIR` at run time rather than embedded.
#[cfg(any(feature = "onnx_gen", feature = "onnx_gen_v2_1"))]
pub mod onnx_gen {
    use burn::prelude::*;

    /// The directory the build generated the weights into.
    fn out_dir() -> &'static std::path::Path {
        std::path::Path::new(env!("SORTFORMER_ONNX_OUT_DIR"))
    }

    /// The reference Nemotron diarization step.
    // Machine-generated: not held to this crate's lint bar.
    #[cfg(feature = "onnx_gen")]
    #[allow(warnings, clippy::all)]
    mod nemotron {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/nemotron_diarization.rs"));

        impl Model {
            /// Loads the reference step from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("nemotron_diarization.bpk"), device)
            }
        }
    }
    #[cfg(feature = "onnx_gen")]
    pub use nemotron::Model as NemotronDiarizationModel;

    /// The reference Sortformer v2.1 step.
    // Machine-generated: not held to this crate's lint bar.
    #[cfg(feature = "onnx_gen_v2_1")]
    #[allow(warnings, clippy::all)]
    mod v2_1 {
        use super::*;

        include!(concat!(env!("OUT_DIR"), "/sortformer_v2_1.rs"));

        impl Model {
            /// Loads the reference step from the generated weights.
            pub fn load_pretrained(device: &Device) -> Self {
                Self::from_file(out_dir().join("sortformer_v2_1.bpk"), device)
            }
        }
    }
    #[cfg(feature = "onnx_gen_v2_1")]
    pub use v2_1::Model as SortformerV21Model;
}

#[cfg(test)]
mod tests {
    /// Without a feature the crate is deliberately empty; this documents that
    /// rather than leaving a suite that silently has nothing in it.
    #[cfg(not(any(
        feature = "checkpoint_nemotron",
        feature = "checkpoint_v2",
        feature = "checkpoint_v2_1",
        feature = "onnx_gen",
        feature = "onnx_gen_v2_1"
    )))]
    #[test]
    #[ignore = "no assets fetched; rerun with a checkpoint_* or onnx_gen* feature"]
    fn test_assets_need_a_feature() {}

    /// Every pretrained asset sits in the cache layout under the root the
    /// crate exposes: `pretrained/sortformer/<namespace>/<sha256>/<file>`.
    #[cfg(feature = "checkpoint_nemotron")]
    #[test]
    fn test_assets_are_laid_out_as_a_pretrained_cache() {
        let root = super::cache_dir().join("pretrained").join("sortformer");
        for path in [super::nemotron_safetensors(), super::nemotron_config()] {
            assert!(path.is_file(), "{} is missing", path.display());
            let rel = path.strip_prefix(root.join("nvidia")).unwrap();
            assert_eq!(rel.components().count(), 2, "{}", path.display());
        }
    }

    /// The `.nemo`s sit in the same layout.
    #[cfg(all(feature = "checkpoint_v2", feature = "checkpoint_v2_1"))]
    #[test]
    fn test_nemos_are_laid_out_as_a_pretrained_cache() {
        let root = super::cache_dir().join("pretrained").join("sortformer");
        for path in [super::v2_nemo(), super::v2_1_nemo()] {
            assert!(path.is_file(), "{} is missing", path.display());
            let rel = path.strip_prefix(root.join("nvidia")).unwrap();
            assert_eq!(rel.components().count(), 2, "{}", path.display());
        }
    }
}
