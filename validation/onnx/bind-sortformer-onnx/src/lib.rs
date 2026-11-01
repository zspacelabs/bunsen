//! Bunsen Sortformer reference assets.
//!
//! **This crate hosts nothing.** The ONNX exports of NVIDIA's
//! `Nemotron-3-Diarization` and streaming Sortformer v2.1 are 401 and 492 MB,
//! far too large to commit, so this crate *fetches* them, pins each file to a
//! SHA-256, keeps them in `OUT_DIR`, and generates the reference models from
//! them. The checkpoints are not here; `sortformer-model-validation`'s tests
//! load them through bunsen's pinned rows, at run time, into the weights
//! cache.
//!
//! ## What it exposes
//!
//! Each item exists only under the feature that fetches its asset:
//!
//! | Item | Feature | What it is |
//! |---|---|---|
//! | `onnx_gen::NemotronDiarizationModel` | `onnx_gen` | the reference model generated from `altunenes/parakeet-rs`'s Nemotron export |
//! | `onnx_gen::SortformerV21Model` | `onnx_gen_v2_1` | the reference model generated from `Scrybl`'s v2.1 export |
//!
//! ## Using local copies: the `SORTFORMER_ONNX_*` variables
//!
//! A build without the network, or one that should not download hundreds of
//! megabytes again after a `cargo clean`, can point the build script at local
//! files. Each variable replaces one asset, and is read only when the feature
//! that fetches it is on:
//!
//! | Variable | Replaces |
//! |---|---|
//! | `SORTFORMER_ONNX_NEMOTRON` | `nemotron3_diar_v3.onnx`, behind `onnx_gen::NemotronDiarizationModel` |
//! | `SORTFORMER_ONNX_V2_1` | `diar_streaming_sortformer_4spk-v2.1.onnx`, behind `onnx_gen::SortformerV21Model` |
//!
//! The variable must name a file, or the build fails. Nothing is downloaded
//! for that asset. A file that is the pinned asset is put where the download
//! would have gone (a symlink on Unix, a copy elsewhere); any other file is
//! used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

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
    #[cfg(not(any(feature = "onnx_gen", feature = "onnx_gen_v2_1")))]
    #[test]
    #[ignore = "no assets fetched; rerun with an onnx_gen* feature"]
    fn test_assets_need_a_feature() {}
}
