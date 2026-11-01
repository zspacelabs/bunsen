//! # Whisper cross-check against a pretrained ONNX reference.
//!
//! Validates [`bunsen`]'s Whisper implementation against an independent one,
//! generated from `onnx-community/whisper-base` by `burn_onnx::ModelGen`. Both
//! run inside `burn`, so a disagreement is a bunsen bug rather than a framework
//! difference.
//!
//! ## This crate hosts nothing, and generates nothing
//!
//! The checkpoint and vocabularies come through bunsen's own pinned rows,
//! fetched at run time into the weights cache by the tests that read them
//! (feature **`checkpoints`**). The generated reference comes from
//! `bind-whisper-onnx`'s `onnx_gen`, fetched and generated at build time,
//! with the bundle the conversion is cross-checked with: `base.pt`, the
//! vocabularies and `transformers`' `model.safetensors`, laid out as a
//! pretrained cache (feature **`reference`**, ~725 MB). Both are off by
//! default, so a plain `cargo test` here reaches no network and runs the
//! fixture-integrity checks alone.
//!
//! `validation/tools/validate.py` runs this crate under a profile, which
//! chooses the model (`validation/models.yml`). Every check of it sits under
//! `base` (`base::staged::...`, `base::audio::...`, `base::bundled::...`):
//!
//! ```sh
//! python3 validation/tools/validate.py run --profile small-core --crate whisper
//! ```
//!
//! `gpu-tests` compiles the tests that run a model on a backend, and a
//! backend feature says which. Pass a backend deliberately:
//! `performance_device()` falls through to `Flex` when none reaches `bunsen`,
//! so a run without one does not fail — it quietly measures the CPU and
//! passes. `--release` matters for the same reason: the work is inside
//! `burn`'s kernels, not in this crate.
//!
//! `WHISPER_ONNX_ENCODER`, `WHISPER_ONNX_DECODER` and `WHISPER_BASE_PT` point
//! the build at local files instead; they are read by
//! `bind-whisper-onnx`' build script, not this crate's.
//!
//! ## Why an ONNX reference
//!
//! bunsen's Whisper is a by-inspection transliteration. Its unit tests check it
//! against itself, which cannot catch a shared misreading of the reference —
//! and did not: the encoder was silently wrong in several separate ways until
//! it was stepped against the real implementation. A cross-check pins the whole
//! thing to something with independent provenance.
//!
//! ## Tests
//!
//! * `staged` steps each layer against the reference on synthetic input, so a
//!   disagreement cannot be inherited from upstream.
//! * `audio` runs the composition over real speech and judges it by word error
//!   rate against a ground-truth transcript.
//! * `bundled` runs the kit's pretrained pathway, vocabularies and hook on the
//!   bundle's files, and holds the `transformers` loader on
//!   `openai/whisper-base`'s `model.safetensors` to `base.pt`, under
//!   `reference`.
//!
//! `onnx-community/whisper-base` is a conversion of `OpenAI`'s multilingual
//! `base.pt`, so the two agree only if bunsen's loader and forward pass are
//! both right.

#![cfg_attr(
    not(all(feature = "checkpoints", feature = "gpu-tests")),
    allow(unused)
)]

use burn::tensor::Device;
/// The reference models, generated from the `onnx-community` export.
///
/// These used to be generated here. They moved to `bind-whisper-onnx` so
/// that one crate owns every Whisper asset — the checkpoint bunsen loads and
/// the export it is judged against — leaving this crate as only the
/// comparison.
#[cfg(feature = "reference")]
pub mod reference {
    pub use bind_whisper_onnx::onnx_gen::*;
}

/// Whisper's fixed analysis window: 30 s at 16 kHz, 3000 mel frames.
pub const N_FRAMES: usize = 3000;

/// The weights cache the tests load the checkpoint and its vocabulary
/// through: bunsen's own (`BUNSEN_CACHE_DIR`, or the user's cache
/// directory), which fetches a pinned file the first time it is asked for;
/// offline when `BUNSEN_VALIDATE_OFFLINE` is set. Routing the assets through
/// the library's own cache is also what this crate is validating. The
/// bundle's own cache, the build-time layout, is the `bundled` checks'.
pub fn weights_cache() -> bunsen::data::pretrained::PretrainedCache {
    use bunsen::data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    };
    let offline = std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some();
    PretrainedCache::new(PretrainedCacheOptions::default().with_offline(offline))
        .expect("open the pretrained cache")
}

/// bunsen's Whisper `openai/base`, at the precision it ships in (fp16), with
/// its token layout and vocabulary, shared.
pub fn load_base(
    device: &Device
) -> std::sync::Arc<bunsen::kits::speech::whisper::driver::WhisperBundle> {
    bunsen::kits::speech::whisper::pretrained::default_whisper_factory()
        .expect("the whisper factory")
        .load_bundle("openai/base", &weights_cache(), device)
        .expect("load openai/base")
}

/// The mel channel count for `base`.
pub const N_MELS: usize = 80;

/// The encoder's model width for `base`.
pub const D_MODEL: usize = 512;

/// The multilingual `base` vocabulary. `base.en` is one smaller.
pub const N_VOCAB: usize = 51865;

/// A short, deterministic token sequence, well inside the vocabulary.
///
/// The comparison is numerical, so these need only be valid ids — they are not
/// meant to decode to anything.
pub const TOKENS: [i64; 4] = [50258, 50259, 50359, 1770];

/// A deterministic `[1, N_MELS, N_FRAMES]` stand-in for real
/// log-mels.
///
/// Cheap and seed-free, so both implementations get bit-identical input
/// without needing an audio fixture.
pub fn synthetic_mels(device: &Device) -> burn::prelude::Tensor<3> {
    use burn::prelude::*;

    let data: Vec<f64> = (0..N_MELS * N_FRAMES)
        .map(|k| {
            let (m, f) = (k / N_FRAMES, k % N_FRAMES);
            // Bounded, non-separable, and varying along both axes.
            ((m * 7 + f * 13) % 211) as f64 / 105.0 - 1.0
        })
        .collect();

    Tensor::from_data(TensorData::new(data, [1, N_MELS, N_FRAMES]), device)
}

/// A deterministic `[1, N_FRAMES / 2, D_MODEL]` stand-in for encoder output.
///
/// Using a synthetic `xa` rather than either encoder's real output isolates
/// the decoder: a disagreement cannot be inherited from upstream.
pub fn synthetic_encoder_output(device: &Device) -> burn::prelude::Tensor<3> {
    use burn::prelude::*;

    let (seq, width) = (N_FRAMES / 2, D_MODEL);
    let data: Vec<f64> = (0..seq * width)
        .map(|k| {
            let (t, c) = (k / width, k % width);
            ((t * 11 + c * 5) % 197) as f64 / 98.0 - 1.0
        })
        .collect();

    Tensor::from_data(TensorData::new(data, [1, seq, width]), device)
}

/// `openai/base`'s checks: staged, over real speech, and of the bundle.
#[cfg(test)]
mod base;

/// A fixture, and the accuracy it is held to.
pub struct Fixture {
    /// The basename of its reference, `testdata/{name}.reference.json`.
    pub name: &'static str,

    /// The clip, and its ground-truth transcript.
    pub clip: bunsen_test_audio::AudioFixture,

    /// Expected number of 30 s windows, as a cheap shape check.
    pub windows: usize,

    /// **The accuracy knob.** Ceiling on word error rate against the
    /// ground-truth transcript.
    ///
    /// Raise it to accept a weaker model or a harder clip; lower it to hold a
    /// gain. A failure here means the pipeline got worse at transcribing,
    /// which is the thing worth knowing.
    pub max_wer: f64,

    /// Ceiling on word error rate against the committed `openai-whisper`
    /// decode.
    ///
    /// Separate from [`Self::max_wer`] because it measures a different thing:
    /// not "is this accurate" but "does this agree with the implementation
    /// bunsen was transliterated from". A backend whose arithmetic flips an
    /// argmax moves this without moving accuracy much, so it has its own knob.
    pub max_reference_wer: f64,
}
