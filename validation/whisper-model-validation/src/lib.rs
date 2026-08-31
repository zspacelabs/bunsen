//! # Whisper cross-check against a pretrained ONNX reference.
//!
//! Validates [`bunsen`]'s Whisper implementation against an independent one,
//! generated from `onnx-community/whisper-base` by `burn_onnx::ModelGen`. Both
//! run inside `burn`, so a disagreement is a bunsen bug rather than a framework
//! difference.
//!
//! ## This crate hosts nothing, and generates nothing
//!
//! Both sides of the comparison come from `bunsen-bundled-whisper`: the
//! checkpoint through `bunsen/whisper-weights`, and the generated reference
//! through its `onnx_gen` feature. That crate fetches ~435 MB of assets, pins
//! each to a SHA-256, and caches them — all under the **`download`** feature,
//! which is off by default so an ordinary `cargo build --workspace` never
//! reaches the network.
//!
//! ```sh
//! cargo test --release -p whisper-model-validation --features download,wgpu
//! ```
//!
//! A backend feature is **required**: cross-checking a live model on a CPU
//! backend is not worth the wall clock, and `PerformanceBackend` falls back to
//! CPU silently when none is selected, so this crate refuses to build instead.
//! `--release` matters for the same reason — the work is inside `burn`'s
//! kernels, not in this crate.
//!
//! `WHISPER_ONNX_ENCODER`, `WHISPER_ONNX_DECODER` and `WHISPER_BASE_PT` point
//! the build at local files instead; they are read by
//! `bunsen-bundled-whisper`' build script, not this crate's.
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
//!
//! `onnx-community/whisper-base` is a conversion of `OpenAI`'s multilingual
//! `base.pt`, so the two agree only if bunsen's loader and forward pass are
//! both right.

#![cfg_attr(not(feature = "download"), allow(unused))]

// `bunsen::support::testing::PerformanceBackend` resolves to `Flex` (CPU) when
// no accelerator feature is set. That silent fallback is exactly what makes a
// cross-check useless — it would still pass, just far too slowly to ever be
// run. Refuse rather than mislead.
#[cfg(all(
    feature = "download",
    not(any(feature = "wgpu", feature = "cuda", feature = "metal")),
))]
compile_error!(
    "whisper-model-validation needs a backend: enable one of `wgpu`, `cuda`, or \
     `metal` alongside `download`. Cross-checking a live model on the CPU \
     backend is not worth the wall clock."
);

/// The reference models, generated from the `onnx-community` export.
///
/// These used to be generated here. They moved to `bunsen-bundled-whisper` so
/// that one crate owns every Whisper asset — the checkpoint bunsen loads and
/// the export it is judged against — leaving this crate as only the
/// comparison.
#[cfg(feature = "download")]
pub mod reference {
    pub use bunsen_bundled_whisper::onnx_gen::*;
}

#[cfg(feature = "download")]
pub use reference::{
    decoder,
    encoder,
};

/// Whisper's fixed analysis window: 30 s at 16 kHz, 3000 mel frames.
pub const N_FRAMES: usize = 3000;

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

/// A deterministic `[1, N_MELS, N_FRAMES]` stand-in for real log-mels.
///
/// Cheap and seed-free, so both implementations get bit-identical input
/// without needing an audio fixture.
pub fn synthetic_mels<B: burn::prelude::Backend>(
    device: &B::Device
) -> burn::prelude::Tensor<B, 3> {
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
pub fn synthetic_encoder_output<B: burn::prelude::Backend>(
    device: &B::Device
) -> burn::prelude::Tensor<B, 3> {
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

/// Staged cross-checks on synthetic input.
#[cfg(all(test, feature = "download"))]
mod staged;

/// End-to-end validation over the committed speech fixtures.
#[cfg(test)]
mod audio;
