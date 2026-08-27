//! # Whisper cross-check against a pretrained ONNX reference.
//!
//! Validates [`bunsen`]'s Whisper implementation against an independent one,
//! generated from `onnx-community/whisper-base` by `burn_onnx::ModelGen`. Both
//! run inside `burn`, so a disagreement is a bunsen bug rather than a framework
//! difference.
//!
//! ## This crate hosts nothing
//!
//! The ONNX graph is ~82 MB and is not committed. `build.rs` fetches it, pins
//! it to a SHA-256, and caches it under `.cache/`. That only happens under the
//! **`download`** feature, which is off by default so an ordinary
//! `cargo build --workspace` never reaches the network.
//!
//! ```sh
//! cargo test --release -p whisper-onnx-crosscheck --features download,wgpu
//! ```
//!
//! A backend feature is **required**: cross-checking a live model on a CPU
//! backend is not worth the wall clock, and `PerformanceBackend` falls back to
//! CPU silently when none is selected, so this crate refuses to build instead.
//! `--release` matters for the same reason — the work is inside `burn`'s
//! kernels, not in this crate.
//!
//! Point `WHISPER_ONNX_ENCODER` at a local `.onnx` to skip the fetch, and
//! `WHISPER_BASE_PT` at `OpenAI`'s `base.pt` to enable the full comparison
//! against bunsen (see [`tests`](self#tests)).
//!
//! ## Why an ONNX reference
//!
//! bunsen's Whisper is a by-inspection transliteration. Its unit tests check it
//! against itself, which cannot catch a shared misreading of the reference —
//! and did not: the encoder was silently wrong in three separate ways until it
//! was stepped against the real implementation. A cross-check pins the whole
//! thing to something with independent provenance.
//!
//! ## Tests
//!
//! * The generated model runs and is deterministic — no extra assets.
//! * With `WHISPER_BASE_PT` set, bunsen's encoder is compared against it on
//!   identical weights. `onnx-community/whisper-base` is a conversion of
//!   `OpenAI`'s multilingual `base.pt`, so the two agree only if bunsen's
//!   loader and forward pass are both right.

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
    "whisper-onnx-crosscheck needs a backend: enable one of `wgpu`, `cuda`, or \
     `metal` alongside `download`. Cross-checking a live model on the CPU \
     backend is not worth the wall clock."
);

/// The generated reference encoder.
///
/// Only present under the `download` feature; without it there is no graph to
/// generate from.
#[cfg(feature = "download")]
// Machine-generated: not held to this crate's lint bar.
#[allow(warnings, clippy::all)]
pub mod encoder {
    use burn::prelude::*;

    include!(concat!(env!("OUT_DIR"), "/whisper_base_encoder.rs"));

    impl<B: Backend> Model<B> {
        /// Loads the reference encoder from the generated weights.
        pub fn load_reference(device: &B::Device) -> Self {
            Self::from_file(
                std::path::Path::new(env!("WHISPER_ONNX_OUT_DIR")).join("whisper_base_encoder.bpk"),
                device,
            )
        }
    }
}

/// The generated reference decoder.
///
/// This is the KV-cache-free export: it consumes a whole token sequence at
/// once, matching `TextDecoder::forward`. Its `forward` returns the logits
/// followed by 24 present-key/value tensors, which this crate ignores.
#[cfg(feature = "download")]
#[allow(warnings, clippy::all)]
pub mod decoder {
    use burn::prelude::*;

    include!(concat!(env!("OUT_DIR"), "/whisper_base_decoder.rs"));

    impl<B: Backend> Model<B> {
        /// Loads the reference decoder from the generated weights.
        pub fn load_reference(device: &B::Device) -> Self {
            Self::from_file(
                std::path::Path::new(env!("WHISPER_ONNX_OUT_DIR")).join("whisper_base_decoder.bpk"),
                device,
            )
        }
    }
}

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

#[cfg(all(test, feature = "download"))]
mod tests {
    use bunsen::{
        burner::{
            module::DTypeMapper,
            tensor::TensorElemOpExt,
        },
        support::testing::PerformanceBackend,
    };
    use burn::{
        module::Module as _,
        prelude::*,
        tensor::{
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;

    type B = PerformanceBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// The reference model loads, runs, and is deterministic.
    ///
    /// Needs no assets beyond the fetched graph, so it is the smoke test that
    /// tells you the fetch-and-generate path itself is healthy.
    #[test]
    fn test_reference_encoder_runs() {
        let device = Default::default();
        let model = encoder::Model::<B>::load_reference(&device);

        let out = model.forward(synthetic_mels::<B>(&device));
        assert_eq!(out.dims(), [1, N_FRAMES / 2, D_MODEL]);

        let values: Vec<f32> = out.clone().into_data().convert::<f32>().to_vec().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "the reference encoder produced a non-finite value",
        );

        // Same input, same output — nothing in the graph is order-dependent.
        let again = model.forward(synthetic_mels::<B>(&device));
        out.to_data_as::<F>()
            .assert_approx_eq::<F>(&again.to_data_as::<F>(), Tolerance::default());
    }

    /// **The cross-check.** bunsen's encoder must match the reference on
    /// identical weights.
    ///
    /// The checkpoint is OpenAI's multilingual `base.pt`, which is what
    /// `onnx-community/whisper-base` was converted from; `build.rs` fetches it
    /// alongside the graph, so this never skips. `WHISPER_BASE_PT` overrides
    /// the path.
    #[test]
    fn test_bunsen_encoder_matches_reference() {
        use bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner;

        let pt = env!("WHISPER_BASE_PT_PATH");

        let device = Default::default();
        let mels = synthetic_mels::<B>(&device);

        let reference = encoder::Model::<B>::load_reference(&device).forward(mels.clone());

        let (model, cfg) = PytorchWhisperScanner::new()
            .load::<B, _>(std::path::PathBuf::from(pt), &device)
            .expect("load base.pt");
        assert_eq!(cfg.n_mels, N_MELS, "WHISPER_BASE_PT is not a `base` model");

        // OpenAI ships these checkpoints in fp16. The reference graph is f32,
        // so compare like with like.
        let model = model.map(&mut DTypeMapper::new(burn::tensor::DType::F32));

        let ours = model.forward_encoder(mels);
        assert_eq!(ours.dims(), reference.dims());

        // Two independent implementations, six transformer blocks deep, on
        // whatever precision and reduction order the vendor's matmul chooses.
        // The agreement is accumulation-limited, and the binding constraint is
        // the *backend*, not the implementations: wgpu lands inside 1e-3
        // absolute, while CUDA drifts to ~1.1e-2 (7.8e-2 relative on
        // near-zero elements), which is the signature of a reduced-precision
        // matmul rather than a disagreement about the model.
        //
        // Set from measurement with headroom over CUDA. Still far tighter
        // than any real defect: the three this crate was built to catch were
        // each wrong by 100% or more.
        ours.to_data_as::<F>()
            .assert_approx_eq::<F>(&reference.to_data_as::<F>(), Tolerance::rel_abs(1e-1, 2e-2));
    }

    /// Loads bunsen's Whisper from the fetched checkpoint, in f32.
    fn load_bunsen() -> (
        bunsen::kits::speech::whisper::blocks::Whisper<B>,
        burn::prelude::Device<B>,
    ) {
        use bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner;

        let device: burn::prelude::Device<B> = Default::default();
        let (model, _) = PytorchWhisperScanner::new()
            .load::<B, _>(
                std::path::PathBuf::from(env!("WHISPER_BASE_PT_PATH")),
                &device,
            )
            .expect("load base.pt");

        // OpenAI ships these checkpoints in fp16; the reference graph is f32.
        // Feeding f32 input to an f16 model does not error here, it just
        // returns wrong numbers, so this cast is load-bearing.
        (
            model.map(&mut DTypeMapper::new(burn::tensor::DType::F32)),
            device,
        )
    }

    /// The decoder inputs both implementations see.
    fn decoder_inputs(
        device: &burn::prelude::Device<B>
    ) -> (Tensor<B, 2, burn::tensor::Int>, Tensor<B, 3>) {
        let tokens = Tensor::from_data(TensorData::new(TOKENS.to_vec(), [1, TOKENS.len()]), device);
        (tokens, synthetic_encoder_output::<B>(device))
    }

    /// **The decoder cross-check.** bunsen's text decoder must match the
    /// reference on identical weights, tokens and encoder output.
    #[test]
    fn test_bunsen_decoder_matches_reference() {
        let (model, device) = load_bunsen();
        let (tokens, xa) = decoder_inputs(&device);

        // `.0` is the logits; the rest of the tuple is the present KV cache.
        let reference = decoder::Model::<B>::load_reference(&device)
            .forward(tokens.clone(), xa.clone())
            .0;
        assert_eq!(reference.dims(), [1, TOKENS.len(), N_VOCAB]);

        let ours = model.forward_decoder(tokens, xa);
        assert_eq!(ours.dims(), reference.dims());

        ours.to_data_as::<F>()
            .assert_approx_eq::<F>(&reference.to_data_as::<F>(), Tolerance::rel_abs(1e-1, 2e-2));
    }

    /// The predicted token must agree at every position.
    ///
    /// Logits span a wide range over 51865 classes, so an elementwise
    /// tolerance can pass while the argmax differs — which is the only thing a
    /// decoder is actually judged on.
    #[test]
    fn test_bunsen_decoder_argmax_matches_reference() {
        let (model, device) = load_bunsen();
        let (tokens, xa) = decoder_inputs(&device);

        let reference = decoder::Model::<B>::load_reference(&device)
            .forward(tokens.clone(), xa.clone())
            .0;
        let ours = model.forward_decoder(tokens, xa);

        let pick = |t: Tensor<B, 3>| -> Vec<i64> {
            t.argmax(2)
                .flatten::<1>(0, 2)
                .into_data()
                .convert::<i64>()
                .to_vec()
                .unwrap()
        };

        assert_eq!(
            pick(ours),
            pick(reference),
            "the decoders disagree on the predicted token",
        );
    }
}
