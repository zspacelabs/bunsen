//! # Parity against `librosa`.
//!
//! These compare the converter to fixtures generated once by
//! `tools/gen_mel_fixtures.py` and committed under `testdata/mels/`. Nothing
//! here shells out to Python — `librosa` is not a build or test dependency,
//! and regenerating the fixtures is a deliberate manual step.
//!
//! Everything else in this module tests the implementation against itself or
//! against a transcription of the reference algorithm. This file is the only
//! place that checks it against the reference *as published*, which is the one
//! thing a transcription cannot do.
//!
//! Fixtures are `f32`, as `librosa` emits, so the tolerances below are bounded
//! by the reference's own precision rather than by this implementation's.

use std::path::PathBuf;

use burn::{
    Tensor,
    prelude::TensorData,
    tensor::{
        Tolerance,
        backend::BackendTypes,
    },
};

use crate::{
    burner::{
        module::ModuleInit,
        tensor::TensorElemOpExt,
    },
    errors::WithOkOrPanic,
    ops::signal::{
        SamplingWindowBuilder,
        mels::{
            FilterNorm,
            MelConverter,
            MelConverterMeta,
            MelConverterOptions,
            MelScale,
            PaddingMode,
        },
    },
    support::testing::{
        PerformanceBackend,
        assert_close_to_vec,
    },
};

type B = PerformanceBackend;
type F = <B as BackendTypes>::FloatElem;

/// Loads a flat little-endian `f32` fixture as `f64`.
fn fixture(name: &str) -> Vec<f64> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/mels")
        .join(name);

    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()));

    assert_eq!(
        bytes.len() % 4,
        0,
        "fixture {name} is not a whole number of f32",
    );

    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

/// Asserts a tensor matches a fixture, comparing as `TensorData` so `burn`
/// reports the mismatch.
fn assert_matches_fixture<const D: usize>(
    actual: &Tensor<B, D>,
    name: &str,
    tolerance: Tolerance<F>,
) {
    let expected = TensorData::new(fixture(name), actual.dims()).convert::<F>();
    actual
        .to_data_as::<F>()
        .assert_approx_eq::<F>(&expected, tolerance);
}

/// The signal the log-mel fixtures were generated from.
fn signal_tensor(device: &burn::prelude::Device<B>) -> (Tensor<B, 2>, usize) {
    let samples = fixture("signal_2s_16k.f32");
    let n = samples.len();
    assert_eq!(n, 32_000);

    (
        Tensor::from_data(TensorData::new(samples, [1, n]), device),
        n,
    )
}

/// Tolerance for the log-mel comparisons.
///
/// The measured worst case against these fixtures is `6.2e-6` absolute
/// (`2.9e-6` relative) on the `center=False` path and `9.5e-7` on the streamed
/// one, so this leaves roughly 8x headroom. The floor is set by `librosa`
/// emitting `f32` and by `f32` accumulation over the 400-term DFT and the
/// 201-term mel matmul — not by anything this implementation could tighten.
fn logmel_tolerance() -> Tolerance<F> {
    Tolerance::<F>::rel_abs(1e-4, 5e-5)
}

/// Options matching the fixture generator: raw `log10` with no clamp and no
/// affine tail, so the comparison is against the spectrogram itself rather
/// than Whisper's packaging of it.
fn parity_options() -> MelConverterOptions {
    MelConverterOptions::default()
        .with_range_clamp(None)
        .with_affine(None)
}

#[test]
fn test_hann_window_matches_librosa() {
    let opts = MelConverterOptions::default();
    assert_close_to_vec(
        &opts.window.to_vec_window(opts.n_fft),
        &fixture("hann_400_periodic.f32"),
        1e-7,
    );
}

#[test]
fn test_filterbank_matches_librosa() {
    let opts = MelConverterOptions::default();
    assert_close_to_vec(
        &opts.to_vec_filterbank().unwrap(),
        &fixture("mel_fb_slaney_16k_400_80.f32"),
        1e-8,
    );

    let htk = MelConverterOptions::default()
        .with_mel_scale(MelScale::Htk)
        .with_filter_norm(FilterNorm::None);

    assert_close_to_vec(
        &htk.to_vec_filterbank().unwrap(),
        &fixture("mel_fb_htk_16k_400_80_nonorm.f32"),
        1e-6,
    );
}

/// The unpadded batch path, against `center=False`.
#[test]
fn test_batch_logmel_matches_librosa_center_false() {
    let device = Default::default();
    let opts = parity_options().with_start_padding(PaddingMode::None);
    let conv: MelConverter<B> = opts.try_init(&device).ok_or_panic();

    let (x, samples) = signal_tensor(&device);

    let mels = conv.forward(x);

    // librosa: 1 + (32000 - 400) / 160 == 198.
    assert_eq!(conv.frame_count(samples), 198);
    assert_eq!(mels.dims(), [1, 198, opts.n_mels]);

    assert_matches_fixture(&mels, "logmel_center_false.f32", logmel_tolerance());
}

/// **The parity milestone.** The streamed path — reflect start padding, a
/// carry across the chunk boundary, and a reflect-padded `finish` — must
/// reproduce `librosa` with `center=True`, frame for frame.
#[test]
fn test_streaming_logmel_matches_librosa_center_true() {
    let device = Default::default();
    let conv: MelConverter<B> = parity_options().try_init(&device).ok_or_panic();

    let (x, _) = signal_tensor(&device);

    let (mels, ctx) = conv.new_context(1).transform(x).unwrap();
    let tail = ctx.finish().expect("reflect end padding yields a tail");

    // 199 from the padded first call, 2 from the flush; librosa gives 201.
    assert_eq!(mels.dims()[1], 199);
    assert_eq!(tail.dims()[1], 2);

    let joined: Tensor<B, 3> = Tensor::cat(vec![mels, tail], 1);
    assert_eq!(joined.dims(), [1, 201, conv.n_mels()]);

    assert_matches_fixture(&joined, "logmel_center_true.f32", logmel_tolerance());
}

/// The same parity, reached by feeding the signal in uneven pieces.
#[test]
fn test_chunked_streaming_matches_librosa_center_true() {
    let device = Default::default();
    let conv: MelConverter<B> = parity_options().try_init(&device).ok_or_panic();

    let samples = fixture("signal_2s_16k.f32");

    let mut ctx = conv.new_context(1);
    let mut pieces = Vec::new();
    let mut at = 0;

    for n in [3_200, 1_600, 12_800, 6_400, 8_000] {
        let chunk = Tensor::from_data(
            TensorData::new(samples[at..at + n].to_vec(), [1, n]),
            &device,
        );
        let (mels, next) = ctx.transform(chunk).unwrap();
        ctx = next;
        pieces.push(mels);
        at += n;
    }
    assert_eq!(at, samples.len());

    pieces.push(ctx.finish().unwrap());

    let joined: Tensor<B, 3> = Tensor::cat(pieces, 1);
    assert_eq!(joined.dims(), [1, 201, conv.n_mels()]);

    assert_matches_fixture(&joined, "logmel_center_true.f32", logmel_tolerance());
}
