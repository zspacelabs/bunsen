//! Staged cross-checks against the ONNX reference, on synthetic input.
//!
//! Each stage is fed input that isolates it: the decoder sees a synthetic
//! encoder output rather than either encoder's real one, so a disagreement
//! cannot be inherited from upstream. `audio` runs the composition.

use bunsen::{
    burner::{
        module::DTypeMapper,
        tensor::TensorElemOpExt,
    },
    kits::speech::whisper::blocks::Whisper,
    support::testing::PerformanceBackend,
};
use burn::{
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
    let model = reference::EncoderModel::<B>::load_pretrained(&device);

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
/// `onnx-community/whisper-base` was converted from. It comes from
/// `bunsen-bundled-whisper`, which this crate's `download` feature pulls
/// in, so this never skips.
#[test]
fn test_bunsen_encoder_matches_reference() {
    let device = Default::default();
    let mels = synthetic_mels::<B>(&device);

    let reference = reference::EncoderModel::<B>::load_pretrained(&device).forward(mels.clone());

    let (model, cfg) = Whisper::<B>::load_pretrained(&device).expect("load base.pt");
    assert_eq!(cfg.n_mels, N_MELS, "the checkpoint is not a `base` model");

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
    // than any real defect: the ones this crate was built to catch were
    // each wrong by 100% or more.
    ours.to_data_as::<F>()
        .assert_approx_eq::<F>(&reference.to_data_as::<F>(), Tolerance::rel_abs(1e-1, 2e-2));
}

/// Loads bunsen's Whisper from the fetched checkpoint, in f32.
fn load_bunsen() -> (Whisper<B>, burn::prelude::Device<B>) {
    let device: burn::prelude::Device<B> = Default::default();
    let (model, _) = Whisper::<B>::load_pretrained(&device).expect("load base.pt");

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
    let reference = reference::DecoderModel::<B>::load_pretrained(&device)
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

    let reference = reference::DecoderModel::<B>::load_pretrained(&device)
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
