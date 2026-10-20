//! bunsen's Sortformer against the generated reference, one streaming step on
//! synthetic input and identical weights (`Nemotron-3-Diarization`).

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// The largest difference allowed, with room for backends whose matmuls
/// drift further than wgpu's.
const TOLERANCE: f32 = 1e-4;

/// The largest elementwise difference.
fn max_diff(
    a: Tensor<3>,
    b: Tensor<3>,
) -> f32 {
    (a - b).abs().max().into_scalar::<f32>()
}

/// bunsen's step is the reference's: the chunk's pre-encode embeddings, and
/// both resolutions of the speaker probabilities, at each geometry.
///
/// The cache and FIFO are given whole, every frame valid. A stream's cache
/// holds exactly its valid frames, and the graph reads a partly valid one
/// differently from a packed one (the `reference` tests still cover that
/// path against onnxruntime). Measured on wgpu, the worst difference is
/// 6e-7.
#[test]
#[serial_test::serial]
fn test_bunsen_step_matches_reference() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let bundle = load_bundle(&device);
    let model = &bundle.model;
    let reference = reference::NemotronDiarizationModel::load_pretrained(&device);

    for step in onnx_reference().nemotron {
        let what = format!("{} mel frames", step.mel_frames);
        let chunk = synthetic_chunk(step.mel_frames, &device);
        let spkcache = synthetic_embeddings(step.spkcache_frames, 1, &device);
        let fifo = synthetic_embeddings(step.fifo_frames, 2, &device);
        let (r_diar, r_hires, r_embs, _) = reference.forward(
            chunk.clone(),
            graph_length(step.mel_frames, &device),
            spkcache.clone(),
            graph_length(step.spkcache_frames, &device),
            fifo.clone(),
            graph_length(step.fifo_frames, &device),
        );

        let (embs, _) = model.pre_encode(chunk, None);
        let d = max_diff(embs.clone(), r_embs);
        eprintln!("{what}: pre-encode embeddings max diff {d:.3e}");
        assert!(d < TOLERANCE, "{what}: pre-encode embeddings off by {d}");

        let input = Tensor::cat(vec![spkcache, fifo, embs], 1);
        let frames = input.dims()[1];
        let probs = model.forward_pre_encoded(input, None);
        let d = max_diff(probs.encoder_rate, r_diar.slice_dim(1, 0..frames));
        eprintln!("{what}: 80 ms probabilities max diff {d:.3e}");
        assert!(d < TOLERANCE, "{what}: 80 ms probabilities off by {d}");
        let d = max_diff(probs.output, r_hires.slice_dim(1, 0..frames * SUBSAMPLING));
        eprintln!("{what}: 10 ms probabilities max diff {d:.3e}");
        assert!(d < TOLERANCE, "{what}: 10 ms probabilities off by {d}");
    }
}
