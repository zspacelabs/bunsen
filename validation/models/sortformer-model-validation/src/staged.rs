//! bunsen's Sortformer against the references, one streaming step.

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

/// Against the generated reference, on synthetic input and identical weights.
mod nemotron_3_diarization {
    use super::*;

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
    fn step_matches_reference() {
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
}

/// Against onnxruntime's record of its export, on synthetic input and
/// identical weights.
mod sortformer_v2_1 {
    use super::*;

    /// bunsen's Sortformer v2.1, loaded from NVIDIA's `.nemo` through the
    /// factory, steps as onnxruntime ran `Scrybl`'s export of it: the chunk's
    /// pre-encode embeddings, and the speaker probabilities over the cache, the
    /// FIFO and the chunk.
    ///
    /// As for Nemotron, only the steps whose cache and FIFO are wholly valid
    /// are compared (the `reference` tests cover the others); one of them has
    /// neither. Measured on wgpu, the worst sample is off by 7.2e-7 (`preds`)
    /// and 1.6e-4 (`chunk_pre_encode_embs`, which run to the hundreds).
    #[test]
    #[serial_test::serial]
    fn step_matches_reference() {
        use bunsen::kits::speech::sortformer::{
            SortformerFamily,
            SortformerMeta,
        };

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let bundle = load_named_bundle("sortformer-v2.1", &device);
        assert_eq!(bundle.family, SortformerFamily::SortformerV2);
        let model = &bundle.model;
        assert_eq!(model.n_speakers(), V2_N_SPEAKERS);

        let mut compared = 0;
        for step in onnx_reference().v2_1 {
            if step.spkcache_valid != step.spkcache_frames || step.fifo_valid != step.fifo_frames {
                continue;
            }
            let what = format!(
                "v2.1, {} mel frames, cache {}, fifo {}",
                step.mel_frames, step.spkcache_frames, step.fifo_frames
            );
            let (embs, lengths) = model.pre_encode(synthetic_chunk(step.mel_frames, &device), None);
            assert_eq!(
                lengths.iter().map(|&l| l as i64).collect::<Vec<_>>(),
                step.chunk_pre_encode_lengths,
                "{what}: lengths"
            );
            let embs_error = step.chunk_pre_encode_embs.assert_matches(
                &format!("{what}: chunk_pre_encode_embs"),
                embs.clone(),
                1e-3,
                1e-3,
            );

            // A part with no frames is left out: a zero-length tensor is never
            // built.
            let parts: Vec<Tensor<3>> = [(step.spkcache_frames, 1), (step.fifo_frames, 2)]
                .into_iter()
                .filter(|&(frames, _)| frames > 0)
                .map(|(frames, salt)| synthetic_embeddings(frames, salt, &device))
                .chain([embs])
                .collect();
            let probs = model.forward_pre_encoded(Tensor::cat(parts, 1), None);
            let preds_error = step.preds.assert_matches(
                &format!("{what}: preds"),
                probs.encoder_rate,
                1e-3,
                1e-4,
            );
            eprintln!("{what}: preds {preds_error}, embeddings {embs_error}");
            compared += 1;
        }
        assert_eq!(compared, 2);
    }
}
