//! The generated reference model against onnxruntime's run of the published
//! graph.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// The generated streaming step computes what onnxruntime computes for the
/// same graph and input, at each geometry: both resolutions of the
/// predictions, the chunk's pre-encode embeddings, and their lengths.
///
/// Measured on wgpu, the worst sample is off by 1.3e-6 (`preds_diar`), 1.6e-6
/// (`preds_hires`) and 3.4e-5 on a value of 20.5 (`chunk_pre_encode_embs`);
/// the tolerances leave room for backends whose matmuls drift further.
#[test]
#[serial_test::serial]
fn test_reference_step_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let model = reference::NemotronDiarizationModel::load_pretrained(&device);

    for step in onnx_reference().nemotron {
        let what = format!("{} mel frames", step.mel_frames);
        let (preds, hires, embs, lengths) = model.forward(
            synthetic_chunk(step.mel_frames, &device),
            graph_length(step.mel_frames, &device),
            synthetic_embeddings(step.spkcache_frames, 1, &device),
            graph_length(step.spkcache_valid, &device),
            synthetic_embeddings(step.fifo_frames, 2, &device),
            graph_length(step.fifo_valid, &device),
        );

        let lengths: Vec<i64> = lengths.into_data().try_into_vec_as::<i64>().unwrap();
        assert_eq!(lengths, step.chunk_pre_encode_lengths, "{what}: lengths");
        step.preds_diar
            .assert_matches(&format!("{what}: preds_diar"), preds, 1e-3, 1e-4);
        step.preds_hires
            .assert_matches(&format!("{what}: preds_hires"), hires, 1e-3, 1e-4);
        step.chunk_pre_encode_embs.assert_matches(
            &format!("{what}: chunk_pre_encode_embs"),
            embs,
            1e-3,
            1e-3,
        );
    }
}

/// The generated v2.1 step computes what onnxruntime computes for the same
/// graph and input, at each geometry, a first step with an empty cache and
/// FIFO among them: the 80 ms predictions, the chunk's pre-encode
/// embeddings, and their lengths.
///
/// Measured on wgpu, the worst sample is off by 7.7e-7 (`preds`, the empty
/// first step) and 1.8e-4 (`chunk_pre_encode_embs`); the test prints what it
/// measures, and the tolerances leave room for backends whose convolutions
/// drift further.
#[test]
#[serial_test::serial]
fn test_v2_1_reference_step_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let model = reference::SortformerV21Model::load_pretrained(&device);

    for step in onnx_reference().v2_1 {
        let what = format!(
            "v2.1, {} mel frames, cache {}",
            step.mel_frames, step.spkcache_frames
        );
        let (preds, embs, lengths) = model.forward(
            synthetic_chunk(step.mel_frames, &device),
            graph_length(step.mel_frames, &device),
            synthetic_embeddings(step.spkcache_frames, 1, &device),
            graph_length(step.spkcache_valid, &device),
            synthetic_embeddings(step.fifo_frames, 2, &device),
            graph_length(step.fifo_valid, &device),
        );

        let lengths: Vec<i64> = lengths.into_data().try_into_vec_as::<i64>().unwrap();
        assert_eq!(lengths, step.chunk_pre_encode_lengths, "{what}: lengths");
        let preds = step
            .preds
            .assert_matches(&format!("{what}: preds"), preds, 1e-3, 1e-4);
        let embs = step.chunk_pre_encode_embs.assert_matches(
            &format!("{what}: chunk_pre_encode_embs"),
            embs,
            1e-3,
            1e-3,
        );
        eprintln!("{what}: preds {preds}, embeddings {embs}");
    }
}
