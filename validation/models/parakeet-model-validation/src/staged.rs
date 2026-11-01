//! bunsen's Parakeet TDT against the generated reference, stage by stage.

use bunsen::{
    kits::speech::parakeet::blocks::TdtPredictionState,
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};
use burn::tensor::Tolerance;

use super::*;

/// The base model's name in bunsen's index: the weights the reference
/// graphs carry.
const BASE: &str = "nvidia/parakeet-tdt-0.6b-v3";

/// On synthetic input, with the weights the reference carries.
mod parakeet_tdt_0_6b_v3 {
    use super::*;

    /// bunsen's encoder matches the reference encoder on identical weights.
    #[test]
    #[serial_test::serial]
    fn encoder_matches_reference() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let length = Tensor::<1, Int>::from_data(TensorData::from([N_FRAMES as i64]), &device);
        let (reference, _) = reference::EncoderModel::load_pretrained(&device)
            .forward(synthetic_features(&device), length);

        let bundle = load_bundle(BASE, &device);
        let (ours, lengths) = bundle
            .model
            .encoder
            .forward(synthetic_features(&device), None);
        assert_eq!(lengths, [N_FRAMES / 8]);
        ours.swap_dims(1, 2)
            .to_data_as::<f32>()
            .assert_approx_eq::<f32>(
                &reference.to_data_as::<f32>(),
                Tolerance::rel_abs(1e-2, 2e-3),
            );
    }

    /// bunsen's encoder projection, prediction network and joint match the
    /// reference's one-step graph: logits, both argmaxes, and both LSTM states.
    #[test]
    #[serial_test::serial]
    fn decoder_joint_matches_reference() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference().decoder_joint;
        let reference = reference::DecoderJointModel::load_pretrained(&device);
        let bundle = load_bundle(BASE, &device);
        let model = &bundle.model;

        for (i, step) in fixture.steps.iter().enumerate() {
            let what = format!("step {i} (target {})", step.target);
            let (s1, s2) = (
                synthetic_state(2 * i, &device),
                synthetic_state(2 * i + 1, &device),
            );
            let (ref_logits, _, ref_h, ref_c) = reference.forward(
                synthetic_encoder_frame(&device),
                Tensor::<2, Int>::from_data(TensorData::from([[step.target as i32]]), &device),
                Tensor::<1, Int>::from_data(TensorData::from([1i32]), &device),
                s1.clone(),
                s2.clone(),
            );

            let frame = model
                .encoder_projector
                .forward(synthetic_encoder_frame(&device).reshape([1, D_ENC]));
            let token = Tensor::<1, Int>::from_data(TensorData::from([step.target]), &device);
            let (prediction, state) = model.decoder.step(
                token,
                TdtPredictionState {
                    hidden: s1,
                    cell: s2,
                },
            );
            let logits = model.joint.forward(frame, prediction);

            let tolerance = Tolerance::rel_abs(1e-3, 1e-3);
            logits.to_data_as::<f32>().assert_approx_eq::<f32>(
                &ref_logits
                    .reshape([1, BLANK + 1 + N_DURATIONS])
                    .to_data_as::<f32>(),
                tolerance,
            );
            state
                .hidden
                .to_data_as::<f32>()
                .assert_approx_eq::<f32>(&ref_h.to_data_as::<f32>(), tolerance);
            state
                .cell
                .to_data_as::<f32>()
                .assert_approx_eq::<f32>(&ref_c.to_data_as::<f32>(), tolerance);
            eprintln!("{what}: ok");
        }
    }
}
