//! The generated reference models against onnxruntime's run of the same graphs.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;

/// The generated encoder computes what onnxruntime computes for the same
/// graph and input.
#[test]
#[serial_test::serial]
fn test_reference_encoder_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().encoder;

    let model = reference::EncoderModel::load_pretrained(&device);
    let length = Tensor::<1, Int>::from_data(TensorData::from([N_FRAMES as i64]), &device);
    let (outputs, lengths) = model.forward(synthetic_features(&device), length);

    let lengths: Vec<i64> = lengths.into_data().try_into_vec_as::<i64>().unwrap();
    assert_eq!(lengths, fixture.encoded_lengths);
    fixture
        .outputs
        .assert_matches("encoder outputs", outputs, 1e-2, 2e-3);
}

/// The generated prediction network and joint compute what onnxruntime
/// computes, step by step: logits, both argmaxes, and both LSTM states.
#[test]
#[serial_test::serial]
fn test_reference_decoder_joint_matches_onnxruntime() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().decoder_joint;

    let model = reference::DecoderJointModel::load_pretrained(&device);
    for (i, step) in fixture.steps.iter().enumerate() {
        let targets =
            Tensor::<2, Int>::from_data(TensorData::from([[step.target as i32]]), &device);
        let target_length = Tensor::<1, Int>::from_data(TensorData::from([1i32]), &device);
        let (logits, _, s1, s2) = model.forward(
            synthetic_encoder_frame(&device),
            targets,
            target_length,
            synthetic_state(2 * i, &device),
            synthetic_state(2 * i + 1, &device),
        );

        let what = format!("step {i} (target {})", step.target);
        let row = logits.clone().reshape([BLANK + 1 + N_DURATIONS]);
        let token = row
            .clone()
            .slice(0..BLANK + 1)
            .argmax(0)
            .into_scalar::<i64>() as usize;
        let duration = row
            .slice(BLANK + 1..BLANK + 1 + N_DURATIONS)
            .argmax(0)
            .into_scalar::<i64>() as usize;
        assert_eq!(token, step.token_argmax, "{what}: token argmax");
        assert_eq!(duration, step.duration_argmax, "{what}: duration argmax");

        step.logits
            .assert_matches(&format!("{what} logits"), logits, 1e-3, 1e-3);
        step.output_states_1
            .assert_matches(&format!("{what} state 1"), s1, 1e-3, 1e-4);
        step.output_states_2
            .assert_matches(&format!("{what} state 2"), s2, 1e-3, 1e-4);
    }
}
