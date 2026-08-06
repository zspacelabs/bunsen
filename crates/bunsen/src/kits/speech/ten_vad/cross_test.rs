#[cfg(test)]
mod test {
    use burn::tensor::{
        Distribution,
        Tensor,
        Tolerance,
        backend::BackendTypes,
    };

    use crate::{
        blocks::rnn::lstm::ExtLstmState,
        kits::speech::ten_vad::{
            TenVad,
            reference::ReferenceModel,
        },
        prelude::*,
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial_test::serial]
    #[allow(unused)]
    fn test_reference_model_forward_cross_test() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();

        let ref_vad: ReferenceModel<B> = ReferenceModel::load_pretrained(&device);

        let vad: TenVad<B> = TenVad::load_pretrained(&device).unwrap();

        // TODO: batch support appears to be broken?
        let shape = [1, 64];

        let input = Tensor::random([1, 3, 41], Distribution::Default, &device);

        let state1_init = ExtLstmState::initial(shape, &device);
        let state2_init = ExtLstmState::initial(shape, &device);

        let (ref_prob, ref_lstm1_hidden, ref_lstm1_cell, ref_lstm2_hidden, ref_lstm2_cell) =
            ref_vad.forward(
                input.clone(),
                state1_init.hidden.clone(),
                state1_init.cell.clone(),
                state2_init.hidden.clone(),
                state2_init.cell.clone(),
            );

        let (mod_prob, mod_lstm1_state, mod_lstm2_state) = vad.forward(input.clone(), None, None);

        mod_prob
            .unsqueeze_dim::<3>(2)
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_prob.to_data_as::<F>(), Tolerance::permissive());

        mod_lstm1_state
            .hidden
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_lstm1_hidden.to_data_as::<F>(), Tolerance::permissive());

        mod_lstm1_state
            .cell
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_lstm1_cell.to_data_as::<F>(), Tolerance::permissive());

        mod_lstm2_state
            .hidden
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_lstm2_hidden.to_data_as::<F>(), Tolerance::permissive());

        mod_lstm2_state
            .cell
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_lstm2_cell.to_data_as::<F>(), Tolerance::permissive());
    }
}
