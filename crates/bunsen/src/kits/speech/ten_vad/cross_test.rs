#[cfg(test)]
mod test {
    use burn::{
        nn::LstmState,
        tensor::{
            Distribution,
            Tensor,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use crate::{
        kits::speech::ten_vad::{
            TenVad,
            reference::ReferenceModel,
        },
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
        let batch = 1;

        let input = Tensor::random([batch, 3, 41], Distribution::Default, &device);

        let state1_init = LstmState::new(
            Tensor::zeros([batch, 64], &device).clone(),
            Tensor::zeros([batch, 64], &device).clone(),
        );
        let state2_init = LstmState::new(
            Tensor::zeros([batch, 64], &device).clone(),
            Tensor::zeros([batch, 64], &device).clone(),
        );

        let (ref_prob, ref_lstm1_hidden, ref_lstm1_cell, ref_lstm2_hidden, ref_lstm2_cell) =
            ref_vad.forward(
                input.clone(),
                state1_init.hidden.clone(),
                state1_init.cell.clone(),
                state2_init.hidden.clone(),
                state2_init.cell.clone(),
            );

        // TODO: LstmState is not Clone! Fix/Work-around.
        let (mod_prob, mod_lstm1_state, mod_lstm2_state) = vad.forward(input.clone(), None, None);

        mod_prob
            .to_data()
            .assert_approx_eq::<F>(&ref_prob.to_data(), Tolerance::permissive());

        mod_lstm1_state
            .hidden
            .clone()
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm1_hidden.to_data(), Tolerance::permissive());

        mod_lstm1_state
            .cell
            .clone()
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm1_cell.to_data(), Tolerance::permissive());

        mod_lstm2_state
            .hidden
            .clone()
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm2_hidden.to_data(), Tolerance::permissive());

        mod_lstm2_state
            .cell
            .clone()
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm2_cell.to_data(), Tolerance::permissive());
    }
}
