#[cfg(test)]
mod test {
    use burn::tensor::{
        Distribution,
        Tensor,
        Tolerance,
        backend::BackendTypes,
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

        let vad: TenVad<B> = TenVad::load_pretrained(&device);

        let batch = 1;
        let input = Tensor::random([batch, 3, 41], Distribution::Default, &device);

        let lstm1_hidden = Tensor::zeros([batch, 64], &device);
        let lstm1_cell = Tensor::zeros([batch, 64], &device);
        let lstm2_hidden = Tensor::zeros([batch, 64], &device);
        let lstm2_cell = Tensor::zeros([batch, 64], &device);

        let (ref_prob, ref_lstm1_hidden, ref_lstm1_cell, ref_lstm2_hidden, ref_lstm2_cell) =
            ref_vad.forward(
                input.clone(),
                lstm1_hidden.clone(),
                lstm1_cell.clone(),
                lstm2_hidden.clone(),
                lstm2_cell.clone(),
            );

        let (mod_prob, mod_lstm1_hidden, mod_lstm1_cell, mod_lstm2_hidden, mod_lstm2_cell) = vad
            .forward(
                input.clone(),
                lstm1_hidden.clone(),
                lstm1_cell.clone(),
                lstm2_hidden.clone(),
                lstm2_cell.clone(),
            );

        mod_prob
            .to_data()
            .assert_approx_eq::<F>(&ref_prob.to_data(), Tolerance::permissive());

        mod_lstm1_hidden
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm1_hidden.to_data(), Tolerance::permissive());

        mod_lstm1_cell
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm1_cell.to_data(), Tolerance::permissive());

        mod_lstm2_hidden
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm2_hidden.to_data(), Tolerance::permissive());

        mod_lstm2_cell
            .to_data()
            .assert_approx_eq::<F>(&ref_lstm2_cell.to_data(), Tolerance::permissive());
    }
}
