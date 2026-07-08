#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use crate::{
        errors::*,
        kits::speech::silero_vad::{
            SileroVadCollection,
            SileroVadMeta,
            reference::Model as ReferenceModel,
        },
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial_test::serial]
    fn test_reference_model_forward_cross_test() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();

        let sc: SileroVadCollection<B> =
            SileroVadCollection::load_pretrained(&device).ok_or_panic();

        let r_mod: ReferenceModel<B> = ReferenceModel::load_pretrained(&device);

        let batch = 8;

        for sample_rate in [16000, 8000] {
            let vad = sc.expect_branch(sample_rate);

            if sample_rate == 16000 {
                assert_eq!(vad.chunk_size(), 512)
            }

            let input =
                Tensor::<B, 2>::random([batch, vad.chunk_size()], Distribution::Default, &device);
            let state = vad.init_state(batch, &device);

            // ([batch], [2, batch, d_hidden])
            let input1 = input.clone();
            let state1 = state.clone();
            let (s_out, s_state) = vad.forward(input1, state1);

            // ([batch, 1], [2, batch, d_hidden])
            let (r_out, r_state) = r_mod.forward(input, sample_rate as i64, state.clone());

            s_out
                .reshape([batch, 1])
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }
    }
}
