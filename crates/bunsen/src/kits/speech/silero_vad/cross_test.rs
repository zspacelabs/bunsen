#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        prelude::TensorData,
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use crate::{
        errors::*,
        kits::speech::silero_vad::{
            SileroVad,
            SileroVadCollection,
            SileroVadContextConfig,
            SileroVadMeta,
            reference::ReferenceModel,
        },
        prelude::*,
        support::testing::{
            PerformanceBackend,
            audio::load_audio_mono_sr,
        },
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
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&r_out.to_data_as::<F>(), Tolerance::permissive());

            s_state
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&r_state.to_data_as::<F>(), Tolerance::permissive());
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_golden_context() -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        eprintln!("This test is known to fail on the CUDA backend.\n");

        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.wav");
        let expected_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.json");
        let sample_rate = 16000;

        type B = PerformanceBackend;
        let device = Default::default();

        let vad: SileroVad<B> = SileroVadCollection::load_pretrained(&device)?
            .try_branch(sample_rate)?
            .clone();

        let (_, mut wav_vec) = load_audio_mono_sr(wav_path, sample_rate)?;

        // [steps, 1, samples=chunk_size]
        let chunk_seq: Tensor<B, 3> = {
            let chunk_size = vad.chunk_size();

            // Pad the audio to the chunk size.
            let tail_len = wav_vec.len() % chunk_size;
            if tail_len != 0 {
                let pad_len = chunk_size - tail_len;
                wav_vec.resize(wav_vec.len() + pad_len, 0.0);
            }

            // Convert to tensor.
            let samples = Tensor::<B, 1>::from_floats(wav_vec.as_slice(), &device);

            // Chunk the audio into chunks of size `chunk_size`.
            samples.reshape([-1, 1, chunk_size as isize])
        };

        // [steps, batch=1]
        let (chunk_probs, _ctx) = vad.context_forward_sequence(
            chunk_seq,
            SileroVadContextConfig::new(sample_rate).init(&vad, &device),
        );

        // [steps]
        let chunk_probs = chunk_probs.squeeze_dim::<1>(1).to_data();

        // [steps]
        let expected: Vec<f32> = serde_json::from_reader(
            std::fs::File::open(expected_path).map_err(BunsenError::external)?,
        )
        .map_err(BunsenError::external)?;
        let expected: TensorData = TensorData::from(expected.as_slice());

        chunk_probs.assert_approx_eq(&expected, Tolerance::<f32>::default());

        Ok(())
    }
}
