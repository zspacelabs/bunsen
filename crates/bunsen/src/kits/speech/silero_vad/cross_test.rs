#[cfg(test)]
mod tests {
    use std::path::Path;

    use burn::{
        Tensor,
        prelude::{
            Backend,
            TensorData,
        },
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };
    use hound::{
        SampleFormat,
        WavReader,
        WavSpec,
    };

    use crate::{
        burner::tensor::*,
        errors::*,
        kits::speech::silero_vad::{
            SileroVad,
            SileroVadCollection,
            SileroVadContextConfig,
            SileroVadMeta,
            reference::ReferenceModel,
        },
        prelude::*,
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
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&r_out.to_data_as::<F>(), Tolerance::permissive());

            s_state
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&r_state.to_data_as::<F>(), Tolerance::permissive());
        }
    }

    static WAV_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.wav");
    static WAV_SR: usize = 16000;
    static CTX_PROBS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.json");
    static FSEQ_PROBS_PATH: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/fseq.json");

    #[test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    fn backend_golden_crosstest() -> Result<(), Box<dyn std::error::Error>> {
        type A = burn::backend::Cuda;
        type B = burn::backend::Flex;

        let device_a = Default::default();
        let device_b = Default::default();

        let vad_a: SileroVad<A> = SileroVadCollection::load_pretrained(&device_a)?
            .try_branch(WAV_SR)?
            .clone();
        let vad_b: SileroVad<B> = SileroVadCollection::load_pretrained(&device_b)?
            .try_branch(WAV_SR)?
            .clone();

        println!("approx_eq?(vad_a, vad_b, $mod::lstm.features.weight");
        vad_a
            .lstm
            .features
            .weight
            .val()
            .clone()
            .to_data()
            .assert_approx_eq(
                &vad_b.lstm.features.weight.val().clone().to_data(),
                Tolerance::<f32>::default(),
            );
        println!("approx_eq?(vad_a, vad_b, $mod::lstm.hidden.weight");
        vad_a
            .lstm
            .hidden
            .weight
            .val()
            .clone()
            .to_data()
            .assert_approx_eq(
                &vad_b.lstm.hidden.weight.val().clone().to_data(),
                Tolerance::<f32>::default(),
            );

        let batch = 1;
        let state_a = vad_a.init_state(batch, &device_a);
        let state_b = vad_b.init_state(batch, &device_b);

        // [1, chunk_size]
        let chunk_a = load_golden_wav_tensor::<A>(&device_a, vad_a.chunk_size())?.select_dim(0, 0);
        let chunk_b = load_golden_wav_tensor::<B>(&device_b, vad_b.chunk_size())?.select_dim(0, 0);

        println!("unrolled::frame_features");
        let f_a = vad_a.frame_features(chunk_a.clone());
        let f_b = vad_b.frame_features(chunk_b.clone());
        f_a.clone()
            .to_data()
            .assert_approx_eq(&f_b.clone().to_data(), Tolerance::<f32>::default());

        let (h_a, c_a) = SileroVad::unpack_state(state_a.clone());
        let (h_b, c_b) = SileroVad::unpack_state(state_b.clone());

        println!("unrolled::lstm_step::gates::features.forward");
        let ff_a = vad_a.lstm.features.forward(f_a.clone());
        let ff_b = vad_b.lstm.features.forward(f_b.clone());
        ff_a.clone()
            .to_data()
            .assert_approx_eq(&ff_b.clone().to_data(), Tolerance::<f32>::default());

        println!("unrolled::lstm_step::gates::hidden.forward");
        let hf_a = vad_a.lstm.hidden.forward(h_a.clone());
        let hf_b = vad_b.lstm.hidden.forward(h_b.clone());
        hf_a.clone()
            .to_data()
            .assert_approx_eq(&hf_b.clone().to_data(), Tolerance::<f32>::default());

        println!("unrolled::lstm_step::gates");
        let gates_a = ff_a + hf_a;
        let gates_b = ff_b + hf_b;
        gates_a
            .clone()
            .to_data()
            .assert_approx_eq(&gates_b.clone().to_data(), Tolerance::<f32>::default());

        println!("unrolled::lstm_step");
        let (h_a, c_a) = vad_a.lstm_step(f_a, h_a, c_a);
        let (h_b, c_b) = vad_b.lstm_step(f_b, h_b, c_b);
        h_a.clone()
            .to_data()
            .assert_approx_eq(&h_b.clone().to_data(), Tolerance::<f32>::default());
        c_a.clone()
            .to_data()
            .assert_approx_eq(&c_b.clone().to_data(), Tolerance::<f32>::default());

        println!("unrolled::output_head");
        let o_a = vad_a.output_head(h_a.clone());
        let o_b = vad_b.output_head(h_b.clone());
        o_a.to_data()
            .assert_approx_eq(&o_b.to_data(), Tolerance::<f32>::default());

        println!("VAD::forward");
        let (p_a, _) = vad_a.forward(chunk_a, state_a);
        let (p_b, _) = vad_b.forward(chunk_b, state_b);

        p_a.to_data()
            .assert_approx_eq(&p_b.to_data(), Tolerance::<f32>::default());

        Ok(())
    }

    fn load_golden_wav_tensor<B: Backend>(
        device: &B::Device,
        chunk_size: usize,
    ) -> BunsenResult<Tensor<B, 3>> {
        let (_, mut wav_vec) = load_audio_mono_sr(WAV_PATH, WAV_SR)?;
        // [steps, 1, samples=chunk_size]
        let chunk_seq: Tensor<B, 3> = {
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
        Ok(chunk_seq)
    }

    #[test]
    #[serial_test::serial]
    fn test_golden_context() -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        eprintln!("This test is known to fail on the CUDA backend.\n");

        type B = PerformanceBackend;
        let device = Default::default();

        let vad: SileroVad<B> = SileroVadCollection::load_pretrained(&device)?
            .try_branch(WAV_SR)?
            .clone();

        let chunk_seq = load_golden_wav_tensor::<B>(&device, vad.chunk_size())?;

        {
            let state = vad.init_state(1, &device);
            let (chunk_probs, _state) = vad.forward_sequence(chunk_seq.clone(), state);
            let chunk_probs: Tensor<B, 1> = chunk_probs.squeeze_dim::<1>(1);

            let expected: Vec<f32> = serde_json::from_reader(
                std::fs::File::open(FSEQ_PROBS_PATH).map_err(BunsenError::external)?,
            )
            .map_err(BunsenError::external)?;
            let expected: TensorData = TensorData::from(expected.as_slice());

            chunk_probs
                .to_data()
                .assert_approx_eq(&expected, Tolerance::<f32>::default());

            // let _probs: Vec<f32> =
            // chunk_probs.cast(DType::F32).to_data().to_vec()?;
            //  println!("chunk_probs: {:?}", _probs);
        }

        // Context processing.
        {
            // [steps, batch=1]
            let (chunk_probs, _ctx) = vad.context_forward_sequence(
                chunk_seq,
                SileroVadContextConfig::new(WAV_SR).init(&vad, &device),
            );

            // [steps]
            let chunk_probs = chunk_probs.squeeze_dim::<1>(1).to_data();

            // [steps]
            let expected: Vec<f32> = serde_json::from_reader(
                std::fs::File::open(CTX_PROBS_PATH).map_err(BunsenError::external)?,
            )
            .map_err(BunsenError::external)?;
            let expected: TensorData = TensorData::from(expected.as_slice());

            chunk_probs.assert_approx_eq(&expected, Tolerance::<f32>::default());
        }

        Ok(())
    }

    /// Loads a mono audio file.
    ///
    /// # Arguments
    /// * `filename` - path to an audio file.
    /// * `sample_rate` - sample rate of the audio file.
    pub fn load_audio_mono_sr<P: AsRef<Path>>(
        filename: P,
        sample_rate: usize,
    ) -> BunsenResult<(WavSpec, Vec<f32>)> {
        let filename = filename.as_ref();

        let mut reader = WavReader::open(filename).map_err(BunsenError::external)?;
        let spec = reader.spec();

        if spec.channels != 1 {
            return Err(BunsenError::Invalid(
                "The audio must be single-channel".to_string(),
            ));
        }
        if spec.sample_rate as usize != sample_rate {
            return Err(BunsenError::Invalid(format!(
                "Expected sample_rate = {}, but found {}",
                sample_rate, spec.sample_rate
            )));
        }

        let spec = reader.spec();
        let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .map(|s| s.unwrap())
                .collect::<Vec<f32>>(),
            (SampleFormat::Int, bits) => {
                let scale = (1i64 << (bits - 1)) as f32;
                reader
                    .samples::<i32>()
                    .collect::<Result<Vec<i32>, _>>()
                    .map_err(BunsenError::external)?
                    .into_iter()
                    .map(|s| s as f32 / scale)
                    .collect()
            }
            _ => unreachable!("hound rejects other formats at open"),
        };

        Ok((spec, samples))
    }
}
