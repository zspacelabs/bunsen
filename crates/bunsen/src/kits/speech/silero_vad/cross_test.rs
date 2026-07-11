#[cfg(test)]
mod tests {
    use std::path::Path;

    use burn::{
        Tensor,
        prelude::TensorData,
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
        errors::*,
        kits::speech::silero_vad::{
            SileroVad,
            SileroVadCollection,
            SileroVadContextConfig,
            SileroVadMeta,
            reference::ReferenceModel,
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
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::permissive());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::permissive());
        }
    }

    #[test]
    #[serial_test::serial]
    /// This is stable for `flex`, `wgpu`.
    ///
    /// This produces inaccurate results for `cuda`.
    fn test_golden_context() -> Result<(), Box<dyn std::error::Error>> {
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
