//! Cross-checks against the generated ONNX reference, and a golden trace.
//!
//! `test_reference_model_forward_cross_test` steps bunsen against the
//! reference on identical random input, which is what catches a transliteration
//! error. `test_context_matches_upstream_wrapper` runs a real recording
//! through the streaming context at both rates and steps the reference the
//! way upstream's `OnnxWrapper` does. `test_golden_context` runs a real
//! recording through the streaming context and pins the per-chunk
//! probabilities.

#[cfg(test)]
mod tests {
    use bunsen::{
        errors::*,
        kits::speech::silero_vad::{
            SileroVad,
            SileroVadCollection,
            SileroVadContextConfig,
            SileroVadMeta,
        },
        prelude::*,
        support::{
            audio::load_audio_mono_sr,
            testing::{
                DeviceMemoryGuard,
                PerformanceBackend,
                default_device,
            },
        },
    };
    use burn::{
        Tensor,
        prelude::TensorData,
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use crate::reference::ReferenceModel;

    #[test]
    #[serial_test::serial]
    fn test_reference_model_forward_cross_test() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

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

    /// The streaming context against upstream's `OnnxWrapper`
    /// (`utils_vad.py`), at both rates: each chunk is prefixed with the tail
    /// of the input before it, `64 if sr == 16000 else 32` samples, and the
    /// state is carried. The context bunsen's config opens by default must
    /// give the reference's probabilities, chunk for chunk, and its final
    /// state, on a real recording.
    #[test]
    #[serial_test::serial]
    fn test_context_matches_upstream_wrapper() -> Result<(), Box<dyn std::error::Error>> {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/test.wav");

        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let sc: SileroVadCollection<B> = SileroVadCollection::load_pretrained(&device)?;
        let r_mod: ReferenceModel<B> = ReferenceModel::load_pretrained(&device);

        // The recording is 16 kHz. For the 8 kHz branch, each pair of
        // samples is averaged into one: a crude low-pass and decimation, but
        // both implementations see the same samples, which is all this
        // comparison needs.
        let wav_16k = load_audio_mono_sr(wav_path, 16000)?;
        let wav_8k: Vec<f32> = wav_16k
            .chunks_exact(2)
            .map(|pair| (pair[0] + pair[1]) / 2.0)
            .collect();

        for (sample_rate, upstream_context, mut wav) in [(16000, 64, wav_16k), (8000, 32, wav_8k)] {
            let vad = sc.try_branch(sample_rate)?;
            let chunk = vad.chunk_size();

            wav.truncate(wav.len() / chunk * chunk);
            let steps = wav.len() / chunk;

            // [steps, 1, chunk]
            let chunks: Tensor<B, 3> =
                Tensor::<B, 1>::from_floats(wav.as_slice(), &device).reshape([steps, 1, chunk]);

            // [steps, 1]
            let (probs, context) = vad.context_forward_sequence(
                chunks.clone(),
                SileroVadContextConfig::new(sample_rate).init(vad, &device),
            );

            // Upstream's wrapper, a chunk at a time.
            let mut tail = Tensor::<B, 2>::zeros([1, upstream_context], &device);
            let mut state = vad.init_state(1, &device);
            let mut expected = Vec::with_capacity(steps);
            for step in 0..steps {
                let at = step as isize;
                let x = Tensor::cat(
                    vec![
                        tail,
                        chunks.clone().slice_dim(0, at..at + 1).reshape([1, chunk]),
                    ],
                    1,
                );
                let (out, next) = r_mod.forward(x.clone(), sample_rate as i64, state);
                state = next;
                tail = x.slice_dim(1, chunk as isize..(chunk + upstream_context) as isize);
                expected.push(out.reshape([1, 1]));
            }
            let expected = Tensor::cat(expected, 0);

            probs
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&expected.to_data_as::<F>(), Tolerance::permissive());
            context
                .state
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&state.to_data_as::<F>(), Tolerance::permissive());
        }

        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_golden_context() -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        eprintln!("This test is known to fail on the CUDA backend.\n");

        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/test.wav");
        let expected_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/test.json");
        let sample_rate = 16000;

        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let vad: SileroVad<B> = SileroVadCollection::load_pretrained(&device)?
            .try_branch(sample_rate)?
            .clone();

        let mut wav_vec = load_audio_mono_sr(wav_path, sample_rate)?;

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
        let expected: Vec<f32> = serde_json::from_reader(std::fs::File::open(expected_path)?)?;
        let expected: TensorData = TensorData::from(expected.as_slice());

        chunk_probs.assert_approx_eq(&expected, Tolerance::<f32>::default());

        Ok(())
    }
}
