#[cfg(test)]
mod tests {
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
            TenVadContextConfig,
            TenVadContextMeta,
            TenVadMeta,
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

        let ref_vad: ReferenceModel<B> = ReferenceModel::load_pretrained(&device);

        let vad: TenVad<B> = TenVad::load_pretrained(&device).unwrap();

        // The leading axis is the graph's sequence axis, not a stream batch;
        // see `TenVad::forward`. Both are 1 here.
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

    /// Drives real audio end-to-end through the context driver, and pins
    /// three independent arms against each other:
    ///
    /// 1. [`TenVad::context_forward_sequence`] — the batched front end,
    /// 2. [`TenVad::context_forward`] — the same work, one hop at a time,
    /// 3. the ONNX reference graph, stepped by hand over the *driver's own*
    ///    feature stacks with the recurrence threaded externally.
    ///
    /// Arms 1 and 2 pin that the sequence form really is "iterating the
    /// non-sequence form"; arm 3 pins that the widened stack the driver builds
    /// is what the reference model expects, and that bunsen's port of the
    /// network agrees with the graph over a whole stream rather than a single
    /// random frame.
    #[test]
    #[serial_test::serial]
    fn test_reference_model_context_forward_sequence_cross_test()
    -> Result<(), Box<dyn std::error::Error>> {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();

        let vad: TenVad<B> = TenVad::load_pretrained(&device)?;
        let ref_vad: ReferenceModel<B> = ReferenceModel::load_pretrained(&device);

        let cfg = TenVadContextConfig::new();
        let sample_rate = cfg.sample_rate();
        let hop_size = cfg.hop_size();

        // A bounded slice of the shared 16 kHz fixture: the reference arm runs
        // one ONNX call per hop, and the full file is 3750 of them.
        const STEPS: usize = 400;

        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.wav");
        let (_, wav_vec) = load_audio_mono_sr(wav_path, sample_rate)?;
        assert!(
            wav_vec.len() >= STEPS * hop_size,
            "fixture is too short: {} samples",
            wav_vec.len(),
        );

        // [steps, batch=1, hop_size]
        let hop_seq: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(&wav_vec[..STEPS * hop_size], &device)
                .reshape([STEPS, 1, hop_size]);

        // Arm 1: the batched sequence path.
        let mut seq_ctx = vad.init_context(&cfg, &device)?;
        let seq_probs = vad.context_forward_sequence(hop_seq.clone(), &mut seq_ctx);
        assert_eq!(seq_probs.dims(), [STEPS, 1]);

        // Arm 2: the same stream, one hop at a time.
        let mut step_ctx = vad.init_context(&cfg, &device)?;
        let mut step_probs = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            let hop = hop_seq.clone().select_dim::<2>(0, step);
            step_probs.push(vad.context_forward(hop, &mut step_ctx));
        }
        let step_probs: Tensor<B, 2> = Tensor::stack(step_probs, 0);

        // Arm 3: the reference graph, fed the driver's own feature stacks.
        let mut ref_ctx = vad.init_context(&cfg, &device)?;
        let mut ref_state1 = ExtLstmState::initial([1, vad.d_hidden()], &device);
        let mut ref_state2 = ExtLstmState::initial([1, vad.d_hidden()], &device);
        let mut ref_probs = Vec::with_capacity(STEPS);

        for step in 0..STEPS {
            let hop = hop_seq.clone().select_dim::<2>(0, step);

            // [batch, n_freq] -> [batch, d_ctx, n_freq]
            let feats = ref_ctx.features.forward(hop);
            let stack = ref_ctx.push_features(feats);

            // The graph takes `(features, h1, c1, h2, c2)` positionally.
            let (prob, h1, c1, h2, c2) = ref_vad.forward(
                stack,
                ref_state1.hidden.clone(),
                ref_state1.cell.clone(),
                ref_state2.hidden.clone(),
                ref_state2.cell.clone(),
            );
            ref_state1 = ExtLstmState::new(c1, h1);
            ref_state2 = ExtLstmState::new(c2, h2);

            // [1, 1, 1] -> [1]
            ref_probs.push(prob.reshape([1]));
        }
        let ref_probs: Tensor<B, 2> = Tensor::stack(ref_probs, 0);

        // The sequence and stepwise arms run identical arithmetic in a
        // different order.
        let probs = seq_probs.clone();
        seq_probs
            .clone()
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&step_probs.to_data_as::<F>(), Tolerance::permissive());

        // The reference arm crosses an independently generated graph.
        seq_probs
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_probs.to_data_as::<F>(), Tolerance::permissive());

        // The carried recurrence must land in the same place, or a resumed
        // stream would diverge from the reference after the first chunk.
        for (ours, theirs) in [
            (&seq_ctx.state1, &ref_state1),
            (&seq_ctx.state2, &ref_state2),
        ] {
            ours.hidden
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&theirs.hidden.to_data_as::<F>(), Tolerance::permissive());
            ours.cell
                .to_data_as::<F>()
                .assert_approx_eq::<F>(&theirs.cell.to_data_as::<F>(), Tolerance::permissive());
        }

        // And the driver's feature history must match what the reference arm
        // was actually fed.
        seq_ctx
            .stack
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&ref_ctx.stack.to_data_as::<F>(), Tolerance::permissive());

        // All three arms share the driver's features, so agreement alone
        // would also hold for a driver emitting constants. Check the output
        // is actually tracking the audio: the fixture is continuous speech
        // entered from a zeroed context, so the probabilities must be valid,
        // must span a real range, and must sit mostly high.
        let host: Vec<f32> = probs.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        assert!(
            host.iter().all(|&p| (0.0..=1.0).contains(&p)),
            "probabilities outside [0, 1]",
        );

        let min = host.iter().copied().fold(f32::INFINITY, f32::min);
        let max = host.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 0.2,
            "probabilities are nearly constant ({min} ..= {max}); \
             the front end is not tracking the audio",
        );

        let voiced = host.iter().filter(|&&p| p > 0.5).count();
        assert!(
            voiced * 2 > host.len(),
            "only {voiced}/{} frames read as speech on a speech fixture",
            host.len(),
        );

        Ok(())
    }
}
