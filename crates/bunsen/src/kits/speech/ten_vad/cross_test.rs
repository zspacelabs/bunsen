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
            context::{
                FEATURE_EPS,
                FEATURE_MEANS,
                FEATURE_STDS,
                N_MELS,
                TenVadFeatureConfig,
                TenVadFeatureMeta,
                TenVadPitchEstimator,
                TenVadPitchSourceInit,
                tensor::hybrid::HybridPitchInit,
            },
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

    /// Drives the whole golden fixture through a pitch source and recovers the
    /// per-hop pitch in Hz, alongside the reference values to compare against.
    ///
    /// `testdata/ten/pitch.json` holds one pitch estimate per 256-sample hop
    /// over the 60 s fixture, produced by driving the reference `AUP_PE_proc`
    /// (`src/pitch_est.cc`) from the reference `AUP_Analyzer` STFT — the same
    /// wiring `AUP_Aed_runOneFrm` uses, so the estimator sees the raw hop and
    /// the un-normalized bin powers exactly as it does in the C driver.
    fn golden_pitch_hz<I>(pitch: I) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>>
    where
        I: TenVadPitchSourceInit<PerformanceBackend>,
    {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let cfg = TenVadFeatureConfig::new();
        let hop_size = cfg.hop_size();
        let sample_rate = cfg.sample_rate();

        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.wav");
        let golden_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ten/pitch.json");

        let (_, wav_vec) = load_audio_mono_sr(wav_path, sample_rate)?;
        let expected: Vec<f32> = serde_json::from_reader(
            std::fs::File::open(golden_path).map_err(BunsenError::external)?,
        )
        .map_err(BunsenError::external)?;

        let steps = expected.len();
        assert!(
            wav_vec.len() >= steps * hop_size,
            "fixture is too short for {steps} hops: {} samples",
            wav_vec.len(),
        );

        // [steps, batch=1, hop_size]
        let hop_seq: Tensor<B, 3> =
            Tensor::<B, 1>::from_floats(&wav_vec[..steps * hop_size], &device)
                .reshape([steps, 1, hop_size]);

        let mut ctx = cfg.try_init_context::<B, _>(1, pitch, &device)?;

        // [steps, batch=1, n_freq]
        let feats = ctx.forward_sequence(hop_seq);
        let flat: Vec<f32> = feats.to_data_as::<F>().to_vec_as::<f32>().ok_or_panic();

        // Undo the standardization to recover Hz, which is what the reference
        // reports and what stays legible in a failure message.
        let n_freq = N_MELS + 1;
        let scale = FEATURE_STDS[N_MELS] + FEATURE_EPS;
        let got: Vec<f32> = (0..steps)
            .map(|t| flat[t * n_freq + N_MELS] * scale + FEATURE_MEANS[N_MELS])
            .collect();

        Ok((got, expected))
    }

    /// Asserts a recovered pitch track against the C reference.
    ///
    /// The two arms reach their bin powers through different FFTs, so they are
    /// never bit-identical. Two things are asserted instead:
    ///
    /// * the **voicing decision** agrees on every frame — the discrete output,
    ///   and the one a porting error would flip; and
    /// * where both call a frame voiced, the estimates agree within
    ///   `rel_bound`.
    fn assert_golden_pitch(
        got: &[f32],
        expected: &[f32],
        rel_bound: f32,
    ) {
        let steps = expected.len();
        let mut voiced_frames = 0usize;
        for (t, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g > 0.0,
                e > 0.0,
                "frame {t}: voicing disagrees, got {g} Hz vs reference {e} Hz",
            );
            if e > 0.0 {
                voiced_frames += 1;
                let rel = (g - e).abs() / e;
                assert!(
                    rel < rel_bound,
                    "frame {t}: {g} Hz vs reference {e} Hz (rel err {rel})",
                );
            }
        }

        // Guard the guard: a golden of all-unvoiced frames would pass the loop
        // above without exercising the estimator at all.
        assert!(
            voiced_frames * 2 > steps,
            "fixture should be mostly voiced, got {voiced_frames} of {steps}",
        );
    }

    /// Pins the whole driver against the reference implementation.
    ///
    /// `testdata/ten/probs.json` holds one speech probability per hop over the
    /// 60 s fixture, produced by driving the shipped `libten_vad.so` through
    /// the reference Python binding -- the same `ten_vad_process` path every
    /// other binding takes. See `testdata/ten/README.md` for the recipe.
    ///
    /// Unlike the other tests here, this one is end to end: the reference's own
    /// front end and its own inference engine, against bunsen's front end and
    /// its burn port of the graph. Nothing is shared between the two arms
    /// except the audio.
    ///
    /// This form is capped short of the periodic-reset boundary; see
    /// [`test_reference_probability_golden_full`] for the whole fixture.
    #[test]
    #[serial_test::serial]
    fn test_reference_probability_golden() -> Result<(), Box<dyn std::error::Error>> {
        // Capped like the neighbouring cross test: this arm runs the model once
        // per hop on the device, since the LSTM state threads through the loop.
        // Raising the cap costs patience, not correctness.
        run_probability_golden(Some(400))
    }

    /// The probability golden over the *whole* 60 s fixture.
    ///
    /// Ignored by default because it runs for over a quarter of an hour -- the
    /// per-hop cost of [`TenVad::context_forward_sequence`], not anything this
    /// test does. Run it explicitly with:
    ///
    /// ```text
    /// cargo test -p bunsen --lib --features wgpu -- \
    ///     test_reference_probability_golden_full --ignored --exact
    /// ```
    ///
    /// It is worth the wait for one reason: the fixture is 3750 hops, which is
    /// **exactly two [`RESET_FRAMES`] periods**, so this is the only test that
    /// can show the periodic LSTM reset fires on the same hop as the
    /// reference's. A reset at the wrong hop -- or a missing one -- shows up
    /// as a divergence immediately after hop 1875. The capped test above
    /// never reaches the boundary.
    ///
    /// [`RESET_FRAMES`]: crate::kits::speech::ten_vad::context::coeff::RESET_FRAMES
    #[test]
    #[ignore = "runs the full 3750-hop fixture; see the rustdoc"]
    #[serial_test::serial]
    fn test_reference_probability_golden_full() -> Result<(), Box<dyn std::error::Error>> {
        run_probability_golden(None)
    }

    /// Drives the probability golden, optionally capped to `steps_cap` hops.
    fn run_probability_golden(steps_cap: Option<usize>) -> Result<(), Box<dyn std::error::Error>> {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let vad: TenVad<B> = TenVad::load_pretrained(&device)?;
        let cfg = TenVadContextConfig::new();

        let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/silero/test.wav");
        let golden_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/ten/probs.json");

        let (_, wav_vec) = load_audio_mono_sr(wav_path, cfg.sample_rate())?;
        let expected: Vec<f32> = serde_json::from_reader(
            std::fs::File::open(golden_path).map_err(BunsenError::external)?,
        )
        .map_err(BunsenError::external)?;

        let steps = steps_cap.unwrap_or(usize::MAX).min(expected.len());
        let samples = steps * cfg.hop_size();
        assert!(
            wav_vec.len() >= samples,
            "fixture too short for {steps} hops"
        );

        let mut ctx = vad.init_context(&cfg, &device)?;
        let probs = vad.context_forward_audio(&wav_vec[..samples], &mut ctx)?;
        let got: Vec<f32> = probs.to_data_as::<F>().to_vec_as::<f32>().ok_or_panic();

        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        let mut sum = 0.0f64;
        let mut decisions = 0usize;
        for (t, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let d = (g - e).abs();
            sum += d as f64;
            if d > worst {
                worst = d;
                worst_at = t;
            }
            if (g >= 0.5) == (e >= 0.5) {
                decisions += 1;
            }
        }
        let mean = sum / steps as f64;
        let agreement = decisions as f64 / steps as f64;

        eprintln!(
            "reference probability golden: mean |diff| = {mean:.3e}, worst = {worst:.3e} \
             at hop {worst_at} (got {}, want {}), decisions agree {:.3}%",
            got[worst_at],
            expected[worst_at],
            100.0 * agreement,
        );

        // Measured, not guessed: two independent front ends and two independent
        // inference engines land within 3e-5 of each other on average, and never
        // disagree on the decision. The bounds sit an order of magnitude above
        // that, so ordinary backend drift passes and a real regression does not.
        assert_eq!(
            decisions,
            steps,
            "speech/no-speech decisions disagree on {} of {steps} hops",
            steps - decisions,
        );
        assert!(
            worst < 5e-3,
            "worst probability error {worst:.3e} at hop {worst_at} is too large"
        );
        assert!(
            mean < 1e-3,
            "mean probability error {mean:.3e} is too large"
        );

        Ok(())
    }

    /// Pins the ported host pitch estimator against the ten-vad C reference.
    ///
    /// This is the anchor of the whole chain: every device stage is validated
    /// differentially against the host estimator, and the host estimator is
    /// validated here. It covers feature `40`; the other 40 are pinned by
    /// [`TenVadFeatureContext`]'s own tests against an independent host
    /// implementation of the mel path.
    ///
    /// [`TenVadFeatureContext`]: crate::kits::speech::ten_vad::context::TenVadFeatureContext
    #[test]
    #[serial_test::serial]
    fn test_pitch_estimator_reference_golden() -> Result<(), Box<dyn std::error::Error>> {
        let (got, expected) = golden_pitch_hz(TenVadPitchEstimator::new())?;
        assert_golden_pitch(&got, &expected, 1e-4);
        Ok(())
    }

    /// The go/no-go gate for the device-side port: stage 1 on the device, the
    /// remaining three stages on the host, measured against the same C golden.
    ///
    /// The stage-level differential tests establish that the device pre-filter
    /// reproduces the host one to a tolerance. They cannot establish that the
    /// tolerance survives the tracker's `argmax` and its voicing threshold,
    /// which are discrete — a single `argmax` step of ±1 moves the reported
    /// pitch by roughly half a percent, some fifty times the bound above. This
    /// test answers that, by differing from the pinned pipeline in exactly one
    /// stage.
    ///
    /// The value bound is looser than the host arm's because the device
    /// contracts the band projections in `f32` where the host accumulates the
    /// autocorrelation in `f64`. The **voicing** bound is not loosened: that is
    /// the assertion with teeth.
    #[test]
    #[serial_test::serial]
    fn test_tensor_prefilter_hybrid_reference_golden() -> Result<(), Box<dyn std::error::Error>> {
        let (got, expected) = golden_pitch_hz(HybridPitchInit::new())?;
        assert_golden_pitch(&got, &expected, 1e-3);
        Ok(())
    }
}
