//! bunsen's Voxtral Realtime against onnxruntime's run of the published
//! graphs, on the same synthetic input.

use bunsen::{
    kits::speech::voxtral::blocks::VoxtralRealtimeMeta,
    ops::transformers::attention::KVCacheModel,
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;

/// The export's decoder is baked at the published delay, 6 positions.
const EXPORT_DELAY: usize = 6;

mod voxtral_mini_4b_realtime_2602 {
    use super::*;

    /// bunsen's audio encoder and projector compute what the published encoder
    /// graph computes, over the whole of both of its calls (the second, through
    /// the first's caches, is the same frames a whole pass reaches later); its
    /// token embedding and text model compute what the embedding and decoder
    /// graphs compute, a prefill then one cached step, choosing the same
    /// tokens.
    #[test]
    #[serial_test::serial]
    fn matches_onnxruntime() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let fixture = onnx_reference();
        let bundle = load_bundle(&device);
        let model = &bundle.model;
        assert_eq!(bundle.layout.delay_positions(), EXPORT_DELAY);

        let e = &fixture.encoder;
        let features = Tensor::cat(
            vec![
                synthetic_features(e.frames, 0, &device),
                synthetic_features(e.step_frames, e.frames, &device),
            ],
            2,
        );
        let audio = model.audio_embeds(features);
        let first = e.frames / model.mel_frames_per_position();
        let error = e.embeds.assert_matches(
            "encoder embeds",
            audio.clone().slice_dim(1, 0..first),
            1e-2,
            2e-3,
        );
        eprintln!("encoder embeds: {error:?}");
        let error = e.step_embeds.assert_matches(
            "encoder step embeds",
            audio.slice_dim(1, first..),
            1e-2,
            2e-3,
        );
        eprintln!("encoder step embeds: {error:?}");

        let lm = &model.language_model;
        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(fixture.embed.ids.clone(), [1, fixture.embed.ids.len()]),
            &device,
        );
        fixture
            .embed
            .embeds
            .assert_matches("token embeds", lm.embed(ids), 1e-6, 1e-6);

        let d = &fixture.decoder;
        let r_emb = lm.rotary(d.prefill + 1, &device);
        let cond = lm.delay_condition(EXPORT_DELAY, &device);
        let mut cache = lm.kv_cache_config().with_seq_len(d.prefill + 1).init();
        let logits = lm.logits(lm.forward_hidden(
            synthetic_embeds(d.prefill, 0, &device),
            &r_emb,
            &cond,
            Some(&mut cache),
        ));
        let argmax: Vec<i64> = logits
            .clone()
            .argmax(2)
            .into_data()
            .try_into_vec_as::<i64>()
            .unwrap();
        assert_eq!(argmax, d.argmax, "prefill argmax");
        let error = d
            .logits
            .assert_matches("prefill logits", logits, 1e-2, 2e-3);
        eprintln!("prefill logits: {error:?}");

        let logits = lm.logits(lm.forward_hidden(
            synthetic_embeds(1, d.prefill, &device),
            &r_emb,
            &cond,
            Some(&mut cache),
        ));
        assert_eq!(
            logits.clone().argmax(2).into_scalar::<i64>(),
            d.step_argmax,
            "step argmax"
        );
        let error = d
            .step_logits
            .assert_matches("step logits", logits, 1e-2, 2e-3);
        eprintln!("step logits: {error:?}");
    }
}
