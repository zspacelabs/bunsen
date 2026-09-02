use bunsen::{
    kits::speech::whisper::{
        ApplyTimestampRules,
        DecodeConfig,
        GreedyDecodeConfig,
        default_filters,
        driver::{
            MaxSeen,
            Task,
            TimestampHistory,
            WhisperDriverConfig,
        },
        mel_windows,
    },
    prelude::TensorElemOpExt,
    support::testing::{
        PerformanceBackend,
        asr::text_error_rate,
    },
};
use burn::{
    Tensor,
    prelude::{
        Backend,
        Device,
        Int,
        TensorData,
    },
    tensor::{
        Tolerance,
        backend::BackendTypes,
    },
};

use crate::{
    N_FRAMES,
    N_MELS,
    audio::{
        FIXTURES,
        Reference,
        SAMPLE_RATE,
        Vocab,
        bunsen_model,
        clip_mels,
        report,
        report_id_diff,
        samples,
        to_text,
        transcript,
        vocab,
    },
    reference,
};

type B = PerformanceBackend;
type F = <B as BackendTypes>::FloatElem;

/// The prompt and stop token, derived from the layout rather than written
/// down: `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`, then
/// `<|endoftext|>`.
fn decode_config(table: &Vocab) -> GreedyDecodeConfig {
    let prompt = table
        .policy
        .sot_sequence(Some("en"), Some(Task::Transcribe), false)
        .expect("a multilingual layout");
    GreedyDecodeConfig::new(prompt, table.policy.ids().eot)
}

/// Splits log-mels into the encoder's fixed windows, zero-padding the
/// last.
fn windows_of<B: Backend>(
    mels: Tensor<B, 3>,
    device: &Device<B>,
) -> Vec<Tensor<B, 3>> {
    let frames = mels.dims()[2];
    (0..frames.div_ceil(N_FRAMES))
        .map(|w| {
            let start = (w * N_FRAMES) as isize;
            let end = ((w + 1) * N_FRAMES).min(frames) as isize;
            let win = mels.clone().slice_dim(2, start..end);

            let have = win.dims()[2];
            if have < N_FRAMES {
                Tensor::cat(
                    vec![win, Tensor::zeros([1, N_MELS, N_FRAMES - have], device)],
                    2,
                )
            } else {
                win
            }
        })
        .collect()
}

/// Greedily decodes one window against the reference decoder.
///
/// The export is KV-cache-free, so the whole prefix is re-fed every
/// step — which is slow, and exactly why bunsen has a cache.
fn greedy_reference<B: Backend>(
    decoder: &reference::DecoderModel<B>,
    xa: Tensor<B, 3>,
    device: &Device<B>,
    config: &GreedyDecodeConfig,
) -> Vec<i64> {
    let mut prefix = config.prompt.clone();
    let mut out = Vec::new();

    for _ in 0..224 {
        let len = prefix.len();
        let tokens: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(prefix.clone(), [1, len]), device);

        let picked: Vec<i64> = decoder
            .forward(tokens, xa.clone())
            .0
            .slice_dim(1, (len - 1) as isize..len as isize)
            .argmax(2)
            .into_data()
            .convert::<i64>()
            .to_vec()
            .unwrap();

        if picked[0] == config.eot_token {
            break;
        }
        out.push(picked[0]);
        prefix.push(picked[0]);
    }

    out
}

/// **The agreement gate.** bunsen must decode what `openai-whisper`
/// decodes, on the same audio and the same windowing.
#[test]
fn test_bunsen_agrees_with_openai_reference() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let model = bunsen_model::<B>(&device);
    let config = decode_config(&table);

    for fixture in FIXTURES {
        let mels = clip_mels(fixture.name, &device);
        let reference = Reference::load(fixture.name);
        let mine = model.decode_chunked(mels, &config);
        report_id_diff(
            "openai-reference",
            fixture.name,
            &mine,
            &reference.window_tokens(),
        );

        let got = to_text(&table, &mine);
        let want = reference.text();
        let wer = text_error_rate(&got, &want);

        eprintln!("{}: bunsen WER vs openai-whisper {wer:.4}", fixture.name);
        assert!(
            wer <= fixture.max_reference_wer,
            "{}",
            report(
                "openai-reference",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_reference_wer,
            ),
        );
    }
}

/// The encoder comparison, on **real** log-mels rather than synthetic
/// ones.
///
/// Real speech is not the same test: `synthetic_mels` is a bounded
/// sawtooth, while a log-mel spectrogram has the dynamic range and the
/// near-silent bins a mis-scaled layer norm would show up in.
#[test]
fn test_onnx_encoder_matches_bunsen_on_real_audio() {
    let device: Device<B> = Default::default();
    let reference = reference::EncoderModel::<B>::load_pretrained(&device);
    let ours = bunsen_model(&device);

    for fixture in FIXTURES {
        let mels = clip_mels(fixture.name, &device);
        for (w, window) in windows_of(mels, &device).into_iter().enumerate() {
            let theirs = reference.forward(window.clone());
            let mine = ours.forward_encoder(window);

            assert_eq!(mine.dims(), theirs.dims(), "{} window {w}", fixture.name);
            mine.to_data_as::<F>()
                .assert_approx_eq::<F>(&theirs.to_data_as::<F>(), Tolerance::rel_abs(1e-1, 2e-2));
        }
    }
}

/// **The ONNX reference transcribes the clip**, independent of bunsen:
/// the ONNX encoder feeds the ONNX decoder.
#[test]
fn test_onnx_reference_transcribes_real_audio() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let reference_enc = reference::EncoderModel::<B>::load_pretrained(&device);
    let reference_dec = reference::DecoderModel::<B>::load_pretrained(&device);
    let config = decode_config(&table);

    for fixture in FIXTURES {
        let mels = clip_mels(fixture.name, &device);
        let ids: Vec<Vec<i64>> = windows_of(mels, &device)
            .into_iter()
            .map(|window| {
                let xa = reference_enc.forward(window);
                greedy_reference(&reference_dec, xa, &device, &config)
            })
            .collect();

        let got = to_text(&table, &ids);
        let want = transcript(fixture.name);
        let wer = text_error_rate(&got, &want);

        eprintln!("{}: onnx WER vs transcript {wer:.4}", fixture.name);
        assert!(
            wer <= fixture.max_wer,
            "{}",
            report(
                "onnx-transcript",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_wer
            ),
        );
    }
}

/// **The agreement gate, end to end.** bunsen and the ONNX reference
/// must transcribe the same clip to the same words.
///
/// This is the one that catches what the staged comparisons let
/// through: each stage can agree inside tolerance while the
/// composition diverges, because a greedy argmax turns a small
/// numerical difference into a different word.
#[test]
fn test_onnx_reference_and_bunsen_transcribe_alike() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let reference_enc = reference::EncoderModel::<B>::load_pretrained(&device);
    let reference_dec = reference::DecoderModel::<B>::load_pretrained(&device);
    let ours = bunsen_model(&device);
    let config = decode_config(&table);

    for fixture in FIXTURES {
        let mels = clip_mels(fixture.name, &device);

        let mut theirs = Vec::new();
        let mut mine = Vec::new();
        for window in windows_of(mels, &device) {
            let xa = reference_enc.forward(window.clone());
            theirs.push(greedy_reference(&reference_dec, xa, &device, &config));
            mine.push(ours.decode_window(window, &config));
        }
        report_id_diff("onnx-reference", fixture.name, &mine, &theirs);

        let (want, got) = (to_text(&table, &theirs), to_text(&table, &mine));
        let wer = text_error_rate(&got, &want);

        eprintln!("{}: bunsen WER vs onnx reference {wer:.4}", fixture.name);
        assert!(
            wer <= fixture.max_reference_wer,
            "{}",
            report(
                "onnx-reference",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_reference_wer,
            ),
        );
    }
}

/// **The accuracy gate.** Real audio, real weights, judged against the
/// ground-truth transcript.
#[test]
fn test_bunsen_accuracy_against_transcript() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let model = bunsen_model::<B>(&device);
    let config = decode_config(&table);

    for fixture in FIXTURES {
        let mels = clip_mels(fixture.name, &device);
        let per_window = model.decode_chunked(mels, &config);
        assert_eq!(per_window.len(), fixture.windows, "{}", fixture.name);

        let got = to_text(&table, &per_window);
        let want = transcript(fixture.name);
        let wer = text_error_rate(&got, &want);

        eprintln!("{}: bunsen WER vs transcript {wer:.4}", fixture.name);
        assert!(
            wer <= fixture.max_wer,
            "{}",
            report(
                "transcript",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_wer
            ),
        );
    }
}

/// One window's decode under `config` and `filters`, per window of the clip.
fn decode_filtered(
    model: &bunsen::kits::speech::whisper::Whisper<B>,
    mels: Tensor<B, 3>,
    config: &DecodeConfig,
    filters: &[std::sync::Arc<dyn bunsen::kits::speech::whisper::LogitFilter<B>>],
) -> Vec<Vec<i64>> {
    mel_windows(mels, N_FRAMES)
        .into_iter()
        .map(|window| {
            model
                .decode_windows(window, config, filters)
                .pop()
                .expect("one row in, one row out")
        })
        .collect()
}

/// **The filtered gate.** `openai-whisper` decodes under two filters by
/// default, `suppress_blank` and `suppress_tokens=-1`; bunsen under the
/// same filters, derived from the rank file alone, must still decode what
/// the reference decodes.
#[test]
fn test_bunsen_agrees_with_openai_reference_under_default_filters() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let model = bunsen_model::<B>(&device);
    let config = DecodeConfig::from(&decode_config(&table));
    let filters = default_filters::<B>(&table.ranks, table.policy.ids());

    for fixture in FIXTURES {
        let reference = Reference::load(fixture.name);
        let mine = decode_filtered(&model, clip_mels(fixture.name, &device), &config, &filters);
        report_id_diff(
            "openai-reference-filtered",
            fixture.name,
            &mine,
            &reference.window_tokens(),
        );

        let got = to_text(&table, &mine);
        let want = reference.text();
        let wer = text_error_rate(&got, &want);

        eprintln!(
            "{}: bunsen (filtered) WER vs openai-whisper {wer:.4}",
            fixture.name
        );
        assert!(
            wer <= fixture.max_reference_wer,
            "{}",
            report(
                "openai-reference-filtered",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_reference_wer,
            ),
        );
    }
}

/// **The beam gate** (I7 at width five). With upstream's default filters
/// and five beams, bunsen must decode what `openai-whisper` decodes with
/// `beam_size=5`: the same candidates, deduplicated, ranked and finished
/// the same way.
#[test]
fn test_bunsen_beam_agrees_with_openai_reference() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let model = bunsen_model::<B>(&device);
    let config = DecodeConfig::from(&decode_config(&table)).with_beam_size(5);
    let filters = default_filters::<B>(&table.ranks, table.policy.ids());

    for fixture in FIXTURES {
        let reference = Reference::load(fixture.name);
        let mine = decode_filtered(&model, clip_mels(fixture.name, &device), &config, &filters);
        report_id_diff(
            "openai-beam5",
            fixture.name,
            &mine,
            &reference.beam5_tokens(),
        );

        let got = to_text(&table, &mine);
        let want = reference.beam5_text();
        let wer = text_error_rate(&got, &want);

        eprintln!(
            "{}: bunsen beam-5 WER vs openai-whisper beam-5 {wer:.4}",
            fixture.name
        );
        assert!(
            wer <= fixture.max_reference_wer,
            "{}",
            report(
                "openai-beam5",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_reference_wer,
            ),
        );
    }
}

/// **The timestamp gate.** Fixed windows prompted for timestamps, under
/// upstream's default filters and its timestamp rules, decode to the
/// timestamped reference: the rules' every clause, on real logits.
#[test]
fn test_bunsen_timestamps_agree_with_openai_reference() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let model = bunsen_model::<B>(&device);
    let ids = table.policy.ids();
    let prompt = table
        .policy
        .sot_sequence(Some("en"), Some(Task::Transcribe), true)
        .expect("a multilingual layout");
    let config = DecodeConfig::new(prompt, ids.eot);
    let mut filters = default_filters::<B>(&table.ranks, ids);
    filters.push(std::sync::Arc::new(ApplyTimestampRules::new(ids, Some(50))));

    for fixture in FIXTURES {
        let reference = Reference::load(fixture.name);
        let mine = decode_filtered(&model, clip_mels(fixture.name, &device), &config, &filters);
        report_id_diff(
            "openai-timestamps",
            fixture.name,
            &mine,
            &reference.timestamped_tokens(),
        );

        let got = to_text(&table, &mine);
        let want = reference.timestamped_text();
        let wer = text_error_rate(&got, &want);

        eprintln!(
            "{}: bunsen (timestamps) WER vs openai-whisper {wer:.4}",
            fixture.name
        );
        assert!(
            wer <= fixture.max_reference_wer,
            "{}",
            report(
                "openai-timestamps",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_reference_wer,
            ),
        );
    }
}

/// **The seek-loop gate.** The driver, offline with timestamps and the
/// prompt carry, over the whole clip in one push: the segments
/// `transcribe()` produced, with their times, through a stream clock from
/// zero.
#[test]
fn test_bunsen_driver_transcribes_like_openai() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let driver = WhisperDriverConfig::new()
        .with_language(Some("en".to_string()))
        .with_timestamps(true)
        .init_with_policy(bunsen_model::<B>(&device), table.policy.clone(), &device)
        .expect("a multilingual layout with a language")
        .with_logit_filters(default_filters::<B>(&table.ranks, table.policy.ids()));

    for fixture in FIXTURES {
        let reference = Reference::load(fixture.name);
        let mut ctx = driver
            .new_context(TimestampHistory::uniform(SAMPLE_RATE), MaxSeen::new())
            .expect("a stream at the model's rate");
        let mut emissions = ctx.push(&samples(fixture.name)).expect("the push decodes");
        emissions.extend(ctx.flush().expect("the flush decodes"));

        let mine: Vec<Vec<i64>> = emissions
            .iter()
            .map(|e| e.segment().tokens.clone())
            .collect();
        let theirs: Vec<Vec<i64>> = reference
            .transcribe
            .segments
            .iter()
            .map(|s| s.tokens.clone())
            .collect();
        report_id_diff("openai-transcribe", fixture.name, &mine, &theirs);
        for (i, (e, s)) in emissions
            .iter()
            .zip(&reference.transcribe.segments)
            .enumerate()
        {
            let (a, b) = (e.segment().start, e.segment().end);
            eprintln!(
                "{}: segment {i}: bunsen {a:.2}-{b:.2} vs openai {:.2}-{:.2}",
                fixture.name, s.start, s.end
            );
        }

        let got = to_text(&table, &mine);
        let want = reference.transcribe.text.clone();
        let wer = text_error_rate(&got, &want);
        eprintln!(
            "{}: bunsen (driver) WER vs openai transcribe() {wer:.4}",
            fixture.name
        );
        assert!(
            wer <= fixture.max_reference_wer,
            "{}",
            report(
                "openai-transcribe",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_reference_wer,
            ),
        );

        assert_eq!(
            emissions.len(),
            reference.transcribe.segments.len(),
            "{}: segment count",
            fixture.name
        );
        for (i, (e, s)) in emissions
            .iter()
            .zip(&reference.transcribe.segments)
            .enumerate()
        {
            assert!(e.is_committed(), "offline commits everything");
            assert!(
                (e.segment().start - s.start).abs() < 1e-6
                    && (e.segment().end - s.end).abs() < 1e-6,
                "{}: segment {i} times: {:.4}-{:.4} vs {:.4}-{:.4}",
                fixture.name,
                e.segment().start,
                e.segment().end,
                s.start,
                s.end,
            );
        }
    }
}

/// **Language detection.** One step over `<|startoftranscript|>` with only
/// the language block to choose from says the clip is English.
#[test]
fn test_bunsen_detects_the_language() {
    let device: Device<B> = Default::default();
    let table = vocab();
    let model = bunsen_model::<B>(&device);

    for fixture in FIXTURES {
        let windows = mel_windows(clip_mels(fixture.name, &device), N_FRAMES);
        let xa = model.forward_encoder(windows[0].clone());
        let token = model.detect_language(xa, table.policy.ids())[0];
        assert_eq!(
            table.policy.language_code(token),
            Some("en"),
            "{}",
            fixture.name
        );
    }
}
