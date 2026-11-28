use bunsen::kits::{
    speech::{
        asr::whisper::driver::WhisperTask,
        support::testing::wer::text_error_rate,
    },
    support::tokens::Detokenizer,
};

use super::{
    super::*,
    *,
};

/// The committed references must decode to their own committed text, every
/// variant.
///
/// Needs no model, so it runs in an ordinary `cargo test`, and it is what
/// checks the bundled vocabulary and bunsen's rank parser end to end, since a
/// wrong table would move the decoded text away from what `openai-whisper`
/// printed. With timestamps kept, it also pins the spelling of every special
/// token against `tiktoken`'s, and the segment times against the timestamp
/// arithmetic the stream driver will use.
#[test]
fn test_reference_tokens_decode_to_reference_text() {
    let table = vocab();
    assert_eq!(
        table.detokenizer.vocab_size(),
        N_VOCAB,
        "the vocabulary does not cover the multilingual model",
    );

    for fixture in FIXTURES {
        let name = fixture.name;
        let reference = Reference::load(name);
        assert_eq!(
            reference.windows.len(),
            fixture.windows,
            "{name}: reference covers {} window(s), audio is {}",
            reference.windows.len(),
            fixture.windows,
        );

        // The layout the decodes were made with is the layout bunsen derives
        // from the vocabulary size alone.
        let ids = table.policy.ids();
        assert_eq!(reference.vocab_size, N_VOCAB, "{name}: vocabulary size");
        assert_eq!(
            reference.prompt,
            table
                .policy
                .sot_sequence(Some("en"), Some(WhisperTask::Transcribe), true)
                .unwrap(),
            "{name}: prompt",
        );
        assert_eq!(reference.no_timestamps, ids.no_timestamps, "{name}");
        assert_eq!(reference.eot, ids.eot, "{name}");
        assert_eq!(reference.timestamp_begin, ids.timestamp_begin, "{name}");

        for (w, window) in reference.windows.iter().enumerate() {
            assert_eq!(
                table.decode(&window.tokens).trim(),
                window.text.trim(),
                "{name}: greedy window {w} does not decode to its committed text",
            );
        }

        for (w, window) in reference.beam5.windows.iter().enumerate() {
            assert_eq!(
                table.decode(&window.tokens).trim(),
                window.text.trim(),
                "{name}: beam-5 window {w} does not decode to its committed text",
            );
        }

        for (w, window) in reference.with_timestamps.windows.iter().enumerate() {
            assert_eq!(
                table.decode(&window.tokens).trim(),
                window.text.trim(),
                "{name}: timestamped window {w} does not decode to its committed text",
            );
            assert!(
                window
                    .tokens
                    .first()
                    .is_some_and(|&t| table.policy.is_timestamp(t)),
                "{name}: timestamped window {w} does not open with a timestamp",
            );

            // Rendered whole, the specials must spell exactly as tiktoken
            // spells them.
            let rendered = table
                .detokenizer
                .detokenize(&window.tokens)
                .expect("every id is inside the vocabulary");
            assert_eq!(
                Some(rendered),
                window.text_with_timestamps,
                "{name}: timestamped window {w} does not render as tiktoken does",
            );
        }

        let transcribe = &reference.transcribe;
        assert!(!transcribe.segments.is_empty(), "{name}: no segments");
        for seg in &transcribe.segments {
            assert_eq!(
                table.decode(&seg.tokens).trim(),
                seg.text.trim(),
                "{name}: segment at {} does not decode to its committed text",
                seg.start,
            );
            assert_eq!(
                seg.temperature, 0.0,
                "{name}: segment at {} fell back to sampling; the fixture is not deterministic",
                seg.start,
            );

            // A segment's times are its bounding timestamp tokens, offset by
            // where its window was cut: `seek` mel frames, at 100 per second.
            // This is the arithmetic the driver's clock will do.
            let offset = seg.seek as f64 / 100.0;
            let (first, last) = (seg.tokens[0], *seg.tokens.last().unwrap());
            let expect_start = table
                .policy
                .timestamp_seconds(first)
                .map_or(offset, |t| offset + t);
            assert!(
                (seg.start - expect_start).abs() < 1e-6,
                "{name}: segment start {} is not {expect_start}",
                seg.start,
            );
            if let Some(t) = table.policy.timestamp_seconds(last) {
                assert!(
                    (seg.end - (offset + t)).abs() < 1e-6,
                    "{name}: segment end {} is not {}",
                    seg.end,
                    offset + t,
                );
            }
        }
    }
}

/// The references themselves must meet the fixture's accuracy bar.
///
/// Needs no model. If this fails, the threshold is wrong or the transcript
/// is, and either way the fixture is broken before bunsen is involved, so
/// `max_wer` would be measuring nothing. The decode variants a later phase
/// will be judged against are held to it too, so each is known good before
/// its phase starts.
///
/// Two variants are reported rather than gated, and the reasons are worth
/// keeping. A fixed 30 s window decoded *with* timestamps stops at its last
/// timestamp and drops whatever followed it, which is the very loss the seek
/// loop exists to recover. And beam 5 on fixed windows without timestamps is
/// not a mode upstream ever transcribes with, so its accuracy is beside the
/// point; what its phase needs from it is agreement. The `transcribe()`
/// reference, which is a real transcription, is gated like the greedy one.
#[test]
fn test_reference_meets_the_accuracy_bar() {
    for fixture in FIXTURES {
        let reference = Reference::load(fixture.name);
        let want = transcript(fixture.name);

        for (label, got, gated) in [
            ("reference", reference.text(), true),
            (
                "timestamps-reference",
                join_windows(&reference.with_timestamps.windows),
                false,
            ),
            (
                "beam5-reference",
                join_windows(&reference.beam5.windows),
                false,
            ),
            (
                "transcribe-reference",
                reference.transcribe.text.clone(),
                true,
            ),
        ] {
            let wer = text_error_rate(&got, &want);
            eprintln!("{}: {label} WER vs transcript {wer:.4}", fixture.name);

            assert!(
                !gated || wer <= fixture.max_wer,
                "{}",
                report(label, fixture.name, &got, &want, wer, fixture.max_wer),
            );
        }
    }
}

/// The audio must be loadable and correctly shaped, so a broken asset is
/// caught without a model.
#[test]
fn test_speech_fixtures_are_well_formed() {
    for fixture in FIXTURES {
        let wav = samples(fixture.name);

        let windows = wav.len().div_ceil(N_SAMPLES).max(1);
        assert_eq!(
            windows,
            fixture.windows,
            "{}: {} samples is {windows} window(s), expected {}",
            fixture.name,
            wav.len(),
            fixture.windows,
        );

        assert!(
            wav.iter().all(|s| s.is_finite()),
            "{}: decoded to non-finite samples",
            fixture.name,
        );

        assert!(
            !transcript(fixture.name).trim().is_empty(),
            "{}: transcript is empty",
            fixture.name,
        );
    }
}
