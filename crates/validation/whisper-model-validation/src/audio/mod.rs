//! End-to-end validation over the committed speech fixtures.
//!
//! [`staged`](super::staged) feeds each stage synthetic input, which isolates
//! it but says nothing about the pipeline a user drives. This runs the whole
//! thing — mp3, mel front end, encoder, KV-cached decoder, greedy search —
//! over `testdata/`, and judges it the way a transcription is judged: **word
//! error rate against a ground-truth transcript**.
//!
//! ## What is asserted
//!
//! The transcript is the authority. Every gate is on text, and every gate is
//! tunable per fixture:
//!
//! * **Accuracy** — WER against `{name}.txt` must be at or under
//!   [`Fixture::max_wer`]. This is what says the pipeline works.
//! * **Agreement** — WER against `{name}.reference.json`, what `openai-whisper`
//!   itself decodes, must be at or under [`Fixture::max_reference_wer`]. This
//!   is what says bunsen agrees with the implementation it transliterates.
//!
//! Nothing compares token ids. A greedy decode argmaxes over 51865 logits at
//! every step, so a backend differing in the last few digits can flip a token
//! and cascade — while the *text* barely moves.
//!
//! ## What runs when
//!
//! The fixture-integrity checks need no model at all and so run in an ordinary
//! `cargo test`. Everything that loads weights is behind `download`, which is
//! also what makes `build.rs` fetch the checkpoint — so within this crate a
//! gated test never has to skip.

use bunsen::{
    burner::module::{
        DTypeMapper,
        ModuleInit,
    },
    kits::{
        speech::whisper::{
            Task,
            TiktokenRanks,
            TokenPolicy,
            Whisper,
            mel_options,
            package_mels,
            pretrained::bundled,
            text::detokenizer,
        },
        tokens::{
            Detokenizer,
            WordchipperDetokenizer,
        },
    },
    ops::signal::mels::MelConverter,
    support::testing::asr::text_error_rate,
};
use burn::{
    Tensor,
    module::Module,
    prelude::{
        Backend,
        Device,
        TensorData,
    },
};

use super::*;

/// Measured on `wgpu` against `whisper-base`:
///
/// | | WER |
/// |---|---|
/// | bunsen vs transcript | 0.068 |
/// | `openai-whisper` vs transcript | 0.068 |
/// | ONNX reference vs transcript | 0.068 |
/// | bunsen vs `openai-whisper` | 0.000 |
/// | bunsen vs ONNX reference | 0.000 |
///
/// `max_wer` is set with roughly 1.5x headroom over the measurement. Of the
/// eight word errors, three are the model (`fly` -> `why`, `in this decade` ->
/// `and disdicate`, `not` -> `that`) and two are the normalizer rather than
/// the model (`thirty-five` vs `35`, `we are` vs `we're`) —
/// `normalize_transcript` deliberately does not reconcile number words or
/// contractions.
///
/// `max_reference_wer` is 0.0 because all three implementations reproduce each
/// other token for token on this clip. A backend with a reduced-precision
/// matmul may need it raised; that is what the knob is for.
pub const FIXTURES: &[Fixture] = &[Fixture {
    name: "jfk_moon",
    windows: 2,
    max_wer: 0.10,
    max_reference_wer: 0.0,
}];

/// The rate Whisper's front end is defined at, and the rate fixtures are
/// stored at.
const SAMPLE_RATE: usize = 16_000;

/// Whisper's fixed analysis window: 30 s.
const N_SAMPLES: usize = 30 * SAMPLE_RATE;

fn testdata(rel: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata")).join(rel)
}

/// The committed `openai-whisper` decodes of a fixture, several ways.
///
/// `tools/gen_speech_fixtures.py` says what each is. `windows` is the
/// agreement gate; the rest are the references later phases of the stream
/// driver pin against, stocked here so that no phase invents its own oracle.
/// The token layout the decodes were made with is in the fixture too, so the
/// vocabulary pairing is checked rather than assumed.
#[derive(serde::Deserialize)]
pub struct Reference {
    /// The vocabulary size of the checkpoint the decodes came from.
    vocab_size: usize,
    /// The sot sequence: `<|startoftranscript|> <|en|> <|transcribe|>`.
    prompt: Vec<i64>,
    no_timestamps: i64,
    eot: i64,
    timestamp_begin: i64,
    /// Greedy, without timestamps, per fixed 30 s window.
    windows: Vec<ReferenceWindow>,
    /// The same decode with timestamp tokens on.
    with_timestamps: ReferenceWindows,
    /// Beam 5, without timestamps.
    beam5: ReferenceWindows,
    /// `transcribe()` at temperature 0: the seek loop and its segments.
    transcribe: ReferenceTranscribe,
}

#[derive(serde::Deserialize)]
pub struct ReferenceWindows {
    windows: Vec<ReferenceWindow>,
}

#[derive(serde::Deserialize)]
pub struct ReferenceWindow {
    /// What the decoder emitted, prompt and stop token excluded.
    tokens: Vec<i64>,
    /// `Tokenizer.decode`: timestamps dropped.
    text: String,
    /// `Tokenizer.decode_with_timestamps`, only where timestamps were on.
    #[serde(default)]
    text_with_timestamps: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct ReferenceTranscribe {
    text: String,
    segments: Vec<ReferenceSegment>,
}

#[derive(serde::Deserialize)]
pub struct ReferenceSegment {
    /// The mel frame the window this segment came from was cut at.
    seek: usize,
    start: f64,
    end: f64,
    text: String,
    /// Bounded by its timestamp tokens, where the decode emitted them.
    tokens: Vec<i64>,
    temperature: f64,
}

impl Reference {
    /// Load a reference from a file.
    pub fn load(name: &str) -> Self {
        let path = testdata(&format!("{name}.reference.json"));
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("opening {}: {e}", path.display()));
        serde_json::from_reader(file).expect("the reference failed to parse")
    }

    /// The reference decode as one string, windows joined in order.
    pub fn text(&self) -> String {
        join_windows(&self.windows)
    }

    /// The reference decode's ids, per window.
    pub fn window_tokens(&self) -> Vec<Vec<i64>> {
        self.windows.iter().map(|w| w.tokens.clone()).collect()
    }

    /// The beam-5 reference decode as one string, windows joined in order.
    pub fn beam5_text(&self) -> String {
        join_windows(&self.beam5.windows)
    }

    /// The beam-5 reference decode's ids, per window.
    pub fn beam5_tokens(&self) -> Vec<Vec<i64>> {
        self.beam5
            .windows
            .iter()
            .map(|w| w.tokens.clone())
            .collect()
    }

    /// The timestamped reference decode as one string, windows joined in
    /// order.
    pub fn timestamped_text(&self) -> String {
        join_windows(&self.with_timestamps.windows)
    }

    /// The timestamped reference decode's ids, per window, timestamp tokens
    /// included.
    pub fn timestamped_tokens(&self) -> Vec<Vec<i64>> {
        self.with_timestamps
            .windows
            .iter()
            .map(|w| w.tokens.clone())
            .collect()
    }
}

/// Windows' text as one string, joined in order.
fn join_windows(windows: &[ReferenceWindow]) -> String {
    windows
        .iter()
        .map(|w| w.text.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The ground-truth transcript.
///
/// Leading and trailing `...` mark that the clip is cut from a longer
/// recording; they are not words and must not count as errors.
fn transcript(name: &str) -> String {
    std::fs::read_to_string(testdata(&format!("{name}.txt")))
        .expect("the transcript failed to read")
        .replace("...", " ")
}

/// The multilingual vocabulary, for turning ids into text.
///
/// The layout comes from the vocabulary size these fixtures are for, and the
/// ranks from the bundled `multilingual.tiktoken` — the same assets bunsen
/// itself would use, so this checks them as well as using them.
pub struct Vocab {
    pub policy: TokenPolicy,
    pub detokenizer: WordchipperDetokenizer<u16>,
    pub ranks: TiktokenRanks,
}

impl Vocab {
    /// The text of a window's ids: prompt, timestamps and stop token dropped.
    fn decode(
        &self,
        ids: &[i64],
    ) -> String {
        self.detokenizer
            .detokenize(&self.policy.text_ids(ids))
            .expect("every id is inside the vocabulary")
    }
}

/// The shared vocabulary, for turning ids into text.
fn vocab() -> Vocab {
    let policy = TokenPolicy::from_vocab_size(N_VOCAB).expect("a Whisper vocabulary size");
    let ranks = TiktokenRanks::load(bundled::multilingual_tiktoken())
        .expect("the vocabulary failed to load");
    let decoder = detokenizer(&ranks, policy.ids()).expect("the vocabulary failed to load");
    Vocab {
        policy,
        detokenizer: decoder,
        ranks,
    }
}

/// A fixture's samples, decoded from mp3.
fn samples(name: &str) -> Vec<f32> {
    bunsen::support::audio::load_audio_mono_sr(testdata(&format!("{name}.mp3")), SAMPLE_RATE)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Formats a WER failure so the reader can see whether it is a real
/// regression or a wording difference.
fn report(
    label: &str,
    name: &str,
    got: &str,
    want: &str,
    wer: f64,
    ceiling: f64,
) -> String {
    format!(
        "{name}: {label} word error rate {wer:.4} exceeds {ceiling:.4}\n  \
         expected: {want}\n  actual:   {got}",
    )
}

/// Everything that loads weights. `download` is what fetches them.
#[cfg(feature = "download")]
mod download;
#[cfg(feature = "gpu-tests")]
mod gpu_tests;

/// A fixture as `[1, N_MELS, frames]` log-mels, through bunsen's front end.
///
/// The clip is zero-padded up to whole 30 s windows first, matching how
/// Whisper pads short audio, and then converted in a single call — the
/// streaming context is a homomorphism over chunking, so one call and many
/// give the same spectrogram.
pub fn clip_mels<B: Backend>(
    name: &str,
    device: &Device<B>,
) -> Tensor<B, 3> {
    let wav = samples(name);

    let windows = wav.len().div_ceil(N_SAMPLES).max(1);
    let mut values: Vec<f64> = wav.iter().map(|&v| v as f64).collect();
    values.resize(windows * N_SAMPLES, 0.0);
    let n = values.len();

    let converter: MelConverter<B> = mel_options(SAMPLE_RATE, N_MELS)
        .try_init(device)
        .expect("mel converter");
    let ctx = converter.new_context(1);
    let (mels, ctx) = ctx
        .transform(Tensor::from_data(TensorData::new(values, [1, n]), device))
        .expect("mel conversion");

    let joined = match ctx.finish() {
        Some(tail) => Tensor::cat(vec![mels, tail], 1),
        None => mels,
    };

    package_mels(joined)
}

/// bunsen's Whisper, from the fetched checkpoint, in f32.
///
/// OpenAI ships fp16; the mel front end produces the backend's default
/// float. Feeding f32 input to an f16 model does not error, it just
/// returns wrong numbers, so the cast is load-bearing.
pub fn bunsen_model<B: Backend>(device: &Device<B>) -> Whisper<B> {
    let (model, cfg) = Whisper::load_pretrained(device).expect("load base.pt");

    assert_eq!(cfg.n_mels, N_MELS, "not a `base` model");
    assert_eq!(
        cfg.vocab_size, N_VOCAB,
        "these fixtures are for a multilingual checkpoint; an English-only \
             one numbers its special tokens differently",
    );

    model.map(&mut DTypeMapper::new(burn::tensor::DType::F32))
}

/// Prints where two per-window id sequences first diverge.
///
/// The gates assert on text, because a backend can flip a near-tied argmax
/// without moving the words; this is what says whether that happened, and
/// where, so a failure, or a pass that hides a flipped token, can be read at
/// token level.
pub fn report_id_diff(
    label: &str,
    name: &str,
    mine: &[Vec<i64>],
    theirs: &[Vec<i64>],
) {
    for (w, (a, b)) in mine.iter().zip(theirs).enumerate() {
        match a.iter().zip(b).position(|(x, y)| x != y) {
            None if a.len() == b.len() => {
                eprintln!("{name}: {label} window {w}: {} ids, identical", a.len());
            }
            None => eprintln!(
                "{name}: {label} window {w}: identical for {} ids, then lengths differ ({} vs {})",
                a.len().min(b.len()),
                a.len(),
                b.len(),
            ),
            Some(i) => eprintln!(
                "{name}: {label} window {w}: first divergence at {i}: {:?} vs {:?}",
                &a[i..(i + 5).min(a.len())],
                &b[i..(i + 5).min(b.len())],
            ),
        }
    }
    if mine.len() != theirs.len() {
        eprintln!(
            "{name}: {label}: {} window(s) vs {}",
            mine.len(),
            theirs.len()
        );
    }
}

/// Joins per-window ids into one transcript.
pub fn to_text(
    table: &Vocab,
    windows: &[Vec<i64>],
) -> String {
    windows
        .iter()
        .map(|ids| table.decode(ids).trim().to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

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
                .sot_sequence(Some("en"), Some(Task::Transcribe), true)
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
