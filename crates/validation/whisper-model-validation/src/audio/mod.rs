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
            TokenPolicy,
            Whisper,
            mel_options,
            package_mels,
            pretrained::bundled,
            text::load_detokenizer,
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

/// The committed `openai-whisper` decode of a fixture.
#[derive(serde::Deserialize)]
pub(crate) struct Reference {
    windows: Vec<ReferenceWindow>,
}

#[derive(serde::Deserialize)]
pub(crate) struct ReferenceWindow {
    tokens: Vec<i64>,
    text: String,
}

impl Reference {
    fn load(name: &str) -> Self {
        let path = testdata(&format!("{name}.reference.json"));
        let file = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("opening {}: {e}", path.display()));
        serde_json::from_reader(file).expect("the reference failed to parse")
    }

    /// The reference decode as one string, windows joined in order.
    fn text(&self) -> String {
        self.windows
            .iter()
            .map(|w| w.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
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
    policy: TokenPolicy,
    detokenizer: WordchipperDetokenizer<u16>,
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
    let detokenizer = load_detokenizer(bundled::multilingual_tiktoken(), policy.ids())
        .expect("the vocabulary failed to load");
    Vocab {
        policy,
        detokenizer,
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

/// The committed reference must decode to its own committed text.
///
/// Needs no model, so it runs in an ordinary `cargo test` — and it is what
/// checks the bundled vocabulary and bunsen's rank parser end to end, since
/// a wrong table would move the decoded text away from what `openai-whisper`
/// printed.
#[test]
fn test_reference_tokens_decode_to_reference_text() {
    let table = vocab();
    assert_eq!(
        table.detokenizer.vocab_size(),
        N_VOCAB,
        "the vocabulary does not cover the multilingual model",
    );

    for fixture in FIXTURES {
        let reference = Reference::load(fixture.name);
        assert_eq!(
            reference.windows.len(),
            fixture.windows,
            "{}: reference covers {} window(s), audio is {}",
            fixture.name,
            reference.windows.len(),
            fixture.windows,
        );

        for (w, window) in reference.windows.iter().enumerate() {
            assert_eq!(
                table.decode(&window.tokens).trim(),
                window.text.trim(),
                "{}: window {w} tokens do not decode to the committed text",
                fixture.name,
            );
        }
    }
}

/// The reference itself must meet the fixture's accuracy bar.
///
/// Needs no model. If this fails, the threshold is wrong or the transcript is
/// — either way the fixture is broken before bunsen is involved, and
/// `max_wer` would be measuring nothing.
#[test]
fn test_reference_meets_the_accuracy_bar() {
    for fixture in FIXTURES {
        let got = Reference::load(fixture.name).text();
        let want = transcript(fixture.name);

        let wer = text_error_rate(&got, &want);
        eprintln!("{}: reference WER vs transcript {wer:.4}", fixture.name);

        assert!(
            wer <= fixture.max_wer,
            "{}",
            report(
                "reference-vs-transcript",
                fixture.name,
                &got,
                &want,
                wer,
                fixture.max_wer,
            ),
        );
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
