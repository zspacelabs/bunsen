//! The fixtures, their references and the helpers the checks share.

use bunsen::{
    burner::module::{
        DTypeMapper,
        ModuleInit,
    },
    kits::{
        speech::{
            asr::whisper::{
                Whisper,
                WhisperMeta,
                driver::WhisperTokenLayout,
                pretrained::WhisperVocabulary,
            },
            support::log_mel::LogMelFrontEndConfig,
        },
        support::tokens::{
            Detokenizer,
            TiktokenRanks,
            WordchipperDetokenizer,
        },
    },
    ops::signal::perceptive_audio::PerceptiveAudioConverter,
};
use burn::{
    Tensor,
    module::Module,
    prelude::{
        Device,
        TensorData,
    },
};

use super::super::*;

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
    clip: bunsen_test_audio::JFK_MOON,
    windows: 2,
    max_wer: 0.10,
    max_reference_wer: 0.0,
}];

/// The rate Whisper's front end is defined at, and the rate fixtures are
/// stored at.
pub(crate) const SAMPLE_RATE: usize = 16_000;

/// Whisper's fixed analysis window: 30 s.
pub(super) const N_SAMPLES: usize = 30 * SAMPLE_RATE;

pub(super) fn testdata(rel: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata")).join(rel)
}

/// The committed `openai-whisper` decodes of a fixture, several ways.
///
/// `validation/tools/gen_speech_fixtures.py` says what each is. `windows` is
/// the agreement gate; the rest are the references later phases of the stream
/// driver pin against, stocked here so that no phase invents its own oracle.
/// The token layout the decodes were made with is in the fixture too, so the
/// vocabulary pairing is checked rather than assumed.
#[derive(serde::Deserialize)]
pub struct Reference {
    /// The vocabulary size of the checkpoint the decodes came from.
    pub(super) vocab_size: usize,
    /// The sot sequence: `<|startoftranscript|> <|en|> <|transcribe|>`.
    pub(super) prompt: Vec<i64>,
    pub(super) no_timestamps: i64,
    pub(super) eot: i64,
    pub(super) timestamp_begin: i64,
    /// Greedy, without timestamps, per fixed 30 s window.
    pub(super) windows: Vec<ReferenceWindow>,
    /// The same decode with timestamp tokens on.
    pub(super) with_timestamps: ReferenceWindows,
    /// Beam 5, without timestamps.
    pub(super) beam5: ReferenceWindows,
    /// `transcribe()` at temperature 0: the seek loop and its segments.
    pub(super) transcribe: ReferenceTranscribe,
}

#[derive(serde::Deserialize)]
pub struct ReferenceWindows {
    pub(super) windows: Vec<ReferenceWindow>,
}

#[derive(serde::Deserialize)]
pub struct ReferenceWindow {
    /// What the decoder emitted, prompt and stop token excluded.
    pub(super) tokens: Vec<i64>,
    /// `Tokenizer.decode`: timestamps dropped.
    pub(super) text: String,
    /// `Tokenizer.decode_with_timestamps`, only where timestamps were on.
    #[serde(default)]
    pub(super) text_with_timestamps: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct ReferenceTranscribe {
    pub(super) text: String,
    pub(super) segments: Vec<ReferenceSegment>,
}

#[derive(serde::Deserialize)]
pub struct ReferenceSegment {
    /// The mel frame the window this segment came from was cut at.
    pub(super) seek: usize,
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) text: String,
    /// Bounded by its timestamp tokens, where the decode emitted them.
    pub(super) tokens: Vec<i64>,
    pub(super) temperature: f64,
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
pub(super) fn join_windows(windows: &[ReferenceWindow]) -> String {
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
pub(super) fn transcript(name: &str) -> String {
    clip(name)
        .transcript()
        .expect("the clip has a transcript")
        .replace("...", " ")
}

/// The clip of the fixture named `name`.
pub(super) fn clip(name: &str) -> bunsen_test_audio::AudioFixture {
    FIXTURES
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("no fixture named {name}"))
        .clip
}

/// The multilingual vocabulary, for turning ids into text.
///
/// The layout comes from the vocabulary size these fixtures are for, and the
/// ranks from the bundled `multilingual.tiktoken` — the same assets bunsen
/// itself would use, so this checks them as well as using them.
pub struct Vocab {
    pub policy: WhisperTokenLayout,
    pub detokenizer: WordchipperDetokenizer<u16>,
    pub ranks: TiktokenRanks,
}

impl Vocab {
    /// The text of a window's ids: prompt, timestamps and stop token dropped.
    pub(super) fn decode(
        &self,
        ids: &[i64],
    ) -> String {
        self.detokenizer
            .detokenize(&self.policy.text_ids(ids))
            .expect("every id is inside the vocabulary")
    }
}

/// The shared vocabulary, for turning ids into text.
pub(crate) fn vocab() -> Vocab {
    let policy = WhisperTokenLayout::from_vocab_size(N_VOCAB).expect("a Whisper vocabulary size");
    let ranks = WhisperVocabulary::for_layout(policy.ids())
        .load(&crate::weights_cache())
        .expect("the vocabulary failed to load");
    let decoder = policy
        .detokenizer(&ranks)
        .expect("the vocabulary failed to load");
    Vocab {
        policy,
        detokenizer: decoder,
        ranks,
    }
}

/// A fixture's samples, decoded from mp3.
pub(crate) fn samples(name: &str) -> Vec<f32> {
    bunsen::support::audio::load_audio_mono_sr(clip(name).path(), SAMPLE_RATE)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Formats a WER failure so the reader can see whether it is a real
/// regression or a wording difference.
pub(super) fn report(
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

/// A fixture as `[1, N_MELS, frames]` log-mels, through bunsen's
/// front end.
///
/// The clip is zero-padded up to whole 30 s windows first, matching how
/// Whisper pads short audio, and then converted in a single call — the
/// streaming context is a homomorphism over chunking, so one call and many
/// give the same spectrogram.
pub fn clip_mels(
    name: &str,
    device: &Device,
) -> Tensor<3> {
    let wav = samples(name);

    let windows = wav.len().div_ceil(N_SAMPLES).max(1);
    let mut values: Vec<f64> = wav.iter().map(|&v| v as f64).collect();
    values.resize(windows * N_SAMPLES, 0.0);
    let n = values.len();

    let front_end = LogMelFrontEndConfig::new().with_sample_rate(SAMPLE_RATE);
    let converter: PerceptiveAudioConverter = front_end
        .mel_converter_options(N_MELS)
        .expect("a Whisper rate")
        .try_init(device)
        .expect("mel converter");
    let mut ctx = converter.new_stream_context(1);
    let mels = converter
        .transform_stream(
            Tensor::from_data(TensorData::new(values, [1, n]), device),
            &mut ctx,
        )
        .expect("mel conversion");

    let joined = match converter.finish_stream(&mut ctx) {
        Some(tail) => Tensor::cat(vec![mels, tail], 1),
        None => mels,
    };

    front_end.package_mels(joined)
}

/// bunsen's Whisper, from the fetched checkpoint, in f32.
///
/// `OpenAI` ships fp16; the mel front end produces the backend's default
/// float. Feeding f32 input to an f16 model does not error, it just
/// returns wrong numbers, so the cast is load-bearing.
pub fn bunsen_model(device: &Device) -> Whisper {
    let construct = crate::load_base(device);
    let model = &construct.model;

    assert_eq!(model.n_mels(), N_MELS, "not a `base` model");
    assert_eq!(
        model.front_end().sample_rate,
        SAMPLE_RATE,
        "the fixtures are stored at Whisper's rate"
    );
    assert_eq!(
        model.vocab_size(),
        N_VOCAB,
        "these fixtures are for a multilingual checkpoint; an English-only \
             one numbers its special tokens differently",
    );

    model
        .clone()
        .map(&mut DTypeMapper::new(burn::tensor::DType::F32))
}

/// Prints where two per-window id sequences first diverge.
///
/// The gates assert on text, because a backend can flip a near-tied argmax
/// without moving the words; this is what says whether that happened, and
/// where so a failure, or a pass that hides a flipped token, can be read at
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
