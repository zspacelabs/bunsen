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

use bunsen::support::testing::asr::{
    BpeDecodeTable,
    WHISPER_FIRST_SPECIAL,
    text_error_rate,
};

use super::*;

/// A fixture, and the accuracy it is held to.
pub struct Fixture {
    /// Basename under `testdata/`, without extension.
    pub name: &'static str,

    /// Expected number of 30 s windows, as a cheap shape check.
    pub windows: usize,

    /// **The accuracy knob.** Ceiling on word error rate against the
    /// ground-truth transcript.
    ///
    /// Raise it to accept a weaker model or a harder clip; lower it to hold a
    /// gain. A failure here means the pipeline got worse at transcribing,
    /// which is the thing worth knowing.
    pub max_wer: f64,

    /// Ceiling on word error rate against the committed `openai-whisper`
    /// decode.
    ///
    /// Separate from [`Self::max_wer`] because it measures a different thing:
    /// not "is this accurate" but "does this agree with the implementation
    /// bunsen was transliterated from". A backend whose arithmetic flips an
    /// argmax moves this without moving accuracy much, so it has its own knob.
    pub max_reference_wer: f64,
}

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

/// `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`.
const PROMPT: [i64; 4] = [50258, 50259, 50359, 50363];

/// `<|endoftext|>`.
const EOT: i64 = 50257;

fn testdata(rel: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata")).join(rel)
}

/// The committed `openai-whisper` decode of a fixture.
#[derive(serde::Deserialize)]
struct Reference {
    windows: Vec<ReferenceWindow>,
}

#[derive(serde::Deserialize)]
struct ReferenceWindow {
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

/// The shared vocabulary, for turning ids into text.
fn vocab() -> BpeDecodeTable {
    BpeDecodeTable::load(testdata("whisper_vocab.bin")).expect("the vocabulary failed to load")
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

/// The committed reference must decode to its own committed text.
///
/// Needs no model, so it runs in an ordinary `cargo test` — and it is what
/// checks the vocabulary table end to end, since a wrong table would move the
/// decoded text away from what `openai-whisper` printed.
#[test]
fn test_reference_tokens_decode_to_reference_text() {
    let table = vocab();
    assert_eq!(
        table.len(),
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
                table.decode(&window.tokens, WHISPER_FIRST_SPECIAL).trim(),
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

/// Everything that loads weights. `download` is what fetches them.
#[cfg(feature = "download")]
mod with_weights {
    use bunsen::{
        burner::{
            module::{
                DTypeMapper,
                ModuleInit,
            },
            tensor::TensorElemOpExt,
        },
        kits::speech::whisper::{
            blocks::Whisper,
            decode::GreedyDecodeConfig,
            mel::{
                mel_options,
                package_mels,
            },
        },
        ops::signal::mels::MelConverter,
        support::testing::PerformanceBackend,
    };
    use burn::{
        module::Module as _,
        prelude::*,
        tensor::{
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;

    type B = PerformanceBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// A fixture as `[1, N_MELS, frames]` log-mels, through bunsen's front end.
    ///
    /// The clip is zero-padded up to whole 30 s windows first, matching how
    /// Whisper pads short audio, and then converted in a single call — the
    /// streaming context is a homomorphism over chunking, so one call and many
    /// give the same spectrogram.
    fn clip_mels(
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

    /// Splits log-mels into the encoder's fixed windows, zero-padding the last.
    fn windows_of(
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

    /// bunsen's Whisper, from the fetched checkpoint, in f32.
    ///
    /// OpenAI ships fp16; the mel front end produces the backend's default
    /// float. Feeding f32 input to an f16 model does not error, it just
    /// returns wrong numbers, so the cast is load-bearing.
    fn bunsen_model(device: &Device<B>) -> Whisper<B> {
        let (model, cfg) = Whisper::load_pretrained(device).expect("load base.pt");

        assert_eq!(cfg.n_mels, N_MELS, "not a `base` model");
        assert_eq!(
            cfg.vocab_size, N_VOCAB,
            "these fixtures are for a multilingual checkpoint; an English-only \
             one numbers its special tokens differently",
        );

        model.map(&mut DTypeMapper::new(burn::tensor::DType::F32))
    }

    /// Greedily decodes one window against the reference decoder.
    ///
    /// The export is KV-cache-free, so the whole prefix is re-fed every step —
    /// which is slow, and exactly why bunsen has a cache.
    fn greedy_reference(
        decoder: &reference::decoder::Model<B>,
        xa: Tensor<B, 3>,
        device: &Device<B>,
    ) -> Vec<i64> {
        let mut prefix = PROMPT.to_vec();
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

            if picked[0] == EOT {
                break;
            }
            out.push(picked[0]);
            prefix.push(picked[0]);
        }

        out
    }

    /// Joins per-window ids into one transcript.
    fn to_text(
        table: &BpeDecodeTable,
        windows: &[Vec<i64>],
    ) -> String {
        windows
            .iter()
            .map(|ids| table.decode(ids, WHISPER_FIRST_SPECIAL).trim().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// **The accuracy gate.** Real audio, real weights, judged against the
    /// ground-truth transcript.
    #[test]
    fn test_bunsen_accuracy_against_transcript() {
        let device: Device<B> = Default::default();
        let table = vocab();
        let model = bunsen_model(&device);
        let config = GreedyDecodeConfig::new(PROMPT.to_vec(), EOT);

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

    /// **The agreement gate.** bunsen must decode what `openai-whisper`
    /// decodes, on the same audio and the same windowing.
    #[test]
    fn test_bunsen_agrees_with_openai_reference() {
        let device: Device<B> = Default::default();
        let table = vocab();
        let model = bunsen_model(&device);
        let config = GreedyDecodeConfig::new(PROMPT.to_vec(), EOT);

        for fixture in FIXTURES {
            let mels = clip_mels(fixture.name, &device);
            let got = to_text(&table, &model.decode_chunked(mels, &config));
            let want = Reference::load(fixture.name).text();
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

    /// The encoder comparison, on **real** log-mels rather than synthetic ones.
    ///
    /// Real speech is not the same test: `synthetic_mels` is a bounded
    /// sawtooth, while a log-mel spectrogram has the dynamic range and the
    /// near-silent bins a mis-scaled layer norm would show up in.
    #[test]
    fn test_onnx_encoder_matches_bunsen_on_real_audio() {
        let device: Device<B> = Default::default();
        let reference = reference::encoder::Model::<B>::load_pretrained(&device);
        let ours = bunsen_model(&device);

        for fixture in FIXTURES {
            let mels = clip_mels(fixture.name, &device);
            for (w, window) in windows_of(mels, &device).into_iter().enumerate() {
                let theirs = reference.forward(window.clone());
                let mine = ours.forward_encoder(window);

                assert_eq!(mine.dims(), theirs.dims(), "{} window {w}", fixture.name);
                mine.to_data_as::<F>().assert_approx_eq::<F>(
                    &theirs.to_data_as::<F>(),
                    Tolerance::rel_abs(1e-1, 2e-2),
                );
            }
        }
    }

    /// **The ONNX reference transcribes the clip**, independent of bunsen: the
    /// ONNX encoder feeds the ONNX decoder.
    #[test]
    fn test_onnx_reference_transcribes_real_audio() {
        let device: Device<B> = Default::default();
        let table = vocab();
        let reference_enc = reference::encoder::Model::<B>::load_pretrained(&device);
        let reference_dec = reference::decoder::Model::<B>::load_pretrained(&device);

        for fixture in FIXTURES {
            let mels = clip_mels(fixture.name, &device);
            let ids: Vec<Vec<i64>> = windows_of(mels, &device)
                .into_iter()
                .map(|window| {
                    let xa = reference_enc.forward(window);
                    greedy_reference(&reference_dec, xa, &device)
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

    /// **The agreement gate, end to end.** bunsen and the ONNX reference must
    /// transcribe the same clip to the same words.
    ///
    /// This is the one that catches what the staged comparisons let through:
    /// each stage can agree inside tolerance while the composition diverges,
    /// because a greedy argmax turns a small numerical difference into a
    /// different word.
    #[test]
    fn test_onnx_reference_and_bunsen_transcribe_alike() {
        let device: Device<B> = Default::default();
        let table = vocab();
        let reference_enc = reference::encoder::Model::<B>::load_pretrained(&device);
        let reference_dec = reference::decoder::Model::<B>::load_pretrained(&device);
        let ours = bunsen_model(&device);
        let config = GreedyDecodeConfig::new(PROMPT.to_vec(), EOT);

        for fixture in FIXTURES {
            let mels = clip_mels(fixture.name, &device);

            let mut theirs = Vec::new();
            let mut mine = Vec::new();
            for window in windows_of(mels, &device) {
                let xa = reference_enc.forward(window.clone());
                theirs.push(greedy_reference(&reference_dec, xa, &device));
                mine.push(ours.decode_window(window, &config));
            }

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
}
