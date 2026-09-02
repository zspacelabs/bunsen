//! Loads a pretrained Whisper checkpoint, converts an audio file to log-mels
//! with [`bunsen::ops::signal::mels`], and greedily decodes each 30 s window.
//!
//! The mel front end is driven in streaming chunks, which is the shape a live
//! transcription loop wants; feeding the whole clip in one call gives the same
//! result. The prompt and stop token are derived from the checkpoint's own
//! vocabulary size, so an English-only and a multilingual model each get the
//! ids they were trained on without anyone typing them in.
//!
//! The backend is [`bunsen::support::testing::PerformanceBackend`]: the one
//! bunsen's own compute-heavy tests run on, selected by bunsen's backend
//! feature at build time (`--features bunsen/wgpu`, `bunsen/cuda`,
//! `bunsen/metal`; `flex` with none). What this example runs on is what the
//! tests ran on.

use std::path::PathBuf;

use bunsen::{
    burner::module::{
        DTypeMapper,
        ModuleInit,
    },
    kits::{
        speech::whisper::{
            WhisperMeta,
            blocks::WhisperFrontEndConfig,
            decode::{
                GreedyDecodeConfig,
                mel_windows,
            },
            driver::{
                Task,
                load_detokenizer,
            },
            pretrained::PytorchWhisperScanner,
        },
        tokens::Detokenizer,
    },
    ops::signal::mels::{
        MelConverter,
        MelConverterMeta,
    },
    support::{
        audio::load_audio_mono_sr,
        testing::PerformanceBackend,
    },
};
use burn::{
    module::Module,
    prelude::{
        Backend,
        Tensor,
        TensorData,
    },
    tensor::DType,
};
use clap::Parser;

/// Prints min / mean / max, so the output is checkable against a reference
/// rather than just shaped correctly.
fn summarize<B: Backend, const D: usize>(
    label: &str,
    t: &Tensor<B, D>,
) {
    let min = t.clone().min().into_scalar();
    let max = t.clone().max().into_scalar();
    let mean = t.clone().mean().into_scalar();

    println!(
        "{label}: {:?} min {min:.6} mean {mean:.6} max {max:.6}",
        t.dims()
    );
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the source model.
    #[arg(long)]
    pub source: String,

    /// Path to the source model.
    #[arg(long, default_value = "model_state_dict")]
    pub top_level_key: Option<String>,

    /// Path to the audio file.
    #[arg(long)]
    pub audio: String,

    /// The rate the checkpoint is declared at, and the audio file is decoded
    /// at; a multiple of 200 Hz.
    #[arg(long, default_value = "16000")]
    pub sample_rate: usize,

    /// Milliseconds of audio per streaming chunk; must be a whole number of
    /// 10 ms hops.
    #[arg(long, default_value = "1000")]
    pub chunk_ms: usize,

    /// Cap on generated tokens per 30 s window.
    #[arg(long, default_value = "32")]
    pub max_tokens: usize,

    /// Language of the speech, as a Whisper code (`en`, `ja`, ...).
    ///
    /// Used by multilingual checkpoints; an English-only checkpoint takes no
    /// language token and ignores this.
    #[arg(long, default_value = "en")]
    pub language: String,

    /// `transcribe` or `translate` (to English). Ignored by an English-only
    /// checkpoint, as `--language` is.
    #[arg(long, default_value = "transcribe")]
    pub task: String,

    /// Let the model emit timestamp tokens.
    #[arg(long)]
    pub timestamps: bool,

    /// A `.tiktoken` vocabulary matching the checkpoint, to print text as
    /// well as ids: `multilingual.tiktoken` for a multilingual checkpoint,
    /// `gpt2.tiktoken` for an English-only one.
    #[arg(long)]
    pub vocab: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{args:#?}");

    let wav = load_audio_mono_sr(&args.audio, args.sample_rate)?;

    run::<PerformanceBackend>(args, wav)
}

/// Pads with silence or trims to a whole number of windows.
fn pad_or_trim(
    mut wav: Vec<f32>,
    n_samples: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let had = wav.len();
    wav.resize(n_samples, 0.0);

    println!(
        "audio: {had} samples ({:.2} s) -> {n_samples} ({})",
        had as f64 / sample_rate as f64,
        if had < n_samples {
            "zero-padded"
        } else {
            "trimmed"
        },
    );

    wav
}

/// Converts a waveform to Whisper-ready log-mels, `[batch, n_mels, frames]`.
///
/// Streams the signal in `chunk`-sample blocks, then packages the joined
/// result once with the front end's `package_mels` — which must see the
/// whole spectrogram, since its clamp reduces over what it is given.
fn to_whisper_mels<B: Backend>(
    front_end: &WhisperFrontEndConfig,
    conv: &MelConverter<B>,
    wav: &[f32],
    chunk: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let mut ctx = conv.new_context(1);
    let mut pieces = Vec::new();

    for block in wav.chunks(chunk) {
        let samples: Vec<f64> = block.iter().map(|&v| v as f64).collect();
        let n = samples.len();

        let x = Tensor::from_data(TensorData::new(samples, [1, n]), device);
        let (mels, next) = ctx.transform(x)?;

        ctx = next;
        pieces.push(mels);
    }

    if let Some(tail) = ctx.finish() {
        pieces.push(tail);
    }

    Ok(front_end.package_mels(Tensor::cat(pieces, 1)))
}

#[allow(unused)]
fn run<B: Backend>(
    args: Args,
    wav: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = B::Device::default();

    let (model, cfg) = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
        .with_front_end(WhisperFrontEndConfig::new().with_sample_rate(args.sample_rate))
        .load::<B, _>(PathBuf::from(args.source.clone()), &device)?;

    println!("{cfg:#?}");

    // OpenAI ships these checkpoints in fp16, so the loaded weights are f16
    // while the mel front end produces the backend's default float. Cast the
    // model up rather than the mels down: the front end is where precision is
    // cheap, and f16 conv support varies by backend.
    //
    // This works because the model's weights are `Param`s. A `ModuleMapper`
    // does *not* reach bare `Tensor` fields — see the note on `MelConverter`.
    let model = model.map(&mut DTypeMapper::new(DType::F32));

    // The front end must produce exactly the channel count the encoder was
    // trained on; `n_mels` comes from the checkpoint, and the rate is what
    // the loader declared on it.
    let options = cfg.front_end.mel_converter_options(cfg.n_mels)?;

    let conv: MelConverter<B> = options.try_init(&device)?;

    let hop = conv.hop();
    let chunk = args.chunk_ms * args.sample_rate / 1000;
    if chunk == 0 || !chunk.is_multiple_of(hop) {
        return Err(format!(
            "--chunk-ms {} is {chunk} samples, which is not a whole number of \
             {hop}-sample hops",
            args.chunk_ms,
        )
        .into());
    }

    // Round up to whole 30 s windows instead of trimming, so batching has
    // more than one window to work with. A window is the model's audio
    // context in frames, times the hop.
    let n_samples = model.max_audio_ctx() * hop;
    let windows_needed = wav.len().div_ceil(n_samples).max(1);
    let wav = pad_or_trim(wav, windows_needed * n_samples, cfg.front_end.sample_rate);
    let mels = to_whisper_mels(&cfg.front_end, &conv, &wav, chunk, &device)?;

    println!("streamed in {chunk}-sample chunks");
    summarize("log-mels", &mels);

    let windows = mel_windows(mels, model.max_audio_ctx());
    println!(
        "{} window(s) of {} frames",
        windows.len(),
        model.max_audio_ctx()
    );

    // The prompt and stop token fall out of the checkpoint's vocabulary size:
    // a multilingual model and an English-only one number their specials
    // differently, and getting that wrong is silent garbage, not an error.
    let policy = cfg.tokens.policy_for_vocab(cfg.vocab_size)?;
    let (language, task) = if policy.ids().is_multilingual() {
        let task = match args.task.as_str() {
            "transcribe" => Task::Transcribe,
            "translate" => Task::Translate,
            other => {
                return Err(
                    format!("--task must be transcribe or translate, got {other:?}").into(),
                );
            }
        };
        (Some(args.language.as_str()), Some(task))
    } else {
        println!("English-only checkpoint: --language and --task do not apply");
        (None, None)
    };
    let prompt = policy.sot_sequence(language, task, args.timestamps)?;
    println!("prompt: {prompt:?}  eot: {}", policy.ids().eot);

    let decode = GreedyDecodeConfig::new(prompt, policy.ids().eot).with_max_tokens(args.max_tokens);

    let detokenizer = match &args.vocab {
        Some(path) => Some(load_detokenizer(path, &policy)?),
        None => None,
    };

    // All windows decoded as one batch. Each window is independent, so this
    // is the same result as decoding them one at a time.
    let batch: Tensor<B, 3> = Tensor::cat(windows, 0);
    println!("batched decode: {:?}", batch.dims());

    for (i, tokens) in model
        .decode_window_batched(batch, &decode)
        .into_iter()
        .enumerate()
    {
        println!("window {i}: {} tokens", tokens.len());
        println!("  {tokens:?}");
        if let Some(detokenizer) = &detokenizer {
            println!("  {:?}", detokenizer.detokenize(&policy.text_ids(&tokens))?);
            if args.timestamps {
                println!("  {:?}", detokenizer.detokenize(&tokens)?);
            }
        }
    }

    Ok(())
}
