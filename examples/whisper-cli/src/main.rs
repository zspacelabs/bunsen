//! Transcribes an audio file with the bundled Whisper `base` checkpoint and
//! its vocabulary, through the stream driver: the audio is pushed in chunks
//! as a live loop would feed it, and segments are printed as they become
//! final — or, under the responsive preset, as drafts first.
//!
//! Everything comes from bunsen's own features. `whisper-weights` bundles the
//! checkpoint and the `.tiktoken` vocabulary that matches it, which is what
//! gives text rather than ids and upstream's default suppress list;
//! `silero-weights` bundles the VAD the real-time presets need. The backend
//! is [`bunsen::support::testing::PerformanceBackend`], chosen by bunsen's
//! backend feature at build time (`--features bunsen/wgpu`; see the README).

use std::sync::Arc;

use bunsen::{
    burner::module::DTypeMapper,
    kits::speech::{
        silero_vad::SileroVad,
        whisper::{
            Whisper,
            WhisperFallbackConfig,
            decode::default_filters,
            driver::{
                EmissionPolicy,
                RunningMaxClamp,
                StreamClock,
                WhisperEmission,
                WhisperStreamDriver,
                WhisperStreamDriverConfig,
                WhisperTask,
            },
            pretrained::bundled_vocabulary,
        },
    },
    support::{
        audio::load_audio_mono_sr,
        testing::PerformanceBackend,
    },
};
use burn::{
    module::Module,
    prelude::Backend,
    tensor::DType,
};
use clap::{
    Parser,
    ValueEnum,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TaskArg {
    Transcribe,
    Translate,
}

/// The three deployment targets: when to decode, and when a decode is
/// final.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Preset {
    /// Decode each full window and commit all of it.
    Offline,

    /// Decode at the end of each speech region as well; every emission is
    /// final. Needs the bundled VAD.
    Conservative,

    /// Conservative, plus a draft every 600 ms of speech. Needs the bundled
    /// VAD.
    Responsive,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the audio file; decoded to mono at the model's rate.
    #[arg(long)]
    audio: String,

    /// Milliseconds of audio per push, as a live loop would feed it.
    #[arg(long, default_value = "1000")]
    chunk_ms: usize,

    /// Language of the speech, as a Whisper code (`en`, `ja`, ...); detected
    /// from the first window when omitted.
    #[arg(long)]
    language: Option<String>,

    /// Transcribe, or translate to English.
    #[arg(long, value_enum, default_value_t = TaskArg::Transcribe)]
    task: TaskArg,

    /// Emit timestamp tokens and split segments on them, seeking to the last
    /// closed timestamp, as upstream's `transcribe()` does.
    #[arg(long)]
    timestamps: bool,

    /// Beams per window; one is greedy.
    #[arg(long, default_value = "1")]
    beam_size: usize,

    /// Cap on generated tokens per window.
    #[arg(long, default_value = "224")]
    max_tokens: usize,

    /// Do not prompt each window with the transcript so far.
    #[arg(long)]
    no_prompt_carry: bool,

    /// Climb upstream's temperature ladder when a window's decode fails its
    /// thresholds; without this, temperature zero alone.
    #[arg(long)]
    fallback: bool,

    /// When to decode, and when a decode is final.
    #[arg(long, value_enum, default_value_t = Preset::Offline)]
    preset: Preset,

    /// Print each segment's ids beside its text.
    #[arg(long)]
    ids: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    run::<PerformanceBackend>(args)
}

fn run<B: Backend>(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let device = B::Device::default();

    // The checkpoint ships in fp16 while the mel front end works in the
    // backend's float; cast the model up, where precision is cheap.
    let (model, cfg) = Whisper::<B>::load_pretrained(&device)?;
    let model = model.map(&mut DTypeMapper::new(DType::F32));
    println!(
        "model: {} mels, vocabulary {}, d_model {}, {} + {} layers",
        cfg.n_mels, cfg.vocab_size, cfg.d_model, cfg.n_encoder_layers, cfg.n_decoder_layers,
    );

    // The token layout follows from the vocabulary size, and the bundled
    // vocabulary follows from the layout: nothing here is typed in.
    let policy = cfg.token_layout.policy_for_vocab(cfg.vocab_size)?;
    let ids = *policy.ids();
    let ranks = bundled_vocabulary(&ids)?;
    let detokenizer = policy.detokenizer(&ranks)?;
    let filters = default_filters::<B>(&ranks, &ids);

    let language = if ids.is_multilingual() {
        args.language.clone()
    } else {
        if args.language.is_some() {
            println!("English-only checkpoint: --language and --task do not apply");
        }
        None
    };
    let task = match args.task {
        TaskArg::Transcribe => WhisperTask::Transcribe,
        TaskArg::Translate => WhisperTask::Translate,
    };
    let emission = match args.preset {
        Preset::Offline => EmissionPolicy::offline(),
        Preset::Conservative => EmissionPolicy::conservative(),
        Preset::Responsive => EmissionPolicy::responsive(),
    };
    let fallback = if args.fallback {
        WhisperFallbackConfig::upstream()
    } else {
        WhisperFallbackConfig::new()
    };

    let mut driver: WhisperStreamDriver<B> = WhisperStreamDriverConfig::new()
        .with_language(language)
        .with_task(task)
        .with_timestamps(args.timestamps)
        .with_beam_size(args.beam_size)
        .with_max_tokens(args.max_tokens)
        .with_condition_on_previous_text(!args.no_prompt_carry)
        .with_emission(emission)
        .with_fallback(fallback)
        .init_with_layout(model, policy, &device)?
        .with_detokenizer(Arc::new(detokenizer))
        .with_logit_filters(filters);
    if args.preset != Preset::Offline {
        driver = driver.with_vad(
            SileroVad::<B>::load_16khz_pretrained(&device)?,
            Default::default(),
        )?;
    }
    if driver.detects_language() {
        println!("language: detected from the first window");
    } else {
        println!("prompt: {:?}", driver.prompt());
    }

    // The audio is decoded at the model's rate: the checkpoint's to
    // declare, not the caller's.
    let wav = load_audio_mono_sr(&args.audio, driver.sample_rate())?;
    println!(
        "audio: {} samples, {:.2} s",
        wav.len(),
        wav.len() as f64 / driver.sample_rate() as f64
    );

    // A bare stream: a clock from zero at the model's rate, and the running
    // maximum as the mel clamp reference.
    let mut ctx = driver.new_context(
        StreamClock::uniform(driver.sample_rate()),
        RunningMaxClamp::new(),
    )?;
    let chunk = (args.chunk_ms * driver.sample_rate() / 1000).max(1);
    let mut announced = false;
    for block in wav.chunks(chunk) {
        let emissions = ctx.write_read(block)?;
        // Detection runs on the first window decoded, so the language is
        // known once anything has been emitted; say so before the text.
        if !announced
            && driver.detects_language()
            && let Some(code) = ctx.language()
        {
            println!("language: {code}");
            announced = true;
        }
        for emission in emissions {
            report(&emission, args.ids);
        }
    }
    for emission in ctx.end_read()? {
        report(&emission, args.ids);
    }

    Ok(())
}

/// One line per emission: a draft is marked `~`, a commit is not.
fn report(
    emission: &WhisperEmission,
    ids: bool,
) {
    let segment = emission.segment();
    let mark = if emission.is_committed() { ' ' } else { '~' };
    println!(
        "{mark}[{:>8.2} --> {:>8.2}] {}",
        segment.start,
        segment.end,
        segment.text.as_deref().unwrap_or("").trim(),
    );
    if ids {
        println!("    {:?}", segment.tokens);
    }
}
