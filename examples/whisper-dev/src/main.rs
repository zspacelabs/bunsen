//! Loads a pretrained Whisper checkpoint, converts an audio file to log-mels
//! with [`bunsen::ops::signal::mels`], and runs the audio encoder over them.
//!
//! The mel front end is driven in streaming chunks, which is the shape a live
//! transcription loop wants; feeding the whole clip in one call gives the same
//! result.

use std::path::PathBuf;

use bunsen::{
    burner::module::{
        DTypeMapper,
        ModuleInit,
    },
    kits::speech::whisper::pretrained::PytorchWhisperScanner,
    ops::signal::mels::{
        AffineCompress,
        MelConverter,
        MelConverterMeta,
        MelConverterOptions,
        RangeClamp,
    },
    support::audio::load_audio_mono_sr,
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

/// Whisper's fixed analysis window: 30 s at 16 kHz.
const N_SAMPLES: usize = 30 * 16_000;

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

    /// Sample rate of the audio file.
    #[arg(long, default_value = "16000")]
    pub sample_rate: usize,

    /// Milliseconds of audio per streaming chunk; must be a whole number of
    /// 10 ms hops.
    #[arg(long, default_value = "1000")]
    pub chunk_ms: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{args:#?}");

    let (_, wav) = load_audio_mono_sr(&args.audio, args.sample_rate)?;

    cfg_select! {
        feature = "cuda" => run::<burn::backend::cuda::Cuda>(args, wav),
        feature = "metal" => run::<burn::backend::Metal>(args, wav),
        feature = "wgpu" => run::<burn::backend::wgpu::Wgpu>(args, wav),
        feature = "flex" => run::<burn::backend::flex::Flex>(args, wav),
        _ => {
            compile_error!("No Backend enabled");
        }
    }
}

/// Pads with silence or trims to Whisper's fixed 30 s window.
fn pad_or_trim(
    mut wav: Vec<f32>,
    n_samples: usize,
) -> Vec<f32> {
    let had = wav.len();
    wav.resize(n_samples, 0.0);

    println!(
        "audio: {had} samples ({:.2} s) -> {n_samples} ({})",
        had as f64 / 16_000.0,
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
/// The clamp and the affine tail are applied **once, after** the stream is
/// joined. `RangeClamp::PerCall` reduces over a single call's frames, so
/// leaving them enabled during streaming would clamp each chunk against its
/// own maximum — see the note on `MelConversionContext`.
fn to_whisper_mels<B: Backend>(
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

    let joined: Tensor<B, 3> = Tensor::cat(pieces, 1);
    let frames = joined.dims()[1];

    // Whisper slices `stft[..., :-1]`.
    let cut = joined.slice_dim(1, 0..frames as isize - 1);

    let packaged = AffineCompress::default().apply(RangeClamp::PerCall { db: 8.0 }.apply(cut));

    // `[batch, frames, n_mels]` -> the `[batch, n_mels, seq]` the encoder wants.
    Ok(packaged.swap_dims(1, 2))
}

#[allow(unused)]
fn run<B: Backend>(
    args: Args,
    wav: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = B::Device::default();

    let (model, cfg) = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
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
    // trained on; `n_mels` comes from the checkpoint rather than a constant.
    let options = MelConverterOptions::default()
        .with_sample_rate(args.sample_rate)
        .with_n_mels(cfg.n_mels)
        .with_range_clamp(None)
        .with_affine(None);

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

    let wav = pad_or_trim(wav, N_SAMPLES);
    let mels = to_whisper_mels(&conv, &wav, chunk, &device)?;

    println!("streamed in {chunk}-sample chunks");
    summarize("log-mels", &mels);

    let encoded = model.forward_encoder(mels);
    summarize("encoder ", &encoded);

    Ok(())
}
