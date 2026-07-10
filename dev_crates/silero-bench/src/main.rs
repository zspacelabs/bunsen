use std::path::Path;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::silero_vad::{
        SileroVad,
        SileroVadCollection,
        SileroVadMeta,
        VadRunningContextConfig,
    },
    support::testing::PerformanceBackend,
};
use burn::Tensor;
use clap::Parser;
use hound::{
    SampleFormat,
    WavSpec,
};

type B = PerformanceBackend;

/// Silero VAD Benchmark tool.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// Path to `.wav` file. Must be mono, with the target sample rate.
    #[arg(long)]
    pub path: String,

    /// The sample rate.
    #[arg(long, default_value = "16000")]
    pub sample_rate: usize,
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();
    println!("* {:#?}", args);

    println!("\n> Loading models");
    let device = Default::default();
    println!("* device: {:?}", device);

    println!("* SileroVad");
    let vad: SileroVad<B> = SileroVadCollection::load_pretrained(&device)?
        .try_branch(args.sample_rate)?
        .clone();

    let chunk_size = vad.chunk_size();
    println!("  - {} chunk_size: {}", args.sample_rate, chunk_size);

    println!("\n> Loading audio file: \"{}\"", args.path);
    let chunk_seq: Tensor<B, 3> = {
        let (spec, mut wav_vec) = load_audio_mono_sr(&args.path, args.sample_rate)?;
        println!("* {:?}", spec);

        let tail_len = wav_vec.len() % chunk_size;
        if tail_len != 0 {
            let pad_len = chunk_size - tail_len;
            wav_vec.resize(wav_vec.len() + pad_len, 0.0);
        }

        let samples = Tensor::<B, 1>::from_floats(wav_vec.as_slice(), &device);

        // [chunks, 1, samples=chunk_size]
        samples.reshape([-1, 1, chunk_size as isize])
    };
    println!("* chunk_seq.dims: {:?}", chunk_seq.dims());

    println!("\n> VadRunningContext::predict_chunk_sequence([steps, batch, chunk_size]):");
    let ctx = VadRunningContextConfig::new(args.sample_rate).init(&vad, &device);

    let (_ctx, seq_out) = ctx.predict_chunk_sequence(chunk_seq.clone(), &vad);
    let seq_out = seq_out
        .squeeze_dim::<1>(1)
        .to_data()
        .to_vec::<f32>()
        .map_err(BunsenError::external)?;
    println!("{:0.4?}", seq_out);

    Ok(())
}

/// Loads a mono audio file.
///
/// # Arguments
/// * `filename` - path to an audio file.
/// * `sample_rate` - sample rate of the audio file.
pub fn load_audio_mono_sr<P: AsRef<Path>>(
    filename: P,
    sample_rate: usize,
) -> BunsenResult<(WavSpec, Vec<f32>)> {
    let filename = filename.as_ref();

    let mut reader = hound::WavReader::open(filename).map_err(BunsenError::external)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(BunsenError::Invalid(
            "The audio must be single-channel".to_string(),
        ));
    }
    if spec.sample_rate as usize != sample_rate {
        return Err(BunsenError::Invalid(format!(
            "Expected sample_rate = {}, but found {}",
            sample_rate, spec.sample_rate
        )));
    }

    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.unwrap())
            .collect::<Vec<f32>>(),
        (SampleFormat::Int, bits) => {
            let scale = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .map_err(BunsenError::external)?
                .into_iter()
                .map(|s| s as f32 / scale)
                .collect()
        }
        _ => unreachable!("hound rejects other formats at open"),
    };

    Ok((spec, samples))
}
