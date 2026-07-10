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
        reference::ReferenceModel,
    },
    support::testing::PerformanceBackend,
};
use burn::{
    Tensor,
    tensor::Tolerance,
};
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

    println!("* ONNX ReferenceModel");
    let reference = ReferenceModel::<B>::load_pretrained(&device);

    println!("\n> Loading audio file: \"{}\"", args.path);
    let samples = {
        let (spec, mut wav_vec) = load_audio_mono_sr(&args.path, args.sample_rate)?;
        println!("* {:?}", spec);

        let tail_len = wav_vec.len() % chunk_size;
        if tail_len != 0 {
            let pad_len = chunk_size - tail_len;
            wav_vec.resize(wav_vec.len() + pad_len, 0.0);
        }

        let samples = Tensor::<B, 1>::from_floats(wav_vec.as_slice(), &device);

        // [chunks, samples=chunk_size]
        samples.reshape([-1, chunk_size as isize])
    };
    println!("* samples.dims: {:?}", samples.dims());

    println!("\n> VadRunningContext::predict_chunk_sequence([steps, batch, chunk_size]):");
    {
        let chunk_seq = samples.clone().reshape([-1, 1, chunk_size as isize]);
        let ctx = VadRunningContextConfig::new(args.sample_rate).init(&vad, &device);

        let (_ctx, seq_out) = ctx.predict_chunk_sequence(chunk_seq, &vad);

        let seq_out = seq_out
            .squeeze_dim::<1>(1)
            .to_data()
            .to_vec::<f32>()
            .map_err(BunsenError::external)?;
        println!("{:0.4?}", seq_out);
    }

    let steps = samples.dims()[0];

    println!("\n> Testing SileroVad::forward([batch, chunk_size], state):");
    let state0 = vad.init_state(steps, &device);
    let (mod_out, _) = vad.forward(samples.clone(), state0.clone());
    let mod_out = mod_out.to_data();

    println!(
        "{:0.4?}",
        mod_out
            .clone()
            .to_vec::<f32>()
            .map_err(BunsenError::external)?
    );

    print!("* ReferenceModel::forward: ");
    // [batch, 1]
    let (ref_out, _) = reference.forward(samples.clone(), args.sample_rate as i64, state0.clone());
    // [batch]
    let ref_out = ref_out.flatten::<1>(0, 1).to_data();

    mod_out.assert_approx_eq::<f32>(&ref_out, Tolerance::permissive());
    println!("approx_eq");

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
