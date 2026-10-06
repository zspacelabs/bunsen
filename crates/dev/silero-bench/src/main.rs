use bunsen::{
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        ParseError,
        sys_at,
    },
    kits::speech::silero_vad::{
        SileroVad,
        SileroVadCollection,
        SileroVadContextConfig,
        SileroVadMeta,
    },
    support::{
        audio::load_audio_mono_sr,
        testing::{
            PerformanceBackend,
            backend_device,
        },
    },
};
use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Tolerance,
};
use clap::Parser;

type B = PerformanceBackend;

/// Silero VAD Benchmark tool.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// Path to `.wav` file. Must be mono, with the target sample rate.
    #[arg(long)]
    pub path: String,

    /// Expected output; json array file path.
    #[arg(long)]
    pub expected: Option<String>,

    /// The sample rate.
    #[arg(long, default_value = "16000")]
    pub sample_rate: usize,
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();
    println!("* {:#?}", args);

    println!("\n> Loading models");
    let device = backend_device::<B>();
    println!("* device: {:?}", device);

    println!("* SileroVad");
    let vad: SileroVad<B> = SileroVadCollection::load_pretrained(&device)?
        .try_branch(args.sample_rate)?
        .clone();

    let chunk_size = vad.chunk_size();
    println!("  - {} chunk_size: {}", args.sample_rate, chunk_size);

    println!("\n> Loading audio file: \"{}\"", args.path);
    let mut wav_vec = load_audio_mono_sr(&args.path, args.sample_rate)?;
    println!("* {} samples", wav_vec.len());

    // [steps, 1, samples=chunk_size]
    let chunk_seq: Tensor<B, 3> = {
        // Pad the audio to the chunk size.
        let tail_len = wav_vec.len() % chunk_size;
        if tail_len != 0 {
            let pad_len = chunk_size - tail_len;
            wav_vec.resize(wav_vec.len() + pad_len, 0.0);
        }

        // Convert to tensor.
        let samples = Tensor::<B, 1>::from_floats(wav_vec.as_slice(), &device);

        // Chunk the audio into chunks of size `chunk_size`.
        samples.reshape([-1, 1, chunk_size as isize])
    };
    println!("* chunk_seq.dims: {:?}", chunk_seq.dims());

    println!("\n> SileroVad::context_forward_sequence([steps, batch, chunk_size], ctx):");
    // [steps, batch=1]
    let (chunk_probs, _ctx) = vad.context_forward_sequence(
        chunk_seq,
        SileroVadContextConfig::new(args.sample_rate).init(&vad, &device),
    );
    // [steps]
    let chunk_probs = chunk_probs.squeeze_dim::<1>(1).to_data();
    println!(
        "{:0.4?}",
        chunk_probs
            .clone()
            .to_vec::<f32>()
            .map_err(|e| BunsenError::illegal("the probabilities are not f32").with_cause(e))?
    );

    if let Some(expected_path) = &args.expected {
        println!(
            "\n> Checking against expected output: \"{}\"",
            expected_path
        );
        let file = std::fs::File::open(expected_path).map_err(sys_at("open", expected_path))?;
        let expected: Vec<f32> = serde_json::from_reader(file).map_err(|e| {
            BunsenError::from_cause(
                BunsenErrorKind::InvalidResource,
                ParseError::new("expected output")
                    .at(expected_path)
                    .with_source(e),
            )
        })?;

        let expected: TensorData = TensorData::from(expected.as_slice());
        chunk_probs.assert_approx_eq(&expected, Tolerance::<f32>::permissive());
        println!("  - OK");
    }

    Ok(())
}
