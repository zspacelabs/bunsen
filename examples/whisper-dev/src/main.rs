use std::path::PathBuf;

use bunsen::{
    kits::speech::whisper::pretrained::PytorchWhisperScanner,
    support::audio::load_audio_mono_sr,
};
use burn::prelude::Backend;
use clap::{
    Parser,
    arg,
};

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:#?}", &args);

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

#[allow(unused)]
fn run<B: Backend>(
    args: Args,
    wav: Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = B::Device::default();

    let (module, cfg) = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
        .load::<B, _>(PathBuf::from(args.source.clone()), &device)?;

    println!("{:#?}", cfg);

    Ok(())
}
