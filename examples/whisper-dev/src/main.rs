use std::path::PathBuf;

use bunsen::kits::speech::whisper::{
    blocks::Whisper,
    pretrained::PytorchWhisperScanner,
};
use burn::prelude::Backend;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the source model.
    #[arg(long)]
    pub source: String,

    /// Path to the source model.
    #[arg(long, default_value = "model_state_dict")]
    pub top_level_key: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:#?}", args);

    cfg_select! {
        feature = "cuda" => {
            run::<burn::backend::cuda::Cuda>(args)
        }
        feature = "metal" => {
            run::<burn::backend::metal::Metal>(args)
        }
        feature = "wgpu" => {
            run::<burn::backend::wgpu::Wgpu>(args)
        }
        feature = "flex" => {
            run::<burn::backend::flex::Flex>(args)
        }
        _ => {
            panic!("No Backend enabled");
        }
    }
}

#[allow(unused)]
fn run<B: Backend>(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let device = B::Device::default();

    let (module, cfg): (Whisper<B>, _) = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
        .load(PathBuf::from(args.source.clone()), &device)?;

    println!("{:#?}", cfg);

    Ok(())
}
