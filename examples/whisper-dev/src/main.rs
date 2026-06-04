use std::path::PathBuf;

use bunsen::{
    burner::module::ModuleInit,
    errors::WithOkOrPanic,
    kits::speech::whisper::{
        blocks::Whisper,
        pretrained::PytorchWhisperScanner,
    },
};
use burn::prelude::Backend;
use burn_store::ModuleSnapshot;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the source model.
    #[arg(long, default_value = "/media/Data/models/whisper/base.en.pt")]
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
    let (mut store, cfg) = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
        .scan_cfg(PathBuf::from(args.source.clone()))?;

    println!("{:#?}", cfg);
    // println!("keys: {:#?}", store.keys());

    let device = Default::default();

    let mut whisper: Whisper<B> = cfg.try_init(&device)?;
    whisper.load_from(&mut store).ok_or_panic();

    Ok(())
}
