use std::path::PathBuf;

use bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner;
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

    let cfg = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
        .scan_cfg(PathBuf::from(args.source.clone()))?;

    println!("{:#?}", cfg);

    Ok(())
}
