use std::{
    collections::BTreeMap,
    path::{
        Path,
        PathBuf,
    },
};

use bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner;
use burn_store::{
    ModuleStore,
    PytorchStore,
    TensorSnapshot,
};
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

pub fn pytorch_snapshots<P: AsRef<Path>>(
    path: P
) -> Result<BTreeMap<String, TensorSnapshot>, Box<dyn std::error::Error>> {
    let mut store = PytorchStore::from_file(path.as_ref());
    let snapshots = store.get_all_snapshots()?.clone();
    Ok(snapshots)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("{:#?}", args);

    let cfg = PytorchWhisperScanner::new()
        .with_top_level_key(args.top_level_key.clone())
        .scan_cfg(PathBuf::from(args.source.clone()))?;

    println!("{:#?}", cfg);

    let snapshots = pytorch_snapshots(args.source)?;
    println!("keys: {:#?}", snapshots.keys());

    // mlp.([01]) => mlp.linear\1

    Ok(())
}
