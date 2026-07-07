use std::fs::File;

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use clap::Parser;
use polars::prelude::*;

/// Silero VAD Benchmark tool.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The data path.
    #[arg(long)]
    pub path: String,
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();
    println!("{:#?}", args);

    // 1. Open the .feather / .arrow file
    let file = File::open(args.path).expect("Could not open file");

    let df = IpcReader::new(file)
        .finish()
        .map_err(BunsenError::external)?;

    println!("{:#?}", df);
    Ok(())
}
