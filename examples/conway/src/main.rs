#![allow(unused)]

pub mod commands;
pub mod sim;

use bunsen::{
    errors::BunsenResult,
    prelude::TensorOpExt,
};
use bunsen_app::device::{
    DeviceArgs,
    DevicePrefs,
    Precision,
};
use burn::tensor::{
    Device,
    IntDType,
};
use clap::Parser;
use piston::{
    EventLoop,
    OpenGLWindow,
    input::RenderEvent,
};

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The device to simulate on.
    #[command(flatten)]
    pub device: DeviceArgs,

    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Visual Simulation.
    Visual(commands::visual_cmd::VisualCmd),

    /// Non-Visual Benchmark.
    Benchmark(commands::benchmark_cmd::BenchmarkCmd),
}

impl Commands {
    pub fn run(
        &self,
        device: &Device,
    ) -> BunsenResult<()> {
        match self {
            Commands::Visual(cmd) => cmd.run(device),
            Commands::Benchmark(cmd) => cmd.run(device),
        }
    }
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();

    // The boards are 0/1 cells: any half-precision float, and the narrowest
    // int the backend has.
    let prefs = DevicePrefs::new()
        .with_precision(Precision::AnyHalf)
        .with_half(Precision::AnyHalf)
        .with_ints([IntDType::I8, IntDType::I32]);
    let device = args.device.init(&prefs).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{}", bunsen_app::device::describe(&device));

    args.command.run(&device)
}
