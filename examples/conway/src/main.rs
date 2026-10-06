#![allow(unused)]

pub mod commands;
pub mod sim;

use bunsen::{
    errors::BunsenResult,
    prelude::TensorOpExt,
};
use burn::tensor::{
    Device,
    DeviceConfig,
    FloatDType,
    IntDType,
};
use clap::Parser;
use clap_common::device::{
    DeviceArgs,
    DeviceChoice,
};
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

    let mut device = args.device.init().unwrap_or_else(|e| panic!("{e}"));
    // The boards are 0/1 cells: half-precision floats, and the narrowest
    // ints where the backend has them.
    let config = match args.device.choice() {
        DeviceChoice::Flex => None,
        DeviceChoice::Wgpu => Some(DeviceConfig::default().float_dtype(FloatDType::F16)),
        _ => Some(
            DeviceConfig::default()
                .float_dtype(FloatDType::F16)
                .int_dtype(IntDType::I8),
        ),
    };
    if let Some(config) = config {
        device.configure(config).unwrap_or_else(|e| panic!("{e}"));
    }
    eprintln!("device: {device:?}");

    args.command.run(&device)
}
