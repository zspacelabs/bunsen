#![allow(unused)]

pub mod commands;
pub mod sim;

use bunsen::{
    errors::BunsenResult,
    prelude::TensorOpExt,
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
    pub fn run(&self) -> BunsenResult<()> {
        match self {
            Commands::Visual(cmd) => cmd.run(),
            Commands::Benchmark(cmd) => cmd.run(),
        }
    }
}

fn main() -> BunsenResult<()> {
    let args = Args::parse();
    cfg_select! {
        feature = "cuda" => {
            eprintln!("CUDA enabled");
        }
        feature = "metal" => {
            eprintln!("Metal enabled");
        }
        feature = "vulkan" => {
            eprintln!("Vulkan enabled");
        }
        feature = "wgpu" => {
            eprintln!("WGPU enabled");
        }
        feature = "flex" => {
            eprintln!("Flex enabled");
        }
        _ => {
            compile_error!("No backend selected");
        }
    }

    args.command.run()
}
