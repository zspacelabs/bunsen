#![allow(unused)]

pub mod commands;
pub mod sim;

use bunsen::{
    errors::BunsenResult,
    prelude::{
        TensorElemOpExt,
        TensorOpExt,
    },
};
use burn::prelude::Backend;
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
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        match self {
            Commands::Visual(cmd) => cmd.run::<B>(),
            Commands::Benchmark(cmd) => cmd.run::<B>(),
        }
    }
}

fn main() {
    let args = Args::parse();
    cfg_select! {
        feature = "cuda" => {
            eprintln!("CUDA enabled");
            type B = burn::backend::Cuda<burn::tensor::f16, i8>;
        }
        feature = "metal" => {
            eprintln!("Metal enabled");
            type B = burn::backend::Metal<burn::tensor::f16, i8>;
        }
        feature = "vulkan" => {
            eprintln!("Vulkan enabled");
            type B = burn::backend::Vulkan<burn::tensor::f16, i8>;
        }
        feature = "wgpu" => {
            eprintln!("WGPU enabled");
            type B = burn::backend::Wgpu<burn::tensor::f16>;
        }
        feature = "flex" => {
            eprintln!("Flex enabled");
            type B = burn::backend::Flex;
        }
        _ => {
            complie_error!("No backend selected");
        }
    }

    args.command.run::<B>();
}
