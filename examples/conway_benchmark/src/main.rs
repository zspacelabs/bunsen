#![allow(unused)]
use std::time::Instant;

use bunsen::kits::sims::conway::{
    life2d::{
        ConwayLife2DConfig,
        ConwayLife2DState,
    },
    life3d::{
        ConwayLife3DConfig,
        ConwayLife3DState,
        LifeRules,
    },
};
use burn::prelude::{
    Backend,
    s,
};
use clap::Parser;
use indicatif::ProgressBar;

/// Conway's Game of Life benchmark for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The number of steps to run.
    #[arg(long, default_value = "1000")]
    pub steps: usize,

    /// The number of dimensions.
    #[arg(long, default_value_t = 2)]
    pub dims: usize,

    /// The width and height of the grid.
    #[arg(long, default_value = "100")]
    pub grid_size: usize,

    /// The fraction of steps to use for warmup.
    #[arg(long, default_value_t = 10)]
    pub warmup_fraction: usize,

    /// Show progress bar.
    #[arg(short, long, default_value_t = false)]
    pub progress: bool,
}

fn main() {
    let args = Args::parse();
    println!("{:#?}", args);

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu<burn::tensor::f16, i8>>(&args);

    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda<burn::tensor::f16, i8>>(&args);

    #[cfg(feature = "metal")]
    run::<burn::backend::Metal<burn::tensor::f16, i8>>(&args);

    #[cfg(feature = "flex")]
    run::<burn::backend::Flex>(&args);
}

fn run<B: Backend>(args: &Args) {
    match args.dims {
        2 => run2d::<B>(args),
        3 => run3d::<B>(args),
        _ => panic!("unsupported dims"),
    }
}

fn run2d<B: Backend>(args: &Args) {
    let device = Default::default();

    let warmup = args.steps / args.warmup_fraction;

    let mut conway: ConwayLife2DState<B> = ConwayLife2DConfig {
        shape: [args.grid_size, args.grid_size],
    }
    .init(&device);
    conway.fuzz(0.2);

    let mut t0: Instant = Instant::now();
    let bar = if args.progress {
        Some(ProgressBar::new(args.steps as u64))
    } else {
        None
    };

    for step in 0..args.steps {
        if step == warmup {
            t0 = Instant::now();
        }
        conway.step();

        if let Some(bar) = &bar {
            bar.inc(1);
        }
    }
    // Force final observation.
    conway.state.clone().slice(s![0, 0]).into_scalar();

    let t1: Instant = Instant::now();
    if let Some(bar) = &bar {
        bar.finish();
    }

    let step_rate = (args.steps - warmup) as f64 / (t1 - t0).as_secs_f64();
    println!("{:.2} steps/sec", step_rate);
}

fn run3d<B: Backend>(args: &Args) {
    let device = Default::default();

    let warmup = args.steps / args.warmup_fraction;

    let mut conway: ConwayLife3DState<B> = ConwayLife3DConfig {
        shape: [args.grid_size, args.grid_size, args.grid_size],
        rules: LifeRules::default(),
    }
    .init(&device);
    conway.fuzz(0.2);

    let mut t0: Instant = Instant::now();
    let bar = if args.progress {
        Some(ProgressBar::new(args.steps as u64))
    } else {
        None
    };

    for step in 0..args.steps {
        if step == warmup {
            t0 = Instant::now();
        }
        conway.step();

        if let Some(bar) = &bar {
            bar.inc(1);
        }
    }
    // Force final observation.
    conway.state.clone().slice(s![0, 0, 0]).into_scalar();

    let t1: Instant = Instant::now();
    if let Some(bar) = &bar {
        bar.finish();
    }

    let step_rate = (args.steps - warmup) as f64 / (t1 - t0).as_secs_f64();
    println!("{:.2} steps/sec", step_rate);
}
