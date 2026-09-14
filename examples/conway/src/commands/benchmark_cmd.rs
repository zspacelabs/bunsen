use std::time::{
    Duration,
    Instant,
};

use bunsen::{
    errors::BunsenResult,
    kits::sims::conway::{
        life2d::{
            ConwayLife2DConfig,
            ConwayLife2DState,
        },
        life3d::{
            ConwayLife3DConfig,
            ConwayLife3DState,
        },
        util::{
            ConwayRules,
            ConwaySim,
        },
    },
    support::{
        geometry::GridShape2D,
        testing::backend_device,
    },
};
use burn::prelude::Backend;
use clap::Parser;
use clap_common::logging::LogArgs;
use indicatif::ProgressBar;

/// Conway's Game of Life benchmark for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct BenchmarkCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The grid shape as `[ WIDTH, HEIGHT ]`, or `X` => `[X, X]`.
    #[arg(long, default_value = "200")]
    pub grid_shape: GridShape2D,

    /// The number of dimensions.
    ///
    /// Shape is taken as `[WIDTH, HEIGHT, HEIGHT]`.
    #[arg(long, default_value_t = 2)]
    pub dims: usize,

    /// The number of steps to run.
    #[arg(long, default_value_t = 1000)]
    pub steps: usize,

    /// Initial fuzz factor for randomizing state.
    #[arg(long, default_value_t = 0.2)]
    pub initial_fuzz: f64,

    /// The fraction of steps to use for warmup.
    #[arg(long, default_value_t = 30)]
    pub warmup_fraction: usize,

    /// Show progress bar.
    #[arg(long, default_value_t = false)]
    pub progress: bool,
}

impl BenchmarkCmd {
    pub fn run<B: Backend>(&self) -> BunsenResult<()> {
        let device = backend_device::<B>();
        self.logging.init(None);

        match self.dims {
            2 => self.run2d::<B>(&device),
            3 => self.run3d::<B>(&device),
            _ => panic!("unsupported dims"),
        }
    }

    fn run2d<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<()> {
        eprintln!("2D Shape: {:?}", self.grid_shape.as_width_height());

        let mut sim: ConwayLife2DState<B> = ConwayLife2DConfig {
            shape: self.grid_shape,
        }
        .init(device);
        sim.fuzz(self.initial_fuzz);

        self.time_steps(&mut || sim.step())?;

        Ok(())
    }

    fn run3d<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<()> {
        let depth = self.grid_shape.height;

        let shape3d: [usize; 3] = [self.grid_shape.width, self.grid_shape.height, depth];
        eprintln!("3D Shape: {:?}", shape3d);

        let mut sim: ConwayLife3DState<B> = ConwayLife3DConfig {
            shape: shape3d,
            rules: ConwayRules::default(),
        }
        .init(device);
        sim.fuzz(self.initial_fuzz);

        self.time_steps(&mut || sim.step())?;

        Ok(())
    }

    fn time_steps<F: FnMut()>(
        &self,
        f: &mut F,
    ) -> BunsenResult<(usize, Duration)> {
        let mut t0: Instant = Instant::now();
        let bar = if self.progress {
            Some(ProgressBar::new(self.steps as u64))
        } else {
            None
        };

        let warmup = self.steps / self.warmup_fraction;
        let steps = self.steps;

        for step in 0..self.steps {
            if step == warmup {
                t0 = Instant::now();
            }
            f();

            if let Some(bar) = &bar {
                bar.inc(1);
            }
        }

        let t1: Instant = Instant::now();
        if let Some(bar) = &bar {
            bar.finish();
        }

        let counted_steps = steps - warmup;
        let elapsed = t1 - t0;

        let step_rate = counted_steps as f64 / elapsed.as_secs_f64();
        eprintln!("Warmup Steps: {warmup:?}");
        eprintln!("Counted Steps: {counted_steps:?}");
        eprintln!("Counted Time: {elapsed:0.2?}");
        eprintln!("{:.2} steps/sec", step_rate);

        Ok((counted_steps, elapsed))
    }
}
