use crate::sim::grid_args::GridArgs;

#[derive(clap::Args, Debug)]
pub struct SimArgs {
    #[clap(flatten)]
    pub grid: GridArgs,

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 10)]
    pub init_skip_steps: usize,

    /// The initial density of the grid.
    #[arg(long, default_value_t = 0.1)]
    pub initial_density: f64,

    /// The noise to apply to the grid on each step.
    #[arg(long, default_value_t = 0.0001)]
    pub update_noise: f64,
}
