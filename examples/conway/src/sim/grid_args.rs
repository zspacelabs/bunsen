use bunsen::support::geometry::GridShape2D;

/// Common arguments for grid-related commands.
#[derive(clap::Args, Debug)]
pub struct GridArgs {
    /// The grid shape as `[ WIDTH, HEIGHT ]`, or `X` => `[X, X]`.
    #[arg(long, default_value = "1000")]
    pub grid_shape: GridShape2D,
}
