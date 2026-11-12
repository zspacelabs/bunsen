//! What the command trees share: a lookup expression with presets or a patch
//! added, a row's presets and a checkpoint's row as the listings print them
//! ([`print_model_row`]), the `--preset` flags, the `--format` choice
//! ([`Format`]), and "did you mean?" on a failed lookup ([`did_you_mean`]).

mod choose_args;
pub mod did_you_mean;
mod model_row;
mod output_format;
mod presets;

pub use choose_args::*;
pub use model_row::*;
pub use output_format::*;
pub use presets::*;

#[cfg(test)]
mod tests;
