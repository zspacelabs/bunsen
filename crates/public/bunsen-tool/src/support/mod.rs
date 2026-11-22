//! What the command trees share: a lookup expression with presets or a patch
//! added, a row's presets and a checkpoint's row as the listings print them
//! ([`print_model_row`]), the `--preset` flags, the `--format` choice
//! ([`Format`]), an ordered document for it to render ([`Value`]), and "did you
//! mean?" on a failed lookup ([`did_you_mean`]); with `live`, the host's audio
//! devices (`find_audio_device`, `list_audio_devices`) and the microphone,
//! captured at a rate asked for (`Capture`), resampled when the device's
//! differs (bunsen's `ops::signal::resample::StreamingResampler`).

#[cfg(feature = "live")]
mod audio_devices;
#[cfg(feature = "live")]
mod capture;
mod choose_args;
pub mod did_you_mean;
mod model_row;
mod output_format;
mod presets;
mod value;

#[cfg(feature = "live")]
pub use audio_devices::*;
#[cfg(feature = "live")]
pub use capture::*;
pub use choose_args::*;
pub use model_row::*;
pub use output_format::*;
pub use presets::*;
pub use value::*;

#[cfg(test)]
mod tests;
