//! Pretrained models

pub mod load;

#[cfg(feature = "silero-weights")]
pub use bunsen_bundled_silero as bundled;
