//! Whisper Component Blocks: the model, its configs, and the conventions
//! a checkpoint is read under.
//!
//! [`WhisperApiConfig`] describes a model as upstream's `ModelDimensions`
//! does, and lowers through
//! [`ToStructureConfig`](crate::burner::module::ToStructureConfig)
//! to a [`WhisperStructureConfig`] (an [`AudioEncoderConfig`] and a
//! [`TextDecoderConfig`]), which
//! [`ModuleInit`](crate::burner::module::ModuleInit) builds into a [`Whisper`].
//! Both answer [`WhisperMeta`]. [`WhisperGeometry`] is the comparable core of
//! an API config: what a checkpoint fixes, and what a prefab promises.
//!
//! A checkpoint records its weights, and through their shapes the geometry;
//! it does not record the conventions it was trained under. Two configs on
//! the model declare them, defaulting to upstream's:
//! [`WhisperFrontEndConfig`], the audio front end (rate, mel grid, clamp
//! range), and [`WhisperTokenLayoutConfig`], the token layout (languages,
//! base vocabularies, spellings, timestamp grid). Everything the driver
//! derives in samples or ids comes from those two, not from constants.
//!
//! The text decoder's incremental decode keeps its state in a
//! [`TextDecoderCache`] the caller opens per decode, never on the model.

/// Default Whisper Head Dimensionality.
pub const WHISPER_DEFAULT_D_MODEL: usize = 64;

mod audio_encoder;
mod decoder_block;
mod encoder_block;
mod front_end;
mod geometry;
mod text_decoder;
mod token_layout;
mod whisper_model;

pub use audio_encoder::*;
pub use decoder_block::*;
pub use encoder_block::*;
pub use front_end::*;
pub use geometry::*;
pub use text_decoder::*;
pub use token_layout::*;
pub use whisper_model::*;
