//! Parakeet's front end: `NeMo`'s mel front end, by Parakeet's names.

/// Parakeet's feature extractor: `NeMo`'s mel front end, whose defaults are
/// `parakeet-tdt-0.6b-v3`'s.
pub type ParakeetFrontEnd = crate::kits::speech::support::nemo::NemoMelFrontEnd;

/// Config for [`ParakeetFrontEnd`]: `NeMo`'s, its defaults
/// `parakeet-tdt-0.6b-v3`'s.
pub type ParakeetFrontEndConfig = crate::kits::speech::support::nemo::NemoMelFrontEndConfig;

/// Parakeet's front end, streamed.
pub type ParakeetFeatures = crate::kits::speech::support::nemo::NemoMelFeatures;

pub use crate::kits::speech::support::nemo::NemoMelFrontEndMeta as ParakeetFrontEndMeta;
