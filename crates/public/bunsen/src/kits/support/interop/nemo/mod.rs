//! # `nemo`: `NeMo`'s configs and state-dict names, for any kit
//!
//! What NVIDIA's `NeMo` speech models (the Sortformer diarizers, Parakeet)
//! share in their files. The `.nemo` container itself, a tar of a YAML config
//! and a `PyTorch` state dict, is
//! [`data::formats::nemo`](crate::data::formats::nemo) (feature `store`).
//!
//! - **Mirrors:** typed sections of a `NeMo` YAML config
//!   ([`NemoPreprocessorMirror`], [`NemoConformerEncoderMirror`],
//!   [`NemoTransformerEncoderMirror`]), their fields the YAML keys as `NeMo`
//!   spells them, so a field greps to `NeMo`'s source and to a `.nemo`'s
//!   config.
//! - **Conversions:** the bunsen configs a section gives, as the section's
//!   methods: [`NemoPreprocessorMirror::mel_front_end_config`],
//!   [`NemoConformerEncoderMirror::conformer_encoder_config`],
//!   [`NemoTransformerEncoderMirror::transformer_encoder_block_config`] and
//!   [`NemoTransformerEncoderMirror::transformer_encoder_final_norm`].
//! - **Names:** `NeMo`'s state-dict names as bunsen's module paths:
//!   [`conformer_encoder_rekey`] (a `FastConformer`, with its
//!   [`conv_subsampling_rekey`] and [`conformer_layer_rekey`]) and
//!   [`transformer_encoder_rekey`] (with its
//!   [`transformer_encoder_layer_rekey`]), `Rekey`s for the loaders.
//!
//! The `.nemo` fixture in `testdata/nemo/` is a toy Sortformer v2 that `NeMo`
//! saved itself, written by `tools/gen_nemo_fixtures.py`.

mod mirrors;
mod rekey;

#[cfg(all(test, feature = "store"))]
mod archive_tests;

pub use mirrors::*;
pub use rekey::*;
