//! Parakeet TDT component blocks: the model, its config, and the transducer's
//! two networks.
//!
//! [`ParakeetTdtConfig`] builds a [`ParakeetTdt`]: a
//! [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
//! (a
//! [`ConformerEncoderConfig`](crate::blocks::transformers::encoders::conformer::ConformerEncoderConfig)'s
//! `FastConformer`) with an
//! encoder projection, a [`TdtPredictionNetwork`] and a [`TdtJoint`]. Both
//! answer [`ParakeetTdtMeta`]. The prediction network's recurrent state is a
//! [`TdtPredictionState`] the caller opens per decode, never on the model.

mod joint;
mod model;
mod prediction;

pub use joint::*;
pub use model::*;
pub use prediction::*;
