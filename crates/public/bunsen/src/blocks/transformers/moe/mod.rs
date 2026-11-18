//! # Mixtures of experts
//!
//! The parts of a sparse mixture-of-experts feed-forward:
//!
//! - [`experts`]: [`ExpertStack`](experts::ExpertStack), a packed stack of
//!   `SwiGLU` experts, run through the dispatch seam
//!   ([`expert_dispatch`](crate::ops::transformers::moe::expert_dispatch)).

pub mod experts;
