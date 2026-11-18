//! # Mixtures of experts
//!
//! The parts of a sparse mixture-of-experts feed-forward:
//!
//! - [`experts`]: [`ExpertStack`](experts::ExpertStack), a packed stack of
//!   `SwiGLU` experts, run through the dispatch seam
//!   ([`expert_dispatch`](crate::ops::transformers::moe::expert_dispatch)).
//! - [`router`]: [`TopKRouter`](router::TopKRouter), each row's `k` experts and
//!   their weights, with an optional input norm, choice bias, renormalization
//!   and scale.
//! - [`sparse`]: [`SparseMoe`](sparse::SparseMoe), the feed-forward: the router
//!   and the experts, in a latent when it has one;
//!   [`TransformerFeedForward::Moe`](super::mlp::TransformerFeedForward::Moe)
//!   in a decoder layer.

pub mod experts;
pub mod router;
pub mod sparse;
