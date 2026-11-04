//! # Kit support: what kits of more than one family share
//!
//! Not a model: the machinery that kits across families build on. A module
//! shared within one family stays with that family
//! ([`gpts::driver_support`](crate::kits::gpts::driver_support),
//! `speech::testing`, behind the `testing` feature); one shared across
//! families is here.
//!
//! - [`tokens`]: the ids-to-text seam, [`Detokenizer`](tokens::Detokenizer),
//!   and the vocabulary readers behind it.
//! - [`construct`]: the construct side of the pretrained pathway: what a
//!   language model's pretrained loads to; and, with the feature `cache`, the
//!   hook a kit builds through, deferred models, the factory, and the loaded
//!   handle.
//! - [`decoding`]: the search machinery, from a step's logits to the ids a
//!   decode returns: logit filters, greedy and beam search, sampling, ranking,
//!   and the temperature ladder.
//! - [`model_config`]: the config side, apart from building: reading the
//!   configs a checkpoint ships with, such as a `transformers` `config.json`
//!   ([`HfConfigJson`](model_config::HfConfigJson)).

pub mod construct;
pub mod decoding;
pub mod model_config;
pub mod tokens;
