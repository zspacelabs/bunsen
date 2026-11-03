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
//! - [`decoding`]: the search machinery, from a step's logits to the ids a
//!   decode returns: logit filters, greedy and beam search, sampling, ranking,
//!   and the temperature ladder.

pub mod decoding;
pub mod tokens;
