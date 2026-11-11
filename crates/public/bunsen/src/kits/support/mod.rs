//! # Kit support: what kits of more than one family share
//!
//! Not a model: the machinery that kits across families build on. A module
//! shared within one family stays with that family
//! ([`language::driver`](crate::kits::language::driver),
//! `speech::testing`, behind the `testing` feature); one shared across
//! families is here.
//!
//! - [`tokens`]: the token seams, [`Detokenizer`](tokens::Detokenizer) (ids to
//!   text) and [`Tokenizer`](tokens::Tokenizer) (text to ids, and back), their
//!   implementations, and the vocabulary readers behind them.
//! - [`construct`]: the construct side of the pretrained pathway: what a
//!   language model's pretrained loads to; and, with the feature `cache`, the
//!   hook a kit builds through, deferred models, the factory, and the loaded
//!   handle.
//! - [`decoding`]: the search machinery, from a step's logits to the ids a
//!   decode returns: logit filters, greedy and beam search, sampling, ranking,
//!   and the temperature ladder.
//! - [`catalog_contracts`]: each kit's build-catalog contract, the signature
//!   its loader implements, gathered for the catalog builder (with `store`, as
//!   the pretrained indexes are).
//! - [`model_config`]: the config side, apart from building: reading the
//!   configs a checkpoint ships with, such as a `transformers` `config.json`
//!   ([`TransformersConfigJson`](model_config::TransformersConfigJson)).

#[cfg(feature = "store")]
pub mod catalog_contracts;
#[cfg(all(test, feature = "store", feature = "speech"))]
mod catalog_equivalence;
pub mod construct;
pub mod decoding;
pub mod model_config;
pub mod tokens;
