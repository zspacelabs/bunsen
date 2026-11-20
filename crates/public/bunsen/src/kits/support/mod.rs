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
//! - [`construct`]: what a language model's pretrained loads to
//!   ([`LmConstruct`](construct::LmConstruct)).
//! - [`decoding`]: the search machinery, from a step's logits to the ids a
//!   decode returns: logit filters, greedy and beam search, sampling, ranking,
//!   and the temperature ladder.
//! - [`pretrained`] (feature `store`, as the kits' pretrained indexes are): the
//!   shared pretrained tier: each kit's build-catalog contract
//!   ([`catalog_contracts`](pretrained::catalog_contracts)), and the construct
//!   hook the `transformers` language models share
//!   ([`TransformersLmKit`](pretrained::TransformersLmKit)).
//! - [`interop`]: what an external ecosystem's files and names mean, for any
//!   kit: a `transformers` `config.json`
//!   ([`TransformersConfigJson`](interop::transformers::TransformersConfigJson)).
//! - [`model_config`]: the config side, apart from building and from an
//!   upstream's files: presets, generation policy, the catalog's probes.
//! - [`text`]: text normalization ([`TextNormalizer`](text::TextNormalizer),
//!   the chain, the normalizers bunsen ships) and language tags
//!   ([`BunsenLanguageTag`](text::BunsenLanguageTag)).

pub mod construct;
pub mod decoding;
pub mod interop;
pub mod model_config;
#[cfg(feature = "store")]
pub mod pretrained;
pub mod text;
pub mod tokens;
