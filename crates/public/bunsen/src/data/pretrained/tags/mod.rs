//! # Tags: what a pretrained model is, as facts a selection can read
//!
//! A build carries tags ([`ResolvedBuild::tags`]): its catalog's typed facts
//! about the model (`thinking: "switch"`, `language: ["en", "*"]`,
//! `parameters: 600000000`). Tags are distinct from presets: a preset is a
//! delta over the config a model is built from; a tag is a fact about the
//! model, for listings and for choosing among models.
//!
//! [`ResolvedBuild::tags`]: crate::data::catalog::ResolvedBuild::tags
//!
//! The keys form a loosely closed, shared vocabulary. Shared keys are bare
//! (`language`); a key only one kit uses is prefixed with the kit's name and a
//! dot (`whisper.multilingual`). Values are any JSON; each key's shape is a
//! convention of the vocabulary. Nothing at run time refuses a tag.
//!
//! ## The vocabulary
//!
//! Each shared key is a [`TagDef`]: its key, its description, the shape of
//! its value ([`TagShape`]), and who writes it ([`TagAuthor`]). They are
//! data and docs; bunsen runs no checks of them.
//!
//! | key | shape | written by |
//! |---|---|---|
//! | [`kit`](KIT) | text: the kit's name | the catalog, and a factory |
//! | [`groups`](GROUPS) | a list of names: the variation groups the model is part of | the row |
//! | [`version`](VERSION) | `{group, version}`: a release in a sequence | the row |
//! | [`language`](LANGUAGE) | a list of canonical BCP 47 tags; `"*"` for unspecified others | the row |
//! | [`parameters`](PARAMETERS) | a count: total parameters, tied weights once | the row |
//! | [`active_parameters`](ACTIVE_PARAMETERS) | a count: per-token active parameters (mixture of experts) | the row |
//! | [`dtype`](DTYPE) | `f32`, `f16` or `bf16`: the stored weights' element type | the catalog, else the row |
//! | [`scores`](SCORES) | `{name: number}` | the row |
//! | [`reasoning`](REASONING) | bool: trained to reason | the row |
//! | [`thinking`](THINKING) | `always`, `switch` or `never`: a thinking block, and whether it switches | the row |
//! | [`base`](BASE) | bool: a pretrained model, not tuned to follow instructions | the row |
//! | [`translate`](TRANSLATE) | bool: it can translate | the row |

mod vocabulary;

pub use vocabulary::*;
