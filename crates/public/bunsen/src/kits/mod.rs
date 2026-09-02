//! # Domain-Purpose Complete Kits
//!
//! `bunsen::kits` collects *complete* domain implementations &mdash; full
//! models, families of models, and runnable simulations &mdash; built on
//! top of the lower-level building blocks elsewhere in the crate
//! ([`crate::blocks`], [`crate::ops`], [`crate::contracts`]).
//!
//! Whereas [`crate::blocks`] is a library of reusable sub-module
//! components, each kit here is a *whole* thing you can pick up and use:
//! a model architecture with pretrained weight loaders, a GPT variant
//! ready to fine-tune, a simulation kernel you can drop into a scene.
//!
//! ## Current kits
//!
//! - [`bimm`] &mdash; *Bunsen/Burn Image Models.* An incremental port of the [`timm`](https://github.com/huggingface/pytorch-image-models)
//!   ecosystem to `burn`. Currently includes the `ResNet` family with
//!   pretrained-weight loaders, and the Swin Transformer V2 family.
//! - [`gpts`] &mdash; Full GPT / LLM variants. Currently includes `NanoChat`, a
//!   compact GPT suitable for experimentation and fine-tuning.
//! - [`sims`] &mdash; Iterative tensor simulations. Currently includes Conway's
//!   Game of Life in 2D and 3D, and a D2Q9 lattice-Boltzmann fluid solver.
//! - [`speech`] &mdash; Full speech recognition models. Currently includes
//!   `Whisper` and `Hubert`.
//! - [`tokens`] &mdash; What kits share on the token side: the ids-to-text
//!   seam, and its `wordchipper` implementation.
//!
//! ## Stability
//!
//! Kits are added incrementally and their surfaces may evolve faster
//! than the rest of the crate while their respective domains are being
//! filled in. Pin a specific `bunsen` version when you need a stable
//! API.

pub mod bimm;
pub mod gpts;
pub mod sims;
pub mod speech;
pub mod tokens;
