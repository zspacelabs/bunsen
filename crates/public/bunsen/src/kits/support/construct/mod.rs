//! # Construction: from a resolved pretrained to what a kit builds
//!
//! The construct side of the pretrained pathway; all but [`LmConstruct`]
//! needs the `cache` feature. [`data::pretrained`] resolves a name to a row
//! and its resource bundle, and brings the bundle's files local; what turns
//! those files into an in-memory construct is here:
//!
//! - [`ConstructInit`]: the kit's hook, which completes a resolved ref's
//!   resource bundle before it is loaded and builds from the loaded parts.
//! - [`Deferred`]: a model not yet loaded, its ref paired with the hook that
//!   will build it.
//! - [`PretrainedFactory`]: a kit's providers in search order, resolving lookup
//!   expressions to [`Deferred`] models.
//! - [`LookupExpr`]: a lookup expression, `[provider:]name[preset, …]{patch}`,
//!   split into the name and what it asks of the row's presets.
//! - [`Loaded`]: the handle the pathway hands back.
//! - [`LmConstruct`]: what a language model's pretrained loads to, its model
//!   and its tokenizer and generation files, with the [`LmPolicy`] its presets
//!   resolved to.
//!
//! [`data::pretrained`]: crate::data::pretrained

#[cfg(feature = "cache")]
mod deferred;
#[cfg(feature = "cache")]
mod factory;
#[cfg(feature = "cache")]
mod init;
mod lm_construct;
#[cfg(feature = "cache")]
mod load_policy;
#[cfg(feature = "cache")]
mod lookup;

#[cfg(feature = "cache")]
pub use deferred::*;
#[cfg(feature = "cache")]
pub use factory::*;
#[cfg(feature = "cache")]
pub use init::*;
pub use lm_construct::*;
#[cfg(feature = "cache")]
pub use load_policy::*;
#[cfg(feature = "cache")]
pub use lookup::*;
