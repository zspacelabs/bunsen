//! # Construction: from a resolved pretrained to what a kit builds
//!
//! The construct side of the pretrained pathway. [`data::pretrained`]
//! resolves a name to a row and its resource bundle, and brings the bundle's
//! files local; what turns those files into an in-memory construct is here:
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
//!
//! [`data::pretrained`]: crate::data::pretrained

mod deferred;
mod factory;
mod init;
mod load_policy;
mod lookup;

pub use deferred::*;
pub use factory::*;
pub use init::*;
pub use load_policy::*;
pub use lookup::*;
