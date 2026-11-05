//! # Math operations as modules
//!
//! A math operation that a module tree holds as a part, rather than calls:
//! a kind of the [activation seam](crate::blocks::activation), say, where
//! loading relies on each kind being a module of its own.
//!
//! - [`PowOpModule`]: `x^p`, the activation seam's `Pow` kind.
//!
//! Each is named `{Name}OpModule`, with `{Name}OpModuleConfig` and
//! `{Name}OpModuleMeta`, spelled out because `ops` names its value objects
//! `{Name}Op` ([`ClampOp`](crate::ops::clamp::ClampOp)).

mod pow;

pub use pow::*;
