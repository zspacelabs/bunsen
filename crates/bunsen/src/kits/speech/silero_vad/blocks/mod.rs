//! Silero VAD blocks

mod container;
mod module;

#[doc(inline)]
pub use container::*;
#[doc(inline)]
pub use module::*;

#[cfg(any(test, feature = "testing"))]
pub mod reference;
