//! Pretrained Whisper Models.

#[cfg(all(feature = "store_pytorch", feature = "cache"))]
mod load_named;
mod prefabs;
mod providers;
#[cfg(feature = "cache")]
mod vocab;

#[cfg(all(feature = "store_pytorch", feature = "cache"))]
#[doc(inline)]
pub use load_named::*;
#[doc(inline)]
pub use prefabs::*;
#[doc(inline)]
pub use providers::*;
#[cfg(feature = "cache")]
#[doc(inline)]
pub use vocab::*;

#[cfg(feature = "store_pytorch")]
mod pytorch_utils;

#[cfg(feature = "store_pytorch")]
#[doc(inline)]
pub use pytorch_utils::*;

#[cfg(all(feature = "whisper-weights", feature = "store_pytorch"))]
mod load;
