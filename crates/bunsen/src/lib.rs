#![cfg_attr(feature = "wgpu", recursion_limit = "512")]
//!# bunsen burn(er)
#![warn(missing_docs)]

extern crate alloc;

extern crate core;

/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;

#[allow(dead_code)]
pub mod compat;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;

#[cfg(feature = "train")]
pub mod training;

#[cfg(feature = "cache")]
pub use bunsen_cache as cache;

pub mod errors;
pub mod functional;
pub mod meta;
pub mod modules;
pub mod nn;
pub mod record;
pub mod utility;
pub mod zspace;

#[doc(inline)]
pub use bunsen_contracts as contracts;

#[cfg(test)]
cfg_select! {
    feature = "cuda" => {
        /// Selected burn backend for unittests.
        pub type BunsenTestBackend = burn::backend::Cuda;
    }
    feature = "metal" => {
        /// Selected burn backend for unittests.
        pub type BunsenTestBackend = burn::backend::Metal;
    }
    feature = "wgpu" => {
        /// Selected burn backend for unittests.
        pub type BunsenTestBackend = burn::backend::Wgpu;
    }
    _ => {
        /// Selected burn backend for unittests.
        pub type BunsenTestBackend = burn::backend::Flex;
    }
}
