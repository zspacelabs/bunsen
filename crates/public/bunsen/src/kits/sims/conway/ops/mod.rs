//! Functional operations for Conway's Game of Life: 2D

mod advance;
mod fuzz_state;
mod wrap_state;

#[doc(inline)]
pub use advance::*;
#[doc(inline)]
pub use fuzz_state::*;
#[doc(inline)]
pub use wrap_state::*;
