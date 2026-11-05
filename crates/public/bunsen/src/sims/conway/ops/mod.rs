//! Functional operations for Conway's Game of Life, in 2D and 3D.
//!
//! Each is a function from a board tensor to a board tensor:
//!
//! - the step: [`next_interior_2d`] and [`next_interior_3d`] compute the next
//!   interior, and [`next_state_wrapped_2d`] and [`next_state_wrapped_3d`] the
//!   whole next board, halo included;
//! - the halo: [`project_wrapped_toroidal_boarders`] rewrites a board's outer
//!   ring from the opposite edges of its interior, in any rank;
//! - noise: [`fuzz_state`], [`fuzz_state_2d`] and [`fuzz_state_3d`] flip each
//!   cell with a given probability.
//!
//! The sims in [`life2d`](super::life2d) and [`life3d`](super::life3d) are
//! these kernels, applied in place. The [`conway`](super) docs explain the
//! halo.

mod advance;
mod fuzz_state;
mod wrap_state;

pub use advance::*;
pub use fuzz_state::*;
pub use wrap_state::*;
