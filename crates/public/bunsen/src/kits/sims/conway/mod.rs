//! # Conway's Game of Life
//!
//! [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life),
//! on 2D and 3D boolean boards that wrap at the edges.
//!
//! # Lifecycle
//!
//! - [`life2d`]: [`ConwayLife2DConfig`](life2d::ConwayLife2DConfig) builds a
//!   [`ConwayLife2DState`](life2d::ConwayLife2DState) on a device.
//! - [`life3d`]: [`ConwayLife3DConfig`](life3d::ConwayLife3DConfig) builds a
//!   [`ConwayLife3DState`](life3d::ConwayLife3DState) on a device.
//!
//! A new board is all dead. Seed it with
//! [`ConwaySim::fuzz`](util::ConwaySim::fuzz), or by writing its `state`
//! tensor (the 2D state also has `read_slice` and `write_slice`), then call
//! [`ConwaySim::step`](util::ConwaySim::step) to advance one generation in
//! place.
//!
//! The 2D rule is fixed: the standard B3/S23, a dead cell with three live
//! neighbours is born, and a live cell with two or three survives. The 3D
//! sim takes its rule from a [`ConwayRules`](util::ConwayRules), the
//! neighbour counts that spawn a cell and that keep one alive; its default
//! is B3/S23 too.
//!
//! # Kernels and wrapping
//!
//! [`ops`] has the step as functions from a board to the next, in 2D and
//! 3D. [`next_interior_2d`](ops::next_interior_2d) and
//! [`next_interior_3d`](ops::next_interior_3d) count each cell's `3x3` (or
//! `3x3x3`) neighbourhood with `unfold` and apply the rule, which yields
//! the board less its outer ring.
//!
//! That ring is how the board wraps. A `[H, W]` board holds a
//! `[H-2, W-2]` torus in its interior, and its outer ring is a halo: each
//! halo cell is a copy of the interior cell on the opposite edge.
//! [`project_wrapped_toroidal_boarders`](ops::project_wrapped_toroidal_boarders)
//! rewrites the halo from the interior, and
//! [`next_state_wrapped_2d`](ops::next_state_wrapped_2d) and
//! [`next_state_wrapped_3d`](ops::next_state_wrapped_3d) are the whole
//! step: the next interior, then a fresh halo. A step reads the halo as it
//! stands, so the sims keep it fresh between steps: `fuzz` and
//! `write_slice` rewrite it after they change the board. A caller who
//! writes a board's `state` tensor directly must do the same, with
//! `state.inplace(project_wrapped_toroidal_boarders)`. The rewrite replaces
//! whatever was written in the halo, so write the interior.
//!
//! # Example
//!
//! ```rust
//! use bunsen::{
//!     kits::sims::conway::{
//!         life2d::{
//!             ConwayLife2DConfig,
//!             ConwayLife2DState,
//!         },
//!         util::ConwaySim,
//!     },
//!     support::{
//!         geometry::GridShape2D,
//!         testing::cpu_device,
//!     },
//! };
//! use burn::prelude::s;
//!
//! let device = cpu_device();
//!
//! // A 7x7 board: a 5x5 torus inside a one-cell halo.
//! let mut life: ConwayLife2DState =
//!     ConwayLife2DConfig::new(GridShape2D::square(7)).init(&device);
//!
//! // A blinker: three in a row turn to three in a column.
//! life.write_slice(s![3, 2..5], vec![vec![true, true, true]]);
//! life.step();
//! assert_eq!(
//!     life.read_slice(s![2..5, 3]),
//!     vec![vec![true], vec![true], vec![true]]
//! );
//! ```

pub mod life2d;
pub mod life3d;
pub mod ops;
pub mod util;
