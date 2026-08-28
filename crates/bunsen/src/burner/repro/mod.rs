//! # Reproductions of defects in `burn` and its ecosystem.
//!
//! A home for backend-generic harnesses that pin bugs living outside this
//! crate. Each is written so that the behaviour under test rests only on
//! public API, keeping the reproduction portable: taking one upstream should
//! mean swapping this crate's test helpers, not rewriting the test. A module
//! may also pin this crate's own workaround for the defect, which is not
//! portable and does not travel with it.
//!
//! Each module documents one defect, and carries two kinds of test:
//!
//! | | asserts | fails when |
//! |---|---|---|
//! | the reproduction | the **correct** semantics | the bug is present |
//! | the behaviour pin | the **current** behaviour | the bug is fixed |
//!
//! The reproduction is `#[ignore]`d so the suite stays green; the behaviour
//! pin is not, so a fix announces itself and names the workarounds it makes
//! redundant rather than silently turning them into new bugs.

#[cfg(feature = "store")]
pub mod pytorch_strided_weights;

pub mod unfold;
