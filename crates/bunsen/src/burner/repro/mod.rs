//! # Reproductions of defects in `burn` and its ecosystem.
//!
//! A home for backend-generic harnesses that pin bugs living outside this
//! crate. Each is written to be lifted out and taken upstream unchanged, so
//! they use only public API.
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

pub mod pytorch_strided_weights;
