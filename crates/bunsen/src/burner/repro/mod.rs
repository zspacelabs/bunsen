//! # Standalone reproductions of upstream `burn` bugs.
//!
//! Backend-generic, dependency-free harnesses for defects in `burn` itself,
//! written so they can be lifted out of this crate and taken upstream
//! unchanged. Each one uses only `burn`'s public tensor surface.
//!
//! ## How this differs from the pins
//!
//! These two modules assert *opposite* things, deliberately:
//!
//! | | asserts | fails when |
//! |---|---|---|
//! | `burner::repro` | the **correct** semantics | the bug is present |
//! | `burner::tensor::burn_behavior` | the **current** behavior | the bug is fixed |
//!
//! The pins keep bunsen's own suite green while guaranteeing that a fix
//! announces itself and names the workarounds it makes redundant. The
//! reproductions here are the bug report: run one against a candidate fix and
//! a pass means the fix works.
//!
//! So the functions here **fail on affected backends today**. That is their
//! job, and it is why they are not wired into the default test suite — the
//! tests that drive them are `#[ignore]`d.
//!
//! ## Running them
//!
//! ```text
//! cargo test -p bunsen --lib --features wgpu -- burner::repro --ignored --nocapture
//! ```

pub mod unfold;
