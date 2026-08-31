//! # Reproductions of defects in `burn` and its ecosystem.
//!
//! A home for backend-generic harnesses that pin bugs living outside bunsen.
//! Each is written so the behaviour under test rests only on public API,
//! keeping the reproduction portable: taking one upstream should mean swapping
//! the test helpers, not rewriting the test. A module may also pin bunsen's own
//! workaround for the defect, which is not portable and does not travel with
//! it.
//!
//! Each module documents one defect, and carries two kinds of test:
//!
//! | | asserts | fails when |
//! |---|---|---|
//! | the reproduction | the **correct** semantics | the bug is present |
//! | the behaviour pin | the **current** behaviour | the bug is fixed |
//!
//! The reproduction is `#[ignore]`d so the suite stays green; the behaviour pin
//! is not, so a fix announces itself and names the workarounds it makes
//! redundant rather than silently turning them into new bugs.
//!
//! ## Why this is not in `bunsen`
//!
//! It used to be, as `bunsen::burner::repro`. A reproduction is a statement
//! about somebody else's code: it carries a fixture, it wants a real
//! accelerator to say anything, and its whole purpose is to stop being true.
//! None of that belongs on a published library's public surface.
//!
//! ## Running
//!
//! ```sh
//! cargo test --release -p burn_bug_repro --features wgpu
//! ```
//!
//! A backend feature is optional but nearly always wanted. Without one,
//! `bunsen::support::testing::PerformanceBackend` resolves to `Flex` (CPU) —
//! which is *correct* for the `unfold` defect, so the behaviour pin is gated
//! off rather than reporting a fix that has not happened. The CPU-correctness
//! checks still run.

pub mod pytorch_strided_weights;
pub mod unfold;
