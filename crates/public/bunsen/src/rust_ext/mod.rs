//! # Rust-language extensions
//!
//! Helpers that extend the Rust language and its standard library: traits
//! over plain types, array and range arithmetic, source locations. The rest
//! of bunsen builds on them.
//!
//! What belongs here: a Rust-language helper with no tensor or burn
//! dependency. An item in this module depends only on the standard library
//! and plain Rust crates (such as `serde`), never on burn, a backend or a
//! tensor; a helper that needs those belongs with the code that uses it, or
//! in [`support`](crate::support).
//!
//! | Module | What it holds |
//! |--------|---------------|
//! | [`reflection`] | [`LocationDesc`](reflection::LocationDesc), a serializable source location, for audit event headers. **Not** module reflection; see below. |
//! | [`CloneBox`] / [`CloneRef`] | A clonable, downcastable `dyn Any` (inside `DynTensor`); a borrowed-or-owned value (the audit probe's data argument). |
//! | [`arrays`] | [`scalar_to_array`](arrays::scalar_to_array), one value repeated into a `[T; D]`. |
//! | [`range_util`] | [`range_into`](range_util::range_into) and [`shift_range`](range_util::shift_range), `i32` range arithmetic for the Conway kernels. |
//!
//! ## Two modules named `reflection`
//!
//! [`rust_ext::reflection`](reflection) holds only [`LocationDesc`]: it
//! describes a place in Rust source. Module reflection, the XML tree and
//! `XPath` queries over a module's structure, is
//! [`burner::module::reflection`](crate::burner::module::reflection). The two
//! are unrelated.
//!
//! [`LocationDesc`]: reflection::LocationDesc

pub mod arrays;
pub mod range_util;
pub mod reflection;

mod clone_box;
mod clone_ref;

pub use clone_box::*;
pub use clone_ref::*;
