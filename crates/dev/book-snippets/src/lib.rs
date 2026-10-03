//! Code the bunsen book includes.
//!
//! The book (`book/`) shows code only by `{{#include}}`-ing it from compiled
//! sources, so an example cannot drift from the API. Each file here holds one
//! snippet between `// ANCHOR: name` and `// ANCHOR_END: name` markers, inside
//! a function a test calls, so `cargo test` runs it as well as compiling it.
//!
//! A chapter includes a snippet with
//! `{{#include ../../../crates/dev/book-snippets/src/<file>.rs:<name>}}`.
//! Rename or remove an anchor only together with the chapters that include it.

pub mod first_use;
