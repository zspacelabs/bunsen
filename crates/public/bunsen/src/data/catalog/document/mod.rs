//! The catalog document: [`BunsenCatalog`] and its entries, with the
//! keyed form they are read from and written to (YAML or JSON).
//!
//! The document is the namespace tree. **Every key the catalog language
//! defines starts with `$`; a bare key is a name.**
//!
//! ```yaml
//! $description: "Qwen3 chat models."
//! qwen:                                  # an entry, keyed by its segment
//!   $kind: namespace
//!   $merge: update
//!   qwen3-0.6b:                          # a child: a map with `$kind`
//!     $kind: build
//!     $description: "Qwen3-0.6B."
//!     $template_class: qwen3/hybrid
//!     parameters: 600000000              # a tag
//!     checkpoint: {$kind: resource, $file: model.safetensors, $format: safetensors, $sources: []}
//! ```
//!
//! - **An entry** is a map with `$kind`, keyed by its segment. Its name is its
//!   path of keys: nesting is the only placement.
//! - **Bare keys under an entry:** under a namespace or a contract, children;
//!   under a template class, tags; under a build, a child when the value is a
//!   map with `$kind`, and a tag otherwise. Other kinds have none.
//! - **`$contracts`**, at the root of the base catalog only, holds the
//!   contracts and their `type` entries: the interfaces bunsen's loaders
//!   implement. A build names its contract with `$contract`.
//! - **`$presets`** holds `$common`, `$axes` (keyed by the axis), `$default`,
//!   and the presets, keyed by name.
//! - **Foreign documents are data:** a preset's `$patch` and `$common` keep the
//!   model's keys.
//! - **Refused:** a `$` word the kind doesn't define, a child without `$kind`,
//!   a bare key where the kind has none, and a repeated key.
//! - **Order is the file's:** `entries`, children, presets and axis uses keep
//!   the order they are written in.
//!
//! A reference to another entry ([`CatalogRef`](super::name::CatalogRef))
//! resolves relative to the entry it is written in, climbing the namespace to
//! the root; a leading `/` resolves from the root only.

mod alias;
mod axis;
mod build;
mod catalog;
mod contract;
mod entry;
mod group;
mod keyed;
mod preset;
mod release;
mod resource;
mod support;
mod template_class;
#[cfg(test)]
mod tests;
mod types;

pub use alias::*;
pub use axis::*;
pub use build::*;
pub use catalog::*;
pub use contract::*;
pub use entry::*;
pub use group::*;
pub use keyed::OrderedNode;
pub use preset::*;
pub use release::*;
pub use resource::*;
pub use template_class::*;
pub use types::*;
