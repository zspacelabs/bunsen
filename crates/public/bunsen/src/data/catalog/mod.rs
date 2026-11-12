//! # Build catalogs
//!
//! The catalog of things bunsen can build, as a stack of catalog layers: how
//! a catalog is read, validated, queried and resolved, and how a build is
//! chosen from it. Probing, building, caching and model loading are out of
//! its scope.
//!
//! - [`document`]: the catalog document ([`BunsenCatalog`]) and the serde form
//!   it is read from and written to, as YAML or JSON. Every entry names its
//!   kind; groups and builds may nest their children.
//! - [`layer`]: one loaded catalog, flattened to full names.
//! - [`stack`]: [`BunsenCatalogStack`], an ordered list of layers. Each new
//!   layer is checked against the layers below it; queries fall through the
//!   list. A group written in several places has one declaring occurrence and
//!   any number of extending occurrences.
//! - [`resolve`]: [`ResolvedBuild`], a build with its template-class chain,
//!   memberships, facets, tags, presets and dependencies resolved.
//! - [`presets`]: folding preset sets along a chain (axis uses narrow, never
//!   widen), and selecting from the result.
//! - [`lookup`]: finding entries by shorter names, and naming them for people.
//! - [`select`]: the selection pipeline: a query of stages over weighted rows,
//!   composed from a tool's preset prefix, its defaults and a request. Axis
//!   flags such as `--think` are explained shortcuts into presets.
//! - [`builtin`]: the built-in catalog, `bunsen-catalog`'s layers, imported
//!   ([`builtin_stack`]).
//! - `testing::fixtures` (the `testing` feature): example catalogs, for tests,
//!   examples and these docs.
//! - [`release_time`], [`tags`], [`name`]: the value types and rules the rest
//!   is built from. Preset patches are [merge
//!   patches](crate::support::merge_patch).
//!
//! [`catalog_format`] describes the catalog document and the stack, and
//! [`usage`] is the guide: the API from importing catalogs to choosing a build,
//! and the select language.
//!
//! # Example
//!
//! ```
//! use bunsen::data::catalog::{
//!     BunsenCatalogStack,
//!     presets::PresetSelection,
//! };
//! use serde_json::json;
//!
//! let mut stack = BunsenCatalogStack::new();
//! stack
//!     .push_yaml(
//!         "prelude",
//!         r#"
//! $contracts:
//!   demo-lm: { $kind: contract, $description: "What the demo loader reads." }
//! family: { $kind: facet, $description: "A line of related models." }
//! thinking: { $kind: axis, $axis: bool, $description: "Think first." }
//! template_class_root:
//!   $kind: template_class
//!   $description: "The root of every chain."
//!   $required_facets: [family]
//! "#,
//!     )
//!     .unwrap();
//! stack
//!     .push_yaml(
//!         "models",
//!         r#"
//! demo:
//!   $kind: namespace
//!   $facet: family
//!   $description: "A demo family."
//!   thinker:
//!     $kind: build
//!     $description: "Always thinks."
//!     $contract: demo-lm
//!     $presets:
//!       $axes: { thinking: { $offers: [thinking], $default: thinking } }
//!       thinking: { $description: "Think.", $patch: { think: true } }
//! "#,
//!     )
//!     .unwrap();
//!
//! let name = stack.find_build("thinker").unwrap();
//! let build = stack.resolve_build(&name).unwrap();
//! assert_eq!(build.tag("thinking"), Some(&json!("always")));
//! assert_eq!(build.tag("family"), Some(&json!("demo")));
//! assert_eq!(build.tag("contract"), Some(&json!("demo-lm")));
//! let off = build
//!     .presets
//!     .check_selection(&PresetSelection::of(["no-thinking"]));
//! assert!(off.unwrap_err().message().contains("fixed at `thinking`"));
//! ```

pub mod builtin;
pub mod document;
pub mod error;
pub mod layer;
mod loader_form;
pub mod lookup;
pub mod name;
pub mod presets;
pub mod release_time;
pub mod resolve;
pub mod select;
pub mod stack;
pub mod tags;

#[cfg(feature = "testing")]
pub mod testing;

#[doc = include_str!("catalog_format.md")]
pub mod catalog_format {}

#[doc = include_str!("usage.md")]
pub mod usage {}

pub use builtin::{
    BuiltinCatalog,
    builtin_stack,
};
pub use document::BunsenCatalog;
pub use error::{
    CatalogErrorExt,
    CatalogProblem,
    CatalogRule,
    catalog_error,
    catalog_errors,
};
pub use name::{
    CatalogName,
    CatalogRef,
};
pub use resolve::ResolvedBuild;
pub use stack::BunsenCatalogStack;
