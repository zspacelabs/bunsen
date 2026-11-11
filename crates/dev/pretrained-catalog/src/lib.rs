//! The pretrained catalog's runtime logic, on its own: how a catalog is read,
//! validated, queried and resolved. Probing, building, caching and model
//! loading are out of its scope.
//!
//! - [`document`]: the catalog document ([`PretrainedCatalog`]) and the serde
//!   form it is read from and written to, as YAML or JSON. Every entry names
//!   its kind; groups and builds may nest their children.
//! - [`layer`]: one loaded catalog, flattened to full names.
//! - [`stack`]: [`PretrainedCatalogStack`], an ordered list of layers. Each new
//!   layer is checked against the layers below it; queries fall through the
//!   list. A group written in several places has one declaring occurrence and
//!   any number of extending occurrences.
//! - [`resolve`]: [`ResolvedBuild`], a build with its template-class chain,
//!   memberships, facets, tags, presets and dependencies resolved.
//! - [`presets`]: folding preset sets along a chain (axis uses narrow, never
//!   widen), and selecting from the result.
//! - [`lookup`]: finding entries by shorter names, and naming them for people.
//! - [`select`]: choosing a build by its tags, names and presets, with axis
//!   flags such as `--think` as explained shortcuts into presets.
//! - [`fixtures`]: the example catalogs in `catalogs/`, for tests and examples.
//! - [`release_time`], [`tags`], [`merge_patch`], [`name`]: the value types and
//!   rules the rest is built from.
//!
//! [`usage`] is the guide: the API from importing catalogs to choosing a
//! build, and the select language.
//!
//! # Example
//!
//! ```
//! use pretrained_catalog::{
//!     PretrainedCatalogStack,
//!     presets::PresetSelection,
//! };
//! use serde_json::json;
//!
//! let mut stack = PretrainedCatalogStack::new();
//! stack
//!     .push_yaml(
//!         "prelude",
//!         r#"
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
//! demo-lm: { $kind: contract, $description: "What the demo loader reads." }
//! demo:
//!   $kind: namespace
//!   $facet: family
//!   $description: "A demo family."
//!   thinker:
//!     $kind: build
//!     $description: "Always thinks."
//!     $join: [demo-lm]
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
//! assert!(off.unwrap_err().message.contains("fixed at `thinking`"));
//! ```

pub mod document;
pub mod error;
pub mod fixtures;
pub mod layer;
pub mod lookup;
pub mod merge_patch;
pub mod name;
pub mod presets;
pub mod release_time;
pub mod resolve;
pub mod select;
pub mod stack;
pub mod tags;

#[doc = include_str!("../docs/usage.md")]
pub mod usage {}

pub use document::PretrainedCatalog;
pub use error::{
    CatalogError,
    CatalogErrorKind,
    CatalogErrors,
};
pub use name::{
    CatalogName,
    CatalogRef,
};
pub use resolve::ResolvedBuild;
pub use stack::PretrainedCatalogStack;
