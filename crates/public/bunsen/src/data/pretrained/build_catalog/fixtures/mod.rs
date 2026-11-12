//! Example catalogs, compiled in, and the stacks they make: what the tests,
//! the examples and the [`usage`](super::usage) guide run against.

use crate::data::pretrained::build_catalog::{
    error::CatalogResult,
    stack::PretrainedCatalogStack,
};

/// `prelude.yml`: facets, registered axes and tags, and the
/// template-class root.
pub const PRELUDE: &str = include_str!("prelude.yml");

/// `commons.yml`: the shared publishers and contracts.
pub const COMMONS: &str = include_str!("commons.yml");

/// `qwen3.yml`: Qwen3 chat builds.
pub const QWEN3: &str = include_str!("qwen3.yml");

/// `whisper.yml`: Whisper builds and the `whisper-large` releases.
pub const WHISPER: &str = include_str!("whisper.yml");

/// `supplemental.yml`: a supplemental catalog with one Qwen3 build.
pub const SUPPLEMENTAL: &str = include_str!("supplemental.yml");

/// The built-in stack: prelude, commons, qwen3, whisper.
///
/// # Errors
/// Whatever a push finds, which would be a bug in the fixtures.
pub fn example_stack() -> CatalogResult<PretrainedCatalogStack> {
    let mut stack = PretrainedCatalogStack::new();
    for (label, yaml) in [
        ("prelude", PRELUDE),
        ("commons", COMMONS),
        ("qwen3", QWEN3),
        ("whisper", WHISPER),
    ] {
        stack.push_yaml(label, yaml)?;
    }
    Ok(stack)
}

/// The built-in stack with the supplemental catalog on top.
///
/// # Errors
/// As [`example_stack`].
pub fn supplemented_example_stack() -> CatalogResult<PretrainedCatalogStack> {
    let mut stack = example_stack()?;
    stack.push_yaml("supplemental", SUPPLEMENTAL)?;
    Ok(stack)
}
