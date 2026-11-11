//! The example catalogs in `catalogs/`, compiled in, and the stacks they
//! make. The tests and the examples use them.

use crate::{
    error::CatalogErrors,
    stack::PretrainedCatalogStack,
};

/// `catalogs/prelude.yml`: facets, registered axes and tags, and the
/// template-class root.
pub const PRELUDE: &str = include_str!("../catalogs/prelude.yml");

/// `catalogs/commons.yml`: the shared publishers and contracts.
pub const COMMONS: &str = include_str!("../catalogs/commons.yml");

/// `catalogs/qwen3.yml`: Qwen3 chat builds.
pub const QWEN3: &str = include_str!("../catalogs/qwen3.yml");

/// `catalogs/whisper.yml`: Whisper builds and the `whisper-large` releases.
pub const WHISPER: &str = include_str!("../catalogs/whisper.yml");

/// `catalogs/supplemental.yml`: a supplemental catalog with one Qwen3 build.
pub const SUPPLEMENTAL: &str = include_str!("../catalogs/supplemental.yml");

/// The built-in stack: prelude, commons, qwen3, whisper.
///
/// # Errors
/// Whatever a push finds, which would be a bug in the fixtures.
pub fn builtin_stack() -> Result<PretrainedCatalogStack, CatalogErrors> {
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
/// As [`builtin_stack`].
pub fn supplemented_stack() -> Result<PretrainedCatalogStack, CatalogErrors> {
    let mut stack = builtin_stack()?;
    stack.push_yaml("supplemental", SUPPLEMENTAL)?;
    Ok(stack)
}
