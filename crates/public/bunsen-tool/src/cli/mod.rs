//! `bunsen-tool`'s command trees, one per branch.

pub mod asr;
pub mod card;
pub mod chat;

use bunsen::{
    errors::BunsenResult,
    kits::support::construct::LookupExpr,
};

/// `model`, a lookup expression, with `presets` added to its box.
///
/// # Errors
/// As [`LookupExpr::parse`].
pub fn with_presets(
    model: &str,
    presets: &[&str],
) -> BunsenResult<String> {
    let mut expr = LookupExpr::parse(model)?;
    expr.selection
        .presets
        .extend(presets.iter().map(|p| p.to_string()));
    Ok(expr.to_string())
}
