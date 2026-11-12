use std::path::PathBuf;

use bunsen::{
    data::catalog::{
        BunsenCatalogStack,
        CatalogErrorExt,
        CatalogName,
        document::EntryKind,
        lookup::LeafMatch,
    },
    errors::BunsenError,
};
use serde_json::Value;

/// Writes a line of output.
macro_rules! emit {
    ($out:expr, $($arg:tt)*) => {{
        use std::fmt::Write as _;
        writeln!($out, $($arg)*).expect("writing to a String");
    }};
}

pub(super) fn errors(e: BunsenError) -> String {
    let lines = e.problem_lines();
    if lines.is_empty() {
        e.to_string()
    } else {
        lines
    }
}

/// `base` with each of `extensions` imported over it, in order.
pub(super) fn extend(
    mut stack: BunsenCatalogStack,
    extensions: &[PathBuf],
) -> Result<BunsenCatalogStack, String> {
    for path in extensions {
        stack.push_file(path).map_err(errors)?;
    }
    Ok(stack)
}

pub(super) fn parse_kind(kind: &str) -> Result<EntryKind, String> {
    EntryKind::from_name(kind).ok_or_else(|| {
        let known: Vec<&str> = EntryKind::ALL.iter().map(|k| k.as_str()).collect();
        format!("no kind `{kind}`; the kinds are {}", known.join(", "))
    })
}

pub(super) fn find(
    stack: &BunsenCatalogStack,
    query: &str,
    kinds: &[EntryKind],
) -> Result<CatalogName, String> {
    stack
        .find(query, kinds, LeafMatch::Exact)
        .map_err(|e| e.problem_message())
}

pub(super) fn parse_value(text: &str) -> Result<Value, String> {
    serde_saphyr::from_str(text).map_err(|e| format!("`{text}` is not YAML: {e}"))
}
