use bunsen::data::catalog::{
    BunsenCatalogStack,
    CatalogErrorExt,
    document::EntryKind,
    lookup::SEARCHABLE,
};

use super::support::*;

pub(super) fn layers(
    out: &mut String,
    stack: &BunsenCatalogStack,
) {
    for (i, layer) in stack.layers().iter().enumerate() {
        let path = layer
            .path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(built in)".into());
        emit!(
            out,
            "{i:>2}  {:<14} {:>3} occurrences  {path}",
            layer.label(),
            layer.entries().len()
        );
    }
}

pub(super) fn show(
    out: &mut String,
    stack: &BunsenCatalogStack,
    query: &str,
) -> Result<(), String> {
    let mut kinds = SEARCHABLE.to_vec();
    kinds.push(EntryKind::Contract);
    let name = find(stack, query, &kinds)?;
    let entry = stack.declaration(&name).expect("found");
    emit!(out, "{name}  ({})", entry.kind());
    for change in entry.history() {
        emit!(
            out,
            "  {:<8} in {}",
            format!("{:?}", change.merge).to_lowercase(),
            stack.layers()[change.layer].label()
        );
    }
    let yaml = serde_saphyr::to_string(entry.body()).map_err(|e| e.to_string())?;
    emit!(out, "{yaml}");
    let aliases = stack.aliases_of(&name);
    if !aliases.is_empty() {
        emit!(out, "aliases:");
        for alias in aliases {
            emit!(out, "  {alias}");
        }
    }
    let kids = stack.children(&name);
    if !kids.is_empty() {
        emit!(out, "children:");
        for kid in kids {
            emit!(out, "  {} ({})", kid.name().leaf(), kid.kind());
        }
    }
    Ok(())
}

pub(super) fn releases(
    out: &mut String,
    stack: &BunsenCatalogStack,
    query: &str,
) -> Result<(), String> {
    let group = find(stack, query, &[EntryKind::Release])?;
    let members = stack.releases(&group).map_err(|e| e.problem_message())?;
    let latest = members.last().map(|(n, _)| n.clone());
    for (build, released) in &members {
        let mark = if Some(build) == latest.as_ref() {
            "  (latest)"
        } else {
            ""
        };
        emit!(out, "{:<12} {build}{mark}", released.to_string());
    }
    Ok(())
}
