use bunsen::data::catalog::{
    BunsenCatalogStack,
    CatalogErrorExt,
    CatalogName,
    document::EntryKind,
    presets::PresetSelection,
    resolve::TagSource,
};
use serde_json::Value;

use super::{
    PresetArgs,
    support::*,
};

pub(super) fn resolve(
    out: &mut String,
    stack: &BunsenCatalogStack,
    query: &str,
    args: &PresetArgs,
    patch: Option<&str>,
) -> Result<(), String> {
    let name = find(stack, query, &[EntryKind::Build])?;
    let b = stack.resolve_build(&name).map_err(errors)?;
    emit!(out, "{}  {}", b.name, b.description);
    let chain: Vec<String> = b.template_chain.iter().map(ToString::to_string).collect();
    emit!(out, "  chain        {}", chain.join(" > "));
    let groups: Vec<String> = b.memberships.iter().map(ToString::to_string).collect();
    emit!(out, "  memberships  {}", groups.join(", "));
    if let Some(contract) = b.contract.as_ref().and_then(CatalogName::contract_path) {
        emit!(out, "  contract     {contract}");
    }
    if let Some(release) = &b.release {
        emit!(
            out,
            "  release      {} ({})",
            release.group,
            release.released
        );
    }
    emit!(out, "  tags");
    for (tag, value) in &b.tags {
        let source = match &value.source {
            TagSource::Written(owner) => format!("written by {owner}"),
            TagSource::Axis => "implied by the axis".into(),
            TagSource::Facet => "implied by the facet".into(),
            TagSource::Contract => "implied by the contract".into(),
        };
        emit!(
            out,
            "    {:<30} {:<28} {source}",
            tag.to_string(),
            value.value.to_string()
        );
    }
    emit!(out, "  axes");
    for using in &b.presets.axes {
        let fixed = if using.is_fixed() { "  (fixed)" } else { "" };
        emit!(
            out,
            "    {:<30} offers {:?}, default {}{fixed}",
            using.axis.name.to_string(),
            using.offers,
            using.default
        );
    }
    let names: Vec<&str> = b.presets.presets.iter().map(|p| p.name.as_str()).collect();
    emit!(out, "  presets      {}", names.join(", "));
    emit!(out, "  resources");
    for (slot, bound) in &b.resources {
        emit!(
            out,
            "    {slot:<18} {} ({})",
            bound.resource,
            bound.entry.file
        );
    }
    if !b.args.is_empty() {
        emit!(out, "  args");
        for (slot, bound) in &b.args {
            emit!(out, "    {slot:<18} {}", bound.value);
        }
    }
    // The flags as shortcuts, then the presets' document.
    let mut presets = args.presets.clone();
    for (axis, on) in args
        .on
        .iter()
        .map(|a| (a, true))
        .chain(args.off.iter().map(|a| (a, false)))
    {
        let flag = format!("--{} {axis}", if on { "on" } else { "off" });
        let name = find(stack, axis, &[EntryKind::Axis])?;
        let member = b
            .presets
            .bool_member(&name, on)
            .map_err(|e| format!("`{flag}` doesn't apply: {}", e.problem_message()))?;
        emit!(out, "  `{flag}` implies `--preset {member}`");
        presets.push(member);
    }
    let selection = PresetSelection {
        presets,
        patch: patch.map(parse_value).transpose()?,
    };
    let applied = b
        .presets
        .select(Value::Object(Default::default()), &selection)
        .map_err(|e| e.problem_message())?;
    emit!(out, "  applied      {}", applied.applied.join(", "));
    emit!(out, "  document     {}", applied.document);
    Ok(())
}
