//! `tree`, a directory view of the namespace, and `dump`.

use bunsen::data::catalog::{
    BunsenCatalogStack,
    CatalogErrorExt,
    CatalogName,
    document::{
        CatalogEntry,
        EntryKind,
        OrderedNode,
    },
    lookup::SEARCHABLE,
    presets::ResolvedAxisUse,
};
use serde_json::Value;

use super::support::*;

/// The format `dump` writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DumpFormat {
    /// YAML.
    #[value(alias = "yaml")]
    Yml,
    /// JSON.
    Json,
}

pub(super) fn dump(
    out: &mut String,
    stack: &BunsenCatalogStack,
    format: DumpFormat,
) -> Result<(), String> {
    let layers = stack
        .layers()
        .iter()
        .map(|layer| {
            let mut fields = vec![(
                "layer".to_string(),
                OrderedNode::Scalar(Value::String(layer.label().to_string())),
            )];
            if let Some(path) = layer.path() {
                fields.push((
                    "path".to_string(),
                    OrderedNode::Scalar(Value::String(path.display().to_string())),
                ));
            }
            fields.push(("catalog".to_string(), layer.document().to_node()));
            OrderedNode::Map(fields)
        })
        .collect();
    let node = OrderedNode::Seq(layers);
    let text = match format {
        DumpFormat::Yml => node.to_yaml(),
        DumpFormat::Json => node.to_json(),
    }
    .map_err(|e| e.problem_message())?;
    emit!(out, "{}", text.trim_end());
    Ok(())
}

/// Where `tree` shows aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum AliasView {
    /// On the line of the build they name.
    Inline,
    /// As entries, where they sit.
    Entries,
    /// Both.
    Both,
    /// Not at all.
    None,
}

/// The kinds `tree` shows by default: namespaces and builds, and the
/// aliases that name one.
const TREE_KINDS: [EntryKind; 3] = [EntryKind::Namespace, EntryKind::Build, EntryKind::Alias];

/// Whether `tree` shows `name` for `kinds`: an alias only when its target's
/// kind is shown too.
fn tree_shows(
    stack: &BunsenCatalogStack,
    name: &CatalogName,
    kinds: &[EntryKind],
) -> bool {
    let Some(entry) = stack.declaration(name) else {
        return false;
    };
    if !kinds.contains(&entry.kind()) {
        return false;
    }
    match entry.body() {
        CatalogEntry::Alias(a) => kinds.contains(&a.target.kind),
        _ => true,
    }
}

/// An entry's description, when its kind has one and it is written.
fn description(entry: &CatalogEntry) -> Option<&str> {
    match entry {
        CatalogEntry::Namespace(g) => g.description.as_deref(),
        CatalogEntry::Contract(e) => Some(&e.description),
        CatalogEntry::Type(e) => Some(&e.description),
        CatalogEntry::Release(e) => Some(&e.description),
        CatalogEntry::Build(e) => Some(&e.description),
        CatalogEntry::TemplateClass(e) => Some(&e.description),
        CatalogEntry::Tag(e) => Some(&e.description),
        CatalogEntry::Axis(e) => Some(&e.description),
        CatalogEntry::Facet(e) => Some(&e.description),
        CatalogEntry::Resource(e) => e.description.as_deref(),
        CatalogEntry::Alias(e) => e.description.as_deref(),
    }
}

/// Axis uses, `{axis: default|other, ..}`: the default first, then the
/// rest of what the use offers.
fn axes_text(axes: &[ResolvedAxisUse]) -> Option<String> {
    if axes.is_empty() {
        return None;
    }
    let uses: Vec<String> = axes
        .iter()
        .map(|a| {
            let mut offers = vec![a.default.as_str()];
            offers.extend(
                a.offers
                    .iter()
                    .map(String::as_str)
                    .filter(|o| *o != a.default),
            );
            format!("{}: {}", a.axis.name.leaf(), offers.join("|"))
        })
        .collect();
    Some(format!("{{{}}}", uses.join(", ")))
}

/// An alias's name as a build's line shows it: its leaf when it sits in
/// the build's scope or above it, its full name otherwise.
fn alias_text(
    alias: &CatalogName,
    build: &CatalogName,
) -> String {
    let near = match (alias.scope(), build.scope()) {
        (None, _) => true,
        (Some(a), Some(b)) => b == a || b.is_within(&a),
        (Some(_), None) => false,
    };
    if near {
        alias.leaf().to_string()
    } else {
        alias.to_string()
    }
}

/// An entry's tree line, after its tree glyph.
fn tree_line(
    stack: &BunsenCatalogStack,
    name: &CatalogName,
    label: &str,
    inline_aliases: bool,
) -> String {
    let Some(entry) = stack.declaration(name) else {
        return label.to_string();
    };
    let body = entry.body();
    let mut parts: Vec<String> = Vec::new();
    match body {
        CatalogEntry::Build(_) => {
            let aliases: Vec<String> = stack
                .aliases_of(name)
                .iter()
                .map(|a| alias_text(a, name))
                .collect();
            if inline_aliases && !aliases.is_empty() {
                parts.push(aliases.join(", "));
            }
            if let Ok(build) = stack.resolve_build(name) {
                parts.extend(axes_text(&build.presets.axes));
            }
        }
        CatalogEntry::TemplateClass(_) => {
            if let Ok(class) = stack.resolve_template_class(name) {
                parts.extend(axes_text(&class.presets.axes));
            }
        }
        CatalogEntry::Resource(r) => {
            parts.push(r.file.clone());
            if !r.shards.is_empty() {
                parts.push(format!("+{} shards", r.shards.len()));
            }
        }
        CatalogEntry::Contract(c) => {
            let resources: Vec<&str> = c.resources.keys().map(String::as_str).collect();
            let args: Vec<&str> = c.args.keys().map(String::as_str).collect();
            if !resources.is_empty() {
                parts.push(format!("[{}]", resources.join(", ")));
            }
            if !args.is_empty() {
                parts.push(format!("{{{}}}", args.join(", ")));
            }
        }
        CatalogEntry::Alias(a) => {
            let target = stack
                .resolve_ref(name, &a.target.path)
                .map(|t| t.name().clone());
            let shown = target.map_or_else(
                || a.target.path.to_string(),
                |t| {
                    if t.scope() == name.scope() {
                        t.leaf().to_string()
                    } else {
                        t.to_string()
                    }
                },
            );
            return format!("{label} -> {shown}");
        }
        _ => {}
    }
    parts.extend(description(body).map(str::to_string));
    let head = match body {
        CatalogEntry::Contract(c) if c.abstract_ => format!("{label} (contract, abstract)"),
        _ => match body.kind() {
            EntryKind::Namespace | EntryKind::Build => label.to_string(),
            kind => format!("{label} ({kind})"),
        },
    };
    if parts.is_empty() {
        head
    } else {
        format!("{head}: {}", parts.join(" "))
    }
}

pub(super) fn tree(
    out: &mut String,
    stack: &BunsenCatalogStack,
    root: Option<&str>,
    kinds: &[String],
    all: bool,
    aliases: AliasView,
    depth: Option<usize>,
) -> Result<(), String> {
    let mut kinds: Vec<EntryKind> = if all {
        EntryKind::ALL.to_vec()
    } else if kinds.is_empty() {
        TREE_KINDS.to_vec()
    } else {
        kinds
            .iter()
            .map(|k| parse_kind(k))
            .collect::<Result<_, _>>()?
    };
    if matches!(aliases, AliasView::Inline | AliasView::None) {
        kinds.retain(|k| *k != EntryKind::Alias);
    }
    let inline = matches!(aliases, AliasView::Inline | AliasView::Both);
    // Whether `name` is shown: it is of a wanted kind, or something below it
    // is.
    fn shown(
        stack: &BunsenCatalogStack,
        name: &CatalogName,
        kinds: &[EntryKind],
    ) -> bool {
        tree_shows(stack, name, kinds)
            || stack
                .children(name)
                .iter()
                .any(|c| shown(stack, c.name(), kinds))
    }
    fn walk(
        out: &mut String,
        stack: &BunsenCatalogStack,
        names: &[CatalogName],
        prefix: &str,
        kinds: &[EntryKind],
        inline: bool,
        depth: Option<usize>,
    ) {
        let names: Vec<&CatalogName> = names.iter().filter(|n| shown(stack, n, kinds)).collect();
        for (i, name) in names.iter().enumerate() {
            let last = i + 1 == names.len();
            emit!(
                out,
                "{prefix}{} {}",
                if last { "└──" } else { "├──" },
                tree_line(stack, name, name.leaf(), inline)
            );
            if depth == Some(0) {
                continue;
            }
            let children: Vec<CatalogName> = stack
                .children(name)
                .iter()
                .map(|c| c.name().clone())
                .collect();
            let prefix = format!("{prefix}{}", if last { "    " } else { "│   " });
            walk(
                out,
                stack,
                &children,
                &prefix,
                kinds,
                inline,
                depth.map(|d| d - 1),
            );
        }
    }
    let children = |name: &CatalogName| -> Vec<CatalogName> {
        stack
            .children(name)
            .iter()
            .map(|c| c.name().clone())
            .collect()
    };
    let below = depth.map(|d| d.saturating_sub(1));
    match root {
        // One entry heads its subtree.
        Some(query) => {
            let name = find(stack, query, &SEARCHABLE)?;
            emit!(
                out,
                "{}",
                tree_line(stack, &name, &name.to_string(), inline)
            );
            walk(out, stack, &children(&name), "", &kinds, inline, below);
        }
        // The root is not an entry: its entries are the top level.
        None => {
            let top: Vec<CatalogName> = stack
                .declarations()
                .filter(|e| e.name().scope().is_none())
                .map(|e| e.name().clone())
                .filter(|n| shown(stack, n, &kinds))
                .collect();
            // The top level is the first level below the root.
            for name in &top {
                emit!(out, "{}", tree_line(stack, name, name.leaf(), inline));
                if depth.is_none_or(|d| d > 1) {
                    let below = depth.map(|d| d - 2);
                    walk(out, stack, &children(name), "", &kinds, inline, below);
                }
            }
            // `$contracts` is a section of its own, shown when its kinds are.
            let contracts = CatalogName::contracts_root();
            let section = children(&contracts);
            if section.iter().any(|n| shown(stack, n, &kinds)) {
                emit!(out, "$contracts");
                if depth.is_none_or(|d| d > 1) {
                    walk(
                        out,
                        stack,
                        &section,
                        "",
                        &kinds,
                        inline,
                        depth.map(|d| d - 2),
                    );
                }
            }
        }
    }
    Ok(())
}
