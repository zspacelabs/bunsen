//! `pretrained-catalog`: import catalogs, then list, show, resolve and
//! select over the state they make.
//!
//! ```text
//! pretrained-catalog --builtin list --kind build
//! pretrained-catalog --catalog a.yml --catalog b.yml show qwen3
//! pretrained-catalog --builtin resolve large-v3 --preset translate
//! pretrained-catalog --builtin select --select qwen3 --off thinking --limit 1
//! pretrained-catalog --builtin select --select whisper --prefer '{release: latest}' --order -weight --rank
//! pretrained-catalog --builtin releases release
//! ```
//!
//! Catalogs are imported in the order given, `--builtin` (the example
//! catalogs) first when it is given.

use std::{
    path::PathBuf,
    process::ExitCode,
};

use clap::{
    Parser,
    Subcommand,
};
use pretrained_catalog::{
    CatalogErrors,
    CatalogName,
    PretrainedCatalogStack,
    document::{
        CatalogEntry,
        EntryKind,
    },
    fixtures,
    lookup::{
        LeafMatch,
        SEARCHABLE,
    },
    presets::PresetSelection,
    resolve::TagSource,
    select::choose::{
        AxisFlag,
        ChoiceRequest,
        SelectionContext,
    },
};
use serde_json::Value;

/// Writes a line of output.
macro_rules! emit {
    ($out:expr, $($arg:tt)*) => {{
        use std::fmt::Write as _;
        writeln!($out, $($arg)*).expect("writing to a String");
    }};
}

/// Import pretrained catalogs, and query the state they make.
#[derive(Parser, Debug)]
#[command(name = "pretrained-catalog", version)]
struct Cli {
    /// A catalog to import, in order; repeatable. YAML, or JSON by its
    /// `.json` extension.
    #[arg(long = "catalog", value_name = "PATH")]
    catalogs: Vec<PathBuf>,

    /// Import the example catalogs (prelude, commons, qwen3, whisper) first.
    #[arg(long)]
    builtin: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// The imported catalogs, in import order.
    Layers,

    /// Entries, by their shortest unique names.
    List {
        /// Only entries of this kind (`build`, `namespace`, `tag`, ...).
        #[arg(long)]
        kind: Option<String>,
        /// Only the groups of this facet ("all families").
        #[arg(long)]
        facet: Option<String>,
        /// Only the builds that are members of this group.
        #[arg(long)]
        members: Option<String>,
        /// Only the entries directly in this scope.
        #[arg(long)]
        children: Option<String>,
    },

    /// One entry: its kind, its history, and its merged fields.
    Show {
        /// The entry, by any unambiguous name.
        name: String,
    },

    /// A build, resolved: chain, memberships, tags, presets, dependencies.
    Resolve {
        /// The build, by any unambiguous name.
        build: String,
        #[command(flatten)]
        presets: PresetArgs,
        /// A merge patch applied after the presets (YAML or JSON).
        #[arg(long)]
        patch: Option<String>,
    },

    /// Select builds, printed as YAML: the merged query, why, and the
    /// result in order, each build resolved with its weight and presets
    /// (all of it, unless `--limit` says how many). Each flag fills the
    /// query key of its name; patches merge in this order: the defaults,
    /// `--query`s, `--select {..}`s, `--prefer`s, then `--order`, `--cutoff`
    /// and `--limit`; names add to `select.from`.
    Select {
        /// A name (added to `select.from`), or a patch of `select`
        /// (`{require: [hybrid], where: {base: false}}`); repeatable.
        #[arg(long = "select", value_name = "NAME|{..}")]
        selects: Vec<String>,
        /// A patch of `prefer` (`{where: {language: en}, release: latest}`);
        /// repeatable.
        #[arg(long, value_name = "{..}")]
        prefer: Vec<String>,
        /// An order key, most significant first; `-` descends; repeatable.
        #[arg(long, value_name = "KEY", allow_hyphen_values = true)]
        order: Vec<String>,
        /// The least weight a build in the result has.
        #[arg(long, value_name = "W")]
        cutoff: Option<f64>,
        /// How many builds the result keeps; all, when absent.
        #[arg(long, value_name = "N")]
        limit: Option<usize>,
        /// A patch of the whole query (`{select, prefer, order, cutoff,
        /// limit}`); repeatable.
        #[arg(long, value_name = "{..}")]
        query: Vec<String>,
        #[command(flatten)]
        presets: PresetArgs,
        /// The default query, applied unless the only name is one build.
        #[arg(long, value_name = "{..}")]
        defaults: Option<String>,
        /// List every ranked candidate, not only the result; one outside it
        /// carries `excluded: below cutoff` or `excluded: past limit`.
        #[arg(long)]
        rank: bool,
    },

    /// The namespace as a tree, from the root or from one entry.
    Tree {
        /// The entry to start from, by any unambiguous name; the root when
        /// absent.
        root: Option<String>,
        /// Show only entries of these kinds (repeatable), with the
        /// containers on the way to them.
        #[arg(long = "kind", value_name = "KIND")]
        kinds: Vec<String>,
        /// How many levels below the start to show.
        #[arg(long)]
        depth: Option<usize>,
    },

    /// A release group's members, oldest first.
    Releases {
        /// The release group, by any unambiguous name.
        group: String,
    },
}

#[derive(clap::Args, Debug)]
struct PresetArgs {
    /// A preset to select; repeatable.
    #[arg(long = "preset", value_name = "NAME")]
    presets: Vec<String>,
    /// Select the positive member of a bool axis (`--on thinking`).
    #[arg(long = "on", value_name = "AXIS")]
    on: Vec<String>,
    /// Select the negative member of a bool axis (`--off thinking`).
    #[arg(long = "off", value_name = "AXIS")]
    off: Vec<String>,
}

impl PresetArgs {
    fn flags(&self) -> Vec<AxisFlag> {
        let flag = |axis: &String, on: bool| AxisFlag {
            flag: format!("--{} {axis}", if on { "on" } else { "off" }),
            axis: axis.clone(),
            on,
        };
        self.on
            .iter()
            .map(|a| flag(a, true))
            .chain(self.off.iter().map(|a| flag(a, false)))
            .collect()
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut out = String::new();
    let result = run(cli, &mut out);
    print!("{out}");
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `cli`, writing its output to `out`.
fn run(
    cli: Cli,
    out: &mut String,
) -> Result<(), String> {
    let stack = load(&cli)?;
    match cli.command {
        Command::Layers => layers(out, &stack),
        Command::List {
            kind,
            facet,
            members,
            children,
        } => list(out, &stack, kind, facet, members, children)?,
        Command::Show { name } => show(out, &stack, &name)?,
        Command::Resolve {
            build,
            presets,
            patch,
        } => resolve(out, &stack, &build, &presets, patch.as_deref())?,
        Command::Select {
            selects,
            prefer,
            order,
            cutoff,
            limit,
            query,
            presets,
            defaults,
            rank,
        } => {
            let request = request(&selects, &prefer, &order, cutoff, limit, &query, &presets)?;
            select(out, &stack, request, defaults.as_deref(), rank)?
        }
        Command::Tree { root, kinds, depth } => tree(out, &stack, root.as_deref(), &kinds, depth)?,
        Command::Releases { group } => releases(out, &stack, &group)?,
    }
    Ok(())
}

fn errors(e: CatalogErrors) -> String {
    e.to_string()
}

fn load(cli: &Cli) -> Result<PretrainedCatalogStack, String> {
    let mut stack = if cli.builtin {
        fixtures::builtin_stack().map_err(errors)?
    } else {
        PretrainedCatalogStack::new()
    };
    for path in &cli.catalogs {
        stack.push_file(path).map_err(errors)?;
    }
    if stack.layers().is_empty() {
        return Err("no catalogs: give `--catalog PATH` (repeatable) or `--builtin`".into());
    }
    Ok(stack)
}

fn parse_kind(kind: &str) -> Result<EntryKind, String> {
    SEARCHABLE
        .iter()
        .copied()
        .find(|k| k.as_str() == kind)
        .ok_or_else(|| {
            let known: Vec<&str> = SEARCHABLE.iter().map(|k| k.as_str()).collect();
            format!("no kind `{kind}`; the kinds are {}", known.join(", "))
        })
}

fn find(
    stack: &PretrainedCatalogStack,
    query: &str,
    kinds: &[EntryKind],
) -> Result<CatalogName, String> {
    stack
        .find(query, kinds, LeafMatch::Exact)
        .map_err(|e| e.message)
}

fn layers(
    out: &mut String,
    stack: &PretrainedCatalogStack,
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

fn list(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    kind: Option<String>,
    facet: Option<String>,
    members: Option<String>,
    children: Option<String>,
) -> Result<(), String> {
    let mut names: Vec<CatalogName> = if let Some(facet) = facet {
        let facet = find(stack, &facet, &[EntryKind::Facet])?;
        stack
            .facet_groups(&facet)
            .iter()
            .map(|e| e.name().clone())
            .collect()
    } else if let Some(group) = members {
        let group = find(
            stack,
            &group,
            &[
                EntryKind::Namespace,
                EntryKind::Contract,
                EntryKind::Release,
            ],
        )?;
        stack.members_of(&group)
    } else if let Some(scope) = children {
        let scope = find(stack, &scope, &SEARCHABLE)?;
        stack
            .children(&scope)
            .iter()
            .map(|e| e.name().clone())
            .collect()
    } else {
        stack.declarations().map(|e| e.name().clone()).collect()
    };
    if let Some(kind) = kind {
        let kind = parse_kind(&kind)?;
        names.retain(|n| stack.declaration(n).is_some_and(|d| d.kind() == kind));
    }
    for name in names {
        let Some(entry) = stack.declaration(&name) else {
            continue;
        };
        let shown = stack.display_name(&name);
        let facet = stack
            .group_facet(entry)
            .map(|f| format!("  ({f})"))
            .unwrap_or_default();
        emit!(
            out,
            "{:<28} {:<44} {}{facet}",
            shown.short,
            shown.full.to_string(),
            entry.kind()
        );
    }
    Ok(())
}

fn show(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    query: &str,
) -> Result<(), String> {
    let name = find(stack, query, &SEARCHABLE)?;
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
    let kids = stack.children(&name);
    if !kids.is_empty() {
        emit!(out, "children:");
        for kid in kids {
            emit!(out, "  {} ({})", kid.name().leaf(), kid.kind());
        }
    }
    Ok(())
}

fn parse_value(text: &str) -> Result<Value, String> {
    serde_saphyr::from_str(text).map_err(|e| format!("`{text}` is not YAML: {e}"))
}

fn resolve(
    out: &mut String,
    stack: &PretrainedCatalogStack,
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
    if let Some(contract) = &b.contract {
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
    emit!(out, "  dependencies");
    for (key, dep) in &b.dependencies {
        emit!(out, "    {key:<18} {} ({})", dep.resource, dep.entry.file);
    }
    // The flags as shortcuts, then the presets' document.
    let mut presets = args.presets.clone();
    for flag in args.flags() {
        let axis = find(stack, &flag.axis, &[EntryKind::Axis])?;
        let member = b
            .presets
            .bool_member(&axis, flag.on)
            .map_err(|e| format!("`{}` doesn't apply: {}", flag.flag, e.message))?;
        emit!(out, "  `{}` implies `--preset {member}`", flag.flag);
        presets.push(member);
    }
    let selection = PresetSelection {
        presets,
        patch: patch.map(parse_value).transpose()?,
    };
    let applied = b
        .presets
        .select(Value::Object(Default::default()), &selection)
        .map_err(|e| e.message)?;
    emit!(out, "  applied      {}", applied.applied.join(", "));
    emit!(out, "  document     {}", applied.document);
    Ok(())
}

/// The `select` flags, as a request: each fills its query key.
fn request(
    selects: &[String],
    prefer: &[String],
    order: &[String],
    cutoff: Option<f64>,
    limit: Option<usize>,
    query: &[String],
    args: &PresetArgs,
) -> Result<ChoiceRequest, String> {
    let mut request = ChoiceRequest::default();
    for patch in query {
        request = request.query(parse_object(patch)?);
    }
    for select in selects {
        request = if select.trim_start().starts_with('{') {
            request.select(parse_object(select)?)
        } else {
            request.name(select.trim())
        };
    }
    for patch in prefer {
        request = request.prefer(parse_object(patch)?);
    }
    if !order.is_empty() {
        request = request.order(order);
    }
    if let Some(cutoff) = cutoff {
        request = request.cutoff(cutoff);
    }
    if let Some(limit) = limit {
        request = request.limit(limit);
    }
    for preset in &args.presets {
        request = request.preset(preset.clone());
    }
    for axis in &args.on {
        request = request.flag(axis.clone(), true);
    }
    for axis in &args.off {
        request = request.flag(axis.clone(), false);
    }
    Ok(request)
}

fn parse_object(text: &str) -> Result<Value, String> {
    let value = parse_value(text)?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(format!("`{text}` is not a mapping"))
    }
}

fn select(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    request: ChoiceRequest,
    defaults: Option<&str>,
    rank: bool,
) -> Result<(), String> {
    let context = SelectionContext {
        defaults: defaults
            .map(parse_object)
            .transpose()?
            .unwrap_or(Value::Null),
        named: Vec::new(),
    };
    let ranking = context
        .rank(stack, None, &[], &request)
        .map_err(|e| e.message)?;
    let report = ranking.report(rank);
    if report.results.is_empty() {
        return Err(ranking.empty_error().message);
    }
    let yaml = serde_saphyr::to_string(&report).map_err(|e| e.to_string())?;
    emit!(out, "{}", yaml.trim_end());
    Ok(())
}

/// What an entry's tree line adds after its name and kind.
fn annotation(
    stack: &PretrainedCatalogStack,
    name: &CatalogName,
) -> String {
    let Some(entry) = stack.declaration(name) else {
        return String::new();
    };
    let note = match entry.body() {
        CatalogEntry::Namespace(_) | CatalogEntry::Contract(_) => {
            stack.group_facet(entry).map(|f| f.to_string())
        }
        CatalogEntry::Build(_) => stack
            .resolve_build(name)
            .ok()
            .and_then(|b| b.release)
            .map(|r| format!("{} {}", r.group.leaf(), r.released)),
        CatalogEntry::TemplateClass(t) => t.base.as_ref().map(|b| format!("base {b}")),
        CatalogEntry::Resource(r) => Some(r.file.clone()),
        CatalogEntry::Axis(a) => Some(format!("{:?}", a.axis).to_lowercase()),
        CatalogEntry::Tag(t) => Some(format!("{:?}", t.tag_type).to_lowercase()),
        _ => None,
    };
    note.map(|n| format!("  [{n}]")).unwrap_or_default()
}

fn tree(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    root: Option<&str>,
    kinds: &[String],
    depth: Option<usize>,
) -> Result<(), String> {
    let kinds: Vec<EntryKind> = kinds
        .iter()
        .map(|k| parse_kind(k))
        .collect::<Result<_, _>>()?;
    // Whether `name` is shown: it is of a wanted kind, or something below it
    // is.
    fn shown(
        stack: &PretrainedCatalogStack,
        name: &CatalogName,
        kinds: &[EntryKind],
    ) -> bool {
        kinds.is_empty()
            || stack
                .declaration(name)
                .is_some_and(|d| kinds.contains(&d.kind()))
            || stack
                .children(name)
                .iter()
                .any(|c| shown(stack, c.name(), kinds))
    }
    fn walk(
        out: &mut String,
        stack: &PretrainedCatalogStack,
        names: &[CatalogName],
        prefix: &str,
        kinds: &[EntryKind],
        depth: Option<usize>,
    ) {
        let names: Vec<&CatalogName> = names.iter().filter(|n| shown(stack, n, kinds)).collect();
        for (i, name) in names.iter().enumerate() {
            let last = i + 1 == names.len();
            let kind = stack
                .declaration(name)
                .map(|d| d.kind().as_str())
                .unwrap_or("?");
            emit!(
                out,
                "{prefix}{} {} ({kind}){}",
                if last { "└──" } else { "├──" },
                name.leaf(),
                annotation(stack, name)
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
            walk(out, stack, &children, &prefix, kinds, depth.map(|d| d - 1));
        }
    }
    let (start, names): (String, Vec<CatalogName>) = match root {
        Some(query) => {
            let name = find(stack, query, &SEARCHABLE)?;
            let kids = stack
                .children(&name)
                .iter()
                .map(|c| c.name().clone())
                .collect();
            let kind = stack.declaration(&name).map(|d| d.kind()).expect("found");
            (format!("{name} ({kind}){}", annotation(stack, &name)), kids)
        }
        None => (
            "/".to_string(),
            stack
                .declarations()
                .filter(|e| e.name().scope().is_none())
                .map(|e| e.name().clone())
                .collect(),
        ),
    };
    emit!(out, "{start}");
    walk(
        out,
        stack,
        &names,
        "",
        &kinds,
        depth.map(|d| d.saturating_sub(1)),
    );
    Ok(())
}

fn releases(
    out: &mut String,
    stack: &PretrainedCatalogStack,
    query: &str,
) -> Result<(), String> {
    let group = find(stack, query, &[EntryKind::Release])?;
    let members = stack.releases(&group).map_err(|e| e.message)?;
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Runs the command line `args` (after the program name) over the
    /// example catalogs, as files.
    fn cli(args: &str) -> Result<String, String> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/catalogs");
        let mut argv = vec!["pretrained-catalog".to_string()];
        for file in ["prelude", "commons", "qwen3", "whisper", "supplemental"] {
            argv.push("--catalog".into());
            argv.push(format!("{dir}/{file}.yml"));
        }
        argv.extend(shell_words(args));
        let parsed = Cli::try_parse_from(argv).map_err(|e| e.to_string())?;
        let mut out = String::new();
        run(parsed, &mut out)?;
        Ok(out)
    }

    /// `args` split on spaces, keeping single-quoted runs together.
    fn shell_words(args: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut word = String::new();
        let mut quoted = false;
        for c in args.chars() {
            match c {
                '\'' => quoted = !quoted,
                ' ' if !quoted => {
                    if !word.is_empty() {
                        words.push(std::mem::take(&mut word));
                    }
                }
                c => word.push(c),
            }
        }
        if !word.is_empty() {
            words.push(word);
        }
        words
    }

    #[test]
    fn test_layers_and_lists() {
        let out = cli("layers").unwrap();
        assert_eq!(out.lines().count(), 5);
        assert!(out.contains("supplemental"));
        let builds = cli("list --kind build").unwrap();
        assert_eq!(builds.lines().count(), 8);
        let families = cli("list --facet family").unwrap();
        assert!(families.contains("qwen/qwen3") && families.contains("openai/whisper"));
        let members = cli("list --members release").unwrap();
        assert_eq!(members.lines().count(), 3);
        let classes = cli("list --children qwen3 --kind template_class").unwrap();
        assert!(classes.contains("qwen/qwen3/hybrid"));
        assert!(
            cli("list --kind family")
                .unwrap_err()
                .contains("no kind `family`")
        );
    }

    #[test]
    fn test_show_tells_the_history() {
        let out = cli("show qwen").unwrap();
        assert!(out.contains("deny     in commons"));
        assert!(out.contains("update   in supplemental"));
        assert!(out.contains("facet: publisher"));
        assert!(cli("show nothing-here").is_err());
    }

    #[test]
    fn test_resolve_with_shortcuts_and_a_patch() {
        let out = cli(
            "resolve qwen3-0.6b-base --off thinking --patch '{generation: {temperature: 0.1}}'",
        )
        .unwrap();
        assert!(out.contains("(fixed)"));
        assert!(out.contains("`--off thinking` implies `--preset no-thinking`"));
        assert!(out.contains("\"temperature\":0.1"));
        let err = cli("resolve qwen3-0.6b-base --on thinking").unwrap_err();
        assert!(err.contains("fixed at `no-thinking`"), "{err}");
    }

    #[test]
    fn test_select_and_rank() {
        // `select` prints a YAML report; `results` holds each build, in order.
        let report = |args: &str| -> Value {
            let out = cli(args).unwrap();
            serde_saphyr::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"))
        };
        let field = |report: &Value, key: &str| -> Vec<Value> {
            report["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r[key].clone())
                .collect()
        };
        let all = report("select --select qwen3 --off thinking");
        assert_eq!(
            field(&all, "name"),
            [
                "qwen/qwen3/qwen3-0.6b",
                "qwen/qwen3/qwen3-0.6b-base",
                "qwen/qwen3/qwen3-1.7b",
            ]
        );
        assert_eq!(all["results"][0]["presets"], json!(["no-thinking"]));
        assert_eq!(all["results"][0]["weight"], json!(1.0));
        assert_eq!(all["results"][0]["tags"]["thinking"], json!("switch"));
        assert_eq!(all["results"][0]["contract"], json!("qwen3-lm"));
        assert_eq!(
            all["why"][1],
            json!("`--off thinking` is `select.where.thinking: false`")
        );
        assert!(
            all["dropped"][0]
                .as_str()
                .unwrap()
                .contains("qwen3-4b-thinking-2507")
        );
        let one = report("select --select qwen3 --off thinking --limit 1");
        assert_eq!(field(&one, "name"), ["qwen/qwen3/qwen3-0.6b"]);
        assert_eq!(one["query"]["limit"], json!(1));
        let marked = report(
            "select --select whisper --prefer '{release: {step: 0.25}}' --order parameters \
             --cutoff 0.6 --limit 2 --rank",
        );
        assert_eq!(field(&marked, "weight"), [1.0, 0.5, 0.75, 1.0]);
        assert_eq!(
            field(&marked, "excluded"),
            [
                Value::Null,
                json!("below cutoff"),
                Value::Null,
                json!("past limit")
            ]
        );
        assert_eq!(
            marked["results"][3]["release"],
            json!({"group": "openai/whisper/large/release", "released": "2023-11-06"})
        );
        let ranked = report(
            "select --select whisper --prefer '{release: {step: 0.25}}' --order -weight --order \
             parameters --rank",
        );
        let names = field(&ranked, "name");
        assert_eq!(names.len(), 4);
        assert_eq!(names[0], "openai/whisper/large/large-v3-turbo");
        assert_eq!(names[3], "openai/whisper/large/large-v1");
        let dropped = report("select --select whisper --preset translate --rank");
        assert!(
            dropped["dropped"][0]
                .as_str()
                .unwrap()
                .starts_with("`openai/whisper/large/large-v3-turbo`")
        );
        let defaults = report("select --select qwen3 --defaults '{order: [-parameters]}'");
        assert_eq!(
            field(&defaults, "name")[0],
            "qwen/qwen3/qwen3-4b-thinking-2507"
        );
        let err = cli("select --select '{where: {kit: qwen3}}'").unwrap_err();
        assert!(err.contains("every key a query tests is declared"), "{err}");
        let typed = cli("select --select '{where: {base: maybe}}'").unwrap_err();
        assert!(typed.contains("is not `true` or `false`"), "{typed}");
        let empty = cli("select --select qwen3 --cutoff 2").unwrap_err();
        assert!(empty.contains("no build is left"), "{empty}");
        let query = report(
            "select --query '{select: {require: [whisper-asr]}, prefer: {where: {timestamps: \
             true}}}' --order -weight",
        );
        assert_eq!(field(&query, "name")[0], "openai/whisper/large/large-v3");
    }

    #[test]
    fn test_tree() {
        let out = cli("tree").unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "/");
        assert!(out.contains("├── qwen (namespace)  [publisher]"), "{out}");
        assert!(out.contains("qwen3 (namespace)  [family]"), "{out}");
        assert!(
            out.contains("thinking-only (template_class)  [base hybrid]"),
            "{out}"
        );
        assert!(
            out.contains("large-v3 (build)  [release 2023-11-06]"),
            "{out}"
        );
        // Children from the supplemental catalog sit under the group it
        // extends.
        assert!(out.contains("qwen3-1.7b (build)"), "{out}");

        let builds = cli("tree whisper --kind build").unwrap();
        let lines: Vec<&str> = builds.lines().collect();
        assert_eq!(lines[0], "openai/whisper (namespace)  [family]");
        assert!(lines[1].ends_with("large (namespace)"), "{builds}");
        assert_eq!(lines.len(), 6, "{builds}");
        assert!(!builds.contains("checkpoint"));
        assert!(!builds.contains("vocabulary"));

        let shallow = cli("tree --depth 1").unwrap();
        assert!(shallow.contains("qwen (namespace)"));
        assert!(!shallow.contains("qwen3 (namespace)"), "{shallow}");
        assert!(cli("tree --kind nope").is_err());
    }

    #[test]
    fn test_releases() {
        let out = cli("releases release").unwrap();
        let rows: Vec<&str> = out.lines().collect();
        assert_eq!(rows.len(), 3);
        assert!(rows[2].ends_with("(latest)"));
    }

    #[test]
    fn test_needs_catalogs() {
        let parsed = Cli::try_parse_from(["pretrained-catalog", "layers"]).unwrap();
        let err = run(parsed, &mut String::new()).unwrap_err();
        assert!(err.contains("no catalogs"));
    }
}
