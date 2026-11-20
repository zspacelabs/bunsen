//! The `interop` check: external-format code in `bunsen::kits` lives in
//! `interop` modules.
//!
//! Three rules, over every non-test item of the tree it scans
//! (`crates/public/bunsen/src/kits` by default):
//!
//! - **`mirror`:** a type named `*Mirror` (a typed record of an upstream file's
//!   fields) sits in an `interop` module.
//! - **`rekey`:** a function that returns a `Rekey`, or is named
//!   `load_*_tensors` / `load_*_weights`, sits in an `interop` module. A trait
//!   impl's method (`CanonicalKeys::canonical_rekey`, a module's own names) is
//!   bunsen's, and not checked.
//! - **`store-gate`:** a `feature = "store"` cfg sits in an `interop` or a
//!   `pretrained` module: model modules carry no `store` gate.
//!
//! An item is in a module named `interop` (or `pretrained`) when any segment
//! of its module path is: a directory, a file, or an inline `mod`. Test code
//! is exempt: a file named `*tests.rs`, a `testing` or `tests` module, an
//! item under `#[cfg(test)]` (or a `cfg` that names `test`), and a `#[test]`.
//!
//! A fourth rule, **`mirror-rename`**, tracks the upstream mirrors whose
//! names don't say so yet (`LayaAgentConfig`): the
//! check can't find a mirror by its name until it is renamed, so these
//! entries are listed by hand, and one holds while its type still exists in
//! its file. A renamed (or moved) type makes its entry stale.
//!
//! Findings the migration hasn't reached are listed in the allowlist
//! (`tools/interop_allowlist.yml`), each by rule, file and item. The check
//! fails on a finding the allowlist doesn't hold, and on an entry that no
//! longer matches one; `--bless` rewrites the allowlist from the findings.

use std::{
    collections::BTreeSet,
    fmt,
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use quote::ToTokens;
use serde::Deserialize;
use syn::{
    Attribute,
    ImplItem,
    Item,
    ReturnType,
    Signature,
};

/// The `interop` check's arguments.
#[derive(Debug, clap::Args)]
pub struct InteropArgs {
    /// The tree to scan.
    #[arg(long, default_value = "crates/public/bunsen/src/kits")]
    pub root: PathBuf,

    /// The allowlist: findings the migration hasn't reached.
    #[arg(long, default_value = "tools/interop_allowlist.yml")]
    pub allowlist: PathBuf,

    /// Rewrite the allowlist from the findings instead of checking them.
    #[arg(long)]
    pub bless: bool,
}

/// What a finding breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Rule {
    /// A `*Mirror` type outside an `interop` module.
    Mirror,

    /// A rekey or a loader outside an `interop` module.
    Rekey,

    /// A `feature = "store"` cfg outside an `interop` or `pretrained` module.
    StoreGate,

    /// An upstream mirror not yet named `*Mirror`: listed by hand, held
    /// while the type exists in its file.
    MirrorRename,
}

impl Rule {
    /// The rule's name, as the allowlist spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mirror => "mirror",
            Self::Rekey => "rekey",
            Self::StoreGate => "store-gate",
            Self::MirrorRename => "mirror-rename",
        }
    }

    /// What breaking it means, for the report.
    pub fn explain(self) -> &'static str {
        match self {
            Self::Mirror => "a `*Mirror` type outside an `interop` module",
            Self::Rekey => "a rekey or loader outside an `interop` module",
            Self::StoreGate => "a `store` gate outside an `interop` or `pretrained` module",
            Self::MirrorRename => "an upstream mirror not yet named `*Mirror`",
        }
    }
}

/// One item that breaks a rule: also an allowlist entry.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct Finding {
    /// The rule it breaks.
    pub rule: Rule,

    /// Its file, as given to the check (relative to the repository root).
    pub path: String,

    /// The item: a type or a function (`Type::method` for an inherent
    /// method), or the module a gate sits on (`mod name`).
    pub item: String,
}

impl fmt::Display for Finding {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "{}: {} ({}: {})",
            self.path,
            self.item,
            self.rule.name(),
            self.rule.explain()
        )
    }
}

/// Runs the check; `Ok(false)` when it fails.
///
/// # Errors
/// When the tree or the allowlist can't be read or parsed.
pub fn run(args: &InteropArgs) -> Result<bool, String> {
    let scanned = scan(&args.root)?;
    let allowed = if args.bless && !args.allowlist.exists() {
        BTreeSet::new()
    } else {
        read_allowlist(&args.allowlist)?
    };
    let (renames, discovered): (BTreeSet<Finding>, BTreeSet<Finding>) = allowed
        .into_iter()
        .partition(|f| f.rule == Rule::MirrorRename);
    let held = |f: &&Finding| scanned.types.contains(&(f.path.clone(), f.item.clone()));
    if args.bless {
        let mut entries = scanned.findings.clone();
        entries.extend(renames.iter().filter(held).cloned());
        fs::write(&args.allowlist, render_allowlist(&entries))
            .map_err(|e| format!("{}: {e}", args.allowlist.display()))?;
        println!(
            "interop: wrote {} entries to {}",
            entries.len(),
            args.allowlist.display()
        );
        return Ok(true);
    }
    let new: Vec<_> = scanned.findings.difference(&discovered).collect();
    let mut stale: Vec<_> = discovered.difference(&scanned.findings).collect();
    stale.extend(renames.iter().filter(|f| !held(f)));
    for finding in &new {
        println!("new: {finding}");
    }
    for entry in &stale {
        println!("stale allowlist entry (fixed, so remove it): {entry}");
    }
    let ok = new.is_empty() && stale.is_empty();
    println!(
        "interop: {} findings and {} mirrors to rename allowlisted; {} new, {} stale",
        discovered.len(),
        renames.len(),
        new.len(),
        stale.len()
    );
    Ok(ok)
}

/// The allowlist's entries.
///
/// # Errors
/// When the file can't be read or isn't a list of findings.
pub fn read_allowlist(path: &Path) -> Result<BTreeSet<Finding>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let entries: Vec<Finding> =
        serde_saphyr::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(entries.into_iter().collect())
}

/// The allowlist file for `findings`.
pub fn render_allowlist(findings: &BTreeSet<Finding>) -> String {
    let mut out = String::from(
        "# Findings of `cargo run -p source-checks -- interop` that the interop\n\
         # migration (scratch/kit-interop/NOTES.md) hasn't reached yet. Each kit's\n\
         # layer removes its own; the check fails on a stale entry. Regenerate\n\
         # with `--bless`.\n",
    );
    if findings.is_empty() {
        out.push_str("[]\n");
    }
    for f in findings {
        out.push_str(&format!(
            "- rule: {}\n  path: {}\n  item: {}\n",
            f.rule.name(),
            f.path,
            yaml_scalar(&f.item)
        ));
    }
    out
}

/// `s` as a YAML scalar: quoted when plain YAML would misread it.
fn yaml_scalar(s: &str) -> String {
    if s.contains(':') || s.contains('#') || s.starts_with(['*', '&', '!', '[', '{']) {
        format!("'{}'", s.replace('\'', "''"))
    } else {
        s.to_string()
    }
}

/// What a scan found.
#[derive(Debug, Default)]
pub struct Scan {
    /// Every finding of the discovered rules.
    pub findings: BTreeSet<Finding>,

    /// Every non-test struct and enum, by file and name: what a
    /// `mirror-rename` entry is held against.
    pub types: BTreeSet<(String, String)>,
}

/// Every finding in the tree under `root`, and every type it defines.
///
/// # Errors
/// When a file can't be read or parsed.
pub fn scan(root: &Path) -> Result<Scan, String> {
    let mut files = Vec::new();
    collect_rs(root, &mut files)?;
    files.sort();
    let mut scanned = Scan::default();
    for file in files {
        let segments = module_segments(root, &file);
        if is_test_path(&file, &segments) {
            continue;
        }
        let text = fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let parsed = syn::parse_file(&text).map_err(|e| format!("{}: {e}", file.display()))?;
        let mut scope = Scope {
            path: file.display().to_string(),
            segments,
            findings: &mut scanned.findings,
            types: &mut scanned.types,
        };
        scope.items(&parsed.items);
    }
    Ok(scanned)
}

/// Every `.rs` file under `dir`.
fn collect_rs(
    dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        if path.is_dir() {
            collect_rs(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// `file`'s module path below `root`: its directories, then its stem, unless
/// that is `mod`.
fn module_segments(
    root: &Path,
    file: &Path,
) -> Vec<String> {
    let rel = file.strip_prefix(root).unwrap_or(file);
    let mut segments: Vec<String> = rel
        .parent()
        .into_iter()
        .flat_map(Path::components)
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if let Some(stem) = rel.file_stem().map(|s| s.to_string_lossy().into_owned())
        && stem != "mod"
    {
        segments.push(stem);
    }
    segments
}

/// Whether `file` is test code by its name or its module path.
fn is_test_path(
    file: &Path,
    segments: &[String],
) -> bool {
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    stem.ends_with("tests") || segments.iter().any(|s| s == "testing" || s == "tests")
}

/// Where the walk is: the file, the module path, and what it found.
struct Scope<'a> {
    path: String,
    segments: Vec<String>,
    findings: &'a mut BTreeSet<Finding>,
    types: &'a mut BTreeSet<(String, String)>,
}

impl Scope<'_> {
    /// Whether the module path is in a module named `name`.
    fn within(
        &self,
        name: &str,
    ) -> bool {
        self.segments.iter().any(|s| s == name)
    }

    fn report(
        &mut self,
        rule: Rule,
        item: String,
    ) {
        self.findings.insert(Finding {
            rule,
            path: self.path.clone(),
            item,
        });
    }

    /// Checks `items`, recursing into inline modules.
    fn items(
        &mut self,
        items: &[Item],
    ) {
        for item in items {
            if is_test(item_attrs(item)) {
                continue;
            }
            self.store_gate(item);
            match item {
                Item::Struct(s) => self.mirror(&s.ident),
                Item::Enum(e) => self.mirror(&e.ident),
                Item::Fn(f) => self.rekey(&f.sig, None),
                Item::Impl(imp) if imp.trait_.is_none() => {
                    let owner = imp.self_ty.to_token_stream().to_string().replace(' ', "");
                    for member in &imp.items {
                        if let ImplItem::Fn(f) = member
                            && !is_test(&f.attrs)
                        {
                            self.rekey(&f.sig, Some(&owner));
                            self.method_store_gate(&f.attrs, &owner, &f.sig.ident);
                        }
                    }
                }
                Item::Mod(m) => {
                    if let Some((_, content)) = &m.content {
                        let name = m.ident.to_string();
                        if name == "tests" || name == "testing" {
                            continue;
                        }
                        self.segments.push(name);
                        self.items(content);
                        self.segments.pop();
                    }
                }
                _ => {}
            }
        }
    }

    fn mirror(
        &mut self,
        ident: &syn::Ident,
    ) {
        let name = ident.to_string();
        self.types.insert((self.path.clone(), name.clone()));
        if name.ends_with("Mirror") && !self.within("interop") {
            self.report(Rule::Mirror, name);
        }
    }

    fn rekey(
        &mut self,
        sig: &Signature,
        owner: Option<&str>,
    ) {
        if self.within("interop") {
            return;
        }
        let name = sig.ident.to_string();
        let loader =
            name.starts_with("load_") && (name.ends_with("_tensors") || name.ends_with("_weights"));
        let returns_rekey = match &sig.output {
            ReturnType::Default => false,
            ReturnType::Type(_, ty) => returns_rekey(&ty.to_token_stream().to_string()),
        };
        if loader || returns_rekey {
            let item = owner.map_or_else(|| name.clone(), |o| format!("{o}::{name}"));
            self.report(Rule::Rekey, item);
        }
    }

    fn method_store_gate(
        &mut self,
        attrs: &[Attribute],
        owner: &str,
        ident: &syn::Ident,
    ) {
        if !self.within("interop") && !self.within("pretrained") && attrs.iter().any(gates_on_store)
        {
            self.report(Rule::StoreGate, format!("{owner}::{ident}"));
        }
    }

    fn store_gate(
        &mut self,
        item: &Item,
    ) {
        if self.within("interop") || self.within("pretrained") {
            return;
        }
        if !item_attrs(item).iter().any(gates_on_store) {
            return;
        }
        let label = match item {
            Item::Mod(m) => {
                // A gate on `pretrained` (or an interop module) is the
                // module's own: allowed.
                let name = m.ident.to_string();
                if name == "pretrained" || name == "interop" {
                    return;
                }
                format!("mod {name}")
            }
            // A gated re-export follows its module's gate, the finding.
            Item::Use(_) => return,
            other => item_name(other),
        };
        self.report(Rule::StoreGate, label);
    }
}

/// Whether a return type's tokens are a `Rekey` or a result of one.
fn returns_rekey(ty: &str) -> bool {
    let compact = ty.replace(' ', "");
    compact == "Rekey" || compact.ends_with("::Rekey") || compact.ends_with("<Rekey>")
}

/// An item's attributes.
fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

/// An item's name, for a finding.
fn item_name(item: &Item) -> String {
    match item {
        Item::Const(i) => i.ident.to_string(),
        Item::Enum(i) => i.ident.to_string(),
        Item::Fn(i) => i.sig.ident.to_string(),
        Item::Mod(i) => format!("mod {}", i.ident),
        Item::Static(i) => i.ident.to_string(),
        Item::Struct(i) => i.ident.to_string(),
        Item::Trait(i) => i.ident.to_string(),
        Item::Type(i) => i.ident.to_string(),
        Item::Impl(i) => format!("impl {}", i.self_ty.to_token_stream()).replace(' ', ""),
        _ => "an item".to_string(),
    }
}

/// Whether the attributes mark test code: `#[test]`, or a `cfg` that names
/// `test`.
fn is_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("test")
            || (a.path().is_ident("cfg")
                && cfg_tokens(a)
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .any(|w| w == "test"))
    })
}

/// Whether the attribute is a `cfg` that names `feature = "store"`.
fn gates_on_store(attr: &Attribute) -> bool {
    attr.path().is_ident("cfg")
        && cfg_tokens(attr)
            .replace(' ', "")
            .contains("feature=\"store\"")
}

/// A `cfg` attribute's tokens.
fn cfg_tokens(attr: &Attribute) -> String {
    attr.meta.to_token_stream().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn findings(
        segments: &[&str],
        source: &str,
    ) -> Vec<(Rule, String)> {
        let mut found = BTreeSet::new();
        let mut types = BTreeSet::new();
        let mut scope = Scope {
            path: "f.rs".into(),
            segments: segments.iter().map(|s| s.to_string()).collect(),
            findings: &mut found,
            types: &mut types,
        };
        scope.items(&syn::parse_file(source).unwrap().items);
        found.into_iter().map(|f| (f.rule, f.item)).collect()
    }

    /// Each rule fires outside `interop`, and not inside it.
    #[test]
    fn test_rules() {
        let source = r#"
            pub struct ConfigMirror { a: u32 }
            pub struct ConfigJson { a: u32 }
            pub fn talker_rekey() -> Rekey { todo!() }
            pub fn load_talker_tensors() {}
            impl Talker { pub fn rekey(&self) -> BunsenResult<Rekey> { todo!() } }
            impl CanonicalKeys for Talker { fn canonical_rekey(&self) -> Rekey { todo!() } }
            #[cfg(feature = "store")]
            pub mod weights;
            #[cfg(all(feature = "store", feature = "cache"))]
            pub mod pretrained;
        "#;
        assert_eq!(
            findings(&["kits", "talker"], source),
            [
                (Rule::Mirror, "ConfigMirror".to_string()),
                (Rule::Rekey, "Talker::rekey".to_string()),
                (Rule::Rekey, "load_talker_tensors".to_string()),
                (Rule::Rekey, "talker_rekey".to_string()),
                (Rule::StoreGate, "mod weights".to_string()),
            ]
        );
        assert!(findings(&["kits", "talker", "interop", "qwen_tts"], source).is_empty());
    }

    /// Test code is exempt: `#[cfg(test)]` items and modules, `#[test]`s, a
    /// `tests` module.
    #[test]
    fn test_test_code_is_exempt() {
        let source = r#"
            #[cfg(all(test, feature = "store"))]
            mod toy_tests;
            #[cfg(test)]
            mod tests { pub struct FixtureMirror; }
            mod testing { pub fn load_fixture_tensors() {} }
            #[test]
            #[cfg(feature = "store")]
            fn test_load() {}
        "#;
        assert!(findings(&["kits", "talker"], source).is_empty());
    }

    /// The allowlist reads back what `--bless` writes.
    #[test]
    fn test_allowlist_round_trip() {
        let found: BTreeSet<Finding> = [
            Finding {
                rule: Rule::StoreGate,
                path: "a/b.rs".into(),
                item: "mod weights".into(),
            },
            Finding {
                rule: Rule::Rekey,
                path: "a/b.rs".into(),
                item: "Talker::rekey".into(),
            },
        ]
        .into_iter()
        .collect();
        let read: Vec<Finding> = serde_saphyr::from_str(&render_allowlist(&found)).unwrap();
        assert_eq!(read.into_iter().collect::<BTreeSet<_>>(), found);
        let empty: Vec<Finding> =
            serde_saphyr::from_str(&render_allowlist(&BTreeSet::new())).unwrap();
        assert!(empty.is_empty());
    }
}
