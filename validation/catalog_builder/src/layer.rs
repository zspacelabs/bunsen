//! One layer built: a probe-contract source, its builder keys expanded, its
//! pins read from the remotes, and its probes run, into a catalog layer.
//!
//! A source is a keyed catalog layer
//! ([`catalog_format`](bunsen::data::pretrained::build_catalog::catalog_format))
//! with three things the builder fills in, and strips:
//!
//! - **`$hub: {$repo, $revision}` on a build**: the source of each of the
//!   build's own resources that writes no `$sources`.
//! - **Pins left out.** A resource or shard had from a `$hub` takes its
//!   `$sha256` and `$size` from the repo's listing (an LFS file) or from its
//!   bytes (a small file, which is fetched); one had only from URLs must write
//!   its `$sha256`, and takes its `$size` from a `HEAD`. A pin the source
//!   writes must agree with the one read. A build's safetensors resources take
//!   their `$dtype` from the build's `config.json`.
//! - **`$probes: [{$probe, $resource, $into, $token?}]` on a build**: each
//!   probe reads the resource bound to the slot `$resource` and patches the
//!   build's `$presets`, into `$common` or the presets `$into` names. What
//!   probes find accumulates
//!   ([`accumulate_probed`]),
//!   and the source's own patches are laid over it as merge patches.

use std::collections::BTreeMap;

use bunsen::{
    data::pretrained::{
        CatalogProbe,
        build_catalog::document::OrderedNode,
    },
    kits::support::model_config::{
        Probe,
        accumulate_probed,
        builtin_probes,
    },
    support::merge_patch,
};
use icu_locale::{
    Locale,
    LocaleCanonicalizer,
};
use serde_json::{
    Map,
    Value,
};

use crate::remote::{
    HubFile,
    HubRepo,
    Remote,
    config_dtype,
    sha256_of,
};

/// The formats whose resources are checkpoints: their `$dtype` is the one the
/// build's `config.json` names.
const CHECKPOINT_FORMATS: &[&str] = &["safetensors", "safetensors-index"];

/// The key a preset set's common patch is written under.
const COMMON: &str = "$common";

/// What a layer's build reads, and how hard it checks.
pub struct LayerBuilder<'a> {
    remote: &'a dyn Remote,
    verify_remote: bool,
    listings: BTreeMap<HubRepo, BTreeMap<String, HubFile>>,
    /// The bytes of the files fetched, by the resource's path in the layer.
    fetched: BTreeMap<String, Vec<u8>>,
}

/// One `$probes` entry of a build.
#[derive(Clone, Debug)]
struct ProbeSpec {
    probe: String,
    slot: String,
    into: Vec<String>,
    token: Option<String>,
}

/// A build whose resources are filled before its dtypes and probes are.
#[derive(Clone, Debug)]
struct PendingBuild {
    path: Vec<String>,
    probes: Vec<ProbeSpec>,
}

impl<'a> LayerBuilder<'a> {
    /// A builder reading from `remote`; with `verify_remote`, it fetches every
    /// file and checks its bytes hash to its pin.
    pub fn new(
        remote: &'a dyn Remote,
        verify_remote: bool,
    ) -> Self {
        Self {
            remote,
            verify_remote,
            listings: BTreeMap::new(),
            fetched: BTreeMap::new(),
        }
    }

    /// Builds the layer `source` writes. `contracts`, for the base layer,
    /// is the `$contracts` section it is given; a source never writes one.
    ///
    /// # Errors
    /// What is wrong with the source, or what the remotes don't agree with,
    /// naming the entry.
    pub fn build(
        &mut self,
        mut source: OrderedNode,
        contracts: Option<OrderedNode>,
    ) -> Result<OrderedNode, String> {
        let OrderedNode::Map(root) = &mut source else {
            return Err("a source is a map".into());
        };
        if root.iter().any(|(k, _)| k == "$contracts") {
            return Err(
                "a source writes no `$contracts`: the builder gives the base layer the kits' \
                 contracts"
                    .into(),
            );
        }
        if let Some(contracts) = contracts {
            let at = usize::from(root.first().is_some_and(|(k, _)| k == "$description"));
            root.insert(at, ("$contracts".to_string(), contracts));
        }

        let mut pending = Vec::new();
        expand_builds(&mut source, &mut Vec::new(), &mut pending)?;
        self.fill_resources(&mut source, &mut Vec::new())?;
        for build in &pending {
            self.finish_build(&mut source, build)
                .map_err(|e| format!("{}: {e}", build.path.join("/")))?;
        }
        check_languages(&source, &mut Vec::new())?;
        Ok(source)
    }

    fn listing(
        &mut self,
        repo: &HubRepo,
    ) -> Result<&BTreeMap<String, HubFile>, String> {
        if !self.listings.contains_key(repo) {
            let listing = crate::remote::hub_listing(self.remote, repo)?;
            self.listings.insert(repo.clone(), listing);
        }
        Ok(&self.listings[repo])
    }

    /// Fills every resource's pins, under `node`.
    fn fill_resources(
        &mut self,
        node: &mut OrderedNode,
        path: &mut Vec<String>,
    ) -> Result<(), String> {
        let OrderedNode::Map(entries) = node else {
            return Ok(());
        };
        for (key, child) in entries.iter_mut() {
            if key.starts_with('$') || !child.is_entry() {
                continue;
            }
            path.push(key.clone());
            if kind_of(child) == Some("resource") {
                self.fill_resource(child, path)
                    .map_err(|e| format!("{}: {e}", path.join("/")))?;
            } else {
                self.fill_resources(child, path)?;
            }
            path.pop();
        }
        Ok(())
    }

    /// Fills one resource's pins: its file's and each shard's.
    fn fill_resource(
        &mut self,
        resource: &mut OrderedNode,
        path: &[String],
    ) -> Result<(), String> {
        let file = string_field(resource, "$file")?.to_string();
        let sources = sources_of(resource)?;
        let hub = hub_of(&sources)?;
        let url = first_url(&sources, hub.as_ref(), &file);

        let pin = self.pin(
            &file,
            string_opt(resource, "$sha256")?,
            u64_opt(resource, "$size")?,
            hub.as_ref(),
            url.as_deref(),
        )?;
        if let Some(bytes) = pin.bytes {
            self.fetched.insert(path.join("/"), bytes);
        }
        set(
            resource,
            "$sha256",
            OrderedNode::Scalar(Value::String(pin.sha256)),
        );
        if let Some(size) = pin.size {
            set(resource, "$size", OrderedNode::Scalar(Value::from(size)));
        }

        let Some(OrderedNode::Seq(shards)) = get_mut(resource, "$shards") else {
            return Ok(());
        };
        let place_bases: Vec<String> = sources
            .iter()
            .filter_map(|s| s.get("$base_url").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
        for shard in shards.iter_mut() {
            let file = string_field(shard, "$file")?.to_string();
            let own = sources_of(shard)?;
            let url = own
                .iter()
                .find_map(|s| s.get("$url").and_then(Value::as_str).map(str::to_string))
                .or_else(|| hub.as_ref().map(|h| h.file_url(&file)))
                .or_else(|| place_bases.first().map(|b| join_url(b, &file)));
            let pin = self
                .pin(
                    &file,
                    string_opt(shard, "$sha256")?,
                    u64_opt(shard, "$size")?,
                    hub.as_ref(),
                    url.as_deref(),
                )
                .map_err(|e| format!("the shard {file}: {e}"))?;
            let size = pin
                .size
                .ok_or_else(|| format!("the shard {file}: no size: write its `$size`"))?;
            set(
                shard,
                "$sha256",
                OrderedNode::Scalar(Value::String(pin.sha256)),
            );
            set(shard, "$size", OrderedNode::Scalar(Value::from(size)));
        }
        Ok(())
    }

    /// One file's pin: what the source writes, reconciled with what the
    /// remotes say.
    fn pin(
        &mut self,
        file: &str,
        written_sha: Option<String>,
        written_size: Option<u64>,
        hub: Option<&HubRepo>,
        url: Option<&str>,
    ) -> Result<Pin, String> {
        let (derived, size, bytes) = match hub {
            Some(repo) => {
                let entry = self
                    .listing(repo)?
                    .get(file)
                    .cloned()
                    .ok_or_else(|| format!("{file} is not in {}'s listing", repo.repo))?;
                match entry.lfs {
                    Some(lfs) => (Some(lfs.oid), Some(lfs.size), None),
                    None => {
                        // A small file: fetched, so its digest is derived and
                        // a probe can read it.
                        let bytes = self.remote.get(&repo.file_url(file))?;
                        (Some(sha256_of(&bytes)), Some(entry.size), Some(bytes))
                    }
                }
            }
            None => {
                let size = match (written_size, url) {
                    (Some(size), _) => Some(size),
                    (None, Some(url)) => self.remote.head_size(url)?,
                    (None, None) => None,
                };
                (None, size, None)
            }
        };
        let sha256 = match (written_sha, derived) {
            (Some(w), Some(d)) if w != d => {
                return Err(format!(
                    "{file}: the source's $sha256 {w} disagrees with the remote's {d}"
                ));
            }
            (Some(w), _) => w,
            (None, Some(d)) => d,
            (None, None) => {
                return Err(format!(
                    "{file}: no $sha256: it is not on the Hub, so the source writes one"
                ));
            }
        };
        if let (Some(w), Some(s)) = (written_size, size)
            && w != s
        {
            return Err(format!(
                "{file}: the source's $size {w} disagrees with the remote's {s}"
            ));
        }
        if let Some(url) = url
            && url.contains("openaipublic.azureedge.net")
            && !url.contains(&sha256)
        {
            return Err(format!(
                "{file}: the URL {url} doesn't carry its $sha256 {sha256}"
            ));
        }
        if self.verify_remote {
            let url = url.ok_or_else(|| format!("{file}: no URL to verify"))?;
            let hashed = self.remote.sha256(url)?;
            if hashed != sha256 {
                return Err(format!("{file}: {url} hashes to {hashed}, not {sha256}"));
            }
        }
        Ok(Pin {
            sha256,
            size: size.or(written_size),
            bytes,
        })
    }

    /// A build's dtypes and probes, once its resources are filled.
    fn finish_build(
        &mut self,
        root: &mut OrderedNode,
        pending: &PendingBuild,
    ) -> Result<(), String> {
        let build_key = pending.path.join("/");
        let build = node_at(root, &pending.path).ok_or("the build is gone")?;

        // The dtype of its checkpoints, from its own `config.json`.
        let dtype = children(build)
            .find(|(_, child)| {
                kind_of(child) == Some("resource")
                    && string_field(child, "$file").is_ok_and(|f| f == "config.json")
            })
            .and_then(|(name, _)| self.fetched.get(&format!("{build_key}/{name}")))
            .and_then(|bytes| config_dtype(bytes));

        // What its probes find, by where it goes.
        let mut found: BTreeMap<String, Value> = BTreeMap::new();
        if !pending.probes.is_empty() {
            let probes = builtin_probes();
            for spec in &pending.probes {
                let at = format!("the {} probe of `{}`", spec.probe, spec.slot);
                let bytes = self
                    .slot_bytes(root, &pending.path, &spec.slot)
                    .map_err(|e| format!("{at}: {e}"))?;
                let probe: &dyn Probe = probes
                    .iter()
                    .map(Box::as_ref)
                    .find(|p| p.name() == spec.probe)
                    .ok_or_else(|| {
                        let names: Vec<_> = probes.iter().map(|p| p.name()).collect();
                        format!("{at}: no such probe; there are {}", names.join(", "))
                    })?;
                let entry = CatalogProbe {
                    probe: spec.probe.clone(),
                    file: spec.slot.clone(),
                    into: spec.into.clone(),
                    token: spec.token.clone(),
                };
                let patch = probe
                    .probe(&bytes, &entry)
                    .map_err(|e| format!("{at}: {e}"))?;
                for into in &spec.into {
                    accumulate_probed(
                        found
                            .entry(into.clone())
                            .or_insert_with(|| Value::Object(Map::new())),
                        &patch,
                    );
                }
            }
        }

        let build = node_at_mut(root, &pending.path).ok_or("the build is gone")?;
        if let Some(dtype) = dtype {
            let OrderedNode::Map(entries) = build else {
                unreachable!("a build is a map")
            };
            for (_, child) in entries.iter_mut() {
                if kind_of(child) == Some("resource")
                    && string_field(child, "$format").is_ok_and(|f| CHECKPOINT_FORMATS.contains(&f))
                    && get(child, "$dtype").is_none()
                {
                    set(
                        child,
                        "$dtype",
                        OrderedNode::Scalar(Value::String(dtype.clone())),
                    );
                }
            }
        }
        if !found.is_empty() {
            let presets = get_or_insert_map(build, "$presets");
            for (into, probed) in found {
                let (slot, key) = if into == COMMON {
                    (&mut *presets, COMMON)
                } else {
                    (get_or_insert_map(&mut *presets, &into), "$patch")
                };
                let mut patch = probed;
                if let Some(written) = get(slot, key) {
                    merge_patch::apply(&mut patch, &written.clone().into_value());
                }
                let patch = OrderedNode::from_serialize(&patch).map_err(|e| e.to_string())?;
                set(slot, key, patch);
            }
        }
        Ok(())
    }

    /// The bytes of the resource a build binds to `slot`.
    fn slot_bytes(
        &mut self,
        root: &OrderedNode,
        build_path: &[String],
        slot: &str,
    ) -> Result<Vec<u8>, String> {
        let build = node_at(root, build_path).ok_or("the build is gone")?;
        let reference = get(build, "$resources")
            .and_then(|r| get(r, slot))
            .and_then(|r| match r {
                OrderedNode::Scalar(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("the build binds no resource to the slot `{slot}`"))?;
        let path = resolve_in_layer(root, build_path, &reference).ok_or_else(|| {
            format!("`{reference}` names no resource in this layer, from the build")
        })?;
        let key = path.join("/");
        if let Some(bytes) = self.fetched.get(&key) {
            return Ok(bytes.clone());
        }
        let resource = node_at(root, &path).ok_or("the resource is gone")?;
        let file = string_field(resource, "$file")?;
        let sources = sources_of(resource)?;
        let url = first_url(&sources, hub_of(&sources)?.as_ref(), file)
            .ok_or_else(|| format!("`{reference}` has no URL to fetch"))?;
        let bytes = self.remote.get(&url)?;
        self.fetched.insert(key, bytes.clone());
        Ok(bytes)
    }
}

/// A file's pin, and its bytes when they were fetched.
struct Pin {
    sha256: String,
    size: Option<u64>,
    bytes: Option<Vec<u8>>,
}

/// Expands each build's `$hub` onto its resources, and takes its
/// `$probes`, under `node`.
fn expand_builds(
    node: &mut OrderedNode,
    path: &mut Vec<String>,
    pending: &mut Vec<PendingBuild>,
) -> Result<(), String> {
    let OrderedNode::Map(entries) = node else {
        return Ok(());
    };
    for (key, child) in entries.iter_mut() {
        if key.starts_with('$') || !child.is_entry() {
            continue;
        }
        path.push(key.clone());
        let at = path.join("/");
        for builder_key in ["$hub", "$probes"] {
            if kind_of(child) != Some("build") && get(child, builder_key).is_some() {
                return Err(format!("{at}: only a build writes `{builder_key}`"));
            }
        }
        if kind_of(child) == Some("build") {
            if let Some(hub) = take(child, "$hub") {
                let repo = hub_repo(&hub.clone().into_value())
                    .map_err(|e| format!("{at}: `$hub`: {e}"))?;
                let source = hub_source(&repo);
                let OrderedNode::Map(fields) = child else {
                    unreachable!("an entry is a map")
                };
                for (_, resource) in fields.iter_mut() {
                    if kind_of(resource) == Some("resource") && get(resource, "$sources").is_none()
                    {
                        set(resource, "$sources", OrderedNode::Seq(vec![source.clone()]));
                    }
                }
            }
            let probes = match take(child, "$probes") {
                Some(probes) => read_probes(&probes.clone().into_value())
                    .map_err(|e| format!("{at}: `$probes`: {e}"))?,
                None => Vec::new(),
            };
            pending.push(PendingBuild {
                path: path.clone(),
                probes,
            });
        }
        expand_builds(child, path, pending)?;
        path.pop();
    }
    Ok(())
}

fn read_probes(value: &Value) -> Result<Vec<ProbeSpec>, String> {
    let Value::Array(items) = value else {
        return Err("a list of probes".into());
    };
    items
        .iter()
        .map(|item| {
            let Value::Object(fields) = item else {
                return Err("a probe is a map".to_string());
            };
            for key in fields.keys() {
                if !["$probe", "$resource", "$into", "$token"].contains(&key.as_str()) {
                    return Err(format!(
                        "`{key}`: a probe writes `$probe`, `$resource`, `$into` and `$token`"
                    ));
                }
            }
            let text = |key: &str| {
                fields
                    .get(key)
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| format!("a probe writes `{key}`"))
            };
            let into: Vec<String> =
                serde_json::from_value(fields.get("$into").cloned().unwrap_or(Value::Null))
                    .map_err(|_| "`$into` is a list of `$common` and preset names".to_string())?;
            if into.is_empty() {
                return Err("`$into` names at least one place".into());
            }
            Ok(ProbeSpec {
                probe: text("$probe")?,
                slot: text("$resource")?,
                into,
                token: fields
                    .get("$token")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}

fn hub_repo(value: &Value) -> Result<HubRepo, String> {
    let field = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("a hub writes `{key}`"))
    };
    Ok(HubRepo {
        repo: field("$repo")?,
        revision: field("$revision")?,
    })
}

fn hub_source(repo: &HubRepo) -> OrderedNode {
    let scalar = |s: &str| OrderedNode::Scalar(Value::String(s.to_string()));
    OrderedNode::Map(vec![(
        "$hub".to_string(),
        OrderedNode::Map(vec![
            ("$repo".to_string(), scalar(&repo.repo)),
            ("$revision".to_string(), scalar(&repo.revision)),
        ]),
    )])
}

/// A resource's `$sources`, as values.
fn sources_of(node: &OrderedNode) -> Result<Vec<Value>, String> {
    match get(node, "$sources") {
        None => Ok(Vec::new()),
        Some(OrderedNode::Seq(items)) => Ok(items.iter().map(|n| n.clone().into_value()).collect()),
        Some(_) => Err("`$sources` is a list".into()),
    }
}

/// The first `$hub` of `sources`.
fn hub_of(sources: &[Value]) -> Result<Option<HubRepo>, String> {
    sources
        .iter()
        .find_map(|s| s.get("$hub"))
        .map(hub_repo)
        .transpose()
}

/// The URL `file` is fetched from first: its own `$url`, else the hub's,
/// else under the first `$base_url`.
fn first_url(
    sources: &[Value],
    hub: Option<&HubRepo>,
    file: &str,
) -> Option<String> {
    sources
        .iter()
        .find_map(|s| s.get("$url").and_then(Value::as_str).map(str::to_string))
        .or_else(|| hub.map(|h| h.file_url(file)))
        .or_else(|| {
            sources
                .iter()
                .find_map(|s| s.get("$base_url").and_then(Value::as_str))
                .map(|b| join_url(b, file))
        })
}

fn join_url(
    base: &str,
    file: &str,
) -> String {
    format!("{}/{file}", base.trim_end_matches('/'))
}

/// The path of the resource `reference` names from the build at
/// `build_path`, among this layer's entries: the first of
/// `build/reference`, `parent/reference`, ..., `reference` that is a
/// resource; from the root alone for a leading `/`.
fn resolve_in_layer(
    root: &OrderedNode,
    build_path: &[String],
    reference: &str,
) -> Option<Vec<String>> {
    let (absolute, reference) = match reference.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, reference),
    };
    let tail: Vec<String> = reference.split('/').map(str::to_string).collect();
    let depths: Vec<usize> = if absolute {
        vec![0]
    } else {
        (0..=build_path.len()).rev().collect()
    };
    depths.into_iter().find_map(|depth| {
        let mut path = build_path[..depth].to_vec();
        path.extend(tail.iter().cloned());
        node_at(root, &path)
            .filter(|n| kind_of(n) == Some("resource"))
            .map(|_| path)
    })
}

/// `tag`, canonicalized as BCP 47 (`EN-us` -> `en-US`, `eng` -> `en`).
fn canonical_language(tag: &str) -> Result<String, String> {
    let mut locale =
        Locale::try_from_str(tag).map_err(|e| format!("{tag:?} is not a BCP 47 tag: {e:?}"))?;
    LocaleCanonicalizer::new_extended().canonicalize(&mut locale);
    Ok(locale.to_string())
}

/// Refuses a `language` tag whose tags aren't canonical BCP 47, under
/// `node`.
fn check_languages(
    node: &OrderedNode,
    path: &mut Vec<String>,
) -> Result<(), String> {
    let OrderedNode::Map(entries) = node else {
        return Ok(());
    };
    for (key, child) in entries {
        if key == "language" && !child.is_entry() && !path.is_empty() {
            let Value::Array(tags) = child.clone().into_value() else {
                return Err(format!("{}: `language` is a list", path.join("/")));
            };
            for tag in tags {
                let tag = tag.as_str().unwrap_or_default().to_string();
                if tag == "*" {
                    continue;
                }
                let canonical =
                    canonical_language(&tag).map_err(|e| format!("{}: {e}", path.join("/")))?;
                if canonical != tag {
                    return Err(format!(
                        "{}: the language {tag:?} is written {canonical:?} in BCP 47",
                        path.join("/")
                    ));
                }
            }
        }
        if key.starts_with('$') || !child.is_entry() {
            continue;
        }
        path.push(key.clone());
        check_languages(child, path)?;
        path.pop();
    }
    Ok(())
}

fn kind_of(node: &OrderedNode) -> Option<&str> {
    match get(node, "$kind") {
        Some(OrderedNode::Scalar(Value::String(kind))) => Some(kind),
        _ => None,
    }
}

fn children(node: &OrderedNode) -> impl Iterator<Item = (&String, &OrderedNode)> {
    let entries: &[(String, OrderedNode)] = match node {
        OrderedNode::Map(entries) => entries,
        _ => &[],
    };
    entries
        .iter()
        .filter(|(k, v)| !k.starts_with('$') && v.is_entry())
        .map(|(k, v)| (k, v))
}

fn node_at<'n>(
    root: &'n OrderedNode,
    path: &[String],
) -> Option<&'n OrderedNode> {
    path.iter()
        .try_fold(root, |node, segment| get(node, segment))
}

fn node_at_mut<'n>(
    root: &'n mut OrderedNode,
    path: &[String],
) -> Option<&'n mut OrderedNode> {
    path.iter()
        .try_fold(root, |node, segment| get_mut(node, segment))
}

fn get<'n>(
    node: &'n OrderedNode,
    key: &str,
) -> Option<&'n OrderedNode> {
    node.get(key)
}

fn get_mut<'n>(
    node: &'n mut OrderedNode,
    key: &str,
) -> Option<&'n mut OrderedNode> {
    match node {
        OrderedNode::Map(entries) => entries.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

/// Sets `key` in a map: in place when it is there, else at the end.
fn set(
    node: &mut OrderedNode,
    key: &str,
    value: OrderedNode,
) {
    if let OrderedNode::Map(entries) = node {
        match entries.iter_mut().find(|(k, _)| k == key) {
            Some((_, slot)) => *slot = value,
            None => entries.push((key.to_string(), value)),
        }
    }
}

fn take(
    node: &mut OrderedNode,
    key: &str,
) -> Option<OrderedNode> {
    let OrderedNode::Map(entries) = node else {
        return None;
    };
    let at = entries.iter().position(|(k, _)| k == key)?;
    Some(entries.remove(at).1)
}

fn get_or_insert_map<'n>(
    node: &'n mut OrderedNode,
    key: &str,
) -> &'n mut OrderedNode {
    if get(node, key).is_none() {
        set(node, key, OrderedNode::Map(Vec::new()));
    }
    get_mut(node, key).expect("just set")
}

fn string_field<'n>(
    node: &'n OrderedNode,
    key: &str,
) -> Result<&'n str, String> {
    match get(node, key) {
        Some(OrderedNode::Scalar(Value::String(s))) => Ok(s),
        _ => Err(format!("`{key}` is a string")),
    }
}

fn string_opt(
    node: &OrderedNode,
    key: &str,
) -> Result<Option<String>, String> {
    match get(node, key) {
        None => Ok(None),
        Some(OrderedNode::Scalar(Value::String(s))) => Ok(Some(s.clone())),
        Some(_) => Err(format!("`{key}` is a string")),
    }
}

fn u64_opt(
    node: &OrderedNode,
    key: &str,
) -> Result<Option<u64>, String> {
    match get(node, key) {
        None => Ok(None),
        Some(OrderedNode::Scalar(v)) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("`{key}` is a whole number")),
        Some(_) => Err(format!("`{key}` is a whole number")),
    }
}

#[cfg(test)]
mod tests {
    use bunsen::data::pretrained::build_catalog::PretrainedCatalog;

    use super::*;
    use crate::remote::testing::FakeRemote;

    const CONFIG: &[u8] = br#"{"torch_dtype": "bfloat16"}"#;
    const GENERATION: &[u8] =
        br#"{"do_sample": true, "eos_token_id": [7, 8], "temperature": 0.6, "top_k": 20}"#;
    const WEIGHTS: &[u8] = b"weights";

    fn repo() -> HubRepo {
        HubRepo {
            repo: "o/m".into(),
            revision: "abc".into(),
        }
    }

    fn fake() -> FakeRemote {
        let mut fake = FakeRemote::default().with_repo(
            &repo(),
            &[
                ("config.json", CONFIG),
                ("generation_config.json", GENERATION),
                ("model.safetensors.index.json", b"{}"),
            ],
            &[
                ("model.safetensors", WEIGHTS),
                ("model-00001.safetensors", b"one"),
                ("model-00002.safetensors", b"two"),
            ],
        );
        fake.bodies
            .insert("https://x.example/v.txt".into(), b"vocab".to_vec());
        fake.sizes.insert("https://x.example/v.txt".into(), 5);
        fake
    }

    fn build(
        remote: &FakeRemote,
        yaml: &str,
    ) -> Result<OrderedNode, String> {
        let node = OrderedNode::from_yaml(yaml).unwrap();
        let built = LayerBuilder::new(remote, false).build(node, None)?;
        PretrainedCatalog::from_node(built.clone()).map_err(|e| e.to_string())?;
        Ok(built)
    }

    fn at<'n>(
        node: &'n OrderedNode,
        path: &str,
    ) -> &'n OrderedNode {
        let path: Vec<String> = path.split('/').map(str::to_string).collect();
        node_at(node, &path).unwrap_or_else(|| panic!("no {path:?}"))
    }

    fn value(
        node: &OrderedNode,
        path: &str,
    ) -> Value {
        at(node, path).clone().into_value()
    }

    const LAYER: &str = r#"
$description: "A test layer."
o:
  $kind: namespace
  $merge: update
  m:
    $kind: build
    $description: "M."
    $hub: { $repo: o/m, $revision: abc }
    $resources: { checkpoint: checkpoint, config: config, generation_config: generation_config, vocabulary: vocabulary }
    $probes:
      - { $probe: generation_config, $resource: generation_config, $into: [$common, thinking] }
    $presets:
      $common: { generation: { max_new_tokens: 512 } }
      thinking: { $description: "Think first.", $patch: { generation: { template_kwargs: { enable_thinking: true } } } }
    checkpoint: { $kind: resource, $file: model.safetensors, $format: safetensors }
    config: { $kind: resource, $file: config.json, $format: json }
    generation_config: { $kind: resource, $file: generation_config.json, $format: generation_config.json }
  vocabulary:
    $kind: resource
    $file: v.txt
    $format: text
    $sha256: "SHA"
    $sources: [{ $url: "https://x.example/v.txt" }]
"#;

    fn layer() -> String {
        LAYER.replace("SHA", &sha256_of(b"vocab"))
    }

    /// A build's `$hub` becomes its resources' source; a Hub file takes its
    /// pins from the listing (LFS) or its bytes (small); a checkpoint takes
    /// the `config.json`'s dtype; a URL-only file keeps its written digest
    /// and takes a `HEAD`'s size; and the builder keys are gone.
    #[test]
    fn test_pins_and_sources() {
        let built = build(&fake(), &layer()).unwrap();
        let m = at(&built, "o/m");
        assert!(m.get("$hub").is_none() && m.get("$probes").is_none());
        assert_eq!(
            value(&built, "o/m/checkpoint/$sources"),
            serde_json::json!([{"$hub": {"$repo": "o/m", "$revision": "abc"}}])
        );
        assert_eq!(
            value(&built, "o/m/checkpoint/$sha256"),
            Value::String(sha256_of(WEIGHTS))
        );
        assert_eq!(value(&built, "o/m/checkpoint/$size"), 7000);
        assert_eq!(value(&built, "o/m/checkpoint/$dtype"), "bfloat16");
        assert_eq!(
            value(&built, "o/m/config/$sha256"),
            Value::String(sha256_of(CONFIG))
        );
        assert_eq!(value(&built, "o/m/config/$size"), CONFIG.len());
        assert!(at(&built, "o/m/config").get("$dtype").is_none());
        assert_eq!(value(&built, "o/vocabulary/$size"), 5);
    }

    /// Probes patch `$common` and the presets they name; the source's own
    /// patches are laid over what they find.
    #[test]
    fn test_probes_patch_the_presets() {
        let built = build(&fake(), &layer()).unwrap();
        let common = value(&built, "o/m/$presets/$common");
        assert_eq!(common["generation"]["stop_ids"], serde_json::json!([7, 8]));
        assert_eq!(common["generation"]["max_new_tokens"], 512);
        let thinking = value(&built, "o/m/$presets/thinking");
        assert_eq!(thinking["$description"], "Think first.");
        assert_eq!(
            thinking["$patch"]["generation"]["sampling"]["temperature"],
            0.6
        );
        assert_eq!(
            thinking["$patch"]["generation"]["template_kwargs"]["enable_thinking"],
            true
        );
    }

    /// A sharded resource's shards take their pins from the listing too.
    #[test]
    fn test_shards() {
        let yaml = r#"
o:
  $kind: namespace
  m:
    $kind: build
    $description: "M."
    $hub: { $repo: o/m, $revision: abc }
    checkpoint:
      $kind: resource
      $file: model.safetensors.index.json
      $format: safetensors-index
      $shards: [{ $file: model-00001.safetensors }, { $file: model-00002.safetensors }]
    config: { $kind: resource, $file: config.json, $format: json }
"#;
        let built = build(&fake(), yaml).unwrap();
        let shards = value(&built, "o/m/checkpoint/$shards");
        assert_eq!(shards[0]["$sha256"], Value::String(sha256_of(b"one")));
        assert_eq!(shards[1]["$size"], 3000);
        assert_eq!(value(&built, "o/m/checkpoint/$dtype"), "bfloat16");
    }

    /// The base layer is given its `$contracts` after its description.
    #[test]
    fn test_the_base_layer_takes_the_contracts() {
        let node = OrderedNode::from_yaml("$description: \"Base.\"\n").unwrap();
        let contracts = OrderedNode::from_yaml("c: { $kind: contract, $description: C. }").unwrap();
        let built = LayerBuilder::new(&fake(), false)
            .build(node, Some(contracts))
            .unwrap();
        let OrderedNode::Map(entries) = &built else {
            unreachable!()
        };
        assert_eq!(entries[1].0, "$contracts");
    }

    #[test]
    fn test_refusals() {
        let fake = fake();
        let refused = |yaml: &str, why: &str| {
            let err = build(&fake, yaml).unwrap_err();
            assert!(err.contains(why), "{err}");
        };
        // A file off the Hub with no digest.
        refused(&LAYER.replace("    $sha256: \"SHA\"\n", ""), "writes one");
        // A digest the Hub disagrees with.
        refused(
            &layer().replace(
                "$format: safetensors }",
                &format!("$format: safetensors, $sha256: \"{}\" }}", "b".repeat(64)),
            ),
            "disagrees",
        );
        // A file the Hub doesn't list.
        refused(
            &layer().replace("model.safetensors", "gone.safetensors"),
            "not in o/m",
        );
        // Builder keys on a namespace.
        refused(
            "o: { $kind: namespace, $hub: { $repo: o/m, $revision: abc } }",
            "only a build",
        );
        // A probe of a slot the build doesn't bind.
        refused(
            &layer().replace(
                "$resource: generation_config,",
                "$resource: tokenizer_config,",
            ),
            "no resource to the slot `tokenizer_config`",
        );
        // An unknown probe.
        refused(
            &layer().replace("$probe: generation_config,", "$probe: telepathy,"),
            "no such probe",
        );
        // A language tag that isn't canonical.
        refused(
            "o: { $kind: namespace, m: { $kind: build, $description: M., language: [EN-us] } }",
            "\"en-US\"",
        );
        // A source that writes `$contracts`.
        let err = LayerBuilder::new(&fake, false)
            .build(OrderedNode::from_yaml("$contracts: {}").unwrap(), None)
            .unwrap_err();
        assert!(err.contains("writes no `$contracts`"), "{err}");
    }

    /// `--verify-remote` refuses bytes that don't hash to the pin.
    #[test]
    fn test_verify_remote() {
        let stale = LAYER.replace("SHA", &"c".repeat(64));
        assert!(build(&fake(), &stale).is_ok());
        let node = OrderedNode::from_yaml(&stale).unwrap();
        let err = LayerBuilder::new(&fake(), true)
            .build(node, None)
            .unwrap_err();
        assert!(err.contains("hashes to"), "{err}");
    }
}
