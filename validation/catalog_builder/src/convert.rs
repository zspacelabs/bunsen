//! `cargo catalog convert`: today's per-kit catalog sources,
//! `validation/catalogs/{kit}.yml`, written as probe-contract sources.
//!
//! The layout: a family with one publisher nests in it
//! (`qwen/qwen3/qwen3-0.6b`), its template classes, groups and shared
//! resources beside its builds; a family that spans publishers is a root
//! namespace, and its builds stay under their publishers and `$join` it.
//! A row becomes a build of its kit's contract (resnet's, qwen3-asr's and
//! voxtral's state their geometry, the prefab's config); its bundles become
//! its resources (a shared bundle's beside the family, shards as one
//! resource), its family its template class, its probes `$probes`, its
//! `groups` and `version` tags memberships, its `base` tag `instruct`, and a
//! `thinking` it writes without an axis a fixed use. What the old form
//! can't say is reported, not invented.

use std::{
    collections::BTreeMap,
    fs,
    path::Path,
};

use bunsen::data::pretrained::{
    build_catalog::document::OrderedNode,
    tags::{
        TagShape,
        VOCABULARY,
    },
};
use serde_json::Value;

use crate::{
    catalog::{
        MANIFEST,
        ManifestEntry,
    },
    tree::*,
};

/// One kit: its family, its contract, the feature it needs, and the tags
/// its rows must carry.
struct Kit {
    kit: &'static str,
    family: &'static str,
    contract: &'static str,
    feature: &'static str,
    required: &'static [&'static str],
}

/// The kits, in import order.
fn kits() -> Vec<Kit> {
    use bunsen::kits::{
        decision::laya::pretrained::LAYA_TAGS,
        language::{
            gemma::pretrained::GEMMA_TAGS,
            nanochat::pretrained::NANOCHAT_TAGS,
            qwen3::pretrained::QWEN3_TAGS,
            qwen3_5::pretrained::QWEN3_5_TAGS,
        },
        speech::asr::{
            parakeet::pretrained::PARAKEET_TAGS,
            qwen3_asr::pretrained::QWEN3_ASR_TAGS,
            voxtral::pretrained::VOXTRAL_TAGS,
            whisper::pretrained::WHISPER_TAGS,
        },
    };
    let kit = |kit, family, contract, feature, required| Kit {
        kit,
        family,
        contract,
        feature,
        required,
    };
    vec![
        kit("laya", "laya", "laya-decision", "kits", LAYA_TAGS.required),
        kit("resnet", "resnet", "resnet-classifier", "kits", &[]),
        kit("gemma", "gemma", "gemma-lm", "kits", GEMMA_TAGS.required),
        kit(
            "nanochat",
            "nanochat",
            "nanochat-lm",
            "kits",
            NANOCHAT_TAGS.required,
        ),
        kit("qwen3", "qwen3", "qwen3-lm", "kits", QWEN3_TAGS.required),
        kit(
            "qwen3.5",
            "qwen3.5",
            "qwen3.5-lm",
            "kits",
            QWEN3_5_TAGS.required,
        ),
        kit(
            "parakeet",
            "parakeet",
            "parakeet-asr",
            "speech",
            PARAKEET_TAGS.required,
        ),
        kit(
            "qwen3-asr",
            "qwen3-asr",
            "qwen3-asr/stated",
            "speech",
            QWEN3_ASR_TAGS.required,
        ),
        kit(
            "voxtral",
            "voxtral",
            "voxtral-asr/stated",
            "speech",
            VOXTRAL_TAGS.required,
        ),
        kit(
            "whisper",
            "whisper",
            "whisper-asr",
            "speech",
            WHISPER_TAGS.required,
        ),
        kit(
            "sortformer",
            "sortformer",
            "sortformer-diarizer",
            "speech",
            &[],
        ),
        kit("silero", "silero_vad", "silero-vad", "speech", &[]),
    ]
}

/// The publishers, as the commons declare them.
const PUBLISHERS: &[(&str, &str)] = &[
    ("laya", "convaiinnovations, Laya's authors."),
    (
        "torchvision",
        "TorchVision, PyTorch's computer vision library.",
    ),
    ("timm", "pytorch-image-models (timm), Ross Wightman's."),
    (
        "gemma",
        "Google's Gemma models, from their unsloth/ mirrors.",
    ),
    (
        "nanochat-students",
        "nanochat-students, who publish nanochat training runs.",
    ),
    (
        "nitishpandey04",
        "nitishpandey04, who publishes nanochat training runs.",
    ),
    ("qwen", "Qwen, Alibaba Cloud's model team."),
    ("fermion", "FermionResearch, Phonon's authors."),
    ("nvidia", "NVIDIA."),
    ("mistralai", "Mistral AI."),
    ("openai", "OpenAI."),
    ("silero", "Silero, snakers4/silero-vad's authors."),
];

/// The descriptions of the families that span publishers.
const SHARED_FAMILIES: &[(&str, &str)] = &[
    (
        "nanochat",
        "karpathy's nanochat GPT: community runs of upstream's 2025 and 2026 models.",
    ),
    ("parakeet", "NVIDIA's Parakeet TDT speech recognizers."),
    ("qwen3-asr", "Qwen's Qwen3-ASR speech recognizers."),
    ("resnet", "ResNet ImageNet-1k classifiers."),
];

/// The groups rows join through their `groups` tag.
const GROUPS: &[(&str, &str)] = &[
    (
        "qwen3-2507",
        "The Qwen3 2507 updates: one thinking mode each, and a 256K context.",
    ),
    (
        "qwen3guard",
        "Qwen3Guard-Gen: Qwen3 tuned to moderate prompts and responses.",
    ),
];

/// A family's series: builds whose names start with the prefix nest in a
/// namespace of its name.
const SERIES: &[(&str, &str, &str)] = &[("whisper", "large", "The large Whisper models.")];

/// A release sequence a `version` tag names.
struct Release {
    /// The sequence, as the tag names it.
    sequence: &'static str,
    /// The series namespace it sits in.
    series: &'static str,
    /// Its name there.
    name: &'static str,
    description: &'static str,
    /// Each version's release date.
    dates: &'static [(&'static str, &'static str)],
}

/// The release sequences the `version` tags name.
const RELEASES: &[Release] = &[Release {
    sequence: "whisper-large",
    series: "large",
    name: "release",
    description: "The large Whisper releases, each a drop-in update.",
    dates: &[
        ("1.0.0", "2022-09"),
        ("2.0.0", "2022-12"),
        ("3.0.0", "2023-11-06"),
    ],
}];

/// The registered axes' old names.
fn axis_name(old: &str) -> &str {
    match old {
        "thinking-mode" => "thinking",
        other => other,
    }
}

/// A dtype tag's spelling as a resource's `$dtype`.
fn long_dtype(short: &str) -> &str {
    match short {
        "f32" => "float32",
        "f16" => "float16",
        "bf16" => "bfloat16",
        other => other,
    }
}

/// What a conversion found and couldn't carry over, by topic.
#[derive(Debug, Default)]
pub struct Notes(BTreeMap<&'static str, Vec<String>>);

impl Notes {
    fn note(
        &mut self,
        topic: &'static str,
        text: impl Into<String>,
    ) {
        self.0.entry(topic).or_default().push(text.into());
    }

    /// The notes, a topic at a time.
    pub fn lines(&self) -> Vec<String> {
        self.0
            .iter()
            .flat_map(|(topic, items)| {
                std::iter::once(format!("== {topic} ({})", items.len()))
                    .chain(items.iter().map(|i| format!("   {i}")))
            })
            .collect()
    }
}

/// An axis the presets use: its kind, its members, its description.
#[derive(Clone, Debug)]
struct AxisDecl {
    bool_kind: bool,
    members: Vec<String>,
    description: String,
}

/// What the conversion gathers across kits.
#[derive(Default)]
struct Converter {
    notes: Notes,
    axes: Vec<(String, AxisDecl)>,
}

fn seq(node: Option<&OrderedNode>) -> Vec<OrderedNode> {
    match node {
        Some(OrderedNode::Seq(items)) => items.clone(),
        _ => Vec::new(),
    }
}

fn strings(node: Option<&OrderedNode>) -> Vec<String> {
    seq(node)
        .into_iter()
        .filter_map(|n| match n {
            OrderedNode::Scalar(Value::String(s)) => Some(s),
            _ => None,
        })
        .collect()
}

fn str_of<'n>(
    node: &'n OrderedNode,
    key: &str,
) -> Option<&'n str> {
    string_field(node, key).ok()
}

fn map_entries(node: Option<&OrderedNode>) -> Vec<(String, OrderedNode)> {
    match node {
        Some(OrderedNode::Map(entries)) => entries.clone(),
        _ => Vec::new(),
    }
}

fn is_empty_map(node: Option<&OrderedNode>) -> bool {
    match node {
        None => true,
        Some(OrderedNode::Map(entries)) => entries.is_empty(),
        _ => false,
    }
}

/// A sentence: as written, with a full stop. A description may open with
/// a name written in lowercase (`snakers4/silero-vad`), so its case stays.
fn sentence(text: &str) -> String {
    let mut out = text.to_string();
    if !out.ends_with('.') {
        out.push('.');
    }
    out
}

/// `text` with its first letter capitalized: a kit's description opens with
/// "the".
fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

impl Converter {
    fn register_axis(
        &mut self,
        name: &str,
        members: &[String],
        description: &str,
    ) {
        let bool_kind = name == "thinking";
        match self.axes.iter_mut().find(|(n, _)| n == name) {
            Some((_, decl)) if !decl.bool_kind => {
                for m in members {
                    if !decl.members.contains(m) {
                        decl.members.push(m.clone());
                    }
                }
            }
            Some(_) => {}
            None => self.axes.push((
                name.to_string(),
                AxisDecl {
                    bool_kind,
                    members: members.to_vec(),
                    description: description.to_string(),
                },
            )),
        }
    }

    /// An old preset set as `$presets`, registering the axes it uses.
    fn presets(
        &mut self,
        set: Option<&OrderedNode>,
        owner: &str,
    ) -> MapBuilder {
        let mut out = MapBuilder::new();
        let Some(set) = set else {
            return out;
        };
        if !is_empty_map(get(set, "base")) {
            out.push("$common", get(set, "base").cloned().expect("not empty"));
        }
        let mut default = strings(get(set, "default"));
        let mut uses = MapBuilder::new();
        for axis in seq(get(set, "axes")) {
            let old = str_of(&axis, "name").unwrap_or_default().to_string();
            let name = axis_name(&old).to_string();
            let members = strings(get(&axis, "members"));
            if name == "thinking" && members != ["thinking", "no-thinking"] {
                self.notes.note(
                    "axes",
                    format!("{owner}: the thinking axis's members are {members:?}"),
                );
            }
            self.register_axis(
                &name,
                &members,
                str_of(&axis, "description").unwrap_or_default(),
            );
            let on = match default.iter().find(|d| members.contains(d)) {
                Some(on) => on.clone(),
                None => {
                    self.notes.note(
                        "defaults",
                        format!(
                            "{owner}: the axis `{old}` has no default; took `{}`",
                            members[0]
                        ),
                    );
                    members[0].clone()
                }
            };
            default.retain(|d| !members.contains(d));
            uses.push(
                name,
                MapBuilder::new()
                    .with("$offers", texts(members))
                    .with("$default", text(on))
                    .build(),
            );
        }
        let uses = uses.build();
        if !is_empty_map(Some(&uses)) {
            out.push("$axes", uses);
        }
        if !default.is_empty() {
            out.push("$default", texts(default));
        }
        for preset in seq(get(set, "presets")) {
            let name = str_of(&preset, "name").unwrap_or_default().to_string();
            let description = str_of(&preset, "description").unwrap_or_default();
            let mut entry = MapBuilder::new();
            if description.is_empty() {
                self.notes.note(
                    "presets",
                    format!("{owner}: the preset `{name}` has no description (an override)"),
                );
            } else {
                entry.push("$description", text(description));
            }
            if let Some(patch) = get(&preset, "patch")
                && !is_empty_map(Some(patch))
            {
                entry.push("$patch", patch.clone());
            }
            out.push(name, entry.build());
        }
        out
    }

    /// Whether `presets` uses the thinking axis.
    fn uses_thinking(presets: &MapBuilder) -> bool {
        let built = presets.clone().build();
        get(&built, "$axes").is_some_and(|axes| get(axes, "thinking").is_some())
    }

    /// A fixed thinking use, for a `thinking` tag written without an axis:
    /// `true` when one was added.
    fn fix_thinking(
        &mut self,
        presets: &mut MapBuilder,
        thinking: Option<&str>,
        owner: &str,
    ) -> bool {
        let member = match thinking {
            None => return false,
            Some("always") => "thinking",
            Some("never") => "no-thinking",
            Some(other) => {
                self.notes.note(
                    "thinking",
                    format!("{owner}: thinking `{other}` without a thinking axis"),
                );
                return false;
            }
        };
        self.register_axis(
            "thinking",
            &["thinking".into(), "no-thinking".into()],
            "whether the model thinks before it answers",
        );
        let mut built = std::mem::take(presets).build();
        get_or_insert_map(&mut built, "$axes");
        set(
            get_mut(&mut built, "$axes").expect("just set"),
            "thinking",
            MapBuilder::new()
                .with("$offers", texts([member]))
                .with("$default", text(member))
                .build(),
        );
        if get(&built, member).is_none() {
            let description = if member == "thinking" {
                "Think before answering: this model always does."
            } else {
                "Answer at once: this model doesn't think."
            };
            set(
                &mut built,
                member,
                MapBuilder::new()
                    .with("$description", text(description))
                    .build(),
            );
        }
        let OrderedNode::Map(entries) = built else {
            unreachable!()
        };
        for (k, v) in entries {
            presets.push(k, v);
        }
        true
    }
}

/// An old tag set, as the new form's tags, and what it says besides.
#[derive(Default)]
struct Tags {
    tags: Vec<(String, OrderedNode)>,
    thinking: Option<String>,
    groups: Vec<String>,
    version: Option<(String, String)>,
    dtype: Option<String>,
}

fn convert_tags(entries: &[(String, OrderedNode)]) -> Tags {
    let mut out = Tags::default();
    for (key, value) in entries {
        match key.as_str() {
            "base" => {
                let base = matches!(value, OrderedNode::Scalar(Value::Bool(true)));
                out.tags
                    .push(("instruct".into(), OrderedNode::Scalar(Value::Bool(!base))));
            }
            "thinking" => out.thinking = str_of_scalar(value),
            "groups" => out.groups = strings(Some(value)),
            "version" => {
                out.version = Some((
                    str_of(value, "group").unwrap_or_default().to_string(),
                    str_of(value, "version").unwrap_or_default().to_string(),
                ));
            }
            "dtype" => out.dtype = str_of_scalar(value),
            "kit" => {}
            _ => out.tags.push((key.clone(), value.clone())),
        }
    }
    out
}

fn str_of_scalar(node: &OrderedNode) -> Option<String> {
    match node {
        OrderedNode::Scalar(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Group tags, then row tags over them.
fn merged_tags(
    group: Option<&OrderedNode>,
    row: Option<&OrderedNode>,
) -> Vec<(String, OrderedNode)> {
    let mut merged = map_entries(group);
    for (key, value) in map_entries(row) {
        match merged.iter_mut().find(|(k, _)| *k == key) {
            Some((_, slot)) => *slot = value,
            None => merged.push((key, value)),
        }
    }
    merged
}

/// An old base or source, as a new source.
fn base_source(base: &OrderedNode) -> Option<OrderedNode> {
    if let Some(url) = str_of(base, "url") {
        return Some(MapBuilder::new().with("$base_url", text(url)).build());
    }
    let dir = get(base, "local_dir")?;
    Some(
        MapBuilder::new()
            .with(
                "$local_dir",
                MapBuilder::new()
                    .with("$name", text(str_of(dir, "name").unwrap_or_default()))
                    .with("$default", texts(strings(get(dir, "default"))))
                    .build(),
            )
            .build(),
    )
}

/// A resource's own old source, as a new source: a URL names its file.
fn own_source(base: &OrderedNode) -> Option<OrderedNode> {
    match str_of(base, "url") {
        Some(url) => Some(MapBuilder::new().with("$url", text(url)).build()),
        None => base_source(base),
    }
}

fn hub_node(bundle: &OrderedNode) -> Option<OrderedNode> {
    let hub = get(bundle, "hub")?;
    Some(
        MapBuilder::new()
            .with("$repo", text(str_of(hub, "repo").unwrap_or_default()))
            .with(
                "$revision",
                text(str_of(hub, "revision").unwrap_or_default()),
            )
            .build(),
    )
}

/// One old resource's fields.
fn resource_fields(
    resource: &OrderedNode,
    file_key: &str,
) -> MapBuilder {
    let mut entry = MapBuilder::new()
        .with("$kind", text("resource"))
        .with(
            "$file",
            text(str_of(resource, file_key).unwrap_or_default()),
        )
        .with(
            "$format",
            text(str_of(resource, "kind").unwrap_or("unknown")),
        );
    for (old, new) in [
        ("sha256", "$sha256"),
        ("size", "$size"),
        ("dtype", "$dtype"),
    ] {
        if let Some(v) = get(resource, old) {
            entry.push(new, v.clone());
        }
    }
    entry
}

/// A bundle's resources, by key: shards (`K.index` and `K.NNNNN`) as one
/// sharded resource `K`. `sources` are those every resource is given.
fn bundle_resources(
    bundle: &OrderedNode,
    sources: &[OrderedNode],
) -> Vec<(String, OrderedNode)> {
    let resources = seq(get(bundle, "resources"));
    let mut out: Vec<(String, MapBuilder)> = Vec::new();
    let mut shards: BTreeMap<String, Vec<OrderedNode>> = BTreeMap::new();
    for resource in &resources {
        let key = str_of(resource, "key").unwrap_or_default().to_string();
        if let Some((stem, n)) = key.rsplit_once('.')
            && n.chars().all(|c| c.is_ascii_digit())
        {
            let mut shard =
                MapBuilder::new().with("$file", text(str_of(resource, "file").unwrap_or_default()));
            for (old, new) in [("sha256", "$sha256"), ("size", "$size")] {
                if let Some(v) = get(resource, old) {
                    shard.push(new, v.clone());
                }
            }
            shards
                .entry(stem.to_string())
                .or_default()
                .push(shard.build());
            continue;
        }
        let name = key.strip_suffix(".index").unwrap_or(&key).to_string();
        let mut entry = resource_fields(resource, "file");
        let mut all_sources: Vec<OrderedNode> = seq(get(resource, "sources"))
            .iter()
            .filter_map(own_source)
            .collect();
        all_sources.extend(sources.iter().cloned());
        if !all_sources.is_empty() {
            entry.push("$sources", OrderedNode::Seq(all_sources));
        }
        out.push((name, entry));
    }
    // A shard's dtype is its resource's.
    for resource in &resources {
        let key = str_of(resource, "key").unwrap_or_default();
        if let (Some((stem, _)), Some(dtype)) = (key.rsplit_once('.'), get(resource, "dtype"))
            && let Some((_, entry)) = out.iter_mut().find(|(n, _)| n == stem)
            && !entry.contains("$dtype")
        {
            entry.push("$dtype", dtype.clone());
        }
    }
    out.into_iter()
        .map(|(name, mut entry)| {
            if let Some(shards) = shards.remove(&name) {
                entry.push("$shards", OrderedNode::Seq(shards));
            }
            (name, entry.build())
        })
        .collect()
}

/// The stated geometry of `kit`'s prefab `name`, when the kit's contract
/// takes one.
fn geometry(
    kit: &str,
    name: &str,
) -> Result<Option<Value>, String> {
    use bunsen::kits::{
        images::resnet::pretrained::RESNET_PREFABS,
        speech::asr::{
            qwen3_asr::pretrained::QWEN3_ASR_PREFABS,
            voxtral::pretrained::VOXTRAL_PREFABS,
        },
    };
    let found = match kit {
        "resnet" => RESNET_PREFABS
            .try_lookup_prefab(name)
            .map(|p| serde_json::to_value(p.to_config())),
        "qwen3-asr" => QWEN3_ASR_PREFABS
            .try_lookup_prefab(name)
            .map(|p| serde_json::to_value(p.to_config())),
        "voxtral" => VOXTRAL_PREFABS
            .try_lookup_prefab(name)
            .map(|p| serde_json::to_value(p.to_config())),
        _ => return Ok(None),
    };
    found
        .map_err(|e| format!("{kit}: the prefab `{name}`: {e}"))?
        .map(Some)
        .map_err(|e| format!("{kit}: the prefab `{name}` doesn't serialize: {e}"))
}

impl Converter {
    /// One kit's source as a probe-contract layer.
    fn kit_layer(
        &mut self,
        kit: &Kit,
        source: &OrderedNode,
    ) -> Result<OrderedNode, String> {
        let groups = seq(get(source, "groups"));
        let bundles = seq(get(source, "bundles"));
        let families = seq(get(source, "families"));
        let bundle = |name: &str| {
            bundles
                .iter()
                .find(|b| str_of(b, "name") == Some(name))
                .cloned()
                .ok_or_else(|| format!("{}: no bundle `{name}`", kit.kit))
        };
        let publishers: Vec<String> = groups
            .iter()
            .map(|g| str_of(g, "name").unwrap_or_default().to_string())
            .collect();
        let shared_family = publishers.len() > 1;
        let rows: Vec<(OrderedNode, OrderedNode)> = groups
            .iter()
            .flat_map(|g| seq(get(g, "rows")).into_iter().map(move |r| (g.clone(), r)))
            .collect();
        let uses = |name: &str| {
            rows.iter()
                .filter(|(_, r)| strings(get(r, "bundles")).iter().any(|b| b == name))
                .count()
        };

        // The family: its template classes, groups and shared resources.
        let mut family = MapBuilder::new()
            .with("$kind", text("namespace"))
            .with("$facet", text("family"));
        let family_description = if shared_family {
            SHARED_FAMILIES
                .iter()
                .find(|(f, _)| *f == kit.family)
                .map(|(_, d)| d.to_string())
                .ok_or_else(|| format!("{}: no description for its family", kit.kit))?
        } else {
            sentence(str_of(&groups[0], "description").unwrap_or_default())
        };
        family.push("$description", text(family_description));
        let requires: Vec<String> = kit
            .required
            .iter()
            .filter_map(|k| match *k {
                "base" => Some("instruct".to_string()),
                "dtype" | "kit" => None,
                other => Some(other.to_string()),
            })
            .collect();
        if !requires.is_empty() {
            family.push("$requires", texts(requires));
        }

        let mut class_thinking: BTreeMap<String, bool> = BTreeMap::new();
        for old in &families {
            let name = str_of(old, "name").unwrap_or_default().to_string();
            let owner = format!("{} family {name}", kit.kit);
            let tags = convert_tags(&map_entries(get(old, "tags")));
            let mut presets = self.presets(get(old, "presets"), &owner);
            let handled = Converter::uses_thinking(&presets)
                || self.fix_thinking(&mut presets, tags.thinking.as_deref(), &owner);
            class_thinking.insert(name.clone(), handled);
            let mut class = MapBuilder::new()
                .with("$kind", text("template_class"))
                .with(
                    "$description",
                    text(sentence(str_of(old, "description").unwrap_or_default())),
                );
            for (k, v) in tags.tags {
                class.push(k, v);
            }
            let presets = presets.build();
            if !is_empty_map(Some(&presets)) {
                class.push("$presets", presets);
            }
            family.push(name, class.build());
        }

        let mut group_names: Vec<String> = Vec::new();
        let mut shared_resources: Vec<String> = Vec::new();
        let mut series: BTreeMap<&str, MapBuilder> = BTreeMap::new();
        let mut releases: BTreeMap<&str, ()> = BTreeMap::new();
        let mut by_publisher: BTreeMap<String, MapBuilder> = BTreeMap::new();
        let mut family_builds: Vec<(String, OrderedNode)> = Vec::new();

        for (group, row) in &rows {
            let publisher = str_of(group, "name").unwrap_or_default().to_string();
            let name = str_of(row, "name").unwrap_or_default().to_string();
            let owner = format!("{}:{publisher}/{name}", kit.kit);
            let tags = convert_tags(&merged_tags(get(group, "tags"), get(row, "tags")));
            let family_name = str_of(row, "family").map(str::to_string);
            let in_series = SERIES
                .iter()
                .find(|(k, prefix, _)| *k == kit.kit && name.starts_with(prefix))
                .map(|(_, prefix, description)| (*prefix, *description));

            let mut build = MapBuilder::new().with("$kind", text("build")).with(
                "$description",
                text(sentence(str_of(row, "description").unwrap_or_default())),
            );
            for key in ["license", "origin"] {
                if let Some(v) = str_of(row, key).or_else(|| str_of(group, key)) {
                    build.push(format!("${key}"), text(v));
                }
            }
            if let Some(class) = &family_name {
                let class = if shared_family {
                    format!("{}/{class}", kit.family)
                } else {
                    class.clone()
                };
                build.push("$template_class", text(class));
            }
            build.push("$contract", text(kit.contract));

            let mut joins: Vec<OrderedNode> = Vec::new();
            if shared_family {
                joins.push(text(kit.family));
            }
            for g in &tags.groups {
                if !group_names.contains(g) {
                    group_names.push(g.clone());
                }
                joins.push(text(if shared_family {
                    format!("{}/{g}", kit.family)
                } else {
                    g.clone()
                }));
            }
            if let Some((sequence, version)) = &tags.version {
                let release = RELEASES
                    .iter()
                    .find(|r| r.sequence == sequence)
                    .ok_or_else(|| format!("{owner}: no release sequence `{sequence}`"))?;
                let date = release
                    .dates
                    .iter()
                    .find(|(v, _)| v == version)
                    .map(|(_, d)| *d)
                    .ok_or_else(|| format!("{owner}: no date for {sequence} {version}"))?;
                releases.insert(release.sequence, ());
                joins.push(
                    MapBuilder::new()
                        .with("$group", text(release.name))
                        .with("$released", text(date))
                        .build(),
                );
                self.notes.note(
                    "releases",
                    format!("{owner}: {sequence} {version} released {date}, by hand"),
                );
            }
            if !joins.is_empty() {
                build.push("$join", OrderedNode::Seq(joins));
            }
            for (k, v) in &tags.tags {
                build.push(k.clone(), v.clone());
            }

            if let Some(prefab) = str_of(row, "prefab")
                && let Some(geometry) = geometry(kit.kit, prefab)?
            {
                build.push(
                    "$args",
                    MapBuilder::new().with("geometry", value(geometry)).build(),
                );
            }

            // Resources: shared bundles beside the family, the rest the
            // build's own.
            let mut bindings = MapBuilder::new();
            let mut own: Vec<(String, OrderedNode)> = Vec::new();
            let mut build_hub: Option<OrderedNode> = None;
            let own_bundles: Vec<OrderedNode> = strings(get(row, "bundles"))
                .iter()
                .filter(|b| uses(b) == 1)
                .map(|b| bundle(b))
                .collect::<Result<_, _>>()?;
            let hub_bundles = own_bundles
                .iter()
                .filter(|b| get(b, "hub").is_some())
                .count();
            for bundle_name in strings(get(row, "bundles")) {
                let b = bundle(&bundle_name)?;
                let mut sources: Vec<OrderedNode> = Vec::new();
                let bases: Vec<OrderedNode> = seq(get(&b, "bases"))
                    .iter()
                    .filter_map(base_source)
                    .collect();
                let resources_have_sources = seq(get(&b, "resources"))
                    .iter()
                    .any(|r| get(r, "sources").is_some());
                let hub = hub_node(&b);
                let shared = uses(&bundle_name) > 1;
                match &hub {
                    Some(hub)
                        if !shared
                            && hub_bundles == 1
                            && bases.is_empty()
                            && !resources_have_sources =>
                    {
                        build_hub = Some(hub.clone());
                    }
                    Some(hub) => sources.push(MapBuilder::new().with("$hub", hub.clone()).build()),
                    None => {}
                }
                sources.extend(bases);
                let resources = bundle_resources(&b, &sources);
                if shared {
                    let [(key, resource)] = resources.as_slice() else {
                        return Err(format!(
                            "{owner}: the shared bundle `{bundle_name}` has more than one resource"
                        ));
                    };
                    let resource_name = bundle_name
                        .split_once('/')
                        .map_or(bundle_name.as_str(), |(_, tail)| tail)
                        .to_string();
                    if !shared_resources.contains(&resource_name) {
                        shared_resources.push(resource_name.clone());
                        family.push(resource_name.clone(), resource.clone());
                    }
                    bindings.push(
                        key.clone(),
                        text(if shared_family {
                            format!("{}/{resource_name}", kit.family)
                        } else {
                            resource_name
                        }),
                    );
                } else {
                    for (key, mut resource) in resources {
                        if key == "checkpoint"
                            && let Some(short) = &tags.dtype
                            && get(&resource, "$dtype").is_none()
                        {
                            set(&mut resource, "$dtype", text(long_dtype(short)));
                            self.notes.note(
                                "dtype",
                                format!("{owner}: its dtype tag `{short}` is its checkpoint's"),
                            );
                        }
                        bindings.push(key.clone(), text(key.clone()));
                        own.push((key, resource));
                    }
                }
            }
            if let Some(hub) = build_hub {
                build.push("$hub", hub);
            }

            let probes: Vec<OrderedNode> = seq(get(row, "probe"))
                .iter()
                .map(|p| {
                    let into: Vec<String> = strings(get(p, "into"))
                        .into_iter()
                        .map(|i| {
                            if i == "base" {
                                "$common".to_string()
                            } else {
                                i
                            }
                        })
                        .collect();
                    MapBuilder::new()
                        .with("$probe", text(str_of(p, "probe").unwrap_or_default()))
                        .with("$resource", text(str_of(p, "file").unwrap_or_default()))
                        .with("$into", texts(into))
                        .with_opt("$token", str_of(p, "token").map(text))
                        .build()
                })
                .collect();
            if !probes.is_empty() {
                build.push("$probes", OrderedNode::Seq(probes));
            }
            build.push("$resources", bindings.build());

            let mut presets = self.presets(get(row, "presets"), &owner);
            let class_handles = family_name
                .as_ref()
                .is_some_and(|f| class_thinking.get(f).copied().unwrap_or(false));
            if !Converter::uses_thinking(&presets) && !class_handles {
                self.fix_thinking(&mut presets, tags.thinking.as_deref(), &owner);
            }
            let presets = presets.build();
            if !is_empty_map(Some(&presets)) {
                build.push("$presets", presets);
            }
            for (key, resource) in own {
                build.push(key, resource);
            }

            let aliases: Vec<(String, OrderedNode)> = strings(get(row, "aliases"))
                .into_iter()
                .map(|alias| {
                    (
                        alias,
                        MapBuilder::new()
                            .with("$kind", text("alias"))
                            .with(
                                "$ref",
                                MapBuilder::new()
                                    .with("$path", text(name.clone()))
                                    .with("$kind", text("build"))
                                    .build(),
                            )
                            .build(),
                    )
                })
                .collect();

            let mut placed = vec![(name.clone(), build.build())];
            placed.extend(aliases);
            match (shared_family, in_series) {
                (false, Some((prefix, description))) => {
                    let ns = series.entry(prefix).or_insert_with(|| {
                        MapBuilder::new()
                            .with("$kind", text("namespace"))
                            .with("$description", text(description))
                    });
                    for (k, v) in placed {
                        ns.push(k, v);
                    }
                }
                (false, None) => family_builds.extend(placed),
                (true, _) => {
                    let ns = by_publisher.entry(publisher.clone()).or_insert_with(|| {
                        MapBuilder::new()
                            .with("$kind", text("namespace"))
                            .with("$merge", text("update"))
                    });
                    for (k, v) in placed {
                        ns.push(k, v);
                    }
                }
            }
        }

        for g in &group_names {
            let description = GROUPS
                .iter()
                .find(|(n, _)| n == g)
                .map(|(_, d)| *d)
                .ok_or_else(|| format!("{}: no description for the group `{g}`", kit.kit))?;
            family.push(
                g.clone(),
                MapBuilder::new()
                    .with("$kind", text("namespace"))
                    .with("$description", text(description))
                    .build(),
            );
        }
        for (prefix, mut ns) in series {
            for release in RELEASES {
                if release.series == prefix && releases.contains_key(release.sequence) {
                    let mut with_release = MapBuilder::new();
                    let built = ns.build();
                    let OrderedNode::Map(entries) = built else {
                        unreachable!()
                    };
                    let mut entries = entries.into_iter();
                    for (k, v) in entries.by_ref().take(2) {
                        with_release.push(k, v);
                    }
                    with_release.push(
                        release.name,
                        MapBuilder::new()
                            .with("$kind", text("release"))
                            .with("$description", text(release.description))
                            .build(),
                    );
                    for (k, v) in entries {
                        with_release.push(k, v);
                    }
                    ns = with_release;
                }
            }
            family.push(prefix, ns.build());
        }
        // An alias whose name the family already holds goes up a scope, into
        // the publisher.
        let mut outer_aliases: Vec<(String, OrderedNode)> = Vec::new();
        for (k, mut v) in family_builds {
            if kind_of(&v) == Some("alias") && family.contains(&k) {
                let target = get(&v, "$ref")
                    .and_then(|r| str_of(r, "$path"))
                    .unwrap_or_default()
                    .to_string();
                let path = format!("{}/{target}", kit.family);
                if let Some(r) = get_mut(&mut v, "$ref") {
                    set(r, "$path", text(path));
                }
                self.notes.note(
                    "aliases",
                    format!(
                        "{}: the alias `{k}` is the publisher's: the family holds the name",
                        kit.kit
                    ),
                );
                outer_aliases.push((k, v));
            } else {
                family.push(k, v);
            }
        }

        let mut layer = MapBuilder::new().with(
            "$description",
            text(capitalized(&sentence(
                str_of(source, "description").unwrap_or_default(),
            ))),
        );
        if shared_family {
            layer.push(kit.family, family.build());
            for publisher in &publishers {
                if let Some(ns) = by_publisher.remove(publisher) {
                    layer.push(publisher.clone(), ns.build());
                }
            }
        } else {
            let mut publisher = MapBuilder::new()
                .with("$kind", text("namespace"))
                .with("$merge", text("update"))
                .with(kit.family, family.build());
            for (k, v) in outer_aliases {
                publisher.push(k, v);
            }
            layer.push(publishers[0].clone(), publisher.build());
        }
        Ok(layer.build())
    }

    /// The prelude: the facets, the registered axes, the shared tags and
    /// the template-class root.
    fn prelude(&self) -> OrderedNode {
        let mut prelude = MapBuilder::new()
            .with(
                "$description",
                text(
                    "bunsen's vocabulary: the contracts, facets, registered axes and tags, and \
                     the template-class root.",
                ),
            )
            .with(
                "publisher",
                MapBuilder::new()
                    .with("$kind", text("facet"))
                    .with("$description", text("Who publishes the checkpoints."))
                    .build(),
            )
            .with(
                "family",
                MapBuilder::new()
                    .with("$kind", text("facet"))
                    .with("$description", text("A line of related architectures."))
                    .build(),
            );
        for (name, decl) in &self.axes {
            let mut axis = MapBuilder::new()
                .with("$kind", text("axis"))
                .with(
                    "$axis",
                    text(if decl.bool_kind {
                        "bool"
                    } else {
                        "categorical"
                    }),
                )
                .with("$description", text(sentence(&decl.description)));
            if !decl.bool_kind {
                axis.push("$members", texts(decl.members.clone()));
            }
            prelude.push(name.clone(), axis.build());
        }
        for def in VOCABULARY {
            let (key, description) = match def.key {
                "kit" | "groups" | "version" | "dtype" | "thinking" => continue,
                "base" => (
                    "instruct",
                    "Tuned to follow instructions; `false` for a pretrained base checkpoint."
                        .to_string(),
                ),
                key => (key, sentence(def.description)),
            };
            let type_expr = match def.shape {
                TagShape::Bool => text("$bool"),
                TagShape::Count => text("$count"),
                TagShape::Text => text("$text"),
                TagShape::OneOf(members) => MapBuilder::new()
                    .with("$one_of", texts(members.iter().copied()))
                    .build(),
                TagShape::TextList => MapBuilder::new().with("$list", text("$text")).build(),
                TagShape::Languages => text("$languages"),
                TagShape::Scores => MapBuilder::new().with("$map", text("$number")).build(),
                TagShape::Version => continue,
            };
            prelude.push(
                key,
                MapBuilder::new()
                    .with("$kind", text("tag"))
                    .with("$type", type_expr)
                    .with("$description", text(description))
                    .build(),
            );
        }
        prelude.push(
            "template_class_root",
            MapBuilder::new()
                .with("$kind", text("template_class"))
                .with(
                    "$description",
                    text("The root of every template-class chain."),
                )
                .with("$required_facets", texts(["publisher", "family"]))
                .build(),
        );
        prelude.build()
    }
}

/// The commons: the publishers.
fn commons() -> OrderedNode {
    let mut commons = MapBuilder::new().with(
        "$description",
        text("The publishers, which more than one kit's builds are in."),
    );
    for (name, description) in PUBLISHERS {
        commons.push(
            *name,
            MapBuilder::new()
                .with("$kind", text("namespace"))
                .with("$facet", text("publisher"))
                .with("$description", text(*description))
                .build(),
        );
    }
    commons.build()
}

/// Converts `validation/catalogs/` into `out`: a manifest, the prelude, the
/// commons and one layer per kit.
///
/// # Errors
/// A source that doesn't read, or says what the conversion can't carry.
pub fn convert(
    catalogs: &Path,
    out: &Path,
) -> Result<Notes, String> {
    let mut converter = Converter::default();
    let mut layers: Vec<(String, Option<&str>, String, OrderedNode)> = Vec::new();
    for kit in kits() {
        let path = catalogs.join(format!("{}.yml", kit.kit));
        let text_in =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let source = OrderedNode::from_yaml(&text_in).map_err(|e| e.to_string())?;
        let layer = converter.kit_layer(&kit, &source)?;
        layers.push((
            kit.kit.to_string(),
            Some(kit.feature),
            format!(
                "# The {} kit's builds: a probe-contract source of the built-in catalog.\n",
                kit.kit
            ),
            layer,
        ));
    }
    layers.insert(
        0,
        (
            "commons".into(),
            None,
            "# The publishers, which more than one kit's builds are in.\n".into(),
            commons(),
        ),
    );
    layers.insert(
        0,
        (
            "prelude".into(),
            None,
            "# The base catalog: the facets, the registered axes and tags, and the\n\
             # template-class root. The builder gives it the kits' contracts, as its\n\
             # `$contracts`.\n"
                .into(),
            converter.prelude(),
        ),
    );

    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut manifest = String::from(
        "# The built-in catalog's layers, in import order, each with the bunsen feature it\n\
         # needs; the first is the base catalog.\n",
    );
    for (name, feature, header, layer) in &layers {
        let entry = ManifestEntry {
            file: format!("{name}.yml"),
            feature: feature.map(str::to_string),
        };
        match &entry.feature {
            Some(f) => manifest.push_str(&format!("{} {f}\n", entry.file)),
            None => manifest.push_str(&format!("{}\n", entry.file)),
        }
        let yaml = layer.to_yaml().map_err(|e| e.to_string())?;
        let path = out.join(&entry.file);
        fs::write(&path, format!("{header}{yaml}"))
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let path = out.join(MANIFEST);
    fs::write(&path, manifest).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(converter.notes)
}
