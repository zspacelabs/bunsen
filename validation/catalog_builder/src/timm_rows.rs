//! `cargo catalog timm-rows`: a kit's timm hub builds, from the Hub.
//!
//! timm publishes each set of weights as a Hub repo named for its
//! architecture and its tag, `timm/<arch>.<tag>`
//! (`timm/resnet50.a1_in1k`). A kit that loads timm's repos builds an
//! architecture from its prefab of the same name, so the rows it can load are
//! the repos of its prefabs' names. This module writes those rows into the
//! kit's probe-contract source, between the [`BEGIN`] and [`END`] lines,
//! never by hand:
//!
//! - every repo the Hub lists for each of the kit's architectures, in prefab
//!   order, then by tag;
//! - one that has a `model.safetensors` and a `config.json` whose head has an
//!   `ImageNet` label set's class count ([`LABEL_SETS`]), so the labels the row
//!   binds are the head's; or, for a kit that loads them, any other head (none,
//!   or a projection), with no labels;
//! - except a `tv_in1k` repo whose architecture has a `torchvision/` row of its
//!   name: that row holds the same weights.
//!
//! Each row is pinned to the repo's revision and card license, as the Hub's
//! model API gives them. A row the source already has keeps its pins, so the
//! Hub is asked only about new repos; `refresh` pins every row again. A row
//! the Hub no longer lists is an error: the source names it, and a person
//! decides. Every hub build of the source's timm namespace is a generated
//! row.
//!
//! The digests and sizes are not written here: `cargo catalog update` reads
//! them from the Hub's listing, as for any hub build.

use std::collections::BTreeMap;

use bunsen::{
    data::{
        catalog::document::OrderedNode,
        pretrained::Prefabs,
        resources::hub::{
            HUB_ORIGIN,
            HubRepo,
        },
    },
    kits::images::classification::resnet::ResNetContractConfig,
};
use serde::Deserialize;

use crate::{
    remote::Remote,
    tree::{
        children,
        get,
        kind_of,
        string_field,
    },
};

/// The line that opens a source's generated rows, at the timm namespace's
/// indent.
pub const BEGIN: &str = "# BEGIN `cargo catalog timm-rows`: generated, don't edit by hand.";

/// The line that closes a source's generated rows.
pub const END: &str = "# END `cargo catalog timm-rows`.";

/// The Hub organization timm publishes under.
const TIMM_ORG: &str = "timm";

/// The tag of torchvision's own weights, as timm republishes them.
const TORCHVISION_TAG: &str = "tv_in1k";

/// The `ImageNet` label sets the catalog holds, by their class count: the
/// subsets of timm's `infer_imagenet_subset`
/// ([`imagenet_subset_for`](bunsen::kits::images::support::labels::imagenet_subset_for)),
/// each the `synsets-` resource of its name.
pub const LABEL_SETS: &[(u64, &str)] = &[
    (1000, "/imagenet/synsets-1k"),
    (11221, "/imagenet/synsets-21k-miil"),
    (11821, "/imagenet/synsets-12k"),
    (21843, "/imagenet/synsets-21k-goog"),
];

/// The readable names of every label set's synsets.
const LABEL_NAMES: &str = "/imagenet/lemmas";

/// The label set of a head of `classes` classes, if the catalog holds one.
fn labels_for(classes: u64) -> Option<&'static str> {
    LABEL_SETS
        .iter()
        .find(|(n, _)| *n == classes)
        .map(|(_, labels)| *labels)
}

/// A kit whose timm rows are generated.
#[derive(Clone, Debug)]
pub struct TimmKit {
    /// The kit's name, and its source's layer name (`resnet`).
    pub name: &'static str,

    /// The contract its builds name (`resnet-classifier`).
    pub contract: &'static str,

    /// The namespaces its builds join.
    pub join: &'static [&'static str],

    /// The architectures it builds, by timm's names: its prefabs'.
    pub architectures: Vec<&'static str>,

    /// Whether a row may have a head no label set fits: none (a headless
    /// tower), or a projection (CLIP's). Such a row binds no labels.
    pub unlabelled: bool,
}

impl TimmKit {
    /// The source file of the kit's builds, in `probe_contracts/`.
    pub fn source_file(&self) -> String {
        format!("{}.yml", self.name)
    }
}

/// The kits whose timm rows are generated.
pub fn timm_kits() -> Vec<TimmKit> {
    vec![TimmKit {
        name: "resnet",
        contract: "resnet-classifier",
        join: &["resnet"],
        architectures: ResNetContractConfig::prefab_names(),
        unlabelled: false,
    }]
}

/// One generated build's pins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimmRow {
    /// The architecture (`resnet50`).
    pub architecture: String,

    /// The weights' tag (`a1_in1k`).
    pub tag: String,

    /// The commit it is pinned to.
    pub revision: String,

    /// Its card's license.
    pub license: String,

    /// The label set it binds ([`LABEL_SETS`]), if its head has one.
    pub labels: Option<String>,
}

impl TimmRow {
    /// The build's name, and the repo's name in timm's organization:
    /// `resnet50.a1_in1k`.
    pub fn name(&self) -> String {
        format!("{}.{}", self.architecture, self.tag)
    }

    /// The build, as the source writes it, at the timm namespace's indent.
    fn render(
        &self,
        kit: &TimmKit,
    ) -> String {
        let name = self.name();
        let join: String = kit.join.iter().map(|j| format!("      - {j}\n")).collect();
        let labels = self.labels.as_ref().map_or(String::new(), |labels| {
            format!("      labels: {labels}\n      label_names: {LABEL_NAMES}\n")
        });
        format!(
            "  {name}:
    $kind: build
    $description: \"timm's `{arch}`, its `{tag}` weights.\"
    $license: {license}
    $origin: {HUB_ORIGIN}/{TIMM_ORG}/{name}
    $contract: {contract}
    $join:
{join}    $hub:
      $repo: {TIMM_ORG}/{name}
      $revision: {revision}
    $resources:
      checkpoint: checkpoint
      config: config
{labels}    checkpoint:
      $kind: resource
      $file: model.safetensors
      $format: safetensors
    config:
      $kind: resource
      $file: config.json
      $format: json
",
            arch = self.architecture,
            tag = self.tag,
            license = self.license,
            contract = kit.contract,
            revision = self.revision,
        )
    }
}

/// A repo the Hub lists that is not a row, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippedRepo {
    /// `timm/<arch>.<tag>`.
    pub repo: String,

    /// Why it is not a row.
    pub reason: String,
}

/// What [`generate`] writes, and what it left out.
#[derive(Clone, Debug)]
pub struct Generated {
    /// The source, its generated rows replaced.
    pub text: String,

    /// The rows, in order.
    pub rows: Vec<TimmRow>,

    /// The repos that are not rows.
    pub skipped: Vec<SkippedRepo>,
}

/// One model of the Hub's model listing.
#[derive(Debug, Deserialize)]
struct ListedModel {
    id: String,
}

/// What the Hub's model API says of one repo.
#[derive(Debug, Deserialize)]
struct ModelInfo {
    sha: String,
    #[serde(default, rename = "cardData")]
    card_data: Option<CardData>,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
struct CardData {
    license: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
}

/// The tags of `architecture`'s repos in timm's organization, sorted: every
/// page of the Hub's search, the names that are `<architecture>.<tag>`.
///
/// # Errors
/// A page that can't be fetched or read.
pub fn listed_tags(
    remote: &dyn Remote,
    architecture: &str,
) -> Result<Vec<String>, String> {
    let prefix = format!("{TIMM_ORG}/{architecture}.");
    let mut url = Some(format!(
        "{HUB_ORIGIN}/api/models?author={TIMM_ORG}&search={architecture}&limit=1000"
    ));
    let mut tags = Vec::new();
    while let Some(page) = url {
        let (body, next) = remote.get_page(&page)?;
        let models: Vec<ListedModel> = serde_json::from_slice(&body)
            .map_err(|e| format!("{page}: not a model listing: {e}"))?;
        tags.extend(
            models
                .into_iter()
                .filter_map(|m| m.id.strip_prefix(&prefix).map(str::to_string)),
        );
        url = next;
    }
    tags.sort();
    tags.dedup();
    Ok(tags)
}

/// The rows the source's timm namespace has, by name: their architecture
/// and tag from the name, their pins from the build.
fn existing_rows(source: &OrderedNode) -> Result<BTreeMap<String, TimmRow>, String> {
    let Some(timm) = get(source, TIMM_ORG) else {
        return Ok(BTreeMap::new());
    };
    let mut rows = BTreeMap::new();
    for (name, build) in children(timm) {
        if kind_of(build) != Some("build") {
            continue;
        }
        let Some(hub) = get(build, "$hub") else {
            continue;
        };
        let (architecture, tag) = name
            .split_once('.')
            .ok_or_else(|| format!("timm/{name}: a hub row is named `<arch>.<tag>`"))?;
        rows.insert(
            name.clone(),
            TimmRow {
                architecture: architecture.to_string(),
                tag: tag.to_string(),
                revision: string_field(hub, "$revision")
                    .map_err(|e| format!("timm/{name}: {e}"))?
                    .to_string(),
                license: string_field(build, "$license")
                    .map_err(|e| format!("timm/{name}: {e}"))?
                    .to_string(),
                labels: match get(build, "$resources") {
                    Some(resources) if get(resources, "labels").is_some() => Some(
                        string_field(resources, "labels")
                            .map_err(|e| format!("timm/{name}: {e}"))?
                            .to_string(),
                    ),
                    _ => None,
                },
            },
        );
    }
    Ok(rows)
}

/// A new row for `timm/<architecture>.<tag>`, pinned from the Hub, or why it
/// is not one.
fn new_row(
    remote: &dyn Remote,
    kit: &TimmKit,
    architecture: &str,
    tag: &str,
) -> Result<Result<TimmRow, String>, String> {
    let repo = format!("{TIMM_ORG}/{architecture}.{tag}");
    let url = format!("{HUB_ORIGIN}/api/models/{repo}");
    let info: ModelInfo = serde_json::from_slice(&remote.get(&url)?)
        .map_err(|e| format!("{url}: not a model's info: {e}"))?;
    for file in ["model.safetensors", "config.json"] {
        if !info.siblings.iter().any(|s| s.rfilename == file) {
            return Ok(Err(format!("no `{file}`")));
        }
    }
    let Some(license) = info.card_data.and_then(|c| c.license) else {
        return Ok(Err("its card states no license".into()));
    };
    let pinned = HubRepo::new(&repo, &info.sha);
    let config_url = pinned.file_url("config.json");
    let config: serde_json::Value = serde_json::from_slice(&remote.get(&config_url)?)
        .map_err(|e| format!("{config_url}: not JSON: {e}"))?;
    let classes = config
        .get("num_classes")
        .and_then(serde_json::Value::as_u64);
    let Some(classes) = classes else {
        return Ok(Err("its `config.json` states no `num_classes`".into()));
    };
    let labels = labels_for(classes);
    if labels.is_none() && !kit.unlabelled {
        return Ok(Err(format!(
            "{classes} classes: no ImageNet label set of the catalog's has that many"
        )));
    }
    Ok(Ok(TimmRow {
        architecture: architecture.to_string(),
        tag: tag.to_string(),
        revision: info.sha,
        license,
        labels: labels.map(str::to_string),
    }))
}

/// `source_text` with `kit`'s generated rows written from the Hub.
///
/// # Errors
/// A source that doesn't read or has no [`BEGIN`] / [`END`] lines, a Hub
/// answer that can't be fetched or read, or a row of the source the Hub no
/// longer lists.
pub fn generate(
    remote: &dyn Remote,
    kit: &TimmKit,
    source_text: &str,
    refresh: bool,
) -> Result<Generated, String> {
    let source = OrderedNode::from_yaml(source_text).map_err(|e| e.to_string())?;
    let mut existing = existing_rows(&source)?;
    let torchvision: Vec<&String> = get(&source, "torchvision")
        .map(|ns| children(ns).map(|(name, _)| name).collect())
        .unwrap_or_default();

    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for &architecture in &kit.architectures {
        for tag in listed_tags(remote, architecture)? {
            let repo = format!("{TIMM_ORG}/{architecture}.{tag}");
            let name = format!("{architecture}.{tag}");
            if tag == TORCHVISION_TAG && torchvision.iter().any(|t| *t == architecture) {
                skipped.push(SkippedRepo {
                    repo,
                    reason: format!("`torchvision/{architecture}` holds these weights"),
                });
                continue;
            }
            match existing.remove(&name) {
                Some(row) if !refresh => rows.push(row),
                _ => match new_row(remote, kit, architecture, &tag)? {
                    Ok(row) => rows.push(row),
                    Err(reason) => skipped.push(SkippedRepo { repo, reason }),
                },
            }
        }
    }
    if let Some(gone) = existing.keys().next() {
        return Err(format!(
            "timm/{gone}: not a repo the Hub lists for the kit's architectures; remove its row"
        ));
    }

    let rendered: String = rows.iter().map(|row| row.render(kit)).collect();
    let text = splice(source_text, &rendered)?;
    Ok(Generated {
        text,
        rows,
        skipped,
    })
}

/// `text` with the lines between its [`BEGIN`] and [`END`] lines replaced
/// by `rows`.
///
/// # Errors
/// When it hasn't one of each, in order.
fn splice(
    text: &str,
    rows: &str,
) -> Result<String, String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let at = |marker: &str| {
        let found: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.trim() == marker)
            .map(|(i, _)| i)
            .collect();
        match found.as_slice() {
            [i] => Ok(*i),
            _ => Err(format!("the source has one `{marker}` line")),
        }
    };
    let (begin, end) = (at(BEGIN)?, at(END)?);
    if end < begin {
        return Err(format!("`{END}` comes after `{BEGIN}`"));
    }
    let mut out: String = lines[..=begin].concat();
    out.push_str(rows);
    out.push_str(&lines[end..].concat());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::testing::FakeRemote;

    /// A kit of two architectures.
    fn kit() -> TimmKit {
        TimmKit {
            architectures: vec!["resnet18", "resnet50"],
            ..timm_kits().remove(0)
        }
    }

    /// The source: a torchvision `resnet18`, one generated row, an alias of
    /// it after the generated rows.
    fn source(row: &str) -> String {
        format!(
            "torchvision:
  $kind: namespace
  resnet18:
    $kind: build
    $contract: resnet-classifier
timm:
  $kind: namespace
  $merge: update
  {BEGIN}
{row}  {END}
  resnet18_a1:
    $kind: alias
    $ref:
      $path: resnet18.a1_in1k
      $kind: build
"
        )
    }

    /// The pinned row of `resnet18.a1_in1k`.
    fn a1() -> TimmRow {
        TimmRow {
            architecture: "resnet18".into(),
            tag: "a1_in1k".into(),
            revision: "r18a1".into(),
            license: "apache-2.0".into(),
            labels: Some("/imagenet/synsets-1k".into()),
        }
    }

    /// The Hub: `resnet18`'s `a1_in1k` and `tv_in1k`, `resnet18d`'s (a
    /// search hit of another architecture), and `resnet50`'s `a1_in1k` and
    /// headless `bamboo`, its license-less `x`, its 12k-class `in12k` and its
    /// 512-wide `proj`.
    fn hub() -> FakeRemote {
        let mut remote = FakeRemote::default();
        let mut listing = |arch: &str, ids: &[&str]| {
            let models: Vec<_> = ids.iter().map(|id| serde_json::json!({"id": id})).collect();
            remote.bodies.insert(
                format!("{HUB_ORIGIN}/api/models?author=timm&search={arch}&limit=1000"),
                serde_json::to_vec(&models).unwrap(),
            );
        };
        listing(
            "resnet18",
            &[
                "timm/resnet18.tv_in1k",
                "timm/resnet18.a1_in1k",
                "timm/resnet18d.ra2_in1k",
            ],
        );
        listing(
            "resnet50",
            &[
                "timm/resnet50.x",
                "timm/resnet50.bamboo",
                "timm/resnet50.a1_in1k",
                "timm/resnet50.in12k",
                "timm/resnet50.proj",
            ],
        );
        let mut model = |name: &str, sha: &str, license: Option<&str>, classes: u64| {
            let files = ["config.json", "model.safetensors"];
            let info = serde_json::json!({
                "sha": sha,
                "cardData": {"license": license},
                "siblings": files.iter().map(|f| serde_json::json!({"rfilename": f}))
                    .collect::<Vec<_>>(),
            });
            remote.bodies.insert(
                format!("{HUB_ORIGIN}/api/models/timm/{name}"),
                serde_json::to_vec(&info).unwrap(),
            );
            remote.bodies.insert(
                HubRepo::new(format!("timm/{name}"), sha).file_url("config.json"),
                serde_json::to_vec(&serde_json::json!({"num_classes": classes})).unwrap(),
            );
        };
        model("resnet18.a1_in1k", "r18a1-new", Some("apache-2.0"), 1000);
        model("resnet50.a1_in1k", "r50a1", Some("apache-2.0"), 1000);
        model("resnet50.bamboo", "bamboo", Some("apache-2.0"), 0);
        model("resnet50.x", "x", None, 1000);
        model("resnet50.in12k", "in12k", Some("apache-2.0"), 11821);
        model("resnet50.proj", "proj", Some("mit"), 512);
        remote
    }

    #[test]
    fn test_generate_writes_the_loadable_repos() {
        let remote = hub();
        let kit = kit();
        let generated = generate(&remote, &kit, &source(&a1().render(&kit)), false).unwrap();
        let r50 = TimmRow {
            architecture: "resnet50".into(),
            tag: "a1_in1k".into(),
            revision: "r50a1".into(),
            license: "apache-2.0".into(),
            labels: Some("/imagenet/synsets-1k".into()),
        };
        let in12k = TimmRow {
            tag: "in12k".into(),
            revision: "in12k".into(),
            labels: Some("/imagenet/synsets-12k".into()),
            ..r50.clone()
        };
        // The existing row keeps its pin; the new ones are pinned from the
        // Hub, each binding its head's labels.
        assert_eq!(generated.rows, [a1(), r50.clone(), in12k.clone()]);
        assert_eq!(
            generated
                .skipped
                .iter()
                .map(|s| (s.repo.as_str(), s.reason.as_str()))
                .collect::<Vec<_>>(),
            [
                (
                    "timm/resnet18.tv_in1k",
                    "`torchvision/resnet18` holds these weights"
                ),
                (
                    "timm/resnet50.bamboo",
                    "0 classes: no ImageNet label set of the catalog's has that many"
                ),
                (
                    "timm/resnet50.proj",
                    "512 classes: no ImageNet label set of the catalog's has that many"
                ),
                ("timm/resnet50.x", "its card states no license"),
            ]
        );
        assert_eq!(
            generated.text,
            source(&format!(
                "{}{}{}",
                a1().render(&kit),
                r50.render(&kit),
                in12k.render(&kit)
            ))
        );
        // The written source reads back, its labels and its alias after the
        // rows.
        let written = OrderedNode::from_yaml(&generated.text).unwrap();
        let rows = existing_rows(&written).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows["resnet50.in12k"], in12k);
    }

    /// A kit that loads unlabelled heads writes them as rows with no labels.
    #[test]
    fn test_an_unlabelled_kit_writes_every_head() {
        let remote = hub();
        let kit = TimmKit {
            unlabelled: true,
            ..kit()
        };
        let generated = generate(&remote, &kit, &source(""), false).unwrap();
        let unlabelled: Vec<_> = generated
            .rows
            .iter()
            .filter(|r| r.labels.is_none())
            .map(TimmRow::name)
            .collect();
        assert_eq!(unlabelled, ["resnet50.bamboo", "resnet50.proj"]);
        let proj = generated.rows.iter().find(|r| r.tag == "proj").unwrap();
        let rendered = proj.render(&kit);
        assert!(!rendered.contains("labels"), "{rendered}");
        let written = OrderedNode::from_yaml(&generated.text).unwrap();
        assert_eq!(existing_rows(&written).unwrap()["resnet50.proj"], *proj);
    }

    /// Each label set is the subset bunsen infers from its class count, by
    /// the subset's name.
    #[test]
    fn test_label_sets_are_the_inferred_subsets() {
        use bunsen::kits::images::support::labels::imagenet_subset_for;
        for &(classes, labels) in LABEL_SETS {
            let subset = imagenet_subset_for(classes as usize).unwrap();
            assert_eq!(
                labels,
                format!(
                    "/imagenet/synsets-{}",
                    subset.strip_prefix("imagenet-").unwrap()
                ),
                "{classes}"
            );
        }
    }

    #[test]
    fn test_refresh_pins_again() {
        let remote = hub();
        let kit = kit();
        let generated = generate(&remote, &kit, &source(&a1().render(&kit)), true).unwrap();
        assert_eq!(generated.rows[0].revision, "r18a1-new");
    }

    #[test]
    fn test_a_row_the_hub_dropped_is_an_error() {
        let remote = hub();
        let kit = kit();
        let gone = TimmRow {
            tag: "gone_in1k".into(),
            ..a1()
        };
        let err = generate(&remote, &kit, &source(&gone.render(&kit)), false).unwrap_err();
        assert!(err.contains("timm/resnet18.gone_in1k"), "{err}");
    }

    #[test]
    fn test_splice_needs_the_markers() {
        assert!(splice("a: 1\n", "").unwrap_err().contains(BEGIN));
        assert_eq!(
            splice(&format!("x\n  {BEGIN}\nold\n  {END}\ny\n"), "new\n").unwrap(),
            format!("x\n  {BEGIN}\nnew\n  {END}\ny\n")
        );
    }
}
