//! `bunsen-tool card`: what this build is, its cache, and the models it knows.

#[cfg(feature = "speech")]
use bunsen::kits::speech::{
    asr::{
        parakeet::{
            ParakeetPolicy,
            pretrained::default_parakeet_factory,
        },
        qwen3_asr::{
            Qwen3AsrPolicy,
            pretrained::default_qwen3_asr_factory,
        },
        voxtral::{
            VoxtralPolicy,
            pretrained::default_voxtral_factory,
        },
        whisper::{
            driver::WhisperPolicy,
            pretrained::default_whisper_factory,
        },
    },
    diarization::sortformer::{
        SortformerPolicy,
        pretrained::default_sortformer_factory,
    },
    vad::silero_vad::{
        SileroPolicy,
        pretrained::default_silero_factory,
    },
};
use bunsen::{
    data::{
        builds::{
            BuildFactory,
            ConstructInit,
        },
        catalog::presets::ResolvedPresetSet,
        resources::ResourceCache,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        decision::laya::pretrained::default_laya_factory,
        language::{
            gemma::{
                GemmaPolicy,
                pretrained::default_gemma_factory,
            },
            nanochat::pretrained::default_nanochat_factory,
            qwen3::{
                Qwen3Policy,
                pretrained::default_qwen3_factory,
            },
            qwen3_5::{
                Qwen3_5Policy,
                pretrained::default_qwen3_5_factory,
            },
        },
    },
};
use bunsen_app::cache::{
    WeightsCacheArgs,
    human_bytes,
};
use serde::{
    Serialize,
    Serializer,
    de::DeserializeOwned,
    ser::{
        SerializeMap,
        SerializeSeq,
    },
};

use crate::support::Format;

/// `bunsen-tool card`'s flags.
#[derive(clap::Args, Debug)]
pub struct CardCommand {
    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The output's format: YAML, or JSON on one line.
    #[arg(long, value_enum, default_value_t = Format::Yml)]
    format: Format,

    /// One section alone, under its name: `build`, `config` or `kits`.
    #[arg(long, value_enum)]
    section: Option<Section>,

    /// Each model's presets as data, and its default selection's policy.
    ///
    /// The presets with their patches included, and the policy the model's
    /// default selection resolves to (`policy`).
    #[arg(long)]
    policies: bool,
}

/// `--section`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    /// The build: its version and features.
    Build,

    /// The configuration: the weights cache.
    Config,

    /// The models, by kind and family.
    Kits,
}

impl Section {
    fn key(self) -> &'static str {
        match self {
            Section::Build => "build",
            Section::Config => "config",
            Section::Kits => "kits",
        }
    }
}

/// A card's document: a tree of strings, booleans, lists and maps whose keys
/// keep the order they were added in.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A string.
    Str(String),

    /// A boolean.
    Bool(bool),

    /// A list.
    List(Vec<Value>),

    /// A map, in the order its keys were added.
    Map(Vec<(String, Value)>),

    /// Data as it serializes: a preset set, a resolved policy.
    Json(serde_json::Value),
}

impl Value {
    /// A map of `entries`, in their order.
    pub fn map<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Value {
        Value::Map(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    pub fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    /// Whether it is an empty string, list or map.
    fn is_empty(&self) -> bool {
        match self {
            Value::Str(s) => s.is_empty(),
            Value::Bool(_) => false,
            Value::List(items) => items.is_empty(),
            Value::Map(entries) => entries.is_empty(),
            Value::Json(json) => match json {
                serde_json::Value::Null => true,
                serde_json::Value::Array(items) => items.is_empty(),
                serde_json::Value::Object(entries) => entries.is_empty(),
                _ => false,
            },
        }
    }

    /// The value without its empty fields: every map entry whose value is an
    /// empty string, list or map, at any depth, once its own empty fields are
    /// gone; in data ([`Json`](Value::Json)), every object entry that is
    /// `null` or an empty object or array, too. A list keeps its items; a
    /// boolean is never empty.
    pub fn elide_empty(self) -> Value {
        match self {
            Value::Json(json) => Value::Json(elide_empty_json(json)),
            Value::List(items) => Value::List(items.into_iter().map(Value::elide_empty).collect()),
            Value::Map(entries) => Value::Map(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, v.elide_empty()))
                    .filter(|(_, v)| !v.is_empty())
                    .collect(),
            ),
            other => other,
        }
    }

    /// The value under `key`, in a map.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&Value> {
        match self {
            Value::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match self {
            Value::Str(s) => serializer.serialize_str(s),
            Value::Bool(b) => serializer.serialize_bool(*b),
            Value::List(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Value::Map(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (k, v) in entries {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
            Value::Json(json) => json.serialize(serializer),
        }
    }
}

impl CardCommand {
    /// Builds the card and writes it to stdout.
    pub fn run(&self) -> BunsenResult<()> {
        let cache = self.cache.init()?;
        let card = card(&cache, self.policies)?;
        let card = match self.section {
            None => card,
            Some(section) => Value::map([(
                section.key(),
                card.get(section.key())
                    .cloned()
                    .expect("every section is in the card"),
            )]),
        };
        let text = render(&card.elide_empty(), self.format)?;
        use std::io::Write;
        std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(sys_at("write to", "stdout"))
    }
}

/// `json` without its empty fields: every object entry that is `null`, or an
/// empty object or array once its own empty fields are gone, at any depth.
/// An array keeps its items.
fn elide_empty_json(json: serde_json::Value) -> serde_json::Value {
    use serde_json::Value as J;
    match json {
        J::Array(items) => J::Array(items.into_iter().map(elide_empty_json).collect()),
        J::Object(entries) => J::Object(
            entries
                .into_iter()
                .map(|(k, v)| (k, elide_empty_json(v)))
                .filter(|(_, v)| match v {
                    J::Null => false,
                    J::Array(items) => !items.is_empty(),
                    J::Object(entries) => !entries.is_empty(),
                    _ => true,
                })
                .collect(),
        ),
        other => other,
    }
}

/// `value` as `format`: YAML, or JSON on one line, each ending in a newline.
pub fn render(
    value: &Value,
    format: Format,
) -> BunsenResult<String> {
    match format {
        Format::Yml => serde_saphyr::to_string(value)
            .map_err(|e| BunsenError::internal(format!("render the card as YAML: {e}"))),
        Format::Json => serde_json::to_string(value)
            .map(|json| json + "\n")
            .map_err(|e| BunsenError::internal(format!("render the card as JSON: {e}"))),
    }
}

/// The whole card: `build`, `config` (the weights cache) and `kits` (the
/// models this build knows by name, by kind and family; under `policies`,
/// each one's presets as data and its resolved policy). Every field is
/// present, an empty one included; the output elides those
/// ([`Value::elide_empty`]).
///
/// # Errors
/// As a family's default factory, if two of its providers share a name; as
/// a row's policy, under `policies`.
pub fn card(
    cache: &ResourceCache,
    policies: bool,
) -> BunsenResult<Value> {
    Ok(Value::map([
        ("build", build()),
        ("config", Value::map([("cache", cache_config(cache))])),
        ("kits", kits(policies)?),
    ]))
}

/// The build: the package, its version, and the features compiled in.
fn build() -> Value {
    let features = [
        ("speech", cfg!(feature = "speech")),
        ("live", cfg!(feature = "live")),
        ("webhook", cfg!(feature = "webhook")),
        ("mqtt", cfg!(feature = "mqtt")),
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
        ("vulkan", cfg!(feature = "vulkan")),
        ("wgpu", cfg!(feature = "wgpu")),
    ];
    Value::map([
        ("package", Value::str(env!("CARGO_PKG_NAME"))),
        ("version", Value::str(env!("CARGO_PKG_VERSION"))),
        (
            "features",
            Value::List(
                features
                    .iter()
                    .filter(|(_, on)| *on)
                    .map(|(name, _)| Value::str(*name))
                    .collect(),
            ),
        ),
    ])
}

/// The weights cache: where it is, whether this run keeps it off the
/// network (`--offline`), whether the build can fetch at all, and that a
/// load marks the files it finds used (`cache list`'s last use).
fn cache_config(cache: &ResourceCache) -> Value {
    Value::map([
        ("dir", Value::str(cache.cache_dir().display().to_string())),
        ("offline", Value::Bool(cache.offline())),
        ("fetch", Value::Bool(true)),
        ("marks_use", Value::Bool(true)),
    ])
}

/// The models, by kind (`asr`, `chat`, `decide`) and family: each family a list
/// of entries, its providers (`- provider: {name}`) then its named models
/// (`- model: {name, aliases, description, size, dtype}`).
fn kits(policies: bool) -> BunsenResult<Value> {
    // A family's rows, their policies resolved as `C` under `policies`.
    macro_rules! family {
        ($factory:expr, $policy:ty) => {
            family($factory, policies.then_some(policy::<_, $policy>))?
        };
    }
    let mut kinds = Vec::new();
    #[cfg(feature = "speech")]
    kinds.push((
        "asr",
        Value::map([
            (
                "parakeet",
                family!(default_parakeet_factory()?, ParakeetPolicy),
            ),
            (
                "qwen3-asr",
                family!(default_qwen3_asr_factory()?, Qwen3AsrPolicy),
            ),
            (
                "voxtral",
                family!(default_voxtral_factory()?, VoxtralPolicy),
            ),
            (
                "whisper",
                family!(default_whisper_factory()?, WhisperPolicy),
            ),
            (
                "sortformer",
                family!(default_sortformer_factory()?, SortformerPolicy),
            ),
            (
                "silero-vad",
                family!(default_silero_factory()?, SileroPolicy),
            ),
        ]),
    ));
    kinds.extend([
        (
            "chat",
            Value::map([
                ("qwen3", family!(default_qwen3_factory()?, Qwen3Policy)),
                (
                    "qwen3.5",
                    family!(default_qwen3_5_factory()?, Qwen3_5Policy),
                ),
                ("gemma", family!(default_gemma_factory()?, GemmaPolicy)),
                // nanochat has no policy to resolve.
                (
                    "nanochat",
                    family(default_nanochat_factory()?, None::<fn(&_, &str) -> _>)?,
                ),
            ]),
        ),
        (
            "decide",
            // Laya has no policy to resolve.
            Value::map([(
                "laya",
                family(default_laya_factory()?, None::<fn(&_, &str) -> _>)?,
            )]),
        ),
    ]);
    Ok(Value::map(kinds))
}

/// The policy `C` the row `name` resolves to under its default selection,
/// as data: what its kit's hook would build it under, with nothing fetched.
///
/// # Errors
/// As [`BuildFactory::resolve_presets`];
/// [`Internal`](bunsen::errors::BunsenErrorKind::Internal) if the policy
/// does not serialize.
fn policy<H: ConstructInit, C: Default + Serialize + DeserializeOwned>(
    factory: &BuildFactory<H>,
    name: &str,
) -> BunsenResult<serde_json::Value> {
    let resolved = factory.resolve_presets(name, &C::default())?;
    serde_json::to_value(resolved.config)
        .map_err(|e| BunsenError::internal(format!("the policy of {name} as data: {e}")))
}

/// A row's presets, in brief: each axis and its members, the presets on no
/// axis (`other`), and the `default` selection; under `full`, the preset
/// set as data, in its fields' order: `base`, `modes` (the presets, each
/// its `name`, `description` and `patch`), `axes` (each its `name`,
/// `description` and `members`) and `default`. Empty (and
/// elided) for a row with none.
fn presets(
    set: &ResolvedPresetSet,
    full: bool,
) -> BunsenResult<Value> {
    if set.presets.is_empty() {
        return Ok(Value::map::<&str>([]));
    }
    let strs =
        |names: &[String]| Value::List(names.iter().map(|n| Value::str(n.as_str())).collect());
    if full {
        // In the structs' field order, each preset's and axis's name first,
        // rather than a JSON map's sorted keys; the patches as data.
        return Ok(Value::map([
            ("base", Value::Json(set.common.clone())),
            (
                "modes",
                Value::List(
                    set.presets
                        .iter()
                        .map(|p| {
                            Value::map([
                                ("name", Value::str(p.name.as_str())),
                                ("description", Value::str(p.description.as_str())),
                                ("patch", Value::Json(p.patch.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "axes",
                Value::List(
                    set.axes
                        .iter()
                        .map(|a| {
                            Value::map([
                                ("name", Value::str(a.axis.name.to_string())),
                                ("description", Value::str(a.axis.description.as_str())),
                                ("members", strs(&a.offers)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("default", strs(&set.default_names())),
        ]));
    }
    let names = |names: &mut dyn Iterator<Item = &String>| {
        Value::List(names.map(|n| Value::str(n.as_str())).collect())
    };
    let mut entries: Vec<(String, Value)> = set
        .axes
        .iter()
        .map(|a| (a.axis.name.to_string(), names(&mut a.offers.iter())))
        .collect();
    let other: Vec<&String> = set
        .presets
        .iter()
        .map(|p| &p.name)
        .filter(|n| set.axis_of(n).is_none())
        .collect();
    entries.push(("other".to_string(), names(&mut other.into_iter())));
    let defaults = set.default_names();
    entries.push(("default".to_string(), names(&mut defaults.iter())));
    Ok(Value::Map(entries))
}

/// One family: a list of entries, each a one-key map. First a `provider:`
/// entry for each provider its factory searches, in order (`catalog`,
/// `hf`, ...), its name; then a `model:` entry for each row its factory
/// lists, a summary card: its name, its aliases (none is an empty list,
/// which the output elides), its description, the size of its files, the
/// dtypes its files declare (none is an empty string, also elided), and its
/// presets, in brief ([`presets`]). With a `policy`, the presets are the
/// set as data, and `policy` is what the row's default selection resolves
/// to.
///
/// # Errors
/// As `policy`, for a row.
fn family<H: ConstructInit>(
    factory: BuildFactory<H>,
    policy: Option<impl Fn(&BuildFactory<H>, &str) -> BunsenResult<serde_json::Value>>,
) -> BunsenResult<Value> {
    let providers = factory.providers().iter().map(|provider| {
        Value::map([(
            "provider",
            Value::map([("name", Value::str(provider.name()))]),
        )])
    });
    let mut models = Vec::new();
    for (provider, row) in factory.list() {
        let presets = presets(&row.presets, policy.is_some())?;
        let policy = match &policy {
            Some(policy) => Value::Json(policy(&factory, &format!("{provider}:{}", row.name))?),
            None => Value::map::<&str>([]),
        };
        models.push(Value::map([(
            "model",
            Value::map([
                ("name", Value::str(row.name)),
                (
                    "aliases",
                    Value::List(row.aliases.into_iter().map(Value::Str).collect()),
                ),
                ("description", Value::str(row.description)),
                (
                    "size",
                    Value::str(row.resources.size().map(human_bytes).unwrap_or_default()),
                ),
                ("dtype", Value::str(row.resources.dtypes().join(", "))),
                ("presets", presets),
                ("policy", policy),
            ]),
        )]));
    }
    Ok(Value::List(providers.chain(models).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "speech")]
    fn offline_cache() -> ResourceCache {
        WeightsCacheArgs {
            cache_dir: Some(std::env::temp_dir().join("bunsen-tool-card-test")),
            offline: true,
        }
        .init()
        .unwrap()
    }

    /// The card has its three sections in order; every family is a list of
    /// one-key entries, its providers first (the catalog provider first of
    /// them), then its models, each a name, its aliases, a description, a
    /// size and its dtypes, in that order; every model's size is known.
    #[cfg(feature = "speech")]
    #[test]
    fn test_card_shape() {
        let card = card(&offline_cache(), true).unwrap();
        let Value::Map(sections) = &card else {
            panic!("a map");
        };
        let keys: Vec<&str> = sections.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["build", "config", "kits"]);
        assert_eq!(
            card.get("config")
                .unwrap()
                .get("cache")
                .unwrap()
                .get("offline"),
            Some(&Value::Bool(true))
        );
        let kits = card.get("kits").unwrap();
        // A multilingual Whisper row: its task axis, and the policy its
        // default selection resolves to.
        let Some(Value::List(whisper)) = kits.get("asr").and_then(|k| k.get("whisper")) else {
            panic!("whisper");
        };
        let small = whisper
            .iter()
            .filter_map(|e| e.get("model"))
            .find(|m| m.get("name") == Some(&Value::str("openai/whisper/small")))
            .unwrap();
        let presets = small.get("presets").unwrap();
        assert_eq!(
            presets.get("default"),
            Some(&Value::List(vec![Value::str("transcribe")]))
        );
        // A preset's name comes first, then its description, then its patch.
        let Some(Value::List(each)) = presets.get("modes") else {
            panic!("the presets, a list under `modes`");
        };
        let Value::Map(first) = &each[0] else {
            panic!("a preset, a map");
        };
        let keys: Vec<&str> = first.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["name", "description", "patch"]);
        assert_eq!(first[0].1, Value::str("transcribe"));
        let Some(Value::Json(policy)) = small.get("policy") else {
            panic!("a policy as data");
        };
        assert_eq!(policy["task"], "Transcribe");
        assert_eq!(policy["beam_size"], 5);
        for (kind, families) in [
            (
                "asr",
                &[
                    "parakeet",
                    "qwen3-asr",
                    "voxtral",
                    "whisper",
                    "sortformer",
                    "silero-vad",
                ][..],
            ),
            ("chat", &["qwen3", "nanochat"]),
            ("decide", &["laya"]),
        ] {
            for family in families {
                let Some(Value::List(entries)) = kits.get(kind).and_then(|k| k.get(family)) else {
                    panic!("{kind}/{family}: a list of entries");
                };
                // Each entry is one key, `provider` or `model`, the providers
                // first.
                let tags: Vec<&str> = entries
                    .iter()
                    .map(|e| match e {
                        Value::Map(m) if m.len() == 1 => m[0].0.as_str(),
                        other => panic!("{kind}/{family}: {other:?}"),
                    })
                    .collect();
                let providers = tags.iter().take_while(|t| **t == "provider").count();
                assert!(providers >= 1, "{kind}/{family}: {tags:?}");
                assert!(
                    tags[providers..].iter().all(|t| *t == "model") && tags.len() > providers,
                    "{kind}/{family}: {tags:?}"
                );
                assert_eq!(
                    entries[0].get("provider").and_then(|p| p.get("name")),
                    Some(&Value::str("catalog")),
                    "{kind}/{family}"
                );
                for entry in &entries[providers..] {
                    let Some(Value::Map(fields)) = entry.get("model") else {
                        panic!("{kind}/{family}: a card is a map");
                    };
                    let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
                    assert_eq!(
                        keys,
                        [
                            "name",
                            "aliases",
                            "description",
                            "size",
                            "dtype",
                            "presets",
                            "policy"
                        ],
                        "{kind}/{family}"
                    );
                    assert!(
                        fields.iter().any(|(k, v)| k == "size" && !v.is_empty()),
                        "{kind}/{family}: every catalog file is sized"
                    );
                    // A kit with a policy resolves every row's default.
                    if !["nanochat", "laya"].contains(family) {
                        assert!(
                            fields.iter().any(|(k, v)| k == "policy" && !v.is_empty()),
                            "{kind}/{family}: a policy"
                        );
                    }
                }
            }
        }
    }

    /// Without `--policies`, a row's presets are in brief, its axes and
    /// default, and there is no policy.
    #[cfg(feature = "speech")]
    #[test]
    fn test_card_presets_in_brief() {
        let card = card(&offline_cache(), false).unwrap().elide_empty();
        let Some(Value::List(whisper)) = card
            .get("kits")
            .and_then(|k| k.get("asr"))
            .and_then(|k| k.get("whisper"))
        else {
            panic!("whisper");
        };
        let small = whisper
            .iter()
            .filter_map(|e| e.get("model"))
            .find(|m| m.get("name") == Some(&Value::str("openai/whisper/small")))
            .unwrap();
        assert_eq!(
            small.get("presets"),
            Some(&Value::map([
                (
                    "task",
                    Value::List(vec![Value::str("transcribe"), Value::str("translate")])
                ),
                ("default", Value::List(vec![Value::str("transcribe")])),
            ]))
        );
        assert_eq!(small.get("policy"), None);
    }

    /// YAML keeps the keys' order; JSON is one line.
    /// Empty strings, lists and maps go, at any depth, and so does a map
    /// emptied by it, nulls in data too; booleans and list items stay.
    #[test]
    fn test_elide_empty() {
        let card = Value::map([
            (
                "model",
                Value::List(vec![
                    Value::map([("name", Value::str("a")), ("aliases", Value::List(vec![]))]),
                    Value::map([
                        ("name", Value::str("b")),
                        ("aliases", Value::List(vec![Value::str("bee")])),
                    ]),
                ]),
            ),
            ("features", Value::List(vec![])),
            ("offline", Value::Bool(false)),
            ("note", Value::str("")),
            ("nested", Value::map([("empty", Value::map::<&str>([]))])),
            (
                "policy",
                Value::Json(serde_json::json!({
                    "generation": {
                        "max_new_tokens": null,
                        "sampling": null,
                        "stop_ids": [151645, 151643],
                        "template_kwargs": {},
                    },
                    "options": {"a": null, "b": []},
                    "flags": [null, {}],
                    "off": false,
                    "zero": 0,
                })),
            ),
            ("unset", Value::Json(serde_json::json!({"a": null}))),
        ]);
        assert_eq!(
            card.elide_empty(),
            Value::map([
                (
                    "model",
                    Value::List(vec![
                        Value::map([("name", Value::str("a"))]),
                        Value::map([
                            ("name", Value::str("b")),
                            ("aliases", Value::List(vec![Value::str("bee")])),
                        ]),
                    ]),
                ),
                ("offline", Value::Bool(false)),
                // Data loses its nulls and empties, at any depth; an array
                // keeps its items, and `false` and `0` stay.
                (
                    "policy",
                    Value::Json(serde_json::json!({
                        "generation": {"stop_ids": [151645, 151643]},
                        "flags": [null, {}],
                        "off": false,
                        "zero": 0,
                    })),
                ),
            ])
        );
    }

    #[test]
    fn test_render() {
        let card = Value::map([(
            "kits",
            Value::map([(
                "chat",
                Value::map([(
                    "qwen3",
                    Value::map([(
                        "model",
                        Value::List(vec![Value::map([
                            ("name", Value::str("qwen/qwen3/qwen3-0.6b")),
                            ("description", Value::str("Qwen3-0.6B")),
                        ])]),
                    )]),
                )]),
            )]),
        )]);
        let yml = render(&card, Format::Yml).unwrap();
        assert!(
            yml.find("name:").unwrap() < yml.find("description:").unwrap(),
            "{yml}"
        );
        let json = render(&card, Format::Json).unwrap();
        assert_eq!(
            json,
            "{\"kits\":{\"chat\":{\"qwen3\":{\"model\":[{\"name\":\"qwen/qwen3/qwen3-0.6b\",\"description\":\"Qwen3-0.6B\"}]}}}}\n"
        );
    }
}
