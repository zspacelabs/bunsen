//! `bunsen-tool card`: what this build is, its cache, and the models it knows.

use bunsen::{
    data::pretrained::{
        Construct,
        PretrainedCache,
        PretrainedFactory,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        gpts::{
            nanochat::pretrained::default_nanochat_factory,
            qwen3::pretrained::default_qwen3_factory,
        },
        speech::{
            parakeet::pretrained::default_parakeet_factory,
            qwen3_asr::pretrained::default_qwen3_asr_factory,
            silero_vad::pretrained::default_silero_factory,
            sortformer::pretrained::default_sortformer_factory,
            voxtral::pretrained::default_voxtral_factory,
            whisper::pretrained::default_whisper_factory,
        },
    },
};
use bunsen_app::cache::WeightsCacheArgs;
use serde::{
    Serialize,
    Serializer,
    ser::{
        SerializeMap,
        SerializeSeq,
    },
};

/// `bunsen-tool card`'s flags.
#[derive(clap::Args, Debug)]
pub struct CardCmd {
    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The output's format: YAML, or JSON on one line.
    #[arg(long, value_enum, default_value_t = Format::Yml)]
    format: Format,

    /// One section alone, under its name: `build`, `config` or `kits`.
    #[arg(long, value_enum)]
    section: Option<Section>,
}

/// `--format`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// YAML.
    #[value(alias = "yaml")]
    Yml,

    /// JSON, unindented.
    Json,
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
}

impl Value {
    /// A map of `entries`, in their order.
    fn map<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Value {
        Value::Map(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    /// Whether it is an empty string, list or map.
    fn is_empty(&self) -> bool {
        match self {
            Value::Str(s) => s.is_empty(),
            Value::Bool(_) => false,
            Value::List(items) => items.is_empty(),
            Value::Map(entries) => entries.is_empty(),
        }
    }

    /// The value without its empty fields: every map entry whose value is an
    /// empty string, list or map, at any depth, once its own empty fields are
    /// gone. A list keeps its items; a boolean is never empty.
    pub fn elide_empty(self) -> Value {
        match self {
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
        }
    }
}

impl CardCmd {
    /// Builds the card and writes it to stdout.
    pub fn run(&self) -> BunsenResult<()> {
        let cache = self.cache.init()?;
        let card = card(&cache)?;
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
/// models this build knows by name, by kind and family). Every field is
/// present, an empty one included; the output elides those
/// ([`Value::elide_empty`]).
///
/// # Errors
/// As a family's default factory, if two of its providers share a name.
pub fn card(cache: &PretrainedCache) -> BunsenResult<Value> {
    Ok(Value::map([
        ("build", build()),
        ("config", Value::map([("cache", cache_config(cache))])),
        ("kits", kits()?),
    ]))
}

/// The build: the package, its version, and the features compiled in.
fn build() -> Value {
    let features = [
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
fn cache_config(cache: &PretrainedCache) -> Value {
    Value::map([
        ("dir", Value::str(cache.cache_dir().display().to_string())),
        ("offline", Value::Bool(cache.offline())),
        ("fetch", Value::Bool(true)),
        ("marks_use", Value::Bool(true)),
    ])
}

/// The models, by kind (`asr`, `chat`) and family: each family's providers,
/// as `provider: [{name}]`, then its named models, as
/// `model: [{name, aliases, description}]`.
fn kits() -> BunsenResult<Value> {
    Ok(Value::map([
        (
            "asr",
            Value::map([
                ("parakeet", family(default_parakeet_factory()?)),
                ("qwen3-asr", family(default_qwen3_asr_factory()?)),
                ("voxtral", family(default_voxtral_factory()?)),
                ("whisper", family(default_whisper_factory()?)),
                ("sortformer", family(default_sortformer_factory()?)),
                ("silero-vad", family(default_silero_factory()?)),
            ]),
        ),
        (
            "chat",
            Value::map([
                ("qwen3", family(default_qwen3_factory()?)),
                ("nanochat", family(default_nanochat_factory()?)),
            ]),
        ),
    ]))
}

/// One family: `provider:`, each provider its factory searches, in order
/// (`well-known`, `hf`, ...); then `model:`, a summary card per row its
/// factory lists: its name, its aliases (none is an empty list, which the
/// output elides) and its description.
fn family<H: Construct>(factory: PretrainedFactory<H>) -> Value {
    let providers = factory
        .providers()
        .iter()
        .map(|provider| Value::map([("name", Value::str(provider.name()))]))
        .collect();
    let models = factory
        .list()
        .into_iter()
        .map(|(_, row)| {
            Value::map([
                ("name", Value::str(row.name)),
                (
                    "aliases",
                    Value::List(row.aliases.into_iter().map(Value::Str).collect()),
                ),
                ("description", Value::str(row.description)),
            ])
        })
        .collect();
    Value::map([
        ("provider", Value::List(providers)),
        ("model", Value::List(models)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offline_cache() -> PretrainedCache {
        WeightsCacheArgs {
            cache_dir: Some(std::env::temp_dir().join("bunsen-tool-card-test")),
            offline: true,
        }
        .init()
        .unwrap()
    }

    /// The card has its three sections in order; every family lists its
    /// providers, the well-known table among them, then its models, each a
    /// name, its aliases and a description, in that order.
    #[test]
    fn test_card_shape() {
        let card = card(&offline_cache()).unwrap();
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
        ] {
            for family in families {
                let entry = kits.get(kind).and_then(|k| k.get(family)).unwrap();
                let Value::Map(groups) = entry else {
                    panic!("{kind}/{family}: a map");
                };
                let keys: Vec<&str> = groups.iter().map(|(k, _)| k.as_str()).collect();
                assert_eq!(keys, ["provider", "model"], "{kind}/{family}");
                let Some(Value::List(providers)) = entry.get("provider") else {
                    panic!("{kind}/{family}: no providers");
                };
                assert!(
                    providers
                        .iter()
                        .any(|p| p.get("name") == Some(&Value::str("well-known"))),
                    "{kind}/{family}: {providers:?}"
                );
                let Some(Value::List(models)) = entry.get("model") else {
                    panic!("{kind}/{family}: no models");
                };
                assert!(!models.is_empty(), "{kind}/{family}");
                for model in models {
                    let Value::Map(fields) = model else {
                        panic!("{kind}/{family}: a card is a map");
                    };
                    let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
                    assert_eq!(keys, ["name", "aliases", "description"], "{kind}/{family}");
                }
            }
        }
    }

    /// YAML keeps the keys' order; JSON is one line.
    /// Empty strings, lists and maps go, at any depth, and so does a map
    /// emptied by it; booleans and list items stay.
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
                            ("name", Value::str("qwen/qwen3-0.6b")),
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
            "{\"kits\":{\"chat\":{\"qwen3\":{\"model\":[{\"name\":\"qwen/qwen3-0.6b\",\"description\":\"Qwen3-0.6B\"}]}}}}\n"
        );
    }
}
