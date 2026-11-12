//! Probes: readers of a pretrained's files that find its presets' values.

use alloc::fmt;

use serde_json::{
    Map,
    Value,
    json,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// One probe of a pretrained's files: a named reader of one of its files,
/// and where what it finds goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogProbe {
    /// The probe, by name: `generation_config`, say.
    pub probe: String,

    /// The file it reads, by its resource key.
    pub file: String,

    /// Where its patch goes: `base`, the set's common patch, or presets by
    /// name.
    pub into: Vec<String>,

    /// The token it reads, for a probe that reads one by name (the
    /// `tokenizer_config` probe's added token: `<end_of_turn>`, say); `None`,
    /// the probe's own.
    pub token: Option<String>,
}

/// A reader of one of a pretrained's files that finds values for its
/// presets: a merge patch over the family's config.
///
/// Probes are found by [`name`](Self::name). The built-in ones
/// ([`builtin_probes`]) read the files a Hugging Face repo carries.
pub trait Probe: Send + Sync + fmt::Debug {
    /// Its name, as a catalog row's `probe` entry names it.
    fn name(&self) -> &'static str;

    /// The patch `file`'s bytes say, as `entry` asks: the file's resource
    /// key is `entry.file`, and a probe that reads a token by name reads
    /// `entry.token`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when the file does not read as the probe expects;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a `token` the
    /// probe takes none of.
    fn probe(
        &self,
        file: &[u8],
        entry: &CatalogProbe,
    ) -> BunsenResult<Value>;
}

/// Refuses a `token` for a probe that reads none by name.
fn no_token(
    probe: &str,
    token: Option<&str>,
) -> BunsenResult<()> {
    match token {
        None => Ok(()),
        Some(token) => Err(BunsenError::illegal(format!(
            "the {probe} probe reads no token by name, but was given {token:?}"
        ))),
    }
}

/// `generation_config.json`: the stop tokens (`eos_token_id`, one or
/// several) and the sampling (`do_sample`, `temperature`, `top_k`, `top_p`,
/// `min_p`; the argmax without `do_sample`), and `max_new_tokens` when it
/// names one. Its patch is under `generation`
/// ([`GenerationPolicy`](super::GenerationPolicy)).
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationConfigProbe;

impl Probe for GenerationConfigProbe {
    fn name(&self) -> &'static str {
        "generation_config"
    }

    fn probe(
        &self,
        file: &[u8],
        entry: &CatalogProbe,
    ) -> BunsenResult<Value> {
        no_token(self.name(), entry.token.as_deref())?;
        let c = read_json(file, "a generation_config.json")?;
        let mut generation = Map::new();
        match c.get("eos_token_id") {
            Some(Value::Array(ids)) => {
                generation.insert("stop_ids".into(), Value::Array(ids.clone()));
            }
            Some(id @ Value::Number(_)) => {
                generation.insert("stop_ids".into(), json!([id]));
            }
            _ => {}
        }
        let sampling = if c.get("do_sample").and_then(Value::as_bool) == Some(true) {
            json!({
                "temperature": c.get("temperature").cloned().unwrap_or(json!(1.0)),
                "top_k": c.get("top_k").cloned().unwrap_or(Value::Null),
                "top_p": c.get("top_p").cloned().unwrap_or(Value::Null),
                "min_p": c.get("min_p").cloned().unwrap_or(Value::Null),
            })
        } else {
            Value::Null
        };
        generation.insert("sampling".into(), sampling);
        if let Some(n) = c.get("max_new_tokens").filter(|v| v.is_u64()) {
            generation.insert("max_new_tokens".into(), n.clone());
        }
        Ok(json!({ "generation": generation }))
    }
}

/// `tokenizer_config.json`: a stop token, by the id its
/// `added_tokens_decoder` gives it: the one its `eos_token` names, or, given
/// a token by name, that one (the turn's end a chat template closes each
/// turn with: Gemma's `<end_of_turn>`). Its patch is under `generation`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenizerConfigProbe;

impl Probe for TokenizerConfigProbe {
    fn name(&self) -> &'static str {
        "tokenizer_config"
    }

    fn probe(
        &self,
        file: &[u8],
        entry: &CatalogProbe,
    ) -> BunsenResult<Value> {
        let c = read_json(file, "a tokenizer_config.json")?;
        let eos = match (entry.token.as_deref(), c.get("eos_token")) {
            (Some(token), _) => token,
            (None, Some(Value::String(s))) => s.as_str(),
            (None, Some(Value::Object(o))) => {
                o.get("content").and_then(Value::as_str).unwrap_or("")
            }
            _ => {
                return Err(BunsenError::invalid_resource(
                    "a tokenizer_config.json without an `eos_token`",
                ));
            }
        };
        let id = c
            .get("added_tokens_decoder")
            .and_then(Value::as_object)
            .and_then(|tokens| {
                tokens.iter().find_map(|(id, t)| {
                    (t.get("content").and_then(Value::as_str) == Some(eos))
                        .then(|| id.parse::<i64>().ok())
                        .flatten()
                })
            })
            .ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "a tokenizer_config.json whose stop token {eos:?} is no added token"
                ))
            })?;
        Ok(json!({ "generation": { "stop_ids": [id] } }))
    }
}

/// A chat template's file: a `tokenizer_config.json` that holds a
/// `chat_template`, or a template file of its own (`chat_template.jinja`),
/// as the row's [`ChatTemplateSource::File`](super::ChatTemplateSource) by
/// its resource key; a `tokenizer_config.json` without one says nothing.
/// Its patch is under `generation`.
///
/// Probed after a `tokenizer_config.json`, a `chat_template.jinja` is the
/// template, as `transformers` prefers it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChatTemplateProbe;

impl Probe for ChatTemplateProbe {
    fn name(&self) -> &'static str {
        "chat_template"
    }

    fn probe(
        &self,
        file: &[u8],
        entry: &CatalogProbe,
    ) -> BunsenResult<Value> {
        no_token(self.name(), entry.token.as_deref())?;
        let holds = match serde_json::from_slice::<Value>(file) {
            Ok(Value::Object(config)) => config.get("chat_template").is_some_and(Value::is_string),
            _ => !core::str::from_utf8(file)
                .map_err(|e| {
                    BunsenError::invalid_resource("a chat template that is not UTF-8").with_cause(e)
                })?
                .trim()
                .is_empty(),
        };
        Ok(if holds {
            json!({ "generation": { "chat_template": { "file": entry.file } } })
        } else {
            json!({})
        })
    }
}

fn read_json(
    file: &[u8],
    what: &'static str,
) -> BunsenResult<Value> {
    let value: Value = serde_json::from_slice(file)
        .map_err(|e| BunsenError::invalid_resource(what).with_cause(e))?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(BunsenError::invalid_resource(format!(
            "{what} that is not an object"
        )))
    }
}

/// The probes bunsen has: [`GenerationConfigProbe`] and
/// [`TokenizerConfigProbe`].
pub fn builtin_probes() -> Vec<Box<dyn Probe>> {
    vec![
        Box::new(GenerationConfigProbe),
        Box::new(TokenizerConfigProbe),
        Box::new(ChatTemplateProbe),
    ]
}

/// `into`, with what one more probe found: what probes find accumulates.
/// Objects join field by field, an array gains the values it lacks, in
/// order (two probes' stop ids join: `generation_config.json`'s end of the
/// text and `tokenizer_config.json`'s end of a turn), and anything else is
/// the later probe's. What is laid over probes (a row's or family's own
/// presets, a lookup's patch) is a merge patch, which replaces.
pub fn accumulate_probed(
    into: &mut Value,
    found: &Value,
) {
    match (into, found) {
        (Value::Object(into), Value::Object(found)) => {
            for (key, value) in found {
                match into.get_mut(key) {
                    Some(slot) => accumulate_probed(slot, value),
                    None => {
                        into.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (Value::Array(into), Value::Array(found)) => {
            for value in found {
                if !into.contains(value) {
                    into.push(value.clone());
                }
            }
        }
        (into, found) => *into = found.clone(),
    }
}

/// The presets of a checkpoint no catalog build describes (an `hf:` repo's),
/// as the catalog builder probes a build's: `family`'s, with what each of
/// `entries` whose file `loaded` has finds laid over them, the rest skipped.
/// What probes find accumulates ([`accumulate_probed`]); an entry's `base`
/// is the common patch, and a preset it names must be one of `family`'s.
///
/// # Errors
/// - [`Sys`](crate::errors::BunsenErrorKind::Sys) when a file cannot be read.
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup): an unknown probe, or a
///   preset `family` lacks.
/// - As a probe's [`probe`](Probe::probe), under a frame naming the entry.
#[cfg(feature = "cache")]
pub fn probe_loaded(
    loaded: &crate::data::resources::LoadedResources,
    entries: &[CatalogProbe],
    family: &crate::data::catalog::presets::ResolvedPresetSet,
) -> BunsenResult<crate::data::catalog::presets::ResolvedPresetSet> {
    use alloc::collections::BTreeMap;

    use crate::{
        errors::LookupError,
        support::merge_patch,
    };

    let probes = builtin_probes();
    let mut common = Value::Object(Map::new());
    let mut found: BTreeMap<String, Value> = BTreeMap::new();
    for entry in entries {
        let Some(part) = loaded.get(&entry.file) else {
            continue;
        };
        let bytes = std::fs::read(&part.path).map_err(crate::errors::sys_at("read", &part.path))?;
        let probe = probes
            .iter()
            .find(|p| p.name() == entry.probe)
            .ok_or_else(|| {
                BunsenError::lookup(
                    LookupError::missing("probe", entry.probe.clone())
                        .with_candidates(probes.iter().map(|p| p.name())),
                )
            })?;
        let patch = probe
            .probe(&bytes, entry)
            .map_err(|e| e.context(format!("the {} probe of {:?}", entry.probe, entry.file)))?;
        for into in &entry.into {
            if into == "base" {
                accumulate_probed(&mut common, &patch);
                continue;
            }
            if family.get(into).is_none() {
                return Err(BunsenError::lookup(
                    LookupError::missing("preset", into.clone()).with_candidates(family.names()),
                )
                .context(format!("the {} probe's `into`", entry.probe)));
            }
            accumulate_probed(
                found
                    .entry(into.clone())
                    .or_insert_with(|| Value::Object(Map::new())),
                &patch,
            );
        }
    }
    let mut out = family.clone();
    merge_patch::apply(&mut out.common, &common);
    for preset in &mut out.presets {
        if let Some(probed) = found.remove(&preset.name) {
            merge_patch::apply(&mut preset.patch, &probed);
        }
    }
    Ok(out)
}

impl CatalogProbe {
    /// The probe `probe` of the file under `file`, into the base.
    pub fn into_base(
        probe: impl Into<String>,
        file: impl Into<String>,
    ) -> Self {
        Self {
            probe: probe.into(),
            file: file.into(),
            into: vec!["base".to_string()],
            token: None,
        }
    }

    /// This probe, reading `token` by name.
    pub fn with_token(
        self,
        token: impl Into<String>,
    ) -> Self {
        Self {
            token: Some(token.into()),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::support::model_config::{
            ChatTemplateSource,
            GenerationPolicy,
            SamplingPolicy,
        },
        support::merge_patch,
    };

    const QWEN3_GENERATION: &str = r#"{
        "bos_token_id": 151643, "do_sample": true,
        "eos_token_id": [151645, 151643], "pad_token_id": 151643,
        "temperature": 0.6, "top_k": 20, "top_p": 0.95,
        "transformers_version": "4.51.0"
    }"#;

    /// A probe's entry, under the `file` key, with `token`.
    fn args(token: Option<&str>) -> CatalogProbe {
        CatalogProbe {
            probe: String::new(),
            file: "file".to_string(),
            into: vec!["base".to_string()],
            token: token.map(str::to_string),
        }
    }

    fn probed_policy(patch: &Value) -> GenerationPolicy {
        serde_json::from_value(patch["generation"].clone()).unwrap()
    }

    #[test]
    fn test_generation_config() {
        let patch = GenerationConfigProbe
            .probe(QWEN3_GENERATION.as_bytes(), &args(None))
            .unwrap();
        assert_eq!(
            probed_policy(&patch),
            GenerationPolicy {
                stop_ids: vec![151645, 151643],
                sampling: Some(SamplingPolicy {
                    temperature: 0.6,
                    top_k: Some(20),
                    top_p: Some(0.95),
                    min_p: None,
                }),
                ..Default::default()
            }
        );

        let greedy = GenerationConfigProbe
            .probe(br#"{"eos_token_id": 2, "max_new_tokens": 64}"#, &args(None))
            .unwrap();
        let p = probed_policy(&greedy);
        assert_eq!(
            (p.stop_ids, p.sampling, p.max_new_tokens),
            (vec![2], None, Some(64))
        );

        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .assert_err(&GenerationConfigProbe.probe(b"[1]", &args(None)));
    }

    #[test]
    fn test_tokenizer_config() {
        let config = br#"{
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {
                "151643": {"content": "<|endoftext|>"},
                "151645": {"content": "<|im_end|>"}
            }
        }"#;
        let patch = TokenizerConfigProbe.probe(config, &args(None)).unwrap();
        assert_eq!(patch, json!({"generation": {"stop_ids": [151645]}}));

        let object = br#"{"eos_token": {"content": "</s>"}, "added_tokens_decoder": {"2": {"content": "</s>"}}}"#;
        assert_eq!(
            TokenizerConfigProbe.probe(object, &args(None)).unwrap(),
            json!({"generation": {"stop_ids": [2]}})
        );
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("no added token")
            .assert_err(&TokenizerConfigProbe.probe(br#"{"eos_token": "?"}"#, &args(None)));
    }

    #[cfg(feature = "cache")]
    fn entry(
        probe: &str,
        file: &str,
        into: &[&str],
    ) -> CatalogProbe {
        CatalogProbe {
            probe: probe.to_string(),
            file: file.to_string(),
            into: into.iter().map(|s| s.to_string()).collect(),
            token: None,
        }
    }

    /// What probes find goes into the common patch or the presets they
    /// name, over the family's own; a file the checkpoint lacks is skipped.
    #[cfg(feature = "cache")]
    #[test]
    fn test_probe_loaded() {
        use std::path::PathBuf;

        use crate::data::{
            catalog::{
                document::{
                    MergeMode,
                    PresetEntry,
                },
                presets::ResolvedPresetSet,
            },
            resources::{
                LoadedResources,
                Provenance,
                ResolvedResource,
                ResourceBundle,
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let path: PathBuf = dir.path().join("generation_config.json");
        std::fs::write(&path, QWEN3_GENERATION).unwrap();
        let loaded = LoadedResources {
            map: ResourceBundle::new("qwen3"),
            parts: [(
                "generation_config".to_string(),
                ResolvedResource {
                    path,
                    provenance: Provenance::LocalDir,
                },
            )]
            .into_iter()
            .collect(),
        };
        let preset = |name: &str, description: &str, patch: Value| PresetEntry {
            name: name.to_string(),
            merge: MergeMode::Update,
            description: description.to_string(),
            patch,
        };
        let family = ResolvedPresetSet {
            common: json!({"generation": {"max_new_tokens": 512}}),
            presets: vec![
                preset(
                    "thinking",
                    "think first",
                    json!({"generation": {"template_kwargs": {"enable_thinking": true}}}),
                ),
                preset(
                    "no-thinking",
                    "answer at once",
                    json!({"generation": {"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": null}}}),
                ),
            ],
            ..ResolvedPresetSet::default()
        };
        let probed = probe_loaded(
            &loaded,
            &[entry(
                "generation_config",
                "generation_config",
                &["base", "thinking"],
            )],
            &family,
        )
        .unwrap();

        let common = probed_policy(&probed.common);
        assert_eq!(common.stop_ids, [151645, 151643]);
        assert_eq!(
            common.max_new_tokens,
            Some(512),
            "the family's common patch kept"
        );
        let thinking = &probed.get("thinking").unwrap().patch;
        assert_eq!(thinking["generation"]["sampling"]["temperature"], 0.6);
        assert_eq!(
            thinking["generation"]["template_kwargs"]["enable_thinking"],
            true
        );
        assert_eq!(
            probed.get("no-thinking").unwrap().patch,
            family.get("no-thinking").unwrap().patch
        );

        assert_eq!(
            probe_loaded(
                &loaded,
                &[entry("tokenizer_config", "tokenizer_config", &["base"])],
                &family,
            )
            .unwrap(),
            family,
            "a file the checkpoint lacks is skipped"
        );
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no probe \"sampling_card\"")
            .assert_err(&probe_loaded(
                &loaded,
                &[entry("sampling_card", "generation_config", &["base"])],
                &family,
            ));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no preset \"fast\"")
            .assert_err(&probe_loaded(
                &loaded,
                &[entry("generation_config", "generation_config", &["fast"])],
                &family,
            ));
    }

    /// Given a token by name, the `tokenizer_config` probe gives that added
    /// token's id; one it lacks is refused; a probe that reads no token
    /// refuses one.
    #[test]
    fn test_tokenizer_config_named_token() {
        let config = br#"{"eos_token": "<eos>", "added_tokens_decoder": {
            "1": {"content": "<eos>"}, "107": {"content": "<end_of_turn>"}}}"#;
        let patch = TokenizerConfigProbe
            .probe(config, &args(Some("<end_of_turn>")))
            .unwrap();
        assert_eq!(probed_policy(&patch).stop_ids, [107]);
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("<start_of_turn>")
            .assert_err(&TokenizerConfigProbe.probe(config, &args(Some("<start_of_turn>"))));
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("reads no token")
            .assert_err(&GenerationConfigProbe.probe(b"{}", &args(Some("<eos>"))));
    }

    /// Probes accumulate: two probes' stop ids join, in order, without a
    /// repeat. A hand-written patch laid over them replaces the array.
    #[test]
    fn test_probes_accumulate_and_hand_written_replaces() {
        let tokenizer_config = br#"{"eos_token": "<eos>", "added_tokens_decoder": {
            "1": {"content": "<eos>"}, "107": {"content": "<end_of_turn>"}}}"#;
        let found = [
            GenerationConfigProbe
                .probe(br#"{"eos_token_id": [1]}"#, &args(None))
                .unwrap(),
            TokenizerConfigProbe
                .probe(tokenizer_config, &args(Some("<end_of_turn>")))
                .unwrap(),
            TokenizerConfigProbe
                .probe(tokenizer_config, &args(None))
                .unwrap(),
        ];
        let mut probed = Value::Object(Map::new());
        for patch in &found {
            accumulate_probed(&mut probed, patch);
        }
        assert_eq!(probed_policy(&probed).stop_ids, [1, 107]);

        merge_patch::apply(&mut probed, &json!({"generation": {"stop_ids": [7]}}));
        assert_eq!(probed_policy(&probed).stop_ids, [7]);
    }

    /// A `tokenizer_config.json` that holds a template, or a template file,
    /// is the row's template by its key; one without says nothing; probed
    /// in order, the later file is the template.
    #[test]
    fn test_chat_template() {
        let keyed = |file: &str| CatalogProbe {
            file: file.to_string(),
            ..args(None)
        };
        let holds = br#"{"chat_template": "{{ bos_token }}"}"#;
        let source = |patch: Value| probed_policy(&patch).chat_template;
        assert_eq!(
            source(
                ChatTemplateProbe
                    .probe(holds, &keyed("tokenizer_config"))
                    .unwrap()
            ),
            Some(ChatTemplateSource::File {
                file: "tokenizer_config".to_string()
            })
        );
        assert_eq!(
            ChatTemplateProbe
                .probe(br#"{"eos_token": "<eos>"}"#, &keyed("tokenizer_config"))
                .unwrap(),
            json!({})
        );
        assert_eq!(
            source(
                ChatTemplateProbe
                    .probe(
                        b"{{ bos_token }}\n{%- for m in messages -%}",
                        &keyed("chat_template")
                    )
                    .unwrap()
            ),
            Some(ChatTemplateSource::File {
                file: "chat_template".to_string()
            })
        );

        let mut probed = Value::Object(Map::new());
        for (file, bytes) in [
            ("tokenizer_config", &holds[..]),
            ("chat_template", b"{{ bos_token }}"),
        ] {
            accumulate_probed(
                &mut probed,
                &ChatTemplateProbe.probe(bytes, &keyed(file)).unwrap(),
            );
        }
        assert_eq!(
            probed_policy(&probed).chat_template,
            Some(ChatTemplateSource::File {
                file: "chat_template".to_string()
            })
        );
    }
}
