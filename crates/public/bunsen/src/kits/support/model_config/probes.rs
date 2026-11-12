//! Probes: readers of a pretrained's files that find its presets' values.

use alloc::{
    collections::BTreeMap,
    fmt,
};

use serde_json::{
    Map,
    Value,
    json,
};

use crate::{
    data::pretrained::{
        CatalogProbe,
        Preset,
        PresetSet,
    },
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    support::merge_patch,
};

/// A reader of one of a pretrained's files that finds values for its
/// presets: a merge patch over the family's config.
///
/// Probes are found by [`name`](Self::name). The built-in ones
/// ([`builtin_probes`]) read the files a Hugging Face repo carries; a caller
/// with more, a template renderer say, passes them to
/// [`apply_probes_with`].
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

/// A pretrained's files, by resource key, for its probes to read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProbeFiles {
    files: BTreeMap<String, Vec<u8>>,
}

impl ProbeFiles {
    /// No files.
    pub fn new() -> Self {
        Self::default()
    }

    /// The files with `bytes` under `key`.
    pub fn with(
        mut self,
        key: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        self.insert(key, bytes);
        self
    }

    /// Puts `bytes` under `key`.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) {
        self.files.insert(key.into(), bytes.into());
    }

    /// The file under `key`.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when there is none;
    /// its candidates are the keys there are.
    pub fn get(
        &self,
        key: &str,
    ) -> BunsenResult<&[u8]> {
        self.files.get(key).map(Vec::as_slice).ok_or_else(|| {
            BunsenError::lookup(
                LookupError::missing("probe file", key).with_candidates(self.files.keys().cloned()),
            )
        })
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

/// `set` with what `entries` find in `files` beneath it, by the built-in
/// probes: [`apply_probes_with`] over [`builtin_probes`].
///
/// # Errors
/// As [`apply_probes_with`].
pub fn apply_probes(
    set: &PresetSet,
    family: &PresetSet,
    entries: &[CatalogProbe],
    files: &ProbeFiles,
) -> BunsenResult<PresetSet> {
    let probes = builtin_probes();
    let probes: Vec<&dyn Probe> = probes.iter().map(Box::as_ref).collect();
    apply_probes_with(set, family, entries, files, &probes)
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

/// `set` with what `entries` find in `files` beneath it: each entry's probe
/// reads its file, and its patch goes into the set's base (`into: [base]`)
/// or into the presets it names, accumulating with the other probes'
/// (arrays join: two probes' stop ids together); then `set`'s own base and
/// presets are laid over what was found, as merge patches, so a
/// hand-written patch extends or corrects a probed one (and replaces a
/// probed array). A preset an entry names must be one `set` or its `family`
/// declares (empty for a row with none); one only the family declares joins
/// `set` as the probed patch alone, with no description, to be laid over the
/// family's ([`PresetSet::over`]).
///
/// # Errors
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup): an unknown probe (its
///   candidates are `probes`), a file `files` lacks, or a preset `set` does not
///   declare.
/// - As a probe's [`probe`](Probe::probe), under a frame naming the entry.
pub fn apply_probes_with(
    set: &PresetSet,
    family: &PresetSet,
    entries: &[CatalogProbe],
    files: &ProbeFiles,
    probes: &[&dyn Probe],
) -> BunsenResult<PresetSet> {
    let mut base = Value::Object(Map::new());
    let mut found: BTreeMap<String, Value> = BTreeMap::new();
    for entry in entries {
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
            .probe(files.get(&entry.file)?, entry)
            .map_err(|e| e.context(format!("the {} probe of {:?}", entry.probe, entry.file)))?;
        for into in &entry.into {
            if into == "base" {
                accumulate_probed(&mut base, &patch);
                continue;
            }
            if set.get(into).is_none() && family.get(into).is_none() {
                return Err(BunsenError::lookup(
                    LookupError::missing("preset", into.clone())
                        .with_candidates(set.names().chain(family.names())),
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

    let mut out = set.clone();
    merge_patch::apply(&mut base, &set.base);
    out.base = base;
    for preset in &mut out.presets {
        if let Some(probed) = found.remove(&preset.name) {
            let mut patch = probed;
            merge_patch::apply(&mut patch, &preset.patch);
            preset.patch = patch;
        }
    }
    // What is left went into presets only the family declares.
    for (name, patch) in found {
        out.presets.push(Preset {
            name,
            description: String::new(),
            patch,
        });
    }
    Ok(out)
}

/// The presets of a checkpoint no catalog row describes (an `hf:` repo's),
/// as the catalog tool probes a row's: each of `entries` whose file `loaded`
/// has, the rest skipped, into a set laid over `family`'s.
///
/// # Errors
/// [`Sys`](crate::errors::BunsenErrorKind::Sys) when a file cannot be read;
/// as [`apply_probes`].
#[cfg(feature = "cache")]
pub fn probe_loaded(
    loaded: &crate::data::pretrained::LoadedResources,
    entries: &[CatalogProbe],
    family: &PresetSet,
) -> BunsenResult<PresetSet> {
    let mut files = ProbeFiles::new();
    let mut kept = Vec::new();
    for entry in entries {
        let Some(part) = loaded.get(&entry.file) else {
            continue;
        };
        if files.get(&entry.file).is_err() {
            let bytes =
                std::fs::read(&part.path).map_err(crate::errors::sys_at("read", &part.path))?;
            files.insert(entry.file.clone(), bytes);
        }
        kept.push(entry.clone());
    }
    Ok(apply_probes(&PresetSet::default(), family, &kept, &files)?.over(family))
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

    /// What probes find goes into the base or the presets they name, and the
    /// set's own patches are laid over it.
    #[test]
    fn test_apply_probes() {
        let set = PresetSet {
            base: json!({"generation": {"max_new_tokens": 512}}),
            presets: vec![
                Preset {
                    name: "thinking".to_string(),
                    description: "think first".to_string(),
                    patch: json!({"generation": {"template_kwargs": {"enable_thinking": true}}}),
                },
                Preset {
                    name: "no-thinking".to_string(),
                    description: "answer at once".to_string(),
                    patch: json!({"generation": {"sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": null}}}),
                },
            ],
            ..Default::default()
        };
        let files = ProbeFiles::new().with("generation_config", QWEN3_GENERATION);
        let probed = apply_probes(
            &set,
            &PresetSet::default(),
            &[entry(
                "generation_config",
                "generation_config",
                &["base", "thinking"],
            )],
            &files,
        )
        .unwrap();

        let base = probed_policy(&probed.base);
        assert_eq!(base.stop_ids, [151645, 151643]);
        assert_eq!(
            base.max_new_tokens,
            Some(512),
            "the set's own base over the probe's"
        );
        let thinking = &probed.get("thinking").unwrap().patch;
        assert_eq!(thinking["generation"]["sampling"]["temperature"], 0.6);
        assert_eq!(
            thinking["generation"]["template_kwargs"]["enable_thinking"],
            true
        );
        assert_eq!(
            probed.get("no-thinking").unwrap().patch,
            set.get("no-thinking").unwrap().patch
        );

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no probe \"sampling_card\"")
            .assert_err(&apply_probes(
                &set,
                &PresetSet::default(),
                &[entry("sampling_card", "generation_config", &["base"])],
                &files,
            ));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no probe file \"tokenizer_config\"")
            .assert_err(&apply_probes(
                &set,
                &PresetSet::default(),
                &[entry("tokenizer_config", "tokenizer_config", &["base"])],
                &files,
            ));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no preset \"fast\"")
            .assert_err(&apply_probes(
                &set,
                &PresetSet::default(),
                &[entry("generation_config", "generation_config", &["fast"])],
                &files,
            ));
    }

    /// A probe into a preset only the row's family declares makes a partial
    /// row preset, which the family's is laid under at load.
    #[test]
    fn test_apply_probes_into_a_family_preset() {
        let family = PresetSet {
            presets: vec![Preset {
                name: "thinking".to_string(),
                description: "think first".to_string(),
                patch: json!({"generation": {"template_kwargs": {"enable_thinking": true}}}),
            }],
            ..Default::default()
        };
        let files = ProbeFiles::new().with("generation_config", QWEN3_GENERATION);
        let row = apply_probes(
            &PresetSet::default(),
            &family,
            &[entry(
                "generation_config",
                "generation_config",
                &["thinking"],
            )],
            &files,
        )
        .unwrap();
        let partial = row.get("thinking").unwrap();
        assert!(partial.description.is_empty());
        let merged = row.over(&family);
        merged.validate().unwrap();
        let thinking = &merged.get("thinking").unwrap().patch["generation"];
        assert_eq!(thinking["template_kwargs"]["enable_thinking"], true);
        assert_eq!(thinking["sampling"]["top_k"], 20);
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
    /// repeat. A hand-written base laid over them replaces the array.
    #[test]
    fn test_probes_accumulate_and_hand_written_replaces() {
        let files = ProbeFiles::new()
            .with("generation_config", br#"{"eos_token_id": [1]}"#.to_vec())
            .with(
                "tokenizer_config",
                br#"{"eos_token": "<eos>", "added_tokens_decoder": {
                    "1": {"content": "<eos>"}, "107": {"content": "<end_of_turn>"}}}"#
                    .to_vec(),
            );
        let turn = CatalogProbe {
            token: Some("<end_of_turn>".to_string()),
            ..entry("tokenizer_config", "tokenizer_config", &["base"])
        };
        let entries = [
            entry("generation_config", "generation_config", &["base"]),
            turn.clone(),
            entry("tokenizer_config", "tokenizer_config", &["base"]),
        ];
        let probed = apply_probes(
            &PresetSet::default(),
            &PresetSet::default(),
            &entries,
            &files,
        )
        .unwrap();
        assert_eq!(probed_policy(&probed.base).stop_ids, [1, 107]);

        let hand = PresetSet {
            base: json!({"generation": {"stop_ids": [7]}}),
            ..PresetSet::default()
        };
        let over = apply_probes(&hand, &PresetSet::default(), &entries, &files).unwrap();
        assert_eq!(probed_policy(&over.base).stop_ids, [7]);
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

        let files = ProbeFiles::new()
            .with("tokenizer_config", holds.to_vec())
            .with("chat_template", b"{{ bos_token }}".to_vec());
        let probed = apply_probes(
            &PresetSet::default(),
            &PresetSet::default(),
            &[
                entry("chat_template", "tokenizer_config", &["base"]),
                entry("chat_template", "chat_template", &["base"]),
            ],
            &files,
        )
        .unwrap();
        assert_eq!(
            probed_policy(&probed.base).chat_template,
            Some(ChatTemplateSource::File {
                file: "chat_template".to_string()
            })
        );
    }
}
