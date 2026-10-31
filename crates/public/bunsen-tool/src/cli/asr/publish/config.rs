//! `--publish`: the targets, as a JSON or YAML config, and the shorthands.
use std::{
    collections::BTreeMap,
    path::{
        Path,
        PathBuf,
    },
};

use bunsen::errors::BunsenResult;
use serde::Deserialize;
use serde_json::{
    Map,
    Value,
};

use super::{
    Clock,
    Cx,
    Publisher,
    Sink,
    writer::{
        Format,
        WriterSink,
    },
};

/// Where the transcript goes, whichever command makes it.
#[derive(clap::Args, Clone, Debug)]
pub struct PublishArgs {
    /// Where the transcript goes, in place of stdout's text: a config,
    /// inline or a file's path, in JSON or YAML (`.json`, `.yaml`; inline,
    /// either). The config is
    /// `{targets: [...]}`, with `times`, `speakers` and `events` set for
    /// every target, or the list of targets alone. A target is
    /// `{type: stdout|file|log|webhook|mqtt, ...}`, and each may set its own
    /// `times` (start and end; from 0 for files, the wall clock for the
    /// microphone), `speakers` (under `--diarize`) and `events` (of
    /// `statement`, `draft`, `transcript`). The README has each target's
    /// fields.
    #[arg(
        long,
        value_name = "CONFIG",
        value_parser = PublishConfig::parse_arg,
        conflicts_with_all = ["json", "rttm"]
    )]
    pub publish: Option<PublishConfig>,

    /// Print JSON lines: `--publish '{type: stdout, format: jsonl}'`.
    #[arg(long)]
    pub json: bool,

    /// Print NIST RTTM, a `SPEAKER` line per statement: `--publish '{type:
    /// stdout, format: rttm}'`. Needs `--diarize`.
    #[arg(long, requires = "diarize", conflicts_with = "json")]
    pub rttm: bool,
}

impl PublishArgs {
    /// The config: `--publish`'s, or stdout's, in the format a shorthand
    /// names.
    fn config(&self) -> PublishConfig {
        if let Some(config) = &self.publish {
            return config.clone();
        }
        let format = if self.json {
            Format::Jsonl
        } else if self.rttm {
            Format::Rttm
        } else {
            Format::Text
        };
        PublishConfig {
            targets: vec![Target {
                kind: Kind::Stdout { format },
                opts: OptsConfig::default(),
            }],
        }
    }

    /// Opens every target: the files, the connections. `clock` reads the
    /// stream's times; `diarized` is whether statements carry speakers.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for an RTTM
    /// target without `--diarize`, or a broker that does not parse;
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) for a
    /// network target this build lacks the feature for; as opening a file.
    pub fn init(
        &self,
        clock: Clock,
        diarized: bool,
    ) -> BunsenResult<Publisher> {
        let mut sinks = Vec::new();
        for target in self.config().targets {
            let opts = target.opts.resolve(&target.kind);
            let sink: Box<dyn Sink> = match target.kind {
                Kind::Stdout { format } => Box::new(WriterSink::stdout(format, diarized)?),
                Kind::File {
                    path,
                    format,
                    append,
                } => Box::new(WriterSink::file(&path, format, append, diarized)?),
                Kind::Log { level, target } => Box::new(LogSink {
                    level: level.level(),
                    target,
                }),
                #[cfg(feature = "webhook")]
                Kind::Webhook { url, headers } => {
                    Box::new(super::webhook::WebhookSink::open(url, headers))
                }
                #[cfg(feature = "mqtt")]
                Kind::Mqtt(mqtt) => Box::new(super::mqtt::MqttSink::open(mqtt)?),
                #[cfg(not(feature = "webhook"))]
                Kind::Webhook { .. } => return Err(unbuilt("webhook")),
                #[cfg(not(feature = "mqtt"))]
                Kind::Mqtt(_) => return Err(unbuilt("mqtt")),
            };
            sinks.push((opts, sink));
        }
        Ok(Publisher {
            sinks,
            cx: Cx { clock, diarized },
        })
    }
}

/// The error for a target whose feature this build lacks.
#[cfg(any(not(feature = "webhook"), not(feature = "mqtt")))]
fn unbuilt(feature: &str) -> bunsen::errors::BunsenError {
    bunsen::errors::BunsenError::unsupported(format!(
        "a `{feature}` target needs bunsen-tool built with `--features {feature}`"
    ))
}

/// The events a record can be.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Statement,
    Draft,
    Transcript,
}

impl EventKind {
    pub const ALL: [EventKind; 3] = [
        EventKind::Statement,
        EventKind::Draft,
        EventKind::Transcript,
    ];

    /// The name a record's `event` gives it: an MQTT topic's last level.
    #[cfg_attr(not(feature = "mqtt"), allow(dead_code))]
    pub fn name(self) -> &'static str {
        match self {
            EventKind::Statement => "statement",
            EventKind::Draft => "draft",
            EventKind::Transcript => "transcript",
        }
    }
}

/// What a target shows, as the config sets it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct OptsConfig {
    times: Option<bool>,
    speakers: Option<bool>,
    events: Option<Vec<EventKind>>,
}

impl OptsConfig {
    /// `self`, each unset field `base`'s.
    fn over(
        self,
        base: &OptsConfig,
    ) -> OptsConfig {
        OptsConfig {
            times: self.times.or(base.times),
            speakers: self.speakers.or(base.speakers),
            events: self.events.or_else(|| base.events.clone()),
        }
    }

    /// What a target of `kind` shows: times on for records, off for text;
    /// speakers on; every event.
    fn resolve(
        &self,
        kind: &Kind,
    ) -> Opts {
        let text = matches!(
            kind,
            Kind::Stdout {
                format: Format::Text
            } | Kind::File {
                format: Format::Text,
                ..
            }
        );
        Opts {
            times: self.times.unwrap_or(!text),
            speakers: self.speakers.unwrap_or(true),
            events: self
                .events
                .clone()
                .unwrap_or_else(|| EventKind::ALL.to_vec()),
        }
    }
}

/// What a target shows.
#[derive(Clone, Debug, PartialEq)]
pub struct Opts {
    /// Each statement's and draft's start and end.
    pub times: bool,

    /// Under `--diarize`, each statement's speaker.
    pub speakers: bool,

    /// The events it takes.
    pub events: Vec<EventKind>,
}

/// A `log` target's level.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LogLevel {
    Error,
    Warn,
    #[default]
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn level(self) -> log::Level {
        match self {
            LogLevel::Error => log::Level::Error,
            LogLevel::Warn => log::Level::Warn,
            LogLevel::Info => log::Level::Info,
            LogLevel::Debug => log::Level::Debug,
            LogLevel::Trace => log::Level::Trace,
        }
    }
}

fn default_log_target() -> String {
    "bunsen_tool::transcript".to_string()
}

/// An `mqtt` target's fields.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MqttConfig {
    /// `[mqtt://]HOST[:PORT]`, port 1883 by default.
    pub broker: String,

    /// Each record goes to `{prefix}/{event}`.
    #[serde(default = "default_mqtt_prefix")]
    pub prefix: String,

    /// 0, 1 or 2.
    #[serde(default = "default_mqtt_qos")]
    pub qos: u8,

    #[serde(default)]
    pub retain: bool,

    /// `bunsen-tool-{pid}` by default.
    pub client_id: Option<String>,

    pub username: Option<String>,

    pub password: Option<String>,
}

fn default_mqtt_prefix() -> String {
    "bunsen-tool".to_string()
}

fn default_mqtt_qos() -> u8 {
    1
}

/// A target's kind, and its own fields.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Kind {
    Stdout {
        #[serde(default)]
        format: Format,
    },
    File {
        path: PathBuf,
        #[serde(default)]
        format: Format,
        #[serde(default)]
        append: bool,
    },
    Log {
        #[serde(default)]
        level: LogLevel,
        #[serde(default = "default_log_target")]
        target: String,
    },
    Webhook {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
    Mqtt(MqttConfig),
}

/// One target.
#[derive(Clone, Debug, PartialEq)]
struct Target {
    kind: Kind,
    opts: OptsConfig,
}

/// `--publish`'s config: the targets, each with what it shows.
#[derive(Clone, Debug, PartialEq)]
pub struct PublishConfig {
    targets: Vec<Target>,
}

/// What a config is written in.
#[derive(Clone, Copy, Debug)]
enum Syntax {
    Json,
    Yaml,
    Either,
}

/// The keys of [`OptsConfig`], which a target and the config share.
const OPTS_KEYS: [&str; 3] = ["times", "speakers", "events"];

/// The fields of `object` named by [`OPTS_KEYS`], taken out of it.
fn take_opts(object: &mut Map<String, Value>) -> Result<OptsConfig, String> {
    let opts: Map<String, Value> = OPTS_KEYS
        .iter()
        .filter_map(|key| object.remove(*key).map(|v| (key.to_string(), v)))
        .collect();
    serde_json::from_value(Value::Object(opts)).map_err(|e| e.to_string())
}

impl PublishConfig {
    /// `--publish`'s value: a file's path, read as JSON for `.json`, as YAML
    /// for `.yaml` and `.yml`, and as either otherwise; else the config
    /// itself, read as either.
    pub fn parse_arg(arg: &str) -> Result<Self, String> {
        let path = Path::new(arg);
        if !path.is_file() {
            return Self::parse(arg, Syntax::Either);
        }
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let syntax = match path.extension().and_then(|e| e.to_str()) {
            Some("json") => Syntax::Json,
            Some("yaml" | "yml") => Syntax::Yaml,
            _ => Syntax::Either,
        };
        Self::parse(&text, syntax).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// `text` as a config, in `syntax`: either is JSON when it parses as
    /// JSON, and YAML otherwise (YAML's flow style, `{type: stdout}`, is
    /// not JSON).
    fn parse(
        text: &str,
        syntax: Syntax,
    ) -> Result<Self, String> {
        let json = || serde_json::from_str::<Value>(text).map_err(|e| format!("not JSON: {e}"));
        let yaml = || serde_saphyr::from_str::<Value>(text).map_err(|e| format!("not YAML: {e}"));
        let value = match syntax {
            Syntax::Json => json()?,
            Syntax::Yaml => yaml()?,
            Syntax::Either => match json() {
                Ok(value) => value,
                Err(json) => yaml().map_err(|yaml| format!("{json}; {yaml}"))?,
            },
        };
        Self::from_value(value)
    }

    /// The config `value` holds: `{targets: [...], times?, speakers?,
    /// events?}`, or the list of targets alone.
    fn from_value(value: Value) -> Result<Self, String> {
        let (targets, base) = match value {
            Value::Array(targets) => (targets, OptsConfig::default()),
            Value::Object(mut object) => {
                let base = take_opts(&mut object)?;
                let targets = match object.remove("targets") {
                    Some(Value::Array(targets)) => targets,
                    Some(_) => return Err("`targets` is not a list".to_string()),
                    None => return Err("the config has no `targets`".to_string()),
                };
                if let Some(key) = object.keys().next() {
                    return Err(format!("unknown field `{key}` in the config"));
                }
                (targets, base)
            }
            Value::String(s) => return Err(format!("no such file, and not a config: {s}")),
            _ => return Err("a config is a mapping, or a list of targets".to_string()),
        };
        if targets.is_empty() {
            return Err("the config has no targets".to_string());
        }
        let targets = targets
            .into_iter()
            .enumerate()
            .map(|(i, target)| {
                let Value::Object(mut object) = target else {
                    return Err(format!("target {i} is not a mapping"));
                };
                let opts = take_opts(&mut object)
                    .map_err(|e| format!("target {i}: {e}"))?
                    .over(&base);
                let kind: Kind = serde_json::from_value(Value::Object(object))
                    .map_err(|e| format!("target {i}: {e}"))?;
                Ok(Target { kind, opts })
            })
            .collect::<Result<_, String>>()?;
        Ok(PublishConfig { targets })
    }
}

/// Logs each record, a JSON line, at its level under its target.
struct LogSink {
    level: log::Level,
    target: String,
}

impl Sink for LogSink {
    fn push(
        &mut self,
        event: &super::Event,
        cx: &Cx,
        opts: &Opts,
    ) -> BunsenResult<()> {
        log::log!(
            target: &self.target,
            self.level,
            "{}",
            super::record::json_line(event, opts, cx)
        );
        Ok(())
    }

    fn finish(self: Box<Self>) -> BunsenResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> PublishConfig {
        PublishConfig::parse_arg(text).unwrap()
    }

    /// JSON and YAML read alike; the config's `times` is each target's
    /// unless the target sets its own.
    #[test]
    fn test_parse() {
        let json = parse(
            r#"{"times": true, "targets": [
                {"type": "stdout"},
                {"type": "mqtt", "broker": "localhost", "times": false},
                {"type": "log", "level": "debug", "events": ["statement"]}
            ]}"#,
        );
        let yaml = parse(
            "times: true\n\
             targets:\n\
             - type: stdout\n\
             - {type: mqtt, broker: localhost, times: false}\n\
             - type: log\n  level: debug\n  events: [statement]\n",
        );
        assert_eq!(json, yaml);
        let t = &json.targets;
        assert_eq!(
            t[0].kind,
            Kind::Stdout {
                format: Format::Text
            }
        );
        assert_eq!(t[0].opts.times, Some(true));
        assert_eq!(t[1].opts.times, Some(false));
        match &t[1].kind {
            Kind::Mqtt(mqtt) => {
                assert_eq!(
                    (mqtt.prefix.as_str(), mqtt.qos, mqtt.retain),
                    ("bunsen-tool", 1, false)
                );
            }
            kind => panic!("{kind:?}"),
        }
        assert_eq!(t[2].opts.events, Some(vec![EventKind::Statement]));

        // A list of targets alone.
        let list = parse("[{type: file, path: out.jsonl, format: jsonl}]");
        assert_eq!(list.targets.len(), 1);
        assert_eq!(list.targets[0].opts, OptsConfig::default());
    }

    #[test]
    fn test_parse_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("publish.yaml");
        std::fs::write(&path, "targets:\n- {type: stdout, format: jsonl}\n").unwrap();
        let config = parse(path.to_str().unwrap());
        assert_eq!(
            config.targets[0].kind,
            Kind::Stdout {
                format: Format::Jsonl
            }
        );
        let path = dir.path().join("publish.json");
        std::fs::write(&path, r#"[{"type": "stdout", "format": "rttm"}]"#).unwrap();
        assert_eq!(
            parse(path.to_str().unwrap()).targets[0].kind,
            Kind::Stdout {
                format: Format::Rttm
            }
        );
    }

    /// A misspelled field, kind or option fails, as does a missing file.
    #[test]
    fn test_parse_errors() {
        for bad in [
            r#"{"targets": [{"type": "stdout", "fromat": "jsonl"}]}"#,
            r#"{"targets": [{"type": "carrier-pigeon"}]}"#,
            r#"{"targets": [{"type": "stdout", "times": "yes"}]}"#,
            r#"{"targets": [{"type": "stdout"}], "time": true}"#,
            r#"{"targets": []}"#,
            r#"{"type": "stdout"}"#,
            "no/such/publish.yaml",
        ] {
            assert!(PublishConfig::parse_arg(bad).is_err(), "{bad}");
        }
    }

    /// Times are off for text and on for records, unless set; every event
    /// and speakers by default.
    #[test]
    fn test_resolve() {
        let opts = OptsConfig::default();
        let text = Kind::Stdout {
            format: Format::Text,
        };
        let jsonl = Kind::Log {
            level: LogLevel::Info,
            target: default_log_target(),
        };
        assert!(!opts.resolve(&text).times);
        assert!(opts.resolve(&jsonl).times);
        assert!(opts.resolve(&jsonl).speakers);
        assert_eq!(opts.resolve(&jsonl).events, EventKind::ALL);
        let set = OptsConfig {
            times: Some(true),
            ..Default::default()
        };
        assert!(set.resolve(&text).times);
    }
}
