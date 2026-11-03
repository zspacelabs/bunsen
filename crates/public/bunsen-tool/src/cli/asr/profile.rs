//! Profiles: speech to text by what it is for, not by which model does it.

use std::{
    collections::BTreeMap,
    fmt,
    str::FromStr,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::construct::LookupExpr,
};
use clap::{
    Arg,
    Command,
    Parser,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::cli::asr::{
    family::Family,
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    stream::Emission,
    voxtral::VoxtralFlags,
    whisper::WhisperFlags,
};

/// Which of a profile's recipes a command runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Files: `transcribe`.
    Offline,

    /// The microphone: `live`.
    Live,
}

impl Mode {
    /// The command that runs this recipe, a profile's or a family's.
    pub fn command(self) -> &'static str {
        match self {
            Mode::Offline => "transcribe",
            Mode::Live => "live",
        }
    }
}

/// What the speech is: the table's language axis.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Speech {
    /// English.
    En,

    /// Any language a model finds from the speech; English among them.
    Multi,
}

/// What to make of the speech.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// Text in the language spoken.
    Transcribe,

    /// English text.
    Translate,
}

/// How much compute to spend on it.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Fast,
    Balanced,
    Good,
}

/// A point in the table: what the speech is, what to make of it, how much
/// to spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Need {
    pub speech: Speech,
    pub task: Task,
    pub quality: Quality,
}

impl fmt::Display for Need {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        use clap::ValueEnum;
        let name = |v: Option<clap::builder::PossibleValue>| {
            v.expect("no variant is skipped").get_name().to_string()
        };
        write!(
            f,
            "{}/{}/{}",
            name(self.speech.to_possible_value()),
            name(self.task.to_possible_value()),
            name(self.quality.to_possible_value())
        )
    }
}

impl Need {
    /// This need as it is served: each of the catalog's implications that
    /// matches it is applied, in order. English speech translated to English
    /// is transcribed, for one: out of every model's training, and Whisper's
    /// output for it is its transcript.
    pub fn settled(self) -> Need {
        self.settled_why().0
    }

    /// [`settled`](Self::settled), with the reasons of the implications
    /// that changed it, in order.
    pub fn settled_why(self) -> (Need, Vec<&'static str>) {
        let mut reasons = Vec::new();
        let need = catalog().implications.iter().fold(self, |need, rule| {
            if rule.when.matches(need) && rule.then.apply(need) != need {
                reasons.push(rule.reason.as_str());
                rule.then.apply(need)
            } else {
                need
            }
        });
        (need, reasons)
    }

    /// The points that can stand in for this settled one, nearest first,
    /// this one first, by the catalog's fallback: another quality before
    /// other speech, as a multilingual model also hears English. The task is
    /// never changed.
    fn candidates(self) -> Vec<Need> {
        let fallback = &catalog().fallback;
        let speeches = fallback
            .speech
            .get(&self.speech)
            .cloned()
            .unwrap_or_else(|| vec![self.speech]);
        let qualities = fallback
            .quality
            .get(&self.quality)
            .cloned()
            .unwrap_or_else(|| vec![self.quality]);
        let mut out = Vec::new();
        for &speech in &speeches {
            for &quality in &qualities {
                out.push(Need {
                    speech,
                    task: self.task,
                    quality,
                });
            }
        }
        out
    }
}

/// One way to run a family: what `bunsen-tool asr family <family>
/// transcribe|live` would be given. The engine's model is a lookup
/// expression, its presets and patch included
/// (`openai/large-v3[translate]{beam_size: 5}`), so a recipe is always a
/// command `bunsen-tool asr family` runs.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// The engine slot: the family that runs, and its model.
    pub engine: EngineSlot,

    pub emission: Emission,

    /// Milliseconds of audio per push.
    pub chunk_ms: usize,
}

/// A recipe's engine: a family, and its model as a lookup expression.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineSlot {
    pub family: Family,

    /// The model, as `--model` names it.
    pub model: String,
}

/// One cell of the table: what runs for a need, in a mode.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub need: Need,
    pub mode: Mode,
    pub recipe: Recipe,
}

/// A named, semantic choice of model: a point of the table, as `--preset`
/// names it.
///
/// A preset names what the speech is for (`en`, `multi`, `translate-en`),
/// with one recipe for files (`transcribe`) and one for the microphone
/// (`live`). The table is this binary's data: when a better model arrives, a
/// cell changes, and the command lines that name the preset do not.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub description: String,
    pub need: Need,
}

impl Profile {
    /// The recipe `mode` runs.
    pub fn recipe(
        &self,
        mode: Mode,
    ) -> &'static Recipe {
        &pick(self.need, mode).0.recipe
    }

    /// The recipe `mode` runs, as it is, resolved.
    ///
    /// # Errors
    /// As [`Recipe::resolve`].
    pub fn resolve(
        &self,
        mode: Mode,
    ) -> BunsenResult<Resolved> {
        self.recipe(mode)
            .resolve(&self.name, self.need, mode, None, &[])
    }
}

/// One axis of the table, and its members, each described.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axis {
    pub name: String,
    pub description: String,
    pub members: Vec<AxisMember>,
}

/// A member of an [`Axis`].
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisMember {
    pub name: String,
    pub description: String,
}

/// Where a need without a cell looks next: per axis, each value's
/// candidates in order, the value itself first.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fallback {
    pub quality: BTreeMap<Quality, Vec<Quality>>,
    pub speech: BTreeMap<Speech, Vec<Speech>>,
}

/// Part of a need: the axes it names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeedPattern {
    #[serde(default)]
    pub speech: Option<Speech>,
    #[serde(default)]
    pub task: Option<Task>,
    #[serde(default)]
    pub quality: Option<Quality>,
}

impl NeedPattern {
    /// Whether `need` has every axis this names.
    fn matches(
        &self,
        need: Need,
    ) -> bool {
        self.speech.is_none_or(|s| s == need.speech)
            && self.task.is_none_or(|t| t == need.task)
            && self.quality.is_none_or(|q| q == need.quality)
    }

    /// `need` with the axes this names set.
    fn apply(
        &self,
        need: Need,
    ) -> Need {
        Need {
            speech: self.speech.unwrap_or(need.speech),
            task: self.task.unwrap_or(need.task),
            quality: self.quality.unwrap_or(need.quality),
        }
    }
}

/// A need that is served as another: when it matches `when`, `then` is set
/// over it.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Implication {
    pub when: NeedPattern,
    pub then: NeedPattern,
    pub reason: String,
}

/// The ASR construct's table, as data: `asr_construct.yml`, embedded and
/// read once ([`catalog`]).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub axes: Vec<Axis>,
    pub presets: Vec<Profile>,
    pub fallback: Fallback,
    pub implications: Vec<Implication>,
    pub points: Vec<Cell>,
}

/// The table, from the binary's `asr_construct.yml`.
///
/// # Panics
/// If the embedded file is not a catalog: a broken build, which the tests
/// catch.
pub fn catalog() -> &'static Catalog {
    static CATALOG: std::sync::OnceLock<Catalog> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_saphyr::from_str(include_str!("asr_construct.yml"))
            .unwrap_or_else(|e| panic!("the embedded asr_construct.yml: {e}"))
    })
}

/// The cell for `need`, [settled](Need::settled), in `mode`, and, when it
/// is not `need`'s own, a note of what stood in.
pub fn pick(
    need: Need,
    mode: Mode,
) -> (&'static Cell, Option<String>) {
    let table = catalog();
    for candidate in need.settled().candidates() {
        if let Some(cell) = table
            .points
            .iter()
            .find(|c| c.need == candidate && c.mode == mode)
        {
            let note = (candidate != need).then(|| {
                format!(
                    "no {} recipe for {need}: {candidate} stands in",
                    mode.command()
                )
            });
            return (cell, note);
        }
    }
    unreachable!("every speech and task reaches a balanced cell; see asr_construct.yml")
}

/// The preset `--preset` names when it is not given: an alias, so what it
/// means can change too.
pub const DEFAULT_PROFILE: &str = "default";

/// The presets: names for the points of the table people ask for.
pub fn profiles() -> &'static [Profile] {
    &catalog().presets
}

/// The preset `name` names, by its name or an alias.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when no preset has
/// the name: it comes from the command line.
pub fn lookup(name: &str) -> BunsenResult<&'static Profile> {
    profiles()
        .iter()
        .find(|p| p.name == name || p.aliases.iter().any(|a| a == name))
        .ok_or_else(|| {
            let names: Vec<&str> = profiles().iter().map(|p| p.name.as_str()).collect();
            BunsenError::policy(format!(
                "no preset `{name}`; the presets: {}",
                names.join(", ")
            ))
        })
}

/// One `--opt`: a family's own option, for whichever family the profile
/// runs, or only for `family`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opt {
    /// The family it is for; any, when `None`.
    pub family: Option<Family>,

    /// The option's long name, or `model`.
    pub key: String,

    pub value: String,
}

impl FromStr for Opt {
    type Err = String;

    /// `[FAMILY.]KEY=VALUE`, or `[FAMILY.]KEY` for `KEY=true`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (key, value) = s.split_once('=').unwrap_or((s, "true"));
        let (family, key) = match key.split_once('.') {
            Some((family, key)) => (
                Some(Family::from_name(family).ok_or_else(|| {
                    let names: Vec<&str> = Family::ALL.iter().map(|f| f.name()).collect();
                    format!("`{family}` is not a family: {}", names.join(", "))
                })?),
                key,
            ),
            None => (None, key),
        };
        let key = key.trim_start_matches("--");
        if key.is_empty() {
            return Err(format!("`{s}` names no option"));
        }
        Ok(Opt {
            family,
            key: key.to_string(),
            value: value.to_string(),
        })
    }
}

impl fmt::Display for Opt {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if let Some(family) = self.family {
            write!(f, "{}.", family.name())?;
        }
        write!(f, "{}={}", self.key, self.value)
    }
}

/// The family's own options, as its command declares them.
fn family_args(family: Family) -> Vec<Arg> {
    let command = Command::new(family.name());
    let command = match family {
        Family::Parakeet => <ParakeetFlags as clap::Args>::augment_args(command),
        Family::Qwen3Asr => <Qwen3AsrFlags as clap::Args>::augment_args(command),
        Family::Voxtral => <VoxtralFlags as clap::Args>::augment_args(command),
        Family::Whisper => <WhisperFlags as clap::Args>::augment_args(command),
    };
    command
        .get_arguments()
        .filter(|a| a.get_long().is_some())
        .cloned()
        .collect()
}

/// The keys `--opt` takes for `family`: `model`, and its own options.
fn family_keys(family: Family) -> String {
    let mut keys = vec!["model".to_string()];
    keys.extend(
        family_args(family)
            .iter()
            .filter_map(|a| a.get_long().map(String::from)),
    );
    keys.join(", ")
}

/// Whether `family` has an option `key`; `model` is every family's.
fn has_key(
    family: Family,
    key: &str,
) -> bool {
    key == "model"
        || family_args(family)
            .iter()
            .any(|a| a.get_long() == Some(key))
}

/// The family's own flags alone, parsed as its command parses them.
#[derive(Parser, Debug)]
#[command(no_binary_name = true)]
struct FlagsOnly<F: clap::Args> {
    #[command(flatten)]
    flags: F,
}

/// A profile's recipe, with the command line's choices over it: what runs.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The profile's name, not the alias it was named by; or the need the
    /// table was asked for (`multi/translate/fast`).
    pub profile: String,

    /// The need it serves, [settled](Need::settled): its task is the one
    /// the model runs, an `--opt task=` included.
    pub need: Need,

    pub mode: Mode,
    pub family: Family,
    pub model: String,
    pub emission: Emission,
    pub chunk_ms: usize,

    /// The family's own flags, by long name, each once, in order.
    pub flags: Vec<(String, String)>,
}

impl Recipe {
    /// This recipe of `profile`, serving `need`, with `language` and `opts`
    /// over it.
    ///
    /// `language` is Whisper's `--language`; a family that finds the
    /// language itself is logged, at `INFO`, as not taking it. An `opt` for
    /// another family is logged as not applied; one for this family, or for
    /// none, replaces the recipe's value. Whisper's `--task`, as it ends,
    /// is the need's task; another family transcribes.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when an `opt`
    /// names an option its family does not have (or, for none, this
    /// family), or a flag that takes no value is given one but `true` or
    /// `false`: they come from the command line.
    pub fn resolve(
        &self,
        profile: &str,
        need: Need,
        mode: Mode,
        language: Option<&str>,
        opts: &[Opt],
    ) -> BunsenResult<Resolved> {
        let family = self.engine.family.name();
        let mut resolved = Resolved {
            profile: profile.to_string(),
            need: need.settled(),
            mode,
            family: self.engine.family,
            model: self.engine.model.clone(),
            emission: self.emission,
            chunk_ms: self.chunk_ms,
            flags: Vec::new(),
        };
        if let Some(language) = language {
            if self.engine.family == Family::Whisper {
                resolved.set("language", language);
            } else {
                log::info!(
                    "profile {profile} runs {family}, which finds the language itself: \
                     --language {language} not applied"
                );
            }
        }
        for opt in opts {
            let owner = opt.family.unwrap_or(self.engine.family);
            if !has_key(owner, &opt.key) {
                return Err(BunsenError::policy(format!(
                    "--opt {opt}: {} has no `{}`; it takes: {}",
                    owner.name(),
                    opt.key,
                    family_keys(owner)
                )));
            }
            if owner != self.engine.family {
                log::info!("profile {profile} runs {family}: --opt {opt} not applied");
            } else if opt.key == "model" {
                resolved.model = opt.value.clone();
            } else {
                resolved.set(&opt.key, &opt.value);
            }
        }
        resolved.argv()?;
        resolved.need.task = resolved.task();
        Ok(resolved)
    }
}

impl Resolved {
    /// The task the model runs: Whisper's `--task`, as its flags have it,
    /// else the `translate` preset in its model's box (`transcribe`
    /// without); another family transcribes.
    pub fn task(&self) -> Task {
        match self.family {
            Family::Whisper => self
                .flags
                .iter()
                .find(|(k, _)| k == "task")
                .and_then(|(_, v)| <Task as clap::ValueEnum>::from_str(v, false).ok())
                .unwrap_or_else(|| {
                    let translates = LookupExpr::parse(&self.model)
                        .is_ok_and(|e| e.selection.presets.iter().any(|p| p == "translate"));
                    if translates {
                        Task::Translate
                    } else {
                        Task::Transcribe
                    }
                }),
            Family::Parakeet | Family::Qwen3Asr | Family::Voxtral => Task::Transcribe,
        }
    }

    /// Sets the family flag `key`, in place when it is already set.
    fn set(
        &mut self,
        key: &str,
        value: &str,
    ) {
        match self.flags.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value.to_string(),
            None => self.flags.push((key.to_string(), value.to_string())),
        }
    }

    /// The family flags as its command takes them: `--key value`, or
    /// `--key` alone for a flag that takes no value and is `true`.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when such a flag
    /// is neither `true` nor `false`, or a key is not the family's.
    pub fn argv(&self) -> BunsenResult<Vec<String>> {
        let args = family_args(self.family);
        let mut argv = Vec::new();
        for (key, value) in &self.flags {
            let arg = args
                .iter()
                .find(|a| a.get_long() == Some(key.as_str()))
                .ok_or_else(|| {
                    BunsenError::policy(format!(
                        "profile {}: {} has no `--{key}`",
                        self.profile,
                        self.family.name()
                    ))
                })?;
            if arg.get_action().takes_values() {
                argv.extend([format!("--{key}"), value.clone()]);
            } else {
                match value.as_str() {
                    "true" => argv.push(format!("--{key}")),
                    "false" => {}
                    _ => {
                        return Err(BunsenError::policy(format!(
                            "profile {}: `--{key}` takes no value; `{value}` is neither \
                             `true` nor `false`",
                            self.profile
                        )));
                    }
                }
            }
        }
        Ok(argv)
    }

    /// The family flags, parsed by `F`, the family's own flags.
    ///
    /// # Errors
    /// As [`argv`](Self::argv);
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when `F` does
    /// not take a value.
    pub fn parse_flags<F: clap::Args>(&self) -> BunsenResult<F> {
        FlagsOnly::<F>::try_parse_from(self.argv()?)
            .map(|parsed| parsed.flags)
            .map_err(|e| {
                BunsenError::policy(format!(
                    "profile {}: {}",
                    self.profile,
                    e.render().to_string().trim()
                ))
            })
    }

    /// The `bunsen-tool asr family` command this runs, word by word.
    ///
    /// # Errors
    /// As [`argv`](Self::argv).
    pub fn command_line(&self) -> BunsenResult<Vec<String>> {
        let mut line: Vec<String> = [
            "bunsen-tool",
            "asr",
            "family",
            self.family.name(),
            self.mode.command(),
            "--model",
            &self.model,
            "--emission",
            emission_name(self.emission),
            "--chunk-ms",
        ]
        .map(String::from)
        .to_vec();
        line.push(self.chunk_ms.to_string());
        line.extend(self.argv()?);
        Ok(line)
    }

    /// What `--json` records of this: the profile, the need it served, and
    /// what it ran.
    pub fn stamp(&self) -> Stamp {
        Stamp {
            profile: self.profile.clone(),
            need: self.need,
            recipe: RecipeStamp {
                family: self.family.name().to_string(),
                model: self.model.clone(),
                emission: emission_name(self.emission).to_string(),
                chunk_ms: self.chunk_ms,
                flags: self.flags.iter().cloned().collect(),
            },
        }
    }
}

/// The name `--emission` takes for `emission`.
pub fn emission_name(emission: Emission) -> &'static str {
    match emission {
        Emission::Offline => "offline",
        Emission::Conservative => "conservative",
        Emission::Drafting => "drafting",
    }
}

/// `words` as one shell line, a word quoted where the shell would split it.
pub fn shell_line(words: &[String]) -> String {
    words
        .iter()
        .map(|w| {
            if !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./:=+,@".contains(c))
            {
                w.clone()
            } else {
                format!("'{}'", w.replace('\'', r"'\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// What chose a transcript's model, as `--json` records it.
#[derive(Clone, Debug, Serialize)]
pub struct Stamp {
    pub profile: String,

    /// The need served: its task is the one the model ran.
    pub need: Need,

    pub recipe: RecipeStamp,
}

/// A recipe, as `--json` records it.
#[derive(Clone, Debug, Serialize)]
pub struct RecipeStamp {
    pub family: String,
    pub model: String,
    pub emission: String,
    pub chunk_ms: usize,
    pub flags: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(s: &str) -> Opt {
        s.parse().unwrap()
    }

    fn resolve(
        name: &str,
        mode: Mode,
        language: Option<&str>,
        opts: &[&str],
    ) -> BunsenResult<Resolved> {
        let profile = lookup(name)?;
        let opts: Vec<Opt> = opts.iter().map(|s| opt(s)).collect();
        profile
            .recipe(mode)
            .resolve(&profile.name, profile.need, mode, language, &opts)
    }

    /// Every recipe parses as its family's flags; the names and aliases are
    /// each one profile's, and the default is one of them.
    #[test]
    fn test_profiles_parse() {
        let mut names = Vec::new();
        for profile in profiles() {
            names.push(profile.name.as_str());
            names.extend(profile.aliases.iter().map(String::as_str));
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = resolve(&profile.name, mode, None, &[]).unwrap();
                match resolved.family {
                    Family::Parakeet => drop(resolved.parse_flags::<ParakeetFlags>().unwrap()),
                    Family::Qwen3Asr => drop(resolved.parse_flags::<Qwen3AsrFlags>().unwrap()),
                    Family::Voxtral => drop(resolved.parse_flags::<VoxtralFlags>().unwrap()),
                    Family::Whisper => drop(resolved.parse_flags::<WhisperFlags>().unwrap()),
                }
            }
        }
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "{names:?}");
        assert_eq!(lookup(DEFAULT_PROFILE).unwrap().name, "en");
        assert!(lookup("klingon").is_err());
    }

    /// Every need reaches a cell in each mode; a profile is its own cell,
    /// and a missing cell falls through to the nearest.
    #[test]
    fn test_pick() {
        use clap::ValueEnum;
        for &speech in Speech::value_variants() {
            for &task in Task::value_variants() {
                for &quality in Quality::value_variants() {
                    for mode in [Mode::Offline, Mode::Live] {
                        let need = Need {
                            speech,
                            task,
                            quality,
                        };
                        let (cell, note) = pick(need, mode);
                        assert_eq!(cell.mode, mode);
                        assert_eq!(note.is_none(), cell.need == need);
                    }
                }
            }
        }
        for profile in profiles() {
            for mode in [Mode::Offline, Mode::Live] {
                let (cell, note) = pick(profile.need, mode);
                assert_eq!(note, None, "{} {mode:?}", profile.name);
                assert_eq!(cell.need, profile.need);
            }
        }

        let need = |speech, task, quality| Need {
            speech,
            task,
            quality,
        };
        // A better quality before a worse.
        let (cell, _) = pick(
            need(Speech::Multi, Task::Transcribe, Quality::Good),
            Mode::Live,
        );
        assert_eq!(cell.recipe.engine.model, "qwen/qwen3-asr-0.6b");
        // English translated to English is transcribed.
        let (cell, note) = pick(
            need(Speech::En, Task::Translate, Quality::Fast),
            Mode::Offline,
        );
        assert_eq!(cell.recipe.engine.model, "fermion/phonon-2");
        assert!(note.is_some());
    }

    #[test]
    fn test_opt_parse() {
        assert_eq!(
            opt("whisper.beam-size=1"),
            Opt {
                family: Some(Family::Whisper),
                key: "beam-size".into(),
                value: "1".into()
            }
        );
        assert_eq!(
            opt("--window-seconds=20"),
            Opt {
                family: None,
                key: "window-seconds".into(),
                value: "20".into()
            }
        );
        assert_eq!(opt("whisper.timestamps").value, "true");
        assert_eq!(opt("model=openai/v1.2=x").value, "openai/v1.2=x");
        assert_eq!(opt("qwen3-asr.model=a").to_string(), "qwen3-asr.model=a");
        assert!("klingon.beam-size=1".parse::<Opt>().is_err());
        assert!("=1".parse::<Opt>().is_err());
    }

    /// A family's `--opt` is applied to that family alone; one for no
    /// family must be the profile's family's; a key no family has fails
    /// even where it would not apply.
    #[test]
    fn test_opts() {
        // `translate-en` is Whisper: its options apply, a user's winning.
        let r = resolve(
            "translate-en",
            Mode::Offline,
            Some("de"),
            &[
                "whisper.beam-size=2",
                "timestamps",
                "qwen3-asr.window-seconds=20",
            ],
        )
        .unwrap();
        // The task and the beam are the model's (`[translate]{beam_size: 5}`);
        // the flags are what the command line added.
        assert_eq!(r.model, "openai/large-v3[translate]{beam_size: 5}");
        assert_eq!(
            r.argv().unwrap(),
            ["--language", "de", "--beam-size", "2", "--timestamps"]
        );
        let flags = r.parse_flags::<WhisperFlags>().unwrap();
        assert_eq!(flags.beam_size, Some(2));
        assert_eq!(flags.language.as_deref(), Some("de"));
        assert!(flags.timestamps);

        // `en` is Parakeet: Whisper's are skipped, as is the language.
        let r = resolve(
            "en",
            Mode::Live,
            Some("de"),
            &["whisper.beam-size=1", "window-seconds=20"],
        )
        .unwrap();
        assert_eq!(r.argv().unwrap(), ["--window-seconds", "20"]);
        assert_eq!(
            r.parse_flags::<ParakeetFlags>().unwrap().window_seconds,
            Some(20.0)
        );

        // A bare key must be the profile's family's.
        assert!(resolve("en", Mode::Offline, None, &["beam-size=1"]).is_err());
        // A family's key it does not have fails, applied or not.
        assert!(resolve("en", Mode::Offline, None, &["whisper.beam-sise=1"]).is_err());
        // A flag that takes no value is `true` or `false`.
        assert!(resolve("translate-en", Mode::Offline, None, &["timestamps=yes"]).is_err());
        let r = resolve("translate-en", Mode::Offline, None, &["timestamps=false"]).unwrap();
        assert!(!r.parse_flags::<WhisperFlags>().unwrap().timestamps);
        // A value the family rejects fails when parsed.
        let r = resolve("translate-en", Mode::Offline, None, &["task=sing"]).unwrap();
        assert!(r.parse_flags::<WhisperFlags>().is_err());

        // `model` replaces the model, for any family or the profile's.
        let r = resolve("multi", Mode::Offline, None, &["model=fermion/phonon-1"]).unwrap();
        assert_eq!(r.model, "fermion/phonon-1");
        let r = resolve("multi", Mode::Offline, None, &["whisper.model=openai/tiny"]).unwrap();
        assert_eq!(r.model, "qwen/qwen3-asr-1.7b");
    }

    #[test]
    fn test_command_line() {
        let r = resolve("translate-en", Mode::Live, None, &[]).unwrap();
        assert_eq!(
            shell_line(&r.command_line().unwrap()),
            "bunsen-tool asr family whisper live --model 'openai/small[translate]{beam_size: 1}' \
             --emission conservative --chunk-ms 250"
        );
        assert_eq!(
            shell_line(&["a b".into(), "it's".into(), "".into()]),
            r"'a b' 'it'\''s' ''"
        );
        let stamp = serde_json::to_value(r.stamp()).unwrap();
        assert_eq!(
            stamp["recipe"]["model"],
            "openai/small[translate]{beam_size: 1}"
        );
        assert_eq!(stamp["profile"], "translate-en");
        assert_eq!(
            stamp["need"],
            serde_json::json!({"speech": "multi", "task": "translate", "quality": "balanced"})
        );

        // An `--opt` that changes Whisper's task changes the need served.
        let r = resolve("translate-en", Mode::Live, None, &["task=transcribe"]).unwrap();
        assert_eq!(r.need.task, Task::Transcribe);
        assert_eq!(r.task(), Task::Transcribe);
    }

    /// The catalog's axes are the enums' values, each described; every
    /// recipe's model is a lookup expression its family's factory resolves,
    /// its presets and patch checked, with nothing fetched.
    #[test]
    fn test_catalog() {
        use bunsen::kits::speech::{
            parakeet::pretrained::default_parakeet_factory,
            qwen3_asr::pretrained::default_qwen3_asr_factory,
            voxtral::pretrained::default_voxtral_factory,
            whisper::pretrained::default_whisper_factory,
        };
        use clap::ValueEnum;

        let table = catalog();
        let members = |axis: &str| -> Vec<String> {
            let axis = table.axes.iter().find(|a| a.name == axis).unwrap();
            assert!(!axis.description.is_empty());
            axis.members
                .iter()
                .inspect(|m| assert!(!m.description.is_empty(), "{}", m.name))
                .map(|m| m.name.clone())
                .collect()
        };
        let names = |values: Vec<Option<clap::builder::PossibleValue>>| -> Vec<String> {
            values
                .into_iter()
                .map(|v| v.unwrap().get_name().to_string())
                .collect()
        };
        assert_eq!(
            members("speech"),
            names(
                Speech::value_variants()
                    .iter()
                    .map(|v| v.to_possible_value())
                    .collect()
            )
        );
        assert_eq!(
            members("task"),
            names(
                Task::value_variants()
                    .iter()
                    .map(|v| v.to_possible_value())
                    .collect()
            )
        );
        assert_eq!(
            members("quality"),
            names(
                Quality::value_variants()
                    .iter()
                    .map(|v| v.to_possible_value())
                    .collect()
            )
        );

        for cell in &table.points {
            let model = &cell.recipe.engine.model;
            let looked = match cell.recipe.engine.family {
                Family::Parakeet => default_parakeet_factory()
                    .and_then(|f| f.lookup(model))
                    .map(drop),
                Family::Qwen3Asr => default_qwen3_asr_factory()
                    .and_then(|f| f.lookup(model))
                    .map(drop),
                Family::Voxtral => default_voxtral_factory()
                    .and_then(|f| f.lookup(model))
                    .map(drop),
                Family::Whisper => default_whisper_factory()
                    .and_then(|f| f.lookup(model))
                    .map(drop),
            };
            looked.unwrap_or_else(|e| panic!("{} {model}: {e}", cell.need));
        }

        // English translated to English is transcribed, and says why.
        let (need, why) = Need {
            speech: Speech::En,
            task: Task::Translate,
            quality: Quality::Fast,
        }
        .settled_why();
        assert_eq!(need.task, Task::Transcribe);
        assert_eq!(why, ["English speech translated to English is transcribed"]);
    }
}
