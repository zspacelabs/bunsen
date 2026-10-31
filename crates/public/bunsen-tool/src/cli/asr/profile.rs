//! Profiles: speech to text by what it is for, not by which model does it.

use std::{
    collections::BTreeMap,
    fmt,
    str::FromStr,
};

use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use clap::{
    Arg,
    Command,
    Parser,
};
use serde::Serialize;

use crate::cli::asr::{
    family::Family,
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    stream::Preset,
    voxtral::VoxtralFlags,
    whisper::WhisperFlags,
};

/// Which of a profile's recipes a command runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// One way to run a family: what `bunsen-tool asr family <family>
/// transcribe|live` would be given.
///
/// The flags are kept as the family command's long flags and parsed by the
/// family's own `clap` struct, so no second copy of any family's options
/// lives here, and a recipe is always a command `bunsen-tool asr family` runs.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    pub family: Family,

    /// The model, as `--model` names it.
    pub model: &'static str,

    pub preset: Preset,

    /// Milliseconds of audio per push.
    pub chunk_ms: usize,

    /// The family's own flags, by long name, in order; a flag that takes no
    /// value is `"true"`.
    pub flags: &'static [(&'static str, &'static str)],
}

/// What the speech is: the table's language axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Speech {
    /// English.
    En,

    /// Any language a model finds from the speech; English among them.
    Multi,
}

/// What to make of the speech.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    /// Text in the language spoken.
    Transcribe,

    /// English text.
    Translate,
}

/// How much compute to spend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Fast,
    Balanced,
    Good,
}

impl Quality {
    /// This, then the others nearest first, a better one before a worse.
    fn nearest(self) -> [Quality; 3] {
        match self {
            Quality::Fast => [Quality::Fast, Quality::Balanced, Quality::Good],
            Quality::Balanced => [Quality::Balanced, Quality::Good, Quality::Fast],
            Quality::Good => [Quality::Good, Quality::Balanced, Quality::Fast],
        }
    }
}

/// A point in the table: what the speech is, what to make of it, how much
/// to spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
    /// This need as it is served: English speech translated to English is
    /// transcribed.
    ///
    /// Translating English to English is out of every model's training, and
    /// Whisper's output for it is its transcript, so the table has no such
    /// cell; a need for one is the `transcribe` need.
    pub fn settled(self) -> Need {
        match (self.speech, self.task) {
            (Speech::En, Task::Translate) => Need {
                task: Task::Transcribe,
                ..self
            },
            _ => self,
        }
    }

    /// The points that can stand in for this settled one, nearest first,
    /// this one first: another quality before other speech, as a
    /// multilingual model also hears English. The task is never changed.
    fn candidates(self) -> Vec<Need> {
        let speeches: &[Speech] = match self.speech {
            Speech::En => &[Speech::En, Speech::Multi],
            Speech::Multi => &[Speech::Multi],
        };
        let mut out = Vec::new();
        for &speech in speeches {
            for quality in self.quality.nearest() {
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

/// One cell of the table: what runs for a need, in a mode.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub need: Need,
    pub mode: Mode,
    pub recipe: Recipe,
}

/// A recipe for files: whole windows, a second of audio per push.
const fn offline(
    family: Family,
    model: &'static str,
    flags: &'static [(&'static str, &'static str)],
) -> Recipe {
    Recipe {
        family,
        model,
        preset: Preset::Offline,
        chunk_ms: 1000,
        flags,
    }
}

/// A recipe for the microphone: speech regions, a quarter second per push.
const fn live(
    family: Family,
    model: &'static str,
    flags: &'static [(&'static str, &'static str)],
) -> Recipe {
    Recipe {
        family,
        model,
        preset: Preset::Conservative,
        chunk_ms: 250,
        flags,
    }
}

const fn cell(
    speech: Speech,
    task: Task,
    quality: Quality,
    mode: Mode,
    recipe: Recipe,
) -> Cell {
    Cell {
        need: Need {
            speech,
            task,
            quality,
        },
        mode,
        recipe,
    }
}

/// The table: a cell where a model was seen to do the job; a need without
/// one falls through to the nearest that has one ([`pick`]).
///
/// Every speech and task reaches a cell: `multi` `transcribe` and `multi`
/// `translate` each have a `balanced` cell in each mode.
pub const TABLE: &[Cell] = {
    use Family::*;
    use Mode::{
        Live,
        Offline,
    };
    use Quality::*;
    use Speech::*;
    use Task::*;
    &[
        // Parakeet TDT, as Phonon-2: each word timed.
        cell(
            En,
            Transcribe,
            Balanced,
            Offline,
            offline(Parakeet, "fermion/phonon-2", &[]),
        ),
        cell(
            En,
            Transcribe,
            Balanced,
            Live,
            live(Parakeet, "fermion/phonon-2", &[]),
        ),
        // Qwen3-ASR: 30 languages. Phonon-1 is an English fine-tune, and
        // writes other speech as broken English. 1.7B is about 2.5x 0.6B's
        // time, too slow to be seen keeping up live.
        cell(
            Multi,
            Transcribe,
            Fast,
            Offline,
            offline(Qwen3Asr, "qwen/qwen3-asr-0.6b", &[]),
        ),
        cell(
            Multi,
            Transcribe,
            Balanced,
            Offline,
            offline(Qwen3Asr, "qwen/qwen3-asr-1.7b", &[]),
        ),
        cell(
            Multi,
            Transcribe,
            Balanced,
            Live,
            live(Qwen3Asr, "qwen/qwen3-asr-0.6b", &[]),
        ),
        // Whisper alone translates (large-v3-turbo was not trained to).
        cell(
            Multi,
            Translate,
            Balanced,
            Offline,
            offline(
                Whisper,
                "openai/large-v3",
                &[("task", "translate"), ("beam-size", "5")],
            ),
        ),
        cell(
            Multi,
            Translate,
            Balanced,
            Live,
            live(
                Whisper,
                "openai/small",
                &[("task", "translate"), ("beam-size", "1")],
            ),
        ),
    ]
};

/// The cell for `need`, [settled](Need::settled), in `mode`, and, when it
/// is not `need`'s own, a note of what stood in.
pub fn pick(
    need: Need,
    mode: Mode,
) -> (&'static Cell, Option<String>) {
    for candidate in need.settled().candidates() {
        if let Some(cell) = TABLE.iter().find(|c| c.need == candidate && c.mode == mode) {
            let note = (candidate != need).then(|| {
                format!(
                    "no {} recipe for {need}: {candidate} stands in",
                    mode.command()
                )
            });
            return (cell, note);
        }
    }
    unreachable!("every speech and task reaches a balanced cell; see TABLE")
}

/// A named, semantic choice of model: a point in the [`TABLE`].
///
/// A profile names what the speech is for (`en`, `multi`, `translate-en`),
/// with one recipe for files (`transcribe`) and one for the microphone
/// (`live`). The table is this binary's: when a better model arrives, a row
/// changes, and the command lines that name the profile do not.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
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
            .resolve(self.name, self.need, mode, None, &[])
    }
}

/// The profile `--profile` names when it is not given: an alias, so what it
/// means can change too.
pub const DEFAULT_PROFILE: &str = "default";

/// The profiles: names for the points of the table people ask for.
pub const PROFILES: &[Profile] = &[
    Profile {
        name: "en",
        aliases: &[DEFAULT_PROFILE],
        description: "English: NVIDIA's Parakeet TDT, as FermionResearch's Phonon-2; each word timed",
        need: Need {
            speech: Speech::En,
            task: Task::Transcribe,
            quality: Quality::Balanced,
        },
    },
    Profile {
        name: "multi",
        aliases: &[],
        description: "Any of 30 languages, found from the speech: Qwen's Qwen3-ASR (1.7B; 0.6B live)",
        need: Need {
            speech: Speech::Multi,
            task: Task::Transcribe,
            quality: Quality::Balanced,
        },
    },
    Profile {
        name: "translate-en",
        aliases: &[],
        description: "Any of Whisper's languages, translated to English text: OpenAI's Whisper \
                      (large-v3-turbo was not trained to translate)",
        need: Need {
            speech: Speech::Multi,
            task: Task::Translate,
            quality: Quality::Balanced,
        },
    },
];

/// The profile `name` names, by its name or an alias.
///
/// # Errors
/// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when no profile has
/// the name: it comes from the command line.
pub fn lookup(name: &str) -> BunsenResult<&'static Profile> {
    PROFILES
        .iter()
        .find(|p| p.name == name || p.aliases.contains(&name))
        .ok_or_else(|| {
            let names: Vec<&str> = PROFILES.iter().map(|p| p.name).collect();
            BunsenError::policy(format!(
                "no profile `{name}`; the profiles: {}",
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
    pub preset: Preset,
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
        let family = self.family.name();
        let mut resolved = Resolved {
            profile: profile.to_string(),
            need: need.settled(),
            mode,
            family: self.family,
            model: self.model.to_string(),
            preset: self.preset,
            chunk_ms: self.chunk_ms,
            flags: self
                .flags
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        };
        if let Some(language) = language {
            if self.family == Family::Whisper {
                resolved.set("language", language);
            } else {
                log::info!(
                    "profile {profile} runs {family}, which finds the language itself: \
                     --language {language} not applied"
                );
            }
        }
        for opt in opts {
            let owner = opt.family.unwrap_or(self.family);
            if !has_key(owner, &opt.key) {
                return Err(BunsenError::policy(format!(
                    "--opt {opt}: {} has no `{}`; it takes: {}",
                    owner.name(),
                    opt.key,
                    family_keys(owner)
                )));
            }
            if owner != self.family {
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
    /// The task the model runs: Whisper's `--task`, as its flags have it
    /// (`transcribe` when they do not set it); another family transcribes.
    pub fn task(&self) -> Task {
        match self.family {
            Family::Whisper => self
                .flags
                .iter()
                .find(|(k, _)| k == "task")
                .and_then(|(_, v)| <Task as clap::ValueEnum>::from_str(v, false).ok())
                .unwrap_or(Task::Transcribe),
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
            "--preset",
            preset_name(self.preset),
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
                preset: preset_name(self.preset).to_string(),
                chunk_ms: self.chunk_ms,
                flags: self.flags.iter().cloned().collect(),
            },
        }
    }
}

/// The name `--preset` takes for `preset`.
pub fn preset_name(preset: Preset) -> &'static str {
    match preset {
        Preset::Offline => "offline",
        Preset::Conservative => "conservative",
        Preset::Responsive => "responsive",
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
    pub preset: String,
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
            .resolve(profile.name, profile.need, mode, language, &opts)
    }

    /// Every recipe parses as its family's flags; the names and aliases are
    /// each one profile's, and the default is one of them.
    #[test]
    fn test_profiles_parse() {
        let mut names = Vec::new();
        for profile in PROFILES {
            names.push(profile.name);
            names.extend(profile.aliases);
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = resolve(profile.name, mode, None, &[]).unwrap();
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
        for profile in PROFILES {
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
        assert_eq!(cell.recipe.model, "qwen/qwen3-asr-0.6b");
        // English translated to English is transcribed.
        let (cell, note) = pick(
            need(Speech::En, Task::Translate, Quality::Fast),
            Mode::Offline,
        );
        assert_eq!(cell.recipe.model, "fermion/phonon-2");
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
        assert_eq!(
            r.argv().unwrap(),
            [
                "--task",
                "translate",
                "--beam-size",
                "2",
                "--language",
                "de",
                "--timestamps"
            ]
        );
        let flags = r.parse_flags::<WhisperFlags>().unwrap();
        assert_eq!(flags.beam_size, 2);
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
            20.0
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
            "bunsen-tool asr family whisper live --model openai/small --preset conservative \
             --chunk-ms 250 --task translate --beam-size 1"
        );
        assert_eq!(
            shell_line(&["a b".into(), "it's".into(), "".into()]),
            r"'a b' 'it'\''s' ''"
        );
        let stamp = serde_json::to_value(r.stamp()).unwrap();
        assert_eq!(stamp["recipe"]["flags"]["task"], "translate");
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
}
