//! Profiles: speech to text by what it is for (`en`, `multi`,
//! `translate-en`), not by which model does it.
//!
//! A profile is two recipes, one for files (`transcribe`) and one for the
//! microphone (`live`); a recipe is a family, a model, a preset, and the
//! family's own flags, as the family's command line would give them. The
//! table is this binary's: when a better model arrives, a row changes, and
//! the command lines that name the profile do not.
//!
//! The flags are kept as the family command's long flags and parsed by the
//! family's own `clap` struct, so the profiles hold no second copy of any
//! family's options, and a recipe is always a command `bunsen-asr family`
//! runs.
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

use crate::{
    family::Family,
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    stream::Preset,
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

/// One way to run a family: what `bunsen-asr family <family> transcribe|live`
/// would be given.
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

/// A named, semantic choice of model: a recipe for files, one for the
/// microphone.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,

    /// What `transcribe` runs.
    pub offline: Recipe,

    /// What `live` runs.
    pub live: Recipe,
}

impl Profile {
    /// The recipe `mode` runs.
    pub fn recipe(
        &self,
        mode: Mode,
    ) -> &Recipe {
        match mode {
            Mode::Offline => &self.offline,
            Mode::Live => &self.live,
        }
    }
}

/// The profile `--profile` names when it is not given: an alias, so what it
/// means can change too.
pub const DEFAULT_PROFILE: &str = "default";

/// The profiles.
pub const PROFILES: &[Profile] = &[
    Profile {
        name: "en",
        aliases: &[DEFAULT_PROFILE],
        description: "English: NVIDIA's Parakeet TDT, as FermionResearch's Phonon-2; each word timed",
        offline: Recipe {
            family: Family::Parakeet,
            model: "fermion/phonon-2",
            preset: Preset::Offline,
            chunk_ms: 1000,
            flags: &[],
        },
        live: Recipe {
            family: Family::Parakeet,
            model: "fermion/phonon-2",
            preset: Preset::Conservative,
            chunk_ms: 250,
            flags: &[],
        },
    },
    Profile {
        name: "multi",
        aliases: &[],
        description: "Any of 30 languages, found from the speech: Qwen's Qwen3-ASR, as FermionResearch's Phonon-1",
        offline: Recipe {
            family: Family::Qwen3Asr,
            model: "fermion/phonon-1-big",
            preset: Preset::Offline,
            chunk_ms: 1000,
            flags: &[],
        },
        live: Recipe {
            family: Family::Qwen3Asr,
            model: "fermion/phonon-1",
            preset: Preset::Conservative,
            chunk_ms: 250,
            flags: &[],
        },
    },
    Profile {
        name: "translate-en",
        aliases: &[],
        description: "Any of Whisper's languages, translated to English text: OpenAI's Whisper \
                      (large-v3-turbo was not trained to translate)",
        offline: Recipe {
            family: Family::Whisper,
            model: "openai/large-v3",
            preset: Preset::Offline,
            chunk_ms: 1000,
            flags: &[("task", "translate"), ("beam-size", "5")],
        },
        live: Recipe {
            family: Family::Whisper,
            model: "openai/small",
            preset: Preset::Conservative,
            chunk_ms: 250,
            flags: &[("task", "translate"), ("beam-size", "1")],
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
    /// The profile's name, not the alias it was named by.
    pub profile: &'static str,
    pub mode: Mode,
    pub family: Family,
    pub model: String,
    pub preset: Preset,
    pub chunk_ms: usize,

    /// The family's own flags, by long name, each once, in order.
    pub flags: Vec<(String, String)>,
}

impl Recipe {
    /// This recipe of `profile`, with `language` and `opts` over it.
    ///
    /// `language` is Whisper's `--language`; a family that finds the
    /// language itself is logged, at `INFO`, as not taking it. An `opt` for
    /// another family is logged as not applied; one for this family, or for
    /// none, replaces the recipe's value.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when an `opt`
    /// names an option its family does not have (or, for none, this
    /// family), or a flag that takes no value is given one but `true` or
    /// `false`: they come from the command line.
    pub fn resolve(
        &self,
        profile: &'static str,
        mode: Mode,
        language: Option<&str>,
        opts: &[Opt],
    ) -> BunsenResult<Resolved> {
        let family = self.family.name();
        let mut resolved = Resolved {
            profile,
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
        Ok(resolved)
    }
}

impl Resolved {
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

    /// The `bunsen-asr family` command this runs, word by word.
    ///
    /// # Errors
    /// As [`argv`](Self::argv).
    pub fn command_line(&self) -> BunsenResult<Vec<String>> {
        let mut line: Vec<String> = [
            "bunsen-asr",
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

    /// What `--json` records of this: the profile, and what it ran.
    pub fn stamp(&self) -> Stamp {
        Stamp {
            profile: self.profile.to_string(),
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
            .resolve(profile.name, mode, language, &opts)
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
        assert_eq!(r.model, "fermion/phonon-1-big");
    }

    #[test]
    fn test_command_line() {
        let r = resolve("translate-en", Mode::Live, None, &[]).unwrap();
        assert_eq!(
            shell_line(&r.command_line().unwrap()),
            "bunsen-asr family whisper live --model openai/small --preset conservative \
             --chunk-ms 250 --task translate --beam-size 1"
        );
        assert_eq!(
            shell_line(&["a b".into(), "it's".into(), "".into()]),
            r"'a b' 'it'\''s' ''"
        );
        let stamp = serde_json::to_value(r.stamp()).unwrap();
        assert_eq!(stamp["recipe"]["flags"]["task"], "translate");
        assert_eq!(stamp["profile"], "translate-en");
    }
}
