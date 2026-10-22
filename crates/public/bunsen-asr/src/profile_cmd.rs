//! `bunsen-asr transcribe`, `bunsen-asr live` and `bunsen-asr profiles`:
//! speech to text by profile.
use std::io::Write;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::speech::silero_vad::pretrained::default_silero_factory,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

#[cfg(feature = "live")]
use crate::live_cmd::LiveIo;
use crate::{
    family::{
        Family,
        Kit,
        fetch_with,
    },
    parakeet::ParakeetFlags,
    profile::{
        DEFAULT_PROFILE,
        Mode,
        Need,
        Opt,
        PROFILES,
        Profile,
        Quality,
        Resolved,
        Speech,
        Task,
        lookup,
        pick,
        shell_line,
    },
    qwen3_asr::Qwen3AsrFlags,
    stream::{
        Preset,
        VAD_MODEL,
    },
    transcribe_cmd::TranscribeIo,
    whisper::WhisperFlags,
};

/// The profile, and what the command line changes of it.
#[derive(clap::Args, Debug)]
pub struct ProfileArgs {
    /// The profile, by name or alias: `profiles` lists them, and what each
    /// runs. Without it, `--speech`, `--task` and `--quality` choose the
    /// model, over `default`'s; with it, they may only adjust it.
    #[arg(long)]
    pub profile: Option<String>,

    /// What the speech is. Without `--profile`, in place of `default`'s
    /// (`multi` with `--task translate`); with it, only as the profile
    /// already hears it (`multi` hears English).
    #[arg(long, value_enum)]
    pub speech: Option<Speech>,

    /// What to make of the speech. Without `--profile`, in place of
    /// `default`'s; with it, set on a model that takes a task (Whisper's
    /// `--task`), and an error on one that does not.
    #[arg(long, value_enum)]
    pub task: Option<Task>,

    /// How much compute to spend, in place of the profile's. A quality the
    /// table has no model for falls through to the nearest that has one,
    /// logged at `INFO`.
    #[arg(long, value_enum)]
    pub quality: Option<Quality>,

    /// The spoken language (`en`, `de`, ...), for a model that can be told
    /// it (Whisper); one that finds the language itself goes on doing so.
    #[arg(long)]
    pub language: Option<String>,

    /// One of the family's own options, as `bunsen-asr family <family>`
    /// takes it, without the `--` (`KEY` alone is `KEY=true`). With
    /// `FAMILY.`, it applies only while the profile runs that family, so a
    /// script keeps working when the profile moves; without, the profile's
    /// family must have it. `model=NAME` replaces the profile's model.
    #[arg(long = "opt", value_name = "[FAMILY.]KEY[=VALUE]")]
    pub opts: Vec<Opt>,

    /// When to decode, in place of the profile's.
    #[arg(long, value_enum)]
    pub preset: Option<Preset>,

    /// Milliseconds of audio per push, in place of the profile's.
    #[arg(long)]
    pub chunk_ms: Option<usize>,
}

impl ProfileArgs {
    /// The profile's name and need, `--quality` over it, and an `--opt`
    /// setting `--task` on its model, when it is named; otherwise the need
    /// the axis flags ask for, over `default`'s, named for itself.
    ///
    /// # Errors
    /// As [`lookup`];
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when the profile
    /// does not hear `--speech`, or its model in `mode` cannot do `--task`.
    fn need(
        &self,
        mode: Mode,
    ) -> BunsenResult<(String, Need, Option<Opt>)> {
        let Some(name) = &self.profile else {
            let base = lookup(DEFAULT_PROFILE)?.need;
            let task = self.task.unwrap_or(base.task);
            let need = Need {
                // English speech, translated to English, is only transcribed.
                speech: self.speech.unwrap_or(match task {
                    Task::Translate => Speech::Multi,
                    Task::Transcribe => base.speech,
                }),
                task,
                quality: self.quality.unwrap_or(base.quality),
            };
            let name = if self.speech.is_none() && self.task.is_none() && self.quality.is_none() {
                lookup(DEFAULT_PROFILE)?.name.to_string()
            } else {
                need.to_string()
            };
            return Ok((name, need, None));
        };
        let profile = lookup(name)?;
        let need = Need {
            quality: self.quality.unwrap_or(profile.need.quality),
            ..profile.need
        };
        if let Some(speech) = self.speech
            && speech != need.speech
            && need.speech != Speech::Multi
        {
            return Err(BunsenError::policy(format!(
                "profile {} hears {}; --speech {} needs another profile, or none",
                profile.name,
                value_name(need.speech),
                value_name(speech)
            )));
        }
        let task = match self.task {
            Some(task) if task != need.task => {
                let family = pick(need, mode).0.recipe.family;
                if family != Family::Whisper {
                    let task = value_name(task);
                    return Err(BunsenError::policy(format!(
                        "profile {} runs {}, which cannot {task}; --task {task} needs another \
                         profile, or none",
                        profile.name,
                        family.name()
                    )));
                }
                Some(Opt {
                    family: Some(family),
                    key: "task".to_string(),
                    value: value_name(task),
                })
            }
            _ => None,
        };
        Ok((profile.name.to_string(), need, task))
    }

    /// The `mode` recipe of the profile, or of the table's cell for the
    /// need the axis flags ask for, with the command line over it (an
    /// `--opt` last), logged at `INFO` as the family command it runs.
    ///
    /// # Errors
    /// As [`need`](Self::need) and
    /// [`Recipe::resolve`](crate::profile::Recipe::resolve).
    pub fn resolve(
        &self,
        mode: Mode,
    ) -> BunsenResult<Resolved> {
        let (name, need, task) = self.need(mode)?;
        let (cell, note) = pick(need, mode);
        if let Some(note) = note {
            log::info!("{note}");
        }
        let opts: Vec<Opt> = task.into_iter().chain(self.opts.iter().cloned()).collect();
        let mut resolved = cell
            .recipe
            .resolve(&name, mode, self.language.as_deref(), &opts)?;
        if let Some(preset) = self.preset {
            resolved.preset = preset;
        }
        if let Some(chunk_ms) = self.chunk_ms {
            resolved.chunk_ms = chunk_ms;
        }
        log::info!(
            "profile {}: {}",
            resolved.profile,
            shell_line(&resolved.command_line()?)
        );
        Ok(resolved)
    }
}

/// The name a `--speech`, `--task` or `--quality` value is given by.
fn value_name(value: impl clap::ValueEnum) -> String {
    value
        .to_possible_value()
        .expect("no variant is skipped")
        .get_name()
        .to_string()
}

/// `$body` with `$flags` the resolved family's own flags, parsed.
macro_rules! with_flags {
    ($resolved:expr, $flags:ident => $body:expr) => {
        match $resolved.family {
            Family::Parakeet => {
                let $flags = $resolved.parse_flags::<ParakeetFlags>()?;
                $body
            }
            Family::Qwen3Asr => {
                let $flags = $resolved.parse_flags::<Qwen3AsrFlags>()?;
                $body
            }
            Family::Whisper => {
                let $flags = $resolved.parse_flags::<WhisperFlags>()?;
                $body
            }
        }
    };
}

/// Transcribes audio files with a profile's model for files.
///
/// The profile picks the family, the model, when to decode and the family's
/// own flags; `--opt` changes them, `profiles show` prints them, and `-v`
/// logs them as the `bunsen-asr family` command they are. `--json` records
/// the profile and its recipe with each transcript.
#[derive(clap::Args, Debug)]
pub struct ProfileTranscribeCmd {
    #[clap(flatten)]
    pub profile: ProfileArgs,

    #[clap(flatten)]
    pub io: TranscribeIo,
}

impl ProfileTranscribeCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.io.init_logging()?;
        let resolved = self.profile.resolve(Mode::Offline)?;
        let stamp = resolved.stamp();
        with_flags!(resolved, flags => self.io.run(
            resolved.family,
            &resolved.model,
            &flags,
            resolved.preset,
            resolved.chunk_ms,
            Some(&stamp),
        ))
    }
}

/// Transcribes the microphone, as it speaks, with a profile's model for
/// the microphone.
///
/// The profile picks the family, the model, when to decode and the family's
/// own flags; `--opt` changes them, `profiles show` prints them, and `-v`
/// logs them as the `bunsen-asr family` command they are.
#[cfg(feature = "live")]
#[derive(clap::Args, Debug)]
pub struct ProfileLiveCmd {
    #[clap(flatten)]
    pub profile: ProfileArgs,

    #[clap(flatten)]
    pub io: LiveIo,
}

#[cfg(feature = "live")]
impl ProfileLiveCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.io.init_logging()?;
        let resolved = self.profile.resolve(Mode::Live)?;
        with_flags!(resolved, flags => self.io.run(
            resolved.family,
            &resolved.model,
            &flags,
            resolved.preset,
            resolved.chunk_ms,
        ))
    }
}

/// The profiles' commands; `list` when none is named.
#[derive(clap::Args, Debug)]
pub struct ProfilesCmd {
    #[command(subcommand)]
    action: Option<ProfilesAction>,
}

#[derive(clap::Subcommand, Debug)]
enum ProfilesAction {
    /// List the profiles: what each is for, and the model each runs for
    /// files and for the microphone.
    List,

    /// Show profiles: each recipe, as the `bunsen-asr family` command it
    /// runs.
    Show(ShowCmd),

    /// Bring every file the named profiles run into the weights cache: both
    /// recipes' models, and the VAD where a recipe finds speech regions.
    Fetch(FetchCmd),
}

impl ProfilesCmd {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            None | Some(ProfilesAction::List) => list(&mut std::io::stdout().lock()),
            Some(ProfilesAction::Show(show)) => show.run(),
            Some(ProfilesAction::Fetch(fetch)) => fetch.run(),
        }
    }
}

/// A profile's name, and its aliases when it has any.
fn title(profile: &Profile) -> String {
    match profile.aliases {
        [] => profile.name.to_string(),
        aliases => format!("{} ({})", profile.name, aliases.join(", ")),
    }
}

/// Writes each profile, and what each of its recipes runs.
fn list(out: &mut impl Write) -> BunsenResult<()> {
    let w = |e| sys_at("write to", "stdout")(e);
    for (i, profile) in PROFILES.iter().enumerate() {
        if i > 0 {
            writeln!(out).map_err(w)?;
        }
        writeln!(out, "{}", title(profile)).map_err(w)?;
        writeln!(out, "  {}", profile.description).map_err(w)?;
        for mode in [Mode::Offline, Mode::Live] {
            let recipe = profile.recipe(mode);
            writeln!(
                out,
                "  {:<11} {} {}",
                format!("{}:", mode.command()),
                recipe.family.name(),
                recipe.model
            )
            .map_err(w)?;
        }
    }
    Ok(())
}

/// Shows profiles: each recipe, as the `bunsen-asr family` command it runs.
#[derive(clap::Args, Debug)]
struct ShowCmd {
    /// Profiles, by name or alias; every profile when none is named.
    names: Vec<String>,
}

impl ShowCmd {
    fn run(&self) -> BunsenResult<()> {
        let profiles: Vec<&Profile> = if self.names.is_empty() {
            PROFILES.iter().collect()
        } else {
            self.names
                .iter()
                .map(|n| lookup(n))
                .collect::<BunsenResult<_>>()?
        };
        let mut out = std::io::stdout().lock();
        let w = |e| sys_at("write to", "stdout")(e);
        for (i, profile) in profiles.into_iter().enumerate() {
            if i > 0 {
                writeln!(out).map_err(w)?;
            }
            writeln!(out, "{}", title(profile)).map_err(w)?;
            writeln!(out, "  {}", profile.description).map_err(w)?;
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = profile
                    .recipe(mode)
                    .resolve(profile.name, mode, None, &[])?;
                writeln!(
                    out,
                    "  {}: {}",
                    mode.command(),
                    shell_line(&resolved.command_line()?)
                )
                .map_err(w)?;
            }
        }
        Ok(())
    }
}

/// Brings every file the named profiles run into the weights cache, and
/// prints where each one is: a deployment that runs offline fills its cache
/// this way ahead of time.
#[derive(clap::Args, Debug)]
struct FetchCmd {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Re-hash each pinned file once it is local, the cached ones included.
    #[arg(long)]
    verify: bool,

    /// Profiles, by name or alias (`profiles` lists them).
    #[arg(required = true)]
    names: Vec<String>,
}

impl FetchCmd {
    fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let mut models: Vec<(Family, &str)> = Vec::new();
        let mut vad = false;
        for name in &self.names {
            let profile = lookup(name)?;
            for recipe in [Mode::Offline, Mode::Live].map(|m| profile.recipe(m)) {
                if !models.contains(&(recipe.family, recipe.model)) {
                    models.push((recipe.family, recipe.model));
                }
                vad |= recipe.preset != Preset::Offline;
            }
        }
        let mut out = std::io::stdout().lock();
        for (family, model) in models {
            Kit::Asr(family).fetch(&[model.to_string()], &cache, self.verify, &mut out)?;
        }
        if vad {
            fetch_with(
                default_silero_factory()?,
                &[VAD_MODEL.to_string()],
                &cache,
                self.verify,
                &mut out,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser, Debug)]
    #[command(no_binary_name = true)]
    struct Line {
        #[command(flatten)]
        args: ProfileArgs,
    }

    fn resolve(
        line: &[&str],
        mode: Mode,
    ) -> BunsenResult<Resolved> {
        Line::try_parse_from(line).unwrap().args.resolve(mode)
    }

    /// Without `--profile`, the axis flags pick from the table, over the
    /// default profile's need. With it, the profile's model is kept: they
    /// adjust it, or fail.
    #[test]
    fn test_need_flags() {
        let r = resolve(&[], Mode::Offline).unwrap();
        assert_eq!(
            (r.profile.as_str(), r.model.as_str()),
            ("en", "fermion/phonon-2")
        );

        // Translating is of speech that is not English.
        let r = resolve(&["--task", "translate"], Mode::Live).unwrap();
        assert_eq!(r.profile, "multi/translate/balanced");
        assert_eq!(r.model, "openai/small");

        let r = resolve(&["--speech", "multi", "--quality", "fast"], Mode::Offline).unwrap();
        assert_eq!(r.profile, "multi/transcribe/fast");
        assert_eq!(r.model, "qwen/qwen3-asr-0.6b");

        // No `fast` live cell: `balanced` stands in.
        let r = resolve(&["--speech", "multi", "--quality", "fast"], Mode::Live).unwrap();
        assert_eq!(r.model, "qwen/qwen3-asr-0.6b");
        assert_eq!(r.preset, Preset::Conservative);

        // `--quality` moves along the profile's row.
        let r = resolve(&["--profile", "multi", "--quality", "fast"], Mode::Offline).unwrap();
        assert_eq!(
            (r.profile.as_str(), r.model.as_str()),
            ("multi", "qwen/qwen3-asr-0.6b")
        );

        // `--task` sets Whisper's task; a user's `--opt` still wins.
        let r = resolve(
            &["--profile", "translate-en", "--task", "transcribe"],
            Mode::Live,
        )
        .unwrap();
        assert_eq!(r.model, "openai/small");
        assert_eq!(
            r.parse_flags::<WhisperFlags>().unwrap().task,
            crate::whisper::Task::Transcribe
        );
        let r = resolve(
            &[
                "--profile",
                "translate-en",
                "--task",
                "transcribe",
                "--opt",
                "task=translate",
            ],
            Mode::Live,
        )
        .unwrap();
        assert_eq!(
            r.parse_flags::<WhisperFlags>().unwrap().task,
            crate::whisper::Task::Translate
        );
        // A task the profile already does is no change.
        assert!(resolve(&["--profile", "multi", "--task", "transcribe"], Mode::Live).is_ok());
        // Qwen3-ASR cannot translate.
        assert!(resolve(&["--profile", "multi", "--task", "translate"], Mode::Live).is_err());

        // `multi` hears English; `en` hears nothing else.
        assert!(resolve(&["--profile", "multi", "--speech", "en"], Mode::Live).is_ok());
        assert!(resolve(&["--profile", "en", "--speech", "multi"], Mode::Live).is_err());
    }

    /// Every profile is listed, with both recipes.
    #[test]
    fn test_list() {
        let mut out = Vec::new();
        list(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("en (default)\n"), "{text}");
        for profile in PROFILES {
            for mode in [Mode::Offline, Mode::Live] {
                assert!(text.contains(profile.recipe(mode).model), "{text}");
            }
        }
        assert!(
            text.contains("  live:       qwen3-asr qwen/qwen3-asr-0.6b\n"),
            "{text}"
        );
    }
}
