//! `bunsen-asr transcribe`, `live` and `profiles`: speech to text by profile.
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
///
/// How the flags choose, in order (the README has them as a table):
///
/// 1. `--language` implies `--speech`: `en` is `en`, any other code `multi`.
///    `--speech en` with another language is an error.
/// 2. Without `--profile`, each of `--speech`, `--task` and `--quality` not
///    given is `default`'s, but `--task translate` makes the speech `multi`.
///    With it, they are the profile's.
/// 3. English speech translated to English is transcribed ([`Need::settled`]),
///    logged at `INFO`.
/// 4. Without `--profile`, the table's cell for that need runs. With it, the
///    profile's recipe runs, at `--quality`; `--speech` must be one it hears
///    (`multi` hears English), and a task not its own is set on its model
///    (Whisper's `--task`) or is an error.
///
/// The task, unless a translating profile or `--task translate` asks
/// otherwise, is `transcribe`; the need served, its task the one the model
/// runs, is logged and recorded by `--json`.
#[derive(clap::Args, Debug)]
pub struct ProfileArgs {
    /// The profile, by name or alias: `profiles` lists them, and what each
    /// runs. Without it, `--speech`, `--task`, `--quality` and `--language`
    /// choose the model, over `default`'s; with it, they may only adjust it.
    #[arg(long)]
    pub profile: Option<String>,

    /// What the speech is; `--language` implies it. Without `--profile`, in
    /// place of `default`'s (`multi` with `--task translate`); with it, only
    /// as the profile already hears it (`multi` hears English).
    #[arg(long, value_enum)]
    pub speech: Option<Speech>,

    /// What to make of the speech: `transcribe` unless the profile
    /// translates. Without `--profile`, in place of `default`'s; with it,
    /// set on a model that takes a task (Whisper's `--task`), and an error
    /// on one that does not. English speech is transcribed either way.
    #[arg(long, value_enum)]
    pub task: Option<Task>,

    /// How much compute to spend, in place of the profile's. A quality the
    /// table has no model for falls through to the nearest that has one,
    /// logged at `INFO`.
    #[arg(long, value_enum)]
    pub quality: Option<Quality>,

    /// The spoken language (`en`, `de`, ...): it implies `--speech` (`en`,
    /// or `multi` for any other), and is given to a model that can be told
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

/// What the command line asks of the table: a name for it, the need it is
/// for, the cell's need, and the `--opt` that sets the task on its model.
struct Asked {
    /// The profile's name, or the need's.
    name: String,

    /// The need, [settled](Need::settled).
    need: Need,

    /// The need whose cell runs: the profile's, at the need's quality; or
    /// the need itself.
    cell: Need,

    /// Whisper's `--task`, when the need's task is not the cell's.
    task: Option<Opt>,
}

impl ProfileArgs {
    /// `--speech`, or the speech `--language` implies: `en` for English,
    /// `multi` for any other.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for `--speech
    /// en` with a language that is not English.
    fn speech(&self) -> BunsenResult<Option<Speech>> {
        let implied = self.language.as_deref().map(|l| {
            if l.eq_ignore_ascii_case("en") {
                Speech::En
            } else {
                Speech::Multi
            }
        });
        match (self.speech, implied) {
            (Some(Speech::En), Some(Speech::Multi)) => Err(BunsenError::policy(format!(
                "--speech en, but --language {} is not English",
                self.language.as_deref().unwrap_or_default()
            ))),
            (Some(speech), _) => Ok(Some(speech)),
            (None, implied) => Ok(implied),
        }
    }

    /// The flag that set `speech`: `--speech`, or `--language`.
    fn speech_flag(&self) -> String {
        match (self.speech, &self.language) {
            (Some(speech), _) => format!("--speech {}", value_name(speech)),
            (None, Some(language)) => format!("--language {language}"),
            (None, None) => unreachable!("no flag set the speech"),
        }
    }

    /// What the flags ask of the table in `mode`, by the rules on
    /// [`ProfileArgs`].
    ///
    /// # Errors
    /// As [`lookup`] and [`speech`](Self::speech);
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when the profile
    /// does not hear the speech, or its model in `mode` cannot do the task.
    fn asked(
        &self,
        mode: Mode,
    ) -> BunsenResult<Asked> {
        let speech = self.speech()?;
        let settle = |asked: Need| {
            let need = asked.settled();
            if need != asked {
                log::info!("{asked}: English speech translated to English is transcribed");
            }
            need
        };
        let Some(name) = &self.profile else {
            let base = lookup(DEFAULT_PROFILE)?;
            let task = self.task.unwrap_or(base.need.task);
            let need = settle(Need {
                speech: speech.unwrap_or(match task {
                    Task::Translate => Speech::Multi,
                    Task::Transcribe => base.need.speech,
                }),
                task,
                quality: self.quality.unwrap_or(base.need.quality),
            });
            let named = self.speech.is_some()
                || self.task.is_some()
                || self.quality.is_some()
                || self.language.is_some();
            return Ok(Asked {
                name: if named {
                    need.to_string()
                } else {
                    base.name.to_string()
                },
                need,
                cell: need,
                task: None,
            });
        };
        let profile = lookup(name)?;
        if let Some(speech) = speech
            && speech != profile.need.speech
            && profile.need.speech != Speech::Multi
        {
            return Err(BunsenError::policy(format!(
                "profile {} hears {}; {} needs another profile, or none",
                profile.name,
                value_name(profile.need.speech),
                self.speech_flag()
            )));
        }
        let quality = self.quality.unwrap_or(profile.need.quality);
        let need = settle(Need {
            speech: speech.unwrap_or(profile.need.speech),
            task: self.task.unwrap_or(profile.need.task),
            quality,
        });
        let cell = Need {
            quality,
            ..profile.need
        };
        let task = if need.task != cell.task {
            let family = pick(cell, mode).0.recipe.family;
            if family != Family::Whisper {
                let task = value_name(need.task);
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
                value: value_name(need.task),
            })
        } else {
            None
        };
        Ok(Asked {
            name: profile.name.to_string(),
            need,
            cell,
            task,
        })
    }

    /// The `mode` recipe of the profile, or of the table's cell for the
    /// need the axis flags ask for, with the command line over it (an
    /// `--opt` last), logged at `INFO` as the need it serves and the family
    /// command it runs.
    ///
    /// # Errors
    /// As [`asked`](Self::asked) and
    /// [`Recipe::resolve`](crate::profile::Recipe::resolve).
    pub fn resolve(
        &self,
        mode: Mode,
    ) -> BunsenResult<Resolved> {
        let asked = self.asked(mode)?;
        let (cell, note) = pick(asked.cell, mode);
        if let Some(note) = note {
            log::info!("{note}");
        }
        let opts: Vec<Opt> = asked
            .task
            .into_iter()
            .chain(self.opts.iter().cloned())
            .collect();
        let mut resolved = cell.recipe.resolve(
            &asked.name,
            asked.need,
            mode,
            self.language.as_deref(),
            &opts,
        )?;
        if let Some(preset) = self.preset {
            resolved.preset = preset;
        }
        if let Some(chunk_ms) = self.chunk_ms {
            resolved.chunk_ms = chunk_ms;
        }
        log::info!(
            "profile {} ({}): {}",
            resolved.profile,
            resolved.need,
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
                let resolved = profile.resolve(mode)?;
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

    /// What `line` runs in `mode`: the profile, the need served, the model,
    /// and the task the model runs; the family's flags parse.
    fn ran(
        line: &[&str],
        mode: Mode,
    ) -> BunsenResult<(String, String, String, Task)> {
        let r = resolve(line, mode)?;
        with_flags!(r, _flags => ());
        assert_eq!(r.need.task, r.task(), "{line:?}");
        Ok((
            r.profile.clone(),
            r.need.to_string(),
            r.model.clone(),
            r.task(),
        ))
    }

    /// Each rule on [`ProfileArgs`], line by line: the profile recorded, the
    /// need served, the model, and the task it runs; or the error.
    #[test]
    fn test_rules() {
        use Mode::{
            Live,
            Offline,
        };
        use Task::{
            Transcribe,
            Translate,
        };
        const PHONON: &str = "fermion/phonon-2";
        const QWEN: &str = "qwen/qwen3-asr-1.7b";
        const QWEN_LIVE: &str = "qwen/qwen3-asr-0.6b";
        const LARGE: &str = "openai/large-v3";
        const SMALL: &str = "openai/small";
        type Ran<'a> = Result<(&'a str, &'a str, &'a str, Task), &'a str>;
        let rows: &[(&[&str], Mode, Ran)] = &[
            // Nothing given: `default`, which transcribes.
            (
                &[],
                Offline,
                Ok(("en", "en/transcribe/balanced", PHONON, Transcribe)),
            ),
            (
                &[],
                Live,
                Ok(("en", "en/transcribe/balanced", PHONON, Transcribe)),
            ),
            // 1. `--language` implies `--speech`.
            (
                &["--language", "de"],
                Offline,
                Ok((
                    "multi/transcribe/balanced",
                    "multi/transcribe/balanced",
                    QWEN,
                    Transcribe,
                )),
            ),
            (
                &["--language", "en"],
                Offline,
                Ok((
                    "en/transcribe/balanced",
                    "en/transcribe/balanced",
                    PHONON,
                    Transcribe,
                )),
            ),
            (
                &["--language", "de", "--task", "translate"],
                Offline,
                Ok((
                    "multi/translate/balanced",
                    "multi/translate/balanced",
                    LARGE,
                    Translate,
                )),
            ),
            (
                &["--speech", "multi", "--language", "en"],
                Offline,
                Ok((
                    "multi/transcribe/balanced",
                    "multi/transcribe/balanced",
                    QWEN,
                    Transcribe,
                )),
            ),
            (
                &["--speech", "en", "--language", "de"],
                Offline,
                Err("is not English"),
            ),
            // 2. Without `--profile`, `default`'s need under the flags given;
            // `--task translate` makes the speech `multi`.
            (
                &["--task", "transcribe"],
                Offline,
                Ok((
                    "en/transcribe/balanced",
                    "en/transcribe/balanced",
                    PHONON,
                    Transcribe,
                )),
            ),
            (
                &["--task", "translate"],
                Offline,
                Ok((
                    "multi/translate/balanced",
                    "multi/translate/balanced",
                    LARGE,
                    Translate,
                )),
            ),
            (
                &["--task", "translate"],
                Live,
                Ok((
                    "multi/translate/balanced",
                    "multi/translate/balanced",
                    SMALL,
                    Translate,
                )),
            ),
            (
                &["--speech", "multi"],
                Offline,
                Ok((
                    "multi/transcribe/balanced",
                    "multi/transcribe/balanced",
                    QWEN,
                    Transcribe,
                )),
            ),
            (
                &["--speech", "multi", "--quality", "fast"],
                Offline,
                Ok((
                    "multi/transcribe/fast",
                    "multi/transcribe/fast",
                    QWEN_LIVE,
                    Transcribe,
                )),
            ),
            // No `fast` live cell: `balanced` stands in.
            (
                &["--speech", "multi", "--quality", "fast"],
                Live,
                Ok((
                    "multi/transcribe/fast",
                    "multi/transcribe/fast",
                    QWEN_LIVE,
                    Transcribe,
                )),
            ),
            // 3. English speech translated to English is transcribed.
            (
                &["--speech", "en", "--task", "translate"],
                Offline,
                Ok((
                    "en/transcribe/balanced",
                    "en/transcribe/balanced",
                    PHONON,
                    Transcribe,
                )),
            ),
            (
                &["--language", "en", "--task", "translate"],
                Live,
                Ok((
                    "en/transcribe/balanced",
                    "en/transcribe/balanced",
                    PHONON,
                    Transcribe,
                )),
            ),
            (
                &["--profile", "en", "--task", "translate"],
                Offline,
                Ok(("en", "en/transcribe/balanced", PHONON, Transcribe)),
            ),
            (
                &["--profile", "translate-en", "--speech", "en"],
                Offline,
                Ok(("translate-en", "en/transcribe/balanced", LARGE, Transcribe)),
            ),
            (
                &["--profile", "translate-en", "--language", "en"],
                Live,
                Ok(("translate-en", "en/transcribe/balanced", SMALL, Transcribe)),
            ),
            // 4. With `--profile`, its recipe runs, adjusted: `--quality` along
            // its row,
            (
                &["--profile", "multi", "--quality", "fast"],
                Offline,
                Ok(("multi", "multi/transcribe/fast", QWEN_LIVE, Transcribe)),
            ),
            // `--speech` it hears,
            (
                &["--profile", "multi", "--speech", "en"],
                Live,
                Ok(("multi", "en/transcribe/balanced", QWEN_LIVE, Transcribe)),
            ),
            (
                &["--profile", "multi", "--language", "de"],
                Offline,
                Ok(("multi", "multi/transcribe/balanced", QWEN, Transcribe)),
            ),
            (
                &["--profile", "translate-en", "--language", "de"],
                Offline,
                Ok(("translate-en", "multi/translate/balanced", LARGE, Translate)),
            ),
            (
                &["--profile", "en", "--speech", "multi"],
                Live,
                Err("profile en hears en"),
            ),
            (
                &["--profile", "en", "--language", "de"],
                Offline,
                Err("--language de needs another profile"),
            ),
            // and a task set on Whisper, an `--opt` still winning.
            (
                &["--profile", "translate-en", "--task", "transcribe"],
                Live,
                Ok((
                    "translate-en",
                    "multi/transcribe/balanced",
                    SMALL,
                    Transcribe,
                )),
            ),
            (
                &[
                    "--profile",
                    "translate-en",
                    "--task",
                    "transcribe",
                    "--opt",
                    "task=translate",
                ],
                Live,
                Ok(("translate-en", "multi/translate/balanced", SMALL, Translate)),
            ),
            (
                &["--profile", "multi", "--task", "transcribe"],
                Live,
                Ok(("multi", "multi/transcribe/balanced", QWEN_LIVE, Transcribe)),
            ),
            (
                &["--profile", "multi", "--task", "translate"],
                Live,
                Err("cannot translate"),
            ),
        ];
        for (line, mode, expect) in rows {
            let got = ran(line, *mode);
            match (expect, got) {
                (Ok((profile, need, model, task)), Ok(got)) => assert_eq!(
                    (got.0.as_str(), got.1.as_str(), got.2.as_str(), got.3),
                    (*profile, *need, *model, *task),
                    "{line:?} {mode:?}"
                ),
                (Err(message), Err(e)) => {
                    assert!(e.to_string().contains(message), "{line:?}: {e}")
                }
                (expect, got) => panic!("{line:?} {mode:?}: expected {expect:?}, got {got:?}"),
            }
        }
    }

    /// Over every combination of `--profile`, `--speech`, `--task` and
    /// `--language`, in each mode: the task is `transcribe` unless a
    /// translating profile or `--task translate` asks otherwise, English
    /// speech is transcribed, a language not English is never given to an
    /// English model, and an error says what to change.
    #[test]
    fn test_rules_hold_everywhere() {
        let mut ran_lines = 0;
        for profile in [None, Some("en"), Some("multi"), Some("translate-en")] {
            for speech in [None, Some("en"), Some("multi")] {
                for task in [None, Some("transcribe"), Some("translate")] {
                    for language in [None, Some("en"), Some("de")] {
                        for mode in [Mode::Offline, Mode::Live] {
                            let mut line = Vec::new();
                            for (flag, value) in [
                                ("--profile", profile),
                                ("--speech", speech),
                                ("--task", task),
                                ("--language", language),
                            ] {
                                if let Some(value) = value {
                                    line.extend([flag, value]);
                                }
                            }
                            let r = match resolve(&line, mode) {
                                Ok(r) => r,
                                Err(e) => {
                                    let e = e.to_string();
                                    assert!(
                                        e.contains("needs another profile")
                                            || e.contains("is not English"),
                                        "{line:?}: {e}"
                                    );
                                    continue;
                                }
                            };
                            ran_lines += 1;
                            // The speech, from the flags alone: `--speech`,
                            // what `--language` implies, the profile's, or
                            // `default`'s (`multi` to translate).
                            let english = match (speech, language, profile) {
                                (Some(speech), _, _) => speech == "en",
                                (None, Some(language), _) => language == "en",
                                (None, None, Some(profile)) => profile == "en",
                                (None, None, None) => task != Some("translate"),
                            };
                            let asked = match (task, profile) {
                                (Some(task), _) => task == "translate",
                                (None, profile) => profile == Some("translate-en"),
                            };
                            let want = if asked && !english {
                                Task::Translate
                            } else {
                                Task::Transcribe
                            };
                            assert_eq!(r.task(), want, "{line:?} {mode:?}");
                            assert_eq!(r.need.task, want, "{line:?} {mode:?}");
                            if language == Some("de") {
                                assert_eq!(r.need.speech, Speech::Multi, "{line:?}");
                                assert_ne!(r.family, Family::Parakeet, "{line:?}");
                            }
                        }
                    }
                }
            }
        }
        // 4 x 3 x 3 x 3 = 108 lines in each mode, which refuses the same 29:
        // `--speech en --language de` (12), `en` asked for other speech (12),
        // and `multi` asked to translate other speech (5).
        assert_eq!(ran_lines, 2 * (108 - 29));
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
