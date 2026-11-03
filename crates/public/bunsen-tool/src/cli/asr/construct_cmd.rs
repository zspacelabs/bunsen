//! `asr transcribe`, `live` and `presets`: speech to text by preset.
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
use crate::cli::asr::live_cmd::LiveIo;
use crate::cli::asr::{
    construct::{
        AsrConstruct,
        AsrPreset,
        DEFAULT_PRESET,
        Mode,
        Need,
        Quality,
        Speech,
        Task,
        construct_config,
        lookup,
        presets,
        shell_line,
    },
    family::{
        Family,
        Kit,
        fetch_with,
    },
    parakeet::ParakeetFlags,
    qwen3_asr::Qwen3AsrFlags,
    stream::{
        Emission,
        VadArgs,
    },
    transcribe_cmd::TranscribeIo,
    voxtral::VoxtralFlags,
    whisper::WhisperFlags,
};

/// The preset, and what the command line changes of it.
///
/// How the flags choose, in order (the README has them as a table):
///
/// 1. `--language` implies `--speech`, by the construct's `language` table:
///    `en` is `en`, any other code `multi`. A `--speech` that does not hear it
///    (`fallback.speech`: `multi` hears English) is an error.
/// 2. Without `--preset`, each of `--speech`, `--task` and `--quality` not
///    given is `default`'s, but `--task translate` makes the speech `multi`.
///    With it, they are the preset's.
/// 3. English speech translated to English is transcribed ([`Need::settled`]),
///    logged at `INFO`.
/// 4. Without `--preset`, the table's point for that need runs. With it, the
///    preset's recipe runs, at `--quality`; `--speech` must be one it hears
///    (`multi` hears English), and a task not its own is set on its model
///    (Whisper's `[translate]` preset) or is an error.
///
/// The task, unless a translating preset or `--task translate` asks
/// otherwise, is `transcribe`; the need served, its task the one the model
/// runs, is logged and recorded by `--json`.
#[derive(clap::Args, Debug)]
pub struct AsrConstructArgs {
    /// The preset, by name or alias: `presets` lists them, and what each
    /// runs. Without it, `--speech`, `--task`, `--quality` and `--language`
    /// choose the model, over `default`'s; with it, they may only adjust it.
    #[arg(long = "preset", value_name = "PRESET")]
    pub preset: Option<String>,

    /// What the speech is; `--language` implies it. Without `--preset`, in
    /// place of `default`'s (`multi` with `--task translate`); with it, only
    /// as the preset already hears it (`multi` hears English).
    #[arg(long, value_enum)]
    pub speech: Option<Speech>,

    /// What to make of the speech: `transcribe` unless the preset
    /// translates. Without `--preset`, in place of `default`'s; with it,
    /// set on a model that takes a task (Whisper's `[translate]`), and an error
    /// on one that does not. English speech is transcribed either way.
    #[arg(long, value_enum)]
    pub task: Option<Task>,

    /// How much compute to spend, in place of the preset's. A quality the
    /// table has no model for falls through to the nearest that has one,
    /// logged at `INFO`.
    #[arg(long, value_enum)]
    pub quality: Option<Quality>,

    /// The spoken language (`en`, `de`, ...): it implies `--speech` (`en`,
    /// or `multi` for any other), and is given to a model that can be told
    /// it (Whisper); one that finds the language itself goes on doing so.
    #[arg(long)]
    pub language: Option<String>,

    /// The model, as a lookup expression (`'openai/small[translate]{beam_size:
    /// 1}'`), in place of the preset's. The family that lists it runs, the
    /// preset's when it does; a name no family lists (`hf:`, a path) runs
    /// in the preset's.
    #[arg(long, value_name = "EXPR")]
    pub model: Option<String>,

    #[clap(flatten)]
    pub vad: VadArgs,

    /// When to decode, in place of the preset's.
    #[arg(long, value_enum)]
    pub emission: Option<Emission>,

    /// Milliseconds of audio per push, in place of the preset's.
    #[arg(long)]
    pub chunk_ms: Option<usize>,
}

/// What the command line asks of the table: a name for it, the need it is
/// for, the point's need, and the task to set on its model.
struct Asked {
    /// The preset's name, or the need's.
    name: String,

    /// The need, [settled](Need::settled).
    need: Need,

    /// The need whose point runs: the preset's, at the need's quality; or
    /// the need itself.
    point: Need,

    /// The task to set on the model (Whisper's), when the need's is not
    /// the point's.
    task: Option<Task>,
}

impl AsrConstructArgs {
    /// `--speech`, or the speech `--language` implies, by the construct's
    /// [`language`](crate::cli::asr::construct::AsrConstructConfig::language)
    /// table (`en` for English, `multi` for any other).
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a `--speech`
    /// that does not hear the speech `--language` implies
    /// ([`Fallback::hears`](crate::cli::asr::construct::Fallback::hears)):
    /// `--speech en --language de`.
    fn speech(&self) -> BunsenResult<Option<Speech>> {
        let config = construct_config();
        let implied = self.language.as_deref().map(|l| config.language.speech(l));
        match (self.speech, implied) {
            (Some(given), Some(implied)) if !config.fallback.hears(given, implied) => {
                Err(BunsenError::policy(format!(
                    "--speech {}, but --language {} is {} speech, which {} does not hear",
                    value_name(given),
                    self.language.as_deref().unwrap_or_default(),
                    value_name(implied),
                    value_name(given)
                )))
            }
            (Some(speech), _) => Ok(Some(speech)),
            (None, implied) => Ok(implied),
        }
    }

    /// The family that runs where the table picked `picked`: without
    /// `--model`, `picked`; with it, the family that lists the model's name,
    /// `picked` when it does, or `picked` when none does (an `hf:` repo, a
    /// checkpoint path). A switch is logged at `INFO`.
    ///
    /// # Errors
    /// As [`LookupExpr::parse`](bunsen::kits::support::construct::LookupExpr::parse)
    /// on `--model`; [`Policy`](bunsen::errors::BunsenErrorKind::Policy)
    /// when more than one other family lists it.
    fn family(
        &self,
        picked: Family,
    ) -> BunsenResult<Family> {
        let Some(model) = &self.model else {
            return Ok(picked);
        };
        let name = bunsen::kits::support::construct::LookupExpr::parse(model)?.name;
        if Kit::Asr(picked).lists(&name)? {
            return Ok(picked);
        }
        let mut others = Vec::new();
        for family in Family::ALL {
            if family != picked && Kit::Asr(family).lists(&name)? {
                others.push(family);
            }
        }
        match others.as_slice() {
            [] => Ok(picked),
            [family] => {
                log::info!(
                    "--model {model}: a {} model, so {} runs in place of {}",
                    family.name(),
                    family.name(),
                    picked.name()
                );
                Ok(*family)
            }
            many => {
                let names: Vec<&str> = many.iter().map(|f| f.name()).collect();
                Err(BunsenError::policy(format!(
                    "--model {model}: {} each list it; `asr family <family>` names one",
                    names.join(" and ")
                )))
            }
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
    /// [`AsrConstructArgs`].
    ///
    /// # Errors
    /// As [`lookup`] and [`speech`](Self::speech);
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) when the preset
    /// does not hear the speech, or its model in `mode` cannot do the task.
    fn asked(
        &self,
        mode: Mode,
    ) -> BunsenResult<Asked> {
        let speech = self.speech()?;
        let settle = |asked: Need| {
            let (need, reasons) = asked.settled_why();
            for reason in reasons {
                log::info!("{asked}: {reason}");
            }
            need
        };
        let Some(name) = &self.preset else {
            let base = lookup(DEFAULT_PRESET)?;
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
                point: need,
                task: None,
            });
        };
        let preset = lookup(name)?;
        if let Some(speech) = speech
            && !construct_config()
                .fallback
                .hears(preset.need.speech, speech)
        {
            return Err(BunsenError::policy(format!(
                "preset {} hears {}; {} needs another preset, or none",
                preset.name,
                value_name(preset.need.speech),
                self.speech_flag()
            )));
        }
        let quality = self.quality.unwrap_or(preset.need.quality);
        let need = settle(Need {
            speech: speech.unwrap_or(preset.need.speech),
            task: self.task.unwrap_or(preset.need.task),
            quality,
        });
        let point = Need {
            quality,
            ..preset.need
        };
        let task = if need.task != point.task {
            let family =
                self.family(construct_config().point(point, mode).0.recipe.engine.family)?;
            let does = construct_config()
                .engine(family)
                .tasks
                .as_ref()
                .is_some_and(|tasks| tasks.does(need.task));
            if !does {
                let task = value_name(need.task);
                return Err(BunsenError::policy(format!(
                    "preset {} runs {}, which cannot {task}; --task {task} needs another \
                     preset, or none",
                    preset.name,
                    family.name()
                )));
            }
            Some(need.task)
        } else {
            None
        };
        Ok(Asked {
            name: preset.name.to_string(),
            need,
            point,
            task,
        })
    }

    /// The `mode` recipe of the preset, or of the table's point for the
    /// need the axis flags ask for, with the command line over it (`--model`
    /// first, then the task and `--language` set on it), logged at `INFO` as
    /// the need it serves and the family command it runs.
    ///
    /// # Errors
    /// As [`asked`](Self::asked) and
    /// [`AsrRecipe::resolve`](crate::cli::asr::construct::AsrRecipe::resolve).
    pub fn resolve(
        &self,
        mode: Mode,
    ) -> BunsenResult<AsrConstruct> {
        let asked = self.asked(mode)?;
        let (point, note) = construct_config().point(asked.point, mode);
        if let Some(note) = note {
            log::info!("{note}");
        }
        let mut recipe = point.recipe.clone();
        if let Some(model) = &self.model {
            recipe.engine.family = self.family(recipe.engine.family)?;
            recipe.engine.model = model.clone();
        }
        let mut resolved = recipe.resolve(
            &asked.name,
            asked.need,
            mode,
            asked.task,
            self.language.as_deref(),
        )?;
        if let Some(emission) = self.emission {
            resolved.emission = emission;
        }
        resolved.vad = self.vad.slot(resolved.emission)?;
        if let Some(chunk_ms) = self.chunk_ms {
            resolved.chunk_ms = chunk_ms;
        }
        log::info!(
            "preset {} ({}): {}",
            resolved.preset,
            resolved.need,
            shell_line(&resolved.command_line())
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

/// `$body` with `$flags` the resolved family's own flags: their defaults, as
/// a preset sets no paths.
macro_rules! with_flags {
    ($resolved:expr, $flags:ident => $body:expr) => {
        match $resolved.family {
            Family::Parakeet => {
                let $flags = ParakeetFlags::default();
                $body
            }
            Family::Qwen3Asr => {
                let $flags = Qwen3AsrFlags::default();
                $body
            }
            Family::Voxtral => {
                let $flags = VoxtralFlags::default();
                $body
            }
            Family::Whisper => {
                let $flags = WhisperFlags::default();
                $body
            }
        }
    };
}

/// Transcribes audio files with a preset's model for files.
///
/// The preset picks the family, the model and when to decode; `--model`,
/// `--emission` and `--vad` change them, `presets show` prints them, and `-v`
/// logs them as the `bunsen-tool asr family` command they are. `--json` records
/// the preset and its recipe with each transcript.
#[derive(clap::Args, Debug)]
pub struct AsrTranscribeCmd {
    #[clap(flatten)]
    pub preset: AsrConstructArgs,

    #[clap(flatten)]
    pub io: TranscribeIo,
}

impl AsrTranscribeCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.io.init_logging()?;
        let resolved = self.preset.resolve(Mode::Offline)?;
        let stamp = resolved.stamp();
        with_flags!(resolved, flags => self.io.run(
            resolved.family,
            &resolved.model,
            &flags,
            &resolved.streaming(),
            Some(&stamp),
        ))
    }
}

/// Transcribes the microphone, as it speaks, with a preset's model for
/// the microphone.
///
/// The preset picks the family, the model and when to decode; `--model`,
/// `--emission` and `--vad` change them, `presets show` prints them, and `-v`
/// logs them as the `bunsen-tool asr family` command they are.
#[cfg(feature = "live")]
#[derive(clap::Args, Debug)]
pub struct AsrLiveCmd {
    #[clap(flatten)]
    pub preset: AsrConstructArgs,

    #[clap(flatten)]
    pub io: LiveIo,
}

#[cfg(feature = "live")]
impl AsrLiveCmd {
    pub fn run(&self) -> BunsenResult<()> {
        self.io.init_logging()?;
        let resolved = self.preset.resolve(Mode::Live)?;
        with_flags!(resolved, flags => self.io.run(
            resolved.family,
            &resolved.model,
            &flags,
            &resolved.streaming(),
        ))
    }
}

/// The presets' commands; `list` when none is named.
#[derive(clap::Args, Debug)]
pub struct AsrPresetsCmd {
    #[command(subcommand)]
    action: Option<AsrPresetsAction>,
}

#[derive(clap::Subcommand, Debug)]
enum AsrPresetsAction {
    /// List the presets: what each is for, and the model each runs for
    /// files and for the microphone.
    List,

    /// Show presets: each recipe, as the `bunsen-tool asr family` command it
    /// runs.
    Show(ShowCmd),

    /// Bring every file the named presets run into the weights cache: both
    /// recipes' models, and the VAD where a recipe finds speech regions.
    Fetch(FetchCmd),
}

impl AsrPresetsCmd {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            None | Some(AsrPresetsAction::List) => list(&mut std::io::stdout().lock()),
            Some(AsrPresetsAction::Show(show)) => show.run(),
            Some(AsrPresetsAction::Fetch(fetch)) => fetch.run(),
        }
    }
}

/// A preset's name, and its aliases when it has any.
fn title(preset: &AsrPreset) -> String {
    match preset.aliases.as_slice() {
        [] => preset.name.clone(),
        aliases => format!("{} ({})", preset.name, aliases.join(", ")),
    }
}

/// Writes each preset, and what each of its recipes runs; then the axes
/// they are points of.
fn list(out: &mut impl Write) -> BunsenResult<()> {
    let w = |e| sys_at("write to", "stdout")(e);
    for (i, preset) in presets().iter().enumerate() {
        if i > 0 {
            writeln!(out).map_err(w)?;
        }
        writeln!(out, "{}", title(preset)).map_err(w)?;
        writeln!(out, "  {}", preset.description).map_err(w)?;
        for mode in [Mode::Offline, Mode::Live] {
            let recipe = preset.recipe(mode);
            writeln!(
                out,
                "  {:<11} {} {}",
                format!("{}:", mode.command()),
                recipe.engine.family.name(),
                recipe.engine.model
            )
            .map_err(w)?;
        }
    }
    // The axes the presets are points of, and what each value means.
    writeln!(out).map_err(w)?;
    writeln!(out, "The axes (`--speech`, `--task`, `--quality`):").map_err(w)?;
    for axis in &construct_config().axes {
        writeln!(out, "  {}: {}", axis.name, axis.description).map_err(w)?;
        for member in &axis.members {
            writeln!(out, "    {:<11} {}", member.name, member.description).map_err(w)?;
        }
    }
    Ok(())
}

/// Shows presets: each recipe, as the `bunsen-tool asr family` command it
/// runs.
#[derive(clap::Args, Debug)]
struct ShowCmd {
    /// Presets, by name or alias; every preset when none is named.
    names: Vec<String>,
}

impl ShowCmd {
    fn run(&self) -> BunsenResult<()> {
        let presets: Vec<&AsrPreset> = if self.names.is_empty() {
            presets().iter().collect()
        } else {
            self.names
                .iter()
                .map(|n| lookup(n))
                .collect::<BunsenResult<_>>()?
        };
        let mut out = std::io::stdout().lock();
        let w = |e| sys_at("write to", "stdout")(e);
        for (i, preset) in presets.into_iter().enumerate() {
            if i > 0 {
                writeln!(out).map_err(w)?;
            }
            writeln!(out, "{}", title(preset)).map_err(w)?;
            writeln!(out, "  {}", preset.description).map_err(w)?;
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = preset.resolve(mode)?;
                writeln!(
                    out,
                    "  {}: {}",
                    mode.command(),
                    shell_line(&resolved.command_line())
                )
                .map_err(w)?;
            }
        }
        Ok(())
    }
}

/// Brings every file the named presets run into the weights cache, and
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

    /// Presets, by name or alias (`presets` lists them).
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
            let preset = lookup(name)?;
            for recipe in [Mode::Offline, Mode::Live].map(|m| preset.recipe(m)) {
                let model = (recipe.engine.family, recipe.engine.model.as_str());
                if !models.contains(&model) {
                    models.push(model);
                }
                vad |= recipe.emission != Emission::Offline;
            }
        }
        let mut out = std::io::stdout().lock();
        for (family, model) in models {
            Kit::Asr(family).fetch(&[model.to_string()], &cache, self.verify, &mut out)?;
        }
        if vad {
            fetch_with(
                default_silero_factory()?,
                &[construct_config().vad.clone()],
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
        args: AsrConstructArgs,
    }

    fn resolve(
        line: &[&str],
        mode: Mode,
    ) -> BunsenResult<AsrConstruct> {
        Line::try_parse_from(line).unwrap().args.resolve(mode)
    }

    /// What `line` runs in `mode`: the preset, the need served, the model,
    /// and the task the model runs; the family's flags parse.
    fn ran(
        line: &[&str],
        mode: Mode,
    ) -> BunsenResult<(String, String, String, Task)> {
        let r = resolve(line, mode)?;
        with_flags!(r, _flags => ());
        assert_eq!(r.need.task, r.task(), "{line:?}");
        Ok((
            r.preset.clone(),
            r.need.to_string(),
            r.model.clone(),
            r.task(),
        ))
    }

    /// The slot flags: `--model` replaces the engine's model, in the family
    /// that lists it; `--vad` the VAD's, under an emission that finds
    /// speech regions, and an error under one that does not.
    #[test]
    fn test_slots() {
        let r = resolve(
            &["--preset", "multi", "--model", "fermion/phonon-1"],
            Mode::Offline,
        )
        .unwrap();
        assert_eq!(
            (r.family, r.model.as_str()),
            (Family::Qwen3Asr, "fermion/phonon-1")
        );
        assert_eq!(r.vad, None);

        // A model another family lists runs in that family: Whisper's, under
        // `en`, which runs Parakeet; its expression's box and patch aside,
        // case aside.
        let r = resolve(&["--model", "OpenAI/Small{beam_size: 2}"], Mode::Offline).unwrap();
        assert_eq!(r.family, Family::Whisper);
        assert_eq!(r.model, "OpenAI/Small{beam_size: 2}");
        assert_eq!(r.task(), Task::Transcribe);
        // A task the preset's family cannot do, on a model that can.
        let r = resolve(
            &[
                "--preset",
                "multi",
                "--task",
                "translate",
                "--model",
                "openai/small",
            ],
            Mode::Live,
        )
        .unwrap();
        assert_eq!(
            (r.family, r.model.as_str(), r.need.task),
            (Family::Whisper, "openai/small[translate]", Task::Translate)
        );
        assert!(resolve(&["--preset", "multi", "--task", "translate"], Mode::Live).is_err());
        // A name no family lists runs in the preset's family.
        let r = resolve(&["--model", "hf:org/repo"], Mode::Offline).unwrap();
        assert_eq!(r.family, Family::Parakeet);
        let r = resolve(&["--model", "well-known:openai/tiny"], Mode::Offline).unwrap();
        assert_eq!(r.family, Family::Whisper);

        // Live finds speech regions: the table's VAD, or `--vad`'s.
        let table = construct_config().vad.as_str();
        assert_eq!(
            resolve(&[], Mode::Live).unwrap().vad.as_deref(),
            Some(table)
        );
        let vad = "silero/vad-16k[silero]";
        let r = resolve(&["--vad", vad], Mode::Live).unwrap();
        assert_eq!(r.vad.as_deref(), Some(vad));
        assert_eq!(r.stamp().recipe.vad.as_deref(), Some(vad));
        assert!(r.command_line().ends_with(&["--vad".into(), vad.into()]));

        // `--vad` with no VAD to fill is an error, whichever emission
        // removed it; an emission that adds one takes it.
        let err = resolve(&["--vad", vad], Mode::Offline).unwrap_err();
        assert!(
            err.to_string()
                .contains("`--vad` has no effect with `--emission offline`"),
            "{err}"
        );
        assert!(resolve(&["--vad", vad, "--emission", "offline"], Mode::Live).is_err());
        let r = resolve(&["--vad", vad, "--emission", "conservative"], Mode::Offline).unwrap();
        assert_eq!(r.vad.as_deref(), Some(vad));
        assert_eq!(
            resolve(&["--emission", "offline"], Mode::Live).unwrap().vad,
            None
        );
    }

    /// Each rule on [`AsrConstructArgs`], line by line: the preset recorded,
    /// the need served, the model, and the task it runs; or the error.
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
        const LARGE: &str = "openai/large-v3[translate]{beam_size: 5}";
        const SMALL: &str = "openai/small[translate]{beam_size: 1}";
        const LARGE_DE: &str = "openai/large-v3[translate]{beam_size: 5, language: de}";
        const LARGE_TRANSCRIBE: &str = "openai/large-v3{beam_size: 5}";
        const SMALL_TRANSCRIBE: &str = "openai/small{beam_size: 1}";
        const SMALL_EN: &str = "openai/small{beam_size: 1, language: en}";
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
                    LARGE_DE,
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
                Err("is multi speech, which en does not hear"),
            ),
            // 2. Without `--preset`, `default`'s need under the flags given;
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
            // No `fast` live point: `balanced` stands in.
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
                &["--preset", "en", "--task", "translate"],
                Offline,
                Ok(("en", "en/transcribe/balanced", PHONON, Transcribe)),
            ),
            (
                &["--preset", "translate-en", "--speech", "en"],
                Offline,
                Ok((
                    "translate-en",
                    "en/transcribe/balanced",
                    LARGE_TRANSCRIBE,
                    Transcribe,
                )),
            ),
            (
                &["--preset", "translate-en", "--language", "en"],
                Live,
                Ok((
                    "translate-en",
                    "en/transcribe/balanced",
                    SMALL_EN,
                    Transcribe,
                )),
            ),
            // 4. With `--preset`, its recipe runs, adjusted: `--quality` along
            // its row,
            (
                &["--preset", "multi", "--quality", "fast"],
                Offline,
                Ok(("multi", "multi/transcribe/fast", QWEN_LIVE, Transcribe)),
            ),
            // `--speech` it hears,
            (
                &["--preset", "multi", "--speech", "en"],
                Live,
                Ok(("multi", "en/transcribe/balanced", QWEN_LIVE, Transcribe)),
            ),
            (
                &["--preset", "multi", "--language", "de"],
                Offline,
                Ok(("multi", "multi/transcribe/balanced", QWEN, Transcribe)),
            ),
            (
                &["--preset", "translate-en", "--language", "de"],
                Offline,
                Ok((
                    "translate-en",
                    "multi/translate/balanced",
                    LARGE_DE,
                    Translate,
                )),
            ),
            (
                &["--preset", "en", "--speech", "multi"],
                Live,
                Err("preset en hears en"),
            ),
            (
                &["--preset", "en", "--language", "de"],
                Offline,
                Err("--language de needs another preset"),
            ),
            // and a task set on Whisper's model, or by the model given.
            (
                &["--preset", "translate-en", "--task", "transcribe"],
                Live,
                Ok((
                    "translate-en",
                    "multi/transcribe/balanced",
                    SMALL_TRANSCRIBE,
                    Transcribe,
                )),
            ),
            (
                &[
                    "--preset",
                    "translate-en",
                    "--model",
                    "openai/small{task: Translate}",
                ],
                Live,
                Ok((
                    "translate-en",
                    "multi/translate/balanced",
                    "openai/small{task: Translate}",
                    Translate,
                )),
            ),
            (
                &["--preset", "multi", "--task", "transcribe"],
                Live,
                Ok(("multi", "multi/transcribe/balanced", QWEN_LIVE, Transcribe)),
            ),
            (
                &["--preset", "multi", "--task", "translate"],
                Live,
                Err("cannot translate"),
            ),
        ];
        for (line, mode, expect) in rows {
            let got = ran(line, *mode);
            match (expect, got) {
                (Ok((preset, need, model, task)), Ok(got)) => assert_eq!(
                    (got.0.as_str(), got.1.as_str(), got.2.as_str(), got.3),
                    (*preset, *need, *model, *task),
                    "{line:?} {mode:?}"
                ),
                (Err(message), Err(e)) => {
                    assert!(e.to_string().contains(message), "{line:?}: {e}")
                }
                (expect, got) => panic!("{line:?} {mode:?}: expected {expect:?}, got {got:?}"),
            }
        }
    }

    /// Over every combination of `--preset`, `--speech`, `--task` and
    /// `--language`, in each mode: the task is `transcribe` unless a
    /// translating preset or `--task translate` asks otherwise, English
    /// speech is transcribed, a language not English is never given to an
    /// English model, and an error says what to change.
    #[test]
    fn test_rules_hold_everywhere() {
        let mut ran_lines = 0;
        for preset in [None, Some("en"), Some("multi"), Some("translate-en")] {
            for speech in [None, Some("en"), Some("multi")] {
                for task in [None, Some("transcribe"), Some("translate")] {
                    for language in [None, Some("en"), Some("de")] {
                        for mode in [Mode::Offline, Mode::Live] {
                            let mut line = Vec::new();
                            for (flag, value) in [
                                ("--preset", preset),
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
                                        e.contains("needs another preset")
                                            || e.contains("does not hear"),
                                        "{line:?}: {e}"
                                    );
                                    continue;
                                }
                            };
                            ran_lines += 1;
                            // The speech, from the flags alone: `--speech`,
                            // what `--language` implies, the preset's, or
                            // `default`'s (`multi` to translate).
                            let english = match (speech, language, preset) {
                                (Some(speech), _, _) => speech == "en",
                                (None, Some(language), _) => language == "en",
                                (None, None, Some(preset)) => preset == "en",
                                (None, None, None) => task != Some("translate"),
                            };
                            let asked = match (task, preset) {
                                (Some(task), _) => task == "translate",
                                (None, preset) => preset == Some("translate-en"),
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

    /// Every preset is listed, with both recipes.
    #[test]
    fn test_list() {
        let mut out = Vec::new();
        list(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("en (default)\n"), "{text}");
        for preset in presets() {
            for mode in [Mode::Offline, Mode::Live] {
                assert!(text.contains(&preset.recipe(mode).engine.model), "{text}");
            }
        }
        assert!(
            text.contains("  live:       qwen3-asr qwen/qwen3-asr-0.6b\n"),
            "{text}"
        );
    }
}
