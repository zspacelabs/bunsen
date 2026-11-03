//! Profiles: speech to text by what it is for, not by which model does it.

use std::{
    collections::BTreeMap,
    fmt,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::construct::LookupExpr,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::cli::{
    asr::{
        family::Family,
        stream::{
            Emission,
            Streaming,
            VadArgs,
        },
    },
    with_patch,
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
            .resolve(&self.name, self.need, mode, None, None)
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

    /// The VAD slot's filler, as a lookup expression: what finds the speech
    /// regions of every emission but [`Offline`](Emission::Offline).
    pub vad: String,

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

/// A profile's recipe, with the command line's choices over it: what runs.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The profile's name, not the alias it was named by; or the need the
    /// table was asked for (`multi/translate/fast`).
    pub profile: String,

    /// The need it serves, [settled](Need::settled): its task is the one
    /// the model runs.
    pub need: Need,

    pub mode: Mode,
    pub family: Family,
    pub model: String,
    pub emission: Emission,
    pub chunk_ms: usize,

    /// The VAD slot's filler, a lookup expression; `None` under an emission
    /// that finds no speech regions.
    pub vad: Option<String>,
}

impl Recipe {
    /// This recipe of `profile`, serving `need`, with `task` and `language`
    /// set on its model.
    ///
    /// `task`, when given, is Whisper's: its `[translate]` preset in the
    /// model's box, or none for `transcribe`. `language` is Whisper's
    /// `{language: ...}` in the model's patch; a family that finds the
    /// language itself is logged, at `INFO`, as not taking it. The task the
    /// model runs, as it ends, is the need's.
    ///
    /// # Errors
    /// As [`LookupExpr::parse`] on the model, and [`VadArgs::slot`].
    pub fn resolve(
        &self,
        profile: &str,
        need: Need,
        mode: Mode,
        task: Option<Task>,
        language: Option<&str>,
    ) -> BunsenResult<Resolved> {
        let mut model = self.engine.model.clone();
        if let Some(task) = task {
            model = with_task(&model, task)?;
        }
        if let Some(language) = language {
            if self.engine.family == Family::Whisper {
                model = with_patch(&model, &serde_json::json!({ "language": language }))?;
            } else {
                log::info!(
                    "profile {profile} runs {}, which finds the language itself: \
                     --language {language} not applied",
                    self.engine.family.name()
                );
            }
        }
        let mut resolved = Resolved {
            profile: profile.to_string(),
            need: need.settled(),
            mode,
            family: self.engine.family,
            model,
            emission: self.emission,
            chunk_ms: self.chunk_ms,
            vad: VadArgs::default().slot(self.emission)?,
        };
        resolved.need.task = resolved.task();
        Ok(resolved)
    }
}

/// Whisper's `model` set to `task`: its box's `translate` and `transcribe`
/// presets replaced by `translate`, or by none for `transcribe` (the task
/// axis's default), and a `task` its patch sets removed.
///
/// # Errors
/// As [`LookupExpr::parse`].
fn with_task(
    model: &str,
    task: Task,
) -> BunsenResult<String> {
    let mut expr = LookupExpr::parse(model)?;
    expr.selection
        .presets
        .retain(|p| p != "translate" && p != "transcribe");
    if task == Task::Translate {
        expr.selection.presets.push("translate".to_string());
    }
    if let Some(patch) = expr
        .selection
        .patch
        .as_mut()
        .and_then(|p| p.as_object_mut())
    {
        patch.remove("task");
    }
    Ok(expr.to_string())
}

impl Resolved {
    /// The task the model runs: Whisper's, as its model's patch sets it
    /// (`{task: Translate}`), else `translate` when its box names the
    /// `translate` preset, `transcribe` without; another family transcribes.
    pub fn task(&self) -> Task {
        if self.family != Family::Whisper {
            return Task::Transcribe;
        }
        let Ok(expr) = LookupExpr::parse(&self.model) else {
            return Task::Transcribe;
        };
        let patched = expr
            .selection
            .patch
            .as_ref()
            .and_then(|p| p.get("task"))
            .and_then(|t| t.as_str());
        match patched {
            Some("Translate") => Task::Translate,
            Some(_) => Task::Transcribe,
            None if expr.selection.presets.iter().any(|p| p == "translate") => Task::Translate,
            None => Task::Transcribe,
        }
    }

    /// The `bunsen-tool asr family` command this runs, word by word.
    pub fn command_line(&self) -> Vec<String> {
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
        if let Some(vad) = &self.vad {
            line.extend(["--vad".to_string(), vad.clone()]);
        }
        line
    }

    /// How this streams.
    pub fn streaming(&self) -> Streaming {
        Streaming {
            emission: self.emission,
            vad: self.vad.clone(),
            chunk_ms: self.chunk_ms,
        }
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
                vad: self.vad.clone(),
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

    /// The VAD, when the emission finds speech regions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vad: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(
        name: &str,
        mode: Mode,
        task: Option<Task>,
        language: Option<&str>,
    ) -> BunsenResult<Resolved> {
        let profile = lookup(name)?;
        profile
            .recipe(mode)
            .resolve(&profile.name, profile.need, mode, task, language)
    }

    /// Every recipe resolves, serving its profile's need; the names and
    /// aliases are each one profile's, and the default is one of them.
    #[test]
    fn test_profiles_parse() {
        let mut names = Vec::new();
        for profile in profiles() {
            names.push(profile.name.as_str());
            names.extend(profile.aliases.iter().map(String::as_str));
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = resolve(&profile.name, mode, None, None).unwrap();
                assert_eq!(resolved.need, profile.need.settled(), "{}", profile.name);
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

    /// The model's selection, parsed: its box and its patch.
    fn selection(model: &str) -> (Vec<String>, Option<serde_json::Value>) {
        let expr = LookupExpr::parse(model).unwrap();
        (expr.selection.presets, expr.selection.patch)
    }

    /// The task is set in the model's box, the language in its patch, on
    /// Whisper alone; the need's task is what the model runs.
    #[test]
    fn test_task_and_language() {
        use serde_json::json;

        // `translate-en` is Whisper: the language joins the recipe's patch.
        let r = resolve("translate-en", Mode::Offline, None, Some("de")).unwrap();
        assert_eq!(
            selection(&r.model),
            (
                vec!["translate".to_string()],
                Some(json!({"beam_size": 5, "language": "de"}))
            )
        );
        assert_eq!(r.need.task, Task::Translate);

        // Transcribing takes `translate` out of the box: the axis's default.
        let r = resolve("translate-en", Mode::Live, Some(Task::Transcribe), None).unwrap();
        assert_eq!(selection(&r.model), (vec![], Some(json!({"beam_size": 1}))));
        assert_eq!(
            (r.task(), r.need.task),
            (Task::Transcribe, Task::Transcribe)
        );

        // `en` is Parakeet, which finds the language itself.
        let r = resolve("en", Mode::Live, None, Some("de")).unwrap();
        assert_eq!(r.model, "fermion/phonon-2");

        // The patch's task wins over the box, and setting a task clears it.
        let mut r = resolve("translate-en", Mode::Live, None, None).unwrap();
        r.model = "openai/small{task: Transcribe}".to_string();
        assert_eq!(r.task(), Task::Transcribe);
        r.model = "openai/small{task: Translate}".to_string();
        assert_eq!(r.task(), Task::Translate);
        assert_eq!(
            selection(&with_task(&r.model, Task::Transcribe).unwrap()),
            (vec![], Some(json!({})))
        );
        assert_eq!(
            selection(&with_task("openai/small[transcribe]", Task::Translate).unwrap()).0,
            ["translate"]
        );
    }

    #[test]
    fn test_command_line() {
        let r = resolve("translate-en", Mode::Live, None, None).unwrap();
        assert_eq!(
            shell_line(&r.command_line()),
            "bunsen-tool asr family whisper live --model 'openai/small[translate]{beam_size: 1}' \
             --emission conservative --chunk-ms 250 --vad 'silero/vad-16k[faster-whisper]'"
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
