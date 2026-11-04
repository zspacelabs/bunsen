//! Presets: speech to text by what it is for, not by which model does it.

use std::{
    collections::BTreeMap,
    fmt,
};

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
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

/// Which of a preset's recipes a command runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Files: `transcribe`.
    Offline,

    /// The microphone: `live`.
    Live,
}

impl Mode {
    /// The command that runs this recipe, a preset's or a family's.
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
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum, Serialize, Deserialize,
)]
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
        let need = construct_config()
            .implications
            .iter()
            .fold(self, |need, rule| {
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
        let fallback = &construct_config().fallback;
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
pub struct AsrRecipe {
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

/// One point of the table: what runs for a need, in a mode.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsrPoint {
    pub need: Need,
    pub mode: Mode,
    pub recipe: AsrRecipe,
}

/// A named, semantic choice of model: a point of the table, as `--preset`
/// names it.
///
/// A preset names what the speech is for (`en`, `multi`, `translate-en`),
/// with one recipe for files (`transcribe`) and one for the microphone
/// (`live`). The table is this binary's data: when a better model arrives, a
/// point changes, and the command lines that name the preset do not.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsrPreset {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub description: String,
    pub need: Need,
}

impl AsrPreset {
    /// The recipe `mode` runs.
    pub fn recipe(
        &self,
        mode: Mode,
    ) -> &'static AsrRecipe {
        &construct_config().point(self.need, mode).0.recipe
    }

    /// The recipe `mode` runs, as it is, resolved.
    ///
    /// # Errors
    /// As [`AsrRecipe::resolve`].
    pub fn resolve(
        &self,
        mode: Mode,
    ) -> BunsenResult<AsrConstruct> {
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

/// Where a need without a point looks next: per axis, each value's
/// candidates in order, the value itself first.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fallback {
    pub quality: BTreeMap<Quality, Vec<Quality>>,
    pub speech: BTreeMap<Speech, Vec<Speech>>,
}

impl Fallback {
    /// Whether `speech` hears a need of `need`'s speech: `need`'s fallback
    /// names it (a multilingual model also hears English), or it is `need`.
    pub fn hears(
        &self,
        speech: Speech,
        need: Speech,
    ) -> bool {
        speech == need || self.speech.get(&need).is_some_and(|c| c.contains(&speech))
    }
}

/// What a family's engine takes from the command line: the patch key
/// `--language` joins (none: the engine finds the language itself), and the
/// tasks its models do (none: it only transcribes).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineTraits {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub tasks: Option<EngineTasks>,
}

/// The tasks a family's models do, and how a model's expression sets one:
/// a preset in its box, or a value under a key in its patch.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineTasks {
    /// The task axis's default: its preset is left out of a box.
    pub default: Task,

    /// Each task's preset.
    pub presets: BTreeMap<Task, String>,

    /// The patch key, and each task's value under it.
    pub patch: TaskPatch,
}

/// The patch key that sets a task, and each task's value under it.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPatch {
    pub key: String,
    pub values: BTreeMap<Task, String>,
}

impl EngineTasks {
    /// Whether the family's models do `task`.
    pub fn does(
        &self,
        task: Task,
    ) -> bool {
        self.presets.contains_key(&task)
    }

    /// `model` set to `task`: its box's task presets replaced by `task`'s
    /// (by none for the default), and a task its patch sets removed.
    ///
    /// # Errors
    /// As [`LookupExpr::parse`].
    pub fn set(
        &self,
        model: &str,
        task: Task,
    ) -> BunsenResult<String> {
        let mut expr = LookupExpr::parse(model)?;
        expr.selection
            .presets
            .retain(|p| !self.presets.values().any(|t| t == p));
        if task != self.default
            && let Some(preset) = self.presets.get(&task)
        {
            expr.selection.presets.push(preset.clone());
        }
        if let Some(patch) = expr
            .selection
            .patch
            .as_mut()
            .and_then(|p| p.as_object_mut())
        {
            patch.remove(&self.patch.key);
        }
        Ok(expr.to_string())
    }

    /// The task `model` runs: as its patch sets it, else as its box names
    /// it, else the default.
    pub fn read(
        &self,
        model: &str,
    ) -> Task {
        let Ok(expr) = LookupExpr::parse(model) else {
            return self.default;
        };
        let patched = expr
            .selection
            .patch
            .as_ref()
            .and_then(|p| p.get(&self.patch.key))
            .and_then(|t| t.as_str());
        if let Some(value) = patched {
            return self
                .patch
                .values
                .iter()
                .find(|(_, v)| v.as_str() == value)
                .map_or(self.default, |(task, _)| *task);
        }
        self.presets
            .iter()
            .find(|(task, preset)| {
                **task != self.default && expr.selection.presets.contains(preset)
            })
            .map_or(self.default, |(task, _)| *task)
    }
}

/// What `--language` implies: the speech each listed code is, case aside,
/// and `otherwise` for any code not listed.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageRule {
    pub speech: BTreeMap<String, Speech>,
    pub otherwise: Speech,
}

impl LanguageRule {
    /// The speech `language` is.
    pub fn speech(
        &self,
        language: &str,
    ) -> Speech {
        self.speech
            .iter()
            .find(|(code, _)| code.eq_ignore_ascii_case(language))
            .map_or(self.otherwise, |(_, speech)| *speech)
    }
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
pub struct AsrConstructConfig {
    pub axes: Vec<Axis>,
    pub presets: Vec<AsrPreset>,
    pub fallback: Fallback,
    pub implications: Vec<Implication>,

    /// What an axis the command line leaves unset is, by what it sets,
    /// without `--preset` ([`fill`](Self::fill)).
    #[serde(default)]
    pub defaults: Vec<Implication>,

    /// What each family's engine takes from the command line, by family;
    /// [`engine`](Self::engine) gives a family not named none.
    #[serde(default)]
    pub engines: BTreeMap<Family, EngineTraits>,

    /// What `--language` implies.
    pub language: LanguageRule,

    /// The VAD slot's filler, as a lookup expression: what finds the speech
    /// regions of every emission but [`Offline`](Emission::Offline).
    pub vad: String,

    pub points: Vec<AsrPoint>,
}

/// The table, from the binary's `asr_construct.yml`.
///
/// # Panics
/// If the embedded file is not a catalog: a broken build, which the tests
/// catch.
pub fn construct_config() -> &'static AsrConstructConfig {
    static CATALOG: std::sync::OnceLock<AsrConstructConfig> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_saphyr::from_str(include_str!("asr_construct.yml"))
            .unwrap_or_else(|e| panic!("the embedded asr_construct.yml: {e}"))
    })
}

impl AsrConstructConfig {
    /// The need `given` asks for, without `--preset`: each axis it sets, then
    /// each default whose `when` the need matches (`given` over `base`) for
    /// the axes `given` leaves unset, then `base`'s; with the reasons of the
    /// defaults that set an axis, in order.
    pub fn fill(
        &self,
        given: NeedPattern,
        base: Need,
    ) -> (Need, Vec<&str>) {
        let asked = given.apply(base);
        let mut filled = given;
        let mut reasons = Vec::new();
        for rule in &self.defaults {
            if !rule.when.matches(asked) {
                continue;
            }
            let mut set = false;
            if filled.speech.is_none() && rule.then.speech.is_some() {
                filled.speech = rule.then.speech;
                set = true;
            }
            if filled.task.is_none() && rule.then.task.is_some() {
                filled.task = rule.then.task;
                set = true;
            }
            if filled.quality.is_none() && rule.then.quality.is_some() {
                filled.quality = rule.then.quality;
                set = true;
            }
            if set {
                reasons.push(rule.reason.as_str());
            }
        }
        (filled.apply(base), reasons)
    }

    /// What `family`'s engine takes from the command line: nothing, for a
    /// family `engines` does not name.
    pub fn engine(
        &self,
        family: Family,
    ) -> &EngineTraits {
        static NONE: EngineTraits = EngineTraits {
            language: None,
            tasks: None,
        };
        self.engines.get(&family).unwrap_or(&NONE)
    }

    /// The point for `need`, [settled](Need::settled), in `mode`, and, when
    /// it is not `need`'s own, a note of what stood in.
    pub fn point(
        &'static self,
        need: Need,
        mode: Mode,
    ) -> (&'static AsrPoint, Option<String>) {
        for candidate in need.settled().candidates() {
            if let Some(point) = self
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
                return (point, note);
            }
        }
        unreachable!("every speech and task reaches a balanced point; see asr_construct.yml")
    }
}

/// The preset `--preset` names when it is not given: an alias, so what it
/// means can change too.
pub const DEFAULT_PRESET: &str = "default";

/// The presets: names for the points of the table people ask for.
pub fn presets() -> &'static [AsrPreset] {
    &construct_config().presets
}

/// The preset `name` names, by its name or an alias.
///
/// # Errors
/// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup), with a
/// [`LookupError`] of every name and alias, when no preset has the name.
pub fn lookup(name: &str) -> BunsenResult<&'static AsrPreset> {
    presets()
        .iter()
        .find(|p| p.name == name || p.aliases.iter().any(|a| a == name))
        .ok_or_else(|| {
            let names = presets()
                .iter()
                .flat_map(|p| std::iter::once(&p.name).chain(&p.aliases))
                .cloned();
            BunsenError::lookup(LookupError::missing("ASR preset", name).with_candidates(names))
        })
}

/// A preset's recipe, with the command line's choices over it: what runs.
#[derive(Clone, Debug)]
pub struct AsrConstruct {
    /// The preset's name, not the alias it was named by; or the need the
    /// table was asked for (`multi/translate/fast`).
    pub preset: String,

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

impl AsrRecipe {
    /// This recipe of `preset`, serving `need`, with `task` and `language`
    /// set on its model.
    ///
    /// `task`, when given, is set as the family's [`EngineTasks`] say
    /// (Whisper's `[translate]` preset, or none for `transcribe`).
    /// `language` joins the model's patch under the family's
    /// [`language`](EngineTraits::language) key (Whisper's
    /// `{language: ...}`); a family with none, which finds the language
    /// itself, is logged, at `INFO`, as not taking it. The task the model
    /// runs, as it ends, is the need's.
    ///
    /// # Errors
    /// As [`LookupExpr::parse`] on the model, and [`VadArgs::slot`].
    pub fn resolve(
        &self,
        preset: &str,
        need: Need,
        mode: Mode,
        task: Option<Task>,
        language: Option<&str>,
    ) -> BunsenResult<AsrConstruct> {
        let engine = construct_config().engine(self.engine.family);
        let mut model = self.engine.model.clone();
        if let Some(task) = task
            && let Some(tasks) = &engine.tasks
        {
            model = tasks.set(&model, task)?;
        }
        if let Some(language) = language {
            if let Some(key) = &engine.language {
                let mut patch = serde_json::Map::new();
                patch.insert(key.clone(), serde_json::Value::from(language));
                model = with_patch(&model, &serde_json::Value::Object(patch))?;
            } else {
                log::info!(
                    "preset {preset} runs {}, which finds the language itself: \
                     --language {language} not applied",
                    self.engine.family.name()
                );
            }
        }
        let mut resolved = AsrConstruct {
            preset: preset.to_string(),
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

impl AsrConstruct {
    /// The task the model runs, as its family's [`EngineTasks`] read its
    /// expression (Whisper's: `{task: Translate}`, else `[translate]`, else
    /// `transcribe`); a family without tasks transcribes.
    pub fn task(&self) -> Task {
        construct_config()
            .engine(self.family)
            .tasks
            .as_ref()
            .map_or(Task::Transcribe, |tasks| tasks.read(&self.model))
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

    /// What `--json` records of this: the preset, the need it served, and
    /// what it ran.
    pub fn stamp(&self) -> Stamp {
        Stamp {
            preset: self.preset.clone(),
            need: self.need,
            recipe: AsrRecipeStamp {
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
    pub preset: String,

    /// The need served: its task is the one the model ran.
    pub need: Need,

    pub recipe: AsrRecipeStamp,
}

/// A recipe, as `--json` records it.
#[derive(Clone, Debug, Serialize)]
pub struct AsrRecipeStamp {
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
    ) -> BunsenResult<AsrConstruct> {
        let preset = lookup(name)?;
        preset
            .recipe(mode)
            .resolve(&preset.name, preset.need, mode, task, language)
    }

    /// Every recipe resolves, serving its preset's need; the names and
    /// aliases are each one preset's, and the default is one of them.
    #[test]
    fn test_profiles_parse() {
        let mut names = Vec::new();
        for preset in presets() {
            names.push(preset.name.as_str());
            names.extend(preset.aliases.iter().map(String::as_str));
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = resolve(&preset.name, mode, None, None).unwrap();
                assert_eq!(resolved.need, preset.need.settled(), "{}", preset.name);
            }
        }
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "{names:?}");
        assert_eq!(lookup(DEFAULT_PRESET).unwrap().name, "en");
        assert!(lookup("klingon").is_err());
    }

    /// Every need reaches a point in each mode; a preset is its own point,
    /// and a missing point falls through to the nearest.
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
                        let (point, note) = construct_config().point(need, mode);
                        assert_eq!(point.mode, mode);
                        assert_eq!(note.is_none(), point.need == need);
                    }
                }
            }
        }
        for preset in presets() {
            for mode in [Mode::Offline, Mode::Live] {
                let (point, note) = construct_config().point(preset.need, mode);
                assert_eq!(note, None, "{} {mode:?}", preset.name);
                assert_eq!(point.need, preset.need);
            }
        }

        let need = |speech, task, quality| Need {
            speech,
            task,
            quality,
        };
        // A better quality before a worse.
        let (point, _) = construct_config().point(
            need(Speech::Multi, Task::Transcribe, Quality::Good),
            Mode::Live,
        );
        assert_eq!(point.recipe.engine.model, "qwen/qwen3-asr-0.6b");
        // English translated to English is transcribed.
        let (point, note) = construct_config().point(
            need(Speech::En, Task::Translate, Quality::Fast),
            Mode::Offline,
        );
        assert_eq!(point.recipe.engine.model, "fermion/phonon-2");
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
        let tasks = construct_config()
            .engine(Family::Whisper)
            .tasks
            .as_ref()
            .unwrap();
        assert_eq!(
            selection(&tasks.set(&r.model, Task::Transcribe).unwrap()),
            (vec![], Some(json!({})))
        );
        assert_eq!(
            selection(
                &tasks
                    .set("openai/small[transcribe]", Task::Translate)
                    .unwrap()
            )
            .0,
            ["translate"]
        );
        // A family the construct names no engine traits for transcribes, and
        // takes no language.
        let parakeet = construct_config().engine(Family::Parakeet);
        assert!(parakeet.tasks.is_none() && parakeet.language.is_none());
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
        assert_eq!(stamp["preset"], "translate-en");
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

        let table = construct_config();
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

        for point in &table.points {
            let model = &point.recipe.engine.model;
            let looked = match point.recipe.engine.family {
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
            looked.unwrap_or_else(|e| panic!("{} {model}: {e}", point.need));
        }

        // Every engine's task presets are presets its family's rows have.
        for (family, engine) in &table.engines {
            let Some(tasks) = &engine.tasks else {
                continue;
            };
            let listing = crate::cli::asr::family::Kit::Asr(*family)
                .listing()
                .unwrap();
            for preset in tasks.presets.values() {
                assert!(
                    listing
                        .rows
                        .iter()
                        .any(|(_, row)| row.presets.get(preset).is_some()),
                    "{} {preset}",
                    family.name()
                );
            }
            assert!(tasks.does(tasks.default));
        }

        // Without a preset, translating makes an unset speech `multi`, and
        // says why; a speech given stays, and nothing set takes the base.
        let base = lookup(DEFAULT_PRESET).unwrap().need;
        let translate = NeedPattern {
            task: Some(Task::Translate),
            ..NeedPattern::default()
        };
        let (need, why) = table.fill(translate, base);
        assert_eq!((need.speech, need.task), (Speech::Multi, Task::Translate));
        assert_eq!(why, ["translating is from speech that is not English"]);
        let (need, why) = table.fill(
            NeedPattern {
                speech: Some(Speech::En),
                ..translate
            },
            base,
        );
        assert_eq!((need.speech, why.len()), (Speech::En, 0));
        assert_eq!(table.fill(NeedPattern::default(), base), (base, vec![]));

        // `--language` by the table, case aside; what hears what by the
        // speech fallback.
        assert_eq!(table.language.speech("EN"), Speech::En);
        assert_eq!(table.language.speech("de"), Speech::Multi);
        assert!(table.fallback.hears(Speech::Multi, Speech::En));
        assert!(!table.fallback.hears(Speech::En, Speech::Multi));
        assert!(table.fallback.hears(Speech::En, Speech::En));

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
