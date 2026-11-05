//! Laya's questions, prompts and answers.
use std::{
    collections::BTreeMap,
    fmt,
};

use burn::{
    Tensor,
    prelude::{
        Bool,
        Int,
        TensorData,
    },
    tensor::Device,
};
use serde::{
    Deserialize,
    Deserializer,
    Serialize,
    Serializer,
    ser::SerializeMap,
};

use super::{
    LayaAgentConfig,
    LayaModel,
    LayaQuestionKind,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::tokens::Tokenizer,
};

/// The longest an option's text runs, in tokens, before its prompt's budget
/// is spent (upstream's cut, `[:48]`).
pub const LAYA_OPTION_TOKENS: usize = 48;

/// An option of a question: a name, and what it means, if said.
///
/// Read from a bare name (`retry`) or a one-entry map (`{retry: "try the
/// call again"}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayaOption {
    /// The name: the option's key in an answer.
    pub name: String,

    /// What it means, rendered after its name.
    pub description: Option<String>,
}

impl LayaOption {
    /// An option of `name` alone.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
        }
    }
}

impl<'de> Deserialize<'de> for LayaOption {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Name(String),
            Described(BTreeMap<String, Option<String>>),
        }
        match Repr::deserialize(deserializer)? {
            Repr::Name(name) => Ok(Self::named(name)),
            Repr::Described(map) if map.len() == 1 => {
                let (name, description) = map.into_iter().next().expect("one entry");
                Ok(Self { name, description })
            }
            Repr::Described(map) => Err(serde::de::Error::custom(format!(
                "an option is a name or a one-entry map, not {} entries",
                map.len()
            ))),
        }
    }
}

impl Serialize for LayaOption {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match &self.description {
            None => serializer.serialize_str(&self.name),
            Some(description) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(&self.name, description)?;
                map.end()
            }
        }
    }
}

/// A question about a state.
///
/// - [`Choice`](LayaQuestionKind::Choice): `options` are the names to pick
///   among, each with what it means if said.
/// - [`Score`](LayaQuestionKind::Score): `options` are the scale's levels, in
///   order, each a level's description; the answer keys them `0`, `1`, ….
/// - [`Noul`](LayaQuestionKind::Noul): yes or no; `options`, when given, say
///   what `false` and `true` mean, as `{false: …}` and `{true: …}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayaQuestion {
    /// Its kind.
    #[serde(rename = "type")]
    pub kind: LayaQuestionKind,

    /// What is asked.
    pub question: String,

    /// Its options, as its kind reads them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<LayaOption>,
}

impl LayaQuestion {
    /// The options' texts, in order, as the prompt holds them (upstream's
    /// `render_options`): a choice's `name` or `name: description`; a
    /// score's `level i: description`; a yes/no question's `false: …` and
    /// `true: …`, each with upstream's words where the question gives none.
    pub fn rendered_options(&self) -> Vec<String> {
        match self.kind {
            LayaQuestionKind::Choice => self
                .options
                .iter()
                .map(|o| match &o.description {
                    Some(d) if !d.is_empty() => format!("{}: {d}", o.name),
                    _ => o.name.clone(),
                })
                .collect(),
            LayaQuestionKind::Score => self
                .options
                .iter()
                .enumerate()
                .map(|(i, o)| format!("level {i}: {}", o.description.as_ref().unwrap_or(&o.name)))
                .collect(),
            LayaQuestionKind::Noul => {
                let said = |key: &str| {
                    self.options
                        .iter()
                        .find(|o| o.name == key)
                        .and_then(|o| o.description.clone())
                        .filter(|d| !d.is_empty())
                };
                vec![
                    format!(
                        "false: {}",
                        said("false").unwrap_or("no, the statement does not hold".into())
                    ),
                    format!(
                        "true: {}",
                        said("true").unwrap_or("yes, the statement holds".into())
                    ),
                ]
            }
        }
    }

    /// The options' keys in an answer, in order: a choice's names; a score's
    /// level indices; `false` and `true`.
    pub fn option_keys(&self) -> Vec<String> {
        match self.kind {
            LayaQuestionKind::Choice => self.options.iter().map(|o| o.name.clone()).collect(),
            LayaQuestionKind::Score => (0..self.options.len()).map(|i| i.to_string()).collect(),
            LayaQuestionKind::Noul => vec!["false".into(), "true".into()],
        }
    }
}

/// A Laya tokenizer's special tokens, by spelling and by id: the prompt's
/// framing, its option markers, and its padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayaSpecials {
    /// `[CLS]` (English), `<bos>` (multilingual).
    pub cls: i64,

    /// `[SEP]`, `<eos>`.
    pub sep: i64,

    /// The option marker, `[MASK]`, `<mask>`, by id.
    pub mask: i64,

    /// The marker's spelling, which the question's and the state's texts
    /// have replaced by a space.
    pub mask_spelling: String,

    /// `[PAD]`, `<pad>`.
    pub pad: i64,
}

impl LayaSpecials {
    /// The specials a `tokenizer_config.json` names (`cls_token`,
    /// `sep_token`, `mask_token`, `pad_token`, each a spelling or an added
    /// token's `{content: …}`), by their ids in `tokenizer`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when the file does not parse, lacks one of them, or names a token
    /// `tokenizer` does not have.
    pub fn from_tokenizer_config(
        json: &str,
        tokenizer: &dyn Tokenizer,
    ) -> BunsenResult<Self> {
        let config: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e))?;
        let spelling = |field: &str| -> BunsenResult<String> {
            let value = &config[field];
            value
                .as_str()
                .or_else(|| value["content"].as_str())
                .map(str::to_string)
                .ok_or_else(|| {
                    BunsenError::invalid_resource(format!("tokenizer_config.json: no `{field}`"))
                })
        };
        let id = |field: &str| -> BunsenResult<(String, i64)> {
            let token = spelling(field)?;
            let id = tokenizer.token_id(&token).ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "tokenizer_config.json: `{field}` is {token:?}, which the tokenizer lacks"
                ))
            })?;
            Ok((token, id))
        };
        let (mask_spelling, mask) = id("mask_token")?;
        Ok(Self {
            cls: id("cls_token")?.1,
            sep: id("sep_token")?.1,
            mask,
            mask_spelling,
            pad: id("pad_token")?.1,
        })
    }
}

/// A prompt, as the model reads it: its ids, and each option's marker
/// position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayaPrompt {
    /// The ids: `[CLS] <kind> question: … [SEP] [MASK] option … [SEP] state
    /// [SEP]`, cut to the budget.
    pub ids: Vec<i64>,

    /// Each option's `[MASK]` position, in order; an option past the
    /// budget has none.
    pub markers: Vec<usize>,
}

/// Builds a question's prompt over `state`, as upstream's `build_sequence`
/// does, budget rules and all.
///
/// The question and its options come first, within `head_max_len` tokens:
/// each option is its marker and at most [`LAYA_OPTION_TOKENS`] of its text;
/// if they leave the question fewer than 16 tokens, every option is cut to
/// `max(4, (head_max_len - 16) / options)`; the question then takes what is
/// left, at least 8. The state takes the room left under `max_len`, its
/// start (or, with `truncate_left`, its end); the whole is cut to `max_len`.
///
/// `state` is text; upstream renders a structured state as
/// `json.dumps(state, ensure_ascii=False)` (`", "` and `": "` between
/// items), which a caller passes as that text.
///
/// # Errors
/// As the tokenizer.
pub fn build_prompt(
    tokenizer: &dyn Tokenizer,
    specials: &LayaSpecials,
    state: &str,
    question: &LayaQuestion,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> BunsenResult<LayaPrompt> {
    let unmask = |text: &str| text.replace(&specials.mask_spelling, " ");
    let mut head = tokenizer.encode(&format!(
        "{} question: {}",
        question.kind.name(),
        unmask(&question.question)
    ))?;
    let mut options = question
        .rendered_options()
        .iter()
        .map(|o| {
            let mut ids = vec![specials.mask];
            let text = tokenizer.encode(&format!(" {}", unmask(o)))?;
            ids.extend(text.into_iter().take(LAYA_OPTION_TOKENS));
            Ok(ids)
        })
        .collect::<BunsenResult<Vec<Vec<i64>>>>()?;
    let used = |options: &[Vec<i64>]| options.iter().map(Vec::len).sum::<usize>();
    let mut budget = head_max_len as i64 - used(&options) as i64;
    if budget < 16 {
        let per = ((head_max_len as i64 - 16) / options.len().max(1) as i64).max(4) as usize;
        for o in &mut options {
            o.truncate(per);
        }
        budget = head_max_len as i64 - used(&options) as i64;
    }
    head.truncate(budget.max(8) as usize);

    let mut ids = vec![specials.cls];
    ids.extend(head);
    ids.push(specials.sep);
    let mut markers = Vec::with_capacity(options.len());
    for o in options {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(specials.sep);
    let room = max_len.saturating_sub(ids.len() + 1);
    let state = tokenizer.encode(&unmask(state))?;
    let state: &[i64] = if !truncate_left {
        &state[..room.min(state.len())]
    } else if room == 0 {
        // Upstream's `st[-0:]` is the whole state; the cut below trims it.
        &state
    } else {
        &state[state.len().saturating_sub(room)..]
    };
    ids.extend_from_slice(state);
    ids.push(specials.sep);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    Ok(LayaPrompt { ids, markers })
}

/// A question's answer: each option's calibrated probability, the most
/// probable option, and what its kind adds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayaAnswer {
    /// The question asked.
    pub question: String,

    /// Its kind.
    #[serde(rename = "type")]
    pub kind: LayaQuestionKind,

    /// Each option's probability, by its key, in order.
    #[serde(serialize_with = "ordered_map")]
    pub probabilities: Vec<(String, f64)>,

    /// The most probable option's key.
    pub answer: String,

    /// One less the answer distribution's entropy over its maximum, `ln k`:
    /// 1 when certain, 0 when uniform.
    pub confidence: f64,

    /// A score's expected level, `Σ i · p_i`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,

    /// The act head's probability of its first act (`escalate`).
    pub act_probability: f64,
}

fn ordered_map<S: Serializer>(
    entries: &[(String, f64)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut map = serializer.serialize_map(Some(entries.len()))?;
    for (k, v) in entries {
        map.serialize_entry(k, v)?;
    }
    map.end()
}

impl fmt::Display for LayaAnswer {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}: {}", self.question, self.answer)
    }
}

impl LayaAnswer {
    /// The answer to `question` from its calibrated `probabilities` and its
    /// act probability.
    pub fn new(
        question: &LayaQuestion,
        probabilities: Vec<f64>,
        act_probability: f64,
    ) -> Self {
        let k = probabilities.len();
        let best = probabilities.iter().enumerate().fold(0, |best, (i, &p)| {
            if p > probabilities[best] { i } else { best }
        });
        let confidence = if k < 2 {
            1.0
        } else {
            let entropy: f64 = probabilities
                .iter()
                .map(|&p| -p * p.clamp(1e-12, 1.0).ln())
                .sum();
            1.0 - entropy / (k as f64).ln()
        };
        let score = (question.kind == LayaQuestionKind::Score).then(|| {
            probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| i as f64 * p)
                .sum()
        });
        let keys = question.option_keys();
        Self {
            question: question.question.clone(),
            kind: question.kind,
            answer: keys[best].clone(),
            probabilities: keys.into_iter().zip(probabilities).collect(),
            confidence,
            score,
            act_probability,
        }
    }
}

impl LayaModel {
    /// Answers `questions` about `state`: one prompt each
    /// ([`build_prompt`], under `agent`'s budgets), run as one padded batch,
    /// each question's logits calibrated by `agent`'s temperature for its
    /// kind and option count.
    ///
    /// # Errors
    /// As the tokenizer;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a question
    /// with no options, or whose options do not fit in `head_max_len`.
    pub fn decide(
        &self,
        tokenizer: &dyn Tokenizer,
        specials: &LayaSpecials,
        agent: &LayaAgentConfig,
        state: &str,
        questions: &[LayaQuestion],
        device: &Device,
    ) -> BunsenResult<Vec<LayaAnswer>> {
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        let prompts = questions
            .iter()
            .map(|q| {
                let n = q.rendered_options().len();
                if q.kind != LayaQuestionKind::Noul && n == 0 {
                    return Err(BunsenError::illegal(format!(
                        "question {:?}: no options",
                        q.question
                    )));
                }
                let prompt = build_prompt(
                    tokenizer,
                    specials,
                    state,
                    q,
                    agent.max_len,
                    agent.head_max_len,
                    false,
                )?;
                if prompt.markers.len() != n {
                    return Err(BunsenError::illegal(format!(
                        "question {:?}: its options do not fit in head_max_len={} tokens",
                        q.question, agent.head_max_len
                    )));
                }
                Ok(prompt)
            })
            .collect::<BunsenResult<Vec<_>>>()?;

        let b = prompts.len();
        let t = prompts.iter().map(|p| p.ids.len()).max().unwrap_or(0);
        let k = prompts.iter().map(|p| p.markers.len()).max().unwrap_or(0);
        let mut ids = vec![specials.pad; b * t];
        let mut markers = vec![0i64; b * k];
        let mut options = vec![false; b * k];
        for (row, p) in prompts.iter().enumerate() {
            ids[row * t..row * t + p.ids.len()].copy_from_slice(&p.ids);
            for (j, &m) in p.markers.iter().enumerate() {
                markers[row * k + j] = m as i64;
                options[row * k + j] = true;
            }
        }
        let lengths: Vec<usize> = prompts.iter().map(|p| p.ids.len()).collect();
        let kinds: Vec<i64> = questions.iter().map(|q| q.kind.index() as i64).collect();
        let out = self.forward(
            Tensor::<2, Int>::from_data(TensorData::new(ids, [b, t]), device),
            &lengths,
            Tensor::<1, Int>::from_data(TensorData::new(kinds, [b]), device),
            Tensor::<2, Int>::from_data(TensorData::new(markers, [b, k]), device),
            Tensor::<2, Bool>::from_data(TensorData::new(options, [b, k]), device),
        );
        let logits: Vec<f32> = out
            .logits
            .into_data()
            .convert::<f32>()
            .try_into_vec()
            .map_err(|e| BunsenError::internal(format!("the logits: {e:?}")))?;
        let act = burn::tensor::activation::softmax(out.act_logits, 1)
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .map_err(|e| BunsenError::internal(format!("the act logits: {e:?}")))?;
        let n_act = act.len() / b;
        Ok(questions
            .iter()
            .zip(&prompts)
            .enumerate()
            .map(|(row, (q, p))| {
                let row_logits = &logits[row * k..row * k + p.markers.len()];
                LayaAnswer::new(
                    q,
                    agent.probabilities(q.kind, row_logits),
                    f64::from(act[row * n_act]),
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer of a test's own: `[CLS]` 1, `[SEP]` 2, `[MASK]` 3, `[PAD]`
    /// 0 as whole tokens; every other character its own id, `10 +` its
    /// position in the alphabet below.
    #[derive(Debug)]
    pub(crate) struct Chars;

    const ALPHABET: &str = " abcdefghijklmnopqrstuvwxyz:0123456789.,?";

    impl Tokenizer for Chars {
        fn encode(
            &self,
            text: &str,
        ) -> BunsenResult<Vec<i64>> {
            let mut ids = Vec::new();
            let mut rest = text;
            while let Some(c) = rest.chars().next() {
                if let Some((id, tail)) = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
                    .iter()
                    .enumerate()
                    .find_map(|(i, s)| rest.strip_prefix(s).map(|t| (i as i64, t)))
                {
                    ids.push(id);
                    rest = tail;
                    continue;
                }
                ids.push(10 + ALPHABET.find(c).unwrap_or(0) as i64);
                rest = &rest[c.len_utf8()..];
            }
            Ok(ids)
        }

        fn token_id(
            &self,
            content: &str,
        ) -> Option<i64> {
            ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
                .iter()
                .position(|s| *s == content)
                .map(|i| i as i64)
        }
    }

    pub(crate) fn specials() -> LayaSpecials {
        LayaSpecials::from_tokenizer_config(
            r#"{"cls_token": "[CLS]", "sep_token": "[SEP]",
                "mask_token": {"content": "[MASK]"}, "pad_token": "[PAD]"}"#,
            &Chars,
        )
        .unwrap()
    }

    fn text(ids: &[i64]) -> String {
        ids.iter()
            .map(|&id| match id {
                0 => "_".to_string(),
                1 => "<".to_string(),
                2 => "|".to_string(),
                3 => "#".to_string(),
                id => ALPHABET[(id - 10) as usize..][..1].to_string(),
            })
            .collect()
    }

    fn choice() -> LayaQuestion {
        serde_json::from_str(
            r#"{"type": "choice", "question": "which next?",
                "options": ["retry", {"close": "end it"}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn test_specials() {
        let s = specials();
        assert_eq!((s.cls, s.sep, s.mask, s.pad), (1, 2, 3, 0));
        assert_eq!(s.mask_spelling, "[MASK]");
        assert!(LayaSpecials::from_tokenizer_config(r#"{"cls_token": "[X]"}"#, &Chars).is_err());
    }

    /// Upstream's option texts, and the keys an answer gives them.
    #[test]
    fn test_rendered_options() {
        assert_eq!(choice().rendered_options(), ["retry", "close: end it"]);
        assert_eq!(choice().option_keys(), ["retry", "close"]);
        let score: LayaQuestion = serde_json::from_str(
            r#"{"type": "score", "question": "how bad?", "options": ["fine", "bad"]}"#,
        )
        .unwrap();
        assert_eq!(score.rendered_options(), ["level 0: fine", "level 1: bad"]);
        assert_eq!(score.option_keys(), ["0", "1"]);
        let noul: LayaQuestion = serde_json::from_str(
            r#"{"type": "noul", "question": "done?", "options": [{"true": "it is done"}]}"#,
        )
        .unwrap();
        assert_eq!(
            noul.rendered_options(),
            ["false: no, the statement does not hold", "true: it is done"]
        );
        assert_eq!(noul.option_keys(), ["false", "true"]);
        assert!(serde_json::from_str::<LayaOption>(r#"{"a": "x", "b": "y"}"#).is_err());
    }

    /// `[CLS] <kind> question: … [SEP] [MASK] opt … [SEP] state [SEP]`, the
    /// markers at the options', and a `[MASK]` in the texts a space.
    #[test]
    fn test_prompt_layout() {
        let mut q = choice();
        q.question = "which[MASK]next?".into();
        let p = build_prompt(&Chars, &specials(), "ok", &q, 512, 192, false).unwrap();
        assert_eq!(
            text(&p.ids),
            "<choice question: which next?|# retry# close: end it|ok|"
        );
        assert_eq!(p.markers, [30, 37]);
        assert_eq!(p.ids[p.markers[0]], 3);
    }

    /// The budgets: the state takes what `max_len` leaves, from its start or
    /// its end; options past 48 tokens are cut; when the options leave the
    /// question under 16, each is cut evenly and the question kept at 8.
    #[test]
    fn test_prompt_budgets() {
        let q = choice();
        let full = build_prompt(&Chars, &specials(), "abcdef", &q, 512, 192, false).unwrap();
        let short = full.ids.len() - 3;
        let p = build_prompt(&Chars, &specials(), "abcdef", &q, short, 192, false).unwrap();
        assert!(text(&p.ids).ends_with("|abc|"), "{}", text(&p.ids));
        let p = build_prompt(&Chars, &specials(), "abcdef", &q, short, 192, true).unwrap();
        assert!(text(&p.ids).ends_with("|def|"), "{}", text(&p.ids));

        let long: LayaQuestion = serde_json::from_str(&format!(
            r#"{{"type": "choice", "question": "q?", "options": ["{}"]}}"#,
            "x".repeat(60)
        ))
        .unwrap();
        let p = build_prompt(&Chars, &specials(), "", &long, 512, 192, false).unwrap();
        assert_eq!(
            p.ids.len(),
            1 + "choice question: q?".len() + 1 + 1 + 48 + 1 + 1
        );

        // Two options of 1 + 48 tokens in a 100-token head: 2 left for the
        // question, under 16, so each option is cut to (100 - 16) / 2 = 42
        // and the question takes the 16 left.
        let two: LayaQuestion = serde_json::from_str(&format!(
            r#"{{"type": "choice", "question": "{}", "options": ["{x}", "{x}"]}}"#,
            "w".repeat(30),
            x = "x".repeat(60)
        ))
        .unwrap();
        let p = build_prompt(&Chars, &specials(), "", &two, 512, 100, false).unwrap();
        assert_eq!(p.markers, [1 + 16 + 1, 1 + 16 + 1 + 42]);
        assert_eq!(p.ids.len(), 1 + 16 + 1 + 42 * 2 + 1 + 1);
    }

    /// A marker past `max_len` is dropped with its option.
    #[test]
    fn test_prompt_markers_past_the_budget() {
        let p = build_prompt(&Chars, &specials(), "", &choice(), 32, 192, false).unwrap();
        assert_eq!(p.ids.len(), 32);
        assert_eq!(p.markers, [30]);
    }

    /// The answer's choice is its most probable option; its confidence one
    /// less its normalized entropy; a score's expectation its levels'.
    #[test]
    fn test_answer() {
        let a = LayaAnswer::new(&choice(), vec![0.25, 0.75], 0.1);
        assert_eq!(a.answer, "close");
        let h = -(0.25f64 * 0.25f64.ln() + 0.75 * 0.75f64.ln());
        assert!((a.confidence - (1.0 - h / 2f64.ln())).abs() < 1e-12);
        assert_eq!(a.score, None);
        let yaml = serde_json::to_string(&a).unwrap();
        assert!(
            yaml.contains(r#""probabilities":{"retry":0.25,"close":0.75}"#),
            "{yaml}"
        );

        let score: LayaQuestion = serde_json::from_str(
            r#"{"type": "score", "question": "q", "options": ["a", "b", "c"]}"#,
        )
        .unwrap();
        let a = LayaAnswer::new(&score, vec![0.2, 0.3, 0.5], 0.0);
        assert_eq!((a.answer.as_str(), a.score), ("2", Some(1.3)));
    }
}
