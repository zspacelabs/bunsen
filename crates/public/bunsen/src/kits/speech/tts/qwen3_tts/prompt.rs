//! Qwen3-TTS's prompt: the rows the talker reads before it generates, and
//! the text each generated frame adds.
//!
//! A row is a text part, a codec part, or both, summed: a text id through
//! the text table and its projection, a codec id through the codec table
//! ([`Qwen3TtsTalker::text_embed`],
//! [`codec_embed`](Qwen3TtsTalker::codec_embed)). In order:
//!
//! 1. the instruction, when there is one: its chat-wrapped ids, text alone;
//! 2. the role, `<|im_start|>assistant\n` (the text's first three ids), text
//!    alone;
//! 3. the codec prefix, each row a codec id over the text pad (the last over
//!    the text BOS): the think tags around a language (`think`, `think_bos`,
//!    the language, `think_eos`), or, without a language, `nothink`,
//!    `think_bos`, `think_eos`; the speaker's row, when there is one; the codec
//!    pad;
//! 4. the text, by [placement](Qwen3TtsTextPlacement):
//!    - interleaved (upstream's "streaming"): its first id over the codec BOS;
//!      each generated frame then adds the next text id, then the text EOS,
//!      then the text pad ([`trailing`](Qwen3TtsPromptRows::trailing));
//!    - upfront ("non-streaming"): every text id, then the text EOS, each over
//!      the codec pad, then the text pad over the codec BOS; each frame adds
//!      the text pad.
//!
//! The text is chat-wrapped and tokenized by the caller:
//! `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`, whose
//! first [`role_ids`](Qwen3TtsPromptConfig::role_ids) ids (three) are the role
//! and last [`close_ids`](Qwen3TtsPromptConfig::close_ids) (five) the close;
//! [`Qwen3TtsPromptRows::build`] checks both against the config's chat ids. An
//! instruction is `<|im_start|>user\n{instruction}<|im_end|>\n`.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    prelude::*,
};

use super::blocks::Qwen3TtsTalker;
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::speech::tts::driver::TtsVoiceInfo,
};

/// Where the text goes: all before the first frame, or a token a frame.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum Qwen3TtsTextPlacement {
    /// A text token joins each generated frame (upstream's "streaming").
    Interleaved,

    /// The whole text before the first frame (upstream's "non-streaming").
    Upfront,
}

/// The ids the prompt is laid out of, and the languages and speakers by
/// the model's keys.
#[derive(Config, Debug)]
pub struct Qwen3TtsPromptConfig {
    /// The text pad.
    pub tts_pad: u32,
    /// The text BOS.
    pub tts_bos: u32,
    /// The text EOS.
    pub tts_eos: u32,
    /// `<|im_start|>`, which opens a chat turn.
    pub im_start: u32,
    /// `<|im_end|>`, which closes a chat turn.
    pub im_end: u32,
    /// `assistant`, the role of the text's turn.
    pub assistant: u32,
    /// The ids of the role a wrapped text opens with,
    /// `<|im_start|>assistant\n` (upstream's text is `input_ids[:, 3:-5]`).
    #[config(default = 3)]
    pub role_ids: usize,
    /// The ids of the close a wrapped text ends with,
    /// `<|im_end|>\n<|im_start|>assistant\n`.
    #[config(default = 5)]
    pub close_ids: usize,
    /// The codec pad.
    pub codec_pad: u32,
    /// The codec BOS.
    pub codec_bos: u32,
    /// The codec EOS, which ends a take.
    pub codec_eos: u32,
    /// The think tag.
    pub codec_think: u32,
    /// The no-think tag.
    pub codec_nothink: u32,
    /// The think block's open.
    pub codec_think_bos: u32,
    /// The think block's close.
    pub codec_think_eos: u32,
    /// Each language's codec row, by the model's key (`english`,
    /// `sichuan_dialect`).
    pub languages: BTreeMap<String, u32>,
    /// Each stock speaker's codec row.
    pub speakers: BTreeMap<String, u32>,
    /// The stock speakers that speak a dialect, and its language key: the
    /// language `auto` means for them.
    pub speaker_languages: BTreeMap<String, String>,
}

impl Qwen3TtsPromptConfig {
    /// What the driver says of the stock speakers: each one's dialect, a
    /// language key, as its default language.
    pub fn voices(&self) -> Vec<TtsVoiceInfo> {
        self.speakers
            .keys()
            .map(|name| {
                let mut info = TtsVoiceInfo::stock(name.clone());
                info.default_language = self.speaker_languages.get(name).cloned();
                info
            })
            .collect()
    }

    /// A stock speaker's codec row.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with the
    /// speakers, for one the model does not have.
    pub fn speaker(
        &self,
        name: &str,
    ) -> BunsenResult<u32> {
        self.speakers.get(name).copied().ok_or_else(|| {
            LookupError::missing("speaker", name)
                .with_candidates(self.speakers.keys().cloned())
                .into()
        })
    }

    /// The language row of `key`, a model key, for `speaker`: `auto` is the
    /// speaker's own language (its dialect), or no language row for a
    /// speaker without one; any other key is that language, whoever speaks.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with the
    /// languages, for a key the model does not list.
    pub fn language(
        &self,
        key: &str,
        speaker: Option<&str>,
    ) -> BunsenResult<Option<u32>> {
        if key == "auto" {
            return Ok(speaker
                .and_then(|s| self.speaker_languages.get(s))
                .and_then(|key| self.languages.get(key))
                .copied());
        }
        self.languages.get(key).copied().map(Some).ok_or_else(|| {
            LookupError::missing("language", key)
                .with_candidates(self.languages.keys().cloned())
                .into()
        })
    }
}

/// What a take's prompt is built from: the tokenized text, the instruction,
/// and the resolved speaker and language rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3TtsPromptRequest {
    /// The text, chat-wrapped and tokenized: the role's ids, the text's, the
    /// close's ([`Qwen3TtsPromptConfig::role_ids`],
    /// [`close_ids`](Qwen3TtsPromptConfig::close_ids)).
    pub input_ids: Vec<u32>,

    /// The instruction, chat-wrapped and tokenized, when there is one.
    pub instruct_ids: Option<Vec<u32>>,

    /// The speaker's codec row; `None`, no speaker row (a described voice).
    pub speaker: Option<u32>,

    /// The language's codec row; `None`, no language (`auto` for a speaker
    /// without one).
    pub language: Option<u32>,

    /// Where the text goes.
    pub placement: Qwen3TtsTextPlacement,
}

/// A prompt row's text id and codec id; a row has at least one, and both
/// are summed.
pub type Qwen3TtsPromptRow = (Option<u32>, Option<u32>);

/// A prompt's rows, as ids: each row's text id, codec id, or both, and the
/// text ids generated frames add, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3TtsPromptRows {
    /// Each row's text id and codec id.
    pub rows: Vec<Qwen3TtsPromptRow>,

    /// The text ids each generated frame adds, in order; past the last, the
    /// text pad.
    pub trailing: Vec<u32>,
}

impl Qwen3TtsPromptRows {
    /// The rows of `request`, laid out by `config` (the [module docs](self)).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a text of no
    /// ids between its role and its close, or one that is not wrapped in the
    /// config's chat ids: the role opening with `<|im_start|>` and
    /// `assistant`, the close opening with `<|im_end|>` and ending with the
    /// role again.
    pub fn build(
        config: &Qwen3TtsPromptConfig,
        request: &Qwen3TtsPromptRequest,
    ) -> BunsenResult<Self> {
        let ids = &request.input_ids;
        let [role_n, close_n] = [config.role_ids, config.close_ids];
        if ids.len() < role_n + 1 + close_n {
            return Err(BunsenError::illegal(format!(
                "a Qwen3-TTS text of {} ids: the role's {role_n}, at least one, the close's \
                 {close_n}",
                ids.len()
            )));
        }
        let role = &ids[..role_n];
        let close = &ids[ids.len() - close_n..];
        let opens = role.len() >= 2 && role[..2] == [config.im_start, config.assistant];
        let closes = close.first() == Some(&config.im_end) && close.ends_with(role);
        if !(opens && closes) {
            return Err(BunsenError::illegal(format!(
                "a Qwen3-TTS text wrapped as {role:?} … {close:?}; the role opens with \
                 <|im_start|> and assistant ({}, {}), the close opens with <|im_end|> ({}) and \
                 ends with the role",
                config.im_start, config.assistant, config.im_end
            )));
        }
        let text = &ids[role_n..ids.len() - close_n];
        let mut rows: Vec<Qwen3TtsPromptRow> = Vec::new();
        if let Some(instruct) = &request.instruct_ids {
            rows.extend(instruct.iter().map(|&id| (Some(id), None)));
        }
        rows.extend(role.iter().map(|&id| (Some(id), None)));

        let mut codec = match request.language {
            Some(language) => vec![
                config.codec_think,
                config.codec_think_bos,
                language,
                config.codec_think_eos,
            ],
            None => vec![
                config.codec_nothink,
                config.codec_think_bos,
                config.codec_think_eos,
            ],
        };
        codec.extend(request.speaker);
        codec.extend([config.codec_pad, config.codec_bos]);
        let prefix = codec.len() - 1;
        rows.extend(codec[..prefix].iter().enumerate().map(|(i, &id)| {
            let text = if i + 1 == prefix {
                config.tts_bos
            } else {
                config.tts_pad
            };
            (Some(text), Some(id))
        }));

        let trailing = match request.placement {
            Qwen3TtsTextPlacement::Interleaved => {
                rows.push((Some(text[0]), Some(config.codec_bos)));
                text[1..].iter().copied().chain([config.tts_eos]).collect()
            }
            Qwen3TtsTextPlacement::Upfront => {
                rows.extend(
                    text.iter()
                        .copied()
                        .chain([config.tts_eos])
                        .map(|id| (Some(id), Some(config.codec_pad))),
                );
                rows.push((Some(config.tts_pad), Some(config.codec_bos)));
                vec![config.tts_pad]
            }
        };
        Ok(Self { rows, trailing })
    }

    /// The rows and the trailing text, embedded by `talker`: `[1, L, W]` and
    /// `[1, t, W]`.
    pub fn embed(
        &self,
        talker: &Qwen3TtsTalker,
        device: &Device,
    ) -> Qwen3TtsPrompt {
        let n = self.rows.len();
        let ids = |pick: fn(&Qwen3TtsPromptRow) -> Option<u32>| {
            let values: Vec<i64> = self
                .rows
                .iter()
                .map(|row| pick(row).map_or(0, i64::from))
                .collect();
            let mask: Vec<f32> = self
                .rows
                .iter()
                .map(|row| if pick(row).is_some() { 1.0 } else { 0.0 })
                .collect();
            (
                Tensor::<2, Int>::from_data(TensorData::new(values, [1, n]), device),
                Tensor::<1>::from_floats(mask.as_slice(), device).reshape([1, n, 1]),
            )
        };
        let (text, text_mask) = ids(|row| row.0);
        let (codec, codec_mask) = ids(|row| row.1);
        // The masks in the tables' dtype: a half model's embeddings are not
        // f32.
        let text = talker.text_embed(text);
        let codec = talker.codec_embed(codec);
        let embeds = text.clone() * text_mask.cast(text.dtype())
            + codec.clone() * codec_mask.cast(codec.dtype());
        let t = self.trailing.len();
        let trailing: Vec<i64> = self.trailing.iter().map(|&id| i64::from(id)).collect();
        let trailing =
            talker.text_embed(Tensor::from_data(TensorData::new(trailing, [1, t]), device));
        Qwen3TtsPrompt { embeds, trailing }
    }
}

/// A prompt, embedded: the talker's prefill input and the trailing text.
#[derive(Clone, Debug)]
pub struct Qwen3TtsPrompt {
    /// `[1, L, W]`: the prefill's rows.
    pub embeds: Tensor<3>,

    /// `[1, t, W]`: what generated frames add, in order; past the last,
    /// the text pad.
    pub trailing: Tensor<3>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::BunsenErrorKind,
        kits::speech::tts::qwen3_tts::interop::qwen_tts::*,
    };

    fn config() -> Qwen3TtsPromptConfig {
        let json: Qwen3TtsPromptMirror = serde_json::from_str(
            r#"{
                "tts_pad_token_id": 151671, "tts_bos_token_id": 151672,
                "tts_eos_token_id": 151673, "im_start_token_id": 151644,
                "im_end_token_id": 151645, "assistant_token_id": 77091,
                "talker_config": {
                    "codec_pad_id": 2148, "codec_bos_id": 2149, "codec_eos_token_id": 2150,
                    "codec_think_id": 2154, "codec_nothink_id": 2155,
                    "codec_think_bos_id": 2156, "codec_think_eos_id": 2157,
                    "codec_language_id": {"english": 2050, "chinese": 2055,
                                          "sichuan_dialect": 2062},
                    "spk_id": {"ryan": 3061, "eric": 2875},
                    "spk_is_dialect": {"ryan": false, "eric": "sichuan_dialect"}
                }
            }"#,
        )
        .expect("parse");
        Qwen3TtsPromptMirror::prompt_config(&json).expect("the release's ids")
    }

    /// `<|im_start|>assistant\n`, the release's ids.
    const ROLE: [u32; 3] = [151_644, 77_091, 198];

    /// `<|im_end|>\n`, then the role.
    const CLOSE: [u32; 5] = [151_645, 198, 151_644, 77_091, 198];

    /// `text`, wrapped in the release's role and close.
    fn wrapped(text: &[u32]) -> Vec<u32> {
        [&ROLE[..], text, &CLOSE[..]].concat()
    }

    fn request(placement: Qwen3TtsTextPlacement) -> Qwen3TtsPromptRequest {
        Qwen3TtsPromptRequest {
            input_ids: wrapped(&[10, 11, 12]),
            instruct_ids: None,
            speaker: Some(3061),
            language: Some(2050),
            placement,
        }
    }

    /// The release's speakers and languages, by the model's keys; `auto` is
    /// a dialect speaker's dialect and no row for anyone else.
    #[test]
    fn test_speakers_and_languages() {
        let config = config();
        assert_eq!(config.speaker("ryan").unwrap(), 3061);
        assert_eq!(
            config.speaker("nobody").unwrap_err().kind(),
            BunsenErrorKind::Lookup
        );
        assert_eq!(config.language("auto", Some("eric")).unwrap(), Some(2062));
        assert_eq!(config.language("auto", Some("ryan")).unwrap(), None);
        assert_eq!(config.language("auto", None).unwrap(), None);
        // an explicit language wins over a speaker's dialect
        assert_eq!(
            config.language("chinese", Some("eric")).unwrap(),
            Some(2055)
        );
        assert_eq!(
            config.language("klingon", None).unwrap_err().kind(),
            BunsenErrorKind::Lookup
        );
    }

    /// Interleaved: the role, the codec prefix over the text pad and BOS,
    /// the first text id over the codec BOS; the rest trails, then the EOS.
    #[test]
    fn test_interleaved_rows() {
        let rows =
            Qwen3TtsPromptRows::build(&config(), &request(Qwen3TtsTextPlacement::Interleaved))
                .unwrap();
        let (pad, bos) = (151_671, 151_672);
        assert_eq!(
            rows.rows,
            vec![
                (Some(ROLE[0]), None),
                (Some(ROLE[1]), None),
                (Some(ROLE[2]), None),
                (Some(pad), Some(2154)),
                (Some(pad), Some(2156)),
                (Some(pad), Some(2050)),
                (Some(pad), Some(2157)),
                (Some(pad), Some(3061)),
                (Some(bos), Some(2148)),
                (Some(10), Some(2149)),
            ]
        );
        assert_eq!(rows.trailing, vec![11, 12, 151_673]);
    }

    /// Upfront, without a language or a speaker, after an instruction: the
    /// no-think prefix, every text id and the EOS over the codec pad, the
    /// text pad over the codec BOS; the text pad trails.
    #[test]
    fn test_upfront_rows() {
        let mut request = request(Qwen3TtsTextPlacement::Upfront);
        request.instruct_ids = Some(vec![7, 8]);
        request.speaker = None;
        request.language = None;
        let rows = Qwen3TtsPromptRows::build(&config(), &request).unwrap();
        let (pad, bos, eos) = (151_671, 151_672, 151_673);
        assert_eq!(
            rows.rows,
            vec![
                (Some(7), None),
                (Some(8), None),
                (Some(ROLE[0]), None),
                (Some(ROLE[1]), None),
                (Some(ROLE[2]), None),
                (Some(pad), Some(2155)),
                (Some(pad), Some(2156)),
                (Some(pad), Some(2157)),
                (Some(bos), Some(2148)),
                (Some(10), Some(2148)),
                (Some(11), Some(2148)),
                (Some(12), Some(2148)),
                (Some(eos), Some(2148)),
                (Some(pad), Some(2149)),
            ]
        );
        assert_eq!(rows.trailing, vec![pad]);
    }

    #[test]
    fn test_empty_text_refused() {
        let mut request = request(Qwen3TtsTextPlacement::Interleaved);
        request.input_ids = wrapped(&[]);
        assert_eq!(
            Qwen3TtsPromptRows::build(&config(), &request)
                .unwrap_err()
                .kind(),
            BunsenErrorKind::Illegal
        );
    }

    /// A text not wrapped in the config's chat ids is refused: another
    /// opening, another role, another close, a close without the role.
    #[test]
    fn test_unwrapped_text_refused() {
        let text = [10, 11, 12];
        let other_close = [151_645, 198, 151_644, 872, 198];
        for input_ids in [
            [&[151_643, 77_091, 198][..], &text, &CLOSE].concat(),
            [&[151_644, 872, 198][..], &text, &CLOSE].concat(),
            [&ROLE[..], &text, &[151_643, 198, 151_644, 77_091, 198]].concat(),
            [&ROLE[..], &text, &other_close].concat(),
        ] {
            let mut request = request(Qwen3TtsTextPlacement::Interleaved);
            request.input_ids = input_ids;
            assert_eq!(
                Qwen3TtsPromptRows::build(&config(), &request)
                    .unwrap_err()
                    .kind(),
                BunsenErrorKind::Illegal
            );
        }
    }

    /// The spans are the config's: a role of 2 ids and a close of 3.
    #[test]
    fn test_spans_from_config() {
        let config = config().with_role_ids(2).with_close_ids(3);
        let mut request = request(Qwen3TtsTextPlacement::Upfront);
        request.input_ids = vec![151_644, 77_091, 10, 151_645, 151_644, 77_091];
        request.speaker = None;
        request.language = None;
        let rows = Qwen3TtsPromptRows::build(&config, &request).unwrap();
        assert_eq!(
            &rows.rows[..2],
            [(Some(151_644), None), (Some(77_091), None)]
        );
        assert_eq!(rows.rows[6], (Some(10), Some(2148)));
    }
}

/// The toy prompts against `qwen_tts`' own: every mode's prefill rows and
/// trailing text, as `generate` builds them
/// (`tools/gen_qwen3_tts_toy_fixtures.py`), from the same ids, speaker and
/// language.
#[cfg(all(test, feature = "store"))]
mod toy_tests {

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            speech::tts::qwen3_tts::interop::qwen_tts::*,
            support::interop::transformers::TransformersConfigJson,
        },
        support::testing::{
            ToyFixture,
            assert_io_close,
            performance_device,
            testdata,
        },
    };

    /// The fixture's modes: speaker, language key, whether instructed, and
    /// placement (`MODES` in the generator).
    const MODES: [(&str, Option<&str>, &str, Qwen3TtsTextPlacement); 6] = [
        (
            "english",
            Some("ryan"),
            "english",
            Qwen3TtsTextPlacement::Interleaved,
        ),
        (
            "english_upfront",
            Some("ryan"),
            "english",
            Qwen3TtsTextPlacement::Upfront,
        ),
        (
            "auto",
            Some("ryan"),
            "auto",
            Qwen3TtsTextPlacement::Interleaved,
        ),
        (
            "dialect",
            Some("eric"),
            "auto",
            Qwen3TtsTextPlacement::Interleaved,
        ),
        (
            "instructed",
            Some("ryan"),
            "chinese",
            Qwen3TtsTextPlacement::Upfront,
        ),
        ("described", None, "english", Qwen3TtsTextPlacement::Upfront),
    ];

    fn ids(
        fixture: &ToyFixture,
        key: &str,
    ) -> Vec<u32> {
        fixture
            .io::<2>(key)
            .into_data()
            .iter::<f32>()
            .map(|v| v as u32)
            .collect()
    }

    #[test]
    fn test_toy_prompts() {
        let device = performance_device();
        let dir = testdata!("qwen3_tts");
        let text = std::fs::read_to_string(dir.join("toy.config.json")).expect("read");
        let value: serde_json::Value = serde_json::from_str(&text).expect("parse");
        let json: Qwen3TtsPromptMirror =
            serde_json::from_value(value["model"].clone()).expect("the toy's ids");
        let config = Qwen3TtsPromptMirror::prompt_config(&json).expect("the toy's prompt config");
        let talker_config = talker_config(
            &TransformersConfigJson::parse(&text)
                .and_then(|c| c.section("model"))
                .and_then(|m| m.section("talker_config"))
                .expect("the toy's talker_config"),
        )
        .expect("the toy's talker");
        let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
        let mut talker: Qwen3TtsTalker = talker_config.init(&device);
        load_talker_tensors(&mut talker, fixture.take_weights(), None).expect("the talker");

        for (mode, speaker, language, placement) in MODES {
            let instruct_key = format!("{mode}.instruct_ids");
            let request = Qwen3TtsPromptRequest {
                input_ids: ids(&fixture, &format!("{mode}.input_ids")),
                instruct_ids: fixture
                    .has(&instruct_key)
                    .then(|| ids(&fixture, &instruct_key)),
                speaker: speaker.map(|s| config.speaker(s).expect("a toy speaker")),
                language: config.language(language, speaker).expect("a toy language"),
                placement,
            };
            let prompt = Qwen3TtsPromptRows::build(&config, &request)
                .expect("the rows")
                .embed(&talker, &device);
            assert_io_close(
                &format!("{mode}: prompt"),
                prompt.embeds,
                fixture.io(&format!("{mode}.prompt_embeds")),
            );
            assert_io_close(
                &format!("{mode}: trailing"),
                prompt.trailing,
                fixture.io(&format!("{mode}.trailing_text")),
            );
        }
    }
}
