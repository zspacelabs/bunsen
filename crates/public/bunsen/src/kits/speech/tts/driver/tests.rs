//! The driver over a scripted engine.

use std::sync::Mutex;

use super::*;
use crate::{
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        testing::ErrorMatcher,
    },
    kits::support::text::{
        BunsenLanguageTag,
        TextNormalizationConfig,
    },
};

/// What the scripted engine was handed for a take.
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    text: String,
    voice: String,
    language: Option<String>,
    instruction: Option<String>,
    seed: Option<u64>,
    max_seconds: Option<f64>,
    temperature: Option<f64>,
}

/// An engine of two stock voices and two languages, whose take is the text's
/// length in samples.
#[derive(Debug)]
struct Scripted {
    capabilities: TtsCapabilities,
    seen: Mutex<Vec<Seen>>,
}

impl Scripted {
    fn new(instruction: bool) -> Self {
        Self {
            capabilities: TtsCapabilities {
                languages: vec![
                    TtsLanguage {
                        key: "english".into(),
                        tag: BunsenLanguageTag::parse("en").unwrap(),
                    },
                    TtsLanguage {
                        key: "french".into(),
                        tag: BunsenLanguageTag::parse("fr").unwrap(),
                    },
                ],
                default_voice: Some("ann".into()),
                stock_voices: true,
                described_voices: false,
                instruction,
            },
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl TtsEngine for Scripted {
    type Take = (usize, TtsStop);
    type Voice = String;

    fn capabilities(&self) -> &TtsCapabilities {
        &self.capabilities
    }

    fn sample_rate(&self) -> u32 {
        8
    }

    fn voices(&self) -> Vec<TtsVoiceInfo> {
        vec![TtsVoiceInfo::stock("ann"), TtsVoiceInfo::stock("bob")]
    }

    fn resolve_voice(
        &self,
        choice: &TtsVoiceChoice,
    ) -> BunsenResult<String> {
        match choice {
            TtsVoiceChoice::Default => Ok("ann".into()),
            TtsVoiceChoice::Stock(name) if name == "ann" || name == "bob" => Ok(name.clone()),
            TtsVoiceChoice::Stock(name) => Err(BunsenError::lookup(
                crate::errors::LookupError::missing("stock voice", name),
            )),
            _ => Err(BunsenError::unsupported("no designed voices")),
        }
    }

    fn language_tag(
        &self,
        language: Option<&str>,
        _voice: &String,
    ) -> BunsenResult<Option<BunsenLanguageTag>> {
        let key = language.unwrap_or("english");
        self.capabilities
            .languages
            .iter()
            .find(|l| l.key == key)
            .map(|l| Some(l.tag.clone()))
            .ok_or_else(|| BunsenError::illegal(format!("no language {key:?}")))
    }

    fn generate(
        &self,
        take: &TtsTakeRequest<'_, String>,
    ) -> BunsenResult<(usize, TtsStop)> {
        self.seen.lock().unwrap().push(Seen {
            text: take.text.into(),
            voice: take.voice.clone(),
            language: take.language.map(Into::into),
            instruction: take.instruction.map(Into::into),
            seed: take.seed,
            max_seconds: take.max_seconds,
            temperature: take.sampling.temperature,
        });
        Ok((take.text.len(), TtsStop::End))
    }

    fn render(
        &self,
        take: &(usize, TtsStop),
    ) -> BunsenResult<Vec<f32>> {
        Ok(vec![0.25; take.0])
    }

    fn take_info(
        &self,
        take: &(usize, TtsStop),
    ) -> TtsTakeInfo {
        TtsTakeInfo {
            frames: take.0,
            steps: take.0,
            stop: take.1,
        }
    }
}

fn driver(
    instruction: bool,
    config: TtsDriverConfig,
) -> TtsDriver<Scripted> {
    TtsDriver::new(Scripted::new(instruction), config).unwrap()
}

/// A request reaches the engine whole: its voice resolved, its language,
/// seed, length and sampling; the speech is the engine's render at its rate.
#[test]
fn test_request_reaches_the_engine() {
    let driver = driver(false, TtsDriverConfig::new());
    let request = TtsRequest::new("Hello\nthere")
        .with_voice(TtsVoiceChoice::Stock("bob".into()))
        .with_language("french")
        .with_seed(7)
        .with_max_seconds(2.5)
        .with_sampling(TtsSamplingOverrides {
            temperature: Some(0.5),
            ..TtsSamplingOverrides::default()
        });
    let speech = driver.synthesize(&request).unwrap();
    assert_eq!(
        driver.engine().seen(),
        [Seen {
            text: "Hello\nthere".into(),
            voice: "bob".into(),
            language: Some("french".into()),
            instruction: None,
            seed: Some(7),
            max_seconds: Some(2.5),
            temperature: Some(0.5),
        }]
    );
    assert_eq!(speech.samples.len(), 11);
    assert_eq!(speech.sample_rate, 8);
    assert_eq!(speech.seconds(), 11.0 / 8.0);
    assert_eq!(
        speech.takes,
        [TtsTakeInfo {
            frames: 11,
            steps: 11,
            stop: TtsStop::End
        }]
    );
    assert!(speech.notes.is_empty());

    let default = driver.synthesize(&TtsRequest::new("hi")).unwrap();
    assert_eq!(default.samples.len(), 2);
    assert_eq!(driver.engine().seen()[1].voice, "ann");
}

/// The text stage normalizes the text, and an injected normalizer replaces
/// the config's.
#[test]
fn test_text_stage() {
    let driver = driver(
        false,
        TtsDriverConfig::new().with_normalization(TextNormalizationConfig::CollapseLineBreaks),
    );
    driver.synthesize(&TtsRequest::new("Hello\nthere")).unwrap();
    assert_eq!(driver.engine().seen()[0].text, "Hello there");

    let driver = driver.with_normalizer(TextNormalizationConfig::None.init().unwrap());
    driver.synthesize(&TtsRequest::new("Hello\nthere")).unwrap();
    assert_eq!(driver.engine().seen()[1].text, "Hello\nthere");
}

/// An instruction reaches an engine that takes one. One that takes none
/// refuses it, naming the opt-in, unless the driver strips it, which the
/// speech's notes say.
#[test]
fn test_instruction() {
    let request = TtsRequest::new("hi").with_instruction("whisper");

    let taking = driver(true, TtsDriverConfig::new());
    taking.synthesize(&request).unwrap();
    assert_eq!(
        taking.engine().seen()[0].instruction.as_deref(),
        Some("whisper")
    );

    let refusing = driver(false, TtsDriverConfig::new());
    ErrorMatcher::kind(BunsenErrorKind::Illegal)
        .message_contains("strip_instruction")
        .assert_err(&refusing.synthesize(&request));
    assert!(refusing.engine().seen().is_empty());

    let stripping = driver(false, TtsDriverConfig::new().with_strip_instruction(true));
    let speech = stripping.synthesize(&request).unwrap();
    assert_eq!(stripping.engine().seen()[0].instruction, None);
    assert_eq!(
        speech.notes,
        [TtsStageNote {
            stage: "strip-instruction".into(),
            note: "dropped the instruction \"whisper\"".into(),
        }]
    );
}

/// The engine's refusals pass through: an unknown voice or language, a kind
/// of voice it cannot speak in.
#[test]
fn test_refusals() {
    let driver = driver(false, TtsDriverConfig::new());
    ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(
        &driver.synthesize(&TtsRequest::new("hi").with_voice(TtsVoiceChoice::Stock("eve".into()))),
    );
    ErrorMatcher::kind(BunsenErrorKind::Illegal)
        .message_contains("klingon")
        .assert_err(&driver.synthesize(&TtsRequest::new("hi").with_language("klingon")));
    ErrorMatcher::kind(BunsenErrorKind::Unsupported).assert_err(
        &driver.synthesize(
            &TtsRequest::new("hi").with_voice(TtsVoiceChoice::Described("calm".into())),
        ),
    );
    assert!(driver.engine().seen().is_empty());
}

/// The driver is a synthesizer: the engine's capabilities, rate and voices
/// through the trait object.
#[test]
fn test_synthesizer() {
    let synthesizer: std::sync::Arc<dyn TtsSynthesizer> =
        std::sync::Arc::new(driver(true, TtsDriverConfig::new()));
    assert!(synthesizer.capabilities().instruction);
    assert_eq!(synthesizer.sample_rate(), 8);
    assert_eq!(synthesizer.voices().len(), 2);
    assert_eq!(
        synthesizer
            .synthesize(&TtsRequest::new("abc"))
            .unwrap()
            .samples
            .len(),
        3
    );
}
