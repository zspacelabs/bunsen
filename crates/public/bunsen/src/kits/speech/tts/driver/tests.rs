//! The driver over a scripted engine.

use std::sync::Mutex;

use pretty_assertions::assert_eq;

use super::{
    stream::TtsWholeTakeStream,
    *,
};
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
    /// Whether its stream returns a take a step at a time, a sample a
    /// frame, decoded as the chunk policy groups them; else the default
    /// whole-take stream.
    stepped: bool,
}

/// [`Scripted`]'s own take stream: a frame a step, the frames' samples
/// returned as the chunk policy groups them.
struct ScriptedStream {
    frames: usize,
    generated: usize,
    decoded: usize,
    chunks: TtsChunkPolicy,
}

impl TtsTakeStream for ScriptedStream {
    fn step(&mut self) -> BunsenResult<TtsStreamStep> {
        self.generated += 1;
        let done = self.generated == self.frames;
        let ready = self.generated - self.decoded;
        let samples = if done || ready >= self.chunks.need(self.decoded) {
            self.decoded = self.generated;
            vec![0.25; ready]
        } else {
            Vec::new()
        };
        Ok(TtsStreamStep {
            samples,
            end: done.then_some(TtsTakeInfo {
                frames: self.frames,
                steps: self.frames,
                stop: TtsStop::End,
            }),
        })
    }
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
                clone_modes: Vec::new(),
            },
            seen: Mutex::new(Vec::new()),
            stepped: false,
        }
    }

    fn stepped(instruction: bool) -> Self {
        Self {
            stepped: true,
            ..Self::new(instruction)
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

    fn start_stream<'a>(
        &'a self,
        take: &TtsTakeRequest<'_, String>,
        chunks: TtsChunkPolicy,
    ) -> BunsenResult<Box<dyn TtsTakeStream + 'a>> {
        if !self.stepped {
            return Ok(Box::new(TtsWholeTakeStream {
                engine: self,
                take: Some(self.generate(take)?),
            }));
        }
        let (frames, _) = self.generate(take)?;
        Ok(Box::new(ScriptedStream {
            frames,
            generated: 0,
            decoded: 0,
            chunks,
        }))
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

    let driver = driver.with_normalizer(TextNormalizationConfig::None.try_init().unwrap());
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
    ErrorMatcher::illegal()
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
    ErrorMatcher::lookup().assert_err(
        &driver.synthesize(&TtsRequest::new("hi").with_voice(TtsVoiceChoice::Stock("eve".into()))),
    );
    ErrorMatcher::illegal()
        .message_contains("klingon")
        .assert_err(&driver.synthesize(&TtsRequest::new("hi").with_language("klingon")));
    ErrorMatcher::unsupported().assert_err(
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

/// A long text is several takes, split at its sentences under the take
/// limit (the engine's when lower), each seeded in turn and joined by the
/// gap; unsplit, one take; no text, refused.
#[test]
fn test_takes() {
    let config = TtsDriverConfig::new()
        .with_max_take_chars(12)
        .with_take_gap_seconds(0.25);
    let driver = driver(false, config.clone());
    let text = "One two. Three four. Five.";
    assert_eq!(driver.takes(text), ["One two.", "Three four.", "Five."]);
    let speech = driver
        .synthesize(&TtsRequest::new(text).with_seed(10))
        .unwrap();
    let seen = driver.engine().seen();
    assert_eq!(
        seen.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        ["One two.", "Three four.", "Five."]
    );
    assert_eq!(
        seen.iter().map(|s| s.seed).collect::<Vec<_>>(),
        [Some(10), Some(11), Some(12)]
    );
    assert_eq!(speech.takes.len(), 3);
    // 8 + 11 + 5 samples, and two gaps of 2 (0.25 s at 8 Hz).
    assert_eq!(speech.samples.len(), 24 + 2 * 2);
    assert_eq!(&speech.samples[8..10], [0.0, 0.0]);

    let unsplit = self::driver(false, config.with_split(false));
    assert_eq!(unsplit.takes(text), [text]);
    assert_eq!(
        unsplit
            .synthesize(&TtsRequest::new(text))
            .unwrap()
            .takes
            .len(),
        1
    );

    ErrorMatcher::illegal()
        .message_contains("no text")
        .assert_err(&driver.synthesize(&TtsRequest::new("  ")));
    for bad in [
        TtsDriverConfig::new().with_max_take_chars(0),
        TtsDriverConfig::new().with_take_gap_seconds(-1.0),
    ] {
        ErrorMatcher::illegal().assert_err(&TtsDriver::new(Scripted::new(false), bad).map(|_| ()));
    }
}

/// A reference of one clip, with `transcript`.
fn reference(transcript: Option<&str>) -> TtsVoiceReference {
    let record = TtsClipRecord {
        file: "a.wav".into(),
        sha256: "00".into(),
        sample_rate: 8,
        channels: 1,
        samples: 8,
        edits: Vec::new(),
    };
    TtsVoiceReference {
        record: TtsVoiceReferenceRecord {
            version: TTS_VOICE_REFERENCE_VERSION,
            name: Some("ada".into()),
            transcript: transcript.map(Into::into),
            language: None,
            gender: None,
            style: None,
            created_unix: None,
            clips: vec![record.clone()],
        },
        clips: vec![TtsReferenceClip {
            record,
            channels: vec![vec![0.0; 8]],
        }],
        digest: "ab".into(),
    }
}

/// The clone mode: the one asked for, if the reference satisfies it; else the
/// first it satisfies; refusals name the need.
#[test]
fn test_clone_mode() {
    let capabilities = TtsCapabilities {
        clone_modes: vec![
            TtsCloneMode {
                name: "icl".into(),
                needs_transcript: true,
            },
            TtsCloneMode::clip("xvector"),
        ],
        ..TtsCapabilities::default()
    };
    let with = reference(Some("hello"));
    let without = reference(None);
    assert_eq!(capabilities.clone_mode(&with, None).unwrap().name, "icl");
    assert_eq!(
        capabilities.clone_mode(&without, None).unwrap().name,
        "xvector"
    );
    assert_eq!(
        capabilities
            .clone_mode(&with, Some("xvector"))
            .unwrap()
            .name,
        "xvector"
    );
    let err = capabilities.clone_mode(&without, Some("icl")).unwrap_err();
    ErrorMatcher::illegal()
        .display_contains("transcript")
        .assert(&err);
    assert_eq!(
        capabilities
            .clone_mode(&with, Some("nope"))
            .unwrap_err()
            .kind(),
        BunsenErrorKind::Lookup
    );
    let mut empty = without.clone();
    empty.clips.clear();
    assert_eq!(
        capabilities.clone_mode(&empty, None).unwrap_err().kind(),
        BunsenErrorKind::Illegal
    );
    assert_eq!(
        TtsCapabilities::default()
            .clone_mode(&with, None)
            .unwrap_err()
            .kind(),
        BunsenErrorKind::Unsupported
    );
}

/// An engine that clones none refuses to make a voice.
#[test]
fn test_build_voice_unsupported() {
    let driver = TtsDriver::new(Scripted::new(false), TtsDriverConfig::new()).unwrap();
    let err = TtsSynthesizer::build_voice(&driver, &reference(None), None).unwrap_err();
    ErrorMatcher::unsupported().assert(&err);
}

/// The chunk policy's ramp: 1, 2, 4, 8, then 15 at a time; a policy of no
/// first frames, or a max under its first, is refused.
#[test]
fn test_chunk_policy() {
    let policy = TtsChunkPolicy::new();
    assert_eq!(
        [0, 1, 3, 7, 15, 30].map(|decoded| policy.need(decoded)),
        [1, 2, 4, 8, 15, 15]
    );
    policy.check().unwrap();
    for bad in [
        TtsChunkPolicy::new().with_first(0),
        TtsChunkPolicy::new().with_first(4).with_max(3),
    ] {
        ErrorMatcher::illegal().assert_err(&bad.check());
    }
}

/// A stream's events, gathered, are the synthesis, for the default
/// whole-take stream and a stepped one; the stepped one's audio comes as
/// the policy groups its frames, each take between its start and end, the
/// gap before every take but the first.
#[test]
fn test_stream_is_the_synthesis() {
    let config = TtsDriverConfig::new()
        .with_max_take_chars(12)
        .with_take_gap_seconds(0.25);
    let text = "One two. Three four. Five.";
    for stepped in [false, true] {
        let engine = if stepped {
            Scripted::stepped(false)
        } else {
            Scripted::new(false)
        };
        let driver = TtsDriver::new(engine, config.clone()).unwrap();
        let request = TtsRequest::new(text).with_seed(10);
        let whole = driver.synthesize(&request).unwrap();
        let mut stream = driver.stream(&request, TtsChunkPolicy::new()).unwrap();
        assert_eq!(
            TtsSpeech::from_stream(&mut stream).unwrap(),
            whole,
            "stepped {stepped}"
        );
        // Each take saw the same seed as the whole synthesis gave it.
        let seeds: Vec<_> = driver.engine().seen().iter().map(|s| s.seed).collect();
        assert_eq!(seeds[..3], seeds[3..]);
    }

    let driver = TtsDriver::new(Scripted::stepped(false), config).unwrap();
    let mut stream = driver
        .stream(&TtsRequest::new(text), TtsChunkPolicy::new())
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next_event().unwrap() {
        events.push(match event {
            TtsEvent::TakeStart { index, .. } => format!("start {index}"),
            TtsEvent::Audio(samples) if samples.iter().all(|&x| x == 0.0) => {
                format!("gap {}", samples.len())
            }
            TtsEvent::Audio(samples) => format!("audio {}", samples.len()),
            TtsEvent::TakeEnd { index, .. } => format!("end {index}"),
            TtsEvent::Notes(_) => "notes".into(),
        });
    }
    // "One two." is 8 frames: 1, 2, 4, then the last 1 at the end.
    assert_eq!(
        events,
        [
            "start 0", "audio 1", "audio 2", "audio 4", "audio 1", "end 0", "gap 2", "start 1",
            "audio 1", "audio 2", "audio 4", "audio 4", "end 1", "gap 2", "start 2", "audio 1",
            "audio 2", "audio 2", "end 2",
        ]
    );
}

/// Through the synthesizer as a trait object, the stream is the driver's;
/// a stripped instruction's note comes first.
#[test]
fn test_synthesizer_stream() {
    let driver = TtsDriver::new(
        Scripted::stepped(false),
        TtsDriverConfig::new().with_strip_instruction(true),
    )
    .unwrap();
    let synthesizer: &dyn TtsSynthesizer = &driver;
    let request = TtsRequest::new("hello").with_instruction("whisper");
    let mut stream = synthesizer.stream(&request, TtsChunkPolicy::new()).unwrap();
    assert_eq!(stream.sample_rate(), 8);
    assert!(matches!(
        stream.next_event().unwrap(),
        Some(TtsEvent::Notes(notes)) if notes.len() == 1
    ));
    let rest = TtsSpeech::from_stream(&mut *stream).unwrap();
    assert_eq!(
        rest.samples,
        synthesizer.synthesize(&request).unwrap().samples
    );

    ErrorMatcher::illegal().assert_err(
        &synthesizer
            .stream(&request, TtsChunkPolicy::new().with_first(0))
            .map(|_| ()),
    );
}
