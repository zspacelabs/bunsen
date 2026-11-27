//! bunsen's greedy take against `qwen_tts`' own, code for code.

use bunsen::{
    kits::speech::tts::{
        TtsStop,
        driver::{
            TtsEngine,
            TtsSamplingOverrides,
            TtsTakeRequest,
            TtsVoiceChoice,
        },
        qwen3_tts::{
            blocks::Qwen3TtsTakeConfig,
            driver::{
                Qwen3TtsEngine,
                Qwen3TtsEngineConfig,
            },
            prompt::{
                Qwen3TtsPromptRows,
                Qwen3TtsTextPlacement,
            },
            sampler::Qwen3TtsSamplerConfig,
        },
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;
use crate::fixtures::{
    FRAMES,
    REFERENCE,
    REQUESTS,
};

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// Every request's greedy take (the release's repetition penalty, the
    /// argmax for both samplers) is upstream's: its frames, code for code.
    /// The TTS engine lays out each request's prompt as upstream does: its
    /// chat-wrapped ids, speaker and language rows, and placement.
    #[test]
    #[serial_test::serial]
    fn take() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let reference = FixtureFile::load(REFERENCE);
        let greedy = Qwen3TtsTakeConfig::new()
            .with_talker(
                Qwen3TtsSamplerConfig::new()
                    .with_sample(false)
                    .with_repetition_penalty(1.05),
            )
            .with_code_predictor(Qwen3TtsSamplerConfig::new().with_sample(false))
            .with_max_frames(FRAMES);
        for name in REQUESTS {
            let request = reference_request(&construct.prompt, &reference, name);
            let facts = reference.request(name);
            let placement = if facts.non_streaming {
                Qwen3TtsTextPlacement::Upfront
            } else {
                Qwen3TtsTextPlacement::Interleaved
            };
            let engine = Qwen3TtsEngine::new(
                construct.clone(),
                construct.tokenizer.load().expect("the release's tokenizer"),
                Qwen3TtsEngineConfig::new().with_placement(placement),
            )
            .expect("the engine");
            let voice = engine
                .resolve_voice(&TtsVoiceChoice::Stock(facts.speaker.clone()))
                .expect("a stock voice");
            let sampling = TtsSamplingOverrides::default();
            let engine_request = engine
                .prompt_request(&TtsTakeRequest {
                    text: &facts.text,
                    voice: &voice,
                    language: Some(&facts.language),
                    instruction: None,
                    seed: None,
                    max_seconds: None,
                    sampling: &sampling,
                })
                .expect("the engine's prompt request");
            assert_eq!(engine_request, request, "{name}: the engine's prompt");
            let prompt = Qwen3TtsPromptRows::build(&construct.prompt, &request)
                .expect("the rows")
                .embed(&construct.model.talker, &device);
            let take = construct
                .model
                .generate_codes(&prompt, &construct.prompt, &greedy);
            let expected = reference_codes(&reference, name);
            let agree = take
                .frames
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .filter(|(a, b)| a == b)
                .count();
            eprintln!("{name}: {agree}/{} codes agree", FRAMES * 16);
            assert_eq!(take.frames, expected, "{name}");
            assert_eq!(take.stop, TtsStop::Budget, "{name}");
        }
    }
}
