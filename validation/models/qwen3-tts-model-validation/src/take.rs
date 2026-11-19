//! bunsen's greedy take against `qwen_tts`' own, code for code.

use bunsen::{
    kits::speech::tts::qwen3_tts::{
        Qwen3TtsStop,
        Qwen3TtsTakeConfig,
        prompt::Qwen3TtsPromptRows,
        sampler::Qwen3TtsSamplerConfig,
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
            assert_eq!(take.stop, Qwen3TtsStop::MaxFrames, "{name}");
        }
    }
}
