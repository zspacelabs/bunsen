//! bunsen's codec decoder in float32 against `qwen_tts`' own, from fixed
//! codes, through upstream's chunked decode.

use bunsen::{
    kits::speech::tts::qwen3_tts::codec::{
        QWEN3_TTS_DECODE_CHUNK,
        QWEN3_TTS_DECODE_CONTEXT,
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;
use crate::fixtures::{
    CODEC,
    SAMPLE_STRIDE,
};

/// The most a sample may stray from the reference's. Measured: at most
/// 3.5e-6 over the takes, 2.2e-5 over the long case.
pub const AUDIO_ABS: f32 = 2e-4;

/// The most the audio's RMS may stray, relative to the reference's.
/// Measured: at most 7.6e-7.
pub const RMS_REL: f64 = 1e-5;

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// Every case: the audio's length and RMS, its windows and its sample
    /// within [`AUDIO_ABS`]; the long case's windows across the first
    /// chunk's end too.
    #[test]
    #[serial_test::serial]
    fn codec() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let fixture = FixtureFile::load(CODEC);
        let cases: Vec<String> = fixture.metadata_json("cases");
        for case in cases {
            let at = |k: &str| format!("{case}.{k}");
            let codes = fixture.ints(&at("codes"));
            let frames = codes.len() / fixtures::CODE_GROUPS;
            let codes = Tensor::<2, Int>::from_data(
                TensorData::new(codes, [frames, fixtures::CODE_GROUPS]),
                &device,
            )
            .transpose()
            .unsqueeze_dim(0);
            let audio = construct.codec.chunked_decode(
                codes,
                QWEN3_TTS_DECODE_CHUNK,
                QWEN3_TTS_DECODE_CONTEXT,
            );
            let audio = audio
                .into_data()
                .convert::<f32>()
                .try_into_vec::<f32>()
                .unwrap();
            let stats = fixture.doubles(&at("audio.stats"));
            assert_eq!(audio.len() as f64, stats[0], "{case}: samples");
            let rms = (audio.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                / audio.len() as f64)
                .sqrt();
            let rms_rel = (rms - stats[3]).abs() / stats[3];

            let window = fixture.shape(&at("audio.windows"))[1];
            let windows = fixture.floats(&at("audio.windows"));
            let mut gap = 0.0f32;
            for (w, &start) in fixture.ints(&at("audio.window_starts")).iter().enumerate() {
                let start = start as usize;
                for (i, &want) in windows[w * window..(w + 1) * window].iter().enumerate() {
                    gap = gap.max((audio[start + i] - want).abs());
                }
            }
            for (&got, &want) in audio
                .iter()
                .step_by(SAMPLE_STRIDE)
                .zip(&fixture.floats(&at("audio.sample")))
            {
                gap = gap.max((got - want).abs());
            }
            eprintln!("{case}: off by at most {gap:.2e}, RMS rel {rms_rel:.1e}");
            assert!(gap <= AUDIO_ABS, "{case}: {gap}");
            assert!(rms_rel <= RMS_REL, "{case}: RMS {rms_rel}");
        }
    }
}
