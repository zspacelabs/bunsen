//! bunsen's codec decoder in float32 against `qwen_tts`' own, from fixed
//! codes, through upstream's chunked decode, whole and as a stream.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
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

/// Every case's codes, decoded by `decode`, against the reference: the
/// audio's length and RMS, its windows and its sample within
/// [`AUDIO_ABS`].
fn check_cases(
    label: &str,
    mut decode: impl FnMut(&Qwen3TtsConstruct, Tensor<3, Int>) -> Tensor<3>,
) {
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
        let audio = decode(&construct, codes)
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap();
        let stats = fixture.doubles(&at("audio.stats"));
        assert_eq!(audio.len() as f64, stats[0], "{label} {case}: samples");
        let rms =
            (audio.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / audio.len() as f64).sqrt();
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
        eprintln!("{label} {case}: off by at most {gap:.2e}, RMS rel {rms_rel:.1e}");
        assert!(gap <= AUDIO_ABS, "{label} {case}: {gap}");
        assert!(rms_rel <= RMS_REL, "{label} {case}: RMS {rms_rel}");
    }
}

/// The pushes a stream is fed in: upstream ZONOS3's ramp, 1, 2, 4, 8, then
/// 15 frames at a time.
const STREAM_PUSHES: [usize; 5] = [1, 2, 4, 8, 15];

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// Every case through the render's chunked decode; the long case's
    /// windows across the first chunk's end too.
    #[test]
    #[serial_test::serial]
    fn codec() {
        check_cases("render", |construct, codes| {
            construct.codec.chunked_decode(codes, &construct.decode)
        });
    }

    /// Every case through the chunked stream, pushed in [`STREAM_PUSHES`]
    /// (the last repeated): the render's audio, as upstream decodes it.
    #[test]
    #[serial_test::serial]
    fn codec_stream() {
        check_cases("stream", |construct, codes| {
            let t = codes.dims()[2];
            let mut ctx = construct.codec.new_chunked_stream_context(construct.decode);
            let (mut at, mut i, mut parts) = (0, 0, Vec::new());
            while at < t {
                let n = STREAM_PUSHES[i.min(STREAM_PUSHES.len() - 1)].min(t - at);
                parts.push(
                    construct.codec.forward_chunked_stream(
                        codes.clone().slice(s![.., .., at..at + n]),
                        &mut ctx,
                    ),
                );
                at += n;
                i += 1;
            }
            Tensor::cat(parts, 2)
        });
    }
}
