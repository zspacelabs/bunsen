//! bunsen's codec against the `-hf` codec, in float32: the audio from each
//! case's fixed codes, decoded whole and as a stream.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
    tensor_to_vec_as,
};

use super::*;
use crate::fixtures::CODEC;

/// The most a sample may stray from the `-hf` codec's.
pub const SAMPLE_ABS: f32 = 1e-5;

/// The most the audio's sums and RMS may stray, relative to the RMS times
/// the sample count (the sums) or to the RMS.
pub const STATS_REL: f64 = 3e-5;

/// Every case's codes, decoded by `decode`, against the `-hf` codec's:
/// the audio's length, its sums and RMS, its strided sample and its
/// contiguous windows.
fn check_cases(
    label: &str,
    mut decode: impl FnMut(&Zonos3Construct, Tensor<3, Int>) -> Tensor<2>,
) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let construct = load_construct(&device);
    let fixture = FixtureFile::load(CODEC);
    let stride: usize = fixture.metadata("sample_stride").parse().unwrap();
    let cases: Vec<String> = fixture.metadata_json("cases");
    for case in cases {
        let at = |k: &str| format!("{case}.{k}");
        let frames = fixture.shape(&at("codes"))[0];
        let codes = Tensor::<2, Int>::from_data(
            TensorData::new(fixture.ints(&at("codes")), [frames, CODEBOOKS]),
            &device,
        )
        .transpose()
        .unsqueeze::<3>();
        let audio: Vec<f32> =
            tensor_to_vec_as::<f32, _>(&decode(&construct, codes).cast(FloatDType::F32));
        let stats = fixture.doubles(&at("audio.stats"));
        assert_eq!(audio.len() as f64, stats[0], "{label} {case}: the samples");
        let n = audio.len() as f64;
        let sum: f64 = audio.iter().map(|&x| f64::from(x)).sum();
        let abs_sum: f64 = audio.iter().map(|&x| f64::from(x).abs()).sum();
        let rms = (audio.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n).sqrt();
        let scale = stats[3] * n;
        let rel = [
            (sum - stats[1]).abs() / scale,
            (abs_sum - stats[2]).abs() / scale,
            (rms - stats[3]).abs() / stats[3],
        ];

        let mut diffs: Vec<f32> = audio
            .iter()
            .step_by(stride)
            .zip(fixture.floats(&at("audio.sample")))
            .map(|(a, b)| (a - b).abs())
            .collect();
        let windows = fixture.floats(&at("audio.windows"));
        let width = fixture.shape(&at("audio.windows"))[1];
        for (start, window) in fixture
            .ints(&at("audio.window_starts"))
            .into_iter()
            .zip(windows.chunks(width))
        {
            let start = start as usize;
            diffs.extend(
                audio[start..start + width]
                    .iter()
                    .zip(window)
                    .map(|(a, b)| (a - b).abs()),
            );
        }
        let worst = diffs.iter().copied().fold(0.0, f32::max);
        eprintln!(
            "{label} {case}: rms {:.3e}, stats rel {rel:?}, sample max {worst:.2e}",
            stats[3]
        );
        assert!(
            rel.iter().all(|&r| r <= STATS_REL),
            "{label} {case}: stats {rel:?}"
        );
        assert!(worst <= SAMPLE_ABS, "{label} {case}: sample {worst}");
    }
}

/// The pushes a stream is fed in: upstream's prefix ramp, 1, 2, 4, 8, then
/// 15 frames at a time.
const STREAM_PUSHES: [usize; 5] = [1, 2, 4, 8, 15];

mod zonos3_mini {
    use super::*;

    /// Every case's audio, decoded whole.
    #[test]
    #[serial_test::serial]
    fn codec() {
        check_cases("whole", |construct, codes| construct.codec.forward(codes));
    }

    /// Every case's audio through the stream, pushed in [`STREAM_PUSHES`]
    /// (the last repeated), then finished: the whole decode's.
    #[test]
    #[serial_test::serial]
    fn codec_stream() {
        check_cases("stream", |construct, codes| {
            let t = codes.dims()[2];
            let codec = &construct.codec;
            let mut ctx = codec.new_stream_context(1, &codes.device());
            let (mut at, mut i, mut parts) = (0, 0, Vec::new());
            while at < t {
                let n = STREAM_PUSHES[i.min(STREAM_PUSHES.len() - 1)].min(t - at);
                parts.extend(
                    codec.forward_stream(codes.clone().slice(s![.., .., at..at + n]), &mut ctx),
                );
                at += n;
                i += 1;
            }
            parts.extend(codec.finish_stream(&mut ctx));
            Tensor::cat(parts, 1)
        });
    }
}
