//! bunsen's speaker encoder against the `-hf` port's, in float32: each stock
//! voice's codec render, from the clip at 44.1 kHz to the embedding.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    cosine_similarity,
    performance_device,
    tensor_to_vec_as,
};

use super::*;
use crate::fixtures::SPEAKER;

// Measured on wgpu f32 (2026-11-21), over the three voices: the stats within
// 1.8e-7, the strided features within 8.3e-4 (the quietest bins, float32's
// spectrum rounding, as `KaldiFbank`'s own test finds), the pooling within
// 1.2e-5 of its largest, the embedding's cosine 1.0000000 to seven places, the
// stock cosine within 3.7e-6 of the reference's.

/// The most the render's, the prepared clip's and the features' sums and RMS
/// may stray, relative to the RMS times the count (the sums) or to the RMS.
pub const STATS_REL: f64 = 1e-5;

/// The most a strided feature may stray from the reference's, absolute (a
/// log energy).
pub const FEATURE_ABS: f32 = 2e-3;

/// The most a pooled statistic may stray, relative to the largest.
pub const POOLED_REL: f32 = 1e-4;

/// The least cosine between bunsen's embedding and the reference's.
pub const EMBEDDING_COSINE: f64 = 0.999_999;

/// The most the cosine to the voice's shipped embedding may stray from the
/// reference's. (The cosine itself is no bound: the shipped embeddings come
/// from upstream's source recordings, not from these renders, and stand at
/// 0.86, 0.82 and 0.52.)
pub const STOCK_COSINE_ABS: f64 = 1e-4;

/// `x`'s count, sum, absolute sum and RMS.
fn stats(x: &[f32]) -> [f64; 4] {
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|&v| f64::from(v)).sum();
    let abs: f64 = x.iter().map(|&v| f64::from(v).abs()).sum();
    let rms = (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n).sqrt();
    [n, sum, abs, rms]
}

/// `actual`'s stats against `expected`'s: the counts equal, the sums and RMS
/// within [`STATS_REL`]; the worst relative gap.
fn stats_rel(
    what: &str,
    actual: &[f32],
    expected: &[f64],
) -> f64 {
    let s = stats(actual);
    assert_eq!(s[0], expected[0], "{what}: the count");
    let scale = expected[3] * s[0];
    let rel = [
        (s[1] - expected[1]).abs() / scale,
        (s[2] - expected[2]).abs() / scale,
        (s[3] - expected[3]).abs() / expected[3],
    ];
    let worst = rel.iter().copied().fold(0.0, f64::max);
    assert!(worst <= STATS_REL, "{what}: stats {rel:?}");
    worst
}

mod zonos3_mini {
    use super::*;

    /// Each voice: bunsen's codec renders its stock codes; the encoder's
    /// prepared clip, features, pooling and embedding against the
    /// reference's; and the embedding's cosine to the shipped one, against
    /// the reference's cosine.
    #[test]
    #[serial_test::serial]
    fn speaker_encoder() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let encoder = &construct
            .speaker
            .as_ref()
            .expect("the release's speaker encoder")
            .encoder;
        let fixture = FixtureFile::load(SPEAKER);
        let stride: usize = fixture.metadata("sample_stride").parse().unwrap();
        let voices: Vec<String> = fixture.metadata_json("voices");
        for voice in voices {
            let at = |k: &str| format!("{voice}.{k}");
            let stock = &construct.voices[&voice];
            let frames = stock.codes.len();
            let codes = Tensor::<2, Int>::from_data(
                TensorData::new(
                    stock
                        .codes
                        .iter()
                        .flatten()
                        .map(|&c| i64::from(c))
                        .collect::<Vec<_>>(),
                    [frames, CODEBOOKS],
                ),
                &device,
            )
            .transpose()
            .unsqueeze::<3>();
            let audio = tensor_to_vec_as::<f32, _>(&construct.codec.forward(codes));
            let audio_rel = stats_rel(&at("audio"), &audio, &fixture.doubles(&at("audio.stats")));

            let prepared = encoder.prepare(&audio, construct.sample_rate as usize);
            let prepared_rel = stats_rel(
                &at("prepared"),
                &prepared,
                &fixture.doubles(&at("prepared.stats")),
            );

            let features = encoder.features(&prepared).expect("the features");
            let trace = encoder.forward_trace(features.clone());
            let features = tensor_to_vec_as::<f32, _>(&features);
            let features_rel = stats_rel(
                &at("features"),
                &features,
                &fixture.doubles(&at("features.stats")),
            );
            let feature_gap = features
                .iter()
                .step_by(stride)
                .zip(fixture.floats(&at("features.sample")))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);

            let pooled = tensor_to_vec_as::<f32, _>(&trace.pooled);
            let expected = fixture.floats(&at("pooled"));
            let peak = expected.iter().map(|v| v.abs()).fold(0.0, f32::max);
            let pooled_rel = pooled
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
                / peak;

            let embedding = tensor_to_vec_as::<f32, _>(&trace.embedding);
            let embedding_cosine = cosine_similarity(&embedding, &fixture.floats(&at("embedding")));
            let stock_cosine = cosine_similarity(&embedding, &stock.embedding);
            let stock_gap = (stock_cosine - fixture.doubles(&at("stock_cosine"))[0]).abs();

            eprintln!(
                "{voice}: audio {audio_rel:.2e}, prepared {prepared_rel:.2e}, features \
                 {features_rel:.2e} (sample max {feature_gap:.2e}), pooled {pooled_rel:.2e}, \
                 embedding cosine {embedding_cosine:.7}, stock cosine {stock_cosine:.4} (gap \
                 {stock_gap:.2e})"
            );
            assert!(
                feature_gap <= FEATURE_ABS,
                "{voice}: features {feature_gap}"
            );
            assert!(pooled_rel <= POOLED_REL, "{voice}: pooled {pooled_rel}");
            assert!(
                embedding_cosine >= EMBEDDING_COSINE,
                "{voice}: embedding cosine {embedding_cosine}"
            );
            assert!(
                stock_gap <= STOCK_COSINE_ABS,
                "{voice}: stock cosine {stock_gap}"
            );
        }
    }
}
