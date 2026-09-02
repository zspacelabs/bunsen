//! The Whisper decode loop at `base.en`'s shapes, cold and warm.
//!
//! Random weights: the cost is in the shapes, and a stop token that is
//! never emitted makes every decode run to its cap, so runs are
//! comparable. The encoder is measured on its own; the decode cases run on
//! encoder output already in hand, so they time the loop and its caches
//! and nothing else. The first call of each case is timed once and printed
//! as `cold`, because this backend's shape-keyed autotune makes a cold call
//! and a warm one differ by more than any copy the towers could remove;
//! criterion's numbers are the warm ones.
//!
//! Run with the backend the kit is meant for:
//! `cargo bench -p bunsen --features wgpu --bench whisper_decode`.

use std::time::Instant;

use bunsen::{
    burner::module::ModuleInit,
    kits::speech::whisper::{
        DecodeConfig,
        Whisper,
        WhisperApiConfig,
    },
    support::testing::PerformanceBackend,
};
use burn::{
    Tensor,
    prelude::Device,
    tensor::Distribution,
};
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

type B = PerformanceBackend;

/// `base.en`: 80 mels, a 51 864 vocabulary, `d_model` 512 in eight heads,
/// six layers each side, a 3000-frame window, a 448-token context.
fn base_en(device: &Device<B>) -> Whisper<B> {
    WhisperApiConfig::new(80, 51_864, 512, 3000, 6, 448, 6).init(device)
}

/// A stop token no decode emits, so every case runs to its cap.
const NEVER: i64 = -1;

fn bench_whisper_decode(c: &mut Criterion) {
    let device = Default::default();
    let model = base_en(&device);
    let mels: Tensor<B, 3> = Tensor::random([1, 80, 3000], Distribution::Default, &device);

    let cold = Instant::now();
    let xa = model.forward_encoder(mels.clone());
    let _ = xa.clone().to_data();
    eprintln!("cold encode: {:?}", cold.elapsed());
    c.bench_function("encode", |b| {
        b.iter(|| model.forward_encoder(mels.clone()).to_data())
    });

    let prompt = vec![50_257, 50_362];
    let cases = [
        ("decode greedy 32", 1, 1, 32, true),
        ("decode greedy 224", 1, 1, 224, true),
        ("decode beam5 32", 5, 1, 32, true),
        ("decode beam5 32 materialized cross-KV", 5, 1, 32, false),
        ("decode batch4 greedy 32", 1, 4, 32, true),
    ];
    for (name, beam, batch, tokens, shared) in cases {
        let config = DecodeConfig::new(prompt.clone(), NEVER)
            .with_max_tokens(tokens)
            .with_beam_size(beam)
            .with_shared_cross_kv(shared);
        let features = if batch == 1 {
            xa.clone()
        } else {
            Tensor::cat(vec![xa.clone(); batch], 0)
        };

        let cold = Instant::now();
        let out = model.decode_features(features.clone(), &config, &[]);
        eprintln!(
            "cold {name}: {:?} ({} rows x {} tokens)",
            cold.elapsed(),
            out.len(),
            out[0].len()
        );

        c.bench_function(name, |b| {
            b.iter(|| model.decode_features(features.clone(), &config, &[]))
        });
    }
}

criterion_group! {
    name = whisper_decode;
    config = Criterion::default().sample_size(10);
    targets = bench_whisper_decode
}
criterion_main!(whisper_decode);
