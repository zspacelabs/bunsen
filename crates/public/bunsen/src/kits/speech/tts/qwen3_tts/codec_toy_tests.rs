//! The toy codec decoder against `qwen_tts`' own.
//!
//! `testdata/qwen3_tts/toy.safetensors` holds a `Qwen3TTSTokenizerV2Decoder`
//! at toy sizes under the codec file's names (`decoder.*`), and its answers
//! over 20 frames of 4 codebooks, computed in float64
//! (`tools/gen_qwen3_tts_toy_fixtures.py`): the quantizer's latent, the
//! pre-conv, the transformer, each upsampler and vocoder step, the audio,
//! and `chunked_decode`'s audio in chunks of 8 frames with 3 of context. The
//! geometry comes from `toy.config.json`'s `codec.decoder_config`, and the
//! weights load through [`load_codec_decoder_tensors`].

use burn::{
    prelude::*,
    tensor::Tolerance,
};

use crate::{
    burner::module::ModuleInit,
    kits::speech::tts::qwen3_tts::{
        blocks::{
            Qwen3TtsCodecDecoder,
            Qwen3TtsCodecDecoderContractConfig,
            Qwen3TtsCodecDecoderMeta,
            Qwen3TtsDecodeConfig,
        },
        interop::qwen_tts::*,
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        assert_io_close_within,
        performance_device,
        testdata,
    },
};

/// The tolerance of the vocoder's steps after its input convolution, and
/// of the audio: the toy vocoder's activations reach 30, inside `SnakeBeta`'s
/// `sin`, so float32 drifts there. Upstream's own decoder in float32 is off
/// its float64 answers by `6.6e-5` after the first block, `1.1e-3` after the
/// second, `1.9e-3` after the activation and `6.3e-5` on the audio; these
/// are about twice that.
fn vocoder_tolerance(step: usize) -> Tolerance<f32> {
    match step {
        0 => Tolerance::rel_abs(1e-3, 1e-4),
        1 => Tolerance::rel_abs(1e-3, 2e-4),
        4 => Tolerance::rel_abs(1e-3, 2e-4),
        _ => Tolerance::rel_abs(1e-3, 4e-3),
    }
}

fn load() -> (
    Qwen3TtsCodecDecoderContractConfig,
    Qwen3TtsCodecDecoder,
    ToyFixture,
) {
    let device = performance_device();
    let dir = testdata!("qwen3_tts");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("toy.config.json")).expect("read the toy config"),
    )
    .expect("parse the toy config");
    let section: Qwen3TtsCodecDecoderMirror =
        serde_json::from_value(json["codec"]["decoder_config"].clone())
            .expect("the toy's decoder_config");
    let config =
        Qwen3TtsCodecDecoderMirror::codec_decoder_config(&section).expect("the toy's config");
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    let mut model: Qwen3TtsCodecDecoder = config.init(&device);
    load_codec_decoder_tensors(&mut model, &config, fixture.take_weights(), None)
        .expect("load the toy weights");
    (config, model, fixture)
}

/// Each step of the decode, and the audio, are the reference's.
#[test]
fn test_toy_decode() {
    let (config, model, fixture) = load();
    assert_eq!(config.samples_per_frame(), 24);
    let codes = fixture.io::<3>("codec.codes").int();
    let (audio, trace) = model.forward_traced(codes);
    assert_io_close("quantized", trace.quantized, fixture.io("codec.quantized"));
    assert_io_close("pre_conv", trace.pre_conv, fixture.io("codec.pre_conv"));
    assert_io_close(
        "transformer",
        trace.transformer,
        fixture.io("codec.transformer"),
    );
    for (i, z) in trace.upsample.into_iter().enumerate() {
        assert_io_close(
            &format!("upsample_{i}"),
            z,
            fixture.io(&format!("codec.upsample_{i}")),
        );
    }
    for (i, y) in trace.vocoder.into_iter().enumerate() {
        assert_io_close_within(
            &format!("decoder_{i}"),
            y,
            fixture.io(&format!("codec.decoder_{i}")),
            vocoder_tolerance(i),
        );
    }
    assert_io_close_within(
        "audio",
        audio,
        fixture.io("codec.audio"),
        vocoder_tolerance(4),
    );
}

/// Decoded in chunks, the audio is upstream's `chunked_decode`'s: the whole
/// decode's up to the first chunk's end, and not after.
#[test]
fn test_toy_chunked_decode() {
    let (config, model, fixture) = load();
    let codes = fixture.io::<3>("codec.codes").int();
    let decode = Qwen3TtsDecodeConfig::new().with_chunk(8).with_context(3);
    let chunked = model.chunked_decode(codes.clone(), &decode);
    assert_io_close_within(
        "chunked_audio",
        chunked.clone(),
        fixture.io("codec.chunked_audio"),
        vocoder_tolerance(4),
    );
    let whole = model.forward(codes);
    let boundary = 8 * config.samples_per_frame();
    assert_io_close(
        "the first chunk",
        chunked.clone().slice(s![.., .., 0..boundary]),
        whole.clone().slice(s![.., .., 0..boundary]),
    );
    let gap = (chunked - whole).abs().max().into_scalar::<f32>();
    assert!(
        gap > 1e-2,
        "past the first chunk, not the whole decode: {gap}"
    );
}

/// One chunk that holds the take is the whole decode.
#[test]
fn test_toy_one_chunk_is_whole() {
    let (_, model, fixture) = load();
    let codes = fixture.io::<3>("codec.codes").int();
    assert_io_close(
        "one chunk",
        model.chunked_decode(
            codes.clone(),
            &Qwen3TtsDecodeConfig::new().with_chunk(64).with_context(3),
        ),
        model.forward(codes),
    );
}

/// `codes` (`[B, K, T]`) cut into pushes of `sizes` (cycled), each through
/// `push`: the audio joined.
fn pushed(
    codes: &Tensor<3, Int>,
    sizes: &[usize],
    mut push: impl FnMut(Tensor<3, Int>) -> Tensor<3>,
) -> Tensor<3> {
    let t = codes.dims()[2];
    let (mut at, mut i, mut parts) = (0, 0, Vec::new());
    while at < t {
        let n = sizes[i % sizes.len()].min(t - at);
        parts.push(push(codes.clone().slice(s![.., .., at..at + n])));
        at += n;
        i += 1;
    }
    Tensor::cat(parts, 2)
}

/// The exact stream's audio, joined, is the whole decode's, whatever the
/// pushes; past the transformer's window (20 frames against the toy's).
#[test]
fn test_toy_stream_is_whole() {
    let (_, model, fixture) = load();
    let codes = fixture.io::<3>("codec.codes").int();
    let whole = model.forward(codes.clone());
    for sizes in [&[1][..], &[2], &[3], &[7], &[1, 4, 2], &[20]] {
        let mut ctx = model.new_stream_context(1, &codes.device());
        let streamed = pushed(&codes, sizes, |piece| model.forward_stream(piece, &mut ctx));
        assert_eq!(ctx.frames, 20);
        assert_io_close_within(
            &format!("stream by {sizes:?}"),
            streamed,
            whole.clone(),
            vocoder_tolerance(4),
        );
    }
}

/// The chunked stream's audio, joined, is the render's (`chunked_decode`,
/// in chunks of 8 with 3 of context) and upstream's, whatever the pushes:
/// pushes that end inside a chunk, on its edge, and across two.
#[test]
fn test_toy_chunked_stream_is_render() {
    let (_, model, fixture) = load();
    let codes = fixture.io::<3>("codec.codes").int();
    let decode = Qwen3TtsDecodeConfig::new().with_chunk(8).with_context(3);
    let render = model.chunked_decode(codes.clone(), &decode);
    for sizes in [&[1][..], &[3], &[8], &[5, 11], &[20]] {
        let mut ctx = model.new_chunked_stream_context(decode);
        let streamed = pushed(&codes, sizes, |piece| {
            model.forward_chunked_stream(piece, &mut ctx)
        });
        assert_eq!(ctx.frames, 20);
        assert_io_close_within(
            &format!("chunked stream by {sizes:?}"),
            streamed.clone(),
            render.clone(),
            vocoder_tolerance(4),
        );
        assert_io_close_within(
            &format!("chunked stream by {sizes:?}, upstream's"),
            streamed,
            fixture.io("codec.chunked_audio"),
            vocoder_tolerance(4),
        );
    }
}

/// With no left context, each chunk is decoded alone.
#[test]
fn test_toy_chunked_stream_without_context() {
    let (_, model, fixture) = load();
    let codes = fixture.io::<3>("codec.codes").int();
    let decode = Qwen3TtsDecodeConfig::new().with_chunk(6).with_context(0);
    let mut ctx = model.new_chunked_stream_context(decode);
    let streamed = pushed(&codes, &[4], |piece| {
        model.forward_chunked_stream(piece, &mut ctx)
    });
    assert_io_close_within(
        "no context",
        streamed,
        model.chunked_decode(codes, &decode),
        vocoder_tolerance(4),
    );
}
