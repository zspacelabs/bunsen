//! The toy codec decoder against the `-hf` port's.
//!
//! `testdata/zonos3/toy.safetensors` holds a two-stage `Zonos3MiniCodec`
//! (the `-hf` port's, op for op with upstream's) under the `-hf` file's
//! names, and its answers over 12 frames of 3 codebooks, computed in float32
//! (`tools/gen_zonos3_toy_fixtures.py`): the latent, each trunk step, both
//! iSTFT branches (FFT sizes 24 and 72), the mix and the resampled audio.
//! The geometry comes from `toy.config.json`'s `codec` section, and the
//! weights load through [`load_zonos3_codec_tensors`].

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    Zonos3Codec,
    Zonos3CodecConfig,
    Zonos3CodecJson,
    weights::load_zonos3_codec_tensors,
};
use crate::{
    burner::module::ModuleInit,
    support::testing::{
        ToyFixture,
        assert_io_close_within,
        performance_device,
    },
};

/// The toy's tolerance: relative `1e-3`, absolute `1e-4`; the reference is
/// float32's.
fn close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    assert_io_close_within(
        what,
        actual,
        expected,
        burn::tensor::Tolerance::rel_abs(1e-3, 1e-4),
    );
}

fn load() -> (Zonos3CodecConfig, Zonos3Codec, ToyFixture) {
    let device = performance_device();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/zonos3");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("toy.config.json")).expect("read the toy config"),
    )
    .expect("parse the toy config");
    let codec: Zonos3CodecJson =
        serde_json::from_value(json["codec"].clone()).expect("the toy's codec section");
    let config = Zonos3CodecConfig::from_json(&codec).expect("the toy's config");
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    let mut model: Zonos3Codec = config.init(&device);
    load_zonos3_codec_tensors(&mut model, &config, fixture.take_weights(), None)
        .expect("load the toy weights");
    (config, model, fixture)
}

/// Every stage, from the codes to the audio. The fixture's trunk tensors
/// are channels first, `[B, C, T]`.
#[test]
fn test_toy_codec() {
    let (config, model, fixture) = load();
    assert_eq!(config.stages.len(), 2);
    assert_eq!(config.n_ffts, [24, 72]);
    let codes: Tensor<3, Int> = fixture.io::<3>("codec.codes").int();
    let (audio, trace) = model.forward_traced(codes);
    let first = |t: Tensor<3>| t.swap_dims(1, 2);
    close(
        "latent",
        first(trace.latent),
        fixture.io::<3>("codec.latent"),
    );
    for (i, name) in ["stage_0", "patch_0", "stage_1"].iter().enumerate() {
        close(
            name,
            first(trace.trunk[i].clone()),
            fixture.io::<3>(&format!("codec.{name}")),
        );
    }
    close("y0", trace.branches[0].clone(), fixture.io::<2>("codec.y0"));
    close("y1", trace.branches[1].clone(), fixture.io::<2>("codec.y1"));
    close(
        "mixer",
        trace.mixed.unsqueeze_dim::<3>(1),
        fixture.io::<3>("codec.mixer"),
    );
    close(
        "audio",
        audio.unsqueeze_dim::<3>(1),
        fixture.io::<3>("codec.audio"),
    );
}
