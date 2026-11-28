use super::*;
use crate::{
    blocks::signals::DwStridingSubsamplingConfig,
    kits::speech::asr::parakeet::ParakeetTdtConfig,
    support::testing::assert_config_eq,
};

#[test]
fn test_parakeet_tdt_rekey() {
    for (hf, bunsen) in [
        (
            "encoder.subsampling.layers.0.weight",
            Some("encoder.subsampling.stem.conv.weight"),
        ),
        (
            "encoder.subsampling.layers.2.bias",
            Some("encoder.subsampling.stages.0.depthwise.conv.bias"),
        ),
        (
            "encoder.subsampling.layers.3.weight",
            Some("encoder.subsampling.stages.0.pointwise.conv.weight"),
        ),
        (
            "encoder.subsampling.layers.5.weight",
            Some("encoder.subsampling.stages.1.depthwise.conv.weight"),
        ),
        (
            "encoder.subsampling.layers.6.bias",
            Some("encoder.subsampling.stages.1.pointwise.conv.bias"),
        ),
        ("encoder.subsampling.layers.8.weight", None),
        (
            "encoder.subsampling.linear.weight",
            Some("encoder.subsampling.linear.weight"),
        ),
        (
            "encoder.layers.3.self_attn.q_proj.weight",
            Some("encoder.transformer.layers.3.attn.query.weight"),
        ),
        (
            "encoder.layers.3.self_attn.relative_k_proj.weight",
            Some("encoder.transformer.layers.3.attn.pos.weight"),
        ),
        (
            "encoder.layers.3.self_attn.bias_u",
            Some("encoder.transformer.layers.3.attn.pos_bias_u"),
        ),
        (
            "encoder.layers.3.conv.depthwise_conv.weight",
            Some("encoder.transformer.layers.3.conv.depthwise.conv.weight"),
        ),
        (
            "encoder.layers.3.conv.norm.running_var",
            Some("encoder.transformer.layers.3.conv.depthwise.norm.running_var"),
        ),
        ("encoder.layers.3.conv.norm.num_batches_tracked", None),
        (
            "encoder.layers.3.conv.pointwise_conv1.weight",
            Some("encoder.transformer.layers.3.conv.pointwise_conv1.weight"),
        ),
        (
            "encoder.layers.3.norm_out.weight",
            Some("encoder.transformer.layers.3.out_norm.weight"),
        ),
        (
            "decoder.lstm.weight_ih_l1",
            Some("decoder.lstm.1.features.weight"),
        ),
        (
            "decoder.lstm.bias_hh_l0",
            Some("decoder.lstm.0.hidden.bias"),
        ),
        (
            "decoder.decoder_projector.bias",
            Some("decoder.projector.bias"),
        ),
        ("joint.head.weight", Some("joint.head.weight")),
        ("encoder_projector.weight", Some("encoder_projector.weight")),
    ] {
        assert_eq!(
            parakeet_tdt_rekey(&DwStridingSubsamplingConfig::new(128, 256, 1024))
                .apply(hf)
                .as_deref(),
            bunsen,
            "{hf}"
        );
    }
}

/// `nvidia/parakeet-tdt-0.6b-v3`'s `config.json` is the built-in
/// geometry; a CTC model's is refused by its `model_type`.
#[test]
fn test_parakeet_tdt_config() {
    const V3: &str = r#"{"architectures": ["ParakeetForTDT"], "blank_token_id": 8192,
        "decoder_hidden_size": 640, "durations": [0, 1, 2, 3, 4],
        "encoder_config": {"attention_bias": false, "conv_kernel_size": 9,
            "convolution_bias": false, "hidden_size": 1024, "intermediate_size": 4096,
            "model_type": "parakeet_encoder", "num_attention_heads": 8,
            "num_hidden_layers": 24, "num_mel_bins": 128, "scale_input": false,
            "subsampling_conv_channels": 256, "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2, "subsampling_factor": 8},
        "model_type": "parakeet_tdt", "num_decoder_layers": 2, "vocab_size": 8193}"#;
    let config = parakeet_tdt_config_from_str(V3).unwrap();
    assert_config_eq("the config", &config, &ParakeetTdtConfig::tdt_0_6b_v3());

    let ctc = V3.replace("parakeet_tdt", "parakeet_ctc");
    crate::errors::testing::ErrorMatcher::unsupported()
        .message_contains("`model_type`")
        .assert_err(&parakeet_tdt_config_from_str(&ctc));
    let mels = V3.replace("\"num_mel_bins\": 128", "\"num_mel_bins\": 80");
    crate::errors::testing::ErrorMatcher::unsupported()
        .message_contains("`encoder_config.num_mel_bins` is 80")
        .assert_err(&parakeet_tdt_config_from_str(&mels));
}
