//! The language decoder's tests.
use burn::{
    Tensor,
    module::Module,
    nn::norm::RmsNormConfig,
    prelude::Int,
    tensor::Tolerance,
};

use super::*;
use crate::{
    blocks::{
        mlp::BunsenGatedMlpConfig,
        norm::{
            BunsenNormalizationConfig,
            GatedNormConfig,
        },
        transformers::{
            attention::csa::CausalSelfAttentionConfig,
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::{
                    DecoderStackMeta,
                    DecoderStackVariantConfig,
                    PatternedStackConfig,
                    UniformStackConfig,
                },
            },
            embedding::{
                RotaryEmbeddingConfig,
                TokenEmbeddingConfig,
            },
            head::LmHeadConfig,
            mixers::GatedDeltaNetConfig,
        },
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    },
    ops::transformers::attention::{
        KVCacheConfig,
        KVCacheMeta,
        KVCacheModel,
    },
    support::testing::{
        assert_tensors_close,
        cpu_device,
    },
};

fn norm() -> BunsenNormalizationConfig {
    BunsenNormalizationConfig::Rms(RmsNormConfig::new(0))
}

fn attention_layer() -> CausalDecoderBlockConfig {
    CausalDecoderBlockConfig::new(
        CausalSelfAttentionConfig::new(3, 1, 12)
            .with_head_dim(Some(6))
            .into(),
        BunsenGatedMlpConfig::new(12, 20).into(),
    )
    .with_norm(norm())
}

fn linear_layer() -> CausalDecoderBlockConfig {
    CausalDecoderBlockConfig::new(
        GatedDeltaNetConfig::new(12, 2, 4, 4, 3)
            .with_norm(GatedNormConfig::rms(0, 1e-6))
            .into(),
        BunsenGatedMlpConfig::new(12, 20).into(),
    )
    .with_norm(norm())
}

fn uniform(n_layers: usize) -> LmDecoderVariantConfig {
    LmDecoderVariantConfig::new(
        TokenEmbeddingConfig::new(40, 12),
        LmHeadConfig::new(12, 40),
        UniformStackConfig::new(
            n_layers,
            attention_layer(),
            norm(),
            RotaryEmbeddingConfig::new(0, 6),
        )
        .into(),
    )
}

fn hybrid() -> LmDecoderVariantConfig {
    LmDecoderVariantConfig::new(
        TokenEmbeddingConfig::new(40, 12),
        LmHeadConfig::new(12, 40).with_tied(false),
        PatternedStackConfig::new(
            vec![attention_layer(), linear_layer()],
            vec![1, 0],
            norm(),
            RotaryEmbeddingConfig::new(0, 6),
        )
        .into(),
    )
}

/// The variant and its structure build the same model, and the meta
/// agrees across the structure and the model; the KV cache takes the
/// stack's geometry and the model's pre-size, uncapped where the stack
/// builds its rotary tables to order.
#[test]
fn test_pathways_and_meta() {
    let device = cpu_device();
    let variant = uniform(2).with_kv_cache_pre_size(16);
    let structure = variant.to_structure();
    let a: LmDecoderModel = variant.init(&device);
    let b: LmDecoderModel = structure.init(&device);
    for meta in [&a as &dyn LmDecoderMeta, &b, &structure] {
        assert_eq!(
            (
                meta.n_vocab(),
                meta.d_model(),
                meta.n_layers(),
                meta.n_kv_heads(),
                meta.head_dim()
            ),
            (40, 12, 2, 1, 6)
        );
    }
    assert_eq!(a.num_params(), b.num_params());
    assert_eq!(a.kv_cache_pre_size(), 16);
    assert_eq!(a.n_logits(), 40, "an unsliced head");
    let mut sliced = uniform(1);
    sliced.head = sliced.head.clone().with_vocab(Some(30));
    let sliced: LmDecoderModel = sliced.init(&device);
    assert_eq!((sliced.n_vocab(), sliced.n_logits()), (40, 30));

    let config = a.kv_cache_config();
    assert_eq!(
        (
            config.num_layers(),
            config.num_heads(),
            config.head_dim(),
            config.seq_len(),
            config.max_seq_len()
        ),
        (2, 1, 6, 16, None)
    );
    assert_eq!(
        uniform(1).kv_cache_pre_size,
        LM_DECODER_KV_CACHE_PRE_SIZE,
        "the default pre-size"
    );
}

/// A stack that fixes its rotary table's length caps the cache there.
#[test]
fn test_cache_capped_at_a_fixed_rotary_table() {
    let mut variant = uniform(1);
    if let DecoderStackVariantConfig::Uniform(stack) = &mut variant.stack {
        stack.rotary.seq_len = 24;
    }
    let model: LmDecoderModel = variant.init(&cpu_device());
    assert_eq!(model.kv_cache_config().max_seq_len(), Some(24));
}

/// A cached decode's hidden states are the uncached forward's; the
/// logits cover the vocabulary.
#[test]
fn test_cached_matches_uncached() {
    let device = cpu_device();
    let model: LmDecoderModel = uniform(2).init(&device);
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 7, 2, 9]], &device);
    let embeds = model.embed(tokens);
    let r_emb = model.rotary(8, &device);
    let whole = model.forward_hidden(embeds.clone(), &r_emb, None);
    let mut cache = model.kv_cache_config().with_seq_len(8).init(&device);
    let head = model.forward_hidden(embeds.clone().slice_dim(1, 0..3), &r_emb, Some(&mut cache));
    let tail = model.forward_hidden(embeds.slice_dim(1, 3..5), &r_emb, Some(&mut cache));
    assert_tensors_close(
        &Tensor::cat(vec![head, tail], 1),
        &whole,
        Tolerance::rel_abs(1e-4, 1e-5),
    );
    assert_eq!(model.logits(whole).dims(), [1, 5, 40]);
}

/// A stack of any mixers builds, and runs uncached; one with a
/// linear-attention layer decodes only through a stack cache, so a flat
/// cache is refused.
#[test]
fn test_hybrid_stack() {
    let device = cpu_device();
    let model: LmDecoderModel = hybrid().init(&device);
    assert_eq!(model.decoder.stack.layer_attends(), [false, true]);
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 7]], &device);
    let hidden = model.forward_hidden(model.embed(tokens), &model.rotary(3, &device), None);
    assert_eq!(model.logits(hidden).dims(), [1, 3, 40]);
    let _: LmDecoderModel = uniform(1).init(&device);
}

/// A hybrid model handed a flat cache refuses it.
#[test]
#[should_panic(expected = "DeltaNetCache")]
fn test_hybrid_refuses_a_flat_cache() {
    let device = cpu_device();
    let model: LmDecoderModel = hybrid().init(&device);
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 7]], &device);
    let mut cache = model.new_kv_cache(1, &device);
    let _ = model.forward_hidden(
        model.embed(tokens),
        &model.rotary(3, &device),
        Some(&mut cache),
    );
}

/// A wrapper of an attention-only model, as a kit's.
#[derive(Module, Debug)]
struct Wrapped {
    lm: LmDecoderModel,
}

impl LmDecoderHolder for Wrapped {
    fn lm(&self) -> &LmDecoderModel {
        &self.lm
    }

    fn lm_mut(&mut self) -> &mut LmDecoderModel {
        &mut self.lm
    }

    fn from_lm(lm: LmDecoderModel) -> Self {
        Self { lm }
    }
}

impl LmDecoderWrapper for Wrapped {}

/// A wrapper of a hybrid model.
#[derive(Module, Debug)]
struct WrappedHybrid {
    lm: LmDecoderModel,
}

impl LmDecoderHolder for WrappedHybrid {
    fn lm(&self) -> &LmDecoderModel {
        &self.lm
    }

    fn lm_mut(&mut self) -> &mut LmDecoderModel {
        &mut self.lm
    }

    fn from_lm(lm: LmDecoderModel) -> Self {
        Self { lm }
    }
}

impl LmDecoderWrapper for WrappedHybrid {}

/// A wrapper's forward and cache are its model's.
#[test]
fn test_wrapper_passthrough() {
    let device = cpu_device();
    let wrapped = Wrapped {
        lm: uniform(2).with_kv_cache_pre_size(16).init(&device),
    };
    let geometry = |c: KVCacheConfig| {
        (
            c.num_layers(),
            c.num_heads(),
            c.head_dim(),
            c.seq_len(),
            c.max_seq_len(),
        )
    };
    assert_eq!(
        geometry(wrapped.kv_cache_config()),
        geometry(wrapped.lm.kv_cache_config())
    );
    let tokens = Tensor::<2, Int>::from_data([[1, 5, 7]], &device);
    let embeds = wrapped.lm().embed(tokens.clone());
    let r_emb = wrapped.lm().rotary(3, &device);
    assert_tensors_close(
        &wrapped.forward_hidden(embeds.clone(), &r_emb, None),
        &wrapped.lm.forward_hidden(embeds, &r_emb, None),
        Tolerance::default(),
    );

    let hybrid = WrappedHybrid {
        lm: hybrid().init(&device),
    };
    let embeds = hybrid.lm().embed(tokens);
    assert_tensors_close(
        &hybrid.forward_hidden(embeds.clone(), &r_emb, None),
        &hybrid.lm.forward_hidden(embeds, &r_emb, None),
        Tolerance::default(),
    );
}

/// A stack that does not lower is refused through the variant, with a
/// frame naming the step.
#[test]
fn test_lowering_frame() {
    let mut variant = hybrid();
    if let DecoderStackVariantConfig::Patterned(stack) = &mut variant.stack {
        stack.layer_kinds = vec![0, 2];
    }
    ErrorMatcher::kind(BunsenErrorKind::Illegal)
        .message_contains("layer_kinds[1]")
        .frame_contains("LmDecoderVariantConfig's stack")
        .assert_err(&variant.try_to_structure());
}

/// Parts that disagree are refused, naming the fields: an embedding or a
/// head whose width is not the stack's, a tied head whose rows are not the
/// embedding's.
#[test]
fn test_parts_agree() {
    let device = cpu_device();
    let structure = uniform(1).to_structure();
    let cases = [
        (
            LmDecoderStructureConfig {
                embedding: TokenEmbeddingConfig::new(40, 10),
                ..structure.clone()
            },
            "embedding.d_model",
        ),
        (
            LmDecoderStructureConfig {
                decoder: LmStackDecoderConfig {
                    head: LmHeadConfig::new(10, 40).with_tied(false),
                    ..structure.decoder.clone()
                },
                ..structure.clone()
            },
            "head.d_model",
        ),
        (
            LmDecoderStructureConfig {
                decoder: LmStackDecoderConfig {
                    head: LmHeadConfig::new(12, 48),
                    ..structure.decoder.clone()
                },
                ..structure.clone()
            },
            "head.n_rows",
        ),
    ];
    for (config, field) in cases {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains(field)
            .assert_err(&ModuleInit::<LmDecoderModel>::try_init(&config, &device));
    }
    let untied = LmDecoderStructureConfig {
        decoder: LmStackDecoderConfig {
            head: LmHeadConfig::new(12, 48).with_tied(false),
            ..structure.decoder.clone()
        },
        ..structure
    };
    let _: LmDecoderModel = untied.init(&device);
}

/// The model is its embedding and its decoder: the decoder built from the
/// structure's half holds the same parameters but the embedding's, and the
/// model's logits are the decoder's handed the embedding's table; a decoder
/// with a head of its own needs no table, and its meta is the stack's.
#[test]
fn test_stack_decoder() {
    let device = cpu_device();
    let structure = uniform(2).to_structure();
    let model: LmDecoderModel = structure.init(&device);
    let decoder: LmStackDecoder = structure.decoder.clone().init(&device);
    assert_eq!(
        decoder.num_params() + model.embedding.num_params(),
        model.num_params()
    );
    let meta = |m: &dyn LmStackDecoderMeta| {
        (
            m.n_logits(),
            m.d_model(),
            m.n_layers(),
            m.n_kv_heads(),
            m.head_dim(),
        )
    };
    assert_eq!(meta(&decoder), (40, 12, 2, 1, 6));
    assert_eq!(meta(&structure.decoder.clone()), meta(&decoder));
    assert_eq!(meta(&model.decoder), meta(&decoder));

    let tokens = Tensor::<2, Int>::from_data([[1, 2, 3, 4]], &device);
    let embeds = model.embed(tokens);
    let r_emb = model.rotary(4, &device);
    let hidden = model.forward_hidden(embeds.clone(), &r_emb, None);
    assert_tensors_close(
        &model.decoder.forward_hidden(embeds, &r_emb, None),
        &hidden,
        Tolerance::rel_abs(1e-6, 1e-6),
    );
    assert_tensors_close(
        &model.logits(hidden.clone()),
        &model
            .decoder
            .logits(hidden.clone(), Some(&model.embedding.weight)),
        Tolerance::rel_abs(1e-6, 1e-6),
    );

    let own = LmStackDecoderConfig::new(
        LmHeadConfig::new(12, 40).with_tied(false),
        structure.decoder.stack.clone(),
    );
    let own: LmStackDecoder = own.init(&device);
    assert_eq!(own.logits(hidden, None).dims(), [1, 4, 40]);
}

/// A decoder refuses a head whose width is not its stack's; a stack of any
/// mixers builds.
#[test]
fn test_stack_decoder_refusals() {
    let device = cpu_device();
    let structure = uniform(1).to_structure();
    let wide =
        LmStackDecoderConfig::new(LmHeadConfig::new(16, 40), structure.decoder.stack.clone());
    assert!(ModuleInit::<LmStackDecoder>::try_init(&wide, &device).is_err());
    let hybrid = hybrid().to_structure();
    assert!(ModuleInit::<LmStackDecoder>::try_init(&hybrid.decoder.clone(), &device).is_ok());
}
