//! The reference models' flat signatures, as calls over their caches.

use burn::prelude::*;

use super::{
    DECODER_HEAD_DIM,
    DECODER_KV_HEADS,
    DECODER_LAYERS,
    ENCODER_HEAD_DIM,
    ENCODER_HEADS,
    ENCODER_LAYERS,
    PADDING_CACHE,
    reference::{
        VoxtralAudioEncoderOnnx,
        VoxtralDecoderOnnx,
        VoxtralEmbedTokensOnnx,
    },
};

/// The reference audio encoder's carry: each layer's key and value caches,
/// `[1, 32, past, 64]`, and the packed conv carry, [`PADDING_CACHE`].
#[derive(Clone, Debug)]
pub struct EncoderCache {
    /// One `(key, value)` pair per layer.
    pub kv: Vec<(Tensor<4>, Tensor<4>)>,
    /// The packed conv carry.
    pub padding: Tensor<3>,
}

impl EncoderCache {
    /// The carry of a stream that has seen nothing: every cache
    /// `[1, 32, 0, 64]`, as the export reads an empty one, and a zero conv
    /// carry.
    pub fn empty(device: &Device) -> Self {
        let t = Tensor::<4>::zeros([1, ENCODER_HEADS, 0, ENCODER_HEAD_DIM], device);
        Self {
            kv: (0..ENCODER_LAYERS)
                .map(|_| (t.clone(), t.clone()))
                .collect(),
            padding: Tensor::zeros(PADDING_CACHE, device),
        }
    }
}

/// Runs the reference audio encoder once over `features`, after what `past`
/// has seen.
///
/// # Arguments
/// * `features` - `[1, 128, frames]` log-mels, `frames` a multiple of 8.
///
/// # Returns
/// The `[1, frames/8, 3072]` audio embeddings, and the carry after the call.
#[allow(clippy::too_many_lines)]
pub fn reference_encode(
    model: &VoxtralAudioEncoderOnnx,
    features: Tensor<3>,
    past: &EncoderCache,
) -> (Tensor<3>, EncoderCache) {
    assert_eq!(past.kv.len(), ENCODER_LAYERS, "one cache per layer");
    let device = features.device();
    let seq = features.dims()[2] / 2;
    let start = past.kv[0].0.dims()[2];
    let attention_mask = Tensor::<2, Int>::ones([1, start + seq], &device);
    let positions =
        Tensor::<1, Int>::arange(start as i64..(start + seq) as i64, &device).unsqueeze_dim(0);
    let kv = &past.kv;
    let (
        embeds,
        padding,
        k0,
        v0,
        k1,
        v1,
        k2,
        v2,
        k3,
        v3,
        k4,
        v4,
        k5,
        v5,
        k6,
        v6,
        k7,
        v7,
        k8,
        v8,
        k9,
        v9,
        k10,
        v10,
        k11,
        v11,
        k12,
        v12,
        k13,
        v13,
        k14,
        v14,
        k15,
        v15,
        k16,
        v16,
        k17,
        v17,
        k18,
        v18,
        k19,
        v19,
        k20,
        v20,
        k21,
        v21,
        k22,
        v22,
        k23,
        v23,
        k24,
        v24,
        k25,
        v25,
        k26,
        v26,
        k27,
        v27,
        k28,
        v28,
        k29,
        v29,
        k30,
        v30,
        k31,
        v31,
    ) = model.forward(
        features,
        attention_mask,
        positions,
        past.padding.clone(),
        kv[0].0.clone(),
        kv[0].1.clone(),
        kv[1].0.clone(),
        kv[1].1.clone(),
        kv[2].0.clone(),
        kv[2].1.clone(),
        kv[3].0.clone(),
        kv[3].1.clone(),
        kv[4].0.clone(),
        kv[4].1.clone(),
        kv[5].0.clone(),
        kv[5].1.clone(),
        kv[6].0.clone(),
        kv[6].1.clone(),
        kv[7].0.clone(),
        kv[7].1.clone(),
        kv[8].0.clone(),
        kv[8].1.clone(),
        kv[9].0.clone(),
        kv[9].1.clone(),
        kv[10].0.clone(),
        kv[10].1.clone(),
        kv[11].0.clone(),
        kv[11].1.clone(),
        kv[12].0.clone(),
        kv[12].1.clone(),
        kv[13].0.clone(),
        kv[13].1.clone(),
        kv[14].0.clone(),
        kv[14].1.clone(),
        kv[15].0.clone(),
        kv[15].1.clone(),
        kv[16].0.clone(),
        kv[16].1.clone(),
        kv[17].0.clone(),
        kv[17].1.clone(),
        kv[18].0.clone(),
        kv[18].1.clone(),
        kv[19].0.clone(),
        kv[19].1.clone(),
        kv[20].0.clone(),
        kv[20].1.clone(),
        kv[21].0.clone(),
        kv[21].1.clone(),
        kv[22].0.clone(),
        kv[22].1.clone(),
        kv[23].0.clone(),
        kv[23].1.clone(),
        kv[24].0.clone(),
        kv[24].1.clone(),
        kv[25].0.clone(),
        kv[25].1.clone(),
        kv[26].0.clone(),
        kv[26].1.clone(),
        kv[27].0.clone(),
        kv[27].1.clone(),
        kv[28].0.clone(),
        kv[28].1.clone(),
        kv[29].0.clone(),
        kv[29].1.clone(),
        kv[30].0.clone(),
        kv[30].1.clone(),
        kv[31].0.clone(),
        kv[31].1.clone(),
    );
    (
        embeds,
        EncoderCache {
            kv: vec![
                (k0, v0),
                (k1, v1),
                (k2, v2),
                (k3, v3),
                (k4, v4),
                (k5, v5),
                (k6, v6),
                (k7, v7),
                (k8, v8),
                (k9, v9),
                (k10, v10),
                (k11, v11),
                (k12, v12),
                (k13, v13),
                (k14, v14),
                (k15, v15),
                (k16, v16),
                (k17, v17),
                (k18, v18),
                (k19, v19),
                (k20, v20),
                (k21, v21),
                (k22, v22),
                (k23, v23),
                (k24, v24),
                (k25, v25),
                (k26, v26),
                (k27, v27),
                (k28, v28),
                (k29, v29),
                (k30, v30),
                (k31, v31),
            ],
            padding,
        },
    )
}

/// Runs the reference token embedding over `ids`.
///
/// # Returns
/// `[1, ids, 3072]`.
pub fn reference_embed(
    model: &VoxtralEmbedTokensOnnx,
    ids: &[i64],
    device: &Device,
) -> Tensor<3> {
    let ids = Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device);
    model.forward(ids)
}

/// The reference decoder's per-layer key and value caches, `[1, 8, past, 128]`
/// each, one pair per layer.
pub type DecoderCache = Vec<(Tensor<4>, Tensor<4>)>;

/// An empty [`DecoderCache`]: every layer's key and value `[1, 8, 0, 128]`,
/// as the export reads an empty cache.
pub fn empty_decoder_cache(device: &Device) -> DecoderCache {
    let t = Tensor::<4>::zeros([1, DECODER_KV_HEADS, 0, DECODER_HEAD_DIM], device);
    (0..DECODER_LAYERS)
        .map(|_| (t.clone(), t.clone()))
        .collect()
}

/// Runs the reference decoder once over `embeds`, at the positions after
/// `past`, every key visible.
///
/// # Arguments
/// * `embeds` - `[1, seq, 3072]` input embeddings.
///
/// # Returns
/// `[1, seq, 131072]` logits, and the cache after the call.
#[allow(clippy::too_many_lines)]
pub fn reference_decode(
    model: &VoxtralDecoderOnnx,
    embeds: Tensor<3>,
    past: &DecoderCache,
) -> (Tensor<3>, DecoderCache) {
    assert_eq!(past.len(), DECODER_LAYERS, "one cache per layer");
    let device = embeds.device();
    let seq = embeds.dims()[1];
    let start = past[0].0.dims()[2];
    let attention_mask = Tensor::<2, Int>::ones([1, start + seq], &device);
    let (
        logits,
        k0,
        v0,
        k1,
        v1,
        k2,
        v2,
        k3,
        v3,
        k4,
        v4,
        k5,
        v5,
        k6,
        v6,
        k7,
        v7,
        k8,
        v8,
        k9,
        v9,
        k10,
        v10,
        k11,
        v11,
        k12,
        v12,
        k13,
        v13,
        k14,
        v14,
        k15,
        v15,
        k16,
        v16,
        k17,
        v17,
        k18,
        v18,
        k19,
        v19,
        k20,
        v20,
        k21,
        v21,
        k22,
        v22,
        k23,
        v23,
        k24,
        v24,
        k25,
        v25,
    ) = model.forward(
        embeds,
        attention_mask,
        past[0].0.clone(),
        past[0].1.clone(),
        past[1].0.clone(),
        past[1].1.clone(),
        past[2].0.clone(),
        past[2].1.clone(),
        past[3].0.clone(),
        past[3].1.clone(),
        past[4].0.clone(),
        past[4].1.clone(),
        past[5].0.clone(),
        past[5].1.clone(),
        past[6].0.clone(),
        past[6].1.clone(),
        past[7].0.clone(),
        past[7].1.clone(),
        past[8].0.clone(),
        past[8].1.clone(),
        past[9].0.clone(),
        past[9].1.clone(),
        past[10].0.clone(),
        past[10].1.clone(),
        past[11].0.clone(),
        past[11].1.clone(),
        past[12].0.clone(),
        past[12].1.clone(),
        past[13].0.clone(),
        past[13].1.clone(),
        past[14].0.clone(),
        past[14].1.clone(),
        past[15].0.clone(),
        past[15].1.clone(),
        past[16].0.clone(),
        past[16].1.clone(),
        past[17].0.clone(),
        past[17].1.clone(),
        past[18].0.clone(),
        past[18].1.clone(),
        past[19].0.clone(),
        past[19].1.clone(),
        past[20].0.clone(),
        past[20].1.clone(),
        past[21].0.clone(),
        past[21].1.clone(),
        past[22].0.clone(),
        past[22].1.clone(),
        past[23].0.clone(),
        past[23].1.clone(),
        past[24].0.clone(),
        past[24].1.clone(),
        past[25].0.clone(),
        past[25].1.clone(),
    );
    (
        logits,
        vec![
            (k0, v0),
            (k1, v1),
            (k2, v2),
            (k3, v3),
            (k4, v4),
            (k5, v5),
            (k6, v6),
            (k7, v7),
            (k8, v8),
            (k9, v9),
            (k10, v10),
            (k11, v11),
            (k12, v12),
            (k13, v13),
            (k14, v14),
            (k15, v15),
            (k16, v16),
            (k17, v17),
            (k18, v18),
            (k19, v19),
            (k20, v20),
            (k21, v21),
            (k22, v22),
            (k23, v23),
            (k24, v24),
            (k25, v25),
        ],
    )
}
