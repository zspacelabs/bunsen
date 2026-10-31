//! The reference model's flat signature, as a step over a cache.

use burn::prelude::*;

use super::{
    N_LAYERS,
    reference::Qwen3OnnxModel,
};

/// The reference model's per-layer key and value caches, `[1, 8, past, 128]`
/// each, one pair per layer.
pub type ReferenceCache = Vec<(Tensor<4>, Tensor<4>)>;

/// An empty [`ReferenceCache`]: every layer's key and value
/// `[1, 8, 0, 128]`, as the export reads an empty cache.
pub fn empty_reference_cache(device: &Device) -> ReferenceCache {
    (0..N_LAYERS)
        .map(|_| {
            let t = Tensor::<4>::zeros([1, super::N_KV_HEADS, 0, super::HEAD_DIM], device);
            (t.clone(), t)
        })
        .collect()
}

/// Runs the reference model once over `ids`, at the positions after `past`,
/// every key visible.
///
/// # Returns
/// `[1, ids, vocab]` logits, and the cache after the call.
#[allow(clippy::too_many_lines)]
pub fn reference_decode(
    model: &Qwen3OnnxModel,
    ids: &[i64],
    past: &ReferenceCache,
    device: &Device,
) -> (Tensor<3>, ReferenceCache) {
    assert_eq!(past.len(), N_LAYERS, "one cache per layer");
    let seq = ids.len();
    let start = past[0].0.dims()[2];
    let input_ids = Tensor::<2, Int>::from_data(TensorData::new(ids.to_vec(), [1, seq]), device);
    let attention_mask = Tensor::<2, Int>::ones([1, start + seq], device);
    let positions =
        Tensor::<1, Int>::arange(start as i64..(start + seq) as i64, device).unsqueeze_dim(0);
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
        k26,
        v26,
        k27,
        v27,
    ) = model.forward(
        input_ids,
        attention_mask,
        positions,
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
        past[26].0.clone(),
        past[26].1.clone(),
        past[27].0.clone(),
        past[27].1.clone(),
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
            (k26, v26),
            (k27, v27),
        ],
    )
}
