//! One linear-attention layer's carried state.

use alloc::vec::Vec;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Int,
        TensorData,
    },
    tensor::{
        Device,
        FloatDType,
    },
};

/// Config for [`DeltaNetState`]: the decode's rows, and the layer's
/// convolution and delta-rule geometry.
#[derive(Config, Debug)]
pub struct DeltaNetStateConfig {
    /// The decode rows.
    pub rows: usize,

    /// The convolution's channels (Qwen3.5's `2 · H_k · d_k + H_v · d_v`).
    pub conv_channels: usize,

    /// The convolution's taps, `K`; the state keeps `K - 1` inputs.
    pub conv_kernel: usize,

    /// The delta rule's heads (the value heads).
    pub heads: usize,

    /// The key width per head.
    pub key_dim: usize,

    /// The value width per head.
    pub value_dim: usize,
}

impl DeltaNetStateConfig {
    /// A state of zeros, the start of a sequence, in f32.
    pub fn init(
        &self,
        device: &Device,
    ) -> DeltaNetState {
        assert!(
            self.conv_kernel >= 2,
            "a causal convolution of at least 2 taps; got {}",
            self.conv_kernel
        );
        let f32 = (device, FloatDType::F32.into());
        DeltaNetState {
            conv: Tensor::zeros([self.rows, self.conv_kernel - 1, self.conv_channels], f32),
            recurrent: Tensor::zeros([self.rows, self.heads, self.key_dim, self.value_dim], f32),
        }
    }
}

/// What one linear-attention layer carries between decode steps: its
/// convolution's last inputs and its delta rule's state, per row.
///
/// The convolution's inputs are kept in f32, which holds a half model's
/// exactly; the delta rule's state accumulates in f32. A model in any dtype
/// carries the same state.
///
/// Made by [`DeltaNetStateConfig`]; a state of zeros is the start of a
/// sequence. A `Module` over bare tensors: state, not parameters.
#[derive(Module, Debug)]
pub struct DeltaNetState {
    /// `[rows, K - 1, channels]`, the convolution's last inputs, oldest
    /// first ([`causal_depthwise_conv1d`](super::causal_depthwise_conv1d)'s
    /// `state`).
    pub conv: Tensor<3>,

    /// `[rows, heads, d_k, d_v]`, the delta rule's state
    /// ([`GatedDeltaRule::recurrent`](super::GatedDeltaRule::recurrent)'s).
    pub recurrent: Tensor<4>,
}

impl DeltaNetState {
    /// The decode rows.
    pub fn rows(&self) -> usize {
        self.conv.dims()[0]
    }

    /// Back to the start of a sequence: zeros, in place of what was carried.
    pub fn reset(&mut self) {
        self.conv = self.conv.zeros_like();
        self.recurrent = self.recurrent.zeros_like();
    }

    /// Permutes the rows: row `r` becomes row `sources[r]` (beams
    /// branching).
    ///
    /// # Panics
    /// If `sources` is not one entry per row, or names a row past them.
    pub fn reorder(
        &mut self,
        sources: &[usize],
    ) {
        let rows = self.rows();
        assert_eq!(sources.len(), rows, "one source row per row");
        assert!(
            sources.iter().all(|&s| s < rows),
            "a source row past the rows"
        );
        if sources.iter().enumerate().all(|(r, &s)| r == s) {
            return;
        }
        let indices: Tensor<1, Int> = Tensor::from_data(
            TensorData::new(
                sources.iter().map(|&s| s as i64).collect::<Vec<_>>(),
                [rows],
            ),
            &self.conv.device(),
        );
        self.conv = self.conv.clone().select(0, indices.clone());
        self.recurrent = self.recurrent.clone().select(0, indices);
    }
}

/// Every linear-attention layer's [`DeltaNetState`], by its slot: what a
/// hybrid model's decode carries beside its KV cache.
///
/// Each linear-attention layer reads and writes its own slot (its mixer's
/// `layer_index`), as an attention layer does its slot of a KV cache. An
/// attention-only model's is empty.
///
/// A state summarizes the whole prefix, so it cannot rewind; it can only
/// reset, or go back to a [snapshot](Self::take_snapshot) of every layer
/// taken at an earlier position. The cache keeps the most recent snapshot.
#[derive(Module, Debug, Default)]
pub struct DeltaNetCache {
    /// The layers' states, by slot.
    pub layers: Vec<DeltaNetState>,

    /// The most recent snapshot, if one was taken since the last reset.
    pub snapshot: Option<DeltaNetSnapshot>,
}

/// Every linear-attention layer's state at a position: what a
/// [`DeltaNetCache`] goes back to.
#[derive(Module, Debug)]
pub struct DeltaNetSnapshot {
    /// The position the states were taken at: the tokens they summarize.
    pub pos: usize,

    /// The layers' states, by slot.
    pub layers: Vec<DeltaNetState>,
}

impl DeltaNetCache {
    /// The cache of these states, by slot, with no snapshot.
    pub fn new(layers: Vec<DeltaNetState>) -> Self {
        Self {
            layers,
            snapshot: None,
        }
    }

    /// Whether it holds no layer's state.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The state of the layer at `slot`.
    ///
    /// # Panics
    /// If there is no such slot.
    pub fn layer_mut(
        &mut self,
        slot: usize,
    ) -> &mut DeltaNetState {
        let len = self.layers.len();
        self.layers
            .get_mut(slot)
            .unwrap_or_else(|| panic!("linear-attention slot {slot} of {len}"))
    }

    /// Every layer back to the start of a sequence; the snapshot dropped.
    pub fn reset(&mut self) {
        self.layers.iter_mut().for_each(DeltaNetState::reset);
        self.snapshot = None;
    }

    /// Keeps every layer's state as it is, as the snapshot at `pos` (the
    /// tokens the states summarize), in place of any older one. The states
    /// are shared, not copied: a step replaces a state, it never writes into
    /// one.
    pub fn take_snapshot(
        &mut self,
        pos: usize,
    ) {
        self.snapshot = Some(DeltaNetSnapshot {
            pos,
            layers: self.layers.clone(),
        });
    }

    /// The snapshot's position, if there is one.
    pub fn snapshot_pos(&self) -> Option<usize> {
        self.snapshot.as_ref().map(|s| s.pos)
    }

    /// Every layer back to the snapshot's states, which it keeps.
    ///
    /// # Panics
    /// If there is no snapshot.
    pub fn restore_snapshot(&mut self) {
        let snapshot = self.snapshot.as_ref().expect("a snapshot to restore");
        self.layers = snapshot.layers.clone();
    }

    /// Permutes every layer's rows, and the snapshot's: row `r` becomes row
    /// `sources[r]`.
    pub fn reorder(
        &mut self,
        sources: &[usize],
    ) {
        let snapshot = self.snapshot.iter_mut().flat_map(|s| s.layers.iter_mut());
        for layer in self.layers.iter_mut().chain(snapshot) {
            layer.reorder(sources);
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        DType,
        Distribution,
    };

    use super::*;
    use crate::support::testing::{
        cpu_device,
        seeded_tensor,
    };

    fn state() -> DeltaNetState {
        let device = cpu_device();
        let mut state = DeltaNetStateConfig::new(3, 5, 4, 2, 3, 2).init(&device);
        state.conv = seeded_tensor::<3>(1, [3, 3, 5], Distribution::Default, &device);
        state.recurrent = seeded_tensor::<4>(2, [3, 2, 3, 2], Distribution::Default, &device);
        state
    }

    /// A new state is zeros of the config's shapes, in f32.
    #[test]
    fn test_init() {
        let device = cpu_device();
        let state = DeltaNetStateConfig::new(2, 10, 4, 3, 5, 6).init(&device);
        assert_eq!(state.rows(), 2);
        assert_eq!(state.conv.dims(), [2, 3, 10]);
        assert_eq!(state.recurrent.dims(), [2, 3, 5, 6]);
        for dtype in [state.conv.dtype(), state.recurrent.dtype()] {
            assert_eq!(dtype, DType::F32);
        }
        assert_eq!(state.recurrent.abs().max().into_scalar::<f32>(), 0.0);
    }

    /// Row `r` takes row `sources[r]`'s convolution inputs and delta-rule
    /// state together; the identity changes nothing.
    #[test]
    fn test_reorder() {
        let before = state();
        let mut after = state();
        after.reorder(&[2, 0, 0]);
        for (r, s) in [(0, 2), (1, 0), (2, 0)] {
            assert_eq!(
                after.conv.clone().slice_dim(0, r..=r).into_data(),
                before.conv.clone().slice_dim(0, s..=s).into_data()
            );
            assert_eq!(
                after.recurrent.clone().slice_dim(0, r..=r).into_data(),
                before.recurrent.clone().slice_dim(0, s..=s).into_data()
            );
        }
        let mut same = state();
        same.reorder(&[0, 1, 2]);
        assert_eq!(same.conv.into_data(), before.conv.into_data());
    }

    /// A reset is the start of a sequence again, at the same shapes.
    #[test]
    fn test_reset() {
        let mut state = state();
        state.reset();
        assert_eq!(state.conv.dims(), [3, 3, 5]);
        assert_eq!(state.conv.abs().max().into_scalar::<f32>(), 0.0);
        assert_eq!(state.recurrent.abs().max().into_scalar::<f32>(), 0.0);
    }

    /// A cache's reset and reorder reach every layer; a slot past its
    /// layers is refused.
    #[test]
    fn test_cache() {
        let mut cache = DeltaNetCache::new(vec![state(), state()]);
        assert!(!cache.is_empty() && DeltaNetCache::default().is_empty());
        let before = state();
        cache.reorder(&[1, 1, 0]);
        for layer in &cache.layers {
            assert_eq!(
                layer.conv.clone().slice_dim(0, 0..1).into_data(),
                before.conv.clone().slice_dim(0, 1..2).into_data()
            );
        }
        cache.reset();
        assert_eq!(
            cache
                .layer_mut(1)
                .recurrent
                .clone()
                .abs()
                .max()
                .into_scalar::<f32>(),
            0.0
        );
        // A snapshot keeps the states as they were; a restore brings them
        // back; a reset drops it.
        let mut cache = DeltaNetCache::new(vec![state()]);
        cache.take_snapshot(5);
        let kept = cache.layer_mut(0).recurrent.clone();
        cache.layer_mut(0).reset();
        assert_eq!(cache.snapshot_pos(), Some(5));
        cache.restore_snapshot();
        assert_eq!(
            cache.layer_mut(0).recurrent.clone().into_data(),
            kept.clone().into_data()
        );
        cache.reorder(&[2, 1, 0]);
        assert_eq!(
            cache.snapshot.as_ref().unwrap().layers[0]
                .recurrent
                .clone()
                .slice_dim(0, 0..1)
                .into_data(),
            kept.slice_dim(0, 2..3).into_data(),
            "the snapshot's rows reorder too"
        );
        cache.reset();
        assert_eq!(cache.snapshot_pos(), None);
        let refused = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            let _ = cache.layer_mut(2);
        }));
        assert!(refused.is_err());
    }

    #[test]
    #[should_panic(expected = "a source row past the rows")]
    fn test_reorder_past_the_rows() {
        state().reorder(&[0, 1, 3]);
    }
}
