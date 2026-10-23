//! The Conformer's block stack, past its subsampling.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use super::{
    ConformerBlock,
    ConformerBlockConfig,
    ConformerBlockMeta,
};
use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::transformers::attention::{
        length_mask,
        rel_position_table,
    },
};

/// Common meta for [`ConformerTransformer`] and
/// [`ConformerTransformerConfig`].
pub trait ConformerTransformerMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The number of Conformer blocks.
    fn n_layers(&self) -> usize;
}

/// Config for [`ConformerTransformer`]: `n_layers` of one block.
///
/// Implements [`ConformerTransformerMeta`].
#[derive(Config, Debug)]
pub struct ConformerTransformerConfig {
    /// The block every layer is built from.
    pub block: ConformerBlockConfig,

    /// The number of Conformer blocks.
    pub n_layers: usize,

    /// Whether the input is scaled by `sqrt(d_model)` (`NeMo`'s `xscaling`).
    #[config(default = "false")]
    pub scale_input: bool,
}

impl ConformerTransformerMeta for ConformerTransformerConfig {
    fn d_model(&self) -> usize {
        self.block.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

impl ModuleInit<ConformerTransformer> for ConformerTransformerConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConformerTransformer> {
        Ok(ConformerTransformer {
            layers: (0..self.n_layers)
                .map(|_| self.block.try_init(device))
                .collect::<BunsenResult<_>>()?,
            scale_input: self.scale_input,
        })
    }
}

/// The Conformer's stack: [`ConformerBlock`]s sharing one relative-position
/// table over the frames given, after an optional input scaling. The frame
/// rate in is the frame rate out; a subsampling front, if any, runs before
/// (as a [`ConformerEncoderConfig`](super::ConformerEncoderConfig)'s encoder
/// runs one).
///
/// Built by [`ConformerTransformerConfig`].
/// Implements [`ConformerTransformerMeta`].
#[derive(Module, Debug)]
pub struct ConformerTransformer {
    /// The Conformer blocks.
    pub layers: Vec<ConformerBlock>,

    /// Whether the input is scaled by `sqrt(d_model)`.
    pub scale_input: bool,
}

impl ConformerTransformerMeta for ConformerTransformer {
    fn d_model(&self) -> usize {
        self.layers.first().map_or(0, |l| l.d_model())
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl ConformerTransformer {
    /// The input scaling (when it scales), the relative-position table over
    /// the frames given, and the Conformer blocks.
    ///
    /// # Arguments
    /// * `x` - `[batch, frames, d_model]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    ///
    /// # Returns
    /// `[batch, frames, d_model]`. Frames past an item's length hold values
    /// that mean nothing.
    pub fn forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        let d_model = self.d_model();
        let [batch, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "d_model"],
            &x.dims(),
            &["batch", "frames"],
            &[("d_model", d_model)],
        );

        let mut x = x;
        if self.scale_input {
            x = x.mul_scalar((d_model as f64).sqrt());
        }
        let device = x.device();
        let positions = rel_position_table(frames, d_model, &device);
        let valid = lengths.map(|lengths| length_mask(lengths, frames, &device));
        for layer in &self.layers {
            x = layer.forward(x, positions.clone(), valid.clone());
        }

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "frames", "d_model"],
            &x.dims(),
            &[("batch", batch), ("frames", frames), ("d_model", d_model)],
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;

        x
    }
}
