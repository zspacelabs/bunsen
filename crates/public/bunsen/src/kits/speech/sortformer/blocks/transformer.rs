//! The transformer a Sortformer runs between its projection and its head.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    blocks::transformers::encoders::residual::{
        ResidualEncoderAttentionBlock,
        ResidualEncoderAttentionBlockConfig,
        ResidualEncoderAttentionBlockMeta,
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    ops::transformers::attention::length_mask,
};

/// Config for [`SortformerTransformer`]: `n_layers` of one block.
#[derive(Config, Debug)]
pub struct SortformerTransformerConfig {
    /// Every layer's block.
    pub block: ResidualEncoderAttentionBlockConfig,

    /// The layers.
    pub n_layers: usize,
}

impl SortformerTransformerConfig {
    /// The width it reads and writes.
    pub fn d_model(&self) -> usize {
        self.block.d_model()
    }
}

impl ModuleInit<SortformerTransformer> for SortformerTransformerConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SortformerTransformer> {
        Ok(SortformerTransformer {
            layers: (0..self.n_layers)
                .map(|_| self.block.try_init(device))
                .collect::<BunsenResult<_>>()?,
        })
    }
}

/// A stack of [`ResidualEncoderAttentionBlock`]s over the projected frames,
/// without position embeddings: Sortformer v2's post-norm `ReLU` transformer
/// (`NeMo`'s `transformer_encoders.TransformerEncoder`). Padding frames are
/// masked out as keys, as `NeMo`'s padding mask does.
///
/// Built by [`SortformerTransformerConfig`].
#[derive(Module, Debug)]
pub struct SortformerTransformer {
    /// The layers.
    pub layers: Vec<ResidualEncoderAttentionBlock>,
}

impl SortformerTransformer {
    /// The width it reads and writes.
    pub fn d_model(&self) -> usize {
        self.layers.first().map_or(0, |l| l.d_model())
    }

    /// Runs the layers.
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
        let [batch, frames] = unpack_shape_contract!(
            ["batch", "frames", "d_model"],
            &x.dims(),
            &["batch", "frames"],
            &[("d_model", self.d_model())],
        );
        // `[batch, frames, frames]`, true where key `j` is padding.
        let mask = lengths.map(|lengths| {
            length_mask(lengths, frames, &x.device())
                .bool_not()
                .unsqueeze_dim::<3>(1)
                .expand([batch, frames, frames])
        });
        self.layers
            .iter()
            .fold(x, |x, layer| layer.forward_masked(x, mask.clone()))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        blocks::transformers::encoders::residual::NormPlacement,
        support::testing::cpu_device,
    };

    fn toy() -> SortformerTransformer {
        SortformerTransformerConfig::new(
            ResidualEncoderAttentionBlockConfig::new(8)
                .with_d_head(4)
                .with_key_bias(true)
                .with_norm_placement(NormPlacement::Post),
            2,
        )
        .init(&cpu_device())
    }

    /// Each layer runs in turn, and padding is masked out as keys: a short
    /// item's valid frames, batched behind a longer one, are what it gives
    /// alone.
    #[test]
    fn test_layers_and_padding() {
        let device = cpu_device();
        let model = toy();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.d_model(), 8);

        let x = Tensor::<3>::random([2, 6, 8], Distribution::Default, &device);
        let whole = model.forward(x.clone(), None);
        let by_hand = model.layers[1].forward(model.layers[0].forward(x.clone()));
        whole
            .to_data()
            .assert_approx_eq::<f32>(&by_hand.to_data(), Tolerance::absolute(1e-6));

        let batched = model.forward(x.clone(), Some(&[6, 4]));
        let alone = model.forward(x.slice([1..2, 0..4]), None);
        batched
            .slice([1..2, 0..4])
            .to_data()
            .assert_approx_eq::<f32>(&alone.to_data(), Tolerance::absolute(1e-5));
    }
}
