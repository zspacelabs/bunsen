//! The `FastConformer` encoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use super::{
    ConformerBlock,
    ConformerBlockConfig,
    DwStridingSubsampling,
    DwStridingSubsamplingConfig,
    DwStridingSubsamplingMeta,
};
use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::{
        length_mask,
        rel_position_table,
    },
};

/// Common meta for [`ConformerEncoder`] and [`ConformerEncoderConfig`].
pub trait ConformerEncoderMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The number of Conformer blocks.
    fn n_layers(&self) -> usize;

    /// The time reduction of the subsampling.
    fn subsampling_factor(&self) -> usize;
}

/// Config for [`ConformerEncoder`].
///
/// Implements [`ConformerEncoderMeta`].
#[derive(Config, Debug)]
pub struct ConformerEncoderConfig {
    /// The subsampling front of the encoder.
    pub subsampling: DwStridingSubsamplingConfig,

    /// The block every layer is built from.
    pub block: ConformerBlockConfig,

    /// The number of Conformer blocks.
    pub n_layers: usize,

    /// Whether the subsampled input is scaled by `sqrt(d_model)` (`NeMo`'s
    /// `xscaling`).
    #[config(default = "false")]
    pub scale_input: bool,
}

impl ConformerEncoderMeta for ConformerEncoderConfig {
    fn d_model(&self) -> usize {
        self.block.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling.factor()
    }
}

impl ModuleInit<ConformerEncoder> for ConformerEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConformerEncoder> {
        if self.subsampling.d_model != self.block.d_model {
            return Err(ConstraintError::new(
                "ConformerEncoderConfig",
                "",
                Rule::Relation {
                    lhs: (
                        "subsampling.d_model".into(),
                        self.subsampling.d_model.to_string(),
                    ),
                    op: "==",
                    rhs: ("block.d_model".into(), self.block.d_model.to_string()),
                },
            )
            .into());
        }
        Ok(ConformerEncoder {
            subsampling: self.subsampling.try_init(device)?,
            layers: (0..self.n_layers)
                .map(|_| self.block.try_init(device))
                .collect::<BunsenResult<_>>()?,
            scale_input: self.scale_input,
        })
    }
}

/// The `FastConformer` encoder: [`DwStridingSubsampling`], then a stack of
/// [`ConformerBlock`]s sharing one relative-position table.
///
/// This is the encoder of NVIDIA's Parakeet and Canary models.
///
/// Built by [`ConformerEncoderConfig`].
/// Implements [`ConformerEncoderMeta`].
#[derive(Module, Debug)]
pub struct ConformerEncoder {
    /// Subsamples the features and projects them to `d_model`.
    pub subsampling: DwStridingSubsampling,

    /// The Conformer blocks.
    pub layers: Vec<ConformerBlock>,

    /// Whether the subsampled input is scaled by `sqrt(d_model)`.
    pub scale_input: bool,
}

impl ConformerEncoderMeta for ConformerEncoder {
    fn d_model(&self) -> usize {
        self.subsampling.d_model()
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling.factor()
    }
}

impl ConformerEncoder {
    /// The encoder's output length for `len` input frames.
    pub fn output_len(
        &self,
        len: usize,
    ) -> usize {
        self.subsampling.output_len(len)
    }

    /// Encodes features.
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, time]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]`, and each item's output length.
    /// Frames past an item's length hold values that mean nothing.
    pub fn forward(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        #[cfg(any(debug_assertions, test))]
        let [batch, time_in] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_mels", "time"],
            &features.dims(),
            &["batch", "time"],
        );

        let padded = lengths.is_some();
        let (mut x, lengths) = self.subsampling.forward(features, lengths);
        if self.scale_input {
            x = x.mul_scalar((self.d_model() as f64).sqrt());
        }
        let [_, time, d_model] = x.dims();
        let device = x.device();
        let positions = rel_position_table(time, d_model, &device);
        let valid = padded.then(|| length_mask(&lengths, time, &device));
        for layer in &self.layers {
            x = layer.forward(x, positions.clone(), valid.clone());
        }

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &[
                ("batch", batch),
                ("time", self.output_len(time_in)),
                ("d_model", self.d_model()),
            ],
        );

        (x, lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 16),
            ConformerBlockConfig::new(16, 2, 32, 5),
            2,
        );
        let module: ConformerEncoder = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_layers(), config.n_layers());
        assert_eq!(module.subsampling_factor(), config.subsampling_factor());

        let (y, lengths) = module.forward(Tensor::zeros([1, 24, 20], &cpu_device()), None);
        assert_eq!(y.dims(), [1, 3, 16]);
        assert_eq!(lengths, [3]);
    }

    #[test]
    fn test_widths_must_agree() {
        let config = ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 12),
            ConformerBlockConfig::new(16, 2, 32, 5),
            2,
        );
        assert!(ModuleInit::<ConformerEncoder>::try_init(&config, &cpu_device()).is_err());
    }
}
