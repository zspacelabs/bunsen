//! The Voxtral Realtime audio projector.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::{
        Device,
        activation::gelu,
    },
};

use crate::{
    blocks::signals::{
        FeatureStacking,
        FeatureStackingConfig,
        FeatureStackingMeta,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`VoxtralProjector`] and [`VoxtralProjectorConfig`].
pub trait VoxtralProjectorMeta {
    /// The audio encoder's width.
    fn d_audio(&self) -> usize;

    /// The text model's width.
    fn d_text(&self) -> usize;

    /// The encoder frames stacked into one decoder position.
    fn downsample(&self) -> usize;
}

/// Config for [`VoxtralProjector`].
///
/// Implements [`VoxtralProjectorMeta`].
#[derive(Config, Debug)]
pub struct VoxtralProjectorConfig {
    /// The audio encoder's width.
    pub d_audio: usize,

    /// The text model's width.
    pub d_text: usize,

    /// The encoder frames stacked into one decoder position.
    #[config(default = "4")]
    pub downsample: usize,
}

impl VoxtralProjectorMeta for VoxtralProjectorConfig {
    fn d_audio(&self) -> usize {
        self.d_audio
    }

    fn d_text(&self) -> usize {
        self.d_text
    }

    fn downsample(&self) -> usize {
        self.downsample
    }
}

impl ModuleInit<VoxtralProjector> for VoxtralProjectorConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VoxtralProjector> {
        Ok(VoxtralProjector {
            stacking: FeatureStackingConfig::new(self.d_audio, self.d_text)
                .with_factor(self.downsample)
                .try_init(device)?,
            linear_2: LinearConfig::new(self.d_text, self.d_text)
                .with_bias(false)
                .init(device),
        })
    }
}

/// Encoder frames into the text model's embedding space, one per decoder
/// position: `downsample` consecutive frames stacked side by side and
/// projected (`linear_1`), GELU, and a second projection (`linear_2`);
/// `transformers`' `multi_modal_projector`, unbiased.
///
/// The stacking is the shared
/// [`FeatureStacking`](crate::blocks::signals::FeatureStacking), frames
/// outer, whose projection is `transformers`' `linear_1`.
///
/// Built by [`VoxtralProjectorConfig`].
/// Implements [`VoxtralProjectorMeta`].
#[derive(Module, Debug)]
pub struct VoxtralProjector {
    /// `downsample` frames stacked, then `d_audio * downsample -> d_text`.
    pub stacking: FeatureStacking,

    /// `d_text -> d_text`.
    pub linear_2: Linear,
}

impl VoxtralProjectorMeta for VoxtralProjector {
    fn d_audio(&self) -> usize {
        self.stacking.d_input()
    }

    fn d_text(&self) -> usize {
        self.linear_2.weight.dims()[1]
    }

    fn downsample(&self) -> usize {
        self.stacking.factor()
    }
}

impl VoxtralProjector {
    /// Projects encoder frames.
    ///
    /// # Arguments
    /// * `x` - `[batch, frames, d_audio]`, `frames` a multiple of `downsample`.
    ///
    /// # Returns
    /// `[batch, frames/downsample, d_text]`.
    ///
    /// # Panics
    /// If `frames` is not a multiple of `downsample`: a partial group has no
    /// decoder position.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "d_audio"],
            &x.dims(),
            &["batch", "frames"],
            &[("d_audio", self.d_audio())]
        );
        assert!(
            frames.is_multiple_of(self.downsample()),
            "{frames} encoder frames are not whole groups of {}",
            self.downsample()
        );
        let (stacked, _) = self.stacking.forward(x, None);
        let y = self.linear_2.forward(gelu(stacked));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "positions", "d_text"],
            &y.dims(),
            &[
                ("batch", batch),
                ("positions", frames / self.downsample()),
                ("d_text", self.d_text())
            ]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
        y
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// A config non-default in every field, and the projector built from it,
    /// answer the meta alike; the projection is `linear_2(gelu(linear_1(
    /// stacked)))`, frames outer.
    #[test]
    fn test_meta_and_forward() {
        let device = cpu_device();
        let config = VoxtralProjectorConfig::new(5, 7).with_downsample(3);
        let projector: VoxtralProjector = config.init(&device);
        assert_eq!(projector.d_audio(), config.d_audio());
        assert_eq!(projector.d_text(), config.d_text());
        assert_eq!(projector.downsample(), config.downsample());
        assert!(projector.stacking.projection.bias.is_none());
        assert!(projector.linear_2.bias.is_none());

        let x = seeded_tensor::<3>(1, [2, 6, 5], Distribution::Default, &device);
        let stacked = x.clone().reshape([2, 2, 15]);
        let expected = projector
            .linear_2
            .forward(gelu(projector.stacking.projection.forward(stacked)));
        assert_tensors_close(&projector.forward(x), &expected, Tolerance::default());
    }
}
