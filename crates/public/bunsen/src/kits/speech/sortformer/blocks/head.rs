//! A Sortformer's speaker head.
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
        activation::relu,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`SortformerHead`] and [`SortformerHeadConfig`].
pub trait SortformerHeadMeta {
    /// The width it reads.
    fn d_hidden(&self) -> usize;

    /// The speakers it scores.
    fn n_speakers(&self) -> usize;
}

/// Config for [`SortformerHead`].
///
/// Implements [`SortformerHeadMeta`].
#[derive(Config, Debug)]
pub struct SortformerHeadConfig {
    /// The width it reads.
    pub d_hidden: usize,

    /// The speakers it scores.
    pub n_speakers: usize,
}

impl SortformerHeadMeta for SortformerHeadConfig {
    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn n_speakers(&self) -> usize {
        self.n_speakers
    }
}

impl ModuleInit<SortformerHead> for SortformerHeadConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SortformerHead> {
        Ok(SortformerHead {
            dense: LinearConfig::new(self.d_hidden, self.d_hidden).init(device),
            out: LinearConfig::new(self.d_hidden, self.n_speakers).init(device),
        })
    }
}

/// A Sortformer's speaker head: `out(relu(dense(relu(x))))`, one logit per
/// speaker per frame (`NeMo`'s `forward_speaker_logits`, `transformers`'
/// classifier).
///
/// Built by [`SortformerHeadConfig`].
/// Implements [`SortformerHeadMeta`].
#[derive(Module, Debug)]
pub struct SortformerHead {
    /// `[d_hidden] -> [d_hidden]`.
    pub dense: Linear,

    /// `[d_hidden] -> [n_speakers]`.
    pub out: Linear,
}

impl SortformerHeadMeta for SortformerHead {
    fn d_hidden(&self) -> usize {
        self.dense.weight.dims()[0]
    }

    fn n_speakers(&self) -> usize {
        self.out.weight.dims()[1]
    }
}

impl SortformerHead {
    /// The speaker logits.
    ///
    /// # Arguments
    /// * `x` - `[batch, frames, d_hidden]`.
    ///
    /// # Returns
    /// `[batch, frames, n_speakers]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        self.out.forward(relu(self.dense.forward(relu(x))))
    }
}
