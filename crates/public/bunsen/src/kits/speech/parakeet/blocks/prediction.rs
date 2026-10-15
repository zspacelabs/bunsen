//! The TDT prediction network.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        Linear,
        LinearConfig,
    },
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::rnn::lstm::{
        FusedLstm,
        FusedLstmConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common meta for [`TdtPredictionNetwork`] and
/// [`TdtPredictionNetworkConfig`].
pub trait TdtPredictionNetworkMeta {
    /// The embedding rows: the vocabulary and the blank.
    fn n_vocab(&self) -> usize;

    /// The network's width: embedding, LSTM state and output.
    fn d_pred(&self) -> usize;

    /// The LSTM layer count.
    fn n_layers(&self) -> usize;
}

/// Config for [`TdtPredictionNetwork`].
///
/// Implements [`TdtPredictionNetworkMeta`].
#[derive(Config, Debug)]
pub struct TdtPredictionNetworkConfig {
    /// The embedding rows: the vocabulary and the blank.
    pub n_vocab: usize,

    /// The network's width.
    pub d_pred: usize,

    /// The LSTM layer count.
    #[config(default = "2")]
    pub n_layers: usize,
}

impl TdtPredictionNetworkMeta for TdtPredictionNetworkConfig {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn d_pred(&self) -> usize {
        self.d_pred
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

impl ModuleInit<TdtPredictionNetwork> for TdtPredictionNetworkConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TdtPredictionNetwork> {
        if self.n_layers == 0 {
            return Err(ConstraintError::new(
                "TdtPredictionNetworkConfig",
                "n_layers",
                Rule::ZeroOrEmpty,
            )
            .into());
        }
        Ok(TdtPredictionNetwork {
            embedding: EmbeddingConfig::new(self.n_vocab, self.d_pred).init(device),
            lstm: (0..self.n_layers)
                .map(|_| FusedLstmConfig::new(self.d_pred).try_init(device))
                .collect::<BunsenResult<_>>()?,
            projector: LinearConfig::new(self.d_pred, self.d_pred).init(device),
        })
    }
}

/// The recurrent state of a [`TdtPredictionNetwork`]: one hidden and one
/// cell state per LSTM layer.
///
/// A decode opens one with [`TdtPredictionNetwork::new_state`] and passes it
/// back in at every step; the network never holds it.
#[derive(Clone, Debug)]
pub struct TdtPredictionState {
    /// `[n_layers, batch, d_pred]`.
    pub hidden: Tensor<3>,

    /// `[n_layers, batch, d_pred]`.
    pub cell: Tensor<3>,
}

impl TdtPredictionState {
    /// The batch size.
    pub fn batch(&self) -> usize {
        self.hidden.dims()[1]
    }

    /// Takes `next`'s rows where `mask` is true, and keeps this state's
    /// elsewhere.
    ///
    /// # Arguments
    /// * `mask` - `[batch]`, true for the rows to take from `next`.
    pub fn select(
        self,
        mask: Tensor<1, burn::prelude::Bool>,
        next: Self,
    ) -> Self {
        let [layers, batch, d] = self.hidden.dims();
        let mask = mask.reshape([1, batch, 1]).expand([layers, batch, d]);
        Self {
            hidden: self.hidden.mask_where(mask.clone(), next.hidden),
            cell: self.cell.mask_where(mask, next.cell),
        }
    }
}

/// The prediction network of a token-and-duration transducer (TDT): an
/// embedding of the previous token, a stack of LSTMs, and a projection.
///
/// The transducer's language model. It reads only the tokens emitted so far
/// (the blank, which starts every decode, embeds to the decoder's "nothing
/// yet" input) and summarizes them for the
/// [`TdtJoint`](super::TdtJoint).
///
/// Built by [`TdtPredictionNetworkConfig`].
/// Implements [`TdtPredictionNetworkMeta`].
#[derive(Module, Debug)]
pub struct TdtPredictionNetwork {
    /// The token embedding, `n_vocab` rows.
    pub embedding: Embedding,

    /// The LSTM stack.
    pub lstm: Vec<FusedLstm>,

    /// The output projection.
    pub projector: Linear,
}

impl TdtPredictionNetworkMeta for TdtPredictionNetwork {
    fn n_vocab(&self) -> usize {
        self.embedding.weight.dims()[0]
    }

    fn d_pred(&self) -> usize {
        self.embedding.weight.dims()[1]
    }

    fn n_layers(&self) -> usize {
        self.lstm.len()
    }
}

impl TdtPredictionNetwork {
    /// A zero state for `batch` decodes.
    pub fn new_state(
        &self,
        batch: usize,
        device: &Device,
    ) -> TdtPredictionState {
        let shape = [self.n_layers(), batch, self.d_pred()];
        TdtPredictionState {
            hidden: Tensor::zeros(shape, device),
            cell: Tensor::zeros(shape, device),
        }
    }

    /// Advances the network by one token.
    ///
    /// # Arguments
    /// * `tokens` - `[batch]`, the previous token of each decode.
    /// * `state` - the state before it.
    ///
    /// # Returns
    /// `[batch, d_pred]`, the network's output, and the state after it.
    pub fn step(
        &self,
        tokens: Tensor<1, Int>,
        state: TdtPredictionState,
    ) -> (Tensor<2>, TdtPredictionState) {
        let [batch] = tokens.dims();
        let d = self.d_pred();
        let mut x = self
            .embedding
            .forward(tokens.unsqueeze_dim::<2>(1))
            .reshape([batch, d]);
        let mut hidden = Vec::with_capacity(self.lstm.len());
        let mut cell = Vec::with_capacity(self.lstm.len());
        for (l, lstm) in self.lstm.iter().enumerate() {
            let h = state
                .hidden
                .clone()
                .slice_dim(0, l..l + 1)
                .reshape([batch, d]);
            let c = state
                .cell
                .clone()
                .slice_dim(0, l..l + 1)
                .reshape([batch, d]);
            let (h, c) = lstm.step(x, h, c);
            x = h.clone();
            hidden.push(h.unsqueeze_dim::<3>(0));
            cell.push(c.unsqueeze_dim::<3>(0));
        }
        let state = TdtPredictionState {
            hidden: Tensor::cat(hidden, 0),
            cell: Tensor::cat(cell, 0),
        };
        (self.projector.forward(x), state)
    }

    /// Runs the network over whole token sequences from a zero state.
    ///
    /// # Arguments
    /// * `tokens` - `[batch, len]`.
    ///
    /// # Returns
    /// `[batch, len, d_pred]`: the output after each token.
    pub fn forward(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        let [batch, len] = tokens.dims();
        let mut state = self.new_state(batch, &tokens.device());
        let mut outputs = Vec::with_capacity(len);
        for u in 0..len {
            let step = tokens.clone().slice_dim(1, u..u + 1).reshape([batch]);
            let (y, next) = self.step(step, state);
            state = next;
            outputs.push(y.unsqueeze_dim::<3>(1));
        }
        Tensor::cat(outputs, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = TdtPredictionNetworkConfig::new(11, 6).with_n_layers(3);
        let module: TdtPredictionNetwork = config.init(&cpu_device());
        assert_eq!(module.n_vocab(), config.n_vocab());
        assert_eq!(module.d_pred(), config.d_pred());
        assert_eq!(module.n_layers(), config.n_layers());

        let tokens = Tensor::<2, Int>::from_data([[1, 2, 3], [4, 5, 10]], &cpu_device());
        assert_eq!(module.forward(tokens).dims(), [2, 3, 6]);
    }

    /// `select` takes the masked rows of the second state and keeps the rest.
    #[test]
    fn test_state_select() {
        let device = cpu_device();
        let zeros = TdtPredictionState {
            hidden: Tensor::zeros([2, 3, 4], &device),
            cell: Tensor::zeros([2, 3, 4], &device),
        };
        let ones = TdtPredictionState {
            hidden: Tensor::ones([2, 3, 4], &device),
            cell: Tensor::ones([2, 3, 4], &device),
        };
        let mask = Tensor::<1, burn::prelude::Bool>::from_data([true, false, true], &device);
        let picked = zeros.select(mask, ones);
        let rows: Vec<f32> = picked
            .hidden
            .slice_dim(0, 1..2)
            .slice_dim(2, 0..1)
            .into_data()
            .try_into_vec_as::<f32>()
            .unwrap();
        assert_eq!(rows, [1.0, 0.0, 1.0]);
    }
}
