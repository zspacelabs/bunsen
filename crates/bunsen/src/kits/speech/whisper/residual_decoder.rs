use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        attention::{
            MhaInput,
            MhaOutput,
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
    },
    prelude::{
        Backend,
        Bool,
    },
};

use crate::blocks::transformers::mlp::{
    Mlp,
    MlpConfig,
};

/// Common meta for [`ResidualDecoderAttentionBlock`] and
/// [`ResidualDecoderAttentionBlockConfig`].
pub trait ResidualDecoderAttentionBlockMeta {
    /// Return the number of states.
    fn n_states(&self) -> usize;

    /// Return the number of heads.
    fn n_heads(&self) -> usize;
}

/// Config for [`ResidualDecoderAttentionBlock`].
#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    /// Number of States.
    pub n_states: usize,

    /// Number of Heads.
    pub n_heads: usize,
}

impl ResidualDecoderAttentionBlockMeta for ResidualDecoderAttentionBlockConfig {
    fn n_states(&self) -> usize {
        self.n_states
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl ResidualDecoderAttentionBlockConfig {
    /// Initialize the residual decoder attention block.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ResidualDecoderAttentionBlock<B> {
        let mha_cfg = MultiHeadAttentionConfig::new(self.n_states, self.n_heads).with_dropout(0.0);
        let ln_cfg = LayerNormConfig::new(self.n_states);

        ResidualDecoderAttentionBlock {
            attn_ln: ln_cfg.init(device),
            attn: mha_cfg.init(device),
            cross_attn_ln: ln_cfg.init(device),
            cross_attn: mha_cfg.init(device),
            mlp_ln: ln_cfg.init(device),
            mlp: MlpConfig::new(self.n_states).init(device),
        }
    }
}

/// Residual Decoder Attention Block for Whisper.
#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    /// Attention Normalization.
    pub attn_ln: LayerNorm<B>,

    /// Attention.
    pub attn: MultiHeadAttention<B>,

    /// Cross Attention Normalization.
    pub cross_attn_ln: LayerNorm<B>,

    /// Cross Attention.
    pub cross_attn: MultiHeadAttention<B>,

    /// MLP Normalization.
    pub mlp_ln: LayerNorm<B>,

    /// MLP.
    pub mlp: Mlp<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlockMeta for ResidualDecoderAttentionBlock<B> {
    fn n_states(&self) -> usize {
        self.attn.d_model
    }

    fn n_heads(&self) -> usize {
        self.attn.n_heads
    }
}

/// Forward record for [`ResidualDecoderAttentionBlock::forward`].
#[derive(Debug, Clone)]
pub struct RdabForwardRecord<B: Backend> {
    /// Block Output: ``[batch, seq_len, n_states]``.
    pub output: Tensor<B, 3>,

    /// Cross-Attention Weights: ``[batch, n_heads, seq_len, seq_len]``.
    pub ca_weights: Tensor<B, 4>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    /// Forward pass of the residual decoder attention block.
    ///
    /// ## Arguments
    /// * `x` - ``[batch, seq_len, n_states]`` input.
    /// * `xa` - ``[batch, seq_len, n_states]`` cross-attention input.
    /// * `mask` - ``[batch, seq_len, seq_len]`` attention mask.
    ///
    /// ## Returns
    /// `RdabForwardRecord` - forward record.
    /// * `fr.output` : ``[batch, seq_len, n_states]``.
    /// * `fr.ca_weights` : ``[batch, n_heads, seq_len, seq_len]``.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> RdabForwardRecord<B> {
        #[cfg(any(debug_assertions, test))]
        let (batch, seq_len) = {
            crate::contracts::define_shape_contract!(CONTRACT, ["batch", "seq_len", "n_states"]);

            let [batch, seq_len] =
                CONTRACT.unpack_shape(&x, &["batch", "seq_len"], &[("n_states", self.n_states())]);

            CONTRACT.assert_shape(
                &xa,
                &[
                    ("batch", batch),
                    ("seq_len", seq_len),
                    ("n_states", self.n_states()),
                ],
            );

            crate::contracts::assert_shape_contract!(
                ["*", "seq_len", "seq_len"],
                &mask,
                &[("seq_len", seq_len)]
            );

            (batch, seq_len)
        };

        let self_attn = self.self_attn_pass(x.clone(), mask.clone());
        let x = x + self_attn.context;

        let cross_attn = self.cross_attn_pass(x.clone(), xa.clone());
        let x = x + cross_attn.context;

        let mlp = self.mlp_pass(x.clone());
        let x = x + mlp;

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract!(
            ["batch", "n_heads", "seq_len", "seq_len"],
            &cross_attn.weights,
            &[
                ("batch", batch),
                ("n_heads", self.n_heads()),
                ("seq_len", seq_len)
            ],
        );

        RdabForwardRecord {
            output: x,
            ca_weights: cross_attn.weights,
        }
    }

    /// Compute the normalized self-attn
    ///
    /// ## Arguments
    /// * `x` - ``[batch, seq_len, n_states]`` input.
    /// * `mask` - ``[batch, seq_len, seq_len]`` attention mask.
    ///
    /// ## Returns
    /// `RdabForwardRecord` - forward record.
    /// * `fr.output` : ``[batch, seq_len, n_states]``.
    /// * `fr.ca_weights` : ``[batch, n_heads, seq_len, seq_len]``.
    pub fn self_attn_pass(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> MhaOutput<B> {
        self.attn
            .forward(MhaInput::self_attn(self.attn_ln.forward(x)).mask_attn(mask))
    }

    /// Compute the normalized cross-attn
    ///
    /// ## Arguments
    /// * `x` - ``[batch, seq_len, n_states]`` input.
    /// * `xa` - ``[batch, seq_len, n_states]`` cross-attention input.
    ///
    /// ## Returns
    /// `RdabForwardRecord` - forward record.
    /// * `fr.output` : ``[batch, seq_len, n_states]``.
    /// * `fr.ca_weights` : ``[batch, n_heads, seq_len, seq_len]``.
    pub fn cross_attn_pass(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
    ) -> MhaOutput<B> {
        self.cross_attn.forward(MhaInput::new(
            self.cross_attn_ln.forward(x.clone()),
            xa.clone(),
            xa,
        ))
    }

    /// Compute the normalized mlp
    ///
    /// ## Arguments
    /// * `x` - ``[batch, seq_len, n_states]`` input.
    ///
    /// ## Returns
    /// ``[batch, seq_len, n_states]``
    pub fn mlp_pass(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.mlp.forward(self.mlp_ln.forward(x))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Shape,
        tensor::Distribution,
    };

    use super::*;
    use crate::contracts::assert_shape_contract;

    #[test]
    #[serial_test::serial]
    fn test_residual_decoder_forward() {
        type B = crate::support::testing::PerformanceBackend;
        let device = Default::default();

        let n_heads = 4;
        let n_states = 32 * n_heads;

        let cfg = ResidualDecoderAttentionBlockConfig::new(n_states, n_heads);

        assert_eq!(cfg.n_states(), n_states);
        assert_eq!(cfg.n_heads(), n_heads);

        let block: ResidualDecoderAttentionBlock<B> = cfg.init(&device);

        assert_eq!(block.n_states(), n_states);
        assert_eq!(block.n_heads(), n_heads);

        let batch = 2;
        let seq_len = 10;
        let shape: Shape = [batch, seq_len, n_states].into();

        let x: Tensor<B, 3> = Tensor::random(shape.clone(), Distribution::Default, &device);
        let xa: Tensor<B, 3> = Tensor::random(shape.clone(), Distribution::Default, &device);

        let mask: Tensor<B, 3> =
            Tensor::random([1, seq_len, seq_len], Distribution::Bernoulli(0.5), &device);
        let mask = mask.bool();

        let fr = block.forward(x.clone(), xa.clone(), mask.clone());

        let expected = {
            let self_attn = block.self_attn_pass(x.clone(), mask.clone());
            let x = x + self_attn.context;

            let cross_attn = block.cross_attn_pass(x.clone(), xa.clone());
            let x = x + cross_attn.context;

            let mlp = block.mlp_pass(x.clone());
            let x = x + mlp;

            RdabForwardRecord::<B> {
                output: x,
                ca_weights: cross_attn.weights,
            }
        };

        fr.output
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.output.clone().into_data(), Default::default());
        fr.ca_weights
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.ca_weights.clone().into_data(), Default::default());

        assert_shape_contract!(
            ["batch", "seq_len", "n_states"],
            &fr.output,
            &[
                ("batch", batch),
                ("seq_len", seq_len),
                ("n_states", n_states),
            ],
        );

        assert_shape_contract!(
            ["batch", "n_heads", "seq_len", "seq_len"],
            &fr.ca_weights,
            &[("batch", batch), ("n_heads", n_heads), ("seq_len", seq_len)],
        );
    }
}
