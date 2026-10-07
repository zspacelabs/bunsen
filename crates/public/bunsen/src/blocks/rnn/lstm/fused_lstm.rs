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
        activation::{
            sigmoid,
            tanh,
        },
    },
};

use crate::burner::module::ModuleInit;

/// Common meta for [`FusedLstm`] and [`FusedLstmConfig`].
pub trait FusedLstmMeta {
    /// The recurrent hidden / cell width of the LSTM.
    fn d_model(&self) -> usize;
}

/// Config for [`FusedLstm`].
///
/// Implements [`FusedLstmMeta`].
#[derive(Config, Debug)]
pub struct FusedLstmConfig {
    /// The recurrent hidden / cell width of the LSTM.
    pub d_model: usize,

    /// Do the LSTM gate projections have a bias?
    #[config(default = "true")]
    pub bias: bool,

    /// What is the configured IO layout of the Linear weights?
    #[config(default = "burn::nn::LinearLayout::Row")]
    pub layout: burn::nn::LinearLayout,
}

impl From<FusedLstmConfig> for LinearConfig {
    fn from(config: FusedLstmConfig) -> Self {
        LinearConfig::new(config.d_model, 4 * config.d_model)
            .with_bias(config.bias)
            .with_layout(config.layout)
    }
}

impl FusedLstmMeta for FusedLstmConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }
}

impl ModuleInit<FusedLstm> for FusedLstmConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> crate::errors::BunsenResult<FusedLstm> {
        let cfg: LinearConfig = self.clone().into();
        Ok(FusedLstm {
            features: cfg.init(device),
            hidden: cfg.init(device),
        })
    }
}

/// A fused LSTM block.
///
/// This is the fused LSTM used by the Silero VAD model.
/// It is an open question as to alignment/abstraction of this
/// with the common/generic burn.nn.Lstm module.
///
/// Built by [`FusedLstmConfig`].
/// Implements [`FusedLstmMeta`].
#[derive(Module, Debug)]
pub struct FusedLstm {
    /// The LSTM input (feature -> gates) projection.
    pub features: Linear,

    /// The LSTM recurrent (hidden -> gates) projection.
    pub hidden: Linear,
}

impl FusedLstmMeta for FusedLstm {
    fn d_model(&self) -> usize {
        self.features.weight.dims()[0]
    }
}

impl FusedLstm {
    /// Runs one LSTM step.
    ///
    /// # Arguments
    ///
    /// * `feature` - `[batch, d_model]` encoder feature frame.
    /// * `cell` - `[batch, d_model]` previous cell state.
    /// * `hidden` - `[batch, d_model]` previous hidden state.
    ///
    /// # Returns
    ///
    /// The `(hidden, cell)` next states, each `[batch, d_model]`.
    pub fn step(
        &self,
        features: Tensor<2>,
        hidden: Tensor<2>,
        cell: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        #[cfg(any(test, debug_assertions))]
        use crate::contracts::{
            assert_shape_contract_periodically,
            unpack_shape_contract,
        };
        #[cfg(any(test, debug_assertions))]
        let batch = {
            let [batch] = unpack_shape_contract!(
                ["batch", "d_model"],
                &features,
                &["batch"],
                &[("d_model", self.d_model())],
            );
            assert_shape_contract_periodically!(
                ["batch", "d_model"],
                &cell,
                &[("batch", batch), ("d_model", self.d_model())]
            );
            assert_shape_contract_periodically!(
                ["batch", "d_model"],
                &hidden,
                &[("batch", batch), ("d_model", self.d_model())]
            );
            batch
        };

        // Gates: recurrent projection of `hidden` plus input projection of
        // `feature`, split into [input, forget, cell, output] gates.
        let gates = self.features.forward(features) + self.hidden.forward(hidden);

        #[cfg(any(test, debug_assertions))]
        assert_shape_contract_periodically!(
            ["batch", 4 * "d_model"],
            &gates,
            &[("batch", batch), ("d_model", self.d_model())]
        );

        /*
        let [g_i, g_f, g_c, g_o] = gates.chunk(4, 1).try_into().unwrap();

        let input_values = sigmoid(g_i);
        let forget_values = sigmoid(g_f);
         */

        // Funny chunking to reduce sigmoid() calls.
        let [g_i_f, g_c_o] = gates.chunk(2, 1).try_into().unwrap();
        let [g_c, g_o] = g_c_o.chunk(2, 1).try_into().unwrap();

        let [input_values, forget_values] = sigmoid(g_i_f).chunk(2, 1).try_into().unwrap();
        let candidate_cell_values = tanh(g_c);
        let output_values = sigmoid(g_o);

        let cell = forget_values * cell + input_values * candidate_cell_values;
        let hidden = output_values * tanh(cell.clone());
        (hidden, cell)
    }
}
