//! Reference ONNX Exported Silero VAD Model.

use burn::{
    Tensor,
    module::{
        Module,
        Param,
        ParamId,
    },
    nn::{
        Linear,
        LinearConfig,
        LinearLayout,
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::{
        Backend,
        Int,
        s,
    },
    tensor::{
        Bytes,
        activation::{
            relu,
            sigmoid,
            tanh,
        },
        ops::PadMode,
    },
};
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
};

/// Reference model for Silero VAD.
#[derive(Module, Debug)]
pub struct ReferenceVAD<B: Backend> {
    constant32: Param<Tensor<B, 1, Int>>,
    constant41: Param<Tensor<B, 1, Int>>,
    constant42: Param<Tensor<B, 1>>,
    conv1d37: Conv1d<B>,
    conv1d38: Conv1d<B>,
    conv1d39: Conv1d<B>,
    conv1d40: Conv1d<B>,
    conv1d41: Conv1d<B>,
    linear13: Linear<B>,
    linear14: Linear<B>,
    conv1d42: Conv1d<B>,
    conv1d43: Conv1d<B>,
    conv1d44: Conv1d<B>,
    conv1d45: Conv1d<B>,
    conv1d46: Conv1d<B>,
    conv1d47: Conv1d<B>,
    linear15: Linear<B>,
    linear16: Linear<B>,
    conv1d48: Conv1d<B>,
}

impl<B: Backend> ReferenceVAD<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(
        file: &str,
        device: &B::Device,
    ) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(
        bytes: Bytes,
        device: &B::Device,
    ) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> ReferenceVAD<B> {
    /// Build a new reference model.
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant32: Param<Tensor<B, 1, Int>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([0i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant41: Param<Tensor<B, 1, Int>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1, Int>::from_data([1i64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant42: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::from_data([2f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let conv1d37 = Conv1dConfig::new(1, 258, 256)
            .with_stride(128)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d38 = Conv1dConfig::new(129, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d39 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d40 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d41 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear13 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear14 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d42 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d43 = Conv1dConfig::new(1, 130, 128)
            .with_stride(64)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d44 = Conv1dConfig::new(65, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d45 = Conv1dConfig::new(128, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d46 = Conv1dConfig::new(64, 64, 3)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d47 = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let linear15 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let linear16 = LinearConfig::new(128, 512)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let conv1d48 = Conv1dConfig::new(128, 1, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant32,
            constant41,
            constant42,
            conv1d37,
            conv1d38,
            conv1d39,
            conv1d40,
            conv1d41,
            linear13,
            linear14,
            conv1d42,
            conv1d43,
            conv1d44,
            conv1d45,
            conv1d46,
            conv1d47,
            linear15,
            linear16,
            conv1d48,
        }
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        sr: usize,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        match sr {
            16000 => self.forward_16khz(input, state),
            8000 => self.forward_8khz(input, state),
            _ => panic!("unsupported sample rate: {sr}"),
        }
    }

    /// (cell, hidden)
    fn unpack_state(state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hidden = state.clone().slice_dim(0, 0).squeeze_dim::<2usize>(0);
        let cell = state.slice_dim(0, 1).squeeze_dim::<2usize>(0);
        (cell, hidden)
    }

    /// Stacks `(cell, hidden)` into a packed `[2, batch, hidden]` state.
    fn pack_state(
        cell: Tensor<B, 2>,
        hidden: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        Tensor::stack(vec![hidden, cell], 0)
    }

    /// Frame Features, 16khz
    pub fn frame_features_16khz(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let x = input.pad([(0, 0), (0, 64)], PadMode::Reflect);
        let x: Tensor<B, 3> = x.unsqueeze_dim::<3>(1);

        let [real_2, imag_2] = self
            .conv1d37
            .forward(x)
            .square()
            .chunk(2, 1)
            .try_into()
            .unwrap();
        let x = (real_2 + imag_2).sqrt();

        // Encoder
        let x = self.conv1d38.forward(x);
        let x = relu(x);
        let x = self.conv1d39.forward(x);
        let x = relu(x);
        let x = self.conv1d40.forward(x);
        let x = relu(x);
        let x = self.conv1d41.forward(x);
        let x = relu(x);

        x.slice_dim(2, 0).squeeze_dim::<2usize>(2)
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward_16khz(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let input = input.clone();
        let state = state.clone();

        let features = self.frame_features_16khz(input);

        let (cell, hidden) = Self::unpack_state(state);

        let gates = self.linear13.forward(hidden) + self.linear14.forward(features);

        let [g_i, g_f, g_c, g_o] = gates.chunk(4, 1).try_into().unwrap();

        let input_values = sigmoid(g_i);
        let forget_values = sigmoid(g_f);
        let candidate_cell_values = tanh(g_c);
        let output_values = sigmoid(g_o);

        let cell = (forget_values * cell) + (input_values * candidate_cell_values);
        let hidden = output_values * tanh(cell.clone());

        let state = Self::pack_state(cell, hidden.clone());

        // output head
        let x: Tensor<B, 3> = hidden.unsqueeze_dim::<3>(2);
        let x = relu(x);
        let x = self.conv1d42.forward(x);
        let x = sigmoid(x);
        let x = x.squeeze_dims::<2>(&[1]);
        let x = x.mean_dim(1);
        // let x: Tensor<B, 2> = x.squeeze_dims::<1>(&[1]).unsqueeze_dims::<2>(&[1]);

        (x, state)
    }

    /// Run the module.
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward_8khz(
        &self,
        input: Tensor<B, 2>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let input = input.clone();
        let state = state.clone();

        let pad8_out1 = input.pad([(0usize, 0usize), (0usize, 32usize)], PadMode::Reflect);
        let unsqueeze36_out1: Tensor<B, 3> = pad8_out1.unsqueeze_dims::<3>(&[1]);
        let conv1d43_out1 = self.conv1d43.forward(unsqueeze36_out1);
        let slice15_out1 = conv1d43_out1.clone().slice(s![.., 0..65, ..]);
        let slice16_out1 = conv1d43_out1.slice(s![.., 65.., ..]);
        let pow15_out1 = slice15_out1.square();
        let pow16_out1 = slice16_out1.square();
        let sqrt8_out1 = (pow15_out1 + pow16_out1).sqrt();
        let conv1d44_out1 = self.conv1d44.forward(sqrt8_out1);
        let relu36_out1 = relu(conv1d44_out1);
        let conv1d45_out1 = self.conv1d45.forward(relu36_out1);
        let relu37_out1 = relu(conv1d45_out1);
        let conv1d46_out1 = self.conv1d46.forward(relu37_out1);
        let relu38_out1 = relu(conv1d46_out1);
        let conv1d47_out1 = self.conv1d47.forward(relu38_out1);
        let relu39_out1 = relu(conv1d47_out1);
        let features = {
            let sliced = relu39_out1.slice(s![.., .., 0i64]);
            sliced.squeeze_dim::<2usize>(2)
        };

        let gather25_out1 = {
            let sliced = state.clone().slice(s![0i64, .., ..]);
            sliced.squeeze_dim::<2usize>(0)
        };
        let gather26_out1 = {
            let sliced = state.slice(s![1i64, .., ..]);
            sliced.squeeze_dim::<2usize>(0)
        };
        let linear15_out1 = self.linear15.forward(gather25_out1);
        let linear16_out1 = self.linear16.forward(features);
        let add23_out1 = linear15_out1.add(linear16_out1);
        let split_tensors = add23_out1.split_with_sizes([128, 128, 128, 128].into(), 1);
        let [split8_out1, split8_out2, split8_out3, split8_out4] =
            split_tensors.try_into().unwrap();
        let sigmoid29_out1 = sigmoid(split8_out1);
        let sigmoid30_out1 = sigmoid(split8_out2);
        let tanh15_out1 = split8_out3.tanh();
        let sigmoid31_out1 = sigmoid(split8_out4);
        let mul22_out1 = sigmoid30_out1.mul(gather26_out1);
        let mul23_out1 = sigmoid29_out1.mul(tanh15_out1);
        let add24_out1 = mul22_out1.add(mul23_out1);
        let tanh16_out1 = add24_out1.clone().tanh();
        let mul24_out1 = sigmoid31_out1.mul(tanh16_out1);
        let unsqueeze37_out1: Tensor<B, 3> = mul24_out1.clone().unsqueeze_dims::<3>(&[-1]);
        let unsqueeze38_out1: Tensor<B, 3> = mul24_out1.unsqueeze_dims::<3>(&[0]);
        let unsqueeze39_out1: Tensor<B, 3> = add24_out1.unsqueeze_dims::<3>(&[0]);
        let concat8_out1 = Tensor::cat([unsqueeze38_out1, unsqueeze39_out1].into(), 0);
        let relu40_out1 = relu(unsqueeze37_out1);
        let conv1d48_out1 = self.conv1d48.forward(relu40_out1);
        let sigmoid32_out1 = sigmoid(conv1d48_out1);
        let squeeze8_out1 = sigmoid32_out1.squeeze_dims::<2>(&[1]);
        let reducemean8_out1 = { squeeze8_out1.mean_dim(1usize).squeeze_dims::<1usize>(&[1]) };
        let unsqueeze40_out1: Tensor<B, 2> = reducemean8_out1.unsqueeze_dims::<2>(&[1]);
        (unsqueeze40_out1, concat8_out1)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::{
            Distribution,
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;
    use crate::{
        errors::*,
        kits::speech::silero_vad::{
            SileroVadCollection,
            SileroVadMeta,
            pretrained::silero_vad_pretrained_bytes,
        },
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial_test::serial]
    fn test_load_forward_pretrained() {
        type B = PerformanceBackend;
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();

        let sc: SileroVadCollection<B> =
            SileroVadCollection::load_pretrained(&device).ok_or_panic();

        let r_mod: ReferenceVAD<B> =
            ReferenceVAD::from_bytes(silero_vad_pretrained_bytes(), &device);

        let batch = 8;

        for sample_rate in [16000, 8000] {
            let vad = sc.expect_branch(sample_rate);

            if sample_rate == 16000 {
                assert_eq!(vad.chunk_size(), 512)
            }

            let input =
                Tensor::<B, 2>::random([batch, vad.chunk_size()], Distribution::Default, &device);
            let state = vad.init_state(batch, &device);

            // ([batch], [2, batch, d_hidden])
            let input1 = input.clone();
            let state1 = state.clone();
            let (s_out, s_state) = vad.forward(input1, state1);

            // ([batch, 1], [2, batch, d_hidden])
            let (r_out, r_state) = r_mod.forward(input, sample_rate, state.clone());

            s_out
                .reshape([batch, 1])
                .to_data()
                .assert_approx_eq::<F>(&r_out.to_data(), Tolerance::default());

            s_state
                .to_data()
                .assert_approx_eq::<F>(&r_state.to_data(), Tolerance::default());
        }
    }
}
