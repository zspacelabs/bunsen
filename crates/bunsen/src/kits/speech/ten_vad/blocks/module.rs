#![allow(missing_docs)]
use burn::{
    module::{
        Param,
        ParamId,
    },
    nn::{
        Linear,
        LinearConfig,
        Lstm,
        LstmConfig,
        LstmState,
        PaddingConfig2d,
        conv::{
            Conv2d,
            Conv2dConfig,
        },
        pool::{
            MaxPool2d,
            MaxPool2dConfig,
        },
    },
    prelude::*,
    tensor::{
        Bytes,
        DType,
        activation::{
            relu,
            sigmoid,
        },
    },
};
use burn_store::{
    BurnpackStore,
    ModuleSnapshot,
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Config for [`TenVad`].
///
/// Builds [`TenVad`].
#[derive(Config, Debug)]
pub struct TenVadStructureConfig {}

impl<B: Backend> ModuleInit<B, TenVad<B>> for TenVadStructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TenVad<B>> {
        Ok(TenVad::new(device))
    }
}

/// ten-vad module.
///
/// Built by [`TenVadStructureConfig`].
#[derive(Module, Debug)]
pub struct TenVad<B: Backend> {
    constant23: Param<Tensor<B, 1>>,
    constant27: Param<Tensor<B, 1>>,
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    lstm1: Lstm<B>,
    lstm2: Lstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

impl<B: Backend> TenVad<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(
        file: P,
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

impl<B: Backend> TenVad<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant23: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([32], (device, DType::F32)),
            device.clone(),
            false,
            [32].into(),
        );
        let constant27: Param<Tensor<B, 1>> = Param::uninitialized(
            ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::zeros([1], (device, DType::F32)),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d1 = Conv2dConfig::new([1, 1], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d2 = Conv2dConfig::new([1, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([1, 3])
            .with_strides([1, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d3 = Conv2dConfig::new([16, 16], [1, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(false)
            .init(device);
        let conv2d4 = Conv2dConfig::new([16, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([16, 16], [1, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 1))
            .with_dilation([1, 1])
            .with_groups(16)
            .with_bias(false)
            .init(device);
        let conv2d6 = Conv2dConfig::new([16, 16], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let lstm1 = LstmConfig::new(80, 64, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm2 = LstmConfig::new(64, 64, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let linear1 = LinearConfig::new(128, 32).with_bias(false).init(device);
        let linear2 = LinearConfig::new(32, 1).with_bias(false).init(device);
        Self {
            constant23,
            constant27,
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            lstm1,
            lstm2,
            linear1,
            linear2,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input_1: Tensor<B, 3>,
        lstm1_hidden: Tensor<B, 2>,
        lstm1_cell: Tensor<B, 2>,
        lstm2_hidden: Tensor<B, 2>,
        lstm2_cell: Tensor<B, 2>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
    ) {
        let reshape1_out1 = input_1.reshape([-1, 1, 3, 41]);
        // ConvSeq1d
        let conv2d1_out1 = self.conv2d1.forward(reshape1_out1);
        let conv2d2_out1 = self.conv2d2.forward(conv2d1_out1);
        let relu1_out1 = relu(conv2d2_out1);

        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1);

        // ConvSeq1d
        let conv2d3_out1 = self.conv2d3.forward(maxpool2d1_out1);
        let conv2d4_out1 = self.conv2d4.forward(conv2d3_out1);
        let relu2_out1 = relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu2_out1);
        let conv2d6_out1 = self.conv2d6.forward(conv2d5_out1);
        let relu3_out1 = relu(conv2d6_out1);

        let transpose1_out1 = relu3_out1.permute([0, 2, 3, 1]);
        let reshape2_out1 = transpose1_out1.reshape([-1, 1, 80]);

        let (lstm1_out1, lstm1_state) = self.lstm1.forward(
            reshape2_out1,
            Some(LstmState::new(lstm1_cell, lstm1_hidden)),
        );
        let reshape3_out1 = lstm1_out1.reshape([1, -1, 64]);
        let transpose2_out1 = reshape3_out1.clone().swap_dims(0, 1);

        let (lstm2_out1, lstm2_state) = self.lstm2.forward(
            transpose2_out1,
            Some(LstmState::new(lstm2_cell, lstm2_hidden)),
        );
        let transpose3_out1 = lstm2_out1.swap_dims(0, 1);

        let concat1_out1 = Tensor::cat([transpose3_out1, reshape3_out1].into(), 2);

        let mut shape1: [usize; 3] = concat1_out1.dims();
        shape1[2] = 32;
        let reshape4_out1 = concat1_out1.reshape([-1, 128]);
        let linear1_out1 = self.linear1.forward(reshape4_out1);
        let reshape5_out1 = linear1_out1.reshape(shape1);

        let add1_out1 = reshape5_out1 + self.constant23.val().unsqueeze();
        let relu4_out1 = relu(add1_out1);

        let mut shape2: [usize; 3] = relu4_out1.dims();
        shape2[2] = 1;
        let reshape6_out1 = relu4_out1.reshape([-1, 32]);
        let linear2_out1 = self.linear2.forward(reshape6_out1);
        let reshape7_out1 = linear2_out1.reshape(shape2);

        let add2_out1 = reshape7_out1 + self.constant27.val().unsqueeze();
        let sigmoid1_out1 = sigmoid(add2_out1);

        (
            sigmoid1_out1,
            lstm1_state.hidden,
            lstm1_state.cell,
            lstm2_state.hidden,
            lstm2_state.cell,
        )
    }
}
