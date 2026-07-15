#![allow(missing_docs)]
use burn::{
    nn::{
        Linear,
        LinearConfig,
        Lstm,
        LstmConfig,
        LstmState,
        PaddingConfig2d,
        activation::ActivationConfig,
        conv::Conv2dConfig,
        pool::{
            MaxPool2d,
            MaxPool2dConfig,
        },
    },
    prelude::*,
    tensor::{
        Bytes,
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
    blocks::conv::{
        ConvBlock2dConfig,
        ConvSeq2d,
        ConvSeq2dConfig,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Config for [`TenVad`].
///
/// Builds [`TenVad`].
#[derive(Config, Debug)]
pub struct TenVadStructureConfig {
    /// The first ConvSeq2d block.
    pub cs1: ConvSeq2dConfig,

    /// The maxpooling block.
    pub pool: MaxPool2dConfig,

    /// The second ConvSeq2d block.
    pub cs2: ConvSeq2dConfig,
}

impl Default for TenVadStructureConfig {
    fn default() -> Self {
        let cs1 = ConvSeq2dConfig {
            blocks: vec![
                ConvBlock2dConfig::new(Conv2dConfig::new([1, 1], [3, 3]).with_bias(false))
                    .with_act(None),
                ConvBlock2dConfig::new(Conv2dConfig::new([1, 16], [1, 1]))
                    .with_act(Some(ActivationConfig::Relu)),
            ],
        };

        let maxpool = MaxPool2dConfig::new([1, 3]).with_strides([1, 2]);

        let cs2 = ConvSeq2dConfig {
            blocks: vec![
                ConvBlock2dConfig::new(
                    Conv2dConfig::new([16, 16], [1, 3])
                        .with_stride([2, 2])
                        .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
                        .with_groups(16)
                        .with_bias(false),
                )
                .with_act(None),
                ConvBlock2dConfig::new(Conv2dConfig::new([16, 16], [1, 1]))
                    .with_act(Some(ActivationConfig::Relu)),
                ConvBlock2dConfig::new(
                    Conv2dConfig::new([16, 16], [1, 3])
                        .with_stride([2, 2])
                        .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 1))
                        .with_groups(16)
                        .with_bias(false),
                )
                .with_act(None),
                ConvBlock2dConfig::new(Conv2dConfig::new([16, 16], [1, 1]))
                    .with_act(Some(ActivationConfig::Relu)),
            ],
        };

        Self {
            cs1,
            pool: maxpool,
            cs2,
        }
    }
}

impl<B: Backend> ModuleInit<B, TenVad<B>> for TenVadStructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TenVad<B>> {
        let cs1 = self.cs1.try_init(device)?;
        let cs2 = self.cs2.try_init(device)?;
        let pool = self.pool.init();

        let lstm1 = LstmConfig::new(80, 64, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm2 = LstmConfig::new(64, 64, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let linear1 = LinearConfig::new(128, 32).with_bias(true).init(device);
        let linear2 = LinearConfig::new(32, 1).with_bias(true).init(device);

        Ok(TenVad {
            cs1,
            pool,
            cs2,
            lstm1,
            lstm2,
            linear1,
            linear2,
            phantom: core::marker::PhantomData,
        })
    }
}

/// ten-vad module.
///
/// Built by [`TenVadStructureConfig`].
#[derive(Module, Debug)]
pub struct TenVad<B: Backend> {
    pub cs1: ConvSeq2d<B>,
    pub pool: MaxPool2d,
    pub cs2: ConvSeq2d<B>,
    pub lstm1: Lstm<B>,
    pub lstm2: Lstm<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> TenVad<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(
        file: P,
        device: &B::Device,
    ) -> Self {
        let mut model = TenVadStructureConfig::default().try_init(device).unwrap();
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
        let mut model = TenVadStructureConfig::default().try_init(device).unwrap();
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> TenVad<B> {
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        input_1: Tensor<B, 3>,
        state1: Option<LstmState<B, 2>>,
        state2: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, LstmState<B, 2>, LstmState<B, 2>) {
        let x = input_1.reshape([-1, 1, 3, 41]);
        let x = self.cs1.forward(x);
        let x = self.pool.forward(x);
        let x = self.cs2.forward(x);

        let x = x.permute([0, 2, 3, 1]);
        let x = x.reshape([-1, 1, 80]);

        let (x, state1) = self.lstm1.forward(x, state1);
        let y = x.reshape([1, -1, 64]);

        let x = y.clone().swap_dims(0, 1);

        let (x, state2) = self.lstm2.forward(x, state2);
        let x = x.swap_dims(0, 1);

        let x = Tensor::cat([x, y].into(), 2);

        let mut shape1: [usize; 3] = x.dims();
        shape1[2] = 32;
        let x = x.reshape([-1, 128]);
        let x = self.linear1.forward(x);
        let x = x.reshape(shape1);

        let x = relu(x);

        let mut shape2: [usize; 3] = x.dims();
        shape2[2] = 1;
        let x = x.reshape([-1, 32]);
        let x = self.linear2.forward(x);
        let x = x.reshape(shape2);

        let x = sigmoid(x);

        (x, state1, state2)
    }
}
