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

    /// The first Lstm block.
    pub lstm1: LstmConfig,

    /// The second Lstm block.
    pub lstm2: LstmConfig,

    /// The first linear output block.
    pub linear1: LinearConfig,

    /// The second linear output block.
    pub linear2: LinearConfig,
}

impl Default for TenVadStructureConfig {
    fn default() -> Self {
        let d_hidden = 64;
        Self {
            cs1: ConvSeq2dConfig {
                blocks: vec![
                    ConvBlock2dConfig::new(Conv2dConfig::new([1, 1], [3, 3]).with_bias(false))
                        .with_act(None),
                    ConvBlock2dConfig::new(Conv2dConfig::new([1, 16], [1, 1]))
                        .with_act(Some(ActivationConfig::Relu)),
                ],
            },
            pool: MaxPool2dConfig::new([1, 3]).with_strides([1, 2]),
            cs2: ConvSeq2dConfig {
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
            },
            lstm1: LstmConfig::new(80, d_hidden, true)
                .with_batch_first(false)
                .with_input_forget(false),
            lstm2: LstmConfig::new(d_hidden, d_hidden, true)
                .with_batch_first(false)
                .with_input_forget(false),
            linear1: LinearConfig::new(2 * d_hidden, d_hidden / 2).with_bias(true),
            linear2: LinearConfig::new(d_hidden / 2, 1).with_bias(true),
        }
    }
}

impl<B: Backend> ModuleInit<B, TenVad<B>> for TenVadStructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TenVad<B>> {
        Ok(TenVad {
            cs1: self.cs1.try_init(device)?,
            pool: self.pool.init(),
            cs2: self.cs2.try_init(device)?,
            lstm1: self.lstm1.init(device),
            lstm2: self.lstm2.init(device),
            linear1: self.linear1.init(device),
            linear2: self.linear2.init(device),
        })
    }
}

/// ten-vad module.
///
/// Built by [`TenVadStructureConfig`].
#[derive(Module, Debug)]
pub struct TenVad<B: Backend> {
    /// The first ConvSeq2d block.
    pub cs1: ConvSeq2d<B>,

    /// The MaxPool2d block.
    pub pool: MaxPool2d,

    /// The second ConvSeq2d block.
    pub cs2: ConvSeq2d<B>,

    /// The first Lstm block.
    pub lstm1: Lstm<B>,

    /// The second Lstm block.
    pub lstm2: Lstm<B>,

    /// The first linear output block.
    pub linear1: Linear<B>,

    /// The second linear output block.
    pub linear2: Linear<B>,
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
    /// Forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state1: Option<LstmState<B, 2>>,
        state2: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, LstmState<B, 2>, LstmState<B, 2>) {
        let x = input.reshape([-1, 1, 3, 41]);
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
