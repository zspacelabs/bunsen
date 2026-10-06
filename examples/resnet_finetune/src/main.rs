#![allow(dead_code)]
#![recursion_limit = "256"]

extern crate core;
mod data;
mod dataset;

use core::clone::Clone;
use std::{
    sync::Arc,
    time::Instant,
};

use anyhow::Context;
use bunsen::{
    burner::module::{
        DTypeMapper,
        ModuleInit,
    },
    data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    },
    kits::images::resnet::{
        ResNet,
        pretrained::{
            RESNET_PREFABS,
            default_resnet_factory,
        },
    },
};
use burn::{
    config::Config,
    data::{
        dataloader::{
            DataLoaderBuilder,
            Dataset,
        },
        dataset::{
            transform::ShuffledDataset,
            vision::ImageFolderDataset,
        },
    },
    lr_scheduler::{
        composed::{
            ComposedLrSchedulerConfig,
            SchedulerReduction,
        },
        cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::Module,
    nn::{
        LeakyReluConfig,
        PReluConfig,
        activation::ActivationConfig,
        loss::BinaryCrossEntropyLossConfig,
    },
    optim::AdamWConfig,
    prelude::{
        Int,
        Tensor,
    },
    tensor::{
        Device,
        FloatDType,
    },
    train::{
        InferenceStep,
        Learner,
        MetricEarlyStoppingStrategy,
        MultiLabelClassificationOutput,
        StoppingCondition,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
        metric::{
            HammingScore,
            LearningRateMetric,
            LossMetric,
            store::{
                Aggregate,
                Direction,
                Split,
            },
        },
    },
};
use clap::{
    Parser,
    ValueEnum,
};
use clap_common::device::{
    DeviceArgs,
    DeviceChoice,
};

use crate::{
    data::{
        ClassificationBatch,
        ClassificationBatcher,
    },
    dataset::{
        CLASSES,
        PlanetLoader,
        download,
    },
};
/*
tracel-ai/models reference:
| Split | Metric                         | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------------------------------|----------|----------|----------|----------|
| Train | Hamming Score @ Threshold(0.5) | 91.311   | 1        | 95.277   | 5        |
| Train | Loss                           | 0.122    | 5        | 0.250    | 1        |
| Valid | Hamming Score @ Threshold(0.5) | 88.490   | 1        | 93.843   | 3        |
| Valid | Loss                           | 0.168    | 3        | 0.512    | 1        |
 */

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ReplaceActivationOption {
    Relu,
    Gelu,
    PRelu,
    LeakyRelu,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Random seed for reproducibility.
    #[arg(short, long, default_value = "0")]
    pub seed: u64,

    /// Train percentage.
    #[arg(long, default_value = "70")]
    pub train_percentage: u8,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/resnet_finetune")]
    pub artifact_dir: String,

    /// Use half precision for training.
    #[arg(long, default_value = "false")]
    pub half_precision: bool,

    /// The device to train on.
    #[command(flatten)]
    pub device: DeviceArgs,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 100)]
    pub batch_size: usize,

    /// Grads accumulation size for processing
    #[arg(short, long, default_value_t = 8)]
    pub grads_accumulation: usize,

    /// Category smoothing factor for training.
    #[arg(long, default_value = "0.05")]
    pub smoothing: Option<f32>,

    /// Number of workers for data loading.
    #[arg(long, default_value = "0")]
    pub num_workers: usize,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "100")]
    pub num_epochs: usize,

    /// Early stopping patience; 0 to disable.
    #[arg(long, default_value_t = 10)]
    pub patience: usize,

    /// Pretrained `ResNet` weights, as the resnet factory names them:
    /// `torchvision/resnet50`, `timm/resnet50_a1`, or a bare name.
    /// Use "list" to list all available pretrained models.
    #[arg(long, default_value = "torchvision/resnet50")]
    pub pretrained: String,

    /// Replace activation function?
    #[arg(long, default_value = "relu")]
    pub replace_activation: Option<ReplaceActivationOption>,

    /// Freeze the body layers during training.
    #[arg(long, default_value = "false")]
    pub freeze_layers: bool,

    /// Drop Block Prob
    #[arg(long, default_value = "0.2")]
    pub drop_block_prob: f64,

    /// Drop Path Prob
    #[arg(long, default_value = "0.1")]
    pub stochastic_depth_prob: f64,

    /// Learning rate
    #[arg(long, default_value_t = 5e-3)]
    pub learning_rate: f64,

    /// Warm-up epochs.
    #[arg(long, default_value_t = 5)]
    pub warmup_epochs: usize,

    /// Enable cautious weight decay.
    #[arg(long, default_value = "false")]
    pub cautious_weight_decay: bool,

    /// Optimizer Weight decay.
    #[arg(long, default_value_t = 5e-3)]
    pub weight_decay: f32,
}

#[allow(clippy::too_many_arguments)]
mod local {
    use bunsen::kits::images::resnet::ResNetContractConfig;
    use burn::config::Config;

    /// Log config.
    ///
    /// Only exists for logging.
    #[derive(Config, Debug)]
    pub struct LogConfig {
        pub seed: u64,
        pub train_percentage: u8,
        pub batch_size: usize,
        pub num_epochs: usize,
        pub resnet_prefab: String,
        pub resnet_pretrained: String,
        pub drop_block_prob: f64,
        pub drop_path_prob: f64,
        pub learning_rate: f64,
        pub patience: usize,
        pub weight_decay: f32,
        pub resnet: ResNetContractConfig,
    }
}
use local::*;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let _source_tree = download();

    train(&args)
}

fn ensure_artifact_dir(artifact_dir: &str) -> anyhow::Result<()> {
    let _ignored = std::fs::remove_dir_all(artifact_dir);
    std::fs::create_dir_all(artifact_dir)?;
    Ok(())
}

#[must_use]
pub fn train(args: &Args) -> anyhow::Result<()> {
    let mut device = args.device.init().map_err(anyhow::Error::msg)?;
    if args.half_precision && args.device.choice() != DeviceChoice::Flex {
        device.configure(FloatDType::BF16)?;
    }
    // Training records gradients: autodiff before the model and inputs.
    let device: Device = device.autodiff();

    let factory = default_resnet_factory()?;

    // TODO: lift to clap parser.
    if args.pretrained == "list" {
        println!("Available pretrained models:");
        for prefab in RESNET_PREFABS.iter() {
            let cfg = (prefab.builder)();
            println!("* \"{}\"", prefab.name);
            println!("{cfg:?}");

            for (provider, row) in factory.for_prefab(prefab.name) {
                println!("  - \"{provider}:{}\": {}", row.name, row.description);
            }
        }
        return Ok(());
    }
    let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
    let mut model_ref = factory.resolve(&args.pretrained, &cache)?;
    let prefab = model_ref
        .model
        .prefab(&RESNET_PREFABS)
        .with_context(|| format!("{}: names no prefab", model_ref.id()))?;
    let resnet_prefab = prefab.name.clone();
    let resnet_pretrained = model_ref.id();

    // Remove existing artifacts before to get an accurate learner summary
    let artifact_dir: &str = args.artifact_dir.as_ref();
    ensure_artifact_dir(artifact_dir)?;

    device.seed(args.seed);

    let mut resnet_config = prefab.to_config();

    if let Some(option) = &args.replace_activation {
        match option {
            ReplaceActivationOption::Relu => {
                resnet_config = resnet_config.with_activation(ActivationConfig::Relu);
            }
            ReplaceActivationOption::Gelu => {
                resnet_config = resnet_config.with_activation(ActivationConfig::Gelu);
            }
            ReplaceActivationOption::PRelu => {
                resnet_config =
                    resnet_config.with_activation(ActivationConfig::PRelu(PReluConfig::new()));
            }
            ReplaceActivationOption::LeakyRelu => {
                resnet_config = resnet_config
                    .with_activation(ActivationConfig::LeakyRelu(LeakyReluConfig::new()));
            }
        }
    }

    let model: ResNet = resnet_config.clone().try_init(&device)?;

    let old_float_type = model.output_fc.weight.dtype();

    // The activation rewrite is this example's own surgery: the deferred
    // model's hook builds from this config rather than the prefab's, and
    // the checkpoint is read into that model.
    model_ref.hook = model_ref.hook.with_config(resnet_config.clone());
    let loaded = model_ref
        .load(&cache, &device)
        .context("Failed to load pretrained weights")?;

    let mut model: ResNet = Arc::unwrap_or_clone(loaded.handle)
        .map(&mut DTypeMapper::new(old_float_type))
        .with_classes(CLASSES.len())
        .with_stochastic_drop_block(args.drop_block_prob)
        .with_stochastic_path_depth(args.stochastic_depth_prob);

    if args.freeze_layers {
        model = model.freeze_layers();
    }

    let host: Host = Host {
        smoothing: args.smoothing,
        resnet: model,
    };

    let optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(args.cautious_weight_decay)
        .with_weight_decay(args.weight_decay)
        .init();

    LogConfig {
        seed: args.seed,
        train_percentage: args.train_percentage,
        batch_size: args.batch_size,
        num_epochs: args.num_epochs,
        resnet_prefab: resnet_prefab.clone(),
        resnet_pretrained: resnet_pretrained.clone(),
        drop_block_prob: args.drop_block_prob,
        drop_path_prob: args.stochastic_depth_prob,
        learning_rate: args.learning_rate,
        patience: args.patience,
        weight_decay: args.weight_decay,
        resnet: resnet_config,
    }
    .save(format!("{artifact_dir}/config.json"))
    .expect("Config should be saved successfully");

    // Dataloaders
    let batcher_train = ClassificationBatcher::new(device.clone());
    let batcher_valid = ClassificationBatcher::new(device.clone());

    let (train, valid) =
        ImageFolderDataset::planet_train_val_split(args.train_percentage, args.seed)?;

    let train_set_size = train.len();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .num_workers(args.num_workers)
        .build(ShuffledDataset::new(train, args.seed));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(args.batch_size)
        .num_workers(args.num_workers)
        .build(valid);

    let iters_per_epoch = train_set_size as f64 / args.batch_size as f64;
    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .linear(LinearLrSchedulerConfig::new(
            1e-7,
            1.0,
            (iters_per_epoch * args.warmup_epochs as f64) as usize,
        ))
        .cosine(CosineAnnealingLrSchedulerConfig::new(
            args.learning_rate,
            (iters_per_epoch * args.num_epochs as f64) as usize,
        ))
        .with_reduction(SchedulerReduction::Prod)
        .init()
        .expect("Failed to initialize learning rate scheduler");

    let now: Instant;
    {
        let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
            .metrics((
                HammingScore::new(),
                LossMetric::new(),
                // CudaMetric::new(), ??
                LearningRateMetric::new(),
            ))
            .with_default_checkpointers()
            .early_stopping(MetricEarlyStoppingStrategy::new(
                &LossMetric::new(),
                Aggregate::Mean,
                Direction::Lowest,
                Split::Valid,
                StoppingCondition::NoImprovementSince {
                    n_epochs: args.patience,
                },
            ))
            .num_epochs(args.num_epochs)
            .grads_accumulation(args.grads_accumulation)
            .summary();

        now = Instant::now();
        let result = training.launch(Learner::new(host, optimizer, lr_scheduler));
        if let Some(error) = result.error {
            anyhow::bail!("training failed: {error}");
        }

        result
            .model
            .resnet
            .save_file(format!("{artifact_dir}/model.bpk"))?;
    }
    let elapsed = now.elapsed().as_secs();
    println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

    println!("{:#?}", args);

    Ok(())
}

#[derive(Module, Debug)]
pub struct Host {
    pub smoothing: Option<f32>,

    pub resnet: ResNet,
}

pub trait MultiLabelClassification {
    fn forward_classification(
        &self,
        images: Tensor<4>,
        targets: Tensor<2, Int>,
    ) -> MultiLabelClassificationOutput;
}

impl MultiLabelClassification for Host {
    fn forward_classification(
        &self,
        images: Tensor<4>,
        targets: Tensor<2, Int>,
    ) -> MultiLabelClassificationOutput {
        let device = images.device();
        let output = self.resnet.forward(images);

        let mut loss_cfg = BinaryCrossEntropyLossConfig::new().with_logits(true);

        if device.is_autodiff() {
            loss_cfg = loss_cfg.with_smoothing(self.smoothing);
        }

        let loss = loss_cfg
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        MultiLabelClassificationOutput::new(loss, output, targets)
    }
}

impl TrainStep for Host {
    type Input = ClassificationBatch;
    type Output = MultiLabelClassificationOutput;

    fn step(
        &self,
        batch: Self::Input,
    ) -> TrainOutput<Self::Output> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Host {
    type Input = ClassificationBatch;
    type Output = MultiLabelClassificationOutput;

    fn step(
        &self,
        batch: Self::Input,
    ) -> Self::Output {
        self.forward_classification(batch.images, batch.targets)
    }
}
