use std::{
    cmp::max,
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
    },
};

use bunsen::{
    burner::{
        module::reflection::XmlModuleTree,
        optim::{
            GroupOptimizerAdaptor2,
            OptimizerGroup,
        },
    },
    kits::gpts::nanochat::{
        NanoChatGpt,
        NanoChatGptConfig,
        NanoChatGptMeta,
    },
    public::hashbrown::HashSet,
};
use bunsen_preview_chat_dataloader::{
    dataloader::ChatDataLoader,
    tokens::{
        DenseTokenBlocksOptions,
        TokenBatchIteratorOptions,
    },
};
use burn::{
    lr_scheduler::linear::LinearLrSchedulerConfig,
    module::{
        Module,
        ParamId,
    },
    nn::loss::CrossEntropyLossConfig,
    optim::{
        AdamWConfig,
        LearningRate,
        MuonConfig,
        decay::WeightDecayConfig,
    },
    prelude::Backend,
    record::CompactRecorder,
    tensor::{
        AsIndex,
        Slice,
        Tensor,
        backend::AutodiffBackend,
        bf16,
        s,
    },
    train::{
        ClassificationOutput,
        InferenceStep,
        Learner,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
        metric::{
            LearningRateMetric,
            LossMetric,
        },
    },
};
use clap::Parser;
use num_traits::Pow;
use rand::{
    SeedableRng,
    rngs::StdRng,
};
use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    disk_cache::WordchipperDiskCache,
};
use wordchipper_cli_util::logging::LogArgs;
use zsl_data_cache::dataset::DatasetCacheConfig;

#[derive(Debug, Clone, clap::Args)]
pub struct TokenBatchOptionsArgs {
    /// The number of sequences to load per batch.
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// The maximum number of tokens in a sequence.
    #[arg(long, default_value_t = 2048)]
    pub batch_seq_len: usize,

    /// The minimum number of sequences to keep in the buffer
    /// before loading more sequences.
    #[arg(long, default_value_t = 1024)]
    pub min_buffer: usize,
}

impl TokenBatchOptionsArgs {
    pub fn options(&self) -> TokenBatchIteratorOptions {
        TokenBatchIteratorOptions {
            batch_size: self.batch_size,
            batch_seq_len: self.batch_seq_len,
            min_buffer: self.min_buffer,
        }
    }
}

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The embedding dimension size.
    #[clap(long, default_value = "768")]
    pub n_embed: usize,

    /// The number of layers.
    #[clap(long, default_value = "8")]
    pub n_layer: usize,

    /// The pretrained vocabulary.
    #[clap(long, default_value = "openai:p50k_edit")]
    pub pretrained_vocab: String,

    /// Beginning-of-Sequence token.
    #[arg(long, default_value = "<|bos|>")]
    pub bos_token: String,

    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to the dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    #[arg(long, default_value_t = 0.008)]
    pub unembedding_lr: f64,

    #[arg(long, default_value_t = 0.3)]
    pub embedding_lr: f64,

    #[arg(long, default_value_t = 0.02)]
    pub matrix_lr: f64,

    #[arg(long, default_value_t = 0.5)]
    pub scalar_lr: f64,

    /// Warm-up steps.
    #[arg(long, default_value_t = 300)]
    pub warmup_steps: usize,

    /// Optimizer Weight decay.
    #[arg(long, default_value_t = 0.28)]
    pub weight_decay: f32,

    /// Number of epochs to train the model.
    #[arg(long, default_value = "100")]
    pub num_epochs: usize,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 4)]
    pub batch_size: usize,

    /// The training seq len.
    #[clap(long, default_value = "2048")]
    pub seq_len: usize,

    /// Grads accumulation size for processing
    #[arg(short, long, default_value_t = 1)]
    pub grads_accumulation: usize,

    /// Directory to save the artifacts.
    #[arg(long, default_value = "/tmp/chat")]
    pub artifact_dir: String,
}

fn ensure_artifact_dir(artifact_dir: &str) -> anyhow::Result<()> {
    let _ignored = std::fs::remove_dir_all(artifact_dir);
    std::fs::create_dir_all(artifact_dir)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // args.logging.setup_logging(3).unwrap();

    run::<burn::backend::Autodiff<burn::backend::cuda::Cuda<bf16>>>(&args)
}

fn run<B: AutodiffBackend>(args: &Args) -> anyhow::Result<()> {
    type T = u32;

    println!("{:#?}", args);

    // Remove existing artifacts before to get an accurate learner summary
    let artifact_dir: &str = args.artifact_dir.as_ref();
    ensure_artifact_dir(artifact_dir)?;

    let device: B::Device = Default::default();

    let data_cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir.clone());
    log::info!("DATASET CACHE: {:#?}", data_cache_config);
    let mut data_cache = data_cache_config.clone().init()?;

    let mut disk_cache = WordchipperDiskCache::default();

    let shards: Vec<usize> = {
        let max_shard = data_cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for slice in &args.shards {
            for idx in slice.into_iter() {
                let shard = idx.expect_elem_index(max_shard);
                collected.insert(shard);
            }
        }
        let mut shards: Vec<usize> = collected.into_iter().collect();
        shards.sort();
        shards
    };

    log::info!("Loading Shards: {shards:?}");
    let shard_paths = data_cache.load_shards(&shards)?;

    let validation_ratio = 0.10;
    let num_validation_shards: usize = max(
        ((shard_paths.len() as f64) * validation_ratio).ceil() as usize,
        1,
    );
    let num_training_shards = shard_paths.len() - num_validation_shards;

    let training_paths: Vec<PathBuf> = shard_paths[..num_training_shards].to_vec();
    let validation_paths: Vec<PathBuf> = shard_paths[num_training_shards..].to_vec();

    let mut vocab: UnifiedTokenVocab<T> =
        wordchipper::load_vocab(&args.pretrained_vocab, &mut disk_cache)?
            .vocab()
            .to_token_type()?;

    let max_token = vocab.max_token().unwrap();

    // This is a stupid hack.
    let mut vocab_size = vocab.len();
    let bos_token: T = {
        let specials = vocab.special_vocab_mut();
        if let Some(tok) = specials.lookup_token(args.bos_token.as_bytes()) {
            tok
        } else {
            let tok = max_token + 1;
            specials.add_str_word(&args.bos_token, tok);
            vocab_size += 1;
            tok
        }
    };
    let vocab = Arc::new(vocab);

    let tok = wordchipper::TokenizerOptions::default()
        .with_accelerated_lexers(true)
        .with_parallel(true)
        .build(vocab);

    let gpt_config = NanoChatGptConfig::new()
        .with_n_embed(args.n_embed)
        .with_n_layer(args.n_layer)
        .with_vocab_size(vocab_size);

    let gpt: NanoChatGpt<B> = gpt_config.clone().init::<B>(&device);

    let host = GptHost { gpt };

    let dl_config = DenseTokenBlocksOptions {
        batch_seq_len: args.seq_len,
        batch_size: args.batch_size,
        bos: vec![bos_token],
        ..Default::default()
    };

    let training_data_loader: ChatDataLoader<B> = ChatDataLoader::new(
        training_paths,
        Some(Arc::new(Mutex::new(StdRng::seed_from_u64(0)))),
        &device,
        tok.clone(),
        dl_config.clone(),
    );

    let validation_data_loader: ChatDataLoader<B::InnerBackend> =
        ChatDataLoader::new(validation_paths, None, &device, tok.clone(), dl_config);

    let training = SupervisedTraining::new(
        artifact_dir,
        Arc::new(training_data_loader),
        Arc::new(validation_data_loader),
    )
    .grads_accumulation(args.grads_accumulation)
    .num_epochs(args.num_epochs)
    .metrics((LossMetric::new(), LearningRateMetric::new()))
    .with_file_checkpointer(CompactRecorder::new())
    .summary();

    let mut mtree = XmlModuleTree::build(&host);

    // See: GPT.setup_optimizer
    // <https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py#L374>
    let matrix_params: HashSet<ParamId> = mtree
        .select_params("GptHost/GPT/*[@name='h']/Linear/*[@name='weight',@rank=2]")
        .to_param_ids()?
        .into_iter()
        .collect();

    // TODO: value_embeds
    // TODO: resid
    // TODO: x0
    // TODO: smear, smear_gate, blackout

    let embedding_params: HashSet<ParamId> = mtree
        .select_params("GptHost/GPT/*[@name='wte']")
        .to_param_ids()?
        .into_iter()
        .collect();

    let lm_head_params: HashSet<ParamId> = mtree
        .select_params("GptHost/GPT/*[@name='lm_head']")
        .to_param_ids()?
        .into_iter()
        .collect();

    let remnant_params: HashSet<ParamId> = mtree
        .param_ids()?
        .into_iter()
        .collect::<HashSet<ParamId>>()
        .difference(&matrix_params)
        .cloned()
        .collect::<HashSet<ParamId>>()
        .difference(&embedding_params)
        .cloned()
        .collect::<HashSet<ParamId>>()
        .difference(&lm_head_params)
        .cloned()
        .collect::<HashSet<ParamId>>();

    let model_dim = gpt_config.n_embed();
    let dmodel_lr_scale: f64 = (model_dim as f64 / 768.0_f64).pow(-0.5);

    let lm_head_lr = args.unembedding_lr * dmodel_lr_scale;
    let embedding_lr = args.embedding_lr * dmodel_lr_scale;
    let scalar_lr = args.scalar_lr;
    let matrix_lr = args.matrix_lr;

    // This is only used to scale the learning rates for each group below.
    // This implements warmup scheduling for learning rate.
    let warmup_scheduler = LinearLrSchedulerConfig::new(1e-10, 1.0, args.warmup_steps)
        .init()
        .expect("Failed to initialize learning rate scheduler");

    // TODO: per-group GradientClipping.

    let optimizer = GroupOptimizerAdaptor2::new(
        vec![
            OptimizerGroup::from_adaptor(
                lm_head_params,
                &AdamWConfig::new()
                    .with_beta_1(0.8)
                    .with_beta_2(0.96)
                    .with_epsilon(1e-10)
                    .with_weight_decay(0.01)
                    .init::<B, GptHost<B>>(),
            )
            .with_lr_selector(
                move |lr: f64,
                      _: &bunsen::public::hashbrown::HashMap<String, LearningRate>|
                      -> LearningRate { lr * lm_head_lr },
            ),
            OptimizerGroup::from_adaptor(
                embedding_params,
                &AdamWConfig::new()
                    .with_beta_1(0.8)
                    .with_beta_2(0.995)
                    .with_epsilon(1e-10)
                    .with_weight_decay(0.001)
                    .init::<B, GptHost<B>>(),
            )
            .with_lr_selector(
                move |lr: LearningRate,
                      _: &bunsen::public::hashbrown::HashMap<String, LearningRate>|
                      -> LearningRate { lr * embedding_lr },
            ),
            OptimizerGroup::from_adaptor(
                remnant_params,
                &AdamWConfig::new()
                    .with_beta_1(0.8)
                    .with_beta_2(0.96)
                    .with_epsilon(1e-10)
                    .with_weight_decay(0.01)
                    .init::<B, GptHost<B>>(),
            )
            .with_lr_selector(
                move |lr: LearningRate,
                      _: &bunsen::public::hashbrown::HashMap<String, LearningRate>|
                      -> LearningRate { lr * scalar_lr },
            ),
        ],
        vec![
            OptimizerGroup::from_adaptor(
                matrix_params,
                &MuonConfig::new()
                    // .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
                    .with_weight_decay(Some(WeightDecayConfig {
                        penalty: args.weight_decay,
                    }))
                    .init::<B, GptHost<B>>(),
            )
            .with_lr_selector(
                move |lr: LearningRate,
                      _: &bunsen::public::hashbrown::HashMap<String, LearningRate>|
                      -> LearningRate { lr * matrix_lr },
            ),
        ],
    )
    .unwrap();

    let result = training.launch(Learner::new(host, optimizer, warmup_scheduler));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}

#[derive(Module, Debug)]
pub struct GptHost<B: Backend> {
    pub gpt: NanoChatGpt<B>,
}

impl<B: Backend> GptHost<B> {
    fn loss_step(
        &self,
        input: Tensor<B, 2, burn::prelude::Int>,
    ) -> ClassificationOutput<B> {
        let inputs: Tensor<B, 2, burn::prelude::Int> = input.clone().slice(s![.., ..-1]);
        let targets: Tensor<B, 2, burn::prelude::Int> = input.slice(s![.., 1..]);

        let mut kv_cache = None;

        // Logits.
        let outputs: Tensor<B, 3> = self.gpt.forward(inputs, &mut kv_cache);

        let output_flatten: Tensor<B, 2> = outputs.flatten(0, 1);
        let targets_flatten: Tensor<B, 1, burn::prelude::Int> = targets.flatten(0, 1);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_flatten.device())
            .forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for GptHost<B> {
    type Input = Tensor<B, 2, burn::prelude::Int>;
    type Output = ClassificationOutput<B>;

    fn step(
        &self,
        input: Self::Input,
    ) -> TrainOutput<Self::Output> {
        let classification_output = self.loss_step(input);
        let grads = classification_output.loss.backward();

        TrainOutput::new(self, grads, classification_output)
    }
}

impl<B: Backend> InferenceStep for GptHost<B> {
    type Input = Tensor<B, 2, burn::prelude::Int>;
    type Output = ClassificationOutput<B>;

    fn step(
        &self,
        input: Self::Input,
    ) -> Self::Output {
        self.loss_step(input)
    }
}
